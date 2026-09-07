from __future__ import annotations

import socket
import ssl
import time
from io import BytesIO
from threading import Event, Lock
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from gpu_fault.cluster_executor import (
    ClusterActionExecutor,
    ClusterExecutorError,
    RegionalExecutorClient,
    RegionalFleetRegistry,
    _persistent_store_from_environment,
    executor_from_environment,
)
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.regional import RemoteCommandResult, RemoteCommandStatus
from gpu_fault.regional_compatibility import CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION


def test_regional_executor_client_strips_secret_file_newline() -> None:
    client = RegionalExecutorClient(
        " https://control-plane.example ", " cluster-a ", " token-value\n"
    )

    assert client.base_url == "https://control-plane.example"
    assert client.cluster_id == "cluster-a"
    assert client.token == "token-value"


def test_regional_executor_client_rejects_embedded_control_character() -> None:
    with pytest.raises(
        ClusterExecutorError, match="token must not contain control characters"
    ):
        RegionalExecutorClient(
            "https://control-plane.example", "cluster-a", "token\nvalue"
        )


def test_regional_executor_client_advertises_current_protocol(monkeypatch) -> None:
    client = RegionalExecutorClient(
        "https://control-plane.example", "cluster-a", "token-value"
    )
    payloads = []

    def post(path, payload):
        payloads.append((path, payload))
        if path.endswith("/claim"):
            return {"commands": []}
        return {"ready": True}

    monkeypatch.setattr(client, "_post", post)

    assert (
        client.claim(
            "executor-a", execution_owners=["owner-a"], max_commands=1, lease_seconds=60
        )
        == []
    )
    client.readiness(
        "executor-a",
        execution_owners=["owner-a"],
        last_successful_claim_age_seconds=1.0,
    )

    assert [payload["executor_protocol_version"] for _, payload in payloads] == [
        CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
        CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
    ]


def test_regional_executor_client_advertises_its_artifact(monkeypatch) -> None:
    artifact = "a" * 64
    client = RegionalExecutorClient(
        "https://control-plane.example",
        "cluster-a",
        "token-value",
        executor_artifact_sha256=artifact,
    )
    payloads = []

    def post(path, payload):
        payloads.append((path, payload))
        return {"commands": []} if path.endswith("/claim") else {"ready": True}

    monkeypatch.setattr(client, "_post", post)

    client.claim("executor-a", max_commands=1, lease_seconds=60)
    client.readiness(
        "executor-a",
        execution_owners=["owner-a"],
        last_successful_claim_age_seconds=1.0,
    )

    assert [payload["executor_artifact_sha256"] for _, payload in payloads] == [
        artifact,
        artifact,
    ]
    assert [payload["executor_compatibility_digest"] for _, payload in payloads] == [
        artifact,
        artifact,
    ]


def test_regional_executor_client_scopes_private_ca_to_control_plane(
    monkeypatch,
) -> None:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    created = []
    sent = []

    def create_default_context(*, cafile=None):
        created.append(cafile)
        return context

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"ready":true}'

    monkeypatch.setattr(
        "gpu_fault.cluster_executor.ssl.create_default_context", create_default_context
    )
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.urlopen",
        lambda request, **kwargs: (
            sent.append((request.full_url, kwargs)) or Response()
        ),
    )
    client = RegionalExecutorClient(
        "https://control-plane.example",
        "cluster-a",
        "token-value",
        ca_file="/etc/gpu-fault/tls/ca.crt",
    )

    assert client._get("/healthz") == {"ready": True}
    assert created == ["/etc/gpu-fault/tls/ca.crt"]
    assert sent == [
        (
            "https://control-plane.example/healthz",
            {"timeout": 15, "ssl_context": context},
        )
    ]


def test_regional_executor_client_preserves_http_status(monkeypatch) -> None:
    error = HTTPError(
        "https://control-plane.example/v1/fleet/readiness",
        503,
        "Service Unavailable",
        {},
        BytesIO(b'{"detail":"store unavailable"}'),
    )
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    client = RegionalExecutorClient(
        "https://control-plane.example", "cluster-a", "token-value"
    )
    registry = RegionalFleetRegistry(client)

    with pytest.raises(ClusterExecutorError) as raised:
        registry.list_agents("cluster-a")

    assert raised.value.status_code == 503
    assert "store unavailable" in str(raised.value)


class FakeClient:
    # _validate compares the claimed command against the client's own
    # cluster, so the fake has to carry one.
    cluster_id = "cluster-a"

    def __init__(self) -> None:
        self.claim_kwargs = None

    def claim(self, executor_id, **kwargs):
        self.claim_kwargs = {"executor_id": executor_id, **kwargs}
        return []


class FakeAdapter:
    def __init__(self, owner: str) -> None:
        self.owner = owner


def test_persistent_store_is_optional(monkeypatch) -> None:
    monkeypatch.delenv("GPU_FAULT_STORE_URL", raising=False)

    assert _persistent_store_from_environment() is None


def test_persistent_store_builds_postgres_pool(monkeypatch) -> None:
    captured = {}

    class FakePostgresStore:
        def __init__(self, url, **kwargs) -> None:
            captured["url"] = url
            captured["kwargs"] = kwargs

    monkeypatch.setattr("gpu_fault.store.PostgresStore", FakePostgresStore)
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://db.example/gpu_fault")
    monkeypatch.setenv("GPU_FAULT_POSTGRES_POOL_MIN_SIZE", "2")
    monkeypatch.setenv("GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "6")
    monkeypatch.setenv("GPU_FAULT_POSTGRES_POOL_TIMEOUT_SECONDS", "3.5")

    store = _persistent_store_from_environment()

    assert isinstance(store, FakePostgresStore)
    assert captured == {
        "url": "postgresql://db.example/gpu_fault",
        "kwargs": {"pool_min_size": 2, "pool_max_size": 6, "pool_timeout_seconds": 3.5},
    }


def test_persistent_store_rejects_unknown_url(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "mysql://db/gpu_fault")

    with pytest.raises(ClusterExecutorError, match="must use sqlite:/// or PostgreSQL"):
        _persistent_store_from_environment()


def test_cluster_executor_advertises_local_adapter_owners() -> None:
    client = FakeClient()
    executor = ClusterActionExecutor(
        client,
        [
            FakeAdapter("gpu-fault-kubernetes-adapter"),
            FakeAdapter("gpu-fault-node-agent"),
        ],
        executor_id="executor-a",
        allowed_namespaces={"training"},
    )

    assert executor.run_once() == 0
    assert client.claim_kwargs["execution_owners"] == [
        "gpu-fault-kubernetes-adapter",
        "gpu-fault-node-agent",
    ]


def test_cluster_executor_claim_failures_use_bounded_backoff(monkeypatch) -> None:
    class FailingClient(FakeClient):
        def claim(self, *_args, **_kwargs):
            raise ClusterExecutorError("expired regional token")

    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 4:
            raise StopIteration

    monkeypatch.setattr("gpu_fault.cluster_executor.time.sleep", sleep)
    executor = ClusterActionExecutor(
        FailingClient(),
        [FakeAdapter("gpu-fault-node-agent")],
        executor_id="executor-a",
        allowed_namespaces={"training"},
        poll_seconds=2,
        claim_backoff_max_seconds=8,
    )

    with pytest.raises(StopIteration):
        executor.run()

    assert sleeps == [2, 4, 8, 8]


def test_cluster_executor_runs_claimed_commands_concurrently() -> None:
    class ConcurrentClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.commands = [
                SimpleNamespace(
                    command_id=f"command-{index}",
                    cluster_id="cluster-a",
                    lease_token=f"lease-{index}",
                    step=SimpleNamespace(operation=WorkflowOperation.VALIDATE_HOST),
                )
                for index in range(2)
            ]
            self.completed = []

        def claim(self, executor_id, **kwargs):
            super().claim(executor_id, **kwargs)
            return self.commands

        def complete(self, command, result):
            self.completed.append((command.command_id, result.status))

        def renew(self, *_args, **_kwargs):
            return None

    client = ConcurrentClient()
    executor = ClusterActionExecutor(
        client,
        [FakeAdapter("owner-a")],
        executor_id="executor-a",
        allowed_namespaces={"training"},
        max_concurrent_commands=2,
    )
    started = 0
    started_lock = Lock()
    both_started = Event()

    def execute(command):
        nonlocal started
        with started_lock:
            started += 1
            if started == 2:
                both_started.set()
        assert both_started.wait(timeout=1)
        time.sleep(0.1)
        return RemoteCommandResult(
            lease_token=command.lease_token, status=RemoteCommandStatus.SUCCEEDED
        )

    executor._execute = execute
    started_at = time.monotonic()
    count = executor.run_once()
    elapsed = time.monotonic() - started_at

    assert count == 2
    assert elapsed < 0.3
    assert len(client.completed) == 2


def test_cluster_executor_renews_remote_command_lease() -> None:
    class RenewClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.renewals = []

        def renew(self, command, executor_id, lease_seconds):
            self.renewals.append((command.command_id, executor_id, lease_seconds))

    class StopAfterOneRenewal:
        def __init__(self) -> None:
            self.calls = 0

        def wait(self, _seconds):
            self.calls += 1
            return self.calls > 1

    client = RenewClient()
    executor = ClusterActionExecutor(
        client,
        [FakeAdapter("owner-a")],
        executor_id="executor-a",
        allowed_namespaces={"training"},
        lease_seconds=120,
    )
    command = SimpleNamespace(
        command_id="command-a", cluster_id="cluster-a", lease_token="lease-a"
    )

    executor._renew_lease(command, StopAfterOneRenewal())

    assert client.renewals == [("command-a", "executor-a", 120)]


def test_lease_renewal_interval_is_a_third_of_the_lease_capped_at_thirty() -> None:
    """``lease_seconds`` is validated to 10..7200, so the interval's only live
    clamp is the 30s ceiling; the old ``max(1.0, ...)`` floor was dead code."""

    class RecordingStop:
        def __init__(self) -> None:
            self.waits: list[float] = []

        def wait(self, seconds: float) -> bool:
            self.waits.append(seconds)
            return True

    for lease_seconds, expected in ((10, 10 / 3), (90, 30.0), (7200, 30.0)):
        executor = ClusterActionExecutor(
            FakeClient(),
            [FakeAdapter("owner-a")],
            executor_id="executor-a",
            allowed_namespaces={"training"},
            lease_seconds=lease_seconds,
        )
        stop = RecordingStop()
        executor._renew_lease(
            SimpleNamespace(command_id="c", cluster_id="cluster-a", lease_token="l"),
            stop,
        )
        assert stop.waits == [expected], (lease_seconds, stop.waits)


def test_a_failed_outcome_without_a_message_is_not_an_executor_internal_error() -> None:
    """The adapter's FAILED verdict with ``error=None`` used to fail the result
    model inside the try block and surface as executor-internal-error."""

    class SilentAdapter(FakeAdapter):
        def supports(self, step) -> bool:
            return True

        def execute(self, context) -> WorkflowStepOutcome:
            return WorkflowStepOutcome(status=WorkflowStepStatus.FAILED, error=None)

    executor = _executor(SilentAdapter("owner-a"))

    result = executor._execute(_remote_command())

    assert result.status is RemoteCommandStatus.FAILED
    assert result.status_source is None
    assert "without an error message" in (result.error or "")
    assert result.details["error_message_missing"] is True
    assert "executor_internal_error" not in result.details
    assert executor.unexpected_failures == 0


def test_cluster_executor_rejects_duplicate_adapter_owner() -> None:
    with pytest.raises(ClusterExecutorError, match="unique owner"):
        ClusterActionExecutor(
            FakeClient(),
            [FakeAdapter("owner-a"), FakeAdapter("owner-a")],
            executor_id="executor-a",
            allowed_namespaces=set(),
        )


def _remote_command(operation=WorkflowOperation.VALIDATE_HOST):
    """A claimed command shaped enough to reach adapter dispatch.

    ``WorkflowStepContext`` is a plain dataclass, so ``_execute`` accepts
    these stand-ins and the test can drive the catch-all without building
    a whole incident graph.
    """

    return SimpleNamespace(
        command_id="command-a",
        cluster_id="cluster-a",
        lease_token="lease-a",
        fencing_token=7,
        step_index=0,
        idempotency_key="idem-a",
        result_details=None,
        restart_authorization=None,
        step=SimpleNamespace(
            operation=operation,
            workload_ids=[],
            node_ids=["node-a"],
            execution_owner="owner-a",
        ),
        # ``executes_safety_steps`` is the model property the executor reads to
        # pick the step set (F-C8); the stand-in answers like a PENDING record.
        workflow=SimpleNamespace(
            fencing_token=7, step_executions=[], executes_safety_steps=False
        ),
        incident=SimpleNamespace(fencing_token=7),
    )


class RaisingAdapter(FakeAdapter):
    def __init__(self, owner: str, error: BaseException) -> None:
        super().__init__(owner)
        self.error = error

    def supports(self, _step) -> bool:
        return True

    def execute(self, _context):
        raise self.error


def _executor(adapter) -> ClusterActionExecutor:
    return ClusterActionExecutor(
        FakeClient(),
        [adapter],
        executor_id="executor-a",
        allowed_namespaces={"training"},
    )


class ExecutingClient(FakeClient):
    def __init__(self, command) -> None:
        super().__init__()
        self.command = command
        self.completed = None

    def claim(self, executor_id, **kwargs):
        super().claim(executor_id, **kwargs)
        command, self.command = self.command, None
        return [command] if command is not None else []

    def complete(self, _command, result) -> None:
        self.completed = result


def _run_command(adapter, command):
    client = ExecutingClient(command)
    executor = ClusterActionExecutor(
        client, [adapter], executor_id="executor-a", allowed_namespaces={"training"}
    )

    assert executor.run_once() == 1
    assert client.completed is not None
    return executor, client.completed


def _botocore_error(name: str) -> Exception:
    return type(name, (Exception,), {"__module__": "botocore.exceptions"})(name)


def test_aws_misconfiguration_is_not_an_executor_internal_error() -> None:
    # The original symptom: a ServiceAccount with no role-arn annotation
    # made DescribeCluster raise NoCredentialsError, which is outside the
    # preflight call site's except tuple, so the operator read a
    # deployment gap as "the executor is broken" -- and it was counted
    # into the internal-error metric alerting watches.
    executor = _executor(
        RaisingAdapter("owner-a", _botocore_error("NoCredentialsError"))
    )

    result = executor._execute(_remote_command())

    assert result.status is RemoteCommandStatus.FAILED
    assert result.status_source == "executor-configuration-error"
    assert result.details["configuration_error"] is True
    assert result.details["exception_type"] == "NoCredentialsError"
    assert "executor_internal_error" not in result.details
    assert "eks.amazonaws.com/role-arn" in result.error
    # store.py counts internal errors by status_source, so this source
    # keeps configuration gaps out of that metric.
    assert executor.unexpected_failures == 0


def test_executor_defect_is_still_an_internal_error() -> None:
    executor = _executor(RaisingAdapter("owner-a", AttributeError("core")))

    result = executor._execute(_remote_command())

    assert result.status_source == "executor-internal-error"
    assert result.details["executor_internal_error"] is True
    assert executor.unexpected_failures == 1


def test_temporary_dns_failure_keeps_command_retryable() -> None:
    executor = _executor(
        RaisingAdapter(
            "owner-a",
            socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution"),
        )
    )

    result = executor._execute(_remote_command())

    assert result.status is RemoteCommandStatus.WAITING
    assert result.status_source == "executor-retryable-transport"
    assert result.details["retryable_transport_error"] is True
    assert result.details["exception_type"] == "gaierror"
    assert "temporary DNS failure" in result.details["reason"]
    assert executor.unexpected_failures == 0


def test_deliberate_rejection_is_neither() -> None:
    executor = _executor(
        RaisingAdapter("owner-a", ClusterExecutorError("stale fencing token"))
    )

    result = executor._execute(_remote_command())

    assert result.status_source == "executor-rejected"
    assert executor.unexpected_failures == 0


@pytest.mark.parametrize("status_code", [408, 425, 429, 500, 503])
def test_control_plane_transient_failure_keeps_command_retryable(
    status_code: int,
) -> None:
    executor, result = _run_command(
        RaisingAdapter(
            "owner-a",
            ClusterExecutorError(
                f"regional control plane rejected request ({status_code})",
                status_code=status_code,
            ),
        ),
        _remote_command(),
    )

    assert result.status is RemoteCommandStatus.WAITING
    assert result.status_source == "executor-retryable-control-plane"
    assert result.details["retryable_control_plane_error"] is True
    assert result.details["status_code"] == status_code
    assert executor.unexpected_failures == 0


def test_control_plane_permanent_rejection_still_fails() -> None:
    executor, result = _run_command(
        RaisingAdapter(
            "owner-a",
            ClusterExecutorError(
                "regional control plane rejected request (403)", status_code=403
            ),
        ),
        _remote_command(),
    )

    assert result.status is RemoteCommandStatus.FAILED
    assert result.status_source == "executor-rejected"
    assert executor.unexpected_failures == 0


def test_remote_executor_rechecks_fleet_before_destructive_action() -> None:
    class TrackingAdapter(FakeAdapter):
        def __init__(self) -> None:
            super().__init__("owner-a")
            self.calls = 0

        def supports(self, _step) -> bool:
            return True

        def execute(self, _context):
            self.calls += 1
            raise AssertionError("destructive adapter must not run")

    class NotReadyRegistry:
        def readiness(self, cluster_id, node_ids):
            assert cluster_id == "cluster-a"
            assert node_ids == ["node-a"]
            return SimpleNamespace(
                ready=False,
                nodes=[
                    SimpleNamespace(
                        node_id="node-a",
                        reasons=["artifact SHA-256 mismatch: expected new, got old"],
                    )
                ],
            )

    adapter = TrackingAdapter()
    adapter.registry = NotReadyRegistry()
    executor = ClusterActionExecutor(
        FakeClient(),
        [adapter],
        executor_id="executor-a",
        allowed_namespaces={"training"},
    )
    command = _remote_command(WorkflowOperation.MARK_UNSCHEDULABLE)
    node_action = SimpleNamespace(
        operation=WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE, node_ids=["node-a"]
    )
    command.workflow_request_id = "workflow-a"
    command.workflow.blocked_reasons = []
    command.workflow.completed_step_indexes = []
    command.workflow.official_steps = [command.step, node_action]
    command.workflow.safety_steps = []
    command.incident.cluster_id = "cluster-a"

    result = executor._execute(command)

    assert result.status is RemoteCommandStatus.WAITING
    assert result.details["fleet_preflight_blocked"] is True
    assert "blocked destructive workflow" in result.details["reason"]
    assert adapter.calls == 0


def _executor_environment(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_URL", "https://control-plane.example")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_TOKEN", "token")
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "cluster-a")
    monkeypatch.setenv("GPU_FAULT_ENABLE_HYPERPOD_ADAPTER", "true")
    monkeypatch.setenv("GPU_FAULT_HYPERPOD_CLUSTER", "hp-cluster")
    monkeypatch.setenv("GPU_FAULT_HYPERPOD_CONFIRM_CLUSTER", "hp-cluster")
    # The adapter refuses to start without resolvable AWS credentials.
    # Pin the probe instead of inheriting the developer's environment:
    # otherwise these tests pass on a box with an instance role and fail
    # on a laptop without one, for a reason that has nothing to do with
    # what they assert. The guard itself is covered by
    # test_executor_refuses_hyperpod_adapter_without_aws_credentials.
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.missing_aws_credentials", lambda: None
    )


def test_executor_does_not_create_spare_coordinator_by_default(monkeypatch) -> None:
    _executor_environment(monkeypatch)
    captured = {}

    class FakeKubernetesAdapter:
        owner = "gpu-fault-kubernetes-adapter"
        core = object()

        def __init__(self, **_kwargs) -> None:
            pass

    class FakeLifecycle:
        def __init__(self, _config, *, store=None) -> None:
            self.store = store
            captured["lifecycle_store"] = store

    class FakeHyperPodStepAdapter:
        def __init__(self, _lifecycle, **kwargs) -> None:
            self.owner = kwargs["owner"]
            captured.update(kwargs)

    monkeypatch.setattr(
        "gpu_fault.cluster_executor.KubernetesWorkflowAdapter", FakeKubernetesAdapter
    )
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.HyperPodLifecycleAdapter", FakeLifecycle
    )
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.HyperPodLifecycleStepAdapter",
        FakeHyperPodStepAdapter,
    )

    executor_from_environment()

    assert captured["spare_coordinator"] is None
    assert captured["registry"].client.cluster_id == "cluster-a"


def test_hyperpod_executor_requires_independent_confirmation(monkeypatch) -> None:
    _executor_environment(monkeypatch)
    monkeypatch.delenv("GPU_FAULT_HYPERPOD_CONFIRM_CLUSTER", raising=False)

    class FakeKubernetesAdapter:
        owner = "gpu-fault-kubernetes-adapter"
        core = object()

        def __init__(self, **_kwargs) -> None:
            pass

    monkeypatch.setattr(
        "gpu_fault.cluster_executor.KubernetesWorkflowAdapter", FakeKubernetesAdapter
    )

    with pytest.raises(
        ClusterExecutorError, match="GPU_FAULT_HYPERPOD_CONFIRM_CLUSTER is required"
    ):
        executor_from_environment()


def test_executor_refuses_hyperpod_adapter_without_aws_credentials(monkeypatch) -> None:
    _executor_environment(monkeypatch)
    # Undo the pin _executor_environment installs: this is the one test
    # that exercises the probe rather than working around it.
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.missing_aws_credentials",
        lambda: "no AWS credentials are resolvable in this pod",
    )

    class FakeKubernetesAdapter:
        owner = "gpu-fault-kubernetes-adapter"
        core = object()

        def __init__(self, **_kwargs) -> None:
            pass

    monkeypatch.setattr(
        "gpu_fault.cluster_executor.KubernetesWorkflowAdapter", FakeKubernetesAdapter
    )
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.HyperPodLifecycleAdapter",
        lambda _config, **_kwargs: pytest.fail(
            "the credential check must run before the adapter is built"
        ),
    )

    with pytest.raises(
        ClusterExecutorError, match="eks.amazonaws.com/role-arn"
    ) as raised:
        executor_from_environment()

    # Fail at startup, not at the first real GPU fault hours later.
    assert "HyperPod adapter is enabled" in str(raised.value)


def test_executor_starts_without_credentials_when_hyperpod_is_off(monkeypatch) -> None:
    _executor_environment(monkeypatch)
    monkeypatch.setenv("GPU_FAULT_ENABLE_HYPERPOD_ADAPTER", "false")
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.missing_aws_credentials",
        lambda: pytest.fail(
            "an executor that makes no AWS calls must not require credentials"
        ),
    )

    class FakeKubernetesAdapter:
        owner = "gpu-fault-kubernetes-adapter"
        core = object()

        def __init__(self, **_kwargs) -> None:
            pass

    monkeypatch.setattr(
        "gpu_fault.cluster_executor.KubernetesWorkflowAdapter", FakeKubernetesAdapter
    )

    executor = executor_from_environment()

    assert executor.confirm_cluster_name is None


def test_hyperpod_executor_requires_fleet_registry(monkeypatch) -> None:
    _executor_environment(monkeypatch)

    class FakeKubernetesAdapter:
        owner = "gpu-fault-kubernetes-adapter"
        core = object()

        def __init__(self, **_kwargs) -> None:
            pass

    class FakeLifecycle:
        def __init__(self, _config, *, store=None) -> None:
            self.store = store

    monkeypatch.setattr(
        "gpu_fault.cluster_executor.KubernetesWorkflowAdapter", FakeKubernetesAdapter
    )
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.HyperPodLifecycleAdapter", FakeLifecycle
    )
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.RegionalFleetRegistry", lambda _client: None
    )

    with pytest.raises(ClusterExecutorError, match="requires a fleet registry"):
        executor_from_environment()


def test_executor_builds_spare_coordinator_when_explicitly_enabled(monkeypatch) -> None:
    _executor_environment(monkeypatch)
    monkeypatch.setenv("GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER", "true")
    captured = {}

    class FakeKubernetesAdapter:
        owner = "gpu-fault-kubernetes-adapter"
        core = object()

        def __init__(self, **_kwargs) -> None:
            pass

    class FakeLifecycle:
        def __init__(self, _config, *, store=None) -> None:
            self.store = store
            captured["lifecycle_store"] = store

    class FakeRegistry:
        def __init__(self, client) -> None:
            self.store = object()
            captured["registry_client"] = client

    class FakeSpareCoordinator:
        def __init__(self, lifecycle, actual_store, core, **kwargs) -> None:
            captured["coordinator_args"] = (lifecycle, actual_store, core, kwargs)

    class FakeHyperPodStepAdapter:
        def __init__(self, _lifecycle, **kwargs) -> None:
            self.owner = kwargs["owner"]
            captured["step_kwargs"] = kwargs

    monkeypatch.setattr(
        "gpu_fault.cluster_executor._persistent_store_from_environment",
        lambda: pytest.fail("remote state must not open a database"),
    )
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.KubernetesWorkflowAdapter", FakeKubernetesAdapter
    )
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.HyperPodLifecycleAdapter", FakeLifecycle
    )
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.RegionalFleetRegistry", FakeRegistry
    )
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.HyperPodSpareCoordinator", FakeSpareCoordinator
    )
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.HyperPodLifecycleStepAdapter",
        FakeHyperPodStepAdapter,
    )

    executor_from_environment()

    assert captured["registry_client"].cluster_id == "cluster-a"
    assert captured["coordinator_args"][1] is captured["step_kwargs"]["registry"].store
    assert (
        captured["coordinator_args"][3]["remote_health_provider"]
        is captured["step_kwargs"]["registry"]
    )
    assert (
        captured["step_kwargs"]["spare_coordinator"].__class__ is FakeSpareCoordinator
    )
    assert captured["step_kwargs"]["store"] is None


@pytest.mark.parametrize("enabled", ["true", "1", "yes", "ON"])
def test_executor_spare_failover_requires_remote_state(monkeypatch, enabled) -> None:
    # Every enabled token must arm the switch (S9): `=1` used to read as off.
    _executor_environment(monkeypatch)
    monkeypatch.setenv("GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER", enabled)
    monkeypatch.setenv("GPU_FAULT_CLUSTER_EXECUTOR_REMOTE_STATE", "false")
    monkeypatch.delenv("GPU_FAULT_STORE_URL", raising=False)
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.KubernetesWorkflowAdapter",
        lambda **_kwargs: FakeAdapter("gpu-fault-kubernetes-adapter"),
    )
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.HyperPodLifecycleAdapter",
        lambda _config, **_kwargs: object(),
    )

    with pytest.raises(ClusterExecutorError, match="requires.*REMOTE_STATE=true"):
        executor_from_environment()


def test_executor_remote_state_does_not_require_agent_secret(monkeypatch) -> None:
    _executor_environment(monkeypatch)
    monkeypatch.setenv("GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER", "true")
    monkeypatch.setenv("GPU_FAULT_AGENT_REGISTRATION_SECRET", "short")
    monkeypatch.setattr(
        "gpu_fault.cluster_executor._persistent_store_from_environment",
        lambda: pytest.fail("remote state must not open a database"),
    )

    class FakeKubernetesAdapter:
        owner = "gpu-fault-kubernetes-adapter"
        core = object()

        def __init__(self, **_kwargs) -> None:
            pass

    monkeypatch.setattr(
        "gpu_fault.cluster_executor.KubernetesWorkflowAdapter", FakeKubernetesAdapter
    )
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.HyperPodLifecycleAdapter",
        lambda _config, **_kwargs: object(),
    )

    executor = executor_from_environment()

    hyperpod = next(
        adapter
        for adapter in executor.adapters
        if adapter.owner == "gpu-fault-hyperpod-adapter"
    )
    assert hyperpod.registry is not None


def test_executor_confirms_cluster_from_its_own_configuration(monkeypatch) -> None:
    _executor_environment(monkeypatch)

    class FakeKubernetesAdapter:
        owner = "gpu-fault-kubernetes-adapter"
        core = object()

        def __init__(self, **_kwargs) -> None:
            pass

    monkeypatch.setattr(
        "gpu_fault.cluster_executor.KubernetesWorkflowAdapter", FakeKubernetesAdapter
    )
    monkeypatch.setattr(
        "gpu_fault.cluster_executor.HyperPodLifecycleAdapter",
        lambda _config, **_kwargs: object(),
    )

    executor = executor_from_environment()

    # Not the cluster_id the command carries: the HyperPod adapter's
    # confirmation gate has to compare against an independent source.
    assert executor.confirm_cluster_name == "hp-cluster"


def test_execution_request_does_not_let_a_command_confirm_itself() -> None:
    executor = ClusterActionExecutor(
        FakeClient(),
        [FakeAdapter("gpu-fault-hyperpod-adapter")],
        executor_id="executor-a",
        allowed_namespaces=set(),
        confirm_cluster_name="hp-cluster",
    )

    class FakeCommand:
        fencing_token = 3
        cluster_id = "attacker-supplied-cluster"
        restart_authorization = None

    request = executor._execution_request(FakeCommand())

    assert request.confirm_cluster_name == "hp-cluster"
    assert request.confirm_cluster_name != FakeCommand.cluster_id


def test_execution_request_confirmation_is_absent_without_config() -> None:
    executor = ClusterActionExecutor(
        FakeClient(),
        [FakeAdapter("gpu-fault-kubernetes-adapter")],
        executor_id="executor-a",
        allowed_namespaces=set(),
    )

    class FakeCommand:
        fencing_token = 3
        cluster_id = "cluster-a"
        restart_authorization = None

    # An executor that owns no HyperPod mutations must not manufacture a
    # confirmation; the adapter's gate then fails closed.
    assert executor._execution_request(FakeCommand()).confirm_cluster_name is None


class _OutcomeAdapter(FakeAdapter):
    """An adapter that reports one fixed step status.

    The idle-path tests are about what ``run_once`` concludes from a batch of
    results, so they drive the real ``_execute`` and let the adapter decide the
    status, rather than replacing ``_execute`` itself.
    """

    def __init__(self, status: WorkflowStepStatus) -> None:
        super().__init__("owner-a")
        self.status = status
        self.calls = 0

    def supports(self, _step) -> bool:
        return True

    def execute(self, _context) -> WorkflowStepOutcome:
        self.calls += 1
        return WorkflowStepOutcome(status=self.status)


class _ReclaimingClient(FakeClient):
    """Hands out the same claimed command on every claim."""

    def __init__(self) -> None:
        super().__init__()
        self.claims = 0
        self.completions: list[RemoteCommandStatus] = []

    def claim(self, executor_id, **kwargs):
        super().claim(executor_id, **kwargs)
        self.claims += 1
        return [_remote_command()]

    def complete(self, _command, result) -> None:
        self.completions.append(result.status)

    def renew(self, *_args, **_kwargs) -> None:
        return None


def test_a_held_command_polls_instead_of_spinning(monkeypatch) -> None:
    """WAITING is not progress, so it must not skip the idle sleep.

    A command that reports WAITING is immediately re-claimable, so the claim
    loop went straight back to claim() with nothing changed. On 2026-09-04 one
    held STOP_WORKLOADS drove ~25 claim/execute/complete round trips a second
    across two replicas and 759 identical log lines a minute -- for ten
    minutes, against a control plane that had nothing new to say.
    """

    client = _ReclaimingClient()
    sleeps: list[float] = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 3:
            raise StopIteration

    monkeypatch.setattr("gpu_fault.cluster_executor.time.sleep", sleep)
    executor = ClusterActionExecutor(
        client,
        [_OutcomeAdapter(WorkflowStepStatus.WAITING)],
        executor_id="executor-a",
        allowed_namespaces={"training"},
        poll_seconds=2,
    )

    with pytest.raises(StopIteration):
        executor.run()

    assert sleeps == [2, 2, 2], "a held command must take the idle path"
    assert client.claims == 3
    assert executor.last_cycle_advanced is False


def test_a_command_that_advances_does_not_wait_for_the_next_poll() -> None:
    """The backoff must not slow a queue that is actually draining."""

    client = _ReclaimingClient()
    executor = ClusterActionExecutor(
        client,
        [_OutcomeAdapter(WorkflowStepStatus.SUCCEEDED)],
        executor_id="executor-a",
        allowed_namespaces={"training"},
    )

    assert executor.run_once() == 1
    assert client.completions == [RemoteCommandStatus.SUCCEEDED]
    assert executor.last_cycle_advanced is True
