from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import Any

import pytest
from kubernetes.client.exceptions import ApiException
from urllib3.exceptions import HTTPError

from gpu_fault.regional import RemoteCommandResult
from gpu_fault.remote_command_models import RemoteCommandStatus
from scripts.e2e.regional.probes import destr008_cancellation_probe as probe
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from tests.regional.test_destr008_cancellation_probe import (
    NOW,
    FakeApi,
    MemoryCPU,
    acknowledge,
    armed_submission,
    claim,
    command,
    plan,
    seed,
    setup,
    terminalize,
)


@pytest.fixture
def restore_logging() -> Iterator[None]:
    before = logging.root.manager.disable
    try:
        yield
    finally:
        logging.disable(before)


def main_bridge(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tls: bool = True,
    host: str = "https://cpu.invalid",
) -> tuple[wire.Plan, FakeApi, MemoryCPU, list[bool]]:
    bound = plan()
    api = FakeApi(bound)
    store = MemoryCPU()
    closed: list[bool] = []

    def incluster(*, client_configuration: Any) -> None:
        client_configuration.verify_ssl = tls
        client_configuration.host = host

    class Client:
        def __init__(self, configuration: Any) -> None:
            assert configuration.host == host, (
                "bridge must use the incluster configuration"
            )

        def __enter__(self) -> object:
            return self

        def __exit__(self, *args: Any) -> None:
            closed.append(True)

    monkeypatch.setattr(probe.config, "load_incluster_config", incluster)
    monkeypatch.setattr(probe.client, "ApiClient", Client)
    monkeypatch.setattr(probe.client, "CoreV1Api", lambda _: api)
    monkeypatch.setattr(probe, "cpu_store", lambda: store)
    monkeypatch.setattr(probe.time, "time", lambda: float(NOW))
    return bound, api, store, closed


def arguments(bound: wire.Plan) -> list[str]:
    return [
        "--namespace",
        "cpu",
        "--configmap",
        "watchdog",
        "--uid",
        "cm-uid",
        "--plan-sha256",
        wire.digest(bound),
    ]


def test_real_main_bridge_uses_incluster_api_and_persists_final_receipt(
    monkeypatch: pytest.MonkeyPatch, capsys, restore_logging
) -> None:
    bound, api, store, closed = main_bridge(monkeypatch)
    run = probe.run
    ticks = iter([NOW, bound.deadline_at, bound.deadline_at + 5])
    monkeypatch.setattr(
        probe,
        "run",
        lambda port, store: run(
            port, store, clock=lambda: next(ticks), sleep=lambda _: None
        ),
    )
    assert probe.main(arguments(bound)) == 0
    outputs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item["state"] for item in outputs] == ["ARMED", "REVOKED", "QUIESCENT"]
    final = wire.decode(wire.Receipt, api.value.data["status.json"])
    assert final.state == "QUIESCENT" and final.source_complete
    assert final.configmap_uid == "cm-uid" and final.fence == bound.fence
    assert store.closed and closed == [True]
    assert all(not item["fence_release_authorized"] for item in outputs), (
        "CPU receipts must never authorize GPU fence release"
    )


def test_main_reuses_done_tombstone_without_opening_the_store(
    monkeypatch: pytest.MonkeyPatch, capsys, restore_logging
) -> None:
    bound, api, _, closed = main_bridge(monkeypatch)
    port = probe.KubernetesControlMap(
        api,
        namespace="cpu",
        name="watchdog",
        uid="cm-uid",
        plan_sha256=wire.digest(bound),
        probe_sha256=bound.probe_sha256,
    )
    watchdog = probe.Watchdog(port, MemoryCPU(), sleep=lambda _: None)
    watchdog.tick(NOW)
    watchdog.tick(bound.deadline_at)
    final = watchdog.tick(bound.deadline_at + 5)
    monkeypatch.setattr(probe.time, "time", lambda: float(bound.deadline_at + 10))
    monkeypatch.setattr(
        probe,
        "cpu_store",
        lambda: (_ for _ in ()).throw(AssertionError("Store must remain unopened")),
    )
    before = list(api.patches)
    assert probe.main(arguments(bound)) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "QUIESCENT"
    assert wire.decode(wire.Receipt, api.value.data["status.json"]) == final
    assert api.patches == before and closed == [True]


def test_main_sanitizes_store_error_and_suppresses_upstream_credential_logging(
    monkeypatch: pytest.MonkeyPatch, capsys, caplog, restore_logging
) -> None:
    bound, api, _, closed = main_bridge(monkeypatch)
    private = "postgresql://local-user:private-test-value@invalid.example/database"

    def unavailable() -> Any:
        logging.getLogger("psycopg.pool").error(private)
        raise RuntimeError(private)

    monkeypatch.setattr(probe, "cpu_store", unavailable)
    assert probe.main(arguments(bound)) == 1
    output = capsys.readouterr()
    assert private not in output.out + output.err + caplog.text
    status = wire.decode(wire.Receipt, api.value.data["status.json"])
    assert status.state == "FAILED" and status.error_code == "CPU_STORE_ERROR"
    assert status.producer_revoked and not status.fence_release_authorized
    assert json.loads(output.out)["configmap_uid"] == "cm-uid"
    assert closed == [True]


@pytest.mark.parametrize("field", ["control.json", "status.json"])
def test_bad_control_or_status_has_a_durable_sanitized_failure(
    field: str, monkeypatch: pytest.MonkeyPatch, capsys, restore_logging
) -> None:
    bound, api, store, _ = main_bridge(monkeypatch)
    api.value.data[field] = '{"unknown":"private-input-must-not-be-echoed"}'
    assert probe.main(arguments(bound)) == 1
    output = capsys.readouterr().out
    assert "private-input-must-not-be-echoed" not in output
    status = wire.decode(wire.Receipt, api.value.data["status.json"])
    assert status.state == "FAILED" and status.error_code == "DOCUMENT_SHAPE"
    assert status.configmap_uid == "cm-uid" and not status.monitoring
    assert not store.closed, (
        "invalid control must be rejected before Store construction"
    )


@pytest.mark.parametrize(
    ("tls", "host"), [(False, "https://cpu.invalid"), (True, "http://cpu.invalid")]
)
def test_main_refuses_unverified_api_transport(
    tls: bool, host: str, monkeypatch: pytest.MonkeyPatch, capsys, restore_logging
) -> None:
    bound, api, _, closed = main_bridge(monkeypatch, tls=tls, host=host)
    assert probe.main(arguments(bound)) == 1
    assert json.loads(capsys.readouterr().out)["error_code"] == "CPU_API_TLS_REQUIRED"
    assert api.patches == [] and closed == []


def test_main_never_loads_a_gpu_kubeconfig(
    monkeypatch: pytest.MonkeyPatch, capsys, restore_logging
) -> None:
    monkeypatch.setenv("KUBECONFIG", "/not-an-authorized-kubeconfig")
    monkeypatch.setattr(
        probe.config,
        "load_incluster_config",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("must refuse before API configuration")
        ),
    )
    assert probe.main(arguments(plan())) == 1
    assert json.loads(capsys.readouterr().out)["error_code"] == "KUBECONFIG_FORBIDDEN"


def test_cpu_store_uses_existing_env_without_schema_or_large_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    memory = MemoryCPU()
    url = "postgresql://localhost/isolated?sslmode=verify-full"
    monkeypatch.setenv("GPU_FAULT_STORE_URL", url)
    monkeypatch.setenv("GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT", "false")
    monkeypatch.setenv("GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "80")

    def factory(dsn: str, **kwargs: Any) -> MemoryCPU:
        calls.append((dsn, kwargs))
        return memory

    monkeypatch.setattr(probe, "PostgresStore", factory)
    assert probe.cpu_store() is memory
    assert calls == [
        (
            url,
            {
                "initialize_schema": False,
                "pool_min_size": 0,
                "pool_max_size": 2,
                "pool_timeout_seconds": 2,
            },
        )
    ]
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "sqlite:///must-not-be-used")
    with pytest.raises(wire.ProbeError, match="CPU_STORE_CONFIGURATION"):
        probe.cpu_store()
    monkeypatch.setenv("GPU_FAULT_STORE_URL", url)
    monkeypatch.setenv("GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT", "true")
    with pytest.raises(wire.ProbeError, match="CPU_STORE_CONFIGURATION"):
        probe.cpu_store()
    assert len(calls) == 1, (
        "unsafe Store settings must fail before opening a connection"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"namespace": "../cpu"},
        {"name": ""},
        {"uid": "bad/uid"},
        {"plan_sha256": "x"},
        {"probe_sha256": "x"},
    ],
)
def test_api_reference_shape_is_strict(change: dict[str, str]) -> None:
    bound = plan()
    values = {
        "namespace": "cpu",
        "name": "watchdog",
        "uid": "cm-uid",
        "plan_sha256": wire.digest(bound),
        "probe_sha256": bound.probe_sha256,
        **change,
    }
    with pytest.raises(wire.ProbeError, match="CONFIGMAP_BINDING"):
        probe.KubernetesControlMap(FakeApi(bound), **values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("api_version", "v2"),
        ("kind", "Secret"),
        ("binary_data", {"x": "ignored"}),
        ("immutable", True),
        ("metadata.name", "foreign"),
        ("metadata.namespace", "foreign"),
        ("metadata.uid", "replacement"),
        ("metadata.resource_version", None),
        ("metadata.resource_version", "invalid version"),
        ("metadata.deletion_timestamp", "deleting"),
        ("data", {}),
        ("data", None),
    ],
)
def test_unknown_or_recreated_configmap_is_not_patched(field: str, value: Any) -> None:
    bound, api, port, _, _ = setup()
    target: Any = api.value
    if "." in field:
        parent, field = field.split(".")
        target = getattr(target, parent)
    setattr(target, field, value)
    with pytest.raises(wire.ProbeError, match="CONFIGMAP_BINDING"):
        port.read()
    assert api.patches == [], "name alone never authorizes a write"


def test_plan_or_source_drift_is_refused_before_store_access() -> None:
    _, api, port, _, _ = setup()
    api.value.data["plan.json"] = wire.encode(plan(release_id="drift"))
    with pytest.raises(wire.ProbeError, match="PLAN_SOURCE"):
        port.read()
    assert api.patches == []
    bound = plan()
    changed = probe.KubernetesControlMap(
        FakeApi(bound),
        namespace="cpu",
        name="watchdog",
        uid="cm-uid",
        plan_sha256=wire.digest(bound),
        probe_sha256="b" * 64,
    )
    with pytest.raises(wire.ProbeError, match="PLAN_SOURCE"):
        changed.read()


def test_patch_carries_uid_version_and_all_data_preconditions() -> None:
    bound, api, port, _, watchdog = setup()
    before = port.read()
    status = watchdog.tick(NOW)
    patch = api.patches[0]
    tests = {item["path"]: item["value"] for item in patch if item["op"] == "test"}
    assert tests == {
        "/metadata/uid": "cm-uid",
        "/metadata/resourceVersion": before.envelope.version,
        **{f"/data/{key}": value for key, value in before.envelope.data.items()},
    }
    assert wire.decode(wire.Receipt, api.value.data["status.json"]) == status
    with pytest.raises(ApiException) as caught:
        port.write(before, before.control, status)
    assert caught.value.status == 409
    assert port.read().status == status
    api.value.metadata.uid = "replacement"
    with pytest.raises(wire.ProbeError, match="RECEIPT_UNAVAILABLE"):
        watchdog.tick(bound.deadline_at)
    assert api.value.metadata.uid == "replacement"


def test_cas_claim_racing_deadline_prevents_false_not_started_tombstone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, api, port, store, watchdog = setup()
    watchdog.tick(NOW)
    submitting = wire.claim_submission(
        bound, port.read().control, claim_id="claim-a", now=bound.deadline_at - 1
    )
    patch = api.patch_namespaced_config_map
    fired: list[bool] = []

    def race(*args: Any, **kwargs: Any) -> Any:
        if not fired:
            fired.append(True)
            api.value.data["control.json"] = wire.encode(submitting)
            api.value.metadata.resource_version = str(
                int(api.value.metadata.resource_version) + 1
            )
        return patch(*args, **kwargs)

    monkeypatch.setattr(api, "patch_namespaced_config_map", race)
    status = watchdog.tick(bound.deadline_at)
    assert status.state == "FAILED" and status.error_code == "SOURCE_UNRESOLVED"
    assert status.producer is not None and status.producer.state == "SUBMITTING"
    assert port.read().control.revocation.producer_state == "SUBMITTING"
    assert store.list_workflows() == []


def test_lost_revocation_ack_resumes_the_committed_state() -> None:
    bound, api, port, _, watchdog = setup()
    watchdog.tick(NOW)

    def lost_ack() -> None:
        raise TimeoutError("private-server-response")

    api.after_patch = lost_ack
    result = watchdog.tick(bound.deadline_at)
    assert result.state == "REVOKED" and result.revocation.at == bound.deadline_at
    assert watchdog.tick(bound.deadline_at + 5).state == "QUIESCENT"
    assert port.read().control.revocation.at == bound.deadline_at


@pytest.mark.parametrize(
    "error",
    [
        ApiException(status=409),
        ApiException(status=422),
        ApiException(status=429),
        ApiException(status=500),
        ApiException(status=502),
        ApiException(status=503),
        ApiException(status=504),
        HTTPError("local-api"),
        TimeoutError("local-api"),
    ],
)
def test_transient_api_errors_are_bounded_and_retried(error: Exception) -> None:
    _, api, port, store, _ = setup()
    api.read_errors = [error]
    waits: list[float] = []
    status = probe.Watchdog(port, store, sleep=waits.append).tick(NOW)
    assert status.state == "ARMED"
    assert waits == [0.25] and port.read().status == status


def test_api_retry_exhaustion_persists_failure_without_raw_response() -> None:
    _, api, port, store, _ = setup()
    api.read_errors = [
        ApiException(status=503, reason="private-server-body")
    ] * probe.RETRY_ATTEMPTS
    waits: list[float] = []
    status = probe.Watchdog(port, store, sleep=waits.append).tick(NOW)
    assert status.state == "FAILED" and status.error_code == "API_UNAVAILABLE"
    assert waits == [0.25, 0.5, 1.0]
    assert "private-server-body" not in wire.encode(status)
    assert port.read().status == status


def test_nontransient_api_failure_does_not_retry_as_a_conflict() -> None:
    _, api, port, store, _ = setup()
    api.read_errors = [ApiException(status=403, reason="private-authorization")]
    waits: list[float] = []
    status = probe.Watchdog(port, store, sleep=waits.append).tick(NOW)
    assert status.state == "FAILED" and status.error_code == "API_UNAVAILABLE"
    assert waits == [] and not status.fence_release_authorized


def test_transient_store_conflict_is_retried_using_existing_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from psycopg.errors import SerializationFailure

    bound, _, port, store, _, _, _ = armed_submission()
    read = store.list_job_recovery_workflow_incidents
    calls: list[int] = []

    def serialization(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise SerializationFailure("private-dsn-context")
        return read(*args, **kwargs)

    monkeypatch.setattr(store, "list_job_recovery_workflow_incidents", serialization)
    waits: list[float] = []
    result = probe.Watchdog(port, store, sleep=waits.append).tick(bound.deadline_at)
    assert result.state == "REVOKED" and waits == [0.25]
    assert "private-dsn-context" not in wire.encode(result)


def test_failure_receipt_retries_its_own_cas_and_never_swallows_exhaustion() -> None:
    _, api, port, _, _ = setup()
    api.patch_errors = [ApiException(status=409)]
    waits: list[float] = []
    result = probe.persist_failure(
        port, code="LOCAL_FAILURE", now=NOW, monitoring=False, sleep=waits.append
    )
    assert result.state == "FAILED" and waits == [0.25]
    api.patch_errors = [ApiException(status=503)] * probe.RETRY_ATTEMPTS
    with pytest.raises(wire.ProbeError, match="RECEIPT_UNAVAILABLE"):
        probe.persist_failure(
            port, code="LOCAL_FAILURE", now=NOW, monitoring=False, sleep=waits.append
        )
    assert len(waits) == 4, "failure publication has a bounded independent retry budget"


@pytest.mark.parametrize("change", ["no_version", "wrong_control", "wrong_status"])
def test_api_must_acknowledge_the_exact_cas_write(
    change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound, api, port, _, _ = setup()
    snapshot = port.read()
    status = wire.receipt(
        bound, snapshot.control, None, uid=port.uid, now=NOW, state="ARMED"
    )
    patch = api.patch_namespaced_config_map

    def changed_response(*args: Any, **kwargs: Any) -> Any:
        response = patch(*args, **kwargs)
        if change == "no_version":
            response.metadata.resource_version = snapshot.envelope.version
        elif change == "wrong_control":
            response.data["control.json"] = "{}"
        else:
            response.data["status.json"] = "null"
        return response

    monkeypatch.setattr(api, "patch_namespaced_config_map", changed_response)
    with pytest.raises(wire.ProbeError, match="CAS_NOT_ACKNOWLEDGED"):
        port.write(snapshot, snapshot.control, status)


def test_changed_done_record_cannot_be_returned_as_success_by_failure_handler() -> None:
    bound, api, _, _, watchdog = setup()
    watchdog.tick(NOW)
    watchdog.tick(bound.deadline_at)
    watchdog.tick(bound.deadline_at + 5)
    previous = json.loads(api.value.data["status.json"])
    previous["run_id"] = "wrong-run"
    api.value.data["status.json"] = wire.encode(previous)
    result = watchdog.tick(bound.deadline_at + 6)
    assert result.state == "FAILED" and result.error_code == "RECEIPT_BINDING"
    assert result.run_id == bound.run_id and not result.fence_release_authorized


LOCAL_AUTH = b"destr008-local-process-test"


def fresh_tick() -> None:
    """Entry point for a new interpreter connected only to local controlled fakes."""
    manager = BaseManager(address=("127.0.0.1", int(sys.argv[1])), authkey=LOCAL_AUTH)
    manager.register("api")
    manager.register("store")
    manager.connect()
    bound = wire.Plan.model_validate_json(sys.argv[2])
    port = probe.KubernetesControlMap(
        manager.api(),
        namespace="cpu",
        name="watchdog",
        uid="cm-uid",
        plan_sha256=wire.digest(bound),
        probe_sha256=bound.probe_sha256,
    )
    options = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    later_ticks = options.pop("later_ticks", [])
    watchdog = probe.Watchdog(port, manager.store(), sleep=lambda _: None, **options)
    for at in [int(sys.argv[3]), *later_ticks]:
        status = watchdog.tick(at)
    print(wire.encode({"pid": os.getpid(), "receipt": status.model_dump(mode="json")}))


def test_fresh_process_parent_loss_late_event_and_still_leased_control() -> None:
    bound, api, port, store, _ = setup()
    manager = BaseManager(address=("127.0.0.1", 0), authkey=LOCAL_AUTH)
    manager.register("api", callable=lambda: api)
    manager.register("store", callable=lambda: store)
    server = manager.get_server()

    def serve() -> None:
        try:
            server.serve_forever()
        except SystemExit as exit_status:
            assert exit_status.code == 0, "local server must stop normally"

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    env = {
        "HOME": "/tmp",
        "PATH": "/home/ubuntu/GPU-Fault-Recovery-System/GPU-Fault-Recovery-System/.venv/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": "src:.",
        "PYTHONDONTWRITEBYTECODE": "1",
        "AWS_CONFIG_FILE": "/dev/null",
        "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
        "AWS_EC2_METADATA_DISABLED": "true",
        "KUBECONFIG": "/dev/null",
    }
    pids: list[int] = []

    def tick(now: int, **options: Any) -> wire.Receipt:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from tests.regional.test_destr008_cancellation_bridge import fresh_tick; fresh_tick()",
                str(server.address[1]),
                wire.encode(bound),
                str(now),
                wire.encode(options),
            ],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        assert not completed.stderr, "local child must emit no raw diagnostic output"
        value = json.loads(completed.stdout)
        pids.append(value["pid"])
        return wire.Receipt.model_validate(value["receipt"])

    try:
        assert tick(NOW).state == "ARMED"
        claim(bound, port)
        unknown = tick(bound.deadline_at)
        assert unknown.state == "FAILED" and unknown.error_code == "SOURCE_UNRESOLVED"
        incident, workflow = seed(store, bound)
        remote = command(store, incident, workflow, status=RemoteCommandStatus.LEASED)
        late = tick(bound.deadline_at + 1)
        assert late.state == "FAILED" and late.commands_active == 1
        assert store.get_workflow(workflow.request_id).workload_withdrawn_at is not None
        assert (
            store.get_remote_command(remote.command_id).cancellation_requested_at
            is not None
        )
        unresolved = tick(bound.deadline_at + wire.DRAIN_SECONDS + 1)
        assert (
            unresolved.state == "FAILED"
            and unresolved.producer_revoked
            and unresolved.monitoring
        )
        assert (
            unresolved.commands_active == 1 and not unresolved.fence_release_authorized
        )
        assert (
            store.get_remote_command(remote.command_id).status
            is RemoteCommandStatus.LEASED
        )
        assert port.read().status == unresolved
        later = bound.deadline_at + wire.DRAIN_SECONDS + 2
        acknowledge(bound, port, now=later)
        failed = probe.persist_failure(
            port, code="LOCAL_WATCHDOG_TIMEOUT", now=later + 1, monitoring=False
        )
        store.complete_remote_command(
            bound.cluster_id,
            remote.command_id,
            RemoteCommandResult(
                lease_token="local-test-lease",
                status=RemoteCommandStatus.FAILED,
                error="local physical execution completed",
            ),
        )
        terminalize(store)
        options = {
            "cleanup_only": True,
            "cleanup_attempt_id": "process-cleanup",
            "cleanup_seconds": 30,
        }
        observed = tick(later + 2, **options)
        assert observed.state == "REVOKED" and observed.failure == failed.failure
        done = tick(later + 3, **options, later_ticks=[later + 8])
        assert done.state == "QUIESCENT" and done.case_failed
        assert done.failure == failed.failure and done.cleanup == observed.cleanup
        assert done.sequence > observed.sequence and port.read().status == done
        assert len(set(pids)) == 6, (
            "every watchdog/cleanup restart must use a fresh interpreter"
        )
    finally:
        server.stop_event.set()
        thread.join(timeout=5)
    assert not thread.is_alive(), "the local control server must be reaped"


def test_standalone_probe_loads_split_helpers_without_repo_package_context() -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(probe.__file__).resolve()), "--help"],
        cwd="/tmp",
        env={
            "HOME": "/tmp",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "AWS_CONFIG_FILE": "/dev/null",
            "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
            "AWS_EC2_METADATA_DISABLED": "true",
            "KUBECONFIG": "/dev/null",
        },
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    assert "--namespace" in completed.stdout and "--plan-sha256" in completed.stdout
    assert not completed.stderr, "standalone probe help must not emit stderr"
