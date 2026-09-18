from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.regional import RemoteCommandResult, RemoteCommandStatus
from gpu_fault.store import InMemoryStore
from scripts.e2e.regional import run_ha005_rollout_continuity as ha005
from scripts.e2e.regional import run_ha006_executor_takeover as ha006
from scripts.e2e.regional import run_ha009_aurora_credential_rotation as ha009
from tests.regional._cov95_ha001_harness import Clock, deployment, pod


@pytest.mark.parametrize("module", [ha005, ha006])
def test_residual_readers_preserve_identity_but_do_not_export_fixture_credentials(
    module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        module,
        "load_registry",
        lambda: [
            {"cluster_id": "regular"},
            {
                "cluster_id": "perf-cap-000",
                "synthetic_run_id": "unit",
                "token": "REPLACE_WITH_UNIT",
            },
            {"cluster_id": "synthetic-other", "synthetic": True},
        ],
    )
    result = module.registry_residuals()
    assert result["count"] == 2
    assert all(
        set(item) == {"cluster_id", "synthetic_run_id"} for item in result["entries"]
    ), "residual evidence must contain identities, not credentials"

    def dataplane(*args: str, **kwargs: Any) -> str:
        assert kwargs.get("check", True) is True
        return args[2] if args[1] == "configmap" else ""

    monkeypatch.setattr(module, "dataplane", dataplane)
    assert module.kubernetes_residuals()["count"] == 1


@pytest.mark.parametrize("module", [ha005, ha006])
@pytest.mark.parametrize("success", [False, True])
def test_probe_file_waits_are_bounded(
    module: Any, monkeypatch: pytest.MonkeyPatch, success: bool
) -> None:
    clock = Clock()
    monkeypatch.setattr(module, "time", clock)
    monkeypatch.setattr(
        module,
        "dataplane",
        lambda *a, **kw: "present" if success and clock.now >= 1 else "",
    )
    arguments = (
        ("/state/ready.json", 2)
        if module is ha005
        else ("unit-pod", "/state/ready.json", 2)
    )
    if success:
        assert module.wait_file(*arguments) is None
    else:
        with pytest.raises(module.CaseError, match="did not create"):
            module.wait_file(*arguments)
    assert clock.now <= 2
    monkeypatch.setattr(module, "dataplane", lambda *a, **kw: '{"unit":1}')
    assert (
        module.read_probe() if module is ha005 else module.read_state("pod", "state")
    ) == {"unit": 1}


@pytest.mark.parametrize("missing", [False, True])
def test_executor_lease_is_read_from_the_deployed_container(
    monkeypatch: pytest.MonkeyPatch, missing: bool
) -> None:
    entries = [{"name": "OTHER", "value": "0"}]
    if not missing:
        entries.append(
            {"name": "GPU_FAULT_CLUSTER_EXECUTOR_LEASE_SECONDS", "value": "30"}
        )
    monkeypatch.setattr(
        ha006,
        "dataplane",
        lambda *a, **kw: json.dumps(
            {"spec": {"template": {"spec": {"containers": [{"env": entries}]}}}}
        ),
    )
    if missing:
        with pytest.raises(ha006.CaseError, match="setting is missing"):
            ha006.production_lease_seconds()
    else:
        assert ha006.production_lease_seconds() == 30


@pytest.mark.parametrize("success", [False, True])
def test_takeover_waits_retain_waiting_and_terminal_observations(
    monkeypatch: pytest.MonkeyPatch, success: bool
) -> None:
    clock = Clock()
    monkeypatch.setattr(ha006, "time", clock)
    seed = {"command_id": "unit-command", "deduplication_key": "unit-dedup"}

    def command(command_id: str) -> dict:
        assert command_id == seed["command_id"]
        return {
            "status": "LEASED" if success and clock.now >= 0.5 else "WAITING",
            "lease_owner": ha006.PODS[0],
            "last_lease_owner": ha006.PODS[1],
            "result_details": {"round": 1},
        }

    monkeypatch.setattr(ha006, "command_snapshot", command)
    monkeypatch.setattr(
        ha006, "notification_snapshot", lambda _: {"notification_id": "unit-notice"}
    )
    if success:
        _, _, timeline = ha006.wait_first_owner(seed, timeout_seconds=2)
        assert ha006.waiting_branch(timeline)["reclaimed_by_other_replica"] is True
    else:
        with pytest.raises(ha006.CaseError, match="did not acquire"):
            ha006.wait_first_owner(seed, timeout_seconds=1)
    monkeypatch.setattr(
        ha006,
        "command_snapshot",
        lambda _: {"status": "SUCCEEDED" if success else "WAITING"},
    )
    if success:
        assert ha006.wait_terminal(seed, timeout_seconds=1)[0]["status"] == "SUCCEEDED"
    else:
        with pytest.raises(ha006.CaseError, match="did not reach SUCCEEDED"):
            ha006.wait_terminal(seed, timeout_seconds=1)


def test_takeover_verdict_rejects_each_broken_contract_field() -> None:
    result = ha006.takeover_errors(
        final={"status": "WAITING", "last_lease_owner": "old", "result_details": {}},
        survivor="new",
        notification_id="unit",
        timing={"takeover_seconds": -1, "tolerance_seconds": 1},
        remaining_lease=None,
        survivor_state={"claimed_total": 0, "unexpected_failures": 1},
        physical_total=1,
        notification_final={"dedup_link_count": 0, "objects": {}},
    )
    assert len(result) == 12
    assert "survivor did not reuse the shared action ledger" in result
    assert "command completed before the recorded kill" in result
    assert "shared ledger drill notification was not suppressed" in result


def test_seed_and_command_snapshot_execute_against_real_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryStore()
    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        classmethod(lambda cls: SimpleNamespace(store=store)),
    )

    def cpu_python(script: str, *arguments: str) -> dict:
        output = io.StringIO()
        with monkeypatch.context() as local, redirect_stdout(output):
            local.setattr(sys, "argv", ["unit-cpu", *arguments])
            exec(script, {})
        return json.loads(output.getvalue())

    monkeypatch.setattr(ha006, "cpu_python", cpu_python)
    monkeypatch.setattr(ha005, "cpu_python", cpu_python)
    first = ha006.seed_command("ha006-unit")
    assert ha006.command_snapshot(first["command_id"])["status"] == "PENDING"
    command = store.claim_remote_commands(
        "perf-cap-000",
        "unit-executor",
        limit=1,
        lease_seconds=60,
        execution_owners={ha006.OWNER},
    )[0]
    assert ha006.command_snapshot(first["command_id"])["lease_owner"] == "unit-executor"
    store.complete_remote_command(
        command.cluster_id,
        command.command_id,
        RemoteCommandResult(
            status=RemoteCommandStatus.SUCCEEDED,
            lease_token=command.lease_token,
            details={"simulated": True},
        ),
    )
    assert ha006.command_snapshot(first["command_id"])["status"] == "SUCCEEDED"
    second = ha009.seed_runtime_records("ha009-unit")
    assert store.get_notification(second["notification_id"]).drill_id == "ha009-unit"
    assert (
        store.get_remote_command(second["command_id"]).step.execution_owner
        == "gpu-fault-ha005-noop"
    )
    assert ha005.processor_receipts([]) == {"requests": [], "missing": []}
    assert ha005.processor_receipts(["missing"]) == {
        "requests": [],
        "missing": ["missing"],
    }


@pytest.mark.parametrize("module", [ha005, ha009])
def test_deployment_snapshots_retain_every_pod_and_its_readiness(
    module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = ha005.ALL_DEPLOYMENTS
    documents = {name: deployment(name, 1) for name in names}
    for document in documents.values():
        document["metadata"]["generation"] = 1
        document["status"].update(
            updatedReplicas=1, availableReplicas=1, observedGeneration=1
        )
        document["spec"]["template"]["spec"]["containers"][0]["ports"][0]["name"] = (
            "http"
        )

    def control(*args: str, **kwargs: Any) -> str:
        if args[1] == "deployment":
            return json.dumps(
                documents[args[2]]
                if module is ha005
                else {"items": list(documents.values())}
            )
        name = args[3].removeprefix("app=")
        value = pod(f"{name}-pod", name)
        value["status"]["containerStatuses"][0]["restartCount"] = 2
        return json.dumps({"items": [value]})

    monkeypatch.setattr(ha005, "control", control)
    value = (
        ha005.deployment_snapshot()
        if module is ha005
        else ha009.deployment_snapshot()[names[0]]
    )
    assert value["replicas"] == value["ready"] == 1
    assert value["pods"][0][1]["ready"] is True
    assert value["pods"][0][1]["restarts"] == 2
    if module is ha009:
        documents[names[0]]["spec"]["template"]["spec"]["containers"][0]["ports"] = []
        with pytest.raises(ha009.CaseError, match="unique HTTP port"):
            ha009.deployment_snapshot()


@pytest.mark.parametrize(
    "raw,expected",
    [("", 300), ("1.1", 2), ("0", None), ("nan", None), ("invalid", None)],
)
def test_pool_idle_budget_uses_finite_positive_configuration(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: int | None
) -> None:
    monkeypatch.setattr(ha005, "control", lambda *a, **kw: raw)
    if expected is None:
        with pytest.raises(ha009.CaseError, match="max_idle"):
            ha009.pool_max_idle_seconds()
    else:
        assert ha009.pool_max_idle_seconds() == expected


@pytest.mark.parametrize("empty", [False, True])
def test_credential_digests_never_return_the_projected_value(
    monkeypatch: pytest.MonkeyPatch, empty: bool
) -> None:
    value = b"postgresql://unit.invalid/database"
    encoded = "" if empty else base64.b64encode(value).decode()
    monkeypatch.setattr(ha005, "control", lambda *a, **kw: encoded)
    if empty:
        for function in (
            ha009.kubernetes_secret_digest,
            ha009.kubernetes_secret_dsn_digest,
        ):
            with pytest.raises(ha009.CaseError, match="no postgres-url"):
                function()
    else:
        assert (
            ha009.kubernetes_secret_digest()
            == hashlib.sha256(encoded.encode()).hexdigest()
        )
        assert ha009.kubernetes_secret_dsn_digest() == hashlib.sha256(value).hexdigest()
    monkeypatch.setattr(ha005, "control", lambda *a, **kw: "unit-digest\n")
    assert ha009.pod_dsn_file_digest("unit-pod") == "unit-digest"


@pytest.mark.parametrize("success", [False, True])
def test_rotation_and_projection_waits_are_bounded(
    monkeypatch: pytest.MonkeyPatch, success: bool
) -> None:
    clock = Clock()
    monkeypatch.setattr(ha009, "time", clock)
    monkeypatch.setattr(
        ha009,
        "PHASE_BUDGETS",
        {**ha009.PHASE_BUDGETS, "secret_propagation": 10, "managed_rotation": 10},
    )
    monkeypatch.setattr(
        ha009,
        "pod_dsn_file_digest",
        lambda _: "new" if success and clock.now >= 5 else "old",
    )
    result = ha009.wait_secret_propagated(["pod-a", "pod-b"], "new")
    assert result["pods"] == {
        pod: "new" if success else "old" for pod in ("pod-a", "pod-b")
    }
    for pods, digest in (([], "new"), (["pod"], "")):
        with pytest.raises(ha009.CaseError, match="requires Pod identities"):
            ha009.wait_secret_propagated(pods, digest)
    clock.now = 0
    monkeypatch.setattr(
        ha009,
        "secret_versions",
        lambda _: {
            "stages": {"AWSCURRENT": "new" if success and clock.now >= 5 else "old"}
        },
    )
    monkeypatch.setattr(
        ha009, "aws", lambda *a: {"DBClusters": [{"Status": "available"}]}
    )
    if success:
        assert (
            ha009.wait_rotated_secret("old", "unit-reference")["stages"]["AWSCURRENT"]
            == "new"
        )
    else:
        with pytest.raises(ha009.CaseError, match="did not complete"):
            ha009.wait_rotated_secret("old", "unit-reference")


def test_idle_observations_and_log_counts_cover_each_named_pod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    seen = []
    monkeypatch.setattr(ha009, "time", clock)
    monkeypatch.setattr(
        ha009,
        "probe_pod",
        lambda pod, port: seen.append((pod, port)) or {"fresh_connection": True},
    )
    monkeypatch.setattr(
        ha005, "control", lambda *a, **kw: ha009.AUTH_FAILURE_LOG_MARKER + "\nnormal\n"
    )
    result = ha009.observe_after_idle(
        ["a", "b"], ports={"a": 8080, "b": 8081}, samples=2, interval=1
    )
    assert seen == [("a", 8080), ("b", 8081)] * 2
    assert result["auth_failures_in_logs"] == {"a": 1, "b": 1}
    assert clock.now == 1


@pytest.mark.parametrize("success", [False, True])
def test_continuity_receipts_and_probe_shutdown_wait_for_confirmation(
    monkeypatch: pytest.MonkeyPatch, success: bool
) -> None:
    clock = Clock()
    monkeypatch.setattr(ha005, "time", clock)
    monkeypatch.setattr(
        ha005,
        "processor_receipts",
        lambda _: {
            "missing": [],
            "requests": [
                {
                    "request_id": "unit",
                    "status": "COMPLETED" if success and clock.now >= 2 else "LEASED",
                    "response_status": 200,
                }
            ],
        },
    )
    if success:
        assert (
            ha005.wait_receipts(["unit"], timeout_seconds=3)["requests"][0]["status"]
            == "COMPLETED"
        )
    else:
        with pytest.raises(ha005.CaseError, match="receipts did not converge"):
            ha005.wait_receipts(["unit"], timeout_seconds=1)
    clock.now = 0
    commands = []
    monkeypatch.setattr(ha005, "dataplane", lambda *a, **kw: commands.append(a) or "")
    monkeypatch.setattr(
        ha005, "read_probe", lambda: {"stopped": success and clock.now >= 1}
    )
    if success:
        assert ha005.stop_probe(timeout_seconds=2)["stopped"] is True
    else:
        with pytest.raises(ha005.CaseError, match="did not stop"):
            ha005.stop_probe(timeout_seconds=2)
    assert len(commands) == 1
    assert commands[0][-1] == "/state/stop"


@pytest.mark.parametrize("success", [False, True])
def test_runtime_records_need_success_and_all_three_notification_records(
    monkeypatch: pytest.MonkeyPatch, success: bool
) -> None:
    clock = Clock()
    monkeypatch.setattr(ha009, "time", clock)
    monkeypatch.setattr(
        ha009,
        "runtime_snapshot",
        lambda _: {
            "command": {
                "status": "SUCCEEDED" if success and clock.now >= 2 else "WAITING"
            },
            "notification": {
                kind: {"count": 1}
                for kind in (
                    "notification",
                    "notification_delivery",
                    "notification_result",
                )
            },
        },
    )
    if success:
        assert (
            ha009.wait_runtime_records({}, timeout_seconds=3)["command"]["status"]
            == "SUCCEEDED"
        )
    else:
        with pytest.raises(ha009.CaseError, match="did not converge"):
            ha009.wait_runtime_records({}, timeout_seconds=1)


@pytest.mark.parametrize("module", [ha006, ha009])
@pytest.mark.parametrize("remaining", [0, 1])
def test_seed_cleanup_binds_exact_ids_and_rejects_reported_residuals(
    monkeypatch: pytest.MonkeyPatch, module: Any, remaining: int
) -> None:
    seed = {
        key: f"unit-{key}"
        for key in (
            "incident_id",
            "event_id",
            "workflow_id",
            "command_id",
            "notification_id",
            "deduplication_key",
        )
    }
    calls = []

    def query(script: str, *args: str) -> dict:
        calls.append(args)
        return {"remaining_objects": remaining, "remaining_links": 0}

    monkeypatch.setattr(ha005 if module is ha009 else ha006, "cpu_python", query)
    action = (
        (lambda: ha009.cleanup_runtime_records(seed))
        if module is ha009
        else (lambda: ha006.cleanup_seed(seed, None))
    )
    if remaining:
        with pytest.raises(module.CaseError, match="cleanup left residuals"):
            action()
    else:
        assert action()["remaining_objects"] == 0
    assert calls[0][:4] == tuple(
        seed[key] for key in ("incident_id", "event_id", "workflow_id", "command_id")
    )
    assert calls[0][4:] == (
        (seed["notification_id"], seed["deduplication_key"])
        if module is ha009
        else (seed["deduplication_key"], "")
    )


def test_rotation_metadata_and_aws_wrapper_do_not_relax_response_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        ha009,
        "run_fixture_command",
        lambda argv, **kw: calls.append(argv)
        or subprocess.CompletedProcess(
            argv,
            0,
            '{"Versions":[{"VersionId":"v1","VersionStages":["AWSCURRENT"]}]}',
            "",
        ),
    )
    assert ha009.secret_versions("unit-reference")["stages"] == {"AWSCURRENT": "v1"}
    assert calls[0][1:3] == ["secretsmanager", "list-secret-version-ids"]
    monkeypatch.setattr(
        ha009, "run_fixture_command", lambda *a, **kw: SimpleNamespace(stdout="[]")
    )
    with pytest.raises(ha009.CaseError, match="not an object"):
        ha009.aws("rds", "describe-db-clusters")
    monkeypatch.setattr(
        ha009,
        "aurora_guard",
        lambda: SimpleNamespace(
            read=lambda: {
                "identity": {"database": {"master_secret_arn": "unit-reference"}}
            }
        ),
    )
    assert ha009.master_secret_arn() == "unit-reference"
