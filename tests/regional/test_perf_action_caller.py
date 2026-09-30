from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.e2e.regional.regional_live_fixture import component_python
from scripts.perf import regional_action_capacity_suite as action
from scripts.perf import regional_capacity_registry as registry
from tests.regional._perf_caller_support import CapacityWire, action_args


def test_action_main_uses_real_registration_and_receipted_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)

    assert action.main(action_args(tmp_path)) == 0, (
        "the fake complete action run should finish"
    )

    receipt = json.loads((tmp_path / "capacity-resources.json").read_text())
    assert set(receipt["resources"]) == {
        f"configmap/{action.SCRIPT_CONFIGMAP}",
        f"serviceaccount/{action.LOAD_SERVICE_ACCOUNT}",
        f"job/{action.JOB_NAME}",
    }, "every action resource, the load account included, reaches the run journal"
    assert all(value["uid"] for value in receipt["resources"].values()), (
        "action creations must preserve UIDs"
    )
    assert wire.events.index("publish") < wire.events.index(
        f"create:job/{action.JOB_NAME}"
    ), "action Jobs need acknowledged registration and token identity"
    assert wire.events.index(f"delete:job/{action.JOB_NAME}") < wire.events.index(
        "data-cleanup"
    ), "simulated executors must stop before exact data cleanup"
    assert wire.events[-2:] == ["data-cleanup", "teardown"], (
        "action cleanup must precede registry/token teardown"
    )
    assert wire.objects == {}, "the fake action run must not leave resources"
    assert len(wire.teardowns) == 1, (
        "shared teardown must consume the original artifacts once"
    )


@pytest.mark.parametrize("failure", ["missing-sink", "transport"])
def test_action_amp_refusal_cannot_register_or_authorize_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    if failure == "missing-sink":
        wire.amp.config["route"]["routes"] = []
        message = "must not mail drill alerts"
    else:
        wire.amp.error = TimeoutError("AMP preflight unavailable")
        message = "AMP preflight unavailable"

    with pytest.raises((RuntimeError, TimeoutError), match=message) as caught:
        action.main(action_args(tmp_path))

    if wire.amp.error is not None:
        assert caught.value is wire.amp.error, "the original AMP failure must survive"
    assert wire.creations == [], "AMP refusal must precede token or executor creation"
    assert wire.events == [], "AMP refusal must not publish, inspect, or clean run data"
    assert wire.teardowns == [], "AMP refusal cannot invent registry cleanup authority"
    assert not (tmp_path / "registry-registration-intent.json").exists(), (
        "failed AMP preflight must not grant registration ownership"
    )
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "aborted", (
        "failed AMP preflight must invalidate the run"
    )


@pytest.mark.parametrize("resource", [action.SCRIPT_CONFIGMAP, action.JOB_NAME])
def test_action_lost_create_ack_is_tracked_before_finally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resource: str
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    wire.fail_create = resource

    with pytest.raises(TimeoutError, match="create ACK lost"):
        action.main(action_args(tmp_path))

    created = [item for item in wire.creations if item["metadata"]["name"] == resource]
    assert len(created) == 1, "action create must never be blindly retried"
    key = f"{created[0]['kind'].lower()}/{resource}"
    receipt = json.loads((tmp_path / "capacity-resources.json").read_text())
    assert receipt["resources"][key]["uid"] == created[0]["metadata"]["uid"], (
        "lost ACK UID must be recoverable"
    )
    assert wire.objects == {}, (
        "final cleanup must stop the resource created before the lost ACK"
    )
    assert wire.events[-2:] == ["data-cleanup", "teardown"], (
        "lost ACK cannot bypass exact data cleanup"
    )


def test_action_lost_delete_ack_uses_checked_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    wire.fail_delete = f"job/{action.JOB_NAME}"

    assert action.main(action_args(tmp_path)) == 0, (
        "confirmed absence should reconcile a lost delete ACK"
    )
    assert wire.events.count(f"delete:job/{action.JOB_NAME}") == 1, (
        "delete ACK loss must not replay deletion"
    )
    assert wire.events[-2:] == ["data-cleanup", "teardown"], (
        "checked resource absence must allow exact cleanup"
    )


@pytest.mark.parametrize("remaining", [1, False, None])
def test_action_refuses_unknown_or_nonempty_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remaining: object
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)

    def seed(**_kwargs):
        wire.remaining = remaining
        return {"workflows_created": 1}

    monkeypatch.setattr(action, "seed", seed)
    with pytest.raises(RuntimeError, match="exact run-owned cleanup left data"):
        action.main(action_args(tmp_path))

    assert wire.teardowns == [], "failed data proof must keep registry and token intact"
    assert set(wire.objects) == {f"secret/{registry.TOKEN_SECRET}"}, (
        "producer cleanup still precedes data refusal"
    )
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "aborted", (
        "cleanup refusal must fail the run"
    )


def test_action_cleanup_transport_failure_is_not_suppressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    wire.fail_data = True

    with pytest.raises(RuntimeError, match="data cleanup unavailable"):
        action.main(action_args(tmp_path))

    assert wire.teardowns == [], "transport failure cannot revoke ownership proof"
    assert (tmp_path / "registry-registration-intent.json").is_file(), (
        "retry must retain registration intent"
    )
    assert (tmp_path / "registry-token-proof.json").is_file(), (
        "retry must retain the original token UID"
    )


def test_action_registration_ack_failure_keeps_intent_and_token_proof_for_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    wire.fail_publish = True

    with pytest.raises(TimeoutError, match="registration ACK lost"):
        action.main(action_args(tmp_path))

    assert len(wire.creations) == 1, (
        "failed registration must not start executor resources"
    )
    assert wire.events[-2:] == ["data-cleanup", "teardown"], (
        "partial registration must still unwind its exact scope"
    )
    assert wire.objects == {}, (
        "partially published registration must not leak its token"
    )


def test_action_rejects_replaced_registration_token_before_load_or_data_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)

    def publish(entries, **kwargs):
        wire.publish(entries, **kwargs)
        wire.objects[f"secret/{registry.TOKEN_SECRET}"]["metadata"]["uid"] = (
            "replacement"
        )

    monkeypatch.setattr(registry, "write_registry", publish)
    with pytest.raises(RuntimeError, match="token Secret changed"):
        action.main(action_args(tmp_path))

    assert len(wire.creations) == 1, (
        "token UID drift must block all executor resource creation"
    )
    assert "data-cleanup" not in wire.events, (
        "rejected token identity cannot authorize data deletion"
    )
    assert wire.teardowns == [], "a replaced token must never be adopted by teardown"


def test_action_preflight_failure_does_not_invent_cleanup_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)
    wire.remaining = 1

    with pytest.raises(RuntimeError, match="data already exists"):
        action.main(action_args(tmp_path))

    assert wire.creations == [], "nonempty preflight must reject token creation"
    assert wire.teardowns == [], (
        "failed preflight cannot authorize an unregistered cleanup"
    )
    assert "data-cleanup" not in wire.events, (
        "preexisting data must never be purged by a failed registration"
    )


def test_action_replaced_job_blocks_deletion_and_data_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire = CapacityWire(tmp_path, monkeypatch)

    def seed(**_kwargs):
        wire.objects[f"job/{action.JOB_NAME}"]["metadata"]["uid"] = "replacement"
        return {"workflows_created": 1}

    monkeypatch.setattr(action, "seed", seed)
    with pytest.raises(RuntimeError, match="ownership or UID changed"):
        action.main(action_args(tmp_path))

    assert wire.teardowns == [], "replacement detection must block shared teardown"
    assert "data-cleanup" not in wire.events, (
        "data cleanup requires the original producer to have stopped"
    )
    assert not any(event.startswith("delete:") for event in wire.events), (
        "a replaced Job must never be deleted"
    )


@pytest.mark.parametrize("probe", ["seed", "database_snapshot", "database_details"])
def test_action_database_transport_uses_cpu_component_python(
    monkeypatch: pytest.MonkeyPatch, probe: str
) -> None:
    calls = []

    def control(*args, **_kwargs):
        calls.append(args)
        return "api" if args[0] == "get" else "{}"

    monkeypatch.setattr(action, "control", control)
    if probe == "seed":
        action.seed(
            run_id="run-a",
            clusters=1,
            workflows_per_cluster=1,
            nodes_per_workflow=1,
            agent_identity=None,
        )
    else:
        getattr(action, probe)("run-a")

    assert component_python("cpu") in calls[-1], (
        "database probes must run in the installed CPU component"
    )
    assert "python3" not in calls[-1], (
        "system Python cannot stand in for the CPU runtime"
    )


def test_budget_tier_relaxes_the_executor_floor_under_the_production_budget() -> None:
    """§8.2 runs the same model under the production budget and the raw
    capacity tier; the run records the tier and only the raw tier demands full
    executor concurrency (live 2026-09-22: 20/5/2 failed the floor of 8)."""

    raw = {
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION": 128,
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER": 4,
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS": 4,
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_NODE": 1,
        "GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_FAILURE_DOMAIN": 1,
    }
    production = {**raw, "GPU_FAULT_REMEDIATION_MAX_ACTIVE_REGION": 20}
    production["GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_CLUSTER"] = 5
    production["GPU_FAULT_REMEDIATION_MAX_ACTIVE_PER_RESOURCE_CLASS"] = 2
    unbounded = {name: None for name in raw}
    model = {"clusters": 32, "workflows_per_cluster": 4, "executor_workers": 8}

    assert action.budget_tier(raw, **model)["tier"] == "raw-capacity", (
        "128/4/4 admits every one of 32 x 4 workflows at once"
    )
    assert action.budget_tier(raw, **model)["min_concurrent_commands"] == 1, (
        "peak concurrency is recorded evidence, not a pass criterion, in any tier"
    )
    tier = action.budget_tier(production, **model)
    assert tier["tier"] == "production-budget", "20/5/2 throttles the model"
    assert tier["min_concurrent_commands"] == 1, (
        "a throttled executor only has to make progress"
    )
    assert tier["required_for_raw_capacity"] == {
        "region": 128,
        "per_cluster": 4,
        "per_resource_class": 4,
    }, "the record names what the raw tier would need"
    assert action.budget_tier(unbounded, **model)["tier"] == "unbounded", (
        "no limits configured is its own tier"
    )
    job = action.executor_job(
        clusters=1,
        expected_commands=40,
        workers=8,
        min_concurrent_commands=1,
        delay_scale=1.0,
        lease_seconds=60,
        inject_renew_failure_once=False,
        executor_protocol_version=1,
        executor_artifact_sha256="a" * 64,
        executor_compatibility_digest="b" * 64,
    )
    env = {
        item["name"]: item["value"]
        for item in job["spec"]["template"]["spec"]["containers"][0]["env"]
        if "value" in item
    }
    assert env["ACTION_MIN_CONCURRENT_COMMANDS"] == "1", (
        "the floor reaches the executor through its environment"
    )
    long_job = action.executor_job(
        clusters=1,
        expected_commands=40,
        workers=8,
        max_seconds=2400,
        delay_scale=1.0,
        lease_seconds=60,
        inject_renew_failure_once=False,
        executor_protocol_version=1,
        executor_artifact_sha256="a" * 64,
        executor_compatibility_digest="b" * 64,
    )
    long_env = {
        item["name"]: item["value"]
        for item in long_job["spec"]["template"]["spec"]["containers"][0]["env"]
        if "value" in item
    }
    assert long_env["ACTION_MAX_SECONDS"] == "2400", (
        "the executors' wall follows the runner's --timeout-seconds"
    )
    assert long_job["spec"]["activeDeadlineSeconds"] == 2700, (
        "the Job deadline leaves teardown slack past the executors' wall"
    )
