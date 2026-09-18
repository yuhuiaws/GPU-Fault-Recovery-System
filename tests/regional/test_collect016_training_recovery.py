"""COLLECT-016: resources are owned from the moment they exist, and the A/B
segments assert what is unique to them rather than only DESTR-009's contract."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from gpu_fault.models import Environment
from gpu_fault.watcher import AttemptObservation, ContainerObservation, WorkloadPhase
from scripts.e2e.regional import run_collect016_training_recovery as collect016
from tests.regional._collector_reset_support import reset_documents


def _state_a(job_id: str = "c016-a-1") -> dict[str, Any]:
    return {
        "incident": {"workload_identity_source": "SOLE_ACTIVE_ATTEMPT_ON_NODE"},
        "workflow": {
            "status": "SUCCEEDED",
            "blocked_reasons": [],
            "official_steps": [
                {"operation": "STOP_WORKLOADS", "parameters": {"job_id": job_id}},
                {"operation": "RESTART_WORKLOAD", "parameters": {"job_id": job_id}},
            ],
        },
    }


def _state_b() -> dict[str, Any]:
    return {
        "workflow": {
            "status": "FAILED",
            "step_executions": [
                {
                    "operation": "RESTART_WORKLOAD",
                    "status": "FAILED",
                    "details": {"reason": "RESTART_BUDGET_EXHAUSTED"},
                }
            ],
        },
        "commands": [],
        "restart_budget": {"restart_count": 1, "budget": 1},
    }


@pytest.mark.parametrize(
    "commands",
    [[], [{"command_id": "stop-1", "step": {"operation": "STOP_WORKLOADS"}}]],
    ids=["no-remote-command", "withheld-restart-keeps-stop"],
)
def test_restart_budget_sections_pass_on_the_documented_readings(
    commands: list[dict[str, Any]],
) -> None:
    state_b = _state_b()
    state_b["commands"] = commands
    assert (
        collect016.restart_budget_section_errors(_state_a(), state_b, job_id="c016-a-1")
        == []
    ), "withholding the exhausted restart does not prohibit STOP_WORKLOADS"


@pytest.mark.parametrize(
    ("mutate_a", "mutate_b", "fragment"),
    [
        (
            lambda a: a["incident"].__setitem__("workload_identity_source", None),
            None,
            "workload_identity_source",
        ),
        (
            lambda a: a["workflow"].__setitem__("blocked_reasons", ["no workload"]),
            None,
            "was blocked",
        ),
        (
            lambda a: a["workflow"]["official_steps"][1]["parameters"].__setitem__(
                "job_id", "other-job"
            ),
            None,
            "compiled for job",
        ),
        (
            None,
            lambda b: b["workflow"].__setitem__("status", "SUCCEEDED"),
            "not budget",
        ),
        (None, lambda b: b.__setitem__("commands", [{"command_id": "x"}]), "remote"),
        (
            None,
            lambda b: b.__setitem__(
                "commands",
                [{"command_id": "x", "step": {"operation": "RESTART_WORKLOAD"}}],
            ),
            "dispatched a RESTART_WORKLOAD",
        ),
        (
            None,
            lambda b: b["workflow"]["step_executions"][0]["details"].__setitem__(
                "reason", "OTHER"
            ),
            "RESTART_BUDGET_EXHAUSTED",
        ),
        (
            None,
            lambda b: b["restart_budget"].__setitem__("restart_count", 2),
            "restart_count",
        ),
    ],
)
def test_each_unique_reading_is_checked(mutate_a, mutate_b, fragment: str) -> None:
    state_a, state_b = _state_a(), _state_b()
    if mutate_a is not None:
        mutate_a(state_a)
    if mutate_b is not None:
        mutate_b(state_b)
    errors = collect016.restart_budget_section_errors(
        state_a, state_b, job_id="c016-a-1"
    )
    assert any(fragment in item for item in errors), (fragment, errors)


def _settings(tmp_path: Path) -> Any:
    class Regional:
        gpu_kubeconfig = "kc"
        gpu_context = "ctx"
        namespace = "ns"
        cluster_id = "cluster-a"

    return collect016.Settings(
        regional=Regional(),  # type: ignore[arg-type]
        site_file=tmp_path / "site.yaml",
        host_probe_image="img",
        predecessor_path=tmp_path / "pred.json",
    )


def test_workload_is_owned_before_its_first_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wait_running timeout used to leak the 24-GPU job: it was only appended
    to the cleanup list after the helper returned."""

    class Workload:
        def submit(self) -> None:
            pass

        def wait_running(self, timeout_seconds: int) -> dict[str, Any]:
            raise TimeoutError("pods never ran")

    workload = Workload()
    monkeypatch.setattr(
        collect016.base, "render_named_training_manifest", lambda p, **_: p
    )
    monkeypatch.setattr(collect016, "managed_fixture", lambda *_, **__: workload)
    workloads: list[Any] = []
    fixtures: list[Any] = []

    with pytest.raises(TimeoutError):
        collect016.run_restart_budget_sections(
            _settings(tmp_path),
            object(),  # type: ignore[arg-type]
            tmp_path,
            "s1",
            workloads=workloads,
            fixtures=fixtures,
        )
    assert workloads == [workload], "the job is registered before wait_running"


def test_probe_is_owned_before_it_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Workload:
        def submit(self) -> None:
            pass

        def wait_running(self, timeout_seconds: int) -> dict[str, Any]:
            return {"pods": [{"uid": "u1", "node": "node-a"}]}

    class Collector:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def create(self) -> None:
            raise RuntimeError("image pull failed")

    monkeypatch.setattr(
        collect016.base, "render_named_training_manifest", lambda p, **_: p
    )
    monkeypatch.setattr(collect016, "managed_fixture", lambda *_, **__: Workload())
    monkeypatch.setattr(
        collect016.workload_case, "wait_observation", lambda *_, **__: None
    )
    monkeypatch.setattr(collect016, "CollectorAcceptanceFixture", Collector)
    workloads: list[Any] = []
    fixtures: list[Any] = []

    with pytest.raises(RuntimeError, match="image pull failed"):
        collect016.run_restart_budget_sections(
            _settings(tmp_path),
            object(),  # type: ignore[arg-type]
            tmp_path,
            "s1",
            workloads=workloads,
            fixtures=fixtures,
        )
    assert len(workloads) == 1
    assert len(fixtures) == 1 and isinstance(fixtures[0], Collector), (
        "the probe is registered before create() can fail"
    )


def test_reset_section_owns_workload_and_probe_before_the_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Workload:
        def submit(self) -> None:
            pass

        def wait_running(self, timeout_seconds: int) -> dict[str, Any]:
            return {"pods": [{"uid": "u1", "node": "node-a"}]}

    class Host:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def create(self) -> None:
            raise RuntimeError("no such node")

    monkeypatch.setattr(
        collect016.base, "render_named_training_manifest", lambda p, **_: p
    )
    monkeypatch.setattr(collect016, "managed_fixture", lambda *_, **__: Workload())
    monkeypatch.setattr(
        collect016.workload_case, "wait_observation", lambda *_, **__: None
    )
    monkeypatch.setattr(collect016, "HostProbeFixture", Host)
    monkeypatch.setattr(
        collect016, "HostProbeSettings", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    workloads: list[Any] = []
    fixtures: list[Any] = []

    with pytest.raises(RuntimeError, match="no such node"):
        collect016.run_reset_section(
            _settings(tmp_path),
            object(),  # type: ignore[arg-type]
            tmp_path,
            "s1",
            workloads=workloads,
            fixtures=fixtures,
        )
    assert len(workloads) == 1 and len(fixtures) == 1


@pytest.fixture
def active_reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    job_id = "c016-d-s1"
    old_attempt, new_attempt = f"{job_id}-a001", f"{job_id}-a002"
    uids = [f"{index:08x}-0000-4000-8000-{index:012x}" for index in range(1, 7)]
    old_uids, new_uids = uids[:3], uids[3:]
    nodes = ["node-a", "node-b", "node-c"]
    pods = [
        {
            "uid": uid,
            "name": f"worker-{index}",
            "node": node,
            "phase": "Running",
            "ready": True,
            "attempt_id": new_attempt,
        }
        for index, (uid, node) in enumerate(zip(new_uids, nodes, strict=True))
    ]
    old_pods = [
        {**pod, "uid": uid, "attempt_id": old_attempt}
        for pod, uid in zip(pods, old_uids, strict=True)
    ]
    before, after, state = reset_documents()
    inventory = [
        {"uuid": f"GPU-{suffix}", "pci_bdf": f"0000:{0x59 + index:02x}:00.0"}
        for index, suffix in enumerate("abcdefgh")
    ]
    gpu_uuids = [item["uuid"] for item in inventory]
    before["gpu_inventory"] = inventory
    after.update(
        gpu_inventory=deepcopy(inventory),
        compute_clients=[
            {"pid": str(100 + index), "pod_uid": new_uids[0], "gpu_uuid": uuid}
            for index, uuid in enumerate(gpu_uuids)
        ],
        sampler={
            "sample_count": 3,
            "min_gpu_count": 7,
            "last": {"gpu_count": 8, "gpu_uuids": gpu_uuids},
            "observed_gpu_uuid_sets": [gpu_uuids, gpu_uuids[1:], gpu_uuids],
        },
    )
    observation = AttemptObservation(
        cluster_id="cluster-a",
        environment=Environment.KUBERNETES,
        job_id=job_id,
        attempt_id=new_attempt,
        workload_phase=WorkloadPhase.RUNNING,
        observed_at=datetime.now(timezone.utc),
        expected_critical_ranks=3,
        runtime_profile_version="profile-a",
        containers=[
            ContainerObservation(
                pod_uid=pod["uid"],
                pod_name=pod["name"],
                container_name="trainer",
                role="worker",
                rank=index,
                node_id=pod["node"],
                gpu_count=8,
                gpu_uuids=(
                    gpu_uuids
                    if index == 0
                    else [f"GPU-{index}-{number}" for number in range(8)]
                ),
            )
            for index, pod in enumerate(pods)
        ],
    ).model_dump(mode="json")
    old_observation = deepcopy(observation)
    old_observation["attempt_id"] = old_attempt
    for container, uid in zip(old_observation["containers"], old_uids, strict=True):
        container["pod_uid"] = uid
    state["event"].update(xid=109, evidence_ref="kmsg://node-a/boot-1/1")
    state["decision"] = {"official_action": "RESET_GPU"}
    reset_step = state["workflow"]["official_steps"][0]
    state["workflow"].update(
        official_steps=[
            reset_step if operation == "RESET_GPU" else {"operation": operation}
            for operation in collect016.WORKLOAD_RESET_STEPS
        ],
        completed_operations=list(collect016.WORKLOAD_RESET_STEPS),
        step_executions=[
            {
                "operation": operation,
                "status": "SUCCEEDED",
                "adapter_operation_id": f"remote/{operation}",
            }
            for operation in collect016.WORKLOAD_RESET_STEPS
        ],
    )
    state["commands"] = [
        {"status": "SUCCEEDED", "step": {"operation": operation}}
        for operation in collect016.WORKLOAD_RESET_STEPS
    ]
    calls: list[str] = []
    workload = SimpleNamespace(
        submit=lambda: calls.append("submit"),
        wait_running=lambda **_: {"pods": deepcopy(old_pods)},
    )

    def authorize_restart(proof: dict[str, Any]) -> None:
        assert proof == state
        calls.append("authorize-restart")

    workload.authorize_restart = authorize_restart

    def wait_restarted(
        source_uids: set[str], *, timeout_seconds: int
    ) -> dict[str, Any]:
        assert source_uids == set(old_uids)
        assert timeout_seconds == 900
        calls.append("replacement-pods")
        return {"pods": deepcopy(pods)}

    workload.wait_restarted = wait_restarted

    def store_snapshot(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["node"] == "node-a" and kwargs["job_id"] == job_id
        attempt = kwargs["attempt_id"]
        assert attempt in {old_attempt, new_attempt}
        calls.append(f"observation:{attempt}")
        value = old_observation if attempt == old_attempt else observation
        return {"observations": [deepcopy(value)]}

    regional = SimpleNamespace(store_snapshot=store_snapshot)
    output = tmp_path / "d-reset"
    persisted_at_stop: list[str] = []

    def host_execute(command: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        if command == "snapshot":
            final = "--since-epoch" in args
            calls.append("host-after" if final else "host-before")
            return deepcopy(after if final else before)
        calls.append(command)
        if command == "stop-reset-sampler":
            persisted_at_stop.extend(path.name for path in output.glob("*.json"))
        return {}

    host = SimpleNamespace(
        host_script="/owned-unit-probe.py",
        create=lambda: calls.append("host-create"),
        execute=host_execute,
    )
    audit_reads = 0

    def audit_execute(command: str) -> dict[str, Any]:
        nonlocal audit_reads
        assert command == "reset-audit"
        audit_reads += 1
        calls.append(f"audit:{audit_reads}")
        return deepcopy(before if audit_reads == 1 else after)

    collector = SimpleNamespace(
        create=lambda: calls.append("collector-create"), execute=audit_execute
    )

    def wait_workflow(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("workflow-terminal")
        return deepcopy(state)

    monkeypatch.setattr(
        collect016.base, "render_named_training_manifest", lambda path, **_: path
    )
    monkeypatch.setattr(collect016, "managed_fixture", lambda *_, **__: workload)
    monkeypatch.setattr(collect016, "HostProbeFixture", lambda _: host)
    monkeypatch.setattr(
        collect016, "HostProbeSettings", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(
        collect016, "CollectorAcceptanceFixture", lambda *_, **__: collector
    )
    monkeypatch.setattr(collect016.base, "wait_xid_workflow", wait_workflow)
    return SimpleNamespace(
        settings=_settings(tmp_path),
        regional=regional,
        workload=workload,
        host=host,
        collector=collector,
        state=state,
        after=after,
        pods=pods,
        observation=observation,
        old_uids=old_uids,
        old_attempt=old_attempt,
        new_attempt=new_attempt,
        calls=calls,
        persisted_at_stop=persisted_at_stop,
        case_dir=tmp_path,
        output=output,
    )


def run_active_reset(environment: SimpleNamespace) -> dict[str, Any]:
    return collect016.run_reset_section(
        environment.settings,
        environment.regional,
        environment.case_dir,
        "s1",
        workloads=[],
        fixtures=[],
    )


def test_active_reset_observes_replacement_before_final_host_snapshot(
    active_reset: SimpleNamespace,
) -> None:
    result = run_active_reset(active_reset)
    assert result["errors"] == [], result["errors"]
    proof = json.loads((active_reset.output / "post-restart-workload.json").read_text())
    assert proof == {
        "cluster_id": "cluster-a",
        "job_id": "c016-d-s1",
        "node": "node-a",
        "source_attempt_id": active_reset.old_attempt,
        "source_pod_uids": active_reset.old_uids,
        "pods": active_reset.pods,
        "observation": active_reset.observation,
    }
    assert result["d"]["target_pod_uids"] == sorted(
        pod["uid"] for pod in active_reset.pods
    )
    ordered = [
        "workflow-terminal",
        "authorize-restart",
        "replacement-pods",
        f"observation:{active_reset.new_attempt}",
        "host-after",
        "audit:2",
        "stop-reset-sampler",
    ]
    indices = [active_reset.calls.index(call) for call in ordered]
    assert indices == sorted(indices)
    assert {
        "post-restart-workload.json",
        "host-after.json",
        "reset-audit-after.json",
    } <= set(active_reset.persisted_at_stop)


@pytest.mark.parametrize(
    "change",
    ["old-uid", "not-ready", "mixed-attempt", "no-attempt", "missing", "no-node"],
)
def test_active_reset_refuses_unproven_replacement_before_final_snapshot(
    active_reset: SimpleNamespace, change: str
) -> None:
    pod = active_reset.pods[0]
    if change == "old-uid":
        pod["uid"] = active_reset.old_uids[0]
    elif change == "not-ready":
        pod["ready"] = False
    elif change == "mixed-attempt":
        pod["attempt_id"] = active_reset.old_attempt
    elif change == "no-attempt":
        for item in active_reset.pods:
            item["attempt_id"] = None
    elif change == "missing":
        active_reset.pods.pop()
    else:
        pod["node"] = None
    with pytest.raises(collect016.RegionalFixtureError, match="replacement"):
        run_active_reset(active_reset)
    assert "host-after" not in active_reset.calls
    assert active_reset.calls[-1] == "stop-reset-sampler"


@pytest.mark.parametrize(
    "change", ["stale", "future", "naive", "invalid", "cluster", "job", "attempt"]
)
def test_active_reset_requires_fresh_bound_running_observation(
    active_reset: SimpleNamespace, change: str
) -> None:
    observation = active_reset.observation
    if change in {"stale", "future"}:
        observation["observed_at"] = (
            datetime.now(timezone.utc)
            + timedelta(seconds=-121 if change == "stale" else 60)
        ).isoformat()
    elif change == "naive":
        observation["observed_at"] = datetime.now().isoformat()
    elif change == "invalid":
        observation["observed_at"] = "unknown"
    else:
        observation[f"{change}_id"] = "foreign"
    with pytest.raises(collect016.RegionalFixtureError, match="observation"):
        run_active_reset(active_reset)
    assert "host-after" not in active_reset.calls
    assert active_reset.calls[-1] == "stop-reset-sampler"


@pytest.mark.parametrize(
    "change",
    [
        "old-client",
        "unknown-client",
        "foreign-gpu",
        "missing-client",
        "allocation",
        "deleted-pod",
        "stale-at-snapshot",
        "wrong-reset-gpu",
    ],
)
def test_active_caller_keeps_client_allocation_and_physical_reset_guards(
    active_reset: SimpleNamespace, change: str
) -> None:
    after = active_reset.after
    if change == "old-client":
        after["compute_clients"][0]["pod_uid"] = active_reset.old_uids[0]
    elif change == "unknown-client":
        after["compute_clients"][0]["pod_uid"] = None
    elif change == "foreign-gpu":
        after["compute_clients"][0]["gpu_uuid"] = "GPU-foreign"
    elif change == "missing-client":
        after["compute_clients"].pop()
    elif change == "allocation":
        active_reset.observation["containers"][0]["gpu_uuids"][0] = "GPU-foreign"
    elif change == "deleted-pod":
        active_reset.observation["containers"][0]["deletion_requested"] = True
    elif change == "stale-at-snapshot":
        after["captured_at"] = (
            datetime.now(timezone.utc) + timedelta(seconds=121)
        ).isoformat()
    else:
        after["sampler"]["observed_gpu_uuid_sets"] = [
            after["sampler"]["last"]["gpu_uuids"][:-1]
        ]
    errors = run_active_reset(active_reset)["errors"]
    expected = "only the target" if change == "wrong-reset-gpu" else "post-restart"
    assert any(expected in error for error in errors), errors
    assert active_reset.calls[-1] == "stop-reset-sampler"


def test_replacement_wait_failure_still_stops_the_sampler(
    active_reset: SimpleNamespace,
) -> None:
    active_reset.workload.wait_restarted = Mock(
        side_effect=TimeoutError("replacement unavailable")
    )
    with pytest.raises(TimeoutError, match="replacement unavailable"):
        run_active_reset(active_reset)
    assert "host-after" not in active_reset.calls
    assert active_reset.calls[-1] == "stop-reset-sampler"


def test_restart_authorization_failure_stops_before_replacement_and_stops_sampler(
    active_reset: SimpleNamespace,
) -> None:
    active_reset.workload.authorize_restart = Mock(
        side_effect=collect016.RegionalFixtureError("restart custody rejected")
    )
    with pytest.raises(collect016.RegionalFixtureError, match="custody rejected"):
        run_active_reset(active_reset)
    active_reset.workload.authorize_restart.assert_called_once_with(active_reset.state)
    assert "replacement-pods" not in active_reset.calls
    assert "host-after" not in active_reset.calls
    assert active_reset.calls[-1] == "stop-reset-sampler"


def test_shared_reset_without_workload_proof_keeps_idle_client_rule(
    active_reset: SimpleNamespace,
) -> None:
    state, errors = collect016.base.run_single_reset(
        SimpleNamespace(node="node-a", case_id=collect016.CASE_ID),
        active_reset.regional,
        active_reset.host,
        active_reset.output,
        collector=active_reset.collector,
        cleanup=collect016.base.CaseCleanup(),
        xid=109,
        marker="unit-reset",
        run_id="unit-reset",
        expected_steps=collect016.WORKLOAD_RESET_STEPS,
    )
    assert "post_restart_workload" not in state
    assert "compute clients remain after reset" in errors
