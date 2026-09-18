"""Adversarial composed safety checks with real cancellation and Store causality."""

from __future__ import annotations

import copy
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from gpu_fault.admin.atomic_json import write_json_atomic
from scripts.e2e.regional import destr008_safety as safety_module
from scripts.e2e.regional import destr008_watchdog_resources as resources
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._destr008_causal import CausalHarness, build_causal


@pytest.fixture
def h(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CausalHarness:
    return build_causal(tmp_path, monkeypatch)


def at(h: CausalHarness, seconds: int) -> datetime:
    return datetime.fromtimestamp(h.cpu.clock.now() + seconds, timezone.utc)


def arm(h: CausalHarness, *, window: int = 360, maintenance: int = 3600) -> None:
    h.safety.arm(
        h.observation(),
        window_seconds=window,
        maintenance_window_end=at(h, maintenance),
    )


def fresh(h: CausalHarness) -> safety_module.ShortageSafety:
    current = h.safety
    return safety_module.ShortageSafety(
        h.cpu.regional,
        run_id=current.run_id,
        attempt_id=current.attempt_id,
        event_id=current.event_id,
        fault_node=current.fault_node,
        spare_node=current.spare_node,
        spare_uid=h.gpu.binding.node_uid,
        release_id=current.release_id,
        directory=current.directory,
    )


def cpu_mutations(h: CausalHarness) -> list[tuple[str, tuple[str, ...]]]:
    return [
        (verb, args)
        for verb, args, _ in h.cpu.calls
        if verb in {"create", "patch", "delete"}
    ]


def gpu_mutations(h: CausalHarness) -> list[tuple[str, ...]]:
    return [
        args
        for kind, args in h.gpu.calls
        if kind == "kube" and args[0] in {"create", "patch", "delete"}
    ]


@pytest.mark.parametrize("window,maintenance", [(119, 3600), (360, 179), (0, 3600)])
def test_insufficient_initial_headroom_never_creates_protection(
    h: CausalHarness, window: int, maintenance: int
) -> None:
    with pytest.raises(RegionalFixtureError, match="insufficient maintenance window"):
        arm(h, window=window, maintenance=maintenance)
    assert h.safety.plan is None and h.safety.watchdog is None
    assert h.cpu.calls == [] and h.gpu.calls == []
    report = h.safety.resume_cleanup()
    assert report == {"quiescent": True, "retired": True, "errors": []}


@pytest.mark.parametrize(
    "window,maintenance,expected", [(120, 3600, 120), (360, 240, 180)]
)
def test_deadline_is_clamped_once_and_boundary_headroom_is_usable(
    h: CausalHarness, window: int, maintenance: int, expected: int
) -> None:
    start = int(h.cpu.clock.now())
    arm(h, window=window, maintenance=maintenance)
    assert h.safety.plan is not None
    assert h.safety.plan.created_at == start
    assert h.safety.plan.deadline_at == start + expected
    h.safety.admit_fixture()
    report = h.safety.finish()
    assert report["quiescent"] and report["retired"] and report["errors"] == [], report
    assert report["receipt"]["source_complete"] is True
    assert h.cpu.journal()["plan"]["deadline_at"] == start + expected


def test_watchdog_startup_consuming_headroom_cannot_admit_fixture(
    h: CausalHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_pod = h.cpu.run_pod

    def delayed_start(pod: dict[str, Any]) -> None:
        run_pod(pod)
        h.cpu.clock.sleep(241)

    monkeypatch.setattr(h.cpu, "run_pod", delayed_start)
    with pytest.raises(RegionalFixtureError, match="startup consumed"):
        arm(h)
    assert h.safety.watchdog is not None and h.safety.control is not None
    assert h.safety.watchdog.record.ever_armed is True
    assert h.safety.fence.record["action_started"] is False
    with pytest.raises(RegionalFixtureError, match="bound complete watchdog"):
        h.safety.admit_fixture()
    report = h.safety.finish()
    assert report["quiescent"] and report["retired"], report
    assert report["receipt"]["producer"]["state"] == "NOT_STARTED"


@pytest.mark.parametrize(
    "bad_observation",
    [
        {"workload_ids": ["training/job-a"]},
        {"runtime_profile_version": "profile-a", "workload_ids": []},
    ],
)
def test_invalid_observation_after_fence_arm_has_no_cpu_producer_and_is_cleanup_only(
    h: CausalHarness, bad_observation: dict[str, Any]
) -> None:
    with pytest.raises((KeyError, ValidationError)):
        h.safety.arm(
            bad_observation, window_seconds=360, maintenance_window_end=at(h, 3600)
        )
    assert h.safety.watchdog is None and h.safety.control is None
    assert h.gpu.objects, "the actual fence arm must precede this interrupted phase"
    assert cpu_mutations(h) == []
    report = h.safety.finish(cleanup_only=True)
    assert report == {"quiescent": True, "retired": True, "errors": []}
    assert h.gpu.objects == {}


@pytest.mark.parametrize(
    "kind", ["validatingadmissionpolicy", "validatingadmissionpolicybinding"]
)
def test_lost_fence_create_ack_remains_unresolved_and_never_adopts(
    h: CausalHarness, kind: str
) -> None:
    h.gpu.lost_create.add(kind)
    with pytest.raises(TimeoutError):
        arm(h)
    assert h.safety.watchdog is None and cpu_mutations(h) == []
    before = copy.deepcopy(h.gpu.objects)
    report = h.safety.resume_cleanup()
    assert report["quiescent"] is False and report["retired"] is False, report
    assert report["errors"] and h.safety.quiescent is False, report
    assert h.gpu.objects == before


@pytest.mark.parametrize(
    "kind", ["configmap", "serviceaccount", "role", "rolebinding", "job"]
)
def test_lost_cpu_creation_ack_preserves_fence_and_forbids_cleanup_adoption(
    h: CausalHarness, kind: str
) -> None:
    h.cpu.lost_create.add(kind)
    with pytest.raises(RegionalFixtureError):
        arm(h)
    assert h.safety.watchdog is not None and h.safety.control is None
    assert h.safety.watchdog.record.ever_armed is False
    report = fresh(h).resume_cleanup()
    assert report["quiescent"] is False and report["retired"] is False, report
    assert report["errors"] and h.gpu.objects, report
    assert not any(verb == "delete" for verb, _ in cpu_mutations(h)), h.cpu.calls


@pytest.mark.parametrize("cleanup_only", [False, True])
def test_known_gated_pod_failure_can_retire_without_claim_or_new_observer(
    h: CausalHarness, cleanup_only: bool
) -> None:
    h.cpu.gate_no_apply = True
    with pytest.raises(RegionalFixtureError, match="gate release is unconfirmed"):
        arm(h)
    assert h.safety.watchdog is not None
    assert h.safety.watchdog.record.ever_armed is False
    creations = sum(verb == "create" for verb, _, _ in h.cpu.calls)
    report = fresh(h).resume_cleanup() if cleanup_only else h.safety.finish()
    assert report == {"quiescent": True, "retired": True, "errors": []}
    assert sum(verb == "create" for verb, _, _ in h.cpu.calls) == creations
    assert h.gpu.objects == {} and h.cpu.journal()["closed"] is True


def test_lost_in_memory_watchdog_handle_does_not_leave_control_as_authority(
    h: CausalHarness,
) -> None:
    arm(h)
    h.safety.admit_fixture()
    h.safety.watchdog = None
    report = h.safety.finish()
    assert report["quiescent"] is False and report["retired"] is False, report
    assert report["errors"] == ["quiescence: RegionalFixtureError"]
    assert h.safety.quiescent is False and h.gpu.objects, report
    assert not any(verb == "delete" for verb, _ in cpu_mutations(h)), h.cpu.calls


@pytest.mark.parametrize(
    "kind", ["validatingadmissionpolicybinding", "validatingadmissionpolicy"]
)
def test_fence_deletion_failure_does_not_reuse_quiescence_from_previous_attempt(
    h: CausalHarness, kind: str
) -> None:
    arm(h)
    h.safety.admit_fixture()
    h.gpu.failed_delete.add(kind)
    report = h.safety.finish()
    assert report["quiescent"] is False and report["retired"] is False, report
    assert report["errors"] and h.safety.quiescent is False, report
    assert kind in h.gpu.objects
    assert h.cpu.journal()["closed"] is False
    h.gpu.failed_delete.clear()
    retried = h.safety.resume_cleanup()
    assert retried["quiescent"] and retried["retired"], retried
    assert retried["receipt"]["case_failed"] is True
    assert h.gpu.objects == {} and h.cpu.journal()["closed"] is True


def test_cpu_retirement_failure_is_not_cleanup_complete_and_can_resume(
    h: CausalHarness,
) -> None:
    arm(h)
    h.safety.admit_fixture()
    h.cpu.failed_delete.add("role")
    report = h.safety.finish()
    assert report["quiescent"] is True and report["retired"] is False, report
    assert report["errors"] == ["watchdog cleanup: RegionalFixtureError"]
    assert h.gpu.objects == {} and h.cpu.journal()["closed"] is False
    h.cpu.failed_delete.clear()
    retried = h.safety.finish()
    assert retried["quiescent"] and retried["retired"], retried
    assert retried["errors"], (
        "the failed original completion cannot become a clean case"
    )
    assert h.cpu.journal()["closed"] is True


@pytest.mark.parametrize(
    "field", ["attempt_id", "event_id", "fault_node", "release_id"]
)
def test_changed_saved_plan_is_rejected_without_any_cleanup_mutation(
    h: CausalHarness, field: str
) -> None:
    arm(h)
    assert h.safety.watchdog is not None
    path = h.safety.watchdog.path
    data = h.cpu.journal()
    data["plan"][field] = "changed-" + field
    write_json_atomic(path, data)
    before = cpu_mutations(h), gpu_mutations(h)
    with pytest.raises(
        RegionalFixtureError, match="watchdog private journal is invalid or unavailable"
    ):
        h.safety.resume_cleanup()
    assert (cpu_mutations(h), gpu_mutations(h)) == before
    assert h.safety.quiescent is False and h.gpu.objects, h.safety.quiescent


@pytest.mark.parametrize("plane", ["cpu", "gpu"])
def test_connection_drift_blocks_production_and_cleanup_without_commands(
    h: CausalHarness, plane: str
) -> None:
    arm(h)
    config = getattr(h.cpu.regional.settings, plane + "_kubeconfig")
    config.write_text("changed controlled configuration\n")
    before = cpu_mutations(h), gpu_mutations(h)
    with pytest.raises(RegionalFixtureError):
        h.safety.before_post()
    report = h.safety.finish()
    assert report["quiescent"] is False and report["retired"] is False, report
    assert h.safety.quiescent is False and h.gpu.objects, report
    assert (cpu_mutations(h), gpu_mutations(h)) == before


def test_source_drift_uses_real_digest_validation_not_fabricated_receipts(
    h: CausalHarness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "owned-probe-source"
    source.mkdir()
    for name in wire.SOURCE_FILES:
        shutil.copyfile(h.cpu.source / name, source / name)
    h.cpu.source = source
    monkeypatch.setattr(resources, "CODE_SOURCE", source)
    arm(h)
    file = source / wire.SOURCE_FILES[0]
    file.write_bytes(file.read_bytes() + b"\n# controlled source identity drift\n")
    before = cpu_mutations(h), gpu_mutations(h)
    with pytest.raises(RegionalFixtureError, match="source changed"):
        h.safety.admit_fixture()
    report = h.safety.finish()
    assert report["quiescent"] is False and report["retired"] is False, report
    assert h.gpu.objects and h.cpu.journal()["closed"] is False, report
    assert (cpu_mutations(h), gpu_mutations(h)) == before


@pytest.mark.parametrize("field", ["attempt_id", "event_id"])
def test_resume_validation_failure_clears_previous_quiescence(
    h: CausalHarness, field: str
) -> None:
    arm(h)
    h.safety.admit_fixture()
    assert h.safety.finish()["retired"] is True
    assert h.safety.quiescent is True
    assert h.safety.watchdog is not None
    data = h.cpu.journal()
    data["plan"][field] = "changed-" + field
    write_json_atomic(h.safety.watchdog.path, data)
    with pytest.raises(
        RegionalFixtureError, match="watchdog private journal is invalid or unavailable"
    ):
        h.safety.resume_cleanup()
    assert h.safety.quiescent is False, (
        "failed fresh validation cannot retain an earlier success"
    )


def test_negative_margin_cannot_admit_expiry_before_cancellation(
    h: CausalHarness,
) -> None:
    arm(h)
    assert h.safety.plan is not None
    expiry = datetime.fromtimestamp(h.safety.plan.deadline_at - 1, timezone.utc)
    before = list(h.cpu.calls), list(h.gpu.calls)
    with pytest.raises(RegionalFixtureError):
        h.safety.require_bound(expiry, margin=-1)
    assert (h.cpu.calls, h.gpu.calls) == before


@pytest.mark.parametrize("margin", [True, False])
def test_boolean_margin_is_not_an_integer_safety_budget(
    h: CausalHarness, margin: bool
) -> None:
    arm(h)
    assert h.safety.plan is not None
    expiry = datetime.fromtimestamp(h.safety.plan.deadline_at + 120, timezone.utc)
    before = list(h.cpu.calls), list(h.gpu.calls)
    with pytest.raises(RegionalFixtureError, match="at least 60 seconds margin"):
        h.safety.require_bound(expiry, margin=margin)
    assert (h.cpu.calls, h.gpu.calls) == before


@pytest.mark.parametrize("margin", [0, 59])
def test_margin_below_sixty_is_rejected_without_io(
    h: CausalHarness, margin: int
) -> None:
    arm(h)
    assert h.safety.plan is not None
    expiry = datetime.fromtimestamp(h.safety.plan.deadline_at + 120, timezone.utc)
    before = list(h.cpu.calls), list(h.gpu.calls)
    with pytest.raises(RegionalFixtureError, match="at least 60 seconds margin"):
        h.safety.require_bound(expiry, margin=margin)
    assert (h.cpu.calls, h.gpu.calls) == before


def test_naive_fixture_deadline_cannot_authorize_a_bound(h: CausalHarness) -> None:
    arm(h)
    assert h.safety.plan is not None
    expiry = datetime.fromtimestamp(h.safety.plan.deadline_at + 120, timezone.utc)
    before = list(h.cpu.calls), list(h.gpu.calls)
    with pytest.raises(RegionalFixtureError, match="aware expiry"):
        h.safety.require_bound(expiry.replace(tzinfo=None), margin=60)
    assert (h.cpu.calls, h.gpu.calls) == before


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_fixture_deadline_cannot_authorize_a_bound(
    h: CausalHarness, timestamp: float
) -> None:
    class InvalidDeadline(datetime):
        def timestamp(self) -> float:
            return timestamp

    arm(h)
    assert h.safety.plan is not None
    expiry = InvalidDeadline.fromtimestamp(
        h.safety.plan.deadline_at + 120, timezone.utc
    )
    before = list(h.cpu.calls), list(h.gpu.calls)
    with pytest.raises(RegionalFixtureError):
        h.safety.require_bound(expiry, margin=60)
    assert (h.cpu.calls, h.gpu.calls) == before


@pytest.mark.parametrize("plane", ["cpu", "gpu"])
def test_failed_resume_connection_validation_clears_old_success(
    h: CausalHarness, plane: str
) -> None:
    arm(h)
    h.safety.admit_fixture()
    assert h.safety.finish()["retired"] is True
    getattr(h.cpu.regional.settings, plane + "_kubeconfig").write_text(
        "changed controlled connection identity\n"
    )
    before = cpu_mutations(h), gpu_mutations(h)
    with pytest.raises(RegionalFixtureError, match="journal identity"):
        h.safety.resume_cleanup()
    assert h.safety.quiescent is False, (
        "connection drift must invalidate cached success"
    )
    assert (cpu_mutations(h), gpu_mutations(h)) == before


@pytest.mark.parametrize("field", ["workload_ids", "runtime_profile_version"])
def test_closed_receipt_must_still_bind_the_loaded_plan(
    h: CausalHarness, field: str
) -> None:
    arm(h)
    h.safety.admit_fixture()
    assert h.safety.finish()["retired"] is True
    assert h.safety.watchdog is not None
    data = h.cpu.journal()
    data["plan"][field] = (
        ["training/another-workload"] if field == "workload_ids" else "another-profile"
    )
    assert data["quiescence"]["plan_sha256"] != wire.digest(data["plan"])
    write_json_atomic(h.safety.watchdog.path, data)
    before = cpu_mutations(h), gpu_mutations(h)
    try:
        report = h.safety.resume_cleanup()
    except RegionalFixtureError:
        pass
    else:
        assert report["quiescent"] is False and report["errors"], (
            "a receipt for the old plan cannot certify the modified plan",
            report,
        )
    assert h.safety.quiescent is False
    assert (cpu_mutations(h), gpu_mutations(h)) == before


@pytest.mark.parametrize("fresh_process", [False, True])
def test_recreated_fence_after_closure_is_preserved_and_invalidates_success(
    h: CausalHarness, fresh_process: bool
) -> None:
    arm(h)
    h.safety.admit_fixture()
    original = copy.deepcopy(h.gpu.objects["validatingadmissionpolicy"])
    assert h.safety.finish()["retired"] is True
    original["metadata"]["uid"] = "foreign-recreated-policy"
    h.gpu.objects["validatingadmissionpolicy"] = original
    before = cpu_mutations(h), gpu_mutations(h)
    controller = fresh(h) if fresh_process else h.safety
    report = controller.resume_cleanup() if fresh_process else controller.finish()
    assert report["quiescent"] is False and report["retired"] is False, report
    assert report["errors"] and controller.quiescent is False, report
    assert h.gpu.objects == {"validatingadmissionpolicy": original}
    assert (cpu_mutations(h), gpu_mutations(h)) == before


def test_a_stale_real_receipt_cannot_complete_a_new_silent_observer(
    h: CausalHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    arm(h)
    h.safety.admit_fixture()
    h.acknowledge(h.safety.before_post())
    h.gpu.failed_delete.add("validatingadmissionpolicybinding")
    failed = h.safety.finish()
    assert failed["quiescent"] is False and h.safety.watchdog is not None
    previous = h.safety.watchdog.record.quiescence
    assert previous is not None and previous.source_complete is True
    assert previous.workflow_ids == ["workflow-a"]
    h.gpu.failed_delete.clear()
    h.observer_enabled = False
    h.cpu.clock.sleep(31)
    sleep = h.cpu.clock.sleep
    monkeypatch.setattr(h.cpu.clock, "sleep", lambda seconds: sleep(max(seconds, 60)))
    controller = fresh(h)
    report = controller.resume_cleanup()
    assert report["quiescent"] is False and report["retired"] is False, report
    assert controller.quiescent is False and h.gpu.objects, report
    assert "receipt" not in report and report["errors"], report
    journal = h.cpu.journal()
    observer = journal["jobs"][-1]
    assert observer["cleanup_id"] is not None
    assert observer["sequence_floor"] >= previous.sequence
    assert journal["quiescence"] is None and journal["closed"] is False
    assert observer["resource"]["removed"] is False


@pytest.mark.parametrize("drift", ["namespace", "deployment", "configuration"])
def test_runtime_drift_cannot_authorize_fixture_post_or_retirement(
    h: CausalHarness, drift: str
) -> None:
    arm(h)
    assert h.safety.watchdog is not None
    if drift == "namespace":
        h.cpu.objects["namespace", h.cpu.regional.settings.namespace]["metadata"][
            "uid"
        ] = "replacement-namespace"
    elif drift == "deployment":
        h.cpu.objects["deployment", resources.DEPLOYMENT]["metadata"]["generation"] += 1
    else:
        name = next(iter(h.safety.watchdog.runtime.config_identities))
        h.cpu.objects["configmap", name]["metadata"]["resourceVersion"] = (
            "changed-version"
        )
    created = sum(verb == "create" for verb, _, _ in h.cpu.calls)
    for action in (h.safety.admit_fixture, h.safety.before_post):
        with pytest.raises(RegionalFixtureError):
            action()
    report = h.safety.finish()
    assert report["quiescent"] is False and report["retired"] is False, report
    assert h.safety.quiescent is False and h.gpu.objects, report
    assert sum(verb == "create" for verb, _, _ in h.cpu.calls) == created


def test_remaining_wait_budget_is_not_permission_to_submit_after_expiry(
    h: CausalHarness,
) -> None:
    arm(h, window=120)
    h.safety.admit_fixture()
    h.cpu.clock.sleep(121)
    assert h.safety.remaining_seconds() == 1
    with pytest.raises(RegionalFixtureError):
        h.safety.before_post()
    assert h.safety.control is not None
    snapshot = h.safety.control.read()
    assert snapshot.control.producer.state == "NOT_STARTED"
    assert snapshot.control.revocation is not None
    assert snapshot.control.revocation.reason == "DEADLINE"
    assert h.gpu.objects, "expiry must not implicitly remove the activation fence"
