"""What the release engine is allowed to do at the same time, and what it is not.

Every test here is about an ordering the transaction depends on: the phases that
may overlap (because they touch disjoint clusters and disjoint AWS resources) and
the joins that must happen before the next phase is allowed to start. The
blocking-`Event` pattern is deliberate -- a serialized implementation does not
fail an assertion, it fails to finish, so each overlap is expressed as two phases
that can only both complete if they really run together.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_admin_commands as ADMIN
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from tests.regional._release_orchestrator_support import phase_release

ROOT = Path(__file__).resolve().parents[2]
OVERLAP_TIMEOUT = 10.0


@pytest.fixture(autouse=True)
def _no_ambient_admin_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase narration must never reach a real operator log from a test."""

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)


def component(name: str):
    return getattr(ORCHESTRATION.ReleaseComponent, name)


def plan_of(*names: str):
    return ORCHESTRATION.ReleaseExecutionPlan(
        nodes=tuple(component(name) for name in names)
    )


def full_diff(*changed: str):
    return ORCHESTRATION.ReleaseDiff(
        kind=ORCHESTRATION.ReleaseChangeKind.FULL, changed=frozenset(changed)
    )


def run_phases(release, *, plan, diff, previous=None, completed_phases=None):
    return ORCHESTRATION.run_upgrade_phases(
        release,
        diff=diff,
        plan=plan,
        previous=previous if previous is not None else {},
        completed_phases=set() if completed_phases is None else set(completed_phases),
        completed_clusters=set(),
        registry_staged=False,
    )


def test_candidate_preflight_gates_schema_and_cpu_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unusable candidate must be detected before DDL or Pod mutations."""

    calls: list[str] = []

    def preflight(_release, _plan) -> None:
        calls.append("preflight")

    monkeypatch.setattr(ORCHESTRATION, "preflight_upgrade_mutations", preflight)
    saves: list[dict] = []
    release = phase_release(
        calls, saves, _apply_cpu=lambda **_kwargs: calls.append("cpu-stage")
    )

    run_phases(
        release,
        plan=plan_of("SCHEMA", "REGISTRY", "CPU_STAGE", "VERIFY"),
        diff=full_diff("database_schema", "control_plane_wheel"),
    )

    assert calls.index("preflight") < calls.index("schema") < calls.index("cpu-stage")
    assert release.state["phase"] == "complete"
    # The proof is persisted by the STARTED checkpoint, before schema mutation.
    first = next(
        item
        for item in saves
        if "candidate-preflight-ready" in item["completed_phases"]
    )
    assert first["phase"] == "candidate-preflight-ready"
    assert first["component_progress"]["global"]["schema"]["status"] == "STARTED"


def test_gpu_rollout_waits_for_candidate_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The data plane may not start until the preflight has passed.

    The preflight is the only thing that proves the candidate images and node
    bundle are usable before a cluster is mutated, and `upgrade_gpu_clusters`
    tells each target it was already preflighted.
    """

    calls: list[str] = []
    preflight_done = threading.Event()

    def preflight(_release, _plan) -> None:
        # Long enough that a rollout which does not join would run first.
        threading.Event().wait(0.05)
        calls.append("preflight")
        preflight_done.set()

    def upgrade_target(target, _diff, _plan, *, progress, candidate_preflighted):
        assert candidate_preflighted is True
        assert preflight_done.is_set(), (
            "a GPU cluster was mutated before the candidate preflight finished"
        )
        calls.append(f"gpu:{target.cluster_id}")
        progress(ORCHESTRATION.ReleaseComponent.EXECUTOR, "COMPLETED", None)

    release = phase_release(
        calls,
        clusters=(SimpleNamespace(cluster_id="gpu-a"),),
        _upgrade_gpu_target=upgrade_target,
        _preflight_gpu_deployments=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(ORCHESTRATION, "preflight_upgrade_mutations", preflight)
    run_phases(
        release,
        plan=plan_of("CPU_STAGE", "EXECUTOR", "VERIFY"),
        diff=full_diff("control_plane_wheel", "executor_wheel"),
    )

    assert calls.index("preflight") < calls.index("gpu:gpu-a")


def test_preflight_failure_leaves_schema_and_cpu_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure is durable, but no unstarted runtime component is compensated."""

    calls: list[str] = []

    def preflight(_release, _plan) -> None:
        raise ORCHESTRATION.ReleaseError("candidate preflight rejected node-a")

    monkeypatch.setattr(ORCHESTRATION, "preflight_upgrade_mutations", preflight)
    monkeypatch.setattr(
        ORCHESTRATION, "prepare_upgrade_credentials", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        ORCHESTRATION, "_validate_upgrade_transaction", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        ORCHESTRATION,
        "_upgrade_context",
        lambda *_args, **_kwargs: ({"metadata": {}}, set(), set(), False),
    )
    release = phase_release(
        calls,
        _apply_cpu=lambda **_kwargs: pytest.fail("CPU moved before candidate proof"),
        _ensure_schema=lambda: pytest.fail("DDL ran before candidate proof"),
        _ensure_contexts=lambda: None,
        _require_cpu_secrets=lambda: None,
        _remote_commands_are_idle=lambda: True,
        _upgrade_gpu_target=lambda *_args, **_kwargs: pytest.fail(
            "the data plane rolled after the preflight failed"
        ),
    )
    release.config.auto_rollback = False

    with pytest.raises(ORCHESTRATION.ReleaseError, match="candidate preflight"):
        ORCHESTRATION.upgrade_release(
            release,
            # `agent_config` is what puts the CPU stage in the plan: the stage
            # exists to hold the compatibility window open across a pin change.
            diff=full_diff("control_plane_wheel", "agent_config", "database_schema"),
        )

    assert "cpu-stage" not in calls
    assert release.state["phase"] == "failed"
    assert "candidate preflight rejected node-a" in release.state["original_failure"]
    assert "cpu-staged" not in release.state["completed_phases"]
    assert release.state["component_progress"]["global"] == {}


def test_endpoint_waits_for_the_staged_ingress_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The NLB wait may not run while the ingress Pods are cycling.

    `_wait_targets_healthy` counts healthy targets behind the listener, and the
    staged apply is restarting exactly those Pods -- so a wait overlapping it can
    pass on a target set that is about to be replaced. It still overlaps whatever
    is left of the candidate node preflight, and it is still joined before the
    data plane, because every cluster verifies the control-plane endpoint as its
    first step.
    """

    calls: list[str] = []
    barrier_done = threading.Event()
    endpoint_done = threading.Event()

    def preflight(_release, _plan) -> None:
        calls.append("preflight")

    monkeypatch.setattr(ORCHESTRATION, "preflight_upgrade_mutations", preflight)

    def apply_nlb() -> None:
        assert barrier_done.is_set(), (
            "the endpoint wait started while the ingress role was still restarting"
        )
        calls.append("endpoint")
        endpoint_done.set()

    def heartbeats(_expected, **kwargs) -> None:
        calls.append("pin-barrier" if kwargs.get("required_identity") else "barrier")
        barrier_done.set()

    def upgrade_target(target, _diff, _plan, *, progress, candidate_preflighted):
        assert endpoint_done.is_set(), (
            "a GPU cluster was rolled before the control-plane endpoint was ready"
        )
        calls.append(f"gpu:{target.cluster_id}")
        progress(ORCHESTRATION.ReleaseComponent.EXECUTOR, "COMPLETED", None)

    saves: list[dict] = []
    release = phase_release(
        calls,
        saves,
        clusters=(SimpleNamespace(cluster_id="gpu-a"),),
        _apply_nlb=apply_nlb,
        _wait_candidate_cpu_agent_heartbeats=heartbeats,
        _upgrade_gpu_target=upgrade_target,
    )

    run_phases(
        release,
        plan=plan_of("CPU_STAGE", "ENDPOINT", "EXECUTOR", "VERIFY"),
        diff=full_diff("control_plane_wheel", "agent_config", "executor_wheel"),
    )

    assert calls.index("cpu-stage") < calls.index("barrier")
    assert calls.index("barrier") < calls.index("endpoint")
    assert calls.index("endpoint") < calls.index("gpu:gpu-a")
    rolling = next(
        item for item in saves if item.get("release_lifecycle") == "ROLLING_CLUSTERS"
    )
    assert "endpoint-ready" in rolling["completed_phases"]


def test_observability_overlaps_data_plane(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure monitoring concurrently, then publish expectations after both join."""

    monkeypatch.setattr(
        ORCHESTRATION, "preflight_upgrade_mutations", lambda _self, _plan: None
    )
    calls: list[str] = []
    data_plane_started = threading.Event()

    def apply_observability() -> None:
        assert data_plane_started.wait(timeout=OVERLAP_TIMEOUT), (
            "observability did not overlap the data plane"
        )
        calls.append("observability")

    def upgrade_target(target, _diff, _plan, *, progress, candidate_preflighted):
        data_plane_started.set()
        calls.append(f"gpu:{target.cluster_id}")
        progress(ORCHESTRATION.ReleaseComponent.EXECUTOR, "COMPLETED", None)

    release = phase_release(
        calls,
        clusters=(SimpleNamespace(cluster_id="gpu-a"),),
        _apply_control_plane_observability=apply_observability,
        _upgrade_gpu_target=upgrade_target,
    )

    run_phases(
        release,
        plan=plan_of("OBSERVABILITY", "EXECUTOR", "CPU_FINALIZE", "VERIFY"),
        diff=full_diff("observability", "executor_wheel", "control_plane_wheel"),
    )

    assert calls.index("gpu:gpu-a") < calls.index("observability")
    assert (
        max(calls.index("observability"), calls.index("gpu:gpu-a"))
        < calls.index("expected-rules")
        < calls.index("cpu-finalize")
    ), "expected collectors were published before both installation branches joined"
    assert "observability-ready" in release.state["completed_phases"]


def _stamps(saves: list[dict]) -> list[str]:
    return [str(item["phase"]) for item in saves]


def test_a_merged_write_never_records_an_earlier_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recorded phase only ever moves forward.

    A write carries every phase that completed since the last one, and the
    candidate preflight -- submitted first, joined last -- is the earliest of
    them. Stamping the *pending* maximum alone therefore moved the record
    backwards at the write that opens the data plane, and a phase earlier than
    the work already done is a resume that redoes it (or, if the phase is not
    resumable at all, a release that starts over).
    """

    monkeypatch.setattr(
        ORCHESTRATION, "preflight_upgrade_mutations", lambda _self, _plan: None
    )
    calls: list[str] = []
    saves: list[dict] = []
    release = phase_release(
        calls,
        saves,
        clusters=(SimpleNamespace(cluster_id="gpu-a"),),
        _upgrade_gpu_target=lambda target, _diff, _plan, *, progress, **_kwargs: (
            progress(ORCHESTRATION.ReleaseComponent.EXECUTOR, "COMPLETED", None)
        ),
    )

    run_phases(
        release,
        # The plan that isolates the stamp: the observability submit writes state
        # after the CPU stage completed, so by the time the preflight is joined
        # the only phase left pending is the *earliest* one in the release. An
        # endpoint in the plan would hide that behind a late phase in the same
        # write.
        plan=plan_of(
            "SCHEMA", "REGISTRY", "CPU_STAGE", "OBSERVABILITY", "EXECUTOR", "VERIFY"
        ),
        diff=full_diff(
            "database_schema",
            "control_plane_wheel",
            "agent_config",
            "observability",
            "executor_wheel",
        ),
    )

    order = ORCHESTRATION.UPGRADE_PHASE_ORDER
    ranked = [phase for phase in _stamps(saves) if phase in order]
    assert ranked == sorted(ranked, key=order.index), _stamps(saves)
    rolling = next(
        item for item in saves if item.get("release_lifecycle") == "ROLLING_CLUSTERS"
    )
    assert rolling["phase"] != "candidate-preflight-ready", (
        "the write that opens the data plane was stamped with the earliest phase "
        "it carried, not the furthest one it proves"
    )


def test_every_recorded_phase_can_be_resumed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A crash at any write this engine makes has to be resumable.

    `next_deploy` decides between resuming the transaction and starting a fresh
    one from the recorded phase alone, and a fresh one discards
    `completed_phases` and `cluster_attempts` -- i.e. it re-runs mutations that
    the state says are already done. So every phase the engine can leave behind
    has to be in `RESUMABLE_PHASES`, including the ones the parallel phases
    introduced.
    """

    admin = ADMIN
    monkeypatch.setattr(
        ORCHESTRATION, "preflight_upgrade_mutations", lambda _self, _plan: None
    )
    monkeypatch.setattr(
        admin,
        "retry_release_diff",
        lambda _release, _state: admin.diff_from_changed({"control_plane_wheel"}),
    )
    calls: list[str] = []
    saves: list[dict] = []
    release = phase_release(
        calls,
        saves,
        clusters=(SimpleNamespace(cluster_id="gpu-a"),),
        _upgrade_gpu_target=lambda target, _diff, _plan, *, progress, **_kwargs: (
            progress(ORCHESTRATION.ReleaseComponent.EXECUTOR, "COMPLETED", None)
        ),
    )

    run_phases(
        release,
        plan=plan_of("CPU_STAGE", "OBSERVABILITY", "EXECUTOR", "VERIFY"),
        diff=full_diff(
            "control_plane_wheel", "agent_config", "observability", "executor_wheel"
        ),
    )

    unresumable = [
        phase
        for phase in _stamps(saves)
        # `complete` is the committed end of the transaction, not a crash point
        # this engine leaves behind mid-flight.
        if phase != "complete" and phase not in admin.RESUMABLE_PHASES
    ]
    assert unresumable == [], unresumable
    rolling = next(
        item for item in saves if item.get("release_lifecycle") == "ROLLING_CLUSTERS"
    )
    assert (
        admin.next_deploy(release, {**rolling, "release_id": "candidate"})["resume"]
        is True
    )


def test_background_failure_is_reported_with_the_error_that_stopped_the_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A phase that dies in the pool must not die silently.

    Nothing joins the observability future once the data plane has failed, so its
    exception would never be retrieved: it would vanish with the pool, leaving no
    line in the log and nothing in the failure record, while a rollback ran
    against a monitoring stack somebody was still mutating.
    """

    monkeypatch.setattr(
        ORCHESTRATION, "preflight_upgrade_mutations", lambda _self, _plan: None
    )
    calls: list[str] = []
    data_plane_failed = threading.Event()

    def apply_observability() -> None:
        assert data_plane_failed.wait(timeout=OVERLAP_TIMEOUT), (
            "observability did not overlap the data plane"
        )
        raise ORCHESTRATION.ReleaseError("amp workspace rejected the rules")

    def upgrade_target(*_args, **_kwargs):
        data_plane_failed.set()
        raise ORCHESTRATION.ClusterLocalReleaseError("gpu-a agents did not converge")

    release = phase_release(
        calls,
        clusters=(SimpleNamespace(cluster_id="gpu-a"),),
        _apply_control_plane_observability=apply_observability,
        _upgrade_gpu_target=upgrade_target,
    )

    with pytest.raises(ORCHESTRATION.PartialClusterRolloutError) as failure:
        run_phases(
            release,
            plan=plan_of("OBSERVABILITY", "EXECUTOR", "CPU_FINALIZE", "VERIFY"),
            diff=full_diff("observability", "executor_wheel", "control_plane_wheel"),
        )

    message = str(failure.value)
    assert "gpu-a agents did not converge" in message
    assert "observability also failed" in message
    assert "amp workspace rejected the rules" in message
    assert "expected-rules" not in calls, (
        "a failed GPU/monitoring join still published collector expectations"
    )
    # The type decides pause-versus-rollback, so it must survive the annotation:
    # one cluster's convergence timeout is not a reason to revert the fleet.
    assert not isinstance(failure.value, ORCHESTRATION.ReleaseError) or isinstance(
        failure.value, ORCHESTRATION.PartialClusterRolloutError
    )


def test_a_merged_checkpoint_narrates_every_phase_it_carried(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Merging the writes must not cost the phases their line in the log.

    `save_state` narrates the phase it was called with, so folding four phases
    into one write left three of them with no `release-phase` line at all -- and
    those lines are the only timestamps in a deploy log, i.e. the only way to see
    which phase is the slow one.
    """

    monkeypatch.setattr(
        ORCHESTRATION, "preflight_upgrade_mutations", lambda _self, _plan: None
    )
    calls: list[str] = []
    saves: list[dict] = []
    release = phase_release(
        calls,
        saves,
        clusters=(SimpleNamespace(cluster_id="gpu-a"),),
        _upgrade_gpu_target=lambda target, _diff, _plan, *, progress, **_kwargs: (
            progress(ORCHESTRATION.ReleaseComponent.EXECUTOR, "COMPLETED", None)
        ),
    )

    run_phases(
        release,
        plan=plan_of(
            "CPU_STAGE", "OBSERVABILITY", "EXECUTOR", "CPU_FINALIZE", "VERIFY"
        ),
        diff=full_diff(
            "control_plane_wheel", "agent_config", "observability", "executor_wheel"
        ),
    )

    # The fake `_save_state` does not narrate, so these lines are exactly the ones
    # the flush is responsible for: the phases that shared somebody else's write.
    # Production narrates the stamp from `save_state`, which is why the invariant
    # is "every completed phase is either a stamp or a line", not "every phase has
    # a line here".
    narrated = {
        line.split()[2]
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("release-phase ")
    }
    stamps = {str(item["phase"]) for item in saves}
    silent = [
        phase
        for phase in sorted(release.state["completed_phases"])
        if phase not in stamps and phase not in narrated
    ]
    assert silent == [], silent
    # Candidate proof is now its own early stamp, before the first CPU mutation.
    assert "candidate-preflight-ready" in stamps, stamps
