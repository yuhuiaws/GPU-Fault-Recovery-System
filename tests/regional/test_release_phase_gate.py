"""Evidence binds only to a committed release (``release_phase_gate``).

Live: after BOOT-020's rollback stage the release state stayed ``rolled-back``,
still naming the candidate that failed while every Pod ran the previous
release, and the next case's preflight bound its evidence to that candidate.
Every phase the release engine writes is refused here except a committed
``complete``; the message names the phase, the recorded id and the remedy.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gpu_fault_release.regional_admin_commands import RESUMABLE_PHASES, ROLLBACK_PHASES
from gpu_fault_release.regional_release_orchestration import SUPERSEDABLE_PHASES
from gpu_fault_release.regional_release_progress import BOOTSTRAP_PHASES
from scripts.e2e.regional import boot020_restore_stage as restore_stage
from scripts.e2e.regional import regional_live_fixture as live
from scripts.e2e.regional import release_phase_gate as gate
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._cov95_common_live import LiveModel

CANDIDATE = "release-candidate"
# Every phase the release engine records short of a committed ``complete``:
# the forward and fail-forward phases, every rollback phase, every bootstrap
# phase, the finished rollback and the data-plane pause.
NON_COMMITTED_PHASES = sorted(
    RESUMABLE_PHASES
    | ROLLBACK_PHASES
    | BOOTSTRAP_PHASES
    | {"rolled-back", "data-plane-paused"}
)
STATE_CONFIGMAP = ("cpu", "configmap", "gpu-fault-regional-release-state")


def state(phase: str | None, **fields: Any) -> dict[str, Any]:
    value: dict[str, Any] = {"release_id": CANDIDATE, **fields}
    if phase is not None:
        value["phase"] = phase
    return value


def committed(**fields: Any) -> dict[str, Any]:
    return state("complete", transaction_committed=True, **fields)


def assert_names_phase_id_and_remedy(message: str | None, phase: str) -> str:
    assert message is not None, f"phase {phase!r} was accepted as evidence"
    assert f"phase {phase!r}" in message, message
    assert f"recorded release_id {CANDIDATE!r}" in message, message
    assert "binds no evidence" in message, message
    assert "gpu-fault-admin deploy` of the site's own release" in message, message
    return message


def serve(fixture: LiveModel, document: dict[str, Any]) -> None:
    fixture.documents[STATE_CONFIGMAP] = {"data": {"state.json": json.dumps(document)}}


@pytest.mark.parametrize("flag", [False, True, None], ids=["false", "true", "unset"])
@pytest.mark.parametrize("phase", NON_COMMITTED_PHASES)
def test_every_non_committed_phase_binds_no_evidence(
    phase: str, flag: bool | None
) -> None:
    """The commit flag alone never makes a phase other than ``complete`` bind."""

    document = state(phase, transaction_committed=flag)
    message = assert_names_phase_id_and_remedy(
        gate.release_phase_error(document), phase
    )

    with pytest.raises(gate.ReleasePhaseError) as refused:
        gate.committed_release_id(document)
    assert str(refused.value) == message, "both entry points must say the same"
    assert (refused.value.phase, refused.value.release_id) == (phase, CANDIDATE), (
        "the refusal must carry the phase and the recorded id it refused"
    )
    assert isinstance(refused.value, RegionalFixtureError), (
        "callers that already stop on a fixture error must stop on this too"
    )


@pytest.mark.parametrize(
    ("phase", "reason"),
    [
        (
            "rolled-back",
            "a rollback finished: the recorded release_id is the candidate that "
            "failed and the site runs its previous release",
        ),
        ("rollback-started", "a rollback is active (in progress or stopped)"),
        ("rollback-failed", "a rollback is active (in progress or stopped)"),
        ("bootstrap-started", "the site's bootstrap has not finished"),
        ("bootstrap-cleaned", "the site's bootstrap has not finished"),
        ("failed", "--supersede-failed-transaction"),
        ("partial-convergence", "--supersede-failed-transaction"),
        ("data-plane-paused", "the data-plane rollout paused on a failed cluster"),
        ("cpu-staged", "a release transaction is in flight or stopped"),
        ("verified", "a release transaction is in flight or stopped"),
    ],
)
def test_each_phase_family_says_why_it_binds_nothing(phase: str, reason: str) -> None:
    message = gate.release_phase_error(state(phase))
    assert message is not None and reason in message, message


def test_a_pending_rollback_cleanup_tells_the_operator_it_takes_two_deploys() -> None:
    message = gate.release_phase_error(
        state("rolled-back", rollback_cleanup_completed=False)
    )
    assert message is not None, "a rolled-back state was accepted"
    assert "the rollback cleanup is still pending" in message, message
    assert "a second one deploys the release" in message, message
    finished = gate.release_phase_error(
        state("rolled-back", rollback_cleanup_completed=True)
    )
    assert finished is not None and "cleanup" not in finished, finished


@pytest.mark.parametrize("flag", [False, None, "true", 1, "missing"], ids=repr)
def test_complete_binds_only_once_its_transaction_is_committed(flag: object) -> None:
    """``complete`` uncommitted can still roll back (the deploy's verify and
    stability gate has not passed), so it is refused like every other phase."""

    document = state("complete")
    if flag != "missing":
        document["transaction_committed"] = flag
    message = assert_names_phase_id_and_remedy(
        gate.release_phase_error(document), "complete"
    )
    assert "its transaction is not committed" in message, message
    assert f"transaction_committed={document.get('transaction_committed')!r}" in message


def test_a_committed_complete_release_is_the_one_evidence_binds_to() -> None:
    document = committed()
    assert gate.release_phase_error(document) is None, "a committed release was refused"
    assert gate.committed_release_id(document) == CANDIDATE


@pytest.mark.parametrize(
    ("document", "fragment"),
    [
        (
            {"release_id": CANDIDATE},
            "phase none, recorded release_id 'release-candidate'",
        ),
        ({"phase": "", "release_id": CANDIDATE}, "no phase is recorded"),
        (
            {"phase": "complete", "transaction_committed": True},
            "recorded release_id none) binds no evidence: the state records no release id",
        ),
        (
            {"phase": "complete", "transaction_committed": True, "release_id": 7},
            "the state records no release id",
        ),
        ({"phase": 3, "release_id": CANDIDATE}, "no phase is recorded"),
    ],
)
def test_a_state_without_a_phase_or_an_id_binds_nothing(
    document: dict[str, Any], fragment: str
) -> None:
    message = gate.release_phase_error(document)
    assert message is not None and fragment in message, message
    with pytest.raises(gate.ReleasePhaseError, match="binds no evidence"):
        gate.committed_release_id(document)


@pytest.mark.parametrize("document", [None, [], "complete", 1])
def test_a_state_that_is_not_an_object_binds_nothing(document: object) -> None:
    assert gate.release_phase_error(document) == gate.NOT_AN_OBJECT
    with pytest.raises(RegionalFixtureError, match=gate.NOT_AN_OBJECT) as refused:
        gate.committed_release_id(document)
    assert not isinstance(refused.value, gate.ReleasePhaseError), (
        "an unreadable state has no phase or id to report"
    )


def test_the_fail_forward_opt_in_admits_only_the_stopped_phases_it_names() -> None:
    allowed = gate.FAIL_FORWARD_STOPPED_PHASES
    assert allowed == SUPERSEDABLE_PHASES, (
        "the witness's stopped phases are the engine's supersedable fail-forward ones"
    )
    for phase in sorted(allowed):
        assert gate.release_phase_error(state(phase), allowed_phases=allowed) is None
        assert (
            gate.committed_release_id(state(phase), allowed_phases=allowed) == CANDIDATE
        )
        missing_id = gate.release_phase_error({"phase": phase}, allowed_phases=allowed)
        assert missing_id is not None and "records no release id" in missing_id, (
            "an allowed phase still needs the release id it binds to"
        )
    for phase in sorted(set(NON_COMMITTED_PHASES) - allowed):
        assert_names_phase_id_and_remedy(
            gate.release_phase_error(state(phase), allowed_phases=allowed), phase
        )
    uncommitted = gate.release_phase_error(state("complete"), allowed_phases=allowed)
    assert uncommitted is not None and "not committed" in uncommitted, uncommitted


def test_complete_can_never_be_allowed_uncommitted_and_a_bare_string_is_refused() -> (
    None
):
    with pytest.raises(ValueError, match="cannot be allowed uncommitted"):
        gate.release_phase_error(state("complete"), allowed_phases={"complete"})
    with pytest.raises(TypeError, match="collection of phase names"):
        gate.release_phase_error(state("failed"), allowed_phases="failed")


def test_the_fixture_refuses_the_rolled_back_candidate_the_live_case_bound_to(
    tmp_path: Path,
) -> None:
    """The live regression: ``release_id``, ``evidence_identity`` and the
    runtime identity all read the state; none may name the failed candidate."""

    fixture = LiveModel(tmp_path)
    serve(fixture, state("rolled-back", transaction_committed=False))

    with pytest.raises(gate.ReleasePhaseError, match="phase 'rolled-back'"):
        fixture.release_id()
    with pytest.raises(gate.ReleasePhaseError, match="phase 'rolled-back'"):
        fixture.evidence_identity()
    identity = fixture.runtime_identity()
    errors = live.runtime_identity_errors(identity)
    assert errors == [gate.release_phase_error(identity["release_state"])], errors


@pytest.mark.parametrize("phase", NON_COMMITTED_PHASES)
def test_runtime_identity_errors_name_every_non_committed_phase(
    tmp_path: Path, phase: str
) -> None:
    fixture = LiveModel(tmp_path)
    serve(fixture, state(phase, transaction_committed=False))
    errors = live.runtime_identity_errors(fixture.runtime_identity())
    assert len(errors) == 1, errors
    assert_names_phase_id_and_remedy(errors[0], phase)


def test_the_fixture_binds_a_committed_release_and_an_explicit_stopped_phase(
    tmp_path: Path,
) -> None:
    fixture = LiveModel(tmp_path)
    serve(fixture, committed())
    assert fixture.evidence_identity() == {
        "release_id": CANDIDATE,
        "cluster_id": fixture.settings.cluster_id,
    }
    serve(fixture, state("partial-convergence", transaction_committed=False))
    with pytest.raises(gate.ReleasePhaseError, match="phase 'partial-convergence'"):
        fixture.evidence_identity()
    assert fixture.evidence_identity(allow_phases=gate.FAIL_FORWARD_STOPPED_PHASES) == {
        "release_id": CANDIDATE,
        "cluster_id": fixture.settings.cluster_id,
    }
    serve(fixture, state("complete", transaction_committed=False))
    with pytest.raises(gate.ReleasePhaseError, match="not committed"):
        fixture.release_id(allow_phases=gate.FAIL_FORWARD_STOPPED_PHASES)


class RestoreStage:
    """The BOOT-020 restore stage's collaborators, with the end state injected."""

    def __init__(self, end_state: dict[str, Any]) -> None:
        self.end_state = end_state
        self.classifications = iter(({"kind": "FULL"}, {"kind": "NOOP"}))
        self.stages: list[str] = []

    def classify(self, _scenario: str) -> dict[str, Any]:
        return next(self.classifications)

    def deploy(self, _scenario: str, **_options: Any) -> dict[str, Any]:
        return {"phase": "complete", "transaction_committed": True}

    def cpu_env_alignment(self, _scenario: str) -> dict[str, Any]:
        return {"aligned": True}

    def stage(self, name: str, observe: Callable[[], Any]) -> Any:
        self.stages.append(name)
        return observe()

    def before(self, *_args: Any) -> dict[str, Any]:
        return {"live": "candidate-d"}

    def after(self, *_args: Any) -> dict[str, Any]:
        return {"live": "site-release", **self.end_state}


def require(condition: bool, message: str, context: Any = None) -> None:
    if not condition:
        raise AssertionError(message if context is None else f"{message}: {context!r}")


@pytest.mark.parametrize(
    "end_state",
    [
        state("complete", transaction_committed=False),
        state("rolled-back", transaction_committed=False),
        {"phase": "complete", "transaction_committed": True},
    ],
    ids=["uncommitted", "rolled-back", "no-release-id"],
)
def test_boot020_restore_ends_only_on_a_committed_release(
    end_state: dict[str, Any],
) -> None:
    """The restore's last observation is BOOT-020's evidence identity; the
    stages before it observe ``rolled-back`` and ``failed`` on purpose."""

    world = RestoreStage(end_state)
    with pytest.raises(AssertionError, match="restore end state: regional release"):
        restore_stage.stage_restore(world, world, world, require=require)
    assert "restore_cpu_env_alignment" not in world.stages, (
        "the restore went on past an end state that binds no evidence"
    )
    committed_world = RestoreStage(committed())
    restore_stage.stage_restore(
        committed_world, committed_world, committed_world, require=require
    )
    assert committed_world.stages[-1] == "restore_next_classification", (
        committed_world.stages
    )
