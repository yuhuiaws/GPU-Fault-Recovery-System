"""BOOT-020 compares live captures without their capture time.

Live 2026-10-13 the NOOP stage failed "NOOP changed live state" on
``cpu_restart_baseline.captured_at`` alone: every ``_capture_previous()``
records when it began, so two captures of an unchanged release never compared
equal. The stage-level proofs are in ``test_boot020_release_rolling``.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import pytest

from scripts.e2e.regional import boot020_live_capture as captures
from scripts.e2e.regional.boot020_deployment_reads import AcceptanceCheckError

API_HA = "gpu-fault-api-ha"
WORKER = "gpu-fault-control-worker"
SPOOL = "gpu-fault-telemetry-spool-worker"
FIRST = 1791900788.36
SECOND = 1791900920.98


def _capture(captured_at: float = FIRST) -> dict[str, Any]:
    return {
        "release_id": "site",
        "cpu_wheel": "cpu-v1",
        "cpu_restart_baseline": {
            "captured_at": captured_at,
            "spec_sha256": {API_HA: "a" * 64, WORKER: "b" * 64, SPOOL: "c" * 64},
        },
        "clusters": {"cluster-a": {"wheel": "executor-v1"}},
    }


def _specs(capture: dict[str, Any]) -> dict[str, str]:
    specs: dict[str, str] = capture["cpu_restart_baseline"]["spec_sha256"]
    return specs


def _rollback(**restarted: dict[str, list[str]]) -> dict[str, Any]:
    """A rollback deploy result whose CPU stages restarted ``restarted``."""

    return {
        "phase": "rolled-back",
        "rollback_timing": {
            "t_safe_seconds": 30.0,
            "phases": {
                phase: {"details": {"cpu_restart": {"restarted": roles, "kept": []}}}
                for phase, roles in restarted.items()
            },
        },
    }


def test_two_captures_that_differ_only_in_their_capture_time_are_the_same() -> None:
    before, after = _capture(FIRST), _capture(SECOND)

    assert before != after, "two live captures always differ in their capture time"
    assert captures.same_capture(before, after), "the capture time is no live state"
    assert captures.same_live({"live": before}, {"live": after}), (
        "two snapshots of an unchanged release observed the same live release"
    )


def test_a_spec_digest_difference_still_counts() -> None:
    before, after = _capture(FIRST), _capture(SECOND)
    _specs(after)[API_HA] = "d" * 64

    assert not captures.same_capture(before, after), "a Deployment spec moved"


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda capture: capture.update(cpu_wheel="cpu-v2"), id="field"),
        pytest.param(
            lambda capture: capture["clusters"]["cluster-a"].update(wheel="v2"),
            id="nested-field",
        ),
        pytest.param(
            lambda capture: capture.update(executor_internal_error_total=1),
            id="added-field",
        ),
        pytest.param(lambda capture: _specs(capture).pop(SPOOL), id="spec-dropped"),
        pytest.param(
            lambda capture: capture["cpu_restart_baseline"].update(spec_sha256={}),
            id="no-spec-digests",
        ),
        pytest.param(
            lambda capture: capture["cpu_restart_baseline"].update(extra=1),
            id="other-baseline-field",
        ),
        pytest.param(
            lambda capture: capture.pop("cpu_restart_baseline"), id="no-baseline"
        ),
    ],
)
def test_every_other_difference_still_counts(
    change: Callable[[dict[str, Any]], object],
) -> None:
    before, after = _capture(FIRST), _capture(SECOND)
    change(after)

    assert not captures.same_capture(before, after), "a live difference was ignored"
    assert not captures.same_capture(after, before), "the comparison is symmetric"


def test_the_comparison_never_changes_its_inputs() -> None:
    before, after = _capture(FIRST), _capture(SECOND)
    _specs(after)[WORKER] = "e" * 64
    pristine = copy.deepcopy((before, after))

    compared = captures.comparable_capture(after, restarted={WORKER})
    captures.same_capture(before, after, restarted={WORKER})
    captures.assert_restored(
        {"live": before}, {"live": after}, _rollback(cpu_restore={"worker": ["x"]})
    )

    assert (before, after) == pristine, "the compared captures were modified"
    assert compared == {
        **pristine[1],
        "cpu_restart_baseline": {"spec_sha256": {API_HA: "a" * 64, SPOOL: "c" * 64}},
    }


@pytest.mark.parametrize(
    ("left", "right", "same"),
    [(None, None, True), (None, _capture(), False), ("rel-a", "rel-a", True)],
)
def test_a_value_that_is_no_capture_compares_as_it_is(
    left: object, right: object, same: bool
) -> None:
    assert captures.same_capture(left, right) is same


def test_a_rollback_names_the_tiers_every_cpu_stage_restarted() -> None:
    rollback = _rollback(
        controller_stage={"worker": ["config-core-changed"]},
        cpu_restore={"ingress": ["pin-proof-failed"], "unknown-tier": ["x"]},
    )

    assert captures.rollback_restarted_deployments(rollback) == {WORKER, API_HA}


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"rollback_timing": None},
        {"rollback_timing": {"t_safe_seconds": 30.0}},
        {"rollback_timing": {"phases": {"restore": {"status": "COMPLETED"}}}},
        {"rollback_timing": {"phases": {"cpu_restore": {"details": {}}}}},
    ],
)
def test_a_rollback_without_restart_evidence_names_no_tier(
    result: dict[str, Any],
) -> None:
    assert captures.rollback_restarted_deployments(result) == frozenset()


def test_a_rollback_may_move_only_the_digests_of_the_tiers_it_restarted() -> None:
    before, rolled_back = {"live": _capture(FIRST)}, {"live": _capture(SECOND)}
    _specs(rolled_back["live"])[WORKER] = "e" * 64
    worker_restarted = _rollback(cpu_restore={"worker": ["spec-changed"]})

    captures.assert_restored(before, rolled_back, worker_restarted)
    with pytest.raises(AcceptanceCheckError, match="did not restore live state"):
        captures.assert_restored(before, rolled_back, _rollback())
    _specs(rolled_back["live"])[SPOOL] = "f" * 64
    with pytest.raises(AcceptanceCheckError, match="did not restore live state"):
        captures.assert_restored(before, rolled_back, worker_restarted)
    _specs(rolled_back["live"])[SPOOL] = "c" * 64
    rolled_back["live"]["cpu_wheel"] = "cpu-v2"
    with pytest.raises(AcceptanceCheckError, match="did not restore live state"):
        captures.assert_restored(before, rolled_back, worker_restarted)
