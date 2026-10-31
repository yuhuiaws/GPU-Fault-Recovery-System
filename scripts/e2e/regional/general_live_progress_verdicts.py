"""GF-GENERAL-LIVE-027 graders over the training records, kept pure.

``general_live_progress`` drives the test job and reads the product's records
(``TRAINING_PROBE``); the graders that need nothing but those records and the
product's answers live here, apart from the fixture module (the architecture
gate's 1500-line file cap). ``old_attempt``: a heartbeat claiming an attempt
the product does not run is graded by the product's ingest and refused, leaves
no state, and the live attempt keeps running and being heard.
``resumed_progress``: a stall is cleared only by the original rank's newer
progress, which the product's training validation records.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.general_live_verdicts import parse_time

#: ``resumed_progress``: the step the stalled rank repeats, then the newer one.
STALLED_STEP = 10
RESUMED_STEP = 11


def rank_states(records: dict[str, Any], rank: int) -> list[dict[str, Any]]:
    """The attempt's stored progress states of ``rank``."""

    return [
        item
        for item in records.get("progress_states") or []
        if item.get("rank") == rank
    ]


def foreign_refusal(answer: dict[str, Any], attempt: str) -> list[str]:
    """The ingest graded the foreign heartbeat and refused it.

    A delivered answer is the product's ``TrainingHealthResult``; anything else
    (a refused connection, an HTTP error, unreadable output) never reached that
    grading, so nothing was refused or accepted: INCONCLUSIVE.
    """

    body = answer.get("body") if isinstance(answer.get("body"), dict) else {}
    if answer.get("delivered") is not True or "accepted" not in body:
        raise VariantInconclusive(
            f"the heartbeat claiming {attempt} was not graded by the product's "
            f"ingest (status {answer.get('status')}: "
            f"{str(answer.get('error') or body)[:300]}); nothing was refused"
        )
    if body.get("accepted") is False:
        return []
    return [
        f"the product accepted a heartbeat for {attempt}, an attempt it does not "
        f"run ({len(body.get('findings') or [])} findings)"
    ]


def foreign_state_errors(records: dict[str, Any], attempt: str) -> list[str]:
    """No progress state of ``attempt`` in the store (``other_progress_states``)."""

    states = (records.get("other_progress_states") or {}).get(attempt)
    if states is None:
        raise VariantInconclusive(
            f"the records read listed no progress entry for {attempt}: whether the "
            "refused heartbeat left state behind is unknown"
        )
    return [f"the product stored progress for {attempt}: {states}"] if states else []


def live_kept_errors(records: dict[str, Any], answer: dict[str, Any]) -> list[str]:
    """The live attempt still RUNNING and its own heartbeats still accepted
    after the foreign one (both stamped by the same Pod's clock)."""

    errors = []
    sent = parse_time((answer.get("heartbeat") or {}).get("observed_at"))
    live = rank_states(records, 0)
    accepted = [parse_time(item.get("observed_at")) for item in live]
    if sent is None or not any(item is not None and item > sent for item in accepted):
        errors.append(
            "no heartbeat of the live attempt was accepted after the foreign one: "
            f"latest {[item.get('observed_at') for item in live]}"
        )
    phase = (records.get("observation") or {}).get("workload_phase")
    if phase != "RUNNING":
        errors.append(f"the live attempt is {phase} after the foreign heartbeat")
    return errors


def validated_by_progress(
    records: dict[str, Any], recovered: dict[str, Any], attempt_id: str
) -> list[str]:
    """The stall was cleared by the original rank's newer progress.

    The product's training validation (``health_evidence_validation``
    ``validate_training``) succeeds only on a fresh heartbeat of the finding's
    own attempt and rank whose step passed the stalled one, and records
    ``validated_attempt``/``validated_rank``/``validated_step`` on the
    SUCCEEDED VALIDATE_FABRIC. A RECOVERED record without that execution was
    cleared by something else (healthy fabric alone is not enough). No
    incident, or one that did not recover, is the ``recovered`` stage's error,
    not this one's.
    """

    if not recovered or recovered.get("state") != "RECOVERED":
        return []
    workflow = (records.get("workflows") or {}).get(
        str(recovered.get("workflow_request_id"))
    ) or {}
    seen = []
    for step in workflow.get("step_executions") or []:
        if step.get("operation") != "VALIDATE_FABRIC":
            continue
        details = step.get("details") or {}
        validated = details.get("validated_step")
        seen.append(
            f"{step.get('status')} attempt={details.get('validated_attempt')} "
            f"rank={details.get('validated_rank')} step={validated}"
        )
        if (
            step.get("status") == "SUCCEEDED"
            and details.get("validated_attempt") == attempt_id
            and details.get("validated_rank") == 0
            and isinstance(validated, int)
            and validated > STALLED_STEP
        ):
            return []
    return [
        f"no VALIDATE_FABRIC of the stall's workflow validated rank 0 of "
        f"{attempt_id} past step {STALLED_STEP}: "
        + ("; ".join(seen) or "none executed")
    ]


__all__ = [
    "RESUMED_STEP",
    "STALLED_STEP",
    "foreign_refusal",
    "foreign_state_errors",
    "live_kept_errors",
    "rank_states",
    "validated_by_progress",
]
