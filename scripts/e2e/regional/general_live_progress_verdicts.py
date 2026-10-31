"""GF-GENERAL-LIVE-027 graders over the training records, kept pure.

``general_live_progress`` drives the test job and reads the product's records
(``TRAINING_PROBE``); the graders that need nothing but those records and the
product's answers live here, apart from the fixture module (the architecture
gate's 1500-line file cap). ``old_attempt``: a heartbeat claiming an attempt
the product does not run is graded by the product's ingest and refused -- the
handler's own answer, not the regional ingress's queue admission -- leaves no
state, and the live attempt keeps running and being heard.
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


def ingest_decision(answer: dict[str, Any], attempt: str) -> dict[str, Any]:
    """The product's own decision on the heartbeat ``HEARTBEAT_POST`` sent.

    Where the ingest graded the POST itself (no processor), the answer's body
    is its ``TrainingHealthResult``. On the regional path the ingress only
    admits the POST to the processor queue: 202 ``{"accepted": true,
    "processor_request_id", "status_url", "coalesced"}``, where ``accepted``
    means "queued" (``app/middleware/dispatch.py``). The decision is the
    handler's answer the processor committed when it replayed the row, which
    ``status_url`` returns once COMPLETED (``app/routes/processor.py``) and the
    script followed (``receipt``). INCONCLUSIVE, never graded: an admission
    that did not reach COMPLETED within the follow bound, one the status route
    refused to show, a spool admission (no completion record), a completion
    that is not a 200 ``TrainingHealthResult`` (the processor's 5xx past its
    retry horizon, a 409, a ``STALE_SUPERSEDED`` disposition the handler never
    saw), or a decision about another heartbeat (a later sample coalesced into
    the row).
    """

    body = answer.get("body") if isinstance(answer.get("body"), dict) else {}
    request_id = body.get("processor_request_id")
    if body.get("spooled"):
        raise VariantInconclusive(
            f"the heartbeat claiming {attempt} was admitted to the telemetry spool, "
            "which keeps no completion record: the product's decision is not "
            "observable"
        )
    if request_id:
        receipt = (
            answer.get("receipt") if isinstance(answer.get("receipt"), dict) else {}
        )
        if receipt.get("terminal") is not True:
            raise VariantInconclusive(
                f"the admission {request_id} of the heartbeat claiming {attempt} did "
                f"not reach a terminal state within {receipt.get('bound_seconds')} s "
                f"({receipt.get('polls')} reads, last HTTP "
                f"{receipt.get('http_status')} {receipt.get('processor_status')}): "
                "the product's decision was not observed"
            )
        decision = receipt.get("body")
        if receipt.get("http_status") != 200 or not isinstance(decision, dict):
            raise VariantInconclusive(
                f"the processor completed {request_id} with HTTP "
                f"{receipt.get('http_status')} {str(decision)[:300]}: not the "
                "ingest's grading of the heartbeat"
            )
    else:
        decision = body
    sent = (answer.get("heartbeat") or {}).get("heartbeat_id")
    graded = decision.get("heartbeat_id")
    if decision.get("status") == "STALE_SUPERSEDED" or "accepted" not in decision:
        raise VariantInconclusive(
            f"the heartbeat claiming {attempt} was not graded by the product's "
            f"ingest: {str(decision)[:300]}"
        )
    if sent and graded and graded != sent:
        raise VariantInconclusive(
            f"the completed decision is about heartbeat {graded}, not the posted {sent}"
        )
    return decision


def foreign_refusal(answer: dict[str, Any], attempt: str) -> list[str]:
    """The product's decision on the foreign heartbeat is a refusal.

    A delivered answer leads to the decision (``ingest_decision``); anything
    else (a refused connection, an HTTP error, unreadable output) never reached
    the ingest, so nothing was refused or accepted: INCONCLUSIVE. The decision
    must be ``accepted=false``; ``accepted=true`` from the ingest itself -- not
    the admission's -- is the FAIL.
    """

    if answer.get("delivered") is not True:
        raise VariantInconclusive(
            f"the heartbeat claiming {attempt} was not graded by the product's "
            f"ingest (status {answer.get('status')}: "
            f"{str(answer.get('error') or answer.get('body'))[:300]}); nothing was "
            "refused"
        )
    decision = ingest_decision(answer, attempt)
    if decision.get("accepted") is False:
        return []
    return [
        f"the product accepted a heartbeat for {attempt}, an attempt it does not "
        f"run ({len(decision.get('findings') or [])} findings)"
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
    "ingest_decision",
    "foreign_state_errors",
    "live_kept_errors",
    "rank_states",
    "validated_by_progress",
]
