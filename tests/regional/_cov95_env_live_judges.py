"""Hand-built GF-ENV-LIVE polls for ``test_cov95_env_live_judges``.

The documents one poll of ``env_live_watch`` hands a judge (the GPU state, the
node activity, the Node), in the shapes ``tests/regional/_env_live_world.py``
serves, on the H200 T.Limit field the world grades.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.adapters.common import QUARANTINE_TAINT
from gpu_fault.health_recovery_contract import COOLDOWN_SECONDS
from scripts.e2e.regional import env_live_judges as judges
from scripts.e2e.regional import env_live_watch as watch
from scripts.e2e.regional.env_live_expectations import Expectation
from tests.regional._env_live_world import BOOT, GPU, NODE, PROXY, REVIEW_OPERATIONS

T0 = datetime(2026, 11, 5, 4, 0, tzinfo=timezone.utc)
FIELD = "gpu_tlimit_c"
SEVERE = -3.0
HEALTHY = 58.0
WARNING = 0.5
SOURCE = "NVIDIA_DEVICE_TLIMIT"
POLICY = "SITE_NVIDIA_DEVICE_LIMIT_DERIVED"
REVIEW_HOLD = COOLDOWN_SECONDS + 15.0
CONSUMED = 100.0
#: max(consumed, the accepted clear at 90 s) + the cooldown hold.
HOLD_END = CONSUMED + REVIEW_HOLD
REVIEW = "GF-ENV-LIVE-004"
STAGES = ("reaction", "grading", "scope", "workflow", "isolation", "task")


def at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def iso(seconds: float) -> str:
    return at(seconds).isoformat()


def expectation(kind: str, **overrides: Any) -> Expectation:
    """A GF-ENV-LIVE rule on the H200 T.Limit field, review-shaped by default."""

    values: dict[str, Any] = {
        "case_id": REVIEW,
        "plan_name": "core_tlimit/sustained",
        "kind": kind,
        "node": NODE,
        "gpu": GPU,
        "field": FIELD,
        "trigger_values": (SEVERE,),
        "planned_reads": 6,
        "names": (FIELD, "gpu_temperature_c"),
        "terms": {},
        "hold_seconds": REVIEW_HOLD,
        "threshold_source": SOURCE,
    }
    values.update(overrides)
    return Expectation(**values)


def warning_rule(**overrides: Any) -> Expectation:
    values: dict[str, Any] = {
        "case_id": "GF-ENV-LIVE-003",
        "plan_name": "core_tlimit/warning",
        "trigger_values": (WARNING,),
        "hold_seconds": 60.0,
        "require_diagnostic": True,
        "fallback_field": "gpu_temperature_c",
    }
    values.update(overrides)
    return expectation("warning", **values)


def unknown_rule() -> Expectation:
    return expectation(
        "unknown",
        case_id="GF-ENV-LIVE-008",
        plan_name="omit_reason",
        field="clock_throttle_reasons",
        trigger_values=(128.5,),
        names=("clock_throttle_reasons", "gpu_temperature_c"),
        hold_seconds=60.0,
        threshold_source=None,
    )


def replay(
    *, consumed: float | None = CONSUMED, armed: bool = True, reads: int = 6
) -> watch.ReplayProgress:
    """The proxy's read count since the arm at ``T0`` (consumed or not)."""

    return watch.ReplayProgress(
        planned_reads=6,
        armed=armed,
        proxy_endpoint=PROXY,
        start_reads=0,
        started_at=T0,
        reads=reads,
        status_reads=3,
        consumed_at=None if consumed is None else at(consumed),
    )


def batch(
    batch_id: str,
    observed: float | str,
    values: dict[str, float],
    *reasons: str,
    boot: str = BOOT,
) -> dict[str, Any]:
    return {
        "batch_id": batch_id,
        "observed_at": observed if isinstance(observed, str) else iso(observed),
        "edge_filter_reasons": list(reasons or ("environmental-observation",)),
        "collection_errors": [],
        "gpus": [GPU],
        "samples": [
            {
                "gpu": GPU,
                "name": name,
                "value": value,
                "labels": {"source_boot_id": boot},
            }
            for name, value in values.items()
        ],
    }


def latest(name: str, observed: float | str, value: float) -> dict[str, Any]:
    return {
        "gpu": GPU,
        "name": name,
        "value": value,
        "observed_at": observed if isinstance(observed, str) else iso(observed),
        "source": "DCGM_EXPORTER",
        "labels": {"source_boot_id": BOOT},
    }


def finding(
    finding_id: str,
    seconds: float,
    *,
    value: float = SEVERE,
    severity: str = "CRITICAL",
    name: str = FIELD,
    **fields: Any,
) -> dict[str, Any]:
    record = {
        "finding_id": finding_id,
        "canonical_name": name,
        "severity": severity,
        "value": value,
        "gpu_uuid": GPU,
        "observed_at": iso(seconds),
        "automatic_action": "DRAIN" if severity == "CRITICAL" else "RUN_DIAGNOSTICS",
        "policy_source": POLICY,
        "threshold_source": SOURCE,
        "confirmation_count": 3 if severity == "CRITICAL" else 1,
        "workload_state": "IDLE",
        "affected_workload_ids": [],
        "affected_gpu_uuids": [],
    }
    record.update(fields)
    return record


def incident(
    finding_id: str,
    seconds: float,
    *,
    state: str = "ESCALATED",
    workflow: str | None = None,
) -> dict[str, Any]:
    return {
        "incident_id": f"inc-gpu-{finding_id}",
        "event_id": f"gpu-{finding_id}",
        "event_type": "NODE_HEALTH",
        "state": state,
        "effective_action": "DRAIN",
        "gpu_uuids": [GPU],
        "node_ids": [NODE],
        "workflow_request_id": workflow,
        "created_at": iso(seconds),
    }


def workflow(
    request_id: str,
    finding_id: str,
    seconds: float,
    operations: Sequence[str],
    *,
    status: str = "SUCCEEDED",
    outcomes: dict[str, str] | None = None,
    deadline: float = 1800.0,
    **fields: Any,
) -> dict[str, Any]:
    """A workflow of ``inc-gpu-<finding_id>``: every step SUCCEEDED once the
    workflow did, else only the step outcomes given."""

    executed = (
        {name: "SUCCEEDED" for name in operations} if status == "SUCCEEDED" else {}
    )
    executed.update(outcomes or {})
    record = {
        "request_id": request_id,
        "incident_id": f"inc-gpu-{finding_id}",
        "status": status,
        "official_steps": [{"operation": name} for name in operations],
        "step_executions": [
            {
                "operation": name,
                "step_index": index,
                "status": executed[name],
                "updated_at": iso(seconds + 30),
                "error": None if executed[name] == "SUCCEEDED" else f"{name} refused",
            }
            for index, name in enumerate(operations)
            if name in executed
        ],
        "completed_operations": [
            name for name in operations if executed.get(name) == "SUCCEEDED"
        ],
        "execution_deadline": iso(seconds + deadline),
        "created_at": iso(seconds),
    }
    record.update(fields)
    return record


def notification(finding_id: str, status: str | None = "SENT") -> dict[str, Any]:
    return {
        "notification_id": f"n-{finding_id}",
        "incident_id": f"inc-gpu-{finding_id}",
        "category": "SUPPORT_ESCALATION",
        "status": status,
    }


def node_doc(*, held: bool) -> dict[str, Any]:
    return {
        "unschedulable": held,
        "taints": [{"key": QUARANTINE_TAINT, "effect": "NoSchedule"}] if held else [],
        "ownership_annotations": {},
    }


def snapshot(
    seconds: float,
    *,
    gpu: dict[str, Any] | None = None,
    activity: dict[str, Any] | None = None,
    node: dict[str, Any] | None = None,
) -> watch.Snapshot:
    return watch.Snapshot(at=at(seconds), gpu_state=gpu, activity=activity, node=node)


def gpu_doc(
    *,
    findings: Sequence[dict[str, Any]] = (),
    active: Sequence[dict[str, Any]] = (),
    batches: Sequence[dict[str, Any]] = (),
    rows: Sequence[dict[str, Any]] = (),
    statuses: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "findings": list(findings),
        "active_findings": list(active),
        "batches": list(batches),
        "latest": list(rows),
        "collector_statuses": list(statuses),
    }


def activity_doc(
    *,
    incidents: Sequence[dict[str, Any]] = (),
    workflows: Sequence[dict[str, Any]] = (),
    notifications: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "incidents": list(incidents),
        "workflows": list(workflows),
        "notifications": list(notifications),
    }


# --------------------------------------------------------------------------- #
# GF-ENV-LIVE-004 review chain
# --------------------------------------------------------------------------- #
SEVERE_BATCHES = [
    batch(f"b-{index}", 45 + 15 * index, {FIELD: SEVERE, "gpu_temperature_c": 30.0})
    for index in range(3)
]
CLEAR_BATCH = batch(
    "b-clear", 90, {FIELD: HEALTHY, "gpu_temperature_c": 30.0}, "candidate-recovered"
)


def review_scene(**changes: Any) -> dict[str, Any]:
    """The product doing 004 right: confirmed at 75 s, reviewed, notified."""

    scene: dict[str, Any] = {
        "findings": [finding("f-crit", 75)],
        "batches": [*SEVERE_BATCHES, CLEAR_BATCH],
        "incidents": [incident("f-crit", 76, workflow="wf-review")],
        "workflows": [workflow("wf-review", "f-crit", 76, REVIEW_OPERATIONS)],
        "notifications": [notification("f-crit")],
    }
    scene.update(changes)
    return scene


def judge_review(
    scene: dict[str, Any],
    *,
    polls: Sequence[float] = (CONSUMED, HOLD_END + 15),
    held: bool = True,
    final: bool = False,
) -> watch.Judgement | None:
    """The review judge over the same records at every poll, decided at the last."""

    judge = judges.judge_for(expectation("review"), repoint_at=at(-60))
    progress = replay()
    for seconds in polls:
        judge.observe(
            snapshot(
                seconds,
                gpu=gpu_doc(findings=scene["findings"], batches=scene["batches"]),
                activity=activity_doc(
                    incidents=scene["incidents"],
                    workflows=scene["workflows"],
                    notifications=scene["notifications"],
                ),
                node=node_doc(held=held),
            ),
            progress,
        )
    return judge.decide(progress, at(polls[-1]), final=final)


def judge_read_only(
    rule: Expectation,
    *,
    findings: Sequence[dict[str, Any]] = (),
    active: Sequence[dict[str, Any]] = (),
    batches: Sequence[dict[str, Any]] = (),
    rows: Sequence[dict[str, Any]] = (),
    incidents: Sequence[dict[str, Any]] = (),
    workflows: Sequence[dict[str, Any]] = (),
    poll_at: float = 120.0,
    final: bool = False,
) -> tuple[judges.ReadOnlyJudge, watch.Judgement | None]:
    """One poll of a read-only case at ``poll_at``, decided at that moment."""

    judge = judges.ReadOnlyJudge(rule)
    progress = replay()
    judge.observe(
        snapshot(
            poll_at,
            gpu=gpu_doc(findings=findings, active=active, batches=batches, rows=rows),
            activity=activity_doc(incidents=incidents, workflows=workflows),
            node=node_doc(held=False),
        ),
        progress,
    )
    return judge, judge.decide(progress, at(poll_at), final=final)
