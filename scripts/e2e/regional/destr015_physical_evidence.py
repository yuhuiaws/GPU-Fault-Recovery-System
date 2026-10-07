"""Identity and conservative clock bounds for actual reset-process overlap."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied

MAX_CLOCK_DRIFT_PPM = 200
CLOCK_MARGIN_NS = 50_000_000


class ResetIntervalScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    run_id: str = Field(min_length=1)
    cluster_id: str = Field(min_length=1)
    release_id: str = Field(min_length=1)
    node: str = Field(min_length=1)
    node_uid: str = Field(min_length=1)
    boot_id: str = Field(min_length=1)
    agent_generation: int = Field(gt=0)
    gpu_uuid: str = Field(pattern=r"^GPU-[A-Za-z0-9-]+$")
    maintenance_end: datetime

    def digest(self) -> str:
        return evidence_digest(self.model_dump(mode="json"))


def evidence_digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _integer(value: Any) -> int:
    if type(value) is not int:
        raise BoundaryDenied("physical interval contains a non-integer clock")
    return value


def guaranteed_interval(
    capture: dict[str, Any],
    *,
    scope: ResetIntervalScope,
    workflow: dict[str, Any],
    host: dict[str, Any],
    baseline: dict[str, Any],
) -> tuple[int, int]:
    """The inner interval guaranteed on the controller's monotonic clock.

    Remote exec/exit timestamps are converted through the witness's continuously
    sampled realtime/monotonic envelope, then through bounded RPC clock exchanges.
    Whole-command or branch timestamps never supply either physical endpoint.
    """
    start, end = capture["start"], capture["end"]
    identity = (
        "scope_sha256",
        "witness_id",
        "tracee",
        "executable_path",
        "executable_sha256",
    )
    if (
        start["scope_sha256"] != scope.digest()
        or any(start.get(key) != end.get(key) for key in identity)
        or end["start_sha256"] != evidence_digest(start)
        or end["trace_complete"] is not True
        or end["closed"] is not True
        or end["lost_events"] != 0
        or _integer(end["trace_bytes"]) <= 0
        or _integer(end["calibration_execs"]) < 1
        or start["tracee"]["boot_id"] != scope.boot_id
        or host["boot_id"] != scope.boot_id
        or baseline["boot_id"] != scope.boot_id
    ):
        raise BoundaryDenied("physical interval witness identity or continuity differs")
    actions = end["actions"]
    if not isinstance(actions, list) or not actions:
        raise BoundaryDenied("physical interval requires a reset invocation")
    for item in actions:
        if (
            item["operation"] != "RESET_GPU"
            or item["gpu_uuid"] != scope.gpu_uuid
            or _integer(item["pid"]) <= 0
        ):
            raise BoundaryDenied("physical reset invocation or target differs")
    applied = [item for item in actions if item["returncode"] == 0]
    if len(applied) != 1:
        raise BoundaryDenied(
            "physical interval requires exactly one applied reset invocation"
        )
    action = applied[0]
    began, ended = _integer(action["started_ns"]), _integer(action["ended_ns"])
    refused = [item for item in actions if item["returncode"] != 0]
    if refused:
        _require_refusals_accounted(
            refused, workflow=workflow, node=scope.node, began=began
        )
    first_began = min(_integer(item["started_ns"]) for item in actions)
    last_ended = max(_integer(item["ended_ns"]) for item in actions)
    before_ids = {
        (row.get("command_id"), row.get("attempt")) for row in baseline["ledger"]
    }
    rows = [
        row
        for row in host["ledger"]
        if row.get("operation") == "RESET_GPU"
        and (row.get("command_id"), row.get("attempt")) not in before_ids
    ]
    if len(rows) != 1:
        raise BoundaryDenied("physical reset has no unique new Agent command")
    row = rows[0]
    if (
        row.get("state") != "SUCCEEDED"
        or row.get("workflow_request_id") != workflow.get("request_id")
        or row.get("incident_id") != workflow.get("incident_id")
        or not workflow.get("request_id")
        or not workflow.get("incident_id")
        or row.get("fencing_token") != workflow.get("fencing_token")
        or type(row.get("fencing_token")) is not int
        or row.get("agent_generation") != scope.agent_generation
        or row.get("gpu_uuids") != [scope.gpu_uuid]
    ):
        raise BoundaryDenied("physical reset command is not bound to this workflow")
    command_times = [
        datetime.fromisoformat(str(row[key]).replace("Z", "+00:00"))
        for key in ("started_at", "completed_at")
    ]
    if (
        any(stamp.tzinfo is None for stamp in command_times)
        or not began < ended
        or not (
            int(command_times[0].timestamp() * 1e9)
            <= first_began
            <= last_ended
            <= int(command_times[1].timestamp() * 1e9)
        )
    ):
        raise BoundaryDenied("physical reset escaped its Agent command interval")
    clocks = capture["clock_exchanges"]
    if not isinstance(clocks, list) or len(clocks) < 2:
        raise BoundaryDenied("physical interval clock calibration is incomplete")
    span = _integer(clocks[-1]["received_ns"]) - _integer(clocks[0]["sent_ns"])
    if span <= 0:
        raise BoundaryDenied("physical interval controller clock is reversed")
    drift = span * MAX_CLOCK_DRIFT_PPM // 1_000_000 + CLOCK_MARGIN_NS
    lower, upper = [], []
    previous = -1
    for clock in clocks:
        sent, received, remote = (
            _integer(clock[key]) for key in ("sent_ns", "received_ns", "monotonic_ns")
        )
        if sent < previous or received < sent or remote <= 0:
            raise BoundaryDenied("physical interval clock exchange is unordered")
        previous = received
        lower.append(sent - remote - drift)
        upper.append(received - remote + drift)
    offset_min, offset_max = max(lower), min(upper)
    wall_min = _integer(end["wall_minus_monotonic_min_ns"])
    wall_max = _integer(end["wall_minus_monotonic_max_ns"])
    if offset_min > offset_max or wall_min > wall_max:
        raise BoundaryDenied("physical interval clocks have no bounded mapping")
    if (
        began - wall_max < clocks[0]["monotonic_ns"]
        or ended - wall_min > clocks[-1]["monotonic_ns"]
    ):
        raise BoundaryDenied("physical reset escaped its calibrated capture window")
    inner = (began - wall_min + offset_max, ended - wall_max + offset_min)
    if inner[0] >= inner[1]:
        raise BoundaryDenied("clock uncertainty consumes the physical reset interval")
    return inner


def reset_receipt_calls(
    workflow: dict[str, Any], *, node: str
) -> list[dict[str, Any]] | None:
    """The Agent's ``physical_calls`` receipt for ``node``'s RESET_GPU step.

    ``None`` when the workflow record carries no such receipt for the node.
    A SUCCEEDED execution wins over earlier attempts; otherwise the last one.
    """

    found: list[dict[str, Any]] | None = None
    for step in workflow.get("step_executions") or []:
        if not isinstance(step, dict) or step.get("operation") != "RESET_GPU":
            continue
        results = (step.get("details") or {}).get("node_results") or {}
        calls = (results.get(node) or {}).get("physical_calls")
        if not isinstance(calls, list):
            continue
        if found is None or step.get("status") == "SUCCEEDED":
            found = calls
    return found


def _require_refusals_accounted(
    refused: list[dict[str, Any]],
    *,
    workflow: dict[str, Any],
    node: str,
    began: int,
) -> None:
    """Refused ``--gpu-reset`` invocations are not physical resets.

    On p5en nodes the driver routinely answers the first ``nvidia-smi
    --gpu-reset`` busy (exit 255); the Agent unloads the display modules and
    its attempt 2 performs the reset, so the witness sees two invocations in
    the window although exactly one reset was applied (live, eks-managed
    attempt 11). A refused invocation is accepted only when the Agent's own
    receipt for this node names exactly that many ``busy`` attempts and one
    applied ``reset``, and it ended before the applied reset began -- so it
    can neither be a second reset nor overlap the interval being proven.
    """

    calls = reset_receipt_calls(workflow, node=node)
    if calls is None:
        raise BoundaryDenied(
            "refused reset invocations are not accounted for by an Agent receipt"
        )
    busy = [
        item
        for item in calls
        if item.get("outcome") == "busy" and "--gpu-reset" in str(item.get("command"))
    ]
    resets = [item for item in calls if item.get("outcome") == "reset"]
    if len(busy) != len(refused) or len(resets) != 1:
        raise BoundaryDenied(
            "refused reset invocations differ from the Agent receipt's busy attempts"
        )
    for item in refused:
        if _integer(item["ended_ns"]) > began:
            raise BoundaryDenied(
                "a refused reset invocation overlaps the applied reset"
            )


def physical_overlap_errors(
    captures: dict[str, dict[str, Any]],
    *,
    scopes: dict[str, ResetIntervalScope],
    workflow: dict[str, Any],
    hosts: dict[str, dict[str, dict[str, Any]]],
) -> list[str]:
    if len(scopes) != 2 or set(captures) != set(scopes) or set(hosts) != set(scopes):
        return ["physical reset witnesses do not cover exactly the two approved nodes"]
    try:
        intervals = [
            guaranteed_interval(
                captures[node],
                scope=scope,
                workflow=workflow,
                host=hosts[node]["after"],
                baseline=hosts[node]["before"],
            )
            for node, scope in scopes.items()
        ]
    except (BoundaryDenied, KeyError, TypeError, ValueError, OverflowError) as exc:
        return [f"physical reset overlap is unproven: {type(exc).__name__}: {exc}"]
    if min(value[1] for value in intervals) <= max(value[0] for value in intervals):
        return ["the actual reset process intervals do not prove parallel execution"]
    return []
