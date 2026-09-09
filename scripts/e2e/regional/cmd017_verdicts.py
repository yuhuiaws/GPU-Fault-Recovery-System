"""Pure verdict functions and constants of GF-REGIONAL-CMD-017.

The case proves ARCH-C6 on a real regional deployment: a remote command whose
step is a multi-node barrier operation is held at the executor's claim boundary
with a reason and a counter, because the regional executor has no barrier
coordinator and must not let ``_execute_multi_node_reset`` answer with a bare
FAILED that reads like the reset itself failed -- or, worse, let a reset arm.

The natural injection (a two-node SXID that compiles to a multi-node
``RESET_ALL_GPUS_NVSWITCHES``) is deliberately *not* used: reaching the barrier
step on real nodes means passing ``VERIFY_NO_GPU_CLIENTS`` on both, at which
point a regressed claim boundary would arm two real full-fabric resets
(COLLECT-014 did exactly that once when inventory lagged). Seeding the barrier
command directly for a synthetic cluster with no Node Agents exercises the same
deployed code path with no reset reachable from it.
"""

from __future__ import annotations

from typing import Any

CASE_ID = "GF-REGIONAL-CMD-017"
CONFIRMATION = "CMD017_LIVE_BARRIER_HOLD"
RUN_PREFIX = "cmd017-"
OWNER = "gpu-fault-cmd017-barrier"
POD = "gpu-fault-cmd017-executor"
CONFIGMAP = "gpu-fault-cmd017-script"
OPERATION = "RESET_ALL_GPUS_NVSWITCHES"
NODE_IDS = ["cmd017-synthetic-node-a", "cmd017-synthetic-node-b"]
STATUS_SOURCE = "executor-barrier-unavailable"
HOLD_LOG = "multi-node barrier unavailable"
POD_DEADLINE_SECONDS = 600
# How many claim/hold rounds the runner waits for; two prove the command stays
# re-claimable and that every round counts, not just the first.
MINIMUM_HOLDS = 2
HOLD_TIMEOUT_SECONDS = 180


def seed_errors(seed: dict[str, Any]) -> list[str]:
    """The seed must be a two-node barrier step for a cluster nobody serves."""

    errors: list[str] = []
    if seed.get("operation") != OPERATION:
        errors.append(f"seeded operation {seed.get('operation')!r} != {OPERATION!r}")
    node_ids = list(seed.get("node_ids") or [])
    if len(node_ids) < 2:
        errors.append(f"seeded step names {len(node_ids)} node(s); a barrier needs two")
    if seed.get("registered_agents"):
        errors.append(
            "the synthetic cluster has registered Node Agents; the hard stop does "
            f"not hold: {seed['registered_agents']}"
        )
    return errors


def ready_errors(ready: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if ready.get("owner") != OWNER:
        errors.append(f"probe owner {ready.get('owner')!r} != {OWNER!r}")
    if ready.get("adapter_has_barrier_coordinator") is not False:
        errors.append("the stand-in adapter claims a barrier coordinator")
    return errors


def held_command_errors(
    command: dict[str, Any],
    *,
    node_ids: list[str],
    executor_id: str,
) -> list[str]:
    """The control plane's record of the hold, read while the probe runs."""

    errors: list[str] = []
    if command.get("status") not in {"WAITING", "PENDING", "LEASED"}:
        errors.append(
            f"held command is {command.get('status')}; a barrier hold must not "
            "reach a terminal status"
        )
    if command.get("status_source") != STATUS_SOURCE:
        errors.append(
            f"status_source {command.get('status_source')!r} != {STATUS_SOURCE!r}"
        )
    details = command.get("result_details") or {}
    if details.get("multi_node_barrier_unavailable") is not True:
        errors.append("result_details.multi_node_barrier_unavailable is not True")
    if details.get("operation") != OPERATION:
        errors.append(
            f"result_details.operation {details.get('operation')!r} != {OPERATION!r}"
        )
    if sorted(details.get("node_ids") or []) != sorted(node_ids):
        errors.append(
            f"result_details.node_ids {details.get('node_ids')!r} != {sorted(node_ids)!r}"
        )
    if details.get("executor_id") != executor_id:
        errors.append(
            f"result_details.executor_id {details.get('executor_id')!r} != {executor_id!r}"
        )
    reason = str(details.get("reason") or "")
    if "barrier coordinator" not in reason:
        errors.append(f"hold reason does not name the missing coordinator: {reason!r}")
    if command.get("error"):
        errors.append(f"held command carries an error: {command.get('error')!r}")
    return errors


# The probe's state recorder rewrites executor-state.json once a second; a
# snapshot the breadcrumb is compared with must be taken at least this long
# after the breadcrumb was read.
STATE_RECORDER_SETTLE_SECONDS = 2.0


def executor_state_errors(
    state: dict[str, Any], *, minimum_holds: int = MINIMUM_HOLDS
) -> list[str]:
    errors: list[str] = []
    holds = int(state.get("barrier_unavailable_holds_total") or 0)
    if holds < minimum_holds:
        errors.append(
            f"barrier_unavailable_holds_total is {holds}, below the {minimum_holds} "
            "claim rounds the case waited for"
        )
    claimed = int(state.get("claimed_total") or 0)
    if claimed < holds:
        errors.append(f"claimed_total {claimed} is below the hold count {holds}")
    if int(state.get("unexpected_failures") or 0):
        errors.append("executor recorded an unexpected failure")
    if int(state.get("reported_failures") or 0):
        errors.append("the control plane rejected a hold result")
    if state.get("adapter_executed") is not False:
        errors.append(
            "the stand-in adapter was executed; the claim boundary did not hold "
            "the barrier step"
        )
    return errors


def breadcrumb_errors(claim_state: dict[str, Any], state: dict[str, Any]) -> list[str]:
    recorded = claim_state.get("counters")
    if not isinstance(recorded, dict):
        return ["claim-state breadcrumb carries no counters"]
    errors: list[str] = []
    for key in ("barrier_unavailable_holds_total", "claimed_total"):
        if int(recorded.get(key) or 0) > int(state.get(key) or 0):
            errors.append(f"breadcrumb {key} runs ahead of the executor snapshot")
    if int(recorded.get("barrier_unavailable_holds_total") or 0) < 1:
        errors.append("claim-state breadcrumb never recorded a barrier hold")
    return errors


def log_errors(logs: str) -> list[str]:
    if HOLD_LOG not in logs:
        return [f"executor log has no {HOLD_LOG!r} line"]
    return []


def adapter_marker_errors(marker: dict[str, Any] | None) -> list[str]:
    if marker:
        return [
            "the stand-in adapter recorded an execution: "
            f"{marker.get('operation')} on {marker.get('node_ids')}"
        ]
    return []


def final_command_errors(command: dict[str, Any]) -> list[str]:
    """After the probe stopped, the command is still open for purge."""

    if command.get("status") in {"SUCCEEDED", "FAILED"}:
        return [
            f"the barrier command reached {command.get('status')} without a "
            "coordinator; the hold was not fail-closed"
        ]
    return []
