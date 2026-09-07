"""Pure verdict functions and constants of GF-REGIONAL-NET-006.

The case proves ARCH-C7/C8 on a real regional deployment: an executor that
loses its command lease while an action is still running must not post the
result under that lease, must count what it withheld, and must expose those
counters in the claim-state breadcrumb. NET-002 proves the complementary half
(a result posted after expiry is refused with 409); here the executor never
gets that far, and the *absence* of the 409 is part of the verdict.

Every function below is judged against evidence documents the runner wrote --
the probe's ready/state files, store snapshots of the seeded command, the Pod
log -- and touches no cluster.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

CASE_ID = "GF-REGIONAL-NET-006"
CONFIRMATION = "NET006_LIVE_LEASE_LOSS_WITHHELD_RESULT"
RUN_PREFIX = "net006-"
OWNER = "gpu-fault-net006-test"
POD = "gpu-fault-net006-executor"
CONFIGMAP = "gpu-fault-net006-script"
OPERATION = "FREEZE_EVIDENCE"
NODE_IDS = ["net006-synthetic-node"]

# Timing. The block must outlast the lease and the hold; the hold must outlast
# the lease so the watch has declared it lost when the action returns; the
# renewal cadence (lease/3, capped at 30s) times the failure limit must fit
# inside the hold so the loss is declared by *failures*, not only by the local
# clock.
BLOCK_SECONDS = 120
BLOCK_ROLLBACK_SECONDS = 150
HTTP_TIMEOUT_SECONDS = 15
ACTION_HOLD_SECONDS = 100
LEASE_SECONDS = 60
LEASE_FAILURE_LIMIT = 3
POD_DEADLINE_SECONDS = 900
RECLAIM_TIMEOUT_SECONDS = 240

WITHHELD_LOG = "withheld a result under a lost lease"
LOST_LOG = "regional command lease treated as lost"
STALE_LEASE_409 = "rejected request (409)"
COUNTER_KEYS = (
    "claimed_total",
    "reported_failures",
    "unexpected_failures",
    "lease_renewal_failures",
    "lease_lost_total",
    "results_withheld_total",
    "cancellations_observed_total",
    "barrier_unavailable_holds_total",
)


def renewal_interval_seconds(lease_seconds: int) -> float:
    """``ClusterActionExecutor._renew_lease``'s cadence for one lease length."""

    return min(30.0, lease_seconds / 3)


def timing_errors(
    *,
    block_seconds: int = BLOCK_SECONDS,
    rollback_seconds: int = BLOCK_ROLLBACK_SECONDS,
    hold_seconds: int = ACTION_HOLD_SECONDS,
    lease_seconds: int = LEASE_SECONDS,
    failure_limit: int = LEASE_FAILURE_LIMIT,
    http_timeout_seconds: float = HTTP_TIMEOUT_SECONDS,
) -> list[str]:
    """Refuse a parameter set that could pass for the wrong reason."""

    errors: list[str] = []
    if hold_seconds <= lease_seconds:
        errors.append(
            f"action hold {hold_seconds}s does not outlast the lease "
            f"{lease_seconds}s; the watch could still vouch for the lease"
        )
    failures_take = renewal_interval_seconds(lease_seconds) * failure_limit
    if failures_take >= hold_seconds:
        errors.append(
            f"{failure_limit} renewal failures at {renewal_interval_seconds(lease_seconds):.1f}s "
            f"take {failures_take:.1f}s, not inside the {hold_seconds}s hold; the loss "
            "would be declared by the local clock alone"
        )
    if block_seconds <= hold_seconds:
        errors.append(
            f"block {block_seconds}s must outlast the hold {hold_seconds}s so the "
            "action returns while the control plane is still unreachable"
        )
    if rollback_seconds <= block_seconds:
        errors.append(
            f"automatic rollback {rollback_seconds}s must be after the intended "
            f"block {block_seconds}s"
        )
    if http_timeout_seconds >= renewal_interval_seconds(lease_seconds):
        errors.append(
            f"HTTP timeout {http_timeout_seconds}s is not below the renewal "
            f"interval {renewal_interval_seconds(lease_seconds):.1f}s; a hanging "
            "renewal would hide the next one"
        )
    return errors


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def ready_errors(ready: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if ready.get("proxy_mode") != "refuse-while-blocked":
        errors.append("probe proxy does not refuse while blocked")
    if ready.get("action_requires_network_block") is not True:
        errors.append("simulated action is not ordered after the network block")
    if ready.get("owner") != OWNER:
        errors.append(f"probe owner {ready.get('owner')!r} != {OWNER!r}")
    errors.extend(
        timing_errors(
            block_seconds=BLOCK_SECONDS,
            rollback_seconds=int(ready.get("block_rollback_seconds") or 0),
            hold_seconds=int(ready.get("action_hold_seconds") or 0),
            lease_seconds=int(ready.get("lease_seconds") or 0),
            failure_limit=int(ready.get("lease_failure_limit") or 0),
            http_timeout_seconds=float(ready.get("http_timeout_seconds") or 0),
        )
    )
    return errors


def counters(document: dict[str, Any]) -> dict[str, int]:
    return {key: int(document.get(key) or 0) for key in COUNTER_KEYS}


def executor_state_errors(
    state: dict[str, Any],
    *,
    failure_limit: int = LEASE_FAILURE_LIMIT,
) -> list[str]:
    """The counters the withheld result must leave behind (C7/C8)."""

    values = counters(state)
    errors: list[str] = []
    if values["claimed_total"] != 2:
        errors.append(
            f"claimed_total is {values['claimed_total']}, not 2 (first claim plus "
            "one reclaim after the lease was lost)"
        )
    if values["results_withheld_total"] != 1:
        errors.append(
            f"results_withheld_total is {values['results_withheld_total']}, not 1"
        )
    if values["lease_lost_total"] < 1:
        errors.append("lease_lost_total never moved; the watch never declared the loss")
    if values["lease_renewal_failures"] < failure_limit:
        errors.append(
            f"lease_renewal_failures {values['lease_renewal_failures']} is below "
            f"the failure limit {failure_limit}; the block did not fail renewals"
        )
    if values["reported_failures"] != 0:
        errors.append(
            f"reported_failures is {values['reported_failures']}; a result was "
            "posted under the lost lease and rejected instead of withheld"
        )
    if values["unexpected_failures"] != 0:
        errors.append("executor recorded an unexpected failure")
    if values["cancellations_observed_total"] != 0:
        errors.append("a cancellation was observed; nothing cancelled this command")
    if not state.get("last_successful_claim_at"):
        errors.append("executor state carries no last_successful_claim_at")
    return errors


def breadcrumb_errors(claim_state: dict[str, Any], state: dict[str, Any]) -> list[str]:
    """The claim-state breadcrumb carries the same counters (C8)."""

    errors: list[str] = []
    recorded = claim_state.get("counters")
    if not isinstance(recorded, dict):
        errors.append("claim-state breadcrumb carries no counters")
        return errors
    if counters(recorded) != counters(state):
        errors.append(
            "claim-state breadcrumb counters differ from the executor's final "
            f"snapshot: {counters(recorded)} != {counters(state)}"
        )
    if claim_state.get("executor_id") != state.get(
        "executor_id", claim_state.get("executor_id")
    ):
        errors.append("claim-state breadcrumb names another executor")
    return errors


def lease_guard_errors(observed: dict[str, Any]) -> list[str]:
    """The node-action guard on the executing thread saw the lost lease."""

    reason = observed.get("reason")
    if not isinstance(reason, str) or not reason:
        return [
            "the lease guard reported no hold reason at the end of the action; a "
            "node action started at that moment would not have been refused"
        ]
    if "lease renewal failed" not in reason and "lease expired locally" not in reason:
        return [f"lease guard reason is not a lost-lease reason: {reason!r}"]
    return []


def log_errors(logs: str) -> list[str]:
    errors: list[str] = []
    if WITHHELD_LOG not in logs:
        errors.append(f"executor log has no {WITHHELD_LOG!r} line")
    if LOST_LOG not in logs:
        errors.append(f"executor log has no {LOST_LOG!r} line")
    if STALE_LEASE_409 in logs:
        errors.append(
            "executor log shows a 409 on result submission; the result was posted "
            "under the lost lease rather than withheld"
        )
    return errors


def command_errors(
    *,
    leased: dict[str, Any],
    withheld: dict[str, Any],
    final: dict[str, Any],
    unblocked_at: datetime,
    executor_id: str,
) -> list[str]:
    """The seeded command's journey: leased, still leased at the withhold,
    reclaimed by the same executor id, finished from the ledger."""

    errors: list[str] = []
    if leased.get("status") != "LEASED" or not leased.get("lease_expires_at"):
        errors.append(f"command was not actively leased before the block: {leased}")
    expires = parse_time(leased.get("lease_expires_at"))
    if expires is not None and expires >= unblocked_at:
        errors.append("command lease had not expired before the block was lifted")
    if withheld.get("status") not in {"LEASED", "PENDING"}:
        errors.append(
            f"command left LEASED/PENDING while the result was withheld: "
            f"{withheld.get('status')}"
        )
    if withheld.get("status_source") not in {None, ""} and withheld.get("status") in {
        "SUCCEEDED",
        "FAILED",
    }:
        errors.append("command reached a terminal status during the block")
    if final.get("status") != "SUCCEEDED":
        errors.append(f"command did not finish SUCCEEDED: {final.get('status')}")
    if (final.get("result_details") or {}).get("cached") is not True:
        errors.append("the reclaim did not finish from the idempotency ledger")
    if final.get("last_lease_owner") != executor_id:
        errors.append(
            f"final lease owner {final.get('last_lease_owner')!r} != {executor_id!r}"
        )
    return errors


def ledger_errors(ledger: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if ledger.get("physical_count") != 1:
        errors.append("physical action count is not one")
    if len(ledger.get("keys") or []) != 1:
        errors.append("idempotency ledger does not contain exactly one key")
    return errors


def interruption_errors(
    *,
    blocked_seconds: float,
    action_gate: dict[str, Any],
    action_returned: dict[str, Any],
    rollback: dict[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    if blocked_seconds < BLOCK_SECONDS:
        errors.append(f"network interruption ended before {BLOCK_SECONDS} seconds")
    if blocked_seconds >= BLOCK_ROLLBACK_SECONDS:
        errors.append("network interruption exceeded the automatic rollback bound")
    if rollback and rollback.get("automatic"):
        errors.append("the automatic rollback fired; the runner did not lift the block")
    gate_at = action_gate.get("observed_at_epoch")
    returned_at = action_returned.get("observed_at_epoch")
    if not isinstance(gate_at, (int, float)) or not isinstance(
        returned_at, (int, float)
    ):
        errors.append("action gate or return timestamps are missing")
        return errors
    held = float(returned_at) - float(gate_at)
    if held < LEASE_SECONDS:
        errors.append(
            f"the action was held {held:.1f}s, not past the {LEASE_SECONDS}s lease"
        )
    if action_returned.get("cached") is not False:
        errors.append("the first execution reported cached=True; it ran nothing")
    return errors
