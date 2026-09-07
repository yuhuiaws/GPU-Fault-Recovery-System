"""Pure verdict functions and constants of GF-REGIONAL-CMD-018.

The case proves ARCH-D5 on a real regional deployment: a remote command's id is
a digest of the whole step, so a merge that rewrites a step's parameters gives
the next dispatch a *different* id while the command minted for the previous
step may still be executing on the node. The adapter must therefore hold the
second dispatch WAITING with ``reason=OPEN_SIBLING_COMMAND`` while a sibling is
PENDING/LEASED/WAITING under the same fencing token, count it on
``open_sibling_holds_total``, and mint nothing -- so the node executes the
action exactly once. Once the sibling is terminal the hold lifts.

The command belongs to the synthetic cluster ``perf-cap-000`` and the only
executor that owns its step is a probe whose "action" is a local ledger; no
node can receive it. Every function here judges documents the runner wrote.
"""

from __future__ import annotations

from typing import Any

CASE_ID = "GF-REGIONAL-CMD-018"
CONFIRMATION = "CMD018_EXECUTE"
PREDECESSOR_CASE_ID = "GF-REGIONAL-CMD-017"
RUN_PREFIX = "cmd018-"
OWNER = "gpu-fault-cmd018-sibling"
POD = "gpu-fault-cmd018-executor"
CONFIGMAP = "gpu-fault-cmd018-script"
OPERATION = "TRIGGER_HEALTH_SNAPSHOT"
NODE_IDS = ["cmd018-synthetic-node"]
HOLD_REASON = "OPEN_SIBLING_COMMAND"
HOLDS_METRIC = "gpu_fault_remote_command_open_sibling_holds_total"
POD_DEADLINE_SECONDS = 600
COMPLETION_TIMEOUT_SECONDS = 180


def dispatch_errors(dispatch: dict[str, Any]) -> list[str]:
    """The in-Pod dispatch script's record of the hold, before any executor ran."""

    errors: list[str] = []
    first = dispatch.get("first") or {}
    held = dispatch.get("held") or {}
    first_id = str(first.get("remote_command_id") or "")
    if first.get("status") != "WAITING" or not first_id:
        errors.append(f"the first dispatch did not mint a PENDING command: {first}")
    if held.get("status") != "WAITING":
        errors.append(
            f"the rewritten dispatch is {held.get('status')!r}, expected WAITING"
        )
    details = held.get("details") or {}
    if details.get("reason") != HOLD_REASON:
        errors.append(
            f"hold reason is {details.get('reason')!r}, expected {HOLD_REASON!r}"
        )
    if details.get("remote_command_id") != first_id:
        errors.append("the hold does not point at the open sibling's command id")
    held_id = str(details.get("held_command_id") or "")
    if not held_id or held_id == first_id:
        errors.append(
            "the rewritten step did not produce a different command id; the hold "
            "proves nothing"
        )
    if details.get("mutation_submitted_by_control_plane") is not False:
        errors.append("the hold claims the control plane submitted a mutation")
    if int(dispatch.get("open_sibling_holds_total") or 0) != 1:
        errors.append(
            f"open_sibling_holds_total is {dispatch.get('open_sibling_holds_total')!r}, "
            "expected 1"
        )
    open_ids = sorted(str(item) for item in dispatch.get("open_command_ids") or [])
    if open_ids != [first_id]:
        errors.append(
            f"open commands after the hold are {open_ids}, expected [{first_id}]"
        )
    if dispatch.get("registered_agents"):
        errors.append(
            "the synthetic cluster has registered Node Agents; the hard stop does not hold"
        )
    return errors


def executed_once_errors(
    ledger: dict[str, Any], command: dict[str, Any], *, first_id: str
) -> list[str]:
    """The probe executed the open sibling exactly once and it reached a terminal state."""

    errors: list[str] = []
    if int(ledger.get("physical_count") or 0) != 1:
        errors.append(
            f"the ledger holds {ledger.get('physical_count')} attempts, expected 1"
        )
    keys = [str(item) for item in ledger.get("keys") or []]
    if len(keys) != 1:
        errors.append(f"the ledger holds {len(keys)} idempotency keys, expected 1")
    if command.get("command_id") != first_id:
        errors.append("the completed command is not the first sibling")
    if command.get("status") != "SUCCEEDED":
        errors.append(
            f"the first sibling is {command.get('status')!r}, expected SUCCEEDED"
        )
    return errors


def release_errors(release: dict[str, Any], *, first_id: str) -> list[str]:
    """With the sibling terminal, the rewritten step mints its own command."""

    errors: list[str] = []
    outcome = release.get("outcome") or {}
    details = outcome.get("details") or {}
    if details.get("reason") == HOLD_REASON:
        errors.append(
            "the dispatch is still held after the sibling reached a terminal state"
        )
    minted = str(details.get("remote_command_id") or "")
    if not minted or minted == first_id:
        errors.append(f"no new command was minted after the release: {details}")
    if release.get("cancelled") is not True:
        errors.append("the newly minted command was not cancelled before purge")
    cancelled = release.get("cancelled_command") or {}
    if cancelled.get("status") != "FAILED":
        errors.append(
            f"the cancelled command is {cancelled.get('status')!r}, expected FAILED"
        )
    return errors


def metric_family_errors(texts: list[str]) -> list[str]:
    if not any(HOLDS_METRIC in text for text in texts):
        return [f"{HOLDS_METRIC} is not exported by any control-plane replica"]
    return []


def case_verdict(stages: dict[str, list[str]]) -> str:
    return "PASS" if not any(stages.values()) else "FAIL"
