"""Keep a reset workflow's terminal record, and observe a restart only after success.

Live 2026-11-03 (GF-REGIONAL-COLLECT-016 D, gpu-eks-cluster, R16): the XID 109
reset of a node running the managed workload ended FAILED at RESET_GPU, the
product's reset rung rebooted the node (``workflow-reboot-after-<id>``), and the
case reported the restart custody's refusal of that FAILED workflow --
"successful workflow or incident identity differs" -- instead of the failed
reset. Nothing on disk said why the reset failed: the terminal store reading
(the step executions, the remote commands with the Node Agent's result details,
the notifications) lived only in memory, and the custody raised before any
host evidence was read.

``settle_reset_terminal`` writes that reading as ``workflow-terminal.json`` the
moment the wait returns, on every path, and observes a post-restart workload
only behind a SUCCEEDED reset workflow. The restart custody adopts nothing else
(``RestartCustody.from_state``), so a failed reset fails the case with its own
failed steps and errors, and the custody is never asked about a restart that
cannot exist. A SUCCEEDED reset is observed exactly as before: the custody
still binds the replacement before the case counts it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.regional_commands import RegionalFixtureError

TERMINAL_RECORD = "workflow-terminal.json"
POST_RESTART_WORKLOAD = "post-restart-workload.json"
#: What the store reading of a finished reset holds that explains it: the
#: workflow with its step executions, the remote commands carrying the Node
#: Agent's result details, the notifications, and the WAITING steps seen.
TERMINAL_RECORD_KEYS = (
    "release_id",
    "event",
    "decision",
    "incident",
    "workflow",
    "commands",
    "notifications",
    "observed_waiting_step_executions",
)
#: Each failed step's error in the case error, bounded; the record has it whole.
FAILURE_TEXT_LIMIT = 400

PostRestartObserver = Callable[[dict[str, Any]], dict[str, Any]]


def terminal_record(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: state.get(key) for key in TERMINAL_RECORD_KEYS}


def failed_steps(workflow: Mapping[str, Any]) -> list[str]:
    """``<operation> (step <index>): <error>`` for every FAILED step execution."""

    executions = workflow.get("step_executions")
    failures = []
    for item in executions if isinstance(executions, list) else []:
        if not isinstance(item, dict) or item.get("status") != "FAILED":
            continue
        error = " ".join(str(item.get("error") or "no error recorded").split())
        failures.append(
            f"{item.get('operation')} (step {item.get('step_index')}): "
            f"{error[:FAILURE_TEXT_LIMIT]}"
        )
    return failures


def reset_failure(state: Mapping[str, Any]) -> str | None:
    """Why the waited-for reset workflow did not succeed; ``None`` when it did."""

    workflow = state.get("workflow")
    workflow = workflow if isinstance(workflow, dict) else {}
    status = workflow.get("status")
    if status == "SUCCEEDED":
        return None
    reasons = failed_steps(workflow)
    terminal = " ".join(str(workflow.get("terminal_failure_reason") or "").split())
    if terminal and not any(terminal[:80] in reason for reason in reasons):
        reasons.append(f"terminal failure: {terminal[:FAILURE_TEXT_LIMIT]}")
    return (
        f"reset workflow {workflow.get('request_id')!r} ended {status!r}"
        + (": " + "; ".join(reasons) if reasons else "")
        + f" ({TERMINAL_RECORD})"
    )


def settle_reset_terminal(
    case_dir: Path,
    state: dict[str, Any],
    observe: PostRestartObserver | None,
) -> dict[str, Any] | None:
    """Write the terminal record; with ``observe``, the post-restart proof.

    The record is written first and unconditionally. ``observe`` runs only on a
    SUCCEEDED reset; its proof must be a populated mapping, is kept in
    ``state["post_restart_workload"]`` and written beside the record.
    """

    write_json_atomic(case_dir / TERMINAL_RECORD, terminal_record(state))
    if observe is None:
        return None
    failure = reset_failure(state)
    if failure is not None:
        raise RegionalFixtureError(
            f"{failure}; the post-restart workload is not observed: the restart "
            "custody adopts only the restart of a SUCCEEDED workflow"
        )
    proof = observe(state)
    if not isinstance(proof, dict) or not proof:
        raise RegionalFixtureError("post-restart workload proof is missing")
    state["post_restart_workload"] = proof
    write_json_atomic(case_dir / POST_RESTART_WORKLOAD, proof)
    return proof
