"""What a FAILED RESTART_FABRIC_MANAGER result of the Node Agent says about the unit.

The agent's restart (``node_agent/operations/reset._restart_fabric_manager``)
checks that its configuration allows it, that no compute client holds the GPUs,
reads the unit, passes the final ownership checkpoint and restarts the unit with
``systemctl``; its executor frames every failure ``<exception class>:
<message>``. The escalation ladder climbs to the reboot only on a unit that
failed (``failure_outcome.fabric_manager_unit_failed``), so the fold records
which of the agent's verdicts a FAILED restart is
(``failure_outcome.FABRIC_MANAGER_RESTART_OUTCOME_DETAIL``). The configuration
refusal and a failed ``systemctl`` call carry no structured detail of their
own, so their wording is the contract -- like the ``clients are still active``
wording the client verification's retry fold matches -- and it is pinned
against the real handler (``tests/execution/test_fabric_manager_restart_outcomes.py``).
A wording this module does not know reads ``NO_UNIT_VERDICT``: no reboot.
"""

from __future__ import annotations

from typing import Any

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.protocol import NodeActionResult, NodeActionStatus
from gpu_fault.orchestration.failure_outcome import (
    FABRIC_MANAGER_RESTART_OUTCOME_DETAIL,
    FM_RESTART_NO_UNIT_VERDICT,
    FM_RESTART_REFUSED_BY_NODE_CONFIGURATION,
    FM_RESTART_REFUSED_GPU_CLIENTS,
    FM_RESTART_UNIT_FAILED,
    gpu_clients_refused,
    unknown_outcome_failure,
)

#: The agent's refusal when ``GPU_FAULT_NODE_ALLOW_FABRIC_MANAGER_RESTART`` is
#: not enabled on the node, as its executor frames it.
NODE_CONFIGURATION_REFUSAL = (
    "RuntimeError: Fabric Manager restart is disabled by node configuration"
)
#: A ``systemctl`` call that exited non-zero (``operations.diagnostics
#: ._run_checked``: ``<argv0> exited with status <code>[: <output tail>]``).
UNIT_CALL_FAILURE_PREFIX = "RuntimeError: systemctl exited with status "


def fabric_manager_restart_outcome(result: NodeActionResult) -> str | None:
    """The outcome of a FAILED RESTART_FABRIC_MANAGER the agent answered, or
    ``None``: another operation or status, or an unknown outcome -- the agent's
    safety refusals and interrupted or abandoned attempts carry their own
    flags, which both ladders judge first. A retryable failure whose
    re-submits ran out (``node_action_retry_exhausted``) may be a restart
    killed at its deadline whose systemd job is still running: never a unit
    failure."""

    if (
        result.operation is not WorkflowOperation.RESTART_FABRIC_MANAGER
        or result.status is not NodeActionStatus.FAILED
        or unknown_outcome_failure(result.details)
    ):
        return None
    if gpu_clients_refused(result.details):
        return FM_RESTART_REFUSED_GPU_CLIENTS
    error = result.error or ""
    if error == NODE_CONFIGURATION_REFUSAL:
        return FM_RESTART_REFUSED_BY_NODE_CONFIGURATION
    if (
        not result.retryable
        and result.details.get("node_action_retry_exhausted") is not True
        and error.startswith(UNIT_CALL_FAILURE_PREFIX)
    ):
        return FM_RESTART_UNIT_FAILED
    return FM_RESTART_NO_UNIT_VERDICT


def fabric_manager_restart_evidence(result: NodeActionResult) -> dict[str, Any]:
    """The step detail :func:`fabric_manager_restart_outcome` gives, or ``{}``."""

    outcome = fabric_manager_restart_outcome(result)
    return {} if outcome is None else {FABRIC_MANAGER_RESTART_OUTCOME_DETAIL: outcome}
