"""Report driver / firmware / EFA-driver install steps that are still in flight.

Request: nothing on stdin.
Response: one JSON object on stdout with ``inflight`` (at most 100 entries,
each ``workflow_id`` / ``incident_id`` / ``workflow_status`` / ``operation`` /
``step_index`` / ``node_ids`` / ``step_status``), ``inflight_count``,
``scanned`` (rows read) and ``bounded`` -- ``false`` when more executable
workflows exist than the 1000-row window, in which case the caller must not
read ``inflight_count: 0`` as "clear".

Why the release engine asks: R4 (2026-09-09) moved the node-action command id
of REMEDIATE_DRIVER / UPDATE_SOFTWARE_FIRMWARE / REMEDIATE_EFA_DRIVER from
``<key>/<node>/agent-N`` to ``<key>/<node>``. A control plane of the *other*
shape that takes over such a step mid-install finds no agent-ledger row under
the id it derives and submits the install a second time, on a node that is
still running the first one (install timeout 1800 s). The release engine
therefore refuses to open an upgrade or rollback transaction while one of these
steps is PENDING or WAITING, and this probe is the one store read behind that
refusal.

Only the executable statuses are scanned: PENDING, SAFETY_PENDING, RUNNING.
BLOCKED is excluded because no DISPATCHED BLOCKED row is ever reopened -- not
because it cannot hold an install execution. It can: the dispatcher blocks a
RUNNING workflow that hits a mid-dispatch ValidationError
(``BlockedKind.INTERNAL_ERROR``) and leaves ``step_executions`` intact, so a
BLOCKED row may still carry a WAITING install. But the executor returns the
recorded result for a BLOCKED row without executing, and the operator levers
close it to SUPERSEDED and refuse rows already dispatched. The one writer that
does rewrite BLOCKED to PENDING -- ``families/node_lifecycle._state`` merging a
second node fault into an open replacement -- only reaches a row whose
``not_before`` is still in the future and that no executor owns, i.e. one the
dispatcher has never claimed, which therefore carries no install execution
(``tests/orchestration/test_replacement_merge_reopens_only_undispatched.py``).
So no control plane, old or new, will derive a command id for a dispatched
step in a BLOCKED row again, and the double submit this gate guards against
cannot start there. BLOCKED is also never archived; scanning it once let the
newest RUNNING row fall past the window and the probe report 0.
``tests/regional/test_release_inflight_install_gate.py`` pins the status set and
the INTERNAL_ERROR shape.

The probe's own failure is evidence too. A fresh store connection that fails
(the Aurora rotation window, while the Pod's warm pool keeps dispatching), a
module the deployed ``gpu_fault`` lacks -- these are answers of the form "I
could not read", not "nobody could run me", and the release engine must refuse
on them. So ``main`` runs under a catch-all that prints
``{"probe_error": "<type>: <message>"}`` on stdout and exits 1; the engine's
shell wrapper marks the exit code either way. Only a failure before this file
runs at all (no Running Pod, kubectl never reaching one) is silence.

A step is in flight when its index is neither completed nor superseded and its
latest execution record is absent (PENDING: not yet handed to an adapter) or
WAITING (the node agent has it). A latest execution that SUCCEEDED (the node is
done; ``completed_step_indexes`` follows in the same transaction) or FAILED
(failure handling, not a retry, follows) is not in flight.

The operation set is spelled out here rather than imported from
``gpu_fault.execution.config.NODE_INSTALL_OPERATIONS``: the engine ships this
file to the Pod as source (see ``probes/README.md``) and it runs against
whatever ``gpu_fault`` is *already deployed* -- for every rollback target, a
module that may predate the name. The same test pins the two sets equal.

The ``except Exception`` arm fails closed in the only direction that is safe
here: a step the probe cannot classify is counted (``UNKNOWN``), never skipped.
"""

import json
import sys
from typing import Any

from gpu_fault.app import ApplicationContext
from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus

INSTALL_OPERATIONS = {
    WorkflowOperation.REMEDIATE_DRIVER,
    WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
    WorkflowOperation.REMEDIATE_EFA_DRIVER,
}
STATUSES = {
    WorkflowStatus.PENDING,
    WorkflowStatus.SAFETY_PENDING,
    WorkflowStatus.RUNNING,
}
WINDOW = 1000
REPORT_LIMIT = 100


def step_state(workflow: Any, index: int) -> str | None:
    """``PENDING`` / ``WAITING`` when the step is in flight, else ``None``."""

    if index in set(workflow.completed_step_indexes) or index in set(
        workflow.superseded_step_indexes
    ):
        return None
    latest = None
    for execution in workflow.step_executions:
        if execution.step_index == index:
            latest = execution
    if latest is None:
        return "PENDING"
    if latest.status is WorkflowStepStatus.WAITING:
        return "WAITING"
    return None


def main() -> None:
    store = ApplicationContext.from_environment().store
    rows = list(store.list_workflows(statuses=STATUSES, limit=WINDOW + 1))
    inflight: list[dict[str, Any]] = []
    for workflow in rows:
        for index, step in enumerate(workflow.official_steps):
            if step.operation not in INSTALL_OPERATIONS:
                continue
            try:
                state = step_state(workflow, index)
            except Exception:
                state = "UNKNOWN"
            if state is None:
                continue
            inflight.append(
                {
                    "workflow_id": workflow.request_id,
                    "incident_id": workflow.incident_id,
                    "workflow_status": workflow.status.value,
                    "operation": step.operation.value,
                    "step_index": index,
                    "node_ids": list(step.node_ids),
                    "step_status": state,
                }
            )
    print(
        json.dumps(
            {
                "inflight": inflight[:REPORT_LIMIT],
                "inflight_count": len(inflight),
                "scanned": len(rows),
                "bounded": len(rows) <= WINDOW,
            },
            sort_keys=True,
        )
    )


def run() -> None:
    """Run ``main``; turn its failure into evidence the caller can refuse on."""

    try:
        main()
    except Exception as exc:  # noqa: BLE001 - the failure IS the report
        print(json.dumps({"probe_error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)


run()
