"""Report activity still pinned to the outgoing Runtime Profile version.

Request on stdin: ``{"previous": "<version>", "desired": "<version>"}``.
Response: one JSON object with the two counts and up to ten ids of each.

Finalizing a profile transition retires the old version, so anything still
running against it would lose the capability set it was planned under. Both
halves matter and neither implies the other: a workflow is control-plane state,
an attempt observation is a workload actually on a GPU.

``verified_restore_successor(store, item) is None`` excludes workflows a later
verified restore already closed out -- without it a finalize waits on records
whose nodes are back in the pool.

Ids are capped at ten while the counts stay exact: the counts are the gate, the
ids are for the operator to go look.
"""

import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.models import WorkflowStatus
from gpu_fault.watcher import WorkloadPhase
from gpu_fault.workflow_resolution import verified_restore_successor

NONTERMINAL = {
    WorkflowStatus.PENDING,
    WorkflowStatus.SAFETY_PENDING,
    WorkflowStatus.BLOCKED,
    WorkflowStatus.RUNNING,
}


def main() -> None:
    expected = json.load(sys.stdin)
    store = ApplicationContext.from_environment().store
    workflows = [
        item
        for item in store.list_workflows(statuses=NONTERMINAL, limit=1001)
        if item.runtime_profile_version != expected["desired"]
        and verified_restore_successor(store, item) is None
    ]
    workloads = [
        item.observation
        for item in store.list_attempt_observation_states()
        if item.observation.workload_phase
        in {WorkloadPhase.PENDING, WorkloadPhase.RUNNING}
        and item.observation.runtime_profile_version != expected["desired"]
    ]
    print(
        json.dumps(
            {
                "workflow_count": len(workflows),
                "workload_count": len(workloads),
                "workflow_ids": [item.request_id for item in workflows[:10]],
                "attempt_ids": [item.attempt_id for item in workloads[:10]],
            },
            sort_keys=True,
        )
    )


main()
