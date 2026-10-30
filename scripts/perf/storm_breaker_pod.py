"""The storm breaker step's view of its cluster, inside a control-plane Pod (§8.5).

Runs in an api-ha Pod through ``kubectl exec -i ... python3 -`` with the
product installed and nothing else; the runner side is
``regional_storm_breaker.py``. One read, every poll of the step:

* the cluster's breaker record (``get_remediation_storm_breaker``);
* the workflows the product itself lists as held by it
  (``workflow_reconcile.storm_held_workflows``, what ``gpu-fault-admin
  remediation-storm release|abort`` acts on);
* the step's own seeded workflows -- status, the scope their last budget
  refusal names, when that wait began, whether the claim that tripped the
  breaker was theirs -- and their remote commands (count, statuses, the first
  creation and the last update), which say whether anything ran on a held
  workflow's nodes.

Nothing is written.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from typing import Any

TRIPPED_CODE = "REMEDIATION_STORM_TRIPPED"


def step_workflow_ids(run_id: str, workflows: int) -> list[str]:
    """The seeded workflows' ids (``seed_regional_action_workflows.action_scope``
    on cluster 0; the runner's tests pin the two equal)."""

    return [
        f"workflow-actionperf-{run_id}-c000-w{index:03d}" for index in range(workflows)
    ]


def stamp(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def workflow_view(workflow: Any) -> dict[str, Any]:
    from gpu_fault.store.shared.remediation_budgets import budget_wait_started_at

    events = [str(event.code or "") for event in workflow.events]
    return {
        "status": str(getattr(workflow.status, "value", workflow.status)),
        "last_blocked_scope": workflow.remediation_budget_last_blocked_scope,
        "wait_count": workflow.remediation_budget_wait_count,
        "wait_started_at": stamp(budget_wait_started_at(workflow)),
        "claims": len(workflow.remediation_budget_claims),
        "tripped_breaker": TRIPPED_CODE in events,
        "created_at": stamp(workflow.created_at),
        "updated_at": stamp(workflow.updated_at),
    }


def command_views(commands: list[Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Any]] = {}
    for command in commands:
        grouped.setdefault(command.workflow_request_id, []).append(command)
    return {
        request_id: {
            "count": len(items),
            "statuses": dict(
                sorted(
                    Counter(
                        str(getattr(item.status, "value", item.status))
                        for item in items
                    ).items()
                )
            ),
            "first_created_at": stamp(min(item.created_at for item in items)),
            "last_updated_at": stamp(max(item.updated_at for item in items)),
        }
        for request_id, items in sorted(grouped.items())
    }


def breaker_view(store: Any, cluster_id: str) -> dict[str, Any] | None:
    record = store.get_remediation_storm_breaker(cluster_id)
    if record is None:
        return None
    document = record.model_dump(mode="json")
    document["open"] = bool(record.open)
    document["window_nodes"] = len({entry.node_id for entry in record.window})
    document.pop("window", None)
    return document


def status(
    store: Any, *, run_id: str, cluster_id: str, workflows: int
) -> dict[str, Any]:
    from gpu_fault.store import NotFoundError
    from gpu_fault.workflow_reconcile import storm_held_workflows

    ids = step_workflow_ids(run_id, workflows)
    views: dict[str, Any] = {}
    for request_id in ids:
        try:
            views[request_id] = workflow_view(store.get_workflow(request_id))
        except NotFoundError:
            views[request_id] = None
    commands = command_views(store.list_remote_commands(workflow_request_ids=ids))
    return {
        "cluster_id": cluster_id,
        "run_id": run_id,
        "breaker": breaker_view(store, cluster_id),
        "held_workflows": sorted(
            item.request_id for item in storm_held_workflows(store, cluster_id)
        ),
        "workflows": views,
        "commands": commands,
    }


def main() -> None:
    from gpu_fault.app import ApplicationContext

    context = ApplicationContext.from_environment()
    try:
        document = status(
            context.store,
            run_id=os.environ["STORM_RUN_ID"],
            cluster_id=os.environ["STORM_CLUSTER_ID"],
            workflows=int(os.environ["STORM_WORKFLOWS"]),
        )
    finally:
        close = getattr(context.store, "close", None)
        if close is not None:
            close()
    print(json.dumps(document, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
