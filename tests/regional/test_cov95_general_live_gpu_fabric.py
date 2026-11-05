"""GF-GENERAL-LIVE fabric recovery failure: held-failure and exit verdict edges."""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional import general_live_gpu_fabric as fab
from tests.regional._general_live_gpu_fakes import gpu_incident, gpu_workflow


def execution(
    index: int, operation: str, status: str, at: str | None, **extra: Any
) -> dict[str, Any]:
    stamp = None if at is None else f"2026-10-27T{at}+00:00"
    return {
        "step_index": index,
        "operation": operation,
        "status": status,
        "started_at": extra.pop("started_at", stamp),
        "updated_at": stamp,
        "error": None,
        "details": {},
        **extra,
    }


def doc(
    executions: list[dict[str, Any]],
    *,
    state: str = "ACTION_PENDING",
    status: str = "RUNNING",
) -> dict[str, Any]:
    operations = [item["operation"] for item in executions]
    return {
        "incidents": [gpu_incident("inc", "b-node-fabric", workflow="wf", state=state)],
        "workflows": [
            gpu_workflow("wf", "inc", operations, status=status, executions=executions)
        ],
    }


def test_no_incident_means_nothing_to_judge() -> None:
    record = doc([execution(0, "RESET_ALL_GPUS_NVSWITCHES", "SUCCEEDED", "05:16:00")])

    assert fab.held_failure_errors(record, []) == []
    assert (
        fab.escalation_exit_errors(record, [], boot_before="b-1", boot_now="b-2") == []
    )


NO_VALIDATION = (
    "the post-reset validation did not fail on the fabric the replay kept failed: "
    "no FAILED VALIDATE_GPU/VALIDATE_FABRIC after RESET_ALL_GPUS_NVSWITCHES "
    "(no validation executed)"
)


def test_incidents_the_read_does_not_hold_are_judged_by_their_own_state() -> None:
    tracked = [
        gpu_incident("inc-x", "b-node-fabric", workflow="wf-x"),
        gpu_incident("inc-y", "b-node-fabric", workflow="wf-y", state="DETECTED"),
    ]
    with_rung = doc(
        [
            execution(4, "RESET_ALL_GPUS_NVSWITCHES", "SUCCEEDED", "05:16:00"),
            # A rung without any time stamp yet (just planned on the node).
            execution(9, "RESTART_NODE", "WAITING", None, started_at=None),
        ]
    )
    without_rung = doc(
        [execution(4, "RESET_ALL_GPUS_NVSWITCHES", "SUCCEEDED", "05:16:00")]
    )

    # The tracked incidents own no workflow in the read: no validation of theirs
    # ran; the node's rung still counts, and without one the first tracked
    # incident's state is what did not escalate.
    assert fab.held_failure_errors(with_rung, tracked) == [NO_VALIDATION]
    assert fab.held_failure_errors(without_rung, tracked) == [
        NO_VALIDATION,
        "a fabric that stayed failed after the reset did not escalate/hold isolation "
        "(incident ACTION_PENDING, workflow None, no RESTART_NODE rung)",
    ]


def test_a_settled_success_needs_one_reset_and_a_recovered_incident() -> None:
    record = doc(
        [
            execution(4, "RESET_ALL_GPUS_NVSWITCHES", "SUCCEEDED", "05:16:00"),
            execution(9, "RESET_ALL_GPUS_NVSWITCHES", "SUCCEEDED", "05:19:00"),
            execution(10, "VALIDATE_GPU", "SUCCEEDED", "05:20:00"),
        ],
        status="SUCCEEDED",
    )

    errors = fab.escalation_exit_errors(
        record,
        record["incidents"],
        boot_before="b-1",
        boot_now="b-1",
        node={"unschedulable": False, "taints": [], "ownership_annotations": {}},
    )

    assert errors == [
        "RESET_ALL_GPUS_NVSWITCHES succeeded 2 times (expected exactly one): the "
        "ladder looped on the reset",
        "the record ended SUCCEEDED but the incident is ACTION_PENDING, not RECOVERED",
    ]
