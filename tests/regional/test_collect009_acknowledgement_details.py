"""COLLECT-009 reads the operator's acknowledgement instructions from wherever
CHECK_MECHANICALS put them.

Regional sites run the step as a remote node action: the step's details are
pointers (remote_command_id, remote_status) and the Node Agent's answer -- the
annotation key and the ``<incident_id>:<fencing_token>`` value -- is on the
remote command's result_details. Attempt 2 on 2026-09-07 failed with "omitted
annotation details" while the command carried them.
"""

from __future__ import annotations

from scripts.e2e.regional import run_collector_acceptance as acceptance

INCIDENT = "inc-kernel-log-kmsg-0000-xid-54-cluster-a"
VALUE = f"{INCIDENT}:1"


def _remote_step() -> dict[str, object]:
    return {
        "operation": "CHECK_MECHANICALS",
        "status": "WAITING",
        "adapter_operation_id": "remote/remote-5eee1d4d",
        "details": {
            "remote_status": "WAITING",
            "remote_cluster_id": "hp-cluster",
            "remote_command_id": "remote-5eee1d4d",
            "mutation_submitted_by_control_plane": False,
        },
    }


def _command(
    command_id: str, operation: str = "CHECK_MECHANICALS"
) -> dict[str, object]:
    return {
        "command_id": command_id,
        "status": "WAITING",
        "step": {"operation": operation},
        "result_details": {
            "pending_nodes": ["node-a"],
            "notification_id": "notification-1",
            "required_annotation": "gpu-fault.io/mechanical-inspection-complete",
            "required_annotation_value": VALUE,
            "required_evidence": "physical seating inspected",
        },
    }


def test_remote_step_follows_its_command_pointer() -> None:
    details = acceptance.mechanical_acknowledgement_details(
        _remote_step(),
        [_command("remote-other", operation="RESET_GPU"), _command("remote-5eee1d4d")],
    )
    assert (
        details["required_annotation"] == "gpu-fault.io/mechanical-inspection-complete"
    )
    assert details["required_annotation_value"] == VALUE
    assert details["remote_command_id"] == "remote-5eee1d4d", "pointers are kept"
    assert details["notification_id"] == "notification-1"


def test_local_step_details_win_when_present() -> None:
    execution = {
        "operation": "CHECK_MECHANICALS",
        "status": "WAITING",
        "details": {
            "required_annotation": "gpu-fault.io/mechanical-inspection-complete",
            "required_annotation_value": "local:3",
        },
    }
    details = acceptance.mechanical_acknowledgement_details(
        execution, [_command("remote-5eee1d4d")]
    )
    assert details["required_annotation_value"] == "local:3", "step answer preferred"


def test_wrong_command_or_missing_answer_stays_incomplete() -> None:
    # A different command id is never read, even when it is a CHECK_MECHANICALS.
    details = acceptance.mechanical_acknowledgement_details(
        _remote_step(), [_command("remote-zzzz")]
    )
    assert "required_annotation" not in details, details

    # Without a pointer, the CHECK_MECHANICALS command is the fallback.
    execution = _remote_step()
    execution["details"] = {}
    details = acceptance.mechanical_acknowledgement_details(
        execution,
        [_command("remote-other", operation="RESET_GPU"), _command("remote-x")],
    )
    assert details["required_annotation_value"] == VALUE

    # A command that has not answered yet leaves the runner waiting, not lying.
    bare = _command("remote-5eee1d4d")
    bare["result_details"] = {"pending_nodes": ["node-a"]}
    details = acceptance.mechanical_acknowledgement_details(_remote_step(), [bare])
    assert "required_annotation_value" not in details, details
    assert (
        acceptance.mechanical_acknowledgement_details(_remote_step(), None)
        == (_remote_step()["details"])
    )
