"""Control-plane evidence required by the COLLECT-015 reboot case."""

from scripts.e2e.regional import run_collector_destructive as destructive


def test_collect015_control_plane_readings() -> None:
    workflow = {
        "status": "SUCCEEDED",
        "step_executions": [{"operation": "RESTART_NODE", "status": "SUCCEEDED"}],
    }
    good = destructive.collect015_workflow_errors(
        workflow,
        submission={"state": "SUBMITTED"},
        node_after={"unschedulable": False},
        node_recovery="None",
        allow_replace="false",
    )
    assert good == [], good
    bad = destructive.collect015_workflow_errors(
        {
            "status": "SUCCEEDED",
            "step_executions": [{"operation": "RESTART_NODE", "status": "FAILED"}],
        },
        submission={"state": "INTENDED"},
        node_after={"unschedulable": True},
        node_recovery="Automatic",
        allow_replace="true",
    )
    assert len(bad) == 5, bad
    assert destructive.collect015_workflow_errors(
        None, submission=None, node_after={}, node_recovery=None, allow_replace=""
    ) == ["no workflow planned RESTART_NODE for the ALWAYS_FATAL SXID"]
