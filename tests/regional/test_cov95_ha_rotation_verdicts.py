from __future__ import annotations

import copy
from typing import Any

import pytest

from scripts.e2e.regional import ha009_observation
from scripts.e2e.regional import run_ha009_aurora_credential_rotation as ha009
from tests.regional.test_acceptance_alignment_ha_telemetry import telemetry_proof
from tests.regional.test_ha_capacity_callers import CallerEnvironment


def observations() -> dict[str, Any]:
    environment = CallerEnvironment(ha009, "")
    deployments = {name: environment.deployment(name) for name in ha009.DEPLOYMENTS}
    pods = [pod for value in deployments.values() for pod, _ in value["pods"]]
    final_probe, receipts = telemetry_proof(spooled=False)
    final_probe["accepted_request_ids"] = [
        item["request_id"] for item in receipts["requests"]
    ]
    return {
        "versions_after": {"stages": {"AWSCURRENT": "v2", "AWSPREVIOUS": "v1"}},
        "current_before": "v1",
        "digest_before": "old",
        "digest_after": "new",
        "first_job": {"logs": ["rotated=True restarted=False"]},
        "second_job": {"logs": ["rotated=False restarted=False"]},
        "deployments_before": deployments,
        "deployments_after": copy.deepcopy(deployments),
        "propagation": {"digest": "new", "pods": dict.fromkeys(pods, "new")},
        "idle_observation": {
            "samples": {
                pod: [
                    {
                        "healthz_status": 200,
                        "metrics_status": 200,
                        "fresh_connection": True,
                        "metrics": {
                            "gpu_fault_postgres_pool_connections_errors_total": 0
                        },
                    }
                ]
                for pod in pods
            },
            "auth_failures_in_logs": dict.fromkeys(pods, 0),
        },
        "final_probe": final_probe,
        "receipts": receipts,
        "runtime": {
            "command": {"status": "SUCCEEDED"},
            "notification": {
                kind: {"count": 1, "status": "SKIPPED"}
                for kind in (
                    "notification",
                    "notification_delivery",
                    "notification_result",
                )
            },
        },
        "digest_before_noop": "new",
        "digest_after_noop": "new",
        "before_noop": copy.deepcopy(deployments),
        "after_noop": copy.deepcopy(deployments),
    }


@pytest.mark.parametrize(
    "defect,fragment",
    [
        ("current", "AWSCURRENT did not change"),
        ("previous", "did not become AWSPREVIOUS"),
        ("digest", "digest did not change"),
        ("first-job", "did not report a rotation"),
        ("first-restart", "path A must not roll"),
        ("command", "command did not succeed"),
        ("notification", "notification count"),
        ("delivery", "notification_delivery count"),
        ("result", "notification_result count"),
        ("suppression", "not safely suppressed"),
        ("noop-log", "not a NOOP"),
        ("noop-digest", "NOOP refresh changed the Kubernetes Secret"),
        ("noop-generation", "generation"),
        ("noop-pods", "NOOP refresh rolled"),
    ],
)
def test_rotation_verdict_requires_every_zero_rollout_and_continuity_proof(
    defect: str, fragment: str
) -> None:
    arguments = observations()
    assert ha009.rotation_errors(**arguments) == [], (
        "the complete rotation and telemetry fixture must pass before mutation"
    )
    if defect == "current":
        arguments["versions_after"]["stages"]["AWSCURRENT"] = "v1"
    elif defect == "previous":
        arguments["versions_after"]["stages"]["AWSPREVIOUS"] = "foreign"
    elif defect == "digest":
        arguments["digest_after"] = "old"
    elif defect == "first-job":
        arguments["first_job"]["logs"] = ["rotated=False restarted=False"]
    elif defect == "first-restart":
        arguments["first_job"]["logs"] = ["rotated=True restarted=True"]
    elif defect == "command":
        arguments["runtime"]["command"]["status"] = "WAITING"
    elif defect in {"notification", "delivery", "result"}:
        kind = {
            "notification": "notification",
            "delivery": "notification_delivery",
            "result": "notification_result",
        }[defect]
        arguments["runtime"]["notification"][kind]["count"] = 0
    elif defect == "suppression":
        arguments["runtime"]["notification"]["notification_result"]["status"] = "SENT"
    elif defect == "noop-log":
        arguments["second_job"]["logs"] = ["rotated=True restarted=False"]
    elif defect == "noop-digest":
        arguments["digest_after_noop"] = "changed"
    elif defect == "noop-generation":
        arguments["after_noop"][ha009.DEPLOYMENTS[0]]["generation"] = 2
    else:
        arguments["after_noop"][ha009.DEPLOYMENTS[0]]["pods"] = []
    errors = ha009.rotation_errors(**arguments)
    assert any(fragment in item for item in errors), errors


def test_incomplete_pod_coverage_and_deployment_uid_do_not_prove_steady_state() -> None:
    arguments = observations()
    arguments["propagation"]["pods"] = {}
    errors = ha009.rotation_errors(**arguments)
    assert "Secret propagation does not cover every expected Pod" in errors
    before = arguments["deployments_before"]
    after = copy.deepcopy(before)
    after[ha009.DEPLOYMENTS[0]]["uid"] = "replacement"
    assert any(
        "UID changed" in error
        for error in ha009_observation.steady_deployments(
            before, after, list(ha009.DEPLOYMENTS)
        )
    ), "a replacement Deployment cannot satisfy zero-rollout evidence"
