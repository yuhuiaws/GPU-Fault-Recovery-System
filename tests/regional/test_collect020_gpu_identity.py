"""Contract tests for GF-REGIONAL-COLLECT-020 (GPU identity finding)."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.e2e.regional import collect020_verdicts as verdicts
from scripts.e2e.regional import run_collect020_gpu_identity as collect020

DROPPED = "GPU-dddd-eeee"


def _text(errors: list[str]) -> str:
    return "\n".join(errors)


def test_preconditions_need_a_known_type_two_gpus_and_a_safe_drop() -> None:
    assert (
        verdicts.preconditions_errors(
            instance_type="ml.p5.48xlarge", gpu_count=8, mismatch_samples=2
        )
        == []
    ), "a p5.48xlarge with 8 GPUs and the shipped threshold passes"
    assert "not in the expected-count table" in _text(
        verdicts.preconditions_errors(
            instance_type="t3.large", gpu_count=8, mismatch_samples=2
        )
    )
    assert "built with 8" in _text(
        verdicts.preconditions_errors(
            instance_type="p5.48xlarge", gpu_count=7, mismatch_samples=2
        )
    )
    assert "reboot path opens" in _text(
        verdicts.preconditions_errors(
            instance_type="p5.48xlarge", gpu_count=8, mismatch_samples=1
        )
    )
    assert "one-GPU node" in _text(
        verdicts.preconditions_errors(
            instance_type="p5.4xlarge", gpu_count=1, mismatch_samples=2
        )
    )
    assert verdicts.expected_count_for("ml.p5.48xlarge") == 8
    assert verdicts.expected_count_for("unknown") is None


def _activity(**overrides: Any) -> dict[str, Any]:
    activity = {
        "incidents": [
            {
                "incident_id": "inc-1",
                "state": "RECOVERED",
                "reasons": [
                    f"GPU inventory identity changed between snapshots: removed=['{DROPPED}'] added=[]"
                ],
            }
        ],
        "workflows": [
            {
                "request_id": "wf-1",
                "incident_id": "inc-1",
                "status": "SUCCEEDED",
                "official_steps": [
                    {"operation": "FREEZE_EVIDENCE"},
                    {"operation": "MARK_UNSCHEDULABLE"},
                    {"operation": "RUN_DCGM_DIAGNOSTIC"},
                    {"operation": "VALIDATE_GPU"},
                    {"operation": "RESTORE_SCHEDULING"},
                ],
            }
        ],
        "notifications": [
            {
                "incident_id": "inc-1",
                "category": "OPERATOR_REVIEW",
                "subject": "review",
            },
            {
                "incident_id": "inc-1",
                "category": "DCGM_DIAGNOSTIC",
                "subject": "DCGM diagnostic",
            },
        ],
        "markers": [
            {
                "marker_id": "m-1",
                "incident_id": "inc-1",
                "retired_at": "2030-01-01T00:00:00+00:00",
                "retired_reason": "restored",
            }
        ],
        "evidence": [
            {
                "record_id": "gpu-inventory/s1",
                "payload": {
                    "expected_gpu_count": 8,
                    "devices": [{"gpu_uuid": f"GPU-{index}"} for index in range(7)],
                },
            }
        ],
    }
    activity.update(overrides)
    return activity


def test_the_identity_finding_contract_passes_on_a_diagnostic_workflow() -> None:
    activity = _activity()
    assert verdicts.identity_finding_errors(activity, dropped_uuid=DROPPED) == []
    assert verdicts.dcgm_notification_errors(activity) == []
    assert (
        verdicts.no_reboot_errors(activity, boot_id_before="b", boot_id_after="b") == []
    )
    assert verdicts.marker_retirement_errors(activity) == []
    assert (
        verdicts.inventory_evidence_errors(
            activity["evidence"], dropped_uuid=DROPPED, expected_count=8
        )
        == []
    )


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"incidents": []}, "no incident names gpu_inventory_identity_changed"),
        (
            {
                "incidents": [
                    {
                        "incident_id": "inc-1",
                        "reasons": ["GPU inventory identity changed"],
                    }
                ]
            },
            "names the removed UUID",
        ),
        (
            {
                "workflows": [
                    {
                        "request_id": "wf-1",
                        "incident_id": "inc-1",
                        "status": "SUCCEEDED",
                        "official_steps": [{"operation": "RESTART_NODE"}],
                    }
                ]
            },
            "compiles ['RESTART_NODE']",
        ),
        (
            {
                "workflows": [
                    {
                        "request_id": "wf-1",
                        "incident_id": "inc-1",
                        "status": "RUNNING",
                        "official_steps": [{"operation": "RUN_DCGM_DIAGNOSTIC"}],
                    }
                ]
            },
            "not terminal",
        ),
        (
            {
                "notifications": [
                    {"incident_id": "inc-1", "category": "DCGM_DIAGNOSTIC"}
                ]
            },
            "no OPERATOR_REVIEW",
        ),
    ],
)
def test_each_identity_finding_deviation_fails(
    overrides: dict[str, Any], fragment: str
) -> None:
    errors = verdicts.identity_finding_errors(
        _activity(**overrides), dropped_uuid=DROPPED
    )
    assert fragment in _text(errors), errors


def test_reboot_mismatch_and_unknown_count_are_hard_failures() -> None:
    assert "rebooted" in _text(
        verdicts.no_reboot_errors(_activity(), boot_id_before="b1", boot_id_after="b2")
    )
    mismatch = _activity(
        incidents=[
            {"incident_id": "inc-2", "reasons": ["gpu_inventory_mismatch: 7 of 8"]}
        ]
    )
    assert "gpu_inventory_mismatch" in _text(
        verdicts.no_reboot_errors(mismatch, boot_id_before="b", boot_id_after="b")
    )
    unknown = _activity(
        incidents=[{"incident_id": "inc-3", "reasons": ["gpu_expected_count_unknown"]}]
    )
    assert "gpu_expected_count_unknown fired" in _text(
        verdicts.no_reboot_errors(unknown, boot_id_before="b", boot_id_after="b")
    )


def test_inventory_evidence_must_carry_the_instance_type_count() -> None:
    evidence = _activity()["evidence"]
    evidence[0]["payload"]["expected_gpu_count"] = None
    assert "expected_gpu_count=None" in _text(
        verdicts.inventory_evidence_errors(
            evidence, dropped_uuid=DROPPED, expected_count=8
        )
    )
    full = [
        {
            "record_id": "s2",
            "payload": {"expected_gpu_count": 8, "devices": [{"gpu_uuid": DROPPED}]},
        }
    ]
    assert "no GPU_INVENTORY evidence without the dropped UUID" in _text(
        verdicts.inventory_evidence_errors(full, dropped_uuid=DROPPED, expected_count=8)
    )


def test_markers_of_a_recovered_incident_must_carry_retirement_fields() -> None:
    bare = _activity(
        markers=[{"marker_id": "m-1", "incident_id": "inc-1", "retired_at": None}]
    )
    errors = verdicts.marker_retirement_errors(bare)
    assert "has no retired_at" in _text(errors) and "has no retired_reason" in _text(
        errors
    ), errors
    assert "no incident of the case reached RECOVERED" in _text(
        verdicts.marker_retirement_errors(
            _activity(incidents=[{"incident_id": "inc-1", "state": "ESCALATED"}])
        )
    )
    assert (
        verdicts.restored_node_errors(
            {"ownership_annotations": {}, "unschedulable": False, "taints": []}
        )
        == []
    )
    assert "still carries taints" in _text(
        verdicts.restored_node_errors(
            {"ownership_annotations": {}, "unschedulable": False, "taints": ["x"]}
        )
    )


def test_the_window_contract_pins_the_unit_the_unset_and_one_drop() -> None:
    opened = {
        "unit": verdicts.UNIT,
        "unset": ["GPU_FAULT_EXPECTED_GPU_COUNT"],
        "shadow": ["drop-uuid", DROPPED, "1"],
        "after": {"ActiveState": "active"},
    }
    assert verdicts.window_errors(opened, dropped_uuid=DROPPED) == []
    assert "did not unset" in _text(
        verdicts.window_errors({**opened, "unset": []}, dropped_uuid=DROPPED)
    )
    assert "hides the GPU for 2" in _text(
        verdicts.window_errors(
            {**opened, "shadow": ["drop-uuid", DROPPED, "2"]}, dropped_uuid=DROPPED
        )
    )
    assert verdicts.DROP_CALLS == 1, (
        "one query stays below the shipped 2-sample threshold"
    )


def test_the_case_constants_and_plan_name_the_hard_stop() -> None:
    assert collect020.CASE_ID == "GF-REGIONAL-COLLECT-020"
    assert collect020.CONFIRMATION == "COLLECT020_EXECUTE"
    assert verdicts.PREDECESSOR_CASE_ID == "GF-REGIONAL-COLLECT-019"
    settings = type(
        "S", (), {"node": "node-a", "regional": type("R", (), {"cluster_id": "c"})()}
    )()
    details = collect020.plan_details(settings, {"predecessor": {"valid": True}})
    assert details["risk"] == "live-isolation", details
    assert "REBOOT_NODE is unreachable" in details["hard_stop"], details["hard_stop"]
    assert "RESTART_NODE" in "\n".join(details["stop_conditions"])
    assert (
        details["rollback"]["quarantine_is_restored_only_by_validation_workflow"]
        is True
    )
