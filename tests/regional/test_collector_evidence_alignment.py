"""Acceptance oracles reject missing evidence and use current finding identities."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from gpu_fault.host_health import NodeHealthCategory, NodeHealthFinding
from gpu_fault.models import RecoveryAction, Severity
from scripts.e2e.regional import collect019_verdicts as c019
from scripts.e2e.regional import collect020_verdicts as c020

NODE = "node-a"
GPU = "GPU-removed"
CLUSTER = "cluster-a"


def census(*, silent: str = "0", erroring: str = "1") -> str:
    labels = f'{{cluster_id="{CLUSTER}",channel="HOST_TELEMETRY"}}'
    return "\n".join(
        f"{family}{labels} {value}"
        for family, value in (
            (c019.SILENT_METRIC, silent),
            (c019.ERRORING_METRIC, erroring),
        )
        if value != ""
    )


@pytest.mark.parametrize("phase", ["before", "during"])
@pytest.mark.parametrize("missing", ["empty", "silence", "partial-replica", "negative"])
def test_silence_proof_requires_present_nonnegative_samples_on_every_replica(
    phase, missing
):
    before, during = [census()], [census()]
    bad = {
        "empty": [],
        "silence": [census(silent="")],
        "partial-replica": [census(), census(silent="")],
        "negative": [census(silent="-1")],
    }[missing]
    if phase == "before":
        before = bad
    else:
        during = bad
    errors = c019.gauge_errors(before, during, cluster_id=CLUSTER)
    assert errors, "missing census evidence must not be interpreted as a zero gauge"


@pytest.mark.parametrize("erroring", ["", "-1", "0"])
def test_erroring_census_is_explicit_and_positive(erroring):
    errors = c019.gauge_errors(
        [census()], [census(erroring=erroring)], cluster_id=CLUSTER
    )
    assert errors, "the hang must be observed as an explicitly positive error census"


def identity_activity():
    finding = NodeHealthFinding(
        finding_id="finding-snapshot-gpu_inventory_identity_changed",
        event_id="snapshot-gpu_inventory_identity_changed",
        cluster_id=CLUSTER,
        node_id=NODE,
        observed_at=datetime(2026, 9, 15, tzinfo=timezone.utc),
        category=NodeHealthCategory.GPU,
        severity=Severity.CRITICAL,
        recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
        metric_name=c020.IDENTITY_METRIC,
        gpu_uuids=[GPU],
        reason=f"GPU inventory identity changed: removed=['{GPU}'] added=[]",
    )
    marker = finding.marker().model_dump(mode="json")
    incident = {
        "incident_id": marker["incident_id"],
        "event_id": finding.event_id,
        "reasons": [finding.reason],
    }
    return {
        "incidents": [incident],
        "markers": [marker],
        "workflows": [
            {
                "request_id": "workflow",
                "incident_id": incident["incident_id"],
                "status": "SUCCEEDED",
                "official_steps": [
                    {"operation": operation}
                    for operation in sorted(c020.DIAGNOSTIC_OPERATIONS)
                ],
            }
        ],
        "notifications": [
            {
                "incident_id": incident["incident_id"],
                "category": c020.OPERATOR_REVIEW_CATEGORY,
            }
        ],
    }


def test_current_marker_model_supplies_the_critical_identity_proof():
    activity = identity_activity()
    assert c020.identity_finding_errors(activity, dropped_uuid=GPU) == [], (
        "the real NodeHealthFinding marker schema must satisfy the live oracle"
    )


@pytest.mark.parametrize(
    "defect", ["absent", "warning", "wrong-gpu", "wrong-incident", "wrong-event"]
)
def test_incident_prose_does_not_replace_a_matching_critical_marker(defect):
    activity = deepcopy(identity_activity())
    marker = activity["markers"][0]
    if defect == "absent":
        activity["markers"] = []
    elif defect == "warning":
        marker["severity"] = "WARNING"
    elif defect == "wrong-gpu":
        marker["scope"]["gpu_uuids"] = ["GPU-unrelated"]
    elif defect == "wrong-incident":
        marker["incident_id"] = "other-incident"
    else:
        marker["marker_id"] = "marker-other-finding"
    errors = c020.identity_finding_errors(activity, dropped_uuid=GPU)
    assert any("CRITICAL" in error for error in errors), (
        "plausible prose cannot authorize a missing or mismatched severity proof"
    )


@pytest.mark.parametrize("metric", [c020.UNKNOWN_COUNT_METRIC, c020.MISMATCH_METRIC])
@pytest.mark.parametrize(
    ("collection", "identity_field"),
    [("incidents", "event_id"), ("markers", "marker_id"), ("findings", "metric_name")],
)
def test_structured_negative_findings_cannot_hide_behind_unrelated_reason_text(
    metric, collection, identity_field
):
    row = {
        identity_field: metric
        if identity_field == "metric_name"
        else f"sample-{metric}",
        "reasons": ["collector could not prove the node inventory invariant"],
    }
    errors = c020.no_reboot_errors(
        {collection: [row]}, boot_id_before="boot", boot_id_after="boot"
    )
    assert any(metric in error for error in errors), (
        "negative inventory evidence must be matched through its actual identity"
    )
