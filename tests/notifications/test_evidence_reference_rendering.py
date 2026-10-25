"""Evidence references in a notification say where the archive can be read.

Live 2026-10-25 (hung-triage hand-off mail): the archives were listed as
``file:///var/lib/gpu-fault/...`` -- a path on the faulting node an operator
reading the mail cannot open, with nothing saying so. A ``file://`` reference
now names its node and states that it is readable only there because the
diagnostic S3 destination (``site.yaml`` ``spec.diagnosticS3Uri`` /
``GPU_FAULT_DIAGNOSTIC_S3_URI``) is not configured on that node or the upload
failed; an ``s3://`` reference carries the console deep link the operator can
click. ``AdvisoryNotification.evidence_refs`` keeps the raw references.
"""

from __future__ import annotations

import pytest

from gpu_fault.notifications.common import (
    describe_evidence_ref,
    render_evidence_refs,
    s3_console_url,
)
from gpu_fault.notifications.dcgm_diagnostic import DcgmDiagnosticEmailBuilder
from gpu_fault.notifications.not_applicable import NotApplicableEmailBuilder
from gpu_fault.notifications.sxid_event import SxidEventEmailBuilder

FILE_REF = "file:///var/lib/gpu-fault/diagnostics/gpu-diagnostic-20261025T0900Z.tar.gz"
S3_REF = "s3://gpu-fault-evidence/site-a/node-a/gpu-diagnostic-20261025T0900Z.tar.gz"
CONSOLE = (
    "https://us-west-2.console.aws.amazon.com/s3/object/gpu-fault-evidence"
    "?region=us-west-2&prefix=site-a/node-a/gpu-diagnostic-20261025T0900Z.tar.gz"
)


@pytest.fixture
def region(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    return "us-west-2"


def test_an_s3_reference_renders_with_its_console_deep_link(region: str) -> None:
    assert render_evidence_refs([S3_REF], node_id="node-a").splitlines() == [
        f"- {S3_REF}",
        f"  Console: {CONSOLE}",
    ]
    assert s3_console_url(S3_REF) == CONSOLE


def test_the_console_link_quotes_the_key_and_falls_back_to_the_global_console(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")
    assert s3_console_url("s3://bucket/a b/c#1.tar.gz") == (
        "https://eu-central-1.console.aws.amazon.com/s3/object/bucket"
        "?region=eu-central-1&prefix=a%20b/c%231.tar.gz"
    )
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    assert s3_console_url("s3://bucket/key") == (
        "https://s3.console.aws.amazon.com/s3/object/bucket?prefix=key"
    )
    assert s3_console_url("s3://bucket-only") is None, "no object, no object link"
    assert s3_console_url(FILE_REF) is None


def test_a_file_reference_names_its_node_and_says_it_is_readable_only_there() -> None:
    lines = render_evidence_refs([FILE_REF], node_id="node-a").splitlines()

    assert lines[0] == f"- {FILE_REF}"
    note = lines[1]
    assert note.startswith("  "), note
    assert "node-a" in note, note
    assert "readable only on" in note, note
    assert "diagnostic_s3_uri" in note and "spec.diagnosticS3Uri" in note, note
    assert "not configured" in note and "upload failed" in note, note
    assert "evidence_upload_error" in note, note
    assert "s3://" not in note.replace("S3", ""), "no made-up S3 location"


def test_a_file_reference_without_a_known_node_still_warns() -> None:
    (reference, note) = render_evidence_refs([FILE_REF]).splitlines()

    assert reference == f"- {FILE_REF}"
    assert "the faulting node" in note and "readable only on" in note, note
    one_line = describe_evidence_ref(FILE_REF)
    assert one_line.startswith(FILE_REF) and "readable only on" in one_line, one_line


def test_other_schemes_and_empty_lists_render_as_before() -> None:
    assert render_evidence_refs(["kmsg://worker-1/boot-1/11"]) == (
        "- kmsg://worker-1/boot-1/11"
    )
    assert render_evidence_refs([]) == "- NONE"
    assert render_evidence_refs([], empty="- none") == "- none"
    assert describe_evidence_ref("kmsg://worker-1/boot-1/11") == (
        "kmsg://worker-1/boot-1/11"
    )


def test_an_upload_error_from_the_step_result_is_quoted() -> None:
    error = (
        "s3://gpu-fault-evidence/site-a/node-a/x.tar.gz: PermissionError: AccessDenied"
    )
    (_, note) = render_evidence_refs(
        [FILE_REF], node_id="node-a", upload_error=error
    ).splitlines()

    assert "upload failed" in note and "AccessDenied" in note, note
    assert "not configured" not in note, (
        "when the node reported the failed upload, the note does not blame the config"
    )


def test_the_dcgm_diagnostic_mail_annotates_each_node_reference(region: str) -> None:
    notification = DcgmDiagnosticEmailBuilder().build(
        cluster_id="cluster-a",
        incident_id="incident-a",
        workflow_id="workflow-a",
        event_id="event-a",
        operation_id="operation-a",
        workload_ids=[],
        control_plane_action="OBSERVE",
        node_results={
            "node-a": {
                "diagnostic_outcome": "PASS",
                "diagnostic_findings": [],
                "evidence_ref": FILE_REF,
            },
            "node-b": {
                "diagnostic_outcome": "PASS",
                "diagnostic_findings": [],
                "evidence_ref": S3_REF,
            },
        },
    )

    body = notification.body_text
    assert f"- {FILE_REF}\n  " in body, body
    assert "readable only on node-a" in body, body
    assert f"- {S3_REF}\n  Console: {CONSOLE}" in body, body
    assert notification.evidence_refs == [FILE_REF, S3_REF], notification.evidence_refs


def test_node_scoped_builders_render_the_node_of_a_file_reference() -> None:
    notification = NotApplicableEmailBuilder().build(
        cluster_id="cluster-a",
        node_id="node-a",
        incident_id="incident-a",
        event_id="event-a",
        xid=31,
        product="H100",
        driver_branch=None,
        cuda_version=None,
        observed_at="2026-10-25T09:00:00+00:00",
        workload_state="RUNNING",
        workload_ids=[],
        official_action="NONE",
        investigatory_action=None,
        policy_source="catalog",
        policy_version="610",
        reasons=["unit"],
        safety_action=None,
        workflow_id="workflow-a",
        workflow_status="SUCCEEDED",
        safety_steps=[],
        official_steps=[],
        evidence_refs=[FILE_REF, "kmsg://node-a/boot-1/11"],
    )

    body = notification.body_text
    assert "readable only on node-a" in body, body
    assert "- kmsg://node-a/boot-1/11" in body, body
    assert notification.evidence_refs == [FILE_REF, "kmsg://node-a/boot-1/11"]


def test_the_sxid_mail_describes_its_single_reference_inline() -> None:
    notification = SxidEventEmailBuilder().build(
        cluster_id="cluster-a",
        node_id="node-a",
        incident_id="incident-a",
        workflow_id="workflow-a",
        event_id="event-a",
        sxid=20034,
        product="H100",
        observed_at="2026-10-25T09:00:00+00:00",
        event_source="fabric-manager",
        evidence_ref=FILE_REF,
        workload_state="RUNNING",
        workload_ids=[],
        classification="FATAL",
        classification_source="catalog",
        link_scope="NODE",
        link_scope_source="catalog",
        switch_id="0",
        port="1",
        pci_bdf="0000:00:00.0",
        fabric_partition="0",
        disposition="DRAIN",
        official_action="DRAIN",
        effective_action="DRAIN",
        safety_action=None,
        investigatory_action=None,
        policy_source="catalog",
        policy_version="610",
        reasons=["unit"],
    )

    assert f"- Evidence: {FILE_REF} (" in notification.body_text, notification.body_text
    assert "readable only on node-a" in notification.body_text
    assert notification.evidence_refs == [FILE_REF]
