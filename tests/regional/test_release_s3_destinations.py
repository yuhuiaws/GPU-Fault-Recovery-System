"""The preflight report's domains, and the two S3 destination checks.

The control-record archive bucket (``spec.retention``) and the diagnostic
evidence destination (``spec.diagnosticS3Uri``; live 2026-10-25 every
hung-triage archive was a ``file://`` on the faulting node because the managed
deploy could not set one) are both proved read-only with
``get-bucket-location`` by ``regional_release_s3_destinations``; the report
registers them next to each other.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_admin_checks as CHECKS
from gpu_fault_release import regional_release_s3_destinations as DESTINATIONS


def _checks_module():
    return CHECKS


class Config:
    site_name = "test-site"
    clusters = []


class Release:
    config = Config()


def test_preflight_report_contains_all_required_domains(monkeypatch) -> None:
    module = _checks_module()
    names = (
        "_check_tools",
        "_check_local_inputs",
        "check_artifact_config_maps",
        "_check_aws_identity",
        "_check_contexts",
        "_check_cpu_capacity",
        "check_cpu_secrets",
        "_check_load_balancer_controller",
        "_check_nlb_inputs",
        "_check_aurora",
        "check_aurora_refresh",
        "check_control_record_archive_bucket",
        "check_diagnostic_evidence_destination",
        "check_email_notifications",
        "_check_monitoring",
        "workflow_safety_snapshot",
    )
    for name in names:
        monkeypatch.setattr(
            module, name, lambda *args, name=name: module.CheckValue(name)
        )
    report = module.build_preflight_report(Release())

    assert report["healthy"] is True
    assert (report["summary"]["PASS"], report["summary"]["WARN"]) == (len(names), 1)
    assert {item["name"] for item in report["checks"]} == {
        "tools",
        "local_inputs",
        "artifact_config_maps",
        "aws_identity",
        "regional_contexts",
        "cpu_capacity",
        "cpu_secrets",
        "load_balancer_controller",
        "nlb_inputs",
        "aurora",
        "aurora_credential_refresh",
        "control_record_archive_bucket",
        "diagnostic_evidence_destination",
        "email_notifications",
        "monitoring",
        "workflow_safety",
        "token_rotation",
    }


def test_archive_bucket_check_requires_the_site_region(monkeypatch) -> None:
    module = _checks_module()
    release = SimpleNamespace(
        config=SimpleNamespace(
            aws_region="us-east-2",
            retention=SimpleNamespace(
                archive_s3_uri="s3://gpu-fault-control-records/site/archive"
            ),
        )
    )
    monkeypatch.setattr(
        DESTINATIONS,
        "aws_json",
        lambda _release, _args: {"LocationConstraint": "us-east-2"},
    )
    value = module.check_control_record_archive_bucket(release)
    assert value.details == {
        "bucket": "gpu-fault-control-records",
        "prefix": "site/archive",
        "region": "us-east-2",
    }

    monkeypatch.setattr(
        DESTINATIONS,
        "aws_json",
        lambda _release, _args: {"LocationConstraint": "eu-west-1"},
    )
    with pytest.raises(module.ReleaseError, match="expected the site Region"):
        module.check_control_record_archive_bucket(release)

    release.config.retention.archive_s3_uri = "s3://bucket-only"
    with pytest.raises(module.ReleaseError, match="s3://bucket/prefix"):
        module.check_control_record_archive_bucket(release)

    release.config.retention.archive_s3_uri = None
    with pytest.raises(module.CheckSkipped):
        module.check_control_record_archive_bucket(release)


def test_diagnostic_evidence_destination_warns_when_unset_and_reads_the_bucket(
    monkeypatch,
) -> None:
    """``spec.diagnosticS3Uri`` unset: a WARN that says the archives stay on
    the nodes (live 2026-10-25: every hung-triage archive was a ``file://``
    on the faulting node). Set: a read-only ``get-bucket-location`` proves the
    bucket exists; another Region is a WARN, not a refusal; the node roles'
    ``s3:PutObject`` is named, not verified (the preflight reads no node role)."""

    module = _checks_module()
    release = SimpleNamespace(
        config=SimpleNamespace(aws_region="us-east-2", diagnostic_s3_uri=None)
    )

    value = module.check_diagnostic_evidence_destination(release)
    assert value.status == "WARN", value
    assert "stay on the GPU nodes" in value.summary, value.summary
    assert "spec.diagnosticS3Uri" in value.summary, value.summary
    assert value.details == {"configured": False}

    release.config.diagnostic_s3_uri = "s3://gpu-fault-evidence/site-a/diagnostics"
    calls = []

    def aws_json(_release, arguments):
        calls.append(arguments)
        return {"LocationConstraint": "us-east-2"}

    monkeypatch.setattr(DESTINATIONS, "aws_json", aws_json)
    value = module.check_diagnostic_evidence_destination(release)
    assert value.status == "PASS", value
    assert calls == [["s3api", "get-bucket-location", "--bucket", "gpu-fault-evidence"]]
    assert value.details == {
        "configured": True,
        "bucket": "gpu-fault-evidence",
        "prefix": "site-a/diagnostics",
        "region": "us-east-2",
        "object_arn": "arn:aws:s3:::gpu-fault-evidence/site-a/diagnostics/*",
    }
    assert "s3:PutObject" in value.summary, value.summary

    monkeypatch.setattr(
        DESTINATIONS, "aws_json", lambda _release, _args: {"LocationConstraint": None}
    )
    release.config.aws_region = "us-east-1"
    assert module.check_diagnostic_evidence_destination(release).details["region"] == (
        "us-east-1"
    ), "a null LocationConstraint is us-east-1"

    monkeypatch.setattr(
        DESTINATIONS,
        "aws_json",
        lambda _release, _args: {"LocationConstraint": "eu-west-1"},
    )
    value = module.check_diagnostic_evidence_destination(release)
    assert value.status == "WARN", value
    assert "eu-west-1" in value.summary and "us-east-1" in value.summary, value.summary

    def missing(_release, _args):
        raise module.ReleaseError("An error occurred (NoSuchBucket)")

    monkeypatch.setattr(DESTINATIONS, "aws_json", missing)
    with pytest.raises(module.ReleaseError, match="NoSuchBucket"):
        module.check_diagnostic_evidence_destination(release)

    release.config.diagnostic_s3_uri = "s3://bucket-only"
    monkeypatch.setattr(
        DESTINATIONS,
        "aws_json",
        lambda _release, _args: {"LocationConstraint": "us-east-1"},
    )
    value = module.check_diagnostic_evidence_destination(release)
    assert value.details["prefix"] == "" and value.details["object_arn"] == (
        "arn:aws:s3:::bucket-only/*"
    ), "a bucket without a prefix is a valid destination"

    release.config.diagnostic_s3_uri = "https://bucket/prefix"
    with pytest.raises(module.ReleaseError, match="s3://bucket"):
        module.check_diagnostic_evidence_destination(release)
