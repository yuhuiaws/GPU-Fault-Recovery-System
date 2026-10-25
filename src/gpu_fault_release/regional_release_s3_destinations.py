"""Preflight checks of the S3 destinations a release writes evidence to.

Two destinations, one grammar (``gpu_fault.admin.site_s3``): the control
record archive the control-worker archives closed incidents into
(``site.yaml`` ``spec.retention``) and the diagnostic evidence destination
every GPU node's Agent uploads its archives to (``spec.diagnosticS3Uri``).
Both checks are read-only -- ``get-bucket-location`` proves a bucket exists
and names its Region -- and neither inspects an IAM role.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from gpu_fault.admin.site_s3 import S3_URI_PATTERN
from gpu_fault_release.regional_check_value import CheckSkipped, CheckValue
from gpu_fault_release.regional_release_errors import ReleaseError
from gpu_fault_release.regional_release_state import aws_json


def _bucket_region(release: Any, bucket: str) -> str:
    """The bucket's Region; S3 reports the us-east-1 location as null."""

    location = aws_json(
        release, ["s3api", "get-bucket-location", "--bucket", bucket]
    ).get("LocationConstraint")
    return str(location or "us-east-1")


def check_control_record_archive_bucket(release: Any) -> CheckValue:
    """The control-record archive target exists in this Region before the
    workers that will archive into it roll out (control-plane review
    2026-09-08, F-8 / Agent 10).

    Retention is on: every control-worker reads
    ``GPU_FAULT_CONTROL_RECORD_ARCHIVE_S3_URI`` (site.yaml ``spec.retention``)
    and the archiver soft-fails (keeps live rows, counts errors) when the
    bucket is missing -- a failure once discovered from an alert hours later;
    this check refuses the release instead. Read-only: ``get-bucket-location``
    proves the bucket exists and sits in the site Region.
    """

    retention = getattr(release.config, "retention", None)
    uri = str(getattr(retention, "archive_s3_uri", "") or "")
    if not uri:
        raise CheckSkipped(
            "control record retention is off (site.yaml spec.retention not declared)"
        )
    parts = urlsplit(uri)
    bucket = parts.netloc
    prefix = parts.path.strip("/")
    if parts.scheme != "s3" or not bucket or not prefix:
        raise ReleaseError(
            f"control record archive target must be s3://bucket/prefix, got {uri}"
        )
    region = _bucket_region(release, bucket)
    if region != release.config.aws_region:
        raise ReleaseError(
            f"control record archive bucket {bucket} is in {region}, "
            f"expected the site Region {release.config.aws_region}"
        )
    return CheckValue(
        f"control record archive bucket {bucket} exists in {region}",
        {"bucket": bucket, "prefix": prefix, "region": region},
    )


def check_diagnostic_evidence_destination(release: Any) -> CheckValue:
    """Where the GPU nodes' diagnostic archives end up (site.yaml
    ``spec.diagnosticS3Uri`` -> node ``GPU_FAULT_DIAGNOSTIC_S3_URI``).

    Unset is a WARN, not a refusal -- it is what every site ran with before the
    setting existed -- but one the operator should read: every evidence
    reference a notification carries is then a ``file://`` path readable only
    on the faulting node (live 2026-10-25, hung-triage hand-off). Set, the
    read-only ``get-bucket-location`` proves the bucket exists; a bucket in
    another Region is a WARN (the node's S3 client follows the redirect). The
    node roles' ``s3:PutObject`` on the prefix is named, not verified: the
    preflight reads no GPU node role, and a node whose upload fails keeps the
    archive and reports ``evidence_upload_error`` with its ``file://``
    reference.
    """

    uri = str(getattr(release.config, "diagnostic_s3_uri", None) or "")
    if not uri:
        return CheckValue(
            "diagnostic evidence archives stay on the GPU nodes: site.yaml "
            "spec.diagnosticS3Uri is not set, so every notification's evidence "
            "reference is a file:// path readable only on the faulting node",
            {"configured": False},
            status="WARN",
        )
    match = S3_URI_PATTERN.fullmatch(uri)
    if match is None:
        raise ReleaseError(
            f"diagnostic evidence destination must be s3://bucket[/prefix], got {uri}"
        )
    bucket = match.group("bucket")
    prefix = (match.group("prefix") or "").strip("/")
    object_arn = f"arn:aws:s3:::{bucket}/{prefix + '/' if prefix else ''}*"
    region = _bucket_region(release, bucket)
    details = {
        "configured": True,
        "bucket": bucket,
        "prefix": prefix,
        "region": region,
        "object_arn": object_arn,
    }
    if region != release.config.aws_region:
        return CheckValue(
            f"diagnostic evidence bucket {bucket} is in {region}, not the site "
            f"Region {release.config.aws_region}; node uploads follow the S3 "
            "redirect, but evidence then leaves the site Region",
            details,
            status="WARN",
        )
    return CheckValue(
        f"diagnostic evidence bucket {bucket} exists in {region}; the GPU node "
        f"roles need s3:PutObject on {object_arn} (not verified by this check)",
        details,
    )
