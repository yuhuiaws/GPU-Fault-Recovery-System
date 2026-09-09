"""Create and harden the control-record archive bucket (site retention).

Retention archives incident bundles to ``s3://bucket/prefix`` before deleting
live rows. The bucket used to be a manual prerequisite the deploy preflight
could only refuse on; the bootstrap task table now ensures it, so turning
retention on is one ``controlRecordRetentionDays`` line in ``site.yaml``.
"""

from __future__ import annotations

import json
from typing import Any

from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.site import S3_URI_PATTERN

ARCHIVE_BUCKET_NOT_FOUND = ("404", "NoSuchBucket", "Not Found")
ARCHIVE_BUCKET_SETTING_ABSENT = (
    "ServerSideEncryptionConfigurationNotFoundError",
    "NoSuchPublicAccessBlockConfiguration",
)
ARCHIVE_PUBLIC_ACCESS_BLOCK = {
    "BlockPublicAcls": True,
    "IgnorePublicAcls": True,
    "BlockPublicPolicy": True,
    "RestrictPublicBuckets": True,
}


def _archive_bucket_setting(
    runner: CommandRunner, region: str, *arguments: str
) -> dict[str, Any] | None:
    """One ``get-bucket-*`` read; the setting's own not-configured code is None."""

    try:
        return runner.aws_json(region, *arguments)
    except BootstrapError as exc:
        if any(code in str(exc) for code in ARCHIVE_BUCKET_SETTING_ABSENT):
            return None
        raise


def ensure_control_record_archive_bucket(
    runner: CommandRunner, *, region: str, archive_s3_uri: str
) -> dict[str, Any]:
    """Create the control-record archive bucket when absent and harden it.

    Retention archives incident bundles to ``s3://bucket/prefix`` before
    deleting live rows; the bucket used to be a manual prerequisite the deploy
    preflight could only refuse on. The bucket is created in the site Region
    (no LocationConstraint in us-east-1, which rejects one); versioning, SSE-S3
    default encryption and the public-access block are read first and written
    only when they drift, so the read-only probe of a converged bucket asks
    for no mutation and an existing bucket converges on the next run. Only the
    ``head-bucket`` not-found codes mean absent; AccessDenied or a broken CLI
    is re-raised rather than turned into a create that fails or succeeds twice.
    """

    match = S3_URI_PATTERN.fullmatch(archive_s3_uri.strip())
    if match is None:
        raise BootstrapError(f"archive URI is not s3://bucket/prefix: {archive_s3_uri}")
    bucket = match.group("bucket")
    prefix = (match.group("prefix") or "").strip("/")
    try:
        runner.aws_text(region, "s3api", "head-bucket", "--bucket", bucket)
        created = False
    except BootstrapError as exc:
        if not any(code in str(exc) for code in ARCHIVE_BUCKET_NOT_FOUND):
            raise
        arguments = ["s3api", "create-bucket", "--bucket", bucket]
        if region != "us-east-1":
            arguments += [
                "--create-bucket-configuration",
                f"LocationConstraint={region}",
            ]
        runner.aws_json(region, *arguments, mutate=True)
        created = True
    changed: list[str] = []
    versioning = (
        {}
        if created
        else _archive_bucket_setting(
            runner, region, "s3api", "get-bucket-versioning", "--bucket", bucket
        )
        or {}
    )
    if versioning.get("Status") != "Enabled":
        runner.aws_text(
            region,
            "s3api",
            "put-bucket-versioning",
            "--bucket",
            bucket,
            "--versioning-configuration",
            "Status=Enabled",
            mutate=True,
        )
        changed.append("versioning")
    encryption = (
        None
        if created
        else _archive_bucket_setting(
            runner, region, "s3api", "get-bucket-encryption", "--bucket", bucket
        )
    )
    rules = ((encryption or {}).get("ServerSideEncryptionConfiguration") or {}).get(
        "Rules"
    ) or []
    if not any(
        (rule.get("ApplyServerSideEncryptionByDefault") or {}).get("SSEAlgorithm")
        in {"AES256", "aws:kms"}
        for rule in rules
    ):
        runner.aws_text(
            region,
            "s3api",
            "put-bucket-encryption",
            "--bucket",
            bucket,
            "--server-side-encryption-configuration",
            json.dumps(
                {
                    "Rules": [
                        {
                            "ApplyServerSideEncryptionByDefault": {
                                "SSEAlgorithm": "AES256"
                            },
                            "BucketKeyEnabled": True,
                        }
                    ]
                }
            ),
            mutate=True,
        )
        changed.append("encryption")
    block = (
        None
        if created
        else _archive_bucket_setting(
            runner, region, "s3api", "get-public-access-block", "--bucket", bucket
        )
    )
    current_block = (block or {}).get("PublicAccessBlockConfiguration") or {}
    if any(current_block.get(key) is not True for key in ARCHIVE_PUBLIC_ACCESS_BLOCK):
        runner.aws_text(
            region,
            "s3api",
            "put-public-access-block",
            "--bucket",
            bucket,
            "--public-access-block-configuration",
            ",".join(f"{key}=true" for key in ARCHIVE_PUBLIC_ACCESS_BLOCK),
            mutate=True,
        )
        changed.append("public_access_block")
    return {
        "bucket": bucket,
        "prefix": prefix,
        "region": region,
        "created": created,
        "changed": changed,
    }
