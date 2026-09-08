"""bootstrap creates and hardens the control-record archive bucket.

Retention is on when ``spec.retention`` names a positive day count; the bucket
used to be a manual prerequisite the deploy preflight only refused on. Now the
notification task table ensures it: create when absent (with the Region
constraint outside us-east-1), and always enforce versioning, SSE and the
public-access block, so a rerun converges an existing bucket too.
"""

from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.admin import notification_bootstrap
from gpu_fault.admin.archive_bucket import ensure_control_record_archive_bucket
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.notifications import NotificationRouting
from tests.admin.test_admin_notification_bootstrap import Runner, _cpu

URI = "s3://gpu-fault-control-records-123456789012-us-west-2/site-a/control-record-archive"


class S3Runner(Runner):
    def __init__(self, *, exists: bool, hardened: bool = False) -> None:
        super().__init__()
        self.exists = exists
        self.hardened = hardened
        self.text_commands: list[tuple[str, ...]] = []
        self.mutations: list[tuple[str, ...]] = []

    def aws_text(
        self, _region: str, *arguments: str, mutate: bool = False, **_: Any
    ) -> str:
        self.text_commands.append(arguments)
        if mutate:
            self.mutations.append(arguments)
        if arguments[:2] == ("s3api", "head-bucket") and not self.exists:
            raise BootstrapError(
                "An error occurred (404) when calling the HeadBucket operation: Not Found"
            )
        return ""

    def aws_json(
        self, _region: str, *arguments: str, mutate: bool = False, **_: Any
    ) -> Any:
        self.commands.append(arguments)
        if mutate:
            self.mutations.append(arguments)
            return {"Location": "/gpu-fault-control-records-123456789012-us-west-2"}
        verb = arguments[1]
        if verb == "get-bucket-versioning":
            return {"Status": "Enabled"} if self.hardened else {}
        if verb == "get-bucket-encryption":
            if not self.hardened:
                raise BootstrapError("ServerSideEncryptionConfigurationNotFoundError")
            return {
                "ServerSideEncryptionConfiguration": {
                    "Rules": [
                        {
                            "ApplyServerSideEncryptionByDefault": {
                                "SSEAlgorithm": "AES256"
                            }
                        }
                    ]
                }
            }
        if verb == "get-public-access-block":
            if not self.hardened:
                raise BootstrapError("NoSuchPublicAccessBlockConfiguration")
            return {
                "PublicAccessBlockConfiguration": {
                    "BlockPublicAcls": True,
                    "IgnorePublicAcls": True,
                    "BlockPublicPolicy": True,
                    "RestrictPublicBuckets": True,
                }
            }
        raise AssertionError(f"unexpected read {arguments}")


def _verbs(runner: S3Runner) -> list[str]:
    return [command[1] for command in runner.mutations]


def test_a_missing_bucket_is_created_in_the_site_region_and_hardened() -> None:
    runner = S3Runner(exists=False)

    result = ensure_control_record_archive_bucket(
        runner, region="us-west-2", archive_s3_uri=URI
    )

    assert result == {
        "bucket": "gpu-fault-control-records-123456789012-us-west-2",
        "prefix": "site-a/control-record-archive",
        "region": "us-west-2",
        "created": True,
        "changed": ["versioning", "encryption", "public_access_block"],
    }
    create = [c for c in runner.commands if c[:2] == ("s3api", "create-bucket")]
    assert create and "LocationConstraint=us-west-2" in " ".join(create[0])
    assert _verbs(runner)[1:] == [
        "put-bucket-versioning",
        "put-bucket-encryption",
        "put-public-access-block",
    ], "hardening must follow creation"


def test_an_existing_bucket_is_not_recreated_but_still_hardened() -> None:
    runner = S3Runner(exists=True, hardened=False)

    result = ensure_control_record_archive_bucket(
        runner, region="us-west-2", archive_s3_uri=URI
    )

    assert result["created"] is False
    assert not [c for c in runner.commands if c[:2] == ("s3api", "create-bucket")]
    assert _verbs(runner) == [
        "put-bucket-versioning",
        "put-bucket-encryption",
        "put-public-access-block",
    ]


def test_a_hardened_bucket_asks_for_no_mutation() -> None:
    """The read-only probe of a converged site must not report drift."""

    runner = S3Runner(exists=True, hardened=True)

    result = ensure_control_record_archive_bucket(
        runner, region="us-west-2", archive_s3_uri=URI
    )

    assert result["created"] is False and result["changed"] == []
    assert runner.mutations == []


def test_us_east_1_takes_no_location_constraint() -> None:
    runner = S3Runner(exists=False)

    ensure_control_record_archive_bucket(
        runner,
        region="us-east-1",
        archive_s3_uri="s3://gpu-fault-control-records-1-us-east-1/site/archive",
    )

    create = [c for c in runner.commands if c[:2] == ("s3api", "create-bucket")]
    assert create and "create-bucket-configuration" not in " ".join(create[0])


def test_other_head_bucket_errors_are_not_treated_as_absent() -> None:
    class DeniedRunner(S3Runner):
        def aws_text(self, _region: str, *arguments: str, **_: Any) -> str:
            raise BootstrapError(
                "An error occurred (AccessDenied) when calling HeadBucket"
            )

    with pytest.raises(BootstrapError, match="AccessDenied"):
        ensure_control_record_archive_bucket(
            DeniedRunner(exists=True), region="us-west-2", archive_s3_uri=URI
        )


def test_the_task_table_ensures_the_bucket_when_retention_names_one(
    tmp_path, monkeypatch
) -> None:
    calls: dict[str, dict[str, Any]] = {}

    def spy(name: str):
        def run(*_args: Any, **kwargs: Any) -> dict[str, Any]:
            calls[name] = kwargs
            return {"task": name}

        return run

    for target in (
        "ensure_control_plane_role",
        "ensure_email_notifications",
        "ensure_monitoring_resources",
        "ensure_control_record_archive_bucket",
    ):
        monkeypatch.setattr(notification_bootstrap, target, spy(target))
    routing = NotificationRouting(
        sender="sender@example.com",
        recipients=("oncall@example.com",),
        subject_prefix="[PROD]",
        channel="sns",
    )

    tasks = notification_bootstrap.notification_bootstrap_tasks(
        Runner(),
        state=None,
        cpu=_cpu(),
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        namespace="gpu-fault-system",
        site_id="site-a",
        admin_email="ops@example.com",
        routing=routing,
        archive_s3_uri=URI,
    )

    assert "control_record_archive_bucket" in tasks
    tasks["control_record_archive_bucket"]()
    assert calls["ensure_control_record_archive_bucket"] == {
        "region": "us-west-2",
        "archive_s3_uri": URI,
    }
    assert calls["ensure_control_record_archive_bucket"]["archive_s3_uri"] == (
        calls.get("ensure_control_plane_role", {}).get("archive_s3_uri", URI)
    )
