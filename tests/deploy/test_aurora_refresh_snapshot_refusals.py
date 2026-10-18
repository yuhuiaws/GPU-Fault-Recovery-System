"""Snapshot and convergence refusals of the Aurora credential refresher.

The transaction tests prove the restore path; these pin the identity checks a
captured or supplied snapshot must pass and the convergence checks that turn a
lingering diff or a still-present object into a hard refusal.
"""

from __future__ import annotations

import base64
import copy
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_release_aurora_refresh as REFRESH
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional.test_release_aurora_refresh_transaction import (
    NAMESPACE,
    ObjectRunner,
    objects,
    release,
)


def encoded_arn(arn: str) -> SimpleNamespace:
    runner = SimpleNamespace(
        dry_run=False,
        run=lambda _arguments, **_options: base64.b64encode(arn.encode()).decode(),
    )
    return release(runner)


@pytest.mark.parametrize(
    "arn,message",
    [
        ("arn:aws:s3:::fixture-bucket", "invalid identity"),
        (
            "arn:aws:secretsmanager:eu-west-1:123456789012:secret:rds!x",
            "invalid identity",
        ),
        (
            "arn:aws:secretsmanager:us-east-1:111122223333:secret:rds!x",
            "belongs to another account",
        ),
    ],
    ids=["service", "region", "account"],
)
def test_master_secret_reference_must_match_region_and_account(
    arn: str, message: str
) -> None:
    with pytest.raises(ReleaseError, match=message):
        REFRESH.aurora_master_secret_arn(encoded_arn(arn))


def snapshot(**overrides: Any) -> dict[str, Any]:
    runner = ObjectRunner(objects())
    captured = REFRESH.capture_aurora_refresh_snapshot(release(runner))
    captured.update(overrides)
    return captured


def test_snapshot_from_another_namespace_is_refused() -> None:
    instance = release(ObjectRunner(objects()))
    with pytest.raises(ReleaseError, match="namespace differs"):
        REFRESH.validate_aurora_refresh_snapshot(
            instance, {"namespace": "other", "objects": [], "absent": []}
        )


def test_snapshot_object_named_differently_is_refused() -> None:
    previous = snapshot()
    previous["objects"][0]["metadata"]["name"] = "someone-elses-refresher"
    with pytest.raises(ReleaseError, match="object identity differs"):
        REFRESH.validate_aurora_refresh_snapshot(
            release(ObjectRunner(objects())), previous
        )


def test_rolebinding_pointing_elsewhere_is_refused() -> None:
    previous = snapshot()
    binding = next(
        item for item in previous["objects"] if item["kind"] == "RoleBinding"
    )
    binding["roleRef"]["kind"] = "ClusterRole"
    with pytest.raises(ReleaseError, match="RBAC identity differs"):
        REFRESH.validate_aurora_refresh_snapshot(
            release(ObjectRunner(objects())), previous
        )


def absent_snapshot(*, verified: bool = True) -> dict[str, Any]:
    document: dict[str, Any] = {
        "namespace": NAMESPACE,
        "objects": [],
        "absent": [
            {"resource": resource, "name": REFRESH.CRONJOB_NAME}
            for resource in sorted(REFRESH.OBJECT_RESOURCES)
        ],
    }
    if verified:
        document["absence_verified"] = True
    return document


def test_absence_recorded_for_another_name_is_refused() -> None:
    previous = absent_snapshot()
    previous["absent"][0]["name"] = "other"
    with pytest.raises(ReleaseError, match="absence identity differs"):
        REFRESH.validate_aurora_refresh_snapshot(release(ObjectRunner([])), previous)


def test_cronjob_absence_must_have_been_verified_against_the_anchor() -> None:
    with pytest.raises(ReleaseError, match="absence is unverified"):
        REFRESH.validate_aurora_refresh_snapshot(
            release(ObjectRunner([])), absent_snapshot(verified=False)
        )


def test_apply_refuses_to_refresh_when_the_candidate_did_not_converge() -> None:
    runner = ObjectRunner(objects())
    runner.diff_code = 1
    instance = release(runner)
    with pytest.raises(ReleaseError, match="did not converge before refresh"):
        REFRESH.apply_aurora_refresh(instance)
    assert "refresh" not in runner.events, "no credential refresh on a drifting program"
    assert runner.events.count("apply") == 1


def test_verify_refuses_a_restored_program_that_still_differs() -> None:
    runner = ObjectRunner(objects())
    previous = REFRESH.capture_aurora_refresh_snapshot(release(runner))
    runner.diff_code = 1
    with pytest.raises(ReleaseError, match="refresher did not converge"):
        REFRESH.verify_aurora_refresh_snapshot(release(runner), previous)


def test_verify_refuses_a_recorded_absence_whose_object_still_exists() -> None:
    runner = ObjectRunner(objects())
    with pytest.raises(ReleaseError, match="absence did not converge"):
        REFRESH.verify_aurora_refresh_snapshot(release(runner), absent_snapshot())
    assert runner.events == ["read"], "the first lingering object stops the check"


def test_verify_accepts_a_confirmed_absence() -> None:
    runner = ObjectRunner([])
    REFRESH.verify_aurora_refresh_snapshot(release(runner), absent_snapshot())
    assert runner.events == ["read"] * len(REFRESH.OBJECT_RESOURCES)
    assert copy.deepcopy(runner.live) == {}


def test_snapshot_object_without_metadata_is_invalid() -> None:
    previous = snapshot()
    previous["objects"][0] = "not-an-object"
    with pytest.raises(ReleaseError, match="object is invalid"):
        REFRESH.validate_aurora_refresh_snapshot(
            release(ObjectRunner(objects())), previous
        )


def cronjob_pod(previous: dict[str, Any]) -> dict[str, Any]:
    cronjob = next(item for item in previous["objects"] if item["kind"] == "CronJob")
    pod: dict[str, Any] = cronjob["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    return pod


def test_cronjob_running_as_another_service_account_is_refused() -> None:
    previous = snapshot()
    cronjob_pod(previous)["serviceAccountName"] = "default"
    with pytest.raises(ReleaseError, match="program identity is incomplete or differs"):
        REFRESH.validate_aurora_refresh_snapshot(
            release(ObjectRunner(objects())), previous
        )


def test_cronjob_pointing_at_another_namespace_is_refused() -> None:
    previous = snapshot()
    env = cronjob_pod(previous)["containers"][0]["env"]
    entry = next(item for item in env if item["name"] == "GPU_FAULT_NAMESPACE")
    entry["value"] = "elsewhere"
    with pytest.raises(ReleaseError, match="program identity is incomplete or differs"):
        REFRESH.validate_aurora_refresh_snapshot(
            release(ObjectRunner(objects())), previous
        )
