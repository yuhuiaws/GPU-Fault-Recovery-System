from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from gpu_fault.aurora_credential_refresh import (
    RESTART_ANNOTATION,
    refresh_once,
    replace_password,
)

DSN = (
    "postgresql://gpu_fault_admin:oldpw@"
    "gpu-fault-regional-aurora.cluster-abc.us-west-2.rds.amazonaws.com"
    ":5432/gpu_fault?sslmode=require"
)
SECRET_ARN = (
    "arn:aws:secretsmanager:us-west-2:123456789012:secret:rds!cluster-1488f4cc-laXSTO"
)


def _encode(value: str) -> str:
    return base64.b64encode(value.encode()).decode()


class FakeCore:
    def __init__(self, data: dict[str, str]):
        self.data = dict(data)
        self.patches: list[dict] = []

    def read_namespaced_secret(self, name, namespace):
        self.name = name
        self.namespace = namespace
        return SimpleNamespace(data=dict(self.data))

    def patch_namespaced_secret(self, name, namespace, body):
        self.patches.append(body)
        self.data.update(body["data"])


class FakeApps:
    def __init__(self):
        self.patches: list[dict] = []

    def patch_namespaced_deployment(self, name, namespace, body):
        self.name = name
        self.patches.append(body)


def _refresh(
    core,
    apps,
    *,
    password="newpw",
    verify=None,
    expected_secret_arn=None,
    deployments=None,
):
    verified: list[str] = []

    def default_verify(dsn: str) -> None:
        verified.append(dsn)

    result = refresh_once(
        core,
        apps,
        namespace="gpu-fault-system",
        secret_name="gpu-fault-aurora",
        secret_key="postgres-url",
        deployment="gpu-fault-api-ha",
        deployments=deployments,
        region_name="us-west-2",
        timestamp="2026-08-04T12:00:00+00:00",
        expected_secret_arn=expected_secret_arn,
        verify=verify or default_verify,
        fetch_password=lambda arn, region: password,
    )
    return result, verified


def test_replace_password_preserves_everything_else() -> None:
    updated = replace_password(DSN, "s3cr#t/pw")
    assert updated.startswith("postgresql://gpu_fault_admin:")
    # Reserved characters must be percent-encoded or the DSN reparses wrong.
    assert "s3cr%23t%2Fpw" in updated
    assert updated.endswith(
        "@gpu-fault-regional-aurora.cluster-abc.us-west-2"
        ".rds.amazonaws.com:5432/gpu_fault?sslmode=require"
    )


def test_rotation_updates_secret_and_restarts_deployment() -> None:
    core = FakeCore(
        {"postgres-url": _encode(DSN), "master-secret-arn": _encode(SECRET_ARN)}
    )
    apps = FakeApps()
    result, verified = _refresh(core, apps)

    assert result.rotated is True
    assert result.restarted is True
    stored = base64.b64decode(core.data["postgres-url"]).decode()
    assert "newpw" in stored
    # The candidate is proven against the database before it is written.
    assert verified == [stored]
    assert apps.patches[0]["spec"]["template"]["metadata"]["annotations"] == {
        RESTART_ANNOTATION: "2026-08-04T12:00:00+00:00"
    }


def test_rotation_restarts_every_configured_consumer() -> None:
    core = FakeCore(
        {"postgres-url": _encode(DSN), "master-secret-arn": _encode(SECRET_ARN)}
    )
    apps = FakeApps()

    _refresh(core, apps, deployments=("gpu-fault-api-ha", "gpu-fault-active-executor"))

    assert len(apps.patches) == 2
    assert apps.name == "gpu-fault-active-executor"


def test_unchanged_password_touches_nothing() -> None:
    """Restarting replicas is the only disruptive step; skip it when idle."""
    core = FakeCore(
        {"postgres-url": _encode(DSN), "master-secret-arn": _encode(SECRET_ARN)}
    )
    apps = FakeApps()
    result, verified = _refresh(core, apps, password="oldpw")

    assert result.rotated is False
    assert result.restarted is False
    assert core.patches == []
    assert apps.patches == []
    # No connection is opened either: a no-op run must stay cheap.
    assert verified == []


def test_stale_master_secret_arn_is_repaired() -> None:
    current_arn = SECRET_ARN + "-current"
    core = FakeCore(
        {"postgres-url": _encode(DSN), "master-secret-arn": _encode(SECRET_ARN)}
    )
    apps = FakeApps()

    result, verified = _refresh(
        core, apps, password="newpw", expected_secret_arn=current_arn
    )

    assert result.rotated is True
    assert result.restarted is True
    assert base64.b64decode(core.data["master-secret-arn"]).decode() == current_arn
    assert len(verified) == 1


def test_failed_verification_leaves_secret_alone() -> None:
    """A refresher that cannot tell a good password from a bad one would turn
    a rotation into an outage of its own making."""
    core = FakeCore(
        {"postgres-url": _encode(DSN), "master-secret-arn": _encode(SECRET_ARN)}
    )
    apps = FakeApps()

    def reject(dsn: str) -> None:
        raise RuntimeError("password authentication failed")

    with pytest.raises(RuntimeError):
        _refresh(core, apps, verify=reject)

    assert core.patches == []
    assert apps.patches == []
    assert base64.b64decode(core.data["postgres-url"]).decode() == DSN


def test_self_managed_password_is_left_alone() -> None:
    """Without master-secret-arn there is no rotation to track."""
    core = FakeCore({"postgres-url": _encode(DSN)})
    apps = FakeApps()
    result, verified = _refresh(core, apps)

    assert result.rotated is False
    assert "master-secret-arn" in result.reason
    assert core.patches == []
    assert apps.patches == []


def test_missing_dsn_key_is_an_error() -> None:
    core = FakeCore({"master-secret-arn": _encode(SECRET_ARN)})
    with pytest.raises(RuntimeError, match="postgres-url"):
        _refresh(core, FakeApps())
