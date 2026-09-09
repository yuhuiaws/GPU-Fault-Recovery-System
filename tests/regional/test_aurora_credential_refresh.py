"""The credential refresher keeps the Secret current and reports how it went.

CP-3 (路 A): running Pods mount the ``gpu-fault-aurora`` Secret as files and
the pool re-reads the DSN on every connect, so a rotation no longer rolls any
Deployment. The refresher's job is now: bring ``postgres-url`` up to date
(verify first), and write ``last-refresh-status.json`` into the same Secret on
every run so the mount turns the outcome into a file the control plane can
export (H1-2). Rolling the Deployments survives only behind
``restart_deployments=True`` (the ``--restart-deployments`` switch, default
off) for a site that has not picked up the mount yet.
"""

from __future__ import annotations

import base64
import hashlib
import json
from types import SimpleNamespace

import pytest

from gpu_fault.aurora_credential_refresh import (
    RESTART_ANNOTATION,
    STATUS_KEY,
    refresh_once,
    replace_password,
    rollout_token_for,
    write_refresh_status,
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
    def __init__(
        self, data: dict[str, str], *, annotations: dict[str, str] | None = None
    ):
        self.data = dict(data)
        self.annotations = dict(annotations or {})
        self.patches: list[dict] = []

    def read_namespaced_secret(self, name, namespace):
        self.name = name
        self.namespace = namespace
        return SimpleNamespace(
            data=dict(self.data),
            metadata=SimpleNamespace(annotations=dict(self.annotations)),
        )

    def patch_namespaced_secret(self, name, namespace, body):
        self.patches.append(body)
        self.data.update(body["data"])
        self.annotations.update((body.get("metadata") or {}).get("annotations") or {})


class FakeApps:
    def __init__(self, *, fail_once: str | None = None):
        self.patches: list[dict] = []
        self.patched_names: list[str] = []
        self.annotations: dict[str, dict[str, str]] = {}
        self.fail_once = fail_once

    def read_namespaced_deployment(self, name, namespace):
        self.namespace = namespace
        annotations = self.annotations.setdefault(name, {})
        return SimpleNamespace(
            spec=SimpleNamespace(
                template=SimpleNamespace(
                    metadata=SimpleNamespace(annotations=dict(annotations))
                )
            )
        )

    def patch_namespaced_deployment(self, name, namespace, body):
        if self.fail_once == name:
            self.fail_once = None
            raise RuntimeError(f"temporary patch failure for {name}")
        self.name = name
        self.patched_names.append(name)
        self.patches.append(body)
        self.annotations.setdefault(name, {}).update(
            body["spec"]["template"]["metadata"]["annotations"]
        )


def _refresh(
    core,
    apps,
    *,
    password="newpw",
    verify=None,
    expected_secret_arn=None,
    deployments=None,
    timestamp="2026-08-04T12:00:00+00:00",
    restart_deployments=False,
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
        timestamp=timestamp,
        expected_secret_arn=expected_secret_arn,
        verify=verify or default_verify,
        fetch_password=lambda arn, region: password,
        restart_deployments=restart_deployments,
    )
    return result, verified


NEW_DSN = replace_password(DSN, "newpw")
NEW_TOKEN = hashlib.sha256(NEW_DSN.encode()).hexdigest()[:16]


def test_replace_password_preserves_everything_else() -> None:
    updated = replace_password(DSN, "s3cr#t/pw")
    assert updated.startswith("postgresql://gpu_fault_admin:"), (
        "the user must be preserved"
    )
    # Reserved characters must be percent-encoded or the DSN reparses wrong.
    assert "s3cr%23t%2Fpw" in updated
    assert updated.endswith(
        "@gpu-fault-regional-aurora.cluster-abc.us-west-2"
        ".rds.amazonaws.com:5432/gpu_fault?sslmode=require"
    ), "host, port, database and query must be preserved"


def test_rotation_updates_the_secret_and_leaves_every_deployment_alone() -> None:
    """Running Pods read the mounted Secret; a rotation is a Secret write."""

    core = FakeCore(
        {"postgres-url": _encode(DSN), "master-secret-arn": _encode(SECRET_ARN)}
    )
    apps = FakeApps()
    result, verified = _refresh(core, apps)

    assert result.rotated is True
    assert result.restarted is False
    stored = base64.b64decode(core.data["postgres-url"]).decode()
    assert stored == NEW_DSN
    # The candidate is proven against the database before it is written.
    assert verified == [stored]
    assert apps.patches == [], "no Deployment may be rolled for a rotation"
    assert "reload" in result.reason or "mounted" in result.reason


def test_the_rollout_token_is_derived_from_the_dsn_not_the_clock() -> None:
    """H1-4: two refreshers running concurrently (the :17 tick and a release
    preflight Job) used to stamp different timestamps and roll everything
    twice. A digest of the candidate DSN converges them."""

    core = FakeCore(
        {"postgres-url": _encode(DSN), "master-secret-arn": _encode(SECRET_ARN)}
    )
    _refresh(core, FakeApps(), timestamp="2026-08-04T12:00:00+00:00")
    first = core.annotations[RESTART_ANNOTATION]
    other = FakeCore(
        {"postgres-url": _encode(DSN), "master-secret-arn": _encode(SECRET_ARN)}
    )
    _refresh(other, FakeApps(), timestamp="2026-08-04T12:00:07+00:00")

    assert first == other.annotations[RESTART_ANNOTATION] == NEW_TOKEN
    assert rollout_token_for(NEW_DSN) == NEW_TOKEN
    assert len(NEW_TOKEN) == 16
    assert "newpw" not in NEW_TOKEN, "the annotation must not leak the password"


def test_rotation_restarts_every_configured_consumer_only_when_asked() -> None:
    """Compatibility for a site whose Pods do not mount the Secret yet."""

    core = FakeCore(
        {"postgres-url": _encode(DSN), "master-secret-arn": _encode(SECRET_ARN)}
    )
    apps = FakeApps()

    result, _ = _refresh(
        core,
        apps,
        deployments=("gpu-fault-api-ha", "gpu-fault-active-executor"),
        restart_deployments=True,
    )

    assert result.restarted is True
    assert apps.patched_names == ["gpu-fault-api-ha", "gpu-fault-active-executor"]
    assert apps.patches[0]["spec"]["template"]["metadata"]["annotations"] == {
        RESTART_ANNOTATION: NEW_TOKEN
    }


def test_partial_rollout_resumes_after_secret_write() -> None:
    targets = (
        "gpu-fault-api-ha",
        "gpu-fault-control-worker",
        "gpu-fault-telemetry-spool-worker",
    )
    core = FakeCore(
        {"postgres-url": _encode(DSN), "master-secret-arn": _encode(SECRET_ARN)}
    )
    apps = FakeApps(fail_once="gpu-fault-control-worker")

    with pytest.raises(RuntimeError, match="temporary patch failure"):
        _refresh(core, apps, deployments=targets, restart_deployments=True)

    stored = base64.b64decode(core.data["postgres-url"]).decode()
    rollout_token = core.annotations[RESTART_ANNOTATION]
    assert "newpw" in stored
    assert apps.patched_names == ["gpu-fault-api-ha"]
    assert len(core.patches) == 1

    result, verified = _refresh(
        core,
        apps,
        password="newpw",
        deployments=targets,
        timestamp="2026-08-04T12:05:00+00:00",
        restart_deployments=True,
    )

    assert result.rotated is False
    assert result.restarted is True
    assert result.reason == "deployment rollout resumed for current credentials"
    assert verified == []
    assert len(core.patches) == 1
    assert apps.patched_names == list(targets)
    assert {
        name: annotations[RESTART_ANNOTATION]
        for name, annotations in apps.annotations.items()
    } == {name: rollout_token for name in targets}


def test_current_secret_and_completed_rollout_are_noop() -> None:
    targets = ("gpu-fault-api-ha", "gpu-fault-control-worker")
    core = FakeCore(
        {"postgres-url": _encode(DSN), "master-secret-arn": _encode(SECRET_ARN)}
    )
    apps = FakeApps()
    _refresh(core, apps, deployments=targets, restart_deployments=True)
    secret_patches = len(core.patches)
    deployment_patches = len(apps.patches)

    result, verified = _refresh(
        core,
        apps,
        password="newpw",
        deployments=targets,
        timestamp="2026-08-04T12:05:00+00:00",
        restart_deployments=True,
    )

    assert result.rotated is False
    assert result.restarted is False
    assert "already carry current credentials" in result.reason
    assert verified == []
    assert len(core.patches) == secret_patches
    assert len(apps.patches) == deployment_patches


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
    assert result.restarted is False
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


def test_a_secret_with_a_leftover_rollout_token_is_a_noop_without_restarts() -> None:
    """A site upgraded from the restarting refresher carries the annotation
    from its last rotation; with restarts off there is nothing to resume."""

    core = FakeCore(
        {"postgres-url": _encode(NEW_DSN), "master-secret-arn": _encode(SECRET_ARN)},
        annotations={RESTART_ANNOTATION: "2026-08-04T12:00:00+00:00"},
    )
    apps = FakeApps()

    result, verified = _refresh(core, apps, password="newpw")

    assert result.rotated is False
    assert result.restarted is False
    assert core.patches == []
    assert apps.patches == []
    assert verified == []


# --- H1-2: every run leaves its outcome in the Secret -------------------------


def test_write_refresh_status_lands_in_the_secret_as_json() -> None:
    core = FakeCore(
        {"postgres-url": _encode(DSN), "master-secret-arn": _encode(SECRET_ARN)}
    )

    write_refresh_status(
        core,
        namespace="gpu-fault-system",
        secret_name="gpu-fault-aurora",
        status="ok",
        finished_at="2026-08-04T12:00:03+00:00",
        error=None,
        rotated=True,
        restarted=False,
        reason="credentials updated",
    )

    assert STATUS_KEY == "last-refresh-status.json"
    assert len(core.patches) == 1
    assert set(core.patches[0]["data"]) == {STATUS_KEY}, (
        "the status write must not touch postgres-url or master-secret-arn"
    )
    payload = json.loads(base64.b64decode(core.data[STATUS_KEY]).decode())
    assert payload == {
        "status": "ok",
        "finished_at": "2026-08-04T12:00:03+00:00",
        "error": None,
        "rotated": True,
        "restarted": False,
        "reason": "credentials updated",
    }


def test_write_refresh_status_records_a_failure_without_the_password() -> None:
    core = FakeCore(
        {"postgres-url": _encode(DSN), "master-secret-arn": _encode(SECRET_ARN)}
    )

    write_refresh_status(
        core,
        namespace="gpu-fault-system",
        secret_name="gpu-fault-aurora",
        status="failed",
        finished_at="2026-08-04T12:00:03+00:00",
        error=f"OperationalError: FATAL: password authentication failed {NEW_DSN}",
        rotated=False,
        restarted=False,
        reason=None,
    )

    payload = json.loads(base64.b64decode(core.data[STATUS_KEY]).decode())
    assert payload["status"] == "failed"
    assert "password authentication failed" in payload["error"]
    assert "newpw" not in payload["error"], "the error text must redact DSNs"


def test_main_writes_ok_status_after_a_converged_run(monkeypatch) -> None:
    from gpu_fault import aurora_credential_refresh as module

    core = FakeCore(
        {"postgres-url": _encode(DSN), "master-secret-arn": _encode(SECRET_ARN)}
    )
    apps = FakeApps()
    _install_main_doubles(monkeypatch, module, core, apps, password="oldpw")

    module.main([])

    payload = json.loads(base64.b64decode(core.data[STATUS_KEY]).decode())
    assert payload["status"] == "ok"
    assert payload["rotated"] is False and payload["restarted"] is False
    assert payload["error"] is None
    assert payload["finished_at"].endswith("+00:00"), (
        "finished_at must be timezone-aware UTC"
    )
    assert apps.patches == []


def test_main_writes_failed_status_and_still_raises(monkeypatch) -> None:
    from gpu_fault import aurora_credential_refresh as module

    core = FakeCore(
        {"postgres-url": _encode(DSN), "master-secret-arn": _encode(SECRET_ARN)}
    )
    apps = FakeApps()
    events: list[dict] = []
    core.create_namespaced_event = lambda namespace, body: events.append(body)

    def refuse(dsn: str) -> None:
        raise RuntimeError("password authentication failed")

    _install_main_doubles(
        monkeypatch, module, core, apps, password="newpw", verify=refuse
    )
    monkeypatch.setenv("GPU_FAULT_AURORA_REFRESH_MAX_ATTEMPTS", "1")

    with pytest.raises(RuntimeError, match="failed after retries"):
        module.main([])

    payload = json.loads(base64.b64decode(core.data[STATUS_KEY]).decode())
    assert payload["status"] == "failed"
    assert "password authentication failed" in payload["error"]
    assert events and events[0]["reason"] == "AuroraCredentialRefreshFailed"
    assert base64.b64decode(core.data["postgres-url"]).decode() == DSN


def test_main_restarts_only_with_the_switch(monkeypatch) -> None:
    from gpu_fault import aurora_credential_refresh as module

    core = FakeCore(
        {"postgres-url": _encode(DSN), "master-secret-arn": _encode(SECRET_ARN)}
    )
    apps = FakeApps()
    _install_main_doubles(monkeypatch, module, core, apps, password="newpw")

    module.main(["--restart-deployments"])

    assert apps.patched_names == ["gpu-fault-api-ha"]
    payload = json.loads(base64.b64decode(core.data[STATUS_KEY]).decode())
    assert payload["rotated"] is True and payload["restarted"] is True


def _install_main_doubles(monkeypatch, module, core, apps, *, password, verify=None):
    """Point ``main()`` at the fakes: no kube config, no AWS, no database."""

    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.delenv("GPU_FAULT_AURORA_REFRESH_RESTART_DEPLOYMENTS", raising=False)
    monkeypatch.delenv("GPU_FAULT_AURORA_RESTART_DEPLOYMENTS", raising=False)
    monkeypatch.setattr(module, "configure_logging", lambda: None)
    monkeypatch.setattr(module, "load_kubernetes_clients", lambda: (core, apps))
    monkeypatch.setattr(module, "read_current_password", lambda arn, region: password)
    monkeypatch.setattr(module, "verify_dsn", verify or (lambda dsn: None))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
