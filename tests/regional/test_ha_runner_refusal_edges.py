"""Refusal and cleanup edges of the HA-005/HA-006/HA-009 runners.

Every path here is a run that stops early: a preflight residual, a managed
secret without a current version, a CPU baseline that is not steady, probe
resources that never earned an ownership receipt. The runners must record the
refusal in their canonical result and must not reach a later mutation.
"""

from __future__ import annotations

import argparse
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from scripts.e2e.regional import ha009_refresh
from scripts.e2e.regional import run_ha005_rollout_continuity as ha005
from scripts.e2e.regional import run_ha006_executor_takeover as ha006
from scripts.e2e.regional import run_ha009_aurora_credential_rotation as ha009

DEADLINE = datetime(2099, 1, 1, tzinfo=timezone.utc)
PAST = datetime(2000, 1, 1, tzinfo=timezone.utc)
SECRET_ARN = "arn:aws:secretsmanager:us-west-2:123456789012:secret:unit-aurora"


class FakeResources:
    """Owned-resource ledger whose deletes are recorded, never issued."""

    def __init__(self, *_arguments: Any, failing: set[str] | None = None) -> None:
        self.deleted: list[tuple[str, str]] = []
        self.records: dict[str, dict[str, Any]] = {}
        self.failing = failing or set()

    def delete(self, kind: str, name: str) -> None:
        if kind in self.failing:
            raise RuntimeError(f"{kind} delete refused")
        self.deleted.append((kind, name))

    def owned(self, kind: str, name: str) -> dict[str, Any] | None:
        return self.records.get(f"{kind}/{name}")


def sequence(*values: dict[str, Any]) -> Callable[[], dict[str, Any]]:
    """Return successive values, then keep returning the last one."""

    remaining = list(values)

    def read() -> dict[str, Any]:
        if len(remaining) > 1:
            return remaining.pop(0)
        return remaining[0]

    return read


def read_report(run_dir: Path, case_id: str) -> dict[str, Any]:
    return json.loads(
        (run_dir / "cases" / case_id / f"{case_id}.json").read_text(encoding="utf-8")
    )


def ha009_residuals(
    monkeypatch: pytest.MonkeyPatch,
    *,
    registry: tuple[dict[str, Any], ...] = ({"count": 0},),
    kubernetes: tuple[dict[str, Any], ...] = ({"count": 0},),
) -> None:
    monkeypatch.setattr(ha009.BASE, "require_window", lambda *a, **k: None)
    monkeypatch.setattr(ha009.BASE, "database_residuals", lambda: {"total": 0})
    monkeypatch.setattr(ha009.BASE, "registry_residuals", sequence(*registry))
    monkeypatch.setattr(ha009.BASE, "kubernetes_residuals", sequence(*kubernetes))


@pytest.mark.parametrize(
    ("registry", "kubernetes", "fragment"),
    [
        ({"count": 3}, {"count": 0}, "registry preflight residuals"),
        ({"count": 0}, {"count": 1}, "Kubernetes preflight residuals"),
    ],
)
def test_ha009_preflight_residuals_refuse_before_the_binding_is_read(
    registry: dict[str, Any],
    kubernetes: dict[str, Any],
    fragment: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ha009_residuals(monkeypatch, registry=(registry,), kubernetes=(kubernetes,))
    reads: list[str] = []
    monkeypatch.setattr(
        ha009,
        "aurora_guard",
        lambda: SimpleNamespace(read=lambda *a: reads.append("x")),
    )
    monkeypatch.setattr(
        ha009.BASE, "teardown", lambda *a, **k: reads.append("teardown")
    )

    assert ha009.run_case(tmp_path, 1, DEADLINE) == 1
    report = read_report(tmp_path, ha009.CASE_ID)
    assert report["verdict"] == "FAIL"
    assert fragment in report["error"], report
    assert reads == [], "a residual preflight must stop before any binding read"
    assert "cleanup_errors" not in report, "cleanup must not be armed by a preflight"
    capsys.readouterr()


def arm_ha009_until_secret_versions(
    monkeypatch: pytest.MonkeyPatch, resources: FakeResources | None
) -> dict[str, Any]:
    """Patch the live seams up to the managed-secret read; return the call log."""

    calls: dict[str, Any] = {"teardown": 0}
    monkeypatch.setattr(
        ha009,
        "aurora_guard",
        lambda: SimpleNamespace(
            read=lambda *a: {
                "identity": {"database": {"master_secret_arn": SECRET_ARN}}
            }
        ),
    )
    monkeypatch.setattr(ha009.BASE, "OwnedProbeResources", lambda *a, **k: resources)
    monkeypatch.setattr(ha009, "pool_max_idle_seconds", lambda: 300)
    monkeypatch.setattr(ha009, "secret_versions", lambda arn: {"stages": {}})

    def teardown(**_keywords: Any) -> None:
        calls["teardown"] += 1

    monkeypatch.setattr(ha009.BASE, "teardown", teardown)
    monkeypatch.setattr(
        ha009,
        "deployment_snapshot",
        lambda: {
            name: {"ready": 1, "replicas": 1, "pods": []} for name in ha009.DEPLOYMENTS
        },
    )
    return calls


@pytest.mark.parametrize(
    ("registry", "kubernetes", "fragment"),
    [
        (({"count": 0}, {"count": 2}), ({"count": 0},), "registry residuals"),
        (({"count": 0},), ({"count": 0}, {"count": 5}), "Kubernetes residuals"),
    ],
)
def test_ha009_missing_current_secret_version_fails_and_postflight_residuals_fail(
    registry: tuple[dict[str, Any], ...],
    kubernetes: tuple[dict[str, Any], ...],
    fragment: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ha009_residuals(monkeypatch, registry=registry, kubernetes=kubernetes)
    resources = FakeResources()
    calls = arm_ha009_until_secret_versions(monkeypatch, resources)

    assert ha009.run_case(tmp_path, 1, DEADLINE) == 1
    report = read_report(tmp_path, ha009.CASE_ID)
    assert "managed secret has no AWSCURRENT version" in report["error"], report
    assert resources.deleted == [
        ("Pod", ha009.BASE.POD),
        ("ConfigMap", ha009.BASE.CONFIGMAP),
    ]
    assert calls["teardown"] == 1, "an armed case must tear its registry down"
    assert fragment in report["postflight_error"], report
    assert report["verdict"] == "FAIL"
    capsys.readouterr()


def test_ha009_postflight_requires_every_deployment_fully_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ha009_residuals(monkeypatch)
    arm_ha009_until_secret_versions(monkeypatch, FakeResources())
    degraded = {
        name: {"ready": 1, "replicas": 1, "pods": []} for name in ha009.DEPLOYMENTS
    }
    degraded[ha009.DEPLOYMENTS[-1]]["ready"] = 0
    monkeypatch.setattr(ha009, "deployment_snapshot", lambda: degraded)

    assert ha009.run_case(tmp_path, 1, DEADLINE) == 1
    report = read_report(tmp_path, ha009.CASE_ID)
    assert report["postflight_error"] == (
        f"CaseError: {ha009.DEPLOYMENTS[-1]} is not fully Ready"
    )
    assert report["postflight"]["deployments"] == degraded
    capsys.readouterr()


def test_ha009_cleanup_without_an_ownership_receipt_preserves_everything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ha009_residuals(monkeypatch)
    calls = arm_ha009_until_secret_versions(monkeypatch, None)

    assert ha009.run_case(tmp_path, 1, DEADLINE) == 1
    report = read_report(tmp_path, ha009.CASE_ID)
    assert report["cleanup_preserved"] == "probe resources have no ownership receipt"
    assert calls["teardown"] == 0, "no receipt means no registry teardown"
    assert "postflight" not in report
    capsys.readouterr()


def test_ha009_unsteady_cpu_baseline_stops_before_rotation_and_keeps_the_probe_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ha009_residuals(monkeypatch)
    resources = FakeResources()
    arm_ha009_until_secret_versions(monkeypatch, resources)
    monkeypatch.setattr(
        ha009, "secret_versions", lambda arn: {"stages": {"AWSCURRENT": "v-before"}}
    )
    monkeypatch.setattr(ha009, "kubernetes_secret_digest", lambda: "digest-before")
    monkeypatch.setattr(ha009.BASE, "register", lambda *a, **k: None)
    monkeypatch.setattr(
        ha009.BASE, "executor_identity", lambda **k: {"service_account": "executor"}
    )
    deployment = {
        "spec": {"template": {"spec": {"containers": [{"image": "img@sha256:ab"}]}}}
    }
    dataplane_calls: list[str] = []

    def dataplane(*arguments: str, **_keywords: Any) -> str:
        dataplane_calls.append(arguments[0])
        if arguments[0] == "get":
            return json.dumps(deployment)
        return "probe log line"

    monkeypatch.setattr(ha009.BASE, "dataplane", dataplane)
    created: list[str] = []
    monkeypatch.setattr(ha009, "create_probe", lambda image, *a: created.append(image))
    ledger = SimpleNamespace(started=0, stopped=0)
    ledger.start = lambda: setattr(ledger, "started", ledger.started + 1)
    ledger.stop = lambda: setattr(ledger, "stopped", ledger.stopped + 1)
    monkeypatch.setattr(ha009.BASE, "ReceiptLedger", lambda *a, **k: ledger)
    monkeypatch.setattr(ha009.BASE, "wait_probe_samples", lambda **k: {"samples": 4})
    monkeypatch.setattr(
        ha009, "deployments_steady", lambda before, after: ["ingress is rolling"]
    )
    rotations: list[str] = []
    monkeypatch.setattr(ha009, "request_rotation", lambda state: rotations.append("x"))

    assert ha009.run_case(tmp_path, 1, DEADLINE) == 1
    report = read_report(tmp_path, ha009.CASE_ID)
    assert (
        report["error"] == "CaseError: CPU baseline is incomplete: ingress is rolling"
    )
    assert rotations == [], "an unsteady baseline must never request a rotation"
    assert created == ["img@sha256:ab"]
    assert (ledger.started, ledger.stopped) == (1, 1)
    assert (tmp_path / "cases" / ha009.CASE_ID / "probe.log").read_text() == (
        "probe log line"
    )
    assert dataplane_calls.count("exec") == 1, "the probe is asked to stop once"
    assert resources.deleted[0] == ("Pod", ha009.BASE.POD)
    capsys.readouterr()


def test_ha009_configure_requires_every_identity_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("RDS_CLUSTER_ID", "AWS_REGION", "CRONJOB", "SECRET_NAME"):
        monkeypatch.setattr(ha009, name, getattr(ha009, name))
    for name in (
        "GPU_FAULT_AURORA_CLUSTER_ID",
        "GPU_FAULT_PERF_AWS_REGION",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    ):
        monkeypatch.delenv(name, raising=False)
    arguments = argparse.Namespace(
        rds_cluster_id="", refresh_cronjob="refresh", aurora_secret_name="secret"
    )
    with pytest.raises(ha009_refresh.CaseError, match="are required"):
        ha009.configure(arguments)
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    arguments.rds_cluster_id = "aurora-unit"
    ha009.configure(arguments)
    assert (ha009.RDS_CLUSTER_ID, ha009.AWS_REGION) == ("aurora-unit", "us-west-2")


def test_ha005_continuity_rejects_negative_counters_and_missing_receipts() -> None:
    probe = {
        "counters": {
            "event_attempts": -1,
            "event_accepted": 0,
            "event_failures": 0,
            "event_buffered": 0,
            "claim_success": 1,
        },
        "outbox": {"records": 0, "replayable": 0},
        "attempted_events": [],
    }
    errors = ha005.continuity_errors(
        probe, {"requests": [], "missing": ["r1"]}, accepted_ids=[]
    )
    assert "event counters are invalid" in errors
    assert "accepted processor request IDs are missing" in errors


def test_ha005_telemetry_replay_waits_until_the_receipt_converges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verdicts = [["spool not drained"], []]
    receipts = [{"batch_id": "b1"}, {"batch_id": "b2"}]
    monkeypatch.setattr(
        ha005, "telemetry_replay_receipt", lambda probe: receipts.pop(0)
    )
    monkeypatch.setattr(
        ha005, "telemetry_replay_errors", lambda probe, receipt: verdicts.pop(0)
    )
    sleeps: list[float] = []
    monkeypatch.setattr(ha005.time, "sleep", lambda seconds: sleeps.append(seconds))

    assert ha005.wait_telemetry_replay({"run_id": "r"}) == {"batch_id": "b2"}
    assert sleeps == [2], "one failed read means exactly one pause"


def test_ha005_telemetry_replay_fails_at_the_deadline_with_the_last_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ha005, "telemetry_replay_receipt", lambda probe: {})
    monkeypatch.setattr(
        ha005, "telemetry_replay_errors", lambda probe, receipt: ["summary not bound"]
    )
    with pytest.raises(ha005.CaseError, match="summary not bound"):
        ha005.wait_telemetry_replay({"run_id": "r"}, timeout_seconds=0)


def test_ha005_receipt_poller_does_not_collect_after_a_stop_raced_its_read() -> None:
    reading = threading.Event()
    release = threading.Event()
    collected: list[list[str]] = []

    def read_accepted_ids() -> list[str]:
        reading.set()
        release.wait(timeout=10)
        return ["r1"]

    ledger = ha005.ReceiptLedger(read_accepted_ids, interval_seconds=0.01)
    ledger.collect = lambda ids: collected.append(ids)  # type: ignore[method-assign]
    ledger.start()
    assert reading.wait(timeout=10), "the poller must start reading"
    stopper = threading.Thread(target=ledger.stop)
    stopper.start()
    stopper.join(timeout=0.2)
    assert stopper.is_alive(), "stop waits for the blocked read before returning"
    release.set()
    stopper.join(timeout=10)
    assert not stopper.is_alive(), "stop must return once the read is released"
    assert collected == [], "ids read while stopping must not be collected"
    assert ledger.polls == 0


def test_ha005_case_without_an_ownership_receipt_preserves_the_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ha005, "require_window", lambda *a, **k: None)
    monkeypatch.setattr(ha005, "database_residuals", lambda: {"total": 0})
    monkeypatch.setattr(ha005, "registry_residuals", lambda: {"count": 0})
    monkeypatch.setattr(ha005, "kubernetes_residuals", lambda: {"count": 0})
    monkeypatch.setattr(ha005, "OwnedProbeResources", lambda *a, **k: None)
    teardowns: list[str] = []
    monkeypatch.setattr(ha005, "teardown", lambda **k: teardowns.append("x"))

    def register(*_arguments: Any, **_keywords: Any) -> None:
        raise RuntimeError("registry is read-only in this unit")

    monkeypatch.setattr(ha005, "register", register)

    assert ha005.run_case(tmp_path, 1, DEADLINE, all_deployments=True) == 1
    report = read_report(tmp_path, ha005.CASE_ID)
    assert report["error"] == "RuntimeError: registry is read-only in this unit"
    assert report["cleanup_preserved"] == "probe resources have no ownership receipt"
    assert report["coverage_scope"] == "all-enabled-roles"
    assert teardowns == [], "without a receipt nothing may be torn down"
    capsys.readouterr()


def test_ha006_parse_timestamp_assumes_utc_for_naive_values() -> None:
    parsed = ha006.parse_timestamp("2026-01-02T03:04:05")
    assert parsed == datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert ha006.parse_timestamp("") is None


def test_ha006_refuses_a_closed_maintenance_window_before_touching_anything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reads: list[str] = []
    monkeypatch.setattr(
        ha006, "database_residuals", lambda: reads.append("db") or {"total": 0}
    )
    with pytest.raises(ha006.CaseError, match="maintenance window has ended"):
        ha006.run_case(tmp_path, 1, PAST)
    assert reads == []
    assert not (tmp_path / "cases").exists(), (
        "no case directory before the window check"
    )


@pytest.mark.parametrize(
    ("registry", "kubernetes", "fragment"),
    [
        ({"count": 4}, {"count": 0}, "registry preflight residuals"),
        ({"count": 0}, {"count": 1}, "Kubernetes preflight residuals"),
    ],
)
def test_ha006_preflight_residuals_refuse_before_resources_are_owned(
    registry: dict[str, Any],
    kubernetes: dict[str, Any],
    fragment: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(ha006, "database_residuals", lambda: {"total": 0})
    monkeypatch.setattr(ha006, "registry_residuals", lambda: registry)
    monkeypatch.setattr(ha006, "kubernetes_residuals", lambda: kubernetes)
    owned: list[str] = []
    monkeypatch.setattr(
        ha006, "OwnedProbeResources", lambda *a, **k: owned.append("resources")
    )
    monkeypatch.setattr(ha006, "teardown", lambda **k: owned.append("teardown"))

    assert ha006.run_case(tmp_path, 1, DEADLINE) == 1
    report = read_report(tmp_path, ha006.CASE_ID)
    assert fragment in report["error"], report
    assert owned == [], "a residual preflight must stop before owning resources"
    capsys.readouterr()


def test_ha006_cleanup_without_a_receipt_preserves_everything(tmp_path: Path) -> None:
    result: dict[str, Any] = {"verdict": "PASS"}
    ha006.cleanup_case(tmp_path, "ha006-unit", {}, None, result, resources=None)
    assert result == {
        "verdict": "FAIL",
        "cleanup_preserved": "probe resources have no ownership receipt",
    }


def test_ha006_cleanup_keeps_the_registry_when_a_probe_pod_delete_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resources = FakeResources(failing={"Pod"})
    for pod in ha006.PODS:
        resources.records[f"Pod/{pod}"] = {"metadata": {"name": pod}}
    monkeypatch.setattr(ha006, "dataplane", lambda *a, **k: "executor log")
    teardowns: list[str] = []
    monkeypatch.setattr(ha006, "teardown", lambda **k: teardowns.append("x"))
    result: dict[str, Any] = {"verdict": "PASS"}

    ha006.cleanup_case(tmp_path, "ha006-unit", {}, None, result, resources=resources)

    assert result["verdict"] == "FAIL"
    assert result["cleanup_preserved"] == (
        "probe shutdown unverified; retain registry and rows"
    )
    assert [error.split(":")[0] for error in result["cleanup_errors"]] == [
        f"delete {pod}" for pod in ha006.PODS
    ]
    assert resources.deleted == [], "a refused delete must not be recorded as done"
    assert teardowns == [], "registry teardown waits for every probe Pod to go"
    for pod in ha006.PODS:
        assert (tmp_path / f"{pod}.log").read_text() == "executor log"
