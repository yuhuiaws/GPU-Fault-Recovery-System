from __future__ import annotations

import json
import subprocess
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest

from gpu_fault.admin import cluster_readiness as readiness
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site, materialized_release_config
from gpu_fault.collector_registry import COLLECTOR_KINDS
from gpu_fault_release import rollout
from gpu_fault_release.regional_release_config import ReleaseError
from tests.admin.test_admin_site import site_file

NODES = ["node-a", "node-b"]
NODE_COLLECTOR_KINDS = tuple(
    kind.value
    for kind, spec in COLLECTOR_KINDS.items()
    if spec.systemd_unit is not None
)


def _report(*, missing: str | None = None) -> dict:
    return {
        "cluster_id": "gpu-a",
        "ready": missing is None,
        "nodes": [
            {
                "node_id": node,
                "ready": missing is None,
                "collectors": {
                    kind.value: {
                        "ready": kind.value != missing,
                        "last_success_at": None if kind.value == missing else "fresh",
                        "unit_state": "active",
                    }
                    for kind in COLLECTOR_KINDS
                    if kind.value in NODE_COLLECTOR_KINDS
                },
            }
            for node in NODES
        ],
    }


@pytest.fixture
def clock(monkeypatch):
    value = [100.0]
    monkeypatch.setattr(readiness.time, "monotonic", lambda: value[0])
    monkeypatch.setattr(
        readiness.time,
        "sleep",
        lambda seconds: value.__setitem__(0, value[0] + seconds),
    )
    return value


@pytest.fixture
def probe(tmp_path, monkeypatch):
    state = SimpleNamespace(
        site=load_site(site_file(tmp_path)), document=_report, calls=[], reports=[]
    )

    def run(arguments, **options):
        state.calls.append((list(arguments), options))
        assert 0 < options["timeout_seconds"] <= 30, (
            "readiness requests must retain their bounded supervised command budget"
        )
        if "exec" in arguments:
            document = state.document()
            state.reports.append(deepcopy(document))
            output = json.dumps(document)
        else:
            output = "cpu-pod"
        return subprocess.CompletedProcess(arguments, 0, output, "")

    monkeypatch.setattr(readiness, "run_command", run)
    return state


def test_complete_reports_use_the_supervised_cpu_readiness_client(probe) -> None:
    result = readiness.wait_collector_readiness(probe.site, "gpu-a")

    assert result == _report()
    assert len(probe.calls) == 2
    discovery, _options = probe.calls[0]
    assert discovery[discovery.index("-l") + 1] == "app=gpu-fault-api-ha"
    assert all(
        arguments[arguments.index("--kubeconfig") + 1]
        == probe.site.release_config["cpu_kubeconfig"]
        for arguments, _options in probe.calls
    ), "collector readiness must query only the CPU control plane"


@pytest.mark.parametrize("kind", NODE_COLLECTOR_KINDS)
def test_join_waits_for_every_collector_report(probe, clock, monkeypatch, kind) -> None:
    monkeypatch.setenv("GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS", "900")
    reports = iter([_report(missing=kind), _report()])
    probe.document = lambda: next(reports)

    result = readiness.wait_collector_readiness(
        probe.site, "gpu-a", timeout_seconds=10, interval_seconds=1
    )

    assert probe.reports == [_report(missing=kind), _report()], (
        f"join must wait for {kind} even when its unit is active or cadence is slow"
    )
    assert result == _report()
    assert clock[0] == 101.0


@pytest.mark.parametrize(
    "failure",
    [
        "non-object",
        "foreign-cluster",
        "string-ready",
        "missing-nodes",
        "empty-nodes",
        "unready-node",
        "malformed-node",
    ],
)
def test_readiness_refuses_missing_or_ambiguous_evidence(probe, clock, failure) -> None:
    report = _report()
    if failure == "non-object":
        report = []
    elif failure == "foreign-cluster":
        report["cluster_id"] = "foreign"
    elif failure == "string-ready":
        report["ready"] = "true"
    elif failure == "missing-nodes":
        report.pop("nodes")
    elif failure == "empty-nodes":
        report["nodes"] = []
    elif failure == "unready-node":
        report["nodes"][0]["ready"] = False
    else:
        report["nodes"][0] = None
    probe.document = lambda: report

    with pytest.raises(BootstrapError, match="did not become ready"):
        readiness.wait_collector_readiness(
            probe.site, "gpu-a", timeout_seconds=3, interval_seconds=1
        )
    assert len(probe.reports) == 3, (
        f"ambiguous evidence must never produce an early success: {failure}"
    )
    assert clock[0] == 103.0


@pytest.mark.parametrize("late_success", [False, True])
def test_readiness_deadline_cannot_be_refreshed(probe, clock, late_success) -> None:
    calls = []

    def report():
        calls.append(clock[0])
        if late_success:
            clock[0] = 111
            return _report()
        raise BootstrapError("readiness unavailable")

    probe.document = report
    with pytest.raises(BootstrapError, match="did not become ready"):
        readiness.wait_collector_readiness(
            probe.site, "gpu-a", timeout_seconds=10, interval_seconds=3
        )
    assert len(calls) == (1 if late_success else 4)
    assert clock[0] == (111 if late_success else 110)


@pytest.mark.parametrize("healthy", [False, True])
def test_verify_dispatch_preserves_the_complete_engine_report(
    tmp_path, monkeypatch, capsys, healthy
) -> None:
    site = load_site(site_file(tmp_path))
    baseline = {"previous": {"release_id": "previous-release"}}
    applied = []
    release = SimpleNamespace(
        _load_state=lambda: baseline, _apply_health_baseline=applied.append
    )
    monkeypatch.setattr(rollout, "RegionalRelease", lambda *_args: release)
    monkeypatch.setattr(
        rollout, "ReleaseKubeconfigCache", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(rollout, "deployment_api_budget", nullcontext)
    calls = []
    report = {
        "healthy": healthy,
        "summary": {"PASS": 12, "FAIL": int(not healthy)},
        "checks": [
            {"name": "monitoring"},
            {"name": "control_api"},
            {"name": "gpu_cluster:gpu-a"},
            {"name": "gpu_cluster:gpu-b"},
        ],
    }

    def verify(actual, *, mode):
        assert actual is release
        calls.append(mode)
        return deepcopy(report)

    monkeypatch.setattr(rollout, "build_health_report", verify)
    with materialized_release_config(site) as config:
        monkeypatch.setattr(
            rollout.sys, "argv", ["rollout", "verify", "--config", str(config)]
        )
        assert rollout.main() == (0 if healthy else 1)

    assert json.loads(capsys.readouterr().out) == report
    assert calls == ["verify"]
    assert applied == [baseline]


@pytest.mark.parametrize("schema_version", [3, 4])
@pytest.mark.parametrize("failure", [None, "image drift", "Agent set mismatch"])
def test_membership_sync_requires_full_capture_before_commit(
    failure, schema_version
) -> None:
    events = []
    document = {
        "live_runtime_image": "cpu-image",
        "runtime_image": "cpu-image",
        "executor_image": "different-executor-image",
        "node_dependencies": {"reference": "previous-dependencies"},
        "clusters": {"gpu-a": {}, "gpu-b": {}},
    }
    original = deepcopy(document)
    committed = []

    class Release:
        config = SimpleNamespace(
            clusters=[
                SimpleNamespace(cluster_id="gpu-a"),
                SimpleNamespace(cluster_id="gpu-b"),
            ],
            release_manifest_schema_version=schema_version,
        )

        def _capture_previous(self, plan=None):
            events.append(("capture", plan))
            if failure:
                raise ReleaseError(failure)
            return document

        def _save_state(self, phase, **values):
            events.append(("save", phase))
            committed.append((phase, values))

    release = Release()
    if failure:
        with pytest.raises(ReleaseError, match=failure):
            rollout.sync_release_state(release)
        assert committed == [], "failed full capture must not commit release state"
        assert events == [("capture", None)]
    else:
        rollout.sync_release_state(release)
        assert events == [("capture", None), ("save", "complete")], (
            "membership sync must use one full-site capture before committing"
        )
        assert committed[0][1]["completed_cluster_ids"] == ["gpu-a", "gpu-b"]
        assert committed[0][1]["adopted_live_runtime_image"] == (
            "cpu-image" if schema_version == 3 else None
        ), "schema v4 must not collapse separate CPU and Executor image identities"
        assert committed[0][1]["transaction_committed"] is True
    assert document == original
