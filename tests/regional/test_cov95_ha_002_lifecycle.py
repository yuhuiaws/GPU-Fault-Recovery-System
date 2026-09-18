from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import run_ha001_control_plane_failover as common
from scripts.e2e.regional import run_ha002_pdb_topology as ha
from tests.regional._cov95_ha002_harness import HA002Harness


@pytest.mark.parametrize("spool", [0, 3])
def test_full_cordon_eviction_and_recovery_preserve_node_and_all_roles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, spool: int
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path, spool=spool)
    baseline = copy.deepcopy(harness.nodes["node-0"])
    code, report = harness.execute()
    assert code == 0, report.get("error", report.get("errors"))
    assert report["cleanup_errors"] == []
    assert report["spool_worker"]["status"] == ("TESTED" if spool else "NOT_APPLICABLE")
    assert len(harness.evictions) == (6 if spool else 4)
    for index, eviction in enumerate(harness.evictions):
        options = eviction["deleteOptions"]
        name = eviction["metadata"]["name"]
        assert options["preconditions"]["uid"] == harness.plan["pod_uids"][name]
        assert options.get("dryRun") == (["All"] if index % 2 else None)
    assert harness.nodes["node-0"]["spec"] == baseline["spec"]
    assert harness.nodes["node-0"]["metadata"]["annotations"] == {}
    assert len(harness.patches) == 2
    assert harness.patches[0][0] == {
        "op": "test",
        "path": "/metadata/uid",
        "value": baseline["metadata"]["uid"],
    }
    assert harness.watchdog.returncode == 0
    assert harness.watchdog.waits == [10]
    assert harness.events.index("watchdog-start") < harness.events.index("cordon")
    assert harness.events.index("uncordon") < harness.events.index("watchdog-stop")
    assert harness.resources == {}
    assert report["topology"]["worker_by_node"] == {
        "node-0": 2,
        "node-1": 2,
        "node-2": 2,
    }


@pytest.mark.parametrize("defect", ["uid", "taints", "ready", "owner", "cordon"])
def test_node_target_drift_blocks_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path)
    node = harness.nodes["node-0"]
    if defect == "uid":
        node["metadata"]["uid"] = "replacement"
    elif defect == "taints":
        node["spec"]["taints"] = [{"key": "foreign"}]
    elif defect == "ready":
        node["status"]["conditions"][0]["status"] = "False"
    elif defect == "owner":
        node["metadata"]["annotations"] = {ha.NODE_OWNER: "foreign"}
    else:
        node["spec"]["unschedulable"] = True
    with pytest.raises(ha.CaseError, match="target node drifted"):
        harness.execute()
    assert harness.events == []
    assert harness.patches == []


@pytest.mark.parametrize("defect", ["uid", "node", "app", "missing"])
def test_named_pod_target_drift_blocks_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path)
    name = harness.plan["target_pods"]["workers"][0]
    value = harness.pods[name]
    if defect == "uid":
        value["metadata"]["uid"] = "replacement"
    elif defect == "node":
        value["spec"]["nodeName"] = "node-1"
    elif defect == "app":
        value["metadata"]["labels"]["app"] = "foreign"
    else:
        del harness.pods[name]
    with pytest.raises(ha.CaseError, match="planned Pod drifted"):
        harness.execute()
    assert harness.evictions == []
    assert harness.events == []


@pytest.mark.parametrize("failure", ["ack-loss", "abort", "supervision-loss"])
def test_interruption_after_cordon_restores_only_when_supervision_is_proven(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path)

    def interrupt() -> None:
        if failure == "supervision-loss":
            raise ProcessSupervisionLost("unit supervision lost")
        raise RuntimeError(f"unit {failure}")

    harness.after_cordon = interrupt
    code, report = harness.execute()
    assert code == 1
    assert harness.evictions == []
    if failure == "supervision-loss":
        assert report["supervision_lost"] is True
        assert harness.events[-1] == "cordon"
        assert len(harness.resources) == 2
        assert harness.signals == []
    else:
        assert report["cleanup_errors"] == []
        assert harness.nodes["node-0"]["spec"]["unschedulable"] is False
        assert len(harness.patches) == 2
        assert harness.resources == {}
        assert harness.watchdog.returncode == 0


def test_node_replacement_is_not_uncordoned_and_cancels_stale_watchdog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path)

    def replace() -> None:
        harness.nodes["node-0"]["metadata"]["uid"] = "replacement"
        raise RuntimeError("unit node replaced")

    harness.after_cordon = replace
    code, report = harness.execute()
    assert code == 1
    assert report["node_identity_changed"] is True
    assert len(harness.patches) == 1
    assert harness.watchdog.returncode == 0
    assert harness.resources == {}


def test_failed_restoration_leaves_watchdog_armed_and_fails_postflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path)
    harness.eviction_failure = "unit API unavailable"
    harness.restore_failure = RuntimeError("unit restore unavailable")
    code, report = harness.execute()
    assert code == 1
    assert "Eviction failed" in report["error"]
    assert report["watchdog_left_armed"] is True
    assert harness.watchdog.returncode is None
    assert harness.signals == []
    assert any(
        "node remains cordoned" in error for error in report["cleanup_errors"]
    ), "failed scheduling restoration must remain visible"
    assert harness.resources == {}


def test_budget_reopening_prevents_the_second_eviction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path)
    harness.pdb_reopened = True
    code, report = harness.execute()
    assert code == 1
    assert "allows 1 disruption" in report["error"]
    assert len(harness.evictions) == 1
    assert harness.nodes["node-0"]["spec"]["unschedulable"] is False
    assert harness.watchdog.returncode == 0
    assert harness.resources == {}


def test_foreign_probe_preflight_never_deletes_or_cordons(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path)
    harness.resources["Pod", common.PROBE_POD] = {
        "metadata": {"name": common.PROBE_POD, "uid": "foreign"}
    }
    code, report = harness.execute()
    assert code == 1
    assert "probe resources already exist" in report["error"]
    assert harness.patches == []
    assert harness.evictions == []
    assert ("Pod", common.PROBE_POD) in harness.resources
    assert "stop-probe" not in harness.events


def test_missing_queue_baseline_refuses_before_probe_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path)
    path = tmp_path / "cases" / ha.CASE_ID / "plan.json"
    envelope = json.loads(path.read_text())
    del envelope["details"]["baseline"]["queue"]
    path.write_text(json.dumps(envelope))
    with pytest.raises(ha.CaseError, match="no observed queue baseline"):
        harness.execute()
    assert harness.events == []
