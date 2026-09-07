"""DESTR-003's cleanup ownership, incident recovery and preflight fan-out.

Three review findings, each pinned by one live failure shape: a validated
restore that was skipped in silence because the node's owner had moved to a
successor incident; a cleanup that did nothing at all because the workflow wait
timed out before an ``incident_id`` was learned; and a read-only preflight that
ran fifteen independent subprocesses one after another.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr003_warm_spare_failover as destr003
from scripts.e2e.regional.regional_live_fixture import RegionalLiveSettings
from scripts.e2e.regional.warm_spare_fixture import QUARANTINE_TAINT

FAULT = "node-a"
SPARE = "node-b"


def _settings(tmp_path: Path) -> destr003.Settings:
    for name in ("cpu.kubeconfig", "gpu.kubeconfig", "site.yaml", "job.yaml"):
        (tmp_path / name).write_text("apiVersion: v1\n", encoding="utf-8")
    return destr003.Settings(
        regional=RegionalLiveSettings(
            cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
            gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
            gpu_context="gpu-context",
            namespace="gpu-fault-system",
            cluster_id="cluster-a",
            region="us-west-2",
        ),
        site_file=tmp_path / "site.yaml",
        manifest=tmp_path / "job.yaml",
        hyperpod_cluster="hp",
        fault_node=FAULT,
        spare_node=SPARE,
        job_id="destr003-job",
        attempt_id="destr003-job-a001",
        predecessor_path=tmp_path / "predecessor.json",
    )


class _RestoreRecorder:
    def __init__(self, owner: str | None, *, isolated: bool) -> None:
        self.owner = owner
        self.isolated = isolated
        self.waited: list[str] = []
        self.restored: list[str] = []

    def node_snapshot(self, node: str) -> dict[str, Any]:
        return {
            "name": node,
            "unschedulable": self.isolated,
            "taints": (
                [{"key": QUARANTINE_TAINT, "effect": "NoSchedule"}]
                if self.isolated
                else []
            ),
            "annotations": {"gpu-fault.io/incident-id": self.owner},
        }

    def wait_incident_idle(self, incident_id: str) -> dict[str, Any]:
        self.waited.append(incident_id)
        return {"incident_id": incident_id}

    def create_restore_workflow(
        self, *, incident_id: str, node: str, profile_version: str, reason: str
    ) -> dict[str, Any]:
        self.restored.append(incident_id)
        return {"workflow_request_id": f"restore-{incident_id}"}

    def wait_workflow_id(self, workflow_id: str) -> dict[str, Any]:
        return {"request_id": workflow_id, "status": "SUCCEEDED"}


def test_restore_goes_through_the_successor_that_owns_the_node(tmp_path: Path) -> None:
    # The old cleanup compared the annotation with its own incident and skipped
    # the restore when they differed, recording nothing -- the node stayed
    # cordoned under the successor and the case failed in postflight instead.
    warm = _RestoreRecorder("inc-successor", isolated=True)

    result = destr003.restore_fault_node(
        warm,  # type: ignore[arg-type]
        settings=_settings(tmp_path),
        incident_id="inc-ours",
        profile_version="v9",
    )

    assert result["errors"] == [], result
    assert result["successor_incident"] == "inc-successor"
    assert warm.waited == ["inc-successor"]
    assert warm.restored == ["inc-successor"]


def test_an_isolated_node_with_no_owner_is_an_error_not_a_skip(tmp_path: Path) -> None:
    warm = _RestoreRecorder(None, isolated=True)

    result = destr003.restore_fault_node(
        warm,  # type: ignore[arg-type]
        settings=_settings(tmp_path),
        incident_id="inc-ours",
        profile_version="v9",
    )

    assert warm.restored == []
    assert result["errors"] == [
        "fault node is still isolated but names no incident owner; "
        "nothing could be restored"
    ], result


def test_an_unowned_schedulable_node_needs_no_restore(tmp_path: Path) -> None:
    warm = _RestoreRecorder(None, isolated=False)

    result = destr003.restore_fault_node(
        warm,  # type: ignore[arg-type]
        settings=_settings(tmp_path),
        incident_id="inc-ours",
        profile_version="v9",
    )

    assert result == {"errors": [], "quarantine_owner": None}


def test_a_lost_incident_id_is_recovered_from_the_injected_event() -> None:
    class _Warm:
        def __init__(self) -> None:
            self.event_ids: list[str] = []

        def store_snapshot(self, *, event_id: str = "", **_: Any) -> dict[str, Any]:
            self.event_ids.append(event_id)
            return {"incident": {"incident_id": "inc-from-event"}}

    warm = _Warm()
    assert destr003.recover_incident_id(warm, "destr003-1") == "inc-from-event"  # type: ignore[arg-type]
    assert warm.event_ids == ["destr003-1"]
    # Nothing was injected: nothing to look up, and no probe is spent on it.
    assert destr003.recover_incident_id(warm, "") == ""  # type: ignore[arg-type]
    assert warm.event_ids == ["destr003-1"]


def test_gather_runs_independent_reads_concurrently_and_keeps_their_names() -> None:
    barrier = threading.Barrier(3, timeout=5)

    def read(value: str) -> Any:
        def _read() -> str:
            # Three readers must all be inside the pool at once for the
            # barrier to release; serial execution would time out here.
            barrier.wait()
            return value

        return _read

    started = time.monotonic()
    result = destr003.gather(
        {"a": read("A"), "b": read("B"), "c": read("C")}, workers=3
    )

    assert result == {"a": "A", "b": "B", "c": "C"}
    assert time.monotonic() - started < 5


def test_gather_surfaces_the_failing_read() -> None:
    def boom() -> None:
        raise RuntimeError("kubectl exploded")

    with pytest.raises(RuntimeError, match="kubectl exploded"):
        destr003.gather({"ok": lambda: 1, "bad": boom}, workers=2)


class _FakeRegional:
    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        return {"profile": {"profile_version": "v9"}, "release_id": "rel-1"}

    def gpu_workloads(self) -> list[dict[str, Any]]:
        return []

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        return {"pods": 1}

    def evidence_identity(self) -> dict[str, str]:
        return {"release_id": "rel-1", "cluster_id": "cluster-a"}


class _FakeWarm:
    def __init__(self, regional: Any, cluster: str) -> None:
        pass

    def node_snapshot(self, node: str) -> dict[str, Any]:
        return {"name": node, "uid": f"uid-{node}", "labels": {}, "annotations": {}}

    def store_snapshot(self, **_: Any) -> dict[str, Any]:
        return {"agents": []}

    def spare_nodes(self) -> list[str]:
        return [SPARE]

    def cluster_recovery(self) -> dict[str, Any]:
        return {"status": "InService", "node_recovery": "None"}

    def executor_environment(self) -> list[dict[str, Any]]:
        return []

    def synthetic_replacement_gates(self) -> list[dict[str, Any]]:
        return []

    def provider_inventory(self) -> dict[str, Any]:
        return {"count": 2, "sha256": "abc"}


def _patch_preflight(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    identity_calls: list[dict[str, Any]] = []

    def predecessor(path: Path, case_id: str, **identity: Any) -> dict[str, Any]:
        identity_calls.append({"case_id": case_id, **identity})
        return {"valid": True}

    monkeypatch.setattr(destr003, "RegionalLiveFixture", lambda _s: _FakeRegional())
    monkeypatch.setattr(destr003, "WarmSpareLiveFixture", _FakeWarm)
    monkeypatch.setattr(destr003, "predecessor_evidence", predecessor)
    monkeypatch.setattr(destr003, "preflight_errors", lambda *a, **k: [])
    return identity_calls


def test_preflight_binds_the_predecessor_to_this_release_and_runs_the_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity_calls = _patch_preflight(monkeypatch)
    ran: list[Path] = []

    def run_tests(case_dir: Path) -> dict[str, Any]:
        ran.append(case_dir)
        return {"passed": True, "returncode": 0}

    monkeypatch.setattr(destr003, "focused_tests", run_tests)

    result = destr003.read_only_preflight(_settings(tmp_path), tmp_path)

    assert identity_calls == [
        {
            "case_id": destr003.PREDECESSOR_CASE_ID,
            "release_id": "rel-1",
            "cluster_id": "cluster-a",
        }
    ]
    assert ran == [tmp_path]
    assert result["focused_tests"] == {"passed": True, "returncode": 0}
    assert result["release_id"] == "rel-1"
    assert result["declared_spares"] == [SPARE]


def test_preflight_reuses_a_recorded_focused_result_instead_of_rerunning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_preflight(monkeypatch)
    monkeypatch.setattr(
        destr003,
        "focused_tests",
        lambda case_dir: pytest.fail("must not re-run a reusable result"),
    )

    result = destr003.read_only_preflight(
        _settings(tmp_path), tmp_path, reusable_tests={"passed": True}
    )

    assert result["focused_tests"] == {"passed": True, "reused": True}


def test_plan_details_record_the_focused_tests_with_a_source_digest(
    tmp_path: Path,
) -> None:
    preflight = {
        "release_id": "rel-1",
        "fault_node": {"uid": "u-a"},
        "spare_node": {"uid": "u-b"},
        "store": {"profile": {"profile_version": "v9"}},
        "provider_inventory": {"sha256": "abc"},
        "predecessor": {"valid": True},
        "focused_tests": {"passed": True, "returncode": 0},
    }

    details = destr003.plan_details(_settings(tmp_path), preflight)

    assert details["focused_tests"] == {"passed": True, "returncode": 0}
    assert len(details["focused_tests_source_digest"]) == 64
