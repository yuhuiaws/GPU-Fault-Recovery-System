from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_ha001_control_plane_failover as common
from scripts.e2e.regional import run_ha002_pdb_topology as ha
from tests.regional._cov95_ha001_harness import DEADLINE, Clock, pod
from tests.regional._cov95_ha002_harness import HA002Harness


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("min_ingress_ready", 1, "ingress Ready fell"),
        ("min_worker_ready", 1, "worker Ready fell"),
        ("min_spool_ready", 0, "spool-worker Ready fell"),
        ("min_endpoint_ready", 1, "NLB endpoint count"),
    ],
)
def test_final_verdict_rejects_each_observed_availability_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, value: int, error: str
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path, spool=3)
    observe = common.observe_phase

    def sample(*args: Any, **kwargs: Any) -> dict:
        phase = observe(*args, **kwargs)
        phase["summary"][field] = value
        return phase

    monkeypatch.setattr(common, "observe_phase", sample)
    code, report = harness.execute()
    assert code == 1
    assert any(error in item for item in report["errors"]), report["errors"]
    assert harness.resources == {}
    assert harness.nodes["node-0"]["spec"]["unschedulable"] is False


@pytest.mark.parametrize("app", list(ha.ROLE_APPS.values()))
def test_final_verdict_requires_balanced_spread_for_every_enabled_role(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, app: str
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path, spool=3)
    wait = ha.wait_recovery

    def recover(**kwargs: Any) -> list[dict]:
        result = wait(**kwargs)
        for value in harness.pods.values():
            if value["metadata"]["labels"]["app"] == app:
                value["spec"]["nodeName"] = "node-0"
        return result

    monkeypatch.setattr(ha, "wait_recovery", recover)
    code, report = harness.execute()
    assert code == 1
    assert any("balanced spread" in item for item in report["errors"]), report["errors"]
    assert report["cleanup_errors"] == []


@pytest.mark.parametrize("defect", ["window", "missing-plan", "confirmation"])
def test_execute_requires_an_approved_existing_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path)
    if defect == "missing-plan":
        (tmp_path / "cases" / ha.CASE_ID / "plan.json").unlink()
    with pytest.raises(ha.CaseError):
        ha.execute(
            tmp_path,
            1,
            "wrong" if defect == "confirmation" else ha.CONFIRMATION,
            maintenance_window_end=None if defect == "window" else DEADLINE,
        )
    assert harness.events == []


def test_plan_ignores_finished_or_unplaced_pods_and_refuses_other_workloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path)
    finished = pod("finished", "foreign")
    finished["status"]["phase"] = "Succeeded"
    pending = pod("unplaced", "foreign")
    pending["status"]["phase"] = "Pending"
    pending["spec"].pop("nodeName")
    harness.pods.update(finished=finished, unplaced=pending)
    envelope = ha.build_plan(tmp_path, 1, arguments=argparse.Namespace())
    assert envelope["details"]["target_node"]["name"] == "node-0"
    for name in harness.nodes:
        value = pod(f"foreign-{name}", "foreign")
        value["spec"]["nodeName"] = name
        harness.pods[value["metadata"]["name"]] = value
    with pytest.raises(ha.CaseError, match="no clean CPU node"):
        ha.build_plan(tmp_path, 1, arguments=argparse.Namespace())


def test_single_spool_replica_cannot_authorize_a_pdb_disruption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ha.CaseError, match="at least two replicas"):
        HA002Harness(monkeypatch, tmp_path, spool=1)


def test_unschedulable_node_is_not_chosen_even_with_eligible_pods(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path)
    harness.nodes["node-0"]["spec"]["unschedulable"] = True
    plan = ha.build_plan(tmp_path, 1, arguments=argparse.Namespace())["details"]
    assert plan["target_node"]["name"] == "node-1"
    assert "node-0" not in plan["cpu_nodes"]


@pytest.mark.parametrize("mode", ["pdb", "replicas"])
def test_recovery_waiters_stop_at_their_deadlines(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    clock = Clock()
    monkeypatch.setattr(ha, "time", clock)
    if mode == "pdb":
        monkeypatch.setattr(ha, "pdb_snapshot", lambda _: {"disruptions_allowed": 1})
        with pytest.raises(ha.CaseError, match="did not enter blocked"):
            ha.wait_pdb_block("unit", timeout_seconds=1)
    else:
        monkeypatch.setattr(
            common,
            "control_sample",
            lambda **kw: {
                "ingress_ready": 1,
                "worker_ready": 1,
                "spool_ready": 0,
                "endpoint_ready": 1,
                "queue": {"depth": 0},
            },
        )
        with pytest.raises(ha.CaseError, match="did not recover"):
            ha.wait_recovery(replicas={common.INGRESS_APP: 3}, timeout_seconds=1)
    assert clock.now <= 2


@pytest.mark.parametrize(
    "code,error", [(0, "unexpectedly succeeded"), (1, "not rejected by PDB")]
)
def test_only_a_pdb_rejection_satisfies_the_second_eviction(
    code: int, error: str
) -> None:
    with pytest.raises(ha.CaseError, match=error):
        ha.require_pdb_rejection(
            subprocess.CompletedProcess([], code, "", "unavailable"), "unit"
        )


def test_watchdog_escalates_and_reaps_after_term_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    waits = 0

    def wait(**kwargs: Any) -> int:
        nonlocal waits
        waits += 1
        if waits == 1:
            raise subprocess.TimeoutExpired("unit", 10)
        return -9

    process = SimpleNamespace(pid=12345, poll=lambda: None, wait=wait)
    handle = SimpleNamespace(close=lambda: events.append("close"))
    monkeypatch.setattr(
        ha, "os", SimpleNamespace(killpg=lambda pid, sig: events.append((pid, sig)))
    )
    ha.stop_watchdog(process, handle)
    assert events == [(12345, ha.signal.SIGTERM), (12345, ha.signal.SIGKILL), "close"]
    assert waits == 2
    ha.stop_watchdog(None, None)


@pytest.mark.parametrize("execute", [False, True])
def test_main_routes_to_guarded_fake_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, execute: bool
) -> None:
    harness = HA002Harness(monkeypatch, tmp_path)
    monkeypatch.setattr(ha, "install_site_profile", lambda: None)
    monkeypatch.setattr(ha, "install_abort_signals", lambda: None)
    monkeypatch.setattr(common, "configure", lambda args: None)
    monkeypatch.setattr(ha.os, "umask", lambda _: None)
    monkeypatch.setattr(ha, "guard_authorize_execution", lambda *a, **kw: DEADLINE)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "unit",
            "--run-dir",
            str(tmp_path),
            *(["--execute", "--confirm", ha.CONFIRMATION] if execute else []),
        ],
    )
    assert ha.main() == 0
    assert len(harness.evictions) == (4 if execute else 0)


def test_standalone_import_keeps_pdb_rejection_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(ha.__file__).parent))
    spec = importlib.util.spec_from_file_location("cov95_ha002_standalone", ha.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.require_pdb_rejection(
        subprocess.CompletedProcess([], 1, "", "disruption budget"), "unit"
    )
    assert result["reason"] == "would violate PodDisruptionBudget"
