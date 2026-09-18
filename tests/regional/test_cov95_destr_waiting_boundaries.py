from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_destr002_hyperpod_reboot as reboot
from scripts.e2e.regional import run_destr003_warm_spare_failover as failover
from scripts.e2e.regional import run_destr008_warm_spare_shortage as shortage
from scripts.e2e.regional import run_destr009_workload_restart as workload
from scripts.e2e.regional.probes import destr016_node_probe as delayed
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_destr_actions import ActionHarness
from tests.regional._cov95_destr_warm import NOW, Clock


@pytest.mark.parametrize("module", [failover, shortage, workload])
@pytest.mark.parametrize("defect", ["missing", "ambiguous", "pending", "partial-gpu"])
@pytest.mark.parametrize("recovers", [False, True])
def test_attempt_observation_polling_is_bounded_and_requires_one_complete_running_attempt(
    module: Any, defect: str, recovers: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    monkeypatch.setattr(module, "time", clock)
    gpu_count = 24 if module is workload else 8
    valid = {
        "workload_phase": "RUNNING",
        "containers": [
            {
                "gpu_count": gpu_count,
                "gpu_uuids": [f"GPU-{index}" for index in range(gpu_count)],
            }
        ],
    }
    invalid = deepcopy(valid)
    if defect == "missing":
        observations = []
    elif defect == "ambiguous":
        observations = [invalid, deepcopy(valid)]
    elif defect == "pending":
        invalid["workload_phase"] = "PENDING"
        observations = [invalid]
    else:
        invalid["containers"][0]["gpu_uuids"].pop()
        observations = [invalid]
    calls: list[dict[str, Any]] = []

    def snapshot(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        rows = [valid] if recovers and len(calls) > 1 else observations
        return {"observations": deepcopy(rows)}

    regional = SimpleNamespace(store_snapshot=snapshot)
    settings = SimpleNamespace(job_id="owned-job", attempt_id="owned-attempt")

    def wait() -> dict[str, Any]:
        if module is shortage:
            return module.wait_observation(
                regional,
                job_id=settings.job_id,
                attempt_id=settings.attempt_id,
                timeout_seconds=10,
            )
        if module is workload:
            return module.wait_observation(
                regional, settings, node="target", timeout_seconds=10
            )
        return module.wait_observation(regional, settings, timeout_seconds=10)

    if recovers:
        assert wait() == valid, {"defect": defect, "calls": calls}
        assert clock.elapsed == 5, clock.elapsed
    else:
        with pytest.raises(
            RegionalFixtureError, match="attempt observation did not appear"
        ):
            wait()
        assert clock.elapsed == 10, clock.elapsed
    assert len(calls) == 2, calls
    assert all(
        row["job_id"] == "owned-job" and row["attempt_id"] == "owned-attempt"
        for row in calls
    ), calls
    if module is workload:
        assert all(
            row["node"] == "target" and row["queue_attempts"] == 1 for row in calls
        ), calls


@pytest.mark.parametrize("defect", ["too-few", "never-replaced", "settles"])
def test_executor_restart_waits_for_replacement_uid_within_a_fixed_budget(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(reboot, tmp_path, monkeypatch)
    command = {"result_details": {"executor_id": "executor-a/7"}}
    before = h.regional.ready_pods("gpu", "executor")
    reads = 0

    def pods(value: Any, _args: Any, _kwargs: Any) -> Any:
        nonlocal reads
        reads += 1
        if defect == "too-few":
            return before[:1]
        if defect == "never-replaced" or reads == 2:
            return deepcopy(before)
        return value

    h.transforms["regional.ready_pods"] = pods
    if defect == "settles":
        result = reboot.restart_executor(h.regional, command)
        assert result["deleted"]["uid"] == "uid-a", result
        assert {row["uid"] for row in result["after"]} == {"uid-b", "uid-c"}, result
        assert reads == 3 and h.clock.elapsed == 5, (reads, h.clock.elapsed)
    else:
        expected = "at least two" if defect == "too-few" else "did not replace"
        with pytest.raises(RegionalFixtureError, match=expected):
            reboot.restart_executor(h.regional, command)
        assert h.clock.elapsed == (0 if defect == "too-few" else 300), h.clock.elapsed
    deletes = [
        args
        for name, args, _ in h.calls
        if name == "regional.kubectl" and "delete" in args
    ]
    assert len(deletes) == (0 if defect == "too-few" else 1), deletes


@pytest.mark.parametrize("defect", ["ambiguous", "uid-drift", "non-object"])
def test_replay_requires_one_stable_replacement_and_an_object_response(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ActionHarness(reboot, tmp_path, monkeypatch)
    restart = {
        "before": [{"name": "executor-a", "uid": "uid-a"}],
        "after": [{"name": "executor-c", "uid": "uid-c"}],
    }
    expected = ""
    if defect == "ambiguous":
        restart["after"].append({"name": "executor-d", "uid": "uid-d"})
        expected = "identity is not unique"
    else:

        def response(value: Any, args: Any, _kwargs: Any) -> Any:
            if defect == "uid-drift" and args[1:3] == ("get", "pod"):
                return json.dumps({"metadata": {"uid": "recreated"}})
            if defect == "non-object" and args[1] == "exec":
                return "[]"
            return value

        h.transforms["regional.kubectl"] = response
        expected = "UID changed" if defect == "uid-drift" else "returned no object"
    with pytest.raises(RegionalFixtureError, match=expected):
        reboot.replay_from_replacement(h.regional, {"command_id": "owned"}, restart)
    execs = [
        args
        for name, args, _ in h.calls
        if name == "regional.kubectl" and "exec" in args
    ]
    assert len(execs) == int(defect == "non-object"), h.calls


@pytest.mark.parametrize("defect", ["bdf", "deadline", "expired", "no-markers"])
def test_delayed_injection_plan_keeps_deadline_and_target_binding_strict(
    defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = Clock()
    monkeypatch.setattr(delayed, "datetime", clock)
    args = argparse.Namespace(
        inject_script="/run/gpu-fault-host-probe-abcdef123456.py",
        pci_bdf="0000:01:00.0",
        max_hold_seconds=120,
        absorb_marker="owned-a",
        absorb_drill_id="owned-a",
        absorb_after_seconds=30,
        maintenance_window_end=(NOW + timedelta(minutes=10)).isoformat(),
    )
    if defect == "bdf":
        args.pci_bdf = "unbound"
        expected = "unsafe PCI BDF"
    elif defect == "deadline":
        args.maintenance_window_end = "unparseable"
        expected = "require a maintenance deadline"
    elif defect == "expired":
        args.maintenance_window_end = NOW.isoformat()
        expected = "window ended"
    else:
        args.absorb_marker = args.absorb_drill_id = ""
        assert delayed.injection_plan(args) == [], vars(args)
        return
    with pytest.raises(delayed.ProbeError, match=expected):
        delayed.injection_plan(args)
