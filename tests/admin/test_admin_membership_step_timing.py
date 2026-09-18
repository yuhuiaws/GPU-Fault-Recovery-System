"""Join and removal state records when each step completed."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from gpu_fault.admin import cluster_readiness
from gpu_fault.admin.cluster_join_state import complete_step
from gpu_fault.admin.site import load_site
from tests.admin.test_admin_site import site_file


def test_each_completed_step_keeps_its_own_timestamp(tmp_path: Path) -> None:
    """``updated_at`` only says when the record last moved; an operator asking
    where a join spent its time needs one timestamp per step."""

    path = tmp_path / "state.json"
    state: dict[str, object] = {"completed_steps": [], "evidence": {}}

    complete_step(path, state, "DISCOVERED", {"target": {}})
    first = dict(state["step_completed_at"])  # type: ignore[call-overload]
    complete_step(path, state, "PRECHECKED")

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert sorted(stored["step_completed_at"]) == ["DISCOVERED", "PRECHECKED"], (
        "every completed step must carry a timestamp"
    )
    assert stored["step_completed_at"]["DISCOVERED"] == first["DISCOVERED"], (
        "completing a later step must not rewrite an earlier step's timestamp"
    )
    assert stored["updated_at"] == stored["step_completed_at"]["PRECHECKED"], (
        "the record's updated_at is the latest step's timestamp"
    )


def test_readiness_completion_records_the_real_gate_result_and_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A readiness timestamp covers every required report, not only fast kinds."""

    site = load_site(site_file(tmp_path))
    collectors = {
        "GPU_INVENTORY": {
            "ready": True,
            "last_success_at": "t",
            "unit_state": "active",
        },
        "GPU_METRICS": {"ready": True, "last_success_at": "t", "unit_state": "active"},
    }
    report = {
        "cluster_id": "gpu-a",
        "ready": True,
        "nodes": [{"node_id": "node-1", "ready": True, "collectors": collectors}],
    }
    monkeypatch.setattr(
        cluster_readiness,
        "run_command",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, 0, json.dumps(report) if "exec" in arguments else "cpu-pod", ""
        ),
    )
    ready = cluster_readiness.wait_collector_readiness(site, "gpu-a")
    path = tmp_path / "state.json"
    state: dict[str, object] = {"completed_steps": [], "evidence": {}}

    complete_step(
        path,
        state,
        "COLLECTORS_READY",
        {"nodes": len(ready["nodes"]), "ready": ready["ready"]},
    )

    stored = json.loads(path.read_text(encoding="utf-8"))
    evidence = stored["evidence"]["COLLECTORS_READY"]
    assert stored["step_completed_at"]["COLLECTORS_READY"] == stored["updated_at"]
    assert evidence["ready"] is True
    assert evidence["nodes"] == 1
    assert ready == report, "the readiness gate must preserve every collector verdict"
