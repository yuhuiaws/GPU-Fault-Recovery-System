from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_collector_acceptance as runner
from tests.regional._collector_power_receipt import candidate_record, load_receipt


def test_same_attempt_in_distinct_campaigns_cannot_share_power_control_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "seconds_until_next_summary", lambda *a, **k: 0)
    monkeypatch.setattr(runner, "require_action_time", lambda *a: None)
    run_ids: list[str] = []
    for directory in (tmp_path / "first", tmp_path / "second", tmp_path / "first"):
        stamps = iter(["2000-01-01T00:00:00Z", datetime.now(timezone.utc).isoformat()])
        monkeypatch.setattr(runner, "gpu_metrics_stamp", lambda _fixture: next(stamps))
        calls: list[tuple[str, str]] = []

        def execute(command: str, *arguments: str, **kwargs: object) -> dict[str, bool]:
            run_id = arguments[arguments.index("--run-id") + 1]
            calls.append((command, run_id))
            if command == "throttle-gpu":
                raise RuntimeError("controlled refusal before mutation")
            return {"mutation_started": False}

        fixture = SimpleNamespace(
            snapshot=lambda: {
                "collector_env": {
                    "GPU_FAULT_METRICS_INTERVAL_SECONDS": "5",
                    "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS": "60",
                    "GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES": "2",
                },
                "gpu_power": [{"index": 0}],
            },
            execute=execute,
        )
        with pytest.raises(RuntimeError, match="controlled refusal"):
            runner.run_collect002(fixture, directory, 1)
        saved = json.loads((directory / "power-operation-a1.json").read_text())
        run_ids.append(saved["run_id"])
        assert calls == [
            ("throttle-gpu", saved["run_id"]),
            ("restore-gpu-power-limit", saved["run_id"]),
        ]
        assert saved["attempt"] == 1
    assert run_ids[0] != run_ids[1]
    assert run_ids[0] == run_ids[2]


@pytest.mark.parametrize(
    "defect", ["throttle-proof", "restore-proof", "restore-no-mutation"]
)
def test_caller_requires_mutation_and_cleanup_proof_even_when_power_looks_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    now = datetime.now(timezone.utc)
    stamps = iter(
        [
            (now - timedelta(seconds=60)).isoformat(),
            now.isoformat(),
            (now + timedelta(seconds=1)).isoformat(),
        ]
    )
    monkeypatch.setattr(runner, "seconds_until_next_summary", lambda *a, **k: 0)
    monkeypatch.setattr(runner, "require_action_time", lambda *a: None)
    monkeypatch.setattr(runner, "gpu_metrics_stamp", lambda _: next(stamps))
    monkeypatch.setattr(
        runner,
        "recent_evidence",
        lambda *a, **k: [
            {"kind": "GPU_METRICS", "payload": {"edge_filter_reasons": ["candidate"]}}
        ],
    )
    calls = []

    def execute(command: str, *arguments: str, **kwargs: object) -> dict[str, Any]:
        calls.append(command)
        proof = {
            **load_receipt(
                arguments[arguments.index("--run-id") + 1], datetime.now(timezone.utc)
            ),
            "mutation_started": True,
            "timer_armed": True,
            "load_unit_active": True,
            "restored": True,
            "load_stopped": True,
            "timer_disarmed": True,
            "cleanup_verified": True,
        }
        if command == "throttle-gpu" and defect == "throttle-proof":
            proof.pop("timer_armed")
        elif command == "restore-gpu-power-limit":
            if defect == "restore-proof":
                proof["cleanup_verified"] = False
            elif defect == "restore-no-mutation":
                proof.update(restored=False, mutation_started=False, no_mutation=True)
        return proof

    fixture = SimpleNamespace(
        node="node-a",
        regional=SimpleNamespace(settings=SimpleNamespace(cluster_id="cluster-a")),
        execute=execute,
        snapshot=lambda: {
            "collector_env": {
                "GPU_FAULT_METRICS_INTERVAL_SECONDS": "5",
                "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS": "60",
                "GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES": "2",
            },
            "gpu_power": [
                {"index": 0, "power_limit_w": 700, "power_default_limit_w": 700}
            ],
        },
    )
    if defect == "throttle-proof":
        with pytest.raises(runner.RegionalFixtureError, match="recovery proof"):
            runner.run_collect002(fixture, tmp_path, 1)
    else:
        result = runner.run_collect002(fixture, tmp_path, 1)
        assert result["verdict"] == "FAIL"
        assert "restoration proof" in result["cleanup_error"]
    assert calls == ["throttle-gpu", "restore-gpu-power-limit"]


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "missing",
        "run",
        "boot",
        "intent",
        "index",
        "boolean-index",
        "gpu",
        "timestamp",
        "naive",
        "before-request",
        "after-response",
    ],
)
def test_load_timing_requires_the_owned_pre_exec_receipt(defect: str) -> None:
    requested = datetime(2026, 9, 17, tzinfo=timezone.utc)
    started = requested + timedelta(seconds=25)
    received = requested + timedelta(seconds=30)
    injection = load_receipt("timing-run", started)
    receipt = injection["load_start"]
    if defect == "missing":
        injection.pop("load_start")
    elif defect in {"run", "boot", "intent", "gpu"}:
        key = {
            "run": "run_id",
            "boot": "boot_id",
            "intent": "intent_sha256",
            "gpu": "gpu_uuid",
        }[defect]
        receipt[key] = "foreign"
    elif defect == "index":
        receipt["gpu_index"] = 1
    elif defect == "boolean-index":
        receipt["gpu_index"] = False
    elif defect == "timestamp":
        receipt["started_at"] = "unknown"
    elif defect == "naive":
        receipt["started_at"] = started.replace(tzinfo=None).isoformat()
    elif defect == "before-request":
        receipt["started_at"] = (requested - timedelta(seconds=1)).isoformat()
    elif defect == "after-response":
        receipt["started_at"] = (received + timedelta(seconds=1)).isoformat()
    if defect == "none":
        assert (
            runner.power_load_started_at(
                injection,
                run_id="timing-run",
                requested_at=requested,
                received_at=received,
            )
            == started
        )
    else:
        with pytest.raises(runner.RegionalFixtureError):
            runner.power_load_started_at(
                injection,
                run_id="timing-run",
                requested_at=requested,
                received_at=received,
            )


def test_first_persisted_delivery_wins_over_a_later_status_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    origin = datetime(2026, 9, 17, tzinfo=timezone.utc)
    current = [origin]
    monkeypatch.setattr(
        runner,
        "datetime",
        SimpleNamespace(
            now=lambda tz=None: current[0], fromisoformat=datetime.fromisoformat
        ),
    )
    monkeypatch.setattr(runner, "seconds_until_next_summary", lambda *a, **k: 0)
    monkeypatch.setattr(runner, "require_action_time", lambda *a: None)
    reads = 0

    def status(_fixture: object) -> str:
        nonlocal reads
        reads += 1
        if reads == 1:
            return (origin - timedelta(seconds=300)).isoformat()
        if reads == 2:
            return origin.isoformat()
        current[0] = origin + timedelta(seconds=76)
        return current[0].isoformat()

    monkeypatch.setattr(runner, "gpu_metrics_stamp", status)
    monkeypatch.setattr(
        runner,
        "recent_evidence",
        lambda *a, **k: [
            candidate_record(origin + timedelta(seconds=76)),
            candidate_record(origin + timedelta(seconds=61)),
        ],
    )

    def execute(command: str, *arguments: str, **kwargs: object) -> dict[str, Any]:
        if command == "throttle-gpu":
            current[0] = origin + timedelta(seconds=10)
            return {
                **load_receipt(arguments[arguments.index("--run-id") + 1], current[0]),
                "mutation_started": True,
                "timer_armed": True,
                "load_unit_active": True,
            }
        return {
            "restored": True,
            "load_stopped": True,
            "timer_disarmed": True,
            "cleanup_verified": True,
        }

    fixture = SimpleNamespace(
        node="node-a",
        regional=SimpleNamespace(settings=SimpleNamespace(cluster_id="cluster-a")),
        execute=execute,
        snapshot=lambda: {
            "collector_env": {
                "GPU_FAULT_METRICS_INTERVAL_SECONDS": "15",
                "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS": "300",
                "GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES": "3",
            },
            "gpu_power": [
                {"index": 0, "power_limit_w": 700, "power_default_limit_w": 700}
            ],
        },
    )
    result = runner.run_collect002(fixture, tmp_path, 1)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["delivery_latency_seconds"] == 51
    assert result["allowed_latency_seconds"] == 60
    assert result["preparation_seconds"] == 10
    assert result["sampled_status_at"] == (origin + timedelta(seconds=76)).isoformat()
    assert result["delivered_at"] == (origin + timedelta(seconds=61)).isoformat()
