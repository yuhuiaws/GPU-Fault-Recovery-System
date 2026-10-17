from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_collector_acceptance as runner
from tests.regional._collector_power_receipt import candidate_record, load_receipt

EXPORTER_POD = {
    "metadata": {"name": "gpu-fault-dcgm-exporter-a"},
    "status": {"phase": "Running"},
    "spec": {
        "containers": [
            {
                "name": "exporter",
                "args": ["-a", "127.0.0.1:9400", "-c", "15000"],
                "env": [{"name": "DCGM_REMOTE_HOSTENGINE_INFO", "value": "x:5555"}],
            }
        ]
    },
}


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

    order: list[str] = []

    def execute(command: str, *arguments: str, **kwargs: object) -> dict[str, Any]:
        order.append(command)
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

    def kubectl(plane: str, *arguments: str, **kwargs: object) -> str:
        order.append(f"kubectl {plane} {' '.join(arguments[:3])}")
        assert "spec.nodeName=node-a" in arguments
        return json.dumps({"items": [EXPORTER_POD]})

    fixture = SimpleNamespace(
        node="node-a",
        regional=SimpleNamespace(
            settings=SimpleNamespace(cluster_id="cluster-a"), kubectl=kubectl
        ),
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
    # The node's own exporter (host DCGM, -c 15000), read before any mutation;
    # the ticks from the quiet delivery to the confirming sample carry 1 s of
    # loop work, and the confirming batch reached the Store at its sample stamp.
    assert order == [
        "kubectl gpu get pod -l",
        "throttle-gpu",
        "restore-gpu-power-limit",
    ]
    assert result["allowed_latency_seconds"] == 62.0
    breakdown = result["allowed_latency_breakdown"]
    assert breakdown["components"] == {
        "nvml_average_seconds": 1.0,
        "dcgm_watch_seconds": 15.0,
        "collector_phase_seconds": 15.0,
        "confirmation_seconds": 30.0,
        "collector_tick_overrun_seconds": 1.0,
        "delivery_seconds": 0.0,
    }
    assert breakdown["exporter"]["dcgm_hostengine"] == "host"
    assert breakdown["collector"]["window"]["ticks"] == 4
    assert result["preparation_seconds"] == 10
    assert result["sampled_status_at"] == (origin + timedelta(seconds=76)).isoformat()
    assert result["delivered_at"] == (origin + timedelta(seconds=61)).isoformat()


def test_a_node_without_the_dcgm_load_generator_is_refused_by_name_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plain EKS (AL2023 NVIDIA) nodes run no host DCGM and ship no dcgmproftester.

    Live 2026-10-05 on gpu-eks-cluster the throttle died inside the probe with
    an opaque ``ProbeError ... at proftester_binary`` and its output withheld;
    the snapshot now reports the generator and the case refuses before it
    touches a power limit.
    """

    monkeypatch.setattr(runner, "seconds_until_next_summary", lambda *a, **k: 0)
    monkeypatch.setattr(runner, "require_action_time", lambda *a: None)
    executed: list[str] = []
    fixture = SimpleNamespace(
        snapshot=lambda: {
            "collector_env": {
                "GPU_FAULT_METRICS_INTERVAL_SECONDS": "5",
                "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS": "60",
                "GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES": "2",
            },
            "gpu_power": [{"index": 0}],
            "dcgm_load_generator": None,
        },
        execute=lambda command, *a, **k: executed.append(command),
    )
    with pytest.raises(runner.RegionalFixtureError, match="no DCGM load generator"):
        runner.run_collect002(fixture, tmp_path, 1)
    assert executed == [], "nothing is executed on the node without the generator"
    assert not (tmp_path / "power-operation-a1.json").exists(), (
        "the refusal happens before the run identity is written"
    )


def test_a_load_that_never_reaches_the_gpu_is_named_and_the_window_ends_early(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Live (plain EKS, seventh round): the load unit stayed active for the
    whole 135-s window while the loaded GPU sat at 0 %/76 W, the chain
    delivered nothing for it and the case failed on the chain. The window now
    ends two cycles in with the load named first, the restore still runs, and
    the load unit's own account from the restore receipt is on the result."""

    origin = datetime(2026, 9, 17, tzinfo=timezone.utc)
    clock = {"monotonic": 5000.0}
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["monotonic"] += seconds

    monkeypatch.setattr(
        runner,
        "time",
        SimpleNamespace(monotonic=lambda: clock["monotonic"], sleep=sleep),
    )
    monkeypatch.setattr(
        runner,
        "datetime",
        SimpleNamespace(
            now=lambda tz=None: origin + timedelta(seconds=clock["monotonic"] - 5000.0),
            fromisoformat=datetime.fromisoformat,
        ),
    )
    monkeypatch.setattr(runner, "seconds_until_next_summary", lambda *a, **k: 0)
    monkeypatch.setattr(runner, "require_action_time", lambda *a: None)
    monkeypatch.setattr(runner, "recent_evidence", lambda *a, **k: [])
    stamps = iter(["2000-01-01T00:00:00Z", "quiet"])
    monkeypatch.setattr(
        runner, "gpu_metrics_stamp", lambda _fixture: next(stamps, "quiet")
    )
    limit = {"w": 700.0}
    calls: list[str] = []

    def execute(command: str, *arguments: str, **kwargs: object) -> dict[str, Any]:
        calls.append(command)
        if command == "throttle-gpu":
            limit["w"] = 200.0
            return {
                **load_receipt(
                    arguments[arguments.index("--run-id") + 1],
                    origin + timedelta(seconds=clock["monotonic"] - 5000.0),
                ),
                "mutation_started": True,
                "timer_armed": True,
                "load_unit_active": True,
            }
        limit["w"] = 700.0
        return {
            "restored": True,
            "load_stopped": True,
            "timer_disarmed": True,
            "cleanup_verified": True,
            "load_unit_active_at_restore": True,
            "load_unit_result": "success",
            "load_exit_status": "0",
            "load_journal_tail": [
                "dcgmproftester12: Error: unable to load the CUDA libraries"
            ],
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
                {
                    "index": 0,
                    "power_limit_w": limit["w"],
                    "power_default_limit_w": 700.0,
                    "power_draw_w": 76.5,
                    "utilization_percent": 0.0,
                }
            ],
        },
    )

    result = runner.run_collect002(fixture, tmp_path, 1)

    assert result["verdict"] == "FAIL"
    assert result["errors"][0].startswith(
        "GPU load generator is not loading GPU 0: 0% utilization and 76 W at a 200 W cap"
    ), result["errors"]
    assert any(
        "GPU metrics chain delivered nothing" in error for error in result["errors"][1:]
    ), "the chain's silence is listed as the consequence, after the cause"
    assert result["load_failure"] == result["errors"][0]
    assert [item["elapsed_seconds"] for item in result["load_status"]] == [15.0, 30.0]
    assert result["load_unit"] == {
        "load_unit_active_at_restore": True,
        "load_unit_result": "success",
        "load_exit_status": "0",
        "load_journal_tail": [
            "dcgmproftester12: Error: unable to load the CUDA libraries"
        ],
    }
    assert sleeps == [15.0, 15.0], "the window ended on the named failure, not at 135 s"
    assert calls == ["throttle-gpu", "restore-gpu-power-limit"]
    assert (
        json.loads((tmp_path / "timeline.json").read_text())["entries"][-1][
            "load_status"
        ]["gpu"]["power_limit_w"]
        == 200.0
    )
