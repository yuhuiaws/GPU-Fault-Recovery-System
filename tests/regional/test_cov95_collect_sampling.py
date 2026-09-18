"""Collector cadence, edge delivery and persisted cursor failure observations."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_collector_acceptance as runner
from tests.regional._collector_power_receipt import candidate_record, load_receipt
from tests.regional._cov95_collect_net import Clock, no_external_effects  # noqa: F401

ORIGIN = datetime(2026, 9, 1, tzinfo=timezone.utc)


def local_clock(monkeypatch: Any) -> Clock:
    clock = Clock()
    monkeypatch.setattr(runner, "time", clock)
    monkeypatch.setattr(
        runner,
        "datetime",
        SimpleNamespace(
            now=lambda tz=None: ORIGIN + timedelta(seconds=clock.now - 1000),
            fromisoformat=datetime.fromisoformat,
        ),
    )
    return clock


@pytest.mark.parametrize(
    "mode",
    ["healthy", "missing", "too-frequent", "persisted-summary", "false-recovery"],
)
def test_cadence_observation_distinguishes_missing_unsuppressed_and_false_edges(
    monkeypatch: Any, tmp_path: Path, mode: str
) -> None:
    clock = local_clock(monkeypatch)
    env = {
        "GPU_FAULT_METRICS_INTERVAL_SECONDS": "2",
        "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS": "10",
        "GPU_FAULT_HOST_INTERVAL_SECONDS": "2",
        "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "10",
    }
    calls = []

    def read(script: str, *args: str) -> dict[str, Any]:
        calls.append(args)
        if len(args) == 3:
            reasons = {
                "persisted-summary": ["health-summary"],
                "false-recovery": ["recovered:gpu"],
            }.get(mode)
            return {
                "records": [
                    {
                        "kind": "HOST_TELEMETRY",
                        "payload": {"edge_filter_reasons": reasons},
                    }
                ]
                if reasons
                else []
            }
        elapsed = clock.now - 1000
        offset = (
            -1
            if mode == "missing"
            else elapsed
            if mode == "too-frequent"
            else int(elapsed // 10) * 10
        )
        return {
            "records": [
                {
                    "collector": "OTHER",
                    "batch_id": "other",
                    "observed_at": ORIGIN.isoformat(),
                },
                {
                    "collector": "GPU_METRICS",
                    "batch_id": "not-dcgm",
                    "observed_at": ORIGIN.isoformat(),
                },
                *[
                    {
                        "collector": channel,
                        "batch_id": prefix + "fixture",
                        "observed_at": (ORIGIN + timedelta(seconds=offset)).isoformat(),
                    }
                    for channel, prefix in runner.COLLECT001_PRODUCERS.items()
                ],
            ]
        }

    regional = SimpleNamespace(
        settings=SimpleNamespace(cluster_id="cluster-a"), cpu_python=read
    )
    fixture = SimpleNamespace(
        node="node-a", regional=regional, snapshot=lambda: {"collector_env": env}
    )
    result = runner.run_collect001(fixture, tmp_path)
    assert result["verdict"] == ("PASS" if mode == "healthy" else "FAIL")
    assert result["observation_seconds"] == 28
    assert clock.now == 1028
    assert calls[-1] == ("cluster-a", "node-a", ORIGIN.isoformat())
    if mode == "healthy":
        assert result["deliveries"]["GPU_METRICS"]["delivered"] == 3
    else:
        assert result["errors"], "bad cadence or mislabelled evidence must fail"


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "no-power",
        "quiet-timeout",
        "no-delivery",
        "late",
        "no-candidate",
        "restore",
    ],
)
def test_power_edge_case_observes_real_sampling_bounds_and_always_restores(
    monkeypatch: Any, tmp_path: Path, problem: str | None
) -> None:
    clock = local_clock(monkeypatch)
    calls = []
    status_reads = 0
    injected = False
    restored = False

    def read(script: str, *args: str) -> dict[str, Any]:
        nonlocal status_reads
        if len(args) == 3:
            return {
                "records": []
                if problem in {"no-candidate", "no-delivery"}
                else [
                    candidate_record(
                        ORIGIN
                        + timedelta(
                            seconds=clock.now - 1000 + (60 if problem == "late" else 0)
                        )
                    )
                ]
            }
        status_reads += 1
        if not injected:
            offset = 0 if status_reads < 3 or problem == "quiet-timeout" else 10
        elif problem == "no-delivery":
            offset = 10
        elif status_reads == 4:
            return {"records": []}
        else:
            offset = clock.now - 1000 + (60 if problem == "late" else 0)
        return {
            "records": [
                {
                    "collector": "GPU_METRICS",
                    "batch_id": "dcgm-fixture",
                    "observed_at": (ORIGIN + timedelta(seconds=offset)).isoformat(),
                }
            ]
        }

    def snapshot() -> dict[str, Any]:
        return {
            "collector_env": {
                "GPU_FAULT_METRICS_INTERVAL_SECONDS": "2",
                "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS": "10",
                "GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES": "2",
            },
            "gpu_power": []
            if problem == "no-power"
            else [
                {
                    "index": 0,
                    "power_default_limit_w": 300,
                    "power_limit_w": 100 if injected and not restored else 300,
                }
            ],
        }

    def execute(command: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        nonlocal injected, restored
        calls.append(command)
        if command == "throttle-gpu":
            injected = True
        else:
            assert command == "restore-gpu-power-limit"
            if problem == "restore":
                raise RuntimeError("restore failed")
            restored = True
        return {
            **load_receipt(
                args[args.index("--run-id") + 1], runner.datetime.now(timezone.utc)
            ),
            "mutation_started": True,
            "timer_armed": True,
            "load_unit_active": True,
            "restored": True,
            "load_stopped": True,
            "timer_disarmed": True,
            "cleanup_verified": True,
        }

    regional = SimpleNamespace(
        settings=SimpleNamespace(cluster_id="cluster-a"), cpu_python=read
    )
    fixture = SimpleNamespace(
        node="node-a", regional=regional, snapshot=snapshot, execute=execute
    )
    if problem in {"no-power", "quiet-timeout"}:
        with pytest.raises(RuntimeError):
            runner.run_collect002(fixture, tmp_path, 1)
        assert calls == [], "unproven baseline must not lower power"
    else:
        result = runner.run_collect002(fixture, tmp_path, 1)
        assert result["verdict"] == ("PASS" if problem is None else "FAIL")
        assert calls == ["throttle-gpu", "restore-gpu-power-limit"]
        if problem is None:
            assert result["delivery_latency_seconds"] == 2


@pytest.mark.parametrize("gpu_count,efa_count", [(2, 2), (1, 2), (2, 1)])
def test_inventory_case_compares_actual_devices_to_live_configuration(
    gpu_count: int, efa_count: int
) -> None:
    from tests.regional.test_alignment_collector_inventory import ENV, inventory_proof

    env = {
        **ENV,
        "GPU_FAULT_EXPECTED_GPU_COUNT": "2",
        "GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT": "2",
    }
    file = {"sha256": "a" * 64}
    proof = inventory_proof()
    for metric in proof["metrics"]:
        metric["value"] = 2
        metric["labels"].update(expected_count="2", observed_count="2")
    fixture = SimpleNamespace(
        node="node-a",
        regional=SimpleNamespace(
            settings=SimpleNamespace(cluster_id="cluster-a"),
            cpu_python=lambda *a: proof,
        ),
        execute=lambda *a: {
            "invocation_id": "invocation",
            "file": file,
            "file_env": env,
            "running_env": dict(env),
        },
        snapshot=lambda: {
            "collector_env": env,
            "collector_env_file": file,
            "gpu_inventory": [{}] * gpu_count,
            "efa_inventory": {"active_count": efa_count},
        },
    )
    result = runner.run_collect003(fixture)
    assert result["verdict"] == ("PASS" if (gpu_count, efa_count) == (2, 2) else "FAIL")


@pytest.mark.parametrize(
    "problem", [None, "initial-count", "replay", "workflow", "cursor-timeout"]
)
def test_fm_case_refuses_bad_initial_receipt_and_detects_replay_after_restart(
    monkeypatch: Any, tmp_path: Path, problem: str | None
) -> None:
    from tests.regional._alignment_fm_receipts import FmProducerFixture

    local_clock(monkeypatch)
    fixture = FmProducerFixture()
    fixture.problem = problem
    calls = fixture.calls
    if problem in {"initial-count", "cursor-timeout", "replay"}:
        with pytest.raises(RuntimeError):
            runner.run_collect005(fixture, tmp_path, 1)
        if problem == "initial-count":
            assert "restart-service" not in calls, (
                "invalid collection cannot authorize restart"
            )
        elif problem == "replay":
            assert len(fixture.store_snapshot("marker")["evidence"]) == 1
    else:
        result = runner.run_collect005(fixture, tmp_path, 1)
        assert result["verdict"] == ("PASS" if problem is None else "FAIL")
        assert calls.index("append-sxid") < calls.index("restart-service")


def test_waiting_and_monitor_only_pollers_stop_at_deadline(
    monkeypatch: Any, tmp_path: Path
) -> None:
    clock = local_clock(monkeypatch)
    reads = []
    fixture = SimpleNamespace(store_snapshot=lambda *a, **k: reads.append(k) or {})
    with pytest.raises(RuntimeError, match="WAITING"):
        runner.waiting_workflow(fixture, "marker", timeout_seconds=4)
    assert len(reads) == 2
    with pytest.raises(RuntimeError, match="did not converge"):
        runner.wait_monitor_only(
            fixture, "marker", case_dir=tmp_path, timeout_seconds=5
        )
    assert clock.now == 1009


@pytest.mark.parametrize("problem", ["stream", "boot", "software", "bad-receipt"])
def test_kernel_case_refuses_unproven_identity_and_stops_after_bad_receipt(
    monkeypatch: Any, tmp_path: Path, problem: str
) -> None:
    local_clock(monkeypatch)
    calls = []
    baseline = {
        "boot_id": "" if problem == "boot" else "boot-a",
        "kernel_collector": {"kmsg_fds": [] if problem == "stream" else ["3"]},
        "gpu_inventory": [{"uuid": "GPU-a", "pci_bdf": "0000:af:00.0"}],
    }

    def execute(command: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(command)
        return (
            {}
            if problem == "software"
            else {"product": "H100", "driver_branch": "550", "cuda_version": "12.4"}
        )

    state = {
        "evidence": [{"record_id": "bad"}],
        "decisions": [{"official_action": "RESTART_APP"}],
        "workflows": [{"status": "BLOCKED"}],
    }
    fixture = SimpleNamespace(
        node="node-a",
        snapshot=lambda: deepcopy(baseline),
        execute=execute,
        store_snapshot=lambda *a, **k: deepcopy(state),
        regional=SimpleNamespace(
            settings=SimpleNamespace(cluster_id="cluster-a"),
            node_snapshot=lambda node: {
                "ready": "True",
                "unschedulable": False,
                "taints": [],
            },
        ),
    )
    if problem != "bad-receipt":
        with pytest.raises(RuntimeError):
            runner.run_collect012(fixture, tmp_path, 1, "profile-a")
        assert "write-xid" not in calls
    else:
        result = runner.run_collect012(fixture, tmp_path, 1, "profile-a")
        assert result["verdict"] == "FAIL"
        assert calls.count("write-xid") == 1, (
            "bad first receipt must stop the remaining injections"
        )
