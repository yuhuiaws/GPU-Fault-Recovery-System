"""NOTIFY-005's latch probe reads the window a new phase can actually inherit.

Each managed attempt times and latches its own low-utilisation window (batch
140). An attempt's row keeps its last, latched state once the job is gone,
because nothing resolves that attempt again; only the unscoped
cluster/node/metric/device/rule window is reset by the idle batches between the
phases. A probe that also counted the attempt rows would hold every phase after
the first until its quiesce deadline and fail it as "latch not released".
"""

from __future__ import annotations

import json
import sys
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.host_health import HostMetricSample, NodeHealthPolicy
from gpu_fault.store import SqliteStore
from gpu_fault.store.shared.health_signals import (
    finding_health_signal_fingerprint,
    finding_health_signal_key,
)
from scripts.e2e.regional import run_notification_acceptance as runner
from tests._builders import (
    attempt_observation,
    container_observation,
    host_telemetry_batch,
)

START = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)
WORKLOAD = "training/job/notify005"


def _batch(offset: int, workload_state: str = "ACTIVE"):
    observed_at = START + timedelta(seconds=offset)
    return host_telemetry_batch(
        f"notify005-{offset}",
        observed_at,
        [
            HostMetricSample(
                name="host_gpu_utilization_percent",
                value=0.0,
                unit="percent",
                device="GPU-0",
            )
        ],
        workload_state=workload_state,
        affected_workload_ids=[WORKLOAD],
        received_at=observed_at,
    )


def test_the_latch_probe_skips_a_finished_attempts_own_window(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with closing(SqliteStore(str(tmp_path / "latch.db"))) as store:
        policy = NodeHealthPolicy(store)
        findings = []
        for offset in (0, 150, 300):
            store.save_attempt_observation(
                attempt_observation(
                    "job-a",
                    "attempt-a",
                    START + timedelta(seconds=offset),
                    workload_ids=[WORKLOAD],
                    containers=[
                        container_observation("pod-a", "worker-a", 0, "node-a")
                    ],
                )
            )
            findings = policy.evaluate_metrics(_batch(offset))
        (finding,) = findings
        attempt_key = finding_health_signal_key(finding)
        store.mark_health_signal_notified(
            attempt_key,
            notified_at=START + timedelta(seconds=300),
            semantic_fingerprint=finding_health_signal_fingerprint(finding),
        )
        # The phase's job is gone: the node reports no active workload.
        assert policy.evaluate_metrics(_batch(400, workload_state="IDLE")) == []

        monkeypatch.setattr(
            ApplicationContext,
            "from_environment",
            classmethod(lambda cls: SimpleNamespace(store=store)),
        )

        def cpu_python(script: str, *args: str) -> dict:
            monkeypatch.setattr(sys, "argv", ["latch-probe", *args])
            exec(compile(script, "<latch-probe>", "exec"), {})
            return json.loads(capsys.readouterr().out)

        records = cpu_python(runner.LATCH_STATE_PROBE, "cluster-a", "node-a")
        polls = runner.wait_low_utilization_latch_disarmed(
            SimpleNamespace(cpu_python=cpu_python),
            cluster_id="cluster-a",
            nodes=("node-a",),
            deadline_seconds=0,
            poll_seconds=0,
        )
        attempt_state = store.get_health_signal_state(attempt_key)

    assert records["records"] == [
        {
            "node_id": "node-a",
            "device": "GPU-0",
            "rule_id": "LOW_GPU_UTILIZATION",
            "active": False,
            "notified": False,
            "active_since": None,
        }
    ]
    assert [poll["latched_nodes"] for poll in polls] == [[]]
    assert attempt_state is not None, "the attempt's own window was never claimed"
    assert (attempt_state.active, attempt_state.notified) == (True, True), (
        "the finished attempt's row is expected to keep its latched state"
    )
