from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.e2e.regional.destr015_physical_evidence import (
    ResetIntervalScope,
    evidence_digest,
    physical_overlap_errors,
)
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from scripts.e2e.regional.probes import destr015_physical_probe as probe

NOW = datetime(2026, 9, 15, tzinfo=timezone.utc)
BASE = int(NOW.timestamp() * 1_000_000_000)
SECOND = 1_000_000_000


def interval_fixture():
    workflow = {"request_id": "workflow", "incident_id": "incident", "fencing_token": 7}
    scopes, captures, hosts = {}, {}, {}
    for node, interval in (("node-a", (5, 20)), ("node-b", (10, 25))):
        scope = ResetIntervalScope(
            run_id="run",
            cluster_id="cluster",
            release_id="release",
            node=node,
            node_uid=f"uid-{node}",
            boot_id=f"boot-{node}",
            agent_generation=4,
            gpu_uuid=f"GPU-{node}",
            maintenance_end=NOW + timedelta(hours=1),
        )
        scopes[node] = scope
        start = {
            "scope_sha256": scope.digest(),
            "witness_id": f"witness-{node}",
            "tracee": {"pid": 1, "boot_id": scope.boot_id, "start_ticks": 1},
            "producer": {"pid": 2, "boot_id": scope.boot_id, "start_ticks": 2},
            "executable_path": "/usr/bin/nvidia-smi",
            "executable_sha256": "a" * 64,
        }
        end = {
            **start,
            "start_sha256": evidence_digest(start),
            "trace_complete": True,
            "closed": True,
            "lost_events": 0,
            "trace_sha256": "b" * 64,
            "trace_bytes": 100,
            "calibration_execs": 1,
            "wall_minus_monotonic_min_ns": BASE - 50 * SECOND,
            "wall_minus_monotonic_max_ns": BASE - 50 * SECOND,
            "actions": [
                {
                    "operation": "RESET_GPU",
                    "gpu_uuid": scope.gpu_uuid,
                    "returncode": 0,
                    "pid": 123,
                    "started_ns": BASE + interval[0] * SECOND,
                    "ended_ns": BASE + interval[1] * SECOND,
                }
            ],
        }
        captures[node] = {
            "start": start,
            "end": end,
            "clock_exchanges": [
                {
                    "sent_ns": 1000 * SECOND,
                    "received_ns": 1000 * SECOND + 20_000_000,
                    "monotonic_ns": 50 * SECOND,
                },
                {
                    "sent_ns": 1100 * SECOND,
                    "received_ns": 1100 * SECOND + 20_000_000,
                    "monotonic_ns": 150 * SECOND,
                },
            ],
        }
        hosts[node] = {
            "before": {"boot_id": scope.boot_id, "ledger": []},
            "after": {
                "boot_id": scope.boot_id,
                "ledger": [
                    {
                        "operation": "RESET_GPU",
                        "command_id": f"command-{node}",
                        "attempt": 1,
                        "state": "SUCCEEDED",
                        "workflow_request_id": "workflow",
                        "incident_id": "incident",
                        "fencing_token": 7,
                        "agent_generation": 4,
                        "gpu_uuids": [scope.gpu_uuid],
                        "started_at": NOW.isoformat(),
                        "completed_at": (NOW + timedelta(seconds=40)).isoformat(),
                    }
                ],
            },
        }
    return captures, scopes, workflow, hosts


def test_parallel_proof_uses_actual_exec_intervals_with_clock_bounds():
    captures, scopes, workflow, hosts = interval_fixture()
    assert (
        physical_overlap_errors(captures, scopes=scopes, workflow=workflow, hosts=hosts)
        == []
    )


@pytest.mark.parametrize(
    "defect",
    [
        "serial-physical",
        "boot",
        "workflow",
        "incident",
        "fence",
        "generation",
        "gpu",
        "duplicate-command",
        "command-window",
        "trace-lost",
        "truncated",
        "uncalibrated",
        "missing-clock",
        "clock-uncertainty",
        "clock-reversed",
        "clock-drift",
        "wrong-witness",
        "scope",
        "failed-reset",
        "duplicate-reset",
    ],
)
def test_branch_overlap_cannot_mask_an_unproven_physical_interval(defect):
    captures, scopes, workflow, hosts = interval_fixture()
    current = captures["node-b"]
    action = current["end"]["actions"][0]
    row = hosts["node-b"]["after"]["ledger"][0]
    if defect == "serial-physical":
        action.update(started_ns=BASE + 22 * SECOND, ended_ns=BASE + 29 * SECOND)
    elif defect == "boot":
        hosts["node-b"]["after"]["boot_id"] = "new-boot"
    elif defect in {"workflow", "incident", "fence", "generation", "gpu"}:
        key = {
            "workflow": "workflow_request_id",
            "incident": "incident_id",
            "fence": "fencing_token",
            "generation": "agent_generation",
            "gpu": "gpu_uuids",
        }[defect]
        row[key] = (
            ["GPU-other"]
            if defect == "gpu"
            else 9
            if defect in {"fence", "generation"}
            else "other"
        )
    elif defect == "duplicate-command":
        hosts["node-b"]["after"]["ledger"].append(dict(row))
    elif defect == "command-window":
        row["started_at"] = (NOW + timedelta(seconds=30)).isoformat()
    elif defect == "trace-lost":
        current["end"]["lost_events"] = 1
    elif defect == "truncated":
        current["end"]["trace_complete"] = False
    elif defect == "uncalibrated":
        current["end"]["calibration_execs"] = 0
    elif defect == "missing-clock":
        current["clock_exchanges"].pop()
    elif defect == "clock-uncertainty":
        for clock in current["clock_exchanges"]:
            clock["received_ns"] += 40 * SECOND
    elif defect == "clock-reversed":
        current["clock_exchanges"][1]["sent_ns"] = 1
    elif defect == "clock-drift":
        current["clock_exchanges"][1]["monotonic_ns"] += 20 * SECOND
    elif defect == "wrong-witness":
        current["end"]["witness_id"] = "other"
    elif defect == "scope":
        current["start"]["scope_sha256"] = "b" * 64
    elif defect == "failed-reset":
        action["returncode"] = 1
    else:
        current["end"]["actions"].append(dict(action))
    assert physical_overlap_errors(
        captures, scopes=scopes, workflow=workflow, hosts=hosts
    ), defect


def quoted(value):
    return '"' + "".join(f"\\x{byte:02x}" for byte in value.encode()) + '"'


def trace_process(pid, start, end, args, code=0, *, executable="/usr/bin/nvidia-smi"):
    def stamp(seconds):
        return f"{BASE // SECOND + seconds}.000000000"

    return (
        f"{pid} {stamp(start)} execve({quoted(executable)}, "
        f"[{', '.join(quoted(item) for item in args)}], NULL) = 0\n"
        f"{pid} {stamp(end)} +++ exited with {code} +++\n"
    ).encode()


def test_physical_probe_parses_actual_exec_exit_pairs_not_workflow_timestamps():
    raw = trace_process(100, 1, 2, ("nvidia-smi", *probe.QUERY_ARGS))
    raw += trace_process(101, 4, 9, ("nvidia-smi", "--gpu-reset", "-i", "GPU-a"))
    actions, calibration = probe.reset_events(
        raw, executable=Path("/usr/bin/nvidia-smi"), gpu_uuid="GPU-a"
    )
    assert calibration == 1
    assert actions[0]["started_ns"] == BASE + 4 * SECOND
    assert actions[0]["ended_ns"] == BASE + 9 * SECOND
    assert actions[0]["pid"] == 101
    with pytest.raises(BoundaryDenied, match="approved GPU"):
        probe.reset_events(
            raw, executable=Path("/usr/bin/nvidia-smi"), gpu_uuid="GPU-b"
        )
    with pytest.raises(BoundaryDenied):
        probe.reset_events(
            raw[:-1], executable=Path("/usr/bin/nvidia-smi"), gpu_uuid="GPU-a"
        )


def test_continuous_clock_envelope_refuses_a_wall_clock_jump(monkeypatch):
    monotonic = iter([SECOND, SECOND, 2 * SECOND, 2 * SECOND])
    wall = iter([BASE, BASE + 100 * SECOND])
    monkeypatch.setattr(
        probe,
        "time",
        SimpleNamespace(
            monotonic_ns=lambda: next(monotonic), time_ns=lambda: next(wall)
        ),
    )
    clock = probe.ClockEnvelope()
    assert clock.sample() == {"monotonic_ns": SECOND}
    with pytest.raises(BoundaryDenied, match="jumped"):
        clock.sample()
