"""GF-GENERAL-LIVE-003 ``ambiguous_ownership`` watches the window the product needs.

Two idle 4-GPU attempts on one node must resolve to none, so no
LOW_GPU_UTILIZATION finding. The variant used to read the node once,
LOW_DURATION plus three collector intervals (345 s) after the jobs ran -- but an
idle GPU is no edge once the node is idle, the host collector delivers its
reading on the 300 s health summaries, and the control plane mints a sustained
finding on the first sample at least 300 s after the first one it graded for
the attempt (``store/shared/xid.py`` ``_next_health_signal_state``): up to
300 s + two summaries after the jobs ran. A wrongly attributed finding minted
at ~600 s passed. The variant now watches that whole window and requires the
product to have graded a low sample on the jobs' GPUs past the sustain --
without one, "no finding" proves nothing (INCONCLUSIVE).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.host_health import NodeHealthPolicy
from scripts.e2e.regional import general_live_utilization as util
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    incident,
    make_context,
    make_settings,
    store_poll_clock,
)

SPEC = next(item for item in util.VARIANTS if item.variant == "ambiguous_ownership")
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
#: The rule id the product mints a low GPU finding under (the product's table).
LOW_GPU = NodeHealthPolicy.SUSTAINED_METRIC_RULES["host_gpu_utilization_percent"][0]


class IdleJob:
    def __init__(self, ctx: Any, *, name: str, gpu_count: int = 8) -> None:
        self.name = name
        self.gpus = [f"GPU-{name}-{index}" for index in range(gpu_count)]

    def start(self) -> dict[str, Any]:
        return {"pods": []}

    def identity(self) -> dict[str, Any]:
        return {
            "job_id": self.name,
            "attempt_id": f"{self.name}-a001",
            "gpu_uuids": list(self.gpus),
            "pods": [],
        }

    def cleanup(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _idle_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(util, "ManagedIdleJob", IdleJob)


def _gpu_sample(since: datetime, *, seconds: float, device: str, value: float) -> Any:
    return {
        "name": "host_gpu_utilization_percent",
        "device": device,
        "value": value,
        "observed_at": (since + timedelta(seconds=seconds)).isoformat(),
        "labels": {},
    }


def _low_incident(since: datetime) -> dict[str, Any]:
    """LOW_GPU_UTILIZATION minted on a GPU of one attempt -- what a product
    that guessed the owner would do."""

    found = incident(
        "inc-low",
        "host_gpu_utilization_percent",
        device="GPU-gl3-ambA-1-0",
        rule_id=LOW_GPU,
    )
    return {
        **found,
        "cluster_id": "gpu-eks-cluster",
        "created_at": (since + timedelta(seconds=600)).isoformat(),
    }


Read = Callable[[datetime, int], dict[str, Any]]


def _run(tmp_path: Path, read: Read, *, summary: int = 300) -> tuple[Any, list[int]]:
    """The variant with every activity read answered by ``read(since, n)``;
    returns the result and the read numbers."""

    reads: list[int] = []

    def cpu_python(script: str, *arguments: str) -> dict[str, Any]:
        reads.append(len(reads) + 1)
        since = datetime.fromisoformat(arguments[2])
        return {
            "since": arguments[2],
            "cluster_id": "gpu-eks-cluster",
            "node_id": NODE,
            "incidents": [],
            "workflows": [],
            "batches": [],
            "markers": [],
            "commands": [],
            "notifications": [],
            "metric_latest": [],
            "collector_statuses": [],
            **read(since, len(reads)),
        }

    tmp_path.mkdir(parents=True, exist_ok=True)
    settings = make_settings(SPEC, node=NODE)
    settings = type(settings)(**{**settings.__dict__, "site_file": tmp_path / "s.yaml"})
    (tmp_path / "s.yaml").write_text("site: test\n", encoding="utf-8")
    node = FakeProbe(
        {
            "survey": {
                "collector_env": {
                    "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
                    "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": str(summary),
                },
                "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
            }
        }
    )
    ctx = make_context(
        SPEC,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=cpu_python),
        probes={f"node@{NODE}": node},
        settings=settings,
    )
    return SPEC.execute(store_poll_clock(ctx)), reads


def _graded(since: datetime, n: int) -> dict[str, Any]:
    """An idle node no attempt resolved on: the jobs' GPUs graded at 0 % on
    the summaries, the latest one past the sustain."""

    return {
        "metric_latest": [
            _gpu_sample(since, seconds=20 * n, device="GPU-gl3-ambA-1-0", value=0.0)
        ]
    }


def test_the_window_is_the_sustain_plus_two_summaries_on_the_nodes_cadence(
    tmp_path: Path,
) -> None:
    result, reads = _run(tmp_path, _graded)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["watch_seconds"] == 300 + 2 * 300 + 15 + util.LOW_CHAIN_MARGIN, (
        result["watch_seconds"]
    )
    assert len(reads) * 20 >= result["watch_seconds"], "watched the whole window"
    longer, _reads = _run(tmp_path / "slow", _graded, summary=600)
    assert longer["watch_seconds"] == 300 + 2 * 600 + 15 + util.LOW_CHAIN_MARGIN, (
        longer["watch_seconds"]
    )


def test_a_late_wrongly_attributed_finding_fails(tmp_path: Path) -> None:
    """The finding a guessed owner would get, minted at ~600 s: the old single
    read at 345 s never saw it."""

    def late(since: datetime, n: int) -> dict[str, Any]:
        document = _graded(since, n)
        if n * 20 >= 600:
            document["incidents"] = [_low_incident(since)]
        return document

    result, reads = _run(tmp_path, late)
    assert result["verdict"] == "FAIL", result
    assert result["stages"]["no_finding"] == [
        "unexpected host_gpu_utilization_percent incident inc-low"
    ], result["stages"]
    assert len(reads) * 20 < result["watch_seconds"], "the watch ended at the finding"


@pytest.mark.parametrize(
    "samples",
    [
        [],
        [{"seconds": 120, "device": "GPU-gl3-ambA-1-0", "value": 0.0}],
        [{"seconds": 900, "device": "GPU-gl3-ambA-1-0", "value": 97.0}],
        [{"seconds": 900, "device": "GPU-other-0", "value": 0.0}],
    ],
    ids=["none", "before-the-sustain", "busy", "not-the-jobs-gpu"],
)
def test_no_low_sample_graded_past_the_sustain_is_inconclusive(
    tmp_path: Path, samples: list[dict[str, Any]]
) -> None:
    def read(since: datetime, n: int) -> dict[str, Any]:
        return {"metric_latest": [_gpu_sample(since, **item) for item in samples]}

    result, _reads = _run(tmp_path, read)
    assert result["verdict"] == "INCONCLUSIVE", result
    assert result["stages"]["no_finding"] == [], result["stages"]
    assert "proves nothing about the attribution" in result["detail"], result


def test_the_spec_names_the_window() -> None:
    assert SPEC.max_minutes >= 30, "two job starts, the ~18-minute watch, cleanup"
    assert "two host health summaries" in SPEC.injection, SPEC.injection
