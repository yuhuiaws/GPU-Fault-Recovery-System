"""GF-GENERAL-LIVE 020 ``recovered`` / 022 ``recovery_evidence``: the detection counts.

Both variants held a counter fault, waited for the product's incident, threw
that read away, ended the replay and passed when ``recover_counter``'s last read
had any accepted batch and no active finding of any kind -- true of a product
that never detected the fault at all. Now no incident on the held counter is
INCONCLUSIVE (there was no fault to recover from), the detection is judged as
the sibling diagnostic variant's (a wrong one is a FAIL), and the recovery needs
the finding cleared by a normal reading after the drop, not by the drop
(``renewal_errors``). The product side of that rule is pinned here through the
real DCGM collector, the control-plane API over an in-memory store and the real
``GPU_STATE_PROBE``.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import itertools
import json
import sys
import types
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from gpu_fault.app import ApplicationContext, default_simulated_profile
from gpu_fault.collectors import CollectorContext, next_stable_phase
from gpu_fault.store import InMemoryStore
from scripts.e2e.regional import general_live_gpu_links as links
from scripts.e2e.regional.general_live_gpu_store import GPU_STATE_PROBE
from scripts.e2e.regional.general_live_gpu_verdicts import renewal_errors
from tests._builders import asgi_client
from tests.collectors._support import DcgmMetricsCollector
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    cleared_state,
    dcgm_probe,
    gpu_incident,
    gpu_workflow,
    held_failure,
    waiting_validation,
)

G1 = "GPU-11111111-1111-4111-8111-111111111111"
NEED = "recovery needs a new normal reading"


# --------------------------------------------------------------------------- #
# The product: the drop keeps the finding, a later normal reading clears it
# --------------------------------------------------------------------------- #
CLUSTER = "eks-cluster"
NODE = "eks-node-a"
GPU = "GPU-9d21f50b-8c80-b4e0-8f24-eff9d453315f"
STEADY_FIELDS = {
    "DCGM_FI_DEV_GPU_TEMP": 30,
    "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL": 0,
    "DCGM_FI_DEV_ROW_REMAP_FAILURE": 0,
    "DCGM_FI_DEV_ROW_REMAP_PENDING": 0,
    "DCGM_FI_DEV_PCIE_REPLAY_COUNTER": 0,
    "DCGM_FI_DEV_NVLINK_ERROR_DL_CRC": 0,
    "DCGM_FI_DEV_NVLINK_ERROR_DL_RECOVERY": 0,
    "DCGM_FI_DEV_NVLINK_ERROR_DL_REPLAY": 0,
    "DCGM_FI_DEV_CLOCKS_EVENT_REASONS": 1,
    "DCGM_FI_DEV_POWER_MGMT_LIMIT": 700,
    "DCGM_FI_DEV_THERMAL_VIOLATION": 0,
}
STATE_CODE = compile(GPU_STATE_PROBE, "<general-live-gpu-state-probe>", "exec")
#: Where the node's first health summary falls after its first read: past the
#: drop (read 11, +150 s) and inside the normal reads after it. The phase is the
#: node's own (``next_stable_phase``), so the reads are placed on it rather than
#: on the wall clock -- a summary on the drop's read made the drop's batch a
#: summary as well.
SUMMARY_AFTER = 200


def gpu_state_read(store: Any, arguments: list[str]) -> dict[str, Any]:
    """Run the real ``GPU_STATE_PROBE`` in-process over ``store``."""

    output = io.StringIO()
    context = types.SimpleNamespace(store=store)
    with (
        mock.patch.object(ApplicationContext, "from_environment", lambda: context),
        mock.patch.object(sys, "argv", ["gpu-state-probe", *arguments]),
        contextlib.redirect_stdout(output),
    ):
        exec(STATE_CODE, {"__name__": "__main__"})
    return json.loads(output.getvalue())


class Node:
    """One GPU node's DCGM collector posting to the control plane."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str):
        boot = tmp_path / "boot"
        boot.write_text("b39c4e22-0873-4642-ba92-895dfd9e76ef\n", encoding="ascii")
        monkeypatch.setenv("GPU_FAULT_BOOT_ID_PATH", str(boot))
        monkeypatch.setenv("GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO", "0")
        self.field = field
        self.store = InMemoryStore()
        self.store.save_profile(default_simulated_profile())
        context = ApplicationContext(store=self.store)
        # Near the wall clock (the store keeps raw evidence until it expires),
        # on the node's summary phase.
        phase = next_stable_phase(
            datetime.now(timezone.utc),
            cluster_id=CLUSTER,
            node_id=NODE,
            channel="gpu-metrics",
            interval_seconds=300,
        )
        self.since = phase - timedelta(seconds=SUMMARY_AFTER)
        ticks = (self.since + timedelta(seconds=15 * n) for n in itertools.count())

        class Sink:
            def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
                async def submit() -> dict[str, Any]:
                    async with asgi_client(context) as client:
                        response = await client.post(path, json=payload)
                        assert response.status_code == 200, response.text
                        return response.json()

                return asyncio.run(submit())

        self.collector = DcgmMetricsCollector(
            Sink(),
            CollectorContext(
                cluster_id=CLUSTER,
                runtime_profile_version="simulated-v1",
                product="H200",
                driver_branch=580,
                cuda_version="13.0",
            ),
            node_id=NODE,
            now=lambda: next(ticks),
            health_summary_seconds=300,
            edge_confirmation_samples=3,
        )

    def scrape(self, *values: int) -> None:
        labels = f'gpu="0",UUID="{GPU}",modelName="NVIDIA H200"'
        for value in values:
            readings = {**STEADY_FIELDS, self.field: value}
            self.collector.collect_text(
                "".join(f"{name}{{{labels}}} {v}\n" for name, v in readings.items())
            )

    def state(self, canonical: str) -> dict[str, Any]:
        since = (self.since - timedelta(seconds=60)).isoformat()
        return gpu_state_read(self.store, [CLUSTER, NODE, since, canonical])


@pytest.mark.parametrize(
    "field,canonical,step",
    [
        # 020 recovered's NVLink replay family as the H200 nodes report it.
        (
            "DCGM_FI_DEV_NVLINK_ERROR_DL_REPLAY",
            "nvlink_replay_aggregate_error_total",
            1,
        ),
        # 022 recovery_evidence's over-limit increment at a 15 s scrape.
        ("DCGM_FI_DEV_PCIE_REPLAY_COUNTER", "pcie_replay_total", 3),
    ],
)
def test_the_drop_keeps_the_finding_and_a_later_normal_reading_clears_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    canonical: str,
    step: int,
) -> None:
    node = Node(monkeypatch, tmp_path, field)
    node.scrape(0, 0, *(step * (index + 1) for index in range(8)))
    node.scrape(0)  # the replay ended: the device's own reading is back
    dropped = node.state(canonical)
    assert dropped["batches"][-1]["edge_filter_reasons"] == ["counter-reset"], dropped
    assert [item["canonical_name"] for item in dropped["active_findings"]] == [
        canonical
    ], "the drop is an invalid delta: the finding stays"
    errors = renewal_errors(dropped, canonical, GPU, need=NEED)
    assert len(errors) == 2, errors
    assert "still active after the replay ended" in errors[0], errors
    assert "the drop alone is a counter zeroing" in errors[1], errors

    node.scrape(*([0] * 25))  # the node's next health summary re-judges it
    renewed = node.state(canonical)
    assert renewed["active_findings"] == [], renewed["active_findings"]
    assert renewal_errors(renewed, canonical, GPU, need=NEED) == [], renewed


# --------------------------------------------------------------------------- #
# The variants
# --------------------------------------------------------------------------- #
#: variant -> (registry case, the canonical the default survey resolves)
VARIANTS = {
    "recovered": (links.CASE_020, "nvlink_replay_error_total"),
    "recovery_evidence": (links.CASE_022, "pcie_replay_total"),
}
START = datetime(2026, 10, 27, 1, 47, tzinfo=timezone.utc)
#: The product's records once the replay ended: the replayed rise, the drop and
#: the normal reading after it, the finding cleared.
RENEWED = (1.0, 2.0, 3.0, 0.0, 0.0)


def _activity(
    canonical: str, *executions: dict[str, Any], operations: tuple[str, ...] = ()
) -> dict[str, Any]:
    incident = gpu_incident("inc", f"b-{G1}-{canonical}", state="RECOVERED")
    workflow = gpu_workflow(
        "wf-1",
        "inc",
        ["FREEZE_EVIDENCE", "VALIDATE_GPU", *operations],
        status="FAILED",
        executions=list(executions),
    )
    return {"incidents": [incident], "workflows": [workflow], "batches": []}


def _quiet() -> dict[str, Any]:
    return {"incidents": [], "workflows": [], "batches": [{"observed_at": "t"}]}


def _run(
    tmp_path: Path,
    variant: str,
    activity: Callable[[str], dict[str, Any]],
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    case, canonical = VARIANTS[variant]
    spec = next(
        item
        for item in links.VARIANTS
        if item.variant == variant and item.case_id == case
    )
    final = state or cleared_state(canonical, G1, values=RENEWED)

    def cpu(script: str, *arguments: str) -> dict[str, Any]:
        if script == GPU_STATE_PROBE:
            return dict(final)
        return {
            "markers": [],
            "commands": [],
            "notifications": [],
            "metric_latest": [],
            "collector_statuses": [],
            "agent": None,
            **activity(canonical),
        }

    regional = FakeRegional(cpu_python=cpu)
    settings = make_settings(spec, node=regional.node_snapshot("")["name"])
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    ticks = itertools.count()
    ctx.clock = lambda: START + timedelta(seconds=20 * next(ticks))
    ctx.probes[f"dcgm@{ctx.settings.node}"] = dcgm_probe(
        audit=audit_records({f"{G1}/{canonical}": 1.0})
    )
    return spec.execute(ctx)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_the_detected_fault_that_recovered_on_a_normal_reading_passes(
    tmp_path: Path, variant: str
) -> None:
    out = _run(tmp_path, variant, lambda c: _activity(c, held_failure(c, G1)))
    assert out["verdict"] == "PASS", out["errors"]
    assert set(out["stages"]) == {"hardware", "detection", "read_only", "recovery"}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_a_product_that_never_detected_the_fault_is_inconclusive(
    tmp_path: Path, variant: str
) -> None:
    """The old PASS: no incident, an accepted batch, no active finding."""

    out = _run(
        tmp_path, variant, lambda _c: _quiet(), state=_quiet() | {"active_findings": []}
    )
    assert out["verdict"] == "INCONCLUSIVE", out
    assert out["detail"].endswith("there was no fault to recover from"), out["detail"]
    assert "recovery" not in out["stages"], "nothing to recover from is not judged"


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize(
    "operations,stage",
    [(("RESET_GPU",), "hardware"), (("MARK_UNSCHEDULABLE", "QUARANTINE"), "read_only")],
    ids=["reset", "isolation"],
)
def test_a_wrong_detection_fails(
    tmp_path: Path, variant: str, operations: tuple[str, ...], stage: str
) -> None:
    out = _run(
        tmp_path,
        variant,
        lambda c: _activity(c, held_failure(c, G1), operations=operations),
    )
    assert out["verdict"] == "FAIL", out
    assert out["stages"][stage], out["stages"]


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_a_validation_passed_while_held_is_a_wrong_detection(
    tmp_path: Path, variant: str
) -> None:
    passed = {"operation": "VALIDATE_GPU", "status": "SUCCEEDED", "details": {}}
    out = _run(tmp_path, variant, lambda c: _activity(c, passed))
    assert out["verdict"] == "FAIL", out
    assert out["stages"]["detection"] == [
        "VALIDATE_GPU succeeded while the replayed reading was held"
    ], out["stages"]


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_a_validation_still_waiting_is_inconclusive(
    tmp_path: Path, variant: str
) -> None:
    out = _run(tmp_path, variant, lambda c: _activity(c, waiting_validation()))
    assert out["verdict"] == "INCONCLUSIVE", out
    assert out["detail"].startswith("VALIDATE_GPU did not finish"), out["detail"]


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize(
    "values,active",
    [((1.0, 2.0, 3.0, 0.0), False), ((1.0, 2.0, 3.0), True)],
    ids=["cleared-on-the-drop", "never-cleared"],
)
def test_a_recovery_without_a_normal_reading_after_the_drop_fails(
    tmp_path: Path, variant: str, values: tuple[float, ...], active: bool
) -> None:
    canonical = VARIANTS[variant][1]
    state = cleared_state(canonical, G1, values=values)
    if active:
        state["active_findings"] = [
            {"finding_id": "f", "canonical_name": canonical, "gpu_uuid": G1}
        ]
    out = _run(tmp_path, variant, lambda c: _activity(c, held_failure(c, G1)), state)
    assert out["verdict"] == "FAIL", out
    assert any(
        "the drop alone is a counter zeroing" in error
        for error in out["stages"]["recovery"]
    ), out["stages"]["recovery"]
    still = any("still active" in error for error in out["stages"]["recovery"])
    assert still is active, out["stages"]["recovery"]
