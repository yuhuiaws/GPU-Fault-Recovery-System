"""GF-GENERAL-LIVE-024 ``independent_xid`` judged on the product's own records.

The variant's own code runs with the node side faked as its unit tests fake it
(the DCGM replay controller, the kmsg kernel replay), but every store read it
makes in the API Pod -- node activity, GPU state, XID decisions -- is the real
probe script run in-process over an in-memory control plane that ingested,
through its own collector routes, what the node sends: the held correctable
pair of the plan's GPU in the exporter's PCI spelling once the replay is armed,
the installed kernel line (XID 92 in the kernel's spelling) once the kmsg
fixture is armed, the flat readings again once the replay closes.

Batch 236 compares PCI devices in provider-event correlation, so the XID's
marker is now associated with the composite's (one GPU, one memory class) and
its decision reads ``duplicate=true``; the R15 base compared the strings and
left the two apart. Either way the XID keeps its own RECOVERED
``inc-<event_id>``, no workflow and no companion correlation, and the variant
must PASS on both shapes -- the R15 harness failed the first on its duplicate
flag alone (S61). When the association takes the XID's incident, the variant
must still FAIL.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import sys
import types
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest import mock

from gpu_fault.app import ApplicationContext, default_simulated_profile
from gpu_fault.channel_registry import GPU_METRICS_PATH, NVIDIA_KERNEL_PATH
from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricSample, GpuMetricSource
from gpu_fault.models import IncidentState, WorkloadState
from gpu_fault.orchestration import coordinator
from gpu_fault.store import InMemoryStore
from scripts.e2e.regional import general_live_gpu_composite as comp
from scripts.e2e.regional import general_live_kernel_xid as kx
from tests._builders import asgi_client, build_context
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
)
from tests.regional._general_live_gpu_fakes import audit_records, dcgm_probe, survey

PAIR = (comp.CORRECTABLE_PRIMARY, "row_remap_correctable_total")
#: The exporter's names of the pair (typed counters arrive ``_total``).
EXPORTER = {
    comp.CORRECTABLE_PRIMARY: "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL",
    "row_remap_correctable_total": "DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS",
}
BOOT = "33333333-3333-4333-8333-333333333333"
#: The GPU the variant picks on this survey (the highest-index one reporting
#: the pair, ``GpuSurvey.component_pair``), checked against the installed plan.
GPU = "GPU-11111111-1111-4111-8111-111111111111"
#: The product's clock sits in the past: no record of this run is future
#: evidence, and the activity window opens at the variant's own ``since``.
T0 = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=30)
INJECTED = T0 + timedelta(seconds=90)


def _survey() -> dict[str, Any]:
    """The H200 survey: the correctable row-remap counter beside the SBE one."""

    value = survey()
    for entry in value["exporter"]["gpus"].values():
        entry["values"]["row_remap_correctable_total"] = 0.0
        entry["metric_names"]["row_remap_correctable_total"] = [
            "DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS_total"
        ]
    return value


def _installed(probe: FakeProbe, command: str, name: str | None = None) -> Any:
    """The JSON the variant installed on the node through ``probe``."""

    for call in probe.calls:
        if call[0] == command and (
            name is None or call[call.index("--name") + 1] == name
        ):
            return json.loads(base64.b64decode(call[call.index("--b64") + 1]))
    raise AssertionError(f"nothing was installed by {command}")


class Product:
    """The in-memory control plane the faked node reports to."""

    def __init__(
        self, survey_value: dict[str, Any], *, cluster: str, node: str
    ) -> None:
        self.store = InMemoryStore()
        self.store.save_profile(default_simulated_profile())
        self.context = build_context(store=self.store)
        self.survey = survey_value
        self.cluster, self.node = cluster, node

    def post(self, path: str, body: dict[str, Any]) -> None:
        async def request() -> None:
            async with asgi_client(self.context) as client:
                response = await client.post(path, json=body)
                assert response.status_code == 200, response.text

        asyncio.run(request())

    def scrape(self, tick: int, value: float) -> None:
        """One exporter scrape of the pair on the plan's GPU, 15 s apart."""

        entry = self.survey["exporter"]["gpus"][GPU]
        labels = {"gpu": entry["gpu_index"], "UUID": GPU}
        samples = [
            GpuMetricSample(
                metric_name=EXPORTER[name],
                canonical_name=name,
                value=value,
                gpu_index=entry["gpu_index"],
                gpu_uuid=GPU,
                pci_bdf=entry["pci_bdf"],
                labels={**labels, "pci_bus_id": entry["pci_bdf"]},
            )
            for name in PAIR
        ]
        batch = GpuMetricBatch(
            batch_id=f"scrape-{tick}",
            cluster_id=self.cluster,
            node_id=self.node,
            observed_at=T0 + timedelta(seconds=15 * tick),
            source=GpuMetricSource.DCGM_EXPORTER,
            product="NVIDIA H200",
            driver_branch=580,
            cuda_version="13.0",
            runtime_profile_version="simulated-v1",
            workload_state=WorkloadState.IDLE,
            samples=samples,
        )
        self.post(GPU_METRICS_PATH, batch.model_dump(mode="json"))

    def kernel_line(self, message: str) -> None:
        self.post(
            NVIDIA_KERNEL_PATH,
            {
                "cluster_id": self.cluster,
                "node_id": self.node,
                "record_id": f"kmsg-{BOOT}-900000064",
                "observed_at": (INJECTED + timedelta(seconds=3)).isoformat(),
                "source_boot_id": BOOT,
                "message": message,
                "product": "NVIDIA H200",
                "driver_branch": 580,
                "cuda_version": "13.0",
                "runtime_profile_version": "simulated-v1",
            },
        )

    def probe(self, script: str, *arguments: str) -> dict[str, Any]:
        """A store probe of the variant, run in-process over this store."""

        output = io.StringIO()
        context = types.SimpleNamespace(store=self.store)
        with (
            mock.patch.object(ApplicationContext, "from_environment", lambda: context),
            mock.patch.object(sys, "argv", ["probe", *arguments]),
            contextlib.redirect_stdout(output),
        ):
            exec(compile(script, "<store-probe>", "exec"), {"__name__": "__main__"})
        return json.loads(output.getvalue().splitlines()[-1])

    def composite_incident_ids(self) -> list[str]:
        return sorted(
            item.incident_id
            for item in self.store.list_incidents_by_state(
                self.cluster, set(IncidentState)
            )
            if item.event_id.endswith(f"-composite-{comp.CORRECTABLE_RULE}")
        )


def _run(tmp_path: Path) -> tuple[dict[str, Any], Product]:
    """The variant over the product: the pair held from the arm, the kernel
    line 90 s later, the real readings again at the close."""

    spec = next(item for item in comp.VARIANTS if item.variant == "independent_xid")
    survey_value = _survey()
    holder: dict[str, Product] = {}
    regional = FakeRegional(
        cpu_python=lambda script, *arguments: holder["product"].probe(
            script, *arguments
        )
    )
    settings = make_settings(spec, node=regional.node_snapshot("")["name"])
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    product = holder["product"] = Product(
        survey_value, cluster=settings.regional.cluster_id, node=settings.node
    )

    def arm(_arguments: tuple[str, ...]) -> dict[str, Any]:
        plan = _installed(dcgm, "install-source", "plan.json")
        targets = {
            gpu
            for item in plan["steps"]
            for operation in item["ops"]
            for gpu in operation.get("gpus") or []
        }
        assert targets == {GPU}, f"the plan targets {targets}, not {GPU}"
        for tick in range(7):  # the baseline, then both counters +1 per scrape
            product.scrape(tick, float(tick))
        return {"armed": True, "status": {"armed": True}}

    def close(_arguments: tuple[str, ...]) -> dict[str, Any]:
        for tick in (9, 10):  # the device's own (flat) readings again
            product.scrape(tick, 8.0)
        return {
            "clean": True,
            "leftover": [],
            "errors": [],
            "disarm": {"disarmed": True},
            "restore": {"dropin_removed": True},
            "stop": {"run_dir_removed": True},
        }

    def kernel_arm(_arguments: tuple[str, ...]) -> dict[str, Any]:
        (record,) = _installed(kernel, "install-records")
        product.kernel_line(record["message"])
        for tick in (7, 8):  # the pair is still held after the line
            product.scrape(tick, float(tick))
        return {"armed": True, "lines_written": kx.EMISSIONS}

    dcgm = dcgm_probe(
        survey_value,
        audit=audit_records({f"{GPU}/{name}": 1.0 for name in PAIR}),
        extra={"arm": arm, "close-run": close},
    )
    kernel = FakeProbe(
        {
            "install-records": {"records": 1},
            "open-window": {"after": {"ActiveState": "active"}},
            "arm": kernel_arm,
            "status": {"injected": {"mode": "kmsg", "fixture_lines": kx.EMISSIONS}},
            "close-window": {"clean": True},
        }
    )
    ctx.probes[f"dcgm@{settings.node}"] = dcgm
    ctx.probes[f"kernel-replay-kmsg@{settings.node}"] = kernel
    with (
        mock.patch.object(comp, "utc_now", lambda: T0),
        mock.patch.object(kx, "utc_now", lambda: INJECTED),
    ):
        return comp.independent_xid(ctx), product


def _xid(out: dict[str, Any]) -> dict[str, Any]:
    (decision,) = out["decisions"]
    return decision


def test_the_variant_passes_on_the_products_association(tmp_path: Path) -> None:
    out, product = _run(tmp_path)

    assert out["verdict"] == "PASS", out["errors"]
    composites = product.composite_incident_ids()
    assert len(composites) == 1, composites
    decision = _xid(out)
    assert decision["duplicate"] is True, (
        "batch 236: the kernel spelling meets the exporter's, so the XID's marker "
        "is associated with the composite's and the decision is flagged"
    )
    assert kx.marker_associations(decision) == composites
    assert out["marker_associations"] == composites
    assert decision["correlated_event_id"] is None
    assert decision["incident_id"] == kx.own_incident_id(decision)
    assert (decision["incident_state"], decision["workflow_request_id"]) == (
        "RECOVERED",
        None,
    )
    assert (
        product.store.get_incident_by_event(decision["event_id"]).incident_id
        == (decision["incident_id"])
    ), "the XID's event is linked to its own incident, not the composite's"


def test_the_variant_passes_where_the_spellings_stay_apart(tmp_path: Path) -> None:
    """The R15 comparison (exact strings): no association, no flag, same verdict."""

    def strings(left: Any, right: Any) -> bool:
        return bool(set(left) & set(right))

    with mock.patch.object(coordinator, "pci_identity_overlaps", strings):
        out, _product = _run(tmp_path)

    assert out["verdict"] == "PASS", out["errors"]
    decision = _xid(out)
    assert (decision["duplicate"], kx.marker_associations(decision)) == (False, [])
    assert out["marker_associations"] == []


def test_the_variant_fails_when_the_association_takes_the_xid(tmp_path: Path) -> None:
    """Fail closed: had the association bound the XID's handling to the marker it
    met -- the shape batch 236 closed for a stale-boot line -- the XID's
    decision would name the composite's incident and overwrite it."""

    real: Callable[..., Any] = coordinator.execution_decision

    def keep_association(event: Any, decision: Any, existing: Any) -> Any:
        return decision if decision.duplicate else real(event, decision, existing)

    with mock.patch.object(coordinator, "execution_decision", keep_association):
        out, _product = _run(tmp_path)

    assert out["verdict"] == "FAIL", out
    independent = out["stages"]["independent"]
    assert any(
        "absorbed" in error or "not its own" in error for error in independent
    ), independent
