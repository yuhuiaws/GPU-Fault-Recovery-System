"""GF-GENERAL-LIVE-016 ``cursor_pagination`` graded on what the product really makes.

The scripted documents of ``test_general_live_kernel.py`` say what the grade
reads; this module makes sure those documents are the product's. Nothing is
scripted between the variant and the product here: the variant's own reviewed
records go through the probe's real journalctl pass-through wrapper
(``probes/general_live_kernel_source_probe.py`` ``WRAPPER_SCRIPT``), the real
``NodeLogCollector`` reads that wrapper as its ``journalctl``, its batch is
posted to the real control-plane API over an in-memory store, and the real
``ACTIVITY_PROBE`` reads the store back for the grade -- the chain the live run
of 2026-10-27 (R13h, plain EKS) took, where the grade read FAIL "only 0/3
severe kernel records reached an incident" on a store holding one
NODE_HEALTH_GROUP incident with all three records' markers.
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext, default_simulated_profile
from gpu_fault.collectors import CollectorContext
from gpu_fault.collectors.logs.node import NodeLogCollector
from gpu_fault.models import IncidentState
from gpu_fault.store import InMemoryStore
from scripts.e2e.regional import general_live_kernel as mod
from scripts.e2e.regional.probes import general_live_kernel_source_probe as probe
from tests._builders import asgi_client
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    activity_read,
    make_context,
    make_settings,
    store_script,
)

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
CLUSTER = "gpu-eks-cluster"
BOOT = "7679ce8a-3ab8-484a-8d34-4ca9eab39fe5"


class _Node:
    """One GPU node: the replay wrapper on disk and the collector reading it."""

    def __init__(self, tmp_path: Path, run_id: str, context: ApplicationContext):
        self.root = tmp_path / "node"
        self.root.mkdir()
        self.records = self.root / "records.json"
        self.armed = self.root / "armed"
        self.state = self.root / "emissions.jsonl"
        real = self.root / "journalctl-real"
        # The node's own journal has nothing new in the window: the reviewed
        # records are the whole burst, as on a quiet node.
        real.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        real.chmod(0o755)
        self.wrapper = self.root / "journalctl"
        self.wrapper.write_text(
            probe.WRAPPER_SCRIPT
            % {
                "real": str(real),
                "records": str(self.records),
                "armed": str(self.armed),
                "state": str(self.state),
                "emissions": 4,
                "boot_id": BOOT,
                "run_id": run_id,
            },
            encoding="utf-8",
        )
        self.wrapper.chmod(0o755)
        self.responses: list[dict[str, Any]] = []
        node = self

        class Sink:
            def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
                async def submit() -> dict[str, Any]:
                    async with asgi_client(context) as client:
                        response = await client.post(path, json=payload)
                        assert response.status_code == 200, response.text
                        return response.json()

                body = asyncio.run(submit())
                node.responses.append(body)
                return body

        self.collector = NodeLogCollector(
            Sink(),
            CollectorContext(
                cluster_id=CLUSTER, runtime_profile_version="simulated-v1"
            ),
            node_id=NODE,
            boot_id=BOOT,
            runner=self._journalctl,
        )

    def _journalctl(self, argv: list[str], **options: Any) -> Any:
        # The collector asks for ``journalctl``; the unit's PATH drop-in makes
        # that the wrapper on the node, which forwards to the real binary.
        return subprocess.Popen(
            [sys.executable, "-I", str(self.wrapper), *argv[1:]], **options
        )

    # -- the host probe's subcommands, answered by this node -------------- #
    def install(self, arguments: tuple[str, ...]) -> dict[str, Any]:
        encoded = arguments[arguments.index("--b64") + 1]
        records = probe.validate_records(json.loads(base64.b64decode(encoded)))
        self.records.write_text(json.dumps(records), encoding="utf-8")
        return {"records": len(records)}

    def arm(self, _arguments: tuple[str, ...]) -> dict[str, Any]:
        self.armed.write_text("armed", encoding="utf-8")
        # The collector's next poll after ``arm`` (10 s live): the first one
        # that reads the reviewed records.
        self.collector.collect_once()
        return {"armed": True, "mode": "journal"}

    def status(self, _arguments: tuple[str, ...]) -> dict[str, Any]:
        counts = json.loads(self.state.read_text(encoding="utf-8"))
        return {"injected": {"armed": True, "emission_counts": counts}}


@pytest.fixture
def burst(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, _Node, Any]:
    monkeypatch.setattr(
        "gpu_fault.collectors.logs.node.shutil.which", lambda _: "/usr/bin/journalctl"
    )
    store = InMemoryStore()
    store.save_profile(default_simulated_profile())
    context = ApplicationContext(store=store)
    spec = next(item for item in mod.VARIANTS if item.variant == "cursor_pagination")
    regional = FakeRegional(
        cpu_python=store_script(
            {"activity": lambda arguments: activity_read(store, arguments)}
        )
    )
    settings = make_settings(spec, node=NODE)
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    node = _Node(tmp_path, ctx.run_id, context)
    ctx.probes[f"kernel-replay-journal@{NODE}"] = FakeProbe(  # type: ignore[assignment]
        {
            "install-records": node.install,
            "open-window": {
                "after": {"ActiveState": "active"},
                "boot_id": BOOT,
                "proof_scope": "live-replay",
            },
            "arm": node.arm,
            "status": node.status,
            "close-window": {"clean": True},
        }
    )
    return ctx, node, store


def test_the_burst_the_product_folds_into_one_isolation_is_a_pass(burst) -> None:
    ctx, node, store = burst
    result = mod.cursor_pagination(ctx)

    # What the product made of the burst -- the live record's shape: the
    # three records in one batch, three findings, one incident.
    [response] = node.responses
    assert len(response["findings"]) == 3, response
    assert len(set(response["incident_ids"])) == 1, response
    [incident] = store.list_incidents_by_state(
        CLUSTER, list(IncidentState), node_ids={NODE}
    )
    assert incident.event_type == "NODE_HEALTH_GROUP", incident
    assert incident.policy_source == "SITE_KERNEL_LOG", incident
    assert len(store.list_markers_for_incident(incident.incident_id)) == 3, (
        "one marker per finding, all three on the incident the burst minted"
    )

    assert result["verdict"] == "PASS", result["errors"]
    assert result["incidents"] == [incident.incident_id], result
    assert {
        label: (link or {}).get("incident_id")
        for label, link in result["records"].items()
    } == dict.fromkeys(("mce", "storage", "lockup"), incident.incident_id), result


def test_a_record_the_product_never_read_fails_by_name(burst) -> None:
    ctx, node, _store = burst

    def arm_after_two(arguments: tuple[str, ...]) -> dict[str, Any]:
        # The lockup record never reaches the collector: its emission budget
        # is spent before arm, so the wrapper skips it.
        node.state.write_text(json.dumps({"2": 4}), encoding="utf-8")
        return node.arm(arguments)

    ctx.probes[f"kernel-replay-journal@{NODE}"].script["arm"] = arm_after_two
    result = mod.cursor_pagination(ctx)
    assert result["verdict"] == "FAIL", result
    [error] = result["stages"]["all_processed"]
    assert error.startswith("only 2/3 severe kernel records"), error
    assert "lockup" in error, error
