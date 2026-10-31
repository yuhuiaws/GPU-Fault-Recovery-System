"""GF-GENERAL-LIVE-016 ``user_text`` / ``old_boot`` proved on what the product really did.

Both negatives used to read only the journal wrapper's emission count and an
activity read without a QUARANTINE incident: a record the control plane never
received, or one it graded into a diagnostic finding, passed alike. Here the
variant's own reviewed record goes through the probe's real journalctl
pass-through wrapper (``WRAPPER_SCRIPT``), the real ``NodeLogCollector``, the
real control-plane API over an in-memory store, and the real
``ACTIVITY_PROBE`` / ``NODE_LOG_ENTRY_PROBE`` read the store back: the product
stores the batch with the record and grades it into nothing -- the trusted-source
gate refuses a journal line that is not kernel transport, the current-boot check
a kernel line stamped with another boot -- and the variant passes on that
evidence only.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from gpu_fault.app import ApplicationContext, default_simulated_profile
from gpu_fault.collectors import CollectorContext
from gpu_fault.collectors.logs.node import NodeLogCollector
from gpu_fault.models import IncidentState, RecoveryAction, Severity
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
ENTRY_CODE = compile(mod.NODE_LOG_ENTRY_PROBE, "<node-log-entry-probe>", "exec")


def entries_read(store: Any, arguments: tuple[str, ...]) -> dict[str, Any]:
    """Run the real ``NODE_LOG_ENTRY_PROBE`` in-process over ``store``."""

    output = io.StringIO()
    context = types.SimpleNamespace(store=store)
    with (
        mock.patch.object(ApplicationContext, "from_environment", lambda: context),
        mock.patch.object(sys, "argv", ["node-log-entry-probe", *arguments]),
        contextlib.redirect_stdout(output),
    ):
        exec(ENTRY_CODE, {"__name__": "__main__"})
    return json.loads(output.getvalue())


class JournalNode:
    """One GPU node: the replay wrapper on disk and the collector reading it,
    posting to the control plane over the in-memory store."""

    def __init__(self, tmp_path: Path, run_id: str, context: ApplicationContext):
        self.root = tmp_path / "node"
        self.root.mkdir()
        self.records = self.root / "records.json"
        self.armed = self.root / "armed"
        self.state = self.root / "emissions.jsonl"
        real = self.root / "journalctl-real"
        # A quiet node's own journal: the reviewed record is all there is.
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
        self.polls_after_arm = 1
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
        return subprocess.Popen(
            [sys.executable, "-I", str(self.wrapper), *argv[1:]], **options
        )

    def install(self, arguments: tuple[str, ...]) -> dict[str, Any]:
        encoded = arguments[arguments.index("--b64") + 1]
        records = probe.validate_records(json.loads(base64.b64decode(encoded)))
        self.records.write_text(json.dumps(records), encoding="utf-8")
        return {"records": len(records)}

    def arm(self, _arguments: tuple[str, ...]) -> dict[str, Any]:
        self.armed.write_text("armed", encoding="utf-8")
        for _ in range(self.polls_after_arm):
            self.collector.collect_once()
        return {"armed": True, "mode": "journal"}

    def status(self, _arguments: tuple[str, ...]) -> dict[str, Any]:
        counts = json.loads(self.state.read_text(encoding="utf-8"))
        return {"injected": {"armed": True, "emission_counts": counts}}


def negative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> tuple[Any, JournalNode, InMemoryStore]:
    monkeypatch.setattr(
        "gpu_fault.collectors.logs.node.shutil.which", lambda _: "/usr/bin/journalctl"
    )
    store = InMemoryStore()
    store.save_profile(default_simulated_profile())
    context = ApplicationContext(store=store)
    spec = next(item for item in mod.VARIANTS if item.variant == variant)
    activity = store_script(
        {"activity": lambda arguments: activity_read(store, arguments)}
    )

    def cpu_python(script: str, *arguments: str) -> dict[str, Any]:
        if script == mod.NODE_LOG_ENTRY_PROBE:
            return entries_read(store, arguments)
        return activity(script, *arguments)

    regional = FakeRegional(cpu_python=cpu_python)
    settings = make_settings(spec, node=NODE)
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    node = JournalNode(tmp_path, ctx.run_id, context)
    ctx.probes[f"kernel-replay-journal@{NODE}"] = FakeProbe(  # type: ignore[assignment]
        {
            "install-records": node.install,
            "open-window": {"after": {"ActiveState": "active"}, "boot_id": BOOT},
            "arm": node.arm,
            "status": node.status,
            "close-window": {"clean": True},
        }
    )
    return ctx, node, store


@pytest.mark.parametrize(
    "variant,source,boot",
    [("user_text", "journal", BOOT), ("old_boot", "dmesg", mod.OLD_BOOT_ID)],
)
def test_the_gate_refuses_the_record_the_product_received(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    source: str,
    boot: str,
) -> None:
    ctx, node, store = negative(tmp_path, monkeypatch, variant)
    result = getattr(mod, variant)(ctx)

    [response] = node.responses
    assert response["findings"] == [] and response["incident_ids"] == [], response
    assert (
        store.list_incidents_by_state(CLUSTER, list(IncidentState), node_ids={NODE})
        == []
    ), "the product graded the record into nothing"
    assert result["verdict"] == "PASS", result["errors"]
    [entry] = result["received"]
    assert entry["entry_id"].startswith(f"gfrs-replay-{ctx.run_id}-0-"), entry
    assert (entry["source"], entry["boot_id"]) == (source, boot), entry


@pytest.mark.parametrize("variant", ["user_text", "old_boot"])
def test_a_record_that_never_reached_the_control_plane_is_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    """The wrapper counted its emissions, the collector never delivered the
    batch -- the shape the old oracle passed on."""

    ctx, node, _store = negative(tmp_path, monkeypatch, variant)
    node.polls_after_arm = 0
    node.state.write_text(json.dumps({"0": 1}), encoding="utf-8")
    result = getattr(mod, variant)(ctx)
    assert node.responses == [], "nothing was posted"
    assert result["verdict"] == "INCONCLUSIVE", result
    assert result["stages"] == {"no_finding": [], "not_isolated": [], "hardware": []}, (
        "the stages alone read clean: the old verdict"
    )


@pytest.mark.parametrize(
    "grade",
    [
        (Severity.WARNING, RecoveryAction.RUN_DIAGNOSTICS),
        (Severity.CRITICAL, RecoveryAction.QUARANTINE),
    ],
    ids=["diagnostic", "quarantine"],
)
def test_a_gate_that_let_the_record_through_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    grade: tuple[Severity, RecoveryAction],
) -> None:
    """A product whose trust gate graded the user-space line: a diagnostic
    finding passed the old stage, which counted QUARANTINE incidents only."""

    monkeypatch.setattr(
        "gpu_fault.host_health.log_rule_decision", lambda *_arguments: grade
    )
    ctx, node, _store = negative(tmp_path, monkeypatch, "user_text")
    result = mod.user_text(ctx)
    [response] = node.responses
    assert len(response["findings"]) == 1, response
    assert result["verdict"] == "FAIL", result
    assert any(
        error.startswith(f"the reviewed record gfrs-replay-{ctx.run_id}-0-")
        for error in result["stages"]["no_finding"]
    ), result["stages"]
