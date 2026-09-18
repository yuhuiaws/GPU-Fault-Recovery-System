"""The exact emitted Store probe widens only identity-bound raw evidence."""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.store import InMemoryStore
from gpu_fault.telemetry import EvidenceKind, RawEvidenceRecord
from scripts.e2e.regional.collector_window_fixture import CollectorWindowFixture
from scripts.e2e.regional.kmsg_clock import KMSG_CLOCK_SKEW_SECONDS
from tests._builders import fault_incident, workflow_request

NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)
MARKER = "c018-unique-a1"


def snapshot(tmp_path, monkeypatch, *, marker: str):
    store = InMemoryStore()
    for record_id, delta, message, cluster, node in (
        ("current", -1, f"marker={MARKER} ", "cluster-a", "node-a"),
        (
            "too-old",
            -KMSG_CLOCK_SKEW_SECONDS - 1,
            f"marker={MARKER} ",
            "cluster-a",
            "node-a",
        ),
        ("different", -1, "marker=another-run ", "cluster-a", "node-a"),
        ("prefix", -1, f"marker={MARKER}-foreign ", "cluster-a", "node-a"),
        ("other-cluster", -1, f"marker={MARKER} ", "cluster-b", "node-a"),
        ("other-node", -1, f"marker={MARKER} ", "cluster-a", "node-b"),
    ):
        store.save_raw_evidence(
            RawEvidenceRecord(
                record_id=record_id,
                cluster_id=cluster,
                node_id=node,
                kind=EvidenceKind.NVIDIA_KERNEL,
                observed_at=NOW + timedelta(seconds=delta),
                ingested_at=NOW,
                expires_at=NOW + timedelta(hours=1),
                payload={"message": message},
            ),
            max_records_per_node=100,
        )
    incident = fault_incident("old-incident", "old-event", workflow_request_id="old")
    store.save_incident(incident)
    store.save_workflow(
        workflow_request(
            "old", incident.incident_id, created_at=NOW - timedelta(seconds=1)
        )
    )
    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        classmethod(lambda cls: SimpleNamespace(store=store)),
    )
    calls = []

    def cpu_python(source, *arguments):
        calls.append(arguments)
        output = io.StringIO()
        with monkeypatch.context() as local:
            local.setattr(sys, "argv", ["probe", *arguments])
            with redirect_stdout(output):
                exec(source, {"__name__": "__main__"})
        return json.loads(output.getvalue())

    kubeconfig = tmp_path / "unused-kubeconfig"
    kubeconfig.touch()
    regional = SimpleNamespace(
        cpu_python=cpu_python,
        settings=SimpleNamespace(
            cluster_id="cluster-a",
            gpu_kubeconfig=kubeconfig,
            gpu_context="unused-context",
            namespace="gpu-fault-system",
        ),
    )
    fixture = CollectorWindowFixture(
        regional,
        node="node-a",
        image="example.invalid/probe@sha256:" + "a" * 64,
        case_id="GF-REGIONAL-COLLECT-018",
        run_id="clock-alignment",
        case_dir=tmp_path,
    )
    result = fixture.node_activity(
        NOW, evidence_kind="NVIDIA_KERNEL", evidence_marker=marker
    )
    return result, calls


@pytest.mark.parametrize("marker", ["", MARKER], ids=["unbound", "marker-bound"])
def test_kernel_clock_allowance_is_applied_once_and_only_to_the_matching_line(
    tmp_path, monkeypatch, marker
):
    result, calls = snapshot(tmp_path, monkeypatch, marker=marker)
    assert result["workflows"] == [] and result["incidents"] == [], (
        "kernel clock allowance must not broaden the control-plane activity window"
    )
    assert [item["record_id"] for item in result["evidence"]] == (
        ["current"] if marker else []
    ), "only the exact marker, cluster and node may use the bounded earlier clock"
    original_since, raw_since = calls[0][2], calls[0][4]
    assert original_since == NOW.isoformat(), "the original event time stays intact"
    expected = NOW - timedelta(seconds=KMSG_CLOCK_SKEW_SECONDS if marker else 0)
    assert raw_since == expected.isoformat(), (
        "neither the caller nor the emitted probe may subtract the allowance twice"
    )
