"""GF-GENERAL-LIVE-008/009 injector/verdict contract over scripted records.

The 008 link-down chain is also driven over the real product: a real host
collector over a fake ``/sys/class/net``, a real control plane over an
in-memory store, the real executor on the simulated Runtime Profile and the
real node-activity read.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.collectors import CollectorContext, HostTelemetryCollector
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation, WorkflowStatus
from gpu_fault.store import InMemoryStore
from scripts.e2e.regional import general_live_network as net
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.general_live_store import ACTIVITY_PROBE
from tests._builders import active_workflow_executor, asgi_client, execute_workflow
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    activity_read,
    host_batch,
    incident,
    make_context,
    store_script,
    workflow,
)

SPECS = {spec.variant: spec for spec in net.VARIANTS}
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
IFA = "gfrsaaaaaaa"
IFB = "gfrsaaaaaab"
IFC = "gfrsaaaaaac"
IFD = "gfrsaaaaaad"


def survey(required: str = "") -> dict[str, Any]:
    return {
        "cpu_count": 8,
        "tools": {name: True for name in ("ip", "tc", "systemd-run")},
        "modules": {
            "veth": {"loaded": True, "available": True},
            "dummy": {"loaded": True, "available": True},
            "sch_netem": {"loaded": True, "available": True},
        },
        "backing": [],
        "collector_env": {
            "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
            "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
            "GPU_FAULT_REQUIRED_INTERFACES": required,
        },
        "residue": {"units": [], "paused_units": {}, "paused_processes": {}},
    }


def net_probe(down: str | None = None, **extra: Any) -> FakeProbe:
    operstate = {IFA: "up", IFB: "up", IFC: "up", IFD: "up"}
    if down:
        operstate[down] = "down"
    script = {
        "create-veth": [
            {"a": IFA, "b": IFB, "link_a": {"operstate": "up"}},
            {"a": IFC, "b": IFD, "link_a": {"operstate": "up"}},
        ],
        "create-dummy": {"name": "gfrsaaaaaad", "link": {"operstate": "unknown"}},
        "set-link": lambda args: {
            "name": args[2],
            "state": args[-1],
            "link": {"operstate": args[-1]},
        },
        "delete-interface": {"deleted": IFA, "present": False},
        "start-drops": {"unit": "gfrs-x-drops"},
        "stop-drops": {"stopped": "gfrs-x-drops"},
        "reading": {
            "interfaces": {
                name: {"operstate": state} for name, state in operstate.items()
            }
        },
        "cleanup": {"clean": True},
    }
    script.update(extra)
    return FakeProbe(script)


def node_probe(
    survey_value: dict[str, Any], oracle: dict[str, Any] | None = None
) -> FakeProbe:
    return FakeProbe(
        {
            "survey": survey_value,
            "oracle-start": {"unit": "oracle"},
            "oracle-stop": {},
            "oracle-read": oracle or {"records": [], "next_offset": 0, "total": 0},
        }
    )


def window_probe() -> FakeProbe:
    return FakeProbe(
        {
            "open-window": {"after": {"ActiveState": "active"}},
            "close-window": {"after": {"ActiveState": "active"}},
        }
    )


def probes(node: FakeProbe, netp: FakeProbe, win: FakeProbe) -> dict[str, FakeProbe]:
    return {f"node@{NODE}": node, f"net@{NODE}": netp, f"window@{NODE}": win}


def quarantine_doc(device: str) -> dict[str, Any]:
    """The isolated link, and the pair read up again once it is put back.

    ``interface_missing`` recreates the deleted pair before it closes the
    window and waits for fresh ``network_link_down=0`` readings of both ends;
    the other variants never look at them.
    """

    inc = incident(
        "inc-l",
        "network_link_down",
        device=device,
        state="QUARANTINED",
        workflow="wf-l",
    )
    wf = workflow(
        "wf-l",
        "inc-l",
        ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"],
        status="SUCCEEDED",
    )
    return {
        "incidents": [inc],
        "workflows": [wf],
        "batches": [
            host_batch(
                f"2026-10-24T00:1{minute}:00+00:00",
                "network_link_down",
                0.0,
                device=end,
            )
            for minute in range(2)
            for end in (IFA, IFB)
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": [],
    }


def drops_doc(device: str) -> dict[str, Any]:
    inc = incident(
        "inc-d",
        "network_drops_delta",
        device=device,
        state="ACTION_PENDING",
        workflow="wf-d",
    )
    wf = workflow(
        "wf-d",
        "inc-d",
        ["FREEZE_EVIDENCE", "VALIDATE_FABRIC"],
        status="FAILED",
        executions=[
            {
                "operation": "VALIDATE_FABRIC",
                "status": "FAILED",
                "details": {"host_diagnostic_failure": True},
            }
        ],
    )
    batches = [
        host_batch(
            "2026-10-24T00:02:00+00:00", "network_drops_delta", 50.0, device=device
        )
    ]
    batches += [
        host_batch(
            f"2026-10-24T00:2{i}:00+00:00", "network_drops_delta", 0.0, device=device
        )
        for i in range(2)
    ]
    return {
        "incidents": [inc],
        "workflows": [wf],
        "batches": batches,
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": [],
    }


def restore_patch(
    monkeypatch: pytest.MonkeyPatch,
    operations=("VALIDATE_FABRIC", "RESTORE_SCHEDULING"),
) -> None:
    monkeypatch.setattr(
        net,
        "restore_incidents",
        lambda ctx, incidents, reason: [
            {"status": "SUCCEEDED", "operations": list(operations)}
        ],
    )


def test_required_link_down_quarantines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restore_patch(monkeypatch)
    spec = SPECS["required_link_down"]
    regional = FakeRegional(cpu_python=store_script({"activity": quarantine_doc(IFA)}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey()), net_probe(down=IFA), window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_required_link_down_inconclusive_if_link_stays_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restore_patch(monkeypatch)
    spec = SPECS["required_link_down"]
    regional = FakeRegional(cpu_python=store_script({"activity": quarantine_doc(IFA)}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey()), net_probe(down=None), window_probe()),
    )
    with pytest.raises(VariantInconclusive):
        spec.execute(ctx)


def test_interface_missing_quarantines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restore_patch(monkeypatch)
    spec = SPECS["interface_missing"]
    regional = FakeRegional(cpu_python=store_script({"activity": quarantine_doc(IFA)}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey()), net_probe(), window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_unknown_state_is_collection_error_no_quarantine(tmp_path: Path) -> None:
    spec = SPECS["unknown_state"]
    doc = {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": [
            {
                "collector": "HOST_TELEMETRY",
                "last_error_at": "2026-10-24T00:05:00+00:00",
                "errors": ["unknown operstate"],
            }
        ],
    }
    regional = FakeRegional(cpu_python=store_script({"activity": doc}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey()), net_probe(), window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_empty_required_list_is_a_site_fail(tmp_path: Path) -> None:
    spec = SPECS["empty_required_list"]
    regional = FakeRegional(cpu_python=store_script({"activity": {}}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey(required="")), net_probe(), window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL"
    assert spec.proof_mode == "observation"


def test_empty_required_list_passes_if_configured(tmp_path: Path) -> None:
    spec = SPECS["empty_required_list"]
    regional = FakeRegional(cpu_python=store_script({"activity": {}}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(
            node_probe(survey(required="enp71s0")), net_probe(), window_probe()
        ),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS"


def test_sustained_drops_passes(tmp_path: Path) -> None:
    spec = SPECS["sustained_drops"]
    oracle = {
        "records": [
            {"netdev": {IFA: {"tx_dropped": 100 * i}, IFB: {"rx_dropped": 100 * i}}}
            for i in range(1, 20)
        ],
        "next_offset": 19,
        "total": 19,
    }
    regional = FakeRegional(cpu_python=store_script({"activity": drops_doc(IFA)}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey(), oracle), net_probe(), window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_sustained_drops_inconclusive_if_counters_flat(tmp_path: Path) -> None:
    spec = SPECS["sustained_drops"]
    oracle = {
        "records": [
            {"netdev": {IFA: {"tx_dropped": 0}, IFB: {"rx_dropped": 0}}}
            for _ in range(10)
        ],
        "next_offset": 10,
        "total": 10,
    }
    regional = FakeRegional(cpu_python=store_script({"activity": drops_doc(IFA)}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey(), oracle), net_probe(), window_probe()),
    )
    with pytest.raises(VariantInconclusive):
        spec.execute(ctx)


def test_wrong_interface_binds_to_b(tmp_path: Path) -> None:
    spec = SPECS["wrong_interface"]
    regional = FakeRegional(cpu_python=store_script({"activity": drops_doc(IFC)}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey()), net_probe(), window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["device"] == IFC


RESET_ERROR = (
    f"_network: HostObservationError: {IFA}: rx_dropped reset (400 -> 0); delta unknown"
)


def _soon(seconds: float) -> str:
    """Batches must postdate the runner's real-clock recreate of the interface."""

    from datetime import datetime, timedelta, timezone

    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _reset_batch(
    seconds: float, samples: list[dict[str, Any]], errors: tuple[str, ...] = ()
) -> dict[str, Any]:
    at = _soon(seconds)
    return {
        "observed_at": at,
        "batch_id": f"host-{at}",
        "edge_filter_reasons": [],
        "collection_errors": list(errors),
        "samples": samples,
    }


def _drops(value: float, device: str = IFA) -> dict[str, Any]:
    return {
        "name": "network_drops_delta",
        "device": device,
        "value": value,
        "labels": {},
    }


def reset_doc(*, marked: bool, reset_tick_reading: bool = False) -> dict[str, Any]:
    """Ramp, the reset tick, a normal tick after it.

    ``marked``: the reset tick carries the ``_network`` reset error and no drops
    reading for the interface (batch 173); otherwise the old clamp shipped a 0.
    ``reset_tick_reading`` puts a drops reading into the marked reset tick.
    """

    reset_tick = (
        [_drops(0.0)] if (not marked or reset_tick_reading) else [_drops(7.0, IFB)]
    )
    return {
        "incidents": [],
        "workflows": [],
        "batches": [
            _reset_batch(30, [_drops(400.0)]),
            _reset_batch(60, reset_tick, errors=(RESET_ERROR,) if marked else ()),
            _reset_batch(90, [_drops(0.0)]),
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": [],
    }


def _run_counter_reset(tmp_path: Path, document: dict[str, Any]) -> dict[str, Any]:
    spec = SPECS["counter_reset"]
    regional = FakeRegional(cpu_python=store_script({"activity": document}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey()), net_probe(), window_probe()),
    )
    return spec.execute(ctx)


def test_counter_reset_reads_the_reset_tick_as_unknown(tmp_path: Path) -> None:
    result = _run_counter_reset(tmp_path, reset_doc(marked=True))
    assert result["verdict"] == "PASS", result["errors"]
    assert result["stages"]["reset_unknown"] == []
    spec = SPECS["counter_reset"]
    assert spec.expected_outcome == "PASS" and spec.proof_mode == "native"
    assert "delta unknown" in spec.expected and "clamped" in spec.expected


def test_counter_reset_fails_on_the_old_clamp(tmp_path: Path) -> None:
    result = _run_counter_reset(tmp_path, reset_doc(marked=False))
    assert result["verdict"] == "FAIL", result
    assert any(
        "old clamp" in error and "delta unknown" in error for error in result["errors"]
    ), result["errors"]


def test_counter_reset_fails_on_a_reading_in_the_reset_tick(tmp_path: Path) -> None:
    result = _run_counter_reset(
        tmp_path, reset_doc(marked=True, reset_tick_reading=True)
    )
    assert result["verdict"] == "FAIL", result
    assert any("in the reset tick" in error for error in result["errors"]), result[
        "errors"
    ]


def test_counter_reset_is_inconclusive_without_evidence(tmp_path: Path) -> None:
    empty = reset_doc(marked=True)
    empty["batches"] = []
    with pytest.raises(VariantInconclusive, match="surviving baseline"):
        _run_counter_reset(tmp_path, empty)


def test_reset_evidence_keeps_only_the_interfaces_own_records() -> None:
    from datetime import datetime, timedelta, timezone

    since = datetime.now(timezone.utc) - timedelta(seconds=5)
    document = reset_doc(marked=True)
    evidence = net.reset_evidence(document, IFA, since)
    assert evidence["errors"] == [RESET_ERROR]
    assert evidence["reset_tick"] == [], "the peer's reading is not the interface's"
    assert [item["value"] for item in evidence["readings"]] == [400.0, 0.0]
    assert net.reset_evidence(document, IFB, since)["errors"] == [], (
        "an error naming another interface does not count"
    )


def test_short_drops_counter_reset_and_recovery(tmp_path: Path) -> None:
    oracle = {
        "records": [
            {"netdev": {IFA: {"tx_dropped": 100 * i}, IFB: {"rx_dropped": 100 * i}}}
            for i in range(1, 8)
        ],
        "next_offset": 7,
        "total": 7,
    }
    empty = {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": [],
    }
    spec = SPECS["short_drops"]
    regional = FakeRegional(cpu_python=store_script({"activity": empty}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey(), oracle), net_probe(), window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    # recovery: a finding then fresh zeros
    spec = SPECS["recovery"]
    zero = drops_doc(IFA)
    regional = FakeRegional(cpu_python=store_script({"activity": zero}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey(), oracle), net_probe(), window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_optional_interface_passes(tmp_path: Path) -> None:
    spec = SPECS["optional_interface"]
    empty = {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": [],
    }
    regional = FakeRegional(cpu_python=store_script({"activity": empty}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey()), net_probe(), window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_deferred_and_capability_variants() -> None:
    # statistics_failed and errors_delta are now implemented as live-replay
    # (lane K) via the host-collector statistics bind; bond_vf stays absent.
    assert SPECS["statistics_failed"].proof_mode == "live-replay", (
        "statistics_failed is a live-replay variant"
    )
    assert SPECS["statistics_failed"].implemented, "statistics_failed is implemented"
    assert SPECS["errors_delta"].implemented, "errors_delta is implemented"
    assert SPECS["bond_vf"].deferred.startswith("CAPABILITY_ABSENT"), (
        "SPECS['bond_vf'].deferred.startswith('CAPABILITY_ABSENT')"
    )


# --------------------------------------------------------------------------- #
# 008 statistics_failed / 009 errors_delta: host-collector statistics bind
# (lane K live-replay). A source-replay FakeProbe stands in for the bind host.
# --------------------------------------------------------------------------- #
def source_probe(**extra: Any) -> FakeProbe:
    script = {
        "install-source": {"name": "rx-errors"},
        "open-window": {
            "after": {"ActiveState": "active"},
            "proof_scope": "live-replay",
        },
        "set-value": {"value": "5000"},
        "status": {"fixtures": {"rx-errors": "5000"}},
        "close-window": {"clean": True},
    }
    script.update(extra)
    return FakeProbe(script)


def probes5(
    node: FakeProbe, netp: FakeProbe, win: FakeProbe, src: FakeProbe
) -> dict[str, FakeProbe]:
    return {
        f"node@{NODE}": node,
        f"net@{NODE}": netp,
        f"window@{NODE}": win,
        f"source-replay@{NODE}": src,
    }


def _errors_doc(device: str) -> dict[str, Any]:
    inc = incident(
        "inc-e",
        "network_errors_delta",
        device=device,
        state="ACTION_PENDING",
        workflow="wf-e",
    )
    wf = workflow(
        "wf-e", "inc-e", ["FREEZE_EVIDENCE", "VALIDATE_FABRIC"], status="RUNNING"
    )
    return {
        "incidents": [inc],
        "workflows": [wf],
        "batches": [],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": [],
    }


def test_errors_delta_steps_a_bound_counter(tmp_path: Path) -> None:
    spec = SPECS["errors_delta"]
    regional = FakeRegional(cpu_python=store_script({"activity": _errors_doc(IFA)}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes5(
            node_probe(survey()), net_probe(), window_probe(), source_probe()
        ),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["proof_scope"] == "live-replay", result


def test_statistics_failed_keeps_link_sample(tmp_path: Path) -> None:
    spec = SPECS["statistics_failed"]
    doc = {
        "incidents": [],
        "workflows": [],
        "batches": [
            host_batch("2026-10-24T00:03:00+00:00", "network_link_up", 1, device=IFA)
        ],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": [],
    }
    regional = FakeRegional(cpu_python=store_script({"activity": doc}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes5(
            node_probe(survey()), net_probe(), window_probe(), source_probe()
        ),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_statistics_failed_flags_a_dropped_link_sample(tmp_path: Path) -> None:
    spec = SPECS["statistics_failed"]
    doc = {
        "incidents": [],
        "workflows": [],
        "batches": [],
        "markers": [],
        "commands": [],
        "health_signals": {},
        "notifications": [],
        "collector_statuses": [],
    }
    regional = FakeRegional(cpu_python=store_script({"activity": doc}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes5(
            node_probe(survey()), net_probe(), window_probe(), source_probe()
        ),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("link sample" in error for error in result["errors"]), result["errors"]


# --------------------------------------------------------------------------- #
# 008 required_link_down / interface_missing: the veth peer goes down with a
# (live 2026-10-25, ip-10-0-32-162, release 6a95b90fcd82: "chain: no
# network_link_down on gfrs0d3ec4a finding reached an incident" while the
# product had quarantined the node under exactly that incident).
# --------------------------------------------------------------------------- #
#: The reason the product appends to the incumbent incident when the peer's
#: finding joins it (``families/drain.py`` ``ingest_terminal_node_quarantine``).
PEER_REASON = f"{NODE}: network_link_down: required network interface is down"


def grouped_doc(
    device: str, *, state: str = "QUARANTINED", status: str = "SUCCEEDED"
) -> dict[str, Any]:
    """The records once the peer's finding joined ``device``'s incident.

    ``ip link set a down`` leaves the veth peer ``lowerlayerdown`` and ``ip
    link del a`` removes both ends; both are declared required, so a second
    critical finding comes in the same tick and the product folds it into the
    first one's isolation: same ``inc-<event id>``, same minting event id,
    re-labelled NODE_HEALTH_GROUP.
    """

    doc = quarantine_doc(device)
    doc["incidents"] = [
        {
            **doc["incidents"][0],
            "event_type": "NODE_HEALTH_GROUP",
            "state": state,
            "reasons": ["required network interface is down", PEER_REASON],
        }
    ]
    doc["workflows"] = [{**doc["workflows"][0], "status": status}]
    return doc


def peer_down_probe() -> FakeProbe:
    return net_probe(
        down=IFA,
        reading={
            "interfaces": {
                IFA: {"operstate": "down", "flags": "0x1002"},
                IFB: {"operstate": "lowerlayerdown", "flags": "0x1003"},
            }
        },
    )


def restore_recorder(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    restored: list[list[str]] = []

    def record(ctx, incidents, reason):
        restored.append([str(item["incident_id"]) for item in incidents])
        return [
            {
                "status": "SUCCEEDED",
                "operations": ["VALIDATE_FABRIC", "RESTORE_SCHEDULING"],
            }
        ]

    monkeypatch.setattr(net, "restore_incidents", record)
    return restored


@pytest.mark.parametrize("variant", ["required_link_down", "interface_missing"])
def test_link_down_takes_the_incident_its_peer_finding_joined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    restored = restore_recorder(monkeypatch)
    spec = SPECS[variant]
    regional = FakeRegional(cpu_python=store_script({"activity": grouped_doc(IFA)}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey()), peer_down_probe(), window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert restored == [["inc-l"]], (
        f"the isolation incident must be handed back through the validated restore: "
        f"{restored}"
    )


@pytest.mark.parametrize("variant", ["required_link_down", "interface_missing"])
def test_link_down_grades_the_chain_once_the_isolation_settled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    # The detection read is the first to list the incident; the isolation
    # workflow finishes seconds later, so that read can still show it
    # ACTION_PENDING behind a RUNNING workflow.
    restore_patch(monkeypatch)
    spec = SPECS[variant]
    running = quarantine_doc(IFA)
    running["incidents"] = [{**running["incidents"][0], "state": "ACTION_PENDING"}]
    running["workflows"] = [{**running["workflows"][0], "status": "RUNNING"}]
    regional = FakeRegional(
        cpu_python=store_script({"activity": [running, quarantine_doc(IFA)]})
    )
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey()), peer_down_probe(), window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_link_down_hands_back_an_incident_the_peer_minted_on_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A peer finding the product took in a later tick, once the first
    # isolation had ended, opens an incident of its own: this variant caused
    # it too, so it goes back through the validated restore as well.
    restored = restore_recorder(monkeypatch)
    spec = SPECS["required_link_down"]
    doc = quarantine_doc(IFA)
    peer = incident(
        "inc-p", "network_link_down", device=IFB, state="QUARANTINED", workflow="wf-p"
    )
    doc["incidents"].append(peer)
    doc["workflows"].append(
        workflow(
            "wf-p",
            "inc-p",
            ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"],
            status="SUCCEEDED",
        )
    )
    regional = FakeRegional(cpu_python=store_script({"activity": doc}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey()), peer_down_probe(), window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert restored == [["inc-l", "inc-p"]], restored


class SimulatedRuntime:
    """Runs every step the simulated Runtime Profile owns, successfully."""

    def __init__(self) -> None:
        self.operations: list[str] = []

    def supports(self, step: Any) -> bool:
        return step.execution_owner == "simulated-runtime"

    def execute(self, context: Any) -> WorkflowStepOutcome:
        self.operations.append(context.step.operation.value)
        return WorkflowStepOutcome.succeeded(
            operation_id=f"simulated-{context.idempotency_key}"
        )


#: Another run's test pair, isolated on the node before this one.
EARLIER_PAIR = ("gfrsbbbbbba", "gfrsbbbbbbb")


class LinkDownWorld:
    """The node's host collector and the control plane, both real.

    The window probe's ``open-window`` (re)starts a real host collector with
    the pair declared required over a fake ``/sys/class/net``; ``set-link``
    flips the pair as the kernel does -- the downed end reads ``down``, its
    peer ``lowerlayerdown`` (NO-CARRIER) -- ``delete-interface`` removes both
    ends and ``create-veth`` puts them back. The collector sees a change at
    its next tick, which runs before each of the runner's reads (one collector
    interval per poll), never inside the probe call: a window closed right
    after a probe call restarts the collector before it read the change.
    Every batch the collector delivers is posted to a real
    ``ApplicationContext`` over an in-memory store, every workflow it plans is
    run by the real executor, and the runner's reads run the real
    ``ACTIVITY_PROBE`` there.
    """

    def __init__(self, root: Path) -> None:
        self.net = root / "net"
        self.interface("enp71s0", physical=True)
        self.collector: HostTelemetryCollector | None = None
        self.delivered: list[dict[str, Any]] = []
        self.context = ApplicationContext(store=InMemoryStore())
        self.runtime = SimulatedRuntime()
        self.executor = active_workflow_executor(
            self.context.store,
            [self.runtime],
            [
                WorkflowOperation.FREEZE_EVIDENCE,
                WorkflowOperation.MARK_UNSCHEDULABLE,
                WorkflowOperation.QUARANTINE,
            ],
        )

    def interface(self, name: str, *, physical: bool = False) -> None:
        statistics = self.net / name / "statistics"
        statistics.mkdir(parents=True, exist_ok=True)
        for counter in ("rx_errors", "tx_errors", "rx_dropped", "tx_dropped"):
            (statistics / counter).write_text("0\n")
        if physical:
            (self.net / name / "device").mkdir(exist_ok=True)
        self.operstate(name, "up")

    def operstate(self, name: str, state: str) -> None:
        (self.net / name / "operstate").write_text(f"{state}\n")

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        async def send() -> Any:
            async with asgi_client(self.context) as client:
                return await client.post(path, json=payload)

        response = asyncio.run(send())
        assert response.status_code == 200, response.text
        self.delivered.append(payload)
        for item in self.context.store.list_workflows(limit=50, newest_first=True):
            if item.status is WorkflowStatus.PENDING:
                execute_workflow(
                    self.executor,
                    item.request_id,
                    expected_fencing_token=item.fencing_token,
                )
        return response.json()

    def host_collector(self, required: list[str]) -> HostTelemetryCollector:
        return HostTelemetryCollector(
            self,
            CollectorContext(
                cluster_id="gpu-eks-cluster", runtime_profile_version="simulated-v1"
            ),
            node_id=NODE,
            net_class_root=str(self.net),
            required_interfaces=required,
            edge_filter_enabled=True,
            health_summary_seconds=300,
        )

    def open_window(self, _arguments: tuple[str, ...]) -> dict[str, Any]:
        self.collector = self.host_collector([IFA, IFB])
        self.collector.collect_once()
        return {"after": {"ActiveState": "active"}}

    def earlier_isolation(self) -> str:
        """An earlier run's required pair on this node: isolated, peer merged,
        ended. From then on the node's merge group names that incident (live:
        ip-10-0-43-5 had run other cases before 008 interface_missing)."""

        first, peer = EARLIER_PAIR
        for name in EARLIER_PAIR:
            self.interface(name)
        self.collector = self.host_collector(list(EARLIER_PAIR))
        self.collector.collect_once()
        self.operstate(first, "down")
        self.operstate(peer, "lowerlayerdown")
        self.tick()
        down = self.delivered[-1]
        self.operstate(first, "up")
        self.operstate(peer, "up")
        self.tick()
        for name in EARLIER_PAIR:
            shutil.rmtree(self.net / name)
        self.collector = None
        self.delivered.clear()
        self.runtime.operations.clear()
        return f"inc-{down['batch_id']}-network_link_down-{first}"

    def tick(self) -> None:
        assert self.collector is not None, "the window opens before the link moves"
        self.collector.collect_once()

    def set_link(self, arguments: tuple[str, ...]) -> dict[str, Any]:
        state = arguments[arguments.index("--state") + 1]
        self.operstate(IFA, state)
        self.operstate(IFB, "up" if state == "up" else "lowerlayerdown")
        return {"name": IFA, "state": state, "link": {"operstate": state}}

    def create_veth(self, _arguments: tuple[str, ...]) -> dict[str, Any]:
        for name in (IFA, IFB):
            self.interface(name)
        return {"a": IFA, "b": IFB, "link_a": {"operstate": "up"}}

    def delete_interface(self, _arguments: tuple[str, ...]) -> dict[str, Any]:
        for name in (IFA, IFB):
            shutil.rmtree(self.net / name)
        return {"deleted": IFA, "present": False}

    def close_window(self, _arguments: tuple[str, ...]) -> dict[str, Any]:
        # The unit restarts without the pair: nothing reads it any more.
        self.collector = None
        return {"after": {"ActiveState": "active"}}

    def pair_readings(self) -> dict[str, float]:
        """The product's latest ``network_link_down`` of each end of the pair."""

        return {
            str(item.device): item.value
            for item in self.context.store.list_telemetry_metrics_latest(
                "gpu-eks-cluster", NODE
            )
            if item.name == "network_link_down" and item.device in (IFA, IFB)
        }

    def reading(self, _arguments: tuple[str, ...]) -> dict[str, Any]:
        return {
            "interfaces": {
                name: {"operstate": (self.net / name / "operstate").read_text().strip()}
                if (self.net / name).exists()
                else {"present": False}
                for name in (IFA, IFB)
            }
        }

    def activity(self, script: str, *arguments: str) -> dict[str, Any]:
        assert script is ACTIVITY_PROBE, "the variant reads only node activity"
        if self.collector is not None:
            self.tick()
        return activity_read(self.context.store, arguments)


@pytest.mark.parametrize("earlier", [False, True], ids=["first", "after-earlier"])
@pytest.mark.parametrize(
    "variant", ["required_link_down", "interface_missing", "recovery"]
)
def test_link_down_over_the_real_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str, earlier: bool
) -> None:
    monkeypatch.setattr(HostTelemetryCollector, "CONTRIBUTORS", ("_network",))
    world = LinkDownWorld(tmp_path)
    previous = world.earlier_isolation() if earlier else None
    restored: list[list[str]] = []
    held: list[dict[str, float]] = []

    def restore(ctx, incidents, reason):
        restored.append([str(item["incident_id"]) for item in incidents])
        held.append(world.pair_readings())
        return [
            {
                "status": "SUCCEEDED",
                "operations": ["VALIDATE_FABRIC", "RESTORE_SCHEDULING"],
            }
        ]

    monkeypatch.setattr(net, "restore_incidents", restore)
    netp = net_probe(
        **{
            "create-veth": world.create_veth,
            "set-link": world.set_link,
            "delete-interface": world.delete_interface,
            "reading": world.reading,
        }
    )
    window = FakeProbe(
        {"open-window": world.open_window, "close-window": world.close_window}
    )
    spec = next(
        item
        for item in net.VARIANTS
        if item.case_id == "GF-GENERAL-LIVE-008" and item.variant == variant
    )
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=world.activity),
        probes=probes(node_probe(survey()), netp, window),
    )

    result = spec.execute(ctx)

    down = next(
        payload
        for payload in world.delivered
        if "threshold:network_link_down" in payload["edge_filter_reasons"]
    )
    reported = {
        sample["device"]: sample["value"]
        for sample in down["samples"]
        if sample["name"] == "network_link_down"
    }
    assert reported == {IFA: 1.0, IFB: 1.0}, (
        f"the down tick must carry both required ends of the pair: {reported}"
    )
    minted = f"inc-{down['batch_id']}-network_link_down-{IFA}"
    stored = world.context.store.get_incident(minted)
    assert stored.event_type == "NODE_HEALTH_GROUP", (
        f"the peer's finding is folded into the target's incident: {stored}"
    )
    assert stored.state.value == "QUARANTINED", stored
    peer_event = f"{down['batch_id']}-network_link_down-{IFB}"
    linked = world.context.store.get_incident_by_event(peer_event)
    assert linked is not None and linked.incident_id == minted, linked
    assert world.runtime.operations == [
        "FREEZE_EVIDENCE",
        "MARK_UNSCHEDULABLE",
        "QUARANTINE",
    ], world.runtime.operations
    assert result["verdict"] == "PASS", result["errors"]
    assert restored == [[minted]], restored
    # Live 2026-10-25 interface_missing: the window was closed with the pair
    # still deleted, so the product's newest reading of both ends was
    # "missing" and the validated restore ended FAILED on it. The pair is put
    # back (or brought up) first, and the window is only closed once the
    # product has read it up: closing it right after the probe call restarts
    # the collector before its next tick could see the change.
    assert held == [{IFA: 0.0, IFB: 0.0}], (
        f"the validated restore must find both ends read up again: {held}"
    )
    if previous is not None:
        # The node's merge group still named the earlier, ended isolation;
        # the peer's finding joined this run's in-flight one all the same.
        ended = world.context.store.get_incident(previous)
        assert ended.event_type == "NODE_HEALTH_GROUP", ended
        assert ended.state.value == "QUARANTINED", ended
