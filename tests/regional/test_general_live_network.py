"""GF-GENERAL-LIVE-008/009 injector/verdict contract over scripted records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_network as net
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
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
        "batches": [],
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


def test_tcp_only_passes_when_counters_stay_zero(tmp_path: Path) -> None:
    spec = SPECS["tcp_only"]
    oracle = {
        "records": [
            {"tcp_retransmits": 10 * i, "netdev": {IFA: {"tx_errors": 0}}}
            for i in range(1, 12)
        ],
        "next_offset": 11,
        "total": 11,
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
    regional = FakeRegional(cpu_python=store_script({"activity": empty}))
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes=probes(node_probe(survey(), oracle), net_probe(), window_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_tcp_only_inconclusive_if_nic_counters_move(tmp_path: Path) -> None:
    spec = SPECS["tcp_only"]
    oracle = {
        "records": [
            {"tcp_retransmits": 10 * i, "netdev": {IFA: {"tx_errors": i}}}
            for i in range(1, 12)
        ],
        "next_offset": 11,
        "total": 11,
    }
    regional = FakeRegional(cpu_python=store_script({"activity": {}}))
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
    for variant in ("short_drops", "counter_reset"):
        spec = SPECS[variant]
        regional = FakeRegional(cpu_python=store_script({"activity": empty}))
        ctx = make_context(
            spec,
            tmp_path=tmp_path,
            regional=regional,
            probes=probes(node_probe(survey(), oracle), net_probe(), window_probe()),
        )
        result = spec.execute(ctx)
        assert result["verdict"] == "PASS", (variant, result["errors"])
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
        "open-window": {"after": {"ActiveState": "active"}, "proof_scope": "live-replay"},
        "set-value": {"value": "5000"},
        "status": {"fixtures": {"rx-errors": "5000"}},
        "close-window": {"clean": True},
    }
    script.update(extra)
    return FakeProbe(script)


def probes5(node: FakeProbe, netp: FakeProbe, win: FakeProbe, src: FakeProbe) -> dict[str, FakeProbe]:
    return {
        f"node@{NODE}": node,
        f"net@{NODE}": netp,
        f"window@{NODE}": win,
        f"source-replay@{NODE}": src,
    }


def _errors_doc(device: str) -> dict[str, Any]:
    inc = incident("inc-e", "network_errors_delta", device=device, state="ACTION_PENDING", workflow="wf-e")
    wf = workflow("wf-e", "inc-e", ["FREEZE_EVIDENCE", "VALIDATE_FABRIC"], status="RUNNING")
    return {
        "incidents": [inc], "workflows": [wf], "batches": [], "markers": [],
        "commands": [], "health_signals": {}, "notifications": [], "collector_statuses": [],
    }


def test_errors_delta_steps_a_bound_counter(tmp_path: Path) -> None:
    spec = SPECS["errors_delta"]
    regional = FakeRegional(cpu_python=store_script({"activity": _errors_doc(IFA)}))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional,
        probes=probes5(node_probe(survey()), net_probe(), window_probe(), source_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["proof_scope"] == "live-replay", result


def test_statistics_failed_keeps_link_sample(tmp_path: Path) -> None:
    spec = SPECS["statistics_failed"]
    doc = {
        "incidents": [], "workflows": [],
        "batches": [host_batch("2026-10-24T00:03:00+00:00", "network_link_up", 1, device=IFA)],
        "markers": [], "commands": [], "health_signals": {}, "notifications": [], "collector_statuses": [],
    }
    regional = FakeRegional(cpu_python=store_script({"activity": doc}))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional,
        probes=probes5(node_probe(survey()), net_probe(), window_probe(), source_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_statistics_failed_flags_a_dropped_link_sample(tmp_path: Path) -> None:
    spec = SPECS["statistics_failed"]
    doc = {
        "incidents": [], "workflows": [], "batches": [],
        "markers": [], "commands": [], "health_signals": {}, "notifications": [], "collector_statuses": [],
    }
    regional = FakeRegional(cpu_python=store_script({"activity": doc}))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional,
        probes=probes5(node_probe(survey()), net_probe(), window_probe(), source_probe()),
    )
    result = spec.execute(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("link sample" in error for error in result["errors"]), result["errors"]
