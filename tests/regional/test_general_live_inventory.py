"""GF-GENERAL-LIVE-013/014 inventory: category-module verdict tests.

013 unreadable_state drives the host-collector PCI-root repoint (a source-replay
FakeProbe) and reads the host telemetry back: a malformed reading is UNKNOWN,
never a missing-device finding or a reboot. Every 014 variant (and the rest of
013) stays NOT_RUN with its reason -- those are asserted to keep the denominator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.host_health import HostMetricSample, efa_inventory_rule
from gpu_fault.models import RecoveryAction, Severity
from scripts.e2e.regional import general_live_inventory as mod
from scripts.e2e.regional.general_live_fixture import (
    VariantBlocked,
    VariantInconclusive,
)
from scripts.e2e.regional.general_live_registry import coverage_rows
from scripts.e2e.regional.probes import general_live_source_probe as probe
from tests.collectors import _cov95_runtime_collect as cov_support
from tests.collectors import _cov95_runtime_collect_host as cov_host
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
    store_script,
)

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
SPECS = {spec.variant: spec for spec in mod.VARIANTS}
isolated_runtime = cov_support.isolated_runtime
host_case = cov_host.host_case


def _source_probe() -> FakeProbe:
    return FakeProbe(
        {
            "build-pci-fixture": {"efa_devices": 16, "malformed": True},
            "open-window": {
                "after": {"ActiveState": "active"},
                "proof_scope": "live-replay",
            },
            "status": {"fixtures": {}},
            "close-window": {"clean": True},
        }
    )


def _doc(
    *, collection_error: bool, incidents=None, workflows=None
) -> dict[str, object]:
    batch = {
        "observed_at": "2026-10-24T00:03:00+00:00",
        "batch_id": "host-1",
        "edge_filter_reasons": [],
        "collection_errors": (
            ["EFA inventory observation unavailable: invalid PCI device ID"]
            if collection_error
            else []
        ),
        "samples": [],
    }
    return {
        "incidents": incidents or [],
        "workflows": workflows or [],
        "batches": [batch],
        "markers": [],
        "metric_latest": [],
        "notifications": [],
        "collector_statuses": [],
    }


def _ctx(variant: str, *, doc: dict[str, object], node_snapshot=None, tmp_path: Path):
    spec = SPECS[variant]
    regional = FakeRegional(
        cpu_python=store_script({"activity": doc}), node_snapshot=node_snapshot
    )
    return make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"source-replay@{NODE}": _source_probe()},
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )


def test_unreadable_state_is_unknown_not_zero(tmp_path: Path) -> None:
    ctx = _ctx("unreadable_state", doc=_doc(collection_error=True), tmp_path=tmp_path)
    result = mod.unreadable_state(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["proof_scope"] == "live-replay", result


def test_unreadable_state_fails_without_an_unknown_marker(tmp_path: Path) -> None:
    ctx = _ctx("unreadable_state", doc=_doc(collection_error=False), tmp_path=tmp_path)
    result = mod.unreadable_state(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("collection error" in error for error in result["errors"]), result[
        "errors"
    ]


def test_unreadable_state_fails_on_a_reboot(tmp_path: Path) -> None:
    incidents = [
        {
            "incident_id": "inc-efa",
            "event_id": "host0-efa_inventory_mismatch-node",
            "event_type": "NODE_HEALTH",
            "effective_action": "REBOOT_NODE",
            "state": "ACTION_PENDING",
            "created_at": "2026-10-24T00:03:00+00:00",
            "workflow_request_id": "wf-efa",
            "node_ids": [NODE],
        }
    ]
    workflows = [
        {
            "request_id": "wf-efa",
            "incident_id": "inc-efa",
            "status": "RUNNING",
            "created_at": "2026-10-24T00:03:00+00:00",
            "official_steps": [{"operation": "RESTART_NODE", "parameters": {}}],
            "step_executions": [],
        }
    ]
    doc = _doc(collection_error=True, incidents=incidents, workflows=workflows)
    ctx = _ctx("unreadable_state", doc=doc, tmp_path=tmp_path)
    result = mod.unreadable_state(ctx)
    assert result["verdict"] == "FAIL", result
    assert any(
        "RESTART_NODE" in error or "efa_inventory_mismatch" in error
        for error in result["errors"]
    ), result["errors"]


def test_013_and_014_cover_every_variant() -> None:
    for case in ("GF-GENERAL-LIVE-013", "GF-GENERAL-LIVE-014"):
        expected = set(coverage_rows()[case])
        got = {item.variant for item in mod.VARIANTS if item.case_id == case}
        assert got == expected, f"{case} registers every coverage variant"
    implemented = {item.variant for item in mod.VARIANTS if item.execute is not None}
    assert implemented == {
        "unreadable_state",
        "unknown_inventory",
        "excess_device",
        "link_inactive",
    }, implemented
    assert all(
        item.deferred for item in mod.VARIANTS if item.case_id == "GF-GENERAL-LIVE-014"
    ), "014 has no valid replay path; every variant is NOT_RUN"
    for name in ("unknown_inventory", "excess_device", "link_inactive"):
        assert (
            SPECS[name].lanes == ("eks", "hp")
            and SPECS[name].proof_mode == "live-replay"
        )
        assert mod.HOST_SOURCE_REPLAY in SPECS[name].approvals
    assert SPECS["link_inactive"].approvals == (mod.HOST_SOURCE_REPLAY, mod.NODE_REBOOT)
    assert SPECS["link_inactive"].expected_outcome == "FAIL"
    assert "missing_inventory_action_completion" in SPECS["link_inactive"].product_gap
    assert SPECS["excess_device"].expected_outcome == "PASS"


# --------------------------------------------------------------------------- #
# whole-tree /sys/class/infiniband replays: link_inactive, excess_device,
# unknown_inventory (HOST_SOURCE_REPLAY)
# --------------------------------------------------------------------------- #
DEVICES = ("rdmap0s0", "rdmap1s0", "rdmap2s0")
EXPECTED_EFA = len(DEVICES)
ACTIVE = {"state": "4: ACTIVE", "phys_state": "5: LinkUp"}


def _tree_summary(**mutations: Any) -> dict[str, Any]:
    return {
        "path": "/run/gfrs-live/source-x/fixtures/infiniband",
        "source": "/sys/class/infiniband",
        "devices": {
            name: {
                "ports": {"1": dict(ACTIVE)},
                "driver": "efa",
                "counters": 4,
                "netdevs": [],
            }
            for name in DEVICES
        },
        "dropped": [],
        "mutations": list(mutations.get("mutations", [])),
    }


def _tree_probe(**extra: Any) -> FakeProbe:
    script: dict[str, Any] = {
        "build-infiniband-fixture": [_tree_summary(), _tree_summary(mutations=["x"])],
        "open-window": {
            "after": {"ActiveState": "active"},
            "proof_scope": "live-replay",
            "verification": {"ok": True, "checks": [{"kind": "tree", "ok": True}]},
        },
        "set-value": {},
        "status": {"fixtures": {}},
        "close-window": {"clean": True},
    }
    script.update(extra)
    return FakeProbe(script)


def _base_doc(**extra: Any) -> dict[str, Any]:
    return {
        "since": "2026-10-24T00:00:00+00:00",
        "cluster_id": "gpu-eks-cluster",
        "node_id": NODE,
        "incidents": [],
        "workflows": [],
        "batches": [],
        "markers": [],
        "metric_latest": [
            {
                "name": "efa_inventory_expected_count",
                "device": None,
                "value": EXPECTED_EFA,
                "observed_at": "2026-10-24T00:00:00+00:00",
                "labels": {},
            }
        ],
        "notifications": [],
        "collector_statuses": [],
        **extra,
    }


def _efa_batch(
    observed_at: str,
    *,
    failure_mode: str,
    active: int,
    mismatch: int,
    errors: list[str] = (),
) -> dict[str, Any]:
    labels = {
        "resource": "EFA",
        "expected_count": str(EXPECTED_EFA),
        "observed_count": str(active),
        "missing_count": str(max(0, EXPECTED_EFA - active)),
        "excess_count": str(max(0, active - EXPECTED_EFA)),
        "failure_mode": failure_mode,
        "expected_efa_device_count": str(EXPECTED_EFA),
        "driver_bound_count": str(active),
    }
    names = {
        "efa_inventory_expected_count": EXPECTED_EFA,
        "efa_inventory_active_count": active,
        "efa_inventory_missing_count": max(0, EXPECTED_EFA - active),
        "efa_inventory_excess_count": max(0, active - EXPECTED_EFA),
        "efa_inventory_mismatch": mismatch,
    }
    return {
        "observed_at": observed_at,
        "batch_id": f"host-{observed_at}",
        "edge_filter_reasons": [],
        "collection_errors": list(errors),
        "samples": [
            {"name": name, "device": None, "value": value, "labels": labels}
            for name, value in names.items()
        ],
    }


def _efa_incident(
    *, workflow_status: str, executions: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "incidents": [
            {
                "incident_id": "inc-efa",
                "event_id": "host-2-efa_inventory_mismatch-node",
                "event_type": "NODE_HEALTH",
                "effective_action": "RUN_DIAGNOSTICS",
                "state": "RECOVERED"
                if workflow_status != "RUNNING"
                else "ACTION_PENDING",
                "created_at": "2026-10-24T00:00:36+00:00",
                "workflow_request_id": "wf-efa",
                "node_ids": [NODE],
                "cluster_id": "gpu-eks-cluster",
            }
        ],
        "workflows": [
            {
                "request_id": "wf-efa",
                "incident_id": "inc-efa",
                "status": workflow_status,
                "created_at": "2026-10-24T00:00:36+00:00",
                "official_steps": [
                    {"operation": "FREEZE_EVIDENCE", "parameters": {}},
                    {
                        "operation": "VALIDATE_FABRIC",
                        "parameters": {"failure_escalation_action": "REBOOT_NODE"},
                    },
                ],
                "step_executions": executions,
            }
        ],
    }


WAITING = {
    "operation": "VALIDATE_FABRIC",
    "status": "WAITING",
    "details": {"node_pending": {NODE: {"missing_inventory_action_completion": ["x"]}}},
}
SUCCEEDED = {"operation": "VALIDATE_FABRIC", "status": "SUCCEEDED", "details": {}}
FAILED = {
    "operation": "VALIDATE_FABRIC",
    "status": "FAILED",
    "details": {"node_pending": {NODE: {"missing_inventory_action_completion": ["x"]}}},
}
REBOOT_WORKFLOW = {
    "request_id": "wf-reboot",
    "incident_id": "inc-efa",
    "status": "SUCCEEDED",
    "created_at": "2026-10-24T00:11:00+00:00",
    "official_steps": [
        {"operation": "MARK_UNSCHEDULABLE", "parameters": {}},
        {"operation": "RESTART_NODE", "parameters": {}},
        {"operation": "VALIDATE_FABRIC", "parameters": {}},
        {"operation": "RESTORE_SCHEDULING", "parameters": {}},
    ],
    "step_executions": [SUCCEEDED],
}


def _tree_ctx(
    variant: str,
    *,
    docs: list[dict[str, Any]],
    tmp_path: Path,
    probe: FakeProbe | None = None,
) -> Any:
    spec = SPECS[variant]
    regional = FakeRegional(cpu_python=store_script({"activity": docs}))
    return make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"source-replay@{NODE}": probe or _tree_probe()},
        settings=make_settings(spec, node=NODE),
        window_minutes=90,
    )


def _build_calls(probe: FakeProbe) -> list[tuple[str, ...]]:
    return [call for call in probe.calls if call[0] == "build-infiniband-fixture"]


def test_unknown_inventory_is_unknown_without_a_count(tmp_path: Path) -> None:
    error = (
        "_efa_inventory: CollectorError: EFA inventory observation unavailable: missing "
        "or ambiguous RDMA driver identity at /sys/class/infiniband/rdmap0s0"
    )
    window = _base_doc(
        batches=[
            {
                "observed_at": "2026-10-24T00:00:20+00:00",
                "batch_id": "host-1",
                "edge_filter_reasons": ["baseline"],
                "collection_errors": [error],
                "samples": [],
            }
        ]
    )
    probe = _tree_probe()
    ctx = _tree_ctx("unknown_inventory", docs=[window], tmp_path=tmp_path, probe=probe)
    result = mod.unknown_inventory(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    baseline, mutated = _build_calls(probe)
    assert baseline[1:] == ("--source", "/sys/class/infiniband", "--run-id", ctx.run_id)
    assert mutated[1:] == (
        "--source",
        "/sys/class/infiniband",
        "--driver",
        "rdmap0s0:ambiguous",
        "--run-id",
        ctx.run_id,
    ), mutated
    opened = next(call for call in probe.calls if call[0] == "open-window")
    assert opened[1:7] == (
        "--mode",
        "tree",
        "--restore-seconds",
        str(SPECS["unknown_inventory"].max_minutes * 60),
        "--tree",
        "infiniband:/sys/class/infiniband",
    ), opened
    assert result["device"] == "rdmap0s0" and result["proof_scope"] == "live-replay"


def test_unknown_inventory_fails_when_a_count_was_reported(tmp_path: Path) -> None:
    window = _base_doc(
        batches=[
            _efa_batch(
                "2026-10-24T00:00:20+00:00",
                failure_mode="HEALTHY",
                active=EXPECTED_EFA,
                mismatch=0,
                errors=[
                    "_efa_inventory: CollectorError: EFA inventory observation unavailable: ambiguous"
                ],
            )
        ]
    )
    ctx = _tree_ctx("unknown_inventory", docs=[window], tmp_path=tmp_path)
    result = mod.unknown_inventory(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("efa_inventory_active_count" in error for error in result["errors"]), (
        result
    )


def _excess_docs(*, reboot: bool = False) -> list[dict[str, Any]]:
    detection = _base_doc(
        batches=[
            _efa_batch(
                "2026-10-24T00:00:20+00:00",
                failure_mode="EXCESS_DEVICE",
                active=4,
                mismatch=0,
            ),
            _efa_batch(
                "2026-10-24T00:00:35+00:00",
                failure_mode="EXCESS_DEVICE",
                active=4,
                mismatch=1,
            ),
        ],
        **_efa_incident(workflow_status="RUNNING", executions=[WAITING]),
    )
    recovery = _base_doc(
        batches=[
            _efa_batch(
                "2026-10-24T00:01:05+00:00",
                failure_mode="HEALTHY",
                active=3,
                mismatch=0,
            )
        ]
    )
    settled = _base_doc(
        batches=recovery["batches"],
        **_efa_incident(workflow_status="FAILED", executions=[FAILED]),
    )
    if reboot:
        settled["workflows"].append(REBOOT_WORKFLOW)
    return [_base_doc(), detection, recovery, settled]


def test_excess_device_diagnoses_only(tmp_path: Path) -> None:
    probe = _tree_probe()
    ctx = _tree_ctx(
        "excess_device", docs=_excess_docs(), tmp_path=tmp_path, probe=probe
    )
    result = mod.excess_device(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    _baseline, mutated = _build_calls(probe)
    assert mutated[1:] == (
        "--source",
        "/sys/class/infiniband",
        "--duplicate",
        f"{mod.EXCESS_DEVICE_NAME}=rdmap0s0",
        "--run-id",
        ctx.run_id,
    ), mutated
    assert result["expected_count"] == EXPECTED_EFA
    assert result["validation_executions"][0]["status"] == "FAILED"
    assert any(call[0] == "close-window" for call in probe.calls), (
        "the tree is closed as soon as the finding exists"
    )


def test_excess_device_fails_on_a_reboot_or_without_the_excess_sample(
    tmp_path: Path,
) -> None:
    ctx = _tree_ctx("excess_device", docs=_excess_docs(reboot=True), tmp_path=tmp_path)
    result = mod.excess_device(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("RESTART_NODE" in error for error in result["errors"]), result["errors"]
    docs = _excess_docs()
    for batch in docs[1]["batches"]:
        for sample in batch["samples"]:
            if sample["name"] == "efa_inventory_excess_count":
                sample["value"] = 0
    ctx = _tree_ctx("excess_device", docs=docs, tmp_path=tmp_path)
    with pytest.raises(VariantInconclusive, match="efa_inventory_excess_count"):
        mod.excess_device(ctx)


def _link_docs(*, concluded: bool) -> list[dict[str, Any]]:
    detection = _base_doc(
        batches=[
            _efa_batch(
                "2026-10-24T00:00:20+00:00",
                failure_mode="LINK_INACTIVE",
                active=2,
                mismatch=0,
            ),
            _efa_batch(
                "2026-10-24T00:00:35+00:00",
                failure_mode="LINK_INACTIVE",
                active=2,
                mismatch=1,
            ),
        ],
        **_efa_incident(workflow_status="RUNNING", executions=[WAITING]),
    )
    recovery = _base_doc(
        batches=[
            _efa_batch(
                "2026-10-24T00:01:05+00:00",
                failure_mode="HEALTHY",
                active=3,
                mismatch=0,
            )
        ]
    )
    if concluded:
        settled = _base_doc(
            batches=recovery["batches"],
            **_efa_incident(workflow_status="SUCCEEDED", executions=[SUCCEEDED]),
        )
    else:
        settled = _base_doc(
            batches=recovery["batches"],
            **_efa_incident(workflow_status="FAILED", executions=[FAILED]),
        )
        settled["workflows"].append(REBOOT_WORKFLOW)
    return [_base_doc(), detection, recovery, settled]


def test_link_inactive_passes_when_the_diagnosis_concludes_on_the_restored_link(
    tmp_path: Path,
) -> None:
    probe = _tree_probe()
    ctx = _tree_ctx(
        "link_inactive", docs=_link_docs(concluded=True), tmp_path=tmp_path, probe=probe
    )
    result = mod.link_inactive(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    _baseline, mutated = _build_calls(probe)
    assert mutated[1:] == (
        "--source",
        "/sys/class/infiniband",
        "--port-state",
        f"rdmap0s0:1:{mod.INACTIVE_STATE}",
        "--phys-state",
        f"rdmap0s0:1:{mod.INACTIVE_PHYS_STATE}",
        "--run-id",
        ctx.run_id,
    ), mutated
    assert result["device"] == "rdmap0s0" and result["port"] == "1"


def test_link_inactive_fails_on_the_deadline_reboot_gap(tmp_path: Path) -> None:
    ctx = _tree_ctx(
        "link_inactive", docs=_link_docs(concluded=False), tmp_path=tmp_path
    )
    result = mod.link_inactive(ctx)
    assert result["verdict"] == "FAIL", result
    joined = "\n".join(result["errors"])
    assert "never SUCCEEDED" in joined and "RESTART_NODE" in joined, joined
    assert "missing_inventory_action_completion" in joined, joined


def test_tree_variants_block_without_an_active_port(tmp_path: Path) -> None:
    down = _tree_summary()
    for summary in down["devices"].values():
        summary["ports"]["1"] = {"state": "1: DOWN", "phys_state": "3: Disabled"}
    probe = _tree_probe(**{"build-infiniband-fixture": [down, down]})
    ctx = _tree_ctx("link_inactive", docs=[_base_doc()], tmp_path=tmp_path, probe=probe)
    with pytest.raises(VariantBlocked, match="ACTIVE/LinkUp"):
        mod.link_inactive(ctx)


# --------------------------------------------------------------------------- #
# the product's rule and collector, pinned
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("mode", "action"),
    [
        ("LINK_INACTIVE", RecoveryAction.RUN_DIAGNOSTICS),
        ("EXCESS_DEVICE", RecoveryAction.RUN_DIAGNOSTICS),
        ("INVENTORY_UNAVAILABLE", RecoveryAction.RUN_DIAGNOSTICS),
        ("PCI_DEVICE_MISSING", RecoveryAction.REBOOT_NODE),
        ("DRIVER_UNBOUND", RecoveryAction.REMEDIATE_EFA_DRIVER),
    ],
)
def test_efa_inventory_rule_resolves_the_failure_modes(
    mode: str, action: RecoveryAction
) -> None:
    sample = HostMetricSample(
        name="efa_inventory_mismatch", value=1, labels={"failure_mode": mode}
    )
    threshold, category, severity, resolved, _reason = efa_inventory_rule(sample)
    assert (threshold, severity, resolved) == (1.0, Severity.CRITICAL, action)
    assert category.value == "RDMA"


def _sysfs_source(tmp_path: Path) -> Path:
    source = tmp_path / "sys" / "class" / "infiniband"
    drivers = tmp_path / "sys" / "bus" / "pci" / "drivers"
    (drivers / "efa").mkdir(parents=True)
    for name in DEVICES:
        device = source / name
        (device / "ports" / "1" / "hw_counters").mkdir(parents=True)
        (device / "ports" / "1" / "state").write_text("4: ACTIVE\n")
        (device / "ports" / "1" / "phys_state").write_text("5: LinkUp\n")
        (device / "ports" / "1" / "hw_counters" / "rx_drops").write_text("0\n")
        (device / "device").mkdir()
        (device / "device" / "uevent").write_text("DRIVER=efa\nPCI_ID=1D0F:EFA2\n")
        (device / "device" / "vendor").write_text("0x1d0f\n")
        (device / "device" / "device").write_text("0xefa2\n")
        (device / "device" / "driver").symlink_to(drivers / "efa")
    return source


def _probe_build(capsys: pytest.CaptureFixture[str], *arguments: str) -> dict[str, Any]:
    code = probe.main([*arguments, "--run-id", "gl013-tree-a1-deadbeef"])
    receipt = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 0, receipt
    return receipt


def _view(batch: Any) -> dict[str, Any]:
    return {
        "observed_at": batch.observed_at.isoformat(),
        "batch_id": batch.batch_id,
        "edge_filter_reasons": [],
        "collection_errors": list(batch.collection_errors),
        "samples": [
            {
                "name": s.name,
                "device": s.device,
                "value": s.value,
                "labels": dict(s.labels),
            }
            for s in batch.samples
        ],
    }


@pytest.mark.parametrize(
    ("mutation", "mode"),
    [
        (
            (
                "--port-state",
                "rdmap0s0:1:1: DOWN",
                "--phys-state",
                "rdmap0s0:1:3: Disabled",
            ),
            "LINK_INACTIVE",
        ),
        (("--duplicate", f"{mod.EXCESS_DEVICE_NAME}=rdmap0s0"), "EXCESS_DEVICE"),
    ],
)
def test_the_real_collector_resolves_the_probe_built_tree(
    host_case,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mutation: tuple[str, ...],
    mode: str,
) -> None:
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "run" / "gfrs-live")
    source = _sysfs_source(tmp_path)
    tree = _probe_build(
        capsys, "build-infiniband-fixture", "--source", str(source), *mutation
    )
    pci = _probe_build(capsys, "build-pci-fixture", "--efa-count", str(EXPECTED_EFA))
    collector = host_case.build(
        "_efa_inventory",
        infiniband_root=tree["path"],
        pci_devices_root=pci["path"],
        expected_efa_device_count=EXPECTED_EFA,
        inventory_mismatch_consecutive_samples=2,
    )
    batches = []
    for _ in range(2):
        batches.append(_view(collector.collect_once()))
        host_case.clock.sleep(15)
    observed = {"batches": batches, "incidents": [], "workflows": []}
    assert mod.efa_failure_mode_errors(observed, mode) == [], batches
    second = {item["name"]: item for item in batches[1]["samples"]}
    assert second["efa_inventory_mismatch"]["value"] == 1
    counted = (
        "efa_inventory_missing_count"
        if mode == "LINK_INACTIVE"
        else "efa_inventory_excess_count"
    )
    assert second[counted]["value"] == 1, second
    assert batches[0]["samples"] and {s["name"] for s in batches[0]["samples"]} >= {
        "efa_inventory_mismatch"
    }
    assert {item["name"]: item["value"] for item in batches[0]["samples"]}[
        "efa_inventory_mismatch"
    ] == 0, "the first sample never confirms"


def test_the_real_collector_reads_an_ambiguous_driver_as_unavailable(
    host_case,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "run" / "gfrs-live")
    source = _sysfs_source(tmp_path)
    tree = _probe_build(
        capsys,
        "build-infiniband-fixture",
        "--source",
        str(source),
        "--driver",
        "rdmap0s0:ambiguous",
    )
    pci = _probe_build(capsys, "build-pci-fixture", "--efa-count", str(EXPECTED_EFA))
    collector = host_case.build(
        "_efa_inventory",
        infiniband_root=tree["path"],
        pci_devices_root=pci["path"],
        expected_efa_device_count=EXPECTED_EFA,
    )
    batch = _view(collector.collect_once())
    assert batch["samples"] == [], batch
    errors = mod.efa_collection_errors({"batches": [batch]})
    assert errors and "ambiguous" in errors[0], errors
    assert mod.port_active(ACTIVE) and not mod.port_active(
        {"state": "1: DOWN", "phys_state": "5: LinkUp"}
    )
