"""GF-GENERAL-LIVE-013 native EFA driver negatives: registry, fixture call
shapes and the PASS / FAIL / INCONCLUSIVE grading over the product's records.

Nothing here runs a node: the on-node ``general_live_efa_probe.py`` is a
``FakeProbe`` and the product's records are a scripted activity document. The
tests prove the two variants read the product's own receipts (the failed or
succeeded ``REMEDIATE_EFA_DRIVER`` step, the post-repair failure, the escalation
reboot, the healthy recovery) and that a dirty cleanup refuses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_efa_driver as mod
from scripts.e2e.regional.general_live_fixture import (
    VariantBlocked,
    VariantInconclusive,
)
from scripts.e2e.regional.general_live_registry import coverage_rows, load_registry
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
    store_script,
)

NODE = "ip-10-0-32-162.us-west-2.compute.internal"
SPECS = {spec.variant: spec for spec in mod.VARIANTS}
BASE_BOOT = "boot-before"
NEW_BOOT = "boot-after"

HEALTH_STEPS = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "REMEDIATE_EFA_DRIVER",
    "RESTART_EFA_DEVICE_PLUGIN",
    "TRIGGER_HEALTH_SNAPSHOT",
    "VALIDATE_FABRIC",
    "RESTORE_SCHEDULING",
)


def _functions(bound: int = 16, *, module: bool = True) -> list[dict[str, Any]]:
    return [
        {
            "pci_bdf": f"0000:{0x30 + index:02x}:00.0",
            "driver": "efa" if index < bound else None,
            "bound": index < bound,
            "driver_override": "(none)",
            "rdma_device": f"rdmap{index}",
        }
        for index in range(16)
    ]


def _probe(
    *,
    bound_after: int = 15,
    module: bool = True,
    trap_fired: dict | None = None,
    release_clean: bool = True,
    release_error: Exception | None = None,
) -> FakeProbe:
    def release(_arguments: tuple[str, ...]) -> dict[str, Any]:
        if release_error is not None:
            raise release_error
        return {
            "clean": release_clean,
            "bound": release_clean,
            "pci_bdf": "0000:30:00.0",
        }

    summary = {
        "bound_count": bound_after,
        "module_loaded": module,
        "boot_id": BASE_BOOT,
        "trap_fired": trap_fired,
        "driver_override": "(none)",
        "bound": False,
    }
    return FakeProbe(
        {
            "functions": {
                "functions": _functions(),
                "module_loaded": True,
                "bound_count": 16,
                "boot_id": BASE_BOOT,
            },
            "summary": summary,
            "hold-unbind": {
                "pci_bdf": "0000:30:00.0",
                "driver_override": "gfrs-live-hold",
                "bound": False,
                "baseline_bound_count": 16,
                "boot_id": BASE_BOOT,
            },
            "trap-unbind": {
                "pci_bdf": "0000:30:00.0",
                "bound": False,
                "grace_seconds": 5,
                "baseline_bound_count": 16,
                "boot_id": BASE_BOOT,
            },
            "release": release,
        }
    )


def _batches() -> list[dict[str, Any]]:
    def sample(metric: str, value: float, failure_mode: str | None = None):
        labels = {"failure_mode": failure_mode} if failure_mode else {}
        return {"name": metric, "device": "node", "value": value, "labels": labels}

    return [
        {
            "observed_at": "2026-10-24T00:03:00+00:00",
            "batch_id": "host-1",
            "edge_filter_reasons": [],
            "collection_errors": [],
            "samples": [
                sample("efa_inventory_mismatch", 1, "DRIVER_UNBOUND"),
                sample("efa_inventory_active_count", 15),
                sample("efa_inventory_missing_count", 1),
            ],
        },
        {
            "observed_at": "2026-10-24T00:40:00+00:00",
            "batch_id": "host-9",
            "edge_filter_reasons": [],
            "collection_errors": [],
            "samples": [
                sample("efa_inventory_mismatch", 0),
                sample("efa_inventory_active_count", 16),
            ],
        },
    ]


def _doc(
    *,
    remediation_status: str = "FAILED",
    remediation_error: str = "EFA driver bind verification failed: 15/16; "
    "bind_errors=0000:30:00.0=OSError: [Errno 19] No such device",
    post_repair: tuple[str, str] | None = None,
    node_results: dict[str, Any] | None = None,
    reboot: bool = True,
    incident: bool = True,
) -> dict[str, Any]:
    executions = [
        {
            "operation": "REMEDIATE_EFA_DRIVER",
            "status": remediation_status,
            "step_index": 2,
            "error": remediation_error if remediation_status == "FAILED" else None,
        }
    ]
    if post_repair is not None:
        operation, status = post_repair
        executions.append(
            {
                "operation": operation,
                "status": status,
                "step_index": 3,
                "error": "device plugin did not restore vpc.amazonaws.com/efa"
                if status == "FAILED"
                else None,
            }
        )
    workflows = [
        {
            "request_id": "wf-health",
            "incident_id": "inc-efa",
            "status": "FAILED",
            "created_at": "2026-10-24T00:05:00+00:00",
            "official_steps": [
                {"operation": name, "parameters": {}} for name in HEALTH_STEPS
            ],
            "step_executions": executions,
        }
    ]
    if reboot:
        workflows.append(
            {
                "request_id": "wf-reboot",
                "incident_id": "inc-reboot",
                "status": "SUCCEEDED",
                "created_at": "2026-10-24T00:10:00+00:00",
                "official_steps": [
                    {"operation": "FREEZE_EVIDENCE", "parameters": {}},
                    {"operation": "RESTART_NODE", "parameters": {}},
                    {"operation": "VALIDATE_FABRIC", "parameters": {}},
                    {"operation": "RESTORE_SCHEDULING", "parameters": {}},
                ],
                "step_executions": [
                    {
                        "operation": "RESTART_NODE",
                        "status": "SUCCEEDED",
                        "step_index": 1,
                    }
                ],
            }
        )
    incidents = (
        [
            {
                "incident_id": "inc-efa",
                "event_id": "batch0-efa_inventory_mismatch-node",
                "event_type": "NODE_HEALTH",
                "effective_action": "REMEDIATE_EFA_DRIVER",
                "state": "RECOVERED",
                "created_at": "2026-10-24T00:03:30+00:00",
                "workflow_request_id": "wf-health",
                "node_ids": [NODE],
                "reasons": [
                    "EFA PCI device is present but the efa driver is not bound"
                ],
            }
        ]
        if incident
        else []
    )
    commands = [
        {
            "operation": "REMEDIATE_EFA_DRIVER",
            "result_details": {"node_results": {NODE: node_results or {}}},
            "error": None if remediation_status == "SUCCEEDED" else remediation_error,
        }
    ]
    return {
        "incidents": incidents,
        "workflows": workflows,
        "batches": _batches(),
        "markers": [],
        "commands": commands,
        "metric_latest": [
            {
                "name": "efa_inventory_expected_count",
                "device": "node",
                "value": 16,
                "observed_at": "2026-10-24T00:03:00+00:00",
                "labels": {},
            }
        ],
        "notifications": [],
        "collector_statuses": [],
    }


def _node_snapshot(*, boot: str = NEW_BOOT, isolated: bool = False) -> dict[str, Any]:
    return {
        "name": NODE,
        "uid": "uid-1",
        "boot_id": boot,
        "ready": "True",
        "unschedulable": bool(isolated),
        "taints": [],
        "ownership_annotations": {},
        "gpu_allocatable": "8",
    }


def _ctx(
    variant: str,
    *,
    doc: dict[str, Any],
    probe: FakeProbe,
    tmp_path: Path,
    node_snapshot=None,
):
    spec = SPECS[variant]
    regional = FakeRegional(
        cluster_id="gpu-eks-cluster",
        cpu_python=store_script({"activity": doc}),
        node_snapshot=node_snapshot or _node_snapshot(),
    )
    return make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"efa-driver@{NODE}": probe},
        settings=make_settings(spec, node=NODE),
        window_minutes=120,
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_both_variants_are_implemented_native_dangerous() -> None:
    reg = load_registry()
    for variant in ("bind_failure", "post_repair_failure"):
        spec = reg[f"GF-GENERAL-LIVE-013.{variant}"]
        assert spec.implemented, f"{variant} must be an implemented variant"
        assert spec.proof_mode == "native"
        assert spec.flags == "X"
        assert spec.risk == "H"
        assert spec.expected_outcome == "PASS"
        assert set(spec.approvals) == {"EFA_DRIVER_UNBIND", "NODE_REBOOT"}
        assert spec.lanes == ("eks", "hp")


def test_013_denominator_is_complete() -> None:
    reg = load_registry()
    expected = set(coverage_rows()["GF-GENERAL-LIVE-013"])
    present = {k.split(".", 1)[1] for k in reg if k.startswith("GF-GENERAL-LIVE-013.")}
    assert present == expected


# --------------------------------------------------------------------------- #
# bind_failure grading
# --------------------------------------------------------------------------- #
def test_bind_failure_passes_on_the_failed_receipt_and_reboot(tmp_path: Path) -> None:
    ctx = _ctx(
        "bind_failure",
        doc=_doc(remediation_status="FAILED"),
        probe=_probe(bound_after=15),
        tmp_path=tmp_path,
    )
    result = mod.bind_failure(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["proof_scope"] == "native"
    # The fixture drove the probe with the chosen BDF.
    efa = ctx.probes[f"efa-driver@{NODE}"]
    hold = [call for call in efa.calls if call[0] == "hold-unbind"]
    assert hold and "--pci-bdf" in hold[0] and "--restore-seconds" in hold[0]
    assert "efa-driver" in ctx.cleanup.pending()


def test_bind_failure_fails_when_the_bind_receipt_is_missing(tmp_path: Path) -> None:
    # REMEDIATE_EFA_DRIVER SUCCEEDED -> no independent bind-failure receipt.
    ctx = _ctx(
        "bind_failure",
        doc=_doc(
            remediation_status="SUCCEEDED",
            node_results={"rebound_pci_bdfs": ["0000:30:00.0"]},
        ),
        probe=_probe(),
        tmp_path=tmp_path,
    )
    result = mod.bind_failure(ctx)
    assert result["verdict"] == "FAIL", result
    assert any(
        "bind_failure_receipt" in error or "receipt_names_bdf" in error
        for error in result["errors"]
    ), result["errors"]


def test_bind_failure_fails_on_a_global_efa_unload(tmp_path: Path) -> None:
    ctx = _ctx(
        "bind_failure",
        doc=_doc(remediation_status="FAILED"),
        probe=_probe(bound_after=15, module=False),
        tmp_path=tmp_path,
    )
    result = mod.bind_failure(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("no_global_unload" in error for error in result["errors"]), result[
        "errors"
    ]


def test_bind_failure_fails_without_the_approved_reboot(tmp_path: Path) -> None:
    ctx = _ctx(
        "bind_failure",
        doc=_doc(remediation_status="FAILED", reboot=False),
        probe=_probe(),
        tmp_path=tmp_path,
    )
    result = mod.bind_failure(ctx)
    assert result["verdict"] == "FAIL", result
    assert any(
        "approved_reboot_branch" in error or "recovered" in error
        for error in result["errors"]
    ), result["errors"]


def test_bind_failure_inconclusive_when_no_finding(tmp_path: Path) -> None:
    ctx = _ctx(
        "bind_failure", doc=_doc(incident=False), probe=_probe(), tmp_path=tmp_path
    )
    with pytest.raises(VariantInconclusive):
        mod.bind_failure(ctx)


# --------------------------------------------------------------------------- #
# post_repair_failure grading
# --------------------------------------------------------------------------- #
def test_post_repair_failure_passes(tmp_path: Path) -> None:
    ctx = _ctx(
        "post_repair_failure",
        doc=_doc(
            remediation_status="SUCCEEDED",
            node_results={
                "rebound_pci_bdfs": ["0000:30:00.0"],
                "already_bound": False,
                "driver_bound_count": 16,
            },
            post_repair=("VALIDATE_FABRIC", "FAILED"),
        ),
        probe=_probe(trap_fired={"fired": True, "bound": False}),
        tmp_path=tmp_path,
    )
    result = mod.post_repair_failure(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["failed_post_repair_step"]["operation"] == "VALIDATE_FABRIC"


def test_post_repair_failure_accepts_a_plugin_restart_failure(tmp_path: Path) -> None:
    ctx = _ctx(
        "post_repair_failure",
        doc=_doc(
            remediation_status="SUCCEEDED",
            node_results={
                "rebound_pci_bdfs": ["0000:30:00.0"],
                "already_bound": False,
                "driver_bound_count": 16,
            },
            post_repair=("RESTART_EFA_DEVICE_PLUGIN", "FAILED"),
        ),
        probe=_probe(trap_fired={"fired": True}),
        tmp_path=tmp_path,
    )
    result = mod.post_repair_failure(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_post_repair_fails_when_remediation_never_succeeded(tmp_path: Path) -> None:
    ctx = _ctx(
        "post_repair_failure",
        doc=_doc(
            remediation_status="FAILED", post_repair=("VALIDATE_FABRIC", "FAILED")
        ),
        probe=_probe(trap_fired={"fired": True}),
        tmp_path=tmp_path,
    )
    result = mod.post_repair_failure(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("action_succeeded" in error for error in result["errors"]), result[
        "errors"
    ]


def test_post_repair_fails_without_a_post_repair_failure(tmp_path: Path) -> None:
    ctx = _ctx(
        "post_repair_failure",
        doc=_doc(
            remediation_status="SUCCEEDED",
            node_results={
                "rebound_pci_bdfs": ["0000:30:00.0"],
                "already_bound": False,
                "driver_bound_count": 16,
            },
            post_repair=("VALIDATE_FABRIC", "SUCCEEDED"),
        ),
        probe=_probe(trap_fired={"fired": True}),
        tmp_path=tmp_path,
    )
    result = mod.post_repair_failure(ctx)
    assert result["verdict"] == "FAIL", result
    assert any(
        "independent_post_repair_failure" in error for error in result["errors"]
    ), result["errors"]


def test_post_repair_fails_when_the_trap_never_fired(tmp_path: Path) -> None:
    ctx = _ctx(
        "post_repair_failure",
        doc=_doc(
            remediation_status="SUCCEEDED",
            node_results={
                "rebound_pci_bdfs": ["0000:30:00.0"],
                "already_bound": False,
                "driver_bound_count": 16,
            },
            post_repair=("VALIDATE_FABRIC", "FAILED"),
        ),
        probe=_probe(trap_fired={"fired": False, "reason": "bind not observed"}),
        tmp_path=tmp_path,
    )
    result = mod.post_repair_failure(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("trap_fired" in error for error in result["errors"]), result["errors"]


# --------------------------------------------------------------------------- #
# Fixture: target selection and cleanup refusal
# --------------------------------------------------------------------------- #
def test_target_bdf_blocked_with_fewer_than_two_functions(tmp_path: Path) -> None:
    probe = FakeProbe(
        {
            "functions": {
                "functions": [
                    {"pci_bdf": "0000:30:00.0", "driver": "efa", "bound": True}
                ],
                "bound_count": 1,
            }
        }
    )
    ctx = _ctx("bind_failure", doc=_doc(), probe=probe, tmp_path=tmp_path)
    fixture = mod.EfaDriverFixture(ctx)
    with pytest.raises(VariantBlocked):
        fixture.target_bdf()


def test_cleanup_refusal_is_recorded_on_the_lifo_stack(tmp_path: Path) -> None:
    from scripts.e2e.regional.host_probe_fixture import HostProbeError

    probe = _probe(release_error=HostProbeError("EFA driver fixture did not restore"))
    ctx = _ctx("bind_failure", doc=_doc(), probe=probe, tmp_path=tmp_path)
    fixture = mod.EfaDriverFixture(ctx)
    fixture.target_bdf()  # select so release has a BDF to act on
    results = ctx.cleanup.run()
    assert any(not item["ok"] and item["name"] == "efa-driver" for item in results), (
        results
    )
    assert ctx.cleanup.failed, "a refused release must be recorded as a failed cleanup"
