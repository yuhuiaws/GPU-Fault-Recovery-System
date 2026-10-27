"""GF-GENERAL-LIVE-013 native EFA driver negatives: registry, fixture call
shapes and the PASS / FAIL / INCONCLUSIVE grading over the product's records.

Nothing here runs a node: the on-node ``general_live_efa_probe.py`` is a
``FakeProbe`` and the product's records are a scripted activity document in the
shape the live site wrote on 2026-10-27 (HyperPod, R13g): the remediation
record ``workflow-<id>`` FAILED at REMEDIATE_EFA_DRIVER (or at a post-repair
step), its ``workflow-reboot-after-workflow-<id>`` successor planned
FREEZE_EVIDENCE, MARK_UNSCHEDULABLE, QUARANTINE, RESTART_NODE, the three
validations and RESTORE_SCHEDULING, rebooted the node and RECOVERED, and the
predecessor was closed RECOVERED by that restore. The tests prove the two
variants read the product's own receipts, accept the successor's QUARANTINE as
the chain's own cordon, refuse a blind loop / a REPLACE_NODE / a ladder that
never reboots, read the trap's record in the probe's real shape, and that the
cleanup is judged by the reboot (plus the product's post-boot inventory) when
the probe died with it -- and still fails without a reboot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_efa_driver as mod
from scripts.e2e.regional import general_live_efa_escalation as esc
from scripts.e2e.regional.general_live_fixture import (
    VariantBlocked,
    VariantInconclusive,
)
from scripts.e2e.regional.general_live_registry import coverage_rows, load_registry
from scripts.e2e.regional.host_probe_fixture import (
    HostProbeError,
    HostProbeTransportError,
)
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    make_context,
    make_settings,
    store_poll_clock,
    store_script,
)

NODE = "hyperpod-test-node-a"
BDF = "0000:30:00.0"
SPECS = {spec.variant: spec for spec in mod.VARIANTS}
BASE_BOOT = "boot-before"
NEW_BOOT = "boot-after"
#: The new boot's Ready transition: before the fake clock's start (so the
#: node reads settled Ready at once) and before the healthy post-boot batch.
READY_SINCE = "2026-10-23T23:50:00+00:00"
RECORD = "workflow-aaf326cf4be4f315223ef25b"
SUCCESSOR = "workflow-reboot-after-" + RECORD
SUCCESSOR_INCIDENT = "inc-reboot-after-" + RECORD
SUPPORT = "workflow-support-after-" + RECORD
SUPPORT_INCIDENT = "inc-support-after-" + RECORD
CAP_ERROR = (
    "step 5/VALIDATE_FABRIC stayed non-terminal for 602s, past the 600s per-step cap"
)
CAP_DETAILS = {
    "step_waiting_timeout_seconds": 600,
    "node_pending": {
        NODE: {
            "missing_post_action_inventory_metric": ["efa_inventory_active_count"],
            "observed_after": ["2026-10-24T00:10:00+00:00"],
        }
    },
}
BIND_ERROR = (
    f"EFA driver bind verification failed: 15/16; bind_errors={BDF}=OSError: "
    "[Errno 19] No such device"
)
TRAP_FIRED_TEXT = json.dumps(
    {"fired": True, "bound": False, "at": "2026-10-27T07:10:04.388473+00:00"}
)
TRAP_GAVE_UP_TEXT = json.dumps(
    {"fired": False, "reason": "bind not observed", "at": "2026-10-27T07:20:00+00:00"}
)
HEALTH_STEPS = esc.REMEDIATION_CHAIN
REBOOT_STEPS = esc.REBOOT_SUCCESSOR_CHAIN


def _functions(bound: int = 16) -> list[dict[str, Any]]:
    return [
        {
            "pci_bdf": f"0000:{0x30 + index:02x}:00.0",
            "driver": "efa" if index < bound else None,
            "bound": index < bound,
            # The probe's parsed value of the kernel's "(null)" read-back.
            "driver_override": None,
            "rdma_device": f"rdmap{index}",
        }
        for index in range(16)
    ]


def _summary(
    *, bound_after: int = 15, module: bool = True, trap_text: str | None = None
) -> dict[str, Any]:
    """The probe's ``summary`` in its live shape (``trap_fired_at`` is the JSON
    text of the trap's record, or ``None`` until the unit wrote it)."""

    return {
        "bound_count": bound_after,
        "module_loaded": module,
        "boot_id": BASE_BOOT,
        "trap_fired_at": trap_text,
        "driver_override": None,
        "driver_override_sysfs": "(null)",
        "bound": False,
        "pci_bdf": BDF,
        "run_units": ["gfrs-deadbeef-efa-deadman.timer"],
        "observed_at": "2026-10-24T00:04:00+00:00",
    }


def _probe(
    *,
    bound_after: int = 15,
    module: bool = True,
    trap_text: str | None = None,
    summary: Any = None,
    release_clean: bool = True,
    release_error: Exception | None = None,
    release: Any = None,
) -> FakeProbe:
    def default_release(_arguments: tuple[str, ...]) -> dict[str, Any]:
        if release_error is not None:
            raise release_error
        return {"clean": release_clean, "bound": release_clean, "pci_bdf": BDF}

    return FakeProbe(
        {
            "functions": {
                "functions": _functions(),
                "module_loaded": True,
                "bound_count": 16,
                "boot_id": BASE_BOOT,
            },
            "summary": summary
            if summary is not None
            else _summary(bound_after=bound_after, module=module, trap_text=trap_text),
            "hold-unbind": {
                "pci_bdf": BDF,
                "driver_override": "gfrs-live-hold",
                "bound": False,
                "baseline_bound_count": 16,
                "boot_id": BASE_BOOT,
            },
            "trap-unbind": {
                "pci_bdf": BDF,
                "bound": False,
                "grace_seconds": 5,
                "baseline_bound_count": 16,
                "boot_id": BASE_BOOT,
            },
            "release": release if release is not None else default_release,
        }
    )


def _failing_after(calls: int, reading: dict[str, Any]) -> Any:
    """A ``summary`` that answers ``calls`` times and then has lost its Pod."""

    seen = {"n": 0}

    def respond(_arguments: tuple[str, ...]) -> dict[str, Any]:
        seen["n"] += 1
        if seen["n"] > calls:
            raise HostProbeTransportError("owned host probe Pod has terminated")
        return dict(reading)

    return respond


def _sample(metric: str, value: float, failure_mode: str | None = None) -> dict:
    labels = {"failure_mode": failure_mode} if failure_mode else {}
    return {"name": metric, "device": "node", "value": value, "labels": labels}


def _batches(*, recovered_active: int = 16) -> list[dict[str, Any]]:
    return [
        {
            "observed_at": "2026-10-24T00:03:00+00:00",
            "batch_id": "host-1",
            "edge_filter_reasons": [],
            "collection_errors": [],
            "samples": [
                _sample("efa_inventory_mismatch", 1, "DRIVER_UNBOUND"),
                _sample("efa_inventory_active_count", 15),
                _sample("efa_inventory_missing_count", 1),
            ],
        },
        {
            "observed_at": "2026-10-24T00:40:00+00:00",
            "batch_id": "host-9",
            "edge_filter_reasons": [],
            "collection_errors": [],
            "samples": [
                _sample(
                    "efa_inventory_mismatch",
                    0 if recovered_active == 16 else 1,
                    None if recovered_active == 16 else "DRIVER_UNBOUND",
                ),
                _sample("efa_inventory_active_count", recovered_active),
            ],
        },
    ]


def _steps(names: tuple[str, ...]) -> list[dict[str, Any]]:
    parameters = {
        "REMEDIATE_EFA_DRIVER": {
            "expected_count": 16,
            "failure_escalation_action": "REBOOT_NODE",
        },
        "RESTART_EFA_DEVICE_PLUGIN": {
            "expected_count": 16,
            "failure_escalation_action": "REBOOT_NODE",
        },
        "VALIDATE_FABRIC": {"failure_escalation_action": "REBOOT_NODE"},
    }
    return [
        {
            "operation": name,
            "parameters": parameters.get(name, {}),
            "node_ids": [NODE],
            "workload_ids": [],
        }
        for name in names
    ]


def _execution(operation: str, status: str, index: int, error: str | None = None):
    return {
        "operation": operation,
        "status": status,
        "step_index": index,
        "error": error,
        "updated_at": f"2026-10-24T00:{5 + index:02d}:00+00:00",
    }


def _successor(
    request_id: str,
    source: str,
    *,
    status: str = "SUCCEEDED",
    steps: tuple[str, ...] = REBOOT_STEPS,
    restart_status: str = "SUCCEEDED",
    rung: str = "RESTART_NODE",
) -> dict[str, Any]:
    executions = [
        _execution("FREEZE_EVIDENCE", "SUCCEEDED", 0),
        _execution("MARK_UNSCHEDULABLE", "SUCCEEDED", 1),
        _execution("QUARANTINE", "SUCCEEDED", 2),
        _execution(rung, restart_status, 3),
    ]
    if status == "SUCCEEDED":
        executions.extend(
            [
                _execution("VALIDATE_GPU", "SUCCEEDED", 4),
                _execution("VALIDATE_HOST", "SUCCEEDED", 5),
                _execution("VALIDATE_FABRIC", "SUCCEEDED", 6),
                _execution("RESTORE_SCHEDULING", "SUCCEEDED", 7),
            ]
        )
    return {
        "request_id": request_id,
        "incident_id": "inc-" + request_id.removeprefix("workflow-"),
        "status": status,
        "official_action": "REPLACE_NODE" if rung == "REPLACE_NODE" else "REBOOT_NODE",
        "created_at": "2026-10-24T00:10:00+00:00",
        "predecessor_workflow_id": source,
        "official_steps": _steps(steps),
        "step_executions": executions,
    }


def _doc(
    *,
    remediation_status: str = "FAILED",
    remediation_error: str = BIND_ERROR,
    post_repair: tuple[str, str] | None = None,
    node_results: dict[str, Any] | None = None,
    reboot: bool = True,
    incident: bool = True,
    record_status: str = "FAILED",
    record_steps: tuple[str, ...] = HEALTH_STEPS,
    record_release: bool = False,
    successor_status: str = "SUCCEEDED",
    successor_steps: tuple[str, ...] = REBOOT_STEPS,
    successor_incident_state: str = "RECOVERED",
    source_state: str = "RECOVERED",
    extra_workflows: tuple[dict[str, Any], ...] = (),
    recovered_active: int = 16,
    post_repair_details: dict[str, Any] | None = None,
    post_repair_error: str | None = None,
    support: bool = False,
    live_commands: bool = False,
) -> dict[str, Any]:
    executions = [
        _execution("FREEZE_EVIDENCE", "SUCCEEDED", 0),
        _execution("MARK_UNSCHEDULABLE", "SUCCEEDED", 1),
        _execution(
            "REMEDIATE_EFA_DRIVER",
            remediation_status,
            2,
            remediation_error if remediation_status == "FAILED" else None,
        ),
    ]
    if post_repair is not None:
        operation, status = post_repair
        index = 3 if operation == "RESTART_EFA_DEVICE_PLUGIN" else 5
        failed_text = post_repair_error or (
            "device plugin did not restore vpc.amazonaws.com/efa"
        )
        execution = _execution(
            operation, status, index, failed_text if status == "FAILED" else None
        )
        if post_repair_details is not None:
            execution["details"] = dict(post_repair_details)
        executions.append(execution)
    if record_release:
        executions.append(_execution("RESTORE_SCHEDULING", "SUCCEEDED", 6))
    workflows = [
        {
            "request_id": RECORD,
            "incident_id": "inc-efa",
            "status": record_status,
            "official_action": "REMEDIATE_EFA_DRIVER",
            "created_at": "2026-10-24T00:05:00+00:00",
            "official_steps": _steps(record_steps),
            "step_executions": executions,
        }
    ]
    incidents = []
    if incident:
        incidents.append(
            {
                "incident_id": "inc-efa",
                "event_id": f"host-{NODE}-1793084182398317-efa_inventory_mismatch-node",
                "event_type": "NODE_HEALTH",
                "effective_action": "REMEDIATE_EFA_DRIVER",
                "state": source_state,
                "created_at": "2026-10-24T00:03:30+00:00",
                "workflow_request_id": RECORD,
                "node_ids": [NODE],
                "reasons": [
                    "EFA PCI device is present but the efa driver is not bound"
                ],
            }
        )
    if reboot:
        workflows.append(
            _successor(
                SUCCESSOR, RECORD, status=successor_status, steps=successor_steps
            )
        )
        incidents.append(
            {
                "incident_id": SUCCESSOR_INCIDENT,
                "event_id": "reboot-after-" + RECORD,
                "event_type": "NODE_HEALTH",
                "effective_action": "REBOOT_NODE",
                "state": successor_incident_state,
                "created_at": "2026-10-24T00:10:00+00:00",
                "workflow_request_id": SUCCESSOR,
                "node_ids": [NODE],
                "reasons": [
                    "efa_recovery remediation failed; failed operation or validation: "
                    f"REMEDIATE_EFA_DRIVER; nodes: {NODE}: validation failed"
                ],
            }
        )
    for item in extra_workflows:
        workflows.append(item)
        incidents.append(
            {
                "incident_id": item["incident_id"],
                "event_id": item["request_id"].removeprefix("workflow-"),
                "event_type": "NODE_HEALTH",
                "effective_action": item.get("official_action"),
                "state": "RECOVERED",
                "created_at": "2026-10-24T00:30:00+00:00",
                "workflow_request_id": item["request_id"],
                "node_ids": [NODE],
                "reasons": [],
            }
        )
    if support:
        workflows.append(
            {
                "request_id": SUPPORT,
                "incident_id": SUPPORT_INCIDENT,
                "status": "SUCCEEDED",
                "official_action": "ESCALATE_OPERATOR",
                "created_at": "2026-10-24T00:20:00+00:00",
                "official_steps": _steps(esc.SUPPORT_HANDOFF_CHAIN),
                "step_executions": [
                    _execution("FREEZE_EVIDENCE", "SUCCEEDED", 0),
                    _execution("ESCALATE_SUPPORT", "SUCCEEDED", 1),
                ],
            }
        )
        incidents.append(
            {
                "incident_id": SUPPORT_INCIDENT,
                "event_id": "support-after-" + RECORD,
                "event_type": "NODE_HEALTH",
                "effective_action": "ESCALATE_OPERATOR",
                "state": "ESCALATED",
                "created_at": "2026-10-24T00:20:00+00:00",
                "workflow_request_id": SUPPORT,
                "node_ids": [NODE],
                "reasons": [
                    "requested escalation REBOOT_NODE withheld: VALIDATE_FABRIC on "
                    f"{NODE} observed no node failure before the 600s per-step cap "
                    "failed it -- missing evidence, not a verdict; an operator "
                    "reviews the node"
                ],
            }
        )
    receipt = {"node_results": {NODE: node_results or {}}}
    commands = [
        # The live RemoteActionCommand names its operation under ``step``.
        {
            "command_id": "remote-dae3d3fc",
            "workflow_request_id": RECORD,
            "step_index": 2,
            "step": {"operation": "REMEDIATE_EFA_DRIVER", "node_ids": [NODE]},
            "status": "SUCCEEDED" if remediation_status == "SUCCEEDED" else "FAILED",
            "result_details": receipt,
            "error": None if remediation_status == "SUCCEEDED" else remediation_error,
        }
        if live_commands
        else {
            "operation": "REMEDIATE_EFA_DRIVER",
            "result_details": receipt,
            "error": None if remediation_status == "SUCCEEDED" else remediation_error,
        }
    ]
    return {
        "incidents": incidents,
        "workflows": workflows,
        "batches": _batches(recovered_active=recovered_active),
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


REPAIRED = {"rebound_pci_bdfs": [BDF], "already_bound": False, "driver_bound_count": 16}


def _node_snapshot(
    *, boot: str = NEW_BOOT, isolated: bool = False, ready_since: str = READY_SINCE
) -> dict[str, Any]:
    return {
        "name": NODE,
        "uid": "uid-1",
        "boot_id": boot,
        "ready": "True",
        "ready_since": ready_since,
        "unschedulable": bool(isolated),
        "taints": (
            [{"key": "gpu-fault.io/quarantined", "value": "inc-efa"}]
            if isolated
            else []
        ),
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


def _errors_of(result: dict[str, Any], stage: str) -> list[str]:
    return list(result["stages"][stage])


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
        assert "reboot-after" in spec.expected, (
            "the registered expectation names the product's reboot successor"
        )
        assert "boot id" in spec.cleanup, (
            "the registered cleanup says the release is judged by the reboot"
        )


def test_013_denominator_is_complete() -> None:
    reg = load_registry()
    expected = set(coverage_rows()["GF-GENERAL-LIVE-013"])
    present = {k.split(".", 1)[1] for k in reg if k.startswith("GF-GENERAL-LIVE-013.")}
    assert present == expected


# --------------------------------------------------------------------------- #
# bind_failure grading
# --------------------------------------------------------------------------- #
def test_bind_failure_passes_on_the_live_escalation_shape(tmp_path: Path) -> None:
    """The 2026-10-27 record: FAILED bind, the reboot successor with its own
    QUARANTINE, one RESTART_NODE, both incidents RECOVERED, node released."""

    ctx = _ctx("bind_failure", doc=_doc(), probe=_probe(), tmp_path=tmp_path)
    result = mod.bind_failure(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["proof_scope"] == "native"
    assert result["rebooted"] is True
    assert result["boot_id_after"] == NEW_BOOT
    assert [item["request_id"] for item in result["escalation"]["successors"]] == [
        SUCCESSOR
    ]
    assert "QUARANTINE" in result["escalation"]["successors"][0]["operations"]
    efa = ctx.probes[f"efa-driver@{NODE}"]
    hold = [call for call in efa.calls if call[0] == "hold-unbind"]
    assert hold and "--pci-bdf" in hold[0] and "--restore-seconds" in hold[0]
    assert "efa-driver" in ctx.cleanup.pending()
    for name in (
        "bind-failure-rung-chain",
        "efa-driver-escalation",
        "efa-driver-settled",
    ):
        assert (ctx.case_dir / f"{name}.json").is_file(), name
    settled = json.loads((ctx.case_dir / "efa-driver-settled.json").read_text())
    assert settled["rebooted"] is True
    assert settled["successors"][0]["incident_state"] == "RECOVERED"


def test_bind_failure_fails_when_the_remediation_record_itself_quarantines(
    tmp_path: Path,
) -> None:
    steps = HEALTH_STEPS[:2] + ("QUARANTINE",) + HEALTH_STEPS[2:]
    ctx = _ctx(
        "bind_failure", doc=_doc(record_steps=steps), probe=_probe(), tmp_path=tmp_path
    )
    result = mod.bind_failure(ctx)
    assert result["verdict"] == "FAIL", result
    errors = _errors_of(result, "not_isolated_beyond_chain")
    assert errors and "itself plans QUARANTINE" in errors[0], errors


def test_bind_failure_fails_on_a_stop_workloads_that_names_no_workload(
    tmp_path: Path,
) -> None:
    steps = REBOOT_STEPS[:3] + ("STOP_WORKLOADS",) + REBOOT_STEPS[3:]
    ctx = _ctx(
        "bind_failure",
        doc=_doc(successor_steps=steps),
        probe=_probe(),
        tmp_path=tmp_path,
    )
    result = mod.bind_failure(ctx)
    assert result["verdict"] == "FAIL", result
    errors = _errors_of(result, "not_isolated_beyond_chain")
    assert errors and "names no workload" in errors[0], errors


def test_bind_failure_fails_on_isolation_by_a_workflow_outside_the_chain(
    tmp_path: Path,
) -> None:
    stranger = {
        "request_id": "workflow-other",
        "incident_id": "inc-other",
        "status": "SUCCEEDED",
        "official_action": "QUARANTINE",
        "created_at": "2026-10-24T00:30:00+00:00",
        "official_steps": _steps(
            ("FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE")
        ),
        "step_executions": [],
    }
    ctx = _ctx(
        "bind_failure",
        doc=_doc(extra_workflows=(stranger,)),
        probe=_probe(),
        tmp_path=tmp_path,
    )
    result = mod.bind_failure(ctx)
    assert result["verdict"] == "FAIL", result
    errors = _errors_of(result, "not_isolated_beyond_chain")
    assert errors == ["workflow workflow-other plans QUARANTINE beyond the chain"]


def test_bind_failure_fails_on_a_blind_reboot_loop(tmp_path: Path) -> None:
    second = _successor("workflow-reboot-after-" + SUCCESSOR, SUCCESSOR)
    ctx = _ctx(
        "bind_failure",
        doc=_doc(extra_workflows=(second,)),
        probe=_probe(),
        tmp_path=tmp_path,
    )
    result = mod.bind_failure(ctx)
    assert result["verdict"] == "FAIL", result
    errors = " | ".join(_errors_of(result, "escalation_shape"))
    assert "did not end the chain" in errors, errors
    assert "RESTART_NODE executed 2 time(s)" in errors, errors
    # The second-generation successor is not mistaken for a sibling rung.
    assert "2 reboot-after successors" not in errors, errors


def test_bind_failure_fails_when_the_ladder_went_on_to_replace(tmp_path: Path) -> None:
    replace = _successor(
        "workflow-replace-after-" + SUCCESSOR,
        SUCCESSOR,
        steps=REBOOT_STEPS[:3] + ("REPLACE_NODE",) + REBOOT_STEPS[4:],
        rung="REPLACE_NODE",
    )
    ctx = _ctx(
        "bind_failure",
        doc=_doc(extra_workflows=(replace,)),
        probe=_probe(),
        tmp_path=tmp_path,
    )
    result = mod.bind_failure(ctx)
    assert result["verdict"] == "FAIL", result
    errors = " | ".join(_errors_of(result, "escalation_shape"))
    assert "REPLACE_NODE planned by" in errors, errors


def test_bind_failure_fails_when_the_remediation_was_re_run(tmp_path: Path) -> None:
    rerun = {
        "request_id": "workflow-second",
        "incident_id": "inc-second",
        "status": "FAILED",
        "official_action": "REMEDIATE_EFA_DRIVER",
        "created_at": "2026-10-24T00:30:00+00:00",
        "official_steps": _steps(HEALTH_STEPS),
        "step_executions": [
            _execution("REMEDIATE_EFA_DRIVER", "FAILED", 2, BIND_ERROR)
        ],
    }
    ctx = _ctx(
        "bind_failure",
        doc=_doc(extra_workflows=(rerun,)),
        probe=_probe(),
        tmp_path=tmp_path,
    )
    result = mod.bind_failure(ctx)
    assert result["verdict"] == "FAIL", result
    errors = " | ".join(_errors_of(result, "escalation_shape"))
    assert "re-run instead of escalated" in errors, errors


def test_bind_failure_fails_when_the_bind_receipt_is_missing(tmp_path: Path) -> None:
    # REMEDIATE_EFA_DRIVER SUCCEEDED -> no independent bind-failure receipt.
    ctx = _ctx(
        "bind_failure",
        doc=_doc(remediation_status="SUCCEEDED", node_results=REPAIRED),
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
        doc=_doc(),
        probe=_probe(bound_after=15, module=False),
        tmp_path=tmp_path,
    )
    result = mod.bind_failure(ctx)
    assert result["verdict"] == "FAIL", result
    assert any("no_global_unload" in error for error in result["errors"]), result[
        "errors"
    ]


def test_bind_failure_fails_fast_when_the_ladder_never_reboots(tmp_path: Path) -> None:
    """The record FAILED, the incident parked QUARANTINED, no successor: the
    settled wait returns on the stopped ladder (two polls) and the stages name
    it instead of waiting the whole reboot window."""

    ctx = _ctx(
        "bind_failure",
        doc=_doc(reboot=False, source_state="QUARANTINED"),
        probe=_probe(),
        tmp_path=tmp_path,
        node_snapshot=_node_snapshot(boot=BASE_BOOT, isolated=True),
    )
    store_poll_clock(ctx)
    result = mod.bind_failure(ctx)
    assert result["verdict"] == "FAIL", result
    assert _errors_of(result, "approved_reboot_branch"), result["errors"]
    assert any(
        "0 reboot-after successors" in e for e in _errors_of(result, "escalation_shape")
    ), result["stages"]["escalation_shape"]
    recovered = " | ".join(_errors_of(result, "recovered"))
    assert "ended without rebooting" in recovered, recovered
    assert "still carries the product's isolation" in recovered, recovered
    timeline = json.loads(
        (ctx.case_dir / "efa-driver-settled-timeline.json").read_text()
    )
    assert len(timeline["entries"]) <= 3, timeline


def test_bind_failure_fails_when_the_successor_left_the_node_isolated(
    tmp_path: Path,
) -> None:
    ctx = _ctx(
        "bind_failure",
        doc=_doc(
            successor_status="FAILED",
            successor_incident_state="QUARANTINED",
            source_state="QUARANTINED",
        ),
        probe=_probe(),
        tmp_path=tmp_path,
        node_snapshot=_node_snapshot(isolated=True),
    )
    store_poll_clock(ctx)
    result = mod.bind_failure(ctx)
    assert result["verdict"] == "FAIL", result
    recovered = " | ".join(_errors_of(result, "recovered"))
    assert "did not return healthy" in recovered, recovered
    assert f"the reboot successor {SUCCESSOR} ended FAILED" in recovered, recovered
    assert "the EFA incident inc-efa is QUARANTINED" in recovered, recovered


def test_bind_failure_fails_when_the_record_released_the_node_on_the_failed_read(
    tmp_path: Path,
) -> None:
    ctx = _ctx(
        "bind_failure", doc=_doc(record_release=True), probe=_probe(), tmp_path=tmp_path
    )
    result = mod.bind_failure(ctx)
    assert result["verdict"] == "FAIL", result
    errors = " | ".join(_errors_of(result, "escalation_shape"))
    assert "restored scheduling although its step failed" in errors, errors


def test_bind_failure_survives_the_probe_dying_with_the_reboot(tmp_path: Path) -> None:
    """The survey and the first oracle reading answer; the reading after the
    incident finds the Pod terminated by the reboot: not a fixture failure."""

    probe = _probe(summary=_failing_after(2, _summary()))
    ctx = _ctx("bind_failure", doc=_doc(), probe=probe, tmp_path=tmp_path)
    result = mod.bind_failure(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert "HostProbeTransportError" in str(result["probe_lost"])
    lost = json.loads((ctx.case_dir / "efa-driver-probe-lost.json").read_text())
    assert lost["boot_id_armed"] == BASE_BOOT and lost["boot_id_now"] == NEW_BOOT
    oracle = [
        json.loads(line)
        for line in (ctx.case_dir / "oracle.jsonl").read_text().splitlines()
    ]
    assert oracle[0]["readings"][-1]["probe_lost"]


def test_bind_failure_inconclusive_when_no_finding(tmp_path: Path) -> None:
    ctx = _ctx(
        "bind_failure", doc=_doc(incident=False), probe=_probe(), tmp_path=tmp_path
    )
    with pytest.raises(VariantInconclusive):
        mod.bind_failure(ctx)


# --------------------------------------------------------------------------- #
# post_repair_failure grading
# --------------------------------------------------------------------------- #
def test_post_repair_failure_passes_on_the_live_shape(tmp_path: Path) -> None:
    """SUCCEEDED bind receipt, the trap's record in the probe's JSON-text shape,
    VALIDATE_FABRIC FAILED on fresh evidence, the reboot successor, RECOVERED."""

    ctx = _ctx(
        "post_repair_failure",
        doc=_doc(
            remediation_status="SUCCEEDED",
            node_results=REPAIRED,
            post_repair=("VALIDATE_FABRIC", "FAILED"),
        ),
        probe=_probe(trap_text=TRAP_FIRED_TEXT),
        tmp_path=tmp_path,
    )
    result = mod.post_repair_failure(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["failed_post_repair_step"]["operation"] == "VALIDATE_FABRIC"
    assert result["trap"]["fired"] is True
    assert result["trap_witness"] == esc.JUDGED_BY_PROBE
    assert result["rebooted"] is True
    assert (ctx.case_dir / "post-repair-trap-timeline.json").is_file(), (
        "the trap read keeps its poll timeline"
    )


def test_post_repair_failure_accepts_a_plugin_restart_failure(tmp_path: Path) -> None:
    ctx = _ctx(
        "post_repair_failure",
        doc=_doc(
            remediation_status="SUCCEEDED",
            node_results=REPAIRED,
            post_repair=("RESTART_EFA_DEVICE_PLUGIN", "FAILED"),
        ),
        probe=_probe(trap_text=TRAP_FIRED_TEXT),
        tmp_path=tmp_path,
    )
    result = mod.post_repair_failure(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_post_repair_trap_read_waits_for_the_unit_to_write_its_record(
    tmp_path: Path,
) -> None:
    """``trap_fired_at`` is ``None`` until the unit fires; the read polls."""

    probe = _probe(
        summary=[_summary(), _summary(), _summary(trap_text=TRAP_FIRED_TEXT)]
    )
    ctx = _ctx(
        "post_repair_failure",
        doc=_doc(
            remediation_status="SUCCEEDED",
            node_results=REPAIRED,
            post_repair=("VALIDATE_FABRIC", "FAILED"),
        ),
        probe=probe,
        tmp_path=tmp_path,
    )
    result = mod.post_repair_failure(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["trap"]["fired"] is True
    summaries = [call for call in probe.calls if call[0] == "summary"]
    assert len(summaries) >= 3, summaries


def test_post_repair_failure_witnessed_by_the_record_when_the_reboot_took_the_probe(
    tmp_path: Path,
) -> None:
    """The probe died before the trap's record was read; the product's
    SUCCEEDED-then-FAILED record on the same function is the witness."""

    probe = _probe(summary=_failing_after(2, _summary()))
    ctx = _ctx(
        "post_repair_failure",
        doc=_doc(
            remediation_status="SUCCEEDED",
            node_results=REPAIRED,
            post_repair=("VALIDATE_FABRIC", "FAILED"),
        ),
        probe=probe,
        tmp_path=tmp_path,
    )
    result = mod.post_repair_failure(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["trap_witness"] == esc.JUDGED_BY_PRODUCT_RECORD
    assert result["trap"] == {}
    assert "HostProbeTransportError" in str(result["probe_lost"])


def test_post_repair_fails_when_the_probe_is_lost_and_nothing_failed_after(
    tmp_path: Path,
) -> None:
    probe = _probe(summary=_failing_after(2, _summary()))
    ctx = _ctx(
        "post_repair_failure",
        doc=_doc(
            remediation_status="SUCCEEDED",
            node_results=REPAIRED,
            post_repair=("VALIDATE_FABRIC", "SUCCEEDED"),
        ),
        probe=probe,
        tmp_path=tmp_path,
    )
    result = mod.post_repair_failure(ctx)
    assert result["verdict"] == "FAIL", result
    errors = " | ".join(_errors_of(result, "trap_fired"))
    assert "probe was lost before the trap's record was read" in errors, errors
    assert _errors_of(result, "independent_post_repair_failure"), result["stages"]


def test_post_repair_fails_when_remediation_never_succeeded(tmp_path: Path) -> None:
    ctx = _ctx(
        "post_repair_failure",
        doc=_doc(
            remediation_status="FAILED", post_repair=("VALIDATE_FABRIC", "FAILED")
        ),
        probe=_probe(trap_text=TRAP_FIRED_TEXT),
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
            node_results=REPAIRED,
            post_repair=("VALIDATE_FABRIC", "SUCCEEDED"),
        ),
        probe=_probe(trap_text=TRAP_FIRED_TEXT),
        tmp_path=tmp_path,
    )
    result = mod.post_repair_failure(ctx)
    assert result["verdict"] == "FAIL", result
    assert any(
        "independent_post_repair_failure" in error for error in result["errors"]
    ), result["errors"]


def test_post_repair_fails_when_the_trap_gave_up(tmp_path: Path) -> None:
    ctx = _ctx(
        "post_repair_failure",
        doc=_doc(
            remediation_status="SUCCEEDED",
            node_results=REPAIRED,
            post_repair=("VALIDATE_FABRIC", "FAILED"),
        ),
        probe=_probe(trap_text=TRAP_GAVE_UP_TEXT),
        tmp_path=tmp_path,
    )
    result = mod.post_repair_failure(ctx)
    assert result["verdict"] == "FAIL", result
    errors = " | ".join(_errors_of(result, "trap_fired"))
    assert "did not fire" in errors and "bind not observed" in errors, errors


def test_post_repair_accepts_the_successors_quarantine(tmp_path: Path) -> None:
    ctx = _ctx(
        "post_repair_failure",
        doc=_doc(
            remediation_status="SUCCEEDED",
            node_results=REPAIRED,
            post_repair=("VALIDATE_FABRIC", "FAILED"),
        ),
        probe=_probe(trap_text=TRAP_FIRED_TEXT),
        tmp_path=tmp_path,
    )
    result = mod.post_repair_failure(ctx)
    assert result["stages"]["not_isolated_beyond_chain"] == []
    assert result["stages"]["escalation_shape"] == []


# --------------------------------------------------------------------------- #
# The chain checks on their own
# --------------------------------------------------------------------------- #
def test_trap_record_reads_the_probe_s_shapes() -> None:
    assert esc.trap_record({"trap_fired_at": TRAP_FIRED_TEXT})["fired"] is True
    assert esc.trap_record({"trap_fired_at": None}) == {}
    assert esc.trap_record({"trap_fired": {"fired": False, "reason": "x"}}) == {
        "fired": False,
        "reason": "x",
    }
    assert esc.trap_record({"trap_fired_at": "not json"}) == {"raw": "not json"}
    assert esc.trap_record({}) == {}


def test_successors_of_matches_one_generation_only() -> None:
    doc = _doc(
        extra_workflows=(_successor("workflow-reboot-after-" + SUCCESSOR, SUCCESSOR),)
    )
    assert [w["request_id"] for w in esc.successors_of(doc, RECORD)] == [SUCCESSOR]
    assert [w["request_id"] for w in esc.successors_of(doc, SUCCESSOR)] == [
        "workflow-reboot-after-" + SUCCESSOR
    ]
    assert esc.successors_of(doc, "workflow-none") == []


def test_ladder_stopped_reads_the_settled_record_without_a_rung() -> None:
    assert esc.ladder_stopped(_doc()) is False, "a SUCCEEDED RESTART_NODE is a rung"
    assert esc.ladder_stopped(_doc(reboot=False, source_state="QUARANTINED")) is True
    assert (
        esc.ladder_stopped(_doc(reboot=False, source_state="ACTION_PENDING")) is False
    )
    running = _doc(reboot=False, source_state="QUARANTINED", record_status="RUNNING")
    assert esc.ladder_stopped(running) is False


def test_exit_errors_name_the_unreleased_node_and_the_open_incidents() -> None:
    doc = _doc(successor_incident_state="ACTION_PENDING", source_state="QUARANTINED")
    errors = esc.exit_errors(doc, esc.efa_chain(doc), _node_snapshot(isolated=True))
    text = " | ".join(errors)
    assert f"escalation incident {SUCCESSOR_INCIDENT} is ACTION_PENDING" in text
    assert "EFA incident inc-efa is QUARANTINED" in text
    assert "still carries the product's isolation" in text
    assert esc.exit_errors(_doc(), esc.efa_chain(_doc()), _node_snapshot()) == []


# --------------------------------------------------------------------------- #
# Fixture: target selection and cleanup
# --------------------------------------------------------------------------- #
def test_target_bdf_blocked_with_fewer_than_two_functions(tmp_path: Path) -> None:
    probe = FakeProbe(
        {
            "functions": {
                "functions": [{"pci_bdf": BDF, "driver": "efa", "bound": True}],
                "bound_count": 1,
            }
        }
    )
    ctx = _ctx("bind_failure", doc=_doc(), probe=probe, tmp_path=tmp_path)
    fixture = mod.EfaDriverFixture(ctx)
    with pytest.raises(VariantBlocked):
        fixture.target_bdf()


def test_cleanup_refusal_is_recorded_on_the_lifo_stack(tmp_path: Path) -> None:
    probe = _probe(release_error=HostProbeError("EFA driver fixture did not restore"))
    ctx = _ctx(
        "bind_failure",
        doc=_doc(),
        probe=probe,
        tmp_path=tmp_path,
        node_snapshot=_node_snapshot(boot=BASE_BOOT),
    )
    store_poll_clock(ctx)
    fixture = mod.EfaDriverFixture(ctx)
    fixture.survey()  # select so release has a BDF to act on, and record the boot
    results = ctx.cleanup.run()
    failed = [
        item for item in results if not item["ok"] and item["name"] == "efa-driver"
    ]
    assert failed, results
    assert "did not restore" in failed[0]["error"], failed
    assert "did not reboot" in failed[0]["error"], failed
    assert ctx.cleanup.failed, "a refused release must be recorded as a failed cleanup"


def test_release_after_the_reboot_is_judged_by_the_boot_and_the_inventory(
    tmp_path: Path,
) -> None:
    """The Pod died with the product's reboot (live 2026-10-27): the release
    is judged by the changed boot id and the product's post-boot inventory."""

    probe = _probe(
        release_error=HostProbeTransportError("owned host probe Pod has terminated")
    )
    ctx = _ctx("bind_failure", doc=_doc(), probe=probe, tmp_path=tmp_path)
    fixture = mod.EfaDriverFixture(ctx)
    fixture.survey()
    fixture.expected_count = 16
    results = ctx.cleanup.run()
    (item,) = [entry for entry in results if entry["name"] == "efa-driver"]
    assert item["ok"], item
    receipt = item["result"]
    assert receipt["clean"] is True
    assert receipt["judged_by"] == "node reboot"
    assert receipt["rebooted"] == {
        "boot_id_before": BASE_BOOT,
        "boot_id_after": NEW_BOOT,
    }
    assert receipt["node_after"]["inventory"]["active"]["value"] == 16
    assert receipt["node_after"]["expected_count"] == 16
    assert "Pod has terminated" in receipt["release_error"]
    written = json.loads((ctx.case_dir / "efa-driver-release.json").read_text())
    assert written["judged_by"] == "node reboot"
    assert (ctx.case_dir / "node-return-timeline.json").is_file(), (
        "the node polls of the probe-loss close are evidence"
    )
    assert not ctx.cleanup.failed, ctx.cleanup.failed


def test_release_after_the_reboot_fails_when_the_inventory_stays_short(
    tmp_path: Path,
) -> None:
    probe = _probe(
        release_error=HostProbeTransportError("owned host probe Pod has terminated")
    )
    ctx = _ctx(
        "bind_failure", doc=_doc(recovered_active=15), probe=probe, tmp_path=tmp_path
    )
    store_poll_clock(ctx)
    fixture = mod.EfaDriverFixture(ctx)
    fixture.survey()
    fixture.expected_count = 16
    results = ctx.cleanup.run()
    (item,) = [entry for entry in results if entry["name"] == "efa-driver"]
    assert not item["ok"], item
    assert "does not prove the function bound" in item["error"], item
    assert "efa_inventory_active_count=16" in item["error"], item


def test_release_without_a_reboot_keeps_the_probe_failure(tmp_path: Path) -> None:
    probe = _probe(
        release_error=HostProbeTransportError("owned host probe Pod has terminated")
    )
    ctx = _ctx(
        "bind_failure",
        doc=_doc(),
        probe=probe,
        tmp_path=tmp_path,
        node_snapshot=_node_snapshot(boot=BASE_BOOT),
    )
    store_poll_clock(ctx)
    fixture = mod.EfaDriverFixture(ctx)
    fixture.survey()
    results = ctx.cleanup.run()
    (item,) = [entry for entry in results if entry["name"] == "efa-driver"]
    assert not item["ok"], item
    assert "Pod has terminated" in item["error"] and "did not reboot" in item["error"]


def test_release_retried_through_the_probe_when_the_node_returns_without_a_reboot(
    tmp_path: Path,
) -> None:
    """A quiesce window (kubelet down) refuses the first release; the node
    comes back on the same boot and the retry through the same Pod cleans."""

    attempts = {"n": 0}

    def release(_arguments: tuple[str, ...]) -> dict[str, Any]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise HostProbeError("host probe response is missing (exit 1)")
        return {"clean": True, "bound": True, "pci_bdf": BDF}

    probe = _probe(release=release)
    ctx = _ctx(
        "bind_failure",
        doc=_doc(),
        probe=probe,
        tmp_path=tmp_path,
        node_snapshot=_node_snapshot(boot=BASE_BOOT),
    )
    fixture = mod.EfaDriverFixture(ctx)
    fixture.survey()
    results = ctx.cleanup.run()
    (item,) = [entry for entry in results if entry["name"] == "efa-driver"]
    assert item["ok"], item
    assert item["result"]["clean"] is True
    assert item["result"].get("judged_by") is None
    assert attempts["n"] == 2


# --------------------------------------------------------------------------- #
# The withheld hand-off (live 2026-10-27 07:08-07:51Z) and the receipt lookup
# --------------------------------------------------------------------------- #
def test_post_repair_failure_passes_on_the_live_command_shape(tmp_path: Path) -> None:
    """The Node Agent's receipt is under the command's ``step.operation`` and
    the step execution's details, never a top-level ``operation``."""

    ctx = _ctx(
        "post_repair_failure",
        doc=_doc(
            remediation_status="SUCCEEDED",
            node_results=REPAIRED,
            post_repair=("VALIDATE_FABRIC", "FAILED"),
            post_repair_details={
                "node_failures": {NODE: ["efa_inventory_active_count=15,expected=16"]}
            },
            live_commands=True,
        ),
        probe=_probe(trap_text=TRAP_FIRED_TEXT),
        tmp_path=tmp_path,
    )
    result = mod.post_repair_failure(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    assert result["node_result"]["rebound_pci_bdfs"] == [BDF]
    assert result["failed_post_repair_step"]["verdict"] == esc.OBSERVED
    assert result["product_gap"] is None


def test_post_repair_failure_withheld_hand_off_is_graded_as_the_products_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VALIDATE_FABRIC failed at the cap without observing the re-unbound
    function; the product withheld the reboot and opened the support hand-off;
    the node stayed cordoned. One stage fails (the claim), the product-side
    question is in ``product_gap``, and the cleanup hands the node back."""

    restores: list[dict[str, Any]] = []
    monkeypatch.setattr(
        mod,
        "restore_incidents",
        lambda ctx, incidents, *, reason: restores.append(
            {"incidents": [item["incident_id"] for item in incidents], "reason": reason}
        )
        or [{"status": "SUCCEEDED"}],
    )
    doc = _doc(
        remediation_status="SUCCEEDED",
        node_results=REPAIRED,
        post_repair=("VALIDATE_FABRIC", "FAILED"),
        post_repair_details=CAP_DETAILS,
        post_repair_error=CAP_ERROR,
        reboot=False,
        support=True,
        source_state="QUARANTINED",
        recovered_active=15,
        live_commands=True,
    )
    ctx = _ctx(
        "post_repair_failure",
        doc=doc,
        probe=_probe(trap_text=TRAP_FIRED_TEXT),
        tmp_path=tmp_path,
        node_snapshot=_node_snapshot(boot=BASE_BOOT, isolated=True),
    )
    store_poll_clock(ctx)
    result = mod.post_repair_failure(ctx)
    assert result["verdict"] == "FAIL", result
    failing = {stage for stage, errors in result["stages"].items() if errors}
    assert failing == {"post_repair_failure_observed"}, result["errors"]
    text = result["errors"][0]
    assert (
        "withheld the reboot" in text and "missing_post_action_inventory_metric" in text
    )
    assert (
        result["product_gap"]
        and "inventory_observation_window" in result["product_gap"]
    )
    assert result["failed_post_repair_step"]["verdict"] == esc.WITHHELD
    assert result["trap_witness"] == esc.JUDGED_BY_PROBE
    assert result["node_isolated_at_exit"] is True
    assert result["rebooted"] is False
    assert any("withheld" in note for note in result["notes"]), result["notes"]
    settled = json.loads((ctx.case_dir / "efa-driver-settled.json").read_text())
    assert settled["successors"][0]["request_id"] == SUPPORT
    assert settled["executions"][-1]["details"]["step_waiting_timeout_seconds"] == 600
    # The cleanup: the release first (the probe answers), then the hand-back.
    results = ctx.cleanup.run()
    assert [item["name"] for item in results] == ["efa-driver", "validated-restore"]
    assert all(item["ok"] for item in results), results
    assert restores and set(restores[0]["incidents"]) == {"inc-efa", SUPPORT_INCIDENT}
    assert "GF-GENERAL-LIVE-013" in restores[0]["reason"]


def test_post_repair_withheld_hand_off_names_a_reboot_that_should_not_have_run(
    tmp_path: Path,
) -> None:
    doc = _doc(
        remediation_status="SUCCEEDED",
        node_results=REPAIRED,
        post_repair=("VALIDATE_FABRIC", "FAILED"),
        post_repair_details=CAP_DETAILS,
        post_repair_error=CAP_ERROR,
        reboot=True,
        support=True,
    )
    errors = " | ".join(esc.handoff_shape_errors(doc, esc.efa_chain(doc)))
    assert "followed a validation that observed nothing" in errors, errors
    assert "RESTART_NODE executed 1 time(s) on a withheld reboot" in errors, errors


def test_post_repair_verdict_classifies_observed_and_withheld() -> None:
    plugin = {
        "operation": "RESTART_EFA_DEVICE_PLUGIN",
        "status": "FAILED",
        "details": {},
    }
    assert esc.post_repair_verdict(plugin) == esc.OBSERVED
    observed = {
        "operation": "VALIDATE_FABRIC",
        "status": "FAILED",
        "details": {
            "node_failures": {NODE: ["efa_inventory_active_count=15,expected=16"]}
        },
    }
    assert esc.post_repair_verdict(observed) == esc.OBSERVED
    capped = {
        "operation": "VALIDATE_FABRIC",
        "status": "FAILED",
        "details": CAP_DETAILS,
    }
    assert esc.post_repair_verdict(capped) == esc.WITHHELD
    by_text = {"operation": "VALIDATE_FABRIC", "status": "FAILED", "error": CAP_ERROR}
    assert esc.post_repair_verdict(by_text) == esc.WITHHELD
    pending_only = {
        "operation": "VALIDATE_FABRIC",
        "status": "FAILED",
        "details": {"node_pending": {NODE: {"missing_or_stale_collectors": ["HOST"]}}},
    }
    assert esc.post_repair_verdict(pending_only) == esc.WITHHELD
    bare = {"operation": "VALIDATE_FABRIC", "status": "FAILED", "details": {}}
    assert esc.post_repair_verdict(bare) == esc.OBSERVED


def test_node_result_reads_the_step_details_then_the_commands_step_operation() -> None:
    doc = _doc(
        remediation_status="SUCCEEDED", node_results=REPAIRED, live_commands=True
    )
    workflow = esc.efa_chain(doc)[1]
    assert (
        esc.node_result(doc, workflow, operation="REMEDIATE_EFA_DRIVER", node=NODE)
        == REPAIRED
    )
    # The step execution's own details win when they carry the receipt.
    workflow["step_executions"][2]["details"] = {
        "node_results": {NODE: {"rebound_pci_bdfs": [BDF], "already_bound": False}}
    }
    value = esc.node_result(doc, workflow, operation="REMEDIATE_EFA_DRIVER", node=NODE)
    assert value == {"rebound_pci_bdfs": [BDF], "already_bound": False}
    # A command of another step index is not this step's receipt.
    doc["commands"][0]["step_index"] = 7
    workflow["step_executions"][2]["details"] = {}
    assert (
        esc.node_result(doc, workflow, operation="REMEDIATE_EFA_DRIVER", node=NODE)
        == {}
    )
    assert (
        esc.node_result(doc, None, operation="REMEDIATE_EFA_DRIVER", node="other") == {}
    )


def test_hand_back_restores_only_an_isolated_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    restores: list[list[str]] = []
    monkeypatch.setattr(
        mod,
        "restore_incidents",
        lambda ctx, incidents, *, reason: restores.append(
            [item["incident_id"] for item in incidents]
        )
        or [{"status": "SUCCEEDED"}],
    )
    released = _ctx("bind_failure", doc=_doc(), probe=_probe(), tmp_path=tmp_path)
    assert mod.hand_back(released, [{"incident_id": "inc-efa"}])["skipped"] is True
    isolated = _ctx(
        "bind_failure",
        doc=_doc(),
        probe=_probe(),
        tmp_path=tmp_path / "isolated",
        node_snapshot=_node_snapshot(isolated=True),
    )
    assert mod.hand_back(isolated, [])["skipped"] is True
    value = mod.hand_back(isolated, [{"incident_id": "inc-efa"}])
    assert value["restored"] == [{"status": "SUCCEEDED"}]
    assert restores == [["inc-efa"]]


def test_ladder_stopped_waits_on_a_successor_blocked_for_an_operator() -> None:
    doc = _doc(reboot=False, source_state="QUARANTINED")
    blocked = _successor(SUCCESSOR, RECORD, status="BLOCKED")
    blocked["blocked_kind"] = "NEEDS_OPERATOR"
    blocked["step_executions"] = []
    doc["workflows"].append(blocked)
    assert esc.ladder_stopped(doc) is False, "an occupying BLOCKED record still gates"
    blocked["blocked_kind"] = "SAFETY_SETTLED"
    assert esc.ladder_stopped(doc) is True


def test_bind_failure_pass_leaves_the_hand_back_skipped(tmp_path: Path) -> None:
    ctx = _ctx("bind_failure", doc=_doc(), probe=_probe(), tmp_path=tmp_path)
    result = mod.bind_failure(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    results = ctx.cleanup.run()
    assert [item["name"] for item in results] == ["efa-driver", "validated-restore"]
    assert results[1]["result"]["skipped"] is True
