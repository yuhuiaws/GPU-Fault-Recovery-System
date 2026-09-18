"""Read-only premises and exact verdicts for collector negative injections."""

from __future__ import annotations

import re

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.e2e.regional.collector_acceptance_fixture import (
    CollectorAcceptanceFixture,
    select_workflow,
)
from scripts.e2e.regional.collector_action_guard import require_action_time
from scripts.e2e.regional.collector_case_cleanup import CaseCleanup
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    component_python,
)

FIRMWARE_REASONS = frozenset(
    {
        "UPDATE_SWFW requires GPU_FAULT_TARGET_FIRMWARE_VERSION",
        "no executable owner for softwareFirmwareUpdate",
    }
)
# The policy engine binds every XID 78 decision to the catalog it resolved
# ("exact NVIDIA Catalog <version> workflow for XID 78: UPDATE_SWFW") and the
# workflow carries that reason alongside the two gate reasons above. It is the
# only other reason the exact firmware block may show; any further block (an
# UNKNOWN workload state, a missing fence) is not this gate.
FIRMWARE_CATALOG_REASON = re.compile(
    r"exact NVIDIA Catalog \S+ workflow for XID 78: UPDATE_SWFW"
)
SCOPE_REASONS = {
    "ACCESS": "fatal access-link recovery requires the affected GPU and all participating workload GPUs",
    "UNKNOWN": "fatal SXID requires a trusted access/trunk port-scope mapping",
}
SCOPE_ACTIONS = {
    "ACCESS": "RESET_PARTICIPATING_GPUS",
    "UNKNOWN": "RESET_ALL_GPUS_AND_NVSWITCHES",
}
SCOPE_DEPENDENT_ACTIONS = frozenset(SCOPE_ACTIONS.values())
COLLECT011_SXIDS = (11001, 12001, 24007)

FIRMWARE_CPU_PROBE = r"""
import json
import os
from gpu_fault.app import ApplicationContext

context = ApplicationContext.from_environment()
print(json.dumps({
    "target_present": bool(context.orchestrator.target_firmware_version),
    "update_command_present": bool(os.getenv("GPU_FAULT_FIRMWARE_UPDATE_COMMAND")),
    "verify_command_present": bool(os.getenv("GPU_FAULT_FIRMWARE_VERIFY_COMMAND")),
}))
"""

SXID_PREMISE_PROBE = r"""
import json
import sys
from datetime import datetime, timezone
from gpu_fault.app import ApplicationContext
from gpu_fault.app.ingest.faults import FaultIngestionService
from gpu_fault.policy import SxidEvent

cluster, node, bdf, product, switch, *variant = sys.argv[1:]
sxid = int(variant[0]) if variant else 11001
if sxid not in {11001, 12001, 24007}:
    raise RuntimeError("SXID scope premise code is not allowlisted")
now = datetime.now(timezone.utc)
context = ApplicationContext.from_environment()
workload = context.topology.resolve(cluster, node, now)
event = SxidEvent(
    event_id="collector-scope-readonly",
    cluster_id=cluster, node_id=node, observed_at=now,
    sxid=sxid, classification="FATAL",
    classification_source="NVIDIA_FABRIC_MANAGER_CATALOG",
    product=product, pci_bdf=bdf, switch_id=switch or None, port="12",
    workload_state=workload.workload_state,
)
event = FaultIngestionService(context)._enrich_sxid_scope(event)
print(json.dumps({
    "cluster_id": cluster, "node_id": node,
    "workload_state": workload.workload_state,
    "link_scope": event.link_scope.value,
    "link_scope_source": event.link_scope_source,
    "participating_gpu_uuids": event.participating_gpu_uuids,
    "switch_id": event.switch_id, "port": event.port, "pci_bdf": event.pci_bdf,
}))
"""


def firmware_premise(
    regional: RegionalLiveFixture, node_projection: dict[str, Any], *, node: str
) -> dict[str, Any]:
    profile = regional.store_snapshot(node=node).get("profile") or {}
    capabilities = [
        entry
        for entry in profile.get("capabilities") or []
        if entry.get("capability") == "softwareFirmwareUpdate"
    ]
    if (
        not profile.get("profile_version")
        or profile.get("warnings")
        or len(capabilities) != 1
        or capabilities[0].get("mode") != "OBSERVE"
    ):
        raise RegionalFixtureError(
            "firmware-negative premise requires the pinned OBSERVE capability"
        )
    replicas = []
    for app in ("gpu-fault-api-ha", "gpu-fault-control-worker"):
        deployment = json.loads(
            regional.kubectl("cpu", "get", "deployment", app, "-o", "json")
        )
        desired = (deployment.get("spec") or {}).get("replicas")
        before = regional.ready_pods("cpu", app)
        if type(desired) is not int or desired <= 0 or len(before) != desired:
            raise RegionalFixtureError(
                "firmware premise requires every CPU replica Ready"
            )
        for pod in before:
            output = regional.kubectl(
                "cpu",
                "exec",
                "-i",
                str(pod["name"]),
                "--",
                component_python("cpu"),
                "-",
                input_text=FIRMWARE_CPU_PROBE,
                timeout=60,
            )
            proof = json.loads(output.splitlines()[-1])
            if not isinstance(proof, dict):
                raise RegionalFixtureError("firmware premise projection is malformed")
            replicas.append({"pod_uid": pod["uid"], "app": app, **proof})
        after = regional.ready_pods("cpu", app)
        current = json.loads(
            regional.kubectl("cpu", "get", "deployment", app, "-o", "json")
        )
        if (
            before != after
            or deployment.get("metadata") != current.get("metadata")
            or desired != (current.get("spec") or {}).get("replicas")
        ):
            raise RegionalFixtureError("CPU population changed during firmware premise")
    if (regional.store_snapshot(node=node).get("profile") or {}) != profile:
        raise RegionalFixtureError("firmware Runtime Profile changed during premise")
    proof = {
        "cpu_replicas": replicas,
        "node_agent": node_projection,
        "profile_version": profile["profile_version"],
        "firmware_mode": "OBSERVE",
    }
    errors = firmware_premise_errors(proof)
    if errors:
        raise RegionalFixtureError("; ".join(errors))
    return proof


def firmware_premise_errors(proof: dict[str, Any]) -> list[str]:
    errors = []
    replicas = proof.get("cpu_replicas")
    if not isinstance(replicas, list) or not replicas:
        errors.append("CPU firmware configuration is unknown")
    elif any(
        row.get(key) is not False
        for row in replicas
        for key in (
            "target_present",
            "update_command_present",
            "verify_command_present",
        )
    ):
        errors.append("CPU firmware target or command is configured or unknown")
    node = proof.get("node_agent") or {}
    if node.get("allow_disabled") is not True or any(
        node.get(key) is not False
        for key in (
            "target_present",
            "update_command_present",
            "verify_command_present",
        )
    ):
        errors.append("Node Agent firmware capability is configured or unknown")
    return errors


def firmware_negative_errors(state: dict[str, Any]) -> list[str]:
    errors = []
    decisions = state.get("decisions") or []
    if len(decisions) != 1 or decisions[0].get("official_action") != "UPDATE_SWFW":
        errors.append("firmware decision is not the injected UPDATE_SWFW event")
    workflows = state.get("workflows") or []
    decision = decisions[0] if len(decisions) == 1 else {}
    workflow: dict[str, Any] = next(
        (
            row
            for row in workflows
            if row.get("request_id") == decision.get("workflow_request_id")
        ),
        {},
    )
    reasons = {str(item) for item in workflow.get("blocked_reasons") or []}
    if (
        workflow.get("status") != "BLOCKED"
        or not FIRMWARE_REASONS <= reasons
        or any(
            FIRMWARE_CATALOG_REASON.fullmatch(item) is None
            for item in reasons - FIRMWARE_REASONS
        )
    ):
        errors.append(
            "firmware workflow did not block for exactly missing target and owner"
        )
    if any(
        step.get("operation") == "UPDATE_SOFTWARE_FIRMWARE"
        for row in workflows
        for step in row.get("official_steps") or []
    ):
        errors.append("blocked workflow contains firmware mutation")
    if any(
        step.get("operation")
        not in {"FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"}
        for row in workflows
        for step in row.get("step_executions") or []
    ) or any(
        (command.get("step") or {}).get("operation")
        not in {"MARK_UNSCHEDULABLE", "QUARANTINE"}
        for command in state.get("commands") or []
    ):
        errors.append("firmware-negative event reached a non-containment action")
    return errors


def scope_premise_errors(
    proof: dict[str, Any], *, cluster_id: str, node: str, scope: str
) -> list[str]:
    if (
        proof.get("cluster_id") != cluster_id
        or proof.get("node_id") != node
        or proof.get("workload_state") != "IDLE"
        or proof.get("link_scope") != scope
        or (
            scope == "ACCESS"
            and proof.get("link_scope_source") != "NVIDIA_PRODUCT_INVARIANT"
        )
    ):
        return ["SXID negative scope or idle-node premise is unproven"]
    if scope == "ACCESS" and proof.get("participating_gpu_uuids") != []:
        return [
            "ACCESS SXID has participating GPUs; refusing a reset-capable injection"
        ]
    return []


def scope_negative_errors(state: dict[str, Any], *, scope: str) -> list[str]:
    return scope_variant_errors(state, scope=scope, sxid=11001)


def scope_variant_errors(state: dict[str, Any], *, scope: str, sxid: int) -> list[str]:
    events = state.get("fabric_events") or []
    decisions = state.get("decisions") or []
    if len(events) != 1 or len(decisions) != 1:
        return [
            "SXID negative requires exactly one normalized event and persisted decision"
        ]
    event, decision = events[0], decisions[0]
    errors = []
    if (
        event.get("sxid") != sxid
        or event.get("classification") != "FATAL"
        or event.get("port") != "12"
        or bool(event.get("switch_id")) != (scope == "ACCESS")
        or decision.get("event_id") != event.get("event_id")
        or decision.get("event_type") != "SXID"
        or decision.get("disposition") != "BLOCKED_MISSING_EVIDENCE"
        or decision.get("official_action") != SCOPE_ACTIONS[scope]
        or decision.get("action") is not None
        or decision.get("reasons") != [SCOPE_REASONS[scope]]
    ):
        errors.append("SXID decision does not prove the exact negative direction")
    workflows = [
        row
        for row in state.get("workflows") or []
        if row.get("request_id") == decision.get("workflow_request_id")
        and row.get("incident_id") == decision.get("incident_id")
    ]
    if len(workflows) != 1 or workflows[0].get("status") != "BLOCKED":
        errors.append("SXID decision has no exactly bound BLOCKED workflow")
    if scope == "ACCESS" and any(
        incident.get("gpu_uuids") != [] for incident in state.get("incidents") or []
    ):
        errors.append("ACCESS negative incident acquired GPU participants")
    if any(
        (command.get("step") or {}).get("operation")
        not in {"MARK_UNSCHEDULABLE", "QUARANTINE"}
        for command in state.get("commands") or []
    ):
        errors.append("SXID negative emitted a non-containment command")
    return errors


def service_state_errors(baseline: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Every unit active before the injection must remain active afterward."""

    errors = []
    for unit, before in sorted((baseline.get("services") or {}).items()):
        if before.get("ActiveState") != "active":
            continue
        current = (after.get("services") or {}).get(unit) or {}
        if current.get("ActiveState") != "active":
            errors.append(
                f"{unit} is {current.get('ActiveState') or 'absent'} after the "
                "case, active before it"
            )
    return errors


def nvswitch_pci_bdf(gpu_inventory: list[dict[str, Any]]) -> str:
    """Select an injected switch address that cannot resolve to a live GPU."""

    taken = {
        str(item.get("pci_bdf") or "").lower().split(".")[0] for item in gpu_inventory
    }
    for bus in ("ab", "ac", "ad", "ae", "af", "ba", "bb", "bc"):
        candidate = f"0000:{bus}:00"
        if candidate not in taken:
            return candidate + ".0"
    raise RegionalFixtureError("no PCI slot free of GPUs for the SXid line")


def run_scope_negative_variant(
    fixture: CollectorAcceptanceFixture,
    case_dir: Path,
    marker: str,
    profile_version: str,
    *,
    sxid: int,
    scope: str,
    cleanup: CaseCleanup,
) -> dict[str, Any]:
    if sxid not in COLLECT011_SXIDS or scope not in {"ACCESS", "UNKNOWN"}:
        raise RegionalFixtureError("SXID negative variant is not allowlisted")
    baseline = fixture.snapshot()
    # A GPU BDF would supply participants and turn this into an executable reset.
    switch_bdf = nvswitch_pci_bdf(baseline["gpu_inventory"])
    if fixture.regional.business_workloads(fixture.node):
        raise RegionalFixtureError("SXID negative requires an idle node")
    software = fixture.execute("gpu-identity")
    premise = fixture.regional.cpu_python(
        SXID_PREMISE_PROBE,
        fixture.regional.settings.cluster_id,
        fixture.node,
        switch_bdf,
        str(software.get("product") or ""),
        "nvidia-nvswitch0" if scope == "ACCESS" else "",
        str(sxid),
    )
    premise_errors = scope_premise_errors(
        premise,
        cluster_id=fixture.regional.settings.cluster_id,
        node=fixture.node,
        scope=scope,
    )
    if premise_errors:
        raise RegionalFixtureError("; ".join(premise_errors))
    injected_at = datetime.now(timezone.utc)
    cleanup.register_seed(fixture, marker)
    require_action_time(180)
    fixture.execute(
        "append-sxid",
        "--sxid",
        str(sxid),
        "--marker",
        marker,
        "--pci-bdf",
        switch_bdf,
        "--classification",
        "Fatal",
        "--message",
        "NVLINK_FATAL_ERROR",
        "--include-switch" if scope == "ACCESS" else "--no-include-switch",
    )
    state = fixture.wait_marker(
        marker,
        case_dir=case_dir,
        timeout_seconds=300,
        terminal_workflow=True,
        observed_after=injected_at,
    )
    cleanup.register_state(fixture, state)
    errors = scope_variant_errors(state, scope=scope, sxid=sxid)
    workflow = select_workflow(
        state.get("workflows") or [], official_actions=SCOPE_DEPENDENT_ACTIONS
    )
    if workflow is None:
        errors.append(f"{fixture.node}: no workflow decided a scope-dependent reset")
    elif workflow.get("status") != "BLOCKED":
        errors.append(f"{fixture.node}: scope-dependent SXID did not fail closed")
    after = fixture.snapshot()
    if after["boot_id"] != baseline["boot_id"]:
        errors.append(f"{fixture.node}: node rebooted during the fail-closed SXID")
    errors.extend(service_state_errors(baseline, after))
    restore = cleanup.restore(
        fixture,
        state,
        profile_version=profile_version,
        reason=f"COLLECT-011 SXID{sxid} {scope} validated cleanup",
    )
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "marker": marker,
        "sxid": sxid,
        "scope": scope,
        "negative_premise": premise,
        "selected_workflow_id": (workflow or {}).get("request_id"),
        "restore_workflows": restore,
        "validation_scope": "live-software-log-fail-closed",
        "physical_fault_injected": False,
    }
