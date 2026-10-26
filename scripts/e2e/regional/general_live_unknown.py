"""GF-GENERAL-LIVE-032 unknown error / missing evidence (reviewed kmsg replay).

The kernel collector reads ``GPU_FAULT_KMSG_PATH``; the kernel-log replay helper
(kmsg mode) repoints that one unit at a reviewed fixture file, so a reviewed
``NVRM: Xid`` line travels through the real ``KernelLogCollector`` parser and the
real pinned XID/SXID catalog (``policy/engine.py``). A code the shipped catalog
does not know resolves to ``SITE_SAFETY`` -- operator-required, with no NVIDIA
recovery action -- and a repeat of the same event must not isolate twice. The
verdict reads the product's own ``FaultPolicyDecision`` records; nothing is
faked and no policy conclusion is posted.

``missing_topology`` rides the same kmsg replay with a reviewed Fabric Manager
SXID line: the kernel collector forwards an ``SXid (PCI:...): <code>, Fatal``
summary line (``collectors/logs/kernel.py`` SXID_SUMMARY_PATTERN), the parser
leaves the link scope UNKNOWN (``nvidia_logs.py`` ``_sxid_scope``), and the
control plane resolves it only from fresh trusted ``nvswitch_port_topology``
telemetry or -- when the line names a port on an H100/H200 -- the product
invariant (``telemetry.py`` NvSwitchPortTopologyService). The host collector's
topology query runs only where ``GPU_FAULT_NVSWITCH_TOPOLOGY_COMMAND`` is set;
the node installer's default is empty and no deploy path on this site sets it,
so the trusted topology is genuinely absent here and nothing is faked: a FATAL
catalog code whose official action is scope-dependent (``SCOPE_DEPENDENT_RESET``)
and whose line names no link must resolve BLOCKED_MISSING_EVIDENCE ("requires a
trusted access/trunk port-scope mapping"), operator-required, with the full
fabric reset withheld -- the gap kept explicit, as the spec demands. The product
quarantines the node under site safety for it; the variant hands the node back
through the product's validated restore.

``missing_version`` is the one decoder gate that reads version evidence: the
NVLink5 workflow (``_workflow_nvlink5``, XID 144-150) blocks when the event
carries no ``driver_branch``, and that branch is stamped by the *kernel
collector's* own ``CollectorContext`` from its ``nvidia-smi
--query-gpu=driver_version`` at unit start (``collectors/context.py``). The
variant opens the kernel collector's command window
(``KERNEL_COLLECTOR_SOURCE_REPLAY``: the reviewed wrapper answers only that
query with a failure, ``GPU_FAULT_DRIVER_BRANCH`` is unset for the window),
proves from the wrapper's journal and the unit's own journal that the restarted
collector asked and got no version, then replays a real-world XID 149
NETIR_LINK_EVT line (IntrInfo/Error Status included) through the kmsg replay
and reads the product's decision and the event it graded. The Catalog's product
gate runs *before* that decoder (``_resolve_xid`` -> ``_product_gate``) and XID
144-150 apply to the B100/GB200 families only, so on this site's H200 nodes
every such XID is NOT_APPLICABLE before the version is ever asked for: the
variant records NOT_RUN(CAPABILITY_ABSENT) from the node's real GPU model
before any mutation, and runs for real on a Blackwell node.

The variants that need a genuinely unknown *product* or an unconfigured Runtime
Profile authority are not reachable on this site and stay NOT_RUN with their
reason -- the denominator keeps them.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_common import (
    activity,
    restore_incidents,
    run_survey,
    wait_settled,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantBlocked,
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
    utc_now,
)
from scripts.e2e.regional.general_live_gpu_replay import PORT as DCGM_PORT
from scripts.e2e.regional.general_live_gpu_replay import GpuSurvey
from scripts.e2e.regional.general_live_kernel_replay import (
    kernel_replay,
    kmsg_follow_error,
    reviewed_record,
)
from scripts.e2e.regional.general_live_kernel_xid import (
    catalog_entry,
    inject_reviewed_kernel_line,
    kmsg_pci_bdf,
    product_family,
)
from scripts.e2e.regional.general_live_registry import (
    LIVE_REPLAY,
    VariantSpec,
    deferred_variant,
)
from scripts.e2e.regional.general_live_source_replay import (
    KERNEL_COLLECTOR_SOURCE_REPLAY,
    HostSourceReplay,
    command_plan,
    kernel_collector_source_replay,
    rule,
)
from scripts.e2e.regional.general_live_verdicts import hardware_errors, verdict

CASE = "GF-GENERAL-LIVE-032"
KERNEL_SOURCE_REPLAY = "KERNEL_SOURCE_REPLAY"
#: A code well outside the pinned catalog (which tops out near XID 150). Chosen
#: so ``policy/engine.py`` ``_resolve_xid`` finds no rule -> SITE_SAFETY.
UNKNOWN_XID = 987
#: A pinned-catalog FATAL SXID (``data/nvidia-fabric-manager-sxid-*.yaml``,
#: SCOPE_DEPENDENT_RESET) that is not Always Fatal: its action depends on the
#: access/trunk scope, so without trusted topology it is BLOCKED_MISSING_EVIDENCE.
MISSING_TOPOLOGY_SXID = 11001
#: The reviewed kmsg line: a Fabric Manager summary line (code, Fatal) from an
#: NVSwitch, deliberately naming no ``Link N`` so the H200 product invariant
#: cannot stand in for the missing topology.
MISSING_TOPOLOGY_LINE = (
    f"nvidia-nvswitch0: SXid (PCI:0000:05:00.0): {MISSING_TOPOLOGY_SXID}, Fatal, "
    "acceptance missing-topology replay without a link"
)
BLOCKED_MISSING_EVIDENCE = "BLOCKED_MISSING_EVIDENCE"
TOPOLOGY_REASON = "port-scope mapping"
DETECTION_BUDGET = 420
#: The site-safety quarantine workflow of a BLOCKED SXID settles in minutes.
SETTLE_BUDGET = 600
#: The WORKFLOW_NVLINK5_ERR XID of ``missing_version``: 149 NVLINK_NETIR_ERROR
#: (``nvidia-xid-catalog-610`` products B100/GB200), the one of the family with
#: a public real B-series kernel line (the gpud project's kmsg fixture, kept in
#: ``tests/nvidia_logs`` as the firmware's two-register ``(0x%08x 0x%08x)``
#: payload shape -- never the Catalog prose's seven-value template).
MISSING_VERSION_XID = 149
#: That line's IntrInfo / Error Status. Under the pinned V1 decode (driver <
#: R575) it is NETIR_LINK_EVT/NETIR_LINK_DOWN -> RESET_GPU; under V2 (R575+,
#: this site's R580) it matches no rule. Both decodes need the driver branch
#: first, so with none stamped the engine must block for the version -- and
#: the variant proves the version is withheld *before* the line is injected,
#: because on a V1 driver a window that never took effect would reset a GPU.
MISSING_VERSION_INTR_INFO = 0x004505C6
MISSING_VERSION_ERROR_STATUS = 0x00000000
MISSING_VERSION_PAYLOAD = "NETIR_LINK_EVT Fatal XC0 i0 Link 08 (0x004505c6 0x00000000)"
NVLINK5_WORKFLOW = "WORKFLOW_NVLINK5_ERR"
#: ``policy/engine.py`` ``_workflow_nvlink5``: "NVLink5 decode requires driver
#: branch, IntrInfo and Error Status".
VERSION_REASON = "driver branch"
#: The reviewed answer the wrapper serves the kernel collector's
#: ``--query-gpu=driver_version``: a non-zero exit with an honest message, which
#: ``collectors/gpu/discovery.py`` reports as GpuSoftwareDiscoveryUnavailable and
#: ``collectors/context.py`` turns into a context with no driver branch.
VERSION_WITHHELD_EXIT = 1
VERSION_WITHHELD_STDERR = (
    "gfrs-live KERNEL_COLLECTOR_SOURCE_REPLAY: driver version withheld by the "
    "reviewed replay (not a driver fault)\n"
)
#: What the kernel collector logs about its discovery (``collectors/context.py``
#: WARNING on failure; ``collectors/gpu/discovery.py`` INFO on success).
VERSION_UNAVAILABLE_LINE = "software version discovery unavailable"
VERSION_DISCOVERED_LINE = "discovered NVIDIA driver branch"
#: How long the restarted collector gets to ask the wrapper for its version.
VERSION_QUERY_BUDGET = 180

#: Read the product's own SXID policy decisions for one node since a moment:
#: every incident the node got since then, joined to the FaultPolicyDecision
#: saved for its event (``store.save_xid_policy_decision`` keeps SXID decisions
#: too). Nothing here is a prefabricated conclusion.
SXID_DECISION_PROBE = r"""
import json
import sys
from datetime import datetime

from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState

cluster_id, node_id, since_text = sys.argv[1:4]
since = datetime.fromisoformat(since_text.replace("Z", "+00:00"))
store = ApplicationContext.from_environment().store
decisions = []
for incident in store.list_incidents_by_state(
    cluster_id, list(IncidentState), node_ids={node_id}
):
    if incident.created_at < since:
        continue
    decision = store.get_xid_policy_decision(incident.event_id)
    if decision is None or decision.event_type.value != "SXID":
        continue
    decisions.append({
        "event_id": decision.event_id,
        "incident_id": incident.incident_id,
        "incident_state": incident.state.value,
        "disposition": decision.disposition.value,
        "official_action": decision.official_action,
        "investigatory_action": decision.investigatory_action,
        "action": decision.action.value if decision.action else None,
        "safety_action": decision.safety_action.value if decision.safety_action else None,
        "pre_actions": [item.value for item in decision.pre_actions],
        "requires_operator": decision.requires_operator,
        "workflow_request_id": decision.workflow_request_id,
        "duplicate": decision.duplicate,
        "reasons": list(decision.reasons),
    })
print(json.dumps({"decisions": decisions}, sort_keys=True, default=str))
"""

#: Read the product's own XID policy decisions for one node since a moment. The
#: engine grades an unknown code SITE_SAFETY with no NVIDIA action; this reads
#: that back rather than a prefabricated conclusion.
XID_DECISION_PROBE = r"""
import json
import sys
from datetime import datetime

from gpu_fault.app import ApplicationContext

cluster_id, node_id, since_text = sys.argv[1:4]
since = datetime.fromisoformat(since_text.replace("Z", "+00:00"))
store = ApplicationContext.from_environment().store
decisions = []
for event in store.list_xid_events(cluster_id, node_id, observed_after=since):
    decision = store.get_xid_policy_decision(event.event_id)
    if decision is None:
        decisions.append({"event_id": event.event_id, "xid": event.xid, "decision": None})
        continue
    decisions.append({
        "event_id": event.event_id,
        "xid": event.xid,
        "disposition": decision.disposition.value,
        "action": decision.action.value if decision.action else None,
        "requires_operator": decision.requires_operator,
        "incident_id": decision.incident_id,
        "workflow_request_id": decision.workflow_request_id,
        "duplicate": decision.duplicate,
        "reasons": list(decision.reasons),
    })
print(json.dumps({"decisions": decisions}, sort_keys=True, default=str))
"""


#: The product's XID events of one code on one node since a moment, with the
#: evidence the NVLink5 decoder gates on (``driver_branch``, ``intr_info``,
#: ``error_status``, ``product``) as the kernel collector stamped and the
#: parser extracted them, joined to the FaultPolicyDecision. Read-only.
VERSION_DECISION_PROBE = r"""
import json
import sys
from datetime import datetime

from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError

cluster_id, node_id, since_text, xid_text = sys.argv[1:5]
since = datetime.fromisoformat(since_text.replace("Z", "+00:00"))
wanted = int(xid_text)
store = ApplicationContext.from_environment().store


def plain(value):
    return getattr(value, "value", value) if value is not None else None


decisions = []
for event in store.list_xid_events(cluster_id, node_id, observed_after=since):
    if event.xid != wanted:
        continue
    entry = {
        "event_id": event.event_id,
        "xid": event.xid,
        "product": event.product,
        "driver_branch": event.driver_branch,
        "intr_info": event.intr_info,
        "error_status": event.error_status,
        "gpu_uuid": event.gpu_uuid,
        "pci_bdf": event.pci_bdf,
        "event_source": plain(event.event_source),
        "observed_at": event.observed_at.isoformat(),
        "decision": False,
    }
    decision = store.get_xid_policy_decision(event.event_id)
    if decision is not None:
        incident_state = None
        if decision.incident_id:
            try:
                incident_state = plain(store.get_incident(decision.incident_id).state)
            except NotFoundError:
                incident_state = None
        entry.update({
            "decision": True,
            "disposition": plain(decision.disposition),
            "official_action": decision.official_action,
            "investigatory_action": decision.investigatory_action,
            "action": plain(decision.action),
            "safety_action": plain(decision.safety_action),
            "pre_actions": [plain(item) for item in decision.pre_actions],
            "requires_operator": decision.requires_operator,
            "incident_id": decision.incident_id,
            "incident_state": incident_state,
            "workflow_request_id": decision.workflow_request_id,
            "duplicate": decision.duplicate,
            "decoded_subcode": decision.decoded_subcode,
            "matched_decode_rules": list(decision.matched_decode_rules),
            "reasons": list(decision.reasons),
        })
    decisions.append(entry)
print(json.dumps({"decisions": decisions}, sort_keys=True, default=str))
"""


def _xid_decisions(ctx: VariantContext, since: Any) -> list[dict[str, Any]]:
    value = ctx.regional.cpu_python(
        XID_DECISION_PROBE,
        ctx.settings.regional.cluster_id,
        ctx.settings.node,
        since.isoformat(),
        timeout=120,
    )
    return list(value.get("decisions") or [])


def _drive_kmsg(ctx: VariantContext, message: str) -> Any:
    replay = kernel_replay(ctx, mode="kmsg")
    replay.create()
    replay.install([reviewed_record(message, transport="kernel", emissions=5)])
    opened = replay.open(approved=True, restore_seconds=ctx.spec.max_minutes * 60)
    ctx.evidence.write("replay-open", opened)
    started = utc_now()
    replay.arm()
    ctx.hold(90, name="inject")
    oracle = replay.read_back()
    ctx.evidence.append_oracle([{"injected": oracle.get("injected")}], source="kmsg")
    injected = oracle.get("injected") or {}
    if not injected.get("fixture_lines"):
        raise VariantInconclusive("the kmsg fixture received no reviewed records")
    # A full fixture the collector never read proves nothing about the grading.
    not_followed = kmsg_follow_error(injected)
    if not_followed:
        raise VariantInconclusive(not_followed)
    return started


def _wait_decisions(
    ctx: VariantContext, since: Any, *, xid: int, name: str
) -> list[dict[str, Any]]:
    last: list[dict[str, Any]] = []

    def accept() -> list[dict[str, Any]] | None:
        nonlocal last
        last = [item for item in _xid_decisions(ctx, since) if item.get("xid") == xid]
        return last or None

    value = ctx.wait_until(
        accept, timeout_seconds=DETECTION_BUDGET, poll_seconds=15, name=name
    )
    return value if value is not None else last


def unknown_xid(ctx: VariantContext) -> dict[str, Any]:
    message = (
        f"NVRM: Xid (PCI:0000:53:00): {UNKNOWN_XID}, acceptance unknown-code replay"
    )
    started = _drive_kmsg(ctx, message)
    decisions = _wait_decisions(ctx, started, xid=UNKNOWN_XID, name="unknown-xid")
    from scripts.e2e.regional.general_live_common import activity

    observed = activity(ctx, started, kinds=("NVIDIA_KERNEL",))
    stages = {
        "graded": _site_safety_errors(decisions),
        "no_nvidia_action": [
            f"an unknown XID produced an NVIDIA recovery action {item['action']}"
            for item in decisions
            if item.get("action") not in (None, "NO_ACTION")
        ],
        "hardware": hardware_errors(observed),
    }
    return _result(stages, extra={"decisions": decisions})


def _site_safety_errors(decisions: list[dict[str, Any]]) -> list[str]:
    if not decisions:
        return ["the unknown XID did not reach a FaultPolicyDecision"]
    errors = []
    for item in decisions:
        if item.get("disposition") != "SITE_SAFETY":
            errors.append(
                f"unknown XID {item.get('xid')} disposition is {item.get('disposition')},"
                " expected SITE_SAFETY"
            )
        if not item.get("requires_operator"):
            errors.append("an unknown XID did not require an operator")
    return errors


def duplicate(ctx: VariantContext) -> dict[str, Any]:
    """A repeat of the same unknown event must not isolate twice."""

    message = f"NVRM: Xid (PCI:0000:53:00): {UNKNOWN_XID}, acceptance duplicate replay"
    started = _drive_kmsg(ctx, message)
    decisions = _wait_decisions(ctx, started, xid=UNKNOWN_XID, name="duplicate")
    non_duplicate = [item for item in decisions if not item.get("duplicate")]
    incidents = {
        item.get("incident_id") for item in decisions if item.get("incident_id")
    }
    stages = {
        "graded": _site_safety_errors(decisions),
        "single_isolation": []
        if len(non_duplicate) <= 1 and len(incidents) <= 1
        else [
            f"a repeated unknown XID isolated {len(incidents)} times "
            f"({len(non_duplicate)} non-duplicate decisions)"
        ],
    }
    return _result(
        stages,
        notes=[
            "the same reviewed event re-posted across polls must dedup to one isolation"
        ],
        extra={"decisions": decisions},
    )


# --------------------------------------------------------------------------- #
# missing_topology: a fatal SXID with no trusted port-scope evidence
# --------------------------------------------------------------------------- #
def _sxid_decisions(
    ctx: VariantContext, since: Any, *, sxid: int
) -> list[dict[str, Any]]:
    value = ctx.regional.cpu_python(
        SXID_DECISION_PROBE,
        ctx.settings.regional.cluster_id,
        ctx.settings.node,
        since.isoformat(),
        timeout=120,
    )
    marker = f"-sxid-{sxid}-"
    return [
        item
        for item in value.get("decisions") or []
        if marker in str(item.get("event_id") or "")
    ]


def _wait_sxid_decisions(
    ctx: VariantContext, since: Any, *, sxid: int, name: str
) -> list[dict[str, Any]]:
    last: list[dict[str, Any]] = []

    def accept() -> list[dict[str, Any]] | None:
        nonlocal last
        last = _sxid_decisions(ctx, since, sxid=sxid)
        return last or None

    value = ctx.wait_until(
        accept, timeout_seconds=DETECTION_BUDGET, poll_seconds=15, name=name
    )
    return value if value is not None else last


def missing_topology_errors(decisions: list[dict[str, Any]]) -> list[str]:
    if not decisions:
        return ["the fatal SXID did not reach a FaultPolicyDecision"]
    errors = []
    for item in decisions:
        if item.get("disposition") != BLOCKED_MISSING_EVIDENCE:
            errors.append(
                f"SXID {MISSING_TOPOLOGY_SXID} disposition is {item.get('disposition')}, "
                f"expected {BLOCKED_MISSING_EVIDENCE}"
            )
        if not any(
            TOPOLOGY_REASON in str(reason) for reason in item.get("reasons") or []
        ):
            errors.append(
                "the decision does not name the missing trusted access/trunk "
                f"port-scope mapping: {item.get('reasons')}"
            )
        if not item.get("requires_operator"):
            errors.append(
                "a fatal SXID without topology evidence did not require an operator"
            )
        if item.get("action") not in (None, "NO_ACTION"):
            errors.append(
                f"a fatal SXID without topology evidence earned action {item['action']}"
            )
    return errors


def missing_topology(ctx: VariantContext) -> dict[str, Any]:
    """A fatal SXID naming no link, with no trusted topology on the node.

    PASS: the product keeps the gap explicit -- BLOCKED_MISSING_EVIDENCE naming
    the missing port-scope mapping, operator-required, the full fabric reset
    withheld -- and plans no reset or reboot. The site-safety quarantine it
    applies is released through the product's validated restore.
    """

    started = _drive_kmsg(ctx, MISSING_TOPOLOGY_LINE)
    decisions = _wait_sxid_decisions(
        ctx, started, sxid=MISSING_TOPOLOGY_SXID, name="missing-topology"
    )
    incident_ids = sorted(
        {str(item["incident_id"]) for item in decisions if item.get("incident_id")}
    )
    observed = (
        wait_settled(
            ctx,
            started,
            incident_ids,
            timeout_seconds=SETTLE_BUDGET,
            name="missing-topology-settled",
        )
        if incident_ids
        else activity(ctx, started, kinds=("NVIDIA_KERNEL",))
    )
    stages = {
        "blocked": missing_topology_errors(decisions),
        "reset_withheld": [
            f"the withheld official action is {item.get('official_action')}, "
            "expected RESET_ALL_GPUS_AND_NVSWITCHES"
            for item in decisions
            if item.get("official_action") != "RESET_ALL_GPUS_AND_NVSWITCHES"
        ],
        "no_hardware_step": hardware_errors(observed),
    }
    restored: list[dict[str, Any]] = []
    if incident_ids:
        restored = restore_incidents(
            ctx,
            [{"incident_id": item} for item in incident_ids],
            reason=f"{ctx.spec.key} acceptance restore",
        )
    return _result(
        stages,
        notes=[
            "GPU_FAULT_NVSWITCH_TOPOLOGY_COMMAND is unset on this site (installer "
            "default), so no trusted nvswitch_port_topology exists; the line names no "
            "link, so the H200 access-only invariant cannot apply either",
            "the fatal SXID is quarantined under site safety and handed back through "
            "the validated restore",
        ],
        extra={
            "decisions": decisions,
            "incidents": incident_ids,
            "restore": restored,
            "sxid_line": MISSING_TOPOLOGY_LINE,
        },
    )


# --------------------------------------------------------------------------- #
# missing_version: a WORKFLOW_NVLINK5_ERR XID from a kernel collector that
# discovered no driver branch
# --------------------------------------------------------------------------- #
def version_gate_errors(
    product: str | None, *, xid: int = MISSING_VERSION_XID
) -> list[str]:
    """Why the NVLink5 decoder's version gate is unreachable for ``product``.

    Empty when a replayed ``xid`` on a GPU of that product reaches
    ``_workflow_nvlink5`` at all. The engine's guards run in a fixed order
    (``_resolve_xid``): catalog membership, then the product gate, then the
    workflow resolver -- so a product outside the rule's families is
    NOT_APPLICABLE before the decoder can ask for the driver branch.
    """

    entry = catalog_entry(xid)
    if entry is None:
        return [f"XID {xid} is absent from the pinned catalog"]
    if entry.immediate_action != NVLINK5_WORKFLOW:
        return [
            f"XID {xid} ({entry.mnemonic}) Immediate Action is "
            f"{entry.immediate_action}, not {NVLINK5_WORKFLOW}; its decoder has no "
            "version gate"
        ]
    family = product_family(product)
    if family is None or family not in entry.products:
        return [
            "the Catalog's product gate (policy/engine.py _resolve_xid -> "
            f"_product_gate, before the {NVLINK5_WORKFLOW} resolver) answers "
            f"NOT_APPLICABLE for XID {xid} on product {product or 'UNKNOWN'} "
            f"(family {family or 'UNKNOWN'}; catalog products {list(entry.products)}): "
            "the decoder never asks for the driver branch, so the version gate is "
            f"reachable only on a {'/'.join(entry.products)}-family node"
        ]
    return []


def missing_version_line(pci_bdf: str, run_id: str) -> str:
    """The reviewed kernel line: the real B-series XID 149 shape, run-tagged."""

    if ":" not in pci_bdf:
        raise ValueError("a kernel XID line needs the GPU's PCI BDF")
    return (
        f"NVRM: Xid (PCI:{pci_bdf}): {MISSING_VERSION_XID}, pid=0, name=gfrs-live, "
        f"{MISSING_VERSION_PAYLOAD} {run_id} reviewed kernel-source replay"
    )


def version_rewrites(audit: dict[str, Any]) -> list[dict[str, Any]]:
    """The wrapper journal entries where a rule answered the driver-version query."""

    return [
        item
        for item in audit.get("entries") or []
        if item.get("query") == "driver_version" and item.get("applied_rules")
    ]


def await_version_withheld(
    ctx: VariantContext, replay: HostSourceReplay, opened_at: Any
) -> dict[str, Any]:
    """Prove the restarted kernel collector asked for, and got, no driver version.

    Two records, both the fixture's own oracle: the wrapper journal must show
    the ``driver_version`` query answered by the plan at least once since the
    restart, and the unit's own journal since ``opened_at`` must not report a
    discovered branch. Anything else is INCONCLUSIVE -- the window never took
    effect -- and no XID line is injected, because on a V1 driver that line
    would otherwise decode to RESET_GPU.
    """

    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        audit = replay.audit("nvidia-smi")
        last = {"audit": audit, "served": version_rewrites(audit)}
        return last if last["served"] else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=VERSION_QUERY_BUDGET,
        poll_seconds=10,
        name="kernel-version-query",
    )
    journal = replay.journal(since=opened_at)
    lines = [str(item) for item in journal.get("lines") or []]
    discovered = [line for line in lines if VERSION_DISCOVERED_LINE in line]
    unavailable = [line for line in lines if VERSION_UNAVAILABLE_LINE in line]
    oracle = {
        "driver_version_calls_served": len(last.get("served") or []),
        "journal_lines": len(lines),
        "journal_unavailable_lines": len(unavailable),
        "journal_discovered_lines": len(discovered),
        **replay.evidence(tool="nvidia-smi"),
    }
    ctx.evidence.write(
        "kernel-version-window", {"audit": last.get("audit"), "journal": journal}
    )
    ctx.evidence.append_oracle([oracle], source="kernel-collector-window")
    if value is None:
        raise VariantInconclusive(
            "the kernel collector did not ask the wrapper for its driver version "
            "after the restart; the command window never took effect"
        )
    if discovered:
        raise VariantInconclusive(
            "the kernel collector's journal reports a discovered driver branch after "
            "the window opened; the command window never took effect"
        )
    return oracle


def _version_decisions(ctx: VariantContext, since: Any) -> list[dict[str, Any]]:
    value = ctx.regional.cpu_python(
        VERSION_DECISION_PROBE,
        ctx.settings.regional.cluster_id,
        ctx.settings.node,
        since.isoformat(),
        str(MISSING_VERSION_XID),
        timeout=120,
    )
    return list(value.get("decisions") or [])


def _wait_version_decisions(ctx: VariantContext, since: Any) -> list[dict[str, Any]]:
    last: list[dict[str, Any]] = []

    def accept() -> list[dict[str, Any]] | None:
        nonlocal last
        last = _version_decisions(ctx, since)
        return last if any(item.get("decision") for item in last) else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=DETECTION_BUDGET,
        poll_seconds=15,
        name="missing-version",
    )
    decisions = value if value is not None else last
    ctx.evidence.write("missing-version-decisions", {"decisions": decisions})
    return decisions


def version_stamped(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Events the kernel collector stamped a driver branch on (the window failed)."""

    return [item for item in decisions if item.get("driver_branch") is not None]


def missing_version_errors(decisions: list[dict[str, Any]]) -> list[str]:
    """Why the product did not block for the missing version (empty: it did).

    The engine's reason names all three decoder inputs at once, so the event's
    own ``intr_info``/``error_status`` must be present: only then is the
    missing branch the one piece of evidence the gate is blocking on.
    """

    graded = [item for item in decisions if item.get("decision")]
    if not graded:
        return [
            f"the reviewed XID {MISSING_VERSION_XID} line did not reach a "
            "FaultPolicyDecision"
        ]
    fresh = [item for item in graded if not item.get("duplicate")] or graded
    errors: list[str] = []
    for item in fresh:
        if item.get("intr_info") is None or item.get("error_status") is None:
            errors.append(
                "the product parsed no IntrInfo/Error Status from the reviewed line, so "
                "the decoder blocks for the registers, not for the version"
            )
        if item.get("disposition") != BLOCKED_MISSING_EVIDENCE:
            errors.append(
                f"XID {MISSING_VERSION_XID} disposition is {item.get('disposition')}, "
                f"expected {BLOCKED_MISSING_EVIDENCE}"
            )
        if not any(
            VERSION_REASON in str(reason) for reason in item.get("reasons") or []
        ):
            errors.append(
                f"the decision does not name the missing driver branch: {item.get('reasons')}"
            )
        if item.get("official_action") != NVLINK5_WORKFLOW:
            errors.append(
                f"the withheld official action is {item.get('official_action')}, "
                f"expected {NVLINK5_WORKFLOW}"
            )
        if not item.get("requires_operator"):
            errors.append(
                "a decoder without its version evidence did not require an operator"
            )
        if item.get("action") not in (None, "NO_ACTION"):
            errors.append(
                f"a decoder without its version evidence earned action {item['action']}"
            )
    return errors


def missing_version(ctx: VariantContext) -> dict[str, Any]:
    """A WORKFLOW_NVLINK5_ERR XID from a kernel collector with no driver branch.

    PASS: the event carries IntrInfo/Error Status but no ``driver_branch`` and
    the product keeps the gap explicit -- BLOCKED_MISSING_EVIDENCE naming the
    driver branch, WORKFLOW_NVLINK5_ERR withheld, operator-required, no reset
    or reboot -- and the site-safety quarantine is handed back through the
    validated restore. NOT_RUN(CAPABILITY_ABSENT) before any mutation when the
    node's GPU family is outside the XID's catalog products (this site's H200);
    INCONCLUSIVE when the command window never took effect.
    """

    run_survey(ctx)
    survey = GpuSurvey(ctx.dcgm("survey", "--port", str(DCGM_PORT), timeout=300))
    ctx.evidence.write("gpu-survey", survey.raw)
    gpus = survey.gpus
    if not gpus:
        raise VariantBlocked("FIXTURE_NOT_READY", "the DCGM survey lists no GPU")
    gpu = gpus[0]
    product = survey.model(gpu)
    unreachable = version_gate_errors(product)
    if unreachable:
        raise VariantNotRun("CAPABILITY_ABSENT", "; ".join(unreachable))
    pci_bdf = kmsg_pci_bdf(survey.pci_bdf(gpu))
    if pci_bdf is None:
        raise VariantBlocked("FIXTURE_NOT_READY", f"GPU {gpu} has no PCI BDF")

    replay = kernel_collector_source_replay(ctx)
    plan = command_plan(
        "nvidia-smi",
        [
            rule(
                "driver_version",
                "exit",
                exit_code=VERSION_WITHHELD_EXIT,
                stderr=VERSION_WITHHELD_STDERR,
            )
        ],
        not_after_seconds=ctx.spec.max_minutes * 60,
    )
    ctx.evidence.write("kernel-nvidia-smi-plan", plan)
    ctx.evidence.write(
        "kernel-wrapper-install", replay.install_wrapper("nvidia-smi", plan)
    )
    opened_at = utc_now()
    opened = replay.open_command(
        ["nvidia-smi"], restore_seconds=ctx.spec.max_minutes * 60
    )
    ctx.evidence.write("kernel-window-open", opened)
    window = await_version_withheld(ctx, replay, opened_at)

    line = missing_version_line(pci_bdf, ctx.run_id)
    injection = inject_reviewed_kernel_line(
        ctx,
        xid=MISSING_VERSION_XID,
        pci_bdf=pci_bdf,
        message=line,
        label="032-missing-version",
    )
    decisions = _wait_version_decisions(ctx, injection.started)
    incident_ids = sorted(
        {str(item["incident_id"]) for item in decisions if item.get("incident_id")}
    )
    stamped = version_stamped(decisions)
    restored: list[dict[str, Any]] = []
    if stamped:
        if incident_ids:
            restored = restore_incidents(
                ctx,
                [{"incident_id": item} for item in incident_ids],
                reason=f"{ctx.spec.key} acceptance restore (window ineffective)",
            )
        raise VariantInconclusive(
            "the kernel collector stamped driver_branch="
            f"{stamped[0].get('driver_branch')} on the XID event: the command window "
            f"never took effect (restored {len(restored)} incident(s))"
        )
    observed = (
        wait_settled(
            ctx,
            injection.started,
            incident_ids,
            timeout_seconds=SETTLE_BUDGET,
            name="missing-version-settled",
        )
        if incident_ids
        else activity(ctx, injection.started, kinds=("NVIDIA_KERNEL",))
    )
    stages = {
        "blocked": missing_version_errors(decisions),
        "no_hardware_step": hardware_errors(observed),
    }
    if incident_ids:
        restored = restore_incidents(
            ctx,
            [{"incident_id": item} for item in incident_ids],
            reason=f"{ctx.spec.key} acceptance restore",
        )
    return _result(
        stages,
        notes=[
            "the kernel collector's CollectorContext discovered no driver branch "
            "(reviewed wrapper on its PATH answered the driver_version query; "
            "GPU_FAULT_DRIVER_BRANCH unset), so the XID event carries none",
            "the reviewed XID 149 line keeps the firmware's two-register payload; "
            "IntrInfo/Error Status were parsed, the driver branch is the one missing "
            "decoder input",
            "the site-safety quarantine is handed back through the validated restore",
        ],
        extra={
            "gpu": gpu,
            "product": product,
            "pci_bdf": pci_bdf,
            "xid_line": line,
            "window": window,
            "injection": injection.evidence(),
            "decisions": decisions,
            "incidents": incident_ids,
            "restore": restored,
        },
    )


def _result(
    stages: dict[str, list[str]],
    *,
    notes: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "verdict": verdict(stages),
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "notes": notes or [],
        "proof_scope": "live-replay",
        **(extra or {}),
    }


def _implemented(
    variant: str, injection: str, expected: str, execute: Any
) -> VariantSpec:
    return VariantSpec(
        case_id=CASE,
        variant=variant,
        proof_mode=LIVE_REPLAY,
        lanes=("eks", "hp"),
        risk="M",
        flags="S",
        injection=injection,
        expected=expected,
        cleanup="restore the kernel collector's /dev/kmsg source; confirm clean",
        max_minutes=16,
        approvals=(KERNEL_SOURCE_REPLAY,),
        execute=execute,
    )


VARIANTS = [
    _implemented(
        "unknown_xid",
        "reviewed NVRM Xid line with a code absent from the pinned catalog (kmsg replay)",
        "graded SITE_SAFETY, operator-required, no NVIDIA recovery action and no reset/reboot",
        unknown_xid,
    ),
    deferred_variant(
        CASE,
        "unknown_product",
        proof_mode=LIVE_REPLAY,
        reason=(
            "CAPABILITY_ABSENT: the XID event's product is stamped from the node's "
            "discovered GPU identity (H200, catalog-known), not from the reviewed source; "
            "a genuinely unknown product needs a GPU generation the catalog does not know"
        ),
        lanes=("eks", "hp"),
        expected="an unknown product -> BLOCKED_MISSING_EVIDENCE, operator-required, no action",
    ),
    VariantSpec(
        case_id=CASE,
        variant="missing_topology",
        proof_mode=LIVE_REPLAY,
        lanes=("eks", "hp"),
        risk="M",
        flags="S",
        injection=(
            f"reviewed kmsg line 'nvidia-nvswitch0: SXid (PCI:...): {MISSING_TOPOLOGY_SXID}, "
            "Fatal, ...' naming no link, on a node with no GPU_FAULT_NVSWITCH_TOPOLOGY_COMMAND "
            "(no trusted nvswitch_port_topology)"
        ),
        expected=(
            "BLOCKED_MISSING_EVIDENCE 'requires a trusted access/trunk port-scope mapping', "
            "operator-required, RESET_ALL_GPUS_AND_NVSWITCHES withheld; site-safety "
            "quarantine only, no reset/reboot; readmitted through the validated restore"
        ),
        cleanup=(
            "restore the kernel collector's /dev/kmsg source; validated restore of the "
            "quarantine incident; confirm clean"
        ),
        max_minutes=30,
        approvals=(KERNEL_SOURCE_REPLAY,),
        execute=missing_topology,
    ),
    VariantSpec(
        case_id=CASE,
        variant="missing_version",
        proof_mode=LIVE_REPLAY,
        lanes=("eks", "hp"),
        risk="M",
        flags="S",
        injection=(
            "the kernel collector's nvidia-smi driver_version query answered with a "
            "reviewed failure inside its own command window "
            "(KERNEL_COLLECTOR_SOURCE_REPLAY; GPU_FAULT_DRIVER_BRANCH unset) so its "
            "restarted CollectorContext stamps no driver branch; then the reviewed kmsg "
            f"line 'NVRM: Xid (PCI:<gpu>): {MISSING_VERSION_XID}, pid=0, name=gfrs-live, "
            f"{MISSING_VERSION_PAYLOAD} ...' through the kmsg replay"
        ),
        expected=(
            "the XID event carries intr_info/error_status but driver_branch=null; "
            "BLOCKED_MISSING_EVIDENCE 'NVLink5 decode requires driver branch, IntrInfo "
            f"and Error Status', {NVLINK5_WORKFLOW} withheld, operator-required, no "
            "reset/reboot; site-safety quarantine readmitted through the validated "
            "restore. NOT_RUN(CAPABILITY_ABSENT) on a non-Blackwell node: the Catalog's "
            "product gate answers NOT_APPLICABLE for XID 144-150 before the decoder runs"
        ),
        cleanup=(
            "restore the kernel collector's /dev/kmsg source; close its nvidia-smi "
            "command window (unit restarted on the real binary, branch re-discovered); "
            "validated restore of the quarantine incident; confirm clean"
        ),
        max_minutes=30,
        approvals=(KERNEL_COLLECTOR_SOURCE_REPLAY, KERNEL_SOURCE_REPLAY),
        execute=missing_version,
    ),
    deferred_variant(
        CASE,
        "observe_profile",
        proof_mode=LIVE_REPLAY,
        reason=(
            "NOT_AUTHORIZED: needs a reviewed OBSERVE Runtime Profile variant; the site's "
            "node-key custody chain is deliberately not configured (AUTH-015 BLOCKED)"
        ),
        lanes=("eks", "hp"),
        expected="OBSERVE profile observes only; no isolation/reset without authority",
    ),
    deferred_variant(
        CASE,
        "permitted_isolation",
        proof_mode=LIVE_REPLAY,
        reason=(
            "NOT_AUTHORIZED: a profile that permits isolation plus the custody/reset "
            "authority is not configured on this site (AUTH-015 BLOCKED)"
        ),
        lanes=("eks", "hp"),
        expected="a profile that permits isolation performs the reviewed isolation only",
    ),
    deferred_variant(
        CASE,
        "known_old_driver",
        proof_mode=LIVE_REPLAY,
        reason=(
            "FIXTURE_NOT_READY: needs a known-XID + old-driver-branch reviewed source and the "
            "XID 154 linkage context under active change in fix/xid154-linkage-drain-p2p; "
            "the coordinator sequences it after that fix"
        ),
        lanes=("eks", "hp"),
        expected="a known ordinary XID on an old driver still gets its original action, not blocked by XID 154 linkage",
    ),
    _implemented(
        "duplicate",
        "the same reviewed unknown-XID line re-posted across polls (kmsg replay)",
        "one FaultPolicyDecision / one isolation; repeats are deduped, no re-isolation",
        duplicate,
    ),
]
