"""GF-GENERAL-LIVE-032 unknown error / missing evidence (reviewed kmsg replay).

The kernel collector reads ``GPU_FAULT_KMSG_PATH``; the kernel-log replay helper
(kmsg mode) repoints that one unit at a reviewed fixture file, so a reviewed
``NVRM: Xid`` line travels through the real ``KernelLogCollector`` parser and the
real pinned XID/SXID catalog (``policy/engine.py``). A code the shipped catalog
does not know resolves to ``SITE_SAFETY`` -- operator-required, with no NVIDIA
recovery action -- and a repeat of the same event must not isolate twice. The
verdict reads the product's own ``FaultPolicyDecision`` records; nothing is
faked and no policy conclusion is posted.

The variants that need a genuinely unknown *product*, an unconfigured Runtime
Profile authority, or reviewed inventory/version/topology source fixtures are
not reachable on this single H200 site and stay NOT_RUN with their reason -- the
denominator keeps them.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    utc_now,
)
from scripts.e2e.regional.general_live_kernel_replay import kernel_replay, reviewed_record
from scripts.e2e.regional.general_live_registry import (
    CAPABILITY,
    LIVE_REPLAY,
    VariantSpec,
    deferred_variant,
)
from scripts.e2e.regional.general_live_verdicts import hardware_errors, verdict

CASE = "GF-GENERAL-LIVE-032"
KERNEL_SOURCE_REPLAY = "KERNEL_SOURCE_REPLAY"
#: A code well outside the pinned catalog (which tops out near XID 150). Chosen
#: so ``policy/engine.py`` ``_resolve_xid`` finds no rule -> SITE_SAFETY.
UNKNOWN_XID = 987
DETECTION_BUDGET = 420

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
    return started


def _wait_decisions(ctx: VariantContext, since: Any, *, xid: int, name: str) -> list[dict[str, Any]]:
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
    message = f"NVRM: Xid (PCI:0000:53:00): {UNKNOWN_XID}, acceptance unknown-code replay"
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
        notes=["the same reviewed event re-posted across polls must dedup to one isolation"],
        extra={"decisions": decisions},
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


def _implemented(variant: str, injection: str, expected: str, execute: Any) -> VariantSpec:
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
        lanes=("eks",),
        expected="an unknown product -> BLOCKED_MISSING_EVIDENCE, operator-required, no action",
    ),
    deferred_variant(
        CASE,
        "missing_topology",
        proof_mode=LIVE_REPLAY,
        reason=(
            "FIXTURE_NOT_READY: needs a reviewed NVSwitch/NVLink topology source fixture "
            "the replay helper does not yet install"
        ),
        lanes=("eks", "hp"),
        expected="an SXID without topology evidence -> BLOCKED_MISSING_EVIDENCE",
    ),
    deferred_variant(
        CASE,
        "missing_version",
        proof_mode=LIVE_REPLAY,
        reason=(
            "FIXTURE_NOT_READY: the driver version is stamped from the collector context; "
            "a reviewed missing-version source fixture is not built this phase"
        ),
        lanes=("eks",),
        expected="a workflow decoder without its version evidence -> BLOCKED_MISSING_EVIDENCE",
    ),
    deferred_variant(
        CASE,
        "observe_profile",
        proof_mode=LIVE_REPLAY,
        reason=(
            "NOT_AUTHORIZED: needs a reviewed OBSERVE Runtime Profile variant; the site's "
            "node-key custody chain is deliberately not configured (AUTH-015 BLOCKED)"
        ),
        lanes=("eks",),
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
        lanes=("eks",),
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
        lanes=("eks",),
        expected="a known ordinary XID on an old driver still gets its original action, not blocked by XID 154 linkage",
    ),
    _implemented(
        "duplicate",
        "the same reviewed unknown-XID line re-posted across polls (kmsg replay)",
        "one FaultPolicyDecision / one isolation; repeats are deduped, no re-isolation",
        duplicate,
    ),
]
