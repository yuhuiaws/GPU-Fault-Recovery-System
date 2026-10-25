"""A reviewed, monitor-grade trusted kernel XID beside another fault (K-reuse).

GF-GENERAL-LIVE-020 ``independent_fatal_event`` and 024 ``independent_xid``
ask that an XID which enters through the *trusted kernel source* while a
counter diagnosis or a weak composite is active on the same GPU keeps its own
catalog handling -- it is neither absorbed into the other fault's incident nor
dropped. This module delivers that XID through the lane-K kernel-source replay
(``general_live_kernel_replay`` kmsg mode: the kernel collector's
``GPU_FAULT_KMSG_PATH`` is repointed at a reviewed fixture file, so the line
travels through the real ``KernelLogCollector`` parser, the real NVIDIA_KERNEL
delivery, GPU-identity enrichment and the pinned XID catalog) and reads the
product's own ``FaultPolicyDecision`` back.

Live safety -- the catalog entry is *monitor grade*: only an XID whose pinned
Immediate Action is ``IGNORE`` is accepted (``require_monitor_grade``). The
engine resolves that to ``NO_ACTION`` / ``MONITOR_ONLY`` whatever the node's
workload state (``policy/engine.py`` ``_direct_resolution``), the coordinator
never merges a non-EXECUTABLE decision into another incident and closes its
own incident ``RECOVERED`` on creation with no workflow, so the live run
isolates nothing and resets nothing. The entries the two variants use:

* XID 137 ``NVLINK_PRIV_ERR`` (020): an NVLink-class event beside the held
  NVLink counter fault;
* XID 92 ``EXCESSIVE_SBE_INTERRUPTS`` (024): a GPU_MEMORY-class event beside
  the weak CORRECTABLE_MEMORY_DEGRADATION composite.

What this proves: the trusted-source path is independent (own decision with
the catalog's action, own incident, not duplicate, not correlated, not merged
into the counter/composite incident, not a component of the composite) and
the concurrent diagnosis is undisturbed. What it does not prove: that a
*fatal-grade* dispatch (RESET/QUARANTINE) survives the concurrent diagnostic
workflow -- that branch needs the reset approvals and is run by the dangerous
kmsg-XID lanes, never here. The one such lane in this runner is 029
``hardware_chain_continues`` (``general_live_attempt_hardware.py``): it reuses
``inject_kernel_xid`` with its own reset-grade catalog gate (``require=``),
under ``GPU_RESET`` and ``KERNEL_SOURCE_REPLAY``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from gpu_fault.policy.catalog import load_xid_policy
from gpu_fault.policy.product_families import ProductFamilyResolver

from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
    utc_now,
)
from scripts.e2e.regional.general_live_kernel_replay import (
    KernelLogReplay,
    kernel_replay,
    records_sha256,
    reviewed_record,
)

#: The only catalog Immediate Action a reviewed kernel XID may carry here: the
#: engine maps it to NO_ACTION regardless of workload state (``DIRECT_ACTION_MAP``);
#: ``RESTART_APP`` is NO_ACTION only on a node the product knows to be IDLE.
MONITOR_GRADE_ACTIONS = frozenset({"IGNORE"})
#: The catalog family of the site's H200 nodes (``productFamilies`` prefix ``H``).
SITE_FAMILY = "H100"
#: Kernel XIDs the product joins with a PCIe replay breach (``gpu_composites``
#: ``_LINK_XIDS``); never independent of a composite, so never reviewed here.
LINK_XIDS = frozenset({32, 79})
#: XIDs with their own workflow resolvers or companion flows in the engine.
WORKFLOW_XIDS = frozenset({45, 48, 63, 64, 74, 154, *range(144, 151)})
#: How long the product gets to show the decision (032 uses the same budget).
DETECTION_BUDGET_SECONDS = 420
#: Copies of the reviewed line the probe appends (the collector reopens the
#: fixture at EOF); they dedup to one event on the stable sequence number.
EMISSIONS = 6
#: How long the fixture is left armed before the oracle is read back.
SETTLE_SECONDS = 90
MONITOR_DISPOSITION = "MONITOR_ONLY"
NO_ACTION = "NO_ACTION"
RECOVERED = "RECOVERED"

_PCI_BDF = re.compile(
    r"^(?:(?P<domain>[0-9a-fA-F]{4}|[0-9a-fA-F]{8}):)?"
    r"(?P<bus>[0-9a-fA-F]{2}):(?P<device>[0-9a-fA-F]{2})(?:\.(?P<function>[0-7]))?$"
)

#: The product's own XID decisions for one node since a moment: the event as
#: ingested (GPU identity, product, source) and the ``FaultPolicyDecision`` the
#: engine recorded for it, plus the state of the incident it names. Read-only.
XID_DECISION_PROBE = r"""
import json
import sys
from datetime import datetime

from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError

cluster_id, node_id, since_text = sys.argv[1:4]
since = datetime.fromisoformat(since_text.replace("Z", "+00:00"))
store = ApplicationContext.from_environment().store


def plain(value):
    return getattr(value, "value", value) if value is not None else None


decisions = []
for event in store.list_xid_events(cluster_id, node_id, observed_after=since):
    entry = {
        "event_id": event.event_id,
        "xid": event.xid,
        "gpu_uuid": event.gpu_uuid,
        "pci_bdf": event.pci_bdf,
        "event_source": plain(event.event_source),
        "product": event.product,
        "observed_at": event.observed_at.isoformat(),
        "source_boot_id": event.source_boot_id,
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
            "severity": plain(decision.severity),
            "requires_operator": decision.requires_operator,
            "incident_id": decision.incident_id,
            "incident_state": incident_state,
            "workflow_request_id": decision.workflow_request_id,
            "investigatory_notification_id": decision.investigatory_notification_id,
            "duplicate": decision.duplicate,
            "correlated_event_id": decision.correlated_event_id,
            "reasons": list(decision.reasons),
        })
    decisions.append(entry)
print(json.dumps({"decisions": decisions}, sort_keys=True, default=str))
"""


# --------------------------------------------------------------------------- #
# The pinned catalog: which XIDs are monitor grade
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CatalogEntry:
    """One row of the pinned NVIDIA XID catalog, as the engine reads it."""

    xid: int
    mnemonic: str
    immediate_action: str | None
    investigatory_action: str | None
    products: tuple[str, ...]
    catalog_version: str


def catalog_entry(xid: int) -> CatalogEntry | None:
    policy = load_xid_policy()
    for rule in policy.catalog_rules:
        if rule.xid == xid:
            return CatalogEntry(
                xid=rule.xid,
                mnemonic=str(rule.mnemonic or ""),
                immediate_action=rule.immediate_action,
                investigatory_action=rule.investigatory_action,
                products=tuple(rule.products),
                catalog_version=policy.catalog_version,
            )
    return None


def product_family(product: str | None) -> str | None:
    """The catalog family of a GPU product name (``NVIDIA H200`` -> ``H100``)."""

    return ProductFamilyResolver(load_xid_policy().product_families).family(product)


def monitor_grade_errors(
    entry: CatalogEntry | None, *, xid: int, family: str = SITE_FAMILY
) -> list[str]:
    """Why ``xid`` may not be replayed live beside another fault (empty: it may)."""

    if entry is None:
        return [
            f"XID {xid} is absent from the pinned catalog (it would be SITE_SAFETY)"
        ]
    errors = []
    if entry.immediate_action not in MONITOR_GRADE_ACTIONS:
        errors.append(
            f"XID {xid} ({entry.mnemonic}) Immediate Action is "
            f"{entry.immediate_action}; only {sorted(MONITOR_GRADE_ACTIONS)} is "
            "monitor grade live"
        )
    if family not in entry.products:
        errors.append(
            f"XID {xid} does not apply to the {family} family ({list(entry.products)});"
            " the product gate would quarantine the node"
        )
    if xid in LINK_XIDS:
        errors.append(f"XID {xid} joins the PCIe replay composite; not independent")
    if xid in WORKFLOW_XIDS:
        errors.append(f"XID {xid} has its own workflow or companion resolver")
    return errors


def require_monitor_grade(xid: int) -> CatalogEntry:
    """The catalog entry of ``xid``, or NOT_RUN before any mutation."""

    entry = catalog_entry(xid)
    errors = monitor_grade_errors(entry, xid=xid)
    if errors or entry is None:
        raise VariantNotRun("FIXTURE_NOT_READY", "; ".join(errors))
    return entry


# --------------------------------------------------------------------------- #
# The reviewed line
# --------------------------------------------------------------------------- #
def kmsg_pci_bdf(value: str | None) -> str | None:
    """The GPU's BDF as an ``NVRM: Xid (PCI:...)`` line prints it.

    The exporter and nvidia-smi print ``00000000:1B:00.0``; the kernel prints
    ``0000:1b:00`` (four-digit domain, no function). The product matches the
    two by numeric domain/bus/device (``gpu_composites.pci_bdf_matches``).
    """

    if not value:
        return None
    match = _PCI_BDF.fullmatch(value.strip())
    if match is None:
        return None
    domain = (match.group("domain") or "0000")[-4:]
    return f"{domain}:{match.group('bus')}:{match.group('device')}".lower()


def xid_message(xid: int, pci_bdf: str, *, run_id: str, label: str) -> str:
    """The reviewed kernel line; names the run so evidence can tell runs apart."""

    if ":" not in pci_bdf:
        raise ValueError("a kernel XID line needs the GPU's PCI BDF")
    return (
        f"NVRM: Xid (PCI:{pci_bdf}): {xid}, pid=0, name=gfrs-live, "
        f"{label} {run_id} reviewed kernel-source replay"
    )


# --------------------------------------------------------------------------- #
# Inject and read back
# --------------------------------------------------------------------------- #
@dataclass
class KernelXidInjection:
    """One armed reviewed XID: the replay window and what was installed."""

    replay: KernelLogReplay
    xid: int
    pci_bdf: str
    message: str
    started: datetime
    records_sha256: str
    closed: bool = False

    def close(self) -> dict[str, Any]:
        """Restore the kernel collector's real /dev/kmsg source now.

        The probe Pod stays on the variant's cleanup stack; closing early keeps
        the window in which the node's real kernel log is unread as short as
        the decision wait.
        """

        if self.closed:
            return {}
        receipt = self.replay.close()
        self.closed = True
        if not receipt.get("clean", True):
            raise RuntimeError(f"kernel replay window did not close clean: {receipt}")
        return receipt

    def evidence(self) -> dict[str, Any]:
        return {
            **self.replay.evidence(),
            "xid": self.xid,
            "pci_bdf": self.pci_bdf,
            "message": self.message,
            "started": self.started.isoformat(),
        }


def inject_kernel_xid(
    ctx: VariantContext,
    *,
    xid: int,
    pci_bdf: str,
    label: str,
    emissions: int = EMISSIONS,
    settle_seconds: int = SETTLE_SECONDS,
    require: Callable[[int], CatalogEntry] = require_monitor_grade,
) -> KernelXidInjection:
    """Deliver one reviewed XID through the kernel source.

    ``require`` is the catalog gate, judged before the node is touched: the
    default accepts only a monitor-grade code (NOT_RUN otherwise); the 029
    hardware-chain lane passes its reset-grade gate instead, under its own
    approvals. INCONCLUSIVE when the fixture file received no reviewed line.
    The window's cleanup is on ``ctx.cleanup``.
    """

    require(xid)
    message = xid_message(xid, pci_bdf, run_id=ctx.run_id, label=label)
    return inject_reviewed_kernel_line(
        ctx,
        xid=xid,
        pci_bdf=pci_bdf,
        message=message,
        label=label,
        emissions=emissions,
        settle_seconds=settle_seconds,
    )


def inject_reviewed_kernel_line(
    ctx: VariantContext,
    *,
    xid: int,
    pci_bdf: str,
    message: str,
    label: str,
    emissions: int = EMISSIONS,
    settle_seconds: int = SETTLE_SECONDS,
) -> KernelXidInjection:
    """Deliver one reviewed kernel line, exactly as given, through the kernel source.

    No catalog grade is checked here: the caller owns the live-safety argument
    for ``message``. ``inject_kernel_xid`` is the monitor-grade caller; 032
    ``missing_version`` first proves the decoder's version evidence is withheld
    inside the kernel collector, so its WORKFLOW_NVLINK5_ERR line can only be
    BLOCKED. INCONCLUSIVE when the fixture file received no reviewed line. The
    window's cleanup is on ``ctx.cleanup``.
    """

    if ":" not in pci_bdf:
        raise ValueError("a kernel XID line needs the GPU's PCI BDF")
    if f": {xid}," not in message:
        raise ValueError(f"the reviewed line does not carry XID {xid}")
    records = [
        reviewed_record(message, transport="kernel", emissions=emissions, label=label)
    ]
    replay = kernel_replay(ctx, mode="kmsg")
    replay.create()
    installed = replay.install(records)
    restore_seconds = min(max(60, ctx.spec.max_minutes * 60), 5400)
    opened = replay.open(approved=True, restore_seconds=restore_seconds)
    ctx.evidence.write(
        f"kernel-xid-{xid}-open", {"installed": installed, "opened": opened}
    )
    started = utc_now()
    armed = replay.arm()
    ctx.evidence.write(f"kernel-xid-{xid}-arm", armed)
    ctx.hold(settle_seconds, name=f"kernel-xid-{xid}-inject")
    oracle = replay.read_back()
    injected = dict(oracle.get("injected") or {})
    ctx.evidence.append_oracle([{"xid": xid, "injected": injected}], source="kmsg")
    if not injected.get("fixture_lines"):
        raise VariantInconclusive(
            f"the kmsg fixture received no reviewed XID {xid} record"
        )
    return KernelXidInjection(
        replay=replay,
        xid=xid,
        pci_bdf=pci_bdf,
        message=message,
        started=started,
        records_sha256=records_sha256(records),
    )


def xid_decisions(
    ctx: VariantContext, since: datetime, *, xid: int
) -> list[dict[str, Any]]:
    """The product's decisions for ``xid`` on the target node since ``since``."""

    value = ctx.regional.cpu_python(
        XID_DECISION_PROBE,
        ctx.settings.regional.cluster_id,
        ctx.settings.node,
        since.isoformat(),
        timeout=120,
    )
    return [item for item in value.get("decisions") or [] if item.get("xid") == xid]


def wait_xid_decisions(
    ctx: VariantContext,
    since: datetime,
    *,
    xid: int,
    name: str,
    timeout_seconds: float = DETECTION_BUDGET_SECONDS,
) -> list[dict[str, Any]]:
    """Poll until ``xid`` has a decision (not just an event), or the budget ends."""

    last: list[dict[str, Any]] = []

    def accept() -> list[dict[str, Any]] | None:
        nonlocal last
        last = xid_decisions(ctx, since, xid=xid)
        return last if any(item.get("decision") for item in last) else None

    value = ctx.wait_until(
        accept, timeout_seconds=timeout_seconds, poll_seconds=15, name=name
    )
    decisions = value if value is not None else last
    ctx.evidence.write(f"kernel-xid-{xid}-decisions", {"decisions": decisions})
    return decisions


# --------------------------------------------------------------------------- #
# Grading: the XID kept its own catalog handling
# --------------------------------------------------------------------------- #
def independence_errors(
    decisions: list[dict[str, Any]],
    *,
    entry: CatalogEntry,
    gpu_uuid: str | None,
    foreign_incident_ids: set[str],
) -> list[str]:
    """Why the reviewed XID was not handled independently (empty: it was).

    The product's own records must show one fresh decision that carries the
    catalog's action for the code, monitor-only with no action, no safety
    action and no operator, bound to its own RECOVERED incident -- not the
    incident of the concurrent counter or composite fault -- with no workflow,
    no correlation and the target GPU's identity.
    """

    xid = entry.xid
    if not decisions:
        return [f"XID {xid} from the trusted kernel source reached no decision"]
    graded = [item for item in decisions if item.get("decision")]
    if not graded:
        return [f"XID {xid} was ingested but has no FaultPolicyDecision"]
    fresh = [item for item in graded if not item.get("duplicate")]
    if not fresh:
        return [
            f"every decision for XID {xid} is a duplicate of an earlier event "
            "(the reviewed line re-used an event id; re-run on the same boot?)"
        ]
    errors: list[str] = []
    if len(fresh) > 1:
        errors.append(
            f"{len(fresh)} non-duplicate decisions for XID {xid}; the reviewed "
            "record's copies did not dedup to one event"
        )
    item = fresh[0]
    if item.get("event_source") not in (None, "KERNEL_LOG"):
        errors.append(
            f"XID {xid} arrived as {item.get('event_source')}, not the trusted "
            "KERNEL_LOG source"
        )
    if item.get("disposition") != MONITOR_DISPOSITION:
        errors.append(
            f"XID {xid} disposition is {item.get('disposition')}, expected "
            f"{MONITOR_DISPOSITION} for catalog action {entry.immediate_action}"
        )
    if item.get("action") not in (None, NO_ACTION):
        errors.append(f"XID {xid} earned recovery action {item.get('action')}")
    if item.get("safety_action"):
        errors.append(f"XID {xid} earned safety action {item.get('safety_action')}")
    if item.get("official_action") != entry.immediate_action:
        errors.append(
            f"XID {xid} official action is {item.get('official_action')}, the pinned "
            f"catalog {entry.catalog_version} says {entry.immediate_action}"
        )
    if item.get("requires_operator"):
        errors.append(f"XID {xid} was handed to an operator")
    if item.get("correlated_event_id"):
        errors.append(
            f"XID {xid} was correlated into event {item.get('correlated_event_id')}"
        )
    incident_id = item.get("incident_id")
    if not incident_id:
        errors.append(f"XID {xid} decision names no incident of its own")
    elif incident_id in foreign_incident_ids:
        errors.append(
            f"XID {xid} was absorbed into the concurrent fault's incident {incident_id}"
        )
    elif item.get("incident_state") not in (None, RECOVERED):
        errors.append(
            f"XID {xid} incident {incident_id} is {item.get('incident_state')}; a "
            "monitor-only decision closes RECOVERED on creation"
        )
    if item.get("workflow_request_id"):
        errors.append(
            f"XID {xid} minted workflow {item.get('workflow_request_id')} "
            "(monitor grade has none)"
        )
    if gpu_uuid and item.get("gpu_uuid") != gpu_uuid:
        errors.append(
            f"XID {xid} resolved to GPU {item.get('gpu_uuid')}, expected the target "
            f"GPU {gpu_uuid} named by its PCI BDF"
        )
    return errors


def isolating_decisions(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Decisions that would isolate or mutate the node (none is expected)."""

    return [
        item
        for item in decisions
        if item.get("decision")
        and (
            item.get("safety_action")
            or item.get("action") not in (None, NO_ACTION)
            or item.get("workflow_request_id")
        )
    ]


def restore_if_isolated(
    ctx: VariantContext, decisions: list[dict[str, Any]]
) -> list[str]:
    """Hand an unexpectedly isolating XID incident back through the validated restore.

    A monitor-grade XID never isolates; if the product did, the variant has
    already failed (``independence_errors``) and this keeps the node from
    being left cordoned. Errors here are cleanup errors.
    """

    isolating = isolating_decisions(decisions)
    incidents = [
        {"incident_id": item["incident_id"]}
        for item in isolating
        if item.get("incident_id") and item.get("incident_state") != RECOVERED
    ]
    if not incidents:
        return []
    from scripts.e2e.regional.general_live_common import restore_incidents

    try:
        restore_incidents(
            ctx, incidents, reason=f"{ctx.spec.key} kernel XID acceptance restore"
        )
    except Exception as exc:  # noqa: BLE001 - recorded as a cleanup error
        return [f"validated restore failed: {type(exc).__name__}: {exc}"]
    return []


__all__ = [
    "DETECTION_BUDGET_SECONDS",
    "EMISSIONS",
    "LINK_XIDS",
    "MONITOR_GRADE_ACTIONS",
    "SITE_FAMILY",
    "WORKFLOW_XIDS",
    "XID_DECISION_PROBE",
    "CatalogEntry",
    "KernelXidInjection",
    "catalog_entry",
    "independence_errors",
    "inject_kernel_xid",
    "inject_reviewed_kernel_line",
    "isolating_decisions",
    "kmsg_pci_bdf",
    "monitor_grade_errors",
    "product_family",
    "require_monitor_grade",
    "restore_if_isolated",
    "wait_xid_decisions",
    "xid_decisions",
    "xid_message",
]
