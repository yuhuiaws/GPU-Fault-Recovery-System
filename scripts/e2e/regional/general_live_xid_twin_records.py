"""The product side of GF-GENERAL-LIVE-024 ``xid_twin_*``: which generation
the control plane runs, what it recorded for the two reports of one XID, and
the oracles that judge those records.

Batch 237 (``gpu_fault.policy.xid_cross_source``) makes the kernel NVRM line
and the DCGM field-230 change of one XID on one GPU one occurrence: the second
report to arrive -- or, for an XID whose decode reads the kernel line (154,
74, 144-150), always the change -- names the first one's event in
``correlated_event_id``, is flagged ``duplicate`` and is linked to the first
one's incident, which runs at most one remediation. Everything here reads the
product's durable records through the installed control-plane interpreter of
an API Pod (``RegionalLiveFixture.cpu_python``): the XID events, their saved
``FaultPolicyDecision`` and the incident each event is linked to
(``XID_TWIN_RECORDS_PROBE``); the workflows with their step executions and the
remote commands with their receipts come from the node activity probe
(``general_live_store.ACTIVITY_PROBE``). Nothing writes to the store, and no
transient state (a node annotation, an in-memory queue) decides a count.

``XID_TWIN_GENERATION_PROBE`` asks the deployed control plane, before anything
is deployed on the node, whether it has the twin rule at all: it imports the
product's own modules and evaluates two in-memory reports of the variant's XID
with the product's own policy engine -- the store is never read or written. A
product without batch 237 decides a field-230 change as a fault of its own;
there the variants end INCONCLUSIVE with the generation named, never FAIL, and
never with a twin injected (it would be a second remediation on the live
node). The gate follows S47 (014's freeze path) and S54.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
)
from scripts.e2e.regional.general_live_verdicts import (
    TERMINAL_WORKFLOWS,
    final_execution,
    operations,
    parse_time,
    subsequence,
)

CASE = "GF-GENERAL-LIVE-024"
KERNEL_SOURCE = "KERNEL_LOG"
GAUGE_SOURCE = "DCGM_EXPORTER"
#: The DCGM last-XID gauge (field 230, ``DCGM_FI_DEV_XID_ERRORS``), canonical.
GAUGE = "xid_last_error"
#: ``policy/xid_cross_source.XID_REPORT_TWIN_WINDOW``: how far apart the kernel
#: line and the field-230 change of one XID may be (pinned by test).
TWIN_WINDOW_SECONDS = 120
#: XIDs whose decode reads the kernel line (catalog actions XID_154,
#: WORKFLOW_NVLINK_ERR, WORKFLOW_NVLINK5_ERR): their change waits for the line
#: and always names it, whichever arrived first
#: (``xid_cross_source.KERNEL_DECODED_ACTIONS``).
KERNEL_DECODED_XIDS = frozenset({154, 74, *range(144, 151)})
GENERATION_EVIDENCE = "product-generation"
#: The named reason of the product-generation INCONCLUSIVE.
PRE_TWIN_GENERATION = "expected on this product generation"
#: The reason text a marker association adds to a decision
#: (``orchestration/provider_correlation.bind_marker_incident``).
MARKER_ASSOCIATION = "Provider marker association with incident "
#: The operations an X variant counts across every workflow of the node.
WATCHED_OPERATIONS = (
    "STOP_WORKLOADS",
    "QUIESCE_GPU_SERVICES",
    "RESET_GPU",
    "RECOVER_DRAIN_P2P",
    "RESTART_WORKLOAD",
    "QUARANTINE",
)


@dataclass(frozen=True)
class TwinExpectation:
    """What the product's catalog decision and its one remediation must be.

    ``executed`` is the number of SUCCEEDED step executions each named
    operation must have across every workflow of the node since the run began;
    ``absent`` operations must have none; ``chain`` is the ordered subsequence
    the one workflow must plan; ``forbidden`` is planned by no workflow of the
    node.
    """

    xid: int
    disposition: str
    official_action: str
    actions: frozenset[str | None]
    executed: dict[str, int] = field(default_factory=dict)
    absent: frozenset[str] = frozenset()
    chain: tuple[str, ...] = ()
    forbidden: frozenset[str] = frozenset()
    #: The hardware operations the variant's approvals let the chain plan.
    hardware: frozenset[str] = frozenset()

    @property
    def remediates(self) -> bool:
        return bool(self.chain)


# --------------------------------------------------------------------------- #
# The product generation (code only; the store is never touched)
# --------------------------------------------------------------------------- #
#: Read-only, in an API Pod: does the deployed control plane carry batch 237's
#: twin rule, and does its own policy engine treat a field-230 change with a
#: kernel line of the same XID on the same GPU, 20 s apart, as that line's
#: report? Two in-memory XidEvents (the kernel spelling ``0000:b9:00`` and the
#: exporter's ``00000000:B9:00.0``), fresh engines, no store, no network.
#: ``argv``: the XID and the reviewed kernel text after ``<xid>,``.
XID_TWIN_GENERATION_PROBE = r"""
import json
import sys
from datetime import datetime, timedelta, timezone

xid, tail = int(sys.argv[1]), sys.argv[2]
result = {"xid": xid, "present": False}
try:
    from gpu_fault.policy import xid_cross_source as twins
except ImportError as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        import gpu_fault.policy.nvlink5_gauge  # noqa: F401
    except ImportError:
        result["nvlink5_gauge"] = False
    else:
        result["nvlink5_gauge"] = True
    print(json.dumps(result, sort_keys=True))
    sys.exit(0)


def plain(value):
    return getattr(value, "value", value) if value is not None else None


def view(decision):
    return {
        "disposition": plain(decision.disposition),
        "official_action": decision.official_action,
        "action": plain(decision.action),
        "safety_action": plain(decision.safety_action),
        "pre_actions": [plain(item) for item in decision.pre_actions],
        "correlated_event_id": decision.correlated_event_id,
    }


try:
    from gpu_fault.app.ingest.faults import FaultIngestionService
    from gpu_fault.orchestration import provider_correlation
    from gpu_fault.policy import GpuFaultPolicyEngine, XidEvent, parse_xid154_action

    window = twins.XID_REPORT_TWIN_WINDOW.total_seconds()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    gpu = "GPU-00000000-0000-4000-8000-0000000b2370"
    message = f"NVRM: Xid (PCI:0000:b9:00): {xid}, {tail}"
    common = {
        "cluster_id": "generation-probe",
        "node_id": "generation-probe-node",
        "xid": xid,
        "gpu_uuid": gpu,
        "product": "NVIDIA H200",
        "driver_branch": 580,
        "cuda_version": "13.0",
        "runtime_profile_version": "generation-probe",
    }
    kernel = XidEvent(
        event_id="generation-probe-kernel",
        observed_at=base,
        event_source=twins.KERNEL_XID_SOURCE,
        pci_bdf="0000:b9:00",
        raw_message=message,
        source_boot_id="generation-probe-boot",
        xid_154_action=parse_xid154_action(message) if xid == 154 else None,
        **common,
    )
    gauge = XidEvent(
        event_id="generation-probe-gauge",
        observed_at=base + timedelta(seconds=20),
        event_source=twins.DCGM_XID_GAUGE_SOURCE,
        pci_bdf="00000000:B9:00.0",
        raw_message=f"DCGM_FI_DEV_XID_ERRORS={xid} gpu={gpu}",
        **common,
    )
    late = gauge.model_copy(
        update={
            "event_id": "generation-probe-late",
            "observed_at": base + timedelta(seconds=window + 1),
        }
    )

    def decide(event, companions, *, closed):
        return GpuFaultPolicyEngine().evaluate_xid(
            event,
            companion_events=companions,
            xid45_window_closed=True,
            xid154_window_closed=True,
            paired_report_ids=frozenset(),
            report_twin_window_closed=closed,
        )

    primary = decide(kernel, [], closed=True)
    followed = decide(gauge, [kernel], closed=False)
    alone = decide(
        gauge.model_copy(update={"event_id": "generation-probe-alone"}), [], closed=False
    )
    result.update(
        present=True,
        twin_window_seconds=window,
        twins_in_window=bool(twins.report_twins(kernel, gauge)),
        late_twin=bool(twins.report_twins(kernel, late)),
        gauge_follows_kernel=followed.correlated_event_id == kernel.event_id,
        same_decision=bool(twins.same_report_decision(primary, followed)),
        primary=view(primary),
        followed=view(followed),
        alone=view(alone),
        kernel_side_join=callable(getattr(twins, "join_gauge_twin", None))
        and callable(getattr(FaultIngestionService, "_join_gauge_twin", None)),
        incident_join=callable(getattr(provider_correlation, "report_twin_incident", None)),
    )
except Exception as exc:
    result["present"] = False
    result["error"] = f"{type(exc).__name__}: {exc}"[:300]
print(json.dumps(result, sort_keys=True, default=str))
"""


@dataclass(frozen=True)
class TwinGeneration:
    """The control plane's answer to ``XID_TWIN_GENERATION_PROBE``."""

    xid: int
    control_plane: dict[str, Any]

    @property
    def present(self) -> bool:
        return self.control_plane.get("present") is True

    @property
    def predates_twins(self) -> bool:
        """No ``xid_cross_source`` module at all: before batch 237."""

        error = str(self.control_plane.get("error") or "")
        return not self.present and "xid_cross_source" in error

    def shortfalls(self) -> list[str]:
        """What of batch 237's rule the deployed engine does not show."""

        answer = self.control_plane
        if not self.present:
            return [str(answer.get("error") or "the probe answered nothing")]
        problems = []
        if answer.get("twin_window_seconds") != TWIN_WINDOW_SECONDS:
            problems.append(
                f"twin window {answer.get('twin_window_seconds')} s, not "
                f"{TWIN_WINDOW_SECONDS} s"
            )
        if answer.get("twins_in_window") is not True:
            problems.append("a kernel line and its change 20 s apart are no twins")
        if answer.get("late_twin") is not False:
            problems.append("a change beyond the window still pairs")
        if answer.get("gauge_follows_kernel") is not True:
            problems.append("the change does not follow its kernel line")
        if answer.get("same_decision") is not True:
            problems.append("the change is not decided the way its line is")
        if answer.get("kernel_side_join") is not True:
            problems.append("no kernel-side join of a change decided first")
        if answer.get("incident_join") is not True:
            problems.append("no incident link of the second report")
        if (
            self.xid in KERNEL_DECODED_XIDS
            and (answer.get("alone") or {}).get("disposition") != "PENDING_CORRELATION"
        ):
            problems.append(
                f"a change of XID {self.xid} alone does not wait for its kernel line"
            )
        return problems

    @property
    def twin_rule(self) -> bool:
        return self.present and not self.shortfalls()

    def describe(self) -> str:
        if self.twin_rule:
            primary = self.control_plane.get("primary") or {}
            return (
                "the control plane carries batch 237's twin rule (twin window "
                f"{TWIN_WINDOW_SECONDS} s; its engine has the change of XID {self.xid} "
                "follow the kernel line, decided "
                f"{primary.get('disposition')} {primary.get('official_action')})"
            )
        if self.predates_twins:
            earlier = (
                "; batch 235's policy/nvlink5_gauge pairs XIDs 144-150 only"
                if self.control_plane.get("nvlink5_gauge")
                else ""
            )
            return (
                "the control plane predates batch 237: gpu_fault.policy."
                f"xid_cross_source is absent{earlier}"
            )
        return "the control plane's twin rule is not batch 237's: " + "; ".join(
            self.shortfalls()
        )

    def record(self) -> dict[str, Any]:
        return {
            "xid": self.xid,
            "present": self.present,
            "twin_rule": self.twin_rule,
            "predates_twins": self.predates_twins,
            "shortfalls": [] if self.twin_rule else self.shortfalls(),
            "control_plane": dict(self.control_plane),
            "description": self.describe(),
        }


def twin_generation(ctx: VariantContext, xid: int, tail: str) -> TwinGeneration:
    """Ask the deployed control plane; the answer is ``product-generation``."""

    answer = ctx.regional.cpu_python(XID_TWIN_GENERATION_PROBE, str(xid), tail)
    value = TwinGeneration(xid, dict(answer) if isinstance(answer, dict) else {})
    ctx.evidence.write(GENERATION_EVIDENCE, value.record())
    return value


def require_twin_generation(
    ctx: VariantContext,
    expectation: TwinExpectation,
    tail: str,
    *,
    consequence: str,
) -> TwinGeneration:
    """INCONCLUSIVE before anything is deployed unless the product dedupes.

    A product without the twin rule, or whose engine decides the XID other
    than the variant expects, is never handed the twin: ``consequence`` says
    what it would do with it (a second incident, a second reset, a
    quarantine). The answer is the evidence ``product-generation``.
    """

    generation = twin_generation(ctx, expectation.xid, tail)
    untouched = "Nothing was deployed and the node was not touched."
    if not generation.twin_rule:
        raise VariantInconclusive(
            f"{PRE_TWIN_GENERATION}: {generation.describe()}. Its field-230 change "
            f"of XID {expectation.xid} would be decided as a fault of its own -- "
            f"{consequence} -- the defect batch 237 fixes, not a fault of this "
            f"variant. {untouched}"
        )
    primary = generation.control_plane.get("primary") or {}
    if (
        primary.get("disposition") != expectation.disposition
        or primary.get("official_action") != expectation.official_action
        or primary.get("action") not in expectation.actions
    ):
        raise VariantInconclusive(
            f"the deployed engine decides XID {expectation.xid} "
            f"{primary.get('disposition')} {primary.get('official_action')} "
            f"(action {primary.get('action')}), not the variant's "
            f"{expectation.disposition} {expectation.official_action}; the twin "
            f"would run another chain than the one judged here. {untouched}"
        )
    return generation


# --------------------------------------------------------------------------- #
# The product's records of the two reports
# --------------------------------------------------------------------------- #
#: Read-only: every XID event of one code on one node since a moment -- source,
#: GPU, the three timestamps, the saved FaultPolicyDecision and the incident
#: the store links the event to (``get_incident_by_event``) -- and those
#: incidents with their markers; the other codes' events by id only.
XID_TWIN_RECORDS_PROBE = r"""
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


def stamp(value):
    return value.isoformat() if value is not None else None


events = []
others = []
incident_ids = set()
for event in store.list_xid_events(cluster_id, node_id, observed_after=since):
    if event.xid != wanted:
        others.append({
            "event_id": event.event_id,
            "xid": event.xid,
            "event_source": plain(event.event_source),
            "observed_at": stamp(event.observed_at),
        })
        continue
    linked = store.get_incident_by_event(event.event_id)
    entry = {
        "event_id": event.event_id,
        "xid": event.xid,
        "event_source": plain(event.event_source),
        "gpu_uuid": event.gpu_uuid,
        "pci_bdf": event.pci_bdf,
        "observed_at": stamp(event.observed_at),
        "collected_at": stamp(event.collected_at),
        "ingested_at": stamp(event.ingested_at),
        "source_boot_id": event.source_boot_id,
        "workload_state": plain(event.workload_state),
        "job_id": event.job_id,
        "attempt_id": event.attempt_id,
        "affected_workload_ids": list(event.affected_workload_ids),
        "raw_message": (event.raw_message or "")[:300],
        "linked_incident_id": linked.incident_id if linked is not None else None,
        "decision": None,
    }
    if linked is not None:
        incident_ids.add(linked.incident_id)
    decision = store.get_xid_policy_decision(event.event_id)
    if decision is not None:
        entry["decision"] = {
            "disposition": plain(decision.disposition),
            "official_action": decision.official_action,
            "investigatory_action": decision.investigatory_action,
            "action": plain(decision.action),
            "safety_action": plain(decision.safety_action),
            "pre_actions": [plain(item) for item in decision.pre_actions],
            "severity": plain(decision.severity),
            "requires_operator": decision.requires_operator,
            "incident_id": decision.incident_id,
            "workflow_request_id": decision.workflow_request_id,
            "duplicate": decision.duplicate,
            "correlated_event_id": decision.correlated_event_id,
            "marker_incident_id": decision.marker.incident_id,
            "marker_active": decision.marker.active,
            "reasons": [str(item)[:400] for item in decision.reasons[-12:]],
        }
        if decision.incident_id:
            incident_ids.add(decision.incident_id)
    events.append(entry)
incidents = []
for incident_id in sorted(incident_ids):
    try:
        incident = store.get_incident(incident_id)
    except NotFoundError:
        incidents.append({"incident_id": incident_id, "missing": True})
        continue
    incidents.append({
        "incident_id": incident.incident_id,
        "event_id": incident.event_id,
        "event_type": incident.event_type,
        "state": plain(incident.state),
        "workflow_request_id": incident.workflow_request_id,
        "official_action": incident.official_action,
        "effective_action": plain(incident.effective_action),
        "safety_action": plain(incident.safety_action),
        "gpu_uuids": list(incident.gpu_uuids),
        "node_ids": list(incident.node_ids),
        "job_id": incident.job_id,
        "attempt_id": incident.attempt_id,
        "created_at": stamp(incident.created_at),
        "updated_at": stamp(incident.updated_at),
        "reasons": [str(item)[:400] for item in incident.reasons[-8:]],
        "markers": [
            {
                "marker_id": item.marker_id,
                "active": item.active,
                "event_source": plain(item.event_source),
            }
            for item in store.list_markers_for_incident(incident.incident_id)
        ],
    })
events.sort(key=lambda item: str(item["observed_at"]))
print(json.dumps({
    "since": since.isoformat(),
    "cluster_id": cluster_id,
    "node_id": node_id,
    "xid": wanted,
    "events": events,
    "incidents": incidents,
    "other_xid_events": others[-50:],
}, sort_keys=True, default=str))
"""


def twin_records(ctx: VariantContext, since: datetime, xid: int) -> dict[str, Any]:
    """The product's records of every report of ``xid`` on the node since ``since``."""

    return ctx.regional.cpu_python(
        XID_TWIN_RECORDS_PROBE,
        ctx.settings.regional.cluster_id,
        ctx.settings.node,
        since.isoformat(),
        str(xid),
        timeout=120,
    )


def reports(records: dict[str, Any], *, gpu: str, source: str) -> list[dict[str, Any]]:
    """The events of one source on the target GPU, oldest first."""

    return [
        item
        for item in records.get("events") or []
        if item.get("event_source") == source and item.get("gpu_uuid") == gpu
    ]


def decided(event: dict[str, Any]) -> bool:
    """A saved decision that is final (not still waiting for a companion)."""

    decision = event.get("decision") or {}
    return bool(decision) and decision.get("disposition") != "PENDING_CORRELATION"


def incident_of(records: dict[str, Any], incident_id: str | None) -> dict[str, Any]:
    for item in records.get("incidents") or []:
        if incident_id and item.get("incident_id") == incident_id:
            return item
    return {}


def marker_associations(decision: dict[str, Any]) -> list[str]:
    """The incidents a marker association named in the decision's reasons."""

    found = []
    for reason in decision.get("reasons") or []:
        text = str(reason)
        if MARKER_ASSOCIATION in text:
            found.append(text.split(MARKER_ASSOCIATION, 1)[1].split(";", 1)[0].strip())
    return found


def seconds_between(later: Any, earlier: Any) -> float | None:
    """``later - earlier`` in seconds for two ISO timestamps (``None`` if not both)."""

    try:
        end, start = parse_time(later), parse_time(earlier)
    except ValueError:
        return None
    if end is None or start is None:
        return None
    return (end - start).total_seconds()


@dataclass(frozen=True)
class TwinPair:
    """The two reports of one occurrence, as the product recorded them."""

    first: dict[str, Any]
    second: dict[str, Any]
    #: The report that must name the other (the joiner): the later one, or the
    #: change for a kernel-decoded XID.
    joiner: dict[str, Any]
    primary: dict[str, Any]

    @property
    def gap_seconds(self) -> float | None:
        return seconds_between(
            self.second.get("observed_at"), self.first.get("observed_at")
        )


def twin_pair(
    records: dict[str, Any], *, gpu: str, first_source: str, xid: int
) -> tuple[TwinPair | None, list[str]]:
    """The planned pair of reports on ``gpu``, or why there is none.

    Exactly one kernel line must lead (kernel-first) or follow the one change
    (gauge-first) as planned; a later repeat of the line (the negative
    control) is not part of the pair. The reasons are premise failures
    (INCONCLUSIVE), not product errors.
    """

    kernels = reports(records, gpu=gpu, source=KERNEL_SOURCE)
    gauges = reports(records, gpu=gpu, source=GAUGE_SOURCE)
    problems = []
    if not gauges:
        problems.append(
            f"no field-230 change of XID {xid} on {gpu} reached the product"
        )
    if not kernels:
        problems.append(f"no kernel line of XID {xid} on {gpu} reached the product")
    if problems:
        return None, problems
    if len(gauges) != 1:
        return None, [
            f"{len(gauges)} field-230 changes of XID {xid} on {gpu}; the plan served "
            "one (a real XID would have aborted the replay)"
        ]
    gauge = gauges[0]
    if first_source == KERNEL_SOURCE:
        kernel = kernels[0]
        first, second = kernel, gauge
    else:
        later = [
            item
            for item in kernels
            if (
                gap := seconds_between(
                    item.get("observed_at"), gauge.get("observed_at")
                )
            )
            is not None
            and gap >= 0
        ]
        if not later:
            return None, [
                "the kernel line was observed before the change it was meant to follow"
            ]
        kernel = later[0]
        first, second = gauge, kernel
    gap = seconds_between(second.get("observed_at"), first.get("observed_at"))
    if gap is None or gap < 0:
        return None, [
            f"the product's clock puts the {second.get('event_source')} report "
            f"before the {first.get('event_source')} one ({gap} s)"
        ]
    if gap > TWIN_WINDOW_SECONDS:
        return None, [
            f"the two reports are {gap:.0f} s apart by the product's own clock, "
            f"outside the {TWIN_WINDOW_SECONDS} s twin window"
        ]
    joiner = gauge if xid in KERNEL_DECODED_XIDS else second
    primary = kernel if joiner is gauge else gauge
    return TwinPair(first=first, second=second, joiner=joiner, primary=primary), []


def decision_class_errors(
    event: dict[str, Any], expectation: TwinExpectation, *, label: str
) -> list[str]:
    decision = event.get("decision") or {}
    if not decision:
        return [f"the {label} has no saved FaultPolicyDecision"]
    errors = []
    if decision.get("disposition") != expectation.disposition:
        errors.append(
            f"the {label} is {decision.get('disposition')}, expected "
            f"{expectation.disposition}"
        )
    if decision.get("official_action") != expectation.official_action:
        errors.append(
            f"the {label}'s official action is {decision.get('official_action')}, "
            f"expected {expectation.official_action}"
        )
    if decision.get("action") not in expectation.actions:
        errors.append(f"the {label} earned recovery action {decision.get('action')}")
    if decision.get("safety_action"):
        errors.append(
            f"the {label} earned safety action {decision.get('safety_action')}"
        )
    return errors


def twin_errors(
    records: dict[str, Any], pair: TwinPair, expectation: TwinExpectation
) -> dict[str, list[str]]:
    """The two reports are one occurrence: one incident, the joiner names the
    primary, both decided the catalog's way; stages ``primary``, ``twin``,
    ``one_incident``."""

    primary, joiner = pair.primary, pair.joiner
    primary_label = f"{primary.get('event_source')} report {primary.get('event_id')}"
    joiner_label = f"{joiner.get('event_source')} report {joiner.get('event_id')}"
    primary_decision = primary.get("decision") or {}
    joiner_decision = joiner.get("decision") or {}
    stages: dict[str, list[str]] = {
        "primary": decision_class_errors(primary, expectation, label=primary_label),
        "twin": decision_class_errors(joiner, expectation, label=joiner_label),
        "one_incident": [],
    }
    incident_id = primary_decision.get("incident_id")
    if primary_decision.get("correlated_event_id"):
        stages["primary"].append(
            f"the {primary_label} names {primary_decision.get('correlated_event_id')}: "
            "the report the occurrence is decided on names no other"
        )
    if not incident_id:
        stages["primary"].append(f"the {primary_label} names no incident")
    if primary.get("linked_incident_id") != incident_id:
        stages["primary"].append(
            f"the store links the {primary_label} to {primary.get('linked_incident_id')}"
            f", its decision names {incident_id}"
        )
    if joiner_decision.get("correlated_event_id") != primary.get("event_id"):
        stages["twin"].append(
            f"the {joiner_label} names {joiner_decision.get('correlated_event_id')} "
            f"in correlated_event_id, not its twin {primary.get('event_id')}"
        )
    if joiner_decision.get("duplicate") is not True:
        stages["twin"].append(f"the {joiner_label} is not flagged duplicate")
    if joiner_decision.get("workflow_request_id") not in (
        None,
        primary_decision.get("workflow_request_id"),
    ):
        stages["twin"].append(
            f"the {joiner_label} has workflow {joiner_decision.get('workflow_request_id')}"
            " of its own"
        )
    if joiner_decision.get("incident_id") != incident_id:
        stages["one_incident"].append(
            f"the two reports of one XID opened two incidents: {incident_id} and "
            f"{joiner_decision.get('incident_id')}"
        )
    if joiner.get("linked_incident_id") != incident_id:
        stages["one_incident"].append(
            f"the store links the {joiner_label} to {joiner.get('linked_incident_id')}"
            f", not to the occurrence's incident {incident_id}"
        )
    pair_ids = {str(primary.get("event_id")), str(joiner.get("event_id"))}
    stray = sorted(
        str(item.get("incident_id"))
        for item in records.get("incidents") or []
        if item.get("incident_id") != incident_id
        and str(item.get("event_id")) in pair_ids
    )
    if stray:
        stages["one_incident"].append(
            f"incident(s) {stray} were opened for a report of the pair as well"
        )
    if not incident_of(records, incident_id):
        stages["one_incident"].append(f"incident {incident_id} is not in the store")
    return stages


def repeat_errors(
    records: dict[str, Any],
    repeat_event_id: str | None,
    *,
    occurrence_incident_id: str | None,
    expectation: TwinExpectation,
) -> list[str]:
    """The repeat outside the twin window is a new occurrence of its own."""

    event = next(
        (
            item
            for item in records.get("events") or []
            if item.get("event_id") == repeat_event_id
        ),
        None,
    )
    if event is None:
        return [f"the repeated kernel line {repeat_event_id} is not in the store"]
    label = f"repeated kernel line {repeat_event_id}"
    errors = decision_class_errors(event, expectation, label=label)
    decision = event.get("decision") or {}
    own = f"inc-{repeat_event_id}"
    if decision.get("correlated_event_id"):
        errors.append(
            f"the {label} names {decision.get('correlated_event_id')}: a repeat "
            "outside the twin window is no report of the earlier occurrence"
        )
    if decision.get("incident_id") != own:
        errors.append(
            f"the {label} is on incident {decision.get('incident_id')}, not its own {own}"
        )
    if occurrence_incident_id and event.get("linked_incident_id") in (
        occurrence_incident_id,
    ):
        errors.append(f"the {label} was linked to the occurrence's incident")
    if decision.get("duplicate") and not marker_associations(decision):
        errors.append(
            f"the {label} is flagged duplicate with no marker association: the "
            "repeat was read as the earlier event"
        )
    if decision.get("workflow_request_id") and not expectation.remediates:
        errors.append(
            f"the {label} minted workflow {decision.get('workflow_request_id')}"
        )
    state = incident_of(records, decision.get("incident_id")).get("state")
    if not expectation.remediates and state not in ("RECOVERED",):
        errors.append(
            f"the {label}'s incident is {state}; a monitor-only decision closes "
            "RECOVERED on creation"
        )
    return errors


# --------------------------------------------------------------------------- #
# Workflows and executed operations (the node activity document)
# --------------------------------------------------------------------------- #
def executed_steps(activity: dict[str, Any]) -> dict[str, set[tuple[str, int]]]:
    """SUCCEEDED step executions per operation, as ``(workflow, step index)``,
    across every workflow the activity read lists for the node."""

    result: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for workflow in activity.get("workflows") or []:
        request_id = str(workflow.get("request_id"))
        for item in workflow.get("step_executions") or []:
            if item.get("status") != "SUCCEEDED":
                continue
            result[str(item.get("operation"))].add(
                (request_id, int(item.get("step_index") or 0))
            )
    return result


def command_receipts(activity: dict[str, Any]) -> dict[str, set[tuple[str, int]]]:
    """The remote commands' SUCCEEDED receipts per operation, the batched steps
    of a compound command included (``result_details.batched_results``)."""

    result: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for command in activity.get("commands") or []:
        request_id = str(command.get("workflow_request_id"))
        if command.get("status") == "SUCCEEDED":
            step = command.get("step") or {}
            result[str(step.get("operation"))].add(
                (request_id, int(command.get("step_index") or 0))
            )
        batched = (command.get("result_details") or {}).get("batched_results") or {}
        for entry in command.get("batched_steps") or []:
            index = int(entry.get("step_index") or 0)
            outcome = batched.get(str(index)) or {}
            if outcome.get("status") == "SUCCEEDED":
                operation = str((entry.get("step") or {}).get("operation"))
                result[operation].add((request_id, index))
    return result


def execution_errors(
    activity: dict[str, Any], expectation: TwinExpectation
) -> tuple[list[str], dict[str, Any]]:
    """Each named operation ran exactly as often as the one remediation needs
    (``executed``), an ``absent`` one not at all, across every workflow of the
    node, by the product's own step executions -- and no remote-command
    receipt shows a run they do not; returns the errors and the counts."""

    steps = executed_steps(activity)
    receipts = command_receipts(activity)
    errors = []
    counts: dict[str, Any] = {}
    for operation in sorted(set(expectation.executed) | expectation.absent):
        wanted = expectation.executed.get(operation, 0)
        found = sorted(steps.get(operation, set()))
        counts[operation] = {
            "steps": [list(item) for item in found],
            "receipts": [list(item) for item in sorted(receipts.get(operation, set()))],
        }
        if len(found) != wanted:
            errors.append(
                f"{operation} executed {len(found)} time(s) on the node "
                f"({', '.join(f'{wf}#{index}' for wf, index in found) or 'none'}), "
                f"expected {wanted}"
            )
        extra = sorted(receipts.get(operation, set()) - set(found))
        if extra:
            errors.append(
                f"{operation} has remote-command receipts {extra} beyond its step "
                "executions"
            )
    return errors, counts


def node_workflows(activity: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        activity.get("workflows") or [], key=lambda item: str(item.get("created_at"))
    )


def workflow_errors(
    activity: dict[str, Any],
    incident_id: str | None,
    expectation: TwinExpectation,
) -> tuple[list[str], dict[str, Any]]:
    """One workflow for the occurrence (none for a monitor-only XID), no other
    workflow on the node acts; returns the errors and the occurrence's workflow."""

    workflows = node_workflows(activity)
    mine = [item for item in workflows if item.get("incident_id") == incident_id]
    others = [item for item in workflows if item.get("incident_id") != incident_id]
    errors = []
    if not expectation.remediates:
        errors.extend(
            f"workflow {item.get('request_id')} of incident {item.get('incident_id')} "
            "was planned for a monitor-only XID"
            for item in workflows
        )
        return errors, {}
    if len(mine) != 1:
        errors.append(
            f"{len(mine)} workflows on the occurrence's incident {incident_id}, "
            "expected one"
        )
    acting = set(WATCHED_OPERATIONS) | set(expectation.hardware)
    for item in others:
        planned = sorted(set(operations(item)) & acting)
        if planned:
            errors.append(
                f"workflow {item.get('request_id')} of incident "
                f"{item.get('incident_id')} also plans {planned}"
            )
    for item in workflows:
        forbidden = sorted(set(operations(item)) & expectation.forbidden)
        if forbidden:
            errors.append(f"workflow {item.get('request_id')} plans {forbidden}")
    return errors, (mine[0] if mine else {})


def chain_errors(workflow: dict[str, Any], expectation: TwinExpectation) -> list[str]:
    """The occurrence's one remediation ran its chain to SUCCEEDED."""

    if not workflow:
        return ["the occurrence's remediation workflow is not in the product's records"]
    errors = []
    steps = operations(workflow)
    if not subsequence(expectation.chain, steps):
        errors.append(
            f"workflow steps {steps} lack the chain {list(expectation.chain)}"
        )
    status = str(workflow.get("status"))
    if status not in TERMINAL_WORKFLOWS:
        errors.append(f"the remediation is still {status} at the end of its budget")
    elif status != "SUCCEEDED":
        errors.append(
            f"the remediation ended {status}: "
            f"{str(workflow.get('terminal_failure_reason') or '')[:240]}"
        )
    for operation in expectation.chain:
        execution = final_execution(workflow, operation)
        if execution is None:
            errors.append(f"{operation} has no step execution")
        elif execution.get("status") != "SUCCEEDED":
            errors.append(
                f"{operation} ended {execution.get('status')}: "
                f"{str(execution.get('error') or '')[:200]}"
            )
    return errors


def incident_state_errors(
    records: dict[str, Any], incident_id: str | None, *, recovered: bool = True
) -> list[str]:
    state = incident_of(records, incident_id).get("state")
    if recovered and state != "RECOVERED":
        return [f"the occurrence's incident {incident_id} is {state}, not RECOVERED"]
    if state == "QUARANTINED":
        return [f"the occurrence's incident {incident_id} is QUARANTINED"]
    return []


def hardware_planned(activity: dict[str, Any], *, allowed: Iterable[str]) -> list[str]:
    """No workflow of the node planned a hardware step outside ``allowed``."""

    from scripts.e2e.regional.general_live_registry import HARDWARE_OPERATIONS

    permitted = set(allowed)
    return [
        f"workflow {item.get('request_id')} plans {sorted(found)}"
        for item in activity.get("workflows") or []
        if (found := (set(operations(item)) & HARDWARE_OPERATIONS) - permitted)
    ]


# --------------------------------------------------------------------------- #
# The partial-GPU job (xid_twin_unbound_job)
# --------------------------------------------------------------------------- #
#: Read-only: the completion watcher's attempt observations of one job -- the
#: attempt, its phase and each container's node and GPU UUIDs (the watcher's
#: own nvidia-smi in the container) -- the GPUs the product binds the job to.
JOB_GPUS_PROBE = r"""
import json
import sys

from gpu_fault.app import ApplicationContext

cluster_id, job_id = sys.argv[1:3]
store = ApplicationContext.from_environment().store
attempts = []
for item in store.list_attempt_observations(cluster_id):
    if item.job_id != job_id:
        continue
    attempts.append({
        "attempt_id": item.attempt_id,
        "workload_phase": getattr(item.workload_phase, "value", item.workload_phase),
        "observed_at": item.observed_at.isoformat(),
        "containers": [
            {
                "pod_uid": container.pod_uid,
                "node_id": container.node_id,
                "gpu_uuids": list(container.gpu_uuids),
                "terminated": container.terminated,
            }
            for container in item.containers
        ],
    })
print(json.dumps({"job_id": job_id, "attempts": attempts}, sort_keys=True, default=str))
"""


def watcher_gpus(document: dict[str, Any], *, attempt_id: str) -> set[str]:
    """The GPUs the product's watcher maps the attempt's live containers to."""

    return {
        gpu
        for item in document.get("attempts") or []
        if item.get("attempt_id") == attempt_id
        for container in item.get("containers") or []
        if not container.get("terminated")
        for gpu in container.get("gpu_uuids") or []
    }


def job_restart_errors(
    records: dict[str, Any], *, attempt_id: str, restart_attempt_id: str | None
) -> list[str]:
    """The job was stopped once by the occurrence and restarted once.

    ``records`` is ``general_live_store.attempt_records`` for the original
    attempt (and its restart): the original attempt ends STOPPED by the
    incident with no completion decision of its own beyond NO_ACTION and no
    passive containment, the job's restart ledger holds exactly one
    reservation.
    """

    entry = (records.get("attempts") or {}).get(attempt_id) or {}
    errors = []
    terminal = (entry.get("event") or {}).get("terminal_status")
    if terminal not in ("STOPPED",):
        errors.append(
            f"the job's attempt terminal is {terminal}, expected STOPPED by the "
            "occurrence's STOP_WORKLOADS"
        )
    decision = entry.get("decision") or {}
    if decision and decision.get("status") != "NO_ACTION":
        errors.append(
            f"the attempt's completion decision is {decision.get('status')}; an "
            "incident-initiated stop is NO_ACTION"
        )
    if entry.get("containment_incident"):
        errors.append("a passive containment incident was opened for the stop")
    if not restart_attempt_id:
        errors.append("RESTART_WORKLOAD recorded no restart_attempt_id")
    ledger = records.get("restart_budget")
    if not isinstance(ledger, dict):
        errors.append("the job has no restart ledger after its restart")
    elif ledger.get("restart_count") != 1:
        errors.append(
            f"the job's restart ledger counts {ledger.get('restart_count')} "
            "restarts, expected one"
        )
    return errors


__all__ = [
    "CASE",
    "GAUGE",
    "GAUGE_SOURCE",
    "GENERATION_EVIDENCE",
    "JOB_GPUS_PROBE",
    "KERNEL_DECODED_XIDS",
    "KERNEL_SOURCE",
    "MARKER_ASSOCIATION",
    "PRE_TWIN_GENERATION",
    "TWIN_WINDOW_SECONDS",
    "WATCHED_OPERATIONS",
    "XID_TWIN_GENERATION_PROBE",
    "XID_TWIN_RECORDS_PROBE",
    "TwinExpectation",
    "TwinGeneration",
    "TwinPair",
    "chain_errors",
    "command_receipts",
    "decided",
    "decision_class_errors",
    "executed_steps",
    "execution_errors",
    "hardware_planned",
    "incident_of",
    "incident_state_errors",
    "job_restart_errors",
    "marker_associations",
    "node_workflows",
    "repeat_errors",
    "reports",
    "require_twin_generation",
    "seconds_between",
    "twin_errors",
    "twin_generation",
    "twin_pair",
    "twin_records",
    "watcher_gpus",
    "workflow_errors",
]
