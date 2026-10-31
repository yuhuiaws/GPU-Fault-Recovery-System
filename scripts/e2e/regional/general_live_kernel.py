"""GF-GENERAL-LIVE-016 severe kernel log (reviewed journal-source replay).

The node log collector reads ``journalctl --output=json`` and grades each entry
through the trusted-source gate (``log_rules`` / ``log_recovery_context``): a
hardware-fatal rule (MCE, STORAGE I/O, RDMA/EFA, kernel lockup) may only fire
from a ``dmesg`` origin (``_TRANSPORT=kernel``) that matches the node's current
boot, and a corrected MCE is downgraded to a diagnostic. This module drives the
reusable kernel-log replay (``general_live_kernel_replay``, journal mode): it
installs a reviewed set of kernel-transport records for the node/boot, arms the
pass-through wrapper, and reads the product's own records back.

Every variant is ``live-replay``: it proves the collector -> trust gate ->
finding -> incident -> cordon/quarantine (+ STOP only for a freshly attributed
workload) -> notification chain on a reviewed source, never a physical kernel
fault. The negative variants are the point of the trust gate: a user-space line
(``user_text``), an old-boot line (``old_boot``) and a corrected MCE
(``corrected_mce``) must not isolate the current node.

The reviewed lines are what the kernel prints, so the product grades the real
format and not a paraphrase (Linux v6.12 unless noted):

* ``corrected_mce``: the ``__print_mce()`` header of a polled record,
  ``arch/x86/kernel/cpu/mce/core.c`` ``pr_emerg(HW_ERR "CPU %d: Machine
  Check%s: %Lx Bank %d: %016Lx\n", ...)`` with no " Exception" (MCG_STATUS.MCIP
  clear) and an IA32_MCi_STATUS whose UC bit 61 is clear -- "the processor was
  able to correct the error condition" (Intel SDM Vol. 3B 253669-093US
  18.3.2.2). The product grades it by those bits (``gpu_fault.mce_grading``),
  so this variant needs a release that carries that grading; an earlier
  release reads the same header as an uncorrected MCE and quarantines.
* ``kernel_lockup``: ``kernel/watchdog.c`` ``watchdog_timer_fn()``
  ``pr_emerg("BUG: soft lockup - CPU#%d stuck for %us! [%s:%d]\n", ...)`` under
  ``pr_fmt "watchdog: "``.
* ``rdma_fatal``: ``drivers/infiniband/hw/efa/efa_com.c``
  ``efa_com_wait_and_process_admin_cq_polling()`` ``ibdev_err_ratelimited(...,
  "Wait for completion (polling) timeout\n")`` -- the device stopped answering
  its admin queue and the driver closes it -- printed through
  ``drivers/infiniband/core/device.c`` ``__ibdev_printk()`` as ``"%s %s %s:
  %pV"`` = ``efa <pci address> efa_0: ...``.
* ``uncorrectable_mce`` and ``readonly_storage`` keep their earlier reviewed
  wording: the rule names the very words they carry (``uncorrected``,
  ``Remounting filesystem read-only``), and the kernel's own forms of them
  (``Machine Check Exception`` with UC set, ``EXT4-fs (dev): Remounting
  filesystem read-only``) are pinned in the product's tests.

Each reviewed record travels under a synthetic journal cursor that carries the
run id (``probes/general_live_kernel_source_probe.py``): the cursor is the
entry id, and the control plane derives the incident's event id from it, so a
second run on the same node and boot must not reuse the first run's identity.
A severe finding on a node whose isolation is already in flight joins that
incident instead of minting one (``NODE_HEALTH_GROUP``), so incidents are
matched under every label the product gives them and ``cursor_pagination``
follows each record through its own finding's marker.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from scripts.e2e.regional.general_live_common import activity
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    utc_now,
)
from scripts.e2e.regional.general_live_kernel_replay import (
    kernel_replay,
    replay_entry_prefix,
    reviewed_record,
)
from scripts.e2e.regional.general_live_registry import (
    LIVE_REPLAY,
    VariantSpec,
)
from scripts.e2e.regional.general_live_verdicts import (
    NODE_HEALTH_EVENT_TYPES,
    hardware_errors,
    minted_in_read,
    node_isolated,
    notification_errors,
    operations,
    subsequence,
    verdict,
    workflows_of,
)

CASE = "GF-GENERAL-LIVE-016"
ENABLE = "ENABLE_NODE_LOG_COLLECTOR"
#: node log interval 10 s, health summary 300 s; a finding lands on the next
#: batch after the record is read, plus the workflow's cordon/quarantine.
DETECTION_BUDGET = 420
SETTLE_BUDGET = 420
ISOLATION_OPERATIONS = ("MARK_UNSCHEDULABLE", "QUARANTINE")
#: The boot id ``old_boot``'s reviewed record carries: never a node's own.
OLD_BOOT_ID = "00000000-0000-0000-0000-0000deadbeef"

#: The reviewed entries the control plane itself received: every NODE_LOGS batch
#: the product stored for the node since a moment -- ``app/ingest/telemetry.py``
#: keeps each batch with entries, the one ``host_health.evaluate_logs`` graded in
#: the same transaction -- reduced to the entries whose entry id (the replay
#: cursor, ``replay_entry_prefix``) starts with a prefix. Read-only.
NODE_LOG_ENTRY_PROBE = r"""
import json
import sys
from datetime import datetime

from gpu_fault.app import ApplicationContext
from gpu_fault.telemetry import EvidenceKind

cluster_id, node_id, since_text, prefix = sys.argv[1:5]
since = datetime.fromisoformat(since_text.replace("Z", "+00:00"))
store = ApplicationContext.from_environment().store
entries = []
for item in store.list_raw_evidence(
    cluster_id, node_id=node_id, kind=EvidenceKind.NODE_LOGS, limit=2000
):
    if item.observed_at < since:
        continue
    payload = item.payload if isinstance(item.payload, dict) else {}
    for entry in payload.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        if not str(entry.get("entry_id") or "").startswith(prefix):
            continue
        fields = entry.get("fields") or {}
        entries.append({
            "record_id": item.record_id,
            "batch_id": payload.get("batch_id"),
            "collected_at": item.observed_at.isoformat(),
            "entry_id": entry.get("entry_id"),
            "source": entry.get("source"),
            "transport": fields.get("_TRANSPORT"),
            "boot_id": fields.get("_BOOT_ID"),
            "collector_boot_id": fields.get("_COLLECTOR_BOOT_ID"),
            "message": str(entry.get("message") or "")[:200],
        })
entries.sort(key=lambda item: item["collected_at"])
print(json.dumps({"entries": entries}, sort_keys=True, default=str))
"""


# --------------------------------------------------------------------------- #
# reviewed record sets (one node/boot)
# --------------------------------------------------------------------------- #
def _severe(
    message: str, *, boot_id: str = "", transport: str = "kernel"
) -> list[dict[str, Any]]:
    return [reviewed_record(message, transport=transport, priority=2, boot_id=boot_id)]


SEVERE_MESSAGES = {
    "uncorrectable_mce": "mce: [Hardware Error]: Machine check events logged: uncorrected error CPU 3",
    "readonly_storage": "EXT4-fs (nvme1n1): Remounting filesystem read-only after I/O error",
    "rdma_fatal": "efa 0000:00:1d.0 efa_0: Wait for completion (polling) timeout",
    "kernel_lockup": "watchdog: BUG: soft lockup - CPU#7 stuck for 23s! [python3:41234]",
}
#: ``__print_mce()`` header of a corrected, polled record: "Machine Check:"
#: (no " Exception") and IA32_MCi_STATUS 0x8c00004000010090 = VAL|MISCV|ADDRV
#: with UC (bit 61) and PCC (bit 57) clear.
CORRECTED_MCE_MESSAGE = (
    "mce: [Hardware Error]: CPU 1: Machine Check: 0 Bank 7: 8c00004000010090"
)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _drive(
    ctx: VariantContext, records: list[dict[str, Any]], *, restore_seconds: int
) -> tuple[Any, dict[str, Any]]:
    """Install, open (approved), arm; return (started_at, injection oracle)."""

    replay = kernel_replay(ctx, mode="journal")
    replay.create()
    replay.install(records)
    opened = replay.open(approved=True, restore_seconds=restore_seconds)
    ctx.evidence.write("replay-open", opened)
    started = utc_now()
    replay.arm()
    # Let the collector poll (10 s) and re-post across the emission budget.
    ctx.hold(90, name="inject")
    oracle = replay.read_back()
    ctx.evidence.append_oracle(
        [{"injected": oracle.get("injected")}], source="kernel-journal"
    )
    injected = oracle.get("injected") or {}
    if not injected.get("emission_counts"):
        raise VariantInconclusive(
            "the journal wrapper reported no reviewed records were ingested"
        )
    return started, oracle


def _log_incidents(
    activity_document: dict[str, Any], *, policy_source: str | None = None
) -> list[dict[str, Any]]:
    """Incidents a node-log entry on this node minted since the read's ``since``.

    Under every label the product gives that record (``NODE_HEALTH_EVENT_TYPES``):
    a second severe finding on a node whose isolation is in flight joins it,
    and the minted incident -- same id, same ``log-`` event id -- becomes
    ``NODE_HEALTH_GROUP`` (``families/drain.py``
    ``ingest_terminal_node_quarantine``). ``minted_in_read`` keeps out an older
    incident the read lists only because a workflow of it is newer.
    """

    result = []
    for incident in activity_document.get("incidents") or []:
        if incident.get("event_type") not in NODE_HEALTH_EVENT_TYPES:
            continue
        if not str(incident.get("event_id") or "").startswith("log-"):
            continue
        if not minted_in_read(activity_document, incident):
            continue
        if policy_source is not None and incident.get("policy_source") != policy_source:
            continue
        result.append(incident)
    return sorted(result, key=lambda item: str(item.get("created_at")))


def _wait_log_incident(
    ctx: VariantContext,
    since: Any,
    *,
    policy_source: str | None,
    minimum: int,
    name: str,
) -> dict[str, Any]:
    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, since, kinds=("NODE_LOGS",))
        found = _log_incidents(last, policy_source=policy_source)
        return last if len(found) >= minimum else None

    value = ctx.wait_until(
        accept, timeout_seconds=DETECTION_BUDGET, poll_seconds=15, name=name
    )
    return value if value is not None else last


def _quarantine_variant(ctx: VariantContext, variant: str) -> dict[str, Any]:
    records = _severe(SEVERE_MESSAGES[variant])
    started, oracle = _drive(ctx, records, restore_seconds=ctx.spec.max_minutes * 60)
    observed = _wait_log_incident(
        ctx,
        started,
        policy_source="SITE_KERNEL_LOG",
        minimum=1,
        name=f"{variant}-detect",
    )
    incidents = _log_incidents(observed, policy_source="SITE_KERNEL_LOG")
    stages = {
        "chain": _quarantine_chain_errors(observed, incidents),
        "hardware": hardware_errors(observed),
        "isolation": _isolated_errors(ctx, observed),
        "notification": notification_errors(
            observed, incident_ids=[item["incident_id"] for item in incidents]
        ),
    }
    return _result(
        ctx, stages, extra={"incidents": [item["incident_id"] for item in incidents]}
    )


def _quarantine_chain_errors(
    activity_document: dict[str, Any], incidents: list[dict[str, Any]]
) -> list[str]:
    if not incidents:
        return ["no severe kernel-log finding reached a NODE_HEALTH incident"]
    errors = []
    incident = incidents[0]
    if incident.get("effective_action") != "QUARANTINE":
        errors.append(
            f"severe kernel log incident action is {incident.get('effective_action')},"
            " expected QUARANTINE"
        )
    workflows = workflows_of(activity_document, str(incident["incident_id"]))
    if not workflows:
        errors.append("the severe kernel-log incident has no workflow")
        return errors
    steps = operations(workflows[0])
    if not subsequence(ISOLATION_OPERATIONS, steps):
        errors.append(f"workflow steps {steps} lack {list(ISOLATION_OPERATIONS)}")
    if incident.get("state") not in {"ACTION_PENDING", "ESCALATED", "QUARANTINED"}:
        errors.append(f"incident state is {incident.get('state')}")
    return errors


def _isolated_errors(
    ctx: VariantContext, activity_document: dict[str, Any]
) -> list[str]:
    node = ctx.regional.node_snapshot(ctx.settings.node)
    if not node_isolated(node):
        return ["the node was not isolated after a severe kernel-log finding"]
    return []


def uncorrectable_mce(ctx: VariantContext) -> dict[str, Any]:
    return _quarantine_variant(ctx, "uncorrectable_mce")


def readonly_storage(ctx: VariantContext) -> dict[str, Any]:
    return _quarantine_variant(ctx, "readonly_storage")


def rdma_fatal(ctx: VariantContext) -> dict[str, Any]:
    return _quarantine_variant(ctx, "rdma_fatal")


def kernel_lockup(ctx: VariantContext) -> dict[str, Any]:
    return _quarantine_variant(ctx, "kernel_lockup")


def corrected_mce(ctx: VariantContext) -> dict[str, Any]:
    """A corrected MCE must be a diagnostic, never isolation."""

    started, _ = _drive(
        ctx,
        _severe(CORRECTED_MCE_MESSAGE),
        restore_seconds=ctx.spec.max_minutes * 60,
    )
    observed = _wait_log_incident(
        ctx,
        started,
        policy_source="SITE_APPLICATION_LOG",
        minimum=1,
        name="corrected-detect",
    )
    incidents = _log_incidents(observed)
    quarantine = [
        item for item in incidents if item.get("effective_action") == "QUARANTINE"
    ]
    stages = {
        "diagnostic": []
        if any(item.get("effective_action") == "RUN_DIAGNOSTICS" for item in incidents)
        else ["a corrected MCE did not produce a RUN_DIAGNOSTICS finding"],
        "no_quarantine": [
            f"a corrected MCE was escalated to QUARANTINE: {item['incident_id']}"
            for item in quarantine
        ],
        "not_isolated": ["node isolated on a corrected MCE"]
        if node_isolated(ctx.regional.node_snapshot(ctx.settings.node))
        else [],
        "hardware": hardware_errors(observed),
    }
    return _result(
        ctx, stages, notes=["corrected MCE is graded WARNING/RUN_DIAGNOSTICS"]
    )


def received_entries(
    ctx: VariantContext, since: Any, prefix: str
) -> list[dict[str, Any]]:
    """The control plane's own copy of the reviewed record since ``since``
    (``NODE_LOG_ENTRY_PROBE``): the batches it stored and graded that carry
    an entry id starting with ``prefix``."""

    value = ctx.regional.cpu_python(
        NODE_LOG_ENTRY_PROBE,
        ctx.settings.regional.cluster_id,
        ctx.settings.node,
        since.isoformat(),
        prefix,
        timeout=120,
    )
    return [item for item in value.get("entries") or [] if isinstance(item, dict)]


def replay_findings(activity_document: dict[str, Any], prefix: str) -> list[str]:
    """Every finding the product made of the reviewed record, whatever its grade.

    Each finding writes its marker before ingestion picks the incident, and the
    marker names the entry it was read from (``raw_evidence_ref`` =
    ``<origin>://<node>/<entry id>``): the record's own incident and one it
    joined both show. The preflight leaves the node no open incident, so any
    incident such a finding could reach was minted inside the read.
    """

    errors = []
    for marker in activity_document.get("markers") or []:
        entry_id = str(marker.get("raw_evidence_ref") or "").rpartition("/")[2]
        if entry_id.startswith(prefix):
            errors.append(
                f"the reviewed record {entry_id} produced a finding: marker "
                f"{marker.get('marker_id')} on incident "
                f"{marker.get('incident_id') or '(none)'}"
            )
    return errors


#: What a received copy of a negative's record must show (None) or why not.
Expectation = Callable[[dict[str, Any]], str | None]


def unread_reasons(
    received: list[dict[str, Any]],
    findings: list[str],
    expect: Expectation,
) -> list[str]:
    """Why the negative is unproven: the product never received the reviewed
    record (and nothing it graded names it), or received it without the trait
    the gate must refuse (``expect`` names the mismatch, None when it holds)."""

    if not received:
        if findings:
            return []
        return [
            "the control plane stored no NODE_LOGS batch carrying the reviewed "
            "record: the collector never delivered it, so 'no finding' proves "
            "nothing about the gate"
        ]
    return sorted({reason for item in received if (reason := expect(item))})


def _negative_variant(
    ctx: VariantContext,
    records: list[dict[str, Any]],
    note: str,
    *,
    expect: Expectation,
) -> dict[str, Any]:
    """A reviewed record that the trust/boot gate must drop: no finding at all.

    The negative is proved on the product's records: the control plane received
    the record with the trait the gate refuses (``received_entries`` -- else the
    variant is INCONCLUSIVE), and no marker or incident came of it, whatever its
    grade, within a full detection budget. The wrapper's emission count only
    says the record left the node's journal wrapper.
    """

    started, _ = _drive(ctx, records, restore_seconds=ctx.spec.max_minutes * 60)
    prefix = replay_entry_prefix(ctx.run_id, 0)
    # Hold a full detection budget; the correct behaviour is that nothing fires.
    ctx.hold(min(DETECTION_BUDGET, ctx.spec.max_minutes * 60 - 120), name="negative")
    observed = activity(ctx, started, kinds=("NODE_LOGS",))
    received = received_entries(ctx, started, prefix)
    ctx.evidence.write("negative-received", {"prefix": prefix, "entries": received})
    findings = replay_findings(observed, prefix)
    stages = {
        "no_finding": [
            *findings,
            *(
                "a severe node-log incident was minted in the window: "
                f"{item['incident_id']}"
                for item in _log_incidents(observed)
                if item.get("effective_action") == "QUARANTINE"
            ),
        ],
        "not_isolated": ["node isolated by an untrusted/old-boot kernel line"]
        if node_isolated(ctx.regional.node_snapshot(ctx.settings.node))
        else [],
        "hardware": hardware_errors(observed),
    }
    return _result(
        ctx,
        stages,
        notes=[note],
        extra={"received": received[:8]},
        inconclusive=unread_reasons(received, findings, expect),
    )


def _user_space(entry: dict[str, Any]) -> str | None:
    if entry.get("source") == "journal" and entry.get("transport") != "kernel":
        return None
    return (
        f"the control plane received the reviewed record as {entry.get('source')} "
        f"(transport {entry.get('transport')}), not the user-space journal origin "
        "the gate must refuse"
    )


def _prior_boot(entry: dict[str, Any]) -> str | None:
    def plain(value: Any) -> str:
        return str(value or "").replace("-", "").lower()

    if (
        entry.get("source") == "dmesg"
        and plain(entry.get("boot_id")) == plain(OLD_BOOT_ID)
        and plain(entry.get("boot_id")) != plain(entry.get("collector_boot_id"))
    ):
        return None
    return (
        f"the control plane received the reviewed record as {entry.get('source')} "
        f"with boot id {entry.get('boot_id')} (collector boot "
        f"{entry.get('collector_boot_id')}), not a prior boot's kernel line"
    )


def user_text(ctx: VariantContext) -> dict[str, Any]:
    records = _severe(SEVERE_MESSAGES["uncorrectable_mce"], transport="syslog")
    return _negative_variant(
        ctx,
        records,
        "a user-space (journal, non-kernel transport) line is never trusted",
        expect=_user_space,
    )


def old_boot(ctx: VariantContext) -> dict[str, Any]:
    records = _severe(SEVERE_MESSAGES["uncorrectable_mce"], boot_id=OLD_BOOT_ID)
    return _negative_variant(
        ctx,
        records,
        "a kernel line from a prior boot must not isolate the current boot",
        expect=_prior_boot,
    )


def unknown_attempt(ctx: VariantContext) -> dict[str, Any]:
    """A severe kernel line with no resolvable attempt: isolate, but no STOP."""

    records = _severe(SEVERE_MESSAGES["rdma_fatal"])
    started, _ = _drive(ctx, records, restore_seconds=ctx.spec.max_minutes * 60)
    observed = _wait_log_incident(
        ctx, started, policy_source="SITE_KERNEL_LOG", minimum=1, name="unknown-attempt"
    )
    incidents = _log_incidents(observed, policy_source="SITE_KERNEL_LOG")
    stages = {
        "chain": _quarantine_chain_errors(observed, incidents),
        "no_stop": _no_stop_without_attribution(observed, incidents),
        "isolation": _isolated_errors(ctx, observed),
        "hardware": hardware_errors(observed),
    }
    return _result(
        ctx,
        stages,
        notes=["no managed attempt: the node is isolated but no workload is stopped"],
    )


def _no_stop_without_attribution(
    activity_document: dict[str, Any], incidents: list[dict[str, Any]]
) -> list[str]:
    errors = []
    for incident in incidents:
        for workflow in workflows_of(activity_document, str(incident["incident_id"])):
            if "STOP_WORKLOADS" in operations(workflow):
                errors.append(
                    f"workflow {workflow.get('request_id')} stopped a workload with no "
                    "resolved attempt"
                )
    return errors


def _record_links(
    activity_document: dict[str, Any],
    incidents: list[dict[str, Any]],
    prefixes: dict[str, str],
) -> dict[str, dict[str, Any] | None]:
    """The incident of this run each reviewed record's finding was routed to.

    A finding's marker (``NodeHealthFinding.marker``) names the entry it was
    read from -- ``raw_evidence_ref`` is ``<origin>://<node>/<entry id>`` and
    the entry id is the replay cursor (``replay_entry_prefix``) -- and, once
    ingestion picked the incident, that incident: the finding's own, or the
    one an earlier finding of the burst minted and this one joined. ``None``
    for a record no marker on one of ``incidents`` names: the product never
    read it, or never graded it onto an incident of this run.
    """

    minted = {str(item.get("incident_id")): item for item in incidents}
    links: dict[str, dict[str, Any] | None] = {}
    for label, prefix in prefixes.items():
        links[label] = None
        for marker in activity_document.get("markers") or []:
            entry_id = str(marker.get("raw_evidence_ref") or "").rpartition("/")[2]
            incident = minted.get(str(marker.get("incident_id") or ""))
            if incident is None or not entry_id.startswith(prefix):
                continue
            links[label] = {
                "entry_id": entry_id,
                "marker_id": marker.get("marker_id"),
                "incident_id": incident.get("incident_id"),
                "event_type": incident.get("event_type"),
            }
            break
    return links


def _wait_records_linked(
    ctx: VariantContext, since: Any, prefixes: dict[str, str], *, name: str
) -> dict[str, Any]:
    """Read the node's activity until every record's finding names an incident."""

    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, since, kinds=("NODE_LOGS",))
        incidents = _log_incidents(last, policy_source="SITE_KERNEL_LOG")
        return last if all(_record_links(last, incidents, prefixes).values()) else None

    value = ctx.wait_until(
        accept, timeout_seconds=DETECTION_BUDGET, poll_seconds=15, name=name
    )
    return value if value is not None else last


def cursor_pagination(ctx: VariantContext) -> dict[str, Any]:
    """A burst of severe records in one read: every one reaches an incident.

    Three reviewed severe records, emitted by one wrapper call with one
    timestamp and three cursors, must each be read, graded and routed to an
    incident of this run -- none lost to the cursor, the entry identity or the
    batch limits. How many incidents that makes is the product's choice: the
    first finding mints one and opens the node's isolation, the others join
    that isolation while it is in flight, as reasons of the same record
    re-labelled ``NODE_HEALTH_GROUP`` (``families/drain.py``
    ``ingest_terminal_node_quarantine``) -- one cordon and quarantine of the
    node, not three racing ones. Each record is therefore followed through its
    own finding's marker (``_record_links``), and the isolation chain is graded
    on the incident the burst minted first.

    Live 2026-10-26 (R13g, HyperPod and plain EKS) and 2026-10-27 (R13h, plain
    EKS) this variant read FAIL "only 0/3 severe kernel records reached an
    incident; the cursor starved the rest" while the first armed poll had
    delivered all three records in one batch and the product had folded them
    into one NODE_HEALTH_GROUP incident and quarantined the node: the grade
    took only ``NODE_HEALTH`` and wanted one incident per record. A page
    boundary between same-timestamp entries cannot be placed through the
    replay (the three land in one read); the collector's resume across one is
    ``tests/collectors/test_node_log_journal_positioning.py``.
    """

    records = [
        reviewed_record(SEVERE_MESSAGES["uncorrectable_mce"], priority=2, label="mce"),
        reviewed_record(
            SEVERE_MESSAGES["readonly_storage"], priority=2, label="storage"
        ),
        reviewed_record(SEVERE_MESSAGES["kernel_lockup"], priority=2, label="lockup"),
    ]
    prefixes = {
        str(record["label"]): replay_entry_prefix(ctx.run_id, index)
        for index, record in enumerate(records)
    }
    started, _ = _drive(ctx, records, restore_seconds=ctx.spec.max_minutes * 60)
    observed = _wait_records_linked(ctx, started, prefixes, name="pagination")
    incidents = _log_incidents(observed, policy_source="SITE_KERNEL_LOG")
    links = _record_links(observed, incidents, prefixes)
    starved = [label for label, link in links.items() if link is None]
    reached = sorted({str(link["incident_id"]) for link in links.values() if link})
    stages = {
        "all_processed": [
            f"only {len(records) - len(starved)}/{len(records)} severe kernel "
            f"records reached an incident of this run; no finding of "
            f"{', '.join(starved)} names one (marker raw_evidence_ref = replay "
            "cursor)"
        ]
        if starved
        else [],
        # Graded once something was minted; nothing minted is all_processed's.
        "chain": _quarantine_chain_errors(observed, incidents) if incidents else [],
        "hardware": hardware_errors(observed),
    }
    return _result(
        ctx,
        stages,
        extra={
            "incidents": [item["incident_id"] for item in incidents],
            "records": links,
        },
        notes=[
            "same-timestamp records of one read must not starve one another; "
            f"{len(records) - len(starved)} record(s) reached {len(reached)} "
            "incident(s): a later severe finding joins the node's in-flight "
            "isolation (NODE_HEALTH_GROUP)"
        ],
    )


# --------------------------------------------------------------------------- #
# shared result
# --------------------------------------------------------------------------- #
def _result(
    ctx: VariantContext,
    stages: dict[str, list[str]],
    *,
    notes: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    inconclusive: list[str] | None = None,
) -> dict[str, Any]:
    # An INCONCLUSIVE carries its reason: ``detail`` for the runner, the errors
    # list for the board (as ``general_live_spares``).
    reasons = list(inconclusive or [])
    return {
        "verdict": verdict(stages, inconclusive=reasons),
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ]
        + [f"inconclusive: {item}" for item in reasons],
        "notes": notes or [],
        "proof_scope": "live-replay",
        **({"inconclusive": reasons, "detail": "; ".join(reasons)} if reasons else {}),
        **(extra or {}),
    }


def _spec(
    variant: str, injection: str, expected: str, execute: Any, *, risk: str = "M"
) -> VariantSpec:
    return VariantSpec(
        case_id=CASE,
        variant=variant,
        proof_mode=LIVE_REPLAY,
        lanes=("eks", "hp"),
        risk=risk,
        flags="S",
        injection=injection,
        expected=expected,
        cleanup="close the journalctl wrapper, disable the node log collector, confirm clean",
        max_minutes=18,
        approvals=(ENABLE,),
        execute=execute,
    )


VARIANTS = [
    _spec(
        "uncorrectable_mce",
        "reviewed kernel-transport MCE 'uncorrected error' line via the node log collector",
        "MCE CRITICAL -> cordon + quarantine + notification; no reset/reboot; no STOP without attribution",
        uncorrectable_mce,
        risk="H",
    ),
    _spec(
        "corrected_mce",
        "reviewed kernel-transport __print_mce header 'CPU n: Machine Check: ... Bank b: <status>' "
        "with IA32_MCi_STATUS.UC clear (needs a release with mce_grading)",
        "downgraded to WARNING RUN_DIAGNOSTICS; no quarantine, no isolation",
        corrected_mce,
        risk="L",
    ),
    _spec(
        "readonly_storage",
        "reviewed kernel-transport 'Remounting filesystem read-only' I/O line",
        "STORAGE CRITICAL -> cordon + quarantine + notification; no reset/reboot",
        readonly_storage,
        risk="H",
    ),
    _spec(
        "rdma_fatal",
        "reviewed kernel-transport EFA admin-queue 'Wait for completion (polling) timeout' line",
        "RDMA CRITICAL -> cordon + quarantine + notification; no unauthorized reboot",
        rdma_fatal,
        risk="H",
    ),
    _spec(
        "kernel_lockup",
        "reviewed kernel-transport 'watchdog: BUG: soft lockup' line (never a real panic)",
        "SYSTEM_LOG CRITICAL -> cordon + quarantine + notification",
        kernel_lockup,
        risk="H",
    ),
    _spec(
        "user_text",
        "the same severe text on a non-kernel transport (user-space journal origin)",
        "the trusted-source gate drops it: no finding, no isolation",
        user_text,
        risk="L",
    ),
    _spec(
        "old_boot",
        "a reviewed kernel-transport severe line stamped with a prior boot id",
        "current-boot evidence check fails: the current node is not isolated",
        old_boot,
        risk="L",
    ),
    _spec(
        "cursor_pagination",
        "three reviewed severe records in one journal read (one timestamp, three "
        "cursors)",
        "every record's finding reaches an incident of this run -- the first mints "
        "it, the others join that one isolation (NODE_HEALTH_GROUP); none is starved",
        cursor_pagination,
        risk="M",
    ),
    _spec(
        "unknown_attempt",
        "a reviewed severe kernel line with no managed attempt to attribute it to",
        "the node is isolated and quarantined, but no workload is stopped (no guessed job)",
        unknown_attempt,
        risk="H",
    ),
]
