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
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_common import activity
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    utc_now,
)
from scripts.e2e.regional.general_live_kernel_replay import (
    kernel_replay,
    reviewed_record,
)
from scripts.e2e.regional.general_live_registry import (
    LIVE_REPLAY,
    VariantSpec,
)
from scripts.e2e.regional.general_live_verdicts import (
    hardware_errors,
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
    """NODE_HEALTH incidents minted from a node-log entry on this node."""

    result = []
    for incident in activity_document.get("incidents") or []:
        if incident.get("event_type") != "NODE_HEALTH":
            continue
        if not str(incident.get("event_id") or "").startswith("log-"):
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

    message = (
        "mce: [Hardware Error]: Machine check events logged: corrected error CPU 1"
    )
    started, _ = _drive(
        ctx, _severe(message), restore_seconds=ctx.spec.max_minutes * 60
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


def _negative_variant(
    ctx: VariantContext, records: list[dict[str, Any]], note: str
) -> dict[str, Any]:
    """A reviewed record that the trust/boot gate must drop: no isolation."""

    started, _ = _drive(ctx, records, restore_seconds=ctx.spec.max_minutes * 60)
    # Hold a full detection budget; the correct behaviour is that nothing fires.
    ctx.hold(min(DETECTION_BUDGET, ctx.spec.max_minutes * 60 - 120), name="negative")
    observed = activity(ctx, started, kinds=("NODE_LOGS",))
    incidents = _log_incidents(observed)
    stages = {
        "no_finding": [
            f"an untrusted/old-boot kernel line produced an incident: {item['incident_id']}"
            for item in incidents
            if item.get("effective_action") == "QUARANTINE"
        ],
        "not_isolated": ["node isolated by an untrusted/old-boot kernel line"]
        if node_isolated(ctx.regional.node_snapshot(ctx.settings.node))
        else [],
        "hardware": hardware_errors(observed),
    }
    return _result(ctx, stages, notes=[note])


def user_text(ctx: VariantContext) -> dict[str, Any]:
    records = _severe(SEVERE_MESSAGES["uncorrectable_mce"], transport="syslog")
    return _negative_variant(
        ctx,
        records,
        "a user-space (journal, non-kernel transport) line is never trusted",
    )


def old_boot(ctx: VariantContext) -> dict[str, Any]:
    records = _severe(
        SEVERE_MESSAGES["uncorrectable_mce"],
        boot_id="00000000-0000-0000-0000-0000deadbeef",
    )
    return _negative_variant(
        ctx,
        records,
        "a kernel line from a prior boot must not isolate the current boot",
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


def cursor_pagination(ctx: VariantContext) -> dict[str, Any]:
    """Several severe records at once: each reaches its own incident, none starved."""

    records = [
        reviewed_record(SEVERE_MESSAGES["uncorrectable_mce"], priority=2, label="mce"),
        reviewed_record(
            SEVERE_MESSAGES["readonly_storage"], priority=2, label="storage"
        ),
        reviewed_record(SEVERE_MESSAGES["kernel_lockup"], priority=2, label="lockup"),
    ]
    started, _ = _drive(ctx, records, restore_seconds=ctx.spec.max_minutes * 60)
    observed = _wait_log_incident(
        ctx,
        started,
        policy_source="SITE_KERNEL_LOG",
        minimum=len(records),
        name="pagination",
    )
    incidents = _log_incidents(observed, policy_source="SITE_KERNEL_LOG")
    stages = {
        "all_processed": []
        if len(incidents) >= len(records)
        else [
            f"only {len(incidents)}/{len(records)} severe kernel records reached an "
            "incident; the cursor starved the rest"
        ],
        "hardware": hardware_errors(observed),
    }
    return _result(
        ctx,
        stages,
        extra={"incidents": [item["incident_id"] for item in incidents]},
        notes=["same-timestamp records across cursor pages must not starve later ones"],
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
        "reviewed kernel-transport MCE 'corrected error' line",
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
        "three reviewed severe records in one window (same-timestamp across cursor pages)",
        "each reaches its own incident; none is starved by the cursor",
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
