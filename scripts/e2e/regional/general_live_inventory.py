"""GF-GENERAL-LIVE-013 EFA device inventory and 014 K8s allocatable under-report.

013: the host collector counts EFA devices three ways -- PCI functions
(``GPU_FAULT_PCI_DEVICES_ROOT``), driver-bound RDMA devices and active ports
(``collectors/host/observation.efa_inventory_counts``) -- and a shortfall
resolves by failure mode (``host_health.efa_inventory_rule``): DRIVER_UNBOUND
remediates the driver, PCI_DEVICE_MISSING reboots, LINK_INACTIVE/EXCESS only
diagnose, and an *unreadable* reading is ``INVENTORY_UNAVAILABLE`` (UNKNOWN),
never zero cards. The safe, reviewed replays on this site are ``unreadable_state``
(a malformed PCI-root fixture repointed into the host collector's own
environment makes the observation UNAVAILABLE without touching a device) and the
whole-tree ``/sys/class/infiniband`` replays of the shared host source replay
(``general_live_source_replay.build_infiniband``: the node's real RDMA tree copied
with one reviewed mutation and bound read-only into the host collector unit's
mount namespace): ``link_inactive`` (one real port's ``state``/``phys_state``
DOWN/Disabled -> LINK_INACTIVE), ``excess_device`` (one real device duplicated ->
EXCESS_DEVICE) and ``unknown_inventory`` (one device's driver identity made
ambiguous -> ``EFA inventory observation unavailable``, UNKNOWN). The PCI root
stays real, so ``pci_discovered`` never drops and the PCI_DEVICE_MISSING reboot
chain cannot be entered by these. PCI_DEVICE_MISSING drives REBOOT on the only
EKS GPU node, and the driver-binding variants need a reserved native
unbind/rebind; those stay NOT_RUN.

The product's design for LINK_INACTIVE is diagnose first, then escalate:
``families/health.py`` ``_resource_escalation`` tags the diagnostic
VALIDATE_FABRIC ``failure_escalation_action=REBOOT_NODE``, and since batch 175
that step judges the *fresh* inventory readings observed after the chain
reached it (``adapters/gpu_validation.py``): SUCCEEDED when they show the
expected count, FAILED on a fresh mismatch -- when the reboot is the designed
answer to a confirmed, still-inactive link -- and the reading it rests on is
recorded in ``details.inventory_evidence``. ``link_inactive`` therefore shows
the inactive port for a bounded exposure only
(``general_live_source_replay.bounded_exposure_seconds``: the window's own
deadman closes it 1.7 collector intervals after it opened, after the second
tick that confirms the shortfall and before a third could be read), so the
diagnostic only ever sees the restored port: PASS is a SUCCEEDED validation
on fresh evidence, the incident RECOVERED, no RESTART_NODE, under
HOST_SOURCE_REPLAY alone. An inactive reading the product recorded after the
restore means the bound did not hold on that node; the designed escalation
is then right and the variant is INCONCLUSIVE, never a verdict on the product.
``excess_device`` has no escalation (``reboot_allowed`` is False for
EXCESS_DEVICE): its validation concludes on fresh evidence either way --
SUCCEEDED on the restored count, or FAILED on a fresh excess reading with
``escalate_failed_hardware_remediation`` planning nothing -- and PASS is a
concluded diagnostic that recorded its evidence, RECOVERED, no reboot.

014: a Kubernetes allocatable shortfall must come from a *real* device-plugin
registration fault (``collectors/cloud/kubernetes``); the spec is explicit that
patching Node.status or faking allocatable is not valid proof, so there is no
live-replay path. The native device-plugin fault needs the DEVICE_PLUGIN_RESTART
approval and a plugin pod-delete/re-register injection with new-Pod-UID
verification, not built this phase -- every 014 variant stays NOT_RUN with its
reason. The denominator keeps them all.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from scripts.e2e.regional.general_live_common import (
    activity,
    wait_for_incident,
    wait_settled,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantBlocked,
    VariantContext,
    VariantInconclusive,
    utc_now,
)
from scripts.e2e.regional.general_live_registry import (
    HARDWARE_OPERATIONS,
    LIVE_REPLAY,
    VariantSpec,
    deferred_variant,
)
from scripts.e2e.regional.general_live_source_replay import (
    HOST_SOURCE_REPLAY,
    INFINIBAND_TREE,
    HostSourceReplay,
    bounded_exposure_seconds,
    host_source_replay,
)
from scripts.e2e.regional.general_live_verdicts import (
    ChainExpectation,
    chain_errors,
    executions,
    finding_incidents,
    hardware_errors,
    isolation_operations,
    node_isolated,
    operations,
    parse_time,
    samples,
    verdict,
    workflows_of,
)

CASE_013 = "GF-GENERAL-LIVE-013"
CASE_014 = "GF-GENERAL-LIVE-014"
DETECTION_BUDGET = 420
EFA_METRICS = {
    "efa_inventory_mismatch",
    "efa_inventory_active_count",
    "efa_inventory_expected_count",
    "efa_inventory_missing_count",
    "efa_inventory_excess_count",
    "efa_inventory_discovered_count",
    "rdma_link_down",
}
INFINIBAND_ROOT = "/sys/class/infiniband"
#: The duplicated device of ``excess_device``; a name no real EFA device has.
EXCESS_DEVICE_NAME = "gfrs-excess0"
#: ``state`` / ``phys_state`` texts of an inactive port (the product reads the
#: leading number: ACTIVE is 4, LinkUp is 5).
INACTIVE_STATE = "1: DOWN"
INACTIVE_PHYS_STATE = "3: Disabled"
#: Isolation steps a diagnostic-only EFA chain must not plan.
ISOLATION_OPERATIONS = frozenset({"MARK_UNSCHEDULABLE", "QUARANTINE", "STOP_WORKLOADS"})
#: The workflow step deadline (GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS, 600 s by
#: default): a validation that regressed to waiting for an action completion
#: fails there, and the settle budget covers it so the FAIL is read off a
#: settled record.
STEP_TIMEOUT_SECONDS = 600
#: The old gate's wait, which must appear in no validation's details any more.
STALE_GATE = "missing_inventory_action_completion"
#: Node and control-plane clocks may differ by a few seconds when a host batch
#: is placed before or after the window's restore.
RESTORE_CLOCK_TOLERANCE_SECONDS = 5


def unreadable_state(ctx: VariantContext) -> dict[str, Any]:
    """Live-replay: a malformed PCI reading is UNKNOWN, not zero cards, no reboot."""

    replay = host_source_replay(ctx)
    replay.create()
    replay.build_pci(efa_count=16, malformed=True)
    opened = replay.open_pci_root(restore_seconds=ctx.spec.max_minutes * 60)
    ctx.evidence.write("pci-root-open", opened)
    started = utc_now()
    # Let the host collector poll the malformed PCI tree a few times.
    ctx.hold(ctx.host_interval * 4 + 30, name="unreadable")
    observed = activity(ctx, started, metrics=EFA_METRICS, kinds=("HOST_TELEMETRY",))
    collection_errors = [
        error
        for batch in observed.get("batches") or []
        for error in batch.get("collection_errors") or []
        if "EFA inventory" in error
    ]
    ctx.evidence.append_oracle(
        [{"efa_collection_errors": collection_errors}], source="pci-root"
    )
    missing_findings = finding_incidents(observed, "efa_inventory_mismatch")
    stages = {
        "unknown_observed": []
        if collection_errors
        else [
            "no 'EFA inventory observation unavailable' collection error was recorded"
        ],
        "no_missing_finding": [
            f"a malformed reading minted an efa_inventory_mismatch incident: {item['incident_id']}"
            for item in missing_findings
        ],
        "no_reboot": hardware_errors(observed),
        "not_isolated": ["node isolated on an UNKNOWN EFA inventory reading"]
        if node_isolated(ctx.regional.node_snapshot(ctx.settings.node))
        else [],
    }
    return _result(
        stages,
        notes=[
            "a malformed PCI vendor/device makes the EFA inventory UNAVAILABLE (UNKNOWN), not zero"
        ],
        extra={"collection_errors": collection_errors, "proof_scope": "live-replay"},
    )


# --------------------------------------------------------------------------- #
# whole-tree /sys/class/infiniband replays (HOST_SOURCE_REPLAY)
# --------------------------------------------------------------------------- #
def port_active(states: dict[str, Any]) -> bool:
    state = str(states.get("state") or "").split(":", 1)[0].strip()
    physical = str(states.get("phys_state") or "").split(":", 1)[0].strip()
    return state == "4" and physical == "5"


def _real_tree(
    ctx: VariantContext, replay: HostSourceReplay
) -> tuple[dict[str, Any], str, str, int]:
    """The node's RDMA tree as the product reads it: a device, its port, the count.

    The unmutated copy is built first so the chosen device and port are real and
    ACTIVE/LinkUp before the mutation, and the active-device count is the
    runner's own oracle of the healthy inventory.
    """

    baseline = replay.build_infiniband(source=INFINIBAND_ROOT)
    ctx.evidence.write("infiniband-baseline", baseline)
    devices = baseline.get("devices") or {}
    active = {
        name: ports
        for name, summary in devices.items()
        if any(port_active(states) for states in (summary.get("ports") or {}).values())
        for ports in [summary.get("ports") or {}]
    }
    if not active:
        raise VariantBlocked(
            "CAPABILITY_ABSENT",
            "the node lists no EFA RDMA device with an ACTIVE/LinkUp port",
        )
    device = sorted(active)[0]
    port = sorted(
        name for name, states in active[device].items() if port_active(states)
    )[0]
    return baseline, device, port, len(active)


def _expected_efa_count(ctx: VariantContext, fallback: int) -> int:
    """The count the host collector grades against, from the product's record."""

    latest = activity(ctx, utc_now(), metrics={"efa_inventory_expected_count"}).get(
        "metric_latest"
    )
    for item in latest or []:
        if item.get("name") == "efa_inventory_expected_count" and isinstance(
            item.get("value"), (int, float)
        ):
            return int(item["value"])
    return fallback


def efa_collection_errors(observed: dict[str, Any]) -> list[str]:
    return [
        str(error)
        for batch in observed.get("batches") or []
        for error in batch.get("collection_errors") or []
        if "EFA inventory" in str(error)
    ]


def _efa_samples(observed: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [
        item
        for item in samples(observed, name)
        if isinstance(item.get("value"), (int, float))
    ]


def efa_failure_mode_errors(observed: dict[str, Any], mode: str) -> list[str]:
    """The confirmed mismatch sample names the failure mode the product resolved."""

    confirmed = [
        item
        for item in _efa_samples(observed, "efa_inventory_mismatch")
        if item["value"] >= 1
    ]
    if not confirmed:
        return ["no efa_inventory_mismatch=1 sample reached the product"]
    modes = sorted(
        {(item.get("labels") or {}).get("failure_mode", "?") for item in confirmed}
    )
    return [] if modes == [mode] else [f"failure_mode {modes}, expected [{mode!r}]"]


def _isolation_errors(ctx: VariantContext, observed: dict[str, Any]) -> list[str]:
    isolation = isolation_operations(observed)
    errors = [f"isolation operations planned: {isolation}"] if isolation else []
    if node_isolated(ctx.regional.node_snapshot(ctx.settings.node)):
        errors.append("the node is isolated")
    return errors


def _validation_executions(
    observed: dict[str, Any], incidents: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "request_id": workflow.get("request_id"),
            "status": item.get("status"),
            "details": item.get("details"),
            "error": str(item.get("error") or "")[:200],
        }
        for incident in incidents
        for workflow in workflows_of(observed, str(incident["incident_id"]))
        for item in executions(workflow, "VALIDATE_FABRIC")
    ]


def stale_gate_errors(validations: list[dict[str, Any]]) -> list[str]:
    """No VALIDATE_FABRIC waited for an action completion no diagnostic step produces."""

    return [
        f"VALIDATE_FABRIC of {item.get('request_id')} waited on {STALE_GATE}: the "
        "diagnostic chain's inventory gate does not judge fresh readings"
        for item in validations
        if STALE_GATE in str(item.get("details") or "")
    ]


def fresh_evidence_errors(
    execution: dict[str, Any],
    *,
    node: str,
    expected: int | None,
    after: datetime | None,
) -> list[str]:
    """The concluded validation recorded the fresh reading it rests on.

    Batch 175 writes ``details.inventory_evidence[node][metric]`` with the
    value, the expected count and the reading's ``observed_at`` for every
    verdict of the inventory gate. For a SUCCEEDED diagnostic the reading must
    be the restored count (``expected``) observed after the restore (``after``).
    """

    details = execution.get("details") or {}
    evidence = (details.get("inventory_evidence") or {}).get(node) or {}
    reading = evidence.get("efa_inventory_active_count")
    if not isinstance(reading, dict):
        return [
            f"VALIDATE_FABRIC {execution.get('status')} without "
            f"details.inventory_evidence[{node}][efa_inventory_active_count]: no fresh "
            "reading recorded"
        ]
    errors = []
    if expected is not None and reading.get("value") != expected:
        errors.append(
            f"the validation judged efa_inventory_active_count={reading.get('value')}, "
            f"not the restored {expected}"
        )
    observed = parse_time(reading.get("observed_at"))
    if after is not None and (
        observed is None
        or observed < after - timedelta(seconds=RESTORE_CLOCK_TOLERANCE_SECONDS)
    ):
        errors.append(
            f"the validation's reading was observed {reading.get('observed_at')}, "
            f"before the restore at {after.isoformat()}"
        )
    return errors


def inactive_after(
    observed: dict[str, Any],
    moment: datetime,
    metric: str = "efa_inventory_missing_count",
) -> list[dict[str, Any]]:
    """Shortfall readings the product recorded after ``moment`` (plus tolerance)."""

    cutoff = moment + timedelta(seconds=RESTORE_CLOCK_TOLERANCE_SECONDS)
    return [
        item
        for item in _efa_samples(observed, metric)
        if item["value"] >= 1
        and (when := parse_time(item.get("observed_at"))) is not None
        and when > cutoff
    ]


def _recovered_errors(
    incidents: list[dict[str, Any]], settled: dict[str, Any]
) -> list[str]:
    """Every finding incident of the variant closed RECOVERED once settled."""

    ids = {str(item["incident_id"]) for item in incidents}
    return [
        f"incident {item.get('incident_id')} is {item.get('state')}, not RECOVERED"
        for item in settled.get("incidents") or []
        if str(item.get("incident_id")) in ids and item.get("state") != "RECOVERED"
    ]


def _await_efa_recovery(
    ctx: VariantContext, since: datetime, expected: int
) -> list[str]:
    """Fresh product evidence that the inventory reads healthy again."""

    def accept() -> dict[str, Any] | None:
        current = activity(ctx, since, metrics=EFA_METRICS)
        active = _efa_samples(current, "efa_inventory_active_count")
        mismatch = _efa_samples(current, "efa_inventory_mismatch")
        if (
            active
            and active[-1]["value"] == expected
            and mismatch
            and mismatch[-1]["value"] == 0
        ):
            return current
        return None

    value = ctx.wait_until(
        accept,
        timeout_seconds=ctx.host_summary + ctx.host_interval * 6,
        poll_seconds=ctx.host_interval,
        name="efa-recovery",
    )
    if value is None:
        return [
            f"no fresh efa_inventory_active_count={expected} with efa_inventory_mismatch=0 "
            "reached the product after the window"
        ]
    return []


def _open_tree(ctx: VariantContext, replay: HostSourceReplay) -> dict[str, Any]:
    opened = replay.open_tree(
        [(INFINIBAND_TREE, INFINIBAND_ROOT)], restore_seconds=ctx.spec.max_minutes * 60
    )
    ctx.evidence.write("source-window-open", opened)
    return opened


def unknown_inventory(ctx: VariantContext) -> dict[str, Any]:
    """An ambiguous driver identity makes the inventory UNKNOWN: no count, no reboot."""

    replay = host_source_replay(ctx)
    _baseline, device, _port, _active = _real_tree(ctx, replay)
    built = replay.build_infiniband(
        source=INFINIBAND_ROOT, driver=[(device, "ambiguous")]
    )
    ctx.evidence.write("infiniband-fixture", built)
    started = utc_now()
    _open_tree(ctx, replay)
    ctx.hold(ctx.host_interval * 4 + 30, name="unknown-inventory")
    observed = activity(ctx, started, metrics=EFA_METRICS)
    collection_errors = efa_collection_errors(observed)
    ctx.evidence.append_oracle(
        [
            {
                "efa_collection_errors": collection_errors,
                **replay.evidence(device=device),
            }
        ],
        source="infiniband",
    )
    stages = {
        "unknown_observed": []
        if any("ambiguous" in error for error in collection_errors)
        else [
            "no 'EFA inventory observation unavailable ... ambiguous RDMA driver "
            "identity' collection error was recorded"
        ],
        "no_count_reported": [
            f"{name}={item['value']:g} was reported from an unknown inventory at "
            f"{item['observed_at']}"
            for name in ("efa_inventory_active_count", "efa_inventory_mismatch")
            for item in _efa_samples(observed, name)
        ],
        "no_missing_finding": [
            f"an unknown inventory minted an efa_inventory_mismatch incident: {item['incident_id']}"
            for item in finding_incidents(observed, "efa_inventory_mismatch")
        ],
        "no_reboot": hardware_errors(observed),
        "not_isolated": _isolation_errors(ctx, observed),
    }
    return _result(
        stages,
        notes=[
            f"{device}'s device/driver identity was made ambiguous in the replayed "
            "tree only; the product's observation is UNAVAILABLE (UNKNOWN), no count"
        ],
        extra={
            "device": device,
            "collection_errors": collection_errors,
            "proof_scope": "live-replay",
        },
    )


def excess_device(ctx: VariantContext) -> dict[str, Any]:
    """One device more than the invariant: EXCESS_DEVICE diagnoses only, no reboot.

    The tree is closed the moment the finding exists. The diagnostic
    VALIDATE_FABRIC judges fresh readings (batch 175): SUCCEEDED when the next
    sample shows the restored count, FAILED when it still shows the excess --
    EXCESS_DEVICE carries no escalation, so a FAILED diagnostic ends the
    incident RECOVERED ("diagnostic inconclusive") and plans nothing. Either is
    a concluded diagnostic; PASS needs its evidence recorded, no stale gate, no
    hardware or isolation step, RECOVERED, and the healthy count after.
    """

    replay = host_source_replay(ctx)
    _baseline, device, _port, active = _real_tree(ctx, replay)
    expected = _expected_efa_count(ctx, active)
    built = replay.build_infiniband(
        source=INFINIBAND_ROOT, duplicate=[(EXCESS_DEVICE_NAME, device)]
    )
    ctx.evidence.write("infiniband-fixture", built)
    started = utc_now()
    _open_tree(ctx, replay)
    detection = wait_for_incident(
        ctx,
        started,
        "efa_inventory_mismatch",
        timeout_seconds=DETECTION_BUDGET,
        metrics=EFA_METRICS,
        name="excess-detection",
    )
    excess = [
        item
        for item in _efa_samples(detection, "efa_inventory_excess_count")
        if item["value"] >= 1
    ]
    if not excess:
        raise VariantInconclusive(
            "the product never reported efa_inventory_excess_count >= 1 from the "
            "replayed tree"
        )
    ctx.evidence.append_oracle(
        [{"excess_samples": excess[-2:], **replay.evidence(device=EXCESS_DEVICE_NAME)}],
        source="infiniband",
    )
    incidents = finding_incidents(detection, "efa_inventory_mismatch")
    restored_at = utc_now()
    ctx.evidence.write("source-window-close", replay.close())
    recovered = _await_efa_recovery(ctx, restored_at, expected)
    settled = wait_settled(
        ctx,
        started,
        [str(item["incident_id"]) for item in incidents],
        timeout_seconds=STEP_TIMEOUT_SECONDS + 300,
        metrics=EFA_METRICS,
        name="excess-settled",
    )
    observed = settled or activity(ctx, started, metrics=EFA_METRICS)
    validations = _validation_executions(observed, incidents)
    concluded = [
        item for item in validations if item.get("status") in {"SUCCEEDED", "FAILED"}
    ]
    stages = {
        "failure_mode": efa_failure_mode_errors(detection, "EXCESS_DEVICE"),
        "diagnostic_only": chain_errors(
            observed,
            ChainExpectation(
                metric="efa_inventory_mismatch",
                operations=("VALIDATE_FABRIC",),
                forbidden=HARDWARE_OPERATIONS | ISOLATION_OPERATIONS,
                single=False,
            ),
        ),
        "validation_concluded": []
        if concluded
        else [
            "the diagnostic VALIDATE_FABRIC never concluded (SUCCEEDED or FAILED on "
            f"fresh evidence): {str(validations)[:400]}"
        ],
        "fresh_evidence": [
            error
            for item in concluded[-1:]
            for error in fresh_evidence_errors(
                item, node=ctx.settings.node, expected=None, after=None
            )
        ],
        "no_stale_gate": stale_gate_errors(validations),
        "no_reboot": hardware_errors(observed),
        "not_isolated": _isolation_errors(ctx, observed),
        "incident_recovered": _recovered_errors(incidents, observed),
        "recovered": recovered,
    }
    return _result(
        stages,
        notes=[
            f"{device} duplicated as {EXCESS_DEVICE_NAME} in the replayed tree only "
            f"(active {active + 1} > expected {expected}); EXCESS_DEVICE is RUN_DIAGNOSTICS",
            "the diagnostic VALIDATE_FABRIC judges the fresh inventory reading after "
            "the chain reached it: SUCCEEDED on the restored count, or FAILED on a "
            "fresh excess reading with no escalation planned (EXCESS_DEVICE authorizes "
            "no reboot)",
        ],
        extra={
            "device": device,
            "expected_count": expected,
            "validation_executions": validations,
            "settled": _statuses(observed),
            "proof_scope": "live-replay",
        },
    )


def link_inactive(ctx: VariantContext) -> dict[str, Any]:
    """One real port DOWN/Disabled for a bounded exposure: LINK_INACTIVE diagnoses
    the restored link and concludes SUCCEEDED; no RESTART_NODE.

    The window's own deadman closes it :func:`bounded_exposure_seconds` after
    it opened -- after the collector's second tick, which confirms the
    shortfall and mints the finding, and before a third could read the port --
    so the diagnostic VALIDATE_FABRIC (judging fresh readings since batch 175)
    only ever sees the real, ACTIVE port and the designed REBOOT_NODE
    escalation has nothing fresh to fire on. The runner never races the
    product: its own close afterwards finds the window already closed.
    """

    replay = host_source_replay(ctx)
    _baseline, device, port, active = _real_tree(ctx, replay)
    expected = _expected_efa_count(ctx, active)
    built = replay.build_infiniband(
        source=INFINIBAND_ROOT,
        port_state=[(device, port, INACTIVE_STATE)],
        phys_state=[(device, port, INACTIVE_PHYS_STATE)],
    )
    ctx.evidence.write("infiniband-fixture", built)
    exposure = bounded_exposure_seconds(ctx.host_interval)
    started = utc_now()
    opened = replay.open_tree(
        [(INFINIBAND_TREE, INFINIBAND_ROOT)], restore_seconds=exposure
    )
    ctx.evidence.write("source-window-open", opened)
    restore_at = utc_now() + timedelta(seconds=exposure)
    detection = wait_for_incident(
        ctx,
        started,
        "efa_inventory_mismatch",
        timeout_seconds=exposure + ctx.host_interval * 3 + 60,
        metrics=EFA_METRICS,
        name="link-inactive-detection",
    )
    incidents = finding_incidents(detection, "efa_inventory_mismatch")
    missing = [
        item
        for item in _efa_samples(detection, "efa_inventory_missing_count")
        if item["value"] >= 1
    ]
    # The deadman has closed the window by now; the runner's close confirms it
    # (already_closed) or, should the timer have been late, closes it itself.
    remaining = (restore_at - utc_now()).total_seconds()
    if remaining > 0:
        ctx.hold(remaining + RESTORE_CLOCK_TOLERANCE_SECONDS, name="exposure")
    closed = replay.close()
    closed_by = "deadman" if closed.get("already_closed") else "runner"
    ctx.evidence.write("source-window-close", {**closed, "closed_by": closed_by})
    ctx.evidence.append_oracle(
        [
            {
                "inactive_port": f"{device}/ports/{port}",
                "exposure_seconds": exposure,
                "restore_at": restore_at.isoformat(),
                "closed_by": closed_by,
                **replay.evidence(device=device),
            }
        ],
        source="infiniband",
    )
    if not incidents or not missing:
        raise VariantInconclusive(
            f"the bounded exposure ({exposure} s) ended before the product confirmed "
            "the inactive port: no efa_inventory_mismatch finding (the collector's "
            "two confirming ticks did not both fall inside the window)"
        )
    recovered = _await_efa_recovery(ctx, restore_at, expected)
    settled = wait_settled(
        ctx,
        started,
        [str(item["incident_id"]) for item in incidents],
        timeout_seconds=STEP_TIMEOUT_SECONDS + 300,
        metrics=EFA_METRICS,
        name="link-inactive-settled",
    )
    observed = settled or activity(ctx, started, metrics=EFA_METRICS)
    late = inactive_after(observed, restore_at)
    if late:
        raise VariantInconclusive(
            f"the exposure bound did not hold: {len(late)} efa_inventory_missing_count "
            f"reading(s) after the restore at {restore_at.isoformat()} (first "
            f"{late[0]['observed_at']}); the product's designed escalation on fresh "
            "inactive evidence applies and is not a verdict on it"
        )
    first_workflows = [
        workflows_of(observed, str(item["incident_id"]))[:1] for item in incidents
    ]
    diagnose_first = []
    for found in first_workflows:
        if not found:
            diagnose_first.append("the LINK_INACTIVE incident has no workflow")
            continue
        steps = operations(found[0])
        if "VALIDATE_FABRIC" not in steps:
            diagnose_first.append(f"the first workflow {steps} does not diagnose")
        banned = sorted(set(steps) & (HARDWARE_OPERATIONS | ISOLATION_OPERATIONS))
        if banned:
            diagnose_first.append(
                f"the first workflow plans {banned} before diagnosing"
            )
    # Only the diagnostic (first) workflow may conclude the diagnosis: a
    # SUCCEEDED VALIDATE_FABRIC inside a later reboot chain is not proof.
    concluded = [
        item
        for found in first_workflows
        for workflow in found
        for item in executions(workflow, "VALIDATE_FABRIC")
        if item.get("status") == "SUCCEEDED"
    ]
    validations = _validation_executions(observed, incidents)
    stages = {
        "failure_mode": efa_failure_mode_errors(detection, "LINK_INACTIVE"),
        "diagnose_first": diagnose_first,
        "validation_concluded": []
        if concluded
        else [
            "VALIDATE_FABRIC never SUCCEEDED on the restored link: "
            + str(validations)[:400]
        ],
        "fresh_evidence": [
            error
            for item in concluded[-1:]
            for error in fresh_evidence_errors(
                item, node=ctx.settings.node, expected=expected, after=restore_at
            )
        ],
        "no_stale_gate": stale_gate_errors(validations),
        "no_reboot_after_recovery": hardware_errors(observed),
        "not_isolated": _isolation_errors(ctx, observed),
        "incident_recovered": _recovered_errors(incidents, observed),
        "recovered": recovered,
    }
    return _result(
        stages,
        notes=[
            f"{device}/ports/{port} read {INACTIVE_STATE!r}/{INACTIVE_PHYS_STATE!r} in the "
            f"replayed tree for a bounded exposure of {exposure} s (closed by "
            f"{closed_by}); the real port stayed up",
            "rdma_link_down co-fires from the same port (010 link_inactive semantics)",
        ],
        extra={
            "device": device,
            "port": port,
            "expected_count": expected,
            "exposure_seconds": exposure,
            "restore_at": restore_at.isoformat(),
            "closed_by": closed_by,
            "validation_executions": validations,
            "settled": _statuses(observed),
            "proof_scope": "live-replay",
        },
    )


def _statuses(activity_document: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "request_id": item.get("request_id"),
            "status": item.get("status"),
            "operations": operations(item),
        }
        for item in activity_document.get("workflows") or []
    ]


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
        **(extra or {}),
    }


def _tree_variant(
    variant: str,
    *,
    injection: str,
    expected: str,
    execute: Any,
    risk: str = "L",
    flags: str = "S",
    max_minutes: int = 14,
    approvals: tuple[str, ...] = (HOST_SOURCE_REPLAY,),
    expected_outcome: str = "PASS",
    product_gap: str = "",
    cleanup: str = "close the /sys/class/infiniband tree window (deadman-backed); confirm clean",
) -> VariantSpec:
    return VariantSpec(
        case_id=CASE_013,
        variant=variant,
        proof_mode=LIVE_REPLAY,
        lanes=("eks", "hp"),
        risk=risk,
        flags=flags,
        injection=injection,
        expected=expected,
        cleanup=cleanup,
        max_minutes=max_minutes,
        approvals=approvals,
        expected_outcome=expected_outcome,
        product_gap=product_gap,
        execute=execute,
    )


_PLUGIN_NATIVE = (
    "NOT_AUTHORIZED: a real kubelet device-plugin registration fault needs the "
    "DEVICE_PLUGIN_RESTART approval and a plugin pod-delete/re-register injection with "
    "new-Pod-UID verification; patching Node.status or faking allocatable is not valid "
    "proof (spec 014). Not built in this lane."
)

VARIANTS = [
    # --- 013 EFA inventory --- #
    VariantSpec(
        case_id=CASE_013,
        variant="unreadable_state",
        proof_mode=LIVE_REPLAY,
        lanes=("eks", "hp"),
        risk="L",
        flags="S",
        injection="repoint GPU_FAULT_PCI_DEVICES_ROOT at a reviewed tree with a malformed EFA vendor/device",
        expected="EFA inventory observation UNAVAILABLE (UNKNOWN); no missing-device finding, no reboot, no isolation",
        cleanup="close the PCI-root window; confirm clean",
        max_minutes=14,
        approvals=("KERNEL_SOURCE_REPLAY",),
        execute=unreadable_state,
    ),
    # 013 pci_missing (the PCI_DEVICE_MISSING -> REBOOT_NODE chain, with the
    # unreadable-root negative stage) lives in general_live_efa_pci.py.
    deferred_variant(
        CASE_013,
        "driver_unbound",
        proof_mode=LIVE_REPLAY,
        reason=(
            "NOT_AUTHORIZED: the reviewed targeted EFA driver unbind needs the reserved "
            "EFA_DRIVER_UNBIND native probe and an independent rebind deadman, not built this phase"
        ),
        lanes=("eks",),
        expected="DRIVER_UNBOUND -> targeted bind + plugin restart + snapshot + validation; not a reboot",
    ),
    _tree_variant(
        "link_inactive",
        injection=(
            "bind a copy of the node's /sys/class/infiniband with one real port's "
            "state/phys_state '1: DOWN'/'3: Disabled' into the host collector unit for "
            "a bounded exposure: the window's deadman closes it 1.7 collector intervals "
            "after it opened (two confirming ticks, never a third)"
        ),
        expected=(
            "efa_inventory_mismatch failure_mode=LINK_INACTIVE -> RUN_DIAGNOSTICS: "
            "FREEZE_EVIDENCE + VALIDATE_FABRIC first, no isolation; the validation "
            "SUCCEEDS on the fresh restored reading (details.inventory_evidence), never "
            "waits on missing_inventory_action_completion; the incident is RECOVERED; "
            "no RESTART_NODE; the healthy count reads back"
        ),
        execute=link_inactive,
        risk="M",
        max_minutes=30,
        cleanup=(
            "the window closes itself at the exposure bound (deadman); the runner's "
            "close confirms it and the diagnostic is waited to settle, never cancelled"
        ),
    ),
    _tree_variant(
        "excess_device",
        injection=(
            f"bind a copy of the node's /sys/class/infiniband with one real device "
            f"duplicated as {EXCESS_DEVICE_NAME} into the host collector unit; closed "
            "the moment the finding exists"
        ),
        expected=(
            "efa_inventory_mismatch failure_mode=EXCESS_DEVICE -> RUN_DIAGNOSTICS with "
            "VALIDATE_FABRIC concluded on fresh evidence (SUCCEEDED on the restored "
            "count or FAILED on a fresh excess reading, details.inventory_evidence "
            "recorded, never missing_inventory_action_completion); no escalation, no "
            "reboot, no isolation; the incident RECOVERED; healthy count after"
        ),
        execute=excess_device,
        max_minutes=25,
    ),
    _tree_variant(
        "unknown_inventory",
        injection=(
            "bind a copy of the node's /sys/class/infiniband with one device's "
            "device/driver identity made ambiguous (two DRIVER= lines, no symlink)"
        ),
        expected=(
            "'EFA inventory observation unavailable: missing or ambiguous RDMA driver "
            "identity' collection error; no EFA count or mismatch sample, no finding, "
            "no reboot, no isolation"
        ),
        execute=unknown_inventory,
    ),
    deferred_variant(
        CASE_013,
        "bind_failure",
        proof_mode=LIVE_REPLAY,
        reason=(
            "NOT_AUTHORIZED: a bind that fails after an unbind needs the reserved "
            "EFA_DRIVER_UNBIND native path"
        ),
        lanes=("eks",),
        expected="a failed bind has its own receipt and is handled per the approved branch; no global efa unload",
    ),
    deferred_variant(
        CASE_013,
        "post_repair_failure",
        proof_mode=LIVE_REPLAY,
        reason=(
            "NOT_AUTHORIZED: a post-action validation failure needs the reserved "
            "EFA_DRIVER_UNBIND native path plus an injected post-repair failure"
        ),
        lanes=("eks",),
        expected="action succeeds but post-action validation fails -> independent receipt and escalation",
    ),
    # --- 014 Kubernetes allocatable under-report (no valid replay) --- #
    deferred_variant(
        CASE_014,
        "gpu_plugin",
        proof_mode=LIVE_REPLAY,
        reason=_PLUGIN_NATIVE,
        lanes=("eks", "hp"),
        expected="a real GPU device-plugin registration shortfall -> restart the correct plugin; new Pod UID; resources re-registered",
    ),
    deferred_variant(
        CASE_014,
        "efa_plugin",
        proof_mode=LIVE_REPLAY,
        reason=_PLUGIN_NATIVE,
        lanes=("eks", "hp"),
        expected="a real EFA device-plugin registration shortfall -> restart the correct plugin",
    ),
    deferred_variant(
        CASE_014,
        "one_sample",
        proof_mode=LIVE_REPLAY,
        reason=_PLUGIN_NATIVE,
        lanes=("eks",),
        expected="a single allocatable mismatch is not confirmed into an action",
    ),
    deferred_variant(
        CASE_014,
        "replacement_uid",
        proof_mode=LIVE_REPLAY,
        reason=_PLUGIN_NATIVE,
        lanes=("eks",),
        expected="after the plugin restart the new Pod UID and re-Ready are verified",
    ),
    deferred_variant(
        CASE_014,
        "expected_unknown",
        proof_mode=LIVE_REPLAY,
        reason=(
            "CAPABILITY_ABSENT: p5en.48xlarge has a known expected allocatable count; an unknown "
            "expected count needs an instance type absent from INSTANCE_ACCELERATOR_COUNTS"
        ),
        lanes=("eks",),
        expected="an unknown expected count is taken for forensics, not treated as a physical shortfall",
    ),
    deferred_variant(
        CASE_014,
        "mig_timeslicing",
        proof_mode=LIVE_REPLAY,
        reason="CAPABILITY_ABSENT: MIG / time-slicing is not configured on this site",
        lanes=("eks",),
        expected="MIG/time-slicing allocatable is not treated as a physical card shortfall",
    ),
    deferred_variant(
        CASE_014,
        "validation_failure",
        proof_mode=LIVE_REPLAY,
        reason=_PLUGIN_NATIVE,
        lanes=("eks",),
        expected="a post-restart validation failure escalates to reboot under a separate authorization",
    ),
    deferred_variant(
        CASE_014,
        "running_training",
        proof_mode=LIVE_REPLAY,
        reason=(
            "FIXTURE_NOT_READY: an allocatable mismatch while training runs needs a managed (scale) "
            "training attempt on the node (lane N territory), not built here"
        ),
        lanes=("eks", "hp"),
        expected="a mismatch while training runs keeps the training identity and valid progress",
    ),
]
