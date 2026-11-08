"""GF-GENERAL-LIVE-013 ``post_repair_isolation_lifted``: a reboot whose
isolation was lifted mid-chain is handed over, not climbed (b265, 82f7b77).

This reuses ``post_repair_failure``'s chain wholesale: the on-node
``probes/general_live_efa_probe.py`` trap re-unbinds one allow-listed EFA data
function after the product's ``REMEDIATE_EFA_DRIVER`` bind, on the product's own
post-repair snapshot (the host-collector restart), so ``VALIDATE_FABRIC`` fails
on fresh evidence and the ladder opens exactly one ``workflow-reboot-after``
successor whose plan is FREEZE_EVIDENCE, MARK_UNSCHEDULABLE, QUARANTINE,
RESTART_NODE, the validations, RESTORE_SCHEDULING (``escalation
._escalation_operations``). The fixture then lifts *only the cordon* of the case
node (``spec.unschedulable=false``, the equivalent of ``kubectl uncordon``; the
quarantine taint, the incident annotation and the fencing token are never
written) inside that successor's isolation window, so the node lifecycle adapter
reads it uncordoned for RESTART_NODE and refuses before the preflight, the agent
revocation and the provider call (``adapters/hyperpod/lifecycle
._observe_isolation`` -> ``_isolation_problems``: ``spec.unschedulable is not
true``). R24 (b265, 82f7b77) marks that refusal ``isolation_not_observed =
NOT_ISOLATED`` and the classifier (``failure_outcome.lifecycle_isolation_refusals``
-> ``escalation._classify``, ``CONTAINMENT_STAGE``) hands the node over at
``containment_or_release`` -- FREEZE_EVIDENCE, MARK_UNSCHEDULABLE, QUARANTINE (the
isolation put back), ESCALATE_SUPPORT (exactly one ticket naming the cordon), no
rung: no REPLACE_NODE, no reboot, no warm spare.

The lift is ``general_live_cordon_lift`` (S74), which carries the window
analysis: QUARANTINE re-writes the cordon unseen, the window from its write to
the lifecycle adapter's read is ~0.3-1.5 s, and a ``kubectl`` call costs ~1.5 s,
so S72's polling hold always landed late (R24 L72). The S74 lift watches the
node over one pre-authenticated client, provokes QUARANTINE into a visible
re-cordon with a first lift on the successor's MARK_UNSCHEDULABLE, and lifts
again the moment that re-cordon arrives: at most two cordon-only writes, each a
JSON patch that tests the successor's annotation, its fencing token and the
cordon, with a restore registered before the first. Detection is read every
:data:`DETECTION_POLL_SECONDS` (not the shared 20 s), so the hold is armed well
before the successor appears (the source chain takes ~20 s).

Verdicts. PASS when the reboot-after RESTART_NODE was refused for want of
observed isolation and handed over at ``containment_or_release``. FAIL when the
product did what the hand-off forbids (a climb to REPLACE_NODE, the wrong stage,
a reboot after the refusal) or when the harness overreached (a write other than
the cordon, more than two lifts, a write under another incident or after the
hold stopped: ``general_live_cordon_lift.overreach_errors``). Otherwise
INCONCLUSIVE, its detail opening with a reason token read off the product's
records and the hold's log: the lift races in :data:`RACE_REASONS`
(``NOT_PROVEN_RACE``) and the outcomes that are not the race in
:data:`OTHER_REASONS`. A miss is a clean reboot (the ``post_repair_failure``
exit, graded across the boot by ``EfaDriverFixture.release``).

The variant is generation-gated: without 82f7b77 in the running control plane it
is NOT_RUN (``require_isolation_refusal_generation``, a read-only probe in an
API Pod, the way the device-plugin and XID-twin generation gates do). It reuses
``EfaDriverFixture`` and ``post_repair_failure``'s settle/trap/cleanup, so it is
registered in its own module (``general_live_efa_driver`` owns exactly the two
bind negatives) and adds nothing under ``src/`` or ``deploy/``.
"""

from __future__ import annotations

import json
from typing import Any

from scripts.e2e.regional.general_live_common import wait_for_incident
from scripts.e2e.regional.general_live_cordon_lift import (
    MAX_LIFTS,
    T_Q_SECONDS,
    LiftPlan,
    decisive_lift,
    hold_cordon_lifted,
    open_cordon_lifter,
    overreach_errors,
    rv_position,
    successor_incident_id,
    successor_workflow_id,
)
from scripts.e2e.regional.general_live_efa_driver import (
    DETECTION_BUDGET,
    EFA_DRIVER_UNBIND,
    EFA_METRICS,
    NODE_REBOOT,
    TRAP_FLOOR_SECONDS,
    TRAP_GRACE_SECONDS,
    CASE_013,
    EfaDriverFixture,
    _escalation_extra,
    _expected_efa_count,
    _incidents_on_node,
    _oracle_no_global_unload,
    _result,
    _settle,
    hand_back,
)
from scripts.e2e.regional.general_live_efa_escalation import (
    EFA_METRIC,
    MARK_UNSCHEDULABLE,
    QUARANTINE,
    RESTART_NODE,
    containment_release_handoff_errors,
    efa_chain,
    isolation_refusal_errors,
    reboot_successors,
    refused_restart_step,
    warm_spare_touched,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
    utc_now,
)
from scripts.e2e.regional.general_live_registry import NATIVE, VariantSpec
from scripts.e2e.regional.general_live_verdicts import finding_incidents, parse_time

#: The deployed control plane's answer to "does this engine classify a reboot
#: refused for want of observed isolation as a hand-off, not a climb?" --
#: read-only inside an API Pod, imports only (no mutation, no node touched).
GENERATION_PROBE = r"""
import json

result = {"present": False}
try:
    from gpu_fault.orchestration import failure_outcome as fo
except ImportError as exc:
    result["error"] = f"{type(exc).__name__}: {exc}"[:200]
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0)

marks = {
    "lifecycle_isolation_refusals": callable(
        getattr(fo, "lifecycle_isolation_refusals", None)
    ),
    "lifecycle_isolation_refused": callable(
        getattr(fo, "lifecycle_isolation_refused", None)
    ),
    "LIFECYCLE_ISOLATION_DETAIL": getattr(fo, "LIFECYCLE_ISOLATION_DETAIL", None),
    "ISOLATION_NOT_ISOLATED": getattr(fo, "ISOLATION_NOT_ISOLATED", None),
    "ISOLATION_UNOBSERVABLE": getattr(fo, "ISOLATION_UNOBSERVABLE", None),
}
try:
    from gpu_fault.orchestration import escalation as esc

    marks["classifier_wired"] = "lifecycle_isolation_refusals" in dir(esc)
    marks["containment_stage"] = getattr(esc, "CONTAINMENT_STAGE", None)
except Exception as exc:  # noqa: BLE001 - reported, the gate decides
    marks["classifier_error"] = f"{type(exc).__name__}: {exc}"[:200]

result["present"] = bool(
    marks["lifecycle_isolation_refusals"]
    and marks["LIFECYCLE_ISOLATION_DETAIL"] == "isolation_not_observed"
    and marks["ISOLATION_NOT_ISOLATED"] == "NOT_ISOLATED"
    and marks.get("classifier_wired")
    and marks.get("containment_stage") == "containment_or_release"
)
result["marks"] = marks
print(json.dumps(result, sort_keys=True, default=str))
"""

#: The trap window reused from ``post_repair_failure``.
TRAP_WINDOW = DETECTION_BUDGET + 180
#: The pause between detection reads (each read is a ~5 s exec into an API
#: Pod). The shared 20 s pause could arm the hold after the successor appeared.
DETECTION_POLL_SECONDS = 2
WINDOW_EVIDENCE = "lifted-window"

#: INCONCLUSIVE reasons that are the lift race (``NOT_PROVEN_RACE``); every
#: INCONCLUSIVE detail of the grading opens with ``<reason>: ``.
RACE_REASONS = {
    "LIFT_LATE": (
        "RESTART_NODE's isolation read saw the cordon the chain left: no decisive "
        "lift had landed yet"
    ),
    "LIFT_UNDONE": (
        "a cordon was put back after the lift and before RESTART_NODE's isolation "
        "read (QUARANTINE re-cordoned the provocation and the decisive re-lift did "
        "not land first, or another write followed the decisive lift)"
    ),
    "REPLAY_WITHOUT_OBSERVE": (
        "RESTART_NODE's first run ended in an executor hold or retry and its "
        "replay resumed it as a submitted reboot without reading isolation (b272)"
    ),
    "NO_RUNG": "no reboot-after rung opened: no RESTART_NODE met the lifted cordon",
}
#: INCONCLUSIVE reasons that are not the race: a harness or product outcome to
#: read (the validation driver grades them FAIL).
OTHER_REASONS = {
    "CONTAINMENT_FAILED": (
        "the successor's MARK_UNSCHEDULABLE or QUARANTINE FAILED, which a cordon "
        "lift cannot cause"
    ),
    "RUNG_UNRESOLVED": (
        "the successor's RESTART_NODE neither refused for want of observed "
        "isolation nor passed its isolation read"
    ),
    "HOLD_NOT_ARMED": "the hold could not name the successor; nothing was lifted",
}
#: A FAILED RESTART_NODE whose error names a stage after the isolation read.
POST_OBSERVE_ERRORS = (
    "preflight failed",
    "preflight cannot run",
    "rejected nodes",
    "failed to revoke node agent",
)
#: Details of an executor hold or retry before the adapter (or the read) ran.
REPLAY_MARKERS = (
    "fleet_preflight_blocked",
    "lease_guard_blocked",
    "multi_node_barrier_unavailable",
    "retryable_transport_error",
    "retryable_adapter_error",
    "retryable_control_plane_error",
)


def reason(token: str, text: str) -> str:
    """An INCONCLUSIVE detail: ``<token>: <text>``."""

    if token not in RACE_REASONS and token not in OTHER_REASONS:
        raise ValueError(f"unknown INCONCLUSIVE reason {token!r}")
    return f"{token}: {text}"


def require_isolation_refusal_generation(ctx: VariantContext) -> dict[str, Any]:
    """NOT_RUN unless the running control plane carries b265's classifier.

    The answer is the evidence ``product-generation``. On a pre-82f7b77 engine
    the refusal is a bare ``safety_rejection`` that climbs to REPLACE_NODE, so
    the lifted cordon would consume a warm spare -- a fault of the old engine,
    not of this variant; nothing is deployed and the node is not touched.
    """

    answer = ctx.regional.cpu_python(GENERATION_PROBE, timeout=120)
    answer = dict(answer) if isinstance(answer, dict) else {"present": False}
    ctx.evidence.write("product-generation", answer)
    if not answer.get("present"):
        detail = answer.get("marks") or answer.get("error") or answer
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            "the deployed control plane does not classify a reboot refused for want "
            "of observed isolation as a containment_or_release hand-off (b265, "
            f"82f7b77 absent): {json.dumps(detail, sort_keys=True, default=str)}. "
            "Nothing was deployed and the node was not touched.",
        )
    return answer


def lift_plan(detection: dict[str, Any], *, node: str, baseline_boot: str) -> LiftPlan:
    """The hold's plan from the detection read: the EFA finding's incident and
    workflow name the one successor the hold may lift."""

    pair = efa_chain(detection)
    found = finding_incidents(detection, EFA_METRIC)
    incident = pair[0] if pair else (found[0] if found else {})
    workflow_id = str(
        (pair[1].get("request_id") if pair else incident.get("workflow_request_id"))
        or ""
    )
    if not workflow_id:
        raise VariantInconclusive(
            reason(
                "HOLD_NOT_ARMED",
                "the EFA finding's workflow is not on record in the detection read, "
                "so the reboot-after successor cannot be named; nothing was lifted",
            )
        )
    return LiftPlan(
        node=node,
        source_workflow_id=workflow_id,
        source_incident_id=str(incident.get("incident_id") or ""),
        baseline_boot=baseline_boot,
    )


def post_repair_isolation_lifted(ctx: VariantContext) -> dict[str, Any]:
    """Reuse ``post_repair_failure``'s chain, lift the reboot successor's cordon
    into its QUARANTINE -> RESTART_NODE window, and grade the R24 hand-off."""

    require_isolation_refusal_generation(ctx)
    incidents: list[dict[str, Any]] = []
    ctx.cleanup.push("validated-restore", lambda: hand_back(ctx, incidents))
    fixture = EfaDriverFixture(ctx)
    survey = fixture.survey()
    bdf = fixture.target_bdf()
    node = ctx.settings.node
    baseline_bound = int(survey.get("bound_count") or 0)
    baseline_boot = str(survey.get("boot_id") or "")
    expected = _expected_efa_count(ctx, baseline_bound)
    fixture.expected_count = expected
    # The one client before anything is injected: a TLS or credential failure
    # is a harness failure on an untouched node.
    lifter = open_cordon_lifter(ctx)
    started = utc_now()
    injected = fixture.trap_unbind(
        restore_seconds=ctx.spec.max_minutes * 60,
        grace_seconds=TRAP_GRACE_SECONDS,
        floor_seconds=TRAP_FLOOR_SECONDS,
        trap_window_seconds=TRAP_WINDOW,
    )
    oracle = [fixture.summary()]
    detection = wait_for_incident(
        ctx,
        started,
        EFA_METRIC,
        timeout_seconds=DETECTION_BUDGET,
        metrics=EFA_METRICS,
        name="lifted-detection",
        poll_seconds=DETECTION_POLL_SECONDS,
    )
    if not finding_incidents(detection, EFA_METRIC):
        oracle.append(fixture.summary_or_loss())
        ctx.evidence.append_oracle(
            [{"pci_bdf": bdf, "injection": injected, "readings": oracle}],
            source="efa-driver",
        )
        raise VariantInconclusive(
            f"the product never minted an efa_inventory_mismatch finding from the "
            f"unbound function {bdf}"
        )
    incidents[:] = _incidents_on_node(detection, node)
    plan = lift_plan(detection, node=node, baseline_boot=baseline_boot)
    lift = hold_cordon_lifted(ctx, lifter, plan, window=TRAP_WINDOW + 120)
    recovered, observed, node_now, pair, settled = _settle(
        ctx,
        started=started,
        expected=expected,
        baseline_boot=baseline_boot,
        rung_name="lifted-rung",
    )
    incidents[:] = _incidents_on_node(observed, node) or incidents
    oracle.append(fixture.summary_or_loss())
    window = window_evidence(observed, lift, node=node)
    ctx.evidence.write(WINDOW_EVIDENCE, window)
    ctx.evidence.append_oracle(
        [
            {
                "pci_bdf": bdf,
                "injection": injected,
                "readings": oracle,
                "cordon_lift": lift_summary(lift),
            }
        ],
        source="efa-driver",
    )
    return grade_isolation_lifted(
        observed=observed,
        pair=pair,
        node_now=node_now,
        settled=settled,
        fixture=fixture,
        bdf=bdf,
        expected=expected,
        baseline_boot=baseline_boot,
        lift=lift,
        oracle=oracle,
        window=window,
    )


# --------------------------------------------------------------------------- #
# what the product's records say about the window
# --------------------------------------------------------------------------- #
def _executions(
    workflow: dict[str, Any] | None, operation: str
) -> list[dict[str, Any]]:
    return [
        item
        for item in (workflow or {}).get("step_executions") or []
        if isinstance(item, dict) and item.get("operation") == operation
    ]


def _for_node(mapping: Any, node: str) -> dict[str, Any]:
    """``mapping[node]``, or the entry whose ``kubernetes_node`` is ``node``."""

    if not isinstance(mapping, dict):
        return {}
    if isinstance(mapping.get(node), dict):
        return dict(mapping[node])
    for value in mapping.values():
        if isinstance(value, dict) and value.get("kubernetes_node") == node:
            return dict(value)
    return {}


def quarantine_baseline(workflow: dict[str, Any] | None, node: str) -> dict[str, Any]:
    """QUARANTINE's recorded ``node_baselines`` for ``node`` (``before`` read
    inside its patch, ``after`` re-read): ``before.unschedulable`` false means it
    re-cordoned a lifted node."""

    for item in reversed(_executions(workflow, QUARANTINE)):
        baseline = _for_node((item.get("details") or {}).get("node_baselines"), node)
        if baseline:
            return baseline
    return {}


def isolation_read(workflow: dict[str, Any] | None, node: str) -> dict[str, Any]:
    """What RESTART_NODE's isolation read recorded for ``node`` (only a read
    that passed records it: ``observed_isolation``)."""

    for item in _executions(workflow, RESTART_NODE):
        observed = _for_node(
            (item.get("details") or {}).get("observed_isolation"), node
        )
        if observed:
            return observed
    return {}


def observe_passed(restarts: list[dict[str, Any]], *, rebooted: bool) -> bool:
    """Whether RESTART_NODE got past its isolation read: it recorded the read,
    submitted, succeeded, rebooted the node, or failed at a later stage."""

    if rebooted:
        return True
    for item in restarts:
        details = item.get("details") or {}
        if (
            item.get("status") == "SUCCEEDED"
            or details.get("observed_isolation")
            or details.get("submitted_nodes")
        ):
            return True
        error = str(item.get("error") or "")
        if item.get("status") == "FAILED" and any(
            text in error for text in POST_OBSERVE_ERRORS
        ):
            return True
    return False


def replay_markers(restarts: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            key
            for item in restarts
            for key in REPLAY_MARKERS
            if (item.get("details") or {}).get(key)
        }
    )


def _hold_text(lift: dict[str, Any]) -> str:
    return (
        f"hold stopped {lift.get('stopped')!r} after "
        f"{len(lift.get('lifts') or [])} lift(s)"
    )


def classify_unrefused(
    reboot_after: dict[str, Any],
    *,
    node: str,
    lift: dict[str, Any],
    rebooted: bool,
    boot_change: str,
) -> tuple[str, str]:
    """The INCONCLUSIVE reason of a reboot rung whose RESTART_NODE was not
    refused for want of observed isolation, read off the records."""

    steps = reboot_after.get("step_executions") or []
    if any(
        item.get("operation") in (QUARANTINE, MARK_UNSCHEDULABLE)
        and item.get("status") == "FAILED"
        for item in steps
    ):
        return (
            "CONTAINMENT_FAILED",
            "the reboot-after successor's MARK_UNSCHEDULABLE or QUARANTINE FAILED; "
            "neither verifies the cordon and a conflict ends WAITING, so a "
            "cordon-only lift cannot cause it -- read the step's error; the "
            "RESTART_NODE refusal was not exercised",
        )
    restarts = _executions(reboot_after, RESTART_NODE)
    if observe_passed(restarts, rebooted=rebooted):
        read = isolation_read(reboot_after, node)
        baseline = quarantine_baseline(reboot_after, node)
        before, after = baseline.get("before") or {}, baseline.get("after") or {}
        version = str(read.get("resource_version") or "")
        evidence = (
            f"(isolation read rv {version or 'not recorded'}; QUARANTINE rv "
            f"{before.get('resource_version')} -> {after.get('resource_version')}, "
            f"unschedulable before {before.get('unschedulable')}; {_hold_text(lift)})"
        )
        if rebooted:
            outcome = (
                f"; the node rebooted ({boot_change}): the post_repair_failure exit, "
                "graded clean across the boot by the reboot-aware cleanup"
            )
        else:
            outcome = (
                "; RESTART_NODE submitted to the provider: a reboot is in progress "
                "(reboot-aware cleanup)"
            )
        if before.get("unschedulable") is False:
            return (
                "LIFT_UNDONE",
                "the provocation landed before QUARANTINE's write and QUARANTINE "
                "re-cordoned the node; the decisive re-lift did not land before "
                f"RESTART_NODE's isolation read {evidence}{outcome}",
            )
        if rv_position(lift, version) == "after":
            return (
                "LIFT_UNDONE",
                "a cordon was put back after the decisive lift and before "
                f"RESTART_NODE's isolation read {evidence}{outcome}",
            )
        return (
            "LIFT_LATE",
            "RESTART_NODE ran before the cordon lift: its isolation read saw the "
            f"cordon the chain left {evidence}{outcome}; the lifted-cordon refusal "
            "was not exercised",
        )
    markers = replay_markers(restarts)
    if markers:
        return (
            "REPLAY_WITHOUT_OBSERVE",
            f"RESTART_NODE's first run ended in an executor hold or retry "
            f"({', '.join(markers)}) and its replay resumes it as a submitted reboot "
            "without reading isolation again (b272): the lifted cordon was never "
            f"read ({_hold_text(lift)})",
        )
    last = restarts[-1] if restarts else {}
    return (
        "RUNG_UNRESOLVED",
        "the reboot-after RESTART_NODE neither refused for want of observed "
        f"isolation nor passed its isolation read (status {last.get('status')!r}, "
        f"error {str(last.get('error') or '')[:160]!r}; {_hold_text(lift)})",
    )


def _first_event(workflow: dict[str, Any] | None, operation: str, code: str) -> Any:
    for event in (workflow or {}).get("events") or []:
        if (
            isinstance(event, dict)
            and event.get("operation") == operation
            and event.get("code") == code
        ):
            return event.get("at")
    return None


def _milliseconds(start: Any, end: Any) -> float | None:
    try:
        begin, finish = parse_time(start), parse_time(end)
    except ValueError:
        return None
    if begin is None or finish is None:
        return None
    return round((finish - begin).total_seconds() * 1000, 3)


def _first(items: Any, key: str) -> Any:
    for item in items or []:
        if isinstance(item, dict) and item.get(key):
            return item.get(key)
    return None


def window_evidence(
    observed: dict[str, Any], lift: dict[str, Any], *, node: str
) -> dict[str, Any]:
    """Where the window was, after the fact: the successor's step records
    (control-plane clock), the hold's writes (harness clock) and the executor's
    RESTART_NODE claim line (the GPU node's runtime clock)."""

    successor_id = str(lift.get("successor_workflow") or "") or (
        successor_workflow_id(str(lift.get("source_workflow") or ""))
    )
    successor = next(
        (
            item
            for item in observed.get("workflows") or []
            if str(item.get("request_id")) == successor_id
        ),
        None,
    )
    claims = lift.get("executor_claims") or {}
    decisive = decisive_lift(lift) or {}
    provocation: dict[str, Any] = next(iter(lift.get("lifts") or []), {})
    marks = {
        "mark_unschedulable_succeeded_at": _first_event(
            successor, MARK_UNSCHEDULABLE, "STEP_SUCCEEDED"
        ),
        "quarantine_succeeded_at": _first_event(
            successor, QUARANTINE, "STEP_SUCCEEDED"
        ),
        "restart_node_minted_at": _first_event(successor, RESTART_NODE, "STEP_WAITING"),
        "restart_node_failed_at": _first_event(successor, RESTART_NODE, "STEP_FAILED"),
        "provocation_acked_at": provocation.get("acked_at"),
        "decisive_acked_at": decisive.get("acked_at"),
        "executor_claimed_at": _first(claims.get("claims"), "at"),
        "executor_finished_at": _first(claims.get("finishes"), "at"),
        "executor_finished_status": _first(claims.get("finishes"), "status"),
    }
    deltas = {
        "decisive_after_quarantine_ms": _milliseconds(
            marks["quarantine_succeeded_at"], marks["decisive_acked_at"]
        ),
        "decisive_after_mint_ms": _milliseconds(
            marks["restart_node_minted_at"], marks["decisive_acked_at"]
        ),
        "claim_after_mint_ms": _milliseconds(
            marks["restart_node_minted_at"], marks["executor_claimed_at"]
        ),
        "decisive_after_claim_ms": _milliseconds(
            marks["executor_claimed_at"], marks["decisive_acked_at"]
        ),
        "finish_after_claim_ms": _milliseconds(
            marks["executor_claimed_at"], marks["executor_finished_at"]
        ),
    }
    return {
        "node": node,
        "successor_workflow": successor_id,
        "marks": marks,
        "deltas_ms": deltas,
        "clocks": "control plane (step records), harness host (lifts), GPU node "
        "runtime (claim lines): NTP-synchronised, not one clock",
        "restart_isolation_read": isolation_read(successor, node),
        "quarantine_baseline": quarantine_baseline(successor, node),
        "latency": lift.get("latency"),
    }


def lift_summary(lift: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": lift.get("schema"),
        "stopped": lift.get("stopped"),
        "roles": lift.get("roles"),
        "lifts": len(lift.get("lifts") or []),
        "writes": [
            {
                key: item.get(key)
                for key in ("seq", "kind", "role", "status", "acked_at", "patch_ack_ms")
            }
            for item in lift.get("writes") or []
            if isinstance(item, dict)
        ],
        "latency": lift.get("latency"),
    }


def grade_isolation_lifted(
    *,
    observed: dict[str, Any],
    pair: Any,
    node_now: dict[str, Any],
    settled: dict[str, Any],
    fixture: EfaDriverFixture,
    bdf: str,
    expected: int,
    baseline_boot: str,
    lift: dict[str, Any],
    oracle: list[dict[str, Any]],
    window: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The R24 verdict, a harness-overreach FAIL, or an INCONCLUSIVE whose detail
    opens with a :data:`RACE_REASONS` / :data:`OTHER_REASONS` token.

    PASS when the reboot-after RESTART_NODE was refused for want of observed
    isolation and handed over at ``containment_or_release``; FAIL when the
    product did something the hand-off forbids (a climb to REPLACE_NODE, the
    wrong stage, a reboot after the refusal) or the hold broke its own rules.
    """

    boot_now = str(node_now.get("boot_id") or "")
    source_id = str(pair[1].get("request_id")) if pair else ""
    successors = reboot_successors(observed, source_id) if source_id else []
    reboot_after = successors[0] if successors else None
    node = str(lift.get("node") or node_now.get("name") or "")
    rebooted = bool(baseline_boot and boot_now and baseline_boot != boot_now)
    overreach = overreach_errors(
        lift, successor=successor_incident_id(source_id) if source_id else ""
    )
    extra = {
        "pci_bdf": bdf,
        "expected_count": expected,
        "baseline_boot_id": baseline_boot,
        "cordon_lifts": lift.get("lifts"),
        "cordon_hold_stopped": lift.get("stopped"),
        "cordon_lift": lift_summary(lift),
        "lift_window": window,
    }
    if overreach:
        return _result(
            {"harness_overreach": overreach},
            notes=[
                "the hold broke its own rules (cordon-only, at most "
                f"{MAX_LIFTS} lifts, only the reboot-after successor, nothing after "
                "it stopped); the product's outcome is not graded on such a run"
            ],
            extra={**extra, **_escalation_extra(settled, fixture)},
        )
    if reboot_after is None:
        raise VariantInconclusive(
            reason(
                "NO_RUNG",
                "no reboot-after rung opened (the post-repair validation was "
                "withheld or the node recovered); there was no RESTART_NODE to meet "
                "the lifted cordon",
            )
        )
    refused = refused_restart_step(reboot_after)
    if refused is None:
        token, text = classify_unrefused(
            reboot_after,
            node=node,
            lift=lift,
            rebooted=rebooted,
            boot_change=f"{baseline_boot} -> {boot_now}",
        )
        raise VariantInconclusive(reason(token, text))

    stages = {
        "isolation_refused": isolation_refusal_errors(observed, pair, refused),
        "containment_or_release_handoff": containment_release_handoff_errors(
            observed, reboot_after, node_now
        ),
        "no_reboot": []
        if not rebooted
        else [f"the node's boot id changed {baseline_boot} -> {boot_now}"],
        "no_warm_spare": [
            f"a warm spare was engaged: {item}" for item in warm_spare_touched(observed)
        ],
        "no_global_unload": _oracle_no_global_unload(oracle, expected),
        "harness_overreach": [],
    }
    return _result(
        stages,
        notes=[
            f"{bdf} unbound; post_repair_failure's trap re-unbound it on the product's "
            "post-repair snapshot so VALIDATE_FABRIC failed and the reboot-after rung "
            "opened",
            "inside the reboot-after successor's isolation window the fixture lifted "
            f"only the cordon (spec.unschedulable=false, at most {MAX_LIFTS} JSON "
            "patches that tested the successor's annotation, its fencing token and "
            f"the cordon; the second within {T_Q_SECONDS:g} s of QUARANTINE's "
            "re-cordon), leaving the quarantine taint, the incident annotation and "
            "the fencing token untouched",
            "R24 (b265): the reboot-after RESTART_NODE refused before any provider call "
            "(isolation_not_observed=NOT_ISOLATED), and the classifier handed the node "
            "over at containment_or_release -- the isolation put back (cordon + "
            "quarantine) and exactly one ticket naming the cordon; no REPLACE_NODE, no "
            "reboot, no warm spare",
        ],
        extra={
            **extra,
            "refused_restart": {
                "status": refused.get("status"),
                "error": str(refused.get("error") or "")[:200],
                "node_failures": (refused.get("details") or {}).get("node_failures"),
                "isolation_not_observed": (refused.get("details") or {}).get(
                    "isolation_not_observed"
                ),
            },
            "node_isolated_at_exit": True,
            **_escalation_extra(settled, fixture),
        },
    )


def _spec() -> VariantSpec:
    return VariantSpec(
        case_id=CASE_013,
        variant="post_repair_isolation_lifted",
        proof_mode=NATIVE,
        lanes=("eks", "hp"),
        risk="H",
        flags="X",
        injection=(
            "reuse post_repair_failure's chain (trap re-unbinds one allow-listed EFA "
            "function on the product's post-repair snapshot so VALIDATE_FABRIC fails "
            "and a reboot-after successor opens); over one watched, pre-authenticated "
            "client lift ONLY the case node's cordon (spec.unschedulable=false, the "
            "equivalent of kubectl uncordon) the moment that successor's "
            "MARK_UNSCHEDULABLE lands and once more the moment its QUARANTINE "
            "re-cordons -- at most two JSON patches, each testing the successor's "
            "incident annotation, its fencing token and the cordon -- leaving the "
            "quarantine taint, the incident annotation and the fencing token untouched"
        ),
        expected=(
            "R24 (b265): the reboot-after RESTART_NODE is refused before any provider "
            "call (isolation_not_observed=NOT_ISOLATED, no submission), and the "
            "classifier opens exactly one workflow-support-after at stage "
            "containment_or_release (FREEZE_EVIDENCE, MARK_UNSCHEDULABLE, QUARANTINE, "
            "ESCALATE_SUPPORT) that re-isolates the node and files exactly one ticket "
            "naming the cordon; no REPLACE_NODE anywhere, no reboot (boot id "
            "unchanged), no warm spare touched. A lift that misses the window is "
            "INCONCLUSIVE with a named race (LIFT_LATE, LIFT_UNDONE, "
            "REPLAY_WITHOUT_OBSERVE, NO_RUNG), never FAIL; a hold that writes more "
            "than the cordon, more than twice, or outside its successor FAILs. "
            "Generation-gated on 82f7b77 (NOT_RUN without it)."
        ),
        execute=post_repair_isolation_lifted,
        cleanup=(
            "re-cordon the node if the hold left it lifted while the successor still "
            "holds it (registered before the first lift); stop the trap, clear any "
            "override and rebind the function (deadman-backed); the product's reboot "
            "chain, if a reboot raced ahead, is waited out and the release is judged "
            "by the new boot id plus the product's post-boot inventory; otherwise the "
            "node is left isolated by the support-after successor's re-applied cordon "
            "and quarantine and handed back through the product's validated restore "
            "(hand_back, LIFO after the release); nothing persists across a reboot"
        ),
        max_minutes=90,
        approvals=(EFA_DRIVER_UNBIND, NODE_REBOOT),
    )


VARIANTS = [_spec()]

__all__ = [
    "DETECTION_POLL_SECONDS",
    "GENERATION_PROBE",
    "OTHER_REASONS",
    "RACE_REASONS",
    "VARIANTS",
    "classify_unrefused",
    "grade_isolation_lifted",
    "isolation_read",
    "lift_plan",
    "lift_summary",
    "observe_passed",
    "post_repair_isolation_lifted",
    "quarantine_baseline",
    "reason",
    "replay_markers",
    "require_isolation_refusal_generation",
    "window_evidence",
]
