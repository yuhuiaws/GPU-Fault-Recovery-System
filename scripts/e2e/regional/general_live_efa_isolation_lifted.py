"""GF-GENERAL-LIVE-013 ``post_repair_isolation_lifted``: a reboot whose
isolation was lifted mid-chain is handed over, not climbed (b265, 82f7b77).

This reuses ``post_repair_failure``'s chain wholesale: the on-node
``probes/general_live_efa_probe.py`` trap re-unbinds one allow-listed EFA data
function after the product's ``REMEDIATE_EFA_DRIVER`` bind, on the product's own
post-repair snapshot (the host-collector restart), so ``VALIDATE_FABRIC`` fails
on fresh evidence and the ladder opens exactly one ``workflow-reboot-after``
successor whose plan is FREEZE_EVIDENCE, MARK_UNSCHEDULABLE, QUARANTINE,
RESTART_NODE, the validations, RESTORE_SCHEDULING (``escalation
._escalation_operations``). Then, once that successor's QUARANTINE has SUCCEEDED
and before its RESTART_NODE has submitted to the provider, the fixture lifts
*only the cordon* of the case node (``spec.unschedulable=false``, the equivalent
of ``kubectl uncordon`` -- the quarantine taint, the incident annotation and the
fencing token are left untouched). The next thing the node lifecycle adapter
does for RESTART_NODE is read the node and refuse, before the preflight, the
agent revocation and the provider call, because ``spec.unschedulable is not
true`` (``adapters/hyperpod/lifecycle._observe_isolation`` ->
``_isolation_problems``).

R23 (live today) carried that refusal as a bare ``safety_rejection`` and climbed
to REPLACE_NODE, which with a healthy warm spare could fail the node over though
no reboot ever ran. R24 (b265, 82f7b77) marks it ``isolation_not_observed =
NOT_ISOLATED`` and the classifier (``failure_outcome
.lifecycle_isolation_refusals`` -> ``escalation._classify``,
``CONTAINMENT_STAGE``) answers it with the hand-off a failed containment step
already gets: ``containment_or_release`` -- FREEZE_EVIDENCE, MARK_UNSCHEDULABLE,
QUARANTINE (the isolation put back), ESCALATE_SUPPORT (exactly one ticket), no
rung. The ticket's reason names the problem: ``RESTART_NODE not submitted,
isolation not observed: spec.unschedulable is not true``. No REPLACE_NODE, no
reboot, no warm spare.

Timing (established from the R23 P13 records and the product code, not a
guess). RESTART_NODE on a regional HyperPod/EKS cluster is a remote command
(``regional.RegionalRemoteWorkflowAdapter``): the control plane mints it right
after QUARANTINE (P13 attempt 2: 26 ms after QUARANTINE's step completed) and
the GPU-cluster executor claims it over a NOTIFY-woken 20 s long-poll
(``app/remote_command_wakeups.py``; ``cluster_executor`` default
``claim_wait_seconds=20``), so the claim is sub-second in a healthy system and
only degrades to the ~2 s poll if the wakeup listener is down. ``_observe
_isolation`` is the adapter's first action on that claim
(``lifecycle.execute`` -> ``require_mutation_ownership`` ->
``_observe_isolation``, before the DescribeCluster preflight that accounts for
the ~10 s to the provider submission). So the window from QUARANTINE SUCCEEDED
to the isolation read is **sub-second**, and QUARANTINE itself does not mutate
the node (its MARK_UNSCHEDULABLE already set the cordon, taint and annotation;
P13: QUARANTINE's node resource version is unchanged before/after) and does not
verify the cordon after its patch (``node_operations._isolate`` re-reads for the
baseline but never fails on the cordon), so a cordon-only lift can never fail
QUARANTINE -- a lift that lands before QUARANTINE is simply re-cordoned.

Because the window is sub-second, a store-poll trigger (one read is a kubectl
exec, ~1-3 s) cannot win and would reboot the node every run. The trigger is
therefore a fast node-read HOLD (``hold_cordon_lifted``): once the node carries
a *successor* incident's ownership annotation (the reboot-after incident, which
takes the node over from the source EFA incident at its MARK_UNSCHEDULABLE) and
is cordoned, it lifts the cordon and keeps re-lifting (bounded by
``LIFT_CAP``, cordon-only) while that same successor owns the cordoned node, so
the cordon is down whenever the reboot rung's ``_observe_isolation`` reads it.
It stops the moment a *different* successor re-isolates the node (the
support-after of the refused reboot -- the window was met) or the node reboots
(boot id change -- the lift lost the race). This relaxes the brief's literal
"one-shot" to a bounded cordon-only hold, the only way to beat a sub-second
window from outside the cluster; the quarantine taint, the incident annotation
and the fencing token are never touched. Expected hit rate is moderate, not
certain: a miss is a clean reboot (the ``post_repair_failure`` exit, graded
across the boot by ``EfaDriverFixture.release``), never a FAIL. Both race
misses stay INCONCLUSIVE:

* a different successor never appears and the node reboots -- RESTART_NODE's
  observe read a cordoned node before the hold could lift it;
* QUARANTINE itself FAILED (a patch-retry exhaustion under the hold) -- the
  pre-existing containment path, not 82f7b77.

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

from gpu_fault.adapters.common import ANNOTATION_INCIDENT

from scripts.e2e.regional.general_live_common import wait_for_incident
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
from scripts.e2e.regional.general_live_verdicts import finding_incidents

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
#: The cordon-hold reads the node directly (no store in the hot path), as fast
#: as a ``kubectl get node`` answers; the sub-second QUARANTINE -> observe window
#: (see the module docstring) is far below one store read, so the store is never
#: on the trigger path. A ``get node`` is ~0.3 s, so the effective cadence is
#: well under a second.
NODE_POLL_SECONDS = 0.2
#: How many cordon lifts the hold will issue for one run. The reboot-after
#: successor's own QUARANTINE re-cordons a lift that landed before it, so the
#: hold must be able to lift again; the cap bounds any pathological loop (a
#: support-after re-isolation the hold has not yet recognised) to a handful of
#: cordon-only patches.
LIFT_CAP = 5


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


def lift_cordon(ctx: VariantContext, record: dict[str, Any]) -> dict[str, Any]:
    """Set ``spec.unschedulable=false`` on the case node (one cordon lift).

    A JSON merge patch that names only ``spec.unschedulable`` -- the quarantine
    taint, the ``gpu-fault.io/`` ownership annotations and the fencing token are
    not in the patch and are left exactly as the product wrote them (the
    equivalent of ``kubectl uncordon``). Each call appends to ``record['lifts']``
    (the hold may re-lift after the successor's QUARANTINE re-cordons); it reads
    nothing (the hold already holds the node reading that triggered it), so a
    re-lift costs one patch, not a node round-trip.
    """

    node = ctx.settings.node
    ctx.regional.kubectl(
        "gpu",
        "patch",
        "node",
        node,
        "--type=merge",
        "-p",
        json.dumps({"spec": {"unschedulable": False}}),
        timeout=60,
    )
    lift = {
        "node": node,
        "lifted_at": utc_now().isoformat(),
        "patch": {"spec": {"unschedulable": False}},
    }
    record.setdefault("lifts", []).append(lift)
    return lift


def hold_cordon_lifted(
    ctx: VariantContext,
    *,
    source_incident_id: str,
    baseline_boot: str,
    window: float,
) -> dict[str, Any]:
    """Hold the case node's cordon lifted across the reboot-after successor's
    isolation steps so its RESTART_NODE's ``_observe_isolation`` reads it
    uncordoned.

    Reads only the node (no store in the hot path). The reboot-after successor
    takes the node over from the source EFA incident at its MARK_UNSCHEDULABLE,
    stamping its own ownership annotation; the instant the node carries a
    successor incident (``!= source_incident_id``) and is cordoned, lift the
    cordon, and keep re-lifting (bounded by :data:`LIFT_CAP`) while that same
    successor owns the cordoned node -- its own QUARANTINE re-cordons a lift
    that landed before it, so a single lift would be undone. Stop when a
    *different* successor re-isolates the node (the support-after of the refused
    reboot: the window was met), when the node reboots (boot id change: the lift
    lost the race), or when ``window`` elapses. Cordon-only throughout.
    """

    record: dict[str, Any] = {"lifts": [], "stopped": None, "successor_incident": None}

    def accept() -> dict[str, Any] | None:
        node = ctx.regional.node_snapshot(ctx.settings.node)
        boot = str(node.get("boot_id") or "")
        annotation = (node.get("ownership_annotations") or {}).get(ANNOTATION_INCIDENT)
        unschedulable = bool(node.get("unschedulable"))
        record["last"] = {
            "incident": annotation,
            "unschedulable": unschedulable,
            "boot_id": boot,
        }
        if baseline_boot and boot and boot != baseline_boot:
            record["stopped"] = "rebooted"
            return record
        if annotation and annotation != source_incident_id:
            if record["successor_incident"] is None:
                record["successor_incident"] = annotation
            if annotation != record["successor_incident"]:
                # A later successor re-isolated the node: the support-after of
                # the refused reboot. The reboot rung's observe already ran.
                record["stopped"] = "handed_over"
                return record
            if unschedulable:
                if len(record["lifts"]) >= LIFT_CAP:
                    record["stopped"] = "lift_cap"
                    return record
                lift_cordon(ctx, record)
        return None

    ctx.wait_until(
        accept,
        timeout_seconds=window,
        poll_seconds=NODE_POLL_SECONDS,
        name="lifted-cordon-hold",
    )
    record["stopped"] = record["stopped"] or "window"
    ctx.evidence.write("efa-driver-cordon-hold", record)
    return record


def post_repair_isolation_lifted(ctx: VariantContext) -> dict[str, Any]:
    """Reuse ``post_repair_failure``'s chain, lift the reboot successor's cordon
    once in the QUARANTINE->RESTART_NODE window, and grade the R24 hand-off."""

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
    source = efa_chain(detection)
    source_incident_id = str(source[0].get("incident_id")) if source else ""
    lift = hold_cordon_lifted(
        ctx,
        source_incident_id=source_incident_id,
        baseline_boot=baseline_boot,
        window=TRAP_WINDOW + 120,
    )
    recovered, observed, node_now, pair, settled = _settle(
        ctx,
        started=started,
        expected=expected,
        baseline_boot=baseline_boot,
        rung_name="lifted-rung",
    )
    incidents[:] = _incidents_on_node(observed, node) or incidents
    oracle.append(fixture.summary_or_loss())
    ctx.evidence.append_oracle(
        [
            {
                "pci_bdf": bdf,
                "injection": injected,
                "readings": oracle,
                "cordon_lift": lift,
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
    )


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
) -> dict[str, Any]:
    """The R24 verdict, or an INCONCLUSIVE race with a named reason.

    PASS when the reboot-after RESTART_NODE was refused for want of observed
    isolation and handed over at ``containment_or_release``; INCONCLUSIVE when
    the race missed the window (a reboot happened, or QUARANTINE failed its own
    verification, or no reboot rung opened); FAIL only when the product did
    something the hand-off forbids (a climb to REPLACE_NODE, the wrong stage,
    a reboot after the refusal).
    """

    boot_now = str(node_now.get("boot_id") or "")
    source_id = str(pair[1].get("request_id")) if pair else ""
    successors = reboot_successors(observed, source_id) if source_id else []
    reboot_after = successors[0] if successors else None

    if baseline_boot and boot_now and baseline_boot != boot_now:
        raise VariantInconclusive(
            "RESTART_NODE ran before the cordon lift: the node rebooted "
            f"({baseline_boot} -> {boot_now}); the reboot is the post_repair_failure "
            "exit, graded clean across the boot by the reboot-aware cleanup, and the "
            "lifted-cordon refusal was not exercised"
        )
    if reboot_after is None:
        raise VariantInconclusive(
            "no reboot-after rung opened (the post-repair validation was withheld or "
            "the node recovered); there was no RESTART_NODE to meet the lifted cordon"
        )

    steps = reboot_after.get("step_executions") or []
    refused = refused_restart_step(reboot_after)
    if refused is None:
        if any(
            item.get("operation") in (QUARANTINE, MARK_UNSCHEDULABLE)
            and item.get("status") == "FAILED"
            for item in steps
        ):
            raise VariantInconclusive(
                "the cordon lift landed before QUARANTINE succeeded: the reboot-after "
                "QUARANTINE failed its own verification -- the pre-existing containment "
                "path, not the 82f7b77 RESTART_NODE refusal"
            )
        restarts = [item for item in steps if item.get("operation") == RESTART_NODE]
        if restarts and (
            (restarts[-1].get("details") or {}).get("submitted_nodes")
            or restarts[-1].get("status") == "SUCCEEDED"
        ):
            raise VariantInconclusive(
                "RESTART_NODE submitted to the provider before the cordon lift: a "
                "reboot is in progress (reboot-aware cleanup); the lifted-cordon "
                "refusal was not exercised"
            )
        raise VariantInconclusive(
            "the reboot-after RESTART_NODE did not refuse for want of observed "
            f"isolation (cordon hold stopped {lift.get('stopped')!r} after "
            f"{len(lift.get('lifts') or [])} lift(s)); the lift did not land in the "
            "QUARANTINE->RESTART_NODE window"
        )

    stages = {
        "isolation_refused": isolation_refusal_errors(observed, pair, refused),
        "containment_or_release_handoff": containment_release_handoff_errors(
            observed, reboot_after, node_now
        ),
        "no_reboot": []
        if baseline_boot == boot_now
        else [f"the node's boot id changed {baseline_boot} -> {boot_now}"],
        "no_warm_spare": [
            f"a warm spare was engaged: {item}" for item in warm_spare_touched(observed)
        ],
        "no_global_unload": _oracle_no_global_unload(oracle, expected),
    }
    return _result(
        stages,
        notes=[
            f"{bdf} unbound; post_repair_failure's trap re-unbound it on the product's "
            "post-repair snapshot so VALIDATE_FABRIC failed and the reboot-after rung "
            "opened",
            "after the reboot-after successor's QUARANTINE succeeded, the fixture "
            "lifted only the cordon (spec.unschedulable=false), leaving the quarantine "
            "taint, the incident annotation and the fencing token untouched",
            "R24 (b265): the reboot-after RESTART_NODE refused before any provider call "
            "(isolation_not_observed=NOT_ISOLATED), and the classifier handed the node "
            "over at containment_or_release -- the isolation put back (cordon + "
            "quarantine) and exactly one ticket naming the cordon; no REPLACE_NODE, no "
            "reboot, no warm spare",
        ],
        extra={
            "pci_bdf": bdf,
            "expected_count": expected,
            "baseline_boot_id": baseline_boot,
            "cordon_lifts": lift.get("lifts"),
            "cordon_hold_stopped": lift.get("stopped"),
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
            "and a reboot-after successor opens); after that successor's QUARANTINE "
            "SUCCEEDED and before its RESTART_NODE submits, lift ONLY the case node's "
            "cordon (spec.unschedulable=false, the equivalent of kubectl uncordon), "
            "leaving the quarantine taint, the incident annotation and the fencing "
            "token untouched"
        ),
        expected=(
            "R24 (b265): the reboot-after RESTART_NODE is refused before any provider "
            "call (isolation_not_observed=NOT_ISOLATED, no submission), and the "
            "classifier opens exactly one workflow-support-after at stage "
            "containment_or_release (FREEZE_EVIDENCE, MARK_UNSCHEDULABLE, QUARANTINE, "
            "ESCALATE_SUPPORT) that re-isolates the node and files exactly one ticket "
            "naming the cordon; no REPLACE_NODE anywhere, no reboot (boot id "
            "unchanged), no warm spare touched. A lift that misses the window is "
            "INCONCLUSIVE (a reboot happened, or QUARANTINE failed its own "
            "verification), never FAIL. Generation-gated on 82f7b77 (NOT_RUN without "
            "it)."
        ),
        execute=post_repair_isolation_lifted,
        cleanup=(
            "stop the trap, clear any override and rebind the function "
            "(deadman-backed); the product's reboot chain, if a reboot raced ahead, is "
            "waited out and the release is judged by the new boot id plus the product's "
            "post-boot inventory; otherwise the node is left isolated by the "
            "support-after successor's re-applied cordon and quarantine and handed back "
            "through the product's validated restore (hand_back, LIFO after the "
            "release); nothing persists across a reboot"
        ),
        max_minutes=90,
        approvals=(EFA_DRIVER_UNBIND, NODE_REBOOT),
    )


VARIANTS = [_spec()]

__all__ = [
    "GENERATION_PROBE",
    "VARIANTS",
    "grade_isolation_lifted",
    "hold_cordon_lifted",
    "lift_cordon",
    "post_repair_isolation_lifted",
    "require_isolation_refusal_generation",
]
