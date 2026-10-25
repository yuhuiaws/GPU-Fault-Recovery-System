"""GF-GENERAL-LIVE-013 native EFA driver-binding negatives: ``bind_failure``
and ``post_repair_failure``.

These are the two branches of the reserved EFA driver unbind the live-replay
tree variants (``general_live_inventory.py``) cannot reach: the product's own
``REMEDIATE_EFA_DRIVER`` chain (``node_agent/operations/remediation.py``:
``modprobe efa`` + a per-BDF ``printf BDF > /sys/bus/pci/drivers/efa/bind``,
whose ``bind_errors`` raise ``EFA driver bind verification failed``) acting on
one real EFA data function that the fixture unbinds -- the same unbind and
``systemd-run --on-active`` rebind deadman COLLECT-017 A / DESTR-021 already use
(``probes/collector_node_probe.py`` ``unbind-efa``), extended on-node by
``probes/general_live_efa_probe.py`` with the two negatives:

* ``bind_failure``: the function is held un-rebindable by its own
  ``driver_override`` (a driver name no module provides), so every bind of
  *this one function* to ``efa`` returns ENODEV. The product sees DRIVER_UNBOUND
  (``host_health.efa_inventory_rule``), runs ``REMEDIATE_EFA_DRIVER``, and its
  bind of this BDF fails -- an independent FAILED receipt naming the BDF. The
  product's own escalation ladder (a failed EFA remediation whose step carries
  ``failure_escalation_action=REBOOT_NODE``, ``orchestration/escalation.py``
  ``_classify`` ``efa_recovery``) then reboots the node: the spec's approved
  branch for a bind that genuinely will not take, and the one that clears the
  per-device override. No global ``efa`` unload ever happens (the module stays
  loaded, every other function keeps its binding).
* ``post_repair_failure``: no override; the function is unbound, and a bounded
  one-shot trap re-unbinds it a few seconds after the product's bind, so
  ``REMEDIATE_EFA_DRIVER`` records SUCCEEDED (``rebound_pci_bdfs=[BDF]``) and a
  later post-repair step (``RESTART_EFA_DEVICE_PLUGIN`` or ``VALIDATE_FABRIC``)
  independently FAILS on fresh evidence, which escalates to the same reboot --
  the independent failure receipt and the escalation the spec requires.

Both keep the fixture scoped to ONE function, node-reversible and
deadman-restorable (clear the override, rebind), and both grade only on the
product's own records. They are the dangerous-live tail of the lane: risk H,
flag X, ``EFA_DRIVER_UNBIND`` + ``NODE_REBOOT`` approvals, run last.

Reusable: :class:`EfaDriverFixture` wraps ``general_live_efa_probe.py`` so any
category needing one allow-listed EFA function held unbound (or trapped after a
rebind) can call ``hold_unbind`` / ``trap_unbind`` / ``release`` with the same
deadman and LIFO cleanup.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pathlib import Path

from scripts.e2e.regional.collector_action_guard import require_action_time
from scripts.e2e.regional.general_live_common import (
    activity,
    unsettled_workflows,
    wait_for_incident,
)
from scripts.e2e.regional.general_live_fixture import (
    ProbeHandle,
    VariantBlocked,
    VariantContext,
    VariantInconclusive,
    utc_now,
)
from scripts.e2e.regional.general_live_registry import NATIVE, VariantSpec
from scripts.e2e.regional.general_live_verdicts import (
    executions,
    finding_incidents,
    node_isolated,
    operations,
    samples,
    verdict,
    workflows_of,
)

PROBES = Path(__file__).with_name("probes")
#: The on-node EFA driver fixture probe (sibling of the shared node probe).
EFA_DRIVER_PROBE = PROBES / "general_live_efa_probe.py"
CASE_013 = "GF-GENERAL-LIVE-013"
EFA_DRIVER_UNBIND = "EFA_DRIVER_UNBIND"
NODE_REBOOT = "NODE_REBOOT"
#: The host collector's EFA inventory samples the verdict reads.
EFA_METRICS = {
    "efa_inventory_mismatch",
    "efa_inventory_active_count",
    "efa_inventory_expected_count",
    "efa_inventory_missing_count",
    "efa_inventory_discovered_count",
}
REMEDIATE_STEP = "REMEDIATE_EFA_DRIVER"
PLUGIN_STEP = "RESTART_EFA_DEVICE_PLUGIN"
VALIDATE_STEP = "VALIDATE_FABRIC"
RESTART_NODE = "RESTART_NODE"
#: Detection budget (two confirmation ticks + ingestion + planning).
DETECTION_BUDGET = 420
#: The approved reboot chain (reboot, validations, controlled restore) may take
#: this long to settle; the runner waits it out and never cancels it.
REBOOT_SETTLE_BUDGET = 2400
#: Isolation beyond the EFA remediation chain's own cordon is a failure.
ISOLATION_OPERATIONS = frozenset({"QUARANTINE", "STOP_WORKLOADS"})


class EfaDriverFixture:
    """One allow-listed EFA function held unbound / trapped, with cleanup+deadman.

    Wraps ``general_live_efa_probe.py`` on a node-pinned probe Pod. The target
    function is chosen from the node's real EFA data functions (the lowest BDF
    with a bound ``efa`` driver and more than one present), recorded, and the
    coordinator's maintenance BDF allow-list must cover it. ``release`` is
    pushed on the context's LIFO cleanup, so it runs (and verifies the function
    is bound to ``efa`` again) before the probe Pod is torn down; a
    ``systemd-run`` deadman restores the baseline even if the runner, the Pod
    and the control plane all vanish.
    """

    def __init__(self, ctx: VariantContext) -> None:
        self.ctx = ctx
        self._bdf: str | None = None
        self._injected = False
        key = f"efa-driver@{ctx.settings.node}"
        handle = ctx.probes.get(key)
        if handle is None:
            handle = ProbeHandle(
                ctx.regional,
                script=EFA_DRIVER_PROBE,
                node=ctx.settings.node,
                image=ctx.settings.host_probe_image,
                key=ctx.spec.key,
                run_id=ctx.run_id,
                case_dir=ctx.case_dir,
                deadline_seconds=ctx.spec.max_minutes * 60 + 1800,
            )
            ctx.probes[key] = handle
        self.handle = handle
        ctx.cleanup.push("efa-driver", self.release)

    def _execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        return self.handle.execute(
            *arguments, "--run-id", self.ctx.run_id, timeout=timeout
        )

    def survey(self) -> dict[str, Any]:
        bdf = self.target_bdf()
        value = self._execute("summary", "--pci-bdf", bdf, timeout=120)
        self.ctx.evidence.write("efa-driver-survey", value)
        return value

    def target_bdf(self) -> str:
        """The one EFA function this run acts on (deterministic, recorded)."""

        if self._bdf is not None:
            return self._bdf
        listing = self._execute("functions", timeout=120)
        functions = listing.get("functions") or []
        if listing.get("error") and not functions:
            raise VariantBlocked(
                "FIXTURE_NOT_READY",
                f"the EFA driver probe could not enumerate functions: {listing}",
            )
        bound = sorted(
            str(item["pci_bdf"])
            for item in functions
            if item.get("driver") == "efa" and item.get("pci_bdf")
        )
        if len(bound) < 2:
            raise VariantBlocked(
                "CAPABILITY_ABSENT",
                f"the node lists {len(bound)} bound EFA function(s); need at least two",
            )
        self._bdf = bound[0]
        return self._bdf

    def hold_unbind(self, *, restore_seconds: int) -> dict[str, Any]:
        """bind_failure: hold the function un-rebindable, then unbind it."""

        bdf = self.target_bdf()
        require_action_time(120)
        value = self._execute(
            "hold-unbind",
            "--pci-bdf",
            bdf,
            "--restore-seconds",
            str(int(restore_seconds)),
            timeout=180,
        )
        self._injected = True
        self.ctx.evidence.write("efa-driver-hold", value)
        return value

    def trap_unbind(
        self, *, restore_seconds: int, grace_seconds: int, trap_window_seconds: int
    ) -> dict[str, Any]:
        """post_repair_failure: unbind, and re-unbind after the product rebinds."""

        bdf = self.target_bdf()
        require_action_time(120)
        value = self._execute(
            "trap-unbind",
            "--pci-bdf",
            bdf,
            "--restore-seconds",
            str(int(restore_seconds)),
            "--grace-seconds",
            str(int(grace_seconds)),
            "--trap-window-seconds",
            str(int(trap_window_seconds)),
            timeout=180,
        )
        self._injected = True
        self.ctx.evidence.write("efa-driver-trap", value)
        return value

    def summary(self) -> dict[str, Any]:
        value = self._execute("summary", "--pci-bdf", self.target_bdf(), timeout=120)
        self.ctx.evidence.write("efa-driver-state", value)
        return value

    def release(self) -> dict[str, Any]:
        """Restore the baseline (LIFO cleanup). Raises if the node is not clean."""

        if self._bdf is None:
            return {"released": False, "reason": "no target was selected"}
        value = self._execute("release", "--pci-bdf", self._bdf, timeout=180)
        self.ctx.evidence.write("efa-driver-release", value)
        return value


# --------------------------------------------------------------------------- #
# Grading helpers (read only the product's own records)
# --------------------------------------------------------------------------- #
def _efa_samples(observed: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [
        item
        for item in samples(observed, name)
        if isinstance(item.get("value"), (int, float))
    ]


def _expected_efa_count(ctx: VariantContext, fallback: int) -> int:
    latest = activity(ctx, utc_now(), metrics={"efa_inventory_expected_count"}).get(
        "metric_latest"
    )
    for item in latest or []:
        if item.get("name") == "efa_inventory_expected_count" and isinstance(
            item.get("value"), (int, float)
        ):
            return int(item["value"])
    return fallback


def _failure_mode_errors(detection: dict[str, Any], mode: str) -> list[str]:
    confirmed = [
        item
        for item in _efa_samples(detection, "efa_inventory_mismatch")
        if item["value"] >= 1
    ]
    if not confirmed:
        return ["no efa_inventory_mismatch=1 sample reached the product"]
    modes = sorted(
        {(item.get("labels") or {}).get("failure_mode", "?") for item in confirmed}
    )
    return [] if modes == [mode] else [f"failure_mode {modes}, expected [{mode!r}]"]


def _remote_node_result(
    observed: dict[str, Any], *, operation: str, node: str
) -> dict[str, Any]:
    """The Node Agent's result for ``operation`` on ``node`` (remote command)."""

    commands = [
        item
        for item in observed.get("commands") or []
        if item.get("operation") == operation
    ]
    if not commands:
        return {}
    details = commands[-1].get("result_details") or {}
    results = details.get("node_results") or {}
    value = results.get(node)
    return value if isinstance(value, dict) else {}


def _incident_and_first_workflow(
    observed: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    incidents = finding_incidents(observed, "efa_inventory_mismatch")
    if not incidents:
        return None
    incident = incidents[0]
    workflows = workflows_of(observed, str(incident["incident_id"]))
    if not workflows:
        return None
    return incident, workflows[0]


def _reboot_planned_errors(observed: dict[str, Any], node: str) -> list[str]:
    """The spec'd approved branch: a RESTART_NODE planned on the node.

    A failed EFA remediation (bind_failure) or a post-repair validation that
    observed the failure (post_repair_failure) carries
    ``failure_escalation_action=REBOOT_NODE``; the product's ladder opens an
    escalation incident whose workflow plans RESTART_NODE. Found by scanning
    every workflow on the node for the reboot step -- the escalation incident's
    event id does not name the EFA metric.
    """

    for workflow in observed.get("workflows") or []:
        if RESTART_NODE in operations(workflow):
            return []
    return ["no RESTART_NODE escalation was planned (the spec's approved branch)"]


def _await_efa_recovery(
    ctx: VariantContext, since: datetime, expected: int
) -> list[str]:
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
        timeout_seconds=ctx.host_summary + ctx.host_interval * 8,
        poll_seconds=ctx.host_interval,
        name="efa-driver-recovery",
    )
    if value is None:
        return [
            f"no fresh efa_inventory_active_count={expected} with mismatch=0 after recovery"
        ]
    return []


def _oracle_no_global_unload(records: list[dict[str, Any]], expected: int) -> list[str]:
    """From the probe oracle: the module stayed loaded, no function went to 0."""

    errors = []
    for record in records:
        if record.get("module_loaded") is False:
            errors.append(f"the efa module was unloaded at {record.get('observed_at')}")
        bound = record.get("bound_count")
        if isinstance(bound, int) and bound < expected - 1:
            errors.append(
                f"only {bound} EFA functions bound at {record.get('observed_at')} "
                f"(expected >= {expected - 1}; a global unload or extra loss)"
            )
    return errors


def _isolation_errors(ctx: VariantContext, observed: dict[str, Any]) -> list[str]:
    """Isolation beyond the EFA remediation / reboot chain's own cordon."""

    errors = []
    for workflow in observed.get("workflows") or []:
        extra = sorted(set(operations(workflow)) & ISOLATION_OPERATIONS)
        if extra:
            errors.append(f"workflow {workflow.get('request_id')} plans {extra}")
    return errors


def _await_recovered(
    ctx: VariantContext, since: datetime, expected: int, baseline_boot: str
) -> dict[str, Any] | None:
    """Poll until the reboot completed and the node is healthy and clean.

    The reboot is the spec'd branch of both variants, so a changed boot id is
    required: the node returns on a new boot with full EFA inventory, no
    gpu-fault taint/annotation, and no workflow still running.
    """

    def accept() -> dict[str, Any] | None:
        current = activity(ctx, since, metrics=EFA_METRICS)
        if unsettled_workflows(current):
            return None
        node = ctx.regional.node_snapshot(ctx.settings.node)
        if node.get("boot_id") in {None, "", baseline_boot}:
            return None
        if node_isolated(node):
            return None
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

    return ctx.wait_until(
        accept,
        timeout_seconds=REBOOT_SETTLE_BUDGET,
        poll_seconds=max(15, ctx.host_interval),
        name="efa-driver-settled",
    )


def _result(
    stages: dict[str, list[str]],
    *,
    notes: list[str],
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "verdict": verdict(stages),
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "notes": notes,
        "proof_scope": "native",
        **extra,
    }


def bind_failure(ctx: VariantContext) -> dict[str, Any]:
    """The product's bind of one held EFA function fails; its receipt is
    independent and it takes the approved reboot branch; no global efa unload."""

    fixture = EfaDriverFixture(ctx)
    survey = fixture.survey()
    bdf = fixture.target_bdf()
    node = ctx.settings.node
    baseline_bound = int(survey.get("bound_count") or 0)
    baseline_boot = str(survey.get("boot_id") or "")
    expected = _expected_efa_count(ctx, baseline_bound)
    started = utc_now()
    injected = fixture.hold_unbind(restore_seconds=ctx.spec.max_minutes * 60)
    oracle = [fixture.summary()]
    detection = wait_for_incident(
        ctx,
        started,
        "efa_inventory_mismatch",
        timeout_seconds=DETECTION_BUDGET,
        metrics=EFA_METRICS,
        name="bind-failure-detection",
    )
    oracle.append(fixture.summary())
    ctx.evidence.append_oracle(
        [{"pci_bdf": bdf, "injection": injected, "readings": oracle}],
        source="efa-driver",
    )
    if not finding_incidents(detection, "efa_inventory_mismatch"):
        raise VariantInconclusive(
            f"the product never minted an efa_inventory_mismatch finding from the "
            f"unbound function {bdf}"
        )
    recovered = _await_recovered(ctx, started, expected, baseline_boot)
    observed = recovered or activity(ctx, started, metrics=EFA_METRICS)
    pair = _incident_and_first_workflow(observed)
    remediation = [item for item in executions(pair[1], REMEDIATE_STEP)] if pair else []
    failed = [item for item in remediation if item.get("status") == "FAILED"]
    receipt = _remote_node_result(observed, operation=REMEDIATE_STEP, node=node)
    bind_text = (
        " ".join(str(item.get("error") or "") for item in failed) + " " + str(receipt)
    )
    stages = {
        "driver_unbound": _failure_mode_errors(detection, "DRIVER_UNBOUND"),
        "bind_failure_receipt": []
        if failed
        else [f"{REMEDIATE_STEP} did not record a FAILED bind receipt"],
        "receipt_names_bdf": []
        if (failed and ("bind" in bind_text.lower()) and (bdf in bind_text))
        else [
            f"the {REMEDIATE_STEP} receipt does not name a bind failure for {bdf}: "
            f"{bind_text[:200]}"
        ],
        "no_global_unload": _oracle_no_global_unload(oracle, expected),
        "approved_reboot_branch": _reboot_planned_errors(observed, node),
        "not_isolated_beyond_chain": _isolation_errors(ctx, observed),
        "recovered": []
        if recovered
        else ["the node did not return healthy on a new boot within the window"],
    }
    return _result(
        stages,
        notes=[
            f"{bdf} held by driver_override={injected.get('driver_override')!r} and "
            "unbound; the product's targeted bind of this one function fails (ENODEV)",
            "the failed EFA remediation escalates to RESTART_NODE -- the approved "
            "branch for a bind that will not take, which also clears the override",
            "no global efa unload: the module stayed loaded and every other function "
            "kept its binding throughout",
        ],
        extra={
            "pci_bdf": bdf,
            "expected_count": expected,
            "baseline_boot_id": baseline_boot,
            "remediation_executions": [
                {
                    "status": item.get("status"),
                    "error": str(item.get("error") or "")[:200],
                }
                for item in remediation
            ],
            "node_result": receipt,
        },
    )


def _succeeded_remediation_errors(
    workflow: dict[str, Any],
    observed: dict[str, Any],
    *,
    bdf: str,
    node: str,
    expected: int,
) -> tuple[list[str], dict[str, Any]]:
    """The action's own receipt: REMEDIATE_EFA_DRIVER SUCCEEDED, it bound the BDF."""

    remediation = executions(workflow, REMEDIATE_STEP)
    terminal = [
        item for item in remediation if item.get("status") in {"SUCCEEDED", "FAILED"}
    ]
    receipt = _remote_node_result(observed, operation=REMEDIATE_STEP, node=node)
    errors = []
    if not terminal or terminal[-1].get("status") != "SUCCEEDED":
        errors.append(
            f"{REMEDIATE_STEP} did not record a SUCCEEDED receipt: "
            f"{[item.get('status') for item in remediation]}"
        )
    if receipt.get("rebound_pci_bdfs") != [bdf]:
        errors.append(
            f"the node result rebound {receipt.get('rebound_pci_bdfs')!r}, expected [{bdf!r}]"
        )
    if receipt.get("already_bound") is not False:
        errors.append(
            "the node result says the function was already bound (nothing done)"
        )
    return errors, receipt


def _post_repair_failure_errors(
    workflow: dict[str, Any],
) -> tuple[list[str], dict[str, Any] | None]:
    """A step after the SUCCEEDED remediation that independently FAILED."""

    remediation = executions(workflow, REMEDIATE_STEP)
    after = min(
        (
            item.get("step_index", 0)
            for item in remediation
            if item.get("status") == "SUCCEEDED"
        ),
        default=None,
    )
    if after is None:
        return (["no SUCCEEDED remediation to validate after"], None)
    failed = [
        item
        for step in (PLUGIN_STEP, VALIDATE_STEP)
        for item in executions(workflow, step)
        if item.get("status") == "FAILED" and item.get("step_index", -1) > after
    ]
    if not failed:
        return (
            ["no post-repair step (plugin restart or fabric validation) FAILED"],
            None,
        )
    failed.sort(key=lambda item: item.get("step_index", 0))
    return [], failed[-1]


def post_repair_failure(ctx: VariantContext) -> dict[str, Any]:
    """The product binds the function back (SUCCEEDED receipt), the fixture
    re-unbinds it, and the post-repair step fails independently and escalates."""

    fixture = EfaDriverFixture(ctx)
    survey = fixture.survey()
    bdf = fixture.target_bdf()
    node = ctx.settings.node
    baseline_bound = int(survey.get("bound_count") or 0)
    baseline_boot = str(survey.get("boot_id") or "")
    expected = _expected_efa_count(ctx, baseline_bound)
    started = utc_now()
    injected = fixture.trap_unbind(
        restore_seconds=ctx.spec.max_minutes * 60,
        grace_seconds=5,
        trap_window_seconds=DETECTION_BUDGET + 180,
    )
    oracle = [fixture.summary()]
    detection = wait_for_incident(
        ctx,
        started,
        "efa_inventory_mismatch",
        timeout_seconds=DETECTION_BUDGET,
        metrics=EFA_METRICS,
        name="post-repair-detection",
    )
    oracle.append(fixture.summary())
    if not finding_incidents(detection, "efa_inventory_mismatch"):
        raise VariantInconclusive(
            f"the product never minted an efa_inventory_mismatch finding from the "
            f"unbound function {bdf}"
        )
    recovered = _await_recovered(ctx, started, expected, baseline_boot)
    final_summary = fixture.summary() if recovered else {}
    ctx.evidence.append_oracle(
        [
            {
                "pci_bdf": bdf,
                "injection": injected,
                "readings": oracle,
                "final": final_summary,
            }
        ],
        source="efa-driver",
    )
    observed = recovered or activity(ctx, started, metrics=EFA_METRICS)
    pair = _incident_and_first_workflow(observed)
    workflow = pair[1] if pair else {}
    succeeded_errors, receipt = _succeeded_remediation_errors(
        workflow, observed, bdf=bdf, node=node, expected=expected
    )
    post_errors, failed_step = _post_repair_failure_errors(workflow)
    trap_fired = (final_summary or oracle[-1]).get("trap_fired") or {}
    stages = {
        "driver_unbound": _failure_mode_errors(detection, "DRIVER_UNBOUND"),
        "action_succeeded": succeeded_errors,
        "independent_post_repair_failure": post_errors,
        "trap_fired": []
        if trap_fired.get("fired") is True
        else [
            f"the re-unbind trap did not fire after the product's bind: {trap_fired}"
        ],
        "escalation": _reboot_planned_errors(observed, node),
        "no_global_unload": _oracle_no_global_unload(oracle, expected),
        "not_isolated_beyond_chain": _isolation_errors(ctx, observed),
        "recovered": []
        if recovered
        else ["the node did not return healthy on a new boot within the window"],
    }
    return _result(
        stages,
        notes=[
            f"{bdf} unbound; a bounded trap re-unbound it {injected.get('grace_seconds')}s "
            "after the product's bind, so the action receipt is SUCCEEDED and the "
            "post-repair step fails on fresh evidence",
            "the post-repair failure escalates to RESTART_NODE -- the escalation the "
            "spec requires for an action whose validation observed a continuing fault",
        ],
        extra={
            "pci_bdf": bdf,
            "expected_count": expected,
            "baseline_boot_id": baseline_boot,
            "node_result": receipt,
            "failed_post_repair_step": {
                "operation": (failed_step or {}).get("operation"),
                "status": (failed_step or {}).get("status"),
                "error": str((failed_step or {}).get("error") or "")[:200],
            },
            "trap": trap_fired,
        },
    )


def _efa_driver_variant(
    variant: str, *, injection: str, expected: str, execute: Any, cleanup: str
) -> VariantSpec:
    return VariantSpec(
        case_id=CASE_013,
        variant=variant,
        proof_mode=NATIVE,
        lanes=("eks", "hp"),
        risk="H",
        flags="X",
        injection=injection,
        expected=expected,
        cleanup=cleanup,
        max_minutes=90,
        approvals=(EFA_DRIVER_UNBIND, NODE_REBOOT),
        execute=execute,
    )


VARIANTS = [
    _efa_driver_variant(
        "bind_failure",
        injection=(
            "unbind one allow-listed EFA data function and hold it un-rebindable "
            "(its driver_override set to a name no module provides), so the product's "
            "REMEDIATE_EFA_DRIVER bind of this one BDF fails with ENODEV"
        ),
        expected=(
            "DRIVER_UNBOUND -> REMEDIATE_EFA_DRIVER records a FAILED bind receipt naming "
            "the BDF; the product escalates to RESTART_NODE (the approved branch, which "
            "clears the override); no global efa unload; node returns healthy on a new boot"
        ),
        execute=bind_failure,
        cleanup=(
            "stop the trap, clear the driver_override and rebind the function "
            "(deadman-backed); the product's reboot chain is waited out, never cancelled; "
            "fail loudly unless the function is bound to efa with no override"
        ),
    ),
    _efa_driver_variant(
        "post_repair_failure",
        injection=(
            "unbind one allow-listed EFA data function, and a bounded one-shot trap "
            "re-unbinds it a few seconds after the product's REMEDIATE_EFA_DRIVER bind "
            "(no driver_override), so the action receipt SUCCEEDS and the post-repair step fails"
        ),
        expected=(
            "REMEDIATE_EFA_DRIVER SUCCEEDED (rebound_pci_bdfs=[BDF]) then an independent "
            "post-repair failure (RESTART_EFA_DEVICE_PLUGIN or VALIDATE_FABRIC) on fresh "
            "evidence -> RESTART_NODE escalation; no global efa unload; node returns healthy"
        ),
        execute=post_repair_failure,
        cleanup=(
            "stop the trap, clear any override and rebind the function (deadman-backed); "
            "the product's reboot chain is waited out, never cancelled; fail loudly unless "
            "the function is bound to efa"
        ),
    ),
]

__all__ = ["EfaDriverFixture", "VARIANTS", "bind_failure", "post_repair_failure"]
