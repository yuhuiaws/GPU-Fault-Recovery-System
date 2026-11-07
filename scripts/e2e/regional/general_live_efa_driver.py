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
  ``failure_escalation_action=REBOOT_NODE``, ``orchestration/families/health.py``
  ``_efa_parameters``; ``orchestration/escalation.py`` ``_classify``
  ``efa_recovery``) then reboots the node: the spec's approved branch for a bind
  that genuinely will not take, and the one that clears the per-device
  override. No global ``efa`` unload ever happens (the module stays loaded,
  every other function keeps its binding).
* ``post_repair_failure``: no override; the function is unbound, and a bounded
  one-shot trap re-unbinds it after the product's bind -- on the product's own
  post-repair snapshot, the host collector's restart (see below) -- so
  ``REMEDIATE_EFA_DRIVER`` records SUCCEEDED (``rebound_pci_bdfs=[BDF]``) and a
  later post-repair step (``RESTART_EFA_DEVICE_PLUGIN`` or ``VALIDATE_FABRIC``)
  independently FAILS on fresh evidence, which escalates to the same reboot --
  the independent failure receipt and the escalation the spec requires.

The reboot is the product's designed exit, and it takes the fixture with it.
Live 2026-10-27 06:55-07:06Z (HyperPod, R13g): the held function's bind failed
(``bind_errors=<BDF>=ENODEV``), the remediation record ended FAILED, the
ladder opened ``workflow-reboot-after-<record>`` -- FREEZE_EVIDENCE,
MARK_UNSCHEDULABLE, QUARANTINE, RESTART_NODE, VALIDATE_GPU, VALIDATE_HOST,
VALIDATE_FABRIC, RESTORE_SCHEDULING (``escalation.py`` ``_escalation_operations``:
every reboot or replacement successor cordons and quarantines the node it acts
on) -- and rebooted the node; the override, the transient deadman/trap units
and the ``/run`` state went with the boot, the node came back Ready with 16/16
bound and was released, and the variant graded FAIL twice for the harness's
own expectations: ``not_isolated_beyond_chain`` read the successor's QUARANTINE
as isolation beyond the chain, and the cleanup's ``release`` ran through a
probe Pod the reboot had terminated. Both variants are reboot-aware now:

* the escalation is graded on the product's records before and at the reboot
  rung and on the settled exit (``general_live_efa_escalation``): one
  remediation attempt, exactly one reboot-after successor with the designed
  plan (its QUARANTINE is the chain's own cordon), RESTART_NODE executed once,
  no REPLACE_NODE and no further rung, the node back on a new boot and
  released, both incidents RECOVERED (the successor's restore closes its
  QUARANTINED predecessor, ``orchestration/incident_closure.py``
  ``_close_escalation_predecessors``). Isolation beyond the chain is a
  QUARANTINE in the remediation record itself, a STOP_WORKLOADS on a step that
  names no workload, or any isolation step of a workflow outside the chain;
* nothing of the fixture persists across a reboot, by construction (sysfs
  override, ``systemd-run`` transient units, ``/run`` state), so a ``release``
  the probe cannot run is judged by the node's boot id
  (``close_after_probe_loss``) and the product's own post-boot inventory --
  ``efa_inventory_active_count`` at the expected count with ``mismatch=0`` on
  the new boot; without a reboot the probe failure stands and the cleanup
  fails. The trap's record (``post_repair_failure``) is read while the product
  repairs, before the reboot can take the probe; a probe lost before that read
  leaves the product's own SUCCEEDED-then-FAILED record as the trap's witness.

A post-repair validation can also fail at its bound without ever observing
the re-unbound function (live 2026-10-27 07:08-07:51Z, HyperPod: REMEDIATE
SUCCEEDED 07:10:00, the trap re-unbound at 07:10:04, RESTART_EFA_DEVICE_PLUGIN
passed on the plugin's unchanged allocatable, VALIDATE_FABRIC waited out the
600 s cap with no fresh inventory sample). The product then withholds the
reboot it asked for and hands the node to an operator
(``validation_evidence.withheld_escalation`` -> ``workflow-support-after-``,
FREEZE_EVIDENCE + ESCALATE_SUPPORT, the chain's cordon left on the node) --
designed, but not the fresh-evidence failure the variant claims: the variant
grades the hand-off's shape, fails ``post_repair_failure_observed`` with the
product-side question in ``product_gap``, and its cleanup hands the node back
through the product's validated restore once the function is bound again
(``hand_back``, LIFO after the release). The same run showed the Node Agent's
receipt is read from ``node_results`` of the step execution / the remote
command's ``step.operation`` (``general_live_efa_escalation.node_result``), not
a top-level ``operation`` the command record never had.

The trap's own timing lost a race too (live 2026-11-06, HyperPod, R23
validation): REMEDIATE_EFA_DRIVER SUCCEEDED (bound 16/16),
RESTART_EFA_DEVICE_PLUGIN passed, TRIGGER_HEALTH_SNAPSHOT restarted the metrics
and then the host collector right away, and the restarted host collector's
first sample -- 16/16, about 3.6 s after the bind the trap had seen -- reached
the product before the trap's fixed 5 s grace ran out. VALIDATE_FABRIC
SUCCEEDED on it (a sample newer than the last recovery step), the record ended
SUCCEEDED, the late re-unbind read as a new incident whose remediation then
held (the trap is one-shot), and no reboot-after successor followed; earlier
PASS runs had had the opposite order. The trap now fires on that snapshot
itself: once the product's bind shows (polled every 0.25 s), it reads
``gpu-fault-host-collector.service``'s state (``systemctl show``: InvocationID,
main PID, start and activation timestamps -- a read, never a restart, stop or
change of the product's unit) every 0.2 s and re-unbinds the moment the
restart shows, so the restarted collector's first tick, about two seconds
after its start, already reads the function unbound. It never fires sooner
than ``TRAP_FLOOR_SECONDS`` after the bind (the Node Agent's read-back is done
before the product restarts anything; the floor guards a restart nobody
planned) and, when no restart shows, falls back to the old
``TRAP_GRACE_SECONDS``; its record names the trigger (``collector_restart`` or
``grace_fallback``) and the seconds from the bind to the re-unbind.

Both keep the fixture scoped to ONE function, node-reversible and
deadman-restorable (clear the override, rebind), and both grade only on the
product's own records. They are the dangerous-live tail of the lane: risk H,
flag X, ``EFA_DRIVER_UNBIND`` + ``NODE_REBOOT`` approvals, run last.

Reusable: :class:`EfaDriverFixture` wraps ``general_live_efa_probe.py`` so any
category needing one allow-listed EFA function held unbound (or trapped after a
rebind) can call ``hold_unbind`` / ``trap_unbind`` / ``release`` with the same
deadman, LIFO cleanup and reboot-aware release.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.e2e.regional.collector_action_guard import require_action_time
from scripts.e2e.regional.general_live_common import (
    STORE_POLL_SECONDS,
    activity,
    restore_incidents,
    unsettled_workflows,
    wait_for_incident,
)
from scripts.e2e.regional.general_live_efa_escalation import (
    EFA_METRIC,
    PLUGIN_STEP,
    REMEDIATE_STEP,
    VALIDATE_STEP,
    WITHHELD,
    Chain,
    efa_chain,
    escalation_record,
    escalation_shape_errors,
    exit_errors,
    handoff_shape_errors,
    incident_state,
    isolation_errors,
    ladder_stopped,
    node_result,
    post_repair_verdict,
    reboot_planned_errors,
    successors_of,
    trap_fired_errors,
    trap_record,
)
from scripts.e2e.regional.general_live_fixture import (
    VariantBlocked,
    VariantContext,
    VariantInconclusive,
    utc_now,
)
from scripts.e2e.regional.general_live_gpu_replay_close import (
    JUDGED_BY_REBOOT,
    close_after_probe_loss,
    node_boot_id,
    node_facts,
    unreachable_detail,
)
from scripts.e2e.regional.general_live_registry import NATIVE, VariantSpec
from scripts.e2e.regional.general_live_verdicts import (
    TERMINAL_WORKFLOWS,
    executions,
    finding_incidents,
    node_isolated,
    parse_time,
    samples,
    verdict,
)
from scripts.e2e.regional.host_probe_fixture import HostProbeError
from scripts.e2e.regional.probes.general_live_efa_probe import (
    MIN_FLOOR_SECONDS,
    SNAPSHOT_COLLECTOR_UNIT,
    TRIGGER_COLLECTOR_RESTART,
    TRIGGER_GRACE_FALLBACK,
)

PROBES = Path(__file__).with_name("probes")
#: The on-node EFA driver fixture probe (sibling of the shared node probe).
EFA_DRIVER_PROBE = PROBES / "general_live_efa_probe.py"
CASE_013 = "GF-GENERAL-LIVE-013"
EFA_DRIVER_UNBIND = "EFA_DRIVER_UNBIND"
NODE_REBOOT = "NODE_REBOOT"
#: The host collector's EFA inventory samples the verdict reads.
EFA_METRICS = {
    EFA_METRIC,
    "efa_inventory_active_count",
    "efa_inventory_expected_count",
    "efa_inventory_missing_count",
    "efa_inventory_discovered_count",
}
#: Detection budget (two confirmation ticks + ingestion + planning).
DETECTION_BUDGET = 420
#: From the incident to the ladder's decision: the remediation record ends
#: (the failed bind within seconds; a post-repair RESTART_EFA_DEVICE_PLUGIN or
#: VALIDATE_FABRIC after its own evidence wait, up to the 600 s step cap) and
#: the reboot-after successor is planned. Evidence of the record before/at
#: the reboot; the settled wait below is what the verdict needs.
RUNG_BUDGET = 1500
#: The approved reboot chain (reboot, validations, controlled restore) may take
#: this long to settle; the runner waits it out and never cancels it.
REBOOT_SETTLE_BUDGET = 2400
#: The predecessor's closure follows the successor's restore within a reconcile.
SOURCE_CLOSE_BUDGET = 180
#: How often the trap's own record is read while the product repairs.
TRAP_POLL_SECONDS = 10
#: The trap's re-unbind after the product's bind: on the host collector's
#: restart for the post-repair snapshot, never sooner than the floor, at the
#: grace (the old fixed delay) when no restart shows.
TRAP_FLOOR_SECONDS = MIN_FLOOR_SECONDS
TRAP_GRACE_SECONDS = 5


class EfaDriverFixture:
    """One allow-listed EFA function held unbound / trapped, with cleanup+deadman.

    Wraps ``general_live_efa_probe.py`` on a node-pinned probe Pod. The target
    function is chosen from the node's real EFA data functions (the lowest BDF
    with a bound ``efa`` driver and more than one present), recorded, and the
    coordinator's maintenance BDF allow-list must cover it. ``release`` is
    pushed on the context's LIFO cleanup, so it runs (and verifies the function
    is bound to ``efa`` again) before the probe Pod is torn down; a
    ``systemd-run`` deadman restores the baseline even if the runner, the Pod
    and the control plane all vanish. A release the probe can no longer run
    because the product rebooted the node is judged by the boot and the
    product's post-boot inventory (``_release_after_probe_loss``).
    """

    def __init__(self, ctx: VariantContext) -> None:
        self.ctx = ctx
        self._bdf: str | None = None
        self._injected = False
        #: The boot the fixture was armed on (the survey's, then the receipt's).
        self.boot_id = ""
        self.armed_at: datetime | None = None
        #: The product's expected EFA count (the variant sets it); the
        #: baseline bound count stands in until then.
        self.expected_count: int | None = None
        self.baseline_bound: int | None = None
        self.probe_lost: str | None = None
        self.handle = ctx.host_probe(
            "efa-driver",
            script=EFA_DRIVER_PROBE,
            deadline_seconds=ctx.spec.max_minutes * 60 + 1800,
        )
        ctx.cleanup.push("efa-driver", self.release)

    def _execute(self, *arguments: str, timeout: int = 180) -> dict[str, Any]:
        return self.handle.execute(
            *arguments, "--run-id", self.ctx.run_id, timeout=timeout
        )

    def survey(self) -> dict[str, Any]:
        bdf = self.target_bdf()
        value = self._execute("summary", "--pci-bdf", bdf, timeout=120)
        self.boot_id = str(value.get("boot_id") or self.boot_id)
        if isinstance(value.get("bound_count"), int):
            self.baseline_bound = int(value["bound_count"])
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

    def _armed(self, value: dict[str, Any]) -> None:
        self._injected = True
        self.armed_at = utc_now()
        self.boot_id = str(value.get("boot_id") or self.boot_id)
        if isinstance(value.get("baseline_bound_count"), int):
            self.baseline_bound = int(value["baseline_bound_count"])

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
        self._armed(value)
        self.ctx.evidence.write("efa-driver-hold", value)
        return value

    def trap_unbind(
        self,
        *,
        restore_seconds: int,
        grace_seconds: int,
        trap_window_seconds: int,
        floor_seconds: int = TRAP_FLOOR_SECONDS,
    ) -> dict[str, Any]:
        """post_repair_failure: unbind, and re-unbind after the product rebinds
        -- on the host collector's restart for its post-repair snapshot, never
        sooner than ``floor_seconds`` after the bind, at ``grace_seconds`` when
        no restart shows."""

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
            "--floor-seconds",
            str(int(floor_seconds)),
            "--trap-window-seconds",
            str(int(trap_window_seconds)),
            timeout=180,
        )
        self._armed(value)
        self.ctx.evidence.write("efa-driver-trap", value)
        return value

    def summary(self) -> dict[str, Any]:
        value = self._execute("summary", "--pci-bdf", self.target_bdf(), timeout=120)
        self.ctx.evidence.write("efa-driver-state", value)
        return value

    def summary_or_loss(self) -> dict[str, Any]:
        """A reading, or the record of a probe the node took away.

        The product's reboot (and a HyperPod quiesce) terminates the probe Pod
        mid-run; a reading after that is not a fixture failure but the end of
        the fixture's own observations, kept as ``probe_lost`` with the node's
        boot id at that moment. Grading continues on the product's records.
        """

        try:
            return self.summary()
        except HostProbeError as error:
            self.probe_lost = f"{type(error).__name__}: {error}"
            record = {
                "probe_lost": self.probe_lost,
                "observed_at": utc_now().isoformat(),
                "boot_id_armed": self.boot_id,
                "boot_id_now": node_boot_id(self.ctx),
            }
            self.ctx.evidence.write("efa-driver-probe-lost", record)
            return record

    def release(self) -> dict[str, Any]:
        """Restore the baseline (LIFO cleanup). Raises if the node is not clean.

        Through the probe when it answers; otherwise by the node's reboot
        (``_release_after_probe_loss``) -- never a fixture that persists across
        a boot, never a cleanup that counts a dead Pod as residue.
        """

        if self._bdf is None:
            return {"released": False, "reason": "no target was selected"}
        try:
            value = self._execute("release", "--pci-bdf", self._bdf, timeout=180)
        except HostProbeError as error:
            value = self._release_after_probe_loss(error)
        self.ctx.evidence.write("efa-driver-release", value)
        return value

    def _release_after_probe_loss(self, error: HostProbeError) -> dict[str, Any]:
        """Judge a release the probe could not perform.

        ``close_after_probe_loss`` polls the Node object, retries the release
        through the probe once the node is settled Ready (a quiesce window
        ends with the same Pod alive) and, when the node's boot id changed
        against the one the fixture was armed on and the probe still refuses,
        closes by the reboot: the sysfs override, the ``systemd-run`` deadman
        and trap units and the ``/run`` state cannot have survived it. The
        reboot alone is not the proof -- the product's own inventory of the
        new boot must read every function bound (``_node_after_state``), or
        the cleanup fails as unproven. A node that never rebooted keeps the
        probe failure as the cleanup error.
        """

        bdf = str(self._bdf)
        receipts: dict[str, Any] = {}
        closed = close_after_probe_loss(
            self.ctx,
            error,
            run_close=lambda: self._execute("release", "--pci-bdf", bdf, timeout=180),
            node_boot_id=self.boot_id,
            receipts=receipts,
        )
        if closed is None:
            raise HostProbeError(
                f"EFA driver fixture release could not run ({error}); "
                + unreachable_detail(receipts.get("close_after_probe_loss") or [])
            )
        if closed.get("judged_by") != JUDGED_BY_REBOOT:
            # The node came back without a reboot and the retry released it.
            return {**closed, "release_retry": receipts}
        rebooted = dict(closed.get("rebooted") or {})
        after = self._node_after_state()
        if after["errors"]:
            raise HostProbeError(
                f"the node rebooted since the EFA fixture was armed ({rebooted}) but "
                "its inventory does not prove the function bound: "
                + "; ".join(after["errors"])
            )
        return {
            "clean": True,
            "judged_by": JUDGED_BY_REBOOT,
            "rebooted": rebooted,
            "node_after": after,
            "release_error": receipts.get("close_error"),
            "release_retry": receipts.get("close_after_probe_loss"),
            "note": (
                "the node rebooted since the fixture was armed; its driver_override "
                "(sysfs), the systemd-run deadman/trap units and the /run state "
                "cannot have survived the boot, and the product's inventory of the "
                "new boot reads every EFA function bound"
            ),
        }

    def _node_after_state(self) -> dict[str, Any]:
        """The product's reading of the node on the boot it returned on.

        Waits (one host summary plus a few intervals) for an
        ``efa_inventory_active_count`` sample at the expected count with
        ``efa_inventory_mismatch=0`` observed at or after the Node's Ready
        transition of the new boot -- the first post-boot batches may precede
        kubelet's Ready, so the wait takes the next one.
        """

        expected = (
            self.expected_count
            if self.expected_count is not None
            else self.baseline_bound
        )
        facts = node_facts(self.ctx)
        ready_since = _moment(facts.ready_since)
        since = self.armed_at or utc_now()

        def fresh(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if ready_since is None:
                return items
            return [
                item
                for item in items
                if (moment := _moment(item.get("observed_at"))) is not None
                and moment >= ready_since
            ]

        def accept() -> dict[str, Any] | None:
            current = activity(self.ctx, since, metrics=EFA_METRICS)
            active = fresh(_efa_samples(current, "efa_inventory_active_count"))
            mismatch = fresh(_efa_samples(current, EFA_METRIC))
            if (
                active
                and mismatch
                and (expected is None or active[-1]["value"] == expected)
                and mismatch[-1]["value"] == 0
            ):
                return {"active": active[-1], "mismatch": mismatch[-1]}
            return None

        value = self.ctx.wait_until(
            accept,
            timeout_seconds=self.ctx.host_summary + self.ctx.host_interval * 8,
            poll_seconds=max(10, self.ctx.host_interval),
            name="efa-driver-node-after",
        )
        errors = (
            []
            if value is not None
            else [
                f"no efa_inventory_active_count={expected} with mismatch=0 sample "
                f"of the new boot (Ready since {facts.ready_since or 'unknown'})"
            ]
        )
        return {
            "boot_id": facts.boot_id,
            "ready": facts.ready,
            "ready_since": facts.ready_since,
            "expected_count": expected,
            "inventory": value,
            "errors": errors,
        }


# --------------------------------------------------------------------------- #
# Reading the product (the chain checks live in general_live_efa_escalation)
# --------------------------------------------------------------------------- #
def _moment(value: Any) -> datetime | None:
    try:
        moment = parse_time(value)
    except ValueError:
        return None
    if moment is not None and moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


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
        item for item in _efa_samples(detection, EFA_METRIC) if item["value"] >= 1
    ]
    if not confirmed:
        return ["no efa_inventory_mismatch=1 sample reached the product"]
    modes = sorted(
        {(item.get("labels") or {}).get("failure_mode", "?") for item in confirmed}
    )
    return [] if modes == [mode] else [f"failure_mode {modes}, expected [{mode!r}]"]


def _incidents_on_node(observed: dict[str, Any], node: str) -> list[dict[str, Any]]:
    """Every incident of the read that names ``node`` (the chain's and its
    successors'), for the hand-back."""

    return [
        item
        for item in observed.get("incidents") or []
        if node in (item.get("node_ids") or [])
    ]


def hand_back(ctx: VariantContext, incidents: list[dict[str, Any]]) -> dict[str, Any]:
    """Cleanup: the product's validated restore if the node is still isolated.

    The reboot exit releases the node itself (nothing to do). A hand-off --
    the product withheld its reboot and left the chain's cordon with an
    ESCALATED incident -- or a successor that ended short of RESTORE_SCHEDULING
    leaves the node to an operator; this is that operator's exit, on fresh
    evidence, after the fixture's release rebound the function (the LIFO
    stack runs this after the release).
    """

    node = ctx.regional.node_snapshot(ctx.settings.node)
    if not node_isolated(node):
        return {"skipped": True, "reason": "the node is not isolated"}
    if not incidents:
        return {"skipped": True, "reason": "no incident of this run to restore"}
    return {
        "restored": restore_incidents(
            ctx,
            incidents,
            reason=(
                f"{CASE_013}: the EFA function is bound again (fixture released); "
                "hand the node back on fresh evidence"
            ),
        )
    }


def _await_trap(
    ctx: VariantContext, fixture: EfaDriverFixture, *, window_seconds: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read the trap's record while the product repairs, before the reboot.

    Polls the probe every ``TRAP_POLL_SECONDS`` until the trap fired (or gave
    up with a reason), the probe was lost (the reboot or a quiesce took the
    Pod: the reading says so and grading moves to the product's records), or
    the trap window passed. Every reading joins the oracle.
    """

    readings: list[dict[str, Any]] = []

    def accept() -> dict[str, Any] | None:
        reading = fixture.summary_or_loss()
        readings.append(reading)
        if reading.get("probe_lost"):
            return {"lost": reading}
        record = trap_record(reading)
        if record.get("fired") is True or record.get("reason"):
            return {"record": record, "reading": reading}
        return None

    value = ctx.wait_until(
        accept,
        timeout_seconds=window_seconds,
        poll_seconds=TRAP_POLL_SECONDS,
        name="post-repair-trap",
    )
    return readings, dict(value or {})


def _await_rung(ctx: VariantContext, since: datetime, *, name: str) -> dict[str, Any]:
    """The record at the ladder's decision (evidence ``<name>-chain``).

    Accepted once the remediation record is terminal and, when it FAILED, its
    successor is on the record; the settled wait that follows is what the
    verdict needs, this is the record before and at the reboot.
    """

    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = activity(ctx, since, metrics=EFA_METRICS)
        pair = efa_chain(last)
        if pair is None:
            return None
        workflow = pair[1]
        status = str(workflow.get("status"))
        if status not in TERMINAL_WORKFLOWS:
            return None
        if status == "FAILED" and not successors_of(
            last, str(workflow.get("request_id"))
        ):
            return None
        return last

    value = ctx.wait_until(
        accept,
        timeout_seconds=RUNG_BUDGET,
        poll_seconds=STORE_POLL_SECONDS,
        name=name,
    )
    document = value if value is not None else last
    ctx.evidence.write(
        f"{name}-chain",
        {
            "accepted": value is not None,
            "boot_id_now": node_boot_id(ctx),
            "incidents": document.get("incidents") or [],
            "workflows": document.get("workflows") or [],
        },
    )
    return document


def _await_recovered(
    ctx: VariantContext, since: datetime, expected: int, baseline_boot: str
) -> dict[str, Any] | None:
    """Poll until the chain settled: the node healthy on a new boot, or the
    ladder stopped without one.

    The reboot is the spec'd branch of both variants, so the healthy exit
    needs a changed boot id: the node returns on a new boot with full EFA
    inventory, no gpu-fault taint/annotation, and no workflow still running
    (``healthy``). A ladder that ended without rebooting (``ladder_stopped``
    on two consecutive polls) is returned as well, unhealthy, so the stages
    name the deviation instead of waiting the whole window for a reboot.
    """

    stopped = 0

    def accept() -> dict[str, Any] | None:
        nonlocal stopped
        current = activity(ctx, since, metrics=EFA_METRICS)
        node = ctx.regional.node_snapshot(ctx.settings.node)
        if unsettled_workflows(current):
            stopped = 0
            return None
        if ladder_stopped(current):
            stopped += 1
            if stopped >= 2:
                return {"activity": current, "node": node, "healthy": False}
            return None
        stopped = 0
        if node.get("boot_id") in {None, "", baseline_boot}:
            return None
        if node_isolated(node):
            return None
        active = _efa_samples(current, "efa_inventory_active_count")
        mismatch = _efa_samples(current, EFA_METRIC)
        if (
            active
            and active[-1]["value"] == expected
            and mismatch
            and mismatch[-1]["value"] == 0
        ):
            return {"activity": current, "node": node, "healthy": True}
        return None

    value: dict[str, Any] | None = ctx.wait_until(
        accept,
        timeout_seconds=REBOOT_SETTLE_BUDGET,
        poll_seconds=max(15, ctx.host_interval),
        name="efa-driver-settled",
    )
    return value


def _settled_records(
    ctx: VariantContext, since: datetime, recovered: dict[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The records and the Node the exit is judged on.

    After a healthy exit the EFA incident's closure by its successor may lag
    the restore by a reconcile; a short bounded re-read takes it.
    """

    observed = (
        recovered["activity"]
        if recovered
        else activity(ctx, since, metrics=EFA_METRICS)
    )
    node = (
        recovered["node"]
        if recovered
        else ctx.regional.node_snapshot(ctx.settings.node)
    )
    pair = efa_chain(observed)
    if pair is None or not (recovered and recovered.get("healthy")):
        return observed, node
    incident_id = str(pair[0].get("incident_id"))
    if incident_state(observed, incident_id) == "RECOVERED":
        return observed, node

    def accept() -> dict[str, Any] | None:
        current = activity(ctx, since, metrics=EFA_METRICS)
        return current if incident_state(current, incident_id) == "RECOVERED" else None

    later = ctx.wait_until(
        accept,
        timeout_seconds=SOURCE_CLOSE_BUDGET,
        poll_seconds=STORE_POLL_SECONDS,
        name="efa-driver-source-closed",
    )
    return (later if later is not None else observed), node


def _recovered_errors(recovered: dict[str, Any] | None) -> list[str]:
    if recovered and recovered.get("healthy"):
        return []
    if recovered:
        return [
            "the ladder ended without rebooting the node (every workflow terminal, "
            "no RESTART_NODE SUCCEEDED, the incident past ACTION_PENDING)"
        ]
    return ["the node did not return healthy on a new boot within the window"]


def _oracle_no_global_unload(records: list[dict[str, Any]], expected: int) -> list[str]:
    """From the probe oracle: the module stayed loaded, no function went to 0.

    A reading the reboot took (``probe_lost``) carries no counts and is not
    a reading of the module.
    """

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


def _settle(
    ctx: VariantContext,
    *,
    started: datetime,
    expected: int,
    baseline_boot: str,
    rung_name: str,
) -> tuple[
    dict[str, Any] | None, dict[str, Any], dict[str, Any], Chain | None, dict[str, Any]
]:
    """The ladder's decision, then the settled chain: ``(recovered, observed,
    node, chain, settled record)``; evidence ``efa-driver-escalation`` (the
    record before/at the reboot) and ``efa-driver-settled``."""

    rung = _await_rung(ctx, started, name=rung_name)
    ctx.evidence.write(
        "efa-driver-escalation",
        escalation_record(
            rung,
            efa_chain(rung),
            boot_before=baseline_boot,
            boot_now=node_boot_id(ctx),
        ),
    )
    recovered = _await_recovered(ctx, started, expected, baseline_boot)
    observed, node_now = _settled_records(ctx, started, recovered)
    pair = efa_chain(observed)
    settled = escalation_record(
        observed,
        pair,
        boot_before=baseline_boot,
        boot_now=str(node_now.get("boot_id") or ""),
    )
    ctx.evidence.write("efa-driver-settled", settled)
    return recovered, observed, node_now, pair, settled


REBOOT_NOTES = [
    "the failed EFA chain hands the node to the REBOOT_NODE rung: "
    "workflow-reboot-after-<record> cordons, quarantines and reboots the node by "
    "design (escalation.py _escalation_operations) -- its QUARANTINE is the "
    "chain's own cordon, not isolation beyond it -- and the reboot clears the "
    "fixture",
    "graded on the product's records before/at the reboot and on the settled exit; "
    "the override, deadman/trap units and /run state cannot survive the boot, so "
    "the successor's validations grade the real node and both incidents RECOVER "
    "(the successor's restore closes its QUARANTINED predecessor)",
]


WITHHELD_NOTES = [
    "the post-repair validation failed at its bound without observing the re-broken "
    "function, so the product withheld the REBOOT_NODE it asked for (designed: "
    "validation_evidence.withheld_escalation, escalation.py "
    "VALIDATION_EVIDENCE_MISSING_STAGE) and opened workflow-support-after-<record> "
    "(evidence + ticket); the node kept the chain's cordon for an operator",
    "the variant's claim -- a post-repair step FAILED on fresh evidence of the "
    "re-unbound function -- was not demonstrated; the cleanup released the function "
    "and handed the node back through the product's validated restore",
]
PRODUCT_GAP_WITHHELD = (
    "the post-repair VALIDATE_FABRIC ended at its per-step cap with no fresh "
    "efa_inventory_active_count after the recovery steps completed "
    "(adapters/gpu_validation.py inventory_observation_window: a sample newer than "
    "the last recovery step), although TRIGGER_HEALTH_SNAPSHOT had restarted the host "
    "collector (node_agent/operations/diagnostics.py _trigger_health_snapshot) and the "
    "function was unbound again; RESTART_EFA_DEVICE_PLUGIN had passed on the plugin's "
    "unchanged allocatable (adapters/kubernetes/node_operations.py "
    "_restart_device_plugin_node: allocatable >= expected -> already_healthy). The "
    "withheld reboot is the designed response to missing evidence; whether the host "
    "collector failed to deliver the post-action inventory sample is the open "
    "product question -- see failed_post_repair_step.details.node_pending"
)


def _observed_errors(
    failed_step: dict[str, Any] | None, verdict_kind: str
) -> list[str]:
    """The post-repair failure looked at the node (``OBSERVED``), as the
    variant claims; a bound that failed it first is the product's withheld
    path and the claim stands undemonstrated."""

    if failed_step is None:
        return []
    if verdict_kind != WITHHELD:
        return []
    details = failed_step.get("details") or {}
    pending = details.get("node_pending") or {}
    return [
        f"the post-repair {failed_step.get('operation')} failed at its bound without "
        f"observing the re-unbound function (node_pending={pending or 'none'}; "
        f"{str(failed_step.get('error') or '')[:160]}); the product withheld the "
        "reboot and handed the node to an operator -- the fresh-evidence failure "
        "the variant claims was not demonstrated"
    ]


def _escalation_extra(
    settled: dict[str, Any], fixture: EfaDriverFixture
) -> dict[str, Any]:
    return {
        "boot_id_after": settled["boot_id_now"],
        "rebooted": settled["rebooted"],
        "escalation": {
            "successors": settled["successors"],
            "restart_node_executions": settled["restart_node_executions"],
        },
        "probe_lost": fixture.probe_lost,
    }


def bind_failure(ctx: VariantContext) -> dict[str, Any]:
    """The product's bind of one held EFA function fails; its receipt is
    independent and it takes the approved reboot branch; no global efa unload."""

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
    injected = fixture.hold_unbind(restore_seconds=ctx.spec.max_minutes * 60)
    oracle = [fixture.summary()]
    detection = wait_for_incident(
        ctx,
        started,
        EFA_METRIC,
        timeout_seconds=DETECTION_BUDGET,
        metrics=EFA_METRICS,
        name="bind-failure-detection",
    )
    oracle.append(fixture.summary_or_loss())
    ctx.evidence.append_oracle(
        [{"pci_bdf": bdf, "injection": injected, "readings": oracle}],
        source="efa-driver",
    )
    if not finding_incidents(detection, EFA_METRIC):
        raise VariantInconclusive(
            f"the product never minted an efa_inventory_mismatch finding from the "
            f"unbound function {bdf}"
        )
    incidents[:] = _incidents_on_node(detection, node)
    recovered, observed, node_now, pair, settled = _settle(
        ctx,
        started=started,
        expected=expected,
        baseline_boot=baseline_boot,
        rung_name="bind-failure-rung",
    )
    incidents[:] = _incidents_on_node(observed, node) or incidents
    remediation = executions(pair[1], REMEDIATE_STEP) if pair else []
    failed = [item for item in remediation if item.get("status") == "FAILED"]
    receipt = node_result(
        observed, pair[1] if pair else None, operation=REMEDIATE_STEP, node=node
    )
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
        "approved_reboot_branch": reboot_planned_errors(observed),
        "escalation_shape": escalation_shape_errors(observed, pair),
        "not_isolated_beyond_chain": isolation_errors(observed, pair),
        "recovered": _recovered_errors(recovered)
        + exit_errors(observed, pair, node_now),
    }
    return _result(
        stages,
        notes=[
            f"{bdf} held by driver_override={injected.get('driver_override')!r} and "
            "unbound; the product's targeted bind of this one function fails (ENODEV)",
            *REBOOT_NOTES,
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
            **_escalation_extra(settled, fixture),
        },
    )


def _succeeded_remediation_errors(
    workflow: dict[str, Any],
    observed: dict[str, Any],
    *,
    bdf: str,
    node: str,
) -> tuple[list[str], dict[str, Any]]:
    """The action's own receipt: REMEDIATE_EFA_DRIVER SUCCEEDED, it bound the BDF."""

    remediation = executions(workflow, REMEDIATE_STEP)
    terminal = [
        item for item in remediation if item.get("status") in {"SUCCEEDED", "FAILED"}
    ]
    receipt = node_result(observed, workflow, operation=REMEDIATE_STEP, node=node)
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


def _trap_note(bdf: str, record: dict[str, Any], injected: dict[str, Any]) -> str:
    """What the trap did, in the words of its own record: on the collector's
    restart for the post-repair snapshot, or on the grace fallback."""

    after = record.get("seconds_after_bind")
    delay = (
        f" {after}s"
        if isinstance(after, (int, float)) and not isinstance(after, bool)
        else ""
    )
    trigger = record.get("trigger")
    if trigger == TRIGGER_COLLECTOR_RESTART:
        why = (
            f", on the {SNAPSHOT_COLLECTOR_UNIT} restart of its post-repair snapshot "
            "(TRIGGER_HEALTH_SNAPSHOT)"
        )
    elif trigger == TRIGGER_GRACE_FALLBACK:
        grace = record.get("grace_seconds", injected.get("grace_seconds"))
        why = (
            f", on its {grace}s grace fallback (no {SNAPSHOT_COLLECTOR_UNIT} restart "
            "showed between the bind and the grace)"
        )
    elif record.get("fired") is True:
        why = " (its record names no trigger)"
    else:
        why = " (the trap's own record was not read)"
    return (
        f"{bdf} unbound; a bounded one-shot trap re-unbound it{delay} after the "
        f"product's bind{why}, so the action receipt is SUCCEEDED and the "
        "post-repair step fails on fresh evidence"
    )


def post_repair_failure(ctx: VariantContext) -> dict[str, Any]:
    """The product binds the function back (SUCCEEDED receipt), the fixture
    re-unbinds it, and the post-repair step fails independently and escalates."""

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
    trap_window = DETECTION_BUDGET + 180
    injected = fixture.trap_unbind(
        restore_seconds=ctx.spec.max_minutes * 60,
        grace_seconds=TRAP_GRACE_SECONDS,
        floor_seconds=TRAP_FLOOR_SECONDS,
        trap_window_seconds=trap_window,
    )
    oracle = [fixture.summary()]
    detection = wait_for_incident(
        ctx,
        started,
        EFA_METRIC,
        timeout_seconds=DETECTION_BUDGET,
        metrics=EFA_METRICS,
        name="post-repair-detection",
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
    # The trap's own record, read while the product repairs and before its
    # reboot can take the probe; the readings join the oracle.
    readings, trap_outcome = _await_trap(ctx, fixture, window_seconds=trap_window + 60)
    oracle.extend(readings)
    ctx.evidence.append_oracle(
        [
            {
                "pci_bdf": bdf,
                "injection": injected,
                "readings": oracle,
                "trap": trap_outcome.get("record") or {},
                "probe_lost": fixture.probe_lost,
            }
        ],
        source="efa-driver",
    )
    recovered, observed, node_now, pair, settled = _settle(
        ctx,
        started=started,
        expected=expected,
        baseline_boot=baseline_boot,
        rung_name="post-repair-rung",
    )
    incidents[:] = _incidents_on_node(observed, node) or incidents
    workflow = pair[1] if pair else {}
    succeeded_errors, receipt = _succeeded_remediation_errors(
        workflow, observed, bdf=bdf, node=node
    )
    post_errors, failed_step = _post_repair_failure_errors(workflow)
    trap_errors, trap_witness = trap_fired_errors(
        trap_outcome, post_repair_failed=not post_errors
    )
    verdict_kind = post_repair_verdict(failed_step) if failed_step else ""
    withheld = verdict_kind == WITHHELD
    stages = {
        "driver_unbound": _failure_mode_errors(detection, "DRIVER_UNBOUND"),
        "action_succeeded": succeeded_errors,
        "independent_post_repair_failure": post_errors,
        "post_repair_failure_observed": _observed_errors(failed_step, verdict_kind),
        "trap_fired": trap_errors,
        "no_global_unload": _oracle_no_global_unload(oracle, expected),
        "not_isolated_beyond_chain": isolation_errors(observed, pair),
    }
    if withheld:
        # The product kept the reboot it asked for (designed) and handed the
        # node to an operator; the hand-off's shape is graded, the reboot
        # stages are not owed, and the cleanup's validated restore releases
        # the node the chain left cordoned.
        stages["escalation"] = []
        stages["escalation_shape"] = handoff_shape_errors(observed, pair)
        stages["recovered"] = []
    else:
        stages["escalation"] = reboot_planned_errors(observed)
        stages["escalation_shape"] = escalation_shape_errors(observed, pair)
        stages["recovered"] = _recovered_errors(recovered) + exit_errors(
            observed, pair, node_now
        )
    trap = trap_outcome.get("record") or {}
    return _result(
        stages,
        notes=[
            _trap_note(bdf, trap, injected),
            *(WITHHELD_NOTES if withheld else REBOOT_NOTES),
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
                "verdict": verdict_kind or None,
                "details": {
                    key: value
                    for key, value in ((failed_step or {}).get("details") or {}).items()
                    if key in {"node_pending", "node_failures", "inventory_evidence"}
                },
            },
            "trap": trap,
            "trap_trigger": trap.get("trigger"),
            "trap_witness": trap_witness,
            "product_gap": PRODUCT_GAP_WITHHELD if withheld else None,
            "node_isolated_at_exit": node_isolated(node_now),
            **_escalation_extra(settled, fixture),
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
            "the BDF; the record ends FAILED and the product's ladder opens exactly one "
            "workflow-reboot-after-<record> (cordon, QUARANTINE, RESTART_NODE, "
            "validations, RESTORE_SCHEDULING -- the approved branch, which clears the "
            "override), RESTART_NODE runs once, no REPLACE_NODE; no global efa unload; "
            "node returns healthy on a new boot, released, both incidents RECOVERED"
        ),
        execute=bind_failure,
        cleanup=(
            "stop the trap, clear the driver_override and rebind the function "
            "(deadman-backed); the product's reboot chain is waited out, never cancelled; "
            "after the product's reboot the override, deadman and /run state are gone "
            "by construction and the release is judged by the new boot id plus the "
            "product's post-boot inventory (every function bound); otherwise fail "
            "loudly unless the function is bound to efa with no override"
        ),
    ),
    _efa_driver_variant(
        "post_repair_failure",
        injection=(
            "unbind one allow-listed EFA data function, and a bounded one-shot trap "
            "re-unbinds it after the product's REMEDIATE_EFA_DRIVER bind -- when the "
            "product's post-repair snapshot restarts the host collector (read-only "
            f"watch), never sooner than {TRAP_FLOOR_SECONDS} s after the bind, at a "
            f"{TRAP_GRACE_SECONDS} s grace when no restart shows (no driver_override) "
            "-- so the action receipt SUCCEEDS and the post-repair step fails"
        ),
        expected=(
            "REMEDIATE_EFA_DRIVER SUCCEEDED (rebound_pci_bdfs=[BDF]) then an independent "
            "post-repair failure (RESTART_EFA_DEVICE_PLUGIN or VALIDATE_FABRIC) on fresh "
            "evidence -> exactly one workflow-reboot-after-<record> (cordon, QUARANTINE, "
            "RESTART_NODE once, validations, RESTORE_SCHEDULING), no REPLACE_NODE; the "
            "trap's record read before the reboot; no global efa unload; node returns "
            "healthy on a new boot, released, both incidents RECOVERED"
        ),
        execute=post_repair_failure,
        cleanup=(
            "stop the trap, clear any override and rebind the function (deadman-backed); "
            "the product's reboot chain is waited out, never cancelled; after the "
            "product's reboot the trap, deadman and /run state are gone by construction "
            "and the release is judged by the new boot id plus the product's post-boot "
            "inventory; otherwise fail loudly unless the function is bound to efa"
        ),
    ),
]

__all__ = ["EfaDriverFixture", "VARIANTS", "bind_failure", "post_repair_failure"]
