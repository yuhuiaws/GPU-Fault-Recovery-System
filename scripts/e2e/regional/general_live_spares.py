"""GF-GENERAL-LIVE-030 warm spare unhealthy.

Both lanes. One otherwise-idle GPU node of the lane's cluster -- the *spare
candidate*, ``--spare-candidate`` or by default the lane's ``--node-b`` (on
plain EKS the other EKS node, each lane's profile naming the other; on
HyperPod the shared second node, or the profile's dedicated spare when the
recipe passes it explicitly) -- is declared a warm spare for the variant's
duration through the product's admin CLI (``gpu-fault-admin config spare``,
confirmation ``DECLARE_WARM_SPARE_CORDON``: the spare label plus a cordon,
keyed to the provider's failure-domain group -- HyperPod's instance group, the
EKS managed node group -- and graded by the same ``HyperPodSpareCoordinator``
the plain-EKS profile's ``nodeReplace`` warm-spare handover uses). The runner
always releases it (``RELEASE_WARM_SPARE_UNCORDON``), so no spare is left
declared, reserved or isolated; the variant releases before its verdict and
records the candidate's labels, taints, cordon and spare annotations before
and after, plus a cluster-wide census of the spare label
(``warm_spare_residue.py`` is the same read for the lane runner's heal).

Because the candidate is another lane's own node on plain EKS, the runnable
variants hold every lock (``X``) and prove the candidate idle first -- Ready,
schedulable, no gpu-fault ownership/taints/spare marks, only infrastructure
Pods (the product's DaemonSets, kube-system, device plugins), no
GPU-requesting Pod, an ACTIVE Agent with fresh channels, no open incident or
workflow, and on EKS the GPU node groups' node auto repair off (the product's
own join preflight requires it; ``eks_node_repair``) -- at plan time and again
before declaring; a busy candidate is INCONCLUSIVE, never cordoned.
``not_ready`` stops kubelet through the DESTR-008 service window (an on-node
restore timer bounds the stop at 600 s; the Agent identity it pins is the EC2
instance id on eks-managed), waits for the node to leave Ready, grades, reads
the node once more to prove the grade was taken while kubelet was down and
before the on-node restore deadline, then undoes the stop *through the node*
(``service_window_node_restore``): the explicit undo is attempted, a missing
response while the node is not Ready is the expected shape -- the probe's exec
channel is kubelet itself (live 2026-10-27, plain EKS) -- the node is awaited
Ready through the API until ``expires_at`` plus a margin, the window is closed
over the restored channel as the proof, every Pod on the node is awaited Ready,
and only then is the spare released and ``restored`` judged; a node that does
not return by the deadline fails the undo naming ``restore_at``/``expires_at``
and the last condition seen. An earlier window still open on the candidate in
the case directory's ``spare-service`` -- custody an earlier run's undo never
closed -- is retired first, before anything is declared, through the harness's
own host probe and only on proof (``reclaim_service_windows``,
``service_window_retire``); one that cannot be proven stale keeps ``not_ready``
BLOCKED, naming every failed proof. Each
block-only variant makes the declared spare
ineligible for allocation in exactly one way -- NotReady, stale GPU telemetry,
a configuration error (foreign reservation or a non-AVAILABLE pool state) --
and reads the product's own spare-health grading, the exact gate
``HyperPodSpareCoordinator.health_reasons`` applies when a REPLACE_NODE
allocation evaluates a spare. The verdict asserts the spare is rejected with
the right reason class and that the product manufactured no provider recovery
(no spare-health reboot incident/workflow, no spare-health failure annotation,
no REMEDIATING/UNAVAILABLE transition it caused): NotReady/stale/config never
authorize a reboot.

Site constraint (honest scope). The periodic ``HyperPodSpareHealthController``
that turns a *trusted hardware* finding into a provider reboot, marks a spare
UNAVAILABLE after a failed reboot and runs the bounded recheck is enabled only
in a single-cluster control plane; this site's regional/multi-cluster control
plane sets ``GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER=false`` site-wide, so the
executor only grades a spare during an allocation and never autonomously
reboots an idle spare. The hardware-reboot, repair-failure and bounded-recheck
variants therefore stay NOT_RUN(CAPABILITY_ABSENT) on both clusters;
``no_spare`` (the no-provider-replace fallback) lives on the dangerous
REPLACE_NODE failover branch, whose assertions GF-REGIONAL-DESTR-008 S1 already
made live on this site, and stays NOT_RUN(COVERED_BY_REGIONAL). Every required
variant remains in the denominator.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from scripts.e2e.regional.general_live_fixture import (
    QUARANTINE_PREFIX,
    GeneralLiveSettings,
    VariantBlocked,
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
    channel_errors,
    node_provider,
    utc_now,
)
from scripts.e2e.regional.general_live_registry import (
    LIVE_REPLAY,
    NATIVE,
    VariantSpec,
    deferred_variant,
)
from scripts.e2e.regional.general_live_store import preflight_state
from scripts.e2e.regional.general_live_verdicts import operations, verdict
from scripts.e2e.regional.service_window_node_restore import (
    NodeRestoreError,
    open_windows,
    restore_through_node,
    stale_journals,
    window_deadlines,
)
from scripts.e2e.regional.warm_spare_residue import spare_residue_report

CASE = "GF-GENERAL-LIVE-030"
#: ``not_ready``: kubelet is stopped 15 s after the window is armed and the node
#: lifecycle controller marks the node Unknown after its 40 s grace, so the
#: node leaves Ready well inside this; the on-node restore timer (600 s) is the
#: independent bound on the stop itself.
NOT_READY_WAIT_SECONDS = 240
#: After an explicit kubelet restore over a working probe channel: the node
#: Ready again. When the channel was down (kubelet stopped) the wait runs to the
#: window's ``expires_at`` plus ``service_window_node_restore``'s margin
#: instead. Then every Pod on the node Ready.
READY_WAIT_SECONDS = 300
PODS_READY_WAIT_SECONDS = 300
#: After the metrics collector is resumed: a fresh success on every required
#: channel, so the next case on that node (another lane's) meets a clean
#: preflight; one GPU_METRICS interval is at most 180 s on this site.
CHANNELS_FRESH_WAIT_SECONDS = 420
#: The spare's metrics collector: one host systemd unit on both providers
#: (``general_live_node_probe.PAUSABLE_UNITS``; the dcgm-exporter container is
#: 025's target, not this one). ``stale_metrics`` pauses it through the node
#: probe's ``unit-pause`` -- SIGSTOP under the pause drop-in that re-stops a
#: product restart -- the same mechanism on HyperPod and plain EKS.
METRICS_UNIT = "gpu-fault-metrics-collector.service"
#: ``stale_metrics``: the unit produces GPU_INVENTORY (180 s) and GPU_METRICS
#: (420 s, ``general_live_collectors.SILENCE_THRESHOLDS``); the hold carries the
#: metrics channel past its threshold plus one scan before the grade.
STALE_HOLD_SECONDS = 460
#: The pause's on-node deadman (``unit-pause --seconds``): the hold, the grade
#: and the runner's slack -- sized as 031 sizes its pauses -- inside the node
#: probe's pause bound (``general_live_node_pause.MAX_PAUSE_SECONDS``, 1500 s),
#: never the variant's budget. Live 2026-10-27 on both clusters the 30-minute
#: spec budget (``max_minutes * 60`` = 1800 s) shipped as the pause and the
#: probe refused it before stopping anything ("pause seconds is outside
#: 60..1500"; the case read only the digest 6420bbc8692c83af).
METRICS_PAUSE_SECONDS = STALE_HOLD_SECONDS + 120 + 300
#: The on-node service-window helper (``warm_spare_node_probe``) honours a
#: kubelet/agent stop with an independent restart timer only for one fixture
#: identity -- the DESTR-008 service-window fixture. ``SPARE_KUBELET_STOP``
#: authorises 030 to reuse exactly that mechanism ("stop kubelet ... with a
#: host-side automatic restart timer"), so ``not_ready`` opens the window under
#: this fixture identity, not under its own running-case key (which the probe
#: rejects as "service window is not a DESTR008 fixture"). The acceptance
#: evidence still records the running case; only the on-node fixture is shared.
SERVICE_WINDOW_FIXTURE_CASE_ID = "GF-REGIONAL-DESTR-008"
#: Where ``not_ready``'s service window keeps its controller journal and its
#: host-probe record, under the case directory (``open_windows`` reads it).
SERVICE_STATE = "spare-service"
SPARE_LABEL = "gpu-fault.io/spare"
HYPERPOD_HEALTH_LABEL = "sagemaker.amazonaws.com/node-health-status"
RESERVATION_ANNOTATION = "gpu-fault.io/spare-reservation"
RESERVED_AT_ANNOTATION = "gpu-fault.io/spare-reserved-at"
POOL_STATE_ANNOTATION = "gpu-fault.io/spare-pool-state"
HEALTH_ANNOTATION = "gpu-fault.io/spare-health"
FAILURES_ANNOTATION = "gpu-fault.io/spare-health-failures"
INCIDENT_ANNOTATION = "gpu-fault.io/spare-health-incident"
HARDWARE_PREFIX = "confirmed hardware fault: "
#: Classes ``spare_health.SpareHealthReasonClass`` groups reject reasons into.
CONFIGURATION = "CONFIGURATION"
OBSERVATION = "OBSERVATION"
HARDWARE = "HARDWARE"

#: Read-only: the product records a spare-health grade reads -- blocking spare
#: markers, active GPU findings, collector observation reasons, agent
#: readiness -- plus any spare-health incident/workflow/notification for the
#: node. It never mutates; it mirrors ``HyperPodSpareCoordinator`` /
#: ``regional_spare_health``'s store-side reasons so the verdict grades the
#: declared spare exactly as the allocation gate would.
SPARE_STORE_PROBE = r"""
import json
import sys
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.models import IncidentState
from gpu_fault.store import NotFoundError
from gpu_fault.markers import blocking_spare_markers
from gpu_fault.spare_fault_evidence import fault_reasons, observation_reasons

cluster_id, node_id = sys.argv[1:3]
context = ApplicationContext.from_environment()
store = context.store
now = datetime.now(timezone.utc)

agents = [a for a in store.list_agents(cluster_id) if a.node_id == node_id]
agent_reasons = []
if len(agents) != 1:
    agent_reasons.append(f"expected one matching agent, found {len(agents)}")
elif context.fleet_registry is not None and not context.fleet_registry.readiness(
    cluster_id, [agents[0].node_id]
).ready:
    agent_reasons.append("node agent is not fleet-ready")

markers = blocking_spare_markers(store, {node_id}, cluster_id=cluster_id)
findings = store.list_gpu_findings(cluster_id, node_id, active_only=True)
fault = list(fault_reasons(markers, findings))
observation = (
    list(observation_reasons(store, cluster_id, node_id, now)) if len(agents) == 1 else []
)

incidents = []
workflows = {}
for incident in store.list_incidents_by_state(
    cluster_id, list(IncidentState), node_ids={node_id}
):
    if not str(incident.event_id).startswith("spare-health-"):
        continue
    incidents.append({
        "incident_id": incident.incident_id,
        "event_id": incident.event_id,
        "state": incident.state.value,
        "official_action": incident.official_action,
        "workflow_request_id": incident.workflow_request_id,
    })
    if incident.workflow_request_id and incident.workflow_request_id not in workflows:
        try:
            workflows[incident.workflow_request_id] = store.get_workflow(
                incident.workflow_request_id
            ).model_dump(mode="json")
        except NotFoundError:
            pass

print(json.dumps({
    "agent_reasons": agent_reasons,
    "fault_reasons": fault,
    "observation_reasons": observation,
    "spare_health_incidents": incidents,
    "spare_health_workflows": workflows,
}, sort_keys=True, default=str))
"""


class SpareSession:
    """Declares one warm spare, injects one condition, and always releases it.

    Thin wrapper over the admin warm-spare module (the operator lever) and the
    warm-spare live fixture. Tests monkeypatch this class; only its shape and
    the pure grading/verdict functions below matter for the unit tests.
    """

    def __init__(self, ctx: VariantContext) -> None:
        from pathlib import Path

        from scripts.e2e.regional.warm_spare_fixture import WarmSpareLiveFixture

        candidate = spare_candidate(ctx.settings)
        if not candidate:
            raise VariantNotRun(
                "FIXTURE_NOT_READY",
                "a warm spare needs a spare candidate: --spare-candidate <node> "
                "(default: --node-b, the lane's second node)",
            )
        self.ctx = ctx
        #: The spare node *name*; the snapshot accessor is ``node_snapshot``.
        self.node = candidate
        self.scope = spare_scope(ctx.settings)
        self.incident_id = f"inc-{ctx.run_id}"
        self.record = Path(ctx.case_dir) / "warm-spares" / f"{self.node}.json"
        self.warm = WarmSpareLiveFixture(ctx.regional, self.scope)
        self._declared = False
        #: The candidate as found, before anything of ours touched it.
        self.before: dict[str, Any] | None = None
        self.spares_before: list[str] = []

    def verify_idle(self) -> None:
        """Re-prove the candidate idle right before declaring; never cordon a busy node."""

        errors = candidate_errors(
            self.ctx.settings, self.ctx.regional, recovery=self._recovery
        )
        self.ctx.evidence.write(
            "spare-candidate",
            {"node": self.node, "scope": self.scope, "errors": errors},
        )
        if errors:
            raise VariantInconclusive(
                "spare candidate is not idle; nothing declared: " + "; ".join(errors)
            )

    def _recovery(self) -> dict[str, Any]:
        value = self.warm.cluster_recovery()
        self.ctx.evidence.write("spare-cluster-recovery", value)
        return value

    def declare(self) -> dict[str, Any]:
        from gpu_fault.admin import warm_spare

        self.before = self.node_snapshot()
        self.spares_before = self.warm.spare_nodes()
        self.ctx.evidence.write(
            "spare-before", {"node": self.before, "declared_spares": self.spares_before}
        )
        # Register the release before the declaration can touch the cluster, so
        # a node that was labelled/cordoned is always put back even if the
        # declare raises after it mutated. ``release`` reads the on-disk record
        # to decide whether anything was in fact declared.
        self.ctx.cleanup.push("warm-spare", self.release)
        try:
            report = warm_spare.perform_warm_spare(
                self._node_api(),
                warm_spare.WarmSpareRequest(
                    state_dir=self.record.parent,
                    node=self.node,
                    fault_node="",
                    cluster_id=self.ctx.settings.regional.cluster_id,
                    reference=f"general-live-030/{self.ctx.run_id}",
                    mode="declare",
                    confirmation=warm_spare.DECLARE_CONFIRMATION,
                ),
                cluster_id=self.ctx.settings.regional.cluster_id,
                record=self.record,
                agent_lookup=self._agent_lookup(),
            )
        except warm_spare.WarmSpareError as exc:
            # The admin refused (a stale reservation, a half-declared state,
            # an unreadable Agent record) before any patch: the condition was
            # never established and nothing about the product was graded.
            self.ctx.evidence.write("spare-declare", {"refused": str(exc)})
            raise VariantInconclusive(f"spare declaration refused: {exc}") from exc
        self._declared = True
        self.ctx.evidence.write("spare-declare", report)
        if report.get("refusals"):
            raise VariantInconclusive(
                f"spare declaration refused: {report['refusals']}"
            )
        return report

    def release(self) -> dict[str, Any]:
        from gpu_fault.admin import warm_spare

        # The on-disk record is the proof of what reached the cluster: the admin
        # declare writes ``declared_state`` only after it has verified the node
        # is labelled and cordoned. Without such an unreleased record -- a
        # declare refused or failed before its patch, or one already released --
        # there is nothing of ours to put back, so cleanup is "nothing declared"
        # and must leave the node exactly as it was found.
        record = warm_spare.read_record(self.record)
        self._declared = False
        if (
            record is None
            or record.get("released_at")
            or "declared_state" not in record
        ):
            receipt: dict[str, Any] = {
                "released": False,
                "restored": None,
                "declared_spares": [],
            }
            self.ctx.evidence.write("spare-release", receipt)
            return receipt
        api = self._node_api()
        report = warm_spare.perform_warm_spare(
            api,
            warm_spare.WarmSpareRequest(
                state_dir=self.record.parent,
                node=self.node,
                fault_node="",
                cluster_id=self.ctx.settings.regional.cluster_id,
                reference=f"general-live-030/{self.ctx.run_id}",
                mode="release",
                confirmation=warm_spare.RELEASE_CONFIRMATION,
            ),
            cluster_id=self.ctx.settings.regional.cluster_id,
            record=self.record,
            agent_lookup=self._agent_lookup(),
        )
        # The receipt proves the release: the restored node state the admin
        # verified, and the declared-spare set the released node is now out of.
        document = report.get("release") or {}
        receipt = {
            "released": True,
            "restored": document.get("restored_state"),
            "declared_spares": warm_spare.declared_spares(api),
            "report": report,
        }
        self.ctx.evidence.write("spare-release", receipt)
        return receipt

    def _node_api(self) -> Any:
        from scripts.e2e.regional.declare_warm_spare import node_api

        return node_api(self.ctx.regional)

    def _agent_lookup(self) -> Any:
        from scripts.e2e.regional.declare_warm_spare import store_agent_lookup

        return store_agent_lookup(self.warm)

    def node_snapshot(self) -> dict[str, Any]:
        # Not ``node``: that name holds the spare node *name* (a string). An
        # accessor sharing it would shadow the string and be uncallable
        # (TypeError: 'str' object is not callable; live 2026-10-24).
        value = self.warm.node_snapshot(
            self.node,
            extra_annotation_keys=(
                HEALTH_ANNOTATION,
                FAILURES_ANNOTATION,
                INCIDENT_ANNOTATION,
            ),
        )
        self.ctx.evidence.write("spare-node", value)
        return value

    def grade(self) -> dict[str, Any]:
        store = self.ctx.regional.cpu_python(
            SPARE_STORE_PROBE, self.ctx.settings.regional.cluster_id, self.node
        )
        self.ctx.evidence.write("spare-grade", store)
        return store

    def stop_kubelet(self, run_id: str = "") -> dict[str, Any]:
        """Stop kubelet through a new service window under ``run_id``.

        ``run_id`` is the window's identity: the context's run id, or -- once
        the pre-step retired an earlier window still open under it -- the next
        free rerun identity (``reclaim_service_windows``).
        """

        import hashlib
        import json

        run_id = run_id or self.ctx.run_id

        from scripts.e2e.regional.warm_spare_fixture import WarmSpareServiceFixture

        # The service window binds an independent on-node restore to an approved
        # plan digest (64 hex) and an absolute, timezone-aware expiry. This stop
        # carries no remediation Plan, so the digest is over the approved action
        # itself; the expiry is the variant's own maintenance deadline (a
        # ``datetime``, not its ISO string -- the window reads ``.timestamp()``).
        plan_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "case": self.ctx.spec.key,
                    "run_id": run_id,
                    "node": self.node,
                    "service": "kubelet.service",
                    "intent": "warm-spare-unhealthy-not-ready",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        service = WarmSpareServiceFixture(
            self.warm,
            node=self.node,
            image=self.ctx.settings.host_probe_image,
            # The on-node probe trusts only the DESTR-008 service-window fixture
            # identity, not the running case; run_id keeps this run's window and
            # journals distinct from a real DESTR-008 run on the same node.
            case_id=SERVICE_WINDOW_FIXTURE_CASE_ID,
            run_id=run_id,
            state_directory=self.ctx.case_dir / SERVICE_STATE,
            node_uid=self.node_snapshot().get("uid", ""),
            plan_sha256=plan_sha256,
            release_id=self.ctx.regional.release_id(),
            maintenance_expires_at=self.ctx.deadline,
        )
        service.create()
        state: dict[str, Any] = {}

        def undo_service() -> dict[str, Any]:
            # The window's own CLOSED report is the proof: kubelet running
            # again, the stop timer/service and the restore unit quiescent, and
            # ``restore_reason`` naming who restored it ("deadline" = the
            # on-node unit at restore_at, "cleanup" = this undo).
            residuals = service.cleanup()
            return {
                "residuals": residuals,
                "window": service.window.data.get("host_cleanup"),
            }

        def restore_kubelet() -> dict[str, Any]:
            # One undo, through the node: the explicit undo is attempted, but
            # its channel is kubelet's own exec path, so with kubelet stopped
            # the missing response is expected and the on-node restore unit is
            # the restore -- awaited via the API until ``expires_at`` plus a
            # margin, then proven over the restored channel. Then every Pod on
            # the node Ready, before the spare is released (the stack runs this
            # before the release it was pushed after).
            try:
                report = restore_through_node(
                    service="kubelet.service",
                    node=self.node,
                    undo=undo_service,
                    node_snapshot=lambda: self.warm.node_snapshot(self.node),
                    binding=service.window.data.get("binding"),
                    restore_unit=str(
                        (state.get("stopped") or {}).get("restore_unit") or ""
                    ),
                    clock=self.ctx.clock,
                    sleep=self.ctx.sleep,
                    ready_timeout_seconds=READY_WAIT_SECONDS,
                )
            except NodeRestoreError as exc:
                self.ctx.evidence.write("spare-kubelet-restore", exc.report)
                raise
            self.ctx.evidence.write("spare-kubelet-restore", report)
            report["pods_not_ready"] = self.wait_pods_ready()
            return report

        self.ctx.cleanup.push("spare-kubelet", restore_kubelet)
        stopped = service.stop("kubelet.service", restore_seconds=600, delay_seconds=15)
        state["stopped"] = stopped
        self.ctx.evidence.write("spare-kubelet", stopped)
        return stopped

    def await_not_ready(self) -> dict[str, Any]:
        """The candidate's snapshot once kubelet's stop has shown (Ready != True).

        Returns the last snapshot either way; the caller's oracle reads
        ``ready`` and records INCONCLUSIVE when the node never left Ready.
        """

        from scripts.e2e.regional.regional_commands import RegionalFixtureError

        try:
            self.warm.wait_node_ready(
                self.node, ready=False, timeout_seconds=NOT_READY_WAIT_SECONDS
            )
        except RegionalFixtureError:
            pass
        return self.node_snapshot()

    def wait_pods_ready(self) -> list[str]:
        """Pods on the spare still not Ready when the bounded wait ends."""

        def accept() -> list[str] | None:
            pending = pods_not_ready(self.ctx.regional, self.node)
            return [] if not pending else None

        value = self.ctx.wait_until(
            accept,
            timeout_seconds=PODS_READY_WAIT_SECONDS,
            poll_seconds=15,
            name="spare-pods-ready",
        )
        return [] if value == [] else pods_not_ready(self.ctx.regional, self.node)

    def pause_metrics(self) -> dict[str, Any]:
        """SIGSTOP the spare's metrics collector under the node probe's pause drop-in.

        The deadman is ``METRICS_PAUSE_SECONDS`` -- the hold, the grade and the
        runner's slack, inside the node probe's pause bound -- not the variant's
        budget (live 2026-10-27 the budget was refused on both clusters before
        any pause). A unit that did not stop makes the variant INCONCLUSIVE, as
        031's pauses do, with the resume still queued and the receipt kept.
        """

        probe = self.ctx.probe("node", node=self.node)
        paused = self.ctx.pause(
            "unit-pause",
            "--unit",
            METRICS_UNIT,
            "--seconds",
            str(METRICS_PAUSE_SECONDS),
            node=self.node,
        )

        def resume_metrics() -> dict[str, Any]:
            resumed = probe.execute(
                "unit-resume", "--unit", METRICS_UNIT, "--run-id", self.ctx.run_id
            )
            # The candidate is another lane's node: hand it back with fresh
            # channels, or that lane's next preflight reads our pause as its
            # stale collector.
            return {"resumed": resumed, "stale_channels": self.wait_channels_fresh()}

        self.ctx.cleanup.push("spare-metrics", resume_metrics)
        self.ctx.evidence.write("spare-metrics-pause", paused)
        if not paused.get("stopped"):
            raise VariantInconclusive(
                f"{METRICS_UNIT} did not stop under SIGSTOP on the spare {self.node}"
            )
        return paused

    def wait_channels_fresh(self) -> list[str]:
        """Channel errors still present on the spare when the bounded wait ends."""

        def read() -> list[str]:
            store = preflight_state(
                self.ctx.regional, self.ctx.settings.regional.cluster_id, self.node
            )
            return channel_errors(
                list(store.get("collector_statuses") or []), utc_now()
            )

        value = self.ctx.wait_until(
            lambda: [] if not read() else None,
            timeout_seconds=CHANNELS_FRESH_WAIT_SECONDS,
            poll_seconds=30,
            name="spare-channels-fresh",
        )
        return [] if value == [] else read()

    def finish(self) -> dict[str, Any]:
        """Undo every injection, release the spare, read the candidate back.

        Runs the variant's own LIFO undo stack now -- the injection's restore
        first, the warm-spare release last -- so the verdict can carry the
        candidate's after-state and the cluster-wide spare census; the runner's
        teardown finds nothing pending.
        """

        cleanup = self.ctx.cleanup.run()
        after = self.node_snapshot()
        residue = spare_residue_report(
            json.loads(self.ctx.regional.kubectl("gpu", "get", "node", "-o", "json")),
            expected=self.spares_before,
        )
        store = preflight_state(
            self.ctx.regional, self.ctx.settings.regional.cluster_id, self.node
        )
        receipt = {
            "spare_before": self.before,
            "spare_after": after,
            "residue": residue,
            "cleanup": cleanup,
            "release": next(
                (
                    item.get("result")
                    for item in reversed(cleanup)
                    if item.get("name") == "warm-spare"
                ),
                None,
            ),
            # The product's own view of the handed-back node: what the next
            # case's preflight on it will read (incidents a NotReady spare
            # drew are the product's, closed by the lane's pre-case clean).
            "open_incidents_after": list(store.get("open_incidents") or []),
            "active_workflows_after": list(store.get("active_workflows") or []),
        }
        self.ctx.evidence.write("spare-finish", receipt)
        return receipt

    def mutate(self, annotations: dict[str, str | None]) -> Any:
        from scripts.e2e.regional.warm_spare_fixture import (
            NodeMutationFixture,
            NodePatch,
        )

        mutation = NodeMutationFixture(
            self.warm,
            self.node,
            annotation_keys=tuple(annotations),
            allow_reclaimed_reservation=True,
        )
        self.ctx.cleanup.push("spare-annotations", mutation.restore)
        mutation.apply(NodePatch(labels={}, annotations=annotations))
        return mutation


# --------------------------------------------------------------------------- #
# the spare candidate: resolution, read-only idleness proof, hand-back proof
# --------------------------------------------------------------------------- #
def spare_candidate(settings: GeneralLiveSettings) -> str:
    """``--spare-candidate``, else the lane's ``--node-b``; never the target node."""

    return settings.spare_candidate or settings.peer_node


def spare_scope(settings: GeneralLiveSettings) -> str:
    """The provider cluster the warm-spare fixture reads: HyperPod's cluster
    name, or the EKS cluster name (= the lane's cluster id) on plain EKS."""

    return settings.hyperpod_cluster or settings.regional.cluster_id


def candidate_errors(
    settings: GeneralLiveSettings,
    regional: Any,
    *,
    recovery: Callable[[], Mapping[str, Any]] | None = None,
) -> list[str]:
    """Why the spare candidate must not be declared now; empty means idle.

    Read-only, the same facts the lane preflight requires of the target node
    plus the spare-specific ones: a node another lane owns is only borrowed
    when nothing but infrastructure runs on it and the product holds nothing
    on it. ``recovery`` reads the provider's cluster recovery fact (EKS node
    auto repair per GPU node group); on plain EKS an enabled repair would race
    the kubelet stop of ``not_ready`` with a provider reboot/replace.
    """

    candidate = spare_candidate(settings)
    if not candidate:
        return [
            "030 needs a spare candidate: --spare-candidate <node> "
            "(default: --node-b, the lane's second node)"
        ]
    if candidate == settings.node:
        return [f"the spare candidate {candidate} is the target node itself"]
    errors: list[str] = []
    labels = regional.node_metadata(candidate).get("labels") or {}
    provider = node_provider(labels)
    target = node_provider(regional.node_metadata(settings.node).get("labels") or {})
    if provider != target:
        errors.append(
            f"{candidate} is a {provider or 'unknown'} node; the target node "
            f"{settings.node} is {target or 'unknown'}"
        )
    if labels.get(SPARE_LABEL) == "true":
        errors.append(f"{candidate} already carries {SPARE_LABEL}=true")
    node = regional.node_snapshot(candidate)
    if node.get("ready") != "True":
        errors.append(f"{candidate} is not Ready")
    if node.get("unschedulable"):
        errors.append(f"{candidate} is cordoned")
    if node.get("ownership_annotations"):
        errors.append(f"{candidate} has gpu-fault ownership annotations")
    if any(
        str(item.get("key", "")).startswith(QUARANTINE_PREFIX)
        for item in node.get("taints") or []
    ):
        errors.append(f"{candidate} carries gpu-fault taints")
    pool = node.get("spare_pool_annotations") or {}
    if pool.get(RESERVATION_ANNOTATION):
        errors.append(f"{candidate} carries a spare reservation")
    if pool.get(POOL_STATE_ANNOTATION) not in (None, "AVAILABLE"):
        errors.append(
            f"{candidate} spare pool state is {pool.get(POOL_STATE_ANNOTATION)}"
        )
    workloads = regional.business_workloads(candidate)
    if workloads:
        names = sorted(f"{w.get('namespace')}/{w.get('name')}" for w in workloads)
        errors.append(f"{candidate} runs a business workload: {names}")
    gpu_pods = [p for p in regional.gpu_workloads() if p.get("node") == candidate]
    if gpu_pods:
        names = sorted(f"{p.get('namespace')}/{p.get('name')}" for p in gpu_pods)
        errors.append(f"{candidate} has GPU-requesting Pods: {names}")
    store = preflight_state(regional, settings.regional.cluster_id, candidate)
    if (store.get("agent") or {}).get("lifecycle_state") != "ACTIVE":
        errors.append(f"{candidate} Node Agent is not ACTIVE")
    if store.get("open_incidents"):
        errors.append(f"{candidate} has open incidents")
    if store.get("active_workflows"):
        errors.append(f"{candidate} has active workflows")
    errors.extend(
        f"{candidate}: {item}"
        for item in channel_errors(
            list(store.get("collector_statuses") or []), utc_now()
        )
    )
    if provider == "eks" and recovery is not None:
        repair = dict(recovery().get("node_repair_config") or {})
        enabled = sorted(name for name, value in repair.items() if value is True)
        if enabled:
            errors.append(
                f"EKS node auto repair is enabled on node group(s) {enabled}; a "
                "NotReady spare would race the product with a provider repair"
            )
    return errors


def pods_not_ready(regional: Any, node: str) -> list[str]:
    """``namespace/name`` of every Pod on ``node`` that is not Ready yet.

    Running and Pending Pods only: a Pod that already Succeeded or Failed is
    done, not unready.
    """

    document = json.loads(
        regional.kubectl(
            "gpu",
            "get",
            "pod",
            "--field-selector",
            f"spec.nodeName={node}",
            "-o",
            "json",
            all_namespaces=True,
        )
    )
    pending: list[str] = []
    for item in document.get("items") or []:
        status = item.get("status") or {}
        if status.get("phase") not in {"Running", "Pending"}:
            continue
        ready = next(
            (
                condition.get("status")
                for condition in status.get("conditions") or []
                if condition.get("type") == "Ready"
            ),
            None,
        )
        if ready != "True":
            metadata = item.get("metadata") or {}
            pending.append(f"{metadata.get('namespace')}/{metadata.get('name')}")
    return sorted(pending)


def restored_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Why the candidate was not handed back as found; empty means clean.

    Judged on the runner's own reads after the release: the spare label gone,
    the cordon back to the pre-declaration state, the spare annotations as
    before, the node Ready, no spare label newly on any node of the cluster,
    and every undo of the stack reported success.
    """

    before = receipt.get("spare_before") or {}
    after = receipt.get("spare_after") or {}
    errors: list[str] = []
    if (after.get("labels") or {}).get(SPARE_LABEL) == "true":
        errors.append("the spare label is still on the candidate")
    if after.get("unschedulable") and not before.get("unschedulable"):
        errors.append("the candidate is still cordoned")
    if after.get("ready") != "True":
        errors.append(
            f"the candidate is not Ready after the release ({after.get('ready')})"
        )
    for key in (RESERVATION_ANNOTATION, RESERVED_AT_ANNOTATION, POOL_STATE_ANNOTATION):
        was = (before.get("annotations") or {}).get(key)
        now = (after.get("annotations") or {}).get(key)
        if was != now:
            errors.append(f"annotation {key} is {now!r}, was {was!r}")
    residue = receipt.get("residue") or {}
    if residue.get("unexpected"):
        errors.append(f"the spare label is left on {residue['unexpected']}")
    for item in receipt.get("cleanup") or []:
        if not item.get("ok"):
            errors.append(f"undo {item.get('name')} failed: {item.get('error')}")
    return errors


def finalize(result: dict[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the hand-back proof into the verdict: a spare not handed back as
    found fails the variant whatever the grading said."""

    errors = restored_errors(receipt)
    result.setdefault("stages", {})["restored"] = errors
    if errors:
        result["verdict"] = "FAIL"
        result.setdefault("errors", []).extend(f"restored: {item}" for item in errors)
    result.update(
        {
            "spare_candidate": (receipt.get("spare_after") or {}).get("name"),
            "spare_before": receipt.get("spare_before"),
            "spare_after": receipt.get("spare_after"),
            "spare_residue": receipt.get("residue"),
            "spare_release": receipt.get("release"),
            "open_incidents_after": receipt.get("open_incidents_after"),
            "active_workflows_after": receipt.get("active_workflows_after"),
        }
    )
    return result


# --------------------------------------------------------------------------- #
# grading oracle (mirrors HyperPodSpareCoordinator.health_reasons, read-only)
# --------------------------------------------------------------------------- #
def grading_reasons(
    node: dict[str, Any], store: dict[str, Any], *, our_incident: str | None
) -> list[str]:
    """The product's own reject reasons for this spare, node + store combined."""

    reasons: list[str] = []
    labels = node.get("labels") or {}
    annotations = node.get("annotations") or {}
    if node.get("ready") != "True":
        reasons.append("Kubernetes node is not Ready")
    health = labels.get(HYPERPOD_HEALTH_LABEL)
    if health != "Schedulable":
        reasons.append(f"HyperPod node health is {health or 'unknown'}")
    reservation = annotations.get(RESERVATION_ANNOTATION)
    if reservation and reservation != our_incident:
        reasons.append(f"reserved by incident {reservation}")
    pool_state = annotations.get(POOL_STATE_ANNOTATION)
    allowed = {None, "AVAILABLE"}
    if reservation and reservation == our_incident:
        allowed.add("ALLOCATED")
    if pool_state not in allowed:
        reasons.append(f"spare pool state is {pool_state}")
    if not reservation and not node.get("unschedulable"):
        reasons.append("unreserved spare is schedulable")
    reasons.extend(store.get("agent_reasons") or [])
    reasons.extend(store.get("fault_reasons") or [])
    reasons.extend(store.get("observation_reasons") or [])
    return reasons


def reason_class(reason: str) -> str:
    """``spare_health.SpareHealthReasonClass`` of one reason string."""

    normalized = reason.strip().lower()
    if normalized.startswith(
        (
            "unreserved spare is schedulable",
            "reserved by incident ",
            "spare pool state is ",
            "active gpu resource pods exist",
            "active gpu clients",
        )
    ):
        return CONFIGURATION
    if normalized.startswith(HARDWARE_PREFIX):
        return HARDWARE
    return OBSERVATION


def no_recovery_errors(node: dict[str, Any], store: dict[str, Any]) -> list[str]:
    """A block-only condition must not have produced any provider recovery.

    The pool-state annotation is not read here: the runner itself injects it in
    the configuration/reservation variants. A provider recovery shows as a
    spare-health incident/workflow (the controller's reboot), the spare-health
    failure/incident annotations it stamps, or a reboot operation in a workflow.
    """

    errors: list[str] = []
    annotations = node.get("annotations") or {}
    if annotations.get(INCIDENT_ANNOTATION):
        errors.append("the spare carries a spare-health reboot incident annotation")
    failures = annotations.get(FAILURES_ANNOTATION)
    if failures not in (None, "0"):
        errors.append(f"the spare recorded {failures} spare-health failures")
    for incident in store.get("spare_health_incidents") or []:
        workflow = (store.get("spare_health_workflows") or {}).get(
            str(incident.get("workflow_request_id"))
        ) or {}
        reboots = {"RESTART_NODE", "REPLACE_NODE", "RESET_GPU"} & set(
            operations(workflow)
        )
        if reboots:
            errors.append(
                f"spare-health incident {incident.get('incident_id')} planned {sorted(reboots)}"
            )
    return errors


def block_only_verdict(
    node: dict[str, Any],
    store: dict[str, Any],
    *,
    expected_class: str,
    expected_fragment: str,
    our_incident: str | None,
    oracle_ok: bool,
    oracle_detail: str,
) -> dict[str, Any]:
    reasons = grading_reasons(node, store, our_incident=our_incident)
    matched = [item for item in reasons if expected_fragment.lower() in item.lower()]
    stages = {
        "graded_ineligible": []
        if matched
        else [f"no reject reason contains {expected_fragment!r}; reasons={reasons}"],
        "reason_class": []
        if all(reason_class(item) == expected_class for item in matched)
        else [f"a matched reason is not {expected_class}: {matched}"],
        "no_recovery": no_recovery_errors(node, store),
    }
    return _result(
        stages,
        inconclusive=None if oracle_ok else [oracle_detail],
        extra={"reject_reasons": reasons},
    )


def grading_window_errors(
    node: Mapping[str, Any],
    after_grade: Mapping[str, Any],
    graded_at: datetime,
    stopped: Mapping[str, Any],
) -> list[str]:
    """Why "graded while kubelet was down" is not proven; empty means it is.

    ``node`` is the snapshot the NotReady wait ended on, ``after_grade`` the
    one read right after the product's grade, ``graded_at`` the clock then and
    ``stopped`` the window's SCHEDULED report (``restore_at`` is when the
    on-node unit restores kubelet on its own). The claim needs the node not
    Ready on both sides of the grade and the grade before that deadline.
    """

    errors: list[str] = []
    if node.get("ready") in ("True", None):
        errors.append(f"the spare never left Ready: {node.get('ready')}")
    if after_grade.get("ready") == "True":
        errors.append(
            "the spare was Ready again when the grade had been read: kubelet was "
            "already restored"
        )
    deadlines = window_deadlines(stopped)
    if deadlines is not None and graded_at >= deadlines["restore_at"]:
        errors.append(
            f"the grade was read at {graded_at.isoformat()}, not before the on-node "
            f"restore deadline {deadlines['restore_at'].isoformat()}"
        )
    return errors


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #
def reclaim_service_windows(ctx: VariantContext) -> str:
    """Retire an earlier run's provably stale window, or BLOCK naming why.

    The case directory's ``spare-service`` keeps every service window's
    controller journal; one not CLOSED on the candidate is an earlier run's
    custody its undo never closed (live 2026-10-27, plain EKS: the undo before
    ``restore_through_node`` existed). While it is open the node keeps that
    window's claim, drop-in and units, a new window is refused the claim, and a
    rerun under the same run id -- an open record keeps its run id
    (``rerun_run_id``) -- meets the journal itself: live 2026-10-28 the rerun
    declared the spare, then refused "service controller identity or state
    changed". Before anything is declared, each such journal is retired through
    the harness's own host probe (``service_window_retire``: only on proof,
    kubelet untouched); what cannot be retired keeps the variant
    BLOCKED(FIXTURE_NOT_READY) with every failed proof named. Returns the run id
    the new window uses: the context's, or after a retirement the next free
    rerun identity (the retired journal is create-once under the old one).
    """

    from scripts.e2e.regional.general_live_fixture import rerun_run_id, run_identity
    from scripts.e2e.regional.service_window_retire import (
        RetireSettings,
        retire_stale_windows,
    )

    node = spare_candidate(ctx.settings)
    directory = ctx.case_dir / SERVICE_STATE
    if not node or not stale_journals(directory, node):
        return ctx.run_id
    from gpu_fault.admin.operator_identity import (
        local_operator_identity,
        resolve_operator_identity,
    )

    regional = ctx.regional.settings
    reports = retire_stale_windows(
        directory,
        node,
        settings=RetireSettings(
            kubeconfig=regional.gpu_kubeconfig,
            context=regional.gpu_context,
            namespace=regional.namespace,
            image=ctx.settings.host_probe_image,
            node_uid=lambda name: _node_uid(ctx, name),
        ),
        by=resolve_operator_identity(fallback=local_operator_identity()),
        via=f"{ctx.spec.key} pre-step, run {ctx.run_id}",
    )
    ctx.evidence.write("spare-service-retire", {"node": node, "journals": reports})
    held = open_windows(directory, node)
    if held:
        reasons = [
            f"{report['journal']}: {reason}"
            for report in reports
            for reason in report["refusals"]
        ]
        raise VariantBlocked(
            "FIXTURE_NOT_READY",
            f"an earlier run's service window on {node} is still open in "
            f"{directory} and was not retired: "
            + " | ".join(held)
            + ". Retirement refused: "
            + ("; ".join(reasons) or "no reason given")
            + ". Nothing was declared or stopped",
        )
    base = run_identity(
        ctx.spec.case_id, ctx.spec.variant, ctx.attempt, ctx.case_dir.parent.parent
    )
    return rerun_run_id(ctx.case_dir, base)


def _node_uid(ctx: VariantContext, node: str) -> str:
    """The Node's current UID through the API; ``""`` when it is gone."""

    from scripts.e2e.regional.regional_commands import RegionalFixtureError

    try:
        return str(ctx.regional.node_snapshot(node).get("uid") or "")
    except RegionalFixtureError:
        return ""


def not_ready(ctx: VariantContext) -> dict[str, Any]:
    window_run_id = reclaim_service_windows(ctx)
    session = _declared(ctx)
    stopped = session.stop_kubelet(window_run_id)
    node = session.await_not_ready()
    store = session.grade()
    # The claim is "the product graded the spare while kubelet was down": the
    # node is read again after the grade and the grade must precede the on-node
    # restore deadline; the undo then awaits the node's return (through the
    # node, not the stopped kubelet) before the release and ``restored``.
    graded_at = ctx.clock()
    window = grading_window_errors(node, session.node_snapshot(), graded_at, stopped)
    deadlines = window_deadlines(stopped) or {}
    ctx.evidence.write(
        "spare-grading-window",
        {
            "graded_at": graded_at.isoformat(),
            "errors": window,
            **{key: value.isoformat() for key, value in deadlines.items()},
        },
    )
    result = block_only_verdict(
        node,
        store,
        expected_class=OBSERVATION,
        expected_fragment="Kubernetes node is not Ready",
        our_incident=session.incident_id,
        oracle_ok=not window,
        oracle_detail="; ".join(window),
    )
    result["service_window_run_id"] = window_run_id
    return finalize(result, session.finish())


def stale_metrics(ctx: VariantContext) -> dict[str, Any]:
    session = _declared(ctx)
    session.pause_metrics()
    ctx.hold(STALE_HOLD_SECONDS, name="stale-hold")
    store = session.grade()
    observation = store.get("observation_reasons") or []
    result = block_only_verdict(
        session.node_snapshot(),
        store,
        expected_class=OBSERVATION,
        expected_fragment="collector observation UNKNOWN",
        our_incident=session.incident_id,
        oracle_ok=bool(observation),
        oracle_detail="GPU_INVENTORY/GPU_METRICS never went stale (UNKNOWN)",
    )
    return finalize(result, session.finish())


def configuration_error(ctx: VariantContext) -> dict[str, Any]:
    session = _declared(ctx)
    session.mutate({POOL_STATE_ANNOTATION: "UNAVAILABLE"})
    node = session.node_snapshot()
    oracle_ok = (node.get("annotations") or {}).get(
        POOL_STATE_ANNOTATION
    ) == "UNAVAILABLE"
    result = block_only_verdict(
        node,
        session.grade(),
        expected_class=CONFIGURATION,
        expected_fragment="spare pool state is",
        our_incident=session.incident_id,
        oracle_ok=oracle_ok,
        oracle_detail="the injected pool-state annotation was not observed",
    )
    return finalize(result, session.finish())


def reserved_spare(ctx: VariantContext) -> dict[str, Any]:
    session = _declared(ctx)
    from datetime import datetime, timezone

    foreign = f"inc-other-{ctx.run_id}"
    session.mutate(
        {
            RESERVATION_ANNOTATION: foreign,
            RESERVED_AT_ANNOTATION: datetime.now(timezone.utc).isoformat(),
            POOL_STATE_ANNOTATION: "ALLOCATED",
        }
    )
    node = session.node_snapshot()
    store = session.grade()
    oracle_ok = (node.get("annotations") or {}).get(RESERVATION_ANNOTATION) == foreign
    # Grade for a DIFFERENT incident than the reservation holder: the reserved
    # spare must not be reusable by another task.
    result = block_only_verdict(
        node,
        store,
        expected_class=CONFIGURATION,
        expected_fragment="reserved by incident",
        our_incident=f"inc-requester-{ctx.run_id}",
        oracle_ok=oracle_ok,
        oracle_detail="the injected reservation annotation was not observed",
    )
    result.setdefault("notes", []).append(
        "a spare reserved by one incident is rejected for any other incident"
    )
    return finalize(result, session.finish())


def _declared(ctx: VariantContext) -> SpareSession:
    session = SpareSession(ctx)
    session.verify_idle()
    session.declare()
    return session


def spare_preflight(settings: GeneralLiveSettings, regional: Any) -> list[str]:
    """The plan-time proof that the spare candidate is idle (read-only)."""

    from scripts.e2e.regional.warm_spare_fixture import WarmSpareLiveFixture

    def recovery() -> dict[str, Any]:
        return WarmSpareLiveFixture(regional, spare_scope(settings)).cluster_recovery()

    return candidate_errors(settings, regional, recovery=recovery)


def _result(
    stages: dict[str, list[str]],
    *,
    notes: list[str] | None = None,
    inconclusive: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # An INCONCLUSIVE carries its reason: ``detail`` is the key the runner uses
    # for its own verdict reasons, and the errors list is what the board shows.
    return {
        "verdict": verdict(stages, inconclusive=inconclusive),
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ]
        + [f"inconclusive: {item}" for item in inconclusive or []],
        "notes": notes or [],
        "inconclusive": list(inconclusive or []),
        **({"detail": "; ".join(inconclusive)} if inconclusive else {}),
        **(extra or {}),
    }


def _spec(
    variant: str,
    injection: str,
    expected: str,
    execute,
    *,
    risk: str = "M",
    max_minutes: int = 30,
    approvals: tuple[str, ...] = ("DECLARE_WARM_SPARE",),
) -> VariantSpec:
    return VariantSpec(
        case_id=CASE,
        variant=variant,
        proof_mode=NATIVE,
        lanes=("eks", "hp"),
        risk=risk,
        # Every lock: the spare candidate is another lane's own node on plain
        # EKS, so no other case may be in flight while it is cordoned.
        flags="X",
        injection=injection,
        expected=expected,
        cleanup=(
            "restore kubelet through the node (explicit undo attempted; with kubelet "
            "down the on-node restore unit restores it and the node is awaited Ready "
            "via the API until expires_at + margin, then the window is closed over the "
            "restored channel; node and its Pods Ready again) / resume the metrics "
            "collector (channels fresh again) / restore the annotations, then release "
            "the declared spare (uncordon, label gone) before the verdict; the result "
            "records the candidate before and after and the cluster-wide spare census"
        ),
        max_minutes=max_minutes,
        execute=execute,
        approvals=approvals,
        preflight=spare_preflight,
    )


VARIANTS = [
    _spec(
        "not_ready",
        "declare the spare candidate, then stop kubelet on it (DESTR-008 service"
        " window, 600 s on-node restore timer) until the node leaves Ready",
        "graded ineligible (OBSERVATION 'Kubernetes node is not Ready') while kubelet"
        " was down and before the on-node restore deadline; no reboot, no REMEDIATING;"
        " kubelet restored (awaited through the node, not read through the stopped"
        " kubelet) and the node and its Pods Ready again before the release",
        not_ready,
        approvals=("DECLARE_WARM_SPARE", "SPARE_KUBELET_STOP"),
    ),
    _spec(
        "stale_metrics",
        "declare a spare, pause its metrics collector so GPU telemetry goes stale",
        "graded ineligible (OBSERVATION collector UNKNOWN); no reboot",
        stale_metrics,
    ),
    _spec(
        "configuration_error",
        "declare a spare, inject a non-AVAILABLE spare pool state",
        "graded ineligible (CONFIGURATION 'spare pool state is ...'); no reboot",
        configuration_error,
        risk="L",
    ),
    _spec(
        "reserved_spare",
        "declare a spare, inject a reservation held by another incident",
        "rejected for any other requester (CONFIGURATION 'reserved by incident'); not reusable",
        reserved_spare,
        risk="L",
    ),
    deferred_variant(
        CASE,
        "confirmed_hardware",
        proof_mode=LIVE_REPLAY,
        reason="CAPABILITY_ABSENT: turning a trusted hardware finding on an idle spare into a provider reboot is the HyperPodSpareHealthController, disabled in this site's regional control plane (GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER=false); provider reboot is also unauthorized",
        expected="only authorized fresh trusted hardware evidence drives a provider reboot of the spare",
    ),
    deferred_variant(
        CASE,
        "no_spare",
        proof_mode=NATIVE,
        reason=(
            "COVERED_BY_REGIONAL: GF-REGIONAL-DESTR-008 S1 (hp PASS 2026-10-22) -- every "
            "spare label removed, REPLACE_NODE refused with the exact no-fallback literal, "
            "no BatchReplaceClusterNodes in CloudTrail: the assertions this variant would "
            "make on the REPLACE_NODE failover branch already passed live on this site; "
            "no fixture of its own (WARM_SPARE_FAILOVER territory when re-enabled)"
        ),
        expected="no available spare fails the step with the no-fallback literal; provider replacement stays disabled",
    ),
    deferred_variant(
        CASE,
        "repair_failure",
        proof_mode=LIVE_REPLAY,
        reason="CAPABILITY_ABSENT: the UNAVAILABLE-after-failed-reboot path is the spare-health controller, disabled on this regional site",
        expected="a spare still unhealthy after reboot becomes UNAVAILABLE with one alert; no recovery storm",
    ),
    deferred_variant(
        CASE,
        "bounded_recheck",
        proof_mode=LIVE_REPLAY,
        reason="CAPABILITY_ABSENT: the bounded UNAVAILABLE recheck/re-alert loop is the spare-health controller, disabled on this regional site",
        expected="UNAVAILABLE rechecks on a fixed timer and returns to AVAILABLE only after health validation",
    ),
]
