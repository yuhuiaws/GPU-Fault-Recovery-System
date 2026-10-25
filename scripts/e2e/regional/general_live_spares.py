"""GF-GENERAL-LIVE-030 warm spare unhealthy.

This runs on the HyperPod lane. One otherwise-idle node is declared a warm
spare through the product's admin CLI (``gpu-fault-admin config spare``,
confirmation ``DECLARE_WARM_SPARE_CORDON``); the runner always releases it in
cleanup (``RELEASE_WARM_SPARE_UNCORDON``), so no spare is left declared,
reserved or isolated. Each block-only variant makes the declared spare
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
plane sets ``GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER=false``, so the executor
only grades a spare during an allocation and never autonomously reboots an idle
spare. The hardware-reboot, repair-failure and bounded-recheck variants
therefore stay NOT_RUN(CAPABILITY_ABSENT); ``no_spare`` (the no-provider-replace
fallback) lives on the dangerous REPLACE_NODE failover branch, whose assertions
GF-REGIONAL-DESTR-008 S1 already made live on this site, and stays
NOT_RUN(COVERED_BY_REGIONAL). Every required variant remains in the denominator.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
)
from scripts.e2e.regional.general_live_registry import (
    LIVE_REPLAY,
    NATIVE,
    VariantSpec,
    deferred_variant,
)
from scripts.e2e.regional.general_live_verdicts import operations, verdict

CASE = "GF-GENERAL-LIVE-030"
#: The on-node service-window helper (``warm_spare_node_probe``) honours a
#: kubelet/agent stop with an independent restart timer only for one fixture
#: identity -- the DESTR-008 service-window fixture. ``SPARE_KUBELET_STOP``
#: authorises 030 to reuse exactly that mechanism ("stop kubelet ... with a
#: host-side automatic restart timer"), so ``not_ready`` opens the window under
#: this fixture identity, not under its own running-case key (which the probe
#: rejects as "service window is not a DESTR008 fixture"). The acceptance
#: evidence still records the running case; only the on-node fixture is shared.
SERVICE_WINDOW_FIXTURE_CASE_ID = "GF-REGIONAL-DESTR-008"
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

        if not ctx.settings.hyperpod_cluster:
            raise VariantNotRun(
                "FIXTURE_NOT_READY", "a warm spare needs --hyperpod-cluster"
            )
        self.ctx = ctx
        self.node = ctx.settings.node
        self.incident_id = f"inc-{ctx.run_id}"
        self.record = Path(ctx.case_dir) / "warm-spares" / f"{self.node}.json"
        self.warm = WarmSpareLiveFixture(ctx.regional, ctx.settings.hyperpod_cluster)
        self._declared = False

    def declare(self) -> dict[str, Any]:
        from gpu_fault.admin import warm_spare

        # Register the release before the declaration can touch the cluster, so
        # a node that was labelled/cordoned is always put back even if the
        # declare raises after it mutated. ``release`` reads the on-disk record
        # to decide whether anything was in fact declared.
        self.ctx.cleanup.push("warm-spare", self.release)
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

    def stop_kubelet(self) -> dict[str, Any]:
        import hashlib
        import json

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
                    "run_id": self.ctx.run_id,
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
            run_id=self.ctx.run_id,
            state_directory=self.ctx.case_dir / "spare-service",
            node_uid=self.node_snapshot().get("uid", ""),
            plan_sha256=plan_sha256,
            release_id=self.ctx.regional.release_id(),
            maintenance_expires_at=self.ctx.deadline,
        )
        service.create()
        self.ctx.cleanup.push("spare-kubelet", service.cleanup)
        stopped = service.stop("kubelet.service", restore_seconds=600, delay_seconds=15)
        self.ctx.evidence.write("spare-kubelet", stopped)
        return stopped

    def pause_metrics(self) -> dict[str, Any]:
        probe = self.ctx.probe("node", node=self.node)
        paused = self.ctx.pause(
            "unit-pause",
            "--unit",
            "gpu-fault-metrics-collector.service",
            "--seconds",
            str(self.ctx.spec.max_minutes * 60),
            node=self.node,
        )
        self.ctx.cleanup.push(
            "spare-metrics",
            lambda: probe.execute(
                "unit-resume",
                "--unit",
                "gpu-fault-metrics-collector.service",
                "--run-id",
                self.ctx.run_id,
            ),
        )
        self.ctx.evidence.write("spare-metrics-pause", paused)
        return paused

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


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #
def not_ready(ctx: VariantContext) -> dict[str, Any]:
    session = _declared(ctx)
    stopped = session.stop_kubelet()
    node = session.node_snapshot()
    oracle_ok = node.get("ready") not in ("True", None)
    return block_only_verdict(
        node,
        session.grade(),
        expected_class=OBSERVATION,
        expected_fragment="Kubernetes node is not Ready",
        our_incident=session.incident_id,
        oracle_ok=oracle_ok,
        oracle_detail=f"the spare never left Ready: {node.get('ready')} ({stopped})",
    )


def stale_metrics(ctx: VariantContext) -> dict[str, Any]:
    session = _declared(ctx)
    session.pause_metrics()
    ctx.hold(460, name="stale-hold")
    store = session.grade()
    observation = store.get("observation_reasons") or []
    return block_only_verdict(
        session.node_snapshot(),
        store,
        expected_class=OBSERVATION,
        expected_fragment="collector observation UNKNOWN",
        our_incident=session.incident_id,
        oracle_ok=bool(observation),
        oracle_detail="GPU_INVENTORY/GPU_METRICS never went stale (UNKNOWN)",
    )


def configuration_error(ctx: VariantContext) -> dict[str, Any]:
    session = _declared(ctx)
    session.mutate({POOL_STATE_ANNOTATION: "UNAVAILABLE"})
    node = session.node_snapshot()
    oracle_ok = (node.get("annotations") or {}).get(
        POOL_STATE_ANNOTATION
    ) == "UNAVAILABLE"
    return block_only_verdict(
        node,
        session.grade(),
        expected_class=CONFIGURATION,
        expected_fragment="spare pool state is",
        our_incident=session.incident_id,
        oracle_ok=oracle_ok,
        oracle_detail="the injected pool-state annotation was not observed",
    )


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
    return result


def _declared(ctx: VariantContext) -> SpareSession:
    session = SpareSession(ctx)
    session.declare()
    return session


def _result(
    stages: dict[str, list[str]],
    *,
    notes: list[str] | None = None,
    inconclusive: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "verdict": verdict(stages, inconclusive=inconclusive),
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "notes": notes or [],
        **(extra or {}),
    }


def _spec(
    variant: str,
    injection: str,
    expected: str,
    execute,
    *,
    risk: str = "M",
    max_minutes: int = 25,
    approvals: tuple[str, ...] = ("DECLARE_WARM_SPARE",),
) -> VariantSpec:
    return VariantSpec(
        case_id=CASE,
        variant=variant,
        proof_mode=NATIVE,
        lanes=("hp",),
        risk=risk,
        flags="S",
        injection=injection,
        expected=expected,
        cleanup="release the declared spare (uncordon); restore kubelet/metrics/annotations",
        max_minutes=max_minutes,
        execute=execute,
        approvals=approvals,
    )


VARIANTS = [
    _spec(
        "not_ready",
        "declare a spare, then stop kubelet so it goes NotReady",
        "graded ineligible (OBSERVATION 'Kubernetes node is not Ready'); no reboot, no REMEDIATING",
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
        lanes=("hp",),
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
        lanes=("hp",),
        expected="no available spare fails the step with the no-fallback literal; provider replacement stays disabled",
    ),
    deferred_variant(
        CASE,
        "repair_failure",
        proof_mode=LIVE_REPLAY,
        reason="CAPABILITY_ABSENT: the UNAVAILABLE-after-failed-reboot path is the spare-health controller, disabled on this regional site",
        lanes=("hp",),
        expected="a spare still unhealthy after reboot becomes UNAVAILABLE with one alert; no recovery storm",
    ),
    deferred_variant(
        CASE,
        "bounded_recheck",
        proof_mode=LIVE_REPLAY,
        reason="CAPABILITY_ABSENT: the bounded UNAVAILABLE recheck/re-alert loop is the spare-health controller, disabled on this regional site",
        lanes=("hp",),
        expected="UNAVAILABLE rechecks on a fixed timer and returns to AVAILABLE only after health validation",
    ),
]
