"""Node-health incident labels: one set for every GF-GENERAL-LIVE matcher (S57).

The product mints a node-health finding's incident ``NODE_HEALTH`` and
re-labels that record when it routes more evidence into it instead of opening
another: ``NODE_HEALTH_GROUP`` when a critical quarantine joins the node's
in-flight isolation, ``NODE_RESOURCE_GROUP`` when a device-resource finding
joins it, ``GPU_FAULT_GROUP`` for a finding of a managed attempt; a reboot or
replacement finding of a running attempt opens ``NODE_HEALTH_GROUP`` itself.
008 and 016 (b211) learnt it live; the 017 journal stages, 002 ``cgroup_only``'s
log grading and the DCGM and inventory selectors still took ``NODE_HEALTH``
alone.

Nothing is scripted between the product and the grade here. Kernel lines and
DCGM scrapes go through the control-plane API over an in-memory store, as the
node log and GPU collectors post them; host metric findings, which the host
collector's inventory and the control plane's allocatable comparison raise,
go through ``app/ingest/node_health.py`` (``NodeHealthIngestionService``)
under the product's own rule table (``NodeHealthPolicy.METRIC_RULES``) and
event-id shape. Ingestion writes each finding's marker and re-points it at the
incident it routed the finding to, and the real ``ACTIVITY_PROBE`` reads the
store back. The pin drives every routing the product has and holds the labels
they give to ``general_live_verdicts.NODE_HEALTH_EVENT_TYPES`` exactly; the
matcher tests grade a record two findings share, which each matcher missed
while it took ``NODE_HEALTH`` alone.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext, default_simulated_profile
from gpu_fault.app.ingest.node_health import NodeHealthIngestionService
from gpu_fault.channel_registry import GPU_METRICS_PATH, NODE_LOG_PATH
from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricSample
from gpu_fault.host_health import (
    NodeHealthFinding,
    NodeHealthPolicy,
    NodeLogBatch,
    NodeLogEntry,
)
from gpu_fault.models import WorkloadState
from gpu_fault.orchestration.collector_outbox_maintenance import OPERATOR_EVENT_TYPE
from gpu_fault.orchestration.placement_hold import EVENT_TYPE as PLACEMENT_HOLD
from gpu_fault.policy.models import FaultEventType
from gpu_fault.shared_storage_failure_domain import (
    SHARED_STORAGE_FAILURE_DOMAIN_EVENT_TYPE,
)
from gpu_fault.store import InMemoryStore
from scripts.e2e.regional import general_live_applogs as applogs
from scripts.e2e.regional import general_live_memory as memory
from scripts.e2e.regional.general_live_gpu_verdicts import (
    GpuChain,
    chain_errors,
    completeness_incidents,
    composite_incidents,
    fabric_incidents,
    inventory_incidents,
    metric_incidents,
)
from scripts.e2e.regional.general_live_kernel import SEVERE_MESSAGES
from scripts.e2e.regional.general_live_verdicts import NODE_HEALTH_EVENT_TYPES
from tests._builders import asgi_client, attempt_observation, container_observation
from tests.regional._general_live_fakes import (
    FakeProbe,
    FakeRegional,
    activity_read,
    make_context,
    make_settings,
    store_script,
)
from tests.regional.test_general_live_memory import cgroup_context, quiet_activity

CLUSTER = "gpu-eks-cluster"
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
PEER = "ip-10-0-43-6.us-west-2.compute.internal"
GPU = "GPU-5b1e0c7a-3f2d-4e1c-9a8b-7c6d5e4f3a21"
MCE = SEVERE_MESSAGES["uncorrectable_mce"]
LOCKUP = SEVERE_MESSAGES["kernel_lockup"]
DBE = "ecc_dbe_volatile_total"
INVENTORY = "gpu_inventory_mismatch"
ALLOCATABLE = "gpu_kubernetes_allocatable_mismatch"


class _Product:
    """The product's node-health ingestion over an in-memory store."""

    def __init__(self) -> None:
        self.store = InMemoryStore()
        self.store.save_profile(default_simulated_profile())
        self.context = ApplicationContext(store=self.store)
        self.since = datetime.now(timezone.utc) - timedelta(seconds=1)
        self.batches = 0

    def _batch(self) -> str:
        self.batches += 1
        return f"s57-batch-{self.batches}"

    def _post(self, path: str, batch: Any) -> dict[str, Any]:
        async def post() -> dict[str, Any]:
            async with asgi_client(self.context) as client:
                response = await client.post(path, json=batch.model_dump(mode="json"))
                assert response.status_code == 200, response.text
                return dict(response.json())

        return asyncio.run(post())

    def kernel_lines(self, *messages: str, node: str = NODE) -> list[str]:
        """One NODE_LOGS batch, as the node log collector posts it."""

        now = datetime.now(timezone.utc)
        batch = NodeLogBatch(
            batch_id=self._batch(),
            cluster_id=CLUSTER,
            node_id=node,
            collected_at=now,
            runtime_profile_version="simulated-v1",
            entries=[
                NodeLogEntry(
                    entry_id=f"gfrs-s57-{self.batches}-{index}",
                    source="dmesg",
                    observed_at=now,
                    message=message,
                    priority=2,
                    fields={"_TRANSPORT": "kernel"},
                )
                for index, message in enumerate(messages)
            ],
        )
        findings = self._post(NODE_LOG_PATH, batch)["findings"]
        return [str(item["event_id"]) for item in findings]

    def dbe(self, node: str = NODE) -> str:
        """Two DCGM scrapes, the second with one volatile DBE: the finding's event id."""

        now = datetime.now(timezone.utc)
        new: list[dict[str, Any]] = []
        for index, value in enumerate((0.0, 1.0)):
            batch = GpuMetricBatch(
                batch_id=self._batch(),
                cluster_id=CLUSTER,
                node_id=node,
                observed_at=now + timedelta(seconds=15 * index),
                source="DCGM_EXPORTER",
                runtime_profile_version="simulated-v1",
                product="NVIDIA H200",
                samples=[
                    GpuMetricSample(
                        metric_name="DCGM_FI_DEV_ECC_DBE_VOL_TOTAL",
                        canonical_name=DBE,
                        value=value,
                        gpu_uuid=GPU,
                        gpu_index="0",
                    )
                ],
            )
            new = self._post(GPU_METRICS_PATH, batch)["new_findings"]
        [finding] = new
        return f"gpu-{finding['finding_id']}"

    def host(self, metric: str, *, node: str = NODE, **values: Any) -> str:
        """A host metric finding under the product's rule and event-id shape."""

        batch_id = self._batch()
        _threshold, category, severity, action, reason = NodeHealthPolicy.METRIC_RULES[
            metric
        ]
        event_id = f"{batch_id}-{metric}-node"
        finding = NodeHealthFinding(
            finding_id=f"finding-{event_id}",
            event_id=event_id,
            cluster_id=CLUSTER,
            node_id=node,
            observed_at=datetime.now(timezone.utc),
            category=category,
            severity=severity,
            reason=reason,
            recommended_action=action,
            metric_name=metric,
            value=1.0,
            runtime_profile_version="simulated-v1",
            **{"workload_state": WorkloadState.IDLE, **values},
        )
        NodeHealthIngestionService(self.context).ingest(batch_id, [finding])
        return event_id

    def attempt(self, node: str) -> dict[str, Any]:
        """A running managed attempt on ``node``: what the finding names."""

        job = f"job-{node.split('.')[0]}"
        workload = f"training/pytorchjob/{job}"
        self.store.save_attempt_observation(
            attempt_observation(
                job,
                f"{job}-a001",
                datetime.now(timezone.utc),
                cluster_id=CLUSTER,
                containers=[
                    container_observation(
                        f"pod-{job}", f"{job}-worker-0", 0, node, gpu_uuids=[GPU]
                    )
                ],
                workload_ids=[workload],
            )
        )
        return {
            "workload_state": WorkloadState.ACTIVE,
            "affected_workload_ids": [workload],
        }

    def activity(self, node: str = NODE) -> dict[str, Any]:
        """The real ``ACTIVITY_PROBE`` over the store, as ``node_activity`` asks."""

        return activity_read(
            self.store, [CLUSTER, node, self.since.isoformat(), "", "", "NODE_LOGS"]
        )

    def routed(self, event_id: str, node: str = NODE) -> dict[str, Any]:
        """The incident a finding's marker names: where ingestion routed it."""

        document = self.activity(node)
        [marker] = [
            item
            for item in document["markers"]
            if item["marker_id"] == f"marker-{event_id}"
        ]
        [incident] = [
            item
            for item in document["incidents"]
            if item["incident_id"] == marker["incident_id"]
        ]
        return incident


def _shared(document: dict[str, Any]) -> dict[str, Any]:
    """The one incident of ``document``, and that every marker names it."""

    [incident] = document["incidents"]
    markers = [item["incident_id"] for item in document["markers"]]
    assert markers == [incident["incident_id"]] * 2, (
        f"two findings, one incident: {markers}"
    )
    return incident


# --------------------------------------------------------------------------- #
# the pin: the set is what the product's routes give
# --------------------------------------------------------------------------- #
def test_the_set_is_every_label_the_products_routes_give_a_node_health_record() -> None:
    product = _Product()
    labels: dict[str, str] = {}
    # families/health.py: a critical node-log quarantine on a quiet node.
    [first] = product.kernel_lines(MCE)
    labels["minted"] = product.routed(first)["event_type"]
    # families/drain.py ingest_terminal_node_quarantine: one more while the
    # isolation is in flight joins it (016 cursor_pagination's live burst).
    [second] = product.kernel_lines(LOCKUP)
    joined = product.routed(second)
    assert joined["incident_id"] == product.routed(first)["incident_id"], joined
    labels["critical_quarantine_joins"] = joined["event_type"]
    # families/drain.py ingest_grouped_node_resource_finding: the allocatable
    # count of a node whose GPU went missing joins its reboot chain.
    missing = product.host(INVENTORY, node=PEER)
    resource = product.host(ALLOCATABLE, node=PEER)
    assert product.routed(resource, PEER) == product.routed(missing, PEER)
    labels["device_resource_joins"] = product.routed(resource, PEER)["event_type"]
    # families/grouped_health.py: a DCGM finding of a running managed attempt
    # (the API's workload enrichment names the attempt holding the GPU).
    attempt = "ip-10-0-43-7.us-west-2.compute.internal"
    product.attempt(attempt)
    labels["managed_attempt"] = product.routed(product.dbe(attempt), attempt)[
        "event_type"
    ]
    # families/node_lifecycle.py: a reboot finding of a running attempt opens
    # its own group record (derived ``incident-<digest>`` id, its event id).
    reboot = "ip-10-0-43-8.us-west-2.compute.internal"
    lost = product.host(INVENTORY, node=reboot, **product.attempt(reboot))
    replacement = product.routed(lost, reboot)
    assert replacement["event_id"] == lost, replacement
    assert replacement["incident_id"].startswith("incident-"), replacement
    labels["attempt_reboot"] = replacement["event_type"]

    assert labels == {
        "minted": "NODE_HEALTH",
        "critical_quarantine_joins": "NODE_HEALTH_GROUP",
        "device_resource_joins": "NODE_RESOURCE_GROUP",
        "managed_attempt": "GPU_FAULT_GROUP",
        "attempt_reboot": "NODE_HEALTH_GROUP",
    }, "a routing changed its label: update NODE_HEALTH_EVENT_TYPES and its matchers"
    assert set(labels.values()) == NODE_HEALTH_EVENT_TYPES, (
        "the harness set names a label no routing gives, or misses one"
    )


def test_the_products_other_incident_labels_stay_out_of_the_set() -> None:
    # The shared-storage failure domain is a cluster-scoped correlation record
    # with no workflow (``general_live_shared`` reads it by its own label); a
    # placement hold, an operator maintenance record and an XID/SXID decision
    # are not a node-health finding's record.
    others = {
        SHARED_STORAGE_FAILURE_DOMAIN_EVENT_TYPE,
        PLACEMENT_HOLD,
        OPERATOR_EVENT_TYPE,
        *(item.value for item in FaultEventType),
    }
    assert not others & NODE_HEALTH_EVENT_TYPES, others & NODE_HEALTH_EVENT_TYPES


# --------------------------------------------------------------------------- #
# DCGM and inventory selectors (019-026)
# --------------------------------------------------------------------------- #
def test_the_dcgm_selector_takes_the_isolation_a_kernel_finding_joined() -> None:
    # A volatile DBE is a CRITICAL QUARANTINE (``gpu_memory_policy``): its
    # isolation is in flight when a critical node-log quarantine arrives, which
    # joins it, so the DBE's record reads NODE_HEALTH_GROUP. Under NODE_HEALTH
    # alone the selector found nothing: "no ... finding reached an incident".
    product = _Product()
    product.dbe()
    product.kernel_lines(MCE)
    document = product.activity()
    incident = _shared(document)
    assert incident["event_type"] == "NODE_HEALTH_GROUP", incident

    assert metric_incidents(document, DBE, gpu=GPU) == [incident]
    chain = GpuChain(
        label=f"{DBE} on {GPU}",
        operations=("FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"),
        forbidden=frozenset({"RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES"}),
    )
    assert chain_errors(document, metric_incidents(document, DBE, gpu=GPU), chain) == []


@pytest.mark.parametrize(
    ("joiner", "label"),
    [(ALLOCATABLE, "NODE_RESOURCE_GROUP"), ("kernel", "NODE_HEALTH_GROUP")],
)
def test_the_inventory_selector_takes_the_reboot_chain_another_finding_joined(
    joiner: str, label: str
) -> None:
    # A missing GPU is a CRITICAL REBOOT_NODE (MARK_UNSCHEDULABLE, QUARANTINE,
    # RESTART_NODE): the allocatable mismatch it causes joins that chain as a
    # device-resource finding, a critical node-log quarantine as a terminal
    # branch -- either way one record, under a group label.
    product = _Product()
    product.host(INVENTORY)
    if joiner == "kernel":
        product.kernel_lines(MCE)
    else:
        product.host(joiner)
    document = product.activity()
    incident = _shared(document)
    assert incident["event_type"] == label, incident
    assert inventory_incidents(document, INVENTORY) == [incident]


@pytest.mark.parametrize("label", sorted(NODE_HEALTH_EVENT_TYPES))
@pytest.mark.parametrize(
    ("select", "event_id"),
    [
        (
            lambda doc: composite_incidents(doc, "GPU_MEMORY_DEGRADATION"),
            f"gpu-b1-{GPU}-composite-GPU_MEMORY_DEGRADATION",
        ),
        (fabric_incidents, "gpu-b1-node-fabric_manager_status"),
        (completeness_incidents, "gpu-dcgm-b1-dcgm-fields"),
    ],
    ids=["composite", "fabric", "completeness"],
)
def test_every_dcgm_selector_takes_the_minted_record_under_every_label(
    select: Any, event_id: str, label: str
) -> None:
    record = {"incident_id": f"inc-{event_id}", "event_id": event_id}
    document = {"incidents": [{**record, "event_type": label}]}
    assert [item["incident_id"] for item in select(document)] == [record["incident_id"]]
    assert select({"incidents": [{**record, "event_type": "XID"}]}) == [], (
        "an XID record is not a node-health finding's"
    )


# --------------------------------------------------------------------------- #
# 017 journal stages and 002 cgroup_only's log grading
# --------------------------------------------------------------------------- #
def _burst() -> tuple[dict[str, Any], dict[str, Any]]:
    """Two severe node-log findings of one read in one isolation, as written.

    What a reviewed line graded to isolate would leave: the first finding
    mints the QUARANTINE, the second joins it, the record reads
    NODE_HEALTH_GROUP with both markers, the workflow not yet run (the node is
    not cordoned when the variant reads it).
    """

    product = _Product()
    product.kernel_lines(MCE, LOCKUP)
    document = product.activity()
    incident = _shared(document)
    assert (incident["event_type"], incident["effective_action"]) == (
        "NODE_HEALTH_GROUP",
        "QUARANTINE",
    ), incident
    return document, incident


def _journal_ctx(variant: str, document: dict[str, Any], tmp_path: Path) -> Any:
    spec = next(item for item in applogs.VARIANTS if item.variant == variant)
    probe = FakeProbe(
        {
            "install-records": {"records": 1},
            "open-window": {"after": {"ActiveState": "active"}},
            "arm": {"armed": True},
            "status": {"injected": {"emission_counts": {"0": 2}}},
            "close-window": {"clean": True},
        }
    )
    regional = FakeRegional(
        cpu_python=store_script({"activity": document}),
        node_snapshot={
            "name": NODE,
            "ready": "True",
            "unschedulable": False,
            "taints": [],
            "ownership_annotations": {},
        },
    )
    return make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={f"kernel-replay-journal@{NODE}": probe},
        settings=make_settings(spec, node=NODE),
        window_minutes=60,
    )


def test_017_evidence_only_stages_see_the_isolation_two_log_findings_share(
    tmp_path: Path,
) -> None:
    # Under NODE_HEALTH alone the stages read PASS over a node-log isolation
    # opened inside the evidence-only window.
    document, incident = _burst()
    result = applogs.nccl_error(_journal_ctx("nccl_error", document, tmp_path))
    assert result["verdict"] == "FAIL", result
    assert result["stages"]["no_quarantine"] == [
        f"an application error quarantined the node: {incident['incident_id']}"
    ], result["stages"]


@pytest.mark.parametrize("label", sorted(NODE_HEALTH_EVENT_TYPES))
def test_017_cgroup_oom_counts_its_line_under_every_label(
    tmp_path: Path, label: str
) -> None:
    incident = {
        "incident_id": "inc-log-oom",
        "event_id": f"log-{CLUSTER}-{NODE}-0123456789abcdef0123456789abcdef",
        "event_type": label,
        "policy_source": "SITE_APPLICATION_LOG",
        "effective_action": "RUN_DIAGNOSTICS",
        "state": "ACTION_PENDING",
        "created_at": "2026-10-24T00:03:00+00:00",
        "workflow_request_id": "wf-oom",
        "node_ids": [NODE],
    }
    workflow = {
        "request_id": "wf-oom",
        "incident_id": "inc-log-oom",
        "status": "SUCCEEDED",
        "created_at": "2026-10-24T00:03:00+00:00",
        "official_steps": [
            {"operation": "FREEZE_EVIDENCE", "parameters": {}},
            {"operation": "VALIDATE_HOST", "parameters": {}},
        ],
        "step_executions": [],
    }
    document = {"incidents": [incident], "workflows": [workflow], "markers": []}
    result = applogs.cgroup_oom(_journal_ctx("cgroup_oom", document, tmp_path))
    assert result["verdict"] == "PASS", (label, result["errors"])


def test_002_cgroup_only_grades_the_log_isolation_two_findings_share(
    tmp_path: Path,
) -> None:
    # Were the repeated OOM lines graded as critical quarantines, the later
    # ones would join the first one's isolation: log_grading must name that
    # record, not only the isolation stage.
    burst, incident = _burst()
    document = quiet_activity(
        incidents=burst["incidents"], workflows=burst["workflows"]
    )
    spec = next(item for item in memory.VARIANTS if item.variant == "cgroup_only")
    result = spec.execute(cgroup_context(tmp_path, document))
    assert result["verdict"] == "FAIL", result
    assert result["stages"]["log_grading"] == [
        f"OOM log incident {incident['incident_id']} was graded QUARANTINE, "
        "not RUN_DIAGNOSTICS"
    ], result["stages"]
