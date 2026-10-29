"""The notice that a GPU cluster's remediation storm breaker tripped.

``gpu_fault.remediation_storm`` holds a cluster's new destructive remediation
once more nodes entered it in ten minutes than one fault explains: at least
max(3, ceil(20% of the cluster's GPU nodes)) nodes across two or more
workload domains. Nothing resumes without an operator, so this notice is what
tells one: which cluster, which nodes, the numbers it tripped on, that work in
flight goes on, and the two exits. The dispatcher sends it once per trip.

The template lives here, not in ``common.py``: the control plane alone builds
and sends it (the arrangement of ``contained_operator_handoff.py``).
"""

from __future__ import annotations

from gpu_fault.models import AdvisoryNotification
from gpu_fault.remediation_storm import STORM_WINDOW_SECONDS, RemediationStormBreaker

REMEDIATION_STORM_TEMPLATE_VERSION = "remediation-storm-tripped-zh-v1"
#: An operator decides; the product holds the cluster's new destructive work.
REMEDIATION_STORM_CATEGORY = "OPERATOR_REVIEW"
#: Ahead of the per-node mail a storm queues (lower sorts first).
REMEDIATION_STORM_PRIORITY = 0

REMEDIATION_STORM_EMAIL_TEMPLATE = """\
Remediation storm breaker tripped: new destructive remediation in {cluster_id} is held

1. What happened
- Cluster: {cluster_id}
- Tripped at: {tripped_at} (trip {trip_id})
- {node_count} GPU nodes entered destructive remediation within {window}s, across
  {domains} workload domains (a job, or an incident without a job, is one domain).
- Threshold: {threshold} nodes = max(3, ceil(20% of {cluster_nodes} GPU nodes)).
- Nodes: {nodes}
- Workflow that tripped it: {workflow_id}

2. What the product does now
- New destructive work in this cluster (STOP/RESTART of jobs, GPU resets,
  quiesce, reboots, replacements, quarantine, driver/firmware, device plugins)
  is held, not failed: those workflows stay PENDING, each recorded with the
  scope storm:{cluster_id}.
- Work already in flight goes on and is not interrupted.
- Nothing resumes by itself; gpu-fault-admin status shows the breaker and the
  alert GpuFaultRemediationStormBreakerOpen stays firing until an operator acts.

3. Operator exit
- Find the common cause first (a driver, image, policy or firmware rollout, a
  fabric or power event, a bad detector) and stop it.
- Resume: gpu-fault-admin remediation-storm release --state-dir <state-dir>
  --cluster-id {cluster_id} --reason ... --reference ...
  The held workflows run, without counting again; the breaker re-arms for new
  faults.
- Hand the held work to people instead: gpu-fault-admin remediation-storm abort
  --state-dir <state-dir> --cluster-id {cluster_id} --reason ... --reference ...
  The held workflows are closed and their incidents ESCALATED; the breaker stays
  open until a later release.
- Do not edit the breaker or the workflow rows in the Store.

Template: {template_version}
"""


class RemediationStormEmailBuilder:
    """Renders one notice per trip: the key carries the trip id only."""

    def build(self, *, record: RemediationStormBreaker) -> AdvisoryNotification:
        nodes = ", ".join(record.trip_node_ids) or "UNKNOWN"
        body = REMEDIATION_STORM_EMAIL_TEMPLATE.format(
            cluster_id=record.cluster_id,
            tripped_at=record.tripped_at.isoformat()
            if record.tripped_at
            else "UNKNOWN",
            trip_id=record.trip_id or "UNKNOWN",
            node_count=len(record.trip_node_ids),
            window=STORM_WINDOW_SECONDS,
            domains=record.trip_domains,
            threshold=record.trip_threshold,
            cluster_nodes=(
                "UNKNOWN"
                if record.trip_cluster_nodes is None
                else record.trip_cluster_nodes
            ),
            nodes=nodes,
            workflow_id=record.trip_workflow_id or "UNKNOWN",
            template_version=REMEDIATION_STORM_TEMPLATE_VERSION,
        )
        return AdvisoryNotification(
            deduplication_key=(
                f"{record.cluster_id}/remediation-storm-tripped/{record.trip_id}"
            ),
            cluster_name=record.cluster_id,
            incident_id=f"remediation-storm-{record.cluster_id}",
            subject=(
                "[GPU FAULT][remediation storm, operator action required] "
                f"{record.cluster_id}: {len(record.trip_node_ids)} nodes in "
                f"{STORM_WINDOW_SECONDS}s"
            ),
            body_text=body,
            support_case_draft="",
            evidence_refs=[],
            category=REMEDIATION_STORM_CATEGORY,
            priority=REMEDIATION_STORM_PRIORITY,
        )
