"""The notice of a remediation that ended holding its node.

A plan whose last step is the hold itself -- the node-health QUARANTINE plan
FREEZE_EVIDENCE -> MARK_UNSCHEDULABLE -> [STOP_WORKLOADS] -> QUARANTINE of a
BMC sensor, a host signal or a GPU finding graded QUARANTINE, or a critical
quarantine folded into another chain as its terminal branch -- ends SUCCEEDED
with the node cordoned, tainted and the incident QUARANTINED. Nothing in that
plan mails: the isolation step saves no notification and there is no
ESCALATE_SUPPORT. This notice is what the executor sends at that terminal
write (``execution/terminal_quarantine_notice.py``): the node *is* held, by
these steps, for this finding, and these are the two exits (the validated
restore once the cause is repaired, a close on the departure proof once the
node is replaced). The incident stays QUARANTINED; no support case is opened.

The builder only formats facts the terminal record carries. Its template and
version live here rather than in ``common.py``: ``common`` also ships in the
cluster executor wheel, while this notice is built and sent by the control
plane alone (the arrangement of ``contained_operator_handoff.py``).
"""

from __future__ import annotations

from collections.abc import Sequence

from gpu_fault.models import AdvisoryNotification

QUARANTINED_NODE_NOTICE_TEMPLATE_VERSION = "quarantined-node-notice-zh-v1"
#: An operator acts on it; the product did what it could and holds the node.
QUARANTINED_NODE_NOTICE_CATEGORY = "OPERATOR_REVIEW"

QUARANTINED_NODE_NOTICE_EMAIL_TEMPLATE = """\
GPU node quarantined; the repair and the exit need an operator

1. Event
- Cluster: {cluster_id}
- Nodes: {nodes}
- Incident: {incident_id}
- Workflow: {workflow_id}
- Event ID: {event_id}
- Policy source: {policy_source}
- Recommended action: {recommended_action}
- Runtime Profile: {runtime_profile_version}

2. Finding
{reasons}

3. What the product did
- Steps run: {steps_run}
- The nodes are cordoned and carry this incident's gpu-fault.io/quarantined
  taint: no new workload is scheduled onto them. This workflow opened no
  support case.
- Workloads the remediation suspended and did not restart (a validated
  restore does not restart them either; their owner must):
{suspended_workloads}
- Record: workflow SUCCEEDED, incident QUARANTINED. The node stays out of the
  training pool until an operator acts.

4. Operator exit
- Repair the cause the finding names (the component a BMC sensor reads, the
  filesystem, the disk, the link, the host or the GPU), or replace the node.
- Node repaired: run the validated restore under this incident --
  gpu-fault-admin submit-remediation --state-dir <state-dir> --incident-id
  {incident_id} --disposition restore --plan, then without --plan. It validates
  the node, releases the cordon and the taint and ends the incident RECOVERED.
- Node replaced: once the node has left the cluster, the departed-node
  cleanup or gpu-fault-admin workflow-reconcile --state-dir <state-dir>
  --close-quarantined --reason ... --reference ... closes the incident on the
  departure proof. A node still in the cluster with its isolation in place
  cannot be closed; restore it or replace it.
- Do not edit Store rows and do not remove the taint by hand: a taint removed
  by hand orphans the incident's annotations and refuses both exits until the
  evidence pass strips them.

Template: {template_version}
"""


class QuarantinedNodeNoticeEmailBuilder:
    """Renders the notice of a node a completed remediation left quarantined.

    One notice per incident: the deduplication key carries the incident and
    nothing that varies between two terminal writes of the same hold -- not
    the workflow, so a re-run or a second record ending on it sends nothing
    new.
    """

    def build(
        self,
        *,
        cluster_id: str,
        incident_id: str,
        workflow_id: str,
        event_id: str,
        node_ids: Sequence[str],
        steps_run: Sequence[str],
        suspended_workloads: Sequence[str],
        policy_source: str,
        recommended_action: str | None,
        runtime_profile_version: str | None,
        reasons: Sequence[str],
    ) -> AdvisoryNotification:
        nodes = ", ".join(node_ids) or "UNKNOWN"
        body = QUARANTINED_NODE_NOTICE_EMAIL_TEMPLATE.format(
            cluster_id=cluster_id,
            nodes=nodes,
            incident_id=incident_id,
            workflow_id=workflow_id,
            event_id=event_id,
            policy_source=policy_source,
            recommended_action=recommended_action or "NONE",
            runtime_profile_version=runtime_profile_version or "UNKNOWN",
            reasons="\n".join(f"  - {item}" for item in reasons) or "  - UNKNOWN",
            steps_run=" -> ".join(steps_run) or "NONE",
            suspended_workloads=(
                "\n".join(f"  - {item}" for item in suspended_workloads) or "  - NONE"
            ),
            template_version=QUARANTINED_NODE_NOTICE_TEMPLATE_VERSION,
        )
        return AdvisoryNotification(
            deduplication_key=f"{cluster_id}/{incident_id}/quarantined-node-notice",
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=(
                "[GPU FAULT][quarantined, operator action required] "
                f"{cluster_id}: {nodes}"
            ),
            body_text=body,
            support_case_draft="",
            evidence_refs=[],
            category=QUARANTINED_NODE_NOTICE_CATEGORY,
        )
