"""The hand-off notice of a plan that contained its node, then blocked.

A plan the compiler refused only for a missing executable owner (a Runtime
Profile capability declared ``OBSERVE``, or not claimed) runs the containment
and evidence it compiled before that step -- FREEZE_EVIDENCE, the cordon, the
quarantine, the diagnostic bundle -- as its safety steps, and settles
BLOCKED/SAFETY_SETTLED with the incident QUARANTINED
(``WorkflowBuilder.containment_before_block``). This notice is what the
executor sends when that record settles: the node *is* contained, these steps
ran, these did not, this capability needs an operator, and these are the two
exits (a validated restore once the node is cleared, a close on node evidence
once it is replaced). Before it existed the same incident read "ESCALATED"
with nothing on the node and nobody told (live 2026-10-26, gpu-eks-cluster,
``row_remap_failure`` under ``memoryDiagnostics: OBSERVE``).

The builder only formats facts the terminal record carries. Its template and
version live here rather than in ``common.py``: ``common`` also ships in the
cluster executor wheel, while this notice is built and sent by the control
plane alone (the same arrangement as ``hung_support.py``).
"""

from __future__ import annotations

from collections.abc import Sequence

from gpu_fault.models import AdvisoryNotification

CONTAINED_OPERATOR_HANDOFF_TEMPLATE_VERSION = "contained-operator-handoff-zh-v1"
#: An operator acts on it; the product did what it could and holds the node.
CONTAINED_OPERATOR_HANDOFF_CATEGORY = "OPERATOR_REVIEW"

CONTAINED_OPERATOR_HANDOFF_EMAIL_TEMPLATE = """\
GPU node contained; the rest of the recovery plan needs an operator

1. Event
- Cluster: {cluster_id}
- Nodes: {nodes}
- Incident: {incident_id}
- Workflow: {workflow_id}
- Event ID: {event_id}
- Policy source: {policy_source}
- Official action: {official_action}
- Runtime Profile: {runtime_profile_version}

2. What the product did (containment)
- Steps run: {containment_steps}
- The nodes are cordoned and carry this incident's gpu-fault.io/quarantined
  taint; nothing else was changed: no workload was stopped, no GPU reset, no
  node rebooted or replaced by this workflow.
- Record: workflow BLOCKED/SAFETY_SETTLED, incident QUARANTINED. The node
  stays out of the training pool until an operator acts.

3. What needs an operator
- Capabilities without an executable owner in the Runtime Profile:
{missing_owners}
- Steps that did not run: {waiting_steps}
- Blocking reasons:
{blocked_reasons}

4. Operator exit
- Perform the missing step by hand (for memoryDiagnostics: the NVIDIA field
  diagnostic of the GPU named by the incident) and decide on the hardware.
- Node cleared (the fault is gone and the GPU may return): run the validated
  restore under this incident --
  gpu-fault-admin submit-remediation --state-dir <state-dir> --incident-id
  {incident_id} --disposition restore --plan, then without --plan. It validates
  the node, releases the cordon and the taint, ends the incident RECOVERED and
  the dispatcher settles this record with it. The restore does not restart a
  job the plan stopped; its owner does.
- Node replaced (RMA): once the node has left the cluster, the departed-node
  cleanup or gpu-fault-admin workflow-reconcile --state-dir <state-dir>
  --close-quarantined --reason ... --reference ... closes the incident on the
  departure proof. A node still in the cluster with its isolation in place
  cannot be closed; restore it or replace it.
- Do not edit Store rows and do not remove the taint by hand: a taint removed
  by hand orphans the incident's annotations and refuses both exits until the
  evidence pass strips them.

5. Trigger reasons
{reasons}

Template: {template_version}
"""


class ContainedOperatorHandoffEmailBuilder:
    """Renders the hand-off of a contained node whose plan needs an operator.

    One notice per workflow: the deduplication key carries the workflow id and
    nothing that varies between two attempts to send it.
    """

    def build(
        self,
        *,
        cluster_id: str,
        incident_id: str,
        workflow_id: str,
        event_id: str,
        node_ids: Sequence[str],
        containment_operations: Sequence[str],
        waiting_operations: Sequence[str],
        missing_owner_capabilities: Sequence[str],
        blocked_reasons: Sequence[str],
        policy_source: str,
        official_action: str | None,
        runtime_profile_version: str | None,
        reasons: Sequence[str],
    ) -> AdvisoryNotification:
        nodes = ", ".join(node_ids) or "UNKNOWN"
        body = CONTAINED_OPERATOR_HANDOFF_EMAIL_TEMPLATE.format(
            cluster_id=cluster_id,
            nodes=nodes,
            incident_id=incident_id,
            workflow_id=workflow_id,
            event_id=event_id,
            policy_source=policy_source,
            official_action=official_action or "NONE",
            runtime_profile_version=runtime_profile_version or "UNKNOWN",
            containment_steps=" -> ".join(containment_operations) or "NONE",
            missing_owners=(
                "\n".join(f"  - {item}" for item in missing_owner_capabilities)
                or "  - UNKNOWN"
            ),
            waiting_steps=" -> ".join(waiting_operations) or "NONE",
            blocked_reasons=(
                "\n".join(f"  - {item}" for item in blocked_reasons) or "  - UNKNOWN"
            ),
            reasons="\n".join(f"  - {item}" for item in reasons) or "  - UNKNOWN",
            template_version=CONTAINED_OPERATOR_HANDOFF_TEMPLATE_VERSION,
        )
        return AdvisoryNotification(
            deduplication_key=(
                f"{cluster_id}/{incident_id}/contained-operator-handoff/{workflow_id}"
            ),
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=(
                "[GPU FAULT][contained, operator action required] "
                f"{cluster_id}: {nodes}"
            ),
            body_text=body,
            support_case_draft="",
            evidence_refs=[],
            category=CONTAINED_OPERATOR_HANDOFF_CATEGORY,
        )
