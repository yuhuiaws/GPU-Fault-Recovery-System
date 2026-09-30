from __future__ import annotations

import os as os
from threading import RLock as RLock
from typing import Any as Any
from typing import Protocol as Protocol

from pydantic import Field as Field
from pydantic import model_validator as model_validator

from gpu_fault.hyperpod import (
    HyperPodRecoveryAdvisory as HyperPodRecoveryAdvisory,
)
from gpu_fault.models import (
    AdvisoryNotification as AdvisoryNotification,
)
from gpu_fault.models import (
    NotificationResult as NotificationResult,
)
from gpu_fault.models import (
    NotificationStatus as NotificationStatus,
)
from gpu_fault.models import StrictModel as StrictModel

# Template texts are English. The version tags keep their historical "zh"
# lineage on purpose: several deduplication keys embed them, and a retag
# would re-notify every open incident once. Bump a tag only when a
# template's meaning changes, never for wording.
RESTART_GUARD_TEMPLATE_VERSION = "restart-guard-zh-v2"
RESTART_WORKLOAD_TEMPLATE_VERSION = "restart-workload-zh-v1"
RESTART_NODE_TEMPLATE_VERSION = "restart-node-zh-v1"
WARM_SPARE_REPLACEMENT_TEMPLATE_VERSION = "warm-spare-replacement-zh-v1"
RESTART_FABRIC_MANAGER_TEMPLATE_VERSION = "restart-fabric-manager-zh-v1"
GPU_RESET_TEMPLATE_VERSION = "gpu-reset-zh-v1"
FABRIC_RESET_TEMPLATE_VERSION = "fabric-reset-zh-v1"
NOT_APPLICABLE_TEMPLATE_VERSION = "xid-not-applicable-zh-v2"
XID_INVESTIGATORY_TEMPLATE_VERSION = "xid-investigatory-zh-v1"
SXID_EVENT_TEMPLATE_VERSION = "sxid-event-zh-v1"
HARDWARE_ESCALATION_TEMPLATE_VERSION = "hardware-escalation-zh-v2"
NVLINK74_SUPPORT_TEMPLATE_VERSION = "xid74-support-zh-v3"
NVLINK74_MECHANICAL_TEMPLATE_VERSION = "xid74-mechanical-zh-v1"
GPU_MECHANICAL_TEMPLATE_VERSION = "gpu-mechanical-zh-v1"
DCGM_DIAGNOSTIC_TEMPLATE_VERSION = "dcgm-diagnostic-zh-v2"
EFA_RDMA_EVENT_TEMPLATE_VERSION = "efa-rdma-event-zh-v1"
HARDWARE_INVENTORY_TEMPLATE_VERSION = "hardware-inventory-mismatch-zh-v1"
HOST_RESOURCE_EVENT_TEMPLATE_VERSION = "host-resource-event-zh-v1"
DIAGNOSTIC_INCONCLUSIVE_TEMPLATE_VERSION = "diagnostic-inconclusive-zh-v1"

DIAGNOSTIC_INCONCLUSIVE_EMAIL_TEMPLATE = """\
GPU node diagnostic inconclusive

1. Event
- Cluster: {cluster_id}
- Nodes: {nodes}
- Incident: {incident_id}
- Workflow: {workflow_id}
- Event ID: {event_id}
- Policy source: {policy_source}
- Official action: {official_action}

2. Diagnostic result
- Workflow steps: {operations}
- Failed step: {failed_operation}
- Failure reason: {error}
- Conclusion: diagnostic inconclusive. The workflow contained only
  evidence/diagnostic/validation steps; no node was changed or isolated.

3. System handling
- The incident is closed as RECOVERED with reason "diagnostic inconclusive".
- The incident's marker is retired; the node is no longer treated as "under
  repair". Later training failures follow the normal path (fast triage or
  restart).
- No recovery action and no AWS Support case was created. Review manually if
  the node keeps alerting.

4. Trigger reasons
{reasons}

Template: {template_version}
"""

HARDWARE_INVENTORY_EMAIL_TEMPLATE = """\
GPU node hardware inventory mismatch

1. Event
- Cluster: {cluster_id}
- Node: {node_id}
- Incident: {incident_id}
- Workflow: {workflow_id}
- Event ID: {event_id}
- Observed at: {observed_at}
- EC2/HyperPod instance type: {node_instance_type}
- Hardware type: {resource_type}
- Severity: {severity}

2. Inventory check
- Metric: {metric_name}
- Expected count: {expected_count}
- Current ACTIVE count: {observed_count}
- Currently discovered count: {discovered_count}
- Missing: {missing_count}
- Excess: {excess_count}
- Consecutive abnormal samples: {consecutive_samples}
- Samples required to trigger: {required_samples}

3. Related training workloads
- Workload state: {workload_state}
- Affected workloads: {workloads}

4. System handling
- Policy action: {recommended_action}
- Workflow steps: {workflow_steps}
- Policy source: {policy_source}
- Policy version: {policy_version}

5. Evidence
{evidence_refs}

Template: {template_version}
"""

EFA_RDMA_EVENT_EMAIL_TEMPLATE = """\
GPU EFA/RDMA monitoring event

1. Event
- Cluster: {cluster_id}
- Node: {node_id}
- Incident: {incident_id}
- Workflow: {workflow_id}
- Event ID: {event_id}
- Observed at: {observed_at}
- Event type: {event_type}
- Severity: {severity}

2. Monitoring signal
- Metric: {metric_name}
- Device/Port: {device}
- Current value: {value}
- Traffic state: {traffic_signal}
- Traffic baseline: {baseline}
- Trigger reason: {reason}

3. Related training workloads
- Workload state: {workload_state}
- Job ID: {job_id}
- Attempt ID: {attempt_id}
- Affected workloads: {workloads}

4. System handling
- Policy action: {recommended_action}
- Policy source: {policy_source}
- Policy version: {policy_version}
- Workflow steps: {workflow_steps}

5. Evidence
{evidence_refs}

Template: {template_version}
"""

HOST_RESOURCE_EVENT_EMAIL_TEMPLATE = """\
GPU node host resource risk

1. Event
- Cluster: {cluster_id}
- Node: {node_id}
- Incident: {incident_id}
- Workflow: {workflow_id}
- Event ID: {event_id}
- Observed at: {observed_at}
- Severity: {severity}

2. Monitoring signal
- Signal: {signal}
- Metric: {metric_name}
- Device/Mount: {device}
- Current value: {value}
- Comparison: {comparison}
- Configured threshold: {threshold_percent}%
- Minimum duration threshold: {minimum_active_seconds} s
- Trigger reason: {reason}
- Related metrics in the same batch: {related_metrics}

3. Related training workloads
- Workload state: {workload_state}
- Affected workloads: {workloads}

4. System handling
- Policy action: {recommended_action}
- Workflow steps: {workflow_steps}
- Policy source: {policy_source}
- Policy version: {policy_version}

5. Evidence
{evidence_refs}

Template: {template_version}
"""

DCGM_DIAGNOSTIC_EMAIL_TEMPLATE = """\
GPU DCGM quick diagnostic result

1. Event
- Cluster: {cluster_id}
- Incident: {incident_id}
- Workflow: {workflow_id}
- Event ID: {event_id}
- Affected workloads: {workloads}

2. Diagnostic result
{node_results}

3. System handling
- Control-plane action: {control_plane_action}
- PASS/WARN: proceed to the thermal cool-down observation and GPU validation.
- FAIL/INCONCLUSIVE: stop the affected workloads, cordon and isolate the node;
  training is not restarted automatically.
- Configuration-only failures (dcgmErrorSeverity_t=CONFIG, e.g. persistence
  mode off) are treated as WARN: the node is not isolated and workloads are not
  stopped; fix the host configuration per the guidance below and rerun the
  diagnostic.

4. Next steps
{recommended_actions}

5. Evidence
{evidence_refs}

Template: {template_version}
"""

NVLINK74_SUPPORT_EMAIL_TEMPLATE = """\
GPU XID 74 NVLink event: vendor support

1. Event
- Cluster: {cluster_id}
- Incident: {incident_id}
- Workflow: {workflow_id}
- Nodes: {node_ids}
- Affected workloads: {workloads}
- Event ID: {event_id}
- Policy source: {policy_source}
- Official action: {official_action}
- Decode and disposition reasons:
{reasons}

2. System handling
- No single-GPU reset was performed for XID 74.
- Node scheduling and training state follow the workflow's recorded steps.
- An internal vendor-support escalation record was created: {ticket_id}

3. Ticket
- Internal Ticket ID: {ticket_id}
- Workflow operation: ESCALATE_SUPPORT

Template: {template_version}
"""

NVLINK74_MECHANICAL_EMAIL_TEMPLATE = """\
GPU XID 74 NVLink mechanical check required

1. Event
- Cluster: {cluster_id}
- Incident: {incident_id}
- Workflow: {workflow_id}
- Nodes: {node_ids}
- NVLink: {link_id}
- PCI BDF: {pci_bdf}
- Register counts: {occurrence_counts}

2. Current state
- Workflow operation: CHECK_MECHANICALS
- Training and scheduling state follow the workflow's recorded steps.
- The system never confirms the physical check on its own.

3. Confirmation field
- Annotation: {annotation}
- Required value: {annotation_value}

Template: {template_version}
"""

GPU_MECHANICAL_EMAIL_TEMPLATE = """\
GPU mechanical check required

1. Event
- Cluster: {cluster_id}
- Incident: {incident_id}
- Workflow: {workflow_id}
- Nodes: {node_ids}
- XID: {xid}
- PCI BDF: {pci_bdf}

2. Current state
- Workflow operation: CHECK_MECHANICALS
- Training and scheduling state follow the workflow's recorded steps.
- The system never confirms the physical check on its own.

3. Confirmation field
- Annotation: {annotation}
- Required value: {annotation_value}

Template: {template_version}
"""

HARDWARE_ESCALATION_EMAIL_TEMPLATE = """\
GPU node automatic recovery failed: hardware taken out of service

1. Event
- Cluster: {cluster_id}
- Incident: {incident_id}
- Workflow: {workflow_id}
- Faulty nodes: {node_ids}
- Affected workloads: {workloads}
- Event type: {event_type}
- Original event: {event_id}
- Failure reasons:
{reasons}

2. System handling
- Automatic recovery steps that failed: {failed_operations}
- The faulty nodes stay unschedulable and quarantined; scheduling is not
  restored.
- An internal vendor-support escalation record was created: {ticket_id}

3. Ticket
- Internal Ticket ID: {ticket_id}
- Policy source: {policy_source}
- Official action: {official_action}
- Workflow operation: ESCALATE_SUPPORT

Template: {template_version}
"""

NOT_APPLICABLE_EMAIL_TEMPLATE = """\
GPU XID policy not applicable

1. Event
- Cluster: {cluster_id}
- Node: {node_id}
- XID: {xid}
- GPU product: {product}
- Driver branch: {driver_branch}
- CUDA version: {cuda_version}
- Observed at: {observed_at}
- Event ID: {event_id}
- Workload state: {workload_state}
- Affected workloads: {workloads}

2. Policy decision
- Disposition: NOT_APPLICABLE
- NVIDIA catalog action: {official_action}
- NVIDIA investigatory action: {investigatory_action}
- Policy source: {policy_source}
- Policy version: {policy_version}
- Decision reasons:
{reasons}

3. System handling
- Official recovery action: not executed
- Safety action: {safety_action}
- Incident: {incident_id}
- Workflow: {workflow_id}
- Workflow status: {workflow_status}
- Safety steps: {safety_steps}
- Official steps: {official_steps}

4. Evidence
{evidence_refs}

Template: {template_version}
"""

XID_INVESTIGATORY_EMAIL_TEMPLATE = """\
GPU XID investigatory action

1. Event
- Cluster: {cluster_id}
- Node: {node_id}
- XID: {xid}
- GPU product: {product}
- Observed at: {observed_at}
- Event ID: {event_id}
- Incident: {incident_id}
- Workflow: {workflow_id}
- Workload state: {workload_state}
- Affected workloads: {workloads}

2. Policy decision
- Disposition: {disposition}
- NVIDIA immediate action: {official_action}
- Effective action: {effective_action}
- NVIDIA investigatory action: {investigatory_action}
- Policy source: {policy_source}
- Policy version: {policy_version}
- Decision reasons:
{reasons}

3. Investigatory action status
- The system does not execute investigatory actions automatically.
- Whether to run this investigatory action is the administrator's decision.
- Immediate action and training state follow the workflow's recorded steps.

4. Evidence
{evidence_refs}

Template: {template_version}
"""

SXID_EVENT_EMAIL_TEMPLATE = """\
GPU NVSwitch SXID event

1. Event
- Cluster: {cluster_id}
- Node: {node_id}
- SXID: {sxid}
- GPU product: {product}
- Observed at: {observed_at}
- Event ID: {event_id}
- Collector source: {event_source}
- Evidence: {evidence_ref}
- Workload state: {workload_state}
- Affected workloads: {workloads}

2. NVSwitch details
- Classification: {classification}
- Classification source: {classification_source}
- Link scope: {link_scope}
- Link scope source: {link_scope_source}
- Switch: {switch_id}
- Port: {port}
- PCI BDF: {pci_bdf}
- Fabric partition: {fabric_partition}

3. Policy decision
- Disposition: {disposition}
- NVIDIA official action: {official_action}
- Effective action: {effective_action}
- Safety action: {safety_action}
- NVIDIA investigatory action: {investigatory_action}
- Policy source: {policy_source}
- Policy version: {policy_version}
- Decision reasons:
{reasons}

4. System records
- Incident: {incident_id}
- Workflow: {workflow_id}
- Follow-up recovery actions and their results are reported with their own
  fixed templates.

Template: {template_version}
"""

GPU_COUNT_CHANGE_EMAIL_TEMPLATE = """\
GPU training restart paused: administrator decision required

1. What happened
- Training job: {job_id}
- Failed attempt: {attempt_id}
- Cluster: {cluster_id}
- Affected workloads: {workloads}
- GPUs used by the failed attempt: {source_gpu_count}
- GPUs the current template would use: {target_gpu_count}
- The GPU count differs between the failed attempt and the restart, or could
  not be confirmed, so the automatic restart was blocked.
- No new training workload was created and no restart budget was consumed.

2. What the administrator should do
Option A: restore the original resources. Set the workload template back to
{source_gpu_count} GPUs; the restart continues once the counts match again.
Option B: accept {target_gpu_count} GPUs. First change and review:
1. world size and the data/tensor/pipeline parallel strategy;
2. per-device batch size, gradient accumulation and global batch size;
3. learning rate and its schedule;
4. checkpoint compatibility with the new parallel topology;
5. expected critical ranks and resource requests in the training template.
After these changes are made and reviewed, run exactly the approval command
below. Do not approve before the training parameters are adjusted:
{approval_commands}

After approval the control plane re-reads the workload template and continues
only while the actual GPU change is still {approval_annotation}.

3. Tracking
- Incident: {incident_id}
- Approval annotation: gpu-fault.io/approve-gpu-count-change={approval_annotation}
- Template: {template_version}
"""

RESTART_BUDGET_EMAIL_TEMPLATE = """\
GPU training automatic restarts exhausted: administrator decision required

1. What happened
- Training job: {job_id}
- Latest failed attempt: {attempt_id}
- Cluster: {cluster_id}
- Automatic restarts used or reserved: {restart_count}
- Configured maximum automatic restarts: {restart_budget}
- Automatic restarts have stopped to keep a repeatedly failing job from
  holding GPU resources.

2. What the administrator should do
1. Review the GPUs, nodes, training logs and checkpoints of the recent failures;
2. Decide whether the fault is fixed and whether the current resources and
   training parameters still apply;
3. Do not raise the restart budget to hide a repeating fault;
4. Once it is safe to continue, submit a new training job manually under a new
   job ID.

3. Tracking
- Incident: {incident_id}
- Template: {template_version}
"""

RESTART_WORKLOAD_EMAIL_TEMPLATE = """\
GPU training workload restarted automatically

1. Event
- Cluster: {cluster_id}
- Incident: {incident_id}
- Training job: {job_id}
- Affected workloads: {workloads}

2. Restart
- System action: RESTART_WORKLOAD
- Previous attempt: {source_attempt_id}
- New attempt: {restart_attempt_id}
- Previous GPU count: {source_gpu_count}
- Restart GPU count: {target_gpu_count}
- Automatic restarts used by this job: {restart_count}
- Maximum automatic restarts for this job: {restart_budget}
- Kubernetes restart operation: submitted successfully

3. Tracking
- Workflow: {workflow_id}
- Operation: {operation_id}
- Template: {template_version}
"""

RESTART_NODE_EMAIL_TEMPLATE = """\
GPU node restarted automatically

1. Event
- Cluster: {cluster_id}
- Incident: {incident_id}
- Workflow: {workflow_id}
- Triggering event: {event_id}
- Fault type: {event_type}
- Fault identifier: {fault_identifier}
- Collector source: {event_source}
- Policy source: {policy_source}
- Policy action: {official_action}
- Effective action: {effective_action}
- Restart reasons:
{reasons}
- Affected nodes: {node_ids}

2. Restart
- System action: RESTART_NODE
- HyperPod action: BatchRebootClusterNodes
- HyperPod operation: {operation_id}
- Restarted nodes:
{node_observations}
- Restart confirmed by: {confirmation_source}
- Restart status: confirmed complete

3. Follow-up
- GPU validation: the workflow's VALIDATE_GPU step
- Fabric validation: the workflow's VALIDATE_FABRIC step
- Scheduling: restored by the RESTORE_SCHEDULING step after validation passes

Template: {template_version}
"""

WARM_SPARE_REPLACEMENT_EMAIL_TEMPLATE = """\
GPU node replaced by warm spare

1. Event
- Cluster: {cluster_id}
- Incident: {incident_id}
- Workflow: {workflow_id}
- Event ID: {event_id}
- Policy source: {policy_source}
- Official action: {official_action}
- Effective action: {effective_action}

2. Replacement result
- Faulty nodes: {fault_nodes}
- Activated warm spares: {spare_nodes}
- Node rebindings:
{node_rebindings}
- Confirmation source: {confirmation_source}
- Provider replacement API submitted: {provider_mutation_submitted}
- Operation ID: {operation_id}

3. System state
- The REPLACE_NODE branch completed successfully.
- The faulty nodes stay isolated; training continues on the rebound warm
  spares.
- GPU/Fabric validation, scheduling restore and the training restart follow
  the workflow's final state.

4. Disposition reasons
{reasons}

Template: {template_version}
"""

RESTART_FABRIC_MANAGER_EMAIL_TEMPLATE = """\
GPU Fabric Manager restarted automatically

1. Event
- Cluster: {cluster_id}
- Incident: {incident_id}
- Workflow: {workflow_id}
- Triggering event: {event_id}
- Fault type: {event_type}
- Policy source: {policy_source}
- Policy action: {official_action}
- Trigger reasons:
{reasons}
- Affected workloads: {workloads}

2. Restart
- System action: RESTART_FABRIC_MANAGER
- Operation: {operation_id}
- Service: nvidia-fabricmanager
- Per-node results:
{node_results}
- Restart status: confirmed complete

Template: {template_version}
"""

GPU_RESET_EMAIL_TEMPLATE = """\
GPU reset completed automatically

1. Event
- Cluster: {cluster_id}
- Incident: {incident_id}
- Workflow: {workflow_id}
- Triggering event: {event_id}
- Fault type: {event_type}
- Policy source: {policy_source}
- Policy action: {official_action}
- Trigger reasons:
{reasons}
- Affected workloads: {workloads}

2. Execution
- System action: RESET_GPU
- Operation: {operation_id}
- Target nodes: {node_ids}
- Target GPU UUIDs: {gpu_uuids}
- Results:
{node_results}
- Reset status: executed successfully

3. Follow-up
- GPU services: restored by the workflow's RESTORE_GPU_SERVICES step
- GPU validation: the workflow's VALIDATE_GPU step
- Scheduling: restored by the RESTORE_SCHEDULING step after validation passes
- Training: restarted by the RESTART_WORKLOAD step after all validation passes

Template: {template_version}
"""

FABRIC_RESET_EMAIL_TEMPLATE = """\
GPU/NVSwitch reset completed automatically

1. Event
- Cluster: {cluster_id}
- Incident: {incident_id}
- Workflow: {workflow_id}
- Triggering event: {event_id}
- Fault type: {event_type}
- SXID: {sxid}
- Policy source: {policy_source}
- Policy action: {official_action}
- Fabric partition: {fabric_partition}
- Trigger reasons:
{reasons}
- Affected workloads: {workloads}

2. Execution
- System action: RESET_ALL_GPUS_NVSWITCHES
- Operation: {operation_id}
- Per-node results:
{node_results}
- Reset status: executed and re-checked against the node's GPU inventory

3. Follow-up
- GPU services: restored by the workflow's RESTORE_GPU_SERVICES step
- GPU validation: the workflow's VALIDATE_GPU step
- Fabric validation: the workflow's VALIDATE_FABRIC step
- Training: restarted by the RESTART_WORKLOAD step after all validation passes

Template: {template_version}
"""
