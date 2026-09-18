from __future__ import annotations

import json
from threading import RLock

from gpu_fault.operation_registry import (
    NODE_EXCLUSIVE_OPERATIONS,
    WORKLOAD_SCOPED_OPERATIONS,
)
from gpu_fault.orchestration import DagBrancher, RecoveryArbiter
from gpu_fault.orchestration.families import (
    EvidenceOperationService,
    NodeConflictService,
    NodeHealthCallbacks,
    NodeHealthIngestionService,
    NodeHealthPlanBuilder,
    ValidationOperationService,
)
from gpu_fault.orchestration.workflow_merge import WorkflowMergeService
from tests.orchestration._cov95_runtime_builder import builder


def health_service(store):
    arbiter = RecoveryArbiter()
    conflicts = NodeConflictService(store, arbiter)
    evidence = EvidenceOperationService(store, conflicts.active_node_exclusive_workflow)
    merge = WorkflowMergeService(
        arbiter,
        DagBrancher(arbiter),
        preemption_enabled=True,
        workload_scoped_operations=set(WORKLOAD_SCOPED_OPERATIONS),
        node_exclusive_operations=set(NODE_EXCLUSIVE_OPERATIONS),
        workflow_resource_claims_by_node=conflicts.workflow_resource_claims_by_node,
    )
    absorbed = []
    callbacks = NodeHealthCallbacks(
        active_node_exclusive_workflow=conflicts.active_node_exclusive_workflow,
        active_workflow_covers_inventory_finding=conflicts.active_workflow_covers_inventory_finding,
        attempt_observation=evidence.attempt_observation,
        claims_node_exclusively=conflicts.claims_node_exclusively,
        # Neighboring families decline; the tested health-family fallback stays real.
        ingest_grouped_health_finding=lambda finding: None,
        ingest_grouped_node_replacement=lambda finding: None,
        ingest_grouped_node_resource_finding=lambda finding: None,
        ingest_terminal_node_quarantine=lambda finding: None,
        inventory_validation_parameters=ValidationOperationService.inventory_parameters,
        node_group_key=lambda cluster, node: json.dumps(
            ["node-scope", cluster, node], separators=(",", ":")
        ),
        preemption_scope_matches=merge.preemption_scope_matches,
        prepare_preempting_successor=merge.prepare_preempting_successor,
        sample_hung_triage_nodes=evidence.sample_hung_triage_nodes,
        record_host_resource_absorb=lambda: absorbed.append("record-only"),
    )
    compiler = builder()
    compiler.store = store
    service = NodeHealthIngestionService(
        store, RLock(), NodeHealthPlanBuilder(store, compiler, callbacks), callbacks
    )
    return service, absorbed
