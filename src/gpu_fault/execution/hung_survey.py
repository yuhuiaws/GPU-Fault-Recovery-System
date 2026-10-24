"""Use whole-task evidence to choose and, if needed, expand active diagnostics."""

from __future__ import annotations

import json
from typing import Any

from gpu_fault.execution.hung_bundle import bind_hung_bundle
from gpu_fault.execution.hung_classification import classify_hung_signals
from gpu_fault.hung_summary_codec import MAX_SURVEY_RAW_BYTES, decode_summary
from gpu_fault.hung_survey_analysis import survey_analysis
from gpu_fault.hung_survey_series import recheck_details
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.models import WorkflowRequest


def _record(workflow: WorkflowRequest, index: int, **details: Any) -> list[Any]:
    return [
        item.model_copy(update={"details": {**item.details, **details}})
        if item.step_index == index
        else item
        for item in workflow.step_executions
    ]


def _target(
    step: Any, nodes: list[str], all_nodes: list[str], reason: str | None = None
) -> Any:
    parameters = {
        **step.parameters,
        "hung_target_pending": False,
        "not_sampled_nodes": sorted(set(all_nodes) - set(nodes)),
    }
    if reason:
        parameters["hung_skip_reason"] = reason
    return step.model_copy(update={"node_ids": nodes, "parameters": parameters})


def balanced_nodes(
    priority: list[str],
    domains: dict[str, str],
    budget: int,
    required: list[str],
) -> list[str]:
    chosen = list(dict.fromkeys(node for node in required if node in priority))[:budget]
    while len(chosen) < min(budget, len(priority)):
        seen = {domains.get(node, node) for node in chosen}
        available = [node for node in priority if node not in chosen]
        chosen.append(
            next(
                (node for node in available if domains.get(node, node) not in seen),
                available[0],
            )
        )
    return chosen


def after_summary(
    workflow: WorkflowRequest, index: int, details: dict[str, Any]
) -> WorkflowRequest:
    steps = list(workflow.official_steps)
    step = steps[index]
    if step.parameters.get("survey_pass") == "RECHECK":
        baseline = next(
            (
                item
                for item in workflow.step_executions
                if item.operation is Op.COLLECT_HUNG_SUMMARY and item.step_index < index
            ),
            None,
        )
        details = recheck_details(baseline.details if baseline else {}, details)
    analysis = survey_analysis(step.parameters, step.node_ids, details)
    if step.parameters.get("survey_pass") == "BASELINE":
        return workflow.model_copy(
            update={"step_executions": _record(workflow, index, hung_survey=analysis)}
        )
    budget = int(step.parameters["hung_diagnostic_node_budget"])
    priority = list(
        dict.fromkeys(
            [
                *analysis["candidate_nodes"],
                *analysis.get("ras_lead_nodes", []),
                step.parameters["reporting_node_id"],
                *analysis["diagnostic_lead_nodes"],
                *analysis["related_nodes"],
                *analysis["node_priority"],
            ]
        )
    )
    eligible = set(analysis["node_priority"])
    priority = [node for node in priority if node in eligible]
    initial = balanced_nodes(
        priority,
        step.parameters.get("fabric_partition_by_node", {}),
        min(4, budget),
        [
            *analysis["candidate_nodes"],
            *analysis.get("ras_lead_nodes", []),
            step.parameters["reporting_node_id"],
            *analysis["diagnostic_lead_nodes"],
        ],
    )
    skip = "NO_VALID_SURVEY" if not initial else None
    for i, item in enumerate(steps):
        if item.operation is Op.VALIDATE_FABRIC:
            bindings = dict(item.parameters.get("training_fabric_by_node") or {})
            for node, encoded in (details.get("node_results") or {}).items():
                try:
                    scope = decode_summary(encoded).get("training_fabric_scope", {})
                    if scope.get("basis") == "TASK_DEVICE_BINDING":
                        bindings[node] = scope
                except ValueError:
                    continue
            steps[i] = item.model_copy(
                update={
                    "parameters": {
                        **item.parameters,
                        "training_fabric_by_node": bindings,
                    }
                }
            )
        if item.operation is Op.COLLECT_HUNG_TRIAGE:
            steps[i] = _target(
                item,
                [] if skip or item.parameters.get("hung_wave") == 2 else initial,
                step.node_ids,
                skip,
            )
        if item.operation is Op.COLLECT_DIAGNOSTIC_BUNDLE and not item.parameters.get(
            "preserve_existing_fr"
        ):
            steps[i] = item.model_copy(
                update={
                    "node_ids": initial or [step.parameters["reporting_node_id"]],
                    "parameters": {
                        **item.parameters,
                        "survey_coverage": {
                            key: analysis[key]
                            for key in (
                                "expected_node_count",
                                "responding_node_count",
                                "expected_rank_count",
                                "valid_rank_count",
                                "rank_inventory_complete",
                            )
                        },
                        **({"hung_skip_reason": skip} if skip else {}),
                    },
                }
            )
    return workflow.model_copy(
        update={
            "official_steps": steps,
            "dag_revision": workflow.dag_revision + 1,
            "step_executions": _record(
                workflow, index, **details, hung_survey=analysis
            ),
        }
    )


def after_triage(
    workflow: WorkflowRequest, index: int, details: dict[str, Any]
) -> WorkflowRequest:
    steps = list(workflow.official_steps)
    survey = next(
        (
            item
            for item in reversed(workflow.step_executions)
            if item.operation is Op.COLLECT_HUNG_SUMMARY
        ),
        None,
    )
    if survey is None:
        return workflow
    analysis = survey.details.get("hung_survey") or {}
    node_results: dict[str, Any] = {}
    failed: set[str] = set()
    for item in workflow.step_executions:
        if item.operation is Op.COLLECT_HUNG_TRIAGE and item.step_index <= index:
            node_results.update(item.details.get("node_results") or {})
            failed.update(item.details.get("undetermined_nodes") or [])
    node_results.update(details.get("node_results") or {})
    decoded = {}
    decoded_bytes = 0
    for node, value in node_results.items():
        try:
            document = decode_summary(value)
            if document.get("status") in {
                "NODE_COLLECTION_BUSY",
                "TASK_COLLECTION_RATE_LIMIT",
                "COLLECTION_BUDGET_UNAVAILABLE",
            }:
                failed.add(node)
            decoded_bytes += len(json.dumps(document, separators=(",", ":")).encode())
            if decoded_bytes > MAX_SURVEY_RAW_BYTES:
                raise ValueError("combined active evidence exceeds decoded budget")
            decoded[node] = document
        except ValueError:
            failed.add(node)
    node_results = decoded
    signals = [
        {**row, "node_id": node}
        for node, value in node_results.items()
        for row in value.get("ranks", [])
        if isinstance(row, dict)
    ]
    decision = classify_hung_signals(signals, undetermined_nodes=sorted(failed))
    all_nodes = next(
        item.node_ids for item in steps if item.operation is Op.COLLECT_HUNG_SUMMARY
    )
    sampled = sorted(
        {
            node
            for item in steps[: index + 1]
            if item.operation is Op.COLLECT_HUNG_TRIAGE
            for node in item.node_ids
        }
    )
    decision["not_sampled_nodes"] = sorted(set(all_nodes) - set(sampled))
    decision["active_requested_node_count"] = len(sampled)
    decision["active_responding_node_count"] = len(node_results)
    decision["active_node_coverage_fraction"] = (
        len(node_results) / len(all_nodes) if all_nodes else 0
    )
    decision["diagnostic_node_budget"] = steps[index].parameters[
        "hung_diagnostic_node_budget"
    ]
    decision["budget_exhausted"] = len(sampled) >= decision["diagnostic_node_budget"]
    decision["survey_coverage"] = {
        key: analysis.get(key)
        for key in (
            "expected_rank_count",
            "valid_rank_count",
            "rank_inventory_complete",
            "communication_progress_rank_count",
            "cross_node_time_comparable",
        )
    }
    if steps[index].parameters.get("hung_wave") == 1:
        expansion_index = next(
            i
            for i, step in enumerate(steps)
            if step.operation is Op.COLLECT_HUNG_TRIAGE
            and step.parameters.get("hung_wave") == 2
        )
        remaining = max(
            0,
            int(steps[index].parameters["hung_diagnostic_node_budget"]) - len(sampled),
        )
        rank_nodes = analysis.get("rank_node_map", {})
        # Small FR groups can identify unsampled peers even when their passive
        # communication files were not enabled. These are leads, not culprits.
        peer_nodes = list(
            dict.fromkeys(
                rank_nodes[str(rank)]
                for group in decision.get("process_groups", [])
                for rank in group.get("expected_members", [])
                if str(rank) in rank_nodes
            )
        )
        priorities = list(
            dict.fromkeys(
                [
                    *analysis.get("candidate_nodes", []),
                    *peer_nodes,
                    *analysis.get("diagnostic_lead_nodes", []),
                    *analysis.get("related_nodes", []),
                    *analysis.get("node_priority", []),
                ]
            )
        )
        extra = balanced_nodes(
            [node for node in priorities if node not in sampled],
            steps[index].parameters.get("fabric_partition_by_node", {}),
            remaining,
            [
                node
                for node in [
                    *analysis.get("candidate_nodes", []),
                    *peer_nodes,
                    *analysis.get("diagnostic_lead_nodes", []),
                ]
                if node not in sampled
            ],
        )
        reason = (
            "SUFFICIENT_LOCAL_EVIDENCE"
            if decision["targeted_capture_eligible"]
            else steps[index].parameters.get("hung_skip_reason")
            or ("DIAGNOSTIC_BUDGET_EXHAUSTED" if not extra else None)
        )
        steps[expansion_index] = _target(
            steps[expansion_index], [] if reason else extra, all_nodes, reason
        )
    else:
        bundle_index = next(
            i
            for i, step in enumerate(steps)
            if step.operation is Op.COLLECT_DIAGNOSTIC_BUNDLE
            and index in step.depends_on_step_indexes
        )
        steps[bundle_index] = bind_hung_bundle(
            steps[bundle_index].model_copy(update={"node_ids": all_nodes}),
            decision,
            signals,
            sampled or steps[bundle_index].node_ids,
        )
    return workflow.model_copy(
        update={
            "official_steps": steps,
            "dag_revision": workflow.dag_revision + 1,
            "step_executions": _record(workflow, index, hung_triage_decision=decision),
        }
    )
