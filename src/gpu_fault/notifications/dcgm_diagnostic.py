from gpu_fault.notifications.common import (
    DCGM_DIAGNOSTIC_EMAIL_TEMPLATE,
    DCGM_DIAGNOSTIC_TEMPLATE_VERSION,
    AdvisoryNotification,
    Any,
)


class DcgmDiagnosticEmailBuilder:
    """Renders fixed DCGM result and remediation guidance."""

    def build(
        self,
        *,
        cluster_id: str,
        incident_id: str,
        workflow_id: str,
        event_id: str,
        operation_id: str,
        workload_ids: list[str],
        node_results: dict[str, dict[str, Any]],
        control_plane_action: str,
    ) -> AdvisoryNotification:
        result_lines = []
        guidance_lines = []
        evidence_refs = []
        outcomes = []
        for node_id in sorted(node_results):
            result = node_results[node_id]
            outcome = str(result.get("diagnostic_outcome", "INCONCLUSIVE"))
            outcomes.append(outcome)
            findings = result.get("diagnostic_findings", [])
            if not isinstance(findings, list) or not findings:
                result_lines.append(
                    f"- Node：{node_id}；Result：{outcome}；Checks：NONE"
                )
            else:
                for finding in findings:
                    if not isinstance(finding, dict):
                        continue
                    result_lines.append(
                        "- Node：{node}；Test：{test}；Status："
                        "{status}；Entity：{entities}；Error code："
                        "{codes}；Message：{messages}".format(
                            node=node_id,
                            test=finding.get("test_name", "UNKNOWN"),
                            status=finding.get("status", "UNKNOWN"),
                            entities=", ".join(
                                str(item) for item in finding.get("entities", [])
                            )
                            or "NONE",
                            codes=", ".join(
                                str(item) for item in finding.get("error_codes", [])
                            )
                            or "NONE",
                            messages="; ".join(
                                str(item) for item in finding.get("messages", [])
                            )
                            or "NONE",
                        )
                    )
            for action in result.get("recommended_actions", []):
                if not isinstance(action, dict):
                    continue
                guidance_lines.append(
                    "- Node：{node}；Priority：{priority}；"
                    "Action code：{code}；Trigger：{triggers}；"
                    "Instruction：{instruction}".format(
                        node=node_id,
                        priority=action.get("priority", "UNKNOWN"),
                        code=action.get("action_code", "UNKNOWN"),
                        triggers=", ".join(
                            str(item) for item in action.get("trigger_tests", [])
                        )
                        or "NONE",
                        instruction=action.get("instruction", "NONE"),
                    )
                )
            evidence_ref = result.get("evidence_ref")
            if isinstance(evidence_ref, str) and evidence_ref:
                evidence_refs.append(evidence_ref)
        outcome_rank = {
            "PASS": 0,
            "WARN": 1,
            "INCONCLUSIVE": 2,
            "FAIL": 3,
        }
        aggregate_outcome = max(
            outcomes or ["INCONCLUSIVE"],
            key=lambda item: outcome_rank.get(item, 2),
        )
        body = DCGM_DIAGNOSTIC_EMAIL_TEMPLATE.format(
            cluster_id=cluster_id,
            incident_id=incident_id,
            workflow_id=workflow_id,
            event_id=event_id,
            workloads=", ".join(workload_ids) or "NONE",
            node_results="\n".join(result_lines) or "- NONE",
            control_plane_action=control_plane_action,
            recommended_actions=(
                "\n".join(guidance_lines)
                or "- 无额外人工处置；继续执行冷却观察和 GPU validation。"
            ),
            evidence_refs=(
                "\n".join(f"- {evidence_ref}" for evidence_ref in evidence_refs)
                or "- NONE"
            ),
            template_version=DCGM_DIAGNOSTIC_TEMPLATE_VERSION,
        )
        return AdvisoryNotification(
            deduplication_key=(
                f"{cluster_id}/{incident_id}/dcgm-diagnostic/{operation_id}"
            ),
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=(
                "[GPU故障][DCGM诊断结果 "
                f"{aggregate_outcome}] {cluster_id}: "
                f"{', '.join(sorted(node_results))}"
            ),
            body_text=body,
            support_case_draft="",
            evidence_refs=list(dict.fromkeys(evidence_refs)),
        )
