from gpu_fault.notifications.common import (
    AdvisoryNotification,
    Any,
    FABRIC_RESET_EMAIL_TEMPLATE,
    FABRIC_RESET_TEMPLATE_VERSION,
    GPU_COUNT_CHANGE_EMAIL_TEMPLATE,
    GPU_RESET_EMAIL_TEMPLATE,
    GPU_RESET_TEMPLATE_VERSION,
    RESTART_BUDGET_EMAIL_TEMPLATE,
    RESTART_FABRIC_MANAGER_EMAIL_TEMPLATE,
    RESTART_FABRIC_MANAGER_TEMPLATE_VERSION,
    RESTART_GUARD_TEMPLATE_VERSION,
    RESTART_NODE_EMAIL_TEMPLATE,
    RESTART_NODE_TEMPLATE_VERSION,
    RESTART_WORKLOAD_EMAIL_TEMPLATE,
    RESTART_WORKLOAD_TEMPLATE_VERSION,
)


class RestartGuardEmailBuilder:
    """Renders an administrator decision request for restart guards."""

    @staticmethod
    def _approval_commands(
        workload_ids: list[str], approval_annotation: str
    ) -> list[str]:
        commands = []
        for workload_id in workload_ids:
            parts = workload_id.split("/", 2)
            if len(parts) != 3 or not all(parts):
                continue
            namespace, kind, name = parts
            commands.append(
                f"kubectl -n {namespace} annotate {kind} {name} "
                "gpu-fault.io/approve-gpu-count-change="
                f"'{approval_annotation}' --overwrite"
            )
        return commands

    def build_gpu_count_change(
        self,
        *,
        cluster_id: str,
        incident_id: str,
        job_id: str,
        attempt_id: str,
        workload_ids: list[str],
        source_gpu_count: int,
        target_gpu_count: int | None,
        approval_annotation: str | None,
    ) -> AdvisoryNotification:
        source = str(source_gpu_count) if source_gpu_count > 0 else "unknown"
        target = str(target_gpu_count) if target_gpu_count is not None else "unknown"
        approval_commands = (
            self._approval_commands(workload_ids, approval_annotation)
            if approval_annotation
            else []
        )
        approval = (
            "\n".join(f"   {command}" for command in approval_commands)
            if approval_commands
            else (
                "   无法生成审批命令：源或目标 GPU 数量未知，或 workload "
                "标识不完整。请修正资源元数据后人工重新提交任务。"
            )
        )
        body = GPU_COUNT_CHANGE_EMAIL_TEMPLATE.format(
            job_id=job_id,
            attempt_id=attempt_id,
            cluster_id=cluster_id,
            workloads=", ".join(workload_ids) or "未知",
            source_gpu_count=source,
            target_gpu_count=target,
            approval_commands=approval,
            approval_annotation=approval_annotation or "不可用",
            incident_id=incident_id,
            template_version=RESTART_GUARD_TEMPLATE_VERSION,
        )
        return AdvisoryNotification(
            deduplication_key=(
                f"{cluster_id}/{job_id}/{attempt_id}/gpu-count/{source}/{target}"
            ),
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=(
                f"[需处理][GPU训练重启已暂停] {job_id}: {source} GPU -> {target} GPU"
            ),
            body_text=body,
            support_case_draft=(
                "需要管理员决定恢复原 GPU 资源，或调整并行策略和训练参数后"
                "批准新的 GPU 数量。"
            ),
        )

    def build_budget_exhausted(
        self,
        *,
        cluster_id: str,
        incident_id: str,
        job_id: str,
        attempt_id: str,
        restart_count: int,
        restart_budget: int,
    ) -> AdvisoryNotification:
        body = RESTART_BUDGET_EMAIL_TEMPLATE.format(
            job_id=job_id,
            attempt_id=attempt_id,
            cluster_id=cluster_id,
            restart_count=restart_count,
            restart_budget=restart_budget,
            incident_id=incident_id,
            template_version=RESTART_GUARD_TEMPLATE_VERSION,
        )
        return AdvisoryNotification(
            deduplication_key=(
                f"{cluster_id}/{job_id}/restart-budget-exhausted/{restart_budget}"
            ),
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=(
                f"[需处理][GPU训练停止自动重启] {job_id}: "
                f"已达到 {restart_budget} 次上限"
            ),
            body_text=body,
            support_case_draft=(
                "自动重启次数已耗尽，需要管理员调查重复故障并决定是否以"
                "新的 job ID 重新提交。"
            ),
        )

    def build_workload_restarted(
        self,
        *,
        cluster_id: str,
        incident_id: str,
        workflow_id: str,
        operation_id: str,
        job_id: str,
        source_attempt_id: str,
        restart_attempt_id: str,
        workload_ids: list[str],
        source_gpu_count: int,
        target_gpu_count: int,
        restart_count: int,
        restart_budget: int,
    ) -> AdvisoryNotification:
        body = RESTART_WORKLOAD_EMAIL_TEMPLATE.format(
            cluster_id=cluster_id,
            incident_id=incident_id,
            workflow_id=workflow_id,
            operation_id=operation_id,
            job_id=job_id,
            source_attempt_id=source_attempt_id,
            restart_attempt_id=restart_attempt_id,
            workloads=", ".join(workload_ids) or "未知",
            source_gpu_count=source_gpu_count,
            target_gpu_count=target_gpu_count,
            restart_count=restart_count,
            restart_budget=restart_budget,
            template_version=RESTART_WORKLOAD_TEMPLATE_VERSION,
        )
        return AdvisoryNotification(
            deduplication_key=(
                f"{cluster_id}/{job_id}/workload-restarted/{operation_id}"
            ),
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=(
                f"[通知][GPU训练任务已自动重启] {job_id}: "
                f"{source_attempt_id} -> {restart_attempt_id}"
            ),
            body_text=body,
            support_case_draft="",
            category="ACTION_COMPLETED",
            priority=10,
        )

    def build_node_restarted(
        self,
        *,
        cluster_id: str,
        incident_id: str,
        workflow_id: str,
        event_id: str,
        event_type: str,
        xid: int | None,
        policy_source: str,
        official_action: str | None,
        effective_action: str | None,
        reasons: list[str],
        operation_id: str,
        node_ids: list[str],
        source_boot_id: str | None,
        agent_baselines: dict[str, dict[str, str]],
        agent_observations: list[dict[str, Any]],
        provider_observations: list[dict[str, Any]],
        confirmation_source: str,
        event_source: str | None = None,
    ) -> AdvisoryNotification:
        providers = {item.get("node_id"): item for item in provider_observations}
        lines = []
        for item in agent_observations:
            node_id = item.get("node_id", "UNKNOWN")
            provider = providers.get(node_id, {})
            baseline = agent_baselines.get(node_id, {})
            lines.append(
                "- Node：{node}; InstanceId：{instance}; "
                "NodeLogicalId：{logical}; 重启前 boot ID：{old_boot}; "
                "重启后 boot ID：{new_boot}; "
                "Agent incarnation：{incarnation}; "
                "HyperPod status：{status}".format(
                    node=node_id,
                    instance=provider.get("instance_id", "UNKNOWN"),
                    logical=provider.get("node_logical_id", "UNKNOWN"),
                    old_boot=baseline.get("boot_id", source_boot_id or "UNKNOWN"),
                    new_boot=item.get("boot_id", "UNKNOWN"),
                    incarnation=item.get("agent_incarnation_id", "UNKNOWN"),
                    status=provider.get("status", "UNKNOWN"),
                )
            )
        body = RESTART_NODE_EMAIL_TEMPLATE.format(
            cluster_id=cluster_id,
            incident_id=incident_id,
            workflow_id=workflow_id,
            event_id=event_id,
            event_type=event_type,
            fault_identifier=(f"XID {xid}" if xid is not None else event_id),
            event_source=event_source or "UNKNOWN",
            policy_source=policy_source,
            official_action=official_action or "UNKNOWN",
            effective_action=effective_action or "UNKNOWN",
            reasons=("\n".join(f"  - {item}" for item in reasons) or "  - UNKNOWN"),
            node_ids=", ".join(node_ids) or "UNKNOWN",
            operation_id=operation_id,
            node_observations="\n".join(lines) or "- UNKNOWN",
            confirmation_source=confirmation_source,
            template_version=RESTART_NODE_TEMPLATE_VERSION,
        )
        return AdvisoryNotification(
            deduplication_key=(
                f"{cluster_id}/{incident_id}/node-restarted/{operation_id}"
            ),
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=(f"[通知][GPU节点已自动重启] {cluster_id}: {', '.join(node_ids)}"),
            body_text=body,
            support_case_draft="",
            category="ACTION_COMPLETED",
            priority=10,
        )

    def build_fabric_manager_restarted(
        self,
        *,
        cluster_id: str,
        incident_id: str,
        workflow_id: str,
        event_id: str,
        event_type: str,
        policy_source: str,
        official_action: str | None,
        reasons: list[str],
        operation_id: str,
        node_results: dict[str, dict[str, Any]],
        workload_ids: list[str],
    ) -> AdvisoryNotification:
        result_lines = []
        for node_id in sorted(node_results):
            result = node_results[node_id]
            result_lines.append(
                "- Node：{node}; 重启前 MainPID：{previous}; "
                "重启后 MainPID：{current}; 服务状态：{status}".format(
                    node=node_id,
                    previous=result.get("previous_main_pid", "UNKNOWN"),
                    current=result.get("current_main_pid", "UNKNOWN"),
                    status=("active" if result.get("active") is True else "UNKNOWN"),
                )
            )
        body = RESTART_FABRIC_MANAGER_EMAIL_TEMPLATE.format(
            cluster_id=cluster_id,
            incident_id=incident_id,
            workflow_id=workflow_id,
            event_id=event_id,
            event_type=event_type,
            policy_source=policy_source,
            official_action=official_action or "UNKNOWN",
            reasons=("\n".join(f"  - {item}" for item in reasons) or "  - UNKNOWN"),
            workloads=", ".join(workload_ids) or "无",
            operation_id=operation_id,
            node_results="\n".join(result_lines) or "- UNKNOWN",
            template_version=(RESTART_FABRIC_MANAGER_TEMPLATE_VERSION),
        )
        return AdvisoryNotification(
            deduplication_key=(
                f"{cluster_id}/{incident_id}/fabric-manager-restarted/{operation_id}"
            ),
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=(
                "[通知][GPU Fabric Manager 已自动重启] "
                f"{cluster_id}: {', '.join(sorted(node_results))}"
            ),
            body_text=body,
            support_case_draft="",
        )

    def build_fabric_reset_completed(
        self,
        *,
        cluster_id: str,
        incident_id: str,
        workflow_id: str,
        event_id: str,
        event_type: str,
        policy_source: str,
        official_action: str | None,
        reasons: list[str],
        operation_id: str,
        node_results: dict[str, dict[str, Any]],
        workload_ids: list[str],
        fabric_partition: object,
        sxid: object,
    ) -> AdvisoryNotification:
        if isinstance(sxid, int):
            sxid_text = str(sxid)
        elif isinstance(sxid, dict):
            sxid_text = (
                "; ".join(
                    f"{node_id}={','.join(str(item) for item in values)}"
                    for node_id, values in sorted(sxid.items())
                    if isinstance(values, list)
                )
                or "UNKNOWN"
            )
        else:
            sxid_text = "UNKNOWN"
        if isinstance(fabric_partition, str):
            fabric_partition_text = fabric_partition
        elif isinstance(fabric_partition, dict):
            fabric_partition_text = (
                "; ".join(
                    f"{node_id}={value}"
                    for node_id, value in sorted(fabric_partition.items())
                    if isinstance(value, str)
                )
                or "UNKNOWN"
            )
        else:
            fabric_partition_text = "UNKNOWN"
        result_lines = []
        for node_id in sorted(node_results):
            result = node_results[node_id]
            gpu_uuids = result.get("reset_gpu_uuids", [])
            result_lines.append(
                "- Node：{node}; Scope：{scope}; GPU UUID：{gpus}; "
                "无客户端校验：{clients}; Reset 后 inventory：{after}".format(
                    node=node_id,
                    scope=result.get("reset_scope", "UNKNOWN"),
                    gpus=", ".join(gpu_uuids) or "UNKNOWN",
                    clients=result.get("verified_no_gpu_clients", False),
                    after=result.get("inventory_verified_after", False),
                )
            )
        body = FABRIC_RESET_EMAIL_TEMPLATE.format(
            cluster_id=cluster_id,
            incident_id=incident_id,
            workflow_id=workflow_id,
            event_id=event_id,
            event_type=event_type,
            sxid=sxid_text,
            policy_source=policy_source,
            official_action=official_action or "UNKNOWN",
            fabric_partition=fabric_partition_text,
            reasons=("\n".join(f"  - {item}" for item in reasons) or "  - UNKNOWN"),
            workloads=", ".join(workload_ids) or "无",
            operation_id=operation_id,
            node_results="\n".join(result_lines) or "- UNKNOWN",
            template_version=FABRIC_RESET_TEMPLATE_VERSION,
        )
        return AdvisoryNotification(
            deduplication_key=(
                f"{cluster_id}/{incident_id}/fabric-reset/{operation_id}"
            ),
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=(
                "[通知][GPU/NVSwitch 已自动重置] "
                f"{cluster_id}: {', '.join(sorted(node_results))}"
            ),
            body_text=body,
            support_case_draft="",
        )

    def build_gpu_reset_completed(
        self,
        *,
        cluster_id: str,
        incident_id: str,
        workflow_id: str,
        event_id: str,
        event_type: str,
        policy_source: str,
        official_action: str | None,
        reasons: list[str],
        operation_id: str,
        node_ids: list[str],
        gpu_uuids: list[str],
        node_results: dict[str, object],
        workload_ids: list[str],
    ) -> AdvisoryNotification:
        result_lines = []
        for node_id in sorted(node_results):
            result = node_results[node_id]
            if isinstance(result, dict):
                reset_gpus = result.get(
                    "reset_gpu_uuids",
                    result.get("gpu_uuids", gpu_uuids),
                )
                if not isinstance(reset_gpus, list):
                    reset_gpus = gpu_uuids
                status = result.get(
                    "status",
                    result.get("reset_status", "SUCCEEDED"),
                )
                result_lines.append(
                    f"- Node：{node_id}; Status：{status}; "
                    f"GPU UUID：{', '.join(map(str, reset_gpus)) or 'UNKNOWN'}"
                )
            else:
                result_lines.append(f"- Node：{node_id}; Result：{result}")
        body = GPU_RESET_EMAIL_TEMPLATE.format(
            cluster_id=cluster_id,
            incident_id=incident_id,
            workflow_id=workflow_id,
            event_id=event_id,
            event_type=event_type,
            policy_source=policy_source,
            official_action=official_action or "UNKNOWN",
            reasons=("\n".join(f"  - {item}" for item in reasons) or "  - UNKNOWN"),
            workloads=", ".join(workload_ids) or "无",
            operation_id=operation_id,
            node_ids=", ".join(node_ids) or "UNKNOWN",
            gpu_uuids=", ".join(gpu_uuids) or "UNKNOWN",
            node_results="\n".join(result_lines) or "- UNKNOWN",
            template_version=GPU_RESET_TEMPLATE_VERSION,
        )
        return AdvisoryNotification(
            deduplication_key=(f"{cluster_id}/{incident_id}/gpu-reset/{operation_id}"),
            cluster_name=cluster_id,
            incident_id=incident_id,
            subject=(
                f"[通知][GPU 已自动重置] {cluster_id}: "
                f"{', '.join(gpu_uuids) or 'UNKNOWN'}"
            ),
            body_text=body,
            support_case_draft="",
            category="ACTION_COMPLETED",
            priority=10,
        )
