from gpu_fault.notifications.common import (
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
    AdvisoryNotification,
    Any,
)


class RestartGuardEmailBuilder:
    """Renders an administrator decision request for restart guards."""

    @staticmethod
    def _approval_commands(
        cluster_id: str,
        workload_ids: list[str],
        approval_annotation: str,
    ) -> list[str]:
        commands = []
        context_guard = (
            "${GPU_FAULT_KUBE_CONTEXT:"
            f"?set GPU_FAULT_KUBE_CONTEXT for cluster {cluster_id}"
            "}"
        )
        for workload_id in workload_ids:
            parts = workload_id.split("/", 2)
            if len(parts) != 3 or not all(parts):
                continue
            namespace, kind, name = parts
            commands.append(
                f'kubectl --context "{context_guard}" '
                f"-n {namespace} annotate {kind} {name} "
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
            self._approval_commands(
                cluster_id,
                workload_ids,
                approval_annotation,
            )
            if approval_annotation
            else []
        )
        approval = (
            "\n".join(f"   {command}" for command in approval_commands)
            if approval_commands
            else (
                "   No approval command can be generated: the source or target GPU "
                "count is unknown or the workload identity is incomplete. Fix the "
                "resource metadata and resubmit the job manually."
            )
        )
        body = GPU_COUNT_CHANGE_EMAIL_TEMPLATE.format(
            job_id=job_id,
            attempt_id=attempt_id,
            cluster_id=cluster_id,
            workloads=", ".join(workload_ids) or "UNKNOWN",
            source_gpu_count=source,
            target_gpu_count=target,
            approval_commands=approval,
            approval_annotation=approval_annotation or "UNAVAILABLE",
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
                f"[ACTION REQUIRED][GPU training restart paused] {job_id}: "
                f"{source} GPU -> {target} GPU"
            ),
            body_text=body,
            support_case_draft=(
                "An administrator must either restore the original GPU resources or "
                "adjust the parallel strategy and training parameters and then "
                "approve the new GPU count."
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
                f"[ACTION REQUIRED][GPU training automatic restarts stopped] {job_id}: "
                f"limit of {restart_budget} reached"
            ),
            body_text=body,
            support_case_draft=(
                "Automatic restarts are exhausted; an administrator must investigate "
                "the repeated failures and decide whether to resubmit under a new "
                "job ID."
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
            workloads=", ".join(workload_ids) or "UNKNOWN",
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
                f"[NOTICE][GPU training workload restarted automatically] {job_id}: "
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
                "- Node: {node}; InstanceId: {instance}; "
                "NodeLogicalId: {logical}; boot ID before restart: {old_boot}; "
                "boot ID after restart: {new_boot}; "
                "Agent incarnation: {incarnation}; "
                "HyperPod status: {status}".format(
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
            subject=(
                f"[NOTICE][GPU node restarted automatically] {cluster_id}: "
                f"{', '.join(node_ids)}"
            ),
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
                "- Node: {node}; MainPID before restart: {previous}; "
                "MainPID after restart: {current}; service state: {status}".format(
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
            workloads=", ".join(workload_ids) or "NONE",
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
                "[NOTICE][GPU Fabric Manager restarted automatically] "
                f"{cluster_id}: {', '.join(sorted(node_results))}"
            ),
            body_text=body,
            support_case_draft="",
            category="ACTION_COMPLETED",
            priority=10,
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
                "- Node: {node}; Scope: {scope}; GPU UUID: {gpus}; "
                "no-client check: {clients}; inventory after reset: {after}".format(
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
            workloads=", ".join(workload_ids) or "NONE",
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
                "[NOTICE][GPU/NVSwitch reset automatically] "
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
                    f"- Node: {node_id}; Status: {status}; "
                    f"GPU UUID: {', '.join(map(str, reset_gpus)) or 'UNKNOWN'}"
                )
            else:
                result_lines.append(f"- Node: {node_id}; Result: {result}")
        body = GPU_RESET_EMAIL_TEMPLATE.format(
            cluster_id=cluster_id,
            incident_id=incident_id,
            workflow_id=workflow_id,
            event_id=event_id,
            event_type=event_type,
            policy_source=policy_source,
            official_action=official_action or "UNKNOWN",
            reasons=("\n".join(f"  - {item}" for item in reasons) or "  - UNKNOWN"),
            workloads=", ".join(workload_ids) or "NONE",
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
                f"[NOTICE][GPU reset automatically] {cluster_id}: "
                f"{', '.join(gpu_uuids) or 'UNKNOWN'}"
            ),
            body_text=body,
            support_case_draft="",
            category="ACTION_COMPLETED",
            priority=10,
        )
