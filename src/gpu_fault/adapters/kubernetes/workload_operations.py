from __future__ import annotations

import gzip
import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Callable
from urllib import parse as urllib_parse

from gpu_fault.execution import (
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.regional import RemoteEvidenceCaptureRequest
from gpu_fault.telemetry import (
    EvidenceKind,
)
from gpu_fault.hyperpod import (
    HYPERPOD_JOB_AUTO_RESUME_ANNOTATION,
    HyperPodRecoveryState,
    HyperPodWorkloadRecoveryEvidence,
)
from gpu_fault.models import (
    WorkflowStepStatus,
)


from gpu_fault.adapters.common import (
    ANNOTATION_EXECUTION_EPOCH,
    ANNOTATION_FENCING,
    ANNOTATION_INCIDENT,
    ANNOTATION_OPERATION,
    ANNOTATION_RESTART_BUDGET,
    ANNOTATION_RESTART_COUNT,
    ANNOTATION_STEP_INDEX,
    ANNOTATION_TERMINATION_INCIDENT,
    ANNOTATION_TRAINING_CONTAINER,
    ANNOTATION_WORKFLOW,
    LABEL_ATTEMPT_ID,
)
from gpu_fault.adapters.kubernetes.restart_source_guard import (
    _WorkloadMutation,
    refresh_restart_workloads,
    restart_source_failure,
    workload_lifecycle_identity,
)


class KubernetesWorkloadOperationsMixin:
    # Attributes supplied by the composed concrete implementation.
    _parse_workload: Callable[..., Any]
    _patch_workload: Callable[..., Any]
    _read_workload: Callable[..., Any]

    _annotations: Callable[..., Any]
    _declared_gpu_count: Callable[..., Any]
    _labels: Callable[..., Any]
    _metadata: Callable[..., Any]
    _node_rebindings: Callable[..., Any]
    _operation_already_applied: Callable[..., Any]
    _resource_version: Callable[..., Any]
    _restart_attempt_id: Callable[..., Any]
    _restart_custom_spec: Callable[..., Any]
    _restart_guard: Callable[..., Any]
    _restart_terminal_custom_workload: Callable[..., Any]
    _restart_terminal_job: Callable[..., Any]
    _retry_job_name: Callable[..., Any]
    _serialize_workload: Callable[..., Any]
    _suspend_spec: Callable[..., Any]
    _workload_active: Callable[..., Any]
    _workload_terminal: Callable[..., Any]
    alert_sender: Callable[..., Any]
    core: Any
    evidence_service: Any
    evidence_sink: Any
    notification_sink: Any
    restart_email_builder: Any
    store: Any
    workload_log_max_bytes: Any
    workload_log_s3_max_bytes: Any
    workload_log_s3_uri: Any
    workload_log_tail_lines: Any
    workload_log_uploader: Callable[..., Any]

    def _managed_job_recovery_conflict(
        self,
        workloads: list[tuple[str, str, str, str, Any]],
    ) -> WorkflowStepOutcome | None:
        """Refuse to mutate a workload HyperPod is also recovering.

        The plan owns workload recovery outright, so
        ``enable-job-auto-resume=true`` is a non-compliant configuration
        rather than a delegation signal. Acting anyway would make this
        adapter a second writer against the managed control loop, which
        the plan cannot observe or serialise against.

        There is deliberately no per-workload override annotation: the
        only party able to set one is the same party that set the
        offending annotation, so an in-object escape hatch would not be
        a guard at all. Remediation is to remove the annotation.
        """
        enabled = []
        for (
            namespace,
            kind,
            name,
            workload_id,
            workload,
        ) in workloads:
            evidence = HyperPodWorkloadRecoveryEvidence.from_eks_annotations(
                self._annotations(workload),
                workload_id=workload_id,
            )
            if evidence.state is not HyperPodRecoveryState.ENABLED:
                continue
            enabled.append((namespace, kind, name, workload_id))
        if not enabled:
            return None
        workload_ids = sorted(item[-1] for item in enabled)
        return WorkflowStepOutcome.failed(
            "workload recovery is owned by this system but "
            "sagemaker.amazonaws.com/enable-job-auto-resume is enabled "
            f"on {', '.join(workload_ids)}; HyperPod job auto-resume "
            "must be disabled before this system mutates the workload",
            details={
                "managed_job_recovery_workloads": workload_ids,
                "required_annotation": HYPERPOD_JOB_AUTO_RESUME_ANNOTATION,
                "required_annotation_value": "absent or false",
                "remediation_commands": [
                    f"kubectl annotate {kind} {name} -n {namespace} "
                    f"{HYPERPOD_JOB_AUTO_RESUME_ANNOTATION}-"
                    for namespace, kind, name, _ in sorted(
                        enabled, key=lambda item: item[-1]
                    )
                ],
            },
        )

    def _set_workloads(
        self,
        context: WorkflowStepContext,
        *,
        suspend: bool,
    ) -> WorkflowStepOutcome:
        prepared = self._prepare_workload_mutation(context, suspend)
        if isinstance(prepared, WorkflowStepOutcome):
            return prepared
        if suspend and not prepared.workloads:
            return WorkflowStepOutcome.succeeded(
                operation_id=context.idempotency_key,
                details={
                    "workloads": context.step.workload_ids,
                    "suspended": True,
                    "already_absent_workloads": (prepared.absent_workload_ids),
                },
            )
        try:
            self._apply_workload_mutation(context, prepared, suspend)
        except Exception as exc:
            if not suspend and getattr(exc, "status", None) in {404, 410}:
                return restart_source_failure(
                    "RESTART_SOURCE_WORKLOAD_NOT_FOUND_DURING_MUTATION",
                    "restart source workload disappeared during mutation",
                    list(context.step.workload_ids),
                    kubernetes_status=getattr(exc, "status", None),
                )
            raise
        if suspend:
            self._delete_terminating_pods(prepared.terminating_pods)
            outcome = self._suspend_outcome(context, prepared)
            if outcome is not None:
                return outcome
        notification_id, notification_context = self._restart_notification(
            context, prepared, suspend
        )
        restarted_workload_ids = list(
            dict.fromkeys(
                [
                    *prepared.retry_workload_ids,
                    *prepared.created_retry_ids,
                ]
            )
        )
        return WorkflowStepOutcome.succeeded(
            operation_id=context.idempotency_key,
            details={
                "workloads": context.step.workload_ids,
                "suspended": suspend,
                **(
                    {"restart_attempt_id": prepared.restart_attempt_id}
                    if prepared.restart_attempt_id is not None
                    else {}
                ),
                **(
                    {"notification_id": notification_id}
                    if notification_id is not None
                    else {}
                ),
                **(
                    {"notification_context": notification_context}
                    if notification_context is not None
                    else {}
                ),
                **(
                    {"restarted_workload_ids": restarted_workload_ids}
                    if restarted_workload_ids
                    else {}
                ),
            },
        )

    def _prepare_workload_mutation(
        self,
        context: WorkflowStepContext,
        suspend: bool,
    ) -> _WorkloadMutation | WorkflowStepOutcome:
        if not context.step.workload_ids:
            return WorkflowStepOutcome.failed(
                "workload operation requires workload_ids"
            )
        parsed_workloads = [
            (
                *self._parse_workload(workload_id),
                workload_id,
            )
            for workload_id in context.step.workload_ids
        ]
        workloads = []
        absent_workload_ids = []
        source_workload_uids = {}
        source_resource_versions = {}
        deleting_workload_ids = []
        for namespace, kind, name, workload_id in parsed_workloads:
            try:
                workload = self._read_workload(namespace, kind, name)
            except Exception as exc:
                if getattr(exc, "status", None) in {404, 410}:
                    absent_workload_ids.append(workload_id)
                    continue
                raise
            uid, resource_version, deleting = workload_lifecycle_identity(
                workload,
                metadata_reader=self._metadata,
                resource_version_reader=self._resource_version,
            )
            source_workload_uids[workload_id] = uid
            source_resource_versions[workload_id] = resource_version
            if not suspend and deleting:
                deleting_workload_ids.append(workload_id)
            workloads.append((namespace, kind, name, workload_id, workload))
        if not suspend and absent_workload_ids:
            return restart_source_failure(
                "RESTART_SOURCE_WORKLOAD_NOT_FOUND",
                "restart source workload is missing: "
                + ", ".join(sorted(absent_workload_ids)),
                absent_workload_ids,
            )
        if deleting_workload_ids:
            return restart_source_failure(
                "RESTART_SOURCE_WORKLOAD_DELETING",
                "restart source workload is being deleted: "
                + ", ".join(sorted(deleting_workload_ids)),
                deleting_workload_ids,
            )
        conflict = self._managed_job_recovery_conflict(workloads)
        if conflict is not None:
            return conflict
        state = _WorkloadMutation(
            workloads=workloads,
            parsed=parsed_workloads,
            absent_workload_ids=absent_workload_ids,
            source_workload_uids=source_workload_uids,
            source_resource_versions=source_resource_versions,
        )
        if suspend:
            initiator = context.step.parameters.get("termination_initiator_incident_id")
            if initiator:
                (
                    state.terminating_pods,
                    state.log_evidence,
                    state.log_errors,
                ) = self._mark_terminating_pods(workloads, context)
        if not suspend:
            refresh_failure = refresh_restart_workloads(
                state,
                read_workload=self._read_workload,
                metadata_reader=self._metadata,
                resource_version_reader=self._resource_version,
            )
            if refresh_failure is not None:
                return refresh_failure
            refreshed_conflict = self._managed_job_recovery_conflict(state.workloads)
            if refreshed_conflict is not None:
                return refreshed_conflict
            guard, state.restart_count = self._restart_guard(context, state.workloads)
            if guard is not None:
                return guard
            state.restart_attempt_id = self._restart_attempt_id(
                str(context.step.parameters["source_attempt_id"]),
                context.idempotency_key,
            )
            declared_gpu_counts = [
                self._declared_gpu_count(kind, workload)
                for _, kind, _, _, workload in state.workloads
            ]
            if all(count is not None for count in declared_gpu_counts):
                state.target_gpu_count = sum(declared_gpu_counts)
        state.retry_workload_ids = [
            f"{namespace}/job/{self._retry_job_name(name, context.idempotency_key)}"
            for namespace, kind, name, _ in parsed_workloads
            if kind == "job"
        ]
        return state

    def _apply_workload_mutation(
        self,
        context: WorkflowStepContext,
        state: _WorkloadMutation,
        suspend: bool,
    ) -> None:
        for (
            namespace,
            kind,
            name,
            _,
            workload,
        ) in state.workloads:
            annotations = self._annotations(workload)
            if self._operation_already_applied(annotations, context):
                continue
            annotations = {
                ANNOTATION_INCIDENT: context.incident.incident_id,
                ANNOTATION_FENCING: str(context.workflow.fencing_token),
                ANNOTATION_WORKFLOW: context.workflow.request_id,
                ANNOTATION_OPERATION: context.idempotency_key,
                ANNOTATION_EXECUTION_EPOCH: str(context.workflow.execution_epoch),
                ANNOTATION_STEP_INDEX: str(context.step_index),
            }
            if suspend:
                annotations[ANNOTATION_TERMINATION_INCIDENT] = (
                    context.incident.incident_id
                )
            elif state.restart_count is not None:
                annotations[ANNOTATION_RESTART_BUDGET] = str(
                    context.step.parameters["restart_budget"]
                )
                annotations[ANNOTATION_RESTART_COUNT] = str(state.restart_count)
            metadata = {"annotations": annotations}
            if state.restart_attempt_id is not None:
                metadata["labels"] = {LABEL_ATTEMPT_ID: state.restart_attempt_id}
            resource_version = self._resource_version(workload)
            if resource_version is not None:
                metadata["resourceVersion"] = resource_version
            body = {
                "metadata": metadata,
                "spec": self._suspend_spec(kind, suspend),
            }
            if not suspend and kind != "job":
                body["spec"] = self._restart_custom_spec(
                    kind,
                    workload,
                    state.restart_attempt_id,
                    int(context.step.parameters["restart_budget"]),
                    state.restart_count,
                    self._node_rebindings(context),
                )
            if kind == "job":
                if not suspend:
                    self._restart_terminal_job(
                        namespace,
                        name,
                        workload,
                        context,
                        annotations,
                        state.retry_workload_ids,
                        state.restart_count,
                        state.restart_attempt_id,
                    )
                    continue
                self._patch_workload(namespace, kind, name, body)
            else:
                if not suspend and self._workload_terminal(kind, workload):
                    state.created_retry_ids.append(
                        self._restart_terminal_custom_workload(
                            namespace,
                            kind,
                            name,
                            workload,
                            context,
                            annotations,
                            state.restart_count,
                            state.restart_attempt_id,
                        )
                    )
                    continue
                self._patch_workload(namespace, kind, name, body)

    def _restart_notification(
        self,
        context: WorkflowStepContext,
        state: _WorkloadMutation,
        suspend: bool,
    ) -> tuple[str | None, dict | None]:
        if (
            not suspend
            and state.restart_count is not None
            and state.restart_attempt_id is not None
            and state.target_gpu_count is not None
        ):
            parameters = context.step.parameters
            notification = self.restart_email_builder.build_workload_restarted(
                cluster_id=str(parameters["cluster_id"]),
                incident_id=context.incident.incident_id,
                workflow_id=context.workflow.request_id,
                operation_id=context.idempotency_key,
                job_id=str(parameters["job_id"]),
                source_attempt_id=str(parameters["source_attempt_id"]),
                restart_attempt_id=state.restart_attempt_id,
                workload_ids=context.step.workload_ids,
                source_gpu_count=int(parameters["source_gpu_count"]),
                target_gpu_count=state.target_gpu_count,
                restart_count=state.restart_count,
                restart_budget=int(parameters["restart_budget"]),
            )
            notification_context = {
                "job_id": str(parameters["job_id"]),
                "source_attempt_id": str(parameters["source_attempt_id"]),
                "restart_attempt_id": state.restart_attempt_id,
                "source_gpu_count": int(parameters["source_gpu_count"]),
                "target_gpu_count": state.target_gpu_count,
                "restart_count": state.restart_count,
                "restart_budget": int(parameters["restart_budget"]),
            }
            if self.store is not None:
                notification = self.notification_sink.save_notification_if_absent(
                    notification
                )
                notification_id = notification.notification_id
                if self.alert_sender is not None:
                    self.alert_sender(notification_id)
                return notification_id, notification_context
            return None, notification_context
        return None, None

    def _suspend_outcome(
        self,
        context: WorkflowStepContext,
        state: _WorkloadMutation,
    ) -> WorkflowStepOutcome | None:
        previous_waiting = any(
            item.step_index == context.step_index
            and item.status is WorkflowStepStatus.WAITING
            for item in context.workflow.step_executions
        )
        if not previous_waiting:
            return WorkflowStepOutcome.waiting(
                operation_id=context.idempotency_key,
                details={
                    "workloads": context.step.workload_ids,
                    "suspended": True,
                    "waiting_for_active_workloads": (context.step.workload_ids),
                    "deleted_pods": [
                        f"{namespace}/{name}"
                        for namespace, name in state.terminating_pods
                    ],
                    "workload_log_evidence": state.log_evidence,
                    "workload_log_errors": state.log_errors,
                    "already_absent_workloads": (state.absent_workload_ids),
                },
            )
        if (
            context.step.parameters.get("termination_initiator_incident_id")
            and not state.terminating_pods
        ):
            return WorkflowStepOutcome.succeeded(
                operation_id=context.idempotency_key,
                details={
                    "workloads": context.step.workload_ids,
                    "suspended": True,
                    "attempt_pods_absent": True,
                    "workload_log_evidence": state.log_evidence,
                    "workload_log_errors": state.log_errors,
                    "already_absent_workloads": (state.absent_workload_ids),
                },
            )
        active = []
        unknown = []
        for (
            namespace,
            kind,
            name,
            workload_id,
            workload,
        ) in state.workloads:
            active_state = self._workload_active(
                namespace,
                kind,
                name,
                workload,
            )
            if active_state is True:
                active.append(workload_id)
            elif active_state is None:
                unknown.append(workload_id)
        if active or unknown:
            return WorkflowStepOutcome.waiting(
                operation_id=context.idempotency_key,
                details={
                    "workloads": context.step.workload_ids,
                    "suspended": True,
                    "waiting_for_active_workloads": active,
                    "unknown_stop_state": unknown,
                    "already_absent_workloads": (state.absent_workload_ids),
                },
            )
        return None

    def _delete_terminating_pods(
        self,
        pods: list[tuple[str, str]],
    ) -> None:
        for namespace, pod_name in pods:
            try:
                self.core.delete_namespaced_pod(
                    pod_name,
                    namespace,
                    grace_period_seconds=0,
                )
            except Exception as exc:
                if getattr(exc, "status", None) != 404:
                    raise

    def _mark_terminating_pods(
        self,
        workloads: list[tuple[str, str, str, str, Any]],
        context: WorkflowStepContext,
    ) -> tuple[
        list[tuple[str, str]],
        list[dict[str, Any]],
        list[dict[str, str]],
    ]:
        incident_id = str(
            context.step.parameters.get("termination_initiator_incident_id")
            or context.incident.incident_id
        )
        attempts = {
            (
                namespace,
                self._labels(workload).get(LABEL_ATTEMPT_ID),
            )
            for namespace, _, _, _, workload in workloads
        }
        terminating_pods = []
        evidence = []
        errors = []
        for namespace, attempt_id in attempts:
            if not attempt_id:
                raise ValueError(
                    "workload requires an attempt ID before "
                    "incident-initiated termination"
                )
            response = self.core.list_namespaced_pod(
                namespace,
                label_selector=(
                    f"gpu-fault.io/managed=true,{LABEL_ATTEMPT_ID}={attempt_id}"
                ),
            )
            pods = (
                response.get("items", [])
                if isinstance(response, dict)
                else response.items
            )
            for pod in pods:
                metadata = self._metadata(pod)
                name = (
                    metadata.get("name")
                    if isinstance(metadata, dict)
                    else metadata.name
                )
                try:
                    evidence.append(
                        self._capture_workload_log(
                            pod,
                            namespace=namespace,
                            context=context,
                            attempt_id=str(attempt_id),
                        )
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "pod": f"{namespace}/{name}",
                            "error": str(exc),
                        }
                    )
                self.core.patch_namespaced_pod(
                    name,
                    namespace,
                    {
                        "metadata": {
                            "annotations": {
                                ANNOTATION_TERMINATION_INCIDENT: (incident_id)
                            }
                        }
                    },
                )
                terminating_pods.append((namespace, name))
        return (
            sorted(set(terminating_pods)),
            evidence,
            errors,
        )

    def _capture_workload_log(
        self,
        pod: Any,
        *,
        namespace: str,
        context: WorkflowStepContext,
        attempt_id: str,
    ) -> dict[str, Any]:
        serialized = self._serialize_workload(pod)
        metadata = serialized.get("metadata") or {}
        spec = serialized.get("spec") or {}
        pod_name = str(metadata.get("name") or "")
        pod_uid = str(metadata.get("uid") or pod_name)
        if not pod_name:
            raise ValueError("cannot capture workload log without Pod name")
        annotations = metadata.get("annotations") or {}
        container_name = annotations.get(ANNOTATION_TRAINING_CONTAINER)
        containers = spec.get("containers") or []
        if not container_name:
            if len(containers) != 1:
                raise ValueError(
                    f"Pod {namespace}/{pod_name} must identify its "
                    "training container before log capture"
                )
            container_name = containers[0].get("name")
        if not container_name:
            raise ValueError(f"Pod {namespace}/{pod_name} has no training container")

        captured_at = datetime.now(timezone.utc)
        if self.workload_log_s3_uri:
            value = self.core.read_namespaced_pod_log(
                pod_name,
                namespace,
                container=container_name,
                timestamps=True,
                limit_bytes=self.workload_log_s3_max_bytes,
            )
            raw = (
                value
                if isinstance(value, bytes)
                else str(value).encode("utf-8", errors="replace")
            )
            archive_truncated = len(raw) >= self.workload_log_s3_max_bytes
        else:
            value = self.core.read_namespaced_pod_log(
                pod_name,
                namespace,
                container=container_name,
                timestamps=True,
                tail_lines=self.workload_log_tail_lines,
            )
            raw = (
                value
                if isinstance(value, bytes)
                else str(value).encode("utf-8", errors="replace")
            )
            archive_truncated = len(raw.splitlines()) >= self.workload_log_tail_lines

        tail = raw[-self.workload_log_max_bytes :]
        tail_truncated = len(raw) > len(tail) or archive_truncated
        digest = hashlib.sha256(raw).hexdigest()
        s3_uri = None
        if self.workload_log_s3_uri:
            s3_uri = self._upload_workload_log(
                data=raw,
                cluster_id=context.incident.cluster_id,
                incident_id=context.incident.incident_id,
                workflow_id=context.workflow.request_id,
                pod_uid=pod_uid,
                container_name=str(container_name),
                captured_at=captured_at,
            )

        record_hash = hashlib.sha256(
            (f"{context.idempotency_key}\x1f{pod_uid}\x1f{container_name}").encode()
        ).hexdigest()[:24]
        record_id = f"workload-log/{record_hash}"
        node_id = str(spec.get("nodeName") or "UNKNOWN")
        payload = {
            "incident_id": context.incident.incident_id,
            "workflow_id": context.workflow.request_id,
            "step_index": context.step_index,
            "workload_ids": context.step.workload_ids,
            "attempt_id": attempt_id,
            "pod_name": pod_name,
            "pod_uid": pod_uid,
            "container_name": container_name,
            "captured_at": captured_at.isoformat(),
            "sha256": digest,
            "tail": tail.decode("utf-8", errors="replace"),
            "tail_bytes": len(tail),
            "tail_lines": self.workload_log_tail_lines,
            "truncated": tail_truncated,
            "s3_uri": s3_uri,
            "archive_truncated": archive_truncated,
        }
        request = RemoteEvidenceCaptureRequest(
            cluster_id=context.incident.cluster_id,
            record_id=record_id,
            node_id=node_id,
            kind=EvidenceKind.WORKLOAD_LOG,
            observed_at=captured_at,
            attempt_ids=[attempt_id],
            payload=payload,
        )
        if self.evidence_sink is not None:
            record = self.evidence_sink.capture_evidence(request)
        elif self.evidence_service is not None:
            record = self.evidence_service.capture(
                record_id=request.record_id,
                cluster_id=request.cluster_id,
                node_id=request.node_id,
                kind=request.kind,
                observed_at=request.observed_at,
                attempt_ids=request.attempt_ids,
                payload=request.payload,
            )
        else:
            raise RuntimeError("workload log evidence sink is not configured")
        return {
            "record_id": record.record_id,
            "pod": f"{namespace}/{pod_name}",
            "container": container_name,
            "sha256": digest,
            "tail_bytes": len(tail),
            "truncated": tail_truncated,
            "s3_uri": s3_uri,
        }

    def _upload_workload_log(
        self,
        *,
        data: bytes,
        cluster_id: str,
        incident_id: str,
        workflow_id: str,
        pod_uid: str,
        container_name: str,
        captured_at: datetime,
    ) -> str:
        if self.workload_log_s3_uri is None:
            raise RuntimeError("workload log S3 URI is not configured")
        parsed = urllib_parse.urlparse(self.workload_log_s3_uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError("GPU_FAULT_WORKLOAD_LOG_S3_URI must be an s3:// URI")
        safe_container = re.sub(r"[^A-Za-z0-9_.-]+", "-", container_name)
        suffix = (
            f"{cluster_id}/{incident_id}/{workflow_id}/{pod_uid}/"
            f"{safe_container}-{captured_at.strftime('%Y%m%dT%H%M%SZ')}"
            ".log.gz"
        )
        prefix = parsed.path.strip("/")
        key = f"{prefix}/{suffix}" if prefix else suffix
        destination = f"s3://{parsed.netloc}/{key}"
        compressed = gzip.compress(data)
        if self.workload_log_uploader is not None:
            return str(self.workload_log_uploader(destination, compressed))
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for workload log S3 archival"
            ) from exc
        boto3.client("s3").put_object(
            Bucket=parsed.netloc,
            Key=key,
            Body=compressed,
            ContentType="text/plain",
            ContentEncoding="gzip",
            Metadata={"sha256": hashlib.sha256(data).hexdigest()},
        )
        return destination
