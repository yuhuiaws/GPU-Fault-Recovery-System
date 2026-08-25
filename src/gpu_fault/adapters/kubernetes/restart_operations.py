from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from typing import Any, Callable

from gpu_fault.execution import (
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowStepStatus,
)


from gpu_fault.adapters.common import (
    ANNOTATION_APPROVE_GPU_COUNT_CHANGE,
    ANNOTATION_RESTART_BUDGET,
    ANNOTATION_RESTART_COUNT,
    ANNOTATION_TARGET_GPU_COUNT,
    LABEL_ATTEMPT_ID,
    LABEL_JOB_ID,
)


class KubernetesRestartOperationsMixin:
    # Attributes supplied by the composed concrete implementation.
    _annotations: Callable[..., Any]
    _labels: Callable[..., Any]
    _read_workload: Callable[..., Any]
    alert_sender: Callable[..., Any]
    batch: Any
    custom: Any
    notification_sink: Any
    restart_email_builder: Any
    store: Any

    def _restart_guard(
        self,
        context: WorkflowStepContext,
        workloads: list[tuple[str, str, str, str, Any]],
    ) -> tuple[WorkflowStepOutcome | None, int | None]:
        parameters = context.step.parameters
        required = {
            "cluster_id",
            "job_id",
            "source_attempt_id",
            "source_gpu_count",
            "restart_budget",
        }
        if not required.issubset(parameters):
            return (
                WorkflowStepOutcome.failed(
                    "restart safety context is missing: "
                    + ", ".join(sorted(required - set(parameters)))
                ),
                None,
            )
        declared_job_ids = {
            value
            for *_, workload in workloads
            if (value := self._labels(workload).get(LABEL_JOB_ID))
        }
        expected_job_id = str(parameters["job_id"])
        if declared_job_ids and declared_job_ids != {expected_job_id}:
            return (
                WorkflowStepOutcome.failed(
                    "restart workload job identity does not match "
                    f"{expected_job_id}: " + ", ".join(sorted(declared_job_ids))
                ),
                None,
            )
        declared_attempt_ids = {
            value
            for *_, workload in workloads
            if (value := self._labels(workload).get(LABEL_ATTEMPT_ID))
        }
        if len(declared_attempt_ids) > 1:
            return (
                WorkflowStepOutcome.failed(
                    "restart workloads have inconsistent attempt IDs: "
                    + ", ".join(sorted(declared_attempt_ids))
                ),
                None,
            )
        if declared_attempt_ids:
            parameters["source_attempt_id"] = next(iter(declared_attempt_ids))

        source_gpu_count = int(parameters["source_gpu_count"])
        if source_gpu_count <= 0:
            observed_count = self._observed_source_gpu_count(
                str(parameters["cluster_id"]),
                str(parameters["job_id"]),
                str(parameters["source_attempt_id"]),
            )
            if observed_count > 0:
                source_gpu_count = observed_count
                parameters["source_gpu_count"] = observed_count
        declared = [
            self._declared_gpu_count(kind, workload)
            for _, kind, _, _, workload in workloads
        ]
        target_gpu_count = (
            sum(declared)
            if declared and all(item is not None for item in declared)
            else None
        )
        approval = (
            f"{source_gpu_count}:{target_gpu_count}"
            if source_gpu_count > 0 and target_gpu_count is not None
            else None
        )
        count_matches = (
            source_gpu_count >= 0
            and target_gpu_count is not None
            and source_gpu_count == target_gpu_count
        )
        approved = bool(approval) and all(
            self._annotations(workload).get(ANNOTATION_APPROVE_GPU_COUNT_CHANGE)
            == approval
            for _, _, _, _, workload in workloads
        )
        if not count_matches and not approved:
            if self.notification_sink is None:
                return (
                    WorkflowStepOutcome.failed(
                        "restart GPU count changed without local "
                        "persistent notification support"
                    ),
                    None,
                )
            notification = self.restart_email_builder.build_gpu_count_change(
                cluster_id=str(parameters["cluster_id"]),
                incident_id=context.incident.incident_id,
                job_id=str(parameters["job_id"]),
                attempt_id=str(parameters["source_attempt_id"]),
                workload_ids=context.step.workload_ids,
                source_gpu_count=source_gpu_count,
                target_gpu_count=target_gpu_count,
                approval_annotation=approval,
            )
            notification = self.notification_sink.save_notification_if_absent(
                notification
            )
            if self.alert_sender is not None:
                self.alert_sender(notification.notification_id)
            return (
                WorkflowStepOutcome.waiting(
                    operation_id=context.idempotency_key,
                    details={
                        "approval_required": True,
                        "reason": "GPU_COUNT_CHANGED",
                        "source_gpu_count": source_gpu_count,
                        "target_gpu_count": target_gpu_count,
                        "required_approval_annotation": approval,
                        "notification_id": (notification.notification_id),
                    },
                ),
                None,
            )

        if self.store is None:
            authorization = context.request.restart_authorization
            if authorization is None:
                return (
                    WorkflowStepOutcome.failed(
                        "restart safety guard requires regional authorization"
                    ),
                    None,
                )
            expected = {
                "cluster_id": str(parameters["cluster_id"]),
                "job_id": str(parameters["job_id"]),
                "source_attempt_id": str(parameters["source_attempt_id"]),
                "source_gpu_count": source_gpu_count,
                "restart_budget": int(parameters["restart_budget"]),
                "reservation_id": context.idempotency_key,
            }
            actual = {
                "cluster_id": authorization.cluster_id,
                "job_id": authorization.job_id,
                "source_attempt_id": (authorization.source_attempt_id),
                "source_gpu_count": (authorization.source_gpu_count),
                "restart_budget": authorization.restart_budget,
                "reservation_id": authorization.reservation_id,
            }
            if actual != expected:
                return (
                    WorkflowStepOutcome.failed(
                        "regional restart authorization does not "
                        "match the workload safety context"
                    ),
                    None,
                )
            return None, authorization.restart_count

        state, reserved = self.store.reserve_job_restart(
            str(parameters["cluster_id"]),
            str(parameters["job_id"]),
            int(parameters["restart_budget"]),
            context.idempotency_key,
        )
        if not reserved:
            notification = self.restart_email_builder.build_budget_exhausted(
                cluster_id=state.cluster_id,
                incident_id=context.incident.incident_id,
                job_id=state.job_id,
                attempt_id=str(parameters["source_attempt_id"]),
                restart_count=state.restart_count,
                restart_budget=state.budget,
            )
            notification = self.store.save_notification_if_absent(notification)
            if self.alert_sender is not None:
                self.alert_sender(notification.notification_id)
            return (
                WorkflowStepOutcome.failed(
                    "restart budget exhausted for "
                    f"{state.cluster_id}/{state.job_id}: "
                    f"{state.restart_count}/{state.budget}",
                    details={
                        "reason": "RESTART_BUDGET_EXHAUSTED",
                        "restart_count": state.restart_count,
                        "restart_budget": state.budget,
                        "notification_id": (notification.notification_id),
                    },
                ),
                None,
            )
        return None, state.restart_count

    def _observed_source_gpu_count(
        self, cluster_id: str, job_id: str, attempt_id: str
    ) -> int:
        if self.store is None:
            # The regional executor runs against remote state and owns
            # no local store, so historical observations are simply not
            # reachable here. Reporting "unknown" lets the caller keep
            # its declared count instead of crashing the step.
            return 0
        gpu_uuids = {
            gpu_uuid
            for observation in self.store.list_attempt_observations(cluster_id)
            if observation.attempt_id == attempt_id
            for container in observation.containers
            for gpu_uuid in container.gpu_uuids
        }
        if gpu_uuids:
            return len(gpu_uuids)
        latest_job_observations = [
            observation
            for observation in self.store.list_attempt_observations(cluster_id)
            if observation.job_id == job_id
        ]
        if latest_job_observations:
            latest_attempt = max(
                latest_job_observations,
                key=lambda item: item.observed_at,
            ).attempt_id
            gpu_uuids = {
                gpu_uuid
                for observation in latest_job_observations
                if observation.attempt_id == latest_attempt
                for container in observation.containers
                for gpu_uuid in container.gpu_uuids
            }
            if gpu_uuids:
                return len(gpu_uuids)
        historical_counts = [
            int(step.parameters["source_gpu_count"])
            for workflow in self.store.list_workflows(limit=10000)
            for step in workflow.official_steps
            if (
                step.operation is WorkflowOperation.RESTART_WORKLOAD
                and step.parameters.get("cluster_id") == cluster_id
                and step.parameters.get("job_id") == job_id
                and step.parameters.get("source_attempt_id") == attempt_id
                and int(step.parameters.get("source_gpu_count", 0)) > 0
            )
        ]
        return max(historical_counts, default=0)

    def _restart_terminal_job(
        self,
        namespace: str,
        name: str,
        workload: Any,
        context: WorkflowStepContext,
        annotations: dict[str, str],
        retry_workload_ids: list[str],
        restart_count: int | None,
        restart_attempt_id: str,
    ) -> None:
        retry_name = self._retry_job_name(name, context.idempotency_key)
        try:
            self.batch.read_namespaced_job(retry_name, namespace)
            return
        except Exception as exc:
            if getattr(exc, "status", 404) != 404:
                raise
        if isinstance(workload, dict):
            body = deepcopy(workload)
        else:
            try:
                from kubernetes import client
            except ImportError as exc:
                raise RuntimeError(
                    "install gpu-fault-control-plane[collectors]"
                ) from exc
            body = client.ApiClient().sanitize_for_serialization(workload)
        body.pop("status", None)
        metadata = body.setdefault("metadata", {})
        for key in {
            "creationTimestamp",
            "deletionTimestamp",
            "generation",
            "managedFields",
            "namespace",
            "resourceVersion",
            "selfLink",
            "uid",
        }:
            metadata.pop(key, None)
        metadata["name"] = retry_name
        metadata.pop("generateName", None)
        metadata["annotations"] = {
            **(metadata.get("annotations") or {}),
            **annotations,
        }
        metadata.setdefault("labels", {})[LABEL_ATTEMPT_ID] = restart_attempt_id
        spec = body.setdefault("spec", {})
        spec["suspend"] = False
        spec.pop("selector", None)
        spec.pop("manualSelector", None)
        template_spec = spec.setdefault("template", {}).setdefault("spec", {})
        rebindings = self._node_rebindings(context)
        node_name = template_spec.get("nodeName")
        if node_name in rebindings:
            template_spec["nodeName"] = rebindings[node_name]
        node_selector = template_spec.get("nodeSelector")
        if isinstance(node_selector, dict):
            hostname = node_selector.get("kubernetes.io/hostname")
            if hostname in rebindings:
                node_selector["kubernetes.io/hostname"] = rebindings[hostname]
        template_metadata = spec.setdefault("template", {}).setdefault("metadata", {})
        labels = template_metadata.setdefault("labels", {})
        for key in {
            "batch.kubernetes.io/controller-uid",
            "batch.kubernetes.io/job-name",
            "controller-uid",
            "job-name",
        }:
            labels.pop(key, None)
        labels[LABEL_ATTEMPT_ID] = restart_attempt_id
        pod_annotations = template_metadata.setdefault("annotations", {})
        pod_annotations["gpu-fault.io/workload-ids"] = json.dumps(retry_workload_ids)
        if restart_count is not None:
            pod_annotations[ANNOTATION_RESTART_BUDGET] = str(
                context.step.parameters["restart_budget"]
            )
            pod_annotations[ANNOTATION_RESTART_COUNT] = str(restart_count)
        self.batch.create_namespaced_job(namespace, body)

    def _restart_terminal_custom_workload(
        self,
        namespace: str,
        kind: str,
        name: str,
        workload: Any,
        context: WorkflowStepContext,
        annotations: dict[str, str],
        restart_count: int | None,
        restart_attempt_id: str,
    ) -> str:
        group, version, plural = {
            "pytorchjob": (
                "kubeflow.org",
                "v1",
                "pytorchjobs",
            ),
            "jobset": (
                "jobset.x-k8s.io",
                "v1alpha2",
                "jobsets",
            ),
        }[kind]
        retry_name = self._retry_job_name(name, context.idempotency_key)
        retry_workload_id = f"{namespace}/{kind}/{retry_name}"
        try:
            self.custom.get_namespaced_custom_object(
                group,
                version,
                namespace,
                plural,
                retry_name,
            )
            return retry_workload_id
        except Exception as exc:
            if getattr(exc, "status", 404) != 404:
                raise

        body = deepcopy(self._serialize_workload(workload))
        body.pop("status", None)
        metadata = body.setdefault("metadata", {})
        for key in {
            "creationTimestamp",
            "deletionTimestamp",
            "generation",
            "managedFields",
            "namespace",
            "resourceVersion",
            "selfLink",
            "uid",
        }:
            metadata.pop(key, None)
        metadata["name"] = retry_name
        metadata.pop("generateName", None)
        metadata["annotations"] = {
            **(metadata.get("annotations") or {}),
            **annotations,
        }
        metadata.setdefault("labels", {})[LABEL_ATTEMPT_ID] = restart_attempt_id
        patch = self._restart_custom_spec(
            kind,
            workload,
            restart_attempt_id,
            int(context.step.parameters["restart_budget"]),
            restart_count,
            self._node_rebindings(context),
            workload_ids=[retry_workload_id],
        )
        spec = body.setdefault("spec", {})
        if kind == "pytorchjob":
            spec.setdefault("runPolicy", {}).update(patch["runPolicy"])
            replicas = spec.setdefault("pytorchReplicaSpecs", {})
            for role, role_patch in patch["pytorchReplicaSpecs"].items():
                replicas.setdefault(role, {}).update(role_patch)
        else:
            spec.update(patch)
        self.custom.create_namespaced_custom_object(
            group,
            version,
            namespace,
            plural,
            body,
        )
        return retry_workload_id

    @classmethod
    def _restart_attempt_id(cls, source_attempt_id: str, idempotency_key: str) -> str:
        return f"{source_attempt_id[:52]}-r-{cls._retry_suffix(idempotency_key)}"

    @classmethod
    def _restart_custom_spec(
        cls,
        kind: str,
        workload: Any,
        restart_attempt_id: str,
        restart_budget: int,
        restart_count: int | None,
        node_rebindings: dict[str, str] | None = None,
        workload_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        annotations = {
            ANNOTATION_RESTART_BUDGET: str(restart_budget),
        }
        if restart_count is not None:
            annotations[ANNOTATION_RESTART_COUNT] = str(restart_count)

        def update_template(
            template: dict[str, Any],
        ) -> None:
            metadata = template.setdefault("metadata", {})
            metadata.setdefault("labels", {})[LABEL_ATTEMPT_ID] = restart_attempt_id
            metadata.setdefault("annotations", {}).update(annotations)
            if workload_ids:
                metadata["annotations"]["gpu-fault.io/workload-ids"] = json.dumps(
                    workload_ids
                )
            pod_spec = template.setdefault("spec", {})
            node_name = pod_spec.get("nodeName")
            if node_name in (node_rebindings or {}):
                pod_spec["nodeName"] = node_rebindings[node_name]
            node_selector = pod_spec.get("nodeSelector")
            if isinstance(node_selector, dict):
                hostname = node_selector.get("kubernetes.io/hostname")
                if hostname in (node_rebindings or {}):
                    node_selector["kubernetes.io/hostname"] = node_rebindings[hostname]

        serialized = cls._serialize_workload(workload)
        spec = serialized.get("spec") or {}
        if kind == "pytorchjob":
            replicas = spec.get("pytorchReplicaSpecs") or {}
            patches = {}
            for role, replica in replicas.items():
                if not isinstance(replica, dict):
                    continue
                template = deepcopy(replica.get("template") or {})
                update_template(template)
                patches[role] = {"template": template}
            return {
                "runPolicy": {"suspend": False},
                "pytorchReplicaSpecs": patches,
            }
        if kind == "jobset":
            replicated_jobs = deepcopy(spec.get("replicatedJobs") or [])
            for replicated in replicated_jobs:
                template = (
                    (replicated.setdefault("template", {}))
                    .setdefault("spec", {})
                    .setdefault("template", {})
                )
                update_template(template)
            return {
                "suspend": False,
                "replicatedJobs": replicated_jobs,
            }
        raise ValueError(f"unsupported custom workload kind: {kind}")

    @staticmethod
    def _node_rebindings(
        context: WorkflowStepContext,
    ) -> dict[str, str]:
        rebindings: dict[str, str] = {}
        for execution in context.workflow.step_executions:
            if (
                execution.status is WorkflowStepStatus.SUCCEEDED
                and execution.operation is WorkflowOperation.REPLACE_NODE
            ):
                values = execution.details.get("node_rebindings")
                if isinstance(values, dict):
                    rebindings.update(
                        {
                            str(source): str(target)
                            for source, target in values.items()
                            if source and target
                        }
                    )
        return rebindings

    @classmethod
    def _declared_gpu_count(cls, kind: str, workload: Any) -> int | None:
        body = cls._serialize_workload(workload)
        annotations = (body.get("metadata") or {}).get("annotations") or {}
        override = annotations.get(ANNOTATION_TARGET_GPU_COUNT)
        if override is not None:
            try:
                value = int(override)
            except (TypeError, ValueError):
                return None
            return value if value >= 0 else None

        spec = body.get("spec") or {}
        if kind == "job":
            per_pod = cls._pod_template_gpu_count(spec.get("template"))
            if per_pod is None:
                return None
            return per_pod * int(spec.get("completions") or 1)
        if kind == "pytorchjob":
            replica_specs = spec.get("pytorchReplicaSpecs") or {}
            if not isinstance(replica_specs, dict):
                return None
            total = 0
            for replica in replica_specs.values():
                if not isinstance(replica, dict):
                    return None
                per_pod = cls._pod_template_gpu_count(replica.get("template"))
                if per_pod is None:
                    return None
                total += per_pod * int(replica.get("replicas") or 1)
            return total
        if kind == "jobset":
            replicated_jobs = spec.get("replicatedJobs") or []
            if not isinstance(replicated_jobs, list):
                return None
            total = 0
            for replicated in replicated_jobs:
                if not isinstance(replicated, dict):
                    return None
                job_spec = (replicated.get("template") or {}).get("spec") or {}
                per_pod = cls._pod_template_gpu_count(job_spec.get("template"))
                if per_pod is None:
                    return None
                total += (
                    per_pod
                    * int(job_spec.get("completions") or 1)
                    * int(replicated.get("replicas") or 1)
                )
            return total
        return None

    @staticmethod
    def _serialize_workload(
        workload: Any,
    ) -> dict[str, Any]:
        if isinstance(workload, dict):
            return workload
        try:
            from kubernetes import client
        except ImportError:
            return {}
        value = client.ApiClient().sanitize_for_serialization(workload)
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _pod_template_gpu_count(
        template: Any,
    ) -> int | None:
        if not isinstance(template, dict):
            return None
        pod_spec = template.get("spec") or {}
        containers = pod_spec.get("containers") or []
        if not isinstance(containers, list) or not containers:
            return None
        total = 0
        for container in containers:
            if not isinstance(container, dict):
                return None
            resources = container.get("resources") or {}
            requests = resources.get("requests") or {}
            limits = resources.get("limits") or {}
            resource_names = {
                key
                for key in {*requests, *limits}
                if (key == "nvidia.com/gpu" or key.startswith("nvidia.com/mig-"))
            }
            for resource_name in resource_names:
                try:
                    requested = int(requests.get(resource_name, 0))
                    limited = int(limits.get(resource_name, 0))
                except (TypeError, ValueError):
                    return None
                total += max(requested, limited)
        return total

    @staticmethod
    def _retry_suffix(idempotency_key: str) -> str:
        return hashlib.sha256(idempotency_key.encode()).hexdigest()[:8]

    @classmethod
    def _retry_job_name(cls, name: str, idempotency_key: str) -> str:
        return f"{name[:52]}-r-{cls._retry_suffix(idempotency_key)}"

    def _workload_active(
        self,
        namespace: str,
        kind: str,
        name: str,
        workload: Any | None = None,
    ) -> bool | None:
        if workload is None:
            workload = self._read_workload(namespace, kind, name)
        if kind == "job":
            status = (
                workload.get("status", {})
                if isinstance(workload, dict)
                else workload.status
            )
            active = status.get("active") if isinstance(status, dict) else status.active
            terminating = (
                status.get("terminating")
                if isinstance(status, dict)
                else getattr(status, "terminating", None)
            )
            return bool((active or 0) or (terminating or 0))

        status = workload.get("status")
        if not isinstance(status, dict):
            return None
        counts = []
        for value in status.get("replicaStatuses", {}).values():
            if isinstance(value, dict):
                counts.append(value.get("active"))
        for value in status.get("replicatedJobs", []):
            if isinstance(value, dict):
                counts.append(value.get("active"))
        known = [value for value in counts if value is not None]
        if known and any(int(value) > 0 for value in known):
            return True
        conditions = status.get("conditions", [])
        for condition in conditions:
            if str(condition.get("status", "")).lower() == "true" and condition.get(
                "type"
            ) in {
                "Suspended",
                "Succeeded",
                "Failed",
            }:
                return False
            if (
                str(condition.get("status", "")).lower() == "true"
                and condition.get("type") == "Running"
            ):
                return True
        return False if known else None
