from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Callable, Protocol

from gpu_fault.adapters.common import (
    ANNOTATION_APPROVE_GPU_COUNT_CHANGE,
    ANNOTATION_RESTART_BUDGET,
    ANNOTATION_RESTART_COUNT,
    ANNOTATION_TARGET_GPU_COUNT,
    LABEL_ATTEMPT_ID,
    LABEL_JOB_ID,
)
from gpu_fault.execution import (
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.models import (
    IncidentState,
    WorkflowEventCode,
    WorkflowOperation,
    WorkflowStepStatus,
)


class IncidentOwnershipReport(Protocol):
    """The slice of the control plane's ownership answer the premise reads."""

    known: bool
    incident_state: str | None


class IncidentOwnershipProvider(Protocol):
    """How a storeless (regional) executor asks about an incident's state."""

    def incident_ownership(self, incident_id: str) -> IncidentOwnershipReport: ...


class KubernetesRestartOperationsMixin:
    # Attributes supplied by the composed concrete implementation.
    _annotations: Callable[..., Any]
    _labels: Callable[..., Any]
    _read_workload: Callable[..., Any]
    alert_sender: Callable[..., Any]
    batch: Any
    custom: Any
    notification_sink: Any
    ownership_provider: IncidentOwnershipProvider | None
    restart_email_builder: Any
    store: Any

    #: Incident states from which ``RECOVERED`` is no longer reachable.
    _UNRECOVERABLE_INCIDENT_STATES = frozenset(
        {IncidentState.QUARANTINED.value, IncidentState.ESCALATED.value}
    )
    #: The shared "node under repair" vocabulary (review item 4). The
    #: dispatcher's busy-node hold uses the same code; the executor turns a
    #: WAITING outcome carrying it into one HOLD audit event.
    HOLD_REASON_NODE_UNDER_REMEDIATION = WorkflowEventCode.NODE_UNDER_REMEDIATION.value
    HOLD_REASON_INCIDENT_NOT_RECOVERABLE = "INCIDENT_NOT_RECOVERABLE"
    #: The premise this adapter is waiting on, kept beside the shared code.
    PREMISE_REASON_INCIDENT_NOT_RECOVERED = "INCIDENT_NOT_RECOVERED"

    @staticmethod
    def _rejected(error: str, **details: Any) -> WorkflowStepOutcome:
        """A refusal made before any restart was submitted.

        ``restart_submitted: False`` is what the control plane's terminal write
        reads to hand the preflight's reservation back
        (``release_unattempted_restart_reservations``): this adapter neither
        reserves nor releases budget itself.
        """

        return WorkflowStepOutcome.failed(
            error, details={**details, "restart_submitted": False}
        )

    def _restart_guard(
        self,
        context: WorkflowStepContext,
        workloads: list[tuple[str, str, str, str, Any]],
    ) -> tuple[WorkflowStepOutcome | None, int | None]:
        """Compare the step's ``RestartAuthorization`` with the workload.

        The control plane reserves the job's restart budget in its preflight
        and signs that reservation into the authorization; this guard only
        checks that the authorization describes the workload in front of it.
        ``(None, restart_count)`` clears the restart; otherwise the outcome
        says why not.
        """

        parameters = context.step.parameters
        authorization = context.request.restart_authorization
        if authorization is None:
            return (
                self._rejected("restart safety guard requires a restart authorization"),
                None,
            )
        premise = self._incident_premise(context)
        if premise is not None:
            return premise, None
        avoided = self._avoid_node_ids(context)
        if avoided:
            rebindings = self._node_rebindings(context)
            pinned = sorted(
                {
                    node
                    for _, kind, _, _, workload in workloads
                    for node in self._pinned_nodes(kind, workload, rebindings)
                    if node in avoided
                }
            )
            if pinned:
                return (
                    self._rejected(
                        "restart target is pinned to an avoided node: "
                        + ", ".join(pinned),
                        reason="RESTART_TARGET_AVOIDED",
                        avoided_node_ids=pinned,
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
                self._rejected(
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
                self._rejected(
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
                    self._rejected(
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
                        # A hold, not a submission: a workflow cancelled
                        # while it waits here owes no restart budget.
                        "restart_submitted": False,
                    },
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
            "source_attempt_id": authorization.source_attempt_id,
            "source_gpu_count": authorization.source_gpu_count,
            "restart_budget": authorization.restart_budget,
            "reservation_id": authorization.reservation_id,
        }
        if actual != expected:
            return (
                self._rejected(
                    "restart authorization does not match the workload safety context"
                ),
                None,
            )
        return None, authorization.restart_count

    def _incident_premise(
        self, context: WorkflowStepContext
    ) -> WorkflowStepOutcome | None:
        """Honour ``requires_incident_state`` at execution time.

        A restart planned after a repair depends on the incident that owns the
        nodes being ``RECOVERED``. The compiler writes that premise onto the
        step from ``RecoveryPlan.restart_after_incident_id``; both executors
        read it -- the simulated one straight off the plan, this adapter off
        the compiled step -- because for a while only the simulated executor
        did, and a production restart went ahead mid-repair (F-G5).
        Not yet recovered: wait (the executor re-runs the step). Can never
        recover (quarantined, escalated): fail. Cannot be verified: fail
        closed rather than restart on an assumption.
        """
        parameters = context.step.parameters
        required = parameters.get("requires_incident_state")
        if not required:
            return None
        required_state = str(required)
        incident_id = str(parameters.get("incident_id") or context.incident.incident_id)
        state, remediation_workflow_id = self._incident_snapshot(incident_id)
        details: dict[str, Any] = {
            "incident_id": incident_id,
            "required_incident_state": required_state,
            "incident_state": state,
            # The workflow repairing the nodes: the same object the
            # dispatcher's busy-node hold names, so both waits point at it.
            "remediation_workflow_id": remediation_workflow_id,
        }
        if state is None:
            return self._rejected(
                f"cannot verify that incident {incident_id} is {required_state}: "
                "no store or ownership provider can answer",
                **details,
                reason="INCIDENT_STATE_UNVERIFIABLE",
            )
        if state == required_state:
            return None
        if state in self._UNRECOVERABLE_INCIDENT_STATES:
            return self._rejected(
                f"incident {incident_id} is {state}; it will never be "
                f"{required_state}, so the workload cannot be restarted on its nodes",
                **details,
                reason=self.HOLD_REASON_INCIDENT_NOT_RECOVERABLE,
            )
        return WorkflowStepOutcome.waiting(
            operation_id=context.idempotency_key,
            details={
                **details,
                "reason": self.HOLD_REASON_NODE_UNDER_REMEDIATION,
                "premise_reason": self.PREMISE_REASON_INCIDENT_NOT_RECOVERED,
                "restart_submitted": False,
            },
        )

    def _incident_state(self, incident_id: str) -> str | None:
        return self._incident_snapshot(incident_id)[0]

    def _incident_snapshot(self, incident_id: str) -> tuple[str | None, str | None]:
        """``(state, workflow_request_id)`` of an incident, or ``(None, None)``
        when nothing here can answer -- the premise then fails closed."""

        if self.store is not None:
            try:
                incident = self.store.get_incident(incident_id)
            except Exception:  # noqa: BLE001 - unknown incident is "unverifiable"
                return None, None
            state = incident.state
            workflow_id = incident.workflow_request_id
            return (
                str(getattr(state, "value", state)),
                None if workflow_id is None else str(workflow_id),
            )
        provider = getattr(self, "ownership_provider", None)
        if provider is None or not hasattr(provider, "incident_ownership"):
            # Older providers answer only incident_workflow_is_terminal.
            return None, None
        try:
            report = provider.incident_ownership(incident_id)
        except Exception:  # noqa: BLE001 - a failed lookup is "unverifiable"
            return None, None
        if not report.known:
            return None, None
        workflow_id = getattr(report, "workflow_request_id", None)
        return (
            str(report.incident_state) if report.incident_state else None,
            None if workflow_id is None else str(workflow_id),
        )

    @staticmethod
    def _avoid_node_ids(context: WorkflowStepContext) -> list[str]:
        values = context.step.parameters.get("avoid_node_ids")
        if not isinstance(values, list):
            return []
        return sorted({str(value) for value in values if value})

    @classmethod
    def _pod_specs(cls, kind: str, workload: Any) -> list[dict[str, Any]]:
        spec = cls._serialize_workload(workload).get("spec") or {}
        if kind == "job":
            pod_spec = (spec.get("template") or {}).get("spec")
            return [pod_spec] if isinstance(pod_spec, dict) else []
        if kind == "pytorchjob":
            return [
                pod_spec
                for replica in (spec.get("pytorchReplicaSpecs") or {}).values()
                if isinstance(replica, dict)
                and isinstance(
                    pod_spec := (replica.get("template") or {}).get("spec"), dict
                )
            ]
        if kind == "jobset":
            specs: list[dict[str, Any]] = []
            for replicated in spec.get("replicatedJobs") or []:
                if not isinstance(replicated, dict):
                    continue
                job_spec = (replicated.get("template") or {}).get("spec") or {}
                pod_spec = (job_spec.get("template") or {}).get("spec")
                if isinstance(pod_spec, dict):
                    specs.append(pod_spec)
            return specs
        return []

    @classmethod
    def _pinned_nodes(
        cls, kind: str, workload: Any, rebindings: dict[str, str]
    ) -> set[str]:
        """Nodes a workload's Pod templates are pinned to, after rebinding."""
        pinned: set[str] = set()
        for pod_spec in cls._pod_specs(kind, workload):
            node_name = pod_spec.get("nodeName")
            if isinstance(node_name, str) and node_name:
                pinned.add(rebindings.get(node_name, node_name))
            node_selector = pod_spec.get("nodeSelector")
            if isinstance(node_selector, dict):
                hostname = node_selector.get("kubernetes.io/hostname")
                if isinstance(hostname, str) and hostname:
                    pinned.add(rebindings.get(hostname, hostname))
        return pinned

    @staticmethod
    def _apply_node_avoidance(pod_spec: dict[str, Any], avoided: list[str]) -> None:
        """Keep the restarted Pods off ``avoided`` with a required anti-affinity.

        The expression is added to every existing node-selector term (terms
        are OR-ed, expressions within a term AND-ed), or as the only term.
        """
        if not avoided:
            return
        expression = {
            "key": "kubernetes.io/hostname",
            "operator": "NotIn",
            "values": list(avoided),
        }
        affinity = pod_spec.setdefault("affinity", {})
        node_affinity = affinity.setdefault("nodeAffinity", {})
        required = node_affinity.setdefault(
            "requiredDuringSchedulingIgnoredDuringExecution", {}
        )
        terms = required.setdefault("nodeSelectorTerms", [])
        if not terms:
            terms.append({"matchExpressions": [expression]})
            return
        for term in terms:
            expressions = term.setdefault("matchExpressions", [])
            if expression not in expressions:
                expressions.append(expression)

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
        self._apply_node_avoidance(template_spec, self._avoid_node_ids(context))
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
            avoid_node_ids=self._avoid_node_ids(context),
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
        avoid_node_ids: list[str] | None = None,
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
            cls._apply_node_avoidance(pod_spec, list(avoid_node_ids or []))

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
