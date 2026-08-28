from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from enum import StrEnum
from threading import RLock
from typing import Any, Protocol
from uuid import uuid4

from pydantic import Field, model_validator

from gpu_fault.models import (
    CapabilityClaim,
    CapabilityMode,
    CapabilityName,
    Environment,
    ObservedCapability,
    RuntimeProfile,
    StrictModel,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)


LOGGER = logging.getLogger(__name__)

HYPERPOD_JOB_AUTO_RESUME_ANNOTATION = "sagemaker.amazonaws.com/enable-job-auto-resume"


class SageMakerHyperPodClient(Protocol):
    def describe_cluster(self, **kwargs) -> dict[str, Any]: ...

    def list_cluster_nodes(self, **kwargs) -> dict[str, Any]: ...

    def describe_cluster_node(self, **kwargs) -> dict[str, Any]: ...

    def batch_reboot_cluster_nodes(self, **kwargs) -> dict[str, Any]: ...


class HyperPodAction(StrEnum):
    REBOOT = "REBOOT"
    REPLACE = "REPLACE"


class HyperPodRecoveryState(StrEnum):
    ENABLED = "ENABLED"
    DISABLED = "DISABLED"
    UNKNOWN = "UNKNOWN"


class HyperPodRecoveryOwner(StrEnum):
    HYPERPOD_MANAGED = "HYPERPOD_MANAGED"
    GPU_FAULT_ADAPTER = "GPU_FAULT_ADAPTER"
    UNRESOLVED = "UNRESOLVED"


class HyperPodAdvisoryDisposition(StrEnum):
    ADVISE_ONLY = "ADVISE_ONLY"
    ADVISE_AND_EXECUTE = "ADVISE_AND_EXECUTE"
    ADVISE_AND_BLOCK = "ADVISE_AND_BLOCK"


class HyperPodWorkloadRecoveryEvidence(StrictModel):
    state: HyperPodRecoveryState
    source: str
    workload_id: str | None = None

    @classmethod
    def from_eks_annotations(
        cls,
        annotations: dict[str, str] | None,
        *,
        workload_id: str | None = None,
    ) -> HyperPodWorkloadRecoveryEvidence:
        if annotations is None:
            return cls(
                state=HyperPodRecoveryState.UNKNOWN,
                source="eks-annotations-unavailable",
                workload_id=workload_id,
            )
        value = annotations.get(HYPERPOD_JOB_AUTO_RESUME_ANNOTATION)
        enabled = value is not None and value.strip().lower() in {
            "1",
            "true",
            "yes",
            "enabled",
        }
        return cls(
            state=(
                HyperPodRecoveryState.ENABLED
                if enabled
                else HyperPodRecoveryState.DISABLED
            ),
            source="eks-workload-annotation",
            workload_id=workload_id,
        )

    @classmethod
    def from_slurm_auto_resume(
        cls,
        enabled: bool | None,
        *,
        workload_id: str | None = None,
    ) -> HyperPodWorkloadRecoveryEvidence:
        return cls(
            state=(
                HyperPodRecoveryState.UNKNOWN
                if enabled is None
                else (
                    HyperPodRecoveryState.ENABLED
                    if enabled
                    else HyperPodRecoveryState.DISABLED
                )
            ),
            source="slurm-srun-auto-resume",
            workload_id=workload_id,
        )


class HyperPodCapabilityOwnership(StrictModel):
    capability: CapabilityName
    state: HyperPodRecoveryState
    owner_type: HyperPodRecoveryOwner
    mode: CapabilityMode
    owner: str
    adapter: str | None = None
    reason: str


class HyperPodRecoveryAdvisory(StrictModel):
    capability: CapabilityName
    recommended_action: str
    execution_owner: str
    disposition: HyperPodAdvisoryDisposition
    rationale: list[str]
    evidence_refs: list[str] = Field(default_factory=list)


class HyperPodResiliencyOwnership(StrictModel):
    cluster_name: str
    environment: Environment
    node_recovery: HyperPodRecoveryState
    workload_recovery: HyperPodRecoveryState
    workload_evidence_source: str
    capabilities: list[HyperPodCapabilityOwnership]
    runtime_profile: RuntimeProfile
    warnings: list[str] = Field(default_factory=list)


class HyperPodAdapterConfig(StrictModel):
    cluster_name: str
    region_name: str | None = None
    execution_enabled: bool = False
    reboot_enabled: bool | None = None
    replace_enabled: bool = False
    allow_when_node_recovery_automatic: bool = False
    allowed_actions: list[HyperPodAction] = Field(
        default_factory=lambda: [
            HyperPodAction.REBOOT,
            # REPLACE is also the action used by local-only warm-spare
            # failover. Provider replacement is gated independently.
            HyperPodAction.REPLACE,
        ]
    )
    max_batch_size: int = Field(default=25, ge=1, le=25)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def enforce_recovery_invariants(self) -> HyperPodAdapterConfig:
        if self.replace_enabled:
            raise ValueError(
                "provider node replacement is prohibited; "
                "replace_enabled must remain false"
            )
        if self.allow_when_node_recovery_automatic:
            raise ValueError(
                "GPU fault mutation with HyperPod NodeRecovery=Automatic is prohibited"
            )
        return self

    def action_enabled(self, action: HyperPodAction) -> bool:
        if action is HyperPodAction.REPLACE:
            return False
        override = (
            self.reboot_enabled
            if action is HyperPodAction.REBOOT
            else self.replace_enabled
        )
        return self.execution_enabled if override is None else override

    @classmethod
    def from_environment(
        cls,
        *,
        cluster_name: str | None = None,
        region_name: str | None = None,
    ) -> HyperPodAdapterConfig:
        cluster_name = cluster_name or os.getenv("GPU_FAULT_HYPERPOD_CLUSTER")
        if not cluster_name:
            raise ValueError("GPU_FAULT_HYPERPOD_CLUSTER is required")
        legacy_mutation_enabled = (
            os.getenv("GPU_FAULT_ALLOW_HYPERPOD_MUTATION", "").lower() == "true"
        )
        reboot_value = os.getenv("GPU_FAULT_ALLOW_HYPERPOD_REBOOT")
        replace_value = os.getenv("GPU_FAULT_ALLOW_HYPERPOD_REPLACE")
        if replace_value is not None and replace_value.lower() == "true":
            raise ValueError(
                "GPU_FAULT_ALLOW_HYPERPOD_REPLACE must remain false; "
                "provider replacement is outside the supported design"
            )
        if (
            os.getenv(
                "GPU_FAULT_ALLOW_WITH_AUTOMATIC_NODE_RECOVERY",
                "",
            ).lower()
            == "true"
        ):
            raise ValueError(
                "GPU_FAULT_ALLOW_WITH_AUTOMATIC_NODE_RECOVERY must remain false"
            )
        return cls(
            cluster_name=cluster_name,
            region_name=region_name
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION"),
            execution_enabled=legacy_mutation_enabled,
            reboot_enabled=(
                legacy_mutation_enabled
                if reboot_value is None
                else reboot_value.lower() == "true"
            ),
            replace_enabled=False,
            allow_when_node_recovery_automatic=False,
            max_batch_size=int(os.getenv("GPU_FAULT_HYPERPOD_MAX_BATCH_SIZE", "25")),
        )


class HyperPodNode(StrictModel):
    node_logical_id: str
    instance_id: str | None = None
    instance_group_name: str | None = None
    instance_type: str | None = None
    status: str
    status_message: str | None = None
    private_dns_hostname: str | None = None
    private_primary_ip: str | None = None
    availability_zone: str | None = None
    capacity_type: str | None = None
    kubernetes_labels: dict[str, str] = Field(default_factory=dict)
    kubernetes_taints: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def aliases(self) -> set[str]:
        aliases = {self.node_logical_id}
        if self.instance_id:
            aliases.add(self.instance_id)
            aliases.add(f"hyperpod-{self.instance_id}")
        if self.private_dns_hostname:
            aliases.add(self.private_dns_hostname)
            aliases.add(self.private_dns_hostname.split(".", 1)[0])
        if self.private_primary_ip:
            aliases.add(self.private_primary_ip)
        for key in (
            "kubernetes.io/hostname",
            "node.kubernetes.io/instance-id",
        ):
            if self.kubernetes_labels.get(key):
                aliases.add(self.kubernetes_labels[key])
        return aliases


class HyperPodClusterSnapshot(StrictModel):
    cluster_name: str
    cluster_arn: str | None = None
    cluster_status: str
    node_recovery: str | None = None
    orchestrator: dict[str, Any] = Field(default_factory=dict)
    nodes: list[HyperPodNode]


class HyperPodPreflight(StrictModel):
    action: HyperPodAction
    cluster_name: str
    cluster_status: str
    node_recovery: str | None = None
    targets: list[HyperPodNode]
    execution_enabled: bool
    safe_to_submit: bool
    gate_failures: list[str] = Field(default_factory=list)


class HyperPodNodeFailure(StrictModel):
    node_logical_id: str | None = None
    node_id: str | None = None
    error_code: str | None = None
    message: str | None = None


class HyperPodSubmissionResult(StrictModel):
    operation_id: str = Field(default_factory=lambda: f"hyperpod-op-{uuid4()}")
    idempotency_key: str
    action: HyperPodAction
    cluster_name: str
    requested_node_logical_ids: list[str]
    successful_node_logical_ids: list[str]
    failures: list[HyperPodNodeFailure] = Field(default_factory=list)
    submitted: bool = True
    duplicate: bool = False


class HyperPodSubmissionRecord(StrictModel):
    """Durable record of one HyperPod batch mutation attempt.

    Idempotency for reboot/replace cannot live in process memory: the
    executor Pod that submits the batch is frequently the same Pod that
    the mutation restarts, and a control-plane failover moves the retry
    to a different replica entirely. Both cases lose the in-memory dict
    and re-submit a provider lifecycle mutation for a step that
    already ran. The record is written INTENDED before the provider call
    and updated to SUBMITTED/FAILED after, so a crash in between is
    recoverable as "unknown outcome" rather than silently retried.
    """

    cluster_name: str
    idempotency_key: str
    action: HyperPodAction
    requested_node_identifiers: list[str]
    state: str = "INTENDED"
    result: HyperPodSubmissionResult | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def request_identity(self) -> tuple[HyperPodAction, tuple[str, ...]]:
        return (
            self.action,
            tuple(sorted(self.requested_node_identifiers)),
        )


class HyperPodAdapterError(ValueError):
    pass


class HyperPodLifecycleAdapter:
    """Safe wrapper over SageMaker HyperPod node lifecycle APIs."""

    def __init__(
        self,
        config: HyperPodAdapterConfig,
        client: SageMakerHyperPodClient | None = None,
        *,
        store: Any | None = None,
    ) -> None:
        self.config = config
        self.client = client or self._create_client(config)
        # When a store is supplied, submission idempotency survives a
        # process restart and a replica failover. Without one it degrades
        # to the process-local dicts below, which is only safe for tests
        # and for one-shot CLI use.
        self.store = store
        self._submissions: dict[str, HyperPodSubmissionResult] = {}
        self._submission_requests: dict[
            str, tuple[HyperPodAction, tuple[str, ...]]
        ] = {}
        self._lock = RLock()

    @staticmethod
    def _create_client(config: HyperPodAdapterConfig):
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("install gpu-fault-control-plane[hyperpod]") from exc
        return boto3.client("sagemaker", region_name=config.region_name)

    def discover(self) -> HyperPodClusterSnapshot:
        cluster = self.client.describe_cluster(ClusterName=self.config.cluster_name)
        return HyperPodClusterSnapshot(
            cluster_name=cluster["ClusterName"],
            cluster_arn=cluster.get("ClusterArn"),
            cluster_status=cluster["ClusterStatus"],
            node_recovery=cluster.get("NodeRecovery"),
            orchestrator=cluster.get("Orchestrator", {}),
            nodes=self.list_nodes(enrich=True),
        )

    def resolve_recovery_ownership(
        self,
        workload_recovery: HyperPodWorkloadRecoveryEvidence,
    ) -> HyperPodResiliencyOwnership:
        """Select exactly one writer for node and workload recovery."""
        cluster = self.client.describe_cluster(ClusterName=self.config.cluster_name)
        environment = (
            Environment.HYPERPOD_EKS
            if cluster.get("Orchestrator", {}).get("Eks") is not None
            else Environment.HYPERPOD_SLURM
        )
        node_value = cluster.get("NodeRecovery")
        node_state = {
            "Automatic": HyperPodRecoveryState.ENABLED,
            "None": HyperPodRecoveryState.DISABLED,
        }.get(node_value, HyperPodRecoveryState.UNKNOWN)

        capabilities = [
            self._recovery_capability(
                CapabilityName.NODE_REBOOT,
                node_state,
                managed_owner="hyperpod-managed-node-recovery",
                managed_adapter="hyperpod-managed",
                custom_owner="gpu-fault-hyperpod-adapter",
                custom_adapter="hyperpod-lifecycle",
            ),
            self._recovery_capability(
                CapabilityName.NODE_REPLACE,
                node_state,
                managed_owner="hyperpod-managed-node-recovery",
                managed_adapter="hyperpod-managed",
                custom_owner="gpu-fault-hyperpod-adapter",
                custom_adapter="hyperpod-lifecycle",
            ),
            self._recovery_capability(
                CapabilityName.WORKLOAD_STOP,
                workload_recovery.state,
                managed_owner="hyperpod-managed-job-recovery",
                managed_adapter="hyperpod-managed",
                custom_owner="gpu-fault-restart-controller",
                custom_adapter="training-restart",
            ),
            self._recovery_capability(
                CapabilityName.WORKLOAD_RESTART,
                workload_recovery.state,
                managed_owner="hyperpod-managed-job-recovery",
                managed_adapter="hyperpod-managed",
                custom_owner="gpu-fault-restart-controller",
                custom_adapter="training-restart",
            ),
        ]
        claims = [
            CapabilityClaim(
                capability=item.capability,
                mode=item.mode,
                owner=item.owner,
                adapter=item.adapter,
            )
            for item in capabilities
        ]
        observed = [
            ObservedCapability(
                capability=item.capability,
                owner=item.owner,
                available=True,
                version=node_value
                if item.capability
                in {
                    CapabilityName.NODE_REBOOT,
                    CapabilityName.NODE_REPLACE,
                }
                else workload_recovery.source,
            )
            for item in capabilities
            if item.mode in {CapabilityMode.OWN, CapabilityMode.DELEGATE}
        ]
        warnings = []
        if node_state is HyperPodRecoveryState.UNKNOWN:
            warnings.append(
                "NodeRecovery is unknown; node mutation remains observe-only"
            )
        if workload_recovery.state is HyperPodRecoveryState.UNKNOWN:
            warnings.append(
                "workload auto-recovery is unknown; workload restart "
                "remains observe-only"
            )
        profile = RuntimeProfile(
            cluster_id=self.config.cluster_name,
            environment=environment,
            profile_version=(
                "hyperpod-auto-"
                f"{node_state.value.lower()}-"
                f"{workload_recovery.state.value.lower()}"
            ),
            claims=claims,
            observed=observed,
        )
        return HyperPodResiliencyOwnership(
            cluster_name=self.config.cluster_name,
            environment=environment,
            node_recovery=node_state,
            workload_recovery=workload_recovery.state,
            workload_evidence_source=workload_recovery.source,
            capabilities=capabilities,
            runtime_profile=profile,
            warnings=warnings,
        )

    def create_recovery_advisory(
        self,
        ownership: HyperPodResiliencyOwnership,
        *,
        capability: CapabilityName,
        recommended_action: str,
        rationale: list[str],
        evidence_refs: list[str] | None = None,
    ) -> HyperPodRecoveryAdvisory:
        if ownership.cluster_name != self.config.cluster_name:
            raise HyperPodAdapterError(
                "ownership snapshot belongs to a different cluster"
            )
        try:
            selected = next(
                item for item in ownership.capabilities if item.capability is capability
            )
        except StopIteration as exc:
            raise HyperPodAdapterError(
                f"capability is not a HyperPod recovery capability: {capability.value}"
            ) from exc
        disposition = {
            HyperPodRecoveryOwner.HYPERPOD_MANAGED: (
                HyperPodAdvisoryDisposition.ADVISE_ONLY
            ),
            HyperPodRecoveryOwner.GPU_FAULT_ADAPTER: (
                HyperPodAdvisoryDisposition.ADVISE_AND_EXECUTE
            ),
            HyperPodRecoveryOwner.UNRESOLVED: (
                HyperPodAdvisoryDisposition.ADVISE_AND_BLOCK
            ),
        }[selected.owner_type]
        return HyperPodRecoveryAdvisory(
            capability=capability,
            recommended_action=recommended_action,
            execution_owner=selected.owner,
            disposition=disposition,
            rationale=rationale,
            evidence_refs=evidence_refs or [],
        )

    @staticmethod
    def _recovery_capability(
        capability: CapabilityName,
        state: HyperPodRecoveryState,
        *,
        managed_owner: str,
        managed_adapter: str,
        custom_owner: str,
        custom_adapter: str,
    ) -> HyperPodCapabilityOwnership:
        if state is HyperPodRecoveryState.ENABLED:
            return HyperPodCapabilityOwnership(
                capability=capability,
                state=state,
                owner_type=HyperPodRecoveryOwner.HYPERPOD_MANAGED,
                mode=CapabilityMode.DELEGATE,
                owner=managed_owner,
                adapter=managed_adapter,
                reason="HyperPod managed recovery is enabled",
            )
        if state is HyperPodRecoveryState.DISABLED:
            return HyperPodCapabilityOwnership(
                capability=capability,
                state=state,
                owner_type=HyperPodRecoveryOwner.GPU_FAULT_ADAPTER,
                mode=CapabilityMode.OWN,
                owner=custom_owner,
                adapter=custom_adapter,
                reason=(
                    "HyperPod managed recovery is disabled; custom "
                    "adapter owns recovery"
                ),
            )
        return HyperPodCapabilityOwnership(
            capability=capability,
            state=state,
            owner_type=HyperPodRecoveryOwner.UNRESOLVED,
            mode=CapabilityMode.OBSERVE,
            owner="unresolved-recovery-owner",
            reason=("recovery setting is unknown; destructive automation is blocked"),
        )

    def list_nodes(self, *, enrich: bool = False) -> list[HyperPodNode]:
        nodes: list[HyperPodNode] = []
        token = None
        while True:
            request: dict[str, Any] = {
                "ClusterName": self.config.cluster_name,
                "MaxResults": 100,
                "IncludeNodeLogicalIds": True,
            }
            if token:
                request["NextToken"] = token
            response = self.client.list_cluster_nodes(**request)
            nodes.extend(
                self._node_from_summary(item)
                for item in response.get("ClusterNodeSummaries", [])
            )
            token = response.get("NextToken")
            if not token:
                break
        if enrich:
            nodes = [self.describe_node(node.node_logical_id) for node in nodes]
        return nodes

    def describe_node(self, node_logical_id: str) -> HyperPodNode:
        response = self.client.describe_cluster_node(
            ClusterName=self.config.cluster_name,
            NodeLogicalId=node_logical_id,
        )
        return self._node_from_details(response["NodeDetails"])

    def resolve_nodes(
        self,
        identifiers: list[str],
        *,
        nodes: list[HyperPodNode] | None = None,
    ) -> list[HyperPodNode]:
        if not identifiers:
            raise HyperPodAdapterError("at least one target node is required")
        candidates = nodes or self.list_nodes(enrich=True)
        index: dict[str, list[HyperPodNode]] = {}
        for node in candidates:
            for alias in node.aliases:
                index.setdefault(alias, []).append(node)

        resolved = []
        for identifier in identifiers:
            matches = index.get(identifier, [])
            if not matches:
                raise HyperPodAdapterError(
                    f"node is not part of HyperPod cluster: {identifier}"
                )
            unique = {item.node_logical_id: item for item in matches}
            if len(unique) != 1:
                raise HyperPodAdapterError(
                    f"node identifier is ambiguous: {identifier}"
                )
            resolved.append(next(iter(unique.values())))
        return list({node.node_logical_id: node for node in resolved}.values())

    def preflight(
        self,
        action: HyperPodAction,
        node_identifiers: list[str],
        *,
        isolation_verified_nodes: list[str] | None = None,
        require_execution_enabled: bool = True,
    ) -> HyperPodPreflight:
        cluster = self.client.describe_cluster(ClusterName=self.config.cluster_name)
        nodes = self.list_nodes(enrich=True)
        targets = self.resolve_nodes(node_identifiers, nodes=nodes)
        failures: list[str] = []

        if action not in self.config.allowed_actions:
            failures.append(f"action {action.value} is not in adapter allowlist")
        if cluster["ClusterStatus"] != "InService":
            failures.append(
                "cluster must be InService, got " + cluster["ClusterStatus"]
            )
        node_recovery = cluster.get("NodeRecovery")
        if (
            node_recovery == "Automatic"
            and not self.config.allow_when_node_recovery_automatic
        ):
            failures.append(
                "HyperPod automatic node recovery is enabled; direct "
                "batch mutation would create a second lifecycle trigger"
            )
        if len(targets) > self.config.max_batch_size:
            failures.append(f"batch exceeds limit {self.config.max_batch_size}")
        invalid_status = [
            f"{node.node_logical_id}:{node.status}"
            for node in targets
            if node.status not in {"Running", "Failure"}
        ]
        if invalid_status:
            failures.append(
                "nodes have non-actionable status: " + ", ".join(invalid_status)
            )

        isolation_aliases = set(isolation_verified_nodes or [])
        not_isolated = [
            node.node_logical_id
            for node in targets
            if not node.aliases.intersection(isolation_aliases)
        ]
        if not_isolated:
            failures.append(
                "trusted scheduler isolation evidence is missing for: "
                + ", ".join(not_isolated)
            )
        if require_execution_enabled and not self.config.action_enabled(action):
            failures.append(
                f"HyperPod {action.value} mutation is disabled by configuration"
            )

        return HyperPodPreflight(
            action=action,
            cluster_name=self.config.cluster_name,
            cluster_status=cluster["ClusterStatus"],
            node_recovery=node_recovery,
            targets=targets,
            execution_enabled=self.config.action_enabled(action),
            safe_to_submit=not failures,
            gate_failures=failures,
        )

    def submit(
        self,
        action: HyperPodAction,
        node_identifiers: list[str],
        *,
        isolation_verified_nodes: list[str],
        confirm_cluster_name: str,
        workflow_fencing_token: int,
        expected_fencing_token: int,
        idempotency_key: str,
    ) -> HyperPodSubmissionResult:
        if action is HyperPodAction.REPLACE:
            raise HyperPodAdapterError(
                "provider node replacement is prohibited; "
                "use the healthy warm-spare coordinator"
            )
        with self._lock:
            reserved_record = None
            if self.store is not None:
                duplicate = self._replay_durable_submission(
                    action, node_identifiers, idempotency_key
                )
                if duplicate is not None:
                    return duplicate
            else:
                existing = self._submissions.get(idempotency_key)
                if existing is not None:
                    request_identity = (
                        action,
                        tuple(sorted(node_identifiers)),
                    )
                    if self._submission_requests[idempotency_key] != request_identity:
                        raise HyperPodAdapterError(
                            "idempotency key was already used for a "
                            "different HyperPod request"
                        )
                    return existing.model_copy(update={"duplicate": True})
            if confirm_cluster_name != self.config.cluster_name:
                raise HyperPodAdapterError(
                    "cluster confirmation does not match configured HyperPod cluster"
                )
            if workflow_fencing_token != expected_fencing_token:
                raise HyperPodAdapterError("stale workflow fencing token")

            preflight = self.preflight(
                action,
                node_identifiers,
                isolation_verified_nodes=isolation_verified_nodes,
            )
            if not preflight.safe_to_submit:
                raise HyperPodAdapterError(
                    "HyperPod preflight failed: " + "; ".join(preflight.gate_failures)
                )

            logical_ids = [node.node_logical_id for node in preflight.targets]
            if self.store is not None:
                # Write the intent before the provider call. Whoever wins
                # this reservation is the only caller allowed to submit;
                # a loser here means a concurrent replica already holds
                # the key, so replay its outcome instead of submitting a
                # second batch for the same step.
                reserved_record, reserved = self.store.reserve_hyperpod_submission(
                    HyperPodSubmissionRecord(
                        cluster_name=self.config.cluster_name,
                        idempotency_key=idempotency_key,
                        action=action,
                        requested_node_identifiers=list(node_identifiers),
                    )
                )
                if not reserved:
                    duplicate = self._durable_submission_outcome(
                        reserved_record,
                        action,
                        node_identifiers,
                    )
                    if duplicate is not None:
                        return duplicate
                    raise HyperPodAdapterError(
                        "a HyperPod submission for this idempotency "
                        "key is already reserved and has no recorded "
                        "outcome; it is either in flight on another "
                        "replica or was interrupted, so retrying here "
                        "could submit a second batch"
                    )
            request = {
                "ClusterName": self.config.cluster_name,
                "NodeLogicalIds": logical_ids,
            }
            try:
                response = self.client.batch_reboot_cluster_nodes(**request)
            except Exception as exc:
                # The provider call may still have taken effect, so the
                # record must not go back to a state that permits a
                # second submission. UNKNOWN is terminal for retries and
                # forces an operator decision.
                self._record_submission_outcome(
                    reserved_record,
                    state="UNKNOWN",
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise

            failures = [
                HyperPodNodeFailure(
                    node_logical_id=item.get("NodeLogicalId"),
                    node_id=item.get("NodeId"),
                    error_code=item.get("ErrorCode"),
                    message=item.get("Message"),
                )
                for item in (
                    response.get("FailedNodeLogicalIds", [])
                    + response.get("Failed", [])
                )
            ]
            result = HyperPodSubmissionResult(
                idempotency_key=idempotency_key,
                action=action,
                cluster_name=self.config.cluster_name,
                requested_node_logical_ids=logical_ids,
                successful_node_logical_ids=response.get(
                    "SuccessfulNodeLogicalIds", []
                ),
                failures=failures,
            )
            self._submissions[idempotency_key] = result
            self._submission_requests[idempotency_key] = (
                action,
                tuple(sorted(node_identifiers)),
            )
            self._record_submission_outcome(
                reserved_record, state="SUBMITTED", result=result
            )
            return result

    def _replay_durable_submission(
        self,
        action: HyperPodAction,
        node_identifiers: list[str],
        idempotency_key: str,
    ) -> HyperPodSubmissionResult | None:
        """Return the recorded outcome for an already-used key."""

        try:
            record = self.store.get_hyperpod_submission(
                self.config.cluster_name, idempotency_key
            )
        except (KeyError, LookupError):
            return None
        return self._durable_submission_outcome(record, action, node_identifiers)

    def _durable_submission_outcome(
        self,
        record: HyperPodSubmissionRecord,
        action: HyperPodAction,
        node_identifiers: list[str],
    ) -> HyperPodSubmissionResult | None:
        if record.request_identity != (
            action,
            tuple(sorted(node_identifiers)),
        ):
            raise HyperPodAdapterError(
                "idempotency key was already used for a different HyperPod request"
            )
        if record.state == "SUBMITTED" and record.result is not None:
            return record.result.model_copy(update={"duplicate": True})
        if record.state == "UNKNOWN":
            raise HyperPodAdapterError(
                "a previous HyperPod submission for this idempotency "
                "key has an unknown outcome; an operator must confirm "
                "the node state before retrying: "
                + (record.error or "no error recorded")
            )
        return None

    def _record_submission_outcome(
        self,
        record: HyperPodSubmissionRecord | None,
        *,
        state: str,
        result: HyperPodSubmissionResult | None = None,
        error: str | None = None,
    ) -> None:
        if self.store is None or record is None:
            return
        try:
            self.store.save_hyperpod_submission(
                record.model_copy(
                    update={
                        "state": state,
                        "result": result,
                        "error": error,
                        "updated_at": datetime.now(timezone.utc),
                    }
                )
            )
        except Exception:
            # The provider call already happened. Losing the outcome
            # record leaves the key reserved in the INTENDED state,
            # which fails closed on the next attempt rather than
            # re-submitting, so log and do not mask the real result.
            LOGGER.exception(
                "could not persist HyperPod submission outcome: "
                "cluster=%s key=%s state=%s",
                record.cluster_name,
                record.idempotency_key,
                state,
            )

    def execute_step(
        self,
        step: WorkflowStepSpec,
        *,
        isolation_verified_nodes: list[str],
        confirm_cluster_name: str,
        workflow_fencing_token: int,
        expected_fencing_token: int,
        idempotency_key: str,
    ) -> HyperPodSubmissionResult:
        action = {
            WorkflowOperation.RESTART_NODE: HyperPodAction.REBOOT,
            WorkflowOperation.REPLACE_NODE: HyperPodAction.REPLACE,
        }.get(step.operation)
        if action is None:
            raise HyperPodAdapterError(
                f"unsupported HyperPod operation: {step.operation.value}"
            )
        return self.submit(
            action,
            step.node_ids,
            isolation_verified_nodes=isolation_verified_nodes,
            confirm_cluster_name=confirm_cluster_name,
            workflow_fencing_token=workflow_fencing_token,
            expected_fencing_token=expected_fencing_token,
            idempotency_key=idempotency_key,
        )

    @staticmethod
    def _node_from_summary(item: dict[str, Any]) -> HyperPodNode:
        status = item.get("InstanceStatus", {})
        logical_id = item.get("NodeLogicalId")
        if not logical_id:
            raise HyperPodAdapterError(
                "HyperPod API did not return NodeLogicalId; "
                "IncludeNodeLogicalIds support is required"
            )
        return HyperPodNode(
            node_logical_id=logical_id,
            instance_id=item.get("InstanceId"),
            instance_group_name=item.get("InstanceGroupName"),
            instance_type=item.get("InstanceType"),
            status=status.get("Status", "NotFound"),
            status_message=status.get("Message"),
            private_dns_hostname=item.get("PrivateDnsHostname"),
        )

    @staticmethod
    def _node_from_details(item: dict[str, Any]) -> HyperPodNode:
        status = item.get("InstanceStatus", {})
        kubernetes = item.get("KubernetesConfig", {})
        placement = item.get("Placement", {})
        logical_id = item.get("NodeLogicalId")
        if not logical_id:
            raise HyperPodAdapterError(
                "DescribeClusterNode response has no NodeLogicalId"
            )
        return HyperPodNode(
            node_logical_id=logical_id,
            instance_id=item.get("InstanceId"),
            instance_group_name=item.get("InstanceGroupName"),
            instance_type=item.get("InstanceType"),
            status=status.get("Status", "NotFound"),
            status_message=status.get("Message"),
            private_dns_hostname=item.get("PrivateDnsHostname"),
            private_primary_ip=item.get("PrivatePrimaryIp"),
            availability_zone=placement.get("AvailabilityZone"),
            capacity_type=item.get("CapacityType"),
            kubernetes_labels=kubernetes.get("CurrentLabels", {}),
            kubernetes_taints=kubernetes.get("CurrentTaints", []),
        )


class HyperPodWorkflowDispatcher:
    """Dispatches only HyperPod-owned lifecycle steps.

    A successful result means SageMaker accepted the request. A separate
    observer must verify node transition and validation gates before marking
    the incident recovered.
    """

    def __init__(
        self,
        adapter: HyperPodLifecycleAdapter,
        *,
        execution_owner: str = "gpu-fault-hyperpod-adapter",
    ) -> None:
        self.adapter = adapter
        self.execution_owner = execution_owner

    def preflight(
        self,
        workflow: WorkflowRequest,
        step_index: int,
        *,
        isolation_verified_nodes: list[str],
        require_execution_enabled: bool = True,
    ) -> HyperPodPreflight:
        step = self._step(workflow, step_index)
        action = self._action(step)
        return self.adapter.preflight(
            action,
            step.node_ids,
            isolation_verified_nodes=isolation_verified_nodes,
            require_execution_enabled=require_execution_enabled,
        )

    def submit(
        self,
        workflow: WorkflowRequest,
        step_index: int,
        *,
        isolation_verified_nodes: list[str],
        confirm_cluster_name: str,
        expected_fencing_token: int,
    ) -> HyperPodSubmissionResult:
        step = self._step(workflow, step_index)
        return self.adapter.execute_step(
            step,
            isolation_verified_nodes=isolation_verified_nodes,
            confirm_cluster_name=confirm_cluster_name,
            workflow_fencing_token=workflow.fencing_token,
            expected_fencing_token=expected_fencing_token,
            idempotency_key=(
                f"{workflow.request_id}/{step.operation.value}/{step_index}"
            ),
        )

    def _step(self, workflow: WorkflowRequest, step_index: int) -> WorkflowStepSpec:
        if workflow.status not in {
            WorkflowStatus.PENDING,
            WorkflowStatus.RUNNING,
        }:
            raise HyperPodAdapterError(
                "workflow must be PENDING or RUNNING for HyperPod dispatch"
            )
        try:
            step = workflow.official_steps[step_index]
        except IndexError as exc:
            raise HyperPodAdapterError(
                f"workflow step index out of range: {step_index}"
            ) from exc
        if step.execution_owner != self.execution_owner:
            raise HyperPodAdapterError(
                f"workflow step owner {step.execution_owner} does not "
                f"match HyperPod owner {self.execution_owner}"
            )
        self._action(step)
        return step

    @staticmethod
    def _action(step: WorkflowStepSpec) -> HyperPodAction:
        action = {
            WorkflowOperation.RESTART_NODE: HyperPodAction.REBOOT,
            WorkflowOperation.REPLACE_NODE: HyperPodAction.REPLACE,
        }.get(step.operation)
        if action is None:
            raise HyperPodAdapterError(
                "HyperPod lifecycle dispatcher only supports "
                "RESTART_NODE and REPLACE_NODE"
            )
        return action
