"""Node lifecycle provider for a plain EKS GPU cluster (managed node groups).

The same surface ``HyperPodLifecycleAdapter`` offers -- ``discover``,
``preflight``, ``list_nodes``, ``describe_node``, ``resolve_nodes``,
``submit``, ``execute_step`` -- backed by Kubernetes and EC2 instead of
SageMaker. Nodes come from the Kubernetes API under the cluster's label
selector and are joined with ``ec2 describe-instances`` by the instance id in
``spec.providerID`` (``aws:///<zone>/i-...``). Preflight requires the EKS
cluster and every target's managed node group to be ``ACTIVE``. Reboot is
``ec2 reboot-instances`` on exactly the instance ids the targets were bound to
when preflight resolved them; a Node whose providerID moved in between refuses
the submission. Provider-side replacement does not exist here: ``REPLACE`` is
never in the allowlist and ``nodeReplace`` stays OBSERVE in the profile.

Statuses are mapped onto the vocabulary the confirmation code already speaks:
an EC2 instance in state ``running`` reads ``Running``, so the reboot
confirmation ("provider Running plus a new Node Agent incarnation") is the
HyperPod one with a different provider behind it.
"""

from __future__ import annotations

import logging
import os
from threading import RLock
from typing import Any, Mapping, Protocol
from uuid import uuid4

from pydantic import Field, model_validator

from gpu_fault.env import env_bool
from gpu_fault.hyperpod import (
    HyperPodAction,
    HyperPodAdapterError,
    HyperPodCapabilityOwnership,
    HyperPodClusterSnapshot,
    HyperPodLifecycleAdapter,
    HyperPodNode,
    HyperPodPreflight,
    HyperPodRecoveryOwner,
    HyperPodRecoveryState,
    HyperPodResiliencyOwnership,
    HyperPodSubmissionRecord,
    HyperPodSubmissionResult,
    HyperPodWorkloadRecoveryEvidence,
)
from gpu_fault.models import (
    CapabilityClaim,
    CapabilityMode,
    CapabilityName,
    Environment,
    ObservedCapability,
    RuntimeProfile,
    StrictModel,
    WorkflowOperation,
    WorkflowStepSpec,
)
from gpu_fault.node_provider import (
    EKS_NODEGROUP_LABEL,
    EKS_REBOOT_CONFIRMATION_SOURCE,
    NODE_SELECTOR_ENV,
    PROVIDER_CLUSTER_ENV,
    ClusterProvider,
    node_selector,
    parse_provider_id,
    resolve_provider_nodes,
)
from gpu_fault.provider_submissions import ProviderSubmissionLedger

LOGGER = logging.getLogger(__name__)

EKS_ADAPTER_OWNER = "gpu-fault-eks-adapter"
EKS_CLUSTER_NAME_ENV = "GPU_FAULT_EKS_CLUSTER_NAME"
#: The tags EKS stamps on every managed node group instance.
EKS_CLUSTER_TAG = "eks:cluster-name"
EKS_NODEGROUP_TAG = "eks:nodegroup-name"
#: ``DescribeInstances`` filter values per call; well under the API bound.
DESCRIBE_INSTANCES_CHUNK = 100
#: EC2 instance states in the provider vocabulary the confirmation code reads.
EC2_STATE_NAMES: dict[str, str] = {
    "pending": "Pending",
    "running": "Running",
    "shutting-down": "ShuttingDown",
    "terminated": "Terminated",
    "stopping": "Stopping",
    "stopped": "Stopped",
}


class Ec2Client(Protocol):
    def describe_instances(self, **kwargs: Any) -> dict[str, Any]: ...

    def reboot_instances(self, **kwargs: Any) -> dict[str, Any]: ...


class EksClient(Protocol):
    def describe_cluster(self, **kwargs: Any) -> dict[str, Any]: ...

    def describe_nodegroup(self, **kwargs: Any) -> dict[str, Any]: ...


class EksProviderError(HyperPodAdapterError):
    """Subclasses the HyperPod error so every provider-neutral handler catches it."""


class EksProviderConfig(StrictModel):
    """What the EKS provider needs; ``cluster_name`` is the stable cluster id."""

    cluster_id: str
    node_selector: str
    eks_cluster_name: str | None = None
    region_name: str | None = None
    execution_enabled: bool = False
    reboot_enabled: bool | None = None
    replace_enabled: bool = False
    allowed_actions: list[HyperPodAction] = Field(
        default_factory=lambda: [HyperPodAction.REBOOT]
    )
    max_batch_size: int = Field(default=25, ge=1, le=25)

    @model_validator(mode="after")
    def enforce_invariants(self) -> EksProviderConfig:
        if self.replace_enabled or HyperPodAction.REPLACE in self.allowed_actions:
            raise ValueError(
                "provider node replacement is unsupported on eks-managed "
                "clusters; nodeReplace stays OBSERVE"
            )
        if not self.cluster_id.strip():
            raise ValueError("cluster_id must not be blank")
        if not self.node_selector.strip():
            raise ValueError("node_selector must not be blank")
        return self

    @property
    def cluster_name(self) -> str:
        """The provider cluster scope; the cluster id on a plain EKS cluster."""

        return self.cluster_id

    def action_enabled(self, action: HyperPodAction) -> bool:
        if action is not HyperPodAction.REBOOT:
            return False
        if self.reboot_enabled is None:
            return self.execution_enabled
        return self.reboot_enabled

    @classmethod
    def from_environment(
        cls,
        *,
        cluster_id: str | None = None,
        region_name: str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> EksProviderConfig:
        values = os.environ if environ is None else environ
        cluster_id = cluster_id or (values.get(PROVIDER_CLUSTER_ENV) or "").strip()
        if not cluster_id:
            raise ValueError(
                f"{PROVIDER_CLUSTER_ENV} is required (it carries the cluster id "
                "of an eks-managed cluster)"
            )
        selector = values.get(NODE_SELECTOR_ENV, "").strip()
        if not selector:
            raise ValueError(f"{NODE_SELECTOR_ENV} is required for eks-managed")
        if env_bool("GPU_FAULT_ALLOW_HYPERPOD_REPLACE", False, environ=values):
            raise ValueError(
                "GPU_FAULT_ALLOW_HYPERPOD_REPLACE must remain false; "
                "an eks-managed cluster has no provider replacement"
            )
        legacy_mutation_enabled = env_bool(
            "GPU_FAULT_ALLOW_HYPERPOD_MUTATION", False, environ=values
        )
        return cls(
            cluster_id=cluster_id,
            node_selector=selector,
            eks_cluster_name=values.get(EKS_CLUSTER_NAME_ENV, "").strip() or None,
            region_name=region_name
            or values.get("AWS_REGION")
            or values.get("AWS_DEFAULT_REGION"),
            execution_enabled=legacy_mutation_enabled,
            # The same switch the HyperPod executor runs with; the manifest
            # sets it once for every provider.
            reboot_enabled=env_bool(
                "GPU_FAULT_ALLOW_HYPERPOD_REBOOT",
                legacy_mutation_enabled,
                environ=values,
            ),
            max_batch_size=int(values.get("GPU_FAULT_HYPERPOD_MAX_BATCH_SIZE", "25")),
        )


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _tags(instance: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(tag.get("Key")): str(tag.get("Value", ""))
        for tag in instance.get("Tags") or []
        if isinstance(tag, dict) and tag.get("Key")
    }


class EksManagedNodeProvider:
    """Reboot-only lifecycle provider for a plain EKS cluster."""

    provider = ClusterProvider.EKS_MANAGED
    provider_label = "EKS"
    confirmation_source = EKS_REBOOT_CONFIRMATION_SOURCE

    def __init__(
        self,
        config: EksProviderConfig,
        core_api: Any,
        *,
        ec2_client: Ec2Client | None = None,
        eks_client: EksClient | None = None,
        store: Any | None = None,
    ) -> None:
        self.config = config
        self.core = core_api
        self.ec2 = ec2_client or self._create_client("ec2", config.region_name)
        self.eks = eks_client or self._create_client("eks", config.region_name)
        self.store = store
        self._ledger = ProviderSubmissionLedger(
            cluster_name=config.cluster_name,
            store=store,
            error=EksProviderError,
            provider_label=self.provider_label,
            record_type=HyperPodSubmissionRecord,
        )
        self._lock = RLock()
        # ``eks:cluster-name`` / ``eks:nodegroup-name`` of the instances the
        # last listing joined, by node name; preflight binds the EKS cluster
        # it describes to them.
        self._instance_tags: dict[str, dict[str, str]] = {}

    @staticmethod
    def _create_client(service: str, region_name: str | None) -> Any:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("install gpu-fault-control-plane[hyperpod]") from exc
        return boto3.client(service, region_name=region_name)

    # -- inventory --------------------------------------------------------

    def discover(self) -> HyperPodClusterSnapshot:
        nodes = self.list_nodes(enrich=True)
        cluster = self._describe_cluster(self._eks_cluster_name(nodes))
        return HyperPodClusterSnapshot(
            cluster_name=self.config.cluster_name,
            cluster_arn=cluster.get("arn"),
            cluster_status=str(cluster.get("status") or "UNKNOWN"),
            node_recovery=None,
            orchestrator={"Eks": {"ClusterArn": cluster.get("arn")}},
            nodes=nodes,
        )

    def list_nodes(self, *, enrich: bool = False) -> list[HyperPodNode]:
        """The selected Kubernetes nodes, each joined with its EC2 instance."""

        # ``enrich`` is the HyperPod surface's flag; every EKS listing carries
        # the EC2 join because there is no cheaper shape to offer.
        response = self.core.list_node(label_selector=self.config.node_selector)
        items = _value(response, "items", []) or []
        views = [self._node_view(item) for item in items]
        instances = self._describe_instances(
            [view.instance_id for view in views if view.instance_id]
        )
        self._instance_tags = {}
        return [
            self._join(view, instances.get(view.instance_id or "")) for view in views
        ]

    def describe_node(self, node_name: str) -> HyperPodNode:
        node = self.core.read_node(node_name)
        view = self._node_view(node)
        instances = self._describe_instances(
            [view.instance_id] if view.instance_id else []
        )
        return self._join(view, instances.get(view.instance_id or ""))

    def resolve_nodes(
        self,
        identifiers: list[str],
        *,
        nodes: list[HyperPodNode] | None = None,
    ) -> list[HyperPodNode]:
        return resolve_provider_nodes(
            identifiers,
            self.list_nodes(enrich=True) if nodes is None else nodes,
            error=EksProviderError,
            membership="the selected EKS nodes",
        )

    def _node_view(self, node: Any) -> HyperPodNode:
        metadata = _value(node, "metadata", {}) or {}
        name = str(_value(metadata, "name", "") or "")
        if not name:
            raise EksProviderError("Kubernetes node is missing metadata.name")
        labels = dict(_value(metadata, "labels", {}) or {})
        labels.setdefault("kubernetes.io/hostname", name)
        spec = _value(node, "spec", {}) or {}
        provider_id = _value(spec, "providerID", None)
        if provider_id is None:
            provider_id = _value(spec, "provider_id", None)
        parsed = parse_provider_id(provider_id)
        taints = [
            dict(taint)
            if isinstance(taint, dict)
            else {
                "key": getattr(taint, "key", None),
                "value": getattr(taint, "value", None),
                "effect": getattr(taint, "effect", None),
            }
            for taint in (_value(spec, "taints", None) or [])
        ]
        return HyperPodNode(
            node_logical_id=name,
            instance_id=parsed[1] if parsed else None,
            instance_group_name=labels.get(EKS_NODEGROUP_LABEL),
            instance_type=(
                labels.get("node.kubernetes.io/instance-type")
                or labels.get("beta.kubernetes.io/instance-type")
            ),
            status="NotFound",
            status_message=(
                None if parsed else "node has no EC2 providerID (aws:///<zone>/i-...)"
            ),
            availability_zone=(
                (parsed[0] if parsed else None)
                or labels.get("topology.kubernetes.io/zone")
            ),
            capacity_type=labels.get("eks.amazonaws.com/capacityType"),
            kubernetes_labels=labels,
            kubernetes_taints=taints,
        )

    def _join(
        self, view: HyperPodNode, instance: Mapping[str, Any] | None
    ) -> HyperPodNode:
        if instance is None:
            return view
        tags = _tags(instance)
        self._instance_tags[view.node_logical_id] = tags
        state = str((instance.get("State") or {}).get("Name") or "").lower()
        reason = instance.get("StateReason") or {}
        return view.model_copy(
            update={
                "status": EC2_STATE_NAMES.get(state, "Unknown"),
                "status_message": reason.get("Message") if state != "running" else None,
                "instance_type": view.instance_type or instance.get("InstanceType"),
                "instance_group_name": (
                    view.instance_group_name or tags.get(EKS_NODEGROUP_TAG)
                ),
                "private_dns_hostname": instance.get("PrivateDnsName") or None,
                "private_primary_ip": instance.get("PrivateIpAddress") or None,
                "availability_zone": (
                    view.availability_zone
                    or (instance.get("Placement") or {}).get("AvailabilityZone")
                ),
                "capacity_type": (
                    view.capacity_type or instance.get("InstanceLifecycle") or None
                ),
            }
        )

    def _describe_instances(self, instance_ids: list[str]) -> dict[str, dict[str, Any]]:
        """``DescribeInstances`` by filter, so a terminated-and-gone id is absent, not an error."""

        found: dict[str, dict[str, Any]] = {}
        unique = sorted(set(instance_ids))
        for start in range(0, len(unique), DESCRIBE_INSTANCES_CHUNK):
            chunk = unique[start : start + DESCRIBE_INSTANCES_CHUNK]
            token: str | None = None
            while True:
                request: dict[str, Any] = {
                    "Filters": [{"Name": "instance-id", "Values": chunk}]
                }
                if token:
                    request["NextToken"] = token
                response = self.ec2.describe_instances(**request)
                for reservation in response.get("Reservations") or []:
                    for instance in reservation.get("Instances") or []:
                        instance_id = instance.get("InstanceId")
                        if isinstance(instance_id, str) and instance_id:
                            found[instance_id] = instance
                token = response.get("NextToken")
                if not token:
                    break
        return found

    # -- EKS control plane --------------------------------------------------

    def _eks_cluster_name(self, nodes: list[HyperPodNode]) -> str:
        """The EKS cluster the targets belong to; configured or read off their tags."""

        tagged = {
            self._instance_tags.get(node.node_logical_id, {}).get(EKS_CLUSTER_TAG)
            for node in nodes
        }
        tagged.discard(None)
        tagged.discard("")
        configured = self.config.eks_cluster_name
        if configured:
            foreign = sorted(str(name) for name in tagged if name != configured)
            if foreign:
                raise EksProviderError(
                    f"target instances are tagged for EKS cluster(s) {foreign}, "
                    f"not the configured {configured}"
                )
            return configured
        if len(tagged) != 1:
            raise EksProviderError(
                "cannot bind the EKS cluster: targets carry "
                + (
                    "no eks:cluster-name tag"
                    if not tagged
                    else f"conflicting eks:cluster-name tags {sorted(map(str, tagged))}"
                )
                + f"; set {EKS_CLUSTER_NAME_ENV} to pin it"
            )
        return str(next(iter(tagged)))

    def _describe_cluster(self, name: str) -> dict[str, Any]:
        response = self.eks.describe_cluster(name=name)
        cluster = response.get("cluster")
        if not isinstance(cluster, dict):
            raise EksProviderError(f"DescribeCluster returned no cluster for {name}")
        return cluster

    def _nodegroup_status(self, cluster: str, nodegroup: str) -> str:
        response = self.eks.describe_nodegroup(
            clusterName=cluster, nodegroupName=nodegroup
        )
        group = response.get("nodegroup")
        if not isinstance(group, dict):
            raise EksProviderError(
                f"DescribeNodegroup returned no nodegroup for {cluster}/{nodegroup}"
            )
        return str(group.get("status") or "UNKNOWN")

    # -- lifecycle ------------------------------------------------------------

    def preflight(
        self,
        action: HyperPodAction,
        node_identifiers: list[str],
        *,
        isolation_verified_nodes: list[str] | None = None,
        require_execution_enabled: bool = True,
    ) -> HyperPodPreflight:
        nodes = self.list_nodes(enrich=True)
        targets = self.resolve_nodes(node_identifiers, nodes=nodes)
        failures: list[str] = []
        if action not in self.config.allowed_actions:
            failures.append(
                f"action {action.value} is not in adapter allowlist; an "
                "eks-managed cluster has no provider node replacement"
            )
        cluster_status = "UNKNOWN"
        try:
            cluster_name = self._eks_cluster_name(targets)
            cluster_status = str(self._describe_cluster(cluster_name).get("status"))
            if cluster_status != "ACTIVE":
                failures.append(
                    f"EKS cluster {cluster_name} must be ACTIVE, got {cluster_status}"
                )
            missing_group = [
                node.node_logical_id for node in targets if not node.instance_group_name
            ]
            if missing_group:
                failures.append(
                    "nodes carry no managed node group label "
                    f"({EKS_NODEGROUP_LABEL}): " + ", ".join(missing_group)
                )
            for group in sorted(
                {
                    node.instance_group_name
                    for node in targets
                    if node.instance_group_name
                }
            ):
                status = self._nodegroup_status(cluster_name, group)
                if status != "ACTIVE":
                    failures.append(f"node group {group} must be ACTIVE, got {status}")
        except EksProviderError as exc:
            failures.append(str(exc))
        if len(targets) > self.config.max_batch_size:
            failures.append(f"batch exceeds limit {self.config.max_batch_size}")
        unbound = [node.node_logical_id for node in targets if not node.instance_id]
        if unbound:
            failures.append("nodes have no EC2 providerID: " + ", ".join(unbound))
        invalid_status = [
            f"{node.node_logical_id}:{node.status}"
            for node in targets
            if node.status != "Running"
        ]
        if invalid_status:
            failures.append(
                "nodes have non-actionable EC2 state: " + ", ".join(invalid_status)
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
            failures.append(f"EKS {action.value} mutation is disabled by configuration")
        return HyperPodPreflight(
            action=action,
            cluster_name=self.config.cluster_name,
            cluster_status=cluster_status,
            node_recovery=None,
            targets=targets,
            execution_enabled=self.config.action_enabled(action),
            safe_to_submit=not failures,
            gate_failures=failures,
        )

    def _bound_instance_ids(self, targets: list[HyperPodNode]) -> list[str]:
        """The instance ids to reboot, re-read from each Node this instant.

        The preflight resolved each target to an instance id; a Node whose
        providerID now names another instance (replaced under the same name,
        or a stale listing) must not have the *new* instance rebooted.
        """

        bound: list[str] = []
        for target in targets:
            current = self._node_view(self.core.read_node(target.node_logical_id))
            if not target.instance_id or current.instance_id != target.instance_id:
                raise EksProviderError(
                    f"node {target.node_logical_id} is bound to instance "
                    f"{target.instance_id}, but its providerID now names "
                    f"{current.instance_id}; refusing to reboot"
                )
            bound.append(target.instance_id)
        return bound

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
        if action is not HyperPodAction.REBOOT:
            raise EksProviderError(
                "provider node replacement is unsupported on eks-managed clusters"
            )
        with self._lock:
            duplicate = self._ledger.replay(action, node_identifiers, idempotency_key)
            if duplicate is not None:
                return duplicate
            if confirm_cluster_name != self.config.cluster_name:
                raise EksProviderError(
                    "cluster confirmation does not match the configured cluster id"
                )
            if workflow_fencing_token != expected_fencing_token:
                raise EksProviderError("stale workflow fencing token")
            preflight = self.preflight(
                action,
                node_identifiers,
                isolation_verified_nodes=isolation_verified_nodes,
            )
            if not preflight.safe_to_submit:
                raise EksProviderError(
                    "EKS preflight failed: " + "; ".join(preflight.gate_failures)
                )
            instance_ids = self._bound_instance_ids(preflight.targets)
            node_names = [node.node_logical_id for node in preflight.targets]
            reserved_record, duplicate = self._ledger.reserve(
                action, node_identifiers, idempotency_key
            )
            if duplicate is not None:
                return duplicate
            try:
                self.ec2.reboot_instances(InstanceIds=instance_ids)
            except Exception as exc:
                self._ledger.record_outcome(
                    reserved_record,
                    state="UNKNOWN",
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            result = HyperPodSubmissionResult(
                operation_id=f"eks-op-{uuid4()}",
                idempotency_key=idempotency_key,
                action=action,
                cluster_name=self.config.cluster_name,
                requested_node_logical_ids=node_names,
                successful_node_logical_ids=node_names,
            )
            self._ledger.remember(
                idempotency_key, result, action, node_identifiers, reserved_record
            )
            LOGGER.warning(
                "submitted EC2 RebootInstances for %s (nodes %s) key=%s",
                ",".join(instance_ids),
                ",".join(node_names),
                idempotency_key,
            )
            return result

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
            raise EksProviderError(f"unsupported EKS operation: {step.operation.value}")
        return self.submit(
            action,
            step.node_ids,
            isolation_verified_nodes=isolation_verified_nodes,
            confirm_cluster_name=confirm_cluster_name,
            workflow_fencing_token=workflow_fencing_token,
            expected_fencing_token=expected_fencing_token,
            idempotency_key=idempotency_key,
        )

    # -- ownership ------------------------------------------------------------

    def resolve_recovery_ownership(
        self,
        workload_recovery: HyperPodWorkloadRecoveryEvidence,
    ) -> HyperPodResiliencyOwnership:
        """The EKS counterpart of the HyperPod ownership resolution.

        A plain EKS cluster has no managed node recovery: reboot is owned by
        this adapter, replacement is observed only, and workload recovery
        follows the same evidence as on HyperPod.
        """

        capabilities = [
            HyperPodCapabilityOwnership(
                capability=CapabilityName.NODE_REBOOT,
                state=HyperPodRecoveryState.DISABLED,
                owner_type=HyperPodRecoveryOwner.GPU_FAULT_ADAPTER,
                mode=CapabilityMode.OWN,
                owner=EKS_ADAPTER_OWNER,
                adapter="eks-lifecycle",
                reason="plain EKS has no managed node recovery; EC2 RebootInstances",
            ),
            HyperPodCapabilityOwnership(
                capability=CapabilityName.NODE_REPLACE,
                state=HyperPodRecoveryState.DISABLED,
                owner_type=HyperPodRecoveryOwner.GPU_FAULT_ADAPTER,
                mode=CapabilityMode.OBSERVE,
                owner=EKS_ADAPTER_OWNER,
                adapter="eks-lifecycle",
                reason="provider node replacement is unsupported on eks-managed",
            ),
            HyperPodLifecycleAdapter._recovery_capability(
                CapabilityName.WORKLOAD_STOP,
                workload_recovery.state,
                managed_owner="hyperpod-managed-job-recovery",
                managed_adapter="hyperpod-managed",
                custom_owner="gpu-fault-restart-controller",
                custom_adapter="training-restart",
            ),
            HyperPodLifecycleAdapter._recovery_capability(
                CapabilityName.WORKLOAD_RESTART,
                workload_recovery.state,
                managed_owner="hyperpod-managed-job-recovery",
                managed_adapter="hyperpod-managed",
                custom_owner="gpu-fault-restart-controller",
                custom_adapter="training-restart",
            ),
        ]
        profile = RuntimeProfile(
            cluster_id=self.config.cluster_name,
            environment=Environment.EKS,
            profile_version=(
                f"eks-auto-disabled-{workload_recovery.state.value.lower()}"
            ),
            claims=[
                CapabilityClaim(
                    capability=item.capability,
                    mode=item.mode,
                    owner=item.owner,
                    adapter=item.adapter,
                )
                for item in capabilities
            ],
            observed=[
                ObservedCapability(
                    capability=item.capability,
                    owner=item.owner,
                    available=True,
                    version=(
                        "ec2-reboot-v1"
                        if item.capability is CapabilityName.NODE_REBOOT
                        else workload_recovery.source
                    ),
                )
                for item in capabilities
                if item.mode in {CapabilityMode.OWN, CapabilityMode.DELEGATE}
            ],
        )
        warnings = []
        if workload_recovery.state is HyperPodRecoveryState.UNKNOWN:
            warnings.append(
                "workload auto-recovery is unknown; workload restart "
                "remains observe-only"
            )
        return HyperPodResiliencyOwnership(
            cluster_name=self.config.cluster_name,
            environment=Environment.EKS,
            node_recovery=HyperPodRecoveryState.DISABLED,
            workload_recovery=workload_recovery.state,
            workload_evidence_source=workload_recovery.source,
            capabilities=capabilities,
            runtime_profile=profile,
            warnings=warnings,
        )


def eks_provider_from_environment(
    core_api: Any,
    *,
    store: Any | None = None,
    environ: Mapping[str, str] | None = None,
) -> EksManagedNodeProvider:
    """The provider the executor constructs when the cluster is ``eks-managed``.

    Validates the selector through the shared ``node_selector`` reader so the
    executor and the reconciler cannot disagree on the scope.
    """

    config = EksProviderConfig.from_environment(environ=environ)
    selector = node_selector(environ)
    if selector != config.node_selector:
        raise ValueError("node selector readers disagree; refusing to start")
    return EksManagedNodeProvider(config, core_api, store=store)
