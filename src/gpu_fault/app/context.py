from __future__ import annotations

import logging
import os
from datetime import timedelta

from gpu_fault.adapters import (
    ControlPlaneEvidenceAdapter,
    GpuValidationAdapter,
    HyperPodLifecycleStepAdapter,
    KubernetesWorkflowAdapter,
    ManagedRecoveryObserverAdapter,
    NodeActionWorkflowAdapter,
    SimulatedRecoveryExecutor,
    SupportEscalationAdapter,
)
from gpu_fault.app.identity import pod_process_owner
from gpu_fault.capabilities import compile_runtime_profile
from gpu_fault.control_record_archive import ControlRecordArchiver
from gpu_fault.diagnostics import KubernetesDcgmDiagnosticAdapter
from gpu_fault.execution.branch_escalation import BranchEscalator
from gpu_fault.execution.config import validate_timing_from_environment
from gpu_fault.env import env_bool
from gpu_fault.env_validation import validate_gpu_fault_environment
from gpu_fault.execution import (
    ProductionExecutorConfig,
    ProductionWorkflowExecutor,
    WorkflowDispatcher,
    WorkflowDispatcherConfig,
    managed_recovery_timeout_seconds,
)
from gpu_fault.fleet import (
    BarrierCoordinator,
    FleetCompatibilityPolicy,
    FleetRegistry,
    node_action_secrets_from_environment,
    parse_endpoint_networks,
)
from gpu_fault.gpu_metrics import (
    GpuMetricsService,
    GpuMetricsThresholds,
)
from gpu_fault.hma import HyperPodHmaNormalizer
from gpu_fault.host_health import NodeHealthPolicy
from gpu_fault.hyperpod import (
    HyperPodAdapterConfig,
    HyperPodLifecycleAdapter,
)
from gpu_fault.hyperpod_spares import HyperPodSpareCoordinator
from gpu_fault.managed_recovery import (
    HyperPodIdentityRegistry,
    HyperPodManagedRecoveryObserver,
    RegionalHyperPodManagedRecoveryObserver,
)
from gpu_fault.models import (
    CapabilityClaim,
    CapabilityMode,
    CapabilityName,
    EffectiveRuntimeProfile,
    Environment,
    ObservedCapability,
    RuntimeProfile,
)
from gpu_fault.notification_service import AdvisoryNotificationService
from gpu_fault.notifications import notification_notifier_from_environment
from gpu_fault.orchestration import IncidentOrchestrator
from gpu_fault.passive import PassiveWorkflowCompiler
from gpu_fault.policy import GpuFaultPolicyEngine
from gpu_fault.regional import RegionalRemoteWorkflowAdapter
from gpu_fault.regional_registry import (
    sync_regional_cluster_registry,
)
from gpu_fault.service import CompletionService
from gpu_fault.settings import ControlPlaneSettings
from gpu_fault.spare_health import HyperPodSpareHealthController
from gpu_fault.store import (
    InMemoryStore,
    PostgresStore,
    SimulatedDiagnosticAdapter,
    SqliteStore,
)
from gpu_fault.store.contracts import ControlPlaneStore
from gpu_fault.telemetry import (
    EvidenceService,
    NvSwitchPortTopologyService,
    WorkloadTopologyService,
)
from gpu_fault.training_health import (
    TrainingHealthPolicy,
    TrainingHealthService,
)
from gpu_fault.watcher import CompletionWatcher
from gpu_fault.xid_correlation import XidCorrelationCoordinator


LOGGER = logging.getLogger(__name__)
_pod_process_owner = pod_process_owner


def _pending_triage_deadline() -> timedelta:
    """How long a decision may sit in PENDING_TRIAGE before the watchdog closes
    it (F-G2 (4)). The service refuses a non-positive deadline; refusing it
    here names the variable the operator has to fix."""

    seconds = float(os.getenv("GPU_FAULT_PENDING_TRIAGE_DEADLINE_SECONDS", "900"))
    if seconds <= 0:
        raise ValueError("GPU_FAULT_PENDING_TRIAGE_DEADLINE_SECONDS must be positive")
    return timedelta(seconds=seconds)


class ApplicationContext:
    def __init__(
        self,
        notification_notifier=None,
        *,
        store: ControlPlaneStore | None = None,
        production_executor_config: ProductionExecutorConfig | None = None,
        production_adapters=None,
        execution_token: str | None = None,
        processor_replay_secret: str | None = None,
        dispatcher_config: WorkflowDispatcherConfig | None = None,
        fleet_registry: FleetRegistry | None = None,
        barrier_coordinator: BarrierCoordinator | None = None,
    ) -> None:
        self.store: ControlPlaneStore = store if store is not None else InMemoryStore()
        self.diagnostics = SimulatedDiagnosticAdapter(self.store)
        self.evidence = EvidenceService.from_environment(self.store)
        self.policy = GpuFaultPolicyEngine()
        self.completion = CompletionService(
            self.store,
            self.diagnostics,
            evidence_service=self.evidence,
            marker_ttl_seconds=self.policy.policy.marker_ttl_seconds,
            pending_triage_deadline=_pending_triage_deadline(),
        )
        self.xid_correlation = XidCorrelationCoordinator(
            self.store,
            self.policy,
            lambda _event, _decision: (_ for _ in ()).throw(
                RuntimeError("XID correlation finalizer is not configured")
            ),
            owner=_pod_process_owner(),
            poll_interval_seconds=float(
                os.getenv("GPU_FAULT_XID_CORRELATION_POLL_SECONDS", "1")
            ),
            lease_seconds=int(
                os.getenv("GPU_FAULT_XID_CORRELATION_LEASE_SECONDS", "30")
            ),
        )
        self.hma = HyperPodHmaNormalizer()
        self.gpu_metrics = GpuMetricsService(
            GpuMetricsThresholds.from_environment(),
            store=self.store,
        )
        self.node_health = NodeHealthPolicy(self.store)
        self.topology = WorkloadTopologyService(self.store)
        self.nvswitch_topology = NvSwitchPortTopologyService(
            self.store,
            max_age_seconds=int(
                os.getenv(
                    "GPU_FAULT_NVSWITCH_TOPOLOGY_MAX_AGE_SECONDS",
                    "300",
                )
            ),
        )
        # How stale a GPU metrics sample may be and still count as
        # evidence of the node's GPU inventory when enriching an SXID.
        #
        # This has to exceed the interval at which gpu_metric_latest
        # actually gets rewritten, which under edge filtering is the
        # health-summary period (GPU_FAULT_*_HEALTH_SUMMARY_SECONDS,
        # 300s by default) and NOT the 15s sample interval. A window
        # equal to that period leaves a dead zone at the end of every
        # cycle: measured delivery spacing is ~304s, so an event
        # arriving in the last few seconds sees every sample as stale,
        # the inventory comes back empty, and the dual-evidence gate
        # fail-closes a whole-machine reset with "requires
        # fabric_partition and complete node GPU inventory" -- which
        # reads like broken telemetry while telemetry is perfectly
        # healthy. Widening this cannot weaken the reset gate: the node
        # agent independently re-checks requested == local inventory
        # (exact equality, node_agent._reset_all_gpus_nvswitches) both
        # before and after the reset, and VERIFY_NO_GPU_CLIENTS still
        # has to pass.
        inventory_interval_seconds = int(
            os.getenv(
                "GPU_FAULT_GPU_INVENTORY_INTERVAL_SECONDS",
                "60",
            )
        )
        inventory_max_age_seconds = int(
            os.getenv(
                "GPU_FAULT_GPU_INVENTORY_MAX_AGE_SECONDS",
                "180",
            )
        )
        if inventory_max_age_seconds < inventory_interval_seconds * 2:
            raise ValueError(
                "GPU inventory max age must cover at least two "
                "inventory delivery intervals"
            )
        self.gpu_inventory_max_age = timedelta(seconds=inventory_max_age_seconds)
        self.legacy_gpu_metrics_inventory_max_age = timedelta(
            seconds=int(
                os.getenv(
                    "GPU_FAULT_LEGACY_GPU_METRICS_INVENTORY_MAX_AGE_SECONDS",
                    os.getenv(
                        "GPU_FAULT_SXID_INVENTORY_MAX_AGE_SECONDS",
                        "600",
                    ),
                )
            )
        )
        self.sxid_inventory_max_age = self.legacy_gpu_metrics_inventory_max_age
        self.training_health = TrainingHealthService(
            self.store, TrainingHealthPolicy.from_environment()
        )
        self.watcher = CompletionWatcher()
        self.orchestrator = IncidentOrchestrator(self.store)
        self.executor = SimulatedRecoveryExecutor(self.store)
        self.production_executor_config = (
            production_executor_config
            or ProductionExecutorConfig(
                enabled=False,
                executor_id="disabled",
                allowed_operations=frozenset(),
            )
        )
        self.workflow_executor = ProductionWorkflowExecutor(
            self.store,
            production_adapters or [],
            self.production_executor_config,
        )
        self.workflow_executor.fleet_registry = fleet_registry
        self.workflow_executor.branch_escalator = _branch_escalator(self.orchestrator)
        self.execution_token = execution_token
        self.processor_replay_secret = processor_replay_secret
        self.fleet_registry = fleet_registry
        self.barrier_coordinator = barrier_coordinator
        self.spare_health_controller = None
        self.control_record_archiver = None
        self.hyperpod_identity_registry = None
        self.hyperpod_identity_registries = []
        self.regional_mode = False
        self.dispatcher = WorkflowDispatcher(
            self.store,
            self.workflow_executor,
            dispatcher_config
            or WorkflowDispatcherConfig(
                enabled=False,
                poll_interval_seconds=5,
                batch_size=100,
            ),
            failure_handler=(self.orchestrator.escalate_failed_hardware_remediation),
        )
        self.advisory_notifications = AdvisoryNotificationService(
            self.store,
            notification_notifier or notification_notifier_from_environment(),
        )
        if not self.production_executor_config.enabled:
            self.store.save_profile(default_simulated_profile())

    @property
    def executor_mode(self) -> str:
        return (
            "active" if self.production_executor_config.enabled else "simulation-only"
        )

    @classmethod
    def from_environment(cls) -> ApplicationContext:
        validate_gpu_fault_environment(process_name="gpu-fault-api")
        settings = ControlPlaneSettings.from_mapping(os.environ)
        config = settings.executor
        if not config.enabled:
            return cls(production_executor_config=config)
        context = cls._base_context(settings, config)
        cls._configure_agent_registry(context, settings)
        (
            adapters,
            managed_owners,
            regional_remote_adapter,
            kubernetes_adapter,
            hp,
            enable_hyperpod_adapter,
            enable_spare_failover,
            enable_managed_observer,
        ) = cls._base_adapters(context, settings)
        managed_observer = cls._managed_observer(
            context,
            settings,
            managed_owners,
            regional_remote_adapter,
            kubernetes_adapter,
            hp,
            enable_managed_observer,
        )
        cls._add_hyperpod_adapters(
            context,
            settings,
            adapters,
            managed_owners,
            managed_observer,
            hp,
            kubernetes_adapter,
            enable_hyperpod_adapter,
            enable_spare_failover,
        )
        context.workflow_executor = ProductionWorkflowExecutor(
            context.store,
            adapters,
            config,
            notification_sender=context.advisory_notifications.send,
        )
        context.workflow_executor.fleet_registry = context.fleet_registry
        context.workflow_executor.branch_escalator = _branch_escalator(
            context.orchestrator
        )
        context.dispatcher = WorkflowDispatcher(
            context.store,
            context.workflow_executor,
            WorkflowDispatcherConfig.from_environment(config.enabled),
            failure_handler=(context.orchestrator.escalate_failed_hardware_remediation),
        )
        # Review item 3: the timing knobs of executor, dispatcher, node-action
        # verification and branch escalation only make sense in a fixed order;
        # a violation refuses to start, a warning is logged once here.
        for line in validate_timing_from_environment(os.environ):
            LOGGER.warning(line)
        return context

    @classmethod
    def _base_context(cls, settings, config):
        store_settings = settings.store
        assert store_settings is not None
        store_url = store_settings.url
        store_kind = store_settings.kind
        execution_token = settings.execution_token
        assert execution_token is not None
        store = (
            SqliteStore(store_settings.sqlite_path)
            if store_kind == "sqlite"
            else PostgresStore(
                store_url,
                pool_min_size=(store_settings.postgres_pool_min_size),
                pool_max_size=(store_settings.postgres_pool_max_size),
                pool_timeout_seconds=(store_settings.postgres_pool_timeout_seconds),
                initialize_schema=(store_settings.postgres_auto_schema_init),
            )
        )
        context = cls(
            store=store,
            production_executor_config=config,
            execution_token=execution_token,
            processor_replay_secret=settings.processor_replay_secret,
        )
        retention_days = settings.control_record_retention_days
        archive_uri = settings.control_record_archive_uri
        if retention_days > 0:
            context.control_record_archiver = ControlRecordArchiver(
                store_url,
                archive_uri,
                retention=timedelta(days=retention_days),
            )
        # An active control plane executes real recovery actions, so a
        # step that fails, a command no executor claims, or a workflow
        # that stalls has to reach a human. Path A (SES) is the only
        # channel the control plane itself owns; path B (metrics -> AMP ->
        # SNS) lives in an optional collector this process cannot see. The
        # deployed configuration had ALLOW_EMAIL=false, the dispatcher off
        # and the collector at replicas=0, which means no alert existed
        # anywhere -- a silently unexecuted recovery was indistinguishable
        # from a healthy one. Fail closed unless the operator states in
        # the manifest that alerting is handled outside this process.
        if not context.advisory_notifications.delivers_externally():
            if not env_bool("GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL"):
                raise RuntimeError(
                    "active executor has no external alert channel: "
                    + context.advisory_notifications.describe_delivery_mode()
                    + ". Configure GPU_FAULT_EMAIL_SENDER/"
                    "GPU_FAULT_EMAIL_RECIPIENTS with "
                    "GPU_FAULT_ALLOW_EMAIL=true (and either "
                    "GPU_FAULT_NOTIFICATION_DISPATCHER_ENABLED=true or "
                    "GPU_FAULT_NOTIFICATION_ASYNC_DELIVERY=false), or "
                    "set GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL=true to "
                    "declare that alerting is handled outside this "
                    "process (for example by the ADOT/AMP path, which "
                    "must then be scaled above zero replicas)"
                )
            LOGGER.warning(
                "active control plane has NO external alert channel "
                "and GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL=true: a "
                "recovery that never executes will not page anyone "
                "unless the ADOT/AMP path is running"
            )
        context.regional_mode = settings.regional_mode
        if context.regional_mode:
            sync_regional_cluster_registry(
                context.store,
                settings.regional_cluster_values,
            )
        if settings.quick_diagnostics_enabled:
            try:
                from kubernetes import client, config as kube_config
                from kubernetes.config.config_exception import (
                    ConfigException,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "quick diagnostics requires the collectors extra"
                ) from exc
            try:
                kube_config.load_incluster_config()
            except ConfigException:
                kube_config.load_kube_config()
            context.diagnostics = KubernetesDcgmDiagnosticAdapter(
                context.store,
                client.CoreV1Api(),
                dcgm_port=int(os.getenv("GPU_FAULT_DCGM_EXPORTER_PORT", "9400")),
                timeout_seconds=int(
                    os.getenv(
                        "GPU_FAULT_QUICK_DIAGNOSTIC_TIMEOUT_SECONDS",
                        "10",
                    )
                ),
            )
        context.completion = CompletionService(
            context.store,
            context.diagnostics,
            evidence_service=context.evidence,
            workflow_compiler=PassiveWorkflowCompiler(
                context.store,
                evidence_owner=settings.evidence_owner,
            ),
            marker_ttl_seconds=context.policy.policy.marker_ttl_seconds,
            pending_triage_deadline=_pending_triage_deadline(),
        )
        return context

    @staticmethod
    def _configure_agent_registry(context, settings) -> None:
        agent_settings = settings.agent_registry
        if agent_settings.enabled:
            context.fleet_registry = FleetRegistry(
                context.store,
                agent_settings.registration_secret,
                FleetCompatibilityPolicy(
                    require_tls=agent_settings.endpoint_require_tls,
                    required_agent_protocol_version=(
                        agent_settings.required_agent_protocol_version
                    ),
                    compatible_agent_protocol_versions=(
                        agent_settings.compatible_agent_protocol_versions
                    ),
                    required_node_action_key_version=(
                        agent_settings.required_node_action_key_version
                    ),
                    required_agent_version=(agent_settings.required_agent_version),
                    required_artifact_sha256=(agent_settings.required_artifact_sha256),
                    compatible_artifact_sha256s=(
                        agent_settings.compatible_artifact_sha256s
                    ),
                    required_compatibility_digest=(
                        agent_settings.required_compatibility_digest
                    ),
                    compatible_compatibility_digests=(
                        agent_settings.compatible_compatibility_digests
                    ),
                    required_policy_version=(agent_settings.required_policy_version),
                    required_runtime_profile_version=(
                        agent_settings.required_runtime_profile_version
                    ),
                    required_config_digest=(agent_settings.required_config_digest),
                    compatible_config_digests=(
                        agent_settings.compatible_config_digests
                    ),
                    max_heartbeat_age_seconds=(
                        agent_settings.max_heartbeat_age_seconds
                    ),
                ),
                # The advertised endpoint is where signed node actions
                # get POSTed, so it is restricted to the heartbeating
                # node's own name or private address on the agent's
                # port. Widen only for a deployment whose agents are
                # reached through some other name.
                endpoint_allowed_ports=(agent_settings.endpoint_allowed_ports),
                endpoint_allowed_host_suffixes=(
                    agent_settings.endpoint_allowed_host_suffixes
                ),
                # Set this to the node subnets. Unset, any RFC1918
                # address in the VPC is accepted, because a HyperPod
                # node id encodes no address to compare against -- so a
                # node holding the cluster secret can point the control
                # plane at a neighbour's agent port or at a service
                # ClusterIP. A malformed CIDR raises here at startup
                # rather than fencing the fleet one heartbeat at a time.
                endpoint_allowed_networks=parse_endpoint_networks(
                    agent_settings.endpoint_allowed_cidrs
                ),
                endpoint_allowed_networks_by_cluster={
                    registration.cluster_id: parse_endpoint_networks(
                        ",".join(registration.agent_endpoint_allowed_cidrs)
                    )
                    for registration in context.store.list_regional_clusters()
                },
                node_secrets=node_action_secrets_from_environment(),
            )
            context.barrier_coordinator = BarrierCoordinator(context.store)

    @staticmethod
    def _base_adapters(context, settings):
        adapters = [
            ControlPlaneEvidenceAdapter(settings.evidence_owner),
            GpuValidationAdapter(
                context.gpu_metrics,
                store=context.store,
                owner=os.getenv(
                    "GPU_FAULT_VALIDATION_OWNER",
                    "gpu-fault-validation-adapter",
                ),
                max_sample_age=timedelta(
                    seconds=int(
                        os.getenv(
                            "GPU_FAULT_GPU_METRICS_SILENT_AFTER_SECONDS",
                            "420",
                        )
                    )
                ),
                post_action_max_sample_age=timedelta(
                    seconds=int(
                        os.getenv(
                            "GPU_FAULT_POST_ACTION_VALIDATION_MAX_SAMPLE_AGE_SECONDS",
                            "420",
                        )
                    )
                ),
                require_rdma=env_bool("GPU_FAULT_VALIDATION_REQUIRE_RDMA"),
                temperature_warning_grace=timedelta(
                    seconds=int(
                        os.getenv(
                            "GPU_FAULT_TEMPERATURE_WARNING_GRACE_SECONDS",
                            "120",
                        )
                    )
                ),
                transient_warning_grace=timedelta(
                    seconds=int(
                        os.getenv(
                            "GPU_FAULT_TRANSIENT_GPU_WARNING_GRACE_SECONDS",
                            "120",
                        )
                    )
                ),
            ),
            SupportEscalationAdapter(
                context.store,
                owner=os.getenv(
                    "GPU_FAULT_SUPPORT_ESCALATION_OWNER",
                    "gpu-fault-support-escalation",
                ),
                alert_sender=context.advisory_notifications.send,
            ),
        ]
        managed_owners = set(settings.managed_recovery_owners)
        regional_remote_adapter = None
        if context.regional_mode:
            remote_owners = set(settings.remote_execution_owners)
            regional_remote_adapter = RegionalRemoteWorkflowAdapter(
                context.store, owners=remote_owners
            )
            adapters.append(regional_remote_adapter)
        if not context.regional_mode and settings.node_action_adapter_enabled:
            adapters.append(
                NodeActionWorkflowAdapter.from_environment(
                    registry=context.fleet_registry,
                    barriers=context.barrier_coordinator,
                    store=context.store,
                    alert_sender=(context.advisory_notifications.send),
                )
            )
        kubernetes_adapter = None
        if settings.kubernetes_adapter_enabled:
            kubernetes_adapter = KubernetesWorkflowAdapter(
                owner=os.getenv(
                    "GPU_FAULT_KUBERNETES_OWNER",
                    "gpu-fault-kubernetes-adapter",
                ),
                store=context.store,
                alert_sender=context.advisory_notifications.send,
            )
            adapters.append(kubernetes_adapter)
        enable_hyperpod_adapter = settings.hyperpod_adapter_enabled
        enable_spare_failover = settings.spare_failover_enabled
        enable_managed_observer = (
            (context.regional_mode or bool(os.getenv("GPU_FAULT_HYPERPOD_CLUSTER")))
            and env_bool("GPU_FAULT_ENABLE_HYPERPOD_MANAGED_OBSERVER", True)
            and "hyperpod-managed-node-recovery" in managed_owners
        )
        hp = (
            HyperPodLifecycleAdapter(
                HyperPodAdapterConfig.from_environment(),
                # Reboot/replace idempotency must outlive this replica:
                # a failover between the batch submission and the
                # workflow's next poll would otherwise re-submit it.
                store=context.store,
            )
            if enable_hyperpod_adapter
            or (enable_managed_observer and not context.regional_mode)
            else None
        )
        return (
            adapters,
            managed_owners,
            regional_remote_adapter,
            kubernetes_adapter,
            hp,
            enable_hyperpod_adapter,
            enable_spare_failover,
            enable_managed_observer,
        )

    @staticmethod
    def _managed_observer(
        context,
        settings,
        managed_owners,
        regional_remote_adapter,
        kubernetes_adapter,
        hp,
        enable_managed_observer,
    ):
        managed_observer = None
        if enable_managed_observer:
            if context.fleet_registry is None:
                raise RuntimeError(
                    "HyperPod managed recovery observer requires "
                    "GPU_FAULT_ENABLE_AGENT_REGISTRY=true"
                )
            if not context.regional_mode and kubernetes_adapter is None:
                raise RuntimeError(
                    "HyperPod managed recovery observer requires "
                    "GPU_FAULT_ENABLE_KUBERNETES_ADAPTER=true"
                )
            # The same reader the executor's per-operation waiting cap uses, so
            # the observer's window and the cap in front of it cannot drift.
            timeout = timedelta(seconds=managed_recovery_timeout_seconds(os.environ))
            if context.regional_mode:
                assert regional_remote_adapter is not None
                observers = {}
                for registration in context.store.list_regional_clusters():
                    lifecycle = HyperPodLifecycleAdapter(
                        HyperPodAdapterConfig.from_environment(
                            cluster_name=(registration.hyperpod_cluster_name),
                            region_name=registration.region,
                        )
                    )
                    identities = HyperPodIdentityRegistry(lifecycle, context.store)
                    context.hyperpod_identity_registries.append(identities)
                    observers[registration.cluster_id] = (
                        HyperPodManagedRecoveryObserver(
                            identities,
                            context.store,
                            registry=context.fleet_registry,
                            kubernetes_adapter=(regional_remote_adapter),
                            timeout=timeout,
                            alert_sender=(context.advisory_notifications.send),
                        )
                    )
                managed_observer = RegionalHyperPodManagedRecoveryObserver(observers)
            else:
                assert hp is not None
                context.hyperpod_identity_registry = HyperPodIdentityRegistry(
                    hp, context.store
                )
                managed_observer = HyperPodManagedRecoveryObserver(
                    context.hyperpod_identity_registry,
                    context.store,
                    registry=context.fleet_registry,
                    kubernetes_adapter=kubernetes_adapter,
                    timeout=timeout,
                    alert_sender=(context.advisory_notifications.send),
                )
        return managed_observer

    @staticmethod
    def _add_hyperpod_adapters(
        context,
        settings,
        adapters,
        managed_owners,
        managed_observer,
        hp,
        kubernetes_adapter,
        enable_hyperpod_adapter,
        enable_spare_failover,
    ) -> None:
        if managed_owners:
            adapters.append(
                ManagedRecoveryObserverAdapter(managed_owners, managed_observer)
            )
        if enable_hyperpod_adapter:
            assert hp is not None
            spare_coordinator = None
            if enable_spare_failover:
                if kubernetes_adapter is None:
                    raise ValueError(
                        "HyperPod spare failover requires the "
                        "Kubernetes workflow adapter"
                    )
                if context.fleet_registry is None:
                    raise ValueError(
                        "HyperPod spare failover requires the agent registry"
                    )
                spare_coordinator = HyperPodSpareCoordinator(
                    hp,
                    context.store,
                    kubernetes_adapter.core,
                    registry=context.fleet_registry,
                    spare_label=os.getenv(
                        "GPU_FAULT_HYPERPOD_SPARE_LABEL",
                        "gpu-fault.io/spare",
                    ),
                    spare_label_value=os.getenv(
                        "GPU_FAULT_HYPERPOD_SPARE_LABEL_VALUE",
                        "true",
                    ),
                    alert_sender=(context.advisory_notifications.send),
                )
                context.spare_health_controller = HyperPodSpareHealthController(
                    spare_coordinator,
                    context.orchestrator,
                    context.store,
                    failure_threshold=int(
                        os.getenv(
                            "GPU_FAULT_SPARE_HEALTH_FAILURE_THRESHOLD",
                            "2",
                        )
                    ),
                    unavailable_recheck_seconds=float(
                        os.getenv(
                            "GPU_FAULT_SPARE_UNAVAILABLE_RECHECK_SECONDS",
                            "3600",
                        )
                    ),
                    unavailable_alert_seconds=float(
                        os.getenv(
                            "GPU_FAULT_SPARE_UNAVAILABLE_ALERT_SECONDS",
                            "86400",
                        )
                    ),
                    alert_sender=(context.advisory_notifications.send),
                )
            adapters.append(
                HyperPodLifecycleStepAdapter(
                    hp,
                    owner=os.getenv(
                        "GPU_FAULT_HYPERPOD_OWNER",
                        "gpu-fault-hyperpod-adapter",
                    ),
                    registry=context.fleet_registry,
                    spare_coordinator=spare_coordinator,
                    kubernetes_adapter=kubernetes_adapter,
                    store=context.store,
                    alert_sender=(context.advisory_notifications.send),
                    post_reboot_stabilization_seconds=int(
                        os.getenv(
                            "GPU_FAULT_HYPERPOD_POST_REBOOT_STABILIZATION_SECONDS",
                            "60",
                        )
                    ),
                )
            )


def default_simulated_profile() -> EffectiveRuntimeProfile:
    executable = [
        CapabilityName.SCHEDULER_DRAIN,
        CapabilityName.GPU_RESET,
        CapabilityName.NODE_REBOOT,
        CapabilityName.NODE_REPLACE,
        CapabilityName.DEEP_DIAGNOSTICS,
        CapabilityName.WORKLOAD_STOP,
        CapabilityName.WORKLOAD_RESTART,
        CapabilityName.CHECKPOINT_RESTORE,
        CapabilityName.EVIDENCE_CAPTURE,
        CapabilityName.DIAGNOSTIC_BUNDLE_CAPTURE,
        CapabilityName.VM_RESTART,
        CapabilityName.FABRIC_MANAGER_RESTART,
        CapabilityName.NVLINK_DIAGNOSTICS,
        CapabilityName.MEMORY_DIAGNOSTICS,
        CapabilityName.FABRIC_RESET,
        CapabilityName.MECHANICAL_INSPECTION,
        CapabilityName.DRIVER_REMEDIATION,
        CapabilityName.EFA_DRIVER_REMEDIATION,
        CapabilityName.SOFTWARE_FIRMWARE_UPDATE,
        CapabilityName.SUPPORT_ESCALATION,
    ]
    profile = RuntimeProfile(
        cluster_id="development",
        environment=Environment.KUBERNETES,
        profile_version="simulated-v1",
        claims=[
            CapabilityClaim(
                capability=capability,
                mode=CapabilityMode.OWN,
                owner="simulated-runtime",
                adapter="simulated",
            )
            for capability in executable
        ],
        observed=[
            ObservedCapability(
                capability=capability,
                owner="simulated-runtime",
                available=True,
                version="v1",
            )
            for capability in executable
        ],
    )
    return compile_runtime_profile(profile)


def _branch_escalator(orchestrator: IncidentOrchestrator) -> BranchEscalator:
    """F-N1: a failed node branch of a job workflow escalates in place."""

    rungs = int(os.getenv("GPU_FAULT_BRANCH_ESCALATION_MAX_RUNGS", "2"))
    if rungs < 1:
        raise ValueError("GPU_FAULT_BRANCH_ESCALATION_MAX_RUNGS must be at least 1")
    return BranchEscalator(
        orchestrator._brancher,
        orchestrator.compile_branch_steps,
        max_rungs=rungs,
    )
