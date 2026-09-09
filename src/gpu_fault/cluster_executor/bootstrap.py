"""Process bootstrap for the regional executor Pod.

The two console scripts (``gpu-fault-cluster-executor`` and its readiness
probe) land here: ``executor_from_environment`` wires the adapters, the
regional proxies and the executor from ``GPU_FAULT_*``, ``readiness_probe``
asks the control plane whether this executor is useful, and ``main`` runs the
claim loop with the SIGTERM handler installed before the first claim.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from datetime import datetime, timezone

from gpu_fault.adapters import (
    HyperPodLifecycleStepAdapter,
    KubernetesWorkflowAdapter,
    NodeActionWorkflowAdapter,
)
from gpu_fault.aws_errors import missing_aws_credentials
from gpu_fault.dataplane_metrics import (
    HealthPredicate,
    MetricsServer,
    start_metrics_server,
)
from gpu_fault.cluster_executor.executor import (
    LIVENESS_STALE_AFTER_SECONDS,
    ClusterActionExecutor,
)
from gpu_fault.cluster_executor.lease import DEFAULT_MAX_EXECUTION_SECONDS
from gpu_fault.cluster_executor.metrics import loop_breadcrumb_is_fresh
from gpu_fault.cluster_executor.regional_client import (
    ClusterExecutorError,
    RegionalExecutorClient,
    RegionalFleetRegistry,
    RegionalHyperPodSubmissionStore,
    RegionalIncidentOwnershipProvider,
)
from gpu_fault.env import env_bool
from gpu_fault.env_validation import validate_gpu_fault_environment
from gpu_fault.hyperpod import (
    HyperPodAdapterConfig,
    HyperPodLifecycleAdapter,
)
from gpu_fault.hyperpod_spares import HyperPodSpareCoordinator
from gpu_fault.logging_setup import configure_logging
from gpu_fault.spare_reservation_sweep import SpareReservationSweep

# Deliberately the pre-split module's name and not ``__name__``: the log format
# carries ``%(name)s`` and operators filter on ``gpu_fault.cluster_executor``, so
# every layer of the package logs under the one name it always had.
LOGGER = logging.getLogger("gpu_fault.cluster_executor")


def _persistent_store_from_environment():
    store_url = os.getenv("GPU_FAULT_STORE_URL", "").strip()
    if not store_url:
        return None
    if store_url.startswith("sqlite:///"):
        from gpu_fault.store import SqliteStore

        return SqliteStore(store_url.removeprefix("sqlite:///"))
    if store_url.startswith(("postgresql://", "postgres://")):
        from gpu_fault.store import PostgresStore

        return PostgresStore(
            store_url,
            pool_min_size=int(os.getenv("GPU_FAULT_POSTGRES_POOL_MIN_SIZE", "1")),
            pool_max_size=int(os.getenv("GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "4")),
            pool_timeout_seconds=float(
                os.getenv(
                    "GPU_FAULT_POSTGRES_POOL_TIMEOUT_SECONDS",
                    "2",
                )
            ),
        )
    raise ClusterExecutorError("GPU_FAULT_STORE_URL must use sqlite:/// or PostgreSQL")


def _regional_client_from_environment(
    *, timeout_seconds: float = 15
) -> RegionalExecutorClient:
    return RegionalExecutorClient(
        os.environ["GPU_FAULT_CONTROL_PLANE_URL"],
        os.environ["GPU_FAULT_CLUSTER_ID"],
        os.environ["GPU_FAULT_CONTROL_PLANE_TOKEN"],
        timeout_seconds=timeout_seconds,
        ca_file=os.getenv("GPU_FAULT_CONTROL_PLANE_CA_FILE") or None,
        executor_artifact_sha256=(
            os.getenv("GPU_FAULT_EXECUTOR_ARTIFACT_SHA256") or None
        ),
        executor_compatibility_digest=(
            os.getenv("GPU_FAULT_EXECUTOR_COMPATIBILITY_DIGEST") or None
        ),
    )


def executor_from_environment() -> ClusterActionExecutor:
    cluster_id = os.environ["GPU_FAULT_CLUSTER_ID"]
    executor_id = os.getenv(
        "GPU_FAULT_CLUSTER_EXECUTOR_ID",
        f"{cluster_id}/{socket.gethostname()}",
    )
    use_remote_state = env_bool("GPU_FAULT_CLUSTER_EXECUTOR_REMOTE_STATE", True)
    store = None if use_remote_state else _persistent_store_from_environment()
    regional_client = _regional_client_from_environment()
    fleet_registry = RegionalFleetRegistry(regional_client)
    spare_coordinator: HyperPodSpareCoordinator | None = None
    kubernetes_adapter = KubernetesWorkflowAdapter(
        owner=os.getenv(
            "GPU_FAULT_KUBERNETES_OWNER",
            "gpu-fault-kubernetes-adapter",
        ),
        store=store,
        notification_sink=(fleet_registry if store is None else store),
        evidence_sink=(fleet_registry if store is None else None),
        workload_log_tail_lines=int(
            os.getenv("GPU_FAULT_WORKLOAD_LOG_TAIL_LINES", "2000")
        ),
        workload_log_max_bytes=int(
            os.getenv("GPU_FAULT_WORKLOAD_LOG_MAX_BYTES", "262144")
        ),
        workload_log_s3_uri=(os.getenv("GPU_FAULT_WORKLOAD_LOG_S3_URI") or None),
        workload_log_s3_max_bytes=int(
            os.getenv(
                "GPU_FAULT_WORKLOAD_LOG_S3_MAX_BYTES",
                "104857600",
            )
        ),
        # With REMOTE_STATE the adapter has no store, so "has the
        # incident that owns this node already finished?" has to be
        # asked over the API. Without it the takeover branch never
        # fires and a node annotated by a dead workflow is permanently
        # unusable.
        ownership_provider=(
            RegionalIncidentOwnershipProvider(regional_client)
            if store is None
            else None
        ),
    )
    adapters = [kubernetes_adapter]
    hyperpod_confirm_cluster = None
    # Node agent addressing must come from fleet heartbeats, not from a
    # hand-maintained GPU_FAULT_NODE_AGENT_ENDPOINTS map: nodes are
    # replaced over a cluster's life, and a stale map fails closed only
    # at the moment a node action is dispatched, with no prior signal.
    # The control-plane wiring already passes a registry (api.py:615).
    node_action_adapter = None
    if env_bool("GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER"):
        node_action_adapter = NodeActionWorkflowAdapter.from_environment(
            registry=fleet_registry,
        )
        adapters.append(node_action_adapter)
    if env_bool("GPU_FAULT_ENABLE_HYPERPOD_ADAPTER"):
        hyperpod_confirm_cluster = os.getenv(
            "GPU_FAULT_HYPERPOD_CONFIRM_CLUSTER", ""
        ).strip()
        if not hyperpod_confirm_cluster:
            raise ClusterExecutorError(
                "GPU_FAULT_HYPERPOD_CONFIRM_CLUSTER is required "
                "when the HyperPod adapter is enabled"
            )
        # Fail at startup, not at the first real fault. Enabling the
        # adapter without the ServiceAccount's role-arn annotation used
        # to look completely healthy -- Pod Ready, logs clean -- until a
        # GPU broke hours later and REPLACE_NODE died on
        # NoCredentialsError. This costs one local credential-chain
        # resolution and no API call.
        credential_gap = missing_aws_credentials()
        if credential_gap is not None:
            raise ClusterExecutorError(
                "the HyperPod adapter is enabled but "
                + credential_gap
                + "; annotate serviceaccount "
                "gpu-fault-cluster-executor with "
                "eks.amazonaws.com/role-arn=<role> and restart, or set "
                "GPU_FAULT_ENABLE_HYPERPOD_ADAPTER=false"
            )
        lifecycle = HyperPodLifecycleAdapter(
            HyperPodAdapterConfig.from_environment(),
            # Reboot and replace restart this very Pod, so submission
            # idempotency cannot live in its memory. With REMOTE_STATE
            # the record is held by the control plane; with a local
            # store it is held there.
            store=(
                RegionalHyperPodSubmissionStore(regional_client)
                if store is None
                else store
            ),
        )
        # Reboot confirmation always needs the fleet registry to compare
        # the pre-submit boot/incarnation baseline with the new Agent
        # heartbeat. Spare failover controls only replacement
        # allocation; it must not disable the only automatic RESTART_NODE
        # confirmation path.
        registry = fleet_registry
        if registry is None:
            raise ClusterExecutorError(
                "HyperPod lifecycle execution requires a fleet "
                "registry for reboot confirmation"
            )
        if env_bool("GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER"):
            if not use_remote_state:
                raise ClusterExecutorError(
                    "regional HyperPod spare failover requires "
                    "GPU_FAULT_CLUSTER_EXECUTOR_REMOTE_STATE=true"
                )
            spare_coordinator = HyperPodSpareCoordinator(
                lifecycle,
                registry.store,
                kubernetes_adapter.core,
                registry=registry,
                remote_health_provider=registry,
                spare_label=os.getenv(
                    "GPU_FAULT_HYPERPOD_SPARE_LABEL",
                    "gpu-fault.io/spare",
                ),
                spare_label_value=os.getenv(
                    "GPU_FAULT_HYPERPOD_SPARE_LABEL_VALUE",
                    "true",
                ),
            )
        adapters.append(
            HyperPodLifecycleStepAdapter(
                lifecycle,
                owner=os.getenv(
                    "GPU_FAULT_HYPERPOD_OWNER",
                    "gpu-fault-hyperpod-adapter",
                ),
                registry=registry,
                spare_coordinator=spare_coordinator,
                node_action_adapter=node_action_adapter,
                kubernetes_adapter=kubernetes_adapter,
                store=store,
                notification_sink=(fleet_registry if store is None else store),
                post_reboot_stabilization_seconds=int(
                    os.getenv(
                        "GPU_FAULT_HYPERPOD_POST_REBOOT_STABILIZATION_SECONDS",
                        "60",
                    )
                ),
            )
        )
    namespaces = {
        value.strip()
        for value in os.getenv("GPU_FAULT_ALLOWED_WORKLOAD_NAMESPACES", "").split(",")
        if value.strip()
    }
    sweep = SpareReservationSweep(spare_coordinator) if spare_coordinator else None
    return ClusterActionExecutor(
        regional_client,
        adapters,
        executor_id=executor_id,
        allowed_namespaces=namespaces,
        spare_reservation_sweep=sweep,
        # Only set when this executor actually owns HyperPod mutations.
        # Left None otherwise so an unexpectedly routed RESTART_NODE or
        # REPLACE_NODE fails the adapter's confirmation gate instead of
        # confirming itself.
        confirm_cluster_name=(hyperpod_confirm_cluster),
        **_claim_loop_settings_from_environment(),
    )


def _claim_loop_settings_from_environment() -> dict[str, float | int]:
    """The ``GPU_FAULT_CLUSTER_EXECUTOR_*`` knobs that pace the claim loop.

    Every value is validated by ``ClusterActionExecutor`` itself, so a bad
    setting fails at startup with the executor's own message.
    """

    return {
        "poll_seconds": float(
            os.getenv("GPU_FAULT_CLUSTER_EXECUTOR_POLL_SECONDS", "2")
        ),
        # 20 s long-poll by default; 0 restores pure polling and leaves the
        # field out of the claim body for a control plane that predates it.
        "claim_wait_seconds": float(
            os.getenv("GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_WAIT_SECONDS", "20")
        ),
        "lease_seconds": int(
            os.getenv("GPU_FAULT_CLUSTER_EXECUTOR_LEASE_SECONDS", "120")
        ),
        "batch_size": int(os.getenv("GPU_FAULT_CLUSTER_EXECUTOR_BATCH_SIZE", "5")),
        "max_concurrent_commands": int(
            os.getenv("GPU_FAULT_CLUSTER_EXECUTOR_MAX_CONCURRENT_COMMANDS", "5")
        ),
        "lease_renewal_failure_limit": int(
            os.getenv("GPU_FAULT_CLUSTER_EXECUTOR_LEASE_FAILURE_LIMIT", "3")
        ),
        "transport_degraded_backoff_after": int(
            os.getenv(
                "GPU_FAULT_CLUSTER_EXECUTOR_TRANSPORT_DEGRADED_BACKOFF_AFTER", "3"
            )
        ),
        "max_execution_seconds": float(
            os.getenv(
                "GPU_FAULT_CLUSTER_EXECUTOR_MAX_EXECUTION_SECONDS",
                str(DEFAULT_MAX_EXECUTION_SECONDS),
            )
        ),
    }


def readiness_probe() -> int:
    """Authenticated readiness for the executor Pod.

    Replaces ``urlopen(CONTROL_PLANE_URL + "/healthz")``, which was
    anonymous and therefore could not fail for any executor-specific
    reason: a wrong per-cluster token, a revoked registration, a broken
    trust chain to the control plane, or a backlog whose execution owners
    this executor does not implement all left the Pod Ready and claiming
    nothing. Runs as a separate process under the probe, so the last
    successful claim is read from the breadcrumb the claim loop writes.
    """

    configure_logging()
    client = _regional_client_from_environment(
        timeout_seconds=float(
            os.getenv(
                "GPU_FAULT_CLUSTER_EXECUTOR_READINESS_TIMEOUT_SECONDS",
                "8",
            )
        ),
    )
    state_path = os.getenv(
        "GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH",
        "/tmp/executor-claim-state.json",
    )
    executor_id = os.getenv(
        "GPU_FAULT_CLUSTER_EXECUTOR_ID",
        f"{os.environ['GPU_FAULT_CLUSTER_ID']}/{socket.gethostname()}",
    )
    owners: list[str] = []
    claim_age: float | None = None
    try:
        with open(state_path, encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError):
        # No claim has completed yet (or the file is unreadable). The
        # control plane still checks registration, the token and the
        # backlog; it just cannot judge claim freshness.
        state = {}
    if isinstance(state, dict):
        executor_id = state.get("executor_id") or executor_id
        raw_owners = state.get("execution_owners")
        if isinstance(raw_owners, list):
            owners = [owner for owner in raw_owners if isinstance(owner, str)]
        claimed_at = state.get("last_successful_claim_at")
        if isinstance(claimed_at, str):
            try:
                claim_age = max(
                    0.0,
                    (
                        datetime.now(timezone.utc) - datetime.fromisoformat(claimed_at)
                    ).total_seconds(),
                )
            except ValueError:
                claim_age = None
    try:
        report = client.readiness(
            executor_id,
            execution_owners=owners,
            last_successful_claim_age_seconds=claim_age,
        )
    except Exception as exc:
        LOGGER.error("executor readiness probe failed: %s", exc)
        return 1
    if not report.get("ready"):
        LOGGER.error(
            "executor is not ready: %s",
            "; ".join(report.get("reasons") or ["unspecified"]),
        )
        return 1
    return 0


def executor_health(executor: ClusterActionExecutor) -> HealthPredicate:
    """``/healthz``: the exec liveness probe's question, asked of its file.

    The loop breadcrumb, fresher than the manifest's 300 s. Not the claim
    breadcrumb and not a control-plane call: a regional outage must read as a
    live loop that cannot claim, never as a dead Pod.
    """

    return lambda: loop_breadcrumb_is_fresh(
        executor.liveness_state_path, LIVENESS_STALE_AFTER_SECONDS
    )


def start_executor_metrics(executor: ClusterActionExecutor) -> MetricsServer | None:
    """Serve ``/metrics`` + ``/healthz`` on the manifest's ``metrics`` port.

    Best effort, before the first claim: 0 disables, a port that cannot be
    bound is one ERROR line and the claim loop runs without a scrape target
    (the collector reads ``up == 0``, which is the honest value).
    """

    return start_metrics_server(
        executor.metrics.family,
        port=int(os.getenv("GPU_FAULT_CLUSTER_EXECUTOR_METRICS_PORT", "9111")),
        health=executor_health(executor),
    )


def main() -> None:
    configure_logging()
    validate_gpu_fault_environment(process_name="gpu-fault-cluster-executor")
    executor = executor_from_environment()
    start_executor_metrics(executor)
    # Installed before the first claim: a rollout that arrives during the very
    # first cycle must still release its lease instead of parking it.
    executor.install_signal_handlers()
    executor.run()
