from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

from gpu_fault.env import env_bool
from gpu_fault.execution import ProductionExecutorConfig
from gpu_fault.fleet import CURRENT_AGENT_PROTOCOL_VERSION


def _csv(
    values: Mapping[str, str],
    name: str,
    default: str = "",
) -> frozenset[str]:
    return frozenset(
        item.strip() for item in values.get(name, default).split(",") if item.strip()
    )


@dataclass(frozen=True)
class StoreSettings:
    url: str
    kind: str
    sqlite_path: str | None
    postgres_pool_min_size: int
    postgres_pool_max_size: int
    postgres_pool_timeout_seconds: float
    postgres_auto_schema_init: bool

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> StoreSettings:
        url = values.get("GPU_FAULT_STORE_URL", "")
        if url.startswith("sqlite:///"):
            kind = "sqlite"
            sqlite_path = url.removeprefix("sqlite:///")
        elif url.startswith(("postgresql://", "postgres://")):
            kind = "postgres"
            sqlite_path = None
        else:
            raise RuntimeError(
                "active executor requires GPU_FAULT_STORE_URL="
                "sqlite:///path for one replica or a PostgreSQL URL "
                "for multiple replicas"
            )
        minimum = int(values.get("GPU_FAULT_POSTGRES_POOL_MIN_SIZE", "1"))
        maximum = int(values.get("GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "8"))
        if minimum < 0 or maximum < 1 or minimum > maximum:
            raise ValueError("PostgreSQL pool sizes must satisfy 0 <= min <= max")
        return cls(
            url=url,
            kind=kind,
            sqlite_path=sqlite_path,
            postgres_pool_min_size=minimum,
            postgres_pool_max_size=maximum,
            postgres_pool_timeout_seconds=float(
                values.get("GPU_FAULT_POSTGRES_POOL_TIMEOUT_SECONDS", "2")
            ),
            # Default OFF: schema creation/alteration is a privileged, one-time
            # migration concern, not something every API/worker replica should
            # do on boot. Auto-init defaulting to True let any process with the
            # store URL create or alter tables; the deploy/migration paths that
            # legitimately need it set GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT
            # explicitly (the generated regional configs already ship "false").
            postgres_auto_schema_init=env_bool(
                "GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT", False, environ=values
            ),
        )


@dataclass(frozen=True)
class AgentRegistrySettings:
    enabled: bool
    registration_secret: str
    required_agent_protocol_version: int
    compatible_agent_protocol_versions: frozenset[int]
    required_node_action_key_version: int | None
    required_agent_version: str | None
    required_artifact_sha256: str
    compatible_artifact_sha256s: frozenset[str]
    required_compatibility_digest: str
    compatible_compatibility_digests: frozenset[str]
    required_policy_version: str | None
    required_runtime_profile_version: str | None
    required_config_digest: str
    compatible_config_digests: frozenset[str]
    max_heartbeat_age_seconds: int
    endpoint_allowed_ports: frozenset[int]
    endpoint_allowed_host_suffixes: tuple[str, ...]
    endpoint_allowed_cidrs: str
    endpoint_require_tls: bool

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> AgentRegistrySettings:
        enabled = env_bool("GPU_FAULT_ENABLE_AGENT_REGISTRY", False, environ=values)
        artifact = values.get("GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256", "").strip()
        config_digest = values.get("GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST", "").strip()
        compatibility_digest = (
            values.get("GPU_FAULT_REQUIRED_AGENT_COMPATIBILITY_DIGEST") or artifact
        ).strip()
        if enabled and (not artifact or not config_digest):
            raise RuntimeError(
                "agent registry requires non-empty "
                "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256 and "
                "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST pins"
            )
        raw_key_version = values.get(
            "GPU_FAULT_REQUIRED_NODE_ACTION_KEY_VERSION", ""
        ).strip()
        key_version = int(raw_key_version) if raw_key_version else None
        if key_version not in {None, 1, 2}:
            raise ValueError(
                "GPU_FAULT_REQUIRED_NODE_ACTION_KEY_VERSION must be 1 or 2"
            )
        protocol_version = int(
            values.get(
                "GPU_FAULT_REQUIRED_AGENT_PROTOCOL_VERSION",
                str(CURRENT_AGENT_PROTOCOL_VERSION),
            )
        )
        if protocol_version < 1:
            raise ValueError(
                "GPU_FAULT_REQUIRED_AGENT_PROTOCOL_VERSION must be positive"
            )
        compatible_protocol_versions = frozenset(
            int(value)
            for value in _csv(
                values,
                "GPU_FAULT_COMPATIBLE_AGENT_PROTOCOL_VERSIONS",
            )
        ) - {protocol_version}
        if any(value < 1 for value in compatible_protocol_versions):
            raise ValueError(
                "GPU_FAULT_COMPATIBLE_AGENT_PROTOCOL_VERSIONS "
                "must contain positive integers"
            )
        compatible_artifacts = frozenset(
            value.lower()
            for value in _csv(
                values,
                "GPU_FAULT_COMPATIBLE_AGENT_ARTIFACT_SHA256S",
            )
        ) - {artifact.lower()}
        compatible_compatibility = frozenset(
            value.lower()
            for value in _csv(
                values,
                "GPU_FAULT_COMPATIBLE_AGENT_COMPATIBILITY_DIGESTS",
            )
        ) - {compatibility_digest.lower()}
        compatible_config_digests = frozenset(
            value.lower()
            for value in _csv(
                values,
                "GPU_FAULT_COMPATIBLE_AGENT_CONFIG_DIGESTS",
            )
        ) - {config_digest.lower()}
        ports = frozenset(
            int(item.strip())
            for item in values.get(
                "GPU_FAULT_AGENT_ENDPOINT_ALLOWED_PORTS", "9099"
            ).split(",")
            if item.strip()
        )
        if enabled and not ports:
            raise ValueError("agent registry requires an allowed endpoint port")
        return cls(
            enabled=enabled,
            registration_secret=values.get(
                "GPU_FAULT_AGENT_REGISTRATION_SECRET",
                values.get("GPU_FAULT_NODE_ACTION_SECRET", ""),
            ),
            required_agent_protocol_version=protocol_version,
            compatible_agent_protocol_versions=(compatible_protocol_versions),
            required_node_action_key_version=key_version,
            required_agent_version=(
                values.get("GPU_FAULT_REQUIRED_AGENT_VERSION") or None
            ),
            required_artifact_sha256=artifact,
            compatible_artifact_sha256s=compatible_artifacts,
            required_compatibility_digest=compatibility_digest,
            compatible_compatibility_digests=compatible_compatibility,
            required_policy_version=(
                values.get("GPU_FAULT_REQUIRED_POLICY_VERSION") or None
            ),
            required_runtime_profile_version=(
                values.get("GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION") or None
            ),
            required_config_digest=config_digest,
            compatible_config_digests=compatible_config_digests,
            max_heartbeat_age_seconds=int(
                values.get(
                    "GPU_FAULT_AGENT_MAX_HEARTBEAT_AGE_SECONDS",
                    "90",
                )
            ),
            endpoint_allowed_ports=ports,
            endpoint_allowed_host_suffixes=tuple(
                item.strip()
                for item in values.get(
                    "GPU_FAULT_AGENT_ENDPOINT_ALLOWED_HOST_SUFFIXES",
                    "",
                ).split(",")
                if item.strip()
            ),
            endpoint_allowed_cidrs=values.get(
                "GPU_FAULT_AGENT_ENDPOINT_ALLOWED_CIDRS", ""
            ).strip(),
            endpoint_require_tls=env_bool(
                "GPU_FAULT_AGENT_ENDPOINT_REQUIRE_TLS", True, environ=values
            ),
        )


@dataclass(frozen=True)
class ControlPlaneSettings:
    executor: ProductionExecutorConfig
    store: StoreSettings | None
    execution_token: str | None
    processor_replay_secret: str | None
    regional_mode: bool
    regional_cluster_values: tuple[dict, ...]
    quick_diagnostics_enabled: bool
    node_action_adapter_enabled: bool
    kubernetes_adapter_enabled: bool
    hyperpod_adapter_enabled: bool
    spare_failover_enabled: bool
    managed_recovery_owners: frozenset[str]
    remote_execution_owners: frozenset[str]
    agent_registry: AgentRegistrySettings
    control_record_retention_days: int
    control_record_archive_uri: str
    evidence_owner: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> ControlPlaneSettings:
        executor = ProductionExecutorConfig.from_mapping(values)
        registry = AgentRegistrySettings.from_mapping(values)
        if not executor.enabled:
            return cls(
                executor=executor,
                store=None,
                execution_token=None,
                processor_replay_secret=None,
                regional_mode=False,
                regional_cluster_values=(),
                quick_diagnostics_enabled=False,
                node_action_adapter_enabled=False,
                kubernetes_adapter_enabled=False,
                hyperpod_adapter_enabled=False,
                spare_failover_enabled=False,
                managed_recovery_owners=frozenset(),
                remote_execution_owners=frozenset(),
                agent_registry=registry,
                control_record_retention_days=0,
                control_record_archive_uri="",
                evidence_owner="gpu-fault-control-plane",
            )
        store = StoreSettings.from_mapping(values)
        execution_token = values.get("GPU_FAULT_EXECUTION_TOKEN", "")
        if len(execution_token) < 32:
            raise RuntimeError(
                "active executor requires a random "
                "GPU_FAULT_EXECUTION_TOKEN of at least 32 characters"
            )
        if not executor.allowed_operations:
            raise RuntimeError(
                "active executor requires an explicit "
                "GPU_FAULT_ALLOWED_OPERATIONS allowlist"
            )
        # The two modes are not equivalent in security: regional enforces
        # per-cluster tenancy/authorization that single-cluster relaxes. The old
        # ``== "regional"`` test treated *every* other value -- including a typo
        # like "regionl" or an empty string -- as single-cluster, silently
        # downgrading to the less isolated mode. Parse it explicitly so an
        # unrecognised value fails closed instead. An unset value still resolves
        # to single-cluster, but that path additionally requires the operator to
        # opt in with GPU_FAULT_ALLOW_SINGLE_CLUSTER=true below, so the insecure
        # mode is never selected without an explicit choice.
        deployment_mode = (
            values.get("GPU_FAULT_DEPLOYMENT_MODE", "single-cluster").strip().lower()
        )
        if deployment_mode not in ("regional", "single-cluster"):
            raise RuntimeError(
                "GPU_FAULT_DEPLOYMENT_MODE must be 'regional' or "
                f"'single-cluster'; refusing to guess from {deployment_mode!r} "
                "(an unrecognised value must not silently select the less "
                "isolated single-cluster mode)"
            )
        regional_mode = deployment_mode == "regional"
        raw_clusters = values.get("GPU_FAULT_REGIONAL_CLUSTERS_JSON", "")
        cluster_values: tuple[dict, ...] = ()
        if raw_clusters:
            try:
                parsed = json.loads(raw_clusters)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "GPU_FAULT_REGIONAL_CLUSTERS_JSON is invalid"
                ) from exc
            if not isinstance(parsed, list):
                raise RuntimeError("regional cluster registry must be a list")
            if not all(isinstance(item, dict) for item in parsed):
                raise RuntimeError("regional cluster registry entries must be objects")
            cluster_values = tuple(dict(item) for item in parsed)
        if regional_mode:
            if not raw_clusters:
                raise RuntimeError(
                    "regional mode requires GPU_FAULT_REGIONAL_CLUSTERS_JSON"
                )
            if values.get("GPU_FAULT_HYPERPOD_CLUSTER"):
                raise RuntimeError(
                    "regional mode uses the cluster registry; "
                    "GPU_FAULT_HYPERPOD_CLUSTER must be unset"
                )
            if registry.enabled:
                missing_cidrs = sorted(
                    str(item.get("cluster_id") or "<unknown>")
                    for item in cluster_values
                    if not item.get("agent_endpoint_allowed_cidrs")
                )
                if missing_cidrs:
                    raise RuntimeError(
                        "regional cluster registrations require "
                        "agent_endpoint_allowed_cidrs: " + ", ".join(missing_cidrs)
                    )
        else:
            if len(cluster_values) > 1:
                raise RuntimeError(
                    "single-cluster mode cannot load multiple regional "
                    "cluster registrations"
                )
            if not env_bool("GPU_FAULT_ALLOW_SINGLE_CLUSTER", False, environ=values):
                raise RuntimeError(
                    "single-cluster executor mode is Canary-only; set "
                    "GPU_FAULT_ALLOW_SINGLE_CLUSTER=true explicitly"
                )
        quick_diagnostics = env_bool(
            "GPU_FAULT_ENABLE_QUICK_DIAGNOSTICS", False, environ=values
        )
        node_action = env_bool(
            "GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER", False, environ=values
        )
        kubernetes = env_bool(
            "GPU_FAULT_ENABLE_KUBERNETES_ADAPTER", False, environ=values
        )
        hyperpod = env_bool("GPU_FAULT_ENABLE_HYPERPOD_ADAPTER", False, environ=values)
        if regional_mode and quick_diagnostics:
            raise RuntimeError(
                "regional control plane cannot run in-cluster "
                "quick diagnostics against its own EKS"
            )
        if regional_mode and kubernetes:
            raise RuntimeError(
                "regional control plane must not enable the "
                "in-cluster KubernetesWorkflowAdapter"
            )
        if regional_mode and hyperpod:
            raise RuntimeError(
                "regional control plane must delegate HyperPod "
                "mutations to the target cluster executor"
            )
        remote_owners = _csv(
            values,
            "GPU_FAULT_REMOTE_EXECUTION_OWNERS",
            (
                "gpu-fault-kubernetes-adapter,"
                "gpu-fault-node-agent,"
                "gpu-fault-hyperpod-adapter"
            ),
        )
        if regional_mode and not remote_owners:
            raise RuntimeError("regional mode requires remote execution owners")
        retention_days = int(values.get("GPU_FAULT_CONTROL_RECORD_RETENTION_DAYS", "0"))
        archive_uri = values.get("GPU_FAULT_CONTROL_RECORD_ARCHIVE_S3_URI", "").strip()
        if retention_days > 0 and (store.kind != "postgres" or not archive_uri):
            raise RuntimeError(
                "control record retention requires PostgreSQL "
                "and GPU_FAULT_CONTROL_RECORD_ARCHIVE_S3_URI"
            )
        return cls(
            executor=executor,
            store=store,
            execution_token=execution_token,
            processor_replay_secret=(
                values.get("GPU_FAULT_PROCESSOR_REPLAY_SECRET") or None
            ),
            regional_mode=regional_mode,
            regional_cluster_values=cluster_values,
            quick_diagnostics_enabled=quick_diagnostics,
            node_action_adapter_enabled=node_action,
            kubernetes_adapter_enabled=kubernetes,
            hyperpod_adapter_enabled=hyperpod,
            spare_failover_enabled=env_bool(
                "GPU_FAULT_ENABLE_HYPERPOD_SPARE_FAILOVER", False, environ=values
            ),
            managed_recovery_owners=_csv(
                values,
                "GPU_FAULT_MANAGED_RECOVERY_OWNERS",
                ("hyperpod-managed-node-recovery,hyperpod-managed-job-recovery"),
            ),
            remote_execution_owners=remote_owners,
            agent_registry=registry,
            control_record_retention_days=retention_days,
            control_record_archive_uri=archive_uri,
            evidence_owner=values.get(
                "GPU_FAULT_EVIDENCE_OWNER",
                "gpu-fault-control-plane",
            ),
        )
