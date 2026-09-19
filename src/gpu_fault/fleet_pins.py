"""Process-local snapshot of the fleet pins, re-read from the release ConfigMap.

Every pin -- the agent and executor artifact digests, protocol versions, config
digests and the node-action key version -- reaches a control-plane process as
environment the kubelet resolves from ``gpu-fault-release-metadata`` once, at
Pod start. That is why a release that changed only data-plane artifacts still
rolled ``gpu-fault-api-ha``, ``gpu-fault-control-worker`` and the spool worker
twice (stage and finalize, ~5 min of deploy #20): the roll existed only to make
the Pods re-read pins the ConfigMap already held.

``FleetPinRuntime`` polls that ConfigMap through the Kubernetes API, keeps an
immutable snapshot (``.data``, its digest, the resourceVersion, a generation)
and swaps the three derived policies atomically. The ConfigMap stays the single
system of record: ``deploy/control-plane/base/control-plane-deployment.yaml``
keeps the same keys as ``configMapKeyRef`` env, so a Pod that started on the
new ConfigMap and a Pod that hot-reloaded it hold the same pins.

Unlike the regional registry runtime, pins never gate readiness: a stale
snapshot is the previously validated pin set, so a read or validation failure
keeps serving it and is reported in ``status()["error"]``, never treated as
fatal.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event, RLock, Thread
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from gpu_fault.fleet_compatibility import FleetCompatibilityPolicy
from gpu_fault.regional_compatibility import RegionalExecutorCompatibilityPolicy
from gpu_fault.remote_step_batching import RemoteStepBatchingPolicy
from gpu_fault.settings import AgentRegistrySettings

if TYPE_CHECKING:
    from gpu_fault.fleet import FleetRegistry

LOGGER = logging.getLogger(__name__)

CONFIG_MAP_ENV = "GPU_FAULT_FLEET_PIN_CONFIGMAP"
POLL_SECONDS_ENV = "GPU_FAULT_FLEET_PIN_POLL_SECONDS"
NAMESPACE_ENV = "GPU_FAULT_NAMESPACE"
DEFAULT_CONFIG_MAP = "gpu-fault-release-metadata"
DEFAULT_NAMESPACE = "gpu-fault-system"
DEFAULT_POLL_SECONDS = 2.0
# One read may not park the poll thread on a hung API server.
READ_TIMEOUT_SECONDS = 10
THREAD_NAME = "gpu-fault-fleet-pins"

SOURCE_ENVIRONMENT = "environment"
SOURCE_CONFIGMAP = "configmap"

# ConfigMap key -> environment name: exactly the ``configMapKeyRef`` table in
# deploy/control-plane/base/control-plane-deployment.yaml. The runtime maps the
# ConfigMap through this table and derives the policies with the same
# ``from_mapping`` constructors start-up uses, so validation lives in one place.
CONFIG_MAP_KEY_ENVIRONMENT: Mapping[str, str] = MappingProxyType(
    {
        "required-agent-artifact-sha256": "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256",
        "compatible-agent-artifact-sha256s": (
            "GPU_FAULT_COMPATIBLE_AGENT_ARTIFACT_SHA256S"
        ),
        "required-agent-compatibility-digest": (
            "GPU_FAULT_REQUIRED_AGENT_COMPATIBILITY_DIGEST"
        ),
        "compatible-agent-compatibility-digests": (
            "GPU_FAULT_COMPATIBLE_AGENT_COMPATIBILITY_DIGESTS"
        ),
        "required-agent-protocol-version": "GPU_FAULT_REQUIRED_AGENT_PROTOCOL_VERSION",
        "compatible-agent-protocol-versions": (
            "GPU_FAULT_COMPATIBLE_AGENT_PROTOCOL_VERSIONS"
        ),
        "required-regional-executor-protocol-version": (
            "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION"
        ),
        "compatible-regional-executor-protocol-versions": (
            "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS"
        ),
        "required-regional-executor-artifact-sha256": (
            "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256"
        ),
        "compatible-regional-executor-artifact-sha256s": (
            "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S"
        ),
        "required-regional-executor-compatibility-digest": (
            "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST"
        ),
        "compatible-regional-executor-compatibility-digests": (
            "GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_COMPATIBILITY_DIGESTS"
        ),
        "required-agent-config-digest": "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST",
        "compatible-agent-config-digests": "GPU_FAULT_COMPATIBLE_AGENT_CONFIG_DIGESTS",
        "required-node-action-key-version": (
            "GPU_FAULT_REQUIRED_NODE_ACTION_KEY_VERSION"
        ),
    }
)
PIN_ENVIRONMENT_NAMES = frozenset(CONFIG_MAP_KEY_ENVIRONMENT.values())

# The ``FleetCompatibilityPolicy`` fields the ConfigMap governs. Everything
# else on the policy (TLS, agent version, policy and profile versions, the
# heartbeat age, required operations) is start-up configuration and stays as
# ``_configure_agent_registry`` built it.
AGENT_PIN_FIELDS = (
    "required_agent_protocol_version",
    "compatible_agent_protocol_versions",
    "required_node_action_key_version",
    "required_artifact_sha256",
    "compatible_artifact_sha256s",
    "required_compatibility_digest",
    "compatible_compatibility_digests",
    "required_config_digest",
    "compatible_config_digests",
)

ConfigMapReader = Callable[[], tuple[Mapping[str, str], str | None]]


def pin_content_sha256(data: Mapping[str, str]) -> str:
    """Digest of a ConfigMap's ``.data``; byte-identical to ``jq -cS '.data'``.

    The release tooling stamps the same digest on the Deployments and waits
    for every Pod's ``/healthz`` to report it, so the rendering here is fixed:
    sorted keys, no whitespace, UTF-8 without ASCII escaping.
    """

    rendered = json.dumps(
        dict(data), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def pin_environment(
    startup_environment: Mapping[str, str], data: Mapping[str, str]
) -> Mapping[str, str]:
    """``startup_environment`` with every pin variable taken from ``data``.

    A key the ConfigMap does not carry unsets its variable rather than keeping
    the Pod's start-up value: finalize empties the ``compatible-*`` keys, and a
    process that kept the stage-time value would go on admitting the previous
    artifact after the window closed.
    """

    environment = {
        name: value
        for name, value in startup_environment.items()
        if name not in PIN_ENVIRONMENT_NAMES
    }
    for key, name in CONFIG_MAP_KEY_ENVIRONMENT.items():
        if key in data:
            environment[name] = data[key]
    return MappingProxyType(environment)


@dataclass(frozen=True)
class FleetPinSnapshot:
    data: Mapping[str, str]
    content_sha256: str
    resource_version: str | None
    observed_at: datetime | None
    generation: int
    source: str


@dataclass(frozen=True)
class FleetPinPolicies:
    """The three policies one pin set derives to; swapped as a unit."""

    environment: Mapping[str, str]
    agent_pins: FleetCompatibilityPolicy
    executor: RegionalExecutorCompatibilityPolicy
    step_batching: RemoteStepBatchingPolicy


def derive_policies(
    startup_environment: Mapping[str, str], data: Mapping[str, str]
) -> FleetPinPolicies:
    """Every policy the pins govern, through the constructors start-up uses.

    Raises exactly where start-up would for the same values, so a malformed
    ConfigMap is refused before anything is swapped.
    """

    environment = pin_environment(startup_environment, data)
    agent = AgentRegistrySettings.from_mapping(environment)
    # Empty pins read as "none declared" here; the registry, when enabled,
    # has already refused an empty required artifact or config digest above.
    agent_pins = FleetCompatibilityPolicy(
        required_agent_protocol_version=agent.required_agent_protocol_version,
        compatible_agent_protocol_versions=agent.compatible_agent_protocol_versions,
        required_node_action_key_version=agent.required_node_action_key_version,
        required_artifact_sha256=agent.required_artifact_sha256 or None,
        compatible_artifact_sha256s=agent.compatible_artifact_sha256s,
        required_compatibility_digest=agent.required_compatibility_digest or None,
        compatible_compatibility_digests=agent.compatible_compatibility_digests,
        required_config_digest=agent.required_config_digest or None,
        compatible_config_digests=agent.compatible_config_digests,
    )
    return FleetPinPolicies(
        environment=environment,
        agent_pins=agent_pins,
        executor=RegionalExecutorCompatibilityPolicy.from_mapping(environment),
        step_batching=RemoteStepBatchingPolicy.from_environment(environment),
    )


def kubernetes_config_map_reader(name: str, namespace: str) -> ConfigMapReader:
    """The default reader: the ConfigMap through the in-cluster API.

    Outside a cluster ``load_kube_config`` is used only when ``KUBECONFIG`` is
    set explicitly. The test suite deletes that variable for every test, and a
    developer box usually has ``~/.kube/config`` pointing at a live site, so an
    implicit fallback would let a test that runs the app lifespan poll
    production. The API client is built on first use and dropped after a
    failure, so rotated credentials are picked up on the next poll.
    """

    state: dict[str, Any] = {}

    def read() -> tuple[Mapping[str, str], str | None]:
        api = state.get("api")
        if api is None:
            from kubernetes import client, config
            from kubernetes.config.config_exception import ConfigException

            try:
                config.load_incluster_config()
            except ConfigException:
                if not os.environ.get("KUBECONFIG"):
                    raise
                config.load_kube_config()
            api = client.CoreV1Api()
            state["api"] = api
        try:
            body = api.read_namespaced_config_map(
                name, namespace, _request_timeout=READ_TIMEOUT_SECONDS
            )
        except Exception:
            state.pop("api", None)
            raise
        data: dict[str, str] = dict(body.data or {})
        metadata = body.metadata
        resource_version: str | None = (
            None if metadata is None else metadata.resource_version
        )
        return data, resource_version

    return read


class FleetPinRuntime:
    """Immutable pin snapshot with hot-swapped derived policies."""

    def __init__(
        self,
        startup_environment: Mapping[str, str],
        *,
        config_map: str,
        namespace: str,
        poll_seconds: float,
        reader: ConfigMapReader | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError(f"{POLL_SECONDS_ENV} must be positive")
        self.startup_environment: Mapping[str, str] = MappingProxyType(
            dict(startup_environment)
        )
        self.config_map = config_map
        self.namespace = namespace
        self.poll_seconds = float(poll_seconds)
        self.now = now or (lambda: datetime.now(timezone.utc))
        self._reader: ConfigMapReader = reader or kubernetes_config_map_reader(
            config_map, namespace
        )
        self._lock = RLock()
        self._stop = Event()
        self._thread: Thread | None = None
        self._last_error: str | None = None
        self._fleet_registry: FleetRegistry | None = None
        self._fleet_registry_base_policy: FleetCompatibilityPolicy | None = None
        # Snapshot 1 is the environment this process started with: the same
        # ConfigMap, resolved by the kubelet. It is what the process serves
        # until the first successful poll, and what it keeps serving if the
        # ConfigMap can never be read (RBAC, API outage).
        data = {
            key: self.startup_environment[name]
            for key, name in CONFIG_MAP_KEY_ENVIRONMENT.items()
            if name in self.startup_environment
        }
        self._policies = derive_policies(self.startup_environment, data)
        self._snapshot = FleetPinSnapshot(
            data=MappingProxyType(dict(data)),
            content_sha256=pin_content_sha256(data),
            resource_version=None,
            observed_at=None,
            generation=1,
            source=SOURCE_ENVIRONMENT,
        )

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        reader: ConfigMapReader | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> FleetPinRuntime:
        values = os.environ if environment is None else environment
        raw = values.get(POLL_SECONDS_ENV, "").strip()
        try:
            poll_seconds = float(raw) if raw else DEFAULT_POLL_SECONDS
        except ValueError as exc:
            raise ValueError(f"{POLL_SECONDS_ENV} must be a number") from exc
        return cls(
            values,
            config_map=values.get(CONFIG_MAP_ENV, "").strip() or DEFAULT_CONFIG_MAP,
            namespace=values.get(NAMESPACE_ENV, "").strip() or DEFAULT_NAMESPACE,
            poll_seconds=poll_seconds,
            reader=reader,
            now=now,
        )

    def snapshot(self) -> FleetPinSnapshot:
        with self._lock:
            return self._snapshot

    def environment(self) -> Mapping[str, str]:
        """The start-up environment with the served pins laid over it."""

        with self._lock:
            return self._policies.environment

    def status(self) -> dict[str, Any]:
        with self._lock:
            snapshot, error = self._snapshot, self._last_error
        return {
            "source": snapshot.source,
            "config_map": self.config_map,
            "content_sha256": snapshot.content_sha256,
            "resource_version": snapshot.resource_version,
            "observed_at": (
                snapshot.observed_at.isoformat()
                if snapshot.observed_at is not None
                else None
            ),
            "generation": snapshot.generation,
            "error": error,
        }

    def agent_policy(self, base: FleetCompatibilityPolicy) -> FleetCompatibilityPolicy:
        """``base`` with the pinned fields replaced by the served snapshot's."""

        with self._lock:
            pins = self._policies.agent_pins
        return base.model_copy(
            update={name: getattr(pins, name) for name in AGENT_PIN_FIELDS}
        )

    def executor_policy(self) -> RegionalExecutorCompatibilityPolicy:
        with self._lock:
            return self._policies.executor

    def step_batching_policy(self) -> RemoteStepBatchingPolicy:
        with self._lock:
            return self._policies.step_batching

    def bind_fleet_registry(self, registry: FleetRegistry) -> None:
        """Let ``registry.policy`` follow the pins; ``fleet.py`` reads it per call.

        The policy the registry was built with stays the base: only the fields
        in ``AGENT_PIN_FIELDS`` follow the ConfigMap, now and on every apply.
        """

        with self._lock:
            self._fleet_registry = registry
            self._fleet_registry_base_policy = registry.policy
            registry.policy = self.agent_policy(registry.policy)

    def apply(
        self, data: Mapping[str, str], resource_version: str | None = None
    ) -> FleetPinSnapshot:
        """Serve ``data``, a ConfigMap's ``.data``; the poller's and tests' entry.

        A changed digest derives and swaps the three policies, bumps the
        generation and logs once. The same digest only confirms the snapshot
        (source, resourceVersion, observation time). Validation raises to the
        caller before anything is swapped; ``refresh_once`` records it as
        ``error`` instead of raising.
        """

        digest = pin_content_sha256(data)
        observed = self.now()
        with self._lock:
            current = self._snapshot
            changed = digest != current.content_sha256
            if changed:
                policies = derive_policies(self.startup_environment, data)
                self._policies = policies
                if self._fleet_registry is not None:
                    base = (
                        self._fleet_registry_base_policy or self._fleet_registry.policy
                    )
                    self._fleet_registry.policy = self.agent_policy(base)
            snapshot = FleetPinSnapshot(
                data=MappingProxyType(dict(data)),
                content_sha256=digest,
                resource_version=resource_version,
                observed_at=observed,
                generation=current.generation + 1 if changed else current.generation,
                source=SOURCE_CONFIGMAP,
            )
            self._snapshot = snapshot
            recovered = self._last_error is not None
            self._last_error = None
        if changed:
            LOGGER.info(
                "fleet pins applied: generation=%d content_sha256=%s "
                "resource_version=%s config_map=%s/%s",
                snapshot.generation,
                snapshot.content_sha256,
                snapshot.resource_version,
                self.namespace,
                self.config_map,
            )
        elif recovered:
            LOGGER.info(
                "fleet pin refresh recovered: generation=%d content_sha256=%s",
                snapshot.generation,
                snapshot.content_sha256,
            )
        return snapshot

    def refresh_once(self) -> bool:
        """One poll: read, apply. A failure keeps the snapshot and sets ``error``."""

        try:
            data, resource_version = self._reader()
            self.apply(data, resource_version)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            with self._lock:
                repeated = message == self._last_error
                self._last_error = message
                generation = self._snapshot.generation
            if not repeated:
                LOGGER.warning(
                    "fleet pin refresh from ConfigMap %s/%s failed; serving "
                    "generation %d: %s",
                    self.namespace,
                    self.config_map,
                    generation,
                    message,
                )
            return False
        return True

    def run(self, stop: Event) -> None:
        while not stop.is_set():
            self.refresh_once()
            if stop.wait(self.poll_seconds):
                break

    def start(self) -> Thread:
        """Start the daemon poll thread; a second call returns the running one."""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self._thread
            self._stop = Event()
            self._thread = Thread(
                target=self.run,
                args=(self._stop,),
                name=THREAD_NAME,
                daemon=True,
            )
            thread = self._thread
        thread.start()
        return thread

    def stop(self, timeout: float | None = None) -> None:
        with self._lock:
            thread, self._thread = self._thread, None
            self._stop.set()
        if thread is not None:
            thread.join(
                self.poll_seconds + READ_TIMEOUT_SECONDS if timeout is None else timeout
            )

    def wrap_lifespan(self, lifespan: Callable[[Any], Any]) -> Callable[[Any], Any]:
        """Poll for exactly the life of the wrapped application lifespan."""

        @asynccontextmanager
        async def wrapped(app: Any) -> AsyncIterator[None]:
            self.start()
            try:
                async with lifespan(app):
                    yield
            finally:
                await asyncio.to_thread(self.stop)

        return wrapped
