from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Callable
from urllib import request as urllib_request
from urllib.error import HTTPError

from gpu_fault import __version__
from gpu_fault.collector_requirements import (
    COLLECTOR_SYSTEMD_UNITS,
    CollectorServiceState,
)

from gpu_fault.fleet import (
    AgentHeartbeat,
    CURRENT_AGENT_PROTOCOL_VERSION,
    NODE_ACTION_KEY_VERSION_SHARED,
    SignedAgentHeartbeat,
    sign_agent_heartbeat,
)
from gpu_fault.installation_inventory import (
    installed_unit_report,
    read_installed_systemd_units,
)
from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.config import agent_config_digest
from gpu_fault.node_agent.executor import NodeActionExecutor
from gpu_fault.policy import load_xid_policy

LOGGER = logging.getLogger(__name__)


def _collector_units() -> list[str]:
    return sorted(set(COLLECTOR_SYSTEMD_UNITS.values()))


class CollectorServiceStates:
    """Collector unit states, with the enablement answer cached per unit.

    Every heartbeat used to spawn two ``systemctl`` calls per collector unit,
    in sequence, each with a 5 s timeout. While a quiesce holds kubelet down
    systemd answers slowly, so one report took tens of seconds -- and the
    interval is measured *after* the report, so the next tick slipped by the
    same amount.

    ``is-active`` is the live signal and is still asked every tick.
    ``is-enabled`` changes only when a unit is installed or removed, and the
    installer restarts this agent when it does, so a process-lifetime cache
    would already be correct; it is refreshed anyway every ``refresh_every``
    ticks, so a hand-run ``systemctl disable`` is reported within one refresh
    window. A call that fails is never cached, and a unit the cache has no
    answer for -- a failed call, or a unit the installer just added -- is
    asked about on its own: one slow ``systemctl`` must not put the whole
    node's enablement back on the tick that is already running late.
    """

    def __init__(
        self,
        *,
        units: Callable[[], list[str]] = _collector_units,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        refresh_every: int = 30,
    ) -> None:
        self._units = units
        self._runner = runner
        self._refresh_every = max(1, refresh_every)
        self._enabled: dict[str, str] = {}
        self._ticks = 0

    def __call__(self) -> dict[str, CollectorServiceState]:
        units = self._units()
        refresh = self._ticks % self._refresh_every == 0
        self._ticks += 1
        result: dict[str, CollectorServiceState] = {}
        cached: dict[str, str] = {}
        for unit in units:
            active = self._query(["systemctl", "is-active", unit])
            enabled = None if refresh else self._enabled.get(unit)
            if enabled is None:
                # A unit systemd knows nothing about answers "unknown" and
                # that is cached too, so it is not re-queried every tick.
                enabled = self._query(["systemctl", "is-enabled", unit])
            if enabled is not None:
                cached[unit] = enabled
            result[unit] = CollectorServiceState(
                active=active or "unknown",
                enabled=enabled or "unknown",
            )
        self._enabled = cached
        return result

    def _query(self, argv: list[str]) -> str | None:
        """The trimmed answer, or None when the call itself failed."""

        try:
            completed = self._runner(
                argv,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return (completed.stdout or "").strip() or "unknown"


class XidPolicyVersion:
    """The XID policy's mapping version, re-read only when the file changes.

    The policy document was parsed *and* validated on every heartbeat tick for
    a value that changes only when the file is replaced. The packaged catalog
    cannot change under a running process, so it is read once; a file given by
    path is re-read when its mtime or size moves, and an unreadable path is
    handed to the loader so it reports the failure the way it always did.
    """

    def __init__(
        self,
        path: str | None,
        *,
        loader: Callable[[str | None], Any] = load_xid_policy,
    ) -> None:
        self._path = path
        self._loader = loader
        self._stamp: tuple[int, int] | None = None
        self._version: str | None = None

    def __call__(self) -> str:
        reload_now, stamp = self._policy_file_stamp()
        if not reload_now and self._version is not None and stamp == self._stamp:
            return self._version
        version = str(self._loader(self._path).mapping_version)
        self._stamp = stamp
        self._version = version
        return version

    def _policy_file_stamp(self) -> tuple[bool, tuple[int, int] | None]:
        if self._path is None:
            return False, None
        try:
            info = os.stat(self._path)
        except OSError:
            return True, None
        return False, (info.st_mtime_ns, info.st_size)


def _read_boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def heartbeat_reporter_from_environment(
    executor: NodeActionExecutor,
) -> AgentHeartbeatReporter | None:
    control_plane_url = os.getenv("GPU_FAULT_NODE_CONTROL_PLANE_URL", "").strip()
    cluster_id = os.getenv("GPU_FAULT_NODE_CLUSTER_ID", "").strip()
    if not control_plane_url and not cluster_id:
        return None
    if not control_plane_url or not cluster_id:
        raise ValueError("node heartbeat requires control plane URL and cluster ID")
    artifact_sha256 = os.getenv("GPU_FAULT_NODE_ARTIFACT_SHA256", "").strip()
    compatibility_digest = os.getenv(
        "GPU_FAULT_NODE_COMPATIBILITY_DIGEST",
        artifact_sha256,
    ).strip()
    runtime_profile = os.getenv("GPU_FAULT_NODE_RUNTIME_PROFILE_VERSION", "").strip()
    if not artifact_sha256 or not runtime_profile:
        raise ValueError(
            "node heartbeat requires artifact SHA-256 and runtime profile version"
        )
    node_id = os.getenv("NODE_NAME") or socket.getfqdn()
    port = int(os.getenv("GPU_FAULT_NODE_AGENT_PORT", "9099"))
    endpoint = os.getenv(
        "GPU_FAULT_NODE_ADVERTISE_URL",
        (
            f"https://{socket.getfqdn()}:{port}"
            if os.getenv("GPU_FAULT_NODE_AGENT_TLS_CERT")
            else f"http://{socket.getfqdn()}:{port}"
        ),
    )
    certificate_path = os.getenv("GPU_FAULT_NODE_AGENT_TLS_CERT", "").strip()
    tls_certificate_pem = (
        Path(certificate_path).read_text(encoding="ascii") if certificate_path else None
    )
    policy_path = os.getenv("GPU_FAULT_XID_POLICY_PATH", "").strip() or None

    policy_version_provider = XidPolicyVersion(policy_path)
    policy_version = policy_version_provider()
    config_digest = agent_config_digest(executor, runtime_profile)
    boot_id = _read_boot_id()
    node_instance_id = (
        os.getenv("GPU_FAULT_NODE_INSTANCE_ID")
        or os.getenv("NODE_UID")
        or os.getenv("EC2_INSTANCE_ID")
    )
    if not boot_id and not node_instance_id:
        raise ValueError("node heartbeat requires a boot ID or node instance ID")
    incarnation_payload = {
        "cluster_id": cluster_id,
        "node_id": node_id,
        "node_instance_id": node_instance_id,
        "boot_id": boot_id,
    }
    agent_incarnation_id = hashlib.sha256(
        json.dumps(
            incarnation_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return AgentHeartbeatReporter(
        control_plane_url=control_plane_url,
        secret=executor.secret,
        cluster_id=cluster_id,
        node_id=node_id,
        endpoint=endpoint,
        tls_certificate_pem=tls_certificate_pem,
        agent_version=__version__,
        artifact_sha256=artifact_sha256,
        compatibility_digest=compatibility_digest,
        installer_bundle_sha256=(
            os.getenv("GPU_FAULT_NODE_INSTALLER_BUNDLE_SHA256") or None
        ),
        installer_template_sha256=(
            os.getenv("GPU_FAULT_NODE_INSTALLER_TEMPLATE_SHA256") or None
        ),
        policy_version=policy_version,
        policy_version_provider=policy_version_provider,
        runtime_profile_version=runtime_profile,
        config_digest=config_digest,
        allowed_operations=sorted(
            executor.allowed_operations, key=lambda item: item.value
        ),
        collector_status_provider=CollectorServiceStates(),
        installed_units_provider=read_installed_systemd_units,
        boot_id=boot_id,
        node_action_key_version=(executor.node_action_key_version),
        node_instance_id=node_instance_id,
        agent_incarnation_id=agent_incarnation_id,
        generation_sink=executor.set_agent_generation,
        bearer_token=(
            os.getenv("GPU_FAULT_NODE_CONTROL_PLANE_TOKEN")
            or os.getenv("GPU_FAULT_CONTROL_PLANE_TOKEN")
        ),
        interval_seconds=float(
            os.getenv("GPU_FAULT_NODE_HEARTBEAT_INTERVAL_SECONDS", "30")
        ),
    )


class AgentHeartbeatRejected(RuntimeError):
    """The control plane refused the heartbeat with an HTTP error.

    ``urlopen`` raises before anyone reads the response body, so the
    reason the control plane gave -- for example ``agent incarnation has
    been retired`` on a REVOKED agent, which never clears by itself --
    used to be discarded. A node fenced that way logged nothing but a
    bare ``HTTP Error 409: Conflict`` traceback every interval, which is
    how ``hyperpod-i-00000000000000001`` sat unreachable from
    2026-08-12 to 2026-08-16 without anyone noticing.
    """

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"control plane rejected heartbeat ({status}): {detail}")
        self.status = status
        self.detail = detail


class AgentHeartbeatReporter:
    def __init__(
        self,
        *,
        control_plane_url: str,
        secret: str,
        cluster_id: str,
        node_id: str,
        endpoint: str,
        tls_certificate_pem: str | None = None,
        agent_version: str,
        artifact_sha256: str,
        compatibility_digest: str | None = None,
        installer_bundle_sha256: str | None = None,
        installer_template_sha256: str | None = None,
        policy_version: str,
        runtime_profile_version: str,
        config_digest: str,
        allowed_operations: list[WorkflowOperation],
        boot_id: str | None,
        node_action_key_version: int = (NODE_ACTION_KEY_VERSION_SHARED),
        policy_version_provider: Callable[[], str] | None = None,
        collector_status_provider: Callable[[], dict[str, CollectorServiceState]]
        | None = None,
        installed_units_provider: Callable[[], list[str]] | None = None,
        node_instance_id: str | None = None,
        agent_incarnation_id: str | None = None,
        generation_sink=None,
        interval_seconds: float = 30,
        bearer_token: str | None = None,
        sender=None,
        now=None,
    ) -> None:
        if interval_seconds < 5:
            raise ValueError("agent heartbeat interval must be at least five seconds")
        self.control_plane_url = control_plane_url.rstrip("/")
        self.secret = secret
        self.cluster_id = cluster_id
        self.node_id = node_id
        self.endpoint = endpoint
        self.tls_certificate_pem = tls_certificate_pem
        self.agent_version = agent_version
        self.artifact_sha256 = artifact_sha256
        self.compatibility_digest = compatibility_digest or artifact_sha256
        self.installer_bundle_sha256 = installer_bundle_sha256
        self.installer_template_sha256 = installer_template_sha256
        self.policy_version = policy_version
        self.policy_version_provider = policy_version_provider
        self.collector_status_provider = collector_status_provider
        self.installed_units_provider = installed_units_provider
        self._reported_installed_units_digest: str | None = None
        self.runtime_profile_version = runtime_profile_version
        self.config_digest = config_digest
        self.allowed_operations = allowed_operations
        self.boot_id = boot_id
        self.node_action_key_version = node_action_key_version
        self.node_instance_id = node_instance_id
        self.agent_incarnation_id = agent_incarnation_id
        self.generation_sink = generation_sink
        self.interval_seconds = interval_seconds
        self.bearer_token = bearer_token
        self.sender = sender or (
            lambda url, envelope: self._send(
                url,
                envelope,
                cluster_id=self.cluster_id,
                bearer_token=self.bearer_token,
            )
        )
        self.now = now or (lambda: datetime.now(timezone.utc))
        # Local health view for /healthz: the control plane already tracks
        # staleness, so this is informational and never gates the agent.
        self.last_success_at: datetime | None = None
        self.consecutive_failures = 0

    def health_snapshot(self, *, now: datetime | None = None) -> dict[str, Any]:
        reference = now or self.now()
        age = (
            (reference - self.last_success_at).total_seconds()
            if self.last_success_at is not None
            else None
        )
        return {
            "configured": True,
            "consecutive_failures": self.consecutive_failures,
            "last_success_at": (
                self.last_success_at.isoformat()
                if self.last_success_at is not None
                else None
            ),
            "last_success_age_seconds": age,
        }

    def report_once(self):
        policy_version = (
            self.policy_version_provider()
            if self.policy_version_provider is not None
            else self.policy_version
        )
        unit_report = None
        unit_digest = None
        if self.installed_units_provider is not None:
            full_report = installed_unit_report(
                self.installed_units_provider(),
                include_units=True,
            )
            unit_digest = full_report.digest
            unit_report = (
                full_report
                if unit_digest != self._reported_installed_units_digest
                else installed_unit_report(
                    full_report.units or [],
                    include_units=False,
                )
            )
        heartbeat = AgentHeartbeat(
            cluster_id=self.cluster_id,
            node_id=self.node_id,
            endpoint=self.endpoint,
            tls_certificate_pem=self.tls_certificate_pem,
            agent_protocol_version=CURRENT_AGENT_PROTOCOL_VERSION,
            node_action_key_version=self.node_action_key_version,
            agent_version=self.agent_version,
            artifact_sha256=self.artifact_sha256,
            compatibility_digest=self.compatibility_digest,
            installer_bundle_sha256=self.installer_bundle_sha256,
            installer_template_sha256=self.installer_template_sha256,
            policy_version=policy_version,
            runtime_profile_version=self.runtime_profile_version,
            config_digest=self.config_digest,
            allowed_operations=self.allowed_operations,
            collector_services=(
                self.collector_status_provider()
                if self.collector_status_provider is not None
                else {}
            ),
            installed_unit_report=unit_report,
            boot_id=self.boot_id,
            node_instance_id=self.node_instance_id,
            agent_incarnation_id=self.agent_incarnation_id,
            observed_at=self.now(),
        )
        envelope = SignedAgentHeartbeat(
            heartbeat=heartbeat,
            signature=sign_agent_heartbeat(heartbeat, self.secret),
        )
        try:
            response = self.sender(
                self.control_plane_url,
                envelope,
            )
        except AgentHeartbeatRejected as exc:
            if "full installed unit inventory is required" in exc.detail:
                self._reported_installed_units_digest = None
            raise
        if unit_digest is not None:
            self._reported_installed_units_digest = unit_digest
        if (
            self.generation_sink is not None
            and isinstance(response, dict)
            and isinstance(response.get("generation"), int)
        ):
            self.generation_sink(response["generation"])
        return response

    def run(self, stop: Event) -> None:
        refused = 0
        while not stop.is_set():
            try:
                self.report_once()
                refused = 0
                self.consecutive_failures = 0
                self.last_success_at = self.now()
            except AgentHeartbeatRejected as exc:
                refused += 1
                self.consecutive_failures += 1
                if exc.status in {403, 409}:
                    # These never clear on their own: the agent keeps
                    # sending the same identity the control plane just
                    # rejected. Say so on one line instead of dumping a
                    # urllib traceback that names no cause.
                    LOGGER.error(
                        "node agent heartbeat refused %d time(s) in a "
                        "row: %s -- a REVOKED agent or retired "
                        "incarnation does not recover by retrying; "
                        "reactivate the agent or reboot the node",
                        refused,
                        exc,
                    )
                else:
                    LOGGER.error("node agent heartbeat failed: %s", exc)
            except Exception:
                self.consecutive_failures += 1
                LOGGER.exception("node agent heartbeat failed")
            stop.wait(self.interval_seconds)

    @staticmethod
    def _send(
        control_plane_url: str,
        envelope: SignedAgentHeartbeat,
        *,
        cluster_id: str,
        bearer_token: str | None,
    ) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "X-GPU-Fault-Cluster-ID": cluster_id,
        }
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        request = urllib_request.Request(
            control_plane_url + "/v1/fleet/agents/heartbeat",
            data=envelope.model_dump_json().encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=10) as response:
                body = response.read()
                return json.loads(body) if body else {}
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2048]
            raise AgentHeartbeatRejected(exc.code, detail) from exc
