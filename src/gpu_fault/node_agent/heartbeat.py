from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from threading import Event
from typing import Any, Callable
from urllib import request as urllib_request
from urllib.error import HTTPError

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


def collector_service_states() -> dict[str, CollectorServiceState]:
    result = {}
    for unit in sorted(set(COLLECTOR_SYSTEMD_UNITS.values())):
        try:
            active = subprocess.run(
                ["systemctl", "is-active", unit],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            enabled = subprocess.run(
                ["systemctl", "is-enabled", unit],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            result[unit] = CollectorServiceState(
                active=(active.stdout or "").strip() or "unknown",
                enabled=(enabled.stdout or "").strip() or "unknown",
            )
        except (OSError, subprocess.TimeoutExpired):
            result[unit] = CollectorServiceState(
                active="unknown",
                enabled="unknown",
            )
    return result


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
    runtime_profile = os.getenv("GPU_FAULT_NODE_RUNTIME_PROFILE_VERSION", "").strip()
    if not artifact_sha256 or not runtime_profile:
        raise ValueError(
            "node heartbeat requires artifact SHA-256 and runtime profile version"
        )
    node_id = os.getenv("NODE_NAME") or socket.getfqdn()
    port = int(os.getenv("GPU_FAULT_NODE_AGENT_PORT", "9099"))
    endpoint = os.getenv(
        "GPU_FAULT_NODE_ADVERTISE_URL",
        f"http://{socket.getfqdn()}:{port}",
    )
    policy_path = os.getenv("GPU_FAULT_XID_POLICY_PATH", "").strip() or None

    def policy_version_provider() -> str:
        return load_xid_policy(policy_path).mapping_version

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
        agent_version=version("gpu-fault-control-plane"),
        artifact_sha256=artifact_sha256,
        policy_version=policy_version,
        policy_version_provider=policy_version_provider,
        runtime_profile_version=runtime_profile,
        config_digest=config_digest,
        allowed_operations=sorted(
            executor.allowed_operations, key=lambda item: item.value
        ),
        collector_status_provider=collector_service_states,
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
        agent_version: str,
        artifact_sha256: str,
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
        self.agent_version = agent_version
        self.artifact_sha256 = artifact_sha256
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
            agent_protocol_version=CURRENT_AGENT_PROTOCOL_VERSION,
            node_action_key_version=self.node_action_key_version,
            agent_version=self.agent_version,
            artifact_sha256=self.artifact_sha256,
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
            except AgentHeartbeatRejected as exc:
                refused += 1
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
