from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import logging
import os
import re
import socket
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.logging_setup import configure_logging

LOGGER = logging.getLogger(__name__)

INSTALLER_VERSION_ANNOTATION = "gpu-fault.io/installer-version"
INSTALLER_DIGEST_ANNOTATION = "gpu-fault.io/installer-config-digest"
INSTALLER_ARTIFACT_ANNOTATION = "gpu-fault.io/installer-artifact-sha256"
INSTALLER_BUNDLE_ANNOTATION = "gpu-fault.io/installer-bundle-sha256"
INSTALLER_TEMPLATE_ANNOTATION = "gpu-fault.io/installer-template-sha256"
INSTALLER_NODE_UID_ANNOTATION = "gpu-fault.io/installer-node-uid"
INSTALLER_STATE_ANNOTATION = "gpu-fault.io/installer-state"
# The boot the installation was recorded on. HyperPod UpdateClusterSoftware
# re-images a node in place: same Node object, same UID, every annotation kept,
# and an empty root where the Agent used to be. The UID check cannot see that;
# a boot-id change followed by an Agent that does not answer can.
INSTALLER_BOOT_ID_ANNOTATION = "gpu-fault.io/installer-boot-id"
DEFAULT_AGENT_PORT = 9099
DEFAULT_REBOOT_GRACE_SECONDS = 600
INSTALLER_JOB_LABEL = "gpu-fault.io/node-installer"

_INVENTORY = {
    "p5.4xlarge": (1, 1),
    "p5.48xlarge": (8, 32),
    "p5e.48xlarge": (8, 32),
    "p5en.48xlarge": (8, 16),
    "p6-b200.48xlarge": (8, 8),
    "p6-b300.48xlarge": (8, 16),
}


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _metadata(item: Any) -> Any:
    return _value(item, "metadata", {})


def _conditions(item: Any) -> list[Any]:
    return list(_value(_value(item, "status", {}), "conditions", []) or [])


def _is_ready(node: Any) -> bool:
    return any(
        _value(condition, "type") == "Ready"
        and str(_value(condition, "status")).lower() == "true"
        for condition in _conditions(node)
    )


def _boot_id(node: Any) -> str | None:
    info = _value(_value(node, "status"), "node_info")
    if info is None:
        info = _value(_value(node, "status"), "nodeInfo")
    value = _value(info, "boot_id") if info is not None else None
    if value is None and info is not None:
        value = _value(info, "bootID")
    return str(value) if value else None


def _ready_since(node: Any) -> datetime | None:
    for condition in _conditions(node):
        if _value(condition, "type") == "Ready":
            value = _value(condition, "last_transition_time")
            if value is None:
                value = _value(condition, "lastTransitionTime")
            if isinstance(value, datetime):
                return value if value.tzinfo else value.replace(tzinfo=UTC)
            if isinstance(value, str) and value:
                try:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    return None
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def agent_answers(address: str, port: int, *, timeout_seconds: float = 2.0) -> bool:
    """Whether something listens on the Agent port.

    A TCP connect is enough: the Agent serves HTTPS with client certificates,
    so a plain HTTP health call would be refused even by a healthy Agent, and
    the question here is only whether the installed software is running at all.
    """

    try:
        with socket.create_connection((address, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def _job_condition(job: Any, condition_type: str) -> bool:
    return any(
        _value(condition, "type") == condition_type
        and str(_value(condition, "status")).lower() == "true"
        for condition in _conditions(job)
    )


def _api_status(error: Exception) -> int | None:
    return getattr(error, "status", None)


def _job_name(node_name: str, node_uid: str, digest: str) -> str:
    suffix = hashlib.sha256(f"{node_uid}:{digest}".encode()).hexdigest()[:10]
    safe_name = re.sub(r"[^a-z0-9-]+", "-", node_name.lower()).strip("-")
    prefix = f"gpu-fault-install-{safe_name}"[: 63 - len(suffix) - 1].rstrip("-")
    return f"{prefix}-{suffix}"


def _internal_ip(node: Any) -> str:
    addresses = _value(_value(node, "status", {}), "addresses", []) or []
    for address in addresses:
        if _value(address, "type") == "InternalIP":
            return str(_value(address, "address"))
    raise ValueError("node has no InternalIP")


def _inventory(instance_type: str) -> tuple[int, int]:
    normalized = instance_type.removeprefix("ml.")
    try:
        return _INVENTORY[normalized]
    except KeyError as error:
        raise ValueError(
            f"unsupported GPU instance type: {instance_type or 'UNKNOWN'}"
        ) from error


class NodeInstallerReconciler:
    def __init__(
        self,
        core_api: Any,
        batch_api: Any,
        *,
        namespace: str,
        cluster_name: str,
        version: str,
        config_digest: str,
        artifact_sha256: str,
        bundle_sha256: str | None = None,
        template_sha256: str,
        job_template: dict[str, Any],
        dcgm_metrics_url_template: str,
        node_action_keys_secret: str = ("gpu-fault-node-action-keys"),
        retry_seconds: int = 300,
        max_unavailable: int = 1,
        job_active_deadline_seconds: int = 840,
        allowed_node_names: frozenset[str] | None = None,
        wave_config_map: str | None = None,
        now: Callable[[], datetime] | None = None,
        agent_port: int = DEFAULT_AGENT_PORT,
        reboot_grace_seconds: int = DEFAULT_REBOOT_GRACE_SECONDS,
        agent_alive: Callable[[str, int], bool] = agent_answers,
    ) -> None:
        self.core = core_api
        self.batch = batch_api
        self.namespace = namespace
        self.cluster_name = cluster_name
        self.version = version
        self.config_digest = config_digest
        if not re.fullmatch(r"[0-9a-f]{64}", artifact_sha256):
            raise ValueError(
                "installer artifact SHA-256 must be 64 lowercase hex characters"
            )
        self.artifact_sha256 = artifact_sha256
        self.bundle_sha256 = bundle_sha256 or artifact_sha256
        # The release's template-inputs digest, stamped on nodes and Jobs so a
        # template change re-installs. It is an identity pin, not an integrity
        # check of the text this process loaded -- that is
        # ``load_job_template`` -- so it has no default: a made-up value here
        # would let every node "match" a template nobody pinned.
        self.template_sha256 = template_sha256
        for name, digest in (
            ("bundle", self.bundle_sha256),
            ("template", self.template_sha256),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(
                    f"installer {name} SHA-256 must be 64 lowercase hex characters"
                )
        self.job_template = job_template
        self.dcgm_metrics_url_template = dcgm_metrics_url_template
        self.node_action_keys_secret = node_action_keys_secret
        self.retry_seconds = retry_seconds
        if max_unavailable < 1:
            raise ValueError("installer max_unavailable must be positive")
        if job_active_deadline_seconds < 60:
            raise ValueError("installer active deadline must be at least 60 seconds")
        self.max_unavailable = max_unavailable
        self.job_active_deadline_seconds = job_active_deadline_seconds
        self.allowed_node_names = allowed_node_names
        self.wave_config_map = wave_config_map
        self.now = now or (lambda: datetime.now(UTC))
        self.agent_port = agent_port
        self.reboot_grace_seconds = reboot_grace_seconds
        self.agent_alive = agent_alive

    @property
    def node_selector(self) -> str:
        return f"sagemaker.amazonaws.com/cluster-name={self.cluster_name}"

    def reconcile_once(self) -> dict[str, int]:
        result = {
            "current": 0,
            "created": 0,
            "running": 0,
            "succeeded": 0,
            "failed": 0,
            "deferred": 0,
            "not_ready": 0,
            "unsupported": 0,
            "rebooted": 0,
            "recovering": 0,
        }
        allowed_node_names, max_unavailable = self._wave_settings()
        response = self.core.list_node(label_selector=self.node_selector)
        in_flight = 0
        nodes = sorted(
            (
                item
                for item in (_value(response, "items", []) or [])
                if allowed_node_names is None
                or str(_value(_metadata(item), "name")) in allowed_node_names
            ),
            key=lambda item: str(_value(_metadata(item), "name")),
        )
        for node in nodes:
            outcome = self._reconcile_node(
                node,
                allow_create=in_flight < max_unavailable,
            )
            result[outcome] += 1
            if outcome in {"created", "running"}:
                in_flight += 1
        return result

    def _wave_settings(self) -> tuple[frozenset[str] | None, int]:
        if self.wave_config_map is None:
            return self.allowed_node_names, self.max_unavailable
        value = self.core.read_namespaced_config_map(
            self.wave_config_map,
            self.namespace,
        )
        data = dict(_value(value, "data", {}) or {})
        raw_nodes = str(data.get("allowed-nodes") or "").strip()
        if raw_nodes == "*":
            allowed_nodes = None
        else:
            allowed_nodes = frozenset(
                item.strip() for item in raw_nodes.split(",") if item.strip()
            )
        try:
            max_unavailable = int(data.get("max-unavailable") or "")
        except (TypeError, ValueError) as exc:
            raise RuntimeError("installer wave max-unavailable is invalid") from exc
        if max_unavailable < 1:
            raise RuntimeError("installer wave max-unavailable must be positive")
        return allowed_nodes, max_unavailable

    def _reconcile_node(self, node: Any, *, allow_create: bool = True) -> str:
        metadata = _metadata(node)
        node_name = str(_value(metadata, "name"))
        node_uid = str(_value(metadata, "uid"))
        annotations = dict(_value(metadata, "annotations", {}) or {})
        if not _is_ready(node):
            LOGGER.info("node %s is not Ready; installation deferred", node_name)
            return "not_ready"
        identity_matches = (
            annotations.get(INSTALLER_VERSION_ANNOTATION) == self.version
            and annotations.get(INSTALLER_DIGEST_ANNOTATION) == self.config_digest
            and annotations.get(INSTALLER_ARTIFACT_ANNOTATION) == self.artifact_sha256
            and annotations.get(INSTALLER_BUNDLE_ANNOTATION) == self.bundle_sha256
            and annotations.get(INSTALLER_TEMPLATE_ANNOTATION) == self.template_sha256
            and annotations.get(INSTALLER_NODE_UID_ANNOTATION) == node_uid
        )
        installer_state = annotations.get(INSTALLER_STATE_ANNOTATION)
        boot_id = _boot_id(node)
        if identity_matches and installer_state == "Succeeded":
            recorded_boot = annotations.get(INSTALLER_BOOT_ID_ANNOTATION)
            if boot_id is None or recorded_boot == boot_id:
                return "current"
            if recorded_boot is None:
                # Installed before boot ids were recorded: adopt this boot.
                self._mark_node(node_name, node_uid, "Succeeded", boot_id)
                return "current"
            # The node booted since the installation was recorded. A plain
            # reboot keeps the software and the Agent comes back on its own; a
            # re-image (HyperPod UpdateClusterSoftware) keeps the Node object
            # and its annotations but not /opt/gpu-fault. Only the Agent can
            # tell the two apart.
            if self.agent_alive(_internal_ip(node), self.agent_port):
                self._mark_node(node_name, node_uid, "Succeeded", boot_id)
                return "rebooted"
            ready_since = _ready_since(node)
            if (
                ready_since is not None
                and (self.now() - ready_since).total_seconds()
                < self.reboot_grace_seconds
            ):
                return "recovering"
            LOGGER.warning(
                "node %s rebooted (boot %s -> %s) and its Agent does not answer on "
                "port %s; reinstalling",
                node_name,
                recorded_boot,
                boot_id,
                self.agent_port,
            )
            self._mark_node(node_name, node_uid, "Retrying", boot_id)
            installer_state = "Retrying"

        name = _job_name(
            node_name,
            node_uid,
            (
                f"{self.config_digest}:{self.artifact_sha256}:"
                f"{self.bundle_sha256}:{self.template_sha256}"
            ),
        )
        try:
            job = self.batch.read_namespaced_job(name, self.namespace)
        except Exception as error:
            if _api_status(error) != 404:
                raise
            if not allow_create:
                return "deferred"
            try:
                body = self._build_job(node, name)
            except ValueError as build_error:
                LOGGER.error("node %s cannot be installed: %s", node_name, build_error)
                return "unsupported"
            try:
                self.batch.create_namespaced_job(self.namespace, body)
            except Exception as create_error:
                if _api_status(create_error) != 409:
                    raise
            self._mark_node(node_name, node_uid, "Installing", boot_id)
            LOGGER.info("created installer Job %s for node %s", name, node_name)
            return "created"

        if _job_condition(job, "Complete"):
            if not identity_matches or installer_state == "Retrying":
                self.batch.delete_namespaced_job(
                    name,
                    self.namespace,
                    propagation_policy="Background",
                )
                self._mark_node(node_name, node_uid, "Retrying", boot_id)
                LOGGER.warning(
                    "deleted stale completed installer Job %s for replay",
                    name,
                )
                return "running"
            self._mark_node(node_name, node_uid, "Succeeded", boot_id)
            return "succeeded"
        if _job_condition(job, "Failed"):
            if self._retry_due(job):
                self.batch.delete_namespaced_job(
                    name,
                    self.namespace,
                    propagation_policy="Background",
                )
                self._mark_node(node_name, node_uid, "Retrying", boot_id)
                LOGGER.warning("deleted failed installer Job %s for retry", name)
            else:
                self._mark_node(node_name, node_uid, "Failed", boot_id)
            return "failed"
        return "running"

    def _retry_due(self, job: Any) -> bool:
        created = _value(_metadata(job), "creation_timestamp")
        if created is None:
            return True
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        return (self.now() - created).total_seconds() >= self.retry_seconds

    def _mark_node(
        self,
        node_name: str,
        node_uid: str,
        state: str,
        boot_id: str | None = None,
    ) -> None:
        annotations = {
            INSTALLER_VERSION_ANNOTATION: self.version,
            INSTALLER_DIGEST_ANNOTATION: self.config_digest,
            INSTALLER_ARTIFACT_ANNOTATION: self.artifact_sha256,
            INSTALLER_BUNDLE_ANNOTATION: self.bundle_sha256,
            INSTALLER_TEMPLATE_ANNOTATION: self.template_sha256,
            INSTALLER_NODE_UID_ANNOTATION: node_uid,
            INSTALLER_STATE_ANNOTATION: state,
        }
        if boot_id is not None:
            annotations[INSTALLER_BOOT_ID_ANNOTATION] = boot_id
        self.core.patch_node(
            node_name,
            {"metadata": {"annotations": annotations}},
        )

    def _build_job(self, node: Any, job_name: str) -> dict[str, Any]:
        body = copy.deepcopy(self.job_template)
        metadata = _metadata(node)
        node_name = str(_value(metadata, "name"))
        node_uid = str(_value(metadata, "uid"))
        labels = dict(_value(metadata, "labels", {}) or {})
        instance_type = labels.get("node.kubernetes.io/instance-type", "")
        expected_gpus, expected_efa = _inventory(instance_type)
        node_ip = _internal_ip(node)
        metrics_url = self.dcgm_metrics_url_template.replace(
            "{node_name}", node_name
        ).replace("{node_ip}", node_ip)

        body.setdefault("metadata", {})["name"] = job_name
        body["metadata"]["namespace"] = self.namespace
        body["metadata"]["annotations"] = {
            INSTALLER_DIGEST_ANNOTATION: self.config_digest,
            INSTALLER_ARTIFACT_ANNOTATION: self.artifact_sha256,
            INSTALLER_BUNDLE_ANNOTATION: self.bundle_sha256,
            INSTALLER_TEMPLATE_ANNOTATION: self.template_sha256,
        }
        body["metadata"]["labels"] = {
            INSTALLER_JOB_LABEL: "true",
            "gpu-fault.io/node-uid": node_uid,
            "gpu-fault.io/config-digest": hashlib.sha256(
                self.config_digest.encode()
            ).hexdigest()[:16],
        }
        body["spec"]["activeDeadlineSeconds"] = self.job_active_deadline_seconds
        pod_template = body["spec"]["template"]
        pod_template.setdefault("metadata", {})["labels"] = dict(
            body["metadata"]["labels"]
        )
        pod_spec = pod_template["spec"]
        pod_spec["nodeName"] = node_name
        node_secret = next(
            (
                item
                for item in pod_spec.get("volumes", [])
                if item.get("name") == "node-secret"
            ),
            None,
        )
        if node_secret is None:
            raise ValueError("installer template has no node-secret volume")
        node_secret["secret"] = {
            "secretName": self.node_action_keys_secret,
            "items": [
                {
                    "key": node_name,
                    "path": "node-action-secret",
                }
            ],
        }
        container = next(
            item for item in pod_spec["containers"] if item["name"] == "installer"
        )
        updates = {
            "TARGET_NODE_NAME": node_name,
            "TARGET_NODE_IP": node_ip,
            "TARGET_NODE_UID": node_uid,
            "NODE_INSTANCE_TYPE": instance_type,
            "EXPECTED_GPU_COUNT": str(expected_gpus),
            "EXPECTED_EFA_DEVICE_COUNT": str(expected_efa),
            "DCGM_METRICS_URL_B64": base64.b64encode(metrics_url.encode()).decode(),
        }
        for env in container.get("env", []):
            name = env.get("name")
            if name in updates:
                env.clear()
                env.update(name=name, value=updates[name])
        return body


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


TEMPLATE_CONTENT_SHA256_ENV = "GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256"


def load_job_template(
    template_data: bytes | str,
    *,
    expected_sha256: str | None,
    origin: str,
) -> dict[str, Any]:
    """Parse the installer Job template only if it is the one the deploy pinned.

    The template arrives from a ConfigMap in the reconciler's namespace, and
    whatever it says becomes a privileged hostPath Pod on every GPU node with
    that node's HMAC key mounted. Anyone who can update the ConfigMap would
    therefore own the fleet -- unless the deploy also pins the exact bytes it
    put there. ``expected_sha256`` is that pin
    (:data:`TEMPLATE_CONTENT_SHA256_ENV`, the SHA-256 of the ``job.yaml`` text
    the deploy script placed in the ConfigMap). No pin, or a pin that does not
    match, is fatal at startup: the reconciler must not fall back to trusting
    the cluster's copy.
    """

    raw = template_data if isinstance(template_data, bytes) else template_data.encode()
    if not raw.strip():
        raise RuntimeError(f"{origin} is empty")
    expected = (expected_sha256 or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise RuntimeError(
            f"{TEMPLATE_CONTENT_SHA256_ENV} is required and must be the SHA-256 "
            f"of the installer template text ({origin})"
        )
    actual = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise RuntimeError(
            f"installer template {origin} does not match "
            f"{TEMPLATE_CONTENT_SHA256_ENV}: expected {expected[:12]}..., "
            f"loaded {actual[:12]}...; refusing to start"
        )
    template = yaml.safe_load(raw)
    if not isinstance(template, dict):
        raise RuntimeError(f"installer template {origin} is not a Job document")
    return template


def main() -> None:
    from kubernetes import client, config

    configure_logging()
    namespace = os.environ.get("GPU_FAULT_NAMESPACE", "gpu-fault-system")
    template_name = os.environ.get(
        "GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP",
        "gpu-fault-node-installer-template",
    )
    poll_seconds = int(os.environ.get("GPU_FAULT_RECONCILE_SECONDS", "5"))
    retry_seconds = int(os.environ.get("GPU_FAULT_INSTALL_RETRY_SECONDS", "300"))
    max_unavailable = int(os.environ.get("GPU_FAULT_INSTALLER_MAX_UNAVAILABLE", "1"))
    active_deadline_seconds = int(
        os.environ.get("GPU_FAULT_INSTALLER_ACTIVE_DEADLINE_SECONDS", "840")
    )
    allowed_nodes_value = os.environ.get(
        "GPU_FAULT_INSTALLER_ALLOWED_NODES",
        "*",
    ).strip()
    allowed_nodes = (
        None
        if allowed_nodes_value == "*"
        else frozenset(
            item.strip() for item in allowed_nodes_value.split(",") if item.strip()
        )
    )
    config.load_incluster_config()
    core = client.CoreV1Api()
    batch = client.BatchV1Api()
    template_path = os.environ.get("GPU_FAULT_INSTALLER_TEMPLATE_PATH", "").strip()
    template_data: bytes | str
    if template_path:
        # Bytes, not text: the pin is over the exact file the deploy wrote, and
        # read_text() would fold line endings before hashing.
        template_data = Path(template_path).read_bytes()
        origin = template_path
    else:
        config_map = core.read_namespaced_config_map(template_name, namespace)
        template_data = (config_map.data or {}).get("job.yaml", "")
        origin = f"{template_name}/job.yaml"
    job_template = load_job_template(
        template_data,
        expected_sha256=os.environ.get(TEMPLATE_CONTENT_SHA256_ENV),
        origin=origin,
    )
    reconciler = NodeInstallerReconciler(
        core,
        batch,
        namespace=namespace,
        cluster_name=_required_env("GPU_FAULT_HYPERPOD_CLUSTER"),
        version=_required_env("GPU_FAULT_INSTALLER_VERSION"),
        config_digest=_required_env("GPU_FAULT_INSTALLER_CONFIG_DIGEST"),
        artifact_sha256=_required_env("GPU_FAULT_INSTALLER_ARTIFACT_SHA256"),
        bundle_sha256=os.getenv("GPU_FAULT_INSTALLER_BUNDLE_SHA256") or None,
        template_sha256=_required_env("GPU_FAULT_INSTALLER_TEMPLATE_SHA256"),
        job_template=job_template,
        dcgm_metrics_url_template=os.environ.get(
            "GPU_FAULT_DCGM_METRICS_URL",
            "http://127.0.0.1:9400/metrics",
        ),
        node_action_keys_secret=os.environ.get(
            "GPU_FAULT_NODE_ACTION_KEYS_SECRET",
            "gpu-fault-node-action-keys",
        ),
        retry_seconds=retry_seconds,
        max_unavailable=max_unavailable,
        job_active_deadline_seconds=active_deadline_seconds,
        allowed_node_names=allowed_nodes,
        wave_config_map=(os.environ.get("GPU_FAULT_INSTALLER_WAVE_CONFIG_MAP") or None),
        agent_port=int(os.environ.get("GPU_FAULT_NODE_AGENT_PORT", "9099")),
        reboot_grace_seconds=int(
            os.environ.get("GPU_FAULT_INSTALLER_REBOOT_GRACE_SECONDS", "600")
        ),
    )
    while True:
        try:
            LOGGER.info("node installer reconcile: %s", reconciler.reconcile_once())
        except Exception:
            LOGGER.exception("node installer reconcile failed")
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
