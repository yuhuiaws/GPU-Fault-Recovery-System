from __future__ import annotations

import base64
import copy
import hashlib
import logging
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import yaml

LOGGER = logging.getLogger(__name__)

INSTALLER_VERSION_ANNOTATION = "gpu-fault.io/installer-version"
INSTALLER_DIGEST_ANNOTATION = "gpu-fault.io/installer-config-digest"
INSTALLER_ARTIFACT_ANNOTATION = "gpu-fault.io/installer-artifact-sha256"
INSTALLER_NODE_UID_ANNOTATION = "gpu-fault.io/installer-node-uid"
INSTALLER_STATE_ANNOTATION = "gpu-fault.io/installer-state"
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
        job_template: dict[str, Any],
        dcgm_metrics_url_template: str,
        node_action_keys_secret: str = ("gpu-fault-node-action-keys"),
        retry_seconds: int = 300,
        now: Callable[[], datetime] | None = None,
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
        self.job_template = job_template
        self.dcgm_metrics_url_template = dcgm_metrics_url_template
        self.node_action_keys_secret = node_action_keys_secret
        self.retry_seconds = retry_seconds
        self.now = now or (lambda: datetime.now(UTC))

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
            "not_ready": 0,
            "unsupported": 0,
        }
        response = self.core.list_node(label_selector=self.node_selector)
        for node in _value(response, "items", []) or []:
            outcome = self._reconcile_node(node)
            result[outcome] += 1
        return result

    def _reconcile_node(self, node: Any) -> str:
        metadata = _metadata(node)
        node_name = str(_value(metadata, "name"))
        node_uid = str(_value(metadata, "uid"))
        annotations = dict(_value(metadata, "annotations", {}) or {})
        if not _is_ready(node):
            LOGGER.info("node %s is not Ready; installation deferred", node_name)
            return "not_ready"
        if (
            annotations.get(INSTALLER_VERSION_ANNOTATION) == self.version
            and annotations.get(INSTALLER_DIGEST_ANNOTATION) == self.config_digest
            and annotations.get(INSTALLER_ARTIFACT_ANNOTATION) == self.artifact_sha256
            and annotations.get(INSTALLER_NODE_UID_ANNOTATION) == node_uid
            and annotations.get(INSTALLER_STATE_ANNOTATION) == "Succeeded"
        ):
            return "current"

        name = _job_name(
            node_name,
            node_uid,
            f"{self.config_digest}:{self.artifact_sha256}",
        )
        try:
            job = self.batch.read_namespaced_job(name, self.namespace)
        except Exception as error:
            if _api_status(error) != 404:
                raise
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
            self._mark_node(node_name, node_uid, "Installing")
            LOGGER.info("created installer Job %s for node %s", name, node_name)
            return "created"

        if _job_condition(job, "Complete"):
            self._mark_node(node_name, node_uid, "Succeeded")
            return "succeeded"
        if _job_condition(job, "Failed"):
            if self._retry_due(job):
                self.batch.delete_namespaced_job(
                    name,
                    self.namespace,
                    propagation_policy="Background",
                )
                self._mark_node(node_name, node_uid, "Retrying")
                LOGGER.warning("deleted failed installer Job %s for retry", name)
            else:
                self._mark_node(node_name, node_uid, "Failed")
            return "failed"
        return "running"

    def _retry_due(self, job: Any) -> bool:
        created = _value(_metadata(job), "creation_timestamp")
        if created is None:
            return True
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        return (self.now() - created).total_seconds() >= self.retry_seconds

    def _mark_node(self, node_name: str, node_uid: str, state: str) -> None:
        annotations = {
            INSTALLER_VERSION_ANNOTATION: self.version,
            INSTALLER_DIGEST_ANNOTATION: self.config_digest,
            INSTALLER_ARTIFACT_ANNOTATION: self.artifact_sha256,
            INSTALLER_NODE_UID_ANNOTATION: node_uid,
            INSTALLER_STATE_ANNOTATION: state,
        }
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
        body["metadata"]["labels"] = {
            INSTALLER_JOB_LABEL: "true",
            "gpu-fault.io/node-uid": node_uid,
            "gpu-fault.io/config-digest": hashlib.sha256(
                self.config_digest.encode()
            ).hexdigest()[:16],
        }
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


def main() -> None:
    from kubernetes import client, config

    logging.basicConfig(
        level=os.environ.get("GPU_FAULT_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    namespace = os.environ.get("GPU_FAULT_NAMESPACE", "gpu-fault-system")
    template_name = os.environ.get(
        "GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP",
        "gpu-fault-node-installer-template",
    )
    poll_seconds = int(os.environ.get("GPU_FAULT_RECONCILE_SECONDS", "15"))
    retry_seconds = int(os.environ.get("GPU_FAULT_INSTALL_RETRY_SECONDS", "300"))
    config.load_incluster_config()
    core = client.CoreV1Api()
    batch = client.BatchV1Api()
    template_path = os.environ.get("GPU_FAULT_INSTALLER_TEMPLATE_PATH", "").strip()
    if template_path:
        template_text = Path(template_path).read_text()
    else:
        config_map = core.read_namespaced_config_map(template_name, namespace)
        template_text = (config_map.data or {}).get("job.yaml", "")
    if not template_text:
        raise RuntimeError(f"{template_name}/job.yaml is empty")
    reconciler = NodeInstallerReconciler(
        core,
        batch,
        namespace=namespace,
        cluster_name=_required_env("GPU_FAULT_CLUSTER_ID"),
        version=_required_env("GPU_FAULT_INSTALLER_VERSION"),
        config_digest=_required_env("GPU_FAULT_INSTALLER_CONFIG_DIGEST"),
        artifact_sha256=_required_env("GPU_FAULT_INSTALLER_ARTIFACT_SHA256"),
        job_template=yaml.safe_load(template_text),
        dcgm_metrics_url_template=os.environ.get(
            "GPU_FAULT_DCGM_METRICS_URL",
            "http://127.0.0.1:9400/metrics",
        ),
        node_action_keys_secret=os.environ.get(
            "GPU_FAULT_NODE_ACTION_KEYS_SECRET",
            "gpu-fault-node-action-keys",
        ),
        retry_seconds=retry_seconds,
    )
    while True:
        try:
            LOGGER.info("node installer reconcile: %s", reconciler.reconcile_once())
        except Exception:
            LOGGER.exception("node installer reconcile failed")
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
