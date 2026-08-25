from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from gpu_fault.node_installer_reconciler import (
    INSTALLER_ARTIFACT_ANNOTATION,
    INSTALLER_DIGEST_ANNOTATION,
    INSTALLER_NODE_UID_ANNOTATION,
    INSTALLER_STATE_ANNOTATION,
    INSTALLER_VERSION_ANNOTATION,
    NodeInstallerReconciler,
)

NOW = datetime(2026, 7, 29, tzinfo=UTC)
ARTIFACT = "a" * 64


class ApiError(Exception):
    def __init__(self, status: int):
        self.status = status


def node(
    *,
    name: str = "hyperpod-i-123",
    uid: str = "node-uid",
    instance_type: str = "ml.p5en.48xlarge",
    annotations: dict[str, str] | None = None,
    ready: bool = True,
):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name=name,
            uid=uid,
            labels={"node.kubernetes.io/instance-type": instance_type},
            annotations=annotations or {},
        ),
        status=SimpleNamespace(
            conditions=[
                SimpleNamespace(type="Ready", status="True" if ready else "False")
            ],
            addresses=[SimpleNamespace(type="InternalIP", address="10.0.1.25")],
        ),
    )


def job(*, condition: str | None = None, age_seconds: int = 0):
    conditions = []
    if condition:
        conditions.append(SimpleNamespace(type=condition, status="True"))
    return SimpleNamespace(
        metadata=SimpleNamespace(
            creation_timestamp=NOW - timedelta(seconds=age_seconds)
        ),
        status=SimpleNamespace(conditions=conditions),
    )


class CoreApi:
    def __init__(self, nodes):
        self.nodes = nodes
        self.patches = []

    def list_node(self, *, label_selector):
        assert label_selector == "sagemaker.amazonaws.com/cluster-name=hp-cluster-a"
        return SimpleNamespace(items=self.nodes)

    def patch_node(self, name, body):
        self.patches.append((name, body))


class BatchApi:
    def __init__(self, existing=None):
        self.existing = existing
        self.created = []
        self.deleted = []
        self.reads = 0

    def read_namespaced_job(self, name, namespace):
        self.reads += 1
        if self.existing is None:
            raise ApiError(404)
        return self.existing

    def create_namespaced_job(self, namespace, body):
        self.created.append((namespace, body))

    def delete_namespaced_job(self, name, namespace, **kwargs):
        self.deleted.append((name, namespace, kwargs))


def template():
    names = [
        "TARGET_NODE_NAME",
        "TARGET_NODE_IP",
        "TARGET_NODE_UID",
        "NODE_INSTANCE_TYPE",
        "EXPECTED_GPU_COUNT",
        "EXPECTED_EFA_DEVICE_COUNT",
        "DCGM_METRICS_URL_B64",
    ]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": "template"},
        "spec": {
            "template": {
                "spec": {
                    "nodeName": "old",
                    "volumes": [
                        {
                            "name": "node-secret",
                            "secret": {
                                "secretName": "old-secret",
                                "items": [
                                    {"key": "old-node", "path": "node-action-secret"}
                                ],
                            },
                        }
                    ],
                    "containers": [
                        {
                            "name": "installer",
                            "env": [{"name": name, "value": "old"} for name in names],
                        }
                    ],
                }
            }
        },
    }


def reconciler(core, batch):
    return NodeInstallerReconciler(
        core,
        batch,
        namespace="gpu-fault-system",
        cluster_name="hp-cluster-a",
        version="0.10.0",
        config_digest="config-sha",
        artifact_sha256=ARTIFACT,
        job_template=template(),
        dcgm_metrics_url_template="http://{node_ip}:9400/metrics",
        retry_seconds=300,
        now=lambda: NOW,
    )


def test_current_node_is_not_reinstalled():
    annotations = {
        INSTALLER_VERSION_ANNOTATION: "0.10.0",
        INSTALLER_DIGEST_ANNOTATION: "config-sha",
        INSTALLER_ARTIFACT_ANNOTATION: ARTIFACT,
        INSTALLER_NODE_UID_ANNOTATION: "node-uid",
        INSTALLER_STATE_ANNOTATION: "Succeeded",
    }
    core = CoreApi([node(annotations=annotations)])
    batch = BatchApi()

    assert reconciler(core, batch).reconcile_once()["current"] == 1
    assert batch.reads == 0
    assert not core.patches


def test_missing_installation_creates_node_bound_job():
    core = CoreApi([node()])
    batch = BatchApi()

    assert reconciler(core, batch).reconcile_once()["created"] == 1
    namespace, body = batch.created[0]
    assert namespace == "gpu-fault-system"
    assert body["spec"]["template"]["spec"]["nodeName"] == "hyperpod-i-123"
    env = {
        item["name"]: item["value"]
        for item in body["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert env["EXPECTED_GPU_COUNT"] == "8"
    assert env["EXPECTED_EFA_DEVICE_COUNT"] == "16"
    assert env["TARGET_NODE_IP"] == "10.0.1.25"
    node_secret = next(
        item
        for item in body["spec"]["template"]["spec"]["volumes"]
        if item["name"] == "node-secret"
    )
    assert node_secret["secret"] == {
        "secretName": "gpu-fault-node-action-keys",
        "items": [{"key": "hyperpod-i-123", "path": "node-action-secret"}],
    }
    assert (
        core.patches[-1][1]["metadata"]["annotations"][INSTALLER_STATE_ANNOTATION]
        == "Installing"
    )


def test_artifact_change_reinstalls_even_when_config_is_unchanged():
    annotations = {
        INSTALLER_VERSION_ANNOTATION: "0.10.0",
        INSTALLER_DIGEST_ANNOTATION: "config-sha",
        INSTALLER_ARTIFACT_ANNOTATION: "b" * 64,
        INSTALLER_NODE_UID_ANNOTATION: "node-uid",
        INSTALLER_STATE_ANNOTATION: "Succeeded",
    }
    core = CoreApi([node(annotations=annotations)])
    batch = BatchApi()

    assert reconciler(core, batch).reconcile_once()["created"] == 1
    assert len(batch.created) == 1


def test_completed_job_marks_node_current():
    core = CoreApi([node()])
    batch = BatchApi(job(condition="Complete"))

    assert reconciler(core, batch).reconcile_once()["succeeded"] == 1
    annotations = core.patches[-1][1]["metadata"]["annotations"]
    assert annotations[INSTALLER_STATE_ANNOTATION] == "Succeeded"
    assert annotations[INSTALLER_DIGEST_ANNOTATION] == "config-sha"
    assert annotations[INSTALLER_ARTIFACT_ANNOTATION] == ARTIFACT


def test_failed_job_is_deleted_after_retry_delay():
    core = CoreApi([node()])
    batch = BatchApi(job(condition="Failed", age_seconds=301))

    assert reconciler(core, batch).reconcile_once()["failed"] == 1
    assert len(batch.deleted) == 1
    assert (
        core.patches[-1][1]["metadata"]["annotations"][INSTALLER_STATE_ANNOTATION]
        == "Retrying"
    )


def test_unsupported_instance_does_not_create_privileged_job():
    core = CoreApi([node(instance_type="ml.unknown")])
    batch = BatchApi()

    assert reconciler(core, batch).reconcile_once()["unsupported"] == 1
    assert not batch.created
    assert not core.patches
