from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault.node_installer_reconciler import (
    INSTALLER_ARTIFACT_ANNOTATION,
    INSTALLER_BOOT_ID_ANNOTATION,
    INSTALLER_BUNDLE_ANNOTATION,
    INSTALLER_DIGEST_ANNOTATION,
    INSTALLER_NODE_UID_ANNOTATION,
    INSTALLER_STATE_ANNOTATION,
    INSTALLER_TEMPLATE_ANNOTATION,
    INSTALLER_VERSION_ANNOTATION,
    TEMPLATE_CONTENT_SHA256_ENV,
    NodeInstallerReconciler,
    load_job_template,
)

NOW = datetime(2026, 7, 29, tzinfo=UTC)
ARTIFACT = "a" * 64
BUNDLE = "c" * 64
TEMPLATE = "d" * 64


def test_template_text_must_match_the_pinned_content_digest():
    """The ConfigMap is in the cluster; the pin is in the deploy. They must agree.

    Whoever can update gpu-fault-node-installer-template-* would otherwise
    choose what runs as a privileged hostPath Pod on every GPU node, with that
    node's HMAC key mounted. The reconciler therefore hashes the exact bytes it
    loads and compares them to the digest the deploy stamped into its env.
    """

    text = yaml.safe_dump(template(), sort_keys=False)
    digest = hashlib.sha256(text.encode()).hexdigest()

    assert load_job_template(text, expected_sha256=digest, origin="cm/job.yaml") == (
        template()
    )
    # Bytes and str hash the same way, so the mounted-file path and the
    # ConfigMap path cannot disagree about one template.
    assert (
        load_job_template(
            text.encode(), expected_sha256=digest.upper(), origin="cm/job.yaml"
        )
        == template()
    )

    with pytest.raises(RuntimeError, match="does not match"):
        load_job_template(text + "\n# edited", expected_sha256=digest, origin="x")
    with pytest.raises(RuntimeError, match="does not match"):
        load_job_template(text, expected_sha256="0" * 64, origin="x")


@pytest.mark.parametrize("missing", [None, "", "   ", "not-a-digest", "ab" * 31])
def test_template_load_fails_closed_without_a_usable_pin(missing):
    text = yaml.safe_dump(template(), sort_keys=False)

    with pytest.raises(RuntimeError, match=TEMPLATE_CONTENT_SHA256_ENV):
        load_job_template(text, expected_sha256=missing, origin="cm/job.yaml")


def test_template_load_rejects_empty_and_non_document_templates():
    digest = hashlib.sha256(b"").hexdigest()
    with pytest.raises(RuntimeError, match="is empty"):
        load_job_template("", expected_sha256=digest, origin="cm/job.yaml")

    scalar = "just-a-string\n"
    with pytest.raises(RuntimeError, match="not a Job document"):
        load_job_template(
            scalar,
            expected_sha256=hashlib.sha256(scalar.encode()).hexdigest(),
            origin="cm/job.yaml",
        )


def test_reconciler_requires_an_explicit_template_identity_pin():
    """No fallback digest: a value derived from the config digest matched anything."""

    with pytest.raises(TypeError):
        NodeInstallerReconciler(  # type: ignore[call-arg]
            CoreApi([]),
            BatchApi(),
            namespace="gpu-fault-system",
            cluster_name="hp-cluster-a",
            version="0.10.0",
            config_digest="config-sha",
            artifact_sha256=ARTIFACT,
            job_template=template(),
            dcgm_metrics_url_template="http://{node_ip}:9400/metrics",
        )
    with pytest.raises(ValueError, match="template SHA-256"):
        NodeInstallerReconciler(
            CoreApi([]),
            BatchApi(),
            namespace="gpu-fault-system",
            cluster_name="hp-cluster-a",
            version="0.10.0",
            config_digest="config-sha",
            artifact_sha256=ARTIFACT,
            template_sha256="REPLACE_WITH_INSTALLER_TEMPLATE_SHA256",
            job_template=template(),
            dcgm_metrics_url_template="http://{node_ip}:9400/metrics",
        )


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
    def __init__(self, nodes, *, wave_data=None):
        self.nodes = nodes
        self.wave_data = wave_data
        self.patches = []

    def list_node(self, *, label_selector):
        assert label_selector == "sagemaker.amazonaws.com/cluster-name=hp-cluster-a"
        return SimpleNamespace(items=self.nodes)

    def patch_node(self, name, body):
        self.patches.append((name, body))

    def read_namespaced_config_map(self, name, namespace):
        assert name == "gpu-fault-node-installer-wave"
        assert namespace == "gpu-fault-system"
        return SimpleNamespace(data=self.wave_data)


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
        bundle_sha256=BUNDLE,
        template_sha256=TEMPLATE,
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
        INSTALLER_BUNDLE_ANNOTATION: BUNDLE,
        INSTALLER_TEMPLATE_ANNOTATION: TEMPLATE,
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
    assert body["spec"]["activeDeadlineSeconds"] == 840
    assert body["metadata"]["annotations"] == {
        INSTALLER_DIGEST_ANNOTATION: "config-sha",
        INSTALLER_ARTIFACT_ANNOTATION: ARTIFACT,
        INSTALLER_BUNDLE_ANNOTATION: BUNDLE,
        INSTALLER_TEMPLATE_ANNOTATION: TEMPLATE,
    }
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
        INSTALLER_BUNDLE_ANNOTATION: BUNDLE,
        INSTALLER_TEMPLATE_ANNOTATION: TEMPLATE,
        INSTALLER_NODE_UID_ANNOTATION: "node-uid",
        INSTALLER_STATE_ANNOTATION: "Succeeded",
    }
    core = CoreApi([node(annotations=annotations)])
    batch = BatchApi()

    assert reconciler(core, batch).reconcile_once()["created"] == 1
    assert len(batch.created) == 1


def test_completed_job_marks_node_current():
    annotations = {
        INSTALLER_VERSION_ANNOTATION: "0.10.0",
        INSTALLER_DIGEST_ANNOTATION: "config-sha",
        INSTALLER_ARTIFACT_ANNOTATION: ARTIFACT,
        INSTALLER_BUNDLE_ANNOTATION: BUNDLE,
        INSTALLER_TEMPLATE_ANNOTATION: TEMPLATE,
        INSTALLER_NODE_UID_ANNOTATION: "node-uid",
        INSTALLER_STATE_ANNOTATION: "Installing",
    }
    core = CoreApi([node(annotations=annotations)])
    batch = BatchApi(job(condition="Complete"))

    assert reconciler(core, batch).reconcile_once()["succeeded"] == 1
    annotations = core.patches[-1][1]["metadata"]["annotations"]
    assert annotations[INSTALLER_STATE_ANNOTATION] == "Succeeded"
    assert annotations[INSTALLER_DIGEST_ANNOTATION] == "config-sha"
    assert annotations[INSTALLER_ARTIFACT_ANNOTATION] == ARTIFACT
    assert annotations[INSTALLER_BUNDLE_ANNOTATION] == BUNDLE
    assert annotations[INSTALLER_TEMPLATE_ANNOTATION] == TEMPLATE


def test_completed_job_is_replayed_after_rollback():
    annotations = {
        INSTALLER_VERSION_ANNOTATION: "0.10.0",
        INSTALLER_DIGEST_ANNOTATION: "previous-config",
        INSTALLER_ARTIFACT_ANNOTATION: "b" * 64,
        INSTALLER_BUNDLE_ANNOTATION: "e" * 64,
        INSTALLER_TEMPLATE_ANNOTATION: "f" * 64,
        INSTALLER_NODE_UID_ANNOTATION: "node-uid",
        INSTALLER_STATE_ANNOTATION: "Succeeded",
    }
    current_node = node(annotations=annotations)
    core = CoreApi([current_node])
    batch = BatchApi(job(condition="Complete"))
    active = reconciler(core, batch)

    assert active.reconcile_once()["running"] == 1
    assert len(batch.deleted) == 1
    replay_annotations = core.patches[-1][1]["metadata"]["annotations"]
    assert replay_annotations[INSTALLER_STATE_ANNOTATION] == "Retrying"
    assert replay_annotations[INSTALLER_DIGEST_ANNOTATION] == "config-sha"
    assert replay_annotations[INSTALLER_ARTIFACT_ANNOTATION] == ARTIFACT
    assert replay_annotations[INSTALLER_BUNDLE_ANNOTATION] == BUNDLE
    assert replay_annotations[INSTALLER_TEMPLATE_ANNOTATION] == TEMPLATE

    current_node.metadata.annotations = replay_annotations
    assert active.reconcile_once()["running"] == 1
    assert len(batch.deleted) == 2

    batch.existing = None
    assert active.reconcile_once()["created"] == 1
    assert len(batch.created) == 1
    assert (
        core.patches[-1][1]["metadata"]["annotations"][INSTALLER_STATE_ANNOTATION]
        == "Installing"
    )


def test_bundle_or_template_change_reinstalls_node():
    annotations = {
        INSTALLER_VERSION_ANNOTATION: "0.10.0",
        INSTALLER_DIGEST_ANNOTATION: "config-sha",
        INSTALLER_ARTIFACT_ANNOTATION: ARTIFACT,
        INSTALLER_BUNDLE_ANNOTATION: "e" * 64,
        INSTALLER_TEMPLATE_ANNOTATION: TEMPLATE,
        INSTALLER_NODE_UID_ANNOTATION: "node-uid",
        INSTALLER_STATE_ANNOTATION: "Succeeded",
    }
    core = CoreApi([node(annotations=annotations)])
    batch = BatchApi()

    assert reconciler(core, batch).reconcile_once()["created"] == 1


def test_reconcile_limits_new_jobs_to_max_unavailable():
    core = CoreApi(
        [
            node(name="hyperpod-i-001", uid="node-1"),
            node(name="hyperpod-i-002", uid="node-2"),
        ]
    )
    batch = BatchApi()

    result = reconciler(core, batch).reconcile_once()

    assert result["created"] == 1
    assert result["deferred"] == 1
    assert len(batch.created) == 1


def test_reconcile_limits_nodes_to_the_active_fleet_wave():
    core = CoreApi(
        [
            node(name="hyperpod-i-001", uid="node-1"),
            node(name="hyperpod-i-002", uid="node-2"),
        ]
    )
    batch = BatchApi()
    active = reconciler(core, batch)
    active.allowed_node_names = frozenset({"hyperpod-i-002"})

    result = active.reconcile_once()

    assert result["created"] == 1
    assert len(batch.created) == 1
    assert batch.created[0][1]["spec"]["template"]["spec"]["nodeName"] == (
        "hyperpod-i-002"
    )


def test_reconcile_reads_wave_without_restarting_controller():
    core = CoreApi(
        [
            node(name="hyperpod-i-001", uid="node-1"),
            node(name="hyperpod-i-002", uid="node-2"),
        ],
        wave_data={
            "allowed-nodes": "hyperpod-i-001,hyperpod-i-002",
            "max-unavailable": "2",
        },
    )
    batch = BatchApi()
    active = reconciler(core, batch)
    active.wave_config_map = "gpu-fault-node-installer-wave"

    result = active.reconcile_once()

    assert result["created"] == 2
    assert len(batch.created) == 2


def test_invalid_wave_config_fails_closed_without_creating_jobs():
    core = CoreApi(
        [node()], wave_data={"allowed-nodes": "*", "max-unavailable": "invalid"}
    )
    batch = BatchApi()
    active = reconciler(core, batch)
    active.wave_config_map = "gpu-fault-node-installer-wave"

    with pytest.raises(RuntimeError, match="max-unavailable"):
        active.reconcile_once()

    assert batch.created == []


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


# ------------------------------------------------- re-imaged nodes (boot id)


def _installed_annotations(**extra: str) -> dict[str, str]:
    annotations = {
        INSTALLER_VERSION_ANNOTATION: "0.10.0",
        INSTALLER_DIGEST_ANNOTATION: "config-sha",
        INSTALLER_ARTIFACT_ANNOTATION: ARTIFACT,
        INSTALLER_BUNDLE_ANNOTATION: BUNDLE,
        INSTALLER_TEMPLATE_ANNOTATION: TEMPLATE,
        INSTALLER_NODE_UID_ANNOTATION: "node-uid",
        INSTALLER_STATE_ANNOTATION: "Succeeded",
    }
    annotations.update(extra)
    return annotations


def _booted_node(annotations, *, boot_id: str, ready_since: datetime):
    item = node(annotations=annotations)
    item.status.node_info = SimpleNamespace(boot_id=boot_id)
    item.status.conditions = [
        SimpleNamespace(type="Ready", status="True", last_transition_time=ready_since)
    ]
    return item


def _reconciler_with_probe(core, batch, *, alive: bool):
    probes: list[tuple[str, int]] = []

    def probe(address: str, port: int) -> bool:
        probes.append((address, port))
        return alive

    value = NodeInstallerReconciler(
        core,
        batch,
        namespace="gpu-fault-system",
        cluster_name="hp-cluster-a",
        version="0.10.0",
        config_digest="config-sha",
        artifact_sha256=ARTIFACT,
        bundle_sha256=BUNDLE,
        template_sha256=TEMPLATE,
        job_template=template(),
        dcgm_metrics_url_template="http://{node_ip}:9400/metrics",
        retry_seconds=300,
        now=lambda: NOW,
        agent_alive=probe,
        reboot_grace_seconds=600,
    )
    return value, probes


def test_a_reimaged_node_whose_agent_is_gone_is_reinstalled():
    """HyperPod UpdateClusterSoftware keeps the Node object and its annotations
    but wipes /opt/gpu-fault. The UID check cannot see it; a new boot id plus
    an Agent port nobody answers on can (live 2026-09-06 12:0xZ, four nodes)."""

    core = CoreApi(
        [
            _booted_node(
                _installed_annotations(**{INSTALLER_BOOT_ID_ANNOTATION: "boot-old"}),
                boot_id="boot-new",
                ready_since=NOW - timedelta(hours=1),
            )
        ]
    )
    batch = BatchApi()
    value, probes = _reconciler_with_probe(core, batch, alive=False)

    first = value.reconcile_once()

    assert probes == [("10.0.1.25", 9099)], probes
    assert first["created"] == 1, first
    assert batch.created, "no installer Job was created for the re-imaged node"
    states = [
        body["metadata"]["annotations"][INSTALLER_STATE_ANNOTATION]
        for _name, body in core.patches
    ]
    assert states == ["Retrying", "Installing"], states
    assert all(
        body["metadata"]["annotations"][INSTALLER_BOOT_ID_ANNOTATION] == "boot-new"
        for _name, body in core.patches
    ), "the new boot id must be recorded with the reinstall"


def test_a_plain_reboot_with_a_live_agent_only_restamps_the_boot_id():
    core = CoreApi(
        [
            _booted_node(
                _installed_annotations(**{INSTALLER_BOOT_ID_ANNOTATION: "boot-old"}),
                boot_id="boot-new",
                ready_since=NOW - timedelta(hours=1),
            )
        ]
    )
    batch = BatchApi()
    value, probes = _reconciler_with_probe(core, batch, alive=True)

    result = value.reconcile_once()

    assert result["rebooted"] == 1, result
    assert not batch.created, "a node whose Agent answers must not be reinstalled"
    assert batch.reads == 0, "no installer Job lookup for a healthy rebooted node"
    ((name, body),) = core.patches
    assert body["metadata"]["annotations"][INSTALLER_STATE_ANNOTATION] == "Succeeded"
    assert body["metadata"]["annotations"][INSTALLER_BOOT_ID_ANNOTATION] == "boot-new"


def test_a_freshly_booted_node_gets_a_grace_period_before_reinstall():
    core = CoreApi(
        [
            _booted_node(
                _installed_annotations(**{INSTALLER_BOOT_ID_ANNOTATION: "boot-old"}),
                boot_id="boot-new",
                ready_since=NOW - timedelta(seconds=30),
            )
        ]
    )
    batch = BatchApi()
    value, _probes = _reconciler_with_probe(core, batch, alive=False)

    result = value.reconcile_once()

    assert result["recovering"] == 1, result
    assert not batch.created and not core.patches, (
        "an Agent that is still starting after a reboot must not be reinstalled"
    )


def test_a_node_installed_before_boot_ids_adopts_its_current_boot():
    core = CoreApi(
        [
            _booted_node(
                _installed_annotations(),
                boot_id="boot-now",
                ready_since=NOW - timedelta(hours=1),
            )
        ]
    )
    batch = BatchApi()
    value, probes = _reconciler_with_probe(core, batch, alive=False)

    result = value.reconcile_once()

    assert result["current"] == 1, result
    assert probes == [], "adopting a boot id must not probe or reinstall"
    ((name, body),) = core.patches
    assert body["metadata"]["annotations"][INSTALLER_BOOT_ID_ANNOTATION] == "boot-now"
    assert body["metadata"]["annotations"][INSTALLER_STATE_ANNOTATION] == "Succeeded"
