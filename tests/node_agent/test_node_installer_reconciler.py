from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault.node_installer_reconciler import (
    INSTALLER_ARTIFACT_ANNOTATION,
    INSTALLER_ATTEMPTS_ANNOTATION,
    INSTALLER_BOOT_ID_ANNOTATION,
    INSTALLER_BUNDLE_ANNOTATION,
    INSTALLER_DIGEST_ANNOTATION,
    INSTALLER_JOB_LABEL,
    INSTALLER_NODE_UID_ANNOTATION,
    INSTALLER_REASON_ANNOTATION,
    INSTALLER_RETRY_AFTER_ANNOTATION,
    INSTALLER_STATE_ANNOTATION,
    INSTALLER_TEMPLATE_ANNOTATION,
    INSTALLER_VERSION_ANNOTATION,
    POD_NEVER_STARTED_GRACE_SECONDS,
    RECONCILER_HEARTBEAT_PATH,
    REQUEST_TIMEOUT,
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


def job(
    *,
    condition: str | None = None,
    age_seconds: int = 0,
    failed_at: datetime | None = None,
):
    conditions = []
    if condition:
        item = SimpleNamespace(type=condition, status="True")
        if failed_at is not None:
            item.last_transition_time = failed_at
        conditions.append(item)
    return SimpleNamespace(
        metadata=SimpleNamespace(
            creation_timestamp=NOW - timedelta(seconds=age_seconds)
        ),
        status=SimpleNamespace(conditions=conditions),
    )


def installer_pod(*, reason: str | None = None, message: str = ""):
    """A Job Pod whose container is stuck in ``reason`` and never started."""

    waiting = (
        None if reason is None else SimpleNamespace(reason=reason, message=message)
    )
    return SimpleNamespace(
        metadata=SimpleNamespace(name="installer-pod"),
        status=SimpleNamespace(
            phase="Pending",
            container_statuses=[
                SimpleNamespace(
                    name="installer",
                    state=SimpleNamespace(
                        waiting=waiting, running=None, terminated=None
                    ),
                )
            ],
        ),
    )


class CoreApi:
    def __init__(self, nodes, *, wave_data=None, patch_failures=(), pods=None):
        self.nodes = nodes
        self.wave_data = wave_data
        self.patch_failures = set(patch_failures)
        self.pods = dict(pods or {})
        self.patches = []
        self.calls = []

    def list_node(self, *, label_selector, _request_timeout=None):
        assert label_selector == "sagemaker.amazonaws.com/cluster-name=hp-cluster-a"
        self.calls.append(("list_node", _request_timeout))
        return SimpleNamespace(items=self.nodes)

    def patch_node(self, name, body, _request_timeout=None):
        self.calls.append(("patch_node", _request_timeout))
        if name in self.patch_failures:
            raise ApiError(500)
        self.patches.append((name, body))

    def read_namespaced_config_map(self, name, namespace, _request_timeout=None):
        assert name == "gpu-fault-node-installer-wave"
        assert namespace == "gpu-fault-system"
        self.calls.append(("read_namespaced_config_map", _request_timeout))
        return SimpleNamespace(data=self.wave_data)

    def list_namespaced_pod(self, namespace, *, label_selector, _request_timeout=None):
        assert namespace == "gpu-fault-system"
        self.calls.append(("list_namespaced_pod", _request_timeout))
        items = [pod for key, pod in self.pods.items() if key in label_selector]
        return SimpleNamespace(items=items)


class BatchApi:
    """Fake Job API.

    ``existing`` answers for every Job name (the historical behaviour); ``jobs``
    maps a node name to the Job that node's installer Job lookup returns, which
    is what a budget question about two different nodes needs.
    """

    def __init__(self, existing=None, *, jobs=None):
        self.existing = existing
        self.jobs = dict(jobs or {})
        self.created = []
        self.deleted = []
        self.reads = 0
        self.calls = []

    def _for_name(self, name):
        for node_name, value in self.jobs.items():
            if node_name in name:
                return value
        return self.existing

    def read_namespaced_job(self, name, namespace, _request_timeout=None):
        self.reads += 1
        self.calls.append(("read_namespaced_job", _request_timeout))
        found = self._for_name(name)
        if found is None:
            raise ApiError(404)
        return found

    def list_namespaced_job(self, namespace, *, label_selector, _request_timeout=None):
        assert namespace == "gpu-fault-system"
        assert label_selector == f"{INSTALLER_JOB_LABEL}=true"
        self.calls.append(("list_namespaced_job", _request_timeout))
        items = list(self.jobs.values())
        if self.existing is not None:
            items.append(self.existing)
        return SimpleNamespace(items=items)

    def create_namespaced_job(self, namespace, body, _request_timeout=None):
        self.calls.append(("create_namespaced_job", _request_timeout))
        self.created.append((namespace, body))

    def delete_namespaced_job(self, name, namespace, _request_timeout=None, **kwargs):
        self.calls.append(("delete_namespaced_job", _request_timeout))
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


def reconciler(core, batch, *, now=lambda: NOW, max_unavailable=1):
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
        max_unavailable=max_unavailable,
        now=now,
    )


def apply_patches(node_item, core):
    """Fold the reconciler's annotation patches back onto the Node.

    The apiserver does this between passes; a test that asks what the *next*
    pass does has to do it too, because every backoff decision is read from the
    annotations rather than from reconciler memory.
    """

    annotations = dict(node_item.metadata.annotations or {})
    for _name, body in core.patches:
        for key, value in body["metadata"]["annotations"].items():
            if value is None:
                annotations.pop(key, None)
            else:
                annotations[key] = value
    node_item.metadata.annotations = annotations
    return annotations


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


# -------------------------- liveness, budget, isolation, backoff (F3/F5/F6/F8/F14)


def test_reconcile_pass_touches_the_heartbeat_file():
    """Task 16's livenessProbe reads the age of this file.

    Without it a half-open apiserver connection wedges the loop forever while
    the Pod stays Running and Ready (F6). The heartbeat is written even when the
    pass itself fails closed, because liveness answers "is the loop turning",
    not "did reconciliation succeed".
    """

    core = CoreApi([node()])
    batch = BatchApi()
    started = time.time()

    reconciler(core, batch).reconcile_once()

    path = Path(RECONCILER_HEARTBEAT_PATH)
    assert path.exists(), f"{RECONCILER_HEARTBEAT_PATH} was not written"
    assert path.stat().st_mtime >= started - 1, (
        "the heartbeat file was not refreshed by this reconcile pass"
    )

    failing = CoreApi(
        [node()], wave_data={"allowed-nodes": "*", "max-unavailable": "invalid"}
    )
    active = reconciler(failing, BatchApi())
    active.wave_config_map = "gpu-fault-node-installer-wave"
    before_failed_pass = time.time()
    with pytest.raises(RuntimeError, match="max-unavailable"):
        active.reconcile_once()

    assert path.stat().st_mtime >= before_failed_pass - 1, (
        "a pass that fails closed must still prove the loop is alive"
    )


def test_running_job_on_a_later_node_blocks_creation_on_an_earlier_node():
    """max_unavailable was overshot 2x because in_flight was counted in-loop (F3).

    ``hyperpod-a`` sorts before ``hyperpod-b``, so with the old in-loop counter
    it saw in_flight == 0 and created a second install while ``hyperpod-b`` was
    still mid-install: two nodes lose their Agent at once under a budget of 1.
    """

    core = CoreApi(
        [node(name="hyperpod-a", uid="node-a"), node(name="hyperpod-b", uid="node-b")]
    )
    batch = BatchApi(jobs={"hyperpod-b": job()})

    result = reconciler(core, batch).reconcile_once()

    assert batch.created == [], "an install slot already in use was handed out again"
    assert result["created"] == 0, result
    assert result["running"] == 1, result
    assert result["deferred"] == 1, result


def test_a_failing_node_does_not_block_later_nodes():
    """One node's exception used to abort the whole pass, every 5 s, forever (F5)."""

    core = CoreApi(
        [node(name="hyperpod-a", uid="node-a"), node(name="hyperpod-b", uid="node-b")],
        patch_failures={"hyperpod-a"},
    )
    batch = BatchApi()

    result = reconciler(core, batch, max_unavailable=2).reconcile_once()

    assert result["error"] == 1, result
    assert result["created"] == 1, result
    assert [name for name, _body in core.patches] == ["hyperpod-b"], core.patches
    assert [
        body["spec"]["template"]["spec"]["nodeName"] for _ns, body in batch.created
    ] == ["hyperpod-a", "hyperpod-b"], (
        "the node after the failing one must still be reconciled"
    )


def test_reconcile_passes_request_timeouts():
    """Every apiserver call is bounded, so a half-open socket cannot wedge a pass."""

    core = CoreApi(
        [
            node(name="hyperpod-i-001", uid="node-1"),
            node(name="hyperpod-i-002", uid="node-2"),
            node(name="hyperpod-i-003", uid="node-3"),
        ],
        wave_data={"allowed-nodes": "*", "max-unavailable": "2"},
        pods={"hyperpod-i-003": installer_pod(reason="CreateContainerConfigError")},
    )
    batch = BatchApi(
        jobs={
            "hyperpod-i-002": job(condition="Failed", age_seconds=301),
            "hyperpod-i-003": job(age_seconds=300),
        }
    )
    active = reconciler(core, batch)
    active.wave_config_map = "gpu-fault-node-installer-wave"

    active.reconcile_once()

    calls = core.calls + batch.calls
    assert {name for name, _timeout in calls} >= {
        "list_node",
        "patch_node",
        "read_namespaced_config_map",
        "list_namespaced_pod",
        "list_namespaced_job",
        "read_namespaced_job",
        "create_namespaced_job",
        "delete_namespaced_job",
    }, calls
    assert all(timeout == REQUEST_TIMEOUT for _name, timeout in calls), calls
    assert REQUEST_TIMEOUT == (5, 60), REQUEST_TIMEOUT


def test_failed_node_is_not_repatched_every_pass():
    """A node waiting out its retry delay was re-patched with identical
    annotations every 5 s (F14): 17k pointless node writes a day per node."""

    current = node()
    core = CoreApi([current])
    batch = BatchApi(job(condition="Failed", age_seconds=10, failed_at=NOW))
    active = reconciler(core, batch)

    first = active.reconcile_once()

    assert first["failed"] == 1, first
    assert len(core.patches) == 1, core.patches
    marked = core.patches[-1][1]["metadata"]["annotations"]
    assert marked[INSTALLER_STATE_ANNOTATION] == "Failed", marked
    apply_patches(current, core)

    second = active.reconcile_once()

    assert second["failed"] == 1, second
    assert len(core.patches) == 1, (
        "a node already annotated Failed must not be patched again"
    )
    assert batch.deleted == [], "the retry delay had not elapsed"
    assert not [
        name for name, _timeout in core.calls if name == "list_namespaced_pod"
    ], (
        "a Failed Job has no Pods left to read (the Job controller deleted them), "
        "so listing them every 5 s per failed node is pure load"
    )


def test_node_without_action_key_is_reported_not_retried():
    """A node with no key in gpu-fault-node-action-keys held the only install
    slot for ~14 of every 14.5 minutes (F8): CreateContainerConfigError until
    activeDeadlineSeconds, Failed, recreated on the next pass, forever.

    The Job is ``backoffLimit: 0`` + ``restartPolicy: Never``, so a Pod stuck in
    CreateContainerConfigError never fails the Job -- it is only killed by
    activeDeadlineSeconds, and the Job controller deletes the Pod in the same
    action. So the only state in which the waiting reason is readable is the one
    tested here: an *active* Job whose Pod is still Pending. The reconciler has
    no ``secrets`` RBAC, so it classifies on that reason, releases the slot at
    once instead of holding it for the full deadline, and backs off on the same
    curve as any other failure.
    """

    current = node()
    core = CoreApi(
        [current],
        pods={
            "hyperpod-i-123": installer_pod(
                reason="CreateContainerConfigError",
                message='secret "gpu-fault-node-action-keys" key '
                '"hyperpod-i-123" not found',
            )
        },
    )
    batch = BatchApi(job(age_seconds=60))
    clock = [NOW]
    active = reconciler(core, batch, now=lambda: clock[0])

    # 1. Inside the grace period a Pending Pod is just a slow start: no verdict,
    #    and not one pod LIST either.
    early = active.reconcile_once()

    assert early["running"] == 1, early
    assert early["unsupported"] == 0, early
    assert batch.deleted == [], "a Pod inside the start grace must not be judged"
    assert not [
        name for name, _timeout in core.calls if name == "list_namespaced_pod"
    ], f"{POD_NEVER_STARTED_GRACE_SECONDS}s of grace must cost no pod LIST"

    # 2. Past the grace period: classified, and the slot is released early.
    clock[0] = NOW + timedelta(seconds=POD_NEVER_STARTED_GRACE_SECONDS + 1)
    result = active.reconcile_once()

    assert result["unsupported"] == 1, result
    assert result["running"] == 0, result
    assert len(batch.deleted) == 1, (
        "the install slot must be released now, not after activeDeadlineSeconds"
    )
    marked = core.patches[-1][1]["metadata"]["annotations"]
    assert marked[INSTALLER_STATE_ANNOTATION] == "Unsupported", marked
    assert "CreateContainerConfigError" in marked[INSTALLER_REASON_ANNOTATION], marked
    assert "hyperpod-i-123" in marked[INSTALLER_REASON_ANNOTATION], marked
    assert marked[INSTALLER_ATTEMPTS_ANNOTATION] == "1", marked
    assert marked[INSTALLER_RETRY_AFTER_ANNOTATION], marked
    apply_patches(current, core)

    # 3. The Job is gone, so the create path owns the node now -- and it must
    #    not hand out a new Job until the backoff has elapsed.
    batch.existing = None
    again = active.reconcile_once()

    assert again["unsupported"] == 1, again
    assert batch.created == [], "no second install slot may be burned during backoff"
    assert len(core.patches) == 1, "the Unsupported annotation must not be rewritten"

    # 4. After the delay one more attempt is allowed: the missing key may have
    #    been added, and only a Job can find out.
    clock[0] = NOW + timedelta(seconds=POD_NEVER_STARTED_GRACE_SECONDS + 302)
    retried = active.reconcile_once()

    assert retried["created"] == 1, retried
    assert len(batch.created) == 1, batch.created
    retry_marked = core.patches[-1][1]["metadata"]["annotations"]
    assert retry_marked[INSTALLER_STATE_ANNOTATION] == "Installing", retry_marked
    assert retry_marked[INSTALLER_RETRY_AFTER_ANNOTATION] is None, retry_marked
    assert retry_marked[INSTALLER_ATTEMPTS_ANNOTATION] == "1", (
        "the attempt count must survive the retry, or the curve resets"
    )


def test_repeated_failures_back_off_on_a_doubling_curve():
    """300 s x 2^attempts, capped at 3600 s, measured from the failure and read
    back out of the Node annotations so a reconciler restart cannot reset it."""

    current = node()
    core = CoreApi([current])
    batch = BatchApi(job(condition="Failed", age_seconds=900, failed_at=NOW))
    clock = [NOW]

    def pass_now(seconds: int) -> dict[str, int]:
        clock[0] = NOW + timedelta(seconds=seconds)
        # A brand-new instance every pass: the backoff must survive a restart.
        result = reconciler(core, batch, now=lambda: clock[0]).reconcile_once()
        apply_patches(current, core)
        return result

    assert pass_now(299)["failed"] == 1, "the first retry waits 300 s"
    assert batch.deleted == [], batch.deleted
    assert INSTALLER_ATTEMPTS_ANNOTATION not in current.metadata.annotations, (
        current.metadata.annotations
    )

    assert pass_now(301)["failed"] == 1, "at 300 s the failed Job is deleted"
    assert len(batch.deleted) == 1, batch.deleted
    assert current.metadata.annotations[INSTALLER_ATTEMPTS_ANNOTATION] == "1", (
        current.metadata.annotations
    )

    # Second failure: 600 s, not 300 s.
    batch.existing = job(
        condition="Failed", age_seconds=0, failed_at=NOW + timedelta(seconds=400)
    )
    assert pass_now(400 + 599)["failed"] == 1, "the second retry waits 600 s"
    assert len(batch.deleted) == 1, batch.deleted
    assert pass_now(400 + 601)["failed"] == 1, "600 s later it retries"
    assert len(batch.deleted) == 2, batch.deleted
    assert current.metadata.annotations[INSTALLER_ATTEMPTS_ANNOTATION] == "2", (
        current.metadata.annotations
    )

    # Cap: at eight attempts the curve would be 21 h; it must stop at one hour.
    current.metadata.annotations[INSTALLER_ATTEMPTS_ANNOTATION] = "8"
    batch.existing = job(
        condition="Failed", age_seconds=0, failed_at=NOW + timedelta(seconds=5000)
    )
    assert pass_now(5000 + 3599)["failed"] == 1, "capped backoff has not elapsed"
    assert len(batch.deleted) == 2, batch.deleted
    assert pass_now(5000 + 3601)["failed"] == 1, "the cap is one hour, not 21 hours"
    assert len(batch.deleted) == 3, batch.deleted
    assert current.metadata.annotations[INSTALLER_ATTEMPTS_ANNOTATION] == "9", (
        current.metadata.annotations
    )
