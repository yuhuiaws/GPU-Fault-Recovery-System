from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpu_fault import node_installer_reconciler as implementation
from tests.node_agent.test_node_installer_reconciler import (
    ARTIFACT,
    BUNDLE,
    TEMPLATE,
    ApiError,
    BatchApi,
    CoreApi,
    job,
    node,
    reconciler,
)


def installed_node(activation=""):
    annotations = {
        implementation.INSTALLER_VERSION_ANNOTATION: "0.10.0",
        implementation.INSTALLER_DIGEST_ANNOTATION: "config-sha",
        implementation.INSTALLER_ARTIFACT_ANNOTATION: ARTIFACT,
        implementation.INSTALLER_BUNDLE_ANNOTATION: BUNDLE,
        implementation.INSTALLER_TEMPLATE_ANNOTATION: TEMPLATE,
        implementation.INSTALLER_NODE_UID_ANNOTATION: "node-uid",
        implementation.INSTALLER_STATE_ANNOTATION: "Succeeded",
    }
    if activation:
        annotations[implementation.INSTALLER_ACTIVATION_ANNOTATION] = activation
    value = node(annotations=annotations)
    value.metadata.resource_version = "1"
    return value


class MutableCore(CoreApi):
    drift = None

    def read_node(self, name, **kwargs):
        current = self.nodes[0]
        if self.drift == "uid":
            return SimpleNamespace(
                metadata=SimpleNamespace(
                    uid="recreated",
                    resource_version="2",
                    annotations=current.metadata.annotations,
                )
            )
        if self.drift == "activation":
            return SimpleNamespace(
                metadata=SimpleNamespace(
                    uid=current.metadata.uid,
                    resource_version="2",
                    annotations={
                        implementation.INSTALLER_ACTIVATION_ANNOTATION: "b" * 64
                    },
                )
            )
        return current

    def patch_node(self, name, body, **kwargs):
        current = self.nodes[0]
        assert body["metadata"]["uid"] == current.metadata.uid
        assert body["metadata"]["resourceVersion"] == current.metadata.resource_version
        super().patch_node(name, body, **kwargs)
        current.metadata.annotations.update(body["metadata"]["annotations"])
        current.metadata.resource_version = str(
            int(current.metadata.resource_version) + 1
        )


class NamedBatch(BatchApi):
    def read_namespaced_job(self, name, namespace, **kwargs):
        if name not in self.jobs:
            raise ApiError(404)
        return self.jobs[name]

    def create_namespaced_job(self, namespace, body, **kwargs):
        super().create_namespaced_job(namespace, body, **kwargs)
        self.jobs[body["metadata"]["name"]] = job()


def test_same_release_activation_creates_a_distinct_job_and_requires_its_completion():
    current = installed_node("a" * 64)
    core, batch = MutableCore([current]), NamedBatch()
    first = reconciler(core, batch)
    assert first.reconcile_once()["created"] == 1
    name = batch.created[0][1]["metadata"]["name"]
    assert (
        implementation.INSTALLER_ACTIVATED_ANNOTATION
        not in current.metadata.annotations
    )
    assert first.reconcile_once()["running"] == 1
    batch.jobs[name] = job(condition="Complete")
    assert first.reconcile_once()["succeeded"] == 1
    assert (
        current.metadata.annotations[implementation.INSTALLER_ACTIVATED_ANNOTATION]
        == "a" * 64
    )
    assert reconciler(core, batch).reconcile_once()["current"] == 1
    assert len(batch.created) == 1
    current.metadata.annotations[implementation.INSTALLER_ACTIVATION_ANNOTATION] = (
        "b" * 64
    )
    assert reconciler(core, batch).reconcile_once()["created"] == 1
    assert batch.created[-1][1]["metadata"]["name"] != name


@pytest.mark.parametrize("drift", ["uid", "activation"])
def test_reconciler_cannot_acknowledge_activation_for_a_replaced_binding(drift):
    current = installed_node("a" * 64)
    core, batch = MutableCore([current]), NamedBatch()
    controller = reconciler(core, batch)
    assert controller.reconcile_once()["created"] == 1
    name = batch.created[0][1]["metadata"]["name"]
    batch.jobs[name] = job(condition="Complete")
    core.drift = drift
    assert controller.reconcile_once()["error"] == 1
    assert (
        implementation.INSTALLER_ACTIVATED_ANNOTATION
        not in current.metadata.annotations
    )


@pytest.mark.parametrize("identity", ["not-an-activation", "a" * 63, "../unsafe"])
def test_malformed_activation_never_creates_an_installer_job(identity):
    core, batch = MutableCore([installed_node(identity)]), NamedBatch()
    assert reconciler(core, batch).reconcile_once()["error"] == 1
    assert batch.created == [] and core.patches == []
