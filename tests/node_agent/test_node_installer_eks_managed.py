"""Installer rendering, reconciler selection and node snapshots for plain EKS nodes."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault.cluster_scope import NODE_SELECTOR_ENV
from gpu_fault.node_installer_rendering import (
    InstallerNode,
    load_installer_template,
    render_installer_job,
    set_installer_environment,
)
from gpu_fault_release import regional_node_batch as batch
from tests.deploy.test_installer_template_identity import pinned_identity, template_job
from tests.deploy.test_node_preflight_inputs import inputs
from tests.node_agent.test_node_installer_reconciler import (
    BatchApi,
    CoreApi,
    node,
    reconciler,
)
from tests.regional.test_cov95_release_node_batch import SCOPE

PROVIDER_ID = "aws:///us-west-2c/i-0222042756c3a2a68"
FQDN = "ip-10-0-33-205.us-west-2.compute.internal"


def _template_with_provider_id() -> dict:
    job = template_job()
    env = job["spec"]["template"]["spec"]["containers"][0]["env"]
    env.append({"name": "TARGET_NODE_PROVIDER_ID", "value": ""})
    return job


def _load(job: dict):
    text = yaml.safe_dump(job)
    identity = pinned_identity(text)
    return (
        load_installer_template(
            text.encode(),
            expected_sha256=identity.template_content_sha256,
            identity=identity,
            origin="test template",
        ),
        identity,
    )


def _installer_env(rendered: dict) -> dict[str, str]:
    container = rendered["spec"]["template"]["spec"]["containers"][0]
    return {item["name"]: item["value"] for item in container["env"]}


def test_installer_node_carries_an_optional_provider_id() -> None:
    plain = InstallerNode("node-a", "uid-a", "10.0.0.1", "ml.p5.48xlarge")
    assert plain.provider_id == "", "the provider id defaults to empty"
    eks = InstallerNode(FQDN, "uid-e", "10.0.33.205", "p5en.48xlarge", PROVIDER_ID)
    assert eks.provider_id == PROVIDER_ID, "the provider id is kept verbatim"
    with pytest.raises(ValueError, match="provider id"):
        InstallerNode("node-a", "uid-a", "10.0.0.1", "ml.p5.48xlarge", "aws:/// bad")


def test_render_fills_the_provider_id_when_the_template_declares_it() -> None:
    job, identity = _load(_template_with_provider_id())
    eks = InstallerNode(FQDN, "uid-e", "10.0.33.205", "p5en.48xlarge", PROVIDER_ID)
    env = _installer_env(render_installer_job(job, eks, identity, "install-e"))
    assert env["TARGET_NODE_PROVIDER_ID"] == PROVIDER_ID, (
        "the Job must carry the node's spec.providerID"
    )
    assert env["TARGET_NODE_UID"] == "uid-e", "the Node UID is still rendered"


def test_render_tolerates_a_pinned_template_without_the_provider_id() -> None:
    """A rollback's steady template predates the entry; it keeps its own value."""

    job, identity = _load(template_job())
    eks = InstallerNode(FQDN, "uid-e", "10.0.33.205", "p5en.48xlarge", PROVIDER_ID)
    env = _installer_env(render_installer_job(job, eks, identity, "install-e"))
    assert "TARGET_NODE_PROVIDER_ID" not in env, (
        "an older template is rendered without inventing the env entry"
    )
    container = {"env": [{"name": "TARGET_NODE_UID", "value": ""}]}
    with pytest.raises(RuntimeError, match="no env entry for TARGET_NODE_IP"):
        set_installer_environment(
            container, {"TARGET_NODE_UID": "u", "TARGET_NODE_IP": "10.0.0.1"}
        )


def test_reconciler_selector_defaults_to_the_hyperpod_label(monkeypatch) -> None:
    monkeypatch.delenv(NODE_SELECTOR_ENV, raising=False)
    instance = reconciler(CoreApi([]), BatchApi())
    assert (
        instance.node_selector == "sagemaker.amazonaws.com/cluster-name=hp-cluster-a"
    ), "without GPU_FAULT_NODE_SELECTOR the reconciler selects on the HyperPod label"


def test_reconciler_selector_follows_the_connection_secret(monkeypatch) -> None:
    monkeypatch.setenv(NODE_SELECTOR_ENV, "nvidia.com/gpu.present=true")
    instance = reconciler(CoreApi([]), BatchApi())
    assert instance.node_selector == "nvidia.com/gpu.present=true", (
        "GPU_FAULT_NODE_SELECTOR must override the HyperPod label"
    )
    monkeypatch.setenv(NODE_SELECTOR_ENV, "nvidia.com/gpu.present=true,")
    with pytest.raises(ValueError, match="node selector"):
        instance.node_selector


def test_reconciler_passes_the_node_provider_id_into_the_job() -> None:
    job = reconciler(CoreApi([]), BatchApi())
    target = node(name=FQDN, uid="uid-e", instance_type="p5en.48xlarge")
    target.spec = SimpleNamespace(provider_id=PROVIDER_ID)
    template = job.job_template
    template["spec"]["template"]["spec"]["containers"][0]["env"].append(
        {"name": "TARGET_NODE_PROVIDER_ID", "value": "old"}
    )
    rendered = job._build_job(target, "gpu-fault-install-e")
    assert _installer_env(rendered)["TARGET_NODE_PROVIDER_ID"] == PROVIDER_ID, (
        "the reconciler must read spec.providerID off the live Node"
    )
    plain = node()
    rendered = job._build_job(plain, "gpu-fault-install-a")
    assert _installer_env(rendered)["TARGET_NODE_PROVIDER_ID"] == "", (
        "a node without a spec renders an empty provider id"
    )


def test_private_snapshot_carries_the_provider_id(tmp_path: Path) -> None:
    document = inputs()
    document["nodes"]["node-a"]["spec"] = {"providerID": PROVIDER_ID}
    path = tmp_path / "nodes.json"
    path.write_text(json.dumps(document))
    path.chmod(0o600)
    nodes = batch.read_private_snapshot(path, SCOPE)
    assert nodes["node-a"].provider_id == PROVIDER_ID, (
        "spec.providerID from the deploy script's snapshot must reach the node"
    )
    without = copy.deepcopy(inputs())
    path.write_text(json.dumps(without))
    assert batch.read_private_snapshot(path, SCOPE)["node-a"].provider_id == "", (
        "a snapshot without spec stays loadable"
    )
