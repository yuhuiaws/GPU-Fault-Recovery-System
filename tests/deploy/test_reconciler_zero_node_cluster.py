"""``deploy-node-installer-reconciler.sh`` for a GPU cluster without nodes.

With ``GPU_FAULT_ZERO_NODE_CLUSTER=true`` and no node selected the script
requires, keys and preflights none, renders the installer template from the
node the cluster's groups declare (``regional_parked_cluster.render_node``) and
applies the steady-state Reconciler, so a node that joins later installs this
release. A pinned template (a rollback's previous one) is kept as it is, with
no render node. Without the switch an empty node set refuses, as it always did.

The render node is never a node a Job runs for: the Reconciler fills each real
node into the Jobs it builds from the template. So for the same node group, a
template rendered from the declared node and one rendered from a real node of
the group build the very same Job for that node.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.node_installer_rendering import (
    InstallerIdentity,
    InstallerNode,
    load_installer_template,
    render_installer_job,
)
from gpu_fault_release import regional_parked_cluster as PARKED
from tests.regional._parked_cluster_support import GROUP, SITE, nodegroup

ROOT = Path(__file__).resolve().parents[2]
RECONCILER = ROOT / "deploy/node/deploy-node-installer-reconciler.sh"
JOB = ROOT / "deploy/node/run-hyperpod-installer-job.sh"
CLUSTER = "gpu-eks-cluster"
SELECTOR = f"gpu-fault.io/control-plane={SITE}"
REAL = "ip-10-0-33-205.us-west-2.compute.internal"
PROVIDER_ID = "aws:///us-west-2c/i-00000000000000001"
TEMPLATE_PREFIX = "gpu-fault-node-installer-template-"


def _proof() -> dict[str, Any]:
    return PARKED.eks_parked(
        {GROUP: nodegroup()}, selector=SELECTOR, nodepools=lambda: None
    )


def _target() -> Any:
    return SimpleNamespace(
        cluster_id=CLUSTER, provider="eks-managed", hyperpod_cluster_name=""
    )


def _render_node() -> dict[str, Any]:
    return PARKED.render_node(_proof(), _target())


# --- the script ------------------------------------------------------------------


@pytest.fixture
def zero_node_environment(tmp_path: Path) -> dict[str, str]:
    binary = tmp_path / "bin"
    binary.mkdir()
    kubectl = binary / "kubectl"
    kubectl.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """
            import base64
            import json
            import os
            import sys
            from pathlib import Path
            import yaml

            root = Path(os.environ["TEST_ROOT"])
            args = sys.argv[1:]
            with (root / "calls.jsonl").open("a") as handle:
                handle.write(json.dumps(args) + "\\n")
            if "create" in args:
                index = args.index("configmap")
                name = args[index + 1]
                source = next(item for item in args if item.startswith("--from-file="))
                path = source.split("=", 2)[2]
                print(yaml.safe_dump({
                    "apiVersion": "v1", "kind": "ConfigMap",
                    "metadata": {"name": name},
                    "data": {"job.yaml": Path(path).read_text()},
                }))
            elif "get" in args:
                index = args.index("get")
                kind = args[index + 1]
                name = args[index + 2] if len(args) > index + 2 else ""
                if kind == "nodes":
                    print('{"items":[]}')
                elif kind == "configmap" and name == "candidate-bundle":
                    print(json.dumps({"binaryData": {"gpu-fault-node-installer-0.10.0.tar.gz": ""}}))
                elif kind == "configmap" and name == "candidate-wheel":
                    print(json.dumps({"binaryData": {"executor.whl": ""}}))
                elif kind == "secret" and name == "gpu-fault-node-action-keys":
                    print('{"data":{}}')
                elif kind == "secret" and name == "gpu-fault-regional-connection":
                    data = {
                        "cluster-id": "gpu-eks-cluster",
                        "cluster-token": "fixture-only",
                        "ca.crt": "fixture-only",
                        "control-plane-url": "https://control.example",
                    }
                    print(json.dumps({"data": {
                        key: base64.b64encode(value.encode()).decode()
                        for key, value in data.items()
                    }}))
                elif kind == "jobs":
                    print('{"items":[]}')
                elif kind == "configmap" and (root / "configmaps" / name).is_file():
                    print(json.dumps({"data": {
                        "job.yaml": (root / "configmaps" / name).read_text()
                    }}))
                else:
                    raise SystemExit("unexpected read " + json.dumps(args))
            elif "apply" in args:
                text = Path(args[args.index("-f") + 1]).read_text()
                name = "admissions.jsonl" if "--dry-run=server" in args else "applies.jsonl"
                with (root / name).open("a") as handle:
                    handle.write(json.dumps(list(yaml.safe_load_all(text))) + "\\n")
            else:
                raise SystemExit("unexpected kubectl " + json.dumps(args))
            """
        ),
        encoding="utf-8",
    )
    kubectl.chmod(0o755)
    batch = binary / "python3"
    batch.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """
            import os
            import sys
            if sys.argv[1:3] == ["-m", "gpu_fault_release.regional_node_batch"]:
                raise SystemExit("a zero-node cluster must never run the node batch")
            if any(item.endswith("provision_node_action_keys.py") for item in sys.argv):
                raise SystemExit("a zero-node cluster must never provision node keys")
            os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
            """
        ),
        encoding="utf-8",
    )
    batch.chmod(0o755)
    return {
        "HOME": str(tmp_path),
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
        "PYTHONDONTWRITEBYTECODE": "1",
        "KUBECONFIG": "/dev/null",
        "TEST_ROOT": str(tmp_path),
        "GPU_FAULT_KUBECTL_CONTEXT": "gpu-eks-context",
        "GPU_FAULT_CLUSTER_ID": CLUSTER,
        "GPU_FAULT_CLUSTER_PROVIDER": "eks-managed",
        "GPU_FAULT_DCGM_HOSTENGINE": "host",
        "GPU_FAULT_NODE_SELECTOR": SELECTOR,
        "GPU_FAULT_INSTALLER_CONFIG_DIGEST": "candidate-config",
        "GPU_FAULT_INSTALLER_ARTIFACT_SHA256": "1" * 64,
        "GPU_FAULT_INSTALLER_BUNDLE_SHA256": "2" * 64,
        "GPU_FAULT_INSTALLER_TEMPLATE_SHA256": "3" * 64,
        "GPU_FAULT_NODE_COMPATIBILITY_DIGEST": "4" * 64,
        "GPU_FAULT_WHEEL_CONFIG_MAP": "candidate-wheel",
        "GPU_FAULT_EXECUTOR_WHEEL_FILENAME": "executor.whl",
        "GPU_FAULT_INSTALLER_CONFIG_MAP": "candidate-bundle",
        "GPU_FAULT_NODE_INSTALLER_IMAGE": "registry.example/installer@sha256:"
        + "c" * 64,
        "GPU_FAULT_SYNC_INSTALLED_RESOURCE_REGISTRY": "false",
        "GPU_FAULT_WAIT_FOR_RECONCILER_ROLLOUT": "false",
        **PARKED.reconciler_environment(_proof(), _target()),
    }


def _run(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RECONCILER)],
        env=environment,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def _records(root: Path, name: str) -> list[list[dict[str, Any]]]:
    path = root / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_zero_node_cluster_gets_its_steady_reconciler_and_template(
    zero_node_environment: dict[str, str], tmp_path: Path
) -> None:
    result = _run(zero_node_environment)

    assert result.returncode == 0, result.stderr
    assert f"zero-node GPU cluster {CLUSTER}: no node selected" in result.stderr
    applied = [
        document for batch in _records(tmp_path, "applies.jsonl") for document in batch
    ]
    template = next(item for item in applied if item.get("kind") == "ConfigMap")
    job = yaml.safe_load(template["data"]["job.yaml"])
    pod = job["spec"]["template"]["spec"]
    assert pod["nodeName"] == PARKED.RENDER_NODE_NAME, "rendered from the declared node"
    reconciler = next(
        item
        for item in applied
        if item.get("kind") == "Deployment"
        and item["metadata"]["name"] == "gpu-fault-node-installer-reconciler"
    )
    environment = {
        item["name"]: item.get("value")
        for item in reconciler["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert environment["GPU_FAULT_INSTALLER_ARTIFACT_SHA256"] == "1" * 64, (
        "the Reconciler carries the release's node runtime for every joiner"
    )
    assert "*" in json.dumps(reconciler), "the steady state allows every node"


def test_a_zero_node_cluster_provisions_no_node_key_with_a_fleet_master(
    zero_node_environment: dict[str, str], tmp_path: Path
) -> None:
    """With ``GPU_FAULT_FLEET_MASTER_FILE`` the script runs the node-key helper
    for the nodes it selects; with none selected it never does -- the helper
    would refuse the empty inventory -- and only requires the key Secret the
    steady-state Reconciler mounts to exist."""

    master = tmp_path / "fleet-master"
    master.write_text("fixture-only\n", encoding="utf-8")
    master.chmod(0o600)

    result = _run({**zero_node_environment, "GPU_FAULT_FLEET_MASTER_FILE": str(master)})

    assert result.returncode == 0, result.stderr
    assert "must never provision node keys" not in result.stderr
    calls = [
        json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()
    ]
    assert any(
        "secret" in call and "gpu-fault-node-action-keys" in call for call in calls
    ), "the existing key Secret is required, not written"


def test_a_zero_node_preflight_proves_the_template_without_a_node_batch(
    zero_node_environment: dict[str, str], tmp_path: Path
) -> None:
    result = _run(
        {**zero_node_environment, "GPU_FAULT_RECONCILER_PREFLIGHT_ONLY": "true"}
    )

    assert result.returncode == 0, result.stderr
    verdict = json.loads(result.stdout.strip().splitlines()[-1])
    assert verdict["node_count"] == 0 and verdict["status"] == "PASSED"
    assert verdict["zero_node_cluster"] is True
    assert _records(tmp_path, "applies.jsonl") == [], "a preflight applies nothing"
    assert _records(tmp_path, "admissions.jsonl"), (
        "but admits the template and Reconciler"
    )


def test_a_pinned_template_is_kept_without_a_render_node(
    zero_node_environment: dict[str, str], tmp_path: Path
) -> None:
    """A rollback: the template a forward run rendered is pinned by name and
    content, so nothing is rendered and no declared node is needed."""

    first = _run(zero_node_environment)
    assert first.returncode == 0, first.stderr
    rendered = next(
        document
        for batch in _records(tmp_path, "applies.jsonl")
        for document in batch
        if document.get("kind") == "ConfigMap"
        and document["metadata"]["name"].startswith(TEMPLATE_PREFIX)
    )
    name, text = rendered["metadata"]["name"], rendered["data"]["job.yaml"]
    (tmp_path / "configmaps").mkdir()
    (tmp_path / "configmaps" / name).write_text(text)
    (tmp_path / "applies.jsonl").unlink()

    result = _run(
        {
            **zero_node_environment,
            "GPU_FAULT_ZERO_NODE_RENDER_NODE": "",
            "GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP": name,
            "GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256": hashlib.sha256(
                text.encode()
            ).hexdigest(),
        }
    )

    assert result.returncode == 0, result.stderr
    assert f"the pinned installer template {name} is kept" in result.stderr
    applied = [
        document for batch in _records(tmp_path, "applies.jsonl") for document in batch
    ]
    assert not [
        item
        for item in applied
        if item.get("kind") == "ConfigMap"
        and item["metadata"]["name"].startswith(TEMPLATE_PREFIX)
    ], "no template rendered"
    reconciler = next(item for item in applied if item.get("kind") == "Deployment")
    assert name in json.dumps(reconciler), "the Reconciler points at the pinned one"


def test_an_empty_render_node_without_a_pinned_template_refuses(
    zero_node_environment: dict[str, str], tmp_path: Path
) -> None:
    result = _run({**zero_node_environment, "GPU_FAULT_ZERO_NODE_RENDER_NODE": ""})

    assert result.returncode == 2
    assert f"zero-node cluster {CLUSTER} has no valid render node" in result.stderr
    assert _records(tmp_path, "applies.jsonl") == [], "nothing applied"


def test_an_empty_node_set_still_refuses_without_the_zero_node_switch(
    zero_node_environment: dict[str, str],
) -> None:
    environment = {
        key: value
        for key, value in zero_node_environment.items()
        if not key.startswith("GPU_FAULT_ZERO_NODE_")
    }

    result = _run(environment)

    assert result.returncode == 1
    assert "no GPU node found" in result.stderr


# --- the golden render equivalence ----------------------------------------------------


def _inputs_entry(document: dict[str, Any]) -> dict[str, Any]:
    """What the reconciler script keeps of a node for rendering (its jq)."""

    metadata = document["metadata"]
    return {
        "metadata": {
            "uid": metadata["uid"],
            "labels": {
                "node.kubernetes.io/instance-type": metadata["labels"][
                    "node.kubernetes.io/instance-type"
                ]
            },
        },
        "spec": {"providerID": document["spec"].get("providerID", "")},
        "status": {
            "addresses": [
                item
                for item in document["status"]["addresses"]
                if item["type"] == "InternalIP"
            ]
        },
    }


def _render_template(tmp_path: Path, name: str, document: dict[str, Any]) -> str:
    root = tmp_path / name.replace(".", "-")
    root.mkdir()
    inputs = root / "nodes.json"
    inputs.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cluster_id": CLUSTER,
                "context": "gpu-eks-context",
                "namespace": "gpu-fault-system",
                "node_action_keys_secret": "gpu-fault-node-action-keys",
                "connection_secret": "gpu-fault-regional-connection",
                "references_validated": True,
                "nodes": {name: _inputs_entry(document)},
            }
        )
    )
    inputs.chmod(0o600)
    kubectl = root / "kubectl"
    kubectl.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    kubectl.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            str(JOB),
            "--node",
            name,
            "--node-inputs",
            str(inputs),
            "--render-only",
        ],
        env={
            "PATH": str(root) + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(root),
            "GPU_FAULT_CLUSTER_ID": CLUSTER,
            "GPU_FAULT_KUBECTL_CONTEXT": "gpu-eks-context",
            "GPU_FAULT_CONNECTION_MODE": "regional",
            "GPU_FAULT_CLUSTER_PROVIDER": "eks-managed",
            "GPU_FAULT_DCGM_HOSTENGINE": "host",
            "GPU_FAULT_INSTALLER_ARTIFACT_SHA256": "1" * 64,
            "GPU_FAULT_INSTALLER_CONFIG_DIGEST": "candidate-config",
            "GPU_FAULT_INSTALLER_TEMPLATE_SHA256": "3" * 64,
            "GPU_FAULT_VERSION": "0.10.0",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _job_for_real_node(text: str) -> dict[str, Any]:
    identity = InstallerIdentity(
        namespace="gpu-fault-system",
        config_digest="candidate-config",
        artifact_sha256="1" * 64,
        bundle_sha256="2" * 64,
        template_sha256="3" * 64,
        node_action_keys_secret="gpu-fault-node-action-keys",
        deadline_seconds=840,
        metrics_url_template="http://{node_ip}:9400/metrics",
        template_content_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    template = load_installer_template(
        text.encode(),
        expected_sha256=identity.template_content_sha256,
        identity=identity,
        origin="golden template",
    )
    real = InstallerNode(REAL, "uid-real", "10.0.33.205", "p5en.48xlarge", PROVIDER_ID)
    return render_installer_job(template, real, identity, "gpu-fault-install-real")


def test_a_joiner_gets_the_same_job_from_the_declared_render_node(
    tmp_path: Path,
) -> None:
    """Golden: the group's real node and the node it declares render templates
    that build the very same Job for the real node."""

    real_document = {
        "metadata": {
            "name": REAL,
            "uid": "uid-real",
            "labels": {
                **nodegroup()["labels"],
                "node.kubernetes.io/instance-type": "p5en.48xlarge",
                "eks.amazonaws.com/nodegroup": GROUP,
            },
        },
        "spec": {"providerID": PROVIDER_ID},
        "status": {"addresses": [{"type": "InternalIP", "address": "10.0.33.205"}]},
    }
    declared = _render_node()

    from_real = _render_template(tmp_path, REAL, real_document)
    from_declared = _render_template(tmp_path, declared["metadata"]["name"], declared)

    assert from_real != from_declared, "the templates name their render nodes"
    assert _job_for_real_node(from_declared) == _job_for_real_node(from_real)
