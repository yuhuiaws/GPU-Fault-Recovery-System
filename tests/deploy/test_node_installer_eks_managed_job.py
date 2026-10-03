"""The installer Job on a plain EKS (``eks-managed``) node.

EKS nodes are FQDNs (``ip-10-0-33-205.us-west-2.compute.internal``) with an
``aws:///<zone>/i-...`` provider id, so the Job name has to be made DNS-1123 safe
and kept under 63 characters, and the node identity the installer registers is
the EC2 instance id rather than the Kubernetes Node UID. HyperPod nodes keep the
Job name and the UID identity they always had.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

SCRIPT = (
    Path(__file__).resolve().parents[2] / "deploy/node/run-hyperpod-installer-job.sh"
)
FQDN = "ip-10-0-33-205.us-west-2.compute.internal"
PROVIDER_ID = "aws:///us-west-2c/i-00000000000000001"
DNS_1123 = re.compile(r"^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$")


def _inputs(node_name: str, *, provider_id: str | None = PROVIDER_ID) -> dict:
    node = {
        "metadata": {
            "uid": "node-uid-" + node_name[:8],
            "labels": {"node.kubernetes.io/instance-type": "p5en.48xlarge"},
        },
        "status": {"addresses": [{"type": "InternalIP", "address": "10.0.33.205"}]},
    }
    if provider_id is not None:
        node["spec"] = {"providerID": provider_id}
    return {
        "schema_version": 1,
        "cluster_id": "gpu-test",
        "context": "gpu-context",
        "namespace": "gpu-fault-system",
        "node_action_keys_secret": "gpu-fault-node-action-keys",
        "connection_secret": "gpu-fault-regional-connection",
        "references_validated": True,
        "nodes": {node_name: node},
    }


def _render(
    tmp_path: Path,
    node_name: str,
    *,
    provider: str | None = None,
    provider_id: str | None = PROVIDER_ID,
) -> subprocess.CompletedProcess[str]:
    snapshot = tmp_path / "nodes.json"
    snapshot.write_text(json.dumps(_inputs(node_name, provider_id=provider_id)))
    snapshot.chmod(0o600)
    binary = tmp_path / "kubectl"
    binary.write_text(
        '#!/bin/sh\nprintf "unexpected kubectl invocation\\n" >&2\nexit 99\n',
        encoding="utf-8",
    )
    binary.chmod(0o755)
    environment = {
        "PATH": str(tmp_path) + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "GPU_FAULT_CLUSTER_ID": "gpu-test",
        "GPU_FAULT_KUBECTL_CONTEXT": "gpu-context",
        "GPU_FAULT_CONNECTION_MODE": "regional",
        "GPU_FAULT_INSTALLER_ARTIFACT_SHA256": "a" * 64,
        "GPU_FAULT_INSTALLER_CONFIG_DIGEST": "b" * 64,
        "GPU_FAULT_VERSION": "0.10.0",
    }
    if provider is not None:
        environment["GPU_FAULT_CLUSTER_PROVIDER"] = provider
    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--node",
            node_name,
            "--node-inputs",
            str(snapshot),
            "--render-only",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def _job(result: subprocess.CompletedProcess[str]) -> dict:
    assert result.returncode == 0, result.stderr
    return yaml.safe_load(result.stdout)


def _env(job: dict) -> dict[str, str]:
    container = job["spec"]["template"]["spec"]["containers"][0]
    return {item["name"]: item.get("value") for item in container["env"]}


def test_hyperpod_node_keeps_its_historical_job_name(tmp_path: Path) -> None:
    job = _job(_render(tmp_path, "hyperpod-i-0abc123def456"))
    assert job["metadata"]["name"] == "gpu-fault-install-i-0abc123def456", (
        "HyperPod Job names must not change"
    )
    env = _env(job)
    assert env["CLUSTER_PROVIDER"] == "hyperpod", "provider defaults to hyperpod"
    assert env["TARGET_NODE_PROVIDER_ID"] == PROVIDER_ID, (
        "the provider id travels with the Job even on HyperPod (unused there)"
    )


def test_fqdn_node_gets_a_dns_1123_job_name_under_63_characters(tmp_path: Path) -> None:
    first = _job(_render(tmp_path, FQDN, provider="eks-managed"))
    again = _job(_render(tmp_path, FQDN, provider="eks-managed"))
    name = first["metadata"]["name"]
    assert DNS_1123.fullmatch(name), f"Job name {name!r} is not DNS-1123"
    assert len(name) <= 63, f"Job name {name!r} exceeds 63 characters"
    assert name.startswith("gpu-fault-install-ip-10-0-33-205-us-west-2"), (
        "the name must keep a readable prefix of the node name"
    )
    assert name == again["metadata"]["name"], "the Job name must be deterministic"
    other = _job(
        _render(
            tmp_path,
            "ip-10-0-33-206.us-west-2.compute.internal",
            provider="eks-managed",
        )
    )
    assert other["metadata"]["name"] != name, (
        "two nodes whose names differ only in the truncated tail must not collide"
    )
    assert first["spec"]["template"]["spec"]["nodeName"] == FQDN, (
        "the Pod is still pinned to the full node name"
    )


def test_a_very_long_node_name_still_fits(tmp_path: Path) -> None:
    long_name = "ip-10-0-33-205." + "sub." * 12 + "compute.internal"
    job = _job(_render(tmp_path, long_name, provider="eks-managed"))
    name = job["metadata"]["name"]
    assert len(name) <= 63 and DNS_1123.fullmatch(name), name


def test_eks_managed_job_registers_the_ec2_instance_id(tmp_path: Path) -> None:
    job = _job(_render(tmp_path, FQDN, provider="eks-managed"))
    env = _env(job)
    assert env["CLUSTER_PROVIDER"] == "eks-managed", "the provider is baked in"
    assert env["TARGET_NODE_PROVIDER_ID"] == PROVIDER_ID, "spec.providerID is passed"
    script = job["spec"]["template"]["spec"]["containers"][0]["args"][-1]
    assert '--node-instance-id "${node_instance_id}"' in script, (
        "the installer must register the resolved node identity"
    )
    fragment = script[script.index('node_instance_id="${TARGET_NODE_UID}"') :]
    fragment = fragment[: fragment.index("fi\n") + 3]
    for provider, expected in (
        ("eks-managed", "i-00000000000000001"),
        ("hyperpod", "node-uid-ip-10-0-"),
    ):
        resolved = subprocess.run(
            ["bash", "-ceu", fragment + '\nprintf "%s" "${node_instance_id}"'],
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "TARGET_NODE_UID": env["TARGET_NODE_UID"],
                "TARGET_NODE_PROVIDER_ID": PROVIDER_ID,
                "TARGET_NODE_NAME": FQDN,
                "CLUSTER_PROVIDER": provider,
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert resolved.returncode == 0, resolved.stderr
        assert resolved.stdout == expected, (
            f"{provider}: expected node identity {expected!r}, got {resolved.stdout!r}"
        )
    broken = subprocess.run(
        ["bash", "-ceu", fragment],
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "TARGET_NODE_UID": "uid",
            "TARGET_NODE_PROVIDER_ID": "",
            "TARGET_NODE_NAME": FQDN,
            "CLUSTER_PROVIDER": "eks-managed",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert broken.returncode != 0 and "no EC2 instance id" in broken.stderr, (
        "an eks-managed node without a provider id must fail closed, not fall "
        "back to the Node UID"
    )


def test_missing_provider_id_renders_empty_and_hyperpod_is_unaffected(
    tmp_path: Path,
) -> None:
    job = _job(_render(tmp_path, "hyperpod-i-0abc", provider_id=None))
    assert _env(job)["TARGET_NODE_PROVIDER_ID"] == "", (
        "an older input snapshot without spec.providerID renders an empty id"
    )


@pytest.mark.parametrize("provider", ["eks", "EKS-MANAGED", "sagemaker"])
def test_unknown_provider_is_refused_before_rendering(tmp_path: Path, provider) -> None:
    result = _render(tmp_path, FQDN, provider=provider)
    assert result.returncode == 2, result.stderr
    assert "GPU_FAULT_CLUSTER_PROVIDER must be hyperpod or eks-managed" in result.stderr
