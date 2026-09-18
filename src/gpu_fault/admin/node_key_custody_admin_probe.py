"""Read-only live bindings and receipt checks for administrator custody."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gpu_fault.admin.bootstrap_common import (
    ClusterIdentity,
    CommandRunner,
    ReadOnlyProbeRunner,
)
from gpu_fault.admin.node_key_custody import namespace_uid
from gpu_fault.admin.node_key_custody_admin_config import AdminCustodyRegistration
from gpu_fault.admin.node_key_custody_chain import verify_chain
from gpu_fault.admin.node_key_custody_crypto import (
    CustodyCrypto,
    parse,
    private_command_environment,
    read_regular,
)
from gpu_fault.admin.node_key_custody_models import (
    Chain,
    CustodyBinding,
    CustodyError,
    MasterSource,
    ReleaseBinding,
    SiteBinding,
    Transaction,
)
from gpu_fault.admin.node_key_proof import read_node_key_proof
from gpu_fault.admin.release_artifacts import verify_prebuilt_release


@dataclass(frozen=True)
class AdminNodeKeyContext:
    state_dir: Path
    repository_root: Path
    site_id: str
    cpu_eks_arn: str
    cpu_kubeconfig: Path
    gpu_kubeconfig: Path
    namespace: str
    cluster: ClusterIdentity
    cluster_id: str
    release_manifest: Path
    runtime_profile_version: str
    agent_config_digest: str
    on_write: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def kubectl(self, plane: str) -> list[str]:
        if plane == "cpu":
            return ["kubectl", "--kubeconfig", str(self.cpu_kubeconfig)]
        return [
            "kubectl",
            "--kubeconfig",
            str(self.gpu_kubeconfig),
            "--context",
            self.cluster.context,
        ]


class CustodyReadRunner(ReadOnlyProbeRunner):
    def run(self, arguments: Sequence[str], **kwargs: Any) -> str:
        return super().run(
            arguments,
            **{
                **kwargs,
                "env": private_command_environment(),
                "capture": True,
                "sensitive": True,
            },
        )


def private_json(runner: CommandRunner, arguments: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(
            runner.run(
                arguments,
                capture=True,
                sensitive=True,
                timeout_seconds=30,
                env=private_command_environment(),
            )
        )
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (ValueError, TypeError):
        raise CustodyError("administrator custody read returned invalid JSON") from None


def custody_verifier(
    runner: CommandRunner, registration: AdminCustodyRegistration
) -> CustodyCrypto:
    def command(
        arguments: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        if arguments[:3] == ["aws", "kms", "sign"]:
            raise CustodyError("administrator custody probes cannot sign receipts")
        output = runner.run(
            arguments,
            capture=True,
            sensitive=True,
            timeout_seconds=kwargs["timeout_seconds"],
            env=kwargs.get("environment"),
        )
        return subprocess.CompletedProcess(arguments, 0, output, "")

    return CustodyCrypto(
        Path(registration.selection.trust),
        registration.trust_sha256,
        runner=command,
    )


def live_node_uids(
    runner: CommandRunner, context: AdminNodeKeyContext
) -> dict[str, str]:
    document = private_json(
        runner,
        [
            *context.kubectl("gpu"),
            "get",
            "nodes",
            "-l",
            "sagemaker.amazonaws.com/cluster-name=" + context.cluster.hyperpod_name,
            "-o",
            "json",
        ],
    )
    try:
        items = document["items"]
        if (
            document.get("apiVersion") != "v1"
            or document.get("kind") not in {"List", "NodeList"}
            or document.get("metadata", {}).get("continue")
            or not isinstance(items, list)
            or not items
        ):
            raise ValueError
        nodes: dict[str, str] = {}
        for item in items:
            metadata = item["metadata"]
            name, uid = metadata["name"], metadata["uid"]
            if (
                item.get("apiVersion") != "v1"
                or item.get("kind") != "Node"
                or metadata.get("deletionTimestamp")
                or metadata.get("labels", {}).get(
                    "sagemaker.amazonaws.com/cluster-name"
                )
                != context.cluster.hyperpod_name
                or not isinstance(name, str)
                or not name
                or not isinstance(uid, str)
                or not uid
                or name in nodes
                or uid in nodes.values()
            ):
                raise ValueError
            nodes[name] = uid
        return nodes
    except (AttributeError, KeyError, TypeError, ValueError):
        raise CustodyError(
            "administrator custody node inventory is incomplete or ambiguous"
        ) from None


def master_source(runner: CommandRunner, context: AdminNodeKeyContext) -> MasterSource:
    document = private_json(
        runner,
        [
            *context.kubectl("cpu"),
            "-n",
            context.namespace,
            "get",
            "secret",
            "gpu-fault-node-installer",
            "-o",
            "json",
        ],
    )
    try:
        metadata = document["metadata"]
        material = base64.b64decode(
            document["data"]["node-action-secret"], validate=True
        )
        if (
            document.get("apiVersion") != "v1"
            or document.get("kind") != "Secret"
            or document.get("type") != "Opaque"
            or "stringData" in document
            or metadata.get("name") != "gpu-fault-node-installer"
            or metadata.get("namespace") != context.namespace
            or not metadata.get("resourceVersion")
            or metadata.get("deletionTimestamp")
            or len(material.decode("utf-8")) < 32
            or material.strip() != material
        ):
            raise ValueError
        return MasterSource(
            secret_name="gpu-fault-node-installer",
            data_key="node-action-secret",
            secret_uid=metadata["uid"],
            sha256=hashlib.sha256(material).hexdigest(),
        )
    except (AttributeError, KeyError, TypeError, ValueError, UnicodeError):
        raise CustodyError("administrator custody master source is invalid") from None


def cluster_namespace_anchors(
    runner: CommandRunner, context: AdminNodeKeyContext
) -> dict[str, str]:
    anchors = {}
    for plane in ("cpu", "gpu"):
        for name, anchor_field in (
            ("kube-system", plane + "_cluster_uid"),
            (context.namespace, plane + "_namespace_uid"),
        ):
            anchors[anchor_field] = namespace_uid(
                private_json(
                    runner,
                    [*context.kubectl(plane), "get", "namespace", name, "-o", "json"],
                ),
                name,
            )
    return anchors


def current_binding(
    runner: CommandRunner,
    context: AdminNodeKeyContext,
    registration: AdminCustodyRegistration,
) -> CustodyBinding:
    runner = CustodyReadRunner(runner)
    before = read_regular(context.release_manifest)
    canonical_manifest = context.repository_root / "dist/current-release.json"
    if before != read_regular(canonical_manifest):
        raise CustodyError(
            "administrator custody release is not the verified current candidate"
        )
    verify_prebuilt_release(
        runner,
        repository_root=context.repository_root,
        state_dir=context.state_dir,
        staging_only=registration.selection.allow_staging,
    )
    if before != read_regular(context.release_manifest) or before != read_regular(
        canonical_manifest
    ):
        raise CustodyError("administrator custody release changed during verification")
    crypto = custody_verifier(runner, registration)
    release_key = read_regular(context.state_dir / "release-signing/cosign.pub")
    if crypto.public_identity(release_key) in crypto.public_keys:
        raise CustodyError(
            "administrator custody authorities must be independent of the release signer"
        )
    try:
        manifest = json.loads(before)
        if (
            type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 4
        ):
            raise ValueError
        node = manifest["components"]["node_runtime"]
        release = ReleaseBinding(
            release_id=manifest["release_id"],
            manifest_sha256=hashlib.sha256(before).hexdigest(),
            delivery_sha256=manifest["delivery"]["sha256"],
            node_wheel_sha256=node["wheel_sha256"],
            node_digest=node["module_digest"],
            bundle_sha256=manifest["bundle_sha256"],
            template_sha256=manifest["delivery"]["node_template_inputs"]["sha256"],
            config_digest=context.agent_config_digest,
            runtime_profile_version=context.runtime_profile_version,
        )
    except (KeyError, TypeError, ValueError):
        raise CustodyError(
            "administrator custody release binding is incomplete"
        ) from None
    return CustodyBinding(
        release=release,
        site=SiteBinding(
            site_name=context.site_id,
            region=context.cluster.region,
            cpu_eks_arn=context.cpu_eks_arn,
            gpu_eks_arn=context.cluster.eks_arn,
            cluster_id=context.cluster_id,
            hyperpod_cluster=context.cluster.hyperpod_name,
            namespace=context.namespace,
            **cluster_namespace_anchors(runner, context),
        ),
        nodes=live_node_uids(runner, context),
        master=master_source(runner, context),
    )


def verify_completed_custody(
    runner: CommandRunner,
    context: AdminNodeKeyContext,
    crypto: CustodyCrypto,
    binding: CustodyBinding,
    chain_file: Path,
) -> Transaction:
    runner = CustodyReadRunner(runner)
    chain = parse(Chain, read_regular(chain_file))
    head = verify_chain(chain, crypto, binding=binding)
    for plane in ("cpu", "gpu"):
        expected = getattr(head.completed.statement, plane)
        proof = read_node_key_proof(runner, context.kubectl(plane), context.namespace)
        if (
            proof is None
            or proof.rotation_pending
            or proof.uid != expected.uid
            or plane == "gpu"
            and set(proof.digests) != set(binding.nodes)
            or any(
                proof.digests.get(node) != state.sha256
                for node, state in expected.keys.items()
            )
        ):
            raise CustodyError(
                "administrator custody receipt does not match deployed key sources"
            )
    return head
