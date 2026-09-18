"""Owned files and fake cloud I/O around the real administrator custody path."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from gpu_fault.admin import cli
from gpu_fault.admin import node_key_custody_admin_config as config
from gpu_fault.admin.bootstrap import REGIONAL_PROFILE_SOURCE
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
)
from gpu_fault.admin.node_key_custody import ProvisionCustody, ProvisionInputs
from gpu_fault.admin.node_key_custody_admin import CustodyPreparation
from gpu_fault.admin.node_key_custody_admin_probe import AdminNodeKeyContext
from gpu_fault.admin.node_key_custody_crypto import CustodyCrypto, parse
from gpu_fault.admin.node_key_custody_models import Authorization, canonical
from tests.deploy._node_action_key_api import NAMESPACE, Api, node_list
from tests.deploy._node_key_custody_support import MASTER, PROVISION, ROOT, Authorities


class AdminWorld(CommandRunner):
    def __init__(self, root: Path, monkeypatch) -> None:
        super().__init__()
        self.root = root
        self.state_dir = root / "site"
        self.state_dir.mkdir(mode=0o700)
        self.repo = root / "repository"
        self.repo.mkdir()
        for relative in (
            REGIONAL_PROFILE_SOURCE,
            "deploy/node/provision-node-action-keys.sh",
            "deploy/node/provision_node_action_keys.py",
            "scripts/e2e/regional/identity_acceptance_auth.py",
            "scripts/e2e/regional/identity_acceptance_common.py",
            "scripts/e2e/regional/run_identity_acceptance.py",
        ):
            target = self.repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        for source in (ROOT / "scripts/e2e/regional").glob("auth015_*.py"):
            shutil.copyfile(source, self.repo / "scripts/e2e/regional" / source.name)
        shutil.copytree(ROOT / "src/gpu_fault/admin", self.repo / "src/gpu_fault/admin")
        security = root / "authorities"
        security.mkdir(mode=0o700)
        self.authorities = Authorities(security)
        monkeypatch.setattr(
            config,
            "CustodyCrypto",
            lambda path, pin: CustodyCrypto(path, pin, runner=self.authorities.runner),
        )
        self.cpu = ClusterIdentity(
            input_arn="arn:aws:eks:test-1:111122223333:cluster/cpu",
            role="cpu",
            region="test-1",
            account_id="111122223333",
            hyperpod_arn="arn:aws:sagemaker:test-1:111122223333:cluster/cpu",
            hyperpod_name="cpu",
            eks_arn="arn:aws:eks:test-1:111122223333:cluster/cpu",
            eks_name="cpu",
            vpc_id="vpc-cpu",
            subnet_ids=("subnet-cpu",),
            node_recovery="None",
            context="cpu-context",
        )
        self.gpu = replace(
            self.cpu,
            input_arn="arn:aws:eks:test-1:111122223333:cluster/gpu",
            eks_arn="arn:aws:eks:test-1:111122223333:cluster/gpu",
            eks_name="gpu",
            hyperpod_arn="arn:aws:sagemaker:test-1:111122223333:cluster/hyperpod-a",
            hyperpod_name="hyperpod-a",
            role="gpu",
            context="gpu-context",
        )
        self.cluster_id = "hyperpod-a"
        self.site_id = "custody-site"
        self.cpu_kubeconfig = self.state_dir / "cpu.kubeconfig"
        self.gpu_kubeconfig = self.state_dir / "gpu.kubeconfig"
        self.master_file = self.state_dir / "fleet-master"
        self.namespaces = {}
        for plane in ("cpu", "gpu"):
            self.namespaces[plane, "kube-system"] = str(uuid.uuid4())
        self.master_uid = None
        nodes = node_list("node-a", "node-b")
        for item in nodes["items"]:
            item["metadata"]["uid"] = str(uuid.uuid4())
            item["metadata"]["labels"]["sagemaker.amazonaws.com/cluster-name"] = (
                "hyperpod-a"
            )
        self.api = Api({"nodes": nodes, "secrets": {"cpu": None, "gpu": None}})
        self.calls = []
        self.sign_calls = 0
        self.helper_calls = 0
        self.release_ready = False
        self.selection_file = root / "selection.json"
        self.requests = {self.gpu.eks_arn: None}
        dist = self.repo / "dist"
        dist.mkdir()
        self.manifest_path = dist / "current-release.json"
        self.manifest = {
            "schema_version": 4,
            "release_id": "custody-release",
            "deployable": True,
            "bundle_sha256": "c" * 64,
            "components": {
                "node_runtime": {"wheel_sha256": "a" * 64, "module_digest": "b" * 64}
            },
            "delivery": {
                "node_template_inputs": {"sha256": "d" * 64},
                "sha256": "f" * 64,
            },
        }
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.release_key = ec.generate_private_key(ec.SECP256R1())
        signing = self.state_dir / "release-signing"
        signing.mkdir()
        (signing / "cosign.pub").write_bytes(
            self.release_key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        self.attestation = dist / "current-attestation.json"
        self.attestation.write_text(
            json.dumps(
                {
                    "manifest_sha256": hashlib.sha256(
                        self.manifest_path.read_bytes()
                    ).hexdigest()
                }
            )
        )
        (dist / "current-attestation.bundle.json").write_bytes(
            self.release_key.sign(
                self.attestation.read_bytes(), ec.ECDSA(hashes.SHA256())
            )
        )
        self.release = {
            "manifest": str(self.manifest_path),
            "release_id": self.manifest["release_id"],
            "agent_config_digest": "e" * 64,
            "images": {
                name: "registry.invalid/" + name + "@sha256:" + "f" * 64
                for name in (
                    "runtime",
                    "executor",
                    "node_installer",
                    "dcgm_exporter",
                    "adot",
                    "node_dependencies",
                )
            },
        }

    def access(self) -> None:
        for plane, path in (("cpu", self.cpu_kubeconfig), ("gpu", self.gpu_kubeconfig)):
            path.write_text("owned fake kubeconfig")
            self.namespaces.setdefault((plane, NAMESPACE), str(uuid.uuid4()))
        self.master_uid = self.master_uid or str(uuid.uuid4())
        self.master_file.write_bytes(MASTER)
        self.master_file.chmod(0o600)

    def context(self) -> AdminNodeKeyContext:
        return AdminNodeKeyContext(
            state_dir=self.state_dir,
            repository_root=self.repo,
            site_id=self.site_id,
            cpu_eks_arn=self.cpu.eks_arn,
            cpu_kubeconfig=self.cpu_kubeconfig,
            gpu_kubeconfig=self.gpu_kubeconfig,
            namespace=NAMESPACE,
            cluster=self.gpu,
            cluster_id=self.cluster_id,
            release_manifest=self.manifest_path,
            runtime_profile_version="hyperpod-v1",
            agent_config_digest="e" * 64,
        )

    def configure(self) -> None:
        self.selection_file.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "trust": str(self.authorities.trust_path),
                    "clusters": self.requests,
                }
            )
        )
        self.selection_file.chmod(0o600)
        args = cli.parser().parse_args(
            [
                "node-key-custody",
                "configure",
                "--state-dir",
                str(self.state_dir),
                "--file",
                str(self.selection_file),
                "--trust-sha256",
                self.authorities.trust_pin,
            ]
        )
        assert cli.run(args) == 0

    def authorize(self) -> Path:
        files = list((self.state_dir / "node-key-custody/preparations").glob("*.json"))
        assert files, "custody authorization requires a prepared identity binding"
        preparation = parse(CustodyPreparation, files[-1].read_bytes())
        now = datetime.now(timezone.utc)
        authorization = Authorization(
            transaction_id=uuid.uuid4().hex + uuid.uuid4().hex,
            binding=preparation.binding,
            purpose="install",
            rotate_node=None,
            previous_receipt_sha256=None,
            producer_sha256=preparation.producer_sha256,
            witness_sha256=preparation.witness_sha256,
            not_before=now - timedelta(seconds=1),
            expires_at=now + timedelta(minutes=15),
        )
        authorization_path = self.root / "approved.json"
        authorization_path.write_bytes(
            canonical(self.authorities.envelope(authorization, "approval"))
        )
        receipts = self.root / "receipts"
        receipts.mkdir(mode=0o700)
        request = ProvisionInputs(
            trust=str(self.authorities.trust_path),
            authorization=str(authorization_path),
            release_manifest=str(self.manifest_path),
            previous_chain=None,
            state_directory=str(receipts),
            retired_key_file=None,
        )
        request_path = self.root / "request.json"
        request_path.write_bytes(canonical(request))
        self.requests[self.gpu.eks_arn] = str(request_path)
        self.configure()
        return receipts / (authorization.transaction_id + ".chain.json")

    def private_run(self, arguments, **kwargs):
        args = list(arguments)
        plane = "gpu" if "gpu-context" in args else "cpu"
        if "--kubeconfig" in args and args[args.index("--kubeconfig") + 1] == str(
            self.gpu_kubeconfig
        ):
            plane = "gpu"
            index = args.index("--kubeconfig")
            del args[index : index + 2]
        if "configmap" in args and hasattr(self, "activation_io"):
            return subprocess.CompletedProcess(
                args, 0, json.dumps(self.activation_io.wave), ""
            )
        if "namespace" in args:
            name = args[args.index("namespace") + 1]
            uid = self.namespaces.get((plane, name))
            if uid is None:
                return subprocess.CompletedProcess(
                    args, 1, "", "missing owned namespace"
                )
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(
                    {
                        "apiVersion": "v1",
                        "kind": "Namespace",
                        "metadata": {"name": name, "uid": uid},
                    }
                ),
                "",
            )
        if "gpu-fault-node-installer" in args:
            assert plane == "cpu"
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(
                    {
                        "apiVersion": "v1",
                        "kind": "Secret",
                        "type": "Opaque",
                        "metadata": {
                            "name": "gpu-fault-node-installer",
                            "namespace": NAMESPACE,
                            "uid": self.master_uid,
                            "resourceVersion": "1",
                        },
                        "data": {
                            "node-action-secret": base64.b64encode(MASTER).decode()
                        },
                    }
                ),
                "",
            )
        return self.api.run(args, input_text=kwargs.get("input_text"))

    def run(self, arguments, **kwargs):
        args = list(arguments)
        self.calls.append(args)
        if args[:2] in (["openssl", "pkey"], ["openssl", "dgst"]):
            result = self.authorities.runner(
                args,
                capture=True,
                timeout_seconds=kwargs["timeout_seconds"],
                environment=kwargs.get("env", {}),
            )
        elif args[0] == "kubectl":
            result = self.private_run(args, **kwargs)
        elif any(str(item).endswith("verify-release-attestation.py") for item in args):
            assert self.release_ready, "custody used an unfinished release"
            signature = (
                self.repo / "dist/current-attestation.bundle.json"
            ).read_bytes()
            self.release_key.public_key().verify(
                signature, self.attestation.read_bytes(), ec.ECDSA(hashes.SHA256())
            )
            assert (
                json.loads(self.attestation.read_text())["manifest_sha256"]
                == hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
            )
            return ""
        elif args[0].endswith("provision-node-action-keys.sh"):
            assert kwargs["mutate"] is True and kwargs["sensitive"] is True
            self.helper_calls += 1
            custody = None
            if "--custody-request" in args:
                request = Path(args[args.index("--custody-request") + 1])
                pin = args[args.index("--custody-trust-sha256") + 1]
                custody = ProvisionCustody.load(
                    request,
                    pin,
                    crypto_factory=lambda path, digest: CustodyCrypto(
                        path, digest, runner=self.authorities.runner
                    ),
                    expected_input_sha256=args[
                        args.index("--custody-input-sha256") + 1
                    ],
                    **(
                        {
                            "activation_state": Path(
                                args[args.index("--custody-activation-state") + 1]
                            ),
                            "activation_state_sha256": args[
                                args.index("--custody-activation-sha256") + 1
                            ],
                        }
                        if "--custody-activation-state" in args
                        else {}
                    ),
                )
            PROVISION.provision(
                PROVISION.Scope(("kubectl", "--context", "gpu-context"), NAMESPACE),
                PROVISION.Scope(("kubectl", "--context", "cpu-context"), NAMESPACE),
                cluster_id=self.cluster_id,
                hyperpod_cluster=self.gpu.hyperpod_name,
                master_file=self.master_file,
                rotate_node=kwargs["env"].get("GPU_FAULT_ROTATE_NODE_ACTION_KEY", ""),
                runner=self.private_run,
                custody=custody,
                expected_nodes=json.loads(
                    kwargs["env"].get("GPU_FAULT_NODE_KEY_EXPECTED_NODES_JSON", "null")
                ),
            )
            return "provisioned 2 node action keys"
        else:
            raise AssertionError("unexpected fake administrator I/O")
        if result.returncode:
            raise BootstrapError("owned fake command failed")
        return result.stdout

    def site(self, *, managed=True):
        return SimpleNamespace(
            source=self.state_dir / "site.yaml",
            repository_root=self.repo,
            release_config={
                "site_name": self.site_id,
                "cpu_eks_arn": self.cpu.eks_arn,
                "cpu_kubeconfig": str(self.cpu_kubeconfig),
                "gpu_kubeconfig": str(self.gpu_kubeconfig),
                "namespace": NAMESPACE,
                "release": {
                    "manifest": str(self.manifest_path),
                    "agent_config_digest": "e" * 64,
                },
                "runtime_profile": {"version": "hyperpod-v1"},
                "health": {},
                "clusters": [
                    {
                        "cluster_id": self.cluster_id,
                        "eks_cluster_arn": self.gpu.eks_arn,
                        "hyperpod_cluster_name": self.gpu.hyperpod_name,
                        "context": self.gpu.context,
                    }
                ]
                if managed
                else [],
            },
        )
