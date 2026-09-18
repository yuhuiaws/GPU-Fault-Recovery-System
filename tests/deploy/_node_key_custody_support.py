"""Synthetic independent signers and a private, stateful provisioning API."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypeVar

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils

from gpu_fault.admin.node_key_custody import (
    ProvisionCustody,
    ProvisionInputs,
    producer_identity,
)
from gpu_fault.admin.node_key_custody_crypto import CustodyCrypto
from gpu_fault.admin.node_key_custody_models import (
    Activated,
    Authorization,
    Chain,
    CustodyBinding,
    MasterSource,
    PublicAuthority,
    ReleaseBinding,
    RuntimeNode,
    Signed,
    SiteBinding,
    Statement,
    Trust,
    canonical,
    statement_sha256,
)
from scripts.e2e.regional.auth015_custody import witness_identity
from tests._script_loader import lazy_script_module
from tests.deploy._node_action_key_api import NAMESPACE, Api, node_list

ROOT = Path(__file__).resolve().parents[2]
PROVISION_PATH = ROOT / "deploy/node/provision_node_action_keys.py"
PROVISION = lazy_script_module(PROVISION_PATH)
MASTER = b"synthetic-custody-master-" + b"m" * 48
_T = TypeVar("_T", bound=Statement)


class Authorities:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[list[str]] = []
        self.keys = {
            role: ec.generate_private_key(ec.SECP256R1())
            for role in ("approval", "provisioner", "witness")
        }
        records = {}
        for index, (role, key) in enumerate(self.keys.items(), 1):
            public = key.public_key()
            name = role + ".pem"
            (root / name).write_bytes(
                public.public_bytes(
                    serialization.Encoding.PEM,
                    serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
            records[role] = PublicAuthority(
                public_key=name,
                public_key_sha256=hashlib.sha256(
                    public.public_bytes(
                        serialization.Encoding.DER,
                        serialization.PublicFormat.SubjectPublicKeyInfo,
                    )
                ).hexdigest(),
                kms_key_arn=(
                    "arn:aws:kms:test-1:111122223333:key/"
                    f"00000000-0000-0000-0000-{index:012d}"
                ),
            )
        self.trust = Trust(**records)
        self.trust_path = root / "trust.json"
        self.trust_path.write_bytes(canonical(self.trust))
        self.trust_pin = hashlib.sha256(self.trust_path.read_bytes()).hexdigest()

    def envelope(self, statement: _T, role: str) -> Signed[_T]:
        signature = self.keys[role].sign(
            canonical(statement), ec.ECDSA(hashes.SHA256())
        )
        return Signed[type(statement)](
            statement=statement,
            signer_sha256=getattr(self.trust, role).public_key_sha256,
            signature=base64.b64encode(signature).decode(),
        )

    def runner(
        self, arguments: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(arguments))
        assert kwargs["capture"] is True and kwargs["timeout_seconds"] == 20
        assert not any(
            name in kwargs["environment"]
            for name in (
                "GPU_FAULT_EXECUTION_TOKEN",
                "GPU_FAULT_NODE_ACTION_SECRET",
                "AWS_SECRET_ACCESS_KEY",
            )
        ), "signer must not inherit raw credentials"
        code, output = 0, ""
        if arguments[:2] == ["openssl", "pkey"]:
            public = serialization.load_pem_public_key(
                Path(arguments[arguments.index("-in") + 1]).read_bytes()
            )
            Path(arguments[arguments.index("-out") + 1]).write_bytes(
                public.public_bytes(
                    serialization.Encoding.DER,
                    serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
        elif arguments[:2] == ["openssl", "dgst"]:
            public = serialization.load_pem_public_key(
                Path(arguments[arguments.index("-verify") + 1]).read_bytes()
            )
            signature = Path(arguments[arguments.index("-signature") + 1]).read_bytes()
            try:
                public.verify(
                    signature,
                    Path(arguments[-1]).read_bytes(),
                    ec.ECDSA(hashes.SHA256()),
                )
            except InvalidSignature:
                code = 1
        else:
            assert arguments[:3] == ["aws", "kms", "sign"], (
                "unexpected synthetic signer command"
            )
            arn = arguments[arguments.index("--key-id") + 1]
            role = next(
                role
                for role in self.keys
                if getattr(self.trust, role).kms_key_arn == arn
            )
            digest = Path(
                arguments[arguments.index("--message") + 1].removeprefix("fileb://")
            ).read_bytes()
            assert len(digest) == 32, "only a digest may reach the synthetic KMS API"
            signature = self.keys[role].sign(
                digest, ec.ECDSA(utils.Prehashed(hashes.SHA256()))
            )
            output = json.dumps(
                {
                    "KeyId": arn,
                    "SigningAlgorithm": "ECDSA_SHA_256",
                    "Signature": base64.b64encode(signature).decode(),
                }
            )
        return subprocess.CompletedProcess(arguments, code, output, "")

    def crypto(self) -> CustodyCrypto:
        return CustodyCrypto(self.trust_path, self.trust_pin, runner=self.runner)


class ProvisionFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.authorities = Authorities(root)
        self.crypto = self.authorities.crypto()
        self.master_file = root / "master"
        self.master_file.write_bytes(MASTER)
        self.master_file.chmod(0o600)
        self.manifest = root / "manifest.json"
        self.manifest.write_text('{"synthetic":"authorized-release"}')
        self.binding = CustodyBinding(
            release=ReleaseBinding(
                release_id="release-a",
                manifest_sha256=hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                delivery_sha256="1" * 64,
                node_wheel_sha256="a" * 64,
                node_digest="b" * 64,
                bundle_sha256="c" * 64,
                template_sha256="d" * 64,
                config_digest="e" * 64,
                runtime_profile_version="profile-a",
            ),
            site=SiteBinding(
                site_name="synthetic-site",
                region="test-1",
                cpu_eks_arn="arn:aws:eks:test-1:111122223333:cluster/cpu",
                gpu_eks_arn="arn:aws:eks:test-1:111122223333:cluster/test-a",
                cluster_id="cluster-a",
                hyperpod_cluster="hyperpod-a",
                namespace=NAMESPACE,
                cpu_cluster_uid="cpu-cluster-uid",
                gpu_cluster_uid="gpu-cluster-uid",
                cpu_namespace_uid="cpu-namespace-uid",
                gpu_namespace_uid="gpu-namespace-uid",
            ),
            nodes={"node-a": "uid-node-a", "node-b": "uid-node-b"},
            master=MasterSource(
                secret_name="gpu-fault-node-installer",
                secret_uid="cpu-master-uid",
                data_key="node-action-secret",
                sha256=hashlib.sha256(MASTER).hexdigest(),
            ),
        )
        self.api = Api(
            {
                "nodes": node_list("node-a", "node-b"),
                "secrets": {"gpu": None, "cpu": None},
            }
        )
        for item in self.api.state["nodes"]["items"]:
            item["metadata"]["uid"] = self.binding.nodes[item["metadata"]["name"]]
            item["metadata"]["labels"] = {
                "sagemaker.amazonaws.com/cluster-name": "hyperpod-a"
            }
        self.master_document = {
            "apiVersion": "v1",
            "kind": "Secret",
            "type": "Opaque",
            "metadata": {
                "name": self.binding.master.secret_name,
                "namespace": NAMESPACE,
                "uid": self.binding.master.secret_uid,
                "resourceVersion": "1",
            },
            "data": {self.binding.master.data_key: base64.b64encode(MASTER).decode()},
        }
        self.events: list[str] = []
        self.serial = 0

    def authorization(self, previous: Chain | None = None) -> Authorization:
        self.serial += 1
        now = datetime.now(timezone.utc)
        return Authorization(
            transaction_id=f"{self.serial:064x}",
            binding=self.binding,
            purpose="rotate" if previous else "install",
            rotate_node="node-a" if previous else None,
            previous_receipt_sha256=statement_sha256(previous.transactions[-1])
            if previous
            else None,
            producer_sha256=producer_identity(PROVISION_PATH),
            witness_sha256=witness_identity(),
            not_before=now - timedelta(minutes=1),
            expires_at=now + timedelta(minutes=10),
        )

    def session(
        self,
        previous: Chain | None = None,
        *,
        authorization: Authorization | None = None,
    ) -> ProvisionCustody:
        authorization = authorization or self.authorization(previous)
        directory = self.root / ("state-" + authorization.transaction_id)
        directory.mkdir(mode=0o700)
        retired_directory = self.root / "private-retired"
        retired_directory.mkdir(mode=0o700, exist_ok=True)
        inputs = ProvisionInputs(
            trust=str(self.authorities.trust_path),
            authorization=str(self.root / "authorization.json"),
            release_manifest=str(self.manifest),
            previous_chain=str(self.root / "previous.json") if previous else None,
            state_directory=str(directory),
            retired_key_file=str(retired_directory / authorization.transaction_id)
            if previous
            else None,
        )
        signed = self.authorities.envelope(authorization, "approval")
        Path(inputs.authorization).write_bytes(canonical(signed))
        if previous:
            Path(inputs.previous_chain).write_bytes(canonical(previous))
        return ProvisionCustody(inputs, self.crypto, signed, previous)

    def run(
        self, arguments: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        plane = "cpu" if "cpu-context" in arguments else "gpu"
        if "namespace" in arguments:
            name = arguments[arguments.index("namespace") + 1]
            uid = getattr(
                self.binding.site,
                plane + ("_cluster_uid" if name == "kube-system" else "_namespace_uid"),
            )
            return subprocess.CompletedProcess(
                arguments,
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
        if self.binding.master.secret_name in arguments:
            assert plane == "cpu", "the master source must never be queried on GPU"
            return subprocess.CompletedProcess(
                arguments, 0, json.dumps(self.master_document), ""
            )
        if any(verb in arguments for verb in ("create", "replace")):
            self.events.append("write:" + plane)
            assert list(self.root.glob("state-*/*.started.json")), (
                "key write preceded the signed intent"
            )
            assert MASTER.decode() not in kwargs.get("input_text", ""), (
                "master leaked into a key map"
            )
        return self.api.run(arguments, **kwargs)

    def provision(self, session: ProvisionCustody, *, runner: Any = None) -> Chain:
        count = PROVISION.provision(
            PROVISION.Scope(("kubectl", "--context", "gpu-context"), NAMESPACE),
            PROVISION.Scope(("kubectl", "--context", "cpu-context"), NAMESPACE),
            cluster_id="cluster-a",
            hyperpod_cluster="hyperpod-a",
            master_file=self.master_file,
            rotate_node=session.statement.rotate_node or "",
            runner=runner or self.run,
            custody=session,
        )
        assert count == 2
        return Chain.model_validate_json(
            Path(str(session.prefix) + ".chain.json").read_bytes()
        )

    def activate(self, chain: Chain) -> Chain:
        head = chain.transactions[-1]
        activated = Activated(
            completed_sha256=statement_sha256(head.completed.statement),
            binding_sha256=statement_sha256(self.binding),
            observed_at=datetime.now(timezone.utc),
            nodes={
                node: RuntimeNode(
                    node_uid=uid,
                    boot_id="boot-" + node,
                    agent_incarnation_id="incarnation-" + node,
                    agent_generation=7,
                    endpoint=f"https://10.0.1.{index}:9099",
                    certificate_sha256="7" * 64,
                )
                for index, (node, uid) in enumerate(self.binding.nodes.items(), 1)
            },
            runtime_identity_sha256="8" * 64,
            protocol_sha256="9" * 64,
            retired_key_denied=head.authorization.statement.purpose == "rotate",
        )
        return Chain(
            transactions=[
                *chain.transactions[:-1],
                head.model_copy(
                    update={"activated": self.crypto.sign(activated, "witness")}
                ),
            ]
        )
