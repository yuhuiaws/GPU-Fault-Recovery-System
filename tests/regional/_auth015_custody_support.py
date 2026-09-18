"""Provisioning plus real in-process Node Agent applications, with fake I/O."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from gpu_fault.admin.node_key_custody_crypto import CustodyCrypto, parse
from gpu_fault.admin.node_key_custody_models import Chain, canonical
from scripts.e2e.regional import auth015_custody_inputs as inputs_module
from scripts.e2e.regional import identity_acceptance_auth as auth
from tests.deploy._node_key_custody_support import ProvisionFixture
from tests.regional._cov95_auth015_live import LiveSite


class RuntimeFixture:
    def __init__(self, root, monkeypatch):
        self.root = root
        self.provisioning = ProvisionFixture(root)
        self.site = LiveSite(root, monkeypatch)
        fixture = self.provisioning
        fixture.binding = fixture.binding.model_copy(
            update={
                "release": fixture.binding.release.model_copy(
                    update={
                        "manifest_sha256": hashlib.sha256(
                            fixture.manifest.read_bytes()
                        ).hexdigest(),
                        "delivery_sha256": self.site.files.delivery["sha256"],
                    }
                )
            }
        )
        self.site.region = "test-1"
        self.site.config = {
            "site_name": fixture.binding.site.site_name,
            "cpu_eks_arn": fixture.binding.site.cpu_eks_arn,
        }
        cpu = self.site.cpu

        def read_cpu(*arguments, **kwargs):
            if arguments[:2] == ("get", "namespace"):
                return fixture.run(
                    ["kubectl", "--context", "cpu-context", *arguments]
                ).stdout
            self.site.current_key_document = copy.deepcopy(
                fixture.api.state["secrets"]["cpu"]
            )
            return cpu(*arguments, **kwargs)

        def read_gpu(target, *arguments, **kwargs):
            assert target == self.site.target
            self.site.events.append(("gpu", *arguments))
            if arguments[:2] == ("get", "namespace"):
                return fixture.run(
                    ["kubectl", "--context", "gpu-context", *arguments]
                ).stdout
            if arguments[:2] == ("get", "secret"):
                return json.dumps(fixture.api.state["secrets"]["gpu"])
            assert arguments[:2] == ("get", "nodes"), (
                "custody cannot mutate a GPU object"
            )
            return json.dumps(self.site.raw["nodes"])

        self.site.cpu = read_cpu
        self.site.gpu = read_gpu
        self.site.regional = lambda target: SimpleNamespace(
            kubectl=lambda plane, *args, **kwargs: (
                read_cpu(*args, **kwargs)
                if plane == "cpu"
                else read_gpu(target, *args, **kwargs)
            )
        )
        monkeypatch.setattr(
            inputs_module,
            "CustodyCrypto",
            lambda path, pin: CustodyCrypto(
                path, pin, runner=fixture.authorities.runner
            ),
        )
        self.chain = fixture.provision(fixture.session())
        self.retired_path = None
        self.capture_index = 0

    def install_runtime_keys(self):
        from tests.regional._security_runtime_activation import activate_runtime

        activate_runtime(self)

    def inputs(self):
        chain_path = self.root / "custody-chain.json"
        chain_path.write_bytes(canonical(self.chain))
        descriptor = self.root / "custody-proof.json"
        descriptor.write_text(
            json.dumps(
                {
                    "trust": str(self.provisioning.authorities.trust_path),
                    "chain": str(chain_path),
                    "retired_key_file": str(self.retired_path)
                    if self.retired_path
                    else None,
                }
            )
        )
        return inputs_module.Auth015CustodyInputs(
            descriptor, self.provisioning.authorities.trust_pin
        )

    def witness(self):
        self.capture_index += 1
        directory = self.root / f"witness-{self.capture_index}"
        directory.mkdir(mode=0o700)
        result = auth.run_auth015(
            self.site,
            self.site.target,
            nodes=("node-a", "node-b"),
            release_inputs=self.site.files.inputs(),
            custody_inputs=self.inputs(),
            case_dir=directory,
            focused_tests={"passed": True},
        )
        self.chain = parse(Chain, Path(result["custody_chain_file"]).read_bytes())
        return result

    def rotate(self):
        session = self.provisioning.session(self.chain)
        self.chain = self.provisioning.provision(session)
        self.retired_path = Path(session.inputs.retired_key_file)

    def close(self):
        self.site.close()
