from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

from scripts.e2e.regional import boot_acceptance_runtime as runtime
from tests.regional.test_boot_acceptance_behavior import RuntimeFixture


class GreenfieldFixture(RuntimeFixture):
    def __init__(self, root, monkeypatch):
        super().__init__()
        self.production_site = root / "unit-production-site"
        self.production_site.write_text("unit site", encoding="utf-8")
        self.production_config = {"clusters": [{"cluster_id": "production-a"}]}
        self.production_owners = ["production-a/production-pod"]
        self.production_reads = []
        self.pod_names = ["replica-a", "replica-b"]
        self.deployment = runtime.manifest_deployment()
        self.deployment["spec"]["replicas"] = len(self.pod_names)
        self.pod_state = {
            "items": [
                {
                    "metadata": {
                        "name": name,
                        "creationTimestamp": "2026-09-12T00:00:00Z",
                    },
                    "status": {
                        "phase": "Running",
                        "conditions": [
                            {"type": "Ready", "status": "False"},
                            {
                                "type": "Ready",
                                "status": "True",
                                "lastTransitionTime": "2026-09-12T00:00:15Z",
                            },
                        ],
                    },
                }
                for name in self.pod_names
            ]
        }
        self.secret = {
            "data": {
                key: "REPLACE_WITH_UNIT_VALUE"
                for _name, key, _env in runtime.secret_key_contract()["required"]
            }
        }
        self.role = "arn:aws:iam::000000000000:role/unit-isolated-executor"
        self.target = {"executor_irsa_role_arn": self.role}
        self.service_account = {
            "metadata": {"annotations": {"eks.amazonaws.com/role-arn": self.role}}
        }
        self.trust = {
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Federated": "unit-oidc-provider"},
                    "Condition": {
                        "StringEquals": {
                            "unit:sub": "system:serviceaccount:unit:unit-executor"
                        }
                    },
                }
            ]
        }
        self.logs = "healthy isolated replica"
        self.commands = []
        monkeypatch.setattr(runtime, "load_site", self.load_production)
        monkeypatch.setattr(runtime, "SiteFixture", self.production_fixture)
        monkeypatch.setattr(runtime, "run", self.run)

    def load_production(self, path, **kwargs):
        self.production_reads.append(("site", path))
        return SimpleNamespace(release_config=self.production_config)

    def production_fixture(self, path, cluster_id):
        self.production_reads.append(("fixture", path, cluster_id))
        return SimpleNamespace(
            regional=SimpleNamespace(cpu_python=self.read_production_owners)
        )

    def read_production_owners(self, script):
        self.production_reads.append(("owners",))
        return {"owners": list(self.production_owners)}

    def pods(self, plane, app):
        self.calls.append(("pods", plane, app))
        return list(self.pod_names)

    def kubectl(self, plane, *args, **kwargs):
        self.calls.append((plane, args, kwargs))
        if args[0] == "logs":
            if self.log_error:
                raise runtime.BootAcceptanceError("unit log transport unavailable")
            return self.logs
        resources = {
            "deployment": self.deployment,
            "secret": self.secret,
            "pod": self.pod_state,
            "serviceaccount": self.service_account,
        }
        assert args[0] == "get" and args[1] in resources, (
            "greenfield probe requested an unconfigured resource"
        )
        return json.dumps(resources[args[1]])

    def run(self, args, **kwargs):
        self.commands.append((list(args), kwargs))
        assert args[:3] == ["aws", "iam", "get-role"], (
            "greenfield audit requested a non-read-only command"
        )
        return subprocess.CompletedProcess(
            args, 0, json.dumps({"Role": {"AssumeRolePolicyDocument": self.trust}}), ""
        )
