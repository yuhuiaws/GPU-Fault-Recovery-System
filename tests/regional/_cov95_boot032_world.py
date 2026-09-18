from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from gpu_fault.admin import aws_cleanup_ownership, deploy_host_binding
from gpu_fault.admin.aws_cleanup import ResourceProbe
from gpu_fault.admin.bootstrap_common import CommandRunner
from gpu_fault.installation_inventory import installed_units_digest
from scripts.e2e.regional import boot032_contract as contract
from scripts.e2e.regional import boot032_lifecycle as lifecycle
from scripts.e2e.regional import boot032_native as adapter
from scripts.e2e.regional import boot032_observe as observe
from scripts.e2e.regional import live_driver_guard as guard
from tests.regional._cov95_boot032_support import (
    ACCOUNT,
    CA,
    FIXTURE,
    arn,
    make_site,
    resource_snapshot,
)
from tests.regional._cov95_common_live import LiveModel


class World:
    def __init__(self, root, monkeypatch):
        self.root = root
        self.target = make_site(
            root / "cases" / contract.CASE_ID / "sacrificial", target=True
        )
        self.protected = make_site(root / "accepted", target=False)
        self.settings = contract.Settings(
            root,
            FIXTURE,
            self.target,
            self.protected,
            protected_cluster_id=contract.cluster_specs(self.protected)[1][
                "cluster_id"
            ],
        )
        self.calls = []
        self.read_error = False
        self.process = {"pid": 1, "boot_id": "fixture-boot", "start_ticks": "10"}
        self.snapshots = {
            site.metadata_name: resource_snapshot(site)
            for site in (self.target, self.protected)
        }
        self.cloud = {}
        self.uids = {}
        self.inventory = {}
        self.runtime = {}
        self.agents = {}
        for site in (self.target, self.protected):
            self.add_site(site)
        self.harness = None
        monkeypatch.setattr(
            deploy_host_binding,
            "bound_deploy_host_state_dir",
            lambda: self.target.source.parent,
        )
        monkeypatch.setattr(observe, "aws", self.aws)
        monkeypatch.setattr(observe, "fetch_installation_resource_registry", self.fetch)
        monkeypatch.setattr(observe, "ResourceProbe", self.probe)
        monkeypatch.setattr(adapter, "ResourceProbe", self.probe)
        monkeypatch.setattr(
            aws_cleanup_ownership,
            "json_command",
            lambda arguments: {"Account": ACCOUNT}
            if arguments[1:3] == ["sts", "get-caller-identity"]
            else None,
        )
        monkeypatch.setattr(observe, "RegionalLiveFixture", self.regional)

        def run(_runner, arguments, **kwargs):
            return self.command(arguments, **kwargs)

        monkeypatch.setattr(CommandRunner, "run", run)
        monkeypatch.setattr(lifecycle, "predecessor_path", lambda *_a: (None, None))
        monkeypatch.setattr(lifecycle, "process_identity", lambda: dict(self.process))
        monkeypatch.setattr(guard, "source_digest", lambda: "fixture-source")
        monkeypatch.setattr(lifecycle, "source_digest", lambda: "fixture-source")

    def add_site(self, site):
        runtime_root = self.root / ("runtime-" + site.metadata_name)
        runtime_root.mkdir(exist_ok=True)
        runtime = LiveModel(runtime_root).runtime_identity()
        runtime["release_state"]["release_id"] = "release-" + site.metadata_name
        self.runtime[site.metadata_name] = runtime
        specs = contract.cluster_specs(site)
        for spec in specs:
            name = spec["eks_name"]
            self.cloud[name] = {
                "eks": {
                    "cluster": {
                        "arn": spec["eks_arn"],
                        "name": name,
                        "createdAt": "2026-09-12T00:00:00Z",
                        "status": "ACTIVE",
                        "endpoint": f"https://{name}.example.invalid",
                        "certificateAuthority": {"data": CA},
                        "tags": {
                            contract.CASE_TAG: contract.CASE_ID,
                            contract.FIXTURE_TAG: FIXTURE,
                        },
                    }
                },
                "hp": {
                    "ClusterName": name,
                    "ClusterArn": arn(
                        "sagemaker", name + "-incarnation", region=spec["region"]
                    ),
                    "ClusterStatus": "InService",
                    "NodeRecovery": "None",
                    "Orchestrator": {"Eks": {"ClusterArn": spec["eks_arn"]}},
                },
                "tags": {
                    "Tags": [
                        {"Key": contract.CASE_TAG, "Value": contract.CASE_ID},
                        {"Key": contract.FIXTURE_TAG, "Value": FIXTURE},
                    ]
                },
            }
            for kind, resource, namespace in (
                ("namespace", "kube-system", ""),
                ("namespace", "gpu-fault-system", ""),
                ("deployment", "gpu-fault-api", "gpu-fault-system"),
                ("node", "node-a", ""),
            ):
                key = (site.metadata_name, spec["context"], kind, resource, namespace)
                self.uids[key] = (
                    f"{site.metadata_name}-{spec['context']}-{resource}-uid"
                )
            if spec["plane"] == "gpu":
                units = ["gpu-fault-agent.service", "gpu-fault-collector.service"]
                self.agents[(site.metadata_name, spec["context"])] = {
                    "cluster_id": spec["cluster_id"],
                    "node_id": "node-a",
                    "node_instance_id": "i-"
                    + hashlib.sha256(
                        f"{site.metadata_name}:{spec['context']}".encode()
                    ).hexdigest()[:17],
                    "lifecycle_state": "ACTIVE",
                    "installed_unit_inventory": {
                        "units": units,
                        "digest": installed_units_digest(units),
                    },
                }
        resources = [
            {"scope": "namespaced", "kind": "deployment", "name": "gpu-fault-api"}
        ]
        self.inventory[site.metadata_name] = {
            "schema_version": 1,
            "cpu": {"resources": copy.deepcopy(resources)},
            "gpu": {
                "resources": copy.deepcopy(resources),
                "by_context": {
                    spec["context"]: {"resources": copy.deepcopy(resources)}
                    for spec in specs[1:]
                },
            },
            "unregistered_resources": [],
        }

    def fetch(self, site, **_kwargs):
        if self.read_error:
            raise RuntimeError("fixture resource read failed")
        return self.snapshots[site.metadata_name]

    def probe(self, site):
        def exists(resource):
            self.calls.append(("resource", site.metadata_name, resource.resource_key))
            if self.read_error:
                raise RuntimeError("fixture resource read failed")
            if (
                site.metadata_name == self.target.metadata_name
                and self.harness is not None
            ):
                return resource.resource_key in self.harness.existing
            return True

        return SimpleNamespace(
            validate_supported=ResourceProbe(site).validate_supported, exists=exists
        )

    def regional(self, settings):
        site = (
            self.target
            if settings.cpu_kubeconfig == contract.kubeconfig(self.target, "cpu")
            else self.protected
        )
        return SimpleNamespace(
            settings=settings,
            runtime_identity=lambda: copy.deepcopy(self.runtime[site.metadata_name]),
            gpu_nodes=lambda: [
                {
                    "name": "node-a",
                    "uid": self.uids[
                        (site.metadata_name, settings.gpu_context, "node", "node-a", "")
                    ],
                    "ready": "True",
                    "unschedulable": False,
                }
            ],
            store_snapshot=lambda **_kwargs: {
                "agent": copy.deepcopy(
                    self.agents[(site.metadata_name, settings.gpu_context)]
                )
            },
        )

    def aws(self, region, service, operation, *arguments, absent=()):
        self.calls.append(("aws", service, operation))
        if self.read_error:
            raise RuntimeError("fixture cloud read failed")
        if operation == "list-tags":
            name = next(
                name
                for name, values in self.cloud.items()
                if values["hp"] and values["hp"]["ClusterArn"] == arguments[1]
            )
            return copy.deepcopy(self.cloud[name]["tags"])
        return copy.deepcopy(
            self.cloud[arguments[1]]["eks" if service == "eks" else "hp"]
        )

    def command(self, arguments, **kwargs):
        arguments = list(arguments)
        self.calls.append(("command", arguments, kwargs))
        if self.read_error:
            raise RuntimeError("fixture command read failed")
        if arguments[0] == "kubectl":
            path = Path(arguments[arguments.index("--kubeconfig") + 1])
            site = next(
                site
                for site in (self.target, self.protected)
                if path
                in {contract.kubeconfig(site, plane) for plane in ("cpu", "gpu")}
            )
            context = (
                arguments[arguments.index("--context") + 1]
                if "--context" in arguments
                else "cpu"
            )
            namespace = (
                arguments[arguments.index("-n") + 1] if "-n" in arguments else ""
            )
            index = arguments.index("get")
            key = (
                site.metadata_name,
                context,
                arguments[index + 1],
                arguments[index + 2],
                namespace,
            )
            uid = self.uids.get(key)
            return "" if uid is None else "uid=" + uid
        if len(arguments) > 1 and arguments[1].endswith(
            "/collect_installed_resource_registry.py"
        ):
            config = json.loads(
                Path(arguments[arguments.index("--config") + 1]).read_text()
            )
            return json.dumps(self.inventory[config["site_name"]])
        if self.harness is not None:
            return self.harness.run(arguments, **kwargs)
        raise AssertionError("unexpected fake command transport operation")
