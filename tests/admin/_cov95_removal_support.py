from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from pathlib import Path

from gpu_fault.admin import aws_commands, bootstrap_common, cluster_join_nodes
from gpu_fault.admin import cluster_join_evidence as evidence
from gpu_fault.admin import cluster_removal as removal
from gpu_fault.admin import cluster_removal_network as network
from gpu_fault.admin.bootstrap_common import CommandRunner
from gpu_fault.admin.resource_registry import write_installation_resource_snapshot
from gpu_fault.admin.site import load_site
from gpu_fault.installation_resources import InstallationResourceDeletePolicy as Policy
from gpu_fault.installation_resources import InstallationResourceOwnership as Ownership
from gpu_fault.installation_resources import InstallationResourceSnapshot
from tests.admin._aws_cleanup_support import ACCOUNT, REGION, TAGS, resource
from tests.admin.test_admin_site import site_file


class RemovalTransport(CommandRunner):
    """In-memory AWS/Kubernetes transports for the real remove_cluster pipeline."""

    def __init__(self, directory, monkeypatch):
        super().__init__()
        self.directory = directory
        self.path = site_file(directory)
        self.site = load_site(self.path)
        self.gpu = self.site.release_config["clusters"][0]
        self.hyperpod = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/hp-gpu-a"
        self.events = []
        self.failure = None
        self.namespace = True
        self.node_uid = "uid-node-a"
        self.nodes = None
        self.role_alive = True
        self.keys = {"node-a": base64.b64encode(b"a" * 64).decode()}
        self.node_annotations = {removal.INSTALLER_ANNOTATIONS[0]: "example"}
        self.states = {"gpu-a": "ACTIVE"}
        self.generation = 1
        self.runtime_override = None
        self.hyperpod_override = {}
        self.endpoint_override = None
        self.registry_secret_override = None
        self.cleanup_override = None
        self.snapshot = self.initial_snapshot()
        for module in (removal, evidence, network, bootstrap_common):
            monkeypatch.setattr(module, "run_command", self.command)
        monkeypatch.setattr(aws_commands, "bounded_command", self.command)
        monkeypatch.setattr(removal, "run_driver", self.driver)
        monkeypatch.setattr(removal, "fetch_installation_resource_registry", self.fetch)
        monkeypatch.setattr(removal, "sync_installation_resource_snapshot", self.sync)
        monkeypatch.setattr(
            removal,
            "apply_failure_domain_map",
            lambda _site: self.events.append("failure-domains"),
        )

    def initial_snapshot(self):
        resources = [
            resource(
                "gpu_eks",
                self.gpu["eks_cluster_arn"],
                policy=Policy.PRESERVE,
                ownership=Ownership.EXTERNAL,
            ).model_copy(update={"resource_key": "cluster/gpu-a/eks"}),
            resource(
                "gpu_hyperpod",
                "hp-gpu-a",
                arn=self.hyperpod,
                policy=Policy.PRESERVE,
                ownership=Ownership.EXTERNAL,
            ).model_copy(update={"resource_key": "cluster/gpu-a/hyperpod"}),
            resource(
                "iam_role", "executor", arn=self.gpu["executor_irsa_role_arn"]
            ).model_copy(update={"resource_key": "aws/iam/executor/gpu-a/role"}),
        ]
        value = InstallationResourceSnapshot(site_id="test-site", resources=resources)
        return value.model_copy(update={"source_sha256": value.digest()})

    def fetch(self, site, *, output=None):
        self.events.append("registry-read")
        if output is not None:
            write_installation_resource_snapshot(site, self.snapshot, path=output)
        return self.snapshot

    def sync(self, _site, snapshot):
        self.events.append("registry-sync")
        self.snapshot = snapshot

    def request(self):
        return removal.RemoveClusterRequest(
            load_site(self.path),
            "gpu-a",
            removal.CONFIRMATION,
            self.gpu["eks_cluster_arn"],
        )

    def state(self):
        return json.loads(
            (self.directory / "remove-cluster/gpu-a/state.json").read_text()
        )

    def run(self, arguments, **options):
        result = self.command(arguments, **options)
        if result.returncode:
            raise bootstrap_common.BootstrapError("fake read failed")
        return result.stdout

    def aws_json(self, region, service, operation, *arguments, **_options):
        assert region == REGION
        result = self.command(["aws", service, operation, *arguments])
        if result.returncode:
            raise bootstrap_common.BootstrapError("fake AWS read failed")
        return json.loads(result.stdout)

    def result(self, arguments, value=None, *, code=0, error=""):
        return subprocess.CompletedProcess(
            arguments,
            code,
            value if isinstance(value, str) else json.dumps(value),
            error,
        )

    def command(self, arguments, **options):
        arguments = list(arguments)
        if arguments[0] == "aws":
            service, operation = arguments[1:3]
            if self.failure == operation:
                return self.result(arguments, code=1, error="example transport failure")
            if service == "sts":
                return self.result(arguments, {"Account": ACCOUNT})
            if service == "sagemaker":
                return self.result(
                    arguments,
                    {
                        "ClusterArn": self.hyperpod,
                        "ClusterName": "hp-gpu-a",
                        "NodeRecovery": "None",
                        "Orchestrator": {
                            "Eks": {"ClusterArn": self.gpu["eks_cluster_arn"]}
                        },
                        **self.hyperpod_override,
                    },
                )
            if service == "eks":
                name = arguments[arguments.index("--name") + 1]
                return self.result(
                    arguments,
                    {
                        "cluster": {
                            "arn": f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/{name}",
                            "status": "ACTIVE",
                            "createdAt": "2026-01-01",
                            "endpoint": f"https://{name}.example.invalid",
                            "resourcesVpcConfig": {"vpcId": "vpc-" + name},
                        }
                    },
                )
            if service == "ec2":
                assert operation == "describe-nat-gateways"
                return self.result(arguments, {"NatGateways": []})
            assert service == "iam", "unhandled fake AWS service"
            if operation == "get-role":
                return self.result(
                    arguments,
                    {},
                    code=0 if self.role_alive else 254,
                    error="" if self.role_alive else "An error occurred (NoSuchEntity)",
                )
            if operation == "list-role-tags":
                return self.result(arguments, {"Tags": TAGS})
            if operation == "list-instance-profiles-for-role":
                return self.result(arguments, {"InstanceProfiles": []})
            if operation == "list-role-policies":
                return self.result(arguments, {"PolicyNames": []})
            if operation == "list-attached-role-policies":
                return self.result(arguments, {"AttachedPolicies": []})
            assert operation == "delete-role", "unhandled fake IAM operation"
            self.events.append("delete-role")
            self.role_alive = False
            return self.result(arguments, {})
        assert arguments[0] == "kubectl", "unhandled removal process"
        gpu = "--context" in arguments
        if "config" in arguments and "view" in arguments:
            return self.result(
                arguments,
                self.endpoint_override
                or f"https://{'gpu-a' if gpu else 'control'}.example.invalid",
            )
        if "get" in arguments and "namespace" in arguments:
            return self.result(
                arguments,
                ""
                if gpu and not self.namespace
                else {
                    "apiVersion": "v1",
                    "kind": "Namespace",
                    "metadata": {
                        "name": "gpu-fault-system",
                        "uid": "namespace-gpu" if gpu else "namespace-cpu",
                    },
                },
            )
        if "get" in arguments and "nodes" in arguments:
            if any(item.startswith("go-template=") for item in arguments):
                return self.result(arguments, '["node-a"]')
            value = (
                self.nodes
                if self.nodes is not None
                else {
                    "items": [
                        {
                            "kind": "Node",
                            "metadata": {
                                "name": "node-a",
                                "uid": self.node_uid,
                                "resourceVersion": "1",
                                "labels": {
                                    "sagemaker.amazonaws.com/cluster-name": "hp-gpu-a"
                                },
                                "annotations": dict(self.node_annotations),
                            },
                        }
                    ]
                }
            )
            return self.result(arguments, value)
        if "get" in arguments and "secret" in arguments:
            if "gpu-fault-regional-clusters" in arguments:
                value = (
                    self.registry_secret_override
                    if self.registry_secret_override is not None
                    else {
                        "kind": "Secret",
                        "metadata": {
                            "name": "gpu-fault-regional-clusters",
                            "namespace": "gpu-fault-system",
                            "uid": "registry-uid",
                        },
                        "data": {
                            "clusters.json": base64.b64encode(
                                json.dumps(
                                    [{"cluster_id": key} for key in self.states]
                                ).encode()
                            ).decode()
                        },
                    }
                )
                return self.result(arguments, value)
            if any(item.startswith("go-template=") for item in arguments):
                return self.result(arguments, json.dumps(sorted(self.keys)))
            return self.result(
                arguments,
                {
                    "apiVersion": "v1",
                    "kind": "Secret",
                    "type": "Opaque",
                    "metadata": {
                        "name": "gpu-fault-node-action-keys",
                        "namespace": "gpu-fault-system",
                        "uid": "gpu-keys" if gpu else "cpu-keys",
                        "resourceVersion": "1",
                    },
                    "data": dict(self.keys),
                },
            )
        if "get" in arguments and "pod" in arguments:
            return self.result(arguments, "cpu-fixture")
        if "get" in arguments and "configmap" in arguments:
            return self.result(
                arguments, {"data": {"state.json": '{"release_id":"release-a"}'}}
            )
        if "exec" in arguments:
            if arguments[-1] == cluster_join_nodes.AGENT_OWNERS_SCRIPT:
                return self.result(
                    arguments, {"node-a": "gpu-a"} if self.states else {}
                )
            assert arguments[-1] == evidence.REGISTRY_STATUS_CLIENT
            value = (
                self.runtime_override
                if self.runtime_override is not None
                else {
                    "generation": self.generation,
                    "content_sha256": "a" * 64,
                    "cluster_states": dict(self.states),
                    "required_member_ids": [],
                    "acked_member_ids": [],
                    "missing_member_ids": [],
                    "active_member_ids": [],
                    "members": [],
                    "converged": True,
                }
            )
            return self.result(arguments, value)
        if "patch" in arguments:
            flag = "-p" if "-p" in arguments else "--patch"
            patch = json.loads(arguments[arguments.index(flag) + 1])
            assert any(
                item["op"] == "test" and item["path"] == "/metadata/uid"
                for item in patch
            ), "metadata mutation is missing its UID precondition"
            if "secret" in arguments:
                self.keys.clear()
                self.events.append("remove-keys")
            else:
                self.node_annotations.clear()
                self.events.append("clear-annotations")
            return self.result(arguments, {})
        if "--raw" in arguments:
            body = json.loads(options["input_text"])
            assert body["preconditions"]["uid"] == "namespace-gpu"
            self.namespace = False
            self.events.append("delete-namespace")
            return self.result(arguments, {})
        raise AssertionError("unhandled fake Kubernetes operation")

    def driver(self, arguments, **_options):
        if Path(arguments[0]).name == "prepare-clean-redeploy.sh":
            mode = "cleanup"
        else:
            mode = arguments[1]
        self.events.append(mode)
        if mode == self.failure:
            return self.result(arguments, code=1, error="example driver failure")
        if mode == "cleanup":
            configuration = json.loads(
                Path(arguments[arguments.index("--config") + 1]).read_text()
            )
            document = {
                "schema_version": 2,
                "config_sha256": hashlib.sha256(
                    json.dumps(configuration, indent=2, sort_keys=True).encode()
                ).hexdigest(),
                "targets": {
                    "namespace": configuration["namespace"],
                    "cpu_kubeconfig": configuration["cpu_kubeconfig"],
                    "clusters": [
                        {"cluster_id": item["cluster_id"], "context": item["context"]}
                        for item in configuration["clusters"]
                    ],
                },
                "scope": "gpu",
                "mode": "clean",
                "node_mode": "uninstall",
                "phase": "CLEANUP_COMPLETED",
                "status": "COMPLETED",
                "original_resources": [
                    {
                        "scope": "gpu:gpu-a",
                        "context": "gpu-a",
                        "kind": "Namespace",
                        "uid": "namespace-gpu",
                    }
                ],
            }
            if self.cleanup_override:
                document.update(self.cleanup_override)
            document["content_sha256"] = removal.canonical_digest(document)
            Path(arguments[arguments.index("--state-file") + 1]).write_text(
                json.dumps(document)
            )
        elif mode == "drain-cluster":
            self.states["gpu-a"] = "DRAINING"
            self.generation += 1
        elif mode == "remove-cluster":
            self.states.clear()
            self.generation += 1
        else:
            assert mode in {"sync-state", "verify"}, "unhandled fake lifecycle driver"
        return self.result(arguments, {})

    def remove(self):
        return removal.remove_cluster(self.request(), runner=self)
