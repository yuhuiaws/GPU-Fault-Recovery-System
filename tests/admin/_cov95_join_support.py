from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from threading import RLock

import yaml

from gpu_fault.admin import cluster_batch_join as batch
from gpu_fault.admin import cluster_join as join
from gpu_fault.admin import cluster_join_evidence as evidence
from gpu_fault.admin import cluster_readiness as readiness
from gpu_fault.admin import cluster_removal as removal
from gpu_fault.admin import failure_domain_map as maps
from gpu_fault.admin import resource_registry as registry
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
)
from gpu_fault.admin.site import load_site
from tests.admin.test_admin_failure_domain_map import FakeKubectl
from tests.admin.test_admin_site import site_file
from tests.admin.test_cov95_resource_registry import RegistryTransport
from tests.admin.test_registry_transport import snapshot

ACCOUNT = "123456789012"
REGION = "us-east-1"


def target(suffix="b"):
    arn = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/gpu-{suffix}"
    return ClusterIdentity(
        input_arn=arn,
        role="gpu",
        region=REGION,
        account_id=ACCOUNT,
        hyperpod_arn=f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/hp-gpu-{suffix}",
        hyperpod_name=f"hp-gpu-{suffix}",
        eks_arn=arn,
        eks_name=f"gpu-{suffix}",
        vpc_id=f"vpc-gpu-{suffix}",
        subnet_ids=(f"subnet-{suffix}",),
        subnet_cidrs=("10.2.0.0/24",),
        node_recovery="None",
        context=f"gpu-{suffix}",
    )


class JoinRegistry(RegistryTransport):
    def __init__(self, scenario):
        super().__init__()
        self.scenario = scenario
        self.live = snapshot()

    def __call__(self, arguments, **options):
        self.calls.append((arguments, options))
        assert arguments[0] == "kubectl", "unexpected join registry command"
        if "get" in arguments and "pod" in arguments:
            return self.pod
        assert "exec" in arguments
        if arguments[-1] == registry.SYNC_SCRIPT:
            self.scenario.event("registry-sync")
            incoming = registry.InstallationResourceSnapshot.model_validate_json(
                options["input_text"]
            )
            merged = {item.resource_key: item for item in self.live.resources}
            merged.update({item.resource_key: item for item in incoming.resources})
            self.live = registry.InstallationResourceSnapshot(
                site_id=incoming.site_id,
                resources=sorted(merged.values(), key=lambda item: item.resource_key),
            )
            acknowledged = incoming.resources
        else:
            assert arguments[-1] == registry.FETCH_SCRIPT
            self.scenario.event("registry-fetch")
            acknowledged = self.live.resources
        return self.scenario.result(
            arguments,
            json.dumps([item.model_dump(mode="json") for item in acknowledged]),
        )


class JoinScenario(CommandRunner):
    """Stateful public-boundary fake; all filesystem writes stay inside tmp_path."""

    def __init__(self, tmp_path, monkeypatch, *, suffixes=("b", "c"), rollback=False):
        super().__init__()
        self.directory = tmp_path
        self.path = site_file(tmp_path)
        document = yaml.safe_load(self.path.read_text())
        self.gpu_kubeconfig = tmp_path / "secure/gpu.kubeconfig"
        document["spec"]["gpuKubeconfig"] = str(self.gpu_kubeconfig)
        document["spec"]["dns"] = {
            "hostedZoneId": "ZEXAMPLE",
            "hostname": "api.example.invalid",
        }
        document["spec"]["autoRollback"] = rollback
        self.path.write_text(yaml.safe_dump(document, sort_keys=False))
        self.site = load_site(self.path)
        self.targets = {target(suffix).eks_arn: target(suffix) for suffix in suffixes}
        self.events = []
        self.commands = []
        self.driver_calls = []
        self.failure = None
        self.failure_error = BootstrapError("modeled join boundary failure")
        self.namespace_uids = {}
        self.contexts = {"gpu-a"}
        self.vpcs = {"vpc-gpu-a"}
        self.ingress = {"192.0.2.10"}
        self.cluster_states = {"gpu-a": "ACTIVE"}
        self.generation = 1
        self.node_names = {suffix: f"node-{suffix}" for suffix in ("a", *suffixes)}
        self.network_error = None
        self.discovery_error = None
        self.namespace_read_override = None
        self.lock = RLock()
        self.registry = JoinRegistry(self)
        self.map_kubectl = FakeKubectl()
        self.map_calls = []
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        for module in (join, evidence, readiness):
            monkeypatch.setattr(module, "run_command", self.command)
        monkeypatch.setattr(join, "run_driver", self.driver)
        monkeypatch.setattr(removal, "run_driver", self.driver)
        monkeypatch.setattr(registry, "run_command", self.registry)
        monkeypatch.setattr(maps, "run_command", self.map_command)
        monkeypatch.setattr(join, "discover_cluster", self.discover)
        monkeypatch.setattr(join, "ensure_executor_role", self.executor_role)
        monkeypatch.setattr(join, "ensure_adot_writer_role", self.adot_role)
        monkeypatch.setattr(join, "provision_node_action_keys", self.node_keys)
        monkeypatch.setattr(
            join,
            "apply_failure_domain_map",
            lambda _site: self.event("failure-domains"),
        )

    def event(self, name):
        with self.lock:
            self.events.append(name)
            if self.failure == name:
                raise self.failure_error

    def result(self, arguments, output="", code=0, error=""):
        import subprocess

        return subprocess.CompletedProcess(arguments, code, output, error)

    def discover(self, _runner, *, cluster_arn, **_kwargs):
        self.event("discover:" + cluster_arn.rsplit("/", 1)[-1])
        if self.discovery_error is not None:
            raise self.discovery_error
        return self.targets[cluster_arn]

    def request(self, suffix="b", **changes):
        return join.JoinClusterRequest(
            site=load_site(self.path), gpu_cluster_arn=target(suffix).eks_arn, **changes
        )

    def state_path(self, suffix="b"):
        identity = hashlib.sha256(target(suffix).eks_arn.encode()).hexdigest()[:12]
        return self.path.parent / "join-cluster" / identity / "state.json"

    def state(self, suffix="b"):
        return json.loads(self.state_path(suffix).read_text())

    def run(self, arguments, **options):
        self.commands.append((list(arguments), dict(options)))
        if "update-kubeconfig" in arguments:
            self.event("kubeconfig")
            path = Path(arguments[arguments.index("--kubeconfig") + 1])
            assert path.is_relative_to(self.directory), (
                "fake attempted external file mutation"
            )
            path.write_text("example-kubeconfig")
            self.contexts.add(arguments[arguments.index("--alias") + 1])
            return ""
        if "associate-vpc-with-hosted-zone" in arguments:
            self.event("associate")
            value = arguments[arguments.index("--vpc") + 1].split("VPCId=", 1)[1]
            self.vpcs.add(value)
            return ""
        context = (
            arguments[arguments.index("--context") + 1]
            if "--context" in arguments
            else "cpu"
        )
        if "get-contexts" in arguments:
            return "\n".join(sorted(self.contexts))
        if "get" in arguments and "namespace" in arguments:
            if self.namespace_read_override is not None:
                return self.namespace_read_override
            uid = self.namespace_uids.get(context)
            return (
                json.dumps(
                    {
                        "kind": "Namespace",
                        "metadata": {"name": "gpu-fault-system", "uid": uid},
                    }
                )
                if uid
                else ""
            )
        if "create" in arguments and "namespace" in arguments:
            return json.dumps(
                {"kind": "Namespace", "metadata": {"name": "gpu-fault-system"}}
            )
        if "apply" in arguments:
            self.event("namespace:" + context)
            assert json.loads(options["input_text"])["kind"] == "Namespace"
            self.namespace_uids[context] = "namespace-" + context
            return ""
        if "get" in arguments and "secret" in arguments:
            if "gpu-fault-control-plane-active" in arguments:
                assert options.get("sensitive"), (
                    "CPU credential read lost private capture"
                )
                return base64.b64encode(b"f" * 64).decode()
            return ""
        if "get" in arguments and "nodes" in arguments:
            suffix = context.rsplit("-", 1)[-1]
            name = self.node_names[suffix]
            if any(str(item).startswith("go-template=") for item in arguments):
                return json.dumps([name])
            return json.dumps(
                {
                    "items": [
                        {
                            "kind": "Node",
                            "metadata": {
                                "name": name,
                                "uid": "uid-" + name,
                                "resourceVersion": "1",
                                "labels": {
                                    "sagemaker.amazonaws.com/cluster-name": "hp-gpu-"
                                    + suffix,
                                    "sagemaker.amazonaws.com/instance-group-name": "group-"
                                    + suffix,
                                },
                            },
                        }
                    ]
                }
            )
        if "get" in arguments and "pod" in arguments:
            return "cpu-fixture"
        if "exec" in arguments:
            return "{}"
        raise AssertionError("unexpected fake join runner operation")

    def aws_json(self, region, service, operation, *arguments, **_options):
        assert region == REGION
        self.event(f"{service}/{operation}")
        if self.network_error is not None:
            raise self.network_error
        if (service, operation) == ("eks", "describe-cluster"):
            name = arguments[arguments.index("--name") + 1]
            return {
                "cluster": {
                    "arn": f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/{name}",
                    "status": "ACTIVE",
                    "createdAt": "2026-01-01T00:00:00Z",
                    "endpoint": f"https://{name}.example.invalid",
                    "resourcesVpcConfig": {"vpcId": "vpc-" + name},
                }
            }
        if (service, operation) == ("ec2", "describe-nat-gateways"):
            vpc = next(item for item in arguments if "Name=vpc-id" in item)
            suffix = vpc.rsplit("-", 1)[-1]
            return {
                "NatGateways": [
                    {
                        "NatGatewayAddresses": [
                            {"PublicIp": f"192.0.2.{10 * (ord(suffix) - ord('a') + 1)}"}
                        ]
                    }
                ]
            }
        if (service, operation) == ("route53", "get-hosted-zone"):
            return {
                "HostedZone": {"Id": "/hostedzone/ZEXAMPLE"},
                "VPCs": [
                    {"VPCRegion": REGION, "VPCId": value} for value in sorted(self.vpcs)
                ],
            }
        raise AssertionError(f"unexpected fake AWS operation {service}/{operation}")

    def executor_role(self, _runner, *, cluster, **_options):
        self.event("executor-role:" + cluster.eks_name)
        return {
            "role_arn": f"arn:aws:iam::{ACCOUNT}:role/{cluster.eks_name}",
            "ownership": "CREATED",
            "inline_policy_name": "GPUFaultRegionalExecutor",
            "oidc_provider_arn": f"arn:aws:iam::{ACCOUNT}:oidc-provider/{cluster.eks_name}.invalid",
            "oidc_provider_ownership": "CREATED",
            "cluster_name": cluster.eks_name,
        }

    def map_command(self, arguments, **options):
        self.map_calls.append(list(arguments))
        if "get" in arguments and "nodes" in arguments:
            self.event("map-nodes")
            return self.result(arguments, self.run(arguments))
        if "apply" in arguments:
            self.event("map-apply")
        elif "patch" in arguments:
            self.event("map-patch")
        elif "rollout" in arguments:
            self.event("map-rollout")
        else:
            self.event("map-readback")
        return self.map_kubectl(arguments, **options)

    def adot_role(self, _runner, *, cluster, **_options):
        self.event("adot-role:" + cluster.eks_name)
        return {
            "role_arn": f"arn:aws:iam::{ACCOUNT}:role/{cluster.eks_name}-adot",
            "ownership": "CREATED",
            "inline_policy_name": "GPUFaultDataplaneAmpWriter",
        }

    def node_keys(self, runner, *, cluster, cluster_id, **_options):
        self.event("node-keys:" + cluster.eks_name)
        return {"cluster_id": cluster_id}

    def command(self, arguments, **_options):
        if arguments[0] == "aws":
            if arguments[1:3] == ["eks", "update-kubeconfig"]:
                return self.result(arguments, self.run(arguments))
            assert arguments[1:3] == ["ec2", "authorize-security-group-ingress"]
            eip = arguments[arguments.index("--cidr") + 1].split("/")[0]
            self.event("ingress:" + eip)
            if eip in self.ingress:
                return self.result(
                    arguments,
                    code=254,
                    error="An error occurred (InvalidPermission.Duplicate)",
                )
            self.ingress.add(eip)
            return self.result(arguments)
        assert arguments[0] == "kubectl"
        if "config" in arguments:
            if "delete-context" in arguments:
                self.contexts.discard(arguments[-1])
                return self.result(arguments)
            if "view" in arguments:
                return self.result(arguments)
            return self.result(arguments, self.run(arguments))
        if "get" in arguments and "deployment" in arguments:
            return self.result(arguments)
        if "get" in arguments and "pods" in arguments:
            return self.result(arguments, '{"kind":"PodList","items":[]}')
        if "get" in arguments and "nodes" in arguments:
            return self.result(arguments, self.run(arguments))
        if "configmap" in arguments:
            output = {"data": {"state.json": json.dumps({"release_id": "release-a"})}}
        elif "get" in arguments and "pod" in arguments:
            return self.result(arguments, "cpu-fixture")
        elif arguments[-1] == readiness.COLLECTOR_READINESS_SCRIPT:
            self.event("collectors")
            cluster_id = next(
                item.split("=", 1)[1]
                for item in arguments
                if item.startswith("CLUSTER_ID=")
            )
            output = {
                "cluster_id": cluster_id,
                "ready": True,
                "nodes": [{"ready": True}],
            }
        else:
            assert arguments[-1] == evidence.REGISTRY_STATUS_CLIENT
            self.event("membership")
            output = {
                "generation": self.generation,
                "content_sha256": hashlib.sha256(
                    json.dumps(self.cluster_states, sort_keys=True).encode()
                ).hexdigest(),
                "cluster_states": dict(self.cluster_states),
                "required_member_ids": [],
                "acked_member_ids": [],
                "missing_member_ids": [],
                "active_member_ids": [],
                "members": [],
                "converged": True,
            }
        return self.result(arguments, json.dumps(output))

    def driver(self, arguments, **_options):
        mode = arguments[1]
        cluster_id = (
            arguments[arguments.index("--cluster-id") + 1]
            if "--cluster-id" in arguments
            else None
        )
        self.driver_calls.append((mode, cluster_id))
        self.event(mode + (":" + cluster_id if cluster_id else ""))
        with self.lock:
            if mode == "join-cluster":
                self.cluster_states[cluster_id] = "PENDING"
                self.generation += 1
            elif mode == "activate-cluster":
                self.cluster_states[cluster_id] = "ACTIVE"
                self.generation += 1
        return self.result(arguments)

    def join(self, suffix="b"):
        return join.join_cluster(self.request(suffix), runner=self)

    def batch(self, suffixes=("b", "c")):
        return batch.join_clusters(
            tuple(self.request(suffix) for suffix in suffixes),
            runner_factory=lambda: self,
        )
