"""A recorded plain EKS world for ``gpu-fault-admin prepare-eks-nodegroup`` tests.

One GPU cluster ``gpu-eks-cluster`` with a Spot ``p5en.48xlarge`` managed node
group on launch template ``lt-0gpu`` and a CPU node group without one. The fake
runner answers the ``aws`` and ``kubectl`` argument vectors the verb sends and
applies the writes to the world, so a second run sees what the first one did.
Every call is recorded with its ``mutate`` flag.
"""

from __future__ import annotations

import base64
import copy
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import eks_nodegroup_prepare as verb
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
)
from gpu_fault.admin.bootstrap_site import site_identifier

REGION = "us-west-2"
ACCOUNT = "123456789012"
CPU_EKS_ARN = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/control-plane-eks"
CPU_HYPERPOD_ARN = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/fixturecpuhp01"
GPU_NAME = "gpu-eks-cluster"
GPU_ARN = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/{GPU_NAME}"
GPU_GROUP = "gpu-p5en-efa-spot-node-group"
CPU_GROUP = "cpu-node-group"
LT_ID = "lt-0gpu"
RELEASE = "1.32.9-20251016"
FIXED_NOW = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
NODECONFIG_PART = (
    "Content-Type: application/node.eks.aws\n"
    "\n"
    "---\n"
    "apiVersion: node.eks.aws/v1alpha1\n"
    "kind: NodeConfig\n"
    "spec:\n"
    "  kubelet:\n"
    "    config:\n"
    "      maxPods: 110\n"
    "\n"
)
SITE_SCRIPT_PART = (
    'Content-Type: text/x-shellscript; charset="us-ascii"\n'
    'Content-Disposition: attachment; filename="site-tools.sh"\n'
    "\n"
    "#!/bin/bash\n"
    "dnf install -y -q datacenter-gpu-manager-4-core datacenter-gpu-manager-4-cuda13\n"
    "\n"
)


def site_user_data() -> str:
    """A launch template's user data with a NodeConfig and a site part."""

    return (
        "MIME-Version: 1.0\n"
        'Content-Type: multipart/mixed; boundary="==SITE=="\n'
        "\n"
        "--==SITE==\n"
        + NODECONFIG_PART
        + "--==SITE==\n"
        + SITE_SCRIPT_PART
        + "--==SITE==--\n"
    )


def encoded(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def cpu_identity() -> ClusterIdentity:
    return ClusterIdentity(
        input_arn=CPU_EKS_ARN,
        role="cpu",
        region=REGION,
        account_id=ACCOUNT,
        hyperpod_arn=CPU_HYPERPOD_ARN,
        hyperpod_name="control-plane-fixture",
        eks_arn=CPU_EKS_ARN,
        eks_name="control-plane-eks",
        vpc_id="vpc-1",
        subnet_ids=("subnet-a",),
        node_recovery="None",
        context="cpu",
    )


SITE_ID = site_identifier(cpu_identity(), [])


def gpu_nodegroup(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "nodegroupName": GPU_GROUP,
        "status": "ACTIVE",
        "amiType": "AL2023_x86_64_NVIDIA",
        "capacityType": "SPOT",
        "releaseVersion": RELEASE,
        "instanceTypes": None,
        "scalingConfig": {"minSize": 2, "maxSize": 2, "desiredSize": 2},
        "updateConfig": {"maxUnavailable": 1},
        "labels": {"node-type": "gpu", "nvidia.com/gpu.present": "true"},
        "nodeRepairConfig": None,
        "launchTemplate": {
            "name": "gpu-p5en-efa-spot-nodes",
            "version": "3",
            "id": LT_ID,
        },
    }
    document.update(overrides)
    return document


def cpu_nodegroup() -> dict[str, Any]:
    return {
        "nodegroupName": CPU_GROUP,
        "status": "ACTIVE",
        "amiType": "AL2023_x86_64_STANDARD",
        "capacityType": "ON_DEMAND",
        "releaseVersion": RELEASE,
        "instanceTypes": ["m7i.xlarge"],
        "labels": {"node-type": "cpu"},
    }


def _options(arguments: Sequence[str]) -> dict[str, list[str]]:
    options: dict[str, list[str]] = {}
    current: str | None = None
    for word in arguments:
        if word.startswith("--"):
            current = word
            options.setdefault(word, [])
        elif current is not None:
            options[current].append(word)
    return options


class World:
    """The provider state the fake runner reads and changes."""

    def __init__(self, *, user_data: str | None = None) -> None:
        self.cluster_status = "ACTIVE"
        self.nodegroups: dict[str, dict[str, Any]] = {
            GPU_GROUP: gpu_nodegroup(),
            CPU_GROUP: cpu_nodegroup(),
        }
        source = {
            "InstanceType": "p5en.48xlarge",
            "NetworkInterfaces": [{"InterfaceType": "efa"}],
        }
        if user_data is not None:
            source["UserData"] = encoded(user_data)
        self.templates: dict[str, dict[int, dict[str, Any]]] = {
            LT_ID: {
                1: {"data": {"InstanceType": "p5en.48xlarge"}, "description": "v1"},
                3: {"data": source, "description": "efa + placement group"},
            }
        }
        self.hyperpods: list[dict[str, Any]] = []
        self.instances: dict[str, list[str]] = {GPU_GROUP: ["i-old1", "i-old2"]}
        #: Per roll call: (describe-update polls before the end, final status).
        self.roll_results: list[tuple[int, str]] = []
        self.config_polls = 1
        self.updates: dict[str, dict[str, Any]] = {}
        #: What each instance reports to the inventory (default: old_node()).
        self.node_facts: dict[str, dict[str, Any]] = {}
        self.rolls = 0
        self.ssm_send_errors: list[str] = []
        self.namespaces = ["default", "kube-system", "training"]
        self.daemonsets: list[dict[str, Any]] = [nvidia_plugin(), efa_plugin()]
        self.kubernetes_error: str | None = None
        self.sent: dict[str, Any] = {}
        self.calls: list[tuple[list[str], bool]] = []

    # --- views ----------------------------------------------------------------

    def aws_operations(self, *, mutate: bool | None = None) -> list[str]:
        return [
            f"{argv[1]} {argv[2]}"
            for argv, flag in self.calls
            if argv[0] == "aws" and (mutate is None or flag is mutate)
        ]

    def template_user_data(self, version: int) -> str | None:
        raw = self.templates[LT_ID][version]["data"].get("UserData")
        return base64.b64decode(raw).decode() if raw else None

    # --- eks --------------------------------------------------------------------

    def eks_describe_cluster(self, _options: Mapping[str, list[str]]) -> dict[str, Any]:
        return {
            "cluster": {"name": GPU_NAME, "arn": GPU_ARN, "status": self.cluster_status}
        }

    def eks_list_nodegroups(self, _options: Mapping[str, list[str]]) -> dict[str, Any]:
        return {"nodegroups": sorted(self.nodegroups)}

    def eks_describe_nodegroup(
        self, options: Mapping[str, list[str]]
    ) -> dict[str, Any]:
        return {
            "nodegroup": copy.deepcopy(self.nodegroups[options["--nodegroup-name"][0]])
        }

    def new_update(self, group: str, kind: str, **fields: Any) -> dict[str, Any]:
        update_id = f"update-{len(self.updates) + 1}"
        self.updates[update_id] = {"group": group, "kind": kind, **fields}
        return {"update": {"id": update_id, "status": "InProgress", "type": kind}}

    def eks_update_nodegroup_config(
        self, options: Mapping[str, list[str]]
    ) -> dict[str, Any]:
        group = options["--nodegroup-name"][0]
        document = self.nodegroups[group]
        if document["status"] != "ACTIVE":
            raise BootstrapError("ResourceInUseException: the node group is updating")
        if "--node-repair-config" in options:
            document["nodeRepairConfig"] = json.loads(
                options["--node-repair-config"][0]
            )
        if "--labels" in options:
            payload = json.loads(options["--labels"][0])
            labels = dict(document.get("labels") or {})
            labels.update(payload.get("addOrUpdateLabels") or {})
            for key in payload.get("removeLabels") or ():
                labels.pop(key, None)
            document["labels"] = labels
        return self.new_update(
            group, "ConfigUpdate", polls=self.config_polls, final="Successful"
        )

    def eks_update_nodegroup_version(
        self, options: Mapping[str, list[str]]
    ) -> dict[str, Any]:
        group = options["--nodegroup-name"][0]
        document = self.nodegroups[group]
        if document["status"] != "ACTIVE":
            raise BootstrapError("ResourceInUseException: the node group is updating")
        template = json.loads(options["--launch-template"][0])
        polls, final = (
            self.roll_results.pop(0) if self.roll_results else (2, "Successful")
        )
        document["status"] = "UPDATING"
        return self.new_update(
            group,
            "VersionUpdate",
            polls=polls,
            final=final,
            version=str(template["version"]),
            release=(options.get("--release-version") or [None])[0],
        )

    def eks_describe_update(self, options: Mapping[str, list[str]]) -> dict[str, Any]:
        entry = self.updates[options["--update-id"][0]]
        if entry["polls"] > 0:
            entry["polls"] -= 1
            return {"update": {"id": options["--update-id"][0], "status": "InProgress"}}
        if not entry.get("settled"):
            entry["settled"] = True
            document = self.nodegroups[entry["group"]]
            document["status"] = "ACTIVE"
            if entry["kind"] == "VersionUpdate" and entry["final"] == "Successful":
                document["launchTemplate"]["version"] = entry["version"]
                self.rolls += 1
                fresh = [f"i-new{self.rolls}a", f"i-new{self.rolls}b"]
                self.instances[entry["group"]] = fresh
                booted = booted_node(self.template_user_data(int(entry["version"])))
                for instance in fresh:
                    self.node_facts[instance] = copy.deepcopy(booted)
        errors = (
            [
                {
                    "errorCode": "NodeCreationFailure",
                    "errorMessage": "Instances failed to join the kubernetes cluster",
                    "resourceIds": ["i-spot"],
                }
            ]
            if entry["final"] == "Failed"
            else []
        )
        return {
            "update": {
                "id": options["--update-id"][0],
                "status": entry["final"],
                "errors": errors,
            }
        }

    def eks_update_kubeconfig(
        self, _options: Mapping[str, list[str]]
    ) -> dict[str, Any]:
        if self.kubernetes_error:
            raise BootstrapError(self.kubernetes_error)
        return {}

    # --- ec2 --------------------------------------------------------------------

    def version_document(self, template: str, number: int) -> dict[str, Any]:
        entry = self.templates[template][number]
        return {
            "LaunchTemplateId": template,
            "VersionNumber": number,
            "VersionDescription": entry["description"],
            "LaunchTemplateData": copy.deepcopy(entry["data"]),
        }

    def ec2_describe_launch_template_versions(
        self, options: Mapping[str, list[str]]
    ) -> dict[str, Any]:
        template = options["--launch-template-id"][0]
        versions = self.templates[template]
        if "--versions" in options:
            wanted = options["--versions"][0]
            number = max(versions) if wanted in ("$Latest", "$Default") else int(wanted)
            return {"LaunchTemplateVersions": [self.version_document(template, number)]}
        low = int(options["--min-version"][0])
        return {
            "LaunchTemplateVersions": [
                self.version_document(template, number)
                for number in sorted(versions)
                if number >= low
            ]
        }

    def ec2_create_launch_template_version(
        self, options: Mapping[str, list[str]]
    ) -> dict[str, Any]:
        template = options["--launch-template-id"][0]
        source = int(options["--source-version"][0])
        data = {
            **copy.deepcopy(self.templates[template][source]["data"]),
            **json.loads(options["--launch-template-data"][0]),
        }
        number = max(self.templates[template]) + 1
        self.templates[template][number] = {
            "data": data,
            "description": options["--version-description"][0],
        }
        return {"LaunchTemplateVersion": self.version_document(template, number)}

    def ec2_describe_instances(
        self, options: Mapping[str, list[str]]
    ) -> dict[str, Any]:
        filters = dict(item.split(",Values=", 1) for item in options["--filters"])
        group = filters["Name=tag:eks:nodegroup-name"]
        assert filters["Name=tag:eks:cluster-name"] == GPU_NAME, filters
        return {
            "Reservations": [
                {
                    "Instances": [
                        {"InstanceId": item} for item in self.instances.get(group, [])
                    ]
                }
            ]
        }

    # --- ssm / sagemaker --------------------------------------------------------

    def ssm_send_command(self, options: Mapping[str, list[str]]) -> dict[str, Any]:
        if self.ssm_send_errors:
            raise BootstrapError(self.ssm_send_errors.pop(0))
        assert len(options["--comment"][0]) <= 100, "SendCommand's comment limit"
        self.sent = {
            "instances": options["--instance-ids"],
            "commands": json.loads(options["--parameters"][0])["commands"],
        }
        return {"Command": {"CommandId": "cmd-1"}}

    def ssm_get_command_invocation(
        self, options: Mapping[str, list[str]]
    ) -> dict[str, Any]:
        instance = options["--instance-id"][0]
        facts = self.node_facts.get(instance) or old_node()
        lines = [
            f"gpu-fault-node {key} {facts[key]}"
            for key in (
                "python3.12-venv",
                "strace",
                "driver-cuda-major",
                "service",
                "hostengine",
                "dcgmi-discovery",
            )
        ]
        lines += [
            f"gpu-fault-node package {name} {version}"
            for name, version in sorted(facts["packages"].items())
        ]
        return {
            "Status": facts.get("ssm_status", "Success"),
            "StandardOutputContent": "\n".join(lines) + "\n",
        }

    def sagemaker_list_clusters(
        self, _options: Mapping[str, list[str]]
    ) -> dict[str, Any]:
        return {
            "ClusterSummaries": [
                {"ClusterName": item["ClusterName"]} for item in self.hyperpods
            ]
        }

    def sagemaker_describe_cluster(
        self, options: Mapping[str, list[str]]
    ) -> dict[str, Any]:
        name = options["--cluster-name"][0]
        return next(item for item in self.hyperpods if item["ClusterName"] == name)

    # --- kubectl ----------------------------------------------------------------

    def kubectl(self, argv: Sequence[str]) -> str:
        if self.kubernetes_error:
            raise BootstrapError(self.kubernetes_error)
        if "namespaces" in argv:
            return json.dumps(
                {"items": [{"metadata": {"name": name}} for name in self.namespaces]}
            )
        assert "daemonsets" in argv and "--all-namespaces" in argv, argv
        return json.dumps({"items": self.daemonsets})


def old_node(**overrides: Any) -> dict[str, Any]:
    """A node prepared by hand: python3.12 and strace, no DCGM host packages."""

    facts: dict[str, Any] = {
        "python3.12-venv": "ok",
        "strace": "ok",
        "driver-cuda-major": "13",
        "service": "inactive",
        "hostengine": "absent",
        "dcgmi-discovery": "failed",
        "packages": {},
    }
    facts.update(overrides)
    return facts


def booted_node(user_data: str | None) -> dict[str, Any]:
    """What a node booted from ``user_data`` reports: the managed part's
    packages (the DCGM pin in the CUDA 13 build of the driver)."""

    text = user_data or ""
    managed = "gpu-fault.io/user-data-part: node-prerequisites" in text
    facts = old_node(
        **({} if managed else {"python3.12-venv": "missing", "strace": "missing"})
    )
    pin = re.search(r"dcgm_version='([^']+)'", text)
    if pin is not None:
        build = pin.group(1) if "-" in pin.group(1) else f"{pin.group(1)}-2"
        facts.update(
            service="active",
            hostengine="running",
            packages={
                "datacenter-gpu-manager-4-core": build,
                "datacenter-gpu-manager-4-cuda13": build,
            },
        )
        facts["dcgmi-discovery"] = "ok"
    return facts


class FakeAws(CommandRunner):
    """Answers the verb's ``aws``/``kubectl`` vectors from a :class:`World`."""

    def __init__(self, world: World) -> None:
        super().__init__()
        self.world = world

    def run(
        self, arguments: Sequence[str], *, mutate: bool = False, **_kwargs: Any
    ) -> str:
        argv = list(arguments)
        self.world.calls.append((argv, mutate))
        if argv[0] == "kubectl":
            return self.world.kubectl(argv)
        assert argv[0] == "aws", argv
        words = list(argv[3:])
        for flag in ("--region", "--output"):
            if flag in words:
                index = words.index(flag)
                del words[index : index + 2]
        handler = getattr(self.world, f"{argv[1]}_{argv[2].replace('-', '_')}")
        return json.dumps(handler(_options(words)))


def daemonset(
    namespace: str,
    name: str,
    *,
    labels: Mapping[str, str],
    tolerations: Sequence[Mapping[str, Any]] = (),
    volumes: Sequence[Mapping[str, Any]] = (),
    containers: Sequence[Mapping[str, Any]] = (),
    ready: int = 2,
    desired: int = 2,
) -> dict[str, Any]:
    return {
        "metadata": {"namespace": namespace, "name": name},
        "spec": {
            "template": {
                "metadata": {"labels": dict(labels)},
                "spec": {
                    "tolerations": [dict(item) for item in tolerations],
                    "volumes": [dict(item) for item in volumes],
                    "containers": [dict(item) for item in containers]
                    or [{"name": name, "image": "public.ecr.aws/example/agent:1"}],
                },
            }
        },
        "status": {"desiredNumberScheduled": desired, "numberReady": ready},
    }


def nvidia_plugin(**overrides: Any) -> dict[str, Any]:
    return daemonset(
        "kube-system",
        "nvdp-nvidia-device-plugin",
        labels={"app.kubernetes.io/name": "nvidia-device-plugin"},
        tolerations=[
            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
        ],
        volumes=[
            {"name": "dp", "hostPath": {"path": "/var/lib/kubelet/device-plugins"}}
        ],
        containers=[
            {
                "name": "nvidia-device-plugin-ctr",
                "image": "nvcr.io/nvidia/k8s-device-plugin:v0.17.1",
            }
        ],
        **overrides,
    )


def efa_plugin(**overrides: Any) -> dict[str, Any]:
    return daemonset(
        "kube-system",
        "aws-efa-k8s-device-plugin-daemonset",
        labels={"name": "aws-efa-k8s-device-plugin"},
        tolerations=[{"key": "aws.amazon.com/efa", "operator": "Exists"}],
        volumes=[
            {"name": "dp", "hostPath": {"path": "/var/lib/kubelet/device-plugins"}}
        ],
        **overrides,
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def run_verb(
    world: World,
    *arguments: str,
    clock: FakeClock | None = None,
    cpu: ClusterIdentity | None = None,
    gpu_arn: str = GPU_ARN,
) -> int:
    """Run the verb against ``world`` with the CLI's own argument parsing."""

    parsed = admin_cli.parser().parse_args(
        [
            verb.VERB,
            "--cpu-cluster-arn",
            CPU_EKS_ARN,
            "--gpu-cluster-arn",
            gpu_arn,
            *arguments,
        ]
    )
    ticking = clock or FakeClock()
    return verb.run_command(
        parsed,
        runner=FakeAws(world),
        discover=lambda _runner, **_arguments: cpu or cpu_identity(),
        identity=lambda: "arn:aws:sts::123456789012:assumed-role/Admin/op",
        sleep=ticking.sleep,
        clock=ticking.monotonic,
        now=lambda: FIXED_NOW,
    )


def apply_arguments(work_dir: Path, *extra: str) -> tuple[str, ...]:
    return (
        "--work-dir",
        str(work_dir),
        *extra,
        "--apply",
        "--confirm",
        verb.CONFIRMATION,
        "--reference",
        "CHG-121",
    )


def journal_file(root: Path, group: str = GPU_GROUP) -> Path:
    (directory,) = (root / "eks-nodegroup-prepare").iterdir()
    return directory / f"{group}.json"
