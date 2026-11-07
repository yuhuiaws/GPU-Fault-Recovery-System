"""A site whose eks-managed GPU cluster has no node, as the deploy host reads it.

``gpu-a`` is a HyperPod cluster with one node; ``gpu-eks`` is an eks-managed
cluster whose only GPU managed node group is labelled for the site at
``desiredSize`` ``desired`` with no node joined (the live shape: min 0, max 2),
and no Karpenter CRD. ``ZeroNodeSite`` answers the read-only commands the
bootstrap issues for them and records every command, so a test can assert what
was -- and what was never -- run: the node-key helper itself is not executed
here (``tests/deploy/test_node_action_key_provisioning.py`` owns it), its
invocation is recorded with its environment.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
)
from tests.admin._bootstrap_support import _cluster

SITE = "site-a"
EKS_SELECTOR = f"gpu-fault.io/control-plane={SITE}"
EKS_GROUP = "gpu-p5en-group"
HELPER = "provision-node-action-keys.sh"


def hyperpod_cluster(name: str = "gpu-a") -> ClusterIdentity:
    return replace(
        _cluster(), role="gpu", hyperpod_name=name, eks_name=name, context=name
    )


def eks_cluster(name: str = "gpu-eks") -> ClusterIdentity:
    return replace(
        _cluster(),
        role="gpu",
        provider="eks-managed",
        hyperpod_name="",
        hyperpod_arn="",
        eks_name=name,
        eks_arn=f"arn:aws:eks:us-east-1:123456789012:cluster/{name}",
        input_arn=f"arn:aws:eks:us-east-1:123456789012:cluster/{name}",
        context=name,
        node_selector=EKS_SELECTOR,
    )


class ZeroNodeSite(CommandRunner):
    """The two GPU clusters' reads; ``gpu-eks`` lists no node.

    ``unreadable``: the ``gpu-eks`` node list cannot be read at all.
    ``helper_error``: what the node-key helper fails with for a cluster id.
    """

    def __init__(
        self,
        *,
        desired: int = 0,
        unreadable: bool = False,
        helper_error: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.desired = desired
        self.unreadable = unreadable
        self.helper_error = dict(helper_error or {})
        self.commands: list[tuple[list[str], bool, dict[str, str]]] = []
        self._lock = threading.Lock()

    # The calls the tests read back.
    def helper_runs(self) -> list[str]:
        """The cluster ids the node-key helper was run for, in order."""

        return [
            env.get("GPU_FAULT_CLUSTER_ID", "")
            for command, _mutate, env in self.commands
            if Path(command[0]).name == HELPER
        ]

    def commands_for(self, context: str) -> list[list[str]]:
        return [
            command
            for command, _mutate, _env in self.commands
            if "--context" in command
            and command[command.index("--context") + 1] == context
        ]

    def mutating(self) -> list[list[str]]:
        return [command for command, mutate, _env in self.commands if mutate]

    def run(
        self,
        arguments: Sequence[str],
        *,
        input_text: str | None = None,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        capture: bool = True,
        sensitive: bool = False,
        mutate: bool = False,
        timeout_seconds: float | None = None,
    ) -> str:
        command = [str(item) for item in arguments]
        with self._lock:
            self.commands.append((command, mutate, dict(env or {})))
        if Path(command[0]).name == HELPER:
            cluster_id = str((env or {}).get("GPU_FAULT_CLUSTER_ID") or "")
            if cluster_id in self.helper_error:
                raise BootstrapError(
                    f"command failed (1): {HELPER}: ERROR: "
                    + self.helper_error[cluster_id]
                )
            return ""
        context = (
            command[command.index("--context") + 1] if "--context" in command else ""
        )
        if command[0] == "kubectl" and "nodes" in command:
            return self._nodes(command, context)
        if command[0] == "kubectl" and "nodepools.karpenter.sh" in command:
            raise BootstrapError(
                "command failed (1): kubectl: error: the server doesn't have a "
                'resource type "nodepools"'
            )
        if command[0] == "kubectl" and "secret" in command:
            return ""  # no key Secret yet: a populated cluster's probe asks ensure
        if command[:3] == ["aws", "eks", "list-nodegroups"]:
            return json.dumps({"nodegroups": [EKS_GROUP]})
        if command[:3] == ["aws", "eks", "describe-nodegroup"]:
            return json.dumps(
                {
                    "nodegroupName": EKS_GROUP,
                    "status": "ACTIVE",
                    "instanceTypes": ["p5en.48xlarge"],
                    "labels": {
                        "gpu-fault.io/control-plane": SITE,
                        "nvidia.com/gpu.present": "true",
                    },
                    "scalingConfig": {
                        "minSize": 0,
                        "maxSize": 2,
                        "desiredSize": self.desired,
                    },
                }
            )
        raise AssertionError(f"unexpected command: {command[:5]}")

    def _nodes(self, command: list[str], context: str) -> str:
        if context == "gpu-eks" and self.unreadable:
            raise BootstrapError(
                "command failed (1): kubectl: Unable to connect to the server"
            )
        names = [] if context == "gpu-eks" else ["node-a"]
        if "json" in command:
            return json.dumps(
                {
                    "apiVersion": "v1",
                    "kind": "List",
                    "items": [_node(name) for name in names],
                }
            )
        return "\n".join(names)  # the probe's jsonpath projection


def _node(name: str) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Node",
        "metadata": {
            "name": name,
            "uid": f"uid-{name}",
            "labels": {"sagemaker.amazonaws.com/cluster-name": "gpu-a"},
        },
    }
