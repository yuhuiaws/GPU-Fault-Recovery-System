from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

from gpu_fault.admin import uninstall as native
from gpu_fault.admin.bootstrap_common import CommandRunner
from scripts.e2e.regional import boot032_contract as contract
from scripts.e2e.regional.boot032_journal import CLEANUP
from tests.admin.test_uninstall_lifecycle import Harness


def cleanup_document(world, path, config_path):
    inventory = path.parent / "fixture-inventory.json"
    inventory.write_text(json.dumps(world.inventory[world.target.metadata_name]))
    value = CLEANUP.initialize(
        path,
        config_path=config_path,
        inventory_path=inventory,
        scope="all",
        mode="reset",
        node_mode="uninstall",
    )
    value.update(
        cluster_uids={}, namespace_snapshots={}, node_targets={}, node_cleanup={}
    )
    CLEANUP.attach_fleet_snapshot(
        value,
        [
            copy.deepcopy(world.agents[(world.target.metadata_name, spec["context"])])
            for spec in contract.cluster_specs(world.target)[1:]
        ],
    )
    namespace = world.target.release_config["namespace"]
    for spec in contract.cluster_specs(world.target):
        key = "cpu" if spec["plane"] == "cpu" else "gpu:" + spec["context"]
        uid_key = (world.target.metadata_name, spec["context"])
        value["cluster_uids"][key] = world.uids[
            (*uid_key, "namespace", "kube-system", "")
        ]
        value["namespace_snapshots"][key] = {
            "uid": world.uids[(*uid_key, "namespace", namespace, "")],
            "objects": [
                [
                    "apps/v1",
                    "deployment",
                    "gpu-fault-api",
                    world.uids[(*uid_key, "deployment", "gpu-fault-api", namespace)],
                ]
            ],
        }
        if spec["plane"] == "gpu":
            value["node_targets"][key] = {
                "node-a": world.uids[(*uid_key, "node", "node-a", "")]
            }
            value["node_cleanup"][key] = {
                "name": "fixture-node-cleanup",
                "uid": "fixture-daemonset-uid",
                "status": "REMOVED",
                "manifest": {"metadata": {"name": "fixture-node-cleanup"}},
            }
        world.uids[(*uid_key, "namespace", namespace, "")] = None
        world.uids[(*uid_key, "deployment", "gpu-fault-api", namespace)] = None
    for phase in CLEANUP.required_phases(value):
        CLEANUP.transition(value, phase=phase, status="COMPLETED", message="fixture")
    CLEANUP.transition(
        value, phase="CLEANUP_COMPLETED", status="COMPLETED", message="fixture"
    )
    CLEANUP.atomic_write(path, value)
    return value


class NativeHarness(Harness):
    """Real native orchestrator, fake external cleanup and resource transports."""

    def __init__(self, world, monkeypatch):
        CommandRunner.__init__(self)
        self.world = world
        self.site = world.target
        self.snapshot = world.snapshots[self.site.metadata_name]
        self.existing = {item.resource_key for item in self.snapshot.resources}
        self.events = []
        self.exports = 0
        self.syncs = 0
        self.fail_sync = self.fail_cpu = self.fail_aurora = False
        self.fail_cleanup = None
        self.no_mutations = False
        self.native_reads = []
        self.cleanup_calls = 0
        self.kube_read_error = False
        world.harness = self
        monkeypatch.setattr(native, "ResourceCleaner", lambda _site: self)
        monkeypatch.setattr(native, "fetch_installation_resource_registry", self.export)
        monkeypatch.setattr(native, "sync_installation_resource_snapshot", self.sync)
        monkeypatch.setattr(native, "bounded_command", self.kubectl)

    def prepare_cpu_delete(self, _hyperpod, _eks):
        cpu = contract.cluster_specs(self.site)[0]
        cloud = self.world.cloud[cpu["eks_name"]]
        return {
            "cpu_eks_created_at": cloud["eks"]["cluster"]["createdAt"],
            "cpu_hyperpod_arn": cloud["hp"]["ClusterArn"],
        }

    def delete_cpu_cluster(self, hyperpod, eks):
        try:
            super().delete_cpu_cluster(hyperpod, eks)
        finally:
            cloud = self.world.cloud[contract.cluster_specs(self.site)[0]["eks_name"]]
            cloud["eks"] = cloud["hp"] = None

    def kubectl(self, arguments, **kwargs):
        assert kwargs["timeout_seconds"] == 45, "native readback must remain bounded"
        self.native_reads.append(list(arguments))
        if self.kube_read_error:
            return subprocess.CompletedProcess(arguments, 1, "", "unreadable")
        context = (
            arguments[arguments.index("--context") + 1]
            if "--context" in arguments
            else "cpu"
        )
        if context == "cpu":
            assert "cluster/cpu-eks" in self.existing, (
                "CPU Kubernetes must not be queried after cluster deletion"
            )
        namespace = arguments[arguments.index("-n") + 1] if "-n" in arguments else ""
        index = arguments.index("get")
        kind = arguments[index + 1]
        names = arguments[index + 2 : arguments.index("--ignore-not-found")]
        present = [
            name
            for name in names
            if self.world.uids.get(
                (self.site.metadata_name, context, kind, name, namespace)
            )
        ]
        return subprocess.CompletedProcess(arguments, 0, "\n".join(present), "")

    def run(self, arguments, **kwargs):
        if arguments[-1] == "cleanup-owned":
            self.events.append("cleanup-owned")
            return ""
        config_path = Path(arguments[arguments.index("--config") + 1])
        flag = "--path" if "verify" in arguments else "--state-file"
        path = Path(arguments[arguments.index(flag) + 1])
        if arguments[-1] == "verify-targets":
            value = CLEANUP.read_state(path)
            for context, uid in value["cluster_uids"].items():
                if context == "cpu" and "--skip-cpu" in arguments:
                    continue
                context_name = context.removeprefix("gpu:")
                assert (
                    self.world.uids[
                        (
                            self.site.metadata_name,
                            context_name,
                            "namespace",
                            "kube-system",
                            "",
                        )
                    ]
                    == uid
                ), "native verify-targets must compare original cluster UIDs"
            return ""
        if "verify" not in arguments:
            assert not self.no_mutations, "completed replay cannot run cleanup"
            self.cleanup_calls += 1
            if self.fail_cleanup is not None:
                failure, self.fail_cleanup = self.fail_cleanup, None
                raise failure
            if not path.exists():
                self.events.append("cleanup")
                cleanup_document(self.world, path, config_path)
        value = CLEANUP.read_state(path)
        CLEANUP.validate_request(
            value,
            config_path=config_path,
            scope="all",
            mode="reset",
            node_mode="uninstall",
            cluster_ids=[],
        )
        return ""

    def copy_cleanup(self):
        return copy.deepcopy(
            CLEANUP.read_state(
                self.world.settings.native_dir / "kubernetes-cleanup.json"
            )
        )
