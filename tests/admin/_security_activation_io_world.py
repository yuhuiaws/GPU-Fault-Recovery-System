"""In-memory Kubernetes transport for the real custody activation adapter."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone

from gpu_fault.admin import node_key_custody_activation_io as module
from gpu_fault.admin.bootstrap_common import ClusterIdentity, CommandRunner
from gpu_fault.admin.node_key_custody_admin_probe import AdminNodeKeyContext
from gpu_fault.admin.node_key_custody_models import (
    Authorization,
    Completed,
    CustodyBinding,
    KeyMap,
    KeyState,
    MasterSource,
    ReleaseBinding,
    SiteBinding,
)
from gpu_fault.fleet import AgentTransitionRequest, FleetRegistry
from gpu_fault.store import InMemoryStore
from tests.regional._cov95_auth015_support import agent
from tests.regional._cov95_ha002_harness import apply_patch as apply_test_patch
from tests.regional._security_consumer_pods import (
    converged_deployment,
    owned_pod_documents,
)


class ActivationWorld(CommandRunner):
    def __init__(self, root, monkeypatch):
        super().__init__()
        self.calls = []
        self.patches = []
        self.projection = "current"
        self.draining = False
        self.store = InMemoryStore()
        for name in ("node-a", "node-b"):
            self.store.save_agent(agent(name))
        self.registry = FleetRegistry(self.store, "test-shared-unused-" + "x" * 32)
        self.namespace = "gpu-fault-system"
        self.records = {}
        self.cpu_arn = "arn:aws:eks:test-1:111122223333:cluster/cpu"
        self.gpu_arn = "arn:aws:eks:test-1:111122223333:cluster/gpu"
        self.key_values = {
            "node-a": "old-fixture-" + "a" * 40,
            "node-b": "peer-fixture-" + "b" * 40,
        }
        self.images = {
            name: "registry.invalid/" + name + "@sha256:" + "f" * 64
            for name in ("runtime", "executor")
        }
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "delivery": {
                        "images": {
                            name: {"reference": image}
                            for name, image in self.images.items()
                        }
                    }
                }
            )
        )
        now = datetime.now(timezone.utc)
        binding = CustodyBinding(
            release=ReleaseBinding(
                release_id="release-a",
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                delivery_sha256="f" * 64,
                node_wheel_sha256="a" * 64,
                node_digest="b" * 64,
                bundle_sha256="c" * 64,
                template_sha256="d" * 64,
                config_digest="e" * 64,
                runtime_profile_version="profile-a",
            ),
            site=SiteBinding(
                site_name="unit-site",
                region="test-1",
                cpu_eks_arn=self.cpu_arn,
                gpu_eks_arn=self.gpu_arn,
                cluster_id="cluster-a",
                hyperpod_cluster="hyperpod-a",
                namespace=self.namespace,
                cpu_cluster_uid="cpu-cluster",
                gpu_cluster_uid="gpu-cluster",
                cpu_namespace_uid="cpu-namespace",
                gpu_namespace_uid="gpu-namespace",
            ),
            nodes={"node-a": "uid-node-a", "node-b": "uid-node-b"},
            master=MasterSource(
                secret_name="master",
                secret_uid="master-uid",
                data_key="master",
                sha256="9" * 64,
            ),
        )
        self.authorization = Authorization(
            transaction_id="1" * 64,
            binding=binding,
            purpose="rotate",
            rotate_node="node-a",
            previous_receipt_sha256="2" * 64,
            producer_sha256="3" * 64,
            witness_sha256="4" * 64,
            not_before=now - timedelta(seconds=1),
            expires_at=now + timedelta(minutes=30),
        )
        cluster = ClusterIdentity(
            input_arn=self.gpu_arn,
            role="gpu",
            region="test-1",
            account_id="111122223333",
            hyperpod_arn="arn:aws:sagemaker:test-1:111122223333:cluster/hyperpod-a",
            hyperpod_name="hyperpod-a",
            eks_arn=self.gpu_arn,
            eks_name="gpu",
            vpc_id="vpc-test",
            subnet_ids=("subnet-test",),
            node_recovery="None",
            context="gpu-context",
        )
        self.context = AdminNodeKeyContext(
            state_dir=root,
            repository_root=root,
            site_id="unit-site",
            cpu_eks_arn=self.cpu_arn,
            cpu_kubeconfig=root / "cpu.kubeconfig",
            gpu_kubeconfig=root / "gpu.kubeconfig",
            namespace=self.namespace,
            cluster=cluster,
            cluster_id="cluster-a",
            release_manifest=manifest,
            runtime_profile_version="profile-a",
            agent_config_digest="e" * 64,
        )
        self.io = module.CustodyActivationIO(self, self.context, self.authorization)
        monkeypatch.setattr(
            self.io, "host_preflight", lambda: self.calls.append(("host-preflight",))
        )
        self.create_resources()

    def create_resources(self):
        binding = self.authorization.binding
        for plane, names in (
            ("cpu", module.CPU_RUNTIME_DEPLOYMENTS),
            ("gpu", (module.GPU_EXECUTOR_DEPLOYMENT, module.GPU_RECONCILER_DEPLOYMENT)),
        ):
            for name in names:
                container = (
                    "api"
                    if plane == "cpu"
                    else (
                        "executor"
                        if name == module.GPU_EXECUTOR_DEPLOYMENT
                        else "reconciler"
                    )
                )
                env = [{"name": "GPU_FAULT_NODE_ACTION_KEYS_DIR", "value": "/etc/keys"}]
                if name == module.GPU_RECONCILER_DEPLOYMENT:
                    env.extend(
                        [
                            {
                                "name": module.INSTALLER_WAVE_CONFIG_MAP_ENV,
                                "value": "wave",
                            },
                            {
                                "name": "GPU_FAULT_INSTALLER_BUNDLE_SHA256",
                                "value": "c" * 64,
                            },
                            {
                                "name": "GPU_FAULT_INSTALLER_TEMPLATE_SHA256",
                                "value": "d" * 64,
                            },
                        ]
                    )
                value = self.document("Deployment", name, plane)
                value["spec"] = {
                    "replicas": 0 if "spool" in name else 1,
                    "template": {
                        "metadata": {"annotations": {}},
                        "spec": {
                            "containers": [
                                {
                                    "name": container,
                                    "image": self.images[
                                        "runtime" if plane == "cpu" else "executor"
                                    ],
                                    "env": env,
                                }
                            ]
                        },
                    },
                }
                self.records[plane, "deployment", name] = converged_deployment(value)
        for node in binding.nodes:
            value = self.document("Node", node, "gpu")
            value["metadata"]["uid"] = binding.nodes[node]
            value["metadata"]["labels"] = {
                "sagemaker.amazonaws.com/cluster-name": "hyperpod-a"
            }
            value["metadata"]["annotations"] = {
                "gpu-fault.io/installer-artifact-sha256": "a" * 64,
                "gpu-fault.io/installer-bundle-sha256": "c" * 64,
                "gpu-fault.io/installer-template-sha256": "d" * 64,
                "gpu-fault.io/installer-config-digest": "e" * 64,
                "gpu-fault.io/installer-node-uid": binding.nodes[node],
                "gpu-fault.io/installer-state": "Succeeded",
            }
            value["spec"] = {}
            value["status"] = {"conditions": [{"type": "Ready", "status": "True"}]}
            self.records["gpu", "node", node] = value
        release = self.document("ConfigMap", "gpu-fault-regional-release-state", "cpu")
        release["data"] = {
            "state.json": json.dumps(
                {
                    "release_id": "release-a",
                    "phase": "complete",
                    "transaction_committed": True,
                    "release_delivery_sha256": "f" * 64,
                    "bundle_sha256": "c" * 64,
                    "agent_config_digest": "e" * 64,
                    "runtime_profile_version": "profile-a",
                    "node_template_sha256": "d" * 64,
                }
            )
        }
        self.records["cpu", "configmap", release["metadata"]["name"]] = release
        wave = self.document("ConfigMap", "wave", "gpu")
        wave["data"] = {
            "allowed-nodes": "*",
            "max-unavailable": "2",
            "generation": "steady",
        }
        self.records["gpu", "configmap", "wave"] = wave
        self.write_keys()

    def document(self, kind, name, plane):
        return {
            "apiVersion": "v1" if kind != "Deployment" else "apps/v1",
            "kind": kind,
            "metadata": {
                "name": name,
                "namespace": self.namespace,
                "uid": plane + "-" + name,
                "resourceVersion": "1",
            },
        }

    def write_keys(self):
        for plane in ("cpu", "gpu"):
            secret = self.document("Secret", "gpu-fault-node-action-keys", plane)
            secret.update(
                {
                    "type": "Opaque",
                    "data": {
                        node: base64.b64encode(value.encode()).decode()
                        for node, value in self.key_values.items()
                    },
                }
            )
            self.records[plane, "secret", "gpu-fault-node-action-keys"] = secret

    def bind_provisioned(self):
        maps = {
            plane: KeyMap(
                uid=self.records[plane, "secret", "gpu-fault-node-action-keys"][
                    "metadata"
                ]["uid"],
                resource_version="1",
                keys={
                    name: KeyState(
                        node_uid=self.authorization.binding.nodes[name],
                        generation=2 if name == "node-a" else 1,
                        sha256=hashlib.sha256(value.encode()).hexdigest(),
                    )
                    for name, value in self.key_values.items()
                },
            )
            for plane in ("cpu", "gpu")
        }
        self.io.bind_completed(
            Completed(
                started_sha256="7" * 64,
                observed_at=datetime.now(timezone.utc),
                cpu=maps["cpu"],
                gpu=maps["gpu"],
                cpu_other_keys_sha256="8" * 64,
            )
        )

    def pods(self, plane, name):
        deployment = self.records[plane, "deployment", name]
        template = deployment["spec"]["template"]
        marker = template["metadata"]["annotations"].get(
            module.ACTIVATION_ANNOTATION, "old"
        )
        return owned_pod_documents(deployment, marker)[1]

    def run(self, command, **kwargs):
        args = list(command)
        self.calls.append(tuple(args))
        assert kwargs.get("sensitive") is True
        plane = "gpu" if "gpu-context" in args else "cpu"
        if "exec" in args:
            script = args[-1]
            request = json.loads(kwargs["input_text"])
            if script == module.AGENTS_PROBE:
                return json.dumps(
                    {
                        row.node_id: row.model_dump(mode="json")
                        for row in self.store.list_agents("cluster-a")
                    }
                )
            if script == module.TRANSITION_PROBE:
                assert kwargs["mutate"] is True
                record = self.registry.drain_agent(
                    "cluster-a",
                    "node-a",
                    AgentTransitionRequest.model_validate(request["payload"]),
                )
                return record.model_dump_json()
            if script == module.PROJECTED_KEY_PROBE:
                started = datetime(2026, 9, 15, tzinfo=timezone.utc).timestamp()
                epoch = int(started) * 1_000_000_000
                file_time = (
                    epoch + (1 if self.projection == "late" else -1) * 1_000_000_000
                )
                return json.dumps(
                    {
                        "sha256": hashlib.sha256(
                            self.key_values["node-a"].encode()
                        ).hexdigest(),
                        "mtime_ns": file_time,
                        "ctime_ns": file_time,
                        "process_start_ticks": 80000,
                        "ticks_per_second": 100,
                        "boottime_ns": 900_000_000_000,
                        "realtime_before_ns": epoch + 100_000_000_000,
                        "realtime_after_ns": epoch + 100_000_001_000,
                    }
                )
            assert script == module.probe_source("rollout_wave_safety")
            return json.dumps(
                {
                    "open_remote": {"PENDING": 0, "LEASED": 0, "WAITING": 0},
                    "destructive_workflow_count": 0,
                    "agent_blocker_count": 0,
                }
            )
        if "rollout" in args:
            if args[-2] == "deployment/" + module.CPU_INGRESS_DEPLOYMENT:
                current = self.store.get_agent("cluster-a", "node-a")
                assert current.lifecycle_state.value == "DRAINING"
                now = datetime.now(timezone.utc)
                self.store.save_agent(
                    current.model_copy(
                        update={
                            "agent_incarnation_id": "new-agent",
                            "generation": current.generation + 1,
                            "lifecycle_state": type(current.lifecycle_state).ACTIVE,
                            "last_seen_at": now,
                            "lease_expires_at": now + timedelta(seconds=120),
                        }
                    )
                )
            return "rolled out"
        operation = "get" if "get" in args else "patch"
        index = args.index(operation)
        kind = args[index + 1]
        if operation == "get" and kind == "namespace":
            name = args[index + 2]
            field = plane + (
                "_cluster_uid" if name == "kube-system" else "_namespace_uid"
            )
            return json.dumps(
                {
                    "apiVersion": "v1",
                    "kind": "Namespace",
                    "metadata": {
                        "name": name,
                        "uid": getattr(self.authorization.binding.site, field),
                    },
                }
            )
        if operation == "get" and kind == "nodes":
            return json.dumps(
                {
                    "apiVersion": "v1",
                    "kind": "NodeList",
                    "items": [
                        value
                        for (scope, resource, _), value in self.records.items()
                        if scope == plane and resource == "node"
                    ],
                }
            )
        if operation == "get" and kind == "jobs":
            return json.dumps({"items": []})
        if operation == "get" and kind == "pods":
            return json.dumps(
                self.pods(plane, args[args.index("-l") + 1].removeprefix("app="))
            )
        if operation == "get" and kind == "replicasets":
            name = args[args.index("-l") + 1].removeprefix("app=")
            value = self.records[plane, "deployment", name]
            marker = value["spec"]["template"]["metadata"]["annotations"].get(
                module.ACTIVATION_ANNOTATION, "old"
            )
            return json.dumps(owned_pod_documents(value, marker)[0])
        name = args[index + 2]
        if operation == "get" and kind == "pod":
            for scope, resource, deployment_name in self.records:
                if scope == plane and resource == "deployment":
                    for pod in self.pods(plane, deployment_name)["items"]:
                        if pod["metadata"]["name"] == name:
                            return json.dumps(pod)
            raise RuntimeError("unit Pod no longer exists")
        key = plane, kind, name
        if operation == "patch":
            assert kwargs["mutate"] is True
            patch = json.loads(kwargs["input_text"])
            self.records[key] = apply_test_patch(self.records[key], patch)
            self.records[key]["metadata"]["resourceVersion"] = str(
                int(self.records[key]["metadata"]["resourceVersion"]) + 1
            )
            self.patches.append((plane, kind, name))
            if kind == "node":
                annotations = self.records[key]["metadata"]["annotations"]
                annotations[module.INSTALLER_ACTIVATED_ANNOTATION] = annotations[
                    module.INSTALLER_ACTIVATION_ANNOTATION
                ]
            return ""
        return json.dumps(copy.deepcopy(self.records[key]))
