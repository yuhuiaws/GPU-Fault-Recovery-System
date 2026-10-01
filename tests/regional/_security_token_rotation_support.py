from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from gpu_fault.admin.atomic_json import write_json_atomic
from scripts.e2e.regional import auth016_lifecycle as lifecycle
from scripts.e2e.regional import identity_acceptance_auth as auth
from tests.regional._security_consumer_pods import (
    converged_deployment,
    owned_pod_documents,
)


def consumer_snapshot(*, activated: bool, stamp: datetime) -> dict:
    return {
        "cluster_id": "cluster-a",
        "captured_at": (stamp + timedelta(seconds=10)).isoformat(),
        "agents": {
            node: {
                # Live identity model (node_agent/heartbeat.py): the incarnation is
                # sha256(cluster, node, instance, boot_id) -- it survives an agent
                # restart and changes only on reboot. A token-rotation reinstall
                # restarts the agent and re-registers it with the wave identity,
                # which the fleet records as a generation advance.
                "incarnation": "boot-" + node,
                "generation": 2 if activated else 1,
                "lifecycle": "ACTIVE",
                "last_seen_at": (stamp + timedelta(seconds=1)).isoformat(),
                "lease_expires_at": (stamp + timedelta(seconds=120)).isoformat(),
                "required_collectors": ["nvidia-kernel"],
            }
            for node in ("node-a", "node-b")
        },
        "collectors": {
            node + "/nvidia-kernel": {
                "ingested_at": (stamp + timedelta(seconds=2)).isoformat(),
                "last_success_at": (stamp + timedelta(seconds=1)).isoformat(),
                "errors": False,
            }
            for node in ("node-a", "node-b")
        },
        "watcher": {
            "cluster_id": "cluster-a",
            "watcher_instance": "new-watcher",
            "observed_at": (stamp + timedelta(seconds=1)).isoformat(),
        },
    }


class RotationWorld:
    def __init__(self, root: Path, monkeypatch, *, defect="none"):
        self.defect = defect
        self.events = []
        self.phase = "baseline"
        self.rotated = False
        self.old = "old-local-fixture-" + "a" * 40
        self.new = "new-local-fixture-" + "b" * 40
        self.stamp = datetime.now(timezone.utc) - timedelta(seconds=1)
        self.target = SimpleNamespace(
            cluster_id="cluster-a",
            eks_cluster_arn="arn:aws:eks:test-1:111122223333:cluster/gpu",
        )
        self.site_file = root / "site.yaml"
        self.site_file.write_text("unit owned site")
        self.token_file = root / "cluster.token"
        self.token_file.write_text(self.old)
        self.token_file.chmod(0o600)
        self.site = SimpleNamespace(
            site=SimpleNamespace(source=self.site_file, repository_root=root),
            namespace="gpu-fault-system",
            gpu=self.gpu,
            pod_json=self.pod_json,
            any_executor_pod=lambda _target: "executor",
            registry=self.registry,
            regional=lambda _target: self,
        )
        self.state_path = lifecycle.rotation_state_path(self.site.site, "cluster-a")
        self.thread = None
        world = self

        class Event:
            stopped = False
            cycle = True

            def is_set(self):
                return self.stopped or not self.cycle

            def set(self):
                self.stopped = True

            def wait(self, _seconds):
                self.cycle = False

        self.stop = Event()

        class Thread:
            def __init__(self, *, target, daemon):
                self.target = target
                world.thread = self

            def start(self):
                world.tick()

            def join(self, *, timeout):
                world.events.append(("sampler-joined", timeout))

            def is_alive(self):
                return world.defect == "sampler"

        monkeypatch.setattr(lifecycle.threading, "Event", lambda: self.stop)
        monkeypatch.setattr(lifecycle.threading, "Thread", Thread)
        monkeypatch.setattr(lifecycle, "claim", self.claim)
        monkeypatch.setattr(lifecycle, "read_cluster_token", lambda *_args: self.old)
        monkeypatch.setattr(lifecycle, "invoke_rotation", self.invoke)
        monkeypatch.setattr(auth, "executor_claim_identity", lambda *_args: {})
        monkeypatch.setattr(
            auth,
            "direct_claim",
            lambda *_args, **_kwargs: 200 if defect == "completed" else 403,
        )
        if defect == "backlog":
            write_json_atomic(
                self.state_path, {"status": "IN_PROGRESS", "reference": "foreign"}
            )

    def tick(self):
        self.stop.cycle = True
        self.thread.target()

    def evidence_identity(self):
        return {"cluster_id": "cluster-a", "release_id": "unit-release"}

    def registry(self):
        return [
            {
                "cluster_id": "cluster-a",
                "enabled": True,
                "lifecycle_state": "ACTIVE",
                "token_sha256": hashlib.sha256(
                    (self.new if self.rotated else self.old).encode()
                ).hexdigest(),
                "retiring_token_sha256": None,
            },
            {"cluster_id": "peer", "token_sha256": "c" * 64},
        ]

    def gpu(self, _target, *args, **kwargs):
        name = (
            args[args.index("deployment") + 1]
            if "deployment" in args
            else (args[args.index("-l") + 1].removeprefix("app="))
        )
        deployment = converged_deployment(
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": name,
                    "namespace": self.site.namespace,
                    "uid": "deployment-" + name,
                },
                "spec": {
                    "replicas": 1,
                    "template": {
                        "metadata": {},
                        "spec": {
                            "containers": [
                                {"name": "consumer", "image": "unit@sha256:" + "e" * 64}
                            ]
                        },
                    },
                },
            }
        )
        if "deployment" in args:
            return json.dumps(deployment)
        replicasets, pods = owned_pod_documents(
            deployment, "new" if self.rotated else "old"
        )
        return json.dumps(replicasets if "replicasets" in args else pods)

    def pod_json(self, *_args, **_kwargs):
        return {
            "cluster_id": "cluster-a",
            "status": 200,
            "local_scope": True,
            "token_sha256": hashlib.sha256(
                (self.new if self.rotated else self.old).encode()
            ).hexdigest(),
        }

    def cpu_python(self, *_args, **_kwargs):
        return consumer_snapshot(activated=self.rotated, stamp=self.stamp)

    def claim(self, *_args):
        return {"status": 403 if self.defect == self.phase else 200, "command_count": 0}

    def invoke(self, _site, _target, reference):
        self.events.append(("production-rotation", reference))
        self.state = {
            "schema_version": 1,
            "status": "COMPLETED",
            "cluster_id": "cluster-a",
            "reference": reference,
            "old_token_sha256": hashlib.sha256(self.old.encode()).hexdigest(),
            "new_token_sha256": hashlib.sha256(self.new.encode()).hexdigest(),
            "keep_window": False,
            "quiet_seconds": 180,
            "pending_token_cleanup_completed": True,
            "token_file": str(self.token_file),
            "node_rollout": {"completed_nodes": ["node-a", "node-b"]},
            "steps": {
                step: {"completed_at": self.stamp.isoformat(), "evidence": {}}
                for step in lifecycle.ROTATION_STEPS
            },
        }
        self.state["steps"][lifecycle.STEP_DATA_PLANE_ROLLED]["evidence"] = {
            "deployments": list(lifecycle.DEPLOYMENTS)
        }
        self.state["steps"][lifecycle.STEP_CONTROL_PLANE_ROLLED]["evidence"] = {
            "registry_secret_rewritten": True,
            "deployments": list(lifecycle.CPU_RUNTIME_DEPLOYMENTS),
        }
        self.state["steps"][lifecycle.STEP_NODES_ROLLED]["evidence"] = {
            "reinstalled_nodes": ["node-a", "node-b"]
        }
        self.state["steps"][lifecycle.STEP_RETIRING_DROPPED]["evidence"] = {
            "retiring_token_dropped": True
        }
        self.state["steps"][lifecycle.STEP_ACCEPTED]["evidence"] = {
            "quiet_seconds": 180,
            "waited_seconds": 180,
        }
        overlap = copy.deepcopy(self.state)
        overlap["steps"] = {
            lifecycle.STEP_OVERLAP_PUBLISHED: self.state["steps"][
                lifecycle.STEP_OVERLAP_PUBLISHED
            ]
        }
        write_json_atomic(self.state_path, overlap)
        self.phase = "overlap"
        self.tick()
        self.rotated = True
        self.token_file.write_text(self.new)
        write_json_atomic(self.state_path, self.state)
        self.phase = "new-consumers"
        self.tick()
