"""A HyperPod GPU cluster whose nodes leave and join while a release rolls it.

The parties a mid-release departure or join plays between are kept real, as in
``test_release_node_config_waves``: ``FleetRegistry`` on the in-memory store
decides READY from signed heartbeats, ``NodeInstallerReconciler`` decides what to
install from the environment the release hands its script, and the release
engine drives both through ``roll_node_runtime`` -- the real wave loop, wave
safety gate, convergence wait and departure proof. The fleet probe's
``exclude-departed`` operation runs the shipped probe program against the same
registry.

Simulated: the Kubernetes Node API, the installer Job (it completes on the
reconciler's next pass unless the node is ``stuck``), HyperPod's node inventory
(``aws sagemaker list-cluster-nodes``), the read-only control-plane probes
(answered from the registry on the site clock, the clock their leases were
issued against) and time -- one clock the engine's waits sleep on and every live
Agent heartbeats by, so a reclaimed node's lease lapses while the release waits.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import runpy
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from gpu_fault.app import ApplicationContext
from gpu_fault.fleet import (
    CURRENT_AGENT_PROTOCOL_VERSION,
    AgentHeartbeat,
    FleetCompatibilityPolicy,
    FleetDeploymentRequest,
    FleetRegistry,
    SignedAgentHeartbeat,
    sign_agent_heartbeat,
)
from gpu_fault.models import WorkflowOperation
from gpu_fault.node_installer_reconciler import (
    INSTALLER_ARTIFACT_ANNOTATION,
    INSTALLER_BUNDLE_ANNOTATION,
    INSTALLER_DIGEST_ANNOTATION,
    INSTALLER_NODE_UID_ANNOTATION,
    INSTALLER_STATE_ANNOTATION,
    INSTALLER_TEMPLATE_ANNOTATION,
    INSTALLER_VERSION_ANNOTATION,
    NodeInstallerReconciler,
)
from gpu_fault_release.regional_release_probes import probe_source
from tests._builders import build_store
from tests.node_agent.test_node_installer_reconciler import template as job_template
from tests.regional.test_release_node_config_waves import (
    AGENT_VERSION,
    INSTALLER_VERSION,
    NOW,
    POLICY,
    PROFILE,
    SECRET,
    ZONE_LABEL,
    InstallerJobs,
    KubernetesNodes,
    script_version,
)
from tests.regional.test_release_node_image_targets import ARTIFACT, BUNDLE, CONFIG

ROOT = Path(__file__).resolve().parents[2]
FLEET_PROBE = ROOT / "deploy/control-plane/regional/probes/fleet_deployment_command.py"
HYPERPOD_CLUSTER = "hp-gpu-a"
HYPERPOD_LABEL = "sagemaker.amazonaws.com/cluster-name"
#: What the nodes ran before this release: the waves re-install every one.
OLD_TEMPLATE = "9" * 64
#: The probes the site answers, by the source the engine ships.
ANSWERED_PROBES = (
    "agent_lifecycle_facts",
    "agent_convergence",
    "rollout_wave_safety",
    "active_agent_node_sets",
)


def node_name(index: int) -> str:
    """A HyperPod node name: it carries its EC2 instance id."""

    return f"hyperpod-i-{index:012x}"


def instance_of(name: str) -> str:
    return name.removeprefix("hyperpod-")


@dataclass
class SiteClock:
    """The one clock: monotonic seconds for the engine, UTC for the leases."""

    site: ChurnSite
    elapsed: float = 0.0
    sleeps: list[float] = field(default_factory=list)

    def monotonic(self) -> float:
        return self.elapsed

    def time(self) -> float:
        return 2_000_000_000 + self.elapsed

    def sleep(self, seconds: float) -> None:
        seconds = max(0.0, float(seconds))
        self.sleeps.append(seconds)
        self.elapsed += seconds
        self.site.tick(seconds)


class NodeView:
    """The Node API the reconciler sees: ``held`` nodes it has not reached."""

    def __init__(self, site: ChurnSite) -> None:
        self.site = site

    def list_node(self, *, label_selector: str, _request_timeout: Any = None) -> Any:
        listed = self.site.nodes.list_node(label_selector=label_selector)
        return {
            "items": [
                item
                for item in listed["items"]
                if item["metadata"]["name"] not in self.site.held
            ]
        }

    def patch_node(self, name: str, body: dict[str, Any], _request_timeout: Any = None):
        if name in self.site.nodes.items:
            self.site.nodes.patch_node(name, body)

    def list_namespaced_pod(self, namespace: str, **options: Any) -> Any:
        return self.site.nodes.list_namespaced_pod(namespace, **options)


class ChurnSite:
    """One HyperPod cluster, its installer, its Agents and its provider."""

    def __init__(self, target: Any, names: list[str], *, template: str) -> None:
        self.target = target
        self.clock = NOW
        self.registry = FleetRegistry(
            build_store(), SECRET, FleetCompatibilityPolicy(), now=lambda: self.clock
        )
        self.nodes = KubernetesNodes({})
        self.batch = InstallerJobs()
        self.job_identities: dict[str, str] = {}
        #: Instance ids ``list-cluster-nodes`` still returns.
        self.listed: set[str] = set()
        #: Node -> the installer template its live Agent heartbeats with.
        self.agents: dict[str, str] = {}
        #: Nodes whose Agent stopped (the instance is gone).
        self.silent: set[str] = set()
        #: Nodes the running reconciler has not reached yet.
        self.held: set[str] = set()
        #: Nodes whose installer Job never finishes.
        self.stuck: set[str] = set()
        self.reconciler: NodeInstallerReconciler | None = None
        self.deploys: list[SimpleNamespace] = []
        self.created: list[Any] = []
        self.operations: list[str] = []
        #: ``(event, callback)``: run once when ``event`` first happens.
        self.hooks: list[tuple[str, Any]] = []
        self.events: dict[str, int] = {}
        for index, name in enumerate(names):
            self.join(name, zone=f"zone-{'ab'[index % 2]}", installed=template)

    # -- the provider and the nodes -----------------------------------------

    def join(self, name: str, *, zone: str = "zone-a", installed: str | None) -> None:
        """A node enters the cluster's selector; ``installed`` is the template
        an earlier reconciler left on it, None for a brand-new node."""

        index = len(self.nodes.items) + 1
        self.nodes.items[name] = {
            "metadata": {
                "name": name,
                "uid": f"uid-{name}",
                "creationTimestamp": self.clock.isoformat(),
                "labels": {
                    HYPERPOD_LABEL: HYPERPOD_CLUSTER,
                    ZONE_LABEL: zone,
                    "node.kubernetes.io/instance-type": "p5en.48xlarge",
                },
                "annotations": {},
            },
            "spec": {"providerID": f"aws:///us-west-2a/{instance_of(name)}"},
            "status": {
                "conditions": [{"type": "Ready", "status": "True"}],
                "addresses": [{"type": "InternalIP", "address": f"10.0.1.{index}"}],
            },
        }
        self.listed.add(instance_of(name))
        if installed is not None:
            self.install(name, template=installed)

    def install(self, name: str, *, template: str) -> None:
        """What a finished installer Job of ``template`` leaves on ``name``."""

        self.nodes.items[name]["metadata"]["annotations"].update(
            {
                INSTALLER_VERSION_ANNOTATION: INSTALLER_VERSION,
                INSTALLER_DIGEST_ANNOTATION: CONFIG,
                INSTALLER_ARTIFACT_ANNOTATION: ARTIFACT,
                INSTALLER_BUNDLE_ANNOTATION: BUNDLE,
                INSTALLER_TEMPLATE_ANNOTATION: template,
                INSTALLER_NODE_UID_ANNOTATION: f"uid-{name}",
                INSTALLER_STATE_ANNOTATION: "Succeeded",
            }
        )
        self.agents[name] = template
        self.beat(name)

    def reclaim(self, name: str) -> None:
        """The provider takes the instance back: Node, listing and Agent go."""

        self.nodes.items.pop(name, None)
        self.listed.discard(instance_of(name))
        self.silent.add(name)

    def beat(self, name: str) -> None:
        beat = AgentHeartbeat(
            cluster_id=self.target.cluster_id,
            node_id=name,
            endpoint=f"http://{name}:9099",
            agent_protocol_version=CURRENT_AGENT_PROTOCOL_VERSION,
            agent_version=AGENT_VERSION,
            artifact_sha256=ARTIFACT,
            compatibility_digest=ARTIFACT,
            installer_bundle_sha256=BUNDLE,
            installer_template_sha256=self.agents[name],
            policy_version=POLICY,
            runtime_profile_version=PROFILE,
            config_digest=CONFIG,
            allowed_operations=[WorkflowOperation.VERIFY_NO_GPU_CLIENTS],
            boot_id=f"boot-{name}",
            node_instance_id=instance_of(name),
            agent_incarnation_id=f"boot-{name}",
            observed_at=self.clock,
        )
        self.registry.register(
            SignedAgentHeartbeat(
                heartbeat=beat, signature=sign_agent_heartbeat(beat, SECRET)
            )
        )

    def tick(self, seconds: float) -> None:
        """Time passes: the reconciler takes a pass, live Agents heartbeat."""

        self.clock += timedelta(seconds=seconds)
        self.fire("tick")
        if self.reconciler is not None:
            self.reconcile_pass()
        for name in sorted(self.agents):
            if name not in self.silent and name in self.nodes.items:
                self.beat(name)

    def fire(self, event: str) -> None:
        """Run the hooks of ``event`` and of its numbered occurrence
        (``next-wave#2`` is the second ``next-wave``), each once."""

        self.events[event] = self.events.get(event, 0) + 1
        names = {event, f"{event}#{self.events[event]}"}
        for index, (name, callback) in enumerate(list(self.hooks)):
            if name in names and callback is not None:
                self.hooks[index] = (name, None)
                callback()

    # -- the reconciler the release deploys -----------------------------------

    def deploy(self, environment: dict[str, str]) -> None:
        allowed = environment["GPU_FAULT_INSTALLER_ALLOWED_NODES"]
        self.reconciler = NodeInstallerReconciler(
            NodeView(self),
            self.batch,
            namespace=environment["GPU_FAULT_NAMESPACE"],
            cluster_name=environment["GPU_FAULT_HYPERPOD_CLUSTER"],
            node_selector=environment.get("GPU_FAULT_NODE_SELECTOR") or None,
            version=script_version(environment),
            config_digest=environment["GPU_FAULT_INSTALLER_CONFIG_DIGEST"],
            artifact_sha256=environment["GPU_FAULT_INSTALLER_ARTIFACT_SHA256"],
            bundle_sha256=environment["GPU_FAULT_INSTALLER_BUNDLE_SHA256"],
            template_sha256=environment["GPU_FAULT_INSTALLER_TEMPLATE_SHA256"],
            job_template=job_template(),
            dcgm_metrics_url_template="http://{node_ip}:9400/metrics",
            max_unavailable=int(environment["GPU_FAULT_INSTALLER_MAX_UNAVAILABLE"]),
            allowed_node_names=(
                None
                if allowed == "*"
                else frozenset(name for name in allowed.split(",") if name)
            ),
            now=lambda: self.clock,
            agent_alive=lambda *_args: True,
        )
        self.deploys.append(SimpleNamespace(allowed=allowed))
        self.fire(f"deploy:{allowed}")
        for _ in range(16):
            if not self.reconcile_pass():
                return
        raise AssertionError("the simulated reconciler never went idle")

    def reconcile_pass(self) -> bool:
        """One reconciler pass, then every running Job but the stuck ones ends."""

        assert self.reconciler is not None, "a pass needs a deployed reconciler"
        before = set(self.batch.jobs)
        self.reconciler.reconcile_once()
        for name in set(self.batch.jobs) - before:
            self.job_identities[name] = self.reconciler.template_sha256
        running = [
            name
            for name, item in self.batch.jobs.items()
            if not item["done"] and item["node"] not in self.stuck
        ]
        for name in running:
            job = self.batch.jobs[name]
            job["done"] = True
            if job["node"] in self.nodes.items:
                self.agents[job["node"]] = self.job_identities[name]
                self.beat(job["node"])
        return bool(running) or bool(set(self.batch.jobs) - before)

    # -- the control plane ------------------------------------------------

    def fleet_command(self, operation: str, payload: dict[str, Any]) -> Any:
        self.operations.append(operation)
        if operation == "create":
            request = dict(payload["request"])
            request.setdefault("desired_agent_version", AGENT_VERSION)
            request.setdefault("desired_policy_version", POLICY)
            deployment = self.registry.create_deployment(
                FleetDeploymentRequest.model_validate(request)
            )
            self.created.append(deployment)
            return json.loads(deployment.model_dump_json())
        if operation == "get":
            self.fire("get")
            return json.loads(
                self.registry.store.get_fleet_deployment(
                    payload["deployment_id"]
                ).model_dump_json()
            )
        if operation == "next-wave":
            self.fire("next-wave")
            return json.loads(
                self.registry.start_next_wave(
                    payload["deployment_id"]
                ).model_dump_json()
            )
        if operation == "normalize-records":
            return {"normalized": 0}
        if operation == "terminalize-cluster-rollouts":
            return {"terminalized": []}
        return self.run_fleet_probe({"operation": operation, **payload})

    def run_fleet_probe(self, payload: dict[str, Any]) -> Any:
        """The shipped ``fleet_deployment_command`` program, on this registry."""

        context = SimpleNamespace(
            fleet_registry=self.registry, store=self.registry.store
        )
        output = io.StringIO()
        with (
            mock.patch.object(
                ApplicationContext, "from_environment", staticmethod(lambda: context)
            ),
            mock.patch.object(sys, "stdin", io.StringIO(json.dumps(payload))),
            mock.patch.object(sys, "argv", [str(FLEET_PROBE)]),
            contextlib.redirect_stdout(output),
        ):
            try:
                runpy.run_path(str(FLEET_PROBE), run_name="__main__")
            except SystemExit as exit_:
                if exit_.code not in (0, None):
                    raise
        return json.loads(output.getvalue())

    def deployment(self) -> Any:
        (created,) = self.created
        return self.registry.store.get_fleet_deployment(created.deployment_id)

    def answer(self, script: str, request: dict[str, Any] | None) -> str:
        """The read-only probes, answered from the registry on the site clock."""

        name = next(
            (item for item in ANSWERED_PROBES if probe_source(item) == script), None
        )
        if name is None:
            raise AssertionError("the site does not answer this probe")
        agents = {
            item.node_id: item
            for item in self.registry.store.list_agents(self.target.cluster_id)
        }

        def live(item: Any) -> bool:
            return (
                str(getattr(item.lifecycle_state, "value", item.lifecycle_state))
                == "ACTIVE"
                and item.lease_expires_at is not None
                and item.lease_expires_at > self.clock
            )

        if name == "active_agent_node_sets":
            return json.dumps(
                {
                    self.target.cluster_id: sorted(
                        node for node, item in agents.items() if live(item)
                    )
                }
            )
        assert request is not None, f"{name} reads a request"
        if name == "agent_lifecycle_facts":
            wanted = set(request["node_ids"])
            return json.dumps(
                {
                    "read_at": self.clock.isoformat(),
                    "agents": {
                        node: {
                            "lifecycle_state": str(
                                getattr(
                                    item.lifecycle_state, "value", item.lifecycle_state
                                )
                            ),
                            "lease_expires_at": item.lease_expires_at.isoformat()
                            if item.lease_expires_at
                            else None,
                            "last_seen_at": item.last_seen_at.isoformat(),
                            "lease_live": live(item),
                            "node_instance_id": item.node_instance_id,
                        }
                        for node, item in agents.items()
                        if node in wanted
                    },
                }
            )
        if name == "agent_convergence":
            names = set(request["node_names"])
            active = [
                item
                for node, item in agents.items()
                if live(item) and (not names or node in names)
            ]
            aligned = [
                item
                for item in active
                if item.artifact_sha256 == request["artifact"]
                and item.config_digest == request["config"]
                and item.runtime_profile_version == request["profile"]
                and (
                    request["bundle"] is None
                    or item.installer_bundle_sha256 == request["bundle"]
                )
                and (
                    request["template"] is None
                    or item.installer_template_sha256 == request["template"]
                )
            ]
            return json.dumps({"active": len(active), "aligned": len(aligned)})
        # rollout_wave_safety: the Agents outside the wave, as the probe judges.
        required = set(request["nodes"]) - set(request["wave"])
        blockers: list[dict[str, Any]] = []
        remaining: list[float] = []
        for node in sorted(required):
            item = agents.get(node)
            if item is None:
                blockers.append({"node_id": node, "reason": "missing"})
                continue
            lease = (
                None
                if item.lease_expires_at is None
                else (item.lease_expires_at - self.clock).total_seconds()
            )
            if lease is not None:
                remaining.append(lease)
            if lease is None or lease < request["minimum_lease_remaining_seconds"]:
                blockers.append(
                    {
                        "node_id": node,
                        "reason": "lease-margin",
                        "last_seen_at": item.last_seen_at.isoformat(),
                        "lease_remaining_seconds": lease,
                    }
                )
        return json.dumps(
            {
                "open_remote": {},
                "destructive_workflow_count": 0,
                "destructive_workflows": [],
                "never_dispatched_destructive_workflows": [],
                "agent_blocker_count": len(blockers),
                "agent_blockers": blockers,
                "minimum_lease_remaining_seconds": min(remaining)
                if remaining
                else None,
            }
        )

    # -- the Kubernetes reads and the deploy host's commands ------------------

    def get_json(self, arguments: list[str], template_document: Any) -> Any:
        if arguments[-2:] == ["configmap", "previous-template"]:
            return template_document
        if "deployment" in arguments:
            return {"spec": {"replicas": 1}, "status": {"readyReplicas": 1}}
        assert "nodes" in arguments, f"unexpected read {arguments}"
        named: list[str] = []
        tail = iter(arguments[arguments.index("nodes") + 1 :])
        for item in tail:
            if item in {"-l", "-o", "--selector", "--output"}:
                next(tail, None)
            elif not item.startswith("-"):
                named.append(item)
        items = [
            item
            for name, item in sorted(self.nodes.items.items())
            if not named or name in named
        ]
        return {"items": copy.deepcopy(items)}

    def aws(self, arguments: list[str]) -> str:
        assert arguments[:3] == ["aws", "sagemaker", "list-cluster-nodes"], (
            f"unexpected AWS call {arguments[:4]}"
        )
        return json.dumps(
            {
                "ClusterNodeSummaries": [
                    {"InstanceId": instance} for instance in sorted(self.listed)
                ]
            }
        )
