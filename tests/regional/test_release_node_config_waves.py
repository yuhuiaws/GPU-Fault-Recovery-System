"""A node-configuration-only release rolls through the fleet's Agent waves.

The fleet deployment calls a node READY when its Agent's heartbeat reports the
release's nine-field identity (protocol, version, artifact, compatibility,
bundle, template, policy, Runtime Profile, config digest). The node installer
reconciler re-installs a node when *its* node identity differs, and that one is
wider: it also holds the DCGM host engine mode of a plain EKS cluster
(``gpu-fault.io/installer-dcgm-hostengine``), which no heartbeat field carried.
A release that changed only the mode therefore created a deployment that was
SUCCEEDED before its first wave, skipped every wave and handed the steady-state
reconciler ``*``: it re-installed every Agent of the cluster at the policy's
``max_unavailable`` -- no one-node canary, no wave safety gate, no
failure-domain cap -- inside the one steady-state convergence wait.

The harness keeps the three parties real. ``FleetRegistry`` on the in-memory
store decides READY from signed heartbeats; ``NodeInstallerReconciler`` decides
which node to re-install from the very environment the release hands
``deploy-node-installer-reconciler.sh``; ``roll_node_runtime`` drives both. Only
the Kubernetes API and the installer Job are simulated: a Job completes on the
next step and restarts its node's Agent with the identity of the reconciler that
created it, which is what ``install-gpu-fault-collector.sh`` writes into the
Agent environment.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.cluster_scope import DCGM_HOSTENGINE_ENV, selector_matches
from gpu_fault.fleet import (
    CURRENT_AGENT_PROTOCOL_VERSION,
    AgentHeartbeat,
    DeploymentStatus,
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
from gpu_fault.node_installer_rendering import INSTALLER_DCGM_HOSTENGINE_ANNOTATION
from gpu_fault_release import regional_release_fleet_rollout as FLEET
from gpu_fault_release import regional_release_node_runtime_rollout as RUNTIME
from gpu_fault_release.regional_release_config import (
    ClusterLocalReleaseError,
    ReleaseError,
)
from gpu_fault_release.regional_release_diff import ReleaseComponent
from gpu_fault_release.regional_release_gpu_rollout import agents_converged
from gpu_fault_release.regional_release_rollback_target import rollback_target
from tests._builders import build_store
from tests.node_agent.test_node_installer_reconciler import template as job_template
from tests.regional import test_release_node_image_targets as IMAGE_TARGETS
from tests.regional.test_release_node_image_targets import (
    ARTIFACT,
    BUNDLE,
    CONFIG,
    INSTALLER,
    PREVIOUS_EXECUTOR,
    TEMPLATE,
)

#: The node image target scenario (release double, reconciler sink), reused.
node_release = IMAGE_TARGETS.node_release

SECRET = "fleet-secret-" + "x" * 32
NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
AGENT_VERSION = "0.10.0"
POLICY = "test-policy"
PROFILE = "candidate-profile"
INSTALLER_VERSION = "0.10.0"
#: The bundle the release rolls, and the one key its ConfigMap holds.
BUNDLE_FILE = f"gpu-fault-node-installer-{INSTALLER_VERSION}.tar.gz"
#: ``deploy-node-installer-reconciler.sh``'s own ``GPU_FAULT_VERSION`` default.
SCRIPT_DEFAULT_VERSION = "0.10.0"
EKS_SELECTOR = "gpu-fault.io/control-plane=site-a"
ZONE_LABEL = "topology.kubernetes.io/zone"
#: Two failure domains of two nodes: the auto policy allows four at once, one
#: canary first and at most two per domain.
NODES = {"node-a": "zone-a", "node-b": "zone-a", "node-c": "zone-b", "node-d": "zone-b"}
PER_DOMAIN_CAP = 2


class NotFound(Exception):
    status = 404


def script_version(environment: dict[str, str]) -> str:
    """The installer version the reconciler script renders with, and checks the
    bundle ConfigMap for (``gpu-fault-node-installer-<version>.tar.gz``)."""

    version = environment.get("GPU_FAULT_VERSION") or SCRIPT_DEFAULT_VERSION
    bundle = environment["GPU_FAULT_INSTALLER_CONFIG_MAP"]
    if f"gpu-fault-node-installer-{version}.tar.gz" != BUNDLE_FILE:
        # What the script exits with before it renders or applies anything.
        raise ReleaseError(
            f"installer ConfigMap {bundle} is missing the candidate bundle "
            f"gpu-fault-node-installer-{version}.tar.gz"
        )
    return version


class KubernetesNodes:
    """The Node API both the release engine and the reconciler read."""

    def __init__(self, items: dict[str, dict[str, Any]]) -> None:
        self.items = items

    def list_node(self, *, label_selector: str, _request_timeout: Any = None) -> Any:
        return {
            "items": [
                item
                for item in self.items.values()
                if selector_matches(item["metadata"]["labels"], label_selector)
            ]
        }

    def patch_node(self, name: str, body: dict[str, Any], _request_timeout: Any = None):
        annotations = self.items[name]["metadata"]["annotations"]
        for key, value in body["metadata"]["annotations"].items():
            if value is None:
                annotations.pop(key, None)
            else:
                annotations[key] = value

    def list_namespaced_pod(
        self, namespace: str, *, label_selector: str, _request_timeout: Any = None
    ) -> Any:
        return {"items": []}


class InstallerJobs:
    """Installer Jobs: created by the reconciler, completed by ``complete``."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}

    def _job(self, name: str) -> dict[str, Any]:
        item = self.jobs[name]
        conditions = [{"type": "Complete", "status": "True"}] if item["done"] else []
        return {"metadata": {"name": name}, "status": {"conditions": conditions}}

    def read_namespaced_job(
        self, name: str, namespace: str, _request_timeout: Any = None
    ) -> Any:
        if name not in self.jobs:
            raise NotFound()
        return self._job(name)

    def list_namespaced_job(
        self, namespace: str, *, label_selector: str, _request_timeout: Any = None
    ) -> Any:
        return {"items": [self._job(name) for name in self.jobs]}

    def create_namespaced_job(
        self, namespace: str, body: dict[str, Any], _request_timeout: Any = None
    ) -> None:
        self.jobs[body["metadata"]["name"]] = {
            "node": body["spec"]["template"]["spec"]["nodeName"],
            "done": False,
        }

    def delete_namespaced_job(self, name: str, namespace: str, **_options: Any) -> None:
        self.jobs.pop(name, None)


class Site:
    """One GPU cluster: its Nodes, its installer Jobs, its Agents' heartbeats."""

    def __init__(self, target: Any, *, template: str, mode: str) -> None:
        self.target = target
        self.clock = NOW
        self.registry = FleetRegistry(
            build_store(), SECRET, FleetCompatibilityPolicy(), now=lambda: self.clock
        )
        labels = (
            {"gpu-fault.io/control-plane": "site-a"}
            if getattr(target, "provider", None) == "eks-managed"
            else {"sagemaker.amazonaws.com/cluster-name": target.hyperpod_cluster_name}
        )
        self.nodes = KubernetesNodes(
            {
                name: {
                    "metadata": {
                        "name": name,
                        "uid": f"uid-{name}",
                        "labels": {
                            **labels,
                            ZONE_LABEL: zone,
                            "node.kubernetes.io/instance-type": "p5en.48xlarge",
                        },
                        "annotations": {},
                    },
                    "spec": {"providerID": f"aws:///us-west-2a/i-{index:017d}"},
                    "status": {
                        "conditions": [{"type": "Ready", "status": "True"}],
                        "addresses": [
                            {"type": "InternalIP", "address": f"10.0.0.{index}"}
                        ],
                    },
                }
                for index, (name, zone) in enumerate(sorted(NODES.items()), start=1)
            }
        )
        self.batch = InstallerJobs()
        #: The (template, mode) each Job's installer restarts the Agent with.
        self.job_identities: dict[str, tuple[str, str]] = {}
        self.gates: list[tuple[str, ...]] = []
        self.deploys: list[SimpleNamespace] = []
        self.installs: list[SimpleNamespace] = []
        self.created: list[Any] = []
        self.waits: list[SimpleNamespace] = []
        for name in NODES:
            self.install(name, template=template, mode=mode)

    def reset(self) -> None:
        """Forget what one rollout did; the nodes and Agents stay as it left them."""

        for record in (
            self.gates,
            self.deploys,
            self.installs,
            self.created,
            self.waits,
        ):
            record.clear()

    # -- the installer ------------------------------------------------------

    def install(self, name: str, *, template: str, mode: str) -> None:
        """What a finished installer Job leaves behind on ``name``."""

        annotations = self.nodes.items[name]["metadata"]["annotations"]
        annotations.update(
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
        if mode == "host":
            annotations[INSTALLER_DCGM_HOSTENGINE_ANNOTATION] = "host"
        else:
            annotations.pop(INSTALLER_DCGM_HOSTENGINE_ANNOTATION, None)
        self.heartbeat(name, template=template)

    def heartbeat(self, name: str, *, template: str) -> None:
        self.clock += timedelta(seconds=1)
        beat = AgentHeartbeat(
            cluster_id=self.target.cluster_id,
            node_id=name,
            endpoint=f"http://{name}:9099",
            agent_protocol_version=CURRENT_AGENT_PROTOCOL_VERSION,
            agent_version=AGENT_VERSION,
            artifact_sha256=ARTIFACT,
            compatibility_digest=ARTIFACT,
            installer_bundle_sha256=BUNDLE,
            installer_template_sha256=template,
            policy_version=POLICY,
            runtime_profile_version=PROFILE,
            config_digest=CONFIG,
            allowed_operations=[WorkflowOperation.VERIFY_NO_GPU_CLIENTS],
            boot_id=f"boot-{name}",
            node_instance_id=f"instance-{name}",
            agent_incarnation_id=f"boot-{name}",
            observed_at=self.clock,
        )
        self.registry.register(
            SignedAgentHeartbeat(
                heartbeat=beat, signature=sign_agent_heartbeat(beat, SECRET)
            )
        )

    # -- the reconciler the release deploys --------------------------------

    def deploy(self, environment: dict[str, str]) -> None:
        """Run the reconciler ``environment`` describes until it is idle."""

        allowed = environment["GPU_FAULT_INSTALLER_ALLOWED_NODES"]
        version = script_version(environment)
        reconciler = NodeInstallerReconciler(
            self.nodes,
            self.batch,
            namespace=environment["GPU_FAULT_NAMESPACE"],
            cluster_name=environment["GPU_FAULT_HYPERPOD_CLUSTER"],
            node_selector=environment.get("GPU_FAULT_NODE_SELECTOR") or None,
            version=version,
            config_digest=environment["GPU_FAULT_INSTALLER_CONFIG_DIGEST"],
            artifact_sha256=environment["GPU_FAULT_INSTALLER_ARTIFACT_SHA256"],
            bundle_sha256=environment["GPU_FAULT_INSTALLER_BUNDLE_SHA256"],
            template_sha256=environment["GPU_FAULT_INSTALLER_TEMPLATE_SHA256"],
            job_template=job_template(),
            dcgm_metrics_url_template="http://{node_ip}:9400/metrics",
            dcgm_hostengine=environment.get(DCGM_HOSTENGINE_ENV, ""),
            max_unavailable=int(environment["GPU_FAULT_INSTALLER_MAX_UNAVAILABLE"]),
            allowed_node_names=(
                None
                if allowed == "*"
                else frozenset(name for name in allowed.split(",") if name)
            ),
            now=lambda: self.clock,
            agent_alive=lambda *_args: True,
        )
        self.deploys.append(
            SimpleNamespace(
                allowed=allowed,
                max_unavailable=reconciler.max_unavailable,
                template=reconciler.template_sha256,
                version=reconciler.version,
            )
        )
        for _ in range(16):
            before = set(self.batch.jobs)
            reconciler.reconcile_once()
            created = sorted(
                self.batch.jobs[name]["node"] for name in set(self.batch.jobs) - before
            )
            for name in set(self.batch.jobs) - before:
                self.job_identities[name] = (
                    reconciler.template_sha256,
                    reconciler.dcgm_hostengine,
                )
            if created:
                # One pass's Jobs run side by side: this is the set of Agents
                # that are down together.
                self.installs.append(
                    SimpleNamespace(
                        nodes=created,
                        allowed=allowed,
                        gate=self.gates[-1] if self.gates else None,
                    )
                )
            running = [
                name for name, item in self.batch.jobs.items() if not item["done"]
            ]
            if not running and not created:
                return
            for name in running:
                job = self.batch.jobs[name]
                job["done"] = True
                template, mode = self.job_identities[name]
                self.heartbeat(job["node"], template=template)
        raise AssertionError("the simulated reconciler never went idle")

    # -- the fleet probe ------------------------------------------------------

    def fleet_command(self, operation: str, payload: dict[str, Any]) -> Any:
        if operation == "create":
            request = dict(payload["request"])
            # What deploy/control-plane/regional/probes/fleet_deployment_command
            # fills in from the control plane's own version and policy.
            request.setdefault("desired_agent_version", AGENT_VERSION)
            request.setdefault("desired_policy_version", POLICY)
            deployment = self.registry.create_deployment(
                FleetDeploymentRequest.model_validate(request)
            )
            self.created.append(deployment)
            return json.loads(deployment.model_dump_json())
        if operation == "get":
            deployment = self.registry.store.get_fleet_deployment(
                payload["deployment_id"]
            )
            return json.loads(deployment.model_dump_json())
        if operation == "next-wave":
            lease = self.registry.start_next_wave(payload["deployment_id"])
            return json.loads(lease.model_dump_json())
        if operation == "normalize-records":
            return {"normalized": 0}
        if operation == "terminalize-cluster-rollouts":
            return {"terminalized": []}
        raise AssertionError(f"unexpected fleet operation {operation}")

    def wait_agents(self, target: Any, artifact_sha: str, **options: Any) -> None:
        """The release's convergence wait, failing fast instead of polling."""

        names = frozenset(options.get("node_names") or ()) or None
        self.waits.append(SimpleNamespace(nodes=names, **options))
        if not agents_converged(
            list(self.nodes.items.values()),
            target,
            artifact_sha,
            bundle_sha=options.get("bundle_sha"),
            template_sha=options.get("template_sha"),
            config_digest=options.get("config_digest"),
            require_node_uid=True,
            node_names=names,
            dcgm_hostengine=options.get("dcgm_hostengine"),
        ):
            raise ClusterLocalReleaseError(
                f"{target.cluster_id} agents did not converge"
            )


def _eks_target(target: Any, mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            key: value
            for key, value in vars(target).items()
            if key not in {"dcgm_hostengine", "dcgm_hostengine_source"}
        },
        provider="eks-managed",
        node_selector=EKS_SELECTOR,
        node_selector_source="label-convention",
        dcgm_hostengine=mode,
        dcgm_hostengine_source="node-group-label" if mode == "host" else "default",
    )


@pytest.fixture
def site_release(
    node_release: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    """The node image target release, wired to a ``Site`` instead of stubs."""

    release = node_release.release
    # "auto": the cap the node count allows (four), with the canary and the
    # per-domain budget the policy derives from it.
    release.config.upgrade_max_unavailable = 0
    # The node bundle this release rolls (``build-node-installer-bundle.sh``).
    release.config.bundle = Path(BUNDLE_FILE)
    holder = SimpleNamespace(site=None)

    def run(arguments: list[str], *, env: dict[str, str] | None = None, **options: Any):
        if "patch" in arguments:
            return ""
        assert Path(arguments[0]).name == "deploy-node-installer-reconciler.sh", (
            f"unexpected command {arguments}"
        )
        assert env is not None, "the reconciler script runs with an environment"
        node_release.environments.append(dict(env))
        if env.get("GPU_FAULT_RECONCILER_PREFLIGHT_ONLY") == "true":
            script_version(env)
            return json.dumps({"status": "PASSED", "node_count": len(NODES)})
        holder.site.deploy(env)
        return ""

    def get_json(arguments: list[str]) -> dict[str, Any]:
        if arguments[-2:] == ["configmap", "previous-template"]:
            # The rollback's captured installer template.
            return node_release.template_document
        assert "nodes" in arguments, f"unexpected read {arguments}"
        return {"items": copy.deepcopy(list(holder.site.nodes.items.values()))}

    def gate(_release: Any, _target: Any, *, wave: tuple[str, ...], **_options: Any):
        holder.site.gates.append(tuple(sorted(wave)))
        return {"minimum_lease_remaining_seconds": 600}

    release.runner.run = run
    for name, value in {
        "_get_json": get_json,
        # The real deployment id: a forward rollout and its rollback are two
        # records.
        "_fleet_deployment_id": lambda target, **options: (
            FLEET.fleet_deployment_id(release, target, **options)
        ),
        "_target_node_names": lambda _target: tuple(sorted(NODES)),
        "_fleet_command": lambda operation, payload: holder.site.fleet_command(
            operation, payload
        ),
        "_wait_agents": lambda target, artifact, **options: holder.site.wait_agents(
            target, artifact, **options
        ),
    }.items():
        monkeypatch.setattr(release, name, value)
    for module in (FLEET, RUNTIME):
        monkeypatch.setattr(module, "ensure_rollout_wave_safe", gate)
    monkeypatch.setattr(
        RUNTIME, "ensure_pre_node_mutation_barrier", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(RUNTIME, "product_isolated_nodes", lambda *_args: [])
    return SimpleNamespace(release=release, holder=holder, scenario=node_release)


def _roll(scenario: SimpleNamespace, target: Any) -> None:
    RUNTIME.roll_node_runtime(
        scenario.release,
        target,
        phase="upgrade",
        wheel_cm="candidate-wheel",
        bundle_cm=scenario.release.bundle_cm,
        artifact_sha=ARTIFACT,
        config_digest=CONFIG,
        candidate_preflight_completed=True,
    )


def _site(scenario: SimpleNamespace, target: Any, *, template: str, mode: str):
    site = Site(target, template=template, mode=mode)
    scenario.holder.site = site
    return site


def _installed_identity(scenario: SimpleNamespace, target: Any) -> tuple[str, str]:
    """The (template, mode) a reconciler of this release installs on ``target``."""

    site = _site(scenario, target, template=TEMPLATE, mode="embedded")
    _roll(scenario, target)
    deploy = site.deploys[-1]
    return deploy.template, str(target.dcgm_hostengine)


def _assert_rolled_through_waves(site: Site) -> None:
    (deployment,) = site.created
    assert deployment.status is not DeploymentStatus.SUCCEEDED, (
        "a node-configuration change opened a fleet deployment that was already "
        "SUCCEEDED: every wave was skipped"
    )
    assert site.installs, "the configuration change re-installs the Agents"
    assert all(item.allowed != "*" for item in site.installs), (
        "the steady-state reconciler re-installed Agents outside every fleet "
        f"wave: {[(item.allowed, item.nodes) for item in site.installs]}"
    )
    assert len(site.installs[0].nodes) == 1, (
        f"the first re-install is a one-node canary, got {site.installs[0].nodes}"
    )
    for item in site.installs:
        wave = tuple(sorted(item.allowed.split(",")))
        assert set(item.nodes) <= set(wave), "a Job ran outside its wave"
        assert item.gate == wave, (
            f"wave {wave} was handed to the reconciler without its own safety gate "
            f"(last gate {item.gate})"
        )
        domains = Counter(NODES[name] for name in item.nodes)
        assert max(domains.values()) <= PER_DOMAIN_CAP, (
            f"wave {wave} took more than {PER_DOMAIN_CAP} nodes of one failure domain"
        )
    reinstalled = sorted(name for item in site.installs for name in item.nodes)
    assert reinstalled == sorted(NODES), "every Agent is re-installed exactly once"
    final = site.registry.store.get_fleet_deployment(deployment.deployment_id)
    assert final.status is DeploymentStatus.SUCCEEDED, "the rollout completed"
    assert site.deploys[-1].allowed == "*", "the steady-state reconciler is restored"


def test_a_dcgm_mode_flip_rolls_through_the_canary_and_gated_waves(
    site_release: SimpleNamespace,
) -> None:
    """Embedded to host: the paused reconciler names ``host``, every Agent still
    runs the embedded engine and reports exactly the nine-field identity the
    release asks for."""

    target = _eks_target(site_release.scenario.target, "host")
    site = _site(site_release, target, template=TEMPLATE, mode="embedded")

    _roll(site_release, target)

    _assert_rolled_through_waves(site)
    assert {item.get("dcgm_hostengine") for item in map(vars, site.waits)} == {
        "host"
    }, "every wave and the steady-state wait hold the Agents to the new mode"


def test_leaving_host_mode_rolls_through_the_canary_and_gated_waves(
    site_release: SimpleNamespace,
) -> None:
    """Host back to embedded, from Agents this engine installed in host mode."""

    host = _eks_target(site_release.scenario.target, "host")
    template, _mode = _installed_identity(site_release, host)
    embedded = _eks_target(site_release.scenario.target, "embedded")
    site = _site(site_release, embedded, template=template, mode="host")

    _roll(site_release, embedded)

    _assert_rolled_through_waves(site)


@pytest.mark.parametrize("cluster", ["hyperpod", "embedded", "host"])
def test_a_release_without_a_node_change_stays_a_no_op(
    site_release: SimpleNamespace, cluster: str
) -> None:
    """Re-deploying what the cluster already runs opens no wave and re-installs
    nothing: the deployment is SUCCEEDED at creation and the steady-state
    reconciler finds every node current."""

    base = site_release.scenario.target
    target = base if cluster == "hyperpod" else _eks_target(base, cluster)
    template, mode = (
        (TEMPLATE, "embedded")
        if cluster == "hyperpod"
        else _installed_identity(site_release, target)
    )
    site = _site(site_release, target, template=template, mode=mode)

    _roll(site_release, target)

    (deployment,) = site.created
    assert deployment.status is DeploymentStatus.SUCCEEDED, (
        "every Agent already reports the release identity"
    )
    assert site.installs == [], (
        f"a no-change release re-installed Agents: {site.installs}"
    )
    assert [item for item in site.gates if item] == [], "no wave was gated or leased"
    assert [item.allowed for item in site.deploys] == ["", "*"], (
        "the paused reconciler, then the steady-state one, and nothing between"
    )
    if cluster in {"hyperpod", "embedded"}:
        assert {item.template for item in site.deploys} == {TEMPLATE}, (
            "a cluster without a host engine keeps the template source identity "
            "byte for byte"
        )


def test_rolling_back_a_mode_flip_rolls_through_the_rollback_waves(
    site_release: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The release that flipped embedded to host rolled its Agents through the
    waves; its rollback restores the previous release's embedded Agents the
    same way, not through the restored reconciler's ``*``. Everything the
    rollback restores but the template is what the Agents already report."""

    scenario = site_release.scenario
    host = _eks_target(scenario.target, "host")
    site = _site(site_release, host, template=TEMPLATE, mode="embedded")
    _roll(site_release, host)
    site.reset()
    previous = scenario.previous
    previous["metadata"]["required-agent-protocol-version"] = (
        CURRENT_AGENT_PROTOCOL_VERSION
    )
    # The previous release ran the embedded engine from the same source.
    previous["clusters"]["gpu-a"]["dcgm_hostengine"] = "embedded"
    # The key of the previous bundle, as the snapshot captures it; the version
    # the restored reconciler renders with comes from it, never from the shell.
    previous["clusters"]["gpu-a"]["bundle_key"] = BUNDLE_FILE
    monkeypatch.setenv("GPU_FAULT_VERSION", "9.9.9")
    before = len(scenario.environments)
    previous["agent_identities"]["gpu-a"].update(
        agent_protocol_version=CURRENT_AGENT_PROTOCOL_VERSION,
        agent_version=AGENT_VERSION,
        policy_version=POLICY,
        runtime_profile_version=PROFILE,
        node_ids=sorted(NODES),
    )

    rollback_target(
        site_release.release,
        host,
        previous=previous,
        artifact=ARTIFACT,
        config_digest=CONFIG,
        runtime_profile_version=PROFILE,
        executor_artifact=ARTIFACT,
        executor_compatibility=ARTIFACT,
        node_compatibility=ARTIFACT,
        runtime_image=PREVIOUS_EXECUTOR,
        node_installer_image=INSTALLER,
        components=frozenset({ReleaseComponent.AGENT}),
    )

    _assert_rolled_through_waves(site)
    assert {item.template for item in site.deploys} == {TEMPLATE}, (
        "the rollback restores the captured template identity everywhere"
    )
    assert all(
        DCGM_HOSTENGINE_ENV in env and env[DCGM_HOSTENGINE_ENV] == "embedded"
        for env in scenario.environments[-len(site.deploys) :]
    ), "the restored reconciler installs the captured mode"
    assert {env["GPU_FAULT_VERSION"] for env in scenario.environments[before:]} == {
        INSTALLER_VERSION
    }, (
        "the rollback's preflight and reconciler are handed the previous bundle's version"
    )


@pytest.mark.parametrize("cluster", ["hyperpod", "host"])
def test_an_exported_gpu_fault_version_never_moves_the_node_identity(
    site_release: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, cluster: str
) -> None:
    """``GPU_FAULT_VERSION`` exported in the deploy host's shell names a
    version the release never shipped. The reconciler script renders the
    installer Job and its node identity with it, and the environment the engine
    hands the script starts from the deploy host's: the script then found no
    bundle of that version and refused every reconciler deploy of the release.
    The engine pins the version of the bundle it rolls, so the same no-change
    release is still one: every node keeps its identity and no ``*`` re-installs
    anything."""

    base = site_release.scenario.target
    target = base if cluster == "hyperpod" else _eks_target(base, cluster)
    template, mode = (
        (TEMPLATE, "embedded")
        if cluster == "hyperpod"
        else _installed_identity(site_release, target)
    )
    site = _site(site_release, target, template=template, mode=mode)
    monkeypatch.setenv("GPU_FAULT_VERSION", "9.9.9")
    before = len(site_release.scenario.environments)

    RUNTIME.preflight_node_runtime(
        site_release.release,
        target,
        phase="upgrade",
        wheel_cm="candidate-wheel",
        bundle_cm=site_release.release.bundle_cm,
        artifact_sha=ARTIFACT,
        config_digest=CONFIG,
    )
    _roll(site_release, target)

    (deployment,) = site.created
    assert deployment.status is DeploymentStatus.SUCCEEDED, (
        "every Agent still reports the release identity"
    )
    assert site.installs == [], f"Agents were re-installed: {site.installs}"
    assert [item.allowed for item in site.deploys] == ["", "*"], (
        "the paused reconciler, then the steady-state one, and nothing between"
    )
    assert {item.version for item in site.deploys} == {INSTALLER_VERSION}, (
        "the reconciler's node identity keeps the released installer version"
    )
    handed = site_release.scenario.environments[before:]
    assert len(handed) == 3, "the preflight, the paused and the steady-state run"
    assert {env["GPU_FAULT_VERSION"] for env in handed} == {INSTALLER_VERSION}, (
        "every script run is handed the version of the bundle the release rolls"
    )
    assert {
        item["metadata"]["annotations"][INSTALLER_VERSION_ANNOTATION]
        for item in site.nodes.items.values()
    } == {INSTALLER_VERSION}, "no node identity moved"
