"""A DATA_PLANE_COMPATIBLE deploy with Agent waves through the real engine,
over GPU clusters without nodes.

``RegionalRelease.upgrade`` runs as the deploy runs it -- the real phase
orchestration, its thread pools and the real report snapshot
(``read_snapshot``) -- over two GPU clusters: ``gpu-eks`` (eks-managed, its
only GPU node group at ``desiredSize`` 0: parked, no node, no Agent) and
``gpu-a`` (HyperPod) in one of three states: one node with its Agent (the
live site), its instance group parked at ``TargetCount`` 0 too, or
unavailable -- ``TargetCount`` 1, its node lost. The real code covers the
previous-state capture's Agent set, the candidate preflight of both clusters
in parallel, the CPU stage and its heartbeat barrier, both clusters' rollout
(the zero-node path, the fleet waves), the finalize barrier, the commit and,
when the post-rollout verification fails, the automatic rollback -- and its
resume after an interruption.

Only what leaves the process is scripted: every kubectl/aws read answers
from ``EngineRunner``, the reconciler script and the in-Pod probes are
recorded and answered, and the mutations the plan owns outside the GPU
rollout (upload, schema, CPU apply, endpoint, observability) are recorded.
"""

from __future__ import annotations

import json
import threading
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from gpu_fault_release import regional_parked_cluster as PARKED
from gpu_fault_release import regional_release_agent_set as AGENT_SET
from gpu_fault_release import regional_release_fleet_rollout as FLEET
from gpu_fault_release import regional_release_node_runtime_rollout as RUNTIME
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release import regional_release_rollback_target as ROLLBACK_TARGET
from gpu_fault_release import regional_release_state as STATE
from gpu_fault_release import regional_release_validation as VALIDATION
from gpu_fault_release import rollout as ROLLOUT
from gpu_fault_release.regional_admin_commands import run_resume
from gpu_fault_release.regional_release_diff import ReleaseChangeKind, ReleaseDiff
from gpu_fault_release.regional_release_probes import probe_label
from gpu_fault_release.regional_release_rollback_context import (
    rollback_controller_config_for,
)
from tests.regional._cov95_release_support import deployment
from tests.regional._release_orchestrator_support import (
    GPU_EKS_ARN,
    REGION,
    aurora_refresh_snapshot,
    config_file,
    ingress_pod_list_json,
)

SITE = "us-east-1-site-0123abcd"
PREVIOUS_IMAGE = "registry.example/runtime@sha256:" + "0" * 64
SELECTOR = f"gpu-fault.io/control-plane={SITE}"
EKS = "gpu-eks"
HYPERPOD = "gpu-a"
EKS_ARN = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-eks"
GROUP = "gpu-p5en-group"
NODE = "hyperpod-i-1"
PREVIOUS_TEMPLATE = "previous-template"
CONTROLLER_STAGE = "controller-stage"

#: ``gpu-a``'s instance group as ``describe-cluster`` reports it
#: (``TargetCount``, ``CurrentCount``): its node with an Agent, parked, or
#: wanting an instance it has lost.
HYPERPOD_GROUPS = {"nodes": (1, 1), "parked": (0, 0), "unavailable": (1, 0)}
SITES = [
    pytest.param("nodes", id="gpu-eks-parked"),
    pytest.param("parked", id="every-cluster-parked"),
    pytest.param("unavailable", id="parked-and-unavailable"),
]
#: The reconciler script on a cluster without nodes, per release step: the
#: zero-node preflight, then the steady-state Reconciler over every node.
ZERO_NODE_SCRIPTS = [("true", "true", ""), ("", "true", "*")]


def _clusters() -> list[dict[str, Any]]:
    return [
        {
            "cluster_id": HYPERPOD,
            "context": "gpu-a-context",
            "executor_irsa_role_arn": "arn:aws:iam::1:role/a",
            "region": REGION,
            "hyperpod_cluster_name": "hp-gpu-a",
            "eks_cluster_arn": GPU_EKS_ARN,
            "agent_endpoint_allowed_cidrs": ["10.0.0.0/16"],
        },
        {
            "cluster_id": EKS,
            "context": "gpu-eks-context",
            "provider": "eks-managed",
            "node_selector": SELECTOR,
            "executor_irsa_role_arn": "arn:aws:iam::1:role/eks",
            "region": REGION,
            "hyperpod_cluster_name": "",
            "eks_cluster_arn": EKS_ARN,
            "agent_endpoint_allowed_cidrs": ["10.0.0.0/16"],
        },
    ]


class EngineRunner:
    """The engine's commands, answered like the site: ``gpu-a`` as
    ``hyperpod`` says (``HYPERPOD_GROUPS``), ``gpu-eks`` parked; every command
    is recorded. A context in ``interrupt`` fails its next rollback
    Reconciler deploy once (``CONTROLLER_STAGE``: the rollback's controller
    stage)."""

    dry_run = False

    def __init__(self, *, hyperpod: str = "nodes") -> None:
        self.hyperpod = hyperpod
        self.commands: list[list[str]] = []
        self.scripts: list[dict[str, str]] = []
        self.probes: list[str] = []
        self.barriers: list[dict[str, Any]] = []
        self.configmaps: dict[str, dict[str, str]] = {}
        self.interrupt: set[str] = set()
        self.lock = threading.Lock()

    def answer_probe(self, name: str, input_text: str | None) -> str:
        """The control plane's in-Pod answers: one ACTIVE Agent on gpu-a's
        node when it has one, none on gpu-eks; every barrier passes."""

        if name == "active_agent_identities":
            if self.hyperpod != "nodes":
                return "[]"
            return json.dumps(
                [
                    {
                        "cluster_id": HYPERPOD,
                        "node_id": NODE,
                        "agent_protocol_version": 3,
                        "agent_version": "0.10.0",
                        "artifact_sha256": "1" * 64,
                        "compatibility_digest": "1" * 64,
                        "installer_bundle_sha256": "b" * 64,
                        "installer_template_sha256": "c" * 64,
                        "policy_version": "policy-v1",
                        "runtime_profile_version": "hyperpod-v1",
                        "config_digest": "a" * 64,
                        "node_action_key_version": 2,
                    }
                ]
            )
        if name == "agent_heartbeat_barrier":
            request = json.loads(input_text or "{}")
            with self.lock:
                self.barriers.append(request)
            return json.dumps({"status": "PASSED"})
        raise AssertionError(f"unexpected probe: {name}")

    def _context(self, arguments: list[str]) -> str:
        if "--context" in arguments:
            return arguments[arguments.index("--context") + 1]
        return "cpu"

    def run(self, arguments: list[str], **kwargs: Any) -> str:
        arguments = [str(item) for item in arguments]
        with self.lock:
            self.commands.append(arguments)
        if arguments[0].endswith("deploy-node-installer-reconciler.sh"):
            return self._reconciler(dict(kwargs.get("env") or {}))
        if "exec" in arguments and "-c" in arguments:
            source = arguments[arguments.index("-c") + 1]
            name = (probe_label(source) or "").removeprefix("<probe:").rstrip(">")
            with self.lock:
                self.probes.append(name)
            return self.answer_probe(name, kwargs.get("input_text"))
        if arguments[0] == "python3" and arguments[1].endswith(".py"):
            return ""  # the read-only verifiers pass
        context = self._context(arguments)
        if (
            arguments[0] == "kubectl"
            and "get" in arguments
            and "configmap" in arguments
        ):
            name = arguments[arguments.index("configmap") + 1]
            if name in self.configmaps:
                return json.dumps({"data": self.configmaps[name]})
        if (
            arguments[0] == "kubectl"
            and "get" in arguments
            and "deployment" in arguments
        ):
            name = arguments[arguments.index("deployment") + 1]
            return json.dumps(
                deployment(
                    name,
                    image=PREVIOUS_IMAGE,
                    wheel="previous-wheel",
                    container="reconciler" if "reconciler" in name else None,
                )
            )
        if arguments[0] == "kubectl" and "get" in arguments and "pod" in arguments:
            return (
                ingress_pod_list_json("api-pod")
                if "json" in " ".join(arguments)
                else "api-pod"
            )
        if arguments[0] == "kubectl" and "get" in arguments and "nodes" in arguments:
            populated = context == "gpu-a-context" and self.hyperpod == "nodes"
            items = [_node(NODE)] if populated else []
            return json.dumps({"apiVersion": "v1", "kind": "List", "items": items})
        if (
            arguments[0] == "kubectl"
            and "get" in arguments
            and (
                "deployments" in arguments
                or "jobs" in arguments
                or ("secret" in arguments and "-l" in arguments)
            )
        ):
            return json.dumps({"items": []})
        if arguments[:3] == ["aws", "eks", "list-nodegroups"]:
            return json.dumps({"nodegroups": [GROUP]})
        if arguments[:3] == ["aws", "eks", "describe-nodegroup"]:
            return json.dumps(
                {
                    "nodegroupName": GROUP,
                    "status": "ACTIVE",
                    "instanceTypes": ["p5en.48xlarge"],
                    "labels": {
                        "gpu-fault.io/control-plane": SITE,
                        "nvidia.com/gpu.present": "true",
                    },
                    "scalingConfig": {"minSize": 0, "maxSize": 2, "desiredSize": 0},
                }
            )
        if arguments[:3] == ["aws", "sagemaker", "describe-cluster"]:
            target, current = HYPERPOD_GROUPS[self.hyperpod]
            return json.dumps(
                {
                    "EksClusterArn": GPU_EKS_ARN,
                    "InstanceGroups": [
                        {
                            "InstanceGroupName": "gpu-spot-group",
                            "InstanceType": "ml.p5en.48xlarge",
                            "CurrentCount": current,
                            "TargetCount": target,
                            "Status": "InService",
                            "KubernetesConfig": {"CurrentLabels": {}},
                        }
                    ],
                }
            )
        if arguments[0] == "kubectl" and "patch" in arguments:
            return ""
        raise AssertionError(
            f"unexpected command: {arguments[:8]}\n"
            + "".join(traceback.format_stack(limit=14)[:-1])
        )

    def probe_output(
        self, arguments: list[str], *, timeout_seconds: float
    ) -> tuple[int, str, str]:
        if "nodepools.karpenter.sh" in arguments:
            with self.lock:
                self.commands.append([str(item) for item in arguments])
            return 1, "", 'error: the server doesn\'t have a resource type "nodepools"'
        try:
            return 0, self.run(list(arguments)), ""
        except AssertionError:
            raise
        except Exception as error:  # noqa: BLE001 - a probe reports, never raises
            return 1, "", str(error)

    def last_script_environment(self, context: str) -> dict[str, str]:
        with self.lock:
            matching = [
                item
                for item in self.scripts
                if item.get("GPU_FAULT_KUBECTL_CONTEXT") == context
            ]
        assert matching, f"no reconciler deployed on {context}"
        return matching[-1]

    def _reconciler(self, environment: dict[str, str]) -> str:
        zero = environment.get(PARKED.ZERO_NODE_CLUSTER_ENV) == "true"
        preflight = environment.get("GPU_FAULT_RECONCILER_PREFLIGHT_ONLY") == "true"
        context = environment.get("GPU_FAULT_KUBECTL_CONTEXT", "")
        with self.lock:
            self.scripts.append(environment)
            interrupted = (
                context in self.interrupt
                and not preflight
                and environment.get("GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP")
                == PREVIOUS_TEMPLATE
            )
            self.interrupt.discard(context if interrupted else "")
        if interrupted:
            raise ROLLOUT.ReleaseError(f"{context}: the deploy host was interrupted")
        products = environment.get(FLEET.INSTALLER_PRODUCTS_FILE_ENV)
        if products:
            Path(products).write_text(
                json.dumps(
                    {
                        "node_set_sha256": "0" * 64,
                        "node_action_keys_provisioned": False,
                        "template_config_map": "gpu-fault-node-installer-template-x",
                        "template_content_sha256": "c" * 64,
                        "template_rendered": True,
                    }
                ),
                encoding="utf-8",
            )
        if preflight:
            return json.dumps({"node_count": 0 if zero else 1, "status": "PASSED"})
        return ""


def _node(name: str) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Node",
        "metadata": {
            "name": name,
            "uid": f"{name}-uid",
            "labels": {"sagemaker.amazonaws.com/cluster-name": "hp-gpu-a"},
            # The identity the node's Agent runs: the release it started from.
            "annotations": {
                "gpu-fault.io/installer-state": "Succeeded",
                "gpu-fault.io/installer-node-uid": f"{name}-uid",
                "gpu-fault.io/installer-artifact-sha256": "1" * 64,
                "gpu-fault.io/installer-config-digest": "a" * 64,
                "gpu-fault.io/installer-bundle-sha256": "b" * 64,
                "gpu-fault.io/installer-template-sha256": "c" * 64,
            },
        },
        "spec": {},
        "status": {
            "conditions": [{"type": "Ready", "status": "True"}],
            "addresses": [{"type": "InternalIP", "address": "10.0.1.10"}],
        },
    }


def _engine(tmp_path: Path, *, hyperpod: str) -> tuple[Any, EngineRunner]:
    config = ROLLOUT.ReleaseConfig.load(config_file(tmp_path, clusters=_clusters()))
    runner = EngineRunner(hyperpod=hyperpod)
    release = ROLLOUT.RegionalRelease(config, runner)  # type: ignore[arg-type]
    return release, runner


# --- the deploy -------------------------------------------------------------------

DIFF = ReleaseDiff(
    kind=ReleaseChangeKind.DATA_PLANE_COMPATIBLE,
    changed=frozenset({"executor_wheel", "node_runtime_wheel", "node_bundle"}),
)
DIFF_PLAN_NODES = (
    "registry",
    "cpu-stage",
    "executor",
    "watcher",
    "collector",
    "reconciler",
    "agent",
    "cpu-finalize",
    "verify",
)


class Deploy:
    """What the deploy recorded beside the commands: phases, mutations,
    fleet operations; ``store`` is the persisted release state a new
    deploy-host process reads back."""

    def __init__(self, store: dict[str, Any] | None = None) -> None:
        self.phases: list[str] = []
        self.calls: list[tuple[str, Any]] = []
        self.fleet: list[tuple[str, str]] = []
        self.previous: dict[str, Any] = {}
        self.snapshot_openers: list[tuple[str, ...]] = []
        self.store: dict[str, Any] = {} if store is None else store
        self.lock = threading.Lock()

    def snapshot_steps(self) -> set[str]:
        """The steps that opened a report snapshot during the deploy."""

        allowed = ("_capture", "capture_previous")
        return {
            next((name for name in reversed(stack) if name in allowed), stack[-1])
            for stack in self.snapshot_openers
        }

    def record(self, name: str, value: Any = None) -> None:
        with self.lock:
            self.calls.append((name, value))


def _bind(
    release: Any,
    runner: EngineRunner,
    monkeypatch: pytest.MonkeyPatch,
    *,
    store: dict[str, Any] | None = None,
) -> Deploy:
    deploy = Deploy(store)

    def save_state(phase: str, **updates: Any) -> None:
        with deploy.lock:
            release.state.update({"phase": phase, **updates})
            deploy.phases.append(phase)
            deploy.store.clear()
            deploy.store.update(json.loads(json.dumps(release.state, default=str)))

    def method(name: str, value: Any) -> None:
        monkeypatch.setattr(release, name, value)

    method("_save_state", save_state)
    # Every report snapshot opened while the deploy runs, by whom: a snapshot
    # is release-wide state, so only a single-threaded step that owns the whole
    # release may open one -- never a per-cluster step.
    real_snapshot = STATE.read_snapshot

    @contextmanager
    def watched_snapshot(target_release: Any) -> Iterator[None]:
        with deploy.lock:
            deploy.snapshot_openers.append(
                tuple(frame.name for frame in traceback.extract_stack()[:-2])
            )
        with real_snapshot(target_release):
            yield

    monkeypatch.setattr(STATE, "read_snapshot", watched_snapshot)
    method("_read_snapshot", lambda: watched_snapshot(release))
    method("_load_state", lambda: json.loads(json.dumps(deploy.store)))
    for name in (
        "_ensure_contexts",
        "_require_cpu_secrets",
        "_upload_release",
        "_ensure_schema",
        "_apply_rds_ca_bundle",
        "_publish_staged_registry",
        "_apply_nlb",
        "_apply_control_plane_observability",
        "_apply_observability",
        "_apply_dataplane_expected_rules",
        "_commit_registry_update",
        "_cleanup_previous_snapshots",
        "_settle_installer_jobs",
        "_preflight_gpu_dcgm_exporter",
        "_preflight_gpu_adot_collector",
        "_apply_gpu_adot_collector",
        "_ensure_connection_secret",
        "_verify_gpu_control_plane_endpoint",
    ):
        method(name, lambda *args, _name=name, **kwargs: deploy.record(_name, args))
    method("_stage_registry", lambda: False)
    method("_remote_commands_are_idle", lambda: True)
    method("_require_no_inflight_installs", lambda **_k: None)
    method("_capture_previous", lambda plan=None: _capture(release, deploy))
    method("_backup_release_secrets", lambda: {})
    method("_refresh_aurora_credentials", lambda **_k: None)
    method("_aurora_refresh_drift", lambda: False)
    method("_observability_drift", lambda: False)
    method(
        "_apply_cpu",
        lambda **kwargs: deploy.record("cpu", bool(kwargs.get("finalize"))),
    )
    method(
        "_preflight_gpu_deployments",
        lambda target, *_a, **_k: deploy.record(
            "preflight-deployments", target.cluster_id
        ),
    )
    method(
        "_apply_gpu_deployments",
        lambda target, *_a, **_k: deploy.record("deployments", target.cluster_id),
    )
    method(
        "_apply_gpu_dcgm_exporter",
        lambda target, **_k: deploy.record("dcgm", target.cluster_id),
    )
    method(
        "_validate_release_quick",
        lambda plan: deploy.record("verify", tuple(item.value for item in plan.nodes)),
    )
    # The fleet's in-Pod side: the populated cluster's one-node wave.
    for module in (FLEET, RUNTIME):
        monkeypatch.setattr(
            module,
            "ensure_rollout_wave_safe",
            lambda *_a, **_k: {"minimum_lease_remaining_seconds": 120},
        )
    monkeypatch.setattr(FLEET, "hand_wave_to_reconciler", lambda *_a, **_k: None)
    monkeypatch.setattr(FLEET, "wait_deployment_rollout", lambda *_a, **_k: None)
    monkeypatch.setattr(
        FLEET,
        "reconciler_container_env",
        lambda _release, target: runner.last_script_environment(target.context),
    )

    def fleet_command(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        cluster_id = str(payload.get("cluster_id") or "")
        with deploy.lock:
            deploy.fleet.append((operation, cluster_id))
        if operation == "create":
            return {
                "status": "PENDING",
                "waves": [[NODE]],
                "nodes": [{"node_id": NODE, "status": "PENDING"}],
            }
        if operation == "next-wave":
            return {"node_ids": [NODE]}
        if operation == "get":
            return {"status": "SUCCEEDED"}
        return {}

    method("_fleet_command", fleet_command)
    # What the rollback's verification reads back: the previous release live.
    method("_deployment_wheel", lambda *_a, **_k: "previous-wheel")
    method("_deployment_template_name", lambda _target: PREVIOUS_TEMPLATE)
    method("_template_bundle", lambda _target, _template: "previous-bundle")
    method("_agent_heartbeats_converged", lambda target, **_k: True)
    monkeypatch.setattr(VALIDATION, "_validate_cpu_rollback", lambda *_a, **_k: None)

    # The rollback's CPU side: the controller stage keeps its one zero-node
    # question -- the Agent pins it restores, from the clusters with live
    # Agents, or none when no cluster had one -- and records them; the CPU
    # restores are recorded.
    def stage_controller(release_arg: Any, *, previous: Any, **kwargs: Any) -> dict:
        pins = rollback_controller_config_for(
            release_arg, previous, kwargs["runtime_profile_version"]
        )
        deploy.record("rollback-controller-pins", pins)
        with runner.lock:
            interrupted = CONTROLLER_STAGE in runner.interrupt
            runner.interrupt.discard(CONTROLLER_STAGE)
        if interrupted:
            raise ROLLOUT.ReleaseError("the deploy host was interrupted")
        return {}

    monkeypatch.setattr(ORCHESTRATION, "_stage_rollback_controller", stage_controller)
    monkeypatch.setattr(
        ORCHESTRATION,
        "_restore_regional_singletons",
        lambda *_a, **_k: deploy.record("rollback-singletons"),
    )
    monkeypatch.setattr(
        ORCHESTRATION,
        "_restore_rollback_cpu",
        lambda *_a, **_k: deploy.record("rollback-cpu"),
    )
    # The previous installer template's bytes against its captured pin: judged
    # the same for every cluster, outside what this harness proves.
    for module in (ORCHESTRATION, ROLLBACK_TARGET):
        monkeypatch.setattr(
            module, "validate_rollback_node_template", lambda *_a, **_k: None
        )
    method("_fleet_deployment_id", lambda target, **_k: f"fleet-{target.cluster_id}")
    method(
        "_wait_agents",
        lambda target, *_a, **_k: deploy.record("wait-agents", target.cluster_id),
    )
    return deploy


def _capture(release: Any, deploy: Deploy) -> dict[str, Any]:
    """The previous state the deploy starts from: the real report snapshot and
    the real Agent-set capture (a zero-node entry for a cluster without nodes),
    the rest as a committed release recorded it."""

    with STATE.read_snapshot(release):
        identities = AGENT_SET.capture_agent_identities_matching_nodes(release)
    deploy.record("capture", sorted(identities))
    previous = {
        "metadata": {
            "required-agent-artifact-sha256": "1" * 64,
            "required-agent-compatibility-digest": "1" * 64,
            "required-agent-protocol-version": "3",
            "required-agent-config-digest": "a" * 64,
            "required-regional-executor-artifact-sha256": "2" * 64,
            "required-regional-executor-compatibility-digest": "2" * 64,
            "required-regional-executor-protocol-version": "2",
            "required-node-action-key-version": "2",
        },
        "runtime_image": PREVIOUS_IMAGE,
        "executor_image": PREVIOUS_IMAGE,
        "node_installer_image": "registry.example/installer@sha256:" + "9" * 64,
        "cpu_wheel": "previous-cpu-wheel",
        "runtime_profile_version": release.config.runtime_profile_version,
        "clusters": {
            target.cluster_id: {
                "wheel": "previous-wheel",
                "reconciler_wheel": "previous-wheel",
                "reconciler_wheel_key": "executor.whl.xz",
                "template": PREVIOUS_TEMPLATE,
                "template_sha256": "e" * 64,
                "template_content_sha256": "f" * 64,
                "bundle": "previous-bundle",
                "bundle_sha256": "d" * 64,
            }
            for target in release.config.clusters
        },
        "agent_identities": identities,
        "aurora_refresh": aurora_refresh_snapshot(release.config.namespace),
    }
    deploy.previous = previous
    return previous


def _fail_verification(
    release: Any, runner: EngineRunner, deploy: Deploy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The post-rollout verification fails; the release metadata the
    rollback's verification reads back is the previous release's, as the
    restore leaves it."""

    def verify(_plan: Any) -> None:
        raise ROLLOUT.ReleaseError("forced verification failure")

    monkeypatch.setattr(release, "_validate_release_quick", verify)

    def capture(plan: Any = None) -> dict[str, Any]:
        previous = _capture(release, deploy)
        runner.configmaps["gpu-fault-release-metadata"] = dict(previous["metadata"])
        runner.configmaps["gpu-fault-api-ha-config-core"] = {
            "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": str(
                previous["runtime_profile_version"]
            )
        }
        return previous

    monkeypatch.setattr(release, "_capture_previous", capture)


def _scripts(runner: EngineRunner, context: str) -> list[tuple[str, str, str]]:
    """The reconciler script runs on ``context``: (preflight-only, zero-node
    mode, allowed nodes)."""

    return [
        (
            env.get("GPU_FAULT_RECONCILER_PREFLIGHT_ONLY", ""),
            env.get(PARKED.ZERO_NODE_CLUSTER_ENV, ""),
            env.get("GPU_FAULT_INSTALLER_ALLOWED_NODES", ""),
        )
        for env in runner.scripts
        if env.get("GPU_FAULT_KUBECTL_CONTEXT") == context
    ]


def _zero_node_clusters(hyperpod: str) -> dict[str, str]:
    """The clusters without nodes and the state each is in."""

    return {EKS: "parked", **({} if hyperpod == "nodes" else {HYPERPOD: hyperpod})}


def _assert_zero_node_entries(
    release: Any, deploy: Deploy, hyperpod: str, *, phase: str
) -> None:
    """The previous state's explicit entry and the release's record of the
    skipped node work, for each cluster without nodes; a live identity for a
    cluster with one."""

    identities = deploy.previous["agent_identities"]
    expected = _zero_node_clusters(hyperpod)
    for cluster_id, state in expected.items():
        identity = identities[cluster_id]
        assert PARKED.is_zero_node_identity(identity), identity
        assert identity[PARKED.ZERO_NODES_IDENTITY_KEY]["state"] == state, identity
        entry = release.state["zero_node_clusters"][cluster_id]
        assert (entry["state"], entry["phase"]) == (state, phase), entry
    assert sorted(release.state["zero_node_clusters"]) == sorted(expected)
    if hyperpod == "nodes":
        assert identities[HYPERPOD]["node_ids"] == [NODE]


@pytest.mark.parametrize("hyperpod", SITES)
def test_a_deploy_with_agent_waves_over_clusters_without_nodes_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hyperpod: str
) -> None:
    release, runner = _engine(tmp_path, hyperpod=hyperpod)
    deploy = _bind(release, runner, monkeypatch)

    release.upgrade(diff=DIFF)
    release.commit_release()

    assert deploy.phases[:4] == [
        "preflight",
        "uploaded",
        "candidate-preflight-ready",
        "registry-staged",
    ], deploy.phases
    finalized = deploy.phases.index("cpu-finalized")
    assert "data-converged" in deploy.phases[:finalized], deploy.phases
    assert set(deploy.phases[finalized + 1 :]) == {"complete"}, (
        "converged, finalized, verified and committed"
    )
    assert release.state["transaction_committed"] is True
    assert ("verify", DIFF_PLAN_NODES) in deploy.calls
    _assert_zero_node_entries(release, deploy, hyperpod, phase="upgrade")
    # A cluster without nodes: a zero-node preflight and the steady state.
    assert _scripts(runner, "gpu-eks-context") == ZERO_NODE_SCRIPTS
    if hyperpod == "nodes":
        # The populated cluster: its preflight, the paused wave, the wave and
        # the steady state; its fleet and the CPU barriers, for it alone.
        assert _scripts(runner, "gpu-a-context") == [
            ("true", "", NODE),
            ("", "", ""),
            ("", "", NODE),
            ("", "", "*"),
        ]
        assert {cluster for _operation, cluster in deploy.fleet if cluster} == {
            HYPERPOD
        }
        assert [barrier["expected_by_cluster"] for barrier in runner.barriers] == [
            {HYPERPOD: [NODE]}
        ] * 3, runner.barriers
    else:
        # No node anywhere: no fleet, no Agent to wait for.
        assert _scripts(runner, "gpu-a-context") == ZERO_NODE_SCRIPTS
        assert deploy.fleet == []
        assert runner.barriers == []
    assert deploy.snapshot_steps() == {"_capture"}, (
        "only the previous-state capture opens a report snapshot: "
        + repr(deploy.snapshot_openers)
    )


def _assert_rolled_back_on_the_previous_template(
    runner: EngineRunner, context: str, *, forward_runs: int
) -> None:
    """The rollback of a cluster without nodes: its Reconciler alone, on the
    previous template the snapshot pinned -- nothing rendered, no render
    node."""

    rollback = [
        env for env in runner.scripts if env.get("GPU_FAULT_KUBECTL_CONTEXT") == context
    ][forward_runs:]
    assert rollback, f"{context}: the Reconciler was rolled back"
    assert all(env[PARKED.ZERO_NODE_CLUSTER_ENV] == "true" for env in rollback), (
        "every rollback run is in zero-node mode"
    )
    assert all(env[PARKED.RENDER_NODE_ENV] == "" for env in rollback), (
        "the pinned previous template needs no render node"
    )
    assert all(
        env["GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP"] == PREVIOUS_TEMPLATE
        for env in rollback
    ), "on the previous template the snapshot pinned"


def _assert_rollback_pins(deploy: Deploy, hyperpod: str) -> None:
    pins = [value for name, value in deploy.calls if name == "rollback-controller-pins"]
    if hyperpod == "nodes":
        assert [item["GPU_FAULT_REQUIRED_AGENT_VERSION"] for item in pins] == [
            "0.10.0"
        ], "the Agent pins come from the cluster with live Agents"
    else:
        assert pins == [{}], (
            "no Agent ran: no pin to stage, the CPU restore puts the previous "
            "-config-core back"
        )


@pytest.mark.parametrize("hyperpod", SITES)
def test_a_failed_verification_rolls_the_deploy_back_over_clusters_without_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hyperpod: str
) -> None:
    release, runner = _engine(tmp_path, hyperpod=hyperpod)
    deploy = _bind(release, runner, monkeypatch)
    _fail_verification(release, runner, deploy, monkeypatch)

    with pytest.raises(ROLLOUT.ReleaseError, match="forced verification failure"):
        release.upgrade(diff=DIFF)

    assert deploy.phases[-4:] == [
        "rollback-verifying",
        "rollback-verified",
        "rolled-back",
        "rolled-back",
    ], deploy.phases
    assert "rollback-data-restored" in deploy.phases
    for cluster_id in _zero_node_clusters(hyperpod):
        context = f"{cluster_id}-context"
        _assert_rolled_back_on_the_previous_template(
            runner, context, forward_runs=len(ZERO_NODE_SCRIPTS)
        )
    _assert_rollback_pins(deploy, hyperpod)
    _assert_zero_node_entries(release, deploy, hyperpod, phase="rollback")
    assert deploy.snapshot_steps() == {"_capture"}, deploy.snapshot_openers


@pytest.mark.parametrize(
    ("interrupted", "resumed_from", "rolled_on_resume", "staged_on_resume"),
    [
        pytest.param(
            CONTROLLER_STAGE,
            "rollback-controller-staging",
            {"gpu-a-context", "gpu-eks-context"},
            [{}],
            id="in-the-controller-stage",
        ),
        pytest.param(
            "gpu-a-context",
            "rollback-data-restoring",
            {"gpu-a-context"},
            [],
            id="in-gpu-a-s-data-restore",
        ),
    ],
)
def test_an_interrupted_rollback_with_every_cluster_parked_resumes_to_rolled_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted: str,
    resumed_from: str,
    rolled_on_resume: set[str],
    staged_on_resume: list[dict[str, str]],
) -> None:
    """The automatic rollback stops in ``rollback-failed``; a new deploy-host
    process resumes it from the persisted state alone where it stopped --
    the controller stage judges the persisted zero-node entries again, a
    cluster already rolled back is not rolled again -- and rolls back to the
    previous release."""

    release, runner = _engine(tmp_path, hyperpod="parked")
    deploy = _bind(release, runner, monkeypatch)
    _fail_verification(release, runner, deploy, monkeypatch)
    runner.interrupt.add(interrupted)

    with pytest.raises(ROLLOUT.ReleaseError, match="automatic rollback also failed"):
        release.upgrade(diff=DIFF)
    assert deploy.phases[-1] == "rollback-failed", deploy.phases
    assert not runner.interrupt, "the interruption happened"
    assert [
        value for name, value in deploy.calls if name == "rollback-controller-pins"
    ] == [{}], "the controller stage ran with no Agent pin to stage"

    resumed = ROLLOUT.RegionalRelease(release.config, runner)  # type: ignore[arg-type]
    again = _bind(resumed, runner, monkeypatch, store=deploy.store)
    before = len(runner.scripts)

    run_resume(resumed)

    assert (again.phases[0], again.phases[-1]) == (resumed_from, "rolled-back"), (
        again.phases
    )
    assert "rollback-verified" in again.phases
    assert [
        value for name, value in again.calls if name == "rollback-controller-pins"
    ] == staged_on_resume
    scripts = runner.scripts[before:]
    assert {env["GPU_FAULT_KUBECTL_CONTEXT"] for env in scripts} == rolled_on_resume
    for context in rolled_on_resume:
        assert [
            (
                env.get("GPU_FAULT_RECONCILER_PREFLIGHT_ONLY", ""),
                env[PARKED.ZERO_NODE_CLUSTER_ENV],
                env["GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP"],
            )
            for env in scripts
            if env["GPU_FAULT_KUBECTL_CONTEXT"] == context
        ] == [("true", "true", PREVIOUS_TEMPLATE), ("", "true", PREVIOUS_TEMPLATE)]
    entries = resumed.state["zero_node_clusters"]
    assert {cluster: entries[cluster]["phase"] for cluster in entries} == {
        EKS: "rollback",
        HYPERPOD: "rollback",
    }
    assert again.snapshot_openers == [], "a rollback opens no report snapshot"
