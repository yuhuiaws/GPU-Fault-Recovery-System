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

A FULL release that moves ``gpu-a``'s own Runtime Profile (BOOT-020's FULL
stage) runs the same way, with the real state checkpoints, the registry
Secret staged and restored, and both CPU stages of its rollback up to the
role-split apply: the rollback puts every per-cluster pin back, and the next
release's capture of the rolled-back site records the previous release's.

Only what leaves the process is scripted: every kubectl/aws read answers
from ``EngineRunner``, the reconciler script and the in-Pod probes are
recorded and answered, and the mutations the plan owns outside the GPU
rollout (upload, schema, CPU apply, endpoint, observability) are recorded.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import regional_parked_cluster as PARKED
from gpu_fault_release import regional_release_agent_set as AGENT_SET
from gpu_fault_release import regional_release_fleet_rollout as FLEET
from gpu_fault_release import regional_release_node_runtime_rollout as RUNTIME
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release import regional_release_registry as REGISTRY
from gpu_fault_release import regional_release_rollback_context as ROLLBACK_CONTEXT
from gpu_fault_release import regional_release_rollback_target as ROLLBACK_TARGET
from gpu_fault_release import regional_release_state as STATE
from gpu_fault_release import regional_release_validation as VALIDATION
from gpu_fault_release import rollout as ROLLOUT
from gpu_fault_release.regional_admin_commands import run_resume
from gpu_fault_release.regional_release_diff import (
    ReleaseChangeKind,
    ReleaseComponent,
    ReleaseDiff,
    ReleaseExecutionPlan,
)
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
from tests.regional._resource_probe_fakes import resource_probe_result

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
    stage). ``registry`` is the regional cluster registry Secret's data once
    a test seeds it (``create --dry-run`` renders, ``apply`` persists)."""

    dry_run = False

    def __init__(self, *, hyperpod: str = "nodes") -> None:
        self.hyperpod = hyperpod
        self.commands: list[list[str]] = []
        self.scripts: list[dict[str, str]] = []
        self.probes: list[str] = []
        self.barriers: list[dict[str, Any]] = []
        self.transitions: list[dict[str, Any]] = []
        self.configmaps: dict[str, dict[str, str]] = {}
        self.registry: dict[str, str] | None = None
        self.interrupt: set[str] = set()
        self.lock = threading.Lock()

    def registry_payload(self, key: str) -> list[dict[str, Any]] | None:
        encoded = (self.registry or {}).get(key)
        return None if encoded is None else json.loads(base64.b64decode(encoded))

    def _registry_command(self, arguments: list[str], input_text: Any) -> str | None:
        """The registry Secret's render and apply; ``None`` for anything else."""

        if "create" in arguments and REGISTRY.REGISTRY_SECRET in arguments:
            data = {
                key: base64.b64encode(Path(path).read_bytes()).decode()
                for key, path in (
                    item.removeprefix("--from-file=").split("=", 1)
                    for item in arguments
                    if item.startswith("--from-file=")
                )
            }
            return json.dumps(
                {
                    "kind": "Secret",
                    "metadata": {"name": REGISTRY.REGISTRY_SECRET},
                    "data": data,
                }
            )
        if "apply" in arguments and isinstance(input_text, str):
            document = json.loads(input_text)
            if document.get("metadata", {}).get("name") == REGISTRY.REGISTRY_SECRET:
                with self.lock:
                    self.registry = dict(document["data"])
                return ""
        return None

    def answer_probe(self, name: str, input_text: str | None) -> str:
        """The control plane's in-Pod answers: one ACTIVE Agent on gpu-a's
        node when it has one, none on gpu-eks; every barrier passes; no
        workflow or workload holds a Runtime Profile the release leaves."""

        if name == "runtime_profile_transition":
            with self.lock:
                self.transitions.append(json.loads(input_text or "{}"))
            return json.dumps({"workflow_count": 0, "workload_count": 0})

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
        registry = self._registry_command(arguments, kwargs.get("input_text"))
        if registry is not None:
            return registry
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
        if "secret" in arguments and REGISTRY.REGISTRY_SECRET in arguments:
            with self.lock:
                data = None if self.registry is None else dict(self.registry)
            if data is None:
                return 0, "", ""
            code, raw, error = resource_probe_result(list(arguments))
            return code, json.dumps({**json.loads(raw), "data": data}), error
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


def _checkpoints(
    release: Any, deploy: Deploy, monkeypatch: pytest.MonkeyPatch, *, state_writer: bool
) -> Callable[..., None]:
    """The release's ``_save_state``: each checkpoint lands in ``deploy``.

    ``state_writer`` checkpoints through the real ``save_state`` -- every
    field it stamps from the running release; only the ConfigMap write and
    the history entry stay in the process.
    """

    def record(phase: str) -> None:
        with deploy.lock:
            deploy.phases.append(phase)
            deploy.store.clear()
            deploy.store.update(json.loads(json.dumps(release.state, default=str)))

    def save_state(phase: str, **updates: Any) -> None:
        # The state's own lock first, as a phase checkpointing from inside a
        # state transaction holds them.
        with STATE.state_transaction(release):
            STATE.save_state(release, phase, **updates)
            record(phase)

    def update_state(phase: str, **updates: Any) -> None:
        with deploy.lock:
            release.state.update({"phase": phase, **updates})
            deploy.phases.append(phase)
            deploy.store.clear()
            deploy.store.update(json.loads(json.dumps(release.state, default=str)))

    if not state_writer:
        return update_state
    monkeypatch.setattr(
        STATE,
        "persist_state",
        lambda target: json.dumps(target.state, default=str, sort_keys=True),
    )
    monkeypatch.setattr(STATE, "record_release_history", lambda *_a, **_k: None)
    return save_state


def _bind_cpu_rollback(
    release: Any, deploy: Deploy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the rollback's two CPU stages for real -- the restart plan, the
    registry restore, the controller patch -- up to the role-split render and
    apply, recorded with the environment and container env snapshot they were
    handed; the role ConfigMap restore and the registry publish are
    recorded."""

    def apply_cpu(_release: Any, environment: dict[str, str]) -> None:
        container_env = environment.get(ROLLBACK_CONTEXT.CONTAINER_ENV_FILE_VARIABLE)
        deploy.record(
            "rollback-cpu-apply",
            {
                "environment": dict(environment),
                "container_env": (
                    json.loads(Path(container_env).read_text(encoding="utf-8"))
                    if container_env
                    else None
                ),
            },
        )

    def method(name: str, value: Any) -> None:
        monkeypatch.setattr(release, name, value)

    monkeypatch.setattr(ORCHESTRATION, "_apply_rollback_cpu_environment", apply_cpu)
    method(
        "_restore_cpu_role_config_maps",
        lambda snapshots: deploy.record("rollback-role-config-maps", snapshots)
        or bool(snapshots),
    )
    method("_config_map_sha", lambda *_a, **_k: "4" * 64)
    method(
        "_publish_restored_registry", lambda: deploy.record("publish-restored-registry")
    )


def _bind(
    release: Any,
    runner: EngineRunner,
    monkeypatch: pytest.MonkeyPatch,
    *,
    store: dict[str, Any] | None = None,
    state_writer: bool = False,
    cpu_rollback: bool = False,
) -> Deploy:
    """Bind ``release`` to the scripted site (``state_writer``: see
    ``_checkpoints``; ``cpu_rollback``: see ``_bind_cpu_rollback``)."""

    deploy = Deploy(store)

    def method(name: str, value: Any) -> None:
        monkeypatch.setattr(release, name, value)

    method(
        "_save_state",
        _checkpoints(release, deploy, monkeypatch, state_writer=state_writer),
    )
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
    # A seeded registry Secret is staged as the release stages it.
    method(
        "_stage_registry",
        lambda: runner.registry is not None and REGISTRY.stage_registry(release),
    )
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
        lambda target, *_a, **kwargs: deploy.record(
            "deployments", (target.cluster_id, kwargs.get("runtime_profile_version"))
        ),
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
    # The durable registry head the rollback compares with the Secret: this
    # site has no registry API (test_manual_rollback_engine compares for real).
    monkeypatch.setattr(
        release,
        "_verify_rollback_registry_head",
        lambda: deploy.record("verify-registry-head") or {},
        raising=False,
    )

    # The rollback's CPU side: the controller stage keeps its one zero-node
    # question -- the Agent pins it restores, from the clusters with live
    # Agents, or the previous -config-core's when no cluster had one -- and
    # records them; the CPU restores are recorded.
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

    if cpu_rollback:
        _bind_cpu_rollback(release, deploy, monkeypatch)
    else:
        monkeypatch.setattr(
            ORCHESTRATION, "_stage_rollback_controller", stage_controller
        )
        monkeypatch.setattr(
            ORCHESTRATION,
            "_restore_rollback_cpu",
            lambda *_a, **_k: deploy.record("rollback-cpu"),
        )
    monkeypatch.setattr(
        ORCHESTRATION,
        "_restore_regional_singletons",
        lambda *_a, **_k: deploy.record("rollback-singletons"),
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


def _config_core_pins(release: Any) -> dict[str, str]:
    """The Agent pins the previous release's ``-config-core`` carried."""

    return {
        "GPU_FAULT_REQUIRED_AGENT_VERSION": "0.10.0",
        "GPU_FAULT_REQUIRED_POLICY_VERSION": "policy-v1",
        "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": str(
            release.config.runtime_profile_version
        ),
    }


def _capture(release: Any, deploy: Deploy, **recorded: Any) -> dict[str, Any]:
    """The previous state the deploy starts from: the real report snapshot and
    the real Agent-set capture (a zero-node entry for a cluster without nodes),
    the rest as a committed release recorded it; ``recorded`` replaces fields
    of that record (a previous release other than the candidate's own pins)."""

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
        # Every CPU role's ConfigMaps as they were: what the CPU restore puts
        # back.
        "cpu_role_config_maps": {
            f"{deployment}-config-core": {
                "GPU_FAULT_PROCESSOR_WORKERS": "24",
                **_config_core_pins(release),
            }
            for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS
        },
        **recorded,
    }
    deploy.previous = previous
    return previous


def _fail_verification(
    release: Any,
    runner: EngineRunner,
    deploy: Deploy,
    monkeypatch: pytest.MonkeyPatch,
    **recorded: Any,
) -> None:
    """The post-rollout verification fails; the release metadata the
    rollback's verification reads back is the previous release's, as the
    restore leaves it. ``recorded`` is handed to ``_capture``."""

    def verify(_plan: Any) -> None:
        with runner.lock:
            forward = len(runner.scripts)
        deploy.record("forward-scripts", forward)
        raise ROLLOUT.ReleaseError("forced verification failure")

    monkeypatch.setattr(release, "_validate_release_quick", verify)

    def capture(plan: Any = None) -> dict[str, Any]:
        previous = _capture(release, deploy, **recorded)
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


def _staged_pins(deploy: Deploy) -> list[Any]:
    return [value for name, value in deploy.calls if name == "rollback-controller-pins"]


def _assert_rollback_pins(release: Any, deploy: Deploy, hyperpod: str) -> None:
    pins = _staged_pins(deploy)
    if hyperpod == "nodes":
        assert [item["GPU_FAULT_REQUIRED_AGENT_VERSION"] for item in pins] == [
            "0.10.0"
        ], "the Agent pins come from the cluster with live Agents"
    else:
        assert pins == [_config_core_pins(release)], (
            "no Agent ran: the previous release's own -config-core pins"
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
    _assert_rollback_pins(release, deploy, hyperpod)
    _assert_zero_node_entries(release, deploy, hyperpod, phase="rollback")
    assert deploy.snapshot_steps() == {"_capture"}, deploy.snapshot_openers


@pytest.mark.parametrize(
    ("interrupted", "resumed_from", "rolled_on_resume", "staged_again"),
    [
        pytest.param(
            CONTROLLER_STAGE,
            "rollback-controller-staging",
            {"gpu-a-context", "gpu-eks-context"},
            True,
            id="in-the-controller-stage",
        ),
        pytest.param(
            "gpu-a-context",
            "rollback-data-restoring",
            {"gpu-a-context"},
            False,
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
    staged_again: bool,
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
    pins = _config_core_pins(release)
    assert _staged_pins(deploy) == [pins], (
        "the controller stage staged the previous -config-core's pins"
    )

    resumed = ROLLOUT.RegionalRelease(release.config, runner)  # type: ignore[arg-type]
    again = _bind(resumed, runner, monkeypatch, store=deploy.store)
    before = len(runner.scripts)

    run_resume(resumed)

    assert (again.phases[0], again.phases[-1]) == (resumed_from, "rolled-back"), (
        again.phases
    )
    assert "rollback-verified" in again.phases
    assert _staged_pins(again) == ([pins] if staged_again else []), (
        "a resumed controller stage stages the same pins from the persisted state"
    )
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


# --- a FULL release that moves a cluster's own Runtime Profile -------------------
#
# BOOT-020's FULL stage as the live site ran it (2026-11-08): gpu-eks parked on
# the site profile, gpu-a on its own profile with its node and Agent. The
# candidate suffixes both -- the site eks-v1 -> eks-v1-boot020, gpu-a's own
# hyperpod-v1 -> hyperpod-v1-boot020, each with a new Agent config digest --
# and its verification fails: the automatic rollback has to leave every pin as
# the previous release left it, and so does the record the next release starts
# from.

REPOSITORY = Path(__file__).resolve().parents[2]
SITE_PROFILE = "eks-v1"
SITE_DIGEST = "5" * 64
#: What gpu-a's Agent and node report (``EngineRunner``, ``_node``).
OWN_PROFILE = "hyperpod-v1"
OWN_DIGEST = "a" * 64
CANDIDATE_SITE_PROFILE = "eks-v1-boot020"
CANDIDATE_SITE_DIGEST = "6" * 64
CANDIDATE_OWN_PROFILE = "hyperpod-v1-boot020"
CANDIDATE_OWN_DIGEST = "7" * 64
CANDIDATE_PINS = (
    CANDIDATE_SITE_PROFILE,
    CANDIDATE_SITE_DIGEST,
    CANDIDATE_OWN_PROFILE,
    CANDIDATE_OWN_DIGEST,
)
#: The previous release's record of the clusters on their own profile.
PREVIOUS_CLUSTER_PROFILES = {
    HYPERPOD: {
        "version": OWN_PROFILE,
        "agent_config_digest": OWN_DIGEST,
        "policy_sha256": "0" * 64,
    }
}
PROFILE_ENV = "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION"
FULL_DIFF = ReleaseDiff(
    kind=ReleaseChangeKind.FULL,
    changed=frozenset(
        {
            "agent_config",
            "cluster_runtime_profiles",
            "runtime_profile_version",
            "executor_wheel",
            "node_runtime_wheel",
            "node_bundle",
        }
    ),
)
FULL_SITES = [
    pytest.param("nodes", id="gpu-a-with-its-agent"),
    pytest.param("parked", id="gpu-a-parked-too"),
]


def _registry_data(entries: list[dict[str, Any]]) -> dict[str, str]:
    return {
        REGISTRY.REGISTRY_CURRENT_KEY: base64.b64encode(
            json.dumps(entries).encode()
        ).decode()
    }


def _previous_registry(config: Any) -> list[dict[str, Any]]:
    """The registry Secret as the previous release left it: gpu-a registered
    with its own previous profile, gpu-eks on the site profile (no pins)."""

    entries = []
    for target in config.clusters:
        entry = REGISTRY.registry_entry(target)
        entry.pop("runtime_profile_version", None)
        entry.pop("agent_config_digest", None)
        if target.cluster_id == HYPERPOD:
            entry.update(
                runtime_profile_version=OWN_PROFILE, agent_config_digest=OWN_DIGEST
            )
        entries.append(entry)
    return sorted(entries, key=lambda item: str(item["cluster_id"]))


def _full_engine(tmp_path: Path, *, hyperpod: str) -> tuple[Any, EngineRunner]:
    """The FULL candidate's release over a site the previous release left."""

    clusters = _clusters()
    for cluster in clusters:
        token = tmp_path / f"{cluster['cluster_id']}.token"
        token.write_text("t" * 32, encoding="utf-8")
        cluster["token_file"] = str(token)
    clusters[0]["runtime_profile"] = {
        "source": str(
            REPOSITORY / "config/runtime-profile.regional-hyperpod-safe.example.yaml"
        ),
        "version": CANDIDATE_OWN_PROFILE,
        "agent_config_digest": CANDIDATE_OWN_DIGEST,
    }
    path = config_file(
        tmp_path, clusters=clusters, profile_version=CANDIDATE_SITE_PROFILE
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    value["runtime_profile"]["source"] = str(
        REPOSITORY / "config/runtime-profile.regional-eks-safe.example.yaml"
    )
    value["runtime_profile"]["registration_cluster_id"] = EKS
    value["release"]["agent_config_digest"] = CANDIDATE_SITE_DIGEST
    path.write_text(json.dumps(value), encoding="utf-8")
    config = ROLLOUT.ReleaseConfig.load(path)
    runner = EngineRunner(hyperpod=hyperpod)
    runner.registry = _registry_data(_previous_registry(config))
    return ROLLOUT.RegionalRelease(config, runner), runner  # type: ignore[arg-type]


def _previous_release() -> dict[str, Any]:
    """What the release the candidate replaces recorded: the site on eks-v1,
    gpu-a on its own hyperpod-v1, each with its Agent config digest; every
    CPU role's ``-config-core`` and container env as they ran."""

    return {
        "metadata": {
            "required-agent-artifact-sha256": "1" * 64,
            "required-agent-compatibility-digest": "1" * 64,
            "required-agent-protocol-version": "3",
            "required-agent-config-digest": SITE_DIGEST,
            "required-regional-executor-artifact-sha256": "2" * 64,
            "required-regional-executor-compatibility-digest": "2" * 64,
            "required-regional-executor-protocol-version": "2",
            "required-node-action-key-version": "2",
        },
        "runtime_profile_version": SITE_PROFILE,
        "cluster_runtime_profiles": json.loads(json.dumps(PREVIOUS_CLUSTER_PROFILES)),
        "cpu_role_config_maps": {
            f"{deployment}-config-core": {
                "GPU_FAULT_PROCESSOR_WORKERS": "24",
                "GPU_FAULT_REQUIRED_AGENT_VERSION": "0.10.0",
                "GPU_FAULT_REQUIRED_POLICY_VERSION": "policy-v1",
                PROFILE_ENV: SITE_PROFILE,
            }
            for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS
        },
        "cpu_role_container_env": {
            deployment: {
                "app": {
                    "env": [{"name": "GPU_FAULT_ROLE_NAME", "value": deployment}],
                    "envFrom": [
                        {"configMapRef": {"name": f"{deployment}-config-core"}}
                    ],
                }
            }
            for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS
        },
    }


def _full_rollback_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, hyperpod: str
) -> tuple[Any, EngineRunner, Deploy]:
    release, runner = _full_engine(tmp_path, hyperpod=hyperpod)
    deploy = _bind(release, runner, monkeypatch, state_writer=True, cpu_rollback=True)
    _fail_verification(release, runner, deploy, monkeypatch, **_previous_release())
    monkeypatch.setattr(
        ORCHESTRATION,
        "ensure_runtime_profile",
        lambda target, **_k: deploy.record("runtime-profiles-registered"),
    )
    return release, runner, deploy


def _pins(scripts: list[dict[str, str]], context: str) -> set[tuple[str, str]]:
    """The (Runtime Profile, Agent config digest) each Reconciler run on
    ``context`` installs."""

    return {
        (env["GPU_FAULT_RUNTIME_PROFILE"], env["GPU_FAULT_INSTALLER_CONFIG_DIGEST"])
        for env in scripts
        if env.get("GPU_FAULT_KUBECTL_CONTEXT") == context
    }


def _recorded(deploy: Deploy, name: str) -> list[Any]:
    return [value for recorded, value in deploy.calls if recorded == name]


def _next_release_snapshot(
    release: Any, deploy: Deploy, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """What the next transaction records as its previous release over the
    rolled-back site: the real capture of the persisted state, scoped to the
    record (a plan with no GPU or CPU component reads nothing else)."""

    monkeypatch.setattr(release, "_remote_command_stats", lambda: {}, raising=False)
    monkeypatch.setattr(release, "_deployment_snapshot_enabled", False)
    release.state = json.loads(json.dumps(deploy.store))
    return STATE.capture_previous(
        release, ReleaseExecutionPlan(nodes=(ReleaseComponent.VERIFY,))
    )


def _assert_restored_to_the_previous_release(
    release: Any,
    runner: EngineRunner,
    deploys: tuple[Deploy, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``deploys``: every deploy-host process the rollback ran in, in order."""

    # The control plane's per-cluster pins: the registry backup is back.
    assert runner.registry_payload(REGISTRY.REGISTRY_CURRENT_KEY) == (
        _previous_registry(release.config)
    ), "gpu-a's registry entry carries its own previous profile and digest again"
    assert runner.registry_payload(REGISTRY.REGISTRY_BACKUP_KEY) is None
    # The site pins every CPU role hot-loads or starts from: the controller
    # patch, both CPU stages' render environment, the role ConfigMaps and the
    # container env -- the previous release's, never a candidate pin.
    patches = [
        json.loads(command[command.index("-p") + 1])["data"]
        for command in runner.commands
        if "patch" in command and any(item.endswith("-config-core") for item in command)
    ]
    assert patches and all(patch[PROFILE_ENV] == SITE_PROFILE for patch in patches)
    applies = [
        value for deploy in deploys for value in _recorded(deploy, "rollback-cpu-apply")
    ]
    assert len(applies) == 2, "the controller stage and the CPU restore render"
    for apply in applies:
        environment = apply["environment"]
        assert environment[PROFILE_ENV] == SITE_PROFILE
        assert environment["GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST"] == SITE_DIGEST
        assert environment["GPU_FAULT_WHEEL_CONFIGMAP"] == "previous-cpu-wheel"
        leaked = sorted(
            name
            for name, value in environment.items()
            if os.environ.get(name) != value
            and any(pin in value for pin in CANDIDATE_PINS)
        )
        assert not leaked, f"candidate pins reach the CPU templates: {leaked}"
    previous = _previous_release()
    assert applies[-1]["environment"]["GPU_FAULT_RUNTIME_IMAGE"] == PREVIOUS_IMAGE
    assert applies[-1]["container_env"] == previous["cpu_role_container_env"], (
        "the CPU restore renders the previous container env verbatim"
    )
    restored_maps = [
        value
        for deploy in deploys
        for value in _recorded(deploy, "rollback-role-config-maps")
    ]
    assert restored_maps == [previous["cpu_role_config_maps"]]
    # The rolled-back state still names the failed candidate (the retry needs
    # it) and keeps the previous release's own record beside it ...
    deploy = deploys[-1]
    stored = deploy.store
    assert (stored["phase"], stored["rollback_result"]["status"]) == (
        "rolled-back",
        "PASSED",
    )
    assert stored["cluster_runtime_profiles"][HYPERPOD]["version"] == (
        CANDIDATE_OWN_PROFILE
    )
    assert stored["previous"]["cluster_runtime_profiles"] == PREVIOUS_CLUSTER_PROFILES
    # ... and the next release, starting over this site, records what runs: a
    # rollback of that release restores gpu-a to hyperpod-v1, not the failed
    # candidate's hyperpod-v1-boot020 (live 2026-11-08, BOOT-020 FULL).
    snapshot = _next_release_snapshot(release, deploy, monkeypatch)
    assert snapshot["cluster_runtime_profiles"] == PREVIOUS_CLUSTER_PROFILES
    assert snapshot["runtime_profile_version"] == SITE_PROFILE


@pytest.mark.parametrize("hyperpod", FULL_SITES)
def test_a_full_rollback_restores_a_clusters_own_runtime_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hyperpod: str
) -> None:
    release, runner, deploy = _full_rollback_site(
        tmp_path, monkeypatch, hyperpod=hyperpod
    )

    with pytest.raises(ROLLOUT.ReleaseError, match="forced verification failure"):
        release.upgrade(diff=FULL_DIFF)

    assert deploy.phases[-2:] == ["rolled-back", "rolled-back"], deploy.phases
    assert "profile-ready" in deploy.phases
    # The forward judged the transition of each cluster from the profile it
    # ran: gpu-a's own, gpu-eks the site's.
    assert {(item["previous"], item["desired"]) for item in runner.transitions} == {
        (SITE_PROFILE, CANDIDATE_SITE_PROFILE),
        (OWN_PROFILE, CANDIDATE_OWN_PROFILE),
    }
    (forward_runs,) = _recorded(deploy, "forward-scripts")
    forward, rollback = runner.scripts[:forward_runs], runner.scripts[forward_runs:]
    assert _pins(forward, "gpu-a-context") == {
        (CANDIDATE_OWN_PROFILE, CANDIDATE_OWN_DIGEST)
    }
    assert _pins(forward, "gpu-eks-context") == {
        (CANDIDATE_SITE_PROFILE, CANDIDATE_SITE_DIGEST)
    }
    # gpu-a's Agents (or, parked, its Reconciler alone) go back to its own
    # previous profile and digest; gpu-eks's Reconciler to the site's.
    assert _pins(rollback, "gpu-a-context") == {(OWN_PROFILE, OWN_DIGEST)}
    assert _pins(rollback, "gpu-eks-context") == {(SITE_PROFILE, SITE_DIGEST)}
    assert sorted(
        value for value in _recorded(deploy, "deployments") if value[1] is not None
    ) == [(HYPERPOD, OWN_PROFILE), (EKS, SITE_PROFILE)], (
        "the Executor, Watcher and Collector of each cluster render its previous "
        "profile"
    )
    _assert_restored_to_the_previous_release(release, runner, (deploy,), monkeypatch)


def test_a_resumed_full_rollback_restores_a_clusters_own_runtime_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The automatic rollback stops in gpu-a's data restore; a new deploy-host
    process resumes it from the persisted state alone and still puts gpu-a's
    own previous profile back, its record included."""

    release, runner, deploy = _full_rollback_site(
        tmp_path, monkeypatch, hyperpod="nodes"
    )
    runner.interrupt.add("gpu-a-context")

    with pytest.raises(ROLLOUT.ReleaseError, match="automatic rollback also failed"):
        release.upgrade(diff=FULL_DIFF)
    assert deploy.phases[-1] == "rollback-failed", deploy.phases
    assert not runner.interrupt, "the interruption happened"

    resumed = ROLLOUT.RegionalRelease(release.config, runner)  # type: ignore[arg-type]
    again = _bind(
        resumed,
        runner,
        monkeypatch,
        store=deploy.store,
        state_writer=True,
        cpu_rollback=True,
    )
    before = len(runner.scripts)

    run_resume(resumed)

    assert (again.phases[0], again.phases[-1]) == (
        "rollback-data-restoring",
        "rolled-back",
    ), again.phases
    assert _pins(runner.scripts[before:], "gpu-a-context") == {
        (OWN_PROFILE, OWN_DIGEST)
    }, "the resumed restore rolls gpu-a back to its own previous pins"
    _assert_restored_to_the_previous_release(
        resumed, runner, (deploy, again), monkeypatch
    )
