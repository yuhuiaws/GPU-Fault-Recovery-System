"""``deploy --rollback`` of a COMMITTED release, through the real engine.

``test_zero_node_deploy_engine``'s site -- ``gpu-eks`` parked on the site
profile, ``gpu-a`` on its own profile with its node and Agent, or parked too --
runs a FULL release that moves both profiles (the site ``eks-v1`` ->
``eks-v1-boot020``, ``gpu-a``'s own ``hyperpod-v1`` -> ``hyperpod-v1-boot020``,
each with a new Agent config digest) and passes its verification: the
regional registry is staged with the candidate's per-cluster pins, published,
and committed, and the commit drops the registry backup a rollback restores
from. A new deploy-host process then runs what ``gpu-fault-admin deploy
--rollback`` starts -- the engine's manual ``rollback`` with the real state
checkpoints and both CPU stages up to the role-split apply -- and the
administrator's alignment after it (``align_rolled_back_management``:
``membership-baseline.json``, then ``sync-state``).

Only what leaves the process is scripted, as there. The registry Secret is
emulated (``create --dry-run`` renders, ``apply`` persists, ``get`` reads) and
every publish lands in a control-plane store whose ``FleetRegistry`` is the
one that holds each cluster's Agents to their pins.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault.admin.rollback_command import (
    align_rolled_back_management,
    rollback_refusal,
)
from gpu_fault.fleet import (
    FleetCompatibilityPolicy,
    FleetRegistry,
    SignedAgentHeartbeat,
)
from gpu_fault.regional import RegionalClusterRegistration, RegionalRegistryRevision
from gpu_fault.store import InMemoryStore
from gpu_fault.store.shared.errors import NotFoundError
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import regional_release_online_registry as ONLINE
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release import regional_release_registry as REGISTRY
from gpu_fault_release import regional_release_rollback_context as ROLLBACK_CONTEXT
from gpu_fault_release import regional_release_state as STATE
from gpu_fault_release import rollout as ROLLOUT
from gpu_fault_release.regional_release_diff import (
    ReleaseChangeKind,
    ReleaseComponent,
    ReleaseDiff,
    ReleaseExecutionPlan,
)

# The real CPU stages of a rollback; the shared harness stubs both.
from gpu_fault_release.regional_release_orchestration import (
    _restore_rollback_cpu as restore_rollback_cpu,
)
from gpu_fault_release.regional_release_orchestration import (
    _stage_rollback_controller as stage_rollback_controller,
)
from gpu_fault_release.regional_release_profile_model import cluster_runtime_profiles
from tests.fleet._support import ARTIFACT, NOW, SECRET, heartbeat, signed
from tests.regional._release_orchestrator_support import config_file
from tests.regional._resource_probe_fakes import resource_probe_result
from tests.regional.test_zero_node_deploy_engine import (
    EKS,
    HYPERPOD,
    NODE,
    PREVIOUS_IMAGE,
    SELECTOR,
    Deploy,
    EngineRunner,
    _bind,
    _capture,
    _clusters,
)

REPOSITORY = Path(__file__).resolve().parents[2]
SITE_PROFILE = "eks-v1"
SITE_DIGEST = "5" * 64
#: What gpu-a's Agent and node report (``EngineRunner``): its own profile.
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
#: A node whose Agent still ran the candidate's own pins.
STALE_NODE = "hyperpod-i-2"
PREVIOUS_RELEASE = "previous-release"
PREVIOUS_DELIVERY = "d" * 64
PROFILE_ENV = "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION"
DIGEST_ENV = "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST"
REGISTRY_RESTORED = "registry-restored"
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
#: A release that moves no profile: new Executor and Agent wheels only.
WHEELS_DIFF = ReleaseDiff(
    kind=ReleaseChangeKind.DATA_PLANE_COMPATIBLE,
    changed=frozenset({"executor_wheel", "node_runtime_wheel", "node_bundle"}),
)
SITES = [
    pytest.param("nodes", id="gpu-a-with-its-agent"),
    pytest.param("parked", id="gpu-a-parked-too"),
]


class Site(EngineRunner):
    """``EngineRunner`` with the regional cluster registry Secret
    (``registry``: its data), ConfigMap merge patches applied to
    ``configmaps``, and the control plane's registry store every publish lands
    in (``control_plane``; ``published``: the reasons); no workflow or
    workload holds a Runtime Profile a release leaves."""

    def __init__(self, *, hyperpod: str) -> None:
        super().__init__(hyperpod=hyperpod)
        self.registry: dict[str, str] = {}
        self.registry_writes = 0
        self.control_plane = InMemoryStore()
        self.published: list[str] = []

    def entries(self, key: str = REGISTRY.REGISTRY_CURRENT_KEY) -> Any:
        encoded = self.registry.get(key)
        return None if encoded is None else json.loads(base64.b64decode(encoded))

    def answer_probe(self, name: str, input_text: str | None) -> str:
        if name == "runtime_profile_transition":
            return json.dumps({"workflow_count": 0, "workload_count": 0})
        if name == "registry_publish_converge":
            return self._publish(json.loads(input_text or "{}"))
        return super().answer_probe(name, input_text)

    def _publish(self, request: dict[str, Any]) -> str:
        """The registry API: the revision the release sends becomes the head."""

        payload = request["payload"]
        store = self.control_plane
        try:
            generation = store.get_regional_registry_head().generation
        except NotFoundError:
            generation = 0
        revision = RegionalRegistryRevision.build(
            generation=generation + 1,
            registrations=[
                RegionalClusterRegistration.model_validate(item)
                for item in payload["registrations"]
            ],
            previous_generation=generation or None,
            required_member_ids=[],
            reason=payload["reason"],
        )
        head = store.publish_regional_registry_revision(
            revision, expected_generation=generation
        )
        with self.lock:
            self.published.append(payload["reason"])
        return json.dumps({"generation": head.generation, "converged": True})

    def run(self, arguments: list[str], **kwargs: Any) -> str:
        command = [str(item) for item in arguments]
        answer = self._registry_secret(command, kwargs.get("input_text"))
        if answer is None and "patch" in command and "configmap" in command:
            # A merge patch of a ConfigMap (the rollback's ``-config-core``).
            name = command[command.index("configmap") + 1]
            data = json.loads(command[command.index("-p") + 1]).get("data") or {}
            with self.lock:
                self.configmaps.setdefault(name, {}).update(data)
            answer = ""
        if answer is None:
            return super().run(arguments, **kwargs)
        with self.lock:
            self.commands.append(command)
        return answer

    def _registry_secret(self, arguments: list[str], input_text: Any) -> str | None:
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
        if "apply" not in arguments or not isinstance(input_text, str):
            return None
        try:
            document = json.loads(input_text)
        except json.JSONDecodeError:
            return None
        if (document.get("metadata") or {}).get("name") != REGISTRY.REGISTRY_SECRET:
            return None
        with self.lock:
            self.registry = dict(document["data"])
            self.registry_writes += 1
        return ""

    def probe_output(
        self, arguments: list[str], *, timeout_seconds: float
    ) -> tuple[int, str, str]:
        if "secret" not in arguments or REGISTRY.REGISTRY_SECRET not in arguments:
            return super().probe_output(arguments, timeout_seconds=timeout_seconds)
        with self.lock:
            self.commands.append([str(item) for item in arguments])
            data = dict(self.registry)
        if not data:
            return 0, "", ""
        code, raw, error = resource_probe_result(list(arguments))
        return code, json.dumps({**json.loads(raw), "data": data}), error


def _config(directory: Path, *, site: tuple[str, str], own: tuple[str, str]) -> Any:
    """The release config of the site: the site profile and Agent config digest
    ``site`` (gpu-eks runs it), gpu-a on its own ``own``."""

    directory.mkdir()
    clusters = _clusters()
    for cluster in clusters:
        token = directory.parent / f"{cluster['cluster_id']}.token"
        token.write_text("t" * 32, encoding="utf-8")
        cluster["token_file"] = str(token)
    clusters[0]["runtime_profile"] = {
        "source": str(
            REPOSITORY / "config/runtime-profile.regional-hyperpod-safe.example.yaml"
        ),
        "version": own[0],
        "agent_config_digest": own[1],
    }
    path = config_file(directory, clusters=clusters, profile_version=site[0])
    value = json.loads(path.read_text(encoding="utf-8"))
    value["runtime_profile"]["source"] = str(
        REPOSITORY / "config/runtime-profile.regional-eks-safe.example.yaml"
    )
    value["runtime_profile"]["registration_cluster_id"] = EKS
    value["release"]["agent_config_digest"] = site[1]
    path.write_text(json.dumps(value), encoding="utf-8")
    return ROLLOUT.ReleaseConfig.load(path)


def _registry_data(config: Any) -> dict[str, str]:
    """The registry Secret a release of ``config`` committed."""

    entries = sorted(
        (REGISTRY.registry_entry(target) for target in config.clusters),
        key=lambda item: str(item["cluster_id"]),
    )
    return {
        REGISTRY.REGISTRY_CURRENT_KEY: base64.b64encode(
            json.dumps(entries).encode()
        ).decode()
    }


def _previous_release(previous_config: Any) -> dict[str, Any]:
    """What the release the candidate replaces recorded: the site on eks-v1,
    gpu-a on its own hyperpod-v1, each with its Agent config digest; every
    CPU role's ``-config-core`` and container env as they ran."""

    return {
        "release_id": PREVIOUS_RELEASE,
        "release_delivery_sha256": PREVIOUS_DELIVERY,
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
        "runtime_image": PREVIOUS_IMAGE,
        "live_runtime_image": PREVIOUS_IMAGE,
        "runtime_profile_version": SITE_PROFILE,
        "cluster_runtime_profiles": cluster_runtime_profiles(previous_config),
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


class Engine:
    """The site, the candidate's and the previous release's configs, and every
    deploy-host process (``Deploy``) that ran, in order."""

    def __init__(
        self, tmp_path: Path, *, hyperpod: str, moves_profiles: bool = True
    ) -> None:
        self.previous_config = _config(
            tmp_path / "previous",
            site=(SITE_PROFILE, SITE_DIGEST),
            own=(OWN_PROFILE, OWN_DIGEST),
        )
        self.config = (
            _config(
                tmp_path / "candidate",
                site=(CANDIDATE_SITE_PROFILE, CANDIDATE_SITE_DIGEST),
                own=(CANDIDATE_OWN_PROFILE, CANDIDATE_OWN_DIGEST),
            )
            if moves_profiles
            else self.previous_config
        )
        self.site = Site(hyperpod=hyperpod)
        self.site.registry = _registry_data(self.previous_config)
        # The previous release's live release metadata and role ConfigMaps.
        recorded = _previous_release(self.previous_config)
        self.site.configmaps["gpu-fault-release-metadata"] = dict(recorded["metadata"])
        for name, data in recorded["cpu_role_config_maps"].items():
            self.site.configmaps[name] = dict(data)
        self.deploys: list[Deploy] = []

    @property
    def store(self) -> dict[str, Any]:
        return self.deploys[-1].store

    def process(self, monkeypatch: pytest.MonkeyPatch, config: Any = None) -> Any:
        """A new deploy-host process: a release bound to the site and to the
        state the earlier processes persisted."""

        release = ROLLOUT.RegionalRelease(config or self.config, self.site)  # type: ignore[arg-type]
        self.deploys.append(
            _bind_engine(
                release,
                self.site,
                monkeypatch,
                previous=_previous_release(self.previous_config),
                store=self.deploys[-1].store if self.deploys else None,
            )
        )
        return release


def _bind_engine(
    release: Any,
    site: Site,
    monkeypatch: pytest.MonkeyPatch,
    *,
    previous: dict[str, Any],
    store: dict[str, Any] | None,
) -> Deploy:
    """The shared harness, with the real state checkpoints, the registry
    staged, published and committed for real, and both CPU stages of a
    rollback run up to the role-split render and apply (recorded with the
    environment and container env snapshot they were handed)."""

    deploy = _bind(release, site, monkeypatch, store=store)

    def method(name: str, value: Any) -> None:
        monkeypatch.setattr(release, name, value, raising=False)

    monkeypatch.setattr(
        STATE,
        "persist_state",
        lambda target: json.dumps(target.state, default=str, sort_keys=True),
    )
    monkeypatch.setattr(STATE, "record_release_history", lambda *_a, **_k: None)

    def save_state(phase: str, **updates: Any) -> None:
        with STATE.state_transaction(release):
            STATE.save_state(release, phase, **updates)
            with deploy.lock:
                deploy.phases.append(phase)
                deploy.store.clear()
                deploy.store.update(json.loads(json.dumps(release.state, default=str)))

    method("_save_state", save_state)

    def capture(plan: Any = None) -> dict[str, Any]:
        deploy.previous = {**_capture(release, deploy), **previous}
        return deploy.previous

    def apply_cpu(**kwargs: Any) -> None:
        # The CPU apply renders the release's site profile into -config-core.
        deploy.record("cpu", bool(kwargs.get("finalize")))
        for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS:
            site.configmaps[f"{deployment}-config-core"][PROFILE_ENV] = str(
                release.config.runtime_profile_version
            )

    def restore_role_config_maps(snapshots: Any) -> bool:
        deploy.record("rollback-role-config-maps", snapshots)
        for name, data in (snapshots or {}).items():
            site.configmaps[name] = dict(data)
        return bool(snapshots)

    method("_capture_previous", capture)
    method("_apply_cpu", apply_cpu)
    method("_stage_registry", lambda: REGISTRY.stage_registry(release))
    method("_publish_staged_registry", lambda: ONLINE.publish_staged_registry(release))
    method(
        "_publish_restored_registry", lambda: ONLINE.publish_restored_registry(release)
    )
    method("_commit_registry_update", lambda: REGISTRY.commit_registry_update(release))
    method(
        "_delete_stale_release_secret_backups",
        lambda previous: deploy.record("commit-backup-sweep"),
    )
    method(
        "_apply_gpu_deployments",
        lambda target, *_a, **kwargs: deploy.record(
            "deployments", (target.cluster_id, kwargs.get("runtime_profile_version"))
        ),
    )
    monkeypatch.setattr(
        ORCHESTRATION,
        "ensure_runtime_profile",
        lambda target, **_k: deploy.record("runtime-profiles-registered"),
    )
    monkeypatch.setattr(
        ORCHESTRATION, "_stage_rollback_controller", stage_rollback_controller
    )
    monkeypatch.setattr(ORCHESTRATION, "_restore_rollback_cpu", restore_rollback_cpu)

    def apply_rollback_cpu(_release: Any, environment: dict[str, str]) -> None:
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

    monkeypatch.setattr(
        ORCHESTRATION, "_apply_rollback_cpu_environment", apply_rollback_cpu
    )
    method("_restore_cpu_role_config_maps", restore_role_config_maps)
    method("_config_map_sha", lambda *_a, **_k: "4" * 64)
    return deploy


def _fleet(site: Site) -> FleetRegistry:
    """The control plane's Agent registry over the registry head the releases
    published, holding the site to the previous release's site pins."""

    return FleetRegistry(
        site.control_plane,
        SECRET,
        FleetCompatibilityPolicy(
            required_agent_version="0.9.0",
            required_artifact_sha256=ARTIFACT,
            required_policy_version="catalog-a",
            required_runtime_profile_version=SITE_PROFILE,
            required_config_digest=SITE_DIGEST,
        ),
        now=lambda: NOW,
    )


def _agent(node: str, profile: str, digest: str) -> SignedAgentHeartbeat:
    """gpu-a's Agent on ``node`` running ``profile`` and Agent config ``digest``."""

    return signed(
        heartbeat(node).model_copy(
            update={
                "cluster_id": HYPERPOD,
                "runtime_profile_version": profile,
                "config_digest": digest,
            }
        )
    )


def _judged_pins(site: Site) -> dict[str, tuple[str | None, str | None]]:
    fleet = _fleet(site)
    return {
        cluster_id: (
            fleet.cluster_policy(cluster_id).required_runtime_profile_version,
            fleet.cluster_policy(cluster_id).required_config_digest,
        )
        for cluster_id in (HYPERPOD, EKS)
    }


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


def _restart_reasons(state: Mapping[str, Any], stage: str) -> set[str]:
    """Every reason the rollback's ``stage`` restarted a CPU tier for."""

    phase = ((state.get("rollback_timing") or {}).get("phases") or {}).get(stage)
    details = (phase or {}).get("details") or {}
    restarted = (details.get("cpu_restart") or {}).get("restarted") or {}
    return {reason for reasons in restarted.values() for reason in reasons}


def _commit(engine: Engine, monkeypatch: pytest.MonkeyPatch, diff: ReleaseDiff) -> None:
    """The deploy: the release, its verification and its commit."""

    release = engine.process(monkeypatch)
    release.upgrade(diff=diff)
    release.commit_release()
    store = engine.store
    assert (store["phase"], store["transaction_committed"]) == ("complete", True)
    assert store["commit_cleanup_completed"] is True
    assert rollback_refusal(store) is None, "deploy --rollback accepts this state"


def _manual_rollback(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> int:
    """What ``deploy --rollback`` starts, in a new process; returns how many
    Reconciler runs the forward made."""

    forward = len(engine.site.scripts)
    engine.process(monkeypatch).rollback()
    store = engine.store
    assert (store["phase"], store["rollback_result"]["status"]) == (
        "rolled-back",
        "PASSED",
    )
    return forward


def _align(
    engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """The administrator's alignment after the engine rolled back: the
    management site from the attempt's rollback baseline, then ``sync-state``
    under the restored release (a new process). Returns the membership
    evidence it recorded."""

    site_file, prepared = _management_site(tmp_path, engine.store["release_id"])
    synced: list[str] = []

    def run_release_mode(
        _site_file: Path, *, mode: str, root: Path, environment: Mapping[str, str]
    ) -> None:
        synced.append(mode)
        # The aligned site renders the restored release: its config and image.
        monkeypatch.setenv("GPU_FAULT_RUNTIME_IMAGE", PREVIOUS_IMAGE)
        ROLLOUT.sync_release_state(engine.process(monkeypatch, engine.previous_config))

    align_rolled_back_management(
        prepared,
        site_file=site_file,
        root=REPOSITORY,
        environment={},
        live_state=json.loads(json.dumps(engine.store)),
        run_release_mode=run_release_mode,
    )
    assert synced == ["sync-state"]
    evidence = (
        site_file.parent / "rollback-management" / PREVIOUS_RELEASE
    ) / "membership-baseline.json"
    return dict(json.loads(evidence.read_text(encoding="utf-8")))


def _management_site(tmp_path: Path, release_id: str) -> tuple[Path, Any]:
    """The administrator's state directory as the committed deploy left it:
    ``site.yaml`` names the candidate, the attempt's rollback baseline the
    previous site, next to the previous release's immutable manifest and
    Runtime Profile snapshot."""

    state_dir = tmp_path / "state"
    repository = tmp_path / "repository"
    repository.mkdir()
    dist = (
        state_dir / "source-snapshots" / "snapshot-a" / "repository-a" / "dist"
    ) / PREVIOUS_RELEASE
    dist.mkdir(parents=True)
    for name in ("runtime.whl", "bundle.tar.gz"):
        (dist / name).write_bytes(name.encode())
    (dist / "release.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "release_id": PREVIOUS_RELEASE,
                "wheel": f"dist/{PREVIOUS_RELEASE}/runtime.whl",
                "bundle": f"dist/{PREVIOUS_RELEASE}/bundle.tar.gz",
                "delivery": {
                    "sha256": PREVIOUS_DELIVERY,
                    "images": {"runtime": {"reference": PREVIOUS_IMAGE}},
                },
            }
        ),
        encoding="utf-8",
    )
    profile = state_dir / "profiles" / f"{SITE_PROFILE}.yaml"
    profile.parent.mkdir()
    profile.write_text(f"profile_version: {SITE_PROFILE}\n", encoding="utf-8")

    def document(site: str, own: str) -> str:
        return yaml.safe_dump(
            {
                "apiVersion": "gpu-fault.aws/v1alpha1",
                "kind": "RegionalSite",
                "metadata": {"name": "test-site"},
                "spec": {
                    "repositoryRoot": str(repository),
                    "release": {"manifest": "dist/current-release.json"},
                    "runtimeProfile": {"source": str(profile), "version": site},
                    "images": {"runtime": PREVIOUS_IMAGE},
                    "clusters": [
                        {"clusterId": HYPERPOD, "runtimeProfile": {"version": own}},
                        {
                            "clusterId": EKS,
                            "provider": "eks-managed",
                            "nodeSelector": SELECTOR,
                        },
                    ],
                },
            },
            sort_keys=False,
        )

    site_file = state_dir / "site.yaml"
    site_file.write_text(
        document(CANDIDATE_SITE_PROFILE, CANDIDATE_OWN_PROFILE), encoding="utf-8"
    )
    attempt = state_dir / "release-deploy" / release_id
    attempt.mkdir(parents=True)
    (attempt / "site.before.yaml").write_text(
        document(SITE_PROFILE, OWN_PROFILE), encoding="utf-8"
    )
    return site_file, SimpleNamespace(release_id=release_id, state_dir=attempt)


def _next_capture(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """What the next transaction records as its previous release: the real
    capture of the persisted state, scoped to the record (a plan with no GPU
    or CPU component reads nothing else)."""

    release = engine.process(monkeypatch)
    monkeypatch.setattr(release, "_remote_command_stats", lambda: {}, raising=False)
    monkeypatch.setattr(release, "_deployment_snapshot_enabled", False)
    release.state = json.loads(json.dumps(engine.store))
    return STATE.capture_previous(
        release, ReleaseExecutionPlan(nodes=(ReleaseComponent.VERIFY,))
    )


def _own_pins(profiles: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    return {
        cluster_id: (str(entry["version"]), str(entry["agent_config_digest"]))
        for cluster_id, entry in profiles.items()
    }


@pytest.mark.parametrize("hyperpod", SITES)
def test_a_manual_rollback_of_a_committed_release_restores_a_clusters_registry_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hyperpod: str
) -> None:
    engine = Engine(tmp_path, hyperpod=hyperpod)
    site = engine.site
    previous_registry = site.entries()

    _commit(engine, monkeypatch, FULL_DIFF)

    # The commit dropped the backup the automatic rollback restores from, and
    # the control plane holds gpu-a's Agents to the candidate's own pins.
    assert site.entries(REGISTRY.REGISTRY_BACKUP_KEY) is None
    assert site.entries() == json.loads(
        base64.b64decode(_registry_data(engine.config)[REGISTRY.REGISTRY_CURRENT_KEY])
    )
    assert _judged_pins(site)[HYPERPOD] == (CANDIDATE_OWN_PROFILE, CANDIDATE_OWN_DIGEST)

    forward_runs = _manual_rollback(engine, monkeypatch)

    # The registry entry: gpu-a registered with its own previous profile and
    # digest again, gpu-eks with none; no backup left behind.
    assert site.entries() == previous_registry
    assert site.entries(REGISTRY.REGISTRY_BACKUP_KEY) is None
    assert site.published[-1].endswith("registry restored by rollback"), site.published
    # The pins the control plane judges by: the head the rollback published.
    assert _judged_pins(site) == {
        HYPERPOD: (OWN_PROFILE, OWN_DIGEST),
        EKS: (SITE_PROFILE, SITE_DIGEST),
    }
    fleet = _fleet(site)
    fleet.register(_agent(NODE, OWN_PROFILE, OWN_DIGEST))
    fleet.register(_agent(STALE_NODE, CANDIDATE_OWN_PROFILE, CANDIDATE_OWN_DIGEST))
    assert fleet.readiness(HYPERPOD, [NODE]).ready, (
        "an Agent the restored Reconciler installs on gpu-a is ready"
    )
    assert not fleet.readiness(HYPERPOD, [STALE_NODE]).ready, (
        "an Agent left on the candidate's own pins is held back"
    )
    # The tiers restarted onto the restored registry in the controller stage;
    # the CPU restore found the pins registered and wrote nothing.
    rolled_back = engine.store
    assert REGISTRY_RESTORED in _restart_reasons(rolled_back, "controller_stage")
    assert REGISTRY_RESTORED not in _restart_reasons(rolled_back, "cpu_restore")
    # The Agents (parked: the Reconciler alone) and the Executor, Watcher and
    # Collector go back to each cluster's own previous pins.
    rollback_scripts = site.scripts[forward_runs:]
    assert _pins(rollback_scripts, "gpu-a-context") == {(OWN_PROFILE, OWN_DIGEST)}
    assert _pins(rollback_scripts, "gpu-eks-context") == {(SITE_PROFILE, SITE_DIGEST)}
    manual = engine.deploys[-1]
    assert sorted(
        value for value in _recorded(manual, "deployments") if value[1] is not None
    ) == [(HYPERPOD, OWN_PROFILE), (EKS, SITE_PROFILE)]
    _assert_previous_site_pins(site, manual)

    evidence = _align(engine, tmp_path, monkeypatch)

    # membership-baseline.json: both sides name gpu-a's own previous profile.
    assert evidence["committed_cluster_ids"] == sorted((HYPERPOD, EKS))
    assert evidence["baseline_cluster_profile_pins"] == {HYPERPOD: OWN_PROFILE}
    assert evidence["live_cluster_profile_pins"] == {HYPERPOD: OWN_PROFILE}
    # The next release starts from the restored pins: its rollback restores
    # gpu-a to hyperpod-v1 again.
    assert (engine.store["phase"], engine.store["previous"]) == ("complete", None)
    snapshot = _next_capture(engine, monkeypatch)
    assert _own_pins(snapshot["cluster_runtime_profiles"]) == {
        HYPERPOD: (OWN_PROFILE, OWN_DIGEST)
    }
    assert snapshot["runtime_profile_version"] == SITE_PROFILE


def _assert_previous_site_pins(site: Site, deploy: Deploy) -> None:
    """The site pins every CPU role hot-loads or starts from -- the controller
    patch and both CPU stages' render environment -- are the previous
    release's; no candidate pin reaches them."""

    patches = [
        json.loads(command[command.index("-p") + 1])["data"]
        for command in site.commands
        if "patch" in command and any(item.endswith("-config-core") for item in command)
    ]
    assert patches and all(patch[PROFILE_ENV] == SITE_PROFILE for patch in patches)
    applies = _recorded(deploy, "rollback-cpu-apply")
    assert len(applies) == 2, "the controller stage and the CPU restore render"
    for apply in applies:
        environment = apply["environment"]
        assert (environment[PROFILE_ENV], environment[DIGEST_ENV]) == (
            SITE_PROFILE,
            SITE_DIGEST,
        )
        leaked = sorted(
            name
            for name, value in environment.items()
            if os.environ.get(name) != value
            and any(pin in value for pin in CANDIDATE_PINS)
        )
        assert not leaked, f"candidate pins reach the CPU templates: {leaked}"


def test_a_manual_rollback_of_a_release_that_moved_no_profile_leaves_the_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New wheels only: the registry the release found is the one it committed,
    and the rollback neither rewrites nor republishes it."""

    engine = Engine(tmp_path, hyperpod="nodes", moves_profiles=False)
    site = engine.site
    registry = dict(site.registry)

    _commit(engine, monkeypatch, WHEELS_DIFF)
    writes, published = site.registry_writes, list(site.published)
    _manual_rollback(engine, monkeypatch)

    assert site.registry == registry
    assert (site.registry_writes, site.published) == (writes, published) == (0, [])
    for stage in ("controller_stage", "cpu_restore"):
        assert REGISTRY_RESTORED not in _restart_reasons(engine.store, stage), stage
