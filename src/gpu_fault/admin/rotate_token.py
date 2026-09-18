"""Rotate one GPU cluster's token as a single, resumable administrator verb.

REG-9 used to be eight manual steps: read the registry generation through an
``exec``, ``openssl rand``, a ``jq`` rewrite of the registry Secret, a control
plane restart, a GPU Secret patch, three Deployment restarts, a privileged
DaemonSet to rewrite every node's ``systemd`` env file, and two log greps. The
deploy host's own copy of the token (``secure/<cluster-id>.token``) was not one
of them, which is how a stale file broke a release on 2026-09-04.

The rotation rides the overlap window the registry already offers: the new
digest becomes ``token_sha256`` at once and the old one parks in
``retiring_token_sha256`` until ``token_rotation_expires_at``. Both open the
door until the window closes, so every data-plane mutation below happens while
the old credential is still valid, and a failure anywhere before acceptance
leaves a working fleet.

State machine (``<state-dir>/rotate-token/<cluster-id>/state.json``)::

    PREPARED -> REGISTRY_OVERLAP_PUBLISHED -> CONNECTION_SECRET_UPDATED
      -> DATA_PLANE_ROLLED -> NODES_ROLLED -> DATA_PLANE_ACCEPTED
      -> TOKEN_FILE_WRITTEN -> RETIRING_TOKEN_DROPPED

Re-running with the same arguments resumes at the first unfinished step. The
new token lives only in a 0600 file under the state directory until the data
plane has proven it authenticates with it; it is never printed and never put
on a command line. The site's token file is rewritten only after that proof,
atomically, with the old value kept as ``<file>.retired-<timestamp>``.
``--rollback`` walks the data plane back onto the old token and republishes
the old-only registry while the window still accepts both.

Node delivery reuses the release's fleet-wave machinery -- the Agent lease and
open-command safety barrier, the Reconciler wave ConfigMap handoff and the
installer/heartbeat convergence wait -- but not a fleet deployment record: a
token change is not an identity change, so a record would be born SUCCEEDED.
The installer reads the token from the GPU connection Secret, so marking a
node ``installer-state=Retrying`` inside its wave is what carries the new
token onto the host.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, cast

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap import discover_cluster
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    CommandRunner,
    safe_name,
    write_secret,
)
from gpu_fault.admin.cluster_removal import resolve_cluster_id
from gpu_fault.admin.execution import run_command
from gpu_fault.admin.membership_lock import (
    administrator_operation_lock,
    reload_site_for_mutation,
)
from gpu_fault.admin.operator_identity import resolve_operator_identity
from gpu_fault.admin.release_state import live_release_state
from gpu_fault.admin.release_engine import build_release as build_release
from gpu_fault.admin.rotate_token_acceptance import (
    CPU_INGRESS_LABEL as CPU_INGRESS_LABEL,
    RETIRING_TOKEN_LOG_FRAGMENT as RETIRING_TOKEN_LOG_FRAGMENT,
    collect_retiring_token_authentications,
    wait_for_quiet_token_authentications,
)
from gpu_fault.admin.rotate_token_policy import (
    MAX_WINDOW_MINUTES,
    MIN_WINDOW_MINUTES,
    require_rotation_policy,
    rotation_policy,
)
from gpu_fault.admin.site import (
    RenderedSite,
    effective_environment,
)
from gpu_fault.fleet_deployment import FleetDeploymentRequest, deployment_waves
from gpu_fault.regional import cluster_token_sha256
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release.regional_admin_commands import (
    BOOTSTRAP_PHASES,
    RESUMABLE_PHASES,
    ROLLBACK_PHASES,
)
from gpu_fault_release.regional_gpu_bootstrap import ensure_connection_secret
from gpu_fault_release.regional_release_agent_convergence import wait_agents
from gpu_fault_release.regional_release_config import (
    ClusterTarget,
    ReleaseError,
)
from gpu_fault_release.regional_release_fleet_rollout import (
    INSTALLER_WAVE_CONFIG_MAP_ENV,
    FleetWaveContext,
    NodeRolloutPolicy,
    ensure_rollout_wave_safe,
    hand_wave_to_reconciler,
    node_rollout_policy,
    reconciler_container_env,
    reconciler_installer_identity,
    target_node_failure_domains,
    target_node_names,
)
from gpu_fault_release.regional_release_node_preflight import (
    validate_target_node_state,
)
from gpu_fault_release.regional_release_online_registry import (
    current_registrations,
    publish_registry_revision,
)
from gpu_fault_release.regional_release_registry import registry, write_registry
from gpu_fault_release.regional_release_rollout_wait import wait_deployment_rollout
from gpu_fault_release.regional_release_state import remote_command_stats

STATE_SCHEMA_VERSION = 1
STATE_ROOT = "rotate-token"
PENDING_TOKEN_FILE = "pending.token"
DEFAULT_WINDOW_MINUTES = 120
DEFAULT_QUIET_SECONDS = 180
DEFAULT_ACCEPTANCE_TIMEOUT_SECONDS = 1200
DEPLOYMENT_ROLLOUT_TIMEOUT_SECONDS = 600
REGISTRY_PUBLISH_TIMEOUT_SECONDS = 300
REGISTRY_REVISIONS_PATH = "/v1/regional/registry/revisions"
INSTALLER_STATE_ANNOTATION = "gpu-fault.io/installer-state"
REINSTALL_STATE = "Retrying"
STEADY_ALLOWED_NODES = "*"
MIN_TOKEN_LENGTH = 32
# Remote command terminal states whose growth during the rotation means a
# command was cut off by a restart rather than finished by its executor.
REMOTE_COMMAND_LOSS_STATUSES = ("FAILED", "EXPIRED")

STATUS_IN_PROGRESS = "IN_PROGRESS"
STATUS_COMPLETED = "COMPLETED"
STATUS_ROLLED_BACK = "ROLLED_BACK"

STEP_PREPARED = "PREPARED"
STEP_OVERLAP_PUBLISHED = "REGISTRY_OVERLAP_PUBLISHED"
STEP_SECRET_UPDATED = "CONNECTION_SECRET_UPDATED"
STEP_DATA_PLANE_ROLLED = "DATA_PLANE_ROLLED"
STEP_NODES_ROLLED = "NODES_ROLLED"
STEP_ACCEPTED = "DATA_PLANE_ACCEPTED"
STEP_TOKEN_FILE_WRITTEN = "TOKEN_FILE_WRITTEN"
STEP_RETIRING_DROPPED = "RETIRING_TOKEN_DROPPED"
ROTATION_STEPS = (
    STEP_PREPARED,
    STEP_OVERLAP_PUBLISHED,
    STEP_SECRET_UPDATED,
    STEP_DATA_PLANE_ROLLED,
    STEP_NODES_ROLLED,
    STEP_ACCEPTED,
    STEP_TOKEN_FILE_WRITTEN,
    STEP_RETIRING_DROPPED,
)
ROLLBACK_SECRET_RESTORED = "ROLLBACK_SECRET_RESTORED"
ROLLBACK_DATA_PLANE_ROLLED = "ROLLBACK_DATA_PLANE_ROLLED"
ROLLBACK_NODES_ROLLED = "ROLLBACK_NODES_ROLLED"
ROLLBACK_REGISTRY_RESTORED = "ROLLBACK_REGISTRY_RESTORED"
ROLLBACK_STEPS = (
    ROLLBACK_SECRET_RESTORED,
    ROLLBACK_DATA_PLANE_ROLLED,
    ROLLBACK_NODES_ROLLED,
    ROLLBACK_REGISTRY_RESTORED,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RotateTokenRequest:
    site: RenderedSite
    cluster_id: str
    window: timedelta = timedelta(minutes=DEFAULT_WINDOW_MINUTES)
    quiet_seconds: int = DEFAULT_QUIET_SECONDS
    acceptance_timeout_seconds: int = DEFAULT_ACCEPTANCE_TIMEOUT_SECONDS
    keep_window: bool = False
    rollback: bool = False
    reference: str | None = None

    def __post_init__(self) -> None:
        rotation_policy(self)


@dataclass
class RotationContext:
    request: RotateTokenRequest
    release: Any
    target: ClusterTarget
    state_path: Path
    state: dict[str, Any]
    token_file: Path
    now: Callable[[], datetime]

    @property
    def cluster_id(self) -> str:
        return self.request.cluster_id

    @property
    def pending_token_file(self) -> Path:
        return self.state_path.parent / PENDING_TOKEN_FILE

    @property
    def rotation_id(self) -> str:
        return str(self.state["rotation_id"])

    def save(self) -> None:
        write_json_atomic(self.state_path, self.state)


# --------------------------------------------------------------------------
# State file


def rotation_state_path(site: RenderedSite, cluster_id: str) -> Path:
    return site.source.parent / STATE_ROOT / safe_name(cluster_id) / "state.json"


def load_rotation_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BootstrapError(f"rotate-token state is not an object: {path}")
    return cast(dict[str, Any], value)


def step_done(state: Mapping[str, Any], step: str) -> bool:
    return step in (state.get("steps") or {})


def step_started(state: Mapping[str, Any], step: str) -> bool:
    return step_done(state, step) or step in (state.get("started_steps") or {})


def start_step(context: RotationContext, step: str) -> None:
    started = context.state.setdefault("started_steps", {})
    if step not in started:
        started[step] = context.now().isoformat()
        context.save()


def complete_step(
    context: RotationContext,
    step: str,
    evidence: Mapping[str, Any] | None = None,
) -> None:
    steps = context.state.setdefault("steps", {})
    steps[step] = {
        "completed_at": context.now().isoformat(),
        "evidence": dict(evidence or {}),
    }
    context.save()


def _site_cluster(site: RenderedSite, cluster_id: str) -> dict[str, Any]:
    matches = [
        dict(item)
        for item in site.release_config["clusters"]
        if str(item.get("cluster_id")) == cluster_id
    ]
    if len(matches) != 1:
        raise BootstrapError(f"unknown cluster_id: {cluster_id}")
    return matches[0]


def _archive_finished_state(path: Path, state: dict[str, Any]) -> None:
    history = path.parent / "history"
    history.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_json_atomic(history / f"{state['rotation_id']}.json", state)
    path.unlink()


def _start_or_resume_state(
    request: RotateTokenRequest,
    token_file: Path,
    now: Callable[[], datetime],
) -> tuple[Path, dict[str, Any]]:
    policy = rotation_policy(request)
    path = rotation_state_path(request.site, request.cluster_id)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    existing = load_rotation_state(path)
    if existing is not None:
        if existing.get("schema_version") != STATE_SCHEMA_VERSION or existing.get(
            "status"
        ) not in {STATUS_IN_PROGRESS, STATUS_COMPLETED, STATUS_ROLLED_BACK}:
            raise BootstrapError("rotate-token state has an unknown schema or status")
        expected = {
            "site_id": request.site.release_config["site_name"],
            "cluster_id": request.cluster_id,
            "token_file": str(token_file),
        }
        for key, value in expected.items():
            if existing.get(key) != value:
                raise BootstrapError(f"rotate-token state conflicts on {key}")
        cleanup_pending = existing.get("pending_token_cleanup_completed") is False
        if existing.get("status") == STATUS_IN_PROGRESS or cleanup_pending:
            require_rotation_policy(existing, policy)
            if existing.get("site_sha256") != request.site.source_sha256:
                raise BootstrapError(
                    "rotate-token site changed since the rotation started"
                )
            if cleanup_pending and request.rollback != (
                existing.get("status") == STATUS_ROLLED_BACK
            ):
                raise BootstrapError(
                    "rotate-token cleanup must resume in the committed direction"
                )
            return path, existing
        if request.rollback:
            raise BootstrapError(
                f"no rotation is in progress for {request.cluster_id}; "
                f"the last one is {existing.get('status')}"
            )
        _archive_finished_state(path, existing)
    elif request.rollback:
        raise BootstrapError(f"no rotation is in progress for {request.cluster_id}")
    stamp = now()
    state: dict[str, Any] = {
        "schema_version": STATE_SCHEMA_VERSION,
        "site_id": request.site.release_config["site_name"],
        "site_sha256": request.site.source_sha256,
        "cluster_id": request.cluster_id,
        "rotation_id": f"{stamp:%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}",
        "status": STATUS_IN_PROGRESS,
        "started_at": stamp.isoformat(),
        "token_file": str(token_file),
        **policy,
        "reference": request.reference,
        "operator": resolve_operator_identity(),
        "steps": {},
        "node_rollout": {"completed_nodes": []},
        "warnings": [],
    }
    write_json_atomic(path, state)
    return path, state


# --------------------------------------------------------------------------
# Engine access


@contextmanager
def site_process_environment(site: RenderedSite) -> Iterator[None]:
    """Run the release engine in-process with the site's environment.

    The engine reads image overrides and the GPU kubeconfig from the process
    environment because it was written to run as a subprocess of the CLI; the
    rotation composes its primitives directly, so it lends the environment for
    the duration of the command and restores the caller's afterwards.
    """

    previous = dict(os.environ)
    selected = effective_environment(site)
    for name in previous.keys() - selected.keys():
        os.environ.pop(name, None)
    os.environ.update(selected)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def release_transaction_open(state: Mapping[str, Any]) -> str | None:
    """The phase of an unfinished release transaction, or ``None``.

    The same vocabulary ``deploy``/``resume`` use: a phase a resume would pick
    up, a rollback or bootstrap in flight, or a ``complete`` whose commit has
    not happened. A stopped (FAILED/PAUSED) transaction counts as open too: it
    has left the fleet mixed, and a token rotation on top of it would have to
    be undone by whichever of the two the operator resolves first.
    """

    phase = str(state.get("phase") or "")
    if phase in RESUMABLE_PHASES or phase in ROLLBACK_PHASES:
        return phase
    if phase in BOOTSTRAP_PHASES and phase != "bootstrap-cleaned":
        return phase
    if phase == "complete":
        if (
            state.get("transaction_committed") is not True
            or state.get("commit_cleanup_completed") is False
        ):
            return phase
        return None
    if phase == "rolled-back" and state.get("rollback_cleanup_completed") is False:
        return phase
    return None if phase in {"rolled-back", "bootstrap-cleaned"} else phase or "unknown"


def ensure_rotation_allowed(site: RenderedSite, release: Any) -> None:
    """The guards ``join-cluster`` and ``remove-cluster`` apply before mutating."""

    open_phase = release_transaction_open(live_release_state(site))
    if open_phase is not None:
        raise BootstrapError(
            "a release transaction is open "
            f"(phase {open_phase}); finish, resume or roll it back before rotating"
        )
    if not release._remote_commands_are_idle():
        raise BootstrapError(
            "remote commands are PENDING/LEASED/WAITING; rotate when the cluster is idle"
        )


def _read_token(path: Path) -> str:
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise BootstrapError(f"cannot read cluster token file {path}: {exc}") from exc
    if len(token) < MIN_TOKEN_LENGTH:
        raise BootstrapError(f"cluster token in {path} is shorter than 32 characters")
    return token


# --------------------------------------------------------------------------
# Registry


def publish_overlap_revision(
    release: Any,
    cluster_id: str,
    *,
    new_digest: str,
    old_digest: str,
    expires_at: datetime,
    reason: str,
) -> dict[str, Any]:
    """Publish the revision that accepts both tokens for one cluster.

    Every other entry is exactly what a release would publish from the
    bootstrap Secret; only the target's digests move. The Secret's own digest
    for the target must be the one the site's token file yields, or the Secret
    and the durable head have already drifted and a publish from it would
    revert somebody else's rotation.
    """

    registrations = []
    found = False
    for source in current_registrations(release, {}):
        item = dict(source)
        if str(item.get("cluster_id")) == cluster_id:
            found = True
            if item.get("token_sha256") != old_digest:
                raise BootstrapError(
                    f"registry Secret digest for {cluster_id} does not match the "
                    "site's token file; reconcile them before rotating"
                )
            if item.get("retiring_token_sha256"):
                raise BootstrapError(
                    f"{cluster_id} already carries a retiring token; finish or roll "
                    "back that rotation first"
                )
            item["token_sha256"] = new_digest
            item["retiring_token_sha256"] = old_digest
            item["token_rotation_expires_at"] = expires_at.isoformat()
        registrations.append(item)
    if not found:
        raise BootstrapError(f"{cluster_id} is not in the regional registry Secret")
    result = publish_registry_revision(
        release,
        path=REGISTRY_REVISIONS_PATH,
        payload={"registrations": registrations, "reason": reason},
        use_current_generation=True,
        timeout_seconds=REGISTRY_PUBLISH_TIMEOUT_SECONDS,
    )
    missing = list(result.get("missing_member_ids") or [])
    if missing:
        # An engine-level failure of a revision that is already live, so the
        # caller's old-only republish is the right answer; the refusals above
        # happen before anything is published and must never trigger it.
        raise ReleaseError(
            "registry revision did not reach every control-plane member: "
            + ", ".join(str(item) for item in missing)
        )
    return result


def publish_current_revision(release: Any, *, reason: str) -> dict[str, Any]:
    """Publish the bootstrap Secret as-is: the old-only (or, after the Secret
    rewrite, the new-only) registry."""

    return publish_registry_revision(
        release,
        path=REGISTRY_REVISIONS_PATH,
        payload={
            "registrations": current_registrations(release, {}),
            "reason": reason,
        },
        use_current_generation=True,
        timeout_seconds=REGISTRY_PUBLISH_TIMEOUT_SECONDS,
    )


def rewrite_registry_secret_token(release: Any, cluster_id: str, token: str) -> None:
    """Put the accepted token into the bootstrap Secret's plaintext entry.

    The Secret is the disaster-recovery copy and the source of every future
    ``publish_current_registry``; left at the old value, the next release
    would republish the retired credential and lock the cluster out.
    """

    entries = []
    found = False
    for source in registry(release):
        item = dict(source)
        if str(item.get("cluster_id")) == cluster_id:
            found = True
            item["token"] = token
        entries.append(item)
    if not found:
        raise BootstrapError(f"{cluster_id} is not in the regional registry Secret")
    write_registry(release, entries)


# --------------------------------------------------------------------------
# Data plane


def update_connection_secret(
    release: Any, target: ClusterTarget, token_file: Path
) -> None:
    """Render the GPU connection Secret from ``token_file`` instead of the site's."""

    ensure_connection_secret(release, replace(target, token_file=str(token_file)))


def restart_data_plane(release: Any, target: ClusterTarget) -> dict[str, Any]:
    """Restart every GPU Deployment that reads the connection Secret and wait."""

    namespace = release.config.namespace
    for name in inventory.DEPLOYMENTS:
        release.runner.run(
            release._gpu(
                target, "-n", namespace, "rollout", "restart", f"deployment/{name}"
            )
        )
    for name in inventory.DEPLOYMENTS:
        wait_deployment_rollout(
            release,
            target,
            name,
            timeout_seconds=DEPLOYMENT_ROLLOUT_TIMEOUT_SECONDS,
        )
    return {"deployments": list(inventory.DEPLOYMENTS)}


def token_rollout_waves(
    cluster_id: str,
    node_names: tuple[str, ...],
    failure_domains: Mapping[str, str],
    policy: NodeRolloutPolicy,
) -> list[tuple[str, ...]]:
    """The waves a release would roll these nodes in, without a record.

    The digests are placeholders: the request model computes waves from the
    node list, the failure-domain map and the policy alone, and nothing here is
    persisted.
    """

    request = FleetDeploymentRequest(
        cluster_id=cluster_id,
        node_ids=list(node_names),
        desired_agent_version="rotate-token",
        desired_artifact_sha256="0" * 64,
        desired_policy_version="rotate-token",
        desired_runtime_profile_version="rotate-token",
        desired_config_digest="0" * 64,
        max_unavailable=min(policy.max_unavailable, len(node_names)),
        first_wave_max_unavailable=min(
            policy.first_wave_max_unavailable, len(node_names)
        ),
        max_unavailable_per_failure_domain=min(
            policy.max_unavailable_per_failure_domain, len(node_names)
        ),
        node_failure_domains=dict(failure_domains),
    )
    return [tuple(wave) for wave in deployment_waves(request)]


def mark_nodes_for_reinstall(
    release: Any, target: ClusterTarget, wave: tuple[str, ...]
) -> None:
    """Make the Reconciler re-run the installer on ``wave``.

    The installer copies the token out of the connection Secret into the
    host's ``systemd`` env files, so a re-install after the Secret update is
    the sanctioned way to change a node's credential.
    """

    release.runner.run(
        release._gpu(
            target,
            "annotate",
            "node",
            *wave,
            f"{INSTALLER_STATE_ANNOTATION}={REINSTALL_STATE}",
            "--overwrite",
        )
    )


def read_wave_config(release: Any, target: ClusterTarget, name: str) -> dict[str, str]:
    data = (
        release._get_json(
            release._gpu(
                target, "-n", release.config.namespace, "get", "configmap", name
            )
        ).get("data")
        or {}
    )
    return {str(key): str(value) for key, value in data.items()}


def restore_wave_config(
    release: Any,
    target: ClusterTarget,
    name: str,
    steady: Mapping[str, str],
) -> None:
    release.runner.run(
        release._gpu(
            target,
            "-n",
            release.config.namespace,
            "patch",
            "configmap",
            name,
            "--type=merge",
            "-p",
            json.dumps({"data": dict(steady)}, sort_keys=True),
        )
    )
    observed = read_wave_config(release, target, name)
    if {key: observed.get(key, "") for key in steady} != dict(steady):
        raise BootstrapError(
            f"{target.cluster_id} installer wave ConfigMap {name} did not return "
            "to its steady state"
        )


def _wave_context(
    release: Any,
    target: ClusterTarget,
    *,
    rotation_id: str,
    node_names: tuple[str, ...],
    identity: tuple[str, str],
    policy: NodeRolloutPolicy,
) -> FleetWaveContext:
    return FleetWaveContext(
        phase="rotate-token",
        deployment_id=f"rotate-token-{target.cluster_id}-{rotation_id}",
        node_names=node_names,
        paused_identity=identity,
        wheel_cm=release.executor_wheel_cm,
        bundle_cm=release.bundle_cm,
        artifact_sha=release.node_wheel_sha,
        config_digest=release.config.agent_config_digest,
        expected_profile=release.config.runtime_profile_version,
        executor_wheel_filename=None,
        expected_compatibility=(
            release.config.component_digests.get("node_runtime")
            or release.node_wheel_sha
        ),
        desired_bundle=identity[0],
        desired_template=identity[1],
        template_config_map=None,
        max_unavailable=policy.max_unavailable,
        runtime_image=None,
        node_installer_image=None,
        allow_legacy_identity=False,
        agent_identity=None,
    )


def roll_node_tokens(
    release: Any,
    target: ClusterTarget,
    *,
    rotation_id: str,
    progress: dict[str, Any],
    record: Callable[[], None],
    only_nodes: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Re-install the cluster's nodes in release-shaped waves.

    ``progress`` is the state file's ``node_rollout`` mapping; ``record`` is
    called after every mutation so a rerun continues with the nodes still
    holding the old token. ``only_nodes`` restricts a rollback to the nodes the
    forward pass already moved.
    """

    node_names = target_node_names(release, target)
    if progress.get("node_names") is not None and set(progress["node_names"]) != set(
        node_names
    ):
        raise BootstrapError("rotation node inventory changed since the first wave")
    completed = set(progress.get("completed_nodes") or [])
    remaining = tuple(
        name
        for name in node_names
        if name not in completed and (only_nodes is None or name in only_nodes)
    )
    validate_target_node_state(release, target, node_names)
    environment = reconciler_container_env(release, target)
    identity = reconciler_installer_identity(environment)
    config_map = str(environment.get(INSTALLER_WAVE_CONFIG_MAP_ENV) or "")
    if not config_map:
        raise BootstrapError(
            f"{target.cluster_id} Reconciler predates the installer wave ConfigMap; "
            "run a release upgrade before rotating the token"
        )
    steady = cast(dict[str, str] | None, progress.get("steady_wave_config"))
    if steady is None:
        steady = read_wave_config(release, target, config_map)
        if steady.get("allowed-nodes") != STEADY_ALLOWED_NODES:
            raise BootstrapError(
                f"{target.cluster_id} Reconciler is mid-wave "
                f"(allowed-nodes={steady.get('allowed-nodes')!r}); finish or resume "
                "the release that owns it before rotating"
            )
        progress["steady_wave_config"] = steady
        progress["wave_config_map"] = config_map
        progress["node_names"] = list(node_names)
        record()
    waves: list[tuple[str, ...]] = []
    if remaining:
        failure_domains = target_node_failure_domains(release, target, remaining)
        policy = node_rollout_policy(release, failure_domains, phase="upgrade")
        waves = token_rollout_waves(
            target.cluster_id, remaining, failure_domains, policy
        )
        context = _wave_context(
            release,
            target,
            rotation_id=rotation_id,
            node_names=node_names,
            identity=identity,
            policy=policy,
        )
        for wave in waves:
            ensure_rollout_wave_safe(release, target, wave=wave, node_names=node_names)
            progress["started_nodes"] = sorted(
                set(progress.get("started_nodes") or []).union(wave)
            )
            record()
            if hand_wave_to_reconciler(release, target, context, wave) is None:
                raise BootstrapError(
                    f"{target.cluster_id} Reconciler did not accept the wave ConfigMap"
                )
            mark_nodes_for_reinstall(release, target, wave)
            wait_agents(
                release,
                target,
                context.artifact_sha,
                bundle_sha=identity[0],
                template_sha=identity[1],
                config_digest=context.config_digest,
                runtime_profile_version=context.expected_profile,
                node_names=wave,
            )
            completed.update(wave)
            progress["completed_nodes"] = sorted(completed)
            record()
    restore_wave_config(release, target, config_map, steady)
    wait_agents(
        release,
        target,
        release.node_wheel_sha,
        bundle_sha=identity[0],
        template_sha=identity[1],
        config_digest=release.config.agent_config_digest,
        runtime_profile_version=release.config.runtime_profile_version,
    )
    return {
        "node_count": len(node_names),
        "waves": [list(wave) for wave in waves],
        "reinstalled_nodes": sorted(completed),
    }


# --------------------------------------------------------------------------
# Acceptance


def retiring_token_authentications(
    site: RenderedSite,
    cluster_id: str,
    *,
    since_seconds: int,
) -> list[str]:
    """Matching retained log lines from stable Ready control-plane containers."""

    return collect_retiring_token_authentications(
        site,
        cluster_id,
        since_seconds=since_seconds,
        run=run_command,
        now=_utc_now,
    )


def wait_for_new_token_acceptance(
    site: RenderedSite,
    cluster_id: str,
    *,
    quiet_seconds: int,
    timeout_seconds: int,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Wait for a bounded quiet window with current-log prefix continuity.

    The initial quiet period starts after the last data-plane mutation.
    Current log prefixes must cover the fixed window and match after reading.
    """

    return wait_for_quiet_token_authentications(
        site,
        cluster_id,
        quiet_seconds=quiet_seconds,
        timeout_seconds=timeout_seconds,
        read_logs=retiring_token_authentications,
        sleep=sleep,
        monotonic=monotonic,
    )


def remote_command_losses(
    baseline: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, int]:
    """Terminal remote-command counters that grew since the baseline."""

    before = baseline.get("by_status") or {}
    after = current.get("by_status") or {}
    return {
        status: int(after.get(status, 0)) - int(before.get(status, 0))
        for status in REMOTE_COMMAND_LOSS_STATUSES
        if int(after.get(status, 0)) > int(before.get(status, 0))
    }


# --------------------------------------------------------------------------
# Token file


def write_token_file(token_file: Path, token: str, *, now: datetime) -> Path:
    """Replace the site's token file atomically, keeping the old as retired.

    The retired copy is a hard link taken before the replace, so there is no
    instant at which the path is missing, and a crash between the two steps
    leaves a retired copy plus a temporary file rather than a lost token.
    """

    retired = token_file.with_name(f"{token_file.name}.retired-{now:%Y%m%dT%H%M%SZ}")
    if retired.exists() and cluster_token_sha256(
        _read_token(retired)
    ) != cluster_token_sha256(_read_token(token_file)):
        raise BootstrapError("retired token file conflicts with the current token")
    temporary = token_file.with_name(f".{token_file.name}.rotating")
    if temporary.exists():
        temporary.unlink()
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(token)
        handle.flush()
        os.fsync(handle.fileno())
    if token_file.exists() and not retired.exists():
        os.link(token_file, retired)
        os.chmod(retired, 0o600)
    os.replace(temporary, token_file)
    os.chmod(token_file, 0o600)
    descriptor = os.open(token_file.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return retired


# --------------------------------------------------------------------------
# Orchestration


def _prepare(context: RotationContext) -> None:
    ensure_rotation_allowed(context.request.site, context.release)
    old_token = _read_token(context.token_file)
    old_digest = cluster_token_sha256(old_token)
    pending = context.pending_token_file
    if pending.is_file():
        new_token = _read_token(pending)
    else:
        new_token = secrets.token_hex(32)
        write_secret(pending, new_token)
    new_digest = cluster_token_sha256(new_token)
    if new_digest == old_digest:
        raise BootstrapError("the pending token equals the current token")
    baseline = remote_command_stats(context.release)
    context.state.update(
        {
            "old_token_sha256": old_digest,
            "new_token_sha256": new_digest,
            "pending_token_file": str(pending),
            "expires_at": (context.now() + context.request.window).isoformat(),
            "remote_command_baseline": baseline,
        }
    )
    complete_step(context, STEP_PREPARED, {"pending_token_file": str(pending)})


def _verify_pending_token(context: RotationContext) -> str:
    """Verify the new credential and recover a journaled atomic file replacement."""

    current_token = _read_token(context.token_file)
    current = cluster_token_sha256(current_token)
    if step_done(context.state, STEP_TOKEN_FILE_WRITTEN):
        if current != context.state.get("new_token_sha256"):
            raise BootstrapError(
                "the committed token file no longer matches its digest"
            )
        return current_token
    if step_started(
        context.state, STEP_TOKEN_FILE_WRITTEN
    ) and current == context.state.get("new_token_sha256"):
        stamp = datetime.fromisoformat(
            context.state["started_steps"][STEP_TOKEN_FILE_WRITTEN]
        )
        retired = context.token_file.with_name(
            f"{context.token_file.name}.retired-{stamp:%Y%m%dT%H%M%SZ}"
        )
        if cluster_token_sha256(_read_token(retired)) != context.state.get(
            "old_token_sha256"
        ):
            raise BootstrapError("retired token file no longer matches its digest")
        complete_step(
            context, STEP_TOKEN_FILE_WRITTEN, {"retired_token_file": str(retired)}
        )
        return current_token
    if current != context.state.get("old_token_sha256"):
        raise BootstrapError(
            "the site's token file changed under the rotation; refusing to continue"
        )
    pending = context.pending_token_file
    if not pending.is_file():
        raise BootstrapError(
            f"pending token file {pending} is missing; the rotation cannot resume"
        )
    token = _read_token(pending)
    if cluster_token_sha256(token) != context.state.get("new_token_sha256"):
        raise BootstrapError("pending token file no longer matches the recorded digest")
    return token


def _expires_at(context: RotationContext) -> datetime:
    return datetime.fromisoformat(str(context.state["expires_at"]))


def _publish_overlap(context: RotationContext, *, extend: bool = False) -> None:
    expires_at = (
        context.now() + context.request.window if extend else _expires_at(context)
    )
    reason = f"rotate-token {context.cluster_id} {context.rotation_id} " + (
        "window extended" if extend else "overlap"
    )
    try:
        result = publish_overlap_revision(
            context.release,
            context.cluster_id,
            new_digest=str(context.state["new_token_sha256"]),
            old_digest=str(context.state["old_token_sha256"]),
            expires_at=expires_at,
            reason=reason,
        )
    except ReleaseError as exc:
        if extend:
            raise BootstrapError(f"rotation window extension failed: {exc}") from exc
        # Nothing on the data plane has moved, so the old-only registry is the
        # one every caller can still authenticate against.
        try:
            publish_current_revision(
                context.release,
                reason=(
                    f"rotate-token {context.cluster_id} {context.rotation_id} "
                    "rollback after failed overlap publish"
                ),
            )
        except ReleaseError as rollback_exc:
            raise BootstrapError(
                f"overlap publish failed ({exc}) and the old-only republish also "
                f"failed ({rollback_exc}); rerun to resume or --rollback"
            ) from exc
        raise BootstrapError(
            f"overlap registry publish failed; registry restored to the old token "
            f"only: {exc}"
        ) from exc
    context.state["expires_at"] = expires_at.isoformat()
    complete_step(
        context,
        STEP_OVERLAP_PUBLISHED,
        {
            "generation": result.get("generation"),
            "content_sha256": result.get("content_sha256"),
            "expires_at": expires_at.isoformat(),
        },
    )


def _ensure_window_open(context: RotationContext) -> None:
    """Extend a window a resume would otherwise run past.

    An expired retiring digest cuts off every caller that has not moved yet,
    so a resume that finds less than the minimum window left republishes the
    overlap with a fresh deadline (the doc's rule: extend, never rely on
    expiry) rather than racing it.
    """

    remaining = _expires_at(context) - context.now()
    if remaining >= timedelta(minutes=MIN_WINDOW_MINUTES):
        return
    _publish_overlap(context, extend=True)
    context.state.setdefault("warnings", []).append(
        f"rotation window extended on resume to {context.state['expires_at']}"
    )
    context.save()


def _roll_nodes(
    context: RotationContext, *, only_nodes: frozenset[str] | None
) -> dict[str, Any]:
    progress = cast(dict[str, Any], context.state.setdefault("node_rollout", {}))
    progress.setdefault("completed_nodes", [])
    return roll_node_tokens(
        context.release,
        context.target,
        rotation_id=context.rotation_id,
        progress=progress,
        record=context.save,
        only_nodes=only_nodes,
    )


def _accept(context: RotationContext) -> None:
    evidence = wait_for_new_token_acceptance(
        context.request.site,
        context.cluster_id,
        quiet_seconds=context.request.quiet_seconds,
        timeout_seconds=context.request.acceptance_timeout_seconds,
    )
    current = remote_command_stats(context.release)
    losses = remote_command_losses(
        context.state.get("remote_command_baseline") or {}, current
    )
    if losses:
        context.state.setdefault("warnings", []).append(
            "remote command terminal counters grew during the rotation: "
            + json.dumps(losses, sort_keys=True)
        )
    evidence["remote_command_losses"] = losses
    complete_step(context, STEP_ACCEPTED, evidence)


def _finish(context: RotationContext, new_token: str) -> None:
    if not step_done(context.state, STEP_TOKEN_FILE_WRITTEN):
        start_step(context, STEP_TOKEN_FILE_WRITTEN)
        stamp = datetime.fromisoformat(
            context.state["started_steps"][STEP_TOKEN_FILE_WRITTEN]
        )
        retired = write_token_file(context.token_file, new_token, now=stamp)
        complete_step(
            context, STEP_TOKEN_FILE_WRITTEN, {"retired_token_file": str(retired)}
        )
    if not step_done(context.state, STEP_RETIRING_DROPPED):
        rewrite_registry_secret_token(
            context.release, context.cluster_id, _read_token(context.token_file)
        )
        evidence: dict[str, Any] = {"retiring_token_dropped": False}
        if context.request.keep_window:
            context.state.setdefault("warnings", []).append(
                "retiring token left to expire at "
                f"{context.state['expires_at']} (--keep-window)"
            )
        else:
            result = publish_current_revision(
                context.release,
                reason=f"rotate-token {context.cluster_id} {context.rotation_id} final",
            )
            evidence = {
                "retiring_token_dropped": True,
                "generation": result.get("generation"),
                "content_sha256": result.get("content_sha256"),
            }
        complete_step(context, STEP_RETIRING_DROPPED, evidence)
    _complete_rotation(context, STATUS_COMPLETED)


def _complete_rotation(context: RotationContext, status: str) -> None:
    context.state["status"] = status
    context.state.setdefault("completed_at", context.now().isoformat())
    context.state["pending_token_cleanup_completed"] = False
    context.save()
    context.pending_token_file.unlink(missing_ok=True)
    context.state["pending_token_cleanup_completed"] = True
    context.save()


def _rotate(context: RotationContext) -> None:
    state = context.state
    if state.get("rollback_started_at") or any(
        step_started(state, step) for step in ROLLBACK_STEPS
    ):
        raise BootstrapError("token rollback is in progress; resume with --rollback")
    if not step_done(state, STEP_PREPARED):
        _prepare(context)
    new_token = _verify_pending_token(context)
    if not step_done(state, STEP_TOKEN_FILE_WRITTEN):
        ensure_rotation_allowed(context.request.site, context.release)
    if not step_done(state, STEP_OVERLAP_PUBLISHED):
        _publish_overlap(context)
    elif not step_done(state, STEP_TOKEN_FILE_WRITTEN):
        _ensure_window_open(context)
    if not step_done(state, STEP_SECRET_UPDATED):
        start_step(context, STEP_SECRET_UPDATED)
        update_connection_secret(
            context.release, context.target, context.pending_token_file
        )
        complete_step(context, STEP_SECRET_UPDATED)
    if not step_done(state, STEP_DATA_PLANE_ROLLED):
        start_step(context, STEP_DATA_PLANE_ROLLED)
        complete_step(
            context,
            STEP_DATA_PLANE_ROLLED,
            restart_data_plane(context.release, context.target),
        )
    if not step_done(state, STEP_NODES_ROLLED):
        complete_step(context, STEP_NODES_ROLLED, _roll_nodes(context, only_nodes=None))
    if not step_done(state, STEP_TOKEN_FILE_WRITTEN):
        _accept(context)
    _finish(context, new_token)


def _rollback(context: RotationContext) -> None:
    state = context.state
    if state.get("status") == STATUS_ROLLED_BACK:
        _complete_rotation(context, STATUS_ROLLED_BACK)
        return
    if step_started(state, STEP_TOKEN_FILE_WRITTEN):
        raise BootstrapError(
            "the rotation recorded the token file write intent; rollback is "
            "forbidden from this boundary even if the filesystem write has not "
            "started; rerun without --rollback to finish"
        )
    if step_done(state, STEP_PREPARED):
        if cluster_token_sha256(_read_token(context.token_file)) != state.get(
            "old_token_sha256"
        ):
            raise BootstrapError("the site's token file changed; rollback is refused")
        ensure_rotation_allowed(context.request.site, context.release)
    if not state.get("rollback_started_at"):
        state["rollback_started_at"] = context.now().isoformat()
        context.save()
    if step_done(state, STEP_OVERLAP_PUBLISHED) and not step_done(
        state, ROLLBACK_REGISTRY_RESTORED
    ):
        _ensure_window_open(context)
    if step_started(state, STEP_SECRET_UPDATED) and not step_done(
        state, ROLLBACK_SECRET_RESTORED
    ):
        ensure_connection_secret(context.release, context.target)
        complete_step(context, ROLLBACK_SECRET_RESTORED)
    if (
        step_started(state, STEP_SECRET_UPDATED)
        or step_started(state, STEP_DATA_PLANE_ROLLED)
    ) and not step_done(state, ROLLBACK_DATA_PLANE_ROLLED):
        complete_step(
            context,
            ROLLBACK_DATA_PLANE_ROLLED,
            restart_data_plane(context.release, context.target),
        )
    progress = state.get("node_rollout") or {}
    moved = frozenset(
        progress.get("forward_nodes")
        or [
            *(progress.get("completed_nodes") or []),
            *(progress.get("started_nodes") or []),
        ]
    )
    if moved and not step_done(state, ROLLBACK_NODES_ROLLED):
        if "forward_nodes" not in progress:
            state["node_rollout"] = {
                "completed_nodes": [],
                "steady_wave_config": progress.get("steady_wave_config"),
                "wave_config_map": progress.get("wave_config_map"),
                "node_names": progress.get("node_names"),
                "forward_nodes": sorted(moved),
            }
            context.save()
        complete_step(
            context, ROLLBACK_NODES_ROLLED, _roll_nodes(context, only_nodes=moved)
        )
    if step_done(state, STEP_OVERLAP_PUBLISHED) and not step_done(
        state, ROLLBACK_REGISTRY_RESTORED
    ):
        result = publish_current_revision(
            context.release,
            reason=f"rotate-token {context.cluster_id} {context.rotation_id} rollback",
        )
        complete_step(
            context,
            ROLLBACK_REGISTRY_RESTORED,
            {"generation": result.get("generation")},
        )
    _complete_rotation(context, STATUS_ROLLED_BACK)


def rotation_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    """What the operator sees: digests, steps and files -- never a token."""

    steps = state.get("steps") or {}
    return {
        "cluster_id": state.get("cluster_id"),
        "rotation_id": state.get("rotation_id"),
        "status": state.get("status"),
        "old_token_sha256": state.get("old_token_sha256"),
        "new_token_sha256": state.get("new_token_sha256"),
        "expires_at": state.get("expires_at"),
        "token_file": state.get("token_file"),
        "retired_token_file": (
            (steps.get(STEP_TOKEN_FILE_WRITTEN) or {}).get("evidence") or {}
        ).get("retired_token_file"),
        "steps": {name: item.get("completed_at") for name, item in steps.items()},
        "reinstalled_nodes": (state.get("node_rollout") or {}).get("completed_nodes"),
        "warnings": list(state.get("warnings") or []),
        "operator": state.get("operator"),
        "reference": state.get("reference"),
    }


def rotate_cluster_token(
    request: RotateTokenRequest,
    *,
    now: Callable[[], datetime] = _utc_now,
) -> dict[str, Any]:
    """Run (or resume, or roll back) the rotation for ``request.cluster_id``."""

    site = request.site
    with administrator_operation_lock(site.source.parent):
        current = reload_site_for_mutation(site)
        record = _site_cluster(current, request.cluster_id)
        expected = _site_cluster(site, request.cluster_id)
        if any(
            record.get(key) != expected.get(key)
            for key in ("context", "eks_cluster_arn", "hyperpod_cluster_name")
        ):
            raise BootstrapError(
                "rotate-token target cluster identity changed before mutation"
            )
        site = current
        request = replace(request, site=site)
        token_file = Path(str(record.get("token_file") or ""))
        if not token_file.is_file():
            raise BootstrapError(f"cluster token file is missing: {token_file}")
        state_path, state = _start_or_resume_state(request, token_file, now)
        with site_process_environment(site):
            release = build_release(site)
            context = RotationContext(
                request=request,
                release=release,
                target=release._target(request.cluster_id),
                state_path=state_path,
                state=state,
                token_file=token_file,
                now=now,
            )
            if request.rollback:
                _rollback(context)
            else:
                _rotate(context)
    return rotation_summary(state)


# --------------------------------------------------------------------------
# CLI


def add_rotate_token_command(
    commands: Any, add_managed_site_arguments: Callable[[Any], None]
) -> None:
    """Register ``gpu-fault-admin rotate-token``; the CLI passes its site options."""

    command = commands.add_parser(
        "rotate-token",
        usage=(
            "gpu-fault-admin rotate-token --state-dir STATE_DIR "
            "--gpu-cluster-arn GPU_ARN [--window-minutes N] [--quiet-seconds N] "
            "[--acceptance-timeout-seconds N] [--keep-window] [--rollback] "
            "[--reference CHANGE_ID]"
        ),
        help=(
            "rotate one GPU cluster's token end to end: registry overlap window, "
            "GPU Secret, data-plane Deployments, node Agents in fleet waves, then "
            "the site's token file; rerun to resume"
        ),
    )
    add_managed_site_arguments(command)
    command.add_argument(
        "--gpu-cluster-arn",
        required=True,
        metavar="GPU_ARN",
        help="the GPU EKS or HyperPod cluster ARN, as given to deploy/join-cluster",
    )
    command.add_argument(
        "--window-minutes",
        type=int,
        default=DEFAULT_WINDOW_MINUTES,
        metavar="N",
        help=(
            "how long the old token stays accepted after the new one is published "
            f"({MIN_WINDOW_MINUTES}-{MAX_WINDOW_MINUTES}; default "
            f"{DEFAULT_WINDOW_MINUTES})"
        ),
    )
    command.add_argument(
        "--quiet-seconds",
        type=int,
        default=DEFAULT_QUIET_SECONDS,
        metavar="N",
        help=(
            "seconds of quiet with stable current-log prefix coverage "
            f"(default {DEFAULT_QUIET_SECONDS})"
        ),
    )
    command.add_argument(
        "--acceptance-timeout-seconds",
        type=int,
        default=DEFAULT_ACCEPTANCE_TIMEOUT_SECONDS,
        metavar="N",
        help=f"give up waiting for acceptance after N seconds (default {DEFAULT_ACCEPTANCE_TIMEOUT_SECONDS})",
    )
    command.add_argument(
        "--keep-window",
        action="store_true",
        help="leave the retiring token to expire instead of dropping it at the end",
    )
    command.add_argument(
        "--rollback",
        action="store_true",
        help="walk an unfinished rotation back onto the old token",
    )
    command.add_argument(
        "--reference",
        metavar="CHANGE_ID",
        help="change ticket recorded with the rotation state",
    )


def _discover(cluster_arn: str) -> tuple[str, str]:
    identity = discover_cluster(
        CommandRunner(),
        cluster_arn=cluster_arn,
        role="gpu",
        context="gpu-fault-admin-rotate-token",
    )
    return identity.eks_arn, identity.hyperpod_name


def run_rotate_token_command(
    arguments: argparse.Namespace, *, site: RenderedSite
) -> int:
    """Rotate the token of the cluster named by ``--gpu-cluster-arn``."""

    request = RotateTokenRequest(
        site=site,
        cluster_id=resolve_cluster_id(
            site, str(arguments.gpu_cluster_arn), discover=_discover
        ),
        window=timedelta(minutes=int(arguments.window_minutes)),
        quiet_seconds=int(arguments.quiet_seconds),
        acceptance_timeout_seconds=int(arguments.acceptance_timeout_seconds),
        keep_window=bool(arguments.keep_window),
        rollback=bool(arguments.rollback),
        reference=cast(str | None, arguments.reference),
    )
    summary = rotate_cluster_token(request)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
