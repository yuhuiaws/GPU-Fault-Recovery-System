#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release.regional_admin_checks import (
    build_health_report,
    build_preflight_report,
    report_exit_code,
)
from gpu_fault_release.regional_admin_commands import (
    bootstrap_cpu_is_current,
    build_full_status,
    build_release_diff,
    apply_rds_ca_bundle,
    build_release_summary,
    ensure_schema,
    run_deploy,
    run_resume,
    stage_noop_release,
)
from gpu_fault_release.regional_aurora_credentials import refresh_aurora_credentials
from gpu_fault_release.regional_dns import apply_control_plane_nlb
from gpu_fault_release.regional_endpoint_rollback import (
    capture_endpoint_snapshot,
    restore_endpoint_snapshot,
)
from gpu_fault_release.regional_gpu_bootstrap import (
    apply_gpu_dcgm_exporter,
    cancel_active_installer_jobs,
    ensure_connection_secret,
    ensure_gpu_namespace,
    preflight_gpu_dcgm_exporter,
    quiesce_gpu_executor,
    settle_installer_jobs,
    verify_gpu_control_plane_endpoint,
)
from gpu_fault_release.regional_notifications import (
    ensure_notification_secret,
    notification_digest,
)
from gpu_fault_release.regional_observability_rollback import (
    capture_observability_snapshot,
    restore_observability_snapshot,
)
from gpu_fault.admin.artifact_configmaps import artifact_binary_sha
from gpu_fault_release.regional_release_artifacts import (
    require_cpu_secrets,
    upload_config_map,
    upload_release,
)
from gpu_fault_release.regional_release_config import (
    ClusterTarget,
    ReleaseConfig,
    ReleaseError,
)
from gpu_fault_release.regional_release_diff import (
    ReleaseChangeKind,
    ReleaseDiff,
    classify_release,
    control_plane_role_targets,
)
from gpu_fault_release.regional_release_fleet_rollout import (
    agent_heartbeats_converged,
    backup_release_secrets,
    backup_secret,
    candidate_agent_pin_identity,
    capture_active_agent_node_sets,
    delete_release_secret_backups,
    deploy_reconciler,
    fleet_command,
    fleet_deployment_id,
    restore_secret,
    target_node_names,
    wait_agents,
    wait_candidate_cpu_agent_heartbeats,
)
from gpu_fault_release.regional_release_gpu_rollout import (
    agents_converged as agents_converged,
)
from gpu_fault_release.regional_release_gpu_rollout import (
    apply_gpu_deployments,
    join_target,
    preflight_gpu_deployments,
    reassert_completion_watcher_state,
    upgrade_gpu_target,
)
from gpu_fault_release.regional_release_gpu_rollout import (
    executor_pin_rejection as executor_pin_rejection,
)
from gpu_fault_release.regional_release_gpu_rollout import (
    require_executor_pin as require_executor_pin,
)
from gpu_fault_release.regional_release_iam import (
    validate_executor_iam_documents as validate_executor_iam_documents,
)
from gpu_fault_release.regional_release_iam import (
    validate_executor_iam_role,
)
from gpu_fault_release.regional_release_node_runtime_rollout import (
    preflight_node_runtime,
    roll_node_runtime,
)
from gpu_fault_release.regional_release_online_registry import (
    activate_join_registry,
    drain_registry_cluster,
    fail_join_registry,
    prepare_join_registry,
    publish_restored_registry,
    publish_staged_registry,
    purge_registry_cluster,
    revoke_registry_cluster,
    rollback_join_registry,
)
from gpu_fault_release.regional_release_orchestration import (
    bootstrap_gpu_clusters,
    rollback_release,
    upgrade_release,
)
from gpu_fault_release.regional_release_orchestration import (
    build_rollback_environment as build_rollback_environment,
)
from gpu_fault_release.regional_release_preflight import ensure_region_contexts
from gpu_fault_release.regional_release_narration import (
    narrate_release_end,
    narrate_release_start,
)
from gpu_fault_release.regional_release_probes import probe_label
from gpu_fault_release.regional_release_registry import (
    commit_registry_update,
    desired_registry,
    initialize_registry,
    registry,
    registry_config_digest,
    registry_entry,
    registry_payloads,
    restore_registry_backup,
    stage_registry,
    update_registry,
    write_registry,
)
from gpu_fault_release.regional_release_rendering import (
    DEFAULT_DCGM_EXPORTER_IMAGE,
    DEFAULT_RUNTIME_IMAGE,
    build_cpu_apply_environment,
    render_release_payload,
    rendered_release_manifest_sha256,
    stamp_gpu_deployments,
)
from gpu_fault_release.regional_release_reporting import build_release_plan
from gpu_fault_release.regional_release_resume_validation import (
    validate_resume_checkpoint,
)
from gpu_fault_release.regional_release_state import (
    STATE_CONFIG_MAP as STATE_CONFIG_MAP,
)
from gpu_fault_release.regional_release_state import (
    capture_previous,
    cleanup_previous_snapshots,
    config_map_binary_key,
    config_map_data,
    deployment_template_name,
    deployment_wheel,
    get_json,
    load_state,
    prime_deployment_snapshot,
    read_snapshot,
    require_digest_pinned_image,
    save_state,
    template_bundle,
)
from gpu_fault_release.regional_release_store_preflight import (
    INFLIGHT_INSTALLS_REFUSED_EXIT_CODE,
    InflightInstallsRefused,
    remote_commands_are_idle,
    require_no_inflight_installs,
)
from gpu_fault_release.regional_release_transaction import commit_release
from gpu_fault_release.regional_release_validation import (
    critical_amp_alerts,
    ensure_profile_transition_safe,
    stability_snapshot,
    store_io_rejection_series_ready,
    validate_release_components,
    validate_release_quick,
    validate_rollback,
    validate_stability_window,
)
from gpu_fault_release.regional_runtime_profile import (
    ensure_runtime_profile,
    runtime_profile_policy_digest,
)

ROOT = Path(__file__).resolve().parents[2]
FAST_ROLLOUT_TIMEOUT = "5m"
SLOW_COMMAND_SECONDS = 15.0
SLOW_COMMAND_LABEL_LIMIT = 160
SENSITIVE_CONFIG_MARKERS = (
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "CREDENTIAL",
    "PRIVATE_KEY",
)


def _echoed_arguments(arguments: list[str]) -> list[str]:
    """Replace a probe body in the command echo with the probe's name.

    Only the echo changes. ``args`` itself is untouched, so the bytes on the
    wire and the ``-c`` calling convention are exactly what they were -- which
    is the whole point of shipping the probe as source (``probes/README.md``).

    Without this, every probe put its full body on one ``+ python3 -c ...``
    line. The probe programs are commented, so that was 24% of the lines of an
    upgrade, and it pushed the kubectl trace an operator actually reads off the
    screen.
    """

    echoed = list(arguments)
    for index, argument in enumerate(echoed):
        if index == 0 or echoed[index - 1] != "-c":
            continue
        label = probe_label(argument)
        if label is not None:
            echoed[index] = label
    return echoed


def _command_label(args: list[str], *, sensitive: bool = False) -> str:
    """The command exactly as the ``+`` echo has always shown it.

    Not shortened: an argument that is not a probe body is the release's real
    input -- a manifest, a selector, a node name -- and the trace an operator
    reads to reconstruct what ran has to keep it whole. Shortening belongs to the
    elapsed line, which is a pointer rather than a record.
    """

    if sensitive:
        return "<sensitive command>"
    return " ".join([Path(args[0]).name, *_echoed_arguments(args[1:])])


class Runner:
    def __init__(self, *, dry_run: bool = False) -> None:
        self.dry_run = dry_run

    def _narrate_elapsed(self, label: str, started: float) -> None:
        """Name the command that ate the wall clock, when it ate enough of it.

        The `+ command` echo says what ran, never how long it took, so a
        five-minute gap between two `release-phase` lines is spread over a dozen
        commands with nothing to say which one held it -- measured on the
        2026-09-05 upgrade: 2m43s across ten lines, then 4m58s across another
        block, both unattributable. Only slow commands are annotated: a release
        issues thousands of sub-second kubectl calls and a duration on each would
        bury the trace this exists to explain.

        The label is shortened here and not in the echo: this line points back at
        a command the echo already recorded in full, and some of them carry a
        rendered manifest as an argument.
        """

        elapsed = time.monotonic() - started
        if elapsed < SLOW_COMMAND_SECONDS:
            return
        if len(label) > SLOW_COMMAND_LABEL_LIMIT:
            label = label[:SLOW_COMMAND_LABEL_LIMIT] + "..."
        print(
            f"command-elapsed {elapsed:.1f}s {label}",
            file=sys.stderr,
            flush=True,
        )

    def run(
        self,
        args: list[str],
        *,
        env: dict[str, str] | None = None,
        input_text: str | None = None,
        capture: bool = False,
        sensitive: bool = False,
        timeout_seconds: float | None = None,
    ) -> str:
        label = _command_label(args, sensitive=sensitive)
        print("+ " + label, file=sys.stderr, flush=True)
        if self.dry_run and not capture:
            return ""
        started = time.monotonic()
        try:
            completed = subprocess.run(
                args,
                check=False,
                text=True,
                input=input_text,
                capture_output=capture,
                env=env,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise ReleaseError(
                f"command timed out after {timeout_seconds}s: {args[0]}"
            ) from exc
        finally:
            # In a `finally` because a command that failed after four minutes is
            # exactly the one whose duration the reader needs.
            self._narrate_elapsed(label, started)
        if completed.returncode:
            # A captured failure has to say what it found. The verifiers this
            # runner drives report their defects on stdout and exit 1, so
            # forwarding stderr alone turns an actionable list of findings into
            # a bare "command failed (1): python3" -- and the release is then
            # blocked by a reason nobody can read. Sensitive commands stay
            # silent: their output is what they were marked sensitive for.
            # Both go to stderr: stdout carries the machine-readable release
            # report, and a failed command's chatter must not land in it.
            if capture and not sensitive:
                for text in (completed.stdout, completed.stderr):
                    if text:
                        print(text, file=sys.stderr)
            raise ReleaseError(f"command failed ({completed.returncode}): {args[0]}")
        return completed.stdout.strip() if capture else ""

    def probe(self, args: list[str], *, timeout_seconds: float | None = None) -> bool:
        """Return whether ``args`` exits zero, treating non-zero as "absent".

        Release code needs read-only existence checks (``kubectl get`` on a
        ConfigMap, Secret, Deployment or CronJob) whose non-zero exit is an
        answer rather than a failure, so they cannot go through :meth:`run`.
        They still have to go through the runner: every call site used to reach
        ``subprocess.run`` directly, which meant a unit test holding a stub
        runner executed a real ``kubectl`` against whatever kubeconfig the
        release config named.

        A probe runs in dry-run mode too, matching the previous behaviour of
        those call sites. Reading cluster state is what makes a dry run's plan
        accurate, and a ``get`` mutates nothing.

        A probe echoes nothing on the way in -- there are hundreds of them and
        their answers are visible in what the release does next -- so a slow one
        is a wholly silent gap, which is why the elapsed line still applies.
        """
        started = time.monotonic()
        try:
            completed = subprocess.run(
                args,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise ReleaseError(
                f"probe timed out after {timeout_seconds}s: {args[0]}"
            ) from exc
        finally:
            self._narrate_elapsed(_command_label(args), started)
        return completed.returncode == 0

    def probe_output(
        self, args: list[str], *, timeout_seconds: float | None = None
    ) -> tuple[int, str, str]:
        """Run ``args`` for its output, returning ``(returncode, stdout, stderr)``.

        The two callers that need this distinguish "absent" from "unreadable" by
        matching ``NotFound`` in stderr, so they need the streams rather than the
        boolean :meth:`probe` returns. Like :meth:`probe`, this exists so the
        call sites do not reach ``subprocess`` behind the runner's back.
        """
        started = time.monotonic()
        try:
            completed = subprocess.run(
                args,
                check=False,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise ReleaseError(
                f"probe timed out after {timeout_seconds}s: {args[0]}"
            ) from exc
        finally:
            self._narrate_elapsed(_command_label(args), started)
        return completed.returncode, completed.stdout, completed.stderr


def sync_release_state(release: Any) -> None:
    previous = release._capture_previous()
    release.state = {}
    release._save_state(
        "complete",
        previous=None,
        release_diff=ReleaseDiff(
            kind=ReleaseChangeKind.NOOP,
            changed=frozenset(),
        ).as_dict(),
        adopted_live_runtime_image=previous["live_runtime_image"],
        transaction_committed=True,
        release_lifecycle="COMMITTED",
        completed_phases=["complete"],
        completed_cluster_ids=sorted(
            target.cluster_id
            for target in getattr(
                getattr(release, "config", None),
                "clusters",
                (),
            )
        ),
    )


def render_and_apply_cpu_roles(
    release: Any,
    environment: dict[str, str],
) -> None:
    with tempfile.TemporaryDirectory(prefix="gpu-fault-role-split-") as directory:
        generated = Path(directory)
        release.runner.run(
            [
                "bash",
                str(
                    ROOT
                    / "deploy/control-plane/tools/render-control-plane-role-split.sh"
                ),
            ],
            env={
                **environment,
                "GPU_FAULT_ROLE_SPLIT_OUT_DIR": str(generated),
            },
        )
        environment["GPU_FAULT_ROLE_SPLIT_GENERATED_DIR"] = str(generated)
        release.runner.run(
            [
                "bash",
                str(
                    ROOT
                    / "deploy/control-plane/tools/apply-control-plane-role-split.sh"
                ),
            ],
            env=environment,
        )


def bootstrap_resume_context(
    state: dict[str, Any],
    release_id: str,
) -> tuple[set[str], bool, bool]:
    loaded_phase = str(state.get("phase") or "")
    resume_phase = str(state.get("resume_phase") or loaded_phase)
    same_release = state.get("release_id") in {None, release_id}
    cleaned = loaded_phase in {
        "bootstrap-cleanup-started",
        "bootstrap-cleanup-progress",
        "bootstrap-cleanup-failed",
        "bootstrap-cleaned",
    }
    completed_cluster_ids = (
        {str(value) for value in state.get("completed_cluster_ids", []) if value}
        if same_release and not cleaned
        else set()
    )
    cpu_checkpoint = (
        same_release
        and not cleaned
        and resume_phase
        in {
            "bootstrap-cpu-ready",
            "bootstrap-endpoint-ready",
            "bootstrap-data-plane-progress",
        }
    )
    check_live_cpu = (
        same_release
        and not cleaned
        and resume_phase in {"bootstrap-started", "bootstrap-failed"}
    )
    return completed_cluster_ids, cpu_checkpoint, check_live_cpu


class RegionalRelease:
    _deployment_snapshot_enabled = True
    _ensure_contexts = ensure_region_contexts
    _apply_gpu_dcgm_exporter = apply_gpu_dcgm_exporter
    _preflight_gpu_dcgm_exporter = preflight_gpu_dcgm_exporter
    _cancel_active_installer_jobs = cancel_active_installer_jobs
    _apply_gpu_deployments = apply_gpu_deployments
    _preflight_gpu_deployments = preflight_gpu_deployments
    _reassert_completion_watcher_state = reassert_completion_watcher_state
    _apply_nlb = apply_control_plane_nlb
    _bootstrap_cpu_is_current = bootstrap_cpu_is_current
    _capture_previous = capture_previous
    _capture_observability_snapshot = capture_observability_snapshot
    _restore_observability_snapshot = restore_observability_snapshot
    _capture_endpoint_snapshot = capture_endpoint_snapshot
    _restore_endpoint_snapshot = restore_endpoint_snapshot
    _config_map_binary_key = config_map_binary_key
    _config_map_data = config_map_data
    _deployment_template_name = deployment_template_name
    _deployment_wheel = deployment_wheel
    _ensure_connection_secret = ensure_connection_secret
    _ensure_gpu_namespace = ensure_gpu_namespace
    _ensure_schema = ensure_schema
    _apply_rds_ca_bundle = apply_rds_ca_bundle
    _refresh_aurora_credentials = refresh_aurora_credentials
    _require_no_inflight_installs = require_no_inflight_installs
    _get_json = get_json
    _load_state = load_state
    _prime_deployment_snapshot = prime_deployment_snapshot
    _quiesce_gpu_executor = quiesce_gpu_executor
    _registry = registry
    _registry_entry = staticmethod(registry_entry)
    _registry_payloads = registry_payloads
    _settle_installer_jobs = settle_installer_jobs
    _read_snapshot = read_snapshot
    _require_cpu_secrets = require_cpu_secrets
    _restore_registry_backup = restore_registry_backup
    _save_state = save_state
    _cleanup_previous_snapshots = cleanup_previous_snapshots
    _stage_registry = stage_registry
    _publish_staged_registry = publish_staged_registry
    _publish_restored_registry = publish_restored_registry
    _stamp_gpu_deployments = stamp_gpu_deployments
    _template_bundle = template_bundle
    _update_registry = update_registry
    _upload_config_map = upload_config_map
    _upload_release = upload_release
    _upgrade_gpu_target = upgrade_gpu_target
    _validate_executor_iam_role = validate_executor_iam_role
    _verify_gpu_control_plane_endpoint = verify_gpu_control_plane_endpoint
    _write_registry = write_registry
    _desired_registry = desired_registry
    _commit_registry_update = commit_registry_update
    _initialize_registry = initialize_registry

    def __init__(self, config: ReleaseConfig, runner: Runner) -> None:
        self.config = config
        self.runner = runner
        self.release_id = config.release_id
        # The digest the plan was approved against. Left unset during
        # construction so the initial render (which produces the digest itself)
        # is not verified against nothing; the apply path populates it from the
        # approved plan and every later render is checked against it.
        self.approved_manifest_digest: str | None = None
        self.wheel_sha = self._sha256(config.wheel)
        self.executor_wheel_sha = self._sha256(config.executor_wheel)
        self.node_wheel_sha = self._sha256(config.node_wheel)
        self.bundle_sha = self._sha256(config.bundle)
        self.runtime_profile_sha = self._sha256(config.runtime_profile_source)
        self.runtime_profile_template_sha = self._sha256(
            config.runtime_profile_template_source
        )
        self.runtime_profile_policy_sha = runtime_profile_policy_digest(
            config.runtime_profile_source
        )
        self.notification_digest = notification_digest(config.notifications)
        self.cluster_registry_digest = registry_config_digest(config.clusters)
        self.admin_config_digest = config.admin_config.sha256()
        self.admin_config_role_digests = config.admin_config.role_sha256()
        self.wheel_cm = "gpu-fault-control-plane-wheel-0100-" + self.wheel_sha[:12]
        self.executor_wheel_cm = (
            "gpu-fault-executor-wheel-0100-" + self.executor_wheel_sha[:12]
        )
        self.bundle_cm = "gpu-fault-node-installer-0100-" + self.bundle_sha[:12]
        self.runtime_image = self._release_image(
            "runtime",
            "GPU_FAULT_RUNTIME_IMAGE",
            DEFAULT_RUNTIME_IMAGE,
        )
        self.node_installer_image = self._release_image(
            "node_installer",
            "GPU_FAULT_NODE_INSTALLER_IMAGE",
            "public.ecr.aws/amazonlinux/amazonlinux:2023",
        )
        self.dcgm_exporter_image = self._release_image(
            "dcgm_exporter",
            "GPU_FAULT_DCGM_EXPORTER_IMAGE",
            DEFAULT_DCGM_EXPORTER_IMAGE,
        )
        self.adot_image = self._release_image(
            "adot",
            "GPU_FAULT_ADOT_IMAGE",
            (
                "public.ecr.aws/aws-observability/aws-otel-collector@"
                "sha256:bb72328152c72fb9662056759b275f7cc85e115db12bbb114fbea9f68dc4816c"
            ),
        )
        for variable, image in (
            ("GPU_FAULT_RUNTIME_IMAGE", self.runtime_image),
            ("GPU_FAULT_NODE_INSTALLER_IMAGE", self.node_installer_image),
            ("GPU_FAULT_DCGM_EXPORTER_IMAGE", self.dcgm_exporter_image),
            ("GPU_FAULT_ADOT_IMAGE", self.adot_image),
        ):
            if not image or any(
                character.isspace() or character == "#" for character in image
            ):
                raise ReleaseError(
                    f"{variable} must be a non-empty OCI image reference "
                    "without whitespace or #"
                )
        self.state: dict[str, Any] = {}
        self.node_template_sha = config.node_template_sha256 or self.bundle_sha
        self.endpoint_digest = hashlib.sha256(
            json.dumps(
                {
                    "delivery_component_sha256": (
                        config.delivery_component_digests.get("endpoint")
                    ),
                    "dns": {
                        "hosted_zone_id": config.dns.hosted_zone_id,
                        "hostname": config.dns.hostname,
                    },
                    "nlb": config.nlb,
                    "clusters": [
                        {
                            "cluster_id": item.cluster_id,
                            "control_plane_url": item.control_plane_url,
                            "ca_sha256": (
                                self._sha256(Path(item.ca_file))
                                if item.ca_file and Path(item.ca_file).is_file()
                                else None
                            ),
                        }
                        for item in config.clusters
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self.dcgm_digest = hashlib.sha256(
            (
                self.dcgm_exporter_image
                + "\0"
                + config.delivery_component_digests.get("dcgm", "")
                + "\0"
                + self._sha256(ROOT / "deploy/dataplane/dcgm-counters.csv")
            ).encode()
        ).hexdigest()
        self.observability_rules_digest = hashlib.sha256(
            (
                self._sha256(ROOT / "deploy/observability/amp-rules.yaml")
                + "\0"
                + self._sha256(ROOT / "deploy/observability/amp-alertmanager.yaml")
            ).encode()
        ).hexdigest()
        self.observability_adot_digest = self._sha256(
            ROOT / "deploy/observability/adot-control-plane.yaml"
        )
        self.rendered_manifest_digest = rendered_release_manifest_sha256(self)

    def _release_image(
        self,
        name: str,
        environment_name: str,
        legacy_default: str,
    ) -> str:
        configured = os.getenv(environment_name, "").strip()
        if self.config.release_manifest_schema_version < 3:
            return configured or legacy_default
        locked = self.config.locked_images[name]
        source = str(
            (
                self.config.release_delivery_identity.get("images", {})
                .get(name, {})
                .get("source")
                or ""
            )
        )
        if not configured or configured in {source, locked}:
            return locked
        locked_digest = locked.rsplit("@sha256:", 1)[-1]
        if configured.endswith(f"@sha256:{locked_digest}"):
            return configured
        raise ReleaseError(
            f"{environment_name} does not match the schema v3 image lock"
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _cpu(self, *args: str) -> list[str]:
        return [
            "kubectl",
            "--kubeconfig",
            self.config.cpu_kubeconfig,
            *args,
        ]

    @staticmethod
    def _gpu(target: ClusterTarget, *args: str) -> list[str]:
        return [
            "kubectl",
            "--context",
            target.context,
            *args,
        ]

    _remote_commands_are_idle = remote_commands_are_idle
    _ensure_profile_transition_safe = ensure_profile_transition_safe

    def gpu_cluster_rollout_step(self) -> str:
        # The plan is what an operator reads before approving a release, so it has
        # to state the parallelism this site actually configured: cross-cluster
        # parallelism is opt-in and defaults to one cluster at a time.
        parallel = self.config.upgrade_max_parallel_clusters
        if parallel <= 1:
            return "roll affected GPU clusters one at a time"
        return (
            "roll affected GPU clusters with bounded parallelism "
            f"(up to {parallel} at a time)"
        )

    def plan(self, mode: str) -> list[str]:
        steps = build_release_plan(mode)
        if mode != "deploy":
            return steps
        try:
            diff = classify_release(self, self._load_state())
        except Exception:
            return steps
        if diff.kind is ReleaseChangeKind.NOOP:
            execution = [
                "run read-only CPU/GPU verifiers",
                "re-assert the Completion Watcher state objects on every GPU cluster",
                "skip artifact upload, schema, endpoint, DCGM and rollouts",
            ]
        elif diff.kind is ReleaseChangeKind.CONTROL_PLANE_ONLY:
            execution = [
                "upload the control-plane wheel only",
                "roll CPU roles once and preserve all GPU/node artifacts",
                "run final verifiers",
            ]
        elif diff.kind is ReleaseChangeKind.DATA_PLANE_COMPATIBLE:
            execution = [
                "upload only changed Executor/Node artifacts",
                "stage compatibility pins only when an artifact pin changed",
                self.gpu_cluster_rollout_step(),
                "finalize changed pins and run verifiers",
            ]
        else:
            execution = steps
        return [
            f"release classification: {diff.kind.value}",
            "changed inputs: " + (", ".join(sorted(diff.changed)) or "none"),
            *execution,
        ]

    def status(self) -> dict[str, Any]:
        self._apply_health_baseline()
        return build_full_status(self)

    def _apply_health_baseline(self) -> None:
        state = self._load_state()
        previous = state.get("previous")
        rollback_result = state.get("rollback_result")
        if (
            str(state.get("phase") or "") == "rolled-back"
            and isinstance(rollback_result, dict)
            and rollback_result.get("status") == "PASSED"
            and isinstance(previous, dict)
        ):
            self.runtime_image = require_digest_pinned_image(
                "rolled-back runtime",
                previous.get("runtime_image")
                or previous.get("live_runtime_image")
                or self.runtime_image,
            )
            self.node_installer_image = require_digest_pinned_image(
                "rolled-back Node Installer",
                previous.get("node_installer_image") or self.node_installer_image,
            )
            self.adot_image = str(previous.get("adot_image") or self.adot_image)
            dcgm_images = {
                str(item.get("dcgm_image"))
                for item in (previous.get("clusters") or {}).values()
                if isinstance(item, dict) and item.get("dcgm_image")
            }
            if len(dcgm_images) > 1:
                raise ReleaseError("rolled-back clusters disagree on the DCGM image")
            if dcgm_images:
                self.dcgm_exporter_image = next(iter(dcgm_images))
            return
        adopted_runtime = str(state.get("adopted_live_runtime_image") or "").strip()
        if adopted_runtime:
            self.runtime_image = adopted_runtime

    def pin_approved_manifest_plan(self, digest: str | None) -> None:
        """Bind apply to the rendered-manifest digest the plan was approved on.

        Once pinned, every render of the release payload is verified against
        this digest, so a working-tree change made after the plan was approved
        and before it is applied is refused rather than silently applied.
        """

        self.approved_manifest_digest = str(digest).strip() if digest else None

    def enforce_manifest_plan_pin(self) -> None:
        """Fail closed if the working tree drifted from the approved plan.

        A no-op until ``pin_approved_manifest_plan`` has been given a digest.
        Rendering the payload raises when the recomputed digest does not match
        the pin.
        """

        if self.approved_manifest_digest:
            render_release_payload(self)

    def _apply_cpu(
        self,
        *,
        finalize: bool,
        force_restart: bool = False,
        diff: ReleaseDiff | None = None,
    ) -> None:
        """Apply the control-plane roles this release changes.

        Whole roles, one invocation. The apply script cannot be driven per role
        while the image moves: it verifies all three tiers against the candidate
        image afterwards, compares the ingress and spool ConfigMaps against each
        other, and consumes the pin-metadata change in the first invocation --
        so a second, narrower invocation would see no pin change and skip the
        restart that puts those roles on the new compatibility window.
        """

        # Refuse to apply CPU manifests that no longer match the approved plan.
        self.enforce_manifest_plan_pin()
        ensure_notification_secret(self)
        environment = build_cpu_apply_environment(self, finalize=finalize)
        environment["GPU_FAULT_CONTROL_PLANE_ROLE_TARGETS"] = ",".join(
            control_plane_role_targets(diff)
            if diff is not None
            else ("spool", "worker", "ingress")
        )
        if force_restart:
            environment["GPU_FAULT_FORCE_ROLE_RESTART"] = "true"
        render_and_apply_cpu_roles(self, environment)
        cronjob_exists = self.runner.probe(
            self._cpu(
                "-n",
                self.config.namespace,
                "get",
                "cronjob",
                "gpu-fault-aurora-credential-refresh",
            )
        )
        if cronjob_exists:
            self.runner.run(
                self._cpu(
                    "-n",
                    self.config.namespace,
                    "set",
                    "image",
                    "cronjob/gpu-fault-aurora-credential-refresh",
                    f"refresh={self.runtime_image}",
                )
            )

    def _apply_observability(self) -> None:
        topic_name = self.config.health.sns_topic_arn.rsplit(":", 1)[-1]
        environment = {
            **os.environ,
            "AWS_REGION": self.config.aws_region,
            "CPU_EKS_CLUSTER": self.config.cpu_eks_arn.rsplit("/", 1)[-1],
            "CPU_KUBECONFIG": self.config.cpu_kubeconfig,
            "AMP_WORKSPACE_ID": self.config.health.amp_workspace_id,
            "SNS_TOPIC_NAME": topic_name,
            "NAMESPACE": self.config.namespace,
            "RULE_NAMESPACE": self.config.health.amp_rule_namespace,
            "GPU_FAULT_ADOT_IMAGE": self.adot_image,
            "GPU_FAULT_ENABLE_ADOT": "true",
            "GPU_FAULT_ENABLE_AMP": "true",
            "GPU_FAULT_REQUIRE_CONFIRMED_SNS_SUBSCRIPTION": str(
                self.config.health.require_confirmed_sns_subscription
            ).lower(),
        }
        if self.config.notifications.admin_email:
            environment["GPU_FAULT_ALERT_EMAIL"] = self.config.notifications.admin_email
        self.runner.run(
            [
                "bash",
                str(ROOT / "deploy/observability/install-amp-monitoring.sh"),
            ],
            env=environment,
        )

    def _restore_cpu_role_config_maps(
        self,
        snapshots: object,
    ) -> bool:
        if not snapshots:
            return False
        if not isinstance(snapshots, dict):
            raise ReleaseError("previous CPU role ConfigMap snapshot is invalid")
        for name, raw_data in sorted(snapshots.items()):
            if (
                not isinstance(name, str)
                or not name.startswith("gpu-fault-")
                or "-config-" not in name
                or not isinstance(raw_data, dict)
            ):
                raise ReleaseError("previous CPU role ConfigMap snapshot is invalid")
            data = {str(key): str(value) for key, value in raw_data.items()}
            sensitive = sorted(
                key
                for key in data
                if any(marker in key for marker in SENSITIVE_CONFIG_MARKERS)
            )
            if sensitive:
                raise ReleaseError(
                    f"previous role ConfigMap {name} contains "
                    "sensitive-looking keys: " + ", ".join(sensitive)
                )
            document = {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": name,
                    "namespace": self.config.namespace,
                },
                "data": data,
            }
            self.runner.run(
                self._cpu("apply", "-f", "-"),
                input_text=json.dumps(document),
                sensitive=True,
            )
        return True

    _backup_secret = backup_secret
    _restore_secret = restore_secret
    _backup_release_secrets = backup_release_secrets
    _delete_release_secret_backups = delete_release_secret_backups
    _target_node_names = target_node_names
    _fleet_command = fleet_command
    _fleet_deployment_id = fleet_deployment_id
    _roll_node_runtime = roll_node_runtime
    _preflight_node_runtime = preflight_node_runtime

    _deploy_reconciler = deploy_reconciler
    _wait_agents = wait_agents
    _agent_heartbeats_converged = agent_heartbeats_converged
    _wait_candidate_cpu_agent_heartbeats = wait_candidate_cpu_agent_heartbeats
    _capture_active_agent_node_sets = capture_active_agent_node_sets
    _candidate_agent_pin_identity = candidate_agent_pin_identity
    _validate_resume_checkpoint = validate_resume_checkpoint

    def _validate_release(self) -> None:
        validate_release_components(
            self,
            cpu=True,
            data_plane=True,
        )

    _validate_release_quick = validate_release_quick
    _critical_amp_alerts = critical_amp_alerts
    _store_io_rejection_series_ready = store_io_rejection_series_ready
    _stability_snapshot = stability_snapshot
    validate_stability_window = validate_stability_window
    _validate_rollback = validate_rollback

    def noop(self, diff: ReleaseDiff) -> None:
        self._ensure_contexts()
        self._require_cpu_secrets()
        # The only sanctioned way to put a deleted watcher state ConfigMap or
        # ClusterRole back on an unchanged release (the component digest gate
        # would otherwise never touch the watcher again).
        for target in self.config.clusters:
            self._reassert_completion_watcher_state(target)
        self._validate_release()
        self._save_state("complete", release_diff=diff.as_dict())

    upgrade = upgrade_release
    rollback = rollback_release
    commit_release = commit_release

    def _config_map_sha(
        self,
        kubectl: list[str],
        name: str,
        preferred_key: str,
    ) -> str:
        value = self._get_json(
            kubectl + ["-n", self.config.namespace, "get", "configmap", name]
        )
        found = artifact_binary_sha(value.get("binaryData") or {}, preferred_key)
        if found is None:
            raise ReleaseError(f"{name} has no binaryData")
        return found[1]

    def bootstrap(self) -> None:
        self._ensure_contexts()
        if self.state.get("phase") in {
            "bootstrap-cleanup-started",
            "bootstrap-cleanup-progress",
            "bootstrap-cleanup-failed",
        } or (
            self.state.get("phase") == "bootstrap-failed" and self.config.auto_rollback
        ):
            self._cleanup_bootstrap()
        prerequisites = (
            (
                ROOT
                / "deploy/control-plane/regional/regional-control-plane-prerequisites.yaml"
            )
            .read_text(encoding="utf-8")
            .replace(
                "namespace: gpu-fault-system",
                f"namespace: {self.config.namespace}",
            )
        )
        prerequisites = prerequisites.replace(
            "kind: Namespace\nmetadata:\n  name: gpu-fault-system",
            f"kind: Namespace\nmetadata:\n  name: {self.config.namespace}",
        )
        self.runner.run(
            self._cpu("apply", "-f", "-"),
            input_text=prerequisites,
        )
        # The namespace now exists; ship the RDS CA bundle before any Aurora
        # consumer (schema-ensure Job, control-plane roles) mounts it (M-6).
        self._apply_rds_ca_bundle()
        completed_cluster_ids, cpu_checkpoint, check_live_cpu = (
            bootstrap_resume_context(self.state, self.release_id)
        )
        if not cpu_checkpoint and check_live_cpu:
            cpu_checkpoint = self._bootstrap_cpu_is_current()
        checkpoint = "bootstrap-started"
        self._save_state(
            checkpoint,
            previous=None,
            transaction_committed=False,
            completed_cluster_ids=sorted(completed_cluster_ids),
        )
        try:
            self._require_cpu_secrets(include_registry=False)
            self._initialize_registry()
            self._require_cpu_secrets()
            self._upload_release()
            if not cpu_checkpoint:
                self._ensure_schema()
                self._apply_cpu(finalize=True)
            ensure_runtime_profile(self)
            checkpoint = "bootstrap-cpu-ready"
            self._save_state(
                checkpoint,
                previous=None,
                completed_cluster_ids=sorted(completed_cluster_ids),
            )
            self._apply_nlb()
            checkpoint = "bootstrap-endpoint-ready"
            self._save_state(
                checkpoint,
                previous=None,
                completed_cluster_ids=sorted(completed_cluster_ids),
            )
            bootstrap_gpu_clusters(self, completed_cluster_ids)
            if completed_cluster_ids:
                checkpoint = "bootstrap-data-plane-progress"
            self._validate_release()
            self._save_state(
                "complete",
                previous=None,
                completed_cluster_ids=sorted(completed_cluster_ids),
            )
        except Exception:
            self._save_state(
                "bootstrap-failed",
                previous=None,
                resume_phase=checkpoint,
                completed_cluster_ids=sorted(completed_cluster_ids),
            )
            if self.config.auto_rollback:
                self._cleanup_bootstrap()
            raise

    def _cleanup_bootstrap(self) -> None:
        completed = set(self.state.get("bootstrap_cleanup_completed_steps") or [])

        def save(phase: str, **updates: Any) -> None:
            self._save_state(
                phase,
                previous=None,
                resume_phase="bootstrap-started",
                completed_cluster_ids=[],
                bootstrap_cleanup_completed_steps=sorted(completed),
                **updates,
            )

        save("bootstrap-cleanup-started")
        try:
            if "installer-jobs-cancelled" not in completed:
                for target in self.config.clusters:
                    self._scale_if_present(
                        self._gpu(target),
                        inventory.GPU_RECONCILER_DEPLOYMENT,
                        0,
                        wait=True,
                    )
                    self._cancel_active_installer_jobs(target)
                completed.add("installer-jobs-cancelled")
                save("bootstrap-cleanup-progress")
            if "gpu-scaled-down" not in completed:
                for target in self.config.clusters:
                    for deployment in inventory.DEPLOYMENTS:
                        self._scale_if_present(
                            self._gpu(target),
                            deployment,
                            0,
                            wait=True,
                        )
                completed.add("gpu-scaled-down")
                save("bootstrap-cleanup-progress")
            if "cpu-scaled-down" not in completed:
                for deployment in inventory.CPU_DEPLOYMENTS:
                    self._scale_if_present(
                        self._cpu(),
                        deployment,
                        0,
                        wait=True,
                    )
                completed.add("cpu-scaled-down")
                save("bootstrap-cleanup-progress")
        except Exception as exc:
            save(
                "bootstrap-cleanup-failed",
                bootstrap_cleanup_failure=f"{type(exc).__name__}: {exc}",
            )
            raise
        save("bootstrap-cleaned")

    def _scale_if_present(
        self,
        kubectl: list[str],
        deployment: str,
        replicas: int,
        *,
        wait: bool = False,
    ) -> None:
        exists = self.runner.probe(
            kubectl
            + [
                "-n",
                self.config.namespace,
                "get",
                "deployment",
                deployment,
            ]
        )
        if exists:
            self.runner.run(
                kubectl
                + [
                    "-n",
                    self.config.namespace,
                    "scale",
                    f"deployment/{deployment}",
                    f"--replicas={replicas}",
                ]
            )
            if wait:
                self.runner.run(
                    kubectl
                    + [
                        "-n",
                        self.config.namespace,
                        "rollout",
                        "status",
                        f"deployment/{deployment}",
                        "--timeout=5m",
                    ]
                )

    def join_cluster(self, cluster_id: str) -> None:
        target = join_target(self, cluster_id)
        self._ensure_contexts()
        self._update_registry(target, remove=False)
        prepare_join_registry(self, cluster_id)
        ensure_runtime_profile(self)
        self._ensure_gpu_namespace(target)
        self._ensure_connection_secret(target)
        self._quiesce_gpu_executor(target)
        self._verify_gpu_control_plane_endpoint(target)
        self._apply_gpu_dcgm_exporter(target)
        self._upload_config_map(
            self._gpu(target),
            self.executor_wheel_cm,
            self.config.executor_wheel.name,
            self.config.executor_wheel,
            self.executor_wheel_sha,
            compress=True,
        )
        self._upload_config_map(
            self._gpu(target),
            self.bundle_cm,
            self.config.bundle.name,
            self.config.bundle,
            self.bundle_sha,
        )
        metadata = self._config_map_data("gpu-fault-release-metadata")
        required = metadata.get("required-agent-artifact-sha256")
        required_executor = metadata.get("required-regional-executor-artifact-sha256")
        if (
            required != self.node_wheel_sha
            or required_executor != self.executor_wheel_sha
        ):
            raise ReleaseError(
                "join-cluster must use the current required Agent and Executor artifacts"
            )
        self._apply_gpu_deployments(target, self.executor_wheel_cm)
        self._roll_node_runtime(
            target,
            phase="join",
            wheel_cm=self.executor_wheel_cm,
            bundle_cm=self.bundle_cm,
            artifact_sha=self.node_wheel_sha,
            config_digest=self.config.agent_config_digest,
        )

    def activate_cluster(self, cluster_id: str) -> None:
        self._target(cluster_id)
        activate_join_registry(self, cluster_id)

    def remove_cluster(self, cluster_id: str) -> None:
        target = self._target(cluster_id)
        revoke_registry_cluster(self, cluster_id)
        for deployment in (
            *inventory.DEPLOYMENTS,
            inventory.GPU_RECONCILER_DEPLOYMENT,
        ):
            self._scale_if_present(self._gpu(target), deployment, 0)
        self._update_registry(target, remove=True)
        purge_registry_cluster(self, cluster_id)

    def fail_cluster(self, cluster_id: str) -> None:
        self._target(cluster_id)
        fail_join_registry(self, cluster_id)

    def rollback_cluster(self, cluster_id: str) -> None:
        self._target(cluster_id)
        rollback_join_registry(self, cluster_id)

    def _target(self, cluster_id: str) -> ClusterTarget:
        for target in self.config.clusters:
            if target.cluster_id == cluster_id:
                return target
        raise ReleaseError(f"unknown cluster_id: {cluster_id}")

    def _roll_cpu_for_registry(self) -> None:
        for deployment in inventory.CPU_RUNTIME_DEPLOYMENTS:
            self.runner.run(
                self._cpu(
                    "-n",
                    self.config.namespace,
                    "rollout",
                    "restart",
                    f"deployment/{deployment}",
                )
            )
            self.runner.run(
                self._cpu(
                    "-n",
                    self.config.namespace,
                    "rollout",
                    "status",
                    f"deployment/{deployment}",
                    f"--timeout={FAST_ROLLOUT_TIMEOUT}",
                )
            )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Regional GPU fault release orchestrator"
    )
    value.add_argument(
        "mode",
        choices=(
            "plan",
            "preflight",
            "release-summary",
            "release-diff",
            "status",
            "bootstrap",
            "deploy",
            "upgrade",
            "resume",
            "rollback",
            "commit",
            "stage-noop",
            "join-cluster",
            "activate-cluster",
            "fail-cluster",
            "rollback-cluster",
            "drain-cluster",
            "remove-cluster",
            "sync-state",
            "verify",
            "stability",
        ),
    )
    value.add_argument("--config", required=True, type=Path)
    value.add_argument("--cluster-id")
    value.add_argument(
        "--plan-mode",
        choices=(
            "bootstrap",
            "deploy",
            "upgrade",
            "rollback",
            "join-cluster",
            "remove-cluster",
        ),
        default="upgrade",
    )
    value.add_argument("--dry-run", action="store_true")
    # `rollback` only: set by the release driver's automatic rollback so the
    # in-flight install check proceeds (logging) when the store cannot answer.
    value.add_argument("--automatic", action="store_true")
    return value


def _run_mode(arguments: argparse.Namespace) -> int:
    config = ReleaseConfig.load(arguments.config)
    release = RegionalRelease(config, Runner(dry_run=arguments.dry_run))
    exit_code = 0
    if arguments.mode == "plan":
        print(
            json.dumps(
                release.plan(arguments.plan_mode),
                indent=2,
                ensure_ascii=False,
            )
        )
    elif arguments.mode == "preflight":
        report = build_preflight_report(release)
        print(json.dumps(report, indent=2, sort_keys=True))
        exit_code = report_exit_code(report)
    elif arguments.mode == "status":
        report = release.status()
        print(json.dumps(report, indent=2, sort_keys=True))
        exit_code = 0 if report.get("healthy") else 1
    elif arguments.mode == "release-summary":
        report = build_release_summary(release)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif arguments.mode == "release-diff":
        report = build_release_diff(release)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif arguments.mode == "bootstrap":
        release.bootstrap()
    elif arguments.mode == "deploy":
        run_deploy(release)
    elif arguments.mode == "upgrade":
        release.upgrade()
    elif arguments.mode == "resume":
        run_resume(release)
    elif arguments.mode == "rollback":
        release.rollback(automatic=bool(getattr(arguments, "automatic", False)))
    elif arguments.mode == "commit":
        release.commit_release()
    elif arguments.mode == "stage-noop":
        stage_noop_release(release)
    elif arguments.mode == "join-cluster":
        if not arguments.cluster_id:
            raise ReleaseError("--cluster-id is required")
        release.join_cluster(arguments.cluster_id)
    elif arguments.mode == "activate-cluster":
        if not arguments.cluster_id:
            raise ReleaseError("--cluster-id is required")
        release.activate_cluster(arguments.cluster_id)
    elif arguments.mode == "fail-cluster":
        if not arguments.cluster_id:
            raise ReleaseError("--cluster-id is required")
        release.fail_cluster(arguments.cluster_id)
    elif arguments.mode == "rollback-cluster":
        if not arguments.cluster_id:
            raise ReleaseError("--cluster-id is required")
        release.rollback_cluster(arguments.cluster_id)
    elif arguments.mode == "drain-cluster":
        if not arguments.cluster_id:
            raise ReleaseError("--cluster-id is required")
        drain_registry_cluster(release, arguments.cluster_id)
    elif arguments.mode == "remove-cluster":
        if not arguments.cluster_id:
            raise ReleaseError("--cluster-id is required")
        release.remove_cluster(arguments.cluster_id)
    elif arguments.mode == "sync-state":
        sync_release_state(release)
    elif arguments.mode == "verify":
        release._apply_health_baseline()
        report = build_health_report(release, mode="verify")
        print(json.dumps(report, indent=2, sort_keys=True))
        exit_code = report_exit_code(report)
    elif arguments.mode == "stability":
        report = release.validate_stability_window()
        print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


def main() -> int:
    arguments = parser().parse_args()
    narrate_release_start(arguments.mode, dry_run=arguments.dry_run)
    exit_code = 2
    try:
        exit_code = _run_mode(arguments)
    except InflightInstallsRefused as exc:
        # Nothing was changed; the release driver classifies on this code.
        exit_code = INFLIGHT_INSTALLS_REFUSED_EXIT_CODE
        print(f"ERROR: {exc}", file=sys.stderr)
    except (
        ReleaseError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
    finally:
        # `finally` rather than the success path, so an invocation that died --
        # the case where an operator most needs to know it is over and how long
        # it lasted -- still closes its own block of the log.
        narrate_release_end(arguments.mode, exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
