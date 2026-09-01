#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import regional_deployment_inventory as inventory
from regional_admin_checks import (
    build_health_report,
    build_preflight_report,
    report_exit_code,
)
from regional_admin_commands import (
    bootstrap_cpu_is_current,
    build_full_status,
    build_release_summary,
    ensure_schema,
    run_deploy,
    run_resume,
)
from regional_dns import apply_control_plane_nlb
from regional_gpu_bootstrap import (
    apply_gpu_dcgm_exporter,
    cancel_active_installer_jobs,
    ensure_connection_secret,
    ensure_gpu_namespace,
    quiesce_gpu_executor,
    retry_failed_installer_jobs,
    verify_gpu_control_plane_endpoint,
)
from regional_notifications import (
    ensure_notification_secret,
    notification_digest,
)
from regional_release_config import (
    ClusterTarget,
    ReleaseConfig,
    ReleaseError,
)
from regional_release_diff import (
    ReleaseChangeKind,
    ReleaseDiff,
    classify_release,
    control_plane_role_targets,
)
from regional_release_fleet_rollout import (
    agent_heartbeats_converged,
    backup_release_secrets,
    backup_secret,
    deploy_reconciler,
    delete_release_secret_backups,
    fleet_command,
    fleet_deployment_id,
    restore_secret,
    roll_node_runtime,
    target_node_names,
    wait_agents,
)
from regional_release_gpu_rollout import (
    agents_converged as agents_converged,
    apply_gpu_deployments,
    executor_pin_rejection as executor_pin_rejection,
    join_target,
    require_executor_pin as require_executor_pin,
    upgrade_gpu_target,
)
from regional_release_iam import (
    validate_executor_iam_documents as validate_executor_iam_documents,
)
from regional_release_iam import (
    validate_executor_iam_role,
)
from regional_release_orchestration import (
    build_rollback_environment as build_rollback_environment,
    commit_release,
    rollback_release,
    upgrade_release,
)
from regional_release_preflight import ensure_region_contexts
from regional_release_rendering import (
    DEFAULT_DCGM_EXPORTER_IMAGE,
    DEFAULT_RUNTIME_IMAGE,
    build_cpu_apply_environment,
    rendered_release_manifest_sha256,
    stamp_gpu_deployments,
)
from regional_release_reporting import build_release_plan
from regional_release_runtime_identity import (
    CONTROL_PLANE_PYTHON,
    validate_runtime_component_identity,
)
from regional_release_registry import (
    commit_registry_update,
    desired_registry,
    registry,
    registry_config_digest,
    registry_entry,
    registry_payloads,
    restore_registry_backup,
    stage_registry,
    update_registry,
    write_registry,
)
from regional_release_online_registry import (
    activate_join_registry,
    drain_registry_cluster,
    prepare_join_registry,
    purge_registry_cluster,
    revoke_registry_cluster,
)
from regional_release_state import (
    STATE_CONFIG_MAP as STATE_CONFIG_MAP,
)
from regional_release_state import (
    capture_previous,
    config_map_binary_key,
    config_map_data,
    deployment_template_name,
    deployment_wheel,
    get_json,
    load_state,
    save_state,
    template_bundle,
)
from regional_release_validation import (
    critical_amp_alerts,
    ensure_profile_transition_safe,
    stability_snapshot,
    store_io_rejection_series_ready,
    validate_release_quick,
    validate_rollback,
    validate_stability_window,
)
from regional_runtime_profile import (
    ensure_runtime_profile,
    runtime_profile_policy_digest,
)

ROOT = Path(__file__).resolve().parents[3]
CONTROL_PLANE_VERIFY_SCRIPT = (
    ROOT / "deploy/control-plane/tools/verify-control-plane-role-split.sh"
)
DATA_PLANE_VERIFY_SCRIPT = ROOT / "deploy/dataplane/tools/verify-dataplane-executor.sh"
FAST_ROLLOUT_TIMEOUT = "5m"
SENSITIVE_CONFIG_MARKERS = (
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "CREDENTIAL",
    "PRIVATE_KEY",
)


class Runner:
    def __init__(self, *, dry_run: bool = False) -> None:
        self.dry_run = dry_run

    def run(
        self,
        args: list[str],
        *,
        env: dict[str, str] | None = None,
        input_text: str | None = None,
        capture: bool = False,
        sensitive: bool = False,
    ) -> str:
        shown = [Path(args[0]).name, *args[1:]]
        print(
            "+ " + ("<sensitive command>" if sensitive else " ".join(shown)),
            file=sys.stderr,
            flush=True,
        )
        if self.dry_run and not capture:
            return ""
        completed = subprocess.run(
            args,
            check=False,
            text=True,
            input=input_text,
            capture_output=capture,
            env=env,
        )
        if completed.returncode:
            if capture and completed.stderr:
                print(completed.stderr, file=sys.stderr)
            raise ReleaseError(f"command failed ({completed.returncode}): {args[0]}")
        return completed.stdout.strip() if capture else ""


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
    cleaned = loaded_phase == "bootstrap-cleaned"
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
    _ensure_contexts = ensure_region_contexts
    _apply_gpu_dcgm_exporter = apply_gpu_dcgm_exporter
    _cancel_active_installer_jobs = cancel_active_installer_jobs
    _apply_gpu_deployments = apply_gpu_deployments
    _apply_nlb = apply_control_plane_nlb
    _bootstrap_cpu_is_current = bootstrap_cpu_is_current
    _capture_previous = capture_previous
    _config_map_binary_key = config_map_binary_key
    _config_map_data = config_map_data
    _deployment_template_name = deployment_template_name
    _deployment_wheel = deployment_wheel
    _ensure_connection_secret = ensure_connection_secret
    _ensure_gpu_namespace = ensure_gpu_namespace
    _ensure_schema = ensure_schema
    _get_json = get_json
    _load_state = load_state
    _quiesce_gpu_executor = quiesce_gpu_executor
    _registry = registry
    _registry_entry = staticmethod(registry_entry)
    _registry_payloads = registry_payloads
    _retry_failed_installer_jobs = retry_failed_installer_jobs
    _restore_registry_backup = restore_registry_backup
    _save_state = save_state
    _stage_registry = stage_registry
    _stamp_gpu_deployments = stamp_gpu_deployments
    _template_bundle = template_bundle
    _update_registry = update_registry
    _upgrade_gpu_target = upgrade_gpu_target
    _validate_executor_iam_role = validate_executor_iam_role
    _verify_gpu_control_plane_endpoint = verify_gpu_control_plane_endpoint
    _write_registry = write_registry
    _desired_registry = desired_registry
    _commit_registry_update = commit_registry_update

    def __init__(self, config: ReleaseConfig, runner: Runner) -> None:
        self.config = config
        self.runner = runner
        self.release_id = config.release_id
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

    def _require_cpu_secrets(self, *, include_registry: bool = True) -> None:
        names = [
            "gpu-fault-aurora",
            "gpu-fault-control-plane-active",
            "gpu-fault-node-action-keys",
        ]
        if include_registry:
            names.append("gpu-fault-regional-clusters")
        for name in names:
            self.runner.run(
                self._cpu(
                    "-n",
                    self.config.namespace,
                    "get",
                    "secret",
                    name,
                ),
                capture=True,
            )

    def _remote_commands_are_idle(self) -> bool:
        script = (
            "from gpu_fault.app import ApplicationContext;"
            "stats=ApplicationContext.from_environment()"
            ".store.remote_command_stats();"
            "bad={name:int(stats['by_status'].get(name,0)) "
            "for name in ('PENDING','LEASED','WAITING') "
            "if int(stats['by_status'].get(name,0))};"
            "print(bad if bad else '')"
        )
        for attempt in range(3):
            pod = self.runner.run(
                self._cpu(
                    "-n",
                    self.config.namespace,
                    "get",
                    "pod",
                    "-l",
                    f"app={inventory.CPU_INGRESS_DEPLOYMENT}",
                    "--field-selector=status.phase=Running",
                    "-o",
                    "jsonpath={.items[0].metadata.name}",
                ),
                capture=True,
            )
            if not pod:
                return True
            try:
                output = self.runner.run(
                    self._cpu(
                        "-n",
                        self.config.namespace,
                        "exec",
                        pod,
                        "--",
                        CONTROL_PLANE_PYTHON,
                        "-c",
                        script,
                    ),
                    capture=True,
                )
            except ReleaseError:
                if attempt == 2:
                    raise
                time.sleep(2)
                continue
            return not output.strip()
        raise ReleaseError("remote command idle check exhausted retries")

    _ensure_profile_transition_safe = ensure_profile_transition_safe

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
                "roll affected GPU clusters with bounded parallelism",
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
        return build_full_status(self)

    def _upload_config_map(
        self,
        kubectl: list[str],
        name: str,
        key: str,
        path: Path,
        expected_sha: str,
    ) -> None:
        exists = (
            subprocess.run(
                kubectl
                + [
                    "-n",
                    self.config.namespace,
                    "get",
                    "configmap",
                    name,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
        if not exists:
            self.runner.run(
                kubectl
                + [
                    "-n",
                    self.config.namespace,
                    "create",
                    "configmap",
                    name,
                    f"--from-file={key}={path}",
                ]
            )
        if self.runner.dry_run:
            return
        value = self._get_json(
            kubectl
            + [
                "-n",
                self.config.namespace,
                "get",
                "configmap",
                name,
            ]
        )
        encoded = (value.get("binaryData") or {}).get(key)
        if not encoded:
            raise ReleaseError(f"{name}/{key} is missing")
        if hashlib.sha256(base64.b64decode(encoded)).hexdigest() != expected_sha:
            raise ReleaseError(f"{name}/{key} digest mismatch")

    def _upload_release(self, diff: ReleaseDiff | None = None) -> None:
        wheel_key = self.config.wheel.name
        executor_key = self.config.executor_wheel.name
        bundle_key = self.config.bundle.name
        if diff is None or diff.has("control_plane_wheel"):
            self._upload_config_map(
                self._cpu(),
                self.wheel_cm,
                wheel_key,
                self.config.wheel,
                self.wheel_sha,
            )
        for target in self.config.clusters:
            kubectl = self._gpu(target)
            if diff is None or diff.has("executor_wheel"):
                self._upload_config_map(
                    kubectl,
                    self.executor_wheel_cm,
                    executor_key,
                    self.config.executor_wheel,
                    self.executor_wheel_sha,
                )
            if diff is None or diff.has("node_runtime_wheel", "node_bundle"):
                self._upload_config_map(
                    kubectl,
                    self.bundle_cm,
                    bundle_key,
                    self.config.bundle,
                    self.bundle_sha,
                )

    def _apply_cpu(
        self,
        *,
        finalize: bool,
        force_restart: bool = False,
        diff: ReleaseDiff | None = None,
    ) -> None:
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
        cronjob = subprocess.run(
            self._cpu(
                "-n",
                self.config.namespace,
                "get",
                "cronjob",
                "gpu-fault-aurora-credential-refresh",
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if cronjob.returncode == 0:
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

    _deploy_reconciler = deploy_reconciler
    _wait_agents = wait_agents
    _agent_heartbeats_converged = agent_heartbeats_converged

    def _validate_release(self) -> None:
        self.runner.run(
            [
                "bash",
                str(CONTROL_PLANE_VERIFY_SCRIPT),
            ],
            env={
                **os.environ,
                "KUBECONFIG": self.config.cpu_kubeconfig,
                "GPU_FAULT_NAMESPACE": self.config.namespace,
            },
        )
        for target in self.config.clusters:
            self.runner.run(
                [
                    "bash",
                    str(DATA_PLANE_VERIFY_SCRIPT),
                ],
                env={
                    **os.environ,
                    "GPU_FAULT_NAMESPACE": self.config.namespace,
                    "GPU_FAULT_KUBE_CONTEXT": target.context,
                    "GPU_FAULT_CONTROL_PLANE_KUBECONFIG": (self.config.cpu_kubeconfig),
                    "GPU_FAULT_EXPECTED_WHEEL_CONFIGMAP": (self.executor_wheel_cm),
                },
            )
        validate_runtime_component_identity(self)

    _validate_release_quick = validate_release_quick
    _critical_amp_alerts = critical_amp_alerts
    _store_io_rejection_series_ready = store_io_rejection_series_ready
    _stability_snapshot = stability_snapshot
    validate_stability_window = validate_stability_window
    _validate_rollback = validate_rollback

    def noop(self, diff: ReleaseDiff) -> None:
        self._ensure_contexts()
        self._require_cpu_secrets()
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
            kubectl
            + [
                "-n",
                self.config.namespace,
                "get",
                "configmap",
                name,
            ]
        )
        binary = value.get("binaryData") or {}
        encoded = binary.get(preferred_key)
        if encoded is None and binary:
            encoded = next(iter(binary.values()))
        if encoded is None:
            raise ReleaseError(f"{name} has no binaryData")
        return hashlib.sha256(base64.b64decode(encoded)).hexdigest()

    def bootstrap(self) -> None:
        self._ensure_contexts()
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
        completed_cluster_ids, cpu_checkpoint, check_live_cpu = (
            bootstrap_resume_context(self.state, self.release_id)
        )
        if not cpu_checkpoint and check_live_cpu:
            cpu_checkpoint = self._bootstrap_cpu_is_current()
        checkpoint = "bootstrap-started"
        self._save_state(
            checkpoint,
            previous=None,
            completed_cluster_ids=sorted(completed_cluster_ids),
        )
        try:
            self._require_cpu_secrets(include_registry=False)
            for target in self.config.clusters:
                self._update_registry(target, remove=False)
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
            for target in self.config.clusters:
                if target.cluster_id in completed_cluster_ids:
                    continue
                self._ensure_gpu_namespace(target)
                self._ensure_connection_secret(target)
                self._quiesce_gpu_executor(target)
                self._verify_gpu_control_plane_endpoint(target)
                self._apply_gpu_dcgm_exporter(target)
                self._apply_gpu_deployments(target, self.executor_wheel_cm)
                self._roll_node_runtime(
                    target,
                    phase="bootstrap",
                    wheel_cm=self.executor_wheel_cm,
                    bundle_cm=self.bundle_cm,
                    artifact_sha=self.node_wheel_sha,
                    config_digest=self.config.agent_config_digest,
                )
                completed_cluster_ids.add(target.cluster_id)
                checkpoint = "bootstrap-data-plane-progress"
                self._save_state(
                    checkpoint,
                    previous=None,
                    completed_cluster_ids=sorted(completed_cluster_ids),
                )
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
        for target in self.config.clusters:
            for deployment in (
                *inventory.DEPLOYMENTS,
                inventory.GPU_RECONCILER_DEPLOYMENT,
            ):
                self._scale_if_present(self._gpu(target), deployment, 0)
        for deployment in inventory.CPU_DEPLOYMENTS:
            self._scale_if_present(self._cpu(), deployment, 0)
        self._save_state(
            "bootstrap-cleaned",
            previous=None,
            resume_phase="bootstrap-started",
            completed_cluster_ids=[],
        )

    def _scale_if_present(
        self,
        kubectl: list[str],
        deployment: str,
        replicas: int,
    ) -> None:
        exists = (
            subprocess.run(
                kubectl
                + [
                    "-n",
                    self.config.namespace,
                    "get",
                    "deployment",
                    deployment,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
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
            "status",
            "bootstrap",
            "deploy",
            "upgrade",
            "resume",
            "rollback",
            "commit",
            "join-cluster",
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
    return value


def main() -> int:
    arguments = parser().parse_args()
    try:
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
        elif arguments.mode == "bootstrap":
            release.bootstrap()
        elif arguments.mode == "deploy":
            run_deploy(release)
        elif arguments.mode == "upgrade":
            release.upgrade()
        elif arguments.mode == "resume":
            run_resume(release)
        elif arguments.mode == "rollback":
            release.rollback()
        elif arguments.mode == "commit":
            release.commit_release()
        elif arguments.mode == "join-cluster":
            if not arguments.cluster_id:
                raise ReleaseError("--cluster-id is required")
            release.join_cluster(arguments.cluster_id)
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
            report = build_health_report(release, mode="verify")
            print(json.dumps(report, indent=2, sort_keys=True))
            exit_code = report_exit_code(report)
        elif arguments.mode == "stability":
            report = release.validate_stability_window()
            print(json.dumps(report, indent=2, sort_keys=True))
        return exit_code
    except (
        ReleaseError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
