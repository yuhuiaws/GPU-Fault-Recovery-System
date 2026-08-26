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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import regional_deployment_inventory as inventory
import yaml
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
)
from regional_dns import apply_control_plane_nlb
from regional_gpu_bootstrap import (
    apply_gpu_dcgm_exporter,
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
)
from regional_release_iam import (
    validate_executor_iam_documents as validate_executor_iam_documents,
)
from regional_release_iam import (
    validate_executor_iam_role,
)
from regional_release_preflight import ensure_region_contexts
from regional_release_rendering import (
    DEFAULT_DCGM_EXPORTER_IMAGE,
    DEFAULT_RUNTIME_IMAGE,
    build_cpu_apply_environment,
    build_reconciler_environment,
    render_gpu_rollout_manifests,
)
from regional_release_reporting import build_release_plan
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
from regional_runtime_profile import (
    ensure_runtime_profile,
    runtime_profile_policy_digest,
)

ROOT = Path(__file__).resolve().parents[3]


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


def agents_converged(
    items: list[dict[str, Any]],
    target: ClusterTarget,
    artifact_sha: str,
) -> bool:
    nodes = [
        item
        for item in items
        if (
            item.get("metadata", {})
            .get("labels", {})
            .get("sagemaker.amazonaws.com/cluster-name")
            == target.hyperpod_cluster_name
        )
    ]
    aligned = [
        item
        for item in nodes
        if (
            item.get("metadata", {})
            .get("annotations", {})
            .get("gpu-fault.io/installer-state")
            == "Succeeded"
            and item.get("metadata", {})
            .get("annotations", {})
            .get("gpu-fault.io/installer-artifact-sha256")
            == artifact_sha
        )
    ]
    return bool(nodes) and len(aligned) == len(nodes)


def upgrade_gpu_target(
    release: Any,
    target: ClusterTarget,
    diff: ReleaseDiff,
) -> None:
    full = diff.kind == ReleaseChangeKind.FULL
    if full or diff.has("endpoint"):
        release._verify_gpu_control_plane_endpoint(target)
    if full or diff.has("dcgm"):
        release._apply_gpu_dcgm_exporter(target)
    if full or diff.has("executor_wheel"):
        release._apply_gpu_deployments(target, release.executor_wheel_cm)
    if full or diff.has(
        "executor_wheel",
        "node_runtime_wheel",
        "node_bundle",
        "agent_config",
        "agent_protocol",
        "runtime_profile",
        "runtime_profile_version",
    ):
        release._deploy_reconciler(
            target,
            wheel_cm=release.executor_wheel_cm,
            bundle_cm=release.bundle_cm,
            artifact_sha=release.node_wheel_sha,
            config_digest=release.config.agent_config_digest,
        )
        release._wait_agents(target, release.node_wheel_sha)


def join_target(release: Any, cluster_id: str) -> ClusterTarget:
    target = release._target(cluster_id)
    if not release._remote_commands_are_idle():
        raise ReleaseError("remote commands are PENDING/LEASED/WAITING")
    return target


def build_rollback_environment(
    *,
    rollback_config: ReleaseConfig,
    metadata: dict[str, str],
    cpu_wheel: str,
    cpu_sha: str,
    artifact: str,
    config_digest: str,
    runtime_profile_version: str,
) -> dict[str, str]:
    legacy_component_pins = not any(
        metadata.get(name)
        for name in (
            "required-agent-compatibility-digest",
            "required-regional-executor-artifact-sha256",
            "required-regional-executor-compatibility-digest",
        )
    )
    return {
        **os.environ,
        "KUBECONFIG": rollback_config.cpu_kubeconfig,
        "GPU_FAULT_AWS_REGION": rollback_config.aws_region,
        "GPU_FAULT_NAMESPACE": rollback_config.namespace,
        "GPU_FAULT_WHEEL_CONFIGMAP": cpu_wheel,
        "GPU_FAULT_WHEEL_SHA256": cpu_sha,
        "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": artifact,
        "GPU_FAULT_REQUIRED_AGENT_COMPATIBILITY_DIGEST": (
            metadata.get("required-agent-compatibility-digest") or artifact
        ),
        "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": config_digest,
        "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": runtime_profile_version,
        "GPU_FAULT_REQUIRED_AGENT_PROTOCOL_VERSION": metadata.get(
            "required-agent-protocol-version", "3"
        ),
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION": metadata.get(
            "required-regional-executor-protocol-version", "2"
        ),
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256": metadata.get(
            "required-regional-executor-artifact-sha256", ""
        ),
        "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST": (
            metadata.get("required-regional-executor-compatibility-digest")
            or metadata.get("required-regional-executor-artifact-sha256", "")
        ),
        "GPU_FAULT_ALLOW_EMAIL": str(rollback_config.notifications.allow_email).lower(),
        "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL": str(
            rollback_config.notifications.acknowledge_external_alert_channel
        ).lower(),
        "GPU_FAULT_NOTIFICATION_CONFIG_SHA256": notification_digest(
            rollback_config.notifications
        ),
        "GPU_FAULT_LEGACY_COMPONENT_PINS": str(legacy_component_pins).lower(),
        "GPU_FAULT_FINALIZE_AGENT_PIN": "true",
        "GPU_FAULT_FINALIZE_DATA_PLANE_PIN": "true",
    }


def sync_release_state(release: Any) -> None:
    release._save_state(
        "complete",
        previous=None,
        release_diff=ReleaseDiff(
            kind=ReleaseChangeKind.NOOP,
            changed=frozenset(),
        ).as_dict(),
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
    _retry_failed_installer_jobs = retry_failed_installer_jobs
    _save_state = save_state
    _template_bundle = template_bundle
    _upgrade_gpu_target = upgrade_gpu_target
    _validate_executor_iam_role = validate_executor_iam_role
    _verify_gpu_control_plane_endpoint = verify_gpu_control_plane_endpoint

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
        self.wheel_cm = "gpu-fault-control-plane-wheel-0100-" + self.wheel_sha[:12]
        self.executor_wheel_cm = (
            "gpu-fault-executor-wheel-0100-" + self.executor_wheel_sha[:12]
        )
        self.bundle_cm = "gpu-fault-node-installer-0100-" + self.bundle_sha[:12]
        self.runtime_image = os.getenv("GPU_FAULT_RUNTIME_IMAGE", DEFAULT_RUNTIME_IMAGE)
        self.dcgm_exporter_image = os.getenv(
            "GPU_FAULT_DCGM_EXPORTER_IMAGE",
            DEFAULT_DCGM_EXPORTER_IMAGE,
        )
        for variable, image in (
            ("GPU_FAULT_RUNTIME_IMAGE", self.runtime_image),
            ("GPU_FAULT_DCGM_EXPORTER_IMAGE", self.dcgm_exporter_image),
        ):
            if not image or any(
                character.isspace() or character == "#" for character in image
            ):
                raise ReleaseError(
                    f"{variable} must be a non-empty OCI image reference "
                    "without whitespace or #"
                )
        self.state: dict[str, Any] = {}
        self.endpoint_digest = hashlib.sha256(
            json.dumps(
                {
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
                + self._sha256(ROOT / "deploy/dataplane/dcgm-counters.csv")
            ).encode()
        ).hexdigest()

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
        pod = self.runner.run(
            self._cpu(
                "-n",
                self.config.namespace,
                "get",
                "pod",
                "-l",
                f"app={inventory.CPU_INGRESS_DEPLOYMENT}",
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ),
            capture=True,
        )
        if not pod:
            return True
        script = (
            "import urllib.request;"
            "t=urllib.request.urlopen("
            "'http://127.0.0.1:8080/metrics',timeout=10).read().decode();"
            "bad=[];"
            "\nfor s in ('PENDING','LEASED','WAITING'):\n"
            " line=next((x for x in t.splitlines() if "
            "x.startswith('gpu_fault_remote_command_total{status=\"'+s+'\"}')),None);"
            "\n if line and float(line.rsplit(' ',1)[1])!=0: bad.append(line)\n"
            "print('\\n'.join(bad))"
        )
        output = self.runner.run(
            self._cpu(
                "-n",
                self.config.namespace,
                "exec",
                pod,
                "--",
                "python",
                "-c",
                script,
            ),
            capture=True,
        )
        return not output.strip()

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

    def _apply_cpu(self, *, finalize: bool) -> None:
        ensure_notification_secret(self)
        self.runner.run(
            [
                "bash",
                str(
                    ROOT
                    / "deploy/control-plane/tools/apply-control-plane-role-split.sh"
                ),
            ],
            env=build_cpu_apply_environment(self, finalize=finalize),
        )

    def _apply_gpu_deployments(
        self,
        target: ClusterTarget,
        wheel_cm: str,
        *,
        runtime_profile_version: str | None = None,
        executor_wheel_filename: str | None = None,
    ) -> None:
        for deployment, text in render_gpu_rollout_manifests(
            self,
            target,
            wheel_cm,
            runtime_profile_version=runtime_profile_version,
            executor_wheel_filename=executor_wheel_filename,
        ):
            self.runner.run(
                self._gpu(target, "apply", "-f", "-"),
                input_text=text,
            )
            self.runner.run(
                self._gpu(
                    target,
                    "-n",
                    self.config.namespace,
                    "rollout",
                    "status",
                    f"deployment/{deployment}",
                    "--timeout=10m",
                )
            )

    def _stamp_gpu_deployments(self, text: str) -> str:
        documents = list(yaml.safe_load_all(text))
        for document in documents:
            if not isinstance(document, dict) or document.get("kind") != "Deployment":
                continue
            annotations = (
                document.setdefault("spec", {})
                .setdefault("template", {})
                .setdefault("metadata", {})
                .setdefault("annotations", {})
            )
            annotations.update(
                {
                    "gpu-fault.io/artifact-sha256": self.executor_wheel_sha,
                    "gpu-fault.io/release-rollout": self.release_id,
                    "gpu-fault.io/release-sha256": self.executor_wheel_sha,
                    "gpu-fault.io/release-wheel-sha256": (self.executor_wheel_sha),
                    "gpu-fault.io/executor-wheel-sha256": (self.executor_wheel_sha),
                    "gpu-fault.io/executor-compatibility-digest": (
                        self.config.component_digests.get("executor")
                        or self.executor_wheel_sha
                    ),
                    "gpu-fault.io/runtime-image": (self.runtime_image),
                }
            )
        return yaml.safe_dump_all(
            documents,
            sort_keys=False,
            width=72,
        )

    def _deploy_reconciler(
        self,
        target: ClusterTarget,
        *,
        wheel_cm: str,
        bundle_cm: str,
        artifact_sha: str,
        config_digest: str,
        runtime_profile_version: str | None = None,
        executor_wheel_filename: str | None = None,
    ) -> None:
        self._retry_failed_installer_jobs(target)
        environment = build_reconciler_environment(
            self,
            target,
            wheel_cm=wheel_cm,
            bundle_cm=bundle_cm,
            artifact_sha=artifact_sha,
            config_digest=config_digest,
            runtime_profile_version=runtime_profile_version,
            executor_wheel_filename=executor_wheel_filename,
        )
        self.runner.run(
            [str(ROOT / "deploy/node/deploy-node-installer-reconciler.sh")],
            env=environment,
            sensitive=bool(target.fleet_master_file),
        )

    def _wait_agents(
        self,
        target: ClusterTarget,
        artifact_sha: str,
        *,
        timeout_seconds: int = 900,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            value = self._get_json(self._gpu(target, "get", "nodes"))
            if agents_converged(value.get("items", []), target, artifact_sha):
                return
            if self.runner.dry_run:
                return
            time.sleep(5)
        raise ReleaseError(f"{target.cluster_id} agents did not converge")

    def _validate_release(self) -> None:
        self.runner.run(
            [
                "bash",
                str(
                    ROOT
                    / "deploy/control-plane/tools/verify-control-plane-role-split.sh"
                ),
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
                    str(ROOT / "deploy/dataplane/tools/verify-dataplane-executor.sh"),
                ],
                env={
                    **os.environ,
                    "GPU_FAULT_NAMESPACE": self.config.namespace,
                    "GPU_FAULT_KUBE_CONTEXT": target.context,
                    "GPU_FAULT_CONTROL_PLANE_KUBECONFIG": (self.config.cpu_kubeconfig),
                    "GPU_FAULT_EXPECTED_WHEEL_CONFIGMAP": (self.executor_wheel_cm),
                },
            )

    def noop(self, diff: ReleaseDiff) -> None:
        self._ensure_contexts()
        self._require_cpu_secrets()
        self._validate_release()
        self._save_state("complete", release_diff=diff.as_dict())

    def upgrade(
        self,
        *,
        resume: bool = False,
        diff: ReleaseDiff | None = None,
    ) -> None:
        self._ensure_contexts()
        self._require_cpu_secrets()
        if not self._remote_commands_are_idle():
            raise ReleaseError("remote commands are PENDING/LEASED/WAITING")
        previous = (
            self._load_state().get("previous") if resume else self._capture_previous()
        )
        if not previous:
            raise ReleaseError("previous release state is unavailable")
        active_diff = diff or ReleaseDiff(
            kind=ReleaseChangeKind.FULL,
            changed=frozenset(
                {
                    "control_plane_wheel",
                    "executor_wheel",
                    "node_runtime_wheel",
                    "node_bundle",
                    "database_schema",
                    "agent_protocol",
                    "executor_protocol",
                    "agent_config",
                    "runtime_profile",
                    "endpoint",
                    "dcgm",
                }
            ),
        )
        self._save_state(
            "preflight",
            previous=previous,
            release_diff=active_diff.as_dict(),
        )
        try:
            self._upload_release(active_diff)
            self._save_state("uploaded")
            if active_diff.has("database_schema"):
                self._ensure_schema()
            self._save_state("schema-ready")
            pin_changed = active_diff.has(
                "executor_wheel",
                "node_runtime_wheel",
                "agent_protocol",
                "executor_protocol",
                "agent_config",
            )
            control_changed = active_diff.has("control_plane_wheel", "notifications")
            full = active_diff.kind == ReleaseChangeKind.FULL
            if pin_changed or full:
                self._apply_cpu(finalize=False)
                self._save_state("cpu-staged")
            if full or active_diff.has(
                "runtime_profile",
                "runtime_profile_version",
            ):
                ensure_runtime_profile(self)
            if active_diff.kind != ReleaseChangeKind.CONTROL_PLANE_ONLY:
                workers = min(4, max(1, len(self.config.clusters)))
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(
                            self._upgrade_gpu_target,
                            target,
                            active_diff,
                        ): target.cluster_id
                        for target in self.config.clusters
                    }
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as exc:
                            raise ReleaseError(
                                f"{futures[future]} rollout failed: {exc}"
                            ) from exc
            self._save_state("data-converged")
            if pin_changed or full or control_changed:
                self._apply_cpu(finalize=True)
            self._validate_release()
            self._save_state(
                "complete",
                release_diff=active_diff.as_dict(),
            )
        except Exception:
            self._save_state("failed")
            if self.config.auto_rollback:
                self.rollback(state=previous)
            raise

    def rollback(self, *, state: dict[str, Any] | None = None) -> None:
        loaded = self._load_state() if state is None else {}
        previous = state or loaded.get("previous")
        if not previous:
            if loaded.get("phase") in {
                "bootstrap-started",
                "bootstrap-failed",
            }:
                self._cleanup_bootstrap()
                return
            raise ReleaseError("previous release state is unavailable")
        metadata = previous.get("metadata") or {}
        cpu_wheel = previous.get("cpu_wheel")
        artifact = metadata.get("required-agent-artifact-sha256")
        config_digest = metadata.get("required-agent-config-digest")
        runtime_profile_version = (
            previous.get("runtime_profile_version") or "hyperpod-v1"
        )
        if not all((cpu_wheel, artifact, config_digest)):
            raise ReleaseError("previous release pins are incomplete")
        cpu_sha = self._config_map_sha(self._cpu(), cpu_wheel, self.config.wheel.name)
        rollback_config = self.config.for_rollback(config_digest)
        environment = build_rollback_environment(
            rollback_config=rollback_config,
            metadata=metadata,
            cpu_wheel=cpu_wheel,
            cpu_sha=cpu_sha,
            artifact=artifact,
            config_digest=config_digest,
            runtime_profile_version=runtime_profile_version,
        )
        self.runner.run(
            [
                "bash",
                str(
                    ROOT
                    / "deploy/control-plane/tools/apply-control-plane-role-split.sh"
                ),
            ],
            env=environment,
        )
        for target in self.config.clusters:
            old = (previous.get("clusters") or {}).get(target.cluster_id, {})
            wheel = old.get("wheel")
            if wheel:
                self._verify_gpu_control_plane_endpoint(target)
                self._apply_gpu_dcgm_exporter(target)
                self._apply_gpu_deployments(
                    target,
                    wheel,
                    runtime_profile_version=runtime_profile_version,
                    executor_wheel_filename=old.get("wheel_key"),
                )
            if all(old.get(name) for name in ("reconciler_wheel", "bundle")):
                self._deploy_reconciler(
                    target,
                    wheel_cm=old["reconciler_wheel"],
                    bundle_cm=old["bundle"],
                    artifact_sha=artifact,
                    config_digest=config_digest,
                    runtime_profile_version=runtime_profile_version,
                    executor_wheel_filename=old.get("reconciler_wheel_key"),
                )
                self._wait_agents(target, artifact)
        self._save_state("rolled-back", previous=previous)

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
                self._deploy_reconciler(
                    target,
                    wheel_cm=self.executor_wheel_cm,
                    bundle_cm=self.bundle_cm,
                    artifact_sha=self.node_wheel_sha,
                    config_digest=self.config.agent_config_digest,
                )
                self._wait_agents(target, self.node_wheel_sha)
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
        self._roll_cpu_for_registry()
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
        self._deploy_reconciler(
            target,
            wheel_cm=self.executor_wheel_cm,
            bundle_cm=self.bundle_cm,
            artifact_sha=self.node_wheel_sha,
            config_digest=self.config.agent_config_digest,
        )
        self._wait_agents(target, self.node_wheel_sha)

    def remove_cluster(self, cluster_id: str) -> None:
        target = self._target(cluster_id)
        if not self._remote_commands_are_idle():
            raise ReleaseError("remote commands are PENDING/LEASED/WAITING")
        for deployment in (
            *inventory.DEPLOYMENTS,
            inventory.GPU_RECONCILER_DEPLOYMENT,
        ):
            self._scale_if_present(self._gpu(target), deployment, 0)
        self._update_registry(target, remove=True)
        self._roll_cpu_for_registry()

    def _target(self, cluster_id: str) -> ClusterTarget:
        for target in self.config.clusters:
            if target.cluster_id == cluster_id:
                return target
        raise ReleaseError(f"unknown cluster_id: {cluster_id}")

    def _registry(self) -> list[dict[str, Any]]:
        exists = (
            subprocess.run(
                self._cpu(
                    "-n",
                    self.config.namespace,
                    "get",
                    "secret",
                    "gpu-fault-regional-clusters",
                ),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
        if not exists:
            return []
        value = self._get_json(
            self._cpu(
                "-n",
                self.config.namespace,
                "get",
                "secret",
                "gpu-fault-regional-clusters",
            )
        )
        encoded = (value.get("data") or {}).get("clusters.json")
        if not encoded:
            return []
        return json.loads(base64.b64decode(encoded))

    def _update_registry(self, target: ClusterTarget, *, remove: bool) -> None:
        registrations = self._registry()
        remaining = [
            item
            for item in registrations
            if item.get("cluster_id") != target.cluster_id
        ]
        if remove:
            registrations = remaining
        else:
            if not all(
                (
                    target.token_file,
                    target.hyperpod_cluster_name,
                    target.eks_cluster_arn,
                )
            ):
                raise ReleaseError(
                    "join-cluster requires token_file, hyperpod_cluster_name, and eks_cluster_arn"
                )
            token = Path(target.token_file).read_text().strip()
            if len(token) < 32:
                raise ReleaseError("cluster token is too short")
            registrations = [
                *remaining,
                {
                    "cluster_id": target.cluster_id,
                    "region": target.region,
                    "hyperpod_cluster_name": (target.hyperpod_cluster_name),
                    "eks_cluster_arn": target.eks_cluster_arn,
                    "token": token,
                    "allowed_namespaces": list(target.allowed_namespaces),
                },
            ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clusters.json"
            path.write_text(
                json.dumps(registrations, indent=2),
                encoding="utf-8",
            )
            path.chmod(0o600)
            rendered = self.runner.run(
                self._cpu(
                    "-n",
                    self.config.namespace,
                    "create",
                    "secret",
                    "generic",
                    "gpu-fault-regional-clusters",
                    f"--from-file=clusters.json={path}",
                    "--dry-run=client",
                    "-o",
                    "yaml",
                ),
                capture=True,
                sensitive=True,
            )
            self.runner.run(
                self._cpu("apply", "-f", "-"),
                input_text=rendered,
                sensitive=True,
            )

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
                    "--timeout=10m",
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
            "join-cluster",
            "remove-cluster",
            "sync-state",
            "verify",
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
            release.upgrade(resume=True)
        elif arguments.mode == "rollback":
            release.rollback()
        elif arguments.mode == "join-cluster":
            if not arguments.cluster_id:
                raise ReleaseError("--cluster-id is required")
            release.join_cluster(arguments.cluster_id)
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
