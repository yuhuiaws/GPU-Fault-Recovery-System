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
import yaml
from regional_release_config import (
    ClusterTarget,
    ReleaseConfig,
    ReleaseError,
    render_nlb_manifest,
)
from regional_release_preflight import ensure_region_contexts
from regional_release_reporting import build_release_plan, build_release_status
from regional_release_rendering import (
    DEFAULT_RUNTIME_IMAGE,
    build_cpu_apply_environment,
    build_reconciler_environment,
    render_gpu_rollout_manifests,
)
from regional_runtime_profile import ensure_runtime_profile

ROOT = Path(__file__).resolve().parents[3]
STATE_CONFIG_MAP = "gpu-fault-regional-release-state"
EXECUTOR_SAGEMAKER_ACTIONS = frozenset(
    {
        "sagemaker:describecluster",
        "sagemaker:listclusternodes",
        "sagemaker:describeclusternode",
        "sagemaker:batchrebootclusternodes",
    }
)


def validate_executor_iam_documents(
    role_arn: str,
    documents: list[dict[str, Any]],
) -> None:
    forbidden = []
    for document in documents:
        for statement in document.get("Statement", []):
            if statement.get("Effect") != "Allow":
                continue
            if statement.get("NotAction") is not None:
                forbidden.append("Allow/NotAction")
                continue
            raw = statement.get("Action", [])
            actions = raw if isinstance(raw, list) else [raw]
            for action in actions:
                normalized = str(action).lower()
                if normalized.startswith("ses:"):
                    forbidden.append(str(action))
                elif (
                    normalized.startswith("sagemaker:")
                    and normalized not in EXECUTOR_SAGEMAKER_ACTIONS
                ):
                    forbidden.append(str(action))
    if forbidden:
        raise ReleaseError(
            f"executor role {role_arn} exceeds the regional "
            "data-plane boundary: " + ", ".join(sorted(set(forbidden)))
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


class RegionalRelease:
    _ensure_contexts = ensure_region_contexts

    def __init__(self, config: ReleaseConfig, runner: Runner) -> None:
        self.config = config
        self.runner = runner
        self.wheel_sha = self._sha256(config.wheel)
        self.bundle_sha = self._sha256(config.bundle)
        self.runtime_profile_sha = self._sha256(config.runtime_profile_source)
        self.wheel_cm = "gpu-fault-control-plane-wheel-0100-" + self.wheel_sha[:12]
        self.bundle_cm = "gpu-fault-node-installer-0100-" + self.bundle_sha[:12]
        self.runtime_image = os.getenv("GPU_FAULT_RUNTIME_IMAGE", DEFAULT_RUNTIME_IMAGE)
        if not self.runtime_image or any(
            character.isspace() or character == "#" for character in self.runtime_image
        ):
            raise ReleaseError(
                "GPU_FAULT_RUNTIME_IMAGE must be a non-empty "
                "OCI image reference without whitespace or #"
            )
        self.state: dict[str, Any] = {}

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

    def _get_json(self, args: list[str]) -> dict[str, Any]:
        raw = self.runner.run(args + ["-o", "json"], capture=True)
        return json.loads(raw) if raw else {}

    def _config_map_data(self, name: str) -> dict[str, str]:
        value = self._get_json(
            self._cpu(
                "-n",
                self.config.namespace,
                "get",
                "configmap",
                name,
            )
        )
        return dict(value.get("data") or {})

    def _deployment_wheel(self, args: list[str], deployment: str) -> str | None:
        value = self._get_json(
            args
            + [
                "-n",
                self.config.namespace,
                "get",
                "deployment",
                deployment,
            ]
        )
        for volume in (
            value.get("spec", {}).get("template", {}).get("spec", {}).get("volumes", [])
        ):
            if volume.get("name") == "artifact":
                return (volume.get("configMap") or {}).get("name")
        return None

    def _capture_previous(self) -> dict[str, Any]:
        metadata = self._config_map_data("gpu-fault-release-metadata")
        clusters = {}
        for target in self.config.clusters:
            template = self._deployment_template_name(target)
            clusters[target.cluster_id] = {
                "wheel": self._deployment_wheel(
                    self._gpu(target),
                    inventory.GPU_EXECUTOR_DEPLOYMENT,
                ),
                "reconciler_wheel": self._deployment_wheel(
                    self._gpu(target),
                    inventory.GPU_RECONCILER_DEPLOYMENT,
                ),
                "template": template,
                "bundle": (
                    self._template_bundle(target, template) if template else None
                ),
            }
        return {
            "metadata": metadata,
            "runtime_profile_version": self._config_map_data(
                "gpu-fault-api-ha-config-core"
            ).get("GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION"),
            "cpu_wheel": self._deployment_wheel(
                self._cpu(),
                inventory.CPU_INGRESS_DEPLOYMENT,
            ),
            "clusters": clusters,
        }

    def _deployment_template_name(self, target: ClusterTarget) -> str | None:
        value = self._get_json(
            self._gpu(
                target,
                "-n",
                self.config.namespace,
                "get",
                "deployment",
                inventory.GPU_RECONCILER_DEPLOYMENT,
            )
        )
        for volume in (
            value.get("spec", {}).get("template", {}).get("spec", {}).get("volumes", [])
        ):
            if volume.get("name") == "installer-template":
                return (volume.get("configMap") or {}).get("name")
        return None

    def _template_bundle(self, target: ClusterTarget, template_name: str) -> str | None:
        value = self._get_json(
            self._gpu(
                target,
                "-n",
                self.config.namespace,
                "get",
                "configmap",
                template_name,
            )
        )
        text = (value.get("data") or {}).get("job.yaml")
        if not text:
            return None
        for document in yaml.safe_load_all(text):
            for volume in (
                (document or {})
                .get("spec", {})
                .get("template", {})
                .get("spec", {})
                .get("volumes", [])
            ):
                if volume.get("name") == "installer":
                    return (volume.get("configMap") or {}).get("name")
        return None

    def _save_state(self, phase: str, **updates: Any) -> None:
        self.state.update(
            {
                "phase": phase,
                "release_id": self.wheel_sha[:12],
                "wheel_sha256": self.wheel_sha,
                "bundle_sha256": self.bundle_sha,
                "wheel_config_map": self.wheel_cm,
                "bundle_config_map": self.bundle_cm,
                "runtime_profile_version": self.config.runtime_profile_version,
                "runtime_profile_sha256": self.runtime_profile_sha,
                "runtime_profile_registration_cluster_id": (
                    self.config.runtime_profile_registration_cluster_id
                ),
                "updated_at_epoch": int(time.time()),
                **updates,
            }
        )
        if self.runner.dry_run:
            return
        with tempfile.TemporaryDirectory() as directory:
            state_file = Path(directory) / "state.json"
            state_file.write_text(
                json.dumps(self.state, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            rendered = self.runner.run(
                self._cpu(
                    "-n",
                    self.config.namespace,
                    "create",
                    "configmap",
                    STATE_CONFIG_MAP,
                    f"--from-file=state.json={state_file}",
                    "--dry-run=client",
                    "-o",
                    "yaml",
                ),
                capture=True,
            )
            self.runner.run(
                self._cpu("apply", "-f", "-"),
                input_text=rendered,
            )

    def _load_state(self) -> dict[str, Any]:
        value = self._get_json(
            self._cpu(
                "-n",
                self.config.namespace,
                "get",
                "configmap",
                STATE_CONFIG_MAP,
            )
        )
        raw = (value.get("data") or {}).get("state.json")
        if not raw:
            raise ReleaseError("regional release state is missing")
        self.state = json.loads(raw)
        return self.state

    def _validate_executor_iam_role(self, target: ClusterTarget) -> None:
        role_name = target.executor_irsa_role_arn.rsplit("/", 1)[-1]
        inline = json.loads(
            self.runner.run(
                [
                    "aws",
                    "iam",
                    "list-role-policies",
                    "--role-name",
                    role_name,
                    "--output",
                    "json",
                ],
                capture=True,
            )
        )
        documents = []
        for name in inline.get("PolicyNames", []):
            value = json.loads(
                self.runner.run(
                    [
                        "aws",
                        "iam",
                        "get-role-policy",
                        "--role-name",
                        role_name,
                        "--policy-name",
                        name,
                        "--output",
                        "json",
                    ],
                    capture=True,
                )
            )
            documents.append(value["PolicyDocument"])
        attached = json.loads(
            self.runner.run(
                [
                    "aws",
                    "iam",
                    "list-attached-role-policies",
                    "--role-name",
                    role_name,
                    "--output",
                    "json",
                ],
                capture=True,
            )
        )
        for policy in attached.get("AttachedPolicies", []):
            metadata = json.loads(
                self.runner.run(
                    [
                        "aws",
                        "iam",
                        "get-policy",
                        "--policy-arn",
                        policy["PolicyArn"],
                        "--output",
                        "json",
                    ],
                    capture=True,
                )
            )
            version = metadata["Policy"]["DefaultVersionId"]
            value = json.loads(
                self.runner.run(
                    [
                        "aws",
                        "iam",
                        "get-policy-version",
                        "--policy-arn",
                        policy["PolicyArn"],
                        "--version-id",
                        version,
                        "--output",
                        "json",
                    ],
                    capture=True,
                )
            )
            documents.append(value["PolicyVersion"]["Document"])
        validate_executor_iam_documents(
            target.executor_irsa_role_arn,
            documents,
        )

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
        return build_release_plan(mode)

    def status(self) -> dict[str, Any]:
        return build_release_status(self)

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

    def _upload_release(self) -> None:
        wheel_key = self.config.wheel.name
        bundle_key = self.config.bundle.name
        self._upload_config_map(
            self._cpu(),
            self.wheel_cm,
            wheel_key,
            self.config.wheel,
            self.wheel_sha,
        )
        for target in self.config.clusters:
            kubectl = self._gpu(target)
            self._upload_config_map(
                kubectl,
                self.wheel_cm,
                wheel_key,
                self.config.wheel,
                self.wheel_sha,
            )
            self._upload_config_map(
                kubectl,
                self.bundle_cm,
                bundle_key,
                self.config.bundle,
                self.bundle_sha,
            )

    def _apply_cpu(self, *, finalize: bool) -> None:
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
    ) -> None:
        for deployment, text in render_gpu_rollout_manifests(
            self,
            target,
            wheel_cm,
            runtime_profile_version=runtime_profile_version,
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
                    "gpu-fault.io/artifact-sha256": self.wheel_sha,
                    "gpu-fault.io/control-plane-wheel-sha256": (self.wheel_sha),
                    "gpu-fault.io/release-rollout": (self.wheel_sha[:12]),
                    "gpu-fault.io/release-sha256": self.wheel_sha,
                    "gpu-fault.io/release-wheel-sha256": (self.wheel_sha),
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
    ) -> None:
        environment = build_reconciler_environment(
            self,
            target,
            wheel_cm=wheel_cm,
            bundle_cm=bundle_cm,
            artifact_sha=artifact_sha,
            config_digest=config_digest,
            runtime_profile_version=runtime_profile_version,
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
                    "GPU_FAULT_EXPECTED_WHEEL_CONFIGMAP": (self.wheel_cm),
                },
            )

    def upgrade(self, *, resume: bool = False) -> None:
        self._ensure_contexts()
        self._require_cpu_secrets()
        if not self._remote_commands_are_idle():
            raise ReleaseError("remote commands are PENDING/LEASED/WAITING")
        previous = (
            self._load_state().get("previous") if resume else self._capture_previous()
        )
        if not previous:
            raise ReleaseError("previous release state is unavailable")
        self._save_state("preflight", previous=previous)
        try:
            self._upload_release()
            self._save_state("uploaded")
            self._ensure_schema()
            self._save_state("schema-ready")
            self._apply_cpu(finalize=False)
            ensure_runtime_profile(self)
            self._save_state("cpu-staged")
            for target in self.config.clusters:
                self._apply_gpu_deployments(target, self.wheel_cm)
                self._deploy_reconciler(
                    target,
                    wheel_cm=self.wheel_cm,
                    bundle_cm=self.bundle_cm,
                    artifact_sha=self.wheel_sha,
                    config_digest=self.config.agent_config_digest,
                )
                self._wait_agents(target, self.wheel_sha)
            self._save_state("data-converged")
            self._apply_cpu(finalize=True)
            self._validate_release()
            self._save_state("complete")
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
        environment = {
            **os.environ,
            "KUBECONFIG": rollback_config.cpu_kubeconfig,
            "GPU_FAULT_AWS_REGION": rollback_config.aws_region,
            "GPU_FAULT_NAMESPACE": rollback_config.namespace,
            "GPU_FAULT_WHEEL_CONFIGMAP": cpu_wheel,
            "GPU_FAULT_WHEEL_SHA256": cpu_sha,
            "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": artifact,
            "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": config_digest,
            "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": (runtime_profile_version),
            "GPU_FAULT_REQUIRED_AGENT_PROTOCOL_VERSION": metadata.get(
                "required-agent-protocol-version", "3"
            ),
            "GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION": (
                metadata.get(
                    "required-regional-executor-protocol-version",
                    "2",
                )
            ),
            "GPU_FAULT_FINALIZE_AGENT_PIN": "true",
            "GPU_FAULT_FINALIZE_DATA_PLANE_PIN": "true",
        }
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
                self._apply_gpu_deployments(
                    target,
                    wheel,
                    runtime_profile_version=runtime_profile_version,
                )
            if all(old.get(name) for name in ("reconciler_wheel", "bundle")):
                self._deploy_reconciler(
                    target,
                    wheel_cm=old["reconciler_wheel"],
                    bundle_cm=old["bundle"],
                    artifact_sha=artifact,
                    config_digest=config_digest,
                    runtime_profile_version=runtime_profile_version,
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

    def _ensure_schema(self) -> None:
        self.runner.run(
            [str(ROOT / "deploy/control-plane/tools/ensure-postgres-schema.sh")],
            env={
                **os.environ,
                "GPU_FAULT_CONTROL_PLANE_KUBECONFIG": (self.config.cpu_kubeconfig),
                "GPU_FAULT_NAMESPACE": self.config.namespace,
                "GPU_FAULT_WHEEL_CONFIGMAP": self.wheel_cm,
            },
        )

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
        self._save_state("bootstrap-started", previous=None)
        try:
            self._require_cpu_secrets(include_registry=False)
            for target in self.config.clusters:
                self._update_registry(target, remove=False)
            self._require_cpu_secrets()
            self._upload_release()
            self._ensure_schema()
            self._apply_cpu(finalize=True)
            ensure_runtime_profile(self)
            self._apply_nlb()
            for target in self.config.clusters:
                self._ensure_gpu_namespace(target)
                self._ensure_connection_secret(target)
                self._apply_gpu_deployments(target, self.wheel_cm)
                self._deploy_reconciler(
                    target,
                    wheel_cm=self.wheel_cm,
                    bundle_cm=self.bundle_cm,
                    artifact_sha=self.wheel_sha,
                    config_digest=self.config.agent_config_digest,
                )
                self._wait_agents(target, self.wheel_sha)
            self._validate_release()
            self._save_state("complete", previous=None)
        except Exception:
            self._save_state("bootstrap-failed", previous=None)
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
        self._save_state("bootstrap-cleaned", previous=None)

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

    def _apply_nlb(self) -> None:
        if not self.config.nlb:
            return
        text = (
            ROOT / "deploy/control-plane/regional/regional-control-plane-nlb.yaml"
        ).read_text(encoding="utf-8")
        text = render_nlb_manifest(self.config, text)
        self.runner.run(self._cpu("apply", "-f", "-"), input_text=text)

    def _ensure_gpu_namespace(self, target: ClusterTarget) -> None:
        rendered = self.runner.run(
            self._gpu(
                target,
                "create",
                "namespace",
                self.config.namespace,
                "--dry-run=client",
                "-o",
                "yaml",
            ),
            capture=True,
        )
        self.runner.run(
            self._gpu(target, "apply", "-f", "-"),
            input_text=rendered,
        )

    def _ensure_connection_secret(self, target: ClusterTarget) -> None:
        if not all(
            (
                target.token_file,
                target.ca_file,
                target.control_plane_url,
                target.hyperpod_cluster_name,
            )
        ):
            self.runner.run(
                self._gpu(
                    target,
                    "-n",
                    self.config.namespace,
                    "get",
                    "secret",
                    "gpu-fault-regional-connection",
                ),
                capture=True,
            )
            return
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {
                "cluster-token": (
                    Path(target.token_file).read_text(encoding="utf-8").strip().encode()
                ),
                "ca.crt": Path(target.ca_file).read_bytes(),
                "control-plane-url": target.control_plane_url.encode(),
                "cluster-id": target.cluster_id.encode(),
                "allowed-namespaces": ",".join(target.allowed_namespaces).encode(),
                "hyperpod-cluster-name": (target.hyperpod_cluster_name or "").encode(),
                "hyperpod-confirm-cluster-name": (
                    target.hyperpod_cluster_name or ""
                ).encode(),
            }
            arguments = self._gpu(
                target,
                "-n",
                self.config.namespace,
                "create",
                "secret",
                "generic",
                "gpu-fault-regional-connection",
            )
            for name, content in files.items():
                path = root / name
                path.write_bytes(content)
                path.chmod(0o600)
                arguments.append(f"--from-file={name}={path}")
            arguments.extend(["--dry-run=client", "-o", "yaml"])
            rendered = self.runner.run(arguments, capture=True, sensitive=True)
            self.runner.run(
                self._gpu(target, "apply", "-f", "-"),
                input_text=rendered,
                sensitive=True,
            )

    def join_cluster(self, cluster_id: str) -> None:
        target = self._target(cluster_id)
        self._ensure_contexts()
        self._update_registry(target, remove=False)
        self._roll_cpu_for_registry()
        ensure_runtime_profile(self)
        self._ensure_gpu_namespace(target)
        self._ensure_connection_secret(target)
        self._upload_config_map(
            self._gpu(target),
            self.wheel_cm,
            self.config.wheel.name,
            self.config.wheel,
            self.wheel_sha,
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
        if required != self.wheel_sha:
            raise ReleaseError(
                "join-cluster must use the current required Agent artifact"
            )
        self._apply_gpu_deployments(target, self.wheel_cm)
        self._deploy_reconciler(
            target,
            wheel_cm=self.wheel_cm,
            bundle_cm=self.bundle_cm,
            artifact_sha=self.wheel_sha,
            config_digest=self.config.agent_config_digest,
        )
        self._wait_agents(target, self.wheel_sha)

    def remove_cluster(self, cluster_id: str) -> None:
        target = self._target(cluster_id)
        if not self._remote_commands_are_idle():
            raise ReleaseError("remote commands are PENDING/LEASED/WAITING")
        for deployment in (
            *inventory.DEPLOYMENTS,
            inventory.GPU_RECONCILER_DEPLOYMENT,
        ):
            self.runner.run(
                self._gpu(
                    target,
                    "-n",
                    self.config.namespace,
                    "scale",
                    f"deployment/{deployment}",
                    "--replicas=0",
                )
            )
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
            if len(remaining) == 0:
                raise ReleaseError("refusing to remove the last regional cluster")
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
            "status",
            "bootstrap",
            "deploy",
            "upgrade",
            "resume",
            "rollback",
            "join-cluster",
            "remove-cluster",
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
        if arguments.mode == "plan":
            print(
                json.dumps(
                    release.plan(arguments.plan_mode),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        elif arguments.mode == "status":
            print(json.dumps(release.status(), indent=2))
        elif arguments.mode in {"bootstrap", "deploy"}:
            release.bootstrap()
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
        return 0
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
