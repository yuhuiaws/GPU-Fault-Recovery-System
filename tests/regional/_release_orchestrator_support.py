from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from gpu_fault.capabilities import compile_runtime_profile
from gpu_fault.models import RuntimeProfile
from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
RELEASE_MODULE_PATH = ROOT / "deploy/control-plane/regional/rollout_regional_release.py"
RELEASE_MODULE = lazy_script_module(RELEASE_MODULE_PATH)
RUNTIME_PROFILE_MODULE = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_runtime_profile.py"
)
DNS_MODULE = lazy_script_module(ROOT / "deploy/control-plane/regional/regional_dns.py")
REGION = "us-east-1"
CPU_EKS_ARN = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-fault-control-plane"
GPU_EKS_ARN = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a"


def config_file(
    tmp_path: Path, *, clusters=None, profile_version: str = "hyperpod-v1"
) -> Path:
    wheel = tmp_path / "release.whl"
    bundle = tmp_path / "bundle.tar.gz"
    wheel.write_bytes(b"wheel")
    bundle.write_bytes(b"bundle")
    value = {
        "aws_region": REGION,
        "cpu_kubeconfig": "/secure/cpu.kubeconfig",
        "cpu_eks_arn": CPU_EKS_ARN,
        "cpu_hyperpod_cluster_name": "gpu-fault-control-plane",
        "namespace": "gpu-fault-system",
        "runtime_profile": {
            "source": str(
                ROOT / "config/runtime-profile.regional-hyperpod-safe.example.yaml"
            ),
            "version": profile_version,
            "registration_cluster_id": "gpu-a",
        },
        "release": {
            "wheel": str(wheel),
            "bundle": str(bundle),
            "agent_config_digest": "a" * 64,
        },
        "clusters": clusters
        or [
            {
                "cluster_id": "gpu-a",
                "context": "gpu-a-context",
                "executor_irsa_role_arn": "arn:aws:iam::1:role/a",
                "region": REGION,
                "hyperpod_cluster_name": "hp-gpu-a",
                "eks_cluster_arn": GPU_EKS_ARN,
                "agent_endpoint_allowed_cidrs": ["10.0.0.0/16"],
            }
        ],
    }
    path = tmp_path / "release.json"
    path.write_text(json.dumps(value))
    return path


def manifest_config_file(tmp_path: Path) -> Path:
    wheel = tmp_path / "release.whl"
    bundle = tmp_path / "bundle.tar.gz"
    wheel.write_bytes(b"wheel")
    bundle.write_bytes(b"bundle")
    manifest = tmp_path / "current-release.json"
    manifest.write_text(
        json.dumps(
            {
                "wheel": str(wheel),
                "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                "bundle": str(bundle),
                "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
            }
        )
    )
    path = tmp_path / "release-config.json"
    path.write_text(
        json.dumps(
            {
                "aws_region": REGION,
                "cpu_kubeconfig": "/secure/cpu.kubeconfig",
                "cpu_eks_arn": CPU_EKS_ARN,
                "cpu_hyperpod_cluster_name": "gpu-fault-control-plane",
                "runtime_profile": {
                    "source": str(
                        ROOT
                        / "config/runtime-profile.regional-hyperpod-safe.example.yaml"
                    ),
                    "version": "hyperpod-v1",
                    "registration_cluster_id": "gpu-a",
                },
                "release": {"manifest": str(manifest), "agent_config_digest": "a" * 64},
                "clusters": [
                    {
                        "cluster_id": "gpu-a",
                        "context": "gpu-a-context",
                        "executor_irsa_role_arn": "arn:aws:iam::1:role/a",
                        "region": REGION,
                        "hyperpod_cluster_name": "hp-gpu-a",
                        "eks_cluster_arn": GPU_EKS_ARN,
                    }
                ],
            }
        )
    )
    return path


class RuntimeProfileRunner:
    dry_run = False

    def __init__(self, existing=None) -> None:
        self.existing = existing
        self.posted = []

    @staticmethod
    def _desired(input_text: str) -> dict:
        profile = RuntimeProfile.model_validate(json.loads(input_text))
        return compile_runtime_profile(profile).model_dump(mode="json")

    def run(self, args, **kwargs):
        command = " ".join(args)
        if "get pod" in command:
            return "api-pod"
        if "compile_runtime_profile" in command:
            desired = self._desired(kwargs["input_text"])
            return json.dumps({"desired": desired, "existing": self.existing})
        if "/v1/runtime-profiles" in command:
            desired = self._desired(kwargs["input_text"])
            self.posted.append(json.loads(kwargs["input_text"]))
            return json.dumps(desired)
        raise AssertionError(f"unexpected Runtime Profile command: {args}")


def phase_release(
    calls: list,
    saves: list | None = None,
    *,
    clusters: tuple = (),
    max_parallel_clusters: int = 1,
    **overrides: Any,
) -> SimpleNamespace:
    """A release whose only behaviour is recording what the phases did.

    Every hook `run_upgrade_phases` may reach is present, so one stand-in serves
    plans that include the schema, the registry, the CPU stage and the finalize
    barriers alike; the plan passed to `run_upgrade_phases` decides which of them
    actually run. Pass `**overrides` to replace individual hooks -- that is how a
    test makes one phase block, fail, or record its own ordering.
    """

    release: SimpleNamespace

    def save_state(phase, **updates):
        release.state.update({"phase": phase, **updates})
        if saves is not None:
            saves.append(json.loads(json.dumps(release.state, default=str)))

    def stage_registry():
        calls.append("registry")
        return False

    fields: dict[str, Any] = {
        "state": {"phase": "preflight"},
        "config": SimpleNamespace(
            clusters=clusters,
            upgrade_max_parallel_clusters=max_parallel_clusters,
            agent_config_digest="config-a",
        ),
        "executor_wheel_cm": "wheel",
        "bundle_cm": "bundle",
        "node_wheel_sha": "a" * 64,
        "_upload_release": lambda _diff: calls.append("upload"),
        "_ensure_schema": lambda: calls.append("schema"),
        "_stage_registry": stage_registry,
        "_apply_cpu": lambda **kwargs: calls.append(
            "cpu-finalize" if kwargs.get("finalize") else "cpu-stage"
        ),
        "_apply_nlb": lambda: calls.append("endpoint"),
        "_apply_observability": lambda: calls.append("observability"),
        "_ensure_profile_transition_safe": lambda _version: None,
        "_capture_active_agent_node_sets": lambda: {"gpu-a": {"node_ids": ["node-a"]}},
        "_wait_candidate_cpu_agent_heartbeats": (
            lambda _expected, **kwargs: calls.append(
                "pin-barrier" if kwargs.get("required_identity") else "barrier"
            )
        ),
        "_candidate_agent_pin_identity": lambda: {"artifact_sha256": "a" * 64},
        "_validate_release_quick": lambda _plan: calls.append("verify"),
        "_commit_registry_update": lambda: calls.append("commit-registry"),
        "_save_state": save_state,
    }
    fields.update(overrides)
    release = SimpleNamespace(**fields)
    return release
