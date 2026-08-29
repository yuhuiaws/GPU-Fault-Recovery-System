from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gpu_fault.capabilities import compile_runtime_profile
from gpu_fault.models import RuntimeProfile

ROOT = Path(__file__).resolve().parents[2]
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
