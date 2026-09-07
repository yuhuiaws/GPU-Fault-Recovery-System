from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from gpu_fault_release import regional_release_online_registry as REGISTRY

ROOT = Path(__file__).resolve().parents[2]


def test_join_transition_uses_one_cpu_pod_client(monkeypatch) -> None:
    calls = []

    class Runner:
        def run(self, arguments, **kwargs):
            calls.append((arguments, kwargs))
            if "get" in arguments and "pod" in arguments:
                return "ingress-0"
            return json.dumps(
                {"generation": 4, "content_sha256": "a" * 64, "converged": True}
            )

    release = SimpleNamespace(
        runner=Runner(),
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _cpu=lambda *arguments: list(arguments),
    )
    monkeypatch.setattr(
        REGISTRY,
        "registry",
        lambda _release: [
            {
                "cluster_id": "gpu-a",
                "region": "us-east-1",
                "hyperpod_cluster_name": "hp-gpu-a",
                "eks_cluster_arn": ("arn:aws:eks:us-east-1:123456789012:cluster/gpu-a"),
                "token": "t" * 32,
                "allowed_namespaces": ["gpu-fault-system", "training"],
                "agent_endpoint_allowed_cidrs": ["10.0.0.0/16"],
            }
        ],
    )

    status = REGISTRY.transition_join_registry(
        release, "gpu-a", "PENDING", reason="join gpu-a pending"
    )

    assert status["generation"] == 4
    assert len(calls) == 2
    assert calls[0][0][-2:] == ["-o", "jsonpath={.items[0].metadata.name}"]
    transaction = json.loads(calls[1][1]["input_text"])
    assert transaction["path"].endswith("/gpu-a/transition"), (
        "registry client used the wrong cluster transition endpoint"
    )
    assert transaction["payload"]["lifecycle_state"] == "PENDING"
    assert calls[1][1]["timeout_seconds"] == 360
