"""The release engine renders, validates and ships the failure-domain map itself.

Before this the administrator ran `gpu-fault-admin failure-domain-map`, applied
the ConfigMap by hand and restarted the control-worker after every node change.
Now the control-plane apply step of `upgrade`/`bootstrap` does it from the node
inventory it already lists, and the worker rolls through the same pod-template
annotation mechanism the role ConfigMaps use -- only when the map changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.failure_domains import (
    FAILURE_DOMAIN_CONFIGMAP,
    FAILURE_DOMAIN_FILE,
    FAILURE_DOMAIN_LABELS,
    FAILURE_DOMAIN_MAP_ANNOTATION,
    failure_domain_map_sha256,
)
from gpu_fault_release import regional_release_failure_domains as FD
from gpu_fault_release import rollout as MODULE
from tests.regional._release_orchestrator_support import (
    GPU_EKS_ARN,
    REGION,
    config_file,
)

ROOT = Path(__file__).resolve().parents[2]
APPLY_SCRIPT = ROOT / "deploy/control-plane/tools/apply-control-plane-role-split.sh"
GROUP = "sagemaker.amazonaws.com/instance-group-name"
RACK = "topology.k8s.aws/network-node-layer-3"


def _cluster(cluster_id: str) -> dict[str, Any]:
    return {
        "cluster_id": cluster_id,
        "context": f"{cluster_id}-context",
        "executor_irsa_role_arn": f"arn:aws:iam::1:role/{cluster_id}",
        "region": REGION,
        "hyperpod_cluster_name": f"hp-{cluster_id}",
        "eks_cluster_arn": GPU_EKS_ARN.replace(
            "cluster/gpu-a", f"cluster/{cluster_id}"
        ),
        "agent_endpoint_allowed_cidrs": ["10.0.0.0/16"],
    }


def _node(name: str, **labels: str) -> dict[str, Any]:
    return {"metadata": {"name": name, "labels": labels}}


def _nodes() -> dict[str, list[dict[str, Any]]]:
    return {
        "gpu-a-context": [
            _node("a-1", **{GROUP: "group-a", RACK: "rack-1"}),
            _node("a-2", **{GROUP: "group-a", RACK: "rack-2"}),
            _node("a-3"),
        ],
        "gpu-b-context": [_node("b-1", **{GROUP: "group-b"})],
    }


class FakeRunner:
    """Answers `get nodes` per cluster context and records everything else."""

    dry_run = False

    def __init__(self, nodes: dict[str, list[dict[str, Any]]]) -> None:
        self.nodes = nodes
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(self, args: list[str], **kwargs: Any) -> str:
        self.calls.append((list(args), dict(kwargs)))
        if "get" in args and "nodes" in args:
            context = args[args.index("--context") + 1]
            return json.dumps({"items": self.nodes[context]})
        return ""

    def probe(self, *_args: Any, **_kwargs: Any) -> bool:
        return False

    def node_reads(self) -> list[str]:
        return [
            args[args.index("--context") + 1]
            for args, _ in self.calls
            if "get" in args and "nodes" in args
        ]

    def config_map_applies(self) -> list[dict[str, Any]]:
        return [
            json.loads(kwargs["input_text"])
            for args, kwargs in self.calls
            if args[-3:] == ["apply", "-f", "-"] and "input_text" in kwargs
        ]


def _release(
    tmp_path: Path,
    nodes: dict[str, list[dict[str, Any]]],
    *,
    label_keys: list[str] | None = None,
) -> tuple[MODULE.RegionalRelease, FakeRunner]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = config_file(tmp_path, clusters=[_cluster("gpu-a"), _cluster("gpu-b")])
    if label_keys is not None:
        document = json.loads(path.read_text())
        document["failure_domain_labels"] = label_keys
        path.write_text(json.dumps(document))
    runner = FakeRunner(nodes)
    return MODULE.RegionalRelease(MODULE.ReleaseConfig.load(path), runner), runner


def test_apply_lists_every_cluster_once_and_ships_only_labelled_nodes(
    tmp_path: Path,
) -> None:
    release, runner = _release(tmp_path, _nodes())

    digest = FD.apply_failure_domain_map(release)

    assert runner.node_reads() == ["gpu-a-context", "gpu-b-context"]
    (manifest,) = runner.config_map_applies()
    assert manifest["kind"] == "ConfigMap"
    assert manifest["metadata"] == {
        "name": FAILURE_DOMAIN_CONFIGMAP,
        "namespace": release.config.namespace,
    }
    assert json.loads(manifest["data"][FAILURE_DOMAIN_FILE]) == {
        "gpu-a": {"a-1": "group-a", "a-2": "group-a"},
        "gpu-b": {"b-1": "group-b"},
    }, "the unlabelled node a-3 consumes no domain slot"
    assert digest == failure_domain_map_sha256(manifest)
    assert runner.calls[-1][0][:3] == [
        "kubectl",
        "--kubeconfig",
        release.config.cpu_kubeconfig,
    ]


def test_digest_changes_only_when_the_map_content_changes(tmp_path: Path) -> None:
    same_a, _ = _release(tmp_path / "one", _nodes())
    same_b, _ = _release(tmp_path / "two", _nodes())
    moved = _nodes()
    moved["gpu-b-context"] = [_node("b-1", **{GROUP: "group-c"})]
    changed, _ = _release(tmp_path / "three", moved)

    first = FD.apply_failure_domain_map(same_a)

    assert first == FD.apply_failure_domain_map(same_b), "same nodes, same stamp"
    assert first != FD.apply_failure_domain_map(changed), "a moved node re-rolls"
    assert len(first) == 64


def test_site_configured_label_keys_replace_the_default_priority(
    tmp_path: Path,
) -> None:
    release, runner = _release(tmp_path, _nodes(), label_keys=[RACK])

    FD.apply_failure_domain_map(release)

    assert release.config.failure_domain_labels == (RACK,)
    (manifest,) = runner.config_map_applies()
    assert json.loads(manifest["data"][FAILURE_DOMAIN_FILE]) == {
        "gpu-a": {"a-1": "rack-1", "a-2": "rack-2"},
        "gpu-b": {},
    }, "gpu-b has no rack label, so its nodes fall out of the domain tier"


def test_release_config_defaults_and_bounds_the_label_keys(tmp_path: Path) -> None:
    release, _ = _release(tmp_path / "default", _nodes())
    assert release.config.failure_domain_labels == FAILURE_DOMAIN_LABELS

    with pytest.raises(MODULE.ReleaseError, match="failure_domain_labels"):
        _release(tmp_path / "empty", _nodes(), label_keys=[])
    with pytest.raises(MODULE.ReleaseError, match="unique"):
        _release(tmp_path / "dup", _nodes(), label_keys=[RACK, RACK])


def test_a_malformed_map_is_refused_before_anything_is_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, runner = _release(tmp_path, _nodes())
    monkeypatch.setattr(
        FD, "failure_domain_map", lambda cluster_id, *_a, **_k: {cluster_id: {"n": ""}}
    )

    with pytest.raises(MODULE.ReleaseError, match="non-empty failure domain"):
        FD.apply_failure_domain_map(release)

    assert runner.config_map_applies() == [], "nothing reached the cluster"


def test_dry_run_neither_lists_nodes_nor_applies(tmp_path: Path) -> None:
    release, runner = _release(tmp_path, _nodes())
    runner.dry_run = True

    assert FD.apply_failure_domain_map(release) == ""
    assert runner.calls == []


def test_cpu_apply_step_ships_the_map_and_hands_its_digest_to_the_role_split(
    tmp_path: Path,
) -> None:
    release, runner = _release(tmp_path, _nodes())
    environment: dict[str, str] = {"GPU_FAULT_NAMESPACE": release.config.namespace}

    MODULE.render_and_apply_cpu_roles(release, environment)

    scripts = [
        Path(args[1]).name
        for args, _ in runner.calls
        if args[0] == "bash" and args[1].endswith(".sh")
    ]
    assert scripts == [
        "render-control-plane-role-split.sh",
        "apply-control-plane-role-split.sh",
    ]
    (manifest,) = runner.config_map_applies()
    apply_index = next(
        index
        for index, (args, _) in enumerate(runner.calls)
        if args[0] == "bash" and args[1].endswith("apply-control-plane-role-split.sh")
    )
    config_map_index = next(
        index
        for index, (args, kwargs) in enumerate(runner.calls)
        if args[-3:] == ["apply", "-f", "-"] and "input_text" in kwargs
    )
    assert config_map_index < apply_index, "the map exists before the worker rolls"
    digest = failure_domain_map_sha256(manifest)
    assert environment[FD.FAILURE_DOMAIN_MAP_SHA256_ENV] == digest
    apply_env = runner.calls[apply_index][1]["env"]
    assert apply_env[FD.FAILURE_DOMAIN_MAP_SHA256_ENV] == digest


def test_role_split_stamps_the_digest_on_the_worker_pod_template_only() -> None:
    """The stamp is a pod-template annotation, so Kubernetes rolls the worker
    exactly when the digest changes; no `rollout restart` is involved."""

    text = APPLY_SCRIPT.read_text(encoding="utf-8")

    assert f'"${{{FD.FAILURE_DOMAIN_MAP_SHA256_ENV}:-}}"' in text
    assert f'"{FAILURE_DOMAIN_MAP_ANNOTATION}": $sha' in text
    worker = text.index("apply_worker_role() {")
    ingress = text.index("apply_ingress_role() {")
    spool = text.index("apply_spool_role() {")
    assert worker < text.index("stamp_failure_domain_map\n", worker) < ingress
    assert "stamp_failure_domain_map\n" not in text[spool:worker]
    assert "stamp_failure_domain_map\n" not in text[ingress:]
    stamp = text[text.index("stamp_failure_domain_map() {") :]
    assert "rollout restart" not in stamp[: stamp.index("\n}\n")]
