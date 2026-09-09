"""The admin side of the product-managed failure-domain map.

`deploy` renders the map inside the release engine; `join-cluster` and
`remove-cluster` re-render it through `apply_failure_domain_map` once the
managed cluster set is final. The label keys come from `site.yaml`
(`spec.failureDomainLabels`), never from a CLI flag a later render would
forget. `gpu-fault-admin failure-domain-map` is a read-only debugging aid.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import cluster_join as admin_cluster_join
from gpu_fault.admin import cluster_join_commit as admin_cluster_join_commit
from gpu_fault.admin import cluster_removal as admin_cluster_removal
from gpu_fault.admin import failure_domain_map as module
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import SiteConfigError, SiteSpec
from gpu_fault.execution.remediation_budget import (
    FAILURE_DOMAIN_MAP_ENV,
    load_failure_domain_map,
)
from gpu_fault.failure_domains import (
    FAILURE_DOMAIN_LABELS,
    FAILURE_DOMAIN_MAP_ANNOTATION,
    failure_domain_map_sha256,
)

GROUP = "sagemaker.amazonaws.com/instance-group-name"
RACK = "topology.k8s.aws/network-node-layer-3"


def _site(**release_config: Any) -> SimpleNamespace:
    return SimpleNamespace(
        release_config={
            "cpu_kubeconfig": "/secure/cpu.kubeconfig",
            "namespace": "gpu-fault-system",
            "clusters": [
                {
                    "cluster_id": "gpu-a",
                    "context": "gpu-a-context",
                    "hyperpod_cluster_name": "hp-gpu-a",
                },
                {
                    "cluster_id": "gpu-b",
                    "context": "gpu-b-context",
                    "hyperpod_cluster_name": "hp-gpu-b",
                },
            ],
            **release_config,
        },
        environment={},
    )


def _inventory(cluster_id: str) -> dict[str, dict]:
    group = f"{cluster_id}-group"
    return {
        f"{cluster_id}-node-1": {
            "metadata": {
                "name": f"{cluster_id}-node-1",
                "labels": {GROUP: group, RACK: f"{cluster_id}-rack"},
            }
        },
        f"{cluster_id}-node-2": {
            "metadata": {"name": f"{cluster_id}-node-2", "labels": {}}
        },
    }


class FakeKubectl:
    """Records kubectl invocations; the worker Deployment exists unless told otherwise."""

    def __init__(self, *, worker_exists: bool = True) -> None:
        self.worker_exists = worker_exists
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        self.calls.append((list(args), kwargs))
        if "get" in args and "deployment" in args:
            code = 0 if self.worker_exists else 1
            return subprocess.CompletedProcess(args, code, "", "not found")
        if "get" in args and "nodes" in args:
            context = args[args.index("--context") + 1]
            items = list(_inventory(context.removesuffix("-context")).values())
            return subprocess.CompletedProcess(
                args, 0, json.dumps({"items": items}), ""
            )
        return subprocess.CompletedProcess(args, 0, "", "")

    def verbs(self) -> list[str]:
        verbs = []
        for args, _ in self.calls:
            if "apply" in args:
                verbs.append("apply")
            elif "patch" in args:
                verbs.append("patch")
            elif "nodes" in args:
                verbs.append("get nodes")
            elif "deployment" in args:
                verbs.append("get deployment")
        return verbs


def test_map_covers_every_managed_cluster_and_reports_unmapped_nodes() -> None:
    listed: list[str] = []

    def list_nodes(site, cluster_id):
        listed.append(cluster_id)
        return _inventory(cluster_id)

    result = module.build_failure_domain_map(_site(), list_nodes=list_nodes)

    assert listed == ["gpu-a", "gpu-b"]
    assert result.mapping == {
        "gpu-a": {"gpu-a-node-1": "gpu-a-group"},
        "gpu-b": {"gpu-b-node-1": "gpu-b-group"},
    }
    assert result.unmapped == {"gpu-a": ["gpu-a-node-2"], "gpu-b": ["gpu-b-node-2"]}
    assert result.label_keys == FAILURE_DOMAIN_LABELS


def test_label_keys_come_from_the_site_not_a_flag() -> None:
    site = _site(failure_domain_labels=[RACK])

    result = module.build_failure_domain_map(
        site, list_nodes=lambda _site, cluster_id: _inventory(cluster_id)
    )

    assert result.label_keys == (RACK,)
    assert result.mapping == {
        "gpu-a": {"gpu-a-node-1": "gpu-a-rack"},
        "gpu-b": {"gpu-b-node-1": "gpu-b-rack"},
    }


def test_node_listing_selects_the_hyperpod_cluster_like_the_release_engine() -> None:
    kubectl = FakeKubectl()

    nodes = module.cluster_nodes(_site(), "gpu-a", run=kubectl)

    ((args, _),) = kubectl.calls
    assert args[:5] == [
        "kubectl",
        "--kubeconfig",
        args[2],
        "--context",
        "gpu-a-context",
    ]
    assert args[-2:] == ["-l", f"{module.HYPERPOD_CLUSTER_LABEL}=hp-gpu-a"]
    assert sorted(nodes) == ["gpu-a-node-1", "gpu-a-node-2"]
    with pytest.raises(BootstrapError, match="gpu-z"):
        module.cluster_nodes(_site(), "gpu-z", run=kubectl)


def test_configmap_carries_the_document_the_executor_loads(tmp_path: Path) -> None:
    mapping = {"gpu-a": {"node-1": "group-a"}}

    manifest = module.failure_domain_configmap(mapping)

    assert manifest["kind"] == "ConfigMap"
    assert manifest["metadata"] == {
        "name": module.FAILURE_DOMAIN_CONFIGMAP,
        "namespace": "gpu-fault-system",
    }
    document = manifest["data"][module.FAILURE_DOMAIN_FILE]
    assert manifest["data"]["map-path"] == module.FAILURE_DOMAIN_MAP_PATH
    # Round trip through the executor-side loader: what the product renders is
    # exactly what the worker will refuse or accept at start-up.
    path = tmp_path / module.FAILURE_DOMAIN_FILE
    path.write_text(document, encoding="utf-8")
    assert load_failure_domain_map(path) == mapping


def test_apply_ships_the_map_and_stamps_the_worker_pod_template() -> None:
    kubectl = FakeKubectl()

    result = module.apply_failure_domain_map(_site(), run=kubectl)

    assert kubectl.verbs() == [
        "get nodes",
        "get nodes",
        "apply",
        "get deployment",
        "patch",
    ]
    apply_args, apply_kwargs = next(c for c in kubectl.calls if "apply" in c[0])
    assert apply_args[:5] == [
        "kubectl",
        "--kubeconfig",
        "/secure/cpu.kubeconfig",
        "-n",
        "gpu-fault-system",
    ]
    manifest = json.loads(apply_kwargs["input"])
    assert json.loads(manifest["data"][module.FAILURE_DOMAIN_FILE]) == result.mapping
    patch_args, _ = next(c for c in kubectl.calls if "patch" in c[0])
    assert patch_args[-4:-2] == [module.CONTROL_WORKER_DEPLOYMENT, "--type=merge"]
    patch = json.loads(patch_args[-1])
    assert patch["spec"]["template"]["metadata"]["annotations"] == {
        FAILURE_DOMAIN_MAP_ANNOTATION: failure_domain_map_sha256(manifest)
    }


def test_apply_leaves_the_stamp_to_deploy_when_the_worker_does_not_exist_yet() -> None:
    kubectl = FakeKubectl(worker_exists=False)

    module.apply_failure_domain_map(_site(), run=kubectl)

    assert kubectl.verbs() == ["get nodes", "get nodes", "apply", "get deployment"]


def test_apply_refuses_a_malformed_map_before_touching_the_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kubectl = FakeKubectl()
    monkeypatch.setattr(
        module,
        "failure_domain_map",
        lambda cluster_id, *_a, **_k: {cluster_id: {"n": ""}},
    )

    with pytest.raises(BootstrapError, match="non-empty failure domain"):
        module.apply_failure_domain_map(_site(), run=kubectl)

    assert "apply" not in kubectl.verbs()


def test_apply_surfaces_a_failed_kubectl_apply() -> None:
    def failing(args, **_kwargs):
        if "nodes" in args:
            return FakeKubectl()(args)
        return subprocess.CompletedProcess(args, 1, "", "forbidden")

    with pytest.raises(BootstrapError, match="forbidden"):
        module.apply_failure_domain_map(_site(), run=failing)


def test_join_renders_the_map_from_the_committed_site_after_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    committed = SimpleNamespace(release_config={"clusters": []})
    request = SimpleNamespace(
        site=SimpleNamespace(
            source=Path("/state/site.yaml"), repository_root=Path("/repo")
        )
    )
    monkeypatch.setattr(
        admin_cluster_join_commit,
        "activate_and_commit",
        lambda *_a, **_k: order.append("commit"),
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "load_site",
        lambda source, *, repository_root: order.append(f"reload {source}")
        or committed,
    )
    monkeypatch.setattr(
        admin_cluster_join,
        "apply_failure_domain_map",
        lambda site: order.append("render" if site is committed else "wrong site"),
    )

    admin_cluster_join.commit_membership(
        request,
        execution=None,
        state_dir=Path("/state"),
        state_path=Path("/state/s.json"),
        state={},
    )

    assert order == ["commit", "reload /state/site.yaml", "render"]


def test_remove_renders_the_map_for_the_remaining_clusters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []
    remaining = SimpleNamespace(release_config={"clusters": []})
    monkeypatch.setattr(admin_cluster_removal, "apply_failure_domain_map", seen.append)

    admin_cluster_removal.refresh_failure_domain_map(remaining)

    assert seen == [remaining]


def _spec(**overrides: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "repositoryRoot": "/repo",
        "awsRegion": "us-east-1",
        "cpu": {
            "kubeconfig": "/secure/cpu.kubeconfig",
            "eksArn": "arn:aws:eks:us-east-1:123456789012:cluster/control",
            "hyperpodClusterName": "control",
        },
        "release": {
            "manifest": "dist/current-release.json",
            "agentConfigDigest": "a" * 64,
        },
        "runtimeProfile": {
            "source": "config/profile.yaml",
            "version": "hyperpod-v1",
            "registrationClusterId": "gpu-a",
        },
        "nlb": {
            "name": "gpu-fault-regional",
            "publicSubnets": ["subnet-a", "subnet-b"],
            "securityGroup": "sg-123",
            "certificateArn": "arn:aws:acm:us-east-1:123456789012:certificate/test",
        },
        "images": {"runtime": "registry.example/runtime@sha256:" + "b" * 64},
        "health": {
            "auroraClusterId": "gpu-fault-aurora",
            "ampWorkspaceId": "ws-test",
            "snsTopicArn": "arn:aws:sns:us-east-1:123456789012:gpu-fault",
        },
        "clusters": [
            {
                "clusterId": "gpu-a",
                "context": "gpu-a",
                "hyperpodClusterName": "hp-gpu-a",
                "eksClusterArn": "arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
                "executorIrsaRoleArn": "arn:aws:iam::123456789012:role/executor",
                "allowedNamespaces": ["training", "gpu-fault-system"],
                "controlPlaneUrl": "https://control.example",
                "tokenFile": "/secure/token",
                "caFile": "/secure/ca.crt",
                "fleetMasterFile": "/secure/fleet-master",
            }
        ],
    }
    spec.update(overrides)
    return spec


def test_site_failure_domain_labels_default_to_the_shared_priority() -> None:
    assert SiteSpec.from_value(_spec()).failure_domain_labels == FAILURE_DOMAIN_LABELS


def test_site_failure_domain_labels_keep_the_operator_order() -> None:
    spec = SiteSpec.from_value(_spec(failureDomainLabels=[RACK, GROUP]))

    assert spec.failure_domain_labels == (RACK, GROUP), "finest domain first, unsorted"


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([], "at least 1"),
        ([RACK, RACK], "unique"),
        (RACK, "must be a list"),
        (["bad key!"], "invalid label key"),
    ],
)
def test_site_failure_domain_labels_are_validated(value: Any, message: str) -> None:
    with pytest.raises(SiteConfigError, match=message):
        SiteSpec.from_value(_spec(failureDomainLabels=value))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpu-fault-admin")
    commands = parser.add_subparsers(dest="command")
    module.add_failure_domain_map_command(
        commands, lambda command: command.add_argument("--state-dir")
    )
    return parser


def test_cli_is_a_read_only_debugging_aid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(
        module, "cluster_nodes", lambda site, cluster_id: _inventory(cluster_id)
    )
    applied: list[Any] = []
    monkeypatch.setattr(module, "apply_failure_domain_map", applied.append)
    output = tmp_path / "failure-domain-map.json"

    arguments = _parser().parse_args(
        ["failure-domain-map", "--state-dir", "/state", "--output", str(output)]
    )
    exit_code = module.run_failure_domain_map_command(arguments, site=_site())

    assert exit_code == 0
    assert applied == [], "the debugging aid never applies anything"
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(manifest["data"][module.FAILURE_DOMAIN_FILE]) == {
        "gpu-a": {"gpu-a-node-1": "gpu-a-group"},
        "gpu-b": {"gpu-b-node-1": "gpu-b-group"},
    }
    summary = json.loads(capsys.readouterr().out)
    assert summary["unmapped"] == {"gpu-a": ["gpu-a-node-2"], "gpu-b": ["gpu-b-node-2"]}
    assert summary["applied_by"] == "deploy"
    assert summary["sha256"] == failure_domain_map_sha256(manifest)
    assert summary["environment"] == {
        FAILURE_DOMAIN_MAP_ENV: module.FAILURE_DOMAIN_MAP_PATH
    }


def test_cli_no_longer_takes_label_or_cluster_flags(capsys) -> None:
    for flag in ("--label", "--cluster-id"):
        with pytest.raises(SystemExit):
            _parser().parse_args(
                ["failure-domain-map", "--state-dir", "/state", flag, "x"]
            )
        assert "unrecognized arguments" in capsys.readouterr().err
