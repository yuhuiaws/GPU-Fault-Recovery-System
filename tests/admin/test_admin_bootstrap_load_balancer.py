"""Ownership probes for the AWS Load Balancer Controller.

``test_admin_bootstrap_probes.py`` covers the healthy-reuse path, where every
ownership decision is stubbed. These tests drive the decisions themselves: the
readiness scan, the Helm ownership check and the value comparison that decides
whether an existing release is ours to reuse.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pytest

from gpu_fault.admin import bootstrap_load_balancer as lbc
from gpu_fault.admin.bootstrap_common import SITE_TAG_KEY, BootstrapError
from tests.admin._bootstrap_support import _cluster

ROLE_ARN = "arn:aws:iam::123456789012:role/gpu-fault-site-a-lbc"
POLICY_ARN = "arn:aws:iam::123456789012:policy/gpu-fault-site-a-lbc-policy"


def _deployment(image: str, *, replicas: int, ready: int) -> dict[str, Any]:
    return {
        "spec": {
            "replicas": replicas,
            "template": {"spec": {"containers": [{"image": image}]}},
        },
        "status": {"readyReplicas": ready},
    }


def _values(**overrides: Any) -> dict[str, Any]:
    cpu = _cluster()
    return {
        "clusterName": cpu.eks_name,
        "region": cpu.region,
        "vpcId": cpu.vpc_id,
        "replicaCount": 2,
        "serviceAccount": {"create": False, "name": "aws-load-balancer-controller"},
        **overrides,
    }


class Runner:
    """Answers the read-only queries and records everything else."""

    dry_run = False

    def __init__(
        self,
        *,
        deployments: list[dict[str, Any]] | None = None,
        deployment_error: bool = False,
        chart_version: str = lbc.LBC_CHART_VERSION,
        values: dict[str, Any] | None = None,
        policy_tags: list[dict[str, str]] | None = None,
        attached: Sequence[str] = (),
    ) -> None:
        self.deployments = deployments or []
        self.deployment_error = deployment_error
        self.chart_version = chart_version
        self.values = values if values is not None else _values()
        self.policy_tags = policy_tags if policy_tags is not None else []
        self.attached = tuple(attached)
        self.calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def run(self, arguments: Sequence[str], **keywords: Any) -> str:
        command = tuple(arguments)
        self.calls.append((command, keywords))
        if command[0] == "kubectl" and "get" in command:
            if self.deployment_error:
                raise BootstrapError("kubectl get deployment failed")
            return json.dumps({"items": self.deployments})
        if command[0] == "helm" and "status" in command:
            return json.dumps({"chart": {"metadata": {"version": self.chart_version}}})
        if command[0] == "helm" and "values" in command:
            return json.dumps(self.values)
        if "list-policy-tags" in command:
            return json.dumps({"Tags": self.policy_tags})
        return ""

    def aws_json(self, _region: str, *arguments: str, **_keywords: Any) -> Any:
        self.calls.append((("aws", *arguments), {}))
        if "list-attached-role-policies" in arguments:
            return {"AttachedPolicies": [{"PolicyArn": arn} for arn in self.attached]}
        return {}

    def commands(self) -> list[tuple[str, ...]]:
        return [command for command, _keywords in self.calls]

    def mutations(self) -> list[tuple[str, ...]]:
        return [
            command
            for command, keywords in self.calls
            if keywords.get("mutate") is True
        ]


def _stub_collaborators(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the shared IAM and Pod Identity helpers.

    They belong to ``bootstrap_services`` and have their own tests; what is under
    test here is which of them the ownership decision reaches.
    """

    cpu = _cluster()
    monkeypatch.setattr(lbc, "_ensure_pod_identity_agent", lambda *_args: {})
    monkeypatch.setattr(
        lbc,
        "_ensure_role",
        lambda *_args, **_keywords: {"role_arn": ROLE_ARN, "ownership": "CREATED"},
    )
    monkeypatch.setattr(
        lbc, "_ensure_service_account", lambda *_args, **_keywords: None
    )
    monkeypatch.setattr(
        lbc,
        "_ensure_pod_identity_association",
        lambda *_args, **_keywords: {
            "association_id": "assoc-a",
            "ownership": "CREATED",
            "cluster_name": cpu.eks_name,
            "namespace": "kube-system",
            "service_account": "aws-load-balancer-controller",
        },
    )


def _stub_processes(
    monkeypatch: pytest.MonkeyPatch, *, helm: tuple[int, str], policy_exists: bool
) -> None:
    returncode, stderr = helm

    def run(arguments: Sequence[str], **_keywords: Any) -> SimpleNamespace:
        if arguments[0] == "helm":
            return SimpleNamespace(returncode=returncode, stderr=stderr)
        return SimpleNamespace(returncode=0 if policy_exists else 254, stderr="")

    monkeypatch.setattr(lbc.subprocess, "run", run)


def _ensure(runner: Runner, tmp_path: Path) -> dict[str, Any]:
    return lbc.ensure_load_balancer_controller(
        runner,
        cpu=_cluster(),
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        state_dir=tmp_path,
        site_id="site-a",
    )


def test_a_cluster_managed_controller_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ready controller we do not own must not be adopted.

    Managed add-ons and vendor installs run their own controller without a Helm
    release of this name. Installing over it would leave two controllers
    reconciling the same Ingress objects, and uninstall would then take away
    something bootstrap never created.
    """

    _stub_processes(
        monkeypatch, helm=(1, "Error: release: not found"), policy_exists=False
    )
    runner = Runner(
        deployments=[
            _deployment(
                "public.ecr.aws/eks/aws-load-balancer-controller:v2.17.1",
                replicas=2,
                ready=2,
            )
        ]
    )

    result = _ensure(runner, tmp_path)

    assert result == {"external": True, "controller": "cluster-managed"}
    assert runner.mutations() == []


def test_a_partly_ready_controller_is_not_treated_as_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Readiness means every replica, and only the controller's own Deployment.

    Counting an unrelated Deployment, or one replica out of two, would report a
    cluster-managed controller and skip the install -- leaving the site with no
    working Ingress path and no error.
    """

    _stub_processes(
        monkeypatch, helm=(1, "Error: release: not found"), policy_exists=False
    )
    _stub_collaborators(monkeypatch)
    runner = Runner(
        deployments=[
            _deployment(
                "registry.k8s.io/kube-state-metrics:v2.10.0", replicas=1, ready=1
            ),
            _deployment(
                "public.ecr.aws/eks/aws-load-balancer-controller:v2.17.1",
                replicas=2,
                ready=1,
            ),
        ]
    )

    result = _ensure(runner, tmp_path)

    assert result["reused"] is False
    assert result["role_arn"] == ROLE_ARN
    commands = runner.commands()
    assert any("upgrade" in command for command in commands), (
        "a controller that is not ready was not installed"
    )
    assert any("create-policy" in command for command in commands), (
        "the missing IAM policy was not created"
    )
    assert any("attach-role-policy" in command for command in commands), (
        "the policy was not attached to the controller role"
    )
    assert any("rollout" in command for command in commands), (
        "the install did not wait for the controller rollout"
    )


def test_an_unreadable_deployment_list_is_not_read_as_a_ready_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed readiness query must not decide ownership either way.

    Treating the error as "ready" would skip the install on a transient API
    outage, which is the same silent failure as a partly ready controller.
    """

    _stub_processes(
        monkeypatch, helm=(1, "Error: release: not found"), policy_exists=False
    )
    _stub_collaborators(monkeypatch)
    runner = Runner(deployment_error=True)

    assert _ensure(runner, tmp_path)["reused"] is False


def test_a_release_whose_values_drifted_is_upgraded_rather_than_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reuse requires the release to still carry the values bootstrap set.

    A release pinned to another cluster, region, VPC or replica count is ours by
    name only; reusing it would leave the controller talking to the wrong cluster
    while bootstrap reported success.
    """

    _stub_processes(monkeypatch, helm=(0, ""), policy_exists=True)
    _stub_collaborators(monkeypatch)
    runner = Runner(
        deployments=[
            _deployment(
                "public.ecr.aws/eks/aws-load-balancer-controller:v2.17.1",
                replicas=2,
                ready=2,
            )
        ],
        values=_values(replicaCount=1),
        policy_tags=[{"Key": SITE_TAG_KEY, "Value": "site-a"}],
        attached=(POLICY_ARN,),
    )

    result = _ensure(runner, tmp_path)

    assert result["reused"] is False
    commands = runner.commands()
    assert any("upgrade" in command for command in commands), (
        "a drifted release was reused instead of upgraded"
    )
    assert not any("create-policy" in command for command in commands), (
        "an existing IAM policy was recreated"
    )
    assert not any("attach-role-policy" in command for command in commands), (
        "an already attached policy was attached again"
    )


def test_an_untagged_existing_policy_is_adopted_by_tagging_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The site tag is what uninstall reads to decide what it may delete.

    A policy created by an earlier bootstrap that predates the tag is still ours,
    so it is tagged rather than recreated -- and rather than left untagged, which
    would strand it forever.
    """

    _stub_processes(monkeypatch, helm=(0, ""), policy_exists=True)
    _stub_collaborators(monkeypatch)
    runner = Runner(
        deployments=[
            _deployment(
                "public.ecr.aws/eks/aws-load-balancer-controller:v2.17.1",
                replicas=2,
                ready=2,
            )
        ],
        chart_version="0.0.1",
        policy_tags=[],
    )

    _ensure(runner, tmp_path)

    tag = next(command for command in runner.commands() if "tag-policy" in command)
    assert f"Key={SITE_TAG_KEY},Value=site-a" in tag


def test_an_undecidable_helm_status_stops_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only "release: not found" means the release is absent.

    Any other Helm failure leaves ownership unknown, and guessing "absent" would
    install over a release that may belong to something else.
    """

    _stub_processes(
        monkeypatch,
        helm=(1, "Error: Kubernetes cluster unreachable"),
        policy_exists=False,
    )

    with pytest.raises(BootstrapError, match="cannot determine"):
        _ensure(Runner(), tmp_path)
