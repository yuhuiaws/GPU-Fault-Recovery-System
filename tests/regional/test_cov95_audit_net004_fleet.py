"""NET-004 audit: the managed fleet's registry, site and context reads, CPU identity."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import gpu_fault.admin.site as admin_site
from scripts.e2e.regional import audit_net004_dependency_boundary as audit
from tests.regional._cov95_collect_net import no_external_effects  # noqa: F401
from tests.regional.test_cov95_net_dependency_audit import boundary  # noqa: F401

READY_POD = {
    "metadata": {"name": "api-a", "uid": "api-a-uid"},
    "spec": {"containers": [{"name": "api"}]},
    "status": {
        "phase": "Running",
        "conditions": [{"type": "Ready", "status": "True"}],
        "containerStatuses": [{"name": "api", "ready": True, "containerID": "c-1"}],
    },
}


class ControlPlane:
    """``audit.cpu``: the API Pod listing and the registry probe's answer."""

    def __init__(self, *, pods: list[dict[str, Any]], registry: Any) -> None:
        self.pods = pods
        self.registry = registry
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *args: str, stdin: str | None = None, timeout: int = 120) -> str:
        self.calls.append(args)
        if args[:2] == ("get", "pods"):
            return json.dumps({"items": self.pods})
        assert args[0] == "exec" and stdin == audit.REGISTRY_CLUSTERS_PROBE
        return "connecting\n" + json.dumps(self.registry)


@pytest.mark.parametrize(
    ("pods", "registry", "message"),
    [
        ([], {}, "no Ready control-plane API Pod for the registry read"),
        ([READY_POD], {"registrations": []}, "durable registry names no GPU clusters"),
        ([READY_POD], {"registrations": "gpu-a"}, "durable registry names no GPU"),
        (
            [READY_POD],
            {"registrations": [{"eks_cluster_arn": "gpu-a"}]},
            "durable registry registration lacks an EKS cluster ARN",
        ),
    ],
)
def test_the_registry_read_refuses_what_cannot_name_the_fleet(
    monkeypatch: pytest.MonkeyPatch, pods: list[Any], registry: Any, message: str
) -> None:
    monkeypatch.setattr(audit, "cpu", ControlPlane(pods=pods, registry=registry))

    with pytest.raises(audit.CaseError, match=message):
        audit.registered_gpu_eks_names()


def test_the_registry_names_every_cluster_by_its_eks_arn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plane = ControlPlane(
        pods=[READY_POD],
        registry={
            "registrations": [
                {"eks_cluster_arn": "arn:aws:eks:us-west-2:0:cluster/gpu-hp"},
                {"eks_cluster_arn": "arn:aws:eks:us-west-2:0:cluster/gpu-eks"},
            ]
        },
    )
    monkeypatch.setattr(audit, "cpu", plane)

    assert audit.registered_gpu_eks_names() == ("gpu-hp", "gpu-eks")
    assert plane.calls[1][:3] == ("exec", "-i", "api-a")


def site_with(monkeypatch: pytest.MonkeyPatch, *arns: str) -> list[Path]:
    loaded: list[Path] = []

    def load_site(path: Path) -> Any:
        loaded.append(path)
        return SimpleNamespace(
            release_config={"clusters": [{"eks_cluster_arn": arn} for arn in arns]}
        )

    monkeypatch.setattr(admin_site, "load_site", load_site)
    return loaded


def test_a_site_without_gpu_clusters_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded = site_with(monkeypatch)

    with pytest.raises(audit.CaseError, match="site contains no GPU clusters"):
        audit.managed_gpu_eks_names(tmp_path / "site.yaml", selected="gpu-a")

    assert loaded == [(tmp_path / "site.yaml").resolve()]


def test_kubeconfig_contexts_keep_only_eks_cluster_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = {
        "contexts": [
            {"name": "local", "context": {"cluster": "kind-local"}},
            {"name": "gpu-a", "context": {"cluster": "arn:aws:eks:r:0:cluster/gpu-a"}},
            {"name": "alias", "context": {"cluster": "arn:aws:eks:r:0:cluster/gpu-a"}},
            {"name": "empty"},
        ]
    }
    monkeypatch.setattr(audit, "gpu", lambda *args, **kwargs: json.dumps(view))

    assert audit.kubeconfig_eks_contexts() == {"gpu-a": "gpu-a"}


def test_first_running_pod_returns_the_named_pod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        audit, "gpu", lambda *args, **kwargs: calls.append(args) or "executor-a\n"
    )

    assert audit.first_running_pod("app=executor", control=False) == "executor-a"
    assert calls[0][:4] == ("get", "pod", "-l", "app=executor")


def test_a_site_scoped_audit_of_one_cluster_keeps_the_selected_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site_with(monkeypatch, "arn:aws:eks:us-west-2:0:cluster/gpu-a")
    monkeypatch.setattr(audit, "SITE_PATH", tmp_path / "site.yaml")
    monkeypatch.setattr(audit, "GPU_EKS_NAME", "gpu-a")
    monkeypatch.setattr(audit, "GPU_CONTEXT", "context-a")
    monkeypatch.setattr(audit, "GPU_EKS_NAMES", ())
    monkeypatch.setattr(audit, "GPU_CONTEXTS", {})
    monkeypatch.setattr(
        audit, "gpu", lambda *args, **kwargs: pytest.fail("a site needs no kube read")
    )

    assert audit.resolve_managed_fleet() == "site"
    assert (audit.GPU_EKS_NAMES, audit.GPU_CONTEXTS) == (
        ("gpu-a",),
        {"gpu-a": "context-a"},
    )


def test_a_repeated_cpu_role_is_refused_before_any_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "CONTROL_APP", "gpu-fault-control-worker")
    monkeypatch.setattr(
        audit, "cpu", lambda *args, **kwargs: pytest.fail("no read for a bad inventory")
    )

    with pytest.raises(
        audit.CaseError, match="CPU role inventory repeats a Deployment"
    ):
        audit.cpu_credential_boundary({})


@pytest.mark.parametrize(
    "deployment",
    [
        {"metadata": {"generation": 1}, "spec": {"replicas": 1}},
        {"metadata": {"uid": "u", "generation": "1"}, "spec": {"replicas": 1}},
        {"metadata": {"uid": "u", "generation": 1}, "spec": {"replicas": 0}},
    ],
)
def test_a_cpu_deployment_without_a_valid_identity_is_refused(
    monkeypatch: pytest.MonkeyPatch, deployment: dict[str, Any]
) -> None:
    monkeypatch.setattr(audit, "cpu", lambda *args, **kwargs: json.dumps(deployment))

    with pytest.raises(audit.CaseError, match="has no valid CPU Deployment identity"):
        audit.cpu_credential_boundary({})


@pytest.mark.usefixtures("boundary")
def test_a_cpu_deployment_that_changes_during_the_proof_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = audit.run_fixture_command
    reads: list[str] = []

    def changing(argv: list[str], **kwargs: Any) -> Any:
        result = transport(argv, **kwargs)
        if "deployment" in argv:
            name = argv[argv.index("deployment") + 1]
            reads.append(name)
            if reads.count(name) == 2:
                value = json.loads(result.stdout)
                value["metadata"]["generation"] += 1
                return SimpleNamespace(stdout=json.dumps(value))
        return result

    monkeypatch.setattr(audit, "run_fixture_command", changing)

    with pytest.raises(
        audit.CaseError, match="CPU Deployment changed during credential boundary proof"
    ):
        audit.cpu_credential_boundary(audit.eks_and_nat_facts()[0])

    assert reads == [audit.CONTROL_APP, audit.CONTROL_APP]
