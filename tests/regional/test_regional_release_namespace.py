"""The release reconciles the Namespace labels on every deploy and reports the
NLB readiness gate of the ingress Pods.

Live 2026-10-08: the readiness-gate label reached only bootstrapped sites; the
upgrade to cc58602a7540 rolled api-ha surge-first without gates because the
prerequisites manifest is applied by the bootstrap alone."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_release_namespace as namespace
from gpu_fault_release import rollout
from gpu_fault_release.regional_admin_checks import CheckValue
from gpu_fault_release.regional_release_config import ReleaseError

GATE = "target-health.elbv2.k8s.aws/k8s-gpufault-gpufault-0123456789"


class FakeRelease:
    """kubectl reads and writes of one CPU cluster, recorded."""

    def __init__(self, *, labels: dict[str, str], pods: list[dict[str, Any]]) -> None:
        self.config = SimpleNamespace(
            namespace="gpu-fault-system", cpu_kubeconfig="cpu"
        )
        self.labels = dict(labels)
        self.pods = pods
        self.patches: list[list[str]] = []
        self.runner = SimpleNamespace(run=self.run_kubectl)
        self.state: dict[str, Any] = {}

    def _cpu(self, *args: str) -> list[str]:
        return ["kubectl", "--kubeconfig", "cpu", *args]

    def _get_json(self, argv: list[str]) -> dict[str, Any]:
        if "namespace" in argv:
            return {
                "metadata": {"name": "gpu-fault-system", "labels": dict(self.labels)}
            }
        assert "pods" in argv, argv
        return {"items": self.pods}

    def run_kubectl(self, argv: list[str], **_options: Any) -> str:
        assert "patch" in argv and "--type" in argv, "only a metadata merge patch"
        self.patches.append(list(argv))
        self.labels.update(json.loads(argv[-1])["metadata"]["labels"])
        return ""


class UnreadableRelease(FakeRelease):
    """kubectl cannot read the Namespace: a first deploy's preflight runs
    before the prerequisites create it."""

    def _get_json(self, argv: list[str]) -> dict[str, Any]:
        raise ReleaseError('namespaces "gpu-fault-system" not found')


def _pod(name: str, *, gated: bool) -> dict[str, Any]:
    spec = {"readinessGates": [{"conditionType": GATE}]} if gated else {}
    return {"metadata": {"name": name}, "spec": spec}


def test_manifest_declares_the_readiness_gate_label() -> None:
    labels = namespace.manifest_namespace_labels()
    assert labels[namespace.READINESS_GATE_LABEL] == "enabled", labels


def test_reconcile_patches_only_the_missing_labels_metadata_only() -> None:
    release = FakeRelease(
        labels={"kubernetes.io/metadata.name": "gpu-fault-system"}, pods=[]
    )

    evidence = namespace.reconcile_namespace_labels(release)

    assert evidence["patched"] == [namespace.READINESS_GATE_LABEL], evidence
    assert evidence["present"] == [], evidence
    assert evidence["namespace"] == "gpu-fault-system", evidence
    (patch,) = release.patches
    assert patch[:6] == [
        "kubectl",
        "--kubeconfig",
        "cpu",
        "patch",
        "namespace",
        "gpu-fault-system",
    ]
    assert patch[6:9] == ["--type", "merge", "-p"], patch
    assert json.loads(patch[-1]) == {
        "metadata": {"labels": {namespace.READINESS_GATE_LABEL: "enabled"}}
    }, "only metadata.labels travels in the patch"
    assert release.labels["kubernetes.io/metadata.name"] == "gpu-fault-system", (
        "existing labels are untouched"
    )

    again = namespace.reconcile_namespace_labels(release)

    assert again["patched"] == [] and again["present"] == [
        namespace.READINESS_GATE_LABEL
    ]
    assert len(release.patches) == 1, "a label already present is not patched again"


def test_ingress_readiness_gate_reports_injected_partial_and_missing() -> None:
    labeled = {namespace.READINESS_GATE_LABEL: "enabled"}
    injected = namespace.ingress_readiness_gate(
        FakeRelease(labels=labeled, pods=[_pod("a", gated=True), _pod("b", gated=True)])
    )
    assert injected["status"] == "injected" and injected["gated_pods"] == ["a", "b"], (
        injected
    )
    partial = namespace.ingress_readiness_gate(
        FakeRelease(
            labels=labeled, pods=[_pod("a", gated=True), _pod("b", gated=False)]
        )
    )
    assert partial["status"] == "partial" and partial["ungated_pods"] == ["b"], (
        "Pods created before the label have no gate until they are rolled"
    )
    missing = namespace.ingress_readiness_gate(
        FakeRelease(labels=labeled, pods=[_pod("a", gated=False)])
    )
    assert missing["status"] == "missing", missing
    unlabeled = namespace.ingress_readiness_gate(
        FakeRelease(labels={}, pods=[_pod("a", gated=True)])
    )
    assert unlabeled["status"] == "missing" and unlabeled["label"] is None, unlabeled


def test_ingress_readiness_gate_reports_unknown_with_the_reason_when_unreadable() -> (
    None
):
    gate = namespace.ingress_readiness_gate(UnreadableRelease(labels={}, pods=[]))

    assert gate["status"] == "unknown", gate
    assert gate["reason"] == 'ReleaseError: namespaces "gpu-fault-system" not found'
    assert gate["label"] is None and gate["pods"] == 0, gate
    assert gate["gated_pods"] == [] and gate["ungated_pods"] == [], gate
    without_config = namespace.ingress_readiness_gate(SimpleNamespace(state={}))
    assert without_config["status"] == "unknown", without_config
    assert "namespace" in without_config["reason"], without_config


def test_controller_check_warns_with_the_reason_when_the_gate_cannot_be_read() -> None:
    """The gate rides on the controller's verdict; a read that fails must not
    turn a Ready controller into a FAIL that stops the deploy."""

    matches = [{"namespace": "kube-system", "name": "lbc", "ready": 2, "replicas": 2}]

    value = namespace.controller_check_value(
        UnreadableRelease(labels={}, pods=[]), matches, CheckValue
    )

    assert value.status == "WARN", value
    assert value.summary == (
        "AWS Load Balancer Controller is Ready; ingress readiness gate: unknown "
        '(ReleaseError: namespaces "gpu-fault-system" not found)'
    ), value.summary
    assert value.details["ingress_readiness_gate"]["status"] == "unknown", value
    assert value.details["controllers"] == matches, value


def test_reconcile_does_not_tolerate_an_unreadable_namespace() -> None:
    """The deploy-side reconcile mutates; a Namespace it cannot read stops the
    release rather than being skipped."""

    with pytest.raises(ReleaseError, match="not found"):
        namespace.reconcile_namespace_labels(UnreadableRelease(labels={}, pods=[]))


def test_controller_check_value_names_the_gate_and_warns_when_missing() -> None:
    matches = [{"namespace": "kube-system", "name": "lbc", "ready": 2, "replicas": 2}]
    release = FakeRelease(
        labels={namespace.READINESS_GATE_LABEL: "enabled"}, pods=[_pod("a", gated=True)]
    )

    value = namespace.controller_check_value(release, matches, CheckValue)

    assert value.summary == (
        "AWS Load Balancer Controller is Ready; ingress readiness gate: injected"
    ), value
    assert value.status == "PASS" and value.details["controllers"] == matches, value
    warned = namespace.controller_check_value(
        FakeRelease(labels={}, pods=[_pod("a", gated=False)]), matches, CheckValue
    )
    assert warned.status == "WARN", "a missing gate is a finding, not a failed deploy"
    assert warned.summary.endswith("ingress readiness gate: missing"), warned.summary
    with pytest.raises(ReleaseError, match="no AWS Load Balancer Controller"):
        namespace.controller_check_value(release, [], CheckValue)


def test_every_control_plane_role_apply_reconciles_the_namespace_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bootstrap and upgrade both apply the roles through
    render_and_apply_cpu_roles, so the label reaches an upgraded site too."""
    order: list[str] = []

    def reconcile(release: Any) -> dict[str, Any]:
        order.append("namespace-labels")
        return {"patched": [namespace.READINESS_GATE_LABEL], "present": []}

    class Runner:
        def run(self, argv: list[str], **_kwargs: Any) -> str:
            order.append(
                "render"
                if "render-control-plane-role-split.sh" in str(argv)
                else "apply"
            )
            return ""

    release = SimpleNamespace(
        runner=Runner(),
        state={},
        config=SimpleNamespace(namespace="gpu-fault-system", cpu_kubeconfig="cpu"),
    )
    monkeypatch.setattr(rollout, "reconcile_namespace_labels", reconcile)
    monkeypatch.setattr(
        rollout,
        "apply_rendered_cpu_roles",
        lambda *_a, **_k: order.append("apply"),
        raising=False,
    )

    try:
        rollout.render_and_apply_cpu_roles(release, {})
    except Exception:  # noqa: BLE001 - the render itself is not under test here
        pass

    assert order and order[0] == "namespace-labels", (
        "the labels are reconciled before the roles are rendered or applied"
    )
    assert release.state["namespace_labels"]["patched"] == [
        namespace.READINESS_GATE_LABEL
    ], "the release state records what the apply patched"
