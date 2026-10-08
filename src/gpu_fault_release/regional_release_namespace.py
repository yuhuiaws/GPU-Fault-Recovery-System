"""Namespace metadata the release owns on every deploy, not only at bootstrap.

``regional-control-plane-prerequisites.yaml`` is applied by the bootstrap
alone; an upgrade re-applies the role-split Deployments and never touches the
Namespace. The readiness-gate label that lets the AWS Load Balancer Controller
inject ``target-health.elbv2.k8s.aws/...`` gates into the api-ha Pods was
therefore missing on every upgraded site (live 2026-10-08: release cc58602a7540
rolled api-ha surge-first without gates). Every release now reconciles the
Namespace labels the manifest declares with a metadata-only merge patch before
the control plane rolls, records what it patched, and ``status --full`` /
``verify`` report whether the ingress Pods carry the gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault_release import repository_root
from gpu_fault_release.regional_release_config import ReleaseError

if TYPE_CHECKING:
    from gpu_fault_release.regional_admin_checks import CheckValue

READINESS_GATE_LABEL = "elbv2.k8s.aws/pod-readiness-gate-inject"
READINESS_GATE_CONDITION_PREFIX = "target-health.elbv2.k8s.aws/"
INGRESS_SELECTOR = "app=gpu-fault-api-ha"
PREREQUISITES = Path(
    "deploy/control-plane/regional/regional-control-plane-prerequisites.yaml"
)


def manifest_namespace_labels(root: Path | None = None) -> dict[str, str]:
    """The labels the prerequisites manifest declares on the Namespace."""

    text = ((root or repository_root()) / PREREQUISITES).read_text(encoding="utf-8")
    for document in yaml.safe_load_all(text):
        if isinstance(document, dict) and document.get("kind") == "Namespace":
            labels = (document.get("metadata") or {}).get("labels") or {}
            return {str(key): str(value) for key, value in labels.items()}
    raise ReleaseError("the control-plane prerequisites declare no Namespace")


def live_namespace_labels(release: Any) -> dict[str, str]:
    document = release._get_json(
        release._cpu("get", "namespace", release.config.namespace)
    )
    labels = (document.get("metadata") or {}).get("labels") or {}
    return {str(key): str(value) for key, value in labels.items()}


def reconcile_namespace_labels(release: Any) -> dict[str, Any]:
    """Patch the Namespace's declared labels that are missing or differ.

    Metadata-only (``kubectl patch --type merge`` on ``metadata.labels``): no
    other Namespace field is read or written. Returns the evidence the release
    records: the declared labels, the keys patched and the keys already present.
    """

    namespace = release.config.namespace
    expected = manifest_namespace_labels()
    live = live_namespace_labels(release)
    missing = {key: value for key, value in expected.items() if live.get(key) != value}
    if missing:
        release.runner.run(
            release._cpu(
                "patch",
                "namespace",
                namespace,
                "--type",
                "merge",
                "-p",
                json.dumps({"metadata": {"labels": missing}}, sort_keys=True),
            )
        )
    return {
        "namespace": namespace,
        "labels": expected,
        "patched": sorted(missing),
        "present": sorted(key for key in expected if key not in missing),
    }


def ingress_readiness_gate(release: Any) -> dict[str, Any]:
    """Whether the ingress Pods carry the controller's NLB readiness gate.

    ``injected``: the Namespace label is set and every api-ha Pod has a
    ``target-health.elbv2.k8s.aws/...`` readiness gate; ``missing``: the label
    is absent or no Pod has the gate (Pods created before the label existed
    keep running without one until they are rolled); ``partial``: some Pods do.
    """

    namespace = release.config.namespace
    label = live_namespace_labels(release).get(READINESS_GATE_LABEL)
    pods = (
        release._get_json(
            release._cpu("-n", namespace, "get", "pods", "-l", INGRESS_SELECTOR)
        ).get("items")
        or []
    )
    gated: list[str] = []
    ungated: list[str] = []
    for pod in pods:
        gates = (pod.get("spec") or {}).get("readinessGates") or []
        name = str((pod.get("metadata") or {}).get("name") or "")
        if any(
            str(gate.get("conditionType") or "").startswith(
                READINESS_GATE_CONDITION_PREFIX
            )
            for gate in gates
            if isinstance(gate, dict)
        ):
            gated.append(name)
        else:
            ungated.append(name)
    if label == "enabled" and pods and not ungated:
        status = "injected"
    elif label != "enabled" or not gated:
        status = "missing"
    else:
        status = "partial"
    return {
        "status": status,
        "label": label,
        "pods": len(pods),
        "gated_pods": sorted(gated),
        "ungated_pods": sorted(ungated),
    }


def controller_check_value(
    release: Any,
    matches: list[dict[str, Any]],
    check_value: Callable[..., CheckValue],
) -> CheckValue:
    """The ``load_balancer_controller`` check's value, with the gate status.

    ``check_value`` is the report's ``CheckValue`` type (passed in, as the
    checks module imports this one). A missing gate is a WARN, not a failure:
    the ingress still serves, but its roll is not NLB-aware until the Pods are
    rolled under the label.
    """

    if not matches:
        raise ReleaseError("no AWS Load Balancer Controller deployment was found")
    gate = ingress_readiness_gate(release)
    return check_value(
        f"AWS Load Balancer Controller is Ready; ingress readiness gate: {gate['status']}",
        {"controllers": matches, "ingress_readiness_gate": gate},
        status="PASS" if gate["status"] == "injected" else "WARN",
    )


__all__ = [
    "INGRESS_SELECTOR",
    "READINESS_GATE_CONDITION_PREFIX",
    "READINESS_GATE_LABEL",
    "controller_check_value",
    "ingress_readiness_gate",
    "live_namespace_labels",
    "manifest_namespace_labels",
    "reconcile_namespace_labels",
]
