#!/usr/bin/env python3
"""Generate the regional cleanup inventory from source manifests."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "deploy" / "control-plane" / "regional" / "cleanup-inventory.json"
MANIFEST_GLOBS = {
    "cpu": [
        "deploy/control-plane/base/*.yaml",
        "deploy/control-plane/regional/*.yaml",
        "deploy/control-plane/regional/generated/*.yaml",
        "deploy/observability/*.yaml",
    ],
    "gpu": ["deploy/dataplane/**/*.yaml"],
}
IMPLICIT_NAMESPACE_KINDS = [
    "ConfigMap",
    "Kustomization",
    "Namespace",
    "Secret",
]
CLUSTER_KINDS = {
    "ClusterRole",
    "ClusterRoleBinding",
    "CustomResourceDefinition",
}
SUPPORT_KINDS = {
    "ClusterRole",
    "ClusterRoleBinding",
    "PodDisruptionBudget",
    "Role",
    "RoleBinding",
    "Service",
    "ServiceAccount",
}
WORKLOAD_KINDS = {"CronJob", "DaemonSet", "Deployment"}
KNOWN_KINDS = (
    set(IMPLICIT_NAMESPACE_KINDS)
    | CLUSTER_KINDS
    | SUPPORT_KINDS
    | WORKLOAD_KINDS
    | {"PrometheusRule"}
)
PHASE_ORDER = {
    "cpu": {
        "ingress": 0,
        "consumer": 1,
        "auxiliary": 2,
        "support": 3,
        "nlb": 4,
    },
    "gpu": {
        "producer": 0,
        "executor": 1,
        "support": 2,
    },
}
ANNOTATION_PREFIX = "gpu-fault.io/"
PHASE_ANNOTATION = ANNOTATION_PREFIX + "cleanup-phase"
ORDER_ANNOTATION = ANNOTATION_PREFIX + "cleanup-order"
ACTION_ANNOTATION = ANNOTATION_PREFIX + "cleanup-action"
DEPLOY_ANNOTATION = ANNOTATION_PREFIX + "deploy-mode"
DEPLOY_MODES = {"regional_reconciler", "regional_rollout"}
NODE_ANNOTATIONS = [
    "gpu-fault.io/installer-version",
    "gpu-fault.io/installer-config-digest",
    "gpu-fault.io/installer-artifact-sha256",
    "gpu-fault.io/installer-node-uid",
    "gpu-fault.io/installer-state",
    "gpu-fault.io/incident-id",
    "gpu-fault.io/fencing-token",
    "gpu-fault.io/previous-unschedulable",
    "gpu-fault.io/mechanical-inspection-complete",
    "gpu-fault.io/efa-plugin-restart-operation",
    "gpu-fault.io/efa-plugin-restart-pod-uid",
    "gpu-fault.io/efa-plugin-restart-started-at",
    "gpu-fault.io/gpu-plugin-restart-operation",
    "gpu-fault.io/gpu-plugin-restart-pod-uid",
    "gpu-fault.io/gpu-plugin-restart-started-at",
    "gpu-fault.io/spare-reservation",
    "gpu-fault.io/spare-pool-state",
    "gpu-fault.io/spare-health",
    "gpu-fault.io/spare-health-failures",
    "gpu-fault.io/spare-health-incident",
    "gpu-fault.io/spare-health-unavailable-at",
    "gpu-fault.io/spare-health-last-alert-at",
]
NODE_LABELS = ["gpu-fault.io/spare"]


def _documents(
    plane: str,
) -> dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]]:
    found: dict[
        tuple[str, str],
        list[tuple[Path, dict[str, Any]]],
    ] = defaultdict(list)
    for pattern in MANIFEST_GLOBS[plane]:
        paths = sorted(ROOT.glob(pattern))
        if not paths:
            raise RuntimeError(f"manifest glob matched nothing: {pattern}")
        for path in paths:
            for document in yaml.safe_load_all(path.read_text(encoding="utf-8")):
                if not isinstance(document, dict):
                    continue
                kind = document.get("kind")
                name = (document.get("metadata") or {}).get("name")
                if not kind or not name:
                    continue
                if kind not in KNOWN_KINDS:
                    raise RuntimeError(f"unsupported manifest kind {kind} in {path}")
                found[(kind, name)].append((path, document))
    return found


def _annotation_values(
    candidates: list[tuple[Path, dict[str, Any]]],
    key: str,
) -> set[str]:
    values = set()
    for _, document in candidates:
        value = (document.get("metadata") or {}).get("annotations", {}).get(key)
        if value is not None:
            values.add(str(value))
    return values


def _one_annotation(
    identity: tuple[str, str],
    candidates: list[tuple[Path, dict[str, Any]]],
    key: str,
) -> str | None:
    values = _annotation_values(candidates, key)
    if len(values) > 1:
        raise RuntimeError(f"conflicting {key} values for {identity}: {sorted(values)}")
    return next(iter(values), None)


def _resource(
    plane: str,
    identity: tuple[str, str],
    candidates: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any] | None:
    kind, name = identity
    phase = _one_annotation(
        identity,
        candidates,
        PHASE_ANNOTATION,
    )
    if kind in IMPLICIT_NAMESPACE_KINDS and phase is None:
        return None
    if phase is None:
        if kind in WORKLOAD_KINDS:
            paths = ", ".join(str(path.relative_to(ROOT)) for path, _ in candidates)
            raise RuntimeError(f"{kind}/{name} lacks {PHASE_ANNOTATION}: {paths}")
        phase = "support"
    if phase not in PHASE_ORDER[plane]:
        raise RuntimeError(f"invalid {plane} cleanup phase {phase!r} for {kind}/{name}")

    order_value = _one_annotation(
        identity,
        candidates,
        ORDER_ANNOTATION,
    )
    try:
        order = int(order_value or "100")
    except ValueError as exc:
        raise RuntimeError(f"invalid cleanup order for {kind}/{name}") from exc
    if order < 0:
        raise RuntimeError(f"cleanup order must be non-negative: {kind}/{name}")

    action = _one_annotation(
        identity,
        candidates,
        ACTION_ANNOTATION,
    )
    if action is None:
        action = "namespace" if kind == "PrometheusRule" else "delete"
    if action not in {"delete", "namespace", "reset"}:
        raise RuntimeError(f"invalid cleanup action for {kind}/{name}: {action}")

    deploy = _one_annotation(
        identity,
        candidates,
        DEPLOY_ANNOTATION,
    )
    if deploy is not None:
        if plane != "gpu" or kind != "Deployment" or deploy not in DEPLOY_MODES:
            raise RuntimeError(f"invalid deploy mode for {kind}/{name}: {deploy}")
        deploy_sources = [
            path
            for path, document in candidates
            if (
                (document.get("metadata") or {})
                .get("annotations", {})
                .get(DEPLOY_ANNOTATION)
                == deploy
            )
        ]
        if len(deploy_sources) != 1:
            raise RuntimeError(f"{kind}/{name} deploy mode must have one source")

    resource: dict[str, Any] = {
        "kind": kind.lower(),
        "name": name,
        "scope": ("cluster" if kind in CLUSTER_KINDS else "namespaced"),
        "phase": phase,
        "order": order,
        "clean": action,
    }
    if deploy is not None:
        resource["deploy"] = deploy
        resource["manifest"] = deploy_sources[0].name
    return resource


def generate() -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_by": ("scripts/generate-cleanup-inventory.py"),
    }
    for plane in ("cpu", "gpu"):
        resources = [
            resource
            for identity, candidates in _documents(plane).items()
            if (
                resource := _resource(
                    plane,
                    identity,
                    candidates,
                )
            )
            is not None
        ]
        resources.sort(
            key=lambda item: (
                PHASE_ORDER[plane][item["phase"]],
                item["order"],
                item["kind"],
                item["name"],
            )
        )
        section: dict[str, Any] = {
            "manifest_globs": MANIFEST_GLOBS[plane],
            "implicit_namespace_kinds": (IMPLICIT_NAMESPACE_KINDS),
        }
        if plane == "cpu":
            section["database_pod_preference"] = [
                item["name"]
                for item in resources
                if item["kind"] == "deployment" and item["phase"] == "consumer"
            ] + [
                item["name"]
                for item in resources
                if item["kind"] == "deployment" and item["phase"] == "ingress"
            ]
        else:
            section["node_annotations"] = NODE_ANNOTATIONS
            section["node_labels"] = NODE_LABELS
        section["resources"] = resources
        result[plane] = section
    return result


def rendered() -> str:
    return (
        json.dumps(
            generate(),
            indent=2,
            ensure_ascii=True,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = rendered()
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.is_file() else ""
        if current != expected:
            print("cleanup inventory is stale; run make deployment-contracts-update")
            return 1
        print("cleanup inventory is current")
        return 0
    OUTPUT.write_text(expected, encoding="utf-8")
    print(OUTPUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
