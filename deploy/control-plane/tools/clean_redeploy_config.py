#!/usr/bin/env python3
"""Validate the inputs of prepare-clean-redeploy.sh and emit its target records.

Runs on the deploy host (``python3 clean_redeploy_config.py CONFIG INVENTORY
[CLUSTER_ID ...]``). Every line printed is one tab-separated record the script
reads back: ``META namespace cpu_kubeconfig``, ``CLUSTER id context`` for each
selected cluster, ``RESOURCE plane kind name scope phase clean`` for each
inventory resource, ``DBPREF name`` for the CPU database Pod preference and
``ANNOTATION``/``LABEL`` for the GPU node metadata the reset strips. Any
problem exits non-zero with the reason, so nothing is printed half-validated.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

DNS_LABEL = r"[a-z0-9]([-a-z0-9]*[a-z0-9])?"
VALID_SCOPES = {"namespaced", "cluster"}
VALID_CLEAN = {"delete", "namespace", "reset"}
VALID_PHASES = {
    "ingress",
    "consumer",
    "auxiliary",
    "producer",
    "executor",
    "support",
    "nlb",
}


def load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid {label}: {exc}")


def refuse_unregistered(inventory: dict[str, Any]) -> None:
    unregistered = inventory.get("unregistered_resources") or []
    if unregistered:
        details = ", ".join(
            f"{item['context']}:{item.get('namespace') or '_cluster'}:"
            f"{item['kind']}/{item['name']}"
            for item in unregistered
        )
        raise SystemExit(
            "unregistered live gpu-fault resources require "
            "classification before cleanup: " + details
        )


def emit_targets(value: dict[str, Any], selected: list[str]) -> None:
    cpu_kubeconfig = value.get("cpu_kubeconfig")
    namespace = value.get("namespace", "gpu-fault-system")
    clusters = value.get("clusters")
    if not isinstance(cpu_kubeconfig, str) or not cpu_kubeconfig:
        raise SystemExit("cpu_kubeconfig must be a non-empty string")
    if not isinstance(namespace, str) or not re.fullmatch(DNS_LABEL, namespace):
        raise SystemExit("namespace is not a valid DNS label")
    if not isinstance(clusters, list) or not clusters:
        raise SystemExit("clusters must be a non-empty list")

    by_id: dict[str, str] = {}
    for item in clusters:
        if not isinstance(item, dict):
            raise SystemExit("each clusters entry must be an object")
        cluster_id = item.get("cluster_id")
        context = item.get("context")
        if not isinstance(cluster_id, str) or not cluster_id:
            raise SystemExit("each cluster_id must be a non-empty string")
        if not isinstance(context, str) or not context:
            raise SystemExit(f"cluster {cluster_id!r} has no context")
        if any(character in cluster_id + context for character in "\t\r\n"):
            raise SystemExit("cluster_id and context may not contain tabs or newlines")
        if cluster_id in by_id:
            raise SystemExit(f"duplicate cluster_id: {cluster_id}")
        by_id[cluster_id] = context

    if selected:
        if len(selected) != len(set(selected)):
            raise SystemExit("duplicate --cluster-id")
        missing = sorted(set(selected) - set(by_id))
        if missing:
            raise SystemExit("unknown cluster_id: " + ", ".join(missing))
        cluster_ids = selected
    else:
        cluster_ids = list(by_id)

    print(f"META\t{namespace}\t{cpu_kubeconfig}")
    for cluster_id in cluster_ids:
        print(f"CLUSTER\t{cluster_id}\t{by_id[cluster_id]}")


def text_fields(resource: dict[str, Any], plane: str) -> tuple[str, str, str, str, str]:
    values: list[str] = []
    for field in ("kind", "name", "scope", "phase", "clean"):
        value = resource.get(field)
        if not isinstance(value, str) or not value:
            raise SystemExit(f"{plane} resource entry has an empty field")
        values.append(value)
    kind, name, scope, phase, clean = values
    return kind, name, scope, phase, clean


def emit_inventory(inventory: dict[str, Any]) -> None:
    if inventory.get("schema_version") != 1:
        raise SystemExit("cleanup resource inventory schema_version must be 1")
    for plane in ("cpu", "gpu"):
        section = inventory.get(plane)
        if not isinstance(section, dict):
            raise SystemExit(f"cleanup resource inventory lacks {plane}")
        resources = section.get("resources")
        if not isinstance(resources, list) or not resources:
            raise SystemExit(f"cleanup resource inventory {plane}.resources is empty")
        seen: set[tuple[str, str]] = set()
        for resource in resources:
            if not isinstance(resource, dict):
                raise SystemExit(f"{plane} resource entry must be an object")
            kind, name, scope, phase, clean = text_fields(resource, plane)
            if any(
                character in "".join((kind, name, scope, phase, clean))
                for character in "\t\r\n"
            ):
                raise SystemExit(
                    f"{plane} resource fields may not contain whitespace controls"
                )
            if not re.fullmatch(r"[a-z][a-z0-9]*", kind):
                raise SystemExit(f"invalid kubectl resource kind: {kind}")
            if not re.fullmatch(DNS_LABEL, name):
                raise SystemExit(f"invalid resource name: {name}")
            if scope not in VALID_SCOPES:
                raise SystemExit(f"invalid resource scope: {scope}")
            if phase not in VALID_PHASES:
                raise SystemExit(f"invalid cleanup phase: {phase}")
            if clean not in VALID_CLEAN:
                raise SystemExit(f"invalid cleanup action: {clean}")
            identity = (kind, name)
            if identity in seen:
                raise SystemExit(f"duplicate {plane} resource: {kind}/{name}")
            seen.add(identity)
            print("RESOURCE", plane, kind, name, scope, phase, clean, sep="\t")
        if plane == "cpu":
            preferences = section.get("database_pod_preference")
            if not isinstance(preferences, list) or not preferences:
                raise SystemExit("cpu.database_pod_preference is empty")
            for preference in preferences:
                if ("deployment", preference) not in seen:
                    raise SystemExit(
                        f"database pod preference is not a CPU deployment: {preference}"
                    )
                print("DBPREF", preference, sep="\t")
        else:
            annotations = section.get("node_annotations")
            if not isinstance(annotations, list) or not annotations:
                raise SystemExit("gpu.node_annotations is empty")
            for annotation in annotations:
                if not isinstance(annotation, str) or not annotation.startswith(
                    "gpu-fault.io/"
                ):
                    raise SystemExit(f"invalid node annotation: {annotation!r}")
                print("ANNOTATION", annotation, sep="\t")
            labels = section.get("node_labels")
            if not isinstance(labels, list) or not labels:
                raise SystemExit("gpu.node_labels is empty")
            for label in labels:
                if not isinstance(label, str) or not label.startswith("gpu-fault.io/"):
                    raise SystemExit(f"invalid node label: {label!r}")
                print("LABEL", label, sep="\t")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        raise SystemExit(
            "usage: clean_redeploy_config.py CONFIG INVENTORY [CLUSTER_ID ...]"
        )
    value = load_json(Path(argv[0]), "regional release config")
    inventory = load_json(Path(argv[1]), "cleanup resource inventory")
    if not isinstance(value, dict):
        raise SystemExit("regional release config must be an object")
    if not isinstance(inventory, dict):
        raise SystemExit("cleanup resource inventory must be an object")
    refuse_unregistered(inventory)
    emit_targets(value, argv[2:])
    emit_inventory(inventory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
