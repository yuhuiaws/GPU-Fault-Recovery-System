"""Capture release evidence without Secret or binary payloads."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped,unused-ignore]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "releases"


def run_json(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def kubectl_json(
    *,
    kubeconfig: str | None = None,
    context: str | None = None,
    namespace: str | None = None,
    resources: tuple[str, ...],
) -> dict[str, Any]:
    command = ["kubectl"]
    if kubeconfig:
        command.extend(["--kubeconfig", kubeconfig])
    if context:
        command.extend(["--context", context])
    if namespace:
        command.extend(["-n", namespace])
    command.extend(["get", ",".join(resources), "-o", "json"])
    return run_json(command)


def _clean_metadata(metadata: dict[str, Any]) -> None:
    for key in (
        "managedFields",
        "resourceVersion",
        "selfLink",
        "uid",
    ):
        metadata.pop(key, None)
    annotations = metadata.get("annotations") or {}
    annotations.pop(
        "kubectl.kubernetes.io/last-applied-configuration",
        None,
    )
    if annotations:
        metadata["annotations"] = annotations
    else:
        metadata.pop("annotations", None)


def sanitize_resource(
    value: dict[str, Any],
) -> dict[str, Any]:
    item = copy.deepcopy(value)
    _clean_metadata(item.setdefault("metadata", {}))
    item.pop("status", None)
    return item


def sanitize_config_map(
    value: dict[str, Any],
) -> dict[str, Any]:
    item = sanitize_resource(value)
    binary = item.pop("binaryData", None) or {}
    if binary:
        item["evidenceBinaryData"] = {
            key: {
                "sha256": hashlib.sha256(base64.b64decode(encoded)).hexdigest(),
                "decodedBytes": len(base64.b64decode(encoded)),
            }
            for key, encoded in sorted(binary.items())
        }
    return item


def config_map_references(
    resources: list[dict[str, Any]],
) -> set[str]:
    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key in (
                "configMap",
                "configMapRef",
                "configMapKeyRef",
            ):
                reference = value.get(key)
                if isinstance(reference, dict) and reference.get("name"):
                    found.add(str(reference["name"]))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(resources)
    return found


def selected_config_maps(
    config_maps: list[dict[str, Any]],
    workloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    referenced = config_map_references(workloads)
    selected = []
    for item in config_maps:
        name = str(item.get("metadata", {}).get("name", ""))
        if (
            name in referenced
            or name == "gpu-fault-release-metadata"
            or "-config-" in name
        ):
            selected.append(sanitize_config_map(item))
    return selected


def secret_inventory(secrets: list[dict[str, Any]]) -> str:
    lines = ["NAME\tTYPE\tCREATED\tKEY\tSHA256"]
    for item in sorted(
        secrets,
        key=lambda value: value["metadata"]["name"],
    ):
        metadata = item["metadata"]
        data = item.get("data") or {}
        if not data:
            lines.append(
                f"{metadata['name']}\t{item.get('type', '')}\t"
                f"{metadata.get('creationTimestamp', '')}\t-\t-"
            )
            continue
        for key, encoded in sorted(data.items()):
            digest = hashlib.sha256(base64.b64decode(encoded)).hexdigest()
            lines.append(
                f"{metadata['name']}\t{item.get('type', '')}\t"
                f"{metadata.get('creationTimestamp', '')}\t"
                f"{key}\t{digest}"
            )
    return "\n".join(lines) + "\n"


def write_yaml(path: Path, items: list[dict[str, Any]]) -> None:
    document = {
        "apiVersion": "v1",
        "kind": "List",
        "items": items,
    }
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, width=100),
        encoding="utf-8",
    )
    path.chmod(0o600)


def operator_identity() -> str:
    try:
        value = run_json(
            [
                "aws",
                "sts",
                "get-caller-identity",
                "--output",
                "json",
            ]
        )
        return str(value.get("Arn") or "")
    except (OSError, subprocess.SubprocessError):
        return os.getenv("USER", "unknown")


def safe_release_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    if not normalized:
        raise ValueError("release id must contain a safe character")
    return normalized


def capture(args: argparse.Namespace) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    release_id = safe_release_id(args.release_id)
    output = args.output or (args.output_root / f"{release_id}-{stamp}")
    control_dir = output / "control"
    gpu_dir = output / "gpu"
    control_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    gpu_dir.mkdir(parents=True, mode=0o700, exist_ok=True)

    control_workloads = kubectl_json(
        kubeconfig=args.cpu_kubeconfig,
        namespace=args.namespace,
        resources=("deployments", "cronjobs", "pdb"),
    )["items"]
    gpu_workloads = kubectl_json(
        context=args.gpu_context,
        namespace=args.namespace,
        resources=("deployments", "daemonsets", "cronjobs"),
    )["items"]
    control_maps = kubectl_json(
        kubeconfig=args.cpu_kubeconfig,
        namespace=args.namespace,
        resources=("configmaps",),
    )["items"]
    gpu_maps = kubectl_json(
        context=args.gpu_context,
        namespace=args.namespace,
        resources=("configmaps",),
    )["items"]
    control_secrets = kubectl_json(
        kubeconfig=args.cpu_kubeconfig,
        namespace=args.namespace,
        resources=("secrets",),
    )["items"]
    gpu_secrets = kubectl_json(
        context=args.gpu_context,
        namespace=args.namespace,
        resources=("secrets",),
    )["items"]
    nodes = kubectl_json(
        context=args.gpu_context,
        resources=("nodes",),
    )["items"]

    phase = args.phase
    write_yaml(
        control_dir / f"workloads-{phase}.yaml",
        [sanitize_resource(item) for item in control_workloads],
    )
    write_yaml(
        control_dir / f"configmaps-{phase}.yaml",
        selected_config_maps(control_maps, control_workloads),
    )
    write_yaml(
        gpu_dir / f"workloads-{phase}.yaml",
        [sanitize_resource(item) for item in gpu_workloads],
    )
    write_yaml(
        gpu_dir / f"configmaps-{phase}.yaml",
        selected_config_maps(gpu_maps, gpu_workloads),
    )
    write_yaml(
        gpu_dir / f"nodes-{phase}.yaml",
        [sanitize_resource(item) for item in nodes],
    )

    inventory_path = output / "secret-inventory.txt"
    inventory_path.write_text(
        "# control plane\n"
        + secret_inventory(control_secrets)
        + "\n# gpu data plane\n"
        + secret_inventory(gpu_secrets),
        encoding="utf-8",
    )
    inventory_path.chmod(0o600)

    release_metadata = next(
        (
            item.get("data") or {}
            for item in control_maps
            if item.get("metadata", {}).get("name") == "gpu-fault-release-metadata"
        ),
        {},
    )
    inventory = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "operator": args.operator or operator_identity(),
        "release_id": args.release_id,
        "phase": phase,
        "namespace": args.namespace,
        "cpu_kubeconfig": args.cpu_kubeconfig,
        "gpu_context": args.gpu_context,
        "release_metadata": release_metadata,
        "control_configmaps": [
            item["metadata"]["name"]
            for item in selected_config_maps(control_maps, control_workloads)
        ],
        "gpu_configmaps": [
            item["metadata"]["name"]
            for item in selected_config_maps(gpu_maps, gpu_workloads)
        ],
    }
    inventory_file = output / "inventory.json"
    inventory_file.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inventory_file.chmod(0o600)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument(
        "--phase",
        choices=("before", "after"),
        default="after",
    )
    parser.add_argument(
        "--cpu-kubeconfig",
        default="/tmp/gpu-fault-control-plane.kubeconfig",
    )
    parser.add_argument("--gpu-context", required=True)
    parser.add_argument(
        "--namespace",
        default="gpu-fault-system",
    )
    parser.add_argument("--operator")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    output = capture(parse_args())
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
