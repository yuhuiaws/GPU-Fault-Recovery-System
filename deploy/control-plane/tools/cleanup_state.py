#!/usr/bin/env python3
"""Maintain the out-of-band regional cleanup state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

PHASES = (
    "PREFLIGHT",
    "CLUSTERS_DRAINING",
    "GPU_DATA_PLANE_SOURCES_STOPPED",
    "QUEUES_DRAINED",
    "CONTROL_CONSUMERS_STOPPED",
    "INGRESS_STOPPED",
    "GPU_EXECUTORS_STOPPED",
    "NODE_RUNTIMES_STOPPED",
    "CPU_AUXILIARIES_STOPPED",
    "APPLICATION_OBJECTS_DELETED",
    "NAMESPACES_DELETED",
    "CLEANUP_COMPLETED",
    "READY_TO_DELETE_AURORA",
    "AURORA_DELETED",
)
PHASE_INDEX = {phase: index for index, phase in enumerate(PHASES)}
STATUSES = {"IN_PROGRESS", "COMPLETED", "FAILED"}


class CleanupStateError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_payload(document: dict[str, Any]) -> bytes:
    value = {key: item for key, item in document.items() if key != "content_sha256"}
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def content_digest(document: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_payload(document)).hexdigest()


def verify_document(document: dict[str, Any]) -> None:
    if not isinstance(document, dict):
        raise CleanupStateError("cleanup state must be an object")
    require_current_order(document)
    expected = document.get("content_sha256")
    actual = content_digest(document)
    if expected != actual:
        raise CleanupStateError("cleanup state SHA-256 mismatch")
    if document.get("phase") not in PHASE_INDEX:
        raise CleanupStateError("invalid cleanup phase")
    if document.get("status") not in STATUSES:
        raise CleanupStateError("invalid cleanup status")
    history = document.get("history")
    if not isinstance(history, list) or not history:
        raise CleanupStateError("cleanup state lacks phase history")
    previous = -1
    for entry in history:
        if (
            not isinstance(entry, dict)
            or entry.get("phase") not in PHASE_INDEX
            or entry.get("status") not in STATUSES
            or PHASE_INDEX[entry["phase"]] < previous
        ):
            raise CleanupStateError("invalid cleanup phase history")
        previous = PHASE_INDEX[entry["phase"]]
    if any(history[-1].get(key) != document.get(key) for key in ("phase", "status")):
        raise CleanupStateError("cleanup phase differs from history")
    if PHASE_INDEX[document["phase"]] >= PHASE_INDEX["CLEANUP_COMPLETED"]:
        if not set(required_phases(document)).issubset(completed_phases(document)):
            raise CleanupStateError("cleanup evidence lacks completed lifecycle phases")


def require_current_order(document: dict[str, Any]) -> None:
    if (
        type(document.get("schema_version")) is not int
        or document["schema_version"] != 2
    ):
        raise CleanupStateError(
            "unsupported cleanup state schema; legacy phase order requires reconciliation"
        )
    if document.get("phase_order") != list(PHASES):
        raise CleanupStateError(
            "cleanup phase order differs; legacy phase order requires reconciliation"
        )


def completed_phases(document: dict[str, Any]) -> list[str]:
    latest = {entry["phase"]: entry["status"] for entry in document["history"]}
    return [phase for phase in PHASES if latest.get(phase) == "COMPLETED"]


def required_phases(document: dict[str, Any]) -> list[str]:
    required = {
        "PREFLIGHT",
        "GPU_DATA_PLANE_SOURCES_STOPPED",
        "QUEUES_DRAINED",
        "NODE_RUNTIMES_STOPPED",
        "GPU_EXECUTORS_STOPPED",
    }
    if document["scope"] == "all":
        required.update(
            {
                "CLUSTERS_DRAINING",
                "INGRESS_STOPPED",
                "CONTROL_CONSUMERS_STOPPED",
                "CPU_AUXILIARIES_STOPPED",
            }
        )
    if document["mode"] in {"clean", "reset"}:
        required.add("APPLICATION_OBJECTS_DELETED")
    if document["mode"] == "reset":
        required.add("NAMESPACES_DELETED")
    return [phase for phase in PHASES if phase in required]


def request_targets(
    config: dict[str, Any], *, scope: str, cluster_ids: list[str]
) -> dict[str, Any]:
    clusters = config.get("clusters")
    if not isinstance(clusters, list):
        raise CleanupStateError("cleanup clusters must be a list")
    by_id = {cluster["cluster_id"]: cluster for cluster in clusters}
    if len(by_id) != len(clusters):
        raise CleanupStateError("duplicate cleanup cluster identity")
    if scope == "gpu" and (
        not cluster_ids
        or not set(cluster_ids).issubset(by_id)
        or len(cluster_ids) != len(set(cluster_ids))
    ):
        raise CleanupStateError("cleanup requires distinct, known GPU cluster IDs")
    if scope not in {"all", "gpu"} or scope == "all" and cluster_ids:
        raise CleanupStateError("invalid cleanup scope")
    selected = [by_id[key] for key in cluster_ids] if scope == "gpu" else clusters
    return {
        "namespace": config.get("namespace", "gpu-fault-system"),
        "cpu_kubeconfig": config.get("cpu_kubeconfig"),
        "clusters": [
            {"cluster_id": cluster["cluster_id"], "context": cluster["context"]}
            for cluster in selected
        ],
    }


ACCEPT_CONFIG_ENV = "GPU_FAULT_CLEANUP_ACCEPT_CONFIG_SHA256"


def validate_request(
    document: dict[str, Any],
    *,
    config_path: Path,
    scope: str,
    mode: str,
    node_mode: str,
    cluster_ids: list[str],
    state_path: Path | None = None,
) -> None:
    verify_document(document)
    config_text = config_path.read_text(encoding="utf-8")
    expected = {
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "scope": scope,
        "mode": mode,
        "node_mode": node_mode,
        "targets": request_targets(
            json.loads(config_text), scope=scope, cluster_ids=cluster_ids
        ),
    }
    accepted = os.environ.get(ACCEPT_CONFIG_ENV, "").strip()
    if (
        document.get("config_sha256") != expected["config_sha256"]
        and accepted
        and accepted == expected["config_sha256"]
    ):
        # The administrator re-materialized the release config from another
        # reviewed source tree (``uninstall --accept-repository-root-override``);
        # only the config's location-bound fields may differ, and scope, modes
        # and targets are still compared below. Keep the previous digest.
        history = document.get("config_sha256_history")
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "previous": document.get("config_sha256"),
                "accepted": accepted,
                "observed_at": now(),
                "message": "config digest accepted by the administrator",
            }
        )
        document["config_sha256_history"] = history
        document["config_sha256"] = accepted
        if state_path is not None:
            atomic_write(state_path, document)
    for key, value in expected.items():
        if document.get(key) != value:
            raise CleanupStateError(f"cleanup request differs on {key}")


def read_state(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CleanupStateError(f"cannot read cleanup state: {exc}") from exc
    verify_document(document)
    return cast(dict[str, Any], document)


def atomic_write(
    path: Path,
    document: dict[str, Any],
) -> None:
    document["updated_at"] = now()
    document["content_sha256"] = content_digest(document)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(
                document,
                stream,
                indent=2,
                ensure_ascii=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
        path.chmod(0o600)
        directory = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY,
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def initialize(
    path: Path,
    *,
    config_path: Path,
    inventory_path: Path,
    scope: str,
    mode: str,
    node_mode: str,
    cluster_ids: list[str] | None = None,
) -> dict[str, Any]:
    if path.exists():
        raise CleanupStateError(f"cleanup state already exists: {path}")
    config_text = config_path.read_text(encoding="utf-8")
    inventory_text = inventory_path.read_text(encoding="utf-8")
    config = json.loads(config_text)
    inventory = json.loads(inventory_text)
    created_at = now()
    document = {
        "schema_version": 2,
        "phase_order": list(PHASES),
        "run_id": f"cleanup-{uuid4()}",
        "created_at": created_at,
        "updated_at": created_at,
        "mode": mode,
        "scope": scope,
        "node_mode": node_mode,
        "phase": "PREFLIGHT",
        "status": "IN_PROGRESS",
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "inventory_sha256": hashlib.sha256(inventory_text.encode()).hexdigest(),
        "targets": request_targets(config, scope=scope, cluster_ids=cluster_ids or []),
        "inventory_snapshot": inventory,
        "fleet_snapshot": None,
        "fleet_snapshot_sha256": None,
        "original_resources": [],
        "history": [
            {
                "phase": "PREFLIGHT",
                "status": "IN_PROGRESS",
                "observed_at": created_at,
                "message": "cleanup state initialized",
            }
        ],
    }
    atomic_write(path, document)
    return document


def record_resource(
    document: dict[str, Any],
    *,
    resource_scope: str,
    context: str,
    kind: str,
    name: str,
    previous: str,
) -> None:
    identity = (
        resource_scope,
        context,
        kind,
        name,
    )
    resources = document["original_resources"]
    for resource in resources:
        existing = (
            resource["scope"],
            resource["context"],
            resource["kind"],
            resource["name"],
        )
        if existing != identity:
            continue
        if resource["previous"] != previous:
            raise CleanupStateError("original resource state changed during capture")
        return
    resources.append(
        {
            "scope": resource_scope,
            "context": context,
            "kind": kind,
            "name": name,
            "previous": previous,
        }
    )


def attach_fleet_snapshot(
    document: dict[str, Any],
    snapshot: list[dict[str, Any]],
) -> None:
    encoded = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    document["fleet_snapshot"] = snapshot
    document["fleet_snapshot_sha256"] = hashlib.sha256(encoded).hexdigest()


def transition(
    document: dict[str, Any],
    *,
    phase: str,
    status: str,
    message: str,
) -> None:
    if phase not in PHASE_INDEX:
        raise CleanupStateError(f"unknown cleanup phase: {phase}")
    if status not in STATUSES:
        raise CleanupStateError(f"unknown cleanup status: {status}")
    require_current_order(document)
    current_phase = document["phase"]
    current_status = document["status"]
    if PHASE_INDEX[phase] < PHASE_INDEX[current_phase]:
        raise CleanupStateError("cleanup phase cannot move backwards")
    if phase == "CLEANUP_COMPLETED" and status == "COMPLETED":
        if not set(required_phases(document)).issubset(completed_phases(document)):
            raise CleanupStateError(
                "cleanup completion requires completed lifecycle phases"
            )
    if phase == "READY_TO_DELETE_AURORA" and (
        current_phase not in {"CLEANUP_COMPLETED", "READY_TO_DELETE_AURORA"}
        or current_status != "COMPLETED"
    ):
        raise CleanupStateError("Aurora handoff requires completed CLEANUP_COMPLETED")
    if phase == "AURORA_DELETED" and (
        current_phase
        not in {
            "READY_TO_DELETE_AURORA",
            "AURORA_DELETED",
        }
        or current_status != "COMPLETED"
    ):
        raise CleanupStateError(
            "Aurora deletion requires completed READY_TO_DELETE_AURORA"
        )
    document["phase"] = phase
    document["status"] = status
    document["history"].append(
        {
            "phase": phase,
            "status": status,
            "observed_at": now(),
            "message": message,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    init = subparsers.add_parser("init")
    init.add_argument("--path", type=Path, required=True)
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--inventory", type=Path, required=True)
    init.add_argument("--scope", required=True)
    init.add_argument("--mode", required=True)
    init.add_argument("--node-mode", required=True)
    init.add_argument("--cluster-id", action="append", default=[])

    record = subparsers.add_parser("record")
    record.add_argument("--path", type=Path, required=True)
    record.add_argument("--resource-scope", required=True)
    record.add_argument("--context", required=True)
    record.add_argument("--kind", required=True)
    record.add_argument("--name", required=True)
    record.add_argument("--previous", required=True)

    fleet = subparsers.add_parser("attach-fleet")
    fleet.add_argument("--path", type=Path, required=True)
    fleet.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    # Retain a clear refusal for older callers; another journal is not authority.
    reuse = subparsers.add_parser("reuse-fleet")
    reuse.add_argument("--path", type=Path, required=True)
    reuse.add_argument("--from", dest="source", type=Path, required=True)

    update = subparsers.add_parser("transition")
    update.add_argument("--path", type=Path, required=True)
    update.add_argument(
        "--phase",
        choices=PHASES,
        required=True,
    )
    update.add_argument(
        "--status",
        choices=sorted(STATUSES),
        required=True,
    )
    update.add_argument("--message", default="")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--path", type=Path, required=True)
    verify.add_argument("--config", type=Path)
    verify.add_argument("--scope", choices=("all", "gpu"))
    verify.add_argument("--mode", choices=("stop", "clean", "reset"))
    verify.add_argument("--node-mode", choices=("stop", "uninstall", "skip"))
    verify.add_argument("--cluster-id", action="append", default=[])

    resume = subparsers.add_parser("resume")
    resume.add_argument("--path", type=Path, required=True)
    resume.add_argument("--config", type=Path, required=True)
    resume.add_argument("--scope", choices=("all", "gpu"), required=True)
    resume.add_argument("--mode", choices=("stop", "clean", "reset"), required=True)
    resume.add_argument(
        "--node-mode", choices=("stop", "uninstall", "skip"), required=True
    )
    resume.add_argument("--cluster-id", action="append", default=[])
    resume.add_argument("--inventory-output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "init":
        document = initialize(
            args.path,
            config_path=args.config,
            inventory_path=args.inventory,
            scope=args.scope,
            mode=args.mode,
            node_mode=args.node_mode,
            cluster_ids=args.cluster_id,
        )
    else:
        document = read_state(args.path)
        if args.command in {"verify", "resume"} and args.config is not None:
            if not all((args.scope, args.mode, args.node_mode)):
                raise CleanupStateError(
                    "cleanup request verification needs scope and modes"
                )
            validate_request(
                document,
                config_path=args.config,
                scope=args.scope,
                mode=args.mode,
                node_mode=args.node_mode,
                cluster_ids=args.cluster_id,
                state_path=args.path,
            )
            if args.command == "resume":
                if PHASE_INDEX[document["phase"]] > PHASE_INDEX["CLEANUP_COMPLETED"]:
                    raise CleanupStateError(
                        "cleanup cannot resume after Aurora handoff"
                    )
                args.inventory_output.write_text(
                    json.dumps(document["inventory_snapshot"]), encoding="utf-8"
                )
                args.inventory_output.chmod(0o600)
        elif args.command == "record":
            record_resource(
                document,
                resource_scope=args.resource_scope,
                context=args.context,
                kind=args.kind,
                name=args.name,
                previous=args.previous,
            )
            atomic_write(args.path, document)
        elif args.command == "attach-fleet":
            snapshot = json.loads(args.input.read_text(encoding="utf-8"))
            if not isinstance(snapshot, list):
                raise CleanupStateError("fleet snapshot must be a JSON list")
            attach_fleet_snapshot(document, snapshot)
            atomic_write(args.path, document)
        elif args.command == "reuse-fleet":
            raise CleanupStateError(
                "fleet reuse across cleanup journals is unsupported; resume the original bound journal"
            )
        elif args.command == "transition":
            transition(
                document,
                phase=args.phase,
                status=args.status,
                message=args.message,
            )
            atomic_write(args.path, document)
    print(
        json.dumps(
            {
                "path": str(args.path),
                "phase": document["phase"],
                "status": document["status"],
                "content_sha256": document["content_sha256"],
                "completed_phases": completed_phases(document),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
