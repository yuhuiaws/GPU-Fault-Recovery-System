"""Identity-bound checkpoints for one cluster removal attempt."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError, safe_name
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from gpu_fault.admin.site import IDENTIFIER_PATTERN, RenderedSite

SCHEMA_VERSION = 2
STEP_DEPENDENCIES = {
    "DISCOVERED": (),
    "CONTROL_REGISTRY_DRAINING": ("DISCOVERED",),
    "KUBERNETES_QUIESCED": ("CONTROL_REGISTRY_DRAINING",),
    "KUBERNETES_REMOVED": ("KUBERNETES_QUIESCED",),
    "CONTROL_REGISTRY_REMOVED": ("KUBERNETES_REMOVED",),
    "AWS_DETACHED": ("KUBERNETES_REMOVED",),
    "AURORA_UPDATED": ("CONTROL_REGISTRY_REMOVED", "AWS_DETACHED"),
    "SITE_COMMIT_STARTED": ("AURORA_UPDATED",),
    "SITE_UPDATED": ("SITE_COMMIT_STARTED",),
    "RELEASE_STATE_UPDATED": ("SITE_UPDATED",),
    "VERIFIED": ("RELEASE_STATE_UPDATED",),
}


@contextmanager
def removal_command_ownership(path: Path, state: dict[str, Any]) -> Iterator[None]:
    if state.get("phase") == "SUPERVISION_LOST":
        raise ProcessSupervisionLost(
            "previous removal command ownership is unproven; "
            "automatic retry is forbidden"
        )
    try:
        yield
    except ProcessSupervisionLost as exc:
        state["phase"] = "SUPERVISION_LOST"
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        try:
            write_json_atomic(path, state)
        except Exception:
            exc.add_note("removal could not persist unproven command ownership")
        raise


def allowed_registry_lifecycles(state: dict[str, Any]) -> set[str | None]:
    completed = set(state["completed_steps"])
    if "CONTROL_REGISTRY_REMOVED" in completed:
        return {None}
    if "KUBERNETES_REMOVED" in completed:
        return {"DRAINING", "REVOKED", None}
    if "CONTROL_REGISTRY_DRAINING" in completed:
        return {"DRAINING"}
    if "DISCOVERED" in completed:
        return {"ACTIVE", "DRAINING"}
    return {"ACTIVE"}


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def site_identity(site: RenderedSite) -> dict[str, str]:
    return {
        key: str(site.release_config[key])
        for key in ("site_name", "aws_region", "cpu_eks_arn", "namespace")
    }


def site_documents(site: RenderedSite, cluster_id: str) -> tuple[str, str]:
    try:
        document = yaml.safe_load(site.source.read_text(encoding="utf-8"))
        without = copy.deepcopy(document)
        without["spec"]["clusters"] = [
            item
            for item in without["spec"]["clusters"]
            if item["clusterId"] != cluster_id
        ]
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        raise BootstrapError("cannot bind remove-cluster site document") from exc
    return canonical_digest(document), canonical_digest(without)


def release_documents(site: RenderedSite, cluster_id: str) -> tuple[str, str]:
    without = {
        **site.release_config,
        "clusters": [
            item
            for item in site.release_config["clusters"]
            if item["cluster_id"] != cluster_id
        ],
    }
    return canonical_digest(site.release_config), canonical_digest(without)


def read_removal_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BootstrapError("cannot read remove-cluster state") from exc
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise BootstrapError(
            "remove-cluster state lacks current identity binding; "
            "legacy evidence requires reconciliation"
        )
    completed = value.get("completed_steps")
    if (
        not isinstance(completed, list)
        or any(not isinstance(step, str) for step in completed)
        or len(completed) != len(set(completed))
        or set(completed) - STEP_DEPENDENCIES.keys()
        or not isinstance(value.get("evidence"), dict)
        or not isinstance(value.get("target"), dict)
    ):
        raise BootstrapError("remove-cluster state has invalid checkpoints")
    for step in completed:
        if not set(STEP_DEPENDENCIES[step]).issubset(completed):
            raise BootstrapError("remove-cluster state has incomplete safety barriers")
    phase = value.get("phase")
    if (
        not isinstance(phase, str)
        or phase not in {"STARTED", "COMPLETED", "SUPERVISION_LOST", *STEP_DEPENDENCIES}
        or phase == "STARTED"
        and completed
        or phase in STEP_DEPENDENCIES
        and phase not in completed
        or phase == "COMPLETED"
        and set(completed) != STEP_DEPENDENCIES.keys()
    ):
        raise BootstrapError("remove-cluster state has an invalid phase")
    if "DISCOVERED" in completed:
        discovery = value["evidence"].get("DISCOVERED")
        if (
            not isinstance(discovery, dict)
            or discovery.get("target") != value["target"]
            or not isinstance(discovery.get("provider_identity"), dict)
            or not discovery["provider_identity"].get("eks_arn")
            or not discovery["provider_identity"].get("hyperpod_arn")
            or not discovery.get("namespace_uid")
            or not discovery.get("registry_digest")
            or not discovery.get("token_sha256")
        ):
            raise BootstrapError("remove-cluster discovery evidence is incomplete")
    attempt = value.get("attempt_id")
    if (
        not isinstance(attempt, str)
        or len(attempt) != 32
        or any(character not in "0123456789abcdef" for character in attempt)
    ):
        raise BootstrapError("remove-cluster state has an invalid attempt identity")
    return value


def validate_saved_site(
    site: RenderedSite,
    cluster_id: str,
    state: dict[str, Any],
) -> None:
    if (
        state.get("site_identity") != site_identity(site)
        or state.get("site_id") != site.release_config["site_name"]
        or state.get("cluster_id") != cluster_id
        or state["target"].get("cluster_id") != cluster_id
    ):
        raise BootstrapError(
            "remove-cluster state conflicts with site or cluster identity"
        )
    current, _ = site_documents(site, cluster_id)
    release, _ = release_documents(site, cluster_id)
    present = [
        item
        for item in site.release_config["clusters"]
        if item["cluster_id"] == cluster_id
    ]
    if present:
        if (
            len(present) != 1
            or dict(present[0]) != state["target"]
            or current != state.get("source_site_sha256")
            or release != state.get("source_release_sha256")
            or "SITE_UPDATED" in state["completed_steps"]
        ):
            raise BootstrapError(
                "remove-cluster target or source site identity drifted"
            )
    elif (
        "SITE_COMMIT_STARTED" not in state["completed_steps"]
        or current != state.get("remaining_site_sha256")
        or release != state.get("remaining_release_sha256")
    ):
        raise BootstrapError(
            "remove-cluster target is absent without a matching partial site commit"
        )


def load_removal_state(
    site: RenderedSite,
    cluster_id: str,
) -> tuple[Path, Path, dict[str, Any]]:
    if not IDENTIFIER_PATTERN.fullmatch(cluster_id):
        raise BootstrapError("remove-cluster cluster identity is invalid")
    root = site.source.parent / "remove-cluster" / safe_name(cluster_id)
    path = root / "state.json"
    matches = [
        dict(item)
        for item in site.release_config["clusters"]
        if item["cluster_id"] == cluster_id
    ]
    if path.exists():
        value = read_removal_state(path)
        if value.get("site_identity") != site_identity(site):
            raise BootstrapError("remove-cluster state conflicts with site identity")
        if value.get("cluster_id") != cluster_id:
            raise BootstrapError("remove-cluster state conflicts with cluster identity")
        if value["phase"] != "COMPLETED" or not matches:
            validate_saved_site(site, cluster_id, value)
            directory = root / "attempts" / value["attempt_id"]
            if not directory.is_dir():
                raise BootstrapError("remove-cluster attempt evidence is missing")
            return directory, path, value
        # A rejoined member needs new discovery, never the old successful cleanup.
        archive = root / "history" / f"{value['attempt_id']}.json"
        if not archive.exists():
            write_json_atomic(archive, value)
    if len(matches) != 1:
        raise BootstrapError(f"unknown cluster_id: {cluster_id}")
    before, after = site_documents(site, cluster_id)
    release_before, release_after = release_documents(site, cluster_id)
    value = {
        "schema_version": SCHEMA_VERSION,
        "site_id": site.release_config["site_name"],
        "cluster_id": cluster_id,
        "site_identity": site_identity(site),
        "source_site_sha256": before,
        "remaining_site_sha256": after,
        "source_release_sha256": release_before,
        "remaining_release_sha256": release_after,
        "target": matches[0],
        "attempt_id": uuid4().hex,
        "phase": "STARTED",
        "completed_steps": [],
        "evidence": {},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    directory = root / "attempts" / value["attempt_id"]
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_json_atomic(path, value)
    return directory, path, value


def resolve_saved_removal(site: RenderedSite, gpu_cluster_arn: str) -> str | None:
    """Find an exact, post-commit tombstone without inferring a target by name."""
    matches: list[str] = []
    root = site.source.parent / "remove-cluster"
    for path in sorted(root.glob("*/state.json")):
        state = read_removal_state(path)
        discovery = state["evidence"].get("DISCOVERED") or {}
        identity = discovery.get("provider_identity") or {}
        if gpu_cluster_arn not in {
            identity.get("eks_arn"),
            identity.get("hyperpod_arn"),
        }:
            continue
        cluster_id = state.get("cluster_id")
        if (
            not isinstance(cluster_id, str)
            or not IDENTIFIER_PATTERN.fullmatch(cluster_id)
            or path.parent.name != safe_name(cluster_id)
        ):
            raise BootstrapError("remove-cluster tombstone identity is invalid")
        validate_saved_site(site, cluster_id, state)
        if "SITE_COMMIT_STARTED" not in state["completed_steps"]:
            raise BootstrapError("remove-cluster tombstone has not started site commit")
        matches.append(cluster_id)
    if len(matches) > 1:
        raise BootstrapError("multiple remove-cluster tombstones match this ARN")
    return matches[0] if matches else None


def removal_result(
    site: RenderedSite,
    cluster_id: str,
    state_path: Path,
    state: dict[str, Any],
    remaining: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "site_id": site.release_config["site_name"],
        "cluster_id": cluster_id,
        "phase": state["phase"],
        "remaining_cluster_ids": [item["cluster_id"] for item in remaining],
        "state_file": str(state_path),
        "cpu_control_plane": "PRESERVED",
        "gpu_cluster": "PRESERVED",
    }
