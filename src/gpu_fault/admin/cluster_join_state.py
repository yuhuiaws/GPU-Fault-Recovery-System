from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, cast

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_join_evidence import site_non_membership_sha256
from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault.admin.site import RenderedSite, load_site


class JoinStateRequest(Protocol):
    @property
    def site(self) -> RenderedSite: ...

    @property
    def gpu_cluster_arn(self) -> str: ...

    @property
    def cluster_id(self) -> str | None: ...

    @property
    def allowed_namespaces(self) -> tuple[str, ...]: ...

    @property
    def state_dir(self) -> Path | None: ...


RECOVERABLE_ATTEMPT_PHASES = frozenset(
    {"ROLLED_BACK", "ROLLBACK_STARTED", "ROLLBACK_FAILED", "COMPLETED"}
)
JOIN_RELEASE_STEPS = ("JOIN_STARTED", "RELEASE_STARTED", "JOINED")


def join_release_started(state: dict[str, Any]) -> bool:
    return any(step_done(state, step) for step in JOIN_RELEASE_STEPS)


def recovery_scope_sha256(site: RenderedSite) -> str:
    """Bind every site-level destination compensation can still mutate."""
    scope = {
        key: site.release_config.get(key)
        for key in (
            "site_name",
            "aws_region",
            "cpu_eks_arn",
            "cpu_hyperpod_cluster_name",
            "cpu_kubeconfig",
            "namespace",
            "nlb",
            "dns",
        )
    }
    return hashlib.sha256(
        json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _require_recovery_scope(
    request: JoinStateRequest, state_dir: Path, state: dict[str, Any]
) -> None:
    expected = state.get("source_site_recovery_sha256")
    if expected is None:
        evidence = state.get("evidence") or {}
        candidate_file = (evidence.get("CANDIDATE_READY") or {}).get("site_file")
        candidate = Path(
            str(
                candidate_file
                or state_dir
                / f"candidate-site-{int(state.get('attempt') or 1):03d}.yaml"
            )
        )
        if not candidate.is_file() or candidate.is_symlink():
            raise BootstrapError(
                "join retry lacks its original site recovery binding; "
                "legacy evidence requires reconciliation"
            )
        previous = load_site(candidate, repository_root=request.site.repository_root)
        expected = recovery_scope_sha256(previous)
    if expected != recovery_scope_sha256(request.site):
        raise BootstrapError("join-cluster recovery site identity drifted")


def load_join_state(
    request: JoinStateRequest,
) -> tuple[Path, Path, dict[str, Any]]:
    identity = hashlib.sha256(request.gpu_cluster_arn.encode()).hexdigest()[:12]
    state_dir = (
        request.state_dir or request.site.source.parent / "join-cluster" / identity
    ).expanduser()
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    state_dir.chmod(0o700)
    path = state_dir / "state.json"
    expected = {
        "site_id": request.site.release_config["site_name"],
        "gpu_cluster_arn": request.gpu_cluster_arn,
        "requested_cluster_id": request.cluster_id or "",
        "allowed_namespaces": sorted(request.allowed_namespaces),
    }
    if path.exists():
        value = cast(
            dict[str, Any],
            json.loads(path.read_text(encoding="utf-8")),
        )
        for key, item in expected.items():
            if value.get(key) != item:
                raise BootstrapError(f"join-cluster state conflicts on {key}")
        current_non_membership = site_non_membership_sha256(request.site.source)
        recorded_non_membership = str(
            value.get("source_site_non_membership_sha256") or ""
        )
        if (
            recorded_non_membership
            and recorded_non_membership != current_non_membership
        ):
            if value.get("phase") not in RECOVERABLE_ATTEMPT_PHASES:
                raise BootstrapError(
                    "join-cluster source site non-membership fields drifted"
                )
            _require_recovery_scope(request, state_dir, value)
        if not recorded_non_membership:
            value["source_site_non_membership_sha256"] = current_non_membership
            write_json_atomic(path, value)
        return state_dir, path, value
    value = {
        "schema_version": 1,
        **expected,
        "attempt": 1,
        "source_site_sha256": request.site.source_sha256,
        "source_site_recovery_sha256": recovery_scope_sha256(request.site),
        "source_site_non_membership_sha256": site_non_membership_sha256(
            request.site.source
        ),
        "phase": "STARTED",
        "completed_steps": [],
        "evidence": {},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(path, value)
    return state_dir, path, value


def step_done(state: dict[str, Any], step: str) -> bool:
    return step in set(state.get("completed_steps") or [])


def completed_state_is_current(
    request: JoinStateRequest,
    state: dict[str, Any],
) -> bool:
    discovered = (state.get("evidence") or {}).get("DISCOVERED") or {}
    target = discovered.get("target") or {}
    cluster_id = str(discovered.get("cluster_id") or "")
    eks_arn = str(target.get("eks_arn") or "")
    return any(
        item.get("cluster_id") == cluster_id and item.get("eks_cluster_arn") == eks_arn
        for item in request.site.release_config["clusters"]
    )


def note_join_failure(state: dict[str, Any], error: BaseException) -> None:
    """Retain the sanitized cause through rollback; a new attempt archives it."""
    completed_at = state.get("step_completed_at")
    completed = state.get("completed_steps")
    timestamps: dict[str, datetime] = {}
    if isinstance(completed_at, dict) and isinstance(completed, list):
        for step in completed:
            raw = completed_at.get(step) if isinstance(step, str) else None
            if not isinstance(raw, str):
                continue
            try:
                timestamp = datetime.fromisoformat(raw)
            except ValueError:
                continue
            if timestamp.tzinfo is not None:
                timestamps[step] = timestamp
    state["failure"] = {
        "error": f"{type(error).__name__}: {diagnostic_text(str(error))}",
        "after_step": (
            max(timestamps, key=timestamps.__getitem__) if timestamps else None
        ),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def reset_completed_state(
    request: JoinStateRequest,
    *,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
) -> None:
    previous_attempt = int(state.get("attempt") or 1)
    retained_token = state.get("retained_cluster_token")
    archive = state_dir / f"state.attempt-{previous_attempt:03d}.json"
    if not archive.exists():
        write_json_atomic(archive, dict(state))
    state.clear()
    state.update(
        {
            "schema_version": 1,
            "site_id": request.site.release_config["site_name"],
            "gpu_cluster_arn": request.gpu_cluster_arn,
            "requested_cluster_id": request.cluster_id or "",
            "allowed_namespaces": sorted(request.allowed_namespaces),
            "attempt": previous_attempt + 1,
            "source_site_sha256": request.site.source_sha256,
            "source_site_recovery_sha256": recovery_scope_sha256(request.site),
            "source_site_non_membership_sha256": site_non_membership_sha256(
                request.site.source
            ),
            "phase": "STARTED",
            "completed_steps": [],
            "evidence": {},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    if isinstance(retained_token, dict):
        state["retained_cluster_token"] = dict(retained_token)
    write_json_atomic(state_path, state)


def complete_step(
    path: Path,
    state: dict[str, Any],
    step: str,
    evidence: dict[str, Any] | None = None,
) -> None:
    completed = set(state.get("completed_steps") or [])
    completed.add(step)
    state["completed_steps"] = sorted(completed)
    state["phase"] = step
    now = datetime.now(timezone.utc).isoformat()
    state["updated_at"] = now
    # One timestamp per step so an operator can read where a join spent its time.
    state.setdefault("step_completed_at", {})[step] = now
    if evidence is not None:
        state.setdefault("evidence", {})[step] = evidence
    write_json_atomic(path, state)
