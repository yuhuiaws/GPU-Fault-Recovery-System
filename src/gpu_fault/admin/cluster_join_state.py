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
# The two steps a join records before it mutates anything: the baseline
# precheck and the discovery of the target (reads of AWS and the site). The
# driver records them through these names, so the set below is derived from
# the recorder rather than spelled a second time. Every later step -- local
# inputs, node keys, prerequisites, the release -- changes something.
PRECHECKED_STEP = "PRECHECKED"
DISCOVERED_STEP = "DISCOVERED"
PRE_MUTATION_STEPS = frozenset({PRECHECKED_STEP, DISCOVERED_STEP})
# The phases a journal can show after stopping before its first mutation: the
# fresh journal, a refusal, a plain failure, or one of the pre-mutation steps
# itself. Any other phase -- an in-flight step, the rollback machinery's
# phases, or ``SUPERVISION_LOST`` (an attempt that cannot be proven stopped) --
# is never superseded automatically, whatever its steps say.
UNMUTATED_ATTEMPT_PHASES = (
    frozenset({"STARTED", "BLOCKED_IDENTITY", "FAILED"}) | PRE_MUTATION_STEPS
)


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
            if attempt_never_mutated(value):
                # A refusal that changed nothing (an orphaned token, a blocked
                # identity) must not pin the site's release fields: the deploy
                # that fixes the refusal moves them, and the retry would be
                # refused forever. The attempt is archived and a fresh one
                # binds to the site as it stands now.
                return (
                    state_dir,
                    path,
                    supersede_unmutated_attempt(
                        request, state_dir=state_dir, state_path=path, state=value
                    ),
                )
            if value.get("phase") not in RECOVERABLE_ATTEMPT_PHASES:
                raise BootstrapError(
                    "join-cluster source site non-membership fields drifted"
                )
            _require_recovery_scope(request, state_dir, value)
        if not recorded_non_membership:
            value["source_site_non_membership_sha256"] = current_non_membership
            write_json_atomic(path, value)
        return state_dir, path, value
    value = _fresh_attempt_state(request, attempt=1)
    write_json_atomic(path, value)
    return state_dir, path, value


def _fresh_attempt_state(request: JoinStateRequest, *, attempt: int) -> dict[str, Any]:
    """A new attempt's journal, bound to the site as it stands now."""

    return {
        "schema_version": 1,
        "site_id": request.site.release_config["site_name"],
        "gpu_cluster_arn": request.gpu_cluster_arn,
        "requested_cluster_id": request.cluster_id or "",
        "allowed_namespaces": sorted(request.allowed_namespaces),
        "attempt": attempt,
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


def attempt_never_mutated(state: dict[str, Any]) -> bool:
    """Whether the recorded attempt stopped before it changed anything.

    True for a refusal or failure whose phase is one of
    ``UNMUTATED_ATTEMPT_PHASES`` and whose completed steps and evidence lie
    within ``PRE_MUTATION_STEPS`` (the precheck and the target discovery). The
    phase is judged as well as the steps: an in-flight step phase, a phase the
    rollback machinery owns (``RECOVERABLE_ATTEMPT_PHASES``) or an attempt whose
    command cannot be proven stopped (``SUPERVISION_LOST``) is never treated as
    unmutated, whatever its steps say.
    """

    phase = str(state.get("phase") or "")
    if phase not in UNMUTATED_ATTEMPT_PHASES:
        return False
    completed = {str(step) for step in state.get("completed_steps") or []}
    evidence = state.get("evidence") or {}
    recorded = {str(key) for key in evidence} if isinstance(evidence, dict) else set()
    return completed <= PRE_MUTATION_STEPS and recorded <= PRE_MUTATION_STEPS


def supersede_unmutated_attempt(
    request: JoinStateRequest,
    *,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Archive a refused, never-mutated attempt and start one bound to the current site.

    The archive keeps the existing ``state.attempt-NNN.json`` naming and the
    attempt's own phase, plus ``superseded_by_attempt``; the fresh journal
    records the supersession under ``superseded_attempts`` and carries a
    retained or reclaimed token binding forward, since the binding was proven
    against the cluster, not against the site's release fields.
    """

    if not attempt_never_mutated(state):
        raise BootstrapError(
            "join-cluster attempt mutated the site and cannot be superseded"
        )
    previous_attempt = int(state.get("attempt") or 1)
    next_attempt = previous_attempt + 1
    archive = state_dir / f"state.attempt-{previous_attempt:03d}.json"
    if archive.exists():
        raise BootstrapError(f"join-cluster attempt archive already exists: {archive}")
    now = datetime.now(timezone.utc).isoformat()
    reason = (
        "the site's non-membership fields changed after an attempt that was "
        "refused before it mutated anything"
    )
    write_json_atomic(
        archive,
        {
            **state,
            "superseded_by_attempt": next_attempt,
            "superseded_at": now,
            "superseded_reason": reason,
        },
    )
    supersessions = [
        dict(item)
        for item in state.get("superseded_attempts") or []
        if isinstance(item, dict)
    ]
    supersessions.append(
        {
            "attempt": previous_attempt,
            "phase": str(state.get("phase") or ""),
            "completed_steps": sorted(
                str(step) for step in state.get("completed_steps") or []
            ),
            "archive": str(archive),
            "reason": reason,
            "superseded_at": now,
        }
    )
    carried = {
        key: dict(state[key])
        for key in ("retained_cluster_token", "reclaimed_cluster_token")
        if isinstance(state.get(key), dict)
    }
    state.clear()
    state.update(_fresh_attempt_state(request, attempt=next_attempt))
    state["superseded_attempts"] = supersessions
    state.update(carried)
    write_json_atomic(state_path, state)
    return state


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
    state.update(_fresh_attempt_state(request, attempt=previous_attempt + 1))
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
