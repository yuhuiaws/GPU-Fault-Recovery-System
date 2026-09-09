from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

from gpu_fault.admin import cluster_join as join
from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
)
from gpu_fault.admin.cluster_join import (
    JoinAttempt,
    JoinClusterRequest,
    JoinExecution,
)
from gpu_fault.admin.cluster_join_evidence import (
    JoinVerificationExpired,
    build_verified_membership_evidence,
    clear_verified_step,
    membership_runtime_snapshot,
    verification_is_stale,
)
from gpu_fault.admin.cluster_join_state import (
    complete_step,
    completed_state_is_current,
    load_join_state,
    reset_completed_state,
    step_done,
)
from gpu_fault.admin.membership_lock import (
    membership_operation_lock,
    reload_site_for_mutation,
)
from gpu_fault.admin.site import RenderedSite

# Read-only discovery and the per-cluster prerequisite work (IAM role, network
# allow-list, node keys) fan out this wide. How many clusters *roll* at once is
# not decided here: the release engine owns cluster parallelism through
# ``spec.release.upgradeMaxParallelClusters`` and its own node wave caps, and
# the batch reads that value (see ``deploy_concurrency``).
PREPARE_PHASE_WORKERS = 4
REGISTRY_SNAPSHOT_BEFORE = "installation-resources-before.json"


@dataclass
class BatchJoinContext:
    site: RenderedSite
    batch_id: str
    candidate_dir: Path
    runner_factory: Callable[[], CommandRunner]
    results: list[dict[str, Any]]
    failures: dict[str, str]
    network_baseline: list[dict[str, Any]] | None = None


def deploy_concurrency(site: RenderedSite) -> int:
    """Clusters rolled at once: the site's ``upgradeMaxParallelClusters``."""

    configured = site.release_config["release"].get("upgrade_max_parallel_clusters")
    return max(1, int(configured if configured is not None else 1))


def _execution(attempt: JoinAttempt) -> JoinExecution:
    if attempt.execution is None:
        raise BootstrapError("batch join attempt has no prepared execution")
    return attempt.execution


def _initialize_context(
    requests: tuple[JoinClusterRequest, ...],
    *,
    runner_factory: Callable[[], CommandRunner],
) -> tuple[BatchJoinContext, list[JoinAttempt]]:
    site = requests[0].site
    if any(
        request.site.source.resolve() != site.source.resolve() for request in requests
    ):
        raise BootstrapError("batch join requests must use the same managed site")
    arns = [request.gpu_cluster_arn for request in requests]
    if len(arns) != len(set(arns)):
        raise BootstrapError("batch join contains duplicate GPU cluster ARNs")
    context = BatchJoinContext(
        site=site,
        # Correlates the per-cluster VERIFIED evidence of one batch; the
        # per-cluster ``join-cluster/<arn-hash>/state.json`` is the record.
        batch_id=hashlib.sha256("\n".join(sorted(arns)).encode()).hexdigest()[:16],
        candidate_dir=site.source.parent / "join-cluster",
        runner_factory=runner_factory,
        results=[],
        failures={},
    )
    attempts = []
    for request in requests:
        attempt_state_dir, state_path, state = load_join_state(request)
        attempt = JoinAttempt(request, attempt_state_dir, state_path, state)
        if state.get("phase") == "COMPLETED":
            if completed_state_is_current(request, state):
                context.results.append(
                    {
                        "site_id": site.release_config["site_name"],
                        "cluster_id": state["evidence"]["DISCOVERED"]["cluster_id"],
                        "phase": "ALREADY_MANAGED",
                        "state_file": str(state_path),
                    }
                )
                continue
            reset_completed_state(
                request,
                state_dir=attempt_state_dir,
                state_path=state_path,
                state=state,
            )
        attempts.append(attempt)
    return context, attempts


def _run_readonly_tasks(
    context: BatchJoinContext,
    attempts: list[JoinAttempt],
) -> tuple[
    dict[str, tuple[ClusterIdentity, str, dict[str, Any] | None]],
    Path | None,
]:
    undiscovered = [
        attempt for attempt in attempts if not step_done(attempt.state, "DISCOVERED")
    ]
    # One registry export serves the whole batch; it is exported into the first
    # undiscovered cluster's record and copied into the others afterwards.
    registry_path = (
        undiscovered[0].state_dir / REGISTRY_SNAPSHOT_BEFORE if undiscovered else None
    )
    tasks: dict[Any, tuple[str, JoinAttempt | None]] = {}
    discoveries: dict[str, tuple[ClusterIdentity, str, dict[str, Any] | None]] = {}
    failures = []
    count = int(bool(undiscovered)) + len(undiscovered) + 1
    with ThreadPoolExecutor(
        max_workers=min(PREPARE_PHASE_WORKERS, max(1, count))
    ) as executor:
        if undiscovered and registry_path is not None:
            tasks[
                executor.submit(join._export_registry, context.site, registry_path)
            ] = ("registry", None)
            for attempt in undiscovered:
                tasks[
                    executor.submit(
                        join._discover_join_target,
                        attempt.request,
                        context.runner_factory(),
                    )
                ] = ("discovery", attempt)
        tasks[
            executor.submit(
                join._existing_cluster_networks,
                context.runner_factory(),
                context.site,
            )
        ] = ("networks", None)
        for future in as_completed(tasks):
            kind, task_attempt = tasks[future]
            try:
                value = future.result()
            except Exception as exc:
                if kind == "discovery" and task_attempt is not None:
                    context.failures[task_attempt.request.gpu_cluster_arn] = str(exc)
                    task_attempt.state["phase"] = "FAILED"
                    task_attempt.state["updated_at"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    write_json_atomic(task_attempt.state_path, task_attempt.state)
                else:
                    failures.append(f"{kind}: {exc}")
                continue
            if kind == "discovery" and task_attempt is not None:
                discoveries[task_attempt.request.gpu_cluster_arn] = cast(
                    tuple[ClusterIdentity, str, dict[str, Any] | None],
                    value,
                )
            elif kind == "networks":
                context.network_baseline = cast(list[dict[str, Any]], value)
    if failures:
        raise BootstrapError(
            "batch join read-only preparation failed: " + "; ".join(sorted(failures))
        )
    return discoveries, registry_path


def _registry_snapshot_for(attempt: JoinAttempt, exported: Path | None) -> Path:
    target = attempt.state_dir / REGISTRY_SNAPSHOT_BEFORE
    if exported is not None and exported != target and exported.is_file():
        shutil.copyfile(exported, target)
        target.chmod(0o600)
    return target


def _checkpoint_readonly_results(
    context: BatchJoinContext,
    attempts: list[JoinAttempt],
    discoveries: dict[str, tuple[ClusterIdentity, str, dict[str, Any] | None]],
    registry_path: Path | None,
) -> list[JoinAttempt]:
    active = []
    identities: list[tuple[str, str, str, str, str]] = []
    for attempt in attempts:
        arn = attempt.request.gpu_cluster_arn
        if arn in context.failures:
            continue
        if not step_done(attempt.state, "PRECHECKED"):
            complete_step(
                attempt.state_path,
                attempt.state,
                "PRECHECKED",
                {
                    "site_sha256": context.site.source_sha256,
                    "batch_id": context.batch_id,
                },
            )
        if not step_done(attempt.state, "DISCOVERED"):
            target, cluster_id, existing = discoveries[arn]
            if existing is not None:
                complete_step(
                    attempt.state_path,
                    attempt.state,
                    "DISCOVERED",
                    {
                        "target": asdict(target),
                        "cluster_id": str(existing["cluster_id"]),
                        "already_managed": True,
                    },
                )
                attempt.state["phase"] = "COMPLETED"
                attempt.state["updated_at"] = datetime.now(timezone.utc).isoformat()
                write_json_atomic(attempt.state_path, attempt.state)
                context.results.append(
                    {
                        "site_id": context.site.release_config["site_name"],
                        "cluster_id": existing["cluster_id"],
                        "phase": "ALREADY_MANAGED",
                        "state_file": str(attempt.state_path),
                        "site_file": str(context.site.source),
                    }
                )
                continue
            complete_step(
                attempt.state_path,
                attempt.state,
                "DISCOVERED",
                {
                    "target": asdict(target),
                    "cluster_id": cluster_id,
                    "registry_snapshot": str(
                        _registry_snapshot_for(attempt, registry_path)
                    ),
                },
            )
        discovery = attempt.state["evidence"]["DISCOVERED"]
        target = join._identity(discovery["target"])
        identities.append(
            (
                str(discovery["cluster_id"]),
                target.eks_arn,
                target.hyperpod_name,
                target.context,
                target.vpc_id,
            )
        )
        active.append(attempt)
    for index, identity in enumerate(identities):
        for other in identities[index + 1 :]:
            if any(left == right for left, right in zip(identity, other, strict=True)):
                raise BootstrapError(
                    "batch join discovered conflicting cluster identities"
                )
    return active


def _prepare_executions(
    context: BatchJoinContext,
    attempts: list[JoinAttempt],
) -> list[JoinAttempt]:
    def prepare(attempt: JoinAttempt) -> JoinExecution | dict[str, Any]:
        return join._prepare_execution(
            attempt.request,
            runner=context.runner_factory(),
            state_dir=attempt.state_dir,
            state_path=attempt.state_path,
            state=attempt.state,
            candidate_preflight=False,
            network_baseline=context.network_baseline,
        )

    prepared_attempts = []
    with ThreadPoolExecutor(
        max_workers=min(PREPARE_PHASE_WORKERS, len(attempts))
    ) as executor:
        futures = {executor.submit(prepare, attempt): attempt for attempt in attempts}
        for future in as_completed(futures):
            attempt = futures[future]
            try:
                prepared = future.result()
            except Exception as exc:
                _record_failure(context, attempt, None, exc)
                continue
            if isinstance(prepared, dict):
                context.results.append(prepared)
                continue
            attempt.execution = prepared
            prepared_attempts.append(attempt)
    return prepared_attempts


def _record_failure(
    context: BatchJoinContext,
    attempt: JoinAttempt,
    execution: JoinExecution | None,
    error: Exception,
) -> None:
    try:
        join._record_join_failure(attempt, execution)
    except Exception as rollback_error:
        context.failures[attempt.request.gpu_cluster_arn] = (
            f"{error}; rollback failed: {rollback_error}"
        )
    else:
        context.failures[attempt.request.gpu_cluster_arn] = str(error)


def _preflight_candidate(
    context: BatchJoinContext,
    attempts: list[JoinAttempt],
) -> list[JoinAttempt]:
    if not attempts:
        return []
    candidate = join._write_batch_candidate_site(
        context.site,
        [_execution(item) for item in attempts],
        state_dir=context.candidate_dir,
    )
    try:
        join._run_rollout(candidate, "preflight")
    except Exception as exc:
        for attempt in attempts:
            _record_failure(context, attempt, attempt.execution, exc)
        return []
    for attempt in attempts:
        attempt.execution = replace(
            _execution(attempt),
            candidate=candidate,
        )
    return attempts


def _deploy_clusters(
    context: BatchJoinContext,
    attempts: list[JoinAttempt],
) -> list[JoinAttempt]:
    deployed: list[JoinAttempt] = []
    if not attempts:
        return deployed
    with ThreadPoolExecutor(
        max_workers=min(deploy_concurrency(context.site), len(attempts))
    ) as executor:
        futures = {
            executor.submit(
                join._deploy_cluster,
                execution=_execution(attempt),
                state_path=attempt.state_path,
                state=attempt.state,
            ): attempt
            for attempt in attempts
        }
        for future in as_completed(futures):
            attempt = futures[future]
            try:
                future.result()
            except Exception as exc:
                _record_failure(context, attempt, attempt.execution, exc)
                continue
            deployed.append(attempt)
    return deployed


def _verify_deployed(
    context: BatchJoinContext,
    attempts: list[JoinAttempt],
) -> list[JoinAttempt]:
    if not attempts:
        return []
    for attempt in attempts:
        verified = (attempt.state.get("evidence") or {}).get("VERIFIED")
        if step_done(attempt.state, "VERIFIED") and verification_is_stale(
            verified if isinstance(verified, dict) else {}
        ):
            clear_verified_step(attempt.state_path, attempt.state)
    candidate = join._write_batch_candidate_site(
        context.site,
        [_execution(item) for item in attempts],
        state_dir=context.candidate_dir,
    )
    before = membership_runtime_snapshot(candidate)
    try:
        join._run_rollout(candidate, "verify")
    except Exception as exc:
        for attempt in attempts:
            _record_failure(context, attempt, attempt.execution, exc)
        return []
    after = membership_runtime_snapshot(candidate)
    verified_at = datetime.now(timezone.utc)
    candidate_cluster_ids = [
        str(item["cluster_id"]) for item in candidate.release_config["clusters"]
    ]
    for attempt in attempts:
        attempt.execution = replace(
            _execution(attempt),
            candidate=candidate,
        )
        if not step_done(attempt.state, "VERIFIED"):
            complete_step(
                attempt.state_path,
                attempt.state,
                "VERIFIED",
                build_verified_membership_evidence(
                    before,
                    after,
                    candidate_site_sha256=candidate.source_sha256,
                    source_site_sha256=str(attempt.state["source_site_sha256"]),
                    source_site_non_membership_sha256=str(
                        attempt.state["source_site_non_membership_sha256"]
                    ),
                    candidate_cluster_ids=candidate_cluster_ids,
                    cluster_id=_execution(attempt).cluster_id,
                    batch_id=context.batch_id,
                    verified_at=verified_at,
                ),
            )
    return attempts


def _commit_clusters(
    context: BatchJoinContext,
    attempts: list[JoinAttempt],
) -> None:
    def ordered(items: list[JoinAttempt]) -> list[JoinAttempt]:
        return sorted(items, key=lambda item: _execution(item).cluster_id)

    pending = ordered(attempts)
    reverified: set[str] = set()
    while pending:
        attempt = pending.pop(0)
        execution = _execution(attempt)
        try:
            join._activate_and_commit(
                attempt.request,
                execution=execution,
                state_dir=attempt.state_dir,
                state_path=attempt.state_path,
                state=attempt.state,
            )
        except JoinVerificationExpired as exc:
            # Earlier commits of this batch used up the window. The clusters
            # still waiting share one candidate file, so all of them are
            # re-verified together; one re-verify per cluster, then it is a
            # failure like any other.
            remaining = [attempt, *pending]
            stale = [
                item for item in remaining if item.request.gpu_cluster_arn in reverified
            ]
            for item in stale:
                _record_failure(context, item, item.execution, exc)
            remaining = [item for item in remaining if item not in stale]
            for item in remaining:
                reverified.add(item.request.gpu_cluster_arn)
                clear_verified_step(item.state_path, item.state)
            pending = ordered(_verify_deployed(context, remaining))
            continue
        except Exception as exc:
            _record_failure(context, attempt, execution, exc)
            continue
        context.results.append(join._complete_join(attempt, execution))


def _finish(context: BatchJoinContext) -> dict[str, Any]:
    summary = {
        "schema_version": 2,
        "phase": "PARTIAL" if context.failures else "COMPLETED",
        "batch_id": context.batch_id,
        "joined": sorted(
            str(item["cluster_id"])
            for item in context.results
            if item.get("phase") == "COMPLETED"
        ),
        "already_managed": sorted(
            str(item["cluster_id"])
            for item in context.results
            if item.get("phase") == "ALREADY_MANAGED"
        ),
        "failed": dict(sorted(context.failures.items())),
        "deploy_concurrency": deploy_concurrency(context.site),
        "clusters": sorted(
            context.results, key=lambda item: str(item.get("cluster_id") or "")
        ),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if context.failures:
        raise BootstrapError(
            "batch join completed with failed clusters: "
            + ", ".join(sorted(context.failures))
            + "; summary: "
            + json.dumps(summary, sort_keys=True)
        )
    return summary


def _join_clusters_locked(
    requests: tuple[JoinClusterRequest, ...],
    *,
    runner_factory: Callable[[], CommandRunner] = CommandRunner,
) -> dict[str, Any]:
    if not requests:
        return {"phase": "COMPLETED", "joined": [], "already_managed": []}
    context, attempts = _initialize_context(requests, runner_factory=runner_factory)
    if not attempts:
        return _finish(context)
    discoveries, registry_path = _run_readonly_tasks(context, attempts)
    attempts = _checkpoint_readonly_results(
        context,
        attempts,
        discoveries,
        registry_path,
    )
    if not attempts:
        return _finish(context)
    prepared = _prepare_executions(context, attempts)
    prepared = _preflight_candidate(context, prepared)
    deployed = _deploy_clusters(context, prepared)
    verified = _verify_deployed(context, deployed)
    _commit_clusters(context, verified)
    return _finish(context)


def join_clusters(
    requests: tuple[JoinClusterRequest, ...],
    *,
    runner_factory: Callable[[], CommandRunner] = CommandRunner,
) -> dict[str, Any]:
    if not requests:
        return {"phase": "COMPLETED", "joined": [], "already_managed": []}
    with membership_operation_lock(requests[0].site):
        current = reload_site_for_mutation(requests[0].site)
        return _join_clusters_locked(
            tuple(replace(request, site=current) for request in requests),
            runner_factory=runner_factory,
        )
