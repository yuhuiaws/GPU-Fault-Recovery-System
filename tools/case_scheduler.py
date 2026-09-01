from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
import threading
import time
from typing import Any, Callable, Mapping, Sequence


GLOBAL_RESOURCE = "acceptance-global"
LOCK_MODES = frozenset({"shared", "exclusive"})
FAILURE_SCOPES = frozenset({"case", "branch", "global"})
ENVIRONMENT_MODES = frozenset({"inherit", "isolated"})
RESULT_STATUSES = frozenset({"PASS", "FAIL", "BLOCKED", "NOT_RUN"})


@dataclass(frozen=True, order=True)
class ResourceLock:
    resource: str
    mode: str

    def __post_init__(self) -> None:
        if not self.resource:
            raise ValueError("resource lock name must not be empty")
        if self.mode not in LOCK_MODES:
            raise ValueError(f"unsupported resource lock mode: {self.mode}")

    def as_dict(self) -> dict[str, str]:
        return {"resource": self.resource, "mode": self.mode}


@dataclass(frozen=True)
class ExecutionPolicy:
    parallel_safe: bool
    locks: tuple[ResourceLock, ...] = ()
    depends_on: tuple[str, ...] = ()
    failure_scope: str = "branch"
    environment: str = "inherit"

    def __post_init__(self) -> None:
        if self.failure_scope not in FAILURE_SCOPES:
            raise ValueError(f"unsupported failure scope: {self.failure_scope}")
        if self.environment not in ENVIRONMENT_MODES:
            raise ValueError(f"unsupported execution environment: {self.environment}")

    def effective_locks(self) -> tuple[ResourceLock, ...]:
        if not self.parallel_safe:
            return (ResourceLock(GLOBAL_RESOURCE, "exclusive"),)
        combined = (ResourceLock(GLOBAL_RESOURCE, "shared"), *self.locks)
        modes: dict[str, str] = {}
        for lock in combined:
            existing = modes.get(lock.resource)
            if existing is not None and existing != lock.mode:
                raise ValueError(f"resource {lock.resource} has conflicting lock modes")
            modes[lock.resource] = lock.mode
        return tuple(
            sorted(
                (ResourceLock(resource, mode) for resource, mode in modes.items()),
                key=lambda item: item.resource,
            )
        )


@dataclass(frozen=True)
class _ScheduledCase:
    case: dict[str, Any]
    policy: ExecutionPolicy
    locks: tuple[ResourceLock, ...]


class ResourceLockManager:
    def __init__(self) -> None:
        self._readers: Counter[str] = Counter()
        self._writers: set[str] = set()

    def can_acquire(self, locks: Sequence[ResourceLock]) -> bool:
        for lock in locks:
            if lock.mode == "shared":
                if lock.resource in self._writers:
                    return False
                continue
            if lock.resource in self._writers or self._readers[lock.resource]:
                return False
        return True

    def acquire(self, locks: Sequence[ResourceLock]) -> None:
        if not self.can_acquire(locks):
            raise RuntimeError("attempted to acquire conflicting resource locks")
        for lock in locks:
            if lock.mode == "shared":
                self._readers[lock.resource] += 1
            else:
                self._writers.add(lock.resource)

    def release(self, locks: Sequence[ResourceLock]) -> None:
        for lock in locks:
            if lock.mode == "shared":
                self._readers[lock.resource] -= 1
                if self._readers[lock.resource] <= 0:
                    del self._readers[lock.resource]
            else:
                self._writers.remove(lock.resource)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _failure_result(case: Mapping[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        "id": case["id"],
        "title": case.get("title", case["id"]),
        "category": case.get("category"),
        "level": case.get("level"),
        "risk": case.get("risk"),
        "status": "FAIL",
        "output": f"{type(exc).__name__}: {exc}",
    }


def _validated_result(
    case: Mapping[str, Any],
    value: object,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return _failure_result(
            case,
            TypeError("executor result must be a mapping"),
        )
    result = dict(value)
    status = result.get("status")
    if status not in RESULT_STATUSES:
        return _failure_result(
            case,
            ValueError(f"executor returned unsupported status: {status!r}"),
        )
    return result


def _blocked_result(
    case: Mapping[str, Any],
    *,
    reason: str,
    queued_at: str,
    queued_monotonic: float,
    policy: ExecutionPolicy,
    monotonic: Callable[[], float],
    utc_now: Callable[[], str],
) -> dict[str, Any]:
    return {
        "id": case["id"],
        "title": case.get("title", case["id"]),
        "category": case.get("category"),
        "level": case.get("level"),
        "risk": case.get("risk"),
        "status": "BLOCKED",
        "reason": reason,
        "duration_seconds": 0.0,
        "scheduling": {
            "queued_at": queued_at,
            "started_at": None,
            "completed_at": utc_now(),
            "lock_wait_seconds": round(monotonic() - queued_monotonic, 3),
            "execution_seconds": 0.0,
            "worker": None,
            "locks": [lock.as_dict() for lock in policy.effective_locks()],
            "depends_on": list(policy.depends_on),
            "failure_scope": policy.failure_scope,
            "environment": policy.environment,
        },
    }


def _execute_scheduled_cases(
    scheduled: tuple[_ScheduledCase, ...],
    batch_key: str | None,
    *,
    execute: Callable[[dict[str, Any], ExecutionPolicy], dict[str, Any]],
    execute_batch: (
        Callable[
            [Sequence[dict[str, Any]], Sequence[ExecutionPolicy]],
            Mapping[str, dict[str, Any]],
        ]
        | None
    ),
    queued_at: Mapping[str, str],
    queued_monotonic: Mapping[str, float],
    collect_all: bool,
    monotonic: Callable[[], float],
    utc_now: Callable[[], str],
) -> list[dict[str, Any]]:
    started_at = utc_now()
    started = monotonic()
    try:
        if batch_key is None:
            item = scheduled[0]
            raw_results = {
                str(item.case["id"]): execute(item.case, item.policy),
            }
        else:
            assert execute_batch is not None
            raw_results = dict(
                execute_batch(
                    [item.case for item in scheduled],
                    [item.policy for item in scheduled],
                )
            )
    except Exception as exc:
        raw_results = {
            str(item.case["id"]): _failure_result(item.case, exc) for item in scheduled
        }
    completed_at = utc_now()
    expected = {str(item.case["id"]) for item in scheduled}
    if set(raw_results) != expected:
        error = ValueError(
            "batch executor result IDs do not match scheduled cases: "
            f"expected={sorted(expected)} actual={sorted(raw_results)}"
        )
        raw_results = {
            str(item.case["id"]): _failure_result(item.case, error)
            for item in scheduled
        }
    batch_id = (
        f"{batch_key}:{started_at}:{threading.current_thread().name}"
        if batch_key is not None
        else None
    )
    output = []
    for item in scheduled:
        case = item.case
        policy = item.policy
        case_id = str(case["id"])
        result = _validated_result(case, raw_results[case_id])
        result_id = result.get("id", case_id)
        if result_id != case_id:
            result = _failure_result(
                case,
                ValueError(f"executor returned result id {result_id!r} for {case_id}"),
            )
        result.setdefault("id", case_id)
        result.setdefault("title", case.get("title", case_id))
        scheduling = {
            "queued_at": queued_at[case_id],
            "started_at": started_at,
            "completed_at": completed_at,
            "lock_wait_seconds": round(
                started - queued_monotonic[case_id],
                3,
            ),
            "execution_seconds": round(monotonic() - started, 3),
            "worker": threading.current_thread().name,
            "locks": [lock.as_dict() for lock in policy.effective_locks()],
            "depends_on": list(policy.depends_on),
            "failure_scope": policy.failure_scope,
            "environment": policy.environment,
            "collect_all": collect_all,
        }
        if batch_key is not None:
            scheduling["batch"] = {
                "id": batch_id,
                "key": batch_key,
                "size": len(scheduled),
            }
        result["scheduling"] = scheduling
        output.append(result)
    return output


def _dependencies_ready(
    policy: ExecutionPolicy,
    *,
    results: Mapping[str, Mapping[str, Any]],
    selected_ids: set[str],
) -> bool:
    return all(
        dependency in results
        for dependency in policy.depends_on
        if dependency in selected_ids
    )


def _block_failed_dependencies(
    pending: list[dict[str, Any]],
    *,
    policies: Mapping[str, ExecutionPolicy],
    results: dict[str, dict[str, Any]],
    selected_ids: set[str],
    success_predicate: Callable[[Mapping[str, Any]], bool],
    queued_at: Mapping[str, str],
    queued_monotonic: Mapping[str, float],
    on_finish: Callable[[dict[str, Any], dict[str, Any]], None] | None,
    monotonic: Callable[[], float],
    utc_now: Callable[[], str],
) -> bool:
    progressed = False
    for case in list(pending):
        case_id = str(case["id"])
        policy = policies[case_id]
        selected_dependencies = [
            dependency for dependency in policy.depends_on if dependency in selected_ids
        ]
        failed_dependencies = [
            dependency
            for dependency in selected_dependencies
            if dependency in results and not success_predicate(results[dependency])
        ]
        if not failed_dependencies:
            continue
        result = _blocked_result(
            case,
            reason=(
                "blocked by failed prerequisite(s): " + ", ".join(failed_dependencies)
            ),
            queued_at=queued_at[case_id],
            queued_monotonic=queued_monotonic[case_id],
            policy=policy,
            monotonic=monotonic,
            utc_now=utc_now,
        )
        results[case_id] = result
        pending.remove(case)
        if on_finish is not None:
            on_finish(case, result)
        progressed = True
    return progressed


def _block_pending_for_global_failure(
    pending: list[dict[str, Any]],
    *,
    global_failure: str,
    policies: Mapping[str, ExecutionPolicy],
    results: dict[str, dict[str, Any]],
    queued_at: Mapping[str, str],
    queued_monotonic: Mapping[str, float],
    on_finish: Callable[[dict[str, Any], dict[str, Any]], None] | None,
    monotonic: Callable[[], float],
    utc_now: Callable[[], str],
) -> None:
    for case in list(pending):
        case_id = str(case["id"])
        result = _blocked_result(
            case,
            reason=f"blocked by global failure in {global_failure}",
            queued_at=queued_at[case_id],
            queued_monotonic=queued_monotonic[case_id],
            policy=policies[case_id],
            monotonic=monotonic,
            utc_now=utc_now,
        )
        results[case_id] = result
        pending.remove(case)
        if on_finish is not None:
            on_finish(case, result)


def _block_unresolved_pending(
    pending: list[dict[str, Any]],
    *,
    policies: Mapping[str, ExecutionPolicy],
    results: dict[str, dict[str, Any]],
    queued_at: Mapping[str, str],
    queued_monotonic: Mapping[str, float],
    on_finish: Callable[[dict[str, Any], dict[str, Any]], None] | None,
    monotonic: Callable[[], float],
    utc_now: Callable[[], str],
) -> None:
    unresolved = ", ".join(str(case["id"]) for case in pending)
    for case in list(pending):
        case_id = str(case["id"])
        result = _blocked_result(
            case,
            reason=(
                "dependency cycle or unsatisfied selected prerequisite: " + unresolved
            ),
            queued_at=queued_at[case_id],
            queued_monotonic=queued_monotonic[case_id],
            policy=policies[case_id],
            monotonic=monotonic,
            utc_now=utc_now,
        )
        results[case_id] = result
        pending.remove(case)
        if on_finish is not None:
            on_finish(case, result)


def _validated_success_predicate(
    *,
    max_workers: int,
    max_batch_size: int,
    batch_key_for: Callable[[dict[str, Any], ExecutionPolicy], str | None] | None,
    execute_batch: (
        Callable[
            [Sequence[dict[str, Any]], Sequence[ExecutionPolicy]],
            Mapping[str, dict[str, Any]],
        ]
        | None
    ),
    is_successful: Callable[[Mapping[str, Any]], bool] | None,
) -> Callable[[Mapping[str, Any]], bool]:
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be at least 1")
    if (batch_key_for is None) != (execute_batch is None):
        raise ValueError("batch_key_for and execute_batch must be configured together")
    if is_successful is not None:
        return is_successful
    return lambda result: result.get("status") == "PASS"


def _collect_completed(
    active: dict[Future[list[dict[str, Any]]], tuple[_ScheduledCase, ...]],
    *,
    active_case_count: int,
    locks: ResourceLockManager,
    results: dict[str, dict[str, Any]],
    collect_all: bool,
    on_finish: Callable[[dict[str, Any], dict[str, Any]], None] | None,
) -> tuple[int, str | None]:
    global_failure = None
    completed, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
    for future in completed:
        scheduled = active.pop(future)
        active_case_count -= len(scheduled)
        for item in scheduled:
            locks.release(item.locks)
        batch_results = future.result()
        for item, result in zip(scheduled, batch_results):
            case = item.case
            policy = item.policy
            case_id = str(case["id"])
            results[case_id] = result
            if on_finish is not None:
                on_finish(case, result)
            if (
                result.get("status") == "FAIL"
                and policy.failure_scope == "global"
                and not collect_all
            ):
                global_failure = case_id
    return active_case_count, global_failure


def run_scheduled_cases(
    cases: Sequence[dict[str, Any]],
    *,
    policy_for: Callable[[dict[str, Any]], ExecutionPolicy],
    execute: Callable[[dict[str, Any], ExecutionPolicy], dict[str, Any]],
    batch_key_for: (
        Callable[[dict[str, Any], ExecutionPolicy], str | None] | None
    ) = None,
    execute_batch: (
        Callable[
            [Sequence[dict[str, Any]], Sequence[ExecutionPolicy]],
            Mapping[str, dict[str, Any]],
        ]
        | None
    ) = None,
    max_batch_size: int = 1,
    max_workers: int = 1,
    collect_all: bool = False,
    is_successful: Callable[[Mapping[str, Any]], bool] | None = None,
    on_start: Callable[[dict[str, Any], ExecutionPolicy], None] | None = None,
    on_finish: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    utc_now: Callable[[], str] = _utc_now,
) -> list[dict[str, Any]]:
    success_predicate = _validated_success_predicate(
        max_workers=max_workers,
        max_batch_size=max_batch_size,
        batch_key_for=batch_key_for,
        execute_batch=execute_batch,
        is_successful=is_successful,
    )
    case_ids = [str(case["id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("scheduled case IDs must be unique")

    selected_ids = set(case_ids)
    policies = {case_id: policy_for(case) for case_id, case in zip(case_ids, cases)}
    pending = list(cases)
    queued_at = {case_id: utc_now() for case_id in case_ids}
    queued_monotonic = {case_id: monotonic() for case_id in case_ids}
    results: dict[str, dict[str, Any]] = {}
    active: dict[
        Future[list[dict[str, Any]]],
        tuple[_ScheduledCase, ...],
    ] = {}
    active_case_count = 0
    locks = ResourceLockManager()
    global_failure: str | None = None

    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="fault-case",
    ) as pool:
        while pending or active:
            progressed = False

            if global_failure is not None:
                _block_pending_for_global_failure(
                    pending,
                    global_failure=global_failure,
                    policies=policies,
                    results=results,
                    queued_at=queued_at,
                    queued_monotonic=queued_monotonic,
                    on_finish=on_finish,
                    monotonic=monotonic,
                    utc_now=utc_now,
                )
                progressed = True

            progressed = (
                _block_failed_dependencies(
                    pending,
                    policies=policies,
                    results=results,
                    selected_ids=selected_ids,
                    success_predicate=success_predicate,
                    queued_at=queued_at,
                    queued_monotonic=queued_monotonic,
                    on_finish=on_finish,
                    monotonic=monotonic,
                    utc_now=utc_now,
                )
                or progressed
            )

            while active_case_count < max_workers:
                candidate = None
                candidate_policy = None
                candidate_locks = None
                for case in pending:
                    case_id = str(case["id"])
                    policy = policies[case_id]
                    if not _dependencies_ready(
                        policy,
                        results=results,
                        selected_ids=selected_ids,
                    ):
                        continue
                    effective_locks = policy.effective_locks()
                    if not locks.can_acquire(effective_locks):
                        continue
                    candidate = case
                    candidate_policy = policy
                    candidate_locks = effective_locks
                    break
                if candidate is None:
                    break
                assert candidate_policy is not None
                assert candidate_locks is not None
                pending.remove(candidate)
                locks.acquire(candidate_locks)
                scheduled = [
                    _ScheduledCase(
                        case=candidate,
                        policy=candidate_policy,
                        locks=candidate_locks,
                    )
                ]
                batch_key = (
                    batch_key_for(candidate, candidate_policy)
                    if batch_key_for is not None
                    else None
                )
                if batch_key is not None:
                    batch_capacity = min(
                        max_batch_size,
                        max_workers - active_case_count,
                    )
                    for case in list(pending):
                        if len(scheduled) >= batch_capacity:
                            break
                        case_id = str(case["id"])
                        policy = policies[case_id]
                        if batch_key_for(case, policy) != batch_key:
                            continue
                        if not _dependencies_ready(
                            policy,
                            results=results,
                            selected_ids=selected_ids,
                        ):
                            continue
                        effective_locks = policy.effective_locks()
                        if not locks.can_acquire(effective_locks):
                            continue
                        pending.remove(case)
                        locks.acquire(effective_locks)
                        scheduled.append(
                            _ScheduledCase(
                                case=case,
                                policy=policy,
                                locks=effective_locks,
                            )
                        )
                if on_start is not None:
                    for item in scheduled:
                        on_start(item.case, item.policy)
                future = pool.submit(
                    _execute_scheduled_cases,
                    tuple(scheduled),
                    batch_key,
                    execute=execute,
                    execute_batch=execute_batch,
                    queued_at=queued_at,
                    queued_monotonic=queued_monotonic,
                    collect_all=collect_all,
                    monotonic=monotonic,
                    utc_now=utc_now,
                )
                active[future] = tuple(scheduled)
                active_case_count += len(scheduled)
                progressed = True

            if active:
                active_case_count, completed_global_failure = _collect_completed(
                    active,
                    active_case_count=active_case_count,
                    locks=locks,
                    results=results,
                    collect_all=collect_all,
                    on_finish=on_finish,
                )
                if completed_global_failure is not None:
                    global_failure = completed_global_failure
                continue

            if pending and not progressed:
                _block_unresolved_pending(
                    pending,
                    policies=policies,
                    results=results,
                    queued_at=queued_at,
                    queued_monotonic=queued_monotonic,
                    on_finish=on_finish,
                    monotonic=monotonic,
                    utc_now=utc_now,
                )

    return [results[case_id] for case_id in case_ids]
