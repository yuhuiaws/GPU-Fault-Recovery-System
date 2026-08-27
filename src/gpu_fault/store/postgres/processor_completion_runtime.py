from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any, TypeVar


CompletionItem = TypeVar("CompletionItem")


def configure_processor_completion_runtime(
    store: Any,
    pool_max_size: int,
) -> None:
    concurrency = int(
        os.getenv(
            "GPU_FAULT_PROCESSOR_COMPLETION_CLUSTER_CONCURRENCY",
            "1",
        )
    )
    if not 1 <= concurrency <= min(8, pool_max_size):
        store._pool.close()
        raise ValueError(
            "GPU_FAULT_PROCESSOR_COMPLETION_CLUSTER_CONCURRENCY "
            "must be between 1 and min(8, PostgreSQL pool max size)"
        )
    if concurrency == 1:
        executor = None
    else:
        executor = ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="gpu-fault-completion-cluster",
        )
    store.processor_completion_cluster_concurrency = concurrency
    store._processor_completion_executor = executor


def complete_cluster_groups(
    groups: dict[str, list[CompletionItem]],
    *,
    concurrency: int,
    executor: Executor | None,
    complete: Callable[[list[CompletionItem]], dict[str, object]],
) -> dict[str, object]:
    completed: dict[str, object] = {}
    failure: Exception | None = None
    scopes = sorted(groups)
    if concurrency <= 1 or executor is None:
        for scope in scopes:
            try:
                completed.update(complete(groups[scope]))
            except Exception as exc:
                if failure is None:
                    failure = exc
    else:
        futures = {scope: executor.submit(complete, groups[scope]) for scope in scopes}
        for scope in scopes:
            try:
                completed.update(futures[scope].result())
            except Exception as exc:
                if failure is None:
                    failure = exc
    if failure is not None:
        raise failure
    return completed
