from __future__ import annotations

from typing import Any

import pytest

from tools.case_scheduler import (
    ExecutionPolicy,
    ResourceLock,
    ResourceLockManager,
    run_scheduled_cases,
)


def case(name: str) -> dict[str, Any]:
    return {"id": name, "title": name, "risk": "non-destructive"}


def passed(item: dict[str, Any], _policy: ExecutionPolicy) -> dict[str, Any]:
    return {"id": item["id"], "status": "PASS"}


@pytest.mark.parametrize("resource,mode", [("", "shared"), ("example", "invalid")])
def test_lock_definition_rejects_ambiguous_names_or_modes(
    resource: str, mode: str
) -> None:
    with pytest.raises(ValueError, match="resource lock"):
        ResourceLock(resource, mode)


@pytest.mark.parametrize(
    "options,problem",
    [
        ({"failure_scope": "unknown"}, "failure scope"),
        ({"environment": "unknown"}, "execution environment"),
    ],
)
def test_execution_policy_rejects_unknown_safety_modes(
    options: dict[str, Any], problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        ExecutionPolicy(parallel_safe=True, **options)


def test_policy_cannot_downgrade_or_mix_a_resource_lock() -> None:
    policy = ExecutionPolicy(
        parallel_safe=True,
        locks=(
            ResourceLock("shared-db", "shared"),
            ResourceLock("shared-db", "exclusive"),
        ),
    )
    with pytest.raises(ValueError, match="conflicting lock modes"):
        policy.effective_locks()


def test_lock_manager_refuses_conflicts_and_releases_only_after_all_readers() -> None:
    manager = ResourceLockManager()
    read = (ResourceLock("example", "shared"),)
    write = (ResourceLock("example", "exclusive"),)
    manager.acquire(read)
    manager.acquire(read)
    with pytest.raises(RuntimeError, match="conflicting resource locks"):
        manager.acquire(write)
    manager.release(read)
    assert manager.can_acquire(write) is False
    manager.release(read)
    manager.acquire(write)
    assert manager.can_acquire(read) is False
    manager.release(write)
    assert manager.can_acquire(read) is True


@pytest.mark.parametrize(
    "options,problem",
    [({"max_workers": 0}, "max_workers"), ({"max_batch_size": 0}, "max_batch_size")],
)
def test_invalid_scheduler_budget_never_calls_policy_or_executor(
    options: dict[str, Any], problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        run_scheduled_cases(
            [case("a")],
            policy_for=lambda _case: pytest.fail(
                "invalid budget reached policy lookup"
            ),
            execute=lambda *_args: pytest.fail("invalid budget started execution"),
            **options,
        )


def test_duplicate_case_identity_is_refused_before_execution() -> None:
    with pytest.raises(ValueError, match="case IDs must be unique"):
        run_scheduled_cases(
            [case("a"), case("a")],
            policy_for=lambda _case: pytest.fail(
                "duplicate identity reached policy lookup"
            ),
            execute=passed,
        )


@pytest.mark.parametrize("callback", [False, True])
def test_dependency_cycles_produce_complete_blocked_results_without_execution(
    callback: bool,
) -> None:
    finished: list[str] = []
    results = run_scheduled_cases(
        [case("a"), case("b")],
        policy_for=lambda item: ExecutionPolicy(
            parallel_safe=True, depends_on=("b" if item["id"] == "a" else "a",)
        ),
        execute=lambda *_args: pytest.fail("cyclic dependency was executed"),
        on_finish=(lambda item, _result: finished.append(item["id"]))
        if callback
        else None,
    )
    assert [item["id"] for item in results] == ["a", "b"]
    assert [item["status"] for item in results] == ["BLOCKED", "BLOCKED"]
    assert all("dependency cycle" in item["reason"] for item in results), (
        "unresolved dependencies lost their diagnostic"
    )
    assert [item["scheduling"]["started_at"] for item in results] == [None, None]
    assert finished == (["a", "b"] if callback else [])


@pytest.mark.parametrize("global_failure", [False, True])
def test_blocked_successors_are_reported_to_the_completion_callback(
    global_failure: bool,
) -> None:
    executed: list[str] = []
    finished: list[tuple[str, str]] = []

    def policy(item: dict[str, Any]) -> ExecutionPolicy:
        return ExecutionPolicy(
            parallel_safe=True,
            failure_scope="global" if global_failure else "branch",
            depends_on=() if global_failure or item["id"] == "a" else ("a",),
        )

    def fail(item: dict[str, Any], _policy: ExecutionPolicy) -> dict[str, Any]:
        executed.append(item["id"])
        raise RuntimeError("synthetic execution failure")

    results = run_scheduled_cases(
        [case("a"), case("b"), case("c")],
        policy_for=policy,
        execute=fail,
        max_workers=1,
        on_finish=lambda item, result: finished.append((item["id"], result["status"])),
    )
    assert executed == ["a"]
    assert finished == [("a", "FAIL"), ("b", "BLOCKED"), ("c", "BLOCKED")]
    assert "synthetic execution failure" in results[0]["output"]
    reason = "global failure" if global_failure else "failed prerequisite"
    assert all(reason in result["reason"] for result in results[1:]), (
        "blocked cases were attributed to the wrong failure scope"
    )


def test_batch_selection_does_not_cross_keys_or_include_unfinished_dependencies() -> (
    None
):
    batches: list[tuple[str, ...]] = []

    def execute_batch(items: Any, policies: Any) -> dict[str, Any]:
        names = tuple(item["id"] for item in items)
        batches.append(names)
        assert len(items) == len(policies)
        return {
            item["id"]: passed(item, policy) for item, policy in zip(items, policies)
        }

    results = run_scheduled_cases(
        [case("a"), case("b"), case("c")],
        policy_for=lambda item: ExecutionPolicy(
            parallel_safe=True, depends_on=("a",) if item["id"] == "c" else ()
        ),
        execute=lambda *_args: pytest.fail("batch case reached single execution"),
        batch_key_for=lambda item, _policy: "separate" if item["id"] == "b" else "same",
        execute_batch=execute_batch,
        max_workers=3,
        max_batch_size=3,
    )
    assert sorted(batches) == [("a",), ("b",), ("c",)]
    assert [item["status"] for item in results] == ["PASS", "PASS", "PASS"]
    assert [item["scheduling"]["batch"]["size"] for item in results] == [1, 1, 1]


@pytest.mark.parametrize("output", [None, {"id": "a", "status": "PARTIAL"}])
def test_invalid_executor_result_blocks_its_dependent_case(output: Any) -> None:
    results = run_scheduled_cases(
        [case("a"), case("b")],
        policy_for=lambda item: ExecutionPolicy(
            parallel_safe=True, depends_on=("a",) if item["id"] == "b" else ()
        ),
        execute=lambda *_args: output,
    )
    assert results[0]["id"] == "a"
    assert results[0]["status"] == "FAIL"
    assert results[1]["status"] == "BLOCKED"
