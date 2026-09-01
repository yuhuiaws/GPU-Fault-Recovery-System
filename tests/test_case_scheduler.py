from __future__ import annotations

import time
from threading import Barrier, Lock

import pytest

from tools.case_scheduler import ExecutionPolicy, ResourceLock, run_scheduled_cases


def _case(case_id: str) -> dict:
    return {
        "id": case_id,
        "title": case_id,
        "category": "test",
        "level": "unit",
        "risk": "non-destructive",
    }


def test_shared_locks_allow_parallel_execution() -> None:
    cases = [_case("case-a"), _case("case-b")]
    barrier = Barrier(2)

    def execute(case, _policy):
        barrier.wait(timeout=2)
        return {"id": case["id"], "status": "PASS"}

    results = run_scheduled_cases(
        cases,
        policy_for=lambda _case: ExecutionPolicy(
            parallel_safe=True, locks=(ResourceLock("local-test", "shared"),)
        ),
        execute=execute,
        max_workers=2,
    )

    assert [result["status"] for result in results] == ["PASS", "PASS"]
    assert len({result["scheduling"]["worker"] for result in results}) == 2


def test_exclusive_lock_serializes_cases() -> None:
    cases = [_case("case-a"), _case("case-b")]
    state_lock = Lock()
    active = 0
    peak = 0

    def execute(case, _policy):
        nonlocal active, peak
        with state_lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with state_lock:
            active -= 1
        return {"id": case["id"], "status": "PASS"}

    results = run_scheduled_cases(
        cases,
        policy_for=lambda _case: ExecutionPolicy(
            parallel_safe=True, locks=(ResourceLock("postgres-server", "exclusive"),)
        ),
        execute=execute,
        max_workers=2,
    )

    assert [result["status"] for result in results] == ["PASS", "PASS"]
    assert peak == 1
    assert results[1]["scheduling"]["lock_wait_seconds"] >= 0.02


def test_postgres_exclusive_lock_can_overlap_local_shared_lock() -> None:
    cases = [_case("cap005"), _case("local")]
    barrier = Barrier(2)

    def policy(case):
        resource = "postgres-server" if case["id"] == "cap005" else "local-test"
        mode = "exclusive" if case["id"] == "cap005" else "shared"
        return ExecutionPolicy(
            parallel_safe=True, locks=(ResourceLock(resource, mode),)
        )

    def execute(case, _policy):
        barrier.wait(timeout=2)
        return {"id": case["id"], "status": "PASS"}

    results = run_scheduled_cases(
        cases, policy_for=policy, execute=execute, max_workers=2
    )

    assert [result["status"] for result in results] == ["PASS", "PASS"]
    assert len({result["scheduling"]["worker"] for result in results}) == 2


def test_global_exclusive_case_waits_for_parallel_case() -> None:
    cases = [_case("parallel"), _case("exclusive")]
    events = []
    events_lock = Lock()

    def policy(case):
        if case["id"] == "exclusive":
            return ExecutionPolicy(parallel_safe=False, failure_scope="global")
        return ExecutionPolicy(
            parallel_safe=True, locks=(ResourceLock("local-test", "shared"),)
        )

    def execute(case, _policy):
        with events_lock:
            events.append(f"start:{case['id']}")
        time.sleep(0.02)
        with events_lock:
            events.append(f"end:{case['id']}")
        return {"id": case["id"], "status": "PASS"}

    run_scheduled_cases(cases, policy_for=policy, execute=execute, max_workers=2)

    assert events == [
        "start:parallel",
        "end:parallel",
        "start:exclusive",
        "end:exclusive",
    ]


def test_collect_all_continues_after_global_failure() -> None:
    cases = [_case("failure"), _case("independent")]
    executed = []

    def execute(case, _policy):
        executed.append(case["id"])
        return {
            "id": case["id"],
            "status": "FAIL" if case["id"] == "failure" else "PASS",
        }

    results = run_scheduled_cases(
        cases,
        policy_for=lambda _case: ExecutionPolicy(
            parallel_safe=False, failure_scope="global"
        ),
        execute=execute,
        max_workers=2,
        collect_all=True,
    )

    assert executed == ["failure", "independent"]
    assert [result["status"] for result in results] == ["FAIL", "PASS"]
    assert all(result["scheduling"]["collect_all"] is True for result in results), (
        results
    )


def test_failed_prerequisite_blocks_dependent_case() -> None:
    cases = [_case("parent"), _case("child")]
    executed = []

    def policy(case):
        return ExecutionPolicy(
            parallel_safe=True,
            locks=(ResourceLock("local-test", "shared"),),
            depends_on=("parent",) if case["id"] == "child" else (),
        )

    def execute(case, _policy):
        executed.append(case["id"])
        return {
            "id": case["id"],
            "status": "FAIL" if case["id"] == "parent" else "PASS",
        }

    results = run_scheduled_cases(
        cases, policy_for=policy, execute=execute, max_workers=2
    )

    assert executed == ["parent"]
    assert results[0]["status"] == "FAIL"
    assert results[1]["status"] == "BLOCKED"
    assert "parent" in results[1]["reason"]


def test_custom_success_predicate_allows_advisory_pass() -> None:
    cases = [_case("parent"), _case("child")]
    executed = []

    def policy(case):
        return ExecutionPolicy(
            parallel_safe=True,
            locks=(ResourceLock("local-test", "shared"),),
            depends_on=("parent",) if case["id"] == "child" else (),
        )

    def execute(case, _policy):
        executed.append(case["id"])
        if case["id"] == "parent":
            return {"id": case["id"], "status": "NOT_RUN", "analysis_status": "PASS"}
        return {"id": case["id"], "status": "PASS"}

    results = run_scheduled_cases(
        cases,
        policy_for=policy,
        execute=execute,
        max_workers=2,
        is_successful=lambda result: (
            result.get("status") == "PASS" or result.get("analysis_status") == "PASS"
        ),
    )

    assert executed == ["parent", "child"]
    assert [result["status"] for result in results] == ["NOT_RUN", "PASS"]


def test_results_remain_in_declared_order() -> None:
    cases = [_case("slow"), _case("fast")]

    def execute(case, _policy):
        if case["id"] == "slow":
            time.sleep(0.03)
        return {"id": case["id"], "status": "PASS"}

    results = run_scheduled_cases(
        cases,
        policy_for=lambda _case: ExecutionPolicy(
            parallel_safe=True, locks=(ResourceLock("local-test", "shared"),)
        ),
        execute=execute,
        max_workers=2,
    )

    assert [result["id"] for result in results] == ["slow", "fast"]


def test_batch_executor_groups_compatible_cases() -> None:
    cases = [_case("case-a"), _case("case-b"), _case("local")]
    batches = []

    def execute(case, _policy):
        return {"id": case["id"], "status": "PASS"}

    def execute_batch(batch, _policies):
        batches.append([case["id"] for case in batch])
        return {case["id"]: {"id": case["id"], "status": "PASS"} for case in batch}

    results = run_scheduled_cases(
        cases,
        policy_for=lambda _case: ExecutionPolicy(
            parallel_safe=True, locks=(ResourceLock("local-test", "shared"),)
        ),
        execute=execute,
        batch_key_for=(
            lambda case, _policy: "codex-manual"
            if case["id"].startswith("case-")
            else None
        ),
        execute_batch=execute_batch,
        max_batch_size=3,
        max_workers=3,
    )

    assert batches == [["case-a", "case-b"]]
    assert [result["status"] for result in results] == ["PASS", "PASS", "PASS"]
    assert results[0]["scheduling"]["batch"]["key"] == "codex-manual"
    assert results[0]["scheduling"]["batch"]["size"] == 2
    assert (
        results[0]["scheduling"]["batch"]["id"]
        == results[1]["scheduling"]["batch"]["id"]
    )
    assert "batch" not in results[2]["scheduling"]


def test_batch_executor_respects_intra_batch_resource_conflicts() -> None:
    cases = [_case("case-a"), _case("case-b")]
    batches = []

    def execute_batch(batch, _policies):
        batches.append([case["id"] for case in batch])
        return {case["id"]: {"id": case["id"], "status": "PASS"} for case in batch}

    results = run_scheduled_cases(
        cases,
        policy_for=lambda _case: ExecutionPolicy(
            parallel_safe=True, locks=(ResourceLock("shared-target", "exclusive"),)
        ),
        execute=lambda case, _policy: {"id": case["id"], "status": "PASS"},
        batch_key_for=lambda _case, _policy: "codex-manual",
        execute_batch=execute_batch,
        max_batch_size=2,
        max_workers=2,
    )

    assert batches == [["case-a"], ["case-b"]]
    assert [result["status"] for result in results] == ["PASS", "PASS"]


def test_batch_executor_missing_result_fails_closed() -> None:
    cases = [_case("case-a"), _case("case-b")]

    results = run_scheduled_cases(
        cases,
        policy_for=lambda _case: ExecutionPolicy(
            parallel_safe=True, locks=(ResourceLock("local-test", "shared"),)
        ),
        execute=lambda case, _policy: {"id": case["id"], "status": "PASS"},
        batch_key_for=lambda _case, _policy: "codex-manual",
        execute_batch=lambda batch, _policies: {
            batch[0]["id"]: {"id": batch[0]["id"], "status": "PASS"}
        },
        max_batch_size=2,
        max_workers=2,
    )

    assert [result["status"] for result in results] == ["FAIL", "FAIL"]
    assert all(
        "batch executor result IDs do not match" in result["output"]
        for result in results
    ), results


def test_invalid_status_and_malformed_batch_fail_per_case() -> None:
    cases = [_case("case-a"), _case("case-b")]

    results = run_scheduled_cases(
        cases,
        policy_for=lambda _case: ExecutionPolicy(
            parallel_safe=True, locks=(ResourceLock("local-test", "shared"),)
        ),
        execute=lambda case, _policy: {"id": case["id"], "status": "PASS"},
        batch_key_for=lambda _case, _policy: "codex-manual",
        execute_batch=lambda batch, _policies: {
            batch[0]["id"]: {"id": batch[0]["id"], "status": "UNKNOWN"},
            batch[1]["id"]: "not-a-mapping",
        },
        max_batch_size=2,
        max_workers=2,
        collect_all=True,
    )

    assert [result["status"] for result in results] == ["FAIL", "FAIL"]
    assert "unsupported status" in results[0]["output"]
    assert "must be a mapping" in results[1]["output"]


def test_batch_configuration_requires_both_callbacks() -> None:
    with pytest.raises(
        ValueError, match="batch_key_for and execute_batch must be configured together"
    ):
        run_scheduled_cases(
            [_case("case-a")],
            policy_for=lambda _case: ExecutionPolicy(parallel_safe=False),
            execute=lambda case, _policy: {"id": case["id"], "status": "PASS"},
            batch_key_for=lambda _case, _policy: "codex-manual",
        )
