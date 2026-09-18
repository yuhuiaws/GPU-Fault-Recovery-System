from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from gpu_fault.admin import execution
from gpu_fault.admin.bootstrap_task_inputs import TaskInputSpec, task_input_spec


def test_child_deadline_cannot_extend_its_parent(monkeypatch):
    monkeypatch.setattr(execution.time, "monotonic", lambda: 100.0)
    with execution.deadline_scope("parent", 20):
        with execution.deadline_scope("child", 200) as child:
            assert child.remaining() == 20
            assert child.cap(200) == 20
            values = execution.command_environment({})
            assert values[execution.DEADLINE_ENV] == "120.0"


def test_expired_parent_stops_work_but_not_bounded_cleanup(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(execution.time, "monotonic", lambda: clock[0])
    with execution.deadline_scope("parent", 1):
        clock[0] = 102.0
        with pytest.raises(execution.DeploymentDeadlineExceeded):
            execution.command_timeout(["aws", "sts", "get-caller-identity"], None)
        with execution.cleanup_deadline("cleanup", 10):
            assert execution.command_timeout(["kubectl", "delete"], 30) == 10
        with pytest.raises(execution.DeploymentDeadlineExceeded):
            execution.command_timeout(["kubectl", "get"], 1)


def test_root_and_recovery_budgets_reach_nested_threads(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(execution.time, "monotonic", lambda: clock[0])
    monkeypatch.delenv(execution.DEADLINE_ENV, raising=False)
    with execution.deployment_deadline("root", 10):
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(execution.current_deadline).result().expires == 110
        clock[0] = 111
        with execution.recovery_deadline("rollback"):
            with ThreadPoolExecutor(max_workers=1) as pool:
                inherited = pool.submit(execution.current_deadline).result()
                assert inherited.expires == 7310
                assert inherited.label == "rollback"
        assert os.environ[execution.DEADLINE_ENV] == "110.0"
    assert execution.DEADLINE_ENV not in os.environ


def test_driver_preserves_passed_file_descriptors_and_deadline(tmp_path):
    path = tmp_path / "inherited"
    with path.open("w") as handle:
        result = execution.run_driver(
            [sys.executable, "-c", f"import os; os.write({handle.fileno()}, b'held')"],
            pass_fds=(handle.fileno(),),
            capture_output=True,
        )
    assert result.returncode == 0, result.stderr
    assert path.read_text() == "held"


def test_timeout_reaps_the_shells_descendant_processes(tmp_path):
    marker = tmp_path / "survived"
    ready = tmp_path / "started"
    child = (
        "import time; from pathlib import Path; "
        f"Path({str(ready)!r}).touch(); time.sleep(2); "
        f"Path({str(marker)!r}).write_text('unexpected')"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(10)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        execution.run_command([sys.executable, "-c", parent], timeout_seconds=1)
    assert ready.exists(), "the test did not start the descendant before timeout"
    time.sleep(1.2)
    assert not marker.exists(), "a timed-out deployment left a live descendant"


def key(*, scope="site-a", identity="a"):
    return execution.ProofKey(
        execution.ProofSubject.TOOLCHAIN, scope, identity * 64, "b" * 64
    )


def test_proofs_are_scoped_fresh_and_return_independent_copies(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(execution.time, "monotonic", lambda: clock[0])
    cache = execution.ProofCache()
    calls = []

    def verify():
        calls.append(True)
        return {"healthy": True}

    first = cache.verify(key(), verify, max_age=10)
    first["healthy"] = False
    assert cache.verify(key(), verify, max_age=10) == {"healthy": True}
    assert len(calls) == 1
    cache.verify(key(scope="site-b"), verify, max_age=10)
    cache.verify(key(identity="c"), verify, max_age=10)
    assert len(calls) == 3
    clock[0] = 111.0
    cache.verify(key(), verify, max_age=10)
    assert len(calls) == 4
    cache.invalidate("site-a")
    cache.verify(key(), verify, max_age=10)
    assert len(calls) == 5


def test_failed_proofs_are_never_cached():
    cache = execution.ProofCache()
    attempts = []

    def fail():
        attempts.append(True)
        raise ValueError("invalid artifact")

    for _ in range(2):
        with pytest.raises(ValueError, match="invalid artifact"):
            cache.verify(key(), fail, max_age=10)
    assert len(attempts) == 2


def test_credential_proofs_require_a_fresh_resource_version():
    with pytest.raises(ValueError, match="resource version"):
        execution.ProofKey(
            execution.ProofSubject.CREDENTIAL_SHAPE, "site-a", "a" * 64, "b" * 64
        )


def test_distinct_proofs_validate_in_parallel_and_same_identity_is_single_flight():
    cache = execution.ProofCache()
    barrier = threading.Barrier(2)

    def verify():
        barrier.wait(timeout=2)
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(cache.verify, key(scope=scope), verify, max_age=10)
            for scope in ("a", "b")
        ]
        assert [future.result(timeout=3) for future in futures] == [True, True]
    calls = []

    def verify_once():
        calls.append(True)
        time.sleep(0.02)
        return {"verified": True}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(cache.verify, key(scope="same"), verify_once, max_age=10)
            for _ in range(4)
        ]
        assert all(future.result() == {"verified": True} for future in futures), (
            "single-flight waiters received different proofs"
        )
    assert len(calls) == 1


def test_tasks_cannot_omit_their_input_or_revalidation_contract():
    with pytest.raises(ValueError, match="input identity"):
        TaskInputSpec("unbound")
    with pytest.raises(ValueError, match="input policy"):
        task_input_spec("unregistered_task")
    assert task_input_spec("cpu_access").always_revalidate_reason, (
        "CPU access must not reuse an unproved checkpoint"
    )
    assert task_input_spec("aurora").deadline_seconds == 3600
    assert task_input_spec("release").deadline_seconds == 7200
    assert task_input_spec("control_record_archive_bucket").always_revalidate_reason, (
        "archive hardening must re-read the current site"
    )
    with pytest.raises(ValueError, match="FAILED"):
        task_input_spec("grafana_install").result_status(
            {"grafana": {"status": "FAILED"}}
        )
