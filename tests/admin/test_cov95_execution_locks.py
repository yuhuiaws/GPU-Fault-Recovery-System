from __future__ import annotations

import fcntl
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from gpu_fault.admin import deadlines, execution
from gpu_fault.admin import operation_lock as locks


def test_waiting_thread_revalidates_inherited_flock_after_previous_owner_exits(
    tmp_path, monkeypatch
):
    """COV95-ADMIN-003: waiting must not reuse a descriptor whose lock was released."""
    waiting = threading.Event()
    calls = [0]

    class ObservedLock:
        def __init__(self):
            self.delegate = threading.Lock()

        def acquire(self, *, blocking):
            calls[0] += 1
            if calls[0] == 2:
                waiting.set()
            return self.delegate.acquire(blocking=blocking)

        def release(self):
            self.delegate.release()

    monkeypatch.setattr(locks, "Lock", ObservedLock)
    descriptor = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            with locks.site_operation_lock(tmp_path, wait=False) as held:
                descriptor = os.dup(held)
                monkeypatch.setenv(locks.SITE_OPERATION_LOCK_FD_ENV, str(descriptor))

                def contender():
                    with locks.site_operation_lock(tmp_path, wait=True) as acquired:
                        external = os.open(
                            tmp_path / locks.SITE_OPERATION_LOCK, os.O_RDWR
                        )
                        try:
                            with pytest.raises(BlockingIOError):
                                fcntl.flock(external, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        finally:
                            os.close(external)
                        assert acquired != descriptor, (
                            "an unlocked inherited descriptor was reused"
                        )

                future = pool.submit(contender)
                assert waiting.wait(timeout=5), (
                    "contender did not reach the thread-lock wait"
                )
            future.result(timeout=5)
        finally:
            if descriptor is not None:
                os.close(descriptor)


@pytest.mark.parametrize("value", ["invalid", "-1", "9999999", ""])
def test_invalid_inherited_descriptor_does_not_bypass_lock(
    tmp_path, monkeypatch, value
):
    monkeypatch.setenv(locks.SITE_OPERATION_LOCK_FD_ENV, value)
    assert locks.inherited_site_operation_lock_fd(tmp_path) is None
    assert locks.inherited_lock_pass_fds() == ()
    with locks.site_operation_lock(tmp_path, wait=False) as descriptor:
        assert descriptor > 2


@pytest.mark.parametrize(
    "record",
    [
        "not-json",
        "[]",
        "{}",
        '{"pid":null}',
        '{"pid":42}',
        '{"pid":42,"command":"example-task"}',
    ],
)
def test_busy_lock_reports_only_available_holder_metadata(tmp_path, record):
    path = tmp_path / locks.SITE_OPERATION_LOCK
    path.write_text(record)
    with path.open("r+") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            locks.SiteOperationBusy, match="mutation is in progress"
        ) as error:
            with locks.site_operation_lock(tmp_path, wait=False):
                pytest.fail("external holder was bypassed")
        if record in {'{"pid":42}', '{"pid":42,"command":"example-task"}'}:
            assert "held by pid 42" in str(error.value)
        else:
            assert "held by pid" not in str(error.value)


def test_valid_external_inherited_lock_is_not_unlocked_by_borrower(
    tmp_path, monkeypatch
):
    path = tmp_path / locks.SITE_OPERATION_LOCK
    path.touch(mode=0o600)
    with path.open("r+") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setenv(locks.SITE_OPERATION_LOCK_FD_ENV, str(holder.fileno()))
        assert locks.inherited_lock_pass_fds() == (holder.fileno(),)
        with locks.site_operation_lock(tmp_path, wait=False) as descriptor:
            assert descriptor == holder.fileno()
        with path.open("r+") as competitor:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("arguments", [[], ["example", "x" * 400]])
def test_holder_record_has_bounded_command_identity(tmp_path, monkeypatch, arguments):
    monkeypatch.setattr(locks, "sys", SimpleNamespace(argv=arguments))
    with locks.site_operation_lock(tmp_path, wait=False):
        record = json.loads((tmp_path / locks.SITE_OPERATION_LOCK).read_text())
        assert record["pid"] == os.getpid()
        assert len(record["command"]) <= 200
        assert bool(record["command"]) == bool(arguments)


@pytest.mark.parametrize("field", ["ftruncate", "pwrite"])
def test_failed_holder_diagnostics_do_not_release_flock(tmp_path, monkeypatch, field):
    interface = SimpleNamespace(**vars(os))

    def failed(*_args):
        raise OSError("example diagnostic failure")

    setattr(interface, field, failed)
    monkeypatch.setattr(locks, "os", interface)
    with locks.site_operation_lock(tmp_path, wait=False):
        with pytest.raises(locks.SiteOperationBusy):
            with locks.site_operation_lock(tmp_path, wait=False):
                pytest.fail("diagnostic failure released the mutation lock")


@pytest.mark.parametrize(
    "function,arguments",
    [
        (deadlines.Deadline("example", 200).cap, (0,)),
        (deadlines.remaining_timeout, (float("nan"),)),
        (deadlines.remaining_timeout, (float("inf"),)),
    ],
)
def test_invalid_native_timeout_is_refused(function, arguments):
    with pytest.raises(ValueError, match="finite and positive"):
        function(*arguments)


@pytest.mark.parametrize("name", [deadlines.DEADLINE_ENV, deadlines.HARD_DEADLINE_ENV])
@pytest.mark.parametrize("value", ["invalid", "nan", "inf", "1e100"])
def test_inherited_deadlines_require_finite_bounded_values(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    function = (
        deadlines.current_deadline
        if name == deadlines.DEADLINE_ENV
        else deadlines.hard_deadline
    )
    with pytest.raises(ValueError, match="invalid inherited"):
        function()


@pytest.mark.parametrize("reserve", [-1, 7201, float("nan")])
def test_root_budget_rejects_invalid_recovery_reserve(reserve):
    with pytest.raises(ValueError, match="recovery reserve"):
        with deadlines.deployment_deadline("example", 10, recovery_seconds=reserve):
            pytest.fail("invalid reserve was accepted")


@pytest.mark.parametrize("seconds", [0, -1, 121, float("nan")])
def test_cleanup_budget_cannot_be_unbounded(seconds):
    with pytest.raises(ValueError, match="bounded budget"):
        with deadlines.cleanup_deadline("example", seconds):
            pytest.fail("invalid cleanup budget was accepted")


def test_worker_cannot_create_a_new_root_deployment_deadline():
    def worker():
        with deadlines.deployment_deadline("worker", 10):
            pytest.fail("worker thread became root deadline owner")

    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(ValueError, match="CLI thread"):
            pool.submit(worker).result(timeout=5)


def test_proof_cache_eviction_and_failed_verification_are_observable():
    cache = execution.ProofCache(maximum=1)
    first = execution.ProofKey(
        execution.ProofSubject.TOOLCHAIN, "example", "a" * 64, "b" * 64
    )
    second = execution.ProofKey(
        execution.ProofSubject.ARTIFACT, "example", "c" * 64, "b" * 64
    )
    calls = []
    assert cache.verify(first, lambda: calls.append("first") or [], max_age=1) == []
    assert cache.verify(second, lambda: calls.append("second") or {}, max_age=1) == {}
    assert cache.verify(first, lambda: calls.append("reloaded") or [], max_age=1) == []
    assert calls == ["first", "second", "reloaded"]
    cache.invalidate("example")
    with pytest.raises(RuntimeError, match="failed proof"):
        cache.verify(
            first,
            lambda: (_ for _ in ()).throw(RuntimeError("failed proof")),
            max_age=1,
        )
    assert cache.verify(first, lambda: "fresh", max_age=1) == "fresh"


@pytest.mark.parametrize("capacity", [0, -1])
def test_proof_cache_requires_positive_capacity(capacity):
    with pytest.raises(ValueError, match="positive"):
        execution.ProofCache(maximum=capacity)


@pytest.mark.parametrize("maximum", [0, -1, 301, float("nan")])
def test_proof_cache_rejects_unbounded_age(maximum):
    key = execution.ProofKey(
        execution.ProofSubject.ARTIFACT, "example", "a" * 64, "b" * 64
    )
    with pytest.raises(ValueError, match="between zero and 300"):
        execution.ProofCache().verify(
            key, lambda: pytest.fail("invalid age reached verifier"), max_age=maximum
        )


@pytest.mark.parametrize("stdout,encoding", [(1, None), (None, "latin-1")])
def test_driver_rejects_unsupported_output_before_spawn(monkeypatch, stdout, encoding):
    monkeypatch.setattr(
        execution,
        "run_owned_command",
        lambda *_args, **_kwargs: pytest.fail("invalid mode started a driver"),
    )
    with pytest.raises(ValueError, match="unsupported output"):
        execution.run_driver(["fake"], stdout=stdout, encoding=encoding)


def test_driver_check_preserves_failure_status_without_private_output(monkeypatch):
    monkeypatch.setattr(
        execution,
        "run_owned_command",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, 7, "example-output", "example-error"
        ),
    )
    with pytest.raises(subprocess.CalledProcessError) as error:
        execution.run_driver(["fake"], check=True)
    assert error.value.returncode == 7
    assert error.value.output is None and error.value.stderr is None
