"""One private local PG16 allocation and one explicitly serial pytest partition."""

from __future__ import annotations

import argparse
import os
import re
import signal
import sys
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path

import pytest

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import CommandRunner
from gpu_fault.admin.deadlines import (
    cleanup_deadline,
    remaining_timeout,
)
from gpu_fault.admin.postgres_grant import (
    ALLOCATION_ENV,
    CONTAINER_ID,
    LOCAL_DOCKER_HOST,
    MAX_GRANT_BYTES,
    POSTGRES_URL_ENV,
    postgres_test_environment,
    private_directory,
)
from gpu_fault.admin.process_supervisor import (
    ProcessSupervisionLost,
    ensure_supervision_safe,
    interruption_scope,
    run_owned_command,
)
from gpu_fault.admin.release_postgres import (
    ALLOCATION_DIRECTORY,
    IMAGE_ID,
    POSTGRES_IMAGE,
    OwnedPostgres,
    isolated_postgres_allocation,
)
from scripts.postgres_shard_receipts import (
    PostgresShardError,
    read_private_json,
    safe_failure_nodeid,
    validate_targets,
)
from scripts.run_static_gates import isolated_gate_environment
from tools import pytest_case_reporter
from tools.pytest_case_reporter import (
    PARTITION_COUNT_ENV,
    PARTITION_INDEX_ENV,
    REPORT_ENV,
)
from tools.pytest_result_identity import source_identity

ROOT = Path(__file__).resolve().parents[1]
FAILURE_ENV = "PYTEST_GPU_FAULT_LOCAL_POSTGRES_FAILURE"
RECEIPT_FILE = "pytest.json"
FAILURE_FILE = "failure.json"
ALLOCATION_FILE = "allocation.json"
CANCEL_FILE = "cancel.json"
READY_FILE = "pytest-ready.json"
WORK_SECONDS = 7200
TEST_CONTROL_ENV = frozenset(
    {
        "GPU_FAULT_POSTGRES_LOCK_STRESS_WORKERS",
        "GPU_FAULT_POSTGRES_LOCK_STRESS_ROUNDS",
        "GPU_FAULT_TEST_SHUFFLE_SEED",
    }
)
SESSION_CONTROLS = ExitStack()
FAILURE_PATH: Path | None = None


@contextmanager
def cancellation_scope(*, request_only: bool = False) -> Iterator[threading.Event]:
    """Keep lifecycle drivers alive while their workers stop only leaf commands."""
    with interruption_scope(wait_all=True):
        interrupted = threading.Event()
        signals = (signal.SIGINT, signal.SIGTERM) if request_only else (signal.SIGTERM,)
        previous = {signum: signal.getsignal(signum) for signum in signals}

        def receive(_number: int, _frame: object) -> None:
            interrupted.set()
            if not request_only:
                signal.raise_signal(signal.SIGINT)

        for signum in signals:
            signal.signal(signum, receive)
        try:
            yield interrupted
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def cancellation_requested(state: Path) -> bool:
    path = state / CANCEL_FILE
    return path.exists() or path.is_symlink()


@contextmanager
def cancel_pytest_scope(state: Path) -> Iterator[None]:
    stopped = threading.Event()

    def watch() -> None:
        while not stopped.wait(0.02):
            # A configured pytest proves its supervisors are already running.
            # Never signal a newly spawning cleanup supervisor before its handlers exist.
            if cancellation_requested(state) and (state / READY_FILE).is_file():
                if not stopped.is_set():
                    os.kill(os.getpid(), signal.SIGINT)
                return

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def worker_environment(parent: Mapping[str, str], *, root: Path) -> dict[str, str]:
    if parent.get(POSTGRES_URL_ENV, "").strip() and ALLOCATION_ENV not in parent:
        raise PostgresShardError(
            "external PostgreSQL URLs are not eligible for local sharding"
        )
    if ALLOCATION_ENV in parent:
        # This is only admission. Never inspect, update, stop or reuse the parent's DB.
        postgres_test_environment(parent)
    result = {
        name: value
        for name, value in isolated_gate_environment(parent).items()
        if not name.startswith(("PG", "POSTGRES_", "PYTEST_", "COSIGN_", "AWS_"))
        and (not name.startswith("GPU_FAULT_") or name in TEST_CONTROL_ENV)
        and name
        not in {
            POSTGRES_URL_ENV,
            ALLOCATION_ENV,
            "KUBECONFIG",
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONPYCACHEPREFIX",
            "DOCKER_HOST",
            "DOCKER_CONTEXT",
            "DOCKER_CONFIG",
            "DOCKER_TLS_VERIFY",
            "DOCKER_CERT_PATH",
        }
    }
    result.update(
        PYTHONPATH=os.pathsep.join((str(root / "src"), str(root))),
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        AWS_CONFIG_FILE=os.devnull,
        AWS_SHARED_CREDENTIALS_FILE=os.devnull,
        AWS_EC2_METADATA_DISABLED="true",
    )
    return result


def load_owned_allocation(state: Path, runner: CommandRunner) -> OwnedPostgres | None:
    """Restore only allocator-written ownership, never infer ownership from a name."""
    private_directory(state)
    directory = state / ALLOCATION_DIRECTORY
    if not directory.exists() and not directory.is_symlink():
        return None
    info = private_directory(directory)
    value = read_private_json(directory / "ownership.json", maximum=MAX_GRANT_BYTES)
    owner, image = value.get("owner"), value.get("image_id")
    container, port = value.get("container_id"), value.get("port")
    attempted, lost, phase = (
        value.get("create_attempted"),
        value.get("supervision_lost"),
        value.get("phase"),
    )
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
        or not isinstance(owner, str)
        or re.fullmatch(r"[0-9a-f]{32}", owner) is None
        or not isinstance(image, str)
        or IMAGE_ID.fullmatch(image) is None
        or value.get("name") != "gpu-fault-release-postgres-" + owner
        or value.get("image_reference") != POSTGRES_IMAGE
        or value.get("docker_host") != LOCAL_DOCKER_HOST
        or (value.get("directory_device"), value.get("directory_inode"))
        != (info.st_dev, info.st_ino)
        or type(value.get("directory_device")) is not int
        or type(value.get("directory_inode")) is not int
        or container is not None
        and (
            not isinstance(container, str) or CONTAINER_ID.fullmatch(container) is None
        )
        or port is not None
        and (type(port) is not int or not 1 <= port <= 65535)
        or type(attempted) is not bool
        or type(lost) is not bool
        or not isinstance(phase, str)
        or phase
        not in {
            "PREPARED",
            "CREATE_STARTED",
            "CREATED",
            "STARTED",
            "READY",
            "REMOVED",
            "UNCONFIRMED",
            "SUPERVISION_LOST",
        }
        or not attempted
        and (
            container is not None
            or port is not None
            or phase not in {"PREPARED", "UNCONFIRMED"}
            or (directory / "container.cid").exists()
            or (directory / "container.cid").is_symlink()
        )
        or phase in {"CREATED", "STARTED", "READY", "REMOVED"}
        and (not attempted or container is None)
        or phase == "READY"
        and port is None
    ):
        raise PostgresShardError("PostgreSQL reconciliation ownership is invalid")
    if lost or phase == "SUPERVISION_LOST":
        raise ProcessSupervisionLost("PostgreSQL worker recorded lost supervision")
    return OwnedPostgres(
        runner,
        directory,
        image,
        owner,
        (info.st_dev, info.st_ino),
        container=container,
        port=port,
        create_attempted=attempted,
        phase=phase,
        password="",
    )


def reconcile_allocation(state: Path, runner: CommandRunner) -> None:
    ensure_supervision_safe(allow_interrupted=True)
    owned = load_owned_allocation(state, runner)
    if owned is None:
        return
    try:
        with cleanup_deadline("local PostgreSQL shard reconciliation"):
            owned.cleanup()
            owned.remove_directory()
    except ProcessSupervisionLost:
        owned.retain(supervision_lost=True)
        raise
    except BaseException:
        owned.retain()
        raise PostgresShardError("PostgreSQL shard cleanup is unconfirmed") from None


def record_failure(
    path: Path,
    *,
    nodeid: str | None = None,
    phase: str = "process",
    outcome: str = "failed",
) -> None:
    if not path.exists() and not path.is_symlink():
        write_json_atomic(
            path,
            {
                "failed": True,
                "nodeid": nodeid if safe_failure_nodeid(nodeid) else None,
                "phase": phase,
                "outcome": outcome,
            },
        )


def mark_failure(nodeid: str, phase: str, outcome: str) -> None:
    if FAILURE_PATH is not None:
        record_failure(FAILURE_PATH, nodeid=nodeid, phase=phase, outcome=outcome)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    del config
    global FAILURE_PATH
    reference = os.environ.pop(FAILURE_ENV, None)
    FAILURE_PATH = Path(reference) if reference else None
    controls = {}
    for name in (REPORT_ENV, PARTITION_COUNT_ENV, PARTITION_INDEX_ENV):
        value = os.environ.pop(name, None)
        if value is None:
            raise pytest.UsageError("local PostgreSQL reporter binding is incomplete")
        controls[name] = value
    SESSION_CONTROLS.enter_context(
        pytest_case_reporter.bound_session_controls(controls)
    )
    os.environ.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
    if FAILURE_PATH is not None:
        write_json_atomic(FAILURE_PATH.parent / READY_FILE, {"pid": os.getpid()})


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    del config
    global FAILURE_PATH
    SESSION_CONTROLS.close()
    FAILURE_PATH = None


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    # Keep reporter phases/durations, but never copy test output or DSN exceptions.
    report.sections.clear()
    if report.failed or report.skipped:
        report.longrepr = (
            (
                report.location[0],
                report.location[1] or 0,
                "Skipped: local shard refuses skips",
            )
            if report.skipped
            else "local PostgreSQL test failed; private output omitted"
        )
        mark_failure(report.nodeid, report.when, report.outcome)


@pytest.hookimpl(tryfirst=True)
def pytest_collectreport(report: pytest.CollectReport) -> None:
    report.sections.clear()
    if report.failed or report.skipped:
        report.longrepr = (
            (report.nodeid, 0, "Skipped: local shard refuses collection skips")
            if report.skipped
            else "local PostgreSQL collection did not pass"
        )
        mark_failure(report.nodeid, "collect", report.outcome)


def serial_test_command(
    python: str, *, state: Path, tests: Sequence[str], durations: int
) -> list[str]:
    return [
        python,
        "-B",
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "-p",
        "xdist.plugin",
        "-p",
        "tools.pytest_case_reporter",
        "-p",
        "scripts.postgres_shard_worker",
        "-p",
        "no:cacheprovider",
        "-n",
        "0",
        "-q",
        "-x",
        "--tb=no",
        "--basetemp",
        str(state / "pytest-temp"),
        f"--durations={durations}",
        *tests,
    ]


def run_worker(
    *,
    state: Path,
    tests: tuple[str, ...],
    workers: int,
    index: int,
    durations: int,
    identity: str,
) -> int:
    private_directory(state)
    if cancellation_requested(state):
        raise KeyboardInterrupt
    if (
        POSTGRES_URL_ENV in os.environ
        or ALLOCATION_ENV in os.environ
        or any(name.startswith(("PG", "POSTGRES_")) for name in os.environ)
    ):
        raise PostgresShardError("PostgreSQL worker inherited database credentials")
    if source_identity(ROOT) != identity:
        raise PostgresShardError("PostgreSQL source changed before worker allocation")
    runner = CommandRunner()
    with isolated_postgres_allocation(
        runner, repository_root=ROOT, state_dir=state
    ) as allocation:
        if allocation.directory != state / ALLOCATION_DIRECTORY:
            raise PostgresShardError(
                "PostgreSQL worker did not receive its own allocation"
            )
        if cancellation_requested(state):
            raise KeyboardInterrupt
        environment = postgres_test_environment(
            allocation.build_environment(isolated_gate_environment(os.environ))
        )
        environment.update(
            {
                REPORT_ENV: str(state / RECEIPT_FILE),
                PARTITION_COUNT_ENV: str(workers),
                PARTITION_INDEX_ENV: str(index),
                FAILURE_ENV: str(state / FAILURE_FILE),
            }
        )
        ownership = read_private_json(
            allocation.directory / "ownership.json", maximum=MAX_GRANT_BYTES
        )
        write_json_atomic(state / ALLOCATION_FILE, ownership)
        with cancel_pytest_scope(state):
            result = run_owned_command(
                serial_test_command(
                    sys.executable, state=state, tests=tests, durations=durations
                ),
                timeout=remaining_timeout(WORK_SECONDS),
                environment=environment,
                cwd=ROOT,
            )
        if result.returncode:
            record_failure(state / FAILURE_FILE)
        return int(result.returncode)


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--state", type=Path)
    parser.add_argument("--identity")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--durations", type=int, default=50)
    parser.add_argument("--tests", nargs="+")
    options = parser.parse_args(arguments)
    os.umask(0o077)
    try:
        with cancellation_scope():
            if options.identity_only:
                print(source_identity(ROOT))
                return 0
            if (
                options.state is None
                or options.identity is None
                or not 1 <= options.workers <= 16
                or not 0 <= options.index < options.workers
                or options.durations < 0
            ):
                raise PostgresShardError("invalid local PostgreSQL worker request")
            return run_worker(
                state=options.state,
                tests=validate_targets(ROOT, options.tests or []),
                workers=options.workers,
                index=options.index,
                durations=options.durations,
                identity=options.identity,
            )
    except KeyboardInterrupt:
        return 130
    except (Exception, ProcessSupervisionLost):
        # No exception message, captured output, environment or URL is public output.
        print(
            "postgres-shards: worker failed; retained private state requires validation",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
