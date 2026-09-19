from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gpu_fault.admin import api_budget as budget
from gpu_fault.admin.execution import cleanup_deadline, run_command


@pytest.fixture
def tools(tmp_path, monkeypatch):
    executable = (
        f"#!{sys.executable}\n"
        "import json,os,socket,sys,time\n"
        "time.sleep(0.02)\n"
        "if os.path.basename(sys.argv[0]) == 'aws':\n"
        " s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)\n"
        " common={'Version':1,'ClientId':os.environ['AWS_CSM_CLIENT_ID'],"
        "'SessionToken':'example-should-never-be-recorded'}\n"
        " for kind in ['ApiCallAttempt','ApiCallAttempt','ApiCall']:\n"
        "  s.sendto(json.dumps({**common,'Type':kind,'AttemptCount':2}).encode(),"
        "('127.0.0.1',int(os.environ['AWS_CSM_PORT'])))\n"
        " print('fake-aws')\n"
        "else: print('fake-kubectl')\n"
    )
    for name in ("aws", "kubectl"):
        path = tmp_path / name
        path.write_text(executable)
        path.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.delenv(budget.ROOT_ENV, raising=False)
    return tmp_path


def test_api_children_share_budget_and_sdk_retries_are_counted_without_credentials(
    tools,
):
    old_path = os.environ["PATH"]
    with budget.deployment_api_budget():
        root = budget.budget_root()
        assert root is not None
        assert budget.resolve_tool("aws") == str(tools / "aws")

        def run(index):
            with budget.api_phase(f"stage-{index % 2}"):
                result = run_command(
                    ["sh", "-c", "aws sts get-caller-identity && kubectl version"]
                )
                assert result.returncode == 0, result.stderr
                assert result.stdout == "fake-aws\nfake-kubectl\n"

        with ThreadPoolExecutor(max_workers=12) as pool:
            list(pool.map(run, range(12)))
        deadline = time.monotonic() + 3
        while budget.statistics()["backends"]["aws"]["sdk_calls"] != 12:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        stats = budget.statistics()
        assert stats["backends"]["aws"]["commands"] == 12
        assert stats["backends"]["kubectl"]["commands"] == 12
        assert stats["backends"]["aws"]["sdk_attempts"] == 24
        assert stats["backends"]["aws"]["sdk_retries"] == 12
        assert stats["sdk_accounting"] == "observed-csm-lower-bound"
        assert stats["http_rate_limit"] is False
        assert all(
            stats["peak_admitted_weight"][name] <= value
            for name, value in budget.LIMITS.items()
        ), "descendant API commands exceeded a deployment-wide capacity limit"
        assert budget.statistics("stage-0")["backends"]["aws"]["commands"] == 6
        with sqlite3.connect(root / "budget.sqlite3") as database:
            assert database.execute("SELECT COUNT(*) FROM leases").fetchone()[0] == 0
        for path in root.glob("budget.sqlite3*"):
            assert b"example-should-never-be-recorded" not in path.read_bytes()
    assert not root.exists(), "the completed deployment left its private budget ledger"
    assert os.environ["PATH"] == old_path
    assert budget.ROOT_ENV not in os.environ


def test_nested_budget_reuses_outer_scope(tools):
    with budget.deployment_api_budget():
        root = budget.budget_root()
        with budget.api_slot("aws") as identifier:
            before = budget.statistics()
            with budget.deployment_api_budget():
                assert budget.budget_root() == root
                assert budget.statistics() == before
                with sqlite3.connect(root / "budget.sqlite3") as database:
                    assert (
                        dict(
                            database.execute("SELECT backend,max_weight FROM capacity")
                        )
                        == budget.LIMITS
                    )
                    assert database.execute("SELECT id FROM leases").fetchall() == [
                        (identifier,)
                    ]
        assert root.is_dir(), "a nested invocation removed the outer deployment budget"


def test_expired_deadline_never_starts_an_api_call(tools, monkeypatch):
    with budget.deployment_api_budget():
        monkeypatch.setenv(
            "GPU_FAULT_DEPLOY_DEADLINE_MONOTONIC", str(time.monotonic() - 1)
        )
        with pytest.raises(budget.ApiBudgetError, match="deadline"):
            with budget.api_slot("aws"):
                pytest.fail("expired API admission was accepted")
        with cleanup_deadline("expired admission accounting"):
            assert budget.statistics()["backends"] == {}


def test_public_or_symlinked_budget_directory_is_refused(tmp_path, monkeypatch):
    tmp_path.chmod(0o755)
    monkeypatch.setenv(budget.ROOT_ENV, str(tmp_path))
    with pytest.raises(budget.ApiBudgetError, match="private"):
        budget.budget_root()


@pytest.mark.allows_cluster_binaries("aws")
def test_transfer_clients_are_bounded_without_modifying_user_config(
    tools, tmp_path, monkeypatch
):
    source = tmp_path / "config"
    content = (
        "[profile build]\nregion = us-east-1\ns3 =\n max_concurrent_requests = 100\n"
    )
    source.write_text(content)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(source))
    with budget.deployment_api_budget():
        assert budget.resolve_tool("aws") == str(tools / "aws")
        result = run_command(["aws", "s3", "cp", "fixture-source", "fixture-target"])
        assert result.returncode == 0, result.stderr
        root = budget.budget_root()
        configs = list(root.glob("aws-config-*"))
        assert len(configs) == 1
        assert configs[0].stat().st_mode & 0o077 == 0
        assert "max_concurrent_requests = 4" in configs[0].read_text()
        assert "preferred_transfer_client = classic" in configs[0].read_text()
        assert budget.statistics()["peak_admitted_weight"]["aws"] == 4
    assert source.read_text() == content


@pytest.fixture
def recursive_tools(tools, tmp_path, monkeypatch):
    executable = f"#!{sys.executable}\n" + textwrap.dedent(
        """\
        import json
        import os
        from pathlib import Path
        import sqlite3
        import stat
        import subprocess
        import sys
        import threading
        import time

        directory = Path(os.environ["FAKE_API_WORK"])
        role, weight = sys.argv[-2], int(sys.argv[-1])
        pid = os.getpid()
        backend = Path(sys.argv[0]).name
        shim_pid = os.getppid()
        spools = []
        for descriptor in Path(f"/proc/{shim_pid}/fd").iterdir():
            try:
                info = descriptor.stat()
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode) and info.st_nlink == 0:
                spools.append({
                    "fd": int(descriptor.name),
                    "mode": info.st_mode & 0o777, "links": info.st_nlink,
                })
        pending = directory / f"pending-{pid}"
        pending.write_text(json.dumps({
            "pid": pid, "weight": weight, "backend": backend, "shim_pid": shim_pid,
            "lease": os.environ.get("GPU_FAULT_DEPLOY_API_PARENT")
                     or os.environ["AWS_CSM_CLIENT_ID"],
            "output_modes": [os.fstat(fd).st_mode & 0o777 for fd in (1, 2)],
            "output_links": [os.fstat(fd).st_nlink for fd in (1, 2)],
            "spools": spools,
        }))
        pending.rename(directory / f"{role}-{pid}.json")
        deadline = time.monotonic() + 12

        def worker():
            while True:
                time.sleep(0.01)

        threading.Thread(target=worker, daemon=True).start()

        def wait_for(predicate):
            while not predicate():
                if time.monotonic() >= deadline:
                    raise RuntimeError("fake CLI barrier expired")
                time.sleep(0.01)

        if role == "outer":
            count = int(os.environ["FAKE_API_OUTERS"])
            wait_for(lambda: len(list(directory.glob("outer-*.json"))) == count)
            field = os.environ.get("FAKE_API_CORRUPT_IDENTITY")
            if field:
                assert field in {"pid_start", "command_start"}
                root = Path(os.environ["GPU_FAULT_DEPLOY_API_BUDGET_DIR"])
                with sqlite3.connect(root / "budget.sqlite3") as database:
                    database.execute(
                        f"UPDATE leases SET {field}='stale-start' WHERE id=?",
                        (os.environ["AWS_CSM_CLIENT_ID"],),
                    )
        levels = int(os.environ.get("FAKE_API_LEVELS", "1"))
        if role == "outer" or (role == "inner" and levels):
            child_weight = int(os.environ["FAKE_API_CHILD_WEIGHT"])
            service = "s3" if child_weight == 4 else "sts"
            child_backend = "aws"
            if os.environ.get("FAKE_API_ALTERNATE"):
                child_backend = "kubectl" if backend == "aws" else "aws"
            environment = {**os.environ, "FAKE_API_LEVELS": str(levels - 1)}
            children = []
            capture = bool(os.environ.get("FAKE_API_CAPTURE_OUTPUT"))
            try:
                for _ in range(int(os.environ.get("FAKE_API_FANOUT", "1"))):
                    children.append(subprocess.Popen([
                        sys.executable, "-c",
                        "import subprocess,sys; "
                        "sys.exit(subprocess.call(sys.argv[1:]))",
                        child_backend, service, "inner", str(child_weight),
                    ], env=environment,
                       stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.PIPE if capture else None))
                for child in children:
                    output, errors = child.communicate(timeout=15)
                    if capture:
                        sys.stdout.buffer.write(output)
                        sys.stderr.buffer.write(errors)
                result = max(child.returncode for child in children)
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                    child.wait()
            (directory / f"returned-{pid}").touch()
            raise SystemExit(result)
        if role == "inner" and os.environ.get("FAKE_API_FAIL"):
            raise SystemExit(23)
        wait_for(lambda: (directory / "release").exists())
        download = int(os.environ.get("FAKE_API_DOWNLOAD_SIZE", "0"))
        if download:
            (directory / "download").write_bytes(b"d" * download)
        size = int(os.environ.get("FAKE_API_OUTPUT_SIZE", "0"))
        stdout_size = int(os.environ.get("FAKE_API_STDOUT_SIZE", str(size)))
        stderr_size = int(os.environ.get("FAKE_API_STDERR_SIZE", str(size)))
        if stdout_size or stderr_size:
            sys.stdout.write("x" * stdout_size)
            sys.stderr.write("y" * stderr_size)
            sys.stdout.flush()
            sys.stderr.flush()
        else:
            print("fake-complete")
        if os.environ.get("FAKE_API_HOLD_AFTER_OUTPUT"):
            time.sleep(2)
            (directory / "output-complete").touch()
        """
    )
    for name in ("aws", "kubectl"):
        path = tools / name
        path.write_text(executable)
        path.chmod(0o755)
        assert path.is_file() and not path.is_symlink()
        assert shutil.which(name) == str(path)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("FAKE_API_WORK", str(work))
    monkeypatch.setenv("FAKE_API_LEVELS", "1")
    monkeypatch.setenv(
        "GPU_FAULT_DEPLOY_DEADLINE_MONOTONIC", str(time.monotonic() + 18)
    )
    return work


def wait_for_fake_work(work, role, count, futures):
    deadline = time.monotonic() + 10
    while len(list(work.glob(f"{role}-*.json"))) < count:
        for future in futures:
            if future.done():
                result = future.result()
                assert result.returncode == 0, result.stderr
        assert time.monotonic() < deadline, "nested fake CLI admission deadlocked"
        time.sleep(0.01)
    return [json.loads(path.read_text()) for path in work.glob(f"{role}-*.json")]


@pytest.mark.allows_cluster_binaries("aws")
@pytest.mark.parametrize(
    ("outer_weight", "child_weight"), [(1, 1), (4, 1), (4, 4), (1, 4)]
)
def test_saturated_aws_commands_delegate_to_nested_fake_cli(
    recursive_tools, tools, monkeypatch, outer_weight, child_weight
):
    count = budget.LIMITS["aws"] // outer_weight
    monkeypatch.setenv("FAKE_API_OUTERS", str(count))
    monkeypatch.setenv("FAKE_API_CHILD_WEIGHT", str(child_weight))
    with budget.deployment_api_budget():
        assert budget.resolve_tool("aws") == str(tools / "aws")
        with ThreadPoolExecutor(max_workers=count) as pool:
            futures = [
                pool.submit(
                    run_command,
                    [
                        "aws",
                        "s3" if outer_weight == 4 else "sts",
                        "outer",
                        str(outer_weight),
                    ],
                    timeout_seconds=20,
                )
                for _ in range(count)
            ]
            try:
                active_count = 8 // max(outer_weight, child_weight)
                inner = wait_for_fake_work(
                    recursive_tools, "inner", active_count, futures
                )
                assert len(inner) == active_count
                for path in recursive_tools.glob("outer-*.json"):
                    pid = json.loads(path.read_text())["pid"]
                    threads = list(Path(f"/proc/{pid}/task").iterdir())
                    assert len(threads) >= 2
                    assert all(
                        (thread / "stat").read_text().rsplit(")", 1)[1].split()[0]
                        == "T"
                        for thread in threads
                    ), "a parent still had runnable CLI threads during delegation"
                with sqlite3.connect(budget.budget_root() / "budget.sqlite3") as db:
                    active = db.execute(
                        "SELECT weight FROM leases WHERE state='active'"
                    ).fetchall()
                    assert active == [(max(outer_weight, child_weight),)] * active_count
                    assert (
                        db.execute(
                            "SELECT COUNT(*) FROM leases WHERE state='parked'"
                        ).fetchone()[0]
                        == count
                    )
            finally:
                (recursive_tools / "release").touch()
            for future in futures:
                result = future.result()
                assert result.returncode == 0, result.stderr
        stats = budget.statistics()
        assert stats["backends"]["aws"]["commands"] == count * 2
        assert stats["backends"]["aws"]["finished_commands"] == count * 2
        assert stats["peak_admitted_weight"]["aws"] == 8


@pytest.mark.allows_cluster_binaries("aws")
@pytest.mark.parametrize("weight", [1, 4])
def test_independent_fake_commands_still_fill_but_cannot_exceed_global_aws_budget(
    recursive_tools, tools, weight
):
    capacity = 8 // weight
    with budget.deployment_api_budget():
        assert budget.resolve_tool("aws") == str(tools / "aws")
        with ThreadPoolExecutor(max_workers=capacity + 2) as pool:
            futures = [
                pool.submit(
                    run_command,
                    ["aws", "s3" if weight == 4 else "sts", "independent", str(weight)],
                    timeout_seconds=20,
                )
                for _ in range(capacity + 2)
            ]
            try:
                wait_for_fake_work(recursive_tools, "independent", capacity, futures)
                time.sleep(0.2)
                assert len(list(recursive_tools.glob("independent-*.json"))) == capacity
            finally:
                (recursive_tools / "release").touch()
            for future in futures:
                result = future.result()
                assert result.returncode == 0, result.stderr
        assert budget.statistics()["peak_admitted_weight"]["aws"] == 8


@pytest.mark.parametrize(("backend", "capacity"), [("kubectl", 8), ("http", 4)])
def test_other_backend_limits_are_unchanged(tools, backend, capacity):
    release = threading.Event()
    entered = [threading.Event() for _ in range(capacity + 1)]

    def call(index):
        with budget.api_slot(backend):
            entered[index].set()
            assert release.wait(timeout=10), "capacity holder was not released"

    with budget.deployment_api_budget():
        with ThreadPoolExecutor(max_workers=capacity + 1) as pool:
            futures = [pool.submit(call, index) for index in range(capacity)]
            try:
                assert all(event.wait(timeout=5) for event in entered[:capacity]), (
                    "callers did not fill the configured backend capacity"
                )
                futures.append(pool.submit(call, capacity))
                assert not entered[-1].wait(timeout=0.2), (
                    "backend admitted work above its capacity"
                )
            finally:
                release.set()
            for future in futures:
                future.result()
        assert budget.statistics()["peak_admitted_weight"][backend] == capacity


@pytest.mark.allows_cluster_binaries("aws")
@pytest.mark.parametrize("parent_id", ["invalid-parent", "f" * 32])
def test_unvalidated_parent_environment_cannot_start_a_fake_cli(
    recursive_tools, tools, parent_id
):
    with budget.deployment_api_budget():
        assert budget.resolve_tool("aws") == str(tools / "aws")
        result = run_command(
            ["aws", "sts", "independent", "1"],
            environment={**os.environ, budget.PARENT_ENV: parent_id},
            timeout_seconds=5,
        )
        assert result.returncode != 0
        assert parent_id not in result.stderr
        assert not list(recursive_tools.glob("*.json")), (
            "invalid parent metadata started a CLI"
        )
        assert budget.statistics()["backends"] == {}


@pytest.mark.allows_cluster_binaries("aws")
def test_completed_parent_lease_cannot_be_replayed(recursive_tools, tools):
    with budget.deployment_api_budget():
        assert budget.resolve_tool("aws") == str(tools / "aws")
        with budget.api_slot("aws") as parent_id:
            assert parent_id, "test did not establish a parent lease"
        result = run_command(
            ["aws", "sts", "independent", "1"],
            environment={**os.environ, budget.PARENT_ENV: parent_id},
            timeout_seconds=5,
        )
        assert result.returncode != 0
        assert not list(recursive_tools.glob("*.json")), (
            "completed parent lease was replayed"
        )
        assert budget.statistics()["backends"]["aws"]["commands"] == 1


@pytest.mark.allows_cluster_binaries("aws")
def test_a_live_unrelated_parent_cannot_delegate_capacity(recursive_tools, tools):
    with budget.deployment_api_budget():
        assert budget.resolve_tool("aws") == str(tools / "aws")
        with ThreadPoolExecutor(max_workers=1) as pool:
            holder = pool.submit(
                run_command, ["aws", "sts", "holder", "1"], timeout_seconds=20
            )
            try:
                record = wait_for_fake_work(recursive_tools, "holder", 1, [holder])[0]
                result = run_command(
                    ["aws", "sts", "independent", "1"],
                    environment={**os.environ, budget.PARENT_ENV: record["lease"]},
                    timeout_seconds=5,
                )
                assert result.returncode != 0
                assert not list(recursive_tools.glob("independent-*.json")), (
                    "unrelated parent admitted a CLI"
                )
            finally:
                (recursive_tools / "release").touch()
            assert holder.result().returncode == 0
        assert budget.statistics()["backends"]["aws"]["commands"] == 1


@pytest.mark.allows_cluster_binaries("aws")
@pytest.mark.parametrize("field", ["pid_start", "command_start"])
def test_stale_parent_pid_start_identity_is_rejected(
    recursive_tools, tools, monkeypatch, field
):
    monkeypatch.setenv("FAKE_API_OUTERS", "1")
    monkeypatch.setenv("FAKE_API_CHILD_WEIGHT", "1")
    monkeypatch.setenv("FAKE_API_CORRUPT_IDENTITY", field)
    with budget.deployment_api_budget():
        assert budget.resolve_tool("aws") == str(tools / "aws")
        result = run_command(["aws", "sts", "outer", "1"], timeout_seconds=10)
        assert result.returncode != 0
        assert not list(recursive_tools.glob("inner-*.json")), (
            "stale parent identity admitted a nested CLI"
        )
        assert budget.statistics()["backends"]["aws"]["commands"] == 1


@pytest.mark.allows_cluster_binaries("aws")
@pytest.mark.parametrize("unreadable_command", [False, True])
def test_dead_owner_identity_does_not_reap_a_live_cli_reservation(
    recursive_tools, tools, monkeypatch, unreadable_command
):
    with budget.deployment_api_budget():
        assert budget.resolve_tool("aws") == str(tools / "aws")
        with ThreadPoolExecutor(max_workers=1) as pool:
            holder = pool.submit(
                run_command, ["aws", "sts", "holder", "1"], timeout_seconds=20
            )
            try:
                record = wait_for_fake_work(recursive_tools, "holder", 1, [holder])[0]
                path = budget.budget_root() / "budget.sqlite3"
                with sqlite3.connect(path) as db:
                    db.execute(
                        "UPDATE leases SET pid_start='stale-start' WHERE id=?",
                        (record["lease"],),
                    )
                read_text = Path.read_text

                def read_stat(candidate, *args, **kwargs):
                    if candidate == Path(f"/proc/{record['pid']}/stat"):
                        raise PermissionError("procfs identity is unavailable")
                    return read_text(candidate, *args, **kwargs)

                with monkeypatch.context() as probe:
                    if unreadable_command:
                        probe.setattr(Path, "read_text", read_stat)
                    with budget.api_slot("aws"):
                        with sqlite3.connect(path) as db:
                            assert (
                                db.execute(
                                    "SELECT COUNT(*) FROM leases WHERE state='active'"
                                ).fetchone()[0]
                                == 2
                            )
            finally:
                (recursive_tools / "release").touch()
            assert holder.result().returncode == 0


@pytest.mark.allows_cluster_binaries("aws")
def test_parallel_helpers_share_one_delegated_branch(
    recursive_tools, tools, monkeypatch
):
    monkeypatch.setenv("FAKE_API_OUTERS", "1")
    monkeypatch.setenv("FAKE_API_CHILD_WEIGHT", "1")
    monkeypatch.setenv("FAKE_API_FANOUT", "8")
    with budget.deployment_api_budget():
        assert budget.resolve_tool("aws") == str(tools / "aws")
        with ThreadPoolExecutor(max_workers=1) as pool:
            outer = pool.submit(
                run_command, ["aws", "sts", "outer", "1"], timeout_seconds=20
            )
            try:
                wait_for_fake_work(recursive_tools, "inner", 1, [outer])
                time.sleep(0.2)
                assert len(list(recursive_tools.glob("inner-*.json"))) == 1
            finally:
                (recursive_tools / "release").touch()
            result = outer.result()
            assert result.returncode == 0, result.stderr
        assert budget.statistics()["backends"]["aws"]["commands"] == 9
        assert budget.statistics()["peak_admitted_weight"]["aws"] == 1


@pytest.mark.allows_cluster_binaries("aws")
@pytest.mark.parametrize("levels", [3, budget.MAX_COMMAND_DEPTH])
def test_recursive_command_depth_is_bounded(
    recursive_tools, tools, monkeypatch, levels
):
    monkeypatch.setenv("FAKE_API_OUTERS", "1")
    monkeypatch.setenv("FAKE_API_CHILD_WEIGHT", "1")
    monkeypatch.setenv("FAKE_API_LEVELS", str(levels))
    (recursive_tools / "release").touch()
    with budget.deployment_api_budget():
        assert budget.resolve_tool("aws") == str(tools / "aws")
        result = run_command(["aws", "sts", "outer", "1"], timeout_seconds=20)
        assert (result.returncode == 0) == (levels < budget.MAX_COMMAND_DEPTH)
        stats = budget.statistics()
        assert stats["backends"]["aws"]["commands"] == min(
            levels + 1, budget.MAX_COMMAND_DEPTH
        )
        assert stats["peak_admitted_weight"]["aws"] == 1


@pytest.mark.allows_cluster_binaries("aws")
def test_failed_helper_restores_its_parent_before_returning(
    recursive_tools, tools, monkeypatch
):
    monkeypatch.setenv("FAKE_API_OUTERS", "1")
    monkeypatch.setenv("FAKE_API_CHILD_WEIGHT", "1")
    monkeypatch.setenv("FAKE_API_FAIL", "1")
    with budget.deployment_api_budget():
        assert budget.resolve_tool("aws") == str(tools / "aws")
        result = run_command(["aws", "sts", "outer", "1"], timeout_seconds=10)
        assert result.returncode == 23
        assert budget.statistics()["backends"]["aws"]["finished_commands"] == 2
        with sqlite3.connect(budget.budget_root() / "budget.sqlite3") as database:
            assert database.execute("SELECT COUNT(*) FROM leases").fetchone()[0] == 0


@pytest.mark.allows_cluster_binaries("aws")
@pytest.mark.parametrize("consume_return_capacity", [False, True])
def test_cancelled_heavier_helper_never_resumes_an_unfunded_parent(
    recursive_tools, tools, monkeypatch, consume_return_capacity
):
    monkeypatch.setenv("FAKE_API_OUTERS", "1")
    monkeypatch.setenv("FAKE_API_CHILD_WEIGHT", "4")
    with budget.deployment_api_budget():
        assert budget.resolve_tool("aws") == str(tools / "aws")
        with (
            contextlib.ExitStack() as reservations,
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            reservations.enter_context(budget.api_slot("aws", weight=7))
            outer = pool.submit(
                run_command, ["aws", "sts", "outer", "1"], timeout_seconds=20
            )
            record = wait_for_fake_work(recursive_tools, "outer", 1, [outer])[0]
            path = budget.budget_root() / "budget.sqlite3"
            deadline = time.monotonic() + 5
            while True:
                with sqlite3.connect(path) as database:
                    waiter = database.execute(
                        "SELECT child.pid,child.pid_start FROM leases child "
                        "JOIN leases parent ON parent.lent_to=child.id "
                        "WHERE parent.id=? AND parent.state='parked' "
                        "AND child.state='waiting'",
                        (record["lease"],),
                    ).fetchone()
                if waiter:
                    break
                assert time.monotonic() < deadline
                time.sleep(0.01)
            if consume_return_capacity:
                reservations.enter_context(budget.api_slot("aws"))
            pid, start = waiter
            descriptor = os.pidfd_open(pid)
            try:
                assert (
                    Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
                    == start
                )
                signal.pidfd_send_signal(descriptor, signal.SIGTERM)
            finally:
                os.close(descriptor)
            assert outer.result(timeout=5).returncode != 0
            assert not list(recursive_tools.glob("inner-*.json")), (
                "unfunded heavier helper was admitted"
            )
            with sqlite3.connect(path) as database:
                assert (
                    database.execute(
                        "SELECT COUNT(*) FROM leases WHERE command_pid IS NOT NULL "
                        "OR state!='active'"
                    ).fetchone()[0]
                    == 0
                )
        assert budget.statistics()["peak_admitted_weight"]["aws"] == 8


@pytest.mark.allows_cluster_binaries("aws", "kubectl")
def test_aws_recursion_through_kubectl_borrows_the_aws_ancestor(
    recursive_tools, tools, monkeypatch
):
    monkeypatch.setenv("FAKE_API_OUTERS", "8")
    monkeypatch.setenv("FAKE_API_CHILD_WEIGHT", "1")
    monkeypatch.setenv("FAKE_API_LEVELS", "2")
    monkeypatch.setenv("FAKE_API_ALTERNATE", "1")
    with budget.deployment_api_budget():
        assert budget.resolve_tool("aws") == str(tools / "aws")
        assert budget.resolve_tool("kubectl") == str(tools / "kubectl")
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(
                    run_command, ["aws", "sts", "outer", "1"], timeout_seconds=20
                )
                for _ in range(8)
            ]
            try:
                inner = wait_for_fake_work(recursive_tools, "inner", 16, futures)
                assert sum(record["backend"] == "aws" for record in inner) == 8
                assert sum(record["backend"] == "kubectl" for record in inner) == 8
                with sqlite3.connect(budget.budget_root() / "budget.sqlite3") as db:
                    assert dict(
                        db.execute(
                            "SELECT backend,SUM(weight) FROM leases "
                            "WHERE state='active' GROUP BY backend"
                        )
                    ) == {"aws": 8, "kubectl": 8}
                    assert (
                        db.execute(
                            "SELECT COUNT(*) FROM leases WHERE state='parked'"
                        ).fetchone()[0]
                        == 8
                    )
            finally:
                (recursive_tools / "release").touch()
            for future in futures:
                result = future.result()
                assert result.returncode == 0, result.stderr
        stats = budget.statistics()
        assert stats["backends"]["aws"]["finished_commands"] == 16
        assert stats["backends"]["kubectl"]["finished_commands"] == 8
        assert stats["peak_admitted_weight"] == {"aws": 8, "kubectl": 8, "http": 0}


@pytest.mark.parametrize("last_read", ["missing", "denied", "malformed"])
def test_reaping_an_exited_cli_distinguishes_disappearance_from_unknown_reads(
    tools, monkeypatch, last_read
):
    with subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE,
    ) as child:
        path = Path(f"/proc/{child.pid}/stat")
        metadata = path.stat()
        prefix, suffix = path.read_text().rsplit(")", 1)
        fields = suffix.split()
        start = fields[19]
        fields[0] = "Z"
        zombie = prefix + ") " + " ".join(fields)
        child.communicate(b"x", timeout=10)
        assert child.returncode == 0, "the local fixture process must exit cleanly"

    read_text, stat = Path.read_text, Path.stat
    observations = []

    def observed_stat(candidate, *args, **kwargs):
        if candidate == path and len(observations) < 2:
            return metadata
        return stat(candidate, *args, **kwargs)

    def observed_read(candidate, *args, **kwargs):
        if candidate != path:
            return read_text(candidate, *args, **kwargs)
        observations.append(last_read)
        if len(observations) == 1:
            return zombie
        if last_read == "missing":
            raise FileNotFoundError("owned child was reaped between stat and read")
        if last_read == "denied":
            raise PermissionError("owned child identity is unreadable")
        return "malformed procfs observation"

    identifier = "a" * 32
    owner_start = (
        Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]
    )
    with budget.deployment_api_budget():
        root = budget.budget_root()
        assert root is not None, "the regression requires an owned temporary ledger"
        with sqlite3.connect(root / "budget.sqlite3") as database:
            database.execute(
                "INSERT INTO leases(id,backend,weight,pid,pid_start,command_pid,"
                "command_start,state) VALUES(?,'aws',1,?,?,?,?,'active')",
                (identifier, os.getpid(), owner_start, child.pid, start),
            )
            database.execute(
                "INSERT INTO calls(id,phase,backend,wait_seconds) "
                "VALUES(?,'fixture','aws',0)",
                (identifier,),
            )
        with monkeypatch.context() as reads:
            reads.setattr(Path, "stat", observed_stat)
            reads.setattr(Path, "read_text", observed_read)
            # A reaper is a sibling of the lease's owner: an unknown re-read
            # keeps the reservation charged but never fails the reaper's own
            # command (the owner alone refuses to release a running CLI).
            with budget.api_slot("http"):
                assert len(observations) >= 2, (
                    "the fixture must exercise disappearance during the repeated probe"
                )
        with sqlite3.connect(root / "budget.sqlite3") as database:
            held = database.execute("SELECT id FROM leases").fetchall()
            finished = database.execute(
                "SELECT finished FROM calls WHERE id=?", (identifier,)
            ).fetchone()[0]
        assert held == ([] if last_read == "missing" else [(identifier,)]), (
            "only confirmed process disappearance may release the owned reservation"
        )
        assert finished == int(last_read == "missing"), (
            "unreadable or malformed procfs data must not mark a command finished"
        )


@pytest.mark.allows_cluster_binaries("aws")
def test_dead_middle_cli_does_not_resume_an_ancestor_over_an_active_helper(
    recursive_tools, tools, monkeypatch
):
    monkeypatch.setenv("FAKE_API_OUTERS", "1")
    monkeypatch.setenv("FAKE_API_CHILD_WEIGHT", "1")
    monkeypatch.setenv("FAKE_API_LEVELS", "2")
    with budget.deployment_api_budget():
        assert budget.resolve_tool("aws") == str(tools / "aws")
        with ThreadPoolExecutor(max_workers=1) as pool:
            outer = pool.submit(
                run_command, ["aws", "sts", "outer", "1"], timeout_seconds=20
            )
            try:
                wait_for_fake_work(recursive_tools, "inner", 2, [outer])
                record = wait_for_fake_work(recursive_tools, "outer", 1, [outer])[0]
                with sqlite3.connect(budget.budget_root() / "budget.sqlite3") as db:
                    pid, start = db.execute(
                        "SELECT command_pid,command_start FROM leases "
                        "WHERE parent_id=? AND state='parked'",
                        (record["lease"],),
                    ).fetchone()
                descriptor = os.pidfd_open(pid)
                try:
                    assert (
                        Path(f"/proc/{pid}/stat")
                        .read_text()
                        .rsplit(")", 1)[1]
                        .split()[19]
                        == start
                    )
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                finally:
                    os.close(descriptor)
                assert outer.result(timeout=15).returncode != 0
                assert not (recursive_tools / f"returned-{record['pid']}").exists(), (
                    "ancestor resumed before its active helper drained"
                )
            finally:
                (recursive_tools / "release").touch()
        assert budget.statistics()["peak_admitted_weight"]["aws"] == 1


@pytest.mark.allows_cluster_binaries("aws")
@pytest.mark.parametrize("weight", [1, 4])
@pytest.mark.parametrize("size", [1024 * 1024, 4 * 1024 * 1024])
def test_credential_process_output_cannot_fill_a_stopped_callers_pipes(
    recursive_tools, tools, monkeypatch, weight, size
):
    monkeypatch.setenv("FAKE_API_OUTERS", "1")
    monkeypatch.setenv("FAKE_API_CHILD_WEIGHT", str(weight))
    monkeypatch.setenv("FAKE_API_CAPTURE_OUTPUT", "1")
    monkeypatch.setenv("FAKE_API_OUTPUT_SIZE", str(size))
    (recursive_tools / "release").touch()
    with budget.deployment_api_budget():
        assert budget.resolve_tool("aws") == str(tools / "aws")
        result = run_command(
            ["aws", "s3" if weight == 4 else "sts", "outer", str(weight)],
            timeout_seconds=20,
        )
        assert result.returncode == 0, result.stderr[:200]
        assert result.stdout == "x" * size
        assert result.stderr == "y" * size
        child = json.loads(next(recursive_tools.glob("inner-*.json")).read_text())
        assert child["output_modes"] == [0o600, 0o600]
        assert child["output_links"] == [1, 1], "the CLI must write pipes, not spools"
        assert len(child["spools"]) == 2
        assert all(
            spool["mode"] == 0o600 and spool["links"] == 0 for spool in child["spools"]
        ), "nested output spools must be anonymous and owner-private"
        assert budget.statistics()["backends"]["aws"]["finished_commands"] == 2
        assert budget.statistics()["peak_admitted_weight"]["aws"] == weight


@pytest.mark.allows_cluster_binaries("aws")
@pytest.mark.parametrize("stream", ["stdout", "stderr", "combined"])
def test_noisy_helper_cannot_spool_more_than_eight_mib(
    recursive_tools, tools, monkeypatch, stream
):
    limit = 8 * 1024 * 1024
    stdout_size = limit + 1 if stream == "stdout" else 0
    stderr_size = limit + 1 if stream == "stderr" else 0
    if stream == "combined":
        stdout_size, stderr_size = limit // 2, limit // 2 + 1
    monkeypatch.setenv("FAKE_API_OUTERS", "1")
    monkeypatch.setenv("FAKE_API_CHILD_WEIGHT", "1")
    monkeypatch.setenv("FAKE_API_CAPTURE_OUTPUT", "1")
    monkeypatch.setenv("FAKE_API_STDOUT_SIZE", str(stdout_size))
    monkeypatch.setenv("FAKE_API_STDERR_SIZE", str(stderr_size))
    monkeypatch.setenv("FAKE_API_HOLD_AFTER_OUTPUT", "1")
    with budget.deployment_api_budget():
        assert budget.resolve_tool("aws") == str(tools / "aws")
        with ThreadPoolExecutor(max_workers=1) as pool, contextlib.ExitStack() as files:
            outer = pool.submit(
                run_command, ["aws", "sts", "outer", "1"], timeout_seconds=20
            )
            try:
                child = wait_for_fake_work(recursive_tools, "inner", 1, [outer])[0]
                assert len(child["spools"]) == 2
                # Retain the anonymous inodes across cleanup and inspect sizes,
                # never the possibly credential-bearing spool contents.
                spools = [
                    files.enter_context(
                        Path(f"/proc/{child['shim_pid']}/fd/{spool['fd']}").open("rb")
                    )
                    for spool in child["spools"]
                ]
                (recursive_tools / "release").touch()
                result = outer.result(timeout=8)
                sizes = [os.fstat(spool.fileno()).st_size for spool in spools]
                assert 0 < sum(sizes) <= limit
                assert result.returncode != 0
                assert result.stdout == ""
                assert len(result.stderr) < 1024
                assert "x" * 128 not in result.stderr
                assert "y" * 128 not in result.stderr
                assert not (recursive_tools / "output-complete").exists(), (
                    "oversized nested output was allowed to complete"
                )
            finally:
                (recursive_tools / "release").touch()
        with sqlite3.connect(budget.budget_root() / "budget.sqlite3") as database:
            assert database.execute("SELECT COUNT(*) FROM leases").fetchone()[0] == 0
        assert budget.statistics()["backends"]["aws"]["finished_commands"] == 2
        assert budget.statistics()["peak_admitted_weight"]["aws"] == 1


@pytest.mark.allows_cluster_binaries("aws")
def test_nested_output_cap_does_not_limit_regular_file_downloads(
    recursive_tools, tools, monkeypatch
):
    size = 8 * 1024 * 1024 + 1
    monkeypatch.setenv("FAKE_API_OUTERS", "1")
    monkeypatch.setenv("FAKE_API_CHILD_WEIGHT", "4")
    monkeypatch.setenv("FAKE_API_CAPTURE_OUTPUT", "1")
    monkeypatch.setenv("FAKE_API_DOWNLOAD_SIZE", str(size))
    (recursive_tools / "release").touch()
    with budget.deployment_api_budget():
        assert budget.resolve_tool("aws") == str(tools / "aws")
        result = run_command(["aws", "s3", "outer", "4"], timeout_seconds=20)
        assert result.returncode == 0, result.stderr
        assert (recursive_tools / "download").stat().st_size == size
        assert result.stdout == "fake-complete\n"
        assert budget.statistics()["peak_admitted_weight"]["aws"] == 4


@pytest.mark.allows_cluster_binaries("aws")
@pytest.mark.parametrize(
    "version",
    [
        0,
        budget.BUDGET_PROTOCOL_VERSION - 1,
        budget.BUDGET_PROTOCOL_VERSION,
        budget.BUDGET_PROTOCOL_VERSION + 1,
    ],
)
def test_incompatible_inherited_ledger_fails_before_any_cli_or_new_scope(
    recursive_tools, tools, tmp_path, monkeypatch, version
):
    assert shutil.which("aws") == str(tools / "aws")
    root = tmp_path / "inherited"
    root.mkdir(mode=0o700)
    path = root / "budget.sqlite3"
    path.touch(mode=0o600)
    with sqlite3.connect(path) as database:
        database.execute(
            "CREATE TABLE capacity(backend TEXT PRIMARY KEY,next_start REAL,"
            "peak INTEGER,max_weight INTEGER NOT NULL)"
        )
        database.executemany(
            "INSERT INTO capacity VALUES(?,123,7,?)", list(budget.LIMITS.items())
        )
        # A legacy lease shape must fail even with a falsely current version.
        database.execute(
            "CREATE TABLE leases(id TEXT PRIMARY KEY,backend TEXT,weight INTEGER,"
            "pid INTEGER,pid_start TEXT)"
        )
        database.execute(f"PRAGMA user_version={version}")
    original = path.read_bytes()
    monkeypatch.setenv(budget.ROOT_ENV, str(root))
    with pytest.raises(budget.ApiBudgetProtocolError, match="incompatible inherited"):
        with budget.deployment_api_budget():
            pytest.fail("an incompatible inherited budget was silently replaced")
    with pytest.raises(budget.ApiBudgetProtocolError, match="matching deploy-host"):
        run_command(["aws", "sts", "independent", "1"], timeout_seconds=5)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            budget.__file__,
            str(tools / "aws"),
            "sts",
            "independent",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0
    assert "incompatible inherited deployment API budget protocol" in result.stderr
    assert "OperationalError" not in result.stderr
    assert "Traceback" not in result.stderr
    assert not list(recursive_tools.glob("*.json")), (
        "incompatible protocol started a CLI"
    )
    assert os.environ[budget.ROOT_ENV] == str(root)
    assert path.read_bytes() == original
    assert not (root / "bin").exists(), (
        "protocol rejection created a replacement budget"
    )


def test_current_protocol_with_different_owner_limits_is_not_reset(tools):
    with budget.deployment_api_budget():
        root = budget.budget_root()
        path = root / "budget.sqlite3"
        with sqlite3.connect(path) as database:
            database.execute(
                "UPDATE capacity SET max_weight=?,peak=3 WHERE backend='aws'",
                (budget.LIMITS["aws"] - 1,),
            )
        try:
            with pytest.raises(budget.ApiBudgetProtocolError, match="protocol"):
                with budget.deployment_api_budget():
                    pytest.fail("an incompatible owner policy was replaced")
            with sqlite3.connect(path) as database:
                assert database.execute(
                    "SELECT max_weight,peak FROM capacity WHERE backend='aws'"
                ).fetchone() == (budget.LIMITS["aws"] - 1, 3)
            assert os.environ[budget.ROOT_ENV] == str(root)
        finally:
            with sqlite3.connect(path) as database:
                database.execute(
                    "UPDATE capacity SET max_weight=? WHERE backend='aws'",
                    (budget.LIMITS["aws"],),
                )
