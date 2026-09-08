from __future__ import annotations

from ._support import (
    NOW,
    CompletedProcess,
    Event,
    FakeRunner,
    NodeActionExecutor,
    NodeActionStatus,
    Path,
    ThreadPoolExecutor,
    WorkflowOperation,
    _flight_dump_payload,
    _torch_flight_dump,
    _triage_only_executor,
    command,
    envelope,
    executor,
    gzip,
    hashlib,
    heartbeat_reporter_from_environment,
    json,
    no_device_clients,
    node_action_executor,
    os,
    pickle,
    pytest,
    tarfile,
    time,
    timedelta,
)


def test_same_command_is_serialized_while_in_flight(tmp_path) -> None:
    class BlockingRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__()
            self.started = Event()
            self.release = Event()

        def __call__(self, command, **kwargs):
            if not self.started.is_set():
                self.started.set()
                self.release.wait(timeout=5)
            return super().__call__(command, **kwargs)

    runner = BlockingRunner()
    agent = executor(tmp_path, runner)
    signed = envelope(command(WorkflowOperation.RESET_GPU))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(agent.execute, signed)
        assert runner.started.wait(timeout=2), (
            "expected runner.started.wait(timeout=2) to be truthy"
        )
        second = pool.submit(agent.execute, signed)
        runner.release.set()
        results = [first.result(), second.result()]

    assert results[0] == results[1]
    assert sum("--gpu-reset" in item for item in runner.commands) == 1


def test_diagnostic_bundle_records_partial_failures(tmp_path) -> None:
    class DiagnosticRunner:
        def __call__(self, argv, **_):
            if argv[:2] == ["dcgmi", "diag"]:
                return CompletedProcess(argv, 2, stdout="", stderr="DCGM unavailable")
            if argv[0] == "/usr/bin/nvidia-bug-report.sh":
                Path(f"{argv[-1]}.gz").write_bytes(gzip.compress(b"NVIDIA bug report"))
            return CompletedProcess(argv, 0, stdout="captured", stderr="")

    output_dir = tmp_path / "diagnostics"
    agent = node_action_executor(
        tmp_path,
        "diagnostics.db",
        allowed_operations={WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE},
        diagnostic_output_dir=str(output_dir),
        runner=DiagnosticRunner(),
    )

    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
                parameters={"sxid": 20001, "classification": "FATAL"},
            )
        )
    )

    assert result.status is NodeActionStatus.SUCCEEDED
    assert result.details["manifest_summary"] == {
        "capture_count": 8,
        "failed_capture_count": 1,
        "diagnostic_reason": None,
        "capture_process_state": False,
    }
    archive = next(output_dir.glob("*.tar.gz"))
    assert result.details["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    with tarfile.open(archive) as bundle:
        manifest = json.load(bundle.extractfile("diagnostics/manifest.json"))
        names = set(bundle.getnames())
        bug_report = bundle.extractfile("diagnostics/nvidia-bug-report.log.gz").read()
    dcgm = next(
        item for item in manifest["captures"] if item["file"] == "dcgm-diag.json"
    )
    assert dcgm["returncode"] == 2
    assert manifest["fault_context"] == {"sxid": 20001, "classification": "FATAL"}
    assert any(
        item["file"] == "nvidia-bug-report.log.gz" for item in manifest["captures"]
    ), (
        'expected any( item["file"] == "nvidia-bug-report.log.gz" for item in manifest["captures"] ) to be truthy'
    )
    bug_report_capture = next(
        item
        for item in manifest["captures"]
        if item["file"] == "nvidia-bug-report.log.gz"
    )
    assert bug_report_capture["command"][-1].endswith("nvidia-bug-report.log"), (
        'expected bug_report_capture["command"][-1].endswith("nvidia-bug-report.log") to be truthy'
    )
    assert gzip.decompress(bug_report) == b"NVIDIA bug report"
    assert "diagnostics/nvidia-bug-report.log.gz.gz" not in names
    assert not any(path.is_dir() for path in output_dir.iterdir()), (
        "expected any(path.is_dir() for path in output_dir.iterdir()) to be falsy"
    )


def test_diagnostic_archive_retention_prunes_age_and_count(tmp_path) -> None:
    output_dir = tmp_path / "diagnostics"
    output_dir.mkdir()
    old = output_dir / "gpu-diagnostic-old.tar.gz"
    middle = output_dir / "gpu-diagnostic-middle.tar.gz"
    newest = output_dir / "gpu-diagnostic-newest.tar.gz"
    for path in (old, middle, newest):
        path.write_bytes(b"archive")
    os.utime(
        old,
        (
            (NOW - timedelta(hours=2)).timestamp(),
            (NOW - timedelta(hours=2)).timestamp(),
        ),
    )
    os.utime(
        middle,
        (
            (NOW - timedelta(minutes=2)).timestamp(),
            (NOW - timedelta(minutes=2)).timestamp(),
        ),
    )
    os.utime(
        newest,
        (
            (NOW - timedelta(minutes=1)).timestamp(),
            (NOW - timedelta(minutes=1)).timestamp(),
        ),
    )
    agent = node_action_executor(
        tmp_path,
        "retention.db",
        allowed_operations={WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE},
        diagnostic_output_dir=str(output_dir),
        diagnostic_retention_seconds=3600,
        diagnostic_max_archives=1,
    )

    removed = agent._cleanup_diagnostic_archives(now=NOW)

    assert set(removed) == {old.name, middle.name}
    assert newest.exists(), "expected newest.exists() to be truthy"


def test_hung_diagnostic_bundle_captures_gpu_process_state(tmp_path) -> None:
    pid = os.getpid()
    infiniband_root = tmp_path / "infiniband"
    port = infiniband_root / "efa_0" / "ports" / "1"
    (port / "counters").mkdir(parents=True)
    (port / "hw_counters").mkdir()
    (infiniband_root / "efa_0" / "device" / "net" / "eth0").mkdir(parents=True)
    (port / "state").write_text("4: ACTIVE\n")
    (port / "phys_state").write_text("5: LinkUp\n")
    (port / "counters" / "port_rcv_errors").write_text("3\n")
    (port / "hw_counters" / "rx_bytes").write_text("4096\n")
    calls = []

    class HungRunner:
        def __call__(self, argv, **_):
            calls.append(argv)
            if "--query-compute-apps=pid,gpu_uuid,process_name" in argv:
                return CompletedProcess(
                    argv,
                    0,
                    stdout=(f"{pid}, GPU-a, python\n999999, GPU-other, unrelated\n"),
                    stderr="",
                )
            return CompletedProcess(argv, 0, stdout="captured", stderr="")

    output_dir = tmp_path / "diagnostics"
    agent = node_action_executor(
        tmp_path,
        "diagnostics.db",
        allowed_operations={WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE},
        diagnostic_output_dir=str(output_dir),
        infiniband_root=str(infiniband_root),
        expand_python_cgroup_processes=False,
        runner=HungRunner(),
        sleep=lambda _seconds: None,
    )

    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
                parameters={
                    "diagnostic_reason": ("EFA_TRAFFIC_HUNG_SUSPECTED"),
                    "capture_process_state": True,
                    "strace_duration_seconds": 1,
                    "strace_sample_count": 3,
                    "strace_sample_interval_seconds": 0,
                    "job_id": "job-a",
                    "attempt_id": "attempt-a",
                },
            )
        )
    )

    assert result.status is NodeActionStatus.SUCCEEDED
    archive = next(output_dir.glob("*.tar.gz"))
    with tarfile.open(archive) as bundle:
        manifest = json.load(bundle.extractfile("diagnostics/manifest.json"))
        names = set(bundle.getnames())
        process_csv = (
            bundle.extractfile("diagnostics/nvidia-compute-processes.csv")
            .read()
            .decode()
        )
        rdma_link = bundle.extractfile("diagnostics/rdma-link-show.txt").read().decode()
        ethtool = (
            bundle.extractfile("diagnostics/ethtool-eth0-statistics.txt")
            .read()
            .decode()
        )
        infiniband = json.load(
            bundle.extractfile("diagnostics/infiniband-counters.json")
        )
    assert manifest["gpu_processes"][0]["pid"] == pid
    assert manifest["target_gpu_uuids"] == ["GPU-a"]
    assert len(manifest["gpu_processes"]) == 1
    assert "GPU-a" in process_csv
    assert "GPU-other" not in process_csv
    assert rdma_link == "captured"
    assert ethtool == "captured"
    assert (
        infiniband["devices"]["efa_0"]["ports"]["1"]["counters"]["port_rcv_errors"]
        == "3"
    )
    assert (
        infiniband["devices"]["efa_0"]["ports"]["1"]["hw_counters"]["rx_bytes"]
        == "4096"
    )
    assert ["rdma", "link", "show"] in calls
    assert ["ethtool", "-S", "eth0"] in calls
    assert f"diagnostics/proc-{pid}-status.txt" in names
    assert manifest["strace_sampling"] == {
        "sample_count": 3,
        "duration_seconds_per_sample": 1,
        "interval_seconds": 0,
    }
    assert manifest["python_stack_sampling"] == {
        "sample_count": 3,
        "interval_seconds": 0,
        "command_timeout_seconds": 10,
        "process_count": 1,
        "mode": "round_parallel",
    }
    traces = [
        item
        for item in manifest["captures"]
        if item["file"].startswith(f"strace-{pid}-sample-")
    ]
    assert len(traces) == 3
    assert [item["sample_index"] for item in traces] == [1, 2, 3]
    assert all(item["sample_count"] == 3 for item in traces), (
        'expected all(item["sample_count"] == 3 for item in traces) to be truthy'
    )
    assert all(item["started_at"] for item in traces), (
        'expected all(item["started_at"] for item in traces) to be truthy'
    )
    assert all(item["completed_at"] for item in traces), (
        'expected all(item["completed_at"] for item in traces) to be truthy'
    )
    for sample_index in range(1, 4):
        suffix = f"sample-{sample_index:02d}.txt"
        assert f"diagnostics/proc-{pid}-stack-{suffix}" in names
        assert f"diagnostics/proc-{pid}-wchan-{suffix}" in names
    strace_calls = [argv for argv in calls if "strace" in argv]
    assert len(strace_calls) == 3
    python_stack_captures = [
        item for item in manifest["captures"] if item.get("tool") == "py-spy"
    ]
    assert len(python_stack_captures) == 3
    assert [item["sample_index"] for item in python_stack_captures] == [1, 2, 3]
    for sample_index in range(1, 4):
        assert f"diagnostics/python-stack-{pid}-sample-{sample_index:02d}.txt" in names


def test_hung_triage_collects_bounded_read_only_rank_signals(
    tmp_path, monkeypatch
) -> None:
    proc_root = tmp_path / "proc"
    pid = 100
    proc_dir = proc_root / str(pid)
    task_dir = proc_dir / "task" / str(pid)
    root_tmp = proc_dir / "root" / "tmp"
    task_dir.mkdir(parents=True)
    root_tmp.mkdir(parents=True)
    (proc_dir / "environ").write_bytes(
        b"RANK=7\0LOCAL_RANK=0\0"
        b"TORCH_NCCL_DEBUG_INFO_PIPE_FILE=/tmp/nccl_pipe_\0"
        b"TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/nccl_trace_\0"
    )
    (proc_dir / "wchan").write_text("futex_wait\n")
    (proc_dir / "status").write_text("voluntary_ctxt_switches:\t10\n")
    (proc_dir / "comm").write_text("python\n")

    def write_proc_state(cpu_ticks: int, switches: int) -> None:
        fields = ["S", *(["0"] * 50)]
        fields[11] = str(cpu_ticks)
        fields[12] = "0"
        stat_line = f"{pid} (python) " + " ".join(fields) + "\n"
        (proc_dir / "stat").write_text(stat_line)
        (task_dir / "stat").write_text(stat_line)
        (proc_dir / "status").write_text(f"voluntary_ctxt_switches:\t{switches}\n")

    write_proc_state(10, 10)
    pipe = root_tmp / "nccl_pipe_7.pipe"
    pipe.write_text("")
    monkeypatch.setattr(
        "gpu_fault.node_agent.operations.flight_recorder.stat.S_ISFIFO",
        lambda _mode: True,
    )
    efa_counter = (
        tmp_path / "infiniband" / "efa_0" / "ports" / "1" / "hw_counters" / "rx_bytes"
    )
    efa_counter.parent.mkdir(parents=True)
    efa_counter.write_text("100\n")

    class TriageRunner:
        def __call__(self, argv, **_):
            if "--query-compute-apps=pid,gpu_uuid,process_name" in argv:
                return CompletedProcess(
                    argv, 0, stdout=f"{pid}, GPU-a, python\n", stderr=""
                )
            if (
                "--query-gpu=uuid,utilization.gpu,utilization.memory,clocks_throttle_reasons.active"
                in argv
            ):
                return CompletedProcess(argv, 0, stdout="GPU-a, 0, 0, 0x0\n", stderr="")
            if argv and argv[0].endswith("py-spy"):
                return CompletedProcess(
                    argv,
                    0,
                    stdout=(
                        "all_reduce (/workspace/train.py:42)\n"
                        "wait (/torch/distributed_c10d.py:100)\n"
                    ),
                    stderr="",
                )
            return CompletedProcess(argv, 0, stdout="", stderr="")

    def sample_delay(_seconds: float) -> None:
        write_proc_state(25, 12)
        efa_counter.write_text("150\n")
        (root_tmp / "nccl_trace_7.json").write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "pg_name_": "default",
                            "collective_seq_id_": 12,
                            "profiling_name_": "nccl:all_reduce",
                            "time_discovered_started_": None,
                            "time_discovered_completed_": None,
                            "time_created_": "2026-08-15T12:00:00Z",
                        }
                    ]
                }
            )
        )

    agent = node_action_executor(
        tmp_path,
        "triage.db",
        allowed_operations={WorkflowOperation.COLLECT_HUNG_TRIAGE},
        proc_root=str(proc_root),
        infiniband_root=str(tmp_path / "infiniband"),
        python_stack_tool="/opt/gpu-fault/venv/bin/py-spy",
        runner=TriageRunner(),
        sleep=sample_delay,
    )

    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.COLLECT_HUNG_TRIAGE,
                parameters={"triage_timeout_seconds": 10},
            )
        )
    )

    assert result.status is NodeActionStatus.SUCCEEDED
    assert result.details["read_only"] is True
    assert result.details["elapsed_seconds"] < 10
    rank = result.details["ranks"][0]
    assert rank["rank"] == 7
    assert rank["flight_recorder"]["last_entry"]["collective_seq_id"] == 12
    assert rank["python_stack"]["signature"]
    assert rank["python_stack"]["frames"][0] == ("/workspace/train.py:all_reduce")
    assert rank["proc"]["cpu_ticks_delta"] == 15
    assert result.details["efa"]["total_delta"] == 50


def test_flight_recorder_dump_is_awaited_within_triage_budget(tmp_path) -> None:
    dump = tmp_path / "nccl_trace_rank_3"
    sleeps: list[float] = []

    def delayed_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 4:
            # PyTorch dumps from its own monitor thread; on a real
            # p5en hang that took 3.4s, far past the single check the
            # collector used to do.
            dump.write_text(_flight_dump_payload())

    agent = _triage_only_executor(tmp_path, delayed_sleep)
    requests = {
        321: {
            "status": "triggered",
            "pipe": "/tmp/nccl_pipe_3.pipe",
            "dump_candidates": [str(dump)],
            "previous_mtimes": {},
        }
    }
    ranks = [{"pid": 321, "flight_recorder": {"status": "dump_missing"}}]

    waited = agent._await_flight_dumps(requests, ranks, time.monotonic() + 30)

    assert len(sleeps) == 4
    assert ranks[0]["flight_recorder"]["status"] == "dumped"
    assert ranks[0]["flight_recorder"]["last_entry"]["collective_seq_id"] == 12
    assert waited >= 0


def test_flight_recorder_wait_stops_at_the_triage_deadline(tmp_path) -> None:
    sleeps: list[float] = []
    agent = _triage_only_executor(tmp_path, sleeps.append)
    ranks = [{"pid": 321, "flight_recorder": {"status": "dump_missing"}}]

    agent._await_flight_dumps(
        {
            321: {
                "status": "triggered",
                "dump_candidates": [str(tmp_path / "absent")],
                "previous_mtimes": {},
            }
        },
        ranks,
        time.monotonic() - 1,
    )

    assert sleeps == []
    assert ranks[0]["flight_recorder"]["status"] == "dump_missing"


def test_flight_recorder_reads_the_pickled_dump_torch_writes(tmp_path) -> None:
    dump = tmp_path / "nccl_trace_rank_8"
    dump.write_bytes(pickle.dumps(_torch_flight_dump(21, "scheduled"), protocol=2))
    agent = _triage_only_executor(tmp_path, lambda _seconds: None)

    summary = agent._finish_flight_recorder(
        {
            "status": "triggered",
            "pipe": "/tmp/nccl_pipe_8.pipe",
            "dump_candidates": [str(dump)],
            "previous_mtimes": {},
        }
    )

    assert "parse_error" not in summary
    assert summary["status"] == "dumped"
    # torch records the group as ``process_group: (uid, desc)``; a real
    # dump has neither pg_name nor group_name.
    assert summary["last_entry"]["pg_name"] == "0:default_pg"
    assert summary["last_entry"]["collective_seq_id"] == 21
    assert summary["last_entry"]["state"] == "scheduled"
    assert summary["unfinished_entry_count"] == 1


def test_flight_recorder_dump_cannot_execute_planted_code(tmp_path) -> None:
    dump = tmp_path / "nccl_trace_rank_9"
    dump.write_bytes(
        b"\x80\x04\x95\x00\x00\x00\x00\x00\x00\x00\x00"
        + pickle.dumps(os.system, protocol=2)
    )
    agent = _triage_only_executor(tmp_path, lambda _seconds: None)

    summary = agent._finish_flight_recorder(
        {"status": "triggered", "dump_candidates": [str(dump)], "previous_mtimes": {}}
    )

    assert summary["status"] == "dumped"
    assert "refusing to resolve" in summary["parse_error"]


def test_stack_summary_drops_pyspy_banners_and_sees_device_sync() -> None:
    dump = (
        "Process 1394129: /opt/venv/bin/python3 -u train.py\n"
        "Python v3.12.13 (/usr/bin/python3.12)\n"
        "\n"
        'Thread 1394129 (idle): "MainThread"\n'
        "    synchronize (torch/cuda/__init__.py:1064)\n"
        "    <module> (/opt/gpu-fault-test/train.py:63)\n"
    )
    other_rank = dump.replace("1394129", "1394131")

    summary = NodeActionExecutor._stack_summary(dump)

    assert summary["frames"] == [
        "torch/cuda/__init__.py:synchronize",
        "/opt/gpu-fault-test/train.py:<module>",
    ]
    # A rank blocked in the device sync of an unfinished collective is a
    # collective hang, and the pid in the py-spy banner must not make
    # every rank's signature unique.
    assert summary["collective_frames"] is True
    assert (
        NodeActionExecutor._stack_summary(other_rank)["signature"]
        == summary["signature"]
    )


def test_hung_bundle_prefers_exact_target_pids(tmp_path) -> None:
    proc_root = tmp_path / "proc"
    pid = 101
    proc_dir = proc_root / str(pid)
    proc_dir.mkdir(parents=True)
    (proc_dir / "comm").write_text("python\n")
    (proc_dir / "cmdline").write_bytes(b"python\0train.py\0")
    (proc_dir / "status").write_text("State:\tS\n")
    (proc_dir / "stack").write_text("stack\n")
    (proc_dir / "wchan").write_text("wait\n")
    calls = []

    class ExactRunner:
        def __call__(self, argv, **_):
            calls.append(argv)
            if "--query-compute-apps=pid,gpu_uuid,process_name" in argv:
                raise AssertionError("exact target PID must not be rediscovered")
            return CompletedProcess(argv, 0, stdout="captured", stderr="")

    work_dir = tmp_path / "bundle"
    work_dir.mkdir()
    manifest = {"captures": []}
    agent = node_action_executor(
        tmp_path,
        "exact.db",
        allowed_operations={WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE},
        proc_root=str(proc_root),
        python_stack_tool="",
        runner=ExactRunner(),
        sleep=lambda _seconds: None,
    )

    agent._capture_hung_process_state(
        work_dir,
        command(
            WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
            parameters={
                "target_pids_by_node": {"node-a": [pid]},
                "target_gpu_uuids_by_pid_by_node": {"node-a": {str(pid): "GPU-a"}},
                "expand_python_cgroup_processes": False,
                "max_processes": 1,
                "strace_sample_count": 1,
                "strace_duration_seconds": 1,
                "strace_sample_interval_seconds": 0,
            },
        ),
        manifest,
    )

    assert manifest["gpu_processes"] == [
        {
            "pid": pid,
            "gpu_uuid": "GPU-a",
            "process_name": "python",
            "association": "hung_triage_target",
        }
    ]
    assert manifest["strace_sampling"]["sample_count"] == 1
    assert not any(
        "--query-compute-apps=pid,gpu_uuid,process_name" in argv for argv in calls
    ), (
        'expected any( "--query-compute-apps=pid,gpu_uuid,process_name" in argv for argv in calls ) to be falsy'
    )


def test_hung_pyspy_samples_all_python_processes_by_round_before_strace(
    tmp_path,
) -> None:
    proc_root = tmp_path / "proc"
    pids = (100, 101)
    for pid in pids:
        directory = proc_root / str(pid)
        directory.mkdir(parents=True)
        (directory / "cmdline").write_bytes(b"python\0train.py\0")
        (directory / "status").write_text("State:\tS\n")
        (directory / "stack").write_text("kernel stack\n")
        (directory / "wchan").write_text("futex_wait_queue\n")

    events = []
    sleep_calls = []

    class RoundRunner:
        def __call__(self, argv, **_):
            if "--query-compute-apps=pid,gpu_uuid,process_name" in argv:
                return CompletedProcess(
                    argv,
                    0,
                    stdout="".join(f"{pid}, GPU-a, python\n" for pid in pids),
                    stderr="",
                )
            if "py-spy" in argv[0]:
                events.append(("py-spy", int(argv[-1])))
            elif "strace" in argv:
                events.append(("strace", int(argv[argv.index("-p") + 1])))
            return CompletedProcess(argv, 0, stdout="captured", stderr="")

    work_dir = tmp_path / "bundle"
    work_dir.mkdir()
    manifest = {"captures": []}
    agent = node_action_executor(
        tmp_path,
        "agent.db",
        allowed_operations={WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE},
        proc_root=str(proc_root),
        expand_python_cgroup_processes=False,
        runner=RoundRunner(),
        sleep=lambda seconds: sleep_calls.append(seconds),
    )

    agent._capture_hung_process_state(
        work_dir,
        command(
            WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
            parameters={
                "strace_sample_count": 2,
                "strace_duration_seconds": 1,
                "strace_sample_interval_seconds": 0,
                "pyspy_sample_count": 3,
                "pyspy_sample_interval_seconds": 1,
                "pyspy_timeout_seconds": 7,
            },
        ),
        manifest,
    )

    python_captures = [
        item for item in manifest["captures"] if item.get("tool") == "py-spy"
    ]
    assert [(item["sample_index"], item["pid"]) for item in python_captures] == [
        (1, 100),
        (1, 101),
        (2, 100),
        (2, 101),
        (3, 100),
        (3, 101),
    ]
    assert manifest["python_stack_sampling"] == {
        "sample_count": 3,
        "interval_seconds": 1,
        "command_timeout_seconds": 7,
        "process_count": 2,
        "mode": "round_parallel",
    }
    assert sleep_calls == [1, 1]
    first_strace = next(
        index for index, event in enumerate(events) if event[0] == "strace"
    )
    assert all(event[0] == "py-spy" for event in events[:first_strace]), (
        'expected all(event[0] == "py-spy" for event in events[:first_strace]) to be truthy'
    )
    assert events[:first_strace].count(("py-spy", 100)) == 3
    assert events[:first_strace].count(("py-spy", 101)) == 3


def test_hung_diagnostics_expand_python_processes_in_training_cgroup(tmp_path) -> None:
    proc_root = tmp_path / "proc"
    for pid, cgroup, comm, cmdline in (
        (100, "/kubepods/pod-a/container-a", "python", "python rank.py"),
        (
            101,
            "/kubepods/pod-a/container-a",
            "python3",
            "python3 -m torch.distributed.run",
        ),
        (102, "/kubepods/pod-a/container-a", "bash", "bash launch.sh"),
        (103, "/kubepods/pod-b/container-b", "python", "python other.py"),
    ):
        directory = proc_root / str(pid)
        directory.mkdir(parents=True)
        (directory / "cgroup").write_text(f"0::{cgroup}\n")
        (directory / "comm").write_text(f"{comm}\n")
        (directory / "cmdline").write_bytes(cmdline.replace(" ", "\0").encode())
    agent = node_action_executor(
        tmp_path,
        "agent.db",
        allowed_operations={WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE},
        proc_root=str(proc_root),
        now=None,
    )

    expanded = agent._expand_python_processes_in_cgroups(
        [
            {
                "pid": 100,
                "gpu_uuid": "GPU-a",
                "process_name": "python",
                "association": "gpu_compute",
            }
        ]
    )

    assert [item["pid"] for item in expanded] == [100, 101]
    assert expanded[1]["association"] == ("training_container_cgroup")
    assert expanded[1]["source_gpu_pids"] == [100]
    assert expanded[1]["gpu_uuid"] == "GPU-a"


@pytest.mark.parametrize(
    ("status", "returncode", "expected"),
    [
        ("Pass", 0, "PASS"),
        ("Warn", 0, "WARN"),
        ("Fail", 0, "FAIL"),
        ("Pass", 2, "FAIL"),
    ],
)
def test_quick_dcgm_diagnostic_returns_structured_outcome(
    tmp_path, status, returncode, expected
) -> None:
    calls = []

    def runner(argv, **_kwargs):
        calls.append(argv)
        return CompletedProcess(
            argv,
            returncode,
            stdout=json.dumps(
                {
                    "DCGM GPU Diagnostic": {
                        "tests": [
                            {
                                "name": "Deployment",
                                "results": [{"gpu_ids": "0", "status": status}],
                            }
                        ]
                    }
                }
            ),
            stderr="",
        )

    output_dir = tmp_path / "quick-diagnostics"
    agent = node_action_executor(
        tmp_path,
        "quick.db",
        allowed_operations={WorkflowOperation.RUN_DCGM_DIAGNOSTIC},
        diagnostic_output_dir=str(output_dir),
        runner=runner,
    )
    request = envelope(command(WorkflowOperation.RUN_DCGM_DIAGNOSTIC))

    first = agent.execute(request)
    duplicate = agent.execute(request)

    assert first.status is NodeActionStatus.SUCCEEDED
    assert first.details["diagnostic_outcome"] == expected
    assert first.details["evidence_ref"].startswith("file://"), (
        'expected first.details["evidence_ref"].startswith("file://") to be truthy'
    )
    assert first.details["sha256"]
    assert duplicate == first
    assert calls == [["dcgmi", "diag", "-r", "1", "-j"]]


def test_quick_dcgm_diagnostic_invalid_json_is_inconclusive(tmp_path) -> None:
    def runner(argv, **_kwargs):
        return CompletedProcess(argv, 0, stdout="not-json", stderr="")

    agent = node_action_executor(
        tmp_path,
        "invalid.db",
        allowed_operations={WorkflowOperation.RUN_DCGM_DIAGNOSTIC},
        diagnostic_output_dir=str(tmp_path / "quick-invalid"),
        runner=runner,
    )

    result = agent.execute(envelope(command(WorkflowOperation.RUN_DCGM_DIAGNOSTIC)))

    assert result.status is NodeActionStatus.SUCCEEDED
    assert result.details["diagnostic_outcome"] == "INCONCLUSIVE"
    assert result.details["parse_error"]
    assert (
        result.details["recommended_actions"][0]["action_code"]
        == "DCGM_EXECUTION_REVIEW"
    )


def test_quick_dcgm_diagnostic_extracts_fixed_admin_guidance(tmp_path) -> None:
    def runner(argv, **_kwargs):
        return CompletedProcess(
            argv,
            0,
            stdout=json.dumps(
                {
                    "DCGM GPU Diagnostic": {
                        "tests": [
                            {
                                "name": "PCIe",
                                "results": [
                                    {
                                        "gpu_ids": "0",
                                        "status": "Fail",
                                        "warnings": [
                                            {
                                                "warning": (
                                                    "PCIe replay and AER "
                                                    "errors detected"
                                                ),
                                                "error_id": {"code": 17},
                                            }
                                        ],
                                    }
                                ],
                            }
                        ]
                    }
                }
            ),
            stderr="",
        )

    agent = node_action_executor(
        tmp_path,
        "guidance.db",
        allowed_operations={WorkflowOperation.RUN_DCGM_DIAGNOSTIC},
        diagnostic_output_dir=str(tmp_path / "quick-guidance"),
        runner=runner,
    )

    result = agent.execute(envelope(command(WorkflowOperation.RUN_DCGM_DIAGNOSTIC)))

    finding = result.details["diagnostic_findings"][0]
    assert finding["test_name"] == "PCIe"
    assert finding["status"] == "FAIL"
    assert finding["entities"] == ["0"]
    assert finding["messages"][0] == ("PCIe replay and AER errors detected")
    assert (
        result.details["recommended_actions"][0]["action_code"] == "PCIE_AER_INSPECTION"
    )


def test_quick_dcgm_config_severity_failure_is_warn(tmp_path) -> None:
    """Persistence mode disabled fails every GPU with CONFIG severity
    and makes dcgmi exit non-zero. Grading that as FAIL drains every
    node in the fleet, so it must come back as WARN."""

    def runner(argv, **_kwargs):
        return CompletedProcess(
            argv,
            226,
            stdout=json.dumps(
                {
                    "DCGM Diagnostic": {
                        "test_categories": [
                            {
                                "category": "Deployment",
                                "tests": [
                                    {
                                        "name": "software",
                                        "results": [
                                            {
                                                "entity_group": "GPU",
                                                "entity_id": gpu,
                                                "status": "Fail",
                                                "warnings": [
                                                    {
                                                        "error_category": 8,
                                                        "error_id": 29,
                                                        "error_severity": 5,
                                                        "warning": (
                                                            "Persistence Mode: disabled"
                                                        ),
                                                    }
                                                ],
                                            }
                                            for gpu in range(8)
                                        ],
                                        "test_summary": {
                                            "status": "Fail",
                                            "warnings": [
                                                {
                                                    "error_category": 8,
                                                    "error_id": 29,
                                                    "error_severity": 5,
                                                    "warning": (
                                                        "Persistence Mode: disabled"
                                                    ),
                                                }
                                            ],
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                }
            ),
            stderr="",
        )

    agent = node_action_executor(
        tmp_path,
        "config.db",
        allowed_operations={WorkflowOperation.RUN_DCGM_DIAGNOSTIC},
        diagnostic_output_dir=str(tmp_path / "quick-config"),
        runner=runner,
    )

    result = agent.execute(envelope(command(WorkflowOperation.RUN_DCGM_DIAGNOSTIC)))

    assert result.details["diagnostic_outcome"] == "WARN"
    assert result.details["configuration_only_failures"] is True
    assert result.details["returncode"] == 226
    assert result.details["status_counts"] == {"FAIL": 9}
    assert [item["action_code"] for item in result.details["recommended_actions"]] == [
        "GPU_HOST_CONFIG_REMEDIATION"
    ]


def test_field_diagnostic_runs_pinned_command_for_exact_link(tmp_path) -> None:
    executable = tmp_path / "fielddiag"
    executable.write_text("#!/bin/sh\n", encoding="ascii")
    executable.chmod(0o755)
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    runner = FakeRunner()
    agent = node_action_executor(
        tmp_path,
        "fielddiag.db",
        allowed_operations={WorkflowOperation.RUN_NVLINK74_WORKFLOW},
        field_diagnostic_enabled=True,
        field_diagnostic_command=(
            str(executable),
            "--gpu",
            "{gpu_uuid}",
            "--link",
            "{link_id}",
            "--pci",
            "{pci_bdf}",
        ),
        field_diagnostic_sha256=digest,
        runner=runner,
        device_client_finder=no_device_clients,
        sleep=lambda _: None,
    )

    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.RUN_NVLINK74_WORKFLOW,
                parameters={"nvlink_link_id": 3, "pci_bdf": "0000:59:00.0"},
            )
        )
    )

    assert result.status is NodeActionStatus.SUCCEEDED
    assert result.details["field_diagnostic"] == "PASSED"
    assert runner.commands[-1] == (
        [str(executable), "--gpu", "GPU-a", "--link", "3", "--pci", "0000:59:00.0"]
    )


def test_field_diagnostic_rejects_failure_report_with_zero_exit(tmp_path) -> None:
    executable = tmp_path / "fielddiag"
    executable.write_text("#!/bin/sh\n", encoding="ascii")
    executable.chmod(0o755)
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()

    class FailureReportRunner(FakeRunner):
        def __call__(self, command, **kwargs):
            if command and command[0] == str(executable):
                self.commands.append(command)
                return CompletedProcess(
                    command, 0, stdout="Overall Result: FAIL\n", stderr=""
                )
            return super().__call__(command, **kwargs)

    agent = node_action_executor(
        tmp_path,
        "fielddiag-failed.db",
        allowed_operations={WorkflowOperation.RUN_NVLINK74_WORKFLOW},
        field_diagnostic_enabled=True,
        field_diagnostic_command=(
            str(executable),
            "--gpu",
            "{gpu_uuid}",
            "--link",
            "{link_id}",
        ),
        field_diagnostic_sha256=digest,
        runner=FailureReportRunner(),
        device_client_finder=no_device_clients,
        sleep=lambda _: None,
    )

    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.RUN_NVLINK74_WORKFLOW,
                parameters={"nvlink_link_id": 3},
            )
        )
    )

    assert result.status is NodeActionStatus.FAILED
    assert "reported failure despite exit status 0" in result.error


def test_memory_field_diagnostic_does_not_require_nvlink_id(tmp_path) -> None:
    executable = tmp_path / "fielddiag"
    executable.write_text("#!/bin/sh\n", encoding="ascii")
    executable.chmod(0o755)
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    runner = FakeRunner()
    agent = node_action_executor(
        tmp_path,
        "memory-fielddiag.db",
        allowed_operations={WorkflowOperation.RUN_FIELD_DIAGNOSTIC},
        field_diagnostic_enabled=True,
        field_diagnostic_command=(str(executable), "--nvlink", "{link_id}"),
        field_diagnostic_sha256=digest,
        memory_field_diagnostic_command=(
            str(executable),
            "--gpu",
            "{gpu_uuid}",
            "--pci",
            "{pci_bdf}",
        ),
        memory_field_diagnostic_sha256=digest,
        runner=runner,
        device_client_finder=no_device_clients,
        sleep=lambda _: None,
    )

    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.RUN_FIELD_DIAGNOSTIC,
                parameters={
                    "procedure": "NVIDIA_FIELD_DIAGNOSTIC",
                    "pci_bdf": "0000:59:00.0",
                },
            )
        )
    )

    assert result.status is NodeActionStatus.SUCCEEDED
    assert result.details["nvlink_link_id"] is None
    assert runner.commands[-1] == [
        str(executable),
        "--gpu",
        "GPU-a",
        "--pci",
        "0000:59:00.0",
    ]


def test_heartbeat_digest_includes_memory_field_diagnostic(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("GPU_FAULT_NODE_CONTROL_PLANE_URL", "http://control-plane")
    monkeypatch.setenv("GPU_FAULT_NODE_CLUSTER_ID", "cluster-a")
    monkeypatch.setenv("GPU_FAULT_NODE_ARTIFACT_SHA256", "a" * 64)
    monkeypatch.setenv("GPU_FAULT_NODE_RUNTIME_PROFILE_VERSION", "profile-a")
    monkeypatch.setenv("NODE_NAME", "node-a")

    def executor(name: str, command: tuple[str, ...]):
        return node_action_executor(
            tmp_path,
            f"{name}.db",
            allowed_operations={WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE},
            memory_field_diagnostic_command=command,
            memory_field_diagnostic_sha256=hashlib.sha256(name.encode()).hexdigest()
            if command
            else None,
            now=None,
        )

    without_memory = heartbeat_reporter_from_environment(executor("without", ()))
    with_memory = heartbeat_reporter_from_environment(
        executor("with", ("/opt/nvidia/fielddiag", "--memory"))
    )

    assert without_memory is not None
    assert with_memory is not None
    assert without_memory.config_digest != with_memory.config_digest


def test_dcgm_diagnostic_json_depth_is_bounded() -> None:
    value = "Pass"
    for _ in range(66):
        value = {"nested": value}

    with pytest.raises(ValueError, match="maximum depth"):
        NodeActionExecutor._dcgm_diagnostic_statuses(value)


def test_field_diagnostic_refuses_active_gpu_clients(tmp_path) -> None:
    executable = tmp_path / "fielddiag"
    executable.write_text("#!/bin/sh\n", encoding="ascii")
    executable.chmod(0o755)
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    runner = FakeRunner()
    agent = node_action_executor(
        tmp_path,
        "fielddiag-clients.db",
        allowed_operations={WorkflowOperation.RUN_NVLINK74_WORKFLOW},
        field_diagnostic_enabled=True,
        field_diagnostic_command=(str(executable), "{link_id}"),
        field_diagnostic_sha256=digest,
        runner=runner,
        device_client_finder=lambda _: [
            {"gpu_uuid": "GPU-a", "pid": "123", "process_name": "python"}
        ],
        sleep=lambda _: None,
    )

    result = agent.execute(
        envelope(
            command(
                WorkflowOperation.RUN_NVLINK74_WORKFLOW,
                parameters={"nvlink_link_id": 3},
            )
        )
    )

    assert result.status is NodeActionStatus.FAILED
    assert "GPU device clients are still active" in result.error
    assert all(str(executable) not in item for item in runner.commands), (
        "expected all(str(executable) not in item for item in runner.commands) to be truthy"
    )


def test_field_diagnostic_rejects_unpinned_executable(tmp_path) -> None:
    executable = tmp_path / "fielddiag"
    executable.write_text("#!/bin/sh\n", encoding="ascii")
    executable.chmod(0o755)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        node_action_executor(
            tmp_path,
            "fielddiag-invalid.db",
            allowed_operations={WorkflowOperation.RUN_NVLINK74_WORKFLOW},
            field_diagnostic_enabled=True,
            field_diagnostic_command=(str(executable),),
            field_diagnostic_sha256="0" * 64,
        )


# --- M-8: flight-recorder path mapping must not escape the container view ---


def _fr_mapped_path(proc_root, pid, path):
    from gpu_fault.node_agent.operations.flight_recorder import (
        FlightRecorderOperationsMixin,
    )

    return FlightRecorderOperationsMixin.process_namespace_path_for_root(
        Path(proc_root), pid, Path(path)
    )


def test_flight_recorder_maps_absolute_path_into_container_root(tmp_path) -> None:
    proc_root = tmp_path / "proc"
    mapped = _fr_mapped_path(proc_root, 100, "/tmp/nccl_trace_")
    base = (proc_root / "100" / "root").resolve()
    assert mapped.resolve().is_relative_to(base), mapped
    assert mapped.name == "nccl_trace_"


def test_flight_recorder_maps_relative_path_into_container_cwd(tmp_path) -> None:
    proc_root = tmp_path / "proc"
    mapped = _fr_mapped_path(proc_root, 100, "nccl_trace_7.json")
    base = (proc_root / "100" / "cwd").resolve()
    assert mapped.resolve().is_relative_to(base), mapped


def test_flight_recorder_rejects_absolute_traversal(tmp_path) -> None:
    proc_root = tmp_path / "proc"
    with pytest.raises(ValueError):
        _fr_mapped_path(proc_root, 100, "/../../etc/passwd")


def test_flight_recorder_rejects_relative_traversal(tmp_path) -> None:
    proc_root = tmp_path / "proc"
    with pytest.raises(ValueError):
        _fr_mapped_path(proc_root, 100, "../../etc/passwd")


def test_flight_recorder_rejects_embedded_traversal(tmp_path) -> None:
    proc_root = tmp_path / "proc"
    with pytest.raises(ValueError):
        _fr_mapped_path(proc_root, 100, "/tmp/../../etc/passwd")


def _field_diagnostic_agent(tmp_path, ledger_name: str, stdout: str):
    """An agent whose Field Diagnostic exits 0 and prints ``stdout``."""

    executable = tmp_path / "fielddiag"
    executable.write_text("#!/bin/sh\n", encoding="ascii")
    executable.chmod(0o755)
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()

    class ReportRunner(FakeRunner):
        def __call__(self, command, **kwargs):
            if command and command[0] == str(executable):
                self.commands.append(command)
                return CompletedProcess(command, 0, stdout=stdout, stderr="")
            return super().__call__(command, **kwargs)

    return node_action_executor(
        tmp_path,
        ledger_name,
        allowed_operations={WorkflowOperation.RUN_NVLINK74_WORKFLOW},
        field_diagnostic_enabled=True,
        field_diagnostic_command=(str(executable), "--link", "{link_id}"),
        field_diagnostic_sha256=digest,
        runner=ReportRunner(),
        device_client_finder=no_device_clients,
        sleep=lambda _: None,
    )


def _run_field_diagnostic(agent):
    return agent.execute(
        envelope(
            command(
                WorkflowOperation.RUN_NVLINK74_WORKFLOW,
                parameters={"nvlink_link_id": 3},
            )
        )
    )


@pytest.mark.parametrize(
    "line",
    [
        "Error count: 0",
        "ERROR: none detected",
        "Errors: 0",
        "Failure count: 0",
        "NVLink error counters: 0",
        "0 errors",
        "No failures detected",
    ],
)
def test_field_diagnostic_benign_error_lines_are_not_failures(tmp_path, line) -> None:
    """A clean report that spells the word "error" is still a clean report.

    The zero-suppression only understood "0 errors"/"no failures" -- the count
    ahead of the noun. Every NVIDIA Field Diagnostic writes the count *after*
    it ("Error count: 0") and reports "ERROR: none detected", so a GPU that
    passed was reported as "reported failure despite exit status 0". That
    verdict is what decides between an NVLink repair and an RMA, so a false
    failure sends a healthy GPU for replacement.
    """

    agent = _field_diagnostic_agent(
        tmp_path,
        "benign.db",
        f"Running NVLink diagnostic\n{line}\nOverall Result: PASS\n",
    )

    result = _run_field_diagnostic(agent)

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert result.details["field_diagnostic"] == "PASSED", result.details


@pytest.mark.parametrize(
    "line",
    [
        "Overall Result: FAIL",
        "ERROR: GPU 3 link 7 training failed",
        "Error count: 5",
        "FATAL: diagnostic aborted",
        "Error count: 0 but Overall Result: FAIL",
    ],
)
def test_field_diagnostic_real_failure_lines_still_fail(tmp_path, line) -> None:
    """The zero-suppression must never swallow a genuine failure line."""

    agent = _field_diagnostic_agent(
        tmp_path, "real-failure.db", f"Running NVLink diagnostic\n{line}\n"
    )

    result = _run_field_diagnostic(agent)

    assert result.status is NodeActionStatus.FAILED, result.details
    assert "reported failure despite exit status 0" in (result.error or ""), (
        result.error
    )


def test_diagnostic_retention_prunes_quick_diag_json(tmp_path) -> None:
    """``dcgm-quick-diagnostic-*.json`` was written and never swept.

    The retention sweep only globbed ``gpu-diagnostic-*.tar.gz``, so every
    quick diagnostic left a JSON file behind for good in the same 0700
    directory. On a node that flaps XIDs those accumulate one per attempt
    until the root filesystem fills, which takes kubelet with it.
    """

    output_dir = tmp_path / "diagnostics"
    output_dir.mkdir()
    old = output_dir / "dcgm-quick-diagnostic-old.json"
    newest = output_dir / "dcgm-quick-diagnostic-newest.json"
    for path in (old, newest):
        path.write_text("{}", encoding="utf-8")
    stale = (NOW - timedelta(hours=2)).timestamp()
    os.utime(old, (stale, stale))
    fresh = (NOW - timedelta(minutes=1)).timestamp()
    os.utime(newest, (fresh, fresh))
    agent = node_action_executor(
        tmp_path,
        "quick-retention.db",
        allowed_operations={WorkflowOperation.RUN_DCGM_DIAGNOSTIC},
        diagnostic_output_dir=str(output_dir),
        diagnostic_retention_seconds=3600,
        diagnostic_max_archives=5,
    )

    removed = agent._cleanup_diagnostic_archives(now=NOW)

    assert removed == [old.name], removed
    assert newest.exists(), "the fresh quick diagnostic must survive"


def test_diagnostic_retention_budgets_each_evidence_family(tmp_path) -> None:
    """Quick diagnostics must not evict the bundle a support case waits on."""

    output_dir = tmp_path / "diagnostics"
    output_dir.mkdir()
    bundle = output_dir / "gpu-diagnostic-keep.tar.gz"
    bundle.write_bytes(b"archive")
    quick = [output_dir / f"dcgm-quick-diagnostic-{index}.json" for index in range(3)]
    for path in quick:
        path.write_text("{}", encoding="utf-8")
    for offset, path in enumerate((bundle, *quick)):
        stamp = (NOW - timedelta(minutes=10 - offset)).timestamp()
        os.utime(path, (stamp, stamp))
    agent = node_action_executor(
        tmp_path,
        "family-retention.db",
        allowed_operations={WorkflowOperation.RUN_DCGM_DIAGNOSTIC},
        diagnostic_output_dir=str(output_dir),
        diagnostic_retention_seconds=3600,
        diagnostic_max_archives=2,
    )

    removed = agent._cleanup_diagnostic_archives(now=NOW)

    assert bundle.exists(), "the only bundle must not be evicted by JSON files"
    assert removed == [quick[0].name], removed


def test_a_quick_diagnostic_sweeps_the_evidence_it_leaves_behind(tmp_path) -> None:
    """Only a bundle collection ever ran the sweep, so quick diags never did.

    A node whose only diagnostic operation is ``RUN_DCGM_DIAGNOSTIC`` -- the
    common case, since it is the cheap first step of every XID plan -- would
    keep every JSON forever even with the family now in the sweep.
    """

    output_dir = tmp_path / "diagnostics"
    output_dir.mkdir()
    stale_json = output_dir / "dcgm-quick-diagnostic-stale.json"
    stale_json.write_text("{}", encoding="utf-8")
    long_ago = (NOW - timedelta(days=3)).timestamp()
    os.utime(stale_json, (long_ago, long_ago))
    agent = node_action_executor(
        tmp_path,
        "quick-sweep.db",
        allowed_operations={WorkflowOperation.RUN_DCGM_DIAGNOSTIC},
        diagnostic_output_dir=str(output_dir),
        diagnostic_retention_seconds=3600,
        runner=lambda argv, **_: CompletedProcess(argv, 0, stdout="{}", stderr=""),
        now=lambda: NOW,
    )

    result = agent.execute(envelope(command(WorkflowOperation.RUN_DCGM_DIAGNOSTIC)))

    assert result.status is NodeActionStatus.SUCCEEDED, result.error
    assert not stale_json.exists(), "the stale quick diagnostic must be swept"
    fresh = result.details["evidence_ref"].removeprefix("file://")
    assert Path(fresh).exists(), f"the evidence it just wrote must survive: {fresh}"
