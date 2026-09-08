from __future__ import annotations

from ._support import (
    NOW,
    CompletedProcess,
    FakeRunner,
    NodeActionExecutor,
    NodeActionStatus,
    Path,
    WorkflowOperation,
    _flight_dump_payload,
    _triage_only_executor,
    command,
    envelope,
    hashlib,
    no_device_clients,
    node_action_executor,
    os,
    pytest,
    timedelta,
)


def test_flight_recorder_unstattable_dump_does_not_abort_the_triage(tmp_path) -> None:
    """One candidate the Agent cannot stat must not lose the dump it can.

    ``_finish_flight_recorder`` walked nine candidate paths per rank with a
    bare ``path.is_file()``. ``Path.is_file`` only swallows ENOENT-class
    errors, so a candidate whose directory the Agent cannot traverse raises
    PermissionError -- and it escapes ``_await_flight_dumps``, the hung-triage
    step, and the whole COLLECT_HUNG_TRIAGE command. The dumps live in the
    training container's own directories, so their permissions change without
    warning, and the flight recorder is the only evidence that yields a
    CONFIRMED verdict.
    """

    sealed = tmp_path / "sealed"
    sealed.mkdir()
    hidden = sealed / "nccl_trace_rank_4"
    hidden.write_text(_flight_dump_payload())
    dump = tmp_path / "nccl_trace_rank_4"
    dump.write_text(_flight_dump_payload())
    os.chmod(sealed, 0o000)
    agent = _triage_only_executor(tmp_path, lambda _seconds: None)
    try:
        summary = agent._finish_flight_recorder(
            {
                "status": "triggered",
                "pipe": "/tmp/nccl_pipe_4.pipe",
                "dump_candidates": [str(hidden), str(dump)],
                "previous_mtimes": {},
            }
        )
    finally:
        os.chmod(sealed, 0o755)

    assert summary["status"] == "dumped", summary
    assert summary["dump_file"] == str(dump), summary
    assert summary["last_entry"]["collective_seq_id"] == 12, summary


def test_flight_recorder_overlong_dump_candidate_does_not_abort_the_triage(
    tmp_path,
) -> None:
    """The candidate paths come from the container's own environment.

    ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE`` is read out of ``/proc/<pid>/environ``,
    so a training container can hand the Agent a path component longer than
    NAME_MAX. ``Path.is_file`` raises ENAMETOOLONG for it, which used to abort
    the triage of every rank -- a one-line env var that disables the node's
    hang evidence.
    """

    overlong = tmp_path / ("nccl_trace_rank_" + "x" * 300)
    agent = _triage_only_executor(tmp_path, lambda _seconds: None)

    summary = agent._finish_flight_recorder(
        {
            "status": "triggered",
            "pipe": "/tmp/nccl_pipe_5.pipe",
            "dump_candidates": [str(overlong)],
            "previous_mtimes": {},
        }
    )

    assert summary["status"] == "dump_missing", summary


def test_flight_recorder_vanished_dump_is_a_parse_error_not_a_crash(tmp_path) -> None:
    """A dump that cannot be read is one rank's parse_error, not a crash."""

    dump = tmp_path / "nccl_trace_rank_6"
    dump.write_text(_flight_dump_payload())
    os.chmod(dump, 0o000)
    agent = _triage_only_executor(tmp_path, lambda _seconds: None)
    try:
        summary = agent._finish_flight_recorder(
            {
                "status": "triggered",
                "pipe": "/tmp/nccl_pipe_6.pipe",
                "dump_candidates": [str(dump)],
                "previous_mtimes": {},
            }
        )
    finally:
        os.chmod(dump, 0o644)

    assert summary["status"] == "dumped", summary
    assert "PermissionError" in summary["parse_error"], summary


def test_flight_recorder_dump_size_survives_a_vanishing_file(tmp_path) -> None:
    """The reported size is the one measured when the dump was selected.

    The summary re-``stat``ed the winning candidate twice more after choosing
    it, outside the ``parse_error`` guard, so a dump the container rotated in
    that window raised FileNotFoundError out of the whole command.
    """

    dump = tmp_path / "nccl_trace_rank_7"
    payload = _flight_dump_payload()
    dump.write_text(payload)
    agent = _triage_only_executor(tmp_path, lambda _seconds: None)

    summary = agent._finish_flight_recorder(
        {
            "status": "triggered",
            "pipe": "/tmp/nccl_pipe_7.pipe",
            "dump_candidates": [str(dump)],
            "previous_mtimes": {},
        }
    )

    assert summary["dump_size_bytes"] == len(payload), summary


def test_flight_recorder_trigger_survives_an_unstattable_pipe(tmp_path) -> None:
    """The trigger phase reads the same attacker-influenced paths.

    ``TORCH_NCCL_DEBUG_INFO_PIPE_FILE`` is taken from the container's
    ``environ`` too, and ``_trigger_flight_recorder`` probed it with bare
    ``path.exists()``/``path.stat()``. An overlong component raised
    ENAMETOOLONG before the finish phase was ever reached, so guarding only
    the finish phase would have left the same one-env-var kill switch open.
    """

    proc_root = tmp_path / "proc"
    (proc_root / "321" / "root" / "tmp").mkdir(parents=True)
    (proc_root / "321" / "cwd").mkdir()
    agent = NodeActionExecutor(
        secret="s" * 48,
        node_ids={"node-a"},
        allowed_operations={WorkflowOperation.COLLECT_HUNG_TRIAGE},
        reset_enabled=False,
        ledger=None,
        proc_root=str(proc_root),
        runner=lambda argv, **_: CompletedProcess(argv, 0, "", ""),
        sleep=lambda _seconds: None,
        now=lambda: NOW,
    )

    result = agent._trigger_flight_recorder(
        {
            "pid": 321,
            "rank": 4,
            "_environment": {
                "TORCH_NCCL_DEBUG_INFO_PIPE_FILE": "/tmp/" + "p" * 300,
                "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": "/tmp/" + "d" * 300,
            },
        }
    )

    assert result["status"] == "pipe_missing", result


def test_strace_samples_run_round_parallel_across_processes(tmp_path) -> None:
    """The strace samples were the one serial leg of the hung bundle.

    py-spy already samples every process in a round and sleeps the interval
    once; strace ran the full schedule for process 1, then for process 2, and
    slept the interval per process. With the defaults on a 16-rank node that is
    16 x (3 x 3 s + 2 x 2 s), about 208 s of a bundle whose whole point is to
    capture the node *while* it is hung -- and the control plane's step
    deadline does not grow with the rank count.
    """

    proc_root = tmp_path / "proc"
    pids = (100, 101, 102)
    for pid in pids:
        directory = proc_root / str(pid)
        directory.mkdir(parents=True)
        # No "python" anywhere, so py-spy stays out of the event log.
        (directory / "cmdline").write_bytes(b"trainer\0--rank\0")
        (directory / "status").write_text("State:\tD\n")
        (directory / "stack").write_text("kernel stack\n")
        (directory / "wchan").write_text("futex_wait_queue\n")
    events: list[tuple[str, int]] = []
    sleep_calls: list[float] = []

    class StraceRunner:
        def __call__(self, argv, **_):
            if "--query-compute-apps=pid,gpu_uuid,process_name" in argv:
                return CompletedProcess(
                    argv,
                    0,
                    stdout="".join(f"{pid}, GPU-a, trainer\n" for pid in pids),
                    stderr="",
                )
            if "strace" in argv:
                events.append(("strace", int(argv[argv.index("-p") + 1])))
            return CompletedProcess(argv, 0, stdout="captured", stderr="")

    work_dir = tmp_path / "bundle"
    work_dir.mkdir()
    manifest: dict[str, object] = {"captures": []}
    agent = node_action_executor(
        tmp_path,
        "strace-rounds.db",
        allowed_operations={WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE},
        proc_root=str(proc_root),
        expand_python_cgroup_processes=False,
        runner=StraceRunner(),
        sleep=lambda seconds: sleep_calls.append(seconds),
    )

    agent._capture_hung_process_state(
        work_dir,
        command(
            WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
            parameters={
                "strace_sample_count": 2,
                "strace_duration_seconds": 1,
                "strace_sample_interval_seconds": 2,
                "capture_process_state": True,
            },
        ),
        manifest,
    )

    traced = [pid for name, pid in events if name == "strace"]
    assert len(traced) == 6, events
    assert set(traced[:3]) == set(pids), f"round 1 must cover every process: {events}"
    assert set(traced[3:]) == set(pids), f"round 2 must cover every process: {events}"
    assert sleep_calls == [2], (
        f"the interval is slept once per round, not once per process: {sleep_calls}"
    )
    assert manifest["strace_sampling"] == {
        "sample_count": 2,
        "duration_seconds_per_sample": 1,
        "interval_seconds": 2,
        "process_count": 3,
        "mode": "round_parallel",
    }, manifest["strace_sampling"]
    captures = [
        (item["sample_index"], item["pid"])
        for item in manifest["captures"]
        if str(item["file"]).startswith("strace-")
    ]
    assert sorted(captures) == [
        (1, 100),
        (1, 101),
        (1, 102),
        (2, 100),
        (2, 101),
        (2, 102),
    ], captures


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
