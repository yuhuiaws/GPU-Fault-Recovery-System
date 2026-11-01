"""Contract tests for the reusable kernel-log source replay helper.

Nothing here runs systemd, journalctl, kubectl or the kernel collector: the
reviewed-record validators and the ``KernelLogReplay`` controller are driven
over a ``FakeProbe``, and the journalctl pass-through wrapper is rendered and
run as a standalone script against a stub ``journalctl`` to prove it forwards
real lines, appends the reviewed records only when armed, and never leaks its
own synthetic cursor back to the real binary.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.e2e.regional import general_live_kernel_replay as helper
from scripts.e2e.regional.probes import general_live_kernel_source_probe as probe
from tests.regional._general_live_fakes import FakeProbe


# --------------------------------------------------------------------------- #
# reviewed records
# --------------------------------------------------------------------------- #
def test_reviewed_record_defaults_and_digest() -> None:
    record = helper.reviewed_record("NVRM: Xid (PCI:0000:53:00): 79, fell off the bus")
    assert record["transport"] == "kernel", (
        "a reviewed record defaults to the trusted kernel transport"
    )
    assert record["emissions"] == 4, record
    first = helper.records_sha256([record])
    second = helper.records_sha256([dict(record)])
    assert first == second, "the record-set digest is deterministic"
    assert len(first) == 64, "records_sha256 is a sha256 hex digest"


def test_validate_records_accepts_reviewed_set() -> None:
    records = [
        helper.reviewed_record("machine check: uncorrected error", priority=2),
        helper.reviewed_record("user wrote this", transport="syslog"),
    ]
    checked = probe.validate_records(records)
    assert [item["transport"] for item in checked] == ["kernel", "syslog"], checked
    assert checked[0]["priority"] == 2, checked


@pytest.mark.parametrize(
    "records, match",
    [
        ([], "non-empty list"),
        ([{"message": ""}], "non-empty message"),
        ([{"message": "x", "transport": "forged-kernel"}], "not reviewed"),
        ([{"message": "x", "emissions": 99}], "emissions"),
        ([{"message": "x", "priority": 9}], "priority"),
    ],
)
def test_validate_records_rejects_bad_input(records: object, match: str) -> None:
    with pytest.raises(probe.ProbeError, match=match):
        probe.validate_records(records)


def test_validate_records_bounds_count_and_size() -> None:
    too_many = [{"message": "x"} for _ in range(probe.MAX_RECORDS + 1)]
    with pytest.raises(probe.ProbeError, match="at most"):
        probe.validate_records(too_many)
    huge = [{"message": "A" * (probe.MAX_MESSAGE_BYTES + 1)}]
    with pytest.raises(probe.ProbeError, match="too large"):
        probe.validate_records(huge)


def test_probe_digest_is_stable_and_rejects_unsafe_ids() -> None:
    first = probe.digest("gl016-uncorrectable-mce-a1-deadbeef")
    assert first == probe.digest("gl016-uncorrectable-mce-a1-deadbeef"), (
        "digest is stable"
    )
    assert len(first) == 16, "the window digest is 16 hex chars"
    with pytest.raises(probe.ProbeError, match="unsafe run ID"):
        probe.safe_id("bad id; rm -rf /")


# --------------------------------------------------------------------------- #
# the KernelLogReplay controller over a FakeProbe
# --------------------------------------------------------------------------- #
def _records() -> list[dict[str, object]]:
    return [helper.reviewed_record("NVRM: Xid 79", emissions=3)]


def _controller(
    tmp_path: Path, script: dict[str, object], mode: str = "journal"
) -> tuple[helper.KernelLogReplay, FakeProbe]:
    fake = FakeProbe(script)

    class _Regional:
        class settings:
            gpu_kubeconfig = tmp_path / "kubeconfig"
            gpu_context = "ctx"
            namespace = "gpu-fault-system"

    controller = helper.KernelLogReplay(
        _Regional(),
        node="ip-10-0-43-5",
        image="img@sha256:" + "0" * 64,
        case_id="GF-GENERAL-LIVE-016.rdma_fatal",
        run_id="gl016-rdma-fatal-a1-deadbeef",
        case_dir=tmp_path / "case",
        mode=mode,
        host=fake,
    )
    return controller, fake


def test_controller_install_sends_sha256_and_base64(tmp_path: Path) -> None:
    records = _records()
    controller, fake = _controller(tmp_path, {"install-records": {"records": 1}})
    controller.install(records)
    call = next(item for item in fake.calls if item[0] == "install-records")
    flags = dict(zip(call[1::2], call[2::2]))
    assert flags["--sha256"] == helper.records_sha256(records), (
        "install carries the record-set digest"
    )
    decoded = base64.b64decode(flags["--b64"])
    assert json.loads(decoded) == records, (
        "install carries the reviewed records verbatim"
    )


def test_controller_open_arm_readback_and_cleanup(tmp_path: Path) -> None:
    script = {
        "install-records": {"records": 1},
        "open-window": {
            "after": {"ActiveState": "active"},
            "proof_scope": "live-replay",
        },
        "arm": {"armed": True},
        "status": {"injected": {"emission_counts": {"0": 2}}},
        "close-window": {"clean": True},
    }
    controller, fake = _controller(tmp_path, script)
    controller.install(_records())
    controller.open(approved=True)
    opened = next(item for item in fake.calls if item[0] == "open-window")
    assert "--approved" in opened, (
        "journal mode passes the approval through to the probe"
    )
    assert ("--mode", "journal") == (
        opened[opened.index("--mode")],
        opened[opened.index("--mode") + 1],
    ), opened
    controller.arm()
    ingested = controller.read_back()
    assert ingested["injected"]["emission_counts"] == {"0": 2}, (
        "read_back returns the probe's injection oracle"
    )
    receipts = controller.cleanup()
    assert receipts["close"]["clean"] is True, receipts
    assert not any(receipts["probe_residuals"].values()), (
        "a clean probe leaves no residue"
    )
    assert helper.PROOF_SCOPE == "live-replay", (
        "a replay only ever claims the live-replay scope"
    )


def test_controller_cleanup_raises_when_window_will_not_close(tmp_path: Path) -> None:
    controller, _ = _controller(
        tmp_path,
        {
            "open-window": {"after": {"ActiveState": "active"}},
            "close-window": {"clean": False, "residue": {"dropin": True}},
        },
    )
    controller.open(approved=True)
    with pytest.raises(RuntimeError, match="did not close clean"):
        controller.cleanup()


def test_controller_evidence_reports_scope_and_digest(tmp_path: Path) -> None:
    controller, _ = _controller(tmp_path, {"install-records": {}})
    controller.install(_records())
    evidence = controller.evidence()
    assert evidence["proof_scope"] == "live-replay", evidence
    assert evidence["records_sha256"] == helper.records_sha256(_records()), evidence
    assert evidence["record_count"] == 1, evidence


# --------------------------------------------------------------------------- #
# the rendered journalctl pass-through wrapper (journal mode)
# --------------------------------------------------------------------------- #
def _render_wrapper(
    tmp_path: Path, *, armed: bool, run_id: str = "gl016-test-a1-0000beef"
) -> tuple[Path, Path]:
    """Render the wrapper against a stub journalctl; return (wrapper, argv-log)."""

    argv_log = tmp_path / "real-argv.json"
    real = tmp_path / "journalctl-real"
    real.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"open({str(argv_log)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
        "print(json.dumps({'__CURSOR': 'real-1', '__REALTIME_TIMESTAMP': '1',"
        " 'MESSAGE': 'real kernel line', '_TRANSPORT': 'kernel'}))\n",
        encoding="utf-8",
    )
    real.chmod(0o755)
    records = tmp_path / "records.json"
    records.write_text(
        json.dumps(
            [
                helper.reviewed_record(
                    "machine check: uncorrected", priority=2, emissions=2
                )
            ]
        ),
        encoding="utf-8",
    )
    state = tmp_path / "emissions.jsonl"
    armed_path = tmp_path / "armed"
    if armed:
        armed_path.write_text("now", encoding="utf-8")
    wrapper = tmp_path / "journalctl"
    wrapper.write_text(
        probe.WRAPPER_SCRIPT
        % {
            "real": str(real),
            "records": str(records),
            "armed": str(armed_path),
            "state": str(state),
            "emissions": 2,
            "boot_id": "boot-xyz",
            "run_id": run_id,
        },
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper, argv_log


def _run_wrapper(wrapper: Path, *args: str) -> list[dict[str, object]]:
    completed = subprocess.run(
        [sys.executable, "-I", str(wrapper), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]


def test_wrapper_passes_through_when_not_armed(tmp_path: Path) -> None:
    wrapper, _ = _render_wrapper(tmp_path, armed=False)
    lines = _run_wrapper(wrapper, "--output=json", "--since", "@1", "--until", "@2")
    assert len(lines) == 1, "an unarmed wrapper only forwards the real journal"
    assert lines[0]["MESSAGE"] == "real kernel line", lines


def test_wrapper_appends_reviewed_records_when_armed(tmp_path: Path) -> None:
    wrapper, _ = _render_wrapper(tmp_path, armed=True)
    lines = _run_wrapper(wrapper, "--output=json", "--since", "@1", "--until", "@2")
    messages = [item["MESSAGE"] for item in lines]
    assert "real kernel line" in messages, "the real line is still forwarded first"
    assert "machine check: uncorrected" in messages, (
        "the reviewed record is appended when armed"
    )
    reviewed = next(
        item for item in lines if item["MESSAGE"].startswith("machine check")
    )
    assert reviewed["_TRANSPORT"] == "kernel", (
        "a kernel-transport record keeps its trusted origin"
    )
    assert reviewed["_BOOT_ID"] == "boot-xyz", reviewed
    assert reviewed["__CURSOR"].startswith("gfrs-replay-"), (
        "injected records carry a replay cursor"
    )
    assert reviewed["__CURSOR"] == "gfrs-replay-gl016-test-a1-0000beef-0-boot-xyz", (
        "the cursor names the run, the record and the boot"
    )
    # What a grade matches a product record's entry id against (016
    # cursor_pagination follows each record through its finding's marker).
    assert reviewed["__CURSOR"] == (
        helper.replay_entry_prefix("gl016-test-a1-0000beef", 0) + "boot-xyz"
    ), "replay_entry_prefix spells the wrapper's cursor up to the boot"


def test_wrapper_cursor_is_stable_within_a_run_and_new_for_the_next_run(
    tmp_path: Path,
) -> None:
    # The cursor becomes the collector's entry id and then the control plane's
    # event id (sha256 of cluster, node, entry id): the same cursor on the same
    # node means the same incident. Within one run that is the dedup the
    # emissions rely on; across runs on one node and boot it routed the second
    # run's finding into the first run's incident, created before the second
    # run started and therefore never seen by it.
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    first, _ = _render_wrapper(tmp_path / "first", armed=True, run_id="gl016-a-1")
    cursors = set()
    for _poll in range(2):
        lines = _run_wrapper(first, "--output=json", "--since", "@1", "--until", "@2")
        cursors.update(
            item["__CURSOR"] for item in lines if item["MESSAGE"].startswith("machine")
        )
    assert cursors == {"gfrs-replay-gl016-a-1-0-boot-xyz"}, (
        "every emission of a record within one run shares one cursor"
    )
    second, _ = _render_wrapper(tmp_path / "second", armed=True, run_id="gl016-b-2")
    lines = _run_wrapper(second, "--output=json", "--since", "@1", "--until", "@2")
    [other] = [
        item["__CURSOR"] for item in lines if item["MESSAGE"].startswith("machine")
    ]
    assert other == "gfrs-replay-gl016-b-2-0-boot-xyz", other
    assert other not in cursors, (
        "a later run on the same node and boot must mint a new entry identity"
    )


def test_wrapper_bounds_emissions(tmp_path: Path) -> None:
    wrapper, _ = _render_wrapper(tmp_path, armed=True)
    seen = 0
    for _poll in range(5):
        lines = _run_wrapper(wrapper, "--output=json", "--since", "@1", "--until", "@2")
        seen += sum(1 for item in lines if item["MESSAGE"].startswith("machine check"))
    assert seen == 2, "the reviewed record is emitted only up to its emissions budget"


def test_wrapper_never_forwards_its_own_cursor(tmp_path: Path) -> None:
    wrapper, argv_log = _render_wrapper(tmp_path, armed=True)
    _run_wrapper(
        wrapper,
        "--output=json",
        "--since",
        "@1",
        "--until",
        "@2",
        "--after-cursor=gfrs-replay-gl016-test-a1-0000beef-0-boot-xyz",
    )
    forwarded = json.loads(argv_log.read_text(encoding="utf-8"))
    assert not any("gfrs-replay-" in item for item in forwarded), (
        "the wrapper strips its own synthetic cursor before the real journalctl"
    )
    assert "--output=json" in forwarded, "every real flag is still forwarded"
    _run_wrapper(
        wrapper,
        "--output=json",
        "--since",
        "@1",
        "--until",
        "@2",
        "--after-cursor",
        "gfrs-replay-gl016-earlier-run-0-boot-xyz",
    )
    forwarded = json.loads(argv_log.read_text(encoding="utf-8"))
    assert not any("gfrs-replay-" in item for item in forwarded), (
        "a cursor of an earlier run is stripped by its prefix as well"
    )


def test_wrapper_stands_the_emitted_window_in_for_its_own_cursor(
    tmp_path: Path,
) -> None:
    """A stripped synthetic cursor must not leave the real read unbounded.

    The collector positions a resumed poll by the cursor alone -- journalctl
    refuses ``--since`` next to ``--after-cursor`` -- so dropping the wrapper's
    own cursor without a substitute would hand the real binary ``--until`` only:
    the whole journal. The ``--until`` of the call that emitted the cursor is
    where the collector had read to; it stands in as ``--since``.
    """

    wrapper, argv_log = _render_wrapper(tmp_path, armed=True)
    _run_wrapper(wrapper, "--output=json", "--since", "@1", "--until", "@2")

    _run_wrapper(
        wrapper,
        "--after-cursor=gfrs-replay-0-boot-xyz",
        "--until",
        "@3",
        "--output=json",
    )

    forwarded = json.loads(argv_log.read_text(encoding="utf-8"))
    assert not any("gfrs-replay-" in item for item in forwarded), forwarded
    assert forwarded[forwarded.index("--since") + 1] == "@2", (
        f"the emitted window stands in for the synthetic cursor: {forwarded}"
    )
    assert forwarded[forwarded.index("--until") + 1] == "@3", (
        f"the asked-for window end is kept: {forwarded}"
    )
    assert [item for item in forwarded if item in ("--since", "--cursor")] == [
        "--since"
    ], f"exactly one positioning option reaches the real journalctl: {forwarded}"


def test_wrapper_falls_back_to_five_minutes_without_an_emitted_window(
    tmp_path: Path,
) -> None:
    wrapper, argv_log = _render_wrapper(tmp_path, armed=True)

    _run_wrapper(wrapper, "--output=json", "--after-cursor", "gfrs-replay-0-boot-xyz")

    forwarded = json.loads(argv_log.read_text(encoding="utf-8"))
    assert forwarded[forwarded.index("--since") + 1] == "-5min", (
        f"no recorded window: the collector's own cold-start window: {forwarded}"
    )


def test_wrapper_passes_a_real_cursor_through(tmp_path: Path) -> None:
    wrapper, argv_log = _render_wrapper(tmp_path, armed=True)
    _run_wrapper(wrapper, "--output=json", "--after-cursor", "s=real-journal-cursor")
    forwarded = json.loads(argv_log.read_text(encoding="utf-8"))
    assert "s=real-journal-cursor" in forwarded, (
        "a real journal cursor is forwarded untouched"
    )


# --------------------------------------------------------------------------- #
# the probe's per-run kmsg sequence band
# --------------------------------------------------------------------------- #
RUN_ID = "gl020-independent-fatal-event-a1-deadbeef"


def test_kmsg_sequence_is_stable_per_run_and_distinct_across_runs() -> None:
    first = probe.kmsg_sequence(RUN_ID, 0)
    assert first == probe.kmsg_sequence(RUN_ID, 0)
    assert probe.kmsg_sequence(RUN_ID, 1) == first + 1
    assert first >= probe.KMSG_SEQUENCE_BASE
    assert probe.kmsg_sequence(RUN_ID, probe.KMSG_SEQUENCE_STRIDE - 1) < 2**31
    assert probe.kmsg_sequence("gl020-independent-fatal-event-a2-deadbeef", 0) != first
    assert probe.kmsg_sequence("gl024-independent-xid-a1-deadbeef", 0) != first
    assert probe.KMSG_SEQUENCE_STRIDE >= probe.MAX_RECORDS, (
        "a run's records never spill into the next band"
    )
    with pytest.raises(probe.ProbeError):
        probe.kmsg_sequence(RUN_ID, probe.KMSG_SEQUENCE_STRIDE)


# --------------------------------------------------------------------------- #
# the kmsg-mode consumption oracle: did the collector read the fixture?
# --------------------------------------------------------------------------- #
def _kmsg_window(monkeypatch, tmp_path: Path, *, invocation: str = "a" * 32) -> Path:
    """An open kmsg window under a private state root; returns the fixture."""

    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(probe, "PROC_ROOT", tmp_path / "proc")
    probe.save_state(
        RUN_ID,
        {
            "run_id": RUN_ID,
            "mode": "kmsg",
            "unit": probe.KERNEL_UNIT,
            "after": {"InvocationID": invocation, "MainPID": "4242"},
        },
    )
    fixture = probe.kmsg_fixture_path(RUN_ID)
    fixture.write_text(
        "3,900000001,1000,-;NVRM: Xid (PCI:0000:53:00): 987, one\n"
        "3,900000001,2000,-;NVRM: Xid (PCI:0000:53:00): 987, two\n",
        encoding="utf-8",
    )
    return fixture


def _collector_holds(tmp_path: Path, fixture: Path, *, pid: str, offset: int) -> None:
    """A /proc tree in which ``pid`` has the fixture open at ``offset``."""

    fd_dir = tmp_path / "proc" / pid / "fd"
    fd_dir.mkdir(parents=True)
    (fd_dir / "3").symlink_to(tmp_path / "unrelated-file-that-does-not-exist")
    (fd_dir / "7").symlink_to(fixture)
    info_dir = tmp_path / "proc" / pid / "fdinfo"
    info_dir.mkdir()
    (info_dir / "7").write_text(f"pos:\t{offset}\nflags:\t0100000\nmnt_id:\t29\n")


def _journal(monkeypatch, lines: list[str], *, returncode: int = 0) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(command, *, check=True, timeout=120):
        calls.append(list(command))
        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout="\n".join(lines) + ("\n" if lines else ""),
            stderr="",
        )

    monkeypatch.setattr(probe, "run", fake_run)
    return calls


def test_kmsg_consumption_sees_a_collector_that_follows_the_fixture(
    monkeypatch, tmp_path: Path
) -> None:
    fixture = _kmsg_window(monkeypatch, tmp_path)
    _collector_holds(tmp_path, fixture, pid="4242", offset=fixture.stat().st_size)
    calls = _journal(
        monkeypatch, ["starting kernel collector", "health summary posted"]
    )
    state = probe.load_state(RUN_ID)
    oracle = probe.kmsg_consumption(
        RUN_ID, state, {"MainPID": "4242", "InvocationID": "b" * 32}
    )
    assert (
        oracle["fixture_lines"] == 2
        and oracle["fixture_bytes"] == fixture.stat().st_size
    )
    assert oracle["collector"] == {
        "main_pid": "4242",
        "fixture_open": True,
        "offset": fixture.stat().st_size,
    }, oracle
    assert oracle["stream_reopens"] == 0 and oracle["followed"] is True, oracle
    assert oracle["consumed"] is True, oracle
    assert calls and f"_SYSTEMD_INVOCATION_ID={'a' * 32}" in calls[0], (
        "the journal of the invocation open-window restarted is read, not the "
        f"current unit state's: {calls}"
    )


def test_kmsg_consumption_sees_a_collector_that_reopens_the_fixture_at_eof(
    monkeypatch, tmp_path: Path
) -> None:
    """The live R13g failure: the fixture is full, the collector holds no
    descriptor on it and its journal shows a reopen a second."""

    fixture = _kmsg_window(monkeypatch, tmp_path)
    (tmp_path / "proc" / "4242" / "fd").mkdir(parents=True)
    _journal(
        monkeypatch,
        [f"{probe.STREAM_REOPEN_MARKER} {fixture}" for _ in range(37)]
        + ["kernel health summary"],
    )
    oracle = probe.kmsg_consumption(
        RUN_ID, probe.load_state(RUN_ID), {"MainPID": "4242"}
    )
    assert oracle["fixture_lines"] == 2, oracle
    assert (
        oracle["collector"]["fixture_open"] is False
        and oracle["collector"]["offset"] is None
    )
    assert oracle["stream_reopens"] == 37, oracle
    assert oracle["followed"] is False and oracle["consumed"] is False, oracle


def test_kmsg_consumption_without_a_journal_relies_on_the_descriptor(
    monkeypatch, tmp_path: Path
) -> None:
    fixture = _kmsg_window(monkeypatch, tmp_path, invocation="")
    _collector_holds(tmp_path, fixture, pid="77", offset=fixture.stat().st_size)
    _journal(monkeypatch, [], returncode=1)
    oracle = probe.kmsg_consumption(
        RUN_ID, probe.load_state(RUN_ID), {"MainPID": "77", "InvocationID": "not-hex"}
    )
    assert oracle["stream_reopens"] is None, "an unreadable journal is not a count"
    assert oracle["followed"] is True and oracle["consumed"] is True, oracle


def test_kmsg_consumption_is_not_consumed_before_the_offset_reaches_the_end(
    monkeypatch, tmp_path: Path
) -> None:
    fixture = _kmsg_window(monkeypatch, tmp_path)
    _collector_holds(tmp_path, fixture, pid="4242", offset=fixture.stat().st_size - 10)
    _journal(monkeypatch, [])
    oracle = probe.kmsg_consumption(
        RUN_ID, probe.load_state(RUN_ID), {"MainPID": "4242"}
    )
    assert oracle["followed"] is True and oracle["consumed"] is False, oracle


def test_arm_then_status_in_kmsg_mode_report_armed_and_consumption(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(probe, "PROC_ROOT", tmp_path / "proc")
    monkeypatch.setattr(probe.time, "sleep", lambda _seconds: None)
    probe.save_state(
        RUN_ID,
        {
            "run_id": RUN_ID,
            "mode": "kmsg",
            "unit": probe.KERNEL_UNIT,
            "after": {"InvocationID": "c" * 32, "MainPID": "4242"},
        },
    )
    probe.records_path(RUN_ID).write_text(
        json.dumps(
            probe.validate_records(
                [{"message": "NVRM: Xid (PCI:0000:53:00): 987, armed", "emissions": 3}]
            )
        ),
        encoding="utf-8",
    )
    fixture = probe.kmsg_fixture_path(RUN_ID)
    fixture.write_text("", encoding="utf-8")
    probe.arm(argparse.Namespace(run_id=RUN_ID))
    armed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert armed["armed"] is True and armed["lines_written"] == 3, armed
    assert probe.armed_path(RUN_ID).is_file(), "kmsg mode marks the run armed too"
    lines = fixture.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and len({line.split(",")[1] for line in lines}) == 1, (
        f"three copies of one record share one sequence number: {lines}"
    )

    _collector_holds(tmp_path, fixture, pid="4242", offset=fixture.stat().st_size)
    _journal(monkeypatch, [])
    monkeypatch.setattr(
        probe,
        "unit_state",
        lambda unit: {
            "ActiveState": "active",
            "MainPID": "4242",
            "InvocationID": "c" * 32,
        },
    )
    probe.status(argparse.Namespace(run_id=RUN_ID))
    status = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    injected = status["injected"]
    assert injected["mode"] == "kmsg" and injected["armed"] is True, injected
    assert injected["fixture_lines"] == 3 and injected["consumed"] is True, injected
    assert helper.kmsg_follow_error(injected) is None


def test_kmsg_follow_error_judges_only_a_reported_consumed_false() -> None:
    assert helper.kmsg_follow_error({"fixture_lines": 5}) is None, (
        "a probe that reports no consumption is not judged (older fakes/probes)"
    )
    assert helper.kmsg_follow_error({"fixture_lines": 5, "consumed": True}) is None
    message = helper.kmsg_follow_error(
        {
            "fixture_lines": 5,
            "fixture_bytes": 440,
            "consumed": False,
            "followed": False,
            "stream_reopens": 541,
            "collector": {"main_pid": "2073979", "fixture_open": False, "offset": None},
        }
    )
    assert message is not None
    assert "did not follow the reviewed kmsg fixture" in message, message
    assert "541 time(s)" in message, message
    assert "fixture open in the collector=False" in message, message
    assert "read offset None of 440 bytes" in message, message


# --------------------------------------------------------------------------- #
# one record at a time (GF-GENERAL-LIVE-024 xid_twin's repeat outside the window)
# --------------------------------------------------------------------------- #
def _two_record_window(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(probe, "PROC_ROOT", tmp_path / "proc")
    monkeypatch.setattr(probe.time, "sleep", lambda _seconds: None)
    probe.save_state(
        RUN_ID,
        {
            "run_id": RUN_ID,
            "mode": "kmsg",
            "unit": probe.KERNEL_UNIT,
            "after": {"InvocationID": "c" * 32, "MainPID": "4242"},
        },
    )
    probe.records_path(RUN_ID).write_text(
        json.dumps(
            probe.validate_records(
                [
                    {"message": "NVRM: Xid (PCI:0000:53:00): 92, twin", "emissions": 2},
                    {
                        "message": "NVRM: Xid (PCI:0000:53:00): 92, again",
                        "emissions": 3,
                    },
                ]
            )
        ),
        encoding="utf-8",
    )
    fixture = probe.kmsg_fixture_path(RUN_ID)
    fixture.write_text("", encoding="utf-8")
    return fixture


def test_arm_index_appends_one_record_with_its_own_sequence_and_write_time(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    fixture = _two_record_window(monkeypatch, tmp_path)
    clock = iter([1_000.0, 1_001.0, 2_000.0, 2_001.0, 2_002.0])
    monkeypatch.setattr(probe.time, "time", lambda: next(clock))

    probe.arm(argparse.Namespace(run_id=RUN_ID, index=0))
    first = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    lines = fixture.read_text(encoding="utf-8").splitlines()
    assert first["indexes"] == [0] and first["lines_written"] == 2, first
    assert (first["written_at"], first["last_written_at"]) == (1_000.0, 1_001.0)
    assert [line.split(";", 1)[1] for line in lines] == [
        "NVRM: Xid (PCI:0000:53:00): 92, twin"
    ] * 2, lines
    assert {line.split(",")[1] for line in lines} == {
        str(probe.kmsg_sequence(RUN_ID, 0))
    }, "record 0's copies share record 0's sequence"

    probe.arm(argparse.Namespace(run_id=RUN_ID, index=1))
    second = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    later = fixture.read_text(encoding="utf-8").splitlines()[2:]
    assert second["indexes"] == [1] and second["lines_written"] == 3, second
    assert second["written_at"] == 2_000.0, "the repeat exists from its own write"
    assert {line.split(",")[1] for line in later} == {
        str(probe.kmsg_sequence(RUN_ID, 1))
    }, "the repeat is a distinct kernel record, not a copy of the first"
    assert all(line.endswith("92, again") for line in later), later


def test_arm_without_an_index_still_writes_every_record(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    fixture = _two_record_window(monkeypatch, tmp_path)
    probe.arm(argparse.Namespace(run_id=RUN_ID))
    receipt = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert receipt["indexes"] == [0, 1] and receipt["lines_written"] == 6, receipt
    assert len(fixture.read_text(encoding="utf-8").splitlines()) == 6


def test_arm_index_is_refused_out_of_range_and_in_journal_mode(
    monkeypatch, tmp_path: Path
) -> None:
    _two_record_window(monkeypatch, tmp_path)
    with pytest.raises(probe.ProbeError, match="not installed"):
        probe.arm(argparse.Namespace(run_id=RUN_ID, index=2))
    state = probe.load_state(RUN_ID)
    probe.save_state(RUN_ID, {**state, "mode": "journal", "unit": probe.LOG_UNIT})
    with pytest.raises(probe.ProbeError, match="kmsg records"):
        probe.arm(argparse.Namespace(run_id=RUN_ID, index=0))
    assert (
        probe.parser().parse_args(["arm", "--run-id", RUN_ID, "--index", "1"]).index
        == 1
    ), "the parser carries the index"


def test_controller_arms_one_kmsg_record_and_refuses_it_in_journal_mode(
    tmp_path: Path,
) -> None:
    controller, fake = _controller(
        tmp_path, {"arm": {"armed": True, "written_at": 5.0}}, mode="kmsg"
    )
    assert controller.arm(index=1)["written_at"] == 5.0
    assert fake.calls[-1] == ("arm", "--run-id", controller.run_id, "--index", "1")
    controller.arm()
    assert fake.calls[-1] == ("arm", "--run-id", controller.run_id), (
        "without an index every record is armed, as before"
    )
    journal, _ = _controller(tmp_path, {"arm": {"armed": True}})
    with pytest.raises(ValueError, match="kmsg"):
        journal.arm(index=0)
