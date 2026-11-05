"""The GF-GENERAL-LIVE-011 dump responder's lifecycle edges and command line.

``test_general_live_fr_responder.py`` serves the responder in a thread to the
product's own flight-recorder path. These tests drive one responder directly
over real FIFOs under ``tmp_path`` (the poll loop's ``select`` replaced where a
wake-up has to be scripted): the constructor's refusals, a stale stop file, an
interrupted or spurious wake-up, an unreadable pipe, a dump that cannot be
written, the cleanup of artifacts that were removed, replaced or made
unreachable meanwhile, and the ``serve``/``stop``/``cleanup``/``render``
commands -- including the file run as the script the job starts.
"""

from __future__ import annotations

import json
import os
import runpy
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import general_live_fr_replay as replay
from scripts.e2e.regional.probes import general_live_fr_responder as fr

#: A descriptor number no open file ever has (far above every allocation).
STALE_DESCRIPTOR = 2**30
SHAPE_ENV = "GFRS_FR_REPLAY_SHAPE"


def shape_document() -> dict[str, Any]:
    return replay.whole_group_shape().document()


def make_responder(tmp_path: Path, **overrides: Any) -> fr.Responder:
    namespace = tmp_path / "ns" / "tmp"
    namespace.mkdir(parents=True, exist_ok=True)
    arguments: dict[str, Any] = {
        "state_dir": str(tmp_path / "state"),
        "pipe_base": str(namespace / "torch_nccl_debug_pipe_"),
        "dump_base": str(namespace / "nccl_trace_rank_"),
        "ranks": [0],
        "world_size": replay.WORLD_SIZE,
        "shape": fr.parse_shape(shape_document()),
        "max_seconds": 60,
        "poll_seconds": 0.005,
    }
    arguments.update(overrides)
    return fr.Responder(**arguments)


def ledger_of(tmp_path: Path) -> dict[str, Any]:
    return fr.read_ledger(str(tmp_path / "state"))


def scripted_select(
    monkeypatch: pytest.MonkeyPatch, responder: fr.Responder, wakeups: list[Any]
) -> list[list[int]]:
    """Each ``select`` call takes the next wake-up; then the responder stops.

    A wake-up is a list of ready descriptors or an exception to raise.
    """

    calls: list[list[int]] = []

    def select(readers: Any, writers: Any, errors: Any, timeout: Any) -> Any:
        calls.append(list(readers))
        if not wakeups:
            responder.request_stop("test")
            return [], [], []
        wakeup = wakeups.pop(0)
        if isinstance(wakeup, BaseException):
            raise wakeup
        return wakeup, [], []

    monkeypatch.setattr(fr, "select", SimpleNamespace(select=select))
    return calls


def render_cli(document: dict[str, Any], rank: int, request: int) -> Any:
    """What ``render`` prints for the shape: the served dump, as JSON."""

    dump = fr.render_dump(fr.parse_shape(document), rank, request, base_time_ns=0)
    return json.loads(json.dumps(dump, sort_keys=True))


# --------------------------------------------------------------------------- #
# the responder
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ranks": []}, "ranks must be distinct non-negative integers"),
        ({"ranks": [0, 0]}, "ranks must be distinct non-negative integers"),
        ({"ranks": [-1]}, "ranks must be distinct non-negative integers"),
        ({"fmt": "yaml"}, "format must be pickle or json"),
    ],
    ids=["no-ranks", "duplicate", "negative", "format"],
)
def test_bad_ranks_or_a_bad_format_are_refused_before_anything_exists(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(fr.ShapeError, match=message):
        make_responder(tmp_path, **overrides)

    assert not (tmp_path / "state").exists(), "no state directory is created"
    assert list((tmp_path / "ns" / "tmp").iterdir()) == [], "no FIFO is created"


def test_the_first_stop_reason_wins_and_the_baseline_is_restored(
    tmp_path: Path,
) -> None:
    responder = make_responder(tmp_path)
    responder.start()
    pipe = Path(responder.pipe_path(0))
    assert pipe.exists(), "start creates the rank's FIFO"

    responder.request_stop("signal:SIGTERM")
    responder.request_stop("stop")

    assert responder.serve() == "signal:SIGTERM"
    ledger = ledger_of(tmp_path)
    assert (ledger["status"], ledger["finish_reason"]) == ("finished", "signal:SIGTERM")
    assert ledger["removed"] == {"pipes": [str(pipe)], "dumps": []}
    assert not pipe.exists(), "the FIFO is removed on the way out"


def test_a_stop_file_left_by_an_earlier_responder_is_cleared_on_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "stop").write_text("an earlier responder's stop", encoding="utf-8")
    responder = make_responder(tmp_path)

    responder.start()

    assert not (state / "stop").exists(), "the old stop file would end it at once"
    assert ledger_of(tmp_path)["status"] == "serving"
    scripted_select(monkeypatch, responder, [])
    assert responder.serve() == "test", "it serves until its own stop"


def test_an_interrupted_select_and_a_spurious_wakeup_are_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responder = make_responder(tmp_path)
    responder.start()
    reader = responder.readers[0]
    # Readable but nothing written: the read meets EAGAIN (the holder is open).
    calls = scripted_select(monkeypatch, responder, [InterruptedError(), [reader]])

    assert responder.serve() == "test"

    assert calls == [[reader], [reader], [reader]], "polled again after each"
    ledger = ledger_of(tmp_path)
    assert ledger["errors"] == [], ledger["errors"]
    assert ledger["ranks"]["0"]["requests"] == 0, "no request was read"


def test_a_pipe_that_cannot_be_read_is_noted_and_serving_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responder = make_responder(tmp_path)
    responder.start()
    os.close(responder.readers[0])
    # A descriptor whose reads fail with something other than EAGAIN.
    responder.readers[0] = os.open(tmp_path, os.O_RDONLY)
    calls = scripted_select(monkeypatch, responder, [[responder.readers[0]]])

    assert responder.serve() == "test"

    assert len(calls) == 2, "the loop went on after the failed read"
    errors = ledger_of(tmp_path)["errors"]
    assert [item["error"][:31] for item in errors] == [
        "read of rank 0 pipe: [Errno 21]"
    ], errors


def test_an_end_of_file_wakeup_is_not_a_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responder = make_responder(tmp_path)
    responder.start()
    # Without the own write end the FIFO reads end-of-file (zero bytes).
    os.close(responder.holders.pop(0))
    scripted_select(monkeypatch, responder, [[responder.readers[0]]])

    assert responder.serve() == "test"

    ledger = ledger_of(tmp_path)
    assert ledger["ranks"]["0"]["requests"] == 0, "an empty read is no request"
    assert (ledger["unexpected_payloads"], ledger["errors"]) == (0, [])
    assert not Path(responder.dump_path(0)).exists(), "no dump was written"


def test_a_dump_that_cannot_be_written_is_a_failed_event(tmp_path: Path) -> None:
    responder = make_responder(tmp_path)
    responder.start()
    try:
        # The temporary path the dump is written through is taken.
        (tmp_path / "state" / "pending-0").mkdir()

        responder.answer(0, b"1\n")

        ledger = ledger_of(tmp_path)
        record = ledger["ranks"]["0"]
        assert (record["requests"], record["answered"]) == (1, 0), record
        event = record["events"][0]
        assert event["action"] == "failed" and "Is a directory" in event["error"]
        assert ledger["errors"][0]["error"].startswith(
            "rank 0 request 1: [Errno 21]"
        ), ledger["errors"]
        assert not Path(responder.dump_path(0)).exists(), "no dump was published"
    finally:
        responder.finish("test")


def test_finish_without_start_records_that_nothing_was_removed(tmp_path: Path) -> None:
    responder = make_responder(tmp_path)
    (tmp_path / "state").mkdir()

    responder.finish("deadline")

    ledger = ledger_of(tmp_path)
    assert (ledger["status"], ledger["finish_reason"]) == ("finished", "deadline")
    assert ledger["removed"] == {"pipes": [], "dumps": []}
    assert ledger["errors"] == []


def test_a_descriptor_already_closed_does_not_stop_the_cleanup(tmp_path: Path) -> None:
    responder = make_responder(tmp_path)
    responder.start()
    os.close(responder.holders[0])
    responder.holders[0] = STALE_DESCRIPTOR

    responder.finish("deadline")

    ledger = ledger_of(tmp_path)
    assert ledger["removed"]["pipes"] == [responder.pipe_path(0)], ledger["removed"]
    assert ledger["errors"] == [], "a failed close is not an error"
    assert not Path(responder.pipe_path(0)).exists(), "the FIFO is still removed"


def test_artifacts_removed_or_replaced_meanwhile_are_left_alone(tmp_path: Path) -> None:
    responder = make_responder(tmp_path, ranks=[0, 1])
    responder.start()
    responder.answer(1, b"1\n")
    os.unlink(responder.pipe_path(0))  # someone else removed it already
    dump = Path(responder.dump_path(1))
    dump.unlink()
    dump.mkdir()  # and something that is not the responder's dump took its name

    responder.finish("deadline")

    ledger = ledger_of(tmp_path)
    assert ledger["removed"] == {"pipes": [responder.pipe_path(1)], "dumps": []}
    assert ledger["errors"] == [], "a missing artifact is not an error"
    assert dump.is_dir(), "a path that is not its own artifact is never removed"


def test_an_artifact_path_that_cannot_be_inspected_is_an_error(tmp_path: Path) -> None:
    responder = make_responder(tmp_path)
    responder.start()
    namespace = tmp_path / "ns" / "tmp"
    namespace.rename(tmp_path / "ns" / "moved")
    namespace.write_text("a file where the pipe directory was", encoding="utf-8")

    responder.finish("deadline")

    ledger = ledger_of(tmp_path)
    assert ledger["removed"]["pipes"] == []
    expected = f"removing {responder.pipe_path(0)}: [Errno 20] Not a directory"
    assert [item["error"][: len(expected)] for item in ledger["errors"]] == [
        expected
    ], ledger["errors"]
    assert (tmp_path / "ns" / "moved" / "torch_nccl_debug_pipe_0.pipe").exists(), (
        "what it could not inspect is left in place"
    )


# --------------------------------------------------------------------------- #
# the command line
# --------------------------------------------------------------------------- #
def test_render_reads_the_shape_from_the_named_environment_variable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    document = shape_document()
    monkeypatch.setenv("GFRS_TEST_SHAPE", json.dumps(document))

    argv = ["render", "--shape-env", "GFRS_TEST_SHAPE", "--rank", "2", "--request", "1"]
    assert fr.main(argv) == 0

    assert json.loads(capsys.readouterr().out) == render_cli(document, 2, 1)


def test_render_without_a_shape_is_an_error_document(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(SHAPE_ENV, raising=False)

    assert fr.main(["render", "--rank", "0", "--request", "1"]) == 2

    assert json.loads(capsys.readouterr().out) == {
        "error": "ShapeError: no shape: give --shape-json or set the --shape-env "
        "variable"
    }


def test_a_shape_over_one_mebibyte_is_refused_unparsed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "shape.json"
    path.write_bytes(b" " * (fr.MAX_SHAPE_BYTES + 1))

    argv = ["render", "--shape-json", str(path), "--rank", "0", "--request", "1"]
    assert fr.main(argv) == 2

    assert json.loads(capsys.readouterr().out) == {
        "error": "ShapeError: shape exceeds 1 MiB"
    }


@pytest.mark.parametrize(
    ("extra", "environment", "message"),
    [
        (["--procs-per-node", "0"], {"RANK": "0", "WORLD_SIZE": "2"}, "1..64"),
        (["--procs-per-node", "65"], {"RANK": "0", "WORLD_SIZE": "2"}, "1..64"),
        ([], {"WORLD_SIZE": "2"}, "environment unusable: 'RANK'"),
        ([], {"RANK": "one", "WORLD_SIZE": "2"}, "environment unusable: invalid"),
    ],
    ids=["no-procs", "too-many-procs", "no-node-rank", "bad-node-rank"],
)
def test_served_ranks_refuse_an_unusable_pod_layout(
    monkeypatch: pytest.MonkeyPatch,
    extra: list[str],
    environment: dict[str, str],
    message: str,
) -> None:
    for name in ("RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    arguments = fr.build_parser().parse_args(
        [
            "serve",
            "--state-dir",
            "/tmp/s",
            "--pipe-base",
            "/tmp/p_",
            "--dump-base",
            "/tmp/d_",
            "--max-seconds",
            "10",
            *extra,
        ]
    )

    with pytest.raises(fr.ShapeError, match=message):
        fr.served_ranks(arguments)


def test_serve_installs_its_signal_stop_and_restores_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "shape.json"
    path.write_text(json.dumps(shape_document()), encoding="utf-8")
    namespace = tmp_path / "ns"
    namespace.mkdir()
    numbers = (signal.SIGTERM, signal.SIGINT)
    previous = {number: signal.getsignal(number) for number in numbers}
    installed: dict[int, Any] = {}

    def select(readers: Any, writers: Any, errors: Any, timeout: Any) -> Any:
        # The orchestrator's SIGTERM, delivered while the responder waits.
        installed.update({number: signal.getsignal(number) for number in numbers})
        installed[signal.SIGTERM](signal.SIGTERM, None)
        return [], [], []

    monkeypatch.setattr(fr, "select", SimpleNamespace(select=select))
    argv = [
        "serve",
        "--state-dir",
        str(tmp_path / "state"),
        "--pipe-base",
        str(namespace / "pipe_"),
        "--dump-base",
        str(namespace / "dump_"),
        "--max-seconds",
        "60",
        "--ranks",
        "0,1",
        "--world-size",
        "16",
        "--shape-json",
        str(path),
        "--poll-seconds",
        "0.01",
    ]
    try:
        code = fr.main(argv)
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)

    assert code == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "finished",
        "reason": "signal:SIGTERM",
    }
    assert all(installed[number] is not previous[number] for number in numbers), (
        "serve installs its own SIGTERM and SIGINT handlers"
    )
    ledger = ledger_of(tmp_path)
    assert (ledger["finish_reason"], ledger["world_size"]) == ("signal:SIGTERM", 16)
    assert sorted(ledger["removed"]["pipes"]) == [
        str(namespace / "pipe_0.pipe"),
        str(namespace / "pipe_1.pipe"),
    ]
    assert list(namespace.iterdir()) == [], "the no-recorder baseline is back"


@pytest.mark.parametrize("ledger", [None, "{torn"], ids=["no-ledger", "torn-ledger"])
def test_stop_without_a_finished_ledger_times_out_with_exit_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ledger: str | None,
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    if ledger is not None:
        (state / "ledger.json").write_text(ledger, encoding="utf-8")
    clock = {"now": 100.0}
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(
        fr, "time", SimpleNamespace(monotonic=lambda: clock["now"], sleep=sleep)
    )

    assert fr.main(["stop", "--state-dir", str(state), "--timeout", "1"]) == 1

    assert json.loads(capsys.readouterr().out) == {"stopped": False, "ledger": {}}
    assert sleeps and set(sleeps) == {0.2}, sleeps
    assert clock["now"] >= 101.0 - 1e-9, "it waited out the whole timeout"
    assert (state / "stop").read_text(encoding="utf-8"), "the stop file is written"


def test_cleanup_without_a_ledger_reports_and_succeeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert fr.main(["cleanup", "--state-dir", str(tmp_path)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["removed"] == []
    assert output["error"].startswith("no ledger: [Errno 2]"), output


def test_cleanup_removes_only_a_ledger_fifo_and_a_regular_dump(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    fifo = tmp_path / "pipe_1.pipe"
    os.mkfifo(fifo)
    regular_pipe = tmp_path / "pipe_0.pipe"
    regular_pipe.write_text("not a FIFO", encoding="utf-8")
    dump = tmp_path / "dump_0"
    dump.write_bytes(b"a dump")
    dump_directory = tmp_path / "dump_1"
    dump_directory.mkdir()
    ledger = {
        "status": "serving",
        "ranks": {
            "0": {"pipe": str(regular_pipe), "dump": str(dump)},
            "1": {"pipe": str(fifo), "dump": str(dump_directory)},
            "2": {"pipe": str(tmp_path / "missing.pipe"), "dump": ""},
        },
    }
    (state / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")

    assert fr.main(["cleanup", "--state-dir", str(state)]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "removed": [str(dump), str(fifo)],
        "status": "serving",
    }
    assert regular_pipe.exists() and dump_directory.is_dir(), (
        "a path of the wrong kind is never removed"
    )


def test_the_responder_file_runs_as_the_script_the_job_starts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    document = shape_document()
    monkeypatch.setenv(SHAPE_ENV, json.dumps(document))
    monkeypatch.setattr(
        sys, "argv", [fr.__file__, "render", "--rank", "3", "--request", "2"]
    )

    with pytest.raises(SystemExit) as exited:
        runpy.run_path(fr.__file__, run_name="__main__")

    assert exited.value.code == 0
    assert json.loads(capsys.readouterr().out) == render_cli(document, 3, 2)
