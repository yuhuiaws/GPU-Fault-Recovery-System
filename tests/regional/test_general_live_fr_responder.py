"""GF-GENERAL-LIVE-011's test flight-recorder dump responder, end to end.

The responder (``probes/general_live_fr_responder.py``) is served in a thread
over a fake container namespace and a fake ``/proc``, and the *product's own*
flight-recorder path reads it: ``FlightRecorderOperationsMixin`` triggers the
FIFO and reads the dump as the Node Agent does, ``flight_series.
collect_second_flight`` pairs the two dumps, ``classify_hung_signals`` decides
and ``bind_hung_bundle`` binds the bundle. That is the contract every replayed
011 variant rests on: each shape of ``general_live_fr_replay`` must make the
product reach the decision the variant grades, with no product code changed.
The responder's own bounds (an existing pipe, bad shapes, the deadline, the
stop file, the EOF race, the ledger) are pinned here as well.
"""

from __future__ import annotations

import json
import os
import pickle
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.execution.hung_bundle import bind_hung_bundle
from gpu_fault.execution.hung_classification import classify_hung_signals
from gpu_fault.flight_progress import operation_key
from gpu_fault.models import WorkflowOperation, WorkflowStepSpec
from gpu_fault.node_agent.operations.flight_evidence import request_flight
from gpu_fault.node_agent.operations.flight_recorder import (
    FlightRecorderOperationsMixin,
    decode_flight_dump,
)
from gpu_fault.node_agent.operations.flight_series import collect_second_flight
from scripts.e2e.regional import general_live_fr_replay as replay
from scripts.e2e.regional.probes import general_live_fr_responder as fr
from tests.node_agent._cov95_runtime_hung import process
from tests.node_agent.test_process_diagnostics_e2e import proc_stat

NODE = "ip-10-0-43-5.us-west-2.compute.internal"
PEER = "ip-10-0-43-6.us-west-2.compute.internal"
NODES = (NODE, PEER)
PROCS = replay.PROCS_PER_NODE
WORLD = replay.WORLD_SIZE
START_TICKS = 1000
CGROUP = "/training/job"


def request(pipe: Path) -> None:
    """Ask for a dump exactly as ``_trigger_flight_recorder`` does."""

    descriptor = os.open(pipe, os.O_WRONLY | os.O_NONBLOCK)
    try:
        os.write(descriptor, b"1\n")
    finally:
        os.close(descriptor)


def wait_for(predicate: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class Served:
    """One responder serving ``ranks`` in a thread over ``namespace/tmp``."""

    def __init__(
        self,
        tmp_path: Path,
        shape: replay.FlightRecorderReplay,
        ranks: list[int],
        *,
        label: str = "0",
        max_seconds: float = 60,
        fmt: str | None = None,
        namespace: Path | None = None,
    ) -> None:
        self.namespace = namespace or tmp_path / "ns"
        (self.namespace / "tmp").mkdir(parents=True, exist_ok=True)
        self.state_dir = tmp_path / f"state-{label}"
        self.document = fr.parse_shape(shape.document())
        self.responder = fr.Responder(
            state_dir=str(self.state_dir),
            pipe_base=str(self.namespace / "tmp" / "torch_nccl_debug_pipe_"),
            dump_base=str(self.namespace / "tmp" / "nccl_trace_rank_"),
            ranks=ranks,
            world_size=WORLD,
            shape=self.document,
            max_seconds=max_seconds,
            fmt=fmt,
            poll_seconds=0.005,
        )
        self.responder.start()
        self.reason: str | None = None
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        self.reason = self.responder.serve()

    def pipe(self, rank: int) -> Path:
        return Path(self.responder.pipe_path(rank))

    def dump(self, rank: int) -> Path:
        return Path(self.responder.dump_path(rank))

    def ledger(self) -> dict[str, Any]:
        return json.loads((self.state_dir / "ledger.json").read_text())

    def stop(self) -> None:
        self.responder.request_stop("test")
        self.thread.join(timeout=5)
        assert not self.thread.is_alive(), "the responder must stop on request"


class Agent(FlightRecorderOperationsMixin):
    """The Node Agent's flight-recorder surface over a fake proc root."""

    def __init__(self, proc_root: Path) -> None:
        self.proc_root = proc_root
        self.sleep = time.sleep

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def read_dump(self, path: Path) -> dict[str, Any]:
        """What the Agent makes of a dump file it was told to look for."""

        return self._finish_flight_recorder(
            {
                "status": "triggered",
                "dump_candidates": [str(path)],
                "previous_mtimes": {},
            }
        )


class Site:
    """Both nodes' responders plus the product's triage over a fake /proc.

    Rank ``r`` is PID ``100 + r`` in cgroup ``/training/job``; its ``root``
    links to the shared container namespace the responders serve in, so the
    Agent's ``/proc/<pid>/root/tmp/...`` mapping lands on the responders' FIFOs
    and dumps exactly as on a live node.
    """

    def __init__(self, tmp_path: Path, shape: replay.FlightRecorderReplay) -> None:
        self.shape = shape
        self.proc_root = tmp_path / "proc"
        self.namespace = tmp_path / "ns"
        self.served = [
            Served(
                tmp_path,
                shape,
                list(range(index * PROCS, (index + 1) * PROCS)),
                label=str(index),
                namespace=self.namespace,
            )
            for index in range(len(NODES))
        ]
        for rank in range(WORLD):
            pid = 100 + rank
            directory = process(self.proc_root, pid, cgroup=CGROUP)
            (directory / "stat").write_text(proc_stat(pid, 0, "R", START_TICKS))
            (directory / "environ").write_bytes(
                (
                    f"RANK={rank}\0LOCAL_RANK={rank % PROCS}\0WORLD_SIZE={WORLD}\0"
                    f"TORCH_NCCL_DEBUG_INFO_PIPE_FILE={replay.PIPE_BASE}\0"
                    f"TORCH_NCCL_DEBUG_INFO_TEMP_FILE={replay.DUMP_BASE}\0"
                ).encode()
            )
            os.symlink(self.namespace, directory / "root")
        self.agent = Agent(self.proc_root)

    def triage(self, node: str, *, interval: float = 1.0) -> list[dict[str, Any]]:
        """``COLLECT_HUNG_TRIAGE``'s flight-recorder work for one node's ranks."""

        index = NODES.index(node)
        command = SimpleNamespace(
            node_id=node,
            parameters={
                "triage_fr_sample_interval_seconds": interval,
                "workload_cgroup_paths_by_node": {node: [CGROUP]},
            },
        )
        processes = []
        for rank in range(index * PROCS, (index + 1) * PROCS):
            pid = 100 + rank
            processes.append(
                {
                    "pid": pid,
                    "rank": rank,
                    "_environment": self.agent._process_environment(pid),
                }
            )
        identities = {
            row["pid"]: {"start_time_ticks": START_TICKS, "verified": True}
            for row in processes
        }
        requests = {
            row["pid"]: request_flight(self.agent, command, row, "first")
            for row in processes
        }
        started = time.monotonic()
        rows = [
            {
                "node_id": node,
                "pid": row["pid"],
                "rank": row["rank"],
                "gpu_uuid": f"GPU-{row['rank']:02d}",
                "process_identity": identities[row["pid"]],
                "flight_recorder": self.agent._finish_flight_recorder(
                    requests[row["pid"]]
                ),
            }
            for row in processes
        ]
        self.agent._await_flight_dumps(requests, rows, started + 10)
        collect_second_flight(
            self.agent,
            command,
            processes,
            rows,
            identities,
            started + 20,
            time.monotonic,
        )
        return rows

    def decide(
        self, *, interval: float = 1.0, prepare: Any = None
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
        """Wave 1 on the first node, wave 2 on the peer unless wave 1 sufficed.

        ``(the decision the bundle is bound with, every signal, the sampled
        nodes)`` -- the product's ``hung_survey.after_triage`` sequence.
        ``prepare`` maps each node's rows (what the live triage adds beside the
        recorder, e.g. the process series of a write-mode stall).
        """

        def wave(node: str) -> list[dict[str, Any]]:
            rows = self.triage(node, interval=interval)
            return [prepare(row) for row in rows] if prepare else rows

        signals = wave(NODE)
        decision = classify_hung_signals(signals, undetermined_nodes=[])
        sampled = [NODE]
        if not decision["targeted_capture_eligible"]:
            signals = [*signals, *wave(PEER)]
            decision = classify_hung_signals(signals, undetermined_nodes=[])
            sampled = [NODE, PEER]
        return decision, signals, sampled

    def bundle(
        self,
        decision: dict[str, Any],
        signals: list[dict[str, Any]],
        sampled: list[str],
    ) -> dict[str, Any]:
        step = WorkflowStepSpec(
            operation=WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
            execution_owner="test",
            node_ids=list(NODES),
            parameters={"disable_strace": True},
            depends_on_step_indexes=[2, 3],
        )
        return bind_hung_bundle(step, decision, signals, sampled).model_dump(
            mode="json"
        )

    @property
    def pipe_base(self) -> str:
        return self.served[0].responder.pipe_base

    @property
    def dump_base(self) -> str:
        return self.served[0].responder.dump_base

    def ledgers(self) -> dict[str, Any]:
        return {f"pod-{index}": item.ledger() for index, item in enumerate(self.served)}

    def close(self) -> None:
        for item in self.served:
            item.stop()


def with_writes(row: dict[str, Any]) -> dict[str, Any]:
    """Add the process series a write-mode held rank yields live.

    Three samples a second apart, two COLLECTED windows with ``wchar`` moving:
    ``behavior_observations`` reads ``HOST_WRITES_OBSERVED`` from it.
    """

    base = datetime(2026, 10, 25, tzinfo=timezone.utc)
    samples = [
        {
            "observed_at": (base + timedelta(seconds=index)).isoformat(),
            "start_ticks": START_TICKS,
            "threads": {},
            "thread_scan_complete": True,
        }
        for index in range(3)
    ]
    windows = [
        {
            "status": "COLLECTED",
            "elapsed_seconds": 1.0,
            "cpu_cores": 0.5,
            "read_bytes_delta": 0,
            "write_bytes_delta": 0,
            "wchar_delta": 4 * 1024 * 1024,
            "thread_scan_complete": True,
            "wait_threads": [],
        }
        for _ in range(2)
    ]
    return {
        **row,
        "process_series": {
            "version": "process-series/v1",
            "requested_samples": 3,
            "samples": samples,
            "windows": windows,
        },
    }


READER = Agent(Path("/proc"))


def groups_of(path: Path) -> dict[str, dict[str, Any]]:
    """The dump at ``path`` as the Agent summarizes it, by process group."""

    summary = READER.read_dump(path)
    assert summary["status"] == "dumped" and "parse_error" not in summary, summary
    return {item["pg_name"]: item for item in summary["last_entries_by_pg"]}


# --------------------------------------------------------------------------- #
# the responder alone
# --------------------------------------------------------------------------- #
def test_each_request_gets_a_shaped_pickle_dump_the_product_decodes(
    tmp_path: Path,
) -> None:
    served = Served(tmp_path, replay.few_candidates_shape(), [0, 1])
    try:
        dumps, groups = [], []
        for index in (1, 2):
            before = (
                served.dump(0).stat().st_mtime_ns if served.dump(0).exists() else -1
            )
            request(served.pipe(0))
            assert wait_for(
                lambda: served.dump(0).exists()
                and served.dump(0).stat().st_mtime_ns > before
            ), f"request {index} was not answered"
            dumps.append(served.dump(0).read_bytes())
            groups.append(groups_of(served.dump(0)))
        assert dumps[0][:2] == b"\x80\x02", (
            "PyTorch's on-disk format: pickle protocol 2"
        )
        assert pickle.loads(dumps[0])["version"] == "2.10"
        assert decode_flight_dump(dumps[0])["pg_config"]["1"]["ranks"] == str(
            list(range(8))
        )
        first, second = groups
        assert set(first) == {"0:default_pg", "1:tp"}
        assert first["1:tp"]["pg_members"] == list(range(8))
        assert first["0:default_pg"]["pg_members"] == list(range(16))
        # The default group advances two collectives per dump ...
        advance = first["0:default_pg"]["progress"], second["0:default_pg"]["progress"]
        assert (advance[0]["last_completed"], advance[1]["last_completed"]) == (
            1200,
            1202,
        )
        assert advance[1]["watermark_source"] == "PG_STATUS"
        # ... while rank 0 lags a collective behind its node group (799 pending
        # at 800) and the pending operation keeps its identity across dumps.
        lag = first["1:tp"]["progress"], second["1:tp"]["progress"]
        assert [item["oldest_pending"]["collective_seq_id"] for item in lag] == [
            799,
            799,
        ]
        assert [item["last_completed"] for item in lag] == [798, 798]
        pending = [
            next(
                e
                for e in groups["1:tp"]["semantic_entries"]
                if e["collective_seq_id"] == 799
            )
            for groups in (first, second)
        ]
        assert operation_key(pending[0]) == operation_key(pending[1])
        ledger = served.ledger()
        record = ledger["ranks"]["0"]
        assert (record["requests"], record["answered"], record["withheld"]) == (2, 2, 0)
        assert record["events"][1]["watermarks"]["1:tp"]["last_completed"] == 798
        assert (
            record["events"][1]["dump_sha256"]
            == __import__("hashlib").sha256(dumps[1]).hexdigest()
        )
        assert ledger["status"] == "serving" and ledger["errors"] == []
        assert ledger["ranks"]["1"]["requests"] == 0, "rank 1 was never asked"
    finally:
        served.stop()


def test_the_json_format_and_the_render_subcommand_match_the_served_dump(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shape = replay.whole_group_shape()
    served = Served(tmp_path, shape, [8], fmt="json")
    try:
        request(served.pipe(8))
        assert wait_for(served.dump(8).exists), "the json dump was written"
        document = json.loads(served.dump(8).read_bytes())
        assert groups_of(served.dump(8))["2:tp"]["pg_members"] == list(range(8, 16)), (
            "the second node's tp group is uid 2 with its own ranks"
        )
    finally:
        served.stop()
    path = tmp_path / "shape.json"
    path.write_text(json.dumps(shape.document()))
    assert (
        fr.main(["render", "--shape-json", str(path), "--rank", "8", "--request", "1"])
        == 0
    )
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["pg_status"] == document["pg_status"]
    assert [e["collective_seq_id"] for e in rendered["entries"]] == [
        e["collective_seq_id"] for e in document["entries"]
    ]


def test_a_withheld_second_request_leaves_the_first_dump_untouched(
    tmp_path: Path,
) -> None:
    served = Served(tmp_path, replay.second_missing_shape(), [3])
    try:
        request(served.pipe(3))
        assert wait_for(served.dump(3).exists), "the first dump was written"
        stamp = served.dump(3).stat().st_mtime_ns
        request(served.pipe(3))
        assert wait_for(lambda: served.ledger()["ranks"]["3"]["requests"] == 2), (
            "the second request reached the responder"
        )
        time.sleep(0.05)
        assert served.dump(3).stat().st_mtime_ns == stamp, "the second dump is withheld"
        record = served.ledger()["ranks"]["3"]
        assert (record["answered"], record["withheld"]) == (1, 1)
        assert [e["action"] for e in record["events"]] == ["answered", "withheld"]
    finally:
        served.stop()


def test_a_request_opened_between_select_and_read_is_answered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The holder write end keeps the FIFO out of end-of-file.

    Same race as ``tests/execution/test_nccl_chain_fifo.py``: the agent's next
    open lands right after the responder's select() returned and before its
    read(); without an own writer that read raised EAGAIN and the thread died.
    """

    served = Served(tmp_path, replay.few_candidates_shape(), [0])
    try:
        request(served.pipe(0))
        assert wait_for(lambda: served.ledger()["ranks"]["0"]["requests"] == 1), (
            "the first request was counted"
        )
        opened: list[int] = []
        real = fr.select.select

        def select(readers: Any, writers: Any, errors: Any, timeout: Any) -> Any:
            ready = real(readers, writers, errors, timeout)
            if not opened:
                opened.append(os.open(served.pipe(0), os.O_WRONLY | os.O_NONBLOCK))
            return ready

        monkeypatch.setattr(fr, "select", SimpleNamespace(select=select))
        assert wait_for(lambda: bool(opened)), (
            "the responder never returned from select()"
        )
        try:
            os.write(opened[0], b"1\n")
        finally:
            os.close(opened[0])
        assert wait_for(lambda: served.ledger()["ranks"]["0"]["requests"] == 2), (
            "the raced request was answered too"
        )
        assert served.ledger()["errors"] == []
        assert served.thread.is_alive(), "the responder thread survived the race"
    finally:
        monkeypatch.undo()
        served.stop()


def test_junk_on_the_pipe_counts_as_one_request_and_is_noted(tmp_path: Path) -> None:
    served = Served(tmp_path, replay.few_candidates_shape(), [0])
    try:
        descriptor = os.open(served.pipe(0), os.O_WRONLY | os.O_NONBLOCK)
        try:
            os.write(descriptor, b"dump please")
        finally:
            os.close(descriptor)
        assert wait_for(lambda: served.ledger()["ranks"]["0"]["requests"] == 1), (
            "a junk payload still counts as a request"
        )
        ledger = served.ledger()
        assert ledger["unexpected_payloads"] == 1
        assert served.dump(0).exists(), (
            "a request is a request, PyTorch reads it the same"
        )
    finally:
        served.stop()


def test_the_deadline_ends_the_responder_and_restores_the_no_recorder_baseline(
    tmp_path: Path,
) -> None:
    served = Served(tmp_path, replay.whole_group_shape(), [0, 1], max_seconds=1)
    request(served.pipe(1))
    assert wait_for(served.dump(1).exists), "the dump was written before the deadline"
    served.thread.join(timeout=5)
    assert served.reason == "deadline"
    ledger = served.ledger()
    assert ledger["status"] == "finished" and ledger["finish_reason"] == "deadline"
    assert not served.pipe(0).exists() and not served.pipe(1).exists(), "FIFOs removed"
    assert not served.dump(1).exists(), "the dump it wrote is removed"
    assert sorted(ledger["removed"]["pipes"]) == sorted(
        [str(served.pipe(0)), str(served.pipe(1))]
    )
    assert ledger["removed"]["dumps"] == [str(served.dump(1))]


def test_the_stop_subcommand_ends_a_serving_responder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    served = Served(tmp_path, replay.whole_group_shape(), [0])
    assert (
        fr.main(["stop", "--state-dir", str(served.state_dir), "--timeout", "5"]) == 0
    )
    served.thread.join(timeout=5)
    assert served.reason == "stop"
    output = json.loads(capsys.readouterr().out)
    assert output["stopped"] is True and output["ledger"]["finish_reason"] == "stop"
    assert not served.pipe(0).exists(), "the stop subcommand removed the FIFO"
    assert fr.main(["ledger", "--state-dir", str(served.state_dir)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "finished"


def test_a_signal_stops_the_responder_through_its_own_cleanup(tmp_path: Path) -> None:
    served = Served(tmp_path, replay.whole_group_shape(), [0])
    served.responder.request_stop("signal:SIGTERM")
    served.thread.join(timeout=5)
    assert served.reason == "signal:SIGTERM"
    assert served.ledger()["finish_reason"] == "signal:SIGTERM"


def test_cleanup_removes_what_a_killed_responder_left(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    served = Served(tmp_path, replay.whole_group_shape(), [0])
    request(served.pipe(0))
    assert wait_for(served.dump(0).exists), "the dump was written before the kill"
    # Killed: the thread stops without ``finish`` (the FIFOs and dump stay).
    served.responder.finish = lambda reason: None  # type: ignore[method-assign]
    served.responder.stop_requested = "killed"
    served.thread.join(timeout=5)
    for descriptor in (
        *served.responder.readers.values(),
        *served.responder.holders.values(),
    ):
        os.close(descriptor)
    served.responder.readers.clear()
    served.responder.holders.clear()
    served.responder.ledger["status"] = "serving"
    served.responder.save()
    assert served.pipe(0).exists(), "a killed responder leaves its FIFO"
    assert fr.main(["cleanup", "--state-dir", str(served.state_dir)]) == 0
    removed = json.loads(capsys.readouterr().out)["removed"]
    assert sorted(removed) == sorted([str(served.pipe(0)), str(served.dump(0))])
    assert not served.pipe(0).exists() and not served.dump(0).exists()


def test_an_existing_pipe_or_a_relative_base_is_refused(tmp_path: Path) -> None:
    (tmp_path / "tmp").mkdir()
    document = fr.parse_shape(replay.whole_group_shape().document())
    os.mkfifo(tmp_path / "tmp" / "torch_nccl_debug_pipe_0.pipe")
    responder = fr.Responder(
        state_dir=str(tmp_path / "state"),
        pipe_base=str(tmp_path / "tmp" / "torch_nccl_debug_pipe_"),
        dump_base=str(tmp_path / "tmp" / "nccl_trace_rank_"),
        ranks=[0],
        world_size=WORLD,
        shape=document,
        max_seconds=5,
    )
    with pytest.raises(fr.ShapeError, match="a real recorder may own it"):
        responder.start()
    with pytest.raises(fr.ShapeError, match="absolute path"):
        fr.Responder(
            state_dir=str(tmp_path / "state"),
            pipe_base="tmp/relative_",
            dump_base=str(tmp_path / "tmp" / "nccl_trace_rank_"),
            ranks=[0],
            world_size=WORLD,
            shape=document,
            max_seconds=5,
        )
    with pytest.raises(fr.ShapeError, match="no groups for ranks"):
        fr.Responder(
            state_dir=str(tmp_path / "state"),
            pipe_base=str(tmp_path / "tmp" / "p_"),
            dump_base=str(tmp_path / "tmp" / "d_"),
            ranks=[99],
            world_size=WORLD,
            shape=document,
            max_seconds=5,
        )
    with pytest.raises(fr.ShapeError, match="max-seconds"):
        fr.Responder(
            state_dir=str(tmp_path / "state"),
            pipe_base=str(tmp_path / "tmp" / "p_"),
            dump_base=str(tmp_path / "tmp" / "d_"),
            ranks=[0],
            world_size=WORLD,
            shape=document,
            max_seconds=0,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.update(version="other"), "unknown shape version"),
        (lambda d: d.update(withhold_from_request=0), "withhold_from_request"),
        (
            lambda d: d["ranks"]["0"]["groups"][0].update(progress="jump"),
            "bad progress",
        ),
        (lambda d: d["ranks"]["0"]["groups"][0].update(seq=3), "seq must be at least"),
        (lambda d: d["ranks"]["0"]["groups"][0].update(members=[1, 2]), "including 0"),
        (lambda d: d["ranks"]["0"]["groups"][0].update(pg_id=0), "duplicate pg_id"),
        (lambda d: d.update(format="yaml"), "pickle or json"),
        (lambda d: d.update(history=0), "bad history"),
    ],
    ids=[
        "version",
        "withhold",
        "progress",
        "seq",
        "members",
        "pg-id",
        "format",
        "history",
    ],
)
def test_a_bad_shape_document_is_refused(change: Any, message: str) -> None:
    document = replay.few_candidates_shape().document()
    if "pg_id" in message:
        document["ranks"]["0"]["groups"][1]["pg_id"] = 0
    else:
        change(document)
    with pytest.raises(fr.ShapeError, match=message):
        fr.parse_shape(document)


def test_served_ranks_come_from_the_torchrun_pod_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
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
        ]
    )
    assert fr.served_ranks(arguments) == (list(range(8, 16)), 16)
    monkeypatch.setenv("RANK", "2")
    with pytest.raises(fr.ShapeError, match="within the number of nodes"):
        fr.served_ranks(arguments)
    arguments.ranks = "3,4"
    with pytest.raises(fr.ShapeError, match="--world-size"):
        fr.served_ranks(arguments)
    arguments.world_size = 16
    assert fr.served_ranks(arguments) == ([3, 4], 16)


def test_the_cli_reports_a_bad_shape_as_an_error_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "shape.json"
    path.write_text(json.dumps({"version": "nope"}))
    assert (
        fr.main(["render", "--shape-json", str(path), "--rank", "0", "--request", "1"])
        == 2
    )
    assert "unknown shape version" in json.loads(capsys.readouterr().out)["error"]


# --------------------------------------------------------------------------- #
# the shapes
# --------------------------------------------------------------------------- #
def test_the_shapes_expand_to_the_variants_design() -> None:
    few = replay.few_candidates_shape()
    assert few.lagging_ranks() == {0, 8}
    assert few.group_names() == {"0:default_pg", "1:tp", "2:tp"}
    assert [g["uid"] for g in few.rank_groups(9)] == ["0", "2"], (
        "node 1's tp group is uid 2"
    )
    assert [g["seq"] for g in few.rank_groups(8)] == [1200, 799], (
        "rank 8 lags its group"
    )
    assert [g["seq"] for g in few.rank_groups(9)] == [1200, 800]
    ambiguous = replay.ambiguous_candidates_shape()
    assert ambiguous.lagging_ranks() == {0, 1, 2, 3, 8, 9, 10, 11}
    assert {g["uid"] for g in ambiguous.rank_groups(12)} == {"0", "2", "4"}
    different = replay.different_process_groups_shape()
    assert [g["seq"] for g in different.rank_groups(0)] == [1200, 1200]
    assert [g["seq"] for g in different.rank_groups(8)] == [1200, 700]
    assert different.lagging_ranks() == set(), (
        "a lower sequence in another group is no lag"
    )
    assert replay.second_missing_shape().withhold_from_request == 2
    assert replay.temporal_completion_shape().group_names() == {
        "0:default_pg",
        "1:gfrs_enqueue_only",
    }
    for name, factory in replay.SHAPES.items():
        shape = factory()
        assert shape.name == name
        fr.parse_shape(shape.document())
        evidence = shape.evidence()
        assert evidence["proof_scope"] == "live-replay"
        assert evidence["shape_sha256"] == shape.sha256()
        assert evidence["responder_sha256"] == replay.responder_sha256()
    assert replay.node_ranks(NODES) == {NODE: list(range(8)), PEER: list(range(8, 16))}


def test_a_group_shape_is_validated_and_collisions_are_refused() -> None:
    with pytest.raises(ValueError, match="only a pending group"):
        replay.GroupShape("1", "tp", replay.ADVANCE, 800, "node", lagging_local=(0,))
    with pytest.raises(ValueError, match="decimal"):
        replay.GroupShape("tp", "tp", replay.PENDING, 800)
    with pytest.raises(ValueError, match="progress"):
        replay.GroupShape("1", "tp", "jump", 800)
    with pytest.raises(ValueError, match="members"):
        replay.GroupShape("1", "tp", replay.PENDING, 800, "some")
    colliding = replay.FlightRecorderReplay(
        "x",
        (replay.DEFAULT_ADVANCE, replay.GroupShape("0", "again", replay.PENDING, 800)),
    )
    with pytest.raises(ValueError, match="collide"):
        colliding.document()
    partial = replay.FlightRecorderReplay(
        "y", (replay.GroupShape("1", "tp", replay.PENDING, 800, (0, 1, 2)),)
    )
    with pytest.raises(ValueError, match="belongs to no group"):
        partial.document()
    assert replay.DELTAS[replay.ENQUEUE_ONLY] == {
        "last_enqueued": 1,
        "last_started": 0,
        "last_completed": 0,
    }


def test_the_ledger_oracle_separates_fixture_faults_from_product_faults() -> None:
    def ledger(ranks: list[int], requests: int = 2, **extra: Any) -> dict[str, Any]:
        return {
            "version": "gfrs-fr-replay-ledger/v1",
            "status": "serving",
            "errors": [],
            "pipe_base": replay.PIPE_BASE,
            "ranks": {
                str(rank): {
                    "pipe": f"{replay.PIPE_BASE}{rank}.pipe",
                    "requests": requests,
                    "events": [],
                }
                for rank in ranks
            },
            **extra,
        }

    good = {"master": ledger(list(range(8))), "worker": ledger(list(range(8, 16)))}
    assert replay.responder_errors(good, NODES) == []
    assert replay.request_errors(good, NODES, sampled_nodes=[NODE]) == []
    assert replay.responder_errors({"master": good["master"]}, NODES) == [
        f"{PEER}: no responder ledger for its Pod"
    ]
    broken = {
        "master": ledger(list(range(8)), status="starting", errors=[{"error": "x"}]),
        "worker": {"error": "cat: no such file"},
    }
    errors = replay.responder_errors(broken, NODES)
    assert any("responder status 'starting'" in item for item in errors), errors
    assert any("responder errors" in item for item in errors), errors
    assert any("worker: no responder ledger" in item for item in errors), errors
    wrong_pipe = {
        "master": ledger(list(range(8))),
        "worker": ledger(list(range(8, 16))),
    }
    wrong_pipe["worker"]["ranks"]["9"]["pipe"] = "/tmp/elsewhere_9.pipe"
    assert replay.responder_errors(wrong_pipe, NODES) == [
        "worker: rank 9 pipe is /tmp/elsewhere_9.pipe"
    ]
    twice = {
        "a": ledger(list(range(8))),
        "b": ledger(list(range(8))),
        "c": ledger(list(range(8, 16))),
    }
    assert any(
        "second responder" in item for item in replay.responder_errors(twice, NODES)
    ), "a second responder for one node is a fixture fault"
    unasked = {
        "master": ledger(list(range(8)), requests=1),
        "worker": ledger(list(range(8, 16)), 0),
    }
    assert replay.request_errors(unasked, NODES, sampled_nodes=[NODE]) == [
        f"{NODE}: rank {rank} received 1 dump request(s), the product's triage should "
        "have asked 2"
        for rank in range(8)
    ], "an unsampled node owes no requests"
    assert len(replay.request_errors(unasked, NODES, sampled_nodes=list(NODES))) == 16


# --------------------------------------------------------------------------- #
# the product over the responder: the contract of every replayed variant
# --------------------------------------------------------------------------- #
@pytest.fixture(name="site")
def site_fixture(tmp_path: Path, request: pytest.FixtureRequest) -> Any:
    sites: list[Site] = []

    def create(shape: replay.FlightRecorderReplay) -> Site:
        site = Site(tmp_path / shape.name, shape)
        sites.append(site)
        return site

    yield create
    for site in sites:
        site.close()


def test_the_product_pairs_the_responders_dumps_as_a_real_recorders(site: Any) -> None:
    current = site(replay.whole_group_shape())
    rows = current.triage(NODE)
    assert [row["rank"] for row in rows] == list(range(8))
    for row in rows:
        series = row["flight_recorder_series"]
        assert series["status"] == "PAIRED", series
        assert series["interval_seconds"] >= 1
        assert row["flight_recorder"]["status"] == "dumped"
        assert row["flight_recorder"]["dump_file"].endswith(
            f"nccl_trace_rank_{row['rank']}"
        ), "the Agent read the dump under the rank's own name"
        assert {g["pg_name"] for g in series["second"]["groups"]} == {
            "0:default_pg",
            "1:tp",
        }
    ledgers = current.ledgers()
    assert all(r["requests"] == 2 for r in ledgers["pod-0"]["ranks"].values()), (
        "every rank of the sampled node was asked twice"
    )
    assert all(r["requests"] == 0 for r in ledgers["pod-1"]["ranks"].values()), (
        "the unsampled node's ranks were never asked"
    )
    assert replay.responder_errors(ledgers, NODES, pipe_base=current.pipe_base) == []
    assert replay.responder_errors(ledgers, NODES) == [
        f"pod-{index}: the responder serves pipes under {current.pipe_base!r}, the "
        f"product reads {replay.PIPE_BASE!r}"
        for index in range(2)
    ], "a responder under another base is not the one the product reads"
    assert replay.request_errors(ledgers, NODES, sampled_nodes=[NODE]) == []


def test_few_candidates_make_a_targeted_bundle_after_the_first_wave(site: Any) -> None:
    current = site(replay.few_candidates_shape())
    decision, signals, sampled = current.decide()
    assert sampled == [NODE], "sufficient local evidence: the peer is never sampled"
    assert decision["classification"] == "PLAUSIBLE"
    assert decision["culprit_ranks"] == [0] and decision["control_ranks"] == [1]
    assert decision["targeted_capture_eligible"] is True, decision
    assert decision["membership_verified"] and decision["temporal_verified"]
    group = next(g for g in decision["process_groups"] if g["pg_name"] == "1:tp")
    assert group["time_comparable"] is True and group["membership_verified"] is True
    assert group["candidate_reasons"]["0"] == [
        "PERSISTENT_COMPLETION_LAG",
        "PERSISTENT_ENQUEUE_LAG",
    ]
    bundle = current.bundle(decision, signals, sampled)
    assert bundle["parameters"]["bundle_scope"] == "identity_bound_candidates"
    assert bundle["parameters"]["target_pids_by_node"] == {NODE: [100, 101]}
    assert bundle["parameters"]["max_processes"] == 2
    assert bundle["node_ids"] == [NODE]
    assert bundle["parameters"]["operator_escalation_required"] is False


def test_ambiguous_candidates_are_reported_but_keep_the_conservative_scope(
    site: Any,
) -> None:
    current = site(replay.ambiguous_candidates_shape())
    decision, signals, sampled = current.decide()
    assert sampled == [NODE, PEER], "four leads are not sufficient local evidence"
    assert decision["culprit_ranks"] == [0, 1, 2, 3, 8, 9, 10, 11]
    assert decision["candidate_count"] == 8
    assert decision["targeted_capture_eligible"] is False
    bundle = current.bundle(decision, signals, sampled)
    assert bundle["parameters"]["bundle_scope"] == "bounded_triage_nodes"
    assert bundle["parameters"]["target_pids_by_node"] == {}
    assert bundle["parameters"]["max_processes"] == 0
    assert bundle["parameters"]["operator_escalation_required"] is True
    assert sorted(bundle["node_ids"]) == sorted(NODES)


def test_a_whole_group_pending_is_a_stall_without_a_culprit(site: Any) -> None:
    current = site(replay.whole_group_shape())
    decision, _signals, sampled = current.decide()
    assert sampled == [NODE, PEER]
    assert decision["stalled_process_groups"] == ["1:tp", "2:tp"]
    assert decision["culprit_ranks"] == [] and decision["candidate_count"] == 0
    assert decision["classification"] == "PLAUSIBLE"
    assert decision["targeted_capture_eligible"] is False
    by_name = {g["pg_name"]: g for g in decision["process_groups"]}
    assert (
        by_name["1:tp"]["group_stall_observed"]
        and by_name["2:tp"]["group_stall_observed"]
    )
    default = by_name["0:default_pg"]
    assert default["persistent_pending_ranks"] == list(range(16))
    # The cross-node group has no clock bound on this site: the product does
    # not call it stalled, by doctrine, and never assumes the error is zero.
    assert default["group_stall_observed"] is False
    assert default["time_reason"] == "MISSING_CLOCK_BOUND_OR_INVALID_WINDOW"
    codes = {c for item in decision["candidate_processes"] for c in item["reasons"]}
    assert codes == {"FR_PERSISTENT_PENDING_OPERATION"}


def test_different_process_groups_are_never_compared_with_each_other(site: Any) -> None:
    current = site(replay.different_process_groups_shape())
    decision, _signals, _sampled = current.decide()
    assert decision["culprit_ranks"] == []
    by_name = {g["pg_name"]: g for g in decision["process_groups"]}
    assert by_name["1:tp"]["observed_members"] == list(range(8))
    assert by_name["2:tp"]["observed_members"] == list(range(8, 16))
    assert by_name["1:tp"]["membership_sha256"] != by_name["2:tp"]["membership_sha256"]
    assert (
        by_name["1:tp"]["group_stall_observed"]
        and by_name["2:tp"]["group_stall_observed"]
    )
    assert by_name["0:default_pg"]["completion_advanced_ranks"] == list(range(16))


def test_temporal_completion_reads_the_shaped_watermarks_per_rank(site: Any) -> None:
    current = site(replay.temporal_completion_shape())
    decision, _signals, _sampled = current.decide()
    by_name = {g["pg_name"]: g for g in decision["process_groups"]}
    advance, enqueue = by_name["0:default_pg"], by_name["1:gfrs_enqueue_only"]
    assert advance["completion_advanced_ranks"] == list(range(16))
    assert all(
        advance["watermark_deltas_by_rank"][str(rank)] == replay.DELTAS[replay.ADVANCE]
        for rank in range(16)
    ), "every rank's default-group deltas are the advance deltas"
    assert enqueue["completion_advanced_ranks"] == []
    assert enqueue["enqueue_advanced_ranks"] == list(range(16))
    assert enqueue["persistent_pending_ranks"] == list(range(16))
    assert all(
        enqueue["watermark_deltas_by_rank"][str(rank)]
        == replay.DELTAS[replay.ENQUEUE_ONLY]
        for rank in range(16)
    ), "every rank's enqueue-only deltas are +1/0/0"
    assert decision["culprit_ranks"] == []


def test_a_withheld_second_dump_is_a_forensics_gap_not_a_stall(site: Any) -> None:
    current = site(replay.second_missing_shape())
    decision, signals, sampled = current.decide()
    assert sampled == [NODE, PEER]
    assert all(
        s["flight_recorder_series"]["status"] == "SECOND_UNAVAILABLE" for s in signals
    ), "every series is SECOND_UNAVAILABLE"
    assert all(
        s["flight_recorder_series"]["second"]["status"] == "dump_missing"
        for s in signals
    ), "every withheld second dump reads dump_missing"
    assert all(s["flight_recorder"]["status"] == "dumped" for s in signals), (
        "the first stays"
    )
    assert decision["evidence_branches"]["fr_temporal"] is None
    assert decision["evidence_branches"]["snapshot_authoritative"] is True
    assert decision["culprit_ranks"] == [] and "stalled_process_groups" not in decision
    assert decision["classification"] == "UNDETERMINED"
    assert decision["candidate_processes"] == []
    bundle = current.bundle(decision, signals, sampled)
    assert bundle["parameters"]["bundle_scope"] == "bounded_triage_nodes"
    ledger = current.ledgers()["pod-0"]["ranks"]["0"]
    assert (ledger["requests"], ledger["answered"], ledger["withheld"]) == (2, 1, 1)


def test_advancing_completion_leaves_the_process_evidence_in_place(site: Any) -> None:
    current = site(replay.plus_process_shape())
    signals = [with_writes(row) for row in current.triage(NODE)]
    decision = classify_hung_signals(signals, undetermined_nodes=[])
    processes = decision["evidence_branches"]["process_observations"]["processes"]
    assert len(processes) == 8
    for item in processes:
        assert sorted(item["fr_completion_advanced_groups"]) == ["0:default_pg", "1:tp"]
        assert item["fr_persistent_pending_groups"] == []
        codes = {obs["code"] for obs in item["behavior"]["observations"]}
        assert "HOST_WRITES_OBSERVED" in codes, codes
        assert len(item["behavior"]["io_windows"]) == 2
    assert decision["culprit_ranks"] == [] and decision["candidate_processes"] == []
