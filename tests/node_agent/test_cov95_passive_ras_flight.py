"""The passive sample's RAS listener election, RAS query and existing-file flight read.

``listening_owner`` elects the one verified GPU rank whose own network
namespace listens on the loopback RAS port; ``read_ras`` queries it from inside
that namespace and refuses an answer from a process that changed meanwhile;
``local_flight`` reads only a flight recorder file the rank already wrote,
small and fresh, and keeps status watermarks without the operation lists.
Procfs is an owned tree under ``tmp_path``; the RAS socket query and
``setns`` are fakes, so no test opens a socket or changes namespace.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.node_agent import passive_communication
from gpu_fault.node_agent.observation_worker import ObservationAgent
from gpu_fault.node_agent.passive_communication import (
    listening_owner,
    local_flight,
    read_ras,
)
from tests.node_agent.test_hung_collection_hardening import stat_text

RAS_PORT = 28028
WORKLOAD = ["/training/job"]


def agent(tmp_path: Path) -> ObservationAgent:
    return ObservationAgent(
        {
            "proc_root": str(tmp_path / "proc"),
            "infiniband_root": str(tmp_path / "ib"),
            "diagnostic_output_dir": str(tmp_path / "evidence"),
        }
    )


def rank_process(
    tmp_path: Path,
    pid: int,
    *,
    start: int = 1000,
    cgroup: str = "/training/job",
    environment: str = "",
) -> Path:
    directory = tmp_path / "proc" / str(pid)
    directory.mkdir(parents=True)
    (directory / "cgroup").write_text(f"0::{cgroup}\n")
    (directory / "stat").write_text(stat_text(pid, start))
    (directory / "environ").write_bytes(environment.encode())
    (directory / "status").write_text("State:\tS\n")
    return directory


def listen(
    directory: Path,
    *,
    port: int = RAS_PORT,
    inode: str = "123",
    table: str = "tcp",
    address: str = "0100007F",
    descriptor_table: bool = True,
) -> None:
    (directory / "net").mkdir(exist_ok=True)
    (directory / "net" / table).write_text(
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when "
        "retrnsmt   uid  timeout inode\n"
        f"   0: {address}:{port:04X} 00000000:0000 0A 0 0 0 0 0 {inode}\n"
    )
    if descriptor_table:
        (directory / "fd").mkdir(exist_ok=True)
        (directory / "fd" / "7").symlink_to(f"socket:[{inode}]")


def row(pid: int, *, start: int = 1000, **values: Any) -> dict[str, Any]:
    return {
        "pid": pid,
        "rank": values.pop("rank", 0),
        "gpu_uuids": values.pop("gpu_uuids", ["GPU-a"]),
        "process_identity": {
            "verified": values.pop("verified", True),
            "start_time_ticks": start,
        },
        **values,
    }


def test_only_a_verified_gpu_rank_on_a_loopback_port_is_elected(tmp_path):
    # Every candidate below listens on its port; only the guards keep the
    # earlier ones from being elected.
    for pid, environment in [
        (101, ""),
        (102, ""),
        (103, "NCCL_RAS_ADDR=10.0.0.5:28028\0"),
        (104, "NCCL_RAS_ADDR=127.0.0.1:ras\0"),
        (105, ""),
    ]:
        listen(rank_process(tmp_path, pid, environment=environment))
    rows = [
        row(101, gpu_uuids=[]),
        row(102, verified=False),
        row(103),
        row(104),
        row(105),
    ]

    assert listening_owner(agent(tmp_path), rows) == (105, "127.0.0.1", RAS_PORT)


def test_an_ipv6_ras_address_is_matched_in_the_tcp6_table_only(tmp_path):
    directory = rank_process(tmp_path, 100, environment="NCCL_RAS_ADDR=[::1]:28100\0")
    listen(
        directory, port=28100, table="tcp6", address="00000000000000000000000001000000"
    )
    # An IPv4 table naming the same socket must not be what elected it.
    (directory / "net" / "tcp").write_text("header\n")

    assert listening_owner(agent(tmp_path), [row(100)]) == (100, "::1", 28100)


def test_a_rank_whose_tables_cannot_be_read_does_not_stop_the_election(tmp_path):
    # 100 has no socket table at all, 101 has no descriptor table: both are
    # passed over and the readable 102 is still elected.
    no_table = rank_process(tmp_path, 100)
    (no_table / "fd").mkdir()
    (no_table / "fd" / "7").symlink_to("socket:[123]")
    listen(rank_process(tmp_path, 101), descriptor_table=False)
    listen(rank_process(tmp_path, 102))

    owner = listening_owner(agent(tmp_path), [row(100), row(101), row(102)])

    assert owner == (102, "127.0.0.1", RAS_PORT), owner


def test_the_descriptor_scan_reads_at_most_4096_entries(tmp_path, monkeypatch):
    directory = rank_process(tmp_path, 100)
    listen(directory, descriptor_table=False)
    (directory / "fd").mkdir()
    for index in range(4100):
        (directory / "fd" / str(index)).symlink_to("anon_inode:[eventfd]")
    reads: list[str] = []
    real_readlink = os.readlink

    def counting_readlink(path: Any, *args: Any, **kwargs: Any) -> str:
        if str(path).startswith(str(directory / "fd")):
            reads.append(str(path))
        return real_readlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "readlink", counting_readlink)
        owner = listening_owner(agent(tmp_path), [row(100)])

    assert owner is None, owner
    assert len(reads) == 4096, f"read {len(reads)} descriptors of 4100"


class RasQuery:
    """Stands in for ``query_ras``: records the endpoint, answers or raises."""

    def __init__(self, answer: Any = None, *, during: Any = None) -> None:
        self.answer = answer if answer is not None else {"communicators_count": 1}
        self.during = during
        self.calls: list[tuple[str, int]] = []

    def __call__(self, host: str, port: int) -> dict[str, Any]:
        self.calls.append((host, port))
        if self.during is not None:
            self.during()
        if isinstance(self.answer, BaseException):
            raise self.answer
        return dict(self.answer)


def ras_listener(tmp_path: Path, *, same_namespace: bool = True) -> Path:
    directory = rank_process(tmp_path, 100)
    listen(directory)
    (directory / "ns").mkdir()
    if same_namespace:
        (directory / "ns" / "net").symlink_to("/proc/self/ns/net")
    else:
        (directory / "ns" / "net").write_bytes(b"")
    return directory


def refuse_setns(fd: int, nstype: int = 0) -> None:
    raise AssertionError("a listener in this namespace must be queried without setns")


def test_a_coordinator_without_a_scoped_listener_reports_none(tmp_path):
    rank_process(tmp_path, 100)

    result = read_ras(agent(tmp_path), [row(100)], True, WORKLOAD)

    assert result == {"status": "UNAVAILABLE", "reason": "NO_SCOPED_RAS_LISTENER"}


@pytest.mark.parametrize(
    "change,start,paths",
    [
        pytest.param("restarted", 2000, WORKLOAD, id="restarted-listener"),
        pytest.param("foreign", 1000, ["/other-job"], id="listener-outside-scope"),
    ],
)
def test_a_listener_that_is_not_the_sampled_rank_is_never_queried(
    tmp_path, monkeypatch, change, start, paths
):
    query = RasQuery()
    monkeypatch.setattr(passive_communication, "query_ras", query)
    directory = ras_listener(tmp_path)
    (directory / "stat").write_text(stat_text(100, start))

    result = read_ras(agent(tmp_path), [row(100)], True, paths)

    assert result == {"status": "UNAVAILABLE", "reason": "RAS_OWNER_IDENTITY_CHANGED"}
    assert query.calls == [], change


def test_a_listener_in_this_namespace_is_queried_without_setns(tmp_path, monkeypatch):
    query = RasQuery({"communicators_count": 2, "status": "PARSED"})
    monkeypatch.setattr(passive_communication, "query_ras", query)
    monkeypatch.setattr(os, "setns", refuse_setns)
    ras_listener(tmp_path)

    result = read_ras(agent(tmp_path), [row(100)], True, WORKLOAD)

    assert query.calls == [("127.0.0.1", RAS_PORT)], query.calls
    assert result == {
        "communicators_count": 2,
        "status": "COLLECTED",
        "listener_pid": 100,
        "listener_start_ticks": 1000,
    }, result


def test_a_listener_in_another_namespace_is_queried_from_inside_it(
    tmp_path, monkeypatch
):
    entered: list[int] = []
    query = RasQuery()

    def record_setns(fd: int, nstype: int = 0) -> None:
        entered.append(nstype)

    monkeypatch.setattr(passive_communication, "query_ras", query)
    monkeypatch.setattr(os, "setns", record_setns)
    ras_listener(tmp_path, same_namespace=False)

    result = read_ras(agent(tmp_path), [row(100)], True, WORKLOAD)

    assert entered == [0], "the rank's network namespace is entered once"
    assert result["status"] == "COLLECTED" and result["listener_pid"] == 100, result


def test_a_listener_that_restarts_during_the_query_is_refused(tmp_path, monkeypatch):
    directory = ras_listener(tmp_path)
    query = RasQuery(
        during=lambda: (directory / "stat").write_text(stat_text(100, 3000))
    )
    monkeypatch.setattr(passive_communication, "query_ras", query)
    monkeypatch.setattr(os, "setns", refuse_setns)

    result = read_ras(agent(tmp_path), [row(100)], True, WORKLOAD)

    assert query.calls == [("127.0.0.1", RAS_PORT)], query.calls
    assert result == {"status": "UNAVAILABLE", "reason": "RAS owner process changed"}


@pytest.mark.parametrize(
    "failure,reason",
    [
        pytest.param(TimeoutError("RAS handshake deadline"), "RAS handshake deadline"),
        pytest.param(ConnectionRefusedError("refused"), "refused"),
        pytest.param(ValueError("x" * 300), "x" * 256, id="long-reason-truncated"),
    ],
)
def test_a_failed_ras_query_is_unavailable_with_its_bounded_cause(
    tmp_path, monkeypatch, failure, reason
):
    monkeypatch.setattr(passive_communication, "query_ras", RasQuery(failure))
    monkeypatch.setattr(os, "setns", refuse_setns)
    ras_listener(tmp_path)

    result = read_ras(agent(tmp_path), [row(100)], True, WORKLOAD)

    assert result == {"status": "UNAVAILABLE", "reason": reason}, result


def torch_dump(seq: int) -> dict[str, Any]:
    """The torch 2.10 dump shape: one pending all-reduce on the default group."""

    return {
        "version": "2.10",
        "pg_config": {"0": {"name": "0", "desc": "default_pg", "ranks": "[0, 1]"}},
        "pg_status": {
            "0": {
                "last_enqueued_collective": seq,
                "last_started_collective": -1,
                "last_completed_collective": seq - 1,
            }
        },
        "entries": [
            {
                "record_id": seq - 1,
                "pg_id": 0,
                "process_group": ["0", "default_pg"],
                "collective_seq_id": seq,
                "p2p_seq_id": 0,
                "op_id": seq,
                "profiling_name": "nccl:all_reduce",
                "state": "scheduled",
                "retired": False,
                "is_p2p": False,
                "time_created_ns": 1786844121207040384,
                "time_discovered_started_ns": None,
                "time_discovered_completed_ns": None,
                "input_sizes": [[16777216]],
                "output_sizes": [[16777216]],
                "timeout_ms": 600000,
            }
        ],
    }


def written_dump(tmp_path: Path, pid: int = 100, *, content: bytes = b"") -> Path:
    """A rank with HOME=/root whose recorder already wrote its rank-0 file."""

    directory = rank_process(tmp_path, pid, environment="HOME=/root\0RANK=0\0")
    dump = directory / "root/root/.cache/torch/comm_lib_trace_rank_0"
    dump.parent.mkdir(parents=True)
    dump.write_bytes(content or json.dumps(torch_dump(12)).encode())
    return dump


def test_an_unverified_or_gpu_less_rank_reads_no_flight_file(tmp_path):
    written_dump(tmp_path)
    worker = agent(tmp_path)

    assert local_flight(worker, row(100, verified=False)) == {"status": "UNAVAILABLE"}
    assert local_flight(worker, row(100, gpu_uuids=[])) == {"status": "UNAVAILABLE"}


def test_a_fresh_existing_dump_becomes_a_watermark_without_operation_lists(tmp_path):
    dump = written_dump(tmp_path)

    value = local_flight(agent(tmp_path), row(100))

    assert value["status"] == "dumped", value
    assert value["dump_file"] == str(dump), value
    assert value["source"] == "EXISTING_PROCESS_NAMESPACE_FILE", value
    assert value["dump_triggered"] is False, value
    assert value["source_mtime_ns"] == dump.stat().st_mtime_ns, value
    (group,) = value["last_entries_by_pg"]
    assert group["pg_name"] == "0:default_pg" and group["pg_member_count"] == 2, group
    assert group["progress"]["last_enqueued"] == 12, group
    assert "semantic_entries" not in group, "routine samples keep no operation list"
    assert "unfinished_entries" not in value and "last_entry" not in value, value


@pytest.mark.parametrize("defect", ["stale", "future", "oversize", "absent"])
def test_only_a_small_file_written_in_the_last_minute_is_read(tmp_path, defect):
    dump = written_dump(tmp_path)
    if defect == "stale":
        os.utime(dump, (time.time() - 120, time.time() - 120))
    elif defect == "future":
        os.utime(dump, (time.time() + 120, time.time() + 120))
    elif defect == "oversize":
        dump.write_bytes(b" " * (2 * 1024 * 1024 + 1))
    else:
        dump.unlink()

    value = local_flight(agent(tmp_path), row(100))

    assert value == {"status": "UNAVAILABLE", "reason": "NO_FRESH_BOUNDED_FR_FILE"}


class RotatingAgent(ObservationAgent):
    """The rank rotates its file away between the freshness check and the read."""

    def __init__(self, settings: dict[str, Any]) -> None:
        super().__init__(settings)
        self.stats = 0

    def _dump_file_stat(self, path: Path) -> os.stat_result | None:
        info = super()._dump_file_stat(path)
        if info is not None:
            self.stats += 1
        return info if self.stats <= 1 else None


def test_a_dump_rotated_away_before_the_read_keeps_no_watermark(tmp_path):
    dump = written_dump(tmp_path)
    worker = RotatingAgent(
        {
            "proc_root": str(tmp_path / "proc"),
            "infiniband_root": str(tmp_path / "ib"),
            "diagnostic_output_dir": str(tmp_path / "evidence"),
        }
    )

    value = local_flight(worker, row(100))

    assert value["status"] == "dump_missing", value
    assert value["dump_candidates"] == [str(dump)], value
    assert value["source"] == "EXISTING_PROCESS_NAMESPACE_FILE", value
    assert "source_mtime_ns" not in value, "a file never read sets no watermark"
