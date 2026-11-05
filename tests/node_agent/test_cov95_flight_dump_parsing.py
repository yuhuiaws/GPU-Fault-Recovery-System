"""Where the Agent looks for a rank's flight recorder dump, and what it makes of it.

The paths and the bytes both come from the training container: the dump
prefix, the pipe and HOME/XDG_CACHE_HOME are read from the rank's own
environment and the dump is whatever the rank (or anything else in its
container) wrote there. These tests cover the parts of that the live triage
tests never reach: a rank without ``RANK`` (the pipe and the prefix are all
there is to look at), the main-branch XDG cache default, a buffer size that is
not a number, a container path that escapes its root through a symlink, a dump
rotated while it was read, and dumps whose structure is hostile or odd (too
big, nested past the parser, wider than the traversal bound, groups named by a
list or not at all, a member list that is not JSON).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.node_agent.operations import existing_flight, flight_recorder
from gpu_fault.node_agent.operations.flight_evidence import request_flight
from gpu_fault.node_agent.operations.flight_recorder import (
    CACHE_DUMP_NAME,
    LEGACY_DUMP_PREFIX,
    FlightRecorderOperationsMixin,
    decode_flight_dump,
    default_dump_prefixes,
    summarize_flight_dump,
    trace_buffer_configuration,
)

CGROUP = "/training/job"
NODE_ID = "node-a"


class RecorderHost(FlightRecorderOperationsMixin):
    """The flight recorder half of the node Agent over a fake ``/proc``."""

    def __init__(self, proc_root: Path, diagnostic_output_dir: Path) -> None:
        self.proc_root = proc_root
        self.diagnostic_output_dir = diagnostic_output_dir
        self.sleep = lambda _seconds: None


def training_process(proc_root: Path, pid: int, environment: dict[str, str]) -> Path:
    """A rank of the workload cgroup, started at tick 5000; its container root."""

    directory = proc_root / str(pid)
    (directory / "root" / "tmp").mkdir(parents=True)
    (directory / "cgroup").write_text(f"0::{CGROUP}\n")
    (directory / "environ").write_bytes(
        b"".join(f"{key}={value}\0".encode() for key, value in environment.items())
    )
    fields = ["S"] + ["0"] * 19
    fields[19] = "5000"
    (directory / "stat").write_text(f"{pid} (python) " + " ".join(fields))
    return directory / "root"


def existing_flight_command() -> SimpleNamespace:
    return SimpleNamespace(
        command_id="cmd-existing-fr",
        node_id=NODE_ID,
        parameters={"workload_cgroup_paths_by_node": {NODE_ID: [CGROUP]}},
    )


def flight_payload(*entries: dict[str, Any], **extra: Any) -> bytes:
    return json.dumps({"entries": list(entries), **extra}).encode()


def test_the_main_branch_xdg_cache_default_leads_the_default_prefixes() -> None:
    assert default_dump_prefixes({"XDG_CACHE_HOME": "/cache", "HOME": "/root"}) == [
        ("torch_xdg_cache_default", f"/cache/torch/{CACHE_DUMP_NAME}"),
        ("torch_home_cache_default", f"/root/.cache/torch/{CACHE_DUMP_NAME}"),
        ("legacy_tmp_default", LEGACY_DUMP_PREFIX),
    ]
    assert [label for label, _ in default_dump_prefixes({"XDG_CACHE_HOME": ""})] == [
        "torch_home_cache_default",
        "legacy_tmp_default",
    ], "an empty XDG_CACHE_HOME is not a cache root"


def test_a_buffer_size_that_is_not_a_number_is_unknown_not_off() -> None:
    assert trace_buffer_configuration(
        {"TORCH_FR_BUFFER_SIZE": "lots", "TORCH_NCCL_TRACE_BUFFER_SIZE": "2000"}
    ) == {
        "variable": "TORCH_FR_BUFFER_SIZE",
        "value": "lots",
        "enabled": None,
        "pipe_enabled": True,
    }
    assert trace_buffer_configuration({"TORCH_NCCL_TRACE_BUFFER_SIZE": "0x10"}) == {
        "variable": "TORCH_NCCL_TRACE_BUFFER_SIZE",
        "value": "0x10",
        "enabled": None,
        "pipe_enabled": None,
    }, "getCvarInt cannot read 0x10: the Agent cannot say whether the pipe exists"


def test_a_dump_over_64_mib_is_refused_before_it_is_parsed() -> None:
    with pytest.raises(ValueError, match="flight recorder dump exceeds 64 MiB"):
        decode_flight_dump(b" " * (64 * 1024 * 1024 + 1))


def test_json_nested_past_the_parser_limit_is_a_value_error() -> None:
    depth = 200_000
    with pytest.raises(ValueError, match="nesting exceeds parser limit"):
        decode_flight_dump(b"[" * depth + b"]" * depth)


@pytest.mark.parametrize(
    "payload",
    [
        # Wider than the traversal bound in one container ...
        [[] for _ in range(100_001)],
        # ... and more containers than the bound, none of them wide.
        [[[] for _ in range(101)] for _ in range(1000)],
    ],
    ids=["wide", "many"],
)
def test_a_dump_beyond_the_traversal_bound_is_refused(payload: Any) -> None:
    with pytest.raises(ValueError, match="structure exceeds traversal limit"):
        summarize_flight_dump(json.dumps(payload).encode())


def test_groups_named_by_a_list_or_not_at_all_and_members_that_are_not_json() -> None:
    summary = summarize_flight_dump(
        flight_payload(
            {"collective_seq_id": 4, "state": "completed"},
            {"collective_seq_id": 5, "pg_name": ["3", "tp", "ignored"]},
            {"collective_seq_id": 6, "pg_name": [], "state": "scheduled"},
            pg_config={"3": {"ranks": "[0, 1"}},
        )
    )

    assert summary["entry_count"] == 3, summary
    [group] = summary["last_entries_by_pg"]
    assert group["pg_name"] == "3:tp", "a list names the group by its first two parts"
    assert group["pg_config_key"] == "3", "the first part is the pg_config key"
    assert group["collective_seq_id"] == 5, group
    assert "pg_member_count" not in group and "pg_members" not in group, (
        "a member list that is not JSON proves no membership"
    )
    assert summary["last_entry"]["collective_seq_id"] == 6, summary["last_entry"]
    assert summary["last_entry"]["pg_name"] == "", (
        "the newest entry is still the newest, even without a group"
    )
    assert summary["unfinished_entry_count"] == 2, (
        "entries without a group still count as unfinished"
    )


def test_a_container_path_that_resolves_outside_its_root_is_refused(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    root = training_process(proc_root, 77, {})
    outside = tmp_path / "host-etc"
    outside.mkdir()
    (root / "escape").symlink_to(outside)

    with pytest.raises(ValueError, match="namespace path escapes container base"):
        FlightRecorderOperationsMixin.process_namespace_path_for_root(
            proc_root, 77, Path("/escape/nccl_trace_rank_0")
        )
    assert FlightRecorderOperationsMixin.process_namespace_path_for_root(
        proc_root, 77, Path("/tmp/nccl_trace_rank_0")
    ) == (root / "tmp" / "nccl_trace_rank_0"), "a path inside the root maps into it"


def test_a_rank_without_rank_triggers_its_pipe_and_waits_on_the_pipe_siblings(
    tmp_path: Path,
) -> None:
    """No ``RANK``: no rank-suffixed pipe, prefix or default can be named, so
    the Agent writes the pipe itself and waits on its siblings; an empty dump
    prefix contributes no path of its own."""

    proc_root = tmp_path / "proc"
    environment = {
        "TORCH_NCCL_DEBUG_INFO_PIPE_FILE": "/tmp/nccl_pipe_",
        "TORCH_FR_DUMP_TEMP_FILE": "",
        "TORCH_NCCL_TRACE_BUFFER_SIZE": "2000",
    }
    root = training_process(proc_root, 90, environment)
    pipe = root / "tmp" / "nccl_pipe_"
    os.mkfifo(pipe)
    reader = os.open(pipe, os.O_RDONLY | os.O_NONBLOCK)
    try:
        agent = RecorderHost(proc_root, tmp_path / "diagnostics")
        request = request_flight(
            agent,
            SimpleNamespace(command_id="cmd-triage"),
            {"pid": 90, "rank": None, "_environment": environment},
            "first",
        )
        written = os.read(reader, 16)
    finally:
        os.close(reader)

    assert written == b"1\n", "the dump was asked for through the rank's own pipe"
    assert request["status"] == "triggered", request
    assert request["pipe"] == str(pipe), request
    assert request["dump_candidates"] == [str(pipe), f"{pipe}.json"], request
    assert set(request["dump_sources"].values()) == {"pipe_sibling"}, request
    assert request["dump_prefix_variable"] == "TORCH_FR_DUMP_TEMP_FILE", (
        "the empty prefix is still the variable PyTorch resolves"
    )
    assert request["previous_mtimes"] == {}, "the FIFO itself is never a dump file"
    assert request["evidence_path"].endswith("-90-first.dump"), request


def test_an_existing_dump_of_a_rank_without_rank_is_found_by_its_prefix(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    root = training_process(
        proc_root, 91, {"TORCH_NCCL_DEBUG_INFO_TEMP_FILE": "/tmp/fr_dump"}
    )
    (root / "tmp" / "fr_dump.json").write_bytes(
        flight_payload({"collective_seq_id": 3, "pg_name": "0:default_pg"})
    )
    agent = RecorderHost(proc_root, tmp_path / "diagnostics")

    [record] = existing_flight.preserve_existing(agent, existing_flight_command())

    assert record["pid"] == 91 and record["rank"] is None, record
    assert record["capture_mode"] == "EXISTING_FILE_NO_TRIGGER", record
    assert record["status"] == "dumped" and record["parse_error"] is None, record
    assert record["identity_unchanged"] is True, record
    assert record["process_start_ticks"] == 5000, record
    evidence = record["raw_evidence"]
    assert evidence["status"] == "PRESERVED", evidence
    assert (
        Path(evidence["path"]).read_bytes()
        == (root / "tmp" / "fr_dump.json").read_bytes()
    ), "the preserved copy is the dump the rank wrote"


def test_a_dump_rotated_while_it_was_read_is_reported_and_not_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    root = training_process(
        proc_root,
        92,
        {"RANK": "2", "WORLD_SIZE": "4", "TORCH_FR_DUMP_TEMP_FILE": "/tmp/fr_"},
    )
    dump = root / "tmp" / "fr_2"
    dump.write_bytes(flight_payload({"collective_seq_id": 8, "pg_name": "0:pg"}))
    real_read = flight_recorder.bounded_read

    def read_then_rotate(read: Any, *arguments: Any) -> Any:
        value = real_read(read, *arguments)
        # The container replaced the dump in place while the Agent parsed it.
        dump.write_bytes(flight_payload())
        return value

    monkeypatch.setattr(flight_recorder, "bounded_read", read_then_rotate)
    agent = RecorderHost(proc_root, tmp_path / "diagnostics")

    [record] = existing_flight.preserve_existing(agent, existing_flight_command())

    assert record["rank"] == 2, record
    assert record["status"] == "dumped", record
    assert record["parse_error"] == "dump changed while being read", record
    assert record["raw_evidence"] is None, "a changed dump is never preserved"
    assert not (tmp_path / "diagnostics").exists() or not any(
        (tmp_path / "diagnostics").iterdir()
    ), "nothing was written to the evidence directory"
