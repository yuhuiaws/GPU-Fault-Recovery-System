#!/usr/bin/env python3
"""Test flight-recorder dump responder of GF-GENERAL-LIVE-011 (live-replay).

Runs *inside* the ``pytorch`` container of the managed two-node NCCL stall job
(``general_live_nccl_job.py`` writes this file next to the trainer and starts
it before ``exec torchrun``) as a stand-in for PyTorch's flight-recorder
monitor thread. The job itself runs with ``TORCH_NCCL_TRACE_BUFFER_SIZE=0``, so
PyTorch creates no debug pipe and writes no dump; the rank processes still
carry ``TORCH_NCCL_DEBUG_INFO_PIPE_FILE`` / ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE``
in their environment, which is where the product's ``COLLECT_HUNG_TRIAGE``
(``node_agent/operations/flight_recorder.py``) looks: it maps
``<pipe base><rank>.pipe`` into the rank's mount namespace, requires a FIFO,
writes ``b"1\\n"`` into it and then waits for ``<dump base><rank>`` to get a
newer mtime, twice per PID at least two seconds apart (``flight_series.py``).
This responder owns exactly those FIFOs for the ranks of its own Pod and
answers each request with a dump whose collective progress is *shaped* by the
variant (``general_live_fr_replay.py`` renders the shape): completion that
advances, an operation that stays pending, enqueue without completion, ranks
one collective behind their group, or a withheld second dump. Only the raw
reading is replaced; the FIFO protocol, namespace mapping, parsing (the dump is
``pickle`` protocol 2 like PyTorch's, or JSON), the two-dump series, the
classification and the bundle binding are the product's own. Evidence of a
variant built on it says ``proof_scope: live-replay``.

Bounds. Standard library only, Python 3.8+. It touches only
``<pipe base><rank>.pipe`` and ``<dump base><rank>`` of its own ranks inside its
own container and its state directory; it refuses a pipe path that already
exists (a real recorder owns it), holds no GPU and attaches to nothing. It
stops on its own at ``--max-seconds`` (the in-container analogue of a unit's
``RuntimeMaxSec``), on SIGTERM/SIGINT or on a ``stop`` file, and then removes
the FIFOs and dumps it created (only while they are still the inodes it made),
so the container returns to its no-recorder baseline without the runner. Its
ledger (``<state dir>/ledger.json``, written atomically after every event) is
the runner's raw-source oracle: every request received, answered or withheld,
with the watermarks each dump carried.

The FIFO is read through a reader opened ``O_RDONLY | O_NONBLOCK`` while an own
write end stays open: with the agent as the only writer, a request's closed
writer left the empty FIFO at end-of-file, ``select`` reported it readable at
once and a read that met the agent's *next* open with nothing written yet
raised EAGAIN (``tests/execution/_nccl_chain_node.py``). The holder keeps the
FIFO out of that state, so readable only ever means request bytes.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import pickle
import select
import signal
import stat
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SHAPE_VERSION = "gfrs-fr-replay/v1"
LEDGER_VERSION = "gfrs-fr-replay-ledger/v1"
PROGRESS_KINDS = ("advance", "pending", "enqueue_only")
FORMATS = ("pickle", "json")
MAX_SHAPE_BYTES = 1024 * 1024
MAX_RANKS = 4096
MAX_GROUPS = 64
MAX_MEMBERS = 65536
MAX_HISTORY = 64
MAX_SEQ = 2**40
MAX_SECONDS = 86400
PROFILING_NAME = "nccl:all_reduce"
#: 1.5 MiB of fp32, the stall job's payload (``general_live_nccl_job.py``).
PAYLOAD_ELEMENTS = 393216
REQUEST = b"1\n"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ShapeError(ValueError):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ShapeError(message)


def parse_shape(document: Any) -> Dict[str, Any]:
    """Validate a shape document (see ``general_live_fr_replay.py``)."""

    _check(isinstance(document, dict), "shape must be an object")
    _check(document.get("version") == SHAPE_VERSION, "unknown shape version")
    fmt = document.get("format", "pickle")
    _check(fmt in FORMATS, "shape format must be pickle or json")
    version = document.get("dump_version", "2.10")
    _check(isinstance(version, str) and 0 < len(version) <= 16, "bad dump_version")
    history = document.get("history", 6)
    _check(type(history) is int and 1 <= history <= MAX_HISTORY, "bad history")
    withhold = document.get("withhold_from_request")
    _check(
        withhold is None or (type(withhold) is int and withhold >= 1),
        "withhold_from_request must be a request index >= 1 or null",
    )
    ranks = document.get("ranks")
    _check(isinstance(ranks, dict) and 0 < len(ranks) <= MAX_RANKS, "bad ranks")
    for key, value in ranks.items():
        _check(isinstance(key, str) and key.isdecimal(), f"bad rank key {key!r}")
        rank = int(key)
        groups = value.get("groups") if isinstance(value, dict) else None
        _check(
            isinstance(groups, list) and 0 < len(groups) <= MAX_GROUPS,
            f"rank {rank}: bad groups",
        )
        seen = set()
        for group in groups:
            _check(isinstance(group, dict), f"rank {rank}: group must be an object")
            pg_id = group.get("pg_id")
            _check(type(pg_id) is int and 0 <= pg_id < 2**31, f"rank {rank}: bad pg_id")
            _check(pg_id not in seen, f"rank {rank}: duplicate pg_id {pg_id}")
            seen.add(pg_id)
            uid, desc = group.get("uid"), group.get("desc")
            _check(isinstance(uid, str) and uid.isdecimal(), f"rank {rank}: bad uid")
            _check(
                isinstance(desc, str) and 0 < len(desc) <= 64, f"rank {rank}: bad desc"
            )
            members = group.get("members")
            _check(
                isinstance(members, list)
                and 0 < len(members) <= MAX_MEMBERS
                and all(type(item) is int and item >= 0 for item in members)
                and len(set(members)) == len(members)
                and rank in members,
                f"rank {rank}: group {uid} members must be unique ranks including {rank}",
            )
            _check(
                group.get("progress") in PROGRESS_KINDS, f"rank {rank}: bad progress"
            )
            seq = group.get("seq")
            _check(
                type(seq) is int and history + 2 <= seq < MAX_SEQ,
                f"rank {rank}: seq must be at least history + 2 and under 2**40",
            )
    return {
        "version": SHAPE_VERSION,
        "format": fmt,
        "dump_version": version,
        "history": history,
        "withhold_from_request": withhold,
        "ranks": ranks,
    }


def shape_sha256(shape: Dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(shape, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# --------------------------------------------------------------------------- #
# dump rendering (pure)
# --------------------------------------------------------------------------- #
def group_watermarks(group: Dict[str, Any], request: int) -> Dict[str, Any]:
    """Enqueued/started/completed sequence numbers of ``group`` at ``request``.

    ``advance``: two more collectives completed per request, one started;
    ``pending``: the same started collective in every request; ``enqueue_only``:
    that, plus one more collective enqueued (scheduled) per request.
    """

    seq, kind = int(group["seq"]), group["progress"]
    if kind == "advance":
        completed = seq + 2 * (request - 1)
        pending, scheduled = completed + 1, []
    elif kind == "pending":
        completed, pending, scheduled = seq - 1, seq, []
    else:
        completed, pending = seq - 1, seq
        scheduled = list(range(seq + 1, seq + request))
    return {
        "last_enqueued": scheduled[-1] if scheduled else pending,
        "last_started": pending,
        "last_completed": completed,
        "pending_collective_seq_id": pending,
        "scheduled": scheduled,
    }


def _entry(
    group: Dict[str, Any], seq: int, state: str, base_time_ns: int
) -> Dict[str, Any]:
    # A collective keeps its record identity (record id, op id, creation time)
    # for life, so a pending operation reads as the same operation in both
    # dumps (``flight_progress.operation_key``) and a completed one as the
    # completion of what was pending before.
    created = base_time_ns + seq * 1_000_000
    started = created + 100_000 if state in ("started", "completed") else None
    completed = created + 2_000_000 if state == "completed" else None
    return {
        "record_id": int(group["pg_id"]) * 10_000_000 + seq,
        "pg_id": int(group["pg_id"]),
        "process_group": (group["uid"], group["desc"]),
        "collective_seq_id": seq,
        "p2p_seq_id": 0,
        "op_id": seq,
        "profiling_name": PROFILING_NAME,
        "time_created_ns": created,
        "input_sizes": [[PAYLOAD_ELEMENTS]],
        "input_dtypes": ["Float"],
        "output_sizes": [[PAYLOAD_ELEMENTS]],
        "output_dtypes": ["Float"],
        "state": state,
        "time_discovered_started_ns": started,
        "time_discovered_completed_ns": completed,
        "retired": state == "completed",
        "timeout_ms": 3600000,
        "is_p2p": False,
        "frames": [],
        "duration_ms": 1.0 if state == "completed" else None,
    }


def render_dump(
    shape: Dict[str, Any], rank: int, request: int, *, base_time_ns: int
) -> Dict[str, Any]:
    """The dump document rank ``rank`` answers request ``request`` (1-based)."""

    groups = shape["ranks"][str(rank)]["groups"]
    history = int(shape["history"])
    config: Dict[str, Any] = {}
    status: Dict[str, Any] = {}
    entries: List[Dict[str, Any]] = []
    for group in groups:
        marks = group_watermarks(group, request)
        completed = marks["last_completed"]
        for seq in range(completed - history + 1, completed + 1):
            entries.append(_entry(group, seq, "completed", base_time_ns))
        entries.append(_entry(group, marks["last_started"], "started", base_time_ns))
        for seq in marks["scheduled"]:
            entries.append(_entry(group, seq, "scheduled", base_time_ns))
        config[group["uid"]] = {
            "name": group["uid"],
            "desc": group["desc"],
            "ranks": str(list(group["members"])),
        }
        status[str(group["pg_id"])] = {
            "last_enqueued_collective": marks["last_enqueued"],
            "last_started_collective": marks["last_started"],
            "last_completed_collective": marks["last_completed"],
        }
    entries.sort(key=lambda item: item["record_id"])
    return {
        "version": shape["dump_version"],
        "pg_config": config,
        "pg_status": status,
        "entries": entries,
    }


def encode_dump(document: Dict[str, Any], fmt: str) -> bytes:
    if fmt == "pickle":
        return pickle.dumps(document, protocol=2)
    return json.dumps(document).encode("utf-8")


def watermark_summary(shape: Dict[str, Any], rank: int, request: int) -> Dict[str, Any]:
    """What the ledger records about one dump: the watermarks per group."""

    return {
        f"{group['uid']}:{group['desc']}": {
            key: value
            for key, value in group_watermarks(group, request).items()
            if key != "scheduled"
        }
        for group in shape["ranks"][str(rank)]["groups"]
    }


# --------------------------------------------------------------------------- #
# the responder
# --------------------------------------------------------------------------- #
def _absolute_base(value: str, label: str) -> str:
    if (
        not value.startswith("/")
        or "/../" in value + "/"
        or not os.path.isdir(os.path.dirname(value) or "/")
    ):
        raise ShapeError(f"{label} must be an absolute path in an existing directory")
    return value


def write_json_atomic(path: str, document: Dict[str, Any]) -> None:
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(document, stream, sort_keys=True)
    os.replace(temporary, path)


class Responder:
    def __init__(
        self,
        *,
        state_dir: str,
        pipe_base: str,
        dump_base: str,
        ranks: List[int],
        world_size: int,
        shape: Dict[str, Any],
        max_seconds: float,
        fmt: Optional[str] = None,
        poll_seconds: float = 0.05,
    ) -> None:
        if not 1 <= max_seconds <= MAX_SECONDS:
            raise ShapeError(f"max-seconds must be within 1..{MAX_SECONDS}")
        if not ranks or len(set(ranks)) != len(ranks) or min(ranks) < 0:
            raise ShapeError("ranks must be distinct non-negative integers")
        missing = [rank for rank in ranks if str(rank) not in shape["ranks"]]
        if missing:
            raise ShapeError(f"the shape has no groups for ranks {missing}")
        self.state_dir = _absolute_base(state_dir, "state-dir")
        self.pipe_base = _absolute_base(pipe_base, "pipe-base")
        self.dump_base = _absolute_base(dump_base, "dump-base")
        self.ranks = list(ranks)
        self.world_size = world_size
        self.shape = shape
        self.format = fmt or shape["format"]
        if self.format not in FORMATS:
            raise ShapeError("format must be pickle or json")
        self.max_seconds = float(max_seconds)
        self.poll_seconds = poll_seconds
        self.base_time_ns = time.time_ns()
        self.readers: Dict[int, int] = {}
        self.holders: Dict[int, int] = {}
        self.pipe_inodes: Dict[int, int] = {}
        self.dump_inodes: Dict[int, int] = {}
        self.stop_requested: Optional[str] = None
        self.started = time.monotonic()
        self.ledger: Dict[str, Any] = {
            "version": LEDGER_VERSION,
            "status": "starting",
            "finish_reason": None,
            "pid": os.getpid(),
            "started_at": utc_now(),
            "deadline_at": datetime.fromtimestamp(
                time.time() + self.max_seconds, timezone.utc
            ).isoformat(),
            "finished_at": None,
            "max_seconds": self.max_seconds,
            "shape_sha256": shape_sha256(shape),
            "format": self.format,
            "dump_version": shape["dump_version"],
            "withhold_from_request": shape["withhold_from_request"],
            "pipe_base": self.pipe_base,
            "dump_base": self.dump_base,
            "world_size": world_size,
            "ranks": {
                str(rank): {
                    "pipe": self.pipe_path(rank),
                    "dump": self.dump_path(rank),
                    "requests": 0,
                    "answered": 0,
                    "withheld": 0,
                    "events": [],
                }
                for rank in self.ranks
            },
            "unexpected_payloads": 0,
            "errors": [],
        }

    # -- paths --------------------------------------------------------------- #
    def pipe_path(self, rank: int) -> str:
        # PyTorch's own pipe name: ``c10::str(fileStem, rank, ".pipe")``.
        return f"{self.pipe_base}{rank}.pipe"

    def dump_path(self, rank: int) -> str:
        # PyTorch's own dump name: ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE`` + rank.
        return f"{self.dump_base}{rank}"

    @property
    def ledger_path(self) -> str:
        return os.path.join(self.state_dir, "ledger.json")

    @property
    def stop_path(self) -> str:
        return os.path.join(self.state_dir, "stop")

    def save(self) -> None:
        write_json_atomic(self.ledger_path, self.ledger)

    def fail(self, message: str) -> None:
        self.ledger["errors"].append({"at": utc_now(), "error": message[:512]})

    # -- lifecycle ----------------------------------------------------------- #
    def start(self) -> None:
        os.makedirs(self.state_dir, mode=0o700, exist_ok=True)
        if os.path.exists(self.stop_path):
            os.unlink(self.stop_path)
        for rank in self.ranks:
            pipe = self.pipe_path(rank)
            if os.path.lexists(pipe):
                raise ShapeError(f"{pipe} exists already; a real recorder may own it")
            os.mkfifo(pipe, 0o666)
            self.pipe_inodes[rank] = os.stat(pipe).st_ino
            self.readers[rank] = os.open(pipe, os.O_RDONLY | os.O_NONBLOCK)
            self.holders[rank] = os.open(pipe, os.O_WRONLY | os.O_NONBLOCK)
        self.ledger["status"] = "serving"
        self.save()

    def request_stop(self, reason: str) -> None:
        if self.stop_requested is None:
            self.stop_requested = reason

    def serve(self) -> str:
        """Answer requests until the deadline, a stop file or a signal."""

        by_fd = {fd: rank for rank, fd in self.readers.items()}
        deadline = self.started + self.max_seconds
        reason = "deadline"
        while True:
            remaining = deadline - time.monotonic()
            if self.stop_requested is not None:
                reason = self.stop_requested
                break
            if remaining <= 0:
                break
            if os.path.exists(self.stop_path):
                reason = "stop"
                break
            try:
                ready = select.select(
                    list(by_fd), [], [], min(self.poll_seconds, remaining)
                )[0]
            except InterruptedError:
                continue
            for fd in ready:
                try:
                    payload = os.read(fd, 4096)
                except BlockingIOError:
                    continue
                except OSError as exc:
                    self.fail(f"read of rank {by_fd[fd]} pipe: {exc}")
                    self.save()
                    continue
                if payload:
                    self.answer(by_fd[fd], payload)
        self.finish(reason)
        return reason

    def answer(self, rank: int, payload: bytes) -> None:
        record = self.ledger["ranks"][str(rank)]
        requests = payload.count(REQUEST) or 1
        if payload.replace(REQUEST, b""):
            self.ledger["unexpected_payloads"] += 1
        for _ in range(requests):
            record["requests"] += 1
            index = record["requests"]
            event: Dict[str, Any] = {
                "request": index,
                "received_at": utc_now(),
                "payload_bytes": len(payload),
            }
            withhold = self.shape["withhold_from_request"]
            if withhold is not None and index >= withhold:
                event["action"] = "withheld"
                record["withheld"] += 1
            else:
                try:
                    event.update(self.write_dump(rank, index))
                    event["action"] = "answered"
                    record["answered"] += 1
                except OSError as exc:
                    event["action"] = "failed"
                    event["error"] = str(exc)[:256]
                    self.fail(f"rank {rank} request {index}: {exc}")
            record["events"].append(event)
        self.save()

    def write_dump(self, rank: int, request: int) -> Dict[str, Any]:
        raw = encode_dump(
            render_dump(self.shape, rank, request, base_time_ns=self.base_time_ns),
            self.format,
        )
        target = self.dump_path(rank)
        temporary = os.path.join(self.state_dir, f"pending-{rank}")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
        os.replace(temporary, target)
        self.dump_inodes[rank] = os.stat(target).st_ino
        return {
            "answered_at": utc_now(),
            "dump": target,
            "dump_bytes": len(raw),
            "dump_sha256": hashlib.sha256(raw).hexdigest(),
            "watermarks": watermark_summary(self.shape, rank, request),
        }

    def finish(self, reason: str) -> None:
        """Close and remove what this responder created; the baseline is back."""

        removed = {"pipes": [], "dumps": []}  # type: Dict[str, List[str]]
        for rank in self.ranks:
            for mapping in (self.readers, self.holders):
                descriptor = mapping.pop(rank, None)
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
            for kind, path, inode in (
                ("pipes", self.pipe_path(rank), self.pipe_inodes.get(rank)),
                ("dumps", self.dump_path(rank), self.dump_inodes.get(rank)),
            ):
                if inode is None:
                    continue
                try:
                    info = os.lstat(path)
                    if info.st_ino == inode and (
                        stat.S_ISFIFO(info.st_mode) or stat.S_ISREG(info.st_mode)
                    ):
                        os.unlink(path)
                        removed[kind].append(path)
                except OSError as exc:
                    if exc.errno != errno.ENOENT:
                        self.fail(f"removing {path}: {exc}")
        self.ledger.update(
            {
                "status": "finished",
                "finish_reason": reason,
                "finished_at": utc_now(),
                "removed": removed,
            }
        )
        self.save()


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #
def load_shape(arguments: argparse.Namespace) -> Dict[str, Any]:
    if arguments.shape_json:
        with open(arguments.shape_json, "rb") as stream:
            raw = stream.read(MAX_SHAPE_BYTES + 1)
    else:
        raw = os.environ.get(arguments.shape_env, "").encode("utf-8")
    if not raw:
        raise ShapeError("no shape: give --shape-json or set the --shape-env variable")
    if len(raw) > MAX_SHAPE_BYTES:
        raise ShapeError("shape exceeds 1 MiB")
    return parse_shape(json.loads(raw.decode("utf-8")))


def served_ranks(arguments: argparse.Namespace) -> "tuple[List[int], int]":
    """``(ranks, world size)`` from ``--ranks`` or the torchrun Pod environment."""

    if arguments.ranks:
        ranks = [int(item) for item in arguments.ranks.split(",") if item.strip()]
        if arguments.world_size is None:
            raise ShapeError("--ranks needs --world-size")
        return ranks, int(arguments.world_size)
    per_node = int(arguments.procs_per_node)
    if not 1 <= per_node <= 64:
        raise ShapeError("--procs-per-node must be within 1..64")
    try:
        node_rank = int(os.environ[arguments.node_rank_env])
        nodes = int(os.environ[arguments.world_size_env])
    except (KeyError, ValueError) as exc:
        raise ShapeError(f"node rank / world size environment unusable: {exc}") from exc
    if node_rank < 0 or nodes < 1 or node_rank >= nodes:
        raise ShapeError("node rank must be within the number of nodes")
    return list(
        range(node_rank * per_node, (node_rank + 1) * per_node)
    ), nodes * per_node


def read_ledger(state_dir: str) -> Dict[str, Any]:
    with open(os.path.join(state_dir, "ledger.json"), "r", encoding="utf-8") as stream:
        return json.load(stream)


def command_serve(arguments: argparse.Namespace) -> int:
    shape = load_shape(arguments)
    ranks, world_size = served_ranks(arguments)
    responder = Responder(
        state_dir=arguments.state_dir,
        pipe_base=arguments.pipe_base,
        dump_base=arguments.dump_base,
        ranks=ranks,
        world_size=world_size,
        shape=shape,
        max_seconds=float(arguments.max_seconds),
        fmt=arguments.format,
        poll_seconds=float(arguments.poll_seconds),
    )
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(
            signum,
            lambda number, _frame: responder.request_stop(
                "signal:" + signal.Signals(number).name
            ),
        )
    responder.start()
    reason = responder.serve()
    print(json.dumps({"status": "finished", "reason": reason}))
    return 0


def command_stop(arguments: argparse.Namespace) -> int:
    stop_path = os.path.join(arguments.state_dir, "stop")
    with open(stop_path, "w", encoding="utf-8") as stream:
        stream.write(utc_now())
    deadline = time.monotonic() + float(arguments.timeout)
    ledger: Dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            ledger = read_ledger(arguments.state_dir)
        except (OSError, ValueError):
            ledger = {}
        if ledger.get("status") == "finished":
            break
        time.sleep(0.2)
    print(json.dumps({"stopped": ledger.get("status") == "finished", "ledger": ledger}))
    return 0 if ledger.get("status") == "finished" else 1


def command_ledger(arguments: argparse.Namespace) -> int:
    print(json.dumps(read_ledger(arguments.state_dir), sort_keys=True))
    return 0


def command_cleanup(arguments: argparse.Namespace) -> int:
    """Remove what a responder that was killed left behind (ledger-listed)."""

    try:
        ledger = read_ledger(arguments.state_dir)
    except (OSError, ValueError) as exc:
        print(json.dumps({"removed": [], "error": f"no ledger: {exc}"}))
        return 0
    removed = []
    for record in (ledger.get("ranks") or {}).values():
        for key, check in (("pipe", stat.S_ISFIFO), ("dump", stat.S_ISREG)):
            path = record.get(key)
            try:
                if path and check(os.lstat(path).st_mode):
                    os.unlink(path)
                    removed.append(path)
            except OSError:
                continue
    print(json.dumps({"removed": removed, "status": ledger.get("status")}))
    return 0


def command_render(arguments: argparse.Namespace) -> int:
    shape = load_shape(arguments)
    document = render_dump(
        shape, int(arguments.rank), int(arguments.request), base_time_ns=0
    )
    print(json.dumps(document, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--state-dir", required=True)
    serve.add_argument("--pipe-base", required=True)
    serve.add_argument("--dump-base", required=True)
    serve.add_argument("--max-seconds", type=float, required=True)
    serve.add_argument("--format", choices=FORMATS, default=None)
    serve.add_argument("--poll-seconds", type=float, default=0.05)
    serve.add_argument("--ranks", default="")
    serve.add_argument("--world-size", type=int, default=None)
    serve.add_argument("--node-rank-env", default="RANK")
    serve.add_argument("--world-size-env", default="WORLD_SIZE")
    serve.add_argument("--procs-per-node", type=int, default=8)
    for sub in (serve, commands.add_parser("render")):
        sub.add_argument("--shape-json", default="")
        sub.add_argument("--shape-env", default="GFRS_FR_REPLAY_SHAPE")
    render = commands.choices["render"]
    render.add_argument("--rank", type=int, required=True)
    render.add_argument("--request", type=int, required=True)
    for name in ("stop", "ledger", "cleanup"):
        sub = commands.add_parser(name)
        sub.add_argument("--state-dir", required=True)
    commands.choices["stop"].add_argument("--timeout", type=float, default=10.0)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    handlers = {
        "serve": command_serve,
        "stop": command_stop,
        "ledger": command_ledger,
        "cleanup": command_cleanup,
        "render": command_render,
    }
    try:
        return handlers[arguments.command](arguments)
    except (ShapeError, ValueError) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
