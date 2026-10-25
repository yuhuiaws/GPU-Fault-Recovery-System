"""The reviewed test flight-recorder dump replay of GF-GENERAL-LIVE-011.

The deep-triage variants of 011 need the product's ``COLLECT_HUNG_TRIAGE`` to
read flight-recorder dumps of a known shape -- a second dump that completes the
first, a second dump that never comes, a whole process group pending, one to
three ranks behind their group, too many ranks behind, groups whose progress
must never be compared. PyTorch's own recorder cannot be told what to dump, so
the managed stall job (``general_live_nccl_job.ManagedNcclJob``) runs with
``TORCH_NCCL_TRACE_BUFFER_SIZE=0`` -- PyTorch creates no debug pipe and writes
no dump -- while the test responder ``probes/general_live_fr_responder.py``
runs inside each ``pytorch`` container, owns the ``<pipe base><rank>.pipe``
FIFOs the product derives from the ranks' ``TORCH_NCCL_DEBUG_INFO_PIPE_FILE``
and answers every ``b"1\\n"`` with a dump of the variant's shape at
``<dump base><rank>``. Only that raw reading is replaced; the FIFO request
protocol, the mount-namespace mapping, the dump parsing, the two-dump series,
the classification (``execution/hung_classification.py``,
``flight_temporal.py``) and the bundle binding (``hung_bundle.py``) are the
product's. Everything built on it is ``proof_scope: live-replay``.

Shapes. A :class:`FlightRecorderReplay` names process groups; each
:class:`GroupShape` is the job's default group (``members="all"``), one group
per node (``members="node"``: the node's eight ranks, consecutive group uids
per node) or an explicit rank list, with a progress kind -- ``advance`` (two
more collectives complete per dump), ``pending`` (the same started collective
in both dumps) or ``enqueue_only`` (pending, plus one more collective enqueued
per dump) -- a base sequence number and, for ``pending`` groups, the local ranks
that lag one collective behind. :meth:`FlightRecorderReplay.document` expands
that into the per-rank document the responder consumes (``gfrs-fr-replay/v1``).

Why the deciding groups are node-local. The product compares collective
progress across ranks only inside a common observation window; across nodes
that needs a trusted ``clock_uncertainty_seconds`` step parameter the hung DAG
of this site does not carry, and "an unknown clock never authorizes a targeted
bundle" is product doctrine (``docs/components/nccl-hung-hardening.md``,
``tests/execution/test_nccl_review_e2e.py``). A group whose members all run on
one node compares on that node's own clock, so the groups that must yield
leads or a whole-group stall are per-node groups (as tensor-parallel groups are
in practice); the cross-node default group is present in every dump as it is in
a real one, and its watermark deltas are still read per rank.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROOF_SCOPE = "live-replay"
RESPONDER_SCRIPT = Path(__file__).with_name("probes") / "general_live_fr_responder.py"
#: Where the container command writes the responder and keeps its state.
RESPONDER_PATH = "/tmp/gfrs_fr_responder.py"
STATE_DIR = "/tmp/gfrs-fr-replay"
LEDGER_PATH = f"{STATE_DIR}/ledger.json"
LOG_PATH = f"{STATE_DIR}/responder.log"
#: The pipe base and the dump base the operations manual prescribes (the dump
#: base is PyTorch 2.8's default; 2.9 defaults to ``$HOME/.cache/torch/
#: comm_lib_trace_rank_`` and reads ``TORCH_FR_DUMP_TEMP_FILE`` first); all are
#: set explicitly in the job's environment, the dump base under both PyTorch
#: names, so the product and the responder read the same names.
PIPE_BASE = "/tmp/torch_nccl_debug_pipe_"
DUMP_BASE = "/tmp/nccl_trace_rank_"
SHAPE_ENV = "GFRS_FR_REPLAY_SHAPE"
HEREDOC = "GFRS_FR_RESPONDER"
#: The responder's own RuntimeMaxSec analogue: longer than the trainer's whole
#: life (3600 steps of 2 s), so it outlives every triage and still ends alone.
RESPONDER_MAX_SECONDS = 7200
SHAPE_VERSION = "gfrs-fr-replay/v1"
ADVANCE, PENDING, ENQUEUE_ONLY = "advance", "pending", "enqueue_only"
KINDS = (ADVANCE, PENDING, ENQUEUE_ONLY)
#: Watermark deltas between the two dumps the product pairs, per kind
#: (``probes/general_live_fr_responder.py`` ``group_watermarks``).
DELTAS = {
    ADVANCE: {"last_enqueued": 2, "last_started": 2, "last_completed": 2},
    PENDING: {"last_enqueued": 0, "last_started": 0, "last_completed": 0},
    ENQUEUE_ONLY: {"last_enqueued": 1, "last_started": 0, "last_completed": 0},
}
DEFAULT_UID, DEFAULT_DESC = "0", "default_pg"
#: The two-node job: two Pods of eight ranks (``nvidia.com/gpu: 8``); Master is
#: torchrun node rank 0 (ranks 0-7, ``--fault-node``), Worker node rank 1.
NODES = 2
PROCS_PER_NODE = 8
WORLD_SIZE = NODES * PROCS_PER_NODE


def node_ranks(
    nodes: Sequence[str], procs_per_node: int = PROCS_PER_NODE
) -> dict[str, list[int]]:
    """Each attempt node's global ranks, in the job's Master, Worker order."""

    return {
        node: list(range(index * procs_per_node, (index + 1) * procs_per_node))
        for index, node in enumerate(nodes)
    }


def pg_name(uid: str, desc: str) -> str:
    """The product's group name: ``process_group: (uid, desc)`` joined."""

    return f"{uid}:{desc}"


@dataclass(frozen=True)
class GroupShape:
    """One process group of every rank's dump."""

    uid: str
    desc: str
    progress: str
    seq: int
    #: ``"all"``, ``"node"`` or the explicit global ranks.
    members: str | tuple[int, ...] = "all"
    #: Ranks one collective behind (``pending`` groups only): local ranks of
    #: every node for a ``node`` group, global ranks otherwise.
    lagging_local: tuple[int, ...] = ()
    #: A ``node`` group's base sequence per node (``seq`` for every node if unset).
    seq_by_node: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.progress not in KINDS:
            raise ValueError(f"progress must be one of {KINDS}, not {self.progress!r}")
        if not self.uid.isdecimal():
            raise ValueError("a group uid is PyTorch's decimal group counter")
        if self.lagging_local and self.progress != PENDING:
            raise ValueError("only a pending group has lagging ranks")
        if isinstance(self.members, str) and self.members not in {"all", "node"}:
            raise ValueError("members is 'all', 'node' or explicit ranks")

    def groups_for(
        self, rank: int, *, world_size: int, procs_per_node: int
    ) -> list[dict[str, Any]]:
        """The expanded group of ``rank`` (none when the rank is no member)."""

        node = rank // procs_per_node
        if self.members == "all":
            members = list(range(world_size))
            uid, seq = self.uid, self.seq
            lagging = set(self.lagging_local)
        elif self.members == "node":
            members = list(range(node * procs_per_node, (node + 1) * procs_per_node))
            uid = str(int(self.uid) + node)
            seq = self.seq_by_node[node] if self.seq_by_node else self.seq
            lagging = {node * procs_per_node + local for local in self.lagging_local}
        else:
            members = sorted(self.members)
            uid, seq = self.uid, self.seq
            lagging = set(self.lagging_local)
        if rank not in members:
            return []
        return [
            {
                "pg_id": int(uid),
                "uid": uid,
                "desc": self.desc,
                "members": members,
                "progress": self.progress,
                "seq": seq - 1 if rank in lagging else seq,
            }
        ]


@dataclass(frozen=True)
class FlightRecorderReplay:
    """The dump shape one variant replays through the test responder."""

    name: str
    groups: tuple[GroupShape, ...]
    #: The request index (1-based) from which the responder answers nothing.
    withhold_from_request: int | None = None
    dump_version: str = "2.10"
    format: str = "pickle"
    history: int = 6

    def document(
        self, *, world_size: int = WORLD_SIZE, procs_per_node: int = PROCS_PER_NODE
    ) -> dict[str, Any]:
        ranks = {}
        for rank in range(world_size):
            groups = [
                group
                for shape in self.groups
                for group in shape.groups_for(
                    rank, world_size=world_size, procs_per_node=procs_per_node
                )
            ]
            if not groups:
                raise ValueError(f"rank {rank} belongs to no group of {self.name}")
            ids = [group["pg_id"] for group in groups]
            if len(set(ids)) != len(ids):
                raise ValueError(
                    f"rank {rank}: group ids collide in {self.name}: {ids}"
                )
            ranks[str(rank)] = {"groups": groups}
        return {
            "version": SHAPE_VERSION,
            "format": self.format,
            "dump_version": self.dump_version,
            "history": self.history,
            "withhold_from_request": self.withhold_from_request,
            "ranks": ranks,
        }

    def shape_json(self, **options: int) -> str:
        return json.dumps(
            self.document(**options), sort_keys=True, separators=(",", ":")
        )

    def sha256(self, **options: int) -> str:
        return hashlib.sha256(self.shape_json(**options).encode()).hexdigest()

    # -- what the product should read ------------------------------------- #
    def rank_groups(
        self,
        rank: int,
        *,
        world_size: int = WORLD_SIZE,
        procs_per_node: int = PROCS_PER_NODE,
    ) -> list[dict[str, Any]]:
        return self.document(world_size=world_size, procs_per_node=procs_per_node)[
            "ranks"
        ][str(rank)]["groups"]

    def group_names(self, **options: int) -> set[str]:
        """Every group name any rank's dump carries."""

        return {
            pg_name(group["uid"], group["desc"])
            for value in self.document(**options)["ranks"].values()
            for group in value["groups"]
        }

    def lagging_ranks(self, **options: int) -> set[int]:
        """The ranks the shape puts one collective behind their group."""

        document = self.document(**options)
        groups_by_name: dict[str, list[tuple[int, int]]] = {}
        for rank, value in document["ranks"].items():
            for group in value["groups"]:
                groups_by_name.setdefault(
                    f"{group['uid']}:{group['desc']}:{group['members']}", []
                ).append((int(rank), group["seq"]))
        behind = set()
        for rows in groups_by_name.values():
            top = max(seq for _rank, seq in rows)
            behind.update(rank for rank, seq in rows if seq < top)
        return behind

    def evidence(self, **options: int) -> dict[str, Any]:
        return {
            "proof_scope": PROOF_SCOPE,
            "replaced": (
                "the rank processes' flight-recorder dumps (PyTorch recorder off, "
                "TORCH_NCCL_TRACE_BUFFER_SIZE=0); the test responder answers the "
                "product's FIFO requests with dumps of this shape"
            ),
            "shape": self.name,
            "shape_sha256": self.sha256(**options),
            "groups": [
                {
                    "uid": item.uid,
                    "desc": item.desc,
                    "progress": item.progress,
                    "seq": item.seq,
                    "members": item.members,
                    "lagging_local": list(item.lagging_local),
                    "seq_by_node": list(item.seq_by_node) if item.seq_by_node else None,
                }
                for item in self.groups
            ],
            "withhold_from_request": self.withhold_from_request,
            "dump_version": self.dump_version,
            "format": self.format,
            "responder_sha256": responder_sha256(),
        }


# --------------------------------------------------------------------------- #
# the seven shapes
# --------------------------------------------------------------------------- #
DEFAULT_ADVANCE = GroupShape(DEFAULT_UID, DEFAULT_DESC, ADVANCE, 1200)
DEFAULT_PENDING = GroupShape(DEFAULT_UID, DEFAULT_DESC, PENDING, 1200)
#: The per-node group (uid 1 on the first node, 2 on the second), a tensor
#: parallel group in practice; ``cp`` is a second per-node group (uids 3, 4).
TP = ("1", "tp")
CP = ("3", "cp")


def few_candidates_shape() -> FlightRecorderReplay:
    """One rank per node a collective behind its node group: 1-2 candidates."""

    return FlightRecorderReplay(
        "few_candidates",
        (DEFAULT_ADVANCE, GroupShape(*TP, PENDING, 800, "node", lagging_local=(0,))),
    )


def ambiguous_candidates_shape() -> FlightRecorderReplay:
    """Two node groups with two laggards each: four leads on one node alone."""

    return FlightRecorderReplay(
        "ambiguous_candidates",
        (
            DEFAULT_ADVANCE,
            GroupShape(*TP, PENDING, 800, "node", lagging_local=(0, 1)),
            GroupShape(*CP, PENDING, 640, "node", lagging_local=(2, 3)),
        ),
    )


def different_process_groups_shape() -> FlightRecorderReplay:
    """The node groups pend at very different sequence numbers (1200 vs 700)."""

    return FlightRecorderReplay(
        "different_process_groups",
        (
            DEFAULT_ADVANCE,
            GroupShape(*TP, PENDING, 1200, "node", seq_by_node=(1200, 700)),
        ),
    )


def temporal_completion_shape() -> FlightRecorderReplay:
    """Completion advancing in one group, enqueue without completion in another."""

    return FlightRecorderReplay(
        "fr_temporal_completion",
        (DEFAULT_ADVANCE, GroupShape("1", "gfrs_enqueue_only", ENQUEUE_ONLY, 900)),
    )


def second_missing_shape() -> FlightRecorderReplay:
    """The first dump as usual; nothing answers the second request."""

    return FlightRecorderReplay(
        "fr_second_missing", (DEFAULT_PENDING,), withhold_from_request=2
    )


def whole_group_shape() -> FlightRecorderReplay:
    """Every rank pending on the same collective of its node group and the job."""

    return FlightRecorderReplay(
        "fr_whole_group", (DEFAULT_PENDING, GroupShape(*TP, PENDING, 800, "node"))
    )


def plus_process_shape() -> FlightRecorderReplay:
    """Completion advancing everywhere, next to the held ranks' checkpoint I/O."""

    return FlightRecorderReplay(
        "fr_plus_process", (DEFAULT_ADVANCE, GroupShape(*TP, ADVANCE, 800, "node"))
    )


SHAPES = {
    "few_candidates": few_candidates_shape,
    "ambiguous_candidates": ambiguous_candidates_shape,
    "different_process_groups": different_process_groups_shape,
    "fr_temporal_completion": temporal_completion_shape,
    "fr_second_missing": second_missing_shape,
    "fr_whole_group": whole_group_shape,
    "fr_plus_process": plus_process_shape,
}


# --------------------------------------------------------------------------- #
# container embedding
# --------------------------------------------------------------------------- #
def responder_source() -> str:
    text = RESPONDER_SCRIPT.read_text(encoding="utf-8")
    if any(line == HEREDOC for line in text.splitlines()):
        raise ValueError("the responder would end its own heredoc")
    return text


def responder_sha256() -> str:
    return hashlib.sha256(responder_source().encode("utf-8")).hexdigest()


def responder_environment(
    replay: FlightRecorderReplay, *, procs_per_node: int
) -> dict[str, str]:
    """The job environment of a replayed recorder.

    PyTorch's recorder is off (buffer 0, no dump on timeout); the pipe and dump
    bases stay in every rank's environment because the product reads them from
    there, and the shape travels in the Pod's environment for the responder.
    """

    return {
        "TORCH_NCCL_TRACE_BUFFER_SIZE": "0",
        "TORCH_NCCL_DEBUG_INFO_PIPE_FILE": PIPE_BASE,
        "TORCH_FR_DUMP_TEMP_FILE": DUMP_BASE,
        "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": DUMP_BASE,
        SHAPE_ENV: replay.shape_json(
            world_size=NODES * procs_per_node, procs_per_node=procs_per_node
        ),
    }


def responder_launch(*, procs_per_node: int) -> str:
    """The shell lines that write and start the responder before torchrun.

    Started in the background with its output in its state directory, so it
    survives bash's ``exec`` into torchrun and stays in the container's own
    cgroup; it exits by itself at ``RESPONDER_MAX_SECONDS``.
    """

    return (
        f"cat >{RESPONDER_PATH} <<'{HEREDOC}'\n{responder_source()}{HEREDOC}\n"
        f"mkdir -p {STATE_DIR}\n"
        f"python3 -u {RESPONDER_PATH} serve --state-dir {STATE_DIR}"
        f" --pipe-base {PIPE_BASE} --dump-base {DUMP_BASE} --shape-env {SHAPE_ENV}"
        f" --node-rank-env RANK --world-size-env WORLD_SIZE"
        f" --procs-per-node {procs_per_node} --max-seconds {RESPONDER_MAX_SECONDS}"
        f" >{LOG_PATH} 2>&1 &\n"
    )


# --------------------------------------------------------------------------- #
# the ledger oracle (pure)
# --------------------------------------------------------------------------- #
LEDGER_VERSION = "gfrs-fr-replay-ledger/v1"


def ledger_node(ledger: Mapping[str, Any], nodes: Sequence[str]) -> str | None:
    """The attempt node whose rank block a ledger served, or None."""

    ranks = sorted(
        int(key) for key in (ledger.get("ranks") or {}) if str(key).isdecimal()
    )
    for node, expected in node_ranks(nodes).items():
        if ranks == expected:
            return node
    return None


def responder_errors(
    ledgers: Mapping[str, Any], nodes: Sequence[str], *, pipe_base: str = PIPE_BASE
) -> list[str]:
    """Why the responders' ledgers do not prove the replay fixture held.

    Each attempt node's Pod has exactly one responder, serving (or finished
    after its deadline) the node's rank block with the FIFOs at the product's
    names (``pipe_base`` -- the one the job's environment names), no responder
    error and no failed dump write. These are fixture faults -- the variant is
    INCONCLUSIVE on them, the product is not judged.
    """

    errors: list[str] = []
    served: dict[str, str] = {}
    for pod, ledger in sorted(ledgers.items()):
        if not isinstance(ledger, dict) or ledger.get("version") != LEDGER_VERSION:
            errors.append(f"{pod}: no responder ledger ({ledger!r})")
            continue
        if ledger.get("status") not in {"serving", "finished"}:
            errors.append(f"{pod}: responder status {ledger.get('status')!r}")
        if ledger.get("errors"):
            errors.append(f"{pod}: responder errors {ledger['errors'][:3]}")
        if ledger.get("pipe_base") != pipe_base:
            errors.append(
                f"{pod}: the responder serves pipes under {ledger.get('pipe_base')!r}, "
                f"the product reads {pipe_base!r}"
            )
        node = ledger_node(ledger, nodes)
        if node is None:
            errors.append(f"{pod}: the responder serves ranks of no attempt node")
        elif node in served:
            errors.append(f"{pod}: a second responder for {node} ({served[node]})")
        else:
            served[node] = pod
        for key, record in sorted((ledger.get("ranks") or {}).items()):
            if record.get("pipe") != f"{ledger.get('pipe_base')}{key}.pipe":
                errors.append(f"{pod}: rank {key} pipe is {record.get('pipe')}")
            failed = [
                e for e in record.get("events") or [] if e.get("action") == "failed"
            ]
            if failed:
                errors.append(f"{pod}: rank {key} dump writes failed: {failed[:1]}")
    for node in nodes:
        if node not in served:
            errors.append(f"{node}: no responder ledger for its Pod")
    return errors


def request_errors(
    ledgers: Mapping[str, Any],
    nodes: Sequence[str],
    *,
    sampled_nodes: Sequence[str],
    minimum_requests: int = 2,
) -> list[str]:
    """Why the product did not ask the replaced recorder as it should have.

    Every rank of a node the triage sampled must have received at least
    ``minimum_requests`` dump requests (two per PID: the first dump and the
    second of the series). A node the product never sampled owes none. These
    are product faults: the FIFO was there, the product did not ask.
    """

    errors: list[str] = []
    for ledger in ledgers.values():
        if not isinstance(ledger, dict):
            continue
        node = ledger_node(ledger, nodes)
        if node is None or node not in sampled_nodes:
            continue
        for key, record in sorted((ledger.get("ranks") or {}).items()):
            requests = int(record.get("requests") or 0)
            if requests < minimum_requests:
                errors.append(
                    f"{node}: rank {key} received {requests} dump request(s), the "
                    f"product's triage should have asked {minimum_requests}"
                )
    return errors


def answered_watermarks(
    ledgers: dict[str, Any],
) -> dict[int, list[dict[str, Any]]]:
    """Per rank, the watermarks of every dump the responders wrote, in order."""

    result: dict[int, list[dict[str, Any]]] = {}
    for ledger in ledgers.values():
        if not isinstance(ledger, dict):
            continue
        for key, record in (ledger.get("ranks") or {}).items():
            result[int(key)] = [
                event["watermarks"]
                for event in record.get("events") or []
                if event.get("action") == "answered"
                and isinstance(event.get("watermarks"), dict)
            ]
    return result
