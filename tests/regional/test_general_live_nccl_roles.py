"""GF-GENERAL-LIVE-011 role, thread, process and attempt-change variants.

The triage rows are built the way the Node Agent's ``collect_process_series``
emits them -- kernel thread samples with ``NSpid`` namespace ids, windows from
the product's own ``process_window``, py-spy summaries from its
``stack_summary``, the comparison context of a declaring rank -- and the
``hung_triage_decision`` is what the product's ``classify_hung_signals`` makes
of exactly those rows, so the graders are checked against the real
classification (cohorts, peer comparisons, observations), not a transcript
of it. The records then take the stored shape (each node's triage result in
the summary codec's envelope, the decision on the bundle step) and the
variants run through the fake NCCL job and EFA probes of
``test_general_live_nccl``; the trainer's account is the ``ROLES``/``SHARER``/
``HOLDER`` lines the declarations imply.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.execution.hung_classification import classify_hung_signals
from gpu_fault.hung_summary_codec import encode_summary
from gpu_fault.node_agent.operations.process_samples import process_window
from gpu_fault.node_agent.operations.stack_samples import stack_summary
from scripts.e2e.regional import general_live_nccl as nc
from scripts.e2e.regional import general_live_nccl_role_trainer as plan
from scripts.e2e.regional import general_live_nccl_roles as roles
from scripts.e2e.regional import general_live_nccl_triage_view as view
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from tests.regional._general_live_fakes import FakeRegional, make_context
from tests.regional.test_general_live_nccl import (
    NODE,
    PEER,
    FakeNcclJob,
    Traffic,
    _settings,
    efa_probe,
    hung_records,
)

SPECS = {spec.variant: spec for spec in roles.VARIANTS}
NODES = (NODE, PEER)
JOB = "gl11-role-a1-deadbeef"
ATTEMPT = f"{JOB}-a001"
T0 = datetime(2026, 10, 24, 0, 20, tzinfo=timezone.utc)
CLK = 100
PYTHON = "/opt/conda/bin/python3.11"
TORCH = "/opt/conda/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py"
TRAINER = "/tmp/gfrs_nccl_stall.py"
#: Kernel state and wait channel of a helper, by what it does.
BEHAVIOUR = {
    "wait": ("S", "futex_wait_queue"),
    "spin": ("R", None),
    "sleep": ("S", "futex_wait_queue"),
}


def helpers_of(mode: str, rank: int) -> list[tuple[str, str, str]]:
    """``(name, role, behaviour)`` of a rank's declared helper threads."""

    items = [
        ("gfrs-comm", "COMMUNICATION", "wait"),
        ("gfrs-keeper", "BACKGROUND", "wait"),
    ]
    if mode == plan.ROLES and rank % plan.PROCS_PER_NODE in plan.burning_locals(mode):
        items += [
            (f"gfrs-burn-{i}", "BACKGROUND", "spin") for i in range(plan.BURNER_THREADS)
        ]
    if mode == plan.THREADS:
        items += [
            ("gfrs-bg-a", "BACKGROUND", "wait"),
            ("gfrs-bg-b", "BACKGROUND", "wait"),
        ]
    if mode == plan.WCHAN:
        items += [
            ("gfrs-spinner", "COMMUNICATION", "spin"),
            ("gfrs-sleeper", "BACKGROUND", "sleep"),
        ]
    if mode == plan.CPU:
        items.append(("gfrs-cpu-duty", "BACKGROUND", "spin"))
    return items


def cores_of(mode: str, rank: int) -> float:
    local = rank % plan.PROCS_PER_NODE
    if mode == plan.ROLES:
        return 3.4 if local in plan.burning_locals(mode) else 0.4
    return {plan.WCHAN: 1.3, plan.CPU: 0.9}.get(mode, 0.4)


def declaration(mode: str, rank: int) -> view.Declaration:
    """What the trainer of ``rank`` prints in ``mode`` (container-native ids)."""

    context = plan.expected_context(mode, rank, world_size=roles.WORLD_SIZE)
    container_pid = 100 + rank % plan.PROCS_PER_NODE
    thread_roles: dict[str, tuple[str, str]] = {}
    if context is not None:
        thread_roles[str(container_pid)] = ("TRAINING", "MainThread")
        for index, (name, role, _kind) in enumerate(helpers_of(mode, rank)):
            thread_roles[str(200 + 10 * (rank % plan.PROCS_PER_NODE) + index)] = (
                role,
                name,
            )
    group = plan.group_of(mode, rank % plan.PROCS_PER_NODE) or {}
    return view.Declaration(
        rank=rank,
        mode=mode,
        pid=container_pid,
        group=context["group_name"] if context else None,
        stage=context["pipeline_stage"] if context else None,
        expert=context["expert_role"] if context else None,
        members=tuple(context["members"]) if context else (),
        roles=thread_roles,
        clk_tck=CLK,
        burn=bool(group.get("burn")),
        share_gpu=rank % plan.PROCS_PER_NODE in plan.sharing_locals(mode),
        holder=rank % plan.PROCS_PER_NODE in plan.holder_locals(mode),
    )


def roles_line(decl: view.Declaration) -> str:
    roles_token = (
        ",".join(f"{tid}:{role}:{name}" for tid, (role, name) in decl.roles.items())
        if decl.declared
        else "-"
    )
    return (
        f"[rank{decl.rank}]: ROLES mode={decl.mode} node_index={decl.rank // 8} "
        f"group={decl.group or '-'} stage={decl.stage or '-'} expert={decl.expert or '-'} "
        f"members={','.join(map(str, decl.members)) if decl.members else '-'} "
        f"roles={roles_token} clk_tck={decl.clk_tck} pid={decl.pid} "
        f"burn={int(decl.burn)} share_gpu={int(decl.share_gpu)} holder={int(decl.holder)} "
        f"rank={decl.rank} node=node-{decl.rank // 8} elapsed=5.0"
    )


def account_of(
    mode: str, *, drop_ranks: tuple[int, ...] = (), errors: tuple[str, ...] = ()
) -> dict[str, list[str]]:
    """The two Pods' marker lines for ``mode``."""

    account: dict[str, list[str]] = {"master": [], "worker": []}
    for rank in range(roles.WORLD_SIZE):
        pod = "master" if rank < plan.PROCS_PER_NODE else "worker"
        decl = declaration(mode, rank)
        if rank not in drop_ranks:
            account[pod].append(roles_line(decl))
        if decl.share_gpu:
            account[pod].append(
                f"SHARER pid={sharer_pid(rank)} device={rank % 8} rank={rank} node=n elapsed=5.0"
            )
        if decl.holder:
            account[pod].append(
                f"HOLDER pid={holder_pid(rank)} device={rank % 8} exe=/tmp/{plan.HOLDER_EXE}-{rank} "
                f"rank={rank} node=n elapsed=9.0"
            )
    account["master"].extend(errors)
    return account


def sharer_pid(rank: int) -> int:
    return 150 + rank % plan.PROCS_PER_NODE


def holder_pid(rank: int) -> int:
    return 170 + rank % plan.PROCS_PER_NODE


def host_pid(rank: int) -> int:
    return 30_000 + rank


def gpu_uuid(rank: int) -> str:
    return f"GPU-{rank % plan.PROCS_PER_NODE:08d}-0000-0000-0000-000000000000"


# --------------------------------------------------------------------------- #
# triage rows as the Node Agent emits them
# --------------------------------------------------------------------------- #
def pyspy_payload(
    threads: dict[str, tuple[str, str | None, str, str]],
    pid: int,
    rank: int,
    mode: str,
    index: int,
    unstable: bool,
) -> list[dict[str, Any]]:
    """What py-spy's JSON lists for sample ``index`` of a rank: every thread
    with its native id, the blocked rank 0 inside the collective, a held
    rank's main thread moving between samples, one churn thread per sample in
    the threads mode."""

    pyspy = []
    for host, (native, _role, name, kind) in threads.items():
        if name == "gfrs-watch":
            frames = [{"name": "watch", "filename": TRAINER, "line": 160}]
        elif kind == "main" and rank == 0:
            frames = [
                {"name": "all_reduce", "filename": TORCH, "line": 2000},
                {"name": "stall", "filename": TRAINER, "line": 280},
            ]
        elif kind == "main":
            frames = [
                {
                    "name": "gpu_busy_until",
                    "filename": TRAINER,
                    "line": 170 + (index if unstable else 0),
                }
            ]
        else:
            frames = [{"name": kind + "_loop", "filename": TRAINER, "line": 400}]
        pyspy.append(
            {
                "pid": pid,
                "thread_id": 139_000_000 + int(host),
                "thread_name": name,
                "os_thread_id": int(native),
                "active": kind == "spin",
                "owns_gil": False,
                "frames": frames,
            }
        )
    if mode == plan.THREADS:
        pyspy.append(
            {
                "pid": pid,
                "thread_id": 139_900_000 + index,
                "thread_name": "gfrs-churn-x",
                "os_thread_id": 300 + index,
                "active": False,
                "owns_gil": False,
                "frames": [{"name": "wait", "filename": TRAINER, "line": 518}],
            }
        )
    return pyspy


def rank_row(
    mode: str,
    rank: int,
    *,
    decl: view.Declaration | None = None,
    flat_namespace: bool = False,
    spinner_wchan_kept: bool = False,
    unstable_main: bool | None = None,
) -> dict[str, Any]:
    """One rank's triage row: kernel samples, windows, stacks, context, GPU."""

    decl = decl or declaration(mode, rank)
    node = NODE if rank < plan.PROCS_PER_NODE else PEER
    pid, start = host_pid(rank), 500_000 + rank
    cores = cores_of(mode, rank)
    unstable = (rank != 0) if unstable_main is None else unstable_main
    helpers = helpers_of(mode, rank)
    # host tid -> (native tid, role, name, behaviour); the main thread is the pid.
    threads: dict[str, tuple[str, str | None, str, str]] = {
        str(pid): (
            str(decl.pid),
            "TRAINING" if decl.declared else None,
            "MainThread",
            "main",
        )
    }
    natives = (
        [tid for tid in decl.roles if tid != str(decl.pid)] if decl.declared else []
    )
    for index, (name, role, kind) in enumerate(helpers):
        native = natives[index] if decl.declared else str(200 + 10 * (rank % 8) + index)
        threads[str(pid + 1 + index)] = (
            native,
            role if decl.declared else None,
            name,
            kind,
        )
    threads[str(pid + 50)] = (
        "199",
        None,
        "gfrs-watch",
        "wait",
    )  # the GPU load's SIGTERM watcher

    samples = []
    stacks = []
    for index in range(3):
        kernel: dict[str, dict[str, Any]] = {}
        for host, (native, _role, _name, kind) in threads.items():
            if kind == "main":
                state, wchan = (
                    ("R", None) if unstable and index % 2 else ("S", "futex_wait_queue")
                )
            else:
                state, wchan = BEHAVIOUR[kind]
            if kind == "spin" and spinner_wchan_kept:
                wchan = "0"
            kernel[host] = {
                "start_ticks": start + int(host) - pid,
                "state": state,
                "wchan": wchan,
                "namespace_ids": [int(host)]
                if flat_namespace
                else [int(host), int(native)],
            }
        if mode == plan.THREADS:  # a churn thread lives for one sample only
            kernel[str(pid + 60 + index)] = {
                "start_ticks": start + 1000 + index,
                "state": "S",
                "wchan": "futex_wait_queue",
                "namespace_ids": [pid + 60 + index]
                if flat_namespace
                else [pid + 60 + index, 300 + index],
            }
        samples.append(
            {
                "state": kernel[str(pid)]["state"],
                "cpu_ticks": 7_000 + int(round(cores * CLK)) * index,
                "start_ticks": start,
                "read_bytes": 0,
                "write_bytes": 4096 * index if mode == plan.ROLES else 0,
                "wchar": 0,
                "voluntary_ctxt_switches": 50 + index,
                "wchan": kernel[str(pid)]["wchan"],
                "threads": kernel,
                "scanned_thread_count": len(kernel),
                "thread_scan_complete": True,
                "sample_monotonic": 1_000.0 + index,
                "workload_scope_verified": True,
                "observed_at": (T0 + timedelta(seconds=index)).isoformat(),
            }
        )
        pyspy = pyspy_payload(threads, pid, rank, mode, index, unstable)
        role_map = (
            {tid: role for tid, (role, _n) in decl.roles.items()}
            if decl.declared
            else {}
        )
        summary = stack_summary(json.dumps(pyspy), pid, role_map, kernel)
        stacks.append(
            {
                **summary,
                "observed_at": (
                    T0 + timedelta(seconds=index, milliseconds=100)
                ).isoformat(),
                "finished_at": (
                    T0 + timedelta(seconds=index, milliseconds=400)
                ).isoformat(),
                "returncode": 0,
            }
        )
    windows = [process_window(a, b, CLK) for a, b in zip(samples, samples[1:])]
    context: dict[str, Any] = {"verified": False, "host_thread_roles": {}}
    expected = plan.expected_context(mode, rank, world_size=roles.WORLD_SIZE)
    if decl.declared and expected is not None:
        members = expected["members"]
        context = {
            **{key: value for key, value in expected.items() if key != "members"},
            "thread_roles": {tid: role for tid, (role, _n) in decl.roles.items()},
            "membership_sha256": hashlib.sha256(
                json.dumps(sorted(members), separators=(",", ":")).encode()
            ).hexdigest(),
            "member_count": len(members),
            "membership_verified": True,
            "job_id": JOB,
            "attempt_id": ATTEMPT,
            "declared_rank": rank,
            "source": "PROCESS_BOUND_SNAPSHOT",
            "verified": True,
            "host_thread_roles": {
                host: role for host, (_n, role, _name, _k) in threads.items() if role
            },
        }
    uuid = gpu_uuid(rank)
    return {
        "node_id": node,
        "pid": pid,
        "rank": rank,
        "gpu_uuid": uuid,
        "gpu_uuids": [uuid],
        "local_rank": rank % plan.PROCS_PER_NODE,
        "process_name": PYTHON,
        "process_identity": {"verified": True, "start_time_ticks": start},
        "comparison_context": context,
        "process_series": {
            "version": "process-series/v1",
            "requested_samples": 3,
            "samples": samples,
            "windows": windows,
        },
        "stack_series": {
            "version": "stack-series/v1",
            "requested_samples": 3,
            "samples": stacks,
        },
        "gpu_series": gpu_series(uuid, shared=decl.share_gpu or decl.holder),
        "python_stack": stacks[-1],
        "proc": windows[-1],
        "gpu": {"utilization_gpu_percent": 30.0},
        "flight_recorder": {"status": "not_configured"},
    }


def gpu_series(uuid: str, *, shared: bool) -> dict[str, Any]:
    return {
        "scope": roles.DEVICE_NOT_PID,
        "mapping_verified": True,
        "shared_with_other_processes": shared,
        "samples": [
            {
                "observed_at": (
                    T0 + timedelta(seconds=index, milliseconds=500)
                ).isoformat(),
                "devices": {
                    uuid: {
                        "utilization_gpu_percent": 30.0,
                        "utilization_memory_percent": 12.0,
                        "clocks_throttle_reasons_active": "0x0",
                        "device_model": "NVIDIA H200",
                    }
                },
            }
            for index in range(3)
        ],
    }


def extra_process_row(
    rank: int, *, holder: bool, stack_collected: bool = False
) -> dict[str, Any]:
    """The unranked GPU sharer (Python) or the non-Python holder of ``rank``."""

    node = NODE if rank < plan.PROCS_PER_NODE else PEER
    container = holder_pid(rank) if holder else sharer_pid(rank)
    pid, start = 31_000 + rank, 600_000 + rank
    samples = [
        {
            "state": "S",
            "cpu_ticks": 100 + index,
            "start_ticks": start,
            "read_bytes": 0,
            "write_bytes": 0,
            "wchar": 0,
            "voluntary_ctxt_switches": 10 + index,
            "wchan": "hrtimer_nanosleep",
            "threads": {
                str(pid): {
                    "start_ticks": start,
                    "state": "S",
                    "wchan": "hrtimer_nanosleep",
                    "namespace_ids": [pid, container],
                }
            },
            "scanned_thread_count": 1,
            "thread_scan_complete": True,
            "sample_monotonic": 1_000.0 + index,
            "workload_scope_verified": True,
            "observed_at": (T0 + timedelta(seconds=index)).isoformat(),
        }
        for index in range(3)
    ]
    if holder and not stack_collected:
        stacks = [
            {
                "status": "FAILED",
                "error": "py-spy exited with status 1",
                "returncode": 1,
                "observed_at": (
                    T0 + timedelta(seconds=index, milliseconds=100)
                ).isoformat(),
            }
            for index in range(3)
        ]
    else:
        pyspy = [
            {
                "pid": pid,
                "thread_id": 139_000_000 + pid,
                "thread_name": "MainThread",
                "os_thread_id": container,
                "active": False,
                "owns_gil": False,
                "frames": [{"name": "<module>", "filename": "<string>", "line": 9}],
            }
        ]
        stacks = [
            {
                **stack_summary(json.dumps(pyspy), pid, {}, samples[index]["threads"]),
                "observed_at": (
                    T0 + timedelta(seconds=index, milliseconds=100)
                ).isoformat(),
                "finished_at": (
                    T0 + timedelta(seconds=index, milliseconds=300)
                ).isoformat(),
                "returncode": 0,
            }
            for index in range(3)
        ]
    windows = [process_window(a, b, CLK) for a, b in zip(samples, samples[1:])]
    uuid = gpu_uuid(rank)
    return {
        "node_id": node,
        "pid": pid,
        "rank": None,
        "gpu_uuid": uuid,
        "gpu_uuids": [uuid],
        "local_rank": None,
        "process_name": f"/tmp/{plan.HOLDER_EXE}-{rank}" if holder else PYTHON,
        "process_identity": {"verified": True, "start_time_ticks": start},
        "comparison_context": {"verified": False, "host_thread_roles": {}},
        "process_series": {
            "version": "process-series/v1",
            "requested_samples": 3,
            "samples": samples,
            "windows": windows,
        },
        "stack_series": {
            "version": "stack-series/v1",
            "requested_samples": 3,
            "samples": stacks,
        },
        "gpu_series": gpu_series(uuid, shared=True),
        "python_stack": stacks[-1],
        "proc": windows[-1],
        "gpu": {},
        "flight_recorder": {"status": "not_configured"},
    }


def scenario(mode: str, **tweaks: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Both nodes' rows in ``mode`` and the product's own decision over them."""

    rows = [rank_row(mode, rank, **tweaks) for rank in range(roles.WORLD_SIZE)]
    for rank in range(roles.WORLD_SIZE):
        decl = declaration(mode, rank)
        if decl.share_gpu:
            rows.append(extra_process_row(rank, holder=False))
        if decl.holder:
            rows.append(extra_process_row(rank, holder=True))
    decision = classify_hung_signals(rows, undetermined_nodes=[])
    return rows, decision


def records_of(
    rows: list[dict[str, Any]],
    decision: dict[str, Any] | None,
    since: str,
    **options: Any,
) -> dict[str, Any]:
    """The stored records: each node's rows in its wave, the decision bound."""

    records = hung_records(since, **options)
    workflow = records["workflows"]["wf"]
    waves = [
        item
        for item in workflow["step_executions"]
        if item["operation"] == "COLLECT_HUNG_TRIAGE"
    ]
    for execution, node in zip(waves, NODES):
        document = {
            "node_id": node,
            "ranks": [row for row in rows if row["node_id"] == node],
        }
        execution["details"] = {"node_results": {node: encode_summary(document)}}
    if decision is not None:
        waves[-1]["details"]["hung_triage_decision"] = {"placeholder": True}
        for step in workflow["official_steps"]:
            if step["operation"] == "COLLECT_DIAGNOSTIC_BUNDLE":
                step["parameters"] = {
                    "hung_triage_decision": decision,
                    "bundle_scope": "bounded_triage_nodes",
                }
    return records


# --------------------------------------------------------------------------- #
# the fake job and context
# --------------------------------------------------------------------------- #
class FakeSecondAttempt(FakeNcclJob):
    def __init__(self, ctx: Any, *, name: str, **options: Any) -> None:
        super().__init__(ctx, name=name, **options)
        self.attempt_id = f"{name}-a002"

    def start(self) -> dict[str, Any]:
        self.started_at = self.ctx.clock()
        return super().start()


@pytest.fixture(autouse=True)
def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeNcclJob.instances = []
    FakeNcclJob.traffic = Traffic()
    FakeNcclJob.alive_value = True
    FakeNcclJob.events = None
    monkeypatch.setattr(roles, "ManagedNcclJob", FakeNcclJob)
    monkeypatch.setattr(roles, "SecondAttemptNcclJob", FakeSecondAttempt)
    monkeypatch.setattr(roles, "read_account", lambda job: dict(ACCOUNT["value"]))


ACCOUNT: dict[str, Any] = {"value": {}}


def _ctx(variant: str, tmp_path: Path, respond: Any):
    spec = SPECS[variant]
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=FakeRegional(cpu_python=respond),
        probes={},
        settings=_settings(spec, tmp_path),
    )
    start = datetime(2026, 10, 24, tzinfo=timezone.utc)
    ticks = itertools.count()
    ctx.clock = lambda: start + timedelta(seconds=15 * next(ticks))
    ctx.probes[f"node@{NODE}"] = efa_probe(NODE, survey=True)  # type: ignore[assignment]
    ctx.probes[f"node@{PEER}"] = efa_probe(PEER)  # type: ignore[assignment]
    return ctx


def run_variant(
    variant: str,
    mode: str,
    tmp_path: Path,
    *,
    rows=None,
    decision=None,
    account=None,
    none: bool = False,
    **options: Any,
):
    if rows is None:
        rows, decision = scenario(mode)
    ACCOUNT["value"] = account if account is not None else account_of(mode)

    def respond(script: str, *arguments: str) -> dict[str, Any]:
        assert script == nc.HUNG_PROBE
        if none:
            return {"incidents": [], "workflows": {}}
        # The rows' snapshots name the live job, whose identity the runner mints.
        job = FakeNcclJob.instances[-1]
        stamped = copy.deepcopy(rows)
        for row in stamped:
            context = row.get("comparison_context") or {}
            if context.get("job_id"):
                context["job_id"], context["attempt_id"] = job.name, job.attempt_id
        return records_of(stamped, decision, arguments[-1], **options)

    ctx = _ctx(variant, tmp_path, respond)
    return SPECS[variant].execute(ctx), ctx


# --------------------------------------------------------------------------- #
# registry and the view
# --------------------------------------------------------------------------- #
def test_the_variants_are_native_on_both_lanes_with_the_stall_cleanup() -> None:
    assert [spec.variant for spec in roles.VARIANTS] == [
        "role_mismatch",
        "tid_namespace",
        "wchan_zero",
        "cpu_interval",
        "pyspy_nonzero",
        "attempt_change",
    ]
    for spec in roles.VARIANTS:
        assert (
            spec.proof_mode == "native"
            and spec.lanes == ("eks", "hp")
            and spec.implemented
        )
        assert spec.approvals == () and spec.flags == "S"
    assert SPECS["attempt_change"].max_minutes > SPECS["role_mismatch"].max_minutes
    assert roles.A_DELETE_BY_SECONDS < nc.ZERO_HUNG_SECONDS
    assert roles.A_ZERO_SECONDS > nc.ZERO_WARNING_SECONDS + 300, (
        "the warning is delivered within a summary"
    )


def test_the_account_reads_declarations_children_and_fixture_errors() -> None:
    account = account_of(
        plan.CPU,
        errors=("GFRS-FIXTURE-ERROR what=sharer error=boom rank=6 node=n elapsed=1",),
    )
    decls = view.declarations(account)
    assert set(decls) == set(range(roles.WORLD_SIZE))
    assert (
        decls[6].share_gpu
        and decls[6].tids("BACKGROUND", "gfrs-cpu")
        and decls[6].training_tid == "106"
    )
    assert view.children(account, plan.SHARER_MARKER) == {
        rank: {
            "pid": str(sharer_pid(rank)),
            "device": str(rank % 8),
            "rank": str(rank),
            "node": "n",
            "elapsed": "5.0",
        }
        for rank in (6, 7, 14, 15)
    }
    assert view.fixture_errors(account) == ["rank 6: sharer boom"]
    undeclared = view.declarations(account_of(plan.ROLES))[7]
    assert (
        not undeclared.declared
        and undeclared.roles == {}
        and undeclared.training_tid is None
    )
    with pytest.raises(ValueError, match="malformed ROLES"):
        view.read_declaration({"mode": "roles", "rank": "x"})


def test_the_triage_view_decodes_envelopes_and_finds_the_bound_decision() -> None:
    rows, decision = scenario(plan.ROLES)
    records = records_of(rows, decision, T0.isoformat())
    read, unreadable = view.triage_rows(records)
    assert unreadable == [] and len(read) == len(rows)
    assert {row["node_id"] for row in read} == set(NODES)
    assert view.triage_decision(records) == decision, "the bundle step's copy wins"
    for step in records["workflows"]["wf"]["official_steps"]:
        step["parameters"] = {}
    assert view.triage_decision(records) == {"placeholder": True}, (
        "else the last triage execution's"
    )
    broken = records_of(rows, decision, T0.isoformat())
    waves = [
        i
        for i in broken["workflows"]["wf"]["step_executions"]
        if i["operation"] == "COLLECT_HUNG_TRIAGE"
    ]
    waves[0]["details"]["node_results"][NODE] = {
        "encoding": "zlib+json",
        "payload": "!!",
    }
    read, unreadable = view.triage_rows(broken)
    assert len(unreadable) == 1 and "unreadable" in unreadable[0]
    assert {row["node_id"] for row in read} == {PEER}
    assert view.namespaced(rows[0]) and not view.namespaced(
        rank_row(plan.ROLES, 0, flat_namespace=True)
    )
    assert view.host_tids(rows[3], "103") == [str(host_pid(3))]
    assert view.node_ranks(NODES) == {NODE: list(range(8)), PEER: list(range(8, 16))}


# --------------------------------------------------------------------------- #
# graders over the product's own classification
# --------------------------------------------------------------------------- #
def test_the_product_forms_role_cohorts_the_grader_accepts() -> None:
    rows, decision = scenario(plan.ROLES)
    decls = view.declarations(account_of(plan.ROLES))
    reports, groups = view.process_reports(decision), view.cohorts(decision)
    assert roles.coverage_errors(rows, NODES) == []
    assert (
        roles.context_errors(
            rows, decls, mode=plan.ROLES, job_id=JOB, attempt_id=ATTEMPT
        )
        == []
    )
    assert roles.cohort_errors(rows, decls, reports, groups) == []
    compared = {(c["definition"]["pipeline_stage"], c["process_count"]) for c in groups}
    assert compared == {("pp0", 4), ("pp1", 3)}, "per node: pp0 x4 and pp1 x3 compared"
    assert len(groups) == 4
    comparisons = {
        row["rank"]: view.report_for(reports, row)["peer_comparison"] for row in rows
    }
    assert comparisons[7] == roles.NO_COHORT and comparisons[15] == roles.NO_COHORT
    assert all(
        comparisons[rank] == roles.ROLE_COHORT for rank in range(16) if rank % 8 != 7
    ), "every declared rank is compared inside its own cohort"
    assert "4 cohort(s)" in roles.cohort_notes(reports, groups)[0]


def test_cohort_errors_catch_mixed_cohorts_undeclared_members_and_cross_role_leads() -> (
    None
):
    rows, decision = scenario(plan.ROLES)
    decls = view.declarations(account_of(plan.ROLES))
    reports, groups = view.process_reports(decision), view.cohorts(decision)
    pp1 = next(
        c
        for c in groups
        if c["definition"]["pipeline_stage"] == "pp1" and c["processes"][0][0] == NODE
    )
    undeclared = next(r for r in reports if r["rank"] == 7)
    pp1["processes"].append(
        [undeclared["node_id"], undeclared["pid"], undeclared["start_time_ticks"]]
    )
    errors = roles.cohort_errors(rows, decls, reports, groups)
    assert any("holds an undeclared process" in e for e in errors), errors
    pp0 = next(
        c
        for c in groups
        if c["definition"]["pipeline_stage"] == "pp0" and c["processes"][0][0] == NODE
    )
    quiet = next(r for r in reports if r["rank"] == 5)
    quiet["observations"].append(
        {
            "code": "ROLE_COHORT_CPU_DIFFERENCE",
            "candidate": True,
            "cohort": pp0["cohort"],
        }
    )
    errors = roles.cohort_errors(rows, decls, reports, groups)
    assert any("cites a cohort the process is not in" in e for e in errors), errors
    assert any("attributed across roles" in e for e in errors), errors
    # A lead inside the process's own cohort is legitimate.
    busy = next(r for r in reports if r["rank"] == 1)
    busy["observations"].append(
        {
            "code": "ROLE_COHORT_CPU_DIFFERENCE",
            "candidate": True,
            "cohort": pp0["cohort"],
        }
    )
    assert not [
        e for e in roles.cohort_errors(rows, decls, reports, groups) if "rank 1 " in e
    ]


def test_context_errors_read_the_declaration_back_field_by_field() -> None:
    rows, _decision = scenario(plan.ROLES)
    decls = view.declarations(account_of(plan.ROLES))
    rows[2]["comparison_context"]["pipeline_stage"] = "pp9"
    rows[3]["comparison_context"]["verified"] = False
    rows[7]["comparison_context"] = {"verified": True, "pipeline_stage": "pp0"}
    errors = roles.context_errors(
        rows, decls, mode=plan.ROLES, job_id=JOB, attempt_id=ATTEMPT
    )
    assert len(errors) == 3
    assert "pipeline_stage': 'pp9'" in errors[0] and "not verified" in errors[1]
    assert "undeclared rank read back a context" in errors[2]
    assert roles.coverage_errors(rows[:7] + rows[8:], NODES) == [
        f"{NODE}: ranks [7] were not sampled"
    ]
    assert roles.coverage_errors([], NODES) == [
        "COLLECT_HUNG_TRIAGE sampled no process on either node"
    ]
    assert roles.premise_errors(rows, {}, account_of(plan.ROLES)) == [
        f"no ROLES line for sampled ranks {list(range(16))}"
    ]


def test_thread_identity_graders_over_the_namespace_mapping() -> None:
    rows, decision = scenario(plan.THREADS)
    decls = view.declarations(account_of(plan.THREADS))
    reports = view.process_reports(decision)
    assert roles.host_thread_errors(rows, decls) == []
    assert roles.stack_identity_errors(rows, decls) == []
    assert roles.stability_errors(rows, decls, reports) == []
    blocked = view.report_for(reports, rows[0])
    assert any(t["role"] == "TRAINING" for t in blocked["stack"]["stable_threads"]), (
        "the blocked rank's training thread is a stable thread"
    )
    assert all(
        t["role"] != "UNKNOWN" or t["name"] == "gfrs-watch"
        for r in reports
        for t in r["stack"]["stable_threads"]
    ), "churn threads never stabilize; only the undeclared watcher does"
    # Wrong mapping, OS-sourced role, no stable training thread on the blocked rank.
    rows[1]["comparison_context"]["host_thread_roles"][str(host_pid(1))] = "BACKGROUND"
    rows[2]["stack_series"]["samples"][0]["threads"][0]["role_source"] = (
        "OS_MAIN_OR_UNKNOWN"
    )
    errors = roles.host_thread_errors(rows, decls) + roles.stack_identity_errors(
        rows, decls
    )
    assert any("reads BACKGROUND, declared TRAINING" in e for e in errors), errors
    assert any("got role TRAINING without a declaration" in e for e in errors), errors
    reports[0]["stack"]["stable_threads"] = []
    assert roles.stability_errors(rows, decls, reports) == [
        f"{NODE} rank 0 pid {host_pid(0)}: the blocked rank's training thread is not a stable thread"
    ]


def test_wchan_grader_over_a_spinner_and_a_sleeper() -> None:
    rows, decision = scenario(plan.WCHAN)
    decls = view.declarations(account_of(plan.WCHAN))
    reports = view.process_reports(decision)
    assert roles.wchan_errors(rows, decls, reports) == []
    kept, kept_decision = scenario(plan.WCHAN, spinner_wchan_kept=True)
    errors = roles.wchan_errors(kept, decls, view.process_reports(kept_decision))
    assert errors and all("wchan '0' for the spinner" in e for e in errors), errors
    rows[4]["process_series"]["windows"][0]["cpu_cores"] = 0.1
    errors = roles.wchan_errors(rows, decls, reports)
    assert any("0.10 cores with a spinning thread" in e for e in errors), (
        "a window under the spinner floor is named"
    )
    assert any("cpu_cores 0.1 !=" in e for e in errors), (
        "the normalization is recomputed"
    )


def test_cpu_grader_over_intervals_and_a_shared_gpu() -> None:
    rows, decision = scenario(plan.CPU)
    account = account_of(plan.CPU)
    decls = view.declarations(account)
    sharers = view.children(account, plan.SHARER_MARKER)
    reports = view.process_reports(decision)
    assert roles.cpu_errors(rows, decls, reports, sharers) == []
    comparisons = {
        row["rank"]: view.report_for(reports, row)["peer_comparison"]
        for row in rows
        if row["rank"] is not None
    }
    assert comparisons[6] == roles.NO_COHORT and comparisons[5] == roles.ROLE_COHORT
    rows[6]["gpu_series"]["shared_with_other_processes"] = False
    rows[1]["process_series"]["windows"][1]["elapsed_seconds"] = 3.0
    errors = roles.cpu_errors(rows, decls, reports, sharers)
    assert any("shared GPU reads shared=False" in e for e in errors), errors
    assert any("against samples 1.00 s apart" in e for e in errors), errors
    without = [row for row in rows if row["rank"] is not None]
    assert any(
        "0 sharer process(es) sampled, 2 started" in e
        for e in roles.cpu_errors(without, decls, reports, sharers)
    ), "an unsampled sharer is named"


def test_pyspy_grader_over_the_non_python_holder() -> None:
    rows, decision = scenario(plan.PYSPY)
    holders = view.children(account_of(plan.PYSPY), plan.HOLDER_MARKER)
    reports = view.process_reports(decision)
    assert set(holders) == {7, 15}
    assert roles.pyspy_errors(rows, reports, holders) == []
    holder = next(r for r in reports if r["pid"] == 31_007)
    assert (
        holder["stack"]["valid_samples"] == 0
        and holder["behavior"]["valid_windows"] == 2
    )
    collected = [
        row
        if row["pid"] != 31_007
        else extra_process_row(7, holder=True, stack_collected=True)
        for row in rows
    ]
    errors = roles.pyspy_errors(
        collected,
        view.process_reports(classify_hung_signals(collected, undetermined_nodes=[])),
        holders,
    )
    assert any("holder stack 0 is" in e for e in errors), errors
    dropped = [row for row in rows if row["pid"] != 31_007]
    assert any(
        "0 holder process(es) sampled, 1 started" in e
        for e in roles.pyspy_errors(dropped, reports, holders)
    ), "an unsampled holder is named"
    rows[0]["stack_series"]["samples"] = [
        {"status": "FAILED", "error": "x", "returncode": 1}
    ] * 3
    assert any(
        "no real py-spy capture" in e
        for e in roles.pyspy_errors(rows, reports, holders)
    ), "a Python rank without a real capture is named"


# --------------------------------------------------------------------------- #
# the variants through the fakes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("variant", "mode", "stage", "suffix"),
    [
        ("role_mismatch", plan.ROLES, "cohorts", "role"),
        ("tid_namespace", plan.THREADS, "host_threads", "tid"),
        ("wchan_zero", plan.WCHAN, "wchan", "wchan"),
        ("cpu_interval", plan.CPU, "cpu", "cpu"),
        ("pyspy_nonzero", plan.PYSPY, "pyspy", "pyspy"),
    ],
)
def test_each_role_variant_passes_on_a_faithful_triage(
    tmp_path: Path, variant: str, mode: str, stage: str, suffix: str
) -> None:
    result, ctx = run_variant(variant, mode, tmp_path)
    assert result["verdict"] == "PASS", result["errors"]
    assert (
        stage in result["stages"]
        and "triage" in result["stages"]
        and "resumed" in result["stages"]
    )
    job = FakeNcclJob.instances[-1]
    assert job.stall_mode == mode and job.stalled and job.released == 1
    assert result["attempt"]["stall_mode"] == mode
    assert (ctx.evidence.case_dir / "triage-view.json").is_file(), (
        "the triage view is evidence"
    )
    assert (ctx.evidence.case_dir / f"{suffix}-account.json").is_file(), (
        "the trainer's account is evidence"
    )
    assert ctx.cleanup.pending() == [f"job-{job.attempt_id}"], "released, job kept"


def test_a_valid_stall_whose_triage_is_wrong_fails(tmp_path: Path) -> None:
    rows, decision = scenario(plan.ROLES)
    undeclared = next(r for r in view.process_reports(decision) if r["rank"] == 7)
    cohort = view.cohorts(decision)[0]
    cohort["processes"].append(
        [undeclared["node_id"], undeclared["pid"], undeclared["start_time_ticks"]]
    )
    result, _ctx = run_variant(
        "role_mismatch", plan.ROLES, tmp_path, rows=rows, decision=decision
    )
    assert result["verdict"] == "FAIL"
    assert any("undeclared process" in e for e in result["errors"]), result["errors"]
    assert FakeNcclJob.instances[-1].released == 1, "the job is released all the same"


def test_a_missing_decision_or_an_unsampled_rank_is_a_triage_fail(
    tmp_path: Path,
) -> None:
    rows, _decision = scenario(plan.WCHAN)
    result, _ctx = run_variant(
        "wchan_zero", plan.WCHAN, tmp_path, rows=rows[1:], decision=None
    )
    assert result["verdict"] == "FAIL"
    assert f"triage: {NODE}: ranks [0] were not sampled" in result["errors"]
    assert any("no hung_triage_decision" in e for e in result["errors"]), (
        "a missing decision is named"
    )


def test_no_hung_incident_at_all_fails_the_chain_not_the_fixture(
    tmp_path: Path,
) -> None:
    result, _ctx = run_variant("tid_namespace", plan.THREADS, tmp_path, none=True)
    assert result["verdict"] == "FAIL"
    assert (
        "chain: no EFA hung_suspected finding reached an incident" in result["errors"]
    )


def test_a_fixture_that_did_not_hold_is_inconclusive(tmp_path: Path) -> None:
    with pytest.raises(
        VariantInconclusive, match="no ROLES line for sampled ranks \\[3\\]"
    ):
        run_variant(
            "role_mismatch",
            plan.ROLES,
            tmp_path,
            account=account_of(plan.ROLES, drop_ranks=(3,)),
        )
    with pytest.raises(VariantInconclusive, match="holder no_C_compiler"):
        run_variant(
            "pyspy_nonzero",
            plan.PYSPY,
            tmp_path / "holder",
            account=account_of(
                plan.PYSPY,
                errors=(
                    "GFRS-FIXTURE-ERROR what=holder error=no_C_compiler rank=7 node=n elapsed=1",
                ),
            ),
        )
    account = {
        pod: [line for line in lines if plan.HOLDER_MARKER not in line]
        for pod, lines in account_of(plan.PYSPY).items()
    }
    with pytest.raises(
        VariantInconclusive,
        match="holders started on ranks \\[\\], planned \\[7, 15\\]",
    ):
        run_variant("pyspy_nonzero", plan.PYSPY, tmp_path / "noholder", account=account)
    account = {
        pod: [line for line in lines if plan.SHARER_MARKER not in line]
        for pod, lines in account_of(plan.CPU).items()
    }
    with pytest.raises(
        VariantInconclusive, match="sharer processes started on ranks \\[\\]"
    ):
        run_variant("cpu_interval", plan.CPU, tmp_path / "nosharer", account=account)
    rows, decision = scenario(plan.THREADS, flat_namespace=True)
    with pytest.raises(VariantInconclusive, match="shares the host pid namespace"):
        run_variant(
            "tid_namespace",
            plan.THREADS,
            tmp_path / "flat",
            rows=rows,
            decision=decision,
        )


def test_an_inconclusive_stall_never_reads_the_product(tmp_path: Path) -> None:
    FakeNcclJob.traffic.stalled = 500.0
    calls: list[str] = []

    def respond(script: str, *arguments: str) -> dict[str, Any]:
        calls.append(script)
        return {}

    ACCOUNT["value"] = account_of(plan.ROLES)
    ctx = _ctx("role_mismatch", tmp_path, respond)
    with pytest.raises(VariantInconclusive, match="kept moving"):
        SPECS["role_mismatch"].execute(ctx)
    assert calls == []


# --------------------------------------------------------------------------- #
# attempt_change
# --------------------------------------------------------------------------- #
class AttemptProduct:
    """The store over the attempt change: A's warnings, then B's fresh state."""

    def __init__(
        self,
        *,
        b_warning: bool = False,
        state_names_a: bool = False,
        hung_a: bool = False,
        no_warning: bool = False,
    ) -> None:
        self.b_warning, self.state_names_a, self.hung_a, self.no_warning = (
            b_warning,
            state_names_a,
            hung_a,
            no_warning,
        )
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, script: str, *arguments: str) -> dict[str, Any]:
        assert script == roles.ATTEMPT_PROBE
        self.calls.append(arguments)
        _cluster, _nodes, job, attempts, since = arguments
        stalled = FakeNcclJob.instances[0].stall_times[0]
        attempt_a = f"{job}-a001"
        incidents = []
        if not self.no_warning:
            incidents += [
                {
                    "incident_id": f"inc-w-{node}",
                    "event_id": f"host-{node}-77{nc.WARNING_SUFFIX}",
                    "state": "ACTION_PENDING",
                    "attempt_id": attempt_a,
                    "job_id": job,
                    "node_ids": [node],
                    "created_at": (stalled + timedelta(seconds=300)).isoformat(),
                }
                for node in NODES
            ]
        if self.hung_a:
            incidents.append(
                {
                    "incident_id": "inc-h",
                    "event_id": f"host-{NODE}-78{nc.HUNG_SUFFIX}",
                    "state": "ESCALATED",
                    "attempt_id": attempt_a,
                    "job_id": job,
                    "node_ids": list(NODES),
                    "created_at": (stalled + timedelta(seconds=700)).isoformat(),
                }
            )
        states: dict[str, Any] = {node: None for node in NODES}
        if "," in attempts:
            attempt_b = attempts.split(",")[1]
            if self.b_warning:
                incidents.append(
                    {
                        "incident_id": "inc-wb",
                        "event_id": f"host-{PEER}-79{nc.WARNING_SUFFIX}",
                        "state": "ACTION_PENDING",
                        "attempt_id": attempt_b,
                        "job_id": job,
                        "node_ids": [PEER],
                        "created_at": (stalled + timedelta(seconds=900)).isoformat(),
                    }
                )
            second = FakeNcclJob.instances[-1]
            for node in NODES:
                states[node] = {
                    "state_key": f"c/{node}",
                    "attempt_id": attempt_a if self.state_names_a else attempt_b,
                    "job_id": job,
                    "node_id": node,
                    "first_observed_at": (
                        second.started_at + timedelta(seconds=45)
                    ).isoformat(),
                    "observed_at": (
                        second.started_at + timedelta(seconds=300)
                    ).isoformat(),
                    "had_active_traffic": True,
                    "zero_since": None,
                    "consecutive_zero_samples": 0,
                    "signal": "NORMAL",
                }
        return {"incidents": incidents, "workflows": {}, "efa_states": states}


def run_attempt_change(tmp_path: Path, product: AttemptProduct) -> dict[str, Any]:
    ctx = _ctx("attempt_change", tmp_path, product)
    result = SPECS["attempt_change"].execute(ctx)
    return result


def test_attempt_change_warns_on_a_deletes_it_under_the_threshold_and_starts_b_fresh(
    tmp_path: Path,
) -> None:
    product = AttemptProduct()
    result = run_attempt_change(tmp_path, product)
    assert result["verdict"] == "PASS", result["errors"]
    first, second = FakeNcclJob.instances
    assert first.stalled and first.released == 0, "A is deleted stalled, never released"
    assert (
        isinstance(second, FakeSecondAttempt)
        and second.attempt_id == f"{first.name}-a002"
    )
    assert second.name == first.name and not second.stalled
    assert result["attempt_b"]["attempt_id"].endswith("-a002"), (
        "B is attempt 2 of the same job"
    )
    assert set(result["stages"]) == {"a_warned", "no_hung", "fresh_attempt"}
    assert (
        product.calls[0][3] == first.attempt_id
        and product.calls[-1][3] == f"{first.attempt_id},{second.attempt_id}"
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"b_warning": True}, "attempt B raised"),
        ({"state_names_a": True}, "names attempt"),
        ({"hung_a": True}, "HUNG_SUSPECTED"),
        ({"no_warning": True}, "no ZERO_WARNING for attempt"),
    ],
)
def test_attempt_change_fails_on_an_inherited_streak_or_a_missing_warning(
    tmp_path: Path, options: dict[str, bool], message: str
) -> None:
    result = run_attempt_change(tmp_path, AttemptProduct(**options))
    assert result["verdict"] == "FAIL"
    assert any(message in e for e in result["errors"]), result["errors"]


def test_attempt_change_is_inconclusive_when_a_keeps_moving_or_b_never_goes_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeNcclJob.traffic.stalled = 400.0
    with pytest.raises(VariantInconclusive, match="attempt A did not hold at zero"):
        run_attempt_change(tmp_path, AttemptProduct())
    FakeNcclJob.instances = []
    FakeNcclJob.traffic = Traffic()

    class QuietSecondAttempt(FakeSecondAttempt):
        def start(self) -> dict[str, Any]:
            FakeNcclJob.traffic.active = 100.0  # B never reaches the active threshold
            return super().start()

    monkeypatch.setattr(roles, "SecondAttemptNcclJob", QuietSecondAttempt)
    with pytest.raises(VariantInconclusive, match="no active EFA baseline"):
        run_attempt_change(tmp_path / "b", AttemptProduct())
    assert len(FakeNcclJob.instances) == 2 and not FakeNcclJob.instances[1].stalled


def test_fresh_attempt_errors_read_the_state_row() -> None:
    switched = T0
    records = {
        "incidents": [],
        "efa_states": {
            NODE: {
                "attempt_id": "j-a002",
                "first_observed_at": (switched + timedelta(seconds=10)).isoformat(),
                "had_active_traffic": True,
                "zero_since": None,
                "consecutive_zero_samples": 0,
                "signal": "NORMAL",
            },
            PEER: {
                "attempt_id": "j-a002",
                "first_observed_at": (switched - timedelta(seconds=300)).isoformat(),
                "had_active_traffic": False,
                "zero_since": (switched - timedelta(seconds=400)).isoformat(),
                "consecutive_zero_samples": 9,
                "signal": "ZERO_WARNING",
            },
        },
    }
    errors = roles.fresh_attempt_errors(records, "j-a002", switched, NODES)
    assert [e.split(": ", 1)[1][:24] for e in errors] == [
        "B's state was first obse",
        "B's state never saw acti",
        "B carries a zero streak ",
        "B's state reads ZERO_WAR",
    ]
    assert roles.fresh_attempt_errors(
        {"incidents": [], "efa_states": {}}, "j-a002", switched, NODES
    ) == [f"{NODE}: no EFA traffic state row", f"{PEER}: no EFA traffic state row"]
    assert roles.warned_errors({"incidents": []}, "j-a001", switched, NODES) == [
        f"{NODE}: no ZERO_WARNING for attempt j-a001",
        f"{PEER}: no ZERO_WARNING for attempt j-a001",
    ]
    early = {
        "incidents": [
            {
                "attempt_id": "j-a001",
                "event_id": f"x{nc.WARNING_SUFFIX}",
                "node_ids": [NODE],
                "created_at": (switched + timedelta(seconds=20)).isoformat(),
            }
        ]
    }
    assert (
        "before the threshold"
        in roles.warned_errors(early, "j-a001", switched, [NODE])[0]
    )
