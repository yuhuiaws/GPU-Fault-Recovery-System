"""The role, thread and process modes of the GF-GENERAL-LIVE-011 trainer.

The rendered trainer runs end to end against the fake torch, clock and GPU of
``test_general_live_nccl_job`` plus a thread-aware ``threading`` fake (threads
are recorded, never run; each gets a native id as CPython gives one on
``start()``), a ``subprocess`` fake that records the child processes and a
``shutil`` fake for the compiler lookup. What is observed: the ``ROLES``
line, the ``rank-progress/v1`` snapshot the product's own model validates
(``diagnostic_context`` included), which helper threads exist and in which
order, when the stall switches them on and off, the GPU sharer and the
non-Python CUDA holder (their command lines, stripped environment, failure
reporting) and the children's end with the job.
"""

from __future__ import annotations

import builtins
import itertools
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.rank_progress import RankProgressSnapshot
from scripts.e2e.regional import general_live_nccl_job as nj
from scripts.e2e.regional import general_live_nccl_role_trainer as plan
from tests.regional.test_general_live_nccl_job import (
    WORLD,
    Flag,
    Runtime,
    fakes,
    render,
    script_of,
    stall_window,
    trainer_env,
)

__all__ = ["trainer_env"]  # the imported fixture


class FakeEvent(Flag):
    def __init__(self, runtime: Runtime) -> None:
        super().__init__(runtime)
        self.history: list[tuple[float, str]] = []

    def set(self) -> None:
        super().set()
        self.history.append((self.runtime.now, "set"))

    def clear(self) -> None:
        self.value = False
        self.history.append((self.runtime.now, "clear"))


class FakeThread:
    """A thread that never runs; ``start`` hands out the native id."""

    def __init__(
        self, target: Callable[..., Any], daemon: bool, name: str | None = None
    ) -> None:
        self.target = target
        self.daemon = daemon
        self.name = name or "Thread"
        self.native_id: int | None = None
        self.started: list[FakeThread] | None = None

    def start(self) -> None:
        self.native_id = next(THREAD_IDS)
        assert self.started is not None, "the fake binds its registry in run()"
        self.started.append(self)

    def join(self, timeout: float | None = None) -> None:
        return None


class FakePopen:
    def __init__(
        self,
        argv: list[str],
        *,
        env: dict[str, str] | None = None,
        stdout: Any = None,
        stderr: Any = None,
    ) -> None:
        self.argv = list(argv)
        self.env = dict(env or {})
        self.pid = next(CHILD_PIDS)
        self.returncode: int | None = None
        self.terminated = False
        if hasattr(stdout, "close"):
            stdout.close()

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True


THREAD_IDS = itertools.count(40_000)
CHILD_PIDS = itertools.count(900)
MAIN_NATIVE_ID = 777


def run_roles(
    script: str,
    runtime: Runtime,
    *,
    compiler: str | None = "/usr/bin/gcc",
    compile_returncode: int = 0,
    holder_exit: int | None = None,
) -> dict[str, Any]:
    """Run ``script`` like ``test_general_live_nccl_job.run`` with the fakes a
    role mode touches; the scope carries ``threads``, ``popens``, ``runs``."""

    threads: list[FakeThread] = []
    popens: list[FakePopen] = []
    runs: list[list[str]] = []

    def thread(
        target: Callable[..., Any], daemon: bool, name: str | None = None
    ) -> FakeThread:
        created = FakeThread(target, daemon, name)
        created.started = threads
        return created

    def popen(argv: list[str], **options: Any) -> FakePopen:
        child = FakePopen(argv, **options)
        if holder_exit is not None and argv[0].endswith(
            f"{plan.HOLDER_EXE}-{runtime.rank}"
        ):
            child.returncode = holder_exit
        popens.append(child)
        return child

    def run_command(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        runs.append(list(argv))
        return subprocess.CompletedProcess(
            argv, compile_returncode, "", "boom" if compile_returncode else ""
        )

    modules = {
        **fakes(runtime),
        "threading": SimpleNamespace(
            Event=lambda: FakeEvent(runtime),
            Thread=thread,
            get_native_id=lambda: MAIN_NATIVE_ID,
            current_thread=lambda: SimpleNamespace(name="MainThread"),
        ),
        "subprocess": SimpleNamespace(
            Popen=popen,
            run=run_command,
            STDOUT=subprocess.STDOUT,
            SubprocessError=subprocess.SubprocessError,
        ),
        "shutil": SimpleNamespace(
            which=lambda name: compiler if name == "gcc" else None
        ),
    }
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in modules:
            return modules[name]
        return real_import(name, *args, **kwargs)

    def fake_print(*values: Any, sep: str = " ", **_: Any) -> None:
        runtime.printed.append((runtime.now, sep.join(str(item) for item in values)))

    scope: dict[str, Any] = {
        "__builtins__": {
            **vars(builtins),
            "__import__": fake_import,
            "print": fake_print,
        },
        "__name__": "__main__",
    }
    exec(compile(script, "<nccl role trainer>", "exec"), scope)
    scope["threads"], scope["popens"], scope["runs"] = threads, popens, runs
    return scope


@pytest.fixture
def role_env(
    trainer_env: Callable[..., dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., dict[str, Any]]:
    """``trainer_env`` plus the per-rank progress file and a private work dir."""

    def configure(rank: int, mode: str, **values: Any) -> dict[str, Any]:
        progress = tmp_path / "progress" / f"rank-{rank}.json"
        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        paths = trainer_env(
            rank,
            GFRS_STALL_MODE=mode,
            GFRS_STALL_MAX_SECONDS="70",
            GFRS_ROLE_WORK_DIR=str(work),
            **values,
        )
        monkeypatch.setenv("GPU_FAULT_RANK_PROGRESS_FILE", str(progress))
        monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
        monkeypatch.setenv("WORLD_SIZE", str(WORLD))
        return {**paths, "progress": progress, "work": work}

    return configure


def helper_names(scope: dict[str, Any]) -> list[str]:
    """The role fixture's threads, in start order (not the GPU load's watcher)."""

    return [
        thread.name for thread in scope["threads"] if thread.name.startswith("gfrs-")
    ]


def roles_line(runtime: Runtime) -> dict[str, str]:
    [line] = [text for _when, text in runtime.printed if plan.ROLES_MARKER in text]
    tokens = plan.parse_marker(line, plan.ROLES_MARKER)
    assert tokens is not None
    return tokens


def snapshot(path: Path) -> RankProgressSnapshot:
    return RankProgressSnapshot.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def test_the_role_modes_render_a_progress_dir_and_the_extension() -> None:
    assert set(plan.ROLE_MODES) <= set(nj.STALL_MODES)
    for mode in plan.ROLE_MODES:
        document = render(stall_mode=mode)
        script = script_of(document)
        assert "class RoleFixture" in script and plan.ROLES_MARKER.strip() in script
        for role in ("Master", "Worker"):
            container = next(
                c
                for c in document["spec"]["pytorchReplicaSpecs"][role]["template"][
                    "spec"
                ]["containers"]
                if c["name"] == "pytorch"
            )
            values = {item["name"]: item["value"] for item in container["env"]}
            assert values["GFRS_STALL_MODE"] == mode
            assert values["GFRS_RANK_PROGRESS_DIR"] == nj.RANK_PROGRESS_DIR
    quiet = {
        item["name"]
        for item in next(
            c
            for c in render()["spec"]["pytorchReplicaSpecs"]["Master"]["template"][
                "spec"
            ]["containers"]
            if c["name"] == "pytorch"
        )["env"]
    }
    assert "GFRS_RANK_PROGRESS_DIR" not in quiet


def test_the_plan_is_what_the_trainer_carries() -> None:
    [embedded] = [
        line
        for line in plan.ROLE_TRAINER.splitlines()
        if line.startswith("ROLE_PLAN = json.loads(")
    ]
    carried = json.loads(embedded[len('ROLE_PLAN = json.loads("""') : -len('""")')])
    assert carried == plan.PLAN
    assert plan.expected_context(plan.ROLES, 11, world_size=WORLD) == {
        "group_name": "pp0-expert-node1",
        "members": [8, 9, 10, 11],
        "role": plan.ROLE_NAME,
        "pipeline_stage": "pp0",
        "expert_role": "expert",
        "hardware_class": plan.HARDWARE_CLASS,
        "code_version": plan.CODE_VERSION,
    }
    assert plan.expected_context(plan.ROLES, 15, world_size=WORLD) is None
    assert plan.burning_locals(plan.ROLES) == [0, 1, 2, 3]
    assert plan.sharing_locals(plan.CPU) == [6, 7] and plan.holder_locals(
        plan.PYSPY
    ) == [7]
    with pytest.raises(ValueError, match="outside a world"):
        plan.expected_context(plan.ROLES, 16, world_size=WORLD)
    # The C source reaches the container with its escapes intact (a raw literal).
    assert 'fprintf(stderr, "usage: device seconds stopfile\\n")' in plan.ROLE_TRAINER
    assert (
        "cuCtxCreate_v2" in plan.ROLE_TRAINER
        and "PR_SET_PDEATHSIG" in plan.ROLE_TRAINER
    )


def test_marker_lines_parse_with_a_torchrun_prefix_and_refuse_malformed_roles() -> None:
    line = (
        "[rank3]: ROLES mode=roles node_index=0 group=pp0-expert-node0 stage=pp0 "
        "expert=expert members=0,1,2,3 roles=1:TRAINING:MainThread,29:COMMUNICATION:gfrs-comm "
        "clk_tck=100 pid=1 burn=1 share_gpu=0 holder=0 rank=3 node=h elapsed=1.0"
    )
    tokens = plan.parse_marker(line, plan.ROLES_MARKER)
    assert (
        tokens is not None
        and tokens["group"] == "pp0-expert-node0"
        and tokens["rank"] == "3"
    )
    assert plan.declared_roles(tokens) == {
        "1": ("TRAINING", "MainThread"),
        "29": ("COMMUNICATION", "gfrs-comm"),
    }
    assert plan.declared_roles({"roles": "-"}) == {}
    with pytest.raises(ValueError, match="malformed"):
        plan.declared_roles({"roles": "x:TRAINING:main"})
    with pytest.raises(ValueError, match="malformed"):
        plan.declared_roles({"roles": "7:DRIVER:main"})
    assert plan.parse_marker("HEARTBEAT rank=0", plan.ROLES_MARKER) is None
    assert plan.marker_lines([line, "noise"], plan.ROLES_MARKER) == [tokens]


# --------------------------------------------------------------------------- #
# the roles mode
# --------------------------------------------------------------------------- #
def test_a_pp0_rank_declares_its_group_and_burns_only_while_stalled(
    role_env: Callable[..., dict[str, Any]],
) -> None:
    paths = role_env(3, plan.ROLES)
    runtime = Runtime(rank=3, votes={0: 1.0})
    scope = run_roles(script_of(render(stall_mode=plan.ROLES)), runtime)

    tokens = roles_line(runtime)
    assert tokens["mode"] == "roles" and tokens["group"] == "pp0-expert-node0"
    assert (tokens["stage"], tokens["expert"], tokens["members"]) == (
        "pp0",
        "expert",
        "0,1,2,3",
    )
    assert (tokens["burn"], tokens["share_gpu"], tokens["holder"]) == ("1", "0", "0")
    assert tokens["clk_tck"] == str(os.sysconf("SC_CLK_TCK")) and tokens["pid"] == str(
        os.getpid()
    )
    roles = plan.declared_roles(tokens)
    names = {name for _role, name in roles.values()}
    assert names == {
        "MainThread",
        "gfrs-comm",
        "gfrs-keeper",
        "gfrs-burn-0",
        "gfrs-burn-1",
        "gfrs-burn-2",
    }
    assert roles[str(MAIN_NATIVE_ID)] == ("TRAINING", "MainThread")
    assert {name for name in names if name.startswith("gfrs-burn")} == {
        name
        for tid, (role, name) in roles.items()
        if role == "BACKGROUND" and name.startswith("gfrs-burn")
    }
    threads = [thread for thread in scope["threads"] if thread.name.startswith("gfrs-")]
    assert helper_names(scope) == [
        "gfrs-comm",
        "gfrs-keeper",
        "gfrs-burn-0",
        "gfrs-burn-1",
        "gfrs-burn-2",
    ], "helpers exist from the start, in declaration order"
    assert all(thread.daemon for thread in scope["threads"]), (
        "helper threads are daemons"
    )
    assert {str(thread.native_id) for thread in threads} | {str(MAIN_NATIVE_ID)} == set(
        roles
    )

    document = snapshot(paths["progress"])
    assert document.diagnostic_context is not None
    context = document.diagnostic_context
    assert (context.group_name, context.members) == ("pp0-expert-node0", [0, 1, 2, 3])
    assert (context.pipeline_stage, context.expert_role, context.role) == (
        "pp0",
        "expert",
        plan.ROLE_NAME,
    )
    assert context.thread_roles == {tid: role for tid, (role, _n) in roles.items()}
    assert document.rank == 3 and document.world_size == WORLD and document.step == 3

    fixture = scope["ROLE_FIXTURE"]
    stalled, joined = stall_window(runtime)
    assert fixture.active.history[:2] == [(stalled, "set"), (joined, "clear")], (
        "the helpers switch on with the stall and off when the rank joins"
    )
    assert fixture.active.history[2:] == [(runtime.now, "clear")], (
        "finish() leaves the helpers switched off"
    )
    assert fixture.stop.value is True, "finish() stops the helpers with the job"
    assert scope["popens"] == [] and scope["runs"] == []
    assert runtime.lines("SUCCESS"), "the job completes"


def test_pp1_and_undeclared_ranks_stay_quiet(
    role_env: Callable[..., dict[str, Any]],
) -> None:
    paths = role_env(13, plan.ROLES)
    runtime = Runtime(rank=13, votes={0: 1.0})
    scope = run_roles(script_of(render(stall_mode=plan.ROLES), "Worker"), runtime)
    tokens = roles_line(runtime)
    assert tokens["group"] == "pp1-dense-node1" and tokens["members"] == "12,13,14"
    assert tokens["burn"] == "0"
    assert helper_names(scope) == ["gfrs-comm", "gfrs-keeper"]
    context = snapshot(paths["progress"]).diagnostic_context
    assert context is not None and context.pipeline_stage == "pp1"

    paths = role_env(7, plan.ROLES)
    runtime = Runtime(rank=7, votes={0: 1.0})
    scope = run_roles(script_of(render(stall_mode=plan.ROLES)), runtime)
    tokens = roles_line(runtime)
    assert (tokens["group"], tokens["stage"], tokens["members"], tokens["roles"]) == (
        "-",
        "-",
        "-",
        "-",
    )
    assert plan.declared_roles(tokens) == {}
    assert snapshot(paths["progress"]).diagnostic_context is None
    assert helper_names(scope) == ["gfrs-comm", "gfrs-keeper"]


def test_the_keeper_refreshes_the_snapshot_and_a_step_moves_it(
    role_env: Callable[..., dict[str, Any]],
) -> None:
    paths = role_env(1, plan.ROLES, GFRS_STEPS="2")
    runtime = Runtime(rank=1)
    scope = run_roles(script_of(render(stall_mode=plan.ROLES)), runtime)
    fixture = scope["ROLE_FIXTURE"]
    assert snapshot(paths["progress"]).step == 1
    fixture.step(41)
    first = snapshot(paths["progress"])
    assert first.step == 41 and first.diagnostic_context is not None
    fixture.stop.value = False
    fixture.current_step = 42
    # One keeper tick: the wait returns False once (the flag flips on the way).
    waits = iter([False, True])
    fixture.stop.wait = lambda timeout=None: next(waits)
    fixture.keeper_loop()
    second = snapshot(paths["progress"])
    assert second.step == 42 and second.observed_at >= first.observed_at


# --------------------------------------------------------------------------- #
# threads, wchan, cpu, pyspy
# --------------------------------------------------------------------------- #
def test_threads_mode_declares_helpers_in_rank_dependent_order_and_churns(
    role_env: Callable[..., dict[str, Any]],
) -> None:
    expected = ["gfrs-comm", "gfrs-keeper", "gfrs-bg-a", "gfrs-bg-b", "gfrs-churn"]
    for rank, order in ((2, expected), (3, list(reversed(expected)))):
        role_env(rank, plan.THREADS)
        runtime = Runtime(rank=rank, votes={0: 1.0})
        scope = run_roles(script_of(render(stall_mode=plan.THREADS)), runtime)
        assert helper_names(scope) == order, rank
        roles = plan.declared_roles(roles_line(runtime))
        declared = {name: role for _tid, (role, name) in roles.items()}
        assert declared == {
            "MainThread": "TRAINING",
            "gfrs-comm": "COMMUNICATION",
            "gfrs-keeper": "BACKGROUND",
            "gfrs-bg-a": "BACKGROUND",
            "gfrs-bg-b": "BACKGROUND",
        }, "the churn thread is never declared"
        churn = next(
            thread for thread in scope["threads"] if thread.name == "gfrs-churn"
        )
        fixture = scope["ROLE_FIXTURE"]
        # One churn round: a short undeclared thread is started and joined,
        # then the stop flag reads set.
        fixture.active.value = True
        rounds = iter([False, True])
        fixture.stop.is_set = lambda: next(rounds)
        fixture.stop.wait = lambda timeout=None: False
        churn.target()
        assert scope["threads"][-1].name == "gfrs-churn-x"


def test_wchan_mode_declares_a_spinner_and_a_sleeper(
    role_env: Callable[..., dict[str, Any]],
) -> None:
    role_env(4, plan.WCHAN)
    runtime = Runtime(rank=4, votes={0: 1.0})
    scope = run_roles(script_of(render(stall_mode=plan.WCHAN)), runtime)
    roles = plan.declared_roles(roles_line(runtime))
    declared = {name: role for _tid, (role, name) in roles.items()}
    assert (
        declared["gfrs-spinner"] == "COMMUNICATION"
        and declared["gfrs-sleeper"] == "BACKGROUND"
    )
    fixture = scope["ROLE_FIXTURE"]
    spinner = next(
        thread for thread in scope["threads"] if thread.name == "gfrs-spinner"
    )
    # Spinning is CPU matmul on CPU tensors: no GIL, no GPU.
    fixture.active.value = True
    flips = iter([False, True])
    fixture.stop.is_set = lambda: next(flips)
    spinner.target()
    assert runtime.matmuls, "the spinner burns CPU with torch.mm while stalled"
    assert all(call[0] != "set_device" or call[1] == 4 for call in runtime.calls), (
        "the spinner touches no other device"
    )


def test_cpu_mode_shares_the_gpu_of_local_ranks_6_and_7_with_an_unranked_child(
    role_env: Callable[..., dict[str, Any]],
) -> None:
    paths = role_env(14, plan.CPU)
    runtime = Runtime(rank=14, votes={0: 1.0})
    scope = run_roles(script_of(render(stall_mode=plan.CPU), "Worker"), runtime)
    tokens = roles_line(runtime)
    assert tokens["share_gpu"] == "1" and tokens["group"] == "pp0-dense-node1"
    declared = {
        name: role for _tid, (role, name) in plan.declared_roles(tokens).items()
    }
    assert declared["gfrs-cpu-duty"] == "BACKGROUND"
    [child] = scope["popens"]
    [sharer] = [
        plan.parse_marker(t, plan.SHARER_MARKER)
        for _w, t in runtime.printed
        if plan.SHARER_MARKER in t
    ]
    assert (
        sharer is not None
        and sharer["pid"] == str(child.pid)
        and sharer["device"] == "6"
    )
    assert (
        child.argv[:2] == [os.sys.executable, "-c"]
        and "torch.cuda.set_device" in child.argv[2]
    )
    assert child.argv[3:] == [
        "6",
        str(70 + plan.CHILD_MARGIN_SECONDS) + ".0",
        str(paths["work"] / plan.STOP_FILE),
    ]
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "GPU_FAULT_RANK_PROGRESS_FILE",
    ):
        assert key not in child.env, key
    assert child.env["GFRS_STALL_MODE"] == plan.CPU, (
        "the rest of the environment is kept"
    )
    assert (
        child.terminated and (paths["work"] / plan.STOP_FILE).read_text() == "stop\n"
    ), "finish() stops the children with the job"

    role_env(13, plan.CPU)
    runtime = Runtime(rank=13, votes={0: 1.0})
    scope = run_roles(script_of(render(stall_mode=plan.CPU), "Worker"), runtime)
    assert scope["popens"] == [] and roles_line(runtime)["share_gpu"] == "0"


def test_pyspy_mode_builds_and_runs_the_non_python_holder_on_local_rank_7(
    role_env: Callable[..., dict[str, Any]],
) -> None:
    paths = role_env(7, plan.PYSPY)
    runtime = Runtime(rank=7, votes={0: 1.0})
    scope = run_roles(script_of(render(stall_mode=plan.PYSPY)), runtime)
    exe = str(paths["work"] / f"{plan.HOLDER_EXE}-7")
    source = paths["work"] / plan.HOLDER_SOURCE
    assert scope["runs"] == [["/usr/bin/gcc", "-O2", "-o", exe, str(source), "-ldl"]]
    text = source.read_text(encoding="utf-8")
    assert '"usage: device seconds stopfile\\n"' in text and "cuCtxCreate_v2" in text
    assert "prctl(PR_SET_PDEATHSIG, SIGTERM)" in text
    [child] = scope["popens"]
    assert child.argv == [
        exe,
        "7",
        str(70 + plan.CHILD_MARGIN_SECONDS),
        str(paths["work"] / plan.STOP_FILE),
    ]
    assert "RANK" not in child.env
    [holder] = [
        plan.parse_marker(t, plan.HOLDER_MARKER)
        for _w, t in runtime.printed
        if plan.HOLDER_MARKER in t
    ]
    assert (
        holder is not None and holder["pid"] == str(child.pid) and holder["exe"] == exe
    )
    assert roles_line(runtime)["holder"] == "1"
    assert not [t for _w, t in runtime.printed if plan.FIXTURE_ERROR_MARKER in t]

    role_env(6, plan.PYSPY)
    runtime = Runtime(rank=6, votes={0: 1.0})
    scope = run_roles(script_of(render(stall_mode=plan.PYSPY)), runtime)
    assert scope["popens"] == [] and scope["runs"] == []


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"compiler": None}, "no_C_compiler"),
        ({"compile_returncode": 1}, "compile_1:boom"),
        ({"holder_exit": 4}, "exited_4"),
    ],
)
def test_a_holder_the_container_cannot_build_or_keep_is_reported_not_faked(
    role_env: Callable[..., dict[str, Any]], options: dict[str, Any], message: str
) -> None:
    role_env(15, plan.PYSPY)
    runtime = Runtime(rank=15, votes={0: 1.0})
    scope = run_roles(
        script_of(render(stall_mode=plan.PYSPY), "Worker"), runtime, **options
    )
    [error] = [
        plan.parse_marker(t, plan.FIXTURE_ERROR_MARKER)
        for _w, t in runtime.printed
        if plan.FIXTURE_ERROR_MARKER in t
    ]
    assert error is not None and error["what"] == "holder" and message in error["error"]
    assert not [t for _w, t in runtime.printed if plan.HOLDER_MARKER in t]
    assert runtime.lines("SUCCESS"), "a fixture failure never ends the job"
    if "holder_exit" in options:
        assert len(scope["popens"]) == 1


def test_an_unknown_mode_still_stops_before_any_collective(
    role_env: Callable[..., dict[str, Any]],
) -> None:
    role_env(0, "loud")
    runtime = Runtime(rank=0)
    with pytest.raises(SystemExit, match="GFRS_STALL_MODE"):
        run_roles(script_of(render()), runtime)
    assert runtime.calls == []
