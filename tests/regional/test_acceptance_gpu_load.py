"""The acceptance training fixtures keep their GPUs busy between heartbeats.

Every embedded training script is executed end to end against a fake torch, a
fake clock and a fake GPU timeline: the HEARTBEAT lines the cases parse, their
cadence, the step counts, the duty every nvidia-smi sample period would read
and the SIGTERM exit are observed, not grepped. The idle-by-design workloads
are pinned idle, and the NVIDIA definition the duty rests on is pinned in the
helper and the snippet copies.
"""

from __future__ import annotations

import ast
import bisect
import builtins
import re
import runpy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from scripts.e2e.regional import acceptance_gpu_load as load
from scripts.e2e.regional import (
    late_ownership_resources,
    notify005_checks,
    run_collector_destructive,
    run_destr014_branch_exhaustion,
    run_destr015_parallel_branch_join,
    training_topology,
)
from scripts.e2e.regional.collect017_training import STEP
from scripts.e2e.regional.managed_workload_fixture import (
    SUCCESS_RANK_PATTERN,
    heartbeat_healthy,
    render_node_pinned_manifest,
)
from scripts.e2e.regional.warm_spare_fixture import GpuHolderFixture

ROOT = Path(__file__).resolve().parents[2]
REGIONAL = ROOT / "scripts/e2e/regional"
TRAINING = REGIONAL / "manifests/training"
HEREDOC = re.compile(r"<<'PY'\n(.*?)\nPY(?:\n|$)", re.S)
SAMPLE_PERIODS = (1 / 6, 1.0)
KERNEL_SECONDS = 30e-6
LAUNCH_SECONDS = 4e-6
COLLECTIVE_SECONDS = 20e-6
SIMULATED_STEPS = 3
# The step loop of each GPU-active fixture: the bound the case timings were
# built on, the world size its collective proves and whether it prints SUCCESS.
FIXTURES = {
    "xid11-three-node-pytorchjob.yaml": (300, 24, True),
    "two-node-replace-pytorchjob.yaml": (3600, 16, True),
    "single-node-warm-spare-pytorchjob.yaml": (10800, 8, False),
    "xid11-single-node-job.yaml": (3600, 1, False),
}


# --------------------------------------------------------------------------- #
# Manifest reading
# --------------------------------------------------------------------------- #
def documents(path: Path) -> list[dict[str, Any]]:
    return [
        item
        for item in yaml.safe_load_all(path.read_text(encoding="utf-8"))
        if isinstance(item, dict)
    ]


def containers(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        found.extend(
            item for item in value.get("containers") or [] if isinstance(item, dict)
        )
        for item in value.values():
            found.extend(containers(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(containers(item))
    return found


def scripts_of(container: dict[str, Any]) -> list[str]:
    parts = [*(container.get("command") or []), *(container.get("args") or [])]
    return [match.group(1) for part in parts for match in HEREDOC.finditer(str(part))]


def training_scripts(name: str) -> list[tuple[dict[str, Any], str]]:
    return [
        (container, script)
        for document in documents(TRAINING / name)
        for container in containers(document)
        for script in scripts_of(container)
    ]


def duty_env(container: dict[str, Any]) -> list[Any]:
    return [
        item.get("value")
        for item in container.get("env") or []
        if isinstance(item, dict) and item.get("name") == load.GPU_DUTY_ENV
    ]


def step_loop(tree: ast.AST) -> ast.For | ast.While:
    """The fixture's one ``for step in range(N)`` / ``while step < N`` loop."""

    loops: list[ast.For | ast.While] = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "step"
        )
        or (
            isinstance(node, ast.While)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "step"
        )
    ]
    assert len(loops) == 1, [ast.unparse(loop)[:60] for loop in loops]
    return loops[0]


def call_names(node: ast.AST) -> list[str]:
    return [
        ast.unparse(item.func) for item in ast.walk(node) if isinstance(item, ast.Call)
    ]


def normalized(text: str) -> str:
    return " ".join(text.replace("#", " ").split())


# --------------------------------------------------------------------------- #
# The fake runtime a training script runs against
# --------------------------------------------------------------------------- #
class Exited(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


@dataclass
class Runtime:
    environ: dict[str, str]
    rank: int = 0
    world_size: int = 1
    now: float = 1_000.0
    free_at: float = 1_000.0
    sigterm_at: float | None = None
    busy: list[tuple[float, float]] = field(default_factory=list)
    events: list[tuple[Any, ...]] = field(default_factory=list)
    printed: list[tuple[float, str]] = field(default_factory=list)
    threads: list[tuple[Any, bool]] = field(default_factory=list)
    allocated_bytes: int = 0
    matmuls: int = 0
    queued_at_step: list[float] = field(default_factory=list)

    def launch(self, seconds: float) -> None:
        start = max(self.now, self.free_at)
        self.free_at = start + seconds
        self.busy.append((start, self.free_at))
        self.now += LAUNCH_SECONDS

    def synchronize(self, device: Any = None) -> None:
        self.events.append(("synchronize", str(device)))
        self.now = max(self.now, self.free_at)

    def sleep(self, seconds: float) -> None:
        self.events.append(("sleep", seconds))
        self.now += seconds


class Event:
    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.flag = False

    def set(self) -> None:
        self.flag = True

    def is_set(self) -> bool:
        return self.flag

    def wait(self, timeout: float) -> bool:
        runtime = self.runtime
        if not self.flag and runtime.sigterm_at is not None:
            if runtime.now + timeout >= runtime.sigterm_at:
                runtime.now = max(runtime.now, runtime.sigterm_at)
                self.flag = True
                return True
        if not self.flag:
            runtime.now += timeout
        return self.flag


@dataclass
class Tensor:
    runtime: Runtime
    values: list[float]
    shape: tuple[int, ...]
    dtype: Any = None
    device: Any = None

    def item(self) -> float:
        self.runtime.synchronize(self.device)
        return self.values[0]


def fake_modules(runtime: Runtime) -> dict[str, Any]:
    def device(kind: str, index: int) -> Device:
        return Device(kind, index)

    def tensor(values: list[float], device: Any = None) -> Tensor:
        # Every step starts by building its collective's tensor: whatever the
        # load left queued at that moment ran past the previous deadline.
        runtime.queued_at_step.append(max(0.0, runtime.free_at - runtime.now))
        return Tensor(runtime, list(values), (len(values),), device=device)

    def randn(size: tuple[int, ...], *, device: Any, dtype: Any) -> Tensor:
        runtime.allocated_bytes += size[0] * size[1] * 2
        return Tensor(runtime, [], tuple(size), dtype, device)

    def mm(first: Tensor, second: Tensor, *, out: Tensor) -> None:
        assert first.shape[1] == second.shape[0] == out.shape[0], (first, second, out)
        assert first.dtype == second.dtype == out.dtype == "float16", first.dtype
        runtime.matmuls += 1
        runtime.launch(KERNEL_SECONDS)

    def all_reduce(value: Tensor) -> None:
        runtime.launch(COLLECTIVE_SECONDS)
        value.values = [runtime.world_size * (runtime.world_size + 1) / 2]

    distributed = SimpleNamespace(
        init_process_group=lambda backend: runtime.events.append(("init", backend)),
        get_rank=lambda: runtime.rank,
        get_world_size=lambda: runtime.world_size,
        all_reduce=all_reduce,
        barrier=lambda: runtime.events.append(("barrier",)),
        destroy_process_group=lambda: runtime.events.append(("destroy",)),
    )
    torch = SimpleNamespace(
        device=device,
        tensor=tensor,
        randn=randn,
        mm=mm,
        float16="float16",
        cuda=SimpleNamespace(
            set_device=lambda index: runtime.events.append(("set_device", index)),
            synchronize=runtime.synchronize,
        ),
        distributed=distributed,
    )

    def exit_now(code: int) -> None:
        runtime.events.append(("exit", code))
        raise Exited(code)

    def sigwait(signals: set[int]) -> int:
        runtime.events.append(("sigwait", frozenset(signals)))
        return 15

    def thread(*, target: Any, daemon: bool) -> Any:
        return SimpleNamespace(start=lambda: runtime.threads.append((target, daemon)))

    return {
        "os": SimpleNamespace(environ=runtime.environ, _exit=exit_now),
        "signal": SimpleNamespace(
            SIGTERM=15,
            SIG_BLOCK=0,
            sigwait=sigwait,
            pthread_sigmask=lambda how, signals: runtime.events.append(
                ("sigmask", how, frozenset(signals))
            ),
        ),
        "socket": SimpleNamespace(gethostname=lambda: "node-a"),
        "threading": SimpleNamespace(Event=lambda: Event(runtime), Thread=thread),
        "time": SimpleNamespace(
            monotonic=lambda: runtime.now, time=lambda: runtime.now, sleep=runtime.sleep
        ),
        "torch": torch,
        "torch.distributed": torch,
    }


@dataclass(frozen=True)
class Device:
    kind: str
    index: int

    def __str__(self) -> str:
        return f"{self.kind}:{self.index}"


def namespace(runtime: Runtime) -> dict[str, Any]:
    fakes = fake_modules(runtime)
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in fakes:
            runtime.events.append(("import", name))
            return fakes[name]
        return real_import(name, *args, **kwargs)

    def fake_print(*values: Any, sep: str = " ", **_: Any) -> None:
        runtime.printed.append((runtime.now, sep.join(str(item) for item in values)))

    return {
        "__builtins__": {
            **vars(builtins),
            "__import__": fake_import,
            "print": fake_print,
        },
        "__name__": "__main__",
    }


class StepLimit(ast.NodeTransformer):
    """Run ``steps`` iterations of the fixture's step loop; record its bound."""

    def __init__(self, steps: int) -> None:
        self.steps = steps
        self.bounds: list[int] = []

    def visit_For(self, node: ast.For) -> ast.For:
        self.generic_visit(node)
        call = node.iter
        if (
            isinstance(node.target, ast.Name)
            and node.target.id == "step"
            and isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "range"
            and isinstance(call.args[0], ast.Constant)
        ):
            self.bounds.append(call.args[0].value)
            call.args[0] = ast.Constant(self.steps)
        return node

    def visit_While(self, node: ast.While) -> ast.While:
        self.generic_visit(node)
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "step"
            and isinstance(test.comparators[0], ast.Constant)
        ):
            self.bounds.append(test.comparators[0].value)
            test.comparators[0] = ast.Constant(self.steps)
        return node


def run_script(
    script: str, runtime: Runtime, *, steps: int = SIMULATED_STEPS
) -> tuple[list[int], dict[str, Any]]:
    tree = ast.parse(script)
    limit = StepLimit(steps)
    tree = ast.fix_missing_locations(limit.visit(tree))
    scope = namespace(runtime)
    exec(compile(tree, "<training script>", "exec"), scope)
    return limit.bounds, scope


class Timeline:
    """GPU busy time between two instants, from merged kernel intervals."""

    def __init__(self, intervals: list[tuple[float, float]]) -> None:
        merged: list[list[float]] = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        self.merged = merged
        self.starts = [start for start, _ in merged]
        self.cumulative = [0.0]
        for start, end in merged:
            self.cumulative.append(self.cumulative[-1] + end - start)

    def busy_before(self, instant: float) -> float:
        index = bisect.bisect_right(self.starts, instant)
        if index == 0:
            return 0.0
        start, end = self.merged[index - 1]
        return self.cumulative[index - 1] + min(instant, end) - start

    def utilization(self, start: float, end: float) -> float:
        return (self.busy_before(end) - self.busy_before(start)) / (end - start)

    def readings(self, start: float, end: float, period: float) -> list[float]:
        """What a sample of ``period`` reads, ending every millisecond."""

        count = int((end - start - period) * 1000)
        return [
            self.utilization(start + index / 1000, start + index / 1000 + period)
            for index in range(count)
        ]


def simulate(
    name: str, script: str, *, local_rank: int = 0, duty: str | None = None
) -> tuple[Runtime, list[int], dict[str, Any]]:
    _bound, world_size, _success = FIXTURES[name]
    environ = {"LOCAL_RANK": str(local_rank)}
    if duty is not None:
        environ[load.GPU_DUTY_ENV] = duty
    runtime = Runtime(environ=environ, rank=local_rank, world_size=world_size)
    bounds, scope = run_script(script, runtime)
    return runtime, bounds, scope


# --------------------------------------------------------------------------- #
# The copies and the inventory
# --------------------------------------------------------------------------- #
def test_training_fixture_inventory_is_classified() -> None:
    names = {path.name for path in TRAINING.glob("*.yaml")}

    assert names == {*load.GPU_ACTIVE_MANIFESTS, *load.SHORT_LIVED_MANIFESTS}, (
        "classify every training fixture: GPU-active or short-lived, with reason"
    )
    assert set(FIXTURES) == set(load.GPU_ACTIVE_MANIFESTS)
    for name in load.SHORT_LIVED_MANIFESTS:
        scripts = training_scripts(name)
        assert scripts, name
        for container, script in scripts:
            limit = StepLimit(0)
            limit.visit(ast.parse(script))
            assert limit.bounds == [3], (name, limit.bounds)
            assert load.SNIPPET_BEGIN not in script and duty_env(container) == [], name


@pytest.mark.parametrize("name", load.GPU_ACTIVE_MANIFESTS)
def test_every_training_script_carries_the_canonical_snippet(name: str) -> None:
    text = (TRAINING / name).read_text(encoding="utf-8")
    scripts = training_scripts(name)

    assert load.sync_snippet(text) == text, f"{name}: run acceptance_gpu_load --write"
    assert load.snippet_blocks(text) == len(scripts) >= 1, name
    for container, script in scripts:
        assert load.GPU_LOAD_SNIPPET in script, name
        assert duty_env(container) == [str(load.DEFAULT_GPU_DUTY_PERCENT)], name
        tree = ast.parse(script)
        calls = call_names(step_loop(tree))
        assert calls.count("gpu_busy_until") == 1, (name, calls)
        assert "time.sleep" not in calls, (name, calls)
        # The only sleep left is the SIGTERM watcher's bounded grace.
        watchers = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "watch"
        ]
        assert (
            call_names(tree).count("time.sleep")
            == 1
            == sum(call_names(watcher).count("time.sleep") for watcher in watchers)
        ), name


def test_the_check_command_passes_on_the_repository() -> None:
    assert load.main([]) == 0


def test_check_reports_and_write_repairs_a_drifted_copy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in load.GPU_ACTIVE_MANIFESTS:
        (tmp_path / name).write_text(
            (TRAINING / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    drifted = tmp_path / load.GPU_ACTIVE_MANIFESTS[0]
    original = drifted.read_text(encoding="utf-8")
    drifted.write_text(original.replace("GPU_LOAD_BATCH = 8", "GPU_LOAD_BATCH = 9"))

    assert load.main(["--root", str(tmp_path)]) == 1
    assert "drifted" in capsys.readouterr().out
    assert load.main(["--root", str(tmp_path), "--write"]) == 0
    assert "rewrote" in capsys.readouterr().out
    assert drifted.read_text(encoding="utf-8") == original

    stripped = tmp_path / load.GPU_ACTIVE_MANIFESTS[1]
    stripped.write_text(
        "".join(
            line
            for line in stripped.read_text(encoding="utf-8").splitlines(keepends=True)
            if load.SNIPPET_BEGIN not in line and load.SNIPPET_END not in line
        )
    )
    assert load.main(["--root", str(tmp_path), "--write"]) == 1
    assert "missing" in capsys.readouterr().out


def test_the_module_runs_as_a_check_script(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["acceptance_gpu_load"])
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(REGIONAL / "acceptance_gpu_load.py"), run_name="__main__")
    assert caught.value.code == 0


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (f"{load.SNIPPET_BEGIN}\nbody\n", "without END"),
        (f"body\n{load.SNIPPET_END}\n", "without BEGIN"),
        (
            f"{load.SNIPPET_BEGIN}\n  {load.SNIPPET_BEGIN} x\n{load.SNIPPET_END}\n",
            "nested",
        ),
    ],
)
def test_unbalanced_markers_are_refused(text: str, message: str) -> None:
    with pytest.raises(load.SnippetError, match=message):
        load.sync_snippet(text)


def test_sync_reindents_the_snippet_and_keeps_blank_lines_empty() -> None:
    text = f"head\n    {load.SNIPPET_BEGIN} (old)\n    stale\n    {load.SNIPPET_END}\ntail\n"
    synced = load.sync_snippet(text)
    lines = synced.splitlines()

    assert lines[0] == "head" and lines[-1] == "tail", lines
    body = lines[1:-1]
    assert [line.removeprefix("    ") for line in body] == (
        load.GPU_LOAD_SNIPPET.splitlines()
    )
    assert all(line == "" or line.startswith("    ") for line in body), body
    assert not any(line.endswith(" ") for line in synced.splitlines()), (
        "no trailing whitespace for yamllint"
    )


# --------------------------------------------------------------------------- #
# The scripts, executed
# --------------------------------------------------------------------------- #
def test_every_embedded_python_script_compiles() -> None:
    compiled = 0
    for path in sorted(
        [
            *REGIONAL.glob("manifests/**/*.yaml"),
            *(ROOT / "examples/hyperpod").glob("*.yaml"),
        ]
    ):
        for document in documents(path):
            if document.get("kind") == "ConfigMap":
                for key, value in (document.get("data") or {}).items():
                    if key.endswith(".py"):
                        compile(value, f"{path}:{key}", "exec")
                        compiled += 1
            for container in containers(document):
                for script in scripts_of(container):
                    compile(script, str(path), "exec")
                    compiled += 1
    assert compiled >= 14, compiled


@pytest.mark.parametrize(
    ("name", "index", "local_rank"),
    [
        (name, index, local_rank)
        for name in load.GPU_ACTIVE_MANIFESTS
        for index in range(len(training_scripts(name)))
        for local_rank in ((0, 1) if FIXTURES[name][1] > 1 else (0,))
    ],
)
def test_training_scripts_keep_the_heartbeats_and_read_the_duty(
    name: str, index: int, local_rank: int
) -> None:
    bound, world_size, success = FIXTURES[name]
    _container, script = training_scripts(name)[index]
    runtime, bounds, _scope = simulate(name, script, local_rank=local_rank)

    assert bounds == [bound], (name, bounds)
    sigmask = runtime.events.index(("sigmask", 0, frozenset({15})))
    first_torch = runtime.events.index(("import", "torch"))
    assert sigmask < first_torch, "SIGTERM is blocked before torch starts a thread"
    assert [daemon for _target, daemon in runtime.threads] == [True]

    heartbeats = [(when, line) for when, line in runtime.printed if "HEARTBEAT" in line]
    if local_rank:
        assert heartbeats == [], heartbeats
    else:
        assert len(heartbeats) == SIMULATED_STEPS, runtime.printed
        for step, (_when, line) in enumerate(heartbeats):
            match = STEP.search(line)
            assert match is not None and int(match.group(1)) == step, line
            assert heartbeat_healthy(line, world_size=world_size), line
        cadence = [
            later - earlier
            for (earlier, _), (later, _) in zip(heartbeats, heartbeats[1:])
        ]
        assert all(2.0 <= item < 2.002 for item in cadence), cadence
    successes = [line for _when, line in runtime.printed if line.startswith("SUCCESS")]
    if success:
        assert [SUCCESS_RANK_PATTERN.search(line).groups() for line in successes] == [
            (str(local_rank), str(world_size))
        ], successes

    # Nothing of the load is still queued when the next step starts.
    assert len(runtime.queued_at_step) == SIMULATED_STEPS, runtime.queued_at_step
    assert max(runtime.queued_at_step) == 0.0, runtime.queued_at_step
    assert runtime.allocated_bytes == 3 * load.GPU_LOAD_MATRIX**2 * 2
    timeline = Timeline(runtime.busy)
    start, end = runtime.busy[0][0], runtime.busy[-1][1]
    assert end - start >= (SIMULATED_STEPS - 1) * 2, (start, end)
    for period in SAMPLE_PERIODS:
        readings = timeline.readings(start, end, period)
        assert readings, period
        assert 0.25 <= min(readings) and max(readings) <= 0.36, (
            period,
            min(readings),
            max(readings),
        )


@pytest.mark.parametrize(
    ("duty", "low", "high"), [("0", 0.0, 0.001), ("10", 0.08, 0.13), ("50", 0.45, 0.56)]
)
def test_the_duty_follows_the_container_env(duty: str, low: float, high: float) -> None:
    name = load.GPU_ACTIVE_MANIFESTS[0]
    _container, script = training_scripts(name)[0]
    runtime, _bounds, _scope = simulate(name, script, duty=duty)

    timeline = Timeline(runtime.busy)
    start, end = runtime.printed[0][0], runtime.now
    readings = timeline.readings(start, end, SAMPLE_PERIODS[0])
    assert low <= min(readings) and max(readings) <= high, (
        duty,
        min(readings),
        max(readings),
    )
    if duty == "0":
        assert runtime.matmuls == 0 and runtime.allocated_bytes == 0, runtime.matmuls


@pytest.mark.parametrize("value", ["51", "-1", "abc", "３０", "1e1", "", "+5"])
def test_an_invalid_duty_stops_the_script_before_any_gpu_work(value: str) -> None:
    name = load.GPU_ACTIVE_MANIFESTS[0]
    _container, script = training_scripts(name)[0]
    runtime = Runtime(environ={"LOCAL_RANK": "0", load.GPU_DUTY_ENV: value})

    with pytest.raises(SystemExit, match=load.GPU_DUTY_ENV):
        run_script(script, runtime)
    assert ("import", "torch") not in runtime.events
    assert runtime.threads == [], "no SIGTERM watcher for a refused configuration"


def snippet_scope(runtime: Runtime) -> dict[str, Any]:
    scope = namespace(runtime)
    exec(compile("import os, signal, threading, time\n", "<imports>", "exec"), scope)
    exec(compile(load.GPU_LOAD_SNIPPET, "<snippet>", "exec"), scope)
    scope["torch"] = fake_modules(runtime)["torch"]
    return scope


def test_the_snippet_matches_the_helper_constants() -> None:
    runtime = Runtime(environ={"LOCAL_RANK": "0", load.GPU_DUTY_ENV: " 7 "})
    scope = snippet_scope(runtime)

    assert scope["GPU_DUTY_PERCENT"] == 7
    assert scope["GPU_DUTY_ENV"] == load.GPU_DUTY_ENV
    assert scope["GPU_DUTY_PERIOD_SECONDS"] == load.GPU_DUTY_PERIOD_SECONDS
    assert scope["GPU_LOAD_MATRIX"] == load.GPU_LOAD_MATRIX
    assert scope["GPU_LOAD_BATCH"] == load.GPU_LOAD_BATCH
    assert scope["GPU_TERM_GRACE_SECONDS"] == load.GPU_TERM_GRACE_SECONDS
    runtime.environ.pop(load.GPU_DUTY_ENV)
    assert scope["gpu_duty_percent"]() == load.DEFAULT_GPU_DUTY_PERCENT
    runtime.environ[load.GPU_DUTY_ENV] = str(load.MAX_GPU_DUTY_PERCENT)
    assert scope["gpu_duty_percent"]() == load.MAX_GPU_DUTY_PERCENT
    runtime.environ[load.GPU_DUTY_ENV] = str(load.MAX_GPU_DUTY_PERCENT + 1)
    with pytest.raises(SystemExit):
        scope["gpu_duty_percent"]()


def test_sigterm_finishes_the_batch_in_flight_and_exits_143() -> None:
    runtime = Runtime(environ={"LOCAL_RANK": "0"})
    scope = snippet_scope(runtime)
    device = Device("cuda", 0)
    runtime.sigterm_at = runtime.now + 0.5

    with pytest.raises(Exited) as caught:
        scope["gpu_busy_until"](runtime.now + 30, device)

    assert caught.value.code == load.SIGTERM_EXIT_CODE
    assert runtime.now < runtime.sigterm_at + load.GPU_DUTY_PERIOD_SECONDS, runtime.now
    assert runtime.events[-2:] == [("synchronize", "cuda:0"), ("exit", 143)]
    assert runtime.free_at <= runtime.now, "no kernel is running at the exit"


def test_the_watcher_exits_143_when_the_main_thread_is_stuck() -> None:
    runtime = Runtime(environ={"LOCAL_RANK": "0"})
    scope = snippet_scope(runtime)
    [(watch, daemon)] = runtime.threads
    terminating = scope["TERMINATING"]

    assert daemon is True and terminating.is_set() is False
    with pytest.raises(Exited) as caught:
        watch()
    assert caught.value.code == load.SIGTERM_EXIT_CODE
    assert terminating.is_set() is True
    assert runtime.events[-3:] == [
        ("sigwait", frozenset({15})),
        ("sleep", load.GPU_TERM_GRACE_SECONDS),
        ("exit", 143),
    ]


def test_a_deadline_already_past_returns_without_gpu_work() -> None:
    runtime = Runtime(environ={"LOCAL_RANK": "0"})
    scope = snippet_scope(runtime)

    assert scope["gpu_busy_until"](runtime.now - 1, Device("cuda", 0)) is None
    assert runtime.matmuls == 0, "the matrices exist but no kernel ran"


# --------------------------------------------------------------------------- #
# The rendered copies the runners submit
# --------------------------------------------------------------------------- #
def rendered_scripts(path: Path) -> list[tuple[dict[str, Any], str]]:
    return [
        (container, script)
        for document in documents(path)
        for container in containers(document)
        for script in scripts_of(container)
    ]


def assert_loaded(path: Path, expected: int) -> None:
    scripts = rendered_scripts(path)
    assert len(scripts) == expected, (path, len(scripts))
    for container, script in scripts:
        assert load.GPU_LOAD_SNIPPET in script, path
        assert duty_env(container) == [str(load.DEFAULT_GPU_DUTY_PERCENT)], path


def test_the_renderers_keep_the_load_and_its_env(tmp_path: Path) -> None:
    named = run_collector_destructive.render_named_training_manifest(
        tmp_path / "named.yaml", name="gpu-fault-unit"
    )
    assert_loaded(named, 2)
    topology = training_topology.TrainingTopology(
        node_count=2, gpus_per_node=4, candidate_nodes=("a", "b"), efa_per_node=0
    )
    shrunk = training_topology.render_topology_manifest(
        named, topology, tmp_path / "t.yaml"
    )
    assert_loaded(shrunk, 2)
    assert "--nproc-per-node=4" in yaml.safe_dump(documents(shrunk))
    for runner in (run_destr014_branch_exhaustion, run_destr015_parallel_branch_join):
        pinned = runner.render_two_node_manifest(
            TRAINING / "two-node-replace-pytorchjob.yaml",
            tmp_path / f"{runner.__name__}.yaml",
            master_node="a",
            worker_node="b",
        )
        assert_loaded(pinned, 2)
    single = render_node_pinned_manifest(
        TRAINING / "xid11-single-node-job.yaml", tmp_path / "single.yaml", node="a"
    )
    assert_loaded(single, 1)


def test_a_case_can_opt_out_of_the_load() -> None:
    document = documents(TRAINING / "xid11-three-node-pytorchjob.yaml")[0]
    pytorch = containers(document)
    pytorch[0]["env"].insert(0, "not-a-mapping")

    assert load.set_gpu_duty_percent(document, 0) is document
    for container in pytorch:
        assert duty_env(container) == ["0"], container["env"]
        assert {"name": "NCCL_DEBUG", "value": "INFO"} in container["env"]
    assert "not-a-mapping" in pytorch[0]["env"]
    job = documents(TRAINING / "xid11-single-node-job.yaml")[0]
    for container in containers(job):
        container["args"] = container.pop("command")
        container.pop("env")
    load.set_gpu_duty_percent(job, 12)
    assert [duty_env(item) for item in containers(job)] == [["12"]]
    for invalid in (-1, load.MAX_GPU_DUTY_PERCENT + 1, True, "0"):
        with pytest.raises(ValueError, match="integer"):
            load.set_gpu_duty_percent(document, invalid)  # type: ignore[arg-type]
    idle = notify005_checks.low_utilization_manifest(name="n", nodes=("a",), image="i")
    with pytest.raises(ValueError, match="no acceptance GPU load"):
        load.set_gpu_duty_percent(idle, 0)


# --------------------------------------------------------------------------- #
# The workloads that stay idle on purpose
# --------------------------------------------------------------------------- #
def python_calls(code: str) -> set[str]:
    return {
        ast.unparse(node.func)
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call)
    }


def test_the_idle_by_design_workloads_stay_gpu_idle() -> None:
    for nodes in (("a",), ("a", "b", "c")):
        manifest = notify005_checks.low_utilization_manifest(
            name="notify005", nodes=nodes, image="image"
        )
        for container in containers(manifest):
            [script] = scripts_of(container)
            assert "torch" not in script, "NOTIFY-005 burns CPU only"
            assert load.SNIPPET_BEGIN not in script and duty_env(container) == []
            compile(script, "notify005", "exec")
    warm = SimpleNamespace(
        regional=SimpleNamespace(settings=SimpleNamespace(namespace="ns"))
    )
    holder = GpuHolderFixture(warm, node="spare", run_id="run")  # type: ignore[arg-type]
    [container] = containers(holder.manifest())
    code = container["command"][-1].removeprefix("exec python3 -c '").removesuffix("'")
    assert "time.sleep" in python_calls(code), code
    assert not {"torch.mm", "torch.matmul"} & python_calls(code), code
    assert duty_env(container) == []
    calls = python_calls(late_ownership_resources.CUDA_HOLDER)
    assert "threading.Event().wait" in calls, calls
    assert not {"torch.mm", "torch.matmul"} & calls, calls


# --------------------------------------------------------------------------- #
# The vendor evidence
# --------------------------------------------------------------------------- #
def test_the_nvidia_evidence_is_pinned_in_the_helper_and_the_snippet() -> None:
    assert (
        load.NVIDIA_SMI_REFERENCE
        == "https://docs.nvidia.com/deploy/nvidia-smi/index.html"
    )
    assert load.NVIDIA_SMI_SECTION == "GPU ATTRIBUTES > Utilization > GPU"
    assert load.NVIDIA_SMI_DOCS_VERSION == "nvidia-smi v610 Update"
    assert (
        load.NVIDIA_SMI_CHANGE_LOG == "Changes between nvidia-smi v610 Update and v595"
    )
    assert load.NVIDIA_SMI_UTILIZATION_GPU_QUOTE == (
        "Percent of time over the past sample period during which one or more kernels "
        "was executing on the GPU. The sample period may be between 1 second and 1/6 "
        "second depending on the product."
    )
    assert load.NVIDIA_XID_CATALOG_REFERENCE == (
        "https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html"
    )
    assert load.NVIDIA_XID_CATALOG_DOCS_VERSION == "615"
    assert load.NVIDIA_XID45_TRIGGER_QUOTE == (
        "This event is logged when the user application aborts and the kernel driver "
        "tears down the GPU application running on the GPU. Control-C, GPU resets, "
        "sigkill are all examples where the application is aborted and this event is "
        "created."
    )
    snippet = normalized(load.GPU_LOAD_SNIPPET)
    docstring = normalized(load.__doc__ or "")
    for evidence in (
        load.NVIDIA_SMI_UTILIZATION_GPU_QUOTE,
        load.NVIDIA_SMI_REFERENCE,
        load.NVIDIA_SMI_SECTION,
        load.NVIDIA_SMI_DOCS_VERSION,
    ):
        assert evidence in snippet, evidence
        assert evidence in docstring, evidence
    for evidence in (
        load.NVIDIA_SMI_CHANGE_LOG,
        load.NVIDIA_XID_CATALOG_REFERENCE,
        load.NVIDIA_XID45_TRIGGER_QUOTE,
        f"docs version {load.NVIDIA_XID_CATALOG_DOCS_VERSION}",
    ):
        assert evidence in docstring, evidence
