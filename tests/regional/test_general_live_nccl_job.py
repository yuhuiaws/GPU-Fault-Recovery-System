"""GF-GENERAL-LIVE-011's two-node NCCL stall job: the rendering and the trainer.

The rendered manifest is checked field by field against the source fixture, and
the trainer it carries is executed end to end against a fake torch, a fake
clock and a fake GPU: the payload every step, the pause vote, rank 0's extra
collective, the held ranks' release and bounded watchdog, the write/phase side
activity (the phase snapshot is validated by the product's own model) and the
SIGTERM exit are observed, not grepped.
"""

from __future__ import annotations

import builtins
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault.rank_progress import RankProgressSnapshot
from scripts.e2e.regional import acceptance_gpu_load as load
from scripts.e2e.regional import general_live_nccl as nc
from scripts.e2e.regional import general_live_nccl_job as nj
from scripts.e2e.regional import managed_workload_fixture as mwf
from scripts.e2e.regional.managed_workload_fixture import heartbeat_healthy
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings

SOURCE = Path(nj.__file__).with_name("manifests") / "training" / nj.TWO_NODE_MANIFEST
HEREDOC = re.compile(r"<<'PY'\n(.*?)\nPY\n", re.S)
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
PEER = "ip-10-0-43-6.us-west-2.compute.internal"
WORLD = 16
MM_SECONDS = 30e-6


def source() -> dict[str, Any]:
    return yaml.safe_load(SOURCE.read_text(encoding="utf-8"))


def render(**options: Any) -> dict[str, Any]:
    return nj.render_nccl_manifest(
        source(),
        node=NODE,
        peer=PEER,
        name="gl11-a2z-a1-deadbeef",
        attempt_id="gl11-a2z-a1-deadbeef-a001",
        **options,
    )


def pods(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    replicas = document["spec"]["pytorchReplicaSpecs"]
    return {role: replicas[role]["template"]["spec"] for role in ("Master", "Worker")}


def trainer(pod_spec: dict[str, Any]) -> dict[str, Any]:
    [container] = [c for c in pod_spec["containers"] if c["name"] == "pytorch"]
    return container


def env(container: dict[str, Any]) -> dict[str, str]:
    return {item["name"]: item["value"] for item in container["env"]}


def script_of(document: dict[str, Any], role: str = "Master") -> str:
    command = trainer(pods(document)[role])["command"]
    [script] = HEREDOC.findall(command[-1])
    return script + "\n"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_rendering_keeps_the_efa_job_and_pins_master_and_worker() -> None:
    original = pods(source())
    document = render()
    rendered = pods(document)

    assert document["metadata"]["labels"]["gpu-fault.io/job-id"] == (
        "gl11-a2z-a1-deadbeef"
    )
    assert document["metadata"]["labels"]["gpu-fault.io/attempt-id"] == (
        "gl11-a2z-a1-deadbeef-a001"
    )
    assert rendered["Master"]["nodeName"] == NODE
    assert rendered["Worker"]["nodeName"] == PEER
    for role in ("Master", "Worker"):
        before, after = trainer(original[role]), trainer(rendered[role])
        assert after["image"] == before["image"], role
        assert after["resources"] == before["resources"], role
        assert after["resources"]["requests"]["vpc.amazonaws.com/efa"] == "16", role
        assert after["resources"]["limits"]["nvidia.com/gpu"] == "8", role
        assert after["volumeMounts"] == before["volumeMounts"], role
        values = env(after)
        for name in (
            "NCCL_DEBUG",
            "FI_PROVIDER",
            "FI_EFA_USE_DEVICE_RDMA",
            "FI_EFA_FORK_SAFE",
            load.GPU_DUTY_ENV,
        ):
            assert values[name] == env(before)[name], (role, name)
        assert values["FI_PROVIDER"] == "efa", role
        assert (
            values
            | {
                "GFRS_JOB_ID": "gl11-a2z-a1-deadbeef",
                "GFRS_ATTEMPT_ID": "gl11-a2z-a1-deadbeef-a001",
                "GFRS_KEEP_ACTIVE": "1",
                "GFRS_PAUSE_DIR": "/var/run/gfrs-nccl",
                "GFRS_STALL_MODE": "quiet",
                "GFRS_STALL_MAX_SECONDS": str(nj.STALL_MAX_SECONDS),
                "GFRS_PAYLOAD_ELEMENTS": str(nj.PAYLOAD_BYTES // 4),
                "TORCH_NCCL_TRACE_BUFFER_SIZE": "0",
            }
            == values
        ), role
        assert "TORCH_NCCL_DEBUG_INFO_PIPE_FILE" not in values, role
        assert "GFRS_RANK_PROGRESS_DIR" not in values, role
        assert pods(document)[role]["volumes"] == original[role]["volumes"], role


def test_the_command_runs_the_trainer_with_its_hooks_under_torchrun() -> None:
    document = render()
    command = trainer(pods(document)["Master"])["command"]
    script = script_of(document)

    assert command[:2] == ["/bin/bash", "-lc"]
    assert command[-1].startswith(f"cat >{nj.TRAINER_PATH} <<'PY'\n"), (
        "the trainer is written by a quoted heredoc"
    )
    assert "exec torchrun" in command[-1] and "--nproc-per-node=8" in command[-1]
    assert command[-1].rstrip().endswith(nj.TRAINER_PATH), (
        "torchrun runs the rendered trainer"
    )
    assert script == nj.trainer_script()
    assert script_of(document, "Worker") == script
    assert load.GPU_LOAD_SNIPPET in script, "the canonical acceptance GPU load"
    assert script.index(load.SNIPPET_BEGIN) < script.index("import torch\n")
    for hook in (
        "def pause_vote(",
        "def stall(",
        "entered=extra_all_reduce",
        "held=before_extra_all_reduce",
        'file_state("release")',
        "deadline = held_since + STALL_MAX_SECONDS",
        "timeout=timedelta(seconds=STALL_MAX_SECONDS + 900)",
        "dist.barrier()",
        "SUCCESS rank=",
    ):
        assert hook in script, hook
    compile(script, "trainer", "exec")


def test_flight_recorder_and_phase_mode_render_their_environment() -> None:
    on = env(trainer(pods(render(flight_recorder=True))["Worker"]))
    assert {key: on[key] for key in nj.FLIGHT_RECORDER_ENV} == nj.FLIGHT_RECORDER_ENV
    phase = env(trainer(pods(render(stall_mode="phase"))["Master"]))
    assert phase["GFRS_STALL_MODE"] == "phase"
    assert phase["GFRS_RANK_PROGRESS_DIR"] == nj.RANK_PROGRESS_DIR
    low = env(trainer(pods(render(keep_active=False, stall_mode="write"))["Master"]))
    assert low["GFRS_KEEP_ACTIVE"] == "0" and low["GFRS_STALL_MODE"] == "write"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c["env"].remove({"name": "FI_PROVIDER", "value": "efa"}), "EFA"),
        (lambda c: c["resources"]["requests"].pop("vpc.amazonaws.com/efa"), "EFA"),
        (lambda c: c["resources"]["limits"].pop("nvidia.com/gpu"), "nvidia.com/gpu"),
        (lambda c: c.update(name="other"), "exactly one pytorch"),
    ],
    ids=["no-fi-provider", "no-efa-request", "no-gpu-limit", "no-trainer"],
)
def test_a_source_that_is_not_an_efa_two_node_job_is_refused(
    change: Callable[[dict[str, Any]], Any], message: str
) -> None:
    document = source()
    change(trainer(pods(document)["Worker"]))
    with pytest.raises(ValueError, match=message):
        nj.render_nccl_manifest(
            document, node=NODE, peer=PEER, name="n", attempt_id="n-a001"
        )


def test_an_unknown_mode_a_missing_worker_or_an_unbounded_watchdog_is_refused() -> None:
    with pytest.raises(ValueError, match="stall mode"):
        render(stall_mode="loud")
    with pytest.raises(ValueError, match="watchdog"):
        render(stall_max_seconds=0)
    document = source()
    document["spec"]["pytorchReplicaSpecs"].pop("Worker")
    with pytest.raises(ValueError, match="Worker"):
        nj.render_nccl_manifest(
            document, node=NODE, peer=PEER, name="n", attempt_id="n-a001"
        )


def test_the_payload_meets_the_sizing_from_the_product_thresholds() -> None:
    assert nj.PAYLOAD_BYTES == nj.PAYLOAD_ELEMENTS * 4
    # 2x (tree) to 3.75x (ring) of the payload per node per 2 s step stays
    # inside (1, 4) MiB/s: active without a sticky start-up SPIKE.
    low = 2 * nj.PAYLOAD_BYTES / nj.STEP_SECONDS
    high = 3.75 * nj.PAYLOAD_BYTES / nj.STEP_SECONDS
    ceiling = nj.SPIKE_RATIO * nj.MIN_ACTIVE_BPS
    assert nj.MIN_ACTIVE_BPS < low < high < ceiling, (low, high)
    assert nj.PG_TIMEOUT_MARGIN_SECONDS > 0 and nj.STALL_MAX_SECONDS <= 7200


# --------------------------------------------------------------------------- #
# The trainer, executed
# --------------------------------------------------------------------------- #
class Exited(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


class Reexec(Exception):
    pass


@dataclass
class Runtime:
    rank: int
    now: float = 1_000.0
    #: Votes the *other* ranks contribute to the n-th heartbeat all-reduce.
    votes: dict[int, float] = field(default_factory=dict)
    timers: list[tuple[float, Callable[[], None]]] = field(default_factory=list)
    calls: list[tuple[Any, ...]] = field(default_factory=list)
    printed: list[tuple[float, str]] = field(default_factory=list)
    flags: list[Any] = field(default_factory=list)
    heartbeats: int = 0
    matmuls: list[float] = field(default_factory=list)
    opened: list[tuple[float, str, str]] = field(default_factory=list)

    def at(self, moment: float, action: Callable[[], None]) -> None:
        self.timers.append((moment, action))
        self.timers.sort(key=lambda item: item[0])

    def advance(self, seconds: float) -> None:
        target = self.now + max(0.0, seconds)
        while self.timers and self.timers[0][0] <= target:
            moment, action = self.timers.pop(0)
            self.now = max(self.now, moment)
            action()
        self.now = target

    def lines(self, marker: str) -> list[str]:
        return [line for _when, line in self.printed if line.startswith(marker)]


class Flag:
    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.value = False
        runtime.flags.append(self)

    def set(self) -> None:
        self.value = True

    def is_set(self) -> bool:
        return self.value

    def wait(self, timeout: float) -> bool:
        if not self.value:
            self.runtime.advance(timeout)
        return self.value


class Tensor:
    def __init__(self, values: list[float] | None, size: int, dtype: Any = None):
        self.values = values
        self.size = size
        self.dtype = dtype

    def item(self) -> float:
        assert self.values, "item() of a payload tensor"
        return self.values[0]

    def tolist(self) -> list[float]:
        return list(self.values or [])


def fakes(runtime: Runtime) -> dict[str, Any]:
    def all_reduce(tensor: Tensor) -> None:
        runtime.calls.append(("all_reduce", tensor.size, tensor.dtype, runtime.now))
        if tensor.size == 2:
            assert tensor.values is not None, "the heartbeat carries its values"
            other = runtime.votes.get(runtime.heartbeats, 0.0)
            runtime.heartbeats += 1
            tensor.values = [WORLD * (WORLD + 1) / 2, tensor.values[1] + other]
        elif tensor.size == 1:
            tensor.values = [float(WORLD)]

    def mm(first: Tensor, second: Tensor, *, out: Tensor) -> None:
        runtime.matmuls.append(runtime.now)
        runtime.advance(MM_SECONDS)

    distributed = SimpleNamespace(
        init_process_group=lambda backend, timeout: runtime.calls.append(
            ("init", backend, timeout)
        ),
        get_rank=lambda: runtime.rank,
        get_world_size=lambda: WORLD,
        all_reduce=all_reduce,
        barrier=lambda: runtime.calls.append(("barrier",)),
        destroy_process_group=lambda: runtime.calls.append(("destroy",)),
    )
    torch = SimpleNamespace(
        float16="float16",
        float32="float32",
        device=lambda kind, index: f"{kind}:{index}",
        tensor=lambda values, device: Tensor(list(values), len(values)),
        zeros=lambda count, device, dtype: Tensor(None, count, dtype),
        ones=lambda count, device: Tensor([1.0] * count, count),
        randn=lambda shape, device, dtype: Tensor(None, shape[0] * shape[1], dtype),
        mm=mm,
        cuda=SimpleNamespace(
            set_device=lambda index: runtime.calls.append(("set_device", index)),
            synchronize=lambda device=None: None,
        ),
        distributed=distributed,
    )
    return {
        "signal": SimpleNamespace(
            SIGTERM=15,
            SIG_BLOCK=0,
            sigwait=lambda signals: 15,
            pthread_sigmask=lambda how, signals: None,
        ),
        "socket": SimpleNamespace(gethostname=lambda: f"node-{runtime.rank // 8}"),
        "threading": SimpleNamespace(
            Event=lambda: Flag(runtime),
            Thread=lambda target, daemon: SimpleNamespace(start=lambda: None),
        ),
        "time": SimpleNamespace(
            monotonic=lambda: runtime.now,
            time=lambda: 1_700_000_000.0 + runtime.now,
            sleep=runtime.advance,
        ),
        "torch": torch,
        "torch.distributed": torch,
    }


def run(script: str, runtime: Runtime) -> dict[str, Any]:
    modules = fakes(runtime)
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in modules:
            return modules[name]
        return real_import(name, *args, **kwargs)

    def fake_print(*values: Any, sep: str = " ", **_: Any) -> None:
        runtime.printed.append((runtime.now, sep.join(str(item) for item in values)))

    def tracking_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        runtime.opened.append((runtime.now, str(file), mode))
        return builtins.open(file, mode, *args, **kwargs)

    scope: dict[str, Any] = {
        "__builtins__": {
            **vars(builtins),
            "__import__": fake_import,
            "print": fake_print,
            "open": tracking_open,
        },
        "__name__": "__main__",
    }
    exec(compile(script, "<nccl trainer>", "exec"), scope)
    return scope


@pytest.fixture
def trainer_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[..., dict[str, Any]]:
    """Point the trainer at a private pause dir and scratch dir; fake the exits."""

    def exit_now(code: int) -> None:
        raise Exited(code)

    def reexec(path: str, argv: list[str], environment: dict[str, str]) -> None:
        raise Reexec(path, argv, environment)

    monkeypatch.setattr(os, "_exit", exit_now)
    monkeypatch.setattr(os, "execve", reexec)
    monkeypatch.delenv("GPU_FAULT_RANK_PROGRESS_FILE", raising=False)
    monkeypatch.delenv("GFRS_RANK_PROGRESS_DIR", raising=False)
    (tmp_path / "pause").mkdir()
    (tmp_path / "shm").mkdir()

    def configure(rank: int, **values: Any) -> dict[str, Any]:
        settings = {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank % 8),
            load.GPU_DUTY_ENV: "0",
            "GFRS_PAUSE_DIR": str(tmp_path / "pause"),
            "GFRS_SCRATCH_DIR": str(tmp_path / "shm"),
            "GFRS_STEPS": "4",
            "GFRS_STALL_MAX_SECONDS": "50",
            "GFRS_STALL_MODE": "quiet",
            "GFRS_KEEP_ACTIVE": "1",
            "GFRS_PAYLOAD_ELEMENTS": str(nj.PAYLOAD_ELEMENTS),
            "GFRS_WRITE_BYTES_PER_SECOND": "4096",
            "GFRS_JOB_ID": "gl11-pp-a1-deadbeef",
            "GFRS_ATTEMPT_ID": "gl11-pp-a1-deadbeef-a001",
            **{key: str(value) for key, value in values.items()},
        }
        for key, value in settings.items():
            monkeypatch.setenv(key, value)
        return {"pause": tmp_path / "pause", "shm": tmp_path / "shm", **settings}

    return configure


def stall_window(runtime: Runtime) -> tuple[float, float]:
    """When the held rank stopped and when it joined the extra collective."""

    [stall] = [when for when, line in runtime.printed if line.startswith("STALL ")]
    [joined] = [
        call[3] for call in runtime.calls if call[0] == "all_reduce" and call[1] == 1
    ]
    return stall, joined


def test_rank_zero_takes_the_pause_once_and_enters_an_extra_collective(
    trainer_env: Callable[..., dict[str, Any]],
) -> None:
    paths = trainer_env(0)
    runtime = Runtime(rank=0)
    runtime.at(1_001.0, lambda: (paths["pause"] / "pause").touch())
    run(script_of(render()), runtime)

    init = [call for call in runtime.calls if call[0] == "init"]
    assert init == [("init", "nccl", timedelta(seconds=50 + 900))]
    sizes = [call[1] for call in runtime.calls if call[0] == "all_reduce"]
    payload = nj.PAYLOAD_ELEMENTS
    # Steps 0-3: payload + heartbeat; the vote of step 1 stalls after step 1.
    assert sizes == [payload, 2, payload, 2, 1, payload, 2, payload, 2], sizes
    assert all(
        call[2] == "float32"
        for call in runtime.calls
        if call[0] == "all_reduce" and call[1] == payload
    ), "the payload is fp32, PAYLOAD_BYTES per step"
    assert not (paths["pause"] / "pause").exists(), "the pause file is consumed"
    assert (paths["pause"] / "pause.taken").exists(), (
        "rank 0 renames the pause file it took"
    )
    heartbeats = runtime.lines("HEARTBEAT")
    assert len(heartbeats) == 4, runtime.printed
    assert all(heartbeat_healthy(line, world_size=WORLD) for line in heartbeats), (
        heartbeats
    )
    [entered] = runtime.lines("STALL ")
    assert "step=1 entered=extra_all_reduce watchdog_seconds=50" in entered
    [resumed] = runtime.lines("RESUMED")
    assert "extra_all_reduce=16.0" in resumed
    assert runtime.lines("SUCCESS") == [f"SUCCESS rank=0/{WORLD} node=node-0"], (
        runtime.printed
    )
    assert runtime.calls[-2:] == [("barrier",), ("destroy",)]


def test_a_held_rank_keeps_its_gpu_busy_and_joins_on_a_new_release_file(
    trainer_env: Callable[..., dict[str, Any]],
) -> None:
    paths = trainer_env(8, **{load.GPU_DUTY_ENV: "30"}, GFRS_STEPS="2")
    release = paths["pause"] / "release"
    release.touch()  # a stale release file from an earlier stall is ignored
    os.utime(release, ns=(1, 1))
    runtime = Runtime(rank=8, votes={0: 1.0})
    runtime.at(1_012.0, lambda: os.utime(release, ns=(2, 2)))
    run(script_of(render()), runtime)

    stalled, joined = stall_window(runtime)
    assert 1_011.0 <= joined <= 1_013.5, (stalled, joined)
    [resumed] = runtime.lines("RESUMED")
    assert "reason=release" in resumed, resumed
    busy = [moment for moment in runtime.matmuls if stalled < moment < joined]
    assert busy, "a held rank keeps the acceptance duty on its GPU"
    assert len(runtime.lines("HEARTBEAT")) == 2 and runtime.lines("SUCCESS")


def test_without_a_release_the_bounded_watchdog_ends_the_hold(
    trainer_env: Callable[..., dict[str, Any]],
) -> None:
    trainer_env(9)
    runtime = Runtime(rank=9, votes={1: 1.0})
    run(script_of(render()), runtime)

    stalled, joined = stall_window(runtime)
    assert 50.0 <= joined - stalled <= 51.5, (stalled, joined)
    [held] = runtime.lines("STALL ")
    assert "held=before_extra_all_reduce mode=quiet watchdog_seconds=50" in held
    [resumed] = runtime.lines("RESUMED")
    assert "reason=watchdog held_seconds=50" in resumed, resumed


def test_sigterm_ends_a_held_rank_with_143(
    trainer_env: Callable[..., dict[str, Any]],
) -> None:
    trainer_env(4)
    runtime = Runtime(rank=4, votes={0: 1.0})
    runtime.at(1_020.0, lambda: runtime.flags[0].set())
    with pytest.raises(Exited) as caught:
        run(script_of(render()), runtime)
    assert caught.value.code == load.SIGTERM_EXIT_CODE
    assert runtime.lines("STALL ") and not runtime.lines("RESUMED")


def test_write_mode_rewrites_a_checkpoint_while_held(
    trainer_env: Callable[..., dict[str, Any]],
) -> None:
    paths = trainer_env(2, GFRS_STALL_MODE="write")
    runtime = Runtime(rank=2, votes={0: 1.0})
    checkpoint = paths["shm"] / "gfrs-checkpoint-2.bin"
    run(script_of(render(stall_mode="write")), runtime)

    stalled, joined = stall_window(runtime)
    writes = [
        when
        for when, name, mode in runtime.opened
        if name == str(checkpoint) and mode == "wb"
    ]
    assert 49 <= len(writes) <= 51, "one 4 KiB rewrite per one-second poll"
    assert all(stalled <= when < joined for when in writes), (stalled, joined)
    assert checkpoint.stat().st_size == 4096
    assert not runtime.lines("STALL-ACTIVITY-ERROR"), runtime.printed


def test_a_failing_side_activity_is_reported_once_and_never_ends_the_job(
    trainer_env: Callable[..., dict[str, Any]], tmp_path: Path
) -> None:
    trainer_env(2, GFRS_STALL_MODE="write", GFRS_SCRATCH_DIR=str(tmp_path / "missing"))
    runtime = Runtime(rank=2, votes={0: 1.0})
    run(script_of(render(stall_mode="write")), runtime)

    [error] = runtime.lines("STALL-ACTIVITY-ERROR")
    assert "mode=write error=FileNotFoundError" in error
    assert runtime.lines("RESUMED") and runtime.lines("SUCCESS")


def test_phase_mode_gives_each_rank_its_own_progress_file(
    trainer_env: Callable[..., dict[str, Any]], tmp_path: Path
) -> None:
    trainer_env(11, GFRS_STALL_MODE="phase", GFRS_RANK_PROGRESS_DIR=str(tmp_path))
    with pytest.raises(Reexec) as caught:
        run(script_of(render(stall_mode="phase")), Runtime(rank=11))
    path, argv, environment = caught.value.args
    assert argv[:2] == [path, "-u"]
    assert environment["GPU_FAULT_RANK_PROGRESS_FILE"] == str(tmp_path / "rank-11.json")


def test_phase_mode_advances_a_valid_rank_progress_snapshot_while_held(
    trainer_env: Callable[..., dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = tmp_path / "progress" / "rank-11.json"
    trainer_env(11, GFRS_STALL_MODE="phase", GFRS_STALL_MAX_SECONDS="70")
    monkeypatch.setenv("GFRS_RANK_PROGRESS_DIR", str(tmp_path / "progress"))
    monkeypatch.setenv("GPU_FAULT_RANK_PROGRESS_FILE", str(progress))
    seen: list[dict[str, Any]] = []
    runtime = Runtime(rank=11, votes={0: 1.0})
    for second in (1, 10, 40, 65):
        runtime.at(
            1_000.0 + second,
            lambda: seen.append(json.loads(progress.read_text(encoding="utf-8"))),
        )
    run(script_of(render(stall_mode="phase")), runtime)

    for document in seen:
        snapshot = RankProgressSnapshot.model_validate(document)
        assert (snapshot.job_id, snapshot.attempt_id, snapshot.rank) == (
            "gl11-pp-a1-deadbeef",
            "gl11-pp-a1-deadbeef-a001",
            11,
        )
        assert snapshot.world_size == WORLD and snapshot.process_pid == os.getpid()
    phases = [item["phase"] for item in seen]
    steps = [item["step"] for item in seen]
    assert phases == ["TRAINING", "CHECKPOINT_SAVE", "TRAINING", "CHECKPOINT_SAVE"]
    assert steps == sorted(steps) and len(set(steps)) == 4, steps


def test_the_always_low_job_only_all_reduces_its_heartbeat(
    trainer_env: Callable[..., dict[str, Any]],
) -> None:
    trainer_env(0, GFRS_KEEP_ACTIVE="0")
    runtime = Runtime(rank=0)
    run(script_of(render(keep_active=False)), runtime)

    sizes = {call[1] for call in runtime.calls if call[0] == "all_reduce"}
    assert sizes == {2}, sizes
    assert len(runtime.lines("HEARTBEAT")) == 4


def test_an_invalid_mode_stops_the_trainer_before_any_collective(
    trainer_env: Callable[..., dict[str, Any]],
) -> None:
    trainer_env(0, GFRS_STALL_MODE="loud")
    runtime = Runtime(rank=0)
    with pytest.raises(SystemExit, match="GFRS_STALL_MODE"):
        run(script_of(render()), runtime)
    assert runtime.calls == []


# --------------------------------------------------------------------------- #
# The managed job's kubectl surface
# --------------------------------------------------------------------------- #
PODS = [
    {"name": "job-master-0", "uid": "u-master", "node": NODE},
    {"name": "job-worker-0", "uid": "u-worker", "node": PEER},
]


class FakeWorkload:
    current: list[dict[str, Any]] = []

    def __init__(self, regional: Any, settings: Any, *, state_path: Path) -> None:
        self.settings = settings

    def submit(self) -> dict[str, Any]:
        return {}

    def wait_running(self, timeout_seconds: int) -> dict[str, Any]:
        return {"pods": [dict(item) for item in PODS]}

    def pods(self) -> list[dict[str, Any]]:
        return [dict(item) for item in FakeWorkload.current]

    def pods_healthy(self, pods: list[dict[str, Any]]) -> bool:
        return len(pods) == 2

    def delete(self) -> None:
        pass


class FakePrewarm:
    def __init__(self, regional: Any, **_: Any) -> None:
        pass

    def create(self, nodes: list[str]) -> dict[str, Any]:
        return {}

    def cleanup(self) -> dict[str, bool]:
        return {}


class KubectlRegional(FakeRegional):
    def __init__(self) -> None:
        super().__init__()
        self.commands: list[tuple[str, ...]] = []

    def kubectl(self, plane: str, *arguments: str, **_: Any) -> str:
        self.commands.append(arguments)
        if arguments[0] == "logs":
            return (
                "NCCL INFO x\nHEARTBEAT rank=0 step=3\n[default1]:STALL step=3 held=x\n"
                "RESUMED step=3 reason=release\n"
            )
        return ""


def test_the_job_pauses_the_master_releases_every_pod_and_checks_liveness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(mwf, "ManagedWorkloadFixture", FakeWorkload)
    monkeypatch.setattr(mwf, "ImagePrewarmFixture", FakePrewarm)
    spec = next(item for item in nc.VARIANTS if item.variant == "active_to_zero")
    regional = KubectlRegional()
    base = make_settings(spec, node=NODE, peer=PEER)
    (tmp_path / "site.yaml").write_text("site: test\n")
    settings = type(base)(**{**base.__dict__, "site_file": tmp_path / "site.yaml"})
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    FakeWorkload.current = [dict(item) for item in PODS]
    job = nj.ManagedNcclJob(ctx, name="gl11-a2z-a1-deadbeef", stall_mode="write")
    rendered = ctx.case_dir / "gl11-a2z-a1-deadbeef.source.yaml"
    document = yaml.safe_load(rendered.read_text(encoding="utf-8"))
    assert pods(document)["Worker"]["nodeName"] == PEER
    assert env(trainer(pods(document)["Master"]))["GFRS_STALL_MODE"] == "write"
    assert rendered.stat().st_mode & 0o777 == 0o600
    job.start()

    job.stall()
    assert regional.commands == [
        (
            "exec",
            "job-master-0",
            "-c",
            "pytorch",
            "--",
            "/bin/sh",
            "-c",
            "mkdir -p /var/run/gfrs-nccl && touch /var/run/gfrs-nccl/pause",
        )
    ]
    regional.commands.clear()
    assert job.release()["released"] == ["job-master-0", "job-worker-0"]
    assert [command[1] for command in regional.commands] == [
        "job-master-0",
        "job-worker-0",
    ]
    assert all(
        command[-1].endswith("touch /var/run/gfrs-nccl/release")
        for command in regional.commands
    ), regional.commands
    assert job.alive()["alive"] is True
    FakeWorkload.current[1]["uid"] = "u-worker-restarted"
    assert job.alive()["alive"] is False, "a restarted Pod is not the stalled job"
    assert job.trainer_events()["job-master-0"] == [
        "[default1]:STALL step=3 held=x",
        "RESUMED step=3 reason=release",
    ], "the stall lines, prefixed or not, and nothing else"
    assert job.identity()["stall_mode"] == "write"
    assert job.identity()["payload_bytes"] == nj.PAYLOAD_BYTES


# --------------------------------------------------------------------------- #
# The trainer's own flight-recorder files (where PyTorch left its dumps)
# --------------------------------------------------------------------------- #
FR_LISTING = (
    "HOME|/root\n"
    "FILE|/tmp/torch_nccl_debug_pipe_0.pipe|fifo|0|1792940750\n"
    "FILE|/tmp/torch_nccl_debug_pipe_1.pipe|fifo|0|1792940750\n"
    "FILE|/root/.cache/torch/comm_lib_trace_rank_0|regular file|48213|1792941153\n"
    "FILE|/root/.cache/torch/comm_lib_trace_rank_1|regular file|48210|1792941153\n"
    "a line without the listing's shape\n"
    "FILE|/root/.cache/torch/comm_lib_trace_rank_2|regular file|not-a-size|1\n"
)
FR_DEPRECATION = (
    "[rank0]:[W1025 15:14:36.364962229 Utils.hpp:137] Warning: Environment variable "
    "TORCH_NCCL_TRACE_BUFFER_SIZE is deprecated; use TORCH_FR_BUFFER_SIZE instead"
)
FR_DUMP_REQUEST = (
    "[rank0]:[E1025 15:05:53.400000000 ProcessGroupNCCL.cpp:1575] [PG ID 0 PG GUID "
    "0(default_pg) Rank 0] ProcessGroupNCCL preparing to dump debug info. Include "
    "stack trace: 1"
)
FR_LOG = (
    "HEARTBEAT rank=0 step=3\n"
    f"{FR_DEPRECATION}\n"
    f"{FR_DUMP_REQUEST}\n"
    "STALL step=106 held=before_extra_all_reduce mode=quiet rank=1\n"
)


def _fr_pod(
    home: str | None, *dumps: str, fifos: int = 2, requests: int = 0
) -> dict[str, Any]:
    files: list[dict[str, Any]] = [
        {
            "path": f"/tmp/torch_nccl_debug_pipe_{index}.pipe",
            "type": "fifo",
            "size": 0,
            "mtime": 1,
        }
        for index in range(fifos)
    ]
    files += [
        {"path": path, "type": "regular file", "size": 48213, "mtime": 2}
        for path in dumps
    ]
    return {
        "home": home,
        "files": files,
        "log_lines": [FR_DUMP_REQUEST] * requests,
        "error": None,
    }


def test_parse_fr_forensics_reads_the_stat_lines_and_the_fr_log_lines_only() -> None:
    report = nj.parse_fr_forensics(FR_LISTING, FR_LOG)
    assert report["home"] == "/root"
    assert report["files"] == [
        {
            "path": "/tmp/torch_nccl_debug_pipe_0.pipe",
            "type": "fifo",
            "size": 0,
            "mtime": 1792940750,
        },
        {
            "path": "/tmp/torch_nccl_debug_pipe_1.pipe",
            "type": "fifo",
            "size": 0,
            "mtime": 1792940750,
        },
        {
            "path": "/root/.cache/torch/comm_lib_trace_rank_0",
            "type": "regular file",
            "size": 48213,
            "mtime": 1792941153,
        },
        {
            "path": "/root/.cache/torch/comm_lib_trace_rank_1",
            "type": "regular file",
            "size": 48210,
            "mtime": 1792941153,
        },
    ], "malformed stat lines are dropped, never guessed"
    assert report["log_lines"] == [FR_DEPRECATION, FR_DUMP_REQUEST], (
        "only PyTorch's own flight-recorder lines, not the trainer's"
    )
    assert nj.parse_fr_forensics("", "") == {"home": None, "files": [], "log_lines": []}
    assert nj.parse_fr_forensics("HOME|\n", "")["home"] is None, (
        "an empty HOME is unset"
    )


def test_the_forensics_command_stats_every_known_dump_location_and_the_pipes() -> None:
    command = nj.FR_FORENSICS_COMMAND
    assert f"{nj.LEGACY_DUMP_BASE}*" in command, "the product's pre-2.9 default"
    assert f'"${{HOME:-/tmp}}"{nj.TORCH29_DUMP_SUFFIX}*' in command, (
        "PyTorch 2.9's default, resolved inside the container like PyTorch does"
    )
    assert f"{nj.FLIGHT_RECORDER_ENV['TORCH_NCCL_DEBUG_INFO_PIPE_FILE']}*" in command
    assert 'echo "HOME|${HOME:-}"' in command
    assert "stat -c" in command and command.endswith("; true"), (
        "a location with no file is not a failed exec"
    )
    assert not any(verb in command for verb in ("rm ", "mv ", "touch ", ">", "kill")), (
        "the forensics only read the container"
    )


class ForensicRegional(KubectlRegional):
    exec_error: Exception | None = None

    def kubectl(self, plane: str, *arguments: str, **_: Any) -> str:
        self.commands.append(arguments)
        if arguments[0] == "logs":
            return FR_LOG
        if arguments[0] == "exec":
            if self.exec_error is not None:
                raise self.exec_error
            return FR_LISTING
        return ""


def _managed_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, regional: KubectlRegional
) -> nj.ManagedNcclJob:
    monkeypatch.setattr(mwf, "ManagedWorkloadFixture", FakeWorkload)
    monkeypatch.setattr(mwf, "ImagePrewarmFixture", FakePrewarm)
    spec = next(item for item in nc.VARIANTS if item.variant == "fr_enabled")
    base = make_settings(spec, node=NODE, peer=PEER)
    (tmp_path / "site.yaml").write_text("site: test\n")
    settings = type(base)(**{**base.__dict__, "site_file": tmp_path / "site.yaml"})
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    FakeWorkload.current = [dict(item) for item in PODS]
    job = nj.ManagedNcclJob(ctx, name="gl11-fron-a1-deadbeef", flight_recorder=True)
    job.start()
    regional.commands.clear()
    return job


def test_flight_recorder_forensics_stats_and_reads_the_log_of_every_pod(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    regional = ForensicRegional()
    job = _managed_job(monkeypatch, tmp_path, regional)
    report = job.flight_recorder_forensics()
    execs = [command for command in regional.commands if command[0] == "exec"]
    assert [command[1] for command in execs] == ["job-master-0", "job-worker-0"]
    assert all(
        command[2:7] == ("-c", "pytorch", "--", "/bin/sh", "-c") for command in execs
    ), "the listing runs in the trainer container, like the stall and release"
    assert all(command[-1] == nj.FR_FORENSICS_COMMAND for command in execs), execs
    logs = [command for command in regional.commands if command[0] == "logs"]
    assert [command[1] for command in logs] == ["job-master-0", "job-worker-0"]
    assert all(
        ("-c", "pytorch", f"--tail={nj.FR_LOG_TAIL}") == command[2:] for command in logs
    ), "the trainer container's log, deep enough for the dump request lines"
    assert set(report) == {"job-master-0", "job-worker-0"}
    master = report["job-master-0"]
    assert master["home"] == "/root" and master["error"] is None
    assert [
        item["path"] for item in master["files"] if item["type"] == "regular file"
    ] == [
        "/root/.cache/torch/comm_lib_trace_rank_0",
        "/root/.cache/torch/comm_lib_trace_rank_1",
    ]
    assert master["log_lines"] == [FR_DEPRECATION, FR_DUMP_REQUEST]


def test_a_pod_the_forensics_cannot_read_is_recorded_not_raised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    regional = ForensicRegional()
    regional.exec_error = RuntimeError("kubectl exec: container not found")
    job = _managed_job(monkeypatch, tmp_path, regional)
    report = job.flight_recorder_forensics()
    assert set(report) == {"job-master-0", "job-worker-0"}
    for pod in report.values():
        assert pod["error"] == "RuntimeError: kubectl exec: container not found"
        assert pod["files"] == [] and pod["home"] is None
        assert pod["log_lines"] == [], "no half-read Pod: the error names it"


def test_fr_location_errors_say_where_the_trainer_left_its_dumps() -> None:
    moved = {
        "master": _fr_pod(
            "/root",
            "/root/.cache/torch/comm_lib_trace_rank_0",
            "/root/.cache/torch/comm_lib_trace_rank_1",
            requests=2,
        ),
        "worker": _fr_pod(
            "/root", "/root/.cache/torch/comm_lib_trace_rank_8", requests=1
        ),
    }
    assert nj.fr_location_errors(moved) == [
        "the trainer wrote 3 FR dump file(s) under /root/.cache/torch/comm_lib_trace_rank_*"
        " (the PyTorch >= 2.9 default) and 0 under /tmp/nccl_trace_rank_* (the only"
        " default the product reads); 3 'preparing to dump debug info' line(s)"
    ]
    no_home = {
        "master": _fr_pod(None, "/tmp/.cache/torch/comm_lib_trace_rank_0", requests=1)
    }
    assert nj.fr_location_errors(no_home) == [
        "the trainer wrote 1 FR dump file(s) under /tmp/.cache/torch/comm_lib_trace_rank_*"
        " (the PyTorch >= 2.9 default) and 0 under /tmp/nccl_trace_rank_* (the only"
        " default the product reads); 1 'preparing to dump debug info' line(s)"
    ], "without HOME, PyTorch resolves its default against /tmp"
    legacy = {"master": _fr_pod("/root", "/tmp/nccl_trace_rank_0", requests=1)}
    assert nj.fr_location_errors(legacy) == [
        "the trainer's 1 FR dump file(s) are under /tmp/nccl_trace_rank_* where the"
        " product reads, yet the product read none within its triage budget (compare"
        " the dump mtimes with the triage times); 1 'preparing to dump debug info'"
        " line(s)"
    ]
    nothing = {"master": _fr_pod("/root", requests=2), "worker": _fr_pod("/root")}
    assert nj.fr_location_errors(nothing) == [
        "the trainer wrote no FR dump file under /tmp/nccl_trace_rank_* or"
        " /root/.cache/torch/comm_lib_trace_rank_*; 2 'preparing to dump debug info'"
        " line(s)"
    ]
    unread = {
        "master": {
            "home": None,
            "files": [],
            "log_lines": [],
            "error": "RuntimeError: x",
        },
        "worker": _fr_pod("/root", "/root/.cache/torch/comm_lib_trace_rank_8"),
    }
    assert nj.fr_location_errors(unread) == [
        "master: FR forensics unavailable: RuntimeError: x",
        "the trainer wrote 1 FR dump file(s) under /root/.cache/torch/comm_lib_trace_rank_*"
        " (the PyTorch >= 2.9 default) and 0 under /tmp/nccl_trace_rank_* (the only"
        " default the product reads); 0 'preparing to dump debug info' line(s)",
    ]
    assert nj.fr_location_errors({}) == [], "no Pod read: nothing to attribute"
