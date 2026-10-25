"""The disposable two-node NCCL test job GF-GENERAL-LIVE-011 stalls.

Rendered from the Pod templates of ``two-node-replace-pytorchjob.yaml`` (image,
``FI_PROVIDER=efa``, ``FI_EFA_USE_DEVICE_RDMA``, ``FI_EFA_FORK_SAFE``, the GPU
duty, ``nvidia.com/gpu`` 8 and ``vpc.amazonaws.com/efa`` 16 per Pod), pinned
Master -> ``--fault-node`` and Worker -> ``--node-b``, with its training script
replaced by :func:`trainer_script`. That manifest's own trainer all-reduces one
value every two seconds and has no pause point at all: live, a whole 011 case
moved about 74 KB per node over EFA, so the product never saw an active
baseline and could not suspect a hang.

Traffic sizing, from the product's EFA rules (``host_health.py``,
``efa_traffic_state.py``, ``collectors/host/network.py``): the node rate is the
RX+TX byte delta of every EFA port per collector tick. A delivered sample at or
above ``GPU_FAULT_EFA_TRAFFIC_MIN_ACTIVE_BPS`` (1 MiB/s) marks the attempt
``had_active`` and seeds its baseline; a sample at or below
``GPU_FAULT_EFA_TRAFFIC_ZERO_BPS`` (1 B/s) is zero, ``ZERO_WARNING`` after 60 s
and ``HUNG_SUSPECTED`` after 600 s, past the 120 s startup grace. A sample at
4x the baseline is ``SPIKE`` and a ``SPIKE`` never updates the baseline, so a
first active tick that caught only part of the traffic would seed a low baseline
and leave the attempt in ``SPIKE``. Below 4 MiB/s that cannot happen: a partial
tick at a quarter of the rate or less stays under 1 MiB/s and seeds nothing.
The target is therefore 1-4 MiB/s, about 2 MiB/s. Each step all-reduces
:data:`PAYLOAD_BYTES` (1.5 MiB of fp32) across the 16 ranks; on two nodes that
moves 2x the payload per node over EFA with a tree (out + in) and 3.75x with a
ring (2(n-1)/n each way, n = 16), so the 2.0 s step gives 1.5-2.8 MiB/s per
node. The collective adds well under a millisecond of GPU time per step to the
acceptance duty the source manifest already sets (30 %). The runner proves the
rate from its own counter reads before it stalls the job; it does not trust this
estimate.

The stall. Rank 0 (the Master Pod's local rank 0) takes the runner's pause file
``<pause dir>/pause`` once and votes in the step's heartbeat all-reduce, so all
16 ranks agree on the step. Rank 0 then enters an extra all-reduce no other
rank has issued and blocks in it, its NCCL kernel resident; every other rank
holds before that collective, keeping its GPU at the acceptance duty, and joins
it on a new ``<pause dir>/release`` file in its own Pod or when its bounded
watchdog (:data:`STALL_MAX_SECONDS`) expires. The process group timeout is that
watchdog plus :data:`PG_TIMEOUT_MARGIN_SECONDS`, so nothing times out first. No
collective can complete in between, so no byte crosses EFA while every process
stays alive; once the last held rank joins, the extra collective completes and
the loop resumes. SIGTERM exits 143 through the acceptance GPU load's watcher
whether a rank is held or blocked; after the last step every rank leaves through
a barrier and prints ``SUCCESS``.

Stall modes: ``quiet``; ``write`` (each held rank rewrites a 4 MiB file in
``/dev/shm`` once per second, 4x the collector's 1 MiB/s rank-progress write
threshold, without disk I/O); ``phase`` (each rank keeps a ``rank-progress/v1``
snapshot, the product's application progress protocol, at its own
``GPU_FAULT_RANK_PROGRESS_FILE``; held ranks keep advancing its step and switch
its phase every 30 s). ``keep_active=False`` drops the payload: the always-low
job only all-reduces its heartbeat and never reaches 1 MiB/s. The flight
recorder is on only with ``flight_recorder=True``, with the variables the
operations manual prescribes; otherwise its buffer is 0. A
``general_live_fr_replay.FlightRecorderReplay`` as ``flight_recorder`` keeps
PyTorch's recorder off and runs the test dump responder in each Pod instead
(``probes/general_live_fr_responder.py``, written by the container command next
to the trainer and started before ``exec torchrun``): the product's triage then
reads dumps of that shape through its own FIFO protocol -- live-replay.

The role, thread and process modes (``roles``, ``threads``, ``wchan``, ``cpu``,
``pyspy``) are the ``general_live_nccl_role_trainer`` extension rendered into
the same trainer: each rank declares PP/EP roles and thread roles through its
snapshot's ``diagnostic_context`` (a keeper thread keeps it fresh while the rank
is held or blocked), runs the mode's helper threads from the start and switches
them to their stall behaviour with the stall; ``cpu`` adds an unranked GPU
sharer process and ``pyspy`` a non-Python CUDA holder. The ``ROLES``/``SHARER``
/``HOLDER`` lines they print are the runner's record of what was declared.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.e2e.regional.acceptance_gpu_load import GPU_LOAD_SNIPPET
from scripts.e2e.regional.general_live_fixture import VariantContext, VariantNotRun
from scripts.e2e.regional.general_live_fr_replay import (
    LEDGER_PATH,
    SHAPE_ENV,
    FlightRecorderReplay,
    responder_environment,
    responder_launch,
)
from scripts.e2e.regional.general_live_nccl_role_trainer import (
    ROLE_MODES,
    ROLE_TRAINER,
)

TWO_NODE_MANIFEST = "two-node-replace-pytorchjob.yaml"
TRAINER_PATH = "/tmp/gfrs_nccl_stall.py"
PAUSE_DIR = "/var/run/gfrs-nccl"
RANK_PROGRESS_DIR = "/dev/shm/gfrs-rank-progress"
#: The product defaults the sizing derives from (``host_health.py``): the
#: active threshold ``GPU_FAULT_EFA_TRAFFIC_MIN_ACTIVE_BPS`` and the spike
#: ratio ``GPU_FAULT_EFA_TRAFFIC_SPIKE_RATIO``.
MIN_ACTIVE_BPS = 1048576
SPIKE_RATIO = 4
STEP_SECONDS = 2
#: All-reduced once per step: 1.5 MiB of fp32, sized to 1.5-2.8 MiB/s per node.
PAYLOAD_BYTES = 1536 * 1024
PAYLOAD_ELEMENTS = PAYLOAD_BYTES // 4
#: The in-Pod watchdog: the longest a held rank waits before it releases the
#: stall itself. It outlasts the runner's whole wait (the 930 s drop window plus
#: the hung budget) so the product samples a job that is still stalled.
STALL_MAX_SECONDS = 2700
PG_TIMEOUT_MARGIN_SECONDS = 900
#: ``quiet``/``write``/``phase`` (above) and the role, thread and process modes
#: of ``general_live_nccl_role_trainer`` (``roles``, ``threads``, ``wchan``,
#: ``cpu``, ``pyspy``), which all declare a ``rank-progress/v1`` snapshot.
STALL_MODES = ("quiet", "write", "phase", *ROLE_MODES)
PROGRESS_MODES = ("phase", *ROLE_MODES)
WRITE_BYTES_PER_SECOND = 4 * 1024 * 1024
TRAINER_STEPS = 3600
#: ``docs/部署和运维手册.md``: what a training container injects for NCCL hung
#: forensics (the three-node managed fixture carries the same values). The dump
#: file is pinned under both PyTorch names: 2.9 reads ``TORCH_FR_DUMP_TEMP_FILE``
#: first and would otherwise dump to ``$HOME/.cache/torch/comm_lib_trace_rank_``;
#: 2.8 reads ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE`` only. The buffer keeps its old
#: name because the debug pipe is created only when
#: ``TORCH_NCCL_TRACE_BUFFER_SIZE`` > 0 (ProcessGroupNCCL.hpp ``DumpPipe``).
FLIGHT_RECORDER_ENV = {
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "20000",
    "TORCH_NCCL_DEBUG_INFO_PIPE_FILE": "/tmp/torch_nccl_debug_pipe_",
    "TORCH_FR_DUMP_TEMP_FILE": "/tmp/nccl_trace_rank_",
    "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": "/tmp/nccl_trace_rank_",
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
}
FLIGHT_RECORDER_OFF_ENV = {"TORCH_NCCL_TRACE_BUFFER_SIZE": "0"}
#: Where the trainer's own recorder leaves a dump when nothing pins
#: ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE`` (the manual's injection does not):
#: PyTorch <= 2.8 defaulted to ``/tmp/nccl_trace_rank_<rank>``, the only default
#: the product's reader knows; 2.9.0 (the DLC image) moved it to
#: ``$HOME/.cache/torch/comm_lib_trace_rank_<rank>`` (``c10d::DebugInfoWriter::
#: getWriter``, which prefers ``TORCH_FR_DUMP_TEMP_FILE``). The forensics stat
#: both, and the debug FIFOs, inside each Pod while the stall still holds.
LEGACY_DUMP_BASE = "/tmp/nccl_trace_rank_"
TORCH29_DUMP_SUFFIX = "/.cache/torch/comm_lib_trace_rank_"
FR_FORENSICS_COMMAND = (
    'echo "HOME|${HOME:-}"; '
    f'for f in {LEGACY_DUMP_BASE}* "${{HOME:-/tmp}}"{TORCH29_DUMP_SUFFIX}* '
    f"{FLIGHT_RECORDER_ENV['TORCH_NCCL_DEBUG_INFO_PIPE_FILE']}*; do "
    '[ -e "$f" ] && stat -c "FILE|%n|%F|%s|%Y" "$f"; done; true'
)
#: PyTorch's own account of a dump request and its outcome (c10d log lines in
#: the rank's stderr), and its deprecation notice for the <= 2.8 variable names.
FR_LOG_MARKERS = (
    "preparing to dump debug info",
    "Dump signal received through pipe",
    "Flight Recorder debug info",
    "Flight Recorder trace",
    "is deprecated; use TORCH_FR",
)
FR_LOG_TAIL = 2000
FR_DUMP_REQUEST_MARKER = FR_LOG_MARKERS[0]
#: Variables the renderer owns; any copy in the source manifest is replaced
#: (``TORCH_FR_BUFFER_SIZE`` too: 2.9 would read it before the ``0`` above).
MANAGED_ENV = frozenset(
    {
        *FLIGHT_RECORDER_ENV,
        "TORCH_FR_BUFFER_SIZE",
        SHAPE_ENV,
        "GPU_FAULT_RANK_PROGRESS_FILE",
        "GFRS_ATTEMPT_ID",
        "GFRS_JOB_ID",
        "GFRS_KEEP_ACTIVE",
        "GFRS_PAUSE_DIR",
        "GFRS_PAYLOAD_ELEMENTS",
        "GFRS_RANK_PROGRESS_DIR",
        "GFRS_STALL_MAX_SECONDS",
        "GFRS_STALL_MODE",
        "GFRS_STEPS",
        "GFRS_WRITE_BYTES_PER_SECOND",
    }
)
#: The lines the trainer prints about its stall (``ManagedNcclJob.trainer_events``).
TRAINER_MARKERS = ("STALL step=", "RESUMED step=", "STALL-ACTIVITY-ERROR")
#: Each one is required as the source manifest sets it: a 011 job off EFA is
#: not the experiment.
EFA_ENV = {"FI_PROVIDER": "efa", "FI_EFA_USE_DEVICE_RDMA": "1"}
EFA_RESOURCE = "vpc.amazonaws.com/efa"
GPU_RESOURCE = "nvidia.com/gpu"

TRAINER_HEAD = """\
import json
import os
import signal
import socket
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

# GF-GENERAL-LIVE-011 two-node NCCL stall trainer, rendered by
# scripts/e2e/regional/general_live_nccl_job.py (sizing and stall protocol there).
RANK_PROGRESS_DIR = os.environ.get("GFRS_RANK_PROGRESS_DIR", "")
if RANK_PROGRESS_DIR and not os.environ.get("GPU_FAULT_RANK_PROGRESS_FILE"):
    # The Node Agent reads GPU_FAULT_RANK_PROGRESS_FILE from a rank's exec
    # environment, so every rank re-execs once with its own snapshot path.
    os.execve(
        sys.executable,
        [sys.executable, "-u", *sys.argv],
        {
            **os.environ,
            "GPU_FAULT_RANK_PROGRESS_FILE": os.path.join(
                RANK_PROGRESS_DIR, "rank-" + os.environ["RANK"] + ".json"
            ),
        },
    )

"""

TRAINER_BODY = """
import torch
import torch.distributed as dist

KEEP_ACTIVE = os.environ.get("GFRS_KEEP_ACTIVE", "1") == "1"
PAYLOAD_ELEMENTS = int(os.environ.get("GFRS_PAYLOAD_ELEMENTS", "393216"))
STEPS = int(os.environ.get("GFRS_STEPS", "3600"))
STALL_MODE = os.environ.get("GFRS_STALL_MODE", "quiet")
STALL_MAX_SECONDS = float(os.environ.get("GFRS_STALL_MAX_SECONDS", "2700"))
PAUSE_DIR = os.environ.get("GFRS_PAUSE_DIR", "/var/run/gfrs-nccl")
SCRATCH_DIR = os.environ.get("GFRS_SCRATCH_DIR", "/dev/shm")
WRITE_CHUNK = bytes(int(os.environ.get("GFRS_WRITE_BYTES_PER_SECOND", "4194304")))
JOB_ID = os.environ.get("GFRS_JOB_ID", "")
ATTEMPT_ID = os.environ.get("GFRS_ATTEMPT_ID", "")
STEP_SECONDS = 2.0
POLL_SECONDS = 1.0
PHASE_SECONDS = 30.0
PHASES = ("CHECKPOINT_SAVE", "TRAINING")
if STALL_MODE not in ("quiet", "write", "phase", *ROLE_MODES):
    raise SystemExit(
        f"GFRS_STALL_MODE must be quiet, write, phase or one of {ROLE_MODES}, "
        f"not {STALL_MODE!r}"
    )
if not 0 < STALL_MAX_SECONDS <= 7200:
    raise SystemExit("GFRS_STALL_MAX_SECONDS must be above 0 and at most 7200")

# A rank held at the stall, or blocked in the extra collective, must outlive
# the watchdog: the process group times out only after it, plus a margin.
dist.init_process_group(
    backend="nccl", timeout=timedelta(seconds=STALL_MAX_SECONDS + 900)
)
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
payload = (
    torch.zeros(PAYLOAD_ELEMENTS, device=device, dtype=torch.float32)
    if KEEP_ACTIVE
    else None
)
started = time.time()
activity_failed = False
# A role mode's declarations, helper threads and child processes (the
# extension above); None in the quiet/write/phase modes.
ROLE_FIXTURE = (
    RoleFixture(STALL_MODE, rank, world_size, local_rank, device)
    if STALL_MODE in ROLE_MODES
    else None
)


def say(text):
    print(
        f"{text} rank={rank} node={socket.gethostname()} "
        f"elapsed={time.time() - started:.1f}",
        flush=True,
    )


def file_state(name):
    try:
        status = os.stat(os.path.join(PAUSE_DIR, name))
    except OSError:
        return None
    return (status.st_ino, status.st_mtime_ns)


def pause_vote():
    # Rank 0 takes the runner's pause file once; the vote rides the heartbeat.
    if rank != 0:
        return 0.0
    try:
        os.replace(
            os.path.join(PAUSE_DIR, "pause"), os.path.join(PAUSE_DIR, "pause.taken")
        )
    except OSError:
        return 0.0
    return 1.0


def declare_progress(step, phase, context=None):
    # ``context``: a role mode's diagnostic_context (PP/EP roles, thread roles).
    path = os.environ.get("GPU_FAULT_RANK_PROGRESS_FILE")
    if not path:
        return
    with open("/proc/self/stat", encoding="utf-8") as stream:
        stat = stream.read()
    document = {
        "version": "rank-progress/v1",
        "job_id": JOB_ID,
        "attempt_id": ATTEMPT_ID,
        "rank": rank,
        "world_size": world_size,
        "process_start_ticks": int(stat[stat.rfind(")") + 2 :].split()[19]),
        "process_pid": os.getpid(),
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "phase": phase,
        "process_groups": [],
    }
    if context is not None:
        document["diagnostic_context"] = context
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(document, stream)
    os.replace(temporary, path)


def write_checkpoint():
    target = os.path.join(SCRATCH_DIR, f"gfrs-checkpoint-{rank}.bin")
    with open(target, "wb") as stream:
        stream.write(WRITE_CHUNK)


def guarded(action, *arguments):
    # A failed side activity is reported once and never ends the job.
    global activity_failed
    try:
        action(*arguments)
    except (OSError, ValueError) as exc:
        if not activity_failed:
            activity_failed = True
            say(f"STALL-ACTIVITY-ERROR mode={STALL_MODE} error={type(exc).__name__}")


def keep_working(step, held_seconds):
    # What the app keeps doing while its collective cannot complete.
    if STALL_MODE == "write":
        guarded(write_checkpoint)
    elif STALL_MODE == "phase":
        phase = PHASES[int(held_seconds // PHASE_SECONDS) % len(PHASES)]
        guarded(declare_progress, step + 1 + int(held_seconds), phase)


def stall(step):
    # Rank 0 enters an extra collective; every other rank holds before it and
    # joins on a new release file in its own Pod or at its bounded watchdog.
    extra = torch.ones(1, device=device)
    if rank == 0:
        say(
            f"STALL step={step} entered=extra_all_reduce "
            f"watchdog_seconds={STALL_MAX_SECONDS:.0f}"
        )
        dist.all_reduce(extra)
        say(f"RESUMED step={step} extra_all_reduce={extra.item()}")
        return
    baseline = file_state("release")
    held_since = time.monotonic()
    deadline = held_since + STALL_MAX_SECONDS
    say(
        f"STALL step={step} held=before_extra_all_reduce mode={STALL_MODE} "
        f"watchdog_seconds={STALL_MAX_SECONDS:.0f}"
    )
    reason = "watchdog"
    while time.monotonic() < deadline:
        state = file_state("release")
        if state is not None and state != baseline:
            reason = "release"
            break
        keep_working(step, time.monotonic() - held_since)
        gpu_busy_until(min(time.monotonic() + POLL_SECONDS, deadline), device)
    held = time.monotonic() - held_since
    dist.all_reduce(extra)
    say(
        f"RESUMED step={step} reason={reason} held_seconds={held:.0f} "
        f"extra_all_reduce={extra.item()}"
    )


if ROLE_FIXTURE is not None:
    ROLE_FIXTURE.start()

for step in range(STEPS):
    if payload is not None:
        dist.all_reduce(payload)
    value = torch.tensor([rank + 1.0, pause_vote()], device=device)
    dist.all_reduce(value)
    total, votes = value.tolist()
    expected = world_size * (world_size + 1) / 2
    assert total == expected
    if local_rank == 0:
        print(
            f"HEARTBEAT rank={rank} step={step} "
            f"node={socket.gethostname()} "
            f"all_reduce={total} "
            f"elapsed={time.time() - started:.1f}",
            flush=True,
        )
    if STALL_MODE == "phase":
        guarded(declare_progress, step, "TRAINING")
    if ROLE_FIXTURE is not None:
        ROLE_FIXTURE.step(step)
    gpu_busy_until(time.monotonic() + STEP_SECONDS, device)
    if votes:
        # Every rank's helpers switch to their stall behaviour, the blocked
        # rank 0's included, so a role cohort stays uniform.
        if ROLE_FIXTURE is not None:
            ROLE_FIXTURE.stall_begin()
        stall(step)
        if ROLE_FIXTURE is not None:
            ROLE_FIXTURE.stall_end()

if ROLE_FIXTURE is not None:
    ROLE_FIXTURE.finish()
dist.barrier()
print(
    f"SUCCESS rank={rank}/{world_size} "
    f"node={socket.gethostname()}",
    flush=True,
)
dist.destroy_process_group()
"""


def trainer_script() -> str:
    """The trainer: imports, the canonical acceptance GPU load, the NCCL loop.

    The GPU load snippet comes from ``acceptance_gpu_load`` itself, so this job
    cannot drift from the copies the other training fixtures carry.
    """

    script = TRAINER_HEAD + GPU_LOAD_SNIPPET + ROLE_TRAINER + TRAINER_BODY
    if any(line == "PY" for line in script.splitlines()):
        raise ValueError("the trainer would end its own heredoc")
    return script


def trainer_command(
    nproc_per_node: int, *, replay: FlightRecorderReplay | None = None
) -> list[str]:
    """The container command: the trainer heredoc, (the responder,) torchrun."""

    responder = (
        responder_launch(procs_per_node=nproc_per_node) if replay is not None else ""
    )
    return [
        "/bin/bash",
        "-lc",
        f"cat >{TRAINER_PATH} <<'PY'\n{trainer_script()}PY\n"
        f"{responder}"
        "exec torchrun \\\n"
        '  --nnodes="${WORLD_SIZE}" \\\n'
        f"  --nproc-per-node={nproc_per_node} \\\n"
        '  --node-rank="${RANK}" \\\n'
        '  --master-addr="${MASTER_ADDR}" \\\n'
        '  --master-port="${MASTER_PORT}" \\\n'
        f"  {TRAINER_PATH}\n",
    ]


def render_nccl_manifest(
    document: dict[str, Any],
    *,
    node: str,
    peer: str,
    name: str,
    attempt_id: str,
    keep_active: bool = True,
    stall_mode: str = "quiet",
    flight_recorder: bool | FlightRecorderReplay = False,
    pause_dir: str = PAUSE_DIR,
    stall_max_seconds: int = STALL_MAX_SECONDS,
) -> dict[str, Any]:
    """Turn the two-node replace fixture into the 011 stall job, in place.

    Refuses a source whose Pod templates are not one EFA ``pytorch`` container
    each: rendering the trainer onto anything else would not be this experiment.
    ``flight_recorder`` is PyTorch's own recorder (``True``), none (``False``)
    or a :class:`FlightRecorderReplay` the test responder serves instead.
    """

    if stall_mode not in STALL_MODES:
        raise ValueError(f"stall mode must be one of {STALL_MODES}, not {stall_mode!r}")
    if not 0 < stall_max_seconds <= 7200:
        raise ValueError("the stall watchdog must be above 0 and at most 7200 s")
    replicas = (document.get("spec") or {}).get("pytorchReplicaSpecs") or {}
    document.setdefault("metadata", {}).setdefault("labels", {}).update(
        {"gpu-fault.io/job-id": name, "gpu-fault.io/attempt-id": attempt_id}
    )
    for role, pinned in (("Master", node), ("Worker", peer)):
        replica = replicas.get(role)
        if not isinstance(replica, dict) or replica.get("replicas") != 1:
            raise ValueError(f"the two-node fixture needs exactly one {role}")
        pod_spec = (replica.get("template") or {}).get("spec")
        if not isinstance(pod_spec, dict):
            raise ValueError(f"the {role} replica has no Pod spec")
        pod_spec["nodeName"] = pinned
        trainers = [
            item
            for item in pod_spec.get("containers") or []
            if isinstance(item, dict) and item.get("name") == "pytorch"
        ]
        if len(trainers) != 1:
            raise ValueError(f"the {role} Pod needs exactly one pytorch container")
        container = trainers[0]
        _require_efa(role, container)
        replay = (
            flight_recorder
            if isinstance(flight_recorder, FlightRecorderReplay)
            else None
        )
        gpus = _gpus_per_pod(role, container)
        container["command"] = trainer_command(gpus, replay=replay)
        container.pop("args", None)
        container["env"] = [
            item
            for item in container.get("env") or []
            if not (isinstance(item, dict) and item.get("name") in MANAGED_ENV)
        ] + _env(
            name=name,
            attempt_id=attempt_id,
            keep_active=keep_active,
            stall_mode=stall_mode,
            flight_recorder=flight_recorder,
            pause_dir=pause_dir,
            stall_max_seconds=stall_max_seconds,
            procs_per_node=gpus,
        )
    return document


def _gpus_per_pod(role: str, container: dict[str, Any]) -> int:
    limits = (container.get("resources") or {}).get("limits") or {}
    value = str(limits.get(GPU_RESOURCE, ""))
    if not value.isdigit() or int(value) < 1:
        raise ValueError(f"the {role} trainer has no {GPU_RESOURCE} limit")
    return int(value)


def _require_efa(role: str, container: dict[str, Any]) -> None:
    env = {
        item.get("name"): item.get("value")
        for item in container.get("env") or []
        if isinstance(item, dict)
    }
    wrong = sorted(key for key, value in EFA_ENV.items() if env.get(key) != value)
    resources = container.get("resources") or {}
    counts = [
        (resources.get(kind) or {}).get(EFA_RESOURCE) for kind in ("requests", "limits")
    ]
    if wrong or not all(counts) or len(set(counts)) != 1:
        raise ValueError(
            f"the {role} trainer is not an EFA job (env {wrong}, {EFA_RESOURCE} {counts})"
        )


def _env(
    *,
    name: str,
    attempt_id: str,
    keep_active: bool,
    stall_mode: str,
    flight_recorder: bool | FlightRecorderReplay,
    pause_dir: str,
    stall_max_seconds: int,
    procs_per_node: int,
) -> list[dict[str, str]]:
    if isinstance(flight_recorder, FlightRecorderReplay):
        recorder = responder_environment(flight_recorder, procs_per_node=procs_per_node)
    else:
        recorder = FLIGHT_RECORDER_ENV if flight_recorder else FLIGHT_RECORDER_OFF_ENV
    values = {
        "GFRS_JOB_ID": name,
        "GFRS_ATTEMPT_ID": attempt_id,
        "GFRS_KEEP_ACTIVE": "1" if keep_active else "0",
        "GFRS_PAYLOAD_ELEMENTS": str(PAYLOAD_ELEMENTS),
        "GFRS_PAUSE_DIR": pause_dir,
        "GFRS_STALL_MODE": stall_mode,
        "GFRS_STALL_MAX_SECONDS": str(stall_max_seconds),
        "GFRS_STEPS": str(TRAINER_STEPS),
        "GFRS_WRITE_BYTES_PER_SECOND": str(WRITE_BYTES_PER_SECOND),
        **recorder,
    }
    if stall_mode in PROGRESS_MODES:
        values["GFRS_RANK_PROGRESS_DIR"] = RANK_PROGRESS_DIR
    return [{"name": key, "value": value} for key, value in values.items()]


def parse_fr_forensics(listing: str, log_text: str) -> dict[str, Any]:
    """One Pod's :data:`FR_FORENSICS_COMMAND` listing and log, as a record.

    ``home`` is the container's ``$HOME`` (None when unset: PyTorch then
    resolves its default against ``/tmp``); ``files`` are the ``stat`` rows
    (path, type, size, mtime) of whatever exists at the known dump locations
    and the pipe base; ``log_lines`` are PyTorch's own flight-recorder lines.
    A row that does not parse is dropped, never guessed.
    """

    home: str | None = None
    files: list[dict[str, Any]] = []
    for line in listing.splitlines():
        parts = line.strip().split("|")
        if parts[0] == "HOME" and len(parts) == 2:
            home = parts[1] or None
        elif parts[0] == "FILE" and len(parts) == 5:
            try:
                files.append(
                    {
                        "path": parts[1],
                        "type": parts[2],
                        "size": int(parts[3]),
                        "mtime": int(parts[4]),
                    }
                )
            except ValueError:
                continue
    lines = [
        line
        for line in log_text.splitlines()
        if any(marker in line for marker in FR_LOG_MARKERS)
    ]
    return {"home": home, "files": files, "log_lines": lines[:200]}


def fr_location_errors(forensics: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Where the trainer's recorder left its dumps, when the product paired none.

    The product reads ``TORCH_NCCL_DEBUG_INFO_TEMP_FILE`` or, unset, the PyTorch
    <= 2.8 default :data:`LEGACY_DUMP_BASE`; PyTorch 2.9 moved the default to
    ``$HOME`` + :data:`TORCH29_DUMP_SUFFIX`. Dumps found only there are a dump
    the product never looked at; dumps where it reads are a dump it read too
    early; none anywhere is a trainer that never dumped -- each named with
    PyTorch's own count of dump requests. A Pod the forensics could not read
    is named too; nothing read means nothing to attribute.
    """

    errors: list[str] = []
    legacy = moved = requests = 0
    bases: set[str] = set()
    for pod, report in sorted(forensics.items()):
        if report.get("error"):
            errors.append(f"{pod}: FR forensics unavailable: {report['error']}")
            continue
        base = f"{report.get('home') or '/tmp'}{TORCH29_DUMP_SUFFIX}"
        bases.add(base)
        for entry in report.get("files") or []:
            path = str(entry.get("path") or "")
            if entry.get("type") != "regular file":
                continue
            if path.startswith(LEGACY_DUMP_BASE):
                legacy += 1
            elif path.startswith(base):
                moved += 1
        requests += sum(
            1
            for line in report.get("log_lines") or []
            if FR_DUMP_REQUEST_MARKER in line
        )
    if not bases:
        return errors
    where = ", ".join(f"{base}*" for base in sorted(bases))
    asked = f"{requests} '{FR_DUMP_REQUEST_MARKER}' line(s)"
    if legacy:
        errors.append(
            f"the trainer's {legacy} FR dump file(s) are under {LEGACY_DUMP_BASE}* where"
            " the product reads, yet the product read none within its triage budget"
            f" (compare the dump mtimes with the triage times); {asked}"
        )
    elif moved:
        errors.append(
            f"the trainer wrote {moved} FR dump file(s) under {where} (the PyTorch >="
            f" 2.9 default) and 0 under {LEGACY_DUMP_BASE}* (the only default the"
            f" product reads); {asked}"
        )
    else:
        errors.append(
            f"the trainer wrote no FR dump file under {LEGACY_DUMP_BASE}* or {where};"
            f" {asked}"
        )
    return errors


class ManagedNcclJob:
    """The managed two-node stall job (``ManagedWorkloadFixture`` underneath).

    ``stall`` creates the pause file in the Master Pod; ``release`` creates the
    release file in both Pods (the held ranks of each Pod watch their own);
    ``alive`` says whether both Pods still run as started; ``flight_ledgers``
    reads the test dump responders' ledgers of a replayed recorder. The variant
    tests monkeypatch this class; test_general_live_managed_job_start.py runs
    its start with the real fixtures.
    """

    PAUSE_DIR = PAUSE_DIR
    #: The attempt this job runs as: ``<job id>-a<nnn>``. A subclass running
    #: as 2 is the same job's fresh attempt after the first was deleted
    #: (011 attempt_change); its journals and rendered manifest are its own.
    attempt_number = 1

    def __init__(
        self,
        ctx: VariantContext,
        *,
        name: str,
        flight_recorder: bool | FlightRecorderReplay = False,
        keep_active: bool = True,
        stall_mode: str = "quiet",
    ) -> None:
        from scripts.e2e.regional.managed_workload_fixture import (
            ImagePrewarmFixture,
            ManagedWorkloadFixture,
            ManagedWorkloadSettings,
            name_managed_workload,
        )

        yaml = __import__("yaml")
        if ctx.settings.site_file is None:
            raise VariantNotRun(
                "FIXTURE_NOT_READY", "a managed NCCL job needs --site-file"
            )
        if not ctx.settings.peer_node:
            raise VariantNotRun(
                "FIXTURE_NOT_READY", "a two-node NCCL job needs --node-b"
            )
        source = Path(__file__).with_name("manifests") / "training" / TWO_NODE_MANIFEST
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
        if type(self.attempt_number) is not int or self.attempt_number < 1:
            raise ValueError("the attempt number must be a positive integer")
        self.attempt_id = f"{name}-a{self.attempt_number:03d}"
        stem = name if self.attempt_number == 1 else self.attempt_id
        render_nccl_manifest(
            document,
            node=ctx.settings.node,
            peer=ctx.settings.peer_node,
            name=name,
            attempt_id=self.attempt_id,
            keep_active=keep_active,
            stall_mode=stall_mode,
            flight_recorder=flight_recorder,
        )
        name_managed_workload(document, name)
        rendered = ctx.case_dir / f"{stem}.source.yaml"
        rendered.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        rendered.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        rendered.chmod(0o600)
        self.ctx = ctx
        self.name = name
        self.replay = (
            flight_recorder
            if isinstance(flight_recorder, FlightRecorderReplay)
            else None
        )
        self.flight_recorder = bool(flight_recorder)
        self.keep_active = keep_active
        self.stall_mode = stall_mode
        self.started_pods: dict[str, str] = {}
        self.prewarm = ImagePrewarmFixture(
            ctx.regional,
            case_id=ctx.spec.key,
            run_id=f"{ctx.run_id}-{stem}",
            state_path=ctx.case_dir / f"{stem}.prewarm.json",
        )
        self.fixture = ManagedWorkloadFixture(
            ctx.regional,
            ManagedWorkloadSettings(
                manifest=rendered,
                site_file=ctx.settings.site_file,
                job_id=name,
                attempt_id=self.attempt_id,
                restart_budget=0,
                expected_pods=2,
                expected_gpu_count=16,
            ),
            state_path=ctx.case_dir / f"{stem}.ownership.json",
        )

    def start(self) -> dict[str, Any]:
        self.prewarm.create([self.ctx.settings.node, self.ctx.settings.peer_node])
        self.fixture.submit()
        snapshot = self.fixture.wait_running(timeout_seconds=1200)
        self.started_pods = {
            str(item["name"]): str(item["uid"]) for item in snapshot.get("pods") or []
        }
        return snapshot

    def identity(self) -> dict[str, Any]:
        return {
            "job_id": self.name,
            "attempt_id": self.attempt_id,
            "nodes": [self.ctx.settings.node, self.ctx.settings.peer_node],
            "flight_recorder": self.flight_recorder,
            "flight_recorder_replay": self.replay.name if self.replay else None,
            "keep_active": self.keep_active,
            "stall_mode": self.stall_mode,
            "payload_bytes": PAYLOAD_BYTES if self.keep_active else 0,
            "step_seconds": STEP_SECONDS,
            "stall_watchdog_seconds": STALL_MAX_SECONDS,
        }

    def _exec(self, pod: str, command: str) -> str:
        return self.ctx.regional.kubectl(
            "gpu", "exec", pod, "-c", "pytorch", "--", "/bin/sh", "-c", command
        )

    def stall(self) -> dict[str, Any]:
        """Create the pause file in the Master Pod (global rank 0 takes it)."""

        master = next(
            item
            for item in self.fixture.pods()
            if item["node"] == self.ctx.settings.node
        )
        output = self._exec(
            str(master["name"]),
            f"mkdir -p {PAUSE_DIR} && touch {PAUSE_DIR}/pause",
        )
        return {"pod": master["name"], "output": output}

    def release(self) -> dict[str, Any]:
        """Create the release file in every Pod; each Pod's held ranks join."""

        released = {}
        for item in self.fixture.pods():
            released[str(item["name"])] = self._exec(
                str(item["name"]),
                f"mkdir -p {PAUSE_DIR} && touch {PAUSE_DIR}/release",
            )
        return {"released": sorted(released), "outputs": released}

    def alive(self) -> dict[str, Any]:
        """Both Pods still Running and Ready, and the ones that started."""

        pods = self.fixture.pods()
        current = {str(item["name"]): str(item["uid"]) for item in pods}
        healthy = self.fixture.pods_healthy(pods)
        return {
            "alive": bool(healthy and current == self.started_pods),
            "pods": pods,
            "started": self.started_pods,
        }

    def flight_ledgers(self) -> dict[str, Any]:
        """Each Pod's test dump responder ledger (its own account of requests).

        A Pod whose ledger cannot be read or parsed maps to ``{"error": ...}``:
        the replay oracle then says the fixture did not hold, never the product.
        """

        ledgers: dict[str, Any] = {}
        for item in self.fixture.pods():
            pod = str(item["name"])
            try:
                ledgers[pod] = json.loads(self._exec(pod, f"cat {LEDGER_PATH}"))
            except (OSError, RuntimeError, ValueError) as exc:
                ledgers[pod] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        return ledgers

    def trainer_events(self) -> dict[str, list[str]]:
        """The trainer's STALL / RESUMED lines per Pod (its own account)."""

        events = {}
        for item in self.fixture.pods():
            text = self.ctx.regional.kubectl(
                "gpu", "logs", str(item["name"]), "-c", "pytorch", "--tail=400"
            )
            # A substring, not a prefix: torchrun may prefix a worker's lines.
            events[str(item["name"])] = [
                line
                for line in text.splitlines()
                if any(marker in line for marker in TRAINER_MARKERS)
            ]
        return events

    def flight_recorder_forensics(self) -> dict[str, Any]:
        """Where each Pod's trainer left its recorder files, and what it logged.

        Read while the stall still holds, after the product's collection: a
        ``stat`` of every known dump location and the debug FIFOs plus PyTorch's
        own flight-recorder log lines (:func:`parse_fr_forensics`). Evidence
        that attributes a ``dump_missing`` -- a trainer that never dumped, a
        dump written where the product does not read, a dump read too early --
        never the verdict itself, so a Pod that cannot be read is recorded with
        its error, not raised.
        """

        report: dict[str, Any] = {}
        for item in self.fixture.pods():
            pod = str(item["name"])
            listing = log_text = ""
            error = None
            try:
                listing = self._exec(pod, FR_FORENSICS_COMMAND)
                log_text = self.ctx.regional.kubectl(
                    "gpu", "logs", pod, "-c", "pytorch", f"--tail={FR_LOG_TAIL}"
                )
            except (OSError, RuntimeError) as exc:
                listing = log_text = ""
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
            report[pod] = {**parse_fr_forensics(listing, log_text), "error": error}
        return report

    def delete(self) -> dict[str, Any]:
        from scripts.e2e.regional.managed_workload_fixture import remove_managed_job

        return remove_managed_job(self.fixture, self.prewarm)
