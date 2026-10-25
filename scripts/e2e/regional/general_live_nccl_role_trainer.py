"""The role, thread and process extension of the GF-GENERAL-LIVE-011 trainer.

``general_live_nccl_job.trainer_script`` renders :data:`ROLE_TRAINER` into the
two-node stall trainer between the acceptance GPU load and the NCCL loop. It
adds the stall modes the deep-triage variants of 011 need -- each one the same
EFA stall as ``active_to_zero`` with the ranks declaring and doing something
more -- and the markers the runner reads back from the Pod logs:

``roles`` (role_mismatch)
    Mixed parallelism on every node: local ranks 0-3 declare pipeline stage
    ``pp0`` / expert role ``expert``, local ranks 4-6 declare ``pp1`` /
    ``dense``, local rank 7 declares no diagnostic context at all. The groups
    are node-local (a node's own ranks), because the product compares process
    observations only inside one observation window and the two nodes are
    triaged in different waves. While stalled the ``pp0`` ranks burn host CPU
    (:data:`BURNER_THREADS` CPU matmul threads each) and the ``pp1`` and
    undeclared ranks stay quiet: a triage that compared across roles would
    report the quiet ranks as CPU outliers of the busy cohort.
``threads`` (tid_namespace)
    Every rank declares its training thread, a communication helper and two
    background helpers by container-native thread id; odd ranks start the
    helpers in the reverse order; while stalled a churn thread keeps starting
    and ending short undeclared threads. The product must map the declared ids
    to host thread ids through ``NSpid`` and keep the training thread's role
    and start time whatever the thread list looks like.
``wchan`` (wchan_zero)
    One declared COMMUNICATION helper spins on the CPU while stalled (state R,
    ``/proc/<tid>/wchan`` 0) and one declared BACKGROUND helper sleeps on a
    futex (a real wait channel). ``0`` is not a wait location; CPU cores come
    from tick deltas over the measured interval at ``SC_CLK_TCK``.
``cpu`` (cpu_interval)
    One declared BACKGROUND helper burns the CPU at a 50 % duty while stalled;
    local ranks 6 and 7 also run a second, unranked Python process holding a
    CUDA context on their GPU, so that GPU is shared and the product may not
    claim its utilization for either PID.
``pyspy`` (pyspy_nonzero)
    Local rank 7 compiles and runs a non-Python CUDA context holder (a C
    program using the driver API through ``libcuda.so.1``). The product's
    py-spy legitimately fails on it ("Failed to find python version from target
    process", exit 1): a failed stack sample with no signature, while the
    process's /proc evidence is kept and the Python ranks' stacks still parse.

Every mode writes a ``rank-progress/v1`` snapshot per rank at
``GPU_FAULT_RANK_PROGRESS_FILE`` carrying the declaration as
``diagnostic_context`` (``gpu_fault.rank_progress.DiagnosticContext``); a
keeper thread rewrites it every :data:`KEEPER_SECONDS` so the snapshot stays
inside the product's 30 s freshness window while the rank is held or blocked.
Helper threads exist from the start and only switch behaviour at the stall, so
the declared thread ids never change. Child processes are bounded by the stall
watchdog plus a margin, die with their parent (``PR_SET_PDEATHSIG``) and with
the Pod. The lines the trainer prints (:data:`ROLES_MARKER`,
:data:`SHARER_MARKER`, :data:`HOLDER_MARKER`, :data:`FIXTURE_ERROR_MARKER`) are
``key=value`` tokens; :func:`parse_marker` reads them back and
:func:`expected_context` is the runner's copy of the plan.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

ROLES, THREADS, WCHAN, CPU, PYSPY = "roles", "threads", "wchan", "cpu", "pyspy"
ROLE_MODES = (ROLES, THREADS, WCHAN, CPU, PYSPY)
#: The two-node job runs eight ranks per Pod (``nvidia.com/gpu: 8``).
PROCS_PER_NODE = 8
#: The product reads a snapshot observed at most 30 s ago (``hung_summary_io``).
KEEPER_SECONDS = 10
#: CPU matmul threads of a burning rank: three cores over the quiet ranks'
#: fraction of a core, so a cross-role CPU comparison would be a clear outlier.
BURNER_THREADS = 3
#: What a declaring rank reports as its role, hardware and code version.
ROLE_NAME = "trainer"
HARDWARE_CLASS = "p5en-h200"
CODE_VERSION = "gfrs-nccl-stall/2"
#: How long a child process outlives the stall watchdog at most.
CHILD_MARGIN_SECONDS = 600
#: Base names under the trainer's work dir (``GFRS_ROLE_WORK_DIR``, ``/tmp``).
HOLDER_EXE = "gfrs_cuda_hold"
HOLDER_SOURCE = "gfrs_cuda_hold.c"
STOP_FILE = "gfrs-role-children.stop"
ROLES_MARKER = "ROLES "
SHARER_MARKER = "SHARER "
HOLDER_MARKER = "HOLDER "
FIXTURE_ERROR_MARKER = "GFRS-FIXTURE-ERROR "
MARKERS = (ROLES_MARKER, SHARER_MARKER, HOLDER_MARKER, FIXTURE_ERROR_MARKER)
THREAD_ROLES = ("TRAINING", "COMMUNICATION", "BACKGROUND")

#: Per mode: the declaring groups (local ranks, stage, expert role, whether the
#: held ranks burn CPU), the local ranks that share their GPU with a second
#: process, and the local ranks that run the non-Python CUDA holder.
PLAN: dict[str, Any] = {
    "modes": list(ROLE_MODES),
    "procs_per_node": PROCS_PER_NODE,
    "keeper_seconds": KEEPER_SECONDS,
    "burner_threads": BURNER_THREADS,
    "role": ROLE_NAME,
    "hardware_class": HARDWARE_CLASS,
    "code_version": CODE_VERSION,
    "child_margin_seconds": CHILD_MARGIN_SECONDS,
    "holder_exe": HOLDER_EXE,
    "holder_source": HOLDER_SOURCE,
    "stop_file": STOP_FILE,
    "groups": {
        ROLES: [
            {
                "name": "pp0-expert",
                "locals": [0, 1, 2, 3],
                "stage": "pp0",
                "expert": "expert",
                "burn": True,
            },
            {
                "name": "pp1-dense",
                "locals": [4, 5, 6],
                "stage": "pp1",
                "expert": "dense",
                "burn": False,
            },
        ],
        **{
            mode: [
                {
                    "name": "pp0-dense",
                    "locals": list(range(PROCS_PER_NODE)),
                    "stage": "pp0",
                    "expert": "dense",
                    "burn": False,
                }
            ]
            for mode in (THREADS, WCHAN, CPU, PYSPY)
        },
    },
    "share_gpu": {CPU: [6, 7]},
    "holder": {PYSPY: [7]},
}

#: The ``key=value`` tokens of a marker line (``say`` appends rank/node/elapsed).
TOKEN = re.compile(r"(\w+)=(\S*)")


def group_of(mode: str, local_rank: int) -> dict[str, Any] | None:
    """The declaring group of ``local_rank`` in ``mode`` (None: undeclared)."""

    for group in PLAN["groups"].get(mode, []):
        if local_rank in group["locals"]:
            return dict(group)
    return None


def expected_context(
    mode: str, rank: int, *, world_size: int, procs_per_node: int = PROCS_PER_NODE
) -> dict[str, Any] | None:
    """What a rank's ``diagnostic_context`` declares, less the thread roles.

    ``members`` are the node's own ranks of the group: node ``rank //
    procs_per_node``, in the order the trainer writes them.
    """

    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} is outside a world of {world_size}")
    node = rank // procs_per_node
    group = group_of(mode, rank % procs_per_node)
    if group is None:
        return None
    return {
        "group_name": f"{group['name']}-node{node}",
        "members": [node * procs_per_node + local for local in group["locals"]],
        "role": ROLE_NAME,
        "pipeline_stage": group["stage"],
        "expert_role": group["expert"],
        "hardware_class": HARDWARE_CLASS,
        "code_version": CODE_VERSION,
    }


def burning_locals(mode: str) -> list[int]:
    return sorted(
        local
        for group in PLAN["groups"].get(mode, [])
        if group.get("burn")
        for local in group["locals"]
    )


def sharing_locals(mode: str) -> list[int]:
    return list(PLAN["share_gpu"].get(mode, []))


def holder_locals(mode: str) -> list[int]:
    return list(PLAN["holder"].get(mode, []))


def parse_marker(line: str, marker: str) -> dict[str, str] | None:
    """The tokens of ``line`` after ``marker``, or None when it is not one.

    A substring match: torchrun may prefix a worker's lines. ``roles`` tokens
    (``<tid>:<role>,...``) are returned as written; :func:`declared_roles`
    expands them.
    """

    index = line.find(marker)
    if index < 0:
        return None
    return {key: value for key, value in TOKEN.findall(line[index + len(marker) :])}


def declared_roles(tokens: Mapping[str, str]) -> dict[str, tuple[str, str]]:
    """``{native tid: (role, thread name)}`` of a ROLES line (empty for an
    undeclared rank); the token is ``<tid>:<role>:<thread name>``."""

    raw = tokens.get("roles", "-")
    if raw in {"", "-"}:
        return {}
    roles = {}
    for item in raw.split(","):
        tid, role, name = (item.split(":", 2) + ["", ""])[:3]
        if not tid.isdecimal() or role not in THREAD_ROLES or not name:
            raise ValueError(f"malformed thread role token {item!r}")
        roles[tid] = (role, name)
    return roles


def marker_lines(lines: Iterable[str], marker: str) -> list[dict[str, str]]:
    found = []
    for line in lines:
        tokens = parse_marker(line, marker)
        if tokens is not None:
            found.append(tokens)
    return found


# --------------------------------------------------------------------------- #
# the trainer extension (rendered into the heredoc trainer)
# --------------------------------------------------------------------------- #
ROLE_TRAINER_TEMPLATE = r'''
# ---- GF-GENERAL-LIVE-011 role/thread/process extension ----
# Rendered from scripts/e2e/regional/general_live_nccl_role_trainer.py (the
# plan, the markers and the product fields each mode exercises are there).
import shutil
import subprocess

ROLE_PLAN = json.loads("""__ROLE_PLAN__""")
ROLE_MODES = tuple(ROLE_PLAN["modes"])
ROLE_PROCS_PER_NODE = int(ROLE_PLAN["procs_per_node"])
ROLE_KEEPER_SECONDS = float(ROLE_PLAN["keeper_seconds"])
ROLE_BURNER_THREADS = int(ROLE_PLAN["burner_threads"])
# Where the C source, the holder binary, the children's logs and their stop
# file live; a test points it at a private directory.
ROLE_WORK_DIR = os.environ.get("GFRS_ROLE_WORK_DIR", "/tmp")
ROLE_CHILD_ENV_PREFIXES = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "ROLE_NAME",
    "MASTER_",
    "TORCHELASTIC_",
    "GPU_FAULT_RANK_PROGRESS_FILE",
)
ROLE_HOLDER_SOURCE = r"""
#include <dlfcn.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <time.h>
#include <unistd.h>
/* GF-GENERAL-LIVE-011 pyspy_nonzero: a non-Python process holding a CUDA
 * context on one GPU through the driver API, bounded by a deadline, a stop
 * file and its parent's death. py-spy cannot sample it, which is the point. */
typedef int (*cu_init_t)(unsigned int);
typedef int (*cu_device_get_t)(int *, int);
typedef int (*cu_ctx_create_t)(void **, unsigned int, int);
int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: device seconds stopfile\n"); return 2; }
    prctl(PR_SET_PDEATHSIG, SIGTERM);
    void *lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen libcuda.so.1: %s\n", dlerror()); return 3; }
    cu_init_t cu_init = (cu_init_t)dlsym(lib, "cuInit");
    cu_device_get_t cu_device_get = (cu_device_get_t)dlsym(lib, "cuDeviceGet");
    cu_ctx_create_t cu_ctx_create = (cu_ctx_create_t)dlsym(lib, "cuCtxCreate_v2");
    if (!cu_init || !cu_device_get || !cu_ctx_create) { fprintf(stderr, "dlsym\n"); return 3; }
    int device = 0; void *context = NULL; int status;
    if ((status = cu_init(0)) != 0) { fprintf(stderr, "cuInit %d\n", status); return 4; }
    if ((status = cu_device_get(&device, atoi(argv[1]))) != 0) { fprintf(stderr, "cuDeviceGet %d\n", status); return 4; }
    if ((status = cu_ctx_create(&context, 0, device)) != 0) { fprintf(stderr, "cuCtxCreate %d\n", status); return 4; }
    printf("GFRS-HOLDER-READY pid=%d device=%s\n", (int)getpid(), argv[1]);
    fflush(stdout);
    time_t deadline = time(NULL) + (time_t)atol(argv[2]);
    while (time(NULL) < deadline && access(argv[3], F_OK) != 0) sleep(1);
    return 0;
}
"""
ROLE_SHARER_SOURCE = """
import ctypes, os, sys, time
# GF-GENERAL-LIVE-011 cpu_interval: an unranked second process holding a CUDA
# context on the parent rank's GPU; ends at the deadline, the stop file or the
# parent's death.
ctypes.CDLL(None).prctl(1, 15)
import torch
torch.cuda.set_device(int(sys.argv[1]))
anchor = torch.zeros(1, device="cuda")
deadline = time.monotonic() + float(sys.argv[2])
while time.monotonic() < deadline and not os.path.exists(sys.argv[3]):
    time.sleep(1.0)
"""


def role_group(mode, local_rank):
    for group in ROLE_PLAN["groups"].get(mode, []):
        if local_rank in group["locals"]:
            return group
    return None


def role_child_environment():
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(ROLE_CHILD_ENV_PREFIXES)
    }


def role_path(name):
    return os.path.join(ROLE_WORK_DIR, name)


class RoleFixture:
    # One rank's declarations, helper threads and child processes of a role
    # mode. Threads start with the rank and idle until stall_begin(); children
    # live for the job and are bounded on their own.

    def __init__(self, mode, rank, world_size, local_rank, device):
        self.mode = mode
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.node_index = rank // ROLE_PROCS_PER_NODE
        self.device = device
        self.group = role_group(mode, local_rank)
        self.current_step = 0
        self.stop = threading.Event()
        self.active = threading.Event()
        self.thread_roles = {}
        self.thread_names = {}
        self.threads = []
        self.children = []
        self.context = None
        self.cpu_tensors = None
        self.child_seconds = STALL_MAX_SECONDS + float(
            ROLE_PLAN["child_margin_seconds"]
        )
        self.shares_gpu = local_rank in ROLE_PLAN["share_gpu"].get(mode, [])
        self.runs_holder = local_rank in ROLE_PLAN["holder"].get(mode, [])

    # -- declarations -------------------------------------------------------
    def helpers(self):
        helpers = [
            ("gfrs-comm", "COMMUNICATION", self.wait_loop, 0.5),
            ("gfrs-keeper", "BACKGROUND", self.keeper_loop, None),
        ]
        if self.mode == "threads":
            helpers += [
                ("gfrs-bg-a", "BACKGROUND", self.wait_loop, 1.0),
                ("gfrs-bg-b", "BACKGROUND", self.wait_loop, 1.0),
                ("gfrs-churn", None, self.churn_loop, None),
            ]
            if self.rank % 2:
                helpers.reverse()
        if self.mode == "roles" and self.group is not None and self.group["burn"]:
            helpers += [
                (f"gfrs-burn-{index}", "BACKGROUND", self.burn_loop, None)
                for index in range(ROLE_BURNER_THREADS)
            ]
        if self.mode == "wchan":
            helpers += [
                ("gfrs-spinner", "COMMUNICATION", self.burn_loop, None),
                ("gfrs-sleeper", "BACKGROUND", self.sleep_loop, None),
            ]
        if self.mode == "cpu":
            helpers.append(("gfrs-cpu-duty", "BACKGROUND", self.duty_loop, None))
        return helpers

    def start(self):
        main_tid = str(threading.get_native_id())
        self.thread_roles = {main_tid: "TRAINING"}
        self.thread_names = {main_tid: threading.current_thread().name}
        for name, role, target, argument in self.helpers():
            thread = threading.Thread(
                target=target
                if argument is None
                else (lambda t=target, a=argument: t(a)),
                daemon=True,
                name=name,
            )
            thread.start()
            self.threads.append(thread)
            if role is not None:
                self.thread_roles[str(thread.native_id)] = role
                self.thread_names[str(thread.native_id)] = name
        if self.group is not None:
            self.context = {
                "group_name": f"{self.group['name']}-node{self.node_index}",
                "members": [
                    self.node_index * ROLE_PROCS_PER_NODE + local
                    for local in self.group["locals"]
                ],
                "role": ROLE_PLAN["role"],
                "pipeline_stage": self.group["stage"],
                "expert_role": self.group["expert"],
                "hardware_class": ROLE_PLAN["hardware_class"],
                "code_version": ROLE_PLAN["code_version"],
                "thread_roles": dict(self.thread_roles),
            }
        if self.shares_gpu:
            self.start_sharer()
        if self.runs_holder:
            self.start_holder()
        self.declare()
        self.announce()

    def announce(self):
        group = self.group or {}
        if self.context is None:
            group_name = members = roles = "-"
        else:
            group_name = self.context["group_name"]
            members = ",".join(str(member) for member in self.context["members"])
            roles = ",".join(
                f"{tid}:{role}:{self.thread_names[tid]}"
                for tid, role in self.thread_roles.items()
            )
        say(
            f"ROLES mode={self.mode} node_index={self.node_index} "
            f"group={group_name} stage={group.get('stage', '-')} "
            f"expert={group.get('expert', '-')} members={members} roles={roles} "
            f"clk_tck={os.sysconf('SC_CLK_TCK')} pid={os.getpid()} "
            f"burn={int(bool(group.get('burn')))} share_gpu={int(self.shares_gpu)} "
            f"holder={int(self.runs_holder)}"
        )

    def declare(self):
        guarded(declare_progress, self.current_step, "TRAINING", self.context)

    def step(self, step):
        self.current_step = step
        self.declare()

    def stall_begin(self):
        self.active.set()

    def stall_end(self):
        self.active.clear()

    def finish(self):
        self.active.clear()
        self.stop.set()
        if self.children:
            try:
                with open(role_path(ROLE_PLAN["stop_file"]), "w") as stream:
                    stream.write("stop\n")
            except OSError:
                pass
        for child in self.children:
            try:
                child.terminate()
            except OSError:
                pass

    # -- helper threads -------------------------------------------------------
    def wait_loop(self, seconds):
        while not self.stop.wait(seconds):
            pass

    def sleep_loop(self):
        self.stop.wait()

    def keeper_loop(self):
        while not self.stop.wait(ROLE_KEEPER_SECONDS):
            self.declare()

    def matmul(self):
        if self.cpu_tensors is None:
            shape = (512, 512)
            self.cpu_tensors = [
                torch.randn(shape, device="cpu", dtype=torch.float32)
                for _ in range(3)
            ]
        first, second, out = self.cpu_tensors
        torch.mm(first, second, out=out)

    def burn_loop(self):
        while not self.stop.is_set():
            if not self.active.wait(0.5):
                continue
            self.matmul()

    def duty_loop(self):
        while not self.stop.is_set():
            if not self.active.wait(0.5):
                continue
            until = time.monotonic() + 0.1
            while time.monotonic() < until:
                self.matmul()
            self.stop.wait(0.1)

    def churn_loop(self):
        while not self.stop.is_set():
            if not self.active.wait(0.5):
                continue
            short = threading.Thread(
                target=lambda: self.stop.wait(0.2), daemon=True, name="gfrs-churn-x"
            )
            short.start()
            short.join()
            self.stop.wait(0.3)

    # -- child processes ------------------------------------------------------
    def fixture_error(self, what, error):
        text = " ".join(str(error).split())[:160].replace(" ", "_")
        say(f"GFRS-FIXTURE-ERROR what={what} error={text}")

    def spawn(self, what, argv):
        log = open(role_path(f"gfrs-{what}-{self.rank}.log"), "ab")
        child = subprocess.Popen(
            argv, env=role_child_environment(), stdout=log, stderr=subprocess.STDOUT
        )
        self.children.append(child)
        return child

    def start_sharer(self):
        try:
            child = self.spawn(
                "sharer",
                [
                    sys.executable,
                    "-c",
                    ROLE_SHARER_SOURCE,
                    str(self.local_rank),
                    str(self.child_seconds),
                    role_path(ROLE_PLAN["stop_file"]),
                ],
            )
        except OSError as exc:
            self.fixture_error("sharer", exc)
            return
        say(f"SHARER pid={child.pid} device={self.local_rank}")

    def start_holder(self):
        exe = role_path(f"{ROLE_PLAN['holder_exe']}-{self.rank}")
        source = role_path(ROLE_PLAN["holder_source"])
        compiler = shutil.which("gcc") or shutil.which("cc")
        if compiler is None:
            self.fixture_error("holder", "no C compiler (gcc/cc) in the image")
            return
        try:
            with open(source, "w", encoding="utf-8") as stream:
                stream.write(ROLE_HOLDER_SOURCE)
            built = subprocess.run(
                [compiler, "-O2", "-o", exe, source, "-ldl"],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self.fixture_error("holder", exc)
            return
        if built.returncode != 0:
            self.fixture_error("holder", f"compile_{built.returncode}:{built.stderr}")
            return
        try:
            child = self.spawn(
                "holder",
                [
                    exe,
                    str(self.local_rank),
                    str(int(self.child_seconds)),
                    role_path(ROLE_PLAN["stop_file"]),
                ],
            )
        except OSError as exc:
            self.fixture_error("holder", exc)
            return
        time.sleep(3.0)
        if child.poll() is not None:
            self.fixture_error("holder", f"exited_{child.returncode}")
            return
        say(f"HOLDER pid={child.pid} device={self.local_rank} exe={exe}")
# ---- end of the role/thread/process extension ----
'''

PLAN_JSON = json.dumps(PLAN, separators=(",", ":"))
if '"""' in PLAN_JSON or "\\" in PLAN_JSON:
    raise ValueError("the role plan would end or escape the trainer's own literal")
ROLE_TRAINER = ROLE_TRAINER_TEMPLATE.replace("__ROLE_PLAN__", PLAN_JSON)


__all__ = [
    "BURNER_THREADS",
    "CODE_VERSION",
    "CPU",
    "FIXTURE_ERROR_MARKER",
    "HARDWARE_CLASS",
    "HOLDER_EXE",
    "HOLDER_MARKER",
    "KEEPER_SECONDS",
    "MARKERS",
    "PLAN",
    "PROCS_PER_NODE",
    "PYSPY",
    "ROLES",
    "ROLES_MARKER",
    "ROLE_MODES",
    "ROLE_NAME",
    "ROLE_TRAINER",
    "SHARER_MARKER",
    "THREADS",
    "THREAD_ROLES",
    "WCHAN",
    "burning_locals",
    "declared_roles",
    "expected_context",
    "group_of",
    "holder_locals",
    "marker_lines",
    "parse_marker",
    "sharing_locals",
]
