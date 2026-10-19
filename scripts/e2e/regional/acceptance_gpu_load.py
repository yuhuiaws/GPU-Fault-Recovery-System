"""The GPU load the acceptance training fixtures run between their heartbeats.

A real training job keeps kernels running on its GPUs. The long-running managed
fixtures used to all-reduce one value, print a ``HEARTBEAT`` line and sleep two
seconds, so nvidia-smi read about 0 % ``utilization.gpu`` for their whole life.
The product's sustained ``LOW_GPU_UTILIZATION`` rule (``host_health.py``: a GPU
the live managed attempt holds on the node at or below 5 % for 300 s, each GPU
graded on its own) reads that correctly as "GPUs idle while training is
active", so every case whose job lived 300 s could open a host-health incident
(GF-REGIONAL-E2E-002 had its injected XID merged into one).

What ``utilization.gpu`` measures. nvidia-smi documentation,
https://docs.nvidia.com/deploy/nvidia-smi/index.html, GPU ATTRIBUTES >
Utilization > GPU, nvidia-smi v610 Update (the page's newest change-log entry
is "Changes between nvidia-smi v610 Update and v595"):

    "Percent of time over the past sample period during which one or more
    kernels was executing on the GPU. The sample period may be between 1 second
    and 1/6 second depending on the product."

The duty cycle. Between two heartbeats every rank runs fp16 matmuls of two
2048 x 2048 matrices (three 8 MiB tensors per GPU) for
``GPU_FAULT_ACCEPTANCE_GPU_DUTY_PERCENT`` (default 30, at most 50) of every
10 ms period, ``torch.cuda.synchronize`` after every batch of eight, and waits
out the rest of the period, until the deadline the old ``time.sleep(2)`` had.
A sample period of at least 1/6 s spans at least 15 whole periods, so a reading
is the duty less at most one period's share (about 28 % at the default): far
above the 5 % threshold and far below the 80 % utilization that
``POWER_LIMIT_THROTTLING`` correlates with. The load is sized for an average
draw that is a fraction of the power limit, so the temperature grading's
T.Limit margins stay wide (a design bound, not a hardware measurement). The
synchronize bounds what can be queued at the deadline to one batch, so the
HEARTBEAT cadence, the step counts and the total duration are the old ones.
``0`` is the idle heartbeat: a case that needs idle GPUs sets it in its
rendered manifest (:func:`set_gpu_duty_percent`).

SIGTERM. The snippet blocks SIGTERM before torch starts any thread and a
watcher thread takes it with ``sigwait``: the main loop finishes the batch in
flight, synchronizes and exits 143, and the watcher exits 143 itself after
``GPU_TERM_GRACE_SECONDS`` if the main thread is stuck in a collective (inside
the kubelet's minimum two-second grace). A stop therefore tears down a GPU
context with no kernel of ours running, as the idle heartbeat's did. NVIDIA's
Xid catalog (docs version 615,
https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html, "Xids"
sheet of the linked Xid-Catalog.xlsx, code 45 ROBUST_CHANNEL_PREEMPTIVE_REMOVAL,
Trigger Conditions): "This event is logged when the user application aborts and
the kernel driver tears down the GPU application running on the GPU.
Control-C, GPU resets, sigkill are all examples where the application is
aborted and this event is created." A deliberate SIGKILL (COLLECT-021's
kill-workload) can still land inside a batch.

Inventory: every GPU workload the regional harness submits.

* GPU-active, this snippet: ``manifests/training/xid11-three-node-pytorchjob.yaml``
  (E2E-002, WORKLOAD-*, DESTR-009/012 group A, COLLECT-016/017/021 through
  ``run_collector_destructive.render_named_training_manifest`` and
  ``training_topology.render_topology_manifest``),
  ``two-node-replace-pytorchjob.yaml`` (DESTR-014/015/026),
  ``single-node-warm-spare-pytorchjob.yaml`` (DESTR-003/008) and
  ``xid11-single-node-job.yaml`` (DESTR-012 group D; one GPU, so the node's
  seven other GPUs stay idle -- the product grades only the GPUs an attempt
  holds, so they are not graded).
* Short-lived, unchanged: ``three-node-managed-pytorchjob.yaml`` and
  ``examples/hyperpod/three-node-pytorchjob.yaml`` (three DDP steps, done in
  seconds; WORKLOAD-001/002's baseline).
* Idle by design, unchanged: NOTIFY-005's CPU-only burn
  (``notify005_checks.low_utilization_manifest``: it exists to trip
  ``LOW_GPU_UTILIZATION``, so it holds every GPU of its nodes and creates no
  CUDA context, which also leaves ``LOW_CPU_UTILIZATION`` nothing to grade),
  the warm-spare GPU holder Pod
  (``warm_spare_fixture.GpuHolderFixture``, unmanaged) and the late-ownership
  CUDA holder (``late_ownership_resources.CUDA_HOLDER``, 240 s deadline).
* Not submitted by any runner or case procedure, unchanged: ``q118-*.yaml``
  (legacy Q118 drills; the intensive and log-snapshot ones are GPU-bound) and
  ``regional-isolation-gpu-job.yaml``.

The snippet is copied into every training script of the GPU-active manifests
between ``SNIPPET_BEGIN`` and ``SNIPPET_END``; ``--check`` (the default)
proves the copies are this text and ``--write`` rewrites drifted ones.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

TRAINING_MANIFESTS = Path(__file__).with_name("manifests") / "training"
GPU_DUTY_ENV = "GPU_FAULT_ACCEPTANCE_GPU_DUTY_PERCENT"
DEFAULT_GPU_DUTY_PERCENT = 30
MAX_GPU_DUTY_PERCENT = 50
GPU_DUTY_PERIOD_SECONDS = 0.01
GPU_LOAD_MATRIX = 2048
GPU_LOAD_BATCH = 8
GPU_TERM_GRACE_SECONDS = 1.0
SIGTERM_EXIT_CODE = 143

NVIDIA_SMI_REFERENCE = "https://docs.nvidia.com/deploy/nvidia-smi/index.html"
NVIDIA_SMI_SECTION = "GPU ATTRIBUTES > Utilization > GPU"
NVIDIA_SMI_DOCS_VERSION = "nvidia-smi v610 Update"
NVIDIA_SMI_CHANGE_LOG = "Changes between nvidia-smi v610 Update and v595"
NVIDIA_SMI_UTILIZATION_GPU_QUOTE = (
    "Percent of time over the past sample period during which one or more "
    "kernels was executing on the GPU. The sample period may be between 1 "
    "second and 1/6 second depending on the product."
)
NVIDIA_XID_CATALOG_REFERENCE = (
    "https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html"
)
NVIDIA_XID_CATALOG_DOCS_VERSION = "615"
NVIDIA_XID45_TRIGGER_QUOTE = (
    "This event is logged when the user application aborts and the kernel "
    "driver tears down the GPU application running on the GPU. Control-C, GPU "
    "resets, sigkill are all examples where the application is aborted and "
    "this event is created."
)

SNIPPET_BEGIN = "# BEGIN gpu-fault acceptance GPU load"
SNIPPET_END = "# END gpu-fault acceptance GPU load"
#: Runs before ``import torch`` and imports nothing itself: the training script
#: imports ``os``, ``signal``, ``threading`` and ``time`` first, and ``torch``
#: is looked up only when ``gpu_busy_until`` runs. The tests execute it.
GPU_LOAD_SNIPPET = """\
# BEGIN gpu-fault acceptance GPU load (scripts/e2e/regional/acceptance_gpu_load.py)
# A training job keeps kernels running; an idle heartbeat reads 0 % in nvidia-smi
# utilization.gpu, "Percent of time over the past sample period during which one
# or more kernels was executing on the GPU. The sample period may be between 1
# second and 1/6 second depending on the product." (nvidia-smi v610 Update docs,
# GPU ATTRIBUTES > Utilization > GPU,
# https://docs.nvidia.com/deploy/nvidia-smi/index.html). So between heartbeats
# every GPU runs fp16 matmuls for GPU_FAULT_ACCEPTANCE_GPU_DUTY_PERCENT (0-50,
# default 30, 0 = idle) of every 10 ms, synchronized per batch of eight; SIGTERM
# ends the batch in flight and exits 143.
GPU_DUTY_ENV = "GPU_FAULT_ACCEPTANCE_GPU_DUTY_PERCENT"
GPU_DUTY_PERIOD_SECONDS = 0.01
GPU_LOAD_MATRIX = 2048
GPU_LOAD_BATCH = 8
GPU_TERM_GRACE_SECONDS = 1.0
GPU_LOAD = {}
TERMINATING = threading.Event()


def gpu_duty_percent():
    raw = os.environ.get(GPU_DUTY_ENV, "30").strip()
    if not (raw.isascii() and raw.isdigit()) or int(raw) > 50:
        raise SystemExit(f"{GPU_DUTY_ENV} must be an integer 0-50, not {raw!r}")
    return int(raw)


def watch_sigterm():
    def watch():
        signal.sigwait({signal.SIGTERM})
        TERMINATING.set()
        time.sleep(GPU_TERM_GRACE_SECONDS)
        os._exit(143)

    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    threading.Thread(target=watch, daemon=True).start()


def gpu_busy_until(deadline, device):
    load = GPU_LOAD.get(str(device))
    if load is None and GPU_DUTY_PERCENT:
        shape = (GPU_LOAD_MATRIX, GPU_LOAD_MATRIX)
        load = GPU_LOAD[str(device)] = [
            torch.randn(shape, device=device, dtype=torch.float16) for _ in range(3)
        ]
    while not TERMINATING.is_set():
        start = time.monotonic()
        if start >= deadline:
            return
        period = min(GPU_DUTY_PERIOD_SECONDS, deadline - start)
        busy_until = start + period * GPU_DUTY_PERCENT / 100
        while load and time.monotonic() < busy_until:
            for _ in range(GPU_LOAD_BATCH):
                torch.mm(load[0], load[1], out=load[2])
            torch.cuda.synchronize(device)
        TERMINATING.wait(max(0.0, start + period - time.monotonic()))
    torch.cuda.synchronize(device)
    os._exit(143)


GPU_DUTY_PERCENT = gpu_duty_percent()
watch_sigterm()
# END gpu-fault acceptance GPU load
"""
#: The fixtures under ``manifests/training`` that run the snippet, and the ones
#: that do not with the reason; the inventory test fails on a new fixture that
#: is in neither.
GPU_ACTIVE_MANIFESTS = (
    "xid11-three-node-pytorchjob.yaml",
    "two-node-replace-pytorchjob.yaml",
    "single-node-warm-spare-pytorchjob.yaml",
    "xid11-single-node-job.yaml",
)
SHORT_LIVED_MANIFESTS = {
    "three-node-managed-pytorchjob.yaml": "three DDP steps, done in seconds",
}


class SnippetError(ValueError):
    """A manifest's snippet markers are missing, unbalanced or nested."""


def sync_snippet(text: str) -> str:
    """``text`` with every BEGIN..END block replaced by the canonical snippet.

    Each block keeps the indentation of its BEGIN line, so a copy inside a YAML
    block scalar stays in it; blank snippet lines stay empty (no trailing
    whitespace for yamllint).
    """

    lines = text.splitlines(keepends=True)
    result: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.lstrip().startswith(SNIPPET_BEGIN):
            if line.strip() == SNIPPET_END:
                raise SnippetError("acceptance GPU load END marker without BEGIN")
            result.append(line)
            index += 1
            continue
        indent = line[: len(line) - len(line.lstrip())]
        end = index + 1
        while end < len(lines) and lines[end].strip() != SNIPPET_END:
            if lines[end].lstrip().startswith(SNIPPET_BEGIN):
                raise SnippetError("acceptance GPU load blocks are nested")
            end += 1
        if end == len(lines):
            raise SnippetError("acceptance GPU load BEGIN marker without END")
        result.extend(
            f"{indent}{item}\n" if item else "\n"
            for item in GPU_LOAD_SNIPPET.splitlines()
        )
        index = end + 1
    return "".join(result)


def snippet_blocks(text: str) -> int:
    """How many acceptance GPU load blocks ``text`` carries."""

    return sum(
        1 for line in text.splitlines() if line.lstrip().startswith(SNIPPET_BEGIN)
    )


def _containers(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        containers = value.get("containers")
        if isinstance(containers, list):
            yield from (item for item in containers if isinstance(item, dict))
        for item in value.values():
            yield from _containers(item)
    elif isinstance(value, list):
        for item in value:
            yield from _containers(item)


def set_gpu_duty_percent(document: dict[str, Any], percent: int) -> dict[str, Any]:
    """Set the duty on every container of ``document`` that runs the load.

    The opt-out (``0``) or override for a rendered copy of a training fixture.
    A manifest without the snippet is refused rather than silently unchanged.
    """

    if type(percent) is not int or not 0 <= percent <= MAX_GPU_DUTY_PERCENT:
        raise ValueError(f"GPU duty must be an integer 0-{MAX_GPU_DUTY_PERCENT}")
    loaded = [
        container
        for container in _containers(document)
        if any(
            snippet_blocks(str(part))
            for part in (
                *(container.get("command") or ()),
                *(container.get("args") or ()),
            )
        )
    ]
    if not loaded:
        raise ValueError("the manifest runs no acceptance GPU load")
    for container in loaded:
        env = [
            item
            for item in container.get("env") or ()
            if not (isinstance(item, dict) and item.get("name") == GPU_DUTY_ENV)
        ]
        container["env"] = [*env, {"name": GPU_DUTY_ENV, "value": str(percent)}]
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Keep the acceptance GPU load copies equal to the canonical text."
    )
    parser.add_argument("--write", action="store_true", help="rewrite drifted copies")
    parser.add_argument("--root", type=Path, default=TRAINING_MANIFESTS)
    arguments = parser.parse_args(argv)
    failures = []
    for name in GPU_ACTIVE_MANIFESTS:
        path = arguments.root / name
        text = path.read_text(encoding="utf-8")
        if not snippet_blocks(text):
            failures.append(f"acceptance GPU load is missing: {path}")
            continue
        synced = sync_snippet(text)
        if synced == text:
            continue
        if arguments.write:
            path.write_text(synced, encoding="utf-8")
            print(f"rewrote {path}")
        else:
            failures.append(f"acceptance GPU load copy drifted: {path}")
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
