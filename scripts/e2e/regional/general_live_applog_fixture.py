"""Reusable CRI Pod-log source fixture for GF-GENERAL-LIVE (017 and any later user).

One small, documented controller -- ``CriPodLogFixture`` -- gives the node log
collector (``gpu-fault-log-collector.service``) a reviewed CRI Pod log to read
for a managed training attempt the runner itself created, so the Pod UID, the
container name and the attempt the product derives are genuine while only the
log *text* is reviewed input (``docs/components/general-fault-live-acceptance.md``
``live-replay``). Everything runs on the node through the one host probe
``probes/general_live_applog_probe.py``:

    logs = cri_pod_logs(ctx)                     # cleanup registered (LIFO)
    gaps = capability_gaps(logs.preflight())     # read-only: root, layout, unit
    logs.open(container="pytorch", approved=True)  # drop-in + deadman; the unit
                                                 # reads only this run's file
    logs.create_log(pod_uid, container="pytorch", namespace=..., pod_name=...)
    logs.wait_tracked(ctx, pod_uid, container="pytorch")  # the collector saw it
    receipt = logs.write(pod_uid, container="pytorch", records=[reviewed_line(...)])
    logs.rotate(...) / logs.write(..., target="rotated") / logs.restart_collector()
    logs.seed_log(old_uid, ..., records=[...], allow_create_dir=True,
                  await_gc_seconds=75)           # old_pod: one call, after the
                                                 # kubelet GC tick (see the probe)
    view = logs.status()                         # files + the collector's state
    ...                                          # the caller reads product records
    logs.cleanup()                               # files, drop-in, unit, deadman

The product reads a training log only through ``GPU_FAULT_TRAINING_LOG_PATHS``
(empty on this site), so the window points that one unit at
``/var/log/pods/*/<container>/gfrs-live-<digest>.log*`` for a bounded time
(``ENABLE_NODE_LOG_COLLECTOR``; a ``systemd-run --on-active`` deadman closes
it). ``pod_identity`` then binds each record to the UID and container of the
directory kubelet made for the runner's own Pod; ``log_recovery_context``
attributes it to the attempt observation the completion watcher recorded.

A failed probe call raises ``HostProbeError`` whose ``reason`` is the probe's
own token (``probes/general_live_applog_probe.py`` ``REASONS``); the sentence
behind it is in the case's ``probe-error.json``. ``GC_LOST_REASONS`` are the
two a caller may repeat a round on.

A reviewed record is ``{"message", "stream"?, "flag"?}`` (``stdout``/``stderr``,
CRI ``P`` partial or ``F`` full); the node stamps it with its own clock when it
writes, which is the event time the product reports. ``records_sha256`` is the
digest the node checks and the caller records in evidence. Nothing here posts a
finding or patches a Node; verdicts read the product's own records.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.e2e.regional.collector_action_guard import require_action_time
from scripts.e2e.regional.host_probe_fixture import HostProbeFixture, HostProbeSettings

if TYPE_CHECKING:
    from scripts.e2e.regional.general_live_fixture import VariantContext

PROBE_SCRIPT = Path(__file__).with_name("probes") / "general_live_applog_probe.py"
PROOF_SCOPE = "live-replay"
PROBE_KEY = "cri-pod-logs"
#: The node log collector unit the window drives (``probe.LOG_UNIT``).
LOG_UNIT = "gpu-fault-log-collector.service"
#: Where the product binds a CRI log to its Pod (``node_sources.pod_log_root``
#: default, fixed in the collector) and where kubelet keeps them on both of
#: this site's runtimes (``probe.POD_LOG_ROOT``).
POD_LOG_ROOT = "/var/log/pods"
#: Probe reason tokens that mean kubelet's container GC took the fixture's
#: directory between two steps: a round a caller may repeat (017 ``old_pod``).
#: Every other reason is the failure it names.
GC_LOST_REASONS = frozenset({"fixture_gone", "directory_removed"})
#: ``seed-log``: the probe's own bound on ``--await-gc-seconds`` (``probe.
#: MAX_GC_WAIT_SECONDS``); the exec timeout and the action-time guard add it.
MAX_GC_WAIT_SECONDS = 180
#: The Pod annotation the completion watcher reads a container's host PID from
#: (``completion_pod_parsing.HOST_PID_ANNOTATION``): the workload's declaration
#: of attribution element four (detailed design v2 section 5.5). The runner
#: declares it for its own test Pod from the PID the node probe read out of the
#: container's cgroup, never a guess.
HOST_PID_ANNOTATION = "gpu-fault.io/host-pid"

#: Reviewed application lines (``log_rules.NODE_LOG_RULES``). The CUDA OOM line
#: is what a PyTorch trainer prints; it matches the MEMORY rule (WARNING,
#: RUN_DIAGNOSTICS, any origin), so the product's attribution parameters reach
#: the diagnostic bundle step where a verdict can read them. The NCCL line is
#: the NO_ACTION evidence-only rule.
CUDA_OOM_LINE = (
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. "
    "GPU 0 has a total capacity of 139.81 GiB of which 1.12 GiB is free."
)
NCCL_ERROR_LINE = "test-app: NCCL WARN Bootstrap : unhandled error, aborting"
#: A distributed-training line split as containerd splits a long line: neither
#: half matches any log rule on its own, the stitched line matches the TRAINING
#: rule (``collective ... timeout``) -- so one record proves the P/F join.
PARTIAL_HEAD = "test-app: collective operation on rank 3 did not complete; "
PARTIAL_TAIL = "backend status: timeout after 1800000 ms"
#: How long the collector may take to record a new fixture file: its poll is
#: 10 s and the window restart precedes the first poll.
TRACK_TIMEOUT_SECONDS = 90
TRACK_POLL_SECONDS = 5


def reviewed_line(
    message: str, *, stream: str = "stdout", flag: str = "F"
) -> dict[str, str]:
    """One reviewed CRI record for ``write``; ``flag`` ``P`` is a partial."""

    if stream not in {"stdout", "stderr"}:
        raise ValueError("a CRI record stream is stdout or stderr")
    if flag not in {"P", "F"}:
        raise ValueError("a CRI record flag is P or F")
    if not message or "\n" in message:
        raise ValueError("a reviewed record message is one non-empty line")
    return {"message": message, "stream": stream, "flag": flag}


def records_sha256(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_encode(records)).hexdigest()


def _encode(records: list[dict[str, Any]]) -> bytes:
    return json.dumps(records, sort_keys=True).encode("utf-8")


def capability_gaps(preflight: dict[str, Any]) -> list[str]:
    """What the node lacks for a CRI Pod-log window, from the read-only ``preflight``.

    Each sentence names the exact missing capability (``NOT_RUN(CAPABILITY_ABSENT)``
    before anything on the node is touched): kubelet's Pod-log root, at least
    one kubelet-shaped ``<namespace>_<pod>_<uid>/<container>/0.log`` directory
    under it -- the shape the product's ``pod_identity`` binds by and
    ``old_pod`` recreates an old Pod's directory in -- and the installed node
    log collector unit. The receipt is the probe's; a key it did not send is a
    gap, never a pass.
    """

    gaps: list[str] = []
    root = str(preflight.get("pod_log_root_path") or POD_LOG_ROOT)
    if preflight.get("pod_log_root") is not True:
        gaps.append(
            f"kubelet's CRI Pod-log directory {root} does not exist on the node"
        )
    else:
        layout = preflight.get("pod_log_layout") or {}
        if not isinstance(layout, dict) or not int(
            layout.get("with_container_log") or 0
        ):
            gaps.append(
                "no kubelet-shaped <namespace>_<pod>_<uid>/<container>/0.log "
                f"directory under {root}: the collector's pod_identity cannot bind a "
                "CRI log to a Pod on this node"
            )
    unit = preflight.get("unit") or {}
    load_state = unit.get("LoadState") if isinstance(unit, dict) else None
    if load_state != "loaded":
        gaps.append(
            f"{LOG_UNIT} is not installed on the node (LoadState={load_state!r})"
        )
    return gaps


class CriPodLogFixture:
    """Owns the host probe and the reviewed CRI Pod-log window for one node."""

    def __init__(
        self,
        regional: Any,
        *,
        node: str,
        image: str,
        case_id: str,
        run_id: str,
        case_dir: Path,
        active_deadline_seconds: int = 3600,
        host: Any | None = None,
    ) -> None:
        self.run_id = run_id
        self.case_dir = case_dir
        self.container: str | None = None
        self._opened = False
        self._written: list[dict[str, Any]] = []
        self.host = host or HostProbeFixture(
            HostProbeSettings(
                kubeconfig=regional.settings.gpu_kubeconfig,
                context=regional.settings.gpu_context,
                namespace=regional.settings.namespace,
                node=node,
                image=image,
                case_id=case_id,
                run_id=run_id,
                probe_script=PROBE_SCRIPT,
                state_directory=case_dir / "host-probes",
                active_deadline_seconds=active_deadline_seconds,
            )
        )

    # -- lifecycle ------------------------------------------------------- #
    def create(self) -> None:
        if not getattr(self.host, "created", False):
            require_action_time(180)
            self.host.create()

    def preflight(self) -> dict[str, Any]:
        self.create()
        return self.host.execute("preflight", timeout=120)

    def inspect(
        self, pod_uid: str, *, container_id: str | None = None
    ) -> dict[str, Any]:
        """Read-only: the Pod's real log directories, its container's init PID."""

        self.create()
        arguments = ["inspect", "--run-id", self.run_id, "--pod-uid", pod_uid]
        if container_id:
            arguments += ["--container-id", container_id.split("://")[-1]]
        return self.host.execute(*arguments, timeout=120)

    def open(
        self,
        *,
        container: str,
        approved: bool = False,
        restore_seconds: int = 900,
        initial_tail_bytes: int = 0,
    ) -> dict[str, Any]:
        self.create()
        require_action_time(180)
        arguments = [
            "open-window",
            "--run-id",
            self.run_id,
            "--container",
            container,
            "--restore-seconds",
            str(int(restore_seconds)),
            "--initial-tail-bytes",
            str(int(initial_tail_bytes)),
        ]
        if approved:
            arguments.append("--approved")
        receipt = self.host.execute(*arguments, timeout=150)
        self._opened = True
        self.container = container
        return receipt

    def create_log(
        self,
        pod_uid: str,
        *,
        container: str,
        namespace: str,
        pod_name: str,
        allow_create_dir: bool = False,
    ) -> dict[str, Any]:
        """An empty fixture file in the Pod's container log directory.

        ``allow_create_dir`` is for a Pod that no longer exists (``old_pod``):
        the directory is created with the UID that Pod had and removed again.
        """

        arguments = [
            "create-log",
            "--run-id",
            self.run_id,
            "--pod-uid",
            pod_uid,
            "--container",
            container,
            "--namespace",
            namespace,
            "--pod-name",
            pod_name,
        ]
        if allow_create_dir:
            arguments.append("--allow-create-dir")
        return self.host.execute(*arguments, timeout=120)

    def write(
        self,
        pod_uid: str,
        *,
        container: str,
        records: list[dict[str, Any]],
        target: str = "current",
    ) -> dict[str, Any]:
        """Append reviewed records; the node checks the sha256 and stamps them."""

        if not records:
            raise ValueError("write needs at least one reviewed record")
        require_action_time(60)
        receipt = self.host.execute(
            "write",
            "--run-id",
            self.run_id,
            "--pod-uid",
            pod_uid,
            "--container",
            container,
            "--sha256",
            records_sha256(records),
            "--b64",
            base64.b64encode(_encode(records)).decode("ascii"),
            "--target",
            target,
            timeout=120,
        )
        self._written.append(
            {
                "pod_uid": pod_uid,
                "container": container,
                "target": target,
                "records": list(records),
                "sha256": records_sha256(records),
                "stamps": list(receipt.get("stamps") or []),
            }
        )
        return receipt

    def seed_log(
        self,
        pod_uid: str,
        *,
        container: str,
        namespace: str,
        pod_name: str,
        records: list[dict[str, Any]],
        allow_create_dir: bool = False,
        await_gc_seconds: int = 0,
    ) -> dict[str, Any]:
        """``create_log`` and ``write`` in one probe call (``seed-log``).

        For a directory kubelet is going to remove (``old_pod``): with
        ``await_gc_seconds`` the probe first watches the old Pod's directory
        for kubelet's GC tick and seeds right after it, so the file lives a
        whole GC period for the collector's poll. A ``HostProbeError`` whose
        ``reason`` is in ``GC_LOST_REASONS`` is a round the caller may repeat.
        """

        if not records:
            raise ValueError("seed_log needs at least one reviewed record")
        if not 0 <= int(await_gc_seconds) <= MAX_GC_WAIT_SECONDS:
            raise ValueError(f"await_gc_seconds is outside 0..{MAX_GC_WAIT_SECONDS}")
        require_action_time(120 + int(await_gc_seconds))
        arguments = [
            "seed-log",
            "--run-id",
            self.run_id,
            "--pod-uid",
            pod_uid,
            "--container",
            container,
            "--namespace",
            namespace,
            "--pod-name",
            pod_name,
            "--sha256",
            records_sha256(records),
            "--b64",
            base64.b64encode(_encode(records)).decode("ascii"),
            "--await-gc-seconds",
            str(int(await_gc_seconds)),
        ]
        if allow_create_dir:
            arguments.append("--allow-create-dir")
        receipt = self.host.execute(*arguments, timeout=120 + int(await_gc_seconds))
        self._written.append(
            {
                "pod_uid": pod_uid,
                "container": container,
                "target": "current",
                "records": list(records),
                "sha256": records_sha256(records),
                "stamps": list(receipt.get("stamps") or []),
                "seeded": True,
                "gc": receipt.get("gc"),
            }
        )
        return receipt

    def rotate(self, pod_uid: str, *, container: str) -> dict[str, Any]:
        require_action_time(60)
        return self.host.execute(
            "rotate",
            "--run-id",
            self.run_id,
            "--pod-uid",
            pod_uid,
            "--container",
            container,
            timeout=120,
        )

    def restart_collector(self) -> dict[str, Any]:
        require_action_time(120)
        return self.host.execute(
            "restart-collector", "--run-id", self.run_id, timeout=150
        )

    def status(self) -> dict[str, Any]:
        """Fixture files and the collector's own state about them (the oracle)."""

        return self.host.execute("status", "--run-id", self.run_id, timeout=60)

    def tracked(self, status: dict[str, Any], path: str) -> dict[str, Any] | None:
        """The collector's offset record of ``path``, or ``None``."""

        tracked = (status.get("collector") or {}).get("tracked") or {}
        value = tracked.get(path)
        return dict(value) if isinstance(value, dict) else None

    def pending_fragment(
        self, status: dict[str, Any], path: str, *, stream: str = "stdout"
    ) -> dict[str, Any] | None:
        """The CRI fragment the collector holds for ``path``, or ``None``."""

        pending = (status.get("collector") or {}).get("pending") or {}
        value = ((pending.get(path) or {}).get("cri") or {}).get(stream)
        return dict(value) if isinstance(value, dict) else None

    def wait_tracked(
        self, ctx: VariantContext, path: str, *, name: str
    ) -> dict[str, Any]:
        """Poll ``status`` until the collector records ``path`` at an offset.

        A file the collector has not recorded yet is baselined at its size when
        it is first seen (``GPU_FAULT_NODE_LOG_INITIAL_TAIL_BYTES`` is 0), so a
        line written before that would never be read.
        """

        last: dict[str, Any] = {}

        def accept() -> dict[str, Any] | None:
            nonlocal last
            last = self.status()
            return last if self.tracked(last, path) is not None else None

        value = ctx.wait_until(
            accept,
            timeout_seconds=TRACK_TIMEOUT_SECONDS,
            poll_seconds=TRACK_POLL_SECONDS,
            name=name,
        )
        return value if value is not None else last

    def close(self) -> dict[str, Any]:
        receipt = self.host.execute(
            "close-window", "--run-id", self.run_id, timeout=150
        )
        self._opened = False
        return receipt

    def cleanup(self) -> dict[str, Any]:
        """Close the window and delete the probe; raise if anything is left.

        Pushed onto the variant's LIFO cleanup stack, so a window that cannot be
        proved closed fails the variant (``run_general_live`` ``_teardown``).
        """

        receipts: dict[str, Any] = {}
        if self._opened:
            receipts["close"] = self.close()
            if not receipts["close"].get("clean", True):
                raise RuntimeError(
                    f"CRI Pod-log window did not close clean: {receipts['close']}"
                )
        receipts["probe_residuals"] = self.host.cleanup()
        if any(receipts["probe_residuals"].values()):
            raise RuntimeError(
                f"CRI Pod-log probe left residue: {receipts['probe_residuals']}"
            )
        return receipts

    # -- evidence -------------------------------------------------------- #
    def evidence(self) -> dict[str, Any]:
        return {
            "proof_scope": PROOF_SCOPE,
            "container": self.container,
            "opened": self._opened,
            "writes": list(self._written),
        }


def cri_pod_logs(ctx: VariantContext) -> CriPodLogFixture:
    """Build a ``CriPodLogFixture`` from a variant context and register cleanup.

    The cleanup goes on ``ctx.cleanup`` (LIFO), so the runner tears the window
    down in its ``finally`` and a window that will not close clean fails the
    variant. A test may pre-register a fake host under
    ``ctx.probes["cri-pod-logs@<node>"]``; production never sets it.
    """

    host = ctx.probes.get(f"{PROBE_KEY}@{ctx.settings.node}")
    fixture = CriPodLogFixture(
        ctx.regional,
        node=ctx.settings.node,
        image=ctx.settings.host_probe_image,
        case_id=ctx.spec.key,
        run_id=ctx.run_id,
        case_dir=ctx.case_dir,
        active_deadline_seconds=ctx.spec.max_minutes * 60 + 1200,
        host=host,
    )
    ctx.cleanup.push(PROBE_KEY, fixture.cleanup)
    return fixture


__all__ = [
    "CUDA_OOM_LINE",
    "CriPodLogFixture",
    "GC_LOST_REASONS",
    "HOST_PID_ANNOTATION",
    "LOG_UNIT",
    "MAX_GC_WAIT_SECONDS",
    "NCCL_ERROR_LINE",
    "PARTIAL_HEAD",
    "PARTIAL_TAIL",
    "POD_LOG_ROOT",
    "PROBE_KEY",
    "PROOF_SCOPE",
    "capability_gaps",
    "cri_pod_logs",
    "records_sha256",
    "reviewed_line",
]
