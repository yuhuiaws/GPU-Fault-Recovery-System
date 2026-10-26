"""The DCGM replay's end when the node stops answering the probe.

Every replay object lives under ``/run`` (the collector drop-in, the proxy's
transient unit and deadman timer, ``/run/gfrs-live/<digest>/``), and the
runner reaches them through one node-pinned probe Pod (``kubectl exec``).
Two product actions take that Pod away mid-run:

* ``QUIESCE_GPU_SERVICES`` stops kubelet on a HyperPod node until
  ``RESTORE_GPU_SERVICES`` starts it again (``adapters/node_action/
  quiesce_taint``; an eks-managed node keeps kubelet and taints
  ``gpu-fault.io/quiesce`` NoExecute instead, which the probe Pod tolerates).
  Live 2026-10-26 14:54Z (the HyperPod lane's node, R13g, 024
  ``wrong_scope_source``): the repair GPU's reset chain quiesced the node
  ~20 s after its incident, the oracle gate's ``audit-read`` and then the
  cleanup's ``close-run`` returned "host probe response is missing (exit 1)",
  and the variant ended FAIL "cleanup did not prove the node clean" although
  nothing was wrong with the node.
* ``RESTART_NODE`` reboots it (the same run: VALIDATE_GPU failed on the held
  replay, QUARANTINE, reboot 14:57-14:59:45Z). A reboot empties ``/run`` --
  the drop-in, the run directory, the transient proxy unit and its deadman
  cannot survive it -- so the node is clean by construction, and the probe
  Pod's container is gone with it (restartPolicy Never).

``close_after_probe_loss`` turns a failed ``close-run`` into a judgement
instead of a verdict: it polls the Node object (read-only) while the variant
still has budget, notes a boot id that changed against the one the replay
was deployed under, waits for the node to be Ready for ``READY_SETTLE_SECONDS``
with no quiesce taint, then runs ``close-run`` again through the same probe --
the journals are read first, so the oracle is kept -- and only when the node
never came back without having rebooted does the close stay unproven.
``wait_served_or_records`` is the oracle gate's counterpart: when the journal
cannot be read because the node is away, the product's own records (an
incident on the replayed reading) stand in for it, and the journal is read at
the close.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from scripts.e2e.regional.general_live_fixture import VariantContext
from scripts.e2e.regional.host_probe_fixture import HostProbeError

#: The taint the executor puts on an eks-managed node for the quiesce window
#: (``gpu_fault.adapters.common.QUIESCE_TAINT``); a HyperPod node carries none
#: and stops kubelet instead.
QUIESCE_TAINT = "gpu-fault.io/quiesce"
#: How long Ready must have held before the probe is tried again: a HyperPod
#: node that just booted flips Ready once more ~30 s later when kubelet is
#: restarted with its final config, and an exec across that restart fails
#: (``regional_live_fixture.project_node_snapshot``).
READY_SETTLE_SECONDS = 45
#: The poll while the node is away, the longest the close waits for it (a
#: HyperPod reboot took 2 min 45 s live; a quiesce window 1-2 min), and the
#: part of the variant's remaining budget kept for the close itself.
NODE_POLL_SECONDS = 30
NODE_RETURN_CAP_SECONDS = 15 * 60
CLOSE_RESERVE_SECONDS = 180
#: A node that reads Ready while kubelet is already down (the node-monitor
#: grace period, ~40 s) refuses the probe without changing: the probe is tried
#: again when the Ready transition time changes or this long after the last try.
RETRY_INTERVAL_SECONDS = 120
#: ``judged_by`` of a close that no probe performed.
JUDGED_BY_REBOOT = "node reboot"


@dataclass(frozen=True)
class NodeFacts:
    """What the Node object says about reaching the node now."""

    boot_id: str
    ready: str
    ready_since: str
    unschedulable: bool
    taints: tuple[str, ...]

    @property
    def quiesced(self) -> bool:
        return QUIESCE_TAINT in self.taints

    def settled_ready(self, now: datetime) -> bool:
        """Ready for at least ``READY_SETTLE_SECONDS``, not inside a quiesce."""

        if self.ready != "True" or self.quiesced:
            return False
        since = _parse_time(self.ready_since)
        if since is None:
            return True
        return (now - since).total_seconds() >= READY_SETTLE_SECONDS

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "taints": list(self.taints), "quiesced": self.quiesced}


def _parse_time(text: str) -> datetime | None:
    try:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value if value.tzinfo is not None else None


def node_facts(ctx: VariantContext) -> NodeFacts:
    """The node's boot id, readiness and taints from the Node object (read-only)."""

    snapshot = ctx.regional.node_snapshot(ctx.settings.node)
    return NodeFacts(
        boot_id=str(snapshot.get("boot_id") or ""),
        ready=str(snapshot.get("ready") or ""),
        ready_since=str(snapshot.get("ready_since") or ""),
        unschedulable=bool(snapshot.get("unschedulable")),
        taints=tuple(
            str(item.get("key") or "")
            for item in snapshot.get("taints") or []
            if isinstance(item, dict)
        ),
    )


def node_boot_id(ctx: VariantContext) -> str:
    """The node's current boot id, or ``""`` when the Node object cannot say."""

    try:
        return node_facts(ctx).boot_id
    except Exception:  # noqa: BLE001 - the reference is optional; the close says so
        return ""


def rebooted_verdict(boot_id_before: str, boot_id_after: str) -> dict[str, Any]:
    """A ``close-run`` document for a run the node's reboot closed.

    ``/run`` is a tmpfs: the drop-in, the run directory, the transient proxy
    unit and the deadman timer all went with the boot, and the collector unit
    came back on its own original endpoint.
    """

    return {
        "clean": True,
        "leftover": [],
        "errors": [],
        "judged_by": JUDGED_BY_REBOOT,
        "rebooted": {"boot_id_before": boot_id_before, "boot_id_after": boot_id_after},
        "note": (
            "the node rebooted since the replay was deployed; its /run objects "
            "(collector drop-in, proxy unit, deadman, run directory) cannot have "
            "survived the boot"
        ),
    }


def close_after_probe_loss(
    ctx: VariantContext,
    error: HostProbeError,
    *,
    run_close: Callable[[], dict[str, Any]],
    node_boot_id: str,
    receipts: dict[str, Any],
) -> dict[str, Any] | None:
    """Judge a ``close-run`` the probe could not perform.

    Polls the Node object while the variant has budget (``NODE_RETURN_CAP_
    SECONDS`` at most, ``CLOSE_RESERVE_SECONDS`` kept back), retries
    ``run_close`` once the node is settled Ready with no quiesce taint (again
    after each Ready transition, else every ``RETRY_INTERVAL_SECONDS``), and
    returns its document; a node whose boot id changed against
    ``node_boot_id`` closes the run by its reboot (``rebooted_verdict``) when
    the retry cannot run or fails. Returns ``None`` when the node never came
    back and did not reboot -- the caller's cleanup error. Every poll and
    attempt is in ``receipts["close_after_probe_loss"]`` and the evidence
    file ``node-return-timeline``.
    """

    receipts["close_error"] = f"{type(error).__name__}: {error}"
    attempts: list[dict[str, Any]] = []
    rebooted: dict[str, str] | None = None
    started = ctx.clock()
    budget = min(
        float(NODE_RETURN_CAP_SECONDS),
        ctx.remaining_seconds() - CLOSE_RESERVE_SECONDS,
    )
    tried: tuple[str, datetime] | None = None
    while True:
        now = ctx.clock()
        facts = node_facts(ctx)
        if node_boot_id and facts.boot_id and facts.boot_id != node_boot_id:
            rebooted = {"boot_id_before": node_boot_id, "boot_id_after": facts.boot_id}
        entry: dict[str, Any] = {
            "observed_at": now.isoformat(),
            **facts.as_dict(),
            "rebooted": rebooted is not None,
            "settled_ready": facts.settled_ready(now),
        }
        attempts.append(entry)
        closed: dict[str, Any] | None = None
        due = tried is None or (
            tried[0] != facts.ready_since
            or (now - tried[1]).total_seconds() >= RETRY_INTERVAL_SECONDS
        )
        if entry["settled_ready"] and due:
            tried = (facts.ready_since, now)
            try:
                closed = run_close()
            except HostProbeError as retry_error:
                entry["close_run"] = f"{type(retry_error).__name__}: {retry_error}"
            else:
                entry["close_run"] = "ok"
        ctx.evidence.write("node-return-timeline", {"entries": attempts})
        if closed is not None:
            receipts["close_after_probe_loss"] = attempts
            return {**closed, "rebooted": rebooted} if rebooted else closed
        if rebooted and entry.get("close_run"):
            # The rebooted node is reachable and still refuses the probe: its
            # Pod's container went with the boot. The reboot decides.
            break
        if (now - started).total_seconds() >= budget:
            break
        ctx.sleep(NODE_POLL_SECONDS)
    receipts["close_after_probe_loss"] = attempts
    if rebooted:
        return rebooted_verdict(rebooted["boot_id_before"], rebooted["boot_id_after"])
    return None


def unreachable_detail(attempts: Sequence[dict[str, Any]]) -> str:
    """Why the close stayed unproven, from the polls that were made."""

    if not attempts:
        return "the node was not polled"
    last = attempts[-1]
    state = f"Ready={last.get('ready')!s}"
    if last.get("quiesced"):
        state += ", quiesce taint present"
    return (
        f"the node did not reboot (boot id {last.get('boot_id') or 'unknown'} "
        f"unchanged) and the probe could not run after {len(attempts)} poll"
        f"{'s' if len(attempts) != 1 else ''} ({state})"
    )


def wait_served_or_records(
    replay: Any,
    keys: Sequence[str],
    *,
    records_show: Callable[[], bool],
    label: str,
) -> str:
    """The oracle gate, with the product's records standing in for a journal
    the probe can no longer read.

    ``replay.wait_served`` polls the replay journal through the probe; a
    HyperPod node the product quiesces right after the mint (kubelet down)
    makes that read fail before the journal shows the served reading (live
    2026-10-26 14:54Z, 024 ``wrong_scope_source``). When ``records_show``
    finds the product's own record of the replayed reading -- an incident on
    exactly the replayed GPU and field, since the run began -- the replay
    evidently reached the collector, the gate passes on the records
    (``replay-gate-deferred`` evidence) and the journal is read at the close.
    Any other probe failure, or records that show nothing, is re-raised: the
    fixture failed. Returns ``"journal"`` or ``"records"``.
    """

    try:
        replay.wait_served(*keys)
    except HostProbeError as error:
        if not records_show():
            raise
        replay.ctx.evidence.write(
            "replay-gate-deferred",
            {
                "label": label,
                "keys": list(keys),
                "probe_error": f"{type(error).__name__}: {error}",
                "gate": "the product's records show the replayed reading; the "
                "journal is read at the close",
            },
        )
        return "records"
    return "journal"


__all__ = [
    "CLOSE_RESERVE_SECONDS",
    "JUDGED_BY_REBOOT",
    "NODE_POLL_SECONDS",
    "NODE_RETURN_CAP_SECONDS",
    "NodeFacts",
    "QUIESCE_TAINT",
    "READY_SETTLE_SECONDS",
    "RETRY_INTERVAL_SECONDS",
    "close_after_probe_loss",
    "node_boot_id",
    "node_facts",
    "rebooted_verdict",
    "unreachable_detail",
    "wait_served_or_records",
]
