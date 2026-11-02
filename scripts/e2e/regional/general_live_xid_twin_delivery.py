"""The node side of GF-GENERAL-LIVE-024 ``xid_twin_*``: the two replays and
the placement of one XID's two reports on the node's own clock.

The kernel line goes through the kmsg kernel-source replay
(``general_live_kernel_replay``): reviewed records installed for the run,
``arm --index N`` appending record N alone and answering when it wrote it
(``written_at``, the node's wall clock). The field-230 change goes through the
DCGM field replay (``tools/gpu_dcgm_field_replay.py`` profile ``xid_twin``):
the gauge at 0 from the repointed collector's first scrape (its
``initial-baseline`` delivery is the baseline a change is minted against),
held until the runner advances it, then the XID, held; every journal record
carries the node's wall clock (``at``). From those two clocks:

* kernel-first -- the line, then an advance right after the first scrape at or
  after ``line + 15 s - interval``: the change lands on the next scrape, inside
  the 15-30 s production lag for any collector interval up to 15 s;
* gauge-first -- an advance, the product's record of the change, then the line
  ``delay`` seconds after the change was served (inside the 120 s window);
* the negative control -- the same XID's line again, as a record of its own,
  150 s after the change: outside the twin window, the gauge unchanged.

Nothing here decides a verdict; ``general_live_xid_twins`` grades the
product's records (``general_live_xid_twin_records``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from scripts.e2e.regional.collector_action_guard import require_action_time
from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    node_provider,
)
from scripts.e2e.regional.general_live_gpu_replay import (
    DcgmReplay,
    held_reads,
    plan,
    put,
    step,
)
from scripts.e2e.regional.general_live_gpu_replay_close import (
    close_after_probe_loss,
    node_facts,
    unreachable_detail,
)
from scripts.e2e.regional.general_live_gpu_store import gpu_state
from scripts.e2e.regional.general_live_kernel_replay import (
    PROBE_SCRIPT,
    KernelLogReplay,
    kmsg_follow_error,
    records_sha256,
    reviewed_record,
)
from scripts.e2e.regional.general_live_kernel_xid import xid_message
from scripts.e2e.regional.general_live_verdicts import parse_time
from scripts.e2e.regional.general_live_xid_twin_records import (
    GAUGE,
    GAUGE_SOURCE,
    KERNEL_SOURCE,
    TWIN_WINDOW_SECONDS,
    reports,
    seconds_between,
    twin_records,
)
from scripts.e2e.regional.host_probe_fixture import (
    HostProbeError,
    HostProbeMissingResponseError,
    HostProbeTransportError,
)

TWIN_PROFILE = "xid_twin"
#: The production lag of a field-230 change behind its kernel line: the
#: exporter's 15 s refresh plus the collector's 15 s scrape (batch 237).
PRODUCTION_LAG_SECONDS = (15, 30)
#: How long after the product recorded the change the kernel line follows it
#: in the gauge-first safe and XID 154 variants (inside the 120 s window).
GAUGE_FIRST_DELAY_SECONDS = 30
#: The negative control's repeat lands this far past the twin window,
#: measured from the change (the clock skew between runner and node is ~1 s).
REPEAT_MARGIN_SECONDS = 30
#: Copies of each reviewed line (one sequence number: they dedup to one event).
KMSG_EMISSIONS = 2
JOURNAL_POLL_SECONDS = 2
RECORDS_POLL_SECONDS = 5
#: How long the plan may hold the gauge at 0 before the twin is placed (the
#: kernel window's setup, the oracle gate, the product's baseline), and how
#: long it holds the XID after the change: a held step outlasting its read
#: bound passes through (the gauge vanishes), which changes no verdict once
#: the change was recorded.
BASELINE_HOLD_SECONDS = 480
TWIN_HOLD_SECONDS = 420
#: The product's record of the gauge baseline after the repoint.
BASELINE_BUDGET_SECONDS = 180
#: An earlier report of the XID on the GPU within this reach could pair with
#: this run's: ``xid_cross_source.candidate_window`` reads two twin windows.
QUIET_SECONDS = 2 * TWIN_WINDOW_SECONDS
#: The reviewed kernel text of XID 154 the decode reads: the pinned catalog's
#: own example (``nvidia-xid-catalog-610.generated.yaml`` XID 154
#: triggerConditions: "The string will be similar to ""Xid 154 GPU recovery
#: action changed from 0x0 (None) to 0x2 (Node Reboot Required)"" where the
#: expected values of the text are: ""None"", ""Drain P2P"", ...") with the
#: Drain P2P label -- the line ``docs/components/xid154-recovery-e2e.md`` and
#: the product's tests replay; ``policy.models.parse_xid154_action`` reads only
#: the parenthesised label after "to".
DRAIN_P2P_TEXT = "GPU recovery action changed from 0x0 (None) to 0x3 (Drain P2P)"
#: A HyperPod node's exec channel through the product's chain. Every node-side
#: step of the runner is one ``kubectl exec`` into a probe Pod, served by the
#: node's own kubelet, and ``QUIESCE_GPU_SERVICES`` stops kubelet on a HyperPod
#: node until ``RESTORE_GPU_SERVICES`` (``general_live_gpu_replay_close``); the
#: kernel collector is no quiesced service, so a line written once kubelet is
#: back reaches the product at once. Live 2026-11-02 (R16, HyperPod), the XID
#: 48 chain claimed QUIESCE_GPU_SERVICES 6 s (``xid_twin_reset_gpu``) and 15 s
#: (``xid_twin_unbound_job``, its job's STOP first) after the scrape that served
#: the change, and kubelet was back about 74 s and 83 s after it; the journal
#: read and the kernel line's exec in between came back empty while the Node
#: still read Ready (the node-monitor grace period), and both runs ended
#: INCONCLUSIVE "harness fixture failed" with nothing graded. While the node
#: cannot answer, the line is retried this often, each try bounded (a refused
#: dial fails at once; a stream the stop cuts can hang), until the twin window
#: with margin closes.
ARM_RETRY_SECONDS = 2
ARM_RETRY_TIMEOUT_SECONDS = 30
#: How long a lost call's line may take to reach the product's records.
LOST_LINE_SECONDS = 5
#: The product's record of the change, polled this often where the quiesce
#: follows the change within seconds (HyperPod, an immediate line).
FAST_RECORDS_POLL_SECONDS = 1


@dataclass
class Delivery:
    """When the harness placed each report, on the node's own clock (epoch)."""

    order: str
    kernel_written_at: float | None = None
    gauge_served_at: float | None = None
    advanced_at_read: int | None = None
    repeat_written_at: float | None = None
    problems: list[str] = field(default_factory=list)
    #: Where ``gauge_served_at`` comes from: the replay journal's read, or --
    #: when the node could not answer the journal read -- the product's own
    #: record of the change (its ``observed_at``, the scrape's moment).
    served_from: str = "journal"
    #: Where ``kernel_written_at`` comes from: the probe's receipt (``probe``),
    #: a retry's once the node answered again (``probe-retry``), or the
    #: product's record of a line a lost call did write (``product record``).
    written_from: str = "probe"

    def record(self) -> dict[str, Any]:
        lag = (
            self.gauge_served_at - self.kernel_written_at
            if self.gauge_served_at is not None and self.kernel_written_at is not None
            else None
        )
        return {
            "order": self.order,
            "kernel_written_at": self.kernel_written_at,
            "gauge_served_at": self.gauge_served_at,
            "gauge_minus_kernel_seconds": lag,
            "advanced_at_read": self.advanced_at_read,
            "repeat_written_at": self.repeat_written_at,
            "repeat_after_gauge_seconds": (
                self.repeat_written_at - self.gauge_served_at
                if self.repeat_written_at is not None
                and self.gauge_served_at is not None
                else None
            ),
            "problems": list(self.problems),
            "served_from": self.served_from,
            "written_from": self.written_from,
        }


def kernel_tail(xid: int, label: str, run_id: str) -> str:
    """The reviewed kernel text after ``<xid>,`` (it names the run)."""

    if xid == 154:
        return f"{DRAIN_P2P_TEXT}, gfrs-live {label} {run_id} reviewed kernel-source replay"
    message = xid_message(xid, "0000:00:00", run_id=run_id, label=label)
    return message.split(f": {xid}, ", 1)[1]


def kernel_line(xid: int, label: str, pci_bdf: str, run_id: str) -> str:
    return f"NVRM: Xid (PCI:{pci_bdf}): {xid}, " + kernel_tail(xid, label, run_id)


def twin_plan(
    gpu: str, xid: int, *, interval: int, hold_seconds: int
) -> dict[str, Any]:
    """The gauge at 0 from the repointed collector's first scrape (its
    initial-baseline delivery gives the product the baseline a change needs),
    held until the runner places the change; then the XID, held."""

    return plan(
        TWIN_PROFILE,
        step(
            held_reads(interval, BASELINE_HOLD_SECONDS), put(GAUGE, [gpu], 0), hold=True
        ),
        step(held_reads(interval, hold_seconds), put(GAUGE, [gpu], xid), hold=True),
        absent_when_healthy=[GAUGE],
    )


def kernel_records(
    xid: int, label: str, pci_bdf: str, run_id: str, *, repeat: bool
) -> list[dict[str, Any]]:
    """Record 0 the twin's line, record 1 (negative control) its repeat."""

    records = [
        reviewed_record(
            kernel_line(xid, label, pci_bdf, run_id),
            transport="kernel",
            emissions=KMSG_EMISSIONS,
            label=label,
        )
    ]
    if repeat:
        records.append(
            reviewed_record(
                kernel_line(xid, f"{label}-repeat", pci_bdf, run_id),
                transport="kernel",
                emissions=KMSG_EMISSIONS,
                label=f"{label}-repeat",
            )
        )
    return records


def close_kernel_window(
    ctx: VariantContext, replay: KernelLogReplay, boot_id: str
) -> dict[str, Any]:
    """Close the kmsg window and remove its probe; a probe the product took
    away (HyperPod's quiesce stops kubelet) is waited for inside the budget,
    and a reboot closes the window by itself (its drop-in and state are in
    ``/run``). Unproven is an error the runner's teardown records."""

    receipts: dict[str, Any] = {}
    try:
        return replay.cleanup()
    except HostProbeError as error:
        judged = close_after_probe_loss(
            ctx,
            error,
            run_close=replay.cleanup,
            node_boot_id=boot_id,
            receipts=receipts,
        )
        ctx.evidence.write("kernel-replay-close", receipts)
        if judged is None:
            raise RuntimeError(
                "kernel replay window close unproven: "
                + unreachable_detail(receipts.get("close_after_probe_loss") or [])
            ) from error
        return judged


def open_kernel_window(
    ctx: VariantContext, records: list[dict[str, Any]], *, boot_id: str
) -> KernelLogReplay:
    """Install the reviewed lines and repoint the kernel collector (not armed)."""

    host = ctx.host_probe(
        "kernel-replay-kmsg",
        script=PROBE_SCRIPT,
        deadline_seconds=ctx.spec.max_minutes * 60 + 1200,
    )
    replay = KernelLogReplay(
        ctx.regional,
        node=ctx.settings.node,
        image=ctx.settings.host_probe_image,
        case_id=ctx.spec.key,
        run_id=ctx.run_id,
        case_dir=ctx.case_dir,
        mode="kmsg",
        active_deadline_seconds=ctx.spec.max_minutes * 60 + 1200,
        host=host,
    )
    ctx.cleanup.push(
        "kernel-replay-kmsg", lambda: close_kernel_window(ctx, replay, boot_id)
    )
    replay.create()
    installed = replay.install(records)
    opened = replay.open(
        approved=True, restore_seconds=min(max(60, ctx.spec.max_minutes * 60), 5400)
    )
    ctx.evidence.write(
        "kernel-twin-open",
        {
            "installed": installed,
            "opened": opened,
            "records_sha256": records_sha256(records),
            "messages": [item["message"] for item in records],
        },
    )
    return replay


def close_kernel_early(
    ctx: VariantContext, replay: KernelLogReplay, boot_id: str
) -> dict[str, Any]:
    """The real ``/dev/kmsg`` is unread while the window is open: close it as
    soon as the product has the line."""

    receipt = close_kernel_window(ctx, replay, boot_id)
    ctx.cleanup.discard("kernel-replay-kmsg")
    ctx.evidence.write("kernel-twin-close", {"receipt": receipt})
    return receipt


# --------------------------------------------------------------------------- #
# Placing the reports on the node's clock
# --------------------------------------------------------------------------- #
def gauge_key(gpu: str) -> str:
    return f"{gpu}/{GAUGE}"


def served(record: dict[str, Any], key: str) -> float | None:
    value = (record.get("served") or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def wait_read(
    ctx: VariantContext,
    replay: DcgmReplay,
    accept: Callable[[dict[str, Any]], bool],
    *,
    timeout_seconds: float,
    name: str,
    what: str,
) -> dict[str, Any]:
    """The first armed read of the journal ``accept`` takes; INCONCLUSIVE on
    an abort or when none comes in time (the collector did not scrape)."""

    def probe() -> dict[str, Any] | None:
        replay.audit()
        for record in replay.records:
            if record.get("aborted"):
                raise VariantInconclusive(f"the replay aborted: {record['aborted']}")
        for record in replay.armed_reads():
            if accept(record):
                return record
        return None

    found = ctx.wait_until(
        probe,
        timeout_seconds=timeout_seconds,
        poll_seconds=JOURNAL_POLL_SECONDS,
        name=name,
    )
    if found is None:
        reads = replay.armed_reads()
        raise VariantInconclusive(
            f"the replay journal never showed {what} within {timeout_seconds:.0f} s "
            f"({len(reads)} armed reads, the last at plan step "
            f"{reads[-1].get('step') if reads else None})"
        )
    return found


def wait_until_epoch(ctx: VariantContext, epoch: float, *, name: str) -> None:
    """Sleep on the context clock until the node-clock moment ``epoch``."""

    moment = datetime.fromtimestamp(epoch, timezone.utc)
    while True:
        remaining = (moment - ctx.clock()).total_seconds()
        if remaining <= 0:
            return
        require_action_time(60)
        ctx.sleep(min(5.0, remaining))


def wait_gauge_baseline(
    ctx: VariantContext, since: datetime, gpu: str
) -> dict[str, Any]:
    """The product accepted the gauge at 0 for the GPU since the run began.

    ``store.observe_xid_metric`` mints a change event only against a stored
    baseline; a first-seen XID value is baselined silently, so without this
    the twin's change would be no event at all.
    """

    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = gpu_state(
            ctx.regional,
            ctx.settings.regional.cluster_id,
            ctx.settings.node,
            since,
            names=[GAUGE],
        )
        for item in last.get("latest") or []:
            observed = seconds_between(item.get("observed_at"), since.isoformat())
            if (
                item.get("gpu") == gpu
                and item.get("name") == GAUGE
                and item.get("value") == 0
                and observed is not None
                and observed >= 0
            ):
                return item
        return None

    found = ctx.wait_until(
        accept,
        timeout_seconds=BASELINE_BUDGET_SECONDS,
        poll_seconds=RECORDS_POLL_SECONDS,
        name="gauge-baseline",
    )
    ctx.evidence.write("gauge-baseline", {"accepted": found, "state": last})
    if found is None:
        raise VariantInconclusive(
            f"the product never recorded the gauge at 0 for {gpu} after the repoint; "
            "a later change would be a first-seen value it baselines silently"
        )
    return found


def wait_records(
    ctx: VariantContext,
    since: datetime,
    xid: int,
    need: Callable[[dict[str, Any]], bool],
    *,
    timeout_seconds: float,
    name: str,
    poll_seconds: float = RECORDS_POLL_SECONDS,
) -> dict[str, Any]:
    """Poll the twin records until ``need`` holds (the last read otherwise)."""

    last: dict[str, Any] = {}

    def accept() -> dict[str, Any] | None:
        nonlocal last
        last = twin_records(ctx, since, xid)
        return last if need(last) else None

    value = ctx.wait_until(
        accept,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
        name=name,
    )
    return value if value is not None else last


def quiesce_stops_exec(ctx: VariantContext) -> bool:
    """Whether the product's quiesce takes the node's exec channel: kubelet
    stops on a HyperPod node, an eks-managed one keeps it."""

    labels = ctx.regional.node_metadata(ctx.settings.node).get("labels") or {}
    return node_provider(labels) == "hp"


def unreachable(error: BaseException) -> bool:
    """The node never answered the probe (its transport), not a refused host
    action -- the shape a stopped kubelet leaves."""

    return isinstance(error, (HostProbeTransportError, HostProbeMissingResponseError))


def recorded_epoch(event: dict[str, Any], what: str) -> float:
    """A report's ``observed_at`` as an epoch (the node's clock: the scrape's
    moment, or the kernel line's own timestamp)."""

    moment = parse_time(event.get("observed_at"))
    if moment is None:
        raise VariantInconclusive(f"the product's record of {what} has no observed_at")
    return moment.timestamp()


def arm_kernel_line(
    ctx: VariantContext,
    kernel: KernelLogReplay,
    *,
    since: datetime,
    gpu: str,
    xid: int,
    deadline: float,
) -> dict[str, Any]:
    """Write the twin's kernel line (record 0), through a quiesce that took the
    node's exec channel.

    A call the node never answered (``unreachable``) may still have written
    the line: the product's records are read first, and a line they hold is
    taken from them (its ``observed_at`` as the written moment). Else the line
    is retried every ``ARM_RETRY_SECONDS`` while the Node reads Ready, until
    ``deadline`` (the node's clock: the twin window with margin). The record
    keeps one kmsg sequence number, so a line written twice is one event. A
    refused host action is raised; a deadline that passes is INCONCLUSIVE,
    named, with every attempt in ``twin-arm-retry``.
    """

    try:
        return kernel.arm(index=0)
    except HostProbeError as error:
        if not unreachable(error):
            raise
        first = f"{type(error).__name__}: {error}"
    attempts: list[dict[str, Any]] = []
    document = {"first_error": first, "deadline": deadline, "attempts": attempts}

    def written_by_lost_call() -> dict[str, Any] | None:
        lines = reports(twin_records(ctx, since, xid), gpu=gpu, source=KERNEL_SOURCE)
        if not lines:
            return None
        return {
            "written_at": recorded_epoch(lines[0], "the kernel line"),
            "written_from": "product record",
            "event_id": lines[0].get("event_id"),
        }

    ctx.sleep(LOST_LINE_SECONDS)
    found = written_by_lost_call()
    while found is None and ctx.clock().timestamp() < deadline:
        facts = node_facts(ctx)
        entry: dict[str, Any] = {
            "observed_at": ctx.clock().isoformat(),
            "ready": facts.ready,
            "ready_since": facts.ready_since,
        }
        attempts.append(entry)
        if facts.ready == "True":
            try:
                receipt = kernel.arm(index=0, timeout=ARM_RETRY_TIMEOUT_SECONDS)
            except HostProbeError as error:
                if not unreachable(error):
                    raise
                entry["arm"] = f"{type(error).__name__}: {error}"
            else:
                entry["arm"] = "ok"
                found = {**receipt, "written_from": "probe-retry"}
                break
        ctx.evidence.write("twin-arm-retry", document)
        ctx.sleep(ARM_RETRY_SECONDS)
    if found is None:
        found = written_by_lost_call()
    document["outcome"] = found.get("written_from") if found else "window closed"
    ctx.evidence.write("twin-arm-retry", document)
    if found is None:
        raise VariantInconclusive(
            "the twin's kernel line could not be written inside the twin window "
            f"with margin: the node did not answer its probe ({first}; "
            f"{len(attempts)} retries while kubelet was away -- HyperPod's "
            "QUIESCE_GPU_SERVICES stops it until RESTORE_GPU_SERVICES) and the "
            "product holds no line of a lost call; the change alone cannot show "
            "two reports of one XID as one occurrence"
        )
    return found


def kernel_written_at(receipt: dict[str, Any]) -> float:
    value = receipt.get("written_at")
    if not isinstance(value, (int, float)):
        raise VariantInconclusive(
            "the kernel replay probe did not report when it wrote the line "
            "(an older probe copy?); the twin cannot be timed"
        )
    return float(value)


def kernel_never_arrived(replay: KernelLogReplay, what: str) -> VariantInconclusive:
    """Why a written line never reached the product, from the probe's oracle."""

    try:
        injected = dict(replay.read_back().get("injected") or {})
    except HostProbeError as error:
        return VariantInconclusive(f"{what} never reached the product ({error})")
    followed = kmsg_follow_error(injected)
    if followed:
        return VariantInconclusive(f"{what}: {followed}")
    return VariantInconclusive(
        f"{what} never reached the product (the fixture holds "
        f"{injected.get('fixture_lines')} lines)"
    )


def deliver_kernel_first(
    ctx: VariantContext,
    replay: DcgmReplay,
    kernel: KernelLogReplay,
    *,
    gpu: str,
    xid: int,
) -> Delivery:
    """The line, then the change on the first scrape 15-30 s after it.

    The plan holds the gauge at 0 until advanced, and an advance takes effect
    at the next scrape: so the runner advances right after the first scrape at
    or after ``line + 15 s - interval``, whose successor lands in [line + 15,
    line + 15 + interval) -- inside the production lag for any interval up to
    15 s. Both clocks are the node's (the probe's ``written_at``, the
    journal's ``at``).
    """

    delivery = Delivery(order="kernel-first")
    receipt = kernel.arm(index=0)
    ctx.evidence.write("kernel-twin-arm", receipt)
    delivery.kernel_written_at = written = kernel_written_at(receipt)
    interval = max(1, int(replay.interval))
    threshold = written + PRODUCTION_LAG_SECONDS[0] - interval
    key = gauge_key(gpu)
    before = wait_read(
        ctx,
        replay,
        lambda record: float(record.get("at") or 0.0) >= threshold
        and served(record, key) == 0.0,
        timeout_seconds=2 * interval + 30,
        name="twin-scrape",
        what=f"a scrape serving the baseline after {threshold:.0f}",
    )
    replay.advance()
    delivery.advanced_at_read = int(before.get("read") or 0)
    change = wait_read(
        ctx,
        replay,
        lambda record: served(record, key) == float(xid),
        timeout_seconds=2 * interval + 30,
        name="twin-change",
        what=f"the change of {key} to {xid}",
    )
    delivery.gauge_served_at = served_at = float(change.get("at") or 0.0)
    lag = served_at - written
    low, high = PRODUCTION_LAG_SECONDS
    if not low <= lag <= high:
        delivery.problems.append(
            f"the change was served {lag:.1f} s after the kernel line, outside the "
            f"{low}-{high} s production lag the variant places it in"
        )
    return delivery


def deliver_gauge_first(
    ctx: VariantContext,
    replay: DcgmReplay,
    kernel: KernelLogReplay,
    *,
    since: datetime,
    gpu: str,
    xid: int,
    delay_seconds: int,
) -> Delivery:
    """The change, recorded by the product, then the line ``delay_seconds``
    after it was served -- inside the twin window by construction.

    The served moment is the replay journal's read of the change. On a
    HyperPod node an immediate line (``delay_seconds`` 0: XID 48, whose chain
    quiesces the node within seconds of the change's decision) does not wait
    for that read: the product's own record of the change -- what the line
    must follow anyway -- gives the moment (its ``observed_at``, the scrape's),
    and the journal is read at the close. A journal read the node cannot
    answer falls back the same way, and the line goes through
    ``arm_kernel_line``. The plain-EKS path is unchanged: its quiesce keeps
    kubelet.
    """

    delivery = Delivery(order="gauge-first")
    interval = max(1, int(replay.interval))
    key = gauge_key(gpu)
    immediate = delay_seconds == 0 and quiesce_stops_exec(ctx)
    replay.advance()
    lost = ""
    if not immediate:
        try:
            change = wait_read(
                ctx,
                replay,
                lambda record: served(record, key) == float(xid),
                timeout_seconds=2 * interval + 30,
                name="twin-change",
                what=f"the change of {key} to {xid}",
            )
        except HostProbeError as error:
            if not unreachable(error):
                raise
            lost = f"{type(error).__name__}: {error}"
        else:
            delivery.advanced_at_read = int(change.get("read") or 0)
            delivery.gauge_served_at = float(change.get("at") or 0.0)
    # The line must arrive after the product holds the change: the journal
    # entry precedes the collector's own delivery of that scrape.
    seen = wait_records(
        ctx,
        since,
        xid,
        lambda records: bool(reports(records, gpu=gpu, source=GAUGE_SOURCE)),
        timeout_seconds=2 * interval + 60,
        name="twin-gauge-recorded",
        poll_seconds=FAST_RECORDS_POLL_SECONDS if immediate else RECORDS_POLL_SECONDS,
    )
    gauges = reports(seen, gpu=gpu, source=GAUGE_SOURCE)
    if not gauges:
        raise VariantInconclusive(
            f"the product recorded no field-230 change of XID {xid} on {gpu} after "
            "the replay served it"
            + (f" (the replay journal could not be read: {lost})" if lost else "")
        )
    if delivery.gauge_served_at is None:
        delivery.gauge_served_at = recorded_epoch(gauges[0], "the change")
        delivery.served_from = "product record"
        ctx.evidence.write(
            "twin-change-deferred",
            {
                "reason": lost
                or "an immediate line on a node whose quiesce stops "
                "kubelet does not wait for the journal read",
                "event_id": gauges[0].get("event_id"),
                "observed_at": gauges[0].get("observed_at"),
                "journal": "read at the close",
            },
        )
    served_at = delivery.gauge_served_at
    wait_until_epoch(ctx, served_at + delay_seconds, name="twin-delay")
    receipt = arm_kernel_line(
        ctx,
        kernel,
        since=since,
        gpu=gpu,
        xid=xid,
        deadline=served_at + TWIN_WINDOW_SECONDS - REPEAT_MARGIN_SECONDS,
    )
    ctx.evidence.write("kernel-twin-arm", receipt)
    delivery.written_from = str(receipt.get("written_from") or "probe")
    delivery.kernel_written_at = written = kernel_written_at(receipt)
    gap = written - served_at
    if not 0 <= gap <= TWIN_WINDOW_SECONDS - REPEAT_MARGIN_SECONDS:
        delivery.problems.append(
            f"the kernel line was written {gap:.1f} s after the change, not inside "
            f"the {TWIN_WINDOW_SECONDS} s twin window with margin"
        )
    return delivery


def deliver_repeat(
    ctx: VariantContext, kernel: KernelLogReplay, delivery: Delivery
) -> None:
    """The negative control: the line again, past the window, gauge unchanged."""

    if delivery.gauge_served_at is None:
        raise VariantInconclusive("the change was never served; no repeat placed")
    due = delivery.gauge_served_at + TWIN_WINDOW_SECONDS + REPEAT_MARGIN_SECONDS
    wait_until_epoch(ctx, due, name="twin-repeat-due")
    receipt = kernel.arm(index=1)
    ctx.evidence.write("kernel-twin-repeat-arm", receipt)
    delivery.repeat_written_at = kernel_written_at(receipt)


def quiet_before_twin(ctx: VariantContext, xid: int, gpu: str) -> dict[str, Any]:
    """No other report of the XID on the GPU within the product's pairing
    reach (``xid_cross_source.candidate_window``: two twin windows) before this
    run places its own: an earlier run's line left unpaired would be taken as
    this change's twin. Waited out before anything is deployed."""

    now = ctx.clock()
    recent = twin_records(ctx, now - timedelta(seconds=QUIET_SECONDS), xid)
    on_gpu = [
        item for item in recent.get("events") or [] if item.get("gpu_uuid") == gpu
    ]
    if not on_gpu:
        return {"recent_reports": []}
    newest = max(
        moment
        for item in on_gpu
        if (moment := parse_time(item.get("observed_at"))) is not None
    )
    due = newest + timedelta(seconds=QUIET_SECONDS + 5)
    wait_until_epoch(ctx, due.timestamp(), name="twin-quiet")
    document = {
        "recent_reports": [item.get("event_id") for item in on_gpu],
        "waited_until": due.isoformat(),
    }
    ctx.evidence.write("twin-quiet", document)
    return document


__all__ = [
    "BASELINE_BUDGET_SECONDS",
    "BASELINE_HOLD_SECONDS",
    "DRAIN_P2P_TEXT",
    "GAUGE_FIRST_DELAY_SECONDS",
    "JOURNAL_POLL_SECONDS",
    "KMSG_EMISSIONS",
    "PRODUCTION_LAG_SECONDS",
    "QUIET_SECONDS",
    "RECORDS_POLL_SECONDS",
    "REPEAT_MARGIN_SECONDS",
    "TWIN_HOLD_SECONDS",
    "TWIN_PROFILE",
    "Delivery",
    "close_kernel_early",
    "close_kernel_window",
    "deliver_gauge_first",
    "deliver_kernel_first",
    "deliver_repeat",
    "gauge_key",
    "kernel_line",
    "kernel_never_arrived",
    "kernel_records",
    "kernel_tail",
    "kernel_written_at",
    "open_kernel_window",
    "quiet_before_twin",
    "served",
    "twin_plan",
    "wait_gauge_baseline",
    "wait_read",
    "wait_records",
    "wait_until_epoch",
]
