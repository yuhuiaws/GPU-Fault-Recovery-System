"""GF-GENERAL-LIVE-024 ``xid_twin_*`` placement: the waits that give up and why.

``test_general_live_xid_twins`` pins the placements on their main paths (the
kernel-first lag, the repeat past the window, the line through a quiesce). The
cases here pin each place the node side stops and says why: a kmsg window whose
close stays unproven, a replay journal that aborted or never showed the
reading, a gauge baseline the product never recorded, a change record without
a moment, a retried line the probe refused, a probe that never said when it
wrote the line, a line that never reached the product (three readings of the
probe's oracle), a journal read the probe refused, a change the product never
recorded, a line written outside the twin window, a repeat with no change to
follow, and the quiet wait before a run on a GPU with recent reports.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_xid_twin_delivery as placement
from scripts.e2e.regional import general_live_xid_twins as twins
from scripts.e2e.regional.general_live_fixture import (
    HYPERPOD_LABEL,
    VariantContext,
    VariantInconclusive,
)
from scripts.e2e.regional.host_probe_fixture import (
    HostProbeError,
    HostProbeMissingResponseError,
)
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional.test_general_live_xid_twins import (
    G0,
    G1,
    NODE,
    STOPPED_KUBELET,
    QuiescedKernel,
    _event,
)

START = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
KEY = f"{G1}/xid_last_error"


def clocked(
    tmp_path: Path,
    records: dict[str, Any],
    *,
    variant: str = "xid_twin_reset_gpu",
    labels: dict[str, str] | None = None,
) -> tuple[VariantContext, dict[str, datetime]]:
    """A context on a clock that moves only when the runner sleeps; every
    store probe answers ``records``."""

    spec = next(item for item in twins.VARIANTS if item.variant == variant)
    regional = FakeRegional(
        cpu_python=lambda _script, *_arguments: records,
        node_metadata=(
            {"name": NODE, "labels": labels} if labels is not None else None
        ),
    )
    ctx = make_context(
        spec,
        tmp_path=tmp_path,
        regional=regional,
        probes={},
        settings=make_settings(spec, node=NODE),
    )
    now = {"value": START}
    ctx.clock = lambda: now["value"]

    def sleep(seconds: float) -> None:
        now["value"] += timedelta(seconds=seconds)

    ctx.sleep = sleep
    return ctx, now


class Journal:
    """A replay journal read: ``records`` as the probe returns them."""

    def __init__(self, records: list[dict[str, Any]], *, interval: int = 15) -> None:
        self.records = records
        self.interval = interval
        self.audits = 0
        self.advances = 0

    def audit(self) -> list[dict[str, Any]]:
        self.audits += 1
        return self.records

    def armed_reads(self) -> list[dict[str, Any]]:
        return [item for item in self.records if item.get("armed")]

    def advance(self) -> dict[str, Any]:
        self.advances += 1
        return {"advanced": True}


def read(index: int, value: float, *, at: float, aborted: str | None = None) -> dict:
    return {
        "read": index,
        "armed": True,
        "at": at,
        "step": 0,
        "served": {KEY: value},
        "aborted": aborted,
    }


# --------------------------------------------------------------------------- #
# the kmsg window's close
# --------------------------------------------------------------------------- #
class LostWindow:
    """A kmsg replay whose probe never answers the close."""

    def __init__(self) -> None:
        self.cleanups = 0

    def cleanup(self) -> dict[str, Any]:
        self.cleanups += 1
        raise HostProbeMissingResponseError(STOPPED_KUBELET)


def test_a_close_that_stays_unproven_is_a_runtime_error_with_the_polls(
    tmp_path: Path,
) -> None:
    ctx, _now = clocked(tmp_path, {"events": []})
    ctx.deadline = START + timedelta(seconds=60)
    window = LostWindow()

    with pytest.raises(RuntimeError) as raised:
        placement.close_kernel_window(ctx, window, "boot-1")  # type: ignore[arg-type]

    assert str(raised.value) == (
        "kernel replay window close unproven: the node did not reboot (boot id "
        "boot-1 unchanged) and the probe could not run after 1 poll (Ready=True)"
    )
    assert isinstance(raised.value.__cause__, HostProbeMissingResponseError), (
        "the unproven close keeps the probe failure that caused it"
    )
    assert window.cleanups == 2, "tried once, retried once on the settled node"
    receipts = json.loads((ctx.case_dir / "kernel-replay-close.json").read_text())
    assert receipts["close_error"].startswith("HostProbeMissingResponseError"), receipts
    assert len(receipts["close_after_probe_loss"]) == 1, receipts


# --------------------------------------------------------------------------- #
# the replay journal
# --------------------------------------------------------------------------- #
def test_an_aborted_replay_ends_the_wait_inconclusive(tmp_path: Path) -> None:
    ctx, _now = clocked(tmp_path, {"events": []})
    journal = Journal(
        [read(1, 0.0, at=1.0), read(2, 0.0, at=16.0, aborted="real XID 31 on G1")]
    )

    with pytest.raises(
        VariantInconclusive, match=r"^the replay aborted: real XID 31 on G1$"
    ):
        placement.wait_read(
            ctx,
            journal,  # type: ignore[arg-type]
            lambda record: placement.served(record, KEY) == 48.0,
            timeout_seconds=30,
            name="twin-change",
            what="the change",
        )
    assert journal.audits == 1, "an abort is final at the first read"


def test_a_journal_that_never_shows_the_reading_names_its_last_read(
    tmp_path: Path,
) -> None:
    ctx, now = clocked(tmp_path, {"events": []})
    journal = Journal([read(1, 0.0, at=1.0), read(2, 0.0, at=16.0)])

    with pytest.raises(VariantInconclusive) as raised:
        placement.wait_read(
            ctx,
            journal,  # type: ignore[arg-type]
            lambda record: placement.served(record, KEY) == 48.0,
            timeout_seconds=6,
            name="twin-change",
            what=f"the change of {KEY} to 48",
        )

    assert str(raised.value) == (
        f"the replay journal never showed the change of {KEY} to 48 within 6 s "
        "(2 armed reads, the last at plan step 0)"
    )
    assert now["value"] - START == timedelta(seconds=6), "polled to the bound"


# --------------------------------------------------------------------------- #
# the product's baseline of the gauge
# --------------------------------------------------------------------------- #
def test_a_baseline_the_product_never_recorded_is_inconclusive(tmp_path: Path) -> None:
    since = START
    latest = [
        {"gpu": G0, "name": "xid_last_error", "value": 0, "observed_at": "x"},
        {
            "gpu": G1,
            "name": "xid_last_error",
            "value": 0,
            "observed_at": (since - timedelta(seconds=30)).isoformat(),
        },
        {
            "gpu": G1,
            "name": "xid_last_error",
            "value": 13,
            "observed_at": (since + timedelta(seconds=5)).isoformat(),
        },
        {"gpu": G1, "name": "gpu_temperature_c", "value": 0, "observed_at": "x"},
    ]
    ctx, _now = clocked(tmp_path, {"latest": latest})

    with pytest.raises(VariantInconclusive) as raised:
        placement.wait_gauge_baseline(ctx, since, G1)

    assert str(raised.value).startswith(
        f"the product never recorded the gauge at 0 for {G1} after the repoint"
    ), raised.value
    evidence = json.loads((ctx.case_dir / "gauge-baseline.json").read_text())
    assert evidence["accepted"] is None and evidence["state"]["latest"] == latest


def test_the_baseline_is_the_first_zero_of_the_gpu_since_the_run_began(
    tmp_path: Path,
) -> None:
    fresh = {
        "gpu": G1,
        "name": "xid_last_error",
        "value": 0,
        "observed_at": (START + timedelta(seconds=15)).isoformat(),
    }
    ctx, _now = clocked(
        tmp_path, {"latest": [{"gpu": G0, "name": "xid_last_error", "value": 0}, fresh]}
    )

    assert placement.wait_gauge_baseline(ctx, START, G1) == fresh


# --------------------------------------------------------------------------- #
# the kernel line's receipts and the probe's oracle
# --------------------------------------------------------------------------- #
def test_a_receipt_without_a_written_moment_cannot_time_the_twin() -> None:
    with pytest.raises(VariantInconclusive, match="did not report when it wrote"):
        placement.kernel_written_at({"armed": True, "written_at": "12:00:00"})
    assert placement.kernel_written_at({"written_at": 7}) == 7.0


class Oracle:
    """``read_back`` of the kmsg probe: an answer, or a probe that refuses."""

    def __init__(self, answer: dict[str, Any] | Exception) -> None:
        self.answer = answer

    def read_back(self) -> dict[str, Any]:
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def test_a_line_that_never_arrived_is_explained_by_the_probes_oracle() -> None:
    what = "the kernel line of XID 48"

    lost = placement.kernel_never_arrived(
        Oracle(HostProbeError("host probe failed (exit 1)")),  # type: ignore[arg-type]
        what,
    )
    unread = placement.kernel_never_arrived(
        Oracle(  # type: ignore[arg-type]
            {
                "injected": {
                    "consumed": False,
                    "stream_reopens": 3,
                    "fixture_bytes": 512,
                    "collector": {"fixture_open": False, "offset": 0},
                }
            }
        ),
        what,
    )
    held = placement.kernel_never_arrived(
        Oracle({"injected": {"consumed": True, "fixture_lines": 2}}),  # type: ignore[arg-type]
        what,
    )

    assert str(lost) == f"{what} never reached the product (host probe failed (exit 1))"
    assert str(unread).startswith(
        f"{what}: the kernel collector did not follow the reviewed kmsg fixture: "
        "stream reopened at EOF 3 time(s)"
    ), unread
    assert str(held) == f"{what} never reached the product (the fixture holds 2 lines)"
    assert all(
        isinstance(item, VariantInconclusive) for item in (lost, unread, held)
    ), "every explanation is a premise failure, never a product verdict"


class RefusingRetry:
    """The node never answers the first write; the retry's probe refuses."""

    def __init__(self) -> None:
        self.calls = 0

    def arm(self, *, index: int | None = None, timeout: int = 120) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            raise HostProbeMissingResponseError(STOPPED_KUBELET)
        raise HostProbeError("host probe failed (exit 1); output withheld")


def test_a_retried_line_the_probe_refuses_is_raised_not_retried_again(
    tmp_path: Path,
) -> None:
    ctx, _now = clocked(tmp_path, {"events": []})
    kernel = RefusingRetry()

    with pytest.raises(HostProbeError, match="output withheld") as raised:
        placement.arm_kernel_line(
            ctx,
            kernel,  # type: ignore[arg-type]
            since=START,
            gpu=G1,
            xid=48,
            deadline=START.timestamp() + 90,
        )

    assert not isinstance(raised.value, HostProbeMissingResponseError), (
        "the refusal is raised, not the first unanswered call"
    )
    assert kernel.calls == 2, "the refusal ends the retries"


# --------------------------------------------------------------------------- #
# deliver_gauge_first
# --------------------------------------------------------------------------- #
class RefusedJournal(Journal):
    def audit(self) -> list[dict[str, Any]]:
        raise HostProbeError("dcgm audit-read: host probe failed (exit 1)")


class LostJournal(Journal):
    def audit(self) -> list[dict[str, Any]]:
        raise HostProbeMissingResponseError(f"dcgm audit-read: {STOPPED_KUBELET}")


def change(seconds: float, **fields: Any) -> dict[str, Any]:
    event = _event("g", "DCGM_EXPORTER", seconds, incident_id="inc-g")
    event.update(fields)
    return event


def test_a_journal_read_the_probe_refuses_is_raised(tmp_path: Path) -> None:
    ctx, _now = clocked(tmp_path, {"events": [change(12)]})
    kernel = QuiescedKernel(ctx, back=0.0)

    with pytest.raises(HostProbeError, match="dcgm audit-read: host probe failed"):
        placement.deliver_gauge_first(
            ctx,
            RefusedJournal([]),  # type: ignore[arg-type]
            kernel,  # type: ignore[arg-type]
            since=START,
            gpu=G1,
            xid=48,
            delay_seconds=0,
        )
    assert kernel.calls == [], "no line follows a change the journal never showed"


def test_a_change_the_product_never_recorded_names_the_lost_journal(
    tmp_path: Path,
) -> None:
    ctx, now = clocked(tmp_path, {"events": []})
    kernel = QuiescedKernel(ctx, back=0.0)
    journal = LostJournal([])

    with pytest.raises(VariantInconclusive) as raised:
        placement.deliver_gauge_first(
            ctx,
            journal,  # type: ignore[arg-type]
            kernel,  # type: ignore[arg-type]
            since=START,
            gpu=G1,
            xid=48,
            delay_seconds=30,
        )

    assert str(raised.value) == (
        f"the product recorded no field-230 change of XID 48 on {G1} after the "
        "replay served it (the replay journal could not be read: "
        f"HostProbeMissingResponseError: dcgm audit-read: {STOPPED_KUBELET})"
    )
    assert journal.advances == 1 and kernel.calls == [], "no line is written"
    assert now["value"] - START >= timedelta(seconds=2 * 15 + 60), (
        "the product's records were polled to the bound"
    )


def test_an_immediate_line_needs_the_changes_recorded_moment(tmp_path: Path) -> None:
    ctx, _now = clocked(
        tmp_path,
        {"events": [change(12, observed_at=None)]},
        labels={HYPERPOD_LABEL: "hyperpod"},
    )
    journal = Journal([])
    kernel = QuiescedKernel(ctx, back=0.0)

    with pytest.raises(
        VariantInconclusive,
        match=r"^the product's record of the change has no observed_at$",
    ):
        placement.deliver_gauge_first(
            ctx,
            journal,  # type: ignore[arg-type]
            kernel,  # type: ignore[arg-type]
            since=START,
            gpu=G1,
            xid=48,
            delay_seconds=0,
        )
    assert journal.audits == 0, (
        "an immediate line on HyperPod does not read the journal"
    )


def test_a_line_written_before_the_change_is_a_premise_problem(tmp_path: Path) -> None:
    ctx, _now = clocked(tmp_path, {"events": [change(12)]})

    class EarlyStamp(QuiescedKernel):
        def arm(self, *, index: int | None = None, timeout: int = 120) -> dict:
            receipt = super().arm(index=index, timeout=timeout)
            return {**receipt, "written_at": START.timestamp()}

    kernel = EarlyStamp(ctx, back=0.0)

    delivery = placement.deliver_gauge_first(
        ctx,
        LostJournal([]),  # type: ignore[arg-type]
        kernel,  # type: ignore[arg-type]
        since=START,
        gpu=G1,
        xid=48,
        delay_seconds=0,
    )

    assert delivery.served_from == "product record"
    assert delivery.gauge_served_at == START.timestamp() + 12
    assert delivery.problems == [
        "the kernel line was written -12.0 s after the change, not inside the "
        "120 s twin window with margin"
    ]


# --------------------------------------------------------------------------- #
# the negative control and the quiet wait
# --------------------------------------------------------------------------- #
def test_no_repeat_is_placed_without_a_served_change(tmp_path: Path) -> None:
    ctx, _now = clocked(tmp_path, {"events": []}, variant="xid_twin_kernel_first")
    kernel = QuiescedKernel(ctx, back=0.0)

    with pytest.raises(VariantInconclusive, match="never served; no repeat placed"):
        placement.deliver_repeat(
            ctx,
            kernel,  # type: ignore[arg-type]
            placement.Delivery(order="kernel-first"),
        )
    assert kernel.calls == [], "nothing was written"


def test_a_gpu_with_recent_reports_is_waited_out_before_the_twin(
    tmp_path: Path,
) -> None:
    recent = [
        _event("k-old", "KERNEL_LOG", -100),
        _event("k-older", "KERNEL_LOG", -200),
        {**_event("k-undated", "KERNEL_LOG", 0), "observed_at": None},
        {**_event("g-other", "DCGM_EXPORTER", -10), "gpu_uuid": G0},
    ]
    ctx, now = clocked(tmp_path, {"events": recent}, variant="xid_twin_kernel_first")

    document = placement.quiet_before_twin(ctx, 48, G1)

    due = (
        START - timedelta(seconds=100) + timedelta(seconds=placement.QUIET_SECONDS + 5)
    )
    assert document == {
        "recent_reports": ["k-old", "k-older", "k-undated"],
        "waited_until": due.isoformat(),
    }
    assert now["value"] >= due, "the runner slept until the reach was quiet"
    written = json.loads((ctx.case_dir / "twin-quiet.json").read_text())
    assert written == document
