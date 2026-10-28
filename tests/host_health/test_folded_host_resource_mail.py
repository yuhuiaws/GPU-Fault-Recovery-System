"""A host-resource signal folded into another fault's incident is mailed (b215).

The host-resource mail (``host-resource-event-zh-v1``, category
``HEALTH_TREND``) is how the operator hears of a sustained host-resource
signal: one mail per node and signal, the node's GPUs aggregated, deduplicated
per cluster/node/metric/workloads/signal and cooldown window. Ingestion sent it
only when the finding minted its own incident (``app/ingest/node_health.py``).
A finding the product routes into an incident another fault opened mailed
nothing:

* while a managed attempt's recovery is in flight (an XID 13 restart, an XID
  48 reset, an XID 79 reboot) the attempt grouping folds the finding into that
  recovery's ``GPU_FAULT_GROUP`` record -- and the recovery's plan takes on the
  finding's own FREEZE_EVIDENCE + VALIDATE_HOST, a new gate on the recovery;
* on a node the product is rebooting, the reboot's record absorbs it.

The XID's own mail went out before the fold and says nothing of it. Only a
re-arm of the same node's same signal on an incident that signal already
reached stays silent, as a re-arm on the signal's own incident always did.

Everything goes through the product's ingestion (host telemetry and XIDs posted
to the control-plane API over an in-memory store) on a frozen receive clock:
sustained windows run on the time the control plane accepted each batch.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.app.ingest import telemetry as telemetry_ingest
from gpu_fault.host_health import HostMetricSample, HostTelemetryBatch
from gpu_fault.models import FaultIncident, WorkflowOperation
from tests._builders import host_telemetry_batch
from tests.host_health._support import RecordingNotifier, observe_running_attempt
from tests.host_health.test_grouped_node_health_incidents import (
    WORKLOAD,
    at,
    ingest,
    routed,
    running_attempt,
    xid_recovery,
)
from tests.host_health.test_rebooting_member_failure_domain import rebooting

#: The host-resource rules' sample per metric: a reading past the threshold,
#: the device it is graded on, and a healthy reading.
READINGS = {
    "memory_available_percent": (2.0, None, 50.0),
    "local_filesystem_used_percent": (95.0, "/", 40.0),
}
SIGNALS = {
    "memory_available_percent": "LOW_MEMORY_AVAILABLE",
    "local_filesystem_used_percent": "HIGH_LOCAL_FILESYSTEM_USAGE",
}


class Clock:
    """The control plane's receive clock for host telemetry (F-M2)."""

    def __init__(self) -> None:
        self.now = at(0)

    def to(self, seconds: float) -> None:
        self.now = at(seconds)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[Clock]:
    for name in (
        "GPU_FAULT_MEMORY_PRESSURE_DURATION_SECONDS",
        "GPU_FAULT_LOCAL_FILESYSTEM_DURATION_SECONDS",
        "GPU_FAULT_LOW_UTILIZATION_DURATION_SECONDS",
    ):
        monkeypatch.setenv(name, "15")
    current = Clock()

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return current.now if tz is not None else current.now.replace(tzinfo=None)

    monkeypatch.setattr(telemetry_ingest, "datetime", _Frozen)
    yield current


def reading(
    batch_id: str, node: str, seconds: float, samples: list[HostMetricSample]
) -> HostTelemetryBatch:
    return host_telemetry_batch(
        batch_id,
        at(seconds),
        samples,
        node_id=node,
        workload_state="ACTIVE",
        affected_workload_ids=[WORKLOAD],
        runtime_profile_version="simulated-v1",
    )


def sustained(
    context: ApplicationContext,
    clock: Clock,
    metric: str,
    *,
    node: str = "node-a",
    start: float,
    prefix: str = "low",
) -> list[str]:
    """The metric past its threshold for one window: the finding's event ids."""

    value, device, _healthy = READINGS[metric]
    sample = HostMetricSample(name=metric, value=value, unit="percent", device=device)
    clock.to(start)
    assert ingest(context, reading(f"{prefix}-1", node, start, [sample])) == []
    clock.to(start + 15)
    return ingest(context, reading(f"{prefix}-2", node, start + 15, [sample]))


def healthy(
    context: ApplicationContext, clock: Clock, metric: str, seconds: float
) -> None:
    _value, device, normal = READINGS[metric]
    clock.to(seconds)
    sample = HostMetricSample(name=metric, value=normal, unit="percent", device=device)
    assert ingest(context, reading(f"ok-{seconds}", "node-a", seconds, [sample])) == []


def host_resource_mails(notifier: RecordingNotifier) -> list[Any]:
    return [
        item
        for item in notifier.notifications
        if item.subject.startswith("[GPU FAULT][host resource risk]")
    ]


def host_validation_of(
    context: ApplicationContext, incident: FaultIncident, metric: str
) -> list[Any]:
    workflow = context.store.get_workflow(
        context.store.get_incident(incident.incident_id).workflow_request_id
    )
    return [
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.VALIDATE_HOST
        and step.parameters.get("host_diagnostic_probe", {}).get("metric") == metric
    ]


@pytest.mark.parametrize("xid", [13, 48, 79])
def test_a_signal_folded_into_an_attempts_recovery_is_mailed(
    clock: Clock, xid: int
) -> None:
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    running_attempt(context, "node-a")
    recovery = xid_recovery(context, xid)
    assert host_resource_mails(notifier) == [], "the XID's own mail is no host mail"

    [event] = sustained(context, clock, "memory_available_percent", start=1)

    assert routed(context, event).incident_id == recovery.incident_id, (
        "the finding joins the attempt's recovery, under the XID's identity"
    )
    assert host_validation_of(context, recovery, "memory_available_percent"), (
        "the recovery's plan took on the finding's VALIDATE_HOST"
    )
    [mail] = host_resource_mails(notifier)
    assert mail.category == "HEALTH_TREND"
    assert mail.incident_id == recovery.incident_id, "it names the incident it joined"
    body = mail.body_text
    assert "Signal: LOW_MEMORY_AVAILABLE" in body
    assert "Metric: memory_available_percent" in body
    assert f"Incident: {recovery.incident_id}" in body
    assert f"Event ID: {event}" in body
    assert "VALIDATE_HOST" in body.split("Workflow steps:")[1].splitlines()[0], (
        "the workflow steps it lists are the recovery's, with the finding's gate"
    )


def test_a_signal_folded_again_into_the_recovery_it_reached_is_not_mailed_again(
    clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A one-minute cooldown: the re-arm lands in another deduplication bucket,
    # so only the "already told on this incident" rule keeps it silent.
    monkeypatch.setenv("GPU_FAULT_HOST_NOTIFICATION_COOLDOWN_SECONDS", "60")
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    running_attempt(context, "node-a")
    recovery = xid_recovery(context, 13)
    [first] = sustained(context, clock, "memory_available_percent", start=1)
    healthy(context, clock, "memory_available_percent", 31)
    running_attempt(context, "node-a", observed_at=at(120))  # the watcher's next look

    [second] = sustained(
        context, clock, "memory_available_percent", start=121, prefix="again"
    )

    assert second != first
    assert {
        routed(context, first).incident_id,
        routed(context, second).incident_id,
    } == {recovery.incident_id}
    assert len(host_resource_mails(notifier)) == 1, (
        "the re-armed signal reached an incident the operator was told of"
    )


def test_each_signal_folded_into_one_recovery_is_mailed_once(clock: Clock) -> None:
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    running_attempt(context, "node-a")
    recovery = xid_recovery(context, 13)

    [memory] = sustained(context, clock, "memory_available_percent", start=1)
    [disk] = sustained(
        context, clock, "local_filesystem_used_percent", start=40, prefix="disk"
    )

    assert {routed(context, memory).incident_id, routed(context, disk).incident_id} == {
        recovery.incident_id
    }
    mails = host_resource_mails(notifier)
    assert sorted(
        line.removeprefix("- Signal: ")
        for mail in mails
        for line in mail.body_text.splitlines()
        if line.startswith("- Signal: ")
    ) == sorted(SIGNALS.values()), [mail.body_text for mail in mails]
    assert {mail.incident_id for mail in mails} == {recovery.incident_id}


def test_the_node_of_a_two_node_recovery_each_gets_its_mail(clock: Clock) -> None:
    # The mail is per node (GF-REGIONAL-NOTIFY-005): node-b's signal on the
    # recovery node-a's same signal already reached is node-b's first word.
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    running_attempt(context, "node-a", "node-b")
    recovery = xid_recovery(context, 13)

    [on_a] = sustained(context, clock, "memory_available_percent", start=1)
    [on_b] = sustained(
        context, clock, "memory_available_percent", node="node-b", start=1, prefix="b"
    )

    assert {routed(context, on_a).incident_id, routed(context, on_b).incident_id} == {
        recovery.incident_id
    }
    mails = host_resource_mails(notifier)
    assert sorted(
        line.removeprefix("- Node: ")
        for mail in mails
        for line in mail.body_text.splitlines()
        if line.startswith("- Node: ")
    ) == ["node-a", "node-b"], [mail.body_text for mail in mails]


def test_the_gpus_of_one_node_folded_in_one_batch_share_one_mail(clock: Clock) -> None:
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    observe_running_attempt(context.store, at(0), gpus=("GPU-a", "GPU-b"))
    recovery = xid_recovery(context, 13, gpu="GPU-a")
    samples = [
        HostMetricSample(
            name="host_gpu_utilization_percent", value=0, unit="percent", device=gpu
        )
        for gpu in ("GPU-a", "GPU-b")
    ]
    clock.to(1)
    assert ingest(context, reading("idle-1", "node-a", 1, samples)) == []
    clock.to(16)

    events = ingest(context, reading("idle-2", "node-a", 16, samples))

    assert len(events) == 2
    assert {routed(context, event).incident_id for event in events} == {
        recovery.incident_id
    }
    [mail] = host_resource_mails(notifier)
    assert "GPU-a, GPU-b" in mail.body_text, "one mail for the node's two GPUs"


def test_a_signal_on_a_node_the_product_is_rebooting_is_mailed(clock: Clock) -> None:
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    reboot = rebooting(context)

    [event] = sustained(context, clock, "local_filesystem_used_percent", start=1)

    assert routed(context, event).incident_id == reboot.incident_id, (
        "the reboot's record absorbs the finding: no chain of its own"
    )
    [mail] = host_resource_mails(notifier)
    assert mail.incident_id == reboot.incident_id
    assert "Signal: HIGH_LOCAL_FILESYSTEM_USAGE" in mail.body_text
    assert "RESTART_NODE" in mail.body_text, "the workflow it lists is the reboot's"


def test_a_rearm_the_reboot_absorbs_while_the_signals_own_incident_is_open_is_silent(
    clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The node's own incident of the signal is still with the operator when
    # the product reboots the node for another fault and the signal re-arms:
    # the reboot window takes the finding before the same-signal absorb, but
    # the operator was told when that incident opened. A one-minute cooldown
    # puts the re-arm in another deduplication bucket.
    monkeypatch.setenv("GPU_FAULT_HOST_NOTIFICATION_COOLDOWN_SECONDS", "60")
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    [own_event] = sustained(context, clock, "local_filesystem_used_percent", start=1)
    own = routed(context, own_event)
    assert own.event_id == own_event and own.state.value == "ACTION_PENDING"
    assert len(host_resource_mails(notifier)) == 1, "mailed when it opened"
    reboot = rebooting(context)
    healthy(context, clock, "local_filesystem_used_percent", 31)

    [again] = sustained(
        context, clock, "local_filesystem_used_percent", start=121, prefix="again"
    )

    assert routed(context, again).incident_id == reboot.incident_id
    assert len(host_resource_mails(notifier)) == 1, (
        "the signal's own incident on the node is still open: no second mail"
    )
