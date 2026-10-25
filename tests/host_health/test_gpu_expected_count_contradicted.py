"""A configured expected GPU count that the instance type and the node both
contradict is a configuration finding, not a lost GPU.

GF-GENERAL-LIVE-012 ``expectation_wrong``: ``GPU_FAULT_EXPECTED_GPU_COUNT=9``
on an 8-GPU type whose node lists eight GPUs reached the control plane as
``gpu_inventory_mismatch=1`` and was graded CRITICAL / REBOOT_NODE for a healthy
node. The control plane now re-reads the mismatch against its own instance-type
table before grading, whatever collector version produced the batch: the
configured count is the one reading out of three that disagrees, so the node
gets a WARNING ``gpu_expected_count_contradicted`` finding (evidence only,
operator review) and nothing destructive. A node short of the type's own count,
a driver-bound count that disagrees with the listing, an unknown type or a
batch without the labels grade exactly as before.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.host_health import (
    GPU_EXPECTED_COUNT_CONTRADICTED_METRIC,
    HostMetricSample,
    NodeHealthCategory,
    NodeHealthPolicy,
    gpu_expected_count_contradicted_rule,
    gpu_inventory_verdicts,
)
from gpu_fault.models import RecoveryAction, Severity, WorkflowOperation
from gpu_fault.store.shared.health_signals import (
    finding_health_signal_fingerprint,
    finding_health_signal_key,
)
from tests._builders import asgi_client, build_store, host_telemetry_batch
from tests.host_health._support import NOW, RecordingNotifier, observe_running_attempt

INSTANCE_TYPE = "ml.p5en.48xlarge"


def mismatch_sample(
    *,
    expected: int,
    observed: int,
    discovered: int | None = None,
    instance_type: str | None = INSTANCE_TYPE,
    value: float = 1.0,
    consecutive: str = "2",
) -> HostMetricSample:
    """What an already-deployed host collector posts after two mismatching
    ticks: no ``discovered_count`` on the listing path, no instance-type count."""

    labels = {
        "resource": "GPU",
        "expected_count": str(expected),
        "observed_count": str(observed),
        "missing_count": str(max(0, expected - observed)),
        "excess_count": str(max(0, observed - expected)),
        "consecutive_mismatch_samples": consecutive,
        "required_consecutive_samples": "2",
        "expected_gpu_count": str(expected),
    }
    if instance_type is not None:
        labels["node_instance_type"] = instance_type
    if discovered is not None:
        labels["discovered_count"] = str(discovered)
    return HostMetricSample(name="gpu_inventory_mismatch", value=value, labels=labels)


def batch(batch_id: str, samples: list[HostMetricSample], *, at=NOW):
    return host_telemetry_batch(
        batch_id,
        at,
        samples,
        workload_state="ACTIVE",
        affected_workload_ids=["training/job/job-a"],
        runtime_profile_version="simulated-v1",
    )


def post(context: ApplicationContext, payload) -> dict:
    async def scenario() -> dict:
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/collector-events/host-telemetry",
                json=payload.model_dump(mode="json"),
            )
        assert response.status_code == 200, response.text
        return response.json()

    return asyncio.run(scenario())


def test_a_mismatch_on_a_node_complete_by_its_instance_type_is_a_configuration_finding() -> (
    None
):
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    observe_running_attempt(context.store, NOW)

    body = post(context, batch("typo", [mismatch_sample(expected=9, observed=8)]))

    assert [item["metric_name"] for item in body["findings"]] == [
        GPU_EXPECTED_COUNT_CONTRADICTED_METRIC
    ], body["findings"]
    (finding,) = body["findings"]
    assert finding["severity"] == "warning"
    assert finding["recommended_action"] == "COLLECT_EVIDENCE"
    for named in ("9", "8", INSTANCE_TYPE):
        assert named in finding["reason"], finding["reason"]
    assert finding["diagnostic_parameters"]["instance_type_gpu_count"] == "8"
    (incident_id,) = body["incident_ids"]
    incident = context.store.get_incident(incident_id)
    assert incident.effective_action is RecoveryAction.COLLECT_EVIDENCE
    (workflow_id,) = body["workflow_request_ids"]
    workflow = context.store.get_workflow(workflow_id)
    assert [step.operation for step in workflow.official_steps] == [
        WorkflowOperation.FREEZE_EVIDENCE
    ], "a configuration finding must not isolate, stop or reboot"
    assert all(
        WorkflowOperation.RESTART_NODE not in [s.operation for s in item.official_steps]
        for item in context.store.list_workflows()
    ), "a configuration typo rebooted a healthy node"
    (notification,) = notifier.notifications
    assert "[GPU fault review]" in notification.subject, notification.subject
    assert GPU_EXPECTED_COUNT_CONTRADICTED_METRIC in notification.subject
    for named in ("9", "8", INSTANCE_TYPE, "COLLECT_EVIDENCE"):
        assert named in notification.body_text, notification.body_text


def graded(store, policy, payload):
    """Grade one batch and latch what it emitted, as the ingestion that carries a
    finding into an incident does (an un-notified active signal re-emits)."""

    emitted = policy.evaluate_metrics(payload)
    for finding in emitted:
        store.mark_health_signal_notified(
            finding_health_signal_key(finding),
            notified_at=payload.observed_at,
            semantic_fingerprint=finding_health_signal_fingerprint(finding),
        )
    return emitted


def test_the_episode_is_told_once_clears_when_corrected_and_re_arms() -> None:
    store = build_store()
    policy = NodeHealthPolicy(store)
    typo = [mismatch_sample(expected=9, observed=8)]

    first = graded(store, policy, batch("typo-1", typo))
    still = graded(
        store, policy, batch("typo-still", typo, at=NOW + timedelta(seconds=15))
    )
    corrected = graded(
        store,
        policy,
        batch(
            "corrected",
            [mismatch_sample(expected=8, observed=8, value=0.0, consecutive="0")],
            at=NOW + timedelta(seconds=30),
        ),
    )
    repeated = graded(
        store, policy, batch("typo-2", typo, at=NOW + timedelta(seconds=45))
    )

    assert [item.metric_name for item in first] == [
        GPU_EXPECTED_COUNT_CONTRADICTED_METRIC
    ]
    assert still == [], "the same contradiction is one episode, told once"
    assert corrected == [], "a corrected configuration is not a finding"
    assert [item.metric_name for item in repeated] == [
        GPU_EXPECTED_COUNT_CONTRADICTED_METRIC
    ], "the corrected configuration did not close the episode"


@pytest.mark.parametrize(
    ("observed", "discovered"),
    [(7, 7), (7, None), (8, 7)],
    ids=["short-of-the-type", "short-no-bound-count", "driver-bound-disagrees"],
)
def test_a_real_shortfall_under_a_wrong_expectation_still_reboots(
    observed, discovered
) -> None:
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    observe_running_attempt(context.store, NOW)

    body = post(
        context,
        batch(
            "loss",
            [mismatch_sample(expected=9, observed=observed, discovered=discovered)],
        ),
    )

    by_metric = {item["metric_name"]: item for item in body["findings"]}
    mismatch = by_metric["gpu_inventory_mismatch"]
    assert mismatch["severity"] == "critical"
    assert mismatch["recommended_action"] == "REBOOT_NODE"
    assert by_metric[GPU_EXPECTED_COUNT_CONTRADICTED_METRIC]["severity"] == "warning"
    operations = {
        step.operation
        for item in context.store.list_workflows()
        for step in item.official_steps
    }
    assert WorkflowOperation.RESTART_NODE in operations, (
        "the real shortfall no longer reaches the reboot chain"
    )
    subjects = [item.subject for item in notifier.notifications]
    assert any("inventory mismatch" in subject for subject in subjects), subjects


@pytest.mark.parametrize(
    "sample",
    [
        HostMetricSample(name="gpu_inventory_mismatch", value=1.0),
        mismatch_sample(expected=9, observed=8, instance_type=None),
        mismatch_sample(expected=9, observed=8, instance_type="ml.p5test.48xlarge"),
        HostMetricSample(
            name="gpu_inventory_mismatch",
            value=1.0,
            labels={
                "node_instance_type": INSTANCE_TYPE,
                "expected_count": "nine",
                "observed_count": "8",
            },
        ),
    ],
    ids=["no-labels", "no-type", "unknown-type", "malformed-expected"],
)
def test_a_mismatch_the_table_cannot_re_read_grades_as_before(sample) -> None:
    findings = NodeHealthPolicy(build_store()).evaluate_metrics(
        batch("as-before", [sample])
    )

    assert [
        (item.metric_name, item.severity, item.recommended_action) for item in findings
    ] == [("gpu_inventory_mismatch", Severity.CRITICAL, RecoveryAction.REBOOT_NODE)], (
        findings
    )


def test_a_bound_count_that_cannot_be_read_does_not_excuse_the_mismatch() -> None:
    """A ``discovered_count`` the collector reported but that is not a count is
    no proof the driver agrees with the listing: the mismatch keeps its CRITICAL
    grading; the configuration contradiction is still told."""

    sample = HostMetricSample(
        name="gpu_inventory_mismatch",
        value=1.0,
        labels={
            "node_instance_type": INSTANCE_TYPE,
            "expected_count": "9",
            "observed_count": "8",
            "discovered_count": "unreadable",
        },
    )

    findings = NodeHealthPolicy(build_store()).evaluate_metrics(
        batch("unreadable-bound", [sample])
    )

    assert sorted(
        (item.metric_name, item.severity, item.recommended_action) for item in findings
    ) == [
        (
            GPU_EXPECTED_COUNT_CONTRADICTED_METRIC,
            Severity.WARNING,
            RecoveryAction.COLLECT_EVIDENCE,
        ),
        ("gpu_inventory_mismatch", Severity.CRITICAL, RecoveryAction.REBOOT_NODE),
    ], findings


def test_verdicts_copy_the_mismatch_and_leave_everything_else_alone() -> None:
    gpu = mismatch_sample(expected=9, observed=8, discovered=8)
    efa = HostMetricSample(
        name="efa_inventory_mismatch",
        value=1.0,
        labels={"failure_mode": "PCI_DEVICE_MISSING", "expected_count": "16"},
    )
    other = HostMetricSample(name="cpu_usage_percent", value=12.0)
    before = [item.model_copy(deep=True) for item in (gpu, efa, other)]

    verdicts = gpu_inventory_verdicts([gpu, efa, other])

    assert [gpu, efa, other] == before, "verdicts must not mutate the batch"
    assert [item.name for item in verdicts] == [
        "gpu_inventory_mismatch",
        "efa_inventory_mismatch",
        "cpu_usage_percent",
        GPU_EXPECTED_COUNT_CONTRADICTED_METRIC,
    ]
    re_read, kept_efa, kept_other, contradicted = verdicts
    assert re_read.value == 0, "the node is complete by its own type's count"
    assert re_read.labels["instance_type_gpu_count"] == "8"
    assert kept_efa is efa and kept_other is other
    assert contradicted.value == 1
    assert contradicted.labels["expected_count"] == "9"
    assert contradicted.labels["observed_count"] == "8"
    assert contradicted.labels["discovered_count"] == "8"

    agreeing = gpu_inventory_verdicts(
        [mismatch_sample(expected=8, observed=8, value=0.0, consecutive="0")]
    )
    assert [(item.name, item.value) for item in agreeing] == [
        ("gpu_inventory_mismatch", 0.0),
        (GPU_EXPECTED_COUNT_CONTRADICTED_METRIC, 0.0),
    ], "an agreeing configuration still clears the episode with a 0"
    assert gpu_inventory_verdicts([efa, other]) == [efa, other]


@pytest.mark.parametrize("discovered", [8, None])
def test_the_contradiction_rule_names_the_counts_without_mutating_the_sample(
    discovered,
) -> None:
    (sample,) = [
        item
        for item in gpu_inventory_verdicts(
            [mismatch_sample(expected=9, observed=8, discovered=discovered)]
        )
        if item.name == GPU_EXPECTED_COUNT_CONTRADICTED_METRIC
    ]
    before = sample.model_copy(deep=True)

    threshold, category, severity, action, reason = (
        gpu_expected_count_contradicted_rule(sample)
    )

    assert (threshold, category, severity, action) == (
        1.0,
        NodeHealthCategory.GPU,
        Severity.WARNING,
        RecoveryAction.COLLECT_EVIDENCE,
    )
    for named in ("9", INSTANCE_TYPE, "8 GPUs", "lists 8"):
        assert named in reason, reason
    assert ("bound" in reason) is (discovered is not None), reason
    assert sample == before, "pure rule resolution must not change evidence or labels"
