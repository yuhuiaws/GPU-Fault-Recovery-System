"""The device plugin restart's durable receipt names what it restarted.

GF-GENERAL-LIVE-014 asks the product's own record for the deleted plugin Pod
and its replacement. ``record_attempt`` replaces a step's record on every pass,
so the delete attempt's WAITING receipt (``deleted_pod`` / ``deleted_pod_uid``)
was gone once the restart was confirmed and the SUCCEEDED record named only
``replacement_pod_uids``. Every record after the delete now keeps the deleted
UID (read from the node's own restart annotation) and the Pod name (carried
from this step's earlier receipt); a node that was healthy on first sight, or a
restart that only waited for a DaemonSet Pod, names no deleted Pod.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import WorkflowStepStatus
from tests._builders import copy_model, workflow_step_execution
from tests.execution._cov95_runtime_plugins import PluginHarness

DELETED_POD = "kube-system/plugin-old"
READY = {"conditions": [{"type": "Ready", "status": "True"}]}


def _replacement_ready(h: PluginHarness, uid: str = "pod-new") -> None:
    h.node["status"]["allocatable"][h.resource] = "8"
    h.core.pods = [{"metadata": {"name": "plugin-new", "uid": uid}, "status": READY}]


def _node(outcome) -> dict:
    return outcome.details["node_results"]["node-a"]


@pytest.mark.parametrize("family", ["gpu", "efa"])
@pytest.mark.parametrize("prior", ["recorded", "absent", "foreign"])
def test_confirmed_restart_names_the_deleted_pod_and_its_replacement(
    family: str, prior: str
) -> None:
    """The terminal record carries the original UID whatever the executor kept
    of the earlier pass: the node's annotation is the authority. The Pod name
    is lent by this step's own earlier receipt only when it names that UID."""

    h = PluginHarness(family)
    first = h.execute()
    assert first.status is WorkflowStepStatus.WAITING, first
    delete_receipt = _node(first)
    assert delete_receipt["deleted_pod_uid"] == "pod-old", delete_receipt
    assert delete_receipt["deleted_pod"] == DELETED_POD, delete_receipt
    if prior == "recorded":
        h.record(first)
    elif prior == "foreign":
        # A record at this step identity from before the delete (or from an
        # older product) names another Pod or none: it lends no name.
        stale = workflow_step_execution(
            0,
            h.context.step.operation,
            WorkflowStepStatus.WAITING,
            details={
                "node_results": {
                    "node-a": {
                        "deleted_pod": "kube-system/plugin-other",
                        "deleted_pod_uid": "pod-other",
                    }
                }
            },
        )
        workflow = copy_model(h.context.workflow, step_executions=[stale])
        h.context = replace(h.context, workflow=workflow)
    _replacement_ready(h)

    second = h.execute()

    assert second.status is WorkflowStepStatus.SUCCEEDED, second
    receipt = _node(second)
    assert receipt["deleted_pod_uid"] == "pod-old", receipt
    assert receipt["replacement_pod_uids"] == ["pod-new"], receipt
    assert receipt["already_healthy"] is False, receipt
    if prior == "recorded":
        assert receipt["deleted_pod"] == DELETED_POD, receipt
    else:
        assert "deleted_pod" not in receipt, receipt
    assert h.core.deleted == [("kube-system", "plugin-old")], (
        "a confirmed restart must not delete the replacement",
        h.core.deleted,
    )
    assert h.annotations == {}, h.annotations


def test_every_wait_after_the_delete_keeps_the_deleted_pod() -> None:
    h = PluginHarness()
    first = h.execute()
    h.record(first)
    h.core.pods = []

    second = h.execute()

    assert second.status is WorkflowStepStatus.WAITING, second
    receipt = _node(second)
    assert receipt["waiting_for_replacement_pod"] is True, receipt
    assert receipt["deleted_pod_uid"] == "pod-old", receipt
    assert receipt["deleted_pod"] == DELETED_POD, receipt
    assert h.core.deleted == [("kube-system", "plugin-old")], h.core.deleted

    # The chain is continuous: the next pass reads the wait's receipt.
    h.record(second)
    _replacement_ready(h)
    third = h.execute()
    assert third.status is WorkflowStepStatus.SUCCEEDED, third
    assert _node(third)["deleted_pod"] == DELETED_POD, third
    assert _node(third)["deleted_pod_uid"] == "pod-old", third


def test_a_timed_out_restart_keeps_the_deleted_pod_in_its_failure() -> None:
    h = PluginHarness(restart_timeout_seconds=60)
    first = h.execute()
    h.record(first)
    h.annotations[h.key("started-at")] = (
        datetime.now(timezone.utc) - timedelta(minutes=5)
    ).isoformat()
    h.core.pods = []

    outcome = h.execute()

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert outcome.details["deleted_pod_uid"] == "pod-old", outcome.details
    assert outcome.details["deleted_pod"] == DELETED_POD, outcome.details
    assert outcome.details["node_id"] == "node-a", outcome.details
    assert h.annotations == {}, h.annotations


@pytest.mark.parametrize("family", ["gpu", "efa"])
def test_a_node_healthy_on_first_sight_names_no_deleted_pod(family: str) -> None:
    h = PluginHarness(family)
    h.node["status"]["allocatable"][h.resource] = "8"

    outcome = h.execute()

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    receipt = _node(outcome)
    assert receipt["already_healthy"] is True, receipt
    assert "deleted_pod_uid" not in receipt and "deleted_pod" not in receipt, receipt
    assert h.core.deleted == [], h.core.deleted


def test_a_restart_that_only_waited_for_the_daemonset_names_no_deleted_pod() -> None:
    """No Pod was there to delete: the restart marked the node and waited. Its
    confirmation must not invent a deletion."""

    h = PluginHarness()
    h.core.pods = []
    first = h.execute()
    assert first.status is WorkflowStepStatus.WAITING, first
    assert _node(first)["waiting_for_daemonset_pod"] is True, first
    h.record(first)
    _replacement_ready(h)

    second = h.execute()

    assert second.status is WorkflowStepStatus.SUCCEEDED, second
    receipt = _node(second)
    assert receipt["replacement_pod_uids"] == ["pod-new"], receipt
    assert receipt["already_healthy"] is False, receipt
    assert "deleted_pod_uid" not in receipt and "deleted_pod" not in receipt, receipt
    assert h.core.deleted == [], h.core.deleted
