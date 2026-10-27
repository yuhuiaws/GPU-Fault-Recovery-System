"""``RESTART_*_DEVICE_PLUGIN`` for a plugin the node's socket probe found hung.

The hung plugin keeps every device advertised (kubelet's ListAndWatch stream to
it never ends), so a restart judged on allocatable alone would find the node
``already_healthy`` and delete nothing. Stamped ``plugin_unresponsive`` by the
planner, the first pass deletes the Pod regardless and the replacement is
confirmed as for any restart: a new Ready Pod and the full allocatable.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import WorkflowStepStatus
from tests.execution._cov95_runtime_plugins import PluginHarness

READY = {"conditions": [{"type": "Ready", "status": "True"}]}


def _node(outcome) -> dict:
    return outcome.details["node_results"]["node-a"]


def _hung_plugin(family: str, **parameters) -> PluginHarness:
    h = PluginHarness(family, plugin_unresponsive=True, **parameters)
    h.node["status"]["allocatable"][h.resource] = "8"
    return h


@pytest.mark.parametrize("family", ["gpu", "efa"])
def test_a_hung_plugin_is_deleted_although_it_advertises_every_device(
    family: str,
) -> None:
    h = _hung_plugin(family)

    first = h.execute()

    assert first.status is WorkflowStepStatus.WAITING, first
    assert h.core.deleted == [("kube-system", "plugin-old")], h.core.deleted
    receipt = _node(first)
    assert receipt["deleted_pod_uid"] == "pod-old"
    assert receipt["deleted_pod"] == "kube-system/plugin-old"
    assert receipt["restart_reason"] == "DEVICE_PLUGIN_UNRESPONSIVE"
    assert receipt["allocatable"] == 8 and receipt["expected"] == 8
    assert "already_healthy" not in receipt
    assert h.annotations[h.key("pod-uid")] == "pod-old"

    h.record(first)
    # The deleted Pod's stream ended, kubelet marked the devices unhealthy, the
    # DaemonSet's replacement registered and the advertisement is back.
    h.core.pods = [
        {"metadata": {"name": "plugin-new", "uid": "pod-new"}, "status": READY}
    ]
    second = h.execute()

    assert second.status is WorkflowStepStatus.SUCCEEDED, second
    receipt = _node(second)
    assert receipt["already_healthy"] is False
    assert receipt["deleted_pod_uid"] == "pod-old"
    assert receipt["deleted_pod"] == "kube-system/plugin-old"
    assert receipt["replacement_pod_uids"] == ["pod-new"]
    assert h.annotations == {}, "the restart marker is cleared on confirmation"


def test_the_old_pod_still_there_and_ready_does_not_confirm() -> None:
    """Until kubelet kills the frozen container the deleted Pod may still be
    listed Ready; only a Pod of another UID is the replacement."""

    h = _hung_plugin("gpu")
    h.record(h.execute())

    second = h.execute()

    assert second.status is WorkflowStepStatus.WAITING, second
    receipt = _node(second)
    assert receipt["waiting_for_replacement_pod"] is True
    assert receipt["deleted_pod_uid"] == "pod-old"
    assert h.core.deleted == [("kube-system", "plugin-old")], (
        "the delete is issued once; later passes wait, they do not re-delete"
    )


def test_a_replacement_kubelet_never_starts_fails_at_the_timeout() -> None:
    """A Pod that asked for the resource during the hang blocked kubelet's
    sync loop in the plugin's Allocate; kubelet then neither kills the frozen
    container nor starts the replacement. The step fails at
    ``restart_timeout_seconds`` and the planner's
    ``failure_escalation_action=REBOOT_NODE`` takes over."""

    h = _hung_plugin("gpu", restart_timeout_seconds=60)
    h.record(h.execute())
    h.annotations[h.key("started-at")] = (
        datetime.now(timezone.utc) - timedelta(seconds=120)
    ).isoformat()

    outcome = h.execute()

    assert outcome.status is WorkflowStepStatus.FAILED, outcome
    assert "did not restore nvidia.com/gpu on node-a" in (outcome.error or "")
    assert outcome.details["deleted_pod_uid"] == "pod-old"
    assert outcome.details["allocatable"] == 8


def test_without_the_stamp_a_full_advertisement_is_still_already_healthy() -> None:
    """The allocatable-loss restart keeps its contract: a node that advertises
    every device on first sight is left alone."""

    h = PluginHarness("gpu")
    h.node["status"]["allocatable"][h.resource] = "8"

    outcome = h.execute()

    assert outcome.status is WorkflowStepStatus.SUCCEEDED, outcome
    assert _node(outcome)["already_healthy"] is True
    assert h.core.deleted == []
