"""``RESTART_*_DEVICE_PLUGIN`` deletes the plugin Pod gracefully, never forced.

Live GF-GENERAL-LIVE-014 ``efa_plugin`` (plain EKS, p5en.48xlarge): the EFA
plugin process SIGSTOPped, the product's socket probe found it hung and the
restart force-deleted (grace 0) its Pod. The API server removed the Pod at
once, the DaemonSet created the replacement at once, and the replacement
(hostNetwork, no init container) registered with kubelet before kubelet's
SIGKILL -- a stopped process acts on no SIGTERM -- ended the frozen one. When
the frozen plugin's ``ListAndWatch`` stream then ended, kubelet's device
manager (v1.33 ``handler.go`` ``runClient``, unchanged up to release-1.36)
disconnected the client registered last under ``vpc.amazonaws.com/efa`` --
the replacement's -- and marked every device unhealthy: allocatable 0 with a
Ready replacement Pod for the restart's whole 180 s, the step FAILED and the
escalation rebooted the node. The same freeze of the NVIDIA plugin passed:
that replacement (pod network, NVML start-up) registered after the frozen
container had died. A graceful delete keeps the Pod terminating until kubelet
has stopped its containers, and a non-surge DaemonSet creates no replacement
while its node's Pod is terminating, so the replacement registers only after
the hung plugin's stream ended.

``FrozenPluginNode`` models exactly those parts of Kubernetes, on a simulated
clock; the model's own tests pin it to the live shapes.
"""

from __future__ import annotations

import heapq
import itertools
from typing import Any

import pytest

from gpu_fault.adapters.kubernetes.node_operations import (
    KUBELET_DEVICE_PLUGIN_RECONNECT_ISSUE,
    KUBELET_DEVICE_PLUGIN_RECONNECT_REFERENCE,
    KUBERNETES_FORCED_POD_TERMINATION_QUOTE,
    KUBERNETES_FORCED_POD_TERMINATION_REFERENCE,
    PLUGIN_POD_DELETE_GRACE_SECONDS,
)
from gpu_fault.models import WorkflowStepStatus
from tests.execution._cov95_runtime_plugins import PluginCore, PluginHarness

#: kubelet's ``minimumGracePeriodInSeconds`` (``kuberuntime_manager.go``):
#: ``killContainer`` never stops a container faster, even after a force delete.
KUBELET_MINIMUM_GRACE_SECONDS = 2
#: Neither the EFA nor the NVIDIA chart sets ``terminationGracePeriodSeconds``.
POD_SPEC_GRACE_SECONDS = 30
PLUGIN_VOLUMES = [
    {"name": "device-plugin", "hostPath": {"path": "/var/lib/kubelet/device-plugins"}}
]


def _plugin_pod(name: str, uid: str, *, ready: bool) -> dict[str, Any]:
    return {
        "metadata": {"name": name, "uid": uid},
        "spec": {"volumes": PLUGIN_VOLUMES},
        "status": {
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}]
        },
    }


class FrozenPluginNode(PluginCore):
    """One node whose device plugin process is SIGSTOPped, with the parts of
    Kubernetes the restart rests on (seconds since the restart's delete):

    * API server: a Pod delete with grace 0 removes the object at once; a
      positive grace marks it terminating, and kubelet removes it once the
      Pod's containers stopped (``store.go`` ``lastGraceful``,
      ``status_manager.go`` ``canBeDeleted``);
    * a non-surge DaemonSet controller creates the node's daemon Pod only when
      the node has none, a terminating one included (``daemon_controller.go``
      ``podsShouldBeOnNode``);
    * kubelet: the stopped process acts on no SIGTERM and dies on the SIGKILL
      after ``max(grace, 2 s)``;
    * kubelet's device manager (up to release-1.36): when a plugin's
      ``ListAndWatch`` stream ends it disconnects the client registered last
      under the resource name and marks every device unhealthy; nothing makes
      a disconnected plugin register again.
    """

    def __init__(
        self, resource: str, *, devices: int, replacement_start_seconds: float
    ) -> None:
        super().__init__(resource, typed=False)
        self.resource = resource
        self.devices = devices
        self.replacement_start = replacement_start_seconds
        self.clock = 0.0
        self.pending: list[tuple[float, int, str, str]] = []
        self.sequence = itertools.count()
        self.registered: str | None = "pod-old"
        self.deletes: list[tuple[str, Any]] = []
        self.pods = [_plugin_pod("plugin-old", "pod-old", ready=True)]
        # The frozen plugin's stream never ended: every device stays advertised.
        self._advertise(devices)

    def _advertise(self, count: int) -> None:
        self.nodes["node-a"]["status"]["allocatable"][self.resource] = str(count)

    def _at(self, delay: float, kind: str, uid: str) -> None:
        heapq.heappush(
            self.pending, (self.clock + delay, next(self.sequence), kind, uid)
        )

    def allocatable(self) -> int:
        return int(self.nodes["node-a"]["status"]["allocatable"][self.resource])

    def delete_namespaced_pod(self, name: str, namespace: str, **kwargs: Any) -> None:
        grace = kwargs.get("grace_period_seconds")
        self.deletes.append((name, grace))
        pod = next(item for item in self.pods if item["metadata"]["name"] == name)
        period = POD_SPEC_GRACE_SECONDS if grace is None else grace
        uid = pod["metadata"]["uid"]
        self._at(max(period, KUBELET_MINIMUM_GRACE_SECONDS), "kill", uid)
        if period == 0:
            self.pods.remove(pod)
            self._at(0, "create", "pod-new")
        else:
            pod["metadata"]["deletionTimestamp"] = "terminating"

    def advance(self, seconds: float) -> None:
        until = self.clock + seconds
        while self.pending and self.pending[0][0] <= until:
            when, _, kind, uid = heapq.heappop(self.pending)
            self.clock = when
            if kind == "kill":
                self._kill(uid)
            elif kind == "create":
                self.pods.append(_plugin_pod("plugin-new", uid, ready=False))
                self._at(self.replacement_start, "register", uid)
            else:
                pod = next(item for item in self.pods if item["metadata"]["uid"] == uid)
                pod["status"]["conditions"][0]["status"] = "True"
                self.registered = uid
                self._advertise(self.devices)
        self.clock = until

    def _kill(self, uid: str) -> None:
        # The frozen process's stream ends; kubelet disconnects whichever
        # client is registered now and marks every device unhealthy.
        if self.registered is not None:
            self.registered = None
            self._advertise(0)
        listed = len(self.pods)
        self.pods = [pod for pod in self.pods if pod["metadata"]["uid"] != uid]
        if len(self.pods) < listed:
            # kubelet removed the gracefully deleted Pod; the DaemonSet now
            # sees no Pod on the node.
            self._at(0, "create", "pod-new")


#: (family, devices, replacement start-up): the EFA plugin's replacement is up
#: within a second (hostNetwork: no CNI, no init container); the NVIDIA one
#: needs the pod network and NVML first.
SHAPES = [("efa", 16, 1.0), ("gpu", 8, 8.0)]


def _resource(family: str) -> str:
    return "vpc.amazonaws.com/efa" if family == "efa" else "nvidia.com/gpu"


def test_the_model_reproduces_the_live_force_delete_failure() -> None:
    node = FrozenPluginNode(
        "vpc.amazonaws.com/efa", devices=16, replacement_start_seconds=1.0
    )

    node.delete_namespaced_pod("plugin-old", "kube-system", grace_period_seconds=0)
    node.advance(180)

    assert node.allocatable() == 0, "the replacement was disconnected"
    assert [pod["metadata"]["uid"] for pod in node.pods] == ["pod-new"], node.pods
    assert node.pods[0]["status"]["conditions"][0]["status"] == "True", node.pods
    assert node.registered is None, "no client left for vpc.amazonaws.com/efa"


def test_the_model_reproduces_the_live_gpu_plugin_pass_by_timing_alone() -> None:
    """The NVIDIA replacement registered after the frozen container died, so the
    same force delete restored the GPUs: that pass was timing, not safety."""

    node = FrozenPluginNode("nvidia.com/gpu", devices=8, replacement_start_seconds=8.0)

    node.delete_namespaced_pod("plugin-old", "kube-system", grace_period_seconds=0)
    node.advance(180)

    assert node.allocatable() == 8, node.nodes
    assert node.registered == "pod-new", node.registered


@pytest.mark.parametrize(("family", "devices", "start"), SHAPES)
def test_a_hung_plugin_is_deleted_gracefully_and_its_replacement_registers(
    family: str, devices: int, start: float
) -> None:
    node = FrozenPluginNode(
        _resource(family), devices=devices, replacement_start_seconds=start
    )
    h = PluginHarness(
        family, core=node, plugin_unresponsive=True, expected_count=devices
    )

    first = h.execute()
    assert first.status is WorkflowStepStatus.WAITING, first
    delete_receipt = first.details["node_results"]["node-a"]
    assert delete_receipt["deleted_pod_uid"] == "pod-old", delete_receipt
    assert delete_receipt["restart_reason"] == "DEVICE_PLUGIN_UNRESPONSIVE"
    h.record(first)

    # Past kubelet's SIGKILL of a force-deleted container: the restart still
    # waits (the live passes never caught the second in which a fast
    # replacement and the frozen plugin both advertised every device).
    node.advance(KUBELET_MINIMUM_GRACE_SECONDS + 1)
    second = h.execute()
    assert second.status is WorkflowStepStatus.WAITING, second
    assert second.details["node_results"]["node-a"]["waiting_for_replacement_pod"]
    h.record(second)

    node.advance(30)
    third = h.execute()

    assert third.status is WorkflowStepStatus.SUCCEEDED, third
    receipt = third.details["node_results"]["node-a"]
    assert receipt["replacement_pod_uids"] == ["pod-new"], receipt
    assert receipt["allocatable"] == devices, receipt
    assert receipt["deleted_pod_uid"] == "pod-old", receipt
    assert receipt["already_healthy"] is False, receipt
    assert node.registered == "pod-new", "kubelet keeps the replacement's client"
    assert node.deletes == [("plugin-old", PLUGIN_POD_DELETE_GRACE_SECONDS)], (
        "one graceful delete, never a force delete"
    )
    assert (
        delete_receipt["delete_grace_period_seconds"] == PLUGIN_POD_DELETE_GRACE_SECONDS
    ), delete_receipt
    assert delete_receipt["delete_policy_reference"] == (
        KUBERNETES_FORCED_POD_TERMINATION_REFERENCE
    ), delete_receipt


def test_the_delete_grace_is_positive_and_short() -> None:
    """Positive: grace 0 is the force delete. Short: the restart runs only for
    a plugin that is hung or lost its devices; the SIGKILL ends the hang."""

    assert 0 < PLUGIN_POD_DELETE_GRACE_SECONDS <= 10, PLUGIN_POD_DELETE_GRACE_SECONDS


def test_the_restart_records_kubernetes_evidence_verbatim() -> None:
    assert KUBERNETES_FORCED_POD_TERMINATION_REFERENCE == (
        "https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/"
        "#pod-termination-forced"
    )
    assert KUBERNETES_FORCED_POD_TERMINATION_QUOTE == (
        "When a force deletion is performed, the API server does not wait for "
        "confirmation from the kubelet that the Pod has been terminated on the "
        "node it was running on. It removes the Pod in the API immediately so a "
        "new Pod can be created with the same name. On the node, Pods that are "
        "set to terminate immediately will still be given a small grace period "
        "before being force killed."
    )
    assert KUBELET_DEVICE_PLUGIN_RECONNECT_REFERENCE == (
        "https://github.com/kubernetes/kubernetes/blob/v1.33.5/pkg/kubelet/cm/"
        "devicemanager/plugin/v1beta1/handler.go#L107-L118"
    )
    assert KUBELET_DEVICE_PLUGIN_RECONNECT_ISSUE == (
        "https://github.com/kubernetes/kubernetes/issues/121994"
    )
