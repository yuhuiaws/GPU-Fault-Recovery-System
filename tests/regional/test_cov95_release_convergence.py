from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from gpu_fault_release import regional_release_agent_convergence as agents
from gpu_fault_release import regional_release_runtime_identity as identity
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional._cov95_release_support import (
    NEW_IMAGE,
    OLD_IMAGE,
    Clock,
    ResourceRelease,
)
from tests.regional.test_runtime_component_identity import ProbeRelease, Release


class SplitRelease(Release):
    def __init__(self, fault: str = "") -> None:
        super().__init__()
        self.config.release_manifest_schema_version = 4
        self.runtime_image = OLD_IMAGE
        self.executor_image = NEW_IMAGE
        self.fault = fault

    def _get_json(self, arguments: list[str]) -> dict[str, Any]:
        value = super()._get_json(arguments)
        if "deployment" in arguments:
            return value
        expected = (
            self.executor_image if "--context" in arguments else self.runtime_image
        )
        for pod in value["items"]:
            pod["spec"] = {"containers": [{"image": expected}]}
        value["items"].append(
            {"metadata": {"name": "terminating", "deletionTimestamp": "now"}}
        )
        if "app=gpu-fault-api-ha" in arguments:
            if self.fault == "image":
                value["items"][0]["spec"]["containers"][0]["image"] = NEW_IMAGE
            elif self.fault == "container":
                value["items"][0]["spec"]["containers"] = []
            elif self.fault == "missing-pod":
                value["items"].pop(0)
            elif self.fault == "transport":
                raise ReleaseError("fixture read unavailable")
        return value


@pytest.mark.parametrize("fault", ["image", "container", "missing-pod", "transport"])
def test_runtime_identity_rejects_bad_pod_images_counts_and_failed_reads(
    fault: str,
) -> None:
    with pytest.raises(
        ReleaseError, match="runtime component identity validation failed"
    ) as caught:
        identity.validate_runtime_component_identity(SplitRelease(fault))
    assert "control-plane/gpu-fault-api-ha" in str(caught.value)


def test_runtime_identity_ignores_terminating_pods_and_preserves_split_images() -> None:
    result = identity.validate_runtime_component_identity(SplitRelease())
    assert len(result["control_plane"]["deployments"]["gpu-fault-api-ha"]) == 1
    assert result["executor"]["expected"] == "e" * 64


def test_ingress_resolver_caches_only_observed_names_and_rejects_empty_required_read() -> (
    None
):
    release = ProbeRelease(pod="pod-a", live_pods=["pod-a"], failures=0)
    assert identity.cpu_ingress_pod_if_running(release) == "pod-a"
    release.pod = "pod-b"
    assert identity.cpu_ingress_pod_if_running(release) == "pod-a"
    identity.forget_cpu_ingress_pod(release)
    release.pod = ""
    with pytest.raises(ReleaseError, match="no running CPU ingress Pod"):
        identity.resolve_cpu_ingress_pod(release, failure="fixture")


def test_pending_installer_states_exclude_unselected_nodes() -> None:
    release = ResourceRelease()
    target = release.config.clusters[0]
    nodes = copy.deepcopy(release.documents[("gpu-a", "nodes", "")]["items"])
    nodes[0]["metadata"]["annotations"]["gpu-fault.io/installer-artifact-sha256"] = (
        "old"
    )
    nodes.extend([{}, {"metadata": {"name": "unselected"}}])
    pending = agents.installer_node_states(
        nodes,
        target,
        release.node_wheel_sha,
        bundle_sha=release.bundle_sha,
        template_sha=release.node_template_sha,
        config_digest=release.config.agent_config_digest,
        node_names=frozenset({"node-a"}),
    )
    assert pending == {"node-a": "Succeeded"}


def test_failed_installer_stops_wait_before_heartbeat_acceptance() -> None:
    release = ResourceRelease()
    release.config.release_manifest_schema_version = 3
    node = release.documents[("gpu-a", "nodes", "")]["items"][0]
    node["metadata"]["annotations"]["gpu-fault.io/installer-state"] = "Failed"
    with pytest.raises(ReleaseError, match="installer failed on: node-a"):
        agents.wait_agents(release, release.config.clusters[0], release.node_wheel_sha)
    assert release.heartbeat_checks == []


def test_wait_narrates_bounded_pending_set_and_honors_timeout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    release = ResourceRelease()
    release.config.release_manifest_schema_version = 3
    original = release.documents[("gpu-a", "nodes", "")]["items"][0]
    nodes = []
    for index in range(6):
        node = copy.deepcopy(original)
        node["metadata"]["name"] = f"node-{index}"
        node["metadata"]["annotations"]["gpu-fault.io/installer-state"] = "Running"
        nodes.append(node)
    release.documents[("gpu-a", "nodes", "")]["items"] = nodes
    clock = Clock()
    monkeypatch.setattr(agents.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(agents.time, "sleep", clock.sleep)
    with pytest.raises(ReleaseError, match="agents did not converge"):
        agents.wait_agents(
            release,
            release.config.clusters[0],
            release.node_wheel_sha,
            timeout_seconds=5,
        )
    assert clock.sleeps == [5]
    assert ",+1" in capsys.readouterr().err
    release.runner.dry_run = True
    agents.wait_agents(release, release.config.clusters[0], release.node_wheel_sha)
    assert clock.sleeps == [5]


def test_slow_inventory_read_does_not_add_sleep_after_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = ResourceRelease()
    release.config.release_manifest_schema_version = 3
    clock = Clock()
    monkeypatch.setattr(agents.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(agents.time, "sleep", clock.sleep)

    def delayed(*_args: Any) -> list[Any]:
        clock.elapsed += 2
        return []

    monkeypatch.setattr(agents, "wave_node_items", delayed)
    with pytest.raises(ReleaseError, match="agents did not converge"):
        agents.wait_agents(
            release,
            release.config.clusters[0],
            release.node_wheel_sha,
            timeout_seconds=1,
        )
    assert clock.sleeps == []


def test_legacy_wait_does_not_invent_bundle_or_template_identity() -> None:
    release = ResourceRelease()
    agents.wait_agents(
        release,
        release.config.clusters[0],
        release.node_wheel_sha,
        legacy_identity=True,
    )
    assert release.heartbeat_checks[0]["bundle_sha"] is None
    assert release.heartbeat_checks[0]["template_sha"] is None


@pytest.mark.parametrize(
    "count,active,aligned,expected",
    [(0, 0, 0, False), (1, 0, 0, False), (1, 1, 0, False), (1, 1, 1, True)],
)
def test_heartbeat_convergence_requires_the_complete_nonempty_node_set(
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    active: int,
    aligned: int,
    expected: bool,
) -> None:
    release = ResourceRelease()
    monkeypatch.setattr(
        agents,
        "exec_cpu_ingress_probe",
        lambda *_args, **_kwargs: json.dumps({"active": active, "aligned": aligned}),
    )
    arguments = {
        "node_count": count,
        "node_names": ("node-a",),
        "artifact_sha": release.node_wheel_sha,
        "config_digest": release.config.agent_config_digest,
        "runtime_profile_version": release.config.runtime_profile_version,
        "bundle_sha": None,
        "template_sha": None,
    }
    assert (
        agents.agent_heartbeats_converged(
            release, release.config.clusters[0], **arguments
        )
        is expected
    )
    release.runner.dry_run = True
    assert (
        agents.agent_heartbeats_converged(
            release, release.config.clusters[0], **arguments
        )
        is True
    )
