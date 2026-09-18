from __future__ import annotations

import copy
from typing import Any

import pytest

from gpu_fault_release import regional_release_node_preflight as nodes
from gpu_fault_release import regional_release_rollout_wait as waits
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional._cov95_release_support import ResourceRelease, deployment


def node_release() -> ResourceRelease:
    release = ResourceRelease()
    node = release.documents[("gpu-a", "nodes", "")]["items"][0]
    node["status"] = {"conditions": [{"type": "Ready", "status": "True"}]}
    return release


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("missing", "missing"),
        ("uid", "uid-missing"),
        ("duplicate", "uid-duplicate"),
        ("deleting", "deleting"),
        ("taint", "blocking-taint"),
    ],
)
def test_node_gate_rejects_missing_or_ambiguous_live_identity(
    fault: str, problem: str
) -> None:
    release = node_release()
    items = release.documents[("gpu-a", "nodes", "")]["items"]
    names = ("node-a",)
    if fault == "missing":
        items.clear()
    elif fault == "uid":
        items[0]["metadata"].pop("uid")
    elif fault == "duplicate":
        node = copy.deepcopy(items[0])
        node["metadata"]["name"] = "node-b"
        items.append(node)
        names = ("node-a", "node-b")
    elif fault == "deleting":
        items[0]["metadata"]["deletionTimestamp"] = "2026-01-01T00:00:00Z"
    else:
        items[0]["spec"] = {"taints": [None, {"key": "gpu-fault.io/quarantined"}]}
    with pytest.raises(ReleaseError, match=problem):
        nodes.validate_target_node_state(release, release.config.clusters[0], names)
    assert release.runner.calls == []


def candidate() -> nodes.NodeMutationPreflight:
    return nodes.NodeMutationPreflight(
        phase="upgrade",
        node_names=("node-a",),
        wheel_cm="wheel",
        bundle_cm="bundle",
        artifact_sha="a" * 64,
        config_digest="b" * 64,
        runtime_profile_version="profile",
        executor_wheel_filename=None,
        node_compatibility_digest="c" * 64,
        bundle_sha256=None,
        template_sha256=None,
        template_config_map=None,
        max_unavailable=1,
        runtime_image=None,
        node_installer_image=None,
    )


@pytest.mark.parametrize(
    "output,problem",
    [
        ("", "invalid evidence"),
        ("not-json", "invalid evidence"),
        ('{"status":"PASSED","node_count":0}', "did not cover every target"),
    ],
)
def test_installer_preflight_requires_complete_public_receipt(
    monkeypatch: pytest.MonkeyPatch, output: str, problem: str
) -> None:
    release = node_release()
    release.runner.handler = lambda *_args: output
    monkeypatch.setattr(
        nodes, "build_reconciler_environment", lambda *_args, **_kwargs: {}
    )
    with pytest.raises(ReleaseError, match=problem):
        nodes.run_node_installer_preflight(
            release, release.config.clusters[0], candidate()
        )
    environment = release.runner.calls[0][1]["env"]
    assert environment["GPU_FAULT_RECONCILER_PREFLIGHT_ONLY"] == "true"
    assert environment["GPU_FAULT_REQUIRE_ROLLBACK_SLOT"] == "true"


def pending_release() -> ResourceRelease:
    release = ResourceRelease()
    value = deployment("example")
    value["metadata"]["generation"] = 2
    value["spec"]["selector"] = {"matchLabels": {"app": "example"}}
    value["status"] = {
        "observedGeneration": 2,
        "updatedReplicas": 0,
        "readyReplicas": 0,
        "availableReplicas": 0,
        "unavailableReplicas": 1,
    }
    release.documents[("gpu-a", "deployment", "example")] = value
    release.documents[("gpu-a", "pods", "")] = {"items": []}
    return release


@pytest.mark.parametrize("selector", [{}, {"matchLabels": []}])
def test_rollout_refuses_to_observe_pods_without_a_valid_selector(
    selector: dict[str, Any],
) -> None:
    release = pending_release()
    release.documents[("gpu-a", "deployment", "example")]["spec"]["selector"] = selector
    with pytest.raises(ReleaseError, match="selector has no matchLabels"):
        waits.wait_deployment_rollout(release, release.config.clusters[0], "example")
    assert release.reads == [("gpu-a", "deployment", "example")]
    assert release.runner.calls == []


@pytest.mark.parametrize(
    "details,problem",
    [
        ({"reason": "ProgressDeadlineExceeded"}, "ProgressDeadlineExceeded"),
        ({"message": "replica creation rejected"}, "replica creation rejected"),
        ({}, "Deployment stopped progressing"),
    ],
)
def test_rollout_stopped_progressing_is_terminal_before_pod_or_wait_calls(
    details: dict[str, str], problem: str
) -> None:
    release = pending_release()
    release.documents[("gpu-a", "deployment", "example")]["status"]["conditions"] = [
        {"type": "Available", "status": "False"},
        {"type": "Progressing", "status": "True"},
        {"type": "Progressing", "status": "False", **details},
    ]
    with pytest.raises(ReleaseError, match=problem):
        waits.wait_deployment_rollout(release, release.config.clusters[0], "example")
    assert release.reads == [("gpu-a", "deployment", "example")]
    assert release.runner.calls == []


@pytest.mark.parametrize("message", ["namespace quota exceeded", "Insufficient cpu"])
def test_rollout_rejects_unschedulable_pod_after_ignoring_unrelated_conditions(
    message: str,
) -> None:
    release = pending_release()
    release.documents[("gpu-a", "pods", "")]["items"] = [
        {
            "metadata": {"name": "example-pod"},
            "status": {
                "conditions": [
                    {"type": "Ready", "status": "False"},
                    {"type": "PodScheduled", "status": "True"},
                    {"type": "PodScheduled", "status": "False", "reason": "Scheduling"},
                    {
                        "type": "PodScheduled",
                        "status": "False",
                        "reason": "Unschedulable",
                        "message": message,
                    },
                ]
            },
        }
    ]
    with pytest.raises(ReleaseError, match=f"example-pod is unschedulable: {message}"):
        waits.wait_deployment_rollout(
            release, release.config.clusters[0], "example", capacity_grace_seconds=0
        )
    assert release.reads == [("gpu-a", "deployment", "example"), ("gpu-a", "pods", "")]
    assert release.runner.calls == []


def test_rollout_dry_run_keeps_nonminute_timeout_and_does_not_read_resources() -> None:
    release = pending_release()
    release.runner.dry_run = True
    release.runner.handler = lambda *_args: ""
    result = waits.wait_deployment_rollout(
        release, release.config.clusters[0], "example", timeout_seconds=35
    )
    assert result == {"deployment": "example", "dry_run": True}
    assert release.runner.calls == [
        (
            [
                "kubectl",
                "--context",
                "gpu-a",
                "-n",
                "gpu-fault-system",
                "rollout",
                "status",
                "deployment/example",
                "--timeout=35s",
            ],
            {"timeout_seconds": 65},
        )
    ]
    assert release.reads == []
