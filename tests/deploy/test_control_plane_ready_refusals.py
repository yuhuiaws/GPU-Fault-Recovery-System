"""Edge verdicts of the post-roll control-plane readiness wait.

The rotate-token tests cover the happy path and the three transient classes;
these pin what the wait does with Deployments and Pods that are shaped
unusually (no selector, no Ready condition, health documents without a registry
section, several serving replicas) and the publish retry's explicit bounds.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_release_control_plane_ready as READY
from gpu_fault_release.regional_release_config import ReleaseError
from tests.admin.test_admin_rotate_token_control_plane import ControlPlane

API_HA = READY.inventory.CPU_INGRESS_DEPLOYMENT
REVISION = READY.CURRENT_REVISION_ANNOTATION
HASH = READY.POD_TEMPLATE_HASH_LABEL


def pod(name: str, *, conditions: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "metadata": {"name": name, "labels": {"app": API_HA, HASH: "new"}},
        "status": {
            "phase": "Running",
            "conditions": (
                [{"type": "Ready", "status": "True"}]
                if conditions is None
                else conditions
            ),
        },
    }


def release(cluster: ControlPlane, *, dry_run: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        runner=SimpleNamespace(
            dry_run=dry_run, run=cluster.run, probe_output=cluster.probe_output
        ),
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _cpu=lambda *arguments: ["kubectl", "--kubeconfig", "cpu", *arguments],
        _get_json=cluster.get_json,
    )


def registry_health(generation: int) -> dict[str, Any]:
    return {
        "http_status": 200,
        "regional_registry": {
            "ready": True,
            "generation": generation,
            "target_generation": generation,
            "error": None,
        },
    }


def test_deployment_without_match_labels_fails_closed() -> None:
    cluster = ControlPlane().with_rolled(API_HA)
    del cluster.deployments[API_HA]["spec"]["selector"]
    with pytest.raises(ReleaseError, match="no matchLabels"):
        READY.current_replicaset(release(cluster), API_HA)
    assert all("replicaset" not in call for call in cluster.commands), (
        "no ReplicaSet may be listed without a selector"
    )


def test_pod_without_a_ready_condition_is_not_serving() -> None:
    cluster = ControlPlane().with_rolled(API_HA, replicas=1)
    cluster.pods.append(
        pod("api-new-1", conditions=[{"type": "ContainersReady", "status": "True"}])
    )
    current = READY.current_replicaset(release(cluster), API_HA)
    assert [item.ready for item in current.pods] == [False]
    assert current.serving_pods == ()
    assert not current.complete, "an unconditioned Pod cannot complete the roll"


def test_health_documents_without_a_registry_section_are_judged_by_http_status() -> (
    None
):
    cluster = ControlPlane().with_rolled(API_HA, replicas=3)
    cluster.pods.extend([pod("api-a"), pod("api-b"), pod("api-c")])
    cluster.health = {
        "api-a": {"http_status": 200},
        "api-b": {"http_status": 503},
        "api-c": [],
    }
    clock = {"now": 0.0}

    def monotonic() -> float:
        clock["now"] += 100.0
        return clock["now"]

    with pytest.raises(ReleaseError, match="did not become ready") as refused:
        READY.wait_control_plane_ready(
            release(cluster),
            [API_HA],
            timeout_seconds=10.0,
            sleep=lambda _seconds: None,
            monotonic=monotonic,
        )
    message = str(refused.value)
    assert f"{API_HA}/api-b: not ready (HTTP 503)" in message
    assert f"{API_HA}/api-c: health probe failed" in message
    assert "returned a non-object" in message
    assert "api-a" not in message, "a 200 without a registry section is not pending"


def test_several_serving_replicas_on_one_generation_are_ready() -> None:
    cluster = ControlPlane().with_rolled(API_HA, replicas=2)
    cluster.pods.extend([pod("api-a"), pod("api-b")])
    cluster.health = {"api-a": registry_health(7), "api-b": registry_health(7)}
    readiness = READY.wait_control_plane_ready(
        release(cluster), [API_HA], sleep=lambda _seconds: None
    )
    assert readiness["registry_generation"] == 7
    assert readiness["deployments"][API_HA]["ready_pods"] == ["api-a", "api-b"]
    assert sorted(cluster.health_calls) == [("api-a", 8080), ("api-b", 8080)]


def test_a_failure_without_codes_or_exec_target_is_refused() -> None:
    cluster = ControlPlane()
    verdict = READY.classify_publish_failure(
        release(cluster), ReleaseError("registry generation conflict"), pod=None
    )
    assert verdict == READY.FAILURE_REFUSED
    assert cluster.commands == [], "no Pod lookup without an exec target"


def test_publish_needs_at_least_one_attempt() -> None:
    cluster = ControlPlane().with_rolled(API_HA, replicas=1)
    with pytest.raises(ReleaseError, match="at least one attempt"):
        READY.publish_after_control_plane_roll(
            release(cluster, dry_run=True),
            publish=lambda: pytest.fail("zero attempts must publish nothing"),
            attempts=0,
            delay_seconds=0.0,
        )


def test_explicit_bounds_publish_once_without_a_recorder() -> None:
    cluster = ControlPlane().with_rolled(API_HA, replicas=1)
    cluster.pods.append(pod("api-new-1"))
    published: list[int] = []

    def publish() -> dict[str, Any]:
        published.append(1)
        return {"revision": 3}

    outcome = READY.publish_after_control_plane_roll(
        release(cluster, dry_run=True),
        publish=publish,
        deployments=[API_HA],
        attempts=2,
        delay_seconds=0.0,
        timeout_seconds=5.0,
        sleep=lambda _seconds: pytest.fail("a first-attempt success never sleeps"),
    )
    assert published == [1]
    assert outcome["result"] == {"revision": 3}
    assert outcome["control_plane_ready"] == {"dry_run": True, "deployments": [API_HA]}
    assert [entry["outcome"] for entry in outcome["publish_attempts"]] == ["published"]
    assert outcome["publish_attempts"][0]["pod"] == "api-new-1"


def test_a_non_integer_registry_generation_is_not_recorded() -> None:
    cluster = ControlPlane().with_rolled(API_HA, replicas=1)
    cluster.pods.append(pod("api-a"))
    health = registry_health(7)
    health["regional_registry"]["generation"] = "7"
    cluster.health = {"api-a": health}
    readiness = READY.wait_control_plane_ready(
        release(cluster), [API_HA], sleep=lambda _seconds: None
    )
    assert readiness["registry_generation"] is None, "only integers are generations"
    assert readiness["deployments"][API_HA]["ready_pods"] == ["api-a"]
