"""Which CPU ingress Pod a release hands its in-Pod probes to.

The control plane rolls one replica at a time, so for a window two Running
Pods exist: the outgoing one (Terminating, or no longer Ready) and the
incoming one. ``resolve_cpu_ingress_pod`` must prefer the Pod that can carry a
long barrier, fall back to any Running Pod only when none qualifies, and
refuse an unreadable or empty answer instead of guessing a name. The
Deployment probe beside it reports "installed" only from a sane replica count.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_release_runtime_identity as identity
from gpu_fault_release.regional_release_config import ReleaseError

NAMESPACE = "gpu-fault-system"


class PodListRunner:
    """``kubectl get pod -o json`` answered from a fixed string."""

    dry_run = False

    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.commands: list[list[str]] = []

    def run(self, arguments: list[str], **_kwargs: Any) -> str:
        self.commands.append(list(arguments))
        return self.raw


class DeploymentProbeRunner:
    """``probe_output`` answering one Deployment document (or none)."""

    def __init__(self, document: dict[str, Any] | None) -> None:
        self.document = document

    def probe_output(
        self, _arguments: list[str], **_kwargs: Any
    ) -> tuple[int, str, str]:
        if self.document is None:
            return 0, "", ""
        return 0, json.dumps(self.document), ""


def _release(runner: Any) -> SimpleNamespace:
    return SimpleNamespace(
        runner=runner,
        config=SimpleNamespace(namespace=NAMESPACE),
        _cpu=lambda *arguments: ["kubectl", *arguments],
    )


def _pod(name: str, *, ready: bool, terminating: bool = False) -> dict[str, Any]:
    metadata: dict[str, Any] = {"name": name}
    if terminating:
        metadata["deletionTimestamp"] = "2026-10-01T00:00:00Z"
    return {
        "metadata": metadata,
        "status": {
            "phase": "Running",
            "conditions": [
                "not-a-mapping",
                {"type": "PodScheduled", "status": "True"},
                {"type": "Ready", "status": "True" if ready else "False"},
            ],
        },
    }


def test_a_terminating_or_unready_pod_is_passed_over_for_the_ready_one() -> None:
    runner = PodListRunner(
        json.dumps(
            {
                "items": [
                    _pod("api-ha-old", ready=True, terminating=True),
                    _pod("api-ha-mid", ready=False),
                    "garbage-item",
                    _pod("api-ha-new", ready=True),
                ]
            }
        )
    )

    chosen = identity.resolve_cpu_ingress_pod(_release(runner), failure="probe")

    assert chosen == "api-ha-new"
    assert runner.commands[0][-2:] == ["-o", "json"]


def test_without_a_ready_pod_any_running_pod_is_the_fallback() -> None:
    runner = PodListRunner(
        json.dumps(
            {
                "items": [
                    {"metadata": {}, "status": {"phase": "Running"}},
                    _pod("api-ha-only", ready=False),
                ]
            }
        )
    )

    assert identity.resolve_cpu_ingress_pod(_release(runner), failure="x") == (
        "api-ha-only"
    )


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json at all",
        json.dumps({"items": "not-a-list"}),
        json.dumps(["a", "list", "not", "a", "mapping"]),
    ],
    ids=["empty", "unparseable", "items-not-list", "payload-not-mapping"],
)
def test_an_unusable_pod_list_refuses_with_the_callers_failure(raw: str) -> None:
    runner = PodListRunner(raw)

    with pytest.raises(ReleaseError, match="no running CPU ingress Pod for barrier"):
        identity.resolve_cpu_ingress_pod(_release(runner), failure="barrier")


def _deployment(replicas: object, *, spec: object = "unset") -> dict[str, Any]:
    document: dict[str, Any] = {
        "kind": "Deployment",
        "metadata": {
            "name": "gpu-fault-api-ha",
            "namespace": NAMESPACE,
            "uid": "deployment-uid",
        },
    }
    if spec != "unset":
        document["spec"] = spec
    else:
        document["spec"] = {"replicas": replicas}
    return document


def test_a_missing_ingress_deployment_reads_as_not_installed() -> None:
    assert (
        identity.cpu_ingress_deployment_installed(_release(DeploymentProbeRunner(None)))
        is False
    )


@pytest.mark.parametrize(("replicas", "installed"), [(0, False), (2, True)])
def test_the_replica_count_decides_installed(replicas: int, installed: bool) -> None:
    release = _release(DeploymentProbeRunner(_deployment(replicas)))

    assert identity.cpu_ingress_deployment_installed(release) is installed


@pytest.mark.parametrize(
    "document",
    [
        _deployment(None, spec="not-a-mapping"),
        _deployment(True),
        _deployment(-1),
        _deployment("3"),
    ],
    ids=["spec-not-mapping", "bool", "negative", "string"],
)
def test_an_invalid_replica_count_is_refused(document: dict[str, Any]) -> None:
    release = _release(DeploymentProbeRunner(document))

    with pytest.raises(ReleaseError, match="replica count is invalid"):
        identity.cpu_ingress_deployment_installed(release)
