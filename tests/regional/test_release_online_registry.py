from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.regional import RegionalRegistryRevision
from gpu_fault.regional_registry_runtime import (
    active_registry_member_ids,
    registry_revision_converged,
)
from gpu_fault_release import regional_release_online_registry as REGISTRY
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional._release_orchestrator_support import ingress_pod_list_json

ROOT = Path(__file__).resolve().parents[2]


def test_join_transition_uses_one_cpu_pod_client(monkeypatch) -> None:
    calls = []

    class Runner:
        def run(self, arguments, **kwargs):
            calls.append((arguments, kwargs))
            if "get" in arguments and "pod" in arguments:
                if "json" in arguments:
                    return ingress_pod_list_json("ingress-0")
                return "ingress-0"
            return json.dumps(
                {"generation": 4, "content_sha256": "a" * 64, "converged": True}
            )

    release = SimpleNamespace(
        runner=Runner(),
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _cpu=lambda *arguments: list(arguments),
    )
    monkeypatch.setattr(
        REGISTRY,
        "registry",
        lambda _release: [
            {
                "cluster_id": "gpu-a",
                "region": "us-east-1",
                "hyperpod_cluster_name": "hp-gpu-a",
                "eks_cluster_arn": ("arn:aws:eks:us-east-1:123456789012:cluster/gpu-a"),
                "token": "t" * 32,
                "allowed_namespaces": ["gpu-fault-system", "training"],
                "agent_endpoint_allowed_cidrs": ["10.0.0.0/16"],
            }
        ],
    )

    status = REGISTRY.transition_join_registry(
        release, "gpu-a", "PENDING", reason="join gpu-a pending"
    )

    assert status["generation"] == 4
    assert len(calls) == 2
    assert calls[0][0][-2:] == ["-o", "json"]
    transaction = json.loads(calls[1][1]["input_text"])
    assert transaction["path"].endswith("/gpu-a/transition"), (
        "registry client used the wrong cluster transition endpoint"
    )
    assert transaction["payload"]["lifecycle_state"] == "PENDING"
    assert calls[1][1]["timeout_seconds"] == 360


def _publish_release(
    calls: list, *, publish: Exception | str, status: dict | None = None
) -> SimpleNamespace:
    """A release whose publish exec raises ``publish`` (or returns it) and whose
    follow-up status exec returns ``status``."""

    class Runner:
        dry_run = False

        def run(self, arguments, **kwargs):
            calls.append((list(arguments), kwargs))
            if "get" in arguments and "pod" in arguments:
                if "json" in arguments:
                    return ingress_pod_list_json("ingress-0")
                return "ingress-0"
            transaction = json.loads(kwargs.get("input_text") or "{}")
            if "path" in transaction:
                if isinstance(publish, Exception):
                    raise publish
                return publish
            return json.dumps(status or {})

    return SimpleNamespace(
        runner=Runner(),
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _cpu=lambda *arguments: list(arguments),
    )


def _execs(calls: list) -> list[dict]:
    return [
        json.loads(kwargs["input_text"])
        for _args, kwargs in calls
        if "input_text" in kwargs
    ]


def test_publish_that_ran_out_its_window_names_the_members_that_did_not_ack(
    monkeypatch, capsys
) -> None:
    """Convergence is over control-plane processes: when a publish times out,
    the operator needs the member that held it, not "the fleet"."""

    calls: list = []
    clock = iter([100.0, 100.0 + 300.0])
    monkeypatch.setattr(REGISTRY.time, "monotonic", lambda: next(clock))
    release = _publish_release(
        calls,
        publish=ReleaseError("command failed (1): kubectl"),
        status={
            "generation": 9,
            "required_member_ids": ["api-0", "worker-1"],
            "acked_member_ids": ["api-0"],
            "missing_member_ids": ["worker-1"],
            "active_member_ids": ["api-0", "worker-1"],
            "members": [
                {
                    "member_id": "worker-1",
                    "service_role": "control-worker",
                    "ready": False,
                    "generation": 8,
                    "last_seen_at": "2026-09-12T10:00:00Z",
                    "error": "RuntimeError: regional registry head digest mismatch",
                }
            ],
        },
    )

    with pytest.raises(ReleaseError, match="command failed"):
        REGISTRY.publish_registry_revision(
            release,
            path="/v1/regional/registry/revisions",
            payload={"registrations": [], "reason": "remove gpu-b purged"},
            use_current_generation=True,
            timeout_seconds=300,
        )

    execs = _execs(calls)
    assert [item.get("path") for item in execs] == [
        "/v1/regional/registry/revisions",
        None,
    ], "one publish, then exactly one read-only status exec"
    err = capsys.readouterr().err
    assert "generation 9 did not converge within 300s" in err
    assert "missing worker-1: role=control-worker ready=False generation=8" in err
    assert "head digest mismatch" in err


def test_publish_refused_before_its_window_does_not_read_the_status(
    monkeypatch, capsys
) -> None:
    """An identity or generation refusal fails in seconds; it gets no second exec."""

    calls: list = []
    clock = iter([100.0, 103.0])
    monkeypatch.setattr(REGISTRY.time, "monotonic", lambda: next(clock))
    release = _publish_release(
        calls, publish=ReleaseError("command failed (1): kubectl")
    )

    with pytest.raises(ReleaseError, match="command failed"):
        REGISTRY.publish_registry_revision(
            release,
            path="/v1/regional/registry/clusters/gpu-b/transition",
            payload={"lifecycle_state": "ROLLED_BACK", "reason": "rolled back"},
            use_current_generation=False,
            timeout_seconds=300,
        )

    assert len(_execs(calls)) == 1, calls
    assert "did not converge" not in capsys.readouterr().err


def test_status_read_failure_after_a_timeout_never_masks_the_publish_error(
    monkeypatch, capsys
) -> None:
    calls: list = []
    clock = iter([0.0, 400.0])
    monkeypatch.setattr(REGISTRY.time, "monotonic", lambda: next(clock))

    class Runner:
        dry_run = False

        def run(self, arguments, **kwargs):
            calls.append(arguments)
            if "get" in arguments and "pod" in arguments:
                if "json" in arguments:
                    return ingress_pod_list_json("ingress-0")
                return "ingress-0"
            raise ReleaseError("exec failed")

    release = SimpleNamespace(
        runner=Runner(),
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _cpu=lambda *arguments: list(arguments),
    )

    with pytest.raises(ReleaseError, match="exec failed"):
        REGISTRY.publish_registry_revision(
            release,
            path="/v1/regional/registry/revisions",
            payload={"registrations": [], "reason": "purge"},
            use_current_generation=True,
            timeout_seconds=300,
        )

    assert "status is unavailable after the publish ran out its 300s" in (
        capsys.readouterr().err
    )


def test_convergence_has_no_per_cluster_scope_and_an_empty_set_is_immediate() -> None:
    """The set a publish converges on is the control plane's active member rows.

    A GPU cluster contributes none (its executor and agents are not registry
    members), so a cluster being joined, purged or rolled back never widens the
    set and there is nothing per cluster to exclude; with no active member the
    revision is converged on the probe's first poll rather than after a timeout.
    """

    now = datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc)
    revision = RegionalRegistryRevision.build(
        generation=2,
        registrations=[],
        previous_generation=1,
        required_member_ids=[],
        reason="remove gpu-b purged",
        created_at=now,
    )

    assert registry_revision_converged(
        revision, [], observed_at=now, stale_seconds=90
    ), "a revision no member has to ack converges at once"
    assert active_registry_member_ids([], observed_at=now, stale_seconds=90) == []
