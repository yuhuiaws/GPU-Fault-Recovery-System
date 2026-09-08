"""Pre-rollout coverage gate for GPU training namespaces.

Background (2026-09-07 security review, operational residual): the GPU rollout
renders workload Role/RoleBindings *only* for the namespaces in a cluster
registration's ``allowed_namespaces``, and the executor/control plane reject any
workload whose namespace is outside that list. A training Pod living in a
namespace the operator forgot to list therefore gets a *silent* 403 at recovery
time -- the worst possible moment to learn the allow-list was wrong.

These tests pin the fail-closed PRE-rollout signal that replaces that silence:
the rollout enumerates the namespaces of live GPU training Pods (via the exact
``gpu-fault.io/managed=true`` selector the Completion Watcher uses) and refuses
to proceed, naming every uncovered namespace, before any RBAC is rendered. The
gate is strictly read-only: it never widens ``allowed_namespaces`` for the
operator, because silently expanding RBAC would be a privilege escalation.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_release_gpu_rollout as ROLLOUT
from gpu_fault_release.regional_release_config import ReleaseError


def _pod(namespace: str, *, phase: str = "Running") -> dict[str, Any]:
    return {
        "metadata": {"namespace": namespace, "name": f"{namespace}-pod-0"},
        "status": {"phase": phase},
    }


def _release(items: list[dict[str, Any]]) -> SimpleNamespace:
    calls: list[list[str]] = []

    def get_json(args: list[str]) -> dict[str, Any]:
        calls.append(args)
        return {"items": items}

    release = SimpleNamespace(
        _gpu=lambda _target, *args: list(args), _get_json=get_json
    )
    release.calls = calls  # type: ignore[attr-defined]
    return release


def _target(allowed: Any) -> SimpleNamespace:
    return SimpleNamespace(cluster_id="gpu-a", allowed_namespaces=allowed)


def test_managed_workload_namespaces_uses_the_watcher_selector() -> None:
    release = _release([_pod("team-a"), _pod("team-b")])

    result = ROLLOUT.managed_workload_namespaces(release, _target(("team-a",)))

    assert result == ["team-a", "team-b"]
    # The one read must be an all-namespaces list scoped to the managed label,
    # so the preflight sees exactly the Pods the data plane will recover.
    (args,) = release.calls  # type: ignore[attr-defined]
    assert "pods" in args
    assert "--all-namespaces" in args
    assert f"{ROLLOUT.MANAGED_WORKLOAD_LABEL}=true" in args


def test_terminal_phase_pods_are_ignored() -> None:
    release = _release(
        [
            _pod("team-a", phase="Succeeded"),
            _pod("team-b", phase="Failed"),
            _pod("team-c", phase="Pending"),
        ]
    )

    # Succeeded/Failed workloads are done and never need recovery, so their
    # namespaces do not have to be covered.
    assert ROLLOUT.managed_workload_namespaces(release, _target(())) == ["team-c"]


def test_preflight_fails_closed_naming_uncovered_namespaces() -> None:
    release = _release([_pod("team-a"), _pod("rogue-ns"), _pod("other-rogue")])

    with pytest.raises(ReleaseError) as excinfo:
        ROLLOUT.preflight_allowed_namespace_coverage(release, _target(("team-a",)))

    message = str(excinfo.value)
    assert "other-rogue" in message
    assert "rogue-ns" in message
    # It names the fix and refuses to make it itself: no auto-widening.
    assert "allowed_namespaces" in message
    assert "team-a" not in message.split(":")[-1]


def test_preflight_passes_when_every_live_namespace_is_allowed() -> None:
    release = _release([_pod("team-a"), _pod("team-b", phase="Pending")])

    # Fully covered -> no raise.
    ROLLOUT.preflight_allowed_namespace_coverage(
        release, _target(("team-a", "team-b", "spare"))
    )


def test_preflight_prints_coverage_even_when_it_passes(capsys) -> None:
    release = _release([_pod("team-a")])

    ROLLOUT.preflight_allowed_namespace_coverage(release, _target(("team-a",)))

    err = capsys.readouterr().err
    assert "gpu-a" in err
    assert "team-a" in err


def test_empty_allow_list_with_a_live_workload_fails_closed() -> None:
    release = _release([_pod("team-a")])

    with pytest.raises(ReleaseError):
        ROLLOUT.preflight_allowed_namespace_coverage(release, _target(()))


def test_test_double_target_without_allow_list_is_skipped() -> None:
    # A target that never declares allowed_namespaces (a unit-test double) must
    # be left alone, exactly as the RBAC rendering leaves it alone -- and it
    # must not even read the cluster.
    release = _release([_pod("team-a")])

    ROLLOUT.preflight_allowed_namespace_coverage(
        release, SimpleNamespace(cluster_id="gpu-a")
    )

    assert release.calls == []  # type: ignore[attr-defined]


def test_managed_label_matches_the_completion_watcher_selector() -> None:
    # The literal here must never drift from the label the watcher lists by; if
    # it did, the preflight would silently stop covering the real workloads.
    from gpu_fault import completion_observation

    assert ROLLOUT.MANAGED_WORKLOAD_LABEL == completion_observation.MANAGED_LABEL


def test_apply_gpu_deployments_gate_runs_before_any_mutation(monkeypatch) -> None:
    """The gate must fire before the rollout renders or applies anything."""

    release = _release([_pod("rogue-ns")])

    def _boom(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - must not run
        raise AssertionError("rendering ran despite an uncovered namespace")

    monkeypatch.setattr(ROLLOUT, "_gpu_deployment_manifests", _boom)

    with pytest.raises(ReleaseError, match="rogue-ns"):
        ROLLOUT.apply_gpu_deployments(release, _target(("team-a",)), "wheel")


def test_preflight_gpu_deployments_gate_runs_before_any_mutation(monkeypatch) -> None:
    release = _release([_pod("rogue-ns")])

    def _boom(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - must not run
        raise AssertionError("rendering ran despite an uncovered namespace")

    monkeypatch.setattr(ROLLOUT, "_gpu_deployment_manifests", _boom)

    with pytest.raises(ReleaseError, match="rogue-ns"):
        ROLLOUT.preflight_gpu_deployments(release, _target(("team-a",)), "wheel")
