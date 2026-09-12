"""fail-cluster and rollback-cluster settle the registry for a failed join."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_release_online_registry as online
from gpu_fault_release.regional_release_config import ReleaseError


def _release(calls: list[tuple[str, Any]]) -> Any:
    return SimpleNamespace(
        _update_registry=lambda target, *, remove: calls.append(
            ("update_registry", target.cluster_id, remove)
        )
    )


def test_purge_removes_the_entry_even_when_the_transition_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PENDING entry from an earlier attempt carries that attempt's token
    digest, so the ROLLED_BACK transition is refused on identity; the Secret
    entry and the published revision must still lose the cluster."""

    calls: list[tuple[str, Any]] = []

    def refuse(_release: Any, _cluster_id: str, _state: str, *, reason: str) -> None:
        raise ReleaseError("command failed (1): kubectl")

    monkeypatch.setattr(online, "transition_join_registry", refuse)
    monkeypatch.setattr(
        online,
        "purge_registry_cluster",
        lambda _release, cluster_id: calls.append(("purge", cluster_id)),
    )

    online.purge_failed_join(_release(calls), SimpleNamespace(cluster_id="gpu-b"))

    assert calls == [("update_registry", "gpu-b", True), ("purge", "gpu-b")], (
        "the Secret entry and the revision must both drop the cluster"
    )


def test_fail_join_tolerates_a_missing_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(_release: Any, _cluster_id: str, _state: str, *, reason: str) -> None:
        raise ReleaseError("new regional cluster must enter PENDING first")

    monkeypatch.setattr(online, "transition_join_registry", refuse)

    online.fail_join_registry(SimpleNamespace(), "gpu-b")


def test_fail_join_marks_a_live_entry_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(
        online,
        "transition_join_registry",
        lambda _release, cluster_id, state, *, reason: seen.append((cluster_id, state)),
    )

    online.fail_join_registry(SimpleNamespace(), "gpu-b")

    assert seen == [("gpu-b", "FAILED")], "a live entry must be marked FAILED"
