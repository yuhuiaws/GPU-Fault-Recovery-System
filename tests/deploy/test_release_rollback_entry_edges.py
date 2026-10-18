"""Two entry edges of the release engine.

A ``rollback`` asked of a site whose only state is an unfinished bootstrap has
no previous release to restore: the engine cleans the bootstrap up and stops,
rather than refusing with "previous release state is unavailable". And a
``Runner`` built with the kubeconfig token cache routes every command's
arguments and environment through that cache before executing it.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_release_orchestration as orchestration
from gpu_fault_release import rollout
from gpu_fault_release.regional_release_config import ReleaseError


def _bootstrapping_release(phase: str) -> tuple[SimpleNamespace, list[str]]:
    events: list[str] = []
    release = SimpleNamespace(
        config=SimpleNamespace(clusters=()),
        state={},
        _load_state=lambda: {"phase": phase},
        _cleanup_bootstrap=lambda: events.append("cleanup-bootstrap"),
    )
    return release, events


@pytest.mark.parametrize("phase", ["bootstrap-started", "bootstrap-failed"])
def test_a_rollback_of_an_unfinished_bootstrap_cleans_it_up_and_stops(
    phase: str,
) -> None:
    release, events = _bootstrapping_release(phase)

    orchestration.rollback_release(release)

    assert events == ["cleanup-bootstrap"]
    assert release.state == {"phase": phase}, "the live state was adopted as read"


def test_a_rollback_without_a_previous_release_outside_bootstrap_is_refused() -> None:
    release, events = _bootstrapping_release("complete")

    with pytest.raises(ReleaseError, match="previous release state is unavailable"):
        orchestration.rollback_release(release)
    assert events == [], "nothing is cleaned up for a committed release"


class KubeconfigCacheStub:
    """``ReleaseKubeconfigCache.command_inputs`` double recording each routing."""

    def __init__(self) -> None:
        self.routed: list[tuple[list[str], dict[str, str] | None, float]] = []

    def command_inputs(
        self,
        arguments: list[str],
        environment: dict[str, str] | None,
        *,
        timeout_seconds: float,
    ) -> tuple[list[str], dict[str, str] | None]:
        self.routed.append((list(arguments), environment, timeout_seconds))
        return [*arguments, "--routed"], {**(environment or {}), "ROUTED": "1"}


def test_a_runner_with_a_kubeconfig_cache_routes_each_command_through_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[tuple[list[str], dict[str, Any]]] = []

    def execute(
        arguments: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        executed.append((list(arguments), kwargs))
        return subprocess.CompletedProcess(arguments, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(rollout, "run_command", execute)
    cache = KubeconfigCacheStub()
    runner = rollout.Runner(kubeconfigs=cache)

    assert runner.run([sys.executable, "-c", "pass"], capture=True) == "ok"

    assert [routed[0] for routed in cache.routed] == [[sys.executable, "-c", "pass"]]
    assert cache.routed[0][2] > 0, "the cache refresh shares the command deadline"
    assert executed[0][0] == [sys.executable, "-c", "pass", "--routed"]
    assert executed[0][1]["environment"]["ROUTED"] == "1"
