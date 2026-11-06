"""Every perf runner names its data-plane context or refuses to start.

``regional_capacity_registry.dataplane`` runs ``kubectl --context
$GPU_FAULT_DATAPLANE_CONTEXT``, and kubectl reads an empty ``--context`` as "the
current context". Only the burst runner refused an empty value; the action,
correlated, integrated and storm breaker runners would have created their load
or executor Jobs and the token Secret in whichever cluster the operator's
kubeconfig happened to point at.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import pytest

from scripts.perf import regional_action_capacity_suite as action
from scripts.perf import regional_capacity_registry as registry
from scripts.perf import regional_capacity_suite as capacity
from scripts.perf import regional_correlated_action_suite as correlated
from scripts.perf import regional_integrated_workflow_capacity_suite as integrated
from scripts.perf import regional_storm_breaker_suite as storm

LIVE = ["--allow-live-registry", "--confirm-live-registry"]


def argv_main(
    main: Callable[[], int], *argv: str
) -> Callable[[pytest.MonkeyPatch], int]:
    def call(monkeypatch: pytest.MonkeyPatch) -> int:
        monkeypatch.setattr(sys, "argv", ["runner", *argv])
        return main()

    return call


RUNNERS: dict[str, Callable[[pytest.MonkeyPatch], int]] = {
    "burst": lambda _mp: capacity.main(
        ["all", "--suite-id", "run-a", "--clusters", "1"]
    ),
    "action": lambda _mp: action.main(["--suite-id", "run-a", "--clusters", "1"]),
    "correlated": argv_main(correlated.main, "--clusters", "1", "--suite-id", "run-a"),
    "integrated": argv_main(
        integrated.main,
        "--clusters",
        "32",
        "--spool-mode",
        "enabled",
        "--aurora-cluster-id",
        "aurora-a",
        "--configure-aurora-capacity",
        "--confirm-aurora-scaling",
        "SET_AURORA_MIN_124_MAX_128",
        *LIVE,
        registry.LIVE_REGISTRY_CONFIRMATION,
    ),
    "storm-breaker": lambda _mp: storm.main(["--suite-id", "run-a", *LIVE, "x"]),
}


@pytest.fixture
def no_kubectl(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(argv: list[str], **_kwargs: object) -> object:
        calls.append(argv)
        raise AssertionError(f"a command ran before the context was checked: {argv}")

    monkeypatch.setattr(registry, "run", run)
    monkeypatch.setattr(registry, "DATAPLANE_CONTEXT", "")
    monkeypatch.setenv("GPU_FAULT_AURORA_CLUSTER_ID", "aurora-a")
    monkeypatch.setattr(capacity, "AURORA_CLUSTER_ID", "aurora-a")
    return calls


@pytest.mark.parametrize("runner", sorted(RUNNERS))
def test_every_runner_refuses_an_empty_dataplane_context(
    runner: str,
    no_kubectl: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as refused:
        RUNNERS[runner](monkeypatch)

    assert refused.value.code == 2, "a usage error, before any live work"
    assert "GPU_FAULT_DATAPLANE_CONTEXT is required" in capsys.readouterr().err, (
        "the refusal names the variable to set"
    )
    assert no_kubectl == [], "nothing may reach kubectl or AWS without a context"


@pytest.mark.parametrize(
    "client",
    [registry.dataplane, registry.dataplane_identity],
    ids=lambda f: f.__name__,
)
def test_the_dataplane_client_never_falls_back_to_the_current_context(
    client: Callable[..., str], no_kubectl: list[list[str]]
) -> None:
    with pytest.raises(RuntimeError, match="GPU_FAULT_DATAPLANE_CONTEXT is required"):
        client("get", "pod")

    assert no_kubectl == [], "kubectl must not run with an empty --context"
