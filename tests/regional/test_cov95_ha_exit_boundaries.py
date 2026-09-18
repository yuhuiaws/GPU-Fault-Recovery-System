from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_ha007_control_worker_shutdown as ha007
from scripts.e2e.regional import run_ha008_processor_exit_acceptance as ha008
from tests.regional._cov95_ha_exit_harness import ExitHarness


@pytest.mark.parametrize("missing", ["none", "grace", "budget"])
def test_shutdown_budget_parses_relevant_documents_and_requires_both_values(
    tmp_path: Path, missing: str
) -> None:
    generated = tmp_path / "deploy/control-plane/regional/generated"
    generated.mkdir(parents=True)
    grace = "{}" if missing == "grace" else "{terminationGracePeriodSeconds: 120}"
    budget = "{}" if missing == "budget" else f"{{{ha007.LIFESPAN_VARIABLE}: '60'}}"
    (generated / ha007.CONTROL_WORKER_DEPLOYMENT.name).write_text(
        "null\n---\nkind: Pod\n---\nkind: Deployment\n"
        f"spec: {{template: {{spec: {grace}}}}}\n"
    )
    (generated / ha007.CONTROL_WORKER_CONFIG.name).write_text(
        "null\n---\nkind: Pod\n---\nkind: ConfigMap\n" + f"data: {budget}\n"
    )
    if missing == "none":
        result = ha007.control_worker_budgets(tmp_path)
        assert result["lifespan_budget_seconds"] == 60
        assert result["kubernetes_grace_seconds"] == 120
    else:
        with pytest.raises(RuntimeError, match="declares no"):
            ha007.control_worker_budgets(tmp_path)


@pytest.mark.parametrize(
    "duration,updates,fragment",
    [
        (1, {"returncode": 1}, "exited non-zero"),
        (1, {"completed": 0}, "did not complete"),
        (1, {"shutdown_failures": ["worker"]}, "reported as a shutdown failure"),
        (1, {"returncode": -9}, "killed by a signal"),
        (1, {"elapsed_seconds": 11}, "terminationGracePeriod"),
        (1, {"signal_timed_out": True}, "never received SIGTERM"),
        (8, {"completed": 1}, "completed inside"),
        (8, {"shutdown_failures": []}, "coordinator's failure"),
        (8, {"returncode": 0}, "exited zero"),
        (8, {"shutdown_seconds": 1}, "gave up before"),
    ],
)
def test_shutdown_verdict_rejects_each_invalid_completion(
    duration: int, updates: dict[str, Any], fragment: str
) -> None:
    result = {
        "duration_seconds": duration,
        "returncode": 0 if duration == 1 else 1,
        "completed": 1 if duration == 1 else 0,
        "elapsed_seconds": 5,
        "shutdown_failures": [] if duration == 1 else [ha007.WORKER_THREAD_NAME],
        "shutdown_seconds": 5,
        **updates,
    }
    errors = ha007.evaluate_run(
        result, lifespan_budget_seconds=5, kubernetes_grace_seconds=10
    )
    assert any(fragment in item for item in errors), errors


@pytest.mark.parametrize("durations", [[], [0], [float("nan")], [float("inf")]])
def test_invalid_shutdown_durations_never_spawn_children(
    tmp_path: Path, durations: list[float]
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        ha007.run_probe(
            tmp_path,
            durations,
            budgets={"lifespan_budget_seconds": 1, "kubernetes_grace_seconds": 2},
        )


def test_exit_acceptance_without_output_directory_is_explicitly_isolated(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ExitHarness(monkeypatch)
    monkeypatch.setattr(
        ha008, "os", SimpleNamespace(**{**vars(os), "umask": lambda _: None})
    )
    monkeypatch.setattr(sys, "argv", ["unit"])
    assert ha008.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["validation_scope"] == "isolated-source"
    assert result["formal_sequence_satisfied"] is False
