"""The plan ``--plan`` writes compares equal to what ``--execute`` recomputes.

Live 2026-10-08 (round 7, eks-primary): every HA-011 ``--execute`` failed within
13 s with "approved deployment, predecessor, or resource intent drifted" and,
once the receipt named the drift, ``paths: ["preflight_errors"]``. The shared
driver writes ``preflight_errors`` beside the runner's details so a refused
plan is diagnosable from ``plan.json`` (``live_driver_guard
.details_with_preflight_errors``), while the runner's execute-time
``plan_details`` recomputation never emitted the key -- the unit tests wrote
``plan.json`` by hand and never saw the driver's shape. Here the driver's plan
is driven for real (``run_standard_case`` in plan mode), then executed against
an unchanged fake site.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import ha011_contracts as contracts
from scripts.e2e.regional import live_driver_guard
from scripts.e2e.regional import run_ha011_busy_cpu_takeover as runner
from scripts.e2e.regional.site_profile import SITE_PROFILE_ENV
from tests.regional._cov95_ha011_support import (
    blocked_external_transports as blocked_external_transports,
)
from tests.regional._cov95_ha011_support import (
    deadline,
    install_kubernetes,
    settings_at,
)


@pytest.fixture
def restored_umask() -> Iterator[None]:
    original = os.umask(0o022)
    os.umask(original)
    yield
    os.umask(original)


def _plan_argv(settings: contracts.Settings, run_dir: Path) -> list[str]:
    return [
        "run_ha011_busy_cpu_takeover.py",
        "--run-dir",
        str(run_dir),
        "--plan",
        "--cpu-kubeconfig",
        str(settings.cpu_kubeconfig),
        "--cpu-context",
        settings.cpu_context,
        "--cluster-id",
        settings.cluster_id,
        "--region",
        settings.region,
        "--isolation-id",
        settings.isolation_id,
        "--postgres-image",
        settings.postgres_image,
        "--predecessor-evidence",
        str(settings.predecessor_path),
    ]


def _drive_plan(
    monkeypatch: pytest.MonkeyPatch,
    settings: contracts.Settings,
    run_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> dict[str, Any]:
    """The plan exactly as ``run_standard_case`` writes it for this runner."""

    monkeypatch.delenv(SITE_PROFILE_ENV, raising=False)
    monkeypatch.setattr(live_driver_guard, "install_abort_signals", lambda: None)
    monkeypatch.setattr(sys, "argv", _plan_argv(settings, run_dir))
    assert runner.main() == 0, "a clean read-only preflight plans with exit 0"
    printed = json.loads(capsys.readouterr().out)
    plan_path = run_dir / "cases" / contracts.CASE_ID / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert printed["details"] == plan["details"], "the printed plan is the file"
    assert plan["preflight_passed"] is True, plan
    return plan


def test_the_plan_the_driver_writes_executes_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    restored_umask: None,
) -> None:
    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)

    plan = _drive_plan(monkeypatch, settings, tmp_path, capsys)

    assert plan["details"]["preflight_errors"] == [], (
        "the driver records the preflight verdict beside the planned details"
    )
    assert not fake.created, "planning mutates nothing"

    assert runner.execute_case(settings, tmp_path, 1, deadline()) == 0, (
        "the plan the driver wrote must execute when nothing changed"
    )
    result = json.loads(capsys.readouterr().out)
    assert result["verdict"] == "PASS", result
    assert "drift" not in result and "error" not in result, result
    assert fake.armed and fake.deleted, "the proof ran and cleaned up"


def test_a_real_change_after_the_drivers_plan_still_drifts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    restored_umask: None,
) -> None:
    """Setting the verdict key aside does not blind the gate: a release-state
    document that changed after the driver's plan is still a refused drift,
    named by its leaf alone."""

    settings = settings_at(tmp_path)
    fake = install_kubernetes(monkeypatch, settings)
    _drive_plan(monkeypatch, settings, tmp_path, capsys)
    state = fake.object("ConfigMap", "gpu-fault-regional-release-state", business=True)
    state["data"]["state.json"] = json.dumps(
        {
            "release_id": "release-fixture",
            "phase": "complete",
            "transaction_committed": True,
            "verified_at": "later",
        }
    )

    assert runner.execute_case(settings, tmp_path, 1, deadline()) == 1, (
        "a changed release state is still refused"
    )
    result = json.loads(capsys.readouterr().out)
    assert result["error"] == (
        "approved deployment, predecessor, or resource intent drifted"
    )
    assert result["drift"]["paths"] == ["preflight.identity.release_state_sha256"]
    assert not fake.created, "drift admits nothing"


def test_only_the_drivers_verdict_key_is_set_aside_from_the_compared_intent() -> None:
    planned = {
        "preflight_errors": [],
        "preflight": {"errors": [], "intent_sha256": "a" * 64},
        "owned_process_crash_only": True,
    }

    assert runner.approved_intent(planned) == {
        "preflight": {"errors": [], "intent_sha256": "a" * 64},
        "owned_process_crash_only": True,
    }, "preflight.errors, emitted by both sides, stays compared"
    assert runner.approved_intent({"preflight_errors": ["x"]}) == {}
    assert runner.approved_intent(None) is None, (
        "a missing plan still compares unequal to any recomputation"
    )
    assert runner.approved_intent(["not", "a", "plan"]) == ["not", "a", "plan"]
    assert runner.VERDICT_KEYS == frozenset({"preflight_errors"})
