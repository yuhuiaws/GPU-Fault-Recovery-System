from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from scripts.e2e.regional import collector_window_fixture as window


@pytest.mark.parametrize("errors", [[], ["target node is not idle"]])
def test_direct_plan_forwards_cli_identity_and_real_preflight_boolean(
    errors: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(window, "install_site_profile", lambda: None)
    monkeypatch.setattr(window, "install_abort_signals", lambda: None)
    monkeypatch.setattr(window.os, "umask", lambda value: 0o077)
    monkeypatch.setattr(
        window.sys,
        "argv",
        ["collector", "--run-dir", str(tmp_path), "--node", "selected-node", "--plan"],
    )
    settings = SimpleNamespace(environment=lambda: {"CLUSTER": "test-cluster"})
    monkeypatch.setattr(window, "configure", lambda *a: settings)
    monkeypatch.setattr(window, "read_only_preflight", lambda *a: {"errors": errors})
    captured: dict[str, Any] = {}

    def build_plan(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"schema_version": 3, "preflight_passed": kwargs["preflight_passed"]}

    monkeypatch.setattr(window, "build_plan", build_plan)
    execute = Mock(side_effect=AssertionError("plan must not execute"))
    status = window.run_window_case(
        case_id="GF-REGIONAL-COLLECT-019",
        confirmation="COLLECT019_EXECUTE",
        parser_description="mock plan binding",
        plan_details=lambda *a: {},
        execute=execute,
    )
    assert isinstance(captured["arguments"], argparse.Namespace), (
        f"plan builder lost the parsed CLI identity: {type(captured['arguments']).__name__}"
    )
    assert captured["arguments"].node == "selected-node"
    assert captured["arguments"].run_dir == tmp_path
    assert captured["preflight_passed"] is (not errors)
    assert status == (1 if errors else 0)
    execute.assert_not_called()


def test_failed_execute_preflight_never_creates_a_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(window, "install_site_profile", lambda: None)
    monkeypatch.setattr(window, "install_abort_signals", lambda: None)
    monkeypatch.setattr(window.os, "umask", lambda value: 0o077)
    monkeypatch.setattr(
        window.sys,
        "argv",
        [
            "collector",
            "--run-dir",
            str(tmp_path),
            "--node",
            "selected-node",
            "--execute",
            "--confirm",
            "COLLECT019_EXECUTE",
        ],
    )
    monkeypatch.setattr(
        window, "configure", lambda *a: SimpleNamespace(environment=lambda: {})
    )
    monkeypatch.setattr(
        window, "read_only_preflight", lambda *a: {"errors": ["node changed"]}
    )
    monkeypatch.setattr(
        window,
        "authorize_execution",
        lambda *a, **k: datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    probe = Mock(side_effect=AssertionError("failed preflight must not create probe"))
    monkeypatch.setattr(window, "CollectorWindowFixture", probe)
    with pytest.raises(window.RegionalFixtureError, match="node changed"):
        window.run_window_case(
            case_id="GF-REGIONAL-COLLECT-019",
            confirmation="COLLECT019_EXECUTE",
            parser_description="mock execute binding",
            plan_details=lambda *a: {},
            execute=Mock(),
        )
    probe.assert_not_called()
