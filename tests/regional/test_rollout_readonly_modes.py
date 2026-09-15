"""The release engine's read-only modes print what an operator can read.

A passing ``preflight`` dumped about 250 lines of check details in the middle of
every deploy; ``status`` ran every health check whether or not anyone asked.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_admin_commands as ADMIN
from gpu_fault_release import rollout as ROLLOUT


def _preflight_report(healthy: bool) -> dict[str, object]:
    checks = [
        {"name": "tools", "status": "PASS", "summary": "ok", "details": {"aws": 1}},
        {"name": "aurora", "status": "PASS", "summary": "ok", "details": {"x": 2}},
    ]
    if not healthy:
        checks.append(
            {"name": "monitoring", "status": "FAIL", "summary": "AMP", "details": None}
        )
    return {
        "mode": "preflight",
        "site_name": "site",
        "healthy": healthy,
        "summary": {"PASS": 2, "WARN": 0, "FAIL": 0 if healthy else 1, "SKIP": 0},
        "checks": checks,
    }


def _rollout(monkeypatch: pytest.MonkeyPatch, capsys, *argv: str) -> tuple[int, str]:
    """Run the rollout entry point with ``argv`` and return (exit code, stdout)."""

    monkeypatch.setattr(
        ROLLOUT.sys, "argv", ["rollout", *argv, "--config", "release.json"]
    )
    code = ROLLOUT.main()
    return code, capsys.readouterr().out


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    seen: dict[str, object] = {}
    # The entry point rewrites the CPU kubeconfig through the token cache, so
    # the stand-in config carries one; a path that does not exist is passed on
    # unchanged, which keeps the release's view of the config equal to this.
    config = SimpleNamespace(cpu_kubeconfig="release.kubeconfig")
    monkeypatch.setattr(ROLLOUT.ReleaseConfig, "load", staticmethod(lambda _p: config))

    class Release:
        def __init__(self, config, runner):
            seen["config"] = config
            # No image pin disagrees with the lock: the read-only modes run.
            self.image_lock_conflicts: dict[str, tuple[str, str]] = {}

        def status(self, *, full=False):
            seen["status_full"] = full
            return {"mode": "status", "healthy": True, "health_scope": "quick"}

    monkeypatch.setattr(ROLLOUT, "RegionalRelease", Release)
    monkeypatch.delenv(ADMIN.FULL_REPORT_ENV, raising=False)
    return seen


def test_passing_preflight_prints_only_the_names_of_its_checks(
    engine, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ROLLOUT, "build_preflight_report", lambda _release: _preflight_report(True)
    )

    code, out = _rollout(monkeypatch, capsys, "preflight")

    printed = json.loads(out)
    assert code == 0
    assert printed["healthy"] is True
    assert printed["checks"] == []
    assert printed["passed_checks"] == ["tools", "aurora"]


def test_failing_preflight_keeps_the_failure_details(
    engine, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ROLLOUT, "build_preflight_report", lambda _release: _preflight_report(False)
    )

    code, out = _rollout(monkeypatch, capsys, "preflight")

    printed = json.loads(out)
    assert code == 1
    assert [item["name"] for item in printed["checks"]] == ["monitoring"]
    assert printed["checks"][0]["summary"] == "AMP"
    assert printed["passed_checks"] == ["tools", "aurora"]


def test_full_flag_or_environment_prints_the_whole_preflight(
    engine, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ROLLOUT, "build_preflight_report", lambda _release: _preflight_report(True)
    )

    _, by_flag = _rollout(monkeypatch, capsys, "preflight", "--full")
    monkeypatch.setenv(ADMIN.FULL_REPORT_ENV, "1")
    _, by_environment = _rollout(monkeypatch, capsys, "preflight")

    assert json.loads(by_flag) == json.loads(by_environment) == _preflight_report(True)


def test_status_runs_the_quick_report_unless_full_is_asked(
    engine, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, out = _rollout(monkeypatch, capsys, "status")
    assert code == 0
    assert engine["status_full"] is False
    assert json.loads(out)["mode"] == "status"

    _rollout(monkeypatch, capsys, "status", "--full")
    assert engine["status_full"] is True

    monkeypatch.setenv(ADMIN.FULL_REPORT_ENV, "yes")
    _rollout(monkeypatch, capsys, "status")
    assert engine["status_full"] is True


def test_parser_accepts_full_for_every_mode() -> None:
    parser = ROLLOUT.parser()

    assert parser.parse_args(["status", "--config", "c.json", "--full"]).full is True
    assert parser.parse_args(["preflight", "--config", "c.json"]).full is False


def test_release_status_reads_state_once_inside_one_snapshot(monkeypatch) -> None:
    """The baseline, the summary and the checks share a single state read."""

    reads: list[str] = []
    events: list[str] = []

    @contextmanager
    def snapshot():
        events.append("snapshot-open")
        yield
        events.append("snapshot-close")

    class Release:
        def _load_state(self):
            reads.append("state")
            return {"phase": "complete", "release_id": "release-a"}

        def _apply_health_baseline(self, state=None):
            events.append(f"baseline(state={'given' if state else 'none'})")

        def _read_snapshot(self):
            return snapshot()

    monkeypatch.setattr(
        ADMIN,
        "build_quick_status",
        lambda release, *, state: events.append(f"quick(state={state['phase']})")
        or {"health_scope": "quick"},
    )
    monkeypatch.setattr(
        ADMIN,
        "build_full_status",
        lambda release, *, state: events.append(f"full(state={state['phase']})")
        or {"health_scope": "full"},
    )

    quick = ROLLOUT.RegionalRelease.status(Release())
    full = ROLLOUT.RegionalRelease.status(Release(), full=True)

    assert (quick["health_scope"], full["health_scope"]) == ("quick", "full")
    assert reads == ["state", "state"], "one read per status, none for the baseline"
    assert events == [
        "snapshot-open",
        "baseline(state=given)",
        "quick(state=complete)",
        "snapshot-close",
        "snapshot-open",
        "baseline(state=given)",
        "full(state=complete)",
        "snapshot-close",
    ]
