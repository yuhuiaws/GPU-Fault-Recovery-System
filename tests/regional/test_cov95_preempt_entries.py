from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import run_preempt037_dispatcher_liveness as dispatcher
from scripts.e2e.regional import run_preempt038_evidence_pins as evidence


def install_entry(module, tmp_path, monkeypatch, *, execute=False, predecessor=True):
    flags = ["--run-dir", str(tmp_path)]
    if execute:
        flags.extend(["--execute", "--confirm", module.CONFIRMATION])
    parser = module.parser()
    args = parser.parse_args(flags)
    calls = []
    identity = {"release_id": "fixture-release", "cluster_id": "fixture-cluster"}
    regional = SimpleNamespace(
        evidence_identity=lambda: dict(identity),
        settings=SimpleNamespace(environment=lambda: {"context": "fixture-context"}),
    )
    monkeypatch.setattr(
        module, "parser", lambda: SimpleNamespace(parse_args=lambda: args)
    )
    monkeypatch.setattr(
        module, "os", SimpleNamespace(umask=lambda _mask: None, getenv=os.getenv)
    )
    monkeypatch.setattr(module, "install_site_profile", lambda: None)
    monkeypatch.setattr(module, "install_abort_signals", lambda: None)
    monkeypatch.setattr(
        module, "settings_from_arguments", lambda _args: regional.settings
    )
    monkeypatch.setattr(module, "RegionalLiveFixture", lambda _settings: regional)
    monkeypatch.setattr(
        module,
        "predecessor_path",
        lambda *_args: (
            ("GF-REGIONAL-PREEMPT-036", tmp_path / "predecessor.json")
            if predecessor is not None
            else (None, None)
        ),
    )
    monkeypatch.setattr(
        module, "predecessor_evidence", lambda *_args: {"valid": predecessor}
    )

    def plan(**kwargs):
        calls.append(("plan", kwargs))
        return {
            "details": kwargs["details"],
            "preflight_passed": kwargs["preflight_passed"],
        }

    def authorize(*_args, **kwargs):
        calls.append(("authorize", kwargs))
        return datetime.now(timezone.utc) + timedelta(hours=1)

    monkeypatch.setattr(module, "build_plan", plan)
    monkeypatch.setattr(module, "authorize_execution", authorize)
    return args, regional, identity, calls


@pytest.mark.parametrize("module", [dispatcher, evidence])
@pytest.mark.parametrize("predecessor", [True, False, None])
def test_plan_reports_predecessor_without_executing_any_audit(
    module, predecessor, tmp_path, monkeypatch, capsys
) -> None:
    _args, _regional, _identity, calls = install_entry(
        module, tmp_path, monkeypatch, predecessor=predecessor
    )

    def forbidden(*_args):
        raise AssertionError("plan mode must never execute an audit")

    monkeypatch.setattr(module, "execute", forbidden)
    assert module.main() == int(predecessor is False), (
        "failed predecessor must fail planning"
    )
    report = json.loads(capsys.readouterr().out)
    assert report["preflight_passed"] is (predecessor is not False), (
        "the plan must preserve its observed predecessor verdict"
    )
    assert [name for name, _ in calls] == ["plan"], "planning does not grant execution"


@pytest.mark.parametrize("module", [dispatcher, evidence])
@pytest.mark.parametrize("failure", ["", "execute", "identity"])
def test_entry_records_public_outcome_and_rechecks_release_identity(
    module, failure, tmp_path, monkeypatch
) -> None:
    _args, _regional, identity, calls = install_entry(
        module, tmp_path, monkeypatch, execute=True
    )

    def execute(*_args):
        calls.append(("execute", {}))
        if failure == "execute":
            raise RuntimeError("fixture audit failed")
        if failure == "identity":
            identity["release_id"] = "replacement-release"
        return {"verdict": "PASS", "stages": {"observed": []}}

    monkeypatch.setattr(module, "execute", execute)
    assert module.main() == int(bool(failure)), (
        "audit failure or identity drift must fail"
    )
    path = tmp_path / "cases" / module.CASE_ID / f"{module.CASE_ID}.json"
    result = json.loads(path.read_text())
    assert result["verdict"] == ("FAIL" if failure else "PASS"), (
        "the entrypoint must not preserve PASS after a failed audit or identity check"
    )
    assert result["release_id"] == "fixture-release", (
        "retain the originally bound identity"
    )
    assert [name for name, _ in calls] == ["authorize", "execute"], (
        "public execution must occur only after authorization"
    )


@pytest.mark.parametrize("module", [dispatcher, evidence])
@pytest.mark.parametrize("gate", ["confirm", "predecessor"])
def test_entry_refuses_missing_approval_or_failed_predecessor(
    module, gate, tmp_path, monkeypatch
) -> None:
    args, _regional, _identity, calls = install_entry(
        module, tmp_path, monkeypatch, execute=True, predecessor=gate != "predecessor"
    )
    if gate == "confirm":
        args.confirm = "wrong"
    with pytest.raises(module.RegionalFixtureError, match="confirmation|predecessor"):
        module.main()
    assert all(name != "execute" for name, _ in calls), (
        "failed admission must stop the audit"
    )
