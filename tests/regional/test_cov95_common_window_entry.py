from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._cov95_common_windows import WINDOWS, window_fixture


def install_fake_entry(module, tmp_path, monkeypatch):
    api, settings, assignments = window_fixture(module, tmp_path)
    monkeypatch.setattr(module, "RegionalLiveFixture", lambda _settings: api)
    monkeypatch.setattr(module, "settings_from_arguments", lambda args: args)
    monkeypatch.setattr(module, "install_site_profile", lambda: None)
    monkeypatch.setattr(module, "install_abort_signals", lambda: None)
    arguments = [
        "window",
        "--baseline",
        str(settings.baseline),
        "--rollout-timeout-seconds",
        "1",
    ]
    parser = module.parser()
    monkeypatch.setattr(
        module,
        "parser",
        lambda: SimpleNamespace(parse_args=lambda: parser.parse_args(arguments[1:])),
    )
    return api, settings, assignments, arguments


@pytest.mark.parametrize("module", WINDOWS)
@pytest.mark.parametrize("write_report", [False, True])
def test_read_only_entry_survey_never_opens_a_window(
    module, write_report, tmp_path, monkeypatch, capsys
) -> None:
    api, settings, _assignments, args = install_fake_entry(
        module, tmp_path, monkeypatch
    )
    report = tmp_path / "report.json"
    if write_report:
        args.extend(["--report", str(report)])
    assert module.main() == 0, "a read-only fake survey should succeed"
    value = json.loads(capsys.readouterr().out)
    assert value["mode"] == "read-only", "default execution must remain observational"
    assert api.patches == [], "read-only mode must not mutate a deployment"
    assert not settings.baseline.exists(), "a survey must not create an owner record"
    if write_report:
        assert json.loads(report.read_text()) == value, "persist the exact survey"


@pytest.mark.parametrize("module", WINDOWS)
def test_entry_round_trip_honors_confirmations_and_filters_surveys(
    module, tmp_path, monkeypatch, capsys
) -> None:
    api, settings, assignments, args = install_fake_entry(module, tmp_path, monkeypatch)
    common = list(args)
    args.extend(["--open", "--confirm", module.OPEN_CONFIRMATION])
    for name, value in assignments.items():
        args.extend(["--set", f"{name}={value}"])
    assert module.main() == 0, "confirmed fake open should complete"
    opened = json.loads(capsys.readouterr().out)["open"]
    assert opened["state"] == "OPEN", "the entrypoint must report the completed open"
    assert "pre_window_survey" not in opened, "nested surveys are omitted from output"
    args[:] = [*common, "--close", "--confirm", module.CLOSE_CONFIRMATION]
    assert module.main() == 0, "confirmed fake close should complete"
    closed = json.loads(capsys.readouterr().out)["close"]
    assert closed["state"] == "CLOSED", "the entrypoint must report restored state"
    assert "close_survey" not in closed, "nested close inventory stays in the record"
    assert json.loads(settings.baseline.read_text())["state"] == "CLOSED", (
        "the restored state must be persisted"
    )
    assert len(api.patches) == 2, "open and restore each perform one guarded patch"


@pytest.mark.parametrize("module", WINDOWS)
@pytest.mark.parametrize(
    ("mode", "confirmation", "message"),
    [
        ("open", "", "--open requires --confirm"),
        ("close", "", "--close requires --confirm"),
        ("open", "correct", "at least one --set"),
    ],
)
def test_entry_refuses_unconfirmed_or_empty_mutations(
    module, mode, confirmation, message, tmp_path, monkeypatch
) -> None:
    api, settings, _assignments, args = install_fake_entry(
        module, tmp_path, monkeypatch
    )
    args.extend(
        [
            f"--{mode}",
            "--confirm",
            module.OPEN_CONFIRMATION if confirmation == "correct" else confirmation,
        ]
    )
    with pytest.raises(RegionalFixtureError, match=message):
        module.main()
    assert api.patches == [], "invalid approval must fail before any patch"
    assert not settings.baseline.exists(), "invalid approval cannot create ownership"
