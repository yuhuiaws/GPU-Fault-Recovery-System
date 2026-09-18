from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import live_driver_guard
from scripts.e2e.regional import run_notification_acceptance as runner
from tests.regional._cov95_notify_harness import NotificationSite


def notification_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[NotificationSite, list[str]]:
    site = NotificationSite(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "install_site_profile", lambda: None)
    monkeypatch.setattr(runner, "install_abort_signals", lambda: None)
    monkeypatch.setattr(runner, "os", SimpleNamespace(umask=lambda _: None))
    monkeypatch.setattr(runner, "IdentitySite", lambda _: site)
    monkeypatch.setattr(runner, "predecessor_path", lambda *args: (None, None))
    monkeypatch.setattr(live_driver_guard, "source_digest", lambda: "a" * 64)
    argv = [
        "notification",
        "--run-dir",
        str(tmp_path),
        "--site",
        str(site.site_file),
        "--case",
        "GF-REGIONAL-NOTIFY-004",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return site, argv


@pytest.mark.parametrize("changed", ["cpu", "gpu", "site", "release"])
def test_notification_plan_rejects_live_input_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changed: str
) -> None:
    site, argv = notification_cli(monkeypatch, tmp_path)
    assert runner.main() == 0
    plan = json.loads((tmp_path / "cases/GF-REGIONAL-NOTIFY-004/plan.json").read_text())
    assert len(plan["connections"]) == 2
    assert plan["environment"]["DEPLOYED_RELEASE_ID"] == "unit-release"
    if changed == "release":
        monkeypatch.setattr(
            site,
            "regional",
            lambda target: SimpleNamespace(
                evidence_identity=lambda: {
                    "release_id": "another-release",
                    "cluster_id": target.cluster_id,
                }
            ),
        )
    else:
        path = (
            site.site_file
            if changed == "site"
            else getattr(site, f"{changed}_kubeconfig")
        )
        path.write_text("changed unit fixture\n")
    calls = []
    monkeypatch.setattr(
        runner, "run_notify004", lambda *args, **kwargs: calls.append(args)
    )
    argv.extend(
        [
            "--execute",
            "--confirm",
            "NOTIFY004_EXECUTE",
            "--maintenance-window-end",
            (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
        ]
    )
    with pytest.raises(
        RuntimeError, match="drifted at environment|drifted at connections"
    ):
        runner.main()
    assert calls == [], "drift must be rejected before any provider request"


@pytest.mark.parametrize("release_id", ["", "   "])
def test_notification_plan_requires_deployed_release_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, release_id: str
) -> None:
    site, _argv = notification_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(
        site,
        "regional",
        lambda target: SimpleNamespace(
            evidence_identity=lambda: {
                "release_id": release_id,
                "cluster_id": target.cluster_id,
            }
        ),
    )
    with pytest.raises(runner.NotificationAcceptanceError, match="identity is missing"):
        runner.main()
    assert not (tmp_path / "cases/GF-REGIONAL-NOTIFY-004/plan.json").exists(), (
        'test_notification_plan_requires_deployed_release_identity: expected no (tmp_path / "cases/GF-REGIONAL-NOTIFY-004/plan.json").exists()'
    )
