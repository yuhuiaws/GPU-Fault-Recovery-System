from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
ADMIN = lazy_script_module(
    "regional_admin_commands_test",
    ROOT / "deploy/control-plane/regional/regional_admin_commands.py",
)


def _admin_module():
    return ADMIN._load()


def test_full_status_keeps_health_when_release_summary_is_missing(monkeypatch) -> None:
    module = _admin_module()
    release = SimpleNamespace(config=SimpleNamespace(site_name="test-site"))
    monkeypatch.setattr(
        module,
        "build_health_report",
        lambda _release, *, mode: {
            "mode": mode,
            "healthy": False,
            "summary": {"FAIL": 1},
            "checks": [],
        },
    )

    def broken_summary(_release):
        raise RuntimeError("deployment is missing")

    monkeypatch.setattr(module, "build_release_status", broken_summary)

    report = module.build_full_status(release)

    assert report["healthy"] is False
    assert report["release_status_error"] == "deployment is missing"
    assert report["health"]["mode"] == "status"


def test_release_summary_does_not_repeat_health_checks(monkeypatch) -> None:
    module = _admin_module()
    release = SimpleNamespace(
        config=SimpleNamespace(site_name="test-site"), _load_state=lambda: {}
    )
    monkeypatch.setattr(
        module,
        "build_health_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("release summary repeated health checks")
        ),
    )
    monkeypatch.setattr(
        module, "build_release_status", lambda _release: {"site_name": "test-site"}
    )
    monkeypatch.setattr(
        module,
        "classify_release",
        lambda _release, _state: SimpleNamespace(
            as_dict=lambda: {"kind": "NOOP", "changed": []}
        ),
    )

    report = module.build_release_summary(release)

    assert report["mode"] == "release-summary"
    assert report["next_deploy"] == {"kind": "NOOP", "changed": []}


def test_failed_release_diff_is_reused_for_resume() -> None:
    diff = ADMIN.stored_release_diff(
        {
            "phase": "failed",
            "release_diff": {
                "kind": "DATA_PLANE_COMPATIBLE",
                "changed": ["executor_wheel"],
            },
        }
    )

    assert diff is not None
    assert diff.as_dict() == {
        "kind": "DATA_PLANE_COMPATIBLE",
        "changed": ["executor_wheel"],
    }
