from __future__ import annotations

from types import SimpleNamespace

from gpu_fault import api


def test_api_entrypoint_uses_configured_bind_address(monkeypatch) -> None:
    calls = []
    monkeypatch.setenv("GPU_FAULT_API_HOST", "0.0.0.0")
    monkeypatch.setenv("GPU_FAULT_API_PORT", "18080")
    monkeypatch.setitem(
        __import__("sys").modules,
        "uvicorn",
        SimpleNamespace(run=lambda *args, **kwargs: calls.append((args, kwargs))),
    )

    api.run()

    assert calls == [
        (
            ("gpu_fault.app:create_app",),
            {"factory": True, "host": "0.0.0.0", "port": 18080, "reload": False},
        )
    ]


def test_tests_use_the_production_app_import_surface() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    legacy_import = "from gpu_fault." + "api import"
    stale = []
    for directory in ("tests", "scripts", "tools", "deploy"):
        for path in (root / directory).rglob("*.py"):
            if legacy_import in path.read_text(encoding="utf-8"):
                stale.append(path.relative_to(root).as_posix())

    assert stale == []
