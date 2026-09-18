"""The native case entry retains cleanup failures even when interrupted."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from scripts.e2e.regional import run_collector_destructive as runner


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_native_interruption_persists_failed_case_and_final_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cleanup_fails: bool
) -> None:
    calls: list[str] = []
    case_id = "GF-REGIONAL-COLLECT-004"
    regional = SimpleNamespace(
        evidence_identity=lambda: {"release_id": "release-a", "cluster_id": "cluster-a"}
    )

    def cleanup() -> dict[str, bool]:
        calls.append("cleanup")
        if cleanup_fails:
            raise RuntimeError("owned probe cleanup failed")
        return {"host_script": False}

    collector = SimpleNamespace(
        node="node-a", create=lambda: calls.append("create"), cleanup=cleanup
    )

    def interrupted(*args: Any, **kwargs: Any) -> None:
        calls.append("handler")
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "RegionalLiveFixture", lambda *args: regional)
    monkeypatch.setattr(
        runner,
        "read_only_preflight",
        lambda *args, **kwargs: {
            "errors": [],
            "store": {"profile": {"profile_version": "v1"}},
            "reboot_scope": {},
            "cpu_blast": {},
        },
    )
    monkeypatch.setattr(runner, "reset_fixture", lambda *args, **kwargs: collector)
    monkeypatch.setattr(
        runner, "CollectorAcceptanceFixture", lambda *args, **kwargs: collector
    )
    monkeypatch.setattr(runner, "run_collect004", interrupted)
    settings = SimpleNamespace(
        case_id=case_id, node="node-a", host_probe_image="image", regional=object()
    )
    with pytest.raises(KeyboardInterrupt):
        runner.execute_case(
            cast(runner.Settings, settings),
            tmp_path,
            1,
            datetime.now(timezone.utc) + timedelta(hours=1),
        )
    result = json.loads((tmp_path / "cases" / case_id / f"{case_id}.json").read_text())
    assert result["verdict"] == "FAIL"
    assert result["error"] == "case interrupted: KeyboardInterrupt"
    assert bool(result.get("collector_cleanup_error")) is cleanup_fails
    assert calls == ["create", "handler", "cleanup"]
