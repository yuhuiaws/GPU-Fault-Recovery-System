from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import live_driver_guard
from scripts.e2e.regional import run_boot019_admin_lifecycle as boot019
from scripts.e2e.regional import run_boot020_release_rolling as boot020


def entrypoint(
    family: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    predecessor_valid: bool,
) -> tuple[Any, list[str], Path, list[str]]:
    runner = boot019 if family == "019" else boot020
    monkeypatch.setenv("KUBECONFIG", "/dev/null")
    run_dir = tmp_path / "run"
    site = tmp_path / "site.json"
    site.write_text("{}", encoding="utf-8")
    if family == "019":
        protected = tmp_path / "protected.json"
        protected.write_text("{}", encoding="utf-8")
        arguments = [
            "boot019",
            "--site",
            str(site),
            "--protected-site",
            str(protected),
            "--gpu-cluster-arn",
            "arn:aws:eks:us-west-2:000000000000:cluster/test",
        ]
        monkeypatch.setattr(runner, "epoch_targets", lambda *_a: {"isolated": True})
        factory_name = "LiveAdminLifecycleBackend"
        run_name = "run_admin_lifecycle"
    else:
        arguments = [
            "boot020",
            "--admin-state-dir",
            str(tmp_path),
            "--admin-reference",
            "CHG-ENTRY",
        ]
        monkeypatch.setattr(runner, "validate_admin_target", lambda *_a: None)
        monkeypatch.setattr(
            runner, "public_config_roundtrip", lambda *_a, **_k: {"restored": True}
        )
        for flag in (
            "--noop-config",
            "--control-plane-config",
            "--data-plane-config",
            "--agent-config",
            "--full-config",
        ):
            path = tmp_path / (flag.removeprefix("--") + ".json")
            path.write_text("{}", encoding="utf-8")
            if flag == "--control-plane-config":
                from gpu_fault.admin.config import default_admin_config

                path.write_text(
                    json.dumps(
                        {"admin_config": {"config": default_admin_config().as_dict()}}
                    )
                )
            arguments.extend([flag, str(path)])
        site = tmp_path / "full-config.json"
        kubeconfig = tmp_path / "gpu.kubeconfig"
        kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
        arguments.extend(["--gpu-kubeconfig", str(kubeconfig)])
        factory_name = "LiveReleaseRollingBackend"
        run_name = "run_release_rolling"
    arguments.extend(["--run-dir", str(run_dir)])
    monkeypatch.setattr(runner, "install_site_profile", lambda: None)
    monkeypatch.setattr(live_driver_guard, "applied_site_profile", lambda: None)
    monkeypatch.setattr(live_driver_guard, "source_digest", lambda: "a" * 64)
    monkeypatch.setattr(
        runner, "predecessor_path", lambda *_a: ("previous-case", tmp_path / "previous")
    )
    monkeypatch.setattr(
        runner, "predecessor_evidence", lambda *_a, **_k: {"valid": predecessor_valid}
    )
    calls: list[str] = []

    def backend(*_a: Any, **_k: Any) -> object:
        calls.append("backend")
        return object()

    def execute(_backend: Any, recorder: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("execute")
        recorder.document["verdict"] = "PASS"
        return recorder.complete()

    monkeypatch.setattr(runner, factory_name, backend)
    monkeypatch.setattr(runner, run_name, execute)
    return runner, arguments, site, calls


def execution_flags(runner: Any) -> list[str]:
    return [
        "--execute",
        "--confirm",
        runner.CONFIRMATION,
        "--maintenance-window-end",
        (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
    ]


@pytest.mark.parametrize("family", ["019", "020"])
@pytest.mark.parametrize("predecessor_valid", [True, False])
def test_lifecycle_entrypoint_uses_a_bound_plan_and_canonical_evidence(
    family: str,
    predecessor_valid: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, arguments, _site, calls = entrypoint(
        family, tmp_path, monkeypatch, predecessor_valid=predecessor_valid
    )
    monkeypatch.setattr(sys, "argv", arguments)

    assert runner.main() == (0 if predecessor_valid else 1), (
        "plan exit status must reflect its prerequisite"
    )
    case_dir = tmp_path / "run" / "cases" / runner.CASE_ID
    plan = json.loads((case_dir / "plan.json").read_text())
    assert plan["schema_version"] == 3, "legacy unbound plans must not be emitted"
    assert plan["preflight_passed"] is predecessor_valid, (
        "failed preflight must remain failed in the persisted plan"
    )
    assert calls == [], "planning must not construct a mutating backend"

    monkeypatch.setattr(sys, "argv", [*arguments, *execution_flags(runner)])
    if predecessor_valid:
        assert runner.main() == 0, "the exact approved plan should execute"
        result = json.loads((case_dir / f"{runner.CASE_ID}.json").read_text())
        assert result["verdict"] == "PASS", "success must have a canonical PASS record"
        assert calls == ["backend", "execute"], (
            "one authorization permits one invocation"
        )
    else:
        with pytest.raises(RuntimeError, match="preflight"):
            runner.main()
        assert calls == [], "failed plans cannot reach a live backend"


@pytest.mark.parametrize("family", ["019", "020"])
def test_changed_lifecycle_input_file_requires_a_new_plan(
    family: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, arguments, site, calls = entrypoint(
        family, tmp_path, monkeypatch, predecessor_valid=True
    )
    monkeypatch.setattr(sys, "argv", arguments)
    assert runner.main() == 0, "the original input should produce a valid plan"
    site.write_text('{"changed": true}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [*arguments, *execution_flags(runner)])

    with pytest.raises(RuntimeError, match="details"):
        runner.main()

    assert calls == [], "a same-path content change must be rejected before mutation"
