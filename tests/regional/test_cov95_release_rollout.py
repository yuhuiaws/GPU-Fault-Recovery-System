from __future__ import annotations

import copy
import json
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_admin_commands as commands
from gpu_fault_release import rollout
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_diff import diff_from_changed
from tests.regional._cov95_release_support import (
    NEW_IMAGE,
    OLD_IMAGE,
    RecordingRunner,
    json_response,
)
from tests.regional._release_orchestrator_support import config_file


class Driver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.healthy = True
        self.failure: Exception | None = None

    def record(self, name: str, details: Any = None) -> None:
        self.calls.append((name, details))
        if self.failure is not None:
            raise self.failure

    def plan(self, mode: str) -> list[str]:
        self.record("plan", mode)
        return ["fixture plan"]

    def status(self, *, full: bool = False) -> dict[str, Any]:
        self.record("status", full)
        return {"healthy": self.healthy}

    def bootstrap(self) -> None:
        self.record("bootstrap")

    def upgrade(self) -> None:
        self.record("upgrade")

    def rollback(self, *, automatic: bool = False) -> None:
        self.record("rollback", automatic)

    def commit_release(self) -> None:
        self.record("commit")

    def join_cluster(self, cluster_id: str) -> None:
        self.record("join-cluster", cluster_id)

    def activate_cluster(self, cluster_id: str) -> None:
        self.record("activate-cluster", cluster_id)

    def fail_cluster(self, cluster_id: str) -> None:
        self.record("fail-cluster", cluster_id)

    def rollback_cluster(self, cluster_id: str) -> None:
        self.record("rollback-cluster", cluster_id)

    def remove_cluster(self, cluster_id: str) -> None:
        self.record("remove-cluster", cluster_id)

    def _load_state(self) -> dict[str, Any]:
        self.record("load-state")
        return {"phase": "complete"}

    def _apply_health_baseline(self, state: dict[str, Any]) -> None:
        self.record("health-baseline", state)

    def validate_stability_window(self, *, release_kind=None) -> dict[str, Any]:
        self.record("stability", release_kind)
        return {"healthy": self.healthy}


@pytest.fixture
def driver(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Driver, list[Any]]:
    driver = Driver()
    narration = []
    monkeypatch.delenv("KUBECONFIG", raising=False)
    monkeypatch.setattr(
        rollout.ReleaseConfig,
        "load",
        lambda _path: SimpleNamespace(cpu_kubeconfig=str(tmp_path / "missing.config")),
    )
    monkeypatch.setattr(rollout, "RegionalRelease", lambda _config, _runner: driver)
    monkeypatch.setattr(
        rollout, "RegistryDrainContext", lambda _config, _runner: driver
    )
    monkeypatch.setattr(rollout, "deployment_api_budget", nullcontext)
    monkeypatch.setattr(
        rollout, "deployment_deadline", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(
        rollout,
        "narrate_release_start",
        lambda mode, **kwargs: narration.append(("start", mode, kwargs)),
    )
    monkeypatch.setattr(
        rollout,
        "narrate_release_end",
        lambda mode, **kwargs: narration.append(("end", mode, kwargs)),
    )
    for symbol, mode in (
        ("run_deploy", "deploy"),
        ("run_resume", "resume"),
        ("stage_noop_release", "stage-noop"),
        ("sync_release_state", "sync-state"),
    ):
        monkeypatch.setattr(
            rollout,
            symbol,
            lambda release, mode=mode, **kwargs: release.record(
                mode, kwargs.get("cluster_id")
            ),
        )
    monkeypatch.setattr(
        rollout,
        "drain_registry_clusters",
        lambda release, cluster_ids: release.record("drain-cluster", list(cluster_ids)),
    )
    for symbol, name in (
        ("build_release_summary", "release-summary"),
        ("build_release_diff", "release-diff"),
        ("build_preflight_report", "preflight"),
        ("build_deploy_preflight_report", "preflight-for-deploy"),
    ):
        monkeypatch.setattr(
            rollout,
            symbol,
            lambda release, name=name: release.record(name)
            or {"healthy": release.healthy, "checks": []},
        )
    monkeypatch.setattr(
        rollout,
        "build_health_report",
        lambda release, **_kwargs: release.record("verify")
        or {"healthy": release.healthy},
    )
    return driver, narration


@pytest.mark.parametrize(
    "mode",
    [
        "plan",
        "preflight",
        "release-summary",
        "release-diff",
        "status",
        "bootstrap",
        "deploy",
        "upgrade",
        "resume",
        "rollback",
        "commit",
        "stage-noop",
        "join-cluster",
        "activate-cluster",
        "fail-cluster",
        "rollback-cluster",
        "drain-cluster",
        "remove-cluster",
        "sync-state",
        "verify",
        "stability",
    ],
)
def test_rollout_public_entrypoint_dispatches_exactly_one_selected_mode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    driver: tuple[Driver, list[Any]],
    mode: str,
) -> None:
    instance, narration = driver
    monkeypatch.setattr(
        sys, "argv", ["rollout", mode, "--config", "/dev/null", "--cluster-id", "gpu-a"]
    )
    assert rollout.main() == 0
    calls = instance.calls
    if mode == "verify":
        assert [name for name, _details in calls] == [
            "load-state",
            "health-baseline",
            "verify",
        ]
    elif mode == "stability":
        assert calls == [("load-state", None), ("stability", None)]
    else:
        assert len(calls) == 1
        assert calls[0][0] == mode
    if mode == "drain-cluster":
        assert calls[-1][1] == ["gpu-a"]
    elif mode.endswith("-cluster"):
        assert calls[-1][1] == "gpu-a"
    assert narration == [
        ("start", mode, {"dry_run": False}),
        ("end", mode, {"exit_code": 0}),
    ]
    if mode in {
        "plan",
        "preflight",
        "release-summary",
        "release-diff",
        "status",
        "verify",
        "stability",
    }:
        assert json.loads(capsys.readouterr().out) is not None
    else:
        assert capsys.readouterr().out == ""


def test_batch_drain_keeps_every_selector_in_one_public_call(
    monkeypatch: pytest.MonkeyPatch, driver: tuple[Driver, list[Any]]
) -> None:
    instance, narration = driver
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rollout",
            "drain-cluster",
            "--config",
            "/dev/null",
            "--cluster-id",
            "gpu-b",
            "--cluster-id",
            "gpu-a",
        ],
    )

    assert rollout.main() == 0
    assert instance.calls == [("drain-cluster", ["gpu-b", "gpu-a"])]
    assert narration[-1] == ("end", "drain-cluster", {"exit_code": 0})


@pytest.mark.parametrize(
    "mode",
    [
        "join-cluster",
        "activate-cluster",
        "fail-cluster",
        "rollback-cluster",
        "drain-cluster",
        "remove-cluster",
    ],
)
def test_cluster_modes_refuse_missing_target_before_any_action(
    monkeypatch: pytest.MonkeyPatch, driver: tuple[Driver, list[Any]], mode: str
) -> None:
    instance, narration = driver
    monkeypatch.setattr(sys, "argv", ["rollout", mode, "--config", "/dev/null"])
    assert rollout.main() != 0
    assert instance.calls == []
    assert narration[-1][0] == "end"


@pytest.mark.parametrize(
    "mode,flag", [("deploy", "--automatic"), ("deploy", "--for-deploy")]
)
def test_internal_mode_flags_are_not_accepted_on_other_commands(
    mode: str, flag: str
) -> None:
    with pytest.raises(SystemExit) as caught:
        rollout.parse_arguments([mode, "--config", "/dev/null", flag])
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "mode,extra,call",
    [
        ("preflight", ["--for-deploy", "--full"], ("preflight-for-deploy", None)),
        ("status", ["--full"], ("status", True)),
        ("rollback", ["--automatic"], ("rollback", True)),
    ],
)
def test_public_entrypoint_forwards_only_mode_specific_flags(
    monkeypatch: pytest.MonkeyPatch,
    driver: tuple[Driver, list[Any]],
    mode: str,
    extra: list[str],
    call: tuple[str, Any],
) -> None:
    instance, _narration = driver
    monkeypatch.setattr(sys, "argv", ["rollout", mode, "--config", "/dev/null", *extra])
    assert rollout.main() == 0
    assert instance.calls == [call]


@pytest.mark.parametrize("mode", ["preflight", "status", "verify"])
def test_unhealthy_report_propagates_failure_exit_status(
    monkeypatch: pytest.MonkeyPatch, driver: tuple[Driver, list[Any]], mode: str
) -> None:
    instance, narration = driver
    instance.healthy = False
    monkeypatch.setattr(sys, "argv", ["rollout", mode, "--config", "/dev/null"])
    assert rollout.main() == 1
    assert narration[-1] == ("end", mode, {"exit_code": 1})


@pytest.mark.parametrize(
    "exception",
    [
        ReleaseError("fixture failure"),
        OSError("fixture failure"),
        ValueError("fixture failure"),
    ],
)
def test_rollout_failure_still_closes_its_narration(
    monkeypatch: pytest.MonkeyPatch,
    driver: tuple[Driver, list[Any]],
    exception: Exception,
) -> None:
    instance, narration = driver
    instance.failure = exception
    monkeypatch.setattr(sys, "argv", ["rollout", "upgrade", "--config", "/dev/null"])
    assert rollout.main() != 0
    assert narration[-1][0:2] == ("end", "upgrade")


@pytest.mark.parametrize("method", ["run", "condition", "probe_output"])
@pytest.mark.parametrize(
    "timeout", [TimeoutError(), subprocess.TimeoutExpired("fixture", 1)]
)
def test_runner_translates_timeout_in_every_public_transport_method(
    monkeypatch: pytest.MonkeyPatch, method: str, timeout: Exception
) -> None:
    def execute(*_args: Any, **_kwargs: Any) -> None:
        raise timeout

    monkeypatch.setattr(rollout, "run_command", execute)
    runner = rollout.Runner()
    call = getattr(runner, method)
    with pytest.raises(ReleaseError, match="time budget|probe timed out"):
        call(["kubectl", "wait", "deployment/example"], timeout_seconds=1)


def test_dry_runner_suppresses_mutations_but_captured_reads_use_injected_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def execute(
        arguments: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, 0, stdout=" result \n", stderr="")

    monkeypatch.setattr(rollout, "run_command", execute)
    runner = rollout.Runner(dry_run=True)
    assert runner.run(["fixture-command"]) == ""
    assert calls == []
    assert runner.run(["fixture-read"], capture=True) == "result"
    assert len(calls) == 1


def real_release(tmp_path: Path) -> rollout.RegionalRelease:
    return rollout.RegionalRelease(
        rollout.ReleaseConfig.load(config_file(tmp_path)),
        RecordingRunner(json_response({})),
    )


@pytest.mark.parametrize(
    "changed", [set(), {"control_plane_wheel"}, {"executor_wheel"}, {"database_schema"}]
)
def test_deploy_plan_describes_classification_and_selected_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changed: set[str]
) -> None:
    release = real_release(tmp_path)
    release.runner.handler = json_response({"data": {"state.json": "{}"}})
    diff = diff_from_changed(changed)
    monkeypatch.setattr(rollout, "classify_release", lambda *_args: diff)
    result = release.plan("deploy")
    assert result[0] == f"release classification: {diff.kind.value}"
    assert result[1] == "changed inputs: " + (", ".join(sorted(changed)) or "none")
    assert len(result) > 2


@pytest.mark.parametrize(
    "configured",
    ["", OLD_IMAGE, "mirror.example/runtime@sha256:" + "a" * 64, NEW_IMAGE],
)
def test_release_image_lock_accepts_same_digest_but_rejects_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, configured: str
) -> None:
    config = rollout.ReleaseConfig.load(config_file(tmp_path))
    config = replace(
        config,
        release_manifest_schema_version=3,
        locked_images={
            "runtime": OLD_IMAGE,
            "node_installer": OLD_IMAGE,
            "dcgm_exporter": OLD_IMAGE,
            "adot": OLD_IMAGE,
        },
    )
    monkeypatch.setenv("GPU_FAULT_RUNTIME_IMAGE", configured)
    if configured == NEW_IMAGE:
        with pytest.raises(
            ReleaseError, match="does not match the schema v3 image lock"
        ):
            rollout.RegionalRelease(config, RecordingRunner())
    else:
        release = rollout.RegionalRelease(config, RecordingRunner())
        assert release.runtime_image == (configured or OLD_IMAGE)


@pytest.mark.parametrize("fault", [None, "dcgm-conflict", "adopted", "unchanged"])
def test_status_uses_recorded_live_images_after_successful_rollback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str | None
) -> None:
    release = real_release(tmp_path)
    original = release.runtime_image
    previous = {
        "runtime_image": OLD_IMAGE,
        "node_installer_image": OLD_IMAGE,
        "adot_image": OLD_IMAGE,
        "clusters": {"gpu-a": {"dcgm_image": OLD_IMAGE}},
    }
    loaded = {
        "phase": "rolled-back",
        "previous": previous,
        "rollback_result": {"status": "PASSED"},
    }
    if fault == "dcgm-conflict":
        previous["clusters"]["gpu-b"] = {"dcgm_image": NEW_IMAGE}
    elif fault == "adopted":
        loaded = {"phase": "complete", "adopted_live_runtime_image": OLD_IMAGE}
    elif fault == "unchanged":
        loaded = {"phase": "complete"}
    release.runner.handler = json_response({"data": {"state.json": json.dumps(loaded)}})
    received = []

    def quick(instance: Any, **kwargs: Any) -> dict[str, Any]:
        received.append(copy.deepcopy(kwargs["state"]))
        return {"image": instance.runtime_image, "executor": instance.executor_image}

    monkeypatch.setattr(commands, "build_quick_status", quick)
    if fault == "dcgm-conflict":
        with pytest.raises(ReleaseError, match="disagree on the DCGM image"):
            release.status()
        assert received == []
    else:
        report = release.status()
        expected = original if fault == "unchanged" else OLD_IMAGE
        assert report == {"image": expected, "executor": expected}
        assert received[0]["phase"] == loaded["phase"]


@pytest.mark.parametrize(
    "method,symbol",
    [
        ("activate_cluster", "activate_join_registry"),
        ("fail_cluster", "fail_join_registry"),
        ("rollback_cluster", "purge_failed_join"),
    ],
)
def test_membership_commands_bind_known_cluster_before_public_registry_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, method: str, symbol: str
) -> None:
    release = real_release(tmp_path)
    calls = []
    monkeypatch.setattr(
        rollout,
        symbol,
        lambda _release, target: calls.append(
            target.cluster_id if symbol == "purge_failed_join" else target
        ),
    )
    operation = getattr(release, method)
    operation("gpu-a")
    with pytest.raises(ReleaseError, match="unknown cluster_id"):
        operation("foreign")
    assert calls == ["gpu-a"]
