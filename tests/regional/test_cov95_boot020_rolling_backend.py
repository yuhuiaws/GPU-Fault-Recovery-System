"""BOOT-020 live backend reads and the build-at-execute entry path, all offline."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import (
    regional_release_gpu_rollout,
    regional_release_node_preflight,
)
from gpu_fault_release.regional_release_config import ReleaseError
from scripts.e2e.regional import run_boot020_release_rolling as rolling
from tests.regional._cov95_boot_extra_rolling import (
    BackendModel,
    EngineConfig,
    ReleaseEngine,
    RollingEntry,
)
from tests.regional._cov95_boot_extra_safety import (
    boot_extra_isolation as boot_extra_isolation,
)
from tests.regional.test_boot020_runner import FakeReleaseRollingBackend


class CommittingEngine(ReleaseEngine):
    """The recording engine plus the commit and the resumable rollback."""

    def commit_release(self) -> None:
        self.calls.append(("commit",))
        self.state["transaction_committed"] = True

    def rollback(self) -> None:
        self.calls.append(("rollback",))
        self.state.update(phase="rolled-back", rollback_cleanup_completed=True)


class Model(BackendModel):
    def __init__(
        self,
        *,
        state: dict[str, Any] | None = None,
        clusters: tuple[Any, ...] | None = None,
    ) -> None:
        super().__init__()
        if clusters is not None:
            self.config = EngineConfig(clusters=clusters)
        self.initial_state = state

    def release(self, config: Any, runner: Any) -> CommittingEngine:
        engine = CommittingEngine(self, config, runner)
        if self.initial_state is not None:
            engine.state = copy.deepcopy(self.initial_state)
        self.engines.append(engine)
        return engine


def cluster(name: str) -> SimpleNamespace:
    return SimpleNamespace(cluster_id=name, context=f"context-{name}")


def test_node_safety_records_every_clusters_blockers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Model(clusters=tuple(map(cluster, ("clean", "listed", "garbled", "plain"))))
    backend = model.backend()
    listed: list[tuple[str, bool]] = []
    checked: list[tuple[str, tuple[str, ...]]] = []

    def gpu_node_items(release: Any, target: Any, *, fresh: bool) -> list[dict]:
        listed.append((target.cluster_id, fresh))
        return [
            {"metadata": {"name": f"{target.cluster_id}-b"}},
            {"metadata": {"name": f"{target.cluster_id}-a"}},
            {"metadata": None},
        ]

    def validate(release: Any, target: Any, names: tuple[str, ...]) -> None:
        assert release is model.engines[0]
        checked.append((target.cluster_id, names))
        if target.cluster_id == "listed":
            raise ReleaseError(
                'GPU nodes are not safe: [{"node": "listed-a", "reason": "cordoned"}]'
            )
        if target.cluster_id == "garbled":
            raise ReleaseError("GPU nodes are not safe: [cordoned")
        if target.cluster_id == "plain":
            raise ReleaseError("the node listing is unavailable")

    monkeypatch.setattr(regional_release_gpu_rollout, "gpu_node_items", gpu_node_items)
    monkeypatch.setattr(
        regional_release_node_preflight, "validate_target_node_state", validate
    )

    result = backend.node_safety("executor")

    assert result["safe"] is False
    assert result["clusters"] == {
        "clean": [],
        "listed": [{"node": "listed-a", "reason": "cordoned"}],
        "garbled": "GPU nodes are not safe: [cordoned",
        "plain": "the node listing is unavailable",
    }
    assert result["checked_at"], "the safety read is timestamped"
    assert listed == [(name, True) for name in ("clean", "listed", "garbled", "plain")]
    assert checked[0] == ("clean", ("", "clean-a", "clean-b"))
    assert model.loaded == [backend.configs["executor"]]


def test_node_safety_is_safe_when_every_cluster_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = Model().backend()
    monkeypatch.setattr(
        regional_release_gpu_rollout, "gpu_node_items", lambda *_a, **_k: []
    )
    monkeypatch.setattr(
        regional_release_node_preflight,
        "validate_target_node_state",
        lambda *_args: None,
    )

    result = backend.node_safety("agent")

    assert (result["safe"], result["clusters"]) == (True, {"cluster-a": []})


def test_resume_rollback_finishes_a_pending_rollback_the_way_deploy_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Model(
        state={
            "phase": "rollback-failed",
            "release_id": "unit-release",
            "rollback_failure": "agent wave timed out",
        }
    )
    backend = model.backend()
    backend.commands = rolling.regional_admin_commands
    monkeypatch.setattr(rolling, "time", model)

    result = backend.resume_rollback("agent")

    assert result == {
        "resumed": True,
        "phase_before": "rollback-failed",
        "rollback_failure": "agent wave timed out",
        "phase": "rolled-back",
        "operation_duration_seconds": 1.0,
    }
    assert ("rollback",) in model.engines[0].calls


def test_resume_rollback_touches_nothing_without_a_pending_rollback() -> None:
    model = Model()
    backend = model.backend()
    backend.commands = rolling.regional_admin_commands

    assert backend.resume_rollback("agent") == {"resumed": False, "phase": "complete"}
    assert ("rollback",) not in model.engines[0].calls


def test_cpu_env_alignment_reads_the_scenario_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Model()
    backend = model.backend()
    calls: list[tuple[Any, tuple[str, ...]]] = []

    def alignment(release: Any, deployments: tuple[str, ...]) -> dict[str, Any]:
        calls.append((release, deployments))
        return {"aligned": True}

    monkeypatch.setattr(rolling.restore_stage, "live_cpu_env_alignment", alignment)

    assert backend.cpu_env_alignment("full") == {"aligned": True}
    assert calls == [
        (
            model.engines[0],
            rolling.regional_deployment_inventory.CPU_RUNTIME_DEPLOYMENTS,
        )
    ]
    assert model.loaded == [backend.configs["full"]]


@pytest.mark.parametrize("commit", [True, False])
def test_deploy_commits_a_completed_release_only_when_asked(
    monkeypatch: pytest.MonkeyPatch, commit: bool
) -> None:
    model = Model()
    backend = model.backend()
    monkeypatch.setattr(rolling, "time", model)

    result = backend.deploy("noop", diff={"kind": "NOOP", "changed": []}, commit=commit)

    engine = model.engines[0]
    assert result["phase"] == "complete"
    assert result["transaction_committed"] is commit
    assert (("commit",) in engine.calls) is commit


def pending_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[RollingEntry, list[tuple[str, Any]]]:
    entry = RollingEntry(tmp_path, monkeypatch)
    events: list[tuple[str, Any]] = []
    pending = {
        "action": "build-at-execute",
        "candidates_dir": str(tmp_path / "candidates"),
    }

    def resolve(arguments: Any) -> tuple[dict[str, Path], dict[str, Any]]:
        events.append(("resolve", arguments.admin_state_dir))
        return dict(entry.configs), dict(pending)

    def ensure(arguments: Any, record: dict[str, Any]) -> dict[str, Any]:
        events.append(("ensure", (record, len(entry.calls))))
        return {"action": "explicit"}

    monkeypatch.setattr(rolling.prerequisites, "resolve_release_configs", resolve)
    monkeypatch.setattr(rolling.prerequisites, "ensure_for_execute", ensure)
    monkeypatch.setattr(
        rolling,
        "validate_admin_target",
        lambda state_dir, config: events.append(("validate", (state_dir, config))),
    )
    return entry, events


def test_a_pending_build_plans_without_digests_or_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    entry, events = pending_entry(tmp_path, monkeypatch)
    for path in entry.configs.values():
        path.unlink()  # not built yet: a plan never needs the files
    monkeypatch.setattr(sys, "argv", entry.argv[: entry.argv.index("--execute")])

    assert rolling.main() == 0

    plan = json.loads(capsys.readouterr().out)
    details = plan["details"]
    assert details["config_sha256"] == {}
    assert details["release_candidates_identity"] is None
    assert details["release_candidates"]["action"] == "build-at-execute"
    assert [name for name, _ in events] == ["resolve"]
    assert entry.calls == [], "a plan never authorizes"


def test_an_execute_run_builds_pending_candidates_only_after_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    entry, events = pending_entry(tmp_path, monkeypatch)
    backend = FakeReleaseRollingBackend()
    monkeypatch.setattr(rolling, "LiveReleaseRollingBackend", lambda configs: backend)
    monkeypatch.setattr(sys, "argv", entry.argv)

    assert rolling.main() == 0

    names = [name for name, _ in events]
    assert names == ["resolve", "ensure", "validate"]
    record, authorized = events[1][1]
    assert record["action"] == "build-at-execute"
    assert authorized == 1, "the build ran after authorize_execution"
    assert events[2][1] == (tmp_path.resolve(), entry.configs["noop"])
    document = json.loads(entry.path.read_text(encoding="utf-8"))
    assert document["verdict"] == "PASS"
    provenance = document["release_candidates_provenance"]
    assert [item["action"] for item in provenance] == ["explicit"]
    assert json.loads(capsys.readouterr().out)["verdict"] == "PASS"
