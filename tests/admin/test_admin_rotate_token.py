from __future__ import annotations

import hashlib
import json
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import cli
from gpu_fault.admin import rotate_token as module
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseError

OLD_TOKEN = "a" * 64
OLD_DIGEST = hashlib.sha256(OLD_TOKEN.encode()).hexdigest()
OTHER_TOKEN = "b" * 64
GPU_ARN = "arn:aws:eks:us-west-2:123456789012:cluster/gpu-a"
NOW = datetime(2026, 9, 8, 6, 0, tzinfo=timezone.utc)


def _site(tmp_path: Path) -> SimpleNamespace:
    secure = tmp_path / "secure"
    secure.mkdir(mode=0o700)
    token_file = secure / "gpu-a.token"
    token_file.write_text(OLD_TOKEN, encoding="utf-8")
    token_file.chmod(0o600)
    (tmp_path / "site.yaml").write_text("name: staging\n", encoding="utf-8")
    return SimpleNamespace(
        source=tmp_path / "site.yaml",
        repository_root=tmp_path,
        environment={},
        source_sha256="c" * 64,
        release_config={
            "site_name": "staging",
            "aws_region": "us-west-2",
            "cpu_kubeconfig": str(tmp_path / "cpu.kubeconfig"),
            "cpu_eks_arn": "arn:aws:eks:us-west-2:123456789012:cluster/cpu",
            "namespace": "gpu-fault-system",
            "runtime_profile": {"version": "hyperpod-v1"},
            "clusters": [
                {
                    "cluster_id": "gpu-a",
                    "context": "gpu-a-context",
                    "eks_cluster_arn": GPU_ARN,
                    "token_file": str(token_file),
                },
                {
                    "cluster_id": "gpu-b",
                    "context": "gpu-b-context",
                    "eks_cluster_arn": GPU_ARN.replace("gpu-a", "gpu-b"),
                    "token_file": str(secure / "gpu-b.token"),
                },
            ],
        },
    )


def _target(token_file: str = "/secure/gpu-a.token") -> ClusterTarget:
    return ClusterTarget(
        cluster_id="gpu-a",
        context="gpu-a-context",
        executor_irsa_role_arn="arn:aws:iam::123456789012:role/executor",
        region="us-west-2",
        hyperpod_cluster_name="hp-a",
        eks_cluster_arn=GPU_ARN,
        token_file=token_file,
    )


def _release(token_file: str = "/secure/gpu-a.token") -> SimpleNamespace:
    runner = SimpleNamespace(calls=[], dry_run=False)
    runner.run = lambda args, **kwargs: runner.calls.append(list(args)) or ""
    target = _target(token_file)
    return SimpleNamespace(
        runner=runner,
        config=SimpleNamespace(
            namespace="gpu-fault-system",
            agent_config_digest="d" * 64,
            runtime_profile_version="hyperpod-v1",
            component_digests={},
            upgrade_max_unavailable=0,
        ),
        executor_wheel_cm="wheel-cm",
        bundle_cm="bundle-cm",
        node_wheel_sha="e" * 64,
        idle=True,
        _target=lambda cluster_id: target,
        _remote_commands_are_idle=lambda: True,
        _gpu=lambda target, *args: ["kubectl", "--context", target.context, *args],
    )


def _secret_entries() -> list[dict]:
    return [
        {"cluster_id": "gpu-a", "token": OLD_TOKEN, "allowed_namespaces": []},
        {"cluster_id": "gpu-b", "token": OTHER_TOKEN, "allowed_namespaces": []},
    ]


class Harness:
    """The engine seams the rotation composes, replaced by recorders."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.site = _site(tmp_path)
        self.release = _release(self.site.release_config["clusters"][0]["token_file"])
        self.calls: list[tuple] = []
        self.secret = _secret_entries()
        self.publishes: list[dict] = []
        self.acceptance_error: BootstrapError | None = None
        self.publish_errors: list[Exception] = []
        self.clock = NOW

        def registrations(release, overrides):
            result = []
            for item in self.secret:
                entry = dict(item)
                token = entry.pop("token")
                entry["token_sha256"] = hashlib.sha256(token.encode()).hexdigest()
                entry["lifecycle_state"] = "ACTIVE"
                result.append(entry)
            return result

        def publish(release, *, path, payload, use_current_generation, timeout_seconds):
            if self.publish_errors:
                raise self.publish_errors.pop(0)
            self.publishes.append(payload)
            self.calls.append(("publish", payload["reason"]))
            return {
                "generation": len(self.publishes),
                "content_sha256": "f" * 64,
                "converged": True,
                "missing_member_ids": [],
            }

        def write_registry(release, entries, *, backup=None):
            self.secret = [dict(item) for item in entries]
            self.calls.append(
                ("write_registry", [item["cluster_id"] for item in entries])
            )

        def ensure_secret(release, target):
            self.calls.append(
                ("connection_secret", Path(target.token_file).read_text())
            )

        def restart(release, target):
            self.calls.append(("restart_data_plane", target.cluster_id))
            return {"deployments": ["executor"]}

        def roll(release, target, *, rotation_id, progress, record, only_nodes=None):
            progress["completed_nodes"] = sorted(only_nodes or {"node-1", "node-2"})
            record()
            self.calls.append(("roll_node_tokens", only_nodes))
            return {"node_count": 2, "waves": [["node-1"], ["node-2"]]}

        def accept(site, cluster_id, *, quiet_seconds, timeout_seconds):
            self.calls.append(("acceptance", cluster_id, quiet_seconds))
            if self.acceptance_error is not None:
                raise self.acceptance_error
            return {"quiet_seconds": quiet_seconds, "waited_seconds": 1.0}

        monkeypatch.setattr(module, "build_release", lambda site: self.release)
        monkeypatch.setattr(
            module,
            "live_release_state",
            lambda site: {"phase": "complete", "transaction_committed": True},
        )
        monkeypatch.setattr(
            module,
            "remote_command_stats",
            lambda release: {"by_status": {"SUCCEEDED": 4, "FAILED": 1}},
        )
        monkeypatch.setattr(module, "current_registrations", registrations)
        monkeypatch.setattr(module, "publish_registry_revision", publish)
        monkeypatch.setattr(
            module, "registry", lambda release: [dict(i) for i in self.secret]
        )
        monkeypatch.setattr(module, "write_registry", write_registry)
        monkeypatch.setattr(module, "ensure_connection_secret", ensure_secret)
        monkeypatch.setattr(module, "restart_data_plane", restart)
        monkeypatch.setattr(module, "roll_node_tokens", roll)
        monkeypatch.setattr(module, "wait_for_new_token_acceptance", accept)

    def now(self) -> datetime:
        self.clock += timedelta(seconds=1)
        return self.clock

    def request(self, **overrides) -> module.RotateTokenRequest:
        values = {
            "site": self.site,
            "cluster_id": "gpu-a",
            "window": timedelta(minutes=30),
        }
        values.update(overrides)
        return module.RotateTokenRequest(**values)

    def rotate(self, **overrides) -> dict:
        return module.rotate_cluster_token(self.request(**overrides), now=self.now)

    @property
    def token_file(self) -> Path:
        return Path(self.site.release_config["clusters"][0]["token_file"])

    @property
    def state(self) -> dict:
        return json.loads(
            module.rotation_state_path(self.site, "gpu-a").read_text(encoding="utf-8")
        )


def test_rotation_runs_every_step_and_writes_the_token_file_only_after_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)

    summary = harness.rotate()

    kinds = [call[0] for call in harness.calls]
    assert kinds == [
        "publish",
        "connection_secret",
        "restart_data_plane",
        "roll_node_tokens",
        "acceptance",
        "write_registry",
        "publish",
    ]
    overlap = harness.publishes[0]["registrations"]
    target = next(item for item in overlap if item["cluster_id"] == "gpu-a")
    other = next(item for item in overlap if item["cluster_id"] == "gpu-b")
    new_token = harness.token_file.read_text(encoding="utf-8")
    assert target["retiring_token_sha256"] == OLD_DIGEST
    assert target["token_sha256"] == hashlib.sha256(new_token.encode()).hexdigest()
    state = harness.state
    assert target["token_rotation_expires_at"] == state["expires_at"]
    window = datetime.fromisoformat(state["expires_at"]) - datetime.fromisoformat(
        state["started_at"]
    )
    assert timedelta(minutes=30) <= window <= timedelta(minutes=30, seconds=5)
    assert "retiring_token_sha256" not in other, (
        "other clusters are published untouched"
    )
    # The GPU Secret was rendered from the pending file, not the site's token.
    assert harness.calls[1][1] == new_token
    assert new_token != OLD_TOKEN and len(new_token) == 64
    assert stat.S_IMODE(harness.token_file.stat().st_mode) == 0o600
    retired = Path(summary["retired_token_file"])
    assert retired.read_text(encoding="utf-8") == OLD_TOKEN
    # The bootstrap Secret carries the accepted token and the final revision
    # drops the retiring digest.
    assert harness.secret[0]["token"] == new_token
    final = next(
        item
        for item in harness.publishes[1]["registrations"]
        if item["cluster_id"] == "gpu-a"
    )
    assert "retiring_token_sha256" not in final
    assert summary["status"] == module.STATUS_COMPLETED
    assert summary["new_token_sha256"] == target["token_sha256"]
    assert new_token not in json.dumps(summary), "the summary never carries a token"
    assert not (
        tmp_path / module.STATE_ROOT / "gpu-a" / module.PENDING_TOKEN_FILE
    ).exists(), "the pending token file is removed once the site's file holds it"


def test_token_file_stays_old_until_acceptance_and_a_rerun_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.acceptance_error = BootstrapError("still seeing the retiring token")

    with pytest.raises(BootstrapError, match="retiring token"):
        harness.rotate()

    assert harness.token_file.read_text(encoding="utf-8") == OLD_TOKEN
    state = harness.state
    assert module.STEP_NODES_ROLLED in state["steps"]
    assert module.STEP_ACCEPTED not in state["steps"]
    pending = Path(state["pending_token_file"])
    assert pending.is_file(), "the pending token survives for the resume"
    pending_token = pending.read_text(encoding="utf-8")

    harness.acceptance_error = None
    before = len(harness.calls)
    summary = harness.rotate()

    resumed = [call[0] for call in harness.calls[before:]]
    assert resumed == ["acceptance", "write_registry", "publish"], (
        "the resume repeats nothing the state file already recorded"
    )
    assert harness.token_file.read_text(encoding="utf-8") == pending_token
    assert summary["status"] == module.STATUS_COMPLETED
    assert summary["rotation_id"] == state["rotation_id"]


def test_failed_overlap_publish_rolls_the_registry_back_to_the_old_token_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.publish_errors.append(ReleaseError("registry did not converge"))

    with pytest.raises(BootstrapError, match="restored to the old token only"):
        harness.rotate()

    assert [call[0] for call in harness.calls] == ["publish"]
    rollback = harness.publishes[0]
    assert "rollback" in rollback["reason"]
    target = next(
        item for item in rollback["registrations"] if item["cluster_id"] == "gpu-a"
    )
    assert target["token_sha256"] == OLD_DIGEST
    assert "retiring_token_sha256" not in target
    assert module.STEP_OVERLAP_PUBLISHED not in harness.state["steps"]
    assert harness.token_file.read_text(encoding="utf-8") == OLD_TOKEN


def test_refuses_busy_remote_commands_and_open_release_transactions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.release._remote_commands_are_idle = lambda: False

    with pytest.raises(BootstrapError, match="PENDING/LEASED/WAITING"):
        harness.rotate()
    assert harness.publishes == []

    harness.release._remote_commands_are_idle = lambda: True
    monkeypatch.setattr(
        module, "live_release_state", lambda site: {"phase": "data-plane-progress"}
    )
    with pytest.raises(BootstrapError, match="release transaction is open"):
        harness.rotate()
    assert harness.publishes == []


@pytest.mark.parametrize(
    ("state", "open_phase"),
    [
        ({"phase": "complete", "transaction_committed": True}, None),
        ({"phase": "complete", "transaction_committed": False}, "complete"),
        ({"phase": "data-plane-progress"}, "data-plane-progress"),
        ({"phase": "failed"}, "failed"),
        ({"phase": "rollback-data-progress"}, "rollback-data-progress"),
        ({"phase": "bootstrap-started"}, "bootstrap-started"),
        ({"phase": "bootstrap-cleaned"}, None),
        ({"phase": "rolled-back", "rollback_cleanup_completed": False}, "rolled-back"),
        ({"phase": "rolled-back", "rollback_cleanup_completed": True}, None),
    ],
)
def test_release_transaction_open_uses_the_engine_phase_vocabulary(
    state: dict, open_phase: str | None
) -> None:
    assert module.release_transaction_open(state) == open_phase


def test_registry_digest_drift_from_the_token_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.secret[0]["token"] = "z" * 64

    with pytest.raises(BootstrapError, match="does not match the site's token file"):
        harness.rotate()
    assert harness.publishes == [], "a drifted Secret is never republished"


def test_window_bounds_are_enforced(tmp_path: Path) -> None:
    site = _site(tmp_path)
    with pytest.raises(BootstrapError, match="between 10 and"):
        module.RotateTokenRequest(
            site=site, cluster_id="gpu-a", window=timedelta(minutes=5)
        )
    with pytest.raises(BootstrapError, match="between 10 and"):
        module.RotateTokenRequest(
            site=site, cluster_id="gpu-a", window=timedelta(days=8)
        )


def test_keep_window_leaves_the_retiring_token_to_expire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)

    summary = harness.rotate(keep_window=True)

    assert [call[0] for call in harness.calls].count("publish") == 1
    assert harness.secret[0]["token"] == harness.token_file.read_text(encoding="utf-8")
    assert any("--keep-window" in item for item in summary["warnings"]), (
        "the summary says the retiring token was left to expire"
    )
    assert summary["status"] == module.STATUS_COMPLETED


def test_rollback_walks_the_data_plane_back_before_restoring_the_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.acceptance_error = BootstrapError("still seeing the retiring token")
    with pytest.raises(BootstrapError):
        harness.rotate()
    harness.calls.clear()
    harness.publishes.clear()

    summary = harness.rotate(rollback=True)

    assert [call[0] for call in harness.calls] == [
        "connection_secret",
        "restart_data_plane",
        "roll_node_tokens",
        "publish",
    ]
    assert harness.calls[0][1] == OLD_TOKEN, (
        "the Secret is rendered from the site's file"
    )
    assert harness.calls[2][1] == frozenset({"node-1", "node-2"}), (
        "only the nodes the forward pass moved are re-rolled"
    )
    restored = next(
        item
        for item in harness.publishes[0]["registrations"]
        if item["cluster_id"] == "gpu-a"
    )
    assert restored["token_sha256"] == OLD_DIGEST
    assert "retiring_token_sha256" not in restored
    assert summary["status"] == module.STATUS_ROLLED_BACK
    assert harness.token_file.read_text(encoding="utf-8") == OLD_TOKEN
    assert not Path(harness.state["pending_token_file"]).exists(), (
        "the pending token is shredded on rollback"
    )


def test_rollback_is_refused_once_the_token_file_was_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.rotate()

    with pytest.raises(BootstrapError, match="no rotation is in progress"):
        harness.rotate(rollback=True)


def test_a_completed_rotation_is_archived_and_a_rerun_starts_a_new_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    first = harness.rotate()
    second = harness.rotate()

    assert second["rotation_id"] != first["rotation_id"]
    assert second["old_token_sha256"] == first["new_token_sha256"]
    history = tmp_path / module.STATE_ROOT / "gpu-a" / "history"
    assert sorted(path.stem for path in history.glob("*.json")) == [
        first["rotation_id"]
    ]


def test_write_token_file_replaces_atomically_and_keeps_the_retired_copy(
    tmp_path: Path,
) -> None:
    token_file = tmp_path / "gpu-a.token"
    token_file.write_text(OLD_TOKEN, encoding="utf-8")

    retired = module.write_token_file(token_file, "n" * 64, now=NOW)

    assert token_file.read_text(encoding="utf-8") == "n" * 64
    assert retired.read_text(encoding="utf-8") == OLD_TOKEN
    assert retired.name == "gpu-a.token.retired-20260908T060000Z"
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(retired.stat().st_mode) == 0o600
    assert not (tmp_path / ".gpu-a.token.rotating").exists(), (
        "the temporary file is consumed by the replace"
    )


def test_roll_node_tokens_hands_each_wave_marks_it_retrying_then_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = _release()
    target = _target()
    events: list[tuple] = []
    wave_config = {"allowed-nodes": "*", "max-unavailable": "1", "generation": "steady"}

    monkeypatch.setattr(
        module,
        "target_node_names",
        lambda release, target: ("node-1", "node-2", "node-3"),
    )
    monkeypatch.setattr(
        module,
        "target_node_failure_domains",
        lambda release, target, names: {name: "rack-a" for name in names},
    )
    monkeypatch.setattr(
        module,
        "node_rollout_policy",
        lambda release, domains, *, phase: module.NodeRolloutPolicy(
            max_unavailable=2,
            first_wave_max_unavailable=1,
            max_unavailable_per_failure_domain=2,
        ),
    )
    monkeypatch.setattr(
        module, "validate_target_node_state", lambda *args: events.append(("validate",))
    )
    monkeypatch.setattr(
        module,
        "reconciler_container_env",
        lambda release, target: {module.INSTALLER_WAVE_CONFIG_MAP_ENV: "wave-cm"},
    )
    monkeypatch.setattr(
        module, "reconciler_installer_identity", lambda env: ("bundle" * 8, "tmpl" * 16)
    )
    monkeypatch.setattr(
        module, "read_wave_config", lambda release, target, name: dict(wave_config)
    )
    monkeypatch.setattr(
        module,
        "ensure_rollout_wave_safe",
        lambda release, target, *, wave, node_names: events.append(("safe", wave)),
    )
    monkeypatch.setattr(
        module,
        "hand_wave_to_reconciler",
        lambda release, target, context, wave: events.append(("handoff", wave))
        or context.paused_identity,
    )
    monkeypatch.setattr(
        module,
        "wait_agents",
        lambda release, target, artifact, **kwargs: events.append(
            ("wait", kwargs.get("node_names", ()))
        ),
    )
    monkeypatch.setattr(
        module,
        "restore_wave_config",
        lambda release, target, name, steady: events.append(
            ("restore", name, dict(steady))
        ),
    )
    progress: dict = {"completed_nodes": ["node-1"]}
    saves: list[list[str]] = []

    result = module.roll_node_tokens(
        release,
        target,
        rotation_id="r1",
        progress=progress,
        record=lambda: saves.append(list(progress["completed_nodes"])),
    )

    annotate_calls = [call for call in release.runner.calls if "annotate" in call]
    assert [call[-2] for call in annotate_calls] == [
        f"{module.INSTALLER_STATE_ANNOTATION}={module.REINSTALL_STATE}"
    ] * 2, "every wave is marked Retrying exactly once"
    assert events == [
        ("validate",),
        ("safe", ("node-2",)),
        ("handoff", ("node-2",)),
        ("wait", ("node-2",)),
        ("safe", ("node-3",)),
        ("handoff", ("node-3",)),
        ("wait", ("node-3",)),
        ("restore", "wave-cm", wave_config),
        ("wait", ()),
    ], (
        "node-1 was already moved; the rest roll one node per wave then the whole cluster is re-checked"
    )
    # Each wave: handoff before the Retrying mark, mark before the wait.
    handoff_index = release.runner.calls.index(annotate_calls[0])
    assert handoff_index >= 0
    assert saves[-1] == ["node-1", "node-2", "node-3"]
    assert progress["steady_wave_config"] == wave_config
    assert result["reinstalled_nodes"] == ["node-1", "node-2", "node-3"]


def test_roll_node_tokens_refuses_a_reconciler_that_is_mid_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = _release()
    monkeypatch.setattr(
        module, "target_node_names", lambda release, target: ("node-1",)
    )
    monkeypatch.setattr(module, "validate_target_node_state", lambda *args: None)
    monkeypatch.setattr(
        module,
        "reconciler_container_env",
        lambda release, target: {module.INSTALLER_WAVE_CONFIG_MAP_ENV: "wave-cm"},
    )
    monkeypatch.setattr(
        module, "reconciler_installer_identity", lambda env: ("b" * 64, "t" * 64)
    )
    monkeypatch.setattr(
        module,
        "read_wave_config",
        lambda release, target, name: {
            "allowed-nodes": "node-9",
            "max-unavailable": "1",
        },
    )

    with pytest.raises(BootstrapError, match="mid-wave"):
        module.roll_node_tokens(
            release,
            _target(),
            rotation_id="r1",
            progress={"completed_nodes": []},
            record=lambda: None,
        )
    assert release.runner.calls == [], "nothing is annotated or patched"


def test_wait_for_new_token_acceptance_needs_a_full_quiet_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path)
    observations = [
        ["regional cluster gpu-a authenticated with the retiring token"],
        [],
    ]
    monkeypatch.setattr(
        module,
        "retiring_token_authentications",
        lambda site, cluster_id, *, since_seconds: observations.pop(0),
    )
    clock = {"now": 0.0}
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["now"] += seconds

    result = module.wait_for_new_token_acceptance(
        site,
        "gpu-a",
        quiet_seconds=60,
        timeout_seconds=600,
        sleep=sleep,
        monotonic=lambda: clock["now"],
    )

    assert sleeps[0] == 60, "the first quiet window is always waited out"
    assert result["quiet_seconds"] == 60
    assert observations == [], (
        "a retiring-token line inside the window means wait again"
    )

    monkeypatch.setattr(
        module,
        "retiring_token_authentications",
        lambda site, cluster_id, *, since_seconds: ["still the old token"],
    )
    with pytest.raises(BootstrapError, match="still holds the old token"):
        module.wait_for_new_token_acceptance(
            site,
            "gpu-a",
            quiet_seconds=60,
            timeout_seconds=120,
            sleep=sleep,
            monotonic=lambda: clock["now"],
        )


def test_remote_command_losses_reports_only_grown_terminal_counters() -> None:
    baseline = {"by_status": {"FAILED": 1, "EXPIRED": 0, "SUCCEEDED": 3}}
    current = {"by_status": {"FAILED": 1, "EXPIRED": 2, "SUCCEEDED": 9}}
    assert module.remote_command_losses(baseline, current) == {"EXPIRED": 2}


def test_run_rotate_token_command_resolves_the_arn_without_aws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path)
    captured: list[module.RotateTokenRequest] = []
    monkeypatch.setattr(
        module,
        "rotate_cluster_token",
        lambda request: captured.append(request) or {"status": "COMPLETED"},
    )
    arguments = cli.parser().parse_args(
        [
            "rotate-token",
            "--state-dir",
            str(tmp_path),
            "--gpu-cluster-arn",
            GPU_ARN,
            "--window-minutes",
            "45",
            "--reference",
            "CHG-1",
        ]
    )

    assert module.run_rotate_token_command(arguments, site=site) == 0
    request = captured[0]
    assert request.cluster_id == "gpu-a"
    assert request.window == timedelta(minutes=45)
    assert request.reference == "CHG-1"
    assert request.quiet_seconds == module.DEFAULT_QUIET_SECONDS
    assert request.rollback is False and request.keep_window is False


def test_cli_dispatches_rotate_token_with_the_managed_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path)
    seen: list[tuple] = []
    monkeypatch.setattr(cli, "load_site", lambda path, *, repository_root: site)
    monkeypatch.setattr(
        cli,
        "run_rotate_token_command",
        lambda arguments, *, site: seen.append((arguments.command, site)) or 0,
    )
    arguments = cli.parser().parse_args(
        ["rotate-token", "--state-dir", str(tmp_path), "--gpu-cluster-arn", GPU_ARN]
    )

    assert cli.run(arguments) == 0
    assert seen == [("rotate-token", site)]
