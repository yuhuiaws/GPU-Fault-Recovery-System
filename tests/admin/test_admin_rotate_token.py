from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import cli
from gpu_fault.admin import rotate_token as module
from gpu_fault.admin import rotate_token_acceptance as acceptance_module
from gpu_fault.admin import rotate_token_binding as binding
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault_release import regional_release_control_plane_ready as control_plane
from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseError
from tests.admin.test_admin_rotate_token_acceptance import LogHost

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
    # The publish retry asks whether its exec target still exists; every Pod
    # this release is asked about is present unless a test says otherwise.
    runner.probe_output = lambda args, timeout_seconds=None: (
        0,
        json.dumps(
            {
                "kind": "Pod",
                "metadata": {
                    "name": args[args.index("pod") + 1],
                    "namespace": "gpu-fault-system",
                    "uid": "pod-uid",
                },
            }
        ),
        "",
    )
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
        _cpu=lambda *args: ["kubectl", "--kubeconfig", "cpu.kubeconfig", *args],
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
        self.control_plane_error: Exception | None = None
        self.publish_errors: list[Exception] = []
        self.readiness_waits: list[tuple] = []
        # The durable registry head a test wants the Secret to reconstruct;
        # ``None`` derives it from the last published revision (or the Secret).
        self.durable: list[dict] | None = None
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

        def restart_control_plane(release):
            # Recorded with what the control plane would load (the Secret's
            # token for gpu-a) and how many revisions are published so far.
            self.calls.append(
                ("restart_control_plane", self.secret[0]["token"], len(self.publishes))
            )
            if self.control_plane_error is not None:
                raise self.control_plane_error
            return {
                "deployments": ["api-ha"],
                "rollouts": {"api-ha": {"duration_seconds": 1.0, "progress": [1]}},
            }

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

        def wait_ready(release, deployments=None, **options):
            # Recorded with how many revisions are published so far and the
            # steps run before it, so a test can place it between the roll and
            # the final publish.
            self.readiness_waits.append(
                (len(self.publishes), [call[0] for call in self.calls])
            )
            return {
                "waited_seconds": 0.5,
                "registry_generation": len(self.publishes),
                "deployments": {},
            }

        def durable_registrations(release, *, retiring_digests=None):
            entries = self.durable
            if entries is None:
                entries = (
                    self.publishes[-1]["registrations"]
                    if self.publishes
                    else registrations(release, {})
                )
            for item in entries:
                retiring = item.get("retiring_token_sha256")
                if not retiring:
                    continue
                supplied = (retiring_digests or {}).get(item["cluster_id"])
                if supplied is None:
                    # The Secret carries one token per cluster, so a head with
                    # a retiring digest cannot be reconstructed from it alone.
                    raise ReleaseError("regional registry credential identity differs")
                if supplied != retiring:
                    raise ReleaseError("regional registry snapshot identity differs")
            status = SimpleNamespace(
                generation=len(self.publishes), content_sha256="f" * 64
            )
            return status, [
                SimpleNamespace(
                    cluster_id=item["cluster_id"],
                    token_sha256=item["token_sha256"],
                    retiring_token_sha256=item.get("retiring_token_sha256"),
                )
                for item in entries
            ]

        monkeypatch.setattr(binding, "durable_registrations", durable_registrations)
        monkeypatch.setattr(control_plane, "wait_control_plane_ready", wait_ready)
        monkeypatch.setattr(
            control_plane, "select_current_ingress_pod", lambda release: "api-ha-new-1"
        )
        monkeypatch.setattr(control_plane, "PUBLISH_RETRY_DELAY_SECONDS", 0.0)
        monkeypatch.setattr(module, "build_release", lambda site: self.release)
        monkeypatch.setattr(module, "reload_site_for_mutation", lambda site: site)
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
        monkeypatch.setattr(module, "restart_control_plane", restart_control_plane)
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
        "restart_control_plane",
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
    # The control plane is rolled after the Secret rewrite (so the Pods load
    # the accepted token) and before the final publish (so the retiring digest
    # is still accepted while they roll).
    _kind, loaded_token, published = next(
        call for call in harness.calls if call[0] == "restart_control_plane"
    )
    assert loaded_token == new_token and published == 1, (
        "the control plane must roll between the Secret rewrite and the final publish"
    )
    control_plane = state["steps"][module.STEP_CONTROL_PLANE_ROLLED]["evidence"]
    assert control_plane["registry_secret_rewritten"] is True
    assert control_plane["deployments"] == ["api-ha"]
    assert control_plane["rollouts"]["api-ha"]["progress"] == [1]
    assert new_token not in json.dumps(state), "the journal never carries a token"
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
    assert resumed == [
        "acceptance",
        "write_registry",
        "restart_control_plane",
        "publish",
    ], "the resume repeats nothing the state file already recorded"
    assert harness.token_file.read_text(encoding="utf-8") == pending_token
    assert summary["status"] == module.STATUS_COMPLETED
    assert summary["rotation_id"] == state["rotation_id"]


def test_resume_after_final_publish_failure_uses_the_committed_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    publish = module.publish_current_revision

    def fail_publish(*args, **kwargs):
        raise ReleaseError("final registry publish interrupted")

    monkeypatch.setattr(module, "publish_current_revision", fail_publish)
    with pytest.raises(ReleaseError, match="final registry publish"):
        harness.rotate()

    committed_digest = hashlib.sha256(harness.token_file.read_bytes()).hexdigest()
    assert module.STEP_TOKEN_FILE_WRITTEN in harness.state["steps"]
    before = len(harness.calls)
    monkeypatch.setattr(module, "publish_current_revision", publish)
    result = harness.rotate()

    assert result["status"] == module.STATUS_COMPLETED
    assert result["new_token_sha256"] == committed_digest
    assert [item[0] for item in harness.calls[before:]] == ["publish"], (
        "the completed Secret rewrite and control-plane roll are not repeated"
    )
    assert module.STEP_CONTROL_PLANE_ROLLED in harness.state["steps"]


def test_control_plane_rollout_failure_fails_closed_and_the_rerun_resumes_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.control_plane_error = ReleaseError(
        "control plane Deployment gpu-fault-api-ha exceeded 600 seconds"
    )

    with pytest.raises(ReleaseError, match="gpu-fault-api-ha exceeded"):
        harness.rotate()

    state = harness.state
    assert state["status"] == module.STATUS_IN_PROGRESS
    assert module.STEP_TOKEN_FILE_WRITTEN in state["steps"]
    assert module.STEP_CONTROL_PLANE_ROLLED in state["started_steps"]
    assert module.STEP_CONTROL_PLANE_ROLLED not in state["steps"]
    assert module.STEP_RETIRING_DROPPED not in state["steps"]
    assert len(harness.publishes) == 1, (
        "the retiring digest is not dropped while the control plane has not rolled"
    )
    new_token = harness.token_file.read_text(encoding="utf-8")
    assert harness.secret[0]["token"] == new_token, (
        "the Secret rewrite precedes the restart that loads it"
    )
    with pytest.raises(BootstrapError, match="rollback is forbidden"):
        harness.rotate(rollback=True)

    harness.control_plane_error = None
    before = len(harness.calls)
    summary = harness.rotate()

    assert [call[0] for call in harness.calls[before:]] == [
        "write_registry",
        "restart_control_plane",
        "publish",
    ], "the rerun resumes at the roll, re-asserting the Secret it must load"
    assert summary["status"] == module.STATUS_COMPLETED
    assert summary["steps"][module.STEP_CONTROL_PLANE_ROLLED] is not None
    assert harness.token_file.read_text(encoding="utf-8") == new_token


def test_final_publish_waits_for_the_rolled_control_plane_and_retries_transients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live, the publish raced the roll it had just caused and failed in seconds.

    The readiness wait sits between ``restart_control_plane`` and the final
    publish; a transient failure (the exec target reaped under it) is retried
    once the current ReplicaSet is re-selected, and every attempt is journaled.
    """

    harness = Harness(tmp_path, monkeypatch)
    killed = ReleaseError("command failed (137): kubectl")
    setattr(killed, control_plane.FAILURE_EXIT_CODE_ATTRIBUTE, 137)
    harness_publish = module.publish_registry_revision
    publishes_seen: list[str] = []

    def publish(release, *, path, payload, use_current_generation, timeout_seconds):
        # The overlap publish succeeds; the first final publish is killed under
        # the exec (exit 137), the second lands.
        publishes_seen.append(payload["reason"])
        if payload["reason"].endswith(" final") and len(publishes_seen) == 2:
            raise killed
        return harness_publish(
            release,
            path=path,
            payload=payload,
            use_current_generation=use_current_generation,
            timeout_seconds=timeout_seconds,
        )

    monkeypatch.setattr(module, "publish_registry_revision", publish)

    summary = harness.rotate()

    assert summary["status"] == module.STATUS_COMPLETED
    assert len(harness.readiness_waits) == 1
    published_before_wait, steps_before_wait = harness.readiness_waits[0]
    assert published_before_wait == 1 and steps_before_wait[-1] == (
        "restart_control_plane"
    ), "the readiness wait runs after the roll and before the final publish"
    assert publishes_seen[1:] == [
        f"rotate-token gpu-a {summary['rotation_id']} final",
        f"rotate-token gpu-a {summary['rotation_id']} final",
    ], "the killed publish is retried with the same idempotent revision"
    evidence = harness.state["steps"][module.STEP_RETIRING_DROPPED]["evidence"]
    assert evidence["retiring_token_dropped"] is True
    assert evidence["control_plane_ready"]["waited_seconds"] == 0.5
    assert [item["outcome"] for item in evidence["publish_attempts"]] == [
        "failed",
        "published",
    ]
    assert evidence["publish_attempts"][0]["failure_class"] == "exec-killed"
    assert all(
        item["pod"] == "api-ha-new-1" for item in evidence["publish_attempts"]
    ), "every attempt execs into a Pod of the current ReplicaSet"
    assert harness.state["final_publish_attempts"] == evidence["publish_attempts"]


def test_a_refused_final_publish_is_not_retried_and_the_rerun_resumes_at_the_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    publish = module.publish_current_revision
    attempts = []

    def refuse(*args, **kwargs):
        attempts.append("publish")
        raise ReleaseError("regional registry generation conflict")

    monkeypatch.setattr(module, "publish_current_revision", refuse)
    with pytest.raises(ReleaseError, match="generation conflict") as info:
        harness.rotate()

    assert attempts == ["publish"], "a registry refusal is never repeated"
    assert info.value.__notes__ == [
        "registry publish failed on attempt 1/3 (refused); exec target api-ha-new-1"
    ]
    state = harness.state
    assert state["status"] == module.STATUS_IN_PROGRESS
    assert module.STEP_CONTROL_PLANE_ROLLED in state["steps"]
    assert module.STEP_RETIRING_DROPPED not in state["steps"]
    assert [item["failure_class"] for item in state["final_publish_attempts"]] == [
        "refused"
    ], "the journal keeps the failed attempt for the operator"

    monkeypatch.setattr(module, "publish_current_revision", publish)
    before = len(harness.calls)
    summary = harness.rotate()

    assert [call[0] for call in harness.calls[before:]] == ["publish"], (
        "the rerun resumes at RETIRING_TOKEN_DROPPED without rolling again"
    )
    assert len(harness.readiness_waits) == 2, (
        "the resumed drop waits for the control plane again before publishing"
    )
    assert summary["status"] == module.STATUS_COMPLETED
    evidence = harness.state["steps"][module.STEP_RETIRING_DROPPED]["evidence"]
    assert [item["outcome"] for item in evidence["publish_attempts"]] == ["published"]


def test_a_journal_from_before_the_control_plane_step_resumes_through_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A site left mid-finish by a build without the step gets the restart."""

    harness = Harness(tmp_path, monkeypatch)
    publish = module.publish_current_revision

    def fail_final(*args, **kwargs):
        raise ReleaseError("final registry publish interrupted")

    monkeypatch.setattr(module, "publish_current_revision", fail_final)
    with pytest.raises(ReleaseError, match="final registry publish"):
        harness.rotate()
    monkeypatch.setattr(module, "publish_current_revision", publish)
    path = module.rotation_state_path(harness.site, "gpu-a")
    state = json.loads(path.read_text(encoding="utf-8"))
    del state["steps"][module.STEP_CONTROL_PLANE_ROLLED]
    del state["started_steps"][module.STEP_CONTROL_PLANE_ROLLED]
    path.write_text(json.dumps(state), encoding="utf-8")
    before = len(harness.calls)

    summary = harness.rotate()

    assert [call[0] for call in harness.calls[before:]] == [
        "write_registry",
        "restart_control_plane",
        "publish",
    ]
    assert summary["status"] == module.STATUS_COMPLETED
    assert set(harness.state["steps"]) == set(module.ROTATION_STEPS)


def test_token_replace_ack_loss_resumes_forward_and_refuses_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    write = module.write_token_file

    def write_then_interrupt(*args, **kwargs):
        write(*args, **kwargs)
        raise OSError("token replace acknowledgement lost")

    monkeypatch.setattr(module, "write_token_file", write_then_interrupt)
    with pytest.raises(OSError, match="acknowledgement lost"):
        harness.rotate()
    state = harness.state
    assert module.STEP_TOKEN_FILE_WRITTEN not in state["steps"]
    assert (
        hashlib.sha256(harness.token_file.read_bytes()).hexdigest()
        == state["new_token_sha256"]
    )
    before = len(harness.calls)
    with pytest.raises(BootstrapError, match="token file"):
        harness.rotate(rollback=True)
    assert len(harness.calls) == before

    monkeypatch.setattr(module, "write_token_file", write)
    result = harness.rotate()
    assert result["status"] == module.STATUS_COMPLETED
    assert result["rotation_id"] == state["rotation_id"]
    assert (
        hashlib.sha256(Path(result["retired_token_file"]).read_bytes()).hexdigest()
        == OLD_DIGEST
    )


def test_token_write_intent_is_irreversible_before_the_filesystem_write_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    write = module.write_token_file

    def unavailable_write(*args, **kwargs):
        raise OSError("filesystem write has not started")

    monkeypatch.setattr(module, "write_token_file", unavailable_write)
    with pytest.raises(OSError, match="has not started"):
        harness.rotate()
    assert hashlib.sha256(harness.token_file.read_bytes()).hexdigest() == OLD_DIGEST
    assert module.STEP_TOKEN_FILE_WRITTEN in harness.state["started_steps"]
    assert module.STEP_TOKEN_FILE_WRITTEN not in harness.state["steps"]
    before = len(harness.calls)
    with pytest.raises(BootstrapError, match="write intent"):
        harness.rotate(rollback=True)
    assert len(harness.calls) == before

    monkeypatch.setattr(module, "write_token_file", write)
    assert harness.rotate()["status"] == module.STATUS_COMPLETED


def test_pending_cleanup_failure_keeps_the_committed_rotation_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    unlink = Path.unlink
    failures = []

    def fail_pending_once(path, *args, **kwargs):
        if path.name == module.PENDING_TOKEN_FILE and not failures:
            failures.append(path)
            raise OSError("pending cleanup interrupted")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_pending_once)
    with pytest.raises(OSError, match="pending cleanup"):
        harness.rotate()
    state = harness.state
    assert state["status"] == module.STATUS_COMPLETED
    assert module.STEP_RETIRING_DROPPED in state["steps"]
    before = len(harness.calls)
    result = harness.rotate()
    assert result["rotation_id"] == state["rotation_id"]
    assert result["status"] == module.STATUS_COMPLETED
    assert len(harness.calls) == before
    assert not failures[0].exists(), (
        "completed rotation replay must remove the pending token file"
    )


def test_resume_refuses_changed_site_identity_before_another_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.acceptance_error = BootstrapError("quiet proof unavailable")
    with pytest.raises(BootstrapError, match="quiet proof"):
        harness.rotate()
    before = len(harness.calls)
    harness.site.source_sha256 = "9" * 64
    harness.site.release_config["clusters"][0]["context"] = "gpu-a-other-context"

    with pytest.raises(BootstrapError, match="site changed.*clusters"):
        harness.rotate()
    assert len(harness.calls) == before


@pytest.mark.parametrize("membership", ["removed", "rebound"])
def test_rotation_reloads_membership_inside_the_lock_before_reading_the_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, membership: str
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    events = []
    current = SimpleNamespace(**vars(harness.site))
    current.release_config = {
        **harness.site.release_config,
        "clusters": (
            []
            if membership == "removed"
            else [
                {
                    **harness.site.release_config["clusters"][0],
                    "eks_cluster_arn": GPU_ARN.replace("gpu-a", "rebound-gpu"),
                }
            ]
        ),
    }

    @contextmanager
    def lock(path):
        assert path == tmp_path
        events.append("lock")
        try:
            yield
        finally:
            events.append("unlock")

    def reload(site):
        assert events == ["lock"]
        events.append("reload")
        return current

    monkeypatch.setattr(module, "administrator_operation_lock", lock)
    monkeypatch.setattr(module, "reload_site_for_mutation", reload)
    with pytest.raises(
        BootstrapError, match="unknown cluster_id|target cluster identity"
    ):
        harness.rotate()
    assert events == ["lock", "reload", "unlock"]
    assert harness.calls == []
    assert not module.rotation_state_path(harness.site, "gpu-a").exists(), (
        "rejected membership must not create rotation state"
    )


@pytest.mark.parametrize("updates", [{"status": "UNKNOWN"}, {"schema_version": 999}])
def test_unknown_rotation_state_is_not_archived_or_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, updates: dict
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.acceptance_error = BootstrapError("still waiting")
    with pytest.raises(BootstrapError):
        harness.rotate()
    path = module.rotation_state_path(harness.site, "gpu-a")
    state = {**harness.state, **updates}
    path.write_text(json.dumps(state), encoding="utf-8")
    before = len(harness.calls)

    with pytest.raises(BootstrapError, match="unknown schema or status"):
        harness.rotate()
    assert json.loads(path.read_text()) == state
    assert len(harness.calls) == before
    assert not (path.parent / "history").exists(), (
        "unknown rotation state must not be archived"
    )


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
        (
            {
                "phase": "complete",
                "transaction_committed": True,
                "commit_cleanup_completed": False,
            },
            "complete",
        ),
        ({"phase": "unknown-phase"}, "unknown-phase"),
        ({}, "unknown"),
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


@pytest.mark.parametrize("timeout_seconds", [0, 30, 60])
def test_quiet_period_must_leave_time_for_the_acceptance_probe(
    tmp_path: Path, timeout_seconds: int
) -> None:
    with pytest.raises(BootstrapError, match="quiet"):
        module.RotateTokenRequest(
            site=_site(tmp_path),
            cluster_id="gpu-a",
            quiet_seconds=60,
            acceptance_timeout_seconds=timeout_seconds,
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
    # The registry Secret was never rewritten before the rollback boundary, so
    # the running control-plane Pods already hold the old-only registry the
    # rollback republishes: no restart, and the journal says so.
    assert harness.secret[0]["token"] == OLD_TOKEN
    assert "restart_control_plane" not in [call[0] for call in harness.calls]
    assert module.STEP_CONTROL_PLANE_ROLLED not in harness.state.get("steps", {})
    assert module.STEP_CONTROL_PLANE_ROLLED not in harness.state.get(
        "started_steps", {}
    )


def test_rollback_restores_a_secret_whose_update_acknowledgement_was_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    ensure = module.ensure_connection_secret

    def update_then_interrupt(release, target):
        ensure(release, target)
        if Path(target.token_file).name == module.PENDING_TOKEN_FILE:
            raise ReleaseError("connection secret acknowledgement lost")

    monkeypatch.setattr(module, "ensure_connection_secret", update_then_interrupt)
    with pytest.raises(ReleaseError, match="acknowledgement lost"):
        harness.rotate()
    assert module.STEP_SECRET_UPDATED not in harness.state["steps"]
    harness.calls.clear()

    result = harness.rotate(rollback=True)

    assert result["status"] == module.STATUS_ROLLED_BACK
    assert [item[0] for item in harness.calls] == [
        "connection_secret",
        "restart_data_plane",
        "publish",
    ]
    assert hashlib.sha256(harness.calls[0][1].encode()).hexdigest() == OLD_DIGEST


def test_partial_rollback_resumes_the_original_node_set_and_refuses_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.acceptance_error = BootstrapError("still waiting")
    with pytest.raises(BootstrapError):
        harness.rotate()
    roll = module.roll_node_tokens
    attempts = []

    def interrupted_roll(release, target, **kwargs):
        attempts.append(
            (kwargs["only_nodes"], list(kwargs["progress"]["completed_nodes"]))
        )
        if len(attempts) == 1:
            kwargs["progress"]["completed_nodes"] = ["node-1"]
            kwargs["record"]()
            raise ReleaseError("rollback wave interrupted")
        return roll(release, target, **kwargs)

    monkeypatch.setattr(module, "roll_node_tokens", interrupted_roll)
    with pytest.raises(ReleaseError, match="rollback wave"):
        harness.rotate(rollback=True)
    before = len(harness.calls)
    with pytest.raises(BootstrapError, match="resume with --rollback"):
        harness.rotate()
    assert len(harness.calls) == before

    result = harness.rotate(rollback=True)

    assert result["status"] == module.STATUS_ROLLED_BACK
    assert attempts == [
        (frozenset({"node-1", "node-2"}), []),
        (frozenset({"node-1", "node-2"}), ["node-1"]),
    ]


def test_rollback_includes_a_wave_that_started_without_finishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    roll = module.roll_node_tokens
    rolled_back = []

    def interrupted_wave(release, target, **kwargs):
        if kwargs["only_nodes"] is None:
            kwargs["progress"]["started_nodes"] = ["node-1"]
            kwargs["record"]()
            raise ReleaseError("forward wave interrupted")
        rolled_back.append(kwargs["only_nodes"])
        return roll(release, target, **kwargs)

    monkeypatch.setattr(module, "roll_node_tokens", interrupted_wave)
    with pytest.raises(ReleaseError, match="forward wave"):
        harness.rotate()
    assert harness.state["node_rollout"]["completed_nodes"] == []

    result = harness.rotate(rollback=True)
    assert result["status"] == module.STATUS_ROLLED_BACK
    assert rolled_back == [frozenset({"node-1"})]


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
    assert progress["started_nodes"] == ["node-2", "node-3"]
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
    host = LogHost(namespace=site.release_config["namespace"])
    monkeypatch.setattr(acceptance_module, "run_command", host.run)
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


def test_acceptance_does_not_accept_a_quiet_result_after_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = {"now": 0.0}
    host = LogHost(namespace="gpu-fault-system")
    monkeypatch.setattr(acceptance_module, "run_command", host.run)

    def sleep(seconds):
        clock["now"] += seconds

    def late_logs(site, cluster_id, *, since_seconds):
        clock["now"] += 60
        return []

    monkeypatch.setattr(module, "retiring_token_authentications", late_logs)
    with pytest.raises(BootstrapError, match="timeout|timed out|deadline"):
        module.wait_for_new_token_acceptance(
            _site(tmp_path),
            "gpu-a",
            quiet_seconds=10,
            timeout_seconds=30,
            sleep=sleep,
            monotonic=lambda: clock["now"],
        )


def test_quiet_probe_failure_does_not_expose_credential_helper_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = "synthetic-unstructured-auth-value-13579"
    host = LogHost("cpu-api-pod", namespace="gpu-fault-system")

    def command(arguments, **kwargs):
        if "logs" in arguments:
            return subprocess.CompletedProcess(
                arguments, 1, "", f"Forbidden: exec helper failed\n{marker}"
            )
        return host.run(arguments, **kwargs)

    monkeypatch.setattr(module, "run_command", command)
    with pytest.raises(BootstrapError) as failure:
        module.retiring_token_authentications(
            _site(tmp_path), "gpu-a", since_seconds=60
        )
    message = str(failure.value)
    assert marker not in message
    assert "Forbidden" in message
    assert "redacted" in message


@pytest.mark.parametrize("invalid_source", ["empty", "not-ready", "restarted"])
def test_quiet_probe_refuses_missing_or_incomplete_log_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_source: str
) -> None:
    host = LogHost("cpu-api-pod", namespace="gpu-fault-system")
    pod = host.pods[0]
    if invalid_source == "not-ready":
        pod["status"]["conditions"][0]["status"] = "False"
    if invalid_source == "restarted":
        pod["status"]["containerStatuses"][0]["state"]["running"]["startedAt"] = (
            datetime.now(timezone.utc).isoformat()
        )
    if invalid_source == "empty":
        host.pods.clear()
    commands = []

    def command(arguments, **kwargs):
        commands.append(arguments)
        return host.run(arguments, **kwargs)

    monkeypatch.setattr(module, "run_command", command)
    with pytest.raises(BootstrapError, match="full quiet window"):
        module.retiring_token_authentications(
            _site(tmp_path), "gpu-a", since_seconds=60
        )
    assert len(commands) == 3
    assert all("logs" not in arguments for arguments in commands), (
        "incomplete sources must be rejected before reading logs"
    )


@pytest.mark.parametrize("replace_source", [False, True])
def test_quiet_probe_binds_the_ready_sources_before_and_after_log_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replace_source: bool
) -> None:
    host = LogHost("cpu-api-pod", namespace="gpu-fault-system")
    commands = []
    stamp = datetime.now(timezone.utc)
    old = (stamp - timedelta(minutes=5)).isoformat()
    prefix = f"{old} startup complete\n"
    window = (
        f"{stamp.isoformat()} regional cluster gpu-b "
        "authenticated with the retiring token\n"
    )

    def command(arguments, **kwargs):
        commands.append(arguments)
        if "logs" in arguments:
            assert kwargs["timeout_seconds"] == 180
            assert "--timestamps" in arguments
            assert "--container=api" in arguments
            content = (
                prefix
                if any(argument.startswith("--limit-bytes=") for argument in arguments)
                else window
            )
            return subprocess.CompletedProcess(
                arguments, 0, "[pod/cpu-api-pod/api] " + content, ""
            )
        assert kwargs["timeout_seconds"] == 120
        if replace_source and host.gets:
            host.after_pods[0]["metadata"]["uid"] = "replacement-uid"
        return host.run(arguments, **kwargs)

    monkeypatch.setattr(module, "run_command", command)
    if replace_source:
        with pytest.raises(BootstrapError, match="sources changed"):
            module.retiring_token_authentications(
                _site(tmp_path), "gpu-a", since_seconds=60
            )
    else:
        assert (
            module.retiring_token_authentications(
                _site(tmp_path), "gpu-a", since_seconds=60
            )
            == []
        )
    assert ["logs" in arguments for arguments in commands] == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]


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
