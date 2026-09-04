from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
WAIT = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_rollout_wait.py"
)
GPU_ROLLOUT = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_gpu_rollout.py"
)
FLEET_ROLLOUT = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_fleet_rollout.py"
)


def _deployment(*, ready: int = 0) -> dict:
    return {
        "metadata": {"generation": 2},
        "spec": {"replicas": 1, "selector": {"matchLabels": {"app": "executor"}}},
        "status": {
            "observedGeneration": 2,
            "updatedReplicas": ready,
            "readyReplicas": ready,
            "availableReplicas": ready,
            "unavailableReplicas": 0 if ready else 1,
        },
    }


def _pod_failure(reason: str, message: str = "") -> dict:
    pod = {
        "metadata": {"name": "executor-new"},
        "status": {
            "containerStatuses": [{"state": {"waiting": {"reason": reason}}}],
            "conditions": [],
        },
    }
    if reason == "Unschedulable":
        pod["status"]["containerStatuses"] = []
        pod["status"]["conditions"] = [
            {
                "type": "PodScheduled",
                "status": "False",
                "reason": "Unschedulable",
                "message": message,
            }
        ]
    return {"items": [pod]}


def _release(values) -> SimpleNamespace:
    responses = iter(values)
    return SimpleNamespace(
        runner=SimpleNamespace(dry_run=False),
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _gpu=lambda _target, *arguments: list(arguments),
        _get_json=lambda _arguments: next(responses),
    )


def test_rollout_wait_fails_immediately_on_crash_loop() -> None:
    release = _release([_deployment(), _pod_failure("CrashLoopBackOff")])

    with pytest.raises(WAIT.ReleaseError, match="CrashLoopBackOff"):
        WAIT.wait_deployment_rollout(
            release, SimpleNamespace(cluster_id="gpu-a"), "executor", poll_seconds=0
        )


def test_rollout_wait_allows_transient_capacity_pending_with_progress() -> None:
    release = _release(
        [
            _deployment(),
            _pod_failure(
                "Unschedulable", "0/4 nodes are available: 1 Insufficient cpu"
            ),
            _deployment(ready=1),
        ]
    )

    result = WAIT.wait_deployment_rollout(
        release, SimpleNamespace(cluster_id="gpu-a"), "executor", poll_seconds=0
    )

    assert result["progress"] == [2, 1, 1, 1]


def test_rollout_wait_rejects_deterministic_unschedulable_pod() -> None:
    release = _release(
        [
            _deployment(),
            _pod_failure("Unschedulable", "0/4 nodes match Pod node affinity"),
        ]
    )

    with pytest.raises(WAIT.ReleaseError, match="node affinity"):
        WAIT.wait_deployment_rollout(
            release, SimpleNamespace(cluster_id="gpu-a"), "executor", poll_seconds=0
        )


def test_gpu_deployments_apply_dependency_waves_and_wait_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sequence: list[str] = []
    active = 0
    maximum = 0
    lock = threading.Lock()
    two_started = threading.Event()
    manifests = [
        ("watcher", "watcher-yaml"),
        ("executor", "executor-yaml"),
        ("collector", "collector-yaml"),
    ]

    class Runner:
        dry_run = False

        def run(self, arguments, *, input_text=None, **_kwargs):
            if "apply" in arguments:
                prefix = "dry-run" if "--dry-run=server" in arguments else "apply"
                sequence.append(f"{prefix}:{input_text}")
            return ""

        def probe_output(self, _arguments, **_kwargs):
            # The Completion Watcher outbox ConfigMap does not exist yet.
            return 1, "", 'Error from server (NotFound): configmaps "x" not found'

    release = SimpleNamespace(
        runner=Runner(),
        executor_wheel_sha="a" * 64,
        config=SimpleNamespace(
            namespace="gpu-fault-system",
            executor_protocol_version=3,
            component_digests={"executor": "b" * 64},
        ),
        _config_map_data=lambda _name: {
            "required-regional-executor-protocol-version": "3",
            "required-regional-executor-artifact-sha256": "a" * 64,
            "required-regional-executor-compatibility-digest": "b" * 64,
        },
        _gpu=lambda _target, *arguments: list(arguments),
    )
    monkeypatch.setattr(
        GPU_ROLLOUT, "render_gpu_rollout_manifests", lambda *_args, **_kwargs: manifests
    )

    def wait(_release, _target, deployment):
        nonlocal active, maximum
        sequence.append(f"wait:{deployment}")
        if deployment == "executor":
            return {}
        with lock:
            active += 1
            maximum = max(maximum, active)
            if active == 2:
                two_started.set()
        assert two_started.wait(timeout=2), "secondary Deployment waits did not overlap"
        time.sleep(0.01)
        with lock:
            active -= 1
        return {}

    monkeypatch.setattr(GPU_ROLLOUT, "wait_deployment_rollout", wait)
    monkeypatch.setattr(GPU_ROLLOUT.inventory, "GPU_EXECUTOR_DEPLOYMENT", "executor")
    monkeypatch.setattr(GPU_ROLLOUT.inventory, "GPU_WATCHER_DEPLOYMENT", "watcher")
    monkeypatch.setattr(GPU_ROLLOUT.inventory, "GPU_COLLECTOR_DEPLOYMENT", "collector")

    GPU_ROLLOUT.apply_gpu_deployments(
        release, SimpleNamespace(cluster_id="gpu-a"), "wheel"
    )

    assert set(sequence[:3]) == {
        "dry-run:watcher-yaml",
        "dry-run:executor-yaml",
        "dry-run:collector-yaml",
    }
    assert sequence[3:5] == ["apply:executor-yaml", "wait:executor"]
    assert set(sequence[5:7]) == {"apply:watcher-yaml", "apply:collector-yaml"}
    assert maximum == 2


def test_reconciler_deploy_uses_progress_aware_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deploy script must not block; the progress-aware wait must.

    ``kubectl rollout status`` inside the script gives up on a fixed deadline
    and cannot tell a slow image pull from a crash loop, so the deploy hands
    the waiting back to ``wait_deployment_rollout``.
    """

    sequence: list[str] = []
    environments: list[dict[str, str]] = []
    waited: list[tuple[str, int]] = []

    class Runner:
        dry_run = True

        def run(self, arguments, *, env=None, **_kwargs):
            sequence.append(f"run:{Path(arguments[0]).name}")
            environments.append(dict(env or {}))
            return ""

    release = SimpleNamespace(
        runner=Runner(),
        bundle_sha="b" * 64,
        node_template_sha="t" * 64,
        _cancel_active_installer_jobs=lambda _target: sequence.append("cancel"),
        _retry_failed_installer_jobs=lambda _target: sequence.append("retry"),
    )
    monkeypatch.setattr(
        FLEET_ROLLOUT, "build_reconciler_environment", lambda *_a, **_k: {}
    )

    def wait(_release, _target, deployment, *, timeout_seconds):
        sequence.append("wait")
        waited.append((deployment, timeout_seconds))
        return {}

    monkeypatch.setattr(FLEET_ROLLOUT, "wait_deployment_rollout", wait)

    FLEET_ROLLOUT.deploy_reconciler(
        release,
        SimpleNamespace(cluster_id="gpu-a", fleet_master_file=None),
        wheel_cm="wheel",
        bundle_cm="bundle",
        artifact_sha="a" * 64,
        config_digest="c" * 64,
    )

    assert sequence == [
        "cancel",
        "retry",
        "run:deploy-node-installer-reconciler.sh",
        "wait",
    ]
    assert environments == [{"GPU_FAULT_WAIT_FOR_RECONCILER_ROLLOUT": "false"}]
    assert waited == [(FLEET_ROLLOUT.inventory.GPU_RECONCILER_DEPLOYMENT, 600)]


def test_wave_safety_waits_for_agent_lease_margin(monkeypatch) -> None:
    snapshots = iter(
        (
            {
                "open_remote": {},
                "destructive_workflow_count": 0,
                "agent_blocker_count": 1,
                "agent_blockers": [
                    {
                        "node_id": "node-b",
                        "reason": "lease-margin",
                        "lease_remaining_seconds": 12.0,
                    }
                ],
            },
            {
                "open_remote": {},
                "destructive_workflow_count": 0,
                "agent_blocker_count": 0,
                "agent_blockers": [],
            },
        )
    )
    calls = []
    monkeypatch.setattr(
        FLEET_ROLLOUT,
        "rollout_wave_safety_snapshot",
        lambda *_args, **_kwargs: (calls.append(True) or next(snapshots)),
    )

    FLEET_ROLLOUT.ensure_rollout_wave_safe(
        SimpleNamespace(),
        SimpleNamespace(cluster_id="gpu-a"),
        wave=("node-a",),
        node_names=("node-a", "node-b"),
        timeout_seconds=1,
        poll_seconds=0,
    )

    assert len(calls) == 2


def test_wave_safety_fails_immediately_for_remote_commands(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        FLEET_ROLLOUT,
        "rollout_wave_safety_snapshot",
        lambda *_args, **_kwargs: (
            calls.append(True)
            or {
                "open_remote": {"LEASED": 1},
                "destructive_workflow_count": 0,
                "agent_blocker_count": 0,
                "agent_blockers": [],
            }
        ),
    )

    with pytest.raises(FLEET_ROLLOUT.ReleaseError, match="remote commands"):
        FLEET_ROLLOUT.ensure_rollout_wave_safe(
            SimpleNamespace(),
            SimpleNamespace(cluster_id="gpu-a"),
            wave=("node-a",),
            node_names=("node-a", "node-b"),
            timeout_seconds=60,
            poll_seconds=0,
        )

    assert len(calls) == 1


def test_candidate_cpu_heartbeat_barrier_is_one_aggregated_exec() -> None:
    class Runner:
        dry_run = False

        def __init__(self) -> None:
            self.calls = []

        def run(self, args, **kwargs):
            self.calls.append((args, kwargs))
            if "get" in args and "pod" in args:
                return "cpu-pod"
            return '{"status":"PASSED","expected_count":2,"refreshed_count":2}'

    runner = Runner()
    release = SimpleNamespace(
        runner=runner,
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _cpu=lambda *args: list(args),
    )

    FLEET_ROLLOUT.wait_candidate_cpu_agent_heartbeats(
        release,
        {"gpu-a": {"node_ids": ["node-a"]}, "gpu-b": {"node_ids": ["node-b"]}},
        timeout_seconds=75,
        poll_seconds=5,
    )

    assert len(runner.calls) == 2
    payload = json.loads(runner.calls[1][1]["input_text"])
    assert payload["expected_by_cluster"] == {"gpu-a": ["node-a"], "gpu-b": ["node-b"]}
    assert runner.calls[1][1]["timeout_seconds"] == 105


class _BarrierRunner:
    dry_run = False

    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def run(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if "get" in args and "pod" in args:
            return "cpu-pod"
        return self.result


def _barrier_release(runner: object) -> SimpleNamespace:
    return SimpleNamespace(
        runner=runner,
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _cpu=lambda *args: list(args),
    )


def test_candidate_cpu_barrier_forwards_the_required_pin_identity() -> None:
    runner = _BarrierRunner(
        '{"status":"PASSED","expected_count":1,"refreshed_count":1}'
    )
    identity = {
        "artifact_sha256": "a" * 64,
        "compatibility_digest": "b" * 64,
        "agent_protocol_version": "3",
        "config_digest": "c" * 64,
    }

    FLEET_ROLLOUT.wait_candidate_cpu_agent_heartbeats(
        _barrier_release(runner),
        {"gpu-a": {"node_ids": ["node-a"]}},
        required_identity=identity,
        timeout_seconds=75,
        poll_seconds=5,
    )

    payload = json.loads(runner.calls[1][1]["input_text"])
    assert payload["required_identity"] == identity


def test_pin_barrier_failure_says_the_window_stays_open() -> None:
    """The two barrier callers fail for different reasons; say which one it is.

    A liveness failure after the cutover and a fleet that never reached the
    candidate pin are different incidents with different next steps, and the
    second one is the one that leaves the compatibility window open.
    """
    runner = _BarrierRunner(
        json.dumps(
            {
                "status": "FAILED",
                "expected_count": 1,
                "refreshed_count": 0,
                "blocker_count": 1,
                "blockers": [{"node_id": "node-a", "reason": "pin-not-aligned"}],
            }
        )
    )

    with pytest.raises(FLEET_ROLLOUT.ReleaseError, match="window stays open"):
        FLEET_ROLLOUT.wait_candidate_cpu_agent_heartbeats(
            _barrier_release(runner),
            {"gpu-a": {"node_ids": ["node-a"]}},
            required_identity={
                "artifact_sha256": "a" * 64,
                "compatibility_digest": "b" * 64,
                "agent_protocol_version": "3",
                "config_digest": "c" * 64,
            },
            timeout_seconds=1,
            poll_seconds=0,
        )


def test_candidate_pin_identity_matches_the_finalized_metadata_keys() -> None:
    release = SimpleNamespace(
        node_wheel_sha="a" * 64,
        config=SimpleNamespace(
            component_digests={"node_runtime": "b" * 64},
            agent_protocol_version=3,
            agent_config_digest="c" * 64,
        ),
    )

    assert FLEET_ROLLOUT.candidate_agent_pin_identity(release) == {
        "artifact_sha256": "a" * 64,
        "compatibility_digest": "b" * 64,
        "agent_protocol_version": "3",
        "config_digest": "c" * 64,
    }


def test_candidate_pin_identity_falls_back_to_the_artifact_digest() -> None:
    release = SimpleNamespace(
        node_wheel_sha="a" * 64,
        config=SimpleNamespace(
            component_digests={}, agent_protocol_version=3, agent_config_digest="c" * 64
        ),
    )

    identity = FLEET_ROLLOUT.candidate_agent_pin_identity(release)

    assert identity["compatibility_digest"] == "a" * 64


def test_active_agent_node_sets_are_captured_before_cpu_rollout() -> None:
    class Runner:
        dry_run = False

        @staticmethod
        def run(args, **_kwargs):
            if "get" in args and "pod" in args:
                return "cpu-pod"
            return '{"gpu-a":["node-b","node-a"],"gpu-b":["node-c"]}'

    release = SimpleNamespace(
        runner=Runner(),
        config=SimpleNamespace(
            namespace="gpu-fault-system",
            clusters=(
                SimpleNamespace(cluster_id="gpu-a"),
                SimpleNamespace(cluster_id="gpu-b"),
            ),
        ),
        _cpu=lambda *args: list(args),
    )

    result = FLEET_ROLLOUT.capture_active_agent_node_sets(release)

    assert result == {
        "gpu-a": {"node_ids": ["node-a", "node-b"]},
        "gpu-b": {"node_ids": ["node-c"]},
    }
