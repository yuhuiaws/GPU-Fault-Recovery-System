from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_release_fleet_rollout as FLEET_MODULE
from gpu_fault_release import regional_release_node_preflight as NODE_PREFLIGHT_MODULE
from gpu_fault_release import regional_release_node_runtime_rollout as RUNTIME_MODULE
from gpu_fault_release import regional_release_validation as VALIDATION_MODULE
from gpu_fault_release import rollout as MODULE
from tests.regional._release_orchestrator_support import (
    config_file,
    ingress_pod_list_json,
)

ROOT = Path(__file__).resolve().parents[2]


def test_node_runtime_rollout_uses_fleet_waves(tmp_path: Path, monkeypatch) -> None:
    runtime_module = RUNTIME_MODULE
    config = replace(
        MODULE.ReleaseConfig.load(config_file(tmp_path)), upgrade_max_unavailable=2
    )
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    target = config.clusters[0]
    deploy_calls = []
    wait_calls = []
    fleet_calls = []
    safety_calls = []
    preflight_calls = []
    deployment_reads = iter(({"status": "IN_PROGRESS"}, {"status": "SUCCEEDED"}))
    waves = iter(({"node_ids": ["node-a", "node-b"]}, {"node_ids": ["node-c"]}))
    expected_waves = iter((("node-a", "node-b"), ("node-c",)))
    monkeypatch.setattr(
        release, "_target_node_names", lambda _target: ("node-a", "node-b", "node-c")
    )
    monkeypatch.setattr(
        runtime_module,
        "target_node_failure_domains",
        lambda *_args: {"node-a": "zone-a", "node-b": "zone-b", "node-c": "zone-c"},
    )
    monkeypatch.setattr(
        runtime_module,
        "ensure_rollout_wave_safe",
        lambda *_a, **kwargs: safety_calls.append(dict(kwargs)),
    )
    monkeypatch.setattr(
        FLEET_MODULE, "next_deployment_wave", lambda _deployment: next(expected_waves)
    )
    monkeypatch.setattr(
        FLEET_MODULE,
        "ensure_rollout_wave_safe",
        lambda *_a, **kwargs: safety_calls.append(dict(kwargs)),
    )
    monkeypatch.setattr(
        runtime_module,
        "ensure_pre_node_mutation_barrier",
        lambda *_args, **_kwargs: preflight_calls.append("barrier"),
    )
    monkeypatch.setattr(
        release,
        "_deploy_reconciler",
        lambda *_args, **kwargs: (
            deploy_calls.append(dict(kwargs)) or ("b" * 64, "e" * 64)
        ),
    )
    monkeypatch.setattr(
        release,
        "_wait_agents",
        lambda *_args, **kwargs: wait_calls.append(kwargs.get("node_names")),
    )
    wave_patches = []
    installer_job_calls = []
    monkeypatch.setattr(
        FLEET_MODULE,
        "wait_deployment_rollout",
        lambda *_args, **_kwargs: {"deployment": "reconciler"},
    )
    monkeypatch.setattr(
        FLEET_MODULE,
        "reconciler_container_env",
        lambda *_args, **_kwargs: {
            "GPU_FAULT_INSTALLER_WAVE_CONFIG_MAP": "gpu-fault-node-installer-wave",
            "GPU_FAULT_INSTALLER_BUNDLE_SHA256": "b" * 64,
            "GPU_FAULT_INSTALLER_TEMPLATE_SHA256": "e" * 64,
        },
    )
    monkeypatch.setattr(
        release,
        "_settle_installer_jobs",
        lambda _target: installer_job_calls.append("settle"),
    )

    def record_patch(args, **_kwargs):
        arguments = list(args)
        wave_patches.append(json.loads(arguments[arguments.index("-p") + 1]))
        return ""

    monkeypatch.setattr(release.runner, "run", record_patch)
    monkeypatch.setattr(release, "_get_json", lambda _args: wave_patches[-1])

    def fleet_command(operation, payload):
        fleet_calls.append((operation, payload))
        if operation == "normalize-records":
            return {"normalized": 0}
        if operation == "terminalize-cluster-rollouts":
            return {"terminalized": []}
        if operation == "create":
            return {"status": "PLANNED"}
        if operation == "next-wave":
            return next(waves)
        return next(deployment_reads)

    monkeypatch.setattr(release, "_fleet_command", fleet_command)

    identity = runtime_module.roll_node_runtime(
        release,
        target,
        phase="upgrade",
        wheel_cm=release.executor_wheel_cm,
        bundle_cm=release.bundle_cm,
        artifact_sha=release.node_wheel_sha,
        config_digest=release.config.agent_config_digest,
        candidate_preflight_completed=True,
    )

    assert identity == ("b" * 64, "e" * 64)
    assert preflight_calls == ["barrier"]
    # Waves are handed to the running Reconciler through its wave ConfigMap, so
    # the Deployment is only rolled twice: paused, then steady state.
    assert [item.get("allowed_node_names") for item in deploy_calls] == [(), None]
    assert [item.get("sync_registry", True) for item in deploy_calls] == [False, True]
    assert {item.get("max_unavailable") for item in deploy_calls} == {2}
    assert [item["data"]["allowed-nodes"] for item in wave_patches] == [
        "node-a,node-b",
        "node-c",
    ]
    assert {item["data"]["max-unavailable"] for item in wave_patches} == {"2"}
    assert len({item["data"]["generation"] for item in wave_patches}) == 2
    # One settling pass per wave, and one listing inside it: the in-flight and
    # the failed Jobs are decided from the same read.
    assert installer_job_calls == ["settle", "settle"]
    assert wait_calls == [("node-a", "node-b"), ("node-c",), None]
    # The double answers `None` for the lease minimum, so every wave takes the
    # fail-closed fallback and re-probes after the lease; the single-probe path
    # is pinned in tests/regional/test_release_fleet_wave_cost.py.
    assert [item.get("minimum_lease_remaining_seconds") for item in safety_calls] == [
        None,
        30,
        None,
        30,
    ]
    assert fleet_calls[0] == ("normalize-records", {})
    request = next(
        payload["request"]
        for operation, payload in fleet_calls
        if operation == "create"
    )
    assert request["max_unavailable"] == 2
    assert request["first_wave_max_unavailable"] == 1
    assert request["max_unavailable_per_failure_domain"] == 1
    assert request["node_failure_domains"] == {
        "node-a": "zone-a",
        "node-b": "zone-b",
        "node-c": "zone-c",
    }
    assert request["desired_bundle_sha256"] == "b" * 64
    assert request["desired_template_sha256"] == "e" * 64


def _wave_context(**overrides) -> object:
    fields = {
        "phase": "upgrade",
        "deployment_id": "deployment",
        "node_names": ("node-a", "node-b"),
        "paused_identity": ("b" * 64, "e" * 64),
        "wheel_cm": "wheel",
        "bundle_cm": "bundle",
        "artifact_sha": "a" * 64,
        "config_digest": "config",
        "expected_profile": "profile-v1",
        "executor_wheel_filename": None,
        "expected_compatibility": "compatibility",
        "desired_bundle": "b" * 64,
        "desired_template": "e" * 64,
        "template_config_map": None,
        "max_unavailable": 2,
        "runtime_image": None,
        "node_installer_image": None,
        "allow_legacy_identity": False,
        "agent_identity": None,
    }
    fields.update(overrides)
    return FLEET_MODULE.FleetWaveContext(**fields)


def _wave_release(monkeypatch, environment: dict[str, str], observed: object):
    calls: list[list[str]] = []
    release = SimpleNamespace(
        runner=SimpleNamespace(
            dry_run=False, run=lambda args, **_kwargs: calls.append(list(args)) or ""
        ),
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _gpu=lambda _target, *args: list(args),
        _get_json=lambda _args: observed,
        _settle_installer_jobs=lambda _target: calls.append(["settle"]),
    )
    monkeypatch.setattr(
        FLEET_MODULE,
        "wait_deployment_rollout",
        lambda *_args, **_kwargs: {"deployment": "reconciler"},
    )
    monkeypatch.setattr(
        FLEET_MODULE, "reconciler_container_env", lambda *_args, **_kwargs: environment
    )
    return release, calls


def test_wave_handoff_falls_back_when_reconciler_has_no_wave_config_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, calls = _wave_release(
        monkeypatch,
        {
            "GPU_FAULT_INSTALLER_BUNDLE_SHA256": "b" * 64,
            "GPU_FAULT_INSTALLER_TEMPLATE_SHA256": "e" * 64,
        },
        {},
    )

    identity = FLEET_MODULE.hand_wave_to_reconciler(
        release, SimpleNamespace(cluster_id="gpu-a"), _wave_context(), ("node-a",)
    )

    assert identity is None, "legacy Reconciler must fall back to a redeploy"
    assert calls == [], "fallback must not touch installer Jobs or any ConfigMap"


def test_wave_handoff_rejects_installer_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, calls = _wave_release(
        monkeypatch,
        {
            "GPU_FAULT_INSTALLER_WAVE_CONFIG_MAP": "gpu-fault-node-installer-wave",
            "GPU_FAULT_INSTALLER_BUNDLE_SHA256": "c" * 64,
            "GPU_FAULT_INSTALLER_TEMPLATE_SHA256": "e" * 64,
        },
        {},
    )

    with pytest.raises(FLEET_MODULE.ReleaseError, match="identity changed"):
        FLEET_MODULE.hand_wave_to_reconciler(
            release, SimpleNamespace(cluster_id="gpu-a"), _wave_context(), ("node-a",)
        )

    assert calls == [], "identity drift must be rejected before any node is allowed"


def test_wave_handoff_requires_the_patched_wave_to_be_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, calls = _wave_release(
        monkeypatch,
        {
            "GPU_FAULT_INSTALLER_WAVE_CONFIG_MAP": "gpu-fault-node-installer-wave",
            "GPU_FAULT_INSTALLER_BUNDLE_SHA256": "b" * 64,
            "GPU_FAULT_INSTALLER_TEMPLATE_SHA256": "e" * 64,
        },
        {"data": {"allowed-nodes": "node-z", "max-unavailable": "2"}},
    )

    with pytest.raises(FLEET_MODULE.ReleaseError, match="did not accept the wave"):
        FLEET_MODULE.hand_wave_to_reconciler(
            release, SimpleNamespace(cluster_id="gpu-a"), _wave_context(), ("node-a",)
        )

    assert ["settle"] in calls, (
        "the wave was patched without settling the previous wave's installer Jobs"
    )
    patch = next(item for item in calls if "patch" in item)
    assert patch[:5] == [
        "-n",
        "gpu-fault-system",
        "patch",
        "configmap",
        "gpu-fault-node-installer-wave",
    ]
    assert json.loads(patch[patch.index("-p") + 1])["data"]["allowed-nodes"] == "node-a"


def test_pre_node_barrier_failure_starts_no_reconciler_or_fleet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_module = RUNTIME_MODULE
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    target = config.clusters[0]
    monkeypatch.setattr(release, "_target_node_names", lambda _target: ("node-a",))
    monkeypatch.setattr(
        runtime_module,
        "target_node_failure_domains",
        lambda *_args: {"node-a": "zone-a"},
    )
    monkeypatch.setattr(
        runtime_module,
        "ensure_pre_node_mutation_barrier",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runtime_module.ReleaseError("preflight rejected")
        ),
    )
    monkeypatch.setattr(
        release,
        "_deploy_reconciler",
        lambda *_args, **_kwargs: pytest.fail(
            "reconciler started after pre-node barrier failure"
        ),
    )
    monkeypatch.setattr(
        release,
        "_fleet_command",
        lambda *_args, **_kwargs: pytest.fail(
            "Fleet mutation started after pre-node barrier failure"
        ),
    )

    with pytest.raises(runtime_module.ReleaseError, match="preflight rejected"):
        runtime_module.roll_node_runtime(
            release,
            target,
            phase="upgrade",
            wheel_cm=release.executor_wheel_cm,
            bundle_cm=release.bundle_cm,
            artifact_sha=release.node_wheel_sha,
            config_digest=config.agent_config_digest,
            candidate_preflight_completed=True,
        )


def test_pre_node_inventory_rejects_cordoned_or_quarantined_nodes() -> None:
    release = SimpleNamespace(
        _gpu=lambda _target, *args: list(args),
        _get_json=lambda _args: {
            "items": [
                {
                    "metadata": {
                        "name": "node-a",
                        "uid": "uid-a",
                        "annotations": {"gpu-fault.io/installer-state": "Installing"},
                    },
                    "spec": {
                        "unschedulable": True,
                        "taints": [
                            {"key": "gpu-fault.io/quarantined", "effect": "NoSchedule"}
                        ],
                    },
                    "status": {"conditions": [{"type": "Ready", "status": "False"}]},
                }
            ]
        },
    )

    with pytest.raises(NODE_PREFLIGHT_MODULE.ReleaseError) as caught:
        NODE_PREFLIGHT_MODULE.validate_target_node_state(
            release,
            SimpleNamespace(cluster_id="gpu-a", hyperpod_cluster_name="hp-gpu-a"),
            ("node-a",),
        )

    message = str(caught.value)
    assert "not-ready" in message
    assert "cordoned" in message
    assert "blocking-taint" in message
    assert "installer-active" in message


def _spare_node_items(
    annotations: dict[str, str], *, extra_taints: list[dict[str, str]] | None = None
) -> SimpleNamespace:
    taints = [{"key": "node.kubernetes.io/unschedulable", "effect": "NoSchedule"}]
    taints.extend(extra_taints or [])
    return SimpleNamespace(
        _gpu=lambda _target, *args: list(args),
        _get_json=lambda _args: {
            "items": [
                {
                    "metadata": {
                        "name": "spare-a",
                        "uid": "uid-spare-a",
                        "annotations": {
                            "gpu-fault.io/installer-state": "Succeeded",
                            **annotations,
                        },
                    },
                    "spec": {"unschedulable": True, "taints": taints},
                    "status": {"conditions": [{"type": "Ready", "status": "True"}]},
                }
            ]
        },
    )


def test_a_parked_warm_spare_does_not_block_the_node_wave() -> None:
    """A declared warm spare is cordoned by design, and must still be upgraded.

    `HyperPodSpareCoordinator` refuses an unreserved spare whose node is
    schedulable, so every pool member is permanently cordoned. Reading that
    cordon as "a node is being drained or repaired" made the pre-node-mutation
    barrier fail for the whole cluster, roll the release back, and leave the
    site unable to take any fix while it kept spares.
    """

    NODE_PREFLIGHT_MODULE.validate_target_node_state(
        _spare_node_items({"gpu-fault.io/spare-pool-state": "AVAILABLE"}),
        SimpleNamespace(cluster_id="gpu-a", hyperpod_cluster_name="hp-gpu-a"),
        ("spare-a",),
    )


@pytest.mark.parametrize(
    "annotations",
    [
        {},
        {"gpu-fault.io/spare-pool-state": "ALLOCATED"},
        {
            "gpu-fault.io/spare-pool-state": "AVAILABLE",
            "gpu-fault.io/spare-reservation": "incident-a",
        },
    ],
)
def test_only_an_unreserved_available_spare_is_forgiven_its_cordon(
    annotations: dict[str, str],
) -> None:
    # An ordinary cordoned node, a spare mid-allocation and a reserved spare all
    # mean somebody else is acting on the machine, so the cordon still blocks.
    with pytest.raises(NODE_PREFLIGHT_MODULE.ReleaseError) as caught:
        NODE_PREFLIGHT_MODULE.validate_target_node_state(
            _spare_node_items(annotations),
            SimpleNamespace(cluster_id="gpu-a", hyperpod_cluster_name="hp-gpu-a"),
            ("spare-a",),
        )

    assert "cordoned" in str(caught.value), str(caught.value)


def test_a_parked_spare_still_blocks_on_quarantine() -> None:
    # Forgiving the cordon must not forgive the taint that says "under repair".
    release = _spare_node_items(
        {"gpu-fault.io/spare-pool-state": "AVAILABLE"},
        extra_taints=[{"key": "gpu-fault.io/quarantined", "effect": "NoSchedule"}],
    )

    with pytest.raises(NODE_PREFLIGHT_MODULE.ReleaseError) as caught:
        NODE_PREFLIGHT_MODULE.validate_target_node_state(
            release,
            SimpleNamespace(cluster_id="gpu-a", hyperpod_cluster_name="hp-gpu-a"),
            ("spare-a",),
        )

    message = str(caught.value)
    assert "gpu-fault.io/quarantined" in message, message
    assert "cordoned" not in message, message


def test_rollback_parallelism_is_separate_and_small_clusters_stay_serial(
    tmp_path: Path,
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))

    assert (
        FLEET_MODULE.node_rollout_policy(release, 10, phase="upgrade").max_unavailable
        == 1
    )
    single_az = {f"node-{index}": "zone-a" for index in range(10)}
    single_policy = FLEET_MODULE.rollback_node_rollout_policy(release, single_az)
    assert single_policy.max_unavailable == 2
    assert single_policy.max_unavailable_per_failure_domain == 2
    multi_az = {f"node-{index}": f"zone-{index % 2}" for index in range(10)}
    multi_policy = FLEET_MODULE.rollback_node_rollout_policy(release, multi_az)
    assert multi_policy.max_unavailable == 2
    assert multi_policy.max_unavailable_per_failure_domain == 1
    assert (
        FLEET_MODULE.rollback_node_rollout_policy(
            release, {f"node-{index}": "zone-a" for index in range(5)}
        ).max_unavailable
        == 1
    )
    assert (
        FLEET_MODULE.rollback_node_rollout_policy(
            release,
            {**{f"node-{index}": "zone-a" for index in range(9)}, "node-9": "UNKNOWN"},
        ).max_unavailable
        == 1
    )
    large_release = MODULE.RegionalRelease(
        replace(config, rollback_max_unavailable=4), MODULE.Runner(dry_run=True)
    )
    assert (
        FLEET_MODULE.rollback_node_rollout_policy(
            large_release, {f"node-{index}": "zone-a" for index in range(32)}
        ).max_unavailable
        == 4
    )


def test_large_upgrade_uses_canary_then_failure_domain_balanced_waves(
    tmp_path: Path,
) -> None:
    config = replace(
        MODULE.ReleaseConfig.load(config_file(tmp_path)), upgrade_max_unavailable=32
    )
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))
    domains = {f"node-{index:04d}": f"zone-{index % 3}" for index in range(1000)}

    policy = FLEET_MODULE.node_rollout_policy(release, domains, phase="upgrade")

    assert policy.max_unavailable == 32
    assert policy.first_wave_max_unavailable == 1
    assert policy.max_unavailable_per_failure_domain == 11


def test_component_validation_writes_cpu_gpu_and_runtime_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    evidence = tmp_path / "quick-validation.json"
    calls: list[tuple[list[str], dict[str, str]]] = []
    runtime_calls: list[object] = []
    monkeypatch.setenv(VALIDATION_MODULE.QUICK_VALIDATION_EVIDENCE_ENV, str(evidence))
    monkeypatch.setattr(
        release.runner,
        "run",
        lambda command, **kwargs: (
            calls.append((list(command), dict(kwargs.get("env") or {}))) or ""
        ),
    )

    VALIDATION_MODULE.validate_release_components(
        release,
        cpu=True,
        data_plane=True,
        runtime_validator=lambda value: runtime_calls.append(value),
    )

    value = json.loads(evidence.read_text(encoding="utf-8"))
    assert value["checks"] == sorted(
        {
            "control_plane_role_split",
            "runtime_component_identity",
            *(f"data_plane_executor:{target.cluster_id}" for target in config.clusters),
        }
    )
    assert calls[0][1]["GPU_FAULT_RUNTIME_IMAGE"] == release.runtime_image
    assert runtime_calls == [release]


def test_secret_backup_returns_only_reference(tmp_path: Path, monkeypatch) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    calls = []
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
    )
    monkeypatch.setattr(
        release,
        "_get_json",
        lambda _args: {"type": "Opaque", "data": {"sensitive-key": "encoded-value"}},
    )
    monkeypatch.setattr(
        release.runner, "run", lambda args, **kwargs: calls.append((args, kwargs)) or ""
    )

    reference = FLEET_MODULE.backup_secret(
        release,
        ["kubectl"],
        source="gpu-fault-email",
        backup="gpu-fault-email-rollback-release-a",
        required=True,
    )

    assert reference == "gpu-fault-email-rollback-release-a"
    assert calls[0][1]["sensitive"] is True
    assert "encoded-value" in calls[0][1]["input_text"]


def test_profile_finalize_rejects_old_profile_activity(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(
        config_file(tmp_path, profile_version="profile-v2")
    )
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    responses = iter(
        (
            ingress_pod_list_json("cpu-pod"),
            json.dumps({"workflow_count": 1, "workload_count": 0}),
        )
    )
    monkeypatch.setattr(
        release.runner, "run", lambda *_args, **_kwargs: next(responses)
    )

    with pytest.raises(MODULE.ReleaseError, match="old-profile activity"):
        VALIDATION_MODULE.ensure_profile_transition_safe(release, "profile-v1")


def test_rollback_rejects_non_transactional_cluster_change(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))
    release.state = {"release_diff": {"kind": "FULL", "changed": ["clusters"]}}

    with pytest.raises(MODULE.ReleaseError, match="not transactional"):
        release.rollback(state={"metadata": {}, "cpu_wheel": "old-wheel"})


def test_rollback_accepts_endpoint_change_once_compensated(
    tmp_path: Path, monkeypatch
) -> None:
    """The endpoint left ``NON_TRANSACTIONAL_CHANGES`` when it gained a snapshot.

    ``regional_endpoint_rollback`` captures the live NLB Service and the Route53
    record the candidate overwrites, so an endpoint change no longer has to
    withdraw automatic rollback. Reaching the previous-pin check instead of the
    non-transactional refusal is what proves the guard let it through.
    """
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))
    release.state = {
        "release_diff": {"kind": "DATA_PLANE_COMPATIBLE", "changed": ["endpoint"]}
    }
    # The credential refresh preflight probes the live CronJob; this test is
    # about the transactional guard, not the site.
    monkeypatch.setattr(release, "_refresh_aurora_credentials", lambda: None)
    monkeypatch.setattr(
        release, "_require_no_inflight_installs", lambda **_kwargs: None
    )

    with pytest.raises(MODULE.ReleaseError, match="previous Agent identities"):
        release.rollback(state={"metadata": {}, "cpu_wheel": "old-wheel"})


def test_profile_finalize_allows_drained_old_profile(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(
        config_file(tmp_path, profile_version="profile-v2")
    )
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    responses = iter(
        (
            ingress_pod_list_json("cpu-pod"),
            json.dumps({"workflow_count": 0, "workload_count": 0}),
        )
    )
    monkeypatch.setattr(
        release.runner, "run", lambda *_args, **_kwargs: next(responses)
    )

    VALIDATION_MODULE.ensure_profile_transition_safe(release, "profile-v1")


def test_stability_window_accepts_steady_samples(tmp_path: Path, monkeypatch) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    snapshot = {
        "restarts": {"cpu/pod/container": 0},
        "not_ready": [],
        "queue": {"depth": 0, "oldest_age_seconds": 0.0},
        "remote_commands": {"by_status": {}},
        "critical_alerts": {"count": 0, "alerts": []},
    }
    snapshots = iter((snapshot, snapshot))
    times = iter((0.0, 0.0, 0.0, 120.0))
    monkeypatch.setattr(release, "_stability_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(MODULE.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)

    report = release.validate_stability_window(window_seconds=120, sample_seconds=120)

    assert report["healthy"] is True
    assert report["sample_count"] == 2
    assert report["critical_clear"]["wait_seconds"] == 0


def test_stability_waits_for_labeled_store_io_alert_to_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    steady = {
        "restarts": {"cpu/pod/container": 0},
        "not_ready": [],
        "queue": {"depth": 0, "oldest_age_seconds": 0.0},
        "remote_commands": {"by_status": {}},
        "critical_alerts": {"count": 0, "alerts": []},
    }
    firing = {
        **steady,
        "critical_alerts": {
            "count": 1,
            "alerts": [
                {"alertname": "GpuFaultStoreIoRejected", "severity": "critical"}
            ],
        },
    }
    snapshots = iter((firing, steady, steady))
    times = iter((0.0, 0.0, 0.0, 120.0))
    monkeypatch.setattr(release, "_stability_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(
        release,
        "_store_io_rejection_series_ready",
        lambda: {"ready": True, "pods": {"api/pod": {}}, "errors": []},
    )
    monkeypatch.setattr(MODULE.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)

    report = release.validate_stability_window(
        window_seconds=120,
        sample_seconds=120,
        critical_clear_timeout_seconds=30,
        critical_clear_sample_seconds=30,
    )

    assert report["healthy"] is True
    assert report["critical_clear"]["initial_alerts"] == ["GpuFaultStoreIoRejected"]
    assert report["critical_clear"]["wait_seconds"] == 30


def test_stability_rejects_unlabeled_store_io_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    baseline = {
        "restarts": {"cpu/pod/container": 0},
        "not_ready": [],
        "queue": {"depth": 0, "oldest_age_seconds": 0.0},
        "remote_commands": {"by_status": {}},
        "critical_alerts": {
            "count": 1,
            "alerts": [
                {"alertname": "GpuFaultStoreIoRejected", "severity": "critical"}
            ],
        },
    }
    monkeypatch.setattr(release, "_stability_snapshot", lambda: baseline)
    monkeypatch.setattr(
        release,
        "_store_io_rejection_series_ready",
        lambda: {
            "ready": False,
            "pods": {},
            "errors": ["gpu-fault-api-ha/pod has unlabeled Store I/O series"],
        },
    )

    with pytest.raises(MODULE.ReleaseError, match="not labeled zero"):
        release.validate_stability_window(window_seconds=120)


def test_stability_rejects_unexpected_baseline_critical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    baseline = {
        "restarts": {"cpu/pod/container": 0},
        "not_ready": [],
        "queue": {"depth": 0, "oldest_age_seconds": 0.0},
        "remote_commands": {"by_status": {}},
        "critical_alerts": {
            "count": 1,
            "alerts": [
                {"alertname": "GpuFaultRemoteCommandStalled", "severity": "critical"}
            ],
        },
    }
    monkeypatch.setattr(release, "_stability_snapshot", lambda: baseline)
    monkeypatch.setattr(
        release,
        "_store_io_rejection_series_ready",
        lambda: pytest.fail("unexpected critical alert entered settle path"),
    )

    with pytest.raises(MODULE.ReleaseError, match="non-settleable"):
        release.validate_stability_window(window_seconds=120)


def test_store_io_rejection_series_gate_reads_each_running_cpu_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))

    def get_json(command: list[str]) -> dict:
        if "deployment" in command:
            name = command[command.index("deployment") + 1]
            return {
                "spec": {
                    "replicas": (0 if name == "gpu-fault-telemetry-spool-worker" else 1)
                }
            }
        selector = command[command.index("-l") + 1]
        role = selector.split("=", 1)[1]
        return {"items": [{"metadata": {"name": f"{role}-pod"}}]}

    monkeypatch.setattr(release, "_get_json", get_json)
    monkeypatch.setattr(
        release.runner,
        "run",
        lambda *_args, **_kwargs: json.dumps(
            {"series_count": 1, "all_labeled": True, "all_zero": True}
        ),
    )

    report = VALIDATION_MODULE.store_io_rejection_series_ready(release)

    assert report["ready"] is True
    assert sorted(report["pods"]) == [
        "gpu-fault-api-ha/gpu-fault-api-ha-pod",
        "gpu-fault-control-worker/gpu-fault-control-worker-pod",
    ]


def test_store_io_rejection_series_gate_probes_pods_in_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    barrier = threading.Barrier(len(VALIDATION_MODULE.CPU_METRIC_PORTS), timeout=30)

    def get_json(command: list[str]) -> dict:
        if "deployment" in command:
            return {"spec": {"replicas": 1}}
        selector = command[command.index("-l") + 1]
        role = selector.split("=", 1)[1]
        return {"items": [{"metadata": {"name": f"{role}-pod"}}]}

    def run(args, **_kwargs):
        # Every probe waits for the others, so a serial gate would time out here.
        barrier.wait()
        return json.dumps({"series_count": 1, "all_labeled": True, "all_zero": True})

    monkeypatch.setattr(release, "_get_json", get_json)
    monkeypatch.setattr(release.runner, "run", run)

    report = VALIDATION_MODULE.store_io_rejection_series_ready(release)

    assert report["ready"] is True
    assert len(report["pods"]) == 3


def test_store_io_rejection_series_gate_reports_probes_in_stable_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))

    def get_json(command: list[str]) -> dict:
        if "deployment" in command:
            return {"spec": {"replicas": 1}}
        selector = command[command.index("-l") + 1]
        role = selector.split("=", 1)[1]
        return {"items": [{"metadata": {"name": f"{role}-pod"}}]}

    def run(args, **_kwargs):
        pod = args[args.index("exec") + 1]
        # The last role answers first, so the report order can only stay stable
        # if the fan-out consumes results in input order.
        if not pod.startswith("gpu-fault-telemetry-spool-worker"):
            time.sleep(0.05)
        return json.dumps({"series_count": 1, "all_labeled": False, "all_zero": True})

    monkeypatch.setattr(release, "_get_json", get_json)
    monkeypatch.setattr(release.runner, "run", run)

    report = VALIDATION_MODULE.store_io_rejection_series_ready(release)

    assert report["ready"] is False
    assert list(report["pods"]) == [
        "gpu-fault-api-ha/gpu-fault-api-ha-pod",
        "gpu-fault-control-worker/gpu-fault-control-worker-pod",
        "gpu-fault-telemetry-spool-worker/gpu-fault-telemetry-spool-worker-pod",
    ]
    assert report["errors"] == [
        "gpu-fault-api-ha/gpu-fault-api-ha-pod has unlabeled Store I/O series",
        (
            "gpu-fault-control-worker/gpu-fault-control-worker-pod has unlabeled "
            "Store I/O series"
        ),
        (
            "gpu-fault-telemetry-spool-worker/gpu-fault-telemetry-spool-worker-pod "
            "has unlabeled Store I/O series"
        ),
    ]


def test_store_io_rejection_series_gate_reraises_the_first_probe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))

    def get_json(command: list[str]) -> dict:
        if "deployment" in command:
            return {"spec": {"replicas": 1}}
        selector = command[command.index("-l") + 1]
        role = selector.split("=", 1)[1]
        return {"items": [{"metadata": {"name": f"{role}-pod"}}]}

    def run(args, **_kwargs):
        pod = args[args.index("exec") + 1]
        raise MODULE.ReleaseError(f"{pod} exec failed")

    monkeypatch.setattr(release, "_get_json", get_json)
    monkeypatch.setattr(release.runner, "run", run)

    # A failing probe must fail the gate instead of being reported as ready.
    with pytest.raises(MODULE.ReleaseError, match="gpu-fault-api-ha-pod exec failed"):
        VALIDATION_MODULE.store_io_rejection_series_ready(release)


def test_stability_snapshot_normalizes_decimal_store_stats(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    calls = []
    responses = iter(
        (
            ingress_pod_list_json("api-pod"),
            json.dumps(
                {
                    "queue": {"depth": 0, "oldest_age_seconds": 0.5},
                    "remote_commands": {"by_status": {}},
                }
            ),
        )
    )

    def run(command, **_kwargs):
        calls.append(command)
        return next(responses)

    monkeypatch.setattr(release.runner, "run", run)
    monkeypatch.setattr(release, "_get_json", lambda _command: {"items": []})
    monkeypatch.setattr(
        release, "_critical_amp_alerts", lambda: {"count": 0, "alerts": []}
    )

    snapshot = VALIDATION_MODULE.stability_snapshot(release)
    script = calls[1][calls[1].index("-c") + 1]

    assert "default=float" in script
    assert snapshot["queue"]["oldest_age_seconds"] == 0.5


def test_stability_window_rejects_new_restarts(tmp_path: Path, monkeypatch) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=False))
    baseline = {
        "restarts": {"cpu/pod/container": 0},
        "not_ready": [],
        "queue": {"depth": 0, "oldest_age_seconds": 0.0},
        "remote_commands": {"by_status": {}},
        "critical_alerts": {"count": 0, "alerts": []},
    }
    restarted = {**baseline, "restarts": {"cpu/pod/container": 1}}
    snapshots = iter((baseline, restarted))
    times = iter((0.0, 0.0, 0.0))
    monkeypatch.setattr(release, "_stability_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(MODULE.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)

    with pytest.raises(MODULE.ReleaseError, match="observed a restart"):
        release.validate_stability_window(window_seconds=120, sample_seconds=120)
