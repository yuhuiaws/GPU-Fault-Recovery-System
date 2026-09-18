from __future__ import annotations

import copy
import json
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest

from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import regional_release_validation as validation
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_diff import ReleaseComponent as Component
from gpu_fault_release.regional_release_diff import ReleaseExecutionPlan
from tests.regional._cov95_release_support import (
    DCGM_IMAGE,
    NEW_IMAGE,
    OLD_IMAGE,
    Clock,
    ResourceRelease,
    deployment,
    json_response,
    previous_snapshot,
    stable_sample,
)
from tests.regional.test_store_io_zero_view_probe import zero_view_report


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr(validation.time, "sleep", clock.sleep)
    monkeypatch.setattr(validation.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(validation.time, "time", clock.time)
    return clock


@pytest.mark.parametrize(
    "cpu,data_plane", [(False, False), (True, False), (False, True), (True, True)]
)
def test_component_validation_runs_only_selected_readonly_checks_and_binds_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu: bool, data_plane: bool
) -> None:
    release = ResourceRelease(("gpu-a", "gpu-b"))
    release.runner.handler = json_response({})
    runtime_checks = []
    evidence = tmp_path / "evidence" / "quick.json"
    monkeypatch.setenv(validation.QUICK_VALIDATION_EVIDENCE_ENV, str(evidence))
    validation.validate_release_components(
        release,
        cpu=cpu,
        data_plane=data_plane,
        runtime_validator=lambda value: runtime_checks.append(value.release_id),
    )
    written = json.loads(evidence.read_text())
    expected = []
    if cpu:
        expected.append("control_plane_role_split")
    if data_plane:
        expected.extend(["data_plane_executor:gpu-a", "data_plane_executor:gpu-b"])
    if cpu or data_plane:
        expected.append("runtime_component_identity")
    assert written["checks"] == sorted(expected)
    assert written["release_id"] == release.release_id
    assert written["site_identity"]["cluster_ids"] == ["gpu-a", "gpu-b"]
    assert evidence.stat().st_mode & 0o777 == 0o600
    assert evidence.parent.stat().st_mode & 0o777 == 0o700
    assert runtime_checks == ([release.release_id] if cpu or data_plane else [])
    assert len(release.runner.calls) == int(cpu) + 2 * int(data_plane)
    for arguments, kwargs in release.runner.calls:
        assert arguments[0] == "python3"
        assert kwargs["env"]["GPU_FAULT_NAMESPACE"] == release.config.namespace
        assert "GPU_FAULT_EXPECTED_WHEEL_CONFIGMAP" not in kwargs["env"] or (
            kwargs["env"]["GPU_FAULT_EXPECTED_WHEEL_CONFIGMAP"]
            == release.executor_wheel_cm
        )


def test_no_components_and_no_evidence_path_issue_no_commands() -> None:
    release = ResourceRelease()
    validation.validate_release_quick(
        release, ReleaseExecutionPlan(nodes=(Component.VERIFY,))
    )
    assert release.runner.calls == []


@pytest.mark.parametrize(
    "component", [Component.OBSERVABILITY, Component.AURORA_REFRESH]
)
def test_quick_validation_rejects_repair_drift_before_running_verifiers(
    component: Component,
) -> None:
    release = ResourceRelease()
    release.observability_drift = component == Component.OBSERVABILITY
    release.aurora_drift = component == Component.AURORA_REFRESH
    with pytest.raises(ReleaseError, match="remains out of date"):
        validation.validate_release_quick(
            release, ReleaseExecutionPlan(nodes=(component,))
        )
    assert release.runner.calls == []


@pytest.mark.parametrize("previous", [None, "", "candidate-profile"])
def test_unchanged_profile_never_queries_workloads(previous: str | None) -> None:
    release = ResourceRelease()
    validation.ensure_profile_transition_safe(release, previous)
    assert release.runner.calls == []


@pytest.mark.parametrize("payload", [{"count": 0, "alerts": []}, [], "malformed"])
def test_amp_probe_is_object_typed_and_absence_is_only_configuration_absence(
    payload: Any,
) -> None:
    release = ResourceRelease()
    assert validation.critical_amp_alerts(release) == {"count": 0, "alerts": []}
    assert release.runner.calls == []
    release.config.health.amp_workspace_id = "workspace-example"
    release.runner.handler = json_response(payload)
    if isinstance(payload, dict):
        assert validation.critical_amp_alerts(release) == payload
    else:
        with pytest.raises(ReleaseError, match="non-object"):
            validation.critical_amp_alerts(release)
    arguments, kwargs = release.runner.calls[0]
    assert arguments[:2] == ["python3", "-c"]
    assert json.loads(kwargs["input_text"]) == {
        "region": "us-east-1",
        "workspace_id": "workspace-example",
    }


def test_store_series_reports_missing_pods_and_nonzero_counters() -> None:
    release = ResourceRelease()
    for role in validation.CPU_METRIC_PORTS:
        release.documents[("cpu", "deployment", role)] = deployment(role, replicas=0)
    role = inventory.CPU_INGRESS_DEPLOYMENT
    release.documents[("cpu", "deployment", role)]["spec"]["replicas"] = 2
    release.documents[("cpu", "pods", "")] = {
        "items": [{"metadata": {"name": "api-a"}}, {"metadata": {}}]
    }
    release.runner.handler = json_response(
        zero_view_report(all_labeled=False, all_zero=False)
    )
    report = validation.store_io_rejection_series_ready(release)
    assert report["ready"] is False
    assert report["errors"] == [
        "gpu-fault-api-ha has 1/2 Running metric Pods",
        "gpu-fault-api-ha/api-a has unlabeled Store I/O series",
        "gpu-fault-api-ha/api-a has nonzero Store I/O rejections",
    ]
    assert len(release.runner.calls) == 1


@pytest.mark.parametrize(
    "sample_fault,problem",
    [
        ("not-ready", "non-Ready Pods"),
        ("restart", "observed a restart"),
        ("remote", "non-terminal remote commands"),
        ("unexpected", "unexpected critical alerts"),
        ("never-clears", "did not clear"),
    ],
)
def test_critical_clear_wait_rejects_instability_and_honors_deadline(
    clock: Clock, sample_fault: str, problem: str
) -> None:
    release = ResourceRelease()
    baseline = stable_sample()
    baseline["critical_alerts"] = {
        "count": 1,
        "alerts": [{"alertname": "GpuFaultStoreIoRejected"}],
    }
    sample = stable_sample()
    if sample_fault == "not-ready":
        sample["not_ready"] = ["cpu/api"]
    elif sample_fault == "restart":
        sample["restarts"]["cpu/api/api"] = 1
    elif sample_fault == "remote":
        sample["remote_commands"] = {"by_status": {"WAITING": 1}}
    elif sample_fault == "unexpected":
        sample["critical_alerts"] = {
            "count": 1,
            "alerts": [{"alertname": "UnsafeState"}],
        }
    else:
        sample = copy.deepcopy(baseline)
    release.samples = [sample]
    with pytest.raises(ReleaseError, match=problem):
        validation.wait_for_stability_baseline(
            release, baseline, timeout_seconds=3, sample_seconds=2
        )
    assert clock.sleeps == ([2, 1] if sample_fault == "never-clears" else [2])


@pytest.mark.parametrize("timeout,sample", [(0, 1), (1, 0)])
def test_critical_clear_invalid_budget_fails_before_sleep(
    clock: Clock, timeout: int, sample: int
) -> None:
    release = ResourceRelease()
    baseline = stable_sample()
    baseline["critical_alerts"] = {
        "count": 1,
        "alerts": [{"alertname": "GpuFaultStoreIoRejected"}],
    }
    with pytest.raises(ReleaseError, match="configuration is invalid"):
        validation.wait_for_stability_baseline(
            release, baseline, timeout_seconds=timeout, sample_seconds=sample
        )
    assert clock.sleeps == []


@pytest.mark.parametrize(
    "attempts",
    [
        None,
        [],
        {"gpu-a": None},
        {"gpu-a": {"converged_at_epoch": True}},
        {"gpu-a": {"converged_at_epoch": "recent"}},
    ],
)
def test_malformed_convergence_timestamp_cannot_enable_alert_grace(
    clock: Clock, attempts: Any
) -> None:
    release = ResourceRelease()
    release.live_state = {"cluster_attempts": attempts}
    baseline = stable_sample()
    baseline["critical_alerts"] = {
        "count": 1,
        "alerts": [{"alertname": "GpuFaultCollectorSilent"}],
    }
    with pytest.raises(ReleaseError, match="non-settleable"):
        validation.wait_for_stability_baseline(release, baseline)
    assert clock.sleeps == []


def test_unknown_counted_alert_is_not_silently_treated_as_zero(clock: Clock) -> None:
    release = ResourceRelease()
    baseline = stable_sample()
    baseline["critical_alerts"] = {"count": 1, "alerts": []}
    with pytest.raises(ReleaseError, match="<unknown>"):
        validation.wait_for_stability_baseline(release, baseline)
    assert clock.sleeps == []


@pytest.mark.parametrize("window,sample", [(119, 1), (301, 1), (120, 0), (120, 121)])
def test_stability_invalid_window_is_refused_before_any_sampling(
    clock: Clock, window: int, sample: int
) -> None:
    release = ResourceRelease()
    with pytest.raises(ReleaseError, match="window must be|sample interval is invalid"):
        validation.validate_stability_window(
            release, window_seconds=window, sample_seconds=sample
        )
    assert clock.sleeps == []


@pytest.mark.parametrize(
    "when,fault,problem",
    [
        ("baseline", "ready", "baseline has non-Ready"),
        ("sample", "ready", "window has non-Ready"),
        ("sample", "remote", "non-terminal remote"),
    ],
)
def test_stability_not_ready_or_pending_commands_prevent_success(
    clock: Clock, when: str, fault: str, problem: str
) -> None:
    release = ResourceRelease()
    broken = stable_sample()
    if fault == "ready":
        broken["not_ready"] = ["cpu/api"]
    else:
        broken["remote_commands"]["by_status"]["PENDING"] = 1
    release.samples = [broken] if when == "baseline" else [stable_sample(), broken]
    with pytest.raises(ReleaseError, match=problem):
        validation.validate_stability_window(release)
    assert clock.sleeps == ([] if when == "baseline" else [30])


def test_stability_snapshot_filters_only_nonruntime_pods_and_preserves_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = ResourceRelease(("gpu-a", "gpu-b"))
    healthy = {
        "metadata": {"name": "ready", "ownerReferences": [{"kind": "ReplicaSet"}]},
        "status": {
            "phase": "Running",
            "conditions": [{"type": "Ready", "status": "True"}],
            "containerStatuses": [{"name": "api", "restartCount": 2}],
        },
    }
    unhealthy = {
        "metadata": {"name": "pending", "ownerReferences": [{"kind": "DaemonSet"}]},
        "status": {
            "phase": "Pending",
            "conditions": [],
            "containerStatuses": [{"name": "api"}],
        },
    }
    release.documents[("cpu", "pods", "")] = {
        "items": [
            healthy,
            unhealthy,
            {"metadata": {"name": "terminating", "deletionTimestamp": "now"}},
            {"metadata": {"name": "proof", "ownerReferences": [{"kind": "Job"}]}},
        ]
    }
    for target in release.config.clusters:
        release.documents[(target.context, "pods", "")] = {"items": [healthy]}
    monkeypatch.setattr(
        validation,
        "exec_cpu_ingress_probe",
        lambda *_args, **_kwargs: json.dumps(
            {"queue": {"depth": 9}, "remote_commands": {"by_status": {"READY": 1}}}
        ),
    )
    report = validation.stability_snapshot(release)
    assert report["not_ready"] == ["cpu/pending"]
    assert report["restarts"] == {
        "cpu/ready/api": 2,
        "cpu/pending/api": 0,
        "gpu-a/ready/api": 2,
        "gpu-b/ready/api": 2,
    }
    assert report["queue"] == {"depth": 9}
    release.documents[("gpu-b", "pods", "")] = ReleaseError("GPU inventory unreadable")
    with pytest.raises(ReleaseError, match="GPU inventory unreadable"):
        validation.stability_snapshot(release)


def rollback_fixture(
    cluster_ids: tuple[str, ...] = ("gpu-a",),
) -> tuple[ResourceRelease, dict[str, Any]]:
    release = ResourceRelease(cluster_ids)
    previous = previous_snapshot(release)
    release.runner.handler = json_response({})
    for key, value in release.documents.items():
        if key[1] == "deployment":
            pod = value["spec"]["template"]["spec"]
            pod["containers"][0]["image"] = OLD_IMAGE
            pod["volumes"][0]["configMap"]["name"] = "previous-wheel"
    release.documents[("cpu", "deployment", inventory.CPU_INGRESS_DEPLOYMENT)] = (
        deployment(
            inventory.CPU_INGRESS_DEPLOYMENT,
            image=OLD_IMAGE,
            wheel="previous-cpu-wheel",
        )
    )
    release.documents[("cpu", "configmap", "gpu-fault-api-ha-config-core")] = {
        "data": {"GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": "previous-profile"}
    }
    return release, previous


@pytest.mark.parametrize(
    "component",
    [
        Component.EXECUTOR,
        Component.WATCHER,
        Component.COLLECTOR,
        Component.RECONCILER,
        Component.DCGM,
        Component.AGENT,
        Component.ENDPOINT,
    ],
)
def test_gpu_rollback_validates_only_selected_components(component: Component) -> None:
    release, previous = rollback_fixture()
    validation.validate_gpu_rollback_target(
        release,
        previous,
        OLD_IMAGE,
        release.config.clusters[0],
        components=frozenset({component}),
    )
    assert bool(release.heartbeat_checks) is (component == Component.AGENT)
    assert len(release.runner.calls) == int(
        component not in {Component.DCGM, Component.ENDPOINT}
    )
    if release.runner.calls:
        arguments, kwargs = release.runner.calls[0]
        assert arguments[1].endswith("verify_dataplane_executor.py"), (
            "GPU rollback must use the data-plane verifier entrypoint"
        )
        assert kwargs["env"]["GPU_FAULT_KUBE_CONTEXT"] == "gpu-a"
        assert kwargs["env"]["GPU_FAULT_EXPECTED_WHEEL_CONFIGMAP"] == "previous-wheel"


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("executor-wheel", "wheel mismatch"),
        ("executor-image", "image mismatch"),
        ("reconciler-image", "Reconciler image mismatch"),
        ("reconciler-wheel", "Reconciler wheel mismatch"),
        ("template", "installer template mismatch"),
        ("bundle", "node bundle mismatch"),
        ("dcgm", "DCGM image mismatch"),
    ],
)
def test_gpu_rollback_fails_before_claiming_convergence_on_identity_drift(
    fault: str, problem: str
) -> None:
    release, previous = rollback_fixture()
    name = (
        inventory.GPU_RECONCILER_DEPLOYMENT
        if fault.startswith("reconciler")
        else inventory.GPU_EXECUTOR_DEPLOYMENT
    )
    if fault.endswith("-wheel"):
        release.documents[("gpu-a", "deployment", name)]["spec"]["template"]["spec"][
            "volumes"
        ][0]["configMap"]["name"] = "other-wheel"
    elif fault.endswith("-image"):
        release.documents[("gpu-a", "deployment", name)]["spec"]["template"]["spec"][
            "containers"
        ][0]["image"] = NEW_IMAGE
    elif fault == "template":
        release.template_name = "other-template"
    elif fault == "bundle":
        release.template_bundle = "other-bundle"
    else:
        release.documents[("gpu-a", "daemonset", "gpu-fault-dcgm-exporter")]["spec"][
            "template"
        ]["spec"]["containers"][0]["image"] = NEW_IMAGE
    with pytest.raises(ReleaseError, match=problem):
        validation.validate_gpu_rollback_target(
            release, previous, OLD_IMAGE, release.config.clusters[0]
        )
    assert release.runner.calls == []


def test_legacy_rollback_without_optional_snapshot_pins_does_not_invent_them() -> None:
    release, previous = rollback_fixture()
    previous["clusters"] = {}
    validation.validate_gpu_rollback_target(
        release,
        previous,
        OLD_IMAGE,
        release.config.clusters[0],
        components=frozenset(
            {Component.EXECUTOR, Component.RECONCILER, Component.DCGM}
        ),
        run_verifier=False,
    )
    assert release.runner.calls == []
    assert not any(kind == "daemonset" for _, kind, _ in release.reads), (
        "a legacy snapshot without a DCGM pin must not invent a DCGM target"
    )


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("identity", "identity is missing"),
        ("nodes", "node set is empty"),
        ("artifact", "pins are incomplete"),
        ("config", "pins are incomplete"),
        ("annotation", "annotations mismatch"),
        ("heartbeat", "heartbeat mismatch"),
    ],
)
def test_agent_rollback_requires_complete_and_converged_previous_identity(
    fault: str, problem: str
) -> None:
    release, previous = rollback_fixture()
    identity = previous["agent_identities"]["gpu-a"]
    if fault == "identity":
        previous["agent_identities"].clear()
    elif fault == "nodes":
        identity["node_ids"] = []
    elif fault in {"artifact", "config"}:
        identity["artifact_sha256" if fault == "artifact" else "config_digest"] = ""
    elif fault == "annotation":
        release.documents[("gpu-a", "nodes", "")]["items"][0]["metadata"][
            "annotations"
        ] = {}
    else:
        release.heartbeat_ready = False
    with pytest.raises(ReleaseError, match=problem):
        validation.validate_agent_rollback_target(
            release, previous, release.config.clusters[0]
        )


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("wheel", "CPU wheel"),
        ("image", "CPU runtime image"),
        ("refresh", "Aurora refresh image"),
    ],
)
def test_cpu_rollback_rejects_wrong_wheel_image_and_legacy_refresher(
    monkeypatch: pytest.MonkeyPatch, fault: str, problem: str
) -> None:
    release, previous = rollback_fixture()
    pod = release.documents[("cpu", "deployment", inventory.CPU_INGRESS_DEPLOYMENT)][
        "spec"
    ]["template"]["spec"]
    if fault == "wheel":
        pod["volumes"][0]["configMap"]["name"] = "foreign"
    elif fault == "image":
        pod["containers"][0]["image"] = NEW_IMAGE
    refresh = {
        "spec": {
            "jobTemplate": {
                "spec": {"template": {"spec": {"containers": [{"image": NEW_IMAGE}]}}}
            }
        }
    }
    monkeypatch.setattr(
        validation, "read_aurora_refresh_cronjob", lambda _release: refresh
    )
    with pytest.raises(ReleaseError, match=problem):
        validation.validate_cpu_rollback(release, previous, OLD_IMAGE)


@pytest.mark.parametrize("restore_cpu", [True, False])
def test_rollback_full_refresher_snapshot_is_verified_in_both_scopes(
    monkeypatch: pytest.MonkeyPatch, restore_cpu: bool
) -> None:
    release, previous = rollback_fixture()
    previous["aurora_refresh"] = {"objects": [], "absent": []}
    snapshots = []
    monkeypatch.setattr(
        validation,
        "verify_aurora_refresh_snapshot",
        lambda _release, snapshot: snapshots.append(snapshot),
    )
    validation.validate_rollback(
        release, previous, restore_cpu=restore_cpu, cluster_components={}
    )
    assert snapshots == [previous["aurora_refresh"]]
    assert len(release.runner.calls) == int(restore_cpu)


@pytest.mark.parametrize("fault", ["metadata", "profile"])
def test_rollback_rejects_metadata_or_profile_drift_before_verifier(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    release, previous = rollback_fixture()
    if fault == "metadata":
        release.metadata["required-agent-artifact-sha256"] = "other"
    else:
        release.documents[("cpu", "configmap", "gpu-fault-api-ha-config-core")][
            "data"
        ].clear()
    monkeypatch.setattr(
        validation, "read_aurora_refresh_cronjob", lambda _release: None
    )
    with pytest.raises(
        ReleaseError, match="metadata mismatch|Runtime Profile did not converge"
    ):
        validation.validate_rollback(release, previous)
    assert release.runner.calls == []


@pytest.mark.parametrize("count,dry_run", [(0, False), (1, False), (3, False)])
@pytest.mark.parametrize("fail", [False, True])
def test_rollback_cluster_fanout_waits_for_all_selected_probes(
    monkeypatch: pytest.MonkeyPatch, count: int, dry_run: bool, fail: bool
) -> None:
    release, previous = rollback_fixture(
        tuple(f"gpu-{index}" for index in range(count))
    )
    selected = {
        target.cluster_id: frozenset({Component.EXECUTOR})
        for target in release.config.clusters
    }
    barrier = Barrier(count) if count > 1 else None
    checked = []

    def probe(
        _release: Any, _previous: Any, _image: str, target: Any, **kwargs: Any
    ) -> None:
        if barrier is not None:
            barrier.wait(timeout=5)
        checked.append(target.cluster_id)
        assert kwargs["components"] == frozenset({Component.EXECUTOR})
        if fail and target.cluster_id == "gpu-0":
            raise ReleaseError("probe failed")

    monkeypatch.setattr(validation, "validate_gpu_rollback_target", probe)
    if fail and count:
        with pytest.raises(
            ReleaseError, match="gpu-0 rollback validation failed: probe failed"
        ):
            validation.validate_rollback(
                release, previous, restore_cpu=False, cluster_components=selected
            )
    else:
        validation.validate_rollback(
            release, previous, restore_cpu=False, cluster_components=selected
        )
    assert sorted(checked) == sorted(selected), (
        "fanout must join all selected probes on failure"
    )


def test_rollback_default_scope_checks_every_gpu_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, previous = rollback_fixture(("gpu-a", "gpu-b"))
    monkeypatch.setattr(
        validation, "read_aurora_refresh_cronjob", lambda _release: None
    )
    validation.validate_rollback(release, previous)
    assert sorted(check["cluster_id"] for check in release.heartbeat_checks) == [
        "gpu-a",
        "gpu-b",
    ]
    assert len(release.runner.calls) == 3
    assert (
        release.documents[("gpu-a", "daemonset", "gpu-fault-dcgm-exporter")]["spec"][
            "template"
        ]["spec"]["containers"][0]["image"]
        == DCGM_IMAGE
    )
