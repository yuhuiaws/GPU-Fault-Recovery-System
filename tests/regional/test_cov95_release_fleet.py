from __future__ import annotations

import json
from typing import Any

import pytest

from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import regional_release_fleet_rollout as fleet
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_images import NodeDependencyTarget
from tests.regional._cov95_release_support import Clock, ResourceRelease
from tests.regional._resource_probe_fakes import resource_probe_result


class FleetRelease(ResourceRelease):
    def __init__(self) -> None:
        super().__init__()
        self.config.upgrade_max_unavailable = 0
        self.config.rollback_max_unavailable = 2
        self.config.upgrade_max_parallel_clusters = 1
        self.backups: list[Any] = ["cpu-backup", "gpu-backup"]
        self.backup_calls = []
        self.deploy_calls = []
        self.agent_waits = []
        self.fleet_calls = []
        self.lease = ["node-a"]
        self.reconciler_identity = self.bundle_sha, self.node_template_sha

    def _backup_secret(self, arguments: list[str], **kwargs: Any) -> str | None:
        self.backup_calls.append((arguments, kwargs))
        result = self.backups.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def _deploy_reconciler(self, target: Any, **kwargs: Any) -> tuple[str, str]:
        self.deploy_calls.append((target.cluster_id, kwargs))
        return self.reconciler_identity

    def _fleet_command(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.fleet_calls.append((operation, payload))
        if operation == "next-wave":
            return {"node_ids": self.lease}
        return {
            "status": "SUCCEEDED",
            "waves": [["node-a"]],
            "nodes": [{"node_id": "node-a", "status": "READY"}],
        }

    def _wait_agents(self, target: Any, artifact: str, **kwargs: Any) -> None:
        self.agent_waits.append((target.cluster_id, artifact, kwargs))


def wave_context(
    release: FleetRelease, phase: str = "upgrade"
) -> fleet.FleetWaveContext:
    return fleet.FleetWaveContext(
        phase=phase,
        deployment_id="fleet-example",
        node_names=("node-a",),
        paused_identity=(release.bundle_sha, release.node_template_sha),
        wheel_cm="wheel",
        bundle_cm="bundle",
        artifact_sha=release.node_wheel_sha,
        config_digest=release.config.agent_config_digest,
        expected_profile=release.config.runtime_profile_version,
        executor_wheel_filename=None,
        expected_compatibility=release.node_wheel_sha,
        desired_bundle=release.bundle_sha,
        desired_template=release.node_template_sha,
        template_config_map=None,
        max_unavailable=1,
        runtime_image=None,
        node_installer_image=None,
        allow_legacy_identity=False,
        agent_identity=None,
    )


@pytest.mark.parametrize("required", [False, True])
def test_missing_backup_source_is_optional_only_when_explicit(required: bool) -> None:
    release = ResourceRelease()
    release.runner.probe_output = lambda arguments, **_kwargs: resource_probe_result(
        arguments, present=False
    )
    if required:
        with pytest.raises(ReleaseError, match="required Secret is missing"):
            fleet.backup_secret(
                release, ["kubectl"], source="source", backup="backup", required=True
            )
    else:
        assert (
            fleet.backup_secret(
                release, ["kubectl"], source="source", backup="backup", required=False
            )
            is None
        )
    assert release.runner.calls == []


def test_backup_source_data_must_be_strings() -> None:
    release = ResourceRelease()

    def probe(arguments: list[str], **_kwargs: Any) -> tuple[int, str, str]:
        code, output, error = resource_probe_result(arguments)
        document = json.loads(output)
        document["data"] = {"value": 1}
        return code, json.dumps(document), error

    release.runner.probe_output = probe
    with pytest.raises(ReleaseError, match="source data is invalid"):
        fleet.backup_secret(
            release, ["kubectl"], source="source", backup="backup", required=True
        )
    assert release.runner.calls == []


@pytest.mark.parametrize("cpu", [None, "cpu-backup"])
@pytest.mark.parametrize(
    "gpu", [None, "gpu-backup", ReleaseError("GPU backup refused")]
)
def test_backup_batch_cleans_up_only_created_backups_on_failure(
    cpu: str | None, gpu: Any
) -> None:
    release = FleetRelease()
    release.backups = [cpu, gpu]
    release.runner.handler = lambda *_args: ""
    if gpu is None or isinstance(gpu, Exception):
        with pytest.raises(ReleaseError, match="backup failed|backup refused"):
            fleet.backup_release_secrets(release)
        assert len(release.runner.calls) == int(cpu is not None)
        if cpu:
            assert release.runner.calls[0][0][-2:] == [cpu, "--ignore-not-found"]
    else:
        result = fleet.backup_release_secrets(release)
        assert result["cpu"] is None if cpu is None else result["cpu"]["backup"] == cpu
        assert result["clusters"]["gpu-a"]["backup"] == gpu
        assert release.runner.calls == []


def test_backup_cleanup_contacts_only_current_clusters() -> None:
    release = ResourceRelease()
    release.runner.handler = lambda *_args: ""
    previous = {
        "secret_backups": {
            "cpu": {"backup": "cpu-backup"},
            "clusters": {
                "gpu-a": {"backup": "gpu-backup"},
                "removed": {"backup": "foreign"},
            },
        }
    }
    fleet.delete_release_secret_backups(release, previous)
    assert [args[-2] for args, _kwargs in release.runner.calls] == [
        "cpu-backup",
        "gpu-backup",
    ]
    assert all(kwargs["sensitive"] for _args, kwargs in release.runner.calls), (
        "backup cleanup must retain sensitive command handling"
    )
    assert (
        fleet.referenced_secret_backups(
            {"secret_backups": {"cpu": [], "clusters": {"gpu-a": None, "gpu-b": {}}}}
        )
        == set()
    )


def test_target_node_and_failure_domain_inventory_is_complete() -> None:
    release = ResourceRelease()
    target = release.config.clusters[0]
    assert fleet.target_node_names(release, target) == ("node-a",)
    release.documents[("gpu-a", "nodes", "")]["items"].append(
        {"metadata": {"name": "unrelated"}}
    )
    assert fleet.target_node_failure_domains(release, target, ("node-a",)) == {
        "node-a": fleet.UNKNOWN_FAILURE_DOMAIN
    }
    with pytest.raises(ReleaseError, match="inventory is incomplete"):
        fleet.target_node_failure_domains(release, target, ("node-a", "missing"))
    assert not fleet.nodes_have_legacy_installer_identity(
        release,
        target,
        node_names=("missing",),
        artifact_sha=release.node_wheel_sha,
        config_digest=release.config.agent_config_digest,
    ), "a missing node must not satisfy legacy installer convergence"
    release.documents[("gpu-a", "nodes", "")]["items"] = []
    with pytest.raises(ReleaseError, match="no HyperPod nodes"):
        fleet.target_node_names(release, target)


@pytest.mark.parametrize("payload", [{}, [], None])
def test_fleet_command_accepts_only_object_response(
    monkeypatch: pytest.MonkeyPatch, payload: Any
) -> None:
    observed = []

    def execute(_release: Any, **kwargs: Any) -> str:
        observed.append(json.loads(kwargs["input_text"]))
        return json.dumps(payload)

    monkeypatch.setattr(fleet, "exec_cpu_ingress_command", execute)
    if isinstance(payload, dict):
        assert (
            fleet.fleet_command(ResourceRelease(), "get", {"deployment_id": "example"})
            == payload
        )
    else:
        with pytest.raises(ReleaseError, match="non-object"):
            fleet.fleet_command(ResourceRelease(), "get", {"deployment_id": "example"})
    assert observed == [{"operation": "get", "deployment_id": "example"}]


@pytest.mark.parametrize("fault", ["phase", "nodes", "parallel", "rollback-nodes"])
def test_rollout_policy_refuses_missing_nodes_and_invalid_parallelism(
    fault: str,
) -> None:
    release = FleetRelease()
    if fault == "parallel":
        release.config.upgrade_max_parallel_clusters = 9
    with pytest.raises(ReleaseError, match="policy|no nodes|parallelism"):
        if fault == "rollback-nodes":
            fleet.rollback_node_rollout_policy(release, {})
        else:
            fleet.node_rollout_policy(
                release,
                {} if fault == "nodes" else 1,
                phase="rollback" if fault == "phase" else "upgrade",
            )


def test_completed_deployment_has_no_next_wave() -> None:
    assert (
        fleet.next_deployment_wave(
            {"waves": [["a"]], "nodes": [{"node_id": "a", "status": "READY"}]}
        )
        == ()
    )
    assert fleet.next_deployment_wave({"waves": []}) == ()


@pytest.mark.parametrize("raw", ["{", "[]", "null"])
def test_candidate_heartbeat_barrier_rejects_malformed_or_nonobject_evidence(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setattr(fleet, "exec_cpu_ingress_probe", lambda *_args, **_kwargs: raw)
    with pytest.raises(ReleaseError, match="invalid evidence"):
        fleet.wait_candidate_cpu_agent_heartbeats(
            ResourceRelease(), {"gpu-a": {"node_ids": ["node-a"]}}
        )


def test_heartbeat_barrier_skips_only_empty_selection_or_explicit_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = ResourceRelease()
    observed = []
    monkeypatch.setattr(
        fleet,
        "exec_cpu_ingress_probe",
        lambda *_args, **kwargs: observed.append(kwargs) or '{"status":"PASSED"}',
    )
    fleet.wait_candidate_cpu_agent_heartbeats(
        release, {"gpu-a": None, "gpu-b": {"node_ids": []}}
    )
    release.runner.dry_run = True
    fleet.wait_candidate_cpu_agent_heartbeats(
        release, {"gpu-a": {"node_ids": ["node-a"]}}
    )
    assert observed == []
    assert fleet.capture_active_agent_node_sets(release) == {}


@pytest.mark.parametrize(
    "raw,problem",
    [
        ("{", "inventory is invalid"),
        ("[]", "not an object"),
        ('{"gpu-a":[]}', "missing clusters"),
    ],
)
def test_active_node_inventory_does_not_guess_missing_cluster_evidence(
    monkeypatch: pytest.MonkeyPatch, raw: str, problem: str
) -> None:
    monkeypatch.setattr(fleet, "exec_cpu_ingress_probe", lambda *_args, **_kwargs: raw)
    with pytest.raises(ReleaseError, match=problem):
        fleet.capture_active_agent_node_sets(ResourceRelease())


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("executor", "not fully Ready"),
        ("json", "invalid evidence"),
        ("shape", "non-object evidence"),
    ],
)
def test_wave_safety_requires_executor_readiness_and_object_evidence(
    monkeypatch: pytest.MonkeyPatch, fault: str, problem: str
) -> None:
    release = ResourceRelease()
    if fault == "executor":
        release.documents[("gpu-a", "deployment", inventory.GPU_EXECUTOR_DEPLOYMENT)][
            "status"
        ]["readyReplicas"] = 0
    monkeypatch.setattr(
        fleet,
        "exec_cpu_ingress_probe",
        lambda *_args, **_kwargs: "{" if fault == "json" else "[]",
    )
    with pytest.raises(ReleaseError, match=problem):
        fleet.rollout_wave_safety_snapshot(
            release,
            release.config.clusters[0],
            wave=("node-a",),
            node_names=("node-a",),
            minimum_lease_remaining_seconds=40,
        )


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("remote", "blocked by remote commands"),
        ("workflow", "active destructive workflows"),
        ("agent", "Agent safety did not converge"),
    ],
)
def test_wave_safety_refuses_activity_and_bounds_agent_wait(
    monkeypatch: pytest.MonkeyPatch, fault: str, problem: str
) -> None:
    clock = Clock()
    monkeypatch.setattr(fleet.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(fleet.time, "sleep", clock.sleep)
    snapshot = {
        "open_remote": {"PENDING": 1} if fault == "remote" else {},
        "destructive_workflow_count": int(fault == "workflow"),
        "agent_blocker_count": int(fault == "agent"),
    }
    monkeypatch.setattr(
        fleet, "exec_cpu_ingress_probe", lambda *_args, **_kwargs: json.dumps(snapshot)
    )
    release = ResourceRelease()
    with pytest.raises(ReleaseError, match=problem):
        fleet.ensure_rollout_wave_safe(
            release,
            release.config.clusters[0],
            wave=("node-a",),
            node_names=("node-a",),
            timeout_seconds=1,
            poll_seconds=1,
        )
    assert clock.sleeps == ([1] if fault == "agent" else [])


def test_reconciler_read_requires_named_container_and_complete_digest_pair() -> None:
    release = ResourceRelease()
    target = release.config.clusters[0]
    assert fleet.reconciler_installer_identity(
        fleet.reconciler_container_env(release, target)
    ) == (release.bundle_sha, release.node_template_sha)
    release.documents[("gpu-a", "deployment", inventory.GPU_RECONCILER_DEPLOYMENT)][
        "spec"
    ]["template"]["spec"]["containers"].clear()
    with pytest.raises(ReleaseError, match="no reconciler container"):
        fleet.reconciler_container_env(release, target)
    with pytest.raises(ReleaseError, match="identity is invalid"):
        fleet.reconciler_installer_identity({})


@pytest.mark.parametrize("phase", ["upgrade", "rollback"])
@pytest.mark.parametrize(
    "fault", ["failed", "no-wave", "empty-lease", "changed-wave", "changed-identity"]
)
def test_wave_failure_stops_before_agents_and_records_rollback_failure_when_applicable(
    monkeypatch: pytest.MonkeyPatch, phase: str, fault: str
) -> None:
    release = FleetRelease()
    context = wave_context(release, phase)
    deployment = {
        "status": "RUNNING",
        "waves": [["node-a"]],
        "nodes": [{"node_id": "node-a", "status": "PENDING"}],
    }
    if fault == "failed":
        deployment["status"] = "FAILED"
    elif fault == "no-wave":
        deployment["waves"] = []
    elif fault == "empty-lease":
        release.lease = []
    elif fault == "changed-wave":
        release.lease = ["other"]
    else:
        release.reconciler_identity = ("other", "other")
    monkeypatch.setattr(
        fleet,
        "exec_cpu_ingress_probe",
        lambda *_args, **_kwargs: json.dumps({"minimum_lease_remaining_seconds": 60}),
    )
    monkeypatch.setattr(fleet, "hand_wave_to_reconciler", lambda *_args: None)
    events = []
    monkeypatch.setattr(
        fleet,
        "record_rollback_wave_event",
        lambda _release, **kwargs: events.append(kwargs["event"]),
    )
    with pytest.raises(
        ReleaseError,
        match="deployment failed|no pending wave|empty wave|wave changed|identity changed",
    ):
        fleet.run_fleet_waves(release, release.config.clusters[0], context, deployment)
    assert release.agent_waits == []
    if phase == "rollback" and fault not in {"failed", "no-wave"}:
        assert events == [
            "safety_started",
            *(["safety_completed"] if fault == "changed-identity" else []),
            "failed",
        ]
    else:
        assert events == []


def test_legacy_rollback_identity_is_checked_after_reconciler_install() -> None:
    release = FleetRelease()
    release.reconciler_identity = ("other", "other")
    context = wave_context(release, "rollback")
    with pytest.raises(ReleaseError, match="legacy installer identity changed"):
        fleet.finish_legacy_node_runtime_rollback(
            release,
            release.config.clusters[0],
            enabled=True,
            node_names=context.node_names,
            paused_identity=context.paused_identity,
            wheel_cm=context.wheel_cm,
            bundle_cm=context.bundle_cm,
            artifact_sha=context.artifact_sha,
            config_digest=context.config_digest,
            runtime_profile_version=context.expected_profile,
            executor_wheel_filename=None,
            node_compatibility_digest=context.expected_compatibility,
            template_config_map=None,
            runtime_image=None,
            steady_runtime_image=None,
            steady_template_config_map=None,
            node_installer_image=None,
            max_unavailable=1,
            node_dependency_target=NodeDependencyTarget.PREVIOUS,
        )
    assert len(release.deploy_calls) == 1
    assert release.deploy_calls[0][1]["allowed_node_names"] is None
