import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
MODULE = lazy_script_module(
    "rollout_regional_release_rollback_state_tests",
    ROOT / "deploy/control-plane/regional/rollout_regional_release.py",
)
DIFF = lazy_script_module(
    "regional_release_diff_rollback_state_tests",
    ROOT / "deploy/control-plane/regional/regional_release_diff.py",
)
ORCHESTRATION = lazy_script_module(
    "regional_release_orchestration_rollback_state_tests",
    ROOT / "deploy/control-plane/regional/regional_release_orchestration.py",
)
VALIDATION = lazy_script_module(
    "regional_release_validation_rollback_state_tests",
    ROOT / "deploy/control-plane/regional/regional_release_validation.py",
)
STATE = lazy_script_module(
    "regional_release_state_rollback_state_tests",
    ROOT / "deploy/control-plane/regional/regional_release_state.py",
)
FLEET_ROLLOUT = lazy_script_module(
    "regional_release_fleet_rollout_rollback_state_tests",
    ROOT / "deploy/control-plane/regional/regional_release_fleet_rollout.py",
)
LEGACY = lazy_script_module(
    "regional_release_legacy_rollback_state_tests",
    ROOT / "deploy/control-plane/regional/regional_release_legacy.py",
)


def _legacy_agent_identity(node_ids: tuple[str, ...] = ("node-a",)) -> dict:
    return {
        "agent_protocol_version": 3,
        "agent_version": "0.10.0",
        "artifact_sha256": "artifact",
        "compatibility_digest": "compatibility",
        "installer_bundle_sha256": None,
        "installer_template_sha256": None,
        "policy_version": "catalog",
        "runtime_profile_version": "profile-v1",
        "config_digest": "config",
        "node_action_key_version": 2,
        "node_ids": list(node_ids),
    }


def test_legacy_contract_fields_match_the_deployed_release() -> None:
    assert LEGACY.AGENT_IDENTITY_FIELDS == (
        "agent_protocol_version",
        "agent_version",
        "artifact_sha256",
        "compatibility_digest",
        "installer_bundle_sha256",
        "installer_template_sha256",
        "policy_version",
        "runtime_profile_version",
        "config_digest",
        "node_action_key_version",
    )
    assert LEGACY.LEGACY_OPTIONAL_IDENTITY_FIELDS == (
        "installer_bundle_sha256",
        "installer_template_sha256",
    )
    assert LEGACY.LEGACY_NODE_ANNOTATION_FIELDS == (
        "gpu-fault.io/installer-artifact-sha256",
        "gpu-fault.io/installer-config-digest",
        "gpu-fault.io/installer-node-uid",
        "gpu-fault.io/installer-state",
    )


def test_rollback_controller_requires_one_shared_legacy_contract() -> None:
    first = _legacy_agent_identity()
    second = {**_legacy_agent_identity(), "node_ids": ["node-b"]}

    config = LEGACY.rollback_controller_config(
        {"cluster-a": first, "cluster-b": second}
    )

    assert config == {
        "GPU_FAULT_REQUIRED_AGENT_VERSION": "0.10.0",
        "GPU_FAULT_REQUIRED_POLICY_VERSION": "catalog",
        "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": "profile-v1",
    }
    second["policy_version"] = "other"
    with pytest.raises(MODULE.ReleaseError, match="policy_version differs"):
        LEGACY.rollback_controller_config({"cluster-a": first, "cluster-b": second})


def test_schema_job_manifest_change_keeps_automatic_rollback_available() -> None:
    class ValidationPassed(RuntimeError):
        pass

    release = SimpleNamespace(
        config=SimpleNamespace(
            auto_rollback=True, schema_rollback_compatible=False, clusters=()
        ),
        state={"release_diff": {"changed": ["schema_manifests"]}},
        _ensure_contexts=lambda: None,
        _require_cpu_secrets=lambda: None,
        _remote_commands_are_idle=lambda: True,
        _capture_previous=lambda: (_ for _ in ()).throw(ValidationPassed()),
    )
    diff = DIFF.ReleaseDiff(
        kind=DIFF.ReleaseChangeKind.FULL, changed=frozenset({"schema_manifests"})
    )

    with pytest.raises(ValidationPassed):
        ORCHESTRATION.upgrade_release(release, diff=diff)
    with pytest.raises(
        MODULE.ReleaseError, match="previous release pins are incomplete"
    ):
        ORCHESTRATION.rollback_release(release, state={"metadata": {}})


def test_database_schema_change_still_requires_rollback_compatibility() -> None:
    release = SimpleNamespace(
        config=SimpleNamespace(auto_rollback=True, schema_rollback_compatible=False),
        state={"release_diff": {"changed": ["database_schema"]}},
        _ensure_contexts=lambda: None,
        _require_cpu_secrets=lambda: None,
        _remote_commands_are_idle=lambda: True,
    )
    diff = DIFF.ReleaseDiff(
        kind=DIFF.ReleaseChangeKind.FULL, changed=frozenset({"database_schema"})
    )

    with pytest.raises(MODULE.ReleaseError, match="PostgreSQL schema change"):
        ORCHESTRATION.upgrade_release(release, diff=diff)
    with pytest.raises(MODULE.ReleaseError, match="PostgreSQL schema change"):
        ORCHESTRATION.rollback_release(release, state={"metadata": {}})


def test_previous_release_snapshot_reads_live_images() -> None:
    previous_runtime = "registry.example/runtime:previous"
    previous_installer = "registry.example/installer:previous"
    previous_adot = "registry.example/adot:previous"
    previous_dcgm = "registry.example/dcgm:previous"
    target = SimpleNamespace(cluster_id="gpu-a")
    config = SimpleNamespace(
        namespace="gpu-fault-system", clusters=(target,), bundle=Path("bundle.tar.gz")
    )

    def get_json(arguments: list[str]) -> dict:
        resource = arguments[arguments.index("get") + 1]
        name = arguments[arguments.index("get") + 2]
        if resource == "configmap":
            assert name == "installer-template"
            return {
                "data": {
                    "job.yaml": yaml.safe_dump(
                        {
                            "apiVersion": "batch/v1",
                            "kind": "Job",
                            "spec": {
                                "template": {
                                    "spec": {
                                        "containers": [
                                            {
                                                "name": "installer",
                                                "image": previous_installer,
                                            }
                                        ]
                                    }
                                }
                            },
                        }
                    )
                }
            }
        if resource == "daemonset":
            assert name == "gpu-fault-dcgm-exporter"
            image = previous_dcgm
        else:
            image = previous_adot if name == "gpu-fault-adot" else previous_runtime
        return {
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": (
                                    "collector" if name == "gpu-fault-adot" else "app"
                                ),
                                "image": image,
                            }
                        ]
                    }
                }
            }
        }

    release = SimpleNamespace(
        state={
            "release_delivery_sha256": "1" * 64,
            "rendered_manifest_sha256": "2" * 64,
            "node_template_sha256": "3" * 64,
        },
        config=config,
        runtime_image="registry.example/runtime:candidate",
        node_installer_image="registry.example/installer:candidate",
        adot_image="registry.example/adot:candidate",
        _cpu=lambda *args: ["cpu", *args],
        _gpu=lambda _target, *args: ["gpu", *args],
        _get_json=get_json,
        _config_map_data=lambda name: (
            {"GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": "profile-v1"}
            if name == "gpu-fault-api-ha-config-core"
            else {}
        ),
        _deployment_template_name=lambda _target: "installer-template",
        _deployment_wheel=lambda _args, name: f"{name}-wheel",
        _template_bundle=lambda _target, _template: "node-bundle",
        _config_map_binary_key=lambda _args, name: f"{name}-key",
        _config_map_sha=lambda *_args: "4" * 64,
        _capture_agent_identities=lambda: {target.cluster_id: _legacy_agent_identity()},
        _target_node_names=lambda _target: ("node-a",),
    )

    previous = STATE.capture_previous(release)

    assert previous["runtime_image"] == previous_runtime
    assert previous["node_installer_image"] == previous_installer
    assert previous["adot_image"] == previous_adot
    assert previous["runtime_image"] != release.runtime_image
    assert previous["release_delivery_sha256"] == "1" * 64
    assert previous["clusters"][target.cluster_id]["dcgm_image"] == previous_dcgm
    assert previous["agent_identities"][target.cluster_id]["agent_version"] == "0.10.0"


def test_legacy_state_adoption_uses_verified_rollback_runtime_image() -> None:
    previous_runtime = "registry.example/runtime:legacy"
    rollback_runtime = "registry.example/runtime@sha256:" + "a" * 64
    target = SimpleNamespace(cluster_id="gpu-a")
    release = _snapshot_release(
        target=target,
        previous_runtime=previous_runtime,
        previous_installer="registry.example/installer:previous",
        previous_adot="registry.example/adot:previous",
        previous_dcgm="registry.example/dcgm:previous",
        state={
            "release_id": "legacy",
            "runtime_image": rollback_runtime,
            "adopted_live_runtime_image": previous_runtime,
        },
    )

    previous = STATE.capture_previous(release)

    assert previous["live_runtime_image"] == previous_runtime
    assert previous["runtime_image"] == rollback_runtime


def test_legacy_state_adoption_rejects_live_runtime_drift() -> None:
    target = SimpleNamespace(cluster_id="gpu-a")
    release = _snapshot_release(
        target=target,
        previous_runtime="registry.example/runtime:drifted",
        previous_installer="registry.example/installer:previous",
        previous_adot="registry.example/adot:previous",
        previous_dcgm="registry.example/dcgm:previous",
        state={
            "release_id": "legacy",
            "runtime_image": "registry.example/runtime@sha256:" + "a" * 64,
            "adopted_live_runtime_image": "registry.example/runtime:adopted",
        },
    )

    with pytest.raises(MODULE.ReleaseError, match="drifted after legacy"):
        STATE.capture_previous(release)


def test_previous_release_snapshot_rejects_runtime_image_drift() -> None:
    with pytest.raises(MODULE.ReleaseError, match="runtime images are inconsistent"):
        STATE.require_consistent_images(
            "runtime",
            {
                "cpu/ingress": "registry.example/runtime:one",
                "gpu/executor": "registry.example/runtime:two",
            },
        )


def test_agent_identity_snapshot_captures_the_exact_legacy_contract() -> None:
    identity = _legacy_agent_identity(("node-a", "node-b"))
    records = [
        {"cluster_id": "gpu-a", "node_id": node_id, **identity}
        for node_id in identity["node_ids"]
    ]
    for item in records:
        item.pop("node_ids", None)

    class Runner:
        @staticmethod
        def run(arguments, **_kwargs):
            return "api-pod" if "get" in arguments else json.dumps(records)

    release = SimpleNamespace(
        runner=Runner(),
        config=SimpleNamespace(
            namespace="gpu-fault-system",
            clusters=(SimpleNamespace(cluster_id="gpu-a"),),
        ),
        _cpu=lambda *args: list(args),
    )

    captured = STATE.capture_agent_identities(release)

    assert captured["gpu-a"] == identity


def test_legacy_current_nodes_skip_fleet_rollback() -> None:
    target = SimpleNamespace(cluster_id="gpu-a")
    deploy_calls = []
    release = SimpleNamespace(
        runner=SimpleNamespace(dry_run=False),
        config=SimpleNamespace(
            runtime_profile_version="profile-v1",
            component_digests={"node_runtime": "compatibility"},
        ),
        _target_node_names=lambda _target: ("node-a",),
        _deploy_reconciler=lambda _target, **kwargs: (
            deploy_calls.append(kwargs) or ("bundle", "template")
        ),
        _get_json=lambda _args: {
            "items": [
                {
                    "metadata": {
                        "name": "node-a",
                        "uid": "uid-a",
                        "annotations": {
                            "gpu-fault.io/installer-artifact-sha256": "artifact",
                            "gpu-fault.io/installer-config-digest": "config",
                            "gpu-fault.io/installer-node-uid": "uid-a",
                            "gpu-fault.io/installer-state": "Succeeded",
                        },
                    }
                }
            ]
        },
        _gpu=lambda _target, *args: list(args),
        _fleet_deployment_id=lambda *_args, **_kwargs: pytest.fail(
            "legacy current nodes entered FleetDeployment"
        ),
        _fleet_command=lambda *_args, **_kwargs: pytest.fail(
            "legacy current nodes called Fleet API"
        ),
        _wait_agents=lambda *_args, **_kwargs: pytest.fail(
            "legacy current nodes unnecessarily waited for new identity fields"
        ),
    )

    identity = FLEET_ROLLOUT.roll_node_runtime(
        release,
        target,
        phase="rollback",
        wheel_cm="wheel",
        bundle_cm="bundle",
        artifact_sha="artifact",
        config_digest="config",
        bundle_sha256="bundle",
        template_sha256="template",
        allow_legacy_identity=True,
    )

    assert identity == ("bundle", "template")
    assert [call["allowed_node_names"] for call in deploy_calls] == [(), None]


def test_legacy_partial_rollback_omits_new_fleet_identity_fields() -> None:
    target = SimpleNamespace(cluster_id="gpu-a")
    fleet_requests = []
    fleet_identities = []
    waits = []
    commands = iter(
        [{"status": "PENDING"}, {"node_ids": ["node-a"]}, {"status": "SUCCEEDED"}]
    )
    deploy_calls = []
    release = SimpleNamespace(
        runner=SimpleNamespace(dry_run=False),
        release_id="candidate",
        config=SimpleNamespace(
            runtime_profile_version="profile-v1",
            component_digests={"node_runtime": "compatibility"},
        ),
        _target_node_names=lambda _target: ("node-a",),
        _deploy_reconciler=lambda _target, **kwargs: (
            deploy_calls.append(kwargs) or ("bundle", "template")
        ),
        _get_json=lambda _args: {
            "items": [
                {
                    "metadata": {
                        "name": "node-a",
                        "uid": "uid-a",
                        "annotations": {
                            "gpu-fault.io/installer-artifact-sha256": "candidate",
                            "gpu-fault.io/installer-config-digest": "candidate",
                            "gpu-fault.io/installer-node-uid": "uid-a",
                            "gpu-fault.io/installer-state": "Succeeded",
                        },
                    }
                }
            ]
        },
        _gpu=lambda _target, *args: list(args),
        _fleet_deployment_id=lambda _target, **kwargs: (
            fleet_identities.append(kwargs) or "deployment"
        ),
        _fleet_command=lambda operation, payload: (
            fleet_requests.append((operation, payload)) or next(commands)
        ),
        _wait_agents=lambda _target, _artifact, **kwargs: waits.append(kwargs),
    )

    identity = FLEET_ROLLOUT.roll_node_runtime(
        release,
        target,
        phase="rollback",
        wheel_cm="wheel",
        bundle_cm="bundle",
        artifact_sha="old-artifact",
        config_digest="old-config",
        bundle_sha256="old-bundle",
        template_sha256="old-template",
        runtime_image="candidate-runtime",
        steady_runtime_image="previous-runtime",
        steady_template_config_map="previous-template",
        allow_legacy_identity=True,
        agent_identity=_legacy_agent_identity(),
    )

    request = fleet_requests[0][1]["request"]
    assert identity == ("bundle", "template")
    assert request["desired_bundle_sha256"] is None
    assert request["desired_template_sha256"] is None
    assert fleet_identities[0]["bundle_sha"] is None
    assert fleet_identities[0]["template_sha"] is None
    assert request["desired_agent_protocol_version"] == 3
    assert request["desired_agent_version"] == "0.10.0"
    assert request["desired_policy_version"] == "catalog"
    assert waits and all(item["legacy_identity"] is True for item in waits), (
        "legacy rollback wave did not use legacy Agent convergence"
    )
    assert all(item["agent_identity"] == _legacy_agent_identity() for item in waits), (
        "legacy rollback wave did not retain the captured Agent identity"
    )
    assert [item["runtime_image"] for item in deploy_calls] == [
        "candidate-runtime",
        "candidate-runtime",
        "previous-runtime",
    ]
    assert deploy_calls[-1]["template_config_map"] == "previous-template"


def test_rollback_restores_gpu_before_cpu() -> None:
    source = inspect.getsource(ORCHESTRATION.rollback_release)

    assert source.index("_stage_rollback_controller(") < source.index(
        "_rollback_gpu_clusters("
    )
    assert source.index("_rollback_gpu_clusters(") < source.index(
        "_restore_rollback_cpu("
    )


def test_rollback_verifier_receives_previous_runtime_image() -> None:
    source = inspect.getsource(VALIDATION.validate_rollback)

    assert '"GPU_FAULT_RUNTIME_IMAGE": expected_runtime_image' in source


def test_unverified_rollback_data_checkpoint_is_replayed() -> None:
    source = inspect.getsource(ORCHESTRATION.rollback_release)

    assert 'loaded.get("phase") == "rollback-data-restored"' in source
    assert 'completed_phases.discard("rollback-data-restored")' in source
    assert "completed_clusters.clear()" in source


def _snapshot_release(
    *,
    target: SimpleNamespace,
    previous_runtime: str,
    previous_installer: str,
    previous_adot: str,
    previous_dcgm: str,
    state: dict,
) -> SimpleNamespace:
    config = SimpleNamespace(
        namespace="gpu-fault-system", clusters=(target,), bundle=Path("bundle.tar.gz")
    )

    def get_json(arguments: list[str]) -> dict:
        resource = arguments[arguments.index("get") + 1]
        name = arguments[arguments.index("get") + 2]
        if resource == "configmap":
            return {
                "data": {
                    "job.yaml": yaml.safe_dump(
                        {
                            "spec": {
                                "template": {
                                    "spec": {
                                        "containers": [
                                            {
                                                "name": "installer",
                                                "image": previous_installer,
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                    )
                }
            }
        image = (
            previous_dcgm
            if resource == "daemonset"
            else previous_adot
            if name == "gpu-fault-adot"
            else previous_runtime
        )
        return {
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": (
                                    "collector" if name == "gpu-fault-adot" else "app"
                                ),
                                "image": image,
                            }
                        ]
                    }
                }
            }
        }

    return SimpleNamespace(
        state=state,
        config=config,
        _cpu=lambda *args: ["cpu", *args],
        _gpu=lambda _target, *args: ["gpu", *args],
        _get_json=get_json,
        _config_map_data=lambda name: (
            {"GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": "profile-v1"}
            if name == "gpu-fault-api-ha-config-core"
            else {}
        ),
        _deployment_template_name=lambda _target: "installer-template",
        _deployment_wheel=lambda _args, name: f"{name}-wheel",
        _template_bundle=lambda _target, _template: "node-bundle",
        _config_map_binary_key=lambda _args, name: f"{name}-key",
        _config_map_sha=lambda *_args: "4" * 64,
        _capture_agent_identities=lambda: {target.cluster_id: _legacy_agent_identity()},
        _target_node_names=lambda _target: ("node-a",),
    )
