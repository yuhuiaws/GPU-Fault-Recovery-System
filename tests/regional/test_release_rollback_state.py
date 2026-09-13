import base64
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault_release import regional_observability_rollback as OBSERVABILITY
from gpu_fault_release import regional_release_diff as DIFF
from gpu_fault_release import regional_release_fleet_rollout as FLEET_ROLLOUT
from gpu_fault_release import regional_release_legacy as LEGACY
from gpu_fault_release import (
    regional_release_node_runtime_rollout as NODE_RUNTIME_ROLLOUT,
)
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release import regional_release_rollback_context as ROLLBACK_CONTEXT
from gpu_fault_release import regional_release_rollout_cleanup as ROLLOUT_CLEANUP
from gpu_fault_release import regional_release_state as STATE
from gpu_fault_release import regional_release_validation as VALIDATION
from gpu_fault_release import rollout as MODULE

ROOT = Path(__file__).resolve().parents[2]

# A rollback target's runtime image must be digest-pinned (H-13): a mutable tag
# could be re-pointed between plan approval and apply. The sinks that consume the
# previous release's runtime image reject anything that is not `...@sha256:<hex>`.
PREVIOUS_RUNTIME_IMAGE = "registry.example/runtime@sha256:" + "e" * 64
PREVIOUS_INSTALLER_IMAGE = "registry.example/installer@sha256:" + "f" * 64


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


def test_rollback_cleanup_is_scoped_to_agent_components() -> None:
    target = SimpleNamespace(cluster_id="gpu-a")
    calls = []
    release = SimpleNamespace(
        config=SimpleNamespace(
            clusters=(target,),
            agent_config_digest="c" * 64,
            runtime_profile_version="profile-v2",
        ),
        release_id="candidate",
        node_wheel_sha="a" * 64,
        bundle_sha="b" * 64,
        node_template_sha="t" * 64,
        _cancel_active_installer_jobs=lambda item: calls.append(
            ("cancel-jobs", item.cluster_id)
        ),
        _fleet_deployment_id=lambda *_args, **_kwargs: "upgrade-fleet",
        _fleet_command=lambda operation, payload: calls.append((operation, payload))
        or {"status": "CANCELLED"},
    )
    compensation = ORCHESTRATION.RollbackCompensationPlan(
        global_components=frozenset(),
        cluster_components={"gpu-a": frozenset({DIFF.ReleaseComponent.AGENT})},
        conservative=False,
    )

    ORCHESTRATION.cleanup_candidate_rollout_state(release, compensation)

    assert calls == [
        ("cancel-jobs", "gpu-a"),
        (
            "cancel-if-present",
            {
                "deployment_id": "upgrade-fleet",
                "reason": "release candidate was rolled back after verification",
            },
        ),
        (
            "terminalize-release-rollouts",
            {
                "release_id": "candidate",
                "reason": (
                    "release candidate was rolled back; the rollout was "
                    "superseded before reaching a terminal state"
                ),
            },
        ),
    ]


def test_executor_only_rollback_does_not_cancel_agent_resources() -> None:
    """An Executor-only rollback still has to sweep stranded rollouts.

    Scoping the *targeted* cancels to the compensation plan is right. Scoping
    the sweep to it is what stranded a PLANNED fleet deployment for 34 hours:
    the plan is built from recorded component progress, so an upgrade that
    created the record and then died before its progress write produces a plan
    that claims no AGENT component at all.
    """

    target = SimpleNamespace(cluster_id="gpu-a")
    operations = []
    release = SimpleNamespace(
        config=SimpleNamespace(clusters=(target,)),
        release_id="candidate",
        _cancel_active_installer_jobs=lambda _target: pytest.fail(
            "Executor-only rollback cancelled Installer Jobs"
        ),
        _fleet_command=lambda operation, payload: operations.append(
            (operation, payload)
        )
        or {"terminalized": []},
    )
    compensation = ORCHESTRATION.RollbackCompensationPlan(
        global_components=frozenset(),
        cluster_components={"gpu-a": frozenset({DIFF.ReleaseComponent.EXECUTOR})},
        conservative=False,
    )

    ORCHESTRATION.cleanup_candidate_rollout_state(release, compensation)

    assert [operation for operation, _payload in operations] == [
        "terminalize-release-rollouts"
    ], operations
    assert "cancel-if-present" not in {
        operation for operation, _payload in operations
    }, "an Executor-only rollback cancelled an Agent fleet deployment"


def test_the_stranded_rollout_sweep_is_announced_and_returned() -> None:
    """The 34-hour fence was invisible; a sweep that fires must say so."""

    release = SimpleNamespace(
        release_id="2e76ec8bdcda",
        _fleet_command=lambda _operation, _payload: {
            "terminalized": ["release-upgrade-2e76ec8bdcda-73663d3ca08df03d03ed"]
        },
    )

    terminalized = ORCHESTRATION.terminalize_stranded_rollouts(release)

    assert terminalized == ("release-upgrade-2e76ec8bdcda-73663d3ca08df03d03ed",)


def test_cpu_only_rollback_identity_does_not_require_agent_snapshots() -> None:
    release = SimpleNamespace(
        config=SimpleNamespace(clusters=(SimpleNamespace(cluster_id="gpu-a"),)),
        runtime_image="candidate-runtime",
    )
    compensation = ORCHESTRATION.RollbackCompensationPlan(
        global_components=frozenset({DIFF.ReleaseComponent.CPU_FINALIZE}),
        cluster_components={"gpu-a": frozenset()},
        conservative=False,
    )
    previous = {
        "cpu_wheel": "previous-cpu-wheel",
        "runtime_image": PREVIOUS_RUNTIME_IMAGE,
        "metadata": {
            "required-agent-artifact-sha256": "a" * 64,
            "required-agent-config-digest": "b" * 64,
        },
        "agent_identities": {},
    }

    result = ROLLBACK_CONTEXT.rollback_identity_context(release, previous, compensation)

    assert result[1] == "previous-cpu-wheel"
    assert result[-1] == PREVIOUS_RUNTIME_IMAGE


def test_schema_job_manifest_change_keeps_automatic_rollback_available() -> None:
    class ValidationPassed(RuntimeError):
        pass

    release = SimpleNamespace(
        config=SimpleNamespace(auto_rollback=True, clusters=()),
        state={"release_diff": {"changed": ["schema_manifests"]}},
        _ensure_contexts=lambda: None,
        _require_cpu_secrets=lambda: None,
        _apply_rds_ca_bundle=lambda: None,
        _refresh_aurora_credentials=lambda: None,
        _require_no_inflight_installs=lambda **_kwargs: None,
        _remote_commands_are_idle=lambda: True,
        _capture_previous=lambda **_kwargs: (_ for _ in ()).throw(ValidationPassed()),
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


def test_database_schema_change_still_requires_rollback_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This is the *unaccepted* path. The release gate's pytest inherits the
    # deploy's environment, and ``deploy --accept-schema-change`` exports
    # GPU_FAULT_RELEASE_ACCEPT_SCHEMA_CHANGE there (deploy #18, 2026-09-09):
    # with it set the engine took the accepted branch against this fake config.
    monkeypatch.delenv("GPU_FAULT_RELEASE_ACCEPT_SCHEMA_CHANGE", raising=False)
    release = SimpleNamespace(
        config=SimpleNamespace(auto_rollback=True),
        state={"release_diff": {"changed": ["database_schema"]}},
        _ensure_contexts=lambda: None,
        _require_cpu_secrets=lambda: None,
        _apply_rds_ca_bundle=lambda: None,
        _refresh_aurora_credentials=lambda: None,
        _require_no_inflight_installs=lambda **_kwargs: None,
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
        _config_maps_data=lambda names: {
            name: (
                {"GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": "profile-v1"}
                if name == "gpu-fault-api-ha-config-core"
                else {}
            )
            for name in names
        },
        _deployment_template_name=lambda _target: "installer-template",
        _deployment_wheel=lambda _args, name: f"{name}-wheel",
        _template_bundle=lambda _target, _template: "node-bundle",
        _config_map_binary_key=lambda _args, name: f"{name}-key",
        _config_map_sha=lambda *_args: "4" * 64,
        _capture_agent_identities=lambda: {target.cluster_id: _legacy_agent_identity()},
        _capture_observability_snapshot=lambda: {
            "rule_namespace": "rules",
            "rules_data_base64": "cnVsZXM=",
            "alertmanager_data_base64": "YWxlcnRz",
        },
        _remote_command_stats=lambda: {
            "executor_internal_error_total": 2,
            "executor_internal_error_last_seen_timestamp_seconds": 123.0,
        },
        _target_node_names=lambda _target: ("node-a",),
    )

    previous = STATE.capture_previous(release)

    assert previous["runtime_image"] == previous_runtime
    assert previous["cpu_wheel_sha256"] == "4" * 64
    assert previous["node_installer_image"] == previous_installer
    assert previous["adot_image"] == previous_adot
    assert previous["observability"]["rule_namespace"] == "rules"
    assert previous["runtime_image"] != release.runtime_image
    assert previous["release_delivery_sha256"] == "1" * 64
    assert previous["executor_internal_error_total"] == 2
    assert previous["executor_internal_error_last_seen_timestamp_seconds"] == 123.0
    assert previous["clusters"][target.cluster_id]["dcgm_image"] == previous_dcgm
    assert previous["agent_identities"][target.cluster_id]["agent_version"] == "0.10.0"


def test_deployment_snapshot_primes_list_reads_and_returns_independent_values() -> None:
    calls: list[tuple[str, ...]] = []
    target = SimpleNamespace(cluster_id="gpu-a")

    class Runner:
        @staticmethod
        def run(arguments, **_kwargs):
            calls.append(tuple(arguments))
            context = arguments[0]
            return json.dumps(
                {
                    "items": [
                        {
                            "metadata": {"name": f"{context}-deployment"},
                            "spec": {"replicas": 1},
                        }
                    ]
                }
            )

    def cpu_command(*args):
        return ["cpu", *args]

    def gpu_command(_target, *args):
        return ["gpu", *args]

    class SnapshotRelease:
        _deployment_snapshot_enabled = True
        runner = Runner()
        config = SimpleNamespace(namespace="gpu-fault-system", clusters=(target,))
        _cpu = staticmethod(cpu_command)
        _gpu = staticmethod(gpu_command)

        def _get_json(self, args):
            return STATE.get_json(self, args)

    release = SnapshotRelease()

    def get_json(args):
        return STATE.get_json(release, args)

    with STATE.read_snapshot(release):
        STATE.prime_deployment_snapshot(release)
        commands = (
            cpu_command(
                "-n", release.config.namespace, "get", "deployment", "cpu-deployment"
            ),
            gpu_command(
                target,
                "-n",
                release.config.namespace,
                "get",
                "deployment",
                "gpu-deployment",
            ),
        )
        with ThreadPoolExecutor(max_workers=2) as executor:
            first, second = [
                future.result()
                for future in (
                    executor.submit(get_json, commands[0]),
                    executor.submit(get_json, commands[1]),
                )
            ]
        first["spec"]["replicas"] = 9
        again = get_json(commands[0])

    assert first["metadata"]["name"] == "cpu-deployment"
    assert second["metadata"]["name"] == "gpu-deployment"
    assert again["spec"]["replicas"] == 1
    assert len(calls) == 2
    assert all(
        command[-4:] == ("get", "deployment", "-o", "json") for command in calls
    ), "deployment snapshot issued an unexpected Kubernetes read"


def test_nested_read_snapshot_joins_the_one_already_open() -> None:
    """`status` runs two report builders, and each opens a snapshot of its own.

    A nested block used to replace the cache with an empty one and restore it on
    exit, so the inner block re-read everything the outer block had already paid
    for and threw away everything it read itself.
    """

    calls: list[tuple[str, ...]] = []

    class Runner:
        @staticmethod
        def run(arguments, **_kwargs):
            calls.append(tuple(arguments))
            return json.dumps({"data": {"state.json": "{}"}})

    release = SimpleNamespace(runner=Runner())
    release._get_json = lambda args: STATE.get_json(release, args)
    command = ["cpu", "-n", "gpu-fault-system", "get", "configmap", "state"]

    with STATE.read_snapshot(release):
        release._get_json(command)
        with STATE.read_snapshot(release):
            release._get_json(command)
        # The outer block still holds its snapshot after the inner one exits.
        release._get_json(command)

    assert len(calls) == 1
    # And the snapshot really is closed at the end, rather than left behind for
    # whatever the administrator runs next.
    release._get_json(command)
    assert len(calls) == 2


def test_release_state_externalizes_and_hydrates_previous_snapshot() -> None:
    previous = {
        "release_id": "previous",
        "metadata": {"required-agent-artifact-sha256": "a" * 64},
    }
    release = SimpleNamespace(
        state={"phase": "failed", "previous": previous},
        runner=SimpleNamespace(dry_run=True),
    )

    persisted = STATE.persisted_state(release)
    reference = persisted["previous_snapshot"]
    encoded_reference, chunks = STATE.encode_previous_snapshot(previous)
    documents = {
        name: {"binaryData": {"snapshot.part": base64.b64encode(chunk).decode()}}
        for name, chunk in chunks
    }
    loaded_release = SimpleNamespace(
        state={},
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _cpu=lambda *args: list(args),
        _get_json=lambda arguments: (
            {"data": {"state.json": json.dumps(persisted)}}
            if arguments[-1] == STATE.STATE_CONFIG_MAP
            else documents[arguments[-1]]
        ),
    )

    loaded = STATE.load_state(loaded_release)

    assert "previous" not in persisted
    assert reference == encoded_reference
    assert loaded["previous"] == previous


def test_previous_snapshot_configmaps_are_immutable_and_content_addressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = {}
    calls = []

    class Runner:
        dry_run = False

        @staticmethod
        def run(arguments, **kwargs):
            calls.append((list(arguments), kwargs))
            if "--dry-run=client" in arguments:
                source = next(
                    value for value in arguments if value.startswith("--from-file=")
                )
                key, path = source.removeprefix("--from-file=").split("=", 1)
                return yaml.safe_dump(
                    {
                        "apiVersion": "v1",
                        "kind": "ConfigMap",
                        "metadata": {"name": arguments[4]},
                        "binaryData": {
                            key: base64.b64encode(Path(path).read_bytes()).decode()
                        },
                    }
                )
            if arguments[-3:] == ["create", "-f", "-"]:
                document = yaml.safe_load(kwargs["input_text"])
                documents[document["metadata"]["name"]] = document
            return ""

    def exists(arguments, **_kwargs):
        return arguments[-1] in documents

    runner = Runner()
    runner.probe = exists
    release = SimpleNamespace(
        runner=runner,
        config=SimpleNamespace(namespace="gpu-fault-system"),
        release_id="release-a",
        _cpu=lambda *args: list(args),
        _get_json=lambda arguments: documents[arguments[-1]],
    )
    previous = {"release_id": "previous", "metadata": {"key": "value"}}

    reference = STATE.ensure_previous_snapshot(release, previous)
    repeated = STATE.ensure_previous_snapshot(release, previous)

    assert repeated == reference
    assert len(documents) == len(reference["chunks"])
    assert all(document["immutable"] is True for document in documents.values()), (
        "previous snapshot ConfigMap was mutable"
    )
    assert all(
        document["metadata"]["labels"][STATE.PREVIOUS_SNAPSHOT_LABEL] == "true"
        for document in documents.values()
    ), "previous snapshot ConfigMap omitted its lifecycle label"
    creates = [call for call, _kwargs in calls if call[-3:] == ["create", "-f", "-"]]
    assert len(creates) == len(reference["chunks"])


def test_snapshot_cleanup_preserves_current_reference_and_newest_retained() -> None:
    calls = []

    def item(name: str, created: str) -> dict:
        return {"metadata": {"name": name, "creationTimestamp": created}}

    release = SimpleNamespace(
        state={
            "previous_snapshot": {
                "chunks": [{"config_map": "gpu-fault-release-previous-current-000"}]
            }
        },
        runner=SimpleNamespace(
            dry_run=False,
            run=lambda arguments, **_kwargs: calls.append(arguments) or "",
        ),
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _cpu=lambda *args: list(args),
        _get_json=lambda _arguments: {
            "items": [
                item("gpu-fault-release-previous-current-000", "2026-09-07T00:00:00Z"),
                item("gpu-fault-release-previous-recent-000", "2026-09-06T00:00:00Z"),
                item("gpu-fault-release-previous-older-000", "2026-09-05T00:00:00Z"),
                item("gpu-fault-release-previous-stale-000", "2026-09-01T00:00:00Z"),
            ]
        },
    )

    STATE.cleanup_previous_snapshots(release)

    assert calls == [
        [
            "-n",
            "gpu-fault-system",
            "delete",
            "configmap",
            "gpu-fault-release-previous-stale-000",
        ]
    ], f"only snapshots beyond the {STATE.PREVIOUS_SNAPSHOTS_RETAINED} retained go"


AMP_ROOTS = {
    "describe-rule-groups-namespace": "ruleGroupsNamespace",
    "describe-alert-manager-definition": "alertManagerDefinition",
}


def test_observability_snapshot_restores_amp_rules_and_alertmanager() -> None:
    calls = []
    release = SimpleNamespace(
        config=SimpleNamespace(
            namespace="gpu-fault-system",
            aws_region="us-west-2",
            health=SimpleNamespace(amp_workspace_id="ws-test"),
        ),
        runner=SimpleNamespace(
            run=lambda arguments, **_kwargs: calls.append(arguments) or "",
            # AMP reports both definitions ACTIVE before and after each put.
            probe_output=lambda arguments, **_kwargs: (
                0,
                json.dumps(
                    {AMP_ROOTS[arguments[2]]: {"status": {"statusCode": "ACTIVE"}}}
                ),
                "",
            ),
        ),
        _cpu=lambda *arguments: ["kubectl", *arguments],
    )

    OBSERVABILITY.restore_observability_snapshot(
        release,
        {
            "rule_namespace": "gpu-fault-rules",
            "rules_data_base64": "Z3JvdXBzOiBbXQo=",
            "alertmanager_data_base64": "YWxlcnRtYW5hZ2VyOiB7fQo=",
            "adot": {
                "namespace": "gpu-fault-system",
                "objects": [
                    {
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "metadata": {"name": "gpu-fault-adot"},
                    }
                ],
                "absent": [],
            },
        },
    )

    assert [arguments[2] for arguments in calls if arguments[0] == "aws"] == [
        "put-rule-groups-namespace",
        "put-alert-manager-definition",
    ]
    assert "--name" in calls[0]
    assert "gpu-fault-rules" in calls[0]


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


def test_cpu_only_previous_snapshot_skips_gpu_and_agent_inventory() -> None:
    target = SimpleNamespace(cluster_id="gpu-a")
    release = _snapshot_release(
        target=target,
        previous_runtime="registry.example/runtime:previous",
        previous_installer="registry.example/installer:previous",
        previous_adot="registry.example/adot:previous",
        previous_dcgm="registry.example/dcgm:previous",
        state={
            "release_id": "previous",
            "node_installer_image": "registry.example/installer:previous",
            "adot_image": "registry.example/adot:previous",
        },
    )
    plan = STATE.ReleaseExecutionPlan(
        nodes=(STATE.ReleaseComponent.CPU_FINALIZE, STATE.ReleaseComponent.VERIFY)
    )

    previous = STATE.capture_previous(release, plan)

    assert previous["clusters"] == {}
    assert previous["agent_identities"] == {}
    assert previous["observability"] is None


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


def test_legacy_current_nodes_skip_fleet_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = SimpleNamespace(cluster_id="gpu-a", hyperpod_cluster_name="hp-gpu-a")
    deploy_calls = []
    monkeypatch.setattr(
        NODE_RUNTIME_ROLLOUT,
        "ensure_pre_node_mutation_barrier",
        lambda *_args, **_kwargs: None,
    )
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

    identity = NODE_RUNTIME_ROLLOUT.roll_node_runtime(
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
        candidate_preflight_completed=True,
    )

    assert identity == ("bundle", "template")
    assert [call["allowed_node_names"] for call in deploy_calls] == [(), None]


def test_legacy_partial_rollback_omits_new_fleet_identity_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = SimpleNamespace(cluster_id="gpu-a", hyperpod_cluster_name="hp-gpu-a")
    fleet_requests = []
    fleet_identities = []
    waits = []
    commands = iter(
        [
            {"normalized": 0},
            {"terminalized": []},
            {
                "status": "PENDING",
                "waves": [["node-a"]],
                "nodes": [{"node_id": "node-a", "status": "PENDING"}],
            },
            {"node_ids": ["node-a"]},
            {"status": "SUCCEEDED"},
        ]
    )
    deploy_calls = []
    monkeypatch.setattr(
        NODE_RUNTIME_ROLLOUT, "ensure_rollout_wave_safe", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        FLEET_ROLLOUT, "ensure_rollout_wave_safe", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        NODE_RUNTIME_ROLLOUT,
        "ensure_pre_node_mutation_barrier",
        lambda *_args, **_kwargs: None,
    )
    wave_handoffs = []
    monkeypatch.setattr(
        FLEET_ROLLOUT,
        "hand_wave_to_reconciler",
        lambda _release, _target, context, wave: (
            wave_handoffs.append(wave) or context.paused_identity
        ),
    )
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

    identity = NODE_RUNTIME_ROLLOUT.roll_node_runtime(
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
        candidate_preflight_completed=True,
    )

    request = next(
        payload["request"]
        for operation, payload in fleet_requests
        if operation == "create"
    )
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
    assert wave_handoffs == [("node-a",)]
    assert [item["runtime_image"] for item in deploy_calls] == [
        "candidate-runtime",
        "previous-runtime",
    ]
    assert deploy_calls[-1]["template_config_map"] == "previous-template"


def _rollback_previous(*cluster_ids: str) -> dict:
    return {
        "metadata": {
            "required-agent-artifact-sha256": "artifact",
            "required-agent-config-digest": "config",
            "required-regional-executor-artifact-sha256": "executor",
        },
        "cpu_wheel": "wheel",
        "runtime_image": PREVIOUS_RUNTIME_IMAGE,
        "node_installer_image": PREVIOUS_INSTALLER_IMAGE,
        "runtime_profile_version": "profile-v1",
        "agent_identities": {
            cluster_id: _legacy_agent_identity()
            for cluster_id in (cluster_ids or ("gpu-a",))
        },
    }


def stub_rollback_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], list[tuple[str, ...]]]:
    """Replace every rollback phase body with a recorder of its own name.

    The second list records which clusters were already complete each time the
    data-plane phase ran, which is where the resume decisions land.
    """

    calls: list[str] = []
    replayed_from: list[tuple[str, ...]] = []

    def recorder(name: str):
        def phase(*_args, **kwargs) -> None:
            calls.append(name)
            if name == "_rollback_gpu_clusters":
                replayed_from.append(tuple(sorted(kwargs["completed_clusters"])))

        return phase

    for name in (
        "_stage_rollback_controller",
        "_rollback_gpu_clusters",
        "_restore_rollback_cpu",
        "_verify_and_complete_rollback",
    ):
        monkeypatch.setattr(ORCHESTRATION, name, recorder(name))
    monkeypatch.setattr(
        ORCHESTRATION,
        "cleanup_candidate_rollout_state",
        lambda *_args, **_kwargs: calls.append("cleanup"),
    )
    return calls, replayed_from


def rollback_release_fake(
    *,
    saved: list[str],
    cluster_ids: tuple[str, ...] = ("gpu-a",),
    state: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        state=dict(state or {}),
        runtime_image="candidate-runtime",
        node_installer_image="registry.example/installer:candidate",
        config=SimpleNamespace(
            clusters=tuple(
                SimpleNamespace(cluster_id=cluster_id) for cluster_id in cluster_ids
            )
        ),
        _refresh_aurora_credentials=lambda: None,
        _require_no_inflight_installs=lambda **_kwargs: None,
        _save_state=lambda phase, **_updates: saved.append(phase),
    )


def test_rollback_restores_gpu_before_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """The data plane must be back on the old release before the CPU role is.

    Restoring the CPU role first would leave the previous control plane driving
    GPU clusters that are still running the candidate Agent and Executor.
    """

    calls, _replayed_from = stub_rollback_phases(monkeypatch)
    saved: list[str] = []
    release = rollback_release_fake(saved=saved)

    ORCHESTRATION.rollback_release(release, state=_rollback_previous())

    assert calls == [
        "_stage_rollback_controller",
        "_rollback_gpu_clusters",
        "cleanup",
        "_restore_rollback_cpu",
        "_verify_and_complete_rollback",
    ]
    assert saved.index("rollback-controller-staged") < saved.index(
        "rollback-data-restored"
    )
    assert saved.index("rollback-data-restored") < saved.index("rollback-cpu-restored")
    assert saved[-1] == "rollback-restored"


def test_rollback_verifier_receives_previous_runtime_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every rollback check is told the old image, never the candidate's.

    The role-split verifier decides whether the control plane converged by
    comparing the live Deployment against ``GPU_FAULT_RUNTIME_IMAGE``. Handing it
    the candidate image would make a rollback that never happened pass.
    """

    checked: list[str] = []
    for name in ("_validate_cpu_rollback", "_validate_gpu_rollback"):
        monkeypatch.setattr(
            VALIDATION, name, lambda _release, _previous, image: checked.append(image)
        )
    verifier_envs: list[dict[str, str]] = []
    release = SimpleNamespace(
        runner=SimpleNamespace(
            dry_run=False,
            run=lambda _arguments, *, env=None, **_kwargs: verifier_envs.append(
                dict(env or {})
            ),
        ),
        runtime_image="candidate-runtime",
        config=SimpleNamespace(
            namespace="gpu-fault-system", cpu_kubeconfig="/nonexistent/cpu.kubeconfig"
        ),
        _config_map_data=lambda name: (
            {"GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": "profile-v1"}
            if name == "gpu-fault-api-ha-config-core"
            else {}
        ),
    )

    VALIDATION.validate_rollback(
        release,
        {
            "runtime_image": PREVIOUS_RUNTIME_IMAGE,
            "runtime_profile_version": "profile-v1",
        },
    )

    assert checked == [PREVIOUS_RUNTIME_IMAGE, PREVIOUS_RUNTIME_IMAGE]
    assert [env["GPU_FAULT_RUNTIME_IMAGE"] for env in verifier_envs] == [
        PREVIOUS_RUNTIME_IMAGE
    ]


def test_unverified_rollback_data_checkpoint_replays_only_mismatched_clusters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resumed rollback re-checks each finished cluster instead of trusting it.

    The data-plane checkpoint is written before verification, so a cluster that
    is still on the candidate release must lose its completion and be replayed,
    while the clusters that really are on the previous release are left alone.
    """

    drifted: set[str] = {"gpu-b"}

    def validate_target(_release, _previous, _runtime_image, target, **_kwargs) -> None:
        if target.cluster_id in drifted:
            raise ORCHESTRATION.ReleaseError("cluster is still on the candidate")

    monkeypatch.setattr(ORCHESTRATION, "validate_gpu_rollback_target", validate_target)
    calls, replayed_from = stub_rollback_phases(monkeypatch)
    resumed = {
        "phase": "rollback-data-restored",
        "rollback_completed_phases": [
            "rollback-started",
            "rollback-controller-staged",
            "rollback-data-restored",
        ],
        "rollback_completed_cluster_ids": ["gpu-a", "gpu-b"],
    }

    ORCHESTRATION.rollback_release(
        rollback_release_fake(saved=[], cluster_ids=("gpu-a", "gpu-b"), state=resumed),
        state=_rollback_previous("gpu-a", "gpu-b"),
    )

    assert replayed_from == [("gpu-a",)], (
        "the drifted cluster must be replayed and the converged one retained"
    )
    assert calls == [
        "_rollback_gpu_clusters",
        "cleanup",
        "_restore_rollback_cpu",
        "_verify_and_complete_rollback",
    ], "a phase that already completed must not be repeated on resume"

    drifted.clear()
    calls, replayed_from = stub_rollback_phases(monkeypatch)

    ORCHESTRATION.rollback_release(
        rollback_release_fake(saved=[], cluster_ids=("gpu-a", "gpu-b"), state=resumed),
        state=_rollback_previous("gpu-a", "gpu-b"),
    )

    assert replayed_from == [], "no cluster drifted, so nothing needs replaying"
    assert calls == [
        "cleanup",
        "_restore_rollback_cpu",
        "_verify_and_complete_rollback",
    ]


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
        _config_maps_data=lambda names: {
            name: (
                {"GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION": "profile-v1"}
                if name == "gpu-fault-api-ha-config-core"
                else {}
            )
            for name in names
        },
        _deployment_template_name=lambda _target: "installer-template",
        _deployment_wheel=lambda _args, name: f"{name}-wheel",
        _template_bundle=lambda _target, _template: "node-bundle",
        _config_map_binary_key=lambda _args, name: f"{name}-key",
        _config_map_sha=lambda *_args: "4" * 64,
        _capture_agent_identities=lambda: {target.cluster_id: _legacy_agent_identity()},
        _remote_command_stats=lambda: {
            "executor_internal_error_total": 0,
            "executor_internal_error_last_seen_timestamp_seconds": 0.0,
        },
        _target_node_names=lambda _target: ("node-a",),
    )


def _rolled_back_state(*, status: str) -> dict:
    return {
        "phase": "rolled-back",
        "rollback_result": {"status": status},
        "release_id": "candidate",
        "component_digests": {"executor": "candidate-executor"},
        "release_delivery_sha256": "candidate-delivery",
        "rendered_manifest_sha256": "candidate-rendered",
        "node_template_sha256": "candidate-template",
        "node_installer_image": "registry.example/installer:previous",
        "adot_image": "registry.example/adot:previous",
        "previous": {
            "release_id": "previous",
            "component_digests": {"executor": "previous-executor"},
            "release_delivery_sha256": "previous-delivery",
            "rendered_manifest_sha256": "previous-rendered",
            "node_template_sha256": "previous-template",
        },
    }


def _capture_after_rollback(state: dict) -> dict:
    release = _snapshot_release(
        target=SimpleNamespace(cluster_id="gpu-a"),
        previous_runtime="registry.example/runtime:previous",
        previous_installer="registry.example/installer:previous",
        previous_adot="registry.example/adot:previous",
        previous_dcgm="registry.example/dcgm:previous",
        state=state,
    )
    plan = STATE.ReleaseExecutionPlan(
        nodes=(STATE.ReleaseComponent.CPU_FINALIZE, STATE.ReleaseComponent.VERIFY)
    )
    return STATE.capture_previous(release, plan)


def test_snapshot_after_passed_rollback_describes_the_live_previous_release() -> None:
    """A rolled-back state names the failed candidate; live is `previous`.

    BOOT-020 stage 2 (live 2026-09-07) read the candidate's rendered-manifest
    digest out of the post-rollback state, so the next rollback target would
    have carried a rendering that was never live.
    """

    previous = _capture_after_rollback(_rolled_back_state(status="PASSED"))

    assert previous["release_id"] == "previous"
    assert previous["component_digests"] == {"executor": "previous-executor"}
    assert previous["release_delivery_sha256"] == "previous-delivery"
    assert previous["rendered_manifest_sha256"] == "previous-rendered"
    assert previous["node_template_sha256"] == "previous-template"


def test_snapshot_after_failed_rollback_keeps_the_recorded_state() -> None:
    """Only a PASSED rollback proves `previous` is what runs; otherwise the
    state is left alone and the operator's recovery decides."""

    previous = _capture_after_rollback(_rolled_back_state(status="FAILED"))

    assert previous["release_id"] == "candidate"
    assert previous["rendered_manifest_sha256"] == "candidate-rendered"


def _fleet_id(state: dict) -> str:
    release = SimpleNamespace(release_id="cand", state=state)
    return FLEET_ROLLOUT.fleet_deployment_id(
        release,
        SimpleNamespace(cluster_id="gpu-a"),
        phase="upgrade",
        artifact_sha="a" * 64,
        bundle_sha="b" * 64,
        template_sha="c" * 64,
        config_digest="d" * 64,
        runtime_profile_version="profile-v1",
    )


def test_fleet_deployment_id_is_unique_per_upgrade_transaction() -> None:
    """A candidate re-applied after its own rollback must start a fresh fleet
    rollout: the identity-derived id would hand it the FAILED record the
    rollback left behind (live 2026-09-07, BOOT-020 agent stage)."""

    first = _fleet_id({"fleet_rollout_transaction": "1111aaaa2222"})
    second = _fleet_id({"fleet_rollout_transaction": "3333bbbb4444"})
    resumed = _fleet_id({"fleet_rollout_transaction": "1111aaaa2222"})

    assert first != second, "a new transaction must not reuse the old record"
    assert first == resumed, "a resumed transaction must find its own record"
    assert first.startswith("release-upgrade-cand-"), first


def test_fleet_deployment_id_without_nonce_keeps_the_historical_shape() -> None:
    legacy = _fleet_id({})
    assert legacy == _fleet_id({"fleet_rollout_transaction": ""}), legacy
    assert legacy != _fleet_id({"fleet_rollout_transaction": "1111aaaa2222"}), legacy


def test_fleet_create_terminalizes_the_cluster_s_other_active_rollouts() -> None:
    """The release holding the lock is the cluster's only legitimate rollout, so
    right before it creates (or resumes) its own record every *other*
    non-terminal record is stranded -- whatever release id minted it -- and
    while it stands the destructive workflow fence holds the cluster. Fakes
    answering with nothing mean nothing terminalized, not an error."""
    calls = []
    release = SimpleNamespace(
        release_id="candidate",
        _fleet_command=lambda operation, payload: calls.append((operation, payload))
        or {"terminalized": ["release-upgrade-older-aaaa"]},
    )

    terminalized = ROLLOUT_CLEANUP.terminalize_stranded_cluster_rollouts(
        release,
        cluster_id="gpu-a",
        keep_deployment_id="release-upgrade-candidate-bbbb",
        reason="superseded",
    )

    assert terminalized == ("release-upgrade-older-aaaa",)
    assert calls == [
        (
            "terminalize-cluster-rollouts",
            {
                "cluster_id": "gpu-a",
                "keep_deployment_id": "release-upgrade-candidate-bbbb",
                "reason": "superseded",
            },
        )
    ]
    quiet = SimpleNamespace(release_id="c", _fleet_command=lambda *_a, **_k: None)
    assert (
        ROLLOUT_CLEANUP.terminalize_stranded_cluster_rollouts(
            quiet, cluster_id="gpu-a", keep_deployment_id="x", reason="r"
        )
        == ()
    )
