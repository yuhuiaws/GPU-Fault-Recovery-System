from __future__ import annotations

import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin.config import default_admin_config
from gpu_fault_release import regional_deployment_inventory as inventory
from gpu_fault_release import regional_release_state as state
from gpu_fault_release import rollout
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_diff import ReleaseComponent as Component
from gpu_fault_release.regional_release_diff import ReleaseExecutionPlan
from tests.regional._cov95_release_support import (
    NEW_IMAGE,
    OLD_IMAGE,
    RecordingRunner,
    ResourceRelease,
    deployment,
    json_response,
    previous_snapshot,
)
from tests.regional._release_orchestrator_support import config_file


@pytest.mark.parametrize("fail_after", [False, True])
def test_cached_read_publishes_failures_without_refetching_or_partial_success(
    fail_after: bool,
) -> None:
    cache = {}
    lock = threading.Lock()
    fetched = []

    def fetch() -> dict[str, Any]:
        fetched.append(True)
        if not fail_after:
            raise ReleaseError("read unavailable")
        return {"items": []}

    def after(_value: dict[str, Any]) -> None:
        raise ReleaseError("read unavailable")

    for _attempt in range(2):
        with pytest.raises(ReleaseError, match="read unavailable"):
            state.cached_read(cache, lock, ("key",), fetch, after=after)
    assert fetched == [True]


def test_state_transaction_is_reentrant_and_serializes_read_modify_write() -> None:
    release = ResourceRelease()
    release.state = {"counter": 0}

    def increment(_number: int) -> None:
        with state.state_transaction(release), state.state_transaction(release):
            release.state["counter"] += 1

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(increment, range(40)))
    assert release.state == {"counter": 40}
    with pytest.raises(ValueError), state.state_transaction(release):
        raise ValueError("interrupted")
    increment(0)
    assert release.state["counter"] == 41


def test_snapshot_cache_primes_named_items_but_does_not_cache_across_scopes() -> None:
    release = ResourceRelease()
    count = 0

    def fetch(_args: list[str], _kwargs: dict[str, Any]) -> str:
        nonlocal count
        count += 1
        return json.dumps(
            {
                "items": [
                    None,
                    {},
                    {"metadata": {"name": ""}},
                    {"metadata": {"name": "owned"}, "read": count},
                    {"metadata": {"name": "owned"}, "read": 999},
                ]
            }
        )

    release.runner.handler = fetch
    command = ["kubectl", "get", "deployment"]
    with state.read_snapshot(release):
        state.get_json(release, command)
        with state.read_snapshot(release):
            first = state.get_json(release, [*command, "owned"])
            assert first["read"] == 1
            first["read"] = -1
            assert state.get_json(release, [*command, "owned"])["read"] == 1
        assert count == 1
    state.get_json(release, command)
    assert count == 2


def test_aws_snapshot_caches_only_explicit_readonly_requests() -> None:
    release = ResourceRelease()
    release.runner.handler = json_response({"value": 1})
    with state.read_snapshot(release):
        for _attempt in range(2):
            assert state.aws_json(release, ["ec2", "describe-subnets"]) == {"value": 1}
        state.aws_json(release, ["ec2", "describe-subnets"], cached=False)
        state.aws_json(
            release, ["sts", "get-caller-identity"], region=False, sensitive=True
        )
    assert len(release.runner.calls) == 3
    assert "--region" not in release.runner.calls[-1][0]
    assert release.runner.calls[-1][1]["sensitive"] is True
    assert state.aws_read_only(["ec2"]) is False
    assert state.aws_read_only(["ec2", "modify-subnet-attribute"]) is False


@pytest.mark.parametrize(
    "volumes,expected",
    [
        ([], None),
        ([{"name": "other"}], None),
        ([{"name": "artifact", "configMap": {"name": "wheel"}}], "wheel"),
        ([{"name": "artifact"}], None),
    ],
)
def test_deployment_wheel_observes_only_artifact_volume(
    volumes: list[Any], expected: str | None
) -> None:
    release = ResourceRelease()
    item = deployment("app")
    item["spec"]["template"]["spec"]["volumes"] = volumes
    release.documents[("cpu", "deployment", "app")] = item
    assert state.deployment_wheel(release, ["kubectl"], "app") == expected


@pytest.mark.parametrize(
    "name,value,expected",
    [("wanted", "  old  ", "old"), ("wanted", "", None), ("other", "old", None)],
)
def test_deployment_environment_reads_only_named_literal(
    name: str, value: str, expected: str | None
) -> None:
    release = ResourceRelease()
    item = deployment("app")
    item["spec"]["template"]["spec"]["containers"].insert(
        0, {"name": "unrelated", "env": []}
    )
    item["spec"]["template"]["spec"]["containers"][1]["env"] = [
        {"name": "ignored", "value": "x"},
        {"name": name, "value": value},
    ]
    release.documents[("cpu", "deployment", "app")] = item
    assert state.deployment_env_value(release, ["kubectl"], "app", "wanted") == expected
    assert (
        state.deployment_image(release, ["kubectl"], "app", container_name="absent")
        is None
    )
    assert (
        state.deployment_image(release, ["kubectl"], "app", container_name="app")
        == NEW_IMAGE
    )


@pytest.mark.parametrize(
    "matching,image,expected",
    [(True, " image:v1 ", "image:v1"), (True, "", None), (False, "image:v1", None)],
)
def test_template_container_image_is_selected_from_yaml_documents(
    matching: bool, image: str, expected: str | None
) -> None:
    text = yaml.safe_dump_all(
        [
            None,
            {"kind": "ConfigMap"},
            {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {"name": "sidecar", "image": "ignore"},
                                {
                                    "name": "wanted" if matching else "other",
                                    "image": image,
                                },
                            ]
                        }
                    }
                }
            },
        ]
    )
    assert state.template_container_image(text, container_name="wanted") == expected


@pytest.mark.parametrize(
    "keys,expected",
    [
        ({}, None),
        ({"one": "example"}, "one"),
        ({"one": "example", "two": "example"}, None),
    ],
)
def test_binary_artifact_key_must_be_unique(
    keys: dict[str, str], expected: str | None
) -> None:
    release = ResourceRelease()
    assert state.config_map_binary_key(release, ["kubectl"], None) is None
    assert release.reads == []
    release.documents[("cpu", "configmap", "artifact")] = {"binaryData": keys}
    assert state.config_map_binary_key(release, ["kubectl"], "artifact") == expected


def cpu_documents() -> dict[str, Any]:
    return {name: deployment(name) for name in inventory.CPU_RUNTIME_DEPLOYMENTS}


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("name", "without a name"),
        ("empty", "has no containers"),
        ("literal", "sensitive-looking"),
    ],
)
def test_cpu_environment_capture_refuses_incomplete_or_sensitive_snapshot(
    fault: str, problem: str
) -> None:
    documents = cpu_documents()
    pod = documents[inventory.CPU_INGRESS_DEPLOYMENT]["spec"]["template"]["spec"]
    if fault == "name":
        pod["containers"][0]["name"] = ""
    elif fault == "empty":
        pod["containers"] = []
    else:
        pod["containers"][0]["env"] = [
            {"name": "GPU_FAULT_EXECUTION_TOKEN", "value": "example-only"}
        ]
    with pytest.raises(ReleaseError, match=problem):
        state.cpu_role_container_env(ResourceRelease(), documents)


def test_cpu_environment_capture_preserves_init_and_main_refs_without_aliasing() -> (
    None
):
    documents = cpu_documents()
    pod = documents[inventory.CPU_INGRESS_DEPLOYMENT]["spec"]["template"]["spec"]
    pod["initContainers"] = [
        {"name": "init", "env": [{"name": "SAFE", "value": "old"}]}
    ]
    pod["containers"][0]["envFrom"] = [{"secretRef": {"name": "reference-only"}}]
    snapshot = state.cpu_role_container_env(ResourceRelease(), documents)
    assert snapshot[inventory.CPU_INGRESS_DEPLOYMENT]["init"]["env"] == [
        {"name": "SAFE", "value": "old"}
    ]
    snapshot[inventory.CPU_INGRESS_DEPLOYMENT]["init"]["env"][0]["value"] = "changed"
    assert pod["initContainers"][0]["env"][0]["value"] == "old"


def test_role_configmap_snapshot_ignores_unowned_refs_but_rejects_sensitive_keys() -> (
    None
):
    class CapturedRelease(ResourceRelease):
        def _config_maps_data(self, names):
            return {
                name: dict(self.documents[("cpu", "configmap", name)]["data"])
                for name in names
            }

    release = CapturedRelease()
    documents = cpu_documents()
    pod = documents[inventory.CPU_INGRESS_DEPLOYMENT]["spec"]["template"]["spec"]
    pod["containers"][0]["envFrom"] = [
        {"secretRef": {"name": "not-read"}},
        {"configMapRef": {"name": "customer-map"}},
        {"configMapRef": {"name": "gpu-fault-other"}},
        {"configMapRef": {"name": "gpu-fault-api-ha-config-core"}},
    ]
    release.documents[("cpu", "configmap", "gpu-fault-api-ha-config-core")] = {
        "data": {"SAFE": "old"}
    }
    assert state.cpu_role_config_maps(release, documents) == {
        "gpu-fault-api-ha-config-core": {"SAFE": "old"}
    }
    release.documents[("cpu", "configmap", "gpu-fault-api-ha-config-core")]["data"][
        "GPU_FAULT_PASSWORD"
    ] = "example-only"
    with pytest.raises(ReleaseError, match="sensitive-looking keys"):
        state.cpu_role_config_maps(release, documents)


@pytest.mark.parametrize("fault", ["spool-value", "worker-value", "count-value"])
def test_admin_config_capture_rejects_invalid_live_values(fault: str) -> None:
    release = ResourceRelease()
    release.documents[("cpu", "deployment", "gpu-fault-control-worker")] = deployment(
        "worker", replicas=3
    )
    release.documents[("cpu", "deployment", "gpu-fault-telemetry-spool-worker")] = (
        deployment("spool", replicas=0)
    )
    snapshots = {}
    if fault == "spool-value":
        snapshots["gpu-fault-api-ha-config-telemetry"] = {
            "GPU_FAULT_TELEMETRY_SPOOL": "unknown"
        }
    elif fault == "worker-value":
        release.documents[("cpu", "deployment", "gpu-fault-control-worker")]["spec"][
            "replicas"
        ] = "invalid"
    else:
        snapshots["gpu-fault-control-worker-config-core"] = {
            "GPU_FAULT_CAPACITY_MANAGED_NODE_COUNT": "invalid"
        }
    with pytest.raises(ReleaseError, match="cannot capture a valid live"):
        state.captured_admin_config(release, snapshots)


@pytest.mark.parametrize(
    "records,problem",
    [
        ([], "no active Agent identity"),
        (
            [
                {"cluster_id": "gpu-a", "node_id": "a", "artifact_sha256": "a"},
                {"cluster_id": "gpu-a", "node_id": "b", "artifact_sha256": "b"},
            ],
            "identities differ",
        ),
    ],
)
def test_agent_capture_needs_a_complete_consistent_identity(
    monkeypatch: pytest.MonkeyPatch, records: list[Any], problem: str
) -> None:
    monkeypatch.setattr(
        state, "exec_cpu_ingress_probe", lambda *_args, **_kwargs: json.dumps(records)
    )
    with pytest.raises(ReleaseError, match=problem):
        state.capture_agent_identities(ResourceRelease())


def test_remote_command_snapshot_is_readonly_and_sensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def probe(_release: Any, **kwargs: Any) -> str:
        calls.append(kwargs)
        return json.dumps({"by_status": {"WAITING": 2}})

    monkeypatch.setattr(state, "exec_cpu_ingress_probe", probe)
    assert state.remote_command_stats(ResourceRelease()) == {
        "by_status": {"WAITING": 2}
    }
    assert calls[0]["sensitive"] is True
    assert calls[0]["interactive"] is False


class SnapshotRunner(RecordingRunner):
    def __init__(self) -> None:
        super().__init__()
        self.objects: dict[str, dict[str, Any]] = {}
        self.probes: list[str] = []
        self.invalid_render = False
        self.handler = self.render

    def probe_output(
        self, arguments: list[str], **_kwargs: Any
    ) -> tuple[int, str, str]:
        name = arguments[arguments.index("get") + 2]
        self.probes.append(name)
        return 0, json.dumps(self.objects[name]) if name in self.objects else "", ""

    def render(self, arguments: list[str], kwargs: dict[str, Any]) -> str:
        if "--dry-run=client" in arguments:
            if self.invalid_render:
                return "[]"
            option = next(
                argument
                for argument in arguments
                if argument.startswith("--from-file=")
            )
            key, path = option.removeprefix("--from-file=").split("=", 1)
            return json.dumps(
                {
                    "apiVersion": "v1",
                    "kind": "ConfigMap",
                    "metadata": {
                        "name": arguments[arguments.index("configmap") + 1],
                        "namespace": "gpu-fault-system",
                    },
                    "binaryData": {
                        key: base64.b64encode(Path(path).read_bytes()).decode()
                    },
                }
            )
        value = yaml.safe_load(kwargs["input_text"])
        value["metadata"]["uid"] = "uid-" + value["metadata"]["name"]
        self.objects[value["metadata"]["name"]] = value
        return ""


def test_previous_snapshot_create_validate_reuse_and_hydrate() -> None:
    release = ResourceRelease()
    runner = SnapshotRunner()
    release.runner = runner
    previous = {"release_id": "previous", "metadata": {"pin": "example-pin"}}
    reference = state.ensure_previous_snapshot(release, previous)
    assert len(runner.calls) == 2 * len(reference["chunks"])
    assert all(item["immutable"] is True for item in runner.objects.values()), (
        "captured previous-state chunks must be immutable"
    )
    assert state.ensure_previous_snapshot(release, previous) == reference
    assert len(runner.probes) == len(reference["chunks"]), (
        "unchanged in-process snapshot should use its digest cache"
    )
    other = ResourceRelease()
    other.runner = runner
    assert state.ensure_previous_snapshot(other, previous) == reference
    assert len(runner.calls) == 2 * len(reference["chunks"]), (
        "verified immutable chunks need no rewrite"
    )
    release.state = {"release_id": "candidate", "previous": previous}
    stored, text = state.render_persisted_state(release)
    assert "previous" not in stored
    assert json.loads(text)["previous_snapshot"] == reference
    release.documents[("cpu", "configmap", state.STATE_CONFIG_MAP)] = {
        "data": {"state.json": text}
    }
    release.documents.update(
        {("cpu", "configmap", name): item for name, item in runner.objects.items()}
    )
    loaded = state.load_state(release)
    assert loaded["previous"] == previous


@pytest.mark.parametrize("fault", ["mutable", "digest", "binary", "render"])
def test_previous_snapshot_refuses_drift_or_invalid_render(fault: str) -> None:
    release = ResourceRelease()
    runner = SnapshotRunner()
    release.runner = runner
    previous = {"release_id": "old"}
    if fault == "render":
        runner.invalid_render = True
    else:
        original = ResourceRelease()
        original.runner = runner
        state.ensure_previous_snapshot(original, previous)
        item = next(iter(runner.objects.values()))
        if fault == "mutable":
            item["immutable"] = False
        elif fault == "digest":
            item["metadata"]["annotations"][
                state.PREVIOUS_SNAPSHOT_DIGEST_ANNOTATION
            ] = "foreign"
        else:
            item["binaryData"] = {
                key: base64.b64encode(b"corrupt").decode() for key in item["binaryData"]
            }
    before = len(runner.calls)
    with pytest.raises(ReleaseError, match="identity changed|snapshot|cannot render"):
        state.ensure_previous_snapshot(release, previous)
    assert len(runner.calls) == before + int(fault == "render")


@pytest.mark.parametrize("raw", [None, "{", "[]"])
def test_state_load_failure_does_not_replace_the_last_known_state(
    raw: str | None,
) -> None:
    release = ResourceRelease()
    release.state = {"release_id": "last-known"}
    release.documents[("cpu", "configmap", state.STATE_CONFIG_MAP)] = {
        "data": {"state.json": raw}
    }
    with pytest.raises(ReleaseError, match="state is missing|state is invalid"):
        state.load_state(release)
    assert release.state == {"release_id": "last-known"}


def test_snapshot_retention_keeps_current_and_newest_groups() -> None:
    release = ResourceRelease()
    release.runner.handler = json_response({})
    release.state = {
        "previous_snapshot": {"chunks": [{"config_map": "snapshot-old-0"}, None, {}]}
    }
    items = [{"metadata": {}}]
    for index in range(5):
        items.append(
            {
                "metadata": {
                    "name": f"snapshot-{index}-0",
                    "creationTimestamp": f"2026-01-0{index + 1}T00:00:00Z",
                }
            }
        )
    items.extend(
        [
            {
                "metadata": {
                    "name": "snapshot-old-0",
                    "creationTimestamp": "2020-01-01T00:00:00Z",
                }
            },
            {
                "metadata": {
                    "name": "snapshot-old-1",
                    "creationTimestamp": "2020-01-01T00:00:01Z",
                }
            },
        ]
    )
    release.documents[("cpu", "configmap", "")] = {"items": items}
    state.cleanup_previous_snapshots(release)
    assert release.runner.calls[0][0][-3:] == [
        "snapshot-0-0",
        "snapshot-1-0",
        "snapshot-2-0",
    ]
    release.runner.calls.clear()
    release.documents[("cpu", "configmap", "")] = {"items": items[-2:]}
    state.cleanup_previous_snapshots(release)
    assert release.runner.calls == []
    release.runner.dry_run = True
    state.cleanup_previous_snapshots(release)
    assert release.runner.calls == []


def test_persisted_state_drops_stale_snapshot_refs_and_rejects_oversize_payload() -> (
    None
):
    release = ResourceRelease()
    release.state = {
        "previous": None,
        "previous_snapshot": {},
        "previous_snapshot_sha256": "stale",
    }
    assert state.persisted_state(release) == {"previous": None}
    assert release.state == {"previous": None}
    release.state = {"payload": "x" * (state.MAX_RELEASE_STATE_BYTES + 1)}
    with pytest.raises(ReleaseError, match="bounded ConfigMap"):
        state.render_persisted_state(release)


@pytest.mark.parametrize(
    "volumes,expected",
    [
        ([], None),
        ([{"name": "unrelated"}], None),
        (
            [{"name": "installer-template", "configMap": {"name": "template"}}],
            "template",
        ),
    ],
)
def test_reconciler_template_name_is_taken_only_from_named_volume(
    volumes: list[Any], expected: str | None
) -> None:
    release = ResourceRelease()
    release.documents[("gpu-a", "deployment", inventory.GPU_RECONCILER_DEPLOYMENT)][
        "spec"
    ]["template"]["spec"]["volumes"] = volumes
    assert (
        state.deployment_template_name(release, release.config.clusters[0]) == expected
    )


@pytest.mark.parametrize(
    "job,expected",
    [
        (None, None),
        ("---\nnull\n---\nkind: Service\n", None),
        (
            yaml.safe_dump(
                {
                    "spec": {
                        "template": {
                            "spec": {
                                "volumes": [
                                    {"name": "other"},
                                    {
                                        "name": "installer",
                                        "configMap": {"name": "bundle"},
                                    },
                                ]
                            }
                        }
                    }
                }
            ),
            "bundle",
        ),
    ],
)
def test_template_bundle_reads_structured_job_volume(
    job: str | None, expected: str | None
) -> None:
    release = ResourceRelease()
    release.documents[("gpu-a", "configmap", "template")] = {"data": {"job.yaml": job}}
    assert (
        state.template_bundle(release, release.config.clusters[0], "template")
        == expected
    )


def test_checkpoint_persists_candidate_identity_before_recording_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = RecordingRunner(json_response({}))
    release = rollout.RegionalRelease(
        rollout.ReleaseConfig.load(config_file(tmp_path)), runner
    )
    release.state = {"release_id": "different", "adopted_live_runtime_image": OLD_IMAGE}
    history = []

    def record(_release: Any, *, phase: str, state_text: str) -> None:
        assert len(runner.calls) == 1, "history must be recorded after persistence"
        history.append((phase, json.loads(state_text)))

    monkeypatch.setattr(state, "record_release_history", record)
    state.save_state(release, "cpu-staged", completed_phases=["cpu-staged"])
    written = json.loads(runner.calls[0][1]["input_text"])
    payload = json.loads(written["data"]["state.json"])
    assert "adopted_live_runtime_image" not in payload
    assert payload["release_id"] == release.release_id
    assert payload["completed_phases"] == ["cpu-staged"]
    assert history == [("cpu-staged", payload)]


@pytest.mark.parametrize("fault", [None, "capture", "node-set"])
def test_previous_capture_joins_cluster_reads_and_binds_agent_node_set(
    monkeypatch: pytest.MonkeyPatch, fault: str | None
) -> None:
    release = ResourceRelease(("gpu-a", "gpu-b"))
    release.state = {
        "release_id": "old",
        "runtime_image": NEW_IMAGE,
        "admin_config": default_admin_config().as_dict(),
    }
    release.node_installer_image = OLD_IMAGE
    release.adot_image = OLD_IMAGE
    for name in inventory.CPU_RUNTIME_DEPLOYMENTS:
        release.documents[("cpu", "deployment", name)] = deployment(name)
    release.documents[("cpu", "configmap", "gpu-fault-api-ha-config-core")] = {
        "data": {}
    }
    monkeypatch.setattr(state, "remote_command_stats", lambda _release: {})
    monkeypatch.setattr(
        state, "capture_previous_monitoring", lambda *_args, **_kwargs: ({}, OLD_IMAGE)
    )
    identities = previous_snapshot(release)["agent_identities"]
    if fault == "node-set":
        identities["gpu-a"]["node_ids"] = ["other"]
    monkeypatch.setattr(state, "capture_agent_identities", lambda _release: identities)
    barrier = threading.Barrier(2)
    observed = []

    def capture(_release: Any, target: Any, **_kwargs: Any) -> tuple[Any, Any, str]:
        barrier.wait(timeout=5)
        observed.append(target.cluster_id)
        if fault == "capture" and target.cluster_id == "gpu-a":
            raise ReleaseError("unreadable")
        return {}, {target.cluster_id: NEW_IMAGE}, OLD_IMAGE

    monkeypatch.setattr(state, "capture_gpu_cluster_snapshot", capture)
    plan = ReleaseExecutionPlan(nodes=(Component.EXECUTOR,))
    if fault:
        with pytest.raises(
            ReleaseError,
            match="previous-state capture failed|active Agent set does not match",
        ):
            state.capture_previous(release, plan)
    else:
        result = state.capture_previous(release, plan)
        assert result["release_id"] == "old"
        assert result["agent_identities"] == identities
    assert sorted(observed) == ["gpu-a", "gpu-b"]
