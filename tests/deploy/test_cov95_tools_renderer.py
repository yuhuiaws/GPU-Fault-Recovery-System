from __future__ import annotations

import copy
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.container_env_snapshot import pod_container_env
from tests._script_loader import lazy_script_module
from tests.deploy._cov95_tools_support import (
    RENDERER,
    TOOLS,
    base_deployment,
    by_name,
    effective_env,
    render_documents,
)

FILTER = lazy_script_module(TOOLS / "filter_legacy_release_env.py")
ROLES = (
    ("gpu-fault-api-ha", "api", "ingress", 8080, 4),
    ("gpu-fault-control-worker", "control-worker", "worker", 8081, 4),
    (
        "gpu-fault-telemetry-spool-worker",
        "telemetry-spool-worker",
        "spool-worker",
        8082,
        1,
    ),
)


@pytest.mark.parametrize("as_json", [True, False])
def test_renderer_entrypoint_emits_isolated_roles_without_live_metadata(
    monkeypatch: pytest.MonkeyPatch, as_json: bool
) -> None:
    source = base_deployment()
    source["metadata"].update(
        uid="source-uid",
        resourceVersion="19",
        generation=7,
        creationTimestamp="2026-01-01T00:00:00Z",
        managedFields=[{}],
        ownerReferences=[{"uid": "source-owner"}],
    )
    source["status"] = {"readyReplicas": 3}
    source["spec"]["template"]["spec"]["enableServiceLinks"] = True
    documents = render_documents(monkeypatch, source=source, as_json=as_json)
    resources = by_name(documents)
    assert len(resources) == len(documents), "resource names must not collide"
    assert {item["kind"] for item in documents} == {
        "ConfigMap",
        "Deployment",
        "PodDisruptionBudget",
    }
    for name, container_name, role, port, workers in ROLES:
        deployment = resources[name]
        pod = deployment["spec"]["template"]["spec"]
        container = pod["containers"][0]
        env = effective_env(documents, name)
        assert "status" not in deployment
        assert not set(deployment["metadata"]) & {
            "uid",
            "resourceVersion",
            "generation",
            "creationTimestamp",
            "managedFields",
            "ownerReferences",
        }
        assert pod["enableServiceLinks"] is False
        assert [item["name"] for item in pod["containers"]] == [container_name]
        assert env["GPU_FAULT_SERVICE_ROLE"] == role
        assert env["GPU_FAULT_POSTGRES_AUTO_SCHEMA_INIT"] == "false"
        assert all("valueFrom" in item for item in container["env"]), (
            f"{role} retained literal env after ConfigMap externalization"
        )
        assert any(
            item.get("valueFrom", {}).get("secretKeyRef", {}).get("name")
            == "gpu-fault-aurora"
            for item in container["env"]
        ), "DSN must remain a Secret reference in every role"
        assert f"--port {port} " in container["args"][0]
        assert f"--workers {workers} " in container["args"][0]
        assert container["readinessProbe"]["httpGet"]["port"] == port
        assert "--limit-max-requests" not in container["args"][0]
        if role != "worker":
            assert not set(RENDERER.PROCESSOR_POOL_ENV) & env.keys()
        else:
            assert env["GPU_FAULT_PROCESSOR_COMPLETION_CLUSTER_CONCURRENCY"] == "1"
    for document in documents:
        if document["kind"] == "ConfigMap":
            assert all(isinstance(value, str) for value in document["data"].values()), (
                "rendered ConfigMap values must be Kubernetes-compatible strings"
            )
            assert list(document["data"]) == sorted(document["data"])
    assert resources["gpu-fault-api-ha-pdb"]["spec"]["minAvailable"] == 2
    assert resources["gpu-fault-control-worker-pdb"]["spec"]["maxUnavailable"] == 1
    assert (
        resources["gpu-fault-control-worker"]["metadata"]["annotations"][
            "gpu-fault.io/cleanup-phase"
        ]
        == "consumer"
    )
    assert source["metadata"]["uid"] == "source-uid", "render must not mutate its input"


def test_file_and_stream_rendering_match_and_prune_only_stale_role_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = by_name(render_documents(monkeypatch))
    stale = tmp_path / "gpu-fault-obsolete.yaml"
    stale.write_text("obsolete")
    unrelated = tmp_path / "customer.yaml"
    unrelated.write_text("customer-owned")
    actual = by_name(render_documents(monkeypatch, as_json=False, out_dir=tmp_path))
    assert actual == expected
    assert not stale.exists(), "rendering left an obsolete generated role manifest"
    assert unrelated.read_text() == "customer-owned"
    names = (tmp_path / "manifest-list.txt").read_text().splitlines()
    assert names == sorted(names)
    assert set(names) == {path.name for path in tmp_path.glob("gpu-fault-*.yaml")}


@pytest.mark.parametrize("enabled,replicas", [("true", "2"), ("false", "0")])
def test_renderer_applies_capacity_and_scoped_retention(
    monkeypatch: pytest.MonkeyPatch, enabled: str, replicas: str
) -> None:
    for name, value in {
        "GPU_FAULT_TELEMETRY_SPOOL": enabled,
        "GPU_FAULT_TELEMETRY_SPOOL_REPLICAS": replicas,
        "GPU_FAULT_CONTROL_WORKER_REPLICAS": "3",
        "GPU_FAULT_CAPACITY_LARGEST_CLUSTER_NODE_COUNT": "256",
        "GPU_FAULT_CAPACITY_MANAGED_NODE_COUNT": "512",
        "GPU_FAULT_WORKFLOW_POLL_INTERVAL_SECONDS": "2.5",
        "GPU_FAULT_CONTROL_RECORD_RETENTION_DAYS": " 7 ",
        "GPU_FAULT_CONTROL_RECORD_ARCHIVE_S3_URI": " s3://example/archive ",
        "GPU_FAULT_CONTROL_RECORD_ARCHIVE_INTERVAL_SECONDS": " 60 ",
        "GPU_FAULT_NOTIFICATION_CHANNEL": " sns ",
        "GPU_FAULT_SNS_TOPIC_ARN": " arn:aws:sns:us-east-1:123456789012:example ",
    }.items():
        monkeypatch.setenv(name, value)
    documents = render_documents(monkeypatch)
    resources = by_name(documents)
    worker = effective_env(documents, "gpu-fault-control-worker")
    ingress = effective_env(documents, "gpu-fault-api-ha")
    assert resources["gpu-fault-control-worker"]["spec"]["replicas"] == 3
    assert resources["gpu-fault-telemetry-spool-worker"]["spec"]["replicas"] == int(
        replicas
    )
    assert worker["GPU_FAULT_PROCESSOR_CONSUMER_PROCESSES"] == "12"
    assert worker["GPU_FAULT_PROCESSOR_NOTIFICATION_SHARDS"] == "12"
    assert worker["GPU_FAULT_CONTROL_RECORD_RETENTION_DAYS"] == "7"
    assert worker["GPU_FAULT_CONTROL_RECORD_ARCHIVE_S3_URI"] == "s3://example/archive"
    assert ingress["GPU_FAULT_TELEMETRY_SPOOL"] == enabled
    assert ingress["GPU_FAULT_POSTGRES_POOL_MAX_SIZE"] == (
        "48" if enabled == "true" else "40"
    )
    assert "GPU_FAULT_CONTROL_RECORD_RETENTION_DAYS" not in ingress
    for name, *_ in ROLES:
        env = effective_env(documents, name)
        assert env["GPU_FAULT_NOTIFICATION_CHANNEL"] == "sns"
        assert env["GPU_FAULT_WORKFLOW_POLL_INTERVAL_SECONDS"] == "2.5"
        assert env["GPU_FAULT_CAPACITY_MANAGED_NODE_COUNT"] == "512"


@pytest.mark.parametrize(
    "name,domain",
    [
        ("GPU_FAULT_STORE_URL_FILE", "postgres"),
        ("GPU_FAULT_PROCESSOR_WORKERS", "processor"),
        ("GPU_FAULT_EMAIL_RECIPIENTS", "notification"),
        ("GPU_FAULT_NVLINK_MAX_ERRORS", "telemetry"),
        ("GPU_FAULT_NODE_ACTION_URL", "recovery"),
        ("POD_UID", "core"),
    ],
)
def test_literal_configuration_uses_domain_maps(name: str, domain: str) -> None:
    assert RENDERER.config_domain(name) == domain
    container = {"env": [{"name": name, "value": 12}]}
    result = RENDERER.externalize_literal_env(
        container, role="role", namespace="cpu-only"
    )
    assert result == [
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": f"role-config-{domain}", "namespace": "cpu-only"},
            "data": {name: "12"},
        }
    ]
    assert container == {
        "env": [],
        "envFrom": [{"configMapRef": {"name": f"role-config-{domain}"}}],
    }


@pytest.mark.parametrize(
    "entry,problem",
    [
        ({}, "without a name"),
        ({"name": "GPU_FAULT_MODE"}, "exactly one"),
        ({"name": "GPU_FAULT_MODE", "value": "x", "valueFrom": {}}, "exactly one"),
        ({"name": "GPU_FAULT_EXECUTION_TOKEN", "value": "example-only"}, "valueFrom"),
    ],
)
def test_renderer_refuses_ambiguous_or_literal_sensitive_environment(
    monkeypatch: pytest.MonkeyPatch, entry: dict[str, Any], problem: str
) -> None:
    source = base_deployment()
    source["spec"]["template"]["spec"]["containers"][0]["env"].append(entry)
    with pytest.raises(ValueError, match=problem):
        render_documents(monkeypatch, source=source)


@pytest.mark.parametrize("args", [None, [], ["one", "two"], ["python unrelated.py"]])
def test_renderer_rejects_an_unrecognised_api_launcher(
    monkeypatch: pytest.MonkeyPatch, args: list[str] | None
) -> None:
    source = base_deployment()
    source["spec"]["template"]["spec"]["containers"][0]["args"] = args
    with pytest.raises(ValueError, match="API container"):
        render_documents(monkeypatch, source=source)


@pytest.mark.parametrize("raw", ["", "---\nnull\n", "kind: Service\n", "- []\n"])
def test_yaml_input_must_contain_the_named_deployment(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(raw))
    with pytest.raises(SystemExit, match="no gpu-fault-api-ha Deployment"):
        RENDERER.read_source(False)


def test_json_parse_failure_is_not_interpreted_as_an_empty_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("{"))
    with pytest.raises(json.JSONDecodeError):
        RENDERER.read_source(True)


def test_environment_helpers_preserve_probes_and_remove_absent_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = {
        "ports": [
            {"name": "http", "containerPort": 1},
            {"name": "metrics", "containerPort": 2},
        ],
        "startupProbe": {"tcpSocket": {"port": 1}},
        "livenessProbe": {"exec": {"command": ["true"]}},
    }
    RENDERER.set_probe_port(container, 8081)
    RENDERER.unset_env(container, "absent")
    assert container["ports"] == [
        {"name": "http", "containerPort": 8081},
        {"name": "metrics", "containerPort": 2},
    ]
    assert container["startupProbe"] == {"tcpSocket": {"port": 8081}}
    assert container["livenessProbe"] == {"exec": {"command": ["true"]}}
    for key in (
        *RENDERER.CONTROL_RECORD_RETENTION_ENV,
        *RENDERER.NOTIFICATION_CHANNEL_ENV,
    ):
        RENDERER.set_env(container, key, "old")
        monkeypatch.setenv(key, " ")
    RENDERER.configure_control_record_retention(container)
    RENDERER.configure_notification_channel(container)
    assert container["env"] == []


@pytest.mark.parametrize("value,expected", [(2.0, "2"), (2.5, "2.5"), (3, "3")])
def test_environment_numbers_have_stable_text(value: object, expected: str) -> None:
    assert RENDERER.environment_text(value) == expected


@pytest.mark.parametrize("value", ["yes", "1", "", "unknown"])
def test_non_boolean_spool_setting_fails_closed(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL", value)
    with pytest.raises(ValueError, match="must be true or false"):
        RENDERER.renderer_admin_config()


def snapshot_for(documents: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        item["metadata"]["name"]: pod_container_env(item)
        for item in documents
        if item["kind"] == "Deployment"
    }


def test_rollback_restores_all_container_environments_without_aliasing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = base_deployment()
    source["spec"]["template"]["spec"]["initContainers"] = [
        {"name": "init", "env": [{"name": "OLD_INIT", "value": "yes"}]}
    ]
    snapshot = snapshot_for(render_documents(monkeypatch, source=source))
    for containers in snapshot.values():
        for spec in containers.values():
            spec["env"] = [{"name": "PREVIOUS_RELEASE", "value": "old"}]
            spec["envFrom"] = [{"configMapRef": {"name": "previous-config"}}]
    path = tmp_path / "previous.json"
    path.write_text(json.dumps(snapshot))
    monkeypatch.setenv(RENDERER.CONTAINER_ENV_FILE_VARIABLE, str(path))
    actual = render_documents(monkeypatch, source=source)
    assert snapshot_for(actual) == snapshot
    deployments = [item for item in actual if item["kind"] == "Deployment"]
    RENDERER.restore_previous_container_env(deployments, snapshot)
    snapshot["gpu-fault-api-ha"]["api"]["env"].clear()
    assert pod_container_env(deployments[0])["api"]["env"] == [
        {"name": "PREVIOUS_RELEASE", "value": "old"}
    ]


@pytest.mark.parametrize(
    "fault",
    [
        "unknown-deployment",
        "missing-deployment",
        "unknown-container",
        "missing-container",
    ],
)
def test_rollback_rejects_incomplete_or_unknown_container_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str
) -> None:
    source = base_deployment()
    source["spec"]["template"]["spec"]["initContainers"] = [{"name": "init"}]
    snapshot = snapshot_for(render_documents(monkeypatch, source=source))
    if fault == "unknown-deployment":
        snapshot["other-deployment"] = copy.deepcopy(snapshot["gpu-fault-api-ha"])
    elif fault == "missing-deployment":
        del snapshot["gpu-fault-control-worker"]
    elif fault == "unknown-container":
        snapshot["gpu-fault-api-ha"]["other"] = {"env": [], "envFrom": []}
    else:
        del snapshot["gpu-fault-api-ha"]["api"]
    path = tmp_path / "previous.json"
    path.write_text(json.dumps(snapshot))
    monkeypatch.setenv(RENDERER.CONTAINER_ENV_FILE_VARIABLE, str(path))
    with pytest.raises(SystemExit, match="snapshot is invalid"):
        render_documents(monkeypatch, source=source)


@pytest.mark.parametrize("contents", [None, "{", "{}", "[]"])
def test_rollback_does_not_fall_back_to_candidate_on_read_or_shape_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, contents: str | None
) -> None:
    path = tmp_path / "previous.json"
    if contents is not None:
        path.write_text(contents)
    monkeypatch.setenv(RENDERER.CONTAINER_ENV_FILE_VARIABLE, str(path))
    with pytest.raises(SystemExit, match="snapshot is invalid"):
        render_documents(monkeypatch)


def test_legacy_filter_entrypoint_removes_only_component_pins(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = base_deployment()
    pod = source["spec"]["template"]["spec"]
    keep = {"name": "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256", "value": "old-pin"}
    entries = [
        keep,
        *({"name": key, "value": "new-pin"} for key in FILTER.COMPONENT_PIN_ENV),
    ]
    pod["containers"][0]["env"] = copy.deepcopy(entries)
    pod["initContainers"] = [
        {"name": "init", "env": copy.deepcopy(entries)},
        {"name": "empty"},
    ]
    documents = [
        None,
        ["literal"],
        {"kind": "ConfigMap", "data": {"keep": "yes"}},
        source,
    ]
    monkeypatch.setattr(sys, "stdin", io.StringIO(yaml.safe_dump_all(documents)))
    assert FILTER.main() == 0
    filtered = list(yaml.safe_load_all(capsys.readouterr().out))
    assert filtered[:3] == documents[:3]
    filtered_pod = filtered[3]["spec"]["template"]["spec"]
    assert filtered_pod["containers"][0]["env"] == [keep]
    assert filtered_pod["initContainers"] == [
        {"name": "init", "env": [keep]},
        {"name": "empty", "env": []},
    ]
