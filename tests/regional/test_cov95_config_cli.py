from __future__ import annotations

import copy
import io
import json
import runpy
import sys
from pathlib import Path

import pytest
import yaml

from gpu_fault import config_cli as cli
from gpu_fault.container_env_snapshot import ROLE_DEPLOYMENTS, pod_container_env


def deployment(**changes):
    result = {
        "kind": "Deployment",
        "metadata": {"name": "unit-api"},
        "spec": {
            "replicas": 1,
            "template": {
                "spec": {
                    "terminationGracePeriodSeconds": 120,
                    "containers": [
                        {"name": "api", "args": ["uvicorn app:app --no-proxy-headers"]}
                    ],
                }
            },
        },
    }
    result["spec"]["template"]["spec"]["containers"][0].update(changes)
    return result


@pytest.mark.parametrize(
    ("container", "maps", "fragment"),
    [
        ({"envFrom": [{"configMapRef": {"name": "missing"}}]}, {}, "ConfigMap missing"),
        (
            {
                "envFrom": [
                    {"configMapRef": {"name": "a"}},
                    {"configMapRef": {"name": "b"}},
                ]
            },
            {"a": {"X": "1"}, "b": {"X": "2"}},
            "duplicate envFrom",
        ),
        ({"env": [{"value": "unnamed"}]}, {}, "no name"),
        ({"env": [{"name": "X"}]}, {}, "exactly one"),
        ({"env": [{"name": "X", "value": "1", "valueFrom": {}}]}, {}, "exactly one"),
        (
            {
                "envFrom": [{"configMapRef": {"name": "a"}}],
                "env": [{"name": "X", "value": "2"}],
            },
            {"a": {"X": "1"}},
            "duplicates envFrom",
        ),
        (
            {"env": [{"name": "GPU_FAULT_EXECUTION_TOKEN", "value": "example-only"}]},
            {},
            "must use valueFrom",
        ),
    ],
)
def test_environment_conflicts_and_plaintext_credentials_are_rejected(
    container, maps, fragment
) -> None:
    _, errors = cli.resolve_environment(deployment(**container), maps)
    assert any(fragment in error for error in errors), (
        "invalid environment structure must produce a diagnostic"
    )


def test_resolved_config_values_and_secret_references_remain_distinct() -> None:
    value = deployment(
        envFrom=[{"secretRef": {"name": "private"}}],
        env=[
            {
                "name": "LITERAL",
                "valueFrom": {"configMapKeyRef": {"name": "public", "key": "x"}},
            },
            {
                "name": "SECRET",
                "valueFrom": {"secretKeyRef": {"name": "private", "key": "x"}},
            },
        ],
    )
    environment, errors = cli.resolve_environment(value, {"public": {"x": "value"}})
    assert (
        environment == {"LITERAL": "value", "SECRET": "<valueFrom>"} and not errors
    ), "validation resolves public configuration without reading Secret values"


def test_manifest_enumeration_accepts_directories_files_and_ignores_non_objects(
    tmp_path, monkeypatch
) -> None:
    directory = tmp_path / "deploy/control-plane/regional/generated"
    directory.mkdir(parents=True)
    (directory / "b.yaml").write_text("kind: ConfigMap\nmetadata: {name: example}\n")
    (directory / "a.yaml").write_text("null\n---\n- list\n")
    monkeypatch.chdir(tmp_path)
    assert [path.name for path in cli.manifest_paths([])] == ["a.yaml", "b.yaml"], (
        "default manifests are deterministic"
    )
    assert len(cli.documents(cli.manifest_paths([str(directory)]))) == 1, (
        "non-object YAML documents are not resources"
    )
    assert cli.manifest_paths([str(directory / "b.yaml")]) == [directory / "b.yaml"], (
        "explicit files are retained"
    )
    with pytest.raises(ValueError, match="does not exist"):
        cli.documents([tmp_path / "missing.yaml"])


def test_multiple_primary_containers_require_an_explicit_contract() -> None:
    value = deployment()
    value["spec"]["template"]["spec"]["containers"].append({"name": "other"})
    with pytest.raises(ValueError, match="one container"):
        cli.container(value)


@pytest.mark.parametrize(
    ("arguments", "environment", "fragment"),
    [
        ("uvicorn app:app", {}, "disable proxy"),
        ("uvicorn app:app --no-proxy-headers --proxy-headers", {}, "proxy-derived"),
        (
            "uvicorn app:app --no-proxy-headers --forwarded-allow-ips '*'",
            {},
            "proxy-derived",
        ),
        (
            "uvicorn app:app --no-proxy-headers",
            {"FORWARDED_ALLOW_IPS": "*"},
            "proxy-derived",
        ),
        ("worker", {"GPU_FAULT_SERVICE_ROLE": "worker"}, "missing processor"),
        (
            "ingress",
            {"GPU_FAULT_SERVICE_ROLE": "ingress", "GPU_FAULT_PROCESSOR_WORKERS": "4"},
            "contains processor",
        ),
        ("worker", {"GPU_FAULT_TELEMETRY_SPOOL": "invalid"}, "must be"),
        ("worker", {"GPU_FAULT_ENABLE_AGENT_REGISTRY": "true"}, "requires"),
    ],
)
def test_role_lint_rejects_identity_and_worker_configuration_drift(
    arguments, environment, fragment
) -> None:
    errors = cli.validate_role(deployment(args=[arguments]), environment)
    assert errors and any(fragment in error for error in errors), (
        "invalid role configuration must fail before rollout"
    )


def test_role_lint_accepts_complete_registry_pins_and_worker_pool() -> None:
    values = {name: "1" for name in cli.PROCESSOR_POOL_ENV}
    values.update(
        {
            "GPU_FAULT_SERVICE_ROLE": "worker",
            "GPU_FAULT_ENABLE_AGENT_REGISTRY": "true",
            "GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256": "a" * 64,
            "GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST": "b" * 64,
        }
    )
    assert cli.validate_role(deployment(), values) == [], (
        "valid pinned worker configuration stays supported"
    )


def test_shutdown_budget_covers_prestop_and_worker_request_completion() -> None:
    value = deployment(
        args=["uvicorn app:app --no-proxy-headers --timeout-graceful-shutdown 20"],
        lifecycle={"preStop": {"exec": {"command": ["sh", "-c", "sleep 10"]}}},
    )
    value["spec"]["template"]["spec"]["terminationGracePeriodSeconds"] = 25
    errors = cli.validate_shutdown(
        value,
        {
            "GPU_FAULT_SERVICE_ROLE": "worker",
            "GPU_FAULT_PROCESSOR_REQUEST_MAX_EXECUTION_SECONDS": "30",
            "GPU_FAULT_PROCESSOR_EXIT_GRACE_SECONDS": "5",
            "GPU_FAULT_LIFESPAN_SHUTDOWN_MAX_SECONDS": "10",
        },
    )
    assert len(errors) == 2, (
        "both Pod grace and in-process request grace must be sufficient"
    )
    assert cli.integer({"X": "<valueFrom>"}, "X", 9) == 9, (
        "unresolved optional values use the declared default"
    )


def test_operation_vocabularies_and_node_action_scope_are_checked() -> None:
    errors = cli.validate_operation_allowlists(
        {
            "control": {
                "GPU_FAULT_ALLOWED_OPERATIONS": "RESET_GPU,VERIFY_NO_GPU_CLIENTS"
            },
            "node": {
                "GPU_FAULT_NODE_ALLOWED_OPERATIONS": "RESET_GPU,FREEZE_EVIDENCE,unknown"
            },
        }
    )
    assert any("unknown values" in error for error in errors), (
        "unknown operations are not silently accepted"
    )
    assert any("VERIFY_NO_GPU_CLIENTS" in error for error in errors), (
        "required NodeAction operations cannot disappear"
    )
    assert any("disabled by the control plane" in error for error in errors), (
        "node scope cannot exceed control-plane authorization"
    )
    assert any("non-NodeAction" in error for error in errors), (
        "control-plane-only operations cannot run on the node"
    )


def test_connection_budget_includes_listener_connections() -> None:
    value = deployment(args=["uvicorn app:app --workers 2 --no-proxy-headers"])
    value["spec"]["replicas"] = 3
    errors = cli.validate_connections(
        [value],
        {
            "unit-api": {
                "GPU_FAULT_SERVICE_ROLE": "worker",
                "GPU_FAULT_POSTGRES_POOL_MAX_SIZE": "10",
                "GPU_FAULT_POSTGRES_FLEET_CONNECTION_BUDGET": "60",
            }
        },
    )
    assert errors and "ceiling 66" in errors[0], (
        "LISTEN sessions count in addition to the SQL pool"
    )


@pytest.mark.parametrize("ready", [False, True])
def test_readiness_cli_preserves_authentication_and_boolean_result(
    monkeypatch, capsys, ready
) -> None:
    body = {
        "cluster_id": "cluster/a?b",
        "ready": ready,
        "nodes": [{"node_id": "node-a", "ready": ready, "collectors": {}}],
    }

    def urlopen(request, *, timeout):
        assert (
            request.full_url
            == "https://control.invalid/v1/collector-readiness/cluster%2Fa%3Fb"
        ), "cluster identity is one encoded path component"
        assert request.get_header("X-gpu-fault-execution-token") == "example-only", (
            "readiness requires the selected credential"
        )
        assert timeout == 30, "readiness transport is bounded"
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(cli.urllib_request, "urlopen", urlopen)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "config",
            "collector-readiness",
            "--url",
            "https://control.invalid/",
            "--cluster-id",
            "cluster/a?b",
            "--execution-token",
            "example-only",
        ],
    )
    assert cli.main() == (0 if ready else 1), (
        "only explicit healthy state returns success"
    )
    assert json.loads(capsys.readouterr().out) == body, (
        "the report remains available to the caller"
    )


@pytest.mark.parametrize(
    "body",
    [
        [],
        {"ready": "false"},
        {"cluster_id": "other", "ready": True, "nodes": []},
        {"cluster_id": "a", "ready": True, "nodes": []},
        {"cluster_id": "a", "ready": True, "nodes": [{"node_id": "n", "ready": False}]},
    ],
)
def test_unknown_or_foreign_readiness_cannot_succeed(monkeypatch, capsys, body) -> None:
    monkeypatch.setattr(
        cli.urllib_request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(json.dumps(body).encode()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "config",
            "collector-readiness",
            "--url",
            "https://control.invalid",
            "--cluster-id",
            "a",
            "--execution-token",
            "example-only",
        ],
    )
    assert cli.main() == 2, (
        "malformed/foreign readiness is a read error, not a healthy report"
    )
    assert not capsys.readouterr().out, (
        "untrusted report contents are not printed as a valid result"
    )


@pytest.mark.parametrize("missing_token", [False, True])
def test_readiness_unavailable_or_unauthorized_is_a_controlled_error(
    monkeypatch, missing_token
) -> None:
    def unavailable(*args, **kwargs):
        assert not missing_token, "missing authentication must stop before transport"
        raise OSError("service unavailable")

    monkeypatch.delenv("GPU_FAULT_EXECUTION_TOKEN", raising=False)
    monkeypatch.setattr(cli.urllib_request, "urlopen", unavailable)
    arguments = [
        "config",
        "collector-readiness",
        "--url",
        "https://control.invalid",
        "--cluster-id",
        "a",
    ]
    if not missing_token:
        arguments.extend(["--execution-token", "example-only"])
    monkeypatch.setattr(sys, "argv", arguments)
    assert cli.main() == 2, "missing credentials and failed reads cannot pass readiness"


@pytest.mark.parametrize("structured", [False, True])
def test_cli_returns_invalid_for_missing_manifests(
    tmp_path, monkeypatch, capsys, structured
) -> None:
    arguments = ["config", "validate", str(tmp_path / "absent.yaml")]
    if structured:
        arguments.append("--json")
    monkeypatch.setattr(sys, "argv", arguments)
    assert cli.main() == 1, "file errors are validation failures"
    output = capsys.readouterr()
    if structured:
        assert json.loads(output.out)["valid"] is False, "JSON output preserves failure"
    else:
        assert "configuration invalid" in output.err, (
            "text output has a useful diagnostic"
        )


def test_cli_validates_a_real_minimal_manifest_and_node_scope(
    tmp_path, monkeypatch, capsys
) -> None:
    value = deployment(
        env=[{"name": "GPU_FAULT_ALLOWED_OPERATIONS", "value": "VERIFY_NO_GPU_CLIENTS"}]
    )
    path = tmp_path / "deployment.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["config", "validate", str(path), "--node-operations", "VERIFY_NO_GPU_CLIENTS"],
    )
    assert cli.main() == 0, "a valid minimal manifest is still supported"
    assert "configuration valid" in capsys.readouterr().out, (
        "the CLI reports validated resource counts"
    )
    monkeypatch.setattr(sys, "argv", ["config", "validate", str(path), "--json"])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(Path(cli.__file__)), run_name="__main__")
    assert error.value.code == 0, "module entrypoint shares the same validator"
    assert json.loads(capsys.readouterr().out)["valid"] is True, (
        "entrypoint JSON is a real validation result"
    )


def test_snapshot_cli_reports_the_previous_release_contract_in_text_mode(
    tmp_path, monkeypatch, capsys
) -> None:
    template = deployment(
        env=[{"name": "GPU_FAULT_ONLY_PREVIOUS_RELEASE", "value": "true"}]
    )
    snapshot = {}
    manifests = []
    for name in ROLE_DEPLOYMENTS:
        value = copy.deepcopy(template)
        value["metadata"]["name"] = name
        snapshot[name] = pod_container_env(value)
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        manifests.append(str(path))
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["config", "validate", *manifests, "--container-env-snapshot", str(previous)],
    )
    assert cli.main() == 0, (
        "verbatim previous configuration is judged against its own snapshot"
    )
    text = capsys.readouterr().out
    assert (
        "configuration validators skipped" in text and "configuration valid" in text
    ), "the changed validation basis is explicit in non-JSON output"
