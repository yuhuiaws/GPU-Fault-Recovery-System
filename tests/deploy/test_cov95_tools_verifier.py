from __future__ import annotations

import importlib.util
import json
import subprocess
from types import ModuleType
from typing import Any

import pytest

from tests.deploy._cov95_tools_support import TOOLS, by_name, render_documents

INGRESS = "gpu-fault-api-ha"
WORKER = "gpu-fault-control-worker"
SPOOL = "gpu-fault-telemetry-spool-worker"
IMAGE = "registry.example/runtime@sha256:" + "a" * 64


class Kubectl:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.resources = {
            (item["kind"].lower(), item["metadata"]["name"]): item for item in documents
        }
        self.calls: list[tuple[str, str]] = []
        self.failure: tuple[int, str, str] | None = None

    def __call__(
        self, arguments: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        assert arguments[:2] == ["kubectl", "-n"]
        assert arguments[3] == "get"
        assert kwargs["capture_output"] is True
        names = []
        for argument in arguments[5:]:
            if argument.startswith("-"):
                break
            names.append(argument)
        keys = [(arguments[4], name) for name in names]
        self.calls.extend(keys)
        if self.failure is not None:
            code, output, error = self.failure
        else:
            assert "--ignore-not-found" in arguments, "batch must distinguish absence"
            code, output, error = (
                0,
                json.dumps(
                    {
                        "kind": "List",
                        "items": [
                            self.resources[key] for key in keys if key in self.resources
                        ],
                    }
                ),
                "",
            )
        return subprocess.CompletedProcess(arguments, code, stdout=output, stderr=error)


@pytest.fixture
def verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, Kubectl, dict[str, Any]]:
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL", "true")
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL_REPLICAS", "2")
    documents = render_documents(monkeypatch)
    documents.append(
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "gpu-fault-release-metadata"},
            "data": {
                "required-agent-artifact-sha256": "a" * 64,
                "required-agent-compatibility-digest": "b" * 64,
                "required-agent-protocol-version": "3",
                "required-regional-executor-protocol-version": "2",
                "required-agent-config-digest": "c" * 64,
                "required-node-action-key-version": "2",
            },
        }
    )
    for item in documents:
        if item["kind"] == "Deployment":
            item["status"] = {"readyReplicas": item["spec"]["replicas"]}
            item["spec"]["template"]["spec"]["containers"][0]["image"] = IMAGE
    spec = importlib.util.spec_from_file_location(
        "cov95_role_verifier", TOOLS / "verify_control_plane_role_split.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    transport = Kubectl(documents)
    monkeypatch.setattr(module.subprocess, "run", transport)
    monkeypatch.setenv("GPU_FAULT_RUNTIME_IMAGE", IMAGE)
    return module, transport, by_name(documents)


def set_value(resources: dict[str, Any], role: str, name: str, value: Any) -> None:
    container = resources[role]["spec"]["template"]["spec"]["containers"][0]
    for source in container["envFrom"]:
        resources[source["configMapRef"]["name"]]["data"].pop(name, None)
    container["env"] = [entry for entry in container["env"] if entry["name"] != name]
    if value is not None:
        container["env"].append({"name": name, "value": value})


def test_renderer_output_satisfies_public_verifier(
    verifier: tuple[Any, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    module, transport, _resources = verifier
    assert module.main() == 0
    assert "role-split check passed" in capsys.readouterr().out
    assert len(transport.calls) == len(set(transport.calls)), (
        "ConfigMaps should be read once"
    )


@pytest.mark.parametrize(
    "role,name,value,problem",
    [
        (
            INGRESS,
            "GPU_FAULT_SERVICE_ROLE",
            "worker",
            "not GPU_FAULT_SERVICE_ROLE=ingress",
        ),
        (INGRESS, "GPU_FAULT_TELEMETRY_SPOOL", "unknown", "must explicitly set"),
        (INGRESS, "GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "bad", "limits must be integers"),
        (INGRESS, "GPU_FAULT_POSTGRES_POOL_MAX_SIZE", "1", "pool does not cover"),
        (INGRESS, "GPU_FAULT_PROCESSOR_WORKERS", "24", "carries processor pool sizing"),
        (
            WORKER,
            "GPU_FAULT_PROCESSOR_WORKERS",
            None,
            "has no GPU_FAULT_PROCESSOR_WORKERS",
        ),
        (
            WORKER,
            "GPU_FAULT_SERVICE_ROLE",
            "ingress",
            "not GPU_FAULT_SERVICE_ROLE=worker",
        ),
        (WORKER, "GPU_FAULT_TELEMETRY_SPOOL", "true", "still runs the telemetry spool"),
        (
            SPOOL,
            "GPU_FAULT_SERVICE_ROLE",
            "worker",
            "not GPU_FAULT_SERVICE_ROLE=spool-worker",
        ),
        (SPOOL, "GPU_FAULT_TELEMETRY_SPOOL", "false", "consumer disabled"),
        (
            SPOOL,
            "GPU_FAULT_TELEMETRY_SPOOL_MAX_ITEM_BYTES",
            "bad",
            "limits must be numeric",
        ),
        (
            SPOOL,
            "GPU_FAULT_TELEMETRY_SPOOL_MAX_ITEM_BYTES",
            "1",
            "disagree on the maximum",
        ),
        (
            SPOOL,
            "GPU_FAULT_TELEMETRY_SPOOL_REPLAY_BATCH_MAX_BYTES",
            "1",
            "cannot hold one maximum item",
        ),
        (SPOOL, "GPU_FAULT_TELEMETRY_SPOOL_WORKERS", "0", "in-flight byte limit"),
        (
            SPOOL,
            "GPU_FAULT_TELEMETRY_SPOOL_MAX_IN_FLIGHT_BYTES",
            "1",
            "in-flight byte limit",
        ),
        (
            SPOOL,
            "GPU_FAULT_TELEMETRY_SPOOL_REPLAY_BATCH_MAX_ITEMS",
            "1",
            "item limit must be 64",
        ),
        (
            SPOOL,
            "GPU_FAULT_TELEMETRY_SPOOL_NOTIFICATION_FALLBACK_SECONDS",
            "6",
            "between 2 and 5",
        ),
        (
            SPOOL,
            "GPU_FAULT_PROCESSOR_WORKERS",
            "24",
            "carries main processor pool sizing",
        ),
    ],
)
def test_current_contract_rejects_each_capacity_and_role_drift(
    verifier: tuple[Any, ...],
    capsys: pytest.CaptureFixture[str],
    role: str,
    name: str,
    value: Any,
    problem: str,
) -> None:
    module, _transport, resources = verifier
    set_value(resources, role, name, value)
    assert module.main() == 1
    assert problem in capsys.readouterr().out


@pytest.mark.parametrize("role", [INGRESS, WORKER, SPOOL])
@pytest.mark.parametrize("fault", ["image", "port", "workers"])
def test_current_contract_checks_every_roles_image_and_command(
    verifier: tuple[Any, ...], capsys: pytest.CaptureFixture[str], role: str, fault: str
) -> None:
    module, _transport, resources = verifier
    container = resources[role]["spec"]["template"]["spec"]["containers"][0]
    if fault == "image":
        container["image"] = "foreign-image"
        problem = "runtime image does not match"
    elif fault == "port":
        container["args"] = [container["args"][0].replace("--port 808", "--port 909")]
        problem = "does not serve on"
    else:
        container["args"] = [container["args"][0] + " --limit-max-requests 1"]
        problem = "recycles uvicorn workers"
        if role == SPOOL:
            container["args"] = [
                container["args"][0].replace("--workers 1", "--workers 2")
            ]
            problem = "must run exactly 1 uvicorn process"
    assert module.main() == 1
    output = capsys.readouterr().out
    assert role in output
    assert problem in output


@pytest.mark.parametrize(
    "role,fault,problem",
    [
        (WORKER, "zero", "scaled to zero"),
        (WORKER, "unready", "ready"),
        (SPOOL, "zero", "scaled to zero while ingress admits"),
        (SPOOL, "unready", "ready"),
        (SPOOL, "disabled", "while ingress admission is disabled"),
    ],
)
def test_current_contract_rejects_unavailable_or_unexpected_workers(
    verifier: tuple[Any, ...],
    capsys: pytest.CaptureFixture[str],
    role: str,
    fault: str,
    problem: str,
) -> None:
    module, _transport, resources = verifier
    if fault == "zero":
        resources[role]["spec"]["replicas"] = 0
    elif fault == "unready":
        resources[role]["status"]["readyReplicas"] = 0
    else:
        set_value(resources, INGRESS, "GPU_FAULT_TELEMETRY_SPOOL", "false")
    assert module.main() == 1
    assert problem in capsys.readouterr().out


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("volume", "no aurora-credentials volume"),
        ("secret", "must project gpu-fault-aurora"),
        ("items", "whole Secret"),
        ("mount", "does not mount aurora-credentials"),
        ("path", "must mount at"),
        ("subpath", "must not use subPath"),
        ("writable", "must be read-only"),
        ("url", "must point at"),
    ],
)
def test_current_contract_requires_refreshable_readonly_aurora_mount(
    verifier: tuple[Any, ...],
    capsys: pytest.CaptureFixture[str],
    fault: str,
    problem: str,
) -> None:
    module, _transport, resources = verifier
    pod = resources[INGRESS]["spec"]["template"]["spec"]
    container = pod["containers"][0]
    volume = next(
        item for item in pod["volumes"] if item["name"] == "aurora-credentials"
    )
    mount = next(
        item
        for item in container["volumeMounts"]
        if item["name"] == "aurora-credentials"
    )
    if fault == "volume":
        pod["volumes"].remove(volume)
    elif fault == "secret":
        volume["secret"]["secretName"] = "other-secret"
    elif fault == "items":
        volume["secret"]["items"] = [{"key": "postgres-url", "path": "postgres-url"}]
    elif fault == "mount":
        container["volumeMounts"].remove(mount)
    elif fault == "path":
        mount["mountPath"] = "/elsewhere"
    elif fault == "subpath":
        mount["subPathExpr"] = "postgres-url"
    elif fault == "writable":
        mount["readOnly"] = False
    else:
        set_value(resources, INGRESS, "GPU_FAULT_STORE_URL_FILE", "/unmounted")
    assert module.main() == 1
    assert problem in capsys.readouterr().out


def test_missing_named_container_is_an_explicit_failure(
    verifier: tuple[Any, ...],
) -> None:
    module, _transport, resources = verifier
    resources[INGRESS]["spec"]["template"]["spec"]["containers"][0]["name"] = "other"
    with pytest.raises(SystemExit, match="has no api container"):
        module.main()


@pytest.mark.parametrize(
    "code,error",
    [
        (1, "Error from server (Forbidden): access denied"),
        (1, "request timed out"),
        (127, ""),
        (1, "authentication plugin failed: NotFound in local cache"),
    ],
)
def test_deployment_read_failure_is_not_reported_as_absence(
    verifier: tuple[Any, ...], code: int, error: str
) -> None:
    module, transport, _resources = verifier
    transport.failure = code, "", error
    found, problems = module.role_deployments()
    assert not found, "a failed read supplied Deployment evidence"
    assert len(problems) == 1 and "could not be read" in problems[0], problems
    assert "is missing" not in problems[0], "query failure was converted to absence"
    assert transport.calls == [
        ("deployment", INGRESS),
        ("deployment", WORKER),
        ("deployment", SPOOL),
    ]


def test_confirmed_missing_deployment_retains_missing_role_diagnostic(
    verifier: tuple[Any, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    module, transport, _resources = verifier
    del transport.resources[("deployment", WORKER)]
    assert module.main() == 1
    assert (
        "gpu-fault-control-worker is missing: nothing claims" in capsys.readouterr().out
    )


def test_environment_resolution_ignores_unnamed_entries_and_nonconfigmap_sources(
    verifier: tuple[Any, ...],
) -> None:
    module, transport, _resources = verifier
    values = module.env_values(
        {
            "envFrom": [{"secretRef": {"name": "unused"}}, {"configMapRef": {}}],
            "env": [
                {},
                {
                    "name": "POD_UID",
                    "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}},
                },
                {"name": "EXPLICIT", "value": "yes"},
            ],
        }
    )
    assert values == {"POD_UID": None, "EXPLICIT": "yes"}
    assert transport.calls == []


def test_empty_configmap_transport_error_is_explicit_even_when_optional(
    verifier: tuple[Any, ...],
) -> None:
    module, transport, _resources = verifier
    transport.failure = 3, "", ""
    with pytest.raises(SystemExit, match="kubectl exited 3"):
        module.config_map_data("unknown", optional=True)


def test_configmap_authentication_error_cannot_spoof_optional_absence(
    verifier: tuple[Any, ...],
) -> None:
    module, transport, _resources = verifier
    transport.failure = 1, "", "authentication plugin failed: NotFound in local cache"
    with pytest.raises(SystemExit, match="could not be read"):
        module.config_map_data("unknown", optional=True)
