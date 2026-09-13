"""The role-split verifier reads its objects in two kubectl calls, not eleven.

Measured live: the verifier took 28.6 s after every role apply, one
``kubectl get`` (~1.2 s of exec-plugin auth plus API round trip) per role
Deployment and per referenced ConfigMap. It now issues one
``kubectl get deployment <a> <b> <c>`` and one batched ``get configmap`` for
every ConfigMap those Deployments reference, both with ``--ignore-not-found``
so absence is the set difference and a non-zero exit is a real read failure.

These tests pin the call count and that the NotFound semantics survived the
batching: a missing role Deployment is reported as missing (and only it), a
missing required ConfigMap still fails with ``ConfigMap <name> is missing``, a
missing optional one is ``None``, and an unreachable API server is fatal even
for optional references. A fresh module per test: the verifier caches
ConfigMaps in module state, and the batching is what is under test.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from gpu_fault.container_env_snapshot import ROLE_DEPLOYMENTS
from tests.deploy.test_verify_control_plane_role_split_snapshot import (
    _live_control_plane,
)

ROOT = Path(__file__).resolve().parents[2]
VERIFIER = ROOT / "deploy/control-plane/tools/verify_control_plane_role_split.py"
CORE_MAPS = [f"{name}-config-core" for name in ROLE_DEPLOYMENTS]


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_control_plane_role_split_batch_under_test", VERIFIER
    )
    assert spec is not None and spec.loader is not None, "verifier must be loadable"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


@pytest.fixture
def verifier(monkeypatch):
    monkeypatch.delenv("GPU_FAULT_ROLE_SPLIT_CONTAINER_ENV_FILE", raising=False)
    monkeypatch.delenv("GPU_FAULT_RUNTIME_IMAGE", raising=False)
    return _load_verifier()


def _names(command: list[str]) -> list[str]:
    names: list[str] = []
    for word in command[5:]:
        if word.startswith("-"):
            break
        names.append(word)
    return names


class _Kubectl:
    """kubectl under ``--ignore-not-found``: found objects as a List, exit 0."""

    def __init__(
        self,
        deployments: dict[str, dict[str, Any]],
        config_maps: dict[str, dict[str, str]],
        *,
        unreachable: bool = False,
    ) -> None:
        self.deployments = deployments
        self.config_maps = config_maps
        self.unreachable = unreachable
        self.calls: list[list[str]] = []

    def __call__(
        self, command: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess:
        assert command[:4] == ["kubectl", "-n", "gpu-fault-system", "get"], command
        assert command[-3:] == ["--ignore-not-found", "-o", "json"], command
        self.calls.append(command)
        if self.unreachable:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="The connection to the server 10.0.0.1:443 was refused\n",
            )
        kind = command[4]
        if kind == "deployment":
            items = [
                self.deployments[name]
                for name in _names(command)
                if name in self.deployments
            ]
        else:
            assert kind == "configmap", command
            items = [
                {"metadata": {"name": name}, "data": self.config_maps[name]}
                for name in _names(command)
                if name in self.config_maps
            ]
        stdout = (
            json.dumps({"kind": "List", "apiVersion": "v1", "items": items})
            if items
            else ""
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def _install(monkeypatch, verifier, kubectl: _Kubectl) -> None:
    monkeypatch.setattr(verifier.subprocess, "run", kubectl)


def test_default_mode_reads_deployments_and_config_maps_in_one_call_each(
    monkeypatch, verifier, capsys
):
    kubectl = _Kubectl(_live_control_plane(), {name: {} for name in CORE_MAPS})
    _install(monkeypatch, verifier, kubectl)

    assert verifier.main() == 0, capsys.readouterr().out
    assert [call[4] for call in kubectl.calls] == ["deployment", "configmap"], (
        f"exactly one get per kind, got {kubectl.calls}"
    )
    assert _names(kubectl.calls[0]) == list(ROLE_DEPLOYMENTS), (
        "the three role Deployments are asked for together"
    )
    assert _names(kubectl.calls[1]) == CORE_MAPS, (
        "every ConfigMap the Deployments reference is asked for together"
    )


def test_a_missing_role_deployment_is_reported_from_the_batched_answer(
    monkeypatch, verifier, capsys
):
    live = _live_control_plane()
    live.pop("gpu-fault-control-worker")
    kubectl = _Kubectl(live, {})
    _install(monkeypatch, verifier, kubectl)

    assert verifier.main() == 1, "a missing role fails the check"
    out = capsys.readouterr().out
    assert (
        "gpu-fault-control-worker is missing: nothing claims from the processor queue"
        in out
    ), out
    assert "gpu-fault-api-ha is missing" not in out, "found roles are not missing"
    assert [call[4] for call in kubectl.calls] == ["deployment"], (
        "no ConfigMap is read once a role is missing"
    )


def test_an_unreadable_api_server_reports_every_role_missing_as_before(
    monkeypatch, verifier, capsys
):
    _install(monkeypatch, verifier, _Kubectl({}, {}, unreachable=True))

    assert verifier.main() == 1, "an unreadable control plane fails the check"
    out = capsys.readouterr().out
    assert all(f"{name} is missing" in out for name in ROLE_DEPLOYMENTS), out


def test_batched_config_map_read_keeps_kubelet_optional_semantics(
    monkeypatch, verifier
):
    kubectl = _Kubectl({}, {"gpu-fault-api-ha-config-core": {"GPU_FAULT_A": "1"}})
    _install(monkeypatch, verifier, kubectl)
    api = {
        "envFrom": [
            {"configMapRef": {"name": "gpu-fault-api-ha-config-core"}},
            {"configMapRef": {"name": "gpu-fault-site-overrides", "optional": True}},
        ],
        "env": [
            {
                "name": "GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP",
                "valueFrom": {
                    "configMapKeyRef": {
                        "name": "gpu-fault-failure-domain-map",
                        "key": "map-path",
                        "optional": True,
                    }
                },
            }
        ],
    }

    verifier.load_config_maps(verifier.referenced_config_maps([_pod(api)]))
    values = verifier.env_values(api)

    assert values == {
        "GPU_FAULT_A": "1",
        "GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP": None,
    }, "absent optional references contribute nothing or unset"
    assert len(kubectl.calls) == 1, f"one batched read for all three: {kubectl.calls}"
    assert _names(kubectl.calls[0]) == [
        "gpu-fault-api-ha-config-core",
        "gpu-fault-site-overrides",
        "gpu-fault-failure-domain-map",
    ], "envFrom and keyRef references are collected in order, once each"
    with pytest.raises(
        SystemExit, match="ConfigMap gpu-fault-site-overrides is missing"
    ):
        verifier.config_map_data("gpu-fault-site-overrides")
    assert len(kubectl.calls) == 1, "cached absence still fails a required reference"


def test_a_failed_batched_read_is_fatal_and_names_every_config_map(
    monkeypatch, verifier
):
    _install(monkeypatch, verifier, _Kubectl({}, {}, unreachable=True))

    with pytest.raises(SystemExit) as excinfo:
        verifier.load_config_maps(["gpu-fault-a", "gpu-fault-b"])
    message = str(excinfo.value)
    assert "gpu-fault-a" in message and "gpu-fault-b" in message, message
    assert "is missing" not in message, "unreadable is not absent"
    assert "connection to the server" in message, message


def test_kubectl_get_indexes_single_object_and_empty_answers(monkeypatch, verifier):
    answers = iter(
        [
            subprocess.CompletedProcess(
                [], 0, stdout=json.dumps({"metadata": {"name": "only"}}), stderr=""
            ),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        ]
    )
    monkeypatch.setattr(verifier.subprocess, "run", lambda *_a, **_k: next(answers))

    single, error = verifier.kubectl_get("configmap", ["only"])
    assert set(single) == {"only"} and error == "", (
        "one name comes back as the object itself"
    )
    none, error = verifier.kubectl_get("configmap", ["absent"])
    assert none == {} and error == "", "nothing found is an empty answer, not a failure"


def _pod(container: dict[str, Any]) -> dict[str, Any]:
    return {"spec": {"template": {"spec": {"containers": [container]}}}}
