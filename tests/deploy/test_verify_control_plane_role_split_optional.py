"""The role-split verifier honours ``optional: true`` the way kubelet does.

``deploy/control-plane/base/control-plane-deployment.yaml`` deliberately
declares ``GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP`` from ConfigMap
``gpu-fault-failure-domain-map`` with ``optional: true``: the map is rendered
separately by ``gpu-fault-admin failure-domain-map`` and a site without it runs
without the per-failure-domain remediation tier. Production deploy #7
(2026-09-07) failed at verification with ``ConfigMap
gpu-fault-failure-domain-map is missing`` because the verifier resolved every
``configMapKeyRef`` as if it were required. These tests pin kubelet semantics:

* ``configMapKeyRef`` with ``optional: true`` and an absent ConfigMap, or a
  present ConfigMap without the key, leaves the variable unset;
* ``envFrom.configMapRef`` with ``optional: true`` and an absent ConfigMap
  contributes nothing;
* without ``optional: true`` an absent ConfigMap still fails with the same
  ``ConfigMap <name> is missing`` message;
* absence is cached, so a missing optional ConfigMap costs one kubectl call
  and a later required reference to the same name still fails;
* a kubectl failure that is not NotFound is fatal even for optional
  references - kubelet only tolerates a ConfigMap that does not exist, not
  an API server it cannot reach.

Nothing here talks to kubectl: ``subprocess.run`` is replaced by a recorder
that answers from a dict of ConfigMaps.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
VERIFIER = ROOT / "deploy/control-plane/tools/verify_control_plane_role_split.py"


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_control_plane_role_split_under_test", VERIFIER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # A fresh module per test (the verifier caches lookups in module state), so
    # the shared script loader's cache is not used -- but like it, never write a
    # __pycache__ under deploy/: test_deploy_layout rejects the tree if we do.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


class _Kubectl:
    """Answers ``kubectl get configmap <name> -o json`` from a dict."""

    def __init__(
        self, config_maps: dict[str, dict[str, str]], *, unreachable: bool = False
    ) -> None:
        self.config_maps = config_maps
        self.unreachable = unreachable
        self.names: list[str] = []

    def __call__(self, command, **_kwargs) -> subprocess.CompletedProcess:
        assert command[:2] == ["kubectl", "-n"], command
        assert command[3:5] == ["get", "configmap"], command
        name = command[5]
        self.names.append(name)
        if self.unreachable:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr=(
                    "The connection to the server 10.0.0.1:443 was refused "
                    "- did you specify the right host or port?\n"
                ),
            )
        if name not in self.config_maps:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr=f'Error from server (NotFound): configmaps "{name}" not found\n',
            )
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps({"data": self.config_maps[name]}), stderr=""
        )


@pytest.fixture
def verifier(monkeypatch):
    module = _load_verifier()
    return module


def _install(monkeypatch, verifier, kubectl: _Kubectl) -> None:
    monkeypatch.setattr(verifier.subprocess, "run", kubectl)


def _key_ref(name: str, config_map: str, key: str, *, optional: bool | None):
    reference: dict[str, object] = {"name": config_map, "key": key}
    if optional is not None:
        reference["optional"] = optional
    return {"name": name, "valueFrom": {"configMapKeyRef": reference}}


def test_optional_key_ref_to_absent_config_map_is_unset(monkeypatch, verifier):
    """The production defect: the failure-domain map is optional by design."""

    kubectl = _Kubectl({})
    _install(monkeypatch, verifier, kubectl)
    api = {
        "env": [
            {"name": "GPU_FAULT_SERVICE_ROLE", "value": "worker"},
            _key_ref(
                "GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP",
                "gpu-fault-failure-domain-map",
                "map-path",
                optional=True,
            ),
        ]
    }

    values = verifier.env_values(api)

    assert values["GPU_FAULT_SERVICE_ROLE"] == "worker"
    assert values["GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP"] is None
    assert kubectl.names == ["gpu-fault-failure-domain-map"]


def test_optional_key_ref_to_present_config_map_without_key_is_unset(
    monkeypatch, verifier
):
    kubectl = _Kubectl({"gpu-fault-failure-domain-map": {"other": "x"}})
    _install(monkeypatch, verifier, kubectl)
    api = {
        "env": [
            _key_ref(
                "GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP",
                "gpu-fault-failure-domain-map",
                "map-path",
                optional=True,
            )
        ]
    }

    assert verifier.env_values(api)["GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP"] is None


def test_optional_key_ref_to_present_config_map_resolves(monkeypatch, verifier):
    kubectl = _Kubectl(
        {"gpu-fault-failure-domain-map": {"map-path": "/etc/gpu-fault/map.json"}}
    )
    _install(monkeypatch, verifier, kubectl)
    api = {
        "env": [
            _key_ref(
                "GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP",
                "gpu-fault-failure-domain-map",
                "map-path",
                optional=True,
            )
        ]
    }

    assert (
        verifier.env_value(api, "GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP")
        == "/etc/gpu-fault/map.json"
    )


@pytest.mark.parametrize("optional", [None, False])
def test_required_key_ref_to_absent_config_map_still_fails(
    monkeypatch, verifier, optional
):
    """``optional`` absent or false keeps the original fail-loud message."""

    _install(monkeypatch, verifier, _Kubectl({}))
    api = {
        "env": [
            _key_ref(
                "GPU_FAULT_REQUIRED_POLICY_VERSION",
                "gpu-fault-release-metadata",
                "required-policy-version",
                optional=optional,
            )
        ]
    }

    with pytest.raises(
        SystemExit, match="ConfigMap gpu-fault-release-metadata is missing"
    ):
        verifier.env_values(api)


def test_optional_env_from_absent_config_map_contributes_nothing(monkeypatch, verifier):
    kubectl = _Kubectl({"gpu-fault-api-ha-config-core": {"GPU_FAULT_A": "1"}})
    _install(monkeypatch, verifier, kubectl)
    api = {
        "envFrom": [
            {"configMapRef": {"name": "gpu-fault-api-ha-config-core"}},
            {"configMapRef": {"name": "gpu-fault-site-overrides", "optional": True}},
        ],
        "env": [{"name": "GPU_FAULT_SERVICE_ROLE", "value": "ingress"}],
    }

    values = verifier.env_values(api)

    assert values == {"GPU_FAULT_A": "1", "GPU_FAULT_SERVICE_ROLE": "ingress"}
    assert verifier.env_names(api) == {"GPU_FAULT_A", "GPU_FAULT_SERVICE_ROLE"}


def test_required_env_from_absent_config_map_still_fails(monkeypatch, verifier):
    _install(monkeypatch, verifier, _Kubectl({}))
    api = {"envFrom": [{"configMapRef": {"name": "gpu-fault-api-ha-config-core"}}]}

    with pytest.raises(
        SystemExit, match="ConfigMap gpu-fault-api-ha-config-core is missing"
    ):
        verifier.env_values(api)


def test_absent_config_map_is_looked_up_once(monkeypatch, verifier):
    """Absence is cached like presence: one kubectl call per name."""

    kubectl = _Kubectl({})
    _install(monkeypatch, verifier, kubectl)
    api = {
        "envFrom": [
            {"configMapRef": {"name": "gpu-fault-failure-domain-map", "optional": True}}
        ],
        "env": [
            _key_ref(
                "GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP",
                "gpu-fault-failure-domain-map",
                "map-path",
                optional=True,
            ),
            _key_ref(
                "GPU_FAULT_FAILURE_DOMAIN_MAP_DIGEST",
                "gpu-fault-failure-domain-map",
                "map-digest",
                optional=True,
            ),
        ],
    }

    verifier.env_values(api)
    verifier.env_value(api, "GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP")
    verifier.env_names(api)

    assert kubectl.names == ["gpu-fault-failure-domain-map"]


def test_cached_absence_does_not_satisfy_a_required_reference(monkeypatch, verifier):
    """An optional lookup must not turn a later required one into a pass."""

    kubectl = _Kubectl({})
    _install(monkeypatch, verifier, kubectl)
    optional_ref = _key_ref(
        "GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP",
        "gpu-fault-failure-domain-map",
        "map-path",
        optional=True,
    )
    required_ref = _key_ref(
        "GPU_FAULT_FAILURE_DOMAIN_MAP_PATH",
        "gpu-fault-failure-domain-map",
        "map-path",
        optional=None,
    )

    assert verifier.env_values({"env": [optional_ref]}) == {
        "GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP": None
    }
    with pytest.raises(
        SystemExit, match="ConfigMap gpu-fault-failure-domain-map is missing"
    ):
        verifier.env_values({"env": [optional_ref, required_ref]})
    assert kubectl.names == ["gpu-fault-failure-domain-map"]


def test_unreachable_api_server_is_fatal_even_for_optional_references(
    monkeypatch, verifier
):
    """Only NotFound is tolerated; a kubectl transport failure must not be
    read as "the map is absent" and silently unset the variable."""

    _install(monkeypatch, verifier, _Kubectl({}, unreachable=True))
    api = {
        "env": [
            _key_ref(
                "GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP",
                "gpu-fault-failure-domain-map",
                "map-path",
                optional=True,
            )
        ]
    }

    with pytest.raises(SystemExit) as excinfo:
        verifier.env_values(api)
    message = str(excinfo.value)
    assert "gpu-fault-failure-domain-map" in message
    assert "is missing" not in message
    assert "connection to the server" in message


def test_deployment_template_declares_the_failure_domain_map_optional() -> None:
    """The verifier's tolerance exists for this declaration; if the template
    ever stops marking the map optional, the tolerance tests above are
    pinning behaviour nobody relies on and should be revisited."""

    text = (ROOT / "deploy/control-plane/base/control-plane-deployment.yaml").read_text(
        encoding="utf-8"
    )
    index = text.index("name: GPU_FAULT_REMEDIATION_FAILURE_DOMAIN_MAP")
    block = text[index : index + 400]
    assert "name: gpu-fault-failure-domain-map" in block
    assert "key: map-path" in block
    assert "optional: true" in block
