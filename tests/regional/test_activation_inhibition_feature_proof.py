from __future__ import annotations

import hashlib
import importlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.adapters.hyperpod.lifecycle import HyperPodLifecycleStepAdapter
from gpu_fault.host_health import SyntheticNodeReplacementRequest
from gpu_fault.hyperpod_spares import ACTIVATION_NOT_SPECIFIED, HyperPodSpareCoordinator
from gpu_fault.store.memory.remote_commands import MemoryRemoteCommandMixin
from gpu_fault.store.postgres.remote_commands import PostgresRemoteCommandMixin
from gpu_fault.store.sqlite.remote_commands import SqliteRemoteCommandMixin
from scripts.e2e.regional.probes import destr008_inhibition_probe as capability


@pytest.mark.parametrize("component", ["api", "executor"])
def test_installed_feature_proof_is_read_only_and_complete(
    component: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("feature proof must not initialize or execute services")

    for target in (
        HyperPodSpareCoordinator,
        HyperPodLifecycleStepAdapter,
        MemoryRemoteCommandMixin,
        SqliteRemoteCommandMixin,
        PostgresRemoteCommandMixin,
    ):
        monkeypatch.setattr(target, "__init__", forbidden, raising=False)

    def allocate(self, *, activation_forbidden=ACTIVATION_NOT_SPECIFIED, **kwargs):
        return forbidden()

    monkeypatch.setattr(HyperPodSpareCoordinator, "allocate", allocate)
    monkeypatch.setattr(HyperPodSpareCoordinator, "reserve", allocate)
    result = capability.activation_inhibition_feature_proof(component=component)
    assert result == {
        "capability": "synthetic-replacement-activation-inhibition",
        "capability_version": 1,
        "component": component,
        "marker": "activation_forbidden",
        "minimum_executor_protocol_version": 4,
        "executor_protocol_version": 4,
        "step_batching_protocol_version": 3,
        "supported": True,
        "checks": result["checks"],
    }
    assert result["checks"] and all(
        value is True for value in result["checks"].values()
    )


@pytest.mark.parametrize("component", ["api", "executor"])
@pytest.mark.parametrize("version", [1, 2, 3, True, "4"])
def test_old_or_malformed_protocol_cannot_supply_a_feature_proof(
    component: str, version, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        capability, "CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION", version
    )
    result = capability.activation_inhibition_feature_proof(component=component)
    assert result["supported"] is False
    assert result["checks"]["protocol"] is False


def test_unknown_component_is_rejected_before_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(name):
        raise AssertionError("unsupported role must not import a component")

    monkeypatch.setattr(capability.importlib, "import_module", forbidden)
    with pytest.raises(ValueError, match="api or executor"):
        capability.activation_inhibition_feature_proof(component="node-agent")


@pytest.mark.parametrize("component", ["api", "executor"])
def test_missing_component_is_unsupported_without_leaking_diagnostics(
    component: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(name):
        raise ImportError("private-provider-diagnostic-must-not-escape")

    monkeypatch.setattr(capability.importlib, "import_module", missing)
    result = capability.activation_inhibition_feature_proof(component=component)
    assert result["supported"] is False
    assert result["checks"]["inspection_complete"] is False
    assert "private-provider" not in str(result)


@pytest.mark.parametrize(
    "module_name,attribute",
    [
        ("gpu_fault.app.routes.collector_events", None),
        ("gpu_fault.app.routes.regional", None),
        ("gpu_fault.orchestration.families.health", "NodeHealthPlanBuilder"),
        (
            "gpu_fault.orchestration.families.node_lifecycle",
            "NodeLifecycleOperationService",
        ),
    ],
)
def test_old_propagation_or_claim_code_is_not_a_supported_api(
    module_name: str, attribute: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module(module_name)
    target = module if attribute is None else getattr(module, attribute)
    monkeypatch.setattr(target, "activation_inhibition_version", 0)
    result = capability.activation_inhibition_feature_proof(component="api")
    assert result["supported"] is False
    assert result["checks"]["propagation_and_claim_version"] is False


@pytest.mark.parametrize(
    "target",
    [MemoryRemoteCommandMixin, SqliteRemoteCommandMixin, PostgresRemoteCommandMixin],
)
def test_all_claim_backends_must_accept_the_advertised_protocol(
    target, monkeypatch: pytest.MonkeyPatch
) -> None:
    def old(self, cluster_id, executor_id, *, accept_batched_steps=True):
        raise AssertionError("claim method must not execute")

    monkeypatch.setattr(target, "claim_remote_commands", old)
    result = capability.activation_inhibition_feature_proof(component="api")
    assert result["supported"] is False
    assert result["checks"]["claim_protocol_argument"] is False


@pytest.mark.parametrize(
    "defect", ["field", "mutable", "coercion", "default", "true", "serialization"]
)
def test_request_field_and_actual_validation_are_proved(
    defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("gpu_fault.host_health")
    real = SyntheticNodeReplacementRequest

    class Model:
        model_fields = (
            {}
            if defect == "field"
            else {"activation_forbidden": SimpleNamespace(frozen=defect != "mutable")}
        )

        @staticmethod
        def model_validate(value):
            provided = "activation_forbidden" in value
            if defect == "coercion" and provided:
                value = {**value, "activation_forbidden": True}
            result = real.model_validate(value)
            if defect == "default" and not provided:
                return SimpleNamespace(
                    model_dump=lambda **kwargs: {"activation_forbidden": None}
                )
            if defect == "true" and provided:
                return SimpleNamespace(activation_forbidden=False)
            if defect == "serialization" and provided:
                return SimpleNamespace(
                    activation_forbidden=True, model_dump=lambda **kwargs: {}
                )
            return result

    monkeypatch.setattr(module, "SyntheticNodeReplacementRequest", Model)
    result = capability.activation_inhibition_feature_proof(component="api")
    assert result["supported"] is False
    assert result["checks"]["inspection_complete"] is True


@pytest.mark.parametrize(
    "target", [HyperPodSpareCoordinator, HyperPodLifecycleStepAdapter]
)
@pytest.mark.parametrize("version", [0, True, None])
def test_old_coordinator_or_confirmation_is_not_a_supported_executor(
    target, version, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(target, "activation_inhibition_version", version)
    result = capability.activation_inhibition_feature_proof(component="executor")
    assert result["supported"] is False


@pytest.mark.parametrize("method", ["allocate", "reserve"])
def test_executor_requires_explicit_inhibition_keyword_support(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def old(self, *args, **kwargs):
        raise AssertionError("coordinator must not execute")

    monkeypatch.setattr(HyperPodSpareCoordinator, method, old)
    result = capability.activation_inhibition_feature_proof(component="executor")
    assert result["supported"] is False
    assert result["checks"][method + "_keyword"] is False


def test_non_inhibiting_guard_is_rejected_without_executing_an_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("gpu_fault.hyperpod_spares")
    monkeypatch.setattr(
        module, "require_spare_activation_permitted", lambda *args: None
    )
    result = capability.activation_inhibition_feature_proof(component="executor")
    assert result["supported"] is False
    assert result["checks"]["present_values_refused"] is False


@pytest.mark.parametrize("method", ["allocate", "reserve"])
def test_changed_coordinator_default_is_not_a_supported_executor(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def wrong(self, *, activation_forbidden=None):
        raise AssertionError("coordinator must not execute")

    monkeypatch.setattr(HyperPodSpareCoordinator, method, wrong)
    result = capability.activation_inhibition_feature_proof(component="executor")
    assert result["supported"] is False
    assert result["checks"][method + "_keyword"] is False


@pytest.mark.parametrize("default", [True, 4])
def test_unknown_claim_caller_is_not_assumed_to_support_inhibition(
    default, monkeypatch: pytest.MonkeyPatch
) -> None:
    def wrong(self, *, executor_protocol_version=default):
        raise AssertionError("claim method must not execute")

    monkeypatch.setattr(MemoryRemoteCommandMixin, "claim_remote_commands", wrong)
    result = capability.activation_inhibition_feature_proof(component="api")
    assert result["supported"] is False
    assert result["checks"]["claim_protocol_argument"] is False


def test_default_guard_must_preserve_absent_marker_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("gpu_fault.hyperpod_spares")

    def wrong(*args):
        if args:
            raise module.SpareActivationForbidden("refused")
        return False

    monkeypatch.setattr(module, "require_spare_activation_permitted", wrong)
    result = capability.activation_inhibition_feature_proof(component="executor")
    assert result["supported"] is False
    assert result["checks"]["absence_allowed"] is False


def test_feature_probe_does_not_claim_global_admission_or_deployment_proof() -> None:
    result = capability.activation_inhibition_feature_proof(component="api")
    assert result["supported"] is True
    assert "pod_uid" not in result and "release_id" not in result
    assert "admission_converged" not in result and "cluster_ready" not in result


@pytest.mark.parametrize("digest", [None, "", "bad", True, "z" * 64])
def test_probe_entry_requires_a_verified_source_digest(
    monkeypatch: pytest.MonkeyPatch, digest: object
) -> None:
    monkeypatch.setattr(capability, "_PROBE_SHA256", digest, raising=False)
    with pytest.raises(ValueError, match="pinned source"):
        capability.main()


@pytest.mark.parametrize(
    "arguments", [[], ["probe"], ["probe", "other"], ["probe", "cpu", "extra"]]
)
def test_probe_entry_rejects_unknown_role_before_inspection(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    monkeypatch.setattr(capability, "_PROBE_SHA256", "a" * 64, raising=False)
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(ValueError, match="cpu or gpu"):
        capability.main()


@pytest.mark.parametrize("plane", ["cpu", "gpu"])
def test_pinned_entry_emits_the_actual_readonly_feature_proof(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], plane: str
) -> None:
    source = Path(capability.__file__)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(sys, "argv", [str(source), plane])
    runpy.run_path(
        str(source), run_name="__main__", init_globals={"_PROBE_SHA256": digest}
    )
    result = json.loads(capsys.readouterr().out)
    assert result["probe_sha256"] == digest
    assert result["component"] == ("api" if plane == "cpu" else "executor")
    assert result["supported"] is True
    assert all(value is True for value in result["checks"].values()), (
        "the entry must inspect the real installed protocol and inhibition code"
    )
