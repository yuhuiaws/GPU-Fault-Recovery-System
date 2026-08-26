from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
CHECKS = lazy_script_module(
    "regional_admin_checks",
    ROOT / "deploy/control-plane/regional/regional_admin_checks.py",
)


class Config:
    site_name = "test-site"
    clusters = []


class Release:
    config = Config()


def test_health_report_exit_code_tracks_failures() -> None:
    healthy = CHECKS._report(
        "verify",
        Release(),
        [{"name": "a", "status": "PASS", "summary": "ok", "details": None}],
    )
    unhealthy = CHECKS._report(
        "verify",
        Release(),
        [{"name": "a", "status": "FAIL", "summary": "broken", "details": None}],
    )

    assert healthy["healthy"] is True
    assert CHECKS.report_exit_code(healthy) == 0
    assert unhealthy["healthy"] is False
    assert CHECKS.report_exit_code(unhealthy) == 1


def test_check_aggregates_exceptions_without_stopping() -> None:
    def broken():
        raise RuntimeError("boom")

    result = CHECKS._check("broken", broken)

    assert result["status"] == "FAIL"
    assert result["summary"] == "boom"


def test_empty_node_action_keys_are_valid_for_an_empty_gpu_registry(
    monkeypatch,
) -> None:
    values = {
        "gpu-fault-aurora": {"data": {"postgres-url": "x", "master-secret-arn": "y"}},
        "gpu-fault-control-plane-active": {
            "data": {
                "execution-token": base64.b64encode(b"a" * 32).decode(),
                "processor-replay-secret": base64.b64encode(b"b" * 32).decode(),
                "node-action-secret": base64.b64encode(b"c" * 32).decode(),
            }
        },
        "gpu-fault-node-action-keys": {"data": None},
    }
    monkeypatch.setitem(
        CHECKS.check_cpu_secrets.__globals__,
        "_secret",
        lambda _release, name: values[name],
    )
    release = SimpleNamespace(config=SimpleNamespace(clusters=[]))

    result = CHECKS.check_cpu_secrets(release)

    assert result.details["node_action_key_count"] == 0


def test_preflight_report_contains_all_required_domains(monkeypatch) -> None:
    module = CHECKS._load()
    names = (
        "_check_tools",
        "_check_local_inputs",
        "_check_aws_identity",
        "_check_contexts",
        "_check_cpu_capacity",
        "check_cpu_secrets",
        "_check_load_balancer_controller",
        "_check_nlb_inputs",
        "_check_aurora",
        "_check_aurora_refresh",
        "_check_email_notifications",
        "_check_monitoring",
    )
    for name in names:
        monkeypatch.setattr(
            module, name, lambda *args, name=name: module.CheckValue(name)
        )

    report = module.build_preflight_report(Release())

    assert report["healthy"] is True
    assert report["summary"]["PASS"] == len(names)
    assert {item["name"] for item in report["checks"]} == {
        "tools",
        "local_inputs",
        "aws_identity",
        "regional_contexts",
        "cpu_capacity",
        "cpu_secrets",
        "load_balancer_controller",
        "nlb_inputs",
        "aurora",
        "aurora_credential_refresh",
        "email_notifications",
        "monitoring",
    }


def control_api_release() -> SimpleNamespace:
    return SimpleNamespace(
        wheel_sha="a" * 64,
        node_wheel_sha="a" * 64,
        executor_wheel_sha="d" * 64,
        config=SimpleNamespace(
            agent_config_digest="b" * 64,
            component_digests={"node_runtime": "e" * 64, "executor": "f" * 64},
            runtime_profile_version="hyperpod-v1",
            health=SimpleNamespace(remote_command_max_unclaimed_seconds=300),
            clusters=[SimpleNamespace(cluster_id="gpu-a")],
        ),
    )


def healthy_control_api_report() -> dict:
    return {
        "healthz": {"status": "ok"},
        "version": {
            "deployment_mode": "regional",
            "required_agent_artifact_sha256": "a" * 64,
            "required_agent_config_digest": "b" * 64,
            "compatible_agent_artifact_sha256s": [],
            "required_agent_compatibility_digest": "e" * 64,
            "compatible_agent_compatibility_digests": [],
            "compatible_agent_protocol_versions": [],
            "compatible_agent_config_digests": [],
            "required_runtime_profile_version": "hyperpod-v1",
            "required_node_action_key_version": 2,
            "required_regional_executor_protocol_version": 2,
            "compatible_regional_executor_protocol_versions": [],
            "required_regional_executor_artifact_sha256": "d" * 64,
            "compatible_regional_executor_artifact_sha256s": [],
            "required_regional_executor_compatibility_digest": "f" * 64,
            "compatible_regional_executor_compatibility_digests": [],
        },
        "registry": [{"cluster_id": "gpu-a"}],
        "clusters": {
            "gpu-a": {
                "expected_node_ids": ["node-a"],
                "agents": [
                    {
                        "node_id": "node-a",
                        "lifecycle_state": "ACTIVE",
                        "runtime_profile_version": "hyperpod-v1",
                    }
                ],
                "fleet_readiness": {"ready": True},
                "collector_readiness": {"ready": True},
            }
        },
        "remote_commands": {
            "oldest_unclaimed_age_seconds": 0,
            "executor_internal_error_total": 0,
        },
    }


def test_control_api_health_accepts_a_converged_fleet(monkeypatch) -> None:
    module = CHECKS._load()
    report = healthy_control_api_report()
    monkeypatch.setattr(module, "_control_api_report", lambda _release: report)

    value = module._check_control_api(control_api_release())

    assert value.summary == "control-plane API, fleet and collectors are healthy"


def test_control_api_probe_does_not_inject_unknown_gpu_fault_variables() -> None:
    module = CHECKS._load()

    assert "GPU_FAULT_ADMIN_" not in module.CONTROL_API_SCRIPT
    assert "ADMIN_CLUSTER_IDS_JSON" in module.CONTROL_API_SCRIPT
    assert "ADMIN_EXPECTED_NODES_JSON" in module.CONTROL_API_SCRIPT


def test_split_artifact_checks_use_executor_and_node_pins() -> None:
    source = (
        ROOT / "deploy/control-plane/regional/regional_admin_checks.py"
    ).read_text(encoding="utf-8")
    verifier = source.split("def _run_read_only_verifiers", 1)[1].split(
        "def _cpu_ingress_pod", 1
    )[0]
    gpu = source.split("def _check_gpu_cluster", 1)[1].split(
        "def _check_nlb_runtime", 1
    )[0]

    assert '"GPU_FAULT_EXPECTED_WHEEL_CONFIGMAP": release.executor_wheel_cm' in verifier
    assert "!= release.node_wheel_sha" in gpu


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("oldest_unclaimed_age_seconds", 301, "oldest unclaimed"),
        ("executor_internal_error_total", 1, "internal"),
    ],
)
def test_control_api_health_rejects_stuck_or_broken_commands(
    monkeypatch, field: str, value: int, message: str
) -> None:
    module = CHECKS._load()
    report = healthy_control_api_report()
    report["remote_commands"][field] = value
    monkeypatch.setattr(module, "_control_api_report", lambda _release: report)

    with pytest.raises(module.ReleaseError, match=message):
        module._check_control_api(control_api_release())


def test_certificate_hostname_matching_is_single_label() -> None:
    module = CHECKS._load()

    assert module._hostname_matches("control.example", "control.example")
    assert module._hostname_matches("*.example", "control.example")
    assert not module._hostname_matches("*.example", "nested.control.example")
