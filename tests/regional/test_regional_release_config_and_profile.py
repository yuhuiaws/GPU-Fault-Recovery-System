from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from gpu_fault.capabilities import compile_runtime_profile
from gpu_fault.models import RuntimeProfile
from gpu_fault_release import regional_admin_commands as ADMIN
from tests.regional._release_orchestrator_support import (
    DNS_MODULE,
    GPU_EKS_ARN,
    REGION,
    ROOT,
    RUNTIME_PROFILE_MODULE,
    RuntimeProfileRunner,
    config_file,
)
from tests.regional._release_orchestrator_support import RELEASE_MODULE as MODULE


def test_release_config_requires_unique_clusters(tmp_path) -> None:
    cluster = {
        "cluster_id": "gpu-a",
        "context": "gpu-a-context",
        "executor_irsa_role_arn": "arn:aws:iam::1:role/a",
        "region": REGION,
        "hyperpod_cluster_name": "hp-gpu-a",
        "eks_cluster_arn": GPU_EKS_ARN,
    }

    with pytest.raises(MODULE.ReleaseError, match="unique"):
        MODULE.ReleaseConfig.load(config_file(tmp_path, clusters=[cluster, cluster]))


def test_release_config_requires_runtime_profile_inputs(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value.pop("runtime_profile")
    path.write_text(json.dumps(value))

    with pytest.raises(MODULE.ReleaseError, match="runtime_profile.source"):
        MODULE.ReleaseConfig.load(path)


def test_release_config_allows_a_stable_site_profile_anchor(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["runtime_profile"]["registration_cluster_id"] = "missing"
    path.write_text(json.dumps(value))

    config = MODULE.ReleaseConfig.load(path)

    assert config.runtime_profile_registration_cluster_id == "missing"


def test_release_config_allows_an_empty_managed_gpu_set(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["clusters"] = []
    path.write_text(json.dumps(value))

    config = MODULE.ReleaseConfig.load(path)

    assert config.clusters == ()


def test_release_config_requires_existing_profile_source(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["runtime_profile"]["source"] = str(tmp_path / "missing-profile.yaml")
    path.write_text(json.dumps(value))

    with pytest.raises(MODULE.ReleaseError, match="existing file"):
        MODULE.ReleaseConfig.load(path)


def test_release_config_rejects_unsafe_profile_version(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["runtime_profile"]["version"] = "hyperpod-v2&unexpected"
    path.write_text(json.dumps(value))

    with pytest.raises(MODULE.ReleaseError, match="Runtime Profile version"):
        MODULE.ReleaseConfig.load(path)


def test_release_config_requires_explicit_matching_region(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["aws_region"] = "REPLACE_WITH_AWS_REGION"
    path.write_text(json.dumps(value))

    with pytest.raises(MODULE.ReleaseError, match="aws_region"):
        MODULE.ReleaseConfig.load(path)

    value["aws_region"] = REGION
    value["clusters"][0]["region"] = "us-west-2"
    path.write_text(json.dumps(value))
    with pytest.raises(MODULE.ReleaseError, match="does not match aws_region"):
        MODULE.ReleaseConfig.load(path)


def test_release_config_rejects_cross_region_eks_arns(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["clusters"][0]["eks_cluster_arn"] = (
        "arn:aws:eks:us-west-2:123456789012:cluster/gpu-a"
    )
    path.write_text(json.dumps(value))

    with pytest.raises(MODULE.ReleaseError, match="eks_cluster_arn Region"):
        MODULE.ReleaseConfig.load(path)


def test_release_config_rejects_cross_region_nlb_certificate(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["nlb"] = {
        "name": "gpu-fault-regional",
        "public_subnets": "subnet-a,subnet-b",
        "security_group": "sg-0123456789abcdef0",
        "certificate_arn": ("arn:aws:acm:us-west-2:123456789012:certificate/example"),
    }
    path.write_text(json.dumps(value))

    with pytest.raises(MODULE.ReleaseError, match="certificate_arn Region"):
        MODULE.ReleaseConfig.load(path)


def test_nlb_manifest_uses_explicit_name_and_region_certificate(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["nlb"] = {
        "name": "gpu-fault-regional-test",
        "public_subnets": "subnet-a,subnet-b",
        "security_group": "sg-0123456789abcdef0",
        "certificate_arn": ("arn:aws:acm:us-east-1:123456789012:certificate/example"),
    }
    path.write_text(json.dumps(value))
    config = MODULE.ReleaseConfig.load(path)

    source = (
        ROOT / "deploy/control-plane/regional/regional-control-plane-nlb.yaml"
    ).read_text(encoding="utf-8")
    rendered = DNS_MODULE.render_nlb_manifest(config, source)

    assert "gpu-fault-regional-test" in rendered
    assert "arn:aws:acm:us-east-1:" in rendered
    assert "REPLACE_WITH" not in rendered


def test_plan_covers_first_deploy_and_rollback(tmp_path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))

    deploy = release.plan("deploy")
    rollback = release.plan("rollback")
    join = release.plan("join-cluster")

    assert any("prerequisites" in step for step in deploy), (
        f"deploy plan must state its prerequisites: {deploy}"
    )
    assert any("PostgreSQL schema" in step for step in deploy), (
        f"deploy plan must cover the schema migration: {deploy}"
    )
    assert any("Runtime Profile" in step for step in deploy), (
        "deploy plan must include Runtime Profile registration"
    )
    assert any("previous required pins" in step for step in rollback), (
        f"rollback plan must restore the previous pins: {rollback}"
    )
    assert any("installer bundle" in step for step in rollback), (
        f"rollback plan must cover the node installer bundle: {rollback}"
    )
    assert any("Runtime Profile" in step for step in join), (
        "join plan must verify the shared Runtime Profile"
    )


def test_regional_parser_exposes_admin_health_commands() -> None:
    mode = next(action for action in MODULE.parser()._actions if action.dest == "mode")

    assert {"preflight", "verify", "release-summary", "status"} <= set(mode.choices)


def test_status_keeps_health_report_when_release_summary_is_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    module = MODULE
    config = module.ReleaseConfig.load(config_file(tmp_path))
    release = module.RegionalRelease(config, module.Runner(dry_run=True))
    monkeypatch.setattr(module.RegionalRelease, "_load_state", lambda _self: {})
    monkeypatch.setattr(
        module.RegionalRelease, "_apply_health_baseline", lambda _self, _state: None
    )
    monkeypatch.setattr(
        ADMIN,
        "build_full_status",
        lambda _release, *, state: {
            "healthy": False,
            "release_status_error": "deployment is missing",
            "health": {"mode": "status"},
        },
    )

    report = release.status(full=True)

    assert report["healthy"] is False
    assert report["release_status_error"] == "deployment is missing"
    assert report["health"]["mode"] == "status"


def test_runtime_profile_payload_uses_declared_identity(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))

    payload = RUNTIME_PROFILE_MODULE.render_runtime_profile_payload(config)

    assert payload["cluster_id"] == "gpu-a"
    assert payload["profile_version"] == "hyperpod-v1"
    assert payload["cluster_id"] != "hp-gpu-a"


def test_runtime_profile_policy_digest_ignores_identity_and_yaml_order(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(
        yaml.safe_dump(
            {
                "cluster_id": "placeholder",
                "environment": "hyperpod-eks",
                "profile_version": "profile-a",
                "claims": [
                    {
                        "capability": "gpuReset",
                        "mode": "OBSERVE",
                        "owner": "gpu-fault-node-agent",
                    },
                    {
                        "capability": "evidenceCapture",
                        "mode": "OWN",
                        "owner": "gpu-fault-control-plane",
                        "adapter": "control-plane-evidence",
                    },
                ],
                "observed": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    second.write_text(
        yaml.safe_dump(
            {
                "profile_version": "profile-b",
                "cluster_id": "gpu-a",
                "observed": [],
                "claims": list(reversed(yaml.safe_load(first.read_text())["claims"])),
                "environment": "hyperpod-eks",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert RUNTIME_PROFILE_MODULE.runtime_profile_policy_digest(
        first
    ) == RUNTIME_PROFILE_MODULE.runtime_profile_policy_digest(second)


def test_runtime_profile_is_registered_when_missing(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    runner = RuntimeProfileRunner()
    release = MODULE.RegionalRelease(config, runner)

    MODULE.ensure_runtime_profile(release)

    assert len(runner.posted) == 1
    assert runner.posted[0]["cluster_id"] == "gpu-a"
    assert runner.posted[0]["profile_version"] == "hyperpod-v1"


def test_runtime_profile_registration_is_idempotent(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    desired = compile_runtime_profile(
        RuntimeProfile.model_validate(
            RUNTIME_PROFILE_MODULE.render_runtime_profile_payload(config)
        )
    ).model_dump(mode="json")
    runner = RuntimeProfileRunner(existing=desired)
    release = MODULE.RegionalRelease(config, runner)

    MODULE.ensure_runtime_profile(release)

    assert runner.posted == []


def test_runtime_profile_drift_requires_a_new_version(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    existing = compile_runtime_profile(
        RuntimeProfile.model_validate(
            RUNTIME_PROFILE_MODULE.render_runtime_profile_payload(config)
        )
    ).model_dump(mode="json")
    existing["capabilities"][0]["mode"] = "OBSERVE"
    runner = RuntimeProfileRunner(existing=existing)
    release = MODULE.RegionalRelease(config, runner)

    with pytest.raises(MODULE.ReleaseError, match="new profile version"):
        MODULE.ensure_runtime_profile(release)

    assert runner.posted == []


def test_runtime_profile_verify_is_read_only_when_missing(tmp_path: Path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    runner = RuntimeProfileRunner()
    release = MODULE.RegionalRelease(config, runner)

    with pytest.raises(MODULE.ReleaseError, match="is not registered"):
        RUNTIME_PROFILE_MODULE.verify_runtime_profile(release)

    assert runner.posted == []


def test_release_config_bounds_cluster_parallelism(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["release"]["upgrade_max_parallel_clusters"] = 4
    path.write_text(json.dumps(value))

    assert MODULE.ReleaseConfig.load(path).upgrade_max_parallel_clusters == 4

    value["release"]["upgrade_max_parallel_clusters"] = 9
    path.write_text(json.dumps(value))

    with pytest.raises(
        MODULE.ReleaseError,
        match="release.upgrade_max_parallel_clusters must be within 1..8",
    ):
        MODULE.ReleaseConfig.load(path)


def test_plan_states_the_configured_cluster_parallelism(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    serial = MODULE.RegionalRelease(
        MODULE.ReleaseConfig.load(path), MODULE.Runner(dry_run=True)
    )
    value = json.loads(path.read_text())
    value["release"]["upgrade_max_parallel_clusters"] = 3
    path.write_text(json.dumps(value))
    parallel = MODULE.RegionalRelease(
        MODULE.ReleaseConfig.load(path), MODULE.Runner(dry_run=True)
    )

    assert serial.gpu_cluster_rollout_step() == (
        "roll affected GPU clusters one at a time"
    )
    assert parallel.gpu_cluster_rollout_step() == (
        "roll affected GPU clusters with bounded parallelism (up to 3 at a time)"
    )


def test_release_config_loads_health_targets(tmp_path: Path) -> None:
    path = config_file(tmp_path)
    value = json.loads(path.read_text())
    value["site_name"] = "production"
    value["health"] = {
        "aurora_cluster_id": "gpu-fault-aurora",
        "amp_workspace_id": "ws-test",
        "amp_rule_namespace": "gpu-fault-rules",
        "sns_topic_arn": ("arn:aws:sns:us-east-1:123456789012:gpu-fault"),
        "certificate_min_validity_days": 45,
        "remote_command_max_unclaimed_seconds": 240,
        "require_confirmed_sns_subscription": True,
    }
    path.write_text(json.dumps(value))

    config = MODULE.ReleaseConfig.load(path)

    assert config.site_name == "production"
    assert config.upgrade_max_unavailable == 1
    assert config.rollback_max_unavailable == 2
    assert config.upgrade_max_parallel_clusters == 1
    assert config.health.aurora_cluster_id == "gpu-fault-aurora"
    assert config.health.amp_workspace_id == "ws-test"
    assert config.health.certificate_min_validity_days == 45
    assert config.health.remote_command_max_unclaimed_seconds == 240
