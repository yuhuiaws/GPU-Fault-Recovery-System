from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.config import default_admin_config
from gpu_fault_release import regional_release_config as config
from tests.regional._cov95_release_support import NEW_IMAGE
from tests.regional._release_orchestrator_support import config_file


def delivery_manifest(version: int = 4) -> tuple[dict[str, Any], dict[str, Any]]:
    components = {
        "control_plane": {"wheel_sha256": "1" * 64, "module_digest": "2" * 64},
        "executor": {"wheel_sha256": "3" * 64, "module_digest": "4" * 64},
        "node_bundle": {"template_sha256": "5" * 64},
    }
    names = (
        "collector",
        "cpu",
        "dcgm",
        "endpoint",
        "executor",
        "node",
        "observability",
        "schema",
        "watcher",
        "cpu_ingress",
        "cpu_spool",
        "cpu_worker",
    )
    delivery = {
        "runtime_prebuilt": True,
        "components": {name: {"sha256": "6" * 64} for name in names},
        "images": {
            name: {"reference": NEW_IMAGE}
            for name in ("runtime", "node_installer", "dcgm_exporter", "adot")
        },
    }
    if version == 4:
        delivery["image_layout"] = "split-v1"
        delivery["images"]["runtime"]["components"] = {
            "control_plane": dict(components["control_plane"])
        }
        delivery["images"]["executor"] = {
            "reference": NEW_IMAGE,
            "components": {"executor": dict(components["executor"])},
        }
        delivery["images"]["node_dependencies"] = {
            "reference": NEW_IMAGE,
            "wheelhouse_sha256": "7" * 64,
            "dependency_lock_sha256": "8" * 64,
            "tools_lock_sha256": "9" * 64,
        }
    delivery["sha256"] = config.canonical_sha256(delivery)
    return {
        "schema_version": version,
        "deployable": True,
        "delivery": delivery,
    }, components


def reseal(manifest: dict[str, Any]) -> None:
    delivery = manifest["delivery"]
    delivery.pop("sha256", None)
    delivery["sha256"] = config.canonical_sha256(delivery)


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("digest", "identity digest is invalid"),
        ("deployable", "prebuilt deployable"),
        ("prebuilt", "prebuilt deployable"),
        ("components", "components are incomplete"),
        ("component-digest", "component digests are invalid"),
        ("layout", "split-v1"),
        ("dependency-pin", "offline node dependency identity is incomplete"),
        ("wheel-pin", "split image does not match"),
        ("image-components", "split image does not match"),
        ("image-set", "image identity is incomplete"),
        ("mutable-image", "immutable sha256 references"),
        ("template", "node template digest is invalid"),
        ("rollback-schema", "never rollback-compatible"),
        ("schema", "unsupported release manifest schema"),
    ],
)
def test_delivery_identity_rejects_each_unbound_or_mutable_input(
    fault: str, problem: str
) -> None:
    manifest, components = delivery_manifest(3 if fault == "schema" else 4)
    delivery = manifest["delivery"]
    if fault == "digest":
        delivery["sha256"] = "bad"
    elif fault == "deployable":
        manifest["deployable"] = False
    elif fault == "prebuilt":
        delivery["runtime_prebuilt"] = False
    elif fault == "components":
        delivery["components"].pop("cpu")
    elif fault == "component-digest":
        delivery["components"]["cpu"]["sha256"] = "invalid"
    elif fault == "layout":
        delivery["image_layout"] = "shared"
    elif fault == "dependency-pin":
        delivery["images"]["node_dependencies"]["tools_lock_sha256"] = ""
    elif fault == "wheel-pin":
        delivery["images"]["executor"]["components"]["executor"]["wheel_sha256"] = (
            "0" * 64
        )
    elif fault == "image-components":
        delivery["images"]["runtime"]["components"]["executor"] = components["executor"]
    elif fault == "image-set":
        delivery["images"]["extra"] = {"reference": NEW_IMAGE}
    elif fault == "mutable-image":
        delivery["images"]["adot"]["reference"] = "example/adot:latest"
    elif fault == "template":
        components["node_bundle"]["template_sha256"] = ""
    elif fault == "rollback-schema":
        manifest["database"] = {"rollback_compatible": True}
    elif fault == "schema":
        manifest["schema_version"] = 5
    if fault != "digest":
        reseal(manifest)
    with pytest.raises(config.ReleaseError, match=problem):
        config.parse_delivery_identity(manifest, components)


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("roles", [False, True])
def test_delivery_identity_preserves_legacy_role_mapping_and_split_images(
    version: int, roles: bool
) -> None:
    manifest, components = delivery_manifest(version)
    if not roles:
        for name in ("cpu_ingress", "cpu_spool", "cpu_worker"):
            manifest["delivery"]["components"].pop(name)
    components.pop("node_bundle")
    manifest["delivery"]["node_template_inputs"] = {"sha256": "a" * 64}
    reseal(manifest)
    delivery, digest, role_digests, images, template = config.parse_delivery_identity(
        manifest, components
    )
    assert delivery == manifest["delivery"]
    assert digest == manifest["delivery"]["sha256"]
    assert role_digests["cpu_ingress"] == role_digests["cpu"]
    assert template == "a" * 64
    assert ("executor" in images) is (version == 4)
    assert all("@sha256:" in image for image in images.values()), (
        "accepted delivery images must retain immutable digest references"
    )


@pytest.mark.parametrize(
    "field,value,problem",
    [
        ("amp_rule_namespace", "", "must not be empty"),
        ("certificate_min_validity_days", "bad", "must be integers"),
        ("certificate_min_validity_days", 0, "within 1..3650"),
        ("remote_command_max_unclaimed_seconds", 86401, "within 1..86400"),
    ],
)
def test_health_thresholds_are_validated_without_contacting_services(
    field: str, value: Any, problem: str
) -> None:
    with pytest.raises(config.ReleaseError, match=problem):
        config.RegionalHealthConfig.from_mapping(
            {field: value}, expected_region="us-east-1"
        )


@pytest.mark.parametrize(
    "function,value",
    [
        (config.required_email, "invalid"),
        (config.validate_runtime_profile_version, "invalid version"),
        (config.email_subject_prefix, "x" * 65),
        (config.email_subject_prefix, "line\nbreak"),
    ],
)
def test_public_text_parsers_reject_invalid_values(function: Any, value: str) -> None:
    with pytest.raises(config.ReleaseError):
        function(value, "example-field")


@pytest.mark.parametrize("value", [None, "ops@example.com", [], ["bad"]])
def test_email_list_requires_nonempty_valid_addresses(value: Any) -> None:
    with pytest.raises(config.ReleaseError):
        config.email_list(value, "recipients")


@pytest.mark.parametrize("value", [[], "label", ["label", "label"]])
def test_failure_domain_labels_require_distinct_nonempty_list(value: Any) -> None:
    with pytest.raises(config.ReleaseError, match="non-empty list|must be unique"):
        config.failure_domain_labels(value)


@pytest.mark.parametrize(
    "value",
    [
        {"channel": "unknown"},
        {"channel": 1},
        {"channel": "sns", "allow_email": True},
        {"channel": "ses", "admin_email": "ops@example.com"},
        {"allow_email": False, "acknowledge_external_alert_channel": False},
    ],
)
def test_notifications_reject_unknown_channel_or_missing_delivery_identity(
    value: dict[str, Any],
) -> None:
    with pytest.raises(config.ReleaseError, match="notifications"):
        config.RegionalNotificationConfig.from_mapping(value)


@pytest.mark.parametrize(
    "value,problem",
    [
        ({"control_record_retention_days": True}, "integer >= 0"),
        ({"control_record_retention_days": -1}, "integer >= 0"),
        ({"control_record_retention_days": 1}, "is required"),
        ({"archive_s3_uri": "https://example.invalid/archive"}, "s3://"),
        ({"archive_interval_seconds": 0}, "positive integer"),
    ],
)
def test_retention_requires_explicit_valid_archive_configuration(
    value: dict[str, Any], problem: str
) -> None:
    with pytest.raises(config.ReleaseError, match=problem):
        config.RegionalRetentionConfig.from_mapping(value)


@pytest.mark.parametrize(
    "field,value,problem",
    [
        ("upgrade_max_unavailable", "bad", "must be integers"),
        ("upgrade_max_unavailable", 33, "within 0..32"),
        ("rollback_max_unavailable", 0, "within 1..4"),
        ("upgrade_max_parallel_clusters", 9, "within 1..8"),
        ("agent_config_digest", "A" * 64, "lowercase SHA-256"),
    ],
)
def test_release_config_refuses_invalid_rollout_budget_and_digest(
    tmp_path: Path, field: str, value: Any, problem: str
) -> None:
    path = config_file(tmp_path)
    source = json.loads(path.read_text())
    source["release"][field] = value
    path.write_text(json.dumps(source))
    with pytest.raises(config.ReleaseError, match=problem):
        config.ReleaseConfig.load(path)


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("duplicate-cluster", "cluster_id values must be unique"),
        ("missing-profile", "source must be an existing file"),
        ("missing-template", "template_source must be an existing file"),
        ("bad-yaml", "cannot load runtime_profile.source"),
        ("list-yaml", "one RuntimeProfile mapping"),
        ("fields-yaml", "source is missing"),
        ("admin-digest", "config_sha256 does not match"),
        ("admin-roles", "role_sha256 does not match"),
    ],
)
def test_release_config_local_inputs_remain_bound_and_structurally_valid(
    tmp_path: Path, fault: str, problem: str
) -> None:
    path = config_file(tmp_path)
    source = json.loads(path.read_text())
    if fault == "duplicate-cluster":
        source["clusters"].append(copy.deepcopy(source["clusters"][0]))
    elif fault == "missing-profile":
        source["runtime_profile"]["source"] = "missing.yaml"
    elif fault == "missing-template":
        source["runtime_profile"]["template_source"] = "missing.yaml"
    elif fault.endswith("yaml"):
        profile = tmp_path / "profile.yaml"
        profile.write_text(
            {"bad-yaml": "[", "list-yaml": "[]", "fields-yaml": "{}"}[fault]
        )
        source["runtime_profile"]["source"] = profile.name
        source["runtime_profile"]["template_source"] = profile.name
    else:
        admin = default_admin_config()
        source["admin_config"] = {
            "config": admin.as_dict(),
            "config_sha256": admin.sha256(),
            "role_sha256": admin.role_sha256(),
        }
        source["admin_config"][
            "config_sha256" if fault == "admin-digest" else "role_sha256"
        ] = "wrong"
    path.write_text(json.dumps(source))
    with pytest.raises(config.ReleaseError, match=problem):
        config.ReleaseConfig.load(path)


@pytest.mark.parametrize(
    "fault", [None, "missing", "hash", "component", "schema", "protocol"]
)
def test_manifest_artifact_loader_checks_files_hashes_and_versions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str | None
) -> None:
    wheel, bundle = tmp_path / "control.whl", tmp_path / "node.tar"
    wheel.write_bytes(b"example-wheel")
    bundle.write_bytes(b"example-bundle")
    manifest = {
        "schema_version": 2,
        "wheel": wheel.name,
        "bundle": bundle.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "module_digest": "a" * 64,
        "database_schema_version": 17,
        "protocol_versions": {"agent": 3, "executor": 2},
    }
    if fault == "missing":
        bundle.unlink()
    elif fault == "hash":
        manifest["wheel_sha256"] = "b" * 64
    elif fault == "component":
        manifest["module_digest"] = "BAD"
    elif fault == "schema":
        manifest["database_schema_version"] = -1
    elif fault == "protocol":
        manifest["protocol_versions"]["agent"] = 0
    manifest_path = tmp_path / "release-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(config, "ROOT", tmp_path)
    if fault:
        with pytest.raises(
            config.ReleaseError,
            match="must exist|do not match|lowercase SHA-256|must not be negative|must be positive",
        ):
            config.load_release_artifacts(
                {"manifest": manifest_path.name}, config_path=tmp_path / "site.json"
            )
    else:
        artifacts = config.load_release_artifacts(
            {"manifest": manifest_path.name}, config_path=tmp_path / "site.json"
        )
        assert artifacts.wheel == wheel
        assert artifacts.executor_wheel == wheel
        assert artifacts.node_wheel == wheel
        assert artifacts.bundle == bundle
        assert artifacts.release_id == manifest["wheel_sha256"][:12]


def test_nlb_renderer_rejects_missing_configuration_and_leftover_placeholders(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    loaded = config.ReleaseConfig.load(config_file(tmp_path))
    with pytest.raises(config.ReleaseError, match="nlb config is required"):
        config.render_nlb_manifest(loaded, "kind: Service")
    with pytest.raises(config.ReleaseError, match="nlb config is missing"):
        config.render_nlb_manifest(
            replace(loaded, nlb={"name": "example"}), "kind: Service"
        )
    complete = {
        "public_subnets": "subnet-a,subnet-b",
        "security_group": "sg-example",
        "certificate_arn": "arn:aws:acm:us-east-1:123456789012:certificate/example",
    }
    with pytest.raises(config.ReleaseError, match="still contains a placeholder"):
        config.render_nlb_manifest(
            replace(loaded, nlb=complete), "REPLACE_WITH_UNKNOWN"
        )
