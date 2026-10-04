# ruff: noqa: F401, F811
"""Installer product records and the reconciler environment's pin refusals.

``recorded_installer_products`` is advisory: every mismatch (no state, another
release, other inputs, a malformed digest) yields ``None`` rather than a hint.
``build_reconciler_environment`` must refuse a template content pin without a
template and a rollback template whose captured pin disagrees, and a CPU
manifest renderer must never let a ``REPLACE_WITH_*`` placeholder through.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault_release import regional_release_rendering as RENDERING
from gpu_fault_release.regional_release_config import ReleaseError
from gpu_fault_release.regional_release_images import NodeDependencyTarget
from tests.regional.test_release_node_image_targets import (
    ARTIFACT,
    BUNDLE,
    CONFIG,
    TEMPLATE,
    node_release,
)

TARGET = SimpleNamespace(cluster_id="gpu-a")
DIGEST = "1" * 64
NODE_SET = "2" * 64


def record(**overrides: Any) -> dict[str, Any]:
    return {
        "release_id": "rel-1",
        "inputs_sha256": DIGEST,
        "node_set_sha256": NODE_SET,
        **overrides,
    }


def release_with(state: Any, release_id: str = "rel-1") -> SimpleNamespace:
    return SimpleNamespace(state=state, release_id=release_id)


@pytest.mark.parametrize(
    "state,release_id",
    [
        (None, "rel-1"),
        ({}, "rel-1"),
        ({"node_installer_products": {"gpu-a": "not-a-record"}}, "rel-1"),
        ({"node_installer_products": {"gpu-a": record()}}, "rel-2"),
        (
            {"node_installer_products": {"gpu-a": record(inputs_sha256="9" * 64)}},
            "rel-1",
        ),
        (
            {"node_installer_products": {"gpu-a": record(node_set_sha256="short")}},
            "rel-1",
        ),
    ],
    ids=[
        "no-state",
        "no-record",
        "malformed",
        "other-release",
        "other-inputs",
        "digest",
    ],
)
def test_recorded_products_are_a_hint_only_for_this_release_and_these_inputs(
    state: Any, release_id: str
) -> None:
    assert (
        RENDERING.recorded_installer_products(
            release_with(state, release_id), TARGET, inputs_digest=DIGEST
        )
        is None
    )


def test_recorded_products_return_the_matching_record() -> None:
    state = {"node_installer_products": {"gpu-a": record()}}
    assert (
        RENDERING.recorded_installer_products(
            release_with(state), TARGET, inputs_digest=DIGEST
        )
        == record()
    )


def test_record_products_needs_a_state_mapping_and_a_json_object(
    tmp_path: Path,
) -> None:
    report = tmp_path / "products.json"
    report.write_text("[]", encoding="utf-8")
    assert (
        RENDERING.record_installer_products(
            release_with(None), TARGET, inputs_digest=DIGEST, products_path=str(report)
        )
        is None
    )
    state: dict[str, Any] = {}
    assert (
        RENDERING.record_installer_products(
            release_with(state), TARGET, inputs_digest=DIGEST, products_path=str(report)
        )
        is None
    )
    assert state == {}, "a malformed report records nothing"


def test_cpu_manifest_with_an_unknown_placeholder_is_refused() -> None:
    release = SimpleNamespace(
        config=SimpleNamespace(
            namespace="gpu-fault-system",
            aws_region="us-east-1",
            runtime_profile_version="profile-1",
            notifications=SimpleNamespace(
                allow_email=False,
                acknowledge_external_alert_channel=False,
                ses_configuration_set=None,
            ),
        ),
        wheel_cm="gpu-fault-control-plane-wheel-abcd",
        runtime_image="registry.example/runtime@sha256:" + "a" * 64,
        wheel_sha="b" * 64,
    )
    text = "region: REPLACE_WITH_AWS_REGION\nbucket: REPLACE_WITH_ARCHIVE_URI\n"
    with pytest.raises(ReleaseError, match="example.yaml still contains a placeholder"):
        RENDERING.render_cpu_manifest_text(release, "example.yaml", text)


def environment(node_release: SimpleNamespace, **options: Any) -> dict[str, str]:
    return RENDERING.build_reconciler_environment(
        node_release.release,
        node_release.target,
        wheel_cm="candidate-wheel",
        bundle_cm="shared-bundle",
        artifact_sha=ARTIFACT,
        config_digest=CONFIG,
        **options,
    )


def test_fleet_master_file_is_handed_to_the_reconciler_with_the_cpu_kubeconfig(
    node_release: SimpleNamespace,
) -> None:
    node_release.target.fleet_master_file = "/secure/fleet-master"
    node_release.release.config.cpu_kubeconfig = "/secure/cpu.kubeconfig"
    values = environment(node_release)
    assert values["GPU_FAULT_FLEET_MASTER_FILE"] == "/secure/fleet-master"
    assert values["GPU_FAULT_CONTROL_PLANE_KUBECONFIG"] == "/secure/cpu.kubeconfig"


def test_a_content_pin_without_a_template_is_refused(
    node_release: SimpleNamespace,
) -> None:
    with pytest.raises(ReleaseError, match="requires an explicit template"):
        environment(node_release, template_content_sha256="f" * 64)


def test_rollback_template_must_match_its_captured_content_pin(
    node_release: SimpleNamespace,
) -> None:
    with pytest.raises(ReleaseError, match="disagrees with captured content pin"):
        environment(
            node_release,
            bundle_sha256=BUNDLE,
            template_sha256=TEMPLATE,
            template_config_map="previous-template",
            template_content_sha256="0" * 64,
            node_dependency_target=NodeDependencyTarget.PREVIOUS,
        )


def test_rollback_template_with_the_captured_pin_is_accepted(
    node_release: SimpleNamespace,
) -> None:
    captured = node_release.previous["clusters"]["gpu-a"]["template_content_sha256"]
    values = environment(
        node_release,
        bundle_sha256=BUNDLE,
        template_sha256=TEMPLATE,
        template_config_map="previous-template",
        template_content_sha256=captured,
        node_dependency_target=NodeDependencyTarget.PREVIOUS,
    )
    assert values["GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP"] == "previous-template"
    assert values["GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256"] == captured
