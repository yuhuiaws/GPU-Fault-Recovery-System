"""The diagnostic evidence destination is a site setting the managed deploy renders.

Live 2026-10-25 (hung-triage hand-off mail): every evidence archive was a
``file:///var/lib/gpu-fault/...`` reference on the faulting node. The Node
Agent uploads to S3 only when ``GPU_FAULT_DIAGNOSTIC_S3_URI`` is set
(installer ``--diagnostic-s3-uri``), and the managed deploy had no way to set
it: the release engine never put the variable into the reconciler
environment, so every managed site ran with node-local evidence. The setting
is now ``site.yaml`` ``spec.diagnosticS3Uri``: site -> release config
(``diagnostic_s3_uri``) -> reconciler environment -> installer Job template ->
node ``GPU_FAULT_DIAGNOSTIC_S3_URI``; the same value enters the Agent config
digest the deploy pins (``agent_config_payload`` carries ``diagnostic_s3_uri``),
and a regenerated site keeps the operator's declaration.
"""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped,unused-ignore]

from gpu_fault.admin import bootstrap_common
from gpu_fault.admin.bootstrap_site import preserve_existing_site_contract
from gpu_fault.admin.site import RegionalSite, SiteConfigError, load_site
from gpu_fault.admin.site_s3 import (
    DIAGNOSTIC_S3_URI_ENV,
    node_agent_environment,
    site_diagnostic_s3_uri,
)
from gpu_fault_release import regional_release_config as release_config_module
from gpu_fault_release import regional_release_rendering as rendering
from gpu_fault_release import rollout as release_module
from tests.admin.test_admin_site import site_file as write_site_file
from tests.regional._release_orchestrator_support import config_file
from tests.regional.test_regional_release_eks_managed import (
    RecordingRunner,
    _hyperpod_cluster,
)

URI = "s3://gpu-fault-evidence-123456789012/site-a/diagnostics"


@pytest.fixture
def site_file(tmp_path: Path) -> Path:
    return write_site_file(tmp_path)


def _document(site_file: Path, uri: object = None, *, present: bool = True) -> dict:
    document = yaml.safe_load(site_file.read_text(encoding="utf-8"))
    if present:
        document["spec"]["diagnosticS3Uri"] = uri
    site_file.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return document


def test_the_site_renders_the_destination_into_the_release_config(
    site_file: Path,
) -> None:
    _document(site_file, URI)

    rendered = load_site(site_file)

    assert rendered.release_config["diagnostic_s3_uri"] == URI
    assert (
        RegionalSite.from_value(
            yaml.safe_load(site_file.read_text())
        ).spec.diagnostic_s3_uri
        == URI
    )
    assert node_agent_environment(URI) == {DIAGNOSTIC_S3_URI_ENV: URI}
    assert DIAGNOSTIC_S3_URI_ENV == "GPU_FAULT_DIAGNOSTIC_S3_URI", (
        "the node Agent reads exactly this variable (node_agent/config.py)"
    )


def test_a_site_without_the_setting_renders_byte_for_byte_as_before(
    site_file: Path,
) -> None:
    """Default empty: no key enters the release config (its bytes feed identity
    digests), the Agent environment gains nothing, and a raw document reads None."""

    rendered = load_site(site_file)

    assert "diagnostic_s3_uri" not in rendered.release_config
    assert rendered.release_config.get("diagnostic_s3_uri") is None
    assert node_agent_environment(None) == {}
    assert site_diagnostic_s3_uri(yaml.safe_load(site_file.read_text())) is None
    assert site_diagnostic_s3_uri(None) is None


@pytest.mark.parametrize(
    "value",
    [
        "https://gpu-fault-evidence/site-a",
        "s3://",
        "s3:///prefix-only",
        "s3://Bucket_Name/prefix",
        "s3://b/prefix",
        "",
        "   ",
        3,
        ["s3://bucket/prefix"],
    ],
)
def test_the_destination_must_be_an_s3_bucket_uri(
    site_file: Path, value: object
) -> None:
    document = _document(site_file, value)

    with pytest.raises(SiteConfigError, match="spec.diagnosticS3Uri"):
        RegionalSite.from_value(document)


def test_a_bucket_alone_is_a_valid_destination_and_a_trailing_slash_is_dropped(
    site_file: Path,
) -> None:
    for declared, expected in (
        ("s3://gpu-fault-evidence", "s3://gpu-fault-evidence"),
        ("s3://gpu-fault-evidence/", "s3://gpu-fault-evidence"),
        ("s3://gpu-fault-evidence/site-a/", "s3://gpu-fault-evidence/site-a"),
    ):
        document = _document(site_file, declared)
        assert RegionalSite.from_value(document).spec.diagnostic_s3_uri == expected, (
            declared
        )


def test_a_regenerated_site_keeps_the_declared_destination() -> None:
    """``deploy`` re-renders site.yaml; the operator's declaration must survive
    like ``autoRollback`` and ``retention`` do."""

    generated = {"spec": {"autoRollback": True, "clusters": []}}
    existing = {"spec": {"diagnosticS3Uri": URI, "clusters": []}}

    kept = preserve_existing_site_contract(copy.deepcopy(generated), existing)
    assert kept["spec"]["diagnosticS3Uri"] == URI

    absent = preserve_existing_site_contract(
        copy.deepcopy(generated), {"spec": {"clusters": []}}
    )
    assert "diagnosticS3Uri" not in absent["spec"], (
        "a site that never declared one must not gain the key"
    )
    with pytest.raises(bootstrap_common.BootstrapError, match="diagnosticS3Uri"):
        preserve_existing_site_contract(
            copy.deepcopy(generated), {"spec": {"diagnosticS3Uri": 7, "clusters": []}}
        )


def _release_config(tmp_path: Path, **extra: Any) -> Path:
    path = config_file(tmp_path, clusters=[_hyperpod_cluster(tmp_path)])
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(extra)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_the_release_config_reads_and_validates_the_destination(tmp_path: Path) -> None:
    config = release_config_module.ReleaseConfig.load(
        _release_config(tmp_path, diagnostic_s3_uri=URI)
    )
    assert config.diagnostic_s3_uri == URI

    assert (
        release_config_module.ReleaseConfig.load(
            _release_config(tmp_path)
        ).diagnostic_s3_uri
        is None
    )

    for invalid in ("https://bucket/prefix", "s3://", "s3://Bucket/prefix", 7):
        with pytest.raises(
            release_config_module.ReleaseError, match="diagnostic_s3_uri"
        ):
            release_config_module.ReleaseConfig.load(
                _release_config(tmp_path, diagnostic_s3_uri=invalid)
            )


def _reconciler_environment(tmp_path: Path, **extra: Any) -> dict[str, str]:
    release = release_module.RegionalRelease(
        release_config_module.ReleaseConfig.load(_release_config(tmp_path, **extra)),
        RecordingRunner(),
    )
    (target,) = release.config.clusters
    return rendering.build_reconciler_environment(
        release,
        target,
        wheel_cm="wheel-cm",
        bundle_cm="bundle-cm",
        artifact_sha="a" * 64,
        config_digest="b" * 64,
    )


def test_the_reconciler_environment_carries_the_destination_to_the_node_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``deploy-node-installer-reconciler.sh`` renders the installer Job through
    ``run-hyperpod-installer-job.sh``, which reads ``GPU_FAULT_DIAGNOSTIC_S3_URI``
    and hands it to the node installer as ``--diagnostic-s3-uri``."""

    monkeypatch.delenv(DIAGNOSTIC_S3_URI_ENV, raising=False)

    environment = _reconciler_environment(tmp_path, diagnostic_s3_uri=URI)
    assert environment[DIAGNOSTIC_S3_URI_ENV] == URI

    unset = _reconciler_environment(tmp_path)
    assert DIAGNOSTIC_S3_URI_ENV not in unset, (
        "a site without the setting renders the reconciler environment as before"
    )
    assert DIAGNOSTIC_S3_URI_ENV in rendering.INSTALLER_PRODUCT_INPUT_ENV, (
        "a changed destination must not reuse the previous render's products"
    )


def test_the_site_destination_wins_over_a_stray_deploy_host_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DIAGNOSTIC_S3_URI_ENV, "s3://stray-bucket/elsewhere")

    environment = _reconciler_environment(tmp_path, diagnostic_s3_uri=URI)

    assert environment[DIAGNOSTIC_S3_URI_ENV] == URI


def test_the_agent_config_digest_is_computed_with_the_destination(
    tmp_path: Path,
) -> None:
    """``agent_config_payload`` carries ``diagnostic_s3_uri``: the pin the deploy
    writes (``spec.release.agentConfigDigest``) must be computed with the same
    value the nodes will run with, or the fleet consistency gate fails every
    node-owned step after the wave."""

    installer_environment = {
        key: f"/opt/gpu-fault/{key.lower()}"
        for key in bootstrap_common.NODE_INSTALLER_CONFIG_DIGEST_ENVIRONMENT_KEYS
    }
    seen: list[dict[str, str]] = []

    class Runner:
        def run(self, arguments, **kwargs):
            if arguments[0] == "bash":
                return json.dumps(installer_environment)
            seen.append(dict(kwargs["env"]))
            return "c" * 64

    digest = bootstrap_common.compute_agent_config_digest(
        Runner(),
        repository_root=tmp_path,
        runtime_profile_version="profile-a",
        node_agent_environment={DIAGNOSTIC_S3_URI_ENV: URI},
    )

    assert digest == "c" * 64
    assert seen[0][DIAGNOSTIC_S3_URI_ENV] == URI
    for key, value in installer_environment.items():
        assert seen[0][key] == value, "the installer's own digest inputs still win"


@pytest.mark.parametrize(
    "environment",
    [
        {"PATH": "/evil"},
        {DIAGNOSTIC_S3_URI_ENV: "first\nsecond"},
        {DIAGNOSTIC_S3_URI_ENV: ""},
        {"GPU_FAULT_PYTHON_STACK_TOOL": "/elsewhere/py-spy"},
    ],
    ids=["foreign-key", "newline", "empty", "installer-key"],
)
def test_the_agent_digest_refuses_an_invalid_node_agent_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, environment: dict[str, str]
) -> None:
    installer_environment = {
        key: "example"
        for key in bootstrap_common.NODE_INSTALLER_CONFIG_DIGEST_ENVIRONMENT_KEYS
    }
    monkeypatch.setattr(
        bootstrap_common,
        "run_command",
        lambda arguments, **_options: subprocess.CompletedProcess(
            arguments, 0, json.dumps(installer_environment), ""
        ),
    )

    with pytest.raises(bootstrap_common.BootstrapError, match="node Agent environment"):
        bootstrap_common.compute_agent_config_digest(
            bootstrap_common.CommandRunner(),
            repository_root=tmp_path,
            runtime_profile_version="profile-a",
            node_agent_environment=environment,
        )
