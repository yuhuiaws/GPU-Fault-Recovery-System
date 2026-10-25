"""``prepare_site_release`` pins the Agent config digest with the site's node
declarations.

``spec.diagnosticS3Uri`` reaches every GPU node's Agent as its diagnostic S3
destination, and ``agent_config_payload`` folds that value into the digest the
control plane requires of every Agent; the pin the deploy writes into
``spec.release.agentConfigDigest`` must therefore be computed with it, and the
rewrite of ``spec.release`` must keep the operator's declaration.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts import release_deploy
from tests.admin.test_release_deploy import _site


def test_prepare_site_release_pins_the_digest_with_the_site_diagnostic_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``spec.diagnosticS3Uri`` enters the Agent config digest the deploy pins
    (``agent_config_payload`` carries ``diagnostic_s3_uri``), and the rewrite
    of ``spec.release`` keeps the operator's declaration."""

    site = _site(tmp_path, monkeypatch)
    uri = "s3://gpu-fault-evidence-123456789012/test-site/diagnostics"
    document = yaml.safe_load(site.read_text(encoding="utf-8"))
    document["spec"]["diagnosticS3Uri"] = uri
    site.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    seen: list[dict[str, Any]] = []

    def digest(_runner: object, **kwargs: Any) -> str:
        seen.append(kwargs)
        return "b" * 64

    monkeypatch.setattr(release_deploy, "compute_agent_config_digest", digest)
    profile = tmp_path / "repo/config/profile.yaml"
    plan = release_deploy.plan_runtime_profile(
        site, live_profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest()
    )

    release_deploy.prepare_site_release(site, profile_plan=plan)

    assert seen and seen[0]["node_agent_environment"] == {
        "GPU_FAULT_DIAGNOSTIC_S3_URI": uri
    }, seen
    assert seen[0]["runtime_profile_version"] == "profile-v1"
    rewritten = yaml.safe_load(site.read_text(encoding="utf-8"))
    assert rewritten["spec"]["diagnosticS3Uri"] == uri
    assert rewritten["spec"]["release"]["agentConfigDigest"] == "b" * 64


def test_prepare_site_release_pins_the_digest_without_a_destination_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _site(tmp_path, monkeypatch)
    seen: list[dict[str, Any]] = []

    def digest(_runner: object, **kwargs: Any) -> str:
        seen.append(kwargs)
        return "b" * 64

    monkeypatch.setattr(release_deploy, "compute_agent_config_digest", digest)
    profile = tmp_path / "repo/config/profile.yaml"
    plan = release_deploy.plan_runtime_profile(
        site, live_profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest()
    )

    release_deploy.prepare_site_release(site, profile_plan=plan)

    assert seen and seen[0]["node_agent_environment"] == {}, seen
