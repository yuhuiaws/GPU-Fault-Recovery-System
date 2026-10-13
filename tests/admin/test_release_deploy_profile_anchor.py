"""The site base snapshot after remove-cluster moved the registration anchor.

``profiles/<version>.yaml`` is the immutable record of what a version was
registered as, ``cluster_id`` (its registration anchor) included. When the
registration cluster leaves, remove-cluster moves
``spec.runtimeProfile.registrationClusterId`` to a remaining member of its
provider; the next release attempt planned the unchanged base under the new
anchor, rendered a snapshot that differed from the recorded one in that label
only, and refused: "immutable Runtime Profile snapshot differs".
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts import release_deploy
from tests.admin.test_release_deploy import _site

ROOT = Path(__file__).resolve().parents[2]
HYPERPOD_TEMPLATE = "config/runtime-profile.regional-hyperpod-safe.example.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _document(site: Path) -> dict[str, Any]:
    return yaml.safe_load(site.read_text(encoding="utf-8"))


def _released_hyperpod_site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A HyperPod site anchored on ``gpu-a`` whose first release wrote the
    declared ``hyperpod-v1`` snapshot and moved the source to it."""

    site = _site(tmp_path, monkeypatch)
    repo = tmp_path / "repo"
    (repo / HYPERPOD_TEMPLATE).write_bytes((ROOT / HYPERPOD_TEMPLATE).read_bytes())
    document = _document(site)
    document["spec"]["runtimeProfile"] = {
        "source": HYPERPOD_TEMPLATE,
        "templateSource": HYPERPOD_TEMPLATE,
        "version": "hyperpod-v1",
        "registrationClusterId": "gpu-a",
    }
    site.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    plan = release_deploy.plan_runtime_profile(
        site, live_profile_sha256=_sha256(repo / HYPERPOD_TEMPLATE)
    )
    release_deploy.prepare_site_release(site, profile_plan=plan)
    snapshot = site.parent / "profiles/hyperpod-v1.yaml"
    assert yaml.safe_load(snapshot.read_text())["cluster_id"] == "gpu-a", (
        "the first release records the anchor it registered under"
    )
    return site


def _move_anchor_to_gpu_b(site: Path) -> None:
    """What remove-cluster of ``gpu-a`` leaves: ``gpu-b`` alone, anchored."""

    document = _document(site)
    spec = document["spec"]
    (first,) = spec["clusters"]
    spec["clusters"] = [
        {
            **first,
            "clusterId": "gpu-b",
            "context": "gpu-b",
            "hyperpodClusterName": "hp-gpu-b",
            "eksClusterArn": "arn:aws:eks:us-east-1:123456789012:cluster/gpu-b",
        }
    ]
    spec["runtimeProfile"]["registrationClusterId"] = "gpu-b"
    site.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def test_the_unchanged_base_keeps_its_snapshot_after_the_anchor_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _released_hyperpod_site(tmp_path, monkeypatch)
    snapshot = site.parent / "profiles/hyperpod-v1.yaml"
    before = snapshot.read_bytes()
    _move_anchor_to_gpu_b(site)

    plan = release_deploy.plan_runtime_profile(
        site, live_profile_sha256=_sha256(snapshot)
    )
    prepared = release_deploy.prepare_site_release(site, profile_plan=plan)

    assert (plan.change_kind, plan.desired_version) == ("UNCHANGED", "hyperpod-v1")
    assert plan.approval_required is False, "a moved anchor is no policy change"
    assert plan.registration_cluster_id == "gpu-b", "the plan names the site anchor"
    assert plan.snapshot_document["cluster_id"] == "gpu-a", (
        "the immutable snapshot keeps the anchor the version was registered under"
    )
    assert plan.snapshot_sha256 == _sha256(snapshot), (
        "the planned snapshot is the recorded file, byte for byte"
    )
    assert snapshot.read_bytes() == before, "the immutable snapshot is untouched"
    assert prepared.runtime_profile_version == "hyperpod-v1"
    assert _document(site)["spec"]["runtimeProfile"]["source"] == str(snapshot)


def test_a_snapshot_with_other_policy_is_still_refused_after_the_anchor_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keeping the recorded anchor must not hide a policy difference: a site on
    its template whose ``hyperpod-v1`` snapshot was changed is refused."""

    site = _released_hyperpod_site(tmp_path, monkeypatch)
    snapshot = site.parent / "profiles/hyperpod-v1.yaml"
    _move_anchor_to_gpu_b(site)
    changed = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    changed["claims"] = changed["claims"][1:]
    snapshot.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
    document = _document(site)
    document["spec"]["runtimeProfile"]["source"] = HYPERPOD_TEMPLATE
    site.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    template = tmp_path / "repo" / HYPERPOD_TEMPLATE

    plan = release_deploy.plan_runtime_profile(
        site, live_profile_sha256=_sha256(template)
    )

    assert (plan.change_kind, plan.desired_version) == ("UNCHANGED", "hyperpod-v1")
    with pytest.raises(
        release_deploy.ReleaseDeployError,
        match="immutable Runtime Profile snapshot differs",
    ):
        release_deploy.prepare_site_release(site, profile_plan=plan)
