"""The BOOT-020 config chain steps exactly one classification at a time."""

from __future__ import annotations

import json
from pathlib import Path

from gpu_fault.admin.config import AdminConfig
from scripts.e2e.regional.boot020_release_candidates import (
    candidate_edits,
    chain_configs,
    link_release_dist,
)


def _base() -> dict:
    admin = {"schema_version": 1, "capacity": {"control_worker_replicas": 6}}
    parsed = AdminConfig.from_mapping(admin)
    return {
        "release": {"manifest": "dist/current-release.json"},
        "runtime_profile": {"version": "regional-hyperpod-abc123"},
        "admin_config": {
            "config": parsed.as_dict(),
            "config_sha256": parsed.sha256(),
            "role_sha256": parsed.role_sha256(),
        },
    }


def test_chain_changes_one_dimension_per_step() -> None:
    manifests = {
        "B": "/wt-B/dist/m.json",
        "C": "/wt-C/dist/m.json",
        "D": "/wt-D/dist/m.json",
    }
    chain = chain_configs(
        _base(), live_manifest="/live/m.json", candidate_manifests=manifests
    )

    assert list(chain) == ["noop", "control-plane", "data-plane", "agent", "full"]
    assert chain["noop"]["release"]["manifest"] == "/live/m.json"
    assert chain["noop"]["admin_config"] == _base()["admin_config"]

    control = chain["control-plane"]
    assert control["admin_config"]["config"]["capacity"]["control_worker_replicas"] == 7
    assert (
        control["admin_config"]["role_sha256"]["worker"]
        != _base()["admin_config"]["role_sha256"]["worker"]
    )
    assert (
        control["admin_config"]["role_sha256"]["ingress"]
        == _base()["admin_config"]["role_sha256"]["ingress"]
    )
    assert control["release"]["manifest"] == "/live/m.json"

    assert chain["data-plane"]["release"]["manifest"] == manifests["B"]
    assert chain["data-plane"]["admin_config"] == control["admin_config"]
    assert chain["agent"]["release"]["manifest"] == manifests["C"]
    assert chain["agent"]["runtime_profile"]["version"] == "regional-hyperpod-abc123"
    assert chain["full"]["release"]["manifest"] == manifests["D"]
    assert (
        chain["full"]["runtime_profile"]["version"]
        == "regional-hyperpod-abc123-boot020"
    )


def test_candidate_edits_touch_only_the_intended_modules() -> None:
    edits = candidate_edits("src/gpu_fault/x.py", "src/gpu_fault/node_agent/y.py")
    assert [path for path, _ in edits["B"]] == ["src/gpu_fault/x.py"]
    assert [path for path, _ in edits["C"]] == [
        "src/gpu_fault/x.py",
        "src/gpu_fault/node_agent/y.py",
    ]
    # D differs from C in both files, so both wheels move again for FULL.
    assert (
        dict(edits["D"])["src/gpu_fault/x.py"] != dict(edits["C"])["src/gpu_fault/x.py"]
    )
    assert (
        dict(edits["D"])["src/gpu_fault/node_agent/y.py"]
        != dict(edits["C"])["src/gpu_fault/node_agent/y.py"]
    )


def test_link_release_dist_points_repo_dist_at_the_candidate(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "dist").mkdir(parents=True)
    candidate = tmp_path / "wt-B" / "dist"
    (candidate / "rid123").mkdir(parents=True)
    manifest = candidate / "current-release.json"
    manifest.write_text(json.dumps({"release_id": "rid123"}), "utf-8")

    link = link_release_dist(repo, manifest)

    assert link == repo / "dist" / "rid123"
    assert link.resolve() == (candidate / "rid123").resolve()
    assert link_release_dist(repo, manifest) == link
