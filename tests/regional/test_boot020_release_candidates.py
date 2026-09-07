"""The BOOT-020 config chain steps exactly one classification at a time."""

from __future__ import annotations

import json
from pathlib import Path

from gpu_fault.admin.config import AdminConfig
from scripts.e2e.regional.boot020_release_candidates import (
    candidate_edits,
    chain_configs,
    edits_digest,
    link_release_dist,
    parser,
    reusable_candidate,
    unlink_release_dist,
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
        _base(),
        live_manifest="/live/m.json",
        candidate_manifests=manifests,
        full_agent_config_digest="f" * 64,
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
    # The Agent digest follows the profile version; earlier configs keep the live one.
    assert chain["full"]["release"]["agent_config_digest"] == "f" * 64
    assert "agent_config_digest" not in chain["agent"]["release"]


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


def test_edits_digest_distinguishes_candidates_and_modules() -> None:
    edits = candidate_edits("src/gpu_fault/x.py", "src/gpu_fault/node_agent/y.py")
    other = candidate_edits("src/gpu_fault/z.py", "src/gpu_fault/node_agent/y.py")

    assert edits_digest(edits["B"]) == edits_digest(edits["B"])
    assert len({edits_digest(edits[name]) for name in ("B", "C", "D")}) == 3
    assert edits_digest(edits["B"]) != edits_digest(other["B"])


def test_reusable_candidate_requires_the_recorded_edits_digest(tmp_path: Path) -> None:
    manifest = tmp_path / "wt-B" / "dist" / "current-release.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"release_id": "rid-b"}), "utf-8")
    recorded = {"release_id": "rid-b", "edits_sha256": "d" * 64}

    assert reusable_candidate(
        manifest,
        base_release_id="rid-base",
        recorded=recorded,
        expected_edits_digest="d" * 64,
    ), "a manifest built from the recorded edits is the candidate asked for"
    # The same checkout built from different edits is not the candidate asked for.
    assert not reusable_candidate(
        manifest,
        base_release_id="rid-base",
        recorded=recorded,
        expected_edits_digest="e" * 64,
    ), "different edits digest means a different candidate"
    # A build that never diverged from the base, or one nobody recorded, is rebuilt.
    assert not reusable_candidate(
        manifest,
        base_release_id="rid-b",
        recorded=recorded,
        expected_edits_digest="d" * 64,
    ), "a build that never diverged from the base is not a candidate"
    assert not reusable_candidate(
        manifest,
        base_release_id="rid-base",
        recorded=None,
        expected_edits_digest="d" * 64,
    ), "an unrecorded build is rebuilt"
    assert not reusable_candidate(
        tmp_path / "missing.json",
        base_release_id="rid-base",
        recorded=recorded,
        expected_edits_digest="d" * 64,
    ), "a missing manifest is rebuilt"


def test_unlink_release_dist_removes_only_the_link_it_made(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "dist").mkdir(parents=True)
    candidate = tmp_path / "wt-B" / "dist"
    (candidate / "rid123").mkdir(parents=True)
    manifest = candidate / "current-release.json"
    manifest.write_text(json.dumps({"release_id": "rid123"}), "utf-8")
    link = link_release_dist(repo, manifest)

    assert unlink_release_dist(repo, manifest) == link
    assert not link.exists() and not link.is_symlink()
    assert unlink_release_dist(repo, manifest) is None, "idempotent"

    # A real directory of the same name was not made by this tool.
    link.mkdir()
    assert unlink_release_dist(repo, manifest) is None
    assert link.is_dir(), "a real directory of the same name is left in place"

    # A link to a different target is left alone as well.
    link.rmdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    link.symlink_to(elsewhere)
    assert unlink_release_dist(repo, manifest) is None
    assert link.is_symlink(), "a link to another target is left in place"


def test_clean_is_a_subcommand() -> None:
    arguments = parser().parse_args(
        ["clean", "--snapshot-repo", "/tmp/snap", "--work-dir", "/tmp/work"]
    )
    assert arguments.command == "clean"
