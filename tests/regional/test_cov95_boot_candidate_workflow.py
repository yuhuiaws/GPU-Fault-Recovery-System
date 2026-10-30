from __future__ import annotations

import hashlib
import json
import stat
import sys
from argparse import Namespace
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import bootstrap_common, site
from gpu_fault.admin.config import default_admin_config
from gpu_fault.admin.site_s3 import DIAGNOSTIC_S3_URI_ENV
from scripts.e2e.regional import boot020_release_candidates as candidates
from tests.regional._cov95_boot_candidates import CandidateBuild


def test_candidate_build_reuses_only_matching_recorded_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = CandidateBuild(tmp_path, monkeypatch)
    result = candidates.build_candidates(fixture.arguments)
    assert result["base_release_id"] == "base"
    assert len(fixture.build_calls) == 3
    assert len(fixture.git_calls) == 6
    assert all(call["staging_only"] is True for call in fixture.build_calls), (
        "candidate preparation lost its staging-only build restriction"
    )
    assert [arguments[1] for arguments, _kwargs in fixture.git_calls] == [
        "add",
        "-c",
    ] * 3
    components = {
        name: value["components"] for name, value in result["candidates"].items()
    }
    assert {item["control_plane"] for item in components.values()} == {"c" * 12}
    assert components["B"]["executor"] == components["C"]["executor"]
    assert components["C"]["executor"] != components["D"]["executor"]
    assert len({item["node_runtime"] for item in components.values()}) == 3
    assert candidates.build_candidates(fixture.arguments) == result
    assert len(fixture.build_calls) == 3
    summary = fixture.arguments.work_dir / "candidates.json"
    recorded = json.loads(summary.read_text())
    recorded["candidates"]["B"]["edits_sha256"] = "0" * 64
    summary.write_text(json.dumps(recorded))
    candidates.build_candidates(fixture.arguments)
    assert len(fixture.build_calls) == 4
    assert fixture.build_calls[-1]["repository_root"].name == "wt-B"
    assert (
        json.loads(
            (fixture.arguments.snapshot_repo / "dist/current-release.json").read_text()
        )
        == fixture.base
    )


DIAGNOSTIC_S3_URI = "s3://gpu-fault-evidence-unit/site/diagnostics"


def _write_candidate_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    diagnostic_s3_uri: str | None,
    runner: Callable[[], Any] | None = None,
) -> tuple[CandidateBuild, Namespace, Path, dict[str, Any]]:
    """Build B/C/D, then run ``configs`` against a site with or without a
    declared ``spec.diagnosticS3Uri``; returns the build, the arguments, the
    repository the dist links went to and the base release config. ``runner``
    replaces the CommandRunner ``configs`` computes the FULL digest with."""

    fixture = CandidateBuild(tmp_path, monkeypatch)
    candidates.build_candidates(fixture.arguments)
    if runner is not None:
        monkeypatch.setattr(bootstrap_common, "CommandRunner", runner)
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(candidates, "ROOT", repository)
    parsed = default_admin_config()
    base: dict[str, Any] = {
        "release": {"manifest": "old-manifest", "agent_config_digest": "a" * 64},
        "runtime_profile": {"version": "unit-profile"},
        "admin_config": {
            "config": parsed.as_dict(),
            "config_sha256": parsed.sha256(),
            "role_sha256": parsed.role_sha256(),
        },
    }
    if diagnostic_s3_uri is not None:
        # admin.site writes the key only when the site declares one.
        base["diagnostic_s3_uri"] = diagnostic_s3_uri
    monkeypatch.setattr(
        site,
        "load_site",
        lambda *_args, **_kwargs: SimpleNamespace(release_config=base),
    )
    arguments = Namespace(
        **{
            **vars(fixture.arguments),
            "site": tmp_path / "site.yaml",
            "out_dir": tmp_path / "configs",
            "profile_suffix": "-candidate",
            "replicas_delta": -1,
        }
    )
    candidates.write_configs(arguments)
    return fixture, arguments, repository, base


@pytest.mark.parametrize("diagnostic_s3_uri", [None, DIAGNOSTIC_S3_URI])
def test_candidate_config_workflow_writes_private_chained_inputs_and_cleans_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, diagnostic_s3_uri: str | None
) -> None:
    digests = []

    def digest(_runner: Any, **kwargs: Any) -> str:
        digests.append(kwargs)
        return ("e" if kwargs.get("node_agent_environment") else "f") * 64

    monkeypatch.setattr(bootstrap_common, "compute_agent_config_digest", digest)
    fixture, arguments, repository, base = _write_candidate_configs(
        tmp_path, monkeypatch, diagnostic_s3_uri=diagnostic_s3_uri
    )
    documents = {
        name: json.loads((arguments.out_dir / f"{name}.json").read_text())
        for name in candidates.CONFIG_NAMES
    }
    for name in candidates.CONFIG_NAMES:
        assert (
            stat.S_IMODE((arguments.out_dir / f"{name}.json").stat().st_mode) == 0o600
        )
    pinned = ("e" if diagnostic_s3_uri else "f") * 64
    assert documents["noop"]["runtime_profile"]["version"] == "unit-profile"
    assert documents["full"]["runtime_profile"]["version"] == "unit-profile-candidate"
    assert documents["full"]["release"]["agent_config_digest"] == pinned
    assert documents["agent"]["release"]["agent_config_digest"] == "a" * 64
    assert documents["control-plane"]["admin_config"]["config"]["capacity"][
        "control_worker_replicas"
    ] == (base["admin_config"]["config"]["capacity"]["control_worker_replicas"] - 1)
    # Computed like scripts/release_deploy.py pins it: with the site's node-side
    # declarations, which the Agent hashes too (live 2026-10-29).
    assert digests == [
        {
            "repository_root": fixture.arguments.work_dir / "wt-D",
            "runtime_profile_version": "unit-profile-candidate",
            "node_agent_environment": (
                {DIAGNOSTIC_S3_URI_ENV: diagnostic_s3_uri} if diagnostic_s3_uri else {}
            ),
        }
    ]
    handoff = json.loads((arguments.out_dir / "final-noop.json").read_text())
    assert handoff == documents["full"], "BOOT-023's handoff is the FULL config"
    assert len(list((repository / "dist").iterdir())) == 4
    assert candidates.clean_release_dist(arguments) == 0
    assert list((repository / "dist").iterdir()) == []
    assert candidates.clean_release_dist(arguments) == 0


def test_full_candidate_pins_the_digest_the_s3_configured_agents_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the real ``compute_agent_config_digest``: the digest child sees
    ``GPU_FAULT_DIAGNOSTIC_S3_URI`` exactly as every node's Agent environment
    carries it, so the FULL pin -- and BOOT-023's ``final-noop.json``, copied
    from it -- is the digest those Agents report (live 2026-10-29: the pin
    computed without it never matched and the FULL wave stopped in
    partial-convergence)."""

    installer_environment = {
        key: f"/opt/gpu-fault/{key.lower()}"
        for key in bootstrap_common.NODE_INSTALLER_CONFIG_DIGEST_ENVIRONMENT_KEYS
    }

    def agent_digest(environment: Mapping[str, str]) -> str:
        # Stands in for the Agent's own payload: its profile and destination.
        payload = [
            environment.get("GPU_FAULT_NODE_RUNTIME_PROFILE_VERSION"),
            environment.get(DIAGNOSTIC_S3_URI_ENV),
        ]
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()

    class Runner:
        def run(self, arguments: list[str], **kwargs: Any) -> str:
            if arguments[0] == "bash":
                return json.dumps(installer_environment)
            return agent_digest(kwargs["env"])

    _fixture, arguments, _repository, _base = _write_candidate_configs(
        tmp_path, monkeypatch, diagnostic_s3_uri=DIAGNOSTIC_S3_URI, runner=Runner
    )

    version = "unit-profile-candidate"
    reported = agent_digest(
        {
            "GPU_FAULT_NODE_RUNTIME_PROFILE_VERSION": version,
            DIAGNOSTIC_S3_URI_ENV: DIAGNOSTIC_S3_URI,
        }
    )
    unaware = agent_digest({"GPU_FAULT_NODE_RUNTIME_PROFILE_VERSION": version})
    for name in ("full.json", "final-noop.json"):
        document = json.loads((arguments.out_dir / name).read_text())
        pinned = document["release"]["agent_config_digest"]
        assert pinned == reported != unaware, (name, pinned)


def test_link_conflicts_preserve_real_directories_and_replace_only_links(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    target = repository / "dist" / "candidate"
    target.mkdir(parents=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"release_id":"candidate"}')
    with pytest.raises(SystemExit, match="not a symlink"):
        candidates.link_release_dist(repository, manifest)
    assert target.is_dir(), "a real artifact directory was overwritten"
    target.rmdir()
    other = tmp_path / "other"
    target.symlink_to(other)
    assert candidates.link_release_dist(repository, manifest) == target
    assert target.resolve() == tmp_path / "candidate"


@pytest.mark.parametrize("failure", [False, True])
def test_config_check_classifies_each_named_step_and_reports_partial_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: bool,
) -> None:
    calls = []
    initialized = []

    class Backend:
        def __init__(self, paths: dict[str, Path]) -> None:
            initialized.append(paths)

        def classify(self, scenario: str) -> dict[str, Any]:
            calls.append(scenario)
            if failure and scenario == "executor":
                raise RuntimeError("synthetic classification read failure")
            return {"kind": "NOOP", "changed": []}

    monkeypatch.setitem(
        sys.modules,
        "run_boot020_release_rolling",
        SimpleNamespace(LiveReleaseRollingBackend=Backend),
    )
    monkeypatch.setattr(candidates, "sys", SimpleNamespace(path=list(sys.path)))
    result = candidates.check_configs(
        Namespace(out_dir=tmp_path, gpu_kubeconfig=tmp_path / "unused-kubeconfig")
    )
    assert result == int(failure)
    assert calls == list(candidates.DRIVER_KEYS.values())
    assert initialized == [
        {
            candidates.DRIVER_KEYS[name]: tmp_path / f"{name}.json"
            for name in candidates.CONFIG_NAMES
        }
    ]
    output = capsys.readouterr().out
    assert ("data-plane -> ERROR" in output) is failure
    assert "full -> NOOP" in output
