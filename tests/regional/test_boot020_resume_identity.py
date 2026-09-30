"""BOOT-020 evidence identity across built and reused candidates (2026-09-30).

The fifth regional run's first execute built the candidates and recorded
``release_candidates = {"action": "built", "work_dir": ...}`` in the evidence
``inputs``; every later attempt found the candidates present, recorded
``{"action": "reused"}`` without the other keys, and ``EvidenceRecorder``
refused the resume with a message that named nothing. The inputs are now the
run's identity -- what the candidates *are*, not what an attempt did for them
-- and a refused resume names the keys, the values and the way to continue.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import boot020_evidence as evidence
from scripts.e2e.regional import boot020_release_prerequisites as prerequisites
from scripts.e2e.regional import run_boot020_release_rolling as boot020
from scripts.e2e.regional.acceptance_runner_common import EvidenceRecorder
from scripts.e2e.regional.regional_commands import RegionalFixtureError

RUNTIME = "1.dkr.ecr.us-west-2.amazonaws.com/gpu-fault/runtime-abc@sha256:" + "0" * 64


def _state_dir(tmp_path: Path) -> Path:
    state = tmp_path / "state"
    snapshot = state / "source-snapshots" / "deadbeef" / "repository-1-live"
    (snapshot / "dist").mkdir(parents=True)
    (snapshot / "dist" / "current-release.json").write_text(
        json.dumps({"release_id": "rel-live"})
    )
    (state / "source-deploy-success.json").write_text(
        json.dumps(
            {
                "live": {"release_id": "rel-live"},
                "prepared_repository_root": str(snapshot),
            }
        )
    )
    (state / "site.yaml").write_text(
        "apiVersion: gpu-fault.aws/v1alpha1\nkind: Site\nspec:\n"
        f"  awsRegion: us-west-2\n  repositoryRoot: {snapshot.parent}\n"
        f"  images:\n    runtime: {RUNTIME}\n"
    )
    return state


def _inputs(state: Path) -> prerequisites.CandidateInputs:
    return prerequisites.CandidateInputs(
        snapshot_repo=prerequisites.live_snapshot_repository(state),
        region="us-west-2",
        runtime_repository=RUNTIME.split("@")[0],
        cache_repository=None,
        runtime_profile="hyperpod-v1",
        site_file=state / "site.yaml",
    )


def _fake_build(namespace: argparse.Namespace) -> dict[str, Any]:
    """Stand in for ``build``: candidate manifests plus the work-dir summary."""

    work = Path(namespace.work_dir)
    summary: dict[str, Any] = {"base_release_id": "rel-live", "candidates": {}}
    for name in ("B", "C", "D"):
        manifest = work / f"wt-{name}" / "dist" / "current-release.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"release_id": f"rel-{name}"}))
        summary["candidates"][name] = {
            "release_id": f"rel-{name}",
            "edits_sha256": name.lower() * 64,
            "manifest": str(manifest),
        }
    (work / "candidates.json").write_text(json.dumps(summary))
    return summary


def _fake_write(namespace: argparse.Namespace) -> None:
    """Stand in for ``configs``: the five chained files naming their manifests."""

    out = Path(namespace.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(namespace.work_dir)
    live = str(Path(namespace.snapshot_repo) / "dist" / "current-release.json")
    manifests = {
        "noop": live,
        "control_plane": live,
        "executor": str(work / "wt-B" / "dist" / "current-release.json"),
        "agent": str(work / "wt-C" / "dist" / "current-release.json"),
        "full": str(work / "wt-D" / "dist" / "current-release.json"),
    }
    for key, name in prerequisites.CONFIG_FILES.items():
        (out / name).write_text(
            json.dumps(
                {
                    "release": {"manifest": manifests[key]},
                    "runtime_profile": {
                        "version": "hyperpod-v1-boot020"
                        if key == "full"
                        else "hyperpod-v1"
                    },
                }
            )
        )


def _arguments(state: Path, **overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "noop_config": None,
        "control_plane_config": None,
        "data_plane_config": None,
        "agent_config": None,
        "full_config": None,
        "admin_state_dir": state,
        "admin_reference": "REF-1",
        "release_candidates_dir": None,
        "replicas_delta": -1,
        "execute": True,
        "gpu_kubeconfig": None,
        "attempt": 1,
        "resume": False,
        "start_stage": "noop",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _first_and_second_attempt(
    tmp_path: Path,
) -> tuple[Path, dict[str, Path], dict[str, Any], dict[str, Any]]:
    """The records of an attempt that built and of one that reused."""

    state = _state_dir(tmp_path)
    arguments = _arguments(state)
    configs, pending = prerequisites.resolve_release_configs(arguments)
    assert pending["action"] == "build-at-execute", pending
    built = prerequisites.ensure_release_candidates(
        state,
        Path(pending["candidates_dir"]),
        gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
        build=_fake_build,
        write=_fake_write,
        check=lambda namespace: 0,
        inputs=_inputs,
    )
    assert built["action"] == "built" and "work_dir" in built, built
    configs_again, reused = prerequisites.resolve_release_configs(arguments)
    assert reused["action"] == "reused", reused
    assert configs_again == configs, (configs_again, configs)
    return state, configs, built, reused


def test_the_first_and_the_resumed_attempt_record_the_same_inputs(tmp_path) -> None:
    state, configs, built, reused = _first_and_second_attempt(tmp_path)
    kubeconfig = tmp_path / "gpu.kubeconfig"
    first = evidence.evidence_inputs(
        _arguments(state), configs=configs, candidates=built, gpu_kubeconfig=kubeconfig
    )
    second = evidence.evidence_inputs(
        _arguments(state, attempt=2, resume=True),
        configs=configs,
        candidates=reused,
        gpu_kubeconfig=kubeconfig,
    )
    assert first == second, (first, second)
    identity = first["release_candidates"]
    for volatile in prerequisites.VOLATILE_RECORD_KEYS:
        assert volatile not in identity, (volatile, identity)
    assert identity["live_release_id"] == "rel-live", identity
    assert identity["snapshot_release_id"] == "rel-live", identity
    assert identity["release_ids"] == {
        "noop": "rel-live",
        "control_plane": "rel-live",
        "executor": "rel-B",
        "agent": "rel-C",
        "full": "rel-D",
    }, identity
    assert identity["candidates"] == {
        name: {"release_id": f"rel-{name}", "edits_sha256": name.lower() * 64}
        for name in ("B", "C", "D")
    }, "the build summary copied into the candidates dir is part of the identity"
    assert identity["runtime_profile"] == "hyperpod-v1", identity
    assert first["acceptance_contract"] == evidence.ACCEPTANCE_CONTRACT
    assert set(first["config_sha256"]) == set(configs), first["config_sha256"]
    # The volatile facts still reach the reader, as provenance.
    assert prerequisites.candidate_provenance(built) == {
        "action": "built",
        "replicas_delta": -1,
        "work_dir": built["work_dir"],
    }
    assert prerequisites.candidate_provenance(reused)["action"] == "reused"


def test_identity_without_the_metadata_copy_is_still_stable(tmp_path) -> None:
    # Candidates built before the metadata copy existed (the live site's
    # 2026-09-30 directory) identify through the manifests alone.
    state, configs, built, reused = _first_and_second_attempt(tmp_path)
    (Path(built["candidates_dir"]) / prerequisites.CANDIDATES_METADATA_NAME).unlink()
    first = prerequisites.candidate_identity(state, configs, built)
    second = prerequisites.candidate_identity(state, configs, reused)
    assert first == second and first["candidates"] == {}, (first, second)
    assert first["release_ids"]["full"] == "rel-D", first


def test_identity_fails_closed_on_a_config_whose_manifest_is_gone(tmp_path) -> None:
    state, configs, built, _reused = _first_and_second_attempt(tmp_path)
    Path(json.loads(configs["full"].read_text())["release"]["manifest"]).unlink()
    with pytest.raises(
        RegionalFixtureError, match="full config points at an unreadable"
    ):
        prerequisites.candidate_identity(state, configs, built)


def test_explicit_configs_identify_by_their_bytes_alone(tmp_path) -> None:
    # Operator-supplied configs are opaque to the runner (the entrypoint tests
    # hand it ``{}`` files and a state dir without a deploy record); their
    # identity is ``config_sha256``, and no state-dir or manifest is read.
    state = tmp_path / "no-such-state"
    configs = {name: tmp_path / f"{name}.json" for name in ("noop", "full")}
    for path in configs.values():
        path.write_text("{}")
    identity = prerequisites.candidate_identity(state, configs, {"action": "explicit"})
    assert identity == {"source": "explicit"}, identity


def test_refused_resume_names_the_differing_inputs_and_the_way_to_continue(
    tmp_path,
) -> None:
    path = tmp_path / "GF-REGIONAL-BOOT-020.json"
    recorded = {
        "acceptance_contract": 4,
        "admin_reference": "REF-1",
        "release_candidates": {"action": "built", "work_dir": "/w", "live": "rel-1"},
    }
    EvidenceRecorder(path, case_id=boot020.CASE_ID, inputs=recorded)
    current = {
        "acceptance_contract": 5,
        "admin_reference": "REF-1",
        "release_candidates": {"action": "reused", "live": "rel-1"},
    }
    drift = evidence.evidence_input_drift(path, current)
    assert drift == {
        "acceptance_contract": {"recorded": 4, "current": 5},
        "release_candidates.action": {"recorded": "built", "current": "reused"},
        "release_candidates.work_dir": {"recorded": "/w", "current": None},
    }, drift
    with pytest.raises(boot020.AcceptanceCheckError) as failure:
        evidence.open_evidence(path, case_id=boot020.CASE_ID, inputs=current)
    message = str(failure.value)
    assert "acceptance_contract, release_candidates.action" in message, message
    assert '"recorded": "built"' in message and '"current": "reused"' in message
    assert "--resume --plan" in message and "--execute" in message, message
    assert json.loads(path.read_text())["inputs"] == recorded, (
        "a refused resume rewrites nothing"
    )
    # Equal inputs open the same document instead of refusing.
    assert evidence.evidence_input_drift(path, recorded) == {}
    assert (
        evidence.open_evidence(path, case_id=boot020.CASE_ID, inputs=recorded).document[
            "inputs"
        ]
        == recorded
    )


@pytest.mark.parametrize(
    "drift, expectation",
    [
        ("live-driver plan drifted at arguments_sha256", "same --attempt, --resume"),
        ("live-driver plan drifted at attempt", "same --attempt, --resume"),
        ("live-driver plan drifted at details", "another candidate state"),
    ],
)
def test_plan_drift_points_at_the_resume_sequence(tmp_path, drift, expectation) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(drift)

    with pytest.raises(RuntimeError) as failure:
        evidence.authorize_release_rolling(
            _arguments(_state_dir(tmp_path)),
            case_id=boot020.CASE_ID,
            confirmation=boot020.CONFIRMATION,
            environment={},
            details={},
            authorize=refuse,
        )
    message = str(failure.value)
    assert message.startswith(drift) and expectation in message, message
    assert "--resume --plan" in message, message


def test_other_authorization_failures_pass_through_unchanged(tmp_path) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("approved maintenance window has ended")

    with pytest.raises(RuntimeError, match="^approved maintenance window has ended$"):
        evidence.authorize_release_rolling(
            _arguments(_state_dir(tmp_path)),
            case_id=boot020.CASE_ID,
            confirmation=boot020.CONFIRMATION,
            environment={},
            details={},
            authorize=refuse,
        )


def test_provenance_is_appended_per_attempt_and_never_an_input(tmp_path) -> None:
    path = tmp_path / "GF-REGIONAL-BOOT-020.json"
    inputs = {"acceptance_contract": evidence.ACCEPTANCE_CONTRACT}
    recorder = EvidenceRecorder(path, case_id=boot020.CASE_ID, inputs=inputs)
    state = _state_dir(tmp_path)
    evidence.record_candidate_provenance(
        recorder, _arguments(state), {"action": "built", "work_dir": "/w"}
    )
    resumed = evidence.open_evidence(path, case_id=boot020.CASE_ID, inputs=inputs)
    history = evidence.record_candidate_provenance(
        resumed, _arguments(state, attempt=2, resume=True), {"action": "reused"}
    )
    assert [(item["attempt"], item["action"]) for item in history] == [
        (1, "built"),
        (2, "reused"),
    ], history
    assert history[0]["work_dir"] == "/w" and history[1]["resume"] is True, history
    document = json.loads(path.read_text())
    assert document["inputs"] == inputs, "provenance must not enter the inputs"
    assert document["release_candidates_provenance"] == history
