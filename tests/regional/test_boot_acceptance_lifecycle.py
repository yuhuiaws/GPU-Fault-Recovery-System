"""BOOT-016/017/018 lifecycle runner contracts from the 2026-09-07 review.

The isolated bootstrap site (Aurora, NLB, AMP, ECR, IAM) used to survive every
failure except a non-zero first deploy: a deploy timeout, a kubectl error or a
FAIL verdict left it behind, and BOOT-018 -- the only other uninstall -- was
then refused by the predecessor gate. These tests pin the cleanup, the BOOT-017
reuse gate and the BOOT-018 ``status --full`` comparison.
"""

from __future__ import annotations

import json
import subprocess
import threading
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import boot_acceptance_lifecycle as lifecycle
from scripts.e2e.regional.boot_acceptance_common import BootAcceptanceError


def _completed(returncode: int, stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["x"], returncode, stdout=stdout, stderr="")


def _arguments(tmp_path: Path, **overrides: Any) -> Namespace:
    values: dict[str, Any] = {
        "bootstrap_state_dir": tmp_path / "state",
        "cpu_cluster_arn": "arn:aws:eks:us-west-2:000000000000:cluster/cpu",
        "gpu_cluster_arn": ["arn:aws:eks:us-west-2:000000000000:cluster/gpu"],
        "admin_email": "ops@example.invalid",
        "retain_bootstrap_site": False,
    }
    values.update(overrides)
    return Namespace(**values)


class AdminRecorder:
    """Stands in for ``admin_command``: deploy commits a site and then fails."""

    def __init__(self, state_dir: Path, *, deploy_error: BaseException) -> None:
        self.state_dir = state_dir
        self.deploy_error = deploy_error
        self.calls: list[tuple[str, ...]] = []

    def __call__(
        self, *arguments: str, **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(arguments)
        if arguments[0] == "deploy":
            (self.state_dir / "site.yaml").write_text("site: true\n", encoding="utf-8")
            raise self.deploy_error
        return _completed(0)


# -- item 1: BOOT-016 cleans up after every failure -----------------------------


def test_boot016_uninstalls_the_site_after_a_deploy_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(tmp_path)
    recorder = AdminRecorder(
        arguments.bootstrap_state_dir,
        deploy_error=subprocess.TimeoutExpired(cmd="deploy", timeout=21600),
    )
    monkeypatch.setattr(lifecycle, "admin_command", recorder)
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    outcome = lifecycle.run_boot016(arguments, case_dir)

    assert outcome["verdict"] == "FAIL"
    assert "TimeoutExpired" in outcome["error"]
    assert outcome["checks"]["failure_cleanup"] is True
    assert outcome["cleanup"]["uninstall_ran"] is True
    uninstalls = [call for call in recorder.calls if call[0] == "uninstall"]
    assert len(uninstalls) == 1
    assert "--confirm" in uninstalls[0] and "UNINSTALL_GPU_FAULT" in uninstalls[0]
    assert (case_dir / "failed-deploy-cleanup.log").is_file(), (
        "the failed deploy's cleanup log is kept for the operator"
    )


def test_boot016_uninstalls_after_a_fixture_error_and_keeps_the_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(tmp_path)
    recorder = AdminRecorder(
        arguments.bootstrap_state_dir,
        deploy_error=BootAcceptanceError("Pod probe did not return a JSON object"),
    )
    monkeypatch.setattr(lifecycle, "admin_command", recorder)

    outcome = lifecycle.run_boot016(arguments, tmp_path / "case")

    assert outcome["verdict"] == "FAIL"
    assert outcome["checks"]["failure_cleanup"] is True
    assert any("BootAcceptanceError" in line for line in outcome["traceback"]), (
        "the traceback names the BootAcceptanceError that failed the case"
    )


def test_boot016_honours_retain_and_never_uninstalls_a_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(tmp_path, retain_bootstrap_site=True)
    recorder = AdminRecorder(
        arguments.bootstrap_state_dir, deploy_error=BootAcceptanceError("boom")
    )
    monkeypatch.setattr(lifecycle, "admin_command", recorder)

    retained = lifecycle.run_boot016(arguments, tmp_path / "case-a")

    assert retained["cleanup"] == {
        "site_present": True,
        "retained": True,
        "uninstall_ran": False,
        "uninstall_returncode": None,
    }
    assert "failure_cleanup" not in retained["checks"]

    passing = _arguments(tmp_path, bootstrap_state_dir=tmp_path / "state-b")
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        lifecycle, "admin_command", lambda *a, **k: calls.append(a) or _completed(0)
    )
    monkeypatch.setattr(
        lifecycle,
        "_boot016_verified_site",
        lambda command, state_dir, case_dir: (
            (state_dir / "site.yaml").write_text("x", encoding="utf-8"),
            {"verdict": "PASS", "checks": {"status_passed": True}},
        )[1],
    )

    passed = lifecycle.run_boot016(passing, tmp_path / "case-b")

    assert passed["verdict"] == "PASS"
    assert passed["cleanup"]["uninstall_ran"] is False
    assert not calls, "a PASS must leave the site for BOOT-017/018"


# -- item 1: BOOT-018 cleanup lives in finally ---------------------------------


def test_boot018_uninstalls_when_the_build_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(tmp_path)
    arguments.bootstrap_state_dir.mkdir()
    (arguments.bootstrap_state_dir / "site.yaml").write_text("x", encoding="utf-8")
    uninstalls: list[Path] = []
    monkeypatch.setattr(
        lifecycle,
        "uninstall_site",
        lambda state_dir: uninstalls.append(state_dir) or _completed(0),
    )

    def failing_body(state_dir: Path, case_dir: Path) -> dict[str, Any]:
        raise BootAcceptanceError("BOOT-018 build failed under umask 077")

    monkeypatch.setattr(lifecycle, "boot018_body", failing_body)

    outcome = lifecycle.run_boot018(arguments, tmp_path / "case")

    assert outcome["verdict"] == "FAIL"
    assert "build failed" in outcome["error"]
    assert uninstalls == [arguments.bootstrap_state_dir.resolve()]
    assert outcome["checks"]["bootstrap_site_cleanup"] is True
    assert outcome["cleanup"]["uninstall_ran"] is True


def test_boot018_records_cleanup_for_every_verdict_and_fails_a_bad_uninstall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(tmp_path)
    arguments.bootstrap_state_dir.mkdir()
    (arguments.bootstrap_state_dir / "site.yaml").write_text("x", encoding="utf-8")
    monkeypatch.setattr(lifecycle, "uninstall_site", lambda state_dir: _completed(1))
    monkeypatch.setattr(
        lifecycle,
        "boot018_body",
        lambda state_dir, case_dir: {"verdict": "PASS", "checks": {"x": True}},
    )

    outcome = lifecycle.run_boot018(arguments, tmp_path / "case")

    assert outcome["checks"]["bootstrap_site_cleanup"] is False
    assert outcome["verdict"] == "FAIL", "a leaked site is not a PASS"

    retained = _arguments(tmp_path, retain_bootstrap_site=True)
    outcome = lifecycle.run_boot018(retained, tmp_path / "case-r")
    assert outcome["cleanup"]["retained"] is True
    assert outcome["checks"]["bootstrap_site_cleanup"] is True
    assert outcome["verdict"] == "PASS"


# -- item 5: BOOT-017 reuse gate ---------------------------------------------


def _predecessor(verdict: str, path: Path | None) -> dict[str, Any]:
    return {"valid": True, "verdict": verdict, "path": str(path) if path else None}


def test_boot016_reuse_is_not_evaluated_under_selective_scope(tmp_path: Path) -> None:
    result = lifecycle.boot016_reuse(
        _predecessor("SKIPPED_BY_OPERATOR", None),
        expected_state_dir=tmp_path / "state",
        read_document=lambda path: {},
    )

    assert result["reused"] == "NOT_EVALUATED"
    assert result["status_passed"] == "NOT_EVALUATED"
    assert "not PASS" in result["reason"]


def test_boot016_reuse_requires_the_same_state_dir(tmp_path: Path) -> None:
    evidence = tmp_path / "GF-REGIONAL-BOOT-016.json"
    document = {
        "case_id": "GF-REGIONAL-BOOT-016",
        "verdict": "PASS",
        "state_dir": str(tmp_path / "other-state"),
        "checks": {"status_passed": True},
    }

    result = lifecycle.boot016_reuse(
        _predecessor("PASS", evidence),
        expected_state_dir=tmp_path / "state",
        read_document=lambda path: document,
    )
    assert result["reused"] == "NOT_EVALUATED"
    assert "state_dir differs" in result["reason"]

    document["state_dir"] = str(tmp_path / "state")
    document["checks"] = {"status_passed": False}
    result = lifecycle.boot016_reuse(
        _predecessor("PASS", evidence),
        expected_state_dir=tmp_path / "state",
        read_document=lambda path: document,
    )
    assert result["reused"] is True
    assert result["status_passed"] is False, "BOOT-016's own status check is consumed"


def test_boot017_is_partial_when_the_greenfield_run_cannot_be_evaluated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(tmp_path)
    admin_calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(lifecycle, "run", lambda *a, **k: _completed(0))
    monkeypatch.setattr(
        lifecycle,
        "admin_command",
        lambda *a, **k: admin_calls.append(a) or _completed(0),
    )
    monkeypatch.setattr(lifecycle, "site_identity", lambda path: None)

    outcome = lifecycle.run_boot017(
        arguments, _predecessor("SKIPPED_BY_OPERATOR", None), tmp_path / "case"
    )

    assert outcome["verdict"] == "PARTIAL"
    assert outcome["checks"]["greenfield_sequence_reused_boot016"] == "NOT_EVALUATED"
    assert not admin_calls, "BOOT-017 no longer reruns admin status on the same site"


# -- item 6: BOOT-018 compares status --full against the release itself --------


def _report(cpu_digest: str, gpu_digest: str) -> dict[str, Any]:
    return {
        "healthy": True,
        "checks": [
            {"name": "cpu_secrets", "status": "PASS", "details": None},
            {
                "name": "runtime_component_identity",
                "status": "PASS",
                "details": {
                    "control_plane": {
                        "expected": cpu_digest,
                        "deployments": {
                            "gpu-fault-api-ha": {
                                "api-0": cpu_digest,
                                "api-1": cpu_digest,
                            }
                        },
                    },
                    "executor": {
                        "expected": gpu_digest,
                        "clusters": {
                            "cluster-a": {
                                "gpu-fault-cluster-executor": {"ex-0": gpu_digest}
                            }
                        },
                    },
                },
            },
        ],
    }


CPU = "c" * 64
EXEC = "e" * 64
EXEC_WHEEL = "1" * 64
NODE_WHEEL = "2" * 64
NODE_DIGEST = "3" * 64
MANIFEST = {
    "components": {
        "control_plane": {"module_digest": CPU, "wheel_sha256": "0" * 64},
        "executor": {"module_digest": EXEC, "wheel_sha256": EXEC_WHEEL},
        "node_runtime": {"module_digest": NODE_DIGEST, "wheel_sha256": NODE_WHEEL},
    }
}
METADATA = {
    "required-regional-executor-artifact-sha256": EXEC_WHEEL,
    "required-regional-executor-compatibility-digest": EXEC,
    "required-agent-artifact-sha256": NODE_WHEEL,
    "required-agent-compatibility-digest": NODE_DIGEST,
}
AGENTS = [
    {
        "cluster_id": "cluster-a",
        "node_id": "node-1",
        "artifact_sha256": NODE_WHEEL,
        "compatibility_digest": NODE_DIGEST,
    }
]


def test_parse_status_report_skips_wrapper_lines() -> None:
    stdout = "using site /tmp/x\n" + json.dumps({"healthy": True}, indent=2) + "\n"

    assert lifecycle.parse_status_report(stdout) == {"healthy": True}
    with pytest.raises(BootAcceptanceError, match="no JSON report"):
        lifecycle.parse_status_report("nothing here\n")


def test_runtime_identity_passes_only_when_every_replica_matches_the_release() -> None:
    result = lifecycle.runtime_identity_matches_release(
        _report(CPU, EXEC), manifest=MANIFEST, metadata=METADATA, agents=AGENTS
    )
    assert result["passed"] is True, result["reasons"]
    assert result["replica_count"] == 3

    drifted = _report(CPU, EXEC)
    drifted["checks"][1]["details"]["executor"]["clusters"]["cluster-a"][
        "gpu-fault-cluster-executor"
    ]["ex-0"] = "f" * 64
    result = lifecycle.runtime_identity_matches_release(
        drifted, manifest=MANIFEST, metadata=METADATA, agents=AGENTS
    )
    assert result["passed"] is False
    assert any("ex-0 digest differs" in reason for reason in result["reasons"]), (
        "a replica whose digest differs from the release is a reason"
    )


def test_runtime_identity_fails_on_missing_fields_instead_of_skipping() -> None:
    reused = {
        "checks": [
            {
                "name": "runtime_component_identity",
                "status": "PASS",
                "details": {"evidence": "/quick", "verified_at_epoch": 1},
            }
        ]
    }
    result = lifecycle.runtime_identity_matches_release(
        reused, manifest=MANIFEST, metadata=METADATA, agents=AGENTS
    )
    assert result["passed"] is False
    assert any("no replica digests" in reason for reason in result["reasons"]), (
        "an executor with no replica digests is a reason"
    )

    result = lifecycle.runtime_identity_matches_release(
        _report(CPU, EXEC), manifest=MANIFEST, metadata={}, agents=[]
    )
    assert any("release metadata lacks" in reason for reason in result["reasons"]), (
        "release metadata without the pin is a reason"
    )
    assert any("no ACTIVE Agent" in reason for reason in result["reasons"]), (
        "a cluster without an ACTIVE Agent is a reason"
    )

    stale_agent = [{**AGENTS[0], "artifact_sha256": "9" * 64}]
    result = lifecycle.runtime_identity_matches_release(
        _report(CPU, EXEC), manifest=MANIFEST, metadata=METADATA, agents=stale_agent
    )
    assert any("artifact pin differs" in reason for reason in result["reasons"]), (
        "an artifact pin that differs from the release is a reason"
    )


def test_boot018_second_build_runs_only_the_wheel_closure_test(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return _completed(0)

    def copy(destination: Path) -> None:
        (destination / "dist").mkdir(parents=True)
        (destination / "dist/current-release.json").write_text(
            json.dumps({"release_id": "r1"}), encoding="utf-8"
        )

    result = lifecycle.build_release_under_umask(
        "002", base=tmp_path, case_dir=tmp_path / "case", runner=runner, copy=copy
    )

    assert result["manifest"] == {"release_id": "r1"}
    pytest_command = commands[1]
    assert pytest_command[-1].endswith(
        "::test_built_component_wheels_match_their_source_closures"
    ), "the umask build runs the wheel-closure artifact test"
    assert lifecycle.ARTIFACT_TESTS["077"] == "tests/test_artifact_consistency.py"


def test_boot018_builds_the_two_umask_checkouts_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    barrier = threading.Barrier(2, timeout=5)
    started: list[str] = []

    def build(mask: str, *, base: Path, case_dir: Path) -> dict[str, Any]:
        started.append(mask)
        # Both builds must be inside this function at once; a sequential
        # loop never releases the barrier and times out.
        barrier.wait()
        manifest = {
            "release_id": "r1",
            "bundle_sha256": "b" * 64,
            "components": {
                name: {"wheel_sha256": "w" * 64, "module_digest": "m" * 64}
                for name in ("control_plane", "executor", "node_runtime")
            },
        }
        return {"checkout": base / f"repo-{mask}", "manifest": manifest}

    monkeypatch.setattr(lifecycle, "build_release_under_umask", build)

    # After the builds the body tampers with a file in repo-077, which the fake
    # build never created; that FileNotFoundError ends the test after the part
    # under test has run.
    with pytest.raises(FileNotFoundError):
        lifecycle.boot018_body(tmp_path / "state", tmp_path / "case")

    assert sorted(started) == ["002", "077"]
    assert not barrier.broken, "both builds reached the barrier before it timed out"


def test_runtime_identity_reads_the_checks_status_full_nests_under_health() -> None:
    """`status --full` wraps the health report; its checks sit under health.checks.

    Live 2026-09-13: BOOT-018 read the wrapped report at the top level, found no
    runtime_component_identity check and failed a site whose report carried it
    as PASS.
    """

    wrapped = {"live_release": {"release_id": "x"}, "health": _report(CPU, EXEC)}

    result = lifecycle.runtime_identity_matches_release(
        wrapped, manifest=MANIFEST, metadata=METADATA, agents=AGENTS
    )

    assert result["passed"] is True, result["reasons"]
