from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest

from scripts.e2e.regional import run_boot020_release_rolling as rolling
from scripts.e2e.regional.acceptance_runner_common import EvidenceRecorder
from tests.regional._cov95_boot_extra_rolling import RollingEntry
from tests.regional._cov95_boot_extra_safety import (
    boot_extra_isolation as boot_extra_isolation,
)
from tests.regional.test_boot020_runner import FakeReleaseRollingBackend


@pytest.mark.parametrize("resume", [False, True])
def test_rolling_entrypoint_runs_the_real_stage_contract_from_fresh_evidence(
    tmp_path, monkeypatch, capsys, resume
):
    entry = RollingEntry(tmp_path, monkeypatch)
    backend = FakeReleaseRollingBackend()
    constructed = []

    def make_backend(configs):
        constructed.append(configs)
        return backend

    monkeypatch.setattr(rolling, "LiveReleaseRollingBackend", make_backend)
    monkeypatch.setattr(sys, "argv", [*entry.argv, *(["--resume"] if resume else [])])
    assert rolling.main() == 0, (
        "the valid fake release did not finish its stage contract"
    )
    document = json.loads(entry.path.read_text(encoding="utf-8"))
    assert document["status"] == "COMPLETED"
    assert document["verdict"] == "PASS"
    assert document["release_id"] == "restore", (
        "the restored site is the final identity"
    )
    assert document["cluster_ids"] == ["cluster-a"]
    assert document["cluster_id"] == "cluster-a"
    assert set(document["stages"]) >= {f"{stage}_passed" for stage in rolling.STAGES}
    assert constructed == [entry.configs]
    assert len(backend.calls) == 11, "ten chain deploys plus the restore release"
    assert len(entry.calls) == 1
    assert document["inputs"]["config_sha256"] == {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in entry.configs.items()
    }
    assert (
        entry.calls[0][1]["details"]["config_sha256"]
        == (document["inputs"]["config_sha256"])
    )
    assert json.loads(capsys.readouterr().out)["verdict"] == "PASS"
    assert entry.path.stat().st_mode & 0o777 == 0o600
    assert "convergence" not in document, "a fresh --resume invented a prior stage"


@pytest.mark.parametrize(
    "fault", ["resume-stage", "duplicate-config", "missing-config"]
)
def test_rolling_entrypoint_stops_invalid_inputs_before_authorization(
    tmp_path, monkeypatch, fault
):
    entry = RollingEntry(tmp_path, monkeypatch)
    argv = list(entry.argv)
    if fault == "resume-stage":
        argv.extend(["--resume", "--start-stage", "agent"])
        message = "do not also pass"
    elif fault == "duplicate-config":
        argv[argv.index("--agent-config") + 1] = str(entry.configs["executor"])
        message = "five distinct"
    else:
        entry.configs["executor"].unlink()
        message = "does not exist"
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        rolling,
        "LiveReleaseRollingBackend",
        lambda *_args: pytest.fail("invalid inputs constructed a release backend"),
    )
    with pytest.raises(SystemExit, match=message):
        rolling.main()
    assert entry.calls == []
    assert not entry.path.exists(), "input rejection created execution evidence"


@pytest.mark.parametrize("proof", [False, "true", 1, None])
def test_rolling_entrypoint_requires_exact_passed_predecessor_after_authorization(
    tmp_path, monkeypatch, proof
):
    entry = RollingEntry(tmp_path, monkeypatch)
    entry.predecessor["valid"] = proof
    monkeypatch.setattr(sys, "argv", entry.argv)
    monkeypatch.setattr(
        rolling,
        "LiveReleaseRollingBackend",
        lambda *_args: pytest.fail("unpassed predecessor constructed a backend"),
    )
    with pytest.raises(rolling.AcceptanceCheckError, match="predecessor"):
        rolling.main()
    assert len(entry.calls) == 1
    assert not entry.path.exists(), (
        "an unpassed predecessor published execution evidence"
    )


@pytest.mark.parametrize("missing", ["environment", "file"])
def test_gpu_kubeconfig_selection_cannot_fall_back_to_an_unknown_target(
    tmp_path, monkeypatch, missing
):
    monkeypatch.delenv("KUBECONFIG", raising=False)
    path = None if missing == "environment" else tmp_path / "missing-config"
    with pytest.raises(SystemExit, match="requires --gpu-kubeconfig|does not exist"):
        rolling.configure_gpu_kubeconfig(path)
    assert "KUBECONFIG" not in os.environ


def test_gpu_kubeconfig_environment_fallback_is_resolved_without_opening_it(
    tmp_path, monkeypatch
):
    path = tmp_path / "unit-config"
    path.touch()
    monkeypatch.setenv("KUBECONFIG", str(path))
    assert rolling.configure_gpu_kubeconfig(None) == path.resolve()
    assert os.environ["KUBECONFIG"] == str(path.resolve())


def test_unknown_start_stage_invalidates_old_pass_without_running_the_backend(tmp_path):
    recorder = EvidenceRecorder(
        tmp_path / "case.json", case_id=rolling.CASE_ID, inputs={}
    )
    recorder.note("verdict", "PASS")
    recorder.complete()
    backend = FakeReleaseRollingBackend()
    with pytest.raises(ValueError, match="unknown BOOT-020 stage"):
        rolling.run_release_rolling(backend, recorder, start_stage="unknown")
    saved = json.loads(recorder.path.read_text(encoding="utf-8"))
    assert saved["verdict"] == "FAIL"
    assert backend.calls == []
    assert backend.snapshots == []


@pytest.mark.parametrize("clusters", [None, {}, [], {"cluster-a": {}, "cluster-b": {}}])
def test_completed_evidence_needs_actual_final_identity_and_supports_multicluster(
    tmp_path, clusters
):
    recorder = EvidenceRecorder(
        tmp_path / "case.json", case_id=rolling.CASE_ID, inputs={}
    )
    recorder.stage(
        "restore_after",
        lambda: {"release_id": "unit-full", "live": {"clusters": clusters}},
    )
    if isinstance(clusters, dict) and clusters:
        document = rolling.complete_evidence(recorder)
        assert document["cluster_ids"] == ["cluster-a", "cluster-b"]
        assert document["cluster_id"] is None
        assert document["verdict"] == "PASS"
    else:
        with pytest.raises(rolling.AcceptanceCheckError, match="identity is missing"):
            rolling.complete_evidence(recorder)
        saved = json.loads(recorder.path.read_text(encoding="utf-8"))
        assert saved["status"] != "COMPLETED"
        assert saved.get("verdict") != "PASS"


def test_resume_convergence_failure_discards_stale_observations_without_running_next_stage(
    tmp_path,
):
    recorder = EvidenceRecorder(
        tmp_path / "case.json", case_id=rolling.CASE_ID, inputs={}
    )
    recorder.stage("noop_passed", lambda: {"stage": "noop", "passed_at": "unit-before"})
    recorder.stage("control_plane_before", lambda: {"observation": "stale"})
    recorder.stage("resumed_at_control_plane", lambda: {"previous_attempt": True})
    recorder.note("verdict", "PASS")

    class FailingConvergence(FakeReleaseRollingBackend):
        def deploy(self, scenario, **kwargs):
            self.calls.append((scenario, kwargs))
            return {"phase": "failed", "release_id": "unit-unconverged"}

    backend = FailingConvergence()
    with pytest.raises(RuntimeError, match="could not converge.*noop precondition"):
        rolling.resume_release_rolling(backend, recorder)
    deploys = [call for call in backend.calls if call[0] != "resume_rollback"]
    assert len(deploys) == 1, backend.calls
    assert deploys[0][0] == "noop"
    assert backend.snapshots == []
    saved = json.loads(recorder.path.read_text(encoding="utf-8"))
    assert saved["verdict"] == "FAIL"
    assert saved["stages"] == {
        "noop_passed": {"stage": "noop", "passed_at": "unit-before"}
    }
    assert "convergence" not in saved, "failed convergence was recorded as completed"
