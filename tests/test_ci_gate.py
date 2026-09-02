from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from scripts import ci_gate, ci_unit_gate, resolve_ci_run


def _candidate(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    release_dir = dist / "release-a"
    release_dir.mkdir(parents=True)
    manifest = {"schema_version": 3, "deployable": False, "release_id": "release-a"}
    content = json.dumps(manifest).encode()
    (dist / "current-release.json").write_bytes(content)
    (release_dir / "release.json").write_bytes(content)
    (release_dir / "component.whl").write_bytes(b"wheel")
    return dist


def _unit_gate(dist: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = dist / "ci-domains/unit"
    pytest_results = root / ci_unit_gate.PYTEST_RESULTS_NAME
    fault_report = root / ci_unit_gate.FAULT_REPORT_NAME
    root.mkdir(parents=True)
    pytest_results.write_text(
        json.dumps({"schema_version": 1, "records": {}}), encoding="utf-8"
    )
    fault_report.write_text(
        json.dumps({"schema_version": 2, "verdict": "PASS"}), encoding="utf-8"
    )
    identity = ci_unit_gate.unit_identity(
        Path(__file__).resolve().parents[1], postgres_image="sha256:postgres"
    )
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repository")
    monkeypatch.setenv(
        "GITHUB_WORKFLOW_REF",
        "owner/repository/.github/workflows/ci.yml@refs/heads/main",
    )
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    ci_unit_gate.build_unit_gate(
        Path(__file__).resolve().parents[1],
        root,
        identity=identity,
        pytest_results=pytest_results,
        fault_report=fault_report,
    )
    return root / ci_unit_gate.GATE_NAME


def test_ci_gate_binds_source_and_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dist = _candidate(tmp_path)
    monkeypatch.setattr(
        ci_gate,
        "_git",
        lambda *arguments: {
            ("status", "--porcelain", "--untracked-files=normal"): "",
            ("rev-parse", "HEAD"): "a" * 40,
            ("rev-parse", "HEAD^{tree}"): "b" * 40,
        }[arguments],
    )
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repository")
    monkeypatch.setenv(
        "GITHUB_WORKFLOW_REF",
        "owner/repository/.github/workflows/ci.yml@refs/heads/main",
    )
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    unit_gate = _unit_gate(dist, monkeypatch)

    gate = ci_gate.build_gate(dist, unit_gate)
    path = dist / "ci-gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")

    assert ci_gate.verify_gate(path, dist) == gate
    (dist / "release-a/component.whl").write_bytes(b"tampered")
    with pytest.raises(ci_gate.CiGateError, match="artifacts do not match"):
        ci_gate.verify_gate(path, dist)


def test_ci_gate_rejects_manifest_copy_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dist = _candidate(tmp_path)
    (dist / "release-a/release.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        ci_gate,
        "_git",
        lambda *arguments: {
            ("status", "--porcelain", "--untracked-files=normal"): "",
            ("rev-parse", "HEAD"): "a" * 40,
            ("rev-parse", "HEAD^{tree}"): "b" * 40,
        }[arguments],
    )
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repository")
    monkeypatch.setenv(
        "GITHUB_WORKFLOW_REF",
        "owner/repository/.github/workflows/ci.yml@refs/heads/main",
    )
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    unit_gate = _unit_gate(dist, monkeypatch)

    with pytest.raises(ci_gate.CiGateError, match="content-addressed copy"):
        ci_gate.build_gate(dist, unit_gate)


def test_resolve_ci_run_selects_latest_matching_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "workflow_runs": [
            {"id": 10, "head_sha": "a" * 40, "conclusion": "success", "event": "push"},
            {"id": 12, "head_sha": "a" * 40, "conclusion": "success", "event": "push"},
            {"id": 20, "head_sha": "b" * 40, "conclusion": "success", "event": "push"},
        ]
    }

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        resolve_ci_run,
        "urlopen",
        lambda *_args, **_kwargs: Response(json.dumps(payload).encode()),
    )

    assert (
        resolve_ci_run.resolve_ci_run(
            repository="owner/repository", commit="a" * 40, token="token"
        )
        == 12
    )
