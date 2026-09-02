from __future__ import annotations

import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from scripts import ci_unit_gate

ROOT = Path(__file__).resolve().parents[1]


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)


def _identity_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    files = {
        ".github/workflows/ci.yml": "name: CI\n",
        "CONTRIBUTING.md": "contributing\n",
        "README.md": "readme\n",
        "config/ci-unit-gate.json": (ROOT / "config/ci-unit-gate.json").read_text(
            encoding="utf-8"
        ),
        "deploy/manifest.yaml": "kind: Deployment\n",
        "docs/guide.md": "guide\n",
        "pyproject.toml": "[project]\nname='example'\n",
        "requirements/build.lock": "build==1\n",
        "scripts/release_deploy.py": "VALUE = 'deploy'\n",
        "scripts/ci_gate.py": "VALUE = 'ci'\n",
        "src/gpu_fault/runtime.py": "VALUE = 'runtime'\n",
        "testcases/fault-scenarios.yaml": "schema_version: 1\n",
        "tests/test_runtime.py": "def test_runtime(): pass\n",
        "tests/test_script_assets.py": "def test_ci(): pass\n",
        "tools/run_fault_test_cases.py": "VALUE = 'fault'\n",
        "uv.lock": "version = 1\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    return root


def _identity(root: Path) -> dict:
    return ci_unit_gate.unit_identity(
        root,
        postgres_image="sha256:postgres",
        distributions={"pytest": "9.1.1"},
        environment={"ImageOS": "ubuntu24", "ImageVersion": "20260901.1"},
    )


def test_unit_identity_excludes_docs_and_ci_but_splits_execution_domains(
    tmp_path: Path,
) -> None:
    root = _identity_root(tmp_path)
    before = _identity(root)

    (root / "docs/guide.md").write_text("changed docs\n", encoding="utf-8")
    (root / ".github/workflows/ci.yml").write_text("name: changed\n", encoding="utf-8")
    (root / "scripts/ci_gate.py").write_text("VALUE = 'changed'\n", encoding="utf-8")
    (root / "tests/test_script_assets.py").write_text(
        "def test_changed(): pass\n", encoding="utf-8"
    )
    excluded = _identity(root)

    assert excluded["sha256"] == before["sha256"]
    assert set(before["groups"]) == {
        "dependencies",
        "deployment",
        "fault_runner",
        "runtime",
        "tests",
    }


def test_unit_identity_excludes_exact_documentation_test_set() -> None:
    config = ci_unit_gate.load_config(ROOT)
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    block = makefile.split("DOCUMENTATION_TESTS = \\\n", 1)[1].split(
        "\nCI_TOOLING_TESTS =", 1
    )[0]
    documented = {
        line.strip().rstrip("\\").strip() for line in block.splitlines() if line.strip()
    }
    excluded = set(config["exclude_files"])

    assert documented <= excluded

    tooling_block = makefile.split("CI_TOOLING_TESTS = \\\n", 1)[1].split(
        "\nPOSTGRES_TESTS =", 1
    )[0]
    tooling = {
        line.strip().rstrip("\\").strip()
        for line in tooling_block.splitlines()
        if line.strip()
    }
    assert tooling <= excluded


@pytest.mark.parametrize(
    ("relative", "group", "coverage_changes"),
    [
        ("requirements/build.lock", "dependencies", True),
        ("deploy/manifest.yaml", "deployment", False),
        ("tools/run_fault_test_cases.py", "fault_runner", False),
        ("src/gpu_fault/runtime.py", "runtime", True),
        ("tests/test_runtime.py", "tests", True),
    ],
)
def test_unit_identity_changes_only_the_owned_group(
    tmp_path: Path, relative: str, group: str, coverage_changes: bool
) -> None:
    root = _identity_root(tmp_path)
    before = _identity(root)
    path = root / relative
    path.write_text(path.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")

    after = _identity(root)

    assert after["sha256"] != before["sha256"]
    changed = {
        name
        for name in before["groups"]
        if before["groups"][name]["sha256"] != after["groups"][name]["sha256"]
    }
    assert changed == {group}
    assert (after["coverage_sha256"] != before["coverage_sha256"]) is coverage_changes


def _committed_root(tmp_path: Path) -> Path:
    root = _identity_root(tmp_path)
    _git(root, "config", "user.name", "CI")
    _git(root, "config", "user.email", "ci@example.com")
    _git(root, "commit", "-qm", "test")
    return root


def _evidence(root: Path) -> tuple[Path, Path]:
    pytest_results = root / "pytest-results.json"
    pytest_results.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_identity": "a" * 64,
                "records": {"tests/test_runtime.py::test_runtime": {"status": "PASS"}},
            }
        ),
        encoding="utf-8",
    )
    fault_report = root / "fault-report.json"
    fault_report.write_text(
        json.dumps({"schema_version": 2, "verdict": "PASS", "results": []}),
        encoding="utf-8",
    )
    return pytest_results, fault_report


def test_unit_gate_binds_evidence_and_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _committed_root(tmp_path)
    artifact_root = tmp_path / "artifact"
    pytest_results, fault_report = _evidence(tmp_path)
    identity = _identity(root)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repository")
    monkeypatch.setenv(
        "GITHUB_WORKFLOW_REF",
        "owner/repository/.github/workflows/ci.yml@refs/heads/main",
    )
    monkeypatch.setenv("GITHUB_RUN_ID", "123")

    gate = ci_unit_gate.build_unit_gate(
        root,
        artifact_root,
        identity=identity,
        pytest_results=pytest_results,
        fault_report=fault_report,
    )

    path = artifact_root / ci_unit_gate.GATE_NAME
    assert (
        ci_unit_gate.verify_unit_gate(
            path, artifact_root, expected_identity=identity["sha256"]
        )
        == gate
    )
    (artifact_root / ci_unit_gate.FAULT_REPORT_NAME).write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(ci_unit_gate.UnitGateError, match="evidence does not match"):
        ci_unit_gate.verify_unit_gate(path, artifact_root)


def test_restore_reusable_gate_requires_matching_successful_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _committed_root(tmp_path)
    source = tmp_path / "source"
    pytest_results, fault_report = _evidence(tmp_path)
    identity = _identity(root)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repository")
    monkeypatch.setenv(
        "GITHUB_WORKFLOW_REF",
        "owner/repository/.github/workflows/ci.yml@refs/heads/main",
    )
    monkeypatch.setenv("GITHUB_RUN_ID", "99")
    ci_unit_gate.build_unit_gate(
        root,
        source,
        identity=identity,
        pytest_results=pytest_results,
        fault_report=fault_report,
    )
    (source / ci_unit_gate.BUNDLE_NAME).write_text("{}", encoding="utf-8")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as value:
        for path in sorted(source.iterdir()):
            value.write(path, path.name)
    data = archive.getvalue()
    artifact = {
        "archive_download_url": "https://example.invalid/artifact",
        "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
    }
    run = {"id": 99}
    monkeypatch.setattr(
        ci_unit_gate, "find_reusable_artifact", lambda **_kwargs: (artifact, run)
    )
    monkeypatch.setattr(ci_unit_gate, "_download", lambda *_args: data)
    monkeypatch.setattr(ci_unit_gate, "_is_ancestor", lambda *_args: True)
    monkeypatch.setattr(
        ci_unit_gate,
        "change_impact_plan",
        lambda *_args: {"domains": ["docs-only"], "full": False, "postgres": False},
    )

    destination = tmp_path / "restored"
    result = ci_unit_gate.restore_reusable_gate(
        root=root,
        repository="owner/repository",
        token="token",
        identity=identity,
        destination=destination,
        current_run_id=100,
    )

    assert result["reused"] == "true"
    assert result["delta_required"] == "false"
    assert result["source_run_id"] == 99
    assert (destination / ci_unit_gate.BASE_GATE_NAME).is_file(), (
        "reusable unit gate was not restored"
    )
    monkeypatch.setenv("GITHUB_RUN_ID", "100")
    current = ci_unit_gate.build_unit_gate(
        root,
        destination,
        identity=identity,
        pytest_results=destination / ci_unit_gate.PYTEST_RESULTS_NAME,
        fault_report=destination / ci_unit_gate.FAULT_REPORT_NAME,
    )

    assert current["reused_from"]["producer_run_id"] == "99"
    assert current["delta"]["domains"] == ["docs-only"]


def test_reusable_gate_skips_failed_main_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts = {
        "artifacts": [
            {
                "created_at": "2026-09-02T01:00:00Z",
                "expired": False,
                "id": 1,
                "workflow_run": {"head_branch": "main", "id": 10},
            },
            {
                "created_at": "2026-09-01T01:00:00Z",
                "expired": False,
                "id": 2,
                "workflow_run": {"head_branch": "main", "id": 11},
            },
        ]
    }

    def api(url: str, _token: str):
        if "actions/artifacts" in url:
            return artifacts
        run_id = int(url.rsplit("/", 1)[1])
        return {
            "conclusion": "failure" if run_id == 10 else "success",
            "event": "push",
            "head_branch": "main",
            "id": run_id,
            "path": ".github/workflows/ci.yml",
            "status": "completed",
        }

    monkeypatch.setattr(ci_unit_gate, "_api_json", api)

    found = ci_unit_gate.find_reusable_artifact(
        repository="owner/repository",
        token="token",
        name="gpu-fault-unit-gate-" + "a" * 64,
        current_run_id=None,
    )

    assert found is not None
    assert found[0]["id"] == 2


@pytest.mark.parametrize(
    ("plan", "expected"),
    [
        (
            {"domains": ["docs-only", "ci-tooling"], "full": False, "postgres": False},
            False,
        ),
        ({"domains": ["release-rollout"], "full": False, "postgres": False}, True),
        ({"domains": ["runtime-foundation"], "full": True, "postgres": False}, None),
        ({"domains": ["schema-transaction"], "full": False, "postgres": True}, None),
    ],
)
def test_delta_requirement_fails_closed(plan: dict, expected: bool | None) -> None:
    assert (
        ci_unit_gate.delta_required_for_plan(
            plan, static_only_domains={"ci-tooling", "docs-only"}
        )
        is expected
    )


def test_unit_gate_zip_rejects_path_traversal(tmp_path: Path) -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as value:
        value.writestr("../outside", "unsafe")

    with pytest.raises(ci_unit_gate.UnitGateError, match="unsafe"):
        ci_unit_gate.extract_unit_gate_archive(archive.getvalue(), tmp_path)
