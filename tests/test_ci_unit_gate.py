from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import zipfile
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request

import pytest
from coverage import CoverageData

from scripts import ci_coverage_gate, ci_gate_artifacts, ci_unit_gate
from scripts.component_wheels import APPLICATION_COMPONENT_NAMES, component_modules
from tools.pytest_case_reporter import partition_for_nodeid

ROOT = Path(__file__).resolve().parents[1]


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)


def _identity_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    files = {
        ".github/workflows/ci.yml": "name: CI\n",
        "Makefile": "coverage-shard:\n\t@true\n",
        "config/ci-unit-gate.json": (ROOT / "config/ci-unit-gate.json").read_text(
            encoding="utf-8"
        ),
        "deploy/manifest.yaml": "kind: Deployment\n",
        "docs/guide.md": "guide\n",
        "requirements/build.lock": "build==1\n",
        "src/gpu_fault/admin_config.py": "VALUE = 'shared'\n",
        "src/gpu_fault/admin_only.py": "VALUE = 'deployment'\n",
        "src/gpu_fault/runtime.py": "VALUE = 'runtime'\n",
        "testcases/fault-scenarios.yaml": "schema_version: 1\n",
        "tests/admin/test_admin.py": "def test_admin(): pass\n",
        "tests/conftest.py": "VALUE = 'shared tests'\n",
        "tests/store/test_postgres_store.py": "def test_postgres(): pass\n",
        "tests/test_case_scheduler.py": "def test_scheduler(): pass\n",
        "tests/test_runtime.py": "def test_runtime(): pass\n",
        "tools/case_scheduler.py": "VALUE = 'fault'\n",
        "uv.lock": "version = 1\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    return root


def _committed_root(tmp_path: Path) -> Path:
    root = _identity_root(tmp_path)
    _git(root, "config", "user.name", "CI")
    _git(root, "config", "user.email", "ci@example.com")
    _git(root, "commit", "-qm", "test")
    return root


def _identity(root: Path, shard: str) -> dict:
    return ci_coverage_gate.shard_identity(
        root,
        shard,
        postgres_image="sha256:postgres" if shard == "postgres" else "",
        distributions={"coverage": "7.16.0", "pytest": "9.1.1"},
        environment_identity={
            "machine": "x86_64",
            "postgres_image": ("sha256:postgres" if shard == "postgres" else ""),
            "python_cache_tag": "cpython-312",
            "python_implementation": "CPython",
            "python_version": "3.12.14",
            "runner_environment": "github-hosted",
            "runner_image_os": "ubuntu24",
            "runner_image_version": "20260901.1",
            "runner_label": "ubuntu-latest",
            "sysconfig_platform": "linux-x86_64",
        },
        pytest_workers="4",
    )


def test_coverage_test_partition_is_complete_and_disjoint() -> None:
    counts = ci_coverage_gate.validate_test_partition(ROOT)
    config = ci_coverage_gate.load_config(ROOT)
    assigned = {
        relative
        for shard in ci_coverage_gate.SHARDS
        for relative in ci_coverage_gate.pytest_targets(ROOT, shard)
    }
    collectable = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").rglob("test_*.py")
    }

    assert all(counts[domain] > 0 for domain in ci_coverage_gate.TEST_DOMAINS), (
        "every logical coverage domain must retain at least one test file"
    )
    assert assigned == collectable - set(config["tests"]["coverage_excluded_files"])
    assert sum(counts.values()) == len(assigned)


def test_runtime_nodeid_partitions_are_stable_disjoint_and_complete() -> None:
    nodeids = [f"tests/test_runtime.py::test_case[{index}]" for index in range(1000)]
    count = len(ci_coverage_gate.RUNTIME_SHARDS)
    partitions = [
        {nodeid for nodeid in nodeids if partition_for_nodeid(nodeid, count) == index}
        for index in range(count)
    ]

    assert set.union(*partitions) == set(nodeids)
    assert sum(len(partition) for partition in partitions) == len(nodeids)
    assert max(map(len, partitions)) - min(map(len, partitions)) < 80


def test_coverage_excludes_the_static_documentation_and_ci_tests() -> None:
    config = ci_coverage_gate.load_config(ROOT)
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    docs_block = makefile.split("DOCUMENTATION_TESTS = \\\n", 1)[1].split(
        "\nCI_TOOLING_TESTS =", 1
    )[0]
    docs = {
        line.strip().rstrip("\\").strip()
        for line in docs_block.splitlines()
        if line.strip()
    }
    tooling_block = makefile.split("CI_TOOLING_TESTS = \\\n", 1)[1].split(
        "\nPOSTGRES_TESTS =", 1
    )[0]
    tooling = {
        line.strip().rstrip("\\").strip()
        for line in tooling_block.splitlines()
        if line.strip()
    }

    assert docs | tooling <= set(config["tests"]["coverage_excluded_files"])


def test_deployment_only_coverage_scope_matches_distribution_split() -> None:
    application_modules = {
        module
        for component in APPLICATION_COMPONENT_NAMES
        for module in component_modules(component)
    }
    deploy_host_only = set(component_modules("deploy_host")) - application_modules
    expected = {
        (
            "src/"
            + module.replace(".", "/")
            + (
                "/__init__.py"
                if (ROOT / ("src/" + module.replace(".", "/"))).is_dir()
                else ".py"
            )
        )
        for module in deploy_host_only
    }

    assert set(ci_coverage_gate.deployment_only_source_files(ROOT)) == expected


@pytest.mark.parametrize(
    ("relative", "affected"),
    [
        ("requirements/build.lock", set(ci_coverage_gate.SHARDS)),
        ("Makefile", set(ci_coverage_gate.SHARDS)),
        ("src/gpu_fault/runtime.py", set(ci_coverage_gate.SHARDS)),
        ("src/gpu_fault/admin_only.py", {"deployment"}),
        ("tests/test_runtime.py", set(ci_coverage_gate.RUNTIME_SHARDS)),
        ("tests/admin/test_admin.py", {"deployment"}),
        ("tests/test_case_scheduler.py", {"fault_runner"}),
        ("tests/store/test_postgres_store.py", {"postgres"}),
        ("testcases/fault-scenarios.yaml", {"fault_runner"}),
        ("tests/conftest.py", set(ci_coverage_gate.SHARDS)),
        ("docs/guide.md", set()),
        (".github/workflows/ci.yml", set()),
    ],
)
def test_shard_identity_changes_only_affected_domains(
    tmp_path: Path, relative: str, affected: set[str]
) -> None:
    root = _identity_root(tmp_path)
    before = {
        shard: _identity(root, shard)["sha256"] for shard in ci_coverage_gate.SHARDS
    }
    path = root / relative
    path.write_text(path.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")
    after = {
        shard: _identity(root, shard)["sha256"] for shard in ci_coverage_gate.SHARDS
    }

    assert {
        shard for shard in ci_coverage_gate.SHARDS if before[shard] != after[shard]
    } == affected


def _write_coverage_data(root: Path, path: Path, shard: str) -> None:
    data = CoverageData(basename=str(path))
    measured = [
        root / "src/gpu_fault/runtime.py",
        (
            root / "src/gpu_fault/admin_only.py"
            if shard == "deployment"
            else root / "src/gpu_fault/admin_config.py"
        ),
    ]
    data.add_lines({item.relative_to(root).as_posix(): {1} for item in measured})
    data.write()


def _write_pytest_results(path: Path, nodeid: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_identity": "a" * 64,
                "records": {
                    nodeid: {
                        "duration_seconds": 0.01,
                        "output": "",
                        "phases": {"call": "passed"},
                        "status": "PASS",
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_durations(path: Path, shard: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shard": shard,
                "pytest": {"record_count": 1, "slowest": [], "wall_seconds": 1.0},
                "postgres_stress": (
                    {"record_count": 1, "slowest": [], "wall_seconds": 1.0}
                    if shard == "postgres"
                    else None
                ),
            }
        ),
        encoding="utf-8",
    )


def _set_main_environment(monkeypatch: pytest.MonkeyPatch, *, run_id: str) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repository")
    monkeypatch.setenv(
        "GITHUB_WORKFLOW_REF",
        "owner/repository/.github/workflows/ci.yml@refs/heads/main",
    )
    monkeypatch.setenv("GITHUB_RUN_ID", run_id)


def _build_shard(
    root: Path,
    destination: Path,
    shard: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_id: str = "123",
) -> dict:
    _set_main_environment(monkeypatch, run_id=run_id)
    destination.mkdir(parents=True, exist_ok=True)
    coverage_data = destination / ci_coverage_gate.COVERAGE_DATA_NAME
    pytest_results = destination / ci_coverage_gate.PYTEST_RESULTS_NAME
    durations = destination / ci_coverage_gate.DURATIONS_NAME
    stress = (
        destination / ci_coverage_gate.STRESS_RESULTS_NAME
        if shard == "postgres"
        else None
    )
    _write_coverage_data(root, coverage_data, shard)
    _write_pytest_results(pytest_results, f"tests/{shard}.py::test_pass")
    _write_durations(durations, shard)
    if stress is not None:
        _write_pytest_results(stress, "tests/postgres.py::test_stress")
    gate = ci_coverage_gate.build_shard_gate(
        root,
        destination,
        identity=_identity(root, shard),
        coverage_data=coverage_data,
        pytest_results=pytest_results,
        durations=durations,
        stress_results=stress,
    )
    (destination / ci_coverage_gate.BUNDLE_NAME).write_text("{}", encoding="utf-8")
    return gate


def test_coverage_shard_gate_binds_evidence_and_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _committed_root(tmp_path)
    artifact_root = tmp_path / "artifact"
    gate = _build_shard(root, artifact_root, "runtime_0", monkeypatch)

    assert (
        ci_coverage_gate.verify_shard_gate(
            artifact_root / ci_coverage_gate.GATE_NAME,
            artifact_root,
            expected_identity=gate["identity"]["sha256"],
            expected_shard="runtime_0",
            require_trusted=True,
            source_root=root,
        )
        == gate
    )
    (artifact_root / ci_coverage_gate.DURATIONS_NAME).write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(
        ci_gate_artifacts.GateArtifactError, match="evidence does not match"
    ):
        ci_coverage_gate.verify_shard_gate(
            artifact_root / ci_coverage_gate.GATE_NAME, artifact_root, source_root=root
        )


def test_restore_reusable_shard_resigns_current_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _committed_root(tmp_path)
    source = tmp_path / "source"
    gate = _build_shard(root, source, "runtime_0", monkeypatch, run_id="99")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as value:
        for path in sorted(source.iterdir()):
            value.write(path, path.name)
    data = archive.getvalue()
    artifact = {
        "archive_download_url": "https://example.invalid/artifact",
        "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
    }
    monkeypatch.setattr(
        ci_coverage_gate,
        "find_reusable_artifact",
        lambda **_kwargs: (artifact, {"id": 99}),
    )
    monkeypatch.setattr(ci_coverage_gate, "download", lambda *_args: data)
    monkeypatch.setattr(ci_coverage_gate, "_is_ancestor", lambda *_args: True)

    destination = tmp_path / "restored"
    result = ci_coverage_gate.restore_reusable_shard(
        root=root,
        repository="owner/repository",
        token="token",
        identity=gate["identity"],
        destination=destination,
        current_run_id=100,
    )

    assert result["reused"] == "true"
    assert (destination / ci_coverage_gate.BASE_GATE_NAME).is_file(), (
        "restored evidence must retain the signed base gate"
    )
    _set_main_environment(monkeypatch, run_id="100")
    current = ci_coverage_gate.build_shard_gate(
        root,
        destination,
        identity=gate["identity"],
        coverage_data=destination / ci_coverage_gate.COVERAGE_DATA_NAME,
        pytest_results=destination / ci_coverage_gate.PYTEST_RESULTS_NAME,
        durations=destination / ci_coverage_gate.DURATIONS_NAME,
    )
    assert current["producer"]["run_id"] == "100"
    assert current["reused_from"]["producer_run_id"] == "99"


def test_restore_network_failure_falls_back_to_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _identity_root(tmp_path)
    identity = _identity(root, "runtime_0")
    monkeypatch.setattr(
        ci_coverage_gate,
        "find_reusable_artifact",
        lambda **_kwargs: (_ for _ in ()).throw(
            HTTPError("https://api.github.com/example", 503, "unavailable", {}, None)
        ),
    )

    result = ci_coverage_gate.restore_reusable_shard(
        root=root,
        repository="owner/repository",
        token="token",
        identity=identity,
        destination=tmp_path / "restored",
        current_run_id=100,
    )

    assert result["reused"] == "false"


def _unit_evidence(root: Path) -> tuple[Path, Path, Path, Path]:
    coverage = root / "coverage.json"
    coverage.write_text(
        json.dumps(
            {
                "meta": {"branch_coverage": True},
                "files": {"src/gpu_fault/runtime.py": {}},
                "totals": {"percent_covered": 80.0},
            }
        ),
        encoding="utf-8",
    )
    pytest_results = root / "pytest-case-results.json"
    _write_pytest_results(pytest_results, "tests/runtime.py::test_pass")
    fault_report = root / "fault-report.json"
    fault_report.write_text(
        json.dumps({"schema_version": 2, "verdict": "PASS", "results": []}),
        encoding="utf-8",
    )
    durations = root / "test-durations.json"
    durations.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shards": {
                    shard: {
                        "pytest": {"wall_seconds": 1.0},
                        "postgres_stress": None,
                        "producer_run_id": "123",
                        "reused": False,
                    }
                    for shard in ci_coverage_gate.SHARDS
                },
                "slowest": [],
            }
        ),
        encoding="utf-8",
    )
    return coverage, pytest_results, fault_report, durations


def test_unit_gate_aggregates_all_signed_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _committed_root(tmp_path)
    shards_root = tmp_path / "shards"
    for shard in ci_coverage_gate.SHARDS:
        _build_shard(root, shards_root / shard, shard, monkeypatch)
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    coverage, pytest_results, fault_report, durations = _unit_evidence(evidence_root)
    artifact_root = tmp_path / "unit"
    _set_main_environment(monkeypatch, run_id="123")

    gate = ci_unit_gate.build_unit_gate(
        root,
        artifact_root,
        shards_root=shards_root,
        coverage_summary=coverage,
        pytest_results=pytest_results,
        fault_report=fault_report,
        durations=durations,
        require_run_id="123",
    )

    assert set(gate["shards"]) == set(ci_coverage_gate.SHARDS)
    assert (
        ci_unit_gate.verify_unit_gate(
            artifact_root / ci_unit_gate.GATE_NAME,
            artifact_root,
            expected_identity=gate["identity"]["sha256"],
            source_root=root,
        )
        == gate
    )
    bundle = artifact_root / "shards" / "runtime_0" / ci_coverage_gate.BUNDLE_NAME
    bundle.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ci_unit_gate.UnitGateError, match="artifact does not match"):
        ci_unit_gate.verify_unit_gate(
            artifact_root / ci_unit_gate.GATE_NAME, artifact_root, source_root=root
        )


def test_coverage_combine_uses_the_configured_data_file_basename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _committed_root(tmp_path)
    shards_root = tmp_path / "shards"
    for shard in ci_coverage_gate.SHARDS:
        _build_shard(root, shards_root / shard, shard, monkeypatch)
    output_root = tmp_path / "combined"

    ci_coverage_gate.combine_shards(
        root=root,
        shards_root=shards_root,
        output_root=output_root,
        python=sys.executable,
        require_run_id="123",
    )

    summary = json.loads((output_root / "coverage.json").read_text(encoding="utf-8"))
    merged = json.loads(
        (output_root / "pytest-case-results.json").read_text(encoding="utf-8")
    )
    assert summary["totals"]["percent_covered"] >= 78
    assert len(merged["records"]) == len(ci_coverage_gate.SHARDS)


def test_reusable_artifact_requires_a_successful_producer_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        if url.endswith("/jobs?per_page=100"):
            run_id = int(url.rsplit("/", 2)[1])
            return {
                "jobs": [
                    {
                        "conclusion": "failure" if run_id == 10 else "success",
                        "name": "coverage-runtime_0",
                        "status": "completed",
                    }
                ]
            }
        run_id = int(url.rsplit("/", 1)[1])
        return {
            "conclusion": "failure",
            "event": "push",
            "head_branch": "main",
            "id": run_id,
            "path": ".github/workflows/ci.yml",
            "status": "completed",
        }

    monkeypatch.setattr(ci_gate_artifacts, "api_json", api)

    found = ci_gate_artifacts.find_reusable_artifact(
        repository="owner/repository",
        token="token",
        name="gpu-fault-coverage-runtime-" + "a" * 64,
        current_run_id=None,
        required_job_name="coverage-runtime_0",
    )

    assert found is not None
    assert found[0]["id"] == 2


def test_artifact_redirect_drops_authentication_on_cross_origin() -> None:
    request = Request(
        "https://api.github.com/repos/owner/repository/actions/artifacts/1/zip",
        headers={"Authorization": "Bearer example"},
    )
    redirected = ci_gate_artifacts.ArtifactRedirectHandler().redirect_request(
        request,
        None,
        302,
        "Found",
        Message(),
        "https://objects.example.invalid/artifact.zip?signature=example",
    )

    assert redirected is not None
    assert redirected.get_header("Authorization") is None


def test_gate_zip_rejects_path_traversal(tmp_path: Path) -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as value:
        value.writestr("../outside", "unsafe")

    with pytest.raises(ci_gate_artifacts.GateArtifactError, match="unsafe"):
        ci_gate_artifacts.extract_archive(archive.getvalue(), tmp_path)
