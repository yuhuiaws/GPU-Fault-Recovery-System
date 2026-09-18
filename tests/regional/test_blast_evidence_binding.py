from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import blast_acceptance_base as base
from scripts.e2e.regional.blast_acceptance_cases_2 import BlastCasesTwo
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional.test_blast_acceptance_review import (
    SOURCE_DIGEST,
    evidence,
    make_runner,
)


def input_paths(runner: BlastCasesTwo) -> dict[str, Path]:
    return {
        "site": runner.site_path,
        "cpu_kubeconfig": Path(runner.cpu_kubeconfig),
        "gpu_kubeconfig": Path(runner.gpu_kubeconfig),
    }


def change_input(
    runner: BlastCasesTwo, field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if field == "source":
        monkeypatch.setattr(base, "source_digest", lambda: "b" * 64)
    else:
        input_paths(runner)[field].write_text("changed fixture", encoding="ascii")


def write_cache(runner: BlastCasesTwo) -> dict[str, Any]:
    cache = {
        "captured_at": base.utc_now(),
        "site": str(runner.site_path),
        "gpu_clusters": [
            {"cluster_id": target.cluster_id} for target in runner.targets
        ],
        "binding": runner.preflight_binding(),
    }
    base.write_json(runner.root_run_dir / base.PREFLIGHT_CACHE_NAME, cache)
    return cache


def test_unchanged_blast_inputs_record_only_hashes_and_recheck_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[1])
    reads = []
    original = runner.cpu_json

    def cpu_json(*args: str) -> Any:
        reads.append(args)
        return original(*args)

    monkeypatch.setattr(runner, "cpu_json", cpu_json)
    initial = runner.preflight_binding()
    runner.record_case(runner.case_id, "PASS", checks={"permission": True})

    result = evidence(runner)
    expected = {
        f"{name}_sha256": base.sha256_bytes(path.read_bytes())
        for name, path in input_paths(runner).items()
    } | {"source_digest": SOURCE_DIGEST}
    assert result["verdict"] == "PASS"
    assert result["input_binding"] == expected
    assert all(initial[key] == value for key, value in expected.items()), (
        "test_unchanged_blast_inputs_record_only_hashes_and_recheck_release: expected all(initial[key] == value for key, value in expected.items())"
    )
    assert result["release_id"] == initial["identity"]["release_id"] == "release-test"
    assert result["cluster_id"] == initial["identity"]["cluster_id"] == "cluster-0"
    assert len(reads) == 2, "record_case must freshly read the live release identity"
    for path in input_paths(runner).values():
        assert path.read_text() not in json.dumps(result), "input contents leaked"


@pytest.mark.parametrize(
    "field", ["site", "cpu_kubeconfig", "gpu_kubeconfig", "source"]
)
def test_blast_cache_rejects_changed_contents_at_the_same_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[1])
    write_cache(runner)
    assert runner.reusable_preflight() is not None

    change_input(runner, field, monkeypatch)

    assert runner.evidence_identity()["release_id"] == "release-test"
    assert runner.reusable_preflight() is None


@pytest.mark.parametrize(
    "missing",
    ["cpu_kubeconfig_sha256", "gpu_kubeconfig_sha256", "source_digest", "all"],
)
def test_blast_cache_cannot_reuse_path_only_or_partial_input_proofs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing: str
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[1])
    cache = write_cache(runner)
    keys = (
        ("cpu_kubeconfig_sha256", "gpu_kubeconfig_sha256", "source_digest")
        if missing == "all"
        else (missing,)
    )
    for key in keys:
        del cache["binding"][key]
    base.write_json(runner.root_run_dir / base.PREFLIGHT_CACHE_NAME, cache)

    assert runner.reusable_preflight() is None


@pytest.mark.parametrize(
    "field", ["site", "cpu_kubeconfig", "gpu_kubeconfig", "source"]
)
def test_blast_input_drift_during_audit_cannot_earn_pass_with_the_same_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[1])
    initial = runner.input_binding()

    def audit() -> None:
        change_input(runner, field, monkeypatch)
        runner.record_case(runner.case_id, "PASS", checks={"permission": True})

    monkeypatch.setattr(runner, "blast_002", audit)
    assert runner.run() == 1

    result = evidence(runner)
    assert result["verdict"] == "FAIL"
    assert result["release_id"] == "release-test"
    assert result["cluster_id"] == "cluster-0"
    assert result["input_binding"] == initial
    assert "input identity changed" in result["evidence_binding_error"]
    assert "evidence_identity_error" not in result
    summary = json.loads((runner.run_dir / "phase-summary.json").read_text())
    assert summary["status"] == "FAIL"
    assert summary["cases"][0]["verdict"] == "FAIL"


@pytest.mark.parametrize(
    "field", ["site", "cpu_kubeconfig", "gpu_kubeconfig", "source"]
)
def test_blast_input_drift_before_preflight_stops_before_any_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[1])
    write_cache(runner)
    change_input(runner, field, monkeypatch)
    calls = []
    monkeypatch.setattr(runner, "aws", lambda *args: calls.append(args))
    monkeypatch.setattr(runner, "cpu_json", lambda *args: calls.append(args))

    with pytest.raises(base.CheckError, match="input identity changed"):
        base.BlastRunnerBase.preflight(runner)

    assert calls == []
    assert not (runner.run_dir / "execution-scope.json").exists(), (
        'test_blast_input_drift_before_preflight_stops_before_any_probe: expected no (runner.run_dir / "execution-scope.json").exists()'
    )


@pytest.mark.parametrize(
    "field", ["site", "cpu_kubeconfig", "gpu_kubeconfig", "source"]
)
def test_blast_input_read_errors_reject_cache_and_pass_without_exposing_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[1])
    initial = runner.input_binding()
    write_cache(runner)
    private_marker = "private-error-payload-must-not-be-published"
    if field == "source":

        def unavailable_source() -> str:
            raise RuntimeError(private_marker)

        monkeypatch.setattr(base, "source_digest", unavailable_source)
    else:
        unreadable = input_paths(runner)[field]
        original = Path.read_bytes

        def read_bytes(path: Path) -> bytes:
            if path == unreadable:
                raise PermissionError(private_marker)
            return original(path)

        monkeypatch.setattr(Path, "read_bytes", read_bytes)

    with pytest.raises(base.CheckError, match="cannot bind BLAST") as error:
        runner.reusable_preflight()
    assert private_marker not in str(error.value)
    with pytest.raises(base.CheckError, match="lost its input identity"):
        runner.record_case(runner.case_id, "PASS", checks={"permission": True})

    result = evidence(runner)
    assert result["verdict"] == "FAIL"
    assert result["input_binding"] == initial
    assert "cannot bind BLAST" in result["evidence_binding_error"]
    assert private_marker not in json.dumps(result)


@pytest.mark.parametrize("plane", ["cpu", "gpu"])
def test_missing_blast_kubeconfig_cannot_be_bound_before_probes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, plane: str
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[1])
    input_paths(runner)[f"{plane}_kubeconfig"].unlink()

    with pytest.raises(base.CheckError, match=f"{plane}_kubeconfig_sha256"):
        BlastCasesTwo(
            site_path=runner.site_path,
            run_dir=tmp_path / "missing-input",
            case_id=runner.case_id,
            e2e_dir=runner.e2e_dir,
            trusted_cpu_baseline=runner.trusted_cpu_baseline,
            predecessor={"valid": True},
        )

    assert not (tmp_path / "missing-input").exists(), (
        'test_missing_blast_kubeconfig_cannot_be_bound_before_probes: expected no (tmp_path / "missing-input").exists()'
    )


def test_blast_binding_rejects_site_drift_between_parse_and_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[1])
    original = base.load_site

    def load_site(path: Path) -> Any:
        rendered = original(path)
        path.write_text("changed after parsing", encoding="ascii")
        return rendered

    monkeypatch.setattr(base, "load_site", load_site)
    with pytest.raises(base.CheckError, match="site changed while loading"):
        BlastCasesTwo(
            site_path=runner.site_path,
            run_dir=tmp_path / "changed-site",
            case_id=runner.case_id,
            e2e_dir=runner.e2e_dir,
            trusted_cpu_baseline=runner.trusted_cpu_baseline,
            predecessor={"valid": True},
        )

    assert not (tmp_path / "changed-site").exists(), (
        'test_blast_binding_rejects_site_drift_between_parse_and_snapshot: expected no (tmp_path / "changed-site").exists()'
    )


def test_blast_release_drift_still_fails_with_unchanged_input_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[1])
    initial = runner.preflight_binding()
    monkeypatch.setattr(
        runner,
        "cpu_json",
        lambda *args: {"data": {"state.json": '{"release_id":"changed-release"}'}},
    )
    with pytest.raises(base.CheckError, match="lost its deployment identity"):
        runner.record_case(runner.case_id, "PASS", checks={"permission": True})

    result = evidence(runner)
    assert result["verdict"] == "FAIL"
    assert "release identity changed" in result["evidence_identity_error"]
    assert "evidence_binding_error" not in result
    assert all(
        initial[key] == value for key, value in result["input_binding"].items()
    ), (
        'test_blast_release_drift_still_fails_with_unchanged_input_hashes: expected all(initial[key] == value for key, value in result["input_bind...'
    )
