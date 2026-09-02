from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

if __package__:
    from scripts.ci_coverage_gate import (
        BUNDLE_NAME as SHARD_BUNDLE_NAME,
        GATE_NAME as SHARD_GATE_NAME,
        SHARDS,
        load_config,
        verify_current_shards,
        verify_shard_gate,
    )
    from scripts.ci_gate_artifacts import (
        GateArtifactError,
        canonical_sha256,
        evidence_entry,
        git,
        sha256,
        verify_evidence,
        write_outputs,
    )
else:
    from ci_coverage_gate import (
        BUNDLE_NAME as SHARD_BUNDLE_NAME,
        GATE_NAME as SHARD_GATE_NAME,
        SHARDS,
        load_config,
        verify_current_shards,
        verify_shard_gate,
    )
    from ci_gate_artifacts import (
        GateArtifactError,
        canonical_sha256,
        evidence_entry,
        git,
        sha256,
        verify_evidence,
        write_outputs,
    )


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 2
GATE_NAME = "unit-gate.json"
BUNDLE_NAME = "unit-gate.bundle.json"
PYTEST_RESULTS_NAME = "pytest-case-results.json"
FAULT_REPORT_NAME = "fault-report.json"
COVERAGE_SUMMARY_NAME = "coverage.json"
DURATIONS_NAME = "test-durations.json"
ARTIFACT_PREFIX = "gpu-fault-unit-gate-"
QUALITY_GATES = {
    "coverage": "PASSED",
    "fault_catalog": "PASSED",
    "postgres_contract": "PASSED",
    "postgres_stress": "PASSED",
}


class UnitGateError(RuntimeError):
    pass


def artifact_name(coverage_sha256: str) -> str:
    if len(coverage_sha256) != 64:
        raise UnitGateError("unit gate coverage identity is invalid")
    return ARTIFACT_PREFIX + coverage_sha256


def unit_identity(
    shard_gates: Mapping[str, Mapping[str, Any]],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    if set(shard_gates) != set(SHARDS):
        raise UnitGateError("unit gate shard set is incomplete")
    config = load_config(root)
    shards = {
        shard: str(shard_gates[shard]["identity"]["sha256"]) for shard in sorted(SHARDS)
    }
    coverage_payload = {
        "branch": bool(config["coverage"]["branch"]),
        "floor": int(config["coverage"]["floor"]),
        "shards": shards,
        "source": str(config["coverage"]["source"]),
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "domain": "unit",
        "coverage_sha256": canonical_sha256(coverage_payload),
        "coverage": coverage_payload,
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def _load_json(path: Path, message: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UnitGateError(message) from exc
    if not isinstance(value, dict):
        raise UnitGateError(message)
    return value


def _verify_coverage_summary(path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    value = _load_json(path, "unit coverage summary is invalid")
    totals = value.get("totals")
    meta = value.get("meta")
    floor = float(load_config(root)["coverage"]["floor"])
    if (
        not isinstance(meta, dict)
        or meta.get("branch_coverage") is not True
        or not isinstance(totals, dict)
        or float(totals.get("percent_covered") or 0.0) < floor
        or not isinstance(value.get("files"), dict)
    ):
        raise UnitGateError("unit coverage floor did not pass")
    return value


def _verify_pytest_results(path: Path) -> dict[str, Any]:
    value = _load_json(path, "unit pytest evidence is invalid")
    if value.get("schema_version") != 1 or not isinstance(value.get("records"), dict):
        raise UnitGateError("unit pytest evidence is invalid")
    return value


def _verify_fault_report(path: Path) -> dict[str, Any]:
    value = _load_json(path, "unit fault report is invalid")
    if value.get("schema_version") != 2 or value.get("verdict") != "PASS":
        raise UnitGateError("unit fault report did not pass")
    return value


def _verify_durations(path: Path) -> dict[str, Any]:
    value = _load_json(path, "unit duration evidence is invalid")
    if (
        value.get("schema_version") != 1
        or set(value.get("shards", {})) != set(SHARDS)
        or not isinstance(value.get("slowest"), list)
    ):
        raise UnitGateError("unit duration evidence is invalid")
    return value


def _copy_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def _copy_shards(
    artifact_root: Path,
    gates: Mapping[str, tuple[Path, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    result = {}
    target_root = artifact_root / "shards"
    shutil.rmtree(target_root, ignore_errors=True)
    for shard, (gate_path, gate) in sorted(gates.items()):
        destination = target_root / shard
        shutil.copytree(gate_path.parent, destination)
        copied_gate = destination / SHARD_GATE_NAME
        copied_bundle = destination / SHARD_BUNDLE_NAME
        if not copied_bundle.is_file():
            raise UnitGateError(f"coverage shard signature bundle is missing: {shard}")
        result[shard] = {
            "bundle_path": copied_bundle.relative_to(artifact_root).as_posix(),
            "bundle_sha256": sha256(copied_bundle),
            "gate_path": copied_gate.relative_to(artifact_root).as_posix(),
            "gate_sha256": sha256(copied_gate),
            "identity_sha256": gate["identity"]["sha256"],
            "producer_git_commit": gate["producer"]["git_commit"],
            "producer_run_id": gate["producer"]["run_id"],
            "reused": gate.get("reused_from") is not None,
        }
    return result


def build_unit_gate(
    root: Path,
    artifact_root: Path,
    *,
    shards_root: Path,
    coverage_summary: Path,
    pytest_results: Path,
    fault_report: Path,
    durations: Path,
    require_run_id: str,
) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved_artifact_root = artifact_root.resolve()
    if resolved_artifact_root == resolved_root or resolved_root.is_relative_to(
        resolved_artifact_root
    ):
        raise UnitGateError("unit artifact root must not contain the repository root")
    resolved_shards_root = shards_root.resolve()
    if (
        resolved_artifact_root == resolved_shards_root
        or resolved_artifact_root.is_relative_to(resolved_shards_root)
        or resolved_shards_root.is_relative_to(resolved_artifact_root)
    ):
        raise UnitGateError("unit artifact root must not overlap coverage shards")
    gates = verify_current_shards(
        root,
        shards_root,
        require_run_id=require_run_id,
        require_trusted=True,
    )
    _verify_coverage_summary(coverage_summary, root=root)
    _verify_pytest_results(pytest_results)
    _verify_fault_report(fault_report)
    _verify_durations(durations)
    resolved_artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_root = resolved_artifact_root
    copied = {
        "coverage_summary": _copy_file(
            coverage_summary,
            artifact_root / COVERAGE_SUMMARY_NAME,
        ),
        "durations": _copy_file(durations, artifact_root / DURATIONS_NAME),
        "fault_report": _copy_file(
            fault_report,
            artifact_root / FAULT_REPORT_NAME,
        ),
        "pytest_results": _copy_file(
            pytest_results,
            artifact_root / PYTEST_RESULTS_NAME,
        ),
    }
    shard_inventory = _copy_shards(artifact_root, gates)
    shard_gates = {shard: gate for shard, (_, gate) in gates.items()}
    identity = unit_identity(shard_gates, root=root)
    repository = os.getenv("GITHUB_REPOSITORY", "")
    workflow_ref = os.getenv("GITHUB_WORKFLOW_REF", "")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    expected_workflow = f"{repository}/.github/workflows/ci.yml@refs/heads/main"
    if (
        not repository
        or workflow_ref != expected_workflow
        or not run_id.isdecimal()
        or run_id != require_run_id
    ):
        raise UnitGateError("unit gate must be built by the current main CI run")
    reused = sorted(
        shard for shard, value in shard_inventory.items() if value["reused"]
    )
    gate = {
        "schema_version": SCHEMA_VERSION,
        "domain": "unit",
        "artifact_name": artifact_name(str(identity["coverage_sha256"])),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "producer": {
            "git_commit": git(root, "rev-parse", "HEAD"),
            "git_tree": git(root, "rev-parse", "HEAD^{tree}"),
            "repository": repository,
            "run_id": run_id,
            "workflow_ref": workflow_ref,
        },
        "quality_gates": dict(QUALITY_GATES),
        "reuse": {
            "fresh_shards": sorted(set(SHARDS) - set(reused)),
            "reused_shards": reused,
        },
        "shards": shard_inventory,
        "evidence": {
            name: evidence_entry(artifact_root, path)
            for name, path in sorted(copied.items())
        },
    }
    target = artifact_root / GATE_NAME
    target.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return gate


def _safe_artifact_path(artifact_root: Path, raw: object, message: str) -> Path:
    path = (artifact_root / str(raw or "")).resolve()
    try:
        path.relative_to(artifact_root.resolve())
    except ValueError as exc:
        raise UnitGateError(message) from exc
    if not path.is_file():
        raise UnitGateError(message)
    return path


def verify_unit_gate(
    gate_path: Path,
    artifact_root: Path,
    *,
    expected_identity: str | None = None,
    expected_coverage_identity: str | None = None,
    source_root: Path = ROOT,
) -> dict[str, Any]:
    gate = _load_json(gate_path, "unit gate is invalid")
    identity = gate.get("identity")
    identity_sha = str((identity or {}).get("sha256") or "")
    coverage_sha = str((identity or {}).get("coverage_sha256") or "")
    if (
        gate.get("schema_version") != SCHEMA_VERSION
        or gate.get("domain") != "unit"
        or gate.get("quality_gates") != QUALITY_GATES
        or not isinstance(identity, dict)
        or canonical_sha256(
            {key: value for key, value in identity.items() if key != "sha256"}
        )
        != identity_sha
        or gate.get("artifact_name") != artifact_name(coverage_sha)
    ):
        raise UnitGateError("unit gate identity or quality gates are invalid")
    if expected_identity is not None and identity_sha != expected_identity:
        raise UnitGateError("unit gate does not match expected content identity")
    if (
        expected_coverage_identity is not None
        and coverage_sha != expected_coverage_identity
    ):
        raise UnitGateError("unit gate does not match expected coverage identity")
    producer = gate.get("producer")
    if not isinstance(producer, dict):
        raise UnitGateError("unit gate producer identity is invalid")
    expected_workflow = (
        f"{producer.get('repository')}/.github/workflows/ci.yml@refs/heads/main"
    )
    if (
        producer.get("workflow_ref") != expected_workflow
        or not str(producer.get("run_id") or "").isdecimal()
        or not str(producer.get("git_commit") or "")
    ):
        raise UnitGateError("unit gate producer identity is invalid")
    verified_evidence = verify_evidence(
        artifact_root,
        gate.get("evidence"),
        expected_names={
            "coverage_summary",
            "durations",
            "fault_report",
            "pytest_results",
        },
    )
    _verify_coverage_summary(
        artifact_root / str(verified_evidence["coverage_summary"]["path"]),
        root=source_root,
    )
    _verify_durations(artifact_root / str(verified_evidence["durations"]["path"]))
    _verify_fault_report(artifact_root / str(verified_evidence["fault_report"]["path"]))
    _verify_pytest_results(
        artifact_root / str(verified_evidence["pytest_results"]["path"])
    )
    shards = gate.get("shards")
    if not isinstance(shards, dict) or set(shards) != set(SHARDS):
        raise UnitGateError("unit gate shard inventory is invalid")
    verified_shards: dict[str, dict[str, Any]] = {}
    for shard, raw in shards.items():
        if not isinstance(raw, dict):
            raise UnitGateError(f"unit gate shard entry is invalid: {shard}")
        shard_gate_path = _safe_artifact_path(
            artifact_root,
            raw.get("gate_path"),
            f"unit gate shard path is invalid: {shard}",
        )
        shard_bundle = _safe_artifact_path(
            artifact_root,
            raw.get("bundle_path"),
            f"unit gate shard bundle path is invalid: {shard}",
        )
        if sha256(shard_gate_path) != raw.get("gate_sha256") or sha256(
            shard_bundle
        ) != raw.get("bundle_sha256"):
            raise UnitGateError(f"unit gate shard artifact does not match: {shard}")
        shard_gate = verify_shard_gate(
            shard_gate_path,
            shard_gate_path.parent,
            expected_identity=str(raw.get("identity_sha256") or ""),
            expected_shard=shard,
            require_trusted=True,
            source_root=source_root,
        )
        if (
            str(shard_gate["producer"]["run_id"]) != str(producer["run_id"])
            or str(raw.get("producer_run_id") or "")
            != str(shard_gate["producer"]["run_id"])
            or raw.get("producer_git_commit") != shard_gate["producer"]["git_commit"]
            or bool(raw.get("reused")) != (shard_gate.get("reused_from") is not None)
        ):
            raise UnitGateError(f"unit gate shard producer does not match: {shard}")
        verified_shards[shard] = shard_gate
    if unit_identity(verified_shards, root=source_root) != identity:
        raise UnitGateError("unit gate aggregate identity does not match shards")
    reuse = gate.get("reuse")
    actual_reused = sorted(
        shard
        for shard, shard_gate in verified_shards.items()
        if shard_gate.get("reused_from") is not None
    )
    if not isinstance(reuse, dict) or reuse != {
        "fresh_shards": sorted(set(SHARDS) - set(actual_reused)),
        "reused_shards": actual_reused,
    }:
        raise UnitGateError("unit gate reuse inventory is invalid")
    return gate


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--artifact-root", type=Path, required=True)
    build.add_argument("--shards-root", type=Path, required=True)
    build.add_argument("--coverage-summary", type=Path, required=True)
    build.add_argument("--pytest-results", type=Path, required=True)
    build.add_argument("--fault-report", type=Path, required=True)
    build.add_argument("--durations", type=Path, required=True)
    build.add_argument("--require-run-id", required=True)
    build.add_argument("--github-output", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--artifact-root", type=Path, required=True)
    verify.add_argument("--gate", type=Path, required=True)
    verify.add_argument("--expected-identity")
    options = parser.parse_args(arguments)
    try:
        if options.command == "build":
            gate = build_unit_gate(
                ROOT,
                options.artifact_root.resolve(),
                shards_root=options.shards_root.resolve(),
                coverage_summary=options.coverage_summary.resolve(),
                pytest_results=options.pytest_results.resolve(),
                fault_report=options.fault_report.resolve(),
                durations=options.durations.resolve(),
                require_run_id=options.require_run_id,
            )
            outputs = {
                "artifact_name": gate["artifact_name"],
                "coverage_identity": gate["identity"]["coverage_sha256"],
                "identity": gate["identity"]["sha256"],
                "reused": str(bool(gate["reuse"]["reused_shards"])).lower(),
            }
            write_outputs(options.github_output, outputs)
            print(json.dumps(outputs, sort_keys=True))
        else:
            verify_unit_gate(
                options.gate.resolve(),
                options.artifact_root.resolve(),
                expected_identity=options.expected_identity,
                source_root=ROOT,
            )
    except (
        GateArtifactError,
        OSError,
        UnitGateError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"ci-unit-gate: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
