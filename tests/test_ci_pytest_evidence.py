from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from coverage import CoverageData

from scripts import ci_coverage_gate as gate
from scripts import ci_gate_artifacts, ci_unit_gate
from scripts.ci_pytest_evidence import (
    aggregate_pytest_results,
    ci_context,
    validate_shard_receipt,
)
from tests.test_ci_unit_gate import (
    _build_shard,
    _committed_root,
    _identity,
    _set_main_environment,
    _unit_evidence,
)
from tools.pytest_result_identity import (
    CI_CONTEXT_ENV,
    parse_pytest_receipt,
    source_identity,
)


def rebuild(root: Path, artifact: Path, identity: dict) -> dict:
    return gate.build_shard_gate(
        root,
        artifact,
        identity=identity,
        coverage_data=artifact / gate.COVERAGE_DATA_NAME,
        pytest_results=artifact / gate.PYTEST_RESULTS_NAME,
        durations=artifact / gate.DURATIONS_NAME,
        stress_results=(
            artifact / gate.STRESS_RESULTS_NAME
            if identity["shard"] == "postgres"
            else None
        ),
    )


@pytest.fixture
def runtime_shard(tmp_path, monkeypatch):
    root = _committed_root(tmp_path)
    artifact = tmp_path / "runtime"
    current = _build_shard(root, artifact, "runtime_0", monkeypatch)
    report = artifact / gate.PYTEST_RESULTS_NAME
    return root, artifact, current, json.loads(report.read_text(encoding="utf-8"))


@pytest.fixture(params=[shard for shard in gate.SHARDS if shard != "postgres"])
def worker_shard(request, tmp_path, monkeypatch):
    root = _committed_root(tmp_path)
    artifact = tmp_path / request.param
    current = _build_shard(root, artifact, request.param, monkeypatch)
    report = artifact / gate.PYTEST_RESULTS_NAME
    return root, artifact, current, json.loads(report.read_text(encoding="utf-8"))


def identity_with_workers(root: Path, previous: dict, workers: str) -> dict:
    return gate.shard_identity(
        root,
        previous["shard"],
        distributions=previous["installed_distributions"],
        environment_identity=previous["environment"],
        pytest_workers=workers,
    )


@pytest.mark.parametrize("field", ["numprocesses", "requested_numprocesses"])
@pytest.mark.parametrize(
    "reported", [0, 1, 8, -1, True, False, "4", 4.0, None, [], {}, "auto", "logical"]
)
def test_ci_worker_binding_rejects_mismatches_at_build_and_verification(
    worker_shard, field, reported
):
    root, artifact, current, value = worker_shard
    value["session"]["selection"][field] = reported
    with pytest.raises(ValueError, match="workers"):
        validate_shard_receipt(value, root=root, identity=current["identity"])
    report = artifact / gate.PYTEST_RESULTS_NAME
    report.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(gate.CoverageGateError, match="workers"):
        rebuild(root, artifact, current["identity"])

    current["evidence"]["pytest_results"] = ci_gate_artifacts.evidence_entry(
        artifact, report
    )
    path = artifact / gate.GATE_NAME
    path.write_text(json.dumps(current), encoding="utf-8")
    with pytest.raises(gate.CoverageGateError, match="workers"):
        gate.verify_shard_gate(path, artifact, source_root=root)


def test_ci_worker_binding_requires_an_actual_count(worker_shard):
    root, _artifact, current, value = worker_shard
    value["session"]["selection"].pop("numprocesses")
    with pytest.raises(ValueError, match="workers"):
        validate_shard_receipt(value, root=root, identity=current["identity"])


@pytest.mark.parametrize(
    ("configured", "actual"),
    [
        ("0", 0),
        ("1", 1),
        ("4", 4),
        ("8", 8),
        ("04", 4),
        ("+4", 4),
        (" 4 ", 4),
        ("1_0", 10),
        ("auto", 1),
        ("auto", 8),
        ("logical", 2),
    ],
)
def test_ci_worker_binding_accepts_supported_matching_settings(
    worker_shard, configured, actual
):
    root, artifact, current, value = worker_shard
    identity = identity_with_workers(root, current["identity"], configured)
    value["session"]["ci_context"] = ci_context(identity, "pytest")
    selection = value["session"]["selection"]
    selection.update(
        numprocesses=actual,
        requested_numprocesses=(
            configured if configured in ("auto", "logical") else actual
        ),
    )
    report = artifact / gate.PYTEST_RESULTS_NAME
    report.write_text(json.dumps(value), encoding="utf-8")
    rebuilt = rebuild(root, artifact, identity)
    verified = gate.verify_shard_gate(
        artifact / gate.GATE_NAME,
        artifact,
        source_root=root,
        expected_identity=identity["sha256"],
    )
    assert verified["identity"] == rebuilt["identity"] == identity, (
        "worker validation must preserve the declared content identity"
    )
    if configured not in ("auto", "logical"):
        selection.pop("requested_numprocesses")
        validate_shard_receipt(value, root=root, identity=identity)


@pytest.mark.parametrize("field", ["numprocesses", "requested_numprocesses"])
@pytest.mark.parametrize(("configured", "reported"), [("0", False), ("1", True)])
def test_worker_booleans_cannot_alias_zero_or_one(
    runtime_shard, configured, field, reported
):
    root, _artifact, current, value = runtime_shard
    identity = identity_with_workers(root, current["identity"], configured)
    value["session"]["ci_context"] = ci_context(identity, "pytest")
    value["session"]["selection"].update(
        numprocesses=int(configured), requested_numprocesses=int(configured)
    )
    validate_shard_receipt(value, root=root, identity=identity)
    value["session"]["selection"][field] = reported
    with pytest.raises(ValueError, match="workers"):
        validate_shard_receipt(value, root=root, identity=identity)


@pytest.mark.parametrize("mode", ["auto", "logical"])
@pytest.mark.parametrize(
    "defect",
    ["missing-request", "wrong-mode", "numeric-request", "serial", "string-count"],
)
def test_automatic_worker_modes_require_an_explicit_matching_request(
    runtime_shard, mode, defect
):
    root, artifact, current, value = runtime_shard
    identity = identity_with_workers(root, current["identity"], mode)
    value["session"]["ci_context"] = ci_context(identity, "pytest")
    selection = value["session"]["selection"]
    selection.update(numprocesses=2, requested_numprocesses=mode)
    validate_shard_receipt(value, root=root, identity=identity)
    if defect == "missing-request":
        selection.pop("requested_numprocesses")
    elif defect == "wrong-mode":
        selection["requested_numprocesses"] = "logical" if mode == "auto" else "auto"
    elif defect == "numeric-request":
        selection["requested_numprocesses"] = 2
    else:
        selection["numprocesses"] = 0 if defect == "serial" else "2"
    (artifact / gate.PYTEST_RESULTS_NAME).write_text(
        json.dumps(value), encoding="utf-8"
    )
    with pytest.raises(gate.CoverageGateError, match="workers"):
        rebuild(root, artifact, identity)


@pytest.mark.parametrize(
    "invalid", [True, False, -1, 4.0, "", "-1", "4.0", "AUTO", "invalid", [], {}]
)
def test_invalid_worker_configuration_is_rejected_at_identity_and_config_inputs(
    runtime_shard, invalid
):
    root, _artifact, current, value = runtime_shard
    with pytest.raises(gate.CoverageGateError, match="workers"):
        identity_with_workers(root, current["identity"], invalid)
    current["identity"]["protocol"]["pytest_workers"] = invalid
    with pytest.raises(ValueError, match="workers"):
        validate_shard_receipt(value, root=root, identity=current["identity"])
    path = root / "config/ci-unit-gate.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["protocol"]["pytest_workers"] = invalid
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(gate.CoverageGateError, match="workers"):
        gate.load_config(root)


@pytest.mark.parametrize("missing", [True, False])
def test_worker_configuration_cannot_be_missing_or_null(runtime_shard, missing):
    root, _artifact, current, value = runtime_shard
    protocol = current["identity"]["protocol"]
    if missing:
        protocol.pop("pytest_workers")
    else:
        protocol["pytest_workers"] = None
    with pytest.raises(ValueError, match="workers"):
        validate_shard_receipt(value, root=root, identity=current["identity"])
    path = root / "config/ci-unit-gate.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if missing:
        config["protocol"].pop("pytest_workers")
    else:
        config["protocol"]["pytest_workers"] = None
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(gate.CoverageGateError, match="workers"):
        gate.load_config(root)


@pytest.mark.parametrize("invalid", ["", "-1", "4.0", "AUTO", "True"])
def test_invalid_environment_worker_setting_is_not_silently_defaulted(
    runtime_shard, monkeypatch, invalid
):
    root, _artifact, _current, _value = runtime_shard
    monkeypatch.setenv("PYTEST_XDIST_WORKERS", invalid)
    with pytest.raises(gate.CoverageGateError, match="workers"):
        gate.shard_identity(root, "runtime_0")


@pytest.mark.parametrize("configured", [0, 1, 4, "4", "04", "+4", "auto", "logical"])
def test_supported_worker_configuration_preserves_identity_fallback(
    runtime_shard, monkeypatch, configured
):
    root, _artifact, _current, _value = runtime_shard
    monkeypatch.delenv("PYTEST_XDIST_WORKERS", raising=False)
    path = root / "config/ci-unit-gate.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["protocol"]["pytest_workers"] = configured
    path.write_text(json.dumps(config), encoding="utf-8")
    assert gate.load_config(root)["protocol"]["pytest_workers"] == configured
    identity = gate.shard_identity(root, "runtime_0")
    assert identity["protocol"]["pytest_workers"] == str(configured), (
        "the default protocol representation remains compatible with existing identities"
    )
    assert gate.current_shard_identity(root, "runtime_0", identity) == identity, (
        "aggregate validation must resolve the same supported worker identity"
    )


@pytest.mark.parametrize("workers", ["", "-1", "4.0", "AUTO"])
def test_invalid_worker_request_stops_before_run_output_is_replaced(
    runtime_shard, workers
):
    root, artifact, _current, _value = runtime_shard
    config = gate.load_config(root)
    for group in ("coverage_excluded_files", "postgres_files", "fault_runner_files"):
        for relative in config["tests"][group]:
            path = root / relative
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("def test_fixture(): pass\n", encoding="utf-8")
    gate.validate_test_partition(root)
    report = artifact / gate.PYTEST_RESULTS_NAME
    original = report.read_bytes()
    with pytest.raises(gate.CoverageGateError, match="workers"):
        gate.run_shard(
            root=root,
            shard="runtime_0",
            python=sys.executable,
            artifact_root=artifact,
            workers=workers,
            distribution="worksteal",
            durations=50,
            include_stress=False,
        )
    assert report.read_bytes() == original, (
        "invalid worker input must be rejected before clearing or executing a shard"
    )


@pytest.mark.parametrize(
    "defect",
    [
        "schema-bool",
        "empty",
        "record-not-object",
        "missing-session",
        "source-drift",
        "session-failed",
        "session-exit-bool",
        "missing-setup",
        "missing-call",
        "missing-teardown",
        "setup-failed",
        "teardown-failed",
        "skipped",
        "skipped-pass-label",
        "collection-error",
        "collection-skip",
        "missing-clean-collection",
        "missing-discovery",
        "missing-collected-result",
        "partial-partition",
        "foreign-nodeid",
        "wrong-partition",
        "wrong-target",
        "missing-target-file",
        "keyword",
        "markexpr",
        "deselect",
        "wrong-shard-identity",
        "wrong-suite",
        "missing-start",
        "reversed-times",
        "duration-nan",
        "duration-negative",
        "duration-bool",
        "output-not-text",
    ],
)
def test_ci_rejects_incomplete_or_unbound_success_receipts(runtime_shard, defect):
    root, artifact, current, value = runtime_shard
    nodeid = next(iter(value["records"]))
    record = value["records"][nodeid]
    session = value["session"]
    if defect == "schema-bool":
        value["schema_version"] = True
    elif defect == "empty":
        value["records"] = {}
    elif defect == "record-not-object":
        value["records"][nodeid] = "PASS"
    elif defect == "missing-session":
        value.pop("session")
    elif defect == "source-drift":
        session["source_identity"] = "f" * 64
    elif defect.startswith("session-"):
        session["exitstatus"] = True if defect.endswith("bool") else 1
    elif defect in {"missing-setup", "missing-call", "missing-teardown"}:
        record["phases"].pop(defect.removeprefix("missing-"))
    elif defect in {"setup-failed", "teardown-failed"}:
        record["phases"][defect.removesuffix("-failed")] = "failed"
    elif defect in {"skipped", "skipped-pass-label"}:
        record["phases"]["call"] = "skipped"
        record["status"] = "SKIP" if defect == "skipped" else "PASS"
    elif defect == "collection-error":
        session["collection_errors"] = ["tests/test_broken.py"]
    elif defect == "collection-skip":
        session["collection_skips"] = ["tests/test_runtime.py"]
    elif defect == "missing-clean-collection":
        session.pop("collection_errors")
    elif defect == "missing-discovery":
        session.pop("discovered_nodeids")
    elif defect in {"missing-collected-result", "partial-partition"}:
        value["records"].pop(nodeid)
        if defect == "partial-partition":
            session["collected_nodeids"].remove(nodeid)
    elif defect == "foreign-nodeid":
        session["discovered_nodeids"].append("tests/other.py::test_foreign")
    elif defect == "wrong-partition":
        session["selection"]["partition"] = [3, 1]
    elif defect == "wrong-target":
        session["selection"]["targets"] = [nodeid]
    elif defect == "missing-target-file":
        session["collected_files"] = []
    elif defect in {"keyword", "markexpr", "deselect"}:
        session["selection"][defect] = [nodeid] if defect == "deselect" else "subset"
    elif defect == "wrong-shard-identity":
        session["ci_context"]["identity_sha256"] = "f" * 64
    elif defect == "wrong-suite":
        session["ci_context"]["suite"] = "postgres_stress"
    elif defect == "missing-start":
        session.pop("started_at")
    elif defect == "reversed-times":
        session["finished_at"] = "2020-01-01T00:00:00+00:00"
    elif defect.startswith("duration-"):
        record["duration_seconds"] = {
            "duration-nan": float("nan"),
            "duration-negative": -1,
            "duration-bool": True,
        }[defect]
    else:
        record["output"] = {}
    report = artifact / gate.PYTEST_RESULTS_NAME
    report.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(gate.CoverageGateError):
        rebuild(root, artifact, current["identity"])

    current["evidence"]["pytest_results"] = ci_gate_artifacts.evidence_entry(
        artifact, report
    )
    (artifact / gate.GATE_NAME).write_text(json.dumps(current), encoding="utf-8")
    with pytest.raises(gate.CoverageGateError):
        gate.verify_shard_gate(artifact / gate.GATE_NAME, artifact, source_root=root)


@pytest.mark.parametrize(
    "defect",
    ["missing", "skipped", "missing-call", "wrong-suite", "low-rounds", "parallel"],
)
def test_postgres_contract_cannot_substitute_for_mandatory_stress(
    tmp_path, monkeypatch, defect
):
    root = _committed_root(tmp_path)
    artifact = tmp_path / "postgres"
    current = _build_shard(root, artifact, "postgres", monkeypatch)
    report = artifact / gate.STRESS_RESULTS_NAME
    value = json.loads(report.read_text(encoding="utf-8"))
    record = next(iter(value["records"].values()))
    if defect == "missing":
        with pytest.raises(gate.CoverageGateError, match="stress evidence is required"):
            gate.build_shard_gate(
                root,
                artifact,
                identity=current["identity"],
                coverage_data=artifact / gate.COVERAGE_DATA_NAME,
                pytest_results=artifact / gate.PYTEST_RESULTS_NAME,
                durations=artifact / gate.DURATIONS_NAME,
            )
        return
    if defect == "skipped":
        record["status"] = "SKIP"
        record["phases"]["call"] = "skipped"
    elif defect == "missing-call":
        record["phases"].pop("call")
    elif defect == "wrong-suite":
        value["session"]["ci_context"]["suite"] = "pytest"
    elif defect == "low-rounds":
        value["session"]["selection"]["stress_rounds"] = "1"
    else:
        value["session"]["selection"]["numprocesses"] = 4
    report.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(gate.CoverageGateError):
        rebuild(root, artifact, current["identity"])


@pytest.mark.parametrize("suite", ["pytest", "postgres_stress"])
@pytest.mark.parametrize("field", ["numprocesses", "requested_numprocesses"])
@pytest.mark.parametrize("reported", [1, -1, True, False, "0", 0.0, None])
def test_postgres_worker_receipts_remain_strictly_serial(
    tmp_path, monkeypatch, suite, field, reported
):
    root = _committed_root(tmp_path)
    artifact = tmp_path / "postgres"
    current = _build_shard(root, artifact, "postgres", monkeypatch)
    report = artifact / (
        gate.STRESS_RESULTS_NAME
        if suite == "postgres_stress"
        else gate.PYTEST_RESULTS_NAME
    )
    value = json.loads(report.read_text(encoding="utf-8"))
    value["session"]["selection"]["requested_numprocesses"] = 0
    validate_shard_receipt(value, root=root, identity=current["identity"], suite=suite)
    value["session"]["selection"][field] = reported
    report.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(gate.CoverageGateError, match="must run serially"):
        rebuild(root, artifact, current["identity"])


@pytest.mark.parametrize("workers", ["0", "1", "4", "auto", "logical"])
@pytest.mark.parametrize("suite", ["pytest", "postgres_stress"])
def test_postgres_serial_workers_are_independent_of_nonpostgres_worker_budget(
    tmp_path, monkeypatch, workers, suite
):
    root = _committed_root(tmp_path)
    artifact = tmp_path / "postgres"
    current = _build_shard(root, artifact, "postgres", monkeypatch)
    identity = identity_with_workers(root, current["identity"], workers)
    report = artifact / (
        gate.STRESS_RESULTS_NAME
        if suite == "postgres_stress"
        else gate.PYTEST_RESULTS_NAME
    )
    value = json.loads(report.read_text(encoding="utf-8"))
    value["session"]["ci_context"] = ci_context(identity, suite)
    value["session"]["selection"]["requested_numprocesses"] = 0
    validate_shard_receipt(value, root=root, identity=identity, suite=suite)


def test_reuse_keeps_original_execution_across_content_equivalent_checkouts(
    runtime_shard, monkeypatch
):
    root, artifact, current, value = runtime_shard
    original = copy.deepcopy(current["execution"])
    report_bytes = (artifact / gate.PYTEST_RESULTS_NAME).read_bytes()
    for run_id in ("124", "125"):
        (artifact / gate.GATE_NAME).replace(artifact / gate.BASE_GATE_NAME)
        (root / "docs/guide.md").write_text(
            f"documentation run {run_id}\n", encoding="utf-8"
        )
        assert source_identity(root) != value["source_identity"], (
            "the whole checkout identity really differs from the original execution"
        )
        identity = _identity(root, "runtime_0")
        assert identity == current["identity"], (
            "unchanged slice content remains authorized for historical shard reuse"
        )
        _set_main_environment(monkeypatch, run_id=run_id)
        current = rebuild(root, artifact, identity)
        assert current["producer"]["run_id"] == run_id, (
            "the new attestation names its run"
        )
        assert current["execution"] == original, (
            "re-signing cannot relabel the execution"
        )
        assert (artifact / gate.PYTEST_RESULTS_NAME).read_bytes() == report_bytes, (
            "all original source, session, selection and discovery facts stay intact"
        )
        gate.verify_shard_gate(
            artifact / gate.GATE_NAME,
            artifact,
            source_root=root,
            expected_identity=identity["sha256"],
        )


def test_reuse_rejects_a_changed_slice_even_with_an_old_complete_receipt(runtime_shard):
    root, artifact, _current, _value = runtime_shard
    (artifact / gate.GATE_NAME).replace(artifact / gate.BASE_GATE_NAME)
    (root / "src/gpu_fault/runtime.py").write_text(
        "VALUE = 'changed'\n", encoding="utf-8"
    )
    with pytest.raises(gate.CoverageGateError, match="shard content identity"):
        rebuild(root, artifact, _identity(root, "runtime_0"))


@pytest.fixture
def aggregate(tmp_path, monkeypatch):
    root = _committed_root(tmp_path)
    gates = {}
    for shard in gate.SHARDS:
        artifact = tmp_path / "shards" / shard
        current = _build_shard(root, artifact, shard, monkeypatch)
        gates[shard] = (artifact / gate.GATE_NAME, current)
    return (
        root,
        gates,
        aggregate_pytest_results(
            root, gates, resolve_identity=gate.current_shard_identity
        ),
    )


def test_aggregate_preserves_complete_discovery_and_original_provenance(aggregate):
    root, gates, value = aggregate
    receipt = parse_pytest_receipt(
        value,
        root=root,
        expected_identity=source_identity(root),
        aggregate_parser=gate.parse_combined_pytest_receipt,
    )
    assert receipt.discovered_nodeids == frozenset(receipt.records), (
        "the union of legitimate partitions must execute every discovered test"
    )
    assert "source_identity" not in value and "session" not in value, (
        "an aggregate must not impersonate a new pytest execution"
    )
    for shard, (path, current) in gates.items():
        original = json.loads((path.parent / gate.PYTEST_RESULTS_NAME).read_text())
        provenance = value["shards"][shard]
        assert provenance["session"] == original["session"], (
            "each original discovery and selection receipt survives the merge"
        )
        assert provenance["source_identity"] == original["source_identity"], (
            "historical source identities are provenance, not rewritten current identities"
        )
        assert provenance["execution"] == current["execution"], (
            "the original producer is retained independently of the current attestation"
        )
    with pytest.raises(ValueError, match="not a fresh pytest session"):
        parse_pytest_receipt(
            value,
            root=root,
            expected_identity=source_identity(root),
            require_session=True,
        )


def test_aggregate_validates_reused_slices_without_relabeling_original_source(
    aggregate, monkeypatch
):
    root, gates, previous = aggregate
    old_source = previous["validated_source_identity"]
    (root / "docs/guide.md").write_text(
        "a new documentation-only checkout\n", encoding="utf-8"
    )
    _set_main_environment(monkeypatch, run_id="124")
    for shard, (path, current) in list(gates.items()):
        path.replace(path.parent / gate.BASE_GATE_NAME)
        identity = _identity(root, shard)
        assert identity == current["identity"], (
            "documentation does not invalidate test slices"
        )
        gates[shard] = (path, rebuild(root, path.parent, identity))
    value = aggregate_pytest_results(
        root, gates, resolve_identity=gate.current_shard_identity
    )
    assert value["validated_source_identity"] != old_source, (
        "validation names the new checkout, not the historical whole source"
    )
    assert all(
        provenance["source_identity"] == old_source
        and provenance["execution"]["producer"]["run_id"] == "123"
        and provenance["producer"]["run_id"] == "124"
        for provenance in value["shards"].values()
    ), "reused executions keep their actual producer and source provenance"
    receipt = parse_pytest_receipt(
        value,
        root=root,
        expected_identity=source_identity(root),
        aggregate_parser=gate.parse_combined_pytest_receipt,
    )
    assert receipt.discovered_nodeids == frozenset(receipt.records), (
        "content-equivalent reuse still proves a complete combined collection"
    )


@pytest.mark.parametrize(
    "defect",
    [
        "missing-shard",
        "missing-result",
        "missing-stress",
        "relabel",
        "source-drift",
        "discovery-drift",
    ],
)
def test_aggregate_cannot_hide_incomplete_or_incompatible_receipts(aggregate, defect):
    root, _gates, value = aggregate
    if defect == "missing-shard":
        value["shards"].pop("runtime_2")
    elif defect == "missing-result":
        value["records"].pop(next(iter(value["records"])))
    elif defect == "missing-stress":
        value["shards"]["postgres"]["postgres_stress"] = None
    elif defect == "relabel":
        value["source_identity"] = source_identity(root)
    elif defect == "source-drift":
        (root / "src/gpu_fault/runtime.py").write_text(
            "VALUE = 'changed'\n", encoding="utf-8"
        )
        value["validated_source_identity"] = source_identity(root)
    else:
        provenance = value["shards"]["runtime_0"]
        session = provenance["session"]
        unselected = next(
            nodeid
            for nodeid in session["discovered_nodeids"]
            if nodeid not in session["collected_nodeids"]
        )
        session["discovered_nodeids"].remove(unselected)
    with pytest.raises(ValueError):
        parse_pytest_receipt(
            value,
            root=root,
            expected_identity=source_identity(root),
            aggregate_parser=gate.parse_combined_pytest_receipt,
        )


@pytest.mark.parametrize("shard", gate.SHARDS)
@pytest.mark.parametrize(
    ("field", "reported"),
    [
        ("numprocesses", 1),
        ("numprocesses", True),
        ("numprocesses", "4"),
        ("requested_numprocesses", True),
        ("requested_numprocesses", "4"),
    ],
)
def test_aggregate_revalidates_worker_binding_on_creation_and_reuse(
    aggregate, shard, field, reported
):
    root, gates, value = aggregate
    value["shards"][shard]["session"]["selection"][field] = reported
    with pytest.raises(ValueError, match="workers|serially"):
        parse_pytest_receipt(
            value,
            root=root,
            expected_identity=source_identity(root),
            aggregate_parser=gate.parse_combined_pytest_receipt,
        )
    path, current = gates[shard]
    report = path.parent / gate.PYTEST_RESULTS_NAME
    original = json.loads(report.read_text(encoding="utf-8"))
    original["session"]["selection"][field] = reported
    report.write_text(json.dumps(original), encoding="utf-8")
    current["evidence"]["pytest_results"] = ci_gate_artifacts.evidence_entry(
        path.parent, report
    )
    with pytest.raises(ValueError, match="workers|serially"):
        aggregate_pytest_results(
            root, gates, resolve_identity=gate.current_shard_identity
        )


@pytest.mark.parametrize("field", ["numprocesses", "requested_numprocesses"])
def test_aggregate_revalidates_postgres_stress_serial_workers(aggregate, field):
    root, _gates, value = aggregate
    stress = value["shards"]["postgres"]["postgres_stress"]["receipt"]
    stress["session"]["selection"][field] = False
    with pytest.raises(ValueError, match="must run serially"):
        gate.parse_combined_pytest_receipt(value, root, source_identity(root))


@pytest.mark.parametrize("mode", ["auto", "logical"])
def test_aggregate_reuses_automatic_workers_without_relabeling_the_request(
    aggregate, monkeypatch, mode
):
    root, gates, _value = aggregate
    for shard, (path, current) in list(gates.items()):
        if shard == "postgres":
            continue
        identity = identity_with_workers(root, current["identity"], mode)
        report = path.parent / gate.PYTEST_RESULTS_NAME
        original = json.loads(report.read_text(encoding="utf-8"))
        original["session"]["ci_context"] = ci_context(identity, "pytest")
        original["session"]["selection"].update(
            numprocesses=2, requested_numprocesses=mode
        )
        report.write_text(json.dumps(original), encoding="utf-8")
        gates[shard] = (path, rebuild(root, path.parent, identity))
    previous = aggregate_pytest_results(
        root, gates, resolve_identity=gate.current_shard_identity
    )
    (root / "docs/guide.md").write_text("new documentation\n", encoding="utf-8")
    _set_main_environment(monkeypatch, run_id="124")
    for shard, (path, current) in list(gates.items()):
        path.replace(path.parent / gate.BASE_GATE_NAME)
        gates[shard] = (path, rebuild(root, path.parent, current["identity"]))
    combined = aggregate_pytest_results(
        root, gates, resolve_identity=gate.current_shard_identity
    )
    receipt = gate.parse_combined_pytest_receipt(combined, root, source_identity(root))
    assert receipt.discovered_nodeids == frozenset(receipt.records), (
        "supported automatic modes must retain complete aggregate coverage"
    )
    for shard, provenance in combined["shards"].items():
        original = previous["shards"][shard]
        assert provenance["session"] == original["session"], (
            "reuse must not recompute a producer's automatic worker count or request"
        )
        assert provenance["execution"] == original["execution"], (
            "worker validation must preserve original execution provenance"
        )
        assert provenance["producer"]["run_id"] == "124", (
            "only the current attestation, not the original execution, names the new run"
        )


def test_aggregate_requires_an_explicit_current_content_validator(aggregate):
    root, _gates, value = aggregate
    with pytest.raises(ValueError, match="content-identity validator"):
        parse_pytest_receipt(value, root=root, expected_identity=source_identity(root))


def test_a_valid_hash_partition_is_accepted_without_executing_other_partitions(
    runtime_shard,
):
    root, _artifact, current, value = runtime_shard
    receipt = validate_shard_receipt(value, root=root, identity=current["identity"])
    assert len(receipt.records) < len(receipt.discovered_nodeids or ()), (
        "authorized hash deselection must not be confused with missing execution"
    )


def real_reporter_receipt(
    tmp_path, workers, *, extra_args=(), auto_workers=2, xdist=True
):
    root = _committed_root(tmp_path)
    (root / "tests/test_runtime.py").write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('value', range(12))\n"
        "def test_runtime(value):\n"
        "    assert value >= 0\n",
        encoding="utf-8",
    )
    identity = gate.shard_identity(
        root, "runtime_0", pytest_workers=str(workers) if workers is not None else "0"
    )
    report = tmp_path / "report.json"
    environment = {
        "HOME": "/tmp",
        "PATH": os.defpath,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_XDIST_AUTO_NUM_WORKERS": str(auto_workers),
        "PYTEST_GPU_FAULT_CASE_REPORT": str(report),
        "PYTEST_GPU_FAULT_PARTITION_COUNT": "3",
        "PYTEST_GPU_FAULT_PARTITION_INDEX": "0",
        CI_CONTEXT_ENV: json.dumps(ci_context(identity, "pytest")),
        "GPU_FAULT_TEST_POSTGRES_URL": "",
        "AWS_CONFIG_FILE": os.devnull,
        "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
        "AWS_EC2_METADATA_DISABLED": "true",
        "KUBECONFIG": os.devnull,
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *gate.pytest_targets(root, "runtime_0"),
            "-p",
            "tools.pytest_case_reporter",
            *(["-p", "xdist.plugin"] if xdist else []),
            "-p",
            "no:cacheprovider",
            *(["-n", str(workers)] if workers is not None else []),
            "-o",
            "addopts=",
            *extra_args,
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    value = json.loads(report.read_text(encoding="utf-8"))
    return root, identity, value


@pytest.mark.parametrize("workers", [0, 2])
def test_real_reporter_emits_verifiable_ci_selection(tmp_path, workers):
    root, identity, value = real_reporter_receipt(tmp_path, workers)
    receipt = validate_shard_receipt(value, root=root, identity=identity)
    assert len(receipt.discovered_nodeids or ()) == 12, (
        "worker or serial collection preserves every unfiltered parameter variant"
    )
    assert value["session"]["selection"]["numprocesses"] == workers, (
        "the receipt records the actual pytest worker setting"
    )
    assert value["session"]["selection"]["requested_numprocesses"] == workers, (
        "the receipt retains the requested setting independently of the worker pool"
    )


@pytest.mark.parametrize("mode", ["auto", "logical"])
@pytest.mark.parametrize("capped", [False, True])
def test_real_reporter_retains_automatic_mode_and_effective_worker_pool(
    tmp_path, mode, capped
):
    root, identity, value = real_reporter_receipt(
        tmp_path, mode, extra_args=("--maxprocesses=1",) if capped else ()
    )
    receipt = validate_shard_receipt(value, root=root, identity=identity)
    selection = value["session"]["selection"]
    assert selection["requested_numprocesses"] == mode, (
        "xdist's automatic resolution must not erase the originally requested mode"
    )
    assert type(selection["numprocesses"]) is int
    assert selection["numprocesses"] == (1 if capped else 2), (
        "the receipt must count the resolved pool after xdist's maximum-worker cap"
    )
    assert len(receipt.discovered_nodeids or ()) == 12, (
        "automatic worker metadata must not change complete selection evidence"
    )


def test_real_reporter_capped_numeric_pool_cannot_claim_either_worker_identity(
    tmp_path,
):
    root, identity, value = real_reporter_receipt(
        tmp_path, 4, extra_args=("--maxprocesses=2",)
    )
    selection = value["session"]["selection"]
    assert selection["numprocesses"] == 2
    assert selection["requested_numprocesses"] == 4, (
        "actual and requested counts must remain independent facts"
    )
    with pytest.raises(ValueError, match="workers"):
        validate_shard_receipt(value, root=root, identity=identity)
    changed = identity_with_workers(root, identity, "2")
    value["session"]["ci_context"] = ci_context(changed, "pytest")
    with pytest.raises(ValueError, match="workers"):
        validate_shard_receipt(value, root=root, identity=changed)


@pytest.mark.parametrize("mode", ["auto", "logical"])
def test_automatic_mode_resolving_to_no_workers_is_not_parallel_proof(tmp_path, mode):
    root, identity, value = real_reporter_receipt(tmp_path, mode, auto_workers=0)
    selection = value["session"]["selection"]
    assert selection["requested_numprocesses"] == mode
    assert selection["numprocesses"] == 0, (
        "the reporter records serial execution even when automatic mode requested it"
    )
    with pytest.raises(ValueError, match="workers"):
        validate_shard_receipt(value, root=root, identity=identity)


@pytest.mark.parametrize("xdist", [False, True])
def test_real_reporter_without_worker_option_proves_serial_execution(tmp_path, xdist):
    root, identity, value = real_reporter_receipt(tmp_path, None, xdist=xdist)
    selection = value["session"]["selection"]
    assert type(selection["numprocesses"]) is int
    assert selection["numprocesses"] == 0
    assert selection["requested_numprocesses"] == 0
    validate_shard_receipt(value, root=root, identity=identity)


@pytest.mark.parametrize(
    "defect",
    ["statement-branch-94", "old-floor", "runner-root", "runner-identity", "stress"],
)
def test_config_rejects_a_weakened_ci_protocol(tmp_path, defect):
    root = _committed_root(tmp_path)
    path = root / "config/ci-unit-gate.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if defect == "statement-branch-94":
        config["coverage"]["objective_floor"] = 94
    elif defect == "old-floor":
        config["coverage"]["floor"] = 77
    elif defect == "runner-root":
        config["coverage"]["sources"].remove("tools")
    elif defect == "runner-identity":
        config["shards"]["runtime_0"]["identity_groups"].remove("fault_runner_source")
    else:
        config["protocol"]["postgres_stress_rounds"] = 1
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(gate.CoverageGateError):
        gate.load_config(root)


@pytest.mark.parametrize("scope", ["production", "runner"])
def test_normal_combine_rejects_independent_branch_regression(
    aggregate, tmp_path, scope
):
    root, gates, _value = aggregate
    relative = (
        "src/gpu_fault/runtime.py"
        if scope == "production"
        else "tools/case_scheduler.py"
    )
    for _shard, (path, current) in gates.items():
        coverage_path = path.parent / gate.COVERAGE_DATA_NAME
        data = CoverageData(basename=str(coverage_path))
        data.read()
        arcs = {name: set(data.arcs(name) or ()) for name in data.measured_files()}
        arcs[relative] -= {(2, 4), (4, -1)}
        data.erase()
        replacement = CoverageData(basename=str(coverage_path))
        replacement.add_arcs(arcs)
        replacement.write()
        rebuild(root, path.parent, current["identity"])
    with pytest.raises(gate.CoverageGateError, match=scope + " requires >= 95"):
        gate.combine_shards(
            root=root,
            shards_root=next(iter(gates.values()))[0].parent.parent,
            output_root=tmp_path / "combined",
            python=sys.executable,
            require_run_id="123",
        )
    assert not (tmp_path / "combined/pytest-case-results.json").exists(), (
        "a failed independent coverage floor cannot publish combined PASS evidence"
    )


@pytest.mark.parametrize("scope", ["production", "runner"])
def test_unit_gate_cannot_attest_a_94_percent_branch_report(aggregate, tmp_path, scope):
    root, gates, _value = aggregate
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    coverage, results, fault, durations = _unit_evidence(
        evidence_root, source_root=root, gates=gates
    )
    value = json.loads(coverage.read_text(encoding="utf-8"))
    for name, report in value["files"].items():
        if any(name.startswith(prefix + "/") for prefix in gate.SCOPES[scope]):
            report["summary"].update(covered_branches=94, missing_branches=6)
    coverage.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ci_unit_gate.UnitGateError, match=scope + " requires >= 95"):
        ci_unit_gate.build_unit_gate(
            root,
            tmp_path / "unit",
            shards_root=next(iter(gates.values()))[0].parent.parent,
            coverage_summary=coverage,
            pytest_results=results,
            fault_report=fault,
            durations=durations,
            require_run_id="123",
        )


def test_static_promql_evidence_never_enters_reusable_shard_identity(tmp_path):
    root = _committed_root(tmp_path)
    names = (
        "tests/metrics/test_closed_loop_promql.py",
        "tests/test_alert_rules_promtool.py",
    )
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_native(): pass\n", encoding="utf-8")
    before = {shard: _identity(root, shard)["sha256"] for shard in gate.SHARDS}
    for name in names:
        (root / name).write_text("def test_changed_native(): pass\n", encoding="utf-8")
    for shard in gate.SHARDS:
        assert not set(names) & set(gate.pytest_targets(root, shard)), (
            "native-tool behavior belongs to the always-fresh static gate"
        )
        assert _identity(root, shard)["sha256"] == before[shard], (
            "no cached shard may claim native-tool results from an unbound environment"
        )
