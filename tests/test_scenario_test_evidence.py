from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._script_loader import load_script_module
from tools.pytest_result_identity import PytestReceipt, normalized_pytest_nodeid
from tools.scenario_requirements import Check
from tools.scenario_test_evidence import ROOT, PytestEvidence, load_test_evidence

NOW = datetime(2026, 9, 12, 12, tzinfo=timezone.utc)
IDENTITY = "a" * 64
SELECTOR = "tests/test_sample.py::test_result"
PHASES = {"setup": "passed", "call": "passed", "teardown": "passed"}
# Exercise the reporter as an isolated script, not the active pytest plugin.
reporter = load_script_module(
    Path(__file__).resolve().parents[1] / "tools/pytest_case_reporter.py"
)


@pytest.fixture
def payload() -> dict:
    return {
        "schema_version": 1,
        "source_identity": IDENTITY,
        "session": {
            "source_identity": IDENTITY,
            "started_at": (NOW - timedelta(minutes=2)).isoformat(),
            "finished_at": (NOW - timedelta(minutes=1)).isoformat(),
            "exitstatus": 0,
            "collected_nodeids": [SELECTOR],
            "discovered_nodeids": [SELECTOR],
            "collection_errors": [],
            "collection_skips": [],
        },
        "records": {SELECTOR: {"status": "PASS", "phases": PHASES}},
    }


def load(tmp_path: Path, payload: dict):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_test_evidence([path], expected_identity=IDENTITY, now=NOW)


def test_full_phases_and_exact_collection_confirm_one_check(tmp_path, payload) -> None:
    evidence = load(tmp_path, payload)
    assert evidence.confirms(Check(nodeid=SELECTOR)), (
        "complete local execution is counted"
    )
    assert not evidence.confirms(Check(nodeid=SELECTOR + "_other")), (
        "prefix collisions are not matches"
    )
    assert not evidence.confirms(Check(nodeid=SELECTOR, expected=2)), (
        "missing variants are not inferred"
    )


def test_parameterized_selector_requires_every_expected_result(
    tmp_path, payload
) -> None:
    nodes = [SELECTOR + "[first]", SELECTOR + "[second]"]
    payload["session"]["collected_nodeids"] = nodes
    payload["session"]["discovered_nodeids"] = nodes
    payload["records"] = {node: {"status": "PASS", "phases": PHASES} for node in nodes}
    evidence = load(tmp_path, payload)
    assert evidence.confirms(Check(nodeid=SELECTOR, expected=2)), (
        "all expected variants passed"
    )
    payload["records"][nodes[1]]["status"] = "FAIL"
    assert not load(tmp_path, payload).confirms(Check(nodeid=SELECTOR, expected=2)), (
        "one failed or skipped variant prevents the whole requirement from passing"
    )


@pytest.mark.parametrize(
    "field", ["source_identity", "schema_version", "session", "records"]
)
@pytest.mark.parametrize("value", [None, True, [], "invalid"])
def test_malformed_report_is_not_local_verification(
    tmp_path, payload, field, value
) -> None:
    payload[field] = value
    with pytest.raises(ValueError):
        load(tmp_path, payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_identity", "b" * 64),
        ("exitstatus", 1),
        ("exitstatus", False),
        ("collected_nodeids", []),
        ("collected_nodeids", [SELECTOR, SELECTOR]),
        ("collected_nodeids", ["file_without_function"]),
        ("collected_nodeids", [True]),
        ("discovered_nodeids", None),
        ("discovered_nodeids", []),
        ("discovered_nodeids", [True]),
        ("discovered_nodeids", [SELECTOR, SELECTOR]),
        ("discovered_nodeids", [SELECTOR + "_other"]),
        ("collection_errors", ["test_collection_failure.py"]),
        ("collection_errors", None),
        ("collection_skips", ["test_skipped_module.py"]),
        ("collection_skips", None),
        ("collection_skips", [True]),
        ("started_at", None),
        ("started_at", "2026-09-12T11:00:00"),
        ("started_at", "2026-01-01T11:00:00+00:00"),
        ("finished_at", "2026-09-12T13:00:00+00:00"),
        ("finished_at", "2026-09-12T01:00:00+00:00"),
    ],
)
def test_unbound_filtered_or_stale_session_is_refused(
    tmp_path, payload, field, value
) -> None:
    payload["session"][field] = value
    with pytest.raises(ValueError):
        load(tmp_path, payload)


@pytest.mark.parametrize(
    "record",
    [None, {}, {"status": "PASS"}, {"status": "PASS", "phases": {"call": "passed"}}],
)
def test_incomplete_test_phases_do_not_manufacture_a_pass(
    tmp_path, payload, record
) -> None:
    payload["records"][SELECTOR] = record
    with pytest.raises(ValueError):
        load(tmp_path, payload)


def test_unexecuted_collected_test_is_refused(tmp_path, payload) -> None:
    payload["records"].clear()
    with pytest.raises(ValueError, match="complete collection"):
        load(tmp_path, payload)


@pytest.mark.parametrize(
    "field", ["discovered_nodeids", "collection_errors", "collection_skips"]
)
def test_missing_discovery_metadata_cannot_establish_scenario_coverage(
    tmp_path, payload, field
) -> None:
    del payload["session"][field]
    with pytest.raises(ValueError):
        load(tmp_path, payload)


@pytest.mark.parametrize("expected", [1, 2])
def test_filtered_discovered_variant_cannot_pass_even_with_an_understated_count(
    tmp_path, payload, expected
) -> None:
    first, second = SELECTOR + "[first]", SELECTOR + "[second]"
    payload["session"]["collected_nodeids"] = [first]
    payload["session"]["discovered_nodeids"] = [first, second]
    payload["records"] = {first: {"status": "PASS", "phases": PHASES}}
    evidence = load(tmp_path, payload)
    assert not evidence.confirms(Check(nodeid=SELECTOR, expected=expected)), (
        "declared counts cannot erase a discovered but unexecuted parameter"
    )
    assert evidence.confirms(Check(nodeid=first)), (
        "an explicit completed parameter retains its own narrower proof"
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_complementary_shards_can_complete_all_discovered_variants(
    tmp_path, payload, reverse
) -> None:
    nodes = [SELECTOR + "[first]", SELECTOR + "[second]"]
    paths = []
    for index, node in enumerate(nodes):
        shard = deepcopy(payload)
        shard["session"]["collected_nodeids"] = [node]
        shard["session"]["discovered_nodeids"] = nodes
        shard["records"] = {node: {"status": "PASS", "phases": PHASES}}
        path = tmp_path / f"shard-{index}.json"
        path.write_text(json.dumps(shard), encoding="utf-8")
        paths.append(path)
    if reverse:
        paths.reverse()
    evidence = load_test_evidence(paths, expected_identity=IDENTITY, now=NOW)
    assert evidence.confirms(Check(nodeid=SELECTOR, expected=2)), (
        "valid shards may supply complementary three-phase results"
    )
    assert not evidence.confirms(Check(nodeid=SELECTOR)), (
        "merging shards must preserve the complete discovered denominator"
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_path_alias_in_another_shard_cannot_hide_a_failed_result(
    tmp_path, payload, reverse
) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps(payload), encoding="utf-8")
    alias = normalized_pytest_nodeid(SELECTOR, root=ROOT)
    payload["session"]["collected_nodeids"] = [alias]
    payload["session"]["discovered_nodeids"] = [alias]
    payload["records"] = {alias: {"status": "FAIL", "phases": PHASES}}
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    paths = [bad, good] if reverse else [good, bad]
    evidence = load_test_evidence(paths, expected_identity=IDENTITY, now=NOW)
    assert not evidence.confirms(Check(nodeid=SELECTOR)), (
        "relative and absolute names of the same test must share failure precedence"
    )


@pytest.mark.parametrize("missing", ["discovery", "clean-collection"])
def test_direct_receipt_model_without_complete_discovery_cannot_confirm_a_check(
    missing,
) -> None:
    nodeid = normalized_pytest_nodeid(SELECTOR, root=ROOT)
    receipt = PytestReceipt(
        {nodeid: {"status": "PASS", "phases": PHASES}},
        None if missing == "discovery" else frozenset({nodeid}),
        ("skipped",) if missing == "clean-collection" else (),
    )
    assert not PytestEvidence(receipt, ROOT).confirms(Check(nodeid=SELECTOR)), (
        "in-memory models must not restore legacy or skipped discovery as proof"
    )


def test_merging_reports_does_not_hide_an_earlier_failure(tmp_path, payload) -> None:
    good = tmp_path / "good.json"
    good.write_text(json.dumps(payload), encoding="utf-8")
    payload["records"][SELECTOR]["status"] = "FAIL"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    for paths in ([good, bad], [bad, good]):
        evidence = load_test_evidence(paths, expected_identity=IDENTITY, now=NOW)
        assert not evidence.confirms(Check(nodeid=SELECTOR)), (
            "merge order cannot conceal failure"
        )


@pytest.mark.parametrize(
    ("now", "max_age"),
    [(NOW.replace(tzinfo=None), timedelta(days=7)), (NOW, timedelta(0))],
)
def test_invalid_freshness_policy_is_rejected(now, max_age) -> None:
    with pytest.raises(ValueError, match="freshness bound"):
        load_test_evidence([], expected_identity=IDENTITY, now=now, max_age=max_age)


@pytest.mark.parametrize("worker", [False, True])
def test_reporter_binds_start_end_source_and_collected_tests(
    tmp_path, monkeypatch, worker
) -> None:
    path = tmp_path / "pytest.json"
    monkeypatch.setenv(reporter.REPORT_ENV, str(path))
    monkeypatch.setattr(reporter, "source_identity", lambda root: IDENTITY)
    monkeypatch.setattr(reporter, "repository_root", lambda root: root.resolve())
    monkeypatch.chdir(tmp_path)
    reporter.pytest_configure(SimpleNamespace())
    if worker:
        reporter.pytest_xdist_node_collection_finished(SimpleNamespace(), [SELECTOR])
    else:
        reporter.pytest_collection_finish(
            SimpleNamespace(items=[SimpleNamespace(nodeid=SELECTOR)])
        )
    for phase in PHASES:
        reporter.pytest_runtest_logreport(
            SimpleNamespace(
                nodeid=SELECTOR,
                when=phase,
                outcome="passed",
                duration=0.01,
                capstdout="",
                capstderr="",
                failed=False,
            )
        )
    reporter.pytest_sessionfinish(
        SimpleNamespace(config=SimpleNamespace(), exitstatus=0)
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["session"]["source_identity"] == raw["source_identity"] == IDENTITY, (
        "test execution binds both sides of the source snapshot"
    )
    assert raw["session"]["collected_nodeids"] == [SELECTOR], "collection is explicit"
    assert raw["records"][SELECTOR]["phases"] == PHASES, "all test phases are recorded"


def test_mid_run_source_change_cannot_be_relabelled_as_fresh_success(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "pytest.json"
    monkeypatch.setenv(reporter.REPORT_ENV, str(path))
    monkeypatch.setattr(reporter, "source_identity", lambda root: IDENTITY)
    monkeypatch.setattr(reporter, "repository_root", lambda root: root.resolve())
    monkeypatch.chdir(tmp_path)
    reporter.pytest_configure(SimpleNamespace())
    monkeypatch.setattr(reporter, "source_identity", lambda root: "b" * 64)
    reporter.pytest_sessionfinish(
        SimpleNamespace(config=SimpleNamespace(), exitstatus=0)
    )
    with pytest.raises(ValueError, match="source changed"):
        load_test_evidence(
            [path], expected_identity="b" * 64, now=datetime.now(timezone.utc)
        )
