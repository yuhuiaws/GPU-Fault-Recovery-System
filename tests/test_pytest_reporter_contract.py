from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests._script_loader import load_script_module

REPORTER = load_script_module(
    Path(__file__).resolve().parents[1] / "tools/pytest_case_reporter.py"
)


@pytest.fixture
def reporter(tmp_path, monkeypatch):
    monkeypatch.delenv(REPORTER.PARTITION_COUNT_ENV, raising=False)
    monkeypatch.delenv(REPORTER.PARTITION_INDEX_ENV, raising=False)
    monkeypatch.setenv(REPORTER.REPORT_ENV, str(tmp_path / "results.json"))
    monkeypatch.setattr(REPORTER, "source_identity", lambda root: "a" * 64)
    monkeypatch.setattr(REPORTER, "repository_root", lambda root: root.resolve())
    REPORTER.pytest_configure(SimpleNamespace())
    return REPORTER


def emit(
    reporter, *, nodeid="tests/test_sample.py::test_case", outcome="passed", output=""
):
    reporter.pytest_runtest_logreport(
        SimpleNamespace(
            nodeid=nodeid,
            when="call",
            outcome=outcome,
            duration=0.25,
            capstdout=output,
            capstderr="",
            failed=outcome == "failed",
            skipped=outcome == "skipped",
            longrepr="assertion diagnostic" if outcome != "passed" else "",
        )
    )


def test_collected_test_reports_reuse_paths_without_runtime_filesystem_reads(
    reporter, monkeypatch
) -> None:
    nodeids = [
        "tests/test_sample.py::test_case[first]",
        "tests/test_sample.py::test_case[second]",
    ]
    reporter.pytest_collection_finish(
        SimpleNamespace(items=[SimpleNamespace(nodeid=nodeid) for nodeid in nodeids])
    )

    def forbidden_path(*_args, **_kwargs):
        raise AssertionError(
            "runtest reporting must not resolve the collected path again"
        )

    with monkeypatch.context() as patch:
        patch.setattr(reporter, "Path", forbidden_path)
        for nodeid in nodeids:
            emit(reporter, nodeid=nodeid)
    assert set(reporter.REPORTS) == set(nodeids), (
        "both collected variants must retain their established source paths"
    )


def test_a_new_collection_root_invalidates_previous_path_mappings(
    reporter, tmp_path, monkeypatch
) -> None:
    nodeid = "tests/test_sample.py::test_case"
    session = SimpleNamespace(items=[SimpleNamespace(nodeid=nodeid)])
    reporter.pytest_collection_finish(session)
    emit(reporter, nodeid=nodeid)
    assert set(reporter.REPORTS) == {nodeid}, "the first session uses its own root"
    monkeypatch.chdir(tmp_path)
    reporter.pytest_configure(SimpleNamespace(rootpath=tmp_path / "alternate"))
    reporter.pytest_collection_finish(session)
    emit(reporter, nodeid=nodeid)
    assert set(reporter.REPORTS) == {"alternate/" + nodeid}, (
        "a new source and collection root must not reuse an old path mapping"
    )


@pytest.mark.parametrize("outcome", ["passed", "failed", "skipped"])
def test_phase_outcomes_and_diagnostics_are_retained(reporter, outcome) -> None:
    emit(reporter, outcome=outcome, output="captured diagnostic")
    record = reporter.REPORTS["tests/test_sample.py::test_case"]
    assert record["status"] == ("PASS" if outcome == "passed" else "FAIL"), (
        "skips and failures never become successful case evidence"
    )
    assert record["phases"] == {"call": outcome}, (
        "the consumer can detect missing phases"
    )
    assert record["duration_seconds"] == 0.25, "duration survives report normalization"
    assert record["output"][0] == "captured diagnostic", "diagnostics are not lost"
    if outcome != "passed":
        assert record["output"][-1] == "assertion diagnostic", (
            "failure detail is retained"
        )


def test_reporting_disabled_does_not_write_any_file(
    reporter, monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv(reporter.REPORT_ENV)
    reporter.pytest_configure(SimpleNamespace())
    emit(reporter)
    reporter.pytest_sessionfinish(
        SimpleNamespace(config=SimpleNamespace(), exitstatus=0)
    )
    assert list(tmp_path.iterdir()) == [], (
        "ordinary pytest without a report option has no report side effects"
    )


def test_bound_session_controls_survive_without_changing_public_environment_names(
    reporter, monkeypatch, tmp_path
) -> None:
    names = (
        reporter.REPORT_ENV,
        reporter.PARTITION_COUNT_ENV,
        reporter.PARTITION_INDEX_ENV,
    )
    bound_path = tmp_path / "bound.json"
    controls = dict(zip(names, (str(bound_path), "2", "1"), strict=True))
    with reporter.bound_session_controls(controls):
        for name in names:
            monkeypatch.delenv(name, raising=False)
        reporter.pytest_configure(SimpleNamespace())
        items = [
            SimpleNamespace(nodeid=f"tests/test_a.py::test_case[{index}]")
            for index in range(20)
        ]
        expected = {
            item.nodeid
            for item in items
            if reporter.partition_for_nodeid(item.nodeid, 2) == 1
        }
        reporter.pytest_collection_modifyitems(
            SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items: None)),
            items,
        )
        assert {item.nodeid for item in items} == expected
        reporter.pytest_collection_finish(SimpleNamespace(items=items))
        for item in items:
            emit(reporter, nodeid=item.nodeid)
        reporter.pytest_sessionfinish(
            SimpleNamespace(config=SimpleNamespace(), exitstatus=0)
        )
        assert names == (
            reporter.REPORT_ENV,
            reporter.PARTITION_COUNT_ENV,
            reporter.PARTITION_INDEX_ENV,
        ), "nested runners must retain the public environment-name contract"
    report = json.loads(bound_path.read_text())
    assert report["session"]["selection"]["partition"] == [2, 1]
    assert set(report["records"]) == expected


def test_failed_binding_scope_restores_the_callers_report_controls(
    reporter, tmp_path
) -> None:
    bound_path = tmp_path / "bound.json"
    controls = {
        reporter.REPORT_ENV: str(bound_path),
        reporter.PARTITION_COUNT_ENV: "1",
        reporter.PARTITION_INDEX_ENV: "0",
    }
    with pytest.raises(RuntimeError, match="fixture failure"):
        with reporter.bound_session_controls(controls):
            raise RuntimeError("fixture failure")
    reporter.pytest_configure(SimpleNamespace())
    reporter.pytest_sessionfinish(
        SimpleNamespace(config=SimpleNamespace(), exitstatus=0)
    )
    assert (tmp_path / "results.json").is_file(), (
        "the caller's report destination must be restored after a failed scope"
    )
    assert not bound_path.exists(), "a failed scope must not retain its report binding"


@pytest.mark.parametrize("defect", ["missing", "unknown", "non-string"])
def test_bound_session_controls_reject_incomplete_or_unknown_inputs(
    reporter, tmp_path, defect
) -> None:
    controls = {
        reporter.REPORT_ENV: str(tmp_path / "bound.json"),
        reporter.PARTITION_COUNT_ENV: "1",
        reporter.PARTITION_INDEX_ENV: "0",
    }
    if defect == "missing":
        del controls[reporter.PARTITION_INDEX_ENV]
    elif defect == "unknown":
        controls["UNRELATED"] = "value"
    else:
        controls[reporter.PARTITION_COUNT_ENV] = 1
    with pytest.raises(ValueError, match="pytest session controls"):
        with reporter.bound_session_controls(controls):
            pytest.fail("invalid reporter controls were accepted")


@pytest.mark.parametrize(
    ("count", "index"),
    [("bad", "0"), ("2", "bad"), ("0", "0"), ("2", "2"), ("", "0"), ("2", "")],
)
def test_invalid_partitions_are_rejected_before_collection(
    reporter, monkeypatch, count, index
) -> None:
    monkeypatch.setenv(reporter.PARTITION_COUNT_ENV, count)
    monkeypatch.setenv(reporter.PARTITION_INDEX_ENV, index)
    with pytest.raises(RuntimeError, match="partition"):
        reporter.pytest_configure(SimpleNamespace())


def test_partition_validation_requires_positive_count(reporter) -> None:
    with pytest.raises(ValueError, match="positive"):
        reporter.partition_for_nodeid("tests/a.py::test_a", 0)


def test_partitions_are_complete_disjoint_and_keep_stable_order(
    reporter, monkeypatch
) -> None:
    original = [
        SimpleNamespace(nodeid=f"tests/test_a.py::test_case[{index}]")
        for index in range(20)
    ]
    partitions = []
    for index in (0, 1):
        monkeypatch.setenv(reporter.PARTITION_COUNT_ENV, "2")
        monkeypatch.setenv(reporter.PARTITION_INDEX_ENV, str(index))
        deselected = []
        config = SimpleNamespace(
            hook=SimpleNamespace(
                pytest_deselected=lambda items: deselected.extend(items)
            )
        )
        items = list(original)
        reporter.pytest_collection_modifyitems(config, items)
        assert items and deselected, "both partitions contain real test items"
        assert {id(item) for item in items + deselected} == {
            id(item) for item in original
        }, "partitioning neither drops nor duplicates test items"
        assert items == [item for item in original if item in items], (
            "collection order is stable"
        )
        partitions.append({item.nodeid for item in items})
    assert not partitions[0] & partitions[1], (
        "shards cannot execute the same selected nodeid"
    )
    assert partitions[0] | partitions[1] == {item.nodeid for item in original}, (
        "all tests enter exactly one shard"
    )


def test_no_partition_and_single_partition_leave_collection_intact(
    reporter, monkeypatch
) -> None:
    items = [SimpleNamespace(nodeid="tests/test_a.py::test_one")]
    original = list(items)
    config = SimpleNamespace(
        hook=SimpleNamespace(
            pytest_deselected=lambda items: pytest.fail("unexpected deselection")
        )
    )
    reporter.pytest_collection_modifyitems(config, items)
    assert items == original, "disabled partitioning does not change collection"
    monkeypatch.setenv(reporter.PARTITION_COUNT_ENV, "1")
    monkeypatch.setenv(reporter.PARTITION_INDEX_ENV, "0")
    reporter.pytest_collection_modifyitems(config, items)
    assert items == original, "one shard retains every test"


def test_worker_reports_are_merged_and_owned_intermediates_removed(
    reporter, tmp_path
) -> None:
    nodeid = "tests/test_sample.py::test_case"
    emit(reporter, nodeid=nodeid, output="worker output")
    reporter.pytest_sessionfinish(
        SimpleNamespace(
            config=SimpleNamespace(workerinput={"workerid": "gw0"}), exitstatus=0
        )
    )
    path = tmp_path / "results.json"
    assert not path.exists(), "workers do not publish a final controller result"
    reporter.REPORTS.clear()
    reporter.pytest_xdist_node_collection_finished(SimpleNamespace(), [nodeid])
    reporter.pytest_sessionfinish(
        SimpleNamespace(config=SimpleNamespace(), exitstatus=0)
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["records"][nodeid]["output"] == "worker output", (
        "worker diagnostics survive merge"
    )
    assert result["session"]["collected_nodeids"] == [nodeid], (
        "the controller includes worker collection"
    )
    assert list(tmp_path.iterdir()) == [path], (
        "owned worker and normalization files are cleaned"
    )


def test_new_controller_clears_stale_worker_reports(reporter, tmp_path) -> None:
    stale = tmp_path / "results.json.worker-gw0.json"
    stale.write_text('{"old-test": {}}', encoding="utf-8")
    reporter.pytest_configure(SimpleNamespace())
    assert not stale.exists(), "a prior run's worker report cannot enter a new merge"


def test_worker_configuration_does_not_erase_sibling_results(
    reporter, tmp_path
) -> None:
    sibling = tmp_path / "results.json.worker-gw1.json"
    sibling.write_text("{}", encoding="utf-8")
    reporter.pytest_configure(SimpleNamespace(workerinput={"workerid": "gw0"}))
    assert sibling.exists(), "only the controller may clear prior worker files"


def test_malformed_worker_payload_stops_final_report(reporter, tmp_path) -> None:
    path = tmp_path / "results.json.worker-gw0.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="worker report is invalid"):
        reporter.pytest_sessionfinish(
            SimpleNamespace(config=SimpleNamespace(), exitstatus=0)
        )
    assert not (tmp_path / "results.json").exists(), (
        "invalid worker data cannot become successful evidence"
    )


def test_invalid_normalized_document_cannot_be_published(
    reporter, monkeypatch, tmp_path
) -> None:
    emit(reporter)
    original = Path.read_text

    def corrupt_normalized(path, **kwargs):
        if path.name.endswith(".normalized"):
            return "[]"
        return original(path, **kwargs)

    monkeypatch.setattr(Path, "read_text", corrupt_normalized)
    with pytest.raises(RuntimeError, match="normalized pytest report is invalid"):
        reporter.pytest_sessionfinish(
            SimpleNamespace(config=SimpleNamespace(), exitstatus=0)
        )
    assert not (tmp_path / "results.json").exists(), (
        "normalization failure cannot publish a report"
    )


def test_discovery_is_snapshotted_before_explicit_parameter_pruning(reporter) -> None:
    nodeids = [
        "tests/test_sample.py::test_case[a]",
        "tests/test_sample.py::test_case[b]",
    ]
    items = [Mock(spec=pytest.Item, nodeid=nodeid) for nodeid in nodeids]
    report = SimpleNamespace(
        nodeid="tests/test_sample.py", failed=False, result=list(items)
    )
    hook = reporter.pytest_make_collect_report(SimpleNamespace())
    next(hook)
    with pytest.raises(StopIteration) as completed:
        hook.send(report)
    assert completed.value.value is report
    report.result[:] = items[:1]
    reporter.pytest_collectreport(report)
    assert reporter.DISCOVERED == set(nodeids), (
        "late report pruning must not erase an unselected parameter"
    )


def test_worker_discovery_crosses_the_controller_boundary(reporter, tmp_path) -> None:
    nodeid = "tests/test_sample.py::test_case[a]"
    unselected = "tests/test_sample.py::test_case[b]"
    skipped = "tests/test_optional.py"
    reporter.pytest_collectreport(
        SimpleNamespace(
            nodeid="tests/test_sample.py",
            failed=False,
            result=[
                Mock(spec=pytest.Item, nodeid=value) for value in (nodeid, unselected)
            ],
        )
    )
    reporter.pytest_collectreport(
        SimpleNamespace(nodeid=skipped, failed=False, skipped=True, result=[])
    )
    output = {}
    reporter.pytest_sessionfinish(
        SimpleNamespace(
            config=SimpleNamespace(
                workerinput={"workerid": "gw0"}, workeroutput=output
            ),
            exitstatus=0,
        )
    )
    reporter.pytest_configure(SimpleNamespace())
    reporter.pytest_xdist_node_collection_finished(SimpleNamespace(), [nodeid])
    reporter.pytest_testnodedown(SimpleNamespace(workeroutput=output), None)
    reporter.pytest_sessionfinish(
        SimpleNamespace(config=SimpleNamespace(), exitstatus=0)
    )
    value = json.loads((tmp_path / "results.json").read_text())
    assert value["session"]["collected_nodeids"] == [nodeid]
    assert value["session"]["discovered_nodeids"] == [nodeid, unselected]
    assert value["session"]["collection_errors"] == []
    assert value["session"]["collection_skips"] == [skipped]


@pytest.mark.parametrize(
    "facts",
    [
        None,
        {},
        {"nodeids": [True], "errors": []},
        {"nodeids": ["invalid"], "errors": []},
        {"nodeids": [], "errors": "unavailable"},
        {"nodeids": [], "errors": [None]},
        {"nodeids": [], "errors": [], "skips": "unavailable"},
        {"nodeids": [], "errors": [], "skips": [None]},
    ],
)
def test_missing_or_malformed_worker_discovery_cannot_be_published_as_complete(
    reporter, tmp_path, facts
) -> None:
    reporter.pytest_testnodedown(
        SimpleNamespace(workeroutput={reporter.WORKER_DISCOVERY_KEY: facts}), None
    )
    reporter.pytest_sessionfinish(
        SimpleNamespace(config=SimpleNamespace(), exitstatus=0)
    )
    value = json.loads((tmp_path / "results.json").read_text())
    assert value["session"]["collection_errors"], (
        "missing worker discovery must remain an explicit receipt failure"
    )
