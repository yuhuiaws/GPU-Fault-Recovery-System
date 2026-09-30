from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.test_cov95_fault_runner import case
from tools import run_fault_test_cases as runner
from tools.pytest_case_reporter import partition_for_nodeid
from tools.pytest_result_identity import load_pytest_receipt

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = "a" * 64
PHASES = {"setup": "passed", "call": "passed", "teardown": "passed"}
CHILD_BOOTSTRAP = """
import sys
import pytest
from tools import pytest_case_reporter
pytest_case_reporter.source_identity = lambda root: "a" * 64
pytest_case_reporter.repository_root = lambda root: root.resolve()
raise SystemExit(pytest.main(sys.argv[1:]))
"""


@pytest.fixture
def tiny_suite(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setattr(runner, "source_identity", lambda root: IDENTITY)
    (ROOT / ".codex").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="evidence-child-", dir=ROOT / ".codex"
    ) as directory:
        path = Path(directory) / "test_selection.py"
        path.write_text(
            "import pytest\n\n"
            "@pytest.mark.parametrize('value', [0, 1, 2, 3], ids=['shape-0', 'shape-1', 'shape-2', 'shape-3'])\n"
            "def test_matrix(value):\n"
            "    assert 0 <= value < 4\n\n"
            "def test_control():\n"
            "    assert 2 + 2 == 4\n",
            encoding="ascii",
        )
        yield path


def matrix_nodeids(path: Path) -> list[str]:
    # Hyphenated ids: the child suite lives under a random `evidence-child-<8 chars>` directory whose
    # name is drawn from [a-z0-9_]; a plain id like `v0` can occur in that name and then `-k v0`
    # matches every variant through the directory keyword (deploy gate 2026-09-24, dir ...zuhk4tv0).
    selector = path.relative_to(ROOT).as_posix() + "::test_matrix"
    return [f"{selector}[shape-{index}]" for index in range(4)]


def run_child(
    path: Path,
    *,
    options: tuple[str, ...] = (),
    target: str | None = None,
    partition: tuple[int, int] | None = None,
    xdist: bool = False,
    expected_exitstatus: int = 0,
) -> tuple[Path, dict]:
    report = path.parent / "receipt.json"
    environment = {
        "HOME": "/tmp",
        "PATH": os.environ["PATH"],
        "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_GPU_FAULT_CASE_REPORT": str(report),
        "AWS_CONFIG_FILE": os.devnull,
        "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
        "AWS_EC2_METADATA_DISABLED": "true",
        "KUBECONFIG": os.devnull,
    }
    if partition is not None:
        count, index = partition
        environment["PYTEST_GPU_FAULT_PARTITION_COUNT"] = str(count)
        environment["PYTEST_GPU_FAULT_PARTITION_INDEX"] = str(index)
    arguments = [
        sys.executable,
        "-c",
        CHILD_BOOTSTRAP,
        "-q",
        "--noconftest",
        "-p",
        "no:cacheprovider",
        "-p",
        "tools.pytest_case_reporter",
        "-o",
        "addopts=",
        "--rootdir",
        str(ROOT),
    ]
    if xdist:
        arguments.extend(["-p", "xdist.plugin", "-n", "2"])
    arguments.extend([target or str(path), *options])
    result = subprocess.run(
        arguments,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == expected_exitstatus, result.stdout + result.stderr
    return report, json.loads(report.read_text(encoding="utf-8"))


@pytest.mark.parametrize("selection", ["keyword", "explicit-parameter", "deselect"])
def test_filtered_parameterized_receipt_cannot_pass_generic_case(
    tiny_suite: Path, selection: str
) -> None:
    variants = matrix_nodeids(tiny_suite)
    selector = variants[0].partition("[")[0]
    options = (
        ("-k", "shape-0")
        if selection == "keyword"
        else tuple(
            option for nodeid in variants[1:] for option in ("--deselect", nodeid)
        )
        if selection == "deselect"
        else ()
    )
    target = variants[0] if selection == "explicit-parameter" else selector
    report, value = run_child(tiny_suite, options=options, target=target)
    assert value["session"]["collected_nodeids"] == [variants[0]]
    generic = case(pytest_nodeid=selector)
    result = runner.load_pytest_results([generic], path=report)
    assert result[generic["id"]]["status"] == "FAIL", (
        "one selected parameter cannot prove the generic four-variant case"
    )
    exact = case(pytest_nodeid=variants[0])
    assert (
        runner.load_pytest_results([exact], path=report)[exact["id"]]["status"]
        == "PASS"
    )
    assert set(variants) <= set(value["session"]["discovered_nodeids"])


def test_xdist_partition_keeps_the_unfiltered_discovery_inventory(
    tiny_suite: Path,
) -> None:
    variants = matrix_nodeids(tiny_suite)
    selector = variants[0].partition("[")[0]
    partition = next(
        (count, index)
        for count in range(2, 32)
        for index in range(count)
        if 0
        < sum(partition_for_nodeid(nodeid, count) == index for nodeid in variants)
        < len(variants)
    )
    report, value = run_child(
        tiny_suite, target=selector, partition=partition, xdist=True
    )
    assert 0 < len(value["session"]["collected_nodeids"]) < len(variants)
    generic = case(pytest_nodeid=selector)
    assert (
        runner.load_pytest_results([generic], path=report)[generic["id"]]["status"]
        == "FAIL"
    ), "worker discovery must survive the controller merge before generic reuse"
    assert set(variants) <= set(value["session"]["discovered_nodeids"])


@pytest.mark.parametrize("xdist", [False, True])
def test_complete_modern_and_legacy_receipts_still_pass(
    tiny_suite: Path, xdist: bool
) -> None:
    report, value = run_child(tiny_suite, xdist=xdist)
    selector = matrix_nodeids(tiny_suite)[0].partition("[")[0]
    selected = case(pytest_nodeid=selector)
    assert (
        runner.load_pytest_results([selected], path=report)[selected["id"]]["status"]
        == "PASS"
    )
    value.pop("session")
    report.write_text(json.dumps(value), encoding="utf-8")
    assert (
        runner.load_pytest_results([selected], path=report)[selected["id"]]["status"]
        == "PASS"
    )


@pytest.mark.parametrize(
    "discovered", [None, [], ["invalid"], [True], ["tests/test_unit.py::test_case"] * 2]
)
def test_modern_receipt_requires_valid_discovery_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, discovered: object
) -> None:
    monkeypatch.setattr(runner, "source_identity", lambda root: IDENTITY)
    selected = case()
    nodeid = selected["pytest_nodeid"]
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": 1,
        "source_identity": IDENTITY,
        "session": {
            "source_identity": IDENTITY,
            "started_at": now,
            "finished_at": now,
            "exitstatus": 0,
            "collected_nodeids": [nodeid],
            "discovered_nodeids": discovered,
        },
        "records": {nodeid: {"status": "PASS", "phases": PHASES}},
    }
    report = tmp_path / "receipt.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="discovery"):
        runner.load_pytest_results([selected], path=report)


def test_collection_error_is_not_a_complete_discovery_snapshot(
    tiny_suite: Path,
) -> None:
    tiny_suite.with_name("test_broken.py").write_text(
        "def test_broken(:\n    pass\n", encoding="ascii"
    )
    report, value = run_child(
        tiny_suite,
        target=str(tiny_suite.parent),
        options=("--continue-on-collection-errors",),
        expected_exitstatus=1,
    )
    assert value["session"]["collection_errors"], (
        "a collection error cannot disappear behind successful runnable tests"
    )
    with pytest.raises(ValueError, match="collection errors"):
        load_pytest_receipt(
            report,
            root=ROOT,
            expected_identity=IDENTITY,
            require_session=True,
            expected_exitstatus=1,
        )


@pytest.mark.parametrize("defect", ["unseen", "alias"])
def test_receipt_cannot_add_undiscovered_or_alias_colliding_results(
    tiny_suite: Path, defect: str
) -> None:
    report, value = run_child(tiny_suite)
    first = value["session"]["collected_nodeids"][0]
    if defect == "unseen":
        value["session"]["discovered_nodeids"].remove(first)
    else:
        alias = "./" + first
        value["records"][alias] = value["records"][first]
        value["session"]["collected_nodeids"].append(alias)
        value["session"]["discovered_nodeids"].append(alias)
    report.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="discovery|alias"):
        load_pytest_receipt(report, root=ROOT, expected_identity=IDENTITY)


@pytest.mark.parametrize("skips", ["unavailable", [None]])
def test_receipt_rejects_malformed_skipped_collector_inventory(
    tiny_suite: Path, skips: object
) -> None:
    report, value = run_child(tiny_suite)
    value["session"]["collection_skips"] = skips
    report.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="skipped collector inventory"):
        load_pytest_receipt(report, root=ROOT, expected_identity=IDENTITY)


@pytest.mark.parametrize("xdist", [False, True], ids=["serial", "xdist"])
def test_command_wrapper_cannot_pass_a_skipped_collector(
    tiny_suite: Path, monkeypatch: pytest.MonkeyPatch, xdist: bool
) -> None:
    skipped = tiny_suite.with_name("test_skipped.py")
    skipped.write_text(
        "import pytest\n"
        "pytest.skip('synthetic optional dependency', allow_module_level=True)\n"
        "def test_unavailable():\n"
        "    assert 2 + 2 == 4\n",
        encoding="ascii",
    )
    _report, value = run_child(tiny_suite, target=str(tiny_suite.parent), xdist=xdist)
    assert value["session"]["collection_skips"] == [
        skipped.relative_to(ROOT).as_posix()
    ]

    def child(arguments, **options):
        Path(options["env"][runner.PYTEST_BATCH_REPORT_ENV]).write_text(
            json.dumps(value)
        )
        return subprocess.CompletedProcess(arguments, 0, stdout="5 passed, 1 skipped")

    monkeypatch.setattr(runner.subprocess, "run", child)
    wrapped = case(
        automation="command",
        command=[sys.executable, "-m", "pytest", str(tiny_suite.parent)],
    )
    result = runner.run_case(wrapped, environment={})
    assert result["status"] == "FAIL", (
        "collector-level skips have no runtest record but still leave wrapper assertions unexecuted"
    )
    assert "skipped collector" in result["pytest_error"]
    passed_case = case(pytest_nodeid=matrix_nodeids(tiny_suite)[0].partition("[")[0])
    skipped_case = case(
        id="GF-SKIPPED",
        pytest_nodeid=skipped.relative_to(ROOT).as_posix() + "::test_unavailable",
    )
    results = runner.run_pytest_batch([passed_case, skipped_case], environment={})
    assert results[passed_case["id"]]["status"] == "PASS"
    assert results[skipped_case["id"]]["status"] == "FAIL"
