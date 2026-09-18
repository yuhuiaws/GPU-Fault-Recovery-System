from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import execution
from gpu_fault.admin.deadlines import DeploymentDeadlineExceeded
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import (
    acceptance_supervision,
    focused_pytest,
    regional_commands,
)
from scripts.e2e.regional import audit_warm_spare_guardrails as warm
from tests.regional.test_warm_spare_guardrails import pytest_receipt
from tools import pytest_result_identity as evidence
from tools.pytest_case_reporter import REPORT_SCHEMA_VERSION
from tools.pytest_result_identity import PytestReceipt

ROOT = Path(__file__).resolve().parents[2]
GOOD = "test_owned.py::test_good"
BAD = "test_owned.py::test_bad"
PHASES = {"setup": "passed", "call": "passed", "teardown": "passed"}
BOOTSTRAP = """
import hashlib
import sys
from pathlib import Path
import pytest
from tools import pytest_case_reporter
def fixture_identity(root):
    root = Path(root)
    return hashlib.sha256((root / "input.txt").read_bytes()).hexdigest()
pytest_case_reporter.source_identity = fixture_identity
pytest_case_reporter.repository_root = lambda root: root.resolve()
raise SystemExit(pytest.main(sys.argv[1:]))
"""


@pytest.fixture
def child(tmp_path, monkeypatch):
    suite = tmp_path / "test_owned.py"
    suite.write_text("def test_good():\n    assert True, 'control'\n", encoding="utf-8")
    (tmp_path / "input.txt").write_text("baseline", encoding="utf-8")
    state = SimpleNamespace(
        root=tmp_path, suite=suite, calls=[], extra=[], receipt=None, completed=None
    )
    supervised_command = execution.run_command

    def identity(root):
        root = Path(root)
        return hashlib.sha256((root / "input.txt").read_bytes()).hexdigest()

    def command(argv, **options):
        state.calls.append((list(argv), copy.deepcopy(options)))
        completed = supervised_command(
            [
                sys.executable,
                "-c",
                BOOTSTRAP,
                *argv[3:],
                "--noconftest",
                "-p",
                "no:cacheprovider",
                *state.extra,
            ],
            **options,
        )
        path = Path(options["environment"]["PYTEST_GPU_FAULT_CASE_REPORT"])
        state.receipt = json.loads(path.read_text()) if path.exists() else None
        state.completed = completed
        return completed

    monkeypatch.setattr(warm, "ROOT", tmp_path)
    monkeypatch.setattr(evidence, "source_identity", identity)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(ROOT / "src"), str(ROOT))))
    monkeypatch.setattr(execution, "run_command", command)
    return state


def test_real_supervised_child_accepts_complete_evidence_not_failure_prose(child):
    child.suite.write_text(
        "def test_good():\n"
        "    print('FAILED unrelated.py::test_unrelated - misleading prose')\n"
        "    assert True, 'complete control'\n",
        encoding="utf-8",
    )
    child.extra = ["-s"]

    result = warm.run_focused_pytest(child.root, {"good": [GOOD]})

    assert result == {"good": True}, (
        "complete receipt evidence must override misleading prose"
    )
    assert len(child.calls) == 1 and child.completed.returncode == 0, (
        "the audit must run one actual supervised pytest child"
    )
    assert child.receipt["records"][GOOD]["phases"] == PHASES, (
        "the real child must complete setup, call and teardown"
    )
    _, options = child.calls[0]
    assert options["timeout_seconds"] == 300 and options["cwd"] == child.root, (
        "the child must retain the explicit supervised lifetime and working directory"
    )
    log = child.root / "cases/pytest.log"
    assert (
        "FAILED unrelated" in log.read_text() and log.stat().st_mode & 0o777 == 0o600
    ), "prose is retained only as private diagnostics"


@pytest.mark.parametrize(
    "mode", ["skip", "xfail", "call-failure", "teardown-failure", "partial-teardown"]
)
def test_real_child_keeps_complete_neighbor_but_refuses_unpassed_case(child, mode):
    prefix = "import pytest\n"
    good = "def test_good():\n    assert True, 'independent control'\n"
    if mode == "skip":
        bad = "def test_bad():\n    pytest.skip('fixture unavailable')\n"
    elif mode == "xfail":
        bad = (
            "@pytest.mark.xfail(reason='known failure')\n"
            "def test_bad():\n    assert False, 'expected failure is not passing evidence'\n"
        )
    elif mode == "call-failure":
        bad = "def test_bad():\n    assert False, 'actual failed assertion'\n"
    else:
        action = (
            "pytest.exit('owned partial teardown', returncode=0)"
            if mode == "partial-teardown"
            else "raise RuntimeError('owned teardown failure')"
        )
        prefix += f"@pytest.fixture\ndef incomplete():\n    yield\n    {action}\n"
        bad = "def test_bad(incomplete):\n    assert True, 'call alone cannot pass'\n"
    child.suite.write_text(prefix + good + bad, encoding="utf-8")

    result = warm.run_focused_pytest(child.root, {"good": [GOOD], "bad": [BAD]})

    assert result == {"good": True, "bad": False}, (
        "an independently complete passing case survives a known neighboring failure",
        mode,
        child.completed.stdout,
        child.completed.stderr,
    )
    assert child.receipt["records"][GOOD]["phases"] == PHASES, (
        "the passing case must have all three actual child phases"
    )
    bad_record = child.receipt["records"][BAD]
    assert bad_record["status"] != "PASS" or bad_record["phases"] != PHASES, (
        "the rejected case must have observed missing, skipped or failed work"
    )


def test_real_child_filtered_generic_variants_cannot_pass_the_case(child):
    child.suite.write_text(
        "import pytest\n"
        "def test_good():\n    assert True, 'control'\n"
        "@pytest.mark.parametrize('value', [0, 1], ids=['zero', 'one'])\n"
        "def test_matrix(value):\n    assert value in {0, 1}, 'matrix member'\n",
        encoding="utf-8",
    )
    child.extra = ["-k", "not one"]

    result = warm.run_focused_pytest(
        child.root, {"good": [GOOD], "matrix": ["test_owned.py::test_matrix"]}
    )

    assert result == {"good": True, "matrix": False}, (
        "one passed parameter cannot satisfy a generic case's discovered matrix"
    )
    assert child.completed.returncode == 0, (
        "filtered success must not hide the evidence gap"
    )
    assert len(child.receipt["session"]["discovered_nodeids"]) == 3, (
        "the real reporter must retain discovery before filtering"
    )
    assert len(child.receipt["records"]) == 2, (
        "the omitted variant must remain unexecuted"
    )


def test_real_child_zero_exit_without_final_receipt_fails_closed(child):
    child.suite.write_text(
        "import os\n"
        "def test_bad():\n"
        "    print('PASSED all cases', flush=True)\n"
        "    os._exit(0)\n",
        encoding="utf-8",
    )

    result = warm.run_focused_pytest(child.root, {"bad": [BAD]})

    assert child.completed.returncode == 0 and child.receipt is None, (
        "the actual abruptly exiting child must leave no final receipt"
    )
    assert result == {"bad": False}, "an exit-zero child without evidence cannot pass"


def test_real_child_source_input_drift_invalidates_its_receipt(child):
    child.suite.write_text(
        "from pathlib import Path\n"
        "def test_bad():\n    Path('input.txt').write_text('changed source input')\n",
        encoding="utf-8",
    )

    result = warm.run_focused_pytest(child.root, {"bad": [BAD]})

    assert child.completed.returncode == 0 and child.receipt is not None, (
        "the source-changing child must otherwise finish normally"
    )
    assert (
        child.receipt["source_identity"] != child.receipt["session"]["source_identity"]
    ), "the owned source input must actually change between start and finish"
    assert result == {"bad": False}, "a changed source cannot produce passing evidence"


def test_real_child_receives_no_live_credentials_filters_or_foreign_report(
    child, monkeypatch
):
    injected = {
        "GPU_FAULT_STORE_URL": "postgresql://production.invalid/business",
        "GPU_FAULT_TEST_POSTGRES_URL": "postgresql://foreign.invalid/business",
        "GPU_FAULT_EXECUTION_TOKEN": "unit-placeholder",
        "AWS_ACCESS_KEY_ID": "unit-placeholder",
        "AWS_PROFILE": "production",
        "PGHOST": "production.invalid",
        "PGPASSFILE": "/private/foreign-pgpass",
        "KUBECONFIG": "/private/foreign-kubeconfig",
        "PYTEST_ADDOPTS": "-k absent",
        "PYTEST_GPU_FAULT_PARTITION_COUNT": "16",
        "PYTEST_GPU_FAULT_PARTITION_INDEX": "15",
        "PYTEST_XDIST_WORKER": "gw15",
        "PYTEST_GPU_FAULT_CASE_REPORT": "/private/foreign-receipt.json",
    }
    for name, value in injected.items():
        monkeypatch.setenv(name, value)
    child.suite.write_text(
        "import os\n"
        "def test_good():\n"
        "    assert os.environ['GPU_FAULT_STORE_URL'] == '', 'no production Store'\n"
        "    assert os.environ['GPU_FAULT_TEST_POSTGRES_URL'] == '', 'no PostgreSQL'\n"
        "    assert os.environ['KUBECONFIG'] == '/dev/null', 'no cluster access'\n"
        "    assert 'GPU_FAULT_EXECUTION_TOKEN' not in os.environ, 'no execution token'\n"
        "    assert 'AWS_ACCESS_KEY_ID' not in os.environ, 'no AWS credentials'\n"
        "    assert 'PGHOST' not in os.environ, 'no PG host'\n"
        "    assert 'PYTEST_ADDOPTS' not in os.environ, 'no inherited selection'\n",
        encoding="utf-8",
    )

    assert warm.run_focused_pytest(child.root, {"good": [GOOD]}) == {"good": True}, (
        "the actual isolated child must run despite an inherited excluding filter"
    )
    environment = child.calls[0][1]["environment"]
    retained = {
        "GPU_FAULT_STORE_URL",
        "GPU_FAULT_TEST_POSTGRES_URL",
        "KUBECONFIG",
        "PYTEST_GPU_FAULT_CASE_REPORT",
    }
    assert not (set(injected) - retained) & set(environment), (
        "all authority, worker and shard inputs must be stripped"
    )
    assert (
        environment["PYTEST_GPU_FAULT_CASE_REPORT"]
        != injected["PYTEST_GPU_FAULT_CASE_REPORT"]
    ), "the supervised batch must use a new private report"


def receipt_document(root, *, exitstatus=0):
    identity = evidence.source_identity(root)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_identity": identity,
        "session": {
            "source_identity": identity,
            "exitstatus": exitstatus,
            "collected_nodeids": [GOOD, BAD],
            "discovered_nodeids": [GOOD, BAD],
            "collection_errors": [],
            "collection_skips": [],
        },
        "records": {
            nodeid: {"status": "PASS", "phases": dict(PHASES)} for nodeid in (GOOD, BAD)
        },
    }


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "json",
        "identity",
        "session",
        "collection",
        "collection-error",
        "collector-skip",
        "unexplained-exit",
        "abnormal-exit",
    ],
)
def test_untrusted_receipt_failures_cannot_be_replaced_by_passing_stdout(
    child, monkeypatch, defect
):
    code = 1 if defect == "unexplained-exit" else 2 if defect == "abnormal-exit" else 0
    document = receipt_document(child.root, exitstatus=code)
    if defect == "identity":
        document["source_identity"] = "0" * 64
    elif defect == "session":
        del document["session"]
    elif defect == "collection":
        del document["records"][BAD]
    elif defect == "collection-error":
        document["session"]["collection_errors"] = ["owned collection failed"]
    elif defect == "collector-skip":
        document["session"]["collection_skips"] = ["test_optional.py"]
    reports = []

    def execute(command, **options):
        path = Path(options["environment"]["PYTEST_GPU_FAULT_CASE_REPORT"])
        reports.append(path)
        if defect != "missing":
            path.write_text("not-json" if defect == "json" else json.dumps(document))
        return subprocess.CompletedProcess(command, code, "all PASSED\n", "")

    monkeypatch.setattr(execution, "run_command", execute)
    assert warm.run_focused_pytest(child.root, {"good": [GOOD], "bad": [BAD]}) == {
        "good": False,
        "bad": False,
    }, ("globally invalid session evidence must fail every dependent case", defect)
    assert len(reports) == 1 and not reports[0].parent.exists(), (
        "the owned temporary receipt directory must be removed on refusal"
    )


@pytest.mark.parametrize(
    "failure", ["spawn", "timeout", "deadline", "supervision", "marker"]
)
def test_supervised_transport_failures_raise_and_record_lost_custody(
    child, monkeypatch, failure
):
    reports = []
    markers = []
    error = {
        "spawn": OSError(5, "unit-private-startup-detail"),
        "timeout": subprocess.TimeoutExpired(["python"], 300),
        "deadline": DeploymentDeadlineExceeded("owned deadline"),
        "supervision": ProcessSupervisionLost("owned supervision loss"),
        "marker": ProcessSupervisionLost("owned supervision loss"),
    }[failure]

    def execute(command, **options):
        reports.append(Path(options["environment"]["PYTEST_GPU_FAULT_CASE_REPORT"]))
        raise error

    monkeypatch.setattr(execution, "run_command", execute)

    def record_marker():
        markers.append("lost")
        if failure == "marker":
            raise OSError("unit-private-marker-detail")

    monkeypatch.setattr(
        acceptance_supervision, "record_supervision_loss", record_marker
    )
    expected = (
        ProcessSupervisionLost
        if failure in {"supervision", "marker"}
        else regional_commands.RegionalCommandTimeout
        if failure in {"timeout", "deadline"}
        else regional_commands.RegionalFixtureError
    )
    with pytest.raises(expected) as raised:
        warm.run_focused_pytest(child.root, {"good": [GOOD]})
    assert "unit-private" not in str(raised.value), (
        "supervisor failures must retain safe diagnostics only"
    )
    assert len(reports) == 1 and not reports[0].parent.exists(), (
        "transport failure must release the private receipt directory"
    )
    assert markers == (["lost"] if failure in {"supervision", "marker"} else []), (
        "lost custody must invoke the existing durable marker, unlike ordinary failures"
    )


def test_unrecognized_or_empty_command_never_starts_a_child(child, monkeypatch):
    @contextmanager
    def unrecognized(*args, **kwargs):
        yield None

    monkeypatch.setattr(focused_pytest, "prepare_focused_pytest", unrecognized)
    with pytest.raises(RuntimeError, match="not recognized"):
        warm.run_pytest(child.root, [GOOD])
    with pytest.raises(ValueError, match="explicit selector"):
        warm.run_pytest(child.root, [])
    assert child.calls == [], "invalid preparation must not launch a process"


def test_case_selection_requires_phase_completeness_and_deduplicates_shared_tests(
    child, monkeypatch
):
    calls = []
    report = pytest_receipt([GOOD, BAD])
    bad = evidence.normalized_pytest_nodeid(BAD, root=child.root)
    report.records[bad] = {
        "status": "PASS",
        "phases": {"setup": "passed", "call": "passed"},
    }

    def execute(_root, nodeids):
        calls.append(nodeids)
        return report

    monkeypatch.setattr(warm, "run_pytest", execute)
    result = warm.run_focused_pytest(
        child.root, {"first": [GOOD], "second": [GOOD, BAD], "empty": []}
    )
    assert result == {"first": True, "second": False, "empty": False}, (
        "every required selector needs a complete three-phase observation"
    )
    assert calls == [[GOOD, BAD]], (
        "overlapping cases must not execute duplicate selectors"
    )
    assert warm.run_focused_pytest(child.root, {}) == {}, (
        "no cases means no pytest invocation"
    )
    assert warm.run_focused_pytest(child.root, {"empty": []}) == {"empty": False}, (
        "an empty selector set must not accidentally run the repository-wide suite"
    )
    assert len(calls) == 1, "empty case definitions must not start another process"


@pytest.mark.parametrize(
    "invalid", [PytestReceipt({}), PytestReceipt({}, frozenset(), ("skip",))]
)
def test_sessionless_or_skipped_collector_model_cannot_satisfy_a_case(
    child, monkeypatch, invalid
):
    monkeypatch.setattr(warm, "run_pytest", lambda *_args: invalid)
    assert warm.run_focused_pytest(child.root, {"good": [GOOD]}) == {"good": False}, (
        "legacy sessionless models cannot bypass the warm batch's receipt contract"
    )
