from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault.policy import load_xid_policy
from tools import pytest_case_reporter
from tools import run_fault_test_cases as runner
from tools.pytest_result_identity import source_identity
from tools.run_fault_test_cases import (
    TRACE_PREFIX,
    _extract_processing_trace,
    build_isolated_environment,
    execution_policy,
    load_catalog,
    run_case,
    run_pytest_batch,
    select_cases,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "testcases" / "fault-scenarios.yaml"
REGIONAL_DOCUMENT = ROOT / "docs" / "区域模式端到端验收测试用例.md"
FAULT_MANUAL = ROOT / "docs" / "故障模拟测试手册.md"
REGIONAL_CASE_HEADING = re.compile(
    r"^### (GF-REGIONAL-[A-Z0-9-]+)(?:[：:].*)?$", re.MULTILINE
)
REGIONAL_RISK_LABEL = re.compile(r"^\s*[-*] 等级[：:]\s*`([a-z-]+)`", re.MULTILINE)
REGIONAL_RISK_TABLE_HEADING = "### 2.2 风险等级词表（唯一取值来源）"
REGIONAL_RISK_TABLE_ROW = re.compile(r"^\| `([a-z-]+)` \|", re.MULTILINE)
# 手册里"这条用例真跑过"只有这几种写法：小节标题里的执行记录/执行结论，
# 或者一条 ``- <日期> 真机结果：`` 的行内结论。层级从 H2 起算：记录搬进
# 执行史时整段提了一级，写死 #{4,6} 会把 `### 真实环境执行结论` 漏掉。
REGIONAL_EXECUTION_RECORD = re.compile(
    r"^(?:#{2,6}\s|[-*] |\*\*)"
    r".*?(?:执行记录|执行结论|真机结果|真机执行记录|真机验证)",
    re.MULTILINE,
)


def _heading_slug(value: str) -> str:
    result = []
    for character in value.strip().lower():
        if character.isspace():
            result.append("-")
        elif character == "-":
            result.append(character)
        elif unicodedata.category(character).startswith(("P", "S")):
            continue
        else:
            result.append(character)
    return "".join(result)


def _heading_anchors(path: Path) -> set[str]:
    anchors = set()
    occurrences: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*$", line)
        if match is None:
            continue
        base = _heading_slug(match.group(1))
        occurrence = occurrences.get(base, 0)
        occurrences[base] = occurrence + 1
        anchors.add(base if occurrence == 0 else f"{base}-{occurrence}")
    return anchors


def test_fault_scenario_catalog_is_complete_and_unique() -> None:
    # 精确条数而不是下界：``>= 500`` 只增不减，删掉一条用例它照样绿。
    # 显式用例数来自 YAML，两个生成家族直接跟固定 Catalog rule ID 集合。
    raw = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    xid_values = {rule.xid for rule in load_xid_policy().catalog_rules}
    cases = load_catalog(CATALOG)
    ids = {case["id"] for case in cases}

    assert len(cases) == len(raw["test_cases"]) + len(xid_values) * 2
    assert len(ids) == len(cases)
    assert {f"GF-XID-KMSG-{value:03d}" for value in xid_values} <= ids
    assert {f"GF-XID-KMSG-B200-{value:03d}" for value in xid_values} <= ids
    assert {family["generator"] for family in raw["generated_test_families"]} == {
        "xid_catalog_rules"
    }
    assert all(
        "start" not in family and "end" not in family
        for family in raw["generated_test_families"]
    )
    assert {
        "xid-policy",
        "xid-collector",
        "xid-kmsg-replay",
        "xid-kmsg-replay-b200",
        "gpu-product-discovery",
        "sxid-policy",
        "dcgm-diagnostics",
        "gpu-metrics",
        "host-health",
        "training-recovery",
        "restart-guard",
        "notification",
        "closed-loop",
        "hyperpod-managed",
        "spare-failover",
        "live-cluster",
        "destructive-acceptance",
        "attempt-generation-fence",
        "regional-startup-guard",
        "regional-authentication",
        "regional-high-availability",
        "regional-notification",
        "regional-capacity",
        "regional-preemption",
        "regional-collector",
        "regional-workload",
    }.issubset({case["category"] for case in cases})


def test_every_automated_case_references_a_real_pytest_function() -> None:
    cases = load_catalog(CATALOG)
    parsed: dict[Path, set[str]] = {}
    nodeids = []

    for case in cases:
        if case["automation"] != "pytest":
            continue
        path_text, function_name = case["pytest_nodeid"].split("::", 1)
        function_name = function_name.split("[", 1)[0]
        path = ROOT / path_text
        assert path.is_file(), case["id"]
        if path not in parsed:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            names = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            for node in tree.body:
                if not isinstance(node, ast.ImportFrom):
                    continue
                names.update(
                    alias.asname or alias.name
                    for alias in node.names
                    if (alias.asname or alias.name).startswith("test_")
                )
            parsed[path] = names
        assert function_name in parsed[path], case["id"]
        nodeids.append(case["pytest_nodeid"])

    assert len(nodeids) == len(set(nodeids))


def test_every_pytest_reference_is_collectable() -> None:
    cases = load_catalog(CATALOG)
    nodeids = [
        case["pytest_nodeid"] for case in cases if case["automation"] == "pytest"
    ]
    nodeids.extend(case["related_pytest"] for case in cases if "related_pytest" in case)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            *nodeids,
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout


def test_fault_case_runner_direct_script_entrypoint_resolves_scheduler() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/run_fault_test_cases.py"),
            "--case",
            "GF-POL-001",
            "--list",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    assert result.stdout.startswith("GF-POL-001\tcomponent\tnon-destructive\t"), (
        result.stdout
    )


def test_isolated_environment_does_not_inherit_credentials() -> None:
    environment = build_isolated_environment(
        {
            "HOME": "/home/tester",
            "PATH": "/bin",
            "PYTHONPATH": "src",
            "GPU_FAULT_EXECUTION_TOKEN": "sensitive",
            "GPU_FAULT_CLUSTER_TOKEN": "sensitive",
            "GPU_FAULT_STORE_URL": "postgresql://sensitive",
            "AWS_SECRET_ACCESS_KEY": "sensitive",
            "KUBECONFIG": "/sensitive/kubeconfig",
        }
    )

    assert environment == {
        "HOME": "/home/tester",
        "PATH": "/bin",
        "PYTHONPATH": "src",
        "GPU_FAULT_STORE_URL": "",
        "GPU_FAULT_TEST_POSTGRES_URL": "",
    }


def test_manual_cases_reference_real_document_sections() -> None:
    cases = load_catalog(CATALOG)
    parsed: dict[Path, set[str]] = {}

    for case in cases:
        if case["automation"] != "manual":
            continue
        procedure = case["procedure"]
        assert isinstance(procedure, str), case["id"]
        path_text, separator, anchor = procedure.partition("#")
        assert separator and anchor, case["id"]
        path = ROOT / path_text
        assert path.suffix == ".md", case["id"]
        assert path.is_file(), case["id"]
        if path not in parsed:
            parsed[path] = _heading_anchors(path)
        assert anchor in parsed[path], case["id"]


def test_default_selection_excludes_destructive_manual_cases() -> None:
    cases = load_catalog(CATALOG)
    selected = select_cases(
        cases,
        case_ids=set(),
        categories=set(),
        levels=set(),
        include_manual=False,
        include_live=False,
    )

    assert selected
    assert all(case["risk"] == "non-destructive" for case in selected)
    assert all(case["automation"] == "pytest" for case in selected)


def test_live_case_requires_explicit_opt_in() -> None:
    cases = load_catalog(CATALOG)
    selected = select_cases(
        cases,
        case_ids={"GF-LIVE-000"},
        categories=set(),
        levels=set(),
        include_manual=False,
        include_live=True,
    )

    assert [case["id"] for case in selected] == ["GF-LIVE-000"]
    assert selected[0]["risk"] == "live-non-destructive"


def test_include_live_does_not_execute_an_unselected_command() -> None:
    cases = load_catalog(CATALOG)
    selected = select_cases(
        cases,
        case_ids=set(),
        categories=set(),
        levels=set(),
        include_manual=False,
        include_live=True,
    )

    assert all(case["automation"] != "command" for case in selected)


def test_also_case_adds_explicit_command_to_filtered_pytest_selection() -> None:
    cases = load_catalog(CATALOG)
    selected = select_cases(
        cases,
        case_ids=[],
        categories=set(),
        levels={"unit", "component"},
        include_manual=False,
        include_live=True,
        also_case_ids=["GF-REGIONAL-CAP-005"],
    )

    assert selected[-1]["id"] == "GF-REGIONAL-CAP-005"
    assert any(case["automation"] == "pytest" for case in selected[:-1]), selected
    assert all(case["automation"] == "pytest" for case in selected[:-1]), selected


def test_also_case_requires_live_opt_in() -> None:
    cases = load_catalog(CATALOG)

    with pytest.raises(ValueError, match="requires --include-live"):
        select_cases(
            cases,
            case_ids=[],
            categories=set(),
            levels={"unit"},
            include_manual=False,
            include_live=False,
            also_case_ids=["GF-REGIONAL-CAP-005"],
        )


def test_explicit_case_selection_preserves_requested_order() -> None:
    cases = load_catalog(CATALOG)
    requested = ["GF-REGIONAL-NET-005", "GF-REGIONAL-BOOT-022"]

    selected = select_cases(
        cases,
        case_ids=requested,
        categories=set(),
        levels=set(),
        include_manual=False,
        include_live=True,
    )

    assert [case["id"] for case in selected] == requested


def test_net004_uses_the_read_only_dependency_audit() -> None:
    case = next(
        case for case in load_catalog(CATALOG) if case["id"] == "GF-REGIONAL-NET-004"
    )

    assert case["automation"] == "command"
    assert case["command"] == [
        "python3",
        "scripts/e2e/regional/audit_net004_dependency_boundary.py",
    ]
    assert case["risk"] == "read-only-signal-replay"


def test_ha008_uses_the_isolated_fatal_exit_acceptance() -> None:
    case = next(
        case for case in load_catalog(CATALOG) if case["id"] == "GF-REGIONAL-HA-008"
    )

    assert case["automation"] == "command"
    assert case["command"] == [
        "python3",
        "scripts/e2e/regional/run_ha008_processor_exit_acceptance.py",
    ]
    assert case["risk"] == "non-destructive"


@pytest.mark.parametrize(
    "case_id",
    ["GF-REGIONAL-DESTR-005", "GF-REGIONAL-DESTR-006", "GF-REGIONAL-DESTR-007"],
)
def test_warm_spare_guards_use_the_read_only_deployed_audit(case_id: str) -> None:
    case = next(item for item in load_catalog(CATALOG) if item["id"] == case_id)

    assert case["automation"] == "command"
    assert case["command"] == [
        "python3",
        "scripts/e2e/regional/audit_warm_spare_guardrails.py",
        "--case",
        case_id,
    ]
    assert case["risk"] == "live-non-destructive"


def test_promoted_live_drivers_remain_manual_until_revalidated() -> None:
    expected = {
        "GF-REGIONAL-NET-002": "run_net002_command_recovery.py",
        "GF-REGIONAL-NET-003": "run_net003_result_retry.py",
        "GF-REGIONAL-HA-001": "run_ha001_control_plane_failover.py",
        "GF-REGIONAL-HA-002": "run_ha002_pdb_topology.py",
        "GF-REGIONAL-HA-003": "run_ha003_aurora_failover_reset.py",
        "GF-REGIONAL-HA-004": "run_ha004_waiting_reclaim_reset.py",
        "GF-REGIONAL-HA-005": "run_ha005_rollout_continuity.py",
        "GF-REGIONAL-HA-006": "run_ha006_executor_takeover.py",
        "GF-REGIONAL-DESTR-001": "run_destr001_gpu_reset.py",
        "GF-REGIONAL-DESTR-002": "run_destr002_hyperpod_reboot.py",
        "GF-REGIONAL-DESTR-003": "run_destr003_warm_spare_failover.py",
        "GF-REGIONAL-DESTR-008": "run_destr008_warm_spare_shortage.py",
        "GF-REGIONAL-DESTR-009": "run_destr009_workload_restart.py",
        "GF-REGIONAL-DESTR-012": "run_destr012_managed_recovery_guard.py",
        "GF-REGIONAL-DESTR-013": "audit_destr013_replacement_invariant.py",
        "GF-REGIONAL-ISO-006": "run_iso006_cluster_offline.py",
        "GF-REGIONAL-E2E-002": "run_e2e002_multicluster_fault.py",
        "GF-REGIONAL-COLLECT-001": "run_collector_acceptance.py",
        "GF-REGIONAL-COLLECT-002": "run_collector_acceptance.py",
        "GF-REGIONAL-COLLECT-003": "run_collector_acceptance.py",
        "GF-REGIONAL-COLLECT-004": "run_collector_destructive.py",
        "GF-REGIONAL-COLLECT-005": "run_collector_acceptance.py",
        "GF-REGIONAL-COLLECT-008": "run_collector_destructive.py",
        "GF-REGIONAL-COLLECT-009": "run_collector_acceptance.py",
        "GF-REGIONAL-COLLECT-010": "run_collector_acceptance.py",
        "GF-REGIONAL-COLLECT-011": "run_collector_acceptance.py",
        "GF-REGIONAL-COLLECT-012": "run_collector_acceptance.py",
        "GF-REGIONAL-COLLECT-013": "run_collector_destructive.py",
        "GF-REGIONAL-COLLECT-014": "run_collector_destructive.py",
        "GF-REGIONAL-COLLECT-016": "run_collect016_training_recovery.py",
        "GF-REGIONAL-COLLECT-017": "run_collect017_efa_plugin.py",
        "GF-REGIONAL-COLLECT-015": "run_collector_destructive.py",
    }
    cases = {case["id"]: case for case in load_catalog(CATALOG)}
    bodies = _regional_case_bodies()

    for case_id, script_name in expected.items():
        assert cases[case_id]["automation"] == "manual"
        assert "command" not in cases[case_id]
        assert f"scripts/e2e/regional/{script_name}" in bodies[case_id]


def test_parallel_command_cases_declare_resource_locks() -> None:
    cases = {case["id"]: case for case in load_catalog(CATALOG)}
    local_cases = {
        "GF-REGIONAL-BOOT-022",
        "GF-REGIONAL-NET-005",
        "GF-REGIONAL-PREEMPT-017",
        "GF-REGIONAL-PREEMPT-021",
        "GF-REGIONAL-PREEMPT-024",
        "GF-REGIONAL-PREEMPT-025",
        "GF-REGIONAL-PREEMPT-026",
        "GF-REGIONAL-PREEMPT-028",
        "GF-REGIONAL-PREEMPT-029",
        "GF-REGIONAL-PREEMPT-031",
        "GF-REGIONAL-PREEMPT-032",
        "GF-REGIONAL-PREEMPT-033",
        "GF-REGIONAL-PREEMPT-034",
    }

    for case_id in local_cases:
        policy = execution_policy(cases[case_id])
        assert policy.parallel_safe is True
        assert policy.environment == "isolated"
        assert policy.locks == (runner.ResourceLock("local-test", "shared"),)

    cap005 = execution_policy(cases["GF-REGIONAL-CAP-005"])
    assert cap005.parallel_safe is True
    assert cap005.environment == "inherit"
    assert set(cap005.locks) == {
        runner.ResourceLock("postgres-server", "exclusive"),
        runner.ResourceLock("cap005-workdir", "exclusive"),
    }


def test_superseded_manual_case_reports_its_replacement() -> None:
    cases = load_catalog(CATALOG)
    case = next(
        item
        for item in cases
        if (item.get("evidence") or {}).get("verdict") == "SUPERSEDED"
    )

    result = run_case(case)

    assert result["status"] == "NOT_RUN"
    assert result["reason"] == (f"superseded by {case['superseded_by']}")
    assert result["evidence"]["verdict"] == "SUPERSEDED"
    assert result["superseded_by"] == case["superseded_by"]


def test_processing_trace_is_extracted_from_pytest_output() -> None:
    trace, output = _extract_processing_trace(
        "pytest prelude\n"
        + TRACE_PREFIX
        + '{"schema_version":1,"xid":94}\n'
        + ". [100%]\n"
    )

    assert trace == {"schema_version": 1, "xid": 94}
    assert output == "pytest prelude\n. [100%]"


def test_pytest_batch_preserves_per_case_results() -> None:
    common = {
        "title": "batch",
        "category": "test",
        "level": "unit",
        "risk": "non-destructive",
        "problem": "batch",
        "injection": "pytest",
        "expected": ["per-case result"],
        "automation": "pytest",
    }
    cases = [
        {
            **common,
            "id": "GF-BATCH-A",
            "pytest_nodeid": (
                "tests/test_builders.py::"
                "test_shared_builders_preserve_control_plane_defaults"
            ),
        },
        {
            **common,
            "id": "GF-BATCH-B",
            "pytest_nodeid": (
                "tests/test_builders.py::"
                "test_execute_workflow_builds_the_fencing_request"
            ),
        },
        {
            **common,
            "id": "GF-BATCH-PARAM",
            "pytest_nodeid": (
                "tests/host_health/test_dcgm_diagnostic_analysis.py::"
                "test_dcgm_failure_maps_to_fixed_guidance"
            ),
        },
    ]

    results = run_pytest_batch(cases, environment=build_isolated_environment())

    assert results["GF-BATCH-A"]["status"] == "PASS", results
    assert results["GF-BATCH-B"]["status"] == "PASS", results
    assert results["GF-BATCH-PARAM"]["status"] == "PASS", results


def test_pytest_case_reporter_records_failure() -> None:
    pytest_case_reporter.pytest_configure(object())
    pytest_case_reporter.pytest_runtest_logreport(
        SimpleNamespace(
            nodeid="tests/example.py::test_failure",
            duration=0.25,
            when="call",
            outcome="failed",
            capstdout="captured output",
            capstderr="",
            failed=True,
            longrepr="assert False",
        )
    )

    record = pytest_case_reporter.REPORTS["tests/example.py::test_failure"]

    assert record["status"] == "FAIL"
    assert record["duration_seconds"] == 0.25
    assert record["output"] == ["captured output", "assert False"]


def test_pytest_report_control_does_not_use_runtime_environment_namespace() -> None:
    assert not pytest_case_reporter.REPORT_ENV.startswith("GPU_FAULT_"), (
        "pytest reporting control leaked into the strict runtime environment namespace"
    )
    assert runner.PYTEST_BATCH_REPORT_ENV == pytest_case_reporter.REPORT_ENV


def test_pytest_case_reporter_fails_closed_on_skip() -> None:
    pytest_case_reporter.pytest_configure(object())
    pytest_case_reporter.pytest_runtest_logreport(
        SimpleNamespace(
            nodeid="tests/example.py::test_skipped",
            duration=0.01,
            when="setup",
            outcome="skipped",
            capstdout="",
            capstderr="",
            failed=False,
            skipped=True,
            longrepr="requires unavailable dependency",
        )
    )

    record = pytest_case_reporter.REPORTS["tests/example.py::test_skipped"]

    assert record["status"] == "FAIL"
    assert record["output"] == ["requires unavailable dependency"]


def test_pytest_case_reporter_merges_xdist_workers(tmp_path: Path) -> None:
    report = tmp_path / "pytest-results.json"
    environment = build_isolated_environment()
    environment[runner.PYTEST_BATCH_REPORT_ENV] = str(report)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "2",
            "-p",
            "tools.pytest_case_reporter",
            "tests/test_builders.py::test_shared_builders_preserve_control_plane_defaults",
            "tests/test_builders.py::test_execute_workflow_builds_the_fencing_request",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    value = json.loads(report.read_text(encoding="utf-8"))
    assert value["schema_version"] == 1
    assert value["source_identity"] == source_identity(ROOT)
    assert len(value["records"]) == 2


def test_fault_runner_reuses_pytest_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = next(item for item in load_catalog(CATALOG) if item["id"] == "GF-POL-001")
    pytest_results = tmp_path / "pytest-results.json"
    pytest_results.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_identity": source_identity(ROOT),
                "records": {
                    case["pytest_nodeid"]: {
                        "duration_seconds": 0.1,
                        "output": "",
                        "phases": {
                            "setup": "passed",
                            "call": "passed",
                            "teardown": "passed",
                        },
                        "status": "PASS",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "fault-report.json"
    monkeypatch.setattr(
        runner,
        "run_case",
        lambda *_args, **_kwargs: pytest.fail("stored pytest result was re-executed"),
    )
    monkeypatch.setattr(
        runner,
        "run_pytest_batch",
        lambda *_args, **_kwargs: pytest.fail("stored pytest result was re-executed"),
    )

    result = runner.main(
        [
            "--case",
            case["id"],
            "--pytest-results",
            str(pytest_results),
            "--report",
            str(report),
        ]
    )

    assert result == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["results"][0]["status"] == "PASS"


@pytest.mark.parametrize(
    ("catalog_text", "message"),
    [
        (
            """\
schema_version: 1
test_cases:
  - id: GF-BAD-001
    id: GF-BAD-002
""",
            "duplicate YAML key",
        ),
        (
            """\
schema_version: 1
test_cases:
  - id: GF-BAD-001
    title: invalid automation
    category: test
    level: unit
    risk: non-destructive
    problem: invalid
    injection: invalid
    expected: [must fail]
    automation: pyest
""",
            "unsupported test case automation",
        ),
        (
            """\
schema_version: 1
test_cases:
  - id: GF-BAD-001
    title: invalid expected
    category: test
    level: unit
    risk: non-destructive
    problem: invalid
    injection: invalid
    expected: must fail
    automation: pytest
    pytest_nodeid: tests/regional/test_api.py::test_health
""",
            "field expected",
        ),
        (
            """\
schema_version: 1
test_cases:
  - id: GF-BAD-001
    title: invalid risk
    category: test
    level: unit
    risk: destructive-reboot
    problem: invalid
    injection: invalid
    expected: [must fail]
    automation: pytest
    pytest_nodeid: tests/regional/test_api.py::test_health
""",
            "unsupported test case risk",
        ),
        (
            """\
schema_version: 1
test_cases:
  - id: GF-BAD-001
    title: invalid level
    category: test
    level: smoke
    risk: non-destructive
    problem: invalid
    injection: invalid
    expected: [must fail]
    automation: pytest
    pytest_nodeid: tests/regional/test_api.py::test_health
""",
            "unsupported test case level",
        ),
    ],
)
def test_catalog_loader_rejects_malformed_cases(
    tmp_path: Path, catalog_text: str, message: str
) -> None:
    path = tmp_path / "fault-scenarios.yaml"
    path.write_text(catalog_text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_catalog(path)


def test_risk_vocabulary_has_no_dead_or_undeclared_values() -> None:
    # 风险等级只有一份词表：目录的 ``risk`` 和文档的「等级」共用它。
    # 双向相等，所以既挡住未声明的新写法（如 destructive-reboot），
    # 也挡住没人用的僵尸取值。
    catalog_risks = {case["risk"] for case in load_catalog(CATALOG)}
    documented_risks = set(
        REGIONAL_RISK_LABEL.findall(REGIONAL_DOCUMENT.read_text(encoding="utf-8"))
    )

    assert catalog_risks | documented_risks == (runner.RISK_VALUES)


def test_level_vocabulary_has_no_dead_or_undeclared_values() -> None:
    levels = {case["level"] for case in load_catalog(CATALOG)}

    assert levels == runner.LEVEL_VALUES


def test_evidence_verdict_vocabulary_and_supersession_are_closed() -> None:
    cases = load_catalog(CATALOG)
    statuses = {case["evidence"]["verdict"] for case in cases if "evidence" in case}
    case_ids = {case["id"] for case in cases}

    assert statuses == runner.CURRENT_STATUS_VALUES
    for case in cases:
        verdict = (case.get("evidence") or {}).get("verdict")
        if verdict == "SUPERSEDED":
            assert case["automation"] == "manual"
            assert case["superseded_by"] in case_ids
        else:
            assert "superseded_by" not in case


def test_regional_cases_have_machine_readable_verdicts() -> None:
    regional = [
        case for case in load_catalog(CATALOG) if case["id"].startswith("GF-REGIONAL-")
    ]

    assert len(regional) == 157
    assert all((case.get("evidence") or {}).get("verdict") for case in regional)
    # Membership, not equality. The equality form froze the catalog at "no
    # regional case has ever passed", so the first recorded PASS failed this
    # test rather than the catalog validator -- and the only way to keep it
    # green was to not record results, which is the opposite of what a
    # machine-readable verdict is for.
    assert {case["evidence"]["verdict"] for case in regional} <= set(
        runner.CURRENT_STATUS_VALUES
    )
    # A PASS is the one verdict that carries an audit trail, so it may not be
    # written as a bare word: it has to name the case digest it was recorded
    # against and the component builds it was observed on.
    for case in regional:
        if case["evidence"]["verdict"] != "PASS":
            continue
        verified = case["evidence"]["verified"]
        assert verified["case_digest"] == runner.case_definition_digest(case), case[
            "id"
        ]
        assert set(verified["components"]) == set(runner.VERIFIED_COMPONENTS), case[
            "id"
        ]


def test_regional_cases_use_their_maximum_side_effect_risk() -> None:
    cases = {
        case["id"]: case
        for case in load_catalog(CATALOG)
        if case["id"].startswith("GF-REGIONAL-")
    }

    expected = {
        "GF-REGIONAL-AUTH-007": "live-service-action",
        "GF-REGIONAL-AUTH-012": "live-service-action",
        "GF-REGIONAL-AUTH-016": "live-service-action",
        "GF-REGIONAL-E2E-001": "live-workload-restart",
        "GF-REGIONAL-E2E-002": "live-workload-restart",
        "GF-REGIONAL-COLLECT-008": "destructive",
        "GF-REGIONAL-COLLECT-016": "destructive",
    }
    assert {case_id: cases[case_id]["risk"] for case_id in expected} == expected


def test_regional_runtime_inputs_do_not_hardcode_the_legacy_profile() -> None:
    for case in load_catalog(CATALOG):
        if not case["id"].startswith("GF-REGIONAL-"):
            continue
        runtime_text = "\n".join(
            [str(case.get("injection") or ""), *case.get("expected", [])]
        )
        assert "hyperpod-v1" not in runtime_text, case["id"]


def test_catalog_has_only_structured_json_safe_evidence() -> None:
    raw = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    legacy = runner.LEGACY_EVIDENCE_FIELDS

    for case in raw["test_cases"]:
        assert not legacy.intersection(case), case["id"]
        evidence = case.get("evidence")
        if evidence is None:
            continue
        assert set(evidence) <= runner.EVIDENCE_FIELDS, case["id"]
        assert "verdict" in evidence, case["id"]
        json.dumps(evidence)


def test_evidence_bindings_carry_digests_and_nothing_else() -> None:
    """The binding must not become a back door for execution details.

    `docs/evidence/fault/README.md` keeps run times, report paths, cluster names
    and operators out of the public catalog. A digest identifies a code state
    without naming anything about the environment it ran in, which is the only
    reason `evidence.verified` can exist here at all.
    """
    raw = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))

    for case in raw["test_cases"]:
        verified = (case.get("evidence") or {}).get("verified")
        if verified is None:
            continue
        assert set(verified) <= runner.VERIFIED_FIELDS, case["id"]
        values = [verified["case_digest"], *(verified.get("components") or {}).values()]
        assert all(runner.SHA256_PATTERN.match(value) for value in values), case["id"]


def test_pass_verdicts_are_bound_to_the_current_case_definition() -> None:
    """Every recorded `PASS` still describes the case text in the catalog."""
    passed = [
        case
        for case in load_catalog(CATALOG)
        if (case.get("evidence") or {}).get("verdict") == "PASS"
    ]

    assert passed, "the binding contract needs at least one PASS case to guard"
    for case in passed:
        assert case["evidence"]["verified"]["case_digest"] == (
            runner.case_definition_digest(case)
        ), case["id"]


def test_case_definition_digest_ignores_verdict_and_scheduling() -> None:
    """Recording a result or retuning the scheduler must not break a binding."""
    case = next(case for case in load_catalog(CATALOG) if case["id"] == "GF-POL-001")
    baseline = runner.case_definition_digest(case)

    assert (
        runner.case_definition_digest(
            {**case, "evidence": {"verdict": "NOT_RUN"}, "execution": {"locks": []}}
        )
        == baseline
    )
    assert runner.case_definition_digest({**case, "injection": "something else"}) != (
        baseline
    )


def test_unbound_pass_cases_match_the_shrinking_allowlist() -> None:
    """The allowlist is a ratchet, so it must not keep ids that moved on."""
    unbound = {
        case["id"]
        for case in load_catalog(CATALOG)
        if (case.get("evidence") or {}).get("verdict") == "PASS"
        and "components" not in case["evidence"]["verified"]
    }

    assert unbound == set(runner.EVIDENCE_UNBOUND_PASS_CASES)


def test_verified_components_cover_every_shipped_component() -> None:
    from scripts.component_wheels import APPLICATION_COMPONENT_NAMES

    assert set(runner.VERIFIED_COMPONENTS) == set(APPLICATION_COMPONENT_NAMES)


def _case_with(extra: str, *, case_id: str = "GF-STATUS-001") -> str:
    return (
        f"""\
schema_version: 1
test_cases:
  - id: {case_id}
    title: status contract
    category: test
    level: unit
    risk: non-destructive
    problem: invalid status
    injection: none
    expected: [must validate]
    automation: manual
    procedure: docs/故障模拟测试手册.md#1-范围
"""
        + extra
    )


# A sha256-shaped placeholder that YAML still reads as a string: `"0" * 64`
# parses as the integer 0 and would fail the wrong assertion.
DIGEST = "a" * 64


@pytest.mark.parametrize(
    ("extra", "case_id", "message"),
    [
        (
            "    evidence:\n      verdict: PASS\n",
            "GF-STATUS-001",
            "requires an evidence.verified mapping",
        ),
        (
            f"    evidence:\n      verdict: PASS\n      verified:\n"
            f"        case_digest: {DIGEST}\n"
            f"        components:\n          control_plane: {DIGEST}\n",
            "GF-STATUS-001",
            "does not describe this case any more",
        ),
        (
            f"    evidence:\n      verdict: NOT_RUN\n      verified:\n"
            f"        case_digest: {DIGEST}\n",
            "GF-STATUS-001",
            "evidence.verified requires evidence.verdict=PASS",
        ),
        (
            "    evidence:\n      verdict: PASS\n      verified:\n"
            "        case_digest: not-a-digest\n",
            "GF-STATUS-001",
            "must be a lowercase sha256 digest",
        ),
        (
            f"    evidence:\n      verdict: PASS\n      verified:\n"
            f"        case_digest: {DIGEST}\n        note: ran fine\n",
            "GF-STATUS-001",
            "evidence.verified has unknown fields",
        ),
    ],
)
def test_catalog_rejects_broken_pass_bindings(
    tmp_path: Path, extra: str, case_id: str, message: str
) -> None:
    path = tmp_path / "fault-scenarios.yaml"
    path.write_text(_case_with(extra, case_id=case_id), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_catalog(path)


def test_missing_component_binding_is_rejected_outside_the_allowlist(
    tmp_path: Path,
) -> None:
    """A new PASS cannot be recorded without naming the code it ran on."""
    path = tmp_path / "fault-scenarios.yaml"
    path.write_text(_case_with(""), encoding="utf-8")
    case = load_catalog(path)[0]
    digest = runner.case_definition_digest(case)
    path.write_text(
        _case_with(
            f"    evidence:\n      verdict: PASS\n"
            f"      verified:\n        case_digest: {digest}\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="components must name exactly"):
        load_catalog(path)


def test_legacy_unbound_case_cannot_have_components_backfilled(tmp_path: Path) -> None:
    """Stamping today's digest on a 2026-07 live run would invent evidence."""
    case_id = sorted(runner.EVIDENCE_UNBOUND_PASS_CASES)[0]
    path = tmp_path / "fault-scenarios.yaml"
    path.write_text(_case_with("", case_id=case_id), encoding="utf-8")
    digest = runner.case_definition_digest(load_catalog(path)[0])
    path.write_text(
        _case_with(
            f"    evidence:\n      verdict: PASS\n      verified:\n"
            f"        case_digest: {digest}\n        components:\n"
            + "".join(
                f"          {name}: {DIGEST}\n" for name in runner.VERIFIED_COMPONENTS
            ),
            case_id=case_id,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot be backfilled"):
        load_catalog(path)


def test_catalog_contains_no_real_infrastructure_identities() -> None:
    text = CATALOG.read_text(encoding="utf-8")

    assert re.search(r"hyperpod-i-[0-9a-f]{8,}", text) is None
    assert re.search(r"\bhp-cluster-hypd-[A-Za-z0-9-]+\b", text) is None
    assert "control-plane-" + "GPU-fault-solution" not in text
    assert (
        re.search(r"workflow-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", text) is None
    )


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (
            "    evidence:\n      verdict: PASSED\n",
            "unsupported test case evidence verdict",
        ),
        ("    evidence:\n      verdict: SUPERSEDED\n", "superseded_by"),
        (
            "    evidence:\n      verdict: NOT_RUN\n    superseded_by: GF-OTHER\n",
            "superseded_by requires evidence.verdict=SUPERSEDED",
        ),
    ],
)
def test_catalog_rejects_invalid_current_status_contract(
    tmp_path: Path, extra: str, message: str
) -> None:
    path = tmp_path / "fault-scenarios.yaml"
    path.write_text(_case_with(extra), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_catalog(path)


def test_document_declares_every_risk_value() -> None:
    text = REGIONAL_DOCUMENT.read_text(encoding="utf-8")
    start = text.index(REGIONAL_RISK_TABLE_HEADING)
    end = text.index("\n## ", start)
    declared = set(REGIONAL_RISK_TABLE_ROW.findall(text[start:end]))

    assert declared == runner.RISK_VALUES


def test_fault_manual_uses_current_catalog_profile_and_evidence_contracts() -> None:
    text = FAULT_MANUAL.read_text(encoding="utf-8")
    ids = {case["id"] for case in load_catalog(CATALOG)}
    references = set(re.findall(r"`(GF-[A-Z0-9][A-Z0-9_.-]*\*?)`", text))
    missing = sorted(
        value
        for value in references
        if not value.endswith(("*", "-")) and value not in ids
    )

    assert missing == []
    assert "hyperpod-control-plane-recovery-v1" not in text
    assert '--site "${SITE_FILE}"' in text
    assert "evidence.run_at" not in text
    assert "| `destructive-provider-replace`" not in text
    assert "destructive-provider-replace`不是合法risk" in text
    assert "`manual`、`pytest` 和 `command`" in text
    assert "结果记录在" not in text
    assert "历史报告" not in text
    assert re.findall(r"docs/evidence/fault/[^`\\s]+\\.json", text) == []
    assert "厂商批准的" not in text
    assert "硬件厂商批准" not in text
    assert "### 7.8 真实硬件故障注入未实现" in text

    unavailable = next(
        case for case in load_catalog(CATALOG) if case["id"] == "GF-LIVE-004"
    )
    assert unavailable["risk"] == "non-destructive"
    assert unavailable["automation"] == "manual"
    assert unavailable["evidence"]["verdict"] == "BLOCKED"
    assert unavailable["injection"].startswith("不执行故障注入"), (
        "unavailable scenarios must remain explicitly non-injecting"
    )


def _regional_case_bodies() -> dict[str, str]:
    lines = REGIONAL_DOCUMENT.read_text(encoding="utf-8").splitlines()
    starts: list[tuple[int, str | None]] = []
    for index, line in enumerate(lines):
        heading = REGIONAL_CASE_HEADING.match(line)
        if heading is not None:
            starts.append((index, heading.group(1)))
        elif line.startswith("### ") or line.startswith("## "):
            starts.append((index, None))
    bodies = {}
    for position, (index, case_id) in enumerate(starts):
        if case_id is None:
            continue
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        bodies[case_id] = "\n".join(lines[index + 1 : end])
    return bodies


def test_cap005_uses_the_postgres_stress_runner_consistently() -> None:
    case = next(
        case for case in load_catalog(CATALOG) if case["id"] == "GF-REGIONAL-CAP-005"
    )
    command = ["python3", "scripts/e2e/regional/run_cap005_postgres_suite.py"]

    assert case["automation"] == "command"
    assert case["command"] == command
    assert "make test-postgres-stress" in case["injection"]
    assert "8×40" in case["injection"]
    assert any("concurrency" in item and "2与4" in item for item in case["expected"]), (
        "CAP-005 must require real PostgreSQL completion concurrency at 2 and 4"
    )

    runner_path = ROOT / command[1]
    runner_text = runner_path.read_text(encoding="utf-8")
    tree = ast.parse(runner_text)
    literal_commands = [
        [element.value for element in node.elts]
        for node in ast.walk(tree)
        if isinstance(node, ast.List)
        and all(
            isinstance(element, ast.Constant) and isinstance(element.value, str)
            for element in node.elts
        )
    ]
    assert ["make", "test-postgres-stress", "PYTHON=python3"] in literal_commands
    assert "GPU_FAULT_CAP005_WORKDIR" in runner_text
    assert '"/work"' not in runner_text

    body = _regional_case_bodies()["GF-REGIONAL-CAP-005"]
    assert command[1] in body
    assert "make test-postgres-stress" in body
    assert "`2` 与 `4`" in body


def test_every_documented_execution_record_is_reflected_in_the_catalog() -> None:
    # 公开规格不得重新混入执行结果。若某段仍以执行记录措辞出现，
    # catalog 必须至少具有机器 verdict，避免它成为无法判读的散文结论。
    catalog = {
        case["id"]: case
        for case in load_catalog(CATALOG)
        if case["id"].startswith("GF-REGIONAL-")
    }
    bodies = _regional_case_bodies()
    missing = sorted(
        case_id
        for case_id, body in bodies.items()
        if REGIONAL_EXECUTION_RECORD.search(body)
        and not (catalog[case_id].get("evidence") or {}).get("verdict")
    )

    assert missing == []


def test_catalog_and_document_agree_on_the_regional_case_set() -> None:
    # 集合相等，两个方向都守：文档新增用例必须进目录（否则它既不会被
    # runner 选中，也不会进 --collect-only 的引用体检），目录里也不许留
    # 文档已删掉的孤儿条目。与 tests/regional/test_regional_execution_order.py
    # 对执行顺序的守法一致。
    documented = set(
        REGIONAL_CASE_HEADING.findall(REGIONAL_DOCUMENT.read_text(encoding="utf-8"))
    )
    catalogued = {
        case["id"]
        for case in load_catalog(CATALOG)
        if case["id"].startswith("GF-REGIONAL-")
    }

    assert documented == catalogued
