from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.e2e.regional import run_blast_acceptance as blast
from scripts.e2e.regional import run_boot_acceptance as boot
from scripts.e2e.regional import run_capacity_acceptance as capacity
from scripts.e2e.regional import run_identity_acceptance as identity
from scripts.e2e.regional import run_notification_acceptance as notification
from scripts.e2e.regional import run_preempt012_acceptance as preempt012
from scripts.e2e.regional import run_preemption_contracts as preemption
from scripts.e2e.regional import run_workload_acceptance as workload
from scripts.e2e.regional.regional_case_contract import (
    case_metadata,
    formal_predecessor,
)

ROOT = Path(__file__).resolve().parents[2]
REGIONAL = ROOT / "scripts/e2e/regional"
CATALOG = ROOT / "testcases/fault-scenarios.yaml"
SPEC = ROOT / "docs/区域模式端到端验收测试用例.md"

REQUESTED_CASES = {
    *(f"GF-REGIONAL-BOOT-{number:03d}" for number in range(11, 19)),
    "GF-REGIONAL-AUTH-007",
    "GF-REGIONAL-AUTH-010",
    # AUTH-012 is SUPERSEDED by AUTH-016 and has no driver.
    *(f"GF-REGIONAL-AUTH-{number:03d}" for number in range(13, 17)),
    "GF-REGIONAL-WORKLOAD-001",
    "GF-REGIONAL-WORKLOAD-002",
    "GF-REGIONAL-ISO-001",
    "GF-REGIONAL-ISO-003",
    "GF-REGIONAL-ISO-004",
    "GF-REGIONAL-ISO-005",
    *(f"GF-REGIONAL-PREEMPT-{number:03d}" for number in range(1, 10)),
    "GF-REGIONAL-PREEMPT-012",
    "GF-REGIONAL-E2E-001",
    *(f"GF-REGIONAL-BLAST-{number:03d}" for number in range(1, 5)),
    *(f"GF-REGIONAL-NOTIFY-{number:03d}" for number in range(1, 6)),
    *(f"GF-REGIONAL-CAP-{number:03d}" for number in range(1, 5)),
    "GF-REGIONAL-NET-001",
}


def test_requested_cases_have_complete_reusable_entries() -> None:
    implemented = {
        *boot.CASE_IDS,
        *identity.CASE_IDS,
        *workload.CASE_IDS,
        *preemption.CASE_NODEIDS,
        preempt012.CASE_ID,
        *blast.CASE_IDS,
        *notification.CASE_IDS,
        *capacity.CASE_IDS,
        "GF-REGIONAL-NET-001",
    }

    assert implemented == REQUESTED_CASES


def test_reusable_entries_follow_the_formal_predecessor_chain() -> None:
    assert formal_predecessor("GF-REGIONAL-BOOT-011") is None
    assert formal_predecessor("GF-REGIONAL-AUTH-007") == "GF-REGIONAL-AUTH-006"
    # AUTH-012 is skipped in the execution order, so AUTH-013 follows AUTH-011.
    assert formal_predecessor("GF-REGIONAL-AUTH-013") == "GF-REGIONAL-AUTH-011"
    assert formal_predecessor("GF-REGIONAL-WORKLOAD-001") == "GF-REGIONAL-AUTH-016"
    assert formal_predecessor("GF-REGIONAL-ISO-001") == ("GF-REGIONAL-WORKLOAD-002")
    assert formal_predecessor("GF-REGIONAL-PREEMPT-012") == ("GF-REGIONAL-PREEMPT-011")
    assert formal_predecessor("GF-REGIONAL-E2E-001") == ("GF-REGIONAL-PREEMPT-038")
    assert formal_predecessor("GF-REGIONAL-NET-001") == "GF-REGIONAL-CAP-005"
    assert all(
        case_metadata(case_id).predecessor == formal_predecessor(case_id)
        for case_id in REQUESTED_CASES
    ), "case metadata predecessor drifted from the formal execution order"


def test_preempt_001_to_009_are_real_command_entries() -> None:
    cases = {
        item["id"]: item
        for item in yaml.safe_load(CATALOG.read_text(encoding="utf-8"))["test_cases"]
    }
    for case_id in preemption.CASE_NODEIDS:
        case = cases[case_id]
        assert case["automation"] == "command"
        assert case["command"] == [
            "python3",
            "scripts/e2e/regional/run_preemption_contracts.py",
            "--case",
            case_id,
        ]


def test_mutating_new_live_drivers_are_plan_only_by_default() -> None:
    parsed = (
        boot.parser().parse_args(
            [
                "--case",
                "GF-REGIONAL-BOOT-012",
                "--run-dir",
                "/tmp/run",
                "--site",
                "/tmp/site.yaml",
            ]
        ),
        identity.parser().parse_args(
            [
                "--case",
                "GF-REGIONAL-AUTH-010",
                "--run-dir",
                "/tmp/run",
                "--site",
                "/tmp/site.yaml",
            ]
        ),
        workload.parser().parse_args(
            [
                "--case",
                "GF-REGIONAL-WORKLOAD-001",
                "--run-dir",
                "/tmp/run",
                "--site",
                "/tmp/site.yaml",
            ]
        ),
        notification.parser().parse_args(
            [
                "--case",
                "GF-REGIONAL-NOTIFY-001",
                "--run-dir",
                "/tmp/run",
                "--site",
                "/tmp/site.yaml",
            ]
        ),
        preempt012.parser().parse_args(["--run-dir", "/tmp/run"]),
    )

    assert all(item.execute is False for item in parsed), (
        "a mutating acceptance parser defaults to execute mode"
    )


def test_promoted_sources_have_no_site_specific_topology() -> None:
    names = (
        "blast_acceptance_base.py",
        "blast_acceptance_cases_1.py",
        "blast_acceptance_cases_2.py",
        "boot_acceptance_common.py",
        "boot_acceptance_lifecycle.py",
        "boot_acceptance_runtime.py",
        "capacity_acceptance_base.py",
        "capacity_acceptance_cases.py",
        "identity_acceptance_auth.py",
        "identity_acceptance_common.py",
        "identity_acceptance_iso.py",
        "run_blast_acceptance.py",
        "run_boot_acceptance.py",
        "run_capacity_acceptance.py",
        "run_identity_acceptance.py",
        "run_net001_collector_replay.py",
        "run_notification_acceptance.py",
        "run_preempt012_acceptance.py",
        "run_preemption_contracts.py",
        "run_workload_acceptance.py",
        "probes/auth015_node_probe.py",
        "probes/e2e001_node_probe.py",
        "probes/net001_node_probe.py",
        "probes/notification_drill.py",
        "probes/preempt012_node_probe.py",
    )
    for name in names:
        source = (REGIONAL / name).read_text(encoding="utf-8")
        assert "/secure/" not in source, name
        assert re.search(r"\bhyperpod-i-[0-9a-f]{8,}\b", source) is None, name
        assert re.search(r"\bhp-cluster-hypd-[A-Za-z0-9-]+\b", source) is None, name
        assert (
            re.search(r"\barn:aws:[^:\s]+:[^:\s]*:(?!0{12})\d{12}:", source) is None
        ), name


def test_new_public_entrypoints_expose_safety_help() -> None:
    plan_only = (
        "run_boot_acceptance.py",
        "run_capacity_acceptance.py",
        "run_identity_acceptance.py",
        "run_net001_collector_replay.py",
        "run_notification_acceptance.py",
        "run_preempt012_acceptance.py",
        "run_workload_acceptance.py",
    )
    for name in plan_only:
        completed = subprocess.run(
            [sys.executable, str(REGIONAL / name), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        for option in ("--plan", "--execute", "--confirm", "--maintenance-window-end"):
            assert option in completed.stdout, (name, option)


def test_public_spec_points_to_every_new_runner_family() -> None:
    text = SPEC.read_text(encoding="utf-8")
    assert "### 2.3 完整可复用验收入口" in text
    for name in (
        "run_boot_acceptance.py",
        "run_identity_acceptance.py",
        "run_workload_acceptance.py",
        "run_preemption_contracts.py",
        "run_preempt012_acceptance.py",
        "run_blast_acceptance.py",
        "run_notification_acceptance.py",
        "run_capacity_acceptance.py",
        "run_net001_collector_replay.py",
    ):
        assert name in text


def test_external_notification_evidence_is_structured(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "received": True,
                "reference": "mailbox-audit-1",
                "method": "inbox-screenshot",
            }
        ),
        encoding="utf-8",
    )
    dedup = tmp_path / "dedup.json"
    dedup.write_text(
        json.dumps({"send_count_delta": 0, "duplicate_inbox_count": 0}),
        encoding="utf-8",
    )

    assert notification.validate_external_evidence(receipt, "receipt")["valid"], (
        "valid receipt evidence was rejected"
    )
    assert notification.validate_external_evidence(dedup, "dedup")["valid"], (
        "valid deduplication evidence was rejected"
    )

    # `method` is required, and this is the reason: a human inbox check and a
    # windowed SES `Delivery` count both answer NOTIFY-001's question, but they
    # are not interchangeable to whoever audits the report later. Without a
    # named method a bare `received: true` cannot be told apart from a guess.
    unmethodical = tmp_path / "unmethodical.json"
    unmethodical.write_text(
        json.dumps({"received": True, "reference": "mailbox-audit-1"}), encoding="utf-8"
    )
    assert not notification.validate_external_evidence(unmethodical, "receipt")[
        "valid"
    ], "receipt evidence was accepted without saying how delivery was established"
