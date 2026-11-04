"""join-cluster / remove-cluster on the deploy host: the scoped verify, its record.

The engine's verify scoped to a membership verb's target reports other GPU
clusters' own findings instead of failing on them (``regional_verify_scope``).
The verb runs it with its target, keeps those findings in its ``VERIFIED``
evidence and its result, and leaves one record ``status`` prints until the next
membership verb or a passing verify of the whole site.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import cluster_join as join
from gpu_fault.admin import membership_verify as SCOPED
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from gpu_fault.admin.status_report import print_status_report
from tests.admin._cov95_join_support import JoinScenario
from tests.admin._cov95_removal_support import RemovalTransport
from tests.admin.test_admin_site import site_file

FINDING = {
    "check": "control_api",
    "cluster_id": "gpu-eks-cluster",
    "finding": "gpu-eks-cluster Agent coverage drift: expected=['ip-a'], "
    "agents=['ip-a', 'ip-gone']",
}


def scoped_report(*findings: dict[str, str], healthy: bool = True) -> str:
    """A scoped engine verify report, as the engine prints it."""

    return json.dumps(
        {
            "mode": "verify",
            "healthy": healthy,
            "summary": {"PASS": 14, "WARN": len(findings), "FAIL": 0, "SKIP": 0},
            "checks": [],
            "cluster_scope": {
                "verb": "remove-cluster",
                "target_cluster_ids": ["gpu-a"],
                "not_gating_findings": list(findings),
            },
        }
    )


# -- remove-cluster -----------------------------------------------------------------


@pytest.fixture
def removal(tmp_path, monkeypatch) -> tuple[RemovalTransport, list[list[str]]]:
    transport = RemovalTransport(tmp_path, monkeypatch)
    verifies: list[list[str]] = []
    original = transport.driver

    def driver(arguments: list[str], **options: Any) -> Any:
        if arguments[1] != "verify":
            return original(arguments, **options)
        original(arguments, **options)
        verifies.append(list(arguments[1:]))
        assert options.get("stdout") == subprocess.PIPE, "the report is read"
        return transport.result(arguments, scoped_report(FINDING))

    monkeypatch.setattr("gpu_fault.admin.cluster_removal.run_driver", driver)
    return transport, verifies


def test_a_removal_verifies_the_remaining_site_scoped_and_keeps_its_findings(
    removal, capsys
):
    transport, verifies = removal

    result = transport.remove()

    assert result["phase"] == "COMPLETED"
    assert verifies == [
        ["verify", "--config", verifies[0][2], "--removed-cluster-id", "gpu-a"]
    ], "the remaining site's verify is scoped to the removal"
    assert result[SCOPED.FINDINGS_KEY] == [FINDING]
    verified = transport.state()["evidence"]["VERIFIED"]
    assert verified["cpu_control_plane_preserved"] is True
    assert verified[SCOPED.FINDINGS_KEY] == [FINDING]
    record = json.loads(SCOPED.record_path(transport.path.parent).read_text())
    assert (record["verb"], record["cluster_id"]) == ("remove-cluster", "gpu-a")
    assert '"not_gating_findings"' in capsys.readouterr().out, (
        "the captured report is printed again for the command log"
    )


def test_a_removal_whose_remaining_site_fails_still_fails(removal, monkeypatch):
    transport, _verifies = removal

    def failing(arguments: list[str], **options: Any) -> Any:
        if arguments[1] == "verify":
            return transport.result(arguments, scoped_report(healthy=False), code=1)
        return RemovalTransport.driver(transport, arguments, **options)

    monkeypatch.setattr("gpu_fault.admin.cluster_removal.run_driver", failing)

    with pytest.raises(BootstrapError, match="remaining CPU control plane"):
        transport.remove()

    assert transport.state()["phase"] != "COMPLETED"
    assert not SCOPED.record_path(transport.path.parent).exists(), (
        "a removal that did not complete leaves no findings record"
    )


# -- join-cluster (batch) ------------------------------------------------------------


def test_a_batch_join_scopes_its_verify_to_every_joining_cluster(tmp_path, monkeypatch):
    scenario = JoinScenario(tmp_path, monkeypatch)
    verifies: list[list[str]] = []

    def driver(arguments: list[str], **options: Any) -> Any:
        if arguments[1] == "verify":
            verifies.append(list(arguments[1:]))
        return scenario.driver(arguments, **options)

    monkeypatch.setattr(join, "run_driver", driver)

    result = scenario.batch()

    assert verifies and all(
        arguments[3:]
        == ["--joining-cluster-id", "hp-gpu-b", "--joining-cluster-id", "hp-gpu-c"]
        for arguments in verifies
    ), verifies
    assert result["joined"], result


def test_the_join_command_line_scopes_only_its_verify() -> None:
    rollout, config = Path("rollout.sh"), Path("cfg.json")

    assert SCOPED.rollout_arguments(rollout, "verify", config, "hp-b") == [
        "rollout.sh",
        "verify",
        "--config",
        "cfg.json",
        "--joining-cluster-id",
        "hp-b",
    ]
    assert SCOPED.rollout_arguments(rollout, "join-cluster", config, "hp-b") == [
        "rollout.sh",
        "join-cluster",
        "--cluster-id",
        "hp-b",
        "--config",
        "cfg.json",
    ]
    assert SCOPED.rollout_arguments(rollout, "preflight", config, None)[1] == (
        "preflight"
    )
    with pytest.raises(BootstrapError, match="exactly one cluster id"):
        SCOPED.rollout_arguments(rollout, "activate-cluster", config, ["a", "b"])
    with pytest.raises(BootstrapError, match="no cluster-scoped verify"):
        SCOPED.verify_arguments("remove-cluster", ["a", "b"])


# -- the record status prints ---------------------------------------------------------


def completed_state(*findings: dict[str, str]) -> dict[str, Any]:
    return {"evidence": {"VERIFIED": {SCOPED.FINDINGS_KEY: list(findings)}}}


def test_findings_are_redacted_and_capped_before_they_are_kept() -> None:
    secret = {**FINDING, "finding": "drift; token=abc123def456 " + "x" * 2000}

    (kept,) = SCOPED.scoped_evidence(json.loads(scoped_report(secret)))[
        SCOPED.FINDINGS_KEY
    ]

    assert "abc123def456" not in kept["finding"]
    assert kept["finding"].startswith("drift; token="), kept
    assert kept["finding"].endswith(" [...]"), "the head is kept, the tail cut"
    assert len(kept["finding"]) <= SCOPED.FINDING_LIMIT + len(" [...]")
    assert SCOPED.scoped_evidence(None) == {}
    assert SCOPED.scoped_evidence({"mode": "verify"}) == {}


def test_status_prints_the_last_verbs_findings(tmp_path, capsys) -> None:
    path = site_file(tmp_path)
    fields = SCOPED.finish_verb(
        tmp_path, verb="join-cluster", cluster_id="hp-b", state=completed_state(FINDING)
    )

    print_status_report(json.dumps({"mode": "status", "healthy": True}), site_file=path)

    captured = capsys.readouterr()
    assert fields == {SCOPED.FINDINGS_KEY: [FINDING]}
    assert (
        "finding: join-cluster hp-b completed at " in captured.err
        and "past 1 finding(s) on GPU clusters it does not target" in captured.err
    ), captured.err
    assert (
        "finding: join-cluster hp-b: not gating: gpu-eks-cluster control_api: "
        "gpu-eks-cluster Agent coverage drift"
    ) in captured.err
    document = json.loads(captured.out)
    assert document[SCOPED.FIELD][SCOPED.FINDINGS_KEY] == [FINDING]


def test_the_next_verb_without_findings_removes_the_record(tmp_path) -> None:
    SCOPED.finish_verb(
        tmp_path, verb="join-cluster", cluster_id="hp-b", state=completed_state(FINDING)
    )

    assert (
        SCOPED.finish_verb(
            tmp_path, verb="remove-cluster", cluster_id="hp-b", state=completed_state()
        )
        == {}
    )

    assert SCOPED.status_view(tmp_path) == ({}, [])


@pytest.mark.parametrize("code,kept", [(0, False), (1, True)])
def test_a_passing_verify_of_the_whole_site_clears_the_record(
    tmp_path, monkeypatch, code: int, kept: bool
) -> None:
    path = site_file(tmp_path)
    SCOPED.finish_verb(
        tmp_path, verb="join-cluster", cluster_id="hp-b", state=completed_state(FINDING)
    )
    commands: list[list[str]] = []

    def run_driver(arguments: list[str], **_kwargs: Any) -> Any:
        commands.append(list(arguments))
        return subprocess.CompletedProcess(arguments, code, None, None)

    monkeypatch.setattr(admin_cli, "run_driver", run_driver)
    arguments = argparse.Namespace(command="verify", repo_root=None, state_dir=tmp_path)

    assert admin_cli._run_readonly_managed_command(arguments, path) == code

    assert [item[1] for item in commands] == ["verify"]
    assert "--joining-cluster-id" not in commands[0], "the deploy's verify is unscoped"
    assert SCOPED.record_path(tmp_path).exists() is kept
    assert load_site(path).source == path


def test_status_names_five_findings_counts_the_rest_and_an_unreadable_record(
    tmp_path,
) -> None:
    findings = [{**FINDING, "cluster_id": f"gpu-{index}"} for index in range(7)]
    SCOPED.finish_verb(
        tmp_path,
        verb="remove-cluster",
        cluster_id="gpu-x",
        state=completed_state(*findings),
    )

    field, lines = SCOPED.status_view(tmp_path)

    assert len(field[SCOPED.FIELD][SCOPED.FINDINGS_KEY]) == 7
    assert len(lines) == 1 + SCOPED.STATUS_LINE_LIMIT + 1
    assert lines[-1] == "2 more in membership_verify_findings.not_gating_findings"

    SCOPED.record_path(tmp_path).write_text("{not json", encoding="utf-8")
    field, lines = SCOPED.status_view(tmp_path)
    assert field == {}
    assert len(lines) == 1 and "is unreadable" in lines[0], lines


def test_the_echoed_report_is_read_past_lines_ahead_of_it(capsys) -> None:
    document = json.dumps(json.loads(scoped_report(FINDING)), indent=2)
    result = subprocess.CompletedProcess(["verify"], 0, f"narration\n{document}\n", "")

    report = SCOPED.echoed_report(result)

    assert SCOPED.scoped_evidence(report) == {SCOPED.FINDINGS_KEY: [FINDING]}
    assert capsys.readouterr().out == f"narration\n{document}\n", "printed as captured"
    assert SCOPED.echoed_report(subprocess.CompletedProcess([], 1, "", "")) is None
    assert SCOPED.echoed_report(subprocess.CompletedProcess([], 0, None, None)) is None
