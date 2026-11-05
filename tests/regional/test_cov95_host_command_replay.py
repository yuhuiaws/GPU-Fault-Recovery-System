"""The host command replay wrapper's serving path, in the test process.

``tests/regional/test_host_command_replay.py`` runs the wrapper as the
``python -I`` process a collector unit starts, which proves the shim chain end
to end. Here the same entry points -- ``serve``, ``main``, the call counter,
the journal and the plan loader -- run in-process: the real binary is a
recorded stand-in for ``subprocess.run`` (and, for ``main``, a file on a
temporary PATH), the plan and the audit live under ``tmp_path``, and every test
reads back what was served, what reached the real binary and what was
journaled.
"""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from tools import host_command_replay as wrapper

UUID_A = "GPU-11111111-2222-3333-4444-555555555555"
UUID_B = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
NOW = 1_800_000_000.0
HEALTH = ["-H", "-j", "/dev/nvme0"]
SCAN = ["--scan-open"]
STATS = ["-S", "rdmap79s0"]
REAL = "/usr/sbin/smartctl"
NVIDIA_SMI = "/usr/bin/nvidia-smi"
HEALTHY = json.dumps({"smartctl": {"exit_status": 0}, "smart_status": {"passed": True}})
NIC_STATS = "NIC statistics:\n     rx_drops: 7\n     tx_drops: 1\n"
#: Rewrites UUID_A's utilization column to idle.
SET_IDLE = {
    "query": "utilization",
    "action": "set_column",
    "uuid": UUID_A,
    "column": "utilization.gpu",
    "value": "0",
}


def plan_of(tool: str, *rules: dict[str, Any], not_after: float = NOW + 600) -> dict:
    return {"tool": tool, "not_after": not_after, "rules": list(rules)}


def install_plan(
    run_dir: Path, tool: str, *rules: dict[str, Any], not_after: float = NOW + 600
) -> Path:
    path = wrapper.plan_path(str(run_dir), tool)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan_of(tool, *rules, not_after=not_after)))
    return path


def journaled(run_dir: Path, tool: str) -> list[dict[str, Any]]:
    path = wrapper.audit_dir(str(run_dir)) / f"{tool}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class RealBinary:
    """``subprocess.run`` stand-in for the real tool; records every call."""

    def __init__(self) -> None:
        self.stdout, self.stderr, self.returncode = "", "", 0
        self.error: BaseException | None = None
        self.calls: list[dict[str, Any]] = []

    def answers(self, stdout: str, returncode: int = 0, stderr: str = "") -> None:
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr

    def __call__(self, argv: list[str], **options: Any) -> Any:
        self.calls.append({"argv": list(argv), **options})
        if self.error is not None:
            raise self.error
        return subprocess.CompletedProcess(
            argv, self.returncode, self.stdout, self.stderr
        )


@pytest.fixture
def real(monkeypatch: pytest.MonkeyPatch) -> RealBinary:
    binary = RealBinary()
    monkeypatch.setattr(subprocess, "run", binary)
    return binary


# --------------------------------------------------------------------------- #
# plan validation and the pure helpers
# --------------------------------------------------------------------------- #
def test_a_rule_is_an_object_and_an_integer_json_field_is_bounded() -> None:
    with pytest.raises(wrapper.PlanError, match="rule 0 must be an object"):
        wrapper.validate_plan("smartctl", plan_of("smartctl", ["health"]))  # type: ignore[arg-type]
    status = {
        "query": "health",
        "action": "set_json",
        "field": "smartctl.exit_status",
        "value": 4,
    }
    checked = wrapper.validate_plan("smartctl", plan_of("smartctl", status))
    assert checked["rules"] == [status], "an exit status in 0..255 is reviewed"
    with pytest.raises(
        wrapper.PlanError, match=r"rule 1 smartctl\.exit_status must be an integer"
    ):
        wrapper.validate_plan(
            "smartctl", plan_of("smartctl", status, {**status, "value": 256})
        )


def test_a_rule_applies_to_its_query_and_only_to_its_selected_device() -> None:
    device = {"query": "health", "action": "passthrough", "device": "/dev/nvme0"}
    assert wrapper.rule_applies("smartctl", device, "health", HEALTH) is True, (
        "the selected device"
    )
    assert (
        wrapper.rule_applies("smartctl", device, "health", ["-H", "-j", "/dev/nvme1"])
        is False
    ), "another device is not selected"
    assert wrapper.rule_applies("smartctl", device, "health", ["-H", "-j"]) is False, (
        "an argv without the selector position never matches a selected rule"
    )
    assert wrapper.rule_applies("smartctl", device, "scan", SCAN) is False, (
        "a rule never applies to another query"
    )
    every = {"query": "health", "action": "passthrough"}
    assert (
        wrapper.rule_applies("smartctl", every, "health", ["-H", "-j", "/dev/sda"])
        is True
    ), "a rule without a selector covers every device"
    scan = {"query": "scan", "action": "exit", "exit_code": 1}
    assert wrapper.rule_applies("smartctl", scan, "scan", SCAN) is True, (
        "a query without a selector matches on its name"
    )


def test_apply_rule_passthrough_and_json_paths() -> None:
    assert wrapper.apply_rule(
        {"query": "scan", "action": "passthrough"}, "/dev/nvme0\n", 3, "warn"
    ) == ("/dev/nvme0\n", 3, "warn", None), "passthrough serves the real answer"
    passed = {"action": "set_json", "field": "smart_status.passed", "value": False}
    out, code, _err, reason = wrapper.apply_rule(
        passed, json.dumps({"smartctl": {"exit_status": 0}}), 0, ""
    )
    assert (json.loads(out), code, reason) == (
        {"smartctl": {"exit_status": 0}, "smart_status": {"passed": False}},
        0,
        None,
    ), "a missing parent object is created"
    out, _code, _err, _reason = wrapper.apply_rule(
        passed, json.dumps({"smart_status": "PASSED"}), 0, ""
    )
    assert json.loads(out) == {"smart_status": {"passed": False}}, (
        "a parent that is not an object is replaced by one"
    )
    nested = {"action": "delete_json", "field": "smart_status.passed"}
    out, _code, _err, reason = wrapper.apply_rule(
        nested, json.dumps({"smart_status": {"passed": True, "kept": 1}}), 0, ""
    )
    assert (json.loads(out), reason) == ({"smart_status": {"kept": 1}}, None), (
        "a nested delete removes only the leaf"
    )
    out, _code, _err, reason = wrapper.apply_rule(
        nested, json.dumps({"smart_status": "PASSED"}), 0, ""
    )
    assert (json.loads(out), reason) == ({"smart_status": "PASSED"}, None), (
        "a delete through a non-object leaves the document as it was"
    )
    listing = json.dumps([{"smart_status": {"passed": True}}])
    assert wrapper.apply_rule(passed, listing, 0, "") == (
        listing,
        0,
        "",
        "the real output is not a JSON object",
    ), "a JSON array is served as it came"


def test_apply_rule_names_an_action_it_does_not_know() -> None:
    assert wrapper.apply_rule({"action": "rewrite_all"}, "real\n", 0, "") == (
        "real\n",
        0,
        "",
        "unknown action rewrite_all",
    ), "an unknown action changes nothing and says why"


def test_apply_rule_never_claims_a_column_the_row_does_not_carry() -> None:
    rows = f"{UUID_A}\n{UUID_B}, 97\n"
    assert wrapper.apply_rule(SET_IDLE, rows, 0, "") == (
        rows,
        0,
        "",
        f"the row for {UUID_A} has no utilization.gpu column",
    ), "a row too short for the column is served as it came, with the reason"
    assert wrapper.apply_rule(SET_IDLE, f"{UUID_A}, 55\n", 0, "") == (
        f"{UUID_A}, 0\n",
        0,
        "",
        None,
    ), "the same rule rewrites a row that carries the column"


def test_real_binary_skips_empty_entries_and_links_into_the_replay_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replay_root = tmp_path / "gfrs-live"
    planted = replay_root / "run-1" / "bin" / "smartctl"
    planted.parent.mkdir(parents=True)
    planted.write_text("#!/bin/sh\n")
    planted.chmod(0o755)
    monkeypatch.setattr(wrapper, "REPLAY_ROOT", str(replay_root))
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "smartctl").symlink_to(planted)
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "smartctl").write_text("#!/bin/sh\n")
    (real_dir / "smartctl").chmod(0o755)
    path_value = os.pathsep.join(["", str(linked), "", str(real_dir)])
    assert wrapper.real_binary("smartctl", path_value) == str(real_dir / "smartctl"), (
        "a link that resolves below the replay root is never the real binary"
    )
    assert (
        wrapper.real_binary("smartctl", os.pathsep.join(["", str(linked)])) is None
    ), "with only replay-root candidates there is no real binary"


def test_plan_and_audit_live_in_the_run_directory(tmp_path: Path) -> None:
    run_dir = str(tmp_path / "run")
    assert wrapper.plan_path(run_dir, "ethtool") == (
        tmp_path / "run" / "fixtures" / "plan-ethtool.json"
    ), "one reviewed plan per tool under fixtures/"
    assert wrapper.audit_dir(run_dir) == tmp_path / "run" / "audit", (
        "the audit directory is the unit's one writable path"
    )


def test_load_plan_reports_why_a_plan_is_unusable(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert wrapper.load_plan(str(run_dir), "smartctl") == (
        None,
        "no readable plan: FileNotFoundError",
    ), "no plan file"
    path = wrapper.plan_path(str(run_dir), "smartctl")
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert wrapper.load_plan(str(run_dir), "smartctl") == (
        None,
        "no readable plan: JSONDecodeError",
    ), "an unparsable plan"
    path.write_text(json.dumps({"tool": "smartctl", "rules": []}))
    assert wrapper.load_plan(str(run_dir), "smartctl") == (
        None,
        "plan rejected: a plan has exactly the keys tool, not_after and rules",
    ), "a plan that fails review"
    rule = {"query": "scan", "action": "exit", "exit_code": 2}
    install_plan(run_dir, "smartctl", rule)
    plan, problem = wrapper.load_plan(str(run_dir), "smartctl")
    assert problem is None and plan == plan_of("smartctl", rule), (plan, problem)


# --------------------------------------------------------------------------- #
# the call counter and the journal
# --------------------------------------------------------------------------- #
def test_consume_call_counts_per_tool_and_rule_and_stops_at_the_bound(
    tmp_path: Path,
) -> None:
    run_dir = str(tmp_path / "run")
    assert [wrapper.consume_call(run_dir, "smartctl", 0, 2) for _ in range(3)] == [
        True,
        True,
        False,
    ], "a rule bounded to two calls applies twice"
    assert [wrapper.consume_call(run_dir, "smartctl", 1, 0) for _ in range(3)] == [
        True,
        True,
        True,
    ], "a zero bound is unbounded"
    counts = json.loads((wrapper.audit_dir(run_dir) / "calls.json").read_text())
    assert counts == {"smartctl/0": 2, "smartctl/1": 3}, counts
    assert not (wrapper.audit_dir(run_dir) / "calls.tmp").exists(), (
        "the counter is replaced atomically"
    )


@pytest.mark.parametrize(
    "stored, expected",
    [("", True), ("{not json", False), (json.dumps({"ethtool/0": "many"}), False)],
    ids=["empty-file", "unparsable", "non-integer"],
)
def test_consume_call_fails_towards_passthrough_on_an_unreadable_counter(
    tmp_path: Path, stored: str, expected: bool
) -> None:
    run_dir = str(tmp_path / "run")
    audit = wrapper.audit_dir(run_dir)
    audit.mkdir(parents=True)
    (audit / "calls.json").write_text(stored)
    assert wrapper.consume_call(run_dir, "ethtool", 0, 5) is expected, (
        "an empty counter is zero; a counter that cannot be kept is no permission"
    )


def test_consume_call_and_journal_survive_an_unwritable_audit_path(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "audit").write_text("not a directory")
    assert wrapper.consume_call(str(run_dir), "smartctl", 0, 0) is False, (
        "an uncountable rule is not applied"
    )
    assert wrapper.journal(str(run_dir), "smartctl", {"at": NOW}) is None, (
        "a journal that cannot be written never fails the collector's call"
    )
    assert (run_dir / "audit").read_text() == "not a directory", "nothing was written"


def test_journal_appends_one_sorted_json_line_per_call(tmp_path: Path) -> None:
    run_dir = str(tmp_path / "run")
    wrapper.journal(run_dir, "ethtool", {"b": 2, "a": 1})
    wrapper.journal(run_dir, "ethtool", {"c": 3})
    lines = (wrapper.audit_dir(run_dir) / "ethtool.jsonl").read_text().splitlines()
    assert lines == ['{"a": 1, "b": 2}', '{"c": 3}'], lines


# --------------------------------------------------------------------------- #
# serve()
# --------------------------------------------------------------------------- #
def test_serve_without_a_plan_passes_the_real_answer_through_and_journals_it(
    tmp_path: Path, real: RealBinary
) -> None:
    real.answers(HEALTHY + "\n", stderr="note\n")
    run_dir = tmp_path / "run"
    served = wrapper.serve("smartctl", HEALTH, REAL, str(run_dir), now=NOW)
    assert served == (0, HEALTHY + "\n", "note\n"), served
    assert real.calls == [
        {
            "argv": [REAL, *HEALTH],
            "capture_output": True,
            "text": True,
            "errors": "replace",
            "timeout": wrapper.REAL_TIMEOUT_SECONDS,
            "check": False,
        }
    ], "the real binary gets the collector's argv unchanged, bounded"
    assert journaled(run_dir, "smartctl") == [
        {
            "at": NOW,
            "tool": "smartctl",
            "query": "health",
            "argv": HEALTH,
            "real": REAL,
            "real_returncode": 0,
            "real_sha256": digest(HEALTHY + "\n"),
            "served_returncode": 0,
            "served_sha256": digest(HEALTHY + "\n"),
            "applied_rules": [],
            "unchanged_reason": ["no readable plan: FileNotFoundError"],
            "proof_scope": "live-replay",
            "physical_fault_injected": False,
        }
    ], "the journal names why nothing was replaced"


@pytest.mark.parametrize(
    "not_after, rule, reason",
    [
        (
            NOW - 1,
            {"query": "health", "action": "exit", "exit_code": 2},
            "plan deadline passed; passed through",
        ),
        (
            NOW + 600,
            {"query": "health", "action": "exit", "exit_code": 2, "device": "/dev/sdb"},
            "no rule for this call; passed through",
        ),
    ],
    ids=["deadline", "other-device"],
)
def test_serve_passes_through_past_the_deadline_or_without_a_matching_rule(
    tmp_path: Path,
    real: RealBinary,
    not_after: float,
    rule: dict[str, Any],
    reason: str,
) -> None:
    real.answers(HEALTHY)
    run_dir = tmp_path / "run"
    install_plan(run_dir, "smartctl", rule, not_after=not_after)
    assert wrapper.serve("smartctl", HEALTH, REAL, str(run_dir), now=NOW) == (
        0,
        HEALTHY,
        "",
    ), "the real answer is served"
    entry = journaled(run_dir, "smartctl")[-1]
    assert (entry["applied_rules"], entry["unchanged_reason"]) == ([], [reason]), entry
    assert not (wrapper.audit_dir(str(run_dir)) / "calls.json").exists(), (
        "no rule was counted"
    )


def test_serve_answers_a_whole_output_rule_without_running_the_real_binary(
    tmp_path: Path, real: RealBinary
) -> None:
    run_dir = tmp_path / "run"
    install_plan(
        run_dir,
        "smartctl",
        {"query": "scan", "action": "exit", "exit_code": 3, "stderr": "scan down\n"},
        {"query": "scan", "action": "replace_output", "stdout": "never\n"},
    )
    served = wrapper.serve("smartctl", SCAN, REAL, str(run_dir), now=NOW)
    assert served == (3, "", "scan down\n"), served
    assert real.calls == [], "a whole-output rule never needs the real binary"
    entry = journaled(run_dir, "smartctl")[-1]
    assert (entry["applied_rules"], entry["real_returncode"]) == ([0], None), (
        "the first whole-output rule answers alone"
    )
    assert entry["unchanged_reason"] == [], entry
    assert (entry["real_sha256"], entry["served_sha256"]) == (None, digest("")), (
        "no real reading, so no real digest; the served one is the empty stdout"
    )


def test_serve_never_fabricates_an_edit_without_a_real_binary(
    tmp_path: Path, real: RealBinary
) -> None:
    run_dir = tmp_path / "run"
    install_plan(
        run_dir,
        "ethtool",
        {
            "query": "stats",
            "action": "set_counter",
            "counter": "rx_drops",
            "value": "9",
        },
    )
    served = wrapper.serve("ethtool", STATS, None, str(run_dir), now=NOW)
    assert served == (127, "", "ethtool: command not found\n"), served
    entry = journaled(run_dir, "ethtool")[-1]
    assert entry["unchanged_reason"] == [
        "rule 0: no real ethtool to forward to; not applied"
    ], entry
    assert (entry["real"], entry["real_returncode"]) == (None, None), entry


def test_serve_runs_the_real_binary_once_for_several_edits(
    tmp_path: Path, real: RealBinary
) -> None:
    real.answers(NIC_STATS)
    run_dir = tmp_path / "run"
    install_plan(
        run_dir,
        "ethtool",
        {
            "query": "stats",
            "action": "set_counter",
            "counter": "rx_drops",
            "value": "5000",
        },
        {
            "query": "stats",
            "interface": "rdmap79s0",
            "action": "delete_counter",
            "counter": "tx_drops",
        },
        {"query": "stats", "interface": "eth9", "action": "exit", "exit_code": 1},
    )
    served = wrapper.serve("ethtool", STATS, "/usr/sbin/ethtool", str(run_dir), now=NOW)
    assert served == (0, "NIC statistics:\n     rx_drops: 5000\n", ""), served
    assert len(real.calls) == 1, "both edits act on one real reading"
    entry = journaled(run_dir, "ethtool")[-1]
    assert (entry["applied_rules"], entry["real_returncode"]) == ([0, 1], 0), entry
    assert (entry["real_sha256"], entry["served_sha256"]) == (
        digest(NIC_STATS),
        digest("NIC statistics:\n     rx_drops: 5000\n"),
    ), "the journal tells the real reading from the edited one the product saw"


def test_serve_passes_a_failed_real_query_through_unedited(
    tmp_path: Path, real: RealBinary
) -> None:
    real.answers("", returncode=15, stderr="Unable to determine the device handle\n")
    run_dir = tmp_path / "run"
    install_plan(
        run_dir,
        "smartctl",
        {
            "query": "health",
            "action": "set_json",
            "field": "smart_status.passed",
            "value": False,
        },
    )
    served = wrapper.serve("smartctl", HEALTH, REAL, str(run_dir), now=NOW)
    assert served == (15, "", "Unable to determine the device handle\n"), served
    entry = journaled(run_dir, "smartctl")[-1]
    assert entry["unchanged_reason"] == [
        "rule 0: the real query failed; passed through as it failed"
    ], entry
    assert entry["real_returncode"] == 15 and entry["applied_rules"] == [], entry
    assert not (wrapper.audit_dir(str(run_dir)) / "calls.json").exists(), (
        "a rule that did not act is not counted against its bound"
    )


def test_serve_delays_a_passthrough_by_the_planned_seconds(
    tmp_path: Path, real: RealBinary, monkeypatch: pytest.MonkeyPatch
) -> None:
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    real.answers(NIC_STATS)
    run_dir = tmp_path / "run"
    install_plan(
        run_dir,
        "ethtool",
        {"query": "stats", "action": "passthrough", "delay_seconds": 7},
    )
    served = wrapper.serve("ethtool", STATS, "/usr/sbin/ethtool", str(run_dir), now=NOW)
    assert served == (0, NIC_STATS, ""), served
    assert slept == [7], "the reviewed delay is applied once"
    assert journaled(run_dir, "ethtool")[-1]["applied_rules"] == [0], (
        "a delayed passthrough counts as applied"
    )


def test_serve_a_bounded_rule_falls_back_to_the_real_binary_once_spent(
    tmp_path: Path, real: RealBinary
) -> None:
    real.answers("/dev/nvme0 -d nvme\n")
    run_dir = tmp_path / "run"
    install_plan(
        run_dir,
        "smartctl",
        {"query": "scan", "action": "exit", "exit_code": 4, "max_calls": 1},
    )
    first = wrapper.serve("smartctl", SCAN, REAL, str(run_dir), now=NOW)
    second = wrapper.serve("smartctl", SCAN, REAL, str(run_dir), now=NOW + 1)
    assert (first, second) == ((4, "", ""), (0, "/dev/nvme0 -d nvme\n", "")), (
        "the bound applies the rule once, then the real scan is served"
    )
    assert len(real.calls) == 1, "only the spent call reached the real binary"
    assert journaled(run_dir, "smartctl")[-1]["unchanged_reason"] == [
        "rule 0 exhausted or uncountable; not applied"
    ], "the journal says the bound was spent"


def test_serve_journals_an_edit_the_real_output_cannot_take(
    tmp_path: Path, real: RealBinary
) -> None:
    real.answers("SMART overall-health self-assessment test result: PASSED\n")
    run_dir = tmp_path / "run"
    install_plan(
        run_dir,
        "smartctl",
        {"query": "health", "action": "delete_json", "field": "smart_status"},
    )
    served = wrapper.serve("smartctl", HEALTH, REAL, str(run_dir), now=NOW)
    assert served == (
        0,
        "SMART overall-health self-assessment test result: PASSED\n",
        "",
    ), "text output is served untouched"
    entry = journaled(run_dir, "smartctl")[-1]
    assert entry["unchanged_reason"] == ["rule 0: the real output is not JSON"], entry
    assert entry["applied_rules"] == [], entry


def test_serve_journals_a_column_rewrite_the_row_cannot_take_as_not_applied(
    tmp_path: Path, real: RealBinary
) -> None:
    rows = f"{UUID_A}\n{UUID_B}, 97\n"
    real.answers(rows)
    run_dir = tmp_path / "run"
    install_plan(run_dir, "nvidia-smi", SET_IDLE)
    argv = list(wrapper.TOOLS["nvidia-smi"]["queries"]["utilization"]["argv"])
    served = wrapper.serve("nvidia-smi", argv, NVIDIA_SMI, str(run_dir), now=NOW)
    assert served == (0, rows, ""), "the product is served the real reading"
    entry = journaled(run_dir, "nvidia-smi")[-1]
    assert (entry["query"], entry["applied_rules"]) == ("utilization", []), (
        "a rewrite that did not happen is never journaled as applied"
    )
    assert entry["unchanged_reason"] == [
        f"rule 0: the row for {UUID_A} has no utilization.gpu column"
    ], entry
    assert entry["real_sha256"] == entry["served_sha256"] == digest(rows), (
        "the journal shows the served output is the real one"
    )


@pytest.mark.parametrize(
    "error, expected",
    [
        (
            subprocess.TimeoutExpired([REAL], wrapper.REAL_TIMEOUT_SECONDS),
            (124, "", "smartctl: timed out\n"),
        ),
        (
            PermissionError(13, "Permission denied"),
            (126, "", "smartctl: [Errno 13] Permission denied\n"),
        ),
    ],
    ids=["timeout", "not-executable"],
)
def test_serve_maps_a_real_binary_that_hangs_or_cannot_run(
    tmp_path: Path, real: RealBinary, error: BaseException, expected: tuple
) -> None:
    real.error = error
    run_dir = tmp_path / "run"
    assert wrapper.serve("smartctl", SCAN, REAL, str(run_dir), now=NOW) == expected, (
        "the collector sees the shell's timeout / cannot-execute codes"
    )
    assert journaled(run_dir, "smartctl")[-1]["real_returncode"] == expected[0], (
        "the journal keeps the mapped code"
    )


# --------------------------------------------------------------------------- #
# main()
# --------------------------------------------------------------------------- #
class Exec(Exception):
    """Raised by the ``os.execv`` stand-in: execv never returns."""


def configure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tool: str = "smartctl"
) -> tuple[Path, Path, list[tuple[str, list[str]]]]:
    """A run dir whose ``bin/`` shim is first on PATH, a real tool after it."""

    run_dir = tmp_path / "run"
    shim_dir = run_dir / "bin"
    shim_dir.mkdir(parents=True)
    (shim_dir / tool).write_text("#!/bin/sh\n")
    (shim_dir / tool).chmod(0o755)
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / tool).write_text("#!/bin/sh\n")
    (real_dir / tool).chmod(0o755)
    monkeypatch.setenv(wrapper.TOOL_ENV, tool)
    monkeypatch.setenv(wrapper.DIR_ENV, str(run_dir))
    monkeypatch.setenv("PATH", os.pathsep.join([str(shim_dir), str(real_dir)]))
    executed: list[tuple[str, list[str]]] = []

    def execv(path: str, argv: list[str]) -> None:
        executed.append((path, list(argv)))
        raise Exec(path)

    monkeypatch.setattr(os, "execv", execv)
    return run_dir, real_dir / tool, executed


@pytest.mark.parametrize(
    "tool, run_dir",
    [(None, "/run/gfrs-live/x"), ("journalctl", "/run/gfrs-live/x"), ("smartctl", "")],
    ids=["no-tool", "unknown-tool", "no-run-dir"],
)
def test_main_refuses_to_run_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    tool: str | None,
    run_dir: str,
) -> None:
    if tool is None:
        monkeypatch.delenv(wrapper.TOOL_ENV, raising=False)
    else:
        monkeypatch.setenv(wrapper.TOOL_ENV, tool)
    monkeypatch.setenv(wrapper.DIR_ENV, run_dir)
    assert wrapper.main(["--scan-open"]) == 125, (
        "an unconfigured wrapper never forwards"
    )
    assert capsys.readouterr().err == "host command replay wrapper: not configured\n", (
        "the refusal is named"
    )


def test_main_execs_the_real_binary_for_an_uncatalogued_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_dir, real_tool, executed = configure(monkeypatch, tmp_path)
    with pytest.raises(Exec):
        wrapper.main(["--version"])
    assert executed == [(str(real_tool), [str(real_tool), "--version"])], (
        "the shim's own directory is skipped and the argv is forwarded unchanged"
    )


def test_main_answers_a_missing_tool_for_an_uncatalogued_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _run_dir, real_tool, executed = configure(monkeypatch, tmp_path)
    real_tool.unlink()
    assert wrapper.main(["--version"]) == 127, "no real binary, no forwarding"
    assert capsys.readouterr().err == "smartctl: command not found\n", (
        "the collector sees the shell's missing-command answer"
    )
    assert executed == [], "nothing was executed"


def test_main_serves_a_catalogued_call_and_writes_its_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    run_dir, _real_tool, executed = configure(monkeypatch, tmp_path)
    install_plan(
        run_dir,
        "smartctl",
        {"query": "scan", "action": "exit", "exit_code": 3, "stderr": "scan down\n"},
        {
            "query": "health",
            "action": "replace_output",
            "stdout": '{"smart_status": {"passed": false}}\n',
        },
        not_after=time.time() + 600,
    )
    assert wrapper.main(list(SCAN)) == 3, "the planned exit status is returned"
    assert capsys.readouterr() == ("", "scan down\n"), "the planned stderr is written"
    assert wrapper.main(list(HEALTH)) == 0, "the replaced answer exits 0"
    assert capsys.readouterr() == ('{"smart_status": {"passed": false}}\n', ""), (
        "the reviewed stdout is served with nothing on stderr"
    )
    assert executed == [], "catalogued calls are served, never exec'd"
    assert [entry["query"] for entry in journaled(run_dir, "smartctl")] == [
        "scan",
        "health",
    ], "both catalogued calls are journaled"


def test_module_entrypoint_exits_with_mains_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.delenv(wrapper.TOOL_ENV, raising=False)
    monkeypatch.delenv(wrapper.DIR_ENV, raising=False)
    monkeypatch.setattr(sys, "argv", ["smartctl", "--scan-open"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(Path(wrapper.__file__)), run_name="__main__")
    assert stopped.value.code == 125, "the script exits with main()'s status"
    assert "not configured" in capsys.readouterr().err, "and says why"
