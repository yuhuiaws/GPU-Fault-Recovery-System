"""GF-GENERAL-LIVE-021's held fabric plans fit the replay tool's read bounds.

Live 2026-10-26 (HyperPod, the hpb lane's exclusive node, release R13g,
harness live17) ``before_window`` ended INCONCLUSIVE at ``deploy-proxy``:
``reason unit_down``, probe-error.json "the replay unit did not stay up". The
plan's held step asked 400 reads; ``tools/gpu_dcgm_field_replay.py`` serves at
most ``MAX_READS_PER_STEP`` (200) per step and refused the plan at its start
(``error: reads must be an integer in 1..200``, exit 2), so the transient unit
died before ``/status`` answered -- on any node; the ``nvlink`` plans (3/40
reads) deployed minutes earlier on the same one. These tests pin: the runner's
``plan()`` runs the tool's own validator before the node is touched; the held
steps' read bounds come from ``held_reads`` and fit; and a dead unit's journal
and exit result travel in the probe's ``error_detail`` into the case's
redacted ``probe-error.json``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_gpu_fabric as fab
from scripts.e2e.regional import general_live_gpu_replay as replay
from scripts.e2e.regional import host_probe_fixture as host
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.probes import general_live_dcgm_probe as probe
from tests.regional._dcgm_probe_fakes import RUN, FakeSystemd, sandbox
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    dcgm_probe,
    gpu_cpu_python,
)
from tests.regional._host_probe_support import ProbeApi, host_probe
from tools import gpu_dcgm_field_replay as tool

SPECS = {spec.variant: spec for spec in fab.VARIANTS}
GPUS = [
    "GPU-00000000-0000-4000-8000-000000000000",
    "GPU-11111111-1111-4111-8111-111111111111",
]
#: The fake survey's collector interval (``_general_live_gpu_fakes.survey``).
INTERVAL = 15
#: The tool's refusal, as the unit's journal showed it.
TOOL_REFUSAL = "gpu_dcgm_field_replay.py: error: reads must be an integer in 1..200"
NO_FINDING_VARIANTS = ("before_window", "ungraded_product", "monitor_restart")


def _live_plan(reads: int) -> dict[str, Any]:
    """The shape of the hpb plan.json: a baseline, then one held set step."""

    return {
        "profile": "fabric",
        "steps": [
            {"reads": 3, "ops": []},
            {
                "reads": reads,
                "ops": [
                    {
                        "op": "set",
                        "field": "fabric_manager_status",
                        "gpus": GPUS,
                        "value": 2,
                    }
                ],
                "until_advance": True,
            },
        ],
    }


# --------------------------------------------------------------------------- #
# the runner's plan builder runs the tool's validator
# --------------------------------------------------------------------------- #
def test_the_live_plan_is_what_the_tool_refused() -> None:
    with pytest.raises(ValueError, match="reads must be an integer in 1..200"):
        tool.validate_plan(_live_plan(400))
    assert tool.validate_plan(_live_plan(200))["steps"][1]["reads"] == 200


@pytest.mark.parametrize("reads", [400, 800, 201, 0])
def test_plan_refuses_a_held_step_over_the_tools_read_bound(reads: int) -> None:
    with pytest.raises(VariantInconclusive) as refused:
        replay.plan(
            "fabric",
            replay.baseline(),
            replay.step(reads, replay.put("fabric_manager_status", GPUS, 2), hold=True),
        )
    text = str(refused.value)
    assert "refused by the field tool" in text and "reads must be an integer" in text
    assert "nothing was deployed" in text


def test_plan_refuses_what_the_tool_refuses_beyond_reads() -> None:
    # Over the per-plan total; a field outside the profile's allow-list; a
    # counter set instead of added: each the tool's own sentence.
    with pytest.raises(VariantInconclusive, match="exceeds 400 collector reads"):
        replay.plan("fabric", *(replay.step(200) for _ in range(3)))
    with pytest.raises(VariantInconclusive, match="not in this profile's allow-list"):
        replay.plan("fabric", replay.step(1, replay.put("pcie_replay_total", GPUS, 5)))
    with pytest.raises(
        VariantInconclusive, match="'add' is not allowed by this profile"
    ):
        replay.plan("fabric", replay.step(1, replay.add("fabric_manager_status", GPUS)))
    with pytest.raises(VariantInconclusive, match="use add, not set"):
        replay.plan(
            "memory", replay.step(1, replay.put("ecc_sbe_volatile_total", GPUS, 5))
        )


def test_plan_keeps_a_plan_the_tool_accepts_verbatim() -> None:
    steps = (
        replay.baseline(),
        replay.step(40, replay.put("fabric_manager_status", GPUS, 2), hold=True),
    )
    built = replay.plan("fabric", *steps)
    assert built == {"profile": "fabric", "steps": list(steps)}
    assert tool.validate_plan(built)["profile"] == "fabric"


def test_held_reads_outlasts_the_hold_within_the_tools_bound() -> None:
    assert replay.held_reads(15, 270) == 40  # before_window: 18 reads held
    assert replay.held_reads(15, 360) == 52  # ungraded_product: 24 reads held
    assert replay.held_reads(15, 660) == 92  # monitor_restart: 44 reads held
    assert replay.held_reads(15, 1) == 6 and replay.held_reads(0, 30) == 64
    for interval, seconds in ((15, 270), (15, 660), (5, 480), (60, 2700)):
        reads = replay.held_reads(interval, seconds)
        assert reads <= tool.MAX_READS_PER_STEP, (interval, seconds, reads)
        assert reads > -(-seconds // interval), (interval, seconds, reads)


def test_held_reads_refuses_a_hold_the_tool_cannot_bound() -> None:
    # 2700 s at 5 s per read needs 1084 reads: refused, never clipped to 200.
    with pytest.raises(VariantInconclusive, match="at most 200 per step"):
        replay.held_reads(5, 2700)
    with pytest.raises(VariantInconclusive, match="needs 1084 reads"):
        replay.held_reads(5, 2700)


# --------------------------------------------------------------------------- #
# the three no-finding variants install plans the tool accepts
# --------------------------------------------------------------------------- #
def _ctx(tmp_path: Path, variant: str, *, probe_: Any, cpu: Any) -> Any:
    spec = SPECS[variant]
    regional = FakeRegional(cpu_python=cpu)
    settings = make_settings(spec, node=regional.node_snapshot("")["name"])
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    ctx.probes[f"dcgm@{ctx.settings.node}"] = probe_
    return ctx


def _quiet_cpu() -> Any:
    return gpu_cpu_python(
        activity={
            "incidents": [],
            "workflows": [],
            "batches": [],
            "markers": [],
            "commands": [],
            "notifications": [],
            "metric_latest": [],
            "collector_statuses": [],
            "agent": None,
        },
        gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
    )


def _expected_hold(variant: str) -> int:
    budget = min(SPECS[variant].max_minutes * 60, 45 * 60)
    if variant == "before_window":
        return min(fab.WINDOW - 30, budget)
    if variant == "ungraded_product":
        return min(fab.WINDOW + 60, budget)
    return 2 * min(fab.WINDOW - 60, budget) + fab.RESTART_SECONDS


@pytest.mark.parametrize("variant", NO_FINDING_VARIANTS)
def test_no_finding_variants_install_a_plan_the_tool_serves(
    tmp_path: Path, variant: str
) -> None:
    served = audit_records({f"{gpu}/fabric_manager_status": 2.0 for gpu in GPUS})
    fake = dcgm_probe(audit=served)
    ctx = _ctx(tmp_path, variant, probe_=fake, cpu=_quiet_cpu())
    out = getattr(fab, variant)(ctx)
    assert out["verdict"] == "PASS", out
    installed = json.loads((ctx.case_dir / "plan.json").read_text())
    validated = tool.validate_plan(installed)
    held = [s for s in validated["steps"] if s["until_advance"]]
    assert len(held) == 1 and held[0]["reads"] <= tool.MAX_READS_PER_STEP, installed
    assert held[0]["reads"] == replay.held_reads(INTERVAL, _expected_hold(variant))
    assert sum(s["reads"] for s in validated["steps"]) <= tool.MAX_TOTAL_READS
    deploys = [call for call in fake.calls if call[0] == "deploy-proxy"]
    assert len(deploys) == 1, fake.calls
    assert deploys[0][deploys[0].index("--profile") + 1] == "fabric"


# --------------------------------------------------------------------------- #
# the controller: a dead unit's journal travels with unit_down
# --------------------------------------------------------------------------- #
def _dying_unit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, journal_lines: list[str]
) -> FakeSystemd:
    """A sandbox whose ``systemd-run`` unit exits 2 at once (the tool's
    argparse refusal) and whose journal answers with ``journal_lines``."""

    fake = sandbox(monkeypatch, tmp_path)
    real_run = fake.run

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[0] == "journalctl":
            fake.commands.append(list(command))
            return subprocess.CompletedProcess(command, 0, "\n".join(journal_lines), "")
        completed = real_run(command, **kwargs)
        if command[0] == "systemd-run":
            unit = next(a.split("=", 1)[1] for a in command if a.startswith("--unit="))
            fake.units[f"{unit}.service"].update(
                ActiveState="failed",
                SubState="failed",
                Result="exit-code",
                ExecMainStatus="2",
                MainPID="0",
            )
            fake.tool_answers = False
        return completed

    monkeypatch.setattr(probe, "run", run)
    root = probe.run_dir(RUN)
    root.mkdir(parents=True, exist_ok=True)
    (root / probe.FIELD_TOOL).write_text("# tool\n")
    (root / probe.PLAN_NAME).write_text(json.dumps(_live_plan(400)))
    return fake


def _deploy_argv() -> list[str]:
    return [
        "deploy-proxy",
        "--run-id",
        RUN,
        "--tool",
        probe.FIELD_TOOL,
        "--profile",
        "fabric",
        "--max-seconds",
        "3000",
    ]


def test_unit_down_carries_the_units_journal_and_exit_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _dying_unit(
        monkeypatch,
        tmp_path,
        ["usage: gpu_dcgm_field_replay.py [-h] ...", TOOL_REFUSAL],
    )
    assert probe.main(_deploy_argv()) == 1
    failure = fake.emitted[-1]
    assert failure["error_reason"] == "unit_down", failure
    assert failure["error_message"] == "the replay unit did not stay up"
    detail = failure["error_detail"]
    unit = f"{probe.proxy_unit(RUN)}.service"
    assert detail["unit"] == unit
    assert detail["journal"] == [
        "usage: gpu_dcgm_field_replay.py [-h] ...",
        TOOL_REFUSAL,
    ]
    assert detail["service"]["Result"] == "exit-code"
    assert detail["service"]["ExecMainStatus"] == "2"
    read = fake.commands[fake.index("journalctl")]
    assert read[-2:] == ["-u", unit] and "--no-pager" in read and "40" in read


def test_journal_detail_is_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _dying_unit(
        monkeypatch, tmp_path, [f"line {i} " + "x" * 1000 for i in range(100)]
    )
    assert probe.main(_deploy_argv()) == 1
    journal = fake.emitted[-1]["error_detail"]["journal"]
    assert len(journal) == 40 and all(len(line) <= 300 for line in journal)
    assert journal[-1].startswith("line 99 "), journal[-1]


def test_no_status_carries_the_journal_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The unit stays active but the tool never answers /status within the
    # 30 s start window (the fake clock advances 0.5 s per poll).
    fake = sandbox(monkeypatch, tmp_path)
    real_run = fake.run

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[0] == "journalctl":
            return subprocess.CompletedProcess(
                command, 0, "Traceback (most recent)\n", ""
            )
        completed = real_run(command, **kwargs)
        if command[0] == "systemd-run":
            fake.tool_answers = False
        return completed

    monkeypatch.setattr(probe, "run", run)
    root = probe.run_dir(RUN)
    root.mkdir(parents=True, exist_ok=True)
    (root / probe.FIELD_TOOL).write_text("# tool\n")
    (root / probe.PLAN_NAME).write_text(json.dumps(_live_plan(40)))
    assert probe.main(_deploy_argv()) == 1
    failure = fake.emitted[-1]
    assert failure["error_reason"] == "no_status", failure
    assert failure["error_detail"]["journal"] == ["Traceback (most recent)"]
    assert failure["error_detail"]["service"]["ActiveState"] == "active"


def test_unit_state_asks_for_the_exit_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    probe.unit_state(probe.METRICS_UNIT)
    shown = fake.commands[fake.index("systemctl", "show")]
    assert "--property=Result" in shown and "--property=ExecMainStatus" in shown
    assert "--property=ActiveState" in shown and "--property=MainPID" in shown


def test_a_failure_without_detail_reports_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    assert probe.main(["status", "--run-id", RUN, "--port", "9412"]) == 1
    assert fake.emitted[-1]["error_detail"] is None


# --------------------------------------------------------------------------- #
# the host fixture: the journal lands, redacted, in probe-error.json
# --------------------------------------------------------------------------- #
def test_probe_error_json_keeps_the_redacted_journal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    unit = f"{probe.proxy_unit(RUN)}.service"
    detail = {
        "unit": unit,
        "service": {
            "ActiveState": "failed",
            "Result": "exit-code",
            "ExecMainStatus": "2",
        },
        "journal": ["token=abcdef0123456789 leaked into the log", TOOL_REFUSAL],
    }
    payload = probe.error_fields(
        probe.ProbeError("the replay unit did not stay up", "unit_down", detail),
        "deploy-proxy",
    )
    api = ProbeApi()
    monkeypatch.setattr(host, "run_fixture_command", api.run)
    fixture = host_probe(tmp_path)
    fixture.create()
    api.probe_returncode = 1
    api.probe_stdout = json.dumps(payload) + "\n"
    with pytest.raises(host.HostProbeError) as caught:
        fixture.execute(
            "deploy-proxy", "--tool", probe.FIELD_TOOL, "--profile", "fabric"
        )
    text = str(caught.value)
    assert "; reason unit_down; detail in probe-error.json); output withheld" in text
    assert "abcdef0123456789" not in text and "reads must be" not in text
    record = json.loads((tmp_path / "probe-error.json").read_text())
    assert record["error_reason"] == "unit_down"
    assert record["detail"]["unit"] == unit
    assert record["detail"]["service"]["ExecMainStatus"] == "2"
    assert record["detail"]["journal"][0] == "token=<redacted> leaked into the log"
    assert record["detail"]["journal"][1] == TOOL_REFUSAL
