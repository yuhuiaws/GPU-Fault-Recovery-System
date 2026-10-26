"""GF-GENERAL-LIVE-022's XID-join variants deploy their ``pcie_xid`` replay.

Live 2026-10-26 (plain EKS, both lanes, release R13g) ``same_gpu_xid``,
``wrong_gpu_xid`` and ``expired_xid`` ended INCONCLUSIVE before anything was
replayed: ``harness fixture failed: HostProbeError: dcgm deploy-proxy: host
probe failed (exit 1; error ProbeError ae2c099c21f98eb7 at tool_command:800);
output withheld``. The host controller accepted only ``[a-z]{3,16}`` as the
field tool's ``--profile``, and these three plans are the only ones on the
``pcie_xid`` profile -- the one underscored name in
``tools/gpu_dcgm_field_replay.py``; the neighbours' ``pcie`` / ``composite``
deployed. These tests pin the three gates a profile name passes (the tool's
list, the controller's shape, the runner's plan builder) against each other,
and the trail a refused deploy leaves: a fixed reason token in the exception
text and the redacted message in the case's ``probe-error.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_gpu_links as links
from scripts.e2e.regional import general_live_gpu_replay as replay
from scripts.e2e.regional import host_probe_fixture as host
from scripts.e2e.regional.general_live_fixture import VariantInconclusive
from scripts.e2e.regional.probes import general_live_dcgm_probe as probe
from tests.regional._dcgm_probe_fakes import RUN, sandbox
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    dcgm_probe,
    gpu_cpu_python,
    gpu_incident,
    gpu_workflow,
)
from tests.regional._host_probe_support import ProbeApi, host_probe
from tools import gpu_dcgm_field_replay as tool

#: The live refusal: the first 16 hex of the controller's sentence's sha256,
#: raised at ``tool_command``; the three result.json files carry the digest.
LIVE_DIGEST = "ae2c099c21f98eb7"
LIVE_SENTENCE = "the field tool needs its plan's --profile"
XID_VARIANTS = ("same_gpu_xid", "wrong_gpu_xid", "expired_xid")
G1 = "GPU-11111111-1111-4111-8111-111111111111"
SPECS = {
    spec.variant: spec for spec in links.VARIANTS if spec.case_id == links.CASE_022
}


# --------------------------------------------------------------------------- #
# the controller on the node
# --------------------------------------------------------------------------- #
def _deploy_arguments(profile: str, **overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "run_id": RUN,
        "tool": probe.FIELD_TOOL,
        "profile": profile,
        "gpu_uuid": "",
        "max_seconds": 1800,
        "port": 9412,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _installed(root: Path, profile: str) -> None:
    """The run directory as ``install-source`` leaves it: tool and plan."""

    root.mkdir(parents=True, exist_ok=True)
    (root / probe.FIELD_TOOL).write_text("# tool\n")
    (root / probe.PLAN_NAME).write_text(json.dumps({"profile": profile, "steps": []}))


def test_the_live_digest_was_the_profile_refusal() -> None:
    assert hashlib.sha256(LIVE_SENTENCE.encode()).hexdigest()[:16] == LIVE_DIGEST
    assert probe.PROFILE_NAME.fullmatch("pcie_xid"), "the live profile is served"


@pytest.mark.parametrize("profile", sorted(tool.PROFILES))
def test_the_controller_serves_every_field_tool_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, profile: str
) -> None:
    sandbox(monkeypatch, tmp_path)
    root = probe.run_dir(RUN)
    _installed(root, profile)
    command = probe.tool_command(_deploy_arguments(profile), root)
    assert command[command.index("--profile") + 1] == profile
    assert probe.PROFILE_NAME.fullmatch(profile), profile


def test_deploy_proxy_serves_the_pcie_xid_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    _installed(probe.run_dir(RUN), "pcie_xid")
    probe.deploy_proxy(_deploy_arguments("pcie_xid"))
    receipt = fake.emitted[-1]
    assert receipt["tool"] == probe.FIELD_TOOL, receipt
    assert receipt["status"] == {"armed": False, "reads": 0}, receipt
    started = fake.commands[fake.index("systemd-run")]
    assert started[started.index("--profile") + 1] == "pcie_xid"
    assert started[started.index("--plan") + 1].endswith("/" + probe.PLAN_NAME), started


@pytest.mark.parametrize(
    "profile",
    ["", "Pcie_XID", "pcie-xid", "pcie__xid", "_pcie", "pcie_", "pcie xid", "p"],
)
def test_a_profile_outside_the_tool_shape_is_refused_with_a_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, profile: str
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    _installed(probe.run_dir(RUN), "pcie_xid")
    with pytest.raises(probe.ProbeError) as refused:
        probe.deploy_proxy(_deploy_arguments(profile))
    # The sentence is kept: the live digest stays decodable from the source.
    assert str(refused.value) == LIVE_SENTENCE
    assert refused.value.reason == "profile_shape"
    assert all(c[0] != "systemd-run" for c in fake.commands), "nothing started"


def test_main_reports_the_reason_and_the_message_for_the_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    _installed(probe.run_dir(RUN), "pcie_xid")
    argv = ["deploy-proxy", "--run-id", RUN, "--tool", probe.FIELD_TOOL]
    assert probe.main([*argv, "--profile", "Pcie"]) == 1
    failure = fake.emitted[-1]
    assert failure["error_kind"] == "probe" and failure["error_class"] == "ProbeError"
    assert failure["error_code"] == LIVE_DIGEST
    assert failure["error_site"].startswith("tool_command:"), failure["error_site"]
    assert failure["error_reason"] == "profile_shape"
    assert failure["error_message"] == LIVE_SENTENCE
    assert failure["command"] == "deploy-proxy"


def test_main_keeps_the_message_for_a_failure_without_a_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No tool answers the port: a loopback failure, not one of the controller's
    # named refusals. The message still rides along for probe-error.json.
    fake = sandbox(monkeypatch, tmp_path)
    assert probe.main(["status", "--run-id", RUN, "--port", "9412"]) == 1
    failure = fake.emitted[-1]
    assert failure["error_reason"] is None, failure
    assert failure["error_message"].startswith("loopback GET failed"), failure
    assert failure["error_site"].startswith("tool_status:"), failure["error_site"]


def test_deploy_refusals_name_their_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    with pytest.raises(probe.ProbeError) as missing:
        probe.deploy_proxy(_deploy_arguments("pcie_xid"))
    assert missing.value.reason == "source_missing"
    _installed(probe.run_dir(RUN), "pcie_xid")
    fake.tool_answers = True
    with pytest.raises(probe.ProbeError) as busy:
        probe.deploy_proxy(_deploy_arguments("pcie_xid"))
    assert busy.value.reason == "port_busy"


# --------------------------------------------------------------------------- #
# the host fixture: what the runner's exception text and case dir carry
# --------------------------------------------------------------------------- #
def _probe_failure(**overrides: Any) -> str:
    payload = probe.error_fields(
        probe.ProbeError(LIVE_SENTENCE, "profile_shape"), "deploy-proxy"
    )
    payload["error_site"] = "tool_command:800"
    payload.update(overrides)
    return json.dumps(payload) + "\n"


def _refused_deploy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stdout: str
) -> str:
    api = ProbeApi()
    monkeypatch.setattr(host, "run_fixture_command", api.run)
    fixture = host_probe(tmp_path)
    fixture.create()
    api.probe_returncode = 1
    api.probe_stdout = stdout
    with pytest.raises(host.HostProbeError) as caught:
        fixture.execute("deploy-proxy", "--tool", probe.FIELD_TOOL, "--profile", "x")
    return str(caught.value)


def test_host_fixture_names_the_reason_and_keeps_the_message_in_the_case_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    text = _refused_deploy(monkeypatch, tmp_path, _probe_failure())
    assert text == (
        f"host probe failed (exit 1; error ProbeError {LIVE_DIGEST} at "
        "tool_command:800; reason profile_shape; detail in probe-error.json); "
        "output withheld"
    )
    assert LIVE_SENTENCE not in text
    record = json.loads((tmp_path / "probe-error.json").read_text())
    assert record["message"] == LIVE_SENTENCE
    assert record["error_reason"] == "profile_shape"
    assert record["error_code"] == LIVE_DIGEST
    assert record["command"] == "deploy-proxy" and record["redacted"] is True


@pytest.mark.parametrize(
    "reason",
    ["Profile Shape", "/run/gfrs-live/x", LIVE_SENTENCE, "ab", "x" * 42, 7, None],
)
def test_host_fixture_drops_a_reason_outside_the_token_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reason: Any
) -> None:
    text = _refused_deploy(monkeypatch, tmp_path, _probe_failure(error_reason=reason))
    assert "; reason " not in text, text
    assert text.endswith("; detail in probe-error.json); output withheld"), text
    record = json.loads((tmp_path / "probe-error.json").read_text())
    assert record["error_reason"] is None, record


def test_host_fixture_shows_no_reason_for_other_error_kinds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = json.loads(_probe_failure())
    payload["error_kind"] = "collector_env_guard"
    text = _refused_deploy(monkeypatch, tmp_path, json.dumps(payload) + "\n")
    assert "; reason " not in text, text


# --------------------------------------------------------------------------- #
# the runner: the three variants and the plan builder
# --------------------------------------------------------------------------- #
def _activity(incidents: list, workflows: list = ()) -> dict[str, Any]:
    return {
        "incidents": incidents,
        "workflows": list(workflows),
        "batches": [{"observed_at": "t"}],
        "markers": [],
        "commands": [],
        "notifications": [],
        "metric_latest": [],
        "collector_statuses": [],
        "agent": None,
    }


def _composite_cpu() -> Any:
    comp = gpu_incident(
        "inc-c",
        f"b-{G1}-composite-PCIE_XID_LINK_FAILURE",
        action="DRAIN",
        workflow="wf-c",
    )
    wf = gpu_workflow(
        "wf-c", "inc-c", ["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"]
    )
    return gpu_cpu_python(
        activity=_activity([comp], [wf]),
        gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
    )


def _ctx(tmp_path: Path, variant: str, *, probe_: Any, cpu: Any) -> Any:
    spec = SPECS[variant]
    regional = FakeRegional(cpu_python=cpu)
    settings = make_settings(spec, node=regional.node_snapshot("")["name"])
    object.__setattr__(settings, "approvals", ("DCGM_XID_GAUGE",))
    ctx = make_context(
        spec, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    ctx.probes[f"dcgm@{ctx.settings.node}"] = probe_
    return ctx


@pytest.mark.parametrize("variant", XID_VARIANTS)
def test_the_xid_join_variants_deploy_the_pcie_xid_profile(
    tmp_path: Path, variant: str
) -> None:
    fake = dcgm_probe(audit=audit_records({f"{G1}/pcie_replay_total": 9.0}))
    ctx = _ctx(tmp_path, variant, probe_=fake, cpu=_composite_cpu())
    out = getattr(links, variant)(ctx)
    assert out["verdict"] in {"PASS", "FAIL"}, out
    deploys = [call for call in fake.calls if call[0] == "deploy-proxy"]
    assert len(deploys) == 1, fake.calls
    profile = deploys[0][deploys[0].index("--profile") + 1]
    assert profile == "pcie_xid"
    assert profile in tool.PROFILES and probe.PROFILE_NAME.fullmatch(profile)
    installed = json.loads((ctx.case_dir / "plan.json").read_text())
    assert installed["profile"] == profile
    # The controller the fake stands in for serves exactly this tool name.
    assert deploys[0][deploys[0].index("--tool") + 1] == probe.FIELD_TOOL
    verbs = [call[0] for call in fake.calls]
    assert verbs.index("install-source") < verbs.index("deploy-proxy")
    assert verbs.index("deploy-proxy") < verbs.index("repoint-collector")


@pytest.mark.parametrize("profile", ["pcie-xid", "Pcie", "xid", "pcie_xid_", ""])
def test_plan_refuses_a_profile_the_node_would_refuse(profile: str) -> None:
    # ``xid`` fits the shape but is not a tool profile; the rest fail the shape.
    with pytest.raises(VariantInconclusive, match="not a field-tool profile"):
        replay.plan(profile, replay.step(1))


def test_plan_serves_every_tool_profile() -> None:
    for profile in tool.PROFILES:
        assert replay.plan(profile, replay.step(1))["profile"] == profile
    assert replay.passthrough_plan()["profile"] in tool.PROFILES
