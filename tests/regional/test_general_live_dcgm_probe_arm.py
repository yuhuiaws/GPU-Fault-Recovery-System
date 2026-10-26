"""``arm`` of the DCGM field-replay host controller, in both of its modes.

Live 2026-10-26 14:12Z (HyperPod, release R13g, harness live17)
GF-GENERAL-LIVE-022 ``source_boot_change`` ended INCONCLUSIVE at ``arm`` --
"refusing to arm: the collector does not read the replay" (``arm:1092``) -- on
both clusters. b193 had made the variant arm *before* the repoint
(``arm_before_repoint``: the restarted collector's first scrape under the
replayed boot id is the plan's first read) while the probe's one guard still
assumed the repoint had happened; the unit tests mock the probe, so the first
live run was the first time the two met. The guard is now mode-aware: the
default mode requires the collector to read this run's replay;
``--before-repoint`` requires the collector to still read the original exporter
while this run's proxy unit is up, answers, is unarmed and has served no read.
Every refusal names the mode and the endpoint seen, carries the facts as its
detail (the host fixture keeps it in ``probe-error.json``), and nothing is
POSTed to the tool.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_dcgm_probe as probe
from tests.regional._dcgm_probe_fakes import (
    ENDPOINT,
    REAL,
    REPLAY,
    RUN,
    FakeSystemd,
    sandbox,
)

#: An earlier run's proxy (another port) the collector may still be pointed at.
FOREIGN_PROXY = "http://127.0.0.1:9413/metrics"
PROXY_UNIT = f"{probe.proxy_unit(RUN)}.service"


def arm_arguments(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {"run_id": RUN, "port": 9412, "before_repoint": False}
    values.update(overrides)
    return argparse.Namespace(**values)


def proxy_up(fake: FakeSystemd) -> None:
    """This run's proxy unit runs and its tool answers ``/status`` (0 reads)."""

    fake.unit(PROXY_UNIT, MainPID="300")
    fake.tool_answers = True


def refusal(fake: FakeSystemd, arguments: argparse.Namespace) -> probe.ProbeError:
    with pytest.raises(probe.ProbeError) as caught:
        probe.arm(arguments)
    assert fake.posts == [], "a refusal must not touch the tool"
    assert fake.emitted == [], "a refusal emits no receipt"
    return caught.value


# --------------------------------------------------------------------------- #
# after the repoint (the default): the collector must read this run's replay
# --------------------------------------------------------------------------- #
def test_after_the_repoint_the_collector_must_read_this_runs_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    proxy_up(fake)

    error = refusal(fake, arm_arguments())

    assert error.reason == "collector_not_repointed"
    assert str(error) == (
        "refusing to arm after the repoint: "
        f"the collector reads {REAL}, not the replay {REPLAY}"
    )
    assert error.detail == {
        "before_repoint": False,
        "effective_endpoint": REAL,
        "replay_endpoint": REPLAY,
    }


def test_after_the_repoint_arm_posts_and_records_its_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    proxy_up(fake)
    fake.collector("200", {ENDPOINT: REPLAY})

    probe.arm(arm_arguments())

    assert fake.posts == ["arm"]
    [receipt] = fake.emitted
    assert receipt["armed"] is True
    assert receipt["before_repoint"] is False
    assert receipt["effective_endpoint"] == REPLAY == receipt["replay_endpoint"]
    assert "reads_before_arm" not in receipt and "proxy_unit" not in receipt


# --------------------------------------------------------------------------- #
# before the repoint: the original exporter still read, this run's proxy idle
# --------------------------------------------------------------------------- #
def test_before_the_repoint_arms_while_the_collector_reads_the_original_exporter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    proxy_up(fake)

    probe.arm(arm_arguments(before_repoint=True))

    assert fake.posts == ["arm"], (
        "live 2026-10-26 14:12Z this exact state -- proxy up, 0 reads, collector "
        "on the real exporter -- was refused as 'the collector does not read the "
        "replay'"
    )
    [receipt] = fake.emitted
    assert receipt["armed"] is True
    assert receipt["before_repoint"] is True
    assert receipt["effective_endpoint"] == REAL == receipt["original_endpoint"]
    assert receipt["replay_endpoint"] == REPLAY
    assert receipt["proxy_unit"] == PROXY_UNIT
    assert receipt["proxy_unit_state"] == "active"
    assert receipt["reads_before_arm"] == 0


@pytest.mark.parametrize(
    ("seen", "note"), [(REPLAY, " (this run's replay)"), (FOREIGN_PROXY, "")]
)
def test_before_the_repoint_refuses_a_collector_that_left_the_original_exporter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, seen: str, note: str
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    proxy_up(fake)
    fake.collector("200", {ENDPOINT: seen})

    error = refusal(fake, arm_arguments(before_repoint=True))

    assert error.reason == "collector_already_repointed"
    assert str(error) == (
        "refusing to arm before the repoint: "
        f"the collector reads {seen}{note}, not the original exporter {REAL}"
    )
    assert error.detail["before_repoint"] is True
    assert error.detail["effective_endpoint"] == seen
    assert error.detail["original_endpoint"] == REAL
    assert error.detail["reads_before_arm"] == 0


def test_before_the_repoint_requires_this_runs_proxy_unit_to_be_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    fake.tool_answers = True  # a tool answers, but no unit of this run is loaded

    error = refusal(fake, arm_arguments(before_repoint=True))

    assert error.reason == "proxy_down"
    assert str(error) == (
        f"refusing to arm before the repoint: this run's proxy unit {PROXY_UNIT} "
        "is inactive"
    )
    assert error.detail["proxy_unit"] == PROXY_UNIT
    assert error.detail["proxy_unit_state"] == "inactive"
    assert error.detail["effective_endpoint"] == REAL


def test_before_the_repoint_requires_the_proxy_to_answer_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    fake.unit(PROXY_UNIT, MainPID="300")  # the unit is active, the tool silent

    error = refusal(fake, arm_arguments(before_repoint=True))

    assert error.reason == "no_status"
    assert str(error).startswith(
        "refusing to arm before the repoint: this run's proxy on 9412 gave no "
        "status: loopback GET failed"
    ), str(error)
    assert "reads_before_arm" not in error.detail


@pytest.mark.parametrize(
    "status", [{"armed": False, "reads": 2}, {"armed": True, "reads": 0}]
)
def test_before_the_repoint_requires_an_unarmed_proxy_nobody_has_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: dict[str, Any]
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    proxy_up(fake)
    monkeypatch.setattr(probe, "tool_status", lambda port: dict(status))

    error = refusal(fake, arm_arguments(before_repoint=True))

    assert error.reason == "proxy_read_before_arm"
    assert str(error) == (
        "refusing to arm before the repoint: the proxy on port 9412 reports "
        f"armed={status['armed']} after {status['reads']} reads; nothing may read "
        "it before the repoint"
    )
    assert error.detail["reads_before_arm"] == status["reads"]


# --------------------------------------------------------------------------- #
# the command line and the failure trail
# --------------------------------------------------------------------------- #
def test_the_flag_is_an_arm_argument_and_a_refusal_travels_with_its_facts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    proxy_up(fake)
    fake.collector("200", {ENDPOINT: REPLAY})

    code = probe.main(["arm", "--run-id", RUN, "--port", "9412", "--before-repoint"])

    assert code == 1
    [failure] = fake.emitted
    assert failure["command"] == "arm"
    assert failure["error_reason"] == "collector_already_repointed"
    assert failure["error_site"].startswith("arm_preconditions:"), failure["error_site"]
    assert failure["error_detail"]["effective_endpoint"] == REPLAY
    assert failure["error_detail"]["before_repoint"] is True
    assert fake.posts == []
    default = probe.parser().parse_args(["arm", "--run-id", RUN])
    assert default.before_repoint is False and default.port == 9412
