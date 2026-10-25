"""The DCGM replay host controller's repoint, close and leftover contract.

Live 2026-10-25 (plain EKS, release 6a95b90fcd82): GF-GENERAL-LIVE-019
corrected_ecc / volatile_dbe failed inside ``repoint-collector`` after it had
armed the deadman, written the drop-in and restarted the collector, and the
run's cleanup then left the drop-in, the proxy unit and the run directory on
the node. These tests run the probe over a recorded fake systemd
(``_dcgm_probe_fakes``) and pin: the environment of a restarted collector is
read only once it has exec'd; ``close-run`` undoes every object whatever step
failed and names what it cannot; an earlier run's leftovers are reclaimed only
when its deadman is no longer pending, and foreign objects never.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

import pytest

from scripts.e2e.regional.probes import general_live_dcgm_probe as probe
from tests.regional._dcgm_probe_fakes import (
    DIGEST,
    ENDPOINT,
    FIRES,
    FOREIGN_DROPIN,
    METRICS,
    REAL,
    REPLAY,
    RUN,
    FakeSystemd,
    leftover_run,
    restored_on_restart,
    sandbox,
    stuck_on_restart,
)

SERVICE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"


def repoint_arguments(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "run_id": RUN,
        "port": 9412,
        "restore_seconds": 1800,
        "inventory_wrapper": False,
        "boot_id": "",
        "unknown_expected_count": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _execs_after(
    fake: FakeSystemd, monkeypatch: pytest.MonkeyPatch, reads: int, env: dict
):
    """The restarted collector is systemd's pre-exec child for ``reads`` reads."""

    seen = {"pre_exec_reads": 0}

    def environment(pid: str) -> dict[str, str]:
        if pid in fake.execed:
            return dict(fake.environ.get(pid, {}))
        seen["pre_exec_reads"] += 1
        if seen["pre_exec_reads"] > reads:
            fake.collector(pid, env)
            return dict(env)
        return {}

    monkeypatch.setattr(probe, "process_environment", environment)
    fake.on_restart = lambda pid: fake.collector(pid, {}, execed=False)
    return seen


# --------------------------------------------------------------------------- #
# repoint-collector: the environment is read once the collector has exec'd
# --------------------------------------------------------------------------- #
def test_repoint_waits_for_the_restarted_collector_to_exec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    seen = _execs_after(fake, monkeypatch, 2, {ENDPOINT: REPLAY, "PATH": SERVICE_PATH})
    probe.repoint_collector(repoint_arguments())
    receipt = fake.emitted[-1]
    assert receipt["replay_endpoint"] == REPLAY
    assert receipt["process_environment"][ENDPOINT] == REPLAY
    assert seen["pre_exec_reads"] == 3, (
        "the pre-exec reads were waited out instead of being judged a failure"
    )
    # The deadman is armed before the drop-in is written and the unit restarted.
    deadman = fake.index("systemd-run", f"--unit={probe.deadman_unit(RUN)}")
    assert (
        deadman
        < fake.index("systemctl", "daemon-reload")
        < fake.index("systemctl", "restart", METRICS)
    )
    dropin = probe.dropin_path(RUN).read_text()
    assert f"EnvironmentFile=-{probe.override_env_path(RUN)}" in dropin
    assert probe.override_env_path(RUN).read_text().strip() == f"{ENDPOINT}={REPLAY}"


def test_repoint_fails_when_the_collector_never_execs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    _execs_after(fake, monkeypatch, 10**6, {})
    with pytest.raises(probe.ProbeError, match="did not exec within the settle window"):
        probe.repoint_collector(repoint_arguments())
    assert probe.time.slept >= probe.SETTLE_SECONDS - 1, (
        "the whole settle window was used"
    )
    # What the live run then faced: drop-in, deadman and override are on the node.
    assert probe.dropin_path(RUN).exists(), "the failed repoint leaves its drop-in"
    assert f"{probe.deadman_unit(RUN)}.timer" in fake.units, "the deadman stays armed"


def test_repoint_fails_when_the_execed_collector_reads_the_real_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    restored_on_restart(fake)
    with pytest.raises(probe.ProbeError, match="did not pick up the replay endpoint"):
        probe.repoint_collector(repoint_arguments())


# --------------------------------------------------------------------------- #
# close-run: every object of this run goes, whatever step failed before
# --------------------------------------------------------------------------- #
def test_close_run_undoes_a_repoint_that_failed_after_the_drop_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    leftover_run(fake)
    restored_on_restart(fake)
    probe.close_run(argparse.Namespace(run_id=RUN, port=9412))
    receipt = fake.emitted[-1]
    assert receipt["clean"] is True, receipt
    assert receipt["leftover"] == [] and receipt["errors"] == []
    assert receipt["disarm"]["disarmed"] is True and fake.posts == ["disarm"]
    assert not probe.dropin_path(RUN).exists(), "the drop-in is removed"
    assert not probe.run_dir(RUN).exists(), "the run directory is removed"
    assert f"{probe.proxy_unit(RUN)}.service" not in fake.units, "the proxy is stopped"
    assert f"{probe.deadman_unit(RUN)}.timer" not in fake.units, (
        "the deadman is disarmed"
    )
    assert receipt["restore"]["restored_endpoint"] == REAL
    assert receipt["deadman_kept"] is False
    # Restore before stop: the collector reads the real endpoint before the tool goes.
    assert fake.index("systemctl", "restart", METRICS) < fake.index(
        "systemctl", "stop", f"{probe.proxy_unit(RUN)}.service"
    )


def test_close_run_keeps_the_deadman_and_names_every_leftover_when_a_step_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    leftover_run(fake)
    stuck_on_restart(fake)
    probe.close_run(argparse.Namespace(run_id=RUN, port=9412))
    receipt = fake.emitted[-1]
    assert receipt["clean"] is False
    assert receipt["errors"] == [
        "restore: ProbeError: the collector endpoint is not the original after the restore",
        "stop: ProbeError: refusing to stop the replay while the collector reads it",
    ]
    assert receipt["deadman_kept"] is True
    assert f"{probe.deadman_unit(RUN)}.timer" in fake.units, (
        "the deadman stays armed while anything is left"
    )
    leftover = receipt["leftover"]
    assert leftover == [
        f"unit {probe.proxy_unit(RUN)}.service (active)",
        f"run dir {probe.run_dir(RUN)} (run {RUN})",
        f"deadman {probe.deadman_unit(RUN)}.timer (active; fires {FIRES})",
    ], leftover
    assert not probe.dropin_path(RUN).exists(), "the drop-in itself was removed"


def test_close_run_with_nothing_deployed_is_clean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    probe.close_run(argparse.Namespace(run_id=RUN, port=9412))
    receipt = fake.emitted[-1]
    assert receipt["clean"] is True and receipt["leftover"] == []
    assert receipt["disarm"] == {"disarmed": False, "detail": "no replay tool answered"}
    assert fake.index("systemctl", "show", METRICS) >= 0
    assert all(c[:2] != ["systemctl", "restart"] for c in fake.commands), (
        "no drop-in, no restart of the collector"
    )


# --------------------------------------------------------------------------- #
# residue and cleanup-leftover: name everything, reclaim only expired own runs
# --------------------------------------------------------------------------- #
def test_residue_groups_this_fixtures_objects_per_run_and_names_foreign_ones(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    leftover_run(fake)
    (probe.DROPIN_DIR / FOREIGN_DROPIN).write_text("[Service]\n")
    (probe.STATE_ROOT / "not-a-digest").mkdir()
    residue = probe.residue(9412)
    assert set(residue["runs"]) == {DIGEST}
    state = residue["runs"][DIGEST]
    assert state["run_id"] == RUN
    assert state["deadman_pending"] is True and state["reclaimable"] is False
    assert state["objects"] == [
        f"drop-in {probe.dropin_path(RUN)}",
        f"unit {probe.proxy_unit(RUN)}.service (active)",
        f"run dir {probe.run_dir(RUN)} (run {RUN})",
        f"deadman {probe.deadman_unit(RUN)}.timer (active; fires {FIRES})",
    ]
    assert residue["foreign_dropins"] == [FOREIGN_DROPIN]
    assert residue["foreign_run_dirs"] == ["not-a-digest"]
    assert residue["port_answers"] is True and residue["port"] == 9412
    # the flat lists older readers use are still there
    assert residue["metrics_dropins"] == sorted(
        [FOREIGN_DROPIN, probe.dropin_path(RUN).name]
    )
    assert f"{probe.proxy_unit(RUN)}.service" in residue["units"]


def test_residue_marks_a_run_whose_deadman_elapsed_as_reclaimable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    leftover_run(fake, deadman=False)
    state = probe.residue(9412)["runs"][DIGEST]
    assert state["deadman_pending"] is False and state["reclaimable"] is True
    # A restore service still retrying is pending, not reclaimable.
    fake.unit(
        f"{probe.deadman_unit(RUN)}.service",
        ActiveState="activating",
        SubState="auto-restart",
    )
    state = probe.residue(9412)["runs"][DIGEST]
    assert state["deadman_pending"] is True and state["reclaimable"] is False
    assert state["objects"][-1] == (
        f"deadman {probe.deadman_unit(RUN)}.service (activating/auto-restart)"
    )


def test_cleanup_leftover_leaves_a_pending_deadman_to_fire(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    leftover_run(fake)
    probe.cleanup_leftover(
        argparse.Namespace(run_id="gl019-volatile-dbe-a3-00000000", digest=DIGEST)
    )
    receipt = fake.emitted[-1]
    assert receipt["reclaimed"] is False and receipt["clean"] is False
    assert "deadman restore is still pending" in receipt["reason"]
    assert len(receipt["leftover"]) == 4
    assert probe.dropin_path(RUN).exists(), "nothing of a pending run is touched"
    assert f"{probe.proxy_unit(RUN)}.service" in fake.units
    assert all(c[:2] != ["systemctl", "restart"] for c in fake.commands), (
        "the collector is not restarted for a pending run"
    )


def test_cleanup_leftover_reclaims_an_expired_run_and_never_a_foreign_object(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    leftover_run(fake, deadman=False)
    restored_on_restart(fake)
    foreign = probe.DROPIN_DIR / FOREIGN_DROPIN
    foreign.write_text("[Service]\n")
    probe.cleanup_leftover(
        argparse.Namespace(run_id="gl019-volatile-dbe-a3-00000000", digest=DIGEST)
    )
    receipt = fake.emitted[-1]
    assert receipt["reclaimed"] is True and receipt["clean"] is True, receipt
    assert receipt["run_id"] == RUN and receipt["leftover"] == []
    assert not probe.dropin_path(RUN).exists() and not probe.run_dir(RUN).exists()
    assert f"{probe.proxy_unit(RUN)}.service" not in fake.units
    assert foreign.exists(), "a drop-in that is not this fixture's is left in place"
    assert probe.residue(9412)["runs"] == {}


def test_cleanup_leftover_reports_a_reclaim_that_could_not_finish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    leftover_run(fake, deadman=False)
    stuck_on_restart(fake)
    probe.cleanup_leftover(
        argparse.Namespace(run_id="gl019-volatile-dbe-a3-00000000", digest=DIGEST)
    )
    receipt = fake.emitted[-1]
    assert receipt["reclaimed"] is True and receipt["clean"] is False
    assert receipt["errors"][0].startswith("restore: ProbeError: "), receipt["errors"]
    assert receipt["leftover"][0] == f"unit {probe.proxy_unit(RUN)}.service (active)"


def test_cleanup_leftover_rejects_an_unsafe_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sandbox(monkeypatch, tmp_path)
    with pytest.raises(probe.ProbeError, match="unsafe run digest"):
        probe.cleanup_leftover(argparse.Namespace(run_id=RUN, digest="../etc"))


# --------------------------------------------------------------------------- #
# start refusals inside the probe, the run marker, the failure trail
# --------------------------------------------------------------------------- #
def test_deploy_proxy_refuses_a_port_a_tool_still_answers_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    root = probe.run_dir(RUN)
    root.mkdir(parents=True)
    (root / probe.FIELD_TOOL).write_text("# tool\n")
    (root / probe.PLAN_NAME).write_text("{}")
    fake.tool_answers = True
    with pytest.raises(probe.ProbeError, match="already answers on the port"):
        probe.deploy_proxy(
            argparse.Namespace(
                run_id=RUN,
                tool=probe.FIELD_TOOL,
                profile="memory",
                gpu_uuid="",
                max_seconds=1800,
                port=9412,
            )
        )
    assert all(c[0] != "systemd-run" for c in fake.commands), "no second tool started"


def test_install_source_marks_the_run_directory_with_its_run_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sandbox(monkeypatch, tmp_path)
    content = b'{"profile": "memory", "steps": []}'
    arguments = argparse.Namespace(
        run_id=RUN,
        name="plan.json",
        sha256=hashlib.sha256(content).hexdigest(),
        b64=base64.b64encode(content).decode(),
    )
    probe.install_source(arguments)
    marker = json.loads((probe.run_dir(RUN) / probe.RUN_MARKER).read_text())
    assert marker["run_id"] == RUN and marker["digest"] == DIGEST
    assert probe.recorded_run_id(DIGEST) == RUN
    with pytest.raises(probe.ProbeError, match="already installed for this run"):
        probe.install_source(arguments)


def test_main_reports_the_class_a_message_digest_and_the_site(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = sandbox(monkeypatch, tmp_path)
    assert probe.main(["status", "--run-id", RUN, "--port", "9412"]) == 1
    failure = fake.emitted[-1]
    assert failure["error_kind"] == "probe" and failure["error_class"] == "ProbeError"
    assert failure["command"] == "status"
    assert re.fullmatch(r"[0-9a-f]{16}", failure["error_code"]), failure
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*:[1-9][0-9]*", failure["error_site"]), (
        failure
    )
    assert failure["error_site"].startswith("tool_status:"), failure["error_site"]
