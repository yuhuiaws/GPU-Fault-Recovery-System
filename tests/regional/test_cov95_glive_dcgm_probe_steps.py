"""DCGM field-replay host controller: the steps that change the node.

``install-source``, ``deploy-proxy``, ``repoint-collector``, the tool's
``arm`` / ``disarm`` / ``advance``, ``collector-restart``, the restore and
stop steps and the deadman's ``cleanup`` run in-process over the recorded
systemd sandbox (``_dcgm_probe_fakes``): what each refuses and why, what it
starts and writes, and what it leaves for the deadman. Nothing touches the
host.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_dcgm_probe as probe
from tests.regional._cov95_glive_dcgm import (
    BOOT_ID,
    UUID0,
    deploy_arguments,
    installed_sources,
    node,
    repoint_arguments,
)
from tests.regional._dcgm_probe_fakes import (
    ENDPOINT,
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

REPLAYED_BOOT = "0b1c0e3e-1a2b-4c3d-8e4f-0123456789ab"
CONTENT = b"# reviewed plan\n"
DIGEST_OK = hashlib.sha256(CONTENT).hexdigest()
ENCODED = base64.b64encode(CONTENT).decode("ascii")
TOO_BIG = b"#" * (probe.MAX_SOURCE_BYTES + 1)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeSystemd:
    return sandbox(monkeypatch, tmp_path)


def answer(
    monkeypatch: pytest.MonkeyPatch, fake: FakeSystemd, **replies: tuple[int, bytes]
) -> None:
    """The tool answers ``replies[path]`` (POSTs recorded) instead of the fake's."""

    def loopback(url: str, *, method: str = "GET", timeout: int = 10) -> Any:
        path = url.rsplit("/", 1)[-1]
        if path not in replies:
            return fake.loopback(url, method=method, timeout=timeout)
        if method == "POST":
            fake.posts.append(path)
        return replies[path]

    monkeypatch.setattr(probe, "loopback", loopback)


def after_restart(fake: FakeSystemd, environment: dict[str, str], **state: str) -> None:
    """A restarted collector execs with ``environment`` (and unit ``state``)."""

    def restarted(pid: str) -> None:
        fake.collector(pid, environment)
        fake.units[METRICS].update(state)

    fake.on_restart = restarted


# --------------------------------------------------------------------------- #
# install-source
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("name", "sha256", "b64", "message"),
    [
        ("evil.py", DIGEST_OK, ENCODED, "only the reviewed replay tools"),
        (probe.PLAN_NAME, "ABC", ENCODED, "a sha256 is required"),
        (probe.PLAN_NAME, DIGEST_OK, "not base64!", "source is not valid base64"),
        (
            probe.PLAN_NAME,
            hashlib.sha256(TOO_BIG).hexdigest(),
            base64.b64encode(TOO_BIG).decode("ascii"),
            "exceeds its size bound",
        ),
        (probe.PLAN_NAME, hashlib.sha256(b"x").hexdigest(), ENCODED, "does not match"),
    ],
    ids=["unreviewed-name", "no-sha256", "not-base64", "too-big", "sha-mismatch"],
)
def test_install_source_refuses_what_it_cannot_prove(
    fake: FakeSystemd, name: str, sha256: str, b64: str, message: str
) -> None:
    argv = ["install-source", "--run-id", RUN, "--name", name, "--sha256", sha256]
    assert probe.main([*argv, "--b64", b64]) == 1
    assert message in fake.emitted[-1]["error_message"], fake.emitted[-1]
    assert not probe.run_dir(RUN).exists(), "a refused source writes nothing"


# --------------------------------------------------------------------------- #
# deploy-proxy
# --------------------------------------------------------------------------- #
def test_deploy_proxy_refuses_an_unreviewed_tool_and_a_used_audit_file(
    fake: FakeSystemd,
) -> None:
    with pytest.raises(probe.ProbeError, match="only the reviewed replay tools"):
        probe.deploy_proxy(deploy_arguments(tool="other_replay.py"))
    root = installed_sources(probe.FIELD_TOOL, probe.PLAN_NAME)
    (root / "replay.jsonl").write_text("")
    with pytest.raises(probe.ProbeError, match="an audit file already exists"):
        probe.deploy_proxy(deploy_arguments())
    assert fake.commands == [], "nothing is stopped or started"


@pytest.mark.parametrize(
    "endpoint", ["http://10.0.0.7:9400/metrics", "http://127.0.0.1:9400/other"]
)
def test_deploy_proxy_refuses_an_upstream_that_is_not_loopback_metrics(
    fake: FakeSystemd, endpoint: str
) -> None:
    probe.COLLECTOR_ENV.write_text(f"{ENDPOINT}={endpoint}\n")
    installed_sources(probe.FIELD_TOOL, probe.PLAN_NAME)
    with pytest.raises(probe.ProbeError, match="not loopback /metrics"):
        probe.deploy_proxy(deploy_arguments())
    assert fake.commands == [], "nothing is started"


def test_deploy_proxy_serves_the_environment_tool_for_one_physical_gpu(
    fake: FakeSystemd,
) -> None:
    installed_sources(probe.ENVIRONMENT_TOOL, probe.PLAN_NAME)
    environment = {"tool": probe.ENVIRONMENT_TOOL}
    with pytest.raises(probe.ProbeError, match="needs a physical --gpu-uuid"):
        probe.deploy_proxy(deploy_arguments(**environment, gpu_uuid="0"))
    with pytest.raises(probe.ProbeError, match="runs at most 1800 s"):
        probe.deploy_proxy(
            deploy_arguments(**environment, gpu_uuid=UUID0, max_seconds=1801)
        )
    assert fake.commands == [], "a refused command starts nothing"
    probe.deploy_proxy(deploy_arguments(**environment, gpu_uuid=UUID0))
    receipt = fake.emitted[-1]
    assert receipt["tool"] == probe.ENVIRONMENT_TOOL
    assert receipt["endpoint"] == REPLAY and receipt["upstream"] == REAL
    assert receipt["status"] == {"armed": False, "reads": 0}
    started = next(c for c in fake.commands if c[0] == "systemd-run")
    assert started[-2:] == ["--gpu-uuid", UUID0], started
    assert "--profile" not in started, "the environment tool takes no profile"
    assert "--property=RuntimeMaxSec=1920s" in started


def test_deploy_proxy_refuses_a_tool_that_comes_up_armed(
    fake: FakeSystemd, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed_sources(probe.FIELD_TOOL, probe.PLAN_NAME)

    def loopback(url: str, *, method: str = "GET", timeout: int = 10) -> Any:
        fake.loopback(url, method=method, timeout=timeout)  # refused until up
        return 200, json.dumps({"armed": True, "reads": 3}).encode()

    monkeypatch.setattr(probe, "loopback", loopback)
    with pytest.raises(probe.ProbeError) as caught:
        probe.deploy_proxy(deploy_arguments())
    assert caught.value.reason == "armed_at_start", str(caught.value)
    assert fake.emitted == [], "no receipt for a refused tool"


# --------------------------------------------------------------------------- #
# the deadman and its copy of this file
# --------------------------------------------------------------------------- #
def test_probe_copy_keeps_a_copy_that_is_already_installed(fake: FakeSystemd) -> None:
    copy = probe.probe_copy(RUN)
    assert stat.S_IMODE(copy.stat().st_mode) == 0o700
    copy.write_text("# installed by an earlier step\n")
    assert probe.probe_copy(RUN) == copy
    assert copy.read_text() == "# installed by an earlier step\n", "not re-copied"


def test_arm_deadman_refuses_a_timer_that_did_not_come_up(
    fake: FakeSystemd, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(command: list[str], **options: Any) -> Any:
        result = fake.run(command, **options)
        if command[0] == "systemd-run":
            unit = next(a.split("=", 1)[1] for a in command if a.startswith("--unit="))
            fake.units[f"{unit}.timer"]["ActiveState"] = "failed"
        return result

    monkeypatch.setattr(probe, "run", run)
    with pytest.raises(probe.ProbeError, match="restore deadman is not armed"):
        probe.arm_deadman(RUN, 600)
    started = next(c for c in fake.commands if c[0] == "systemd-run")
    assert started[-3:] == ["cleanup", "--run-id", RUN], started


def test_disarm_deadman_leaves_the_restore_that_is_running_this_cleanup(
    fake: FakeSystemd,
) -> None:
    name = probe.deadman_unit(RUN)
    fake.unit(f"{name}.service", MainPID=str(os.getpid()))
    probe.disarm_deadman(RUN)
    stops = [c for c in fake.commands if c[:2] == ["systemctl", "stop"]]
    assert stops == [["systemctl", "stop", f"{name}.timer"]], stops
    assert fake.commands[-1] == [
        "systemctl",
        "reset-failed",
        f"{name}.service",
        f"{name}.timer",
    ]


# --------------------------------------------------------------------------- #
# the override environment file and the inventory wrapper link
# --------------------------------------------------------------------------- #
def test_override_lines_add_the_boot_id_unknown_count_and_wrapper_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    node(monkeypatch, tmp_path)
    root = installed_sources()
    with pytest.raises(probe.ProbeError) as caught:
        probe.override_lines(RUN, 9412, True)
    assert caught.value.reason == "source_missing", str(caught.value)
    installed_sources(probe.INVENTORY_WRAPPER, probe.INVENTORY_PLAN)
    lines = probe.override_lines(
        RUN, 9412, True, boot_id=REPLAYED_BOOT, unknown_expected_count=True
    )
    assert lines == [
        f"{ENDPOINT}={REPLAY}",
        f"GPU_FAULT_BOOT_ID_PATH={root / 'boot_id'}",
        "GPU_FAULT_EXPECTED_GPU_COUNT=",
        f"GPU_FAULT_NODE_INSTANCE_TYPE={probe.UNKNOWN_INSTANCE_TYPE}",
        f"PATH={root / 'bin'}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin",
        f"GFRS_LIVE_INVENTORY_PLAN={root / probe.INVENTORY_PLAN}",
        f"GFRS_LIVE_INVENTORY_AUDIT={root / 'inventory.jsonl'}",
    ]
    replayed = root / "boot_id"
    assert replayed.read_text() == REPLAYED_BOOT + "\n"
    assert stat.S_IMODE(replayed.stat().st_mode) == 0o644, "the collector reads it"
    with pytest.raises(probe.ProbeError, match="the node's current boot"):
        probe.override_lines(RUN, 9412, False, boot_id=BOOT_ID)


def test_install_wrapper_link_replaces_whatever_bin_nvidia_smi_was(
    fake: FakeSystemd, tmp_path: Path
) -> None:
    root = installed_sources(probe.INVENTORY_WRAPPER)
    script = (
        f"#!/bin/sh\nexec {probe.VENV_PYTHON} -I "
        f'{root / probe.INVENTORY_WRAPPER} "$@"\n'
    )
    link = probe.install_wrapper_link(RUN)
    assert link == root / "bin" / "nvidia-smi" and link.read_text() == script
    assert stat.S_IMODE(link.stat().st_mode) == 0o755
    link.unlink()
    link.symlink_to(tmp_path / "dangling-target")
    assert probe.install_wrapper_link(RUN) == link
    assert not link.is_symlink() and link.read_text() == script, "a link is replaced"
    link.write_text("# an older wrapper script\n")
    probe.install_wrapper_link(RUN)
    assert link.read_text() == script, "an older script is replaced"


# --------------------------------------------------------------------------- #
# repoint-collector
# --------------------------------------------------------------------------- #
def test_repoint_refuses_an_inactive_collector_and_existing_drop_ins(
    fake: FakeSystemd,
) -> None:
    fake.unit(METRICS, ActiveState="inactive")
    with pytest.raises(probe.ProbeError, match="is not active at baseline"):
        probe.repoint_collector(repoint_arguments())
    fake.collector("100", {ENDPOINT: REAL})
    probe.DROPIN_DIR.mkdir(parents=True)
    probe.dropin_path(RUN).write_text("[Service]\n")
    with pytest.raises(probe.ProbeError, match="drop-in for this run already exists"):
        probe.repoint_collector(repoint_arguments())
    probe.dropin_path(RUN).unlink()
    (probe.DROPIN_DIR / FOREIGN_DROPIN).write_text("[Service]\n")
    with pytest.raises(probe.ProbeError, match="another runtime drop-in"):
        probe.repoint_collector(repoint_arguments())
    assert not any(c[0] == "systemd-run" for c in fake.commands), "no deadman armed"


def test_repoint_refuses_a_collector_that_is_down_after_the_restart(
    fake: FakeSystemd,
) -> None:
    after_restart(fake, {ENDPOINT: REPLAY}, ActiveState="failed")
    with pytest.raises(probe.ProbeError, match="is not active after the repoint"):
        probe.repoint_collector(repoint_arguments())
    assert probe.dropin_path(RUN).exists(), "the deadman restores what was written"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"inventory_wrapper": True}, "did not pick up the wrapper PATH"),
        ({"boot_id": REPLAYED_BOOT}, "did not pick up the replayed boot id"),
        ({"unknown_expected_count": True}, "did not pick up the unknown count"),
    ],
    ids=["wrapper", "boot-id", "unknown-count"],
)
def test_repoint_refuses_a_collector_that_missed_one_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    options: dict[str, Any],
    message: str,
) -> None:
    host = node(monkeypatch, tmp_path)
    installed_sources(probe.INVENTORY_WRAPPER, probe.INVENTORY_PLAN)
    after_restart(host.fake, {ENDPOINT: REPLAY, "PATH": "/usr/bin"})
    with pytest.raises(probe.ProbeError, match=message):
        probe.repoint_collector(repoint_arguments(**options))
    assert host.fake.emitted == [], "no receipt"


def test_repoint_with_every_override_names_the_wrapper_and_writable_run_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    host = node(monkeypatch, tmp_path)
    root = installed_sources(probe.INVENTORY_WRAPPER, probe.INVENTORY_PLAN)
    environment = {
        ENDPOINT: REPLAY,
        "PATH": f"{root / 'bin'}:/usr/bin",
        "GPU_FAULT_BOOT_ID_PATH": str(root / "boot_id"),
        "GPU_FAULT_NODE_INSTANCE_TYPE": probe.UNKNOWN_INSTANCE_TYPE,
    }
    after_restart(host.fake, environment)
    options = {"inventory_wrapper": True, "unknown_expected_count": True}
    probe.repoint_collector(repoint_arguments(boot_id=REPLAYED_BOOT, **options))
    receipt = host.fake.emitted[-1]
    assert receipt["inventory_wrapper"] is True
    assert receipt["process_environment"] == environment
    assert (root / "bin" / "nvidia-smi").is_file(), "the wrapper link is installed"
    assert f"ReadWritePaths={root}\n" in probe.dropin_path(RUN).read_text()


# --------------------------------------------------------------------------- #
# arm / disarm / advance / collector-restart
# --------------------------------------------------------------------------- #
def test_arm_reports_a_tool_that_refuses_the_arm(
    fake: FakeSystemd, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake.tool_answers = True
    fake.collector("200", {ENDPOINT: REPLAY})
    answer(monkeypatch, fake, arm=(409, b"already armed\n"))
    assert probe.main(["arm", "--run-id", RUN]) == 1
    assert fake.emitted[-1]["error_message"] == "arm returned 409"
    assert fake.posts == ["arm"]


@pytest.mark.parametrize("verb", ["disarm", "advance"])
def test_disarm_and_advance_post_once_and_report_the_tool_status(
    fake: FakeSystemd, monkeypatch: pytest.MonkeyPatch, verb: str
) -> None:
    fake.tool_answers = True
    assert probe.main([verb, "--run-id", RUN]) == 0
    done = {"disarm": "disarmed", "advance": "advanced"}[verb]
    assert fake.emitted[-1] == {
        "run_id": RUN,
        done: True,
        "status": {"armed": False, "reads": 0},
    }
    assert fake.posts == [verb]
    answer(monkeypatch, fake, **{verb: (404, b"not found\n")})
    assert probe.main([verb, "--run-id", RUN]) == 1
    assert fake.emitted[-1]["error_message"] == f"{verb} returned 404"


def test_collector_restart_reports_the_new_invocation_and_endpoint(
    fake: FakeSystemd,
) -> None:
    fake.units[METRICS]["InvocationID"] = "first"
    restored_on_restart(fake)
    assert probe.main(["collector-restart", "--run-id", RUN]) == 0
    receipt = fake.emitted[-1]
    assert (
        receipt["before"]["MainPID"] == "100" and receipt["after"]["MainPID"] != "100"
    )
    assert receipt["restarted"] is True and receipt["effective_endpoint"] == REAL
    after_restart(fake, {ENDPOINT: REAL}, ActiveState="failed")
    assert probe.main(["collector-restart", "--run-id", RUN]) == 1
    assert (
        fake.emitted[-1]["error_message"]
        == f"{METRICS} is not active after the restart"
    )


# --------------------------------------------------------------------------- #
# restore-collector, stop-proxy and the deadman's cleanup
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("record", ["{broken", "{}"], ids=["not-json", "no-endpoint"])
def test_restore_collector_falls_back_to_the_baseline_endpoint(
    fake: FakeSystemd, record: str
) -> None:
    leftover_run(fake)
    (probe.run_dir(RUN) / "repoint.json").write_text(record)
    restored_on_restart(fake)
    assert probe.main(["restore-collector", "--run-id", RUN]) == 0
    receipt = fake.emitted[-1]
    assert receipt["original_endpoint"] == REAL == receipt["restored_endpoint"]
    assert receipt["dropin_removed"] is True and receipt["dropin_was_present"] is True
    assert f"{probe.deadman_unit(RUN)}.timer" not in fake.units, "deadman disarmed"


@pytest.mark.parametrize(
    ("environment", "state", "message"),
    [
        (
            {ENDPOINT: REAL},
            {"ActiveState": "failed"},
            "is not active after the restore",
        ),
        ({}, {}, "restored collector did not exec within the settle window"),
        (
            {ENDPOINT: REAL, "PATH": "{root}/bin:/usr/bin"},
            {},
            "still runs with the replay wrapper PATH",
        ),
        (
            {ENDPOINT: REAL, "GPU_FAULT_BOOT_ID_PATH": "{root}/boot_id"},
            {},
            "still reads the replayed boot id",
        ),
        (
            {
                ENDPOINT: REAL,
                "GPU_FAULT_NODE_INSTANCE_TYPE": probe.UNKNOWN_INSTANCE_TYPE,
            },
            {},
            "still runs with the unknown instance type",
        ),
    ],
    ids=["down", "never-execed", "wrapper-path", "boot-id", "unknown-type"],
)
def test_restore_collector_refuses_a_collector_still_on_the_replay_settings(
    fake: FakeSystemd, environment: dict[str, str], state: dict[str, str], message: str
) -> None:
    leftover_run(fake)
    root = str(probe.run_dir(RUN))
    after_restart(
        fake, {k: v.format(root=root) for k, v in environment.items()}, **state
    )
    assert probe.main(["restore-collector", "--run-id", RUN]) == 1
    assert message in fake.emitted[-1]["error_message"], fake.emitted[-1]
    assert f"{probe.deadman_unit(RUN)}.timer" in fake.units, "the deadman stays armed"


def test_stop_proxy_stops_the_tool_and_removes_the_run_directory(
    fake: FakeSystemd,
) -> None:
    leftover_run(fake)
    fake.collector("200", {ENDPOINT: REAL})
    assert probe.main(["stop-proxy", "--run-id", RUN]) == 0
    assert fake.emitted[-1] == {
        "run_id": RUN,
        "unit": f"{probe.proxy_unit(RUN)}.service",
        "unit_active": False,
        "audit_was_present": True,
        "run_dir_removed": True,
        "port": 9412,
    }
    assert not probe.run_dir(RUN).exists(), "the run directory is gone"


def test_cleanup_returns_once_every_step_succeeded(fake: FakeSystemd) -> None:
    leftover_run(fake)
    restored_on_restart(fake)
    assert probe.main(["cleanup", "--run-id", RUN]) == 0
    result = fake.emitted[-1]
    assert result["errors"] == [] and result["run_id"] == RUN
    assert result["restore"]["restored_endpoint"] == REAL
    assert result["stop"]["run_dir_removed"] is True
    assert result["residue"]["units"] == [], result["residue"]


def test_cleanup_exits_non_zero_through_main_so_the_deadman_retries(
    fake: FakeSystemd,
) -> None:
    leftover_run(fake)
    stuck_on_restart(fake)
    with pytest.raises(SystemExit) as caught:
        probe.main(["cleanup", "--run-id", RUN])
    assert caught.value.code == 1
    result = fake.emitted[-1]
    assert [error.split(":", 1)[0] for error in result["errors"]] == ["restore", "stop"]
    assert "error_kind" not in result, "the result itself, not a probe failure line"
