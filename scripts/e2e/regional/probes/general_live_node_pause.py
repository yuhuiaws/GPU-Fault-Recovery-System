"""Producer pauses of the GF-GENERAL-LIVE node probe: the arm side.

The second node-side file of ``probes/general_live_node_probe.py``. A probe
ships to the node as one self-contained file (``host_probe_fixture.py``: one
ConfigMap key, sha256-checked), so the commands that *start* a producer pause
travel like the host source replay's fixture builders do: the runner installs
this module through the probe's ``install-source`` (base64 + sha256-checked,
under the run's state directory) once per run and node before the first pause,
and the probe imports the installed copy by path. Everything that *restores* --
the SIGCONTs, the drop-in removal, the log collector's runtime disable,
``residue`` and ``cleanup`` -- stays in the probe, so the deadman and the
runner's cleanup need the probe alone.

* ``unit-pause`` -- SIGSTOP every process of one allow-listed collector unit;
  the unit stays active and enabled, which is what the silence scanner must
  see. A ``/run`` drop-in with ``ExecStartPost=kill -STOP $MAINPID`` is loaded
  first: the Node Agent's TRIGGER_HEALTH_SNAPSHOT step
  (``node_agent/operations/diagnostics.py``) restarts the metrics and host
  collectors, and systemd follows SIGTERM with SIGCONT, so a bare SIGSTOP ends
  with the old process while the new one posts freely (live 2026-10-25: the
  GPU inventory channel of the paused metrics unit, 180 s, tripped that
  refresh 183 s into a 420 s GPU_METRICS pause and the restarted collector
  kept the channel FRESH_SUCCESS). The drop-in stops every new main process
  before it discovers a GPU or posts anything.
* ``exporter-pause`` -- SIGSTOP the node's dcgm-exporter process.
* ``log-collector-enable`` -- runtime-only enable of the installed, disabled
  node log collector, behind an explicit approval.
* ``unit-resume``, ``exporter-resume``, ``log-collector-disable`` -- the
  runner's explicit resumes; they call the probe's restore functions, which
  ``cleanup`` (the deadman's entry point) runs without this module.

Standard library only, Python 3.9+. State, bounds, allow-lists, the deadman and
every helper come from the probe module registered under ``PROBE_ALIAS`` (in a
checkout, the sibling module) and are looked up at call time, so the probe's
tests can stub them.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

#: ``sys.modules`` name the probe registers itself under before loading this file.
PROBE_ALIAS = "gfrs_general_live_node_probe"
#: The longest producer pause (``--seconds``, the on-node deadman's bound): 25
#: minutes. A runner sizes a pause from the threshold it waits for plus its
#: grade and slack, never from a case budget -- 030's 30-minute budget (1800 s)
#: was refused here live 2026-10-27 on both clusters before any pause ("pause
#: seconds is outside 60..1500", the case read only the digest);
#: ``general_live_fixture.pause_bound`` checks the bound before shipping.
MAX_PAUSE_SECONDS = 1500

if TYPE_CHECKING:
    from scripts.e2e.regional.probes import general_live_node_probe as probe
else:
    probe = sys.modules.get(PROBE_ALIAS) or importlib.import_module(
        "scripts.e2e.regional.probes.general_live_node_probe"
    )


def install_pause_dropin(run_id: str, unit: str) -> dict[str, Any]:
    """Load a drop-in that SIGSTOPs every new main process of ``unit``.

    ``ExecStartPost`` runs once the main process is forked (``Type=simple``),
    before the collector has discovered a GPU or posted anything, so a product
    restart during the pause leaves the channel silent instead of fresh.
    """

    kill = shutil.which("kill") or "/usr/bin/kill"
    if not Path(kill).is_file():
        raise probe.ProbeError("no kill binary for the pause drop-in")
    path = probe.pause_dropin_path(run_id, unit)
    created_dir = not path.parent.is_dir()
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    path.write_text(
        f"# gpu-fault acceptance pause of run {run_id}: a restarted main process\n"
        "# is stopped again; removed by unit-resume and by the run's deadman.\n"
        "[Service]\n"
        f"ExecStartPost={kill} -STOP $MAINPID\n",
        encoding="utf-8",
    )
    path.chmod(0o644)
    probe.run(["systemctl", "daemon-reload"])
    return {"path": str(path), "created_dir": created_dir}


def unit_pause(arguments: argparse.Namespace) -> None:
    """SIGSTOP every process of one collector unit; the unit stays active.

    A pause is not a disable: systemd keeps the unit ``active`` and the Agent
    keeps reporting it enabled. The pause drop-in is loaded before the SIGSTOP
    so that a product restart of the unit re-stops the new main process.
    """

    run_id = probe.checked_run_id(arguments.run_id)
    unit = probe.checked_pausable(arguments.unit)
    seconds = probe.bounded(
        arguments.seconds, probe.MIN_DEADMAN_SECONDS, MAX_PAUSE_SECONDS, "pause seconds"
    )
    before = probe.unit_state(unit)
    if before.get("ActiveState") != "active":
        raise probe.ProbeError(f"{unit} is not active at baseline")
    kind = f"resume-{unit.split('.')[0].removeprefix('gpu-fault-')}"
    state = probe.load_state(run_id)
    state["paused_units"][unit] = {
        "paused_at": probe.utc_now(),
        "before": before,
        "kind": kind,
    }
    probe.save_state(run_id, state)
    deadman = probe.arm_deadman(run_id, kind, seconds, "--unit", unit)
    try:
        dropin = install_pause_dropin(run_id, unit)
        state = probe.load_state(run_id)
        state["paused_units"][unit]["dropin"] = dropin
        probe.save_state(run_id, state)
        probe.run(["systemctl", "kill", "--kill-whom=all", "--signal=SIGSTOP", unit])
    except Exception:
        probe.resume_unit(run_id, unit)
        raise
    probe.emit(
        {
            "unit": unit,
            "before": before,
            "after": probe.unit_state(unit),
            "stopped": probe.unit_is_stopped(unit),
            "deadman": deadman,
            "dropin": dropin["path"],
            "resume_after_seconds": seconds,
        }
    )


def exporter_pause(arguments: argparse.Namespace) -> None:
    """SIGSTOP the node's dcgm-exporter process: its scrape port stops answering."""

    run_id = probe.checked_run_id(arguments.run_id)
    seconds = probe.bounded(
        arguments.seconds, probe.MIN_DEADMAN_SECONDS, MAX_PAUSE_SECONDS, "pause seconds"
    )
    processes = probe.exporter_processes()
    if len(processes) != 1:
        raise probe.ProbeError(
            f"expected exactly one {probe.EXPORTER_BINARY} process, "
            f"found {len(processes)}"
        )
    target = processes[0]
    state = probe.load_state(run_id)
    state["paused_processes"][str(target["pid"])] = {
        "start_ticks": target["start_ticks"],
        "paused_at": probe.utc_now(),
        "binary": probe.EXPORTER_BINARY,
    }
    probe.save_state(run_id, state)
    deadman = probe.arm_deadman(run_id, "resume-exporter", seconds)
    os.kill(int(target["pid"]), signal.SIGSTOP)
    probe.emit(
        {
            "pid": target["pid"],
            "stopped": probe.process_is_stopped(
                int(target["pid"]), target["start_ticks"]
            ),
            "deadman": deadman,
            "resume_after_seconds": seconds,
        }
    )


def log_collector_enable(arguments: argparse.Namespace) -> None:
    """Runtime-enable and start the disabled node log collector (needs approval).

    ``enable --runtime`` writes the link under ``/run`` only, so a reboot (or
    the deadman) returns the unit to its installed, disabled state.
    """

    if not arguments.approved:
        raise probe.ProbeError(
            "starting the node log collector requires the runner's approval"
        )
    run_id = probe.checked_run_id(arguments.run_id)
    seconds = probe.bounded(
        arguments.seconds,
        probe.MIN_DEADMAN_SECONDS,
        probe.MAX_DEADMAN_SECONDS,
        "seconds",
    )
    unit = probe.LOG_COLLECTOR_UNIT
    before = probe.unit_state(unit)
    if before.get("LoadState") != "loaded":
        raise probe.ProbeError("the node log collector unit is not installed")
    if before.get("ActiveState") == "active":
        raise probe.ProbeError("the node log collector is already running")
    state = probe.load_state(run_id)
    state["log_collector"] = {"before": before, "enabled_at": probe.utc_now()}
    probe.save_state(run_id, state)
    deadman = probe.arm_deadman(run_id, "logcollector", seconds)
    probe.run(["systemctl", "enable", "--runtime", "--now", unit], timeout=120)
    probe.emit({"before": before, "after": probe.unit_state(unit), "deadman": deadman})


def unit_resume(arguments: argparse.Namespace) -> None:
    probe.emit(
        probe.resume_unit(
            probe.checked_run_id(arguments.run_id),
            probe.checked_pausable(arguments.unit),
        )
    )


def exporter_resume(arguments: argparse.Namespace) -> None:
    run_id = probe.checked_run_id(arguments.run_id)
    probe.emit({"processes": probe.resume_processes(run_id)})


def log_collector_disable(arguments: argparse.Namespace) -> None:
    probe.emit(probe.restore_log_collector(probe.checked_run_id(arguments.run_id)))
