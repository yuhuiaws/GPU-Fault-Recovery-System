"""Target-scoped persistence release for the GPU service quiesce.

Persistence keeps a GPU initialised while no client holds it, and a node gets
it from one of two mechanisms that are not ours to choose: ``nvidia-persistenced``
holds every GPU device node open, while the legacy driver flag that
``gpu-fault-gpu-persistence`` sets (``nvidia-smi -pm 1``) needs no process at
all. HyperPod starts the daemon by hand shortly after boot and the unit races
it, so two nodes of one job can boot into different regimes.

The quiesce used to end the daemon's hold by signalling the daemon (or, on EKS,
by stopping its unit). The daemon's shutdown disables persistence on *every*
device it holds, so on a node that relied on the daemon alone every
``nvidia-smi`` of the window re-initialised all of its GPUs -- three to seven
times slower than on a node whose legacy flag kept them up -- and the two halves
of one parallel reset started about a minute apart (live, HyperPod,
GF-REGIONAL-DESTR-015). Nothing brought the daemon back either.

``nvidia-smi -i <bus id> -pm 0`` ends persistence on one GPU in both regimes:
with the daemon running it travels over the daemon's RPC and the daemon closes
that one device; without it, it clears that GPU's legacy flag. Every other GPU
keeps what it had. The sweep stays the fallback for a daemon that still holds a
target afterwards, and everything here is recorded, bounded, in the step result.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from gpu_fault.node_agent.common import (
    PERSISTENCE_DAEMON_PROCESS,
    PERSISTENCE_DAEMON_UNIT,
    PERSISTENCE_RESTORE_UNIT,
)
from gpu_fault.node_agent.operations.holders import process_cgroup_paths

#: Per-GPU records travel in the step result and the state file.
RECORD_LIMIT = 16
TEXT_LIMIT = 200
QUERY_TIMEOUT_SECONDS = 15
RELEASE_TIMEOUT_SECONDS = 30
#: One deadline for the release phase, so N targets cannot add up to N timeouts.
RELEASE_BUDGET_SECONDS = 60
PCI_BUS_ID_PATTERN = re.compile(
    r"^[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$"
)
PERSISTENCE_QUERY = [
    "nvidia-smi",
    "--query-gpu=pci.bus_id,persistence_mode",
    "--format=csv,noheader",
]

Run = Callable[..., subprocess.CompletedProcess[str]]


def bounded(value: Any, limit: int = TEXT_LIMIT) -> str:
    text = "".join(char if char.isprintable() else "?" for char in str(value))
    return text if len(text) <= limit else text[: limit - 3] + "..."


class NvidiaSmiBudget:
    """The time the persistence calls of one step may spend.

    A call that timed out marks the driver unresponsive and every later call of
    the step is skipped: a driver wedged for one call is wedged for the next,
    and the quiesce falls back to the sweep instead of paying a timeout per GPU.
    """

    def __init__(
        self, seconds: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.clock = clock
        self.deadline = clock() + seconds
        self.unresponsive: str | None = None

    def renew(self, seconds: float) -> None:
        self.deadline = self.clock() + seconds

    def timeout(self, cap: int) -> int | None:
        remaining = int(self.deadline - self.clock())
        if self.unresponsive is not None or remaining < 1:
            return None
        return min(cap, remaining)

    def refusal(self) -> str:
        return self.unresponsive or "the persistence calls ran out of time"


def _call(
    run: Run, argv: list[str], budget: NvidiaSmiBudget, cap: int
) -> tuple[subprocess.CompletedProcess[str] | None, str | None]:
    """One bounded ``nvidia-smi``: ``(completed, None)`` or ``(None, why not)``."""

    timeout = budget.timeout(cap)
    if timeout is None:
        return None, budget.refusal()
    try:
        completed = run(argv, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        budget.unresponsive = f"{argv[0]} did not answer within {timeout} s"
        return None, budget.unresponsive
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return None, bounded(f"{type(exc).__name__}: {exc}")
    if completed.returncode:
        output = str(completed.stderr or completed.stdout or "").strip()
        detail = f": {output}" if output else ""
        return None, bounded(f"exit status {completed.returncode}{detail}")
    return completed, None


def persistence_modes(
    run: Run, budget: NvidiaSmiBudget, first: Sequence[str] = ()
) -> tuple[list[dict[str, str]], str | None]:
    """``[{pci_bus_id, persistence_mode}]`` per GPU, ``first`` (the targets) first."""

    completed, error = _call(run, PERSISTENCE_QUERY, budget, QUERY_TIMEOUT_SECONDS)
    if completed is None:
        return [], error
    modes = []
    for line in str(completed.stdout or "").splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 2 and PCI_BUS_ID_PATTERN.fullmatch(values[0]):
            modes.append(
                {"pci_bus_id": values[0], "persistence_mode": bounded(values[1], 32)}
            )
    targets = {bus_id.upper() for bus_id in first}
    modes.sort(key=lambda item: item["pci_bus_id"].upper() not in targets)
    return modes[:RECORD_LIMIT], None


def _start_time_ticks(process: Path) -> int | None:
    try:
        stat = (process / "stat").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    # Field 22 of /proc/<pid>/stat; ``comm`` may hold spaces, so count from ")".
    fields = stat.rpartition(")")[2].split()
    return int(fields[19]) if len(fields) > 19 and fields[19].isdigit() else None


def _identity(pid: str, cgroup_paths: Sequence[str]) -> dict[str, Any]:
    units = [
        path.rsplit("/", 1)[-1] for path in cgroup_paths if path.endswith(".service")
    ]
    return {
        "pid": pid,
        "systemd_owned": bool(units),
        "systemd_unit": bounded(units[0], 128) if units else None,
    }


def persistence_daemon(proc_root: Path) -> dict[str, Any]:
    """The running ``nvidia-persistenced`` as ``/proc`` shows it, nothing spawned.

    ``systemd_owned`` is a ``.service`` cgroup: a unit systemd stops and starts.
    A daemon started by hand from a login session (HyperPod) runs in a
    ``session-N.scope`` and only its pid, executable and start time say which
    process it was.
    """

    observed: dict[str, Any] = {
        "running": False,
        "pid": None,
        "systemd_owned": False,
        "systemd_unit": None,
        "exe": None,
        "start_time_ticks": None,
    }
    try:
        pids = sorted(
            (entry.name for entry in proc_root.iterdir() if entry.name.isdigit()),
            key=int,
        )
    except OSError:
        return observed
    for pid in pids:
        process = proc_root / pid
        try:
            name = (process / "comm").read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            continue
        if name != PERSISTENCE_DAEMON_PROCESS:
            continue
        try:
            exe: str | None = bounded(os.readlink(process / "exe"), 256)
        except OSError:
            exe = None
        return {
            **observed,
            **_identity(pid, process_cgroup_paths(process)),
            "running": True,
            "exe": exe,
            "start_time_ticks": _start_time_ticks(process),
        }
    return observed


class TargetPersistence:
    """The persistence half of one QUIESCE step: release, keep, account for it.

    ``stop_services`` stops the window's services around the release and
    leaves a daemon unit that took the release running; ``finish`` runs after
    the sweep and answers the result fields. Neither raises for a driver that
    fails or hangs -- the release is then incomplete, the sweep remains the
    release, and the record says why -- only for a service stop that fails, as
    the stop always did.
    """

    def __init__(
        self,
        *,
        run: Run,
        holders: Callable[[], list[dict[str, Any]]],
        proc_root: Path,
        clock: Callable[[], float] = time.monotonic,
        kept_services: Sequence[dict[str, str]] = (),
    ) -> None:
        self.run = run
        self.holders = holders
        self.proc_root = proc_root
        self.budget = NvidiaSmiBudget(RELEASE_BUDGET_SECONDS, clock)
        self.daemon: dict[str, Any] = {}
        self.targets: list[str] = []
        self.modes: dict[str, list[dict[str, str]]] = {}
        self.errors: dict[str, str] = {}
        self.release: dict[str, Any] = {}
        # What the window keeps running, with why: a Fabric Manager the
        # inventory kept out of ``active_services``, and below the daemon unit.
        self.kept_services: list[dict[str, str]] = list(kept_services)

    def _snapshot(self, name: str) -> None:
        self.modes[name], error = persistence_modes(self.run, self.budget, self.targets)
        if error is not None:
            self.errors[name] = error

    def stop_services(
        self,
        *,
        active_services: Sequence[str],
        pci_bus_ids: Sequence[str] | None,
        whole_node: bool,
    ) -> None:
        """Stop the window's services, last configured first, around the release.

        The release runs before anything else stops, so a daemon is still there
        to take it, and right after ``gpu-fault-gpu-persistence`` stops:
        whatever restores the window -- this step, the retry of a failed one,
        the fail-safe timer -- starts that oneshot again, and its
        ``nvidia-smi -pm 1`` re-enables the targets. Started while still active
        it would not run, and a stop failing after the release would have left
        the targets without persistence past the restore.
        """

        self.daemon = persistence_daemon(self.proc_root)
        self.targets = list(pci_bus_ids or ())[:RECORD_LIMIT]
        self._snapshot("before_release")
        commands = self._release_commands(active_services, pci_bus_ids, whole_node)
        stopped = set()
        if commands:
            self._stop(PERSISTENCE_RESTORE_UNIT)
            stopped.add(PERSISTENCE_RESTORE_UNIT)
            self._release(commands)
        self.release["daemon"] = self.daemon
        if (
            self.release["complete"]
            and self.daemon["running"]
            and PERSISTENCE_DAEMON_UNIT in active_services
        ):
            self.kept_services.append(
                {
                    "service": PERSISTENCE_DAEMON_UNIT,
                    "reason": (
                        "every target GPU was released through the daemon "
                        "(nvidia-smi -pm 0); stopping the unit would end "
                        "persistence on every other GPU of the node for the window"
                    ),
                }
            )
            stopped.add(PERSISTENCE_DAEMON_UNIT)
        for service in reversed(active_services):
            if service not in stopped:
                self._stop(service)

    def _stop(self, service: str) -> None:
        self.run(["systemctl", "stop", service], check=True, timeout=120)

    def _release_commands(
        self,
        active_services: Sequence[str],
        pci_bus_ids: Sequence[str] | None,
        whole_node: bool,
    ) -> list[list[str]]:
        self.release = {
            "scope": "whole-node" if whole_node else "targets",
            "pci_bus_ids": [] if whole_node else self.targets,
            "commands": [],
            "complete": False,
        }
        if PERSISTENCE_RESTORE_UNIT not in active_services:
            self.release["reason"] = (
                f"{PERSISTENCE_RESTORE_UNIT} is not part of this window's restore, "
                "so nothing would enable persistence on the targets again; the "
                "device sweep remains the release"
            )
            return []
        if whole_node:
            return [["nvidia-smi", "-pm", "0"]]
        if pci_bus_ids and all(
            PCI_BUS_ID_PATTERN.fullmatch(item) for item in pci_bus_ids
        ):
            return [["nvidia-smi", "-i", item, "-pm", "0"] for item in pci_bus_ids]
        self.release["reason"] = (
            "the target GPUs' PCI bus ids are unknown, so no release was "
            "attempted; the device sweep remains the release"
        )
        return []

    def _release(self, commands: list[list[str]]) -> None:
        record = self.release
        failures = []
        for argv in commands[:RECORD_LIMIT]:
            completed, error = _call(
                self.run, argv, self.budget, RELEASE_TIMEOUT_SECONDS
            )
            entry: dict[str, Any] = {
                "command": " ".join(argv),
                "released": error is None,
            }
            if error is not None:
                entry["error"] = error
                failures.append(error)
            record["commands"].append(entry)
            if completed is None and self.budget.timeout(1) is None:
                break
        if failures or len(record["commands"]) < len(commands):
            record["reason"] = bounded(
                "the release failed (" + "; ".join(failures[:2] or ["not run"]) + "); "
                "the device sweep remains the release",
                3 * TEXT_LIMIT,
            )
            return
        holding = [
            holder
            for holder in self.holders()
            if holder.get("process_name") == PERSISTENCE_DAEMON_PROCESS
        ]
        if holding:
            record["reason"] = bounded(
                f"{PERSISTENCE_DAEMON_UNIT} (pid {holding[0].get('pid')}) still "
                f"holds {holding[0].get('devices')} after the release; the device "
                "sweep remains the release",
                3 * TEXT_LIMIT,
            )
            return
        record["complete"] = True
        record["reason"] = "every target GPU's persistence was released " + (
            f"through {PERSISTENCE_DAEMON_UNIT}"
            if self.daemon.get("running")
            else "in the legacy mode (no persistence daemon is running)"
        )

    def finish(self, swept: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """The result fields, read after the sweep: the window as it now is."""

        fields: dict[str, Any] = {}
        swept_daemon = next(
            (
                record
                for record in swept
                if record.get("process_name") == PERSISTENCE_DAEMON_PROCESS
            ),
            None,
        )
        if swept_daemon is not None:
            pid = str(swept_daemon.get("pid"))
            observed = self.daemon if self.daemon.get("pid") == pid else {}
            fields["persistence_daemon_swept"] = {
                **_identity(pid, swept_daemon.get("cgroup_paths") or []),
                "exe": observed.get("exe"),
                "start_time_ticks": observed.get("start_time_ticks"),
            }
        if self.budget.unresponsive is None:
            self.budget.renew(QUERY_TIMEOUT_SECONDS)
        self._snapshot("after_release")
        modes: dict[str, Any] = dict(self.modes)
        if self.errors:
            modes["errors"] = dict(self.errors)
        return {
            "kept_services": self.kept_services,
            "persistence_release": self.release,
            "persistence_mode": modes,
            "persistence_daemon": persistence_daemon(self.proc_root),
            **fields,
        }


def restored_daemon(state: dict[str, Any], proc_root: Path) -> dict[str, Any]:
    """RESTORE's daemon fields: what runs now, and a swept daemon never relaunched.

    The product does not own a daemon somebody else started, so the restore
    does not start one: it says the daemon is gone and leaves the decision to
    whoever started it. Persistence then comes from ``gpu-fault-gpu-persistence``
    alone, in the legacy mode. The services the window kept running (a Fabric
    Manager it never stopped) come back as ``kept_services``: not started here.
    """

    daemon = persistence_daemon(proc_root)
    fields: dict[str, Any] = {"persistence_daemon": daemon}
    kept = state.get("kept_services")
    if isinstance(kept, list) and kept:
        fields["kept_services"] = kept
    swept = state.get("persistence_daemon_swept")
    if not isinstance(swept, dict):
        return fields
    who = bounded(
        f"{PERSISTENCE_DAEMON_UNIT} pid {swept.get('pid')} "
        f"({swept.get('exe') or 'executable unknown'})",
        3 * TEXT_LIMIT,
    )
    fields["persistence_daemon_restored"] = bool(daemon["running"])
    fields["persistence_daemon_restore_reason"] = (
        f"the quiesce sweep signalled {who}; a daemon is running again as pid "
        f"{daemon['pid']}, which the restore did not relaunch"
        if daemon["running"]
        else f"the quiesce sweep signalled {who} and the restore does not "
        "relaunch a process the product does not own: the node runs without "
        "the daemon until whoever started it starts it again"
    )
    return fields


def restored_modes(run: Run) -> dict[str, Any]:
    modes, error = persistence_modes(run, NvidiaSmiBudget(QUERY_TIMEOUT_SECONDS))
    record: dict[str, Any] = {"after_restore": modes}
    if error is not None:
        record["errors"] = {"after_restore": error}
    return record
