"""The host half of retiring a stale DESTR-008 service window (stdlib only).

A window whose run never closed it -- its undo failed or never ran -- keeps its
drop-in, its stop/restore units, the service's claim and its directory on the
node, and its controller journal open in the case directory
(``service_window_retire`` is the controller half). This module never runs on
its own: ``service_window_retire.retire_probe_source`` appends everything below
the host-part marker to ``warm_spare_node_probe.py``'s own source, so every name,
receipt, lock and systemd read here is that module's (``probe``), never
re-derived. In tests it is imported like any module and ``probe`` is
``warm_spare_node_probe``; on the host ``probe`` is the combined script itself.

``retire_window`` acts only on proof, and every failing proof is a named refusal
with nothing changed:

* shape -- the host journal says RESTORED (its restore unit or a controller
  restore brought the service back; CLOSING with this module's own retirement
  marker resumes an interrupted retirement), or the window published nothing
  (PREPARING with no receipt, or no host journal at all), or it is CLOSED;
* the service runs, on the machine the window was bound on, under exactly the
  invocation the window restored (its start receipt, or the untouched original
  when the stop never fired) -- nothing here ever starts or stops it;
* every artifact removed is the window's own: named by its key (drop-in
  ``90-gpu-fault-destr008-<key32>.conf``, the three
  ``gpu-fault-destr008-<key32>-*`` units, the directory ``<key>``), matching its
  publication receipt (inode, digest, mtime, mode), its units inactive with no
  job, process or cgroup member, and the claim naming this window;
* no other window is live for the service (one that published nothing is
  inert and noted) and no other window's drop-in is on it;
* the earlier probe's ``/run`` script copy, when named, carries that probe's
  script digest.

The baseline fingerprints a later node Agent wave rewrites (``node-agent.env``,
``/opt/gpu-fault/current``) are deliberately not compared: every release
changes them, and the window's own cleanup refusing exactly that is what left
it stale. The actions then: mark the journal CLOSING with a retirement marker,
remove the units and the drop-in by receipt, ``systemctl daemon-reload`` (the
service is not restarted, and that is verified), remove the claim, remove the
named script copy, mark the journal CLOSED and remove the window directory with
the journal last -- so an interrupted retirement is resumed by the next call.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import time
from pathlib import Path
from typing import Any

from scripts.e2e.regional.probes import warm_spare_node_probe as probe

# --- retire probe host part ---
#: Where host probes install their script copy (``HostProbeFixture.host_script``).
RETIRE_RUN = Path("/run")
RETIRE_SCRIPT_NAME = re.compile(r"gpu-fault-host-probe-[0-9a-f]{10}\.py")
#: The drop-in every window publishes on its service (``ServiceWindow.dropin``).
RETIRE_DROPIN_PREFIX = "90-gpu-fault-destr008-"
#: The window's own systemd artifacts, removed in the order cleanup removes them.
RETIRE_ROLES = ("bound", "stop-service", "stop-timer", "restore-service")
RETIRE_RESTORED = {"controller", "cleanup", "deadline"}


def retire_window(
    window: Any, stale_script: str = "", stale_sha256: str = ""
) -> dict[str, Any]:
    """Retire ``window`` when every proof holds; else name each failed proof."""

    if not probe.ROOT.exists() and not probe.ROOT.is_symlink():
        # Nothing of any window was ever written on this host.
        problem = retire_script_problem(stale_script, stale_sha256)
        removed = [] if problem else retire_remove_script(stale_script)
        proofs = {"shape": "absent", "root": "absent"}
        return retire_report(window, proofs, [problem] if problem else [], removed)
    entered = False
    try:
        with window.locked():
            entered = True
            refusals, proofs = retire_proofs(window, stale_script, stale_sha256)
            if refusals:
                return retire_report(window, proofs, refusals, [])
            removed = retire_locked(window, proofs["shape"], stale_script)
            return retire_report(window, proofs, [], removed)
    except probe.ProbeError as exc:
        if entered:
            raise
        return retire_report(
            window, {}, [f"the window's host journal is unusable: {exc}"], []
        )


def retire_report(
    window: Any, proofs: dict[str, Any], refusals: list[str], removed: list[str]
) -> dict[str, Any]:
    return {
        "binding_sha256": window.key,
        "service": window.service,
        "retired": not refusals,
        "refusals": refusals,
        "proofs": proofs,
        "removed": removed,
    }


def retire_shape(record: dict[str, Any]) -> str:
    if not record:
        return "absent"
    phase = record["phase"]
    if phase == "CLOSED":
        return "closed"
    if phase == "RESTORED" or (
        phase == "CLOSING" and isinstance(record.get("retirement"), dict)
    ):
        return "restored"
    if (
        phase == "PREPARING"
        and not record["installed"]
        and not record["resources"]
        and not record["stop_requested"]
    ):
        return "unpublished"
    return "unretirable"


def retire_proofs(
    window: Any, stale_script: str, stale_sha256: str
) -> tuple[list[str], dict[str, Any]]:
    """Every proof ``retire_window`` needs, with a named refusal per failure."""

    record = window.record
    shape = retire_shape(record)
    refusals: list[str] = []
    if shape == "unretirable":
        refusals.append(
            f"the window is {record['phase']}: only one its own restore left "
            "RESTORED, or one that published nothing, is retired"
        )
    unit = probe.service_snapshot(window.service)
    proofs: dict[str, Any] = {
        "shape": shape,
        "phase": record.get("phase") if record else None,
        "service": {
            key: unit[key]
            for key in ("InvocationID", "MainPID", "ActiveState", "SubState")
        },
        "same_boot": probe.boot_id() == window.binding["boot_id"],
    }
    if not probe.running(unit):
        refusals.append(
            f"{window.service} is not running ({unit['ActiveState']}/"
            f"{unit['SubState']}); a retirement never starts it"
        )
    machine = [
        probe.MACHINE_ID.read_text().strip(),
        probe.PRODUCT_ID.read_text().strip(),
    ]
    if probe.digest(machine) != window.binding["host_sha256"]:
        refusals.append("this is not the machine the window was bound on")
    if shape == "restored":
        expected = (
            (record.get("start_receipt") or {}).get("invocation_id")
            if record["start_intent"]
            else record["before"].get("InvocationID")
        )
        restored = (record.get("after") or {}).get("InvocationID")
        proofs["restored_invocation"] = expected
        proofs["restored_by"] = record.get("restore_reason")
        if not expected or restored != expected or unit["InvocationID"] != expected:
            refusals.append(
                f"{window.service} runs invocation {unit['InvocationID']!r}, not "
                f"{expected!r}, the one its window restored"
            )
        if record.get("restore_reason") not in RETIRE_RESTORED:
            refusals.append("the journal names no restorer for the service")
    refusals.extend(retire_artifact_refusals(window, shape))
    if shape == "restored":
        live, inert = retire_other_windows(window)
        refusals.extend(live)
        proofs["inert_windows"] = inert
    problem = retire_script_problem(stale_script, stale_sha256)
    if problem:
        refusals.append(problem)
    proofs["stale_script"] = stale_script or None
    return refusals, proofs


def retire_artifact_refusals(window: Any, shape: str) -> list[str]:
    """The window's own artifacts: named by its key, its receipts, quiescent."""

    refusals: list[str] = []
    receipts = window.record.get("resources") or {}
    owned = shape == "restored"
    for role in RETIRE_ROLES:
        target = window.targets[role]
        receipt = receipts.get(role)
        if window.key[:32] not in target.name:
            refusals.append(f"{target} does not carry the window's key")
        if (target.exists() or target.is_symlink()) and (
            not owned
            or receipt is None
            or not probe.matches(target, receipt["identity"])
        ):
            refusals.append(
                f"{target} is not this window's own publication"
                if owned
                else f"{target} is present though the window ({shape}) owns none"
            )
        if role in window.units:
            state = probe.unit_state(window.units[role])
            if not probe.idle(state):
                refusals.append(
                    f"{window.units[role]} is {state['ActiveState']}/"
                    f"{state['SubState']}, not inactive"
                )
            elif state["LoadState"] == "loaded" and state["FragmentPath"] not in {
                "",
                str(target),
            }:
                refusals.append(
                    f"{window.units[role]} is loaded from {state['FragmentPath']}"
                )
    claim = probe.service_claim(window.service)
    if claim is not None:
        receipt = receipts.get("claim")
        if claim.get("key") == window.key:
            if (
                not owned
                or receipt is None
                or not probe.matches(window.targets["claim"], receipt["identity"])
            ):
                refusals.append(
                    "the claim names this window but is not its own receipt"
                )
        elif owned:
            refusals.append(
                f"the {window.service} claim names window "
                f"{str(claim.get('key'))[:32]}, not this one"
            )
        elif claim.get("unreadable") is not False:
            refusals.append(f"the {window.service} claim cannot be read")
    if window.directory.is_dir():
        allowed = {"state.json", "helper.py"} | {
            f"source-{name}" for name in window.targets
        }
        for path in sorted(window.directory.iterdir()):
            if path.name not in allowed and not path.name.startswith(".state-"):
                refusals.append(f"the window directory holds {path.name}")
            elif path.is_symlink() or not path.is_file():
                refusals.append(f"the window directory entry {path.name} is not a file")
    return refusals


def retire_other_windows(window: Any) -> tuple[list[str], list[str]]:
    """Other windows of the service: live ones refuse, inert ones are noted."""

    live: list[str] = []
    inert: list[str] = []
    for path in sorted(probe.ROOT.iterdir()):
        if path.name == window.key or not probe.HEX.fullmatch(path.name):
            continue
        name = path.name[:32]
        state = path / "state.json"
        try:
            if path.is_symlink() or not path.is_dir():
                live.append(f"window entry {name} is not a directory")
                continue
            if not (state.exists() or state.is_symlink()):
                (inert if not any(path.iterdir()) else live).append(
                    f"window {name} has no journal"
                )
                continue
            other = probe.private_read(state)
        except (OSError, ValueError, probe.ProbeError):
            live.append(f"window {name} cannot be read")
            continue
        recorded = other.get("binding")
        binding: dict[str, Any] = recorded if isinstance(recorded, dict) else {}
        if binding.get("service") != window.service or other.get("phase") == "CLOSED":
            continue
        if (
            other.get("installed")
            or other.get("resources")
            or other.get("stop_requested")
        ):
            live.append(
                f"another window {name} ({other.get('phase')}) still holds "
                f"{window.service}"
            )
        else:
            inert.append(f"{name} ({other.get('phase')}, published nothing)")
    if window.dropin.parent.is_dir():
        for path in sorted(window.dropin.parent.glob(RETIRE_DROPIN_PREFIX + "*.conf")):
            if path != window.dropin:
                live.append(f"another window's drop-in {path} is on {window.service}")
    return live, inert


def retire_script_problem(stale_script: str, stale_sha256: str) -> str:
    """Why the named ``/run`` script copy must not be removed; ``""`` if it may."""

    if not stale_script:
        return ""
    path = Path(stale_script)
    if (
        path.parent != Path("/run")
        or not RETIRE_SCRIPT_NAME.fullmatch(path.name)
        or not probe.HEX.fullmatch(stale_sha256)
    ):
        return f"{stale_script} is not a host-probe script copy under /run"
    target = RETIRE_RUN / path.name
    try:
        info = target.lstat()
    except FileNotFoundError:
        return ""
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
        return f"{stale_script} is not a regular file of this user"
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != stale_sha256:
        return f"{stale_script} is not the earlier probe's script (digest differs)"
    return ""


def retire_remove_script(stale_script: str) -> list[str]:
    if not stale_script:
        return []
    target = RETIRE_RUN / Path(stale_script).name
    if not target.exists() and not target.is_symlink():
        return []
    target.unlink()
    probe.sync_directory(target.parent)
    return [stale_script]


def retire_locked(window: Any, shape: str, stale_script: str) -> list[str]:
    """The retirement itself, under the window lock, after every proof held."""

    removed: list[str] = []
    record = window.record
    if shape == "restored":
        if record["phase"] != "CLOSING":
            record.update(
                phase="CLOSING",
                retiring=True,
                retirement={"started_at": time.time()},
            )
            window.save()
        invocation = probe.service_snapshot(window.service)["InvocationID"]
        for role in reversed(tuple(record["resources"])):
            target = window.targets[role]
            if role not in {"claim", "helper"} and (
                target.exists() or target.is_symlink()
            ):
                probe.remove_owned(target, record["resources"][role]["identity"])
                removed.append(str(target))
        probe.systemctl("daemon-reload")
        for name in window.units.values():
            state = probe.unit_state(name)
            if state["LoadState"] != "not-found" or not probe.idle(state):
                raise probe.ProbeError("a retired window unit is still loaded")
        after = probe.service_snapshot(window.service)
        if not probe.running(after) or after["InvocationID"] != invocation:
            raise probe.ProbeError("the service changed while its window was retired")
        claim = window.targets["claim"]
        if "claim" in record["resources"] and (claim.exists() or claim.is_symlink()):
            probe.remove_owned(claim, record["resources"]["claim"]["identity"])
            removed.append(str(claim))
    removed.extend(retire_remove_script(stale_script))
    if record:
        record.update(
            phase="CLOSED",
            retiring=True,
            retirement={
                **(record.get("retirement") or {}),
                "finished_at": time.time(),
                "removed": removed,
            },
        )
        window.save()
    if window.directory.is_dir():
        entries = sorted(
            window.directory.iterdir(), key=lambda p: p.name == "state.json"
        )
        for path in entries:
            path.unlink()
            removed.append(str(path))
        window.directory.rmdir()
        probe.sync_directory(probe.ROOT)
        if entries:  # not the empty directory this call's own lock created
            removed.append(str(window.directory))
    return removed


def retire_script(stale_script: str, stale_sha256: str) -> dict[str, Any]:
    """Only the earlier probe's ``/run`` copy: a journal that bound no window."""

    problem = retire_script_problem(stale_script, stale_sha256)
    return {
        "retired": not problem,
        "refusals": [problem] if problem else [],
        "proofs": {"shape": "unbound", "stale_script": stale_script or None},
        "removed": [] if problem else retire_remove_script(stale_script),
    }


def retire_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("retire", "retire-script"))
    parser.add_argument("--binding")
    parser.add_argument("--stale-script", default="")
    parser.add_argument("--stale-script-sha256", default="")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "retire":
            report = retire_window(
                probe.bound_window(arguments),
                arguments.stale_script,
                arguments.stale_script_sha256,
            )
        else:
            report = retire_script(
                arguments.stale_script, arguments.stale_script_sha256
            )
        probe.emit(report)
        return 0
    except Exception as exc:
        probe.emit(
            {
                "error": type(exc).__name__,
                "error_kind": "probe",
                "error_class": type(exc).__name__,
                "error_reason": "retire_unfinished",
                "error_message": str(exc)
                if isinstance(exc, probe.ProbeError)
                else type(exc).__name__,
                "recovery_required": True,
            }
        )
        return 1
