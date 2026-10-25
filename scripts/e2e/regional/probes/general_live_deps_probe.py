#!/usr/bin/env python3
"""Temporary test packages of one GF-GENERAL-LIVE run (TEMP_TEST_PACKAGE_INSTALL).

Runs on the target node through ``host_probe_fixture.py`` (chroot into the
host, the node's own Python), standard library only, one JSON object on the
last stdout line.

Some variants need a host tool or kernel module the node image does not ship:
GF-GENERAL-LIVE-009 ``tcp_only`` needs ``tc`` (the AL2023 EKS AMI has no
``iproute-tc``) and the ``sch_netem`` qdisc. ``ensure`` makes exactly those
usable and changes nothing that is already there:

* a missing tool or a module the running kernel cannot load is installed with
  ``dnf`` (``--allow-install``) -- only the reviewed package of
  ``TOOL_PACKAGES``/``MODULE_PACKAGES``, the module package always for the
  *running* kernel (``kernel-modules-extra-$(uname -r)``). The transaction is
  previewed first (``--assumeno``) and refused unless every row is
  ``Installing`` one of those packages: no kernel, upgrade, downgrade,
  reinstall, removal, obsolete, other dependency or unreadable row;
* a module that is neither loaded nor built in is loaded with ``modprobe``
  (``--allow-module-load``, the ``KERNEL_MODULE_LOAD`` approval).

What the run installs (NEVRAs, only from the rpm transaction that installed
them) and which module it loads are written to
``/run/gpu-fault-acceptance/general/<digest>/deps-state.json`` *before* the
change, after a ``systemd-run --on-active`` deadman is armed that runs
``cleanup`` from a copy of this file. ``cleanup`` unloads only a module this
run loaded (and only once nothing uses it) and removes only the recorded
packages, through a previewed ``dnf remove`` that must remove exactly them
(never a dependent package or an "unused" dependency). A refusal is a normal
answer (``refused`` with a NOT_RUN reason), never an error, so the runner can
record why the variant did not run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_ROOT = Path("/run/gpu-fault-acceptance/general")
UNIT_PREFIX = "gfrs-"
SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")
MIN_DEADMAN_SECONDS = 60
MAX_DEADMAN_SECONDS = 5400
#: The only tools this probe may install a package for, and that package.
TOOL_PACKAGES = {"tc": "iproute-tc"}
#: The only modules this probe may load, and the package that carries each when
#: the running kernel cannot load it -- installed as ``<package>-<uname -r>``.
MODULE_PACKAGES = {"sch_netem": "kernel-modules-extra"}
#: A package that is (part of) a kernel image: never installed by this probe.
KERNEL_PACKAGE = re.compile(r"^kernel(\d+(\.\d+)*)?(-core|-modules(-core)?)?$")
#: No weak dependencies, and no installonly package removed to make room.
DNF_INSTALL_OPTIONS = (
    "--setopt=install_weak_deps=False",
    "--setopt=installonly_limit=0",
)
#: Removal works on the installed set only and never cascades to "unused"
#: dependencies.
DNF_REMOVE_OPTIONS = (
    "--disablerepo=*",
    "--setopt=clean_requirements_on_remove=False",
)
DNF_TIMEOUT = 420
DNF_SECTION = re.compile(r"^([A-Z][A-Za-z /-]*):$")
SYS_MODULE = Path("/sys/module")
MODULES_ROOT = Path("/lib/modules")
EMPTY_STATE: dict[str, Any] = {"installed": [], "pending": [], "modules": []}


class ProbeError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def run(
    command: list[str],
    *,
    check: bool = True,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
        env={**os.environ, **env} if env else None,
    )
    if check and completed.returncode:
        raise ProbeError(
            f"command failed ({completed.returncode}): {shlex.join(command)}: "
            f"{completed.stderr.strip()[:300]}"
        )
    return completed


def checked_run_id(value: str) -> str:
    if SAFE_RUN_ID.fullmatch(value) is None:
        raise ProbeError("unsafe run ID")
    return value


def bounded(value: int, low: int, high: int, label: str) -> int:
    number = int(value)
    if not low <= number <= high:
        raise ProbeError(f"{label} is outside {low}..{high}")
    return number


def run_digest(run_id: str) -> str:
    return hashlib.sha256(checked_run_id(run_id).encode()).hexdigest()[:6]


def state_dir(run_id: str) -> Path:
    return STATE_ROOT / hashlib.sha256(run_id.encode()).hexdigest()[:12]


def deadman_unit(run_id: str) -> str:
    return f"{UNIT_PREFIX}{run_digest(run_id)}-pdeadman"


def load_state(run_id: str) -> dict[str, Any]:
    path = state_dir(run_id) / "deps-state.json"
    if not path.is_file():
        return {"run_id": run_id, **json.loads(json.dumps(EMPTY_STATE))}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("run_id") != run_id:
        raise ProbeError("deps probe state identity mismatch")
    for key, empty in EMPTY_STATE.items():
        value.setdefault(key, json.loads(json.dumps(empty)))
    return value


def save_state(run_id: str, state: dict[str, Any]) -> None:
    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / "deps-state.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def probe_copy(run_id: str) -> Path:
    """A copy of this file the deadman can run after the probe Pod is gone."""

    directory = state_dir(run_id)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = directory / "deps-probe.py"
    if not target.is_file():
        shutil.copyfile(Path(__file__).resolve(), target)
        target.chmod(0o700)
    return target


def unit_state(unit: str) -> dict[str, str]:
    completed = run(
        ["systemctl", "show", unit, "--property=ActiveState", "--property=MainPID"],
        check=False,
    )
    values = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def arm_deadman(run_id: str, seconds: int) -> str:
    bounded(seconds, MIN_DEADMAN_SECONDS, MAX_DEADMAN_SECONDS, "deadman seconds")
    name = deadman_unit(run_id)
    run(["systemctl", "stop", f"{name}.timer"], check=False)
    run(["systemctl", "reset-failed", f"{name}.service"], check=False)
    run(
        [
            "systemd-run",
            f"--unit={name}",
            f"--on-active={int(seconds)}s",
            "--timer-property=AccuracySec=1s",
            "--property=Restart=on-failure",
            "--property=RestartSec=30s",
            "--property=StartLimitIntervalSec=0",
            sys.executable,
            str(probe_copy(run_id)),
            "cleanup",
            "--run-id",
            run_id,
        ]
    )
    if unit_state(f"{name}.timer").get("ActiveState") != "active":
        raise ProbeError(f"deadman {name} is not armed")
    return name


def disarm_deadman(run_id: str) -> None:
    name = deadman_unit(run_id)
    run(["systemctl", "stop", f"{name}.timer"], check=False)
    if unit_state(f"{name}.service").get("MainPID") != str(os.getpid()):
        run(["systemctl", "stop", f"{name}.service"], check=False)
    run(["systemctl", "reset-failed", f"{name}.service", f"{name}.timer"], check=False)


def tool_present(name: str) -> bool:
    return shutil.which(name) is not None


def kernel_release() -> str:
    return os.uname().release


def module_state(name: str) -> dict[str, bool]:
    """Whether ``name`` is loaded, built in, or loadable for the running kernel."""

    loaded = (SYS_MODULE / name).exists()
    builtin = False
    try:
        listing = (MODULES_ROOT / kernel_release() / "modules.builtin").read_text()
        builtin = any(
            line.strip().endswith(f"/{name}.ko") for line in listing.splitlines()
        )
    except OSError:
        pass
    loadable = (
        loaded
        or builtin
        or (
            tool_present("modprobe")
            and run(
                ["modprobe", "--dry-run", "--quiet", name], check=False, timeout=60
            ).returncode
            == 0
        )
    )
    return {"loaded": loaded, "builtin": builtin, "loadable": loadable}


def nevra(name: str, version: str, arch: str) -> str:
    """``name-[epoch:]version-release.arch`` as dnf prints and accepts it."""

    return f"{name}-{version}.{arch}"


def rpm_packages() -> dict[str, dict[str, str]]:
    """Every installed package by NEVRA, with its rpm transaction id."""

    completed = run(
        [
            "rpm",
            "-qa",
            "--qf",
            r"%{NAME}\t%{EPOCH}\t%{VERSION}\t%{RELEASE}\t%{ARCH}\t%{INSTALLTID}\n",
        ],
        timeout=120,
    )
    packages: dict[str, dict[str, str]] = {}
    for line in completed.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) != 6:
            continue
        name, epoch, version, release, arch, tid = fields
        evr = f"{version}-{release}"
        if epoch not in {"", "0", "(none)"}:
            evr = f"{epoch}:{evr}"
        packages[nevra(name, evr, arch)] = {"name": name, "tid": tid}
    return packages


def dnf(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return run(
        ["dnf", *arguments],
        check=False,
        timeout=DNF_TIMEOUT,
        env={"LC_ALL": "C", "LANG": "C"},
    )


def parse_dnf_transaction(text: str) -> dict[str, list[Any]]:
    """The package rows of a ``dnf --assumeno`` transaction table.

    Rows are read from the first section header (``Installing:``,
    ``Upgrading:``, ``Installing dependencies:``, ``Removing dependent
    packages:`` ...) to ``Transaction Summary``; dnf does not wrap rows when its
    output is a pipe. A ``replacing`` row (an obsolete) and every line that is
    not a five-column package row are kept so the caller refuses them -- an
    unreadable plan is never taken for an empty one.
    """

    entries: list[dict[str, str]] = []
    unparsed: list[str] = []
    section: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("Transaction Summary"):
            break
        if not line.strip():
            continue
        header = DNF_SECTION.match(line)
        if header is not None:
            section = header.group(1)
            continue
        if section is None:
            continue
        tokens = line.split()
        indented = raw[:1].isspace()
        if indented and tokens[0] == "replacing" and len(tokens) >= 3:
            entries.append(
                {"section": "Replacing", "name": tokens[1], "version": tokens[2]}
            )
        elif indented and len(tokens) >= 5:
            entries.append(
                {
                    "section": section,
                    "name": tokens[0],
                    "arch": tokens[1],
                    "version": tokens[2],
                    "repository": tokens[3],
                }
            )
        else:
            unparsed.append(line.strip()[:160])
    return {"entries": entries, "unparsed": unparsed}


def entry_nevra(entry: dict[str, str]) -> str:
    if entry.get("arch"):
        return nevra(entry["name"], entry["version"], entry["arch"])
    return f"{entry['name']} {entry['version']}"


def first_error(completed: subprocess.CompletedProcess[str]) -> str:
    for line in f"{completed.stdout}\n{completed.stderr}".splitlines():
        if line.strip().startswith(("Error", "Problem")):
            return line.strip()[:200]
    return (completed.stderr or completed.stdout).strip()[-200:]


def install_refusal(
    completed: subprocess.CompletedProcess[str],
    plan: dict[str, list[Any]],
    wanted: dict[str, str | None],
) -> str | None:
    """Why the previewed install must not run (``None``: it may).

    It may run only when every row is ``Installing`` one of ``wanted`` -- each
    exactly once and, for a module package, built for the running kernel
    (``wanted[name]`` is then ``<version>-<release>.<arch>``, ``uname -r``).
    """

    output = f"{completed.stdout}\n{completed.stderr}"
    unmatched = sorted(set(re.findall(r"No match for argument: (\S+)", output)))
    if unmatched:
        return f"no package {', '.join(unmatched)} in the enabled repositories"
    if plan["unparsed"]:
        return "unreadable dnf transaction rows: " + "; ".join(plan["unparsed"][:3])
    entries = plan["entries"]
    kernels = [item for item in entries if KERNEL_PACKAGE.fullmatch(item["name"])]
    if kernels:
        return "dnf would pull a kernel: " + ", ".join(map(entry_nevra, kernels))
    extra = [
        f"{item['section'].lower()} {entry_nevra(item)}"
        for item in entries
        if item["section"] != "Installing" or item["name"] not in wanted
    ]
    if extra:
        return (
            f"dnf would also {'; '.join(extra)} "
            f"(only {', '.join(sorted(wanted))} may be installed)"
        )
    if not entries and completed.returncode and "Error" in output:
        return f"dnf could not plan the install: {first_error(completed)}"
    names = [item["name"] for item in entries]
    missing = sorted(set(wanted) - set(names))
    if missing:
        return f"dnf plans no install of {', '.join(missing)}"
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        return f"dnf plans more than one {', '.join(repeated)}"
    for item in entries:
        release = wanted[item["name"]]
        if release is not None and f"{item['version']}.{item['arch']}" != release:
            return (
                f"dnf would install {entry_nevra(item)}, not the running "
                f"kernel's {item['name']}-{release}"
            )
    return None


def removal_refusal(plan: dict[str, list[Any]], present: list[str]) -> str | None:
    """Why the previewed removal must not run: it may only remove ``present``."""

    if plan["unparsed"]:
        return "unreadable dnf removal rows: " + "; ".join(plan["unparsed"][:3])
    rows = [entry_nevra(item) for item in plan["entries"]]
    extra = [
        f"{item['section'].lower()} {row}"
        for item, row in zip(plan["entries"], rows)
        if item["section"] != "Removing" or row not in present
    ]
    if extra:
        return (
            f"dnf would also {'; '.join(extra)}; a package this run did not "
            "install is never removed"
        )
    if sorted(rows) != sorted(present):
        return (
            f"dnf plans to remove {', '.join(sorted(rows)) or 'nothing'} instead "
            f"of {', '.join(sorted(present))}"
        )
    return None


def installed_by_run(
    before: dict[str, dict[str, str]],
    after: dict[str, dict[str, str]],
    planned: list[str],
) -> tuple[list[str], list[str]]:
    """The packages this run's transaction added, and what did not go to plan.

    Only packages new since ``before`` *and* from the rpm transaction that
    installed a planned package count as this run's: a package someone else
    installed meanwhile is never recorded, so never removed.
    """

    added = sorted(set(after) - set(before))
    transactions = {after[item]["tid"] for item in planned if item in after}
    ours = [item for item in added if after[item]["tid"] in transactions]
    problems = []
    missing = [item for item in planned if item not in after]
    if missing:
        problems.append("dnf did not install " + ", ".join(missing))
    unexpected = [item for item in ours if item not in planned]
    if unexpected:
        problems.append("dnf installed more than previewed: " + ", ".join(unexpected))
    lost = sorted(set(before) - set(after))
    if lost:
        problems.append(f"the installed set lost {', '.join(lost)} during the install")
    return ours, problems


def install_plan(
    wanted: dict[str, str | None], result: dict[str, Any]
) -> tuple[list[str], dict[str, dict[str, str]], tuple[str, str] | None]:
    """The exact NEVRAs to install, the rpm baseline, or a refusal."""

    if not (tool_present("dnf") and tool_present("rpm")):
        missing = ", ".join(sorted(wanted))
        return [], {}, ("CAPABILITY_ABSENT", f"no dnf/rpm on the node for {missing}")
    before = rpm_packages()
    present = sorted(
        item
        for item, info in before.items()
        if info["name"] in wanted
        and (
            wanted[info["name"]] is None
            or item == f"{info['name']}-{wanted[info['name']]}"
        )
    )
    if present:
        detail = f"{', '.join(present)} is installed but still unusable; not touched"
        return [], before, ("FIXTURE_NOT_READY", detail)
    specs = [
        name if version is None else f"{name}-{version}"
        for name, version in wanted.items()
    ]
    preview = dnf(["--assumeno", *DNF_INSTALL_OPTIONS, "install", *specs])
    plan = parse_dnf_transaction(preview.stdout)
    result["transaction"] = plan["entries"]
    refusal = install_refusal(preview, plan, wanted)
    if refusal is not None:
        return [], before, ("FIXTURE_NOT_READY", refusal)
    planned = [
        nevra(item["name"], item["version"], item["arch"]) for item in plan["entries"]
    ]
    return planned, before, None


def ensure(arguments: argparse.Namespace) -> None:
    """Make the named tools and modules usable, changing only what is missing."""

    run_id = checked_run_id(arguments.run_id)
    seconds = bounded(
        arguments.deadman_seconds, MIN_DEADMAN_SECONDS, MAX_DEADMAN_SECONDS, "deadman"
    )
    tools = sorted(set(arguments.tool))
    modules = sorted(set(arguments.module))
    if any(item not in TOOL_PACKAGES for item in tools) or any(
        item not in MODULE_PACKAGES for item in modules
    ):
        raise ProbeError("tool or module outside the reviewed allow-list")
    release = kernel_release()
    before_tools = {name: tool_present(name) for name in tools}
    before_modules = {name: module_state(name) for name in modules}
    result: dict[str, Any] = {
        "kernel": release,
        "tools_before": before_tools,
        "modules_before": before_modules,
        "transaction": [],
        "deadman": None,
        "refused": None,
    }

    def finish(refused: tuple[str, str] | None = None) -> None:
        state = load_state(run_id)
        result["installed"] = list(state["installed"])
        result["modules_loaded_by_run"] = list(state["modules"])
        result["tools"] = {name: tool_present(name) for name in tools}
        result["modules"] = {name: module_state(name) for name in modules}
        if refused is not None:
            result["refused"] = {"reason": refused[0], "detail": refused[1]}
        emit(result)

    wanted: dict[str, str | None] = {
        TOOL_PACKAGES[name]: None for name in tools if not before_tools[name]
    }
    for name in modules:
        if not before_modules[name]["loadable"]:
            wanted[MODULE_PACKAGES[name]] = release
    to_load = [
        name
        for name in modules
        if not (before_modules[name]["loaded"] or before_modules[name]["builtin"])
    ]
    if wanted and not arguments.allow_install:
        lacks = [name for name in tools if not before_tools[name]] + [
            f"a loadable {name} for kernel {release}"
            for name in modules
            if not before_modules[name]["loadable"]
        ]
        return finish(
            (
                "CAPABILITY_ABSENT",
                f"the node lacks {' and '.join(lacks)}; installing "
                f"{', '.join(sorted(wanted))} needs TEMP_TEST_PACKAGE_INSTALL",
            )
        )
    if to_load and not arguments.allow_module_load:
        return finish(
            (
                "NOT_AUTHORIZED",
                f"kernel module {', '.join(to_load)} is not loaded; needs "
                "KERNEL_MODULE_LOAD",
            )
        )
    if not wanted and not to_load:
        return finish()  # everything is there: nothing touched, nothing recorded
    planned: list[str] = []
    before: dict[str, dict[str, str]] = {}
    if wanted:
        planned, before, refused = install_plan(wanted, result)
        if refused is not None:
            return finish(refused)
    # From here on the node changes: the deadman first, then the record.
    result["deadman"] = arm_deadman(run_id, seconds)
    if planned:
        state = load_state(run_id)
        state["pending"] = sorted(set(state["pending"]) | set(planned))
        save_state(run_id, state)
        completed = dnf(["-y", *DNF_INSTALL_OPTIONS, "install", *planned])
        ours, problems = installed_by_run(before, rpm_packages(), planned)
        state = load_state(run_id)
        state["installed"] = sorted(set(state["installed"]) | set(ours))
        state["pending"] = []
        save_state(run_id, state)
        if completed.returncode:
            problems.insert(0, f"dnf install ended {completed.returncode}")
            problems.insert(1, first_error(completed))
        missing = [name for name in tools if not tool_present(name)]
        if missing:
            problems.append(f"{', '.join(missing)} still missing after the install")
        if problems:
            return finish(("FIXTURE_NOT_READY", "; ".join(problems)))
    for name in to_load:
        state = load_state(run_id)
        # Recorded before the load: it was not loaded, so loaded later is ours.
        if name not in state["modules"]:
            state["modules"].append(name)
        save_state(run_id, state)
        loaded = run(["modprobe", name], check=False, timeout=60)
        if not (SYS_MODULE / name).exists():
            detail = f"{name} could not be loaded ({loaded.returncode}): "
            return finish(("FIXTURE_NOT_READY", detail + loaded.stderr.strip()[:200]))
    finish()


def unload_modules(state: dict[str, Any], report: dict[str, Any]) -> None:
    """Unload each module this run loaded, once nothing uses it any more."""

    kept = []
    for name in state["modules"]:
        root = SYS_MODULE / str(name)
        if not root.exists():
            continue
        try:
            users = (root / "refcnt").read_text().strip()
        except OSError:
            users = "0"
        if users == "0":
            run(["modprobe", "-r", str(name)], check=False, timeout=60)
        if root.exists():
            report["errors"].append(f"{name} is still loaded (refcnt {users})")
            kept.append(name)
        else:
            report["unloaded"].append(name)
    state["modules"] = kept


def remove_packages(state: dict[str, Any], report: dict[str, Any]) -> None:
    """``dnf remove`` exactly the recorded packages that are still installed."""

    candidates = sorted(set(state["installed"]) | set(state["pending"]))
    if not candidates:
        return
    if not (tool_present("dnf") and tool_present("rpm")):
        report["errors"].append("no dnf/rpm to remove " + ", ".join(candidates))
        return
    current = rpm_packages()
    present = [item for item in candidates if item in current]
    kept = list(present)
    if present:
        preview = dnf(["--assumeno", *DNF_REMOVE_OPTIONS, "remove", *present])
        refusal = removal_refusal(parse_dnf_transaction(preview.stdout), present)
        if refusal is not None:
            report["errors"].append(refusal)
        else:
            done = dnf(["-y", *DNF_REMOVE_OPTIONS, "remove", *present])
            after = rpm_packages()
            kept = [item for item in present if item in after]
            report["removed"] = [item for item in present if item not in after]
            lost = sorted(set(current) - set(after) - set(present))
            if lost:
                report["errors"].append(
                    "the installed set also lost " + ", ".join(lost)
                )
            if done.returncode or kept:
                report["errors"].append(
                    f"dnf remove ended {done.returncode}: {first_error(done)}"
                )
    state["installed"] = kept
    state["pending"] = []


def residue(run_id: str) -> dict[str, Any]:
    state = load_state(run_id)
    recorded = sorted(set(state["installed"]) | set(state["pending"]))
    if recorded and tool_present("rpm"):
        current = rpm_packages()
        recorded = [item for item in recorded if item in current]
    return {
        "packages": recorded,
        "modules": [name for name in state["modules"] if (SYS_MODULE / name).exists()],
    }


def residue_is_clean(value: dict[str, Any]) -> bool:
    return not value["packages"] and not value["modules"]


def cleanup(arguments: argparse.Namespace) -> None:
    """Undo this run's module load and package installs; never anything else.

    The module goes first (a package cannot take a loaded module's file away
    cleanly); what could not be undone stays recorded, so the residue says so,
    and the deadman stays armed to retry.
    """

    run_id = checked_run_id(arguments.run_id)
    state = load_state(run_id)
    report: dict[str, Any] = {"unloaded": [], "removed": [], "errors": []}
    unload_modules(state, report)
    save_state(run_id, state)
    remove_packages(state, report)
    save_state(run_id, state)
    remaining = residue(run_id)
    clean = residue_is_clean(remaining)
    if clean:
        disarm_deadman(run_id)
    emit({"run_id": run_id, "report": report, "residue": remaining, "clean": clean})


def show_residue(arguments: argparse.Namespace) -> None:
    value = residue(checked_run_id(arguments.run_id))
    emit({"residue": value, "clean": residue_is_clean(value)})


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)

    def command(name: str, handler: Any) -> argparse.ArgumentParser:
        item = commands.add_parser(name)
        item.add_argument("--run-id", required=True)
        item.set_defaults(handler=handler)
        return item

    item = command("ensure", ensure)
    item.add_argument("--tool", action="append", default=[], choices=TOOL_PACKAGES)
    item.add_argument("--module", action="append", default=[], choices=MODULE_PACKAGES)
    item.add_argument("--deadman-seconds", type=int, required=True)
    item.add_argument("--allow-install", action="store_true")
    item.add_argument("--allow-module-load", action="store_true")
    command("residue", show_residue)
    command("cleanup", cleanup)
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        arguments.handler(arguments)
    except Exception as exc:  # noqa: BLE001 - reported as the probe's JSON error
        emit(
            {
                "error": type(exc).__name__,
                "error_kind": "probe",
                "error_class": type(exc).__name__,
                "command": arguments.command,
            }
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
