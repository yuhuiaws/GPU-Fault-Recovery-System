#!/usr/bin/env python3
"""Temporary test packages of one GF-GENERAL-LIVE run (TEMP_TEST_PACKAGE_INSTALL).

Runs on the target node through ``host_probe_fixture.py`` (chroot into the
host, the node's own Python), standard library only, one JSON object on the
last stdout line.

Some variants need a host tool or kernel module the node image does not ship:
GF-GENERAL-LIVE-009 ``tcp_only`` needs ``tc`` (the AL2023 EKS AMI has no
``iproute-tc``) and the ``sch_netem`` qdisc, the 007 SMART replay variants
need ``smartctl`` (``smartmontools``) and 004 ``probe_error`` needs
``mkfs.vfat`` (``dosfstools``). ``ensure`` makes exactly those usable and
changes nothing that is already there (the runner-side step is
``general_live_deps.ensure_dependencies``):

* a missing tool or a module the running kernel cannot load is installed with
  ``dnf`` (``--allow-install``) -- only the reviewed package of
  ``TOOL_PACKAGES``/``MODULE_PACKAGES``, the module package always for the
  *running* kernel (``kernel-modules-extra-$(uname -r)``). The transaction is
  previewed first (``--assumeno``) and refused unless every row is
  ``Installing`` one of those packages or ``Installing dependencies`` one of
  their reviewed closures (``PACKAGE_CLOSURES``: on AL2023 ``smartmontools``
  carries ``smartmontools-selinux``, from the same build): no kernel, upgrade,
  downgrade, reinstall, removal, obsolete, undeclared dependency or unreadable
  row -- the refusal names the unexpected packages;
* a module that is neither loaded nor built in is loaded with ``modprobe``
  (``--allow-module-load``, the ``KERNEL_MODULE_LOAD`` approval).

What the run installs (NEVRAs in install order, only from the rpm transaction
that installed them), which module it loads and what the packages brought
beyond their files (``PACKAGE_FOOTPRINT``: a systemd unit the preset enabled, a
SELinux policy module a scriptlet loaded) are written to
``/run/gpu-fault-acceptance/general/<digest>/deps-state.json`` *before* the
change, after a ``systemd-run --on-active`` deadman is armed that runs
``cleanup`` from a copy of this file. A unit the install brought is disabled
and stopped at once: the fixture needs the tool, never the daemon. ``cleanup``
unloads only a module this run loaded (and only once nothing uses it), quiets
its units again and removes only the recorded packages, last installed first
(a package before its dependency), each through a previewed ``dnf remove`` that
must remove exactly it (never a dependent package or an "unused" dependency);
the residue proves the packages gone, the units neither active nor enabled and
the policy module unloaded. A refusal is a normal answer (``refused`` with a
NOT_RUN reason), never an error, so the runner can record why the variant did
not run.
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
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_ROOT = Path("/run/gpu-fault-acceptance/general")
UNIT_PREFIX = "gfrs-"
SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")
MIN_DEADMAN_SECONDS = 60
MAX_DEADMAN_SECONDS = 5400
#: The only tools this probe may install a package for, and that package: the
#: package AL2023 ships the tool in. dnf may add nothing beyond the package and
#: its reviewed closure (``PACKAGE_CLOSURES``); a transaction that pulls more is
#: refused, naming what it would pull.
TOOL_PACKAGES = {
    "tc": "iproute-tc",
    "smartctl": "smartmontools",
    "mkfs.vfat": "dosfstools",
}
#: The reviewed dependency closure of a package: the packages dnf may install
#: with it (``Installing dependencies`` rows), each reviewed on its own and
#: accepted only from the same build. ``smartmontools`` carries
#: ``smartmontools-selinux`` on AL2023: the rpm requires
#: ``(smartmontools-selinux if selinux-policy-targeted)`` and the node image
#: ships ``selinux-policy-targeted`` (AL2023 runs SELinux permissive), so dnf
#: adds the subpackage -- the same source rpm's "Custom SELinux policy module
#: for smartmontools" (noarch, 23 k). Its scriptlets load one policy module,
#: relabel only the file contexts that changed and mark ``smartd.service`` for
#: a restart (nothing, while the daemon is not running); it runs nothing
#: itself, and its own requirements (policycoreutils-python-utils,
#: selinux-policy-targeted, libselinux-utils) are already on the node, so the
#: live previews list exactly these two rows.
PACKAGE_CLOSURES: dict[str, tuple[str, ...]] = {
    "smartmontools": ("smartmontools-selinux",),
}
#: dnf lists what it pulls for a requested package under this header; a closure
#: package is accepted there and nowhere else.
DEPENDENCY_SECTION = "Installing dependencies"
#: What a reviewed package leaves beyond its files, claimed when the install
#: brings it, quieted at once and proven gone by the residue. ``smartmontools``
#: ships ``smartd.service`` and AL2023's ``90-default.preset`` enables it when
#: the package's ``%post`` runs ``systemd-update-helper install-system-units``
#: (a preset, no start); the fixture needs ``smartctl``, never the daemon, so
#: the unit is disabled (and stopped, were it active) right after the install
#: and the package's ``%preun`` disables it again on removal.
#: ``smartmontools-selinux`` loads its ``smartmon`` policy module at priority
#: 200 (``semodule -X 200 -i smartmon.pp.bz2``; ``%postun`` removes it) over
#: the ``smartmon`` module the base ``selinux-policy-targeted`` ships at
#: priority 100, so the footprint is the ``semodule --list-modules=full`` row
#: ``200 smartmon``, never the base policy's.
PACKAGE_FOOTPRINT: dict[str, dict[str, tuple[str, ...]]] = {
    "smartmontools": {"units": ("smartd.service",)},
    "smartmontools-selinux": {"selinux_modules": ("200 smartmon",)},
}
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
#: A dnf section header's first word as the verb of a refusal ("Installing
#: dependencies" -> "install the dependencies").
SECTION_VERBS = {
    "Installing": "install",
    "Upgrading": "upgrade",
    "Downgrading": "downgrade",
    "Reinstalling": "reinstall",
    "Removing": "remove",
    "Replacing": "replace",
    "Skipping": "skip",
}
UNIT_ACTIVE_STATES = frozenset({"active", "activating", "reloading"})
UNIT_ENABLED_STATES = frozenset({"enabled", "enabled-runtime"})
SYS_MODULE = Path("/sys/module")
MODULES_ROOT = Path("/lib/modules")
EMPTY_STATE: dict[str, Any] = {
    "installed": [],
    "pending": [],
    "modules": [],
    "units": [],
    "selinux_modules": [],
}


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
        [
            "systemctl",
            "show",
            unit,
            "--property=LoadState",
            "--property=ActiveState",
            "--property=UnitFileState",
            "--property=MainPID",
        ],
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


def section_phrase(section: str) -> str:
    """A dnf section header as what dnf would do: ``install the dependencies``."""

    head, _, rest = section.partition(" ")
    verb = SECTION_VERBS.get(head, head.lower())
    return f"{verb} the {rest}" if rest else verb


def rows_phrase(items: list[dict[str, str]]) -> str:
    """``install the dependencies a, b; upgrade c``: rows grouped by action."""

    groups: dict[str, list[str]] = {}
    for item in items:
        groups.setdefault(section_phrase(item["section"]), []).append(entry_nevra(item))
    return "; ".join(f"{verb} {', '.join(rows)}" for verb, rows in groups.items())


def closure_of(names: Iterable[str]) -> dict[str, str]:
    """Each reviewed dependency of ``names`` -> the package it belongs to."""

    return {dep: name for name in names for dep in PACKAGE_CLOSURES.get(name, ())}


def allowed_label(names: Iterable[str]) -> str:
    """``smartmontools with its reviewed dependency smartmontools-selinux``."""

    parts = []
    for name in sorted(set(names)):
        closure = PACKAGE_CLOSURES.get(name, ())
        if closure:
            noun = "dependency" if len(closure) == 1 else "dependencies"
            parts.append(f"{name} with its reviewed {noun} {', '.join(closure)}")
        else:
            parts.append(name)
    return ", ".join(parts)


def install_refusal(
    completed: subprocess.CompletedProcess[str],
    plan: dict[str, list[Any]],
    wanted: dict[str, str | None],
) -> str | None:
    """Why the previewed install must not run (``None``: it may).

    It may run only when every row is ``Installing`` one of ``wanted`` -- each
    exactly once and, for a module package, built for the running kernel
    (``wanted[name]`` is then ``<version>-<release>.<arch>``, ``uname -r``) --
    or ``Installing dependencies`` one of their reviewed closures, from the
    same build as the package it belongs to. Anything else is named.
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
    closure = closure_of(wanted)
    extra = [
        item
        for item in entries
        if not (
            (item["section"] == "Installing" and item["name"] in wanted)
            or (item["section"] == DEPENDENCY_SECTION and item["name"] in closure)
        )
    ]
    if extra:
        return (
            f"dnf would also {rows_phrase(extra)} "
            f"(only {allowed_label(wanted)} may be installed)"
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
    builds = {
        item["name"]: item["version"] for item in entries if item["name"] in wanted
    }
    for item in entries:
        if item["name"] in closure:
            parent = closure[item["name"]]
            if item["version"] != builds[parent]:
                return (
                    f"dnf would install {entry_nevra(item)}, not the build of "
                    f"{parent}-{builds[parent]}"
                )
            continue
        release = wanted[item["name"]]
        if release is not None and f"{item['version']}.{item['arch']}" != release:
            return (
                f"dnf would install {entry_nevra(item)}, not the running "
                f"kernel's {item['name']}-{release}"
            )
    return None


def install_order(
    entries: list[dict[str, str]], wanted: dict[str, str | None]
) -> list[dict[str, str]]:
    """The accepted rows as rpm installs them: a package after its closure."""

    by_name = {item["name"]: item for item in entries}
    ordered: list[dict[str, str]] = []
    for name in wanted:
        ordered.extend(
            by_name[dep] for dep in PACKAGE_CLOSURES.get(name, ()) if dep in by_name
        )
        if name in by_name:
            ordered.append(by_name[name])
    return ordered


def removal_refusal(plan: dict[str, list[Any]], present: list[str]) -> str | None:
    """Why the previewed removal must not run: it may only remove ``present``."""

    if plan["unparsed"]:
        return "unreadable dnf removal rows: " + "; ".join(plan["unparsed"][:3])
    rows = [entry_nevra(item) for item in plan["entries"]]
    extra = [
        item
        for item, row in zip(plan["entries"], rows)
        if item["section"] != "Removing" or row not in present
    ]
    if extra:
        return (
            f"dnf would also {rows_phrase(extra)}; a package this run did not "
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
    installed meanwhile is never recorded, so never removed. The planned ones
    come first, in install order (a package after its dependency), so a later
    removal can walk them backwards.
    """

    added = set(after) - set(before)
    transactions = {after[item]["tid"] for item in planned if item in after}
    ours = [item for item in planned if item in added] + sorted(
        item for item in added - set(planned) if after[item]["tid"] in transactions
    )
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
    ordered = install_order(plan["entries"], wanted)
    # What the packages will bring beyond their files, as the node has it now:
    # only what is new afterwards is this run's to quiet and to prove gone.
    footprint = footprint_of(item["name"] for item in ordered)
    result["footprint"] = {"before": footprint_state(footprint)}
    planned = [nevra(item["name"], item["version"], item["arch"]) for item in ordered]
    return planned, before, None


def unit_footprint(unit: str) -> dict[str, bool]:
    """Whether ``unit`` has a unit file, is active, is enabled."""

    values = unit_state(unit)
    return {
        "loaded": values.get("LoadState") == "loaded",
        "active": values.get("ActiveState") in UNIT_ACTIVE_STATES,
        "enabled": values.get("UnitFileState") in UNIT_ENABLED_STATES,
    }


def selinux_module_present(module: str) -> bool:
    """Whether the SELinux module store lists ``module`` (``<priority> <name>``)."""

    if not tool_present("semodule"):
        return False
    listing = run(["semodule", "--list-modules=full"], check=False, timeout=120)
    return any(
        " ".join(line.split()[:2]) == module for line in listing.stdout.splitlines()
    )


def footprint_of(names: Iterable[str]) -> dict[str, list[str]]:
    """The units and SELinux modules the named packages bring."""

    units: list[str] = []
    modules: list[str] = []
    for name in names:
        units.extend(PACKAGE_FOOTPRINT.get(name, {}).get("units", ()))
        modules.extend(PACKAGE_FOOTPRINT.get(name, {}).get("selinux_modules", ()))
    return {"units": units, "selinux_modules": modules}


def footprint_state(footprint: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "units": {unit: unit_footprint(unit) for unit in footprint["units"]},
        "selinux_modules": {
            name: selinux_module_present(name) for name in footprint["selinux_modules"]
        },
    }


def quiet_unit(unit: str) -> tuple[bool, dict[str, bool]]:
    """Disable, and stop, a unit a package of this run brought.

    Returns whether anything had to be done, and the unit's state afterwards.
    """

    state = unit_footprint(unit)
    if state["active"]:
        run(["systemctl", "stop", unit], check=False, timeout=90)
    if state["enabled"]:
        run(["systemctl", "disable", unit], check=False, timeout=90)
    if not (state["active"] or state["enabled"]):
        return False, state
    return True, unit_footprint(unit)


def unit_problem(unit: str, state: dict[str, bool], moment: str = "") -> str | None:
    if state["active"]:
        return f"{unit} is still active{moment}"
    if state["enabled"]:
        return f"{unit} is still enabled{moment}"
    return None


def claim_footprint(run_id: str, result: dict[str, Any]) -> list[str]:
    """Record what the install brought beyond its files, and quiet the units.

    A unit that had no unit file before and has one now, and a policy module
    the store did not list before and lists now, are this run's: recorded, so
    cleanup and residue check them; never claimed when the node had them.
    """

    before = result["footprint"]["before"]
    after = footprint_state(
        {
            "units": list(before["units"]),
            "selinux_modules": list(before["selinux_modules"]),
        }
    )
    state = load_state(run_id)
    for unit, was in before["units"].items():
        if not was["loaded"] and after["units"][unit]["loaded"]:
            if unit not in state["units"]:
                state["units"].append(unit)
    for name, was in before["selinux_modules"].items():
        if not was and after["selinux_modules"][name]:
            if name not in state["selinux_modules"]:
                state["selinux_modules"].append(name)
    save_state(run_id, state)
    quieted: list[str] = []
    problems: list[str] = []
    for unit in state["units"]:
        acted, after["units"][unit] = quiet_unit(unit)
        if acted:
            quieted.append(unit)
        problem = unit_problem(unit, after["units"][unit], " after the install")
        if problem is not None:
            problems.append(problem)
    result["footprint"] = {
        "before": before,
        "after": after,
        "quieted": quieted,
        "claimed": {
            "units": list(state["units"]),
            "selinux_modules": list(state["selinux_modules"]),
        },
    }
    return problems


def perform_install(
    run_id: str,
    planned: list[str],
    before: dict[str, dict[str, str]],
    result: dict[str, Any],
) -> list[str]:
    """Install the previewed NEVRAs, record what arrived, quiet what it brought."""

    state = load_state(run_id)
    state["pending"] = list(state["pending"]) + [
        item for item in planned if item not in state["pending"]
    ]
    save_state(run_id, state)
    completed = dnf(["-y", *DNF_INSTALL_OPTIONS, "install", *planned])
    ours, problems = installed_by_run(before, rpm_packages(), planned)
    state = load_state(run_id)
    state["installed"] = list(state["installed"]) + [
        item for item in ours if item not in state["installed"]
    ]
    state["pending"] = []
    save_state(run_id, state)
    if completed.returncode:
        problems.insert(0, f"dnf install ended {completed.returncode}")
        problems.insert(1, first_error(completed))
    problems.extend(claim_footprint(run_id, result))
    return problems


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
        "footprint": None,
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
                f"{allowed_label(wanted)} needs TEMP_TEST_PACKAGE_INSTALL",
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
        problems = perform_install(run_id, planned, before, result)
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


def quiet_units(state: dict[str, Any], report: dict[str, Any]) -> None:
    """Disable and stop each unit this run's packages brought, before they go."""

    for unit in state["units"]:
        acted, after = quiet_unit(unit)
        if acted:
            report["quieted"].append(unit)
        problem = unit_problem(unit, after)
        if problem is not None:
            report["errors"].append(problem)


def recorded_packages(state: dict[str, Any]) -> list[str]:
    """Every NEVRA this run recorded, in install order (pending ones last)."""

    installed = list(state["installed"])
    return installed + [item for item in state["pending"] if item not in installed]


def remove_packages(state: dict[str, Any], report: dict[str, Any]) -> None:
    """``dnf remove`` the recorded packages still installed, last installed first.

    One previewed transaction per package, walking the record backwards (a
    package before its reviewed dependency, which nothing needs once it is
    gone); each may remove exactly that package and nothing else.
    """

    ordered = recorded_packages(state)
    if not ordered:
        return
    if not (tool_present("dnf") and tool_present("rpm")):
        report["errors"].append("no dnf/rpm to remove " + ", ".join(ordered))
        return
    current = rpm_packages()
    kept: list[str] = []
    for item in reversed(ordered):
        if item not in current:
            continue
        preview = dnf(["--assumeno", *DNF_REMOVE_OPTIONS, "remove", item])
        refusal = removal_refusal(parse_dnf_transaction(preview.stdout), [item])
        if refusal is not None:
            report["errors"].append(refusal)
            kept.append(item)
            continue
        done = dnf(["-y", *DNF_REMOVE_OPTIONS, "remove", item])
        after = rpm_packages()
        lost = sorted(set(current) - set(after) - {item})
        if lost:
            report["errors"].append("the installed set also lost " + ", ".join(lost))
        if item in after or done.returncode:
            report["errors"].append(
                f"dnf remove {item} ended {done.returncode}: {first_error(done)}"
            )
        if item in after:
            kept.append(item)
        else:
            report["removed"].append(item)
        current = after
    state["installed"] = [item for item in ordered if item in kept]
    state["pending"] = []


def residue(run_id: str) -> dict[str, Any]:
    """What of this run is still on the node: packages, modules, units, policy."""

    state = load_state(run_id)
    recorded = recorded_packages(state)
    if recorded and tool_present("rpm"):
        current = rpm_packages()
        recorded = [item for item in recorded if item in current]
    return {
        "packages": recorded,
        "modules": [name for name in state["modules"] if (SYS_MODULE / name).exists()],
        "units": [
            unit
            for unit in state["units"]
            if unit_problem(unit, unit_footprint(unit)) is not None
        ],
        "selinux_modules": [
            name for name in state["selinux_modules"] if selinux_module_present(name)
        ],
    }


def residue_is_clean(value: dict[str, Any]) -> bool:
    return not any(value.values())


def cleanup(arguments: argparse.Namespace) -> None:
    """Undo this run's module load and package installs; never anything else.

    The module goes first (a package cannot take a loaded module's file away
    cleanly), then the units the packages brought are quieted, then the
    packages go, last installed first; what could not be undone stays recorded,
    so the residue says so, and the deadman stays armed to retry.
    """

    run_id = checked_run_id(arguments.run_id)
    state = load_state(run_id)
    report: dict[str, Any] = {
        "unloaded": [],
        "quieted": [],
        "removed": [],
        "errors": [],
    }
    unload_modules(state, report)
    save_state(run_id, state)
    quiet_units(state, report)
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
