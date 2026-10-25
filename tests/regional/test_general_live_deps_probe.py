"""The temporary test package probe (TEMP_TEST_PACKAGE_INSTALL).

A recording fake node stands in for dnf, rpm, modprobe, systemd and semodule:
the tests prove the probe installs only what is missing (009 ``tc`` ->
``iproute-tc``, 007 ``smartctl`` -> ``smartmontools`` with its reviewed closure
``smartmontools-selinux``, 004 ``mkfs.vfat`` -> ``dosfstools``), refuses any dnf
transaction that would pull more than the reviewed packages for the running
kernel (naming the unexpected ones), records what it installed, loaded and
brought (the preset-enabled ``smartd.service``, the ``smartmon`` policy module)
before doing it, quiets the unit at once, and that its cleanup (also the
deadman's command) removes and unloads exactly that, last installed first --
never a package, a module or a policy the run found on the node.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_deps_probe as deps

RUN_ID = "gl009-tcp-only-a1-deadbeef"
KERNEL = "6.1.100-100.100.amzn2023.x86_64"
TC_ROW = ("iproute-tc", "x86_64", "5.10.0-2.amzn2023.0.5", "amazonlinux")
TC_NEVRA = "iproute-tc-5.10.0-2.amzn2023.0.5.x86_64"
KMOD_ROW = ("kernel-modules-extra", "x86_64", "6.1.100-100.100.amzn2023", "amazonlinux")
KMOD_NEVRA = f"kernel-modules-extra-{KERNEL}"
#: The AL2023 build the live nodes previewed: smartmontools and, pulled by its
#: rich dependency ``(smartmontools-selinux if selinux-policy-targeted)``, the
#: same build's SELinux policy subpackage.
SMART_ROW = ("smartmontools", "x86_64", "1:7.2-11.amzn2023.0.1", "amazonlinux")
SMART_NEVRA = "smartmontools-1:7.2-11.amzn2023.0.1.x86_64"
SELINUX_ROW = (
    "smartmontools-selinux",
    "noarch",
    "1:7.2-11.amzn2023.0.1",
    "amazonlinux",
)
SELINUX_NEVRA = "smartmontools-selinux-1:7.2-11.amzn2023.0.1.noarch"
SMART_CLOSURE = [SELINUX_NEVRA, SMART_NEVRA]  # install order: the dependency first
DOSFS_ROW = ("dosfstools", "x86_64", "4.2-3.amzn2023.0.2", "amazonlinux")
DOSFS_NEVRA = "dosfstools-4.2-3.amzn2023.0.2.x86_64"
#: Packages the node already has: the probe must never touch them.
BASELINE = {
    "iproute-5.10.0-2.amzn2023.0.5.x86_64": ("iproute", "5.10.0", "2.amzn2023.0.5"),
    "bash-5.2.15-1.amzn2023.0.2.x86_64": ("bash", "5.2.15", "1.amzn2023.0.2"),
    "kernel-modules-extra-6.1.90-90.90.amzn2023.x86_64": (
        "kernel-modules-extra",
        "6.1.90",
        "90.90.amzn2023",
    ),
}


def dnf_table(sections: dict[str, list[tuple[str, str, str, str]]], verb: str) -> str:
    """A ``dnf --assumeno`` transaction as dnf4 prints it into a pipe."""

    lines = [
        "Last metadata expiration check: 0:01:02 ago on Sat Oct 25 03:00:00 2026.",
        "Dependencies resolved.",
        "=" * 80,
        " Package            Architecture  Version                 Repository   Size",
        "=" * 80,
    ]
    for section, rows in sections.items():
        lines.append(f"{section}:")
        lines.extend(
            f" {name:<20} {arch:<8} {version:<26} {repo:<12} 455 k"
            for name, arch, version, repo in rows
        )
    lines += [
        "",
        "Transaction Summary",
        "=" * 80,
        f"{verb}  {sum(map(len, sections.values()))} Package",
        "",
        "Operation aborted.",
    ]
    return "\n".join(lines) + "\n"


class FakeNode:
    """dnf, rpm, modprobe and systemd of one node; every command is recorded."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        installed: dict[str, tuple[str, str, str]] | None = None,
        loaded: set[str] = frozenset(),
        loadable: set[str] = frozenset(),
        module_files_from: str | None = None,
    ) -> None:
        self.calls: list[list[str]] = []
        self.sys_module = tmp_path / "sys-module"
        self.sys_module.mkdir()
        self.packages: dict[str, dict[str, str]] = {}
        for item, (name, version, release) in (installed or BASELINE).items():
            self.packages[item] = self._record(name, version, release, "1000")
        for name in loaded:
            self._load(name)
        self.loadable = set(loadable)
        #: The package whose install makes the module loadable.
        self.module_files_from = module_files_from
        self.repo = {
            row[0]: row for row in (TC_ROW, KMOD_ROW, SMART_ROW, SELINUX_ROW, DOSFS_ROW)
        }
        self.install_preview: dict[str, list[tuple[str, str, str, str]]] | None = None
        self.remove_preview: dict[str, list[tuple[str, str, str, str]]] | None = None
        self.unmatched: list[str] = []
        self.next_tid = 2000
        #: Units with a unit file, as ``systemctl show`` reports them.
        self.units: dict[str, dict[str, bool]] = {}
        #: ``semodule --list-modules=full`` rows; the base selinux-policy-targeted
        #: ships a ``smartmon`` module at priority 100 on every AL2023 node.
        self.selinux_modules: set[str] = {"100 smartmon"}
        #: Footprint knobs: the install also starts the daemon; the unit cannot
        #: be disabled; the subpackage's %postun leaves its policy module behind.
        self.start_units_on_install = False
        self.undisableable = False
        self.sticky_selinux = False
        if self.has("smartmontools"):
            self.units["smartd.service"] = {
                "loaded": True,
                "active": False,
                "enabled": True,
            }
        if self.has("smartmontools-selinux"):
            self.selinux_modules.add("200 smartmon")

    @staticmethod
    def _record(name: str, version: str, release: str, tid: str) -> dict[str, str]:
        return {"name": name, "version": version, "release": release, "tid": tid}

    def _load(self, name: str) -> None:
        (self.sys_module / name).mkdir(exist_ok=True)
        (self.sys_module / name / "refcnt").write_text("0\n")

    def has(self, package: str) -> bool:
        return any(item["name"] == package for item in self.packages.values())

    def tool(self, name: str) -> bool:
        package = deps.TOOL_PACKAGES.get(name)
        if package is not None:
            return self.has(package)
        return name in {
            "dnf",
            "rpm",
            "modprobe",
            "systemd-run",
            "systemctl",
            "semodule",
        }

    def _done(
        self, command: list[str], code: int = 0, stdout: str = "", stderr: str = ""
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, code, stdout=stdout, stderr=stderr)

    def _rows(self, specs: list[str]) -> dict[str, list[tuple[str, str, str, str]]]:
        """dnf's preview for ``specs`` on this node: AL2023 pulls the
        smartmontools-selinux subpackage with smartmontools unless it is there."""

        rows, dependencies = [], []
        for spec in specs:
            name = "kernel-modules-extra" if spec.startswith("kernel-modules") else spec
            rows.append(self.repo[name])
            if name == "smartmontools" and not self.has("smartmontools-selinux"):
                dependencies.append(SELINUX_ROW)
        plan = {"Installing": rows}
        if dependencies:
            plan["Installing dependencies"] = dependencies
        return plan

    def _scriptlets(self, name: str, *, installed: bool) -> None:
        """What the AL2023 rpm scriptlets of the footprint packages do."""

        if name == "smartmontools":
            if installed:  # %post: systemd-update-helper install-system-units -> preset
                self.units["smartd.service"] = {
                    "loaded": True,
                    "active": self.start_units_on_install,
                    "enabled": True,
                }
            else:  # %preun: remove-system-units -> disable --now; the unit file goes
                self.units.pop("smartd.service", None)
        if name == "smartmontools-selinux":
            if installed:  # %post: semodule -X 200 -i smartmon.pp.bz2
                self.selinux_modules.add("200 smartmon")
            elif not self.sticky_selinux:  # %postun: semodule -X 200 -r smartmon
                self.selinux_modules.discard("200 smartmon")

    def dnf_command(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        verb = "install" if "install" in command else "remove"
        targets = command[command.index(verb) + 1 :]
        if "--assumeno" in command and verb == "install":
            if self.unmatched:
                missing = "\n".join(
                    f"No match for argument: {i}" for i in self.unmatched
                )
                return self._done(
                    command, 1, missing + "\n", "Error: Unable to find a match"
                )
            plan = self.install_preview or self._rows(targets)
            return self._done(command, 1, dnf_table(plan, "Install"))
        if "--assumeno" in command:
            plan = self.remove_preview or {
                "Removing": [
                    (
                        self.packages[item]["name"],
                        item.rsplit(".", 1)[1],
                        f"{self.packages[item]['version']}-{self.packages[item]['release']}",
                        "@amazonlinux",
                    )
                    for item in targets
                ]
            }
            return self._done(command, 1, dnf_table(plan, "Remove"))
        if verb == "install":
            tid = str(self.next_tid)
            self.next_tid += 1
            for item in targets:
                for name, arch, version, _repo in self.repo.values():
                    if item == f"{name}-{version}.{arch}":
                        number, release = version.split("-", 1)
                        self.packages[item] = self._record(name, number, release, tid)
                        self._scriptlets(name, installed=True)
                if self.module_files_from and item.startswith(self.module_files_from):
                    self.loadable.add("sch_netem")
            return self._done(command, 0, "Complete!\n")
        for item in targets:
            info = self.packages.pop(item, None)
            if info is not None:
                self._scriptlets(info["name"], installed=False)
        return self._done(command, 0, "Complete!\n")

    def systemctl(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        action, unit = command[1], command[2]
        if action == "show":
            if unit in self.units:
                state = self.units[unit]
                text = (
                    "LoadState=loaded\n"
                    f"ActiveState={'active' if state['active'] else 'inactive'}\n"
                    f"UnitFileState={'enabled' if state['enabled'] else 'disabled'}\n"
                    "MainPID=0\n"
                )
            elif "pdeadman" in unit:
                text = "LoadState=loaded\nActiveState=active\nMainPID=0\n"
            else:
                text = "LoadState=not-found\nActiveState=inactive\nUnitFileState=\nMainPID=0\n"
            return self._done(command, 0, text)
        if action == "stop" and unit in self.units:
            self.units[unit]["active"] = False
        if action == "disable" and unit in self.units and not self.undisableable:
            self.units[unit]["enabled"] = False
        return self._done(command)

    def run(self, command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(command))
        head = command[0]
        if head == "rpm":
            lines = [
                f"{info['name']}\t(none)\t{info['version']}\t{info['release']}\t"
                f"{item.rsplit('.', 1)[1]}\t{info['tid']}"
                for item, info in self.packages.items()
            ]
            return self._done(command, 0, "\n".join(lines) + "\n")
        if head == "dnf":
            return self.dnf_command(command)
        if head == "modprobe":
            name = command[-1]
            if "--dry-run" in command:
                ok = name in self.loadable or (self.sys_module / name).exists()
                return self._done(command, 0 if ok else 1)
            if "-r" in command:
                if (self.sys_module / name / "refcnt").read_text().strip() == "0":
                    for child in (self.sys_module / name).iterdir():
                        child.unlink()
                    (self.sys_module / name).rmdir()
                return self._done(command)
            if name in self.loadable:
                self._load(name)
                return self._done(command)
            return self._done(
                command, 1, stderr=f"modprobe: FATAL: Module {name} not found"
            )
        if head == "systemctl":
            return self.systemctl(command)
        if head == "semodule":
            rows = "".join(f"{row}   pp\n" for row in sorted(self.selinux_modules))
            return self._done(command, 0, rows)
        return self._done(command)

    def ran(self, *prefix: str) -> list[list[str]]:
        return [call for call in self.calls if call[: len(prefix)] == list(prefix)]


@pytest.fixture
def node_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def build(**kwargs: Any) -> tuple[FakeNode, list[dict[str, Any]]]:
        node = FakeNode(tmp_path, **kwargs)
        emitted: list[dict[str, Any]] = []
        monkeypatch.setattr(deps, "STATE_ROOT", tmp_path / "state")
        monkeypatch.setattr(deps, "SYS_MODULE", node.sys_module)
        monkeypatch.setattr(deps, "MODULES_ROOT", tmp_path / "lib-modules")
        monkeypatch.setattr(deps, "run", node.run)
        monkeypatch.setattr(deps, "tool_present", node.tool)
        monkeypatch.setattr(deps, "kernel_release", lambda: KERNEL)
        monkeypatch.setattr(deps, "emit", emitted.append)
        return node, emitted

    return build


def ensure_args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "run_id": RUN_ID,
        "tool": ["tc"],
        "module": ["sch_netem"],
        "deadman_seconds": 2700,
        "allow_install": True,
        "allow_module_load": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def recorded() -> dict[str, Any]:
    return deps.load_state(RUN_ID)


def with_tc() -> dict[str, tuple[str, str, str]]:
    return {**BASELINE, TC_NEVRA: ("iproute-tc", "5.10.0", "2.amzn2023.0.5")}


# --------------------------------------------------------------------------- #
# install only what is missing
# --------------------------------------------------------------------------- #
def test_ensure_changes_nothing_when_tc_and_netem_are_there(node_factory) -> None:
    node, emitted = node_factory(installed=with_tc(), loaded={"sch_netem"})
    deps.ensure(ensure_args())
    result = emitted[-1]
    assert result["refused"] is None and result["deadman"] is None
    assert result["installed"] == [] and result["modules_loaded_by_run"] == []
    touched = [call for call in node.calls if call[0] in {"dnf", "systemd-run"}]
    assert touched == [], touched
    assert not node.ran("modprobe", "sch_netem"), "a loaded module is never reloaded"


def test_ensure_installs_iproute_tc_only_when_tc_is_missing(node_factory) -> None:
    node, emitted = node_factory(loaded={"sch_netem"})
    deps.ensure(ensure_args())
    result = emitted[-1]
    assert result["refused"] is None, result
    assert result["installed"] == [TC_NEVRA]
    assert result["tools"] == {"tc": True}
    preview, install = node.ran("dnf", "--assumeno"), node.ran("dnf", "-y")
    assert preview[0][-2:] == ["install", "iproute-tc"], preview
    assert install == [["dnf", "-y", *deps.DNF_INSTALL_OPTIONS, "install", TC_NEVRA]]
    # The deadman is armed before the install runs.
    order = [call[0] for call in node.calls if call[0] in {"systemd-run", "dnf"}]
    assert order == ["dnf", "systemd-run", "dnf"], order
    assert recorded()["installed"] == [TC_NEVRA] and recorded()["pending"] == []
    assert not any("kernel-modules-extra" in " ".join(c) for c in node.calls), (
        "a loadable sch_netem never pulls the module package"
    )


def test_ensure_loads_netem_only_with_the_module_approval(node_factory) -> None:
    node, emitted = node_factory(installed=with_tc(), loadable={"sch_netem"})
    deps.ensure(ensure_args(allow_module_load=False))
    refused = emitted[-1]["refused"]
    assert refused["reason"] == "NOT_AUTHORIZED", refused
    assert "KERNEL_MODULE_LOAD" in refused["detail"]
    assert not node.ran("modprobe", "sch_netem") and not node.ran("systemd-run")
    deps.ensure(ensure_args())
    result = emitted[-1]
    assert result["refused"] is None and result["modules_loaded_by_run"] == [
        "sch_netem"
    ]
    assert node.ran("modprobe", "sch_netem") == [["modprobe", "sch_netem"]]
    assert not node.ran("dnf"), "a loadable module needs no package"


def test_ensure_installs_the_running_kernels_module_package(node_factory) -> None:
    node, emitted = node_factory(
        installed=with_tc(), module_files_from="kernel-modules-extra"
    )
    deps.ensure(ensure_args())
    result = emitted[-1]
    assert result["refused"] is None, result
    assert node.ran("dnf", "--assumeno")[0][-1] == f"kernel-modules-extra-{KERNEL}"
    assert result["installed"] == [KMOD_NEVRA]
    assert result["modules_loaded_by_run"] == ["sch_netem"]
    assert node.packages["kernel-modules-extra-6.1.90-90.90.amzn2023.x86_64"], (
        "the other kernel's module package stays installed"
    )


def test_ensure_without_install_approval_keeps_capability_absent(node_factory) -> None:
    node, emitted = node_factory(loaded={"sch_netem"})
    deps.ensure(ensure_args(allow_install=False))
    refused = emitted[-1]["refused"]
    assert refused["reason"] == "CAPABILITY_ABSENT", refused
    assert refused["detail"].startswith("the node lacks tc"), (
        "refused['detail'].startswith('the node lacks tc')"
    )
    assert "TEMP_TEST_PACKAGE_INSTALL" in refused["detail"]
    assert not node.ran("dnf") and not node.ran("systemd-run")


# --------------------------------------------------------------------------- #
# refuse anything beyond the reviewed packages
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "plan,needle",
    [
        (
            {
                "Installing": [TC_ROW],
                "Installing dependencies": [
                    ("libbpf", "x86_64", "2:0.5.0-1.amzn2023", "amazonlinux")
                ],
            },
            "install the dependencies libbpf-2:0.5.0-1.amzn2023.x86_64",
        ),
        (
            {
                "Installing": [TC_ROW],
                "Upgrading": [("iproute", "x86_64", "6.5.0-1.amzn2023", "amazonlinux")],
            },
            "upgrade iproute-6.5.0-1.amzn2023.x86_64",
        ),
        (
            {
                "Installing": [TC_ROW],
                "Installing dependencies": [
                    ("kernel", "x86_64", "6.1.200-200.amzn2023", "amazonlinux")
                ],
            },
            "dnf would pull a kernel: kernel-6.1.200-200.amzn2023.x86_64",
        ),
        (
            {
                "Installing": [TC_ROW],
                "Removing": [("bash", "x86_64", "5.2.15-1.amzn2023.0.2", "@System")],
            },
            "remove bash",
        ),
        ({"Installing": [TC_ROW, ("x", "y", "z", "w")]}, "install x-z.y"),
    ],
)
def test_ensure_refuses_a_transaction_beyond_the_package(
    node_factory, plan: dict[str, Any], needle: str
) -> None:
    node, emitted = node_factory(loaded={"sch_netem"})
    node.install_preview = plan
    deps.ensure(ensure_args())
    refused = emitted[-1]["refused"]
    assert refused["reason"] == "FIXTURE_NOT_READY", refused
    assert needle in refused["detail"], refused["detail"]
    assert not node.ran("dnf", "-y"), "a refused transaction never runs"
    assert not node.ran("systemd-run"), "nothing changed, so no deadman"
    assert recorded()["installed"] == [] and recorded()["pending"] == []


def test_ensure_refuses_an_unreadable_or_obsoleting_plan(node_factory) -> None:
    node, emitted = node_factory(loaded={"sch_netem"})
    original = node.dnf_command

    def obsoleting(command: list[str]) -> subprocess.CompletedProcess[str]:
        done = original(command)
        if "--assumeno" in command:
            text = done.stdout.replace(
                "\n\nTransaction Summary",
                "\n     replacing  iproute.x86_64 5.10.0-1.amzn2023\n\nTransaction Summary",
            )
            return subprocess.CompletedProcess(command, 1, stdout=text, stderr="")
        return done

    node.dnf_command = obsoleting
    deps.ensure(ensure_args())
    assert "replace iproute.x86_64" in emitted[-1]["refused"]["detail"]
    assert not node.ran("dnf", "-y"), "not node.ran('dnf', '-y')"


def test_ensure_refuses_without_a_package_for_the_running_kernel(node_factory) -> None:
    node, emitted = node_factory(installed=with_tc())
    node.unmatched = [f"kernel-modules-extra-{KERNEL}"]
    deps.ensure(ensure_args())
    refused = emitted[-1]["refused"]
    assert refused["reason"] == "FIXTURE_NOT_READY"
    assert refused["detail"] == (
        f"no package kernel-modules-extra-{KERNEL} in the enabled repositories"
    )
    assert not node.ran("dnf", "-y") and not node.ran("modprobe", "sch_netem")


def test_ensure_refuses_another_kernels_module_build(node_factory) -> None:
    node, emitted = node_factory(installed=with_tc())
    node.install_preview = {
        "Installing": [
            ("kernel-modules-extra", "x86_64", "6.1.200-200.amzn2023", "amazonlinux")
        ]
    }
    deps.ensure(ensure_args())
    detail = emitted[-1]["refused"]["detail"]
    assert "not the running kernel's" in detail and KERNEL in detail, detail
    assert not node.ran("dnf", "-y"), "not node.ran('dnf', '-y')"


def test_install_records_only_its_own_rpm_transaction() -> None:
    before = {"bash-1-1.x86_64": {"name": "bash", "tid": "1"}}
    after = {
        **before,
        TC_NEVRA: {"name": "iproute-tc", "tid": "7"},
        "unrelated-1-1.x86_64": {"name": "unrelated", "tid": "8"},
    }
    ours, problems = deps.installed_by_run(before, after, [TC_NEVRA])
    assert ours == [TC_NEVRA] and problems == []
    extra = {**after, "libbpf-1-1.x86_64": {"name": "libbpf", "tid": "7"}}
    ours, problems = deps.installed_by_run(before, extra, [TC_NEVRA])
    assert ours == [TC_NEVRA, "libbpf-1-1.x86_64"]
    assert problems == ["dnf installed more than previewed: libbpf-1-1.x86_64"]


def test_install_records_the_closure_in_install_order() -> None:
    after = {
        SMART_NEVRA: {"name": "smartmontools", "tid": "7"},
        SELINUX_NEVRA: {"name": "smartmontools-selinux", "tid": "7"},
    }
    ours, problems = deps.installed_by_run({}, after, SMART_CLOSURE)
    assert (ours, problems) == (SMART_CLOSURE, []), (
        "the dependency before its package, as rpm installed them -- not sorted"
    )


# --------------------------------------------------------------------------- #
# cleanup (also the deadman's command) undoes exactly this run's changes
# --------------------------------------------------------------------------- #
def test_cleanup_removes_exactly_what_this_run_installed(node_factory) -> None:
    node, emitted = node_factory(
        loadable={"sch_netem"}, installed={**BASELINE}, module_files_from=None
    )
    deps.ensure(ensure_args())
    assert emitted[-1]["installed"] == [TC_NEVRA]
    deadman = node.ran("systemd-run")[0]
    assert deadman[-3:] == ["cleanup", "--run-id", RUN_ID], deadman
    node.calls.clear()
    deps.cleanup(argparse.Namespace(run_id=RUN_ID))
    result = emitted[-1]
    assert result["clean"] is True, result
    assert result["report"]["removed"] == [TC_NEVRA]
    assert result["report"]["unloaded"] == ["sch_netem"]
    removals = node.ran("dnf")
    assert removals == [
        ["dnf", "--assumeno", *deps.DNF_REMOVE_OPTIONS, "remove", TC_NEVRA],
        ["dnf", "-y", *deps.DNF_REMOVE_OPTIONS, "remove", TC_NEVRA],
    ]
    assert "--setopt=clean_requirements_on_remove=False" in deps.DNF_REMOVE_OPTIONS
    assert set(BASELINE) <= set(node.packages), "pre-existing packages stay"
    assert node.ran(
        "systemctl", "stop", f"gfrs-{deps.run_digest(RUN_ID)}-pdeadman.timer"
    ), "node.ran('systemctl', 'stop', f'gfrs-{deps.run_digest(RUN_ID)}-pdeadma"
    assert node.ran("modprobe", "-r", "sch_netem"), (
        "node.ran('modprobe', '-r', 'sch_netem')"
    )


def test_cleanup_never_unloads_a_module_the_run_did_not_load(node_factory) -> None:
    node, emitted = node_factory(loaded={"sch_netem"})
    deps.ensure(ensure_args())
    deps.cleanup(argparse.Namespace(run_id=RUN_ID))
    assert emitted[-1]["clean"] is True
    assert not node.ran("modprobe", "-r", "sch_netem"), (
        "not node.ran('modprobe', '-r', 'sch_netem')"
    )
    assert (node.sys_module / "sch_netem").is_dir(), "the node's module stays loaded"


def test_cleanup_keeps_a_module_still_in_use(node_factory) -> None:
    node, emitted = node_factory(installed=with_tc(), loadable={"sch_netem"})
    deps.ensure(ensure_args())
    (node.sys_module / "sch_netem" / "refcnt").write_text("1\n")
    node.calls.clear()
    deps.cleanup(argparse.Namespace(run_id=RUN_ID))
    result = emitted[-1]
    assert result["clean"] is False and result["residue"]["modules"] == ["sch_netem"]
    assert not node.ran("modprobe", "-r", "sch_netem"), (
        "not node.ran('modprobe', '-r', 'sch_netem')"
    )
    assert not node.ran(
        "systemctl", "stop", f"gfrs-{deps.run_digest(RUN_ID)}-pdeadman.timer"
    ), "an unfinished cleanup leaves the deadman armed"


def test_cleanup_refuses_a_cascading_removal(node_factory) -> None:
    node, emitted = node_factory(loaded={"sch_netem"})
    deps.ensure(ensure_args())
    node.remove_preview = {
        "Removing": [TC_ROW[:2] + (TC_ROW[2], "@amazonlinux")],
        "Removing dependent packages": [
            ("netplug", "x86_64", "1.0-1.amzn2023", "@System")
        ],
    }
    deps.cleanup(argparse.Namespace(run_id=RUN_ID))
    result = emitted[-1]
    assert result["clean"] is False, result
    assert "remove the dependent packages netplug" in result["report"]["errors"][0]
    assert not node.ran("dnf", "-y", *deps.DNF_REMOVE_OPTIONS), (
        "a removal that takes more than this run's package never runs"
    )
    assert result["residue"]["packages"] == [TC_NEVRA]


def test_state_is_recorded_before_the_install_runs(node_factory) -> None:
    node, emitted = node_factory(loaded={"sch_netem"})
    seen: list[dict[str, Any]] = []
    original = node.dnf_command

    def spy(command: list[str]) -> subprocess.CompletedProcess[str]:
        if "-y" in command:
            seen.append(json.loads(json.dumps(deps.load_state(RUN_ID))))
        return original(command)

    node.dnf_command = spy
    deps.ensure(ensure_args())
    assert seen and seen[0]["pending"] == [TC_NEVRA], seen
    assert emitted[-1]["refused"] is None


def test_parse_dnf_transaction_rows_and_leftovers() -> None:
    text = dnf_table(
        {"Installing": [TC_ROW], "Installing dependencies": [KMOD_ROW]}, "Install"
    )
    plan = deps.parse_dnf_transaction(text)
    assert [row["section"] for row in plan["entries"]] == [
        "Installing",
        "Installing dependencies",
    ]
    assert plan["entries"][0]["version"] == TC_ROW[2] and plan["unparsed"] == []
    wrapped = text.replace(" iproute-tc  ", " iproute-tc\n          ")
    assert deps.parse_dnf_transaction(wrapped)["unparsed"] == ["iproute-tc"]
    nothing = deps.parse_dnf_transaction("Dependencies resolved.\nNothing to do.\n")
    assert nothing == {"entries": [], "unparsed": []}


def test_ensure_rejects_a_tool_outside_the_allow_list(node_factory) -> None:
    node, emitted = node_factory()
    with pytest.raises(deps.ProbeError, match="allow-list"):
        deps.ensure(ensure_args(tool=["ipmitool"]))
    assert node.calls == []


# --------------------------------------------------------------------------- #
# the 007 / 004 packages: the same mechanism; 007's package has a reviewed closure
# --------------------------------------------------------------------------- #
def test_the_allow_list_names_the_al2023_package_of_each_tool() -> None:
    assert deps.TOOL_PACKAGES == {
        "tc": "iproute-tc",
        "smartctl": "smartmontools",
        "mkfs.vfat": "dosfstools",
    }
    assert deps.PACKAGE_CLOSURES == {"smartmontools": ("smartmontools-selinux",)}, (
        "the one reviewed closure: AL2023's rich dependency "
        "(smartmontools-selinux if selinux-policy-targeted)"
    )
    assert deps.PACKAGE_FOOTPRINT == {
        "smartmontools": {"units": ("smartd.service",)},
        "smartmontools-selinux": {"selinux_modules": ("200 smartmon",)},
    }, "the preset-enabled unit and the priority-200 policy module, never the base one"
    assert "ipmitool" not in deps.TOOL_PACKAGES, (
        "015 serves ipmitool from the wrapper (allow_missing_real); no package"
    )


@pytest.mark.parametrize(
    ("tool", "package", "nevras"),
    [
        ("smartctl", "smartmontools", SMART_CLOSURE),
        ("mkfs.vfat", "dosfstools", [DOSFS_NEVRA]),
    ],
)
def test_ensure_installs_the_tools_package_with_its_closure_and_cleanup_removes_them(
    node_factory, tool: str, package: str, nevras: list[str]
) -> None:
    """``nevras`` is the install order: a reviewed dependency before its package."""

    node, emitted = node_factory()
    deps.ensure(ensure_args(tool=[tool], module=[]))
    result = emitted[-1]
    assert result["refused"] is None, result
    assert result["installed"] == nevras
    assert result["tools"] == {tool: True}
    assert result["modules_loaded_by_run"] == [] and result["modules"] == {}
    preview, install = node.ran("dnf", "--assumeno"), node.ran("dnf", "-y")
    assert preview[0][-2:] == ["install", package], preview
    assert install == [["dnf", "-y", *deps.DNF_INSTALL_OPTIONS, "install", *nevras]]
    assert result["deadman"] == deps.deadman_unit(RUN_ID)
    assert recorded()["installed"] == nevras and recorded()["pending"] == []
    assert node.tool(tool) is True, "the tool is usable after the install"

    emitted.clear()
    node.calls.clear()
    deps.cleanup(argparse.Namespace(run_id=RUN_ID))
    report = emitted[-1]
    assert report["clean"] is True, report
    # Last installed first: the package before its dependency, each through its
    # own previewed transaction that may remove exactly that one package.
    assert report["report"]["removed"] == list(reversed(nevras))
    expected = []
    for nevra in reversed(nevras):
        expected.append(
            ["dnf", "--assumeno", *deps.DNF_REMOVE_OPTIONS, "remove", nevra]
        )
        expected.append(["dnf", "-y", *deps.DNF_REMOVE_OPTIONS, "remove", nevra])
    assert node.ran("dnf") == expected
    assert node.tool(tool) is False, "the package this run installed is gone"
    assert set(node.packages) == set(BASELINE), "the node's own packages stay"
    assert recorded()["installed"] == []
    assert report["residue"] == {
        "packages": [],
        "modules": [],
        "units": [],
        "selinux_modules": [],
    }, "the residue proves every package gone and nothing they brought left"


def test_the_live_smartmontools_preview_is_accepted_with_its_closure(
    node_factory,
) -> None:
    """The transaction both AL2023 kernels previewed live: ``Installing``
    smartmontools, ``Installing dependencies`` smartmontools-selinux, same build."""

    node, emitted = node_factory()
    node.install_preview = {
        "Installing": [SMART_ROW],
        "Installing dependencies": [SELINUX_ROW],
    }
    deps.ensure(ensure_args(tool=["smartctl"], module=[]))
    result = emitted[-1]
    assert result["refused"] is None, result
    assert [(row["section"], row["name"]) for row in result["transaction"]] == [
        ("Installing", "smartmontools"),
        ("Installing dependencies", "smartmontools-selinux"),
    ], "the evidence keeps dnf's table as printed"
    assert result["installed"] == SMART_CLOSURE
    assert "would also installing" not in json.dumps(emitted), "the old grammar is gone"


def test_ensure_without_the_approval_names_the_tools_package(node_factory) -> None:
    node, emitted = node_factory()
    deps.ensure(ensure_args(tool=["smartctl"], module=[], allow_install=False))
    refused = emitted[-1]["refused"]
    assert refused["reason"] == "CAPABILITY_ABSENT", refused
    assert refused["detail"] == (
        "the node lacks smartctl; installing smartmontools with its reviewed "
        "dependency smartmontools-selinux needs TEMP_TEST_PACKAGE_INSTALL"
    )
    assert not node.ran("dnf") and not node.ran("systemd-run")


@pytest.mark.parametrize("with_closure_row", [True, False])
def test_ensure_refuses_an_undeclared_dependency_naming_only_that(
    node_factory, with_closure_row: bool
) -> None:
    node, emitted = node_factory()
    mailx = ("mailx", "x86_64", "12.5-30.amzn2023", "amazonlinux")
    node.install_preview = {
        "Installing": [SMART_ROW],
        "Installing dependencies": [SELINUX_ROW, mailx]
        if with_closure_row
        else [mailx],
    }
    deps.ensure(ensure_args(tool=["smartctl"], module=[]))
    refused = emitted[-1]["refused"]
    assert refused["reason"] == "FIXTURE_NOT_READY", refused
    assert refused["detail"] == (
        "dnf would also install the dependencies mailx-12.5-30.amzn2023.x86_64 "
        "(only smartmontools with its reviewed dependency smartmontools-selinux "
        "may be installed)"
    ), "the unexpected package is named; the accepted closure never is"
    assert not node.ran("dnf", "-y"), "a refused transaction never runs"
    assert not node.ran("systemd-run"), "nothing changed, so no deadman"
    assert recorded()["installed"] == [] and recorded()["pending"] == []
    assert emitted[-1]["transaction"][0]["name"] == "smartmontools"


def test_ensure_refuses_a_closure_package_from_another_build(node_factory) -> None:
    node, emitted = node_factory()
    node.install_preview = {
        "Installing": [SMART_ROW],
        "Installing dependencies": [
            ("smartmontools-selinux", "noarch", "1:7.2-10.amzn2023.0.1", "amazonlinux")
        ],
    }
    deps.ensure(ensure_args(tool=["smartctl"], module=[]))
    refused = emitted[-1]["refused"]
    assert refused["reason"] == "FIXTURE_NOT_READY", refused
    assert refused["detail"] == (
        "dnf would install smartmontools-selinux-1:7.2-10.amzn2023.0.1.noarch, not "
        "the build of smartmontools-1:7.2-11.amzn2023.0.1"
    )
    assert not node.ran("dnf", "-y"), "a refused transaction never runs"


def test_a_closure_package_the_node_already_has_is_never_touched(node_factory) -> None:
    node, emitted = node_factory(
        installed={
            **BASELINE,
            SELINUX_NEVRA: ("smartmontools-selinux", "1:7.2", "11.amzn2023.0.1"),
        }
    )
    deps.ensure(ensure_args(tool=["smartctl"], module=[]))
    result = emitted[-1]
    assert result["refused"] is None, result
    assert result["installed"] == [SMART_NEVRA], "dnf pulled nothing: the node had it"
    assert result["footprint"]["claimed"] == {
        "units": ["smartd.service"],
        "selinux_modules": [],
    }, "the node's own policy module is not this run's"
    deps.cleanup(argparse.Namespace(run_id=RUN_ID))
    assert emitted[-1]["clean"] is True, emitted[-1]
    assert emitted[-1]["report"]["removed"] == [SMART_NEVRA]
    assert SELINUX_NEVRA in node.packages, "the node's smartmontools-selinux stays"
    assert "200 smartmon" in node.selinux_modules, "and so does its policy module"


# --------------------------------------------------------------------------- #
# what the closure brings beyond its files: smartd.service, the smartmon policy
# --------------------------------------------------------------------------- #
def test_the_install_quiets_the_preset_enabled_smartd_and_claims_the_policy(
    node_factory,
) -> None:
    """AL2023's 90-default.preset enables smartd.service when smartmontools's
    %post presets it; the fixture needs smartctl, never the daemon."""

    node, emitted = node_factory()
    deps.ensure(ensure_args(tool=["smartctl"], module=[]))
    result = emitted[-1]
    assert result["refused"] is None, result
    footprint = result["footprint"]
    assert footprint["before"] == {
        "units": {
            "smartd.service": {"loaded": False, "active": False, "enabled": False}
        },
        "selinux_modules": {"200 smartmon": False},
    }
    assert footprint["after"] == {
        "units": {
            "smartd.service": {"loaded": True, "active": False, "enabled": False}
        },
        "selinux_modules": {"200 smartmon": True},
    }, "disabled again right after the install; the policy stays until cleanup"
    assert footprint["quieted"] == ["smartd.service"]
    assert footprint["claimed"] == {
        "units": ["smartd.service"],
        "selinux_modules": ["200 smartmon"],
    }
    assert node.ran("systemctl", "disable") == [
        ["systemctl", "disable", "smartd.service"]
    ]
    assert node.ran("systemctl", "stop", "smartd.service") == [], "inactive: no stop"
    assert node.units["smartd.service"]["enabled"] is False
    assert recorded()["units"] == ["smartd.service"]
    assert recorded()["selinux_modules"] == ["200 smartmon"]


def test_a_daemon_the_install_started_is_stopped_at_once(node_factory) -> None:
    node, emitted = node_factory()
    node.start_units_on_install = True
    deps.ensure(ensure_args(tool=["smartctl"], module=[]))
    result = emitted[-1]
    assert result["refused"] is None, result
    assert node.ran("systemctl", "stop", "smartd.service") == [
        ["systemctl", "stop", "smartd.service"]
    ]
    assert result["footprint"]["after"]["units"]["smartd.service"] == {
        "loaded": True,
        "active": False,
        "enabled": False,
    }


def test_a_unit_that_cannot_be_quieted_refuses_the_variant(node_factory) -> None:
    node, emitted = node_factory()
    node.undisableable = True
    deps.ensure(ensure_args(tool=["smartctl"], module=[]))
    result = emitted[-1]
    assert result["refused"] == {
        "reason": "FIXTURE_NOT_READY",
        "detail": "smartd.service is still enabled after the install",
    }
    # The packages are on the node: recorded and under the deadman, so the
    # registered cleanup removes them.
    assert recorded()["installed"] == SMART_CLOSURE
    assert result["deadman"] == deps.deadman_unit(RUN_ID)


def test_cleanup_quiets_a_daemon_started_meanwhile_before_the_packages_go(
    node_factory,
) -> None:
    node, emitted = node_factory()
    deps.ensure(ensure_args(tool=["smartctl"], module=[]))
    node.units["smartd.service"]["active"] = True  # started during the run
    node.calls.clear()
    deps.cleanup(argparse.Namespace(run_id=RUN_ID))
    result = emitted[-1]
    assert result["clean"] is True, result
    assert result["report"]["quieted"] == ["smartd.service"]
    stop = node.calls.index(["systemctl", "stop", "smartd.service"])
    removal = next(i for i, call in enumerate(node.calls) if call[:2] == ["dnf", "-y"])
    assert stop < removal, "the daemon is stopped before its package goes"
    assert "smartd.service" not in node.units, "the unit file went with the package"


def test_cleanup_reports_a_policy_module_the_removal_left_behind(node_factory) -> None:
    node, emitted = node_factory()
    deps.ensure(ensure_args(tool=["smartctl"], module=[]))
    node.sticky_selinux = True
    node.calls.clear()
    deps.cleanup(argparse.Namespace(run_id=RUN_ID))
    result = emitted[-1]
    assert result["clean"] is False, result
    assert result["residue"] == {
        "packages": [],
        "modules": [],
        "units": [],
        "selinux_modules": ["200 smartmon"],
    }
    assert result["report"]["removed"] == [SMART_NEVRA, SELINUX_NEVRA], "both went"
    assert not node.ran(
        "systemctl", "stop", f"gfrs-{deps.run_digest(RUN_ID)}-pdeadman.timer"
    ), "an unfinished cleanup leaves the deadman armed"


def test_residue_reads_a_record_from_before_the_footprint_keys(node_factory) -> None:
    node_factory()
    deps.save_state(
        RUN_ID, {"run_id": RUN_ID, "installed": [], "pending": [], "modules": []}
    )
    assert deps.residue(RUN_ID) == {
        "packages": [],
        "modules": [],
        "units": [],
        "selinux_modules": [],
    }


@pytest.mark.parametrize(
    ("section", "phrase"),
    [
        ("Installing", "install"),
        ("Installing dependencies", "install the dependencies"),
        ("Installing weak dependencies", "install the weak dependencies"),
        ("Upgrading", "upgrade"),
        ("Downgrading", "downgrade"),
        ("Removing", "remove"),
        ("Removing dependent packages", "remove the dependent packages"),
        ("Removing unused dependencies", "remove the unused dependencies"),
        ("Replacing", "replace"),
        ("Skipping packages with conflicts", "skip the packages with conflicts"),
        ("Frobnicating", "frobnicating"),
    ],
)
def test_a_refusal_reads_as_what_dnf_would_do(section: str, phrase: str) -> None:
    assert deps.section_phrase(section) == phrase


def test_ensure_touches_nothing_when_the_tool_is_there(node_factory) -> None:
    node, emitted = node_factory(
        installed={**BASELINE, DOSFS_NEVRA: ("dosfstools", "4.2", "3.amzn2023.0.2")}
    )
    deps.ensure(ensure_args(tool=["mkfs.vfat"], module=[]))
    result = emitted[-1]
    assert result["refused"] is None and result["installed"] == []
    assert not node.ran("dnf") and not node.ran("systemd-run")
    assert recorded()["installed"] == [], "nothing recorded, so cleanup removes nothing"
