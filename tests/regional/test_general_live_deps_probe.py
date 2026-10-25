"""The temporary test package probe (GF-GENERAL-LIVE-009 tcp_only dependencies).

A recording fake node stands in for dnf, rpm, modprobe and systemd: the tests
prove the probe installs only what is missing, refuses any dnf transaction that
would pull more than the reviewed package for the running kernel, records what
it installed and loaded before doing it, and that its cleanup (also the
deadman's command) removes and unloads exactly that -- never a package or a
module the run found on the node.
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
        self.repo = {TC_ROW[0]: TC_ROW, KMOD_ROW[0]: KMOD_ROW}
        self.install_preview: dict[str, list[tuple[str, str, str, str]]] | None = None
        self.remove_preview: dict[str, list[tuple[str, str, str, str]]] | None = None
        self.unmatched: list[str] = []
        self.next_tid = 2000

    @staticmethod
    def _record(name: str, version: str, release: str, tid: str) -> dict[str, str]:
        return {"name": name, "version": version, "release": release, "tid": tid}

    def _load(self, name: str) -> None:
        (self.sys_module / name).mkdir(exist_ok=True)
        (self.sys_module / name / "refcnt").write_text("0\n")

    def tool(self, name: str) -> bool:
        if name == "tc":
            return any(item["name"] == "iproute-tc" for item in self.packages.values())
        return name in {"dnf", "rpm", "modprobe", "systemd-run", "systemctl"}

    def _done(
        self, command: list[str], code: int = 0, stdout: str = "", stderr: str = ""
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, code, stdout=stdout, stderr=stderr)

    def _rows(self, specs: list[str]) -> dict[str, list[tuple[str, str, str, str]]]:
        rows = []
        for spec in specs:
            name = "kernel-modules-extra" if spec.startswith("kernel-modules") else spec
            rows.append(self.repo[name])
        return {"Installing": rows}

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
                if self.module_files_from and item.startswith(self.module_files_from):
                    self.loadable.add("sch_netem")
            return self._done(command, 0, "Complete!\n")
        for item in targets:
            self.packages.pop(item, None)
        return self._done(command, 0, "Complete!\n")

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
        if head == "systemctl" and command[1] == "show":
            return self._done(command, 0, "ActiveState=active\nMainPID=0\n")
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
            "installing dependencies libbpf-2:0.5.0-1.amzn2023.x86_64",
        ),
        (
            {
                "Installing": [TC_ROW],
                "Upgrading": [("iproute", "x86_64", "6.5.0-1.amzn2023", "amazonlinux")],
            },
            "upgrading iproute-6.5.0-1.amzn2023.x86_64",
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
            "removing bash",
        ),
        ({"Installing": [TC_ROW, ("x", "y", "z", "w")]}, "installing x-z.y"),
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
    assert "replacing iproute.x86_64" in emitted[-1]["refused"]["detail"]
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
    assert "removing dependent packages netplug" in result["report"]["errors"][0]
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
        deps.ensure(ensure_args(tool=["smartctl"]))
    assert node.calls == []
