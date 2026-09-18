"""Exercise the real Make setup targets with local installer and download fakes."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERSION = "9.8.7"
AUDIT_VERSION = "4.5.6"
CYCLONE_VERSION = "7.8.9"

DRIVER = r"""
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import time

mode, executable, *args = sys.argv[1:]
executable = Path(executable)
prefix = executable.parent.parent
events = Path(os.environ["SETUP_EVENTS"])

def record(kind, **fields):
    with events.open("a") as handle:
        handle.write(json.dumps({"kind": kind, **fields}) + "\n")
    events.with_name(kind + ".ready").touch()

def wait_for(kind):
    deadline = time.monotonic() + 10
    while not events.with_name(kind + ".ready").exists():
        if time.monotonic() >= deadline:
            raise SystemExit("parallel setup did not reach " + kind)
        time.sleep(0.01)

def fail(stage):
    if os.environ.get("FAIL_SETUP_STAGE") == stage:
        record(stage + "-failed")
        print("fixture-private-failed-command-output", file=sys.stderr)
        raise SystemExit(23)

def launcher(path, kind):
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), kind, str(path)]
    path.write_text("#!/bin/sh\nexec " + shlex.join(command) + ' "$@"\n')
    path.chmod(0o755)

if mode == "setup":
    record("host-start", args=args)
    if os.environ.get("REQUIRE_HOST_OVERLAP") == "true":
        wait_for("tool-venv")
    if os.environ.get("HOST_WAIT_FOR"):
        wait_for(os.environ["HOST_WAIT_FOR"])
    fail("host")
    time.sleep(0.03)
    target = Path(args[args.index("--venv") + 1]).absolute()
    launcher(target / "bin/python", "python")
    (target / "bin/gpu-fault-admin").write_text("fixture admin CLI\n")
    (target / "bin/gpu-fault-admin").chmod(0o755)
    (target / "ready").write_text("ready\n")
    record("host-ready", path=str(target))
elif mode == "python":
    if args[:2] == ["-m", "venv"]:
        target = Path(args[-1])
        record("tool-venv", python=str(executable), path=str(target))
        if os.environ.get("REQUIRE_HOST_OVERLAP") == "true":
            wait_for("host-start")
        fail("tool-venv")
        launcher(target / "bin/python", "python")
    elif args[:2] == ["-I", "-c"]:
        state = prefix / "packages.json"
        packages = json.loads(state.read_text()) if state.exists() else {}
        print(json.dumps({
            "prefix": str(prefix),
            "base_prefix": (
                str(prefix) if os.environ.get("FAKE_UNISOLATED_TOOLS") == "true"
                else sys.base_prefix
            ),
            "versions": {name: packages.get(name) for name in args[3:]},
        }))
    elif args[:3] == ["-m", "pip", "install"]:
        if args[-1] == "pip":
            record("pip-prepare", python=str(executable))
            fail("pip-prepare")
        else:
            record("packages", python=str(executable), args=args)
            if os.environ.get("REQUIRE_DOWNLOAD_OVERLAP") == "true":
                wait_for("download")
            if os.environ.get("PACKAGES_WAIT_FOR"):
                wait_for(os.environ["PACKAGES_WAIT_FOR"])
            fail("packages")
            packages = dict(item.split("==", 1) for item in args if "==" in item)
            if os.environ.get("FAIL_SETUP_STAGE") != "package-verification":
                (prefix / "packages.json").write_text(json.dumps(packages))
            launcher(prefix / "bin/pip-audit", "tool")
            launcher(prefix / "bin/cyclonedx-py", "tool")
            record("packages-ready")
    elif args[:3] == ["-m", "pip", "check"]:
        record("pip-check", python=str(executable))
        fail("pip-check")
    elif args and args[0].endswith("setup_supply_chain_tools.py"):
        kind = "tool-preflight" if "--validate-only" in args else "tool-entry"
        record(kind, python=str(executable), args=args)
        os.execv(sys.executable, [sys.executable, *args])
    elif args and args[0].endswith("check-alert-rules.py"):
        record("promtool-check", args=args)
        os.execv(sys.executable, [sys.executable, *args])
    else:
        raise SystemExit("unexpected local Python command")
elif mode == "tool":
    if args != ["--version"]:
        raise SystemExit("unexpected tool command")
    packages = json.loads((prefix / "packages.json").read_text())
    if executable.name == "pip-audit":
        print("pip-audit " + packages["pip-audit"])
    else:
        print(packages["cyclonedx-bom"])
elif mode == "curl":
    record("download", args=args)
    if os.environ.get("REQUIRE_DOWNLOAD_OVERLAP") == "true":
        wait_for("packages")
    if os.environ.get("DOWNLOAD_WAIT_FOR"):
        wait_for(os.environ["DOWNLOAD_WAIT_FOR"])
    fail("download")
    expected = (
        "https://github.com/prometheus/prometheus/releases/download/v"
        + os.environ["EXPECTED_PROMTOOL_VERSION"] + "/prometheus-"
        + os.environ["EXPECTED_PROMTOOL_VERSION"] + ".linux-amd64.tar.gz"
    )
    if args[-1] != expected:
        raise SystemExit("unexpected download URL")
    shutil.copyfile(
        os.environ["FAKE_PROMTOOL_ARCHIVE"], args[args.index("-o") + 1]
    )
    record("download-complete")
else:
    raise SystemExit("unexpected local fake")
"""


def launcher(path: Path, driver: Path, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(driver), mode, str(path)]
    path.write_text(
        "#!/bin/sh\nexec " + shlex.join(command) + ' "$@"\n', encoding="utf-8"
    )
    path.chmod(0o755)


def archive(
    path: Path, *, reported_version: str = VERSION, member: str = "file"
) -> str:
    payload = (
        "#!/bin/sh\n"
        '[ "$#" = 1 ] && [ "$1" = "--version" ] || exit 91\n'
        f"printf '%s\\n' {shlex.quote('promtool, version ' + reported_version)}\n"
    ).encode()
    with tarfile.open(path, "w:gz") as bundle:
        info = tarfile.TarInfo(f"prometheus-{VERSION}.linux-amd64/promtool")
        info.mode = 0o755
        if member == "symlink":
            info.type = tarfile.SYMTYPE
            info.linkname = "/outside-promtool"
            bundle.addfile(info)
        elif member == "file":
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Checkout:
    root: Path
    host: Path
    tools: Path
    bootstrap: Path
    events: Path
    download: Path
    digest: str
    env: dict[str, str]

    def make(
        self,
        target: str = "deploy-host-setup-online",
        *assignments: str,
        environment: dict[str, str] | None = None,
        jobs: int | None = 8,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "make",
                "--no-print-directory",
                *(["-j", str(jobs)] if jobs is not None else []),
                target,
                f"PYTHON={self.bootstrap}",
                f"DEPLOY_HOST_BOOTSTRAP_PYTHON={self.bootstrap}",
                f"DEPLOY_HOST_VENV={self.host}",
                f"SUPPLY_CHAIN_TOOLS_VENV={self.tools}",
                f"PIP_AUDIT_VERSION={AUDIT_VERSION}",
                f"CYCLONEDX_BOM_VERSION={CYCLONE_VERSION}",
                f"PROMTOOL_VERSION={VERSION}",
                f"PROMTOOL_SHA256_LINUX_AMD64={self.digest}",
                "DEPLOY_HOST_PLATFORM=fixture",
                "PYTEST_XDIST_WORKERS=2",
                *assignments,
            ],
            cwd=self.root,
            env={**self.env, **(environment or {})},
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    def recorded(self) -> list[dict[str, Any]]:
        if not self.events.exists():
            return []
        return [json.loads(line) for line in self.events.read_text().splitlines()]


@pytest.fixture
def checkout(tmp_path: Path) -> Checkout:
    root = tmp_path / "checkout with spaces"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (root / "Makefile").symlink_to(ROOT / "Makefile")
    for name in (
        "setup_supply_chain_tools.py",
        "deploy_host_bundle.py",
        "check-alert-rules.py",
    ):
        (scripts / name).symlink_to(ROOT / "scripts" / name)
    driver = root / "local fakes.py"
    driver.write_text(DRIVER, encoding="utf-8")
    binaries = root / "fake commands"
    binaries.mkdir()
    for command in ("make", "env", "dirname"):
        original = shutil.which(command)
        assert original is not None, f"local fixture requires {command}"
        (binaries / command).symlink_to(original)
    launcher(binaries / "curl", driver, "curl")
    launcher(scripts / "setup-deploy-host.sh", driver, "setup")
    bootstrap = root / "bootstrap python/bin/python"
    launcher(bootstrap, driver, "python")
    host, tools = root / "host venv", root / "isolated tools"
    events, download = root / "events.jsonl", root / "release.tar.gz"
    digest = archive(download)
    env = {
        "HOME": str(root),
        "PATH": str(binaries),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(root / "pycache"),
        "AWS_EC2_METADATA_DISABLED": "true",
        "AWS_CONFIG_FILE": "/dev/null",
        "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
        "KUBECONFIG": "/dev/null",
        "SETUP_EVENTS": str(events),
        "EXPECTED_HOST": str(host),
        "EXPECTED_PROMTOOL_VERSION": VERSION,
        "FAKE_PROMTOOL_ARCHIVE": str(download),
    }
    return Checkout(root, host, tools, bootstrap, events, download, digest, env)


@pytest.mark.parametrize("jobs", [None, 8])
def test_online_entry_prepares_host_and_isolated_tools_in_parallel(
    checkout: Checkout, jobs: int | None
) -> None:
    result = checkout.make(
        jobs=jobs,
        environment={
            "REQUIRE_HOST_OVERLAP": "true",
            "REQUIRE_DOWNLOAD_OVERLAP": "true",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    events = checkout.recorded()
    kinds = [event["kind"] for event in events]
    assert kinds.index("tool-preflight") < kinds.index("host-start"), (
        "host preparation started before the path and pin preflight"
    )
    assert kinds.index("tool-venv") < kinds.index("host-ready"), (
        "tool preparation did not overlap the host installation"
    )
    assert kinds.index("packages") < kinds.index("download-complete"), (
        "Prometheus download did not overlap the Python tool installation"
    )
    assert kinds.index("download") < kinds.index("packages-ready"), (
        "Python tool installation waited for the Prometheus download"
    )
    tool_entry = next(event for event in events if event["kind"] == "tool-entry")
    assert tool_entry["python"] == str(checkout.bootstrap), (
        "tools must bootstrap independently of the host environment"
    )
    packages = next(event for event in events if event["kind"] == "packages")
    assert packages["python"] == str(checkout.tools / "bin/python"), (
        "unlocked tool dependencies entered the locked host environment"
    )
    assert f"pip-audit=={AUDIT_VERSION}" in packages["args"], "Make lost its audit pin"
    assert f"cyclonedx-bom=={CYCLONE_VERSION}" in packages["args"], (
        "Make lost its SBOM pin"
    )
    assert (checkout.host / "bin/gpu-fault-admin").is_file(), (
        "admin CLI was not prepared"
    )
    assert (checkout.tools / "bin/promtool").is_file(), (
        "isolated promtool was not prepared"
    )
    assert not (checkout.host / "packages.json").exists(), "host received tool packages"
    assert not (checkout.host / "bin/promtool").exists(), (
        "default tool path fell back to host"
    )
    assert kinds.count("promtool-check") == 1, (
        "the pinned native version was not checked"
    )
    assert kinds.index("promtool-check") > kinds.index("download-complete"), (
        "native tool verification started before the download completed"
    )
    assert kinds.index("promtool-check") > kinds.index("pip-check"), (
        "native tool installation raced with the tool venv preparation"
    )


@pytest.mark.parametrize("binding", ["make", "environment"])
def test_explicit_promtool_path_with_spaces_is_prepared(
    checkout: Checkout, binding: str
) -> None:
    selected = checkout.root / "chosen native tools/promtool"
    assignments = (f"PROMTOOL={selected}",) if binding == "make" else ()
    environment = {"PROMTOOL": str(selected)} if binding == "environment" else {}
    result = checkout.make(
        "deploy-host-setup-online", *assignments, environment=environment
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert os.access(selected, os.X_OK), (
        "the explicit native-tool destination was ignored"
    )
    assert not (checkout.tools / "bin/promtool").exists(), (
        "setup substituted its default path"
    )
    check = next(
        event for event in checkout.recorded() if event["kind"] == "promtool-check"
    )
    assert check["args"][check["args"].index("--promtool") + 1] == str(selected), (
        "version verification ignored the explicit tool"
    )


def test_valid_tools_reuse_the_verified_archive_without_pip_or_downloads(
    checkout: Checkout,
) -> None:
    first = checkout.make()
    assert first.returncode == 0, first.stdout + first.stderr
    previous = len(checkout.recorded())
    repeated = checkout.make(environment={"FAIL_SETUP_STAGE": "download"})
    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    events = checkout.recorded()[previous:]
    assert not {"tool-venv", "pip-prepare", "packages", "download"} & {
        event["kind"] for event in events
    }, "valid isolated tools were reinstalled or downloaded"
    assert any(event["kind"] == "promtool-check" for event in events), (
        "reused tool skipped the exact-version check"
    )


@pytest.mark.parametrize(
    "damage", ["missing", "nonexecutable", "wrong-version", "same-version"]
)
def test_missing_or_changed_promtool_is_repaired_from_the_pinned_archive(
    checkout: Checkout, damage: str
) -> None:
    installed = checkout.make()
    assert installed.returncode == 0, installed.stdout + installed.stderr
    tool = checkout.tools / "bin/promtool"
    expected = hashlib.sha256(tool.read_bytes()).hexdigest()
    if damage == "missing":
        tool.unlink()
    elif damage == "nonexecutable":
        tool.chmod(0o600)
    else:
        version = VERSION if damage == "same-version" else "0.0.0"
        tool.write_text(
            "#!/bin/sh\n"
            f"printf '%s\\n' {shlex.quote('promtool, version ' + version)}\n"
            "# different bytes\n",
            encoding="utf-8",
        )
    previous = len(checkout.recorded())
    repaired = checkout.make(environment={"FAIL_SETUP_STAGE": "download"})
    assert repaired.returncode == 0, repaired.stdout + repaired.stderr
    assert hashlib.sha256(tool.read_bytes()).hexdigest() == expected, (
        "version text was accepted instead of the archive-pinned executable bytes"
    )
    assert os.access(tool, os.X_OK), "repaired promtool is not executable"
    assert not any(
        event["kind"] == "download" for event in checkout.recorded()[previous:]
    ), "repair ignored the already verified archive"


@pytest.mark.parametrize("damage", ["missing", "wrong-version", "missing-command"])
def test_python_tool_damage_is_repaired_without_changing_host_packages(
    checkout: Checkout, damage: str
) -> None:
    installed = checkout.make()
    assert installed.returncode == 0, installed.stdout + installed.stderr
    state = checkout.tools / "packages.json"
    if damage == "missing":
        state.unlink()
    elif damage == "wrong-version":
        state.write_text(json.dumps({"pip-audit": "0.0.0", "cyclonedx-bom": "0.0.0"}))
    else:
        (checkout.tools / "bin/pip-audit").unlink()
    previous = len(checkout.recorded())
    repaired = checkout.make(environment={"FAIL_SETUP_STAGE": "download"})
    assert repaired.returncode == 0, repaired.stdout + repaired.stderr
    installs = [
        event for event in checkout.recorded()[previous:] if event["kind"] == "packages"
    ]
    assert len(installs) == 1, "damaged tool packages were not repaired exactly once"
    assert installs[0]["python"] == str(checkout.tools / "bin/python"), (
        "tool repair installed into host Python"
    )
    assert not (checkout.host / "packages.json").exists(), (
        "host dependency isolation changed"
    )


@pytest.mark.parametrize(
    "stage",
    [
        "host",
        "tool-venv",
        "pip-prepare",
        "packages",
        "package-verification",
        "pip-check",
        "download",
    ],
)
def test_any_failed_substep_fails_the_online_entry(
    checkout: Checkout, stage: str
) -> None:
    result = checkout.make(
        environment={
            "FAIL_SETUP_STAGE": stage,
            **({"HOST_WAIT_FOR": "tool-entry"} if stage == "host" else {}),
        }
    )
    assert result.returncode != 0, "a failed setup substep reported overall success"
    kinds = [event["kind"] for event in checkout.recorded()]
    if stage == "host":
        assert "host-ready" not in kinds, "the failed host was reported ready"
        assert "promtool-check" in kinds, (
            "Make returned before the already-started independent tool branch finished"
        )
    else:
        assert (
            "fixture-private-failed-command-output" not in result.stdout + result.stderr
        ), "a failed installer leaked captured output"
        assert "host-ready" in kinds, "Make returned before host preparation finished"
        assert "promtool-check" not in kinds, (
            "setup continued past an earlier failed tool substep"
        )


@pytest.mark.parametrize("failed", ["packages", "download"])
def test_tool_failure_waits_for_the_inflight_peer(
    checkout: Checkout, failed: str
) -> None:
    waiting = "DOWNLOAD_WAIT_FOR" if failed == "packages" else "PACKAGES_WAIT_FOR"
    result = checkout.make(
        environment={"FAIL_SETUP_STAGE": failed, waiting: failed + "-failed"}
    )
    assert result.returncode != 0, "parallel tool setup hid a worker failure"
    kinds = [event["kind"] for event in checkout.recorded()]
    completed = "download-complete" if failed == "packages" else "packages-ready"
    assert kinds.index(failed + "-failed") < kinds.index(completed), (
        "setup returned without waiting for the in-flight peer"
    )
    assert not list(checkout.tools.glob(".promtool-*")), (
        "a failed setup left unfinished native tool staging directories"
    )
    assert "promtool-check" not in kinds, "failed tool setup proceeded to verification"


def test_tool_preflight_does_not_create_environments_or_downloads(
    checkout: Checkout,
) -> None:
    result = checkout.make(
        "ci-supply-chain-tools", "_SUPPLY_CHAIN_SETUP_VALIDATE_ONLY=1"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"status": "validated"' in result.stdout, (
        "input preflight must distinguish validation from installation"
    )
    assert '"status": "ready"' not in result.stdout, (
        "input preflight claimed installed tools"
    )
    assert [event["kind"] for event in checkout.recorded()] == ["tool-preflight"], (
        "input preflight started an installer or downloader"
    )
    assert not checkout.host.exists() and not checkout.tools.exists(), (
        "input preflight changed an environment"
    )


def test_tool_only_check_runs_without_site_packages(checkout: Checkout) -> None:
    installed = checkout.make()
    assert installed.returncode == 0, installed.stdout + installed.stderr
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(ROOT / "scripts/check-alert-rules.py"),
            "--tool-only",
            "--promtool",
            str(checkout.tools / "bin/promtool"),
            "--version",
            VERSION,
        ],
        env=checkout.env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["status"] == "ready", (
        "tool-only verification still depends on installed Python packages"
    )


def test_archive_hash_mismatch_is_fatal_before_native_install(
    checkout: Checkout,
) -> None:
    result = checkout.make(
        "deploy-host-setup-online", f"PROMTOOL_SHA256_LINUX_AMD64={'0' * 64}"
    )
    assert result.returncode != 0, "an archive with a different digest was accepted"
    assert "SHA-256" in result.stderr, "digest failure lost its explanation"
    assert not (checkout.tools / "bin/promtool").exists(), (
        "unverified archive bytes were installed"
    )


def test_a_corrupted_cached_archive_is_replaced_and_reverified(
    checkout: Checkout,
) -> None:
    first = checkout.make()
    assert first.returncode == 0, first.stdout + first.stderr
    (checkout.tools / "promtool.tar.gz").write_bytes(b"corrupted cached archive")
    previous = len(checkout.recorded())
    result = checkout.make()
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        sum(event["kind"] == "download" for event in checkout.recorded()[previous:])
        == 1
    ), "corrupted archive was reused without downloading the pinned replacement"
    assert (
        hashlib.sha256((checkout.tools / "promtool.tar.gz").read_bytes()).hexdigest()
        == checkout.digest
    ), "replacement archive was not checked against the Make pin"


@pytest.mark.parametrize("defect", ["version", "symlink", "missing-member"])
def test_verified_archive_still_requires_the_right_native_tool(
    checkout: Checkout, defect: str
) -> None:
    checkout.digest = archive(
        checkout.download,
        reported_version="0.0.0" if defect == "version" else VERSION,
        member="file" if defect == "version" else defect,
    )
    result = checkout.make()
    assert result.returncode != 0, "archive digest alone was treated as tool readiness"
    assert '"status": "ready"' not in result.stdout, (
        "invalid native tool was reported ready"
    )


def test_standalone_ci_tool_target_remains_independent_of_host_setup(
    checkout: Checkout,
) -> None:
    result = checkout.make("ci-supply-chain-tools", "CI=true")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(
        event["kind"].startswith("host-") for event in checkout.recorded()
    ), "CI tool preparation unexpectedly invoked host setup"
    assert (checkout.tools / "bin/promtool").is_file(), (
        "standalone tool target stopped installing"
    )


def test_explicit_tools_python_with_spaces_keeps_the_selected_venv(
    checkout: Checkout,
) -> None:
    result = checkout.make(
        "deploy-host-setup-online",
        f"SUPPLY_CHAIN_PYTHON={checkout.tools / 'bin/python'}",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (checkout.tools / "bin/promtool").is_file(), (
        "the explicit tool interpreter changed the installation destination"
    )


def test_tool_python_must_prove_it_is_a_virtual_environment(checkout: Checkout) -> None:
    result = checkout.make(environment={"FAKE_UNISOLATED_TOOLS": "true"})
    assert result.returncode != 0, "a base interpreter was accepted as the tool venv"
    assert not any(event["kind"] == "packages" for event in checkout.recorded()), (
        "packages were installed before Python isolation was proved"
    )


def test_standalone_installer_cannot_modify_its_running_venv(
    checkout: Checkout,
) -> None:
    locked = checkout.root / "running locked venv"
    venv.EnvBuilder(with_pip=False).create(locked)
    result = checkout.make(
        "ci-supply-chain-tools",
        f"PYTHON={locked / 'bin/python'}",
        f"SUPPLY_CHAIN_TOOLS_VENV={locked}",
    )
    assert result.returncode != 0, (
        "the running environment became the tools environment"
    )
    assert "venvs must be separate" in result.stderr, (
        "the running venv was not rejected before package installation"
    )
    assert not checkout.recorded(), "the isolation guard started installer commands"


def test_offline_signed_setup_has_no_tool_dependency(checkout: Checkout) -> None:
    bundle, signature = (
        checkout.root / "signed bundle.tar.gz",
        checkout.root / "bundle signature.json",
    )
    bundle.write_bytes(b"offline signed fixture")
    signature.write_text("{}", encoding="utf-8")
    result = checkout.make(
        "deploy-host-setup",
        f"DEPLOY_HOST_ARCHIVE={bundle}",
        f"DEPLOY_HOST_SIGNATURE_BUNDLE={signature}",
        "PROMTOOL=/nonexistent/native/tool",
        environment={"FAIL_SETUP_STAGE": "download"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    events = checkout.recorded()
    assert [event["kind"] for event in events] == ["host-start", "host-ready"], (
        "offline setup acquired an online tool or test dependency"
    )
    arguments = events[0]["args"]
    assert arguments[arguments.index("--bundle") + 1] == str(bundle), (
        "signed archive was not forwarded"
    )
    assert arguments[arguments.index("--signature-bundle") + 1] == str(signature), (
        "offline signature verification input changed"
    )
    assert "--allow-network" not in arguments, (
        "offline setup was authorized to use the network"
    )


@pytest.mark.parametrize(
    "binding",
    [
        "empty-tool",
        "same-venv",
        "nested-tools",
        "nested-host",
        "mismatched-python",
        "host-promtool",
        "invalid-version",
        "invalid-checksum",
    ],
)
def test_conflicting_or_empty_explicit_paths_fail_closed(
    checkout: Checkout, binding: str
) -> None:
    assignment = {
        "empty-tool": "PROMTOOL=",
        "same-venv": f"SUPPLY_CHAIN_TOOLS_VENV={checkout.host}",
        "nested-tools": f"SUPPLY_CHAIN_TOOLS_VENV={checkout.host / 'tools'}",
        "nested-host": f"DEPLOY_HOST_VENV={checkout.tools / 'host'}",
        "mismatched-python": f"SUPPLY_CHAIN_PYTHON={checkout.root / 'other tools/bin/python'}",
        "host-promtool": f"PROMTOOL={checkout.host / 'bin/promtool'}",
        "invalid-version": "PIP_AUDIT_VERSION=unversioned",
        "invalid-checksum": "PROMTOOL_SHA256_LINUX_AMD64=invalid",
    }[binding]
    result = checkout.make("deploy-host-setup-online", assignment)
    assert result.returncode != 0, "invalid explicit paths were silently replaced"
    assert [event["kind"] for event in checkout.recorded()] == ["tool-preflight"], (
        "invalid inputs allowed host installation, tool installation or downloads"
    )
