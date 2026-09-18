from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tests._script_loader import load_script_module

SLOTS = Path(__file__).resolve().parents[2] / "deploy/node/runtime-slot.sh"
ARTIFACT = "a" * 64


def run_slot(command: str, **environment: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            'source "$1"; ' + command,
            "slots",
            str(SLOTS),
        ],
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHON_COMMAND": sys.executable,
            **environment,
        },
        check=False,
        capture_output=True,
        text=True,
    )


def slot_identity(lock: Path, root: Path) -> tuple[str, str]:
    result = run_slot(
        'RUNTIME_DEPENDENCY_SHA256="$(runtime_dependency_identity)"; '
        'printf "%s\\n" "$RUNTIME_DEPENDENCY_SHA256"; runtime_slot_directory',
        DEPENDENCY_LOCK=str(lock),
        RUNTIME_RELEASES_DIR=str(root),
        WHEEL_SHA256=ARTIFACT,
    )
    assert result.returncode == 0, result.stderr
    digest, path = result.stdout.splitlines()
    return digest, path


def test_same_dependency_bytes_reuse_the_same_slot(tmp_path: Path) -> None:
    first = tmp_path / "first.lock"
    second = tmp_path / "second.lock"
    first.write_text("dependency==1.0\n", encoding="utf-8")
    second.write_bytes(first.read_bytes())

    assert slot_identity(first, tmp_path) == slot_identity(second, tmp_path), (
        "a lock path change must not invalidate identical dependencies"
    )


def test_lock_only_upgrade_gets_a_new_slot_without_replacing_previous(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "dependencies.lock"
    lock.write_text("dependency==1.0\n", encoding="utf-8")
    old_digest, old_path = slot_identity(lock, tmp_path)
    previous = Path(old_path)
    previous.mkdir()
    marker = previous / "previous-runtime"
    marker.write_text("keep", encoding="utf-8")
    lock.write_text("dependency==2.0\n", encoding="utf-8")

    new_digest, new_path = slot_identity(lock, tmp_path)

    assert old_digest != new_digest
    assert old_path != new_path, "same wheel with a new lock reused the old slot"
    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("dependency", [None, "old-lock"])
def test_legacy_or_mismatched_dependency_marker_is_not_reusable(
    tmp_path: Path, dependency: str | None
) -> None:
    slot = tmp_path / "slot"
    slot.mkdir()
    (slot / ".complete").touch()
    (slot / "artifact.sha256").write_text(ARTIFACT, encoding="utf-8")
    (slot / "record.sha256").write_text("record", encoding="utf-8")
    if dependency is not None:
        (slot / "dependency.sha256").write_text(dependency, encoding="utf-8")

    result = run_slot(
        'validate_runtime_slot "$SLOT" "$ARTIFACT"',
        SLOT=str(slot),
        ARTIFACT=ARTIFACT,
        RUNTIME_DEPENDENCY_SHA256="new-lock",
    )

    assert result.returncode == 1, "a stale dependency slot was accepted"


def test_invalid_active_slot_is_never_removed(tmp_path: Path) -> None:
    marker = tmp_path / "active-runtime"
    marker.write_text("keep", encoding="utf-8")

    result = run_slot(
        'die() { printf "%s\\n" "$*" >&2; exit 1; }; '
        'prepare_runtime_slot "$SLOT" "$ARTIFACT"',
        SLOT=str(tmp_path),
        ARTIFACT=ARTIFACT,
        PREVIOUS_CURRENT_TARGET=str(tmp_path),
    )

    assert result.returncode == 1
    assert "active node runtime slot failed integrity validation" in result.stderr
    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("previous_status", [0, 1])
def test_installer_validates_previous_before_preparing_candidate(
    previous_status: int,
) -> None:
    result = run_slot(
        'runtime_dependency_identity() { printf "dependency"; }; '
        'runtime_slot_directory() { printf "candidate"; }; '
        'validate_previous_node_runtime() { printf "previous\\n"; return "$PREVIOUS_STATUS"; }; '
        'prepare_runtime_slot() { printf "prepare:%s:%s\\n" "$1" "$2"; }; '
        "prepare_node_runtime",
        PREVIOUS_STATUS=str(previous_status),
        WHEEL_SHA256=ARTIFACT,
    )
    assert result.returncode == previous_status
    assert result.stdout.splitlines() == (
        ["previous", f"prepare:candidate:{ARTIFACT}"]
        if previous_status == 0
        else ["previous"]
    ), "candidate preparation preceded successful rollback validation"


def wheel(
    directory: Path,
    *,
    project: bool,
    version: str = "1.0",
    collector_plugin: bool = False,
    collector_registry: str | None = None,
) -> Path:
    name = "gpu_fault_node_runtime" if project else "slot_dependency"
    module = "gpu_fault" if project else "slot_dependency"
    metadata = f"{name}-{version}.dist-info"
    members = {
        f"{module}/__init__.py": f'VERSION = "{version}"\ndef main(): pass\n'.encode(),
        f"{metadata}/METADATA": (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            + ("Requires-Dist: slot-dependency==1.0\n" if project else "")
        ).encode(),
        f"{metadata}/WHEEL": b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    if project:
        members[f"{metadata}/entry_points.txt"] = (
            "[console_scripts]\n"
            "gpu-fault-collector = gpu_fault:main\n"
            "gpu-fault-node-agent = gpu_fault:main\n"
            "gpu-fault-restore-gpu-services = gpu_fault:main\n"
            + (
                "[gpu_fault.collectors]\nprobe = gpu_fault:main\n"
                if collector_plugin
                else ""
            )
        ).encode()
        if collector_registry is not None:
            members[f"{module}/collector_registry.py"] = collector_registry.encode()
    record = io.StringIO()
    writer = csv.writer(record)
    for path, data in members.items():
        encoded = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        )
        writer.writerow([path, f"sha256={encoded}", len(data)])
    writer.writerow([f"{metadata}/RECORD", "", ""])
    members[f"{metadata}/RECORD"] = record.getvalue().encode()
    target = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(target, "w") as archive:
        for path, data in members.items():
            archive.writestr(path, data)
    return target


@pytest.mark.parametrize(
    ("registry", "success"),
    [
        (None, False),
        (
            "def collector_registry_with_plugins():\n    return {}\n"
            "def validate_collector_plugins():\n"
            "    raise RuntimeError('invalid installed plugin')\n",
            False,
        ),
        (
            "def collector_registry_with_plugins():\n"
            "    raise AssertionError('full validation should take precedence')\n"
            "def validate_collector_plugins():\n"
            "    print('all-plugins-validated')\n"
            "    return {}\n",
            True,
        ),
        (
            "def collector_registry_with_plugins():\n"
            "    print('all-plugins-validated')\n"
            "    return {}\n",
            True,
        ),
        (
            "from types import SimpleNamespace\n"
            "def collector_registry_with_plugins():\n"
            "    return {'probe': SimpleNamespace(cli_command='probe', "
            "factory='gpu_fault:missing_factory')}\n",
            False,
        ),
        (
            "from types import SimpleNamespace\n"
            "def collector_registry_with_plugins():\n"
            "    return {'probe': SimpleNamespace(cli_command='probe', "
            "factory='gpu_fault:VERSION')}\n",
            False,
        ),
    ],
)
def test_candidate_plugin_validation_precedes_activation(
    tmp_path: Path, registry: str | None, success: bool
) -> None:
    dependency = wheel(tmp_path, project=False)
    project = wheel(
        tmp_path, project=True, collector_plugin=True, collector_registry=registry
    )
    lock = tmp_path / "node.lock"
    lock.write_text(
        "slot-dependency==1.0 --hash=sha256:"
        + hashlib.sha256(dependency.read_bytes()).hexdigest()
        + "\n"
    )
    result = run_slot(
        'die() { printf "%s\\n" "$*" >&2; exit 1; }; '
        'RUNTIME_DEPENDENCY_SHA256="$(runtime_dependency_identity)"; '
        'prepare_runtime_slot "$(runtime_slot_directory)" "$WHEEL_SHA256"; '
        'printf "activation-allowed\\n"',
        DEPENDENCY_LOCK=str(lock),
        WHEEL=str(project),
        WHEEL_SHA256=hashlib.sha256(project.read_bytes()).hexdigest(),
        WHEELHOUSE=str(tmp_path),
        RUNTIME_RELEASES_DIR=str(tmp_path / "releases"),
        PREVIOUS_CURRENT_TARGET="",
    )
    assert (result.returncode == 0) is success, result.stdout + result.stderr
    assert ("activation-allowed" in result.stdout) is success
    if success:
        assert "all-plugins-validated" in result.stdout
    else:
        assert "candidate node runtime slot validation failed" in result.stderr


@pytest.fixture(scope="module")
def installed(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, str]]:
    root = tmp_path_factory.mktemp("verified-node-runtime")
    dependency = wheel(root, project=False)
    project = wheel(root, project=True)
    lock = root / "node.lock"
    lock.write_text(
        f"slot-dependency==1.0 --hash=sha256:{hashlib.sha256(dependency.read_bytes()).hexdigest()}\n"
    )
    environment = {
        "DEPENDENCY_LOCK": str(lock),
        "WHEEL": str(project),
        "WHEEL_SHA256": hashlib.sha256(project.read_bytes()).hexdigest(),
        "WHEELHOUSE": str(root),
        "RUNTIME_RELEASES_DIR": str(root / "releases"),
        "PREVIOUS_CURRENT_TARGET": "",
    }
    result = run_slot(
        'die() { printf "%s\\n" "$*" >&2; exit 1; }; '
        'RUNTIME_DEPENDENCY_SHA256="$(runtime_dependency_identity)"; '
        'prepare_runtime_slot "$(runtime_slot_directory)" "$WHEEL_SHA256"',
        **environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return root, environment


def test_wheel_upgrade_reuses_the_verified_dependency_layer(
    installed: tuple[Path, dict[str, str]],
) -> None:
    root, environment = installed
    layer = next((root / "dependencies").iterdir())
    record = layer / "record.sha256"
    before = (record.stat().st_mtime_ns, record.read_bytes())
    project = wheel(root, project=True, version="2.0")
    result = run_slot(
        'die() { printf "%s\\n" "$*" >&2; exit 1; }; '
        'RUNTIME_DEPENDENCY_SHA256="$(runtime_dependency_identity)"; '
        'prepare_runtime_slot "$(runtime_slot_directory)" "$WHEEL_SHA256"',
        **{
            **environment,
            "WHEEL": str(project),
            "WHEEL_SHA256": hashlib.sha256(project.read_bytes()).hexdigest(),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Reusing verified node dependency layer" in result.stdout
    assert (record.stat().st_mtime_ns, record.read_bytes()) == before
    assert len(list((root / "releases").iterdir())) == 2


@pytest.mark.parametrize(
    "damage",
    [
        "application",
        "dependency",
        "version",
        "extra",
        "record",
        "pth",
        "bytecode",
        "symlink",
    ],
)
def test_cached_runtime_verifies_files_versions_and_isolation(
    installed: tuple[Path, dict[str, str]], tmp_path: Path, damage: str
) -> None:
    original, environment = installed
    root = tmp_path / "copy"
    shutil.copytree(original, root, symlinks=True)
    digest_value, path = slot_identity(
        Path(environment["DEPENDENCY_LOCK"]), root / "releases"
    )
    # The general slot_identity helper uses a fixed wheel digest.
    slot = root / "releases" / f"{environment['WHEEL_SHA256']}-{digest_value}"
    layer = root / "dependencies" / digest_value
    site_relative = Path(
        f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    )
    site = slot / "venv" / site_relative
    dependency_site = layer / "venv" / site_relative
    for candidate in (root / "releases").iterdir():
        (
            candidate / "venv" / site_relative / "gpu-fault-node-dependencies.pth"
        ).write_text(str(dependency_site) + "\n")
    env = {
        **environment,
        "RUNTIME_RELEASES_DIR": str(root / "releases"),
        "RUNTIME_DEPENDENCY_SHA256": digest_value,
        "SLOT": str(slot),
    }
    assert (
        run_slot('validate_runtime_slot "$SLOT" "$WHEEL_SHA256"', **env).returncode == 0
    )
    application = site / "gpu_fault/__init__.py"
    if damage == "application":
        application.write_text("BROKEN = True\n")
    elif damage == "dependency":
        (dependency_site / "slot_dependency/__init__.py").write_text("BROKEN = True\n")
    elif damage == "version":
        metadata = dependency_site / "slot_dependency-1.0.dist-info/METADATA"
        metadata.write_text(
            metadata.read_text().replace("Version: 1.0", "Version: 9.0")
        )
    elif damage == "extra":
        (site / "unexpected.py").write_text("unexpected = True\n")
    elif damage == "record":
        (site / "gpu_fault_node_runtime-1.0.dist-info/RECORD").write_text("")
    elif damage == "pth":
        (site / "gpu-fault-node-dependencies.pth").write_text(
            f"import pathlib; pathlib.Path({str(root / 'must-not-execute')!r}).touch()\n"
        )
    elif damage == "bytecode":
        next((site / "gpu_fault/__pycache__").glob("*.pyc")).write_bytes(
            b"invalid bytecode"
        )
    else:
        application.unlink()
        application.symlink_to(original / "outside-environment.py")
    result = run_slot('validate_runtime_slot "$SLOT" "$WHEEL_SHA256"', **env)
    assert result.returncode != 0, f"{damage} was accepted"
    assert not (root / "must-not-execute").exists(), "an unvalidated .pth file executed"


def test_dependency_seal_precedes_importing_its_requirement_parser(
    installed: tuple[Path, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, environment = installed
    helper = load_script_module(SLOTS.with_name("runtime_integrity.py"))
    monkeypatch.setattr(
        helper,
        "locked_versions",
        lambda *_: pytest.fail(
            "imported the dependency parser before checking its seal"
        ),
    )
    seal = tmp_path / "wrong-seal"
    seal.write_text("0" * 64)
    layer = next((root / "dependencies").iterdir()) / "venv"
    with pytest.raises(ValueError, match="RECORD seal"):
        helper.validate(layer, Path(environment["DEPENDENCY_LOCK"]), None, None, seal)


def test_interrupted_unpublished_dependency_build_can_resume(
    installed: tuple[Path, dict[str, str]], tmp_path: Path
) -> None:
    _root, environment = installed
    dependency, _slot = slot_identity(Path(environment["DEPENDENCY_LOCK"]), tmp_path)
    partial = tmp_path / "dependencies" / dependency
    partial.mkdir(parents=True)
    (partial / ".building").touch()
    (partial / "partial-output").touch()
    result = run_slot(
        'die() { printf "%s\\n" "$*" >&2; exit 1; }; prepare_dependency_layer',
        **{
            **environment,
            "RUNTIME_DEPENDENCIES_DIR": str(tmp_path / "dependencies"),
            "RUNTIME_DEPENDENCY_SHA256": dependency,
        },
    )
    assert result.returncode == 0, result.stderr
    assert (partial / ".complete").is_file(), (
        "resumed layer was not validated and sealed"
    )
    assert not (partial / "partial-output").exists(), (
        "incomplete output survived rebuilding"
    )


def test_published_dependency_layer_is_never_rebuilt_in_place(
    installed: tuple[Path, dict[str, str]], tmp_path: Path
) -> None:
    _root, environment = installed
    dependency, _slot = slot_identity(Path(environment["DEPENDENCY_LOCK"]), tmp_path)
    published = tmp_path / "dependencies" / dependency
    published.mkdir(parents=True)
    (published / ".complete").touch()
    marker = published / "retained-runtime"
    marker.write_text("keep")
    result = run_slot(
        'die() { printf "%s\\n" "$*" >&2; exit 1; }; prepare_dependency_layer',
        **{
            **environment,
            "RUNTIME_DEPENDENCIES_DIR": str(tmp_path / "dependencies"),
            "RUNTIME_DEPENDENCY_SHA256": dependency,
        },
    )
    assert result.returncode == 1
    assert "published node dependency layer" in result.stderr
    assert marker.read_text() == "keep"


def test_valid_python_alias_venv_keeps_binary_identity_and_passes_validation(
    tmp_path: Path,
) -> None:
    helper = load_script_module(SLOTS.with_name("runtime_integrity.py"))
    alias = tmp_path / "python-alias"
    alias.symlink_to(Path(sys.executable).resolve())
    venv = tmp_path / "venv"
    subprocess.run(
        [str(alias), "-m", "venv", "--without-pip", str(venv)],
        check=True,
        capture_output=True,
    )
    helper.validate_interpreter(venv)
    subprocess.run(
        [str(venv / "bin/python"), "-I", "-c", "pass"], check=True, capture_output=True
    )
    lock = tmp_path / "node.lock"
    lock.write_text("same dependency inputs\n")
    canonical = run_slot("runtime_dependency_identity", DEPENDENCY_LOCK=str(lock))
    aliased = run_slot(
        "runtime_dependency_identity",
        DEPENDENCY_LOCK=str(lock),
        PYTHON_COMMAND=str(alias),
    )
    assert canonical.returncode == aliased.returncode == 0
    assert canonical.stdout == aliased.stdout


def test_arbitrary_venv_home_is_not_accepted_as_a_python_alias(tmp_path: Path) -> None:
    helper = load_script_module(SLOTS.with_name("runtime_integrity.py"))
    venv = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)],
        check=True,
        capture_output=True,
    )
    config = venv / "pyvenv.cfg"
    lines = config.read_text().splitlines()
    config.write_text(
        "\n".join(
            f"home = {tmp_path / 'unrelated'}" if line.startswith("home = ") else line
            for line in lines
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="base interpreter configuration"):
        helper.validate_interpreter(venv)


def previous_runtime(
    installed: tuple[Path, dict[str, str]], directory: Path
) -> tuple[Path, Path, Path, str]:
    original, environment = installed
    root = directory / "runtime"
    shutil.copytree(original, root, symlinks=True)
    dependency, _ = slot_identity(Path(environment["DEPENDENCY_LOCK"]), root)
    slot = root / "releases" / f"{environment['WHEEL_SHA256']}-{dependency}"
    layer = root / "dependencies" / dependency / "venv"
    helper = load_script_module(SLOTS.with_name("runtime_integrity.py"))
    for candidate in (root / "releases").iterdir():
        (helper.site_path(candidate / "venv") / helper.LAYER_PTH).write_text(
            str(helper.site_path(layer)) + "\n"
        )
    (root / "current").symlink_to(slot)
    (root / "installed-units.txt").write_text("gpu-fault-node-agent.service\n")
    return root, slot, layer, environment["WHEEL_SHA256"]


def test_previous_slot_is_verified_against_its_own_dependency_lock(
    installed: tuple[Path, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _slot, _layer, artifact = previous_runtime(installed, tmp_path)
    candidate_lock = tmp_path / "candidate.lock"
    candidate_lock.write_text("different-candidate-dependency==99\n")
    monkeypatch.setenv("DEPENDENCY_LOCK", str(candidate_lock))
    helper = load_script_module(SLOTS.with_name("runtime_integrity.py"))
    result = helper.validate_previous_runtime(root, artifact)
    assert result["artifact_sha256"] == artifact


def test_damaged_previous_dependency_blocks_upgrade_even_with_a_different_candidate(
    installed: tuple[Path, dict[str, str]], tmp_path: Path
) -> None:
    root, _slot, layer, artifact = previous_runtime(installed, tmp_path)
    helper = load_script_module(SLOTS.with_name("runtime_integrity.py"))
    (helper.site_path(layer) / "slot_dependency/__init__.py").write_text(
        "damaged = True\n"
    )
    with pytest.raises(ValueError, match="differs from RECORD"):
        helper.validate_previous_runtime(root, artifact)


def test_previous_slot_never_executes_an_unverified_pth(
    installed: tuple[Path, dict[str, str]], tmp_path: Path
) -> None:
    root, slot, _layer, artifact = previous_runtime(installed, tmp_path)
    helper = load_script_module(SLOTS.with_name("runtime_integrity.py"))
    marker = tmp_path / "must-not-execute"
    (helper.site_path(slot / "venv") / helper.LAYER_PTH).write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).touch()\n"
    )
    with pytest.raises(ValueError, match="dependency reference"):
        helper.validate_previous_runtime(root, artifact)
    assert not marker.exists(), "previous environment ran before it was validated"


def test_previous_slot_refuses_an_unexpected_artifact_identity(
    installed: tuple[Path, dict[str, str]], tmp_path: Path
) -> None:
    root, _slot, _layer, _artifact = previous_runtime(installed, tmp_path)
    helper = load_script_module(SLOTS.with_name("runtime_integrity.py"))
    with pytest.raises(ValueError, match="artifact identity differs"):
        helper.validate_previous_runtime(root, "0" * 64)


@pytest.mark.parametrize("legacy_directory", [False, True])
def test_previous_legacy_runtime_is_checked_without_requiring_the_candidate_lock(
    installed: tuple[Path, dict[str, str]], tmp_path: Path, legacy_directory: bool
) -> None:
    root, slot, layer, artifact = previous_runtime(installed, tmp_path)
    helper = load_script_module(SLOTS.with_name("runtime_integrity.py"))
    site = helper.site_path(slot / "venv")
    (site / helper.LAYER_PTH).unlink()
    for dependency in helper.site_path(layer).glob("slot_dependency*"):
        shutil.copytree(dependency, site / dependency.name)
    (slot / "dependency.sha256").unlink()
    _versions, records = helper.verify_records(slot / "venv")
    (slot / "record.sha256").write_text(records[helper.PROJECT])
    if legacy_directory:
        (root / "current").unlink()
        shutil.move(slot / "venv", root / "venv")
        (root / "runtime-artifact-sha256").write_text(artifact)
    result = helper.validate_previous_runtime(root, artifact)
    assert result["artifact_sha256"] == artifact
