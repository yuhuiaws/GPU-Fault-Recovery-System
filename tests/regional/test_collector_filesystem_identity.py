"""Schema-specific filesystem custody, with no real host commands or state."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

import pytest

from scripts.e2e.regional.probes import collector_node_probe as probe
from tests.regional import test_collector_env_safety as env_safety
from tests.regional.test_collector_env_safety import (
    ENV_TEXT,
    OWNER_NONCE,
    EnvHost,
    forbidden,
    restore_main,
)

env_host = env_safety.env_host


LEGACY_INTENT_FIELDS = (
    "schema_version",
    "run_id",
    "owner_nonce_sha256",
    "cluster_id",
    "node_id",
    "node_instance_id",
    "boot_id",
    "baseline",
    "override",
    "baseline_sha256",
    "applied_sha256",
    "created_monotonic",
    "created_epoch",
    "restore_seconds",
    "restore_deadline_monotonic",
    "restore_deadline_epoch",
    "python",
    "python_sha256",
    "runtime_path",
    "runtime_device",
    "runtime_inode",
    "probe_sha256",
    "host_unit_path",
    "host_unit_sha256",
)


def device_enumeration(
    host: EnvHost, monkeypatch: pytest.MonkeyPatch
) -> Callable[[int], None]:
    original_stat = Path.stat
    runtime = probe.COLLECTOR_RUNTIME.resolve()
    current = [os.makedev(259, 2)]

    def stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        info = original_stat(path, follow_symlinks=follow_symlinks)
        if path not in {runtime, host.root}:
            return info
        fields = list(info)
        fields[2] = current[0]
        return os.stat_result(fields)

    def change(device: int) -> None:
        current[0] = device
        host.filesystem["maj:min"] = f"{os.major(device)}:{os.minor(device)}"
        host.write_mountinfo()

    monkeypatch.setattr(Path, "stat", stat)
    change(current[0])
    return change


def legacy_digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {key: record[key] for key in LEGACY_INTENT_FIELDS},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def seed_legacy_operation(host: EnvHost) -> dict[str, Any]:
    """Produce old-format fixture files before invoking any legacy recovery."""
    host.override()
    record = host.record()
    record["schema_version"] = 2
    record.pop("runtime_filesystem")
    record["intent_sha256"] = legacy_digest(record)
    backup, _ = host.paths()
    probe.power_write_json(probe.collector_override_record(backup), record)
    for path in (
        probe.ACCEPTANCE_STATE / "collector-env-owner.json",
        backup.with_suffix(".armed.json"),
    ):
        receipt = json.loads(path.read_bytes())
        receipt["intent_sha256"] = record["intent_sha256"]
        probe.power_write_json(path, receipt)
    return record


def assert_guard_preserved(
    host: EnvHost, record: dict[str, Any], *, automatic: bool = False
) -> None:
    before = probe.COLLECTOR_ENV.read_bytes()
    path = probe.collector_override_record(host.paths()[0])
    stored = path.read_bytes()
    host.calls.clear()
    with pytest.raises((probe.CollectorEnvGuardError, OSError)):
        host.restore(automatic=automatic)
    assert probe.COLLECTOR_ENV.read_bytes() == before
    assert path.read_bytes() == stored
    assert host.record() == record
    assert host.paths()[0].exists(), (
        "assert_guard_preserved: expected host.paths()[0].exists()"
    )
    assert host.mutations() == []


@pytest.mark.parametrize("automatic", [False, True])
def test_same_filesystem_survives_only_cross_boot_nvme_device_renumbering(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch, automatic: bool
) -> None:
    change_device = device_enumeration(env_host, monkeypatch)
    ack = env_host.override()
    original = env_host.record()
    assert original["runtime_device"] == 66306
    probe.BOOT_ID_FILE.write_text("boot-fixture-b")
    change_device(os.makedev(259, 1))
    assert probe.COLLECTOR_RUNTIME.resolve().stat().st_dev == 66305
    restored = env_host.restore(automatic=automatic)
    assert restored["restored"] is True
    assert restored["cleanup_verified"] is not automatic
    assert probe.COLLECTOR_ENV.read_text() == ENV_TEXT
    record = env_host.record()
    for field in probe.COLLECTOR_ENV_V3_INTENT_FIELDS:
        assert record[field] == original[field]
    assert record["intent_sha256"] == ack["intent_sha256"]
    if automatic:
        assert env_host.paths()[0].exists(), (
            "test_same_filesystem_survives_only_cross_boot_nvme_device_renumbering: expected env_host.paths()[0].exists()"
        )
        assert env_host.restore()["cleanup_verified"] is True
    assert env_host.record()["state"] == "CLEANED"
    assert env_host.record()["runtime_device"] == 66306
    assert not env_host.paths()[0].exists(), (
        "test_same_filesystem_survives_only_cross_boot_nvme_device_renumbering: expected no env_host.paths()[0].exists()"
    )


@pytest.mark.parametrize("automatic", [False, True])
def test_same_boot_device_change_cannot_authorize_restoration(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch, automatic: bool
) -> None:
    change_device = device_enumeration(env_host, monkeypatch)
    env_host.override()
    original = env_host.record()
    change_device(os.makedev(259, 1))
    assert_guard_preserved(env_host, original, automatic=automatic)


@pytest.mark.parametrize("boot", ["", " ", "boot\ninjected", "not a boot"])
def test_unknown_boot_cannot_explain_a_device_change(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch, boot: str
) -> None:
    change_device = device_enumeration(env_host, monkeypatch)
    env_host.override()
    original = env_host.record()
    probe.BOOT_ID_FILE.write_text(boot)
    change_device(os.makedev(259, 1))
    assert_guard_preserved(env_host, original)


@pytest.mark.parametrize(
    "drift",
    [
        "uuid",
        "fstype",
        "fsroot",
        "target",
        "runtime-inode",
        "runtime-path",
        "python-hash",
        "python-path",
        "node",
        "cluster",
        "instance",
    ],
)
def test_cross_boot_does_not_allow_any_other_identity_drift(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    env_host.override()
    original = env_host.record()
    probe.BOOT_ID_FILE.write_text("boot-fixture-b")
    runtime = probe.COLLECTOR_RUNTIME.resolve()
    if drift == "uuid":
        env_host.filesystem["uuid"] = "21c1b130-8c9e-437c-a0dc-0a0f5e1a7c70"
    elif drift == "fstype":
        env_host.filesystem["fstype"] = "xfs"
    elif drift == "fsroot":
        env_host.filesystem["fsroot"] = "/different-subvolume"
    elif drift == "target":
        env_host.filesystem["target"] = str(runtime)
    elif drift == "runtime-inode":
        original_stat = Path.stat

        def stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
            info = original_stat(path, follow_symlinks=follow_symlinks)
            if path != runtime:
                return info
            fields = list(info)
            fields[1] += 1
            return os.stat_result(fields)

        monkeypatch.setattr(Path, "stat", stat)
    elif drift == "runtime-path":
        other = runtime.with_name("different-runtime")
        other.mkdir()
        other.chmod(0o755)
        (other / "venv").symlink_to(runtime / "venv", target_is_directory=True)
        probe.COLLECTOR_RUNTIME.unlink()
        probe.COLLECTOR_RUNTIME.symlink_to(other, target_is_directory=True)
    elif drift == "python-hash":
        (runtime / "venv/bin/python").write_text("changed inert Python\n")
    elif drift == "python-path":
        old = runtime / "venv/bin/python"
        changed = runtime / "venv/bin/other-python"
        changed.write_bytes(old.read_bytes())
        changed.chmod(0o700)
        old.unlink()
        old.symlink_to(changed)
        monkeypatch.setattr(getattr(probe, "sys"), "executable", str(changed))
    else:
        probe.COLLECTOR_ENV.write_text(
            probe.COLLECTOR_ENV.read_text().replace(
                f"{drift}-fixture", f"{drift}-changed"
            )
        )
    env_host.write_mountinfo()
    if drift == "runtime-inode":
        current = probe.collector_runtime_identity(schema_version=3)
        assert {key for key, value in current.items() if original[key] != value} == {
            "runtime_inode"
        }
    assert_guard_preserved(env_host, original)


@pytest.mark.parametrize("cross_boot", [False, True])
@pytest.mark.parametrize("changed_device", [False, True])
def test_legacy_v2_device_and_intent_are_never_migrated_or_relaxed(
    env_host: EnvHost,
    monkeypatch: pytest.MonkeyPatch,
    cross_boot: bool,
    changed_device: bool,
) -> None:
    change_device = device_enumeration(env_host, monkeypatch)
    original = seed_legacy_operation(env_host)
    assert probe.collector_intent_digest(original) == legacy_digest(original)
    if cross_boot:
        probe.BOOT_ID_FILE.write_text("boot-fixture-b")
    if changed_device:
        change_device(os.makedev(259, 1))
    monkeypatch.setattr(subprocess, "run", forbidden)
    loaded = probe.collector_load_record(env_host.run_id, OWNER_NONCE)
    assert loaded == original
    if changed_device:
        assert_guard_preserved(env_host, original)
    else:
        assert env_host.restore()["cleanup_verified"] is True
    record = env_host.record()
    assert "runtime_filesystem" not in record
    assert record["schema_version"] == 2
    assert record["runtime_device"] == original["runtime_device"]
    assert record["intent_sha256"] == original["intent_sha256"]
    for key in LEGACY_INTENT_FIELDS:
        assert record[key] == original[key]


def test_v2_default_identity_never_requires_findmnt(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subprocess, "run", forbidden)
    identity = probe.collector_runtime_identity()
    assert set(identity) == {
        "runtime_path",
        "runtime_device",
        "runtime_inode",
        "python",
        "python_sha256",
    }
    assert env_host.findmnt_calls == []


@pytest.mark.parametrize("field", ["uuid", "fstype", "target", "fsroot"])
def test_v3_filesystem_fields_are_bound_by_the_version_specific_digest(
    env_host: EnvHost, field: str
) -> None:
    env_host.override()
    original = env_host.record()
    record = copy.deepcopy(original)
    changed = {
        "uuid": "21c1b130-8c9e-437c-a0dc-0a0f5e1a7c70",
        "fstype": "xfs",
        "target": "/different-mount",
        "fsroot": "/different-subvolume",
    }
    record["runtime_filesystem"][field] = changed[field]
    assert probe.collector_intent_digest(record) != original["intent_sha256"]
    path = probe.collector_override_record(env_host.paths()[0])
    probe.power_write_json(path, record)
    before = path.read_bytes()
    with pytest.raises(probe.CollectorEnvGuardError, match="intent or owner"):
        probe.collector_load_record(env_host.run_id, OWNER_NONCE)
    assert path.read_bytes() == before


def test_v2_digest_retains_exact_original_field_set(env_host: EnvHost) -> None:
    env_host.override()
    record = env_host.record()
    record["schema_version"] = 2
    assert probe.COLLECTOR_ENV_INTENT_FIELDS == LEGACY_INTENT_FIELDS
    expected = legacy_digest(record)
    assert probe.collector_intent_digest(record) == expected
    record["runtime_filesystem"] = {"not": "legacy input"}
    record["state"] = "RESTORING"
    assert probe.collector_intent_digest(record) == expected


@pytest.mark.parametrize("version", [0, 1, 4, True, 2.0, "2", None])
def test_unknown_schema_has_no_implicit_fallback(
    env_host: EnvHost, version: Any
) -> None:
    with pytest.raises(probe.CollectorEnvGuardError, match="schema"):
        probe.collector_runtime_identity(schema_version=version)
    with pytest.raises(probe.CollectorEnvGuardError, match="schema"):
        probe.collector_intent_digest({"schema_version": version})
    assert env_host.findmnt_calls == []


@pytest.mark.parametrize(
    "uuid",
    [
        None,
        "",
        " ",
        "-",
        "not-a-uuid",
        "00000000-0000-0000-0000-000000000000",
        "b953642a73294d11b89ed7a6d9a3ae2f",
        "{b953642a-7329-4d11-b89e-d7a6d9a3ae2f}",
        "b953642a-7329-4d11-b89e-d7a6d9a3ae2g",
        "b953642a-7329-4d11-b89e-d7a6d9a3ae2f\n",
        10,
        [],
    ],
    ids=[
        "null",
        "empty",
        "space",
        "missing",
        "invalid",
        "zero",
        "unhyphenated",
        "braces",
        "nonhex",
        "newline",
        "number",
        "list",
    ],
)
def test_invalid_filesystem_uuid_is_rejected_before_env_mutation(
    env_host: EnvHost, uuid: Any
) -> None:
    env_host.filesystem["uuid"] = uuid
    with pytest.raises(probe.CollectorEnvGuardError, match="UUID"):
        env_host.override()
    assert env_host.mutations() == []
    assert probe.COLLECTOR_ENV.read_text() == ENV_TEXT
    assert not (probe.ACCEPTANCE_STATE / "collector-env-owner.json").exists(), (
        'test_invalid_filesystem_uuid_is_rejected_before_env_mutation: expected no (probe.ACCEPTANCE_STATE / "collector-env-owner.json").exists()'
    )


def test_uuid_is_canonicalized_without_weakening_identity(env_host: EnvHost) -> None:
    env_host.filesystem["uuid"] = env_host.filesystem["uuid"].upper()
    env_host.override()
    assert (
        env_host.record()["runtime_filesystem"]["uuid"]
        == env_host.filesystem["uuid"].lower()
    )
    assert env_host.restore()["cleanup_verified"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("fstype", ""),
        ("fstype", None),
        ("fstype", "ext4\n"),
        ("fstype", "ext4;command"),
        ("fstype", "ext4" * 20),
        ("target", ""),
        ("target", "."),
        ("target", "/tmp/../other"),
        ("target", "/tmp//other"),
        ("target", "//tmp"),
        ("target", "/tmp/\x00"),
        ("target", "/tmp/\n"),
        ("target", True),
        ("fsroot", ""),
        ("fsroot", "relative"),
        ("fsroot", "/../"),
        ("fsroot", "/a/"),
        ("fsroot", "/a//b"),
        ("fsroot", "//"),
        ("fsroot", "/a/\x7f"),
        ("fsroot", []),
        ("fsroot", "/" + "a" * 4096),
        ("maj:min", ""),
        ("maj:min", None),
        ("maj:min", 66306),
        ("maj:min", "259:99"),
        ("maj:min", "0259:2"),
    ],
    ids=[
        "type-empty",
        "type-null",
        "type-newline",
        "type-command",
        "type-long",
        "target-empty",
        "target-relative",
        "target-parent",
        "target-double-slash",
        "target-leading-slash",
        "target-nul",
        "target-newline",
        "target-bool",
        "root-empty",
        "root-relative",
        "root-parent",
        "root-trailing-slash",
        "root-double-slash",
        "root-leading-slash",
        "root-control",
        "root-list",
        "root-long",
        "device-empty",
        "device-null",
        "device-number",
        "device-wrong",
        "device-noncanonical",
    ],
)
def test_malformed_findmnt_fields_fail_closed(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch, field: str, value: Any
) -> None:
    device_enumeration(env_host, monkeypatch)
    env_host.filesystem[field] = value
    with pytest.raises(probe.CollectorEnvGuardError):
        env_host.override()
    assert env_host.mutations() == []
    assert probe.COLLECTOR_ENV.read_text() == ENV_TEXT


@pytest.mark.parametrize(
    "shape",
    [
        "truncated",
        "not-object",
        "no-list",
        "null-list",
        "empty-list",
        "two-mounts",
        "non-object-mount",
        "children",
        "missing-field",
        "unknown-field",
        "duplicate-root",
        "duplicate-uuid",
        "huge",
        "nonzero",
        "timeout",
        "os-error",
    ],
)
def test_ambiguous_or_failed_findmnt_query_does_not_authorize_write(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    document: dict[str, Any] = {"filesystems": [dict(env_host.filesystem)]}
    if shape == "no-list":
        document = {}
    elif shape == "null-list":
        document["filesystems"] = None
    elif shape == "empty-list":
        document["filesystems"] = []
    elif shape == "two-mounts":
        document["filesystems"].append(dict(env_host.filesystem))
    elif shape == "non-object-mount":
        document["filesystems"] = ["mount"]
    elif shape == "children":
        document["filesystems"][0]["children"] = []
    elif shape == "missing-field":
        document["filesystems"][0].pop("uuid")
    elif shape == "unknown-field":
        document["filesystems"][0]["source"] = "/dev/unknown"
    output = json.dumps(document)
    if shape == "truncated":
        output = "{"
    elif shape == "not-object":
        output = "[]"
    elif shape == "duplicate-root":
        output = output[:-1] + ',"filesystems":[]}'
    elif shape == "duplicate-uuid":
        output = output.replace(
            '"uuid":', f'"uuid":"{env_host.filesystem["uuid"]}","uuid":', 1
        )
    elif shape == "huge":
        output += " " * 65537

    def query(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        env_host.findmnt(command, **kwargs)
        if shape == "timeout":
            raise subprocess.TimeoutExpired(command, 10, stderr="private-output")
        if shape == "os-error":
            raise OSError("private-output")
        return subprocess.CompletedProcess(
            command, 1 if shape == "nonzero" else 0, output, "private-output"
        )

    monkeypatch.setattr(subprocess, "run", query)
    with pytest.raises(probe.CollectorEnvGuardError) as rejected:
        env_host.override()
    assert "private-output" not in str(rejected.value)
    assert env_host.mutations() == []
    assert probe.COLLECTOR_ENV.read_text() == ENV_TEXT


@pytest.mark.parametrize(
    "drift",
    [
        "uuid-missing",
        "mount-missing",
        "duplicate-mount",
        "nested-mount",
        "wrong-fsroot",
        "wrong-type",
        "wrong-device",
        "malformed-row",
        "malformed-escape",
        "symlink-mountinfo",
        "writable-mountinfo",
        "mount-symlink",
    ],
)
def test_filesystem_query_requires_exact_current_kernel_mount(
    env_host: EnvHost, drift: str
) -> None:
    mountinfo = probe.PROC_ROOT / "self/mountinfo"
    original = mountinfo.read_text()
    mountinfo.chmod(0o600)
    if drift == "uuid-missing":
        env_host.filesystem["uuid"] = ""
    elif drift == "mount-missing":
        mountinfo.write_text(original.replace(str(env_host.root), "/different-root"))
    elif drift == "duplicate-mount":
        mountinfo.write_text(original + original.replace("7 1 ", "8 1 "))
    elif drift == "nested-mount":
        runtime = probe.COLLECTOR_RUNTIME.resolve()
        mountinfo.write_text(
            original
            + original.replace("7 1 ", "8 7 ").replace(str(env_host.root), str(runtime))
        )
    elif drift == "wrong-fsroot":
        mountinfo.write_text(original.replace(" / ", " /different-root ", 1))
    elif drift == "wrong-type":
        mountinfo.write_text(original.replace(" - ext4 ", " - xfs "))
    elif drift == "wrong-device":
        mountinfo.write_text(original.replace(env_host.filesystem["maj:min"], "99:99"))
    elif drift == "malformed-row":
        mountinfo.write_text("malformed")
    elif drift == "malformed-escape":
        mountinfo.write_text(original.replace(str(env_host.root), r"/invalid\777path"))
    elif drift == "symlink-mountinfo":
        other = mountinfo.with_name("foreign-mountinfo")
        other.write_text(original)
        other.chmod(0o444)
        mountinfo.unlink()
        mountinfo.symlink_to(other)
    elif drift == "writable-mountinfo":
        mountinfo.chmod(0o666)
    else:
        mount = env_host.root / "mount-link"
        mount.symlink_to(env_host.root, target_is_directory=True)
        env_host.filesystem["target"] = str(mount)
    with pytest.raises(probe.CollectorEnvGuardError):
        env_host.override()
    assert env_host.mutations() == []
    assert probe.COLLECTOR_ENV.read_text() == ENV_TEXT


@pytest.mark.parametrize(
    "unsafe",
    [
        "symlink",
        "parent-symlink",
        "writable",
        "parent-writable",
        "nonexec",
        "setuid",
        "hardlink",
    ],
)
def test_findmnt_tool_must_be_trusted_and_never_resolved_through_path(
    env_host: EnvHost, unsafe: str
) -> None:
    executable = probe.COLLECTOR_FINDMNT
    if unsafe == "symlink":
        saved = executable.with_name("saved")
        executable.rename(saved)
        executable.symlink_to(saved)
    elif unsafe == "parent-symlink":
        directory = executable.parent
        saved_dir = directory.with_name("saved-bin")
        directory.rename(saved_dir)
        directory.symlink_to(saved_dir, target_is_directory=True)
    elif unsafe == "hardlink":
        os.link(executable, executable.with_name("alias"))
    elif unsafe == "parent-writable":
        executable.parent.chmod(0o777)
    else:
        executable.chmod(
            {"writable": 0o777, "nonexec": 0o600, "setuid": 0o4700}[unsafe]
        )
    with pytest.raises(probe.CollectorEnvGuardError):
        env_host.override()
    assert env_host.findmnt_calls == []
    assert env_host.mutations() == []
    assert probe.COLLECTOR_ENV.read_text() == ENV_TEXT


@pytest.mark.parametrize("drift", ["tool", "mount", "runtime", "python"])
def test_identity_is_rechecked_after_the_filesystem_query(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    def query(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        completed = env_host.findmnt(command, **kwargs)
        if drift == "tool":
            probe.COLLECTOR_FINDMNT.write_text("changed inert tool")
        elif drift == "mount":
            env_host.filesystem["fsroot"] = "/changed"
            env_host.write_mountinfo()
        elif drift == "runtime":
            probe.COLLECTOR_RUNTIME.unlink()
            probe.COLLECTOR_RUNTIME.symlink_to(env_host.root / "missing")
        else:
            python = probe.COLLECTOR_RUNTIME.resolve() / "venv/bin/python"
            python.write_text("changed inert Python")
        return completed

    monkeypatch.setattr(subprocess, "run", query)
    with pytest.raises((probe.CollectorEnvGuardError, OSError)):
        env_host.override()
    assert env_host.mutations() == []
    assert probe.COLLECTOR_ENV.read_text() == ENV_TEXT


def test_findmnt_read_only_access_time_does_not_change_executable_identity(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    def query(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        completed = env_host.findmnt(command, **kwargs)
        # Real exec can advance atime without changing code or custody.
        with Path(command[0]).open("rb") as stream:
            assert stream.read() == b"inert findmnt executable fixture\n"
        return completed

    monkeypatch.setattr(subprocess, "run", query)
    env_host.override()
    assert env_host.restore()["cleanup_verified"] is True


def test_filesystem_guard_keeps_existing_durable_diagnostics_and_materials(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_host.override()
    original = env_host.record()
    probe.BOOT_ID_FILE.write_text("boot-fixture-b")
    env_host.filesystem["uuid"] = "21c1b130-8c9e-437c-a0dc-0a0f5e1a7c70"
    before = probe.COLLECTOR_ENV.read_bytes()
    env_host.calls.clear()
    assert restore_main(env_host, monkeypatch) == 78
    rejected = env_host.emitted[-1]
    assert rejected["failure_receipt"]["status"] == "RECORDED"
    assert (
        rejected["failure_receipt"]["receipt"]["intent_sha256"]
        == original["intent_sha256"]
    )
    assert rejected["cleanup_verified"] is False
    assert rejected["retryable"] is False
    assert env_host.record() == original
    assert env_host.paths()[0].exists(), (
        "test_filesystem_guard_keeps_existing_durable_diagnostics_and_materials: expected env_host.paths()[0].exists()"
    )
    assert probe.COLLECTOR_ENV.read_bytes() == before
    assert env_host.mutations() == []
    assert OWNER_NONCE not in json.dumps(rejected)
    assert "PRIVATE_FIXTURE_SENTINEL" not in json.dumps(rejected)


@pytest.mark.parametrize("fstype", ["ext4", "xfs", "btrfs"])
def test_stable_subvolume_root_and_filesystem_type_survive_reboot(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch, fstype: str
) -> None:
    change_device = device_enumeration(env_host, monkeypatch)
    env_host.filesystem["fstype"] = fstype
    env_host.filesystem["fsroot"] = "/gpu runtime/subvolume"
    env_host.write_mountinfo()
    env_host.override()
    original = env_host.record()
    assert original["runtime_filesystem"]["fsroot"] == "/gpu runtime/subvolume"
    probe.BOOT_ID_FILE.write_text("boot-fixture-b")
    change_device(os.makedev(259, 1))
    assert env_host.restore()["cleanup_verified"] is True
    assert env_host.record()["runtime_filesystem"] == original["runtime_filesystem"]


@pytest.mark.parametrize("invalid", ["missing", "null", "extra", "noncanonical"])
def test_v3_persisted_filesystem_shape_cannot_fall_back_to_v2(
    env_host: EnvHost, invalid: str
) -> None:
    env_host.override()
    record = env_host.record()
    if invalid == "missing":
        record.pop("runtime_filesystem")
    elif invalid == "null":
        record["runtime_filesystem"] = None
    elif invalid == "extra":
        record["runtime_filesystem"]["source"] = "/dev/unbound"
    else:
        record["runtime_filesystem"]["uuid"] = record["runtime_filesystem"][
            "uuid"
        ].upper()
    path = probe.collector_override_record(env_host.paths()[0])
    probe.power_write_json(path, record)
    before = path.read_bytes()
    with pytest.raises(probe.CollectorEnvGuardError):
        probe.collector_load_record(env_host.run_id, OWNER_NONCE)
    assert path.read_bytes() == before


def test_tool_replacement_between_admission_and_open_is_rejected(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_open = os.open
    executed = False

    def open_file(path: Any, *args: Any, **kwargs: Any) -> int:
        nonlocal executed
        if Path(path) == probe.COLLECTOR_FINDMNT and not executed:
            executed = True
            path.rename(path.with_name("previous-tool"))
            path.write_text("other inert executable")
            path.chmod(0o700)
        return original_open(path, *args, **kwargs)

    # The fake replaces the global os.open, which pytest's own tmp_path teardown
    # also uses; keep it to the call under test.
    with monkeypatch.context() as patch:
        patch.setattr(os, "open", open_file)
        with pytest.raises(probe.CollectorEnvGuardError, match="executable changed"):
            env_host.override()
    assert executed is True
    assert env_host.findmnt_calls == []
    assert env_host.mutations() == []


def test_missing_kernel_mount_table_is_a_hard_guard_failure(
    env_host: EnvHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_host.override()
    original = env_host.record()
    (probe.PROC_ROOT / "self/mountinfo").unlink()
    assert restore_main(env_host, monkeypatch) == 78
    assert env_host.emitted[-1]["failure_receipt"]["status"] == "RECORDED"
    assert env_host.emitted[-1]["retryable"] is False
    assert env_host.record() == original
    assert env_host.paths()[0].exists(), (
        "test_missing_kernel_mount_table_is_a_hard_guard_failure: expected env_host.paths()[0].exists()"
    )
