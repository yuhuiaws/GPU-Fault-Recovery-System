from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpu_fault.node_agent import config


@pytest.mark.parametrize(
    "defect", ["missing", "unicode", "empty", "namespace", "container", "valid"]
)
def test_required_host_process_identity_is_never_guessed(tmp_path, monkeypatch, defect):
    root = tmp_path / "proc"
    (root / "1").mkdir(parents=True)
    if defect != "missing":
        (root / "1" / "comm").write_bytes(
            b"\xff"
            if defect == "unicode"
            else b""
            if defect == "empty"
            else b"python\n"
            if defect == "container"
            else b"systemd\n"
        )
    reads = []

    def stat(path):
        reads.append(str(path))
        if defect == "namespace":
            raise OSError("owned namespace read failure")
        return SimpleNamespace(st_dev=1, st_ino=2)

    monkeypatch.setattr(config, "os", SimpleNamespace(stat=stat))
    if defect == "valid":
        assert config.validate_host_proc_root(str(root)) == "systemd"
        assert reads == [str(root / "1" / "ns" / "pid"), "/proc/1/ns/pid"]
    else:
        with pytest.raises(ValueError, match="GPU_FAULT_PROC_ROOT"):
            config.validate_host_proc_root(str(root))


def test_production_factory_enables_guard_and_validates_host_scope(
    tmp_path, monkeypatch
):
    root = tmp_path / "proc"
    (root / "1").mkdir(parents=True)
    (root / "1" / "comm").write_text("systemd\n")
    checks = []
    monkeypatch.setattr(
        config,
        "validate_host_proc_root",
        lambda value: checks.append(value) or "systemd",
    )
    monkeypatch.setenv("GPU_FAULT_PROC_ROOT", str(root))
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_SECRET", "local-test-only-" + "x" * 32)
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_DB", str(tmp_path / "owned-ledger.db"))
    monkeypatch.setenv("GPU_FAULT_NODE_ALLOWED_OPERATIONS", "VERIFY_NO_GPU_CLIENTS")
    monkeypatch.setenv("GPU_FAULT_NODE_ALLOW_SERVICE_QUIESCE", "false")
    agent = config.executor_from_environment()
    assert agent.ownership_gate is not None
    assert checks == [str(root)]


def test_config_digest_requires_profile_without_reading_or_emitting_credentials(
    monkeypatch, capsys
):
    monkeypatch.delenv("GPU_FAULT_NODE_RUNTIME_PROFILE_VERSION", raising=False)
    monkeypatch.setattr(config, "executor_from_environment", lambda **kw: object())
    with pytest.raises(SystemExit, match="PROFILE_VERSION"):
        config.print_config_digest()
    assert capsys.readouterr().out == ""
