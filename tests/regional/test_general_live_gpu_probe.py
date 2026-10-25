"""Pure-function contract tests for the DCGM field-replay host controller.

The probe runs on the node through the host-probe fixture; nothing here runs
systemd, kubectl or nvidia-smi. These cover the stdlib validators, the
override-environment builder and the per-category safety invariants that
decide what the probe may touch.
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.e2e.regional.probes import general_live_dcgm_probe as probe


def test_identifiers_and_bounds() -> None:
    assert probe.checked_run_id("gl19-a1-deadbeef") == "gl19-a1-deadbeef"
    with pytest.raises(probe.ProbeError, match="unsafe run ID"):
        probe.checked_run_id("bad id; rm -rf /")
    assert probe.digest("gl19-a1-deadbeef") == probe.digest("gl19-a1-deadbeef")
    assert len(probe.digest("gl19-a1-deadbeef")) == 12
    assert probe.checked_port(9412) == 9412
    for bad in (0, 80, 70000):
        with pytest.raises(probe.ProbeError):
            probe.checked_port(bad)


def test_only_reviewed_sources_may_be_installed() -> None:
    assert set(probe.SOURCES) == {
        "gpu_dcgm_field_replay.py",
        "gpu_environment_replay.py",
        "plan.json",
        "nvidia_smi_inventory_replay.py",
        "inventory-plan.json",
    }


def test_the_node_agent_is_never_the_repointed_unit() -> None:
    assert probe.METRICS_UNIT == "gpu-fault-metrics-collector.service"
    assert "node-agent" not in probe.METRICS_UNIT


def test_process_environment_reads_only_allow_listed_keys(
    tmp_path, monkeypatch
) -> None:
    environ = tmp_path / "environ"
    environ.write_bytes(
        b"GPU_FAULT_DCGM_METRICS_URL=http://127.0.0.1:9412/metrics\x00"
        b"AWS_SECRET_ACCESS_KEY=supersecret\x00"
        b"GPU_FAULT_METRICS_INTERVAL_SECONDS=15\x00"
        b"PATH=/run/gfrs-live/x/bin:/usr/bin\x00"
    )

    class FakePath:
        def __init__(self, pid: str) -> None:
            self.pid = pid

        def read_bytes(self) -> bytes:
            return environ.read_bytes()

    monkeypatch.setattr(probe, "Path", lambda value: FakePath(value))
    values = probe.process_environment("123")
    assert values["GPU_FAULT_DCGM_METRICS_URL"].endswith(":9412/metrics"), values
    assert values["GPU_FAULT_METRICS_INTERVAL_SECONDS"] == "15", (
        "the runner paces its oracle gate by the collector's real interval"
    )
    assert "AWS_SECRET_ACCESS_KEY" not in values, "secrets never leave the node"


def test_override_lines_only_add_allow_listed_settings(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "run")
    monkeypatch.setattr(
        probe.Path,
        "read_text",
        lambda self, *a, **k: "ffffffff-ffff-ffff-ffff-ffffffffffff\n"
        if "boot_id" in str(self)
        else "",
        raising=False,
    )
    lines = probe.override_lines("gl22-a1-deadbeef", 9412, False)
    assert lines == ["GPU_FAULT_DCGM_METRICS_URL=http://127.0.0.1:9412/metrics"]
    unknown = probe.override_lines(
        "gl26-a1-deadbeef", 9412, False, unknown_expected_count=True
    )
    assert "GPU_FAULT_EXPECTED_GPU_COUNT=" in unknown
    assert f"GPU_FAULT_NODE_INSTANCE_TYPE={probe.UNKNOWN_INSTANCE_TYPE}" in unknown


def test_override_lines_reject_the_current_boot_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "run")
    current = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

    def read_text(self: Any, *a: Any, **k: Any) -> str:
        return current + "\n"

    monkeypatch.setattr(probe.Path, "read_text", read_text, raising=False)
    with pytest.raises(probe.ProbeError, match="current boot"):
        probe.override_lines("gl22-a1-deadbeef", 9412, False, boot_id=current)
    with pytest.raises(probe.ProbeError, match="dashed boot UUID"):
        probe.override_lines("gl22-a1-deadbeef", 9412, False, boot_id="not-a-boot")


def test_only_the_replay_and_inventory_journals_are_readable(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "run")
    captured: dict[str, Any] = {}
    monkeypatch.setattr(probe, "emit", lambda value: captured.update(value))
    namespace = probe.argparse.Namespace(
        run_id="gl19-a1-deadbeef", journal="secrets", offset=0
    )
    assert probe.main.__module__ == probe.__name__
    with pytest.raises(probe.ProbeError, match="replay and inventory journals"):
        probe.audit_read(namespace)
