"""Contract tests for the shared collector-window host probe.

The probe is what COLLECT-018/019/020 and NET-008 may change on a node, so
what it refuses is the safety contract: only allow-listed collector units,
only a probe-generated wrong token (never a caller value), only one unsettable
variable, a bounded nvidia-smi shadow, and outbox seeds/purges only while the
owning unit is stopped. Nothing here runs systemd or touches a node.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import collector_window_probe as probe


def test_the_unit_allow_list_excludes_the_node_agent() -> None:
    assert "gpu-fault-node-agent.service" not in probe.ALLOWED_UNITS, (
        probe.ALLOWED_UNITS
    )
    assert set(probe.ALLOWED_UNITS) == {
        "gpu-fault-kernel-collector.service",
        "gpu-fault-host-collector.service",
        "gpu-fault-metrics-collector.service",
        "gpu-fault-fabric-manager-collector.service",
    }, probe.ALLOWED_UNITS
    with pytest.raises(probe.ProbeError, match="allow-list"):
        probe.checked_unit("gpu-fault-node-agent.service")


def test_only_a_generated_wrong_token_and_one_unset_variable_are_allowed() -> None:
    assert probe.OVERRIDABLE == {"GPU_FAULT_CONTROL_PLANE_TOKEN": {"@invalid"}}, (
        "a caller must never be able to supply a token value"
    )
    assert probe.UNSETTABLE == {"GPU_FAULT_EXPECTED_GPU_COUNT"}, probe.UNSETTABLE


def test_shadow_modes_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        probe,
        "gpu_inventory",
        lambda: [{"uuid": "GPU-aaaa-bbbb"}, {"uuid": "GPU-cccc"}],
    )
    assert probe.parse_shadow_mode("hang:30") == ("hang", "30", "")
    assert probe.parse_shadow_mode("drop-uuid:GPU-aaaa-bbbb:1") == (
        "drop-uuid",
        "GPU-aaaa-bbbb",
        "1",
    )
    with pytest.raises(probe.ProbeError, match="1..120"):
        probe.parse_shadow_mode("hang:0")
    with pytest.raises(probe.ProbeError, match="at most 1 inventory query"):
        probe.parse_shadow_mode("drop-uuid:GPU-aaaa-bbbb:2")
    with pytest.raises(probe.ProbeError, match="does not have"):
        probe.parse_shadow_mode("drop-uuid:GPU-ffff:1")
    with pytest.raises(probe.ProbeError, match="shadow mode must be"):
        probe.parse_shadow_mode("rm -rf /")
    assert probe.MAX_DROP_CALLS == 1, (
        "one hidden query stays below the mismatch threshold"
    )


def test_window_paths_are_digests_of_the_run_id_under_run() -> None:
    first = probe.window_paths("c019-a1-1")
    second = probe.window_paths("c019-a1-1")
    other = probe.window_paths("c019-a2-1")
    assert first == second, "paths are deterministic per run id"
    assert first["root"] != other["root"], "different runs never share a window"
    assert str(first["root"]).startswith("/run/gpu-fault-acceptance/"), first["root"]
    assert probe.dropin_name("c019-a1-1").endswith(".conf"), "systemd drop-in name"
    assert probe.deadman_unit("c019-a1-1").startswith("gpu-fault-collector-window-"), (
        "the deadman unit is namespaced to the probe"
    )
    with pytest.raises(probe.ProbeError, match="unsafe run ID"):
        probe.window_paths("bad id; rm")


def test_restore_seconds_and_kmsg_lines_are_bounded() -> None:
    assert probe.checked_restore_seconds(600) == 600
    for value in (10, 3601):
        with pytest.raises(probe.ProbeError, match="restore seconds"):
            probe.checked_restore_seconds(value)
    line = probe.kmsg_line("xid63", marker="m1", bdf="0000:59:00")
    assert "user-space injection" in line and "Xid (PCI:0000:59:00): 63," in line, line
    unparsed = probe.kmsg_line("unparsed-xid", marker="m1", bdf="0000:59:00")
    assert "Xid (PCI:0000:59:00): ," in unparsed, unparsed
    assert not any(
        char.isdigit() for char in unparsed.split("): ,", 1)[1].split("acceptance")[0]
    ), "the unparsed line must carry no code"
    with pytest.raises(probe.ProbeError, match="not allow-listed"):
        probe.kmsg_line("xid79", marker="m1", bdf="0000:59:00")
    assert probe.checked_bdf("00000000:59:00.0") == "0000:59:00"
    with pytest.raises(probe.ProbeError, match="unsafe PCI BDF"):
        probe.checked_bdf("0000:59:00; rm")


def test_outbox_seed_and_purge_refuse_a_running_unit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(probe, "unit_state", lambda unit: {"ActiveState": "active"})
    monkeypatch.setattr(probe, "parse_env", lambda: {"GPU_FAULT_CLUSTER_ID": "c"})
    monkeypatch.setattr(probe, "OUTBOX_DIRECTORY", tmp_path)
    with pytest.raises(probe.ProbeError, match="while its collector is active"):
        probe.seed_outbox_record(argparse.Namespace(collector="kernel", marker="m1"))
    with pytest.raises(probe.ProbeError, match="while its collector is active"):
        probe.purge_outbox_record(argparse.Namespace(collector="kernel", marker="m1"))


def test_outbox_purge_removes_only_the_seeded_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(probe, "unit_state", lambda unit: {"ActiveState": "inactive"})
    monkeypatch.setattr(probe, "OUTBOX_DIRECTORY", tmp_path)
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(probe, "emit", emitted.append)
    real = '{"path": "/v1/collector-events/nvidia-kernel", "payload": {"record_id": "r1"}, "replayable": false}'
    seeded = (
        '{"path": "%s", "payload": {"record_id": "acceptance-dead-letter-m1"}, "replayable": false}'
        % probe.RETIRED_CHANNEL_PATH
    )
    (tmp_path / "kernel.ndjson").write_text(
        real + "\n" + seeded + "\n", encoding="utf-8"
    )
    probe.purge_outbox_record(argparse.Namespace(collector="kernel", marker="m1"))
    assert emitted[-1]["removed"] == 1 and emitted[-1]["remaining"] == 1, emitted
    assert (tmp_path / "kernel.ndjson").read_text(encoding="utf-8") == real + "\n", (
        "a real dead letter on the node is never touched"
    )


def test_outbox_records_report_metadata_and_marker_presence_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(probe, "OUTBOX_DIRECTORY", tmp_path)
    (tmp_path / "kernel.ndjson").write_text(
        '{"path": "/p", "payload": {"message": "marker=m1"}, "replayable": true, "error": "HTTP 403", "failed_at": "t"}\n'
        "not json\n",
        encoding="utf-8",
    )
    records = probe.outbox_records("kernel", "m1")
    assert records[0]["marker_present"] is True and records[0]["replayable"] is True, (
        records
    )
    assert "payload" not in records[0], "payload bodies never leave the node"
    assert records[1] == {"malformed": True}, records
    assert probe.outbox_records("kernel", None)[0]["marker_present"] is False


def test_the_probe_parser_exposes_every_subcommand_the_cases_use() -> None:
    parser = probe.parser()
    for argv in (
        ["snapshot"],
        ["open-window", "--run-id", "r", "--unit", probe.ALLOWED_UNITS[0]],
        ["close-window", "--run-id", "r"],
        ["write-kmsg", "--kind", "xid63", "--marker", "m", "--pci-bdf", "0000:59:00"],
        ["post-rejected-event", "--marker", "m", "--node-id", "n"],
        ["outbox", "--collector", "kernel", "--action", "stats"],
        ["seed-outbox-record", "--collector", "kernel", "--marker", "m"],
        ["purge-outbox-record", "--collector", "kernel", "--marker", "m"],
        ["stop-unit", "--unit", probe.ALLOWED_UNITS[0], "--run-id", "r"],
        ["start-unit", "--unit", probe.ALLOWED_UNITS[0], "--run-id", "r"],
        ["block", "--tag", "t", "--ttl-seconds", "300", "--ip", "10.0.0.1"],
        ["unblock", "--tag", "t", "--ip", "10.0.0.1"],
    ):
        assert parser.parse_args(argv).command == argv[0], argv
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["open-window", "--run-id", "r", "--unit", "gpu-fault-node-agent.service"]
        )


def _fake_nvidia_smi(tmp_path: Path) -> Path:
    real = tmp_path / "fake-gpu-query"
    real.write_text(
        "#!/usr/bin/env python3\nimport sys\n"
        "print('0, GPU-aaaa, 0000:10:00.0, NVIDIA H100')\n"
        "print('1, GPU-bbbb, 0000:20:00.0, NVIDIA H100')\n",
        encoding="utf-8",
    )
    real.chmod(0o755)
    return real


def _shadow(tmp_path: Path, state: Path) -> Path:
    script = tmp_path / "shadow-gpu-query.py"
    script.write_text(
        probe.SHADOW_SCRIPT
        % {
            "mode": ("drop-uuid", "GPU-bbbb", "1"),
            "real": str(_fake_nvidia_smi(tmp_path)),
            "state": str(state),
            "inventory_query": probe.INVENTORY_QUERY,
        },
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _run_shadow(script: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), probe.INVENTORY_QUERY, "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_the_drop_uuid_shadow_drops_one_query_and_then_passes_through(
    tmp_path: Path,
) -> None:
    script = _shadow(tmp_path, tmp_path / "calls.json")

    first = _run_shadow(script)
    assert first.returncode == 0, first.stderr
    assert "GPU-bbbb" not in first.stdout, "the first inventory query hides the GPU"
    assert "GPU-aaaa" in first.stdout, first.stdout

    second = _run_shadow(script)
    assert second.returncode == 0, second.stderr
    assert "GPU-bbbb" in second.stdout, "the second query is passed through"


def test_the_drop_uuid_shadow_passes_through_when_its_counter_cannot_be_kept(
    tmp_path: Path,
) -> None:
    """ProtectSystem=strict made the /run counter unwritable and crashed the shadow.

    Every inventory query then failed for the whole window and no snapshot
    reached the control plane. An unwritable counter must never drop a GPU
    (that path walks into REBOOT_NODE) and must never fail the query.
    """

    unwritable = tmp_path / "read-only-dir"
    unwritable.mkdir()
    script = _shadow(tmp_path, unwritable)  # a directory: open() raises

    result = _run_shadow(script)
    assert result.returncode == 0, result.stderr
    assert "GPU-bbbb" in result.stdout, "an unkept counter must not drop a GPU"
    assert "call counter unavailable" in result.stderr, result.stderr


def test_the_shadow_counter_lives_in_the_units_private_tmp() -> None:
    paths = probe.window_paths("c020-a1-1")
    assert str(paths["shadow_state"]).startswith("/tmp/"), paths["shadow_state"]
    assert not str(paths["shadow_state"]).startswith(str(paths["root"])), (
        "the window root under /run is read-only for the unit"
    )
