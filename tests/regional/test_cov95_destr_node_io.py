from __future__ import annotations

import argparse
import hashlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.models import WorkflowStatus
from gpu_fault.node_agent.quiesce import GpuServiceQuiesceManager
from scripts.e2e.regional import destr_barrier_authorization as authorization
from scripts.e2e.regional.probes import destructive_node_probe as probe
from tests.regional._cov95_destr_node_io import NodeIO
from tests.regional._cov95_destr_warm import NOW
from tests.regional.test_destr_barrier_authorization import (
    barrier_state,
    install_ledger,
)


def test_host_snapshot_uses_fake_inventory_services_journal_and_private_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = NodeIO(tmp_path, monkeypatch)
    h.inventory += "\nincomplete\n2, GPU-C, bad-bdf, NVIDIA A100"
    h.compute = "GPU-A, 101, trainer\nbad\nGPU-B, unknown, process"
    pod_uid = "11111111-2222-3333-4444-555555555555"
    process = h.proc / "101"
    process.mkdir()
    (process / "cgroup").write_text(
        f"0::/kubepods/pod{pod_uid}/container", encoding="utf-8"
    )
    h.service_outputs["nvidia-dcgm.service"] = (1, "unavailable")
    h.service_outputs["dcgm.service"] = (0, "LoadState=not-found")
    h.timers = (
        "tomorrow gpu-fault-quiesce-one.timer\nother.timer\ngpu-fault-quiesce-one.timer"
    )
    code, report = h.main(monkeypatch, "snapshot")
    assert code == 0 and report["boot_id"] == "boot-io", report
    assert [item["pci_bdf"] for item in report["gpu_inventory"]] == [
        "0000:0a:00",
        "0000:0b:00",
    ], report
    assert report["compute_clients"] == [
        {
            "gpu_uuid": "GPU-A",
            "pid": "101",
            "process_name": "trainer",
            "pod_uid": pod_uid,
        }
    ], report
    assert set(report["services"]) == {
        "kubelet.service",
        "nvidia-fabricmanager.service",
        "nvidia-persistenced.service",
    }, report
    assert report["gpu_fault_timers"] == ["gpu-fault-quiesce-one.timer"], report
    assert report["ledger"] == report["quiesce_states"] == [], report
    assert report["sampler"] is None, report


def test_missing_gpu_inventory_fails_snapshot_instead_of_reporting_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = NodeIO(tmp_path, monkeypatch)
    h.inventory = "unparseable"
    code, report = h.main(monkeypatch, "snapshot")
    assert code == 1 and "no GPU inventory" in report["error"], report


@pytest.mark.parametrize("pid", ["", "0", "-1", "missing", "102"])
def test_client_ownership_never_guesses_for_missing_or_invalid_pid(
    pid: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    NodeIO(tmp_path, monkeypatch)
    assert probe.client_pod_uid(pid) is None, pid


def test_client_with_ambiguous_cgroup_has_no_inferred_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = NodeIO(tmp_path, monkeypatch)
    path = h.proc / "5"
    path.mkdir()
    (path / "cgroup").write_text(
        "0::/pod11111111-2222-3333-4444-555555555555/a\n"
        "1::/pod99999999-2222-3333-4444-555555555555/a\n",
        encoding="utf-8",
    )
    assert probe.client_pod_uid("5") is None, (
        "two Pod UIDs in one cgroup must not assign client ownership"
    )


def test_quiesce_and_kernel_reports_preserve_unreadable_evidence_and_exclude_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = NodeIO(tmp_path, monkeypatch)
    h.quiesce.mkdir()
    (h.quiesce / "quiesce-invalid.json").write_text("{bad", encoding="utf-8")
    (h.quiesce / "quiesce-owned.json").write_text(
        json.dumps({"phase": "QUIESCED", "incident_id": "owned"}), encoding="utf-8"
    )
    records = probe.quiesce_states()
    assert [record["phase"] for record in records] == ["UNREADABLE", "QUIESCED"], (
        records
    )
    messages = [
        "unrelated kernel line",
        "gpu-fault injected Xid (PCI:0000:0a:00): 46, reset",
        "NVRM: resetting GPU 0000:0a:00",
        "NVRM: resetting GPU 0000:0b:00",
    ]
    h.journal = "bad-json\n" + "\n".join(
        json.dumps({"MESSAGE": line}) for line in messages
    )
    journal = probe.kernel_reset_journal(0, "0000:0a:00")
    assert journal["entry_count"] == 3 and journal["target_reset_count"] == 1, journal
    assert len(journal["sha256"]) == 64, journal


@pytest.mark.parametrize("xid", [46, 79, 63])
def test_xid_writes_only_to_fake_descriptor_and_closes_it(
    xid: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = NodeIO(tmp_path, monkeypatch)
    command = f"write-xid{xid}" if xid in {46, 79} else "write-xid"
    extra = [] if xid in {46, 79} else ["--xid", str(xid)]
    code, report = h.main(
        monkeypatch,
        command,
        *extra,
        "--marker",
        "marker-owned",
        "--drill-id",
        "drill-owned",
        "--pci-bdf",
        "0000:0a:00",
        "--maintenance-window-end",
        (NOW + timedelta(seconds=60)).isoformat(),
    )
    assert code == 0 and report["xid"] == xid, report
    assert len(h.writes) == 1 and h.closed == [99999], h.writes
    assert f"Xid (PCI:0000:0a:00): {xid},".encode() in h.writes[0], h.writes
    assert report["bytes_written"] == len(h.writes[0]), report


@pytest.mark.parametrize("deadline", ["bad", NOW.isoformat(), "2026-09-12T13:00:00"])
def test_xid_deadline_refusal_does_not_open_even_the_fake_device(
    deadline: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = NodeIO(tmp_path, monkeypatch)
    code, report = h.main(
        monkeypatch,
        "write-xid46",
        "--marker",
        "m",
        "--drill-id",
        "d",
        "--pci-bdf",
        "0000:0a:00",
        "--maintenance-window-end",
        deadline,
    )
    assert code == 1 and report["error"], report
    assert h.opened == [], h.opened


@pytest.mark.parametrize("defect", ["marker", "drill", "bdf", "xid", "write"])
def test_xid_guard_rejects_unsafe_inputs_and_closes_descriptor_on_write_error(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = NodeIO(tmp_path, monkeypatch)
    args = argparse.Namespace(
        marker="m",
        drill_id="d",
        pci_bdf="0000:0a:00",
        case_id="GF-UNIT",
        xid=46,
        description="unit fixture",
    )
    if defect == "marker":
        args.marker = "unsafe;marker"
    elif defect == "drill":
        args.drill_id = ""
    elif defect == "bdf":
        args.pci_bdf = "0000:0a:00.0"
    elif defect == "xid":
        args.xid = 999
    else:
        h.write_failure = True
    with pytest.raises((probe.ProbeError, OSError)):
        probe.write_generic_xid(args)
    assert h.writes == [], h.writes
    assert h.closed == ([99999] if defect == "write" else []), h.closed


@pytest.mark.parametrize(
    ("output", "code", "count"),
    [
        ("GPU-A\nGPU-B\n", 0, 2),
        ("GPU-A\nGPU-A\n", 0, None),
        ("GPU-A,extra\n", 0, None),
        ("N/A\n", 0, None),
        ("GPU-A\n", 1, None),
    ],
)
def test_gpu_sampler_requires_successful_unique_uuid_rows(
    output: str,
    code: int,
    count: int | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = NodeIO(tmp_path, monkeypatch)
    h.sample_output, h.sample_returncode = output, code
    sample = probe.gpu_sample()
    assert sample["gpu_count"] == count, sample
    assert sample["timed_out"] is False, sample
    assert sample["sha256"] == hashlib.sha256(output.encode()).hexdigest(), sample


def test_gpu_sampler_records_timeout_and_continues_the_bounded_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = NodeIO(tmp_path, monkeypatch)
    h.sample_timeout = True
    path = h.sampler / "reset-sampler-owned.ndjson"
    code, report = h.main(
        monkeypatch,
        "sample-gpus",
        "--output",
        str(path),
        "--duration-seconds",
        "1",
        "--interval-seconds",
        "0.5",
    )
    assert code == 0, report
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2 and all(row["timed_out"] for row in rows), rows
    assert all(row["gpu_count"] is None for row in rows), rows


@pytest.mark.parametrize("started", [False, True])
def test_detached_sampler_start_is_verified_and_stop_removes_owned_trace(
    started: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = NodeIO(tmp_path, monkeypatch)
    h.start_samples = started
    code, report = h.main(
        monkeypatch,
        "start-reset-sampler",
        "--run-id",
        "owned",
        "--probe-script",
        str(probe.__file__),
    )
    assert code == (0 if started else 1), report
    if started:
        assert report["sample_count"] == 2 and report["active"] is True, report
        code, stopped = h.main(monkeypatch, "stop-reset-sampler", "--run-id", "owned")
        assert code == 0 and stopped["active"] is False, stopped
        assert not Path(stopped["path"]).exists(), stopped
    else:
        assert "unit stopped" in report["error"] and h.active is False, report


def test_sampler_summary_ignores_non_records_and_unproven_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = NodeIO(tmp_path, monkeypatch)
    h.sampler.mkdir()
    _, path = probe.sampler_paths("owned")
    samples = [
        {"returncode": 0, "gpu_count": 2, "gpu_uuids": ["GPU-A", "GPU-B"]},
        {"returncode": 0, "gpu_count": True, "gpu_uuids": [5]},
        {"returncode": 1, "gpu_count": 0, "gpu_uuids": []},
    ]
    path.write_text(
        "bad\n[]\n" + "\n".join(json.dumps(row) for row in samples), encoding="utf-8"
    )
    result = probe.sampler_summary("owned")
    assert result["sample_count"] == 3 and result["min_gpu_count"] == 2, result
    assert result["observed_gpu_uuid_sets"] == [("GPU-A", "GPU-B")], result
    assert result["first"] == samples[0] and result["last"] == samples[-1], result
    absent = probe.sampler_summary("absent")
    assert absent["sample_count"] == 0, absent


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("--duration-seconds", "0"),
        ("--duration-seconds", "1801"),
        ("--interval-seconds", "0.01"),
        ("--interval-seconds", "6"),
    ],
)
def test_sampler_main_rejects_invalid_bounds_without_starting_a_unit(
    field: str, value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = NodeIO(tmp_path, monkeypatch)
    code, report = h.main(
        monkeypatch,
        "start-reset-sampler",
        "--run-id",
        "owned",
        "--probe-script",
        str(probe.__file__),
        field,
        value,
    )
    assert code == 1 and "outside" in report["error"], report
    assert h.calls == [], h.calls


def test_sampler_refuses_unowned_path_and_wrong_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = NodeIO(tmp_path, monkeypatch)
    code, report = h.main(
        monkeypatch,
        "start-reset-sampler",
        "--run-id",
        "owned",
        "--probe-script",
        str(tmp_path / "other.py"),
    )
    assert code == 1 and "identity mismatch" in report["error"], report
    with pytest.raises(probe.ProbeError, match="unsafe sampler output"):
        probe.sample_gpus(argparse.Namespace(output=str(tmp_path / "outside.ndjson")))
    assert h.calls == [], h.calls


def test_legacy_quiesce_probe_never_invokes_the_product_restore_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = NodeIO(tmp_path, monkeypatch)
    paths: list[Path] = []

    def restore(path: Path) -> dict[str, Any]:
        paths.append(path)
        return {"restored": True}

    monkeypatch.setattr(
        GpuServiceQuiesceManager, "restore_state_file", staticmethod(restore)
    )
    code, report = h.main(monkeypatch, "restore-quiesce", "--incident-id", "owned")
    assert code == 0 and report["restore_attempted"] is False, report
    assert report["physical_outcome_proven"] is False
    digest = hashlib.sha256(b"owned").hexdigest()[:20]
    assert not (h.quiesce / f"quiesce-{digest}.json").exists(), (
        "the read-only compatibility probe must not create quiesce state"
    )
    assert paths == [], "the acceptance probe cannot call restore_state_file"


def test_local_barrier_rechecks_real_scratch_ledger_and_returns_verified_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = NodeIO(tmp_path, monkeypatch)
    install_ledger(tmp_path, monkeypatch)
    monkeypatch.setattr(authorization, "datetime", h.clock)
    state = barrier_state()
    state["workflow"]["status"] = WorkflowStatus.RUNNING
    proof = authorization.barrier_authorization(
        state,
        run_id="review",
        node="node-a",
        boot_id="boot-a",
        device="/dev/nvidia0",
        drill_id="review-r",
        maintenance_window_end=NOW + timedelta(minutes=10),
    )
    code, report = h.main(
        monkeypatch, "check-barrier", "--barrier-authorization", json.dumps(proof)
    )
    assert code == 0 and report == {"barrier_verified": True}, report
    assert len(probe.ledger_rows()) == 2, probe.ledger_rows()
    code, report = h.main(
        monkeypatch,
        "write-xid46",
        "--marker",
        "m",
        "--drill-id",
        "d",
        "--pci-bdf",
        "0000:0a:00",
        "--barrier-authorization",
        json.dumps(proof),
    )
    assert code == 0 and len(h.writes) == 1, report
