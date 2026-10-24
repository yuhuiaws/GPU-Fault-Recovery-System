"""Contract tests for the reusable kernel-log source replay helper.

Nothing here runs systemd, journalctl, kubectl or the kernel collector: the
reviewed-record validators and the ``KernelLogReplay`` controller are driven
over a ``FakeProbe``, and the journalctl pass-through wrapper is rendered and
run as a standalone script against a stub ``journalctl`` to prove it forwards
real lines, appends the reviewed records only when armed, and never leaks its
own synthetic cursor back to the real binary.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.e2e.regional import general_live_kernel_replay as helper
from scripts.e2e.regional.probes import general_live_kernel_source_probe as probe
from tests.regional._general_live_fakes import FakeProbe


# --------------------------------------------------------------------------- #
# reviewed records
# --------------------------------------------------------------------------- #
def test_reviewed_record_defaults_and_digest() -> None:
    record = helper.reviewed_record("NVRM: Xid (PCI:0000:53:00): 79, fell off the bus")
    assert record["transport"] == "kernel", (
        "a reviewed record defaults to the trusted kernel transport"
    )
    assert record["emissions"] == 4, record
    first = helper.records_sha256([record])
    second = helper.records_sha256([dict(record)])
    assert first == second, "the record-set digest is deterministic"
    assert len(first) == 64, "records_sha256 is a sha256 hex digest"


def test_validate_records_accepts_reviewed_set() -> None:
    records = [
        helper.reviewed_record("machine check: uncorrected error", priority=2),
        helper.reviewed_record("user wrote this", transport="syslog"),
    ]
    checked = probe.validate_records(records)
    assert [item["transport"] for item in checked] == ["kernel", "syslog"], checked
    assert checked[0]["priority"] == 2, checked


@pytest.mark.parametrize(
    "records, match",
    [
        ([], "non-empty list"),
        ([{"message": ""}], "non-empty message"),
        ([{"message": "x", "transport": "forged-kernel"}], "not reviewed"),
        ([{"message": "x", "emissions": 99}], "emissions"),
        ([{"message": "x", "priority": 9}], "priority"),
    ],
)
def test_validate_records_rejects_bad_input(records: object, match: str) -> None:
    with pytest.raises(probe.ProbeError, match=match):
        probe.validate_records(records)


def test_validate_records_bounds_count_and_size() -> None:
    too_many = [{"message": "x"} for _ in range(probe.MAX_RECORDS + 1)]
    with pytest.raises(probe.ProbeError, match="at most"):
        probe.validate_records(too_many)
    huge = [{"message": "A" * (probe.MAX_MESSAGE_BYTES + 1)}]
    with pytest.raises(probe.ProbeError, match="too large"):
        probe.validate_records(huge)


def test_probe_digest_is_stable_and_rejects_unsafe_ids() -> None:
    first = probe.digest("gl016-uncorrectable-mce-a1-deadbeef")
    assert first == probe.digest("gl016-uncorrectable-mce-a1-deadbeef"), (
        "digest is stable"
    )
    assert len(first) == 16, "the window digest is 16 hex chars"
    with pytest.raises(probe.ProbeError, match="unsafe run ID"):
        probe.safe_id("bad id; rm -rf /")


# --------------------------------------------------------------------------- #
# the KernelLogReplay controller over a FakeProbe
# --------------------------------------------------------------------------- #
def _records() -> list[dict[str, object]]:
    return [helper.reviewed_record("NVRM: Xid 79", emissions=3)]


def _controller(
    tmp_path: Path, script: dict[str, object], mode: str = "journal"
) -> tuple[helper.KernelLogReplay, FakeProbe]:
    fake = FakeProbe(script)

    class _Regional:
        class settings:
            gpu_kubeconfig = tmp_path / "kubeconfig"
            gpu_context = "ctx"
            namespace = "gpu-fault-system"

    controller = helper.KernelLogReplay(
        _Regional(),
        node="ip-10-0-43-5",
        image="img@sha256:" + "0" * 64,
        case_id="GF-GENERAL-LIVE-016.rdma_fatal",
        run_id="gl016-rdma-fatal-a1-deadbeef",
        case_dir=tmp_path / "case",
        mode=mode,
        host=fake,
    )
    return controller, fake


def test_controller_install_sends_sha256_and_base64(tmp_path: Path) -> None:
    records = _records()
    controller, fake = _controller(tmp_path, {"install-records": {"records": 1}})
    controller.install(records)
    call = next(item for item in fake.calls if item[0] == "install-records")
    flags = dict(zip(call[1::2], call[2::2]))
    assert flags["--sha256"] == helper.records_sha256(records), (
        "install carries the record-set digest"
    )
    decoded = base64.b64decode(flags["--b64"])
    assert json.loads(decoded) == records, (
        "install carries the reviewed records verbatim"
    )


def test_controller_open_arm_readback_and_cleanup(tmp_path: Path) -> None:
    script = {
        "install-records": {"records": 1},
        "open-window": {
            "after": {"ActiveState": "active"},
            "proof_scope": "live-replay",
        },
        "arm": {"armed": True},
        "status": {"injected": {"emission_counts": {"0": 2}}},
        "close-window": {"clean": True},
    }
    controller, fake = _controller(tmp_path, script)
    controller.install(_records())
    controller.open(approved=True)
    opened = next(item for item in fake.calls if item[0] == "open-window")
    assert "--approved" in opened, (
        "journal mode passes the approval through to the probe"
    )
    assert ("--mode", "journal") == (
        opened[opened.index("--mode")],
        opened[opened.index("--mode") + 1],
    ), opened
    controller.arm()
    ingested = controller.read_back()
    assert ingested["injected"]["emission_counts"] == {"0": 2}, (
        "read_back returns the probe's injection oracle"
    )
    receipts = controller.cleanup()
    assert receipts["close"]["clean"] is True, receipts
    assert not any(receipts["probe_residuals"].values()), (
        "a clean probe leaves no residue"
    )
    assert helper.PROOF_SCOPE == "live-replay", (
        "a replay only ever claims the live-replay scope"
    )


def test_controller_cleanup_raises_when_window_will_not_close(tmp_path: Path) -> None:
    controller, _ = _controller(
        tmp_path,
        {
            "open-window": {"after": {"ActiveState": "active"}},
            "close-window": {"clean": False, "residue": {"dropin": True}},
        },
    )
    controller.open(approved=True)
    with pytest.raises(RuntimeError, match="did not close clean"):
        controller.cleanup()


def test_controller_evidence_reports_scope_and_digest(tmp_path: Path) -> None:
    controller, _ = _controller(tmp_path, {"install-records": {}})
    controller.install(_records())
    evidence = controller.evidence()
    assert evidence["proof_scope"] == "live-replay", evidence
    assert evidence["records_sha256"] == helper.records_sha256(_records()), evidence
    assert evidence["record_count"] == 1, evidence


# --------------------------------------------------------------------------- #
# the rendered journalctl pass-through wrapper (journal mode)
# --------------------------------------------------------------------------- #
def _render_wrapper(tmp_path: Path, *, armed: bool) -> tuple[Path, Path]:
    """Render the wrapper against a stub journalctl; return (wrapper, argv-log)."""

    argv_log = tmp_path / "real-argv.json"
    real = tmp_path / "journalctl-real"
    real.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"open({str(argv_log)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
        "print(json.dumps({'__CURSOR': 'real-1', '__REALTIME_TIMESTAMP': '1',"
        " 'MESSAGE': 'real kernel line', '_TRANSPORT': 'kernel'}))\n",
        encoding="utf-8",
    )
    real.chmod(0o755)
    records = tmp_path / "records.json"
    records.write_text(
        json.dumps(
            [
                helper.reviewed_record(
                    "machine check: uncorrected", priority=2, emissions=2
                )
            ]
        ),
        encoding="utf-8",
    )
    state = tmp_path / "emissions.jsonl"
    armed_path = tmp_path / "armed"
    if armed:
        armed_path.write_text("now", encoding="utf-8")
    wrapper = tmp_path / "journalctl"
    wrapper.write_text(
        probe.WRAPPER_SCRIPT
        % {
            "real": str(real),
            "records": str(records),
            "armed": str(armed_path),
            "state": str(state),
            "emissions": 2,
            "boot_id": "boot-xyz",
        },
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper, argv_log


def _run_wrapper(wrapper: Path, *args: str) -> list[dict[str, object]]:
    completed = subprocess.run(
        [sys.executable, "-I", str(wrapper), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]


def test_wrapper_passes_through_when_not_armed(tmp_path: Path) -> None:
    wrapper, _ = _render_wrapper(tmp_path, armed=False)
    lines = _run_wrapper(wrapper, "--output=json", "--since", "@1", "--until", "@2")
    assert len(lines) == 1, "an unarmed wrapper only forwards the real journal"
    assert lines[0]["MESSAGE"] == "real kernel line", lines


def test_wrapper_appends_reviewed_records_when_armed(tmp_path: Path) -> None:
    wrapper, _ = _render_wrapper(tmp_path, armed=True)
    lines = _run_wrapper(wrapper, "--output=json", "--since", "@1", "--until", "@2")
    messages = [item["MESSAGE"] for item in lines]
    assert "real kernel line" in messages, "the real line is still forwarded first"
    assert "machine check: uncorrected" in messages, (
        "the reviewed record is appended when armed"
    )
    reviewed = next(
        item for item in lines if item["MESSAGE"].startswith("machine check")
    )
    assert reviewed["_TRANSPORT"] == "kernel", (
        "a kernel-transport record keeps its trusted origin"
    )
    assert reviewed["_BOOT_ID"] == "boot-xyz", reviewed
    assert reviewed["__CURSOR"].startswith("gfrs-replay-"), (
        "injected records carry a replay cursor"
    )


def test_wrapper_bounds_emissions(tmp_path: Path) -> None:
    wrapper, _ = _render_wrapper(tmp_path, armed=True)
    seen = 0
    for _poll in range(5):
        lines = _run_wrapper(wrapper, "--output=json", "--since", "@1", "--until", "@2")
        seen += sum(1 for item in lines if item["MESSAGE"].startswith("machine check"))
    assert seen == 2, "the reviewed record is emitted only up to its emissions budget"


def test_wrapper_never_forwards_its_own_cursor(tmp_path: Path) -> None:
    wrapper, argv_log = _render_wrapper(tmp_path, armed=True)
    _run_wrapper(
        wrapper,
        "--output=json",
        "--since",
        "@1",
        "--until",
        "@2",
        "--after-cursor=gfrs-replay-0-boot-xyz",
    )
    forwarded = json.loads(argv_log.read_text(encoding="utf-8"))
    assert not any("gfrs-replay-" in item for item in forwarded), (
        "the wrapper strips its own synthetic cursor before the real journalctl"
    )
    assert "--output=json" in forwarded, "every real flag is still forwarded"


def test_wrapper_passes_a_real_cursor_through(tmp_path: Path) -> None:
    wrapper, argv_log = _render_wrapper(tmp_path, armed=True)
    _run_wrapper(wrapper, "--output=json", "--after-cursor", "s=real-journal-cursor")
    forwarded = json.loads(argv_log.read_text(encoding="utf-8"))
    assert "s=real-journal-cursor" in forwarded, (
        "a real journal cursor is forwarded untouched"
    )
