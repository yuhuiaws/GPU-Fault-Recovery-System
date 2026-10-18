"""GF-REGIONAL-DESTR-016 node probe: the conditional pre-authorization check
end to end (acceptance and each refusal), barrier verdicts on ledger shapes the
real ledger never writes (a failed or unfinished quiesce, rows without
timestamps, malformed proofs), the holder arm's identity refusal and the
timer-side ``fire-injection`` refusals before any write."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import destr016_node_probe as probe

RUN_ID = "destr016-refusal-a1"
NODE = "node-a"
WORKFLOW = "wf-new"
T0 = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)
NOW = T0 + timedelta(seconds=120)
BOOT_A = "11111111-1111-4111-8111-111111111111"
BOOT_B = "22222222-2222-4222-8222-222222222222"
QUIESCE = probe.LEDGER_SHAPE["quiesce"]
VERIFY = probe.LEDGER_SHAPE["verify"]


def proof(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "kind": probe.CONDITIONAL_KIND,
        "conditional": True,
        "run_id": RUN_ID,
        "node_id": NODE,
        "boot_id": BOOT_A,
        "device": "/dev/nvidia0",
        "drill_id": f"{RUN_ID}-r",
        "marker": f"{RUN_ID}-r",
        "maintenance_window_end": (T0 + timedelta(hours=1)).isoformat(),
        "maintenance_window_seconds": 420,
        "authorized_at": T0.isoformat(),
        "expires_at": (T0 + timedelta(seconds=600)).isoformat(),
        "not_before_seconds": {"absorb": 90, "escalate": 240},
        "ledger": dict(probe.LEDGER_SHAPE),
    }
    value.update(overrides)
    return value


def armed_state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "run_id": RUN_ID,
        "boot_id": BOOT_A,
        "device": "/dev/nvidia0",
        "drill_id": f"{RUN_ID}-r",
        "max_hold_seconds": 900,
    }
    state.update(overrides)
    return state


def parked_state(**overrides: Any) -> dict[str, Any]:
    state = armed_state(
        hold_started_at=(T0 + timedelta(seconds=20)).isoformat(),
        pre_authorization=proof(),
        pre_authorized_at=T0.isoformat(),
        ledger_baseline_workflow_ids=["wf-old"],
        injections_fired={},
    )
    state.update(overrides)
    return state


def row(
    operation: str,
    state: str,
    *,
    index: int,
    started: int | None = 20,
    completed: int | None = 25,
    error: str | None = None,
    attempt: int = 1,
) -> dict[str, Any]:
    return {
        "command_id": f"{WORKFLOW}/{index}/{operation}/{NODE}/agent-4",
        "attempt": attempt,
        "state": state,
        "operation": operation,
        "started_at": None
        if started is None
        else (T0 + timedelta(seconds=started)).isoformat(),
        "completed_at": None
        if completed is None
        else (T0 + timedelta(seconds=completed)).isoformat(),
        "workflow_request_id": WORKFLOW,
        "incident_id": "inc-1",
        "agent_generation": 4,
        "error": error,
    }


REFUSED_VERIFY = row(
    VERIFY,
    "FAILED",
    index=3,
    started=40,
    completed=45,
    error="GPU device clients are still active: GPU-a:4242",
)


# --------------------------------------------------------------------------- #
# Host helpers
# --------------------------------------------------------------------------- #
def test_run_names_the_failed_command_and_its_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append({"command": list(command), **kwargs})
        return subprocess.CompletedProcess(command, 2, "", "denied\n")

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    with pytest.raises(
        probe.ProbeError, match=r"command failed \(2\): systemctl show x: denied"
    ):
        probe.run(["systemctl", "show", "x"])
    assert probe.run(["systemctl", "show", "x"], check=False).returncode == 2
    assert [call["check"] for call in seen] == [False, False]


def test_injection_units_exist_only_for_the_two_phases() -> None:
    assert probe.injection_unit(RUN_ID, "absorb") != probe.injection_unit(
        RUN_ID, "escalate"
    )
    with pytest.raises(probe.ProbeError, match="unknown injection phase"):
        probe.injection_unit(RUN_ID, "reboot")


# --------------------------------------------------------------------------- #
# check_pre_authorization
# --------------------------------------------------------------------------- #
def test_a_binding_pre_authorization_is_accepted_unchanged() -> None:
    value = proof()
    accepted = probe.check_pre_authorization(
        armed_state(), value, run_id=RUN_ID, boot_id=BOOT_A, now=NOW
    )
    assert accepted is value, "the accepted proof is the proof that was offered"


@pytest.mark.parametrize(
    ("state_changes", "proof_changes", "message"),
    [
        ({"run_id": "other-run"}, {}, "no armed holder is recorded"),
        ({"disarmed_at": T0.isoformat()}, {}, "already disarmed"),
        ({}, {"kind": "unconditional"}, "not a conditional barrier"),
        ({}, {"device": "/dev/nvidia1"}, "does not bind this holder"),
        ({}, {"boot_id": BOOT_B}, "belongs to another boot"),
        ({"boot_id": BOOT_B}, {}, "belongs to another boot"),
        ({}, {"node_id": "../node"}, "names no node"),
        ({}, {"node_id": ""}, "names no node"),
        ({}, {"ledger": {"quiesce": "RESET_GPU"}}, "ledger shape is not this probe's"),
        ({}, {"expires_at": "not a time"}, "no valid deadline"),
        ({}, {"expires_at": "2026-09-06T10:10:00"}, "no valid deadline"),
        ({}, {"maintenance_window_end": None}, "no valid deadline"),
        ({}, {"expires_at": T0.isoformat()}, "expired or future-dated"),
        (
            {},
            {"authorized_at": (NOW + timedelta(seconds=91)).isoformat()},
            "expired or future-dated",
        ),
        ({"max_hold_seconds": 60}, {}, "outlives the holder's bounded lifetime"),
        ({}, {"maintenance_window_seconds": 29}, "maintenance window is out of bounds"),
        ({}, {"maintenance_window_seconds": 3601}, "window is out of bounds"),
        ({}, {"maintenance_window_seconds": True}, "window is out of bounds"),
        ({}, {"not_before_seconds": [90]}, "not-before delays are invalid"),
        ({}, {"not_before_seconds": {}}, "not-before delays are invalid"),
        ({}, {"not_before_seconds": {"absorb": -1}}, "not-before delays are invalid"),
        ({}, {"not_before_seconds": {"absorb": "90"}}, "not-before delays are invalid"),
    ],
)
def test_pre_authorization_defects_are_refused(
    state_changes: dict[str, Any], proof_changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(probe.ProbeError, match=message):
        probe.check_pre_authorization(
            armed_state(**state_changes),
            proof(**proof_changes),
            run_id=RUN_ID,
            boot_id=BOOT_A,
            now=NOW,
        )


# --------------------------------------------------------------------------- #
# barrier_condition on shapes a real ledger never writes
# --------------------------------------------------------------------------- #
def _barrier(
    rows: list[dict[str, Any]],
    state: dict[str, Any] | None = None,
    *,
    phase: str = "absorb",
) -> dict[str, Any]:
    return probe.barrier_condition(
        state or parked_state(),
        rows,
        phase=phase,
        now=NOW,
        boot_id=BOOT_A,
        holder_active=True,
    )


def test_the_barrier_holds_on_synthetic_rows_shaped_like_the_ledger() -> None:
    verdict = _barrier([row(QUIESCE, "SUCCEEDED", index=2), REFUSED_VERIFY])
    assert verdict["holds"] is True, verdict
    assert verdict["condition"]["workflow_request_id"] == WORKFLOW
    assert (
        verdict["condition"]["pinned_window_expires_at"]
        == (T0 + timedelta(seconds=20 + 420)).isoformat()
    )


@pytest.mark.parametrize(
    ("proof_changes", "reason"),
    [
        ({"ledger": {"quiesce": "RESET_GPU"}}, "ledger shape is not this probe's"),
        ({"expires_at": "garbage"}, "no valid deadline"),
        ({"maintenance_window_end": "2026-09-06T11:00:00"}, "no valid deadline"),
        ({"not_before_seconds": {"absorb": "90"}}, "names no delay for absorb"),
        ({"not_before_seconds": None}, "names no delay for absorb"),
    ],
)
def test_a_malformed_pre_authorization_is_a_final_refusal(
    proof_changes: dict[str, Any], reason: str
) -> None:
    state = parked_state(pre_authorization=proof(**proof_changes))
    verdict = _barrier([row(QUIESCE, "SUCCEEDED", index=2), REFUSED_VERIFY], state)
    assert verdict["holds"] is False
    assert verdict["final"] is True, verdict
    assert reason in verdict["reason"], verdict


def test_a_failed_quiesce_is_final_and_an_unfinished_one_waits() -> None:
    failed = _barrier([row(QUIESCE, "FAILED", index=2), REFUSED_VERIFY])
    assert (failed["final"], failed["reason"]) == (True, "the quiesce did not succeed")
    pending = _barrier([row(QUIESCE, probe.IN_PROGRESS_STATE, index=2, completed=None)])
    assert (pending["holds"], pending["final"]) == (False, False)
    assert pending["reason"] == "the quiesce has not succeeded"
    absent = _barrier([REFUSED_VERIFY])
    assert absent["reason"] == "the quiesce has not succeeded"


def test_a_quiesce_row_without_timestamps_cannot_pin_the_window() -> None:
    verdict = _barrier(
        [
            row(QUIESCE, "SUCCEEDED", index=2, started=None, completed=None),
            REFUSED_VERIFY,
        ]
    )
    assert verdict["final"] is True
    assert verdict["reason"] == "the pinned maintenance window cannot be derived"


def test_the_window_is_pinned_from_the_only_timestamp_present() -> None:
    verdict = _barrier(
        [
            row(QUIESCE, "SUCCEEDED", index=2, started=None, completed=100),
            REFUSED_VERIFY,
        ]
    )
    assert verdict["holds"] is True, verdict
    assert (
        verdict["condition"]["pinned_window_expires_at"]
        == (T0 + timedelta(seconds=100 + 420)).isoformat()
    )


# --------------------------------------------------------------------------- #
# arm_holder identity
# --------------------------------------------------------------------------- #
def test_arm_holder_refuses_a_foreign_probe_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        probe,
        "run",
        lambda command, **kwargs: pytest.fail("no unit may start on a refusal"),
    )
    arguments = argparse.Namespace(
        run_id=RUN_ID,
        drill_id=RUN_ID,
        device="/dev/nvidia0",
        max_hold_seconds=600,
        after_ledger_op=QUIESCE,
        probe_script="/run/another_probe.py",
    )
    with pytest.raises(probe.ProbeError, match="probe script identity mismatch"):
        probe.arm_holder(arguments)
    arguments.after_ledger_op = "RESET_GPU"
    with pytest.raises(probe.ProbeError, match="not permitted for arming"):
        probe.arm_holder(arguments)


# --------------------------------------------------------------------------- #
# fire_injection before any write
# --------------------------------------------------------------------------- #
@pytest.fixture
def fire_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, list[list[str]]]:
    path = tmp_path / "state.json"
    calls: list[list[str]] = []
    monkeypatch.setattr(probe, "state_path", lambda run_id: path)
    monkeypatch.setattr(probe, "ledger_rows", lambda: [])
    monkeypatch.setattr(probe, "_boot_id", lambda: BOOT_A)
    monkeypatch.setattr(probe, "holder_active", lambda run_id: True)
    monkeypatch.setattr(probe, "POLL_SECONDS", 0)
    monkeypatch.setattr(
        probe,
        "run",
        lambda command, **kwargs: calls.append(list(command))
        or subprocess.CompletedProcess(command, 0, "", ""),
    )
    return path, calls


def _fire(phase: str = "absorb") -> None:
    probe.fire_injection(argparse.Namespace(run_id=RUN_ID, phase=phase))


def test_fire_injection_refuses_an_unknown_phase_before_reading_state(
    fire_host: tuple[Path, list[list[str]]],
) -> None:
    with pytest.raises(probe.ProbeError, match="unknown injection phase"):
        _fire("reboot")
    assert fire_host[1] == []


def test_fire_injection_refuses_state_of_another_run_or_without_its_phase(
    fire_host: tuple[Path, list[list[str]]],
) -> None:
    path, calls = fire_host
    probe.write_state(path, parked_state(run_id="other-run"))
    with pytest.raises(probe.ProbeError, match="not authorized for this run"):
        _fire()
    probe.write_state(path, parked_state(injections=[]))
    with pytest.raises(probe.ProbeError, match="injection phase is missing"):
        _fire()
    probe.write_state(
        path, parked_state(injections=[{"phase": "absorb"}, {"phase": "absorb"}])
    )
    with pytest.raises(probe.ProbeError, match="injection phase is missing"):
        _fire()
    probe.write_state(
        path,
        parked_state(
            injections=[{"phase": "absorb"}],
            injections_fired={"absorb": {"fire_requested_at": T0.isoformat()}},
        ),
    )
    with pytest.raises(probe.ProbeError, match="already consumed"):
        _fire()
    assert calls == [], "no write runs while the intent is refused"


def test_fire_injection_refuses_a_changed_injection_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fire_host: tuple[Path, list[list[str]]],
) -> None:
    path, calls = fire_host
    script = tmp_path / "gpu-fault-host-probe.py"
    script.write_text("print('changed')\n", encoding="utf-8")
    item = {
        "phase": "absorb",
        "script": str(script),
        "subcommand": "write-xid46",
        "marker": "m-absorb",
        "drill_id": f"{RUN_ID}-s",
        "pci_bdf": "0000:59:00",
        "maintenance_window_end": "2099-01-01T00:00:00+00:00",
    }
    probe.write_state(
        path,
        parked_state(
            injections=[item],
            injection_script_sha256=hashlib.sha256(b"what was armed").hexdigest(),
        ),
    )
    monkeypatch.setattr(
        probe,
        "barrier_condition",
        lambda state, rows, **kwargs: {
            "holds": True,
            "final": False,
            "reason": "the barrier holds",
            "condition": {"observed_at": NOW.isoformat()},
        },
    )
    with pytest.raises(probe.ProbeError, match="injection probe script changed"):
        _fire()
    assert calls == [], "a changed script never runs"
    assert "fire_requested_at" not in (
        probe.read_state(path).get("injections_fired") or {}
    ).get("absorb", {}), "no fire intent is recorded for a refused script"
