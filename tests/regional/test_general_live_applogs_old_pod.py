"""GF-GENERAL-LIVE-017 ``old_pod`` and the CRI capability preflight: verdict tests.

The fakes, contexts and scripted ``APPLOG_PROBE`` records are
``test_general_live_applogs``'s (shared the way ``test_general_live_applogs_hung``
shares them); this module holds the variant whose first live run (HyperPod,
2026-10-26) lost the kubelet log-directory GC race between two probe calls:
the one-call ``seed-log`` after the GC tick, the rounds repeated on the probe's
``fixture_gone`` / ``directory_removed`` reason tokens, the refusal that is not
repeated, and the read-only ``preflight`` every CRI variant takes before it
creates a job -- ``NOT_RUN(CAPABILITY_ABSENT)`` naming the missing capability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_applogs as mod
from scripts.e2e.regional.general_live_fixture import VariantInconclusive, VariantNotRun
from scripts.e2e.regional.host_probe_fixture import HostProbeError
from tests.regional.test_general_live_applogs import (
    NEW_PID,
    NEW_UID,
    NODE,
    OLD_PATH,
    POD_UID,
    PREFLIGHT_OK,
    SEEDED,
    FakeApplogJob,
    _Applog,
    _attempt,
    _container,
    _cri_ctx,
    _entry,
    _logs_probe,
    _parameters,
    _records,
    _replacement,
    fake_jobs,
)


@pytest.fixture(autouse=True)
def _patch_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_jobs(monkeypatch)


def _gc_lost(reason: str = "fixture_gone") -> Any:
    """A ``seed-log`` the probe refused with ``reason`` (its token on the error)."""

    def refuse(_arguments: tuple[str, ...]) -> dict[str, Any]:
        failure = HostProbeError(
            "host probe failed (exit 1; error ProbeError 0123456789abcdef at "
            f"_append_records:700; reason {reason}; detail in probe-error.json); "
            "output withheld"
        )
        failure.reason = reason
        raise failure

    return refuse


@pytest.mark.parametrize(
    "preflight, needle",
    [
        ({**PREFLIGHT_OK, "pod_log_root": False}, "/var/log/pods does not exist"),
        (
            {**PREFLIGHT_OK, "pod_log_layout": {"with_container_log": 0}},
            "no kubelet-shaped",
        ),
        ({**PREFLIGHT_OK, "unit": {"LoadState": "not-found"}}, "not installed"),
        ({}, "does not exist on the node"),
    ],
)
def test_cri_variants_are_not_run_on_a_node_without_the_capability(
    tmp_path: Path, preflight: dict[str, Any], needle: str
) -> None:
    FakeApplogJob.replacement = _replacement()
    probe = _logs_probe(preflight=preflight)
    ctx = _cri_ctx(
        "old_pod",
        tmp_path,
        applog=_Applog([_old_pod_records(incident_attempt=None)]),
        probe=probe,
    )
    with pytest.raises(VariantNotRun) as caught:
        mod.old_pod(ctx)
    assert caught.value.reason == "CAPABILITY_ABSENT"
    assert needle in caught.value.detail, caught.value.detail
    assert [call[0] for call in probe.calls] == ["preflight"], (
        "a read-only preflight is the only probe call; the window never opened"
    )
    assert FakeApplogJob.instances == [], "no managed job was created"


# --------------------------------------------------------------------------- #
# old_pod
# --------------------------------------------------------------------------- #
def _old_pod_records(
    *, incident_attempt: str | None, **overrides: Any
) -> dict[str, Any]:
    return _records(
        attempts=[
            _attempt(
                "JOB-a001",
                containers=[_container(terminated=True, exit_code=1)],
                phase="FAILED",
                observed_at="2026-10-23T23:55:00+00:00",
            ),
            _attempt(
                "JOB-a001-r-11223344",
                containers=[_container(pod_uid=NEW_UID, host_pid=NEW_PID)],
            ),
        ],
        entries=[_entry(path=OLD_PATH, pod_uid=POD_UID)],
        incident_attempt=incident_attempt,
        **overrides,
    )


def test_old_pod_passes_when_the_line_stays_unresolved(tmp_path: Path) -> None:
    FakeApplogJob.replacement = _replacement()
    records = _old_pod_records(
        incident_attempt=None,
        ops=("FREEZE_EVIDENCE", "VALIDATE_HOST", "ESCALATE_SUPPORT"),
    )
    ctx = _cri_ctx(
        "old_pod", tmp_path, applog=_Applog([records]), probe=_logs_probe(pid=NEW_PID)
    )
    result = mod.old_pod(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    job = FakeApplogJob.instances[-1]
    assert (job.mode, job.restart_budget) == ("fail_once", 1)
    assert job.annotations == [("pod-1", NEW_PID)], "the new Pod declares its PID"
    assert result["attributed_attempt"] is None and result["old_pod_uid"] == POD_UID
    probe = ctx.probes[f"cri-pod-logs@{NODE}"]
    names = [call[0] for call in probe.calls]  # type: ignore[attr-defined]
    assert names.index("inspect") < names.index("open-window"), (
        "the new Pod's real log directory is read before the window opens"
    )
    opened = next(call for call in probe.calls if call[0] == "open-window")  # type: ignore[attr-defined]
    assert opened[opened.index("--initial-tail-bytes") + 1] == str(
        mod.OLD_POD_INITIAL_TAIL
    )
    seeded = next(call for call in probe.calls if call[0] == "seed-log")  # type: ignore[attr-defined]
    assert "--allow-create-dir" in seeded
    assert seeded[seeded.index("--pod-uid") + 1] == POD_UID
    assert seeded[seeded.index("--await-gc-seconds") + 1] == str(
        mod.OLD_POD_GC_WAIT_SECONDS
    )
    assert "create-log" not in names and "write" not in names, (
        "the old Pod's directory, file and line are one probe call"
    )
    assert result["seed_rounds"] == 1 and result["gc"]["observed"] is True
    assert any(
        "12.5 s after kubelet's log-directory GC tick" in n for n in result["notes"]
    ), result["notes"]
    assert (ctx.case_dir / "old-seed-0.json").is_file(), "the seed receipt is evidence"


def test_old_pod_is_inconclusive_when_the_new_pod_has_no_log_directory(
    tmp_path: Path,
) -> None:
    FakeApplogJob.replacement = _replacement()
    probe = _logs_probe(pid=NEW_PID, directories=[])
    ctx = _cri_ctx(
        "old_pod",
        tmp_path,
        applog=_Applog([_old_pod_records(incident_attempt=None)]),
        probe=probe,
    )
    with pytest.raises(VariantInconclusive, match="no log directory for the new Pod"):
        mod.old_pod(ctx)
    assert "open-window" not in [call[0] for call in probe.calls], (
        "the window is not opened on a node whose layout the convention cannot follow"
    )


def test_old_pod_passes_when_the_line_resolves_to_the_prior_attempt(
    tmp_path: Path,
) -> None:
    FakeApplogJob.replacement = _replacement()
    records = _old_pod_records(
        incident_attempt="JOB-a001",
        ops=("FREEZE_EVIDENCE", "VALIDATE_HOST", "ESCALATE_SUPPORT"),
    )
    ctx = _cri_ctx(
        "old_pod", tmp_path, applog=_Applog([records]), probe=_logs_probe(pid=NEW_PID)
    )
    result = mod.old_pod(ctx)
    assert result["verdict"] == "PASS", result["errors"]


def test_old_pod_fails_when_the_new_attempt_is_blamed(tmp_path: Path) -> None:
    FakeApplogJob.replacement = _replacement()
    records = _old_pod_records(
        incident_attempt="JOB-a001-r-11223344", parameters=_parameters(pid=NEW_PID)
    )
    ctx = _cri_ctx(
        "old_pod", tmp_path, applog=_Applog([records]), probe=_logs_probe(pid=NEW_PID)
    )
    result = mod.old_pod(ctx)
    assert result["verdict"] == "FAIL"
    errors = result["errors"]
    assert any("attributed to the new attempt" in error for error in errors), errors
    assert any(f"PID {NEW_PID} was targeted" in error for error in errors), errors


def test_old_pod_is_inconclusive_without_a_restart(tmp_path: Path) -> None:
    records = _old_pod_records(incident_attempt=None)
    with pytest.raises(VariantInconclusive, match="did not restart"):
        mod.old_pod(_cri_ctx("old_pod", tmp_path, applog=_Applog([records])))


def test_old_pod_retries_a_file_kubelet_removed_before_the_collector_read_it(
    tmp_path: Path,
) -> None:
    FakeApplogJob.replacement = _replacement()
    records = _old_pod_records(
        incident_attempt=None,
        ops=("FREEZE_EVIDENCE", "VALIDATE_HOST", "ESCALATE_SUPPORT"),
    )
    probe = _logs_probe(pid=NEW_PID)
    # The first round's file is gone before the collector visits it (kubelet's
    # GC): nothing is ever delivered from it. The second round's file is read.
    probe.script["seed-log"] = [
        {**SEEDED, "file": {"path": OLD_PATH + ".lost"}},
        dict(SEEDED),
    ]
    ctx = _cri_ctx("old_pod", tmp_path, applog=_Applog([records]), probe=probe)
    result = mod.old_pod(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    names = [call[0] for call in probe.calls]
    assert names.count("seed-log") == 2 and result["seed_rounds"] == 2
    lost = json.loads((ctx.case_dir / "old-lost-0.json").read_text())
    assert lost["stage"] == "read" and "status" in lost


@pytest.mark.parametrize("reason", ["fixture_gone", "directory_removed"])
def test_old_pod_repeats_a_round_the_gc_took_while_seeding(
    tmp_path: Path, reason: str
) -> None:
    # Live 2026-10-26 (HyperPod hpb): kubelet's GC removed the recreated
    # directory between the probe's create-log and write and the variant ended
    # INCONCLUSIVE "output withheld". The probe now names the reason and the
    # runner repeats the round.
    FakeApplogJob.replacement = _replacement()
    records = _old_pod_records(
        incident_attempt=None,
        ops=("FREEZE_EVIDENCE", "VALIDATE_HOST", "ESCALATE_SUPPORT"),
    )
    probe = _logs_probe(pid=NEW_PID)
    outcomes = iter([_gc_lost(reason), lambda _arguments: dict(SEEDED)])
    probe.script["seed-log"] = lambda arguments: next(outcomes)(arguments)
    ctx = _cri_ctx("old_pod", tmp_path, applog=_Applog([records]), probe=probe)
    result = mod.old_pod(ctx)
    assert result["verdict"] == "PASS", result["errors"]
    names = [call[0] for call in probe.calls]
    assert names.count("seed-log") == 2 and result["seed_rounds"] == 2
    lost = json.loads((ctx.case_dir / "old-lost-0.json").read_text())
    assert lost == {
        "stage": "seed",
        "reason": reason,
        "error": (
            "host probe failed (exit 1; error ProbeError 0123456789abcdef at "
            f"_append_records:700; reason {reason}; detail in probe-error.json); "
            "output withheld"
        ),
    }


def test_old_pod_fails_on_any_other_probe_refusal(tmp_path: Path) -> None:
    FakeApplogJob.replacement = _replacement()
    probe = _logs_probe(pid=NEW_PID)
    probe.script["seed-log"] = _gc_lost("fixture_exists")
    ctx = _cri_ctx(
        "old_pod",
        tmp_path,
        applog=_Applog([_old_pod_records(incident_attempt=None)]),
        probe=probe,
    )
    with pytest.raises(HostProbeError) as caught:
        mod.old_pod(ctx)
    assert caught.value.reason == "fixture_exists"
    assert [call[0] for call in probe.calls].count("seed-log") == 1, (
        "a refusal that is not the GC is not repeated"
    )


def test_old_pod_gives_up_after_the_rounds(tmp_path: Path) -> None:
    FakeApplogJob.replacement = _replacement()
    probe = _logs_probe(pid=NEW_PID)
    probe.script["seed-log"] = _gc_lost()
    ctx = _cri_ctx(
        "old_pod",
        tmp_path,
        applog=_Applog([_old_pod_records(incident_attempt=None)]),
        probe=probe,
    )
    with pytest.raises(VariantInconclusive, match=f"in {mod.OLD_POD_ATTEMPTS} rounds"):
        mod.old_pod(ctx)
    assert [call[0] for call in probe.calls].count("seed-log") == mod.OLD_POD_ATTEMPTS
