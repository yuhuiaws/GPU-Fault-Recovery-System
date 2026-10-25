"""The runner side of the DCGM replay's start refusal and close contract.

Over the scripted DCGM probe (``_general_live_gpu_fakes.dcgm_probe``): a
probe call that raises midway is still closed by the LIFO cleanup through
``close-run``; a close that leaves objects names each one in the error that
fails the variant; the survey refuses to start while an earlier run's objects
are on the node (named, with a pending deadman left to fire), reclaims only
this fixture's own expired leftovers first, and never a foreign drop-in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import general_live_gpu_memory as mem
from scripts.e2e.regional.general_live_fixture import VariantNotRun
from scripts.e2e.regional.general_live_gpu_replay import (
    DcgmReplay,
    ReplayCleanupError,
    add,
    baseline,
    plan,
    residue_lines,
    step,
)
from scripts.e2e.regional.host_probe_fixture import HostProbeError
from tests.regional._general_live_fakes import FakeRegional, make_context, make_settings
from tests.regional._general_live_gpu_fakes import (
    audit_records,
    dcgm_probe,
    gpu_cpu_python,
    gpu_incident,
    gpu_workflow,
    survey,
)

SPEC = {spec.variant: spec for spec in mem.VARIANTS}["corrected_ecc"]
GPU = "GPU-11111111-1111-4111-8111-111111111111"
NODE = "ip-10-0-43-5.us-west-2.compute.internal"
EARLIER_RUN = "gl019-corrected-ecc-a2-4116e7cc"
DIGEST = "d96819b19bd0"
FIRES = "Sun 2026-10-25 10:23:10 UTC"
LEFTOVER = [
    "drop-in /run/systemd/system/gpu-fault-metrics-collector.service.d/"
    f"gfrs-live-{DIGEST}.conf",
    f"unit gfrs-live-{DIGEST}-proxy.service (active)",
    f"run dir /run/gfrs-live/{DIGEST} (run {EARLIER_RUN})",
    f"deadman gfrs-live-{DIGEST}-restore.timer (active; fires {FIRES})",
]
WITHHELD = (
    "host probe failed (exit 1; error ProbeError 0123456789abcdef at "
    "repoint_collector:812); output withheld"
)


def _ctx(tmp_path: Path, probe: Any, cpu: Any = None):
    regional = FakeRegional(cpu_python=cpu or gpu_cpu_python())
    settings = make_settings(SPEC, node=NODE)
    ctx = make_context(
        SPEC, tmp_path=tmp_path, regional=regional, probes={}, settings=settings
    )
    ctx.probes[f"dcgm@{NODE}"] = probe
    return ctx


def _residue(**runs: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_dirs": sorted(runs),
        "metrics_dropins": [f"gfrs-live-{d}.conf" for d in sorted(runs)],
        "units": [f"gfrs-live-{d}-proxy.service" for d in sorted(runs)],
        "runs": runs,
        "foreign_dropins": [],
        "foreign_run_dirs": [],
        "port": 9412,
        "port_answers": bool(runs),
    }


def _survey_with(residue: dict[str, Any]) -> dict[str, Any]:
    value = survey()
    value["residue"] = residue
    return value


def _passing_activity() -> dict[str, Any]:
    incident = gpu_incident(
        "inc-ecc", f"b-{GPU}-ecc_sbe_volatile_total", state="RECOVERED"
    )
    workflow = gpu_workflow(
        "wf-1",
        "inc-ecc",
        ["FREEZE_EVIDENCE", "VALIDATE_GPU"],
        status="FAILED",
        executions=[
            {
                "operation": "VALIDATE_GPU",
                "status": "FAILED",
                "details": {"host_diagnostic_failure": True},
            }
        ],
    )
    return {
        "incidents": [incident],
        "workflows": [workflow],
        "batches": [{"observed_at": "t"}],
        "markers": [],
        "commands": [],
        "notifications": [],
        "metric_latest": [],
        "collector_statuses": [],
        "agent": None,
    }


def _passing_cpu() -> Any:
    return gpu_cpu_python(
        activity=_passing_activity(),
        gpu_state={"active_findings": [], "batches": [{"observed_at": "t"}]},
    )


# --------------------------------------------------------------------------- #
# close: whatever step raised, close-run runs and names what it left
# --------------------------------------------------------------------------- #
def test_a_probe_call_that_raises_midway_is_closed_by_the_cleanup_stack(
    tmp_path: Path,
) -> None:
    def repoint(arguments: tuple[str, ...]) -> dict[str, Any]:
        raise HostProbeError(WITHHELD)

    probe = dcgm_probe(extra={"repoint-collector": repoint})
    ctx = _ctx(tmp_path, probe)
    with pytest.raises(
        HostProbeError, match="^dcgm repoint-collector: host probe failed"
    ):
        mem.corrected_ecc(ctx)
    assert ctx.cleanup.pending() == ["dcgm-replay"]
    results = ctx.cleanup.run()
    assert [(r["name"], r["ok"]) for r in results] == [("dcgm-replay", True)]
    calls = [c[0] for c in probe.calls]
    assert calls.index("close-run") > calls.index("repoint-collector")
    assert "restore-collector" not in calls and "stop-proxy" not in calls, (
        "the runner's flags no longer decide what is undone"
    )
    close_call = next(c for c in probe.calls if c[0] == "close-run")
    assert close_call[:3] == ("close-run", "--port", "9412")
    assert json.loads((ctx.case_dir / "dcgm-replay-close.json").read_text())["close"][
        "clean"
    ]


def test_close_names_every_leftover_and_failed_step(tmp_path: Path) -> None:
    probe = dcgm_probe(
        extra={
            "close-run": {
                "clean": False,
                "leftover": LEFTOVER[1:],
                "errors": [
                    "stop: ProbeError: refusing to stop the replay while the collector reads it"
                ],
                "deadman_kept": True,
            }
        }
    )
    ctx = _ctx(tmp_path, probe)
    replay = DcgmReplay(
        ctx,
        plan("memory", baseline(), step(40, add("ecc_sbe_volatile_total", [GPU]))),
        600,
    )
    replay.deploy()
    results = ctx.cleanup.run()
    assert results[0]["ok"] is False
    error = results[0]["error"]
    assert error.startswith(
        f"ReplayCleanupError: replay cleanup on {NODE} -- left on the node: "
    ), error
    for line in LEFTOVER[1:]:
        assert line in error, (line, error)
    assert "failed steps: stop: ProbeError: refusing to stop the replay" in error
    assert ctx.cleanup.pending() == []


def test_close_treats_an_unproven_clean_as_a_failure(tmp_path: Path) -> None:
    probe = dcgm_probe(extra={"close-run": {"leftover": [], "errors": []}})
    ctx = _ctx(tmp_path, probe)
    replay = DcgmReplay(ctx, plan("memory", baseline()), 600)
    replay.deploy()
    with pytest.raises(ReplayCleanupError, match="did not prove the run clean"):
        replay.close()


# --------------------------------------------------------------------------- #
# start: refuse while anything is on the node; reclaim only own expired runs
# --------------------------------------------------------------------------- #
def test_survey_refuses_to_start_while_an_earlier_runs_deadman_is_pending(
    tmp_path: Path,
) -> None:
    dirty = _survey_with(
        _residue(
            **{
                DIGEST: {
                    "digest": DIGEST,
                    "run_id": EARLIER_RUN,
                    "objects": LEFTOVER,
                    "deadman_pending": True,
                    "reclaimable": False,
                }
            }
        )
    )
    probe = dcgm_probe(dirty)
    ctx = _ctx(tmp_path, probe)
    with pytest.raises(VariantNotRun) as info:
        mem.corrected_ecc(ctx)
    assert info.value.reason == "FIXTURE_NOT_READY"
    detail = info.value.detail
    assert detail.startswith(
        "replay objects are still on the node; nothing was started: "
    ), detail
    assert (
        f"earlier run {EARLIER_RUN} (deadman restore pending, left to fire): " in detail
    )
    for line in LEFTOVER:
        assert line in detail, (line, detail)
    assert "a replay tool still answers on 127.0.0.1:9412" in detail
    assert [c[0] for c in probe.calls] == ["survey"], (
        "nothing is reclaimed or installed while a deadman is pending"
    )
    assert probe.calls[0][:3] == ("survey", "--port", "9412")


def test_survey_reclaims_an_expired_leftover_of_this_fixture_then_runs(
    tmp_path: Path,
) -> None:
    expired = _survey_with(
        _residue(
            **{
                DIGEST: {
                    "digest": DIGEST,
                    "run_id": EARLIER_RUN,
                    "objects": LEFTOVER[:3],
                    "deadman_pending": False,
                    "reclaimable": True,
                }
            }
        )
    )
    probe = dcgm_probe(
        audit=audit_records({f"{GPU}/ecc_sbe_volatile_total": 1.0}),
        extra={
            "survey": [expired, survey()],
            "cleanup-leftover": {
                "reclaimed": True,
                "clean": True,
                "leftover": [],
                "errors": [],
                "run_id": EARLIER_RUN,
            },
        },
    )
    ctx = _ctx(tmp_path, probe, cpu=_passing_cpu())
    out = mem.corrected_ecc(ctx)
    assert out["verdict"] == "PASS", out["errors"]
    assert probe.calls[0][0] == "survey"
    assert probe.calls[1][:3] == ("cleanup-leftover", "--digest", DIGEST)
    assert probe.calls[2][0] == "survey", "the node is surveyed again after the reclaim"
    assert probe.calls[3][0] == "install-source", (
        "the replay starts on the clean survey"
    )
    receipts = json.loads((ctx.case_dir / "dcgm-leftover-reclaim.json").read_text())
    assert receipts[DIGEST]["reclaimed"] is True
    written = json.loads((ctx.case_dir / "gpu-survey.json").read_text())
    assert written["residue"]["runs"] == {}, "the evidence survey is the clean one"


def test_a_reclaim_that_leaves_objects_is_named_and_nothing_starts(
    tmp_path: Path,
) -> None:
    state = {
        "digest": DIGEST,
        "run_id": EARLIER_RUN,
        "objects": LEFTOVER[1:3],
        "deadman_pending": False,
        "reclaimable": True,
    }
    expired = _survey_with(_residue(**{DIGEST: state}))
    probe = dcgm_probe(
        extra={
            "survey": [expired, expired],
            "cleanup-leftover": {
                "reclaimed": True,
                "clean": False,
                "leftover": LEFTOVER[1:3],
                "errors": [
                    "stop: ProbeError: refusing to stop the replay while the collector reads it"
                ],
            },
        }
    )
    ctx = _ctx(tmp_path, probe)
    with pytest.raises(VariantNotRun) as info:
        mem.corrected_ecc(ctx)
    detail = info.value.detail
    assert f"earlier run {EARLIER_RUN}: {LEFTOVER[1]}, {LEFTOVER[2]}" in detail
    assert (
        f"reclaim of run {DIGEST} failed: stop: ProbeError: refusing to stop the replay"
        in detail
    )
    assert [c[0] for c in probe.calls] == ["survey", "cleanup-leftover", "survey"]


def test_survey_never_reclaims_a_foreign_dropin_or_directory(tmp_path: Path) -> None:
    foreign = _survey_with(
        {
            "run_dirs": ["not-a-digest"],
            "metrics_dropins": ["gpu-fault-acceptance-0123456789abcdef.conf"],
            "units": [],
            "runs": {},
            "foreign_dropins": ["gpu-fault-acceptance-0123456789abcdef.conf"],
            "foreign_run_dirs": ["not-a-digest"],
            "port": 9412,
            "port_answers": False,
        }
    )
    probe = dcgm_probe(foreign)
    ctx = _ctx(tmp_path, probe)
    with pytest.raises(VariantNotRun) as info:
        mem.corrected_ecc(ctx)
    detail = info.value.detail
    assert (
        "foreign drop-in gpu-fault-acceptance-0123456789abcdef.conf on "
        "gpu-fault-metrics-collector.service (not this fixture's; left in place)"
    ) in detail
    assert (
        "foreign directory /run/gfrs-live/not-a-digest (not this fixture's; left in place)"
        in detail
    )
    assert [c[0] for c in probe.calls] == ["survey"], (
        "foreign objects are never reclaimed"
    )


def test_an_older_probes_flat_residue_is_still_named() -> None:
    lines = residue_lines(
        {"run_dirs": ["abc"], "metrics_dropins": ["x.conf"], "units": ["u.service"]}
    )
    assert lines == ["drop-in x.conf", "unit u.service", "run dir /run/gfrs-live/abc"]
    assert (
        residue_lines({"runs": {}, "foreign_dropins": [], "port_answers": False}) == []
    )
