"""Contract tests for GF-REGIONAL-COLLECT-018.

Every verdict is judged against synthetic evidence -- collector status rows,
``/metrics`` text, control-plane log lines and the node-activity document --
once on the intended run and once per way the run can be wrong. The runner
itself is checked to be plan-only by default. Nothing here touches a cluster.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import collect018_verdicts as verdicts
from scripts.e2e.regional import run_collect018_rejected_event as collect018
from scripts.e2e.regional.collector_window_fixture import add_window_arguments

ROOT = Path(__file__).resolve().parents[2]
CLUSTER = "cluster-a"
NODE = "node-a"


def _text(errors: list[str]) -> str:
    return "\n".join(errors)


def _status(**overrides: Any) -> dict[str, Any]:
    status = {
        "collector": "NVIDIA_KERNEL",
        "last_success_at": "2030-01-01T00:00:00+00:00",
        "last_error_at": "2030-01-01T00:05:00+00:00",
        "errors": ["rejected-event: HTTP 422 acceptance_unknown_field extra forbidden"],
    }
    status.update(overrides)
    return status


def _metrics(
    rejections: int, kernel_4xx: int, *, silent: int = 0, erroring: int = 1
) -> str:
    return "\n".join(
        [
            f"gpu_fault_processor_fault_rejections_total {rejections}",
            "gpu_fault_processor_completions_by_path_status_total"
            f'{{path="{verdicts.KERNEL_PATH}",status_class="4xx"}} {kernel_4xx}',
            "gpu_fault_processor_completions_by_path_status_total"
            f'{{path="{verdicts.KERNEL_PATH}",status_class="2xx"}} 40',
            f'gpu_fault_collector_silent_nodes{{cluster_id="{CLUSTER}",channel="NVIDIA_KERNEL"}} {silent}',
            f'gpu_fault_collector_erroring_nodes{{cluster_id="{CLUSTER}",channel="NVIDIA_KERNEL"}} {erroring}',
            'gpu_fault_ingest_unresolved_fault_signals_total{kind="unparsed_xid_line"} 1',
        ]
    )


def test_the_rejected_status_contract_passes_on_a_rejected_event_row() -> None:
    assert verdicts.rejected_status_errors([_status()]) == [], (
        "a rejected-event status with last_error_at passes"
    )


def test_the_rejected_status_contract_rejects_silence_and_payload_echo() -> None:
    assert "no NVIDIA_KERNEL" in _text(
        verdicts.rejected_status_errors([{"collector": "HOST_TELEMETRY"}])
    )
    assert "no 'rejected-event: HTTP 422'" in _text(
        verdicts.rejected_status_errors([_status(errors=["collection-error"])])
    )
    assert "last_error_at" in _text(
        verdicts.rejected_status_errors([_status(last_error_at=None)])
    )
    assert "echoes the payload" in _text(
        verdicts.rejected_status_errors(
            [_status(errors=["rejected-event: HTTP 422 input acceptance marker"])]
        )
    )


def test_the_metric_contract_needs_both_g1_counters_to_move() -> None:
    before = [_metrics(3, 1)]
    assert verdicts.rejection_metric_errors(before, [_metrics(4, 2)]) == [], (
        "both counters moved"
    )
    errors = verdicts.rejection_metric_errors(before, [_metrics(3, 2)])
    assert "fault_rejections_total did not increase" in _text(errors), errors
    errors = verdicts.rejection_metric_errors(before, [_metrics(4, 1)])
    assert "status_class=4xx" in _text(errors), errors


def test_the_log_contract_wants_the_request_id_but_never_the_payload() -> None:
    good = (
        "WARNING processor replay rejected a fault-layer event; the collector saw "
        "a 202 request_id=req-1 path=/v1/collector-events/nvidia-kernel status=422 "
        "detail=acceptance_unknown_field extra forbidden"
    )
    assert verdicts.rejection_log_errors(good, "acceptance-rejected-m1") == []
    assert "no 'processor replay rejected" in _text(
        verdicts.rejection_log_errors("nothing here", "acceptance-rejected-m1")
    )
    assert "echoes the rejected payload" in _text(
        verdicts.rejection_log_errors(
            good + " acceptance-rejected-m1", "acceptance-rejected-m1"
        )
    )
    assert "request_id" in _text(
        verdicts.rejection_log_errors(
            "processor replay rejected a fault-layer event status=422", "x"
        )
    )


def test_the_silence_contract_separates_erroring_from_silent() -> None:
    before = [_metrics(3, 1, silent=0, erroring=0)]
    assert (
        verdicts.silence_errors(
            before,
            [_metrics(4, 2, silent=0, erroring=1)],
            cluster_id=CLUSTER,
            node=NODE,
        )
        == []
    ), "erroring rose, silent did not"
    errors = verdicts.silence_errors(
        before, [_metrics(4, 2, silent=1, erroring=1)], cluster_id=CLUSTER, node=NODE
    )
    assert "rose from 0.0 to 1.0" in _text(errors), errors
    errors = verdicts.silence_errors(
        before, [_metrics(4, 2, silent=0, erroring=0)], cluster_id=CLUSTER, node=NODE
    )
    assert "not at least 1" in _text(errors), errors
    named = _metrics(4, 2) + (
        f'\ngpu_fault_collector_silent_top_node{{cluster_id="{CLUSTER}",channel="NVIDIA_KERNEL",node_id="{NODE}"}} 1'
    )
    assert "names node-a as silent" in _text(
        verdicts.silence_errors(before, [named], cluster_id=CLUSTER, node=NODE)
    )


def _activity(**overrides: Any) -> dict[str, Any]:
    activity = {
        "incidents": [
            {
                "incident_id": "inc-1",
                "reasons": ["unparsed_xid_line: an Xid token is present but no code"],
                "state": "ACTION_PENDING",
            }
        ],
        "workflows": [
            {
                "request_id": "wf-1",
                "incident_id": "inc-1",
                "status": "SUCCEEDED",
                "official_steps": [{"operation": "FREEZE_EVIDENCE"}],
            }
        ],
        "notifications": [{"incident_id": "inc-1", "category": "OPERATOR_REVIEW"}],
        "evidence": [{"record_id": "kmsg-1", "payload": {"message": "marker=m1"}}],
    }
    activity.update(overrides)
    return activity


def test_the_unparsed_finding_contract_passes_on_a_freeze_only_workflow() -> None:
    before = [
        'gpu_fault_ingest_unresolved_fault_signals_total{kind="unparsed_xid_line"} 1'
    ]
    after = [
        'gpu_fault_ingest_unresolved_fault_signals_total{kind="unparsed_xid_line"} 2'
    ]
    assert (
        verdicts.unparsed_finding_errors(
            _activity(), marker="m1", metrics_before=before, metrics_after=after
        )
        == []
    ), "a freeze-only finding with notification and evidence passes"


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"incidents": []}, "no incident names unparsed_xid_line"),
        (
            {
                "workflows": [
                    {
                        "incident_id": "inc-1",
                        "status": "SUCCEEDED",
                        "official_steps": [
                            {"operation": "FREEZE_EVIDENCE"},
                            {"operation": "MARK_UNSCHEDULABLE"},
                        ],
                    }
                ]
            },
            "compiles ['MARK_UNSCHEDULABLE']",
        ),
        ({"notifications": []}, "no OPERATOR_REVIEW notification"),
        ({"evidence": []}, "not kept as raw evidence"),
    ],
)
def test_each_unparsed_finding_deviation_fails(
    overrides: dict[str, Any], fragment: str
) -> None:
    metrics = [
        'gpu_fault_ingest_unresolved_fault_signals_total{kind="unparsed_xid_line"} 2'
    ]
    errors = verdicts.unparsed_finding_errors(
        _activity(**overrides),
        marker="m1",
        metrics_before=[metrics[0].replace(" 2", " 1")],
        metrics_after=metrics,
    )
    assert fragment in _text(errors), errors


def test_the_unresolved_counter_must_move() -> None:
    metrics = [
        'gpu_fault_ingest_unresolved_fault_signals_total{kind="unparsed_xid_line"} 2'
    ]
    errors = verdicts.unparsed_finding_errors(
        _activity(), marker="m1", metrics_before=metrics, metrics_after=metrics
    )
    assert "did not increase" in _text(errors), errors


def test_the_recovery_contract_needs_a_success_after_the_error() -> None:
    assert (
        verdicts.recovery_errors(
            [
                _status(
                    last_success_at="2030-01-01T00:10:00+00:00",
                    last_error_at="2030-01-01T00:05:00+00:00",
                )
            ]
        )
        == []
    )
    assert "still erroring" in _text(verdicts.recovery_errors([_status()]))
    assert "never recorded a success" in _text(
        verdicts.recovery_errors([_status(last_success_at=None)])
    )
    assert (
        verdicts.node_untouched_errors(
            {"ownership_annotations": {}, "unschedulable": False}
        )
        == []
    )
    assert "workflow ownership" in _text(
        verdicts.node_untouched_errors({"ownership_annotations": {"a": "b"}})
    )


def test_the_runner_is_plan_by_default_with_the_documented_flags(
    tmp_path: Path,
) -> None:
    parser = argparse.ArgumentParser()
    add_window_arguments(parser, verdicts.CONFIRMATION)
    arguments = parser.parse_args(["--run-dir", str(tmp_path), "--node", NODE])
    assert arguments.execute is False, "plan by default"
    assert arguments.confirm == "", "no confirmation by default"
    help_text = parser.format_help()
    for flag in (
        "--plan",
        "--execute",
        "--confirm",
        "--maintenance-window-end",
        "--node",
    ):
        assert flag in help_text, flag
    with pytest.raises(SystemExit):
        parser.parse_args(["--run-dir", str(tmp_path), "--plan", "--execute"])
    assert collect018.CASE_ID == "GF-REGIONAL-COLLECT-018", collect018.CASE_ID
    assert collect018.CONFIRMATION == "COLLECT018_EXECUTE", collect018.CONFIRMATION
    assert verdicts.PREDECESSOR_CASE_ID == "GF-REGIONAL-COLLECT-017"
    details = collect018.plan_details(
        type(
            "S",
            (),
            {"node": NODE, "regional": type("R", (), {"cluster_id": CLUSTER})()},
        )(),
        {"predecessor": {"valid": True}},
    )
    assert details["risk"] == "live-kernel-log-injection", details
    assert "synthetic API injection" in details["mutation"], details["mutation"]
    assert "user-space" in details["mutation"], details["mutation"]
    for path in (
        ROOT / "scripts/e2e/regional/run_collect018_rejected_event.py",
        ROOT / "scripts/e2e/regional/probes/collector_window_probe.py",
    ):
        assert path.stat().st_mode & 0o777 == 0o775, path
