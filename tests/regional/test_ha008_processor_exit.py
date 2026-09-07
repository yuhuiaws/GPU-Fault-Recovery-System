"""HA-008: the release-failure evidence is the processor's own ERROR line."""

from __future__ import annotations

from scripts.e2e.regional import run_ha008_processor_exit_acceptance as ha008

INJECTED_ONLY = """INFO gpu_fault.processor.coordinator processor started
Traceback (most recent call last):
  File "probe.py", line 1, in <module>
RuntimeError: HA008 injected release failure
"""

PROCESSOR_LINES = """INFO gpu_fault.processor.coordinator processor started
ERROR gpu_fault.processor.coordinator could not release request after its execution deadline request_id=r1 path=/v1/x lane=l1
Traceback (most recent call last):
RuntimeError: HA008 injected release failure
ERROR gpu_fault.processor.coordinator processor request execution deadline exceeded request_id=r1 path=/v1/x lane=l1 owner=o epoch=1 max_execution_seconds=1 distinct_requests=1 threshold=1 process_unhealthy=True
"""


def test_release_failure_requires_the_processor_error_line_not_the_injected_text() -> (
    None
):
    assert ha008.release_failure_logged(INJECTED_ONLY) is False, (
        "the injected exception text appears in the traceback whether or not the "
        "processor logged anything; it is not evidence"
    )
    assert ha008.release_failure_logged(PROCESSOR_LINES) is True
    assert ha008.deadline_exceeded_logged(PROCESSOR_LINES) is True
    assert ha008.deadline_exceeded_logged(INJECTED_ONLY) is False


def test_release_failure_line_must_be_error_level_from_the_coordinator_logger() -> None:
    wrong_level = "WARNING gpu_fault.processor.coordinator could not release request after its execution deadline\n"
    wrong_logger = (
        "ERROR gpu_fault.other could not release request after its execution deadline\n"
    )
    assert ha008.release_failure_logged(wrong_level) is False
    assert ha008.release_failure_logged(wrong_logger) is False


def test_acceptance_evaluates_both_branches_on_the_processor_lines(
    monkeypatch, tmp_path
) -> None:
    branches = iter(
        [
            {
                "branch": "release-ok",
                "fail_release": False,
                "exit_code": 70,
                "request_id": "r",
                "first_owner": "a",
                "first_lane_epoch": 1,
                "status_after_exit": "PENDING",
                "second_owner": "b",
                "second_lane_epoch": 2,
                "stale_result_rejected": True,
                "final_status": "COMPLETED",
                "final_response_status": 200,
                "release_failure_logged": False,
                "deadline_exceeded_logged": True,
            },
            {
                "branch": "release-fail",
                "fail_release": True,
                "exit_code": 70,
                "request_id": "r",
                "first_owner": "a",
                "first_lane_epoch": 1,
                "status_after_exit": "LEASED",
                "second_owner": "b",
                "second_lane_epoch": 2,
                "stale_result_rejected": True,
                "final_status": "COMPLETED",
                "final_response_status": 200,
                "release_failure_logged": False,
                "deadline_exceeded_logged": True,
            },
        ]
    )
    monkeypatch.setattr(ha008, "_run_branch", lambda *_a, **_k: next(branches))

    report = ha008.run_acceptance(tmp_path)

    assert report["verdict"] == "FAIL"
    assert any("could not release request" in item for item in report["errors"]), (
        report["errors"]
    )
