"""HA-008: the release-failure evidence is the processor's own ERROR line."""

from __future__ import annotations

import json
import os

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


def test_real_fatal_exit_is_taken_over_by_an_independent_process(tmp_path) -> None:
    result = ha008.run_acceptance(tmp_path)
    assert result["verdict"] == "PASS", result
    assert result["sensitive_temp_files_removed"] is True
    for branch in result["branches"]:
        assert branch["exit_code"] == 70
        assert branch["stale_result_rejected"] is True
        assert (
            len({branch["first_process_id"], branch["second_process_id"], os.getpid()})
            == 3
        )
    assert not list(tmp_path.rglob("*.db")), (
        f"processor takeover must remove temporary databases under {tmp_path}"
    )
    assert not list(tmp_path.rglob("claim.json")), (
        f"processor takeover must remove sensitive claim files under {tmp_path}"
    )
    assert "lease_token" not in json.dumps(result)


def test_probe_failure_still_writes_a_failed_case(monkeypatch, tmp_path) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("probe timeout")

    monkeypatch.setattr(ha008, "_run_branch", fail)
    result = ha008.run_acceptance(tmp_path)
    assert result["verdict"] == "FAIL"
    assert (
        json.loads((tmp_path / f"{ha008.CASE_ID}.json").read_text())["verdict"]
        == "FAIL"
    )
