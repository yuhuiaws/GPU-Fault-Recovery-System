"""admin-lifecycle-sequence.sh --last-stage: stop after stage N, resume with --stage N+1.

The fakes and helpers are the BOOT-029 driver suite's own
(test_admin_lifecycle_sequence_script.py); this module only adds the
--last-stage cases, so that suite stays inside the per-file size limit.
"""

from __future__ import annotations

from pathlib import Path

from tests.regional.test_admin_lifecycle_sequence_script import (
    _admin_verbs,
    _install_fakes,
    _record,
    _run,
    _tool_log,
)


def test_last_stage_stops_after_the_named_stage_and_resumes_from_the_next(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)

    first = _run(tmp_path, env, "--last-stage", "3")

    assert first.returncode == 0, first.stdout + first.stderr
    assert (
        "stopped after stage 3 as asked (--last-stage 3); resume with --stage 4"
        in first.stderr
    ), first.stderr
    assert "all five driver stages PASS" not in first.stderr, first.stderr
    assert _admin_verbs(_tool_log(env)) == [
        "deploy",
        "remove-cluster",
        "join-cluster",
    ], "nothing after the last stage runs: the stage-1 site stays installed"
    record = _record(tmp_path)
    assert {number: stage["status"] for number, stage in record["stages"].items()} == {
        "1": "PASS",
        "2": "PASS",
        "3": "PASS",
    }, record["stages"]
    stage_three_digest = record["stages"]["3"]["log_sha256"]

    resumed = _run(tmp_path, env, "--stage", "4")

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert _admin_verbs(_tool_log(env))[3:] == ["uninstall", "deploy"], (
        "--stage 4 resumes at the next stage without repeating 1 to 3"
    )
    record = _record(tmp_path)
    assert record["resumed_from_stage"] == 4, record
    assert record["stages"]["3"]["log_sha256"] == stage_three_digest, (
        "the stopped run's stage records survive the resume"
    )
    assert record["stages"]["5"]["status"] == "PASS", record["stages"]["5"]
    assert "all five driver stages PASS" in resumed.stderr, resumed.stderr


def test_last_stage_usage_errors_exit_two_without_touching_any_tool(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)

    beyond = _run(tmp_path, env, "--last-stage", "6")
    before_start = _run(tmp_path, env, "--stage", "4", "--last-stage", "3")
    not_a_number = _run(tmp_path, env, "--last-stage", "three")

    for result in (beyond, before_start, not_a_number):
        assert result.returncode == 2, result.stderr
        assert "usage:" in result.stderr, result.stderr
    assert _tool_log(env) == [], "argument errors never reach the tools"
