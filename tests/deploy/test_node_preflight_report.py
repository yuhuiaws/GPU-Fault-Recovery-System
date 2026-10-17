"""The node preflight batch reports every refused node to the release engine.

A sensitive command's failure keeps three ``ERROR:`` lines of 200 characters,
so the batch's notes named one node, cut mid-reason (live 2026-10-17). The
batch now keeps, per failed node, its verdict's reasons whole and the remedy
its Job printed, with every node's DCGM observation, and writes them to the
private file the engine names (``NODE_PREFLIGHT_REPORT_ENV``) on refusal and
on success alike. The engine half is
``tests/regional/test_release_node_preflight_refusal.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_fault_release import regional_node_batch as batch
from tests.deploy.test_node_batch_rendering import Runner, rendered_batch

__all__ = ["rendered_batch"]

MISSING = "host prerequisites unavailable: nvidia-dcgm.service (inactive)"
REMEDY = "install and enable nvidia-dcgm (see administrator-operations §5.0.1)"


def _logs(node: str) -> str:
    verdict = {
        "node_id": node,
        "node_uid": "0f6c1d2e-9a8b-4c7d-8e6f-5a4b3c2d1e0f",
        "status": "FAILED",
        "reasons": [MISSING],
        "dcgm": {"mode": "host", "state": "installed-inactive", "conflict": False},
    }
    return "\n".join(
        [
            "INFO  checking host prerequisites",
            json.dumps(verdict),
            f"ERROR: {MISSING}; {REMEDY}",
        ]
    )


class RefusingRunner(Runner):
    """Host Jobs of ``failing`` nodes fail; their logs carry a FAILED verdict."""

    def __init__(self, jobs: dict[str, str], failing: set[str]) -> None:
        super().__init__()
        self.jobs = jobs
        self.failing = failing

    def run(self, command, **kwargs):
        if "logs" in command:
            job = command[command.index("logs") + 1].removeprefix("job/")
            return _logs(self.jobs[job]) if self.jobs[job] in self.failing else ""
        if command[0] == "bash" and self.jobs.get(command[3]) in self.failing:
            super().run(command, **kwargs)
            raise RuntimeError(f"Job {command[3]} failed: BackoffLimitExceeded")
        return super().run(command, **kwargs)


def test_every_refused_node_is_reported_with_its_reasons_remedy_and_state(
    rendered_batch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, template, output, scope, identity = rendered_batch
    prepared = batch.prepare_node_batch(source, template, output, scope, identity)
    jobs = {node.job_name: node.node.name for node in prepared}
    runner = RefusingRunner(jobs, failing={"node-a", "node-c"})
    observations: dict = {}
    refused: dict = {}

    with pytest.raises(RuntimeError, match="2 node preflights failed"):
        batch.run_node_preflights(
            prepared,
            scope,
            identity,
            runner,
            workers=1,
            observations=observations,
            refused=refused,
        )

    assert refused == {
        "node-a": {"reasons": [MISSING], "remedies": [REMEDY]},
        "node-c": {"reasons": [MISSING], "remedies": [REMEDY]},
    }, "each failed node's verdict reasons whole, and the remedy its Job named"
    assert set(observations) == {"node-a", "node-c"}
    assert observations["node-a"]["state"] == "installed-inactive"

    report = tmp_path / "private" / "report.json"
    report.parent.mkdir(mode=0o700)
    monkeypatch.setenv(batch.NODE_PREFLIGHT_REPORT_ENV, str(report))
    batch.write_preflight_report(len(prepared), observations, refused, failed=True)
    document = json.loads(report.read_text(encoding="utf-8"))
    assert document["status"] == "FAILED" and document["node_count"] == 3
    assert document["refused"] == refused
    assert set(document["dcgm_hostengine"]) == {"node-a", "node-c"}
    assert report.stat().st_mode & 0o777 == 0o600, "the report is private"


def test_a_job_without_a_verdict_reports_its_error_lines_or_its_failure() -> None:
    failure = RuntimeError("Job gpu-fault-preflight-a failed: BackoffLimitExceeded")
    assert batch.preflight_refusal(
        "ERROR: host Python 3.12 venv support is unavailable\n", failure
    ) == {"reasons": ["host Python 3.12 venv support is unavailable"], "remedies": []}
    assert batch.preflight_refusal("", failure) == {
        "reasons": ["Job gpu-fault-preflight-a failed: BackoffLimitExceeded"],
        "remedies": [],
    }, "a Job that never ran names the failure"


def test_the_report_is_written_only_for_an_engine_that_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(batch.NODE_PREFLIGHT_REPORT_ENV, raising=False)
    batch.write_preflight_report(1, {}, {}, failed=False)
    assert list(tmp_path.iterdir()) == [], "an older engine gets no report"

    report = tmp_path / "report.json"
    monkeypatch.setenv(batch.NODE_PREFLIGHT_REPORT_ENV, str(report))
    batch.write_preflight_report(
        1, {"node-a": {"state": "host-active"}}, {}, failed=False
    )
    document = json.loads(report.read_text(encoding="utf-8"))
    assert document == {
        "status": "PASSED",
        "node_count": 1,
        "dcgm_hostengine": {"node-a": {"state": "host-active"}},
        "refused": {},
    }
    batch.write_preflight_report(1, {}, {}, failed=True)
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "PASSED", (
        "an existing file is never replaced (O_EXCL): the engine names a new one"
    )
