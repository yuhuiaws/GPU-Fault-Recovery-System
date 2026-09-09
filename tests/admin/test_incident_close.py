"""``gpu-fault-admin workflow-reconcile --close-incident``: the administrator
entry for closing an ESCALATED incident (DESTR-018 product gap, 2026-09-08).

The verb already reconciles BLOCKED workflow records by running a script in the
CPU ingress Pod; closing an ESCALATED incident is the same kind of operator
disposition, so it rides the same verb and the same in-Pod runner, and it calls
the very service function the API route calls (``IncidentClosureService``).
The operator identity is the resolved STS caller ARN; ``--reference`` is the
approved-change reference; ``--dry-run`` only reports the verdicts. One line
per incident; any refusal exits non-zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import cli, incident_close
from gpu_fault.admin.bootstrap_common import BootstrapError
from tests.admin.conftest import TEST_OPERATOR_ARN


def _site(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        release_config={
            "site_name": "staging",
            "aws_region": "us-west-2",
            "cpu_eks_arn": "arn:aws:eks:us-west-2:123456789012:cluster/cpu-control",
            "namespace": "gpu-fault",
            "cpu_kubeconfig": str(tmp_path / "cpu.kubeconfig"),
            "clusters": [{"cluster_id": "gpu-a", "context": "gpu-a-context"}],
        },
        environment={},
        source_sha256="a" * 64,
    )


def _runner(results: list[dict[str, Any]], calls: list[dict[str, Any]]):
    def run(_site: Any, payload: dict[str, Any], *, script: str) -> dict[str, Any]:
        calls.append({"payload": payload, "script": script})
        return {
            "mode": "incident-close",
            "dry_run": payload["dry_run"],
            "results": results,
        }

    return run


def test_closing_reports_one_line_per_incident_and_archives_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        incident_close,
        "_run_reconcile",
        _runner(
            [
                {"incident_id": "inc-a", "outcome": "closed", "state": "RECOVERED"},
                {
                    "incident_id": "inc-b",
                    "outcome": "already-recovered",
                    "state": "RECOVERED",
                },
            ],
            calls,
        ),
    )

    result = incident_close.run_incident_close(
        _site(tmp_path),
        tmp_path,
        incident_ids=("inc-a", "inc-b"),
        reason="node repaired after vendor visit",
        reference="CHG-2026-0908",
        dry_run=False,
    )

    assert incident_close.result_lines(result) == [
        "inc-a: closed",
        "inc-b: already-recovered",
    ]
    assert incident_close.exit_code(result) == 0
    [call] = calls
    assert call["script"] is incident_close.INCIDENT_CLOSE_SCRIPT
    assert call["payload"] == {
        "mode": "incident-close",
        "dry_run": False,
        "incident_ids": ["inc-a", "inc-b"],
        "reason": "node repaired after vendor visit",
        "reference": "CHG-2026-0908",
        "operator": TEST_OPERATOR_ARN,
    }
    assert result["actor"] == TEST_OPERATOR_ARN
    history = tmp_path / incident_close.INCIDENT_CLOSE_HISTORY_PATH
    [archived] = sorted(history.glob("*.json"))
    assert (
        json.loads(archived.read_text(encoding="utf-8"))["results"] == result["results"]
    )


def test_dry_run_needs_no_reference_writes_nothing_and_reports_verdicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        incident_close,
        "_run_reconcile",
        _runner(
            [
                {
                    "incident_id": "inc-a",
                    "outcome": "would-close",
                    "state": "ESCALATED",
                },
                {
                    "incident_id": "inc-b",
                    "outcome": "refused",
                    "state": "ESCALATED",
                    "reason": "incident inc-b still has an open workflow wf-b (RUNNING)",
                    "open_workflow_id": "wf-b",
                },
            ],
            calls,
        ),
    )

    result = incident_close.run_incident_close(
        _site(tmp_path),
        tmp_path,
        incident_ids=("inc-a", "inc-b"),
        reason="node repaired",
        reference=None,
        dry_run=True,
    )

    assert calls[0]["payload"]["dry_run"] is True
    assert incident_close.result_lines(result) == [
        "inc-a: would-close",
        "inc-b: refused(incident inc-b still has an open workflow wf-b (RUNNING))",
    ]
    assert incident_close.exit_code(result) == 1, "a refusal is reported non-zero"
    assert not (tmp_path / incident_close.INCIDENT_CLOSE_HISTORY_PATH).exists(), (
        "a dry run must not write the close history"
    )


def test_a_refused_close_exits_non_zero_and_names_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        incident_close,
        "_run_reconcile",
        _runner(
            [
                {"incident_id": "inc-a", "outcome": "closed", "state": "RECOVERED"},
                {
                    "incident_id": "inc-q",
                    "outcome": "refused",
                    "state": "QUARANTINED",
                    "reason": "incident inc-q is QUARANTINED; only an ESCALATED incident can be closed by an operator",
                },
            ],
            [],
        ),
    )

    result = incident_close.run_incident_close(
        _site(tmp_path),
        tmp_path,
        incident_ids=("inc-a", "inc-q"),
        reason="x",
        reference="CHG-1",
        dry_run=False,
    )

    assert incident_close.exit_code(result) == 1
    assert incident_close.result_lines(result)[1].startswith(
        "inc-q: refused(incident inc-q is QUARANTINED"
    )
    assert result["refused_incident_ids"] == ["inc-q"]
    assert result["closed_incident_ids"] == ["inc-a"]


@pytest.mark.parametrize(
    ("incident_ids", "reason", "reference", "dry_run", "match"),
    [
        pytest.param((), "x", "CHG-1", False, "at least one", id="no-ids"),
        pytest.param(("inc-a",), "", "CHG-1", False, "--reason", id="no-reason"),
        pytest.param(("inc-a",), "x", None, False, "--reference", id="no-reference"),
        pytest.param(
            ("inc-a",), "x", "bad ref!", False, "reference", id="bad-reference"
        ),
        pytest.param(("inc-a", "inc-a"), "x", "CHG-1", False, "repeated", id="dup-ids"),
    ],
)
def test_the_inputs_are_validated_before_anything_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    incident_ids,
    reason,
    reference,
    dry_run,
    match,
) -> None:
    def never(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("the runner must not be reached")

    monkeypatch.setattr(incident_close, "_run_reconcile", never)

    with pytest.raises(BootstrapError, match=match):
        incident_close.run_incident_close(
            _site(tmp_path),
            tmp_path,
            incident_ids=incident_ids,
            reason=reason,
            reference=reference,
            dry_run=dry_run,
        )


def test_the_in_pod_script_calls_the_same_service_as_the_api() -> None:
    script = incident_close.INCIDENT_CLOSE_SCRIPT

    assert "context.incident_closure" in script
    assert "close_incident(" in script and "preview(" in script
    assert "IncidentNotClosable" in script
    compile(script, "<incident-close>", "exec")


def test_the_verb_takes_close_incident_without_plan_or_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "site.yaml").write_text("placeholder\n", encoding="utf-8")
    monkeypatch.setattr(cli, "load_site", lambda *_args, **_kwargs: _site(tmp_path))
    seen: dict[str, Any] = {}

    def run_incident_close(site: Any, state: Path, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs, state_dir=state)
        return {
            "results": [
                {"incident_id": "inc-a", "outcome": "closed", "state": "RECOVERED"}
            ],
            "closed_incident_ids": ["inc-a"],
            "refused_incident_ids": [],
        }

    monkeypatch.setattr(cli, "run_incident_close", run_incident_close)

    exit_code = cli.run(
        cli.parser().parse_args(
            [
                "workflow-reconcile",
                "--state-dir",
                str(state_dir),
                "--close-incident",
                "inc-a",
                "--reason",
                "node repaired",
                "--reference",
                "CHG-2026-0908",
            ]
        )
    )

    assert exit_code == 0
    assert seen["incident_ids"] == ("inc-a",)
    assert seen["reason"] == "node repaired"
    assert seen["reference"] == "CHG-2026-0908"
    assert seen["dry_run"] is False
    assert seen["state_dir"] == state_dir.resolve()
    assert capsys.readouterr().out.splitlines() == ["inc-a: closed"]
