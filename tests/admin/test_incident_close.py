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

from gpu_fault.adapters.common import quarantine_taint_value
from gpu_fault.admin import cli, incident_close
from gpu_fault.admin import workflow_reconcile as reconcile
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


# --------------------------------------------------------------------------
# --close-escalated: discover the ESCALATED queue instead of naming the ids.
# --------------------------------------------------------------------------


def _discovery_runner(report: dict[str, Any], calls: list[dict[str, Any]]):
    def run(_site: Any, payload: dict[str, Any], *, script: str) -> dict[str, Any]:
        calls.append({"payload": payload, "script": script})
        return {"mode": "incident-close", "dry_run": payload["dry_run"], **report}

    return run


def test_close_escalated_dry_run_needs_no_reason_and_lists_every_discovered_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        incident_close,
        "_run_reconcile",
        _discovery_runner(
            {
                "discovered_total": 3,
                "discovered_cluster_ids": ["gpu-a", "gpu-b"],
                "discovery_limit_reached": [],
                "discovered_incident_ids": ["inc-old", "inc-mid", "inc-new"],
                "results": [
                    {
                        "incident_id": "inc-old",
                        "outcome": "would-close",
                        "state": "ESCALATED",
                        "reason": None,
                        "open_workflow_id": None,
                    },
                    {
                        "incident_id": "inc-mid",
                        "outcome": "refused",
                        "state": "ESCALATED",
                        "reason": "incident inc-mid still has an open workflow wf-m (RUNNING)",
                        "open_workflow_id": "wf-m",
                    },
                    {
                        "incident_id": "inc-new",
                        "outcome": "would-close",
                        "state": "ESCALATED",
                        "reason": None,
                        "open_workflow_id": None,
                    },
                ],
            },
            calls,
        ),
    )

    result = incident_close.run_incident_close(
        _site(tmp_path),
        tmp_path,
        selector=incident_close.escalated_selector(),
        reason="",
        reference=None,
        dry_run=True,
    )

    [call] = calls
    assert call["script"] is incident_close.INCIDENT_CLOSE_SCRIPT
    assert call["payload"]["incident_ids"] == []
    assert call["payload"]["dry_run"] is True
    assert call["payload"]["selector"] == {
        "mode": "escalated",
        "states": ["ESCALATED"],
        "cluster_ids": None,
        "node_ids": None,
        "max_items": None,
        "discovery_limit": incident_close.DISCOVERY_LIMIT,
        "order": "created_at,incident_id",
    }
    assert result["discovered_incident_ids"] == ["inc-old", "inc-mid", "inc-new"]
    assert result["selector"]["mode"] == "escalated"
    assert incident_close.result_lines(result) == [
        "discovered 3 ESCALATED incident(s) across 2 cluster(s): gpu-a, gpu-b",
        "inc-old: would-close",
        "inc-mid: refused(incident inc-mid still has an open workflow wf-m (RUNNING))",
        "inc-new: would-close",
    ]
    assert incident_close.exit_code(result) == 1
    assert not (tmp_path / incident_close.INCIDENT_CLOSE_HISTORY_PATH).exists(), (
        "a dry run must not write the close history"
    )


def test_close_escalated_reports_the_max_items_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        incident_close,
        "_run_reconcile",
        _discovery_runner(
            {
                "discovered_total": 89,
                "discovered_cluster_ids": ["gpu-a"],
                "discovery_limit_reached": [],
                "discovered_incident_ids": ["inc-1", "inc-2"],
                "results": [
                    {
                        "incident_id": "inc-1",
                        "outcome": "would-close",
                        "state": "ESCALATED",
                    },
                    {
                        "incident_id": "inc-2",
                        "outcome": "would-close",
                        "state": "ESCALATED",
                    },
                ],
            },
            calls,
        ),
    )

    result = incident_close.run_incident_close(
        _site(tmp_path),
        tmp_path,
        selector=incident_close.escalated_selector(max_items=2),
        reason="",
        reference=None,
        dry_run=True,
    )

    assert calls[0]["payload"]["selector"]["max_items"] == 2
    assert incident_close.result_lines(result)[0] == (
        "discovered 89 ESCALATED incident(s) across 1 cluster(s): gpu-a; "
        "processing 2 (capped by --max-items 2)"
    )
    assert incident_close.exit_code(result) == 0


def test_close_escalated_applied_archives_the_discovery_with_the_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        incident_close,
        "_run_reconcile",
        _discovery_runner(
            {
                "discovered_total": 2,
                "discovered_cluster_ids": ["gpu-a"],
                "discovery_limit_reached": ["gpu-a"],
                "discovered_incident_ids": ["inc-1", "inc-2"],
                "results": [
                    {"incident_id": "inc-1", "outcome": "closed", "state": "RECOVERED"},
                    {
                        "incident_id": "inc-2",
                        "outcome": "refused",
                        "reason": "incident inc-2 still has an open workflow wf-2 (RUNNING)",
                    },
                ],
            },
            [],
        ),
    )

    result = incident_close.run_incident_close(
        _site(tmp_path),
        tmp_path,
        selector=incident_close.escalated_selector(max_items=50),
        reason="vendor repaired the rack",
        reference="CHG-1",
        dry_run=False,
    )

    lines = incident_close.result_lines(result)
    assert lines[0] == "discovered 2 ESCALATED incident(s) across 1 cluster(s): gpu-a"
    assert lines[1].startswith(
        f"discovery hit the {incident_close.DISCOVERY_LIMIT}-incident ceiling on gpu-a"
    ), "a cluster at the per-cluster discovery ceiling is named in the header"
    assert lines[2:] == [
        "inc-1: closed",
        "inc-2: refused(incident inc-2 still has an open workflow wf-2 (RUNNING))",
    ]
    assert result["closed_incident_ids"] == ["inc-1"]
    assert result["refused_incident_ids"] == ["inc-2"]
    assert result["actor"] == TEST_OPERATOR_ARN
    history = tmp_path / incident_close.INCIDENT_CLOSE_HISTORY_PATH
    [archived] = sorted(history.glob("*.json"))
    document = json.loads(archived.read_text(encoding="utf-8"))
    assert document["discovered_incident_ids"] == ["inc-1", "inc-2"]
    assert document["selector"]["mode"] == "escalated"
    assert document["selector"]["max_items"] == 50
    assert document["results"] == result["results"]
    assert {
        "applied_at",
        "reference",
        "closed_incident_ids",
        "refused_incident_ids",
    } <= (set(document)), "the archived shape of --close-incident is kept"


def test_explicit_ids_still_archive_the_selector_and_an_empty_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        incident_close,
        "_run_reconcile",
        _runner(
            [{"incident_id": "inc-a", "outcome": "closed", "state": "RECOVERED"}], []
        ),
    )

    result = incident_close.run_incident_close(
        _site(tmp_path),
        tmp_path,
        incident_ids=("inc-a",),
        reason="x",
        reference="CHG-1",
        dry_run=False,
    )

    assert result["selector"] == {"mode": "incident-ids", "incident_ids": ["inc-a"]}
    assert result["discovered_incident_ids"] == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param(
            {
                "incident_ids": ("inc-a",),
                "reason": "x",
                "reference": "CHG-1",
                "dry_run": False,
            },
            "cannot be combined",
            id="with-close-incident",
        ),
        pytest.param(
            {"reason": "", "reference": "CHG-1", "dry_run": False},
            "--close-escalated requires --reason",
            id="applied-without-reason",
        ),
        pytest.param(
            {"reason": "x", "reference": None, "dry_run": False},
            "--close-escalated requires --reference",
            id="applied-without-reference",
        ),
    ],
)
def test_close_escalated_never_reaches_the_pod_without_reason_and_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kwargs, match
) -> None:
    def never(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("the runner must not be reached")

    monkeypatch.setattr(incident_close, "_run_reconcile", never)

    with pytest.raises(BootstrapError, match=match):
        incident_close.run_incident_close(
            _site(tmp_path),
            tmp_path,
            selector=incident_close.escalated_selector(),
            **kwargs,
        )


def test_a_zero_max_items_cap_is_refused_before_the_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        incident_close, "_run_reconcile", lambda *a, **k: pytest.fail("unreachable")
    )

    with pytest.raises(BootstrapError, match="--max-items"):
        incident_close.run_incident_close(
            _site(tmp_path),
            tmp_path,
            selector=incident_close.escalated_selector(max_items=0),
            reason="",
            reference=None,
            dry_run=True,
        )


def test_the_in_pod_script_discovers_escalated_only_oldest_first_and_caps(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Runs INCIDENT_CLOSE_SCRIPT against an in-memory control plane: the
    discovery is ESCALATED only (QUARANTINED and RECOVERED never appear),
    spans every registered cluster, orders by created_at then id, honours the
    cap, and judges each id exactly as an explicit list would."""

    import io
    import sys
    from datetime import timedelta

    from gpu_fault.app import ApplicationContext
    from gpu_fault.models import IncidentState
    from tests._builders import build_store, copy_model
    from tests.orchestration._incident_closure_support import NOW, _escalated_reset
    from tests.regional._regional_support import TOKEN_A, TOKEN_B, registration

    store = build_store()
    store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    store.save_regional_cluster(registration("cluster-b", TOKEN_B))
    _escalated_reset(store, incident_id="inc-newest", node_ids=("n1",))
    oldest, _ = _escalated_reset(store, incident_id="inc-z-oldest", node_ids=("n2",))
    store.save_incident(
        copy_model(oldest, created_at=NOW - timedelta(days=2)), expected=oldest
    )
    _escalated_reset(
        store, incident_id="inc-b", node_ids=("n3",), cluster_id="cluster-b"
    )
    quarantined, _ = _escalated_reset(store, incident_id="inc-q", node_ids=("n4",))
    store.save_incident(
        copy_model(quarantined, state=IncidentState.QUARANTINED), expected=quarantined
    )
    recovered, _ = _escalated_reset(store, incident_id="inc-r", node_ids=("n5",))
    store.save_incident(
        copy_model(recovered, state=IncidentState.RECOVERED), expected=recovered
    )
    same_created = store.get_incident("inc-b")
    store.save_incident(
        copy_model(
            same_created, created_at=store.get_incident("inc-newest").created_at
        ),
        expected=same_created,
    )
    context = ApplicationContext(store=store)
    monkeypatch.setattr(
        ApplicationContext, "from_environment", classmethod(lambda cls: context)
    )

    def run(payload: dict[str, Any]) -> dict[str, Any]:
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        exec(
            compile(incident_close.INCIDENT_CLOSE_SCRIPT, "<incident-close>", "exec"),
            {},
        )
        return json.loads(capsys.readouterr().out)

    base = {
        "mode": "incident-close",
        "incident_ids": [],
        "reason": "vendor repaired",
        "reference": "CHG-1",
        "operator": "ops@example",
    }
    preview = run(
        {**base, "dry_run": True, "selector": incident_close.escalated_selector()}
    )
    capped = run(
        {
            **base,
            "dry_run": True,
            "selector": incident_close.escalated_selector(max_items=1),
        }
    )
    still_escalated = store.get_incident("inc-z-oldest").state
    applied = run(
        {**base, "dry_run": False, "selector": incident_close.escalated_selector()}
    )

    assert preview["discovered_incident_ids"] == [
        "inc-z-oldest",
        "inc-b",
        "inc-newest",
    ], "created_at first, then incident_id for the tie"
    assert preview["discovered_total"] == 3
    assert preview["discovered_cluster_ids"] == ["cluster-a", "cluster-b"]
    assert preview["discovery_limit_reached"] == []
    assert [(r["incident_id"], r["outcome"]) for r in preview["results"]] == [
        ("inc-z-oldest", "would-close"),
        ("inc-b", "would-close"),
        ("inc-newest", "would-close"),
    ]
    assert still_escalated is IncidentState.ESCALATED, "a dry run writes nothing"
    assert capped["discovered_total"] == 3
    assert capped["discovered_incident_ids"] == ["inc-z-oldest"]
    assert [r["incident_id"] for r in capped["results"]] == ["inc-z-oldest"]
    assert [(r["incident_id"], r["outcome"]) for r in applied["results"]] == [
        ("inc-z-oldest", "closed"),
        ("inc-b", "closed"),
        ("inc-newest", "closed"),
    ]
    assert store.get_incident("inc-b").state is IncidentState.RECOVERED
    assert store.get_incident("inc-q").state is IncidentState.QUARANTINED, (
        "QUARANTINED is never discovered, let alone closed"
    )
    assert store.get_incident("inc-z-oldest").reasons[-1] == (
        "operator closed: vendor repaired by ops@example"
    )


def test_the_verb_takes_close_escalated_with_dry_run_and_max_items(
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
            "selector": kwargs["selector"],
            "discovered_total": 1,
            "discovered_cluster_ids": ["gpu-a"],
            "discovered_incident_ids": ["inc-a"],
            "results": [
                {"incident_id": "inc-a", "outcome": "would-close", "state": "ESCALATED"}
            ],
            "closed_incident_ids": [],
            "refused_incident_ids": [],
        }

    monkeypatch.setattr(cli, "run_incident_close", run_incident_close)

    exit_code = cli.run(
        cli.parser().parse_args(
            [
                "workflow-reconcile",
                "--state-dir",
                str(state_dir),
                "--close-escalated",
                "--max-items",
                "10",
                "--dry-run",
            ]
        )
    )

    assert exit_code == 0
    assert "incident_ids" not in seen
    assert seen["selector"] == incident_close.escalated_selector(max_items=10)
    assert seen["reason"] == ""
    assert seen["reference"] is None
    assert seen["dry_run"] is True
    assert capsys.readouterr().out.splitlines() == [
        "discovered 1 ESCALATED incident(s) across 1 cluster(s): gpu-a",
        "inc-a: would-close",
    ]


def test_the_verb_refuses_close_escalated_together_with_close_incident(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.parser().parse_args(
            [
                "workflow-reconcile",
                "--state-dir",
                str(tmp_path),
                "--close-escalated",
                "--close-incident",
                "inc-a",
            ]
        )

    assert raised.value.code == 2


# --------------------------------------------------------------------------
# --close-quarantined: QUARANTINED incidents close on node isolation evidence
# read through the site's GPU kubeconfig, in a second pass to the Pod.
# --------------------------------------------------------------------------


def _gpu_site(tmp_path: Path) -> SimpleNamespace:
    site = _site(tmp_path)
    site.release_config["gpu_kubeconfig"] = str(tmp_path / "gpu.kubeconfig")
    return site


def _node(
    name: str,
    *,
    unschedulable: bool = False,
    taint_value: str | None = None,
    annotations: dict[str, str] | None = None,
) -> dict[str, Any]:
    taints = (
        [
            {
                "key": reconcile.QUARANTINE_TAINT,
                "value": taint_value,
                "effect": "NoSchedule",
            }
        ]
        if taint_value is not None
        else []
    )
    return {
        "metadata": {"name": name, "annotations": dict(annotations or {})},
        "spec": {"unschedulable": unschedulable, "taints": taints},
    }


def _kubectl_nodes(*nodes: dict[str, Any], calls: list[list[str]] | None = None):
    def run(args, **_kwargs):
        if calls is not None:
            calls.append(list(args))
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"items": list(nodes)}), stderr=""
        )

    return run


def _quarantined_pending(incident_id: str, *node_ids: str) -> dict[str, Any]:
    """What the first Pod pass reports for a QUARANTINED incident."""

    return {
        "incident_id": incident_id,
        "outcome": "refused",
        "state": "QUARANTINED",
        "reason": f"incident {incident_id} is QUARANTINED; only an ESCALATED ...",
        "open_workflow_id": None,
        "cluster_id": "gpu-a",
        "node_ids": list(node_ids),
        "evidence_required": True,
        "isolation_reasons": [],
    }


def _two_pass_runner(first: dict[str, Any], calls: list[dict[str, Any]]):
    """A Pod double: the first call answers ``first``; the evidence pass judges
    each incident from the evidence it is handed, as the service would."""

    def run(_site: Any, payload: dict[str, Any], *, script: str) -> dict[str, Any]:
        calls.append({"payload": payload, "script": script})
        if "evidence" not in payload:
            return {"mode": "incident-close", "dry_run": payload["dry_run"], **first}
        results = []
        for incident_id in payload["incident_ids"]:
            evidence = payload["evidence"][incident_id]
            own = quarantine_taint_value(incident_id)
            blocked = [
                item
                for item in evidence
                if item["unschedulable"] or item["quarantine_taint_value"] == own
            ]
            if blocked:
                results.append(
                    {
                        "incident_id": incident_id,
                        "outcome": "refused",
                        "state": "QUARANTINED",
                        "reason": (
                            f"incident {incident_id} is QUARANTINED and the node "
                            f"evidence does not clear it: node {blocked[0]['node_id']} "
                            "still carries the gpu-fault.io/quarantined taint of "
                            f"incident {incident_id}"
                        ),
                    }
                )
                continue
            results.append(
                {
                    "incident_id": incident_id,
                    "outcome": "would-close" if payload["dry_run"] else "closed",
                    "state": "QUARANTINED" if payload["dry_run"] else "RECOVERED",
                    "isolation_nodes": sorted(item["node_id"] for item in evidence),
                }
            )
        return {
            "mode": "incident-close",
            "dry_run": payload["dry_run"],
            "results": results,
        }

    return run


def test_close_quarantined_dry_run_reads_node_evidence_and_judges_each_incident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    kubectl: list[list[str]] = []
    monkeypatch.setattr(
        incident_close,
        "_run_reconcile",
        _two_pass_runner(
            {
                "discovered_total": 2,
                "discovered_cluster_ids": ["gpu-a"],
                "discovery_limit_reached": [],
                "discovered_incident_ids": ["inc-q1", "inc-q2"],
                "results": [
                    _quarantined_pending("inc-q1", "node-a"),
                    _quarantined_pending("inc-q2", "node-b"),
                ],
            },
            calls,
        ),
    )
    monkeypatch.setattr(
        reconcile.subprocess,
        "run",
        _kubectl_nodes(
            _node("node-a"),
            _node(
                "node-b",
                unschedulable=True,
                taint_value=quarantine_taint_value("inc-q2"),
                annotations={"gpu-fault.io/incident-id": "inc-q2"},
            ),
            calls=kubectl,
        ),
    )

    result = incident_close.run_incident_close(
        _gpu_site(tmp_path),
        tmp_path,
        selector=incident_close.quarantined_selector(),
        reason="",
        reference=None,
        dry_run=True,
    )

    first, second = calls
    assert first["payload"]["selector"] == {
        "mode": "quarantined",
        "states": ["QUARANTINED"],
        "cluster_ids": None,
        "node_ids": None,
        "max_items": None,
        "discovery_limit": incident_close.DISCOVERY_LIMIT,
        "order": "created_at,incident_id",
    }
    assert "evidence" not in first["payload"]
    assert second["script"] is incident_close.INCIDENT_CLOSE_SCRIPT
    assert "selector" not in second["payload"], "the second pass names the ids"
    assert second["payload"]["incident_ids"] == ["inc-q1", "inc-q2"]
    assert second["payload"]["dry_run"] is True
    assert second["payload"]["evidence"] == {
        "inc-q1": [
            {
                "node_id": "node-a",
                "exists": True,
                "unschedulable": False,
                "quarantine_taint_value": None,
                "isolation_annotations": {},
            }
        ],
        "inc-q2": [
            {
                "node_id": "node-b",
                "exists": True,
                "unschedulable": True,
                "quarantine_taint_value": quarantine_taint_value("inc-q2"),
                "isolation_annotations": {"gpu-fault.io/incident-id": "inc-q2"},
            }
        ],
    }
    [nodes_call] = kubectl
    assert nodes_call[:5] == [
        "kubectl",
        "--kubeconfig",
        str(tmp_path / "gpu.kubeconfig"),
        "--context",
        "gpu-a-context",
    ], "node evidence comes from the site's GPU kubeconfig, once per cluster"
    assert incident_close.result_lines(result) == [
        "discovered 2 QUARANTINED incident(s) across 1 cluster(s): gpu-a",
        "inc-q1: would-close (isolation absent on node-a)",
        "inc-q2: refused(incident inc-q2 is QUARANTINED and the node evidence does "
        "not clear it: node node-b still carries the gpu-fault.io/quarantined taint "
        "of incident inc-q2)",
    ]
    assert result["isolation_evidence"] == second["payload"]["evidence"]
    assert incident_close.exit_code(result) == 1
    assert not (tmp_path / incident_close.INCIDENT_CLOSE_HISTORY_PATH).exists(), (
        "a dry run must not write the close history"
    )


def test_close_quarantined_applied_closes_and_archives_the_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        incident_close,
        "_run_reconcile",
        _two_pass_runner(
            {
                "discovered_total": 1,
                "discovered_cluster_ids": ["gpu-a"],
                "discovery_limit_reached": [],
                "discovered_incident_ids": ["inc-q1"],
                "results": [_quarantined_pending("inc-q1", "node-a")],
            },
            calls,
        ),
    )
    monkeypatch.setattr(reconcile.subprocess, "run", _kubectl_nodes(_node("node-a")))

    result = incident_close.run_incident_close(
        _gpu_site(tmp_path),
        tmp_path,
        selector=incident_close.quarantined_selector(max_items=10),
        reason="isolation released by cleanup, node healthy",
        reference="CHG-2",
        dry_run=False,
    )

    assert calls[1]["payload"]["dry_run"] is False
    assert (
        calls[1]["payload"]["reason"] == "isolation released by cleanup, node healthy"
    )
    assert incident_close.result_lines(result) == [
        "discovered 1 QUARANTINED incident(s) across 1 cluster(s): gpu-a",
        "inc-q1: closed (isolation absent on node-a)",
    ]
    assert result["closed_incident_ids"] == ["inc-q1"]
    assert incident_close.exit_code(result) == 0
    history = tmp_path / incident_close.INCIDENT_CLOSE_HISTORY_PATH
    [archived] = sorted(history.glob("*.json"))
    document = json.loads(archived.read_text(encoding="utf-8"))
    assert document["selector"]["mode"] == "quarantined"
    assert document["isolation_evidence"] == {
        "inc-q1": [
            {
                "node_id": "node-a",
                "exists": True,
                "unschedulable": False,
                "quarantine_taint_value": None,
                "isolation_annotations": {},
            }
        ]
    }, "the evidence each close rested on is archived with the run"


def test_an_explicit_quarantined_id_gets_the_same_evidence_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        incident_close,
        "_run_reconcile",
        _two_pass_runner(
            {
                "results": [
                    {"incident_id": "inc-a", "outcome": "closed", "state": "RECOVERED"},
                    _quarantined_pending("inc-q1", "node-a"),
                ]
            },
            calls,
        ),
    )
    monkeypatch.setattr(reconcile.subprocess, "run", _kubectl_nodes(_node("node-a")))

    result = incident_close.run_incident_close(
        _gpu_site(tmp_path),
        tmp_path,
        incident_ids=("inc-a", "inc-q1"),
        reason="x",
        reference="CHG-1",
        dry_run=False,
    )

    assert calls[1]["payload"]["incident_ids"] == ["inc-q1"], (
        "the ESCALATED close landed in the first pass; only the QUARANTINED id is re-judged"
    )
    assert incident_close.result_lines(result) == [
        "inc-a: closed",
        "inc-q1: closed (isolation absent on node-a)",
    ]
    assert result["closed_incident_ids"] == ["inc-a", "inc-q1"]


def test_without_a_gpu_kubeconfig_the_quarantined_close_is_refused_not_guessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        incident_close,
        "_run_reconcile",
        _two_pass_runner(
            {
                "discovered_total": 1,
                "discovered_cluster_ids": ["gpu-a"],
                "discovery_limit_reached": [],
                "discovered_incident_ids": ["inc-q1"],
                "results": [_quarantined_pending("inc-q1", "node-a")],
            },
            calls,
        ),
    )
    monkeypatch.setattr(
        reconcile.subprocess,
        "run",
        lambda *_a, **_k: pytest.fail("the default kubeconfig must never be read"),
    )

    result = incident_close.run_incident_close(
        _site(tmp_path),  # no gpu_kubeconfig, no KUBECONFIG
        tmp_path,
        selector=incident_close.quarantined_selector(),
        reason="",
        reference=None,
        dry_run=True,
    )

    assert len(calls) == 1, "nothing to hand to the Pod: no second pass"
    [header, line] = incident_close.result_lines(result)
    assert line.startswith(
        "inc-q1: refused(workflow reconcile has no GPU kubeconfig for cluster gpu-a"
    ), line
    assert incident_close.exit_code(result) == 1


def test_a_missing_node_is_reported_as_evidence_the_service_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        incident_close,
        "_run_reconcile",
        _two_pass_runner(
            {"results": [_quarantined_pending("inc-q1", "node-gone")]}, calls
        ),
    )
    monkeypatch.setattr(reconcile.subprocess, "run", _kubectl_nodes(_node("node-a")))

    incident_close.run_incident_close(
        _gpu_site(tmp_path),
        tmp_path,
        incident_ids=("inc-q1",),
        reason="",
        reference=None,
        dry_run=True,
    )

    assert calls[1]["payload"]["evidence"]["inc-q1"] == [
        {
            "node_id": "node-gone",
            "exists": False,
            "unschedulable": False,
            "quarantine_taint_value": None,
            "isolation_annotations": {},
        }
    ]


def test_close_quarantined_cannot_be_combined_with_explicit_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        incident_close, "_run_reconcile", lambda *a, **k: pytest.fail("unreachable")
    )

    with pytest.raises(
        BootstrapError, match="--close-quarantined and --close-incident"
    ):
        incident_close.run_incident_close(
            _site(tmp_path),
            tmp_path,
            incident_ids=("inc-a",),
            selector=incident_close.quarantined_selector(),
            reason="x",
            reference="CHG-1",
            dry_run=False,
        )

    with pytest.raises(BootstrapError, match="--close-quarantined requires --reason"):
        incident_close.run_incident_close(
            _site(tmp_path),
            tmp_path,
            selector=incident_close.quarantined_selector(),
            reason="",
            reference="CHG-1",
            dry_run=False,
        )


def test_the_in_pod_script_closes_a_quarantined_incident_only_with_clearing_evidence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """INCIDENT_CLOSE_SCRIPT against an in-memory control plane: the quarantined
    selector discovers QUARANTINED only; without evidence the incident comes
    back ``evidence_required`` with its cluster and nodes; with clearing
    evidence it closes; with its own taint still on the node it is refused."""

    import io
    import sys

    from gpu_fault.app import ApplicationContext
    from gpu_fault.models import IncidentState
    from tests._builders import build_store, copy_model
    from tests.orchestration._incident_closure_support import _escalated_reset
    from tests.regional._regional_support import TOKEN_A, registration

    store = build_store()
    store.save_regional_cluster(registration("cluster-a", TOKEN_A))
    _escalated_reset(store, incident_id="inc-e", node_ids=("n0",))
    for incident_id, node in (("inc-q1", "n1"), ("inc-q2", "n2")):
        incident, _ = _escalated_reset(store, incident_id=incident_id, node_ids=(node,))
        store.save_incident(
            copy_model(incident, state=IncidentState.QUARANTINED), expected=incident
        )
    context = ApplicationContext(store=store)
    monkeypatch.setattr(
        ApplicationContext, "from_environment", classmethod(lambda cls: context)
    )

    def run(payload: dict[str, Any]) -> dict[str, Any]:
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        exec(
            compile(incident_close.INCIDENT_CLOSE_SCRIPT, "<incident-close>", "exec"),
            {},
        )
        return json.loads(capsys.readouterr().out)

    base = {
        "mode": "incident-close",
        "incident_ids": [],
        "reason": "isolation released",
        "reference": "CHG-1",
        "operator": "ops@example",
    }
    first = run(
        {**base, "dry_run": False, "selector": incident_close.quarantined_selector()}
    )
    state_after_first = store.get_incident("inc-q1").state
    evidence = {
        "inc-q1": [
            {
                "node_id": "n1",
                "exists": True,
                "unschedulable": False,
                "quarantine_taint_value": None,
                "isolation_annotations": {},
            }
        ],
        "inc-q2": [
            {
                "node_id": "n2",
                "exists": True,
                "unschedulable": False,
                "quarantine_taint_value": quarantine_taint_value("inc-q2"),
                "isolation_annotations": {},
            }
        ],
    }
    second = run(
        {
            **base,
            "dry_run": False,
            "incident_ids": ["inc-q1", "inc-q2"],
            "evidence": evidence,
        }
    )

    assert first["discovered_incident_ids"] == ["inc-q1", "inc-q2"], (
        "ESCALATED is not discovered by the quarantined selector"
    )
    assert [
        (r["incident_id"], r["outcome"], r["evidence_required"])
        for r in first["results"]
    ] == [("inc-q1", "refused", True), ("inc-q2", "refused", True)]
    assert first["results"][0]["cluster_id"] == "cluster-a"
    assert first["results"][0]["node_ids"] == ["n1"]
    assert state_after_first is IncidentState.QUARANTINED, (
        "nothing is closed without evidence"
    )
    assert [(r["incident_id"], r["outcome"]) for r in second["results"]] == [
        ("inc-q1", "closed"),
        ("inc-q2", "refused"),
    ]
    assert second["results"][0]["isolation_nodes"] == ["n1"]
    assert "taint of incident inc-q2" in second["results"][1]["reason"]
    assert store.get_incident("inc-q1").state is IncidentState.RECOVERED
    assert store.get_incident("inc-q1").reasons[-2:] == [
        "operator closed: isolation released by ops@example",
        "isolation no longer present on node n1",
    ]
    assert store.get_incident("inc-q2").state is IncidentState.QUARANTINED
    assert store.get_incident("inc-e").state is IncidentState.ESCALATED


def test_the_verb_takes_close_quarantined_and_refuses_it_with_the_other_selectors(
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
            "selector": kwargs["selector"],
            "discovered_total": 1,
            "discovered_cluster_ids": ["gpu-a"],
            "discovered_incident_ids": ["inc-q"],
            "results": [
                {
                    "incident_id": "inc-q",
                    "outcome": "would-close",
                    "state": "QUARANTINED",
                    "isolation_nodes": ["node-a"],
                }
            ],
            "closed_incident_ids": [],
            "refused_incident_ids": [],
        }

    monkeypatch.setattr(cli, "run_incident_close", run_incident_close)

    exit_code = cli.run(
        cli.parser().parse_args(
            [
                "workflow-reconcile",
                "--state-dir",
                str(state_dir),
                "--close-quarantined",
                "--max-items",
                "5",
                "--dry-run",
            ]
        )
    )

    assert exit_code == 0
    assert seen["selector"] == incident_close.quarantined_selector(max_items=5)
    assert seen["dry_run"] is True and "incident_ids" not in seen
    assert capsys.readouterr().out.splitlines() == [
        "discovered 1 QUARANTINED incident(s) across 1 cluster(s): gpu-a",
        "inc-q: would-close (isolation absent on node-a)",
    ]
    for other in (["--close-escalated"], ["--close-incident", "inc-a"]):
        with pytest.raises(SystemExit) as raised:
            cli.parser().parse_args(
                [
                    "workflow-reconcile",
                    "--state-dir",
                    str(state_dir),
                    "--close-quarantined",
                    *other,
                ]
            )
        assert raised.value.code == 2
