"""Collector runner dispatch edges for the ordinary and destructive suites.

Pure verdict helpers with one more defect each (an unschedulable node, a
RESTART_APP decision that points at a workflow, an XID 63 marker resolving to
another XID), the destructive settings' optional site binding, the focused
test reuse path, the COLLECT-013 XID allowlist, a training manifest that is not
a mapping, and an execute_case whose case id owns no collector probe.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_collector_acceptance as ordinary
from scripts.e2e.regional import run_collector_destructive as destructive
from scripts.e2e.regional.regional_commands import RegionalFixtureError

FAR = datetime(2099, 1, 1, tzinfo=timezone.utc)
HYPERPOD = "hp-unit"
ROLE = "arn:aws:iam::123456789012:role/gpu-fault-executor-unit"


def destructive_settings(
    tmp_path: Path, *, case_id: str = "GF-REGIONAL-COLLECT-013", **overrides: Any
) -> destructive.Settings:
    values: dict[str, Any] = dict(
        regional=SimpleNamespace(
            environment=lambda: {"GPU_FAULT_REGIONAL_UNIT": "1"},
            namespace="gpu-fault-system",
        ),
        case_id=case_id,
        node="node-a",
        second_node=None,
        host_probe_image="img@sha256:" + "a" * 64,
        hyperpod_cluster=HYPERPOD,
        executor_role_arn=ROLE,
        site_file=None,
        predecessor_path=tmp_path / "predecessor.json",
    )
    values.update(overrides)
    return destructive.Settings(**values)


def test_node_isolation_errors_report_a_cordoned_node() -> None:
    node = {"ready": "True", "unschedulable": True, "taints": []}
    assert ordinary.node_isolation_errors(node, label="after restore") == [
        "node is unschedulable after restore"
    ]


def test_restart_app_monitor_only_rejects_a_decision_bound_to_a_workflow() -> None:
    decision = {
        "official_action": "RESTART_APP",
        "disposition": "MONITOR_ONLY",
        "action": "NO_ACTION",
        "reasons": ["no managed application to restart on an idle node"],
        "workflow_request_id": "workflow-idle",
    }
    errors = ordinary.restart_app_monitor_only_errors(
        [decision], [], [{"state": "RECOVERED"}]
    )
    assert errors == ["RESTART_APP decision points at a workflow on an idle node"]


def test_destructive_environment_binds_the_site_file_when_given(tmp_path: Path) -> None:
    site_file = tmp_path / "site.yaml"
    bound = destructive_settings(tmp_path, site_file=site_file).environment()
    assert bound["GPU_FAULT_SITE_FILE"] == str(site_file)
    assert bound["GPU_FAULT_COLLECT013_XID"] == str(destructive.DEFAULT_COLLECT013_XID)
    unbound = destructive_settings(
        tmp_path, case_id="GF-REGIONAL-COLLECT-008"
    ).environment()
    assert "GPU_FAULT_SITE_FILE" not in unbound
    assert "GPU_FAULT_COLLECT013_XID" not in unbound


def test_focused_tests_reuse_the_plan_result_on_the_same_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*_arguments: Any, **_keywords: Any) -> Any:
        raise AssertionError("a reusable plan result must not rerun pytest")

    monkeypatch.setattr(
        destructive, "RegionalLiveFixture", SimpleNamespace(run=forbidden)
    )
    monkeypatch.setattr(
        destructive,
        "reusable_focused_tests",
        lambda plan: {"passed": True, "returncode": 0, "command": ["pytest"]},
    )
    plan = tmp_path / "plan.json"
    result = destructive.focused_tests(
        "GF-REGIONAL-COLLECT-013", tmp_path, reuse_plan=plan
    )
    assert result == {
        "passed": True,
        "returncode": 0,
        "command": ["pytest"],
        "reused_from_plan": str(plan),
    }
    assert not (tmp_path / "focused-tests.log").exists(), "reuse writes no log"


def test_companion_xid_errors_require_the_marker_to_resolve_to_xid_63() -> None:
    solo = {
        "event": {"xid": 48, "evidence_ref": "kmsg://unit"},
        "decision": {"disposition": "MONITOR_ONLY"},
    }
    assert destructive.companion_xid_errors(solo) == [
        "XID 63 marker did not resolve to an XID 63 event"
    ]


def test_collect013_refuses_an_xid_outside_the_documented_pair(tmp_path: Path) -> None:
    settings = destructive_settings(tmp_path, xid=79)
    touched: list[str] = []
    host = SimpleNamespace(create=lambda: touched.append("host"))
    collector = SimpleNamespace(snapshot=lambda: touched.append("collector"))
    with pytest.raises(RegionalFixtureError, match="XID must be one of"):
        destructive.run_collect013(
            settings,
            SimpleNamespace(),  # type: ignore[arg-type]
            host,  # type: ignore[arg-type]
            tmp_path,
            1,
            collector=collector,  # type: ignore[arg-type]
            cleanup=destructive.CaseCleanup(),
        )
    assert touched == [], "an unsupported XID must never reach a probe"


def test_named_training_manifest_must_be_a_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "training.yaml"
    source.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    monkeypatch.setattr(destructive, "TRAINING_MANIFEST", source)
    destination = tmp_path / "rendered" / "training.yaml"
    with pytest.raises(RegionalFixtureError, match="not a mapping"):
        destructive.render_named_training_manifest(destination, name="unit-job")
    assert not destination.exists(), "a refused manifest is never rendered"


def test_execute_case_fails_closed_when_the_case_owns_no_collector_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        destructive,
        "read_only_preflight",
        lambda *a, **k: {"errors": [], "store": {}, "cpu_blast": {"nodes": 1}},
    )
    monkeypatch.setattr(
        destructive,
        "RegionalLiveFixture",
        lambda *a: SimpleNamespace(
            evidence_identity=lambda: {"release_id": "r", "cluster_id": "c"},
            cpu_blast_snapshot=lambda: {"nodes": 1},
        ),
    )
    hosts: list[str] = []
    monkeypatch.setattr(
        destructive,
        "reset_fixture",
        lambda *a, **k: SimpleNamespace(
            create=lambda: hosts.append("create"),
            cleanup=lambda: hosts.append("cleanup") or {},
        ),
    )
    probes: list[str] = []
    monkeypatch.setattr(
        destructive,
        "CollectorAcceptanceFixture",
        lambda *a, **k: probes.append("collector"),
    )
    settings = destructive_settings(tmp_path, case_id="GF-REGIONAL-COLLECT-999")

    assert destructive.execute_case(settings, tmp_path, 1, FAR) == 1
    result = json.loads(
        (tmp_path / "cases" / settings.case_id / f"{settings.case_id}.json").read_text()
    )
    assert result["verdict"] == "FAIL"
    assert result["error"] == "RegionalFixtureError: collector case has no owned probe"
    assert result["probe_residuals"] == {}
    assert result["release_id"] == "r"
    assert probes == [] and hosts == [], "an unknown case owns no probe at all"
    capsys.readouterr()
