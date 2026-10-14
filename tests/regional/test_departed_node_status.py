"""``status`` reports what the departed-node cleanup did and what it left.

The control-plane probe ``status`` and the release verification already run
(``control_api_inspect``) carries the cleanup's ledger per cluster; ``status``
prints one summary line and one line per node still pending or left to the
operator, and the Agent coverage refusal names the cleanup's state for each
Agent that has no Kubernetes Node.
"""

from __future__ import annotations

import io
import json
import runpy
import urllib.request
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.departed_nodes import (
    DepartedNodeCleanupRecord,
    DepartedNodeCleanupStatus,
)
from gpu_fault_release import regional_admin_commands as ADMIN
from gpu_fault_release import regional_release_probes as PROBES
from gpu_fault_release.regional_admin_checks_coverage import (
    DEPARTED_NODE_LINE_LIMIT,
    check_agent_coverage,
    departed_node_lines,
)
from gpu_fault_release.regional_release_config import ReleaseError

NOW = datetime.now(timezone.utc)
RETIRE = (
    "gpu-fault-admin workflow-reconcile --state-dir <state-dir> "
    "--retire-departed-agents --node node-b --dry-run"
)


def _record(node_id: str, status: DepartedNodeCleanupStatus, **values: Any) -> Any:
    return DepartedNodeCleanupRecord(
        cluster_id="gpu-a",
        node_id=node_id,
        status=status,
        first_seen_at=NOW,
        updated_at=NOW,
        **values,
    )


def _ledger() -> list[dict[str, Any]]:
    return [
        _record(
            "node-a",
            DepartedNodeCleanupStatus.CLEANED,
            actions=["agent-retired:node-a", "incident-closed:incident-a"],
            cleaned_at=NOW,
        ).model_dump(mode="json"),
        _record(
            "node-b",
            DepartedNodeCleanupStatus.REFUSED,
            blockers=["workflow workflow-b (BLOCKED) is not closed: changed a node"],
            blocker_classes=["workflow_changed_node"],
            admin_commands=[RETIRE],
        ).model_dump(mode="json"),
        _record(
            "node-c",
            DepartedNodeCleanupStatus.PENDING,
            blockers=["agent heartbeat is younger than the 10-minute departure guard"],
            blocker_classes=["agent_alive"],
        ).model_dump(mode="json"),
    ]


class Response(io.BytesIO):
    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _run_probe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], store: Any
) -> dict[str, Any]:
    from gpu_fault.app import ApplicationContext

    def urlopen(request: urllib.request.Request, timeout: float) -> Response:
        path = request.full_url.removeprefix("http://127.0.0.1:8080")
        body: Any = {
            "/healthz": {"status": "ok"},
            "/v1/version": {"deployment_mode": "regional"},
            "/v1/regional/clusters": [],
        }.get(path)
        if body is None:
            body = [] if path.startswith("/v1/fleet/agents") else {"ready": True}
        return Response(json.dumps(body).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(
        ApplicationContext,
        "from_environment",
        staticmethod(lambda: SimpleNamespace(store=store)),
    )
    monkeypatch.setenv("GPU_FAULT_EXECUTION_TOKEN", "fixture-token")
    monkeypatch.setenv("GPU_FAULT_DEPARTED_NODE_CLEANUP_ENABLED", "true")
    monkeypatch.setenv("ADMIN_CLUSTER_IDS_JSON", '["gpu-a"]')
    monkeypatch.setenv("ADMIN_EXPECTED_NODES_JSON", '{"gpu-a": ["node-x"]}')
    runpy.run_path(
        str(PROBES.PROBE_DIR / "control_api_inspect.py"), run_name="__main__"
    )
    return dict(json.loads(capsys.readouterr().out))


def test_the_probe_carries_each_clusters_ledger_without_the_evidence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    records = [
        _record(
            "node-a",
            DepartedNodeCleanupStatus.CLEANED,
            evidence={"sources": {"hyperpod": {"listed": False}}},
        )
    ]
    store = SimpleNamespace(
        remote_command_stats=lambda: {"oldest_unclaimed_age_seconds": 0},
        list_departed_node_cleanups=lambda cluster_id: [
            item for item in records if item.cluster_id == cluster_id
        ],
    )

    document = _run_probe(monkeypatch, capsys, store)

    [line] = document["clusters"]["gpu-a"]["departed_node_cleanup"]
    assert line["node_id"] == "node-a" and line["status"] == "CLEANED"
    assert "evidence" not in line, "the evidence stays in the store"
    assert document["departed_node_cleanup_enabled"] is True


def test_an_older_control_plane_answers_none(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = SimpleNamespace(
        remote_command_stats=lambda: {"oldest_unclaimed_age_seconds": 0}
    )

    document = _run_probe(monkeypatch, capsys, store)

    assert document["clusters"]["gpu-a"]["departed_node_cleanup"] is None


def test_status_lists_cleaned_pending_and_refused_nodes_with_the_manual_path() -> None:
    lines = departed_node_lines(
        {"clusters": {"gpu-a": {"departed_node_cleanup": _ledger()}}}
    )

    assert lines == [
        "departed nodes: 1 cleaned, 1 pending, 1 left to the operator",
        "departed node gpu-a/node-b: REFUSED (workflow workflow-b (BLOCKED) is not "
        f"closed: changed a node); manual: {RETIRE}",
        "departed node gpu-a/node-c: PENDING (agent heartbeat is younger than the "
        "10-minute departure guard)",
    ]
    assert (
        departed_node_lines({"clusters": {"gpu-a": {"departed_node_cleanup": None}}})
        == []
    )


def test_status_says_when_the_cleanup_is_off_and_caps_the_node_lines() -> None:
    many = [
        _record(f"node-{index}", DepartedNodeCleanupStatus.PENDING).model_dump(
            mode="json"
        )
        for index in range(DEPARTED_NODE_LINE_LIMIT + 2)
    ]

    lines = departed_node_lines(
        {
            "departed_node_cleanup_enabled": False,
            "clusters": {"gpu-a": {"departed_node_cleanup": many}},
        }
    )

    assert lines[0].startswith("departed-node cleanup is off"), lines
    assert lines[-1].startswith("departed nodes: 2 more in health.checks"), lines
    assert len(lines) == DEPARTED_NODE_LINE_LIMIT + 3


def test_the_status_header_prints_the_ledger_from_the_control_api_check() -> None:
    report = {
        "healthy": True,
        "health_scope": "quick",
        "live_release": {"release_id": "release-a", "phase": "complete"},
        "next_deploy": {"kind": "NOOP"},
        "health": {
            "summary": {"PASS": 2, "WARN": 0, "FAIL": 0, "SKIP": 0},
            "checks": [
                {"name": "cpu_workloads", "status": "PASS"},
                {
                    "name": "control_api",
                    "status": "PASS",
                    "details": {
                        "clusters": {"gpu-a": {"departed_node_cleanup": _ledger()}}
                    },
                },
            ],
        },
    }

    lines = ADMIN.status_header_lines(report, json_destination="stdout")

    assert lines[3] == "failing checks: none"
    assert lines[4] == "departed nodes: 1 cleaned, 1 pending, 1 left to the operator"
    assert lines[-1] == "full JSON report: stdout"


def test_the_coverage_refusal_names_what_the_cleanup_says_about_each_node() -> None:
    cluster = {
        "expected_node_ids": ["node-x"],
        "kubernetes_node_ids": ["node-x"],
        "agents": [
            {"node_id": "node-x", "lifecycle_state": "ACTIVE"},
            {"node_id": "node-c", "lifecycle_state": "ACTIVE"},
            {"node_id": "node-d", "lifecycle_state": "ACTIVE"},
        ],
        "departed_node_cleanup": _ledger(),
    }

    with pytest.raises(ReleaseError) as refused:
        check_agent_coverage("gpu-a", cluster)

    message = str(refused.value)
    assert "Agents without a Kubernetes Node: ['node-c', 'node-d']" in message
    assert (
        "automatic departed-node cleanup: node-c PENDING (agent heartbeat is "
        "younger than the 10-minute departure guard), node-d not judged yet"
    ) in message
