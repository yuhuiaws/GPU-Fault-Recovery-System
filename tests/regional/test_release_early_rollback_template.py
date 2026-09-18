from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_release_orchestration as ORCHESTRATION
from gpu_fault_release.regional_release_config import ReleaseError


@pytest.mark.parametrize("component", ["agent", "reconciler"])
def test_tampered_pending_template_blocks_all_rollback_mutation(
    monkeypatch: pytest.MonkeyPatch, component: str
) -> None:
    calls: list[str] = []
    release = SimpleNamespace(
        config=SimpleNamespace(clusters=(SimpleNamespace(cluster_id="gpu-a"),)),
        state={
            "execution_plan": {"nodes": [component, "verify"]},
            "component_progress": {
                "schema_version": 1,
                "global": {},
                "clusters": {"gpu-a": {component: {"status": "STARTED"}}},
            },
        },
        _refresh_aurora_credentials=lambda: calls.append("refresh"),
        _save_state=lambda *_args, **_kwargs: calls.append("state-write"),
    )

    def validate(_release, target, *, previous):
        calls.append(f"template:{target.cluster_id}")
        raise ReleaseError("rollback template identity is invalid")

    monkeypatch.setattr(ORCHESTRATION, "validate_rollback_node_template", validate)
    with pytest.raises(ReleaseError, match="template identity"):
        ORCHESTRATION.rollback_release(release, state={"metadata": {}})
    assert calls == ["template:gpu-a"], (
        "rollback changed CPU/refresher state before validating the pending old Job"
    )


def test_early_template_guard_is_scoped_to_pending_node_compensation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    release = SimpleNamespace(
        config=SimpleNamespace(
            clusters=tuple(
                SimpleNamespace(cluster_id=name)
                for name in ("done", "node", "executor")
            )
        ),
        state={
            "execution_plan": {"nodes": ["agent", "executor", "verify"]},
            "rollback_completed_cluster_ids": ["done"],
            "component_progress": {
                "schema_version": 1,
                "global": {},
                "clusters": {
                    "done": {"agent": {"status": "COMPLETED"}},
                    "node": {"agent": {"status": "STARTED"}},
                    "executor": {"executor": {"status": "STARTED"}},
                },
            },
        },
    )
    monkeypatch.setattr(
        ORCHESTRATION,
        "validate_rollback_node_template",
        lambda _release, target, **_kwargs: calls.append(target.cluster_id),
    )

    def reached_identity(*_args, **_kwargs):
        raise ReleaseError("identity boundary")

    monkeypatch.setattr(ORCHESTRATION, "_rollback_identity_context", reached_identity)
    with pytest.raises(ReleaseError, match="identity boundary"):
        ORCHESTRATION.rollback_release(release, state={"metadata": {}})
    assert calls == ["node"], (
        "completed or non-node compensation was unnecessarily probed"
    )
