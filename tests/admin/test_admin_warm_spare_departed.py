"""``config spare --release`` for a spare the cluster no longer has.

The departed-node cleanup ends every record a departed node leaves in the
control plane except a spare's declaration: that lives on the deploy host, so
the cleanup reports it with this command. For a node a spot reclaim took the
command used to fail on the Node read; it now proves the node departed --
Kubernetes, the provider, the Agent, the rule ``departed_nodes`` holds -- and
then releases only the record (the label and the cordon left with the Node)
and tells the control plane, which answers the cleanup's finding.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import warm_spare_departed
from gpu_fault.admin import workflow_reconcile as reconcile
from gpu_fault.admin.warm_spare import (
    RELEASE_CONFIRMATION,
    AgentState,
    KubectlNodeApi,
    WarmSpareError,
    WarmSpareNodeMissing,
    WarmSpareRequest,
    perform_warm_spare,
    read_record,
    record_path,
)
from gpu_fault.departed_nodes import DepartedNodeSpareRelease

NODE = "hyperpod-i-0a000001"
INSTANCE = "i-0a000001"


class MissingNodeApi:
    """A cluster that has no Node by that name; nothing may be patched."""

    def read_node(self, name: str) -> Any:
        raise WarmSpareNodeMissing(f"kubectl get node {name} failed: NotFound")

    def patch_node(self, name: str, body: Any) -> Any:
        raise AssertionError("a departed spare has no Node to patch")

    def list_pod_for_all_namespaces(self) -> Any:
        return {"items": []}

    def list_node(self, label_selector: str) -> Any:
        return {"items": []}


def _declaration(tmp_path: Path, **overrides: Any) -> Path:
    """The record ``config spare --declare`` left for the node."""

    path = record_path(tmp_path, NODE)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "node": NODE,
        "cluster_id": "cluster-a",
        "declared_at": "2026-10-01T00:00:00+00:00",
        "reference": "CHG-0",
        "actor": "ops@example",
        "baseline": {"labels": {"gpu-fault.io/spare": None}, "unschedulable": False},
        "pre_declaration_survey": {
            "node": {
                "name": NODE,
                "uid": "uid-1",
                "provider_id": f"aws:///az/{INSTANCE}",
            }
        },
        "declared_state": {"labels": {"gpu-fault.io/spare": "true"}},
        **overrides,
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _request(tmp_path: Path) -> WarmSpareRequest:
    return WarmSpareRequest(
        state_dir=tmp_path,
        node=NODE,
        cluster_id="cluster-a",
        reference="CHG-1",
        mode="release",
        confirmation=RELEASE_CONFIRMATION,
    )


def _departed(
    proof: dict[str, Any], acknowledged: list[dict[str, Any]]
) -> warm_spare_departed.DepartedSpareChannel:
    def acknowledge(release: Any) -> dict[str, Any]:
        acknowledged.append(dict(release))
        return {"acknowledged": True}

    return warm_spare_departed.DepartedSpareChannel(
        lambda cluster_id, node, record, agent: proof, acknowledge
    )


def _proof(*blockers: str) -> dict[str, Any]:
    return {
        "proven": not blockers,
        "instance_id": INSTANCE,
        "blockers": list(blockers),
        "evidence": {"node_id": NODE, "absent_from_provider": not blockers},
    }


def _release(
    tmp_path: Path, departed: warm_spare_departed.DepartedSpareChannel | None
) -> dict[str, Any]:
    return perform_warm_spare(
        MissingNodeApi(),
        _request(tmp_path),
        cluster_id="cluster-a",
        record=record_path(tmp_path, NODE),
        agent_lookup=lambda cluster_id, node: AgentState(lifecycle_state="REVOKED"),
        actor="ops@example",
        provider="hyperpod",
        departed=departed,
    )


def test_a_departed_spares_record_is_released_without_a_node(tmp_path: Path) -> None:
    record = _declaration(tmp_path)
    acknowledged: list[dict[str, Any]] = []

    report = _release(tmp_path, _departed(_proof(), acknowledged))

    document = read_record(record)
    assert document is not None
    assert document["release_reference"] == "CHG-1"
    assert document["restored_state"] is None
    assert "nothing to restore" in document["release_skipped"]
    assert document["release_survey"]["departure"]["absent_from_provider"] is True
    assert report["ready"] is True and report["departed"] is True
    [release] = acknowledged
    assert DepartedNodeSpareRelease.model_validate(release).declaration == "released"
    assert release["instance_id"] == INSTANCE
    assert report["control_plane"] == {"acknowledged": True}


def test_a_departed_spare_without_a_record_is_still_acknowledged(
    tmp_path: Path,
) -> None:
    acknowledged: list[dict[str, Any]] = []

    report = _release(tmp_path, _departed(_proof(), acknowledged))

    assert not record_path(tmp_path, NODE).exists(), "nothing local to release"
    assert report["release"]["declaration"] == "none"
    [release] = acknowledged
    assert release["declaration"] == "none"


@pytest.mark.parametrize(
    "departed",
    [
        pytest.param(None, id="no-proof-channel"),
        pytest.param(
            _departed(_proof("HyperPod still lists instance i-0a000001"), []),
            id="provider-still-lists",
        ),
    ],
)
def test_an_unproven_departure_is_refused_and_writes_nothing(
    tmp_path: Path, departed: warm_spare_departed.DepartedSpareChannel | None
) -> None:
    record = _declaration(tmp_path)
    before = record.read_text(encoding="utf-8")

    with pytest.raises(WarmSpareError, match="node cannot be released"):
        _release(tmp_path, departed)

    assert record.read_text(encoding="utf-8") == before


def test_kubectl_not_found_is_a_missing_node(tmp_path: Path) -> None:
    def run(args: list[str], **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=f'Error from server (NotFound): nodes "{NODE}" not found',
        )

    api = KubectlNodeApi(["kubectl"], run=run)

    with pytest.raises(WarmSpareNodeMissing):
        api.read_node(NODE)


def _site() -> SimpleNamespace:
    return SimpleNamespace(
        release_config={
            "aws_region": "us-west-2",
            "clusters": [
                {
                    "cluster_id": "cluster-a",
                    "context": "ctx",
                    "hyperpod_cluster_name": "hp-a",
                    "region": "us-west-2",
                }
            ],
        },
        environment={},
    )


def _hyperpod_lists(monkeypatch: pytest.MonkeyPatch, *instances: str) -> None:
    def run(args: list[str], **_kwargs: Any) -> SimpleNamespace:
        assert "list-cluster-nodes" in args, args
        body = {"ClusterNodeSummaries": [{"InstanceId": item} for item in instances]}
        return SimpleNamespace(returncode=0, stdout=json.dumps(body), stderr="")

    monkeypatch.setattr(reconcile, "run_command", run)


def _stamp(**delta: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(**delta)).isoformat()


@pytest.mark.parametrize(
    ("agent", "listed", "blocker"),
    [
        pytest.param(AgentState(lifecycle_state=None), (), None, id="no-agent"),
        pytest.param(
            AgentState(
                lifecycle_state="ACTIVE",
                last_seen_at=_stamp(minutes=30),
                lease_expires_at=_stamp(minutes=28),
                evaluated_at=_stamp(seconds=0),
            ),
            (),
            None,
            id="silent-agent",
        ),
        pytest.param(
            AgentState(lifecycle_state="REVOKED"),
            (INSTANCE,),
            "HyperPod still lists instance i-0a000001",
            id="still-listed",
        ),
        pytest.param(
            AgentState(
                lifecycle_state="ACTIVE",
                last_seen_at=_stamp(minutes=1),
                lease_expires_at=_stamp(minutes=-1),
                evaluated_at=_stamp(seconds=0),
            ),
            (),
            "agent heartbeat is younger than the 10-minute departure guard",
            id="live-agent",
        ),
    ],
)
def test_the_site_proof_is_the_departed_node_rule(
    monkeypatch: pytest.MonkeyPatch,
    agent: AgentState,
    listed: tuple[str, ...],
    blocker: str | None,
) -> None:
    _hyperpod_lists(monkeypatch, *listed)

    proof = warm_spare_departed.departed_spare_proof(
        _site(), "cluster-a", NODE, None, agent
    )

    assert proof["proven"] is (blocker is None), proof["blockers"]
    if blocker is not None:
        assert any(blocker in item for item in proof["blockers"]), proof["blockers"]
    assert proof["evidence"]["sources"]["hyperpod"]["cluster_name"] == "hp-a"


def test_a_declaration_of_another_instance_is_no_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hyperpod_lists(monkeypatch)
    declared = {
        "pre_declaration_survey": {"node": {"provider_id": "aws:///az/i-0a00000f"}}
    }

    proof = warm_spare_departed.departed_spare_proof(
        _site(), "cluster-a", NODE, declared, AgentState(lifecycle_state=None)
    )

    assert proof["proven"] is False
    assert proof["blockers"] == [
        "the declaration names instance i-0a00000f, the node name i-0a000001"
    ]


def _run_in_pod(
    monkeypatch: pytest.MonkeyPatch, script: str, payload: dict[str, Any], store: Any
) -> dict[str, Any]:
    import io
    import sys

    from gpu_fault.app import ApplicationContext

    context = ApplicationContext(store=store)
    output = io.StringIO()
    with monkeypatch.context() as scoped:
        scoped.setattr(
            ApplicationContext, "from_environment", classmethod(lambda cls: context)
        )
        scoped.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        scoped.setattr(sys, "stdout", output)
        exec(compile(script, "<pod>", "exec"), {})
    return dict(json.loads(output.getvalue()))


def test_the_pod_stores_the_release_and_reports_the_agents_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gpu_fault.admin.warm_spare import AGENT_SCRIPT
    from tests._builders import build_store
    from tests.departed_nodes._support import agent

    store = build_store()
    store.save_agent(
        agent(
            "cluster-a",
            NODE,
            instance_id=None,
            last_seen=datetime.now(timezone.utc) - timedelta(minutes=30),
        )
    )
    release = {
        "cluster_id": "cluster-a",
        "node_id": NODE,
        "instance_id": INSTANCE,
        "change_reference": "CHG-1",
        "actor": "ops@example",
        "released_at": _stamp(seconds=0),
        "declaration": "released",
    }

    acknowledged = _run_in_pod(
        monkeypatch,
        warm_spare_departed.SPARE_RELEASE_SCRIPT,
        {"release": release},
        store,
    )
    facts = _run_in_pod(
        monkeypatch, AGENT_SCRIPT, {"cluster_id": "cluster-a", "node_id": NODE}, store
    )

    assert acknowledged == {"acknowledged": True}
    [stored] = store.list_departed_node_spare_releases("cluster-a")
    assert stored.instance_id == INSTANCE and stored.change_reference == "CHG-1"
    assert facts["found"] is True and facts["lifecycle_state"] == "ACTIVE"
    assert facts["last_seen_at"] and facts["lease_expires_at"] and facts["evaluated_at"]
