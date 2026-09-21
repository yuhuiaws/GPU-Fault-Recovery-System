from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.telemetry import CollectorKind, CollectorStatus
from scripts.e2e.regional import audit_auth_boundary as audit
from scripts.e2e.regional import identity_fleet_scope as fleet
from tests.regional.test_cov95_identity_audit import bound_api, matrix_arguments, probe
from tests.regional.test_regional_auth_boundary_acceptance import fleet_agent


def test_stateful_live_probe_ignores_other_collectors_but_sees_its_own_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    context, _ = bound_api(monkeypatch)
    args = audit.parser().parse_args(matrix_arguments(tmp_path))
    args.probe_id = "auth-probe-" + "a" * 32
    now = datetime.now(timezone.utc)
    before = probe(audit.STORE_NEGATIVE_PROBE, "a", "b", args.probe_id)
    audit.validate_matrix(audit.run_matrix(args), cluster_a="a")
    context.store.save_collector_status(
        CollectorStatus(
            cluster_id="b",
            node_id="real-node",
            collector=CollectorKind.NVIDIA_KERNEL,
            observed_at=now,
            ingested_at=now,
            last_success_at=now,
            sample_count=20,
        )
    )
    after = probe(audit.STORE_NEGATIVE_PROBE, "a", "b", args.probe_id)
    assert after == before, "routine traffic belongs outside the denial probe's scope"
    context.store.save_collector_status(
        CollectorStatus(
            cluster_id="b",
            node_id=args.probe_id,
            collector=CollectorKind.NVIDIA_KERNEL,
            observed_at=now,
            ingested_at=now,
            last_success_at=now,
            sample_count=1,
        )
    )
    poisoned = probe(audit.STORE_NEGATIVE_PROBE, "a", "b", args.probe_id)
    assert "GF-REGIONAL-AUTH-009" in audit.store_negative_errors(
        before, poisoned, cluster_a="a", cluster_b="b"
    )


def test_alias_and_nested_distributed_payloads_reach_the_actual_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    context, bridge = bound_api(monkeypatch)
    args = audit.parser().parse_args(matrix_arguments(tmp_path))
    results = audit.run_matrix(args)
    for name in audit.AUTH009_SCOPE_VARIANTS:
        assert results["AUTH-009 " + name] == {
            "status": 403,
            "body": {"detail": audit.PAYLOAD_BINDING_DENIAL},
        }
    requests = [
        request
        for request in bridge.requests
        if request.full_url.endswith(audit.AUTH009_VARIANT_PATH)
    ]
    assert len(requests) == len(audit.AUTH009_SCOPE_VARIANTS) + 1
    assert context.store.list_raw_evidence("b") == []
    assert context.store.list_workflows() == []


def test_authenticated_fleet_runner_reads_local_records_and_denies_real_peer_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    context, _ = bound_api(monkeypatch)
    fleet_agent(context, "a", "node-a")
    fleet_agent(context, "b", "node-b")
    from tests.regional._regional_support import TOKEN_A

    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_TOKEN", TOKEN_A)
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "a")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_URL", "https://unit.invalid")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_CA_FILE", str(tmp_path / "public-ca"))
    site = SimpleNamespace(
        targets={"a": object(), "b": object()},
        api_pod_json=lambda script, *args: probe(script, *args),
        any_executor_pod=lambda _target: "executor-a",
        pod_json=lambda _plane, _target, _pod, script, *args, **kwargs: probe(
            script, *args
        ),
    )
    result = fleet.authenticated_fleet_isolation(site, SimpleNamespace(cluster_id="a"))
    assert result["passed"] is True
    assert result["errors"] == []
    assert TOKEN_A not in json.dumps(result)
    baseline = {"a": ["node-a"], "b": ["node-b"]}
    corrupted = copy.deepcopy(result["results"])
    corrupted["results"]["local-list"]["agents"].append(
        {"cluster_id": "b", "node_id": "node-b"}
    )
    assert fleet.fleet_scope_errors(corrupted, cluster_id="a", baseline=baseline), (
        "fleet isolation proof must reject peer records in the local list"
    )


@pytest.mark.parametrize(
    "baseline", [{}, {"a": []}, {"a": ["node-a"]}, {"a": ["node-a"], "b": []}]
)
def test_fleet_scope_rejects_vacuous_baselines(baseline):
    with pytest.raises(fleet.IdentityAcceptanceError):
        fleet.fleet_scope_requests("a", baseline)


def test_single_cluster_site_proves_fleet_isolation_against_an_unregistered_peer(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    # Live 2026-09-20 (AUTH-014 a1): the site has ONE registered GPU cluster, so
    # ``site.targets`` yields a one-entry baseline and the fleet proof raised
    # "fleet isolation needs populated local and peer baselines". The API denies
    # a foreign fleet read by comparing the requested cluster id with the
    # authenticated header before any store lookup (``routes/fleet.py``), so an
    # UNREGISTERED peer id -- the ISO-003/004 recipe -- is a faithful probe of the
    # same rule; only the local listing needs real records.
    context, _ = bound_api(monkeypatch)
    fleet_agent(context, "a", "node-a")
    from tests.regional._regional_support import TOKEN_A

    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_TOKEN", TOKEN_A)
    monkeypatch.setenv("GPU_FAULT_CLUSTER_ID", "a")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_URL", "https://unit.invalid")
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_CA_FILE", str(tmp_path / "public-ca"))
    site = SimpleNamespace(
        targets={"a": object()},
        api_pod_json=lambda script, *args: probe(script, *args),
        any_executor_pod=lambda _target: "executor-a",
        pod_json=lambda _plane, _target, _pod, script, *args, **kwargs: probe(
            script, *args
        ),
    )
    with pytest.raises(fleet.IdentityAcceptanceError, match="peer"):
        fleet.authenticated_fleet_isolation(site, SimpleNamespace(cluster_id="a"))
    result = fleet.authenticated_fleet_isolation(
        site,
        SimpleNamespace(cluster_id="a"),
        peer=SimpleNamespace(cluster_id="b", registered=False),
    )
    assert result["passed"] is True, result
    assert result["errors"] == []
    assert result["unregistered_peer"] == "b"
    rows = result["results"]["results"]
    assert {name for name in rows if name.startswith("peer-")} == {
        "peer-0-query",
        "peer-0-node",
    }
    assert all(
        rows[name]["status"] == 403 for name in rows if name.startswith("peer-")
    ), "an unregistered peer must be denied on every fleet route"
    assert rows["local-list"]["agents"] == [{"cluster_id": "a", "node_id": "node-a"}]
    assert context.store.list_agents("b") == [], "the peer must stay unregistered"
    with pytest.raises(fleet.IdentityAcceptanceError):
        fleet.authenticated_fleet_isolation(
            site,
            SimpleNamespace(cluster_id="a"),
            peer=SimpleNamespace(cluster_id="a", registered=False),
        )
