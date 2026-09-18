from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin.atomic_json import write_json_atomic
from scripts.e2e.regional import auth016_lifecycle as lifecycle
from scripts.e2e.regional import credential_value_scan as credentials
from scripts.e2e.regional import identity_fleet_scope as fleet
from scripts.e2e.regional.blast_acceptance_base import CheckError
from scripts.e2e.regional.blast_rbac_scope import (
    BoundRule,
    bound_rules,
    unexpected_grants,
)
from tests.regional._cov95_identity_support import Clock
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional._regional_support import TOKEN_A
from tests.regional._security_consumer_pods import converged_deployment
from tests.regional._security_token_rotation_support import (
    RotationWorld,
    consumer_snapshot,
)
from tests.regional.test_cov95_identity_audit import bound_api, probe
from tests.regional.test_regional_auth_boundary_acceptance import fleet_agent
from tests.regional.test_security_alignment_rbac import SA, binding_document


@pytest.mark.parametrize(
    "identity",
    ["gpu-system:gpu-fault-cluster-executor", "system:user:gpu-system:executor"],
)
def test_rbac_requires_an_explicit_service_account(identity: str) -> None:
    with pytest.raises(CheckError, match="explicit ServiceAccount identity"):
        bound_rules(binding_document(), identity)


@pytest.mark.parametrize(
    ("items", "message"),
    [
        (None, "inventory is incomplete"),
        ([None], "invalid object"),
        ([{"kind": "Secret", "metadata": {"name": "foreign"}}], "unexpected resource"),
    ],
    ids=["missing-inventory", "nonobject", "wrong-resource-kind"],
)
def test_rbac_malformed_inventory_is_not_a_denial_proof(
    items: Any, message: str
) -> None:
    with pytest.raises(CheckError, match=message):
        bound_rules({"items": items}, SA)


@pytest.mark.parametrize("missing", ["subjects", "rules"])
def test_rbac_absent_subjects_or_rules_grant_nothing(missing: str) -> None:
    document = binding_document()
    document["items"][1 if missing == "subjects" else 0].pop(missing)
    assert bound_rules(document, SA) == [], (
        "a valid empty binding or role must not invent effective permissions"
    )


@pytest.mark.parametrize(
    ("index", "field", "value", "message"),
    [
        (1, "subjects", [None], "subject is invalid"),
        (0, "rules", {}, "bound role is absent or incomplete"),
        (0, "rules", [None], "bound rule is malformed"),
    ],
    ids=["nonobject-subject", "nonlist-rules", "nonobject-rule"],
)
def test_rbac_rejects_malformed_bound_subjects_and_rules(
    index: int, field: str, value: Any, message: str
) -> None:
    document = binding_document()
    document["items"][index][field] = value
    with pytest.raises(CheckError, match=message):
        bound_rules(document, SA)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("verbs", "get", "inventory is malformed"),
        ("resourceNames", [1], "inventory is malformed"),
        ("apiGroups", [], "resource rule has no API group"),
    ],
    ids=["nonlist-verbs", "nonstring-resource-name", "missing-api-group"],
)
def test_rbac_rule_shape_must_be_known_before_comparing_grants(
    field: str, value: Any, message: str
) -> None:
    document = binding_document()
    document["items"][0]["rules"][0][field] = value
    with pytest.raises(CheckError, match=message):
        unexpected_grants(
            bound_rules(document, SA), expected_cluster={}, expected_namespaces={}
        )


def test_discovery_and_self_reviews_do_not_hide_additional_writes() -> None:
    rules = [
        {"nonResourceURLs": ["/version"], "verbs": ["get", "head"]},
        {"nonResourceURLs": ["/version"], "verbs": ["post"]},
        {
            "apiGroups": ["authorization.k8s.io"],
            "resources": ["selfsubjectaccessreviews"],
            "verbs": ["create"],
        },
        {
            "apiGroups": ["discovery.k8s.io"],
            "resources": ["endpointslices"],
            "verbs": ["list", "patch"],
        },
    ]
    grants = [
        BoundRule(f"ClusterRoleBinding/audit-{index}", None, rule)
        for index, rule in enumerate(rules)
    ]
    assert unexpected_grants(grants, expected_cluster={}, expected_namespaces={}) == [
        {"binding": "ClusterRoleBinding/audit-1", "non_resource_write": True},
        {
            "binding": "ClusterRoleBinding/audit-3",
            "namespace": None,
            "api_group": "discovery.k8s.io",
            "resource": "endpointslices",
            "verb": "patch",
            "resource_names": [],
        },
    ], "read-only discovery and self-review exemptions must not authorize writes"


@pytest.mark.parametrize("limit", ["depth", "values"])
def test_credential_scan_enforces_real_structural_budgets(limit: str) -> None:
    raw = (
        b"[" * (credentials.MAX_DEPTH + 1)
        + b"null"
        + b"]" * (credentials.MAX_DEPTH + 1)
        if limit == "depth"
        else json.dumps([None] * credentials.MAX_VALUES).encode()
    )
    assert len(raw) < credentials.MAX_VALUE_BYTES, (
        "the fixture must reach the structural limit, not the byte-size limit"
    )
    with pytest.raises(credentials.CredentialScanError, match="structural limit"):
        credentials.credential_value_digests(raw, require_json=True)


def test_credential_scan_hashes_strings_but_not_json_scalar_coercions() -> None:
    raw = json.dumps(
        {"metadata": [42, False, None, 1.5], "hint": "fixture-only"}
    ).encode()
    expected = {
        hashlib.sha256(value).hexdigest()
        for value in (raw, b"metadata", b"hint", b"fixture-only")
    }
    assert credentials.credential_value_digests(raw, require_json=True) == expected, (
        "numeric, boolean and null leaves must not become invented credential bytes"
    )


@pytest.fixture
def fleet_window(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    context, bridge = bound_api(monkeypatch)
    fleet_agent(context, "a", "node-a")
    fleet_agent(context, "b", "node-b")
    for key, value in {
        "GPU_FAULT_CONTROL_PLANE_TOKEN": TOKEN_A,
        "GPU_FAULT_CLUSTER_ID": "a",
        "GPU_FAULT_CONTROL_PLANE_URL": "https://unit.invalid",
        "GPU_FAULT_CONTROL_PLANE_CA_FILE": str(tmp_path / "public-ca"),
    }.items():
        monkeypatch.setenv(key, value)
    site = SimpleNamespace(
        targets={"a": object(), "b": object()},
        api_pod_json=lambda script, *args: probe(script, *args),
        any_executor_pod=lambda _target: "executor-a",
        pod_json=lambda _plane, _target, _pod, script, *args, **kwargs: probe(
            script, *args
        ),
    )
    return SimpleNamespace(
        site=site,
        target=SimpleNamespace(cluster_id="a"),
        baseline={"a": ["node-a"], "b": ["node-b"]},
        context=context,
        bridge=bridge,
    )


@pytest.mark.parametrize(
    ("defect", "expected"),
    [
        ("cluster", "authenticated fleet request inventory is incomplete or foreign"),
        ("missing", "authenticated fleet request inventory is incomplete or foreign"),
        ("row", "peer-0-node: malformed response"),
        ("denial", "peer-0-node: wrong authenticated scope denial"),
    ],
)
def test_fleet_proof_rejects_malformed_or_wrongly_bound_responses(
    fleet_window: SimpleNamespace, defect: str, expected: str
) -> None:
    result = fleet.authenticated_fleet_isolation(fleet_window.site, fleet_window.target)
    assert result["passed"] is True, "the unmodified ASGI read proof must be valid"
    document = copy.deepcopy(result["results"])
    if defect == "cluster":
        document["cluster_id"] = "b"
    elif defect == "missing":
        document["results"].pop("peer-0-query")
    elif defect == "row":
        document["results"]["peer-0-node"] = None
    else:
        document["results"]["peer-0-node"]["detail"] = document["results"][
            "peer-0-query"
        ]["detail"]
    assert fleet.fleet_scope_errors(
        document, cluster_id="a", baseline=fleet_window.baseline
    ) == [expected], "a scope proof must preserve the exact request and denial binding"


def test_authentication_failure_cannot_stand_in_for_peer_scope_denial(
    fleet_window: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GPU_FAULT_CONTROL_PLANE_TOKEN", "unregistered-fixture")
    result = fleet.authenticated_fleet_isolation(fleet_window.site, fleet_window.target)
    assert result["passed"] is False, "an unauthenticated caller proves no isolation"
    assert result["results"]["results"]["peer-0-query"]["status"] == 403, (
        "the real auth boundary must reject the unregistered credential"
    )
    assert "peer-0-query: wrong authenticated scope denial" in result["errors"], (
        "the status alone must not substitute for authenticated scope denial"
    )
    assert fleet_window.context.store.list_remote_commands() == [], (
        "the fleet proof must not submit actions"
    )


@pytest.mark.parametrize("baseline", [None, {"a": ["node-a"]}])
def test_incomplete_cpu_baseline_stops_before_any_authenticated_request(
    fleet_window: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, baseline: Any
) -> None:
    monkeypatch.setattr(
        fleet_window.site, "api_pod_json", lambda *_args: {"clusters": baseline}
    )
    with pytest.raises(fleet.IdentityAcceptanceError, match="baseline is incomplete"):
        fleet.authenticated_fleet_isolation(fleet_window.site, fleet_window.target)
    assert fleet_window.bridge.requests == [], (
        "an incomplete cluster census cannot authorize the GPU-side read probe"
    )


def test_fleet_membership_change_invalidates_an_otherwise_valid_read_window(
    fleet_window: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = fleet_window.site.api_pod_json
    reads = 0

    def baseline(script: str, *args: str) -> dict[str, Any]:
        nonlocal reads
        reads += 1
        if reads == 2:
            fleet_agent(fleet_window.context, "b", "new-peer-node")
        return original(script, *args)

    monkeypatch.setattr(fleet_window.site, "api_pod_json", baseline)
    result = fleet.authenticated_fleet_isolation(fleet_window.site, fleet_window.target)
    assert result["passed"] is False, "fleet drift must invalidate the read proof"
    assert result["errors"] == [
        "fleet membership changed during the authenticated read window"
    ], "fresh CPU snapshots must bracket the real authenticated fleet requests"


@pytest.mark.parametrize(
    ("defect", "expected"),
    [
        ("naive", "rotation lifecycle timestamps are incomplete"),
        ("reversed", "rotation lifecycle timestamps are out of order"),
    ],
)
def test_rotation_journal_rejects_unordered_or_unzoned_completion_times(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str, expected: str
) -> None:
    world = RotationWorld(tmp_path, monkeypatch)
    result = lifecycle.run_rotation_acceptance(
        world.site, world.target, case_dir=tmp_path, retired_probe=lambda _token: 403
    )
    assert result["verdict"] == "PASS", "the original bound rotation must complete"
    state = copy.deepcopy(world.state)
    state["steps"][lifecycle.ROTATION_STEPS[0]]["completed_at"] = (
        world.stamp.replace(tzinfo=None).isoformat()
        if defect == "naive"
        else (world.stamp + timedelta(seconds=1)).isoformat()
    )
    assert lifecycle.rotation_journal_errors(
        state,
        reference=state["reference"],
        cluster_id="cluster-a",
        old_digest=state["old_token_sha256"],
        node_names={"node-a", "node-b"},
    ) == [expected], "timestamps must establish a complete ordered lifecycle"


def test_malformed_consumer_timestamp_does_not_prove_post_withdrawal_delivery() -> None:
    stamp = datetime(2026, 9, 15, tzinfo=timezone.utc)
    before = consumer_snapshot(activated=False, stamp=stamp)
    after = consumer_snapshot(activated=True, stamp=stamp)
    after["collectors"]["node-b/nvidia-kernel"]["last_success_at"] = None
    assert lifecycle.consumer_errors(
        before, after, cluster_id="cluster-a", after_withdrawal=stamp
    ) == ["credential consumer evidence is incomplete"], (
        "malformed delivery timestamps must not be treated as fresh evidence"
    )


def test_zero_replica_consumer_is_rejected_after_real_pod_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    world = RotationWorld(tmp_path, monkeypatch)

    def gpu(target: Any, *args: str, **kwargs: Any) -> str:
        document = json.loads(world.gpu(target, *args, **kwargs))
        if args[1] == "deployment":
            document["spec"]["replicas"] = 0
            document = converged_deployment(document)
        elif args[1] == "pods":
            document["items"] = []
        return json.dumps(document)

    def unexpected_probe(*args: Any, **kwargs: Any) -> None:
        pytest.fail("a zero-replica consumer must be refused before token probing")

    monkeypatch.setattr(world.site, "gpu", gpu)
    monkeypatch.setattr(world.site, "pod_json", unexpected_probe)
    with pytest.raises(lifecycle.IdentityAcceptanceError, match="not fully Ready"):
        lifecycle.pod_consumers(world.site, world.target)


def test_consumer_poll_retries_stale_evidence_then_returns_the_fresh_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stamp = datetime(2026, 9, 15, tzinfo=timezone.utc)
    before = consumer_snapshot(activated=False, stamp=stamp)
    after = consumer_snapshot(activated=True, stamp=stamp)
    snapshots = iter([before, after])
    clock = Clock()
    monkeypatch.setattr(lifecycle, "time", clock)
    observed, errors = lifecycle.wait_consumer_snapshot(
        lambda: next(snapshots), before, cluster_id="cluster-a", after_withdrawal=stamp
    )
    assert observed is after and errors == [], (
        "polling must return the newly observed consumer state, not its stale input"
    )
    assert clock.value == 7, "the stale read must trigger exactly one five-second wait"


def test_rotation_without_agent_baseline_stops_before_intent_and_sampler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    world = RotationWorld(tmp_path, monkeypatch)
    monkeypatch.setattr(world, "cpu_python", lambda *_args: {"agents": {}})
    with pytest.raises(
        lifecycle.IdentityAcceptanceError, match="complete Agent baseline"
    ):
        lifecycle.run_rotation_acceptance(
            world.site,
            world.target,
            case_dir=tmp_path,
            retired_probe=lambda _token: 403,
        )
    assert world.events == [] and world.thread is None, (
        "an empty fleet cannot start a rotation or its sampler"
    )
    assert not (tmp_path / "auth016-production-intent.json").exists(), (
        "a failed baseline must not reserve a production rotation intent"
    )


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("journal", "complete bound lifecycle"),
        ("token-file", "token file does not match"),
        ("surviving-pod", "old credential consumer Pod survived"),
    ],
)
def test_rotation_postcondition_failure_drains_sampler_and_keeps_fail_forward_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str, message: str
) -> None:
    world = RotationWorld(tmp_path, monkeypatch)
    old_resources = {
        (kind, "app=" + name): world.gpu(
            world.target, "get", kind, "-l", "app=" + name, "-o", "json"
        )
        for name in lifecycle.DEPLOYMENTS
        for kind in ("pods", "replicasets")
    }

    def gpu(target: Any, *args: str, **kwargs: Any) -> str:
        if (
            defect == "surviving-pod"
            and world.rotated
            and args[1] in {"pods", "replicasets"}
        ):
            return old_resources[args[1], args[args.index("-l") + 1]]
        return world.gpu(target, *args, **kwargs)

    def invoke(*args: Any) -> None:
        world.invoke(*args)
        if defect == "journal":
            world.state["steps"].pop(lifecycle.STEP_ACCEPTED)
            write_json_atomic(world.state_path, world.state)
        elif defect == "token-file":
            world.token_file.write_text(world.old)

    monkeypatch.setattr(world.site, "gpu", gpu)
    monkeypatch.setattr(lifecycle, "invoke_rotation", invoke)
    with pytest.raises(lifecycle.IdentityCaseFailure) as raised:
        lifecycle.run_rotation_acceptance(
            world.site,
            world.target,
            case_dir=tmp_path,
            retired_probe=lambda _token: 403,
        )
    assert isinstance(raised.value.__cause__, lifecycle.IdentityAcceptanceError), (
        "a specific failed postcondition must remain the acceptance failure cause"
    )
    assert message in str(raised.value.__cause__), (
        "the intended postcondition must fail"
    )
    result = raised.value.details
    assert (
        result["verdict"] == "FAIL" and result["failure"] == "IdentityAcceptanceError"
    ), "a completed CLI call cannot override unproven rotation postconditions"
    assert world.stop.is_set(), "the runner must stop sampling after failure"
    assert world.events[-1] == ("sampler-joined", lifecycle.SAMPLER_JOIN_SECONDS), (
        "failure must join the owned sampler within its existing bound"
    )
    assert result["cleanup_complete"] is (defect != "journal"), (
        "cleanup completeness must follow the journal proof, not suppress failure"
    )
    saved = json.loads((tmp_path / "auth016-details.json").read_text())
    assert saved == result, "the persisted failure must include sampler finalization"
    assert world.old not in json.dumps(saved) and world.new not in json.dumps(saved), (
        "rotation evidence must not disclose either fixture credential"
    )
    expected = world.old if defect == "token-file" else world.new
    assert (
        hashlib.sha256(world.token_file.read_bytes()).hexdigest()
        == hashlib.sha256(expected.encode()).hexdigest()
    ), "the runner must not restore or replace the committed credential"
