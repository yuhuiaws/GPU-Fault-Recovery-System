from __future__ import annotations

import copy
import json
import subprocess
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault.admin import cluster_join_evidence as evidence
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from tests.admin.test_admin_site import site_file


class MembershipTransport:
    def __init__(self):
        self.calls = []
        self.release = {"data": {"state.json": json.dumps({"release_id": "release-a"})}}
        self.registry = {
            "generation": 2,
            "content_sha256": "c" * 64,
            "cluster_states": {"gpu-a": "ACTIVE", "gpu-b": "PENDING"},
            "required_member_ids": [],
            "acked_member_ids": [],
            "missing_member_ids": [],
            "active_member_ids": [],
            "members": [],
            "converged": True,
        }
        self.pod = "cpu-fixture"
        self.error = None
        self.after_query = lambda: None

    def __call__(self, arguments, **options):
        self.calls.append((arguments, options))
        kind = (
            "state"
            if "configmap" in arguments
            else "registry"
            if "exec" in arguments
            else "pod"
        )
        if self.error == kind:
            return subprocess.CompletedProcess(arguments, 1, "", "example-read-error")
        output = (
            json.dumps(self.release)
            if kind == "state"
            else json.dumps(self.registry)
            if kind == "registry"
            else self.pod
        )
        if kind == "registry":
            self.after_query()
        return subprocess.CompletedProcess(arguments, 0, output, "")


@pytest.fixture
def context(tmp_path, monkeypatch):
    source = site_file(tmp_path)
    current = load_site(source)
    document = yaml.safe_load(source.read_text())
    cluster = copy.deepcopy(document["spec"]["clusters"][0])
    token = tmp_path / "secure/token-b"
    token.write_text("b" * 64)
    token.chmod(0o600)
    cluster.update(
        clusterId="gpu-b",
        context="gpu-b",
        hyperpodClusterName="hp-gpu-b",
        eksClusterArn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-b",
        tokenFile=str(token),
    )
    document["spec"]["clusters"].append(cluster)
    path = tmp_path / "candidate.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    path.chmod(0o600)
    candidate = load_site(path)
    transport = MembershipTransport()
    monkeypatch.setattr(evidence, "run_command", transport)
    before = evidence.membership_runtime_snapshot(current)
    record = evidence.build_verified_membership_evidence(
        before,
        before,
        candidate_site_sha256=candidate.source_sha256,
        source_site_sha256=current.source_sha256,
        source_site_non_membership_sha256=evidence.site_non_membership_sha256(source),
        candidate_cluster_ids=["gpu-b", "gpu-a", "gpu-b"],
        cluster_id="gpu-b",
    )
    state = {
        key: record[key]
        for key in ("source_site_sha256", "source_site_non_membership_sha256")
    }
    transport.calls.clear()
    return current, candidate, record, state, transport


def validate(context, **kwargs):
    current, candidate, record, state, _transport = context
    evidence.validate_verified_membership(
        evidence=record,
        state=state,
        current_site=current,
        candidate_site=candidate,
        cluster_id="gpu-b",
        **kwargs,
    )


@pytest.mark.parametrize(
    "field",
    [
        "cluster_id",
        "candidate_site_sha256",
        "source_site_sha256",
        "source_site_non_membership_sha256",
        "candidate_cluster_ids",
    ],
)
def test_verification_rejects_each_binding_drift_before_runtime(context, field):
    record = context[2]
    record[field] = ["gpu-a"] if field == "candidate_cluster_ids" else "different"
    with pytest.raises(BootstrapError, match="drifted|changed|differs"):
        validate(context)
    assert context[4].calls == []


@pytest.mark.parametrize("document", ["not: [valid", "[]", "spec: []"])
def test_source_non_membership_hash_rejects_invalid_documents(tmp_path, document):
    path = tmp_path / "site.yaml"
    path.write_text(document)
    with pytest.raises(BootstrapError, match="identity|mapping"):
        evidence.site_non_membership_sha256(path)


def test_source_non_membership_hash_reports_missing_file(tmp_path):
    with pytest.raises(BootstrapError, match="cannot read site identity"):
        evidence.site_non_membership_sha256(tmp_path / "missing")


@pytest.mark.parametrize("timestamp", ["not-a-time", "2026-01-01T00:00:00", None])
def test_verification_rejects_invalid_or_naive_time(timestamp):
    assert evidence.verification_is_stale({"verified_at": timestamp}), (
        "invalid or naive verification time was accepted"
    )


@pytest.mark.parametrize(
    "offset,stale", [(-61, True), (-60, False), (900, False), (901, True)]
)
def test_verification_time_window_has_explicit_boundaries(offset, stale):
    now = datetime(2026, 9, 12, tzinfo=timezone.utc)
    record = {"verified_at": (now - timedelta(seconds=offset)).isoformat()}
    assert evidence.verification_is_stale(record, now) is stale


@pytest.mark.parametrize(
    "field", ["live_release_state_sha256", *evidence.RUNTIME_MEMBERSHIP_FIELDS]
)
def test_candidate_verification_compares_every_runtime_binding(context, field):
    current, candidate, _record, _state, _transport = context
    before = evidence.membership_runtime_snapshot(current)
    after = {**before, field: "different"}
    with pytest.raises(BootstrapError, match="drifted during"):
        evidence.build_verified_membership_evidence(
            before,
            after,
            candidate_site_sha256=candidate.source_sha256,
            source_site_sha256=current.source_sha256,
            source_site_non_membership_sha256="a" * 64,
            candidate_cluster_ids=["gpu-a", "gpu-b"],
            cluster_id="gpu-b",
        )


@pytest.mark.parametrize("states", [{"gpu-b": "ACTIVE"}, [], None])
def test_candidate_verification_requires_pending_target(context, states):
    current, candidate, _record, _state, _transport = context
    baseline = evidence.membership_runtime_snapshot(current)
    baseline["registry_cluster_states"] = states
    with pytest.raises(BootstrapError, match="PENDING|mapping"):
        evidence.build_verified_membership_evidence(
            baseline,
            baseline,
            candidate_site_sha256=candidate.source_sha256,
            source_site_sha256=current.source_sha256,
            source_site_non_membership_sha256="a" * 64,
            candidate_cluster_ids=["gpu-a", "gpu-b"],
            cluster_id="gpu-b",
        )


@pytest.mark.parametrize("stage", ["state", "pod", "registry"])
def test_membership_snapshot_transport_failure_stops_evidence(context, stage):
    current, _candidate, _record, _state, transport = context
    transport.error = stage
    with pytest.raises(BootstrapError, match="cannot read|no Running"):
        evidence.membership_runtime_snapshot(current)
    assert len(transport.calls) == {"state": 1, "pod": 2, "registry": 3}[stage]


@pytest.mark.parametrize(
    "document", [[], {"data": []}, {"data": {}}, {"data": {"state.json": "[]"}}]
)
def test_membership_snapshot_requires_complete_release_state(context, document):
    current, _candidate, _record, _state, transport = context
    transport.release = document
    with pytest.raises(BootstrapError, match="mapping|empty"):
        evidence.membership_runtime_snapshot(current)
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "document", [[], {}, {"converged": False}, {"missing_member_ids": ["unacked"]}]
)
def test_membership_snapshot_requires_valid_converged_registry(context, document):
    current, _candidate, _record, _state, transport = context
    if "converged" in document or "missing_member_ids" in document:
        transport.registry.update(document)
    else:
        transport.registry = document
    with pytest.raises(BootstrapError, match="malformed|not converged"):
        evidence.membership_runtime_snapshot(current)
    assert len(transport.calls) == 3


def test_membership_runtime_recheck_cannot_renew_expired_evidence(context, monkeypatch):
    _current, _candidate, record, _state, transport = context
    now = [datetime.fromisoformat(record["verified_at"])]
    monkeypatch.setattr(
        evidence,
        "datetime",
        SimpleNamespace(
            now=lambda _timezone: now[0], fromisoformat=datetime.fromisoformat
        ),
    )
    transport.after_query = lambda: now.__setitem__(0, now[0] + timedelta(seconds=901))
    with pytest.raises(evidence.JoinVerificationExpired, match="expired"):
        validate(context)
    assert len(transport.calls) == 3


@pytest.mark.parametrize("allow_expired", [False, True])
def test_only_irreversible_resume_may_explicitly_accept_expired_proof(
    context, allow_expired
):
    context[2]["verified_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=1000)
    ).isoformat()
    if allow_expired:
        validate(context, allow_expired=True)
        assert len(context[4].calls) == 3
    else:
        with pytest.raises(evidence.JoinVerificationExpired):
            validate(context)
        assert not context[4].calls, "expired reversible proof reached runtime reads"


@pytest.mark.parametrize(
    "change",
    ["baseline", "already-active", "states", "release", "generation", "content"],
)
def test_batch_proof_advancement_rejects_unrelated_changes(context, change):
    record = context[2]
    committed = copy.deepcopy(record)
    final = {
        **record,
        "registry_cluster_states": {"gpu-a": "ACTIVE", "gpu-b": "ACTIVE"},
        "registry_generation": 3,
        "registry_content_sha256": "d" * 64,
    }
    if change == "baseline":
        committed["registry_generation"] += 1
    elif change == "already-active":
        record["registry_cluster_states"]["gpu-b"] = "ACTIVE"
        committed = copy.deepcopy(record)
    else:
        key = {
            "states": "registry_cluster_states",
            "release": "live_release_identity_sha256",
            "generation": "registry_generation",
            "content": "registry_content_sha256",
        }[change]
        final[key] = (
            2 if change == "generation" else "" if change == "content" else "changed"
        )
    with pytest.raises(BootstrapError, match="baselines diverged|PENDING|drifted"):
        evidence.advance_batch_verification(
            record,
            committed_evidence=committed,
            final_identity=final,
            cluster_id="gpu-b",
        )


def test_batch_proof_advancement_preserves_verification_time(context):
    record = context[2]
    final = {
        **record,
        "registry_cluster_states": {"gpu-a": "ACTIVE", "gpu-b": "ACTIVE"},
        "registry_generation": 3,
        "registry_content_sha256": "d" * 64,
    }
    advanced = evidence.advance_batch_verification(
        record, committed_evidence=record, final_identity=final, cluster_id="gpu-b"
    )
    assert advanced["verified_at"] == record["verified_at"]
    assert advanced["verified_batch_activations"] == ["gpu-b"]
    assert advanced["post_verification_runtime"]["registry_generation"] == 3
    assert record["registry_cluster_states"]["gpu-b"] == "PENDING"


@pytest.mark.parametrize("kind", ["pending", "release-drift", "active"])
def test_final_membership_requires_active_target_and_same_release(context, kind):
    current, candidate, record, _state, transport = context
    if kind != "pending":
        transport.registry["cluster_states"]["gpu-b"] = "ACTIVE"
    if kind == "release-drift":
        record["live_release_identity_sha256"] = "0" * 64
    kwargs = dict(
        candidate_site_sha256=candidate.source_sha256,
        cluster_id="gpu-b",
        verified_at=record["verified_at"],
        verification_evidence=record,
    )
    if kind != "active":
        with pytest.raises(BootstrapError, match="not ACTIVE|release identity drifted"):
            evidence.final_membership_identity(current, **kwargs)
    else:
        result = evidence.final_membership_identity(current, **kwargs)
        assert result["registry_lifecycle"] == "ACTIVE"
        assert result["verified_at"] == record["verified_at"]


def test_clearing_verified_step_tolerates_missing_evidence(tmp_path):
    state = {"completed_steps": ["JOINED", "VERIFIED"], "evidence": []}
    path = tmp_path / "state.json"
    evidence.clear_verified_step(path, state)
    assert json.loads(path.read_text())["completed_steps"] == ["JOINED"]
