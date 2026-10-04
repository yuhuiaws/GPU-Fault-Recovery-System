"""Guards of the shared failure-domain publication, isolated from the transport.

``test_batch_join_failure_domains.py`` runs the real batch with an in-memory
kubectl. These tests pin the refusals that the batch only reaches when a member
is malformed or the fleet moves underneath it: an unprepared member, candidates
that disagree, membership drift that is (or is not) explained by an activation
already in flight, a publication without a worker identity, and a receipt that
is not bound to the join reading it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import cluster_join_failure_domains as barrier
from gpu_fault.admin import failure_domain_map as maps
from gpu_fault.admin.bootstrap_common import BootstrapError, ClusterIdentity
from gpu_fault.admin.cluster_join_types import (
    JoinAttempt,
    JoinClusterRequest,
    JoinExecution,
)
from gpu_fault.admin.site import RenderedSite, load_site
from tests.admin.test_admin_cluster_join_rollback import Attempt

CLUSTER = "hp-gpu-b"
GPU_B_ARN = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-b"
NODE_UIDS = {"node-b": "uid-node-b"}


def _identity() -> ClusterIdentity:
    return ClusterIdentity(
        input_arn=GPU_B_ARN,
        role="gpu",
        region="us-east-1",
        account_id="123456789012",
        hyperpod_arn="arn:aws:sagemaker:us-east-1:123456789012:cluster/hp-gpu-b",
        hyperpod_name=CLUSTER,
        eks_arn=GPU_B_ARN,
        eks_name="gpu-b",
        vpc_id="vpc-gpu-b",
        subnet_ids=("subnet-b",),
        node_recovery="None",
        context="gpu-fault-gpu-2-gpu-b",
    )


def _runtime(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "live_release_identity_sha256": "1" * 64,
        "registry_generation": 7,
        "registry_content_sha256": "2" * 64,
        "registry_cluster_states": {"gpu-a": "ACTIVE", CLUSTER: "PENDING"},
    }
    value.update(overrides)
    return value


class Publication:
    """The attempt under test plus every collaborator stubbed at its public seam."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.attempt = Attempt(tmp_path)
        self.candidate = load_site(self.attempt.candidate_path)
        self.before = _runtime()
        self.after = copy.deepcopy(self.before)
        self.worker_uid: str | None = "worker-uid"
        self.events: list[str] = []
        monkeypatch.setattr(
            barrier, "membership_runtime_snapshot", self._snapshot_membership
        )
        monkeypatch.setattr(
            barrier,
            "validate_verified_membership",
            lambda **_kw: self.events.append("validate"),
        )
        monkeypatch.setattr(maps, "build_failure_domain_map", self._build)
        monkeypatch.setattr(maps, "apply_failure_domain_map", self._apply)
        monkeypatch.setattr(maps, "verify_failure_domain_publication", self._verify)

    def _snapshot_membership(self, _site: RenderedSite) -> dict[str, Any]:
        self.events.append("snapshot")
        return copy.deepcopy(self.before if "apply" not in self.events else self.after)

    def _build(self, site: RenderedSite, **_kw: Any) -> maps.FailureDomainMapResult:
        self.events.append("build")
        return maps.FailureDomainMapResult(
            mapping={"gpu-a": {"node-a": "a"}, CLUSTER: {"node-b": "b"}},
            node_uids={"gpu-a": {"node-a": "uid-node-a"}, CLUSTER: dict(NODE_UIDS)},
        )

    def _apply(self, _site: RenderedSite, **kw: Any) -> maps.FailureDomainMapResult:
        self.events.append("apply")
        return kw["prepared"].__class__(
            mapping=kw["prepared"].mapping, worker_uid=self.worker_uid
        )

    def _verify(self, _site: RenderedSite, **kw: Any) -> dict[str, Any]:
        self.events.append("verify:" + ",".join(sorted(kw)))
        return {"worker_uid": kw["worker_uid"], "configmap_uid": "configmap-uid"}

    def member(
        self,
        suffix: str = "b",
        *,
        candidate: RenderedSite | None = None,
        expected: dict[str, Any] | None = None,
        completed: list[str] | None = None,
        execution: bool = True,
    ) -> JoinAttempt:
        site = candidate or self.candidate
        state_path = self.attempt.state_dir / f"state-{suffix}.json"
        verified = {
            **(expected or self.before),
            "batch_id": "batch-1",
            "candidate_cluster_ids": ["gpu-a", CLUSTER],
        }
        state = {
            "attempt": 1,
            "completed_steps": list(completed or []),
            "evidence": {"VERIFIED": verified},
        }
        state_path.write_text(json.dumps(state), encoding="utf-8")
        prepared = JoinExecution(
            target=_identity(),
            cluster_id=CLUSTER,
            discovery={},
            local={"node_uids": dict(NODE_UIDS)},
            prerequisites={},
            candidate=site,
        )
        return JoinAttempt(
            JoinClusterRequest(
                site=self.attempt.site,
                gpu_cluster_arn=GPU_B_ARN,
                state_dir=self.attempt.state_dir,
            ),
            self.attempt.state_dir,
            state_path,
            state,
            execution=prepared if execution else None,
        )


def test_an_empty_batch_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publication = Publication(tmp_path, monkeypatch)
    barrier.publish_batch_failure_domains([])
    assert publication.events == []


def test_a_member_without_a_prepared_join_cannot_be_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publication = Publication(tmp_path, monkeypatch)
    with pytest.raises(BootstrapError, match="lacks a prepared join"):
        barrier.publish_batch_failure_domains([publication.member(execution=False)])
    assert publication.events == []


def test_members_with_different_candidates_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publication = Publication(tmp_path, monkeypatch)
    members = [
        publication.member("b"),
        publication.member("c", candidate=publication.attempt.site),
    ]
    with pytest.raises(BootstrapError, match="candidates differ"):
        barrier.publish_batch_failure_domains(members)
    assert "build" not in publication.events, "no map may be built for a split batch"


def test_membership_drift_before_publication_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publication = Publication(tmp_path, monkeypatch)
    member = publication.member(expected=_runtime(registry_generation=6))
    with pytest.raises(BootstrapError, match="drifted before map publication"):
        barrier.publish_batch_failure_domains([member])
    assert "build" not in publication.events


def test_an_activation_already_started_explains_its_own_registry_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publication = Publication(tmp_path, monkeypatch)
    publication.before = _runtime(
        registry_generation=8,
        registry_cluster_states={"gpu-a": "ACTIVE", CLUSTER: "ACTIVE"},
    )
    publication.after = copy.deepcopy(publication.before)
    member = publication.member(
        expected=_runtime(registry_cluster_states={"gpu-a": "ACTIVE"}),
        completed=["ACTIVATION_STARTED"],
    )
    barrier.publish_batch_failure_domains([member])
    record = member.state["evidence"][barrier.PUBLICATION_READY]
    assert record["worker_uid"] == "worker-uid"
    assert record["node_uids"][CLUSTER] == NODE_UIDS
    assert record["attempt"] == 1
    assert publication.events.index("build") < publication.events.index("apply")


def test_a_started_activation_does_not_excuse_other_membership_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publication = Publication(tmp_path, monkeypatch)
    member = publication.member(
        expected=_runtime(live_release_identity_sha256="3" * 64),
        completed=["ACTIVATION_STARTED"],
    )
    with pytest.raises(BootstrapError, match="drifted before map publication"):
        barrier.publish_batch_failure_domains([member])


def test_a_publication_without_a_worker_identity_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publication = Publication(tmp_path, monkeypatch)
    publication.worker_uid = None
    member = publication.member()
    with pytest.raises(BootstrapError, match="control-worker identity"):
        barrier.publish_batch_failure_domains([member])
    assert barrier.PUBLICATION_STARTED in member.state["evidence"]
    assert barrier.PUBLICATION_READY not in member.state["evidence"]
    assert not any(event.startswith("verify") for event in publication.events), (
        "a publication without a worker must not be verified as converged"
    )


def test_a_verified_batch_member_publishes_before_checking_its_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publication = Publication(tmp_path, monkeypatch)
    member = publication.member()
    assert member.execution is not None, "the member must carry its execution"
    barrier.require_join_failure_domains(
        member.request,
        execution=member.execution,
        state_dir=member.state_dir,
        state_path=member.state_path,
        state=member.state,
    )
    assert publication.events.count("apply") == 1
    assert publication.events[-1] == "verify:configmap_uid,digest,worker_uid", (
        "the final check must pin the ConfigMap UID the publication recorded"
    )
    saved = json.loads(member.state_path.read_text(encoding="utf-8"))
    assert barrier.PUBLICATION_READY in saved["completed_steps"]


@pytest.mark.parametrize(
    "drift", ["attempt", "candidate_site_sha256", "cluster_ids", "node_uids", "shape"]
)
def test_a_receipt_not_bound_to_this_join_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    publication = Publication(tmp_path, monkeypatch)
    member = publication.member(completed=[barrier.PUBLICATION_READY])
    assert member.execution is not None, "the member must carry its execution"
    record: Any = {
        "attempt": 1,
        "candidate_site_sha256": publication.candidate.source_sha256,
        "cluster_ids": ["gpu-a", CLUSTER],
        "node_uids": {CLUSTER: dict(NODE_UIDS)},
        "map_sha256": "4" * 64,
        "worker_uid": "worker-uid",
        "configmap_uid": "configmap-uid",
    }
    if drift == "attempt":
        record["attempt"] = 2
    elif drift == "candidate_site_sha256":
        record["candidate_site_sha256"] = "0" * 64
    elif drift == "cluster_ids":
        record["cluster_ids"] = [CLUSTER]
    elif drift == "node_uids":
        record["node_uids"] = {CLUSTER: {"node-b": "replacement"}}
    else:
        record = "not a mapping"
    member.state["evidence"][barrier.PUBLICATION_READY] = record
    with pytest.raises(BootstrapError, match="not bound to this verified join"):
        barrier.require_join_failure_domains(
            member.request,
            execution=member.execution,
            state_dir=member.state_dir,
            state_path=member.state_path,
            state=member.state,
        )
    assert publication.events == [], "a bad receipt must trigger no publication"
