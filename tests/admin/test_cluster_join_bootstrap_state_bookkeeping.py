"""How a committed join is recorded in the developer's bootstrap-state.json.

After site.yaml is rewritten the join records its executor role, node keys and
Route53 association in the bootstrap state the next ``bootstrap`` run re-proves.
These tests run the real bookkeeping behind the public commit entry and pin its
refusals: an association in another Region or hosted zone, an ambiguous or
foreign-owned checkpoint, plus the lookup that picks the one matching state
file between the site directory and the operator's home.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import cluster_join as admin_cluster_join
from gpu_fault.admin import cluster_join_commit as commit
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_join import JoinClusterRequest, join_cluster
from gpu_fault.admin.resource_registry_dns import vpc_association_entry
from tests.admin.test_admin_cluster_join_rollback import GPU_B_ARN, Attempt
from tests.admin.test_cluster_join_commit_site_and_rollback import CLUSTER, Join

SITE_ID = "test-site"


class StopAfterBookkeeping(Exception):
    """Raised by the release-state stub, the step right after the bookkeeping."""


def _document(**resources: Any) -> dict[str, Any]:
    return {"site_id": SITE_ID, "resources": resources, "completed_tasks": []}


def _write(path: Path, document: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class Bookkeeping:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.join = Join(tmp_path, monkeypatch)
        self.home = tmp_path / "home"
        monkeypatch.setenv("HOME", str(self.home))
        self.local = self.join.site.source.parent / "bootstrap-state.json"
        monkeypatch.setattr(commit, "validate_verified_membership", lambda **_kw: None)
        monkeypatch.setattr(
            commit, "require_join_failure_domains", lambda *_a, **_kw: None
        )

        def stop(*_args: Any, **_kwargs: Any) -> None:
            raise StopAfterBookkeeping

        monkeypatch.setattr(admin_cluster_join, "_sync_join_release_state", stop)

    def home_state(self, name: str, document: dict[str, Any]) -> Path:
        return _write(
            self.home / ".gpu-fault/bootstrap" / name / "bootstrap-state.json", document
        )

    def pki(self, *associations: dict[str, Any]) -> dict[str, Any]:
        return {"hosted_zone_id": "Z123", "vpc_associations": list(associations)}

    def commit(self) -> None:
        with pytest.raises(StopAfterBookkeeping):
            commit.activate_and_commit(
                self.join.request(),
                execution=self.join.execution(),
                state_dir=self.join.state_dir,
                state_path=self.join.state_dir / "state.json",
                state={
                    "attempt": 1,
                    "completed_steps": [],
                    "evidence": {"VERIFIED": {"candidate_site_sha256": "c" * 64}},
                },
            )

    def read(self, path: Path | None = None) -> dict[str, Any]:
        return json.loads((path or self.local).read_text(encoding="utf-8"))


def _association(
    ownership: str = "CREATED", vpc_id: str = "vpc-gpu-b"
) -> dict[str, Any]:
    return vpc_association_entry(
        vpc_id=vpc_id,
        vpc_region="us-east-1",
        cluster_ids=["gpu-a"],
        ownership=ownership,
    )


def test_a_join_without_a_writer_role_records_only_its_executor_and_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    book = Bookkeeping(tmp_path, monkeypatch)
    book.join.prerequisites.pop("adot_writer_role")
    _write(book.local, _document(pki=book.pki()))
    book.commit()
    state = book.read()
    assert sorted(state["resources"]) == [
        f"executor_role:{CLUSTER}",
        "nlb_network",
        f"node_keys:{CLUSTER}",
        "pki",
    ]
    assert state["completed_tasks"] == [
        f"executor_role:{CLUSTER}",
        f"node_keys:{CLUSTER}",
    ]
    (association,) = state["resources"]["pki"]["vpc_associations"]
    assert (
        association["vpc_id"] == "vpc-gpu-b" and association["ownership"] == "CREATED"
    )
    assert CLUSTER in state["joined_clusters"]


def test_an_association_already_checkpointed_as_created_is_not_duplicated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    book = Bookkeeping(tmp_path, monkeypatch)
    _write(book.local, _document(pki=book.pki(_association())))
    book.commit()
    state = book.read()
    assert len(state["resources"]["pki"]["vpc_associations"]) == 1
    assert f"adot_writer_role:{CLUSTER}" in state["completed_tasks"], (
        "the data-plane writer role must be recorded next to the executor role"
    )


@pytest.mark.parametrize(
    "network, pki, message",
    [
        ({"vpc_region": "eu-west-1"}, (), "Region differs from site"),
        ({"hosted_zone_id": "Z999"}, (), "hosted zone differs"),
        ({}, (_association(), _association()), "checkpoint is ambiguous"),
        ({}, (_association("EXTERNAL"),), "ownership conflicts"),
    ],
)
def test_association_checkpoints_that_do_not_fit_the_site_are_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    network: dict[str, Any],
    pki: tuple[dict[str, Any], ...],
    message: str,
) -> None:
    book = Bookkeeping(tmp_path, monkeypatch)
    book.join.network.update(network)
    document = _document(pki=book.pki(*pki))
    _write(book.local, document)
    with pytest.raises(BootstrapError, match=message):
        book.commit()
    assert book.read() == document, (
        "a refused checkpoint must leave the state as it was"
    )


def test_the_bootstrap_state_is_found_in_the_operator_home_when_the_site_has_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    book = Bookkeeping(tmp_path, monkeypatch)
    remote = book.home_state("site", _document(pki=book.pki()))
    book.commit()
    assert f"executor_role:{CLUSTER}" in book.read(remote)["resources"]
    assert not book.local.exists(), "the join must not create a second state file"


def test_a_home_state_of_another_site_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    book = Bookkeeping(tmp_path, monkeypatch)
    foreign = {**_document(), "site_id": "another-site"}
    other = book.home_state("other", foreign)
    _write(book.local, _document(pki=book.pki()))
    book.commit()
    assert book.read(other) == foreign
    assert f"node_keys:{CLUSTER}" in book.read()["resources"]


def test_a_join_preparation_without_a_candidate_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    attempt.install(monkeypatch)
    monkeypatch.setattr(
        admin_cluster_join, "_prepare_execution", lambda *_a, **_k: None
    )
    with pytest.raises(BootstrapError, match="did not produce a candidate"):
        join_cluster(
            JoinClusterRequest(
                site=attempt.site,
                gpu_cluster_arn=GPU_B_ARN,
                state_dir=attempt.state_dir,
            ),
            runner=SimpleNamespace(dry_run=False),
        )
    assert attempt.rollouts == [], "nothing may be rolled out without a candidate"
