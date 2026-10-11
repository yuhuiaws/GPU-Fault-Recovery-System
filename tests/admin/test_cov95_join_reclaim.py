"""``join-cluster --reclaim-cluster-token``: re-attach a cluster the site lost.

The incident this covers: an automatic rollback rebuilt the management site
from a stale baseline and de-registered a joined cluster whose token file, data
plane, node Agents and IAM roles all kept running. Without the flag the join
still refuses the orphaned token; with it, the token is adopted only when the
committed join journal, the cluster's data plane and the control plane all
prove it belongs to this cluster, and the proof is journaled without the token
material.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin import cli_join
from gpu_fault.admin import cluster_join as join
from gpu_fault.admin import cluster_join_inputs as inputs
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_join_inputs import prepare_local_inputs
from gpu_fault.admin.cluster_join_reclaim import ClusterTokenReclaimRefused
from gpu_fault.admin.cluster_join_state import load_join_state
from gpu_fault.admin.site import SiteConfigError
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceSnapshot,
    InstallationResourceStatus,
)
from tests.admin._cov95_join_support import JoinScenario, target

CLUSTER_ID = "hp-gpu-b"
TOKEN = "t" * 64
TOKEN_SHA256 = hashlib.sha256(TOKEN.encode()).hexdigest()
NAMESPACE_UID = "namespace-committed-b"


def _secret(data: dict[str, str]) -> dict:
    return {
        "kind": "Secret",
        "data": {
            key: base64.b64encode(value.encode()).decode()
            for key, value in data.items()
        },
    }


def _connection_secret(token: str = TOKEN, cluster_id: str = CLUSTER_ID) -> dict:
    return _secret({"cluster-token": token, "cluster-id": cluster_id})


def _registry_secret(rows: list[dict], key: str = "clusters.json") -> dict:
    return _secret({key: json.dumps(rows)})


def _registry_row(**changes) -> dict:
    return {
        "cluster_id": CLUSTER_ID,
        "eks_cluster_arn": target().eks_arn,
        "token": TOKEN,
        "lifecycle_state": "ACTIVE",
        **changes,
    }


def _iam_row(site_id: str, cluster_id: str = CLUSTER_ID) -> InstallationResource:
    return InstallationResource(
        site_id=site_id,
        resource_key=f"aws/iam/executor/{cluster_id}/role",
        resource_type="iam_role",
        resource_id="executor",
        resource_arn="arn:aws:iam::123456789012:role/executor-b",
        region="us-east-1",
        account_id="123456789012",
        ownership=InstallationResourceOwnership.CREATED,
        delete_policy=InstallationResourceDeletePolicy.DELETE,
        status=InstallationResourceStatus.ACTIVE,
    )


def _archived_committed_journal(directory: Path, token_file: Path, **changes) -> Path:
    """What a committed join leaves behind once a later attempt archives it."""

    recorded_target = asdict(target())
    recorded_target.update(changes.pop("target", {}))
    journal = {
        "schema_version": 1,
        "attempt": 1,
        "phase": "COMPLETED",
        "completed_steps": [
            "DISCOVERED",
            "LOCAL_INPUTS_STARTED",
            "LOCAL_INPUTS_READY",
            "JOINED",
            "VERIFIED",
            "SITE_UPDATED",
            "RELEASE_STATE_UPDATED",
            "ACTIVATED",
            "FINAL_VERIFIED",
        ],
        "step_completed_at": {"SITE_UPDATED": "2000-01-01T00:00:00+00:00"},
        "evidence": {
            "DISCOVERED": {"cluster_id": CLUSTER_ID, "target": recorded_target},
            "LOCAL_INPUTS_READY": {
                "token_file": str(token_file),
                "namespace_uid": NAMESPACE_UID,
            },
            "FINAL_VERIFIED": {
                "cluster_id": CLUSTER_ID,
                "committed_at": "2000-01-01T00:00:10+00:00",
                "registry_generation": 3,
                "live_release_state_sha256": "a" * 64,
            },
        },
        **changes,
    }
    path = directory / "state.attempt-001.json"
    path.write_text(json.dumps(journal), encoding="utf-8")
    return path


class Orphan:
    """An orphaned token: the file, the committed journal archive, the data
    plane Secret and the control-plane registry rows, each removable."""

    def __init__(self, tmp_path, monkeypatch, *, reclaim=True):
        self.scenario = JoinScenario(tmp_path, monkeypatch)
        self.request = self.scenario.request(reclaim_cluster_token=reclaim)
        self.directory, self.path, self.state = load_join_state(self.request)
        self.token_file = (
            self.request.site.source.parent / "secure" / f"{CLUSTER_ID}.token"
        )
        self.token_file.write_text(TOKEN, encoding="utf-8")
        self.token_file.chmod(0o600)
        self.journal = _archived_committed_journal(self.directory, self.token_file)
        self.scenario.gpu_kubeconfig.touch(mode=0o600)
        self.scenario.contexts.add(target().context)
        self.scenario.namespace_uids[target().context] = NAMESPACE_UID
        self.scenario.connection_secrets[target().context] = _connection_secret()
        self.scenario.registry_secret = _registry_secret([_registry_row()])
        # The alias check needs a real kubeconfig; the fake writes a stub, so
        # the public reclaim-released-alias seam answers for it here.
        monkeypatch.setattr(
            inputs,
            "reclaim_released_alias",
            lambda *_args, **_kwargs: {"released_by": {"registry": "absent"}},
        )

    def prepare(self):
        prepare_local_inputs(
            self.request,
            runner=self.scenario,
            target=target(),
            cluster_id=CLUSTER_ID,
            state_dir=self.directory,
            state_path=self.path,
            state=self.state,
        )

    def journaled(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))


@pytest.fixture
def orphan(tmp_path, monkeypatch):
    return Orphan(tmp_path, monkeypatch)


def test_reclaim_adopts_the_token_on_all_three_proofs_and_journals_them(orphan):
    before = orphan.token_file.read_bytes()

    orphan.prepare()

    state = orphan.journaled()
    assert "LOCAL_INPUTS_READY" in state["completed_steps"]
    assert orphan.token_file.read_bytes() == before, "the existing token is kept"
    assert state["retained_cluster_token"] == {
        "path": str(orphan.token_file),
        "cluster_id": CLUSTER_ID,
        "eks_arn": target().eks_arn,
        "hyperpod_arn": target().hyperpod_arn,
    }, "the binding the ordinary retry path honours is recorded"
    record = state["reclaimed_cluster_token"]
    assert record["cluster_id"] == CLUSTER_ID
    assert record["token_sha256"] == TOKEN_SHA256
    assert record["identity"]["source"] == "join-journal"
    assert record["identity"]["journal"] == str(orphan.journal)
    assert record["identity"]["committed_at"] == "2000-01-01T00:00:10+00:00"
    assert record["identity"]["registry_generation"] == 3
    assert record["data_plane"]["token_sha256"] == TOKEN_SHA256
    assert record["data_plane"]["context"] == target().context
    assert record["data_plane"]["cluster_id"] == CLUSTER_ID
    assert record["control_plane"]["sources"] == ["gpu-fault-regional-clusters"]
    assert record["control_plane"]["regional_registry"]["rows"] == [
        {
            "key": "clusters.json",
            "lifecycle_state": "ACTIVE",
            "token_sha256_matches": True,
        }
    ]
    for timestamp in (
        record["reclaimed_at"],
        record["data_plane"]["observed_at"],
        record["control_plane"]["observed_at"],
    ):
        assert "T" in timestamp and "+00:00" in timestamp
    assert TOKEN not in json.dumps(state), "the journal never carries token material"
    local = state["evidence"]["LOCAL_INPUTS_READY"]
    assert local["namespace_uid"] == NAMESPACE_UID, (
        "the committed namespace is accepted under its recorded UID"
    )
    assert local["reclaimed_cluster_token"] is True


def test_reclaim_accepts_the_installation_registry_as_control_plane_proof(orphan):
    orphan.scenario.registry_secret = None
    orphan.scenario.registry.live = InstallationResourceSnapshot(
        site_id=orphan.request.site.registry_site_id,
        resources=[_iam_row(orphan.request.site.registry_site_id)],
    )

    orphan.prepare()

    record = orphan.journaled()["reclaimed_cluster_token"]
    assert record["control_plane"]["sources"] == ["installation-registry"]
    assert record["control_plane"]["installation_registry"]["active_keys"] == [
        f"aws/iam/executor/{CLUSTER_ID}/role"
    ]
    assert Path(
        record["control_plane"]["installation_registry"]["snapshot"]
    ).is_file(), (
        "the registry snapshot the proof read is kept in the join state directory"
    )


def test_without_the_flag_the_orphaned_token_is_still_refused(tmp_path, monkeypatch):
    orphan = Orphan(tmp_path, monkeypatch, reclaim=False)

    with pytest.raises(
        join.JoinTargetIdentityError, match="no matching join ownership"
    ):
        orphan.prepare()

    assert orphan.scenario.commands == [], "the refusal runs no command"
    assert "retained_cluster_token" not in orphan.journaled()
    assert "reclaimed_cluster_token" not in orphan.journaled()


def test_the_refusal_without_the_flag_names_the_flag(tmp_path, monkeypatch):
    orphan = Orphan(tmp_path, monkeypatch, reclaim=False)
    with pytest.raises(join.JoinTargetIdentityError) as failure:
        orphan.prepare()
    assert "--reclaim-cluster-token" in str(failure.value)


def test_the_flag_without_an_orphaned_token_is_refused(orphan):
    orphan.token_file.unlink()

    with pytest.raises(
        join.JoinTargetIdentityError, match="no cluster token to reclaim"
    ):
        orphan.prepare()

    assert not orphan.token_file.exists(), "nothing is generated"


def test_the_flag_is_harmless_once_the_journal_owns_the_token(orphan):
    """A retry after a successful reclaim carries the same command line; the
    journal already owns the token, so nothing is proven again."""

    orphan.scenario.namespace_uids.pop(target().context)
    orphan.state["retained_cluster_token"] = {
        "path": str(orphan.token_file),
        "cluster_id": CLUSTER_ID,
        "eks_arn": target().eks_arn,
        "hyperpod_arn": target().hyperpod_arn,
    }

    orphan.prepare()

    state = orphan.journaled()
    assert "LOCAL_INPUTS_READY" in state["completed_steps"]
    assert "reclaimed_cluster_token" not in state, "nothing was reclaimed"


def _refuse(orphan, pattern):
    before = copy.deepcopy(orphan.journaled())
    with pytest.raises(ClusterTokenReclaimRefused, match=pattern) as failure:
        orphan.prepare()
    message = str(failure.value)
    assert message.startswith(
        f"join-cluster --reclaim-cluster-token refused for {CLUSTER_ID}"
    ), "every refusal names the flag and the cluster"
    assert TOKEN not in message
    assert orphan.journaled() == before, (
        "a refused reclaim leaves the journal untouched"
    )
    assert "retained_cluster_token" not in orphan.journaled()
    assert "LOCAL_INPUTS_STARTED" not in orphan.journaled()["completed_steps"]
    return message


def test_reclaim_refuses_without_a_committed_join_journal(orphan):
    orphan.journal.unlink()
    _refuse(orphan, "no committed join journal")


@pytest.mark.parametrize(
    "change",
    ["other-eks", "other-hyperpod", "other-token-file", "other-cluster", "uncommitted"],
)
def test_reclaim_refuses_a_journal_that_does_not_bind_this_identity(orphan, change):
    journal = json.loads(orphan.journal.read_text(encoding="utf-8"))
    if change == "other-eks":
        journal["evidence"]["DISCOVERED"]["target"]["eks_arn"] = (
            "arn:aws:eks:x:1:cluster/z"
        )
    elif change == "other-hyperpod":
        journal["evidence"]["DISCOVERED"]["target"]["hyperpod_arn"] = "arn:other"
    elif change == "other-token-file":
        journal["evidence"]["LOCAL_INPUTS_READY"]["token_file"] = "/elsewhere/x.token"
    elif change == "other-cluster":
        journal["evidence"]["DISCOVERED"]["cluster_id"] = "hp-gpu-z"
    else:
        journal["phase"] = "ROLLED_BACK"
        journal["completed_steps"] = ["DISCOVERED", "LOCAL_INPUTS_READY", "JOINED"]
    orphan.journal.write_text(json.dumps(journal), encoding="utf-8")
    _refuse(orphan, "no committed join journal")


def test_reclaim_refuses_a_symlinked_or_short_token_file(orphan):
    real = orphan.token_file.with_name("real.token")
    real.write_text(TOKEN, encoding="utf-8")
    orphan.token_file.unlink()
    orphan.token_file.symlink_to(real)
    _refuse(orphan, "is a symlink")
    orphan.token_file.unlink()
    orphan.token_file.write_text("short", encoding="utf-8")
    _refuse(orphan, "too short")


def test_reclaim_refuses_when_the_data_plane_cannot_be_reached(orphan):
    orphan.scenario.contexts.discard(target().context)
    _refuse(orphan, "has no context")
    orphan.scenario.contexts.add(target().context)
    orphan.scenario.gpu_kubeconfig.unlink()
    _refuse(orphan, "does not exist")


def test_reclaim_refuses_without_the_data_plane_secret(orphan):
    orphan.scenario.connection_secrets.pop(target().context)
    _refuse(orphan, "no readable gpu-fault-system/gpu-fault-regional-connection")


def test_reclaim_refuses_a_data_plane_holding_another_token(orphan):
    orphan.scenario.connection_secrets[target().context] = _connection_secret(
        token="u" * 64
    )
    message = _refuse(orphan, "different cluster token")
    assert "u" * 64 not in message


def test_reclaim_refuses_a_data_plane_installed_for_another_cluster_id(orphan):
    orphan.scenario.connection_secrets[target().context] = _connection_secret(
        cluster_id="hp-gpu-z"
    )
    _refuse(orphan, "installed for cluster id hp-gpu-z")


def test_reclaim_refuses_a_data_plane_secret_without_a_token(orphan):
    orphan.scenario.connection_secrets[target().context] = _secret(
        {"cluster-id": CLUSTER_ID}
    )
    _refuse(orphan, "carries no cluster-token")


def test_reclaim_refuses_when_the_control_plane_knows_nothing(orphan):
    orphan.scenario.registry_secret = None
    _refuse(orphan, "control plane has no record")


def test_reclaim_refuses_a_registry_row_for_another_eks_cluster(orphan):
    orphan.scenario.registry_secret = _registry_secret(
        [_registry_row(eks_cluster_arn="arn:aws:eks:x:1:cluster/z")]
    )
    _refuse(orphan, "for EKS cluster arn:aws:eks:x:1:cluster/z")


def test_reclaim_refuses_a_registry_row_with_another_token(orphan):
    orphan.scenario.registry_secret = _registry_secret([_registry_row(token="u" * 64)])
    message = _refuse(orphan, "different token than the site's token file")
    assert "u" * 64 not in message


def test_reclaim_accepts_the_registry_backup_payload(orphan):
    orphan.scenario.registry_secret = _registry_secret(
        [_registry_row()], key="previous-clusters.json"
    )

    orphan.prepare()

    rows = orphan.journaled()["reclaimed_cluster_token"]["control_plane"][
        "regional_registry"
    ]["rows"]
    assert rows == [
        {
            "key": "previous-clusters.json",
            "lifecycle_state": "ACTIVE",
            "token_sha256_matches": True,
        }
    ]


def test_reclaim_still_refuses_a_replaced_namespace(orphan):
    orphan.scenario.namespace_uids[target().context] = "replaced-namespace"
    with pytest.raises(BootstrapError, match="existing or replaced namespace"):
        orphan.prepare()


# --- the incident, end to end --------------------------------------------------


def test_reclaim_re_attaches_a_cluster_a_rollback_dropped_from_the_site(
    tmp_path, monkeypatch
):
    """Join, lose the membership the way the rollback did, re-attach."""

    scenario = JoinScenario(tmp_path, monkeypatch, suffixes=("b",))
    assert scenario.join()["phase"] == "COMPLETED"
    token_file = scenario.path.parent / "secure" / f"{CLUSTER_ID}.token"
    document = yaml.safe_load(scenario.path.read_text())
    document["spec"]["clusters"] = [
        item for item in document["spec"]["clusters"] if item["clusterId"] != CLUSTER_ID
    ]
    scenario.path.write_text(yaml.safe_dump(document))
    with pytest.raises(BootstrapError, match="no matching join ownership"):
        scenario.join()
    assert scenario.state()["phase"] == "BLOCKED_IDENTITY"
    # The data plane the first join released still holds the token the site
    # keeps; the installation registry still holds the join's rows.
    scenario.connection_secrets[target().context] = _connection_secret(
        token=token_file.read_text().strip()
    )
    monkeypatch.setattr(
        inputs,
        "reclaim_released_alias",
        lambda *_args, **_kwargs: {"released_by": {"registry": "absent"}},
    )
    rollouts_before = list(scenario.driver_calls)

    result = join.join_cluster(
        scenario.request(reclaim_cluster_token=True), runner=scenario
    )

    assert result["phase"] == "COMPLETED"
    restored = yaml.safe_load(scenario.path.read_text())
    assert [item["clusterId"] for item in restored["spec"]["clusters"]] == [
        "gpu-a",
        CLUSTER_ID,
    ], "the cluster is back in the management site"
    new_calls = scenario.driver_calls[len(rollouts_before) :]
    assert [call for call in new_calls if call[1] == CLUSTER_ID] == [
        ("join-cluster", CLUSTER_ID),
        ("activate-cluster", CLUSTER_ID),
    ], "the normal join released and activated the cluster again"
    state = scenario.state()
    assert (
        state["reclaimed_cluster_token"]["token_sha256"]
        == hashlib.sha256(token_file.read_bytes().strip()).hexdigest()
    )
    assert (
        "installation-registry"
        in state["reclaimed_cluster_token"]["control_plane"]["sources"]
    )
    assert token_file.read_text().strip() not in json.dumps(state)


# --- the command surface ---------------------------------------------------------


def _arguments(**values) -> argparse.Namespace:
    defaults = {
        "gpu_cluster_arn": [target().eks_arn],
        "cluster_id": None,
        "allowed_namespace": [],
        "repo_root": None,
        "approve_profile_plan": None,
        "reference": None,
    }
    return argparse.Namespace(**{**defaults, **values})


def test_cli_passes_the_reclaim_flag_into_the_request(tmp_path, monkeypatch):
    scenario = JoinScenario(tmp_path, monkeypatch, suffixes=("b",))
    seen = []

    status = cli_join.run_join_cluster(
        _arguments(reclaim_cluster_token=True),
        site_file=scenario.path,
        load_site=lambda path, repository_root=None: scenario.site,
        join_cluster=lambda request: (seen.append(request), {"phase": "COMPLETED"})[1],
        join_clusters=lambda requests: pytest.fail("one cluster joins alone"),
    )

    assert status == 0
    assert [request.reclaim_cluster_token for request in seen] == [True]
    assert seen[0].gpu_cluster_arn == target().eks_arn


def test_cli_refuses_the_reclaim_flag_for_several_clusters(tmp_path, monkeypatch):
    scenario = JoinScenario(tmp_path, monkeypatch, suffixes=("b", "c"))

    with pytest.raises(SiteConfigError, match="one cluster"):
        cli_join.run_join_cluster(
            _arguments(
                reclaim_cluster_token=True,
                gpu_cluster_arn=[target("b").eks_arn, target("c").eks_arn],
            ),
            site_file=scenario.path,
            load_site=lambda path, repository_root=None: scenario.site,
            join_cluster=lambda request: pytest.fail("refused before any join"),
            join_clusters=lambda requests: pytest.fail("refused before any join"),
        )


def test_the_parser_knows_the_flag_and_defaults_it_off():
    from gpu_fault.admin import cli

    arguments = cli.parser().parse_args(
        ["join-cluster", "--state-dir", "/tmp/x", "--gpu-cluster-arn", "arn:a"]
    )
    assert arguments.reclaim_cluster_token is False
    flagged = cli.parser().parse_args(
        [
            "join-cluster",
            "--state-dir",
            "/tmp/x",
            "--gpu-cluster-arn",
            "arn:a",
            "--reclaim-cluster-token",
        ]
    )
    assert flagged.reclaim_cluster_token is True
