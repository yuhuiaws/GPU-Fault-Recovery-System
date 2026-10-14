"""The site Runtime Profile registration anchor once its cluster left the site.

``spec.runtimeProfile.registrationClusterId`` names the cluster whose provider
the site base profile family follows and under which a new site profile
version registers. remove-cluster moves it with the removal of that cluster
while members of its provider remain, and keeps it when the last cluster
leaves. Joining *different* clusters into the emptied site then left it on the
removed cluster, and the next deploy refused ("registration cluster gpu-a is not
among the managed GPU clusters"); a site an older release left that way (the
anchor removed while members remained) had no exit once that cluster was gone.

The anchor is now *departed* when it names a cluster that left -- proven by
that cluster's COMPLETED removal journal, which also records its provider:
a join fills it with a member of that provider (one on the site profile first,
then the first by cluster id, as remove-cluster chooses); the first join into
an emptied site must bring such a cluster; a zero-member deploy keeps the
departed anchor's family; and ``deploy --registration-cluster <id> --reference
<CHG>`` moves it explicitly, proof-checked and journaled, on a site an older
release left behind.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import cluster_join as join
from gpu_fault.admin import profile_approval
from gpu_fault.admin.bootstrap_common import BootstrapError, ClusterIdentity
from gpu_fault.admin.bootstrap_discovery import (
    EKS_MANAGED_PROFILE,
    HYPERPOD_PROFILE,
    regional_profile_for,
    registration_cluster_provider,
)
from gpu_fault.admin.bootstrap_site import (
    load_existing_site,
    recover_verified_site_contract,
)
from gpu_fault.admin.cluster_join_profile import JoinProfileApprovalRequired
from gpu_fault.admin.cluster_removal import remove_cluster
from gpu_fault.admin.cluster_removal_state import STEP_DEPENDENCIES
from gpu_fault.admin.site import SiteConfigError
from tests.admin._cluster_removal_support import RemovalScenario
from tests.admin._cov95_join_support import ACCOUNT, REGION, JoinScenario, target
from tests.admin.test_admin_cluster_removal_registration import (
    _add_members,
    _registry_with_members,
)
from tests.admin.test_admin_site import _managed_state
from tests.admin.test_multi_cluster_profile_review import MixedJoinScenario

JOURNAL = Path("registration-anchor/journal.json")


def _document(path: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def _write(path: Path, document: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _anchor(site: Path) -> str:
    return str(_document(site)["spec"]["runtimeProfile"]["registrationClusterId"])


def _journal(state_dir: Path) -> list[dict[str, Any]]:
    document = json.loads((state_dir / JOURNAL).read_text(encoding="utf-8"))
    assert document["schema_version"] == 1, document
    return list(document["entries"])


def _emptied(scenario: JoinScenario, tmp_path: Path, monkeypatch: Any) -> Path:
    """Make ``scenario``'s site what a real remove-cluster of its last
    cluster ``gpu-a`` leaves: no GPU cluster, the anchor still ``gpu-a``, and
    that removal's COMPLETED journal."""

    removed = RemovalScenario(tmp_path / "removed", monkeypatch)
    assert remove_cluster(removed.request())["phase"] == "COMPLETED"
    left = _document(removed.path)
    assert left["spec"]["clusters"] == [], left["spec"]["clusters"]
    assert left["spec"]["runtimeProfile"]["registrationClusterId"] == "gpu-a"
    state_dir = scenario.path.parent
    shutil.copytree(
        removed.path.parent / "remove-cluster", state_dir / "remove-cluster"
    )
    document = _document(scenario.path)
    (member,) = document["spec"]["clusters"]
    document["spec"]["clusters"] = []
    _write(scenario.path, document)
    # With no member left, a join finds the site CA in the bootstrap state.
    pki = {"ca_file": member["caFile"], "hosted_zone_id": "ZEXAMPLE"}
    bootstrap = {"site_id": "test-site", "resources": {"pki": pki}}
    (state_dir / "bootstrap-state.json").write_text(json.dumps(bootstrap))
    scenario.cluster_states = {}
    return state_dir


def _completed_removal(
    state_dir: Path, cluster_id: str, *, provider: str = "hyperpod"
) -> Path:
    """A COMPLETED remove-cluster journal of ``cluster_id``, as it is on disk."""

    target_entry: dict[str, Any] = {
        "cluster_id": cluster_id,
        "context": cluster_id,
        "eks_cluster_arn": f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/{cluster_id}",
    }
    if provider == "eks-managed":
        target_entry["provider"] = provider
    else:
        target_entry["hyperpod_cluster_name"] = f"hp-{cluster_id}"
    state = {
        "schema_version": 2,
        "site_id": "test-site",
        "cluster_id": cluster_id,
        "target": target_entry,
        "registration_reanchor": None,
        "attempt_id": "a" * 32,
        "phase": "COMPLETED",
        "completed_steps": list(STEP_DEPENDENCIES),
        "evidence": {
            "DISCOVERED": {
                "target": target_entry,
                "provider_identity": {
                    "eks_arn": target_entry["eks_cluster_arn"],
                    "hyperpod_arn": ""
                    if provider == "eks-managed"
                    else f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/hp",
                },
                "namespace_uid": "namespace-1",
                "registry_digest": "r" * 64,
                "token_sha256": "t" * 64,
            }
        },
    }
    path = state_dir / "remove-cluster" / cluster_id / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(state), encoding="utf-8")
    return path


def _identity(cluster_id: str, provider: str = "hyperpod") -> ClusterIdentity:
    arn = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/{cluster_id}"
    hyperpod = provider == "hyperpod"
    return ClusterIdentity(
        input_arn=arn,
        role="gpu",
        region=REGION,
        account_id=ACCOUNT,
        hyperpod_arn=f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/hp-{cluster_id}"
        if hyperpod
        else "",
        hyperpod_name=f"hp-{cluster_id}" if hyperpod else "",
        eks_arn=arn,
        eks_name=cluster_id,
        vpc_id="vpc-a",
        subnet_ids=("subnet-a",),
        node_recovery="None" if hyperpod else "",
        context=cluster_id,
        provider=provider,
    )


def _state(tmp_path: Path, *, anchor: str, clusters: list[tuple[str, str]]) -> Path:
    """A ``site.yaml`` whose anchor and member list are given."""

    document = {
        "metadata": {"name": "test-site"},
        "spec": {
            "runtimeProfile": {"version": "x", "registrationClusterId": anchor},
            "clusters": [
                {
                    "clusterId": name,
                    "eksClusterArn": f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/{name}",
                    **(
                        {"provider": provider}
                        if provider == "eks-managed"
                        else {"hyperpodClusterName": f"hp-{name}"}
                    ),
                }
                for name, provider in clusters
            ],
        },
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    _write(tmp_path / "site.yaml", document)
    return tmp_path


# --- a join fills the departed anchor ------------------------------------------


def test_the_first_join_into_an_emptied_site_takes_over_the_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = JoinScenario(tmp_path / "site", monkeypatch)
    state_dir = _emptied(scenario, tmp_path, monkeypatch)

    result = scenario.join("b")

    assert result["phase"] == "COMPLETED", result
    assert result["registration_reanchor"] == "hp-gpu-b", result
    assert _anchor(scenario.path) == "hp-gpu-b", (
        "the joined HyperPod cluster takes over the anchor of the removed gpu-a"
    )
    identities = [target("b")]
    assert registration_cluster_provider(state_dir, identities) == "hyperpod", (
        "the next deploy reads the base family from the new anchor; it refused here"
    )
    assert regional_profile_for(Path("/repo"), identities, state_dir=state_dir) == (
        f"/repo/{HYPERPOD_PROFILE[0]}",
        "hyperpod-v1",
    )
    (entry,) = _journal(state_dir)
    assert (entry["previous"], entry["registration_cluster_id"]) == (
        "gpu-a",
        "hp-gpu-b",
    ), entry
    assert entry["trigger"] == "join-cluster", entry
    assert entry["proof"].endswith("remove-cluster/gpu-a/state.json"), entry

    again = scenario.join("b")
    assert "registration_reanchor" not in again, "a member anchor never moves again"
    assert len(_journal(state_dir)) == 1


def test_a_batch_join_into_an_emptied_site_anchors_on_its_first_cluster_by_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = JoinScenario(tmp_path / "site", monkeypatch)
    state_dir = _emptied(scenario, tmp_path, monkeypatch)

    result = scenario.batch(("c", "b"))

    assert result["phase"] == "COMPLETED", result
    assert result["joined"] == ["hp-gpu-b", "hp-gpu-c"], result
    assert result["registration_reanchor"] == "hp-gpu-b", result
    assert _anchor(scenario.path) == "hp-gpu-b"
    assert registration_cluster_provider(state_dir, [target("b"), target("c")]) == (
        "hyperpod"
    )


def test_a_member_anchor_is_left_where_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = JoinScenario(tmp_path, monkeypatch)

    result = scenario.join("b")

    assert result["phase"] == "COMPLETED", result
    assert "registration_reanchor" not in result, result
    assert _anchor(scenario.path) == "gpu-a"
    assert not (scenario.path.parent / JOURNAL).exists(), "nothing was moved"


def test_the_first_join_into_an_emptied_site_must_bring_the_anchors_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain EKS cluster as the only member of a HyperPod-family site would
    leave no cluster to name the base after: refused before anything changes,
    alone or in a batch; with a HyperPod cluster in the same batch it joins,
    and the HyperPod cluster takes the anchor."""

    scenario = MixedJoinScenario(tmp_path / "site", monkeypatch)
    _emptied(scenario, tmp_path, monkeypatch)

    for attempt in (
        lambda: scenario.join_batch("e"),
        lambda: join.join_cluster(scenario.eks_request("e"), runner=scenario),
    ):
        with pytest.raises(BootstrapError) as refused:
            attempt()
        message = str(refused.value)
        assert "registration cluster gpu-a" in message, message
        assert "the first cluster to join must be a hyperpod cluster" in message, (
            message
        )
        assert message.endswith("Nothing was changed"), message
    assert _document(scenario.path)["spec"]["clusters"] == [], "nothing joined"
    assert not any(event.startswith("executor-role") for event in scenario.events), (
        "no AWS mutation"
    )
    assert not profile_approval.profile_plan_path(scenario.state_dir).exists(), (
        "the refusal precedes the profile review"
    )

    with pytest.raises(JoinProfileApprovalRequired):
        scenario.join_batch("b", "e")
    plan = scenario.plan()
    profile_approval.approve_profile(
        scenario.state_dir,
        reference="CHG-anchor-1",
        expected_plan_sha256=str(plan["plan_sha256"]),
        approver_identity="tester@fixture",
    )
    result = scenario.join_batch("b", "e")

    assert result["phase"] == "COMPLETED", result
    assert result["registration_reanchor"] == "hp-gpu-b", result
    assert _anchor(scenario.path) == "hp-gpu-b"
    assert scenario.entry("gpu-e")["runtimeProfile"]["version"] == "eks-v1", (
        "the plain EKS cluster runs its own profile beside the HyperPod anchor"
    )


# --- deploy --------------------------------------------------------------------


def test_a_zero_member_deploy_keeps_the_family_of_the_departed_anchor(
    tmp_path: Path,
) -> None:
    """An eks-first site whose clusters were all removed re-derived the HyperPod
    template on its next deploy (the fresh-site rule for "no cluster"), and the
    release then offered an eks-v1 -> hyperpod-v1 re-base."""

    state = _state(tmp_path, anchor="gpu-eks-cluster", clusters=[])
    _completed_removal(state, "gpu-eks-cluster", provider="eks-managed")

    assert registration_cluster_provider(state, []) == "eks-managed"
    assert regional_profile_for(Path("/repo"), [], state_dir=state) == (
        f"/repo/{EKS_MANAGED_PROFILE[0]}",
        "eks-v1",
    ), "the emptied site keeps the eks family of the cluster it was anchored on"
    assert (
        registration_cluster_provider(
            _state(tmp_path / "unproven", anchor="gpu-eks-cluster", clusters=[]), []
        )
        is None
    ), "without the removal journal the old rule stands"


def test_a_departed_anchor_with_members_is_refused_with_the_way_out(
    tmp_path: Path,
) -> None:
    members = [("gpu-e", "eks-managed"), ("gpu-b", "hyperpod"), ("gpu-c", "hyperpod")]
    state = _state(tmp_path, anchor="gpu-a", clusters=members)
    _completed_removal(state, "gpu-a")
    identities = [_identity(name, provider) for name, provider in members]

    with pytest.raises(BootstrapError) as refused:
        registration_cluster_provider(state, identities)

    message = str(refused.value)
    assert "registration cluster gpu-a is not among the managed GPU clusters" in (
        message
    )
    assert "remove-cluster/gpu-a/state.json" in message, "the proof is named"
    assert "(gpu-b, gpu-c)" in message, "the members that may take it over"
    assert (
        f"gpu-fault-admin deploy --state-dir {state} --registration-cluster gpu-b "
        "--reference <CHG>"
    ) in message, message

    other = _state(tmp_path / "other", anchor="gpu-a", clusters=members[:1])
    _completed_removal(other, "gpu-a")
    with pytest.raises(BootstrapError, match="no hyperpod cluster remains") as stuck:
        registration_cluster_provider(other, identities[:1])
    assert "join a hyperpod cluster" in str(stuck.value), str(stuck.value)

    unproven = _state(tmp_path / "unproven", anchor="gpu-a", clusters=members)
    with pytest.raises(BootstrapError, match="no completed remove-cluster journal"):
        registration_cluster_provider(unproven, identities)


def _stuck_site(tmp_path: Path) -> Path:
    """The site an older remove-cluster left: ``gpu-a`` (the anchor) removed
    while ``gpu-b`` and ``gpu-c`` remained, the anchor still ``gpu-a``."""

    state_dir = _managed_state(tmp_path, gpu_names=("gpu-b", "gpu-c"))
    assert _anchor(state_dir / "site.yaml") == "gpu-a"
    _completed_removal(state_dir, "gpu-a")
    return state_dir


def _deploy(state_dir: Path, *extra: str) -> int:
    arguments = admin_cli.parser().parse_args(
        ["deploy", "--state-dir", str(state_dir), *extra]
    )
    return int(admin_cli.run(arguments))


def test_deploy_refuses_a_departed_anchor_before_the_source_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = _stuck_site(tmp_path)
    monkeypatch.setattr(
        admin_cli,
        "run_source_deploy",
        lambda **_kwargs: pytest.fail("the deploy went on with a departed anchor"),
    )

    with pytest.raises(BootstrapError, match="--registration-cluster gpu-b"):
        _deploy(state_dir)


def test_deploy_registration_cluster_moves_a_departed_anchor_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state_dir = _stuck_site(tmp_path)
    events: list[str] = []
    monkeypatch.setattr(
        admin_cli,
        "run_source_deploy",
        lambda **_kwargs: events.append(_anchor(state_dir / "site.yaml")) or 0,
    )

    assert (
        _deploy(state_dir, "--registration-cluster", "gpu-c", "--reference", "CHG-7")
        == 0
    )

    assert events == ["gpu-c"], "the deploy continues once the anchor moved"
    (entry,) = _journal(state_dir)
    assert (entry["previous"], entry["registration_cluster_id"]) == ("gpu-a", "gpu-c")
    assert entry["trigger"] == "deploy --registration-cluster", entry
    assert entry["reference"] == "CHG-7", entry
    assert entry["proof"].endswith("remove-cluster/gpu-a/state.json"), entry
    assert "registration cluster moved from gpu-a to gpu-c" in capsys.readouterr().err

    assert _deploy(state_dir) == 0, "the next plain deploy needs nothing more"
    assert events == ["gpu-c", "gpu-c"]


@pytest.mark.parametrize(
    "names,extra,message",
    [
        pytest.param(
            ("gpu-a", "gpu-b"),
            ["--registration-cluster", "gpu-b", "--reference", "CHG-1"],
            "registration cluster gpu-a is a member of the site",
            id="anchor-is-a-member",
        ),
        pytest.param(
            ("gpu-b", "gpu-c"),
            ["--registration-cluster", "gpu-x", "--reference", "CHG-1"],
            "gpu-x cannot take over the registration anchor",
            id="not-a-member",
        ),
        pytest.param(
            ("gpu-b", "gpu-c"),
            ["--registration-cluster", "gpu-b"],
            "--registration-cluster requires --reference",
            id="no-reference",
        ),
        pytest.param(
            ("gpu-b", "gpu-c"),
            ["--reference", "CHG-1"],
            "--reference is only used with --approve-profile-plan or "
            "--registration-cluster",
            id="reference-alone",
        ),
    ],
)
def test_deploy_registration_cluster_refusals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    names: tuple[str, ...],
    extra: list[str],
    message: str,
) -> None:
    state_dir = _managed_state(tmp_path, gpu_names=names)
    _completed_removal(state_dir, "gpu-a")
    before = (state_dir / "site.yaml").read_bytes()
    monkeypatch.setattr(
        admin_cli, "run_source_deploy", lambda **_kwargs: pytest.fail("deployed")
    )

    with pytest.raises((BootstrapError, SiteConfigError)) as refused:
        _deploy(state_dir, *extra)

    assert message in str(refused.value), str(refused.value)
    assert (state_dir / "site.yaml").read_bytes() == before, "site.yaml is untouched"
    assert not (state_dir / JOURNAL).exists(), "no anchor move was journaled"


def test_deploy_registration_cluster_refuses_the_other_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = _stuck_site(tmp_path)
    document = _document(state_dir / "site.yaml")
    plain = document["spec"]["clusters"][1]
    plain.pop("hyperpodClusterName")
    plain.update({"provider": "eks-managed", "nodeSelector": "a=b"})
    _write(state_dir / "site.yaml", document)
    monkeypatch.setattr(
        admin_cli, "run_source_deploy", lambda **_kwargs: pytest.fail("deployed")
    )

    with pytest.raises(BootstrapError) as refused:
        _deploy(state_dir, "--registration-cluster", "gpu-c", "--reference", "CHG-2")

    message = str(refused.value)
    assert "gpu-c cannot take over the registration anchor" in message, message
    assert "hyperpod clusters: gpu-b" in message, message


def test_deploy_rollback_refuses_registration_cluster(tmp_path: Path) -> None:
    state_dir = _stuck_site(tmp_path)

    with pytest.raises(SiteConfigError, match="takes no other deploy option"):
        _deploy(
            state_dir,
            "--rollback",
            "--registration-cluster",
            "gpu-b",
            "--reference",
            "CHG-3",
        )


# --- the next deploy keeps the moved anchor ------------------------------------


def _verified_release(state_dir: Path, document: dict[str, Any]) -> None:
    """The last COMPLETED, verified release and its candidate site: the contract
    the next deploy's bootstrap restores ``spec.release`` and
    ``spec.runtimeProfile`` from."""

    release = state_dir / "release-deploy" / "release-a"
    release.mkdir(parents=True)
    (release / "state.json").write_text(
        json.dumps({"phase": "COMPLETED", "verification": {"status": "PASSED"}}),
        encoding="utf-8",
    )
    _write(release / "site.candidate.yaml", document)


def _recovered_anchor(state_dir: Path) -> str:
    recovered = recover_verified_site_contract(state_dir, load_existing_site(state_dir))
    assert recovered is not None
    return str(recovered["spec"]["runtimeProfile"]["registrationClusterId"])


def test_an_explicitly_moved_anchor_survives_the_next_deploys_contract_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bootstrap restored the anchor with the rest of the verified
    contract's profile fields: the move lasted one deploy, and the deploy
    after it refused again."""

    state_dir = _stuck_site(tmp_path)
    verified = _document(state_dir / "site.yaml")
    verified["spec"]["runtimeProfile"]["version"] = "verified-v1"
    _verified_release(state_dir, verified)
    monkeypatch.setattr(admin_cli, "run_source_deploy", lambda **_kwargs: 0)
    assert (
        _deploy(state_dir, "--registration-cluster", "gpu-b", "--reference", "CHG-8")
        == 0
    )

    assert _recovered_anchor(state_dir) == "gpu-b", (
        "the anchor follows membership, not the last release"
    )
    recovered = recover_verified_site_contract(state_dir, load_existing_site(state_dir))
    assert recovered is not None
    assert recovered["spec"]["runtimeProfile"]["version"] == "verified-v1", (
        "the other profile fields are still the verified contract's"
    )


def test_a_removal_re_anchor_survives_the_next_deploys_contract_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """remove-cluster's own move (batch 84) was undone the same way."""

    scenario = RemovalScenario(tmp_path, monkeypatch)
    _registry_with_members(scenario, monkeypatch)
    _add_members(scenario, ("gpu-b", "hyperpod"))
    _verified_release(scenario.path.parent, _document(scenario.path))

    assert remove_cluster(scenario.request())["registration_reanchor"] == "gpu-b"

    assert _recovered_anchor(scenario.path.parent) == "gpu-b"
