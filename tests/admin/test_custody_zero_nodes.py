"""The custody check of ``preflight``/``deploy -f``/``join-cluster`` over a GPU
cluster without nodes (``custody_zero_nodes``).

The original check probed every custody-registered cluster's binding, and a
cluster without nodes has none: one such cluster refused every preflight and
every join of the site. A registered cluster the release engine proves to have
no node is now skipped and said; every other cluster is probed exactly as
before. The custody modules are node-template inputs and stay untouched.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from gpu_fault.admin import bootstrap, cli, cluster_join
from gpu_fault.admin import custody_zero_nodes as custody
from gpu_fault.admin import node_key_custody_admin_entry as entry
from gpu_fault.admin.bootstrap_common import BootstrapMutationRequired
from gpu_fault.admin.node_key_custody_admin import CustodyReconciliationRequired
from tests.admin._node_key_custody_admin_support import AdminWorld
from tests.admin.test_node_key_custody_admin_join import joining as joining
from tests.regional._cov95_identity_support import offline_guard as offline_guard

ROOT = Path(__file__).resolve().parents[2]
PARKED = {
    "parked": True,
    "provider": "hyperpod",
    "reason": "every instance group the node selector covers is InService at "
    "TargetCount 0 and CurrentCount 0",
    "groups": [],
}


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AdminWorld:
    return AdminWorld(tmp_path, monkeypatch)


def registered(world: AdminWorld, monkeypatch: pytest.MonkeyPatch) -> None:
    world.configure()
    world.access()
    world.release_ready = True
    monkeypatch.setattr(bootstrap, "discover_cluster", lambda *a, **k: world.gpu)


def test_a_registered_cluster_with_nodes_is_probed_exactly_as_before(
    world: AdminWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same outcome as the original entry point, after one node read."""

    registered(world, monkeypatch)
    with pytest.raises(CustodyReconciliationRequired, match="public preflight"):
        entry.assert_site_custody_current(world.site(), world)
    before = len(world.calls)

    with pytest.raises(CustodyReconciliationRequired, match="public preflight"):
        custody.assert_site_custody_current(world.site(), world)

    node_reads = [
        call for call in world.calls[before:] if "get" in call and "nodes" in call
    ]
    assert node_reads, "the zero-node question is asked of the node list first"
    assert world.helper_calls == 0, "the probe never provisions"


def test_a_registered_cluster_without_nodes_is_skipped_and_said(
    world: AdminWorld,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registered(world, monkeypatch)
    world.api.state["nodes"]["items"] = []
    with pytest.raises(CustodyReconciliationRequired, match="node inventory"):
        entry.assert_site_custody_current(world.site(), world)

    custody.assert_site_custody_current(world.site(), world)

    narration = capsys.readouterr().err
    assert "zero-node-gpu-cluster" in narration and "step=custody" in narration, (
        narration
    )
    assert 'recorded="no node, no custody binding to check"' in narration, narration
    assert world.helper_calls == 0, "nothing is provisioned for a cluster without nodes"


def test_an_unconfigured_site_reads_nothing(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    runner = SimpleNamespace(run=lambda arguments, **_k: calls.append(arguments))
    site = SimpleNamespace(
        source=tmp_path / "site.yaml",
        release_config={"clusters": [{"cluster_id": "gpu-a", "eks_cluster_arn": "x"}]},
    )

    custody.assert_site_custody_current(site, runner)  # type: ignore[arg-type]

    assert calls == [], "no custody registration, no read"
    assert custody.custody_registered(site, "x") is False  # type: ignore[arg-type]


def test_an_incomplete_probe_still_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    registration = SimpleNamespace(selection=SimpleNamespace(clusters={"arn-a": None}))
    site = SimpleNamespace(
        source=Path("/state/site.yaml"),
        release_config={
            "site_name": "site-a",
            "aws_region": "us-west-2",
            "clusters": [
                {"cluster_id": "gpu-a", "eks_cluster_arn": "arn-a", "context": "a"},
                {"cluster_id": "gpu-b", "eks_cluster_arn": "arn-b", "context": "b"},
            ],
        },
    )
    probed: list[str] = []

    def provision(_runner: Any, context: Any, **_options: Any) -> None:
        probed.append(context)
        raise BootstrapMutationRequired("custody preparation is incomplete")

    monkeypatch.setattr(custody, "load_admin_custody", lambda _root: registration)
    monkeypatch.setattr(custody, "zero_node_proof_for_entry", lambda *_a, **_k: None)
    monkeypatch.setattr(bootstrap, "discover_cluster", lambda *_a, **_k: "identity")
    monkeypatch.setattr(
        custody, "site_custody_context", lambda _site, cluster, cluster_id: cluster_id
    )
    monkeypatch.setattr(custody, "provision_admin_custody", provision)

    with pytest.raises(CustodyReconciliationRequired, match="public preflight"):
        custody.assert_site_custody_current(site, object())  # type: ignore[arg-type]
    assert probed == ["gpu-a"], "only the registered cluster is probed"


def test_preflight_and_deploy_by_file_use_the_zero_node_aware_check() -> None:
    assert cli.assert_site_custody_current is custody.assert_site_custody_current, (
        "preflight and deploy -f must ask the zero-node-aware check"
    )
    assert entry.assert_site_custody_current is not custody.assert_site_custody_current


def test_join_cluster_asks_the_zero_node_aware_check(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    world, join_request, _state_path, rollouts = request.getfixturevalue("joining")
    asked: list[object] = []

    def check(site: Any, runner: Any) -> None:
        asked.append(runner)
        raise RuntimeError("the zero-node-aware custody check was asked")

    monkeypatch.setattr(custody, "assert_site_custody_current", check)

    with pytest.raises(RuntimeError, match="zero-node-aware custody check"):
        cluster_join.join_cluster(join_request, runner=world)
    assert asked == [world], "join-cluster asks it first, with its own runner"
    assert rollouts == [], "nothing ran before the custody check"


def test_the_zero_node_custody_check_is_no_node_template_input() -> None:
    """An edit to a node-template input rolls every GPU node's Agent: the
    zero-node custody check must live outside those file sets, and the custody
    modules must remain the only custody files the template digest names."""

    identity = yaml.safe_load((ROOT / "config/release-identity.yaml").read_text())
    patterns = identity["node_template_inputs"]
    module = Path(custody.__file__).resolve().relative_to(ROOT).as_posix()
    assert not any(fnmatch.fnmatch(module, pattern) for pattern in patterns), (
        f"{module} matches a node_template_inputs pattern"
    )
    original = Path(entry.__file__).resolve().relative_to(ROOT).as_posix()
    assert any(fnmatch.fnmatch(original, pattern) for pattern in patterns), (
        "the original custody entry point is a node-template input"
    )
