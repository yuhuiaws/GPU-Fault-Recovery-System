"""``join-cluster`` of a GPU cluster without nodes (``cluster_join_zero_nodes``).

The join refused an empty node inventory after it had written the kubeconfig
alias, the namespace and the cluster token, and rolled them back. A cluster
whose selector selects no node now joins with its node work skipped and
recorded; a cluster that cannot be joined that way -- no GPU instance type the
installer knows, or registered for node-key custody -- is refused before
anything is created on it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import cluster_join as admin_cluster_join
from gpu_fault.admin import cluster_join_zero_nodes as zero_nodes
from gpu_fault.admin import custody_zero_nodes
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault_release import regional_parked_cluster as parked_cluster
from tests.admin.test_admin_cluster_join_rollback import Attempt
from tests.admin.test_admin_eks_managed_join import EksJoinScenario, eks_target
from tests.regional._parked_cluster_support import cpu_nodegroup, nodegroup

PARKED = {
    "state": "parked",
    "provider": "eks-managed",
    "reason": "every managed node group the node selector covers is ACTIVE at "
    "desiredSize 0 and no Karpenter NodePool can launch a node it selects",
    "groups": [],
}


class ZeroNodeJoin(EksJoinScenario):
    """The plain EKS join world, with the joining cluster ``gpu-e`` at zero
    nodes: its GPU managed node group parked (or still wanting nodes), no
    Karpenter CRD, and the GPU node-key Secret absent until the join creates it."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        desired: int = 0,
        instance_type: str = "p5en.48xlarge",
    ) -> None:
        super().__init__(tmp_path, monkeypatch)
        group = nodegroup("gpu-group", desired=desired)
        group["instanceTypes"] = [instance_type]
        self.nodegroups = {"gpu-group": group, "cpu-group": cpu_nodegroup()}
        self.created: list[dict[str, Any]] = []

    def run(self, arguments, **options):  # type: ignore[no-untyped-def]
        context = (
            arguments[arguments.index("--context") + 1]
            if "--context" in arguments
            else "cpu"
        )
        if context == "gpu-e" and "get" in arguments and "nodes" in arguments:
            self.commands.append((list(arguments), dict(options)))
            if any(str(item).startswith("go-template=") for item in arguments):
                return json.dumps([])
            return json.dumps({"apiVersion": "v1", "kind": "List", "items": []})
        if any(str(item).startswith("nodepools.karpenter.sh") for item in arguments):
            raise BootstrapError(
                'error: the server doesn\'t have a resource type "nodepools"'
            )
        if list(arguments[:3]) == ["aws", "eks", "list-nodegroups"]:
            return json.dumps({"nodegroups": sorted(self.nodegroups)})
        if list(arguments[:3]) == ["aws", "eks", "describe-nodegroup"]:
            name = arguments[arguments.index("--nodegroup-name") + 1]
            return json.dumps(self.nodegroups[name])
        if "create" in arguments and options.get("input_text"):
            document = json.loads(options["input_text"])
            if document.get("kind") == "Secret":
                assert options.get("mutate") is True, "a write is marked a mutation"
                self.created.append(document)
                self.node_key_secrets[context] = dict(document["data"])
                return ""
        return super().run(arguments, **options)

    def command(self, arguments, **options):  # type: ignore[no-untyped-def]
        # The collector-readiness step re-reads the zero-node proof through
        # the join's own run_command.
        if list(arguments[:2]) == ["aws", "eks"] or any(
            str(item).startswith("nodepools.karpenter.sh") for item in arguments
        ):
            try:
                return self.result(arguments, self.run(arguments))
            except BootstrapError as exc:
                return self.result(arguments, code=1, error=str(exc))
        return super().command(arguments, **options)


@pytest.fixture
def parked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ZeroNodeJoin:
    return ZeroNodeJoin(tmp_path, monkeypatch)


def _state(scenario: ZeroNodeJoin) -> dict[str, Any]:
    path = next((scenario.path.parent / "join-cluster").glob("*/state.json"))
    return json.loads(path.read_text())


@pytest.mark.parametrize(("desired", "state"), [(0, "parked"), (2, "unavailable")])
def test_a_cluster_without_nodes_joins_with_its_node_work_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    desired: int,
    state: str,
) -> None:
    scenario = ZeroNodeJoin(tmp_path, monkeypatch, desired=desired)

    result = scenario.join_eks()

    assert result["phase"] == "COMPLETED", result
    journal = _state(scenario)
    evidence = journal["evidence"]
    local = evidence["LOCAL_INPUTS_READY"]
    assert local["nodes"] == [] and local["node_uids"] == {}, local
    assert local["zero_nodes"]["state"] == state, local
    verified = evidence["NODE_NAMES_VERIFIED"]
    assert verified["nodes"] == [] and verified["expected_key_sha256"] == {}, verified
    assert verified["zero_nodes"] == local["zero_nodes"], verified
    keys = evidence["PREREQUISITES_READY"]["node_keys"]
    assert keys["gpu_node_key_secret"]["created"] is True, keys
    assert keys["gpu_node_key_secret"]["keys"] == 0, keys
    assert scenario.created == [
        {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": "gpu-fault-node-action-keys",
                "namespace": "gpu-fault-system",
            },
            "type": "Opaque",
            "data": {},
        }
    ], "the Secret the Executor mounts is created empty"
    assert scenario.node_key_environments == [], "the key helper never runs"
    readiness = evidence["COLLECTORS_READY"]
    assert readiness["zero_node_not_awaited"] is True, readiness
    assert "collectors" not in scenario.events, "nothing will report, nothing waited"
    assert "NODE_KEYS_STARTED" in journal["completed_steps"], journal
    narration = capsys.readouterr().err
    assert "step=join" in narration and f"state={state}" in narration, narration


def test_a_cluster_whose_groups_declare_no_known_gpu_type_is_refused_up_front(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = ZeroNodeJoin(tmp_path, monkeypatch, instance_type="x9z.unknown")
    scenario.nodegroups["gpu-group"]["labels"]["nvidia.com/gpu.present"] = "true"

    with pytest.raises(
        BootstrapError, match="no GPU instance type the installer knows"
    ):
        scenario.join_eks()
    assert not any(event.startswith("namespace:") for event in scenario.events), (
        "nothing is created on the cluster before the refusal"
    )
    assert scenario.created == [], "no Secret either"


def test_a_custody_registered_cluster_without_nodes_is_refused_up_front(
    parked: ZeroNodeJoin, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        custody_zero_nodes,
        "custody_registered",
        lambda _site, arn: arn == eks_target().eks_arn,
    )

    with pytest.raises(BootstrapError, match="node-key custody is registered for it"):
        parked.join_eks()
    assert not any(event.startswith("namespace:") for event in parked.events), (
        "nothing is created on the cluster before the refusal"
    )


def test_an_existing_node_key_secret_is_kept_as_it_is(parked: ZeroNodeJoin) -> None:
    parked.node_key_secrets["gpu-e"] = {}

    assert parked.join_eks()["phase"] == "COMPLETED"

    keys = _state(parked)["evidence"]["PREREQUISITES_READY"]["node_keys"]
    assert keys["gpu_node_key_secret"]["created"] is False, keys
    assert parked.created == [], "an existing Secret is never replaced"


@pytest.mark.parametrize("nodes_now", [False, True])
def test_the_collector_wait_skips_only_a_cluster_that_still_has_no_node(
    monkeypatch: pytest.MonkeyPatch, nodes_now: bool
) -> None:
    waited: list[str] = []
    monkeypatch.setattr(
        zero_nodes,
        "zero_node_proof",
        lambda *_a, **_k: None if nodes_now else {**PARKED, "parked": True},
    )
    execution = SimpleNamespace(
        local={"zero_nodes": PARKED, "gpu_kubeconfig": "/secure/gpu.kubeconfig"},
        target=eks_target(),
        cluster_id="gpu-e",
        candidate="candidate",
    )

    def wait(candidate: Any, cluster_id: str) -> dict[str, Any]:
        waited.append(cluster_id)
        return {"ready": True, "nodes": [{"ready": True}]}

    evidence = zero_nodes.collector_readiness(execution, wait)  # type: ignore[arg-type]

    if nodes_now:
        assert waited == ["gpu-e"] and evidence == {"nodes": 1, "ready": True}
    else:
        assert waited == [], "nothing is waited for on a cluster without nodes"
        assert evidence[parked_cluster.NOT_AWAITED_KEY] is True, evidence


class ZeroNodeAttempt(Attempt):
    """A zero-node join that reached ``JOINED`` and failed: the real rollback."""

    def __init__(self, tmp_path: Path) -> None:
        super().__init__(tmp_path)
        self.removed: list[Any] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        super().install(monkeypatch)
        monkeypatch.setattr(
            admin_cluster_join,
            "_remove_node_action_keys",
            lambda *arguments, **_keywords: self.removed.append(arguments),
        )

    def evidence(self) -> dict[str, Any]:
        record = super().evidence()
        record["LOCAL_INPUTS_READY"].update(nodes=[], node_uids={}, zero_nodes=PARKED)
        record["NODE_NAMES_VERIFIED"].update(
            nodes=[], node_uids={}, expected_key_sha256={}, zero_nodes=PARKED
        )
        record["PREREQUISITES_READY"]["node_keys"] = {
            "cluster_id": "hp-gpu-b",
            "zero_nodes": PARKED,
            "gpu_node_key_secret": {"uid": "keys", "keys": 0, "created": True},
        }
        return record


def test_the_rollback_of_a_zero_node_join_undoes_what_it_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = ZeroNodeAttempt(tmp_path)

    attempt.run(monkeypatch, error="cluster deploy failed")

    assert attempt.cleared == [], "no node to clear installer annotations from"
    assert attempt.removed == [], "no CPU node key is removed: none was provisioned"
    namespace = attempt.commands.matching("delete --raw")
    assert namespace and "/api/v1/namespaces/gpu-fault-system" in namespace[0], (
        "the namespace, with the empty node-key Secret, is deleted"
    )
    state = attempt.state()
    assert state["phase"] == "ROLLED_BACK", state["phase"]
    assert "rollback_errors" not in state, state.get("rollback_errors")
