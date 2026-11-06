"""Uninstall and remove-cluster node targets on a label-convention cluster.

A cluster joined under the label convention selects its GPU nodes through the
node-group label ``gpu-fault.io/control-plane=<site id>``; an eks-managed group
in DCGM host mode also declares ``gpu-fault.io/dcgm-hostengine=host``. The
administrator sets both on the node group, every node of the group carries
them, and the product never writes either on a Node (``OWNERSHIP_RELEASED``
takes the ownership label back from the node groups, the Kubernetes cleanup
never touches them). The cleanup's capture used to count them as installed
product state like any other ``gpu-fault.io/`` key, so on such a cluster a
replacement node without an Agent could never be an orphaned installer node, a
labelled node the reconciler had not reached yet failed the uninstall as an
"unregistered node runtime", and a join that failed before its node runtime
rollout could not be rolled back ("no proven fleet targets").

These cases drive the capture, the metadata cleanup and the node cleanup
against the scripted API server of the neighbouring tests: the two node-group
declarations are not product state, and every label, annotation and taint the
product does set on a Node still fails closed.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import node_patch_retry as RETRY
from gpu_fault.cluster_scope import DCGM_HOSTENGINE_LABEL, OWNERSHIP_LABEL
from tests._script_loader import lazy_script_module
from tests.deploy._cov95_tools_support import TOOLS
from tests.regional.test_cleanup_kubernetes import CONTEXT, Client, document
from tests.regional.test_cleanup_node_metadata_race import (
    RacingClient,
    bump,
    kubelet_status,
)

KUBE = lazy_script_module(TOOLS / "cleanup_kubernetes.py")
PATCH = lazy_script_module(TOOLS / "cleanup_node_patch.py")
FAILURE = lazy_script_module(TOOLS / "cleanup_failure.py")
SITE_ID = "us-west-2-site-0a1b2c3d"
OWNERSHIP_ONLY = {OWNERSHIP_LABEL: SITE_ID}
HOST_DCGM = {OWNERSHIP_LABEL: SITE_ID, DCGM_HOSTENGINE_LABEL: "host"}
INSTALLER_FAILED = {
    "gpu-fault.io/installer-state": "Failed",
    "gpu-fault.io/installer-attempts": "2",
    "gpu-fault.io/installer-reason": "NodeKeyUnavailable",
    "gpu-fault.io/installer-node-uid": "uid-customer-node",
    "customer.example/owner": "kept",
}


def label_group(client: Client, declarations: dict[str, str]) -> dict[str, str]:
    """Every node of the cluster as its labelled node group creates it."""

    labels = {"eks.amazonaws.com/nodegroup": "gpu-ng", **declarations}
    for node in client.nodes:
        node["metadata"]["labels"] = dict(labels)
    return labels


def replacement(client: Client) -> dict[str, Any]:
    """The node the fleet has no Agent record for (the spot replacement)."""

    return client.nodes[1]


def test_the_node_group_declarations_are_the_only_exempt_labels() -> None:
    assert frozenset({OWNERSHIP_LABEL, DCGM_HOSTENGINE_LABEL}) == (
        PATCH.NODE_GROUP_DECLARATION_LABELS
    ), "exactly the two labels an administrator declares on the node group"


@pytest.mark.parametrize("declarations", [OWNERSHIP_ONLY, HOST_DCGM])
def test_a_labelled_replacement_with_unfinished_installer_annotations_is_an_orphan(
    declarations: dict[str, str],
) -> None:
    client, state = Client(), document()
    labels = label_group(client, declarations)
    node = replacement(client)
    node["metadata"]["annotations"] = dict(INSTALLER_FAILED)

    assert KUBE.node_targets(client, state, CONTEXT, "gpu-a") == {
        "managed-node": "uid-managed-node"
    }, "the fleet node is the only cleanup target"
    assert state["orphaned_installer_nodes"] == {
        CONTEXT: {"customer-node": "uid-customer-node"}
    }, "the labelled replacement is journaled as an orphaned installer node"
    assert client.mutations() == [], "the capture writes nothing"

    KUBE.clear_node_metadata(client, state, CONTEXT, "gpu-a")

    assert node["metadata"]["annotations"] == {"customer.example/owner": "kept"}, (
        "only the installer annotations are removed"
    )
    for item in client.nodes:
        assert item["metadata"]["labels"] == labels, (
            f"{item['metadata']['name']}: the node-group declarations stay; "
            "OWNERSHIP_RELEASED releases the ownership label from the node group"
        )


def test_a_labelled_node_without_an_agent_or_product_state_is_left_alone() -> None:
    """A replacement the node installer reconciler has not reached yet."""

    client, state = Client(), document()
    label_group(client, HOST_DCGM)
    untouched = copy.deepcopy(replacement(client))

    assert KUBE.node_targets(client, state, CONTEXT, "gpu-a") == {
        "managed-node": "uid-managed-node"
    }, "a node the product never installed is no cleanup target"
    assert state["orphaned_installer_nodes"] == {CONTEXT: {}}, "and no orphan"

    KUBE.clear_node_metadata(client, state, CONTEXT, "gpu-a")

    assert client.mutations() == [], "nothing of ours is on either node"
    assert replacement(client) == untouched, "the replacement is not patched"


def test_a_failed_join_on_a_label_convention_cluster_has_nothing_to_clean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The join failed before its node runtime rollout: no fleet record for
    the cluster, every GPU node labelled. The rollback's cleanup records an
    empty, proven target set and skips the node cleanup."""

    client, state = Client(), document()
    label_group(client, HOST_DCGM)
    state["fleet_snapshot"] = []
    monkeypatch.setattr(KUBE, "CleanupClient", lambda _config, _context: client)
    state_path = tmp_path / "state.json"

    KUBE.run_node_cleanup(
        {},
        state,
        state_path,
        context=CONTEXT,
        cluster_id="gpu-a",
        image="example.invalid/cleanup@sha256:" + "0" * 64,
        mode="uninstall",
        host_script_b64="",
    )

    assert state["node_targets"] == {CONTEXT: {}}, "the empty target set is recorded"
    record = state["node_cleanup"][CONTEXT]
    assert record["status"] == "SKIPPED", record
    assert "no node runtime was installed" in record["reason"], record
    assert client.mutations() == [], "no DaemonSet for an empty target set"


def spare_label(node: dict[str, Any]) -> None:
    node["metadata"]["labels"]["gpu-fault.io/spare"] = "true"


def unknown_label(node: dict[str, Any]) -> None:
    node["metadata"]["labels"]["gpu-fault.io/managed"] = "true"


def taint(key: str) -> Any:
    def add(node: dict[str, Any]) -> None:
        node["spec"] = {"taints": [{"key": key, "effect": "NoSchedule"}]}

    return add


def annotation(key: str, value: str = "1") -> Any:
    def add(node: dict[str, Any]) -> None:
        node["metadata"]["annotations"] = {key: value}

    return add


def installer_succeeded(node: dict[str, Any]) -> None:
    node["metadata"]["annotations"] = {
        **INSTALLER_FAILED,
        "gpu-fault.io/installer-state": "Succeeded",
    }


PRODUCT_STATE = {
    "spare-label": spare_label,
    "unknown-label": unknown_label,
    "quarantine-taint": taint("gpu-fault.io/quarantined"),
    "quiesce-taint": taint("gpu-fault.io/quiesce"),
    "node-key-annotation": annotation("gpu-fault.io/node-key-activation"),
    "installer-succeeded": installer_succeeded,
    "ownership-key-as-annotation": annotation(OWNERSHIP_LABEL, SITE_ID),
    "ownership-key-as-taint": taint(OWNERSHIP_LABEL),
}


@pytest.mark.parametrize("variant", sorted(PRODUCT_STATE))
def test_product_state_on_a_labelled_node_without_an_agent_still_blocks(
    variant: str,
) -> None:
    client, state = Client(), document()
    label_group(client, HOST_DCGM)
    PRODUCT_STATE[variant](replacement(client))

    with pytest.raises(KUBE.CleanupStateError, match="unregistered node runtime"):
        KUBE.node_targets(client, state, CONTEXT, "gpu-a")

    assert "orphaned_installer_nodes" not in state, "the node is no orphan"
    assert client.mutations() == [], "the refusal writes nothing"


@pytest.mark.parametrize("key", ["gpu-fault.io/quarantined", "gpu-fault.io/quiesce"])
def test_a_product_taint_alone_on_a_node_without_an_agent_blocks(key: str) -> None:
    """A taint is installed state as much as a label or an annotation: the
    cleanup never removes it, so a node without an Agent keeping one would
    be left tainted by an uninstall that reported success."""

    client, state = Client(), document()
    taint(key)(replacement(client))

    with pytest.raises(KUBE.CleanupStateError, match="unregistered node runtime"):
        KUBE.node_targets(client, state, CONTEXT, "gpu-a")
    assert client.mutations() == [], "the refusal writes nothing"


@pytest.mark.parametrize("variant", ["quiesce-taint", "spare-label", "unknown-label"])
def test_product_state_without_fleet_records_is_missing_proof_not_an_empty_set(
    variant: str,
) -> None:
    client, state = Client(), document()
    label_group(client, OWNERSHIP_ONLY)
    state["fleet_snapshot"] = []
    PRODUCT_STATE[variant](replacement(client))

    with pytest.raises(KUBE.CleanupStateError, match="no proven fleet targets"):
        KUBE.node_targets(client, state, CONTEXT, "gpu-a")
    assert "node_targets" not in state, "nothing is recorded on a refusal"


class LostAcknowledgement(RacingClient):
    def run(self, *arguments: str, payload: dict[str, Any] | None = None) -> str:
        result = super().run(*arguments, payload=payload)
        if arguments[:2] == ("patch", "node"):
            # Applied server-side; the client ran out of time first.
            raise FAILURE.CleanupCommandError(arguments, None)
        return result


def test_a_labelled_orphan_whose_patch_lost_its_acknowledgement_is_done() -> None:
    """The fresh read shows the installer annotations gone and only the
    node-group declarations left: nothing of ours remains, no refusal."""

    client, state = LostAcknowledgement(), document()
    labels = label_group(client, HOST_DCGM)
    node = replacement(client)
    node["metadata"]["annotations"] = dict(INSTALLER_FAILED)

    KUBE.clear_node_metadata(client, state, CONTEXT, "gpu-a")

    assert len(client.patches("customer-node")) == 1, "no second patch"
    assert node["metadata"]["annotations"] == {"customer.example/owner": "kept"}, (
        "the applied patch stands"
    )
    assert node["metadata"]["labels"] == labels, "the declarations stay"


def test_a_labelled_orphan_that_raced_a_kubelet_write_is_cleaned_on_the_fresh_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, state = RacingClient(), document()
    label_group(client, OWNERSHIP_ONLY)
    node = replacement(client)
    node["metadata"]["annotations"] = dict(INSTALLER_FAILED)
    client.writers["customer-node"] = [kubelet_status]
    monkeypatch.setattr(RETRY.time, "sleep", lambda _seconds: None)

    KUBE.clear_node_metadata(client, state, CONTEXT, "gpu-a")

    assert [patch[1]["value"] for patch in client.patches("customer-node")] == [
        "1",
        "2",
    ], "the retry is bound to the fresh resourceVersion"
    assert node["metadata"]["annotations"] == {"customer.example/owner": "kept"}, (
        "only the installer annotations are removed"
    )


def test_a_labelled_orphan_that_gained_a_runtime_during_the_race_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, state = RacingClient(), document()
    label_group(client, OWNERSHIP_ONLY)
    node = replacement(client)
    node["metadata"]["annotations"] = dict(INSTALLER_FAILED)

    def install(target: dict[str, Any]) -> None:
        installer_succeeded(target)
        bump(target)

    client.writers["customer-node"] = [install]
    monkeypatch.setattr(RETRY.time, "sleep", lambda _seconds: None)

    with pytest.raises(KUBE.CleanupStateError, match="unregistered node runtime"):
        KUBE.clear_node_metadata(client, state, CONTEXT, "gpu-a")

    assert len(client.patches("customer-node")) == 1, "no second patch"
    assert node["metadata"]["annotations"]["gpu-fault.io/installer-state"] == (
        "Succeeded"
    ), "an installed node keeps its installer record"
