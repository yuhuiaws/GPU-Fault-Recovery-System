"""The node evidence ``workflow-reconcile`` reads and the refusals around it.

A node document from ``kubectl get nodes -o json`` can carry metadata, spec,
taints or annotations in shapes the reconcile cannot read; those read as
"no evidence of that kind" rather than a traceback. The orphaned-annotation
strip refuses an incomplete node and a failed patch by name. The HyperPod
inventory refuses a site without a region, skips rows without an instance id,
refuses a page token that is not a string, and stops at the page limit
instead of paging forever.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import workflow_reconcile as reconcile
from gpu_fault.admin.bootstrap_common import BootstrapError

CLUSTER = "gpu-a"
NODE = "hyperpod-i-00000000000000001"
INCIDENT = "inc-0001"


def _site(tmp_path: Path, *, region: str | None = "us-west-2") -> SimpleNamespace:
    cluster: dict[str, object] = {
        "cluster_id": CLUSTER,
        "context": "gpu-a-context",
        "hyperpod_cluster_name": "gpu-a-hyperpod",
    }
    if region is not None:
        cluster["region"] = region
    return SimpleNamespace(
        release_config={
            "site_name": "staging",
            "aws_region": region or "",
            "cpu_eks_arn": "arn:aws:eks:us-west-2:123456789012:cluster/cpu-control",
            "clusters": [cluster],
        },
        environment={"KUBECONFIG": str(tmp_path / "gpu.kubeconfig")},
        source_sha256="a" * 64,
    )


def test_a_cluster_outside_the_site_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(BootstrapError, match="not in the managed site: gpu-z"):
        reconcile.cluster_target(_site(tmp_path), "gpu-z")


# ------------------------------------------------------------ node evidence


def test_unreadable_node_shapes_read_as_no_evidence(tmp_path: Path) -> None:
    inventory: dict[str, dict[str, Any]] = {
        "metadata-not-mapping": {"metadata": "x", "spec": "y"},
        "annotations-not-mapping": {
            "metadata": {"annotations": ["list"]},
            "spec": {"taints": "not-a-list", "unschedulable": True},
        },
        "taint-items-not-mappings": {
            "metadata": {"annotations": {}},
            "spec": {"taints": ["bare", {"key": "other", "value": "v"}]},
        },
    }

    evidence = reconcile.node_isolation_evidence(
        _site(tmp_path), CLUSTER, list(inventory) + ["absent-node"], inventory=inventory
    )

    by_node = {item["node_id"]: item for item in evidence}
    assert by_node["metadata-not-mapping"] == {
        "node_id": "metadata-not-mapping",
        "exists": True,
        "unschedulable": False,
        "quarantine_taint_value": None,
        "isolation_annotations": {},
    }
    assert by_node["annotations-not-mapping"]["unschedulable"] is True
    assert by_node["annotations-not-mapping"]["isolation_annotations"] == {}
    assert by_node["taint-items-not-mappings"]["quarantine_taint_value"] is None
    assert by_node["absent-node"]["exists"] is False


def test_the_quarantine_taint_value_is_the_first_matching_taint(tmp_path: Path) -> None:
    inventory = {
        NODE: {
            "metadata": {"annotations": {reconcile.ISOLATION_ANNOTATIONS[0]: INCIDENT}},
            "spec": {
                "taints": [
                    {"key": reconcile.QUARANTINE_TAINT, "value": INCIDENT},
                    {"key": reconcile.QUARANTINE_TAINT, "value": "later"},
                ]
            },
        }
    }

    [item] = reconcile.node_isolation_evidence(
        _site(tmp_path), CLUSTER, [NODE], inventory=inventory
    )

    assert item["quarantine_taint_value"] == INCIDENT
    assert item["isolation_annotations"] == {
        reconcile.ISOLATION_ANNOTATIONS[0]: INCIDENT
    }


# -------------------------------------------------------- annotation strip


def _restored_node(**metadata_overrides: object) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "name": NODE,
        "uid": "node-uid",
        "resourceVersion": "42",
        "annotations": {
            reconcile.ISOLATION_ANNOTATIONS[0]: INCIDENT,
            reconcile.ISOLATION_ANNOTATIONS[1]: "fence-1",
        },
    }
    metadata.update(metadata_overrides)
    return {"metadata": metadata, "spec": {"unschedulable": False, "taints": []}}


def test_an_incomplete_node_cannot_have_its_annotations_stripped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        reconcile, "run_command", lambda command, **_k: calls.append(command)
    )

    with pytest.raises(BootstrapError, match="node is incomplete"):
        reconcile.strip_node_isolation_annotations(
            _site(tmp_path),
            CLUSTER,
            {"metadata": "not-a-mapping"},
            incident_id=INCIDENT,
            expected_node=_restored_node(),
        )
    assert calls == [], "nothing is patched before the node is proven"


def test_a_failed_patch_names_the_node_and_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(command: list[str], **_options: object) -> SimpleNamespace:
        assert command[-7:-2] == ["patch", "node", NODE, "--type", "merge"], command
        return SimpleNamespace(returncode=1, stdout="", stderr="conflict")

    monkeypatch.setattr(reconcile, "run_command", run)

    with pytest.raises(BootstrapError, match=f"{NODE} on {CLUSTER}: ") as info:
        reconcile.strip_node_isolation_annotations(
            _site(tmp_path),
            CLUSTER,
            _restored_node(),
            incident_id=INCIDENT,
            expected_node=_restored_node(),
        )
    assert "conflict" not in str(info.value), "kubectl stderr is treated as sensitive"


def test_a_dry_run_strip_reports_the_patch_without_issuing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a dry run must not run kubectl")

    monkeypatch.setattr(reconcile, "run_command", refuse)

    result = reconcile.strip_node_isolation_annotations(
        _site(tmp_path),
        CLUSTER,
        _restored_node(),
        incident_id=INCIDENT,
        expected_node=_restored_node(),
        dry_run=True,
    )

    assert result == {
        "node_id": NODE,
        "node_uid": "node-uid",
        "resource_version": "42",
        "annotations": list(reconcile.ISOLATION_ANNOTATIONS),
    }


# ------------------------------------------------------ HyperPod inventory


def _page(summaries: list[object], token: object = None) -> SimpleNamespace:
    payload: dict[str, object] = {"ClusterNodeSummaries": summaries}
    if token is not None:
        payload["NextToken"] = token
    return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")


def test_a_hyperpod_cluster_without_a_region_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BootstrapError, match="has no AWS region"):
        reconcile.hyperpod_instance_ids(_site(tmp_path, region=None), CLUSTER)


def test_rows_without_an_instance_id_are_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = [
        _page(["not-a-mapping", {"InstanceId": 7}, {"InstanceId": " i-1 "}], "p2"),
        _page([{"InstanceId": ""}, {"InstanceId": "i-2"}]),
    ]
    commands: list[list[str]] = []

    def run(command: list[str], **_options: object) -> SimpleNamespace:
        commands.append(list(command))
        return pages.pop(0)

    monkeypatch.setattr(reconcile, "run_command", run)

    assert reconcile.hyperpod_instance_ids(_site(tmp_path), CLUSTER) == frozenset(
        {"i-1", "i-2"}
    )
    assert "--next-token" not in commands[0]
    assert commands[1][-2:] == ["--next-token", "p2"]


def test_a_page_token_that_is_not_a_string_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        reconcile, "run_command", lambda *_a, **_k: _page([{"InstanceId": "i-1"}], 5)
    )

    with pytest.raises(BootstrapError, match="repeats its page token"):
        reconcile.hyperpod_instance_ids(_site(tmp_path), CLUSTER)


def test_an_inventory_past_the_page_limit_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reconcile, "HYPERPOD_INVENTORY_PAGE_LIMIT", 3)
    tokens = iter(("a", "b", "c", "d"))
    monkeypatch.setattr(
        reconcile,
        "run_command",
        lambda *_a, **_k: _page([{"InstanceId": "i-1"}], next(tokens)),
    )

    with pytest.raises(BootstrapError, match="exceeds 3 pages"):
        reconcile.hyperpod_instance_ids(_site(tmp_path), CLUSTER)


@pytest.mark.parametrize(
    "stdout",
    [json.dumps({"ClusterNodeSummaries": "not-a-list"}), json.dumps(["bare"])],
    ids=["summaries-not-list", "payload-not-mapping"],
)
def test_an_inventory_without_a_summary_list_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    monkeypatch.setattr(
        reconcile,
        "run_command",
        lambda *_a, **_k: SimpleNamespace(returncode=0, stdout=stdout, stderr=""),
    )

    with pytest.raises(BootstrapError, match=f"inventory of {CLUSTER} is invalid$"):
        reconcile.hyperpod_instance_ids(_site(tmp_path), CLUSTER)
