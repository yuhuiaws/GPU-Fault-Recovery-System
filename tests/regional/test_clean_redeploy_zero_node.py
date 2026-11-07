"""uninstall and remove-cluster cleanup of a GPU cluster without nodes, end to end.

The real ``prepare-clean-redeploy.sh`` with the real ``cleanup_kubernetes.py``,
``cleanup_zero_nodes.py`` and the release engine's zero-node proof, over a fake
Kubernetes API (``test_clean_redeploy_script.FAKE_KUBECTL_CLEANUP``) and a fake
``aws``: the eks-managed cluster's GPU nodes left, its only GPU managed node
group is ACTIVE at desiredSize 0 (parked) or still wants 2 (unavailable), the
fleet holds the retired record of one departed node and the ACTIVE record of
another the departed-node cleanup has not retired yet. Before, the PREFLIGHT
capture refused "node cleanup has no proven fleet targets" and the uninstall
journal stayed open.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.regional._parked_cluster_support import OWNERSHIP, cpu_nodegroup, nodegroup
from tests.regional.test_clean_redeploy_script import (
    FAKE_KUBECTL_CLEANUP,
    CleanupHarness,
    _reset_harness,
    run_script,
)

CLUSTER_ARN = "arn:aws:eks:us-west-2:123456789012:cluster/gpu-a"
DEPARTED = "ip-10-0-33-205.us-west-2.compute.internal"
DEPARTED_INSTANCE = "i-00000000000000002"

# The selector reads (``get nodes -l``) and the Karpenter NodePools of the
# zero-node proof; everything else is the shared fake Kubernetes API.
FAKE_KUBECTL_PROOF = r"""#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
if "get" in args:
    kind = args[args.index("get") + 1]
    if kind.startswith("nodepools"):
        print('error: the server doesn\'t have a resource type "nodepools"', file=sys.stderr)
        raise SystemExit(1)
    if kind == "nodes" and "-l" in args:
        context = args[args.index("--context") + 1]
        key, _, value = args[args.index("-l") + 1].partition("=")
        api = json.loads(Path(os.environ["FAKE_API_STATE"]).read_text())
        with Path(os.environ["FAKE_KUBECTL_LOG"]).open("a") as stream:
            stream.write(json.dumps({"args": args, "phase": None, "completed": []}) + "\n")
        items = [
            item
            for identity, item in api["objects"].items()
            if json.loads(identity)[0] == context
            and json.loads(identity)[2] == "node"
            and (item["metadata"].get("labels") or {}).get(key) == value
        ]
        print(json.dumps({"items": items}))
        raise SystemExit(0)
os.execv(sys.executable, [sys.executable, os.environ["FAKE_KUBECTL_BASE"], *args])
"""

FAKE_AWS = r"""#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
with Path(os.environ["FAKE_AWS_LOG"]).open("a") as stream:
    stream.write(json.dumps(args) + "\n")
provider = json.loads(Path(os.environ["FAKE_PROVIDER"]).read_text())
if args[:2] == ["ec2", "describe-instances"]:
    print(json.dumps(provider["instances"]))
elif args[:2] == ["eks", "list-nodegroups"]:
    print(json.dumps({"nodegroups": sorted(provider["nodegroups"])}))
elif args[:2] == ["eks", "describe-nodegroup"]:
    print(json.dumps(provider["nodegroups"][args[args.index("--nodegroup-name") + 1]]))
else:
    raise SystemExit("unexpected aws command")
"""


def _zero_node_harness(
    tmp_path: Path, *, desired: int = 0, instances: list[list[str]] | None = None
) -> CleanupHarness:
    harness = _reset_harness(tmp_path)
    config = json.loads(harness.config.read_text())
    config["aws_region"] = "us-west-2"
    config["clusters"][0].update(
        provider="eks-managed",
        region="us-west-2",
        node_selector=OWNERSHIP,
        eks_cluster_arn=CLUSTER_ARN,
    )
    harness.config.write_text(json.dumps(config))
    api = json.loads(harness.api.read_text())
    api["interrupt_node"] = False
    # The cluster's only GPU node left; its CPU node stays, nothing of ours on it.
    del api["objects"][json.dumps(["gpu-a-context", "", "node", "node-a"])]
    api["fleet"][0]["lifecycle_state"] = "REVOKED"
    api["fleet"].append(
        {
            "cluster_id": "gpu-a",
            "node_id": DEPARTED,
            "node_instance_id": DEPARTED_INSTANCE,
            "lifecycle_state": "ACTIVE",
            "installed_unit_inventory": {"units": ["gpu-fault-node-agent.service"]},
        }
    )
    harness.api.write_text(json.dumps(api))
    fake_bin = tmp_path / "bin"
    base = fake_bin / "kubectl-base"
    base.write_text(FAKE_KUBECTL_CLEANUP, encoding="utf-8")
    base.chmod(0o755)
    kubectl = fake_bin / "kubectl"
    kubectl.write_text(FAKE_KUBECTL_PROOF, encoding="utf-8")
    kubectl.chmod(0o755)
    aws = fake_bin / "aws"
    aws.write_text(FAKE_AWS, encoding="utf-8")
    aws.chmod(0o755)
    provider = tmp_path / "provider.json"
    provider.write_text(
        json.dumps(
            {
                "instances": [] if instances is None else instances,
                "nodegroups": {
                    "gpu-group": nodegroup("gpu-group", desired=desired),
                    "cpu-group": cpu_nodegroup(),
                },
            }
        ),
        encoding="utf-8",
    )
    harness.env.update(
        FAKE_KUBECTL_BASE=str(base),
        FAKE_AWS_LOG=str(tmp_path / "aws.jsonl"),
        FAKE_PROVIDER=str(provider),
    )
    return harness


def _aws_calls(tmp_path: Path) -> list[list[str]]:
    path = tmp_path / "aws.jsonl"
    if not path.exists():
        return []
    return [json.loads(line)[:2] for line in path.read_text().splitlines()]


def _completed(harness: CleanupHarness) -> dict[str, Any]:
    document = json.loads(harness.state.read_text())
    assert (document["phase"], document["status"]) == (
        "CLEANUP_COMPLETED",
        "COMPLETED",
    ), document["history"]
    return document


def test_an_uninstall_reset_over_a_parked_cluster_completes(tmp_path: Path) -> None:
    harness = _zero_node_harness(tmp_path)

    result = harness.run()

    assert result.returncode == 0, result.stdout + result.stderr
    document = _completed(harness)
    entry = document["zero_node_clusters"]["gpu-a"]
    assert entry["state"] == "parked", entry
    assert entry["retired_agents"] == ["node-a"], entry
    assert entry["departed_agents"] == [DEPARTED], entry
    assert [group["name"] for group in entry["groups"]] == ["gpu-group"], entry
    assert document["node_targets"]["gpu:gpu-a-context"] == {}, "no node target"
    cleanup = document["node_cleanup"]["gpu:gpu-a-context"]
    assert cleanup["status"] == "SKIPPED", cleanup
    assert "GPU cluster without nodes (parked)" in cleanup["reason"], cleanup
    assert (
        "gpu-a: GPU cluster without nodes (parked): no node selected; every managed "
        "node group the node selector covers is ACTIVE at desiredSize 0"
    ) in result.stdout, result.stdout
    assert "(1 retired and 1 departed fleet record(s))" in result.stdout, result.stdout
    assert f"gpu-a: departed fleet node {DEPARTED}" in result.stdout, result.stdout
    assert not any("create" in call["args"] for call in harness.calls()), (
        "no node cleanup DaemonSet for a cluster without nodes"
    )
    assert ["eks", "list-nodegroups"] in _aws_calls(tmp_path), (
        "the provider was asked why the cluster has no node"
    )


def test_a_remove_cluster_cleanup_over_an_unavailable_cluster_completes(
    tmp_path: Path,
) -> None:
    """``remove-cluster``'s own invocation (``--scope gpu --mode clean``) of a
    cluster whose group still wants two nodes: unavailable, and still nothing
    to clean on a node."""

    harness = _zero_node_harness(tmp_path, desired=2)

    result = run_script(
        "--config",
        str(harness.config),
        "--scope",
        "gpu",
        "--cluster-id",
        "gpu-a",
        "--mode",
        "clean",
        "--node-mode",
        "uninstall",
        "--state-file",
        str(harness.state),
        "--timeout-seconds",
        "30",
        "--execute",
        env=harness.env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    document = _completed(harness)
    entry = document["zero_node_clusters"]["gpu-a"]
    assert entry["state"] == "unavailable", entry
    assert "wants 2 node(s)" in entry["reason"], entry
    assert document["node_cleanup"]["gpu:gpu-a-context"]["status"] == "SKIPPED"
    assert "GPU cluster without nodes (unavailable)" in result.stdout, result.stdout


@pytest.mark.parametrize("state", ["running", "stopped"])
def test_a_departed_record_whose_instance_ec2_still_has_refuses_before_any_stop(
    tmp_path: Path, state: str
) -> None:
    """The record not yet retired names an instance EC2 still has: the node
    runtime may still run outside Kubernetes, so the capture refuses and
    nothing is stopped."""

    harness = _zero_node_harness(tmp_path, instances=[[DEPARTED_INSTANCE, state]])

    result = harness.run()

    assert result.returncode != 0, result.stdout + result.stderr
    assert (
        f"an active fleet node is missing: {DEPARTED}; EC2 instance "
        f"{DEPARTED_INSTANCE} is {state}"
    ) in result.stderr, result.stderr
    document = json.loads(harness.state.read_text())
    assert (document["phase"], document["status"]) == ("PREFLIGHT", "FAILED"), document[
        "history"
    ]
    assert "zero_node_clusters" not in document, "a refusal records no proof"
    assert not any(
        "scale" in call["args"] or "delete" in call["args"] for call in harness.calls()
    ), "a refused capture stops nothing"
