"""The provider proof of ``workflow_reconcile_departed_nodes`` at its edges.

A plain EKS node Kubernetes no longer has is restored only when EC2 answers
``terminated`` (or nothing) for the instance id the fleet recorded. These tests
pin the refusals around that read from the administrator side: the Agent-record
answer must be a mapping, EC2's answer must be a list of ``[id, state]`` rows,
a failed or unreadable lookup is cached per cluster and reported as a blocker
for every node of it (never as "absent"), a cluster without a region cannot be
asked, and a HyperPod listing that is not JSON is refused by name.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import workflow_reconcile_departed_nodes as departed_nodes
from gpu_fault.admin.bootstrap_common import BootstrapError

CLUSTER = "gpu-eks"
REGION = "us-west-2"
GONE = "i-00000000000000001"
LIVE = "i-00000000000000002"
MISSING = {
    "node_id": "ip-10-0-1-1.us-west-2.compute.internal",
    "exists": False,
    "restored": False,
    "blockers": ["node is missing from Kubernetes"],
}


class Ec2:
    """``run_command`` double for ``aws ec2 describe-instances`` by filter."""

    def __init__(
        self, *, states: dict[str, str] | None = None, stdout: str | None = None
    ) -> None:
        self.states = states or {}
        self.stdout = stdout
        self.returncode = 0
        self.stderr = ""
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **options: Any) -> SimpleNamespace:
        self.commands.append(list(command))
        assert options == {"timeout_seconds": 120}, options
        if self.stdout is not None:
            stdout = self.stdout
        else:
            wanted = command[command.index("--filters") + 1].split("=", 2)[-1]
            stdout = json.dumps(
                [
                    [instance, self.states[instance]]
                    for instance in wanted.split(",")
                    if instance in self.states
                ]
            )
        return SimpleNamespace(
            returncode=self.returncode, stdout=stdout, stderr=self.stderr
        )


def evidence(
    run_command: Ec2,
    *,
    instance_id: str | None,
    states: dict[str, dict[str, str]] | None = None,
    read_at: dict[str, str] | None = None,
    target: dict[str, Any] | None = None,
    aws_region: object = "",
    node_id: str = MISSING["node_id"],
) -> dict[str, Any]:
    return departed_nodes.departed_node_evidence(
        target if target is not None else {"cluster_id": CLUSTER, "region": REGION},
        CLUSTER,
        node_id,
        instance_id=instance_id,
        aws_region=aws_region,
        missing={**MISSING, "node_id": node_id},
        states={} if states is None else states,
        read_at=read_at,
        run_command=run_command,
    )


# ----------------------------------------------------- recorded instance ids


def test_recorded_instance_ids_ask_once_for_the_sorted_unique_nodes() -> None:
    calls: list[tuple[object, dict[str, Any], str]] = []
    site = object()

    def run_script(site_arg: object, payload: dict[str, Any], *, script: str):
        calls.append((site_arg, payload, script))
        return {
            "mode": "agent-instance-ids",
            "instance_ids": {"node-b": f" {GONE} ", "node-a": "uid-not-an-ec2-id"},
        }

    result = departed_nodes.recorded_instance_ids(
        site, CLUSTER, ["node-b", "node-a", "node-b"], run_script=run_script
    )

    assert result == {"node-b": GONE, "node-a": None}, (
        "a Node UID recorded by a HyperPod-era Agent must not read as an EC2 id"
    )
    assert calls == [
        (
            site,
            {
                "mode": "agent-instance-ids",
                "cluster_id": CLUSTER,
                "node_ids": ["node-a", "node-b"],
            },
            departed_nodes.AGENT_INSTANCE_SCRIPT,
        )
    ]


@pytest.mark.parametrize("answer", [{}, {"instance_ids": ["node-a"]}])
def test_recorded_instance_ids_refuse_an_answer_without_a_mapping(
    answer: dict[str, Any],
) -> None:
    with pytest.raises(BootstrapError, match="agent instance ids .* are invalid"):
        departed_nodes.recorded_instance_ids(
            object(), CLUSTER, ["node-a"], run_script=lambda *_a, **_k: answer
        )


# --------------------------------------------------------- EC2 instance states


def test_blank_instance_ids_are_never_sent_to_ec2() -> None:
    ec2 = Ec2()

    assert departed_nodes.ec2_instance_states(REGION, ["", "  "], run_command=ec2) == {}
    assert ec2.commands == [], "an empty filter would describe every instance"


def test_ec2_states_are_read_by_one_filter_and_normalized() -> None:
    ec2 = Ec2(states={GONE: " Terminated ", LIVE: "RUNNING"})

    states = departed_nodes.ec2_instance_states(
        REGION, [LIVE, GONE, LIVE], run_command=ec2
    )

    assert states == {GONE: "terminated", LIVE: "running"}
    assert len(ec2.commands) == 1
    assert ec2.commands[0][ec2.commands[0].index("--filters") + 1] == (
        f"Name=instance-id,Values={GONE},{LIVE}"
    )


def test_a_failed_describe_names_the_region_but_redacts_the_output() -> None:
    ec2 = Ec2()
    ec2.returncode = 254
    ec2.stderr = "An error occurred (AccessDenied) for account secret-detail"

    with pytest.raises(BootstrapError) as raised:
        departed_nodes.ec2_instance_states(REGION, [GONE], run_command=ec2)

    message = str(raised.value)
    assert message.startswith(f"cannot describe EC2 instances in {REGION}: "), message
    assert "AccessDenied: <sensitive output redacted>" in message
    assert "secret-detail" not in message, "raw AWS output leaked into the error"


@pytest.mark.parametrize(
    ("stdout", "reason"),
    [
        ("not json", "is invalid JSON"),
        (json.dumps({"Reservations": []}), "is invalid$"),
        (json.dumps([[GONE]]), "is invalid$"),
        (json.dumps([[GONE, 7]]), "is invalid$"),
        (json.dumps([f"{GONE} running"]), "is invalid$"),
    ],
    ids=["not-json", "not-a-list", "short-row", "state-not-text", "row-not-list"],
)
def test_an_unreadable_ec2_answer_is_refused(stdout: str, reason: str) -> None:
    with pytest.raises(BootstrapError, match=f"EC2 instance description {reason}"):
        departed_nodes.ec2_instance_states(
            REGION, [GONE], run_command=Ec2(stdout=stdout)
        )


# ------------------------------------------------------- departed-node evidence


@pytest.mark.parametrize("recorded", [None, "", "uid-1234", "i-XYZ"])
def test_no_usable_recorded_instance_id_keeps_the_node_blocked(
    recorded: str | None,
) -> None:
    ec2 = Ec2(states={GONE: "terminated"})

    result = evidence(ec2, instance_id=recorded)

    assert result["restored"] is False
    assert result["provider"] == departed_nodes.EKS_PROVIDER
    assert result["instance_id"] == recorded
    assert result["blockers"] == [
        *MISSING["blockers"],
        departed_nodes.NO_RECORDED_INSTANCE_BLOCKER,
    ]
    assert ec2.commands == [], "EC2 was asked about an id that is not an instance"


def test_a_terminated_instance_restores_and_stamps_the_ec2_read() -> None:
    ec2 = Ec2(states={GONE: "terminated"})
    read_at: dict[str, str] = {}
    states: dict[str, dict[str, str]] = {}

    first = evidence(ec2, instance_id=GONE, states=states, read_at=read_at)
    again = evidence(ec2, instance_id=GONE, states=states, read_at=read_at)

    assert (
        first
        == again
        == {
            "node_id": MISSING["node_id"],
            "exists": False,
            "provider": departed_nodes.EKS_PROVIDER,
            "instance_id": GONE,
            "ec2_state": departed_nodes.TERMINATED,
            "absent_from_provider": True,
            "restored": True,
            "blockers": [],
        }
    )
    assert len(ec2.commands) == 1, "the cached state was read from EC2 again"
    assert set(read_at) == {departed_nodes.EC2_SOURCE}
    assert states == {CLUSTER: {GONE: departed_nodes.TERMINATED}}


def test_an_instance_unknown_to_ec2_is_proof_and_needs_no_read_stamp() -> None:
    ec2 = Ec2(states={})

    result = evidence(ec2, instance_id=GONE, read_at=None)

    assert result["restored"] is True
    assert result["ec2_state"] == departed_nodes.NOT_FOUND
    assert result["blockers"] == []


def test_a_running_instance_keeps_the_node_blocked_with_its_state() -> None:
    result = evidence(Ec2(states={LIVE: "running"}), instance_id=LIVE)

    assert result["restored"] is False
    assert result["absent_from_provider"] is False
    assert result["ec2_state"] == "running"
    assert result["blockers"] == [
        f"node is missing from Kubernetes but EC2 instance {LIVE} is running"
    ]


def test_the_site_region_is_used_when_the_cluster_names_none() -> None:
    ec2 = Ec2(states={GONE: "terminated"})

    result = evidence(
        ec2, instance_id=GONE, target={"cluster_id": CLUSTER}, aws_region=" eu-west-1 "
    )

    assert result["restored"] is True
    assert ec2.commands[0][ec2.commands[0].index("--region") + 1] == "eu-west-1"


def test_a_cluster_without_any_region_is_a_blocker_not_a_lookup() -> None:
    ec2 = Ec2(states={GONE: "terminated"})
    read_at: dict[str, str] = {}

    result = evidence(
        ec2, instance_id=GONE, target={"cluster_id": CLUSTER}, read_at=read_at
    )

    assert result["restored"] is False
    assert result["blockers"][-1] == (
        "provider membership unknown: "
        f"GPU cluster {CLUSTER} has no AWS region in the site"
    )
    assert ec2.commands == [] and read_at == {}, "a region-less lookup was attempted"


def test_a_failed_lookup_blocks_every_node_of_the_cluster_with_one_call() -> None:
    ec2 = Ec2()
    ec2.returncode = 1
    ec2.stderr = "An error occurred (Throttling) when calling DescribeInstances"
    states: dict[str, dict[str, str]] = {}

    first = evidence(ec2, instance_id=GONE, states=states)
    second = evidence(
        ec2, instance_id=LIVE, states=states, node_id="ip-10-0-1-2.compute.internal"
    )

    for result in (first, second):
        assert result["restored"] is False
        assert result["blockers"][-1].startswith(
            f"provider membership unknown: cannot describe EC2 instances in {REGION}"
        ), result["blockers"]
    assert len(ec2.commands) == 1, "a failed cluster lookup was repeated per node"
    assert departed_nodes.LOOKUP_ERROR in states[CLUSTER]


# ------------------------------------------------------------ HyperPod listing


def test_a_hyperpod_listing_that_is_not_json_is_refused() -> None:
    site = SimpleNamespace(release_config={"aws_region": REGION})

    with pytest.raises(BootstrapError, match="inventory of gpu-hp is invalid JSON"):
        departed_nodes.hyperpod_instance_ids(
            site,
            {"cluster_id": "gpu-hp", "hyperpod_cluster_name": "hp"},
            "gpu-hp",
            run_command=lambda *_a, **_k: SimpleNamespace(
                returncode=0, stdout="<html>", stderr=""
            ),
        )


# ------------------------------------------------------- recorded ids per plan


def _targets(site: object, cluster_id: str) -> dict[str, Any]:
    provider = "hyperpod" if cluster_id == "gpu-hp" else departed_nodes.EKS_PROVIDER
    return {"cluster_id": cluster_id, "provider": provider}


def test_recorded_ids_are_read_once_per_eks_cluster_for_the_planned_nodes() -> None:
    payloads: list[dict[str, Any]] = []

    def run_script(_site: object, payload: dict[str, Any], *, script: str):
        payloads.append(payload)
        return {"instance_ids": {"node-a": GONE, "node-b": "uid-b"}}

    recorded = departed_nodes.RecordedInstanceIds(
        object(),
        [
            {"cluster_id": CLUSTER, "node_ids": ["node-b"]},
            {"cluster_id": CLUSTER, "node_ids": ["node-a"]},
            {"cluster_id": "gpu-hp", "node_ids": ["hyperpod-i-0000000000000000"]},
            {"cluster_id": CLUSTER},
        ],
        cluster_target=_targets,
        run_script=run_script,
    )

    assert recorded.get(CLUSTER, "node-a") == GONE
    assert recorded.get(CLUSTER, "node-b") is None
    assert recorded.get(CLUSTER, "node-z") is None
    assert recorded.get("gpu-hp", "hyperpod-i-0000000000000000") is None
    assert [item["node_ids"] for item in payloads] == [["node-a", "node-b"]], (
        "the Pod was asked more than once, or asked about a HyperPod cluster"
    )


def test_a_failed_agent_read_leaves_every_id_of_the_cluster_unknown() -> None:
    attempts: list[str] = []

    def run_script(_site: object, payload: dict[str, Any], *, script: str):
        attempts.append(payload["cluster_id"])
        raise BootstrapError("no Running CPU ingress Pod")

    recorded = departed_nodes.RecordedInstanceIds(
        object(),
        [{"cluster_id": CLUSTER, "node_ids": ["node-a", "node-b"]}],
        cluster_target=_targets,
        run_script=run_script,
    )

    assert recorded.get(CLUSTER, "node-a") is None
    assert recorded.get(CLUSTER, "node-b") is None
    assert attempts == [CLUSTER], "the failed read was retried for every node"
