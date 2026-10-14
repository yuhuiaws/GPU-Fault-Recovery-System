"""The RESTART_NODE notice names the provider that rebooted the node.

Live (GF-REGIONAL-DESTR-001 attempt 2, plain EKS): the EKS adapter rebooted
the node with EC2 RebootInstances, and the notice read "HyperPod action:
BatchRebootClusterNodes" and "HyperPod status: Running". The provider lines now
come from the lifecycle adapter's provider: HyperPod keeps the template's text
byte for byte, a plain EKS node names the EC2 reboot of its managed node group
instance and the EC2 instance state, and a provider the table does not know
gets wording that claims nothing.
"""

from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.notifications import RestartGuardEmailBuilder

HYPERPOD_SUBJECT = "[NOTICE][GPU node restarted automatically] hp-cluster: worker-1"
HYPERPOD_BODY = (
    "GPU node restarted automatically\n"
    "\n"
    "1. Event\n"
    "- Cluster: hp-cluster\n"
    "- Incident: incident-1\n"
    "- Workflow: workflow-1\n"
    "- Triggering event: kernel-log-1\n"
    "- Fault type: XID\n"
    "- Fault identifier: XID 79\n"
    "- Collector source: KERNEL_LOG\n"
    "- Policy source: NVIDIA_CATALOG\n"
    "- Policy action: RESTART_BM\n"
    "- Effective action: REBOOT_NODE\n"
    "- Restart reasons:\n"
    "  - exact NVIDIA Catalog 610 Immediate Action for XID 79\n"
    "- Affected nodes: worker-1\n"
    "\n"
    "2. Restart\n"
    "- System action: RESTART_NODE\n"
    "- HyperPod action: BatchRebootClusterNodes\n"
    "- HyperPod operation: hyperpod-op-1\n"
    "- Restarted nodes:\n"
    "- Node: worker-1; InstanceId: i-123; NodeLogicalId: logical-1; "
    "boot ID before restart: boot-a; boot ID after restart: boot-b; "
    "Agent incarnation: incarnation-b; HyperPod status: Running\n"
    "- Restart confirmed by: hyperpod-running-and-new-agent-incarnation\n"
    "- Restart status: confirmed complete\n"
    "\n"
    "3. Follow-up\n"
    "- GPU validation: the workflow's VALIDATE_GPU step\n"
    "- Fabric validation: the workflow's VALIDATE_FABRIC step\n"
    "- Scheduling: restored by the RESTORE_SCHEDULING step after validation passes\n"
    "\n"
    "Template: restart-node-zh-v1\n"
)


def _notice(**overrides: Any):
    arguments: dict[str, Any] = {
        "cluster_id": "hp-cluster",
        "incident_id": "incident-1",
        "workflow_id": "workflow-1",
        "event_id": "kernel-log-1",
        "event_type": "XID",
        "xid": 79,
        "policy_source": "NVIDIA_CATALOG",
        "official_action": "RESTART_BM",
        "effective_action": "REBOOT_NODE",
        "reasons": ["exact NVIDIA Catalog 610 Immediate Action for XID 79"],
        "operation_id": "hyperpod-op-1",
        "node_ids": ["worker-1"],
        "source_boot_id": "boot-a",
        "agent_baselines": {
            "worker-1": {"boot_id": "boot-a", "agent_incarnation_id": "incarnation-a"}
        },
        "agent_observations": [
            {
                "node_id": "worker-1",
                "boot_id": "boot-b",
                "agent_incarnation_id": "incarnation-b",
            }
        ],
        "provider_observations": [
            {
                "node_id": "worker-1",
                "node_logical_id": "logical-1",
                "instance_id": "i-123",
                "status": "Running",
            }
        ],
        "confirmation_source": "hyperpod-running-and-new-agent-incarnation",
        "event_source": "KERNEL_LOG",
    }
    arguments.update(overrides)
    return RestartGuardEmailBuilder().build_node_restarted(**arguments)


@pytest.mark.parametrize("provider", [None, "HyperPod"])
def test_the_hyperpod_notice_keeps_its_text(provider: str | None) -> None:
    notice = _notice() if provider is None else _notice(provider_label=provider)

    assert notice.subject == HYPERPOD_SUBJECT
    assert notice.body_text == HYPERPOD_BODY, "HyperPod's rendering is unchanged"


def test_an_eks_notice_names_the_ec2_reboot_and_the_instance_state() -> None:
    notice = _notice(
        cluster_id="gpu-eks-cluster",
        node_ids=["eks-gpu-node-a"],
        operation_id="remote/remote-4d3bb8076064a4b9d6d9a3c9",
        agent_observations=[
            {
                "node_id": "eks-gpu-node-a",
                "boot_id": "boot-b",
                "agent_incarnation_id": "incarnation-b",
            }
        ],
        provider_observations=[
            {
                "node_id": "eks-gpu-node-a",
                "node_logical_id": "eks-gpu-node-a",
                "instance_id": "i-eks-node-a",
                "status": "Running",
            }
        ],
        confirmation_source="eks-running-and-new-agent-incarnation",
        provider_label="EKS",
    )

    assert notice.subject == (
        "[NOTICE][GPU node restarted automatically] gpu-eks-cluster: eks-gpu-node-a"
    )
    body = notice.body_text
    assert (
        "- EKS action: EC2 RebootInstances on the managed node group instance\n" in body
    )
    assert "- EKS operation: remote/remote-4d3bb8076064a4b9d6d9a3c9\n" in body
    assert "InstanceId: i-eks-node-a;" in body
    assert body.count("; EC2 instance state: Running\n") == 1
    assert "- Restart confirmed by: eks-running-and-new-agent-incarnation" in body
    for text in (notice.subject, body):
        assert "HyperPod" not in text, text
        assert "BatchRebootClusterNodes" not in text, text


def test_a_provider_the_table_does_not_name_gets_neutral_wording() -> None:
    notice = _notice(provider_label="SomethingElse")

    assert "- Provider action: node reboot\n" in notice.body_text
    assert "- Provider operation: hyperpod-op-1\n" in notice.body_text
    assert "; Provider status: Running\n" in notice.body_text
    assert "BatchRebootClusterNodes" not in notice.body_text
    assert "HyperPod status" not in notice.body_text
