"""``gpu-fault-admin site-id``: the ownership label value before the first deploy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import site_identity_command as verb
from gpu_fault.admin.bootstrap_common import BootstrapError, ClusterIdentity
from gpu_fault.admin.bootstrap_site import site_identifier
from gpu_fault.admin.deploy_host_binding import is_readonly_command
from gpu_fault.cluster_scope import OWNERSHIP_LABEL
from tests.admin.test_admin_site import site_file

CPU_HYPERPOD_ARN = "arn:aws:sagemaker:us-west-2:123456789012:cluster/fixturecpuhp01"
CPU_EKS_ARN = "arn:aws:eks:us-west-2:123456789012:cluster/control-plane-eks"


def _cpu(arn: str = CPU_EKS_ARN) -> ClusterIdentity:
    return ClusterIdentity(
        input_arn=arn,
        role="cpu",
        region="us-west-2",
        account_id="123456789012",
        hyperpod_arn=CPU_HYPERPOD_ARN,
        hyperpod_name="control-plane-fixture",
        eks_arn=CPU_EKS_ARN,
        eks_name="control-plane-eks",
        vpc_id="vpc-1",
        subnet_ids=("subnet-a",),
        node_recovery="None",
        context="cpu",
    )


def test_site_id_parses_without_a_state_dir_and_is_read_only() -> None:
    arguments = admin_cli.parser().parse_args(
        ["site-id", "--cpu-cluster-arn", CPU_EKS_ARN]
    )
    assert arguments.command == "site-id"
    assert arguments.state_dir is None, (
        "no state directory exists before the first deploy"
    )
    assert is_readonly_command(
        arguments, readonly_commands=admin_cli.READONLY_COMMANDS
    ), "site-id changes nothing"
    with pytest.raises(SystemExit):
        admin_cli.parser().parse_args(["site-id"])


def test_site_id_prints_the_deterministic_id_and_the_label(capsys) -> None:
    calls = []

    def discover(_runner, **arguments):
        calls.append(arguments)
        return _cpu()

    code = verb.run_site_id_command(
        admin_cli.parser().parse_args(["site-id", "--cpu-cluster-arn", CPU_EKS_ARN]),
        runner=object(),  # type: ignore[arg-type]
        discover=discover,
    )

    assert code == 0
    assert calls == [
        {
            "cluster_arn": CPU_EKS_ARN,
            "role": "cpu",
            "context": "gpu-fault-admin-site-id",
        }
    ], "the id comes from the same CPU discovery deploy runs"
    captured = capsys.readouterr()
    document = json.loads(captured.out)
    expected = site_identifier(_cpu(), [])
    assert document["site_id"] == expected, (
        "the id is the deterministic site identifier"
    )
    assert document["ownership_label"] == {"key": OWNERSHIP_LABEL, "value": expected}
    assert document["node_selectors"]["eks-managed"] == f"{OWNERSHIP_LABEL}={expected}"
    assert document["node_selectors"]["hyperpod"].startswith(
        f"{OWNERSHIP_LABEL}={expected},sagemaker.amazonaws.com/cluster-name="
    ), "the HyperPod selector adds the cluster-name cross-check"
    assert "update-nodegroup-config" in document["apply"]["eks_managed_node_group"]
    assert "update-cluster" in document["apply"]["hyperpod_instance_group"]
    assert "KubernetesConfig" in document["apply"]["hyperpod_instance_group"]
    assert "spec.template.metadata.labels" in document["apply"]["karpenter_nodepool"]
    assert document["cpu_cluster"]["hyperpod_arn"] == CPU_HYPERPOD_ARN
    assert "recorded_site" not in document, (
        "nothing is recorded before the first deploy"
    )
    assert f"{OWNERSHIP_LABEL}={expected}" in captured.err


def test_site_id_matches_the_recorded_site_name_when_given_a_state_dir(
    tmp_path: Path, capsys
) -> None:
    path = site_file(tmp_path)
    expected = site_identifier(_cpu(), [])
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["metadata"]["name"] = expected
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    code = verb.run_site_id_command(
        admin_cli.parser().parse_args(
            ["site-id", "--cpu-cluster-arn", CPU_EKS_ARN, "--state-dir", str(tmp_path)]
        ),
        runner=object(),  # type: ignore[arg-type]
        discover=lambda _runner, **_arguments: _cpu(),
    )

    assert code == 0
    recorded = json.loads(capsys.readouterr().out)["recorded_site"]
    assert recorded == {
        "state_dir": str(tmp_path),
        "site_name": expected,
        "matches": True,
    }


def test_site_id_fails_closed_when_the_state_dir_records_another_site(
    tmp_path: Path, capsys
) -> None:
    site_file(tmp_path)  # metadata.name is "test-site"

    code = verb.run_site_id_command(
        admin_cli.parser().parse_args(
            ["site-id", "--cpu-cluster-arn", CPU_EKS_ARN, "--state-dir", str(tmp_path)]
        ),
        runner=object(),  # type: ignore[arg-type]
        discover=lambda _runner, **_arguments: _cpu(),
    )

    assert code == 1, "labelling node groups with another site's id must be refused"
    captured = capsys.readouterr()
    assert json.loads(captured.out)["recorded_site"]["matches"] is False
    assert "records site test-site" in captured.err


def test_site_id_refuses_a_cpu_arn_without_a_hyperpod_control_plane() -> None:
    plain = ClusterIdentity(
        input_arn=CPU_EKS_ARN,
        role="cpu",
        region="us-west-2",
        account_id="123456789012",
        hyperpod_arn="",
        hyperpod_name="",
        eks_arn=CPU_EKS_ARN,
        eks_name="control-plane-eks",
        vpc_id="vpc-1",
        subnet_ids=(),
        node_recovery="",
        context="cpu",
        provider="eks-managed",
    )
    with pytest.raises(BootstrapError, match="must be a HyperPod EKS cluster"):
        verb.run_site_id_command(
            admin_cli.parser().parse_args(
                ["site-id", "--cpu-cluster-arn", CPU_EKS_ARN]
            ),
            runner=object(),  # type: ignore[arg-type]
            discover=lambda _runner, **_arguments: plain,
        )


def test_deploy_and_join_take_no_selector_flags_any_more() -> None:
    """The node scope is declared on the node groups; the retired flags are refused."""

    parser = admin_cli.parser()
    for command in (
        ["deploy", "--state-dir", "/tmp/s", "--gpu-nodegroup", "gpu-ng"],
        ["deploy", "--state-dir", "/tmp/s", "--node-selector", "a=b"],
        [
            "join-cluster",
            "--state-dir",
            "/tmp/s",
            "--gpu-cluster-arn",
            CPU_EKS_ARN,
            "--gpu-nodegroup",
            "gpu-ng",
        ],
        [
            "join-cluster",
            "--state-dir",
            "/tmp/s",
            "--gpu-cluster-arn",
            CPU_EKS_ARN,
            "--node-selector",
            "a=b",
        ],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(command)
    join = parser.parse_args(
        ["join-cluster", "--state-dir", "/tmp/s", "--gpu-cluster-arn", CPU_EKS_ARN]
    )
    assert not hasattr(join, "node_selector") and not hasattr(join, "gpu_nodegroup")
