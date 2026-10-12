"""The BOOT-029 driver proves the label gone after a de-registration and puts it
back, with the ``site-id`` commands, before the next re-join or re-deploy."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.cluster_scope import OWNERSHIP_LABEL
from scripts.e2e.regional import boot029_ownership as ownership
from scripts.e2e.regional.boot029_node_scope import cpu_site_id

REGION = "us-west-2"
ACCOUNT = "123456789012"
CPU_ARN = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/cpu"
HP_EKS_ARN = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/gpu-hp-orchestrator"
HP_ARN = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/gpuhp01"
EKS_ARN = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/gpu-eks"
LIFECYCLE = {"SourceS3Uri": "s3://lifecycle/", "OnCreate": "on_create.sh"}
ROLE = f"arn:aws:iam::{ACCOUNT}:role/hyperpod"


class World(CommandRunner):
    """A Region with the CPU HyperPod cluster, one GPU HyperPod cluster and one
    plain EKS cluster; the label state moves only through the update calls and
    converges after ``converge_after`` describes."""

    def __init__(self, *, converge_after: int = 2) -> None:
        super().__init__()
        self.calls: list[tuple[str, ...]] = []
        self.sleeps: list[float] = []
        self.converge_after = converge_after
        self.eks_labels: dict[str, dict[str, str]] = {
            "gpu-ng": {"node-type": "gpu"},
            "cpu-ng": {"node-type": "cpu"},
        }
        self.hp_labels: dict[str, str] = {"team": "research"}
        self.hp_desired: dict[str, str] | None = None
        self.pending: dict[str, tuple[int, dict[str, str]]] = {}
        self.hp_specs: list[list[dict[str, Any]]] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    def site_id(self) -> str:
        return cpu_site_id(CPU_ARN, runner=self)

    def label(self, site_id: str) -> None:
        self.eks_labels["gpu-ng"][OWNERSHIP_LABEL] = site_id
        self.hp_labels[OWNERSHIP_LABEL] = site_id

    def _tick(self, name: str) -> None:
        if name not in self.pending:
            return
        left, labels = self.pending[name]
        if left <= 1:
            del self.pending[name]
            if name in self.eks_labels:
                self.eks_labels[name] = dict(labels)
            else:
                self.hp_labels = dict(labels)
                self.hp_desired = None
        else:
            self.pending[name] = (left - 1, labels)

    def aws_json(
        self, region: str, *arguments: str, mutate: bool = False, **_options: Any
    ) -> dict[str, Any]:
        assert region == REGION
        self.calls.append(arguments)
        service, operation = arguments[0], arguments[1]
        if (service, operation) == ("sagemaker", "list-clusters"):
            return {
                "ClusterSummaries": [
                    {"ClusterName": "cpu-hp"},
                    {"ClusterName": "gpuhp01"},
                ]
            }
        if (service, operation) == ("sagemaker", "describe-cluster"):
            name = arguments[3].rsplit("/", 1)[-1]
            if name == "cpu-hp":
                return {
                    "ClusterName": "cpu-hp",
                    "ClusterArn": f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/cpuhp01",
                    "Orchestrator": {"Eks": {"ClusterArn": CPU_ARN}},
                    "InstanceGroups": [
                        {"InstanceGroupName": "cpu", "InstanceType": "ml.m5.2xlarge"}
                    ],
                }
            assert name == "gpuhp01", name
            self._tick("spot-p5en")
            return {
                "ClusterName": "gpuhp01",
                "ClusterArn": HP_ARN,
                "ClusterStatus": "InService",
                "Orchestrator": {"Eks": {"ClusterArn": HP_EKS_ARN}},
                "InstanceGroups": [
                    {
                        "InstanceGroupName": "spot-p5en",
                        "InstanceType": "ml.p5en.48xlarge",
                        "CurrentCount": 2,
                        "TargetCount": 2,
                        "Status": "InService",
                        "LifeCycleConfig": dict(LIFECYCLE),
                        "ExecutionRole": ROLE,
                        "KubernetesConfig": {
                            "CurrentLabels": dict(self.hp_labels),
                            "DesiredLabels": dict(
                                self.hp_desired
                                if self.hp_desired is not None
                                else self.hp_labels
                            ),
                        },
                    },
                    {
                        "InstanceGroupName": "controller",
                        "InstanceType": "ml.m5.2xlarge",
                    },
                ],
            }
        if (service, operation) == ("sagemaker", "update-cluster"):
            assert mutate, "a label write is a mutation"
            specs = json.loads(arguments[arguments.index("--instance-groups") + 1])
            self.hp_specs.append(specs)
            (spec,) = specs
            labels = dict(spec["KubernetesConfig"]["Labels"])
            self.hp_desired = labels
            self.pending["spot-p5en"] = (self.converge_after, labels)
            return {"ClusterArn": HP_ARN}
        if (service, operation) == ("eks", "list-nodegroups"):
            return {"nodegroups": sorted(self.eks_labels)}
        if (service, operation) == ("eks", "describe-nodegroup"):
            name = arguments[arguments.index("--nodegroup-name") + 1]
            self._tick(name)
            return {
                "nodegroup": {
                    "nodegroupName": name,
                    "status": "UPDATING" if name in self.pending else "ACTIVE",
                    "instanceTypes": [
                        "p5en.48xlarge" if name == "gpu-ng" else "m7i.xlarge"
                    ],
                    "labels": dict(self.eks_labels[name]),
                }
            }
        if (service, operation) == ("eks", "update-nodegroup-config"):
            assert mutate, "a label write is a mutation"
            name = arguments[arguments.index("--nodegroup-name") + 1]
            value = arguments[arguments.index("--labels") + 1]
            prefix = f"addOrUpdateLabels={{{OWNERSHIP_LABEL}="
            assert value.startswith(prefix) and value.endswith("}"), value
            labels = {**self.eks_labels[name], OWNERSHIP_LABEL: value[len(prefix) : -1]}
            self.pending[name] = (self.converge_after, labels)
            return {"update": {"id": "u-1", "status": "InProgress"}}
        raise AssertionError(f"unexpected aws call {arguments}")


def _removal_journal(
    state: Path,
    *,
    eks_arn: str,
    hyperpod_name: str,
    groups: list[tuple[str, str]],
    cluster_id: str,
) -> None:
    path = state / "remove-cluster" / cluster_id / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "phase": "COMPLETED",
                "attempt_id": "a" * 32,
                "target": {
                    "eks_cluster_arn": eks_arn,
                    "hyperpod_cluster_name": hyperpod_name,
                },
                "evidence": {
                    "DISCOVERED": {"provider_identity": {"eks_arn": eks_arn}},
                    "OWNERSHIP_RELEASED": {
                        "outcome": "released",
                        "groups": [
                            {"name": name, "outcome": outcome}
                            for name, outcome in groups
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def _uninstall_journal(state: Path, clusters: list[dict[str, Any]]) -> None:
    path = state / "uninstall" / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "phase": "COMPLETED",
                "ownership_release": {"outcome": "released", "clusters": clusters},
            }
        ),
        encoding="utf-8",
    )


def _sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def test_released_report_distinguishes_a_clean_cluster_from_a_stale_claim() -> None:
    world = World()
    site_id = world.site_id()

    record, lines, released = ownership.released_report(
        CPU_ARN, [HP_EKS_ARN, EKS_ARN], runner=world
    )

    assert released is True
    clusters = record["ownership_label_released"]["clusters"]
    assert [item["status"] for item in clusters] == ["released", "released"]
    assert [item["provider"] for item in clusters] == ["hyperpod", "eks-managed"]
    assert clusters[0]["arn_sha256"] == _sha256(HP_EKS_ARN)
    assert record["ownership_label_released"]["site_id_sha256"] == _sha256(site_id)
    serialized = json.dumps(record)
    for secret in (CPU_ARN, HP_EKS_ARN, EKS_ARN, site_id, "spot-p5en", "gpu-ng"):
        assert secret not in serialized, f"{secret} leaked into the record"
    assert any("carries the label on no node group" in line for line in lines), lines

    world.label(site_id)
    record, lines, released = ownership.released_report(
        CPU_ARN, [HP_EKS_ARN, EKS_ARN], runner=world
    )

    assert released is False, "a group still claiming the site is a stale claim"
    assert [
        item["status"] for item in record["ownership_label_released"]["clusters"]
    ] == (["claimed", "claimed"])
    assert any("spot-p5en" in line and "did not release" in line for line in lines), (
        lines
    )
    assert not any(call[1].startswith("update") for call in world.calls), (
        "the released check is read-only"
    )


def test_reapply_relabels_exactly_the_released_groups_with_the_site_id_commands(
    tmp_path: Path,
) -> None:
    world = World()
    site_id = world.site_id()
    _removal_journal(
        tmp_path,
        eks_arn=HP_EKS_ARN,
        hyperpod_name="gpuhp01",
        groups=[("spot-p5en", "released")],
        cluster_id="gpuhp01",
    )
    _removal_journal(
        tmp_path,
        eks_arn=EKS_ARN,
        hyperpod_name="",
        groups=[("gpu-ng", "released"), ("cpu-ng", "unlabelled")],
        cluster_id="gpu-eks",
    )

    record, lines = ownership.reapply_report(
        CPU_ARN,
        [HP_ARN, EKS_ARN],
        state=tmp_path,
        stage=3,
        runner=world,
        sleep=world.sleep,
    )

    updates = [call for call in world.calls if call[1].startswith("update")]
    assert updates[0][:3] == ("sagemaker", "update-cluster", "--cluster-name")
    assert updates[0][3] == "gpuhp01"
    (specs,) = world.hp_specs
    assert specs == [
        {
            "InstanceGroupName": "spot-p5en",
            "InstanceType": "ml.p5en.48xlarge",
            "InstanceCount": 2,
            "LifeCycleConfig": LIFECYCLE,
            "ExecutionRole": ROLE,
            "KubernetesConfig": {
                "Labels": {"team": "research", OWNERSHIP_LABEL: site_id}
            },
        }
    ], "the full current definition of the instance group, plus the label"
    assert updates[1] == (
        "eks",
        "update-nodegroup-config",
        "--cluster-name",
        "gpu-eks",
        "--nodegroup-name",
        "gpu-ng",
        "--labels",
        f"addOrUpdateLabels={{{OWNERSHIP_LABEL}={site_id}}}",
    ), "the exact command site-id prints; the CPU group is never touched"
    assert len(updates) == 2
    assert world.hp_labels[OWNERSHIP_LABEL] == site_id
    assert world.eks_labels["gpu-ng"][OWNERSHIP_LABEL] == site_id
    assert OWNERSHIP_LABEL not in world.eks_labels["cpu-ng"]
    assert world.sleeps, "the driver waits for the provider to apply the label"
    reapplied = record["ownership_label_reapplied"]
    assert reapplied["stage"] == 3
    assert [item["applied"] for item in reapplied["clusters"]] == [1, 1]
    assert reapplied["clusters"][0]["journal_path_sha256"] == _sha256(
        "remove-cluster/gpuhp01/state.json"
    )
    assert reapplied["clusters"][1]["groups_sha256"] == [_sha256("gpu-ng")]
    assert reapplied["clusters"][0]["arn_sha256"] == _sha256(HP_ARN)
    serialized = json.dumps(record)
    for secret in (HP_ARN, EKS_ARN, site_id, "spot-p5en", "gpu-ng", "gpuhp01"):
        assert secret not in serialized, f"{secret} leaked into the record"
    assert any("applied to spot-p5en" in line for line in lines), lines


def test_reapply_is_idempotent_and_reads_the_uninstall_journal_for_stage_five(
    tmp_path: Path,
) -> None:
    world = World()
    site_id = world.site_id()
    world.label(site_id)  # already labelled: nothing to write
    _uninstall_journal(
        tmp_path,
        [
            {
                "cluster_id": "gpu-eks",
                "eks_cluster_arn": EKS_ARN,
                "hyperpod_cluster_name": "",
                "outcome": "released",
                "groups": [{"name": "gpu-ng", "outcome": "released"}],
            }
        ],
    )

    record, _lines = ownership.reapply_report(
        CPU_ARN, [EKS_ARN], state=tmp_path, stage=5, runner=world, sleep=world.sleep
    )

    assert not any(call[1].startswith("update") for call in world.calls), (
        "a group already labelled needs no write"
    )
    (cluster,) = record["ownership_label_reapplied"]["clusters"]
    assert cluster["journal_path_sha256"] == _sha256("uninstall/state.json")
    assert (cluster["already_labelled"], cluster["applied"]) == (1, 0)
    assert world.sleeps == [], "nothing to wait for"


@pytest.mark.parametrize("stage", [3, 5])
def test_reapply_refuses_without_release_evidence(tmp_path: Path, stage: int) -> None:
    world = World()

    with pytest.raises(BootstrapError, match="records a release"):
        ownership.reapply_report(
            CPU_ARN, [EKS_ARN], state=tmp_path, stage=stage, runner=world
        )

    assert not any(call[1].startswith("update") for call in world.calls), (
        "the driver never guesses which groups to label"
    )


def test_reapply_refuses_a_removal_journal_that_never_ran_the_release_step(
    tmp_path: Path,
) -> None:
    world = World()
    path = tmp_path / "remove-cluster/gpu-eks/state.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "phase": "COMPLETED",
                "target": {"eks_cluster_arn": EKS_ARN, "hyperpod_cluster_name": ""},
                "evidence": {"DISCOVERED": {}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BootstrapError, match="did not run the release step"):
        ownership.reapply_report(
            CPU_ARN, [EKS_ARN], state=tmp_path, stage=3, runner=world
        )


def test_cli_exits_two_when_a_de_registered_cluster_still_claims_the_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    world = World()
    world.label(world.site_id())
    monkeypatch.setattr(ownership, "CommandRunner", lambda: world)
    monkeypatch.setattr(
        "sys.argv",
        ["boot029_ownership", "released", "--cpu-arn", CPU_ARN, "--gpu-arn", EKS_ARN],
    )

    with pytest.raises(SystemExit) as stop:
        ownership.main()

    assert stop.value.code == 2
    captured = capsys.readouterr()
    assert (
        json.loads(captured.out)["ownership_label_released"]["clusters"][0]["status"]
        == "claimed"
    )
    assert "did not release" in captured.err


def test_hyperpod_reapply_refuses_while_the_cluster_is_updating(tmp_path: Path) -> None:
    class Updating(World):
        def aws_json(
            self, region: str, *arguments: str, **options: Any
        ) -> dict[str, Any]:
            value = super().aws_json(region, *arguments, **options)
            if arguments[1] == "describe-cluster" and value["ClusterName"] == "gpuhp01":
                value = copy.deepcopy(value)
                value["ClusterStatus"] = "Updating"
            return value

    world = Updating()
    _removal_journal(
        tmp_path,
        eks_arn=HP_EKS_ARN,
        hyperpod_name="gpuhp01",
        groups=[("spot-p5en", "released")],
        cluster_id="gpuhp01",
    )

    with pytest.raises(BootstrapError, match="Updating"):
        ownership.reapply_report(
            CPU_ARN, [HP_EKS_ARN], state=tmp_path, stage=3, runner=world
        )

    assert world.hp_specs == [], "UpdateCluster is never sent mid-update"
