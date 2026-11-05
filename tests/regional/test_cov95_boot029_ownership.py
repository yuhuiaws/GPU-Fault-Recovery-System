"""BOOT-029 ownership-label reads: the node-scope precondition and the re-apply journals.

The Region is ``tests/regional/test_boot029_ownership.World``: the CPU
HyperPod cluster, one GPU HyperPod cluster and one plain EKS cluster.
"""

from __future__ import annotations

import copy
import hashlib
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

import gpu_fault.admin.bootstrap_common as bootstrap_common
from gpu_fault.admin.bootstrap_common import BootstrapError
from scripts.e2e.regional import boot029_node_scope as node_scope
from scripts.e2e.regional import boot029_ownership as ownership
from tests.regional.test_boot029_ownership import (
    ACCOUNT,
    CPU_ARN,
    EKS_ARN,
    HP_ARN,
    HP_EKS_ARN,
    REGION,
    World,
)

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "e2e" / "regional"
EKS_TARGET = {
    "cluster_id": "gpu-eks",
    "provider": "eks-managed",
    "eks_cluster_arn": EKS_ARN,
    "hyperpod_cluster_name": "",
    "region": REGION,
}


def sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


class Anonymous(World):
    """The named HyperPod cluster answers without the identity fields ``drop``."""

    def __init__(self, cluster: str, *drop: str) -> None:
        super().__init__()
        self.cluster = cluster
        self.drop = drop

    def aws_json(self, region: str, *arguments: str, **options: Any) -> dict[str, Any]:
        value = super().aws_json(region, *arguments, **options)
        if arguments[1] == "describe-cluster" and value["ClusterName"] == self.cluster:
            value = copy.deepcopy(value)
            for name in self.drop:
                value.pop(name)
        return value


def write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def removal(*, phase: str = "COMPLETED", evidence: Any) -> dict[str, Any]:
    """A ``remove-cluster`` tombstone of the plain EKS cluster."""

    return {
        "phase": phase,
        "target": {"eks_cluster_arn": EKS_ARN, "hyperpod_cluster_name": ""},
        "evidence": {"DISCOVERED": {}, "OWNERSHIP_RELEASED": evidence},
    }


def released(*names: str) -> dict[str, Any]:
    return {
        "outcome": "released",
        "groups": [{"name": name, "outcome": "released"} for name in names],
    }


def run_cli(
    monkeypatch: pytest.MonkeyPatch, module: Any, world: World, *argv: str
) -> None:
    monkeypatch.setattr(module, "CommandRunner", lambda: world)
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    module.main()


# --- boot029_node_scope ------------------------------------------------------


def test_cpu_site_id_is_the_same_for_the_sagemaker_and_the_eks_arn() -> None:
    world = World()
    sagemaker_arn = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/cpu-hp"

    assert node_scope.cpu_site_id(sagemaker_arn, runner=world) == world.site_id()
    assert ("sagemaker", "describe-cluster", "--cluster-name", sagemaker_arn) in (
        world.calls
    )


@pytest.mark.parametrize(
    ("arn", "fragment"),
    [
        (
            f"arn:aws:ec2:{REGION}:{ACCOUNT}:instance/i-0123456789abcdef0",
            "a cluster is identified by its EKS or SageMaker ARN",
        ),
        (EKS_ARN, f"no HyperPod cluster orchestrates {EKS_ARN}"),
    ],
)
def test_cpu_site_id_refuses_an_arn_that_names_no_hyperpod_control_plane(
    arn: str, fragment: str
) -> None:
    with pytest.raises(BootstrapError, match=fragment):
        node_scope.cpu_site_id(arn, runner=World())


def test_cpu_site_id_refuses_a_hyperpod_without_its_identity() -> None:
    with pytest.raises(BootstrapError, match="exposes no HyperPod identity"):
        node_scope.cpu_site_id(CPU_ARN, runner=Anonymous("cpu-hp", "ClusterArn"))


def test_precondition_report_names_the_owned_groups_and_the_refusals() -> None:
    world = World()
    site_id = world.site_id()
    world.label(site_id)

    record, lines = node_scope.precondition_report(
        CPU_ARN, [EKS_ARN, HP_EKS_ARN], runner=world
    )

    eks, hyperpod = record["clusters"]
    assert eks == {
        "arn_sha256": sha256(EKS_ARN),
        "provider": "eks-managed",
        "status": "labelled",
        "owned_groups_sha256": [sha256("gpu-ng")],
        "node_groups_seen": 2,
    }
    assert hyperpod == {
        "arn_sha256": sha256(HP_EKS_ARN),
        "provider": "hyperpod",
        "status": "refused",
    }
    assert record["site_id_sha256"] == sha256(site_id)
    assert lines[0].startswith(f"node-scope precondition: site id {site_id}"), lines
    assert "owns gpu-ng; node groups seen: cpu-ng" in lines[1]
    assert "HyperPod cluster gpuhp01 has no instance group labelled" in lines[2]
    serialized = json.dumps(record)
    assert "gpu-eks" not in serialized, serialized
    assert site_id not in serialized, serialized


def test_node_scope_cli_passes_a_labelled_site_and_stops_a_refused_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World()
    world.label(world.site_id())

    run_cli(monkeypatch, node_scope, world, "--cpu-arn", CPU_ARN, "--gpu-arn", EKS_ARN)

    captured = capsys.readouterr()
    record = json.loads(captured.out)["node_scope_precondition"]
    assert [item["status"] for item in record["clusters"]] == ["labelled"]
    assert "owns gpu-ng" in captured.err

    with pytest.raises(SystemExit) as stop:
        run_cli(
            monkeypatch, node_scope, World(), "--cpu-arn", CPU_ARN, "--gpu-arn", EKS_ARN
        )
    assert stop.value.code == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["node_scope_precondition"]["clusters"][0][
        "status"
    ] == ("refused")
    assert "has no managed node group labelled" in captured.err


def test_node_scope_cli_reports_a_cpu_cluster_it_cannot_read(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stop:
        run_cli(
            monkeypatch, node_scope, World(), "--cpu-arn", EKS_ARN, "--gpu-arn", EKS_ARN
        )

    assert stop.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        f"node-scope precondition: no HyperPod cluster orchestrates {EKS_ARN}\n"
    )


def test_node_scope_runs_as_a_program(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World()
    world.label(world.site_id())
    monkeypatch.setattr(bootstrap_common, "CommandRunner", lambda: world)
    monkeypatch.setattr(
        sys, "argv", ["boot029_node_scope", "--cpu-arn", CPU_ARN, "--gpu-arn", EKS_ARN]
    )

    runpy.run_path(str(SCRIPTS / "boot029_node_scope.py"), run_name="__main__")

    record = json.loads(capsys.readouterr().out)["node_scope_precondition"]
    assert record["clusters"][0]["status"] == "labelled"


# --- boot029_ownership: cluster targets ---------------------------------------


def test_cluster_target_refuses_a_hyperpod_that_vanished_after_the_provider_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World()
    monkeypatch.setattr(ownership, "find_hyperpod_for_eks", lambda runner, **_: None)

    with pytest.raises(
        BootstrapError, match=f"no HyperPod cluster orchestrates {HP_EKS_ARN}"
    ):
        ownership.cluster_target(HP_EKS_ARN, runner=world)


def test_cluster_target_refuses_a_hyperpod_without_name_or_orchestrator() -> None:
    with pytest.raises(BootstrapError, match=f"behind {HP_ARN} exposes no identity"):
        ownership.cluster_target(HP_ARN, runner=Anonymous("gpuhp01", "Orchestrator"))


# --- boot029_ownership: release journals --------------------------------------


def test_journals_are_read_before_stage_three_and_five_only(tmp_path: Path) -> None:
    with pytest.raises(BootstrapError, match="before stage 3 and stage 5 only"):
        ownership.released_groups_from_journals(tmp_path, stage=4, target=EKS_TARGET)


def test_stage_three_skips_unreadable_and_unfinished_tombstones(tmp_path: Path) -> None:
    (tmp_path / "remove-cluster" / "a-broken").mkdir(parents=True)
    (tmp_path / "remove-cluster" / "a-broken" / "state.json").write_text(
        "{not json", encoding="utf-8"
    )
    write_json(
        tmp_path / "remove-cluster" / "b-running" / "state.json",
        removal(phase="RUNNING", evidence=released("cpu-ng")),
    )
    write_json(tmp_path / "remove-cluster" / "c-list" / "state.json", ["COMPLETED"])
    write_json(
        tmp_path / "remove-cluster" / "gpu-eks" / "state.json",
        removal(evidence=released("gpu-ng")),
    )

    assert ownership.released_groups_from_journals(
        tmp_path, stage=3, target=EKS_TARGET
    ) == ("remove-cluster/gpu-eks/state.json", ["gpu-ng"])


def test_stage_three_release_evidence_that_is_not_a_record_names_no_group(
    tmp_path: Path,
) -> None:
    write_json(
        tmp_path / "remove-cluster" / "gpu-eks" / "state.json",
        removal(evidence="released"),
    )

    assert ownership.released_groups_from_journals(
        tmp_path, stage=3, target=EKS_TARGET
    ) == ("remove-cluster/gpu-eks/state.json", [])


def test_stage_five_needs_the_uninstall_release_evidence(tmp_path: Path) -> None:
    write_json(tmp_path / "uninstall" / "state.json", {"phase": "COMPLETED"})

    with pytest.raises(BootstrapError, match="has no ownership_release evidence"):
        ownership.released_groups_from_journals(tmp_path, stage=5, target=EKS_TARGET)


def test_stage_five_finds_the_cluster_among_the_uninstall_entries(
    tmp_path: Path,
) -> None:
    other = {"eks_cluster_arn": HP_EKS_ARN, "hyperpod_cluster_name": "gpuhp01"}
    ours = {
        "eks_cluster_arn": EKS_ARN,
        "hyperpod_cluster_name": "",
        **released("gpu-ng"),
    }
    journal = tmp_path / "uninstall" / "state.json"
    write_json(
        journal,
        {
            "phase": "COMPLETED",
            "ownership_release": {"clusters": ["junk", other, ours]},
        },
    )

    assert ownership.released_groups_from_journals(
        tmp_path, stage=5, target=EKS_TARGET
    ) == ("uninstall/state.json", ["gpu-ng"])

    write_json(
        journal,
        {"phase": "COMPLETED", "ownership_release": {"clusters": ["junk", other]}},
    )
    with pytest.raises(BootstrapError, match="records a release for gpu-eks"):
        ownership.released_groups_from_journals(tmp_path, stage=5, target=EKS_TARGET)


def test_reapply_refuses_a_released_group_the_cluster_no_longer_has(
    tmp_path: Path,
) -> None:
    world = World()
    write_json(
        tmp_path / "remove-cluster" / "gpu-eks" / "state.json",
        removal(evidence=released("gpu-ng", "retired-ng")),
    )

    with pytest.raises(
        BootstrapError,
        match="no longer has managed node group\\(s\\) retired-ng that "
        "remove-cluster/gpu-eks/state.json recorded as released",
    ):
        ownership.reapply_report(
            CPU_ARN, [EKS_ARN], state=tmp_path, stage=3, runner=world
        )

    assert not [call for call in world.calls if call[1].startswith("update")], (
        "nothing is labelled while a recorded group is missing"
    )


# --- boot029_ownership: CLI ----------------------------------------------------


def test_reapply_cli_needs_the_state_directory_and_the_stage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stop:
        run_cli(
            monkeypatch,
            ownership,
            World(),
            "reapply",
            "--cpu-arn",
            CPU_ARN,
            "--gpu-arn",
            EKS_ARN,
            "--stage",
            "3",
        )

    assert stop.value.code == 2
    assert "reapply requires --state and --stage" in capsys.readouterr().err


def test_reapply_cli_prints_the_record_for_already_labelled_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World()
    world.label(world.site_id())
    write_json(
        tmp_path / "remove-cluster" / "gpu-eks" / "state.json",
        removal(evidence=released("gpu-ng")),
    )

    run_cli(
        monkeypatch,
        ownership,
        world,
        "reapply",
        "--cpu-arn",
        CPU_ARN,
        "--gpu-arn",
        EKS_ARN,
        "--state",
        str(tmp_path),
        "--stage",
        "3",
    )

    captured = capsys.readouterr()
    [cluster] = json.loads(captured.out)["ownership_label_reapplied"]["clusters"]
    assert (cluster["already_labelled"], cluster["applied"]) == (1, 0)
    assert cluster["groups_sha256"] == [sha256("gpu-ng")]
    assert "applied to none, converged after 0.0s" in captured.err
    assert world.sleeps == [], "nothing to wait for"


def test_reapply_cli_reports_missing_release_evidence_and_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stop:
        run_cli(
            monkeypatch,
            ownership,
            World(),
            "reapply",
            "--cpu-arn",
            CPU_ARN,
            "--gpu-arn",
            EKS_ARN,
            "--state",
            str(tmp_path),
            "--stage",
            "5",
        )

    assert stop.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("ownership reapply: no COMPLETED journal"), (
        captured.err
    )


def test_ownership_runs_as_a_program_and_passes_a_released_cluster(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World()
    monkeypatch.setattr(bootstrap_common, "CommandRunner", lambda: world)
    monkeypatch.setattr(
        sys,
        "argv",
        ["boot029_ownership", "released", "--cpu-arn", CPU_ARN, "--gpu-arn", EKS_ARN],
    )

    runpy.run_path(str(SCRIPTS / "boot029_ownership.py"), run_name="__main__")

    captured = capsys.readouterr()
    [cluster] = json.loads(captured.out)["ownership_label_released"]["clusters"]
    assert cluster["status"] == "released"
    assert "carries the label on no node group" in captured.err
