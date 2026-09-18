from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_fault.admin import bootstrap
from gpu_fault.admin import cluster_join as join
from gpu_fault.admin import resource_registry as registry
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from tests.admin._cov95_join_support import JoinScenario, target


class Discovery:
    def __init__(self):
        member = target()
        self.members = [
            {
                "ClusterArn": member.hyperpod_arn,
                "ClusterName": member.hyperpod_name,
                "NodeRecovery": "None",
                "Orchestrator": {"Eks": {"ClusterArn": member.eks_arn}},
            }
        ]
        self.status = "ACTIVE"
        self.calls = []

    def aws_json(self, _region, service, operation, *arguments, **_options):
        self.calls.append((service, operation))
        if (service, operation) == ("sagemaker", "list-clusters"):
            return {
                "ClusterSummaries": [
                    {"ClusterName": item["ClusterName"]} for item in self.members
                ]
            }
        if (service, operation) == ("sagemaker", "describe-cluster"):
            selected = arguments[arguments.index("--cluster-name") + 1]
            return next(
                item
                for item in self.members
                if selected in {item["ClusterName"], item["ClusterArn"]}
            )
        assert (service, operation) == ("eks", "describe-cluster"), (
            "unexpected discovery transport"
        )
        return {
            "cluster": {
                "status": self.status,
                "resourcesVpcConfig": {"vpcId": "vpc-example", "subnetIds": []},
            }
        }


@pytest.mark.parametrize(
    "failure",
    ["service", "missing", "ambiguous", "orchestrator", "inactive", "recovery"],
)
def test_public_discovery_requires_unique_eks_backed_recovery_disabled_target(failure):
    runner = Discovery()
    arn = target().hyperpod_arn
    if failure == "service":
        arn = "arn:aws:sns:us-east-1:123456789012:example"
    elif failure == "missing":
        runner.members = []
        arn = target().eks_arn
    elif failure == "ambiguous":
        runner.members.append({**runner.members[0], "ClusterName": "other-hyperpod"})
        arn = target().eks_arn
    elif failure == "orchestrator":
        runner.members[0]["Orchestrator"] = {}
    elif failure == "inactive":
        runner.status = "CREATING"
    else:
        runner.members[0]["NodeRecovery"] = "Automatic"
    with pytest.raises(
        BootstrapError, match="service|exactly one|orchestrator|not ACTIVE|NodeRecovery"
    ):
        bootstrap.discover_cluster(
            runner, cluster_arn=arn, role="gpu", context="example"
        )
    assert all(
        operation.startswith(("describe", "list"))
        for _service, operation in runner.calls
    ), "invalid discovery performed a mutation"


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    return JoinScenario(tmp_path, monkeypatch)


def test_public_commit_requires_verification_before_any_membership_write(scenario):
    scenario.failure = "join-cluster:hp-gpu-b"
    with pytest.raises(BootstrapError):
        scenario.join()
    state = scenario.state()
    proof = state["evidence"]
    execution = join.JoinExecution(
        target=target(),
        cluster_id="hp-gpu-b",
        discovery=proof["DISCOVERED"],
        local=proof["LOCAL_INPUTS_READY"],
        prerequisites=proof["PREREQUISITES_READY"],
        candidate=load_site(Path(proof["CANDIDATE_READY"]["site_file"])),
    )
    before = list(scenario.events)
    with pytest.raises(BootstrapError, match="no verification evidence"):
        join.commit_membership(
            scenario.request(),
            execution=execution,
            state_dir=scenario.state_path().parent,
            state_path=scenario.state_path(),
            state=state,
        )
    assert scenario.events == before


def test_final_join_requires_every_registered_resource_and_resumes_without_reactivation(
    scenario, monkeypatch
):
    def missing_role(arguments, **options):
        result = scenario.registry(arguments, **options)
        if (
            arguments[-1] == registry.FETCH_SCRIPT
            and scenario.cluster_states.get("hp-gpu-b") == "ACTIVE"
        ):
            records = [
                item
                for item in json.loads(result.stdout)
                if item["resource_key"] != "aws/iam/executor/hp-gpu-b/role"
            ]
            return scenario.result(arguments, json.dumps(records))
        return result

    monkeypatch.setattr(registry, "run_command", missing_role)
    with pytest.raises(BootstrapError, match="resources are absent"):
        scenario.join()
    assert scenario.state()["phase"] == "FAILED_AFTER_ACTIVATION"
    assert "FINAL_VERIFIED" not in scenario.state()["completed_steps"]
    monkeypatch.setattr(registry, "run_command", scenario.registry)
    assert scenario.join()["phase"] == "COMPLETED"
    assert scenario.driver_calls.count(("activate-cluster", "hp-gpu-b")) == 1
