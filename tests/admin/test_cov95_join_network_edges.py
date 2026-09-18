from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin import cluster_join as join
from gpu_fault.admin.bootstrap_common import BootstrapError
from tests.admin._cov95_join_support import JoinScenario, target


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    return JoinScenario(tmp_path, monkeypatch)


def test_join_cluster_id_collision_is_refused_before_preparation(scenario):
    with pytest.raises(BootstrapError, match="cluster_id already exists"):
        join.join_cluster(scenario.request(cluster_id="gpu-a"), runner=scenario)
    assert scenario.commands == []
    assert scenario.driver_calls == []


@pytest.mark.parametrize(
    "document",
    [
        [],
        {},
        {"kind": "Namespace", "metadata": {"name": "gpu-fault-system", "uid": ""}},
    ],
)
def test_join_namespace_probe_requires_kind_name_and_uid(scenario, document):
    scenario.namespace_read_override = json.dumps(document)
    with pytest.raises(join.JoinTargetIdentityError, match="namespace identity"):
        join.probe_join_namespace(
            scenario,
            site=scenario.site,
            kubeconfig=scenario.gpu_kubeconfig,
            context=target().context,
        )
    assert scenario.driver_calls == []


@pytest.mark.parametrize("retained", [False, True])
def test_join_uses_only_an_existing_cluster_ca_or_retained_bootstrap_ca(
    scenario, retained
):
    current_ca = Path(scenario.site.release_config["clusters"][0]["ca_file"])
    contents = current_ca.read_bytes()
    current_ca.unlink()
    retained_ca = scenario.directory / "retained-ca.crt"
    if retained:
        retained_ca.write_bytes(contents)
        (scenario.directory / "bootstrap-state.json").write_text(
            json.dumps(
                {
                    "site_id": "test-site",
                    "resources": {"pki": {"ca_file": str(retained_ca)}},
                }
            )
        )
        scenario.failure = "preflight"
    with pytest.raises(
        BootstrapError, match="cannot locate the site private CA|modeled join boundary"
    ):
        scenario.join()
    if retained:
        assert scenario.state()["evidence"]["LOCAL_INPUTS_READY"]["ca_file"] == str(
            retained_ca
        )
        assert scenario.driver_calls == [("preflight", None)]
    else:
        assert "LOCAL_INPUTS_READY" not in scenario.state()["completed_steps"]
        assert scenario.driver_calls == []


def test_join_existing_ingress_does_not_claim_a_new_rule(scenario):
    scenario.ingress.add("192.0.2.20")
    assert scenario.join()["phase"] == "COMPLETED"
    network = scenario.state()["evidence"]["PREREQUISITES_READY"]["network"]
    assert network["created_ingress_eips"] == []
    assert network["complete"] is True


def test_join_existing_vpc_association_does_not_repeat_association(scenario):
    scenario.vpcs.add(target().vpc_id)
    assert scenario.join()["phase"] == "COMPLETED"
    assert "associate" not in scenario.events
    assert (
        scenario.state()["evidence"]["PREREQUISITES_READY"]["network"][
            "association_created"
        ]
        is False
    )


def test_unproved_ingress_failure_requires_compensation_before_retry(
    scenario, monkeypatch
):
    calls = []
    command = scenario.command

    def denied(arguments, **options):
        if arguments[0] == "aws":
            calls.append(arguments)
            return scenario.result(arguments, code=1, error="example AccessDenied")
        return command(arguments, **options)

    monkeypatch.setattr(join, "run_command", denied)
    with pytest.raises(BootstrapError, match="cannot authorize NLB ingress"):
        scenario.join()
    network = scenario.state()["evidence"]["PREREQUISITES_READY"]["network"]
    assert network["pending_mutation"] == "ingress:192.0.2.20"
    assert network["complete"] is False
    monkeypatch.setattr(join, "run_command", command)
    with pytest.raises(BootstrapError, match="requires rollback first"):
        scenario.join()
    assert len(calls) == 1
    assert scenario.driver_calls == []


def test_join_requires_a_private_hosted_zone_before_candidate_rollout(scenario):
    document = yaml.safe_load(scenario.path.read_text())
    document["spec"].pop("dns")
    scenario.path.write_text(yaml.safe_dump(document))
    with pytest.raises(BootstrapError, match="requires the site private hosted zone"):
        scenario.join()
    assert scenario.driver_calls == []


@pytest.mark.parametrize("change", ["missing", "duplicate", "conflicting-context"])
def test_batch_candidate_drift_after_rollout_cannot_authorize_verification_or_activation(
    scenario, monkeypatch, change
):
    driver = scenario.driver
    changed = []

    def drift(arguments, **options):
        result = driver(arguments, **options)
        if arguments[1] == "join-cluster" and not changed:
            candidates = list(
                (scenario.directory / "join-cluster").glob(
                    "site.batch-candidate-*.yaml"
                )
            )
            assert len(candidates) == 1, "test did not find its owned batch candidate"
            path = candidates[0]
            document = yaml.safe_load(path.read_text())
            members = document["spec"]["clusters"]
            member = next(item for item in members if item["clusterId"] == "hp-gpu-b")
            if change == "missing":
                members.remove(member)
            elif change == "duplicate":
                members.append(copy.deepcopy(member))
            else:
                member["context"] = "gpu-a"
            path.write_text(yaml.safe_dump(document))
            changed.append(path)
        return result

    monkeypatch.setattr(join, "run_driver", drift)
    with pytest.raises(BootstrapError, match="no unique cluster|conflicting cluster"):
        scenario.batch(("b",))
    assert scenario.state()["phase"] == "FAILED"
    assert not any(
        mode == "activate-cluster" for mode, _cluster in scenario.driver_calls
    ), "candidate file drift authorized activation"
