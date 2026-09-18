from __future__ import annotations

import copy
import subprocess

import pytest

from tests.deploy._node_action_key_api import NAMESPACE, encoded, secret
from tests.deploy._node_key_custody_support import PROVISION, ProvisionFixture
from tests.regional._cov95_identity_support import offline_guard as offline_guard


@pytest.fixture
def fixture(tmp_path):
    return ProvisionFixture(tmp_path)


def synchronize(fixture, *, cpu=True, **kwargs):
    return PROVISION.provision(
        PROVISION.Scope(("kubectl", "--context", "gpu-context"), NAMESPACE),
        PROVISION.Scope(("kubectl", "--context", "cpu-context"), NAMESPACE)
        if cpu
        else None,
        cluster_id="cluster-a",
        hyperpod_cluster="hyperpod-a",
        master_file=fixture.master_file,
        runner=fixture.api.run,
        **kwargs,
    )


def test_uncertified_legacy_gpu_only_rotation_remains_explicitly_without_receipts(
    fixture,
):
    assert synchronize(fixture, cpu=False, rotate_node="node-a") == 2
    assert fixture.api.state["secrets"]["cpu"] is None
    assert not list(fixture.root.glob("state-*/*.chain.json")), (
        "legacy GPU-only rotation must not claim custody provenance"
    )


@pytest.mark.parametrize("failure", ["cluster", "inventory", "rotation-node", "master"])
def test_invalid_provisioning_inputs_refuse_before_any_write(fixture, failure):
    options = {}
    if failure == "inventory":
        options["expected_nodes"] = {"node-a": "foreign"}
    elif failure == "rotation-node":
        options["rotate_node"] = "missing"
    elif failure == "master":
        fixture.master_file.write_bytes(b"short")
    if failure == "cluster":
        with pytest.raises(PROVISION.ProvisionError, match="cluster identity"):
            PROVISION.provision(
                PROVISION.Scope(("kubectl",), NAMESPACE),
                None,
                cluster_id="",
                hyperpod_cluster="hyperpod-a",
                master_file=fixture.master_file,
                runner=fixture.api.run,
            )
    else:
        with pytest.raises(PROVISION.ProvisionError):
            synchronize(fixture, **options)
    assert fixture.api.state["writes"] == []


@pytest.mark.parametrize("failure", ["read", "metadata"])
def test_node_read_or_metadata_failure_is_not_an_empty_inventory(fixture, failure):
    if failure == "read":
        fixture.api.state["events"] = [{"on": "gpu:get-nodes", "returncode": 1}]
    else:
        fixture.api.state["nodes"]["items"][0]["metadata"] = None
    with pytest.raises(PROVISION.ProvisionError):
        synchronize(fixture)
    assert fixture.api.state["write_attempts"] == []


def test_transport_exception_is_bounded_and_sanitized(fixture):
    sentinel = "synthetic-sensitive-cli-output"

    def refused(*args, **kwargs):
        raise subprocess.SubprocessError(sentinel)

    client = PROVISION.SecretClient(PROVISION.Scope(("kubectl",), NAMESPACE), refused)
    with pytest.raises(PROVISION.ProvisionError) as caught:
        client.nodes("hyperpod-a")
    assert sentinel not in str(caught.value)


@pytest.mark.parametrize("ordinal", [3, 4])
def test_node_uid_drift_during_or_after_synchronization_stops_completion(
    fixture, ordinal
):
    changed = copy.deepcopy(fixture.api.state["nodes"])
    changed["items"][0]["metadata"]["uid"] = "recreated-node"
    fixture.api.state["events"] = [
        {"on": "gpu:get-nodes", "occurrence": ordinal, "nodes": changed}
    ]
    with pytest.raises(PROVISION.ProvisionError, match="identities changed"):
        synchronize(fixture)
    assert any(item["scope"] == "gpu" for item in fixture.api.state["writes"]), (
        "the Node UID drift scenario must occur after a GPU key write"
    )
    if ordinal == 3:
        assert not any(
            item["scope"] == "cpu" for item in fixture.api.state["writes"]
        ), "Node UID drift before synchronization must prevent CPU key writes"


def test_final_cpu_digest_drift_cannot_be_reported_converged(fixture):
    fixture.api.state["events"] = [
        {
            "on": "cpu:get-secret",
            "occurrence": 4,
            "merge_data": {"node-a": encoded("foreign")},
        }
    ]
    with pytest.raises(
        PROVISION.ProvisionError, match="CPU node keys did not converge"
    ):
        synchronize(fixture)


def test_a_concurrent_cpu_source_cannot_have_the_gpu_source_uid(fixture):
    fixture.api.state["events"] = [
        {"on": "cpu:get-secret", "occurrence": 2, "document": secret("gpu", {})}
    ]
    with pytest.raises(PROVISION.ProvisionError, match="targets must be distinct"):
        synchronize(fixture)
    assert not any(item["scope"] == "cpu" for item in fixture.api.state["writes"]), (
        "aliased CPU and GPU source identities must prevent CPU key writes"
    )


def test_cpu_drift_before_rotation_marker_removal_preserves_the_pending_marker(fixture):
    assert synchronize(fixture) == 2
    offset = fixture.api.state["counts"]["cpu:get-secret"]
    fixture.api.state["events"] = [
        {
            "on": "cpu:get-secret",
            "occurrence": offset + 4,
            "merge_data": {"node-a": encoded("foreign")},
        }
    ]
    with pytest.raises(PROVISION.ProvisionError, match="confirm the pending rotation"):
        synchronize(fixture, rotate_node="node-a")
    assert (
        PROVISION.ROTATION_ANNOTATION
        in fixture.api.state["secrets"]["gpu"]["metadata"]["annotations"]
    )


def test_repeated_gpu_cas_conflict_exhausts_the_fixed_attempt_budget(fixture):
    fixture.api.state["secrets"]["gpu"] = secret("gpu", {})
    fixture.api.state["events"] = [
        {
            "on": "gpu:replace",
            "occurrence": index,
            "returncode": 1,
            "merge_metadata": {"labels": {"external-owner": str(index)}},
        }
        for index in (1, 2, 3)
    ]
    with pytest.raises(PROVISION.ProvisionError, match="GPU.*changed repeatedly"):
        synchronize(fixture)
    assert len(fixture.api.state["write_attempts"]) == 3
    assert fixture.api.state["secrets"]["gpu"]["data"] == {}
    assert fixture.api.state["secrets"]["cpu"] is None
