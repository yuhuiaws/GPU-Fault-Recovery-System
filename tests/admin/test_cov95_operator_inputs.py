from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest
from kubernetes.client import V1Node, V1ObjectMeta

from gpu_fault.admin import collector_outbox as outbox
from gpu_fault.admin import incident_close, warm_spare
from gpu_fault.admin import workflow_reconcile as reconcile
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from tests.admin.test_admin_site import site_file
from tests.admin.test_admin_warm_spare import FakeCoreApi
from tests.admin.test_cov95_workflow_reconcile import ControlPlane


@pytest.fixture
def site(tmp_path):
    return load_site(site_file(tmp_path))


@pytest.mark.parametrize(
    "change", [{"cluster_id": ""}, {"node_id": ""}, {"wait_seconds": 0}]
)
def test_outbox_invalid_target_or_wait_never_submits(site, change):
    arguments = {
        "site": site,
        "cluster_id": "gpu-a",
        "node_id": "node-a",
        "collector": "kernel",
        "action": "stats",
        "reference": "CHG-EXAMPLE",
        **change,
    }
    with pytest.raises(
        BootstrapError, match="requires --cluster-id|requires --node|wait-seconds"
    ):
        outbox.CollectorOutboxRequest(**arguments)


@pytest.mark.parametrize("late", [False, True])
def test_outbox_read_timeout_cannot_become_terminal_success(site, monkeypatch, late):
    clock = SimpleNamespace(value=0.0)
    calls = []

    def timeout(_site, payload, **_options):
        calls.append(payload)
        clock.value = 5 if late else 1
        raise subprocess.TimeoutExpired("example-status-read", 1)

    monkeypatch.setattr(outbox, "run_control_plane_script", timeout)
    if late:
        assert (
            outbox.wait_for_terminal(
                site, "workflow-example", wait_seconds=5, clock=lambda: clock.value
            )
            == {}
        )
    else:
        with pytest.raises(subprocess.TimeoutExpired):
            outbox.wait_for_terminal(
                site, "workflow-example", wait_seconds=5, clock=lambda: clock.value
            )
    assert calls == [{"mode": "status", "workflow_request_id": "workflow-example"}]


def test_outbox_wait_rejects_invalid_budget_without_a_read(site, monkeypatch):
    calls = []
    monkeypatch.setattr(
        outbox,
        "run_control_plane_script",
        lambda *_args, **_options: calls.append("read"),
    )
    with pytest.raises(BootstrapError, match="wait-seconds"):
        outbox.wait_for_terminal(site, "workflow-example", wait_seconds=0)
    assert calls == []


def test_outbox_public_submission_requires_workflow_identity(site, monkeypatch):
    calls = []
    monkeypatch.setattr(outbox, "resolve_operator_identity", lambda: "example-operator")

    def missing(_site, payload, **_options):
        calls.append(payload)
        return {"workflow": {}}

    monkeypatch.setattr(outbox, "run_control_plane_script", missing)
    request = outbox.CollectorOutboxRequest(
        site, "gpu-a", "node-a", "kernel", "stats", "CHG-EXAMPLE"
    )
    with pytest.raises(BootstrapError, match="returned no workflow"):
        outbox.run_collector_outbox(request)
    assert len(calls) == 1
    assert not (site.source.parent / "collector-outbox").exists(), (
        "submission without workflow identity created success evidence"
    )


def raw_node(name="node-a"):
    return {
        "metadata": {
            "name": name,
            "uid": "uid-" + name,
            "resourceVersion": "1",
            "labels": {
                warm_spare.HYPERPOD_HEALTH_LABEL: "Schedulable",
                warm_spare.INSTANCE_GROUP_LABEL: "example-group",
                "node.kubernetes.io/instance-type": "example-gpu",
            },
            "annotations": {},
        },
        "spec": {"unschedulable": False},
        "status": {
            "conditions": [{"type": "Ready", "status": "True"}],
            "allocatable": {warm_spare.GPU_RESOURCE: "8"},
        },
    }


def test_warm_spare_serializes_client_models_without_a_cluster_transport():
    assert (
        warm_spare.plain(V1Node(metadata=V1ObjectMeta(name="example")))["metadata"][
            "name"
        ]
        == "example"
    )
    assert warm_spare.plain("not-a-node") == {}


@pytest.mark.parametrize("value", [None, "not-an-integer", {}])
def test_warm_spare_invalid_gpu_quantities_never_become_capacity(value):
    node = raw_node()
    node["status"]["allocatable"][warm_spare.GPU_RESOURCE] = value
    assert warm_spare.node_snapshot(node)["gpu_allocatable"] == 0
    assert (
        warm_spare.pod_gpu_count(
            {
                "spec": {
                    "containers": [
                        {"resources": {"limits": {warm_spare.GPU_RESOURCE: value}}}
                    ]
                }
            }
        )
        == 0
    )


@pytest.mark.parametrize("content", ["invalid", "[]"])
def test_warm_spare_record_must_be_a_valid_object(tmp_path, content):
    path = tmp_path / "record.json"
    path.write_text(content)
    with pytest.raises(
        warm_spare.WarmSpareError, match="not valid JSON|not a JSON object"
    ):
        warm_spare.read_record(path)


def test_warm_spare_kubectl_malformed_json_is_not_an_empty_node():
    calls = []

    def response(arguments, **options):
        calls.append((arguments, options))
        return subprocess.CompletedProcess(arguments, 0, "invalid", "")

    api = warm_spare.KubectlNodeApi(["kubectl", "--context", "example"], run=response)
    with pytest.raises(warm_spare.WarmSpareError, match="invalid JSON"):
        api.read_node("node-a")
    assert len(calls) == 1
    assert calls[0][1]["timeout_seconds"] == 120


def test_warm_spare_survey_compares_fault_node_topology_without_mutation():
    api = FakeCoreApi(raw_node(), raw_node("fault-node"))
    report = warm_spare.survey(
        api,
        node="node-a",
        fault_node="fault-node",
        cluster_id="gpu-a",
        agent_lookup=lambda *_args: warm_spare.AgentState("ACTIVE"),
    )
    assert report["fault_node"]["name"] == "fault-node"
    assert report["fault_topology"] == report["topology"]
    assert api.patches == []


@pytest.mark.parametrize("field", ["name", "uid", "resource_version"])
def test_warm_spare_declaration_requires_complete_node_identity_before_patch(
    tmp_path, field
):
    api = FakeCoreApi(raw_node())
    observed = warm_spare.node_snapshot(raw_node())
    observed[field] = None
    path = tmp_path / "record.json"
    with pytest.raises(warm_spare.WarmSpareError, match="identity or resource version"):
        warm_spare.declare(
            api,
            node="node-a",
            record=path,
            reference="CHG-EXAMPLE",
            actor="example",
            survey={"cluster_id": "gpu-a", "node": observed},
        )
    assert api.patches == []
    assert path.is_file(), "failed declaration lost its pre-mutation audit"


def test_warm_spare_declaration_refuses_replaced_node_after_patch(
    tmp_path, monkeypatch
):
    api = FakeCoreApi(raw_node())
    patch = api.patch_node

    def recreated(name, body):
        result = patch(name, body)
        api.nodes[name]["metadata"]["uid"] = "recreated-node"
        return result

    monkeypatch.setattr(api, "patch_node", recreated)
    path = tmp_path / "record.json"
    with pytest.raises(warm_spare.WarmSpareError, match="identity changed"):
        warm_spare.declare(
            api,
            node="node-a",
            record=path,
            reference="CHG-EXAMPLE",
            actor="example",
            survey={
                "cluster_id": "gpu-a",
                "node": warm_spare.node_snapshot(raw_node()),
            },
        )
    assert len(api.patches) == 1
    assert "declared_state" not in json.loads(path.read_text())


def test_released_warm_spare_record_cannot_be_released_twice(tmp_path):
    path = tmp_path / "record.json"
    path.write_text('{"released_at":"example-completed"}')
    api = FakeCoreApi()
    with pytest.raises(warm_spare.WarmSpareError, match="already released"):
        warm_spare.release(
            api,
            node="node-a",
            record=path,
            reference="CHG-EXAMPLE",
            actor="example",
            survey={},
        )
    assert api.patches == []


@pytest.mark.parametrize(
    "mode,reference", [("unknown", "CHG-EXAMPLE"), ("declare", "bad\nreference")]
)
def test_warm_spare_public_action_rejects_unknown_mode_and_invalid_reference(
    tmp_path, mode, reference
):
    api = FakeCoreApi(raw_node())
    request = warm_spare.WarmSpareRequest(
        tmp_path,
        "node-a",
        mode=mode,
        reference=reference,
        confirmation=warm_spare.DECLARE_CONFIRMATION,
    )
    with pytest.raises(
        warm_spare.WarmSpareError, match="unknown warm-spare mode|reference is invalid"
    ):
        warm_spare.perform_warm_spare(
            api,
            request,
            cluster_id="gpu-a",
            record=tmp_path / "record.json",
            agent_lookup=lambda *_args: warm_spare.AgentState("ACTIVE"),
            actor="example",
        )
    assert api.patches == []


@pytest.mark.parametrize(
    "record",
    [
        {"incident_id": "example", "node_ids": ["node-a"]},
        {"incident_id": "example", "cluster_id": "gpu-a", "node_ids": []},
    ],
)
def test_incident_close_missing_binding_is_a_refusal_without_node_reads(site, record):
    evidence, refused = incident_close.gather_isolation_evidence(site, [record])
    assert evidence == {}
    assert "names no cluster or no nodes" in refused["example"]


@pytest.mark.parametrize(
    "options",
    [
        {"selector": {"mode": "escalated", "max_items": 0}},
        {"reference": "bad\nreference"},
    ],
)
def test_incident_close_rejects_invalid_selection_or_approval_before_transport(
    site, tmp_path, options
):
    arguments = {
        "incident_ids": ["example"],
        "reason": "example",
        "reference": "CHG-EXAMPLE",
        "dry_run": False,
        "actor": "example",
    }
    arguments.update(options)
    if "selector" in options:
        arguments["incident_ids"] = []
    with pytest.raises(BootstrapError, match="max-items|reference is invalid"):
        incident_close.run_incident_close(site, tmp_path, **arguments)


@pytest.mark.parametrize(
    "selection,output", [(None, {}), ({"mode": "escalated"}, {"results": []})]
)
def test_incident_close_requires_results_and_discovery_identity(
    site, tmp_path, monkeypatch, selection, output
):
    transport = ControlPlane()
    transport.output = json.dumps(output)
    monkeypatch.setattr(reconcile, "run_command", transport)
    with pytest.raises(
        BootstrapError, match="returned no results|returned no incident ids"
    ):
        incident_close.run_incident_close(
            site,
            tmp_path,
            incident_ids=["example"] if selection is None else [],
            selector=selection,
            reason="example",
            reference=None,
            dry_run=True,
            actor="example",
        )
    assert len(transport.payloads) == 1
