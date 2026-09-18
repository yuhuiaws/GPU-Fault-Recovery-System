from __future__ import annotations

import copy
import json
import subprocess

import pytest
import yaml

from gpu_fault.admin import workflow_reconcile as reconcile
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from tests.admin.test_admin_site import site_file


class ControlPlane:
    def __init__(self):
        self.calls = []
        self.payloads = []
        self.pod = "cpu-fixture"
        self.error = None
        self.output = None
        self.plan_reads = 0
        self.plan = {
            "plan_sha256": "a" * 64,
            "items": [
                {
                    "request_id": "workflow-a",
                    "cluster_id": "gpu-a",
                    "node_ids": ["node-a"],
                    "eligible": True,
                    "reasons": [],
                }
            ],
        }
        self.second_plan = None
        self.nodes = {
            "items": [
                {
                    "kind": "Node",
                    "metadata": {"name": "node-a", "uid": "uid-a", "annotations": {}},
                    "spec": {"unschedulable": False, "taints": []},
                }
            ]
        }

    def __call__(self, arguments, **options):
        self.calls.append((arguments, options))
        if "get" in arguments and "nodes" in arguments:
            stage = "nodes"
            output = json.dumps(self.nodes)
        elif "get" in arguments and "pod" in arguments:
            stage, output = "pod", self.pod
        else:
            stage = "exec"
            payload = json.loads(options["input_text"])
            self.payloads.append(payload)
            if self.output is not None:
                output = self.output
            elif payload["mode"] == "plan":
                self.plan_reads += 1
                value = (
                    self.second_plan
                    if self.plan_reads > 1 and self.second_plan is not None
                    else self.plan
                )
                output = json.dumps(value)
            else:
                output = json.dumps(
                    {
                        "applied_workflow_ids": payload["workflow_ids"],
                        "failed_workflow_ids": [],
                        "records_deleted": 0,
                    }
                )
        return subprocess.CompletedProcess(
            arguments,
            int(self.error == stage),
            output,
            "example read failure" if self.error == stage else "",
        )


@pytest.fixture
def context(tmp_path, monkeypatch):
    path = site_file(tmp_path)
    document = yaml.safe_load(path.read_text())
    gpu = tmp_path / "secure/gpu.kubeconfig"
    gpu.write_text("example-kubeconfig")
    gpu.chmod(0o600)
    document["spec"]["gpuKubeconfig"] = str(gpu)
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    site = load_site(path)
    transport = ControlPlane()
    monkeypatch.setattr(reconcile, "run_command", transport)
    return site, transport, tmp_path


@pytest.mark.parametrize("stage", ["pod", "exec"])
def test_control_plane_read_failure_is_not_a_reconcile_result(context, stage):
    site, transport, _root = context
    transport.error = stage
    with pytest.raises(BootstrapError, match="Running CPU|execution failed"):
        reconcile.run_control_plane_script(site, {"mode": "plan"})
    assert len(transport.calls) == (1 if stage == "pod" else 2)


@pytest.mark.parametrize("output", ["invalid", "[]"])
def test_control_plane_response_requires_valid_json_object(context, output):
    site, transport, _root = context
    transport.output = output
    with pytest.raises(BootstrapError, match="invalid JSON|non-object"):
        reconcile.run_control_plane_script(site, {"mode": "plan"})
    assert len(transport.calls) == 2


@pytest.mark.parametrize("nodes", [[], {}, {"items": "invalid"}])
def test_cluster_nodes_rejects_unavailable_inventory(context, nodes):
    site, transport, _root = context
    transport.nodes = nodes
    with pytest.raises(BootstrapError, match="invalid node inventory"):
        reconcile.cluster_nodes(site, "gpu-a")
    assert len(transport.calls) == 1


def test_cluster_node_read_failure_does_not_fall_back_to_default_context(context):
    site, transport, _root = context
    transport.error = "nodes"
    with pytest.raises(BootstrapError, match="cannot read GPU nodes"):
        reconcile.cluster_nodes(site, "gpu-a")
    assert len(transport.calls) == 1
    assert "--kubeconfig" in transport.calls[0][0]
    assert "--context" in transport.calls[0][0]


def test_node_evidence_retains_unknown_nodes_as_missing(context):
    site, transport, _root = context
    transport.nodes["items"].extend(
        [None, {}, {"metadata": []}, {"metadata": {"name": ""}}]
    )
    result = reconcile.node_isolation_evidence(
        site, "gpu-a", ["node-a", "missing", "node-a"]
    )
    assert [entry["node_id"] for entry in result] == ["missing", "node-a"]
    assert result[0]["exists"] is False
    assert result[1]["exists"] is True
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "options",
    [
        {"workflow_ids": [" "]},
        {"incident_ids": [""]},
        {"workflow_ids": ["workflow-a"], "incident_ids": ["incident-a"]},
        {"workflow_ids": ["workflow-a"], "max_items": 1},
        {"max_items": 0},
        {"reference": "invalid reference"},
    ],
)
def test_reconcile_rejects_ambiguous_selectors_before_reads(context, options):
    site, transport, root = context
    with pytest.raises(BootstrapError):
        reconcile.run_workflow_reconcile(
            site, root, **{"reference": "CHG-EXAMPLE", **options}
        )
    assert transport.calls == []
    assert not (root / reconcile.HISTORY_PATH).exists(), (
        "invalid selectors created reconciliation evidence"
    )


@pytest.mark.parametrize("raw_items", [None, {}, [None]])
def test_reconcile_requires_structured_runtime_plan_items(context, raw_items):
    site, transport, root = context
    transport.plan["items"] = raw_items
    with pytest.raises(BootstrapError, match="runtime plan is invalid"):
        reconcile.run_workflow_reconcile(site, root, dry_run=True)
    assert [payload["mode"] for payload in transport.payloads] == ["plan"]


def test_missing_cluster_or_node_identity_cannot_be_reconciled(context):
    site, transport, root = context
    transport.plan["items"][0].update(cluster_id="", node_ids=[])
    result = reconcile.run_workflow_reconcile(site, root, dry_run=True)
    assert result["items"][0]["eligible"] is False
    assert result["items"][0]["scheduling_evidence"]["restored"] is False
    assert not any("nodes" in arguments for arguments, _options in transport.calls), (
        "unbound plan triggered a GPU inventory read"
    )
    assert not (root / reconcile.HISTORY_PATH).exists(), "dry-run wrote an archive"


@pytest.mark.parametrize("change", ["removed", "added", "field"])
def test_reconcile_replans_and_refuses_changed_records_before_apply(context, change):
    site, transport, root = context
    second = copy.deepcopy(transport.plan)
    if change == "removed":
        second["items"] = []
    elif change == "added":
        second["items"].append({**second["items"][0], "request_id": "workflow-b"})
    else:
        second["items"][0]["node_ids"] = ["missing"]
    transport.second_plan = second
    with pytest.raises(BootstrapError, match="plan changed before apply"):
        reconcile.run_workflow_reconcile(site, root, reference="CHG-EXAMPLE")
    assert [payload["mode"] for payload in transport.payloads] == ["plan", "plan"]
    assert not (root / reconcile.HISTORY_PATH).exists(), (
        "drifted plan was archived as applied"
    )


def test_public_reconcile_archives_only_after_fresh_plan_and_apply(context):
    site, transport, root = context
    result = reconcile.run_workflow_reconcile(site, root, reference="CHG-EXAMPLE")
    assert result["applied_workflow_ids"] == ["workflow-a"]
    assert result["records_deleted"] == 0
    assert [payload["mode"] for payload in transport.payloads] == [
        "plan",
        "plan",
        "apply",
    ]
    payload = transport.payloads[-1]
    assert payload["workflow_ids"] == ["workflow-a"]
    assert payload["plan_sha256"] == "a" * 64
    assert payload["admin_plan_sha256"] == result["admin_plan_sha256"]
    archive = root / reconcile.HISTORY_PATH / result["plan_sha256"]
    assert json.loads((archive / "applied.json").read_text()) == result
    assert (archive / "plan.json").stat().st_mode & 0o077 == 0
