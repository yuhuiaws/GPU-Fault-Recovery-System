from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from scripts.e2e.regional import audit_warm_spare_guardrails as audit
from tests.regional._cov95_warm_guard import Kubernetes, arguments, node, raw_node
from tests.regional._cov95_warm_guard import configured as configured
from tests.regional.test_guardrail_audit_evidence import recording

pytestmark = pytest.mark.usefixtures("configured")


@pytest.mark.parametrize(
    "field,label",
    [
        ("gpu_kubeconfig", "GPU kubeconfig"),
        ("gpu_context", "GPU context"),
        ("region", "AWS Region"),
        ("managed_gpu_cluster_name", "managed GPU cluster"),
        ("cpu_kubeconfig", "CPU kubeconfig"),
        ("cpu_context", "CPU context"),
    ],
)
def test_missing_configuration_names_the_absent_identity(tmp_path, field, label):
    value = arguments(tmp_path, **{field: ""})
    with pytest.raises(RuntimeError, match=label):
        audit.configure(value, {audit.CASE_IDS[0]})


def test_automatic_case_requires_its_separate_negative_cluster(tmp_path):
    value = arguments(tmp_path, automatic_negative_cluster_name="")
    with pytest.raises(RuntimeError, match="Automatic negative cluster"):
        audit.configure(value, {audit.CASE_IDS[1]})
    audit.configure(value, {audit.CASE_IDS[0]})
    assert audit.AUTOMATIC_NEGATIVE_CLUSTER == "", (
        "the unrelated managed-owner case must not require a negative fixture"
    )


@pytest.mark.parametrize("plane", ["cpu", "gpu"])
def test_configuration_requires_existing_kubeconfig_files(tmp_path, plane):
    value = arguments(tmp_path, **{f"{plane}_kubeconfig": str(tmp_path / "absent")})
    with pytest.raises(RuntimeError, match=f"{plane.upper()} kubeconfig"):
        audit.configure(value, {audit.CASE_IDS[0]})


@pytest.mark.parametrize("fallback", [False, True])
def test_environment_fallbacks_preserve_plane_and_region_binding(
    tmp_path, monkeypatch, fallback
):
    value = arguments(tmp_path)
    for name, supplied in (
        ("KUBECONFIG" if fallback else "GPU_KUBECONFIG", value.gpu_kubeconfig),
        ("GPU_FAULT_DATAPLANE_CONTEXT" if fallback else "GPU_EKS_CONTEXT", "gpu"),
        ("AWS_DEFAULT_REGION" if fallback else "AWS_REGION", "us-west-2"),
        ("GPU_FAULT_HYPERPOD_CLUSTER_NAME", "managed"),
        ("GPU_FAULT_AUTOMATIC_NEGATIVE_CLUSTER_NAME", "negative"),
        (
            "CPU_KUBECONFIG" if fallback else "GPU_FAULT_CONTROL_KUBECONFIG",
            value.cpu_kubeconfig,
        ),
        ("CPU_EKS_CONTEXT" if fallback else "GPU_FAULT_CONTROL_CONTEXT", "cpu"),
    ):
        monkeypatch.setenv(name, supplied)
    for name in vars(value):
        setattr(value, name, None)
    audit.configure(value, set(audit.CASE_IDS))
    assert (audit.CPU_CONTEXT, audit.GPU_CONTEXT, audit.AWS_REGION) == (
        "cpu",
        "gpu",
        "us-west-2",
    ), "fallback configuration must keep each identity on its own plane"
    assert audit.NAMESPACE == "gpu-fault-system", "empty namespace keeps its default"


def test_executor_inventory_reads_every_pod_and_binds_both_uid_observations(
    monkeypatch,
):
    transport = Kubernetes()
    monkeypatch.setattr(audit, "kubectl", transport)
    result = audit.executor_env()
    assert [item["pod"] for item in result] == ["pod-0", "pod-1"], (
        "all desired Ready replicas must supply their own environment"
    )
    assert [item["pod_uid"] for item in result] == ["uid-0", "uid-1"], (
        "environment records must retain their actual Pod incarnation"
    )
    assert transport.reads == {"pod-0": 2, "pod-1": 2}, (
        "each environment probe needs matching before and after identity"
    )


@pytest.mark.parametrize("uid", [None, "", 17])
def test_executor_rejects_missing_or_untyped_pod_identity(monkeypatch, uid):
    transport = Kubernetes()
    transport.uid = uid
    monkeypatch.setattr(audit, "kubectl", transport)
    with pytest.raises(RuntimeError, match="Pod UID"):
        audit.executor_env()
    assert not any("exec" in call[2] for call in transport.calls), (
        "missing identity must be rejected before reading a Pod environment"
    )


@pytest.mark.parametrize("defect", ["recreated", "wrong-hostname"])
def test_executor_environment_cannot_be_attributed_to_a_different_pod(
    monkeypatch, defect
):
    transport = Kubernetes()
    transport.drift = defect == "recreated"
    transport.reported_pod = "foreign" if defect == "wrong-hostname" else None
    monkeypatch.setattr(audit, "kubectl", transport)
    with pytest.raises(RuntimeError, match="changed during"):
        audit.executor_env()


def test_identity_requires_complete_release_and_matching_registration(monkeypatch):
    transport = Kubernetes()
    monkeypatch.setattr(audit, "kubectl", transport)
    result = audit.audit_identity()
    assert result == {
        "release_id": "release-a",
        "cluster_id": "cluster-a",
        "eks_cluster_arn": "arn:aws:eks:us-west-2:111122223333:cluster/gpu",
        "registry_generation": "4",
    }, "the audit must bind the deployed release and durable cluster identity"


@pytest.mark.parametrize(
    "release",
    [
        {"phase": "complete"},
        {"release_id": "", "phase": "complete"},
        {"release_id": "  ", "phase": "complete"},
        {"release_id": 1, "phase": "complete"},
        {"release_id": "release-a", "phase": "rolling"},
    ],
)
def test_unbound_or_partial_release_is_not_a_valid_audit_identity(monkeypatch, release):
    transport = Kubernetes()
    transport.release = release
    monkeypatch.setattr(audit, "kubectl", transport)
    with pytest.raises(RuntimeError, match="complete bound release"):
        audit.audit_identity()


@pytest.mark.parametrize(
    "function",
    [
        audit.audit_identity,
        audit.deployed_managed_owner_probe,
        audit.deployed_executor_guard_probes,
    ],
)
def test_missing_complete_population_prevents_a_deployed_probe(monkeypatch, function):
    transport = Kubernetes()
    monkeypatch.setattr(audit, "kubectl", transport)
    monkeypatch.setattr(audit, "complete_pod_population", lambda *_args: [])
    with pytest.raises(RuntimeError, match="no (Ready|running)"):
        function()
    assert not any("exec" in call[2] for call in transport.calls), (
        "an empty population cannot select a Pod for execution"
    )


@pytest.mark.parametrize("body", [{}, [], None])
def test_deployed_probe_requires_a_nonempty_object(monkeypatch, body):
    transport = Kubernetes()
    transport.probe = body
    monkeypatch.setattr(audit, "kubectl", transport)
    with pytest.raises(RuntimeError, match="nonempty object"):
        audit.deployed_managed_owner_probe()


def test_both_deployed_guard_callers_keep_their_selected_plane(monkeypatch):
    transport = Kubernetes()
    monkeypatch.setattr(audit, "kubectl", transport)
    assert audit.deployed_managed_owner_probe() == transport.probe, (
        "managed-owner guard must return its observed response"
    )
    assert audit.deployed_executor_guard_probes() == transport.probe, (
        "executor guard must return its observed response"
    )
    executions = [call for call in transport.calls if "exec" in call[2]]
    assert [call[1] for call in executions] == ["cpu", "gpu"], (
        "CPU and GPU guard probes must not cross their configured contexts"
    )
    assert all(call[3].get("stdin") for call in executions), (
        "portable probe programs must be sent through private stdin"
    )


def test_node_snapshot_preserves_gpu_identity_taints_and_ownership(monkeypatch):
    first = raw_node("node-z")
    first["spec"]["unschedulable"] = True
    first["spec"]["taints"] = [
        {"key": "z", "effect": "NoExecute", "timeAdded": "2026-09-01T00:00:00Z"},
        {"key": "a", "value": "workload", "effect": "NoSchedule"},
    ]
    first["metadata"]["annotations"][audit.OWNERSHIP_ANNOTATIONS[0]] = "incident-a"
    second = raw_node("node-a")
    cpu = raw_node("cpu")
    cpu["status"]["allocatable"] = {}
    monkeypatch.setattr(
        audit,
        "kubectl",
        lambda *_args, **_kwargs: json.dumps({"items": [first, cpu, second]}),
    )
    result = audit.node_snapshot()
    assert [item["name"] for item in result] == ["node-a", "node-z"], (
        "the snapshot must sort GPU nodes and omit the CPU-only node"
    )
    assert result[1]["uid"] == "node-z-uid" and result[1]["unschedulable"] is True, (
        "identity and existing scheduling state must survive projection"
    )
    assert [item["key"] for item in result[1]["taints"]] == ["a", "z"], (
        "stable taint ordering must not erase semantic fields"
    )
    assert result[1]["ownership_annotations"][audit.OWNERSHIP_ANNOTATIONS[0]] == (
        "incident-a"
    ), "existing GPU fault ownership must remain observable"
    assert audit.node_preflight_errors(result), "preexisting ownership must block audit"


def test_invalid_gpu_quantity_cannot_prove_a_nonempty_gpu_inventory(monkeypatch):
    malformed = raw_node()
    malformed["status"]["allocatable"]["nvidia.com/gpu"] = "unreadable"
    monkeypatch.setattr(
        audit, "kubectl", lambda *_args, **_kwargs: json.dumps({"items": [malformed]})
    )
    with pytest.raises(RuntimeError, match="GPU resource quantity"):
        audit.node_snapshot()


@pytest.mark.parametrize(
    "field,value",
    [
        ("uid", "new-incarnation"),
        ("ready", "False"),
        ("gpu_allocatable", 4),
        ("unschedulable", True),
        ("taints", [{"key": "test", "effect": "NoSchedule"}]),
        ("ownership_annotations", {audit.OWNERSHIP_ANNOTATIONS[0]: "incident"}),
    ],
)
def test_every_recorded_node_property_participates_in_drift(field, value):
    before = [node()]
    after = deepcopy(before)
    after[0][field] = value
    errors = audit.node_state_drift(before, after)
    assert len(errors) == 1 and f"changed {field}" in errors[0], (
        "a changed node fact cannot disappear during postflight comparison"
    )
    assert audit.node_state_drift(before, before) == [], "unchanged facts remain stable"


def test_membership_change_is_not_hidden_by_matching_remaining_nodes():
    assert audit.node_state_drift([node()], [node(), node("node-b")]), (
        "added GPU membership is drift even when the first node is unchanged"
    )


@pytest.mark.parametrize("payload", [{}, {"Events": None}, []])
def test_unreadable_cloudtrail_inventory_cannot_prove_absence(monkeypatch, payload):
    monkeypatch.setattr(audit, "command", lambda *_args, **_kwargs: json.dumps(payload))
    now = datetime.now(timezone.utc)
    with pytest.raises(RuntimeError, match="event inventory is unreadable"):
        audit.replace_events(now, now)


def test_automatic_guard_uses_validated_recording_and_keeps_its_digest(monkeypatch):
    transport = Kubernetes()
    snapshot = recording() | {"recorded_by": "unit-auditor"}
    transport.probe = {"observed_node_recovery": "Automatic"}
    monkeypatch.setattr(audit, "kubectl", transport)
    result = audit.deployed_automatic_recovery_probe(snapshot)
    assert result["payload_provenance"] == {
        key: snapshot[key] for key in ("recorded_at", "payload_digest", "recorded_by")
    }, "the observed guard response must retain its provider recording identity"
    assert result["observed_node_recovery"] == "Automatic", (
        "the caller must keep the actual probe observation"
    )


def test_automatic_guard_never_executes_without_a_complete_executor_population(
    monkeypatch,
):
    transport = Kubernetes()
    monkeypatch.setattr(audit, "kubectl", transport)
    monkeypatch.setattr(audit, "complete_pod_population", lambda *_args: [])
    with pytest.raises(RuntimeError, match="no running cluster executor"):
        audit.deployed_automatic_recovery_probe(recording())


def test_provider_recording_paginates_projects_and_binds_actual_node_details(
    monkeypatch,
):
    calls = []
    nodes = [
        {
            "NodeLogicalId": name,
            "InstanceId": f"instance-{name}",
            "InstanceGroupName": "gpu",
            "InstanceStatus": {"Status": "Running"},
            "unrelated": "not-recorded",
        }
        for name in ("node-a", "node-b")
    ]

    def command(argv, **_kwargs):
        calls.append(list(argv))
        operation = argv[2]
        if operation == "describe-cluster":
            return json.dumps(
                {
                    "ClusterName": "negative",
                    "NodeRecovery": "Automatic",
                    "ClusterStatus": "InService",
                    "unrelated": "not-recorded",
                }
            )
        if operation == "list-cluster-nodes":
            later = "--next-token" in argv
            return json.dumps(
                {
                    "ClusterNodeSummaries": [nodes[int(later)]],
                    "NextToken": None if later else "second-page",
                }
            )
        assert operation == "describe-cluster-node", (
            "recording must use only the three read-only provider operations"
        )
        name = argv[argv.index("--node-logical-id") + 1]
        return json.dumps(
            {
                "NodeDetails": next(
                    item for item in nodes if item["NodeLogicalId"] == name
                )
            }
        )

    monkeypatch.setattr(audit, "command", command)
    snapshot = audit.record_provider_snapshot("negative")
    audit.validate_provider_record(snapshot, "negative")
    payload = snapshot["payloads"]
    assert set(payload["describe_cluster_node"]) == {"node-a", "node-b"}, (
        "every paginated node requires matching detail evidence"
    )
    assert "unrelated" not in payload["describe_cluster"], (
        "recording must not retain unrelated provider fields"
    )
    assert all(
        "unrelated" not in value for value in payload["describe_cluster_node"].values()
    ), "detail projection must keep only the adapter's declared inputs"
    pages = [call for call in calls if call[2] == "list-cluster-nodes"]
    assert len(pages) == 2 and pages[1][-2:] == ["--next-token", "second-page"], (
        "the second page must use the exact provider cursor"
    )


@pytest.mark.parametrize("page", [None, [], {}, {"ClusterNodeSummaries": None}])
def test_unreadable_provider_page_is_not_an_empty_inventory(monkeypatch, page):
    def command(argv, **_kwargs):
        if argv[2] == "describe-cluster":
            return '{"ClusterName":"negative","NodeRecovery":"Automatic"}'
        return json.dumps(page)

    monkeypatch.setattr(audit, "command", command)
    with pytest.raises(RuntimeError, match="node page is unreadable"):
        audit.record_provider_snapshot("negative")
