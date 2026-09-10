"""Contract tests for GF-REGIONAL-NET-007.

The case proves that a transient Kubernetes API failure met by the GPU-plane
executor leaves the isolation step WAITING with a retryable marker and lets
the workflow close once the failure clears. These tests pin what the runner
accepts as that proof and how tightly it scopes the outage it injects.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import net007_verdicts as verdicts
from scripts.e2e.regional import run_net007_transient_api_outage as net007

NODE = "node-a"
USERNAME = "system:serviceaccount:gpu-fault-system:gpu-fault-cluster-executor"


# --------------------------------------------------------------------------- #
# Outage manifest
# --------------------------------------------------------------------------- #
def _manifest() -> dict[str, Any]:
    return verdicts.webhook_manifest(
        name="gpu-fault-acceptance-net007-x-a1",
        node=NODE,
        namespace="gpu-fault-system",
        username=USERNAME,
        run_id="net007-x-a1",
    )


def test_the_outage_is_scoped_to_one_node_one_client_and_node_updates() -> None:
    manifest = _manifest()
    hook = manifest["webhooks"][0]
    assert hook["rules"] == [
        {
            "apiGroups": [""],
            "apiVersions": ["v1"],
            "operations": ["UPDATE"],
            "resources": ["nodes"],
            "scope": "Cluster",
        }
    ]
    assert hook["objectSelector"] == {"matchLabels": {"kubernetes.io/hostname": NODE}}
    assert hook["matchConditions"][0]["expression"] == (
        f"request.userInfo.username == '{USERNAME}'"
    )
    assert hook["failurePolicy"] == "Fail", hook
    assert hook["timeoutSeconds"] == 1, hook
    assert hook["clientConfig"]["service"]["name"] == verdicts.ABSENT_SERVICE
    assert verdicts.webhook_errors(manifest, node=NODE, username=USERNAME) == []


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda h: h.pop("matchConditions"), "did not keep the matchConditions"),
        (
            lambda h: h["objectSelector"]["matchLabels"].update(
                {"kubernetes.io/hostname": "b"}
            ),
            "pin the node",
        ),
        (lambda h: h["rules"][0]["operations"].append("CREATE"), "nodes/UPDATE only"),
        (lambda h: h["rules"][0]["resources"].append("pods"), "nodes/UPDATE only"),
        (lambda h: h.update(failurePolicy="Ignore"), "failurePolicy"),
        (lambda h: h.update(timeoutSeconds=10), "timeoutSeconds"),
        (
            lambda h: h["clientConfig"]["service"].update(name="real-svc"),
            "absent Service",
        ),
    ],
)
def test_a_server_that_widened_the_outage_is_refused(
    mutate: Any, fragment: str
) -> None:
    """A dropped matchConditions would hit kubelet's node updates too; every
    other widening is equally not the experiment the case describes."""
    applied = _manifest()
    mutate(applied["webhooks"][0])
    errors = verdicts.webhook_errors(applied, node=NODE, username=USERNAME)
    assert any(fragment in error for error in errors), (fragment, errors)


def test_the_deadman_is_bound_to_exactly_its_webhook() -> None:
    names = net007.resource_names("net007-x-a1")
    rbac = net007.rbac_manifests(
        names, namespace="gpu-fault-system", run_id="net007-x-a1"
    )
    role = next(item for item in rbac if item["kind"] == "ClusterRole")
    assert role["rules"] == [
        {
            "apiGroups": ["admissionregistration.k8s.io"],
            "resources": ["validatingwebhookconfigurations"],
            "resourceNames": [names["webhook"]],
            "verbs": ["get", "delete"],
        }
    ]
    job = net007.deadman_manifest(
        names,
        namespace="gpu-fault-system",
        image="img",
        deadman_seconds=840,
        run_id="r",
    )
    spec = job["spec"]["template"]["spec"]
    assert spec["serviceAccountName"] == names["service_account"]
    assert spec["restartPolicy"] == "Never", spec
    assert job["spec"]["backoffLimit"] == 0, job["spec"]
    assert job["spec"]["activeDeadlineSeconds"] > 840, "the Job outlives its deadline"
    env = {item["name"]: item["value"] for item in spec["containers"][0]["env"]}
    assert env == {
        "NET007_DEADMAN_SECONDS": "840",
        "NET007_WEBHOOK_NAME": names["webhook"],
    }
    assert "delete_validating_webhook_configuration" in net007.DEADMAN_SCRIPT, (
        "the deadman must delete the webhook itself"
    )
    assert "404" in net007.DEADMAN_SCRIPT, "an already-removed webhook is success"


# --------------------------------------------------------------------------- #
# Outage evidence
# --------------------------------------------------------------------------- #
def _sample(
    *,
    command_status: str = "WAITING",
    marker: str | None = "retryable_adapter_error",
    step_status: str = "WAITING",
    operation: str = "MARK_UNSCHEDULABLE",
) -> dict[str, Any]:
    details: dict[str, Any] = {"status_source": "executor-retryable-adapter-error"}
    if marker:
        details[marker] = True
    return {
        "observed_at": "2026-09-08T12:00:00+00:00",
        "workflow_status": "RUNNING",
        "step_executions": [
            {"step_index": 0, "operation": "FREEZE_EVIDENCE", "status": "SUCCEEDED"},
            {"step_index": 1, "operation": operation, "status": step_status},
        ],
        "remote_commands": [
            {
                "command_id": "cmd-1",
                "operation": operation,
                "status": command_status,
                "result_details": details,
            }
        ],
    }


def test_the_proof_needs_the_command_waiting_on_a_marker_and_the_step_waiting() -> None:
    evidence = verdicts.outage_evidence([_sample()])
    assert evidence == {
        "observed_at": "2026-09-08T12:00:00+00:00",
        "operation": "MARK_UNSCHEDULABLE",
        "marker": "retryable_adapter_error",
        "command_id": "cmd-1",
        "status_source": "executor-retryable-adapter-error",
    }
    assert verdicts.outage_errors([_sample()]) == []
    transport = _sample(marker="retryable_transport_error")
    assert (
        verdicts.outage_evidence([transport])["marker"] == "retryable_transport_error"
    )


@pytest.mark.parametrize(
    "sample",
    [
        _sample(marker=None),
        _sample(command_status="SUCCEEDED"),
        _sample(step_status="SUCCEEDED"),
        _sample(operation="REMEDIATE_EFA_DRIVER"),
        {"observed_at": "t", "step_executions": [], "remote_commands": []},
    ],
)
def test_samples_without_the_full_proof_are_not_evidence(
    sample: dict[str, Any],
) -> None:
    assert verdicts.outage_evidence([sample]) is None
    errors = verdicts.outage_errors([sample])
    assert any("never met" in error for error in errors), errors


def test_a_step_that_failed_during_the_outage_fails_the_case() -> None:
    errors = verdicts.outage_errors([_sample(), _sample(step_status="FAILED")])
    assert any("FAILED during the outage" in error for error in errors), errors


# --------------------------------------------------------------------------- #
# Recovery
# --------------------------------------------------------------------------- #
def _bundle(**overrides: Any) -> dict[str, Any]:
    workflow = {
        "request_id": "workflow-1",
        "status": "SUCCEEDED",
        "official_steps": [{"operation": op} for op in verdicts.EFA_REMEDIATION_STEPS],
        "step_executions": [
            {"operation": op, "status": "SUCCEEDED"}
            for op in verdicts.EFA_REMEDIATION_STEPS
        ],
    }
    bundle: dict[str, Any] = {
        "workflow": workflow,
        "incident": {"state": "RECOVERED"},
        "matches": [{"workflow": workflow, "incident": {"state": "RECOVERED"}}],
    }
    bundle.update(overrides)
    return bundle


def test_the_recovery_contract_passes_when_the_waited_step_succeeded() -> None:
    assert (
        verdicts.recovery_errors(_bundle(), waited_operation="MARK_UNSCHEDULABLE") == []
    )


def test_each_recovery_defect_is_named() -> None:
    failed_step = _bundle()
    failed_step["workflow"]["step_executions"][1] = {
        "operation": "MARK_UNSCHEDULABLE",
        "status": "FAILED",
    }
    failed_errors = verdicts.recovery_errors(
        failed_step, waited_operation="MARK_UNSCHEDULABLE"
    )
    assert any("waited on the 500 but ended 'FAILED'" in e for e in failed_errors), (
        failed_errors
    )
    chained = _bundle()
    chained["matches"].append(
        {
            "workflow": {"official_steps": [{"operation": "ESCALATE_SUPPORT"}]},
            "incident": {},
        }
    )
    errors = verdicts.recovery_errors(chained, waited_operation=None)
    assert any("grew 2 workflows" in e for e in errors), errors
    assert any(
        "forbidden operations appeared: ['ESCALATE_SUPPORT']" in e for e in errors
    ), errors
    escalated = verdicts.recovery_errors(
        _bundle(incident={"state": "ESCALATED"}), waited_operation=None
    )
    assert any("incident state 'ESCALATED'" in e for e in escalated), escalated


# --------------------------------------------------------------------------- #
# Cleanup, provider, preflight
# --------------------------------------------------------------------------- #
def test_residuals_provider_and_final_node_contracts() -> None:
    assert verdicts.residual_errors({"job/x": False}) == []
    assert verdicts.residual_errors({"job/x": True}) == [
        "outage resources remain: ['job/x']"
    ]
    assert verdicts.provider_errors([]) == []
    assert verdicts.provider_errors([{"event_name": "UpdateCluster"}]) == [
        "provider mutation verbs appeared: ['UpdateCluster']"
    ]
    clean = {
        "ready": "True",
        "unschedulable": False,
        "ownership_annotations": {},
        "taints": [],
    }
    assert verdicts.node_final_errors(clean) == []
    dirty = {
        **clean,
        "unschedulable": True,
        "taints": [{"key": "gpu-fault.io/quarantined"}],
    }
    errors = verdicts.node_final_errors(dirty)
    assert "target node was left cordoned" in errors
    assert "target node was left with a gpu-fault taint" in errors


def _preflight(**overrides: Any) -> list[str]:
    settings = net007.Settings(
        regional=None,  # type: ignore[arg-type]
        node=NODE,
        site_file=__import__("pathlib").Path(__file__),
        host_probe_image="img",
        outage_seconds=60,
    )
    arguments: dict[str, Any] = {
        "node": {
            "ready": "True",
            "unschedulable": False,
            "ownership_annotations": {},
            "labels": {"kubernetes.io/hostname": NODE},
        },
        "workloads": [],
        "state": {
            "queue": {"depth": 0, "fault_backlog_depth": 0},
            "remote_commands": {},
        },
        "executor": {"service_account": "gpu-fault-cluster-executor", "image": "img"},
        "minor": 33,
        "classifier": True,
        "existing_webhook": "",
        "permissions": {"create validatingwebhookconfigurations": True},
        "tests_passed": True,
    }
    arguments.update(overrides)
    return net007.preflight_errors(settings, **arguments)


def test_the_preflight_passes_on_an_idle_node_and_a_classifying_executor() -> None:
    assert _preflight() == []


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        (
            {
                "node": {
                    "ready": "False",
                    "unschedulable": False,
                    "ownership_annotations": {},
                    "labels": {"kubernetes.io/hostname": NODE},
                }
            },
            "not Ready",
        ),
        (
            {
                "node": {
                    "ready": "True",
                    "unschedulable": False,
                    "ownership_annotations": {},
                    "labels": {"kubernetes.io/hostname": "other"},
                }
            },
            "hostname label",
        ),
        ({"workloads": [{"name": "job"}]}, "non-system workload"),
        (
            {"state": {"queue": {"depth": 2, "fault_backlog_depth": 1}}},
            "processor queue",
        ),
        (
            {
                "state": {
                    "queue": {"depth": 1, "fault_backlog_depth": 0},
                    "remote_commands": {},
                }
            },
            None,
        ),
        ({"minor": 29}, "matchConditions"),
        ({"classifier": False}, "record NOT_RUN"),
        ({"existing_webhook": "validatingwebhookconfiguration/x"}, "already exists"),
        ({"permissions": {"create jobs": False}}, "kubeconfig may not"),
        ({"tests_passed": False}, "focused regression"),
    ],
)
def test_each_preflight_refusal_is_named(
    overrides: dict[str, Any], fragment: str | None
) -> None:
    errors = _preflight(**overrides)
    if fragment is None:
        assert not any("processor queue" in e for e in errors), (
            "routine telemetry above the fault tier must not refuse the case"
        )
    else:
        assert any(fragment in error for error in errors), (fragment, errors)


def test_configure_bounds_the_outage() -> None:
    value = net007.parser()
    arguments = value.parse_args(
        [
            "--run-dir",
            "/tmp/run",
            "--node",
            NODE,
            "--site-file",
            "/tmp/site.yaml",
            "--host-probe-image",
            "img",
            "--outage-seconds",
            "10",
        ]
    )
    with pytest.raises(Exception, match="outage seconds"):
        net007.configure(arguments)
    help_text = value.format_help()
    for flag in (
        "--plan",
        "--execute",
        "--confirm",
        "--maintenance-window-end",
        "--outage-seconds",
    ):
        assert flag in help_text, flag


def test_case_constants_match_the_spec() -> None:
    assert verdicts.CASE_ID == "GF-REGIONAL-NET-007"
    assert verdicts.CONFIRMATION == "NET007_EXECUTE"
    assert verdicts.ISOLATION_OPERATIONS == ("MARK_UNSCHEDULABLE", "RESTORE_SCHEDULING")
    assert set(verdicts.FORBIDDEN_OPERATIONS) == {
        "ESCALATE_SUPPORT",
        "QUARANTINE",
        "RESTART_NODE",
        "REPLACE_NODE",
    }
    assert verdicts.executor_username("ns", "sa") == "system:serviceaccount:ns:sa"
    names = net007.resource_names("net007-x-a1")
    assert len(set(names.values())) == 2, "the webhook name and one deadman name"


def test_the_classifier_probe_reads_the_dispatch_module_and_the_node_labels_come_from_kubectl() -> (
    None
):
    """The exception exit moved to cluster_executor/dispatch.py (F6a) and
    node_snapshot carries no labels; attempt 1 (2026-09-10) failed both
    preflight checks on a release that satisfied them."""
    calls: list[tuple[str, ...]] = []

    def kubectl(plane, *args, **_kwargs):
        calls.append((plane, *args))
        if args[0] == "exec":
            return '{"classifier": true}\n'
        return '{"kubernetes.io/hostname": "hp-node-1", "role": "gpu"}'

    regional = SimpleNamespace(kubectl=kubectl)
    assert net007.executor_has_classifier(regional) is True
    probe = calls[-1][-1]
    assert "gpu_fault.cluster_executor.dispatch" in net007.CLASSIFIER_MODULES
    assert probe == net007.CLASSIFIER_PROBE and "importlib" in probe
    assert net007.node_labels(regional, "hp-node-1") == {
        "kubernetes.io/hostname": "hp-node-1",
        "role": "gpu",
    }
    assert calls[-1][:4] == ("gpu", "get", "node", "hp-node-1")
