from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import (
    live_driver_guard,
    notify008_fixture,
    regional_case_contract,
    run_notify008_commit_ambiguity,
)
from scripts.e2e.regional import notify008_runner as runner
from scripts.e2e.regional import notify008_target as binding
from scripts.e2e.regional.probes.notify008_protocol import ProbeError
from tests.regional._cov95_notify008_lifecycle import (
    POSTGRES_IMAGE,
    setup_run,
    target,
    uid,
)
from tests.regional._cov95_notify008_support import RUN_ID


@pytest.mark.parametrize(
    "failure",
    [
        "release-missing",
        "release-uncommitted",
        "deployment-missing",
        "replicas",
        "generation",
        "convergence",
        "containers",
        "readiness",
        "source-replaced",
        "owner-kind",
        "replica-missing",
        "replica-owner",
        "image",
        "distribution",
        "node-missing",
        "gpu-node",
        "target-drift",
    ],
)
def test_source_binding_refuses_incomplete_or_drifted_deployed_identity(
    tmp_path, monkeypatch, failure
):
    settings, api, _, _ = setup_run(tmp_path, monkeypatch)
    if failure == "release-missing":
        api.source["configmap"] = None
    elif failure == "release-uncommitted":
        api.source["configmap"]["data"]["state.json"] = json.dumps(
            {"phase": "complete", "transaction_committed": False}
        )
    elif failure == "deployment-missing":
        api.source["deployment"] = None
    elif failure == "replicas":
        api.source["deployment"]["spec"]["replicas"] = True
    elif failure == "generation":
        api.source["deployment"]["metadata"]["generation"] = None
    elif failure == "convergence":
        api.source["deployment"]["status"]["availableReplicas"] = 0
    elif failure == "containers":
        api.source["deployment"]["spec"]["template"]["spec"]["containers"] = []
    elif failure == "readiness":
        api.source["pod"]["status"]["conditions"] = []
    elif failure == "source-replaced":
        api.source["pod"]["metadata"]["name"] = "replacement-pod"
    elif failure == "owner-kind":
        api.source["pod"]["metadata"]["ownerReferences"][0]["kind"] = "Job"
    elif failure == "replica-missing":
        api.source["replicaset"] = None
    elif failure == "replica-owner":
        api.source["replicaset"]["metadata"]["ownerReferences"][0]["uid"] = uid(99)
    elif failure == "image":
        api.source["pod"]["status"]["containerStatuses"][0]["imageID"] = "unapproved"
    elif failure == "distribution":
        call = api.call

        def altered(*args, **kwargs):
            if args[0] == "exec":
                return "{}"
            return call(*args, **kwargs)

        api.call = altered
    elif failure == "node-missing":
        api.source["node"] = None
    elif failure == "gpu-node":
        api.source["node"]["metadata"]["labels"]["node.kubernetes.io/instance-type"] = (
            "p5.48xlarge"
        )
    else:
        api.source["node"]["metadata"]["uid"] = uid(99)
    with pytest.raises(ProbeError):
        binding.source_target(api, settings, RUN_ID, expected=target())
    assert api.objects == {}, (
        "source-binding failure must not create isolated resources"
    )


def test_preflight_uses_current_canonical_case_and_never_mutates(tmp_path, monkeypatch):
    settings, api, case_dir, _ = setup_run(tmp_path, monkeypatch)
    monkeypatch.setattr(binding, "case_metadata", regional_case_contract.case_metadata)
    monkeypatch.setattr(
        binding, "predecessor_path", regional_case_contract.predecessor_path
    )
    monkeypatch.setattr(
        runner, "uuid4", lambda: SimpleNamespace(hex="0123456789abcdef" + "0" * 16)
    )

    result = runner.read_only_preflight(settings, case_dir)

    assert result["errors"] == [], (
        "registered canonical NOTIFY007 must unlock preflight"
    )
    assert result["predecessor"]["case_id"] == "GF-REGIONAL-NOTIFY-007", (
        "the host must derive its predecessor from canonical order"
    )
    assert result["target"]["run_id"] == RUN_ID, (
        "the plan must bind a specific isolated run"
    )
    assert not any(call[0] in {"create", "patch", "delete"} for call in api.calls), (
        "preflight must remain read-only"
    )
    details = runner.plan_details(settings, result)
    assert (
        details["provider"] == "SIMULATED" and details["risk"] == "live-non-destructive"
    ), "the plan must preserve the approved scope and risk"


def test_selective_evidence_cannot_satisfy_formal_predecessor(tmp_path, monkeypatch):
    settings, api, case_dir, _ = setup_run(tmp_path, monkeypatch)
    path = tmp_path / "cases/GF-REGIONAL-NOTIFY-007/GF-REGIONAL-NOTIFY-007.json"
    previous = json.loads(path.read_text())
    previous.update(execution_scope="selective", formal_sequence_satisfied=False)
    path.write_text(json.dumps(previous))
    with pytest.raises(ProbeError, match="predecessor"):
        binding.predecessor(tmp_path, target())
    assert runner.read_only_preflight(settings, case_dir)["errors"], (
        "a selective predecessor must keep formal preflight closed"
    )
    assert api.objects == {}, "rejected chain evidence must not create resources"


@pytest.mark.parametrize("problem", ["risk", "missing-predecessor"])
def test_canonical_metadata_refusals_do_not_invent_a_predecessor(
    tmp_path, monkeypatch, problem
):
    setup_run(tmp_path, monkeypatch)
    if problem == "risk":
        monkeypatch.setattr(
            binding,
            "case_metadata",
            lambda case: SimpleNamespace(risk="non-destructive", automation="command"),
        )
    else:
        monkeypatch.setattr(binding, "predecessor_path", lambda *args: (None, None))
    with pytest.raises(ProbeError):
        binding.predecessor(tmp_path, target())


def test_cpu_transport_binds_context_file_and_structured_mutation_body(
    tmp_path, monkeypatch
):
    path = tmp_path / "cpu"
    path.write_text("fake config")
    calls = []

    def command(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "response", "")

    monkeypatch.setattr(notify008_fixture, "run_fixture_command", command)
    api = notify008_fixture.CpuAPI(path, "cpu-context")
    assert (
        api.call("create", "-f", "-", namespace=RUN_ID, body={"kind": "ConfigMap"})
        == "response"
    ), "the transport must return the checked command output"
    argv, options = calls[0]
    assert argv[:5] == [
        "kubectl",
        "--kubeconfig",
        str(path),
        "--context",
        "cpu-context",
    ], "every operation must bind the exact CPU context and kubeconfig"
    assert json.loads(options["input_text"]) == {"kind": "ConfigMap"}, (
        "structured resources must not be interpolated into a shell"
    )
    path.write_text("changed config")
    with pytest.raises(ProbeError, match="kubeconfig changed"):
        api.call("get", "pods")
    assert len(calls) == 1, "connection drift must stop before another transport call"


@pytest.mark.parametrize("reply", ["[]", "{}", '{"metadata":{"name":"other"}}'])
def test_cpu_read_refuses_malformed_or_replaced_identity(tmp_path, monkeypatch, reply):
    path = tmp_path / "cpu"
    path.write_text("fake")
    monkeypatch.setattr(
        notify008_fixture,
        "run_fixture_command",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, reply, ""),
    )
    with pytest.raises(ProbeError):
        notify008_fixture.CpuAPI(path, "cpu").read("pod", "expected")


def test_cpu_read_and_pod_list_have_explicit_absence_and_shape_contracts(
    tmp_path, monkeypatch
):
    path = tmp_path / "cpu"
    path.write_text("fake")
    replies = iter(
        ["", '{"metadata":{"name":"expected"}}', "{}", '{"items":[1]}', '{"items":[]}']
    )
    monkeypatch.setattr(
        notify008_fixture,
        "run_fixture_command",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, next(replies), ""),
    )
    api = notify008_fixture.CpuAPI(path, "cpu")
    assert api.read("pod", "expected") is None, (
        "only successful empty output means absence"
    )
    assert api.read("pod", "expected") == {"metadata": {"name": "expected"}}, (
        "valid resource identity must be preserved"
    )
    with pytest.raises(ProbeError):
        api.pods(RUN_ID, "job")
    with pytest.raises(ProbeError):
        api.pods(RUN_ID, "job")
    assert api.pods(RUN_ID, "job") == [], "a valid empty inventory must remain empty"


def test_entrypoint_uses_the_existing_approved_case_driver(monkeypatch):
    called = []
    monkeypatch.setattr(
        live_driver_guard, "run_standard_case", lambda case: called.append(case) or 0
    )
    assert run_notify008_commit_ambiguity.main() == 0, (
        "the host entry must delegate to the guarded driver"
    )
    assert called == [runner.CASE], (
        "the entrypoint must not select another case or bypass its guard"
    )


def test_unapproved_execute_does_not_supersede_existing_evidence(tmp_path, monkeypatch):
    _, api, case_dir, _ = setup_run(tmp_path, monkeypatch)
    current = case_dir / f"{runner.CASE_ID}.json"
    current.write_text(json.dumps({"verdict": "PASS", "prior": True}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_notify008_commit_ambiguity.py",
            "--run-dir",
            str(tmp_path),
            "--cpu-kubeconfig",
            str(tmp_path / "cpu-kubeconfig"),
            "--cpu-context",
            "cpu",
            "--cluster-id",
            "cluster-local",
            "--region",
            "us-west-2",
            "--postgres-image",
            POSTGRES_IMAGE,
            "--execute",
            "--maintenance-window-end",
            (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
        ],
    )
    with pytest.raises(RuntimeError, match="confirmation"):
        run_notify008_commit_ambiguity.main()
    assert json.loads(current.read_text()) == {"verdict": "PASS", "prior": True}, (
        "only an authorized execute attempt may invalidate canonical evidence"
    )
    assert api.calls == [], "failed authorization must not reach Kubernetes"
