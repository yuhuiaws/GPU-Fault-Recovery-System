from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from scripts.e2e.regional import notify008_fixture as module
from scripts.e2e.regional.notify008_resources import GATE
from scripts.e2e.regional.probes.notify008_protocol import ProbeError
from tests.regional._cov95_notify008_lifecycle import setup_run, target, uid


def sandbox_case(tmp_path, monkeypatch, *, created=True, admitted=False):
    _, api, case_dir, deadline = setup_run(tmp_path, monkeypatch)
    sandbox = module.Sandbox(api, target(), case_dir, deadline, lambda: None)
    if created:
        sandbox.create()
    if admitted:
        sandbox.admit()
    return sandbox, api


def test_first_pod_validation_failure_is_retained_without_untrusted_image_text(
    tmp_path, monkeypatch
):
    sandbox, _api = sandbox_case(tmp_path, monkeypatch, admitted=True)
    pod = deepcopy(sandbox.current_pod())
    assert pod is not None
    state = next(
        item for item in pod["status"]["containerStatuses"] if item["name"] == "runtime"
    )
    state["imageID"] = "example.invalid/other@sha256:" + "9" * 64
    with pytest.raises(ProbeError, match="resolved image"):
        sandbox.check_pod(pod, gated=False)
    saved = json.loads(sandbox.path.read_text())["pod_validation_failure"]
    assert saved["resolved_image_sha256"]["runtime"] == "9" * 64
    assert saved["errors"] == ["admitted runtime resolved image identity differs"]
    state["imageID"] = "private-untrusted-image-diagnostic"
    with pytest.raises(ProbeError, match="resolved image"):
        sandbox.check_pod(pod, gated=False)
    text = sandbox.path.read_text()
    assert "private-untrusted-image-diagnostic" not in text
    assert json.loads(text)["pod_validation_failure"] == saved


@pytest.mark.parametrize("existing", ["journal", "namespace"])
def test_creation_refuses_existing_custody_without_remote_takeover(
    tmp_path, monkeypatch, existing
):
    sandbox, api = sandbox_case(tmp_path, monkeypatch, created=False)
    if existing == "journal":
        sandbox.path.write_text('{"prior_attempt":true}')
    else:
        api.objects["namespace"] = {"metadata": {"name": target().run_id}}
    with pytest.raises(ProbeError, match="already exists|preexisting"):
        sandbox.create()
    assert not any(call[0] in {"create", "patch", "delete"} for call in api.calls), (
        "an existing journal or namespace cannot authorize automatic takeover"
    )


def test_window_expiry_precedes_target_read_or_any_mutation(tmp_path, monkeypatch):
    sandbox, api = sandbox_case(tmp_path, monkeypatch, created=False)
    sandbox.deadline = datetime.now(UTC) - timedelta(seconds=1)
    sandbox.check_target = lambda: pytest.fail("expired window reached target read")
    with pytest.raises(ProbeError, match="window ended"):
        sandbox.create()
    assert api.calls == [], "an expired approval cannot reach the transport"


@pytest.mark.parametrize(
    "kind", ["namespace", "serviceaccount", "configmap", "networkpolicy", "job"]
)
def test_admitted_resource_mutation_is_rejected_before_use(tmp_path, monkeypatch, kind):
    sandbox, api = sandbox_case(tmp_path, monkeypatch)
    value = deepcopy(api.objects[kind])
    if kind == "namespace":
        value["metadata"]["labels"]["pod-security.kubernetes.io/enforce"] = "privileged"
    elif kind == "serviceaccount":
        value["metadata"]["annotations"] = {"eks.amazonaws.com/role-arn": "unapproved"}
    elif kind == "configmap":
        value["data"]["config.json"] = "{}"
    elif kind == "networkpolicy":
        value["spec"]["egress"] = [{}]
    else:
        value["spec"]["suspend"] = False
    before = list(api.calls)
    with pytest.raises(ProbeError, match="specification differs|security labels"):
        sandbox.validate_resource(kind, value)
    assert api.calls == before, "admitted drift must not trigger compensating mutation"


def test_lost_create_without_a_readback_object_preserves_original_failure(
    tmp_path, monkeypatch
):
    sandbox, api = sandbox_case(tmp_path, monkeypatch, created=False)

    def failed(*args, **kwargs):
        raise TimeoutError("no create acknowledgement")

    monkeypatch.setattr(api, "call", failed)
    with pytest.raises(TimeoutError):
        sandbox.create()
    result = sandbox.cleanup()
    assert result["namespace_absent"] and result["process_termination_proven"], (
        "readback absence before any ARM must close an unapplied create attempt"
    )


@pytest.mark.parametrize("field", ["uid", "resourceVersion"])
def test_owned_patch_refuses_missing_identity_before_transport(
    tmp_path, monkeypatch, field
):
    sandbox, api = sandbox_case(tmp_path, monkeypatch)
    job = deepcopy(api.objects["job"])
    job["metadata"][field] = ""
    before = list(api.calls)
    with pytest.raises(ProbeError):
        if field == "uid":
            sandbox.validate_resource("job", job)
        else:
            sandbox.patch("job", job, [])
    assert api.calls == before, (
        "conditional mutation needs both admitted UID and version"
    )


def test_admission_refuses_a_disappeared_job_or_empty_container_identity(
    tmp_path, monkeypatch
):
    sandbox, api = sandbox_case(tmp_path, monkeypatch)
    job = api.objects.pop("job")
    with pytest.raises(ProbeError, match="Job disappeared"):
        sandbox.admit()
    api.objects["job"] = job
    api.mutate_running = lambda pod: pod["status"]["containerStatuses"][0].update(
        containerID=""
    )
    with pytest.raises(ProbeError, match="container identity"):
        sandbox.admit()
    assert not api.armed, "missing container identity must keep the work barrier closed"


@pytest.mark.parametrize("failure", ["multiple", "replaced", "absent", "no-job"])
def test_current_pod_never_selects_a_replacement_or_ambiguous_child(
    tmp_path, monkeypatch, failure
):
    sandbox, api = sandbox_case(tmp_path, monkeypatch, admitted=True)
    if failure == "multiple":
        monkeypatch.setattr(api, "pods", lambda *args: [api.pod, deepcopy(api.pod)])
    elif failure == "replaced":
        api.pod["metadata"]["uid"] = uid(99)
    elif failure == "absent":
        api.pod = None
    else:
        sandbox.record["resources"].pop("job")
    if failure in {"multiple", "replaced"}:
        with pytest.raises(ProbeError, match="more than one|UID or name"):
            sandbox.current_pod()
    else:
        assert sandbox.current_pod() is None, (
            "absence is explicit and must not select another running Pod"
        )


@pytest.mark.parametrize(
    "phase", ["Failed", "Succeeded", "Pending", "absent", "eventually-running"]
)
def test_readiness_wait_is_bounded_and_refuses_early_termination(
    tmp_path, monkeypatch, phase
):
    sandbox, api = sandbox_case(tmp_path, monkeypatch, admitted=True)
    if phase == "absent":
        api.pod = None
    else:
        api.pod["status"]["phase"] = (
            "Pending" if phase == "eventually-running" else phase
        )
    if phase == "eventually-running":
        pods = api.pods
        replies = []

        def progressing(*args):
            replies.append(True)
            if len(replies) == 2:
                api.pod["status"]["phase"] = "Running"
            return pods(*args)

        monkeypatch.setattr(api, "pods", progressing)
        assert sandbox.wait_pod(gated=False)["metadata"]["uid"] == uid(30), (
            "a bounded wait may accept only the same admitted Pod once running"
        )
        assert len(replies) == 2, "pending state must be observed before readiness"
    else:
        with pytest.raises(ProbeError, match="terminated|timed out"):
            sandbox.wait_pod(gated=False)
    assert not api.armed, "readiness polling cannot arm probe work"


@pytest.mark.parametrize(
    "failure", ["namespace", "pod", "container", "not-object", "failed", "ack"]
)
def test_probe_dispatch_refuses_lost_custody_and_invalid_acknowledgements(
    tmp_path, monkeypatch, failure
):
    sandbox, api = sandbox_case(tmp_path, monkeypatch, admitted=True)
    if failure == "namespace":
        api.objects.pop("namespace")
    elif failure == "pod":
        api.pod = None
    elif failure == "container":
        api.pod["status"]["containerStatuses"][0]["containerID"] = "containerd://other"
    else:
        response = {"not-object": [], "failed": {"verdict": "FAIL"}, "ack": {}}[failure]
        monkeypatch.setattr(api, "call", lambda *args, **kwargs: json.dumps(response))
    with pytest.raises(
        ProbeError, match="disappeared|replaced|refused|identity differs"
    ):
        sandbox.execute_probe("inspect")
    assert not api.armed, "invalid custody or acknowledgement must not arm any work"


@pytest.mark.parametrize("armed", [False, True])
def test_pod_absence_only_proves_no_processes_when_arm_was_never_attempted(
    tmp_path, monkeypatch, armed
):
    sandbox, api = sandbox_case(tmp_path, monkeypatch, admitted=True)
    sandbox.record["arm_started"] = armed
    api.pod = None
    assert sandbox.termination_proof() is not armed, (
        "absence after ARM is not independent proof that both containers stopped"
    )


@pytest.mark.parametrize("drift", ["container", "mapping", "scalar", "null-state"])
def test_termination_evidence_requires_the_original_admitted_containers(
    tmp_path, monkeypatch, drift
):
    sandbox, api = sandbox_case(tmp_path, monkeypatch, admitted=True)
    sandbox.arm()
    sandbox.execute_probe("stop", cleanup=True)
    assert sandbox.termination_proof(), (
        "matching terminated containers must prove shutdown"
    )
    states = api.pod["status"]["containerStatuses"]
    if drift == "container":
        states[0]["containerID"] = "containerd://replacement"
    elif drift == "mapping":
        api.pod["status"]["containerStatuses"] = {"runtime": states[0]}
    elif drift == "scalar":
        states[0] = "runtime"
    else:
        states[0]["state"] = None
    assert sandbox.termination_proof() is False, (
        "malformed or replacement-container status cannot prove owned process termination"
    )


@pytest.mark.parametrize("outcome", ["delayed", "timeout", "replacement", "read-error"])
def test_cleanup_observes_uid_bound_deletion_without_retrying_mutation(
    tmp_path, monkeypatch, outcome
):
    sandbox, api = sandbox_case(tmp_path, monkeypatch, admitted=True)
    sandbox.arm()
    call, read = api.call, api.read
    deleting = []
    reads = []

    def delete_once(*args, **kwargs):
        if args[0] == "delete":
            deleting.append(args)
            if outcome == "replacement":
                api.objects["namespace"]["metadata"]["uid"] = uid(98)
            raise TimeoutError("delete acknowledgement lost")
        return call(*args, **kwargs)

    def read_after_delete(*args, **kwargs):
        if deleting:
            reads.append(args)
            if outcome == "read-error":
                raise OSError("readback unavailable")
            if outcome == "delayed" and len(reads) == 2:
                api.objects.clear()
        return read(*args, **kwargs)

    monkeypatch.setattr(api, "call", delete_once)
    monkeypatch.setattr(api, "read", read_after_delete)
    result = sandbox.cleanup()
    namespace_deletes = [
        item for item in deleting if item[2].startswith("/api/v1/namespaces/")
    ]
    priority_deletes = [
        item
        for item in deleting
        if item[2].startswith("/apis/scheduling.k8s.io/v1/priorityclasses/")
    ]
    assert len(namespace_deletes) == 1 and len(priority_deletes) <= 1, (
        "each resource gets at most one delete; a lost ACK must resolve by readback"
    )
    assert result["namespace_absent"] is (outcome == "delayed"), (
        "only observed absence can establish completed namespace cleanup"
    )
    assert bool(result["errors"]) is (outcome != "delayed"), (
        "unreadable, replaced or retained namespaces must remain failed cleanup evidence"
    )
    assert sandbox.record["cleanup"] == result, "cleanup evidence must be journaled"


def test_cleanup_before_pod_creation_does_not_attempt_an_exec(tmp_path, monkeypatch):
    sandbox, api = sandbox_case(tmp_path, monkeypatch)
    assert sandbox.cleanup()["errors"] == [], (
        "an unstarted gated Job has no runtime process requiring a STOP exec"
    )
    assert not any(call[0] == "exec" for call in api.calls), (
        "cleanup cannot invent a Pod target before admission"
    )


def test_cleanup_without_resource_custody_does_not_claim_observed_absence(
    tmp_path, monkeypatch
):
    sandbox, api = sandbox_case(tmp_path, monkeypatch, created=False)
    result = sandbox.cleanup()
    assert result["namespace_absent"] is False and result["errors"], (
        "without a current ownership attempt the fixture cannot claim namespace absence"
    )
    assert result["process_termination_proven"] is True and api.calls == [], (
        "no work was started and cleanup must not acquire a foreign namespace"
    )


def test_unarmed_gated_pod_proves_no_work_without_container_status(
    tmp_path, monkeypatch
):
    sandbox, api = sandbox_case(tmp_path, monkeypatch)
    job = sandbox.read("job")
    sandbox.patch(
        "job", job, [{"op": "replace", "path": "/spec/suspend", "value": False}]
    )
    assert api.pod["spec"]["schedulingGates"] == [{"name": GATE}], (
        "the test Pod must still be held outside scheduling"
    )
    assert sandbox.termination_proof(), (
        "an unscheduled Pod behind the exact gate cannot have run database work"
    )
