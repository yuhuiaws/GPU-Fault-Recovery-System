from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from scripts.e2e.regional import notify008_fixture as fixture
from scripts.e2e.regional import notify008_resources as resources
from scripts.e2e.regional import notify008_runner as runner
from scripts.e2e.regional import notify008_target as binding
from scripts.e2e.regional.probes.notify008_protocol import ProbeError
from tests.regional._cov95_notify008_lifecycle import setup_run, target, uid
from tests.regional.test_cov95_notify008_fixture_edges import sandbox_case


def priority_deletes(api):
    return [
        call
        for call in api.calls
        if call[0] == "delete"
        and call[2].startswith("/apis/scheduling.k8s.io/v1/priorityclasses/")
    ]


def test_priority_class_is_unique_cluster_scoped_and_owned_by_the_namespace():
    names = []
    for run_id in (target().run_id, "notify008-fedcba9876543210"):
        approved = replace(target(), run_id=run_id)
        documents = resources.manifests(approved, uid(10), {"probe.py": "pass\n"})
        priority = documents["priorityclass"]
        assert priority == {
            "apiVersion": "scheduling.k8s.io/v1",
            "kind": "PriorityClass",
            "metadata": {
                "name": run_id,
                "labels": {
                    resources.OWNER_LABEL: run_id,
                    resources.CASE_LABEL: runner.CASE_ID,
                },
                "ownerReferences": [
                    {
                        "apiVersion": "v1",
                        "kind": "Namespace",
                        "name": run_id,
                        "uid": uid(10),
                        "controller": False,
                        "blockOwnerDeletion": False,
                    }
                ],
            },
            "value": 0,
            "globalDefault": False,
            "preemptionPolicy": "Never",
        }, "the class must not grant priority or become a cluster default"
        spec = documents["job"]["spec"]["template"]["spec"]
        assert spec["priorityClassName"] == run_id, "the Job must use its own class"
        assert spec["priority"] == 0 and spec["preemptionPolicy"] == "Never", (
            "Pod and PriorityClass admission must agree without enabling preemption"
        )
        names.append(priority["metadata"]["name"])
    assert len(set(names)) == 2 and "mount-s3-critical" not in names, (
        "each run needs its own class, never an existing system class"
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("value",), 1),
        (("value",), 1000000000),
        (("value",), True),
        (("value",), 0.0),
        (("value",), None),
        (("globalDefault",), True),
        (("globalDefault",), "false"),
        (("globalDefault",), 0),
        (("globalDefault",), None),
        (("preemptionPolicy",), "PreemptLowerPriority"),
        (("preemptionPolicy",), None),
        (("apiVersion",), "v1"),
        (("kind",), "ConfigMap"),
        (("spec",), {}),
        (("metadata", "name"), "mount-s3-critical"),
        (("metadata", "namespace"), target().run_id),
        (("metadata", "namespace"), None),
        (("metadata", "uid"), uid(99)),
        (("metadata", "uid"), ""),
        (("metadata", "uid"), 99),
        (("metadata", "labels", resources.OWNER_LABEL), "another-run"),
        (("metadata", "labels", resources.CASE_LABEL), "another-case"),
        (("metadata", "ownerReferences"), []),
        (("metadata", "ownerReferences", 0, "uid"), uid(99)),
        (("metadata", "ownerReferences", 0, "name"), "another-namespace"),
        (("metadata", "ownerReferences", 0, "kind"), "ConfigMap"),
        (("metadata", "ownerReferences", 0, "apiVersion"), "apps/v1"),
        (("metadata", "ownerReferences", 0, "controller"), True),
        (("metadata", "ownerReferences", 0, "blockOwnerDeletion"), True),
        (("metadata", "finalizers"), ["unapproved"]),
        (("metadata", "deletionTimestamp"), "2026-09-16T00:00:00Z"),
    ],
)
def test_priority_class_identity_and_admission_cannot_drift(
    tmp_path, monkeypatch, path, value
):
    sandbox, api = sandbox_case(tmp_path, monkeypatch)
    observed = deepcopy(api.objects["priorityclass"])
    container = observed
    for key in path[:-1]:
        container = container[key]
    container[path[-1]] = value
    calls = list(api.calls)
    with pytest.raises(ProbeError, match="ownership|UID|specification"):
        sandbox.validate_resource("priorityclass", observed)
    assert api.calls == calls, "validation must not mutate or adopt the drifted class"


@pytest.mark.parametrize("omitted", [False, True])
def test_priority_class_allows_only_the_real_global_default_omission(
    tmp_path, monkeypatch, omitted
):
    sandbox, api = sandbox_case(tmp_path, monkeypatch)
    observed = deepcopy(api.objects["priorityclass"])
    if not omitted:
        observed["globalDefault"] = False
    sandbox.validate_resource("priorityclass", observed)
    observed.pop("value")
    with pytest.raises(ProbeError, match="specification"):
        sandbox.validate_resource("priorityclass", observed)


def test_target_preflight_and_creation_reject_a_preexisting_class(
    tmp_path, monkeypatch
):
    settings, api, case_dir, deadline = setup_run(tmp_path, monkeypatch)
    existing = resources.manifests(target(), uid(10), {"probe.py": "pass\n"})[
        "priorityclass"
    ]
    existing["metadata"]["uid"] = uid(99)
    api.objects["priorityclass"] = existing
    with pytest.raises(ProbeError, match="priorityclass already exists"):
        binding.sandbox_preflight(api, target())
    assert runner.read_only_preflight(settings, case_dir)["errors"], (
        "a matching name or label cannot authorize reuse during plan creation"
    )
    sandbox = fixture.Sandbox(api, target(), case_dir, deadline, lambda: None)
    with pytest.raises(ProbeError, match="preexisting PriorityClass"):
        sandbox.create()
    assert not any(call[0] in {"create", "patch", "delete"} for call in api.calls), (
        "a preexisting class must be rejected before any sandbox mutation"
    )
    assert api.objects["priorityclass"] == existing, "the existing class must survive"


def test_priority_class_reads_are_cluster_scoped_and_use_the_run_name(
    tmp_path, monkeypatch
):
    sandbox, api = sandbox_case(tmp_path, monkeypatch)
    sandbox.read("priorityclass")
    assert api.calls[-1] == ("read", "priorityclass", target().run_id, None), (
        "class reads must not use the fixture namespace or shared resource name"
    )
    entry = sandbox.record["resources"]["priorityclass"]
    assert entry["uid"] == api.objects["priorityclass"]["metadata"]["uid"], (
        "the ownership journal must bind the actual create acknowledgement"
    )
    sandbox.record["resources"]["priorityclass"]["uid"] = None
    with pytest.raises(ProbeError, match="acknowledged UID"):
        sandbox.validate_resource("priorityclass", api.objects["priorityclass"])


@pytest.mark.parametrize("garbage_collected", [False, True])
def test_lost_priority_create_ack_never_adopts_readback_or_arms_work(
    tmp_path, monkeypatch, garbage_collected
):
    settings, api, case_dir, deadline = setup_run(tmp_path, monkeypatch)
    api.lose_create.add("priorityclass")
    api.gc_priorityclass = garbage_collected
    assert runner.execute_case(settings, tmp_path, 1, deadline) == 1, (
        "unknown PriorityClass creation cannot be promoted to a successful execution"
    )
    journal = json.loads((case_dir / "notify008-ownership.json").read_text())
    entry = journal["resources"]["priorityclass"]
    assert entry["create_ack_lost"] is True and entry["uid"] is None, (
        "readback must not replace the missing create acknowledgement"
    )
    assert "job" not in journal["resources"] and not api.armed, (
        "no Pod or database work is authorized after a lost class acknowledgement"
    )
    report = json.loads((case_dir / f"{runner.CASE_ID}.json").read_text())
    cleanup = report["cleanup"]
    assert cleanup["namespace_absent"] is True, "the owned namespace must be removed"
    assert cleanup["priorityclass_absent"] is garbage_collected, (
        "namespace absence cannot stand in for observed class absence"
    )
    assert bool(cleanup["errors"]) is not garbage_collected, (
        "a retained global resource must keep cleanup explicitly failed"
    )
    assert priority_deletes(api) == [], "an unknown create UID cannot authorize delete"
    assert ("priorityclass" in api.objects) is not garbage_collected, (
        "only actual namespace-owner garbage collection may remove the unknown UID"
    )


def test_lost_create_ack_and_absent_namespace_still_wait_for_class_gc(
    tmp_path, monkeypatch
):
    sandbox, api = sandbox_case(tmp_path, monkeypatch, created=False)
    api.lose_create.add("priorityclass")
    with pytest.raises(TimeoutError):
        sandbox.create()
    api.objects.pop("namespace")
    read = api.read
    observed = []

    def delayed_gc(kind, *args, **kwargs):
        if kind == "priorityclass":
            observed.append(True)
            if len(observed) == 3:
                api.objects.pop("priorityclass")
        return read(kind, *args, **kwargs)

    monkeypatch.setattr(api, "read", delayed_gc)
    result = sandbox.cleanup()
    assert result["namespace_absent"] and result["priorityclass_absent"], (
        "cleanup must observe class GC even after the namespace has disappeared"
    )
    assert result["errors"] == [] and len(observed) == 3, (
        "namespace absence must not short-circuit cluster-scoped verification"
    )
    assert sandbox.record["resources"]["priorityclass"]["uid"] is None, (
        "cleanup observation cannot become UID custody"
    )
    assert priority_deletes(api) == [], "unknown-UID cleanup must remain read-only"
    assert sandbox.record["cleanup"] == result, "this path must persist cleanup proof"


def test_rejected_acknowledged_class_keeps_uid_for_cleanup_only(tmp_path, monkeypatch):
    sandbox, api = sandbox_case(tmp_path, monkeypatch, created=False)
    api.gc_priorityclass = False
    call = api.call

    def changed_admission(*args, **kwargs):
        text = call(*args, **kwargs)
        if args[0] == "create" and kwargs["body"]["kind"] == "PriorityClass":
            api.objects["priorityclass"]["value"] = 1
            return json.dumps(api.objects["priorityclass"])
        return text

    monkeypatch.setattr(api, "call", changed_admission)
    with pytest.raises(ProbeError, match="specification"):
        sandbox.create()
    assert sandbox.record["resources"]["priorityclass"]["uid"], (
        "an acknowledged owned UID must remain available for rejected-create cleanup"
    )
    assert "job" not in api.objects, "the rejected class cannot admit a Job"
    assert sandbox.cleanup()["errors"] == [] and api.objects == {}, (
        "the rejected owned class must not be orphaned when namespace GC is delayed"
    )
    assert len(priority_deletes(api)) == 1, "cleanup must conditionally delete that UID"


@pytest.mark.parametrize("stage", ["admit", "inspect", "arm", "prepare", "run"])
@pytest.mark.parametrize("drift", ["value", "uid", "absent"])
def test_class_drift_blocks_admission_and_probe_dispatch(
    tmp_path, monkeypatch, stage, drift
):
    sandbox, api = sandbox_case(tmp_path, monkeypatch, admitted=stage != "admit")
    if drift == "absent":
        api.objects.pop("priorityclass")
    elif drift == "uid":
        api.objects["priorityclass"]["metadata"]["uid"] = uid(99)
    else:
        api.objects["priorityclass"]["value"] = 1000000000
    before = len(api.calls)
    with pytest.raises(ProbeError):
        if stage == "admit":
            sandbox.admit()
        else:
            sandbox.execute_probe(stage)
    assert not any(call[0] in {"patch", "exec"} for call in api.calls[before:]), (
        "no scheduling-gate release or probe work may follow class drift"
    )
    assert not api.armed, "replacement or elevated priority must keep work inert"


@pytest.mark.parametrize("armed", [False, True])
@pytest.mark.parametrize("lost_delete_ack", [False, True])
def test_owned_class_cleanup_remains_required_after_namespace_absence(
    tmp_path, monkeypatch, armed, lost_delete_ack
):
    sandbox, api = sandbox_case(tmp_path, monkeypatch)
    priority = deepcopy(api.objects["priorityclass"])
    api.objects = {"priorityclass": priority}
    sandbox.record["arm_started"] = armed
    api.lose_delete = lost_delete_ack
    result = sandbox.cleanup()
    assert result["namespace_absent"] and result["priorityclass_absent"], (
        "namespace absence must still trigger UID-bound class deletion and verification"
    )
    assert result["process_termination_proven"] is not armed, (
        "class cleanup must not weaken the independent process termination requirement"
    )
    assert bool(result["errors"]) is armed, "unproven processes must remain a failure"
    assert len(priority_deletes(api)) == 1 and api.objects == {}, (
        "the class gets exactly one conditional delete, including lost-ACK recovery"
    )
    assert sandbox.record["cleanup"] == result, "absence proof must be journaled"


@pytest.mark.parametrize("drift", ["uid", "owner", "scope", "custody", "read-error"])
def test_absent_namespace_does_not_authorize_foreign_or_unproven_class_deletion(
    tmp_path, monkeypatch, drift
):
    sandbox, api = sandbox_case(tmp_path, monkeypatch)
    priority = api.objects["priorityclass"]
    api.objects = {"priorityclass": priority}
    if drift == "uid":
        priority["metadata"]["uid"] = uid(99)
    elif drift == "owner":
        priority["metadata"]["ownerReferences"][0]["uid"] = uid(99)
    elif drift == "scope":
        priority["metadata"]["namespace"] = target().run_id
    elif drift == "custody":
        sandbox.record["resources"].pop("priorityclass")
    else:
        read = api.read

        def unavailable(kind, *args, **kwargs):
            if kind == "priorityclass":
                raise OSError("class read unavailable")
            return read(kind, *args, **kwargs)

        monkeypatch.setattr(api, "read", unavailable)
    result = sandbox.cleanup()
    assert result["namespace_absent"] and not result["priorityclass_absent"], (
        "unknown or replaced global resources must never be reported absent"
    )
    assert result["errors"] and priority_deletes(api) == [], (
        "namespace absence cannot authorize adoption or deletion without class custody"
    )
    assert api.objects["priorityclass"] == priority, "the unproven class must survive"


def test_priority_class_replacement_during_delete_is_not_retried_or_adopted(
    tmp_path, monkeypatch
):
    sandbox, api = sandbox_case(tmp_path, monkeypatch)
    original = deepcopy(api.objects["priorityclass"])
    api.objects = {"priorityclass": original}
    call = api.call

    def replaced(*args, **kwargs):
        result = call(*args, **kwargs)
        if args[0] == "delete":
            replacement = deepcopy(original)
            replacement["metadata"]["uid"] = uid(99)
            api.objects["priorityclass"] = replacement
        return result

    monkeypatch.setattr(api, "call", replaced)
    cleanup = sandbox.cleanup()
    assert cleanup["errors"] and cleanup["priorityclass_absent"] is False, (
        "a replacement after a successful delete is not absence"
    )
    assert len(priority_deletes(api)) == 1, "the replacement cannot be deleted on retry"
    assert sandbox.record["resources"]["priorityclass"]["uid"] != uid(99), (
        "cleanup must preserve the originally acknowledged UID"
    )
    assert api.objects["priorityclass"]["metadata"]["uid"] == uid(99), (
        "cleanup must leave the replacement untouched"
    )


def test_class_replacement_after_probe_completion_cannot_produce_pass(
    tmp_path, monkeypatch
):
    settings, api, case_dir, deadline = setup_run(tmp_path, monkeypatch)
    api.gc_priorityclass = False
    api.mutate_after_run = lambda cpu: cpu.objects["priorityclass"]["metadata"].update(
        uid=uid(99)
    )
    assert runner.execute_case(settings, tmp_path, 1, deadline) == 1, (
        "a completed probe cannot excuse a replaced class at the final identity check"
    )
    report = json.loads((case_dir / f"{runner.CASE_ID}.json").read_text())
    assert report["verdict"] == "FAIL" and report["cleanup"]["namespace_absent"], (
        "class drift must preserve failure while cleaning the owned namespace"
    )
    assert report["cleanup"]["priorityclass_absent"] is False, (
        "a replacement class cannot count as absence"
    )
    assert priority_deletes(api) == [], "cleanup cannot delete a replacement UID"
    assert api.objects["priorityclass"]["metadata"]["uid"] == uid(99), (
        "a replaced global resource requires explicit reconciliation"
    )


@pytest.mark.parametrize("missing", [False, True])
def test_runner_requires_independent_class_absence_even_when_probe_passes(
    tmp_path, monkeypatch, missing
):
    settings, api, case_dir, deadline = setup_run(tmp_path, monkeypatch)
    cleanup = fixture.Sandbox.cleanup

    def incomplete(self):
        result = cleanup(self)
        if missing:
            result.pop("priorityclass_absent")
        else:
            result["priorityclass_absent"] = False
        return result

    monkeypatch.setattr(fixture.Sandbox, "cleanup", incomplete)
    assert runner.execute_case(settings, tmp_path, 1, deadline) == 1, (
        "namespace and process cleanup cannot substitute for class cleanup evidence"
    )
    report = json.loads((case_dir / f"{runner.CASE_ID}.json").read_text())
    assert report["verdict"] == "FAIL" and api.objects == {}, (
        "missing cleanup proof must fail even when the fake resources happen to be gone"
    )


def test_plan_reserves_the_additional_bounded_class_cleanup_time(tmp_path, monkeypatch):
    settings, api, _, _ = setup_run(tmp_path, monkeypatch)
    details = runner.plan_details(settings, {})
    assert details["cleanup_seconds"] == fixture.CLEANUP_SECONDS == 150, (
        "the plan must reserve namespace cleanup plus bounded class deletion/readback"
    )
    deadline = datetime.now(UTC) + timedelta(
        seconds=target().seconds + fixture.CLEANUP_SECONDS - 10
    )
    assert runner.execute_case(settings, tmp_path, 1, deadline) == 1, (
        "insufficient cleanup time must refuse before resource creation"
    )
    assert not api.calls, "window rejection must happen before reaching Kubernetes"
