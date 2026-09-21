from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import destr008_controller_lock as locking
from scripts.e2e.regional import destr008_gpu_holder as holder
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.seeded_command_fixture import RUN_LABEL, SeededCommandError
from scripts.e2e.regional.warm_spare_fixture import GpuHolderFixture
from tests.regional._destr008_gpu_holder import (
    HolderHarness,
    admitted_pod,
    build_holder,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> HolderHarness:
    return build_holder(tmp_path, monkeypatch)


def set_field(value: dict[str, Any], path: tuple[str | int, ...], item: Any) -> None:
    current: Any = value
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = copy.deepcopy(item)


def test_create_ack_is_private_durable_and_cleanup_is_uid_rv_bound(
    harness: HolderHarness,
) -> None:
    fixture = harness.controller
    base = GpuHolderFixture(fixture.warm, node=fixture.node, run_id=fixture.run_id)
    expected = base.manifest()
    expected["spec"].update(
        automountServiceAccountToken=False, enableServiceLinks=False
    )
    assert fixture.manifest() == expected, "only credential-free Pod settings differ"
    assert fixture.deadline_at is None, "constructing a holder must not arm it"
    fixture.create()
    record = harness.journal()
    assert record["phase"] == "READY", "readiness must be observed after durable ACK"
    assert record["pod_uid"] == "holder-uid", "only the direct ACK grants custody"
    assert harness.pod is not None, "the local fake committed a Pod"
    assert record["pod_spec_sha256"] == holder.digest(harness.pod["spec"]), (
        "the entire admitted spec must be pinned"
    )
    assert record["deadline_at"] == record["created_at"] + base.HOLD_SECONDS, (
        "the bound begins before CREATE, not readiness"
    )
    assert fixture.path.stat().st_mode & 0o777 == 0o600, "the journal must be private"
    assert fixture.path.parent.stat().st_mode & 0o777 == 0o700, (
        "the ownership directory must be private"
    )
    deadline = fixture.deadline_at
    harness.clock.sleep(base.HOLD_SECONDS + 100)
    resumed = harness.new_holder()
    assert resumed.resumed and resumed.deadline_at == deadline, (
        "reconstruction must recover the original expired bound"
    )
    assert resumed.resume_cleanup() is False, "cleanup must verify actual absence"
    assert harness.pod is None, "the acknowledged Pod must be removed"
    assert harness.journal()["deadline_at"] == record["deadline_at"], (
        "cleanup must never extend the deadline"
    )
    assert harness.journal()["phase"] == "CLOSED", "terminal cleanup must be durable"
    assert fixture.cleanup() is False, (
        "a stale controller must reload the closed journal"
    )
    assert [args[0] for args in harness.mutations()] == ["create", "delete"], (
        "only one CREATE and one UID/RV DELETE are permitted"
    )
    for controller in (fixture, resumed, harness.new_holder()):
        with pytest.raises(RegionalFixtureError, match="cleanup-only"):
            controller.create()


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("present", [False, True])
def test_unknown_create_never_adopts_public_metadata_or_absence(
    harness: HolderHarness, committed: bool, present: bool
) -> None:
    if committed:
        harness.lost_create = True
    else:
        harness.create_error = TimeoutError("test-only private diagnostic")
    with pytest.raises(RegionalFixtureError, match="TimeoutError") as error:
        harness.controller.create()
    assert "private diagnostic" not in str(error.value), "API errors must be sanitized"
    record = harness.journal()
    assert record["create_started"] and record["pod_uid"] is None, (
        "an unknown ACK cannot grant ownership"
    )
    harness.pod = admitted_pod(harness.controller.manifest()) if present else None
    count = len(harness.calls)
    for controller in (harness.controller, harness.new_holder()):
        with pytest.raises(RegionalFixtureError, match="cleanup-only"):
            controller.create()
        with pytest.raises(RegionalFixtureError, match="outcome is unknown"):
            controller.resume_cleanup()
    assert len(harness.calls) == count, (
        "unknown ACK must refuse before any adoption read"
    )
    assert harness.journal() == record, (
        "failure evidence and deadline must survive retry"
    )


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "{",
        "[]",
        '{"kind":"Pod","kind":"Pod"}',
        " " * 1048577,
        '{"spec":{"value":NaN}}',
        '{"spec":{"value":Infinity}}',
        '{"spec":{"value":-Infinity}}',
    ],
)
def test_invalid_direct_ack_is_not_repaired_by_a_good_readback(
    harness: HolderHarness, raw: str
) -> None:
    harness.ack_raw = raw
    with pytest.raises(RegionalFixtureError, match="bounded object"):
        harness.controller.create()
    assert harness.pod is not None, "CREATE was committed before malformed ACK"
    with pytest.raises(RegionalFixtureError, match="outcome is unknown"):
        harness.new_holder().cleanup()
    assert harness.journal()["pod_uid"] is None, "GET must never repair a missing ACK"


POD_DEFECTS: list[tuple[tuple[str | int, ...], Any]] = [
    (("apiVersion",), "batch/v1"),
    (("kind",), "Job"),
    (("metadata",), []),
    (("metadata", "name"), "foreign"),
    (("metadata", "namespace"), "foreign"),
    (("metadata", "uid"), None),
    (("metadata", "uid"), ""),
    (("metadata", "uid"), "bad/uid"),
    (("metadata", "resourceVersion"), None),
    (("metadata", "resourceVersion"), ""),
    (("metadata", "ownerReferences"), [{"uid": "other"}]),
    (("metadata", "labels"), ["bad"]),
    (("metadata", "labels"), {}),
    (("metadata", "labels", RUN_LABEL), "foreign"),
    (("metadata", "labels", "app"), "foreign"),
    (("spec",), []),
    (("spec", "nodeName"), "foreign"),
    (("spec", "automountServiceAccountToken"), True),
    (("spec", "activeDeadlineSeconds"), 9999),
    (("spec", "activeDeadlineSeconds"), True),
    (("spec", "restartPolicy"), "Always"),
    (("spec", "hostNetwork"), True),
    (("spec", "priority"), False),
    (("spec", "volumes"), [{"name": "credentials", "secret": {"secretName": "other"}}]),
    (("spec", "initContainers"), [{"name": "sidecar", "image": "unknown"}]),
    (("spec", "containers"), None),
    (("spec", "containers"), []),
    (("spec", "containers"), [{}, {}]),
    (("spec", "containers"), ["bad"]),
    (("spec", "containers", 0, "image"), "unapproved"),
    (("spec", "containers", 0, "command"), ["sleep", "forever"]),
    (("spec", "containers", 0, "env"), [{"name": "UNAPPROVED", "value": "value"}]),
    (("spec", "containers", 0, "stdin"), 0),
    (("spec", "containers", 0, "resources", "requests", "nvidia.com/gpu"), "2"),
]


@pytest.mark.parametrize(("path", "replacement"), POD_DEFECTS)
@pytest.mark.parametrize("stage", ["ack", "cleanup"])
def test_full_pod_identity_and_spec_must_match_before_custody_or_delete(
    harness: HolderHarness, path: tuple[str | int, ...], replacement: Any, stage: str
) -> None:
    if stage == "ack":
        harness.ack_change = lambda pod: set_field(pod, path, replacement)
        with pytest.raises(RegionalFixtureError):
            harness.controller.create()
        valid_identity = path[0] == "spec" or path[:2] in {
            ("metadata", "labels"),
            ("metadata", "ownerReferences"),
        }
        assert harness.journal()["pod_uid"] == (
            "holder-uid" if valid_identity else None
        ), "a valid direct identity grants deletion-only, never execution, custody"
        assert not harness.journal()["pod_approved"], (
            "unapproved ACKs cannot authorize readiness"
        )
    else:
        harness.controller.create()
        assert harness.pod is not None, "the test requires the acknowledged Pod"
        set_field(harness.pod, path, replacement)
        with pytest.raises(RegionalFixtureError):
            harness.new_holder().cleanup()
    assert [args[0] for args in harness.mutations()] == ["create"], (
        "no mismatched Pod may be deleted"
    )


@pytest.mark.parametrize("change", ["uid", "raw-spec-default"])
def test_readback_must_match_direct_ack_even_when_manifest_semantics_match(
    harness: HolderHarness, change: str
) -> None:
    harness.controller.create()
    assert harness.pod is not None, "the test requires an acknowledged Pod"
    if change == "uid":
        harness.pod["metadata"]["uid"] = "replacement-uid"
    else:
        harness.pod["spec"]["hostNetwork"] = False
    with pytest.raises(RegionalFixtureError, match="replaced|acknowledged spec"):
        harness.new_holder().cleanup()
    assert len(harness.mutations()) == 1, (
        "a different admitted spec must not be deleted"
    )


@pytest.mark.parametrize("stage", ["ack", "readiness"])
def test_deleting_holder_is_never_ready(harness: HolderHarness, stage: str) -> None:
    if stage == "ack":
        harness.ack_change = lambda pod: pod["metadata"].update(deletionTimestamp="now")
    else:

        def deleting(args: tuple[str, ...]) -> None:
            if args[:2] == ("get", "pod") and harness.pod is not None:
                harness.pod["metadata"]["deletionTimestamp"] = "now"

        harness.before = deleting
    with pytest.raises(RegionalFixtureError, match="deleting"):
        harness.controller.create()
    assert harness.journal()["phase"] == "CREATED", (
        "direct ACK custody survives deletion"
    )
    assert harness.journal()["pod_approved"] == (stage == "readiness"), (
        "an already deleting direct ACK is not approved for execution"
    )


@pytest.mark.parametrize(
    "status",
    [
        [],
        {"phase": "Other"},
        {"phase": ["Running"]},
        {"phase": "Succeeded"},
        {"phase": "Failed"},
        {"phase": "Unknown"},
        {"phase": "Running", "conditions": {}},
        {"phase": "Running", "conditions": [None]},
        {"phase": "Running", "conditions": [{"type": None, "status": "True"}]},
        {"phase": "Running", "conditions": [{"type": "", "status": "True"}]},
        {"phase": "Running", "conditions": [{"type": "Ready", "status": True}]},
        {"phase": "Running", "conditions": [{"type": "Ready", "status": ["True"]}]},
        {
            "phase": "Running",
            "conditions": [
                {"type": "Ready", "status": "True"},
                {"type": "Ready", "status": "False"},
            ],
        },
    ],
)
def test_invalid_or_terminal_status_cannot_prove_readiness(
    harness: HolderHarness, status: Any
) -> None:
    def change(args: tuple[str, ...]) -> None:
        if args[:2] == ("get", "pod") and harness.pod is not None:
            harness.pod["status"] = status

    harness.before = change
    with pytest.raises(RegionalFixtureError, match="status|ended"):
        harness.controller.create()
    assert harness.journal()["phase"] == "CREATED", "terminal status is not readiness"


@pytest.mark.parametrize("condition", [[], [{"type": "Ready", "status": "False"}]])
def test_readiness_wait_does_not_extend_holder_deadline(
    harness: HolderHarness, condition: list[dict[str, str]]
) -> None:
    started = harness.clock.time()

    def pending(args: tuple[str, ...]) -> None:
        if args[:2] == ("get", "pod") and harness.pod is not None:
            harness.pod["status"] = (
                {"phase": "Running", "conditions": condition}
                if harness.clock.elapsed < 3
                else {
                    "phase": "Running",
                    "conditions": [{"type": "Ready", "status": "True"}],
                }
            )

    harness.before = pending
    harness.controller.create()
    assert harness.clock.elapsed == 3, "the controller must wait for real readiness"
    assert (
        harness.journal()["deadline_at"] == started + GpuHolderFixture.HOLD_SECONDS
    ), "readiness latency must consume the original lifetime"


@pytest.mark.parametrize("defect", ["pending", "expired", "missing"])
def test_readiness_failure_preserves_ack_for_cleanup_only(
    harness: HolderHarness, defect: str
) -> None:
    def fail(args: tuple[str, ...]) -> None:
        if args[:2] == ("get", "pod") and harness.pod is not None:
            if defect == "missing":
                harness.pod = None
            else:
                harness.pod["status"] = {"phase": "Pending"}
                if defect == "expired":
                    harness.clock.sleep(GpuHolderFixture.HOLD_SECONDS)

    harness.before = fail
    with pytest.raises(RegionalFixtureError, match="deadline|disappeared"):
        harness.controller.create()
    assert harness.journal()["pod_uid"] == "holder-uid", (
        "readiness failure retains custody"
    )
    deadline = harness.journal()["deadline_at"]
    harness.before = None
    harness.new_holder().resume_cleanup()
    assert harness.journal()["deadline_at"] == deadline, "cleanup cannot rearm a holder"
    assert harness.pod is None, "cleanup must remove the originally acknowledged Pod"


def test_slow_identity_check_cannot_submit_after_the_original_deadline(
    harness: HolderHarness,
) -> None:
    reads = 0

    def delay(args: tuple[str, ...]) -> None:
        nonlocal reads
        if args[:2] == ("get", "node"):
            reads += 1
            if reads == 2:
                harness.clock.sleep(GpuHolderFixture.HOLD_SECONDS)

    harness.before = delay
    with pytest.raises(RegionalFixtureError, match="creation deadline"):
        harness.controller.create()
    assert not harness.mutations(), "an expired create intent cannot issue CREATE"
    assert harness.journal()["phase"] == "CREATING", "uncertainty must remain durable"


@pytest.mark.parametrize("budget", ["readiness", "lifetime"])
def test_late_ready_read_cannot_approve_an_expired_holder(
    harness: HolderHarness, budget: str
) -> None:
    def delay(args: tuple[str, ...]) -> None:
        if args[:2] == ("get", "pod") and harness.pod is not None:
            if budget == "readiness":
                harness.clock.elapsed += holder.READY_SECONDS
            else:
                harness.clock.epoch += GpuHolderFixture.HOLD_SECONDS

    harness.before = delay
    with pytest.raises(RegionalFixtureError, match="readiness deadline expired"):
        harness.controller.create()
    assert harness.journal()["phase"] == "CREATED", (
        "a late Ready result is not timely readiness"
    )
    harness.before = None
    harness.new_holder().resume_cleanup()
    assert harness.pod is None, "late evidence only permits cleanup of the owned Pod"


@pytest.mark.parametrize("present", [False, True])
def test_never_created_cleanup_tombstone_does_not_adopt_existing_pod(
    harness: HolderHarness, present: bool
) -> None:
    if present:
        harness.pod = admitted_pod(harness.controller.manifest())
        with pytest.raises(RegionalFixtureError, match="no direct"):
            harness.controller.resume_cleanup()
    else:
        assert harness.controller.resume_cleanup() is False, (
            "absence can close NOT_STARTED"
        )
        assert harness.journal()["phase"] == "CLOSED", (
            "the no-create tombstone is durable"
        )
    with pytest.raises(RegionalFixtureError, match="cleanup-only"):
        harness.new_holder().create()
    assert not harness.mutations(), "neither branch may create or delete anything"


def test_preexisting_pod_never_grants_holder_custody(harness: HolderHarness) -> None:
    harness.pod = admitted_pod(harness.controller.manifest())
    with pytest.raises(RegionalFixtureError, match="already in use"):
        harness.controller.create()
    with pytest.raises(RegionalFixtureError, match="no direct"):
        harness.new_holder().cleanup()
    assert not harness.journal()["create_started"], "preexistence must block CREATE"
    assert not harness.mutations(), "even identical labels/spec cannot prove ownership"


@pytest.mark.parametrize("recreated", [False, True])
def test_absent_acknowledged_holder_is_closed_but_recreated_closed_name_is_refused(
    harness: HolderHarness, recreated: bool
) -> None:
    harness.controller.create()
    harness.pod = None
    harness.new_holder().cleanup()
    if recreated:
        harness.pod = admitted_pod(harness.controller.manifest())
        with pytest.raises(RegionalFixtureError, match="recreated"):
            harness.controller.cleanup()
    else:
        assert harness.controller.cleanup() is False, "absence can be reverified"
    assert len(harness.mutations()) == 1, "cleanup must not delete a recreated name"


@pytest.mark.parametrize("committed", [False, True])
def test_delete_ack_failure_retries_only_the_owned_delete(
    harness: HolderHarness, committed: bool
) -> None:
    harness.controller.create()
    if committed:
        harness.lost_delete = True
        harness.controller.cleanup()
        assert harness.journal()["phase"] == "CLOSED", (
            "known UID absence resolves DELETE"
        )
    else:
        harness.delete_error = TimeoutError("test-only diagnostic")
        with pytest.raises(RegionalFixtureError, match="TimeoutError"):
            harness.controller.cleanup()
        assert harness.journal()["phase"] == "CLEANING", (
            "retryable cleanup retains custody"
        )
        harness.delete_error = None
        harness.new_holder().resume_cleanup()
    assert harness.pod is None, "cleanup must observe removal after the owned delete"
    assert sum(args[0] == "create" for args in harness.mutations()) == 1, (
        "cleanup retry cannot create a replacement"
    )


@pytest.mark.parametrize("field", ["resourceVersion", "uid"])
def test_race_after_full_spec_validation_is_refused_before_delete(
    harness: HolderHarness, field: str
) -> None:
    harness.controller.create()

    def change(args: tuple[str, ...]) -> None:
        if args[-1] == "jsonpath={.metadata}" and harness.pod is not None:
            harness.pod["metadata"][field] = "changed"

    harness.before = change
    with pytest.raises(SeededCommandError, match="changed|replaced"):
        harness.controller.cleanup()
    assert len(harness.mutations()) == 1, (
        "full-read/metadata-read races must not delete"
    )


def test_recreation_after_delete_cannot_be_reported_as_removed(
    harness: HolderHarness,
) -> None:
    harness.controller.create()

    def recreate(args: tuple[str, ...]) -> None:
        if args[:2] == ("get", "pod") and args[-1] == "json" and harness.pod is None:
            harness.pod = admitted_pod(harness.controller.manifest())
            harness.pod["metadata"]["uid"] = "recreated-uid"

    harness.before = recreate
    with pytest.raises(RegionalFixtureError, match="unconfirmed"):
        harness.controller.cleanup()
    assert harness.journal()["phase"] == "CLEANING", (
        "absence cannot be inferred from DELETE"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("node", ""),
        ("node", 3),
        ("run_id", "bad/run"),
        ("node_uid", ""),
        ("release_id", "bad\nrelease"),
        ("plan_sha256", "B" * 64),
        ("plan_sha256", 123),
    ],
)
def test_constructor_requires_strict_scope(
    harness: HolderHarness, field: str, value: Any
) -> None:
    with pytest.raises(RegionalFixtureError, match="identity|digest"):
        harness.new_holder(**{field: value})
    assert not harness.calls, "invalid scope must fail before any API access"


@pytest.mark.parametrize("field", ["node", "node_uid", "plan_sha256", "release_id"])
def test_existing_run_cannot_be_rebound_to_another_target_or_plan(
    harness: HolderHarness, field: str
) -> None:
    harness.controller.create()
    harness.calls.clear()
    value = "b" * 64 if field == "plan_sha256" else "another"
    with pytest.raises(RegionalFixtureError, match="binding changed"):
        harness.new_holder(**{field: value})
    assert not harness.calls, "same run must use the original journal and bindings"


@pytest.mark.parametrize(
    "field",
    [
        "cluster_id",
        "namespace",
        "gpu_context",
        "region",
        "cpu_kubeconfig",
        "gpu_kubeconfig",
    ],
)
def test_connection_drift_is_refused_before_commands(
    harness: HolderHarness, field: str
) -> None:
    harness.controller.create()
    harness.calls.clear()
    settings = harness.regional.settings
    if field.endswith("kubeconfig"):
        getattr(settings, field).write_text("changed local test configuration\n")
    else:
        changes: dict[str, Any] = {field: "different"}
        harness.regional.settings = replace(settings, **changes)
    with pytest.raises(RegionalFixtureError, match="connection or source"):
        harness.controller.cleanup()
    with pytest.raises(RegionalFixtureError, match="binding changed"):
        harness.new_holder()
    assert not harness.calls, "connection drift must not issue commands"


@pytest.mark.parametrize("field", ["name", "uid", "release", "cluster"])
def test_live_node_and_release_revalidated_before_every_mutation(
    harness: HolderHarness, field: str
) -> None:
    harness.controller.create()
    count = len(harness.mutations())
    if field in {"name", "uid"}:
        harness.node["metadata"][field] = "changed"
    else:
        harness.identity[field + "_id"] = "changed"
    with pytest.raises(RegionalFixtureError, match="Node UID|release or cluster"):
        harness.new_holder().cleanup()
    assert len(harness.mutations()) == count, "drift must not permit an owned delete"


def test_identity_read_errors_are_not_absence_and_are_sanitized(
    harness: HolderHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail() -> dict[str, str]:
        raise RuntimeError("test-only sensitive diagnostic")

    monkeypatch.setattr(harness.regional, "evidence_identity", fail)
    with pytest.raises(RegionalFixtureError, match="identity read failed") as error:
        harness.controller.create()
    assert "sensitive diagnostic" not in str(error.value), (
        "external messages are not evidence"
    )
    assert not harness.mutations(), "failed reads must not authorize CREATE"


def test_missing_journal_cannot_reset_a_stale_controller(
    harness: HolderHarness,
) -> None:
    harness.controller.create()
    harness.controller.path.unlink()
    harness.pod = None
    with pytest.raises(RegionalFixtureError, match="journal disappeared"):
        harness.controller.create()
    assert len(harness.mutations()) == 1, "loss of local evidence cannot rearm"


@pytest.mark.parametrize("when", ["before-create", "after-ack", "delete"])
def test_supervision_loss_is_sticky_across_reconstruction(
    harness: HolderHarness, when: str
) -> None:
    if when == "delete":
        harness.controller.create()
        harness.delete_error = ProcessSupervisionLost("local supervision loss")
    elif when == "before-create":
        harness.create_error = ProcessSupervisionLost("local supervision loss")
    else:

        def lose(args: tuple[str, ...]) -> None:
            if args[:2] == ("get", "pod") and harness.pod is not None:
                raise ProcessSupervisionLost("local supervision loss")

        harness.before = lose
    with pytest.raises(ProcessSupervisionLost):
        if when == "delete":
            harness.controller.cleanup()
        else:
            harness.controller.create()
    count = len(harness.calls)
    assert harness.journal()["supervision_lost"], "supervision loss must be durable"
    with pytest.raises(RegionalFixtureError, match="supervision was lost"):
        harness.controller.cleanup()
    with pytest.raises(RegionalFixtureError, match="supervision was lost"):
        harness.new_holder()
    assert len(harness.calls) == count, "restarting must not clear supervision loss"


@pytest.mark.parametrize("custody", ["approved", "rejected", "unknown"])
def test_fresh_interpreters_only_resume_cleanup_never_create_again(
    harness: HolderHarness, tmp_path: Path, custody: str
) -> None:
    if custody == "approved":
        harness.controller.create()
    elif custody == "rejected":

        def reject(pod: dict[str, Any]) -> None:
            pod["metadata"]["labels"] = {}
            harness.pod = copy.deepcopy(pod)

        harness.ack_change = reject
        with pytest.raises(RegionalFixtureError, match="ownership or full spec"):
            harness.controller.create()
    else:
        harness.lost_create = True
        with pytest.raises(RegionalFixtureError, match="TimeoutError"):
            harness.controller.create()
    initial_phase = harness.journal()["phase"]
    cleanup_phase = "CREATING" if custody == "unknown" else "CLOSED"
    cleanup_error = "RegionalFixtureError" if custody == "unknown" else None
    cleanup_verbs = [] if custody == "unknown" else ["delete"]
    deadline = harness.journal()["deadline_at"]
    remote = tmp_path / "remote.json"
    write_json_atomic(remote, {"pod": harness.pod})
    program = """
import json, sys
from pathlib import Path
from pytest import MonkeyPatch
from gpu_fault.admin.atomic_json import write_json_atomic
from tests.regional._destr008_gpu_holder import build_holder
root, action = Path(sys.argv[1]), sys.argv[2]
with MonkeyPatch.context() as patch:
    harness = build_holder(root, patch)
    harness.pod = json.loads((root / "remote.json").read_text())["pod"]
    error = None
    try:
        getattr(harness.controller, action)()
    except Exception as exc:
        error = type(exc).__name__
    write_json_atomic(root / "remote.json", {"pod": harness.pod})
    print(json.dumps({"error": error, "mutations": harness.mutations(),
                      "phase": harness.journal()["phase"],
                      "approved": harness.journal()["pod_approved"],
                      "calls": len(harness.calls),
                      "deadline": harness.journal()["deadline_at"]}))
"""
    for action, error, phase, verbs in [
        ("create", "RegionalFixtureError", initial_phase, []),
        ("resume_cleanup", cleanup_error, cleanup_phase, cleanup_verbs),
        ("create", "RegionalFixtureError", cleanup_phase, []),
        ("resume_cleanup", cleanup_error, cleanup_phase, []),
    ]:
        child = subprocess.run(
            [sys.executable, "-c", program, str(tmp_path), action],
            cwd=ROOT,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
            },
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        result = json.loads(child.stdout)
        assert result["error"] == error, (
            "a fresh interpreter must retain create-once authority"
        )
        assert result["phase"] == phase, (
            "the same private journal governs reconstruction"
        )
        assert result["deadline"] == deadline, (
            "process restart must not rearm the lifetime"
        )
        assert result["approved"] == (custody == "approved"), (
            "a fresh process cannot turn deletion-only or unknown custody into approval"
        )
        assert [args[0] for args in result["mutations"]] == verbs, (
            "a fresh process may only delete an originally acknowledged Pod"
        )
        if custody == "unknown":
            assert result["calls"] == 0, (
                "unknown ACK must refuse before any adoption read"
            )
    assert (json.loads(remote.read_text())["pod"] is None) == (custody != "unknown"), (
        "only directly acknowledged custody may remove the Pod"
    )


def test_journal_and_api_methods_require_real_local_ownership(
    harness: HolderHarness,
) -> None:
    with pytest.raises(RegionalFixtureError, match="controller ownership"):
        harness.controller.save()
    with pytest.raises(RegionalFixtureError, match="controller ownership"):
        harness.controller.read()
    with locking.controller_ownership(harness.controller.path):
        with pytest.raises(RegionalFixtureError, match="creation deadline"):
            harness.controller.gpu("create")
    assert not harness.mutations(), "unowned or unbounded operations cannot mutate"


def test_extended_resource_toleration_admission_is_not_spec_drift() -> None:
    # Live EKS (2026-09-20, DESTR-008 a27 active-gpu-pod): the API server's
    # ExtendedResourceToleration plugin appended
    # ``{key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}`` to the
    # holder Pod, and the planned/observed digests diverged. The appended
    # toleration for a resource the Pod itself requests is admission, not a
    # foreign mutation; any other toleration still is.
    planned = {
        "nodeName": "spare-a",
        "tolerations": [{"operator": "Exists"}],
        "containers": [
            {
                "name": "holder",
                "image": "image@sha256:" + "a" * 64,
                "resources": {
                    "requests": {"nvidia.com/gpu": "1"},
                    "limits": {"nvidia.com/gpu": "1"},
                },
            }
        ],
    }
    observed = copy.deepcopy(planned)
    observed["tolerations"].append(
        {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
    )
    assert holder.digest(holder.normal_spec(observed)) == holder.digest(
        holder.normal_spec(planned)
    )
    foreign = copy.deepcopy(planned)
    foreign["tolerations"].append(
        {"key": "example.com/other", "operator": "Exists", "effect": "NoSchedule"}
    )
    assert holder.digest(holder.normal_spec(foreign)) != holder.digest(
        holder.normal_spec(planned)
    ), "a toleration for a resource the Pod never requested is still drift"
    unrequested = copy.deepcopy(planned)
    unrequested["containers"][0]["resources"] = {"limits": {"cpu": "1"}}
    drifted = copy.deepcopy(unrequested)
    drifted["tolerations"].append(
        {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}
    )
    assert holder.digest(holder.normal_spec(drifted)) != holder.digest(
        holder.normal_spec(unrequested)
    ), "the plugin only adds tolerations for requested resources"
