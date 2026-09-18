"""Crash/resumption and private identity boundaries of the CPU lifecycle."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.atomic_json import write_json_atomic
from scripts.e2e.regional import destr008_cancellation as lifecycle
from scripts.e2e.regional import destr008_controller_lock as locking
from scripts.e2e.regional import destr008_watchdog_resources as resources
from scripts.e2e.regional.probes import destr008_cancellation_protocol as wire
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from scripts.e2e.regional.seeded_command_fixture import SeededCommandError
from tests.regional._destr008_cancellation_controller import CpuApi, build_api


@pytest.fixture
def setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[CpuApi, wire.Plan, resources.CpuRuntime]:
    return build_api(tmp_path, monkeypatch)


def test_private_journal_creation_and_empty_directory_existence(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    api.directory.mkdir(mode=0o700)
    assert not lifecycle.has_saved_plan(api.directory, plan.run_id), (
        "an empty private directory has no plan"
    )
    with pytest.raises(RegionalFixtureError, match="unavailable"):
        lifecycle.load_saved_plan(api.directory, plan.run_id)
    watchdog = api.watchdog(plan, runtime)
    assert watchdog.path.stat().st_mode & 0o777 == 0o600, "the journal must be private"
    assert api.directory.stat().st_mode & 0o777 == 0o700, (
        "the journal directory must be private"
    )
    assert not api.calls, (
        "construction and saved-plan helpers must not contact Kubernetes"
    )
    with pytest.raises(RegionalFixtureError, match="control"):
        watchdog.control_client()
    with pytest.raises(RegionalFixtureError, match="never completed"):
        watchdog.validate_running()
    watchdog.cleanup()
    assert watchdog.record.closed and watchdog.record.quiescence is None, (
        "an empty resource lifecycle can retire without fabricated Store evidence"
    )


@pytest.mark.parametrize("mode", ["symlink", "hardlink", "public", "directory", "fifo"])
def test_existing_journal_must_be_private_regular_and_not_linked(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], mode: str
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    path = watchdog.path
    if mode == "public":
        path.chmod(0o644)
    elif mode == "hardlink":
        os.link(path, path.with_suffix(".linked"))
    else:
        data = path.read_bytes()
        path.unlink()
        if mode == "symlink":
            target = path.with_suffix(".target")
            target.write_bytes(data)
            target.chmod(0o600)
            path.symlink_to(target)
        elif mode == "directory":
            path.mkdir()
        else:
            os.mkfifo(path, 0o600)
    with pytest.raises((RegionalFixtureError, OSError)):
        lifecycle.has_saved_plan(api.directory, plan.run_id)
    assert not api.calls, "invalid local identity must fail before any API request"


@pytest.mark.parametrize(
    "contents",
    ["[]", "null", "not-json", '{"schema_version":1,"schema_version":1}', "x" * 262145],
)
def test_corrupt_or_oversized_journal_is_not_absence(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], contents: str
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.path.write_text(contents)
    with pytest.raises(RegionalFixtureError):
        lifecycle.has_saved_plan(api.directory, plan.run_id)
    with pytest.raises(RegionalFixtureError):
        api.watchdog(plan, runtime)
    assert not api.calls, "journal parse failures must not reach Kubernetes"


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"compatibility": "unsupported"},
        {"runtime": {}},
        {"closed": True},
        {"ever_armed": True},
        {"host_sha256": "a" * 64},
        {"configuration": {}},
        {"source_data": {"other.py": "not-authorized"}},
        {"sources": {"foreign": "a" * 64}},
        {
            "support": {
                "role/foreign": {"kind": "role", "name": "foreign", "uid": "uid-a"}
            }
        },
    ],
)
def test_saved_binding_corruption_is_refused_without_remote_adoption(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], change: dict[str, Any]
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    saved = api.journal()
    saved.update(change)
    write_json_atomic(watchdog.path, saved)
    with pytest.raises(RegionalFixtureError):
        api.watchdog(plan, runtime)
    assert not api.calls, (
        "altered local pins must not authorize remote lookup or adoption"
    )


def test_saved_plan_run_binding_and_runtime_shape_are_validated(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    saved = api.journal()
    saved["plan"]["run_id"] = "foreign"
    write_json_atomic(watchdog.path, saved)
    with pytest.raises(RegionalFixtureError, match="another run"):
        lifecycle.has_saved_plan(api.directory, plan.run_id)
    with pytest.raises(RegionalFixtureError, match="binding"):
        lifecycle.load_saved_plan(api.directory, plan.run_id)
    saved["plan"]["run_id"] = plan.run_id
    saved["runtime"] = {}
    write_json_atomic(watchdog.path, saved)
    with pytest.raises(RegionalFixtureError, match="runtime"):
        lifecycle.load_saved_plan(api.directory, plan.run_id)


@pytest.mark.parametrize("run_id", ["", None, False])
def test_saved_plan_helpers_require_a_real_run_id(tmp_path: Path, run_id: Any) -> None:
    with pytest.raises(RegionalFixtureError, match="run identity"):
        lifecycle.has_saved_plan(tmp_path, run_id)


@pytest.mark.parametrize(
    "identity", ["namespace", "cluster", "configuration", "host", "runtime", "release"]
)
def test_execution_and_cleanup_recheck_bound_identity(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    monkeypatch: pytest.MonkeyPatch,
    identity: str,
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    if identity == "namespace":
        api.objects["namespace", runtime.namespace]["metadata"]["uid"] = "replacement"
    elif identity == "cluster":
        api.regional.settings = replace(api.regional.settings, cluster_id="foreign")
    elif identity == "configuration":
        api.regional.settings.cpu_kubeconfig.write_text("changed inert configuration\n")
    elif identity == "host":
        monkeypatch.setattr(locking, "host_identity", lambda: "a" * 64)
    elif identity == "runtime":
        api.objects["deployment", resources.DEPLOYMENT]["metadata"]["generation"] += 1
    else:
        monkeypatch.setattr(
            api.regional,
            "evidence_identity",
            lambda: {"cluster_id": "cluster-a", "release_id": "foreign"},
        )
    with pytest.raises(RegionalFixtureError):
        watchdog.arm()
    with pytest.raises(RegionalFixtureError):
        watchdog.cleanup()
    assert not [call for call in api.calls if call[0] != "get"], (
        "namespace/configuration/runtime/release drift must fail before mutations"
    )


def test_controller_only_source_change_can_resume_cleanup_but_never_arm(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, plan, runtime = setup
    source = tmp_path / "controller-source"
    source.write_text("first controlled source identity\n")
    monkeypatch.setattr(lifecycle, "__file__", str(source))
    original = api.watchdog(plan, runtime)
    original.arm()
    source.write_text("compatible cleanup-only source identity\n")
    with pytest.raises(RegionalFixtureError, match="source changed"):
        original.validate_running()
    saved_plan, saved_runtime = lifecycle.load_saved_plan(api.directory, plan.run_id)
    fresh = api.watchdog(saved_plan, saved_runtime)
    with pytest.raises(RegionalFixtureError, match="source changed"):
        fresh.arm()
    proof = fresh.resume_cleanup(seconds=10)
    assert proof.case_failed and proof.probe_sha256 == plan.probe_sha256, (
        "cleanup source updates must retain the original probe plan and original failure"
    )
    with pytest.raises(RegionalFixtureError):
        fresh.control_client().claim()
    fresh.cleanup()


def test_probe_source_change_is_not_silently_reclassified_as_compatible(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    (api.source / wire.SOURCE_FILES[0]).write_text("different probe bytes\n")
    fresh = api.watchdog(plan, runtime)
    with pytest.raises(RegionalFixtureError, match="source"):
        fresh.arm()
    with pytest.raises(RegionalFixtureError, match="source"):
        fresh.resume_cleanup(seconds=10)
    assert not api.journal()["closed"], (
        "incompatible probe changes require preserving the original record"
    )


CHILD = """
import json,sys
from pathlib import Path
import pytest
from scripts.e2e.regional.destr008_cancellation import load_saved_plan
from tests.regional._destr008_cancellation_controller import load_api
with pytest.MonkeyPatch.context() as patch:
    api=load_api(Path(sys.argv[1]),patch)
    plan,runtime=load_saved_plan(api.directory,sys.argv[2])
    watchdog=api.watchdog(plan,runtime)
    proof=watchdog.resume_cleanup(seconds=15)
    control=watchdog.control_client()
    control.quiescence()
    watchdog.cleanup()
    print(json.dumps({"state":proof.state,"case_failed":proof.case_failed,
                      "closed":watchdog.record.closed,"control_uid":control.uid}))
"""


def test_fresh_interpreter_resumes_only_cleanup_from_private_state(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], tmp_path: Path
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    snapshot = tmp_path / "fake-api.json"
    api.dump(snapshot)
    result = subprocess.run(
        [sys.executable, "-B", "-c", CHILD, str(snapshot), plan.run_id],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(result.stdout) == {
        "state": "QUIESCENT",
        "case_failed": True,
        "closed": True,
        "control_uid": control.uid,
    }, "a fresh process must use the persisted identities and explicit cleanup observer"
    assert lifecycle.has_saved_plan(api.directory, plan.run_id), (
        "cleanup completion must preserve the private audit record"
    )


def test_local_controller_ownership_blocks_a_concurrent_parent(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    with locking.controller_ownership(watchdog.path):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(watchdog.arm)
            with pytest.raises(RegionalFixtureError, match="another controller"):
                future.result(timeout=5)
    assert not api.calls, (
        "a second parent cannot enter before acquiring controller ownership"
    )


@pytest.mark.parametrize("seconds", [False, 0, 4, 181, 10.0])
def test_cleanup_window_must_be_an_explicit_bounded_integer(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], seconds: Any
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    with pytest.raises(RegionalFixtureError, match="bound"):
        watchdog.resume_cleanup(seconds)
    assert not api.calls, "invalid cleanup windows cannot contact the API"


@pytest.mark.parametrize("change", ["producer", "revocation", "close"])
def test_fresh_controller_rejects_durable_control_regression(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], change: str
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    api.frozen_status = True
    configmap = api.objects["configmap", watchdog.name]
    original = copy.deepcopy(configmap["data"])
    if change == "producer":
        control.claim()
    elif change == "close":
        control.request_close()
    else:
        before = control.read()
        revoked = wire.revoke(
            before.control, now=int(api.clock.now()), reason="FAILURE"
        )
        receipt = wire.receipt(
            plan,
            revoked,
            before.receipt,
            uid=control.uid,
            now=int(api.clock.now()),
            state="REVOKED",
        )
        configmap["data"]["control.json"] = wire.encode(revoked)
        configmap["data"]["status.json"] = wire.encode(receipt)
    watchdog.control_client()
    configmap["data"] = original
    with pytest.raises(RegionalFixtureError, match="history regressed"):
        api.watchdog(plan, runtime).control_client()


@pytest.mark.parametrize(
    "change",
    ["missing", "same-sequence", "sequence", "failure", "workflows", "commands"],
)
def test_fresh_controller_keeps_receipt_failure_and_inventory_history(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], change: str
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    api.frozen_status = True
    before = control.read()
    assert before.receipt is not None, (
        "the fixture must have a real parsed ARMED receipt"
    )
    configmap = api.objects["configmap", watchdog.name]
    previous = wire.receipt(
        plan,
        before.control,
        before.receipt,
        uid=control.uid,
        now=int(api.clock.now()),
        state="FAILED",
        error_code="FIXTURE_FAILURE",
        monitoring=False,
        workflow_ids=["remembered-workflow"],
        command_ids=["remembered-command"],
    )
    configmap["data"]["status.json"] = wire.encode(previous)
    watchdog.control_client()
    changed = previous.model_dump()
    changed["sequence"] += 1
    if change == "missing":
        configmap["data"]["status.json"] = "null"
    else:
        if change == "same-sequence":
            api.clock.sleep(1)
            changed.update(sequence=previous.sequence, observed_at=int(api.clock.now()))
        elif change == "sequence":
            changed["sequence"] = previous.sequence - 1
            changed["failure"]["sequence"] = changed["sequence"]
        elif change == "failure":
            changed["failure"]["code"] = "DIFFERENT_FAILURE"
        elif change == "workflows":
            changed["workflow_ids"] = []
        else:
            changed["command_ids"] = []
        configmap["data"]["status.json"] = wire.encode(
            wire.Receipt.model_validate(changed)
        )
    with pytest.raises(RegionalFixtureError, match="history regressed"):
        api.watchdog(plan, runtime).control_client()


def test_mutable_control_is_validated_by_protocol_and_never_echoed(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    api.frozen_status = True
    api.objects["configmap", watchdog.name]["data"]["control.json"] = (
        "private-malformed-fixture"
    )
    with pytest.raises(RegionalFixtureError) as caught:
        watchdog.control_client()
    assert "private-malformed" not in str(caught.value), (
        "control errors must not leak their input"
    )


def test_delete_uses_full_read_resource_version_and_refuses_intervening_spec_change(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    control.request_close()
    watchdog.wait_quiescence()
    original = api.kube

    def raced(plane: str, *args: str, **kwargs: Any) -> str:
        if (
            args[:3] == ("get", "job", watchdog.name)
            and args[-1] == "jsonpath={.metadata}"
        ):
            item = api.objects["job", watchdog.name]
            item["metadata"]["resourceVersion"] = api.next()
            item["spec"]["backoffLimit"] = 1
        return original(plane, *args, **kwargs)

    monkeypatch.setattr(api.regional, "kubectl", raced)
    with pytest.raises(SeededCommandError, match="full cleanup check"):
        watchdog.cleanup()
    assert not [call for call in api.calls if call[0] == "delete"], (
        "an intervening spec/RV mutation must fail before DELETE"
    )
    with pytest.raises(RegionalFixtureError, match="changed"):
        api.watchdog(plan, runtime).cleanup()
    assert ("job", watchdog.name) in api.objects, (
        "later cleanup must not silently accept the changed Job"
    )


def test_removal_of_unknown_metadata_cannot_hide_a_recreated_resource(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    control.request_close()
    watchdog.wait_quiescence()
    saved = copy.deepcopy(api.objects["role", watchdog.name])
    watchdog.cleanup()
    saved["metadata"]["uid"] = "new-role"
    api.objects["role", watchdog.name] = saved
    with pytest.raises(RegionalFixtureError, match="recreated"):
        watchdog.cleanup()


def test_earlier_observer_must_stop_before_a_cleanup_job_can_be_created(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    api.failed_delete.add("job")
    before = len([call for call in api.calls if call[0] == "create"])
    with pytest.raises(RegionalFixtureError):
        watchdog.resume_cleanup(seconds=10)
    assert len([call for call in api.calls if call[0] == "create"]) == before, (
        "a new observer may not overlap an old or unconfirmed one"
    )
    assert api.journal()["cleanup_only"], (
        "cleanup intent must irreversibly remove local execution authority"
    )


def test_successful_arm_cannot_be_retired_without_terminal_quiescence(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    watchdog.arm()
    with pytest.raises(RegionalFixtureError, match="verified terminal"):
        watchdog.cleanup()
    assert ("configmap", watchdog.name) in api.objects, (
        "control state must survive until an independent terminal proof is available"
    )
    proof = watchdog.resume_cleanup(seconds=10)
    assert proof.case_failed, (
        "interrupted retirement must not restore the original case verdict"
    )
    watchdog.cleanup()


def test_quiescence_requires_close_and_a_live_unretired_lifecycle(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    with pytest.raises(RegionalFixtureError, match="precede quiescence"):
        watchdog.wait_quiescence()
    control.request_close()
    watchdog.wait_quiescence()
    with pytest.raises(RegionalFixtureError, match="rearm"):
        watchdog.arm()
    watchdog.cleanup()
    with pytest.raises(RegionalFixtureError, match="retired"):
        watchdog.wait_quiescence()
    with pytest.raises(RegionalFixtureError, match="retired"):
        watchdog.resume_cleanup(seconds=10)


def test_no_terminal_observer_proof_is_not_a_successful_retirement(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, plan, runtime = setup
    watchdog = api.watchdog(plan, runtime)
    control = watchdog.arm()
    api.auto_quiet = False
    control.request_close()
    monkeypatch.setattr(wire, "DRAIN_SECONDS", 0)
    with pytest.raises(RegionalFixtureError, match="deadline expired"):
        watchdog.wait_quiescence()
    assert api.journal()["quiescence"] is None, (
        "a timeout must not create quiescence evidence"
    )


@pytest.mark.parametrize("field", ["runtime-profile", "image", "namespace"])
def test_constructor_rejects_a_different_saved_target(
    setup: tuple[CpuApi, wire.Plan, resources.CpuRuntime], field: str
) -> None:
    api, plan, runtime = setup
    api.watchdog(plan, runtime)
    if field == "runtime-profile":
        plan = plan.model_copy(update={"runtime_profile_version": "different"})
    elif field == "image":
        runtime = replace(runtime, image="registry.example/other@sha256:" + "b" * 64)
    else:
        runtime = replace(runtime, namespace="different")
    with pytest.raises(RegionalFixtureError, match="binding|identity"):
        api.watchdog(plan, runtime)
    assert not api.calls, "saved targets must be checked before contacting an API"
