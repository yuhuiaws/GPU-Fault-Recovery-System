from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import host_probe_fixture as module
from tests.regional._host_probe_support import ProbeApi, host_probe


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> ProbeApi:
    value = ProbeApi()
    monkeypatch.setattr(module, "run_fixture_command", value.run)
    return value


def test_owned_probe_lifecycle_uses_private_uid_preconditions(
    tmp_path: Path, api: ProbeApi
) -> None:
    probe = host_probe(tmp_path)
    probe.create()
    assert probe.execute("snapshot") == {"ok": True}
    assert not any(probe.cleanup().values()), (
        f"owned cleanup must clear all residuals; resources={sorted(api.objects)}, "
        f"host_script_exists={api.host_script_exists}"
    )
    receipt = json.loads(probe.state_path.read_text())
    assert receipt["closed"] is True
    assert probe.state_path.stat().st_mode & 0o077 == 0
    deletes = [kwargs for args, kwargs in api.calls if args[0] == "delete"]
    assert len(deletes) == 2
    assert all(
        json.loads(item["input_text"])["preconditions"]["uid"] for item in deletes
    ), "both Pod and ConfigMap deletions must carry an owned UID precondition"
    assert all("apply" != args[0] for args, _ in api.calls), (
        "probe ownership must use create, never apply; "
        f"verbs={[a[0] for a, _ in api.calls]}"
    )


@pytest.mark.parametrize("operation", ["create", "cleanup"])
def test_foreign_existing_resources_are_never_deleted(
    tmp_path: Path, api: ProbeApi, operation: str
) -> None:
    probe = host_probe(tmp_path)
    api.objects["pod"] = {"metadata": {"name": probe.pod, "uid": "foreign"}}
    with pytest.raises(module.HostProbeError, match="ownership"):
        getattr(probe, operation)()
    assert not any(args[0] in {"delete", "create", "exec"} for args, _ in api.calls), (
        f"{operation} must reject foreign resources before mutation; "
        f"verbs={[a[0] for a, _ in api.calls]}"
    )


@pytest.mark.parametrize("replacement", ["uid", "node", "script", "owner"])
def test_execution_refuses_changed_target(
    tmp_path: Path, api: ProbeApi, replacement: str
) -> None:
    probe = host_probe(tmp_path)
    probe.create()
    if replacement == "uid":
        api.objects["pod"]["metadata"]["uid"] = "replacement"
    elif replacement == "node":
        api.node_uid = "replacement-node"
    elif replacement == "script":
        api.objects["configmap"]["data"] = {"probe.py": "foreign"}
    else:
        api.objects["pod"]["metadata"]["annotations"]["gpu-fault.io/probe-owner"] = (
            "foreign"
        )
    with pytest.raises(module.HostProbeError):
        probe.execute("snapshot")
    assert not any(args[0] == "exec" for args, _ in api.calls), (
        f"{replacement} drift must block host execution; "
        f"verbs={[a[0] for a, _ in api.calls]}"
    )


def test_cleanup_refuses_same_name_replacement(tmp_path: Path, api: ProbeApi) -> None:
    probe = host_probe(tmp_path)
    probe.create()
    api.objects["pod"]["metadata"]["uid"] = "replacement"
    with pytest.raises(module.HostProbeError, match="UID"):
        probe.cleanup()
    assert not any(args[0] == "delete" for args, _ in api.calls), (
        "cleanup must not delete a same-name Pod whose UID differs from the receipt"
    )


@pytest.mark.parametrize(
    "stdout", ["", "{}", "[]", "not-json", '{"error": "private-response"}']
)
def test_empty_or_failed_probe_output_is_not_success(
    tmp_path: Path, api: ProbeApi, stdout: str
) -> None:
    probe = host_probe(tmp_path)
    probe.create()
    api.probe_stdout = stdout
    api.probe_stderr = "private-diagnostic"
    with pytest.raises(module.HostProbeError) as error:
        probe.execute("snapshot")
    assert "private-response" not in str(error.value)
    assert "private-diagnostic" not in str(error.value)
    assert json.loads(probe.state_path.read_text())["script_may_exist"] is True
    assert not any(probe.cleanup().values()), (
        "rejected probe output must still permit cleanup of every owned residual"
    )


def test_lost_create_ack_is_recovered_for_cleanup(
    tmp_path: Path, api: ProbeApi
) -> None:
    probe = host_probe(tmp_path)
    api.lost_ack = "pod"
    with pytest.raises(module.HostProbeError):
        probe.create()
    assert json.loads(probe.state_path.read_text())["resources"]["pod"]["uid"]
    assert not any(probe.cleanup().values()), (
        "the recovered Pod UID must allow complete cleanup after a lost create ACK"
    )


def test_unresolved_creation_does_not_claim_clean(
    tmp_path: Path, api: ProbeApi
) -> None:
    probe = host_probe(tmp_path)
    api.reject_create = True
    with pytest.raises(module.HostProbeError):
        probe.create()
    with pytest.raises(module.HostProbeError, match="unresolved"):
        probe.cleanup()
    assert json.loads(probe.state_path.read_text())["closed"] is False


def test_cleanup_failure_keeps_channel_and_is_resumable(
    tmp_path: Path, api: ProbeApi
) -> None:
    probe = host_probe(tmp_path)
    probe.create()
    probe.execute("snapshot")
    api.cleanup_error = True
    with pytest.raises(module.HostProbeError):
        probe.cleanup()
    assert set(api.objects) == {"pod", "configmap"}
    resumed = host_probe(tmp_path)
    api.cleanup_error = False
    assert not any(resumed.cleanup().values()), (
        "a new fixture must finish cleanup using the retained ownership receipt"
    )
    assert json.loads(resumed.state_path.read_text())["closed"] is True


def test_execute_timeout_keeps_cleanup_receipt_without_replaying_probe(
    tmp_path: Path, api: ProbeApi
) -> None:
    probe = host_probe(tmp_path)
    probe.create()

    def timeout_after_execution() -> None:
        raise module.RegionalCommandTimeout(["unit-host-probe"], 1)

    api.after_execute = timeout_after_execution
    with pytest.raises(module.HostProbeError, match="timed out"):
        probe.execute("snapshot", timeout=1)
    receipt = json.loads(probe.state_path.read_text())
    assert receipt["script_may_exist"] is True
    assert receipt["closed"] is False
    assert api.host_script_exists, (
        "the timeout must occur after installation to exercise an ambiguous host action"
    )
    assert set(api.objects) == {"pod", "configmap"}
    assert sum(args[0] == "exec" for args, _ in api.calls) == 1, (
        "an ambiguous probe timeout must not replay the host command"
    )

    api.after_execute = None
    assert not any(probe.cleanup().values()), (
        "a timed-out execution must retain enough ownership proof for complete cleanup"
    )
    assert not api.host_script_exists, (
        "cleanup after the ambiguous timeout must remove the installed host script"
    )


def test_residual_audit_loads_pending_host_script_after_restart(
    tmp_path: Path, api: ProbeApi
) -> None:
    original = host_probe(tmp_path)
    original.create()
    original.execute("snapshot")
    api.cleanup_error = True
    with pytest.raises(module.HostProbeError):
        original.cleanup()
    api.objects.clear()

    resumed = module.HostProbeFixture(original.settings)
    residuals = resumed.residuals()
    assert residuals.get("host_script") is True, (
        "missing API objects do not prove that the pending host script is gone"
    )
    assert residuals[f"pod/{original.pod}"] is False
    assert residuals[f"configmap/{original.configmap}"] is False
    assert api.host_script_exists, (
        "the host script must outlive API resource loss in this restart-audit scenario"
    )

    api.cleanup_error = False
    retry = module.HostProbeFixture(original.settings)
    assert not any(retry.cleanup().values()), (
        "a standalone residual audit must release its temporary ownership lock"
    )
    assert not api.host_script_exists, (
        "the reconstructed cleanup channel must remove the orphaned host script"
    )


def test_residual_audit_loads_unresolved_creation_after_restart(
    tmp_path: Path, api: ProbeApi
) -> None:
    original = host_probe(tmp_path)
    api.reject_create = True
    with pytest.raises(module.HostProbeError):
        original.create()
    with pytest.raises(module.HostProbeError, match="unresolved"):
        original.cleanup()

    resumed = module.HostProbeFixture(original.settings)
    assert resumed.residuals().get("creation_unresolved") is True, (
        "a new fixture must not erase an unacknowledged creation intent"
    )


def test_residual_audit_keeps_an_active_lifecycle_lock(
    tmp_path: Path, api: ProbeApi
) -> None:
    original = host_probe(tmp_path)
    original.create()
    assert any(original.residuals().values()), (
        "an active fixture with an owned Pod and ConfigMap must not audit as clean"
    )
    other = module.HostProbeFixture(original.settings)
    with pytest.raises(module.HostProbeError, match="another process"):
        other.create()
    assert not any(original.cleanup().values()), (
        "the original lock owner must remain able to complete its cleanup"
    )


def test_failed_residual_audit_releases_its_temporary_lock(
    tmp_path: Path, api: ProbeApi
) -> None:
    original = host_probe(tmp_path)
    original.create()
    original.execute("snapshot")
    api.cleanup_error = True
    with pytest.raises(module.HostProbeError):
        original.cleanup()

    api.read_error = True
    resumed = module.HostProbeFixture(original.settings)
    with pytest.raises(module.HostProbeError):
        resumed.residuals()
    api.read_error = False
    api.cleanup_error = False
    retry = module.HostProbeFixture(original.settings)
    assert not any(retry.cleanup().values()), (
        "a failed standalone audit must not prevent a later cleanup owner"
    )
    assert not api.host_script_exists, (
        "cleanup following an API-read failure must remove the pending host script"
    )


@pytest.mark.parametrize("operation", ["execute", "cleanup"])
def test_probe_commands_ignore_a_sidecar_default_container(
    tmp_path: Path, api: ProbeApi, operation: str
) -> None:
    api.admit_sidecar = True
    probe = host_probe(tmp_path)
    probe.create()
    if operation == "cleanup":
        probe.execute("snapshot")
        assert api.host_script_exists, (
            "the sidecar cleanup test must start with a script installed on the host"
        )
    api.objects["pod"]["metadata"]["annotations"][
        "kubectl.kubernetes.io/default-container"
    ] = "sidecar"
    if operation == "execute":
        assert probe.execute("snapshot") == {"ok": True}
    assert not any(probe.cleanup().values()), (
        f"sidecar defaults must not prevent owned cleanup; "
        f"exec_containers={api.exec_containers}"
    )
    assert not api.host_script_exists, (
        "successful cleanup in an unmounted sidecar must not close the host receipt"
    )
    assert api.exec_containers == ["probe", "probe"]


def test_changed_script_cannot_hide_a_previous_cleanup_failure(
    tmp_path: Path, api: ProbeApi
) -> None:
    original = host_probe(tmp_path)
    original.create()
    original.execute("snapshot")
    api.cleanup_error = True
    with pytest.raises(module.HostProbeError):
        original.cleanup()
    original.settings.probe_script.write_text("print('{\"different\": true}')\n")
    changed = module.HostProbeFixture(original.settings)
    assert changed.state_path == original.state_path, (
        "script changes must still find the pending receipt"
    )
    api.calls.clear()
    with pytest.raises(module.HostProbeError, match="ownership record"):
        changed.create()
    assert not api.calls, (
        "a new script must not orphan the prior resources and create a second probe"
    )


def test_cleanup_replaces_owned_terminal_pod_on_same_node(
    tmp_path: Path, api: ProbeApi
) -> None:
    probe = host_probe(tmp_path)
    probe.create()
    probe.execute("snapshot")
    api.objects["pod"]["status"]["phase"] = "Failed"
    old_uid = api.objects["pod"]["metadata"]["uid"]
    assert not any(probe.cleanup().values()), (
        "cleanup must replace the owned terminal Pod and clear every residual"
    )
    created = [
        json.loads(kwargs["input_text"])
        for args, kwargs in api.calls
        if args[0] == "create"
    ]
    assert sum(item["kind"] == "Pod" for item in created) == 2
    deleted = [
        json.loads(kwargs["input_text"])
        for args, kwargs in api.calls
        if args[0] == "delete"
    ]
    assert deleted[0]["preconditions"]["uid"] == old_uid


def test_target_replacement_during_execution_invalidates_result(
    tmp_path: Path, api: ProbeApi
) -> None:
    probe = host_probe(tmp_path)
    probe.create()
    api.after_execute = lambda: api.objects["pod"]["metadata"].update(uid="replacement")
    with pytest.raises(module.HostProbeError, match="UID"):
        probe.execute("snapshot")


def test_competing_process_receipt_is_refused(tmp_path: Path, api: ProbeApi) -> None:
    first = host_probe(tmp_path)
    first.create()
    second = host_probe(tmp_path)
    with pytest.raises(module.HostProbeError, match="another process"):
        second.create()
    first.cleanup()
    second.create()
    assert not any(second.cleanup().values()), (
        "a successor owner must complete cleanup after the first owner "
        "releases its lock"
    )


def test_different_contexts_and_scripts_have_distinct_probe_names(
    tmp_path: Path, api: ProbeApi
) -> None:
    first = host_probe(tmp_path)
    second = host_probe(tmp_path, context="gpu-context-b")
    script = tmp_path / "another probe.py"
    script.write_text("print('{\"other\": true}')\n", encoding="utf-8")
    third = host_probe(tmp_path, probe_script=script)
    assert len({first.pod, second.pod, third.pod}) == 3
    third.create()
    third.execute("snapshot")
    call = next(args for args, _ in api.calls if args[0] == "exec")
    assert "/probe/probe.py" in call
    assert set(api.objects["configmap"]["data"]) == {"probe.py"}, (
        "ConfigMap keys must remain Kubernetes-compatible"
    )
    assert all(
        "another probe.py" not in item for item in call if "install -m" in item
    ), (
        "a probe filename containing spaces must not be interpolated "
        "into the shell program"
    )
    third.cleanup()


@pytest.mark.parametrize(
    "change",
    [lambda v: v.update(closed="false"), lambda v: v["resources"]["pod"].update(uid=7)],
)
def test_malformed_persisted_receipt_never_authorizes_mutation(
    tmp_path: Path, api: ProbeApi, change: Any
) -> None:
    probe = host_probe(tmp_path)
    probe.create()
    probe.cleanup()
    receipt = json.loads(probe.state_path.read_text())
    change(receipt)
    probe.state_path.write_text(json.dumps(receipt), encoding="utf-8")
    api.calls.clear()
    with pytest.raises(module.HostProbeError):
        host_probe(tmp_path).create()
    assert not api.calls, (
        f"a malformed receipt must be rejected before any API access; "
        f"verbs={[a[0] for a, _ in api.calls]}"
    )
