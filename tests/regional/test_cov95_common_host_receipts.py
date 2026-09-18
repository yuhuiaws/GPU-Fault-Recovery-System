from __future__ import annotations

import json
import os
from contextlib import suppress
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import host_probe_fixture as module
from tests.regional._host_probe_support import ProbeApi, host_probe


@pytest.fixture
def owned(tmp_path, monkeypatch):
    api = ProbeApi()
    monkeypatch.setattr(module, "run_fixture_command", api.run)
    probe = host_probe(tmp_path)
    yield api, probe
    with suppress(module.HostProbeError):
        probe.cleanup()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kubeconfig", "missing", "kubeconfig does not exist"),
        ("probe_script", "missing", "script does not exist"),
        ("context", "", "empty identity"),
        ("namespace", "", "empty identity"),
        ("node", "", "empty identity"),
        ("run_id", "", "empty identity"),
        ("active_deadline_seconds", 59, "outside 60..7200"),
        ("active_deadline_seconds", 7201, "outside 60..7200"),
    ],
)
def test_settings_fail_before_host_resources_are_constructed(
    owned, field, value, message, tmp_path
) -> None:
    api, probe = owned
    if value == "missing":
        value = tmp_path / "missing"
    with pytest.raises(ValueError, match=message):
        replace(probe.settings, **{field: value})
    assert api.calls == [], "invalid configuration must not reach the transport"


@pytest.mark.parametrize("kind", ["fifo-lock", "symlink-record", "public-record"])
def test_nonprivate_receipts_and_locks_cannot_authorize_a_probe(
    owned, kind, tmp_path
) -> None:
    api, probe = owned
    probe.settings.state_directory.mkdir()
    if kind == "fifo-lock":
        os.mkfifo(probe.state_path.with_suffix(".lock"))
    elif kind == "symlink-record":
        target = tmp_path / "foreign.json"
        target.write_text("{}")
        probe.state_path.symlink_to(target)
    else:
        probe.state_path.write_text("{}")
        probe.state_path.chmod(0o644)
    with pytest.raises(module.HostProbeError, match="invalid|not private"):
        probe.residuals()
    assert api.calls == [], "untrusted receipts must fail before any resource read"


@pytest.mark.parametrize(
    ("location", "value", "message"),
    [
        (("closed",), "false", "ownership record"),
        (("resources",), [], "ownership record"),
        (("resources", "pod"), None, "receipt is malformed"),
        (("resources", "pod", "name"), "foreign", "receipt is malformed"),
        (("resources", "pod", "create_started"), "true", "receipt is malformed"),
        (("resources", "pod", "uid"), "", "receipt is malformed"),
        (("resources", "configmap", "uid"), 1, "receipt is malformed"),
    ],
)
def test_resumed_owner_requires_well_typed_resource_receipts(
    owned, location, value, message
) -> None:
    api, probe = owned
    probe.create()
    probe.cleanup()
    receipt = json.loads(probe.state_path.read_text())
    target = receipt
    for part in location[:-1]:
        target = target[part]
    target[location[-1]] = value
    probe.state_path.write_text(json.dumps(receipt))
    api.calls.clear()
    resumed = module.HostProbeFixture(probe.settings)
    with pytest.raises(module.HostProbeError, match=message):
        resumed.create()
    assert api.calls == [], "a malformed persisted receipt cannot start a lifecycle"


@pytest.mark.parametrize("missing", [False, True])
def test_unavailable_node_uid_prevents_creation(owned, missing) -> None:
    api, probe = owned
    api.node_uid = None if missing else ""
    with pytest.raises(module.HostProbeError, match="node identity is unavailable"):
        probe.create()
    assert all(args[0] == "get" for args, _ in api.calls), (
        "unknown node identity must not create a Pod"
    )


@pytest.mark.parametrize(
    ("kind", "field", "value", "message"),
    [
        ("pod", "node", "replacement", "target node"),
        ("pod", "image", "replacement", "container image"),
        ("pod", "containers", [], "container image"),
        ("configmap", "data", {}, "ConfigMap script"),
    ],
)
def test_execution_checks_all_owned_resource_content(
    owned, kind, field, value, message
) -> None:
    api, probe = owned
    probe.create()
    target = api.objects[kind]
    if field == "node":
        target["spec"]["nodeName"] = value
    elif field == "image":
        target["spec"]["containers"][0]["image"] = value
    elif field == "containers":
        target["spec"]["containers"] = value
    else:
        target[field] = value
    with pytest.raises(module.HostProbeError, match=message):
        probe.execute("snapshot")
    assert all(args[0] != "exec" for args, _ in api.calls), (
        "resource content drift must be refused before the host action"
    )


def test_execute_without_active_receipt_is_refused(owned) -> None:
    api, probe = owned
    with pytest.raises(module.HostProbeError, match="no active ownership"):
        probe.execute("snapshot")
    assert api.calls == [], "execution cannot allocate ownership implicitly"


def test_existing_receipt_does_not_hide_node_replacement_on_create(owned) -> None:
    api, probe = owned
    probe.create()
    api.node_uid = "replacement"
    before = len(api.calls)
    with pytest.raises(module.HostProbeError, match="node UID changed"):
        probe.create()
    assert len(api.calls) == before + 1, "recreate must recheck the node before writes"


def test_missing_resource_blocks_the_execution_channel(owned) -> None:
    api, probe = owned
    probe.create()
    del api.objects["configmap"]
    with pytest.raises(module.HostProbeError, match="resource disappeared"):
        probe.execute("snapshot")
    assert all(args[0] != "exec" for args, _ in api.calls), (
        "a vanished ConfigMap must not be treated as intact execution evidence"
    )


def test_creation_rechecks_pod_and_node_after_readiness(owned, monkeypatch) -> None:
    api, probe = owned
    transport = api.run

    def ready_then_replace(command, **kwargs):
        result = transport(command, **kwargs)
        if command[7] == "wait":
            api.node_uid = "replacement-after-ready"
        return result

    monkeypatch.setattr(module, "run_fixture_command", ready_then_replace)
    with pytest.raises(module.HostProbeError, match="UID changed during creation"):
        probe.create()
    receipt = json.loads(probe.state_path.read_text())
    assert receipt["closed"] is False, "ready on a replaced node is not completion"


def test_unconfirmed_deletions_keep_a_resumable_open_receipt(
    owned, monkeypatch
) -> None:
    api, probe = owned
    probe.create()
    transport = api.run
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        module,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock.now,
            sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
        ),
    )

    def leave_configmap(command, **kwargs):
        if command[7] == "delete" and "/configmaps/" in command[9]:
            api.calls.append((command[7:], kwargs))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return transport(command, **kwargs)

    monkeypatch.setattr(module, "run_fixture_command", leave_configmap)
    with pytest.raises(module.HostProbeError, match="residual resources"):
        probe.cleanup()
    receipt = json.loads(probe.state_path.read_text())
    assert receipt["closed"] is False, "a deletion ACK does not prove absence"
    assert set(api.objects) == {"configmap"}, "independent Pod cleanup must still run"
    monkeypatch.setattr(module, "run_fixture_command", transport)
    assert not any(probe.cleanup().values()), (
        "a valid retained receipt must permit retry"
    )
