from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import host_probe_fixture as module
from tests.regional import test_cov95_common_host_receipts as receipts

owned = receipts.owned


@pytest.mark.parametrize(
    "document", [[], {"metadata": None}, {"metadata": {"name": "foreign"}}]
)
def test_malformed_or_wrong_name_reads_cannot_prove_absence(owned, document) -> None:
    api, probe = owned
    api.objects["pod"] = document
    with pytest.raises(module.HostProbeError, match="malformed|name differs"):
        probe.residuals()
    api.objects.clear()
    assert not any(probe.residuals().values()), (
        "only a subsequent clean read may prove absence"
    )


def test_malformed_create_ack_retains_discoverable_owned_resources(
    owned, monkeypatch
) -> None:
    api, probe = owned
    transport = api.run

    def malformed(command, **kwargs):
        result = transport(command, **kwargs)
        if command[7] == "create":
            return subprocess.CompletedProcess(command, 0, "[]", "")
        return result

    monkeypatch.setattr(module, "run_fixture_command", malformed)
    with pytest.raises(module.HostProbeError, match="create response is malformed"):
        probe.create()
    assert set(api.objects) == {"configmap"}, (
        "the failed acknowledgement may follow a committed create"
    )
    assert not any(probe.cleanup().values()), (
        "owned readback must make the partial create cleanable"
    )


@pytest.mark.parametrize("operation", ["create", "cleanup"])
def test_resource_without_creation_receipt_cannot_be_reused_or_deleted(
    owned, operation
) -> None:
    api, original = owned
    original.create()
    original.execute("snapshot")
    api.cleanup_error = True
    with pytest.raises(
        module.HostProbeError,
        match=r"command failed \(1\): <sensitive output redacted>",
    ):
        original.cleanup()
    receipt = json.loads(original.state_path.read_text())
    receipt["resources"]["pod"]["create_started"] = False
    original.state_path.write_text(json.dumps(receipt))
    resumed = module.HostProbeFixture(original.settings)
    before = len(api.calls)
    try:
        with pytest.raises(
            module.HostProbeError, match="not created|no creation receipt"
        ):
            getattr(resumed, operation)()
    finally:
        with pytest.raises(module.HostProbeError, match="no creation receipt"):
            resumed.cleanup()
    assert all(args[0] == "get" for args, _ in api.calls[before:]), (
        "unproved resource creation cannot authorize reuse, execution or deletion"
    )


def test_pod_loss_between_ready_ack_and_readback_invalidates_create(
    owned, monkeypatch
) -> None:
    api, probe = owned
    transport = api.run

    def vanish(command, **kwargs):
        result = transport(command, **kwargs)
        if command[7] == "wait":
            api.objects.pop("pod", None)
        return result

    monkeypatch.setattr(module, "run_fixture_command", vanish)
    with pytest.raises(module.HostProbeError, match="disappeared during creation"):
        probe.create()
    assert not any(probe.cleanup().values()), (
        "remaining owned resources must still be removable"
    )


def fake_clock(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        module,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock.now,
            sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
        ),
    )
    return clock


def test_terminal_pod_must_disappear_before_recreation(owned, monkeypatch) -> None:
    api, probe = owned
    probe.create()
    api.objects["pod"]["status"]["phase"] = "Failed"
    api.stuck_pod = True
    fake_clock(monkeypatch)
    with pytest.raises(
        module.HostProbeError, match="terminal host probe Pod did not disappear"
    ):
        probe.create()
    assert sum(args[0] == "create" for args, _ in api.calls) == 2, (
        "a replacement must not be created while the old owned Pod is still present"
    )


def test_failed_force_delete_keeps_cleanup_open_until_readback_confirms_absence(
    owned, monkeypatch
) -> None:
    api, probe = owned
    probe.create()
    transport = api.run
    deletes = []
    fake_clock(monkeypatch)

    def ignore_pod_delete(command, **kwargs):
        if command[7] == "delete" and "/pods/" in command[9]:
            deletes.append(json.loads(kwargs["input_text"]))
            return subprocess.CompletedProcess(command, 0, "", "")
        return transport(command, **kwargs)

    monkeypatch.setattr(module, "run_fixture_command", ignore_pod_delete)
    with pytest.raises(module.HostProbeError, match="cleanup is incomplete"):
        probe.cleanup()
    assert len(deletes) == 2 and deletes[-1]["gracePeriodSeconds"] == 0, (
        "cleanup may attempt only its bounded normal and forced deletion paths"
    )
    assert json.loads(probe.state_path.read_text())["closed"] is False, (
        "even a forced-delete acknowledgement is not proof of absence"
    )
    monkeypatch.setattr(module, "run_fixture_command", transport)
    assert not any(probe.cleanup().values()), (
        "retained ownership must allow a later confirmed cleanup"
    )
