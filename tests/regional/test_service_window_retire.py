"""The controller half of retiring a stale service window, end to end over fakes.

A real controller journal is left the way the live 2026-10-27 run left it
(stopped, restored by the on-node unit at ``restore_at``, the undo's ``cleanup``
without a response: CLEANUP_REQUIRED, host-probe record open, the probe's
``/run`` script copy and its Pod/ConfigMap still there). ``retire_journal`` then
runs the retire probe in-process against the fake systemd host and the earlier
probe's objects against an in-memory API: it retires only with every proof,
closes the record and the journal with a retirement record, and leaves
everything as it was -- with each reason named -- when a proof fails.
"""

from __future__ import annotations

import fcntl
import hashlib
import itertools
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote

import pytest

from scripts.e2e.regional import destr008_service_window as controller
from scripts.e2e.regional import service_window_retire as sw
from scripts.e2e.regional.host_probe_fixture import (
    HostProbeFixture,
    HostProbeMissingResponseError,
    HostProbeSettings,
)
from scripts.e2e.regional.probes import warm_spare_node_probe as probe
from scripts.e2e.regional.probes import warm_spare_retire as retire_host
from tests.regional._destr008_service_window import PINS, Host, Transport

POD = "gpu-fault-host-probe-0123456789"


class NamedTransport(Transport):
    """The service-window transport with a host-probe record named as live."""

    def __init__(self, host: Host, directory: Path) -> None:
        super().__init__(host, directory)
        self.pod = self.configmap = POD
        self.state_path = directory / f"{POD}.json"


class RetireProbe:
    """The retire probe's transport: the host part runs in-process."""

    instances: list[RetireProbe] = []

    def __init__(self, settings: HostProbeSettings) -> None:
        self.settings = settings
        self.calls: list[str] = []
        self.arguments: list[str] = []
        self.cleaned = False
        RetireProbe.instances.append(self)

    def create(self) -> None:
        self.calls.append("create")

    def execute(self, command: str, *arguments: str, timeout: int = 180) -> Any:
        self.calls.append(command)
        self.arguments = list(arguments)
        options = dict(zip(arguments[::2], arguments[1::2]))
        script = options.get("--stale-script", "")
        digest = options.get("--stale-script-sha256", "")
        if command == "retire":
            window = probe.ServiceWindow(json.loads(options["--binding"]))
            return retire_host.retire_window(window, script, digest)
        return retire_host.retire_script(script, digest)

    def cleanup(self) -> dict[str, bool]:
        self.cleaned = True
        return {f"pod/{POD}-retire": False, "host_script": False}


class Api:
    """The earlier probe's Pod and ConfigMap, as the API serves them."""

    def __init__(self, record: dict[str, Any]) -> None:
        self.deletes: list[dict[str, Any]] = []
        meta = {
            "labels": {
                "app": "gpu-fault-acceptance-host-probe",
                "gpu-fault.io/acceptance-case": record["scope"]["case_id"],
                "gpu-fault.io/acceptance-run": record["scope"]["run_id"],
            },
            "annotations": {
                "gpu-fault.io/probe-owner": record["owner"],
                "gpu-fault.io/probe-script-sha256": record["scope"]["script_sha256"],
                "gpu-fault.io/probe-node-uid": record["node_uid"],
            },
        }
        self.objects: dict[str, dict[str, Any]] = {
            kind: {
                "metadata": {
                    **json.loads(json.dumps(meta)),
                    "name": POD,
                    "uid": record["resources"][kind]["uid"],
                },
                **(
                    {"spec": {"nodeName": record["scope"]["node"]}}
                    if kind == "pod"
                    else {}
                ),
                "status": {"phase": "Failed"} if kind == "pod" else {},
            }
            for kind in ("pod", "configmap")
        }

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        verb = command[7]
        if verb == "get":
            value = self.objects.get(command[8])
            name = command[9]
            found = value is not None and value["metadata"]["name"] == name
            return subprocess.CompletedProcess(
                command, 0, json.dumps(value) if found else "", ""
            )
        assert verb == "delete" and command[8] == "--raw", command
        options = json.loads(kwargs["input_text"])
        kind = "pod" if "/pods/" in command[9] else "configmap"
        self.deletes.append(
            {**options, "object": kind, "name": unquote(command[9].rsplit("/", 1)[1])}
        )
        current = self.objects.get(kind)
        if current and current["metadata"]["uid"] == options["preconditions"]["uid"]:
            del self.objects[kind]
        return subprocess.CompletedProcess(command, 0, "", "")


@pytest.fixture
def stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The live shape: a window restored by its deadline, never closed."""

    monkeypatch.setenv("INVOCATION_ID", "a" * 32)
    host = Host(tmp_path, monkeypatch)
    run = tmp_path / "host-run"
    run.mkdir(mode=0o700)
    monkeypatch.setattr(retire_host, "RETIRE_RUN", run)
    directory = tmp_path / "cases" / "GF-GENERAL-LIVE-030.not_ready" / "spare-service"
    transport = NamedTransport(host, directory)
    window = controller.ServiceWindowController(
        cast(HostProbeFixture, transport),
        cluster_id=PINS["cluster_id"],
        node_uid=PINS["node_uid"],
        agent_identity=PINS["agent_identity"],
        plan_sha256="d" * 64,
        release_id="37db33f254a1",
        maintenance_expires_at=datetime.fromtimestamp(2_000_001_200, timezone.utc),
        read_node_uid=lambda: PINS["node_uid"],
    )
    window.create()
    window.stop("kubelet.service", delay_seconds=15)
    host.fire_stop()
    host.now = host.binding["restore_at"]
    host.elapsed = host.read()["clock"]["restore"]
    assert host.tick()["restore_reason"] == "deadline"
    host.units[host.window.units["restore-service"]].update(
        ActiveState="inactive", SubState="dead", MainPID="0"
    )
    execute = transport.execute

    def no_response(command: str, *args: str, timeout: int = 180) -> Any:
        if command == "cleanup":
            raise HostProbeMissingResponseError(
                "host probe response is missing (exit 1)"
            )
        return execute(command, *args, timeout=timeout)

    transport.execute = no_response  # type: ignore[method-assign]
    with pytest.raises(HostProbeMissingResponseError):
        window.cleanup()
    window.close()
    record = probe.private_read(transport.state_path)
    record["script_may_exist"] = True
    probe.write_record(transport.state_path, record)
    copy = run / f"{POD}.py"
    copy.write_bytes(Path(probe.__file__).read_bytes())
    copy.chmod(0o700)
    api = Api(record)
    RetireProbe.instances = []
    clock = itertools.count()
    settings = sw.RetireSettings(
        kubeconfig=transport.settings.kubeconfig,
        context="test-context",
        namespace="test-namespace",
        image="test-image",
        node_uid=lambda _node: PINS["node_uid"],
        host_probe=RetireProbe,
        run=api.run,
        sleep=lambda _seconds: None,
        clock=lambda: float(next(clock)),
    )
    return type(
        "Stale",
        (),
        {
            "host": host,
            "journal": window.path,
            "record": transport.state_path,
            "copy": copy,
            "api": api,
            "settings": settings,
            "directory": directory,
        },
    )


def _journal(stale: Any) -> dict[str, Any]:
    return probe.private_read(stale.journal)


def test_the_live_shape_retires_and_closes_its_custody(stale: Any) -> None:
    assert _journal(stale)["phase"] == "CLEANUP_REQUIRED"
    window = probe.ServiceWindow(_journal(stale)["binding"])
    kubelet = stale.host.units["kubelet.service"]["InvocationID"]

    entry = sw.retire_journal(
        stale.journal,
        PINS["node"],
        settings=stale.settings,
        by="operator@test",
        via="test",
    )

    assert entry["retired"] is True and entry["refusals"] == [], entry
    (retirer,) = RetireProbe.instances
    assert retirer.calls == ["create", "retire"] and retirer.cleaned
    assert retirer.settings.run_id == "test-service-run-retire"
    assert retirer.settings.case_id == "GF-REGIONAL-DESTR-008"
    assert retirer.settings.state_directory == stale.directory / "retire"
    assert retirer.settings.probe_script.name == "service-window-retire-probe.py"
    sha = hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest()
    assert retirer.arguments[-4:] == [
        "--stale-script",
        f"/run/{POD}.py",
        "--stale-script-sha256",
        sha,
    ]
    assert not window.targets["claim"].exists() and not window.directory.exists()
    assert not window.dropin.exists() and not stale.copy.exists()
    assert stale.host.units["kubelet.service"]["InvocationID"] == kubelet
    assert stale.api.objects == {}, "the earlier probe's Pod and ConfigMap are gone"
    assert [(d["object"], d["preconditions"]) for d in stale.api.deletes] == [
        ("pod", {"uid": "test-pod-uid"}),
        ("configmap", {"uid": "test-configmap-uid"}),
    ]
    record = probe.private_read(stale.record)
    assert record["closed"] is True and record["script_may_exist"] is False
    assert record["retired"]["by"] == "operator@test"
    journal = _journal(stale)
    assert journal["phase"] == "CLOSED"
    retirement = journal["retirement"]
    assert retirement["retired_by"] == "operator@test" and retirement["via"] == "test"
    assert retirement["phase_before"] == "CLEANUP_REQUIRED"
    assert retirement["proofs"]["shape"] == "restored"
    assert retirement["proofs"]["restored_invocation"] == kubelet
    assert str(window.targets["claim"]) in retirement["host_removed"]
    assert retirement["host_probe_deleted"] == [f"pod/{POD}", f"configmap/{POD}"]
    assert (
        sw.retire_stale_windows(
            stale.directory, PINS["node"], settings=stale.settings, by="x", via="y"
        )
        == []
    ), "nothing is open any more"


def _phase(stale: Any) -> None:
    data = _journal(stale)
    data["phase"] = "STOP_REQUESTED"
    probe.write_record(stale.journal, data)


def _lost(stale: Any) -> None:
    data = _journal(stale)
    data["supervision_lost"] = True
    probe.write_record(stale.journal, data)


def _replaced(stale: Any) -> None:
    object.__setattr__(stale.settings, "node_uid", lambda _node: "replacement-uid")


def _foreign_pod(stale: Any) -> None:
    stale.api.objects["pod"]["metadata"]["annotations"]["gpu-fault.io/probe-owner"] = (
        "f" * 32
    )


@pytest.mark.parametrize(
    ("defect", "needle"),
    [
        (_phase, "the journal is 'STOP_REQUESTED', not a finished run's open custody"),
        (_lost, "the journal records lost supervision"),
        (_replaced, "test-node is gone or was replaced (Node UID replacement-uid"),
        (_foreign_pod, f"pod/{POD} is not the earlier probe's own: owner annotation"),
    ],
)
def test_a_local_proof_fails_before_any_probe(
    stale: Any, defect: Any, needle: str
) -> None:
    defect(stale)
    before = _journal(stale)

    entry = sw.retire_journal(
        stale.journal, PINS["node"], settings=stale.settings, by="o", via="v"
    )

    assert entry["retired"] is False
    assert any(needle in reason for reason in entry["refusals"]), entry["refusals"]
    assert RetireProbe.instances == [], "no probe was created"
    assert _journal(stale) == before and len(stale.api.objects) == 2
    assert probe.private_read(stale.record)["closed"] is False


def test_a_host_refusal_is_named_and_everything_stays(stale: Any) -> None:
    stale.host.units["kubelet.service"]["InvocationID"] = "9" * 32  # restarted
    before = _journal(stale)

    entry = sw.retire_journal(
        stale.journal, PINS["node"], settings=stale.settings, by="o", via="v"
    )

    assert entry["retired"] is False
    assert any("runs invocation '9999" in r for r in entry["refusals"]), entry
    (retirer,) = RetireProbe.instances
    assert retirer.cleaned, "the retire probe is cleaned either way"
    assert _journal(stale) == before and len(stale.api.objects) == 2
    assert stale.copy.exists(), "the script copy is kept"


def test_a_journal_another_process_holds_is_refused(stale: Any) -> None:
    descriptor = os.open(stale.journal.with_suffix(".lock"), os.O_RDWR)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        entry = sw.retire_journal(
            stale.journal, PINS["node"], settings=stale.settings, by="o", via="v"
        )
    finally:
        os.close(descriptor)
    assert entry["refusals"] == ["another process holds the journal"]


def test_an_api_failure_is_a_named_refusal_not_a_crash(stale: Any) -> None:
    def broken(command: list[str], **_kwargs: Any) -> Any:
        raise sw.RegionalFixtureError("kubectl get failed")

    object.__setattr__(stale.settings, "run", broken)
    entry = sw.retire_journal(
        stale.journal, PINS["node"], settings=stale.settings, by="o", via="v"
    )
    assert entry["refusals"] == [
        "the retirement could not complete (RegionalFixtureError: kubectl get failed)"
    ]
    assert _journal(stale)["phase"] == "CLEANUP_REQUIRED"


def test_an_unbound_creating_journal_needs_no_probe(stale: Any) -> None:
    data = _journal(stale)
    data["phase"] = "CREATING"
    data.pop("binding")
    probe.write_record(stale.journal, data)
    record = probe.private_read(stale.record)
    record["script_may_exist"] = False
    probe.write_record(stale.record, record)

    entry = sw.retire_journal(
        stale.journal, PINS["node"], settings=stale.settings, by="o", via="v"
    )

    assert entry["retired"] is True, entry["refusals"]
    assert RetireProbe.instances == [], (
        "nothing was bound, nothing on the node to prove"
    )
    assert stale.api.objects == {} and _journal(stale)["phase"] == "CLOSED"


def test_inspect_journal_reads_only(stale: Any) -> None:
    before = _journal(stale)
    entry = sw.inspect_journal(stale.journal, PINS["node"], settings=stale.settings)
    assert entry["refusals"] == [] and entry["phase"] == "CLEANUP_REQUIRED"
    assert entry["host_probe"] == {
        "record": f"{POD}.json",
        "record_closed": False,
        "objects": [f"pod/{POD}", f"configmap/{POD}"],
    }
    window = probe.ServiceWindow(before["binding"])
    assert entry["window"]["key"] == window.key[:32]
    assert entry["window"]["units"] == sorted(window.units.values())
    assert _journal(stale) == before and stale.api.deletes == []
    assert RetireProbe.instances == []
