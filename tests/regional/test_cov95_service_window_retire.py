"""Retiring a stale service window: every proof that can fail, named.

``test_service_window_retire.py`` builds the live 2026-10-27 leftover (the
``stale`` fixture: a CLEANUP_REQUIRED journal, its open host-probe record, the
probe's Pod/ConfigMap and ``/run`` script copy) and retires it; here the same
world is bent the ways a node, an API or another operator can bend it -- a
journal of another schema, node, case, scope or binding, a journal or record
that cannot be read or is malformed, another cluster context, a node replaced,
a record locked by another process, objects without their record, a retire
probe that fails or leaves residuals, a delete the API never acknowledges -- and
each must leave the journal open with the reason, or (record and objects
already gone) still retire.
"""

from __future__ import annotations

import dataclasses
import fcntl
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import service_window_retire as sw
from scripts.e2e.regional.host_probe_fixture import HostProbeError
from scripts.e2e.regional.probes import warm_spare_node_probe as probe
from tests.regional import test_service_window_retire as retire_tests
from tests.regional._destr008_service_window import PINS

#: The ``stale`` world of test_service_window_retire, reused as it is.
stale = retire_tests.stale
RetireProbe = retire_tests.RetireProbe
POD = retire_tests.POD
NODE = PINS["node"]


def _journal(world: Any) -> dict[str, Any]:
    return probe.private_read(world.journal)


def _retire(world: Any) -> dict[str, Any]:
    return sw.retire_journal(
        world.journal, NODE, settings=world.settings, by="operator@test", via="test"
    )


def _kept(world: Any, before: dict[str, Any]) -> None:
    assert _journal(world) == before, "a refused retirement changes no journal"


# --------------------------------------------------------------------------- #
# The local proofs of the journal itself
# --------------------------------------------------------------------------- #
def test_each_local_proof_of_the_journal_is_named(stale: Any) -> None:
    data = _journal(stale)
    assert sw.journal_refusals([data], stale.journal, NODE) == [
        "the journal is not a JSON object"
    ]
    assert sw.journal_refusals(dict(data, scope=None), stale.journal, NODE) == [
        "the journal has no scope"
    ]
    scope = dict(
        data["scope"],
        node="ip-10-0-99-9",
        case_id="GF-REGIONAL-DESTR-014",
        namespace="",
        host_state_path=str(stale.directory.parent / f"{POD}.json"),
    )
    refusals = sw.journal_refusals(
        dict(data, schema_version=2, scope=scope), stale.journal, NODE
    )
    assert refusals[:5] == [
        "the journal's schema is unknown",
        f"the journal is for 'ip-10-0-99-9', not {NODE}",
        "the journal is not a DESTR-008 service window's",
        "the journal's scope has no namespace",
        "the journal's host-probe record is not beside it",
    ], refusals


def test_the_binding_must_be_valid_its_own_and_never_on_a_creating_journal(
    stale: Any,
) -> None:
    data = _journal(stale)
    broken = dict(data, binding={**data["binding"], "owner": "not-hex"})
    (refusal,) = sw.journal_refusals(broken, stale.journal, NODE)
    assert (
        refusal == "the journal's binding is invalid (service binding owner is invalid)"
    )
    foreign = dict(data, owner="f" * 32)
    assert sw.journal_refusals(foreign, stale.journal, NODE) == [
        "the binding is not this journal's own"
    ]
    creating = dict(data, phase="CREATING")
    assert sw.journal_refusals(creating, stale.journal, NODE) == [
        "a CREATING journal cannot carry a binding"
    ]


# --------------------------------------------------------------------------- #
# The read-only inspection
# --------------------------------------------------------------------------- #
def test_inspection_names_the_first_proof_that_fails_and_touches_nothing(
    stale: Any,
) -> None:
    missing = sw.inspect_journal(
        stale.directory / "absent.json", NODE, settings=stale.settings
    )
    assert missing["refusals"] == ["the journal cannot be read (FileNotFoundError)"]
    data = _journal(stale)
    probe.write_record(stale.journal, dict(data, supervision_lost=True))
    lost = sw.inspect_journal(stale.journal, NODE, settings=stale.settings)
    assert lost["refusals"] == [
        "the journal records lost supervision: a command may still run"
    ]
    assert "host_probe" not in lost, "a failed local proof reads nothing remote"
    probe.write_record(stale.journal, data)

    other = dataclasses.replace(stale.settings, context="other-context")
    assert sw.inspect_journal(stale.journal, NODE, settings=other)["refusals"] == [
        "the journal belongs to another cluster context or namespace"
    ]
    gone = dataclasses.replace(stale.settings, node_uid=lambda _n: "")
    assert sw.inspect_journal(stale.journal, NODE, settings=gone)["refusals"] == [
        f"{NODE} is gone or was replaced (Node UID absent)"
    ]
    assert stale.api.deletes == [] and RetireProbe.instances == []


def test_inspection_of_an_unbound_journal_shows_no_window(stale: Any) -> None:
    data = _journal(stale)
    data.pop("binding")
    probe.write_record(stale.journal, data)
    entry = sw.inspect_journal(stale.journal, NODE, settings=stale.settings)
    assert entry["refusals"] == [] and "window" not in entry, entry
    assert entry["host_probe"]["objects"] == [f"pod/{POD}", f"configmap/{POD}"]


# --------------------------------------------------------------------------- #
# Retirement refusals before and around the host probe
# --------------------------------------------------------------------------- #
def test_another_context_and_a_held_record_refuse_the_retirement(stale: Any) -> None:
    before = _journal(stale)
    other = dataclasses.replace(stale.settings, namespace="kube-system")
    entry = sw.retire_journal(stale.journal, NODE, settings=other, by="o", via="v")
    assert entry["refusals"] == [
        "the journal belongs to another cluster context or namespace"
    ]
    descriptor = os.open(stale.record.with_suffix(".lock"), os.O_CREAT | os.O_RDWR)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        held = _retire(stale)
    finally:
        os.close(descriptor)
    assert held["refusals"] == ["another process holds the earlier host probe"]
    _kept(stale, before)
    assert RetireProbe.instances == [], "no retire probe while the record is held"


@pytest.mark.parametrize(
    ("content", "refusal"),
    [
        ("{not json", "the earlier host-probe record cannot be read (JSONDecodeError)"),
        (
            '{"scope": {}, "resources": {}, "closed": "no", "script_may_exist": true}',
            "the earlier host-probe record is malformed",
        ),
    ],
)
def test_a_record_that_cannot_be_read_or_is_malformed_refuses(
    stale: Any, content: str, refusal: str
) -> None:
    before = _journal(stale)
    stale.record.write_text(content, encoding="utf-8")
    entry = _retire(stale)
    assert entry["retired"] is False and refusal in entry["refusals"], entry
    _kept(stale, before)


def test_objects_without_their_record_are_not_the_probes_to_delete(stale: Any) -> None:
    before = _journal(stale)
    stale.record.unlink()
    entry = _retire(stale)
    assert entry["refusals"] == [
        f"pod/{POD} exists but its ownership record does not",
        f"configmap/{POD} exists but its ownership record does not",
    ]
    assert len(stale.api.objects) == 2 and stale.api.deletes == []
    _kept(stale, before)


def test_with_record_and_objects_already_gone_the_window_still_retires(
    stale: Any,
) -> None:
    stale.record.unlink()
    stale.api.objects.clear()
    stale.copy.unlink()
    entry = _retire(stale)
    assert entry["retired"] is True, entry["refusals"]
    assert entry["host_probe_deleted"] == []
    (retirer,) = RetireProbe.instances
    assert "--stale-script" not in retirer.arguments, "no record: no script to prove"
    assert _journal(stale)["phase"] == "CLOSED"
    assert not stale.record.exists(), "no record is invented"


class FailingRetireProbe(RetireProbe):
    """The retire probe's transport failing its command and its own cleanup."""

    def execute(self, command: str, *arguments: str, timeout: int = 180) -> Any:
        self.calls.append(command)
        raise HostProbeError("the retire probe Pod was evicted")

    def cleanup(self) -> dict[str, bool]:
        raise HostProbeError("delete pod/retire timed out")


class LeakyRetireProbe(RetireProbe):
    def cleanup(self) -> dict[str, bool]:
        return {f"pod/{POD}-retire": True, "host_script": False}


def test_a_failed_or_leaky_retire_probe_keeps_the_window_open(stale: Any) -> None:
    before = _journal(stale)
    failing = dataclasses.replace(stale.settings, host_probe=FailingRetireProbe)
    entry = sw.retire_journal(stale.journal, NODE, settings=failing, by="o", via="v")
    assert entry["refusals"] == [
        "the host retirement did not finish (HostProbeError: the retire probe Pod "
        "was evicted)",
        "the retire probe was not cleaned up (delete pod/retire timed out)",
    ]
    _kept(stale, before)
    leaky = dataclasses.replace(stale.settings, host_probe=LeakyRetireProbe)
    entry = sw.retire_journal(stale.journal, NODE, settings=leaky, by="o", via="v")
    assert entry["refusals"] == [
        f"the retire probe left residuals ['pod/{POD}-retire']"
    ]
    assert len(stale.api.objects) == 2, "nothing is deleted after a leaky retire probe"
    assert _journal(stale)["phase"] == "CLEANUP_REQUIRED"


def test_a_delete_the_api_never_acknowledges_keeps_the_custody(stale: Any) -> None:
    run = stale.api.run

    def ignore_deletes(command: list[str], **kwargs: Any) -> Any:
        if command[7] == "delete":
            stale.api.deletes.append({"ignored": command[9]})
            return subprocess.CompletedProcess(command, 0, "", "")
        return run(command, **kwargs)

    patient = dataclasses.replace(stale.settings, run=ignore_deletes)
    entry = sw.retire_journal(stale.journal, NODE, settings=patient, by="o", via="v")
    assert entry["refusals"] == [f"pod/{POD} deletion was not acknowledged"]
    assert entry["host_probe_deleted"] == []
    assert len(stale.api.objects) == 2 and len(stale.api.deletes) == 1, (
        "the first unacknowledged delete stops the rest"
    )
    assert probe.private_read(stale.record)["closed"] is False
    assert _journal(stale)["phase"] == "CLEANUP_REQUIRED"


# --------------------------------------------------------------------------- #
# The retire probe script
# --------------------------------------------------------------------------- #
def test_a_stale_retire_probe_script_is_rewritten_privately(tmp_path: Path) -> None:
    directory = tmp_path / "retire"
    directory.mkdir()
    stale_copy = directory / sw.RETIRE_SCRIPT
    stale_copy.write_text("# an older retire probe\n", encoding="utf-8")
    path = sw.write_retire_probe(directory)
    assert path == stale_copy
    assert path.read_text(encoding="utf-8") == sw.retire_probe_source()
    assert path.stat().st_mode & 0o777 == 0o600


def test_the_retire_probe_refuses_sources_that_lost_their_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sw, "HOST_PART_MARKER", "# --- no such seam ---\n")
    with pytest.raises(sw.RegionalFixtureError, match="lost their seams"):
        sw.retire_probe_source()
