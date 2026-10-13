"""The deploy host's record of the last installation registry sync, read strictly.

``uninstall`` and ``remove-cluster`` decide on this record whether a deletion
attempt may start, so anything that is not exactly a record the release driver
wrote reads as unreadable (and refuses), a NOOP release keeps it only while it
is a success for the release the NOOP started from, and ``status`` turns it into
one finding line with the time, the reason and the remedy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import resource_registry_outcome as outcome
from tests.admin._registry_sync_support import SYNCED_AT, record_registry_sync

FAILURE = "installation resource registry not synchronized: BootstrapError: boom"


def _raw(state_dir: Path, value: Any) -> None:
    path = outcome.record_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        value if isinstance(value, str) else json.dumps(value), encoding="utf-8"
    )


def _valid() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "release_id": "release-a",
        "status": "SYNCED",
        "path": "/state/installation-resources.json",
        "sha256": "a" * 64,
        "synced_at": SYNCED_AT,
        "noop_release_ids": [],
    }


def test_a_written_outcome_reads_back_and_stays_private(tmp_path: Path) -> None:
    path = record_registry_sync(tmp_path, release_id="release-a")

    assert path == tmp_path / "release-deploy/installation-registry-sync.json"
    assert outcome.read_record(tmp_path) == {
        **_valid(),
        "path": str(tmp_path / "installation-resources.json"),
    }
    assert (path.stat().st_mode & 0o777, path.parent.stat().st_mode & 0o777) == (
        0o600,
        0o700,
    )


def test_an_outcome_the_reader_would_refuse_is_never_written(tmp_path: Path) -> None:
    with pytest.raises(outcome.RegistrySyncRecordError, match="snapshot path"):
        outcome.write_outcome(
            tmp_path,
            release_id="release-a",
            outcome={"status": "SYNCED", "synced_at": SYNCED_AT},
        )

    assert not outcome.record_path(tmp_path).exists(), "nothing half-valid lands"


@pytest.mark.parametrize(
    ("value", "problem"),
    [
        ("{not json", "is not valid JSON"),
        ([], "is not a JSON object"),
        ({**_valid(), "schema_version": 2}, "schema_version 2"),
        ({**_valid(), "schema_version": "1"}, "schema_version '1'"),
        ({**_valid(), "status": "SKIPPED"}, "unknown status 'SKIPPED'"),
        ({**_valid(), "release_id": " "}, "lacks its release_id"),
        ({**_valid(), "sha256": "A" * 64}, "snapshot path and sha256"),
        (
            {**_valid(), "status": "UNAVAILABLE", "error": ""},
            "failed sync without its error",
        ),
        ({**_valid(), "noop_release_ids": "release-b"}, "noop_release_ids"),
        ({**_valid(), "noop_release_ids": [""]}, "noop_release_ids"),
    ],
)
def test_anything_but_a_written_record_is_unreadable(
    tmp_path: Path, value: Any, problem: str
) -> None:
    _raw(tmp_path, value)

    with pytest.raises(outcome.RegistrySyncRecordError, match=problem):
        outcome.read_record(tmp_path)


def test_only_an_absent_record_reads_as_none(tmp_path: Path) -> None:
    assert outcome.read_record(tmp_path) is None
    outcome.record_path(tmp_path).mkdir(parents=True)
    with pytest.raises(outcome.RegistrySyncRecordError, match="cannot be read"):
        outcome.read_record(tmp_path)


def test_bytes_that_are_not_text_are_unreadable_not_a_crash(tmp_path: Path) -> None:
    outcome.record_path(tmp_path).parent.mkdir(parents=True)
    outcome.record_path(tmp_path).write_bytes(b"\xff\xfe\x00garbage")

    with pytest.raises(outcome.RegistrySyncRecordError, match="cannot be read"):
        outcome.read_record(tmp_path)
    assert outcome.status_view(tmp_path, release_id="release-a")["status"] == (
        "UNREADABLE"
    )


# --- the NOOP fast path -------------------------------------------------------------


def test_a_noop_keeps_a_success_of_its_own_release_untouched(tmp_path: Path) -> None:
    path = record_registry_sync(tmp_path, release_id="release-a")
    before = path.read_bytes()

    kept = outcome.noop_kept_record(
        tmp_path, release_id="release-a", live_release_id="release-a"
    )

    assert kept is not None and kept["release_id"] == "release-a"
    assert path.read_bytes() == before, "a NOOP must not rewrite the record"


def test_a_noop_staging_a_new_id_adds_it_once(tmp_path: Path) -> None:
    record_registry_sync(tmp_path, release_id="release-a")

    for _ in range(2):
        kept = outcome.noop_kept_record(
            tmp_path, release_id="release-b", live_release_id="release-a"
        )
        assert kept is not None, "the success of the live release still holds"

    record = outcome.read_record(tmp_path)
    assert record is not None
    assert (record["release_id"], record["noop_release_ids"]) == (
        "release-a",
        ["release-b"],
    )
    assert (
        outcome.noop_kept_record(
            tmp_path, release_id="release-c", live_release_id="release-b"
        )
        is not None
    ), "a NOOP over a NOOP-staged release still keeps the original success"


@pytest.mark.parametrize("state", ["missing", "failed", "unreadable", "foreign"])
def test_a_noop_must_sync_when_the_record_is_no_success_for_it(
    tmp_path: Path, state: str
) -> None:
    if state == "failed":
        record_registry_sync(tmp_path, error=FAILURE)
    elif state == "unreadable":
        _raw(tmp_path, "{")
    elif state == "foreign":
        record_registry_sync(tmp_path, release_id="release-old")

    assert (
        outcome.noop_kept_record(
            tmp_path, release_id="release-a", live_release_id="release-a"
        )
        is None
    )


def test_a_noop_that_cannot_note_its_id_syncs_instead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_registry_sync(tmp_path, release_id="release-a")

    def unwritable(*_args: object) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(outcome, "write_json_atomic", unwritable)

    assert (
        outcome.noop_kept_record(
            tmp_path, release_id="release-b", live_release_id="release-a"
        )
        is None
    )


# --- status -------------------------------------------------------------------------


def test_a_success_for_the_running_release_is_no_finding(tmp_path: Path) -> None:
    record_registry_sync(tmp_path, release_id="release-a")

    view = outcome.status_view(tmp_path, release_id="release-a")

    assert view == {
        "record": str(outcome.record_path(tmp_path)),
        "status": "SYNCED",
        "release_id": "release-a",
        "synced_at": SYNCED_AT,
        "error": None,
        "live_release_id": "release-a",
        "deletion_verbs_refused": False,
        "finding": None,
    }


def test_a_failed_sync_finding_states_time_reason_and_remedy(tmp_path: Path) -> None:
    record_registry_sync(tmp_path, release_id="release-a", error=FAILURE)

    view = outcome.status_view(tmp_path, release_id="release-a")

    assert view["status"] == "UNAVAILABLE"
    assert view["deletion_verbs_refused"] is True
    assert view["finding"] == (
        f"the last installation registry sync failed at {SYNCED_AT} (release "
        f"release-a): {FAILURE}; uninstall and remove-cluster refuse to start "
        f"while it stands -- `gpu-fault-admin deploy --state-dir {tmp_path}` (a "
        "NOOP deploy too: it retries the sync while this record is not a success) "
        "or the next `gpu-fault-admin config` re-roll synchronizes the registry again"
    )


@pytest.mark.parametrize("state", ["missing", "unreadable"])
def test_no_outcome_or_an_unreadable_one_is_a_finding_too(
    tmp_path: Path, state: str
) -> None:
    if state == "unreadable":
        _raw(tmp_path, {**_valid(), "status": "DONE"})

    view = outcome.status_view(tmp_path, release_id="release-a")

    assert view["status"] == state.upper()
    assert view["deletion_verbs_refused"] is True
    expected = (
        "no installation registry sync outcome is recorded in "
        f"{outcome.record_path(tmp_path)}"
        if state == "missing"
        else "the installation registry sync record is unreadable: "
    )
    assert view["finding"].startswith(expected), view["finding"]
    assert "uninstall and remove-cluster refuse to start" in view["finding"]
    if state == "unreadable":
        assert "unknown status 'DONE'" in view["error"]


def test_a_success_of_an_older_release_is_a_finding_without_a_refusal(
    tmp_path: Path,
) -> None:
    record_registry_sync(tmp_path, release_id="release-old")

    view = outcome.status_view(tmp_path, release_id="release-a")
    unknown = outcome.status_view(tmp_path, release_id=None)

    assert view["deletion_verbs_refused"] is False
    assert view["finding"].startswith(
        "no installation registry sync outcome since the live release release-a "
        f"(the last sync succeeded at {SYNCED_AT} for release release-old) -- "
        "`gpu-fault-admin deploy"
    ), view["finding"]
    assert "refuse" not in view["finding"]
    assert unknown["finding"] is None, "no live release to compare with"


# --- the deletion gate and its flag -----------------------------------------------


def test_a_synchronized_registry_passes_the_gate_with_evidence(tmp_path: Path) -> None:
    record_registry_sync(tmp_path, release_id="release-a")

    for accept in (False, True):
        evidence = outcome.deletion_gate(tmp_path, verb="uninstall", accept=accept)
        assert evidence["status"] == "SYNCED"
        assert evidence["accept_flag"] is accept
        assert evidence["accepted_unsynchronized"] is False, "nothing was overridden"
        assert evidence["outcome"] == outcome.read_record(tmp_path)


def test_an_accepted_unsynchronized_registry_keeps_what_was_wrong(
    tmp_path: Path,
) -> None:
    _raw(tmp_path, "{")

    evidence = outcome.deletion_gate(tmp_path, verb="remove-cluster", accept=True)

    assert (evidence["status"], evidence["outcome"]) == ("UNREADABLE", None)
    assert "is not valid JSON" in evidence["error"]
    assert evidence["accepted_unsynchronized"] is True


def test_the_flag_parses_on_its_own(capsys: pytest.CaptureFixture[str]) -> None:
    parser = argparse.ArgumentParser(prog="verb")
    outcome.add_accept_flag(parser, verb="uninstall")

    assert parser.parse_args([]).accept_unsynchronized_registry is False
    assert parser.parse_args([outcome.ACCEPT_FLAG]).accept_unsynchronized_registry, (
        "the flag turns the override on"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    assert "installation-registry-sync.json" in capsys.readouterr().out
