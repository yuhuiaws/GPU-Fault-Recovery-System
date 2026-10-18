"""Fail-closed paths of the installation retirement and retained-database records.

``test_admin_reinstall_handoff.py`` drives the full uninstall -> retire ->
reinstall cycle. These tests pin the individual refusals around it: records
that belong to another site or installation, Aurora identities that are not
a single online cluster, retirement journals that are malformed or whose
records moved, and record paths that are symlinks, FIFOs or contain one.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import installation_lifecycle as records
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from gpu_fault.installation_lifecycle import (
    INSTALLATION_FILE,
    RETIREMENT_FILE,
    content_sha256,
    site_identity,
)
from tests.admin.test_admin_site import site_file
from tests.admin.test_uninstall_lifecycle import Harness

INSTALLATION_ID = "a" * 32
NEXT_INSTALLATION_ID = "b" * 32
ARCHIVE = f"retired-20260101T000000Z-{'c' * 32}"
NAMES = ("site.yaml", "evidence")


def write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def completed_state(**overrides: Any) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "installation_id": INSTALLATION_ID,
        "site_id": "test-site",
        "phase": "COMPLETED",
        "cpu_disposition": "delete",
        "reset_database": False,
        "site_identity": {"site_name": "test-site"},
        **overrides,
    }


def state_dir_with_completed_uninstall(tmp_path: Path) -> Path:
    state_dir = tmp_path / "state"
    (state_dir / "uninstall").mkdir(parents=True)
    write(state_dir / "uninstall/state.json", completed_state())
    (state_dir / "site.yaml").write_text("kind: RegionalSite\n", encoding="utf-8")
    return state_dir


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def uninstall_digest(state_dir: Path) -> str:
    return content_sha256(
        {"state.json": file_digest(state_dir / "uninstall/state.json")}
    )


def prepared_journal(state_dir: Path, **overrides: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "PREPARED",
        "archive": ARCHIVE,
        "next_installation": {
            "schema_version": 1,
            "installation_id": NEXT_INSTALLATION_ID,
            "site_identity": {"site_name": "test-site"},
        },
        "records": {
            "site.yaml": file_digest(state_dir / "site.yaml"),
            "uninstall": uninstall_digest(state_dir),
        },
        **overrides,
    }


def test_missing_record_is_a_bootstrap_error(tmp_path):
    with pytest.raises(BootstrapError, match="missing or unsafe"):
        records.read_record(tmp_path / "absent.json")


def test_installation_of_another_site_is_refused_for_uninstall(tmp_path):
    site = load_site(site_file(tmp_path))
    path = site.source.parent / INSTALLATION_FILE
    write(
        path,
        {
            "schema_version": 1,
            "installation_id": INSTALLATION_ID,
            "site_identity": {
                **site_identity(site.release_config),
                "site_name": "another-site",
            },
        },
    )
    with pytest.raises(BootstrapError, match="belongs to another site"):
        records.installation_for_uninstall(site)
    assert json.loads(path.read_text())["installation_id"] == INSTALLATION_ID


def test_incomplete_retirement_blocks_uninstall_installation(tmp_path):
    site = load_site(site_file(tmp_path))
    write(
        site.source.parent / RETIREMENT_FILE,
        {"schema_version": 1, "phase": "PREPARED", "archive": ARCHIVE},
    )
    with pytest.raises(BootstrapError, match="retirement is incomplete"):
        records.installation_for_uninstall(site)
    assert not (site.source.parent / INSTALLATION_FILE).exists(), (
        "an incomplete retirement must not mint a new installation"
    )


def test_aurora_binding_requires_a_cluster_document():
    with pytest.raises(BootstrapError, match="identity is unavailable"):
        records.aurora_binding(
            None,
            cpu_eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/control",
            aws_region="us-east-1",
            cluster_id="gpu-fault-aurora",
        )


@pytest.mark.parametrize("clusters", [[], [{"a": 1}, {"b": 2}], "not-a-list"])
def test_retained_identity_requires_exactly_one_described_cluster(tmp_path, clusters):
    site = load_site(site_file(tmp_path))

    class Runner:
        calls: list[tuple[str, ...]] = []

        def aws_json(self, region: str, *arguments: str, **_kwargs: Any) -> Any:
            self.calls.append((region, *arguments))
            return {"DBClusters": clusters}

    runner = Runner()
    with pytest.raises(BootstrapError, match="identity is unavailable"):
        records.retained_database_identity(site, runner)  # type: ignore[arg-type]
    assert runner.calls == [
        (
            "us-east-1",
            "rds",
            "describe-db-clusters",
            "--db-cluster-identifier",
            "gpu-fault-aurora",
        )
    ]


def test_retained_database_needs_its_pre_cleanup_incarnation(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch)
    state_path = tmp_path / "state.json"
    state = {"phase": "CPU_VERIFIED"}
    with pytest.raises(BootstrapError, match="pre-cleanup incarnation"):
        records.bind_retained_database(harness.request(), harness, state_path, state)
    assert "retained_database" not in state
    assert not state_path.exists(), "a refused binding must not write a state file"


def test_retained_database_incarnation_must_not_change(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch)
    state_path = tmp_path / "state.json"
    state: dict[str, Any] = {"phase": "STARTED"}
    records.bind_retained_database(harness.request(), harness, state_path, state)
    bound = json.loads(state_path.read_text())["retained_database"]
    assert bound["cluster_resource_id"] == harness.database_resource_id
    harness.database_resource_id = "recreated-database"
    with pytest.raises(BootstrapError, match="incarnation changed"):
        records.bind_retained_database(harness.request(), harness, state_path, state)
    assert state["retained_database"] == bound


def test_retained_database_is_not_bound_for_delete_or_reset(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch)
    state_path = tmp_path / "state.json"
    for request in (harness.request(delete=True), harness.request(reset=True)):
        state: dict[str, Any] = {"phase": "COMPLETED"}
        records.bind_retained_database(request, harness, state_path, state)
        assert state == {"phase": "COMPLETED"}
    assert not state_path.exists(), "a skipped binding must not write a state file"


def test_handoff_with_malformed_archive_belongs_to_another_installation(tmp_path):
    path = tmp_path / INSTALLATION_FILE
    write(
        path,
        {
            "schema_version": 1,
            "installation_id": NEXT_INSTALLATION_ID,
            "retained_uninstall": {"archive": "not-an-archive"},
        },
    )
    write(tmp_path / RETIREMENT_FILE, {"schema_version": 1, "phase": "COMPLETED"})
    with pytest.raises(BootstrapError, match="belongs to another installation"):
        records.retained_handoff(
            path,
            installation_id=NEXT_INSTALLATION_ID,
            identity={},
            site_name="test-site",
        )
    assert "retained_adoption" not in json.loads(path.read_text())


def test_retirement_skips_without_a_completed_uninstall(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    assert records.retire_completed_site(state_dir, NAMES) is None
    (state_dir / "uninstall").mkdir()
    write(state_dir / "uninstall/state.json", completed_state(phase="CPU_VERIFIED"))
    assert records.retire_completed_site(state_dir, NAMES) is None
    assert not (state_dir / RETIREMENT_FILE).exists(), (
        "a refused retirement must not leave a journal"
    )


def test_symlinked_record_cannot_be_retired(tmp_path):
    state_dir = state_dir_with_completed_uninstall(tmp_path)
    real = tmp_path / "real-site.yaml"
    real.write_text("kind: RegionalSite\n", encoding="utf-8")
    (state_dir / "site.yaml").unlink()
    (state_dir / "site.yaml").symlink_to(real)
    with pytest.raises(BootstrapError, match="symlinked installation record"):
        records.retire_completed_site(state_dir, NAMES)
    assert not (state_dir / RETIREMENT_FILE).exists(), (
        "a refused retirement must not leave a journal"
    )


def test_record_that_is_neither_file_nor_directory_is_missing(tmp_path):
    state_dir = state_dir_with_completed_uninstall(tmp_path)
    os.mkfifo(state_dir / "evidence")
    with pytest.raises(BootstrapError, match="record is missing"):
        records.retire_completed_site(state_dir, NAMES)
    assert not (state_dir / RETIREMENT_FILE).exists(), (
        "a refused retirement must not leave a journal"
    )


def test_record_directory_with_a_symlink_inside_is_unsafe(tmp_path):
    state_dir = state_dir_with_completed_uninstall(tmp_path)
    evidence = state_dir / "evidence"
    (evidence / "nested").mkdir(parents=True)
    (evidence / "nested/report.json").write_text("{}", encoding="utf-8")
    (evidence / "link.json").symlink_to(evidence / "nested/report.json")
    with pytest.raises(BootstrapError, match="unsafe installation record"):
        records.retire_completed_site(state_dir, NAMES)
    assert not (state_dir / RETIREMENT_FILE).exists(), (
        "a refused retirement must not leave a journal"
    )


def test_retirement_digests_nested_record_directories(tmp_path):
    state_dir = state_dir_with_completed_uninstall(tmp_path)
    evidence = state_dir / "evidence"
    (evidence / "nested").mkdir(parents=True)
    (evidence / "nested/report.json").write_text("{}", encoding="utf-8")
    archive = records.retire_completed_site(state_dir, NAMES)
    assert archive is not None
    assert (archive / "evidence/nested/report.json").read_text() == "{}"
    assert not evidence.exists(), "retired records must leave the state directory"
    journal = json.loads((state_dir / RETIREMENT_FILE).read_text())
    assert set(journal["records"]) == {"site.yaml", "evidence", "uninstall"}
    assert journal["phase"] == "COMPLETED"


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"schema_version": 2}, "journal is invalid"),
        ({"phase": "STARTED"}, "journal is invalid"),
        ({"archive": "retired-today"}, "journal is invalid"),
        ({"records": ["site.yaml"]}, "journal is invalid"),
        ({"records": {"unexpected": "x", "uninstall": "y"}}, "journal is invalid"),
        ({"records": {"site.yaml": "x"}}, "journal is invalid"),
    ],
)
def test_invalid_prepared_journal_is_refused(tmp_path, overrides, message):
    state_dir = state_dir_with_completed_uninstall(tmp_path)
    journal = prepared_journal(state_dir, **overrides)
    write(state_dir / RETIREMENT_FILE, journal)
    with pytest.raises(BootstrapError, match=message):
        records.retire_completed_site(state_dir, NAMES)
    assert (state_dir / "site.yaml").is_file(), (
        "an invalid journal must not move any record"
    )
    assert json.loads((state_dir / RETIREMENT_FILE).read_text()) == journal


def test_symlinked_archive_is_unsafe(tmp_path):
    state_dir = state_dir_with_completed_uninstall(tmp_path)
    write(state_dir / RETIREMENT_FILE, prepared_journal(state_dir))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (state_dir / ARCHIVE).symlink_to(elsewhere)
    with pytest.raises(BootstrapError, match="archive is unsafe"):
        records.retire_completed_site(state_dir, NAMES)
    assert (state_dir / "site.yaml").is_file(), (
        "an unsafe archive must not move any record"
    )
    assert list(elsewhere.iterdir()) == []


def test_record_already_present_in_archive_means_records_changed(tmp_path):
    state_dir = state_dir_with_completed_uninstall(tmp_path)
    write(state_dir / RETIREMENT_FILE, prepared_journal(state_dir))
    (state_dir / ARCHIVE).mkdir()
    (state_dir / ARCHIVE / "site.yaml").write_text("other\n", encoding="utf-8")
    with pytest.raises(BootstrapError, match="records changed during retirement"):
        records.retire_completed_site(state_dir, NAMES)
    assert (state_dir / "site.yaml").is_file(), "a changed record must stay in place"
    assert (state_dir / "uninstall/state.json").is_file(), (
        "the uninstall record must stay in place"
    )


def test_record_missing_from_both_places_is_reported(tmp_path):
    state_dir = state_dir_with_completed_uninstall(tmp_path)
    write(state_dir / RETIREMENT_FILE, prepared_journal(state_dir))
    (state_dir / "site.yaml").unlink()
    with pytest.raises(BootstrapError, match="missing or changed"):
        records.retire_completed_site(state_dir, NAMES)
    journal = json.loads((state_dir / RETIREMENT_FILE).read_text())
    assert journal["phase"] == "PREPARED"


def test_retirement_resumes_with_records_already_moved(tmp_path):
    state_dir = state_dir_with_completed_uninstall(tmp_path)
    journal = prepared_journal(state_dir)
    write(state_dir / RETIREMENT_FILE, journal)
    archive = state_dir / ARCHIVE
    (archive / "uninstall").mkdir(parents=True)
    (state_dir / "site.yaml").replace(archive / "site.yaml")
    (state_dir / "uninstall/state.json").replace(archive / "uninstall/state.json")
    (state_dir / "uninstall").rmdir()
    assert records.retire_completed_site(state_dir, NAMES) == archive
    assert json.loads((state_dir / RETIREMENT_FILE).read_text())["phase"] == (
        "COMPLETED"
    )
    assert (
        json.loads((state_dir / INSTALLATION_FILE).read_text())
        == journal["next_installation"]
    )


def test_new_installation_identity_must_not_change_during_retirement(tmp_path):
    state_dir = state_dir_with_completed_uninstall(tmp_path)
    journal = prepared_journal(state_dir, phase="RECORDS_RETIRED")
    write(state_dir / RETIREMENT_FILE, journal)
    (state_dir / ARCHIVE).mkdir()
    write(
        state_dir / INSTALLATION_FILE,
        {"schema_version": 1, "installation_id": "d" * 32},
    )
    with pytest.raises(BootstrapError, match="identity changed during retirement"):
        records.retire_completed_site(state_dir, NAMES)
    assert json.loads((state_dir / INSTALLATION_FILE).read_text()) == {
        "schema_version": 1,
        "installation_id": "d" * 32,
    }
    assert not (state_dir / ARCHIVE / RETIREMENT_FILE).exists(), (
        "a refused retirement must not publish the archive journal"
    )


def test_retired_journal_with_incomplete_next_installation_is_refused(tmp_path):
    state_dir = state_dir_with_completed_uninstall(tmp_path)
    journal = prepared_journal(
        state_dir,
        phase="RECORDS_RETIRED",
        next_installation={"schema_version": 1, "installation_id": "short"},
    )
    write(state_dir / RETIREMENT_FILE, journal)
    (state_dir / ARCHIVE).mkdir()
    with pytest.raises(BootstrapError, match="identity is incomplete"):
        records.retire_completed_site(state_dir, NAMES)
    assert not (state_dir / INSTALLATION_FILE).exists(), (
        "an incomplete identity must not be published"
    )
