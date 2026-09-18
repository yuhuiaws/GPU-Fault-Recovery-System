from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import destr008_controller_lock as locking
from scripts.e2e.regional import destr008_gpu_holder as holder
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._destr008_gpu_holder import HolderHarness, build_holder


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> HolderHarness:
    return build_holder(tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"phase": "UNKNOWN"},
        {"unknown": "value"},
        {"binding": {}},
        {"created_at": None},
        {"created_at": 0.0},
        {"created_at": float("nan")},
        {"deadline_at": None},
        {"deadline_at": 2000000841.0},
        {"pod_uid": None},
        {"pod_spec_sha256": None},
        {"pod_ack_sha256": None},
        {"pod_ack_sha256": "invalid-digest"},
        {"pod_uid": "invalid/uid"},
        {"pod_spec_sha256": "invalid-digest"},
        {"pod_approved": False},
        {"create_started": False},
        {"phase": "PREPARING"},
        {"phase": "CREATING"},
    ]
    + [
        {
            "phase": phase,
            "pod_uid": None,
            "pod_spec_sha256": None,
            "pod_ack_sha256": None,
            "pod_approved": False,
        }
        for phase in ("CREATED", "READY", "CLEANING", "CLOSED")
    ]
    + [
        {
            "phase": "CREATING",
            "create_started": False,
            "created_at": None,
            "deadline_at": None,
            "pod_uid": None,
            "pod_spec_sha256": None,
            "pod_ack_sha256": None,
            "pod_approved": False,
        }
    ],
)
def test_inconsistent_durable_custody_is_refused_before_io(
    harness: HolderHarness, changes: dict[str, Any]
) -> None:
    harness.controller.create()
    record = harness.journal()
    record.update(changes)
    write_json_atomic(harness.controller.path, record)
    count = len(harness.calls)
    with pytest.raises(RegionalFixtureError, match="journal|binding"):
        harness.new_holder()
    with pytest.raises(RegionalFixtureError, match="journal|binding"):
        harness.controller.cleanup()
    assert len(harness.calls) == count, (
        "corrupt custody must not query or mutate the cluster"
    )


@pytest.mark.parametrize(
    "defect",
    [
        "malformed",
        "list",
        "duplicate",
        "mode",
        "symlink",
        "dangling-link",
        "hardlink",
        "oversize",
    ],
)
def test_untrusted_or_ambiguous_journals_cannot_authorize_cleanup(
    harness: HolderHarness, tmp_path: Path, defect: str
) -> None:
    harness.controller.create()
    path = harness.controller.path
    if defect == "malformed":
        path.write_text("{")
    elif defect == "list":
        path.write_text("[]")
    elif defect == "duplicate":
        path.write_text('{"schema_version":1,"schema_version":1}')
    elif defect == "mode":
        path.chmod(0o644)
    elif defect in {"symlink", "dangling-link"}:
        destination = tmp_path / "journal-target"
        path.rename(destination)
        path.symlink_to(destination)
        if defect == "dangling-link":
            destination.unlink()
    elif defect == "hardlink":
        os.link(path, tmp_path / "journal-link")
    else:
        path.write_text(" " * 262145)
    count = len(harness.calls)
    with pytest.raises(RegionalFixtureError):
        harness.new_holder()
    assert len(harness.calls) == count, "unsafe files cannot grant cleanup authority"


def test_write_size_bound_preserves_previous_journal(harness: HolderHarness) -> None:
    harness.controller.create()
    original = harness.journal()
    with locking.controller_ownership(harness.controller.path):
        assert harness.controller.record is not None, "the owned record must exist"
        harness.controller.record.binding["oversize"] = "x" * 200000
        with pytest.raises(RegionalFixtureError, match="size limit"):
            harness.controller.save()
    assert harness.journal() == original, (
        "an oversized write must not damage prior custody"
    )


def test_source_binding_rechecked_inside_held_operation(
    harness: HolderHarness, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness.controller.create()
    count = len(harness.calls)
    with harness.controller.operation():
        source = tmp_path / "different-controller.py"
        source.write_text("unapproved local test source\n")
        monkeypatch.setattr(holder, "__file__", str(source))
        with pytest.raises(RegionalFixtureError, match="connection or source"):
            harness.controller.read()
    assert len(harness.calls) == count, (
        "source drift cannot pass the held-operation gate"
    )
    with pytest.raises(RegionalFixtureError, match="binding changed"):
        harness.new_holder()


@pytest.mark.parametrize("field", ["image", "hold_seconds"])
def test_in_memory_manifest_mutation_invalidates_authority(
    harness: HolderHarness, field: str
) -> None:
    harness.controller.create()
    value: Any = "unapproved-image" if field == "image" else 900
    setattr(harness.controller, field, value)
    count = len(harness.calls)
    with pytest.raises(RegionalFixtureError, match="connection or source"):
        harness.controller.cleanup()
    assert len(harness.calls) == count, (
        "modified manifest parameters invalidate cleanup"
    )


def test_connection_changed_during_node_read_cannot_reach_pod_api(
    harness: HolderHarness,
) -> None:
    def change(args: tuple[str, ...]) -> None:
        if args[:2] == ("get", "node"):
            harness.regional.settings.gpu_kubeconfig.write_text(
                "changed local config\n"
            )

    harness.before = change
    with pytest.raises(RegionalFixtureError, match="connection or source"):
        harness.controller.create()
    assert harness.calls == [("get", "node", "spare-a", "-o", "json")], (
        "connections must remain pinned after the live identity read"
    )


@pytest.mark.parametrize("started", [False, True])
def test_supervision_loss_without_a_writable_record_never_runs_cleanup(
    harness: HolderHarness, monkeypatch: pytest.MonkeyPatch, started: bool
) -> None:
    if started:
        harness.controller.create()
    original = harness.journal() if started else None

    def fail_write(*args: Any, **kwargs: Any) -> None:
        raise OSError("local fake: disk write failed")

    monkeypatch.setattr(holder, "write_json_atomic", fail_write)
    count = len(harness.calls)
    with pytest.raises(
        ProcessSupervisionLost, match="could not be recorded" if started else "lost"
    ):
        with harness.controller.operation():
            raise ProcessSupervisionLost("local fake: supervision lost")
    assert len(harness.calls) == count, (
        "loss of supervision never authorizes more commands"
    )
    if started:
        assert harness.journal() == original, (
            "failed persistence must retain previous evidence"
        )
    else:
        assert not harness.controller.path.exists(), (
            "no ownership evidence may be fabricated"
        )


@pytest.mark.parametrize("phase", ["PREPARING", "CREATING", "CREATED"])
def test_interruption_at_each_durable_stage_cannot_rearm(
    harness: HolderHarness, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    def stop_after_save(path: Path, value: dict[str, Any]) -> None:
        write_json_atomic(path, value)
        if value["phase"] == phase:
            raise KeyboardInterrupt()

    monkeypatch.setattr(holder, "write_json_atomic", stop_after_save)
    with pytest.raises(KeyboardInterrupt):
        harness.controller.create()
    record = harness.journal()
    assert record["phase"] == phase, "the interrupted stage must remain durable"
    fresh = harness.new_holder()
    with pytest.raises(RegionalFixtureError, match="cleanup-only"):
        fresh.create()
    if phase == "CREATING":
        with pytest.raises(RegionalFixtureError, match="outcome is unknown"):
            fresh.resume_cleanup()
    else:
        fresh.resume_cleanup()
        assert harness.journal()["phase"] == "CLOSED", (
            "known custody can finish cleanup"
        )
    assert sum(args[0] == "create" for args in harness.mutations()) == (
        phase == "CREATED"
    ), "reconstruction must never repeat CREATE after an interrupted intent"


def test_separate_process_cannot_enter_an_owned_holder_run(
    harness: HolderHarness, tmp_path: Path
) -> None:
    program = """
import json, sys
from pathlib import Path
from pytest import MonkeyPatch
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._destr008_gpu_holder import build_holder
with MonkeyPatch.context() as patch:
    try:
        build_holder(Path(sys.argv[1]), patch)
    except RegionalFixtureError as exc:
        print(json.dumps({"refused": str(exc)}))
    else:
        raise AssertionError("a competing controller must not enter")
"""
    with locking.controller_ownership(harness.controller.path):
        child = subprocess.run(
            [sys.executable, "-c", program, str(tmp_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    assert json.loads(child.stdout)["refused"] == "another controller owns this run", (
        "holder custody must exclude competing fresh interpreters"
    )
    assert not harness.calls, "lock exclusion must precede API access"
