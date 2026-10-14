"""The BOOT-029 driver and the deletion verbs' installation registry sync gate.

``remove-cluster`` (stage 2) and ``uninstall`` (stage 4) refuse a new attempt
while the site's last installation registry sync is failed, missing or
unreadable. Before every removal and before the uninstall the driver judges
that record with the product's own reader, stops the stage with the recorded
outcome and the product's remedy before the verb runs, and never passes
``--accept-unsynchronized-registry``; an attempt that already started is left
to the verb, which resumes it on the verdict it started with.

The fakes come from the single-provider contract
(``test_admin_lifecycle_sequence_script``): the fake deploy records its sync
through the product's writer (``FAKE_REGISTRY_SYNC`` picks the outcome).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_fault.admin.resource_registry_outcome import ACCEPT_FLAG, record_path
from tests.admin._registry_sync_support import record_registry_sync
from tests.regional.test_admin_lifecycle_sequence_script import (
    _admin_verbs,
    _install_fakes,
    _record,
    _run,
    _tool_log,
)


def _never_overridden(env: dict[str, str]) -> bool:
    return not any(ACCEPT_FLAG in line for line in _tool_log(env))


def _stage_log(tmp_path: Path, stage: str) -> str:
    name = _record(tmp_path)["stages"][stage]["log_name"]
    return (tmp_path / "state" / "acceptance" / name).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("record", "status"),
    [("missing", "MISSING"), ("failed", "UNAVAILABLE"), ("unreadable", "UNREADABLE")],
    ids=["missing", "failed", "unreadable"],
)
def test_stage_two_stops_before_any_removal_while_the_registry_is_unsynchronized(
    tmp_path: Path, record: str, status: str
) -> None:
    env = _install_fakes(tmp_path)

    refused = _run(tmp_path, {**env, "FAKE_REGISTRY_SYNC": record})

    assert refused.returncode != 0, refused.stdout
    assert (
        "FAILED at stage 2 (remove-cluster): FAIL: remove-cluster would refuse to "
        f"start (installation registry sync {status})"
    ) in refused.stderr, refused.stderr
    assert (
        f"gpu-fault-admin deploy --state-dir {tmp_path / 'state'}" in refused.stderr
    ), "the refusal names the product's remedy"
    assert f"BOOT-029 never passes {ACCEPT_FLAG}" in refused.stderr, refused.stderr
    assert _admin_verbs(_tool_log(env)) == ["deploy"], (
        "no remove-cluster runs over an unsynchronized registry"
    )
    assert not (tmp_path / "state" / "remove-cluster").exists(), (
        "the refusal leaves no removal journal behind"
    )
    assert _record(tmp_path)["stages"]["2"]["status"] == "FAIL"
    assert "--stage 2" in refused.stderr, "the resume hint names the stage"


def test_a_resynchronized_site_resumes_stage_two_to_the_end(tmp_path: Path) -> None:
    env = _install_fakes(tmp_path)
    refused = _run(tmp_path, {**env, "FAKE_REGISTRY_SYNC": "missing"})
    assert refused.returncode != 0, refused.stdout
    # What the remedial deploy (a NOOP deploy retries the sync) records.
    record_registry_sync(tmp_path / "state")

    resumed = _run(tmp_path, env, "--stage", "2")

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert _admin_verbs(_tool_log(env)) == [
        "deploy",
        "remove-cluster",
        "join-cluster",
        "uninstall",
        "deploy",
    ], "the resumed stage removes the cluster and the sequence ends"
    assert _never_overridden(env), "the driver never passes the override flag"
    log = _stage_log(tmp_path, "2")
    assert (
        "installation registry sync SYNCED (release release-a, synced at "
        "2026-01-01T00:00:00Z): remove-cluster may start a new attempt"
    ) in log, "the stage log keeps the verdict the removal started on"


def test_stage_four_stops_before_the_uninstall_once_a_later_sync_failed(
    tmp_path: Path,
) -> None:
    env = _install_fakes(tmp_path)
    stopped = _run(tmp_path, {**env, "FAKE_ADMIN_FAIL_VERB": "join-cluster"})
    assert stopped.returncode != 0, stopped.stdout
    # A config re-roll between the stages failed to synchronize the registry.
    record_registry_sync(tmp_path / "state", error="fixture re-roll sync failed")

    refused = _run(tmp_path, env, "--stage", "3")

    assert refused.returncode != 0, refused.stdout
    assert (
        "FAILED at stage 4 (uninstall-reset): FAIL: uninstall would refuse to start "
        "(installation registry sync UNAVAILABLE)"
    ) in refused.stderr, refused.stderr
    assert "fixture re-roll sync failed" in refused.stderr, (
        "the refusal names the recorded failure"
    )
    assert "uninstall" not in _admin_verbs(_tool_log(env)), (
        "no uninstall runs over an unsynchronized registry"
    )
    assert not (tmp_path / "state" / "uninstall").exists(), (
        "the refusal leaves no uninstall journal behind"
    )
    record_registry_sync(tmp_path / "state")

    resumed = _run(tmp_path, env, "--stage", "4")

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert _admin_verbs(_tool_log(env))[-2:] == ["uninstall", "deploy"]
    assert _never_overridden(env), "the driver never passes the override flag"


@pytest.mark.parametrize(
    ("stage", "verb"), [(2, "remove-cluster"), (4, "uninstall")], ids=["s2", "s4"]
)
def test_an_attempt_that_already_started_is_left_to_the_verb(
    tmp_path: Path, stage: int, verb: str
) -> None:
    """The verb journaled its attempt and then exited non-zero; the record is
    gone since. Resuming the stage reruns the verb, which continues that
    attempt on the verdict it started with -- only a new attempt is judged."""

    env = _install_fakes(tmp_path)
    failed = _run(tmp_path, {**env, "FAKE_ADMIN_FAIL_AFTER_WRITE": verb})
    assert failed.returncode != 0, failed.stdout
    assert _record(tmp_path)["stages"][str(stage)]["status"] == "FAIL"
    record_path(tmp_path / "state").unlink()

    resumed = _run(tmp_path, env, "--stage", str(stage))

    verbs = _admin_verbs(_tool_log(env))
    assert verbs.count(verb) == 2, f"the started {verb} runs again: {verbs}"
    assert "may continue an attempt it already started" in _stage_log(
        tmp_path, str(stage)
    ), "the stage log says the attempt was left to the verb"
    if stage == 2:
        assert resumed.returncode != 0, resumed.stdout
        assert (
            "FAILED at stage 4 (uninstall-reset): FAIL: uninstall would refuse to "
            "start (installation registry sync MISSING)"
        ) in resumed.stderr, "a new uninstall attempt is still judged"
        assert "uninstall" not in verbs
    else:
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        assert verbs[-1] == "deploy", "the sequence reaches stage 5"
    assert _never_overridden(env), "the driver never passes the override flag"
