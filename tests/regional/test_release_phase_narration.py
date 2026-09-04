from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
STATE = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_state.py"
)

TIMESTAMPED_PHASE = re.compile(
    r"^release-phase \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z (?P<phase>[a-z-]+)"
)


def test_a_checkpoint_line_carries_a_timestamp_and_the_phase(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An upgrade prints thousands of command lines and no phase, until now.

    The line has to carry its own timestamp because nothing else in the deploy
    log has one, so without it there is no way to tell which phase was slow.
    """

    release = SimpleNamespace(
        state={"release_lifecycle": "CPU_STAGED", "cluster_ids": []}
    )

    STATE.narrate_phase(release, "cpu-staged")

    line = capsys.readouterr().err.strip()
    match = TIMESTAMPED_PHASE.match(line)
    assert match is not None, (
        f"a checkpoint line must be greppable and timestamped, got {line!r}"
    )
    assert match.group("phase") == "cpu-staged"
    assert "lifecycle=CPU_STAGED" in line, (
        "the release lifecycle is what an operator maps back to the runbook"
    )


def test_repeated_phases_are_distinguishable_by_cluster_progress(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`data-plane-progress` is checkpointed once per cluster per transition.

    Those writes all carry the same phase string, so a line naming only the
    phase would repeat verbatim and hide whether anything advanced.
    """

    release = SimpleNamespace(
        state={
            "release_lifecycle": "ROLLING_CLUSTERS",
            "cluster_ids": ["gpu-a", "gpu-b"],
            "completed_cluster_ids": ["gpu-a"],
        }
    )

    STATE.narrate_phase(release, "data-plane-progress")

    assert "clusters=1/2" in capsys.readouterr().err


def test_a_failed_write_narrates_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A printed phase is a promise that the checkpoint is durable.

    Resume and rollback both key off the phases persisted in the ConfigMap, so
    an operator who reads `cpu-staged` in the log and then has the deploy die
    must be able to trust that the phase is really recorded. Narrating before
    the write, or regardless of it, would turn the log into a source of false
    resume decisions.
    """

    def explode(_release: object, _phase: str, **_updates: object) -> None:
        raise STATE.ReleaseError("configmap apply rejected")

    monkeypatch.setattr(STATE, "_write_state", explode)
    release = SimpleNamespace(state={"cluster_ids": []})

    with pytest.raises(STATE.ReleaseError):
        STATE.save_state(release, "cpu-staged")

    assert capsys.readouterr().err == "", (
        "a checkpoint that never reached the ConfigMap was announced anyway"
    )


def test_a_successful_write_is_announced_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    written: list[str] = []

    monkeypatch.setattr(
        STATE, "_write_state", lambda _release, phase, **_updates: written.append(phase)
    )
    release = SimpleNamespace(state={"cluster_ids": []})

    STATE.save_state(release, "registry-staged")

    assert written == ["registry-staged"]
    assert capsys.readouterr().err.count("release-phase") == 1
