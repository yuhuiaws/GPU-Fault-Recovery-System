from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_release_narration as NARRATION
from gpu_fault_release import regional_release_state as STATE
from gpu_fault_release import rollout as ROLLOUT

ROOT = Path(__file__).resolve().parents[2]


TIMESTAMPED_PHASE = re.compile(
    r"^release-phase \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z (?P<phase>[a-z-]+)"
)
ELAPSED = re.compile(r"elapsed=(?P<elapsed>\d+\.\d)s total=(?P<total>\d+\.\d)s")
TOTAL = re.compile(r"total=\d+\.\ds$")


@pytest.fixture(autouse=True)
def _fresh_clock() -> None:
    """Every test measures from its own origin, not from the session's."""

    NARRATION.PHASE_CLOCK.start()


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

    NARRATION.narrate_phase(release, "cpu-staged")

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

    NARRATION.narrate_phase(release, "data-plane-progress")

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


def test_a_phase_reports_what_reaching_it_cost(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The stamps alone cannot answer "which phase is slow" without arithmetic.

    They are second-resolution and thousands of lines apart, so an operator
    hunting a five-minute upgrade had to subtract two timestamps by hand for each
    of sixteen checkpoints. `elapsed` is the cost of this phase and `total` the
    cost so far, both measured on a monotonic clock so a clock step mid-release
    cannot produce a negative phase.
    """

    release = SimpleNamespace(state={"cluster_ids": []})

    NARRATION.narrate_phase(release, "uploaded")
    NARRATION.narrate_phase(release, "schema-ready")

    first, second = [
        ELAPSED.search(line) for line in capsys.readouterr().err.strip().splitlines()
    ]
    assert first is not None and second is not None, "no phase carried a duration"
    assert float(second.group("total")) >= float(first.group("total")), (
        "total must accumulate across phases"
    )
    assert float(second.group("elapsed")) <= float(second.group("total")), (
        "a single phase cannot cost more than the whole release"
    )


def test_an_invocation_is_bounded_even_when_it_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One `gpu-fault-admin deploy` drives this script a dozen times.

    Their outputs run together in one stream, so without a begin and an end line
    an operator cannot attribute a slow upgrade to an invocation, and cannot tell
    a process that died from one that is still inside a long command. The end
    line therefore has to survive the failure path too.
    """

    monkeypatch.setattr(
        ROLLOUT,
        "_run_mode",
        lambda _arguments: (_ for _ in ()).throw(ROLLOUT.ReleaseError("no config")),
    )
    monkeypatch.setattr(
        ROLLOUT.sys, "argv", ["rollout", "upgrade", "--config", "absent.json"]
    )

    assert ROLLOUT.main() == 2

    err = capsys.readouterr().err
    assert "release-begin" in err and "mode=upgrade" in err
    assert "dry_run=false" in err
    end = [line for line in err.splitlines() if line.startswith("release-end ")]
    assert len(end) == 1, err
    assert "exit_code=2" in end[0]
    assert TOTAL.search(end[0]) is not None, end[0]


def _succeeding_subprocess(monkeypatch: pytest.MonkeyPatch, **fields: object) -> None:
    monkeypatch.setattr(
        ROLLOUT.subprocess,
        "run",
        lambda *_args, **_keywords: SimpleNamespace(
            returncode=0, stdout="", stderr="", **fields
        ),
    )


def test_only_a_slow_command_is_annotated(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A release issues thousands of sub-second calls and a handful of long ones.

    The `+ command` echo carries no duration, so the long ones are indistinguish-
    able from the rest: measured on the 2026-09-05 upgrade, 2m43s hid inside ten
    consecutive lines. Annotating every command instead would double a
    three-thousand-line log to say nothing.
    """

    runner = ROLLOUT.Runner()
    _succeeding_subprocess(monkeypatch)

    monkeypatch.setattr(ROLLOUT, "SLOW_COMMAND_SECONDS", 3600.0)
    runner.run(["/usr/bin/kubectl", "get", "pods"], capture=True)
    assert "command-elapsed" not in capsys.readouterr().err, (
        "a fast command was annotated, which is what buries the trace"
    )

    monkeypatch.setattr(ROLLOUT, "SLOW_COMMAND_SECONDS", 0.0)
    runner.run(["/bin/bash", "render-control-plane-role-split.sh"], capture=True)

    annotated = [
        line
        for line in capsys.readouterr().err.splitlines()
        if line.startswith("command-elapsed ")
    ]
    assert len(annotated) == 1, annotated
    assert annotated[0].endswith("s bash render-control-plane-role-split.sh"), (
        annotated[0]
    )


def test_a_slow_command_that_failed_still_reports_its_duration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A command that failed after four minutes is the one worth measuring.

    Its failure is already loud; what the log loses without this is that the four
    minutes belonged to it rather than to the phase around it.
    """

    runner = ROLLOUT.Runner()
    monkeypatch.setattr(ROLLOUT, "SLOW_COMMAND_SECONDS", 0.0)
    monkeypatch.setattr(
        ROLLOUT.subprocess,
        "run",
        lambda *_args, **_keywords: SimpleNamespace(
            returncode=1, stdout="", stderr="denied"
        ),
    )

    with pytest.raises(ROLLOUT.ReleaseError):
        runner.run(["/usr/bin/kubectl", "apply", "-f", "-"], capture=True)

    assert "s kubectl apply -f -" in capsys.readouterr().err


def test_a_sensitive_command_is_timed_without_being_named(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Timing must not become the leak the `+` echo was careful to avoid.

    Commands carrying a password or token are echoed as `<sensitive command>`; an
    elapsed line spelling the arguments out would undo that for exactly the slow
    ones an operator goes looking for.
    """

    runner = ROLLOUT.Runner()
    monkeypatch.setattr(ROLLOUT, "SLOW_COMMAND_SECONDS", 0.0)
    _succeeding_subprocess(monkeypatch)

    runner.run(
        ["/usr/bin/kubectl", "create", "secret", "--from-literal=password=hunter2"],
        capture=True,
        sensitive=True,
    )

    err = capsys.readouterr().err
    assert "command-elapsed" in err
    assert "hunter2" not in err, "the elapsed line leaked a secret the echo hid"


def test_a_silent_probe_is_annotated_when_it_stalls(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A probe echoes nothing at all, so a slow one is a wholly silent gap.

    There are hundreds of them and their answers show up in what the release does
    next, which is why they stay unannounced going in -- but that also means a
    probe waiting on an unreachable API server produces no line whatsoever.
    """

    runner = ROLLOUT.Runner()
    monkeypatch.setattr(ROLLOUT, "SLOW_COMMAND_SECONDS", 0.0)
    monkeypatch.setattr(
        ROLLOUT.subprocess,
        "run",
        lambda *_args, **_keywords: SimpleNamespace(returncode=1),
    )

    assert runner.probe(["/usr/bin/kubectl", "get", "configmap", "absent"]) is False

    assert "s kubectl get configmap absent" in capsys.readouterr().err


def test_a_long_command_line_is_shortened_before_it_is_timed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Some commands carry a rendered manifest as an argument, and only the
    elapsed line may shorten it.

    An untruncated elapsed line would run for thousands of columns and push the
    trace off the screen. The echo above it must stay whole regardless: a
    non-probe argument is the release's real input, and shortening it there would
    take the record of what actually ran with it.
    """

    runner = ROLLOUT.Runner()
    monkeypatch.setattr(ROLLOUT, "SLOW_COMMAND_SECONDS", 0.0)
    _succeeding_subprocess(monkeypatch)

    runner.run(["/usr/bin/kubectl", "apply", "-f", "x" * 4000], capture=True)

    lines = capsys.readouterr().err.splitlines()
    elapsed = [line for line in lines if line.startswith("command-elapsed ")]
    assert len(elapsed) == 1, elapsed
    assert len(elapsed[0]) < 260, len(elapsed[0])
    assert elapsed[0].endswith("..."), elapsed[0]
    echo = [line for line in lines if line.startswith("+ ")]
    assert echo == ["+ kubectl apply -f " + "x" * 4000], (
        "the echo is the record of what ran and must not be shortened"
    )
