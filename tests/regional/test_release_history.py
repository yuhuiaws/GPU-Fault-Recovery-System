"""Release/rollback/join/remove audit history (architecture review H5).

Release state was one overwritten ConfigMap, previous snapshots were deleted
at commit, and no path recorded who ran the command. Every state transition
now appends an immutable entry (release id, phase, plan/state digests,
timestamp, operator identity, redacted command line) to a bounded
append-only ``gpu-fault-release-history`` ConfigMap, mirrored to the admin
state directory, and the last few previous snapshots are retained.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault_release import regional_release_history as HISTORY
from gpu_fault_release import regional_release_state as STATE

TOKEN = "t" * 40


class _Runner:
    """Records kubectl calls; serves the existing history back on `get`."""

    dry_run = False

    def __init__(self, existing: list[dict] | None = None):
        self.calls: list[tuple[list[str], dict]] = []
        self.existing = existing

    def run(self, arguments, **kwargs):
        self.calls.append((list(arguments), kwargs))
        return ""

    def probe(self, arguments, **_kwargs):
        return self.existing is not None


def _release(runner: _Runner, *, state: dict | None = None) -> SimpleNamespace:
    release = SimpleNamespace(
        runner=runner,
        release_id="rel-42",
        config=SimpleNamespace(namespace="gpu-fault-system"),
        state=state
        or {
            "phase": "cpu-staged",
            "execution_plan": {"nodes": ["registry", "cpu-stage"]},
        },
        _cpu=lambda *arguments: list(arguments),
        _get_json=lambda _arguments: {
            "data": (
                {
                    HISTORY.HISTORY_KEY: "\n".join(
                        json.dumps(item) for item in runner.existing
                    )
                }
                if runner.existing is not None
                else {}
            )
        },
    )
    return release


def _applied_entries(runner: _Runner) -> list[dict]:
    applies = [
        json.loads(kwargs["input_text"])
        for arguments, kwargs in runner.calls
        if arguments[-3:] == ["apply", "-f", "-"]
    ]
    assert len(applies) == 1, "history must be written by exactly one apply"
    document = applies[0]
    assert document["metadata"]["name"] == HISTORY.HISTORY_CONFIG_MAP
    return [
        json.loads(line)
        for line in document["data"][HISTORY.HISTORY_KEY].splitlines()
        if line
    ]


@pytest.fixture(autouse=True)
def _no_aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """No STS on the test host: identity falls back to user@host."""

    monkeypatch.setattr(
        HISTORY, "resolve_operator_identity", lambda *, fallback: fallback
    )


def test_state_transition_appends_an_entry_with_operator_and_digests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(HISTORY.HISTORY_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(
        "sys.argv",
        ["rollout_regional_release.py", "upgrade", "--config", "/secure/r.json"],
    )
    runner = _Runner(existing=[{"release_id": "rel-41", "phase": "complete"}])
    release = _release(runner)

    HISTORY.record_release_history(release, phase="cpu-staged", state_text='{"a":1}')

    entries = _applied_entries(runner)
    assert [item["release_id"] for item in entries] == ["rel-41", "rel-42"], (
        "history must append, never overwrite"
    )
    entry = entries[-1]
    assert entry["phase"] == "cpu-staged"
    assert entry["state_sha256"] == HISTORY.sha256_text('{"a":1}')
    assert entry["plan_sha256"] == HISTORY.sha256_text(
        json.dumps({"nodes": ["registry", "cpu-stage"]}, sort_keys=True)
    )
    assert entry["timestamp"].endswith("Z"), "timestamps are UTC, marked with Z"
    assert "@" in entry["operator"], "fallback identity must be user@host"
    assert entry["command"] == "upgrade --config /secure/r.json"
    mirrored = (tmp_path / HISTORY.HISTORY_MIRROR_FILE).read_text().splitlines()
    assert json.loads(mirrored[-1]) == entry, "the state-dir mirror must match"


def test_operator_identity_prefers_the_aws_caller_arn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolutions: list[str] = []

    def resolve(*, fallback: str) -> str:
        resolutions.append(fallback)
        return "arn:aws:sts::123456789012:assumed-role/ops/x"

    monkeypatch.setattr(HISTORY, "resolve_operator_identity", resolve)
    runner = _Runner(existing=None)
    release = _release(runner)

    HISTORY.record_release_history(release, phase="preflight", state_text="{}")

    entries = _applied_entries(runner)
    assert entries[-1]["operator"] == "arn:aws:sts::123456789012:assumed-role/ops/x"
    # Identity is resolved once per process, not per checkpoint.
    HISTORY.record_release_history(release, phase="schema-ready", state_text="{}")
    assert len(resolutions) == 1, "STS must be asked once per process"
    assert "@" in resolutions[0], "the fallback offered must be user@host"


def test_history_is_bounded_and_keeps_the_newest_entries() -> None:
    existing = [
        {"release_id": f"rel-{index}", "phase": "complete"}
        for index in range(HISTORY.HISTORY_MAX_ENTRIES + 5)
    ]
    runner = _Runner(existing=existing)

    HISTORY.record_release_history(_release(runner), phase="complete", state_text="{}")

    entries = _applied_entries(runner)
    assert len(entries) == HISTORY.HISTORY_MAX_ENTRIES
    assert entries[-1]["release_id"] == "rel-42", "the newest entry must survive"
    assert entries[0]["release_id"] == f"rel-{6}", "the oldest entries are dropped"


def test_history_never_carries_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "rollout_regional_release.py",
            "join-cluster",
            "--cluster-id",
            "gpu-a",
            f"--token={TOKEN}",
            "--execution-token",
            TOKEN,
            f"--config=/secure/r.json?token={TOKEN}",
        ],
    )
    runner = _Runner(existing=None)

    HISTORY.record_release_history(_release(runner), phase="join", state_text="{}")

    entry = _applied_entries(runner)[-1]
    assert TOKEN not in json.dumps(entry), "a credential leaked into the history"
    assert "join-cluster --cluster-id gpu-a" in entry["command"]
    assert "<redacted>" in entry["command"]


def test_history_write_failure_does_not_fail_the_release(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Broken(_Runner):
        def run(self, arguments, **kwargs):
            if arguments[-3:] == ["apply", "-f", "-"]:
                raise RuntimeError("apiserver unavailable")
            return super().run(arguments, **kwargs)

    HISTORY.record_release_history(
        _release(Broken(existing=None)), phase="verified", state_text="{}"
    )

    assert "release history" in capsys.readouterr().err, (
        "a skipped history write must be announced, not swallowed"
    )


def test_dry_run_records_nothing() -> None:
    runner = _Runner(existing=None)
    runner.dry_run = True

    HISTORY.record_release_history(_release(runner), phase="plan", state_text="{}")

    assert runner.calls == [], "a dry run is not a transition"


def test_save_state_records_history(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        STATE,
        "record_release_history",
        lambda release, *, phase, state_text: recorded.append((phase, state_text)),
    )
    monkeypatch.setattr(STATE, "narrate_phase", lambda _release, _phase: None)
    rendered = ("{}", '{"phase":"verified"}')
    monkeypatch.setattr(STATE, "render_persisted_state", lambda _release: rendered)
    release = SimpleNamespace(
        state={},
        release_id="rel-42",
        runner=SimpleNamespace(dry_run=True),
        config=SimpleNamespace(namespace="gpu-fault-system"),
    )
    monkeypatch.setattr(
        STATE, "_write_state", lambda release, phase, **updates: rendered[1]
    )

    STATE.save_state(release, "verified")

    assert recorded == [("verified", '{"phase":"verified"}')], (
        "every checkpoint must leave a history entry"
    )


def _snapshot_release(items: list[dict], *, current: str) -> SimpleNamespace:
    calls: list[list[str]] = []
    release = SimpleNamespace(
        state={"previous_snapshot": {"chunks": [{"config_map": current}]}},
        runner=SimpleNamespace(
            dry_run=False,
            run=lambda arguments, **_kwargs: calls.append(arguments) or "",
        ),
        config=SimpleNamespace(namespace="gpu-fault-system"),
        _cpu=lambda *args: list(args),
        _get_json=lambda _arguments: {"items": items},
    )
    release.calls = calls
    return release


def _chunk(digest: str, index: int, created: str) -> dict:
    return {
        "metadata": {
            "name": f"gpu-fault-release-previous-{digest}-{index:03d}",
            "creationTimestamp": created,
            "annotations": {STATE.PREVIOUS_SNAPSHOT_DIGEST_ANNOTATION: digest * 3},
        }
    }


def test_snapshot_cleanup_retains_the_newest_previous_snapshots() -> None:
    items = [
        _chunk("current0000000000000", 0, "2026-09-07T10:00:00Z"),
        _chunk("recent00000000000000", 0, "2026-09-06T10:00:00Z"),
        _chunk("recent00000000000000", 1, "2026-09-06T10:00:00Z"),
        _chunk("older000000000000000", 0, "2026-09-05T10:00:00Z"),
        _chunk("oldest00000000000000", 0, "2026-09-01T10:00:00Z"),
    ]
    release = _snapshot_release(
        items, current="gpu-fault-release-previous-current0000000000000-000"
    )

    STATE.cleanup_previous_snapshots(release)

    deleted = [call for call in release.calls if "delete" in call]
    assert len(deleted) == 1, "one delete for the snapshots past the retention"
    assert deleted[0][-1:] == ["gpu-fault-release-previous-oldest00000000000000-000"], (
        f"only the oldest snapshot beyond {STATE.PREVIOUS_SNAPSHOTS_RETAINED} "
        "retained ones may be deleted"
    )


def test_snapshot_cleanup_never_deletes_the_current_reference() -> None:
    items = [
        _chunk("current0000000000000", 0, "2026-09-01T10:00:00Z"),
        _chunk("newer000000000000000", 0, "2026-09-05T10:00:00Z"),
        _chunk("newer100000000000000", 0, "2026-09-06T10:00:00Z"),
        _chunk("newer200000000000000", 0, "2026-09-07T10:00:00Z"),
    ]
    release = _snapshot_release(
        items, current="gpu-fault-release-previous-current0000000000000-000"
    )

    STATE.cleanup_previous_snapshots(release)

    for call in release.calls:
        assert "gpu-fault-release-previous-current0000000000000-000" not in call, (
            "the snapshot the live state references is never garbage"
        )
