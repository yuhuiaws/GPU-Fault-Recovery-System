"""The host fixture's ``error`` key: its presence is the failed call, whatever the value.

Live (plain EKS lane eksb, release R13g, harness live17) GF-GENERAL-LIVE-005
``capacity_full`` reached its fill for the first time and ended INCONCLUSIVE on
``harness fixture failed: HostProbeError: host probe failed (exit 0); output
withheld`` -- no fill evidence, no probe-error.json, no token -- while the
cleanup's unfill removed one file from an export that read 0 % used afterwards.
The shared probe's fill snippet printed ``"error": null`` on success, ``nfs_fill``
spread that reading into its result, and ``HostProbeFixture.execute`` classifies
a result by the presence of an ``error`` key (``"error" in payload``), then shows
only whitelisted tokens; ``null`` matched none. The fixture keeps presence as the
signal -- a probe's failure line whose error is null or empty must never be
graded as a success payload -- and now says ``error key null``; the probes keep
the key out of success results. These tests pin both halves against
representative payloads, the shared probe's own failure line (reason token,
message and reading in the case directory), and its real fill and unfill
readings through the real fixture.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import host_probe_fixture as host
from scripts.e2e.regional.probes import general_live_shared_probe as sharedp
from tests.regional._host_probe_support import ProbeApi, host_probe


def _execute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: dict[str, Any],
    returncode: int = 0,
) -> dict[str, Any]:
    api = ProbeApi()
    monkeypatch.setattr(host, "run_fixture_command", api.run)
    fixture = host_probe(tmp_path)
    fixture.create()
    api.probe_returncode = returncode
    api.probe_stdout = json.dumps(payload) + "\n"
    return fixture.execute("nfs-fill", "--name", "shared", "--percent", "99.0")


def _failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: dict[str, Any],
    returncode: int = 0,
) -> str:
    with pytest.raises(host.HostProbeError) as caught:
        _execute(monkeypatch, tmp_path, payload, returncode)
    return str(caught.value)


def test_a_result_without_an_error_key_is_the_call_s_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = {
        "name": "shared",
        "fill": {"fill_bytes": 253 << 20, "wanted_bytes": 253 << 20, "enospc": False},
        "statvfs": {"used_percent": 99.2},
    }
    assert _execute(monkeypatch, tmp_path, payload) == payload
    assert not (tmp_path / host.PROBE_ERROR_RECORD).exists(), "no failure was recorded"


@pytest.mark.parametrize(
    ("error", "shown"),
    [
        (None, "; error key null"),
        ("", ""),
        (False, ""),
        ("OSError", "; error OSError"),
        ("OSError: 28", ""),
        ("reading failed (1)", ""),
    ],
    ids=["null", "empty", "false", "class", "class-and-errno", "sentence"],
)
def test_an_error_key_is_a_failed_call_whatever_its_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: Any, shown: str
) -> None:
    """Presence, not truthiness: ``null``, ``""`` and ``false`` fail the call too
    (a probe whose failure line carried one of them must not be graded as a
    success payload). Only an identifier-shaped value is shown; ``null`` says so
    in a constant token, anything wordier stays withheld."""

    payload = {"error": error, "name": "shared", "statvfs": {"used_percent": 99.2}}
    text = _failure(monkeypatch, tmp_path, payload)
    assert text == f"host probe failed (exit 0{shown}); output withheld", text
    assert not (tmp_path / host.PROBE_ERROR_RECORD).exists(), (
        "nothing wordier than a class name was said"
    )


def test_a_failed_exit_without_an_error_key_is_not_called_null(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The null marker is about the key's value, never about its absence."""

    text = _failure(monkeypatch, tmp_path, {"status": "refused"}, returncode=1)
    assert text == "host probe failed (exit 1); output withheld", text


def test_the_shared_probe_s_failure_line_names_its_reason_and_keeps_the_reading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reading = {"timed_out": True, "elapsed": 180, "error": "TimeoutExpired"}
    with pytest.raises(sharedp.ProbeError) as refused:
        sharedp.checked_reading(reading, "fill")
    line = sharedp.error_fields(refused.value, "nfs-fill")
    text = _failure(monkeypatch, tmp_path, line, returncode=1)
    digest = sharedp.error_code(refused.value)
    assert re.fullmatch(
        rf"host probe failed \(exit 1; error ProbeError {digest} at "
        r"checked_reading:[0-9]+; reason fill_timeout; detail in probe-error\.json\); "
        r"output withheld",
        text,
    ), text
    record = json.loads((tmp_path / host.PROBE_ERROR_RECORD).read_text())
    assert record["error_reason"] == "fill_timeout" and record["command"] == "nfs-fill"
    assert record["message"] == "the fill did not finish within 180 s"
    assert record["detail"] == reading and record["redacted"] is True


def test_the_shared_probe_s_closed_reason_vocabulary_fits_the_fixture_s_shape() -> None:
    """The tokens ``ProbeError``'s docstring lists are the ones ``checked_reading``
    and ``nfs_unfill`` raise, and every one fits the fixture's token shape."""

    tokens = set(re.findall(r"``([a-z]+_[a-z_]+)``", sharedp.ProbeError.__doc__ or ""))
    raised = set()
    for what in ("fill", "unfill"):
        for reading in ({"timed_out": True}, {"error": "reading failed (1)"}):
            with pytest.raises(sharedp.ProbeError) as refused:
                sharedp.checked_reading(reading, what)
            raised.add(refused.value.reason)
    raised.add("unfill_failed")  # nfs_unfill, on a removal the snippet reported
    assert tokens == raised, (tokens, raised)
    for token in tokens:
        assert re.fullmatch(r"[a-z][a-z0-9_]{2,40}", token), token


def _reading(code: str, root: Path, *arguments: str) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(root), *arguments],
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    value = json.loads(completed.stdout.strip().splitlines()[-1])
    assert isinstance(value, dict), value
    return value


def test_the_real_fill_and_unfill_readings_pass_the_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The snippets run for real (a target below the current usage writes
    nothing; the shape is what counts) and their results, spread the way
    ``nfs_fill`` / ``nfs_unfill`` emit them, are the call's payload."""

    export = tmp_path / "export"
    export.mkdir()
    for name, code, arguments in (
        ("fill", sharedp.FILL_SNIPPET, ("0.0000001",)),
        ("unfill", sharedp.UNFILL_SNIPPET, ()),
    ):
        reading = sharedp.checked_reading(_reading(code, export, *arguments), name)
        payload = {"name": "shared", "mountpoint": str(export), **reading}
        assert "error" not in payload and name in payload, payload
        case = tmp_path / name
        case.mkdir()
        assert _execute(monkeypatch, case, payload) == payload
