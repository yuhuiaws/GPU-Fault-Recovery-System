"""Guard diagnostics expose only stable identifiers, never raw failure output."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from scripts.e2e.regional import host_probe_fixture as host
from scripts.e2e.regional.probes import collector_node_probe as probe
from tests.regional._host_probe_support import ProbeApi, host_probe


def test_node_guard_exposes_a_code_and_own_source_location(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe.py",
            "restore-collector-env",
            "--run-id",
            "fixture",
            "--owner-nonce",
            "a" * 32,
        ],
    )
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    assert probe.main() == 78
    value = json.loads(capsys.readouterr().out)
    expected = "collector env recovery requires root"
    assert value["error_code"] == hashlib.sha256(expected.encode()).hexdigest()[:16]
    assert value["error_site"].startswith("collector_require_root:"), (
        'test_node_guard_exposes_a_code_and_own_source_location: expected value["error_site"].startswith("collector_require_root:")'
    )
    assert "a" * 32 not in json.dumps(value)


@pytest.mark.parametrize("valid", [False, True])
def test_host_guard_diagnostic_is_whitelisted_and_does_not_expose_error_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, valid: bool
) -> None:
    api = ProbeApi()
    monkeypatch.setattr(host, "run_fixture_command", api.run)
    fixture = host_probe(tmp_path)
    fixture.create()
    api.probe_returncode = 78
    api.probe_stdout = json.dumps(
        {
            "error_kind": "collector_env_guard",
            "error_code": "f" * 16 if valid else "private-diagnostic",
            "error_site": "collector_health:77" if valid else "private-diagnostic",
            "error": "private-error-detail",
        }
    )
    with pytest.raises(host.HostProbeError) as caught:
        fixture.execute("restore-collector-env")
    assert "private" not in str(caught.value)
    assert ("f" * 16 in str(caught.value)) is valid
    assert ("collector_health:77" in str(caught.value)) is valid
    assert not isinstance(caught.value, host.HostProbeMissingResponseError), (
        "test_host_guard_diagnostic_is_whitelisted_and_does_not_expose_error_message: expected no isinstance(caught.value, host.HostProbeMissingRe..."
    )
    assert not any(fixture.cleanup().values()), (
        "test_host_guard_diagnostic_is_whitelisted_and_does_not_expose_error_message: expected no any(fixture.cleanup().values())"
    )


@pytest.mark.parametrize("valid", [False, True])
def test_power_identity_failure_names_the_kinds_of_files_left_behind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, valid: bool
) -> None:
    """The restore after a reboot used to read "exit 78; output withheld"; the
    operator now sees the class, digest, site and which kinds of files the
    operation left on the node -- identifiers only, never a path or message."""

    api = ProbeApi()
    monkeypatch.setattr(host, "run_fixture_command", api.run)
    fixture = host_probe(tmp_path)
    fixture.create()
    api.probe_returncode = 78
    api.probe_stdout = json.dumps(
        {
            "error_kind": "power_identity",
            "error_class": "PowerRecoveryIdentityError",
            "error_code": "e" * 16,
            "error_site": "power_load_record:826",
            "error": "private-error-detail /var/lib/private",
            "left_behind": ["owner", "record"] if valid else ["/var/lib/private"],
            "manual_confirmation_required": True,
        }
    )
    with pytest.raises(host.HostProbeError) as caught:
        fixture.execute("restore-gpu-power-limit", "--run-id", "unit")
    message = str(caught.value)
    assert "private" not in message, message
    assert "; power identity " + "e" * 16 + " at power_load_record:826" in message
    assert "; error PowerRecoveryIdentityError" in message, message
    assert ("; left behind: owner,record" in message) is valid, message
    assert message.endswith("; output withheld"), message


@pytest.mark.parametrize(
    ("error", "exposed"),
    [
        ("RecoveryError", True),
        ("FileNotFoundError", True),
        ("Recovery failed at /var/lib/private", False),
        ("private-error-detail", False),
    ],
    ids=["type", "builtin-type", "sentence", "hyphenated"],
)
def test_host_failure_names_the_sanitized_error_type_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: str, exposed: bool
) -> None:
    """A probe that fails prints ``{"error": <ExceptionType>}``; the fixture
    withholds the rest of the output, so the type name is the one clue an
    operator gets (DESTR-014 a8 spent a diagnosis on a bare "exit 1"). Only an
    identifier passes: a sentence or a path-bearing message stays withheld."""

    api = ProbeApi()
    monkeypatch.setattr(host, "run_fixture_command", api.run)
    fixture = host_probe(tmp_path)
    fixture.create()
    api.probe_returncode = 1
    api.probe_stdout = json.dumps({"error": error, "recovery_required": True})
    with pytest.raises(host.HostProbeError) as caught:
        fixture.execute("status")
    message = str(caught.value)
    assert message.endswith("; output withheld"), message
    assert (f"; error {error})" in message) is exposed, message
    if not exposed:
        assert error not in message, message
    assert "recovery_required" not in message, message


@pytest.mark.parametrize(
    ("message", "exposed"),
    [
        ("Node Agent target or release pins changed", True),
        ("recovery helper interpreter cannot run the helper", True),
        ("token=abc", False),
        ("two\nlines", False),
        ("x" * 161, False),
    ],
)
def test_recovery_refusal_message_is_surfaced_only_in_its_constant_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, message: str, exposed: bool
) -> None:
    """The DESTR-014 recovery probe's refusals are constant sentences
    (``error_kind: recovery``); live eks attempt 13 stopped on a bare
    "error RecoveryError" because the sentence stayed withheld. Only the
    constant shape passes: anything with '=', a newline or beyond the length
    bound is still withheld."""

    api = ProbeApi()
    monkeypatch.setattr(host, "run_fixture_command", api.run)
    fixture = host_probe(tmp_path)
    fixture.create()
    api.probe_returncode = 1
    api.probe_stdout = json.dumps(
        {
            "error": "RecoveryError",
            "error_kind": "recovery",
            "message": message,
            "recovery_required": True,
        }
    )
    with pytest.raises(host.HostProbeError) as caught:
        fixture.execute("prepare")
    text = str(caught.value)
    assert "; error RecoveryError" in text and text.endswith("; output withheld"), text
    assert (f"; recovery refused: {message}" in text) is exposed, text
    if not exposed:
        assert "recovery refused" not in text and message[:5] not in text, text


@pytest.mark.parametrize(
    "problem", [None, "class", "code", "site"], ids=["valid", "class", "code", "site"]
)
def test_host_failure_surfaces_sanitized_probe_class_code_and_site(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, problem: str | None
) -> None:
    """A probe's generic failure line carries ``error_kind: probe`` plus the
    exception class, a digest of its message and the probe's own source site.
    The fixture shows exactly those three (each whitelisted); the message itself
    (``error``) never leaves the node, so a wordy ``error`` no longer collapses
    the record to a bare "exit 1" (COLLECT-005, 2026-09-21)."""

    api = ProbeApi()
    monkeypatch.setattr(host, "run_fixture_command", api.run)
    fixture = host_probe(tmp_path)
    fixture.create()
    api.probe_returncode = 1
    payload = {
        "error": "ProbeError: FM journal window is not in journal order /private",
        "error_kind": "probe",
        "error_class": "ProbeError",
        "error_code": "a" * 16,
        "error_site": "fm_delivery_evidence:1509",
    }
    if problem is not None:
        payload[f"error_{problem}"] = "private-diagnostic detail"
    api.probe_stdout = json.dumps(payload)
    with pytest.raises(host.HostProbeError) as caught:
        fixture.execute("fm-delivery-evidence")
    message = str(caught.value)
    assert message.endswith("; output withheld"), message
    assert "private" not in message and "journal order" not in message, message
    if problem is None:
        expected = f"; error ProbeError {'a' * 16} at fm_delivery_evidence:1509)"
        assert expected in message, message
    elif problem == "class":
        assert "; error" not in message, message
    else:
        assert "; error ProbeError" in message, message
        assert ("a" * 16 in message) is (problem != "code"), message
        assert ("fm_delivery_evidence:1509" in message) is (problem != "site"), message
    assert not isinstance(caught.value, host.HostProbeMissingResponseError), message
    assert not any(fixture.cleanup().values()), "cleanup must still remove the probe"


@pytest.mark.parametrize(
    ("message", "exposed"),
    [
        (
            "export server answered ENOENT for the export path (export not initialised or hidden)",
            True,
        ),
        ("TCP/2049 connection to the export server timed out", True),
        ("mount.nfs4: mounting 10.0.9.7:/a failed=1", False),
        ("two\nlines", False),
        ("x" * 161, False),
    ],
)
def test_nfs_mount_sentence_is_surfaced_only_in_its_constant_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, message: str, exposed: bool
) -> None:
    """The shared-storage probe classifies a failed ``mount -t nfs4`` into a
    constant sentence (``error_kind: nfs_mount``); live GF-GENERAL-LIVE-005
    attempt 5 stopped six variants on a bare digest because the mount helper's
    reason stayed withheld. Same shape rule as the recovery refusals."""

    api = ProbeApi()
    monkeypatch.setattr(host, "run_fixture_command", api.run)
    fixture = host_probe(tmp_path)
    fixture.create()
    api.probe_returncode = 1
    api.probe_stdout = json.dumps(
        {
            "error": "MountError",
            "error_kind": "nfs_mount",
            "error_class": "MountError",
            "error_code": "0" * 16,
            "error_site": "nfs_mount:560",
            "message": message,
            "command": "nfs-mount",
        }
    )
    with pytest.raises(host.HostProbeError) as caught:
        fixture.execute("nfs-mount")
    text = str(caught.value)
    assert text.endswith("; output withheld"), text
    assert (f"; nfs mount failed: {message}" in text) is exposed, text
    if not exposed:
        assert "nfs mount failed" not in text and message[:5] not in text, text


def test_probe_message_and_detail_are_kept_redacted_in_the_case_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A probe's generic failure line now carries its message and a structured
    detail (the source-replay probe's verification checks). The exception text
    keeps its whitelisted shape and names the record; the record itself lands
    in the case directory (next to ``host-probes/``) with credential-shaped
    tokens and long opaque blobs masked, so an operator reads what the probe
    refused instead of brute-forcing the digest (two lanes did, 2026-10-26)."""

    api = ProbeApi()
    monkeypatch.setattr(host, "run_fixture_command", api.run)
    fixture = host_probe(tmp_path)
    fixture.create()
    api.probe_returncode = 1
    blob = "A" * 48
    api.probe_stdout = json.dumps(
        {
            "error": "ProbeError",
            "error_kind": "probe",
            "error_class": "ProbeError",
            "error_code": "b" * 16,
            "error_site": "open_window:1161",
            "error_message": (
                "the replacement is not visible only inside the unit; closed: "
                "token=abc123 Authorization: Bearer xyz"
            ),
            "error_detail": {
                "checks": [
                    {
                        "kind": "env",
                        "dest": "GPU_FAULT_EXPECTED_GPU_COUNT",
                        "unit_value": "8",
                        "ok": False,
                        "note": f"secret=hunter2 {blob}",
                    }
                ],
                "close": {"clean": True},
            },
            "command": "open-window",
        }
    )
    with pytest.raises(host.HostProbeError) as caught:
        fixture.execute("open-window")
    message = str(caught.value)
    assert message.endswith("; output withheld"), message
    assert f"; error ProbeError {'b' * 16} at open_window:1161" in message, message
    assert "; detail in probe-error.json" in message, message
    assert "not visible" not in message and "hunter2" not in message, message
    record = json.loads(
        (tmp_path / host.PROBE_ERROR_RECORD).read_text(encoding="utf-8")
    )
    assert record["error_class"] == "ProbeError" and record["error_code"] == "b" * 16
    assert (
        record["error_site"] == "open_window:1161"
        and record["command"] == "open-window"
    )
    assert record["returncode"] == 1 and record["redacted"] is True
    assert record["case_id"] == "GF-REGIONAL-DESTR-001" and record["run_id"] == "run-a"
    assert record["message"].startswith(
        "the replacement is not visible only inside the unit; closed: "
    ), record["message"]
    assert "abc123" not in record["message"] and "xyz" not in record["message"]
    assert "token=<redacted>" in record["message"], record["message"]
    check = record["detail"]["checks"][0]
    assert check["dest"] == "GPU_FAULT_EXPECTED_GPU_COUNT" and check["ok"] is False
    assert check["unit_value"] == "8", "short plain values stay readable"
    assert "hunter2" not in check["note"] and blob not in check["note"], check
    assert record["detail"]["close"] == {"clean": True}
    assert not any(fixture.cleanup().values()), "cleanup must still remove the probe"
    # A probe that said only its class name writes no record.
    (tmp_path / host.PROBE_ERROR_RECORD).unlink()
    fixture = host_probe(tmp_path, run_id="run-b")
    fixture.create()
    api.probe_stdout = json.dumps(
        {"error": "ProbeError", "error_kind": "probe", "error_class": "ProbeError"}
    )
    with pytest.raises(host.HostProbeError) as caught:
        fixture.execute("open-window")
    assert "probe-error.json" not in str(caught.value)
    assert not (tmp_path / host.PROBE_ERROR_RECORD).exists(), (
        "a bare class name is not worth a record"
    )


def test_redaction_masks_credential_shapes_and_bounds_the_record() -> None:
    text = "password: p4ss api_key=K1 bearer tok; path=/run/gfrs-live/source-abc/env ok"
    redacted = host.redact_probe_text(text)
    assert "p4ss" not in redacted and "K1" not in redacted and "tok" not in redacted
    assert "/run/gfrs-live/source-abc/env ok" in redacted, redacted
    assert host.redact_probe_text("x" * 5000) == "<redacted>", "an opaque blob"
    long_text = "word " * 1000
    assert (
        host.redact_probe_text(long_text) == long_text[: host.PROBE_ERROR_MESSAGE_CHARS]
    )
    nested: object = {"a": 1}
    for _ in range(12):
        nested = {"inner": nested}
    flattened = json.dumps(host.redact_probe_detail(nested))
    assert "<nested>" in flattened
    assert host.redact_probe_detail([True, None, 2.5, "ok"]) == [True, None, 2.5, "ok"]
    assert host.redact_probe_detail(object()).startswith("<object object"), (
        "an unknown type is kept as its bounded repr"
    )
