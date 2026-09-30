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
