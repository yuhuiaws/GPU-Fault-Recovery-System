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
