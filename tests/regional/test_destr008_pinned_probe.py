"""The independent loader executes captured bytes, not a changed script path."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.e2e.regional.destr008_admission import SOURCE_LOADER
from scripts.e2e.regional.probes import destr008_admission_probe as probe


def invoke(source: str, expected: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            SOURCE_LOADER,
            expected,
            "--fixture",
            "owned",
        ],
        input=source,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
        env={
            "PATH": os.defpath,
            "HOME": "/tmp",
            "AWS_CONFIG_FILE": "/dev/null",
            "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
            "AWS_EC2_METADATA_DISABLED": "true",
            "KUBECONFIG": "/dev/null",
        },
    )


def test_loader_binds_actual_captured_bytes_and_preserves_arguments() -> None:
    source = "import json,sys\nprint(json.dumps({'sha':_SOURCE_SHA256,'args':sys.argv[1:]}))\n"
    digest = hashlib.sha256(source.encode()).hexdigest()
    result = invoke(source, digest)
    assert result.returncode == 0, "a matching immutable source must execute"
    assert json.loads(result.stdout) == {
        "sha": digest,
        "args": ["--fixture", "owned"],
    }, "the child must bind the bytes it actually compiled"


@pytest.mark.parametrize("oversize", [False, True])
def test_unapproved_or_oversized_source_cannot_execute(
    tmp_path: Path, oversize: bool
) -> None:
    marker = tmp_path / "should-not-exist"
    source = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    if oversize:
        source += " " * 65537
    expected = hashlib.sha256(source.encode()).hexdigest() if oversize else "0" * 64
    result = invoke(source, expected)
    assert result.returncode != 0, "source mismatch must fail before execution"
    assert not marker.exists(), "rejected source must perform no local side effect"


@pytest.mark.parametrize("pinned", ["", "short", 1, "z" * 64])
def test_probe_rejects_invalid_loader_identity(
    monkeypatch: pytest.MonkeyPatch, pinned: object
) -> None:
    monkeypatch.setattr(probe, "_SOURCE_SHA256", pinned, raising=False)
    with pytest.raises(probe.AdmissionProbeError, match="pinned admission"):
        probe.source_identity()


def test_probe_reports_the_verified_loader_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(probe, "_SOURCE_SHA256", "a" * 64, raising=False)
    assert probe.source_identity() == "a" * 64, (
        "the loader's verified hash is the source identity"
    )
