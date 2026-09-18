from __future__ import annotations

import errno
import os
from types import SimpleNamespace

import pytest

from tools import run_regional_acceptance as runner


def test_report_permission_failure_closes_the_temporary_descriptor(
    tmp_path, monkeypatch
) -> None:
    descriptors = []
    mkstemp = runner.tempfile.mkstemp

    def allocate(**kwargs):
        descriptor, name = mkstemp(**kwargs)
        descriptors.append(descriptor)
        return descriptor, name

    def refuse_permission(_descriptor, _mode):
        raise OSError(errno.EPERM, "fixture permission change denied")

    monkeypatch.setattr(runner, "tempfile", SimpleNamespace(mkstemp=allocate))
    monkeypatch.setattr(
        runner,
        "os",
        SimpleNamespace(
            fchmod=refuse_permission,
            fdopen=os.fdopen,
            fsync=os.fsync,
            replace=os.replace,
        ),
    )
    with pytest.raises(OSError, match="permission change denied"):
        runner.secure_write_json(tmp_path / "report.json", {"status": "FAIL"})
    assert len(descriptors) == 1, "the write must allocate exactly one temporary file"
    try:
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(descriptors[0])
    finally:
        try:
            os.close(descriptors[0])
        except OSError as error:
            assert error.errno == errno.EBADF, (
                "only an already closed descriptor is safe"
            )
    assert list(tmp_path.iterdir()) == [], (
        "failed reports must not leave partial output"
    )
