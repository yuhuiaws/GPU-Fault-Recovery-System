from __future__ import annotations

import pytest

from gpu_fault.admin.command_log import ADMIN_LOG_ENVIRONMENT
from gpu_fault.admin.status_report import print_status_report


@pytest.mark.parametrize("stdout", [None, ""])
def test_empty_status_has_a_readable_error_without_inventing_json(
    stdout: str | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(ADMIN_LOG_ENVIRONMENT, raising=False)
    print_status_report(stdout)
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == (
        "gpu-fault-admin status: status printed no JSON report; see the output below\n"
    )
