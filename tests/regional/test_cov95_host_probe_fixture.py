"""Host probe fixture: probe-error records, failure diagnostics and lost deletions."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import host_probe_fixture as module
from tests.regional._host_probe_support import ProbeApi, host_probe


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> ProbeApi:
    value = ProbeApi()
    monkeypatch.setattr(module, "run_fixture_command", value.run)
    return value


def test_earlier_probe_errors_ignore_a_foreign_file(tmp_path: Path) -> None:
    path = tmp_path / module.PROBE_ERROR_RECORD
    path.write_text(json.dumps(["not", "a", "record"]), encoding="utf-8")

    assert module.earlier_probe_errors(path) == []


def test_earlier_probe_errors_keep_the_chain_newest_first_and_bounded(
    tmp_path: Path,
) -> None:
    path = tmp_path / module.PROBE_ERROR_RECORD
    older = [{"message": f"call {index}"} for index in range(12)]
    path.write_text(
        json.dumps({"message": "newest", "previous": [older[0], "junk", *older[1:]]}),
        encoding="utf-8",
    )

    chain = module.earlier_probe_errors(path)

    assert [item["message"] for item in chain] == [
        "newest",
        *[f"call {index}" for index in range(module.PROBE_ERROR_HISTORY - 1)],
    ]


def test_a_power_identity_refusal_without_digest_or_site_names_what_is_left() -> None:
    diagnostic, code, site, _reason = module.failure_diagnostic(
        {
            "error": "PowerRecoveryIdentityError",
            "error_kind": "power_identity",
            "error_code": "not-a-digest",
            "error_site": "/abs/path.py:12",
            "left_behind": ["load_unit", "owner_record"],
        }
    )

    assert (code, site) == (None, None)
    assert "; left behind: load_unit,owner_record" in diagnostic
    assert "power identity" not in diagnostic, diagnostic
    assert " at " not in diagnostic, diagnostic


def failing_call(api: ProbeApi, detail: Any) -> None:
    api.probe_returncode = 1
    api.probe_stdout = (
        json.dumps(
            {
                "error": "ValueError",
                "error_message": "the read-back differs",
                "error_detail": detail,
                "command": "snapshot",
            }
        )
        + "\n"
    )


def test_an_oversized_failure_detail_is_recorded_as_truncated(
    tmp_path: Path, api: ProbeApi
) -> None:
    probe = host_probe(tmp_path)
    probe.create()
    failing_call(api, {"checks": ["gpu 0 field differs " * 45] * 200})

    with pytest.raises(module.HostProbeError) as caught:
        probe.execute("snapshot")

    assert f"detail in {module.PROBE_ERROR_RECORD}" in str(caught.value)
    record = json.loads((tmp_path / module.PROBE_ERROR_RECORD).read_text())
    assert record["detail"] == {"truncated": True}
    assert record["message"] == "the read-back differs"
    assert record["returncode"] == 1


def test_a_failure_record_that_cannot_be_written_never_hides_the_failure(
    tmp_path: Path, api: ProbeApi
) -> None:
    probe = host_probe(tmp_path)
    probe.create()
    (tmp_path / module.PROBE_ERROR_RECORD).mkdir()
    failing_call(api, {"checks": ["mismatch"]})

    with pytest.raises(module.HostProbeError) as caught:
        probe.execute("snapshot")

    message = str(caught.value)
    assert message.startswith("host probe failed (exit 1"), message
    assert "detail in" not in message, message
    assert message.endswith("output withheld"), message


class UnacknowledgedDelete(ProbeApi):
    """The API refuses the Pod deletion and keeps the Pod."""

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        arguments = command[7:]
        if arguments[0] == "delete" and "/pods/" in arguments[2]:
            self.calls.append((arguments, kwargs))
            return subprocess.CompletedProcess(command, 1, "", "etcdserver: timeout")
        return super().run(command, **kwargs)


def test_a_deletion_the_api_did_not_acknowledge_fails_the_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = UnacknowledgedDelete()
    monkeypatch.setattr(module, "run_fixture_command", api.run)
    probe = host_probe(tmp_path)
    probe.create()

    with pytest.raises(
        module.HostProbeError, match="owned host probe deletion was not acknowledged"
    ):
        probe.cleanup()

    assert "pod" in api.objects, "the Pod the API kept is still owned and recorded"
    assert json.loads(probe.state_path.read_text())["closed"] is False
