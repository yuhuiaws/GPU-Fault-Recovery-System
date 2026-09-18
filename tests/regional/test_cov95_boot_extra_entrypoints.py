from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.e2e.regional import run_boot_acceptance as boot
from tests.regional._cov95_boot_extra_entry import BootEntry
from tests.regional._cov95_boot_extra_safety import (
    boot_extra_isolation as boot_extra_isolation,
)
from tests.regional.test_boot_acceptance_behavior import RuntimeFixture


@pytest.mark.parametrize("case", ["012", "013", "014", "021"])
def test_entrypoint_runs_the_selected_readonly_case_against_recording_transports(
    tmp_path, monkeypatch, capsys, case
):
    entry = BootEntry(tmp_path, monkeypatch, case=case)
    fixture = RuntimeFixture()
    entry.fixture = fixture
    result = boot.main()
    assert result == 0, "the valid offline probe observations should pass"
    document = json.loads(entry.path.read_text(encoding="utf-8"))
    assert document["case_id"] == entry.arguments.case
    assert document["verdict"] == "PASS"
    assert document["cluster_id"] == fixture.cluster_id
    assert document["release_id"] == "release-a"
    assert all(document["checks"].values()), "a required replica check did not pass"
    assert document["predecessor"] == entry.prerequisite
    assert entry.path.stat().st_mode & 0o777 == 0o600
    assert json.loads(capsys.readouterr().out)["case_id"] == entry.arguments.case
    authorization = [call for call in entry.calls if call[0] == "authorize"]
    assert len(authorization) == 1
    assert authorization[0][2]["confirmation"] == f"BOOT{case}_EXECUTE"


@pytest.mark.parametrize("case", ["011", "015", "016", "017", "018"])
@pytest.mark.parametrize("interruption", [False, True])
def test_entrypoint_persists_fail_before_dispatch_and_records_failed_or_aborted_body(
    tmp_path, monkeypatch, case, interruption
):
    entry = BootEntry(tmp_path, monkeypatch, case=case)
    invocations = []
    error = (
        KeyboardInterrupt("unit operator abort")
        if interruption
        else RuntimeError("unit transport failure")
    )

    def execute(*args, **kwargs):
        invocations.append((args, kwargs))
        before = json.loads(entry.path.read_text(encoding="utf-8"))
        assert before["verdict"] == "FAIL", "stale PASS survived entry to the case body"
        assert "executed_at" not in before
        assert any(call[0] == "authorize" for call in entry.calls), (
            "the case body ran before authorization"
        )
        raise error

    monkeypatch.setattr(boot, f"run_boot{case}", execute)
    if interruption:
        with pytest.raises(KeyboardInterrupt) as caught:
            boot.main()
        assert caught.value is error
    else:
        assert boot.main() == 1, "an ordinary probe failure must return nonzero"
    document = json.loads(entry.path.read_text(encoding="utf-8"))
    assert document["verdict"] == "FAIL"
    assert document["interrupted"] is interruption
    assert document["release_id"] == "release-unit"
    assert "executed_at" in document
    assert document["error"] == f"{type(error).__name__}: {error}"
    assert document["traceback"], "failure evidence lost the failing call frames"
    assert len(invocations) == 1
    args, kwargs = invocations[0]
    case_dir = entry.path.parent
    if case == "011":
        assert args == (entry.fixture,)
        assert kwargs == {"production_site": entry.arguments.production_site}
    elif case == "015":
        assert args == (entry.fixture,)
        assert kwargs == {"case_dir": case_dir, "attempt": entry.arguments.attempt}
    elif case == "017":
        assert args == (entry.arguments, entry.prerequisite, case_dir)
    else:
        assert args == (entry.arguments, case_dir)


@pytest.mark.parametrize(
    ("case", "field", "value", "message"),
    [
        ("013", "site", None, "requires --site"),
        ("011", "production_site", None, "requires --production-site"),
        ("017", "bootstrap_state_dir", None, "requires --bootstrap-state-dir"),
        ("016", "cpu_cluster_arn", "", "requires CPU/GPU ARNs"),
        ("016", "gpu_cluster_arn", [], "requires CPU/GPU ARNs"),
        ("016", "admin_email", "", "requires CPU/GPU ARNs"),
    ],
)
def test_entrypoint_rejects_missing_inputs_before_reading_identity_or_authorizing(
    tmp_path, monkeypatch, case, field, value, message
):
    entry = BootEntry(tmp_path, monkeypatch, case=case)
    setattr(entry.arguments, field, value)
    with pytest.raises(boot.BootAcceptanceError, match=message):
        boot.main()
    assert entry.calls == [("site-profile",)]
    assert not entry.path.exists(), "invalid input wrote execution evidence"


@pytest.mark.parametrize("fault", ["confirmation", "false-proof", "truthy-proof"])
def test_entrypoint_cannot_dispatch_without_exact_confirmation_and_passed_predecessor(
    tmp_path, monkeypatch, fault
):
    entry = BootEntry(tmp_path, monkeypatch)
    if fault == "confirmation":
        entry.arguments.confirm = "BOOT012_EXECUTE"
    else:
        entry.prerequisite["valid"] = False if fault == "false-proof" else "true"
    monkeypatch.setattr(
        boot,
        "run_boot013",
        lambda *_args: pytest.fail("an unauthorized body was invoked"),
    )
    with pytest.raises(boot.BootAcceptanceError, match="confirmation|predecessor"):
        boot.main()
    authorizations = [call for call in entry.calls if call[0] == "authorize"]
    assert len(authorizations) == (0 if fault == "confirmation" else 1)
    assert not entry.path.exists(), "a failed guard wrote case execution evidence"


def test_identity_without_any_site_is_allowed_only_for_first_boot016(tmp_path):
    arguments = SimpleNamespace(
        case="GF-REGIONAL-BOOT-016", site=None, bootstrap_state_dir=None, cluster_id=""
    )
    assert boot.evidence_identity_for(arguments) is None
    arguments.case = "GF-REGIONAL-BOOT-018"
    with pytest.raises(boot.BootAcceptanceError, match="without the target site"):
        boot.evidence_identity_for(arguments)
