from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from gpu_fault.admin.operation_lock import SITE_OPERATION_LOCK_FD_ENV
from scripts import staging_deploy, staging_live_evidence
from tests._staging_deploy_support import (
    DEPLOY_EVENT,
    RECORD_EVIDENCE_EVENT,
    STUB_LOCK_FD,
    deploy_arguments,
    install_admin_stub,
    live_evidence,
    managed_state,
    signing_material,
    source_identities,
    status_report,
    stub_deploy_orchestration,
)


@pytest.mark.parametrize("old_mode", ["UNCHANGED", "DEPLOY_HOST_ONLY", "QUALITY_ONLY"])
@pytest.mark.parametrize(
    ("repair", "phase"),
    [
        ("observability_drift", "complete"),
        ("aurora_refresh_drift", "complete"),
        ("", "failed"),
        ("", "cpu-staged"),
        ("", "rolled-back"),
        ("observability_drift", "failed"),
    ],
)
def test_repair_and_unfinished_release_reach_admin_deploy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    old_mode: str,
    repair: str,
    phase: str,
) -> None:
    repository, state = managed_state(tmp_path)
    install_admin_stub(state)
    signing = signing_material(state)
    source = staging_deploy.SourceCheckout(
        repository_root=repository,
        git_commit="a" * 40,
        fingerprint="a" * 64,
        snapshot=False,
        isolated=True,
    )
    previous = {
        "identities": source_identities(),
        "source": {"fingerprint": source.fingerprint, "git_commit": source.git_commit},
        "live": live_evidence(),
    }
    if old_mode == "DEPLOY_HOST_ONLY":
        previous["identities"]["deploy_host"]["sha256"] = "different"
    elif old_mode == "QUALITY_ONLY":
        previous["source"]["fingerprint"] = "different"
    assert (
        staging_deploy.classify_source_deploy(
            previous,
            source_identities(),
            source=source,
            site_exists=True,
            live_matches=True,
        )
        == old_mode
    )

    report = status_report()
    report["healthy"] = False
    report["live_release"].update(
        phase=phase, transaction_committed=phase == "complete"
    )
    if repair:
        report["next_deploy"] = {"kind": "CONTROL_PLANE_ONLY", "changed": [repair]}
    events: list[str] = []
    stub_deploy_orchestration(
        monkeypatch,
        state=state,
        source=source,
        signing=signing,
        previous=previous,
        events=events,
    )
    status_event = f"status(lock_fd={STUB_LOCK_FD})"

    def status(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert command == [
            str(state / "deployer-venv/bin/gpu-fault-admin"),
            "status",
            "--state-dir",
            str(state),
        ], "the mocked deploy attempted an unexpected subprocess"
        assert kwargs["pass_fds"] == (STUB_LOCK_FD,)
        environment = kwargs["env"]
        assert isinstance(environment, dict), (
            "status needs the inherited lock environment"
        )
        assert environment[SITE_OPERATION_LOCK_FD_ENV] == str(STUB_LOCK_FD)
        assert events == ["lock-enter"], "status was not read once under the lock"
        events.append(status_event)
        return subprocess.CompletedProcess(command, 1, json.dumps(report), "")

    def refused(*_args: object, **_kwargs: object) -> None:
        pytest.fail(
            "the integration test must not start real commands or a fast-path gate"
        )

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(staging_live_evidence.subprocess, "run", status)
    monkeypatch.setattr(
        staging_deploy, "read_live_status", staging_live_evidence.read_live_status
    )
    monkeypatch.setattr(
        staging_deploy,
        "collect_live_deploy_evidence",
        staging_live_evidence.collect_live_deploy_evidence,
    )
    monkeypatch.setattr(staging_deploy, "run_source_impact_gate", refused)

    result = staging_deploy.deploy(deploy_arguments(repository, state))

    assert result["deploy_mode"] == "APPLICATION_RELEASE"
    assert events == [
        "lock-enter",
        status_event,
        DEPLOY_EVENT,
        RECORD_EVIDENCE_EVENT,
        "success",
        "prune",
        "lock-exit",
    ], "a repair or unfinished transaction bypassed the application release"
