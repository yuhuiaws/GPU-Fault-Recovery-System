from __future__ import annotations

import json

import pytest

from scripts.e2e.regional import notify008_runner as runner
from tests.regional._cov95_notify008_lifecycle import setup_run


@pytest.mark.parametrize("lost_ack", ["none", "namespace", "job", "arm", "delete"])
def test_fake_regional_lifecycle_arms_only_after_admission_and_proves_cleanup(
    tmp_path, monkeypatch, lost_ack
):
    settings, api, case_dir, deadline = setup_run(tmp_path, monkeypatch)
    if lost_ack in {"namespace", "job"}:
        api.lose_create.add(lost_ack)
    api.lose_arm = lost_ack == "arm"
    api.lose_delete = lost_ack == "delete"

    code = runner.execute_case(settings, tmp_path, 1, deadline)

    assert code == 0, f"owned lifecycle did not recover the {lost_ack} acknowledgement"
    result = json.loads((case_dir / f"{runner.CASE_ID}.json").read_text())
    assert result["verdict"] == "PASS" and result["provider"] == "SIMULATED", (
        "only complete simulator evidence may pass this fake transport contract"
    )
    assert result["cleanup"]["namespace_absent"] is True, (
        "owned namespace must be absent"
    )
    assert result["cleanup"]["priorityclass_absent"] is True, (
        "namespace deletion alone cannot prove cluster-scoped resource cleanup"
    )
    assert result["cleanup"]["process_termination_proven"] is True, (
        "Pod absence alone is not process termination proof"
    )
    assert api.objects == {} and api.pod is None, (
        "the fake lifecycle must leave no owned resources"
    )
    modes = [
        item[item.index("scripts.e2e.regional.probes.notify008_probe") + 1]
        for item in api.calls
        if "scripts.e2e.regional.probes.notify008_probe" in item
    ]
    assert (
        modes.index("inspect")
        < modes.index("arm")
        < modes.index("prepare")
        < modes.index("run")
    ), "the actual admitted Pod must be inspected before ARM, DDL or runtime work"
    assert "stop" in modes, "cleanup must request termination before namespace deletion"


@pytest.mark.parametrize("operation", ["inspect", "arm", "prepare", "run"])
def test_probe_failure_is_not_promoted_to_pass_and_still_cleans_owned_namespace(
    tmp_path, monkeypatch, operation
):
    settings, api, case_dir, deadline = setup_run(tmp_path, monkeypatch)
    api.fail_probe = operation

    assert runner.execute_case(settings, tmp_path, 1, deadline) == 1, (
        f"a refused {operation} must not become a passing acceptance"
    )
    result = json.loads((case_dir / f"{runner.CASE_ID}.json").read_text())
    assert result["verdict"] == "FAIL", "the original probe failure must be preserved"
    assert api.objects == {}, "failure must still remove this run's namespace"


def test_admitted_credential_injection_prevents_arm_and_database_work(
    tmp_path, monkeypatch
):
    settings, api, case_dir, deadline = setup_run(tmp_path, monkeypatch)

    def inject(pod):
        pod["spec"]["containers"][0]["env"].append(
            {"name": "AWS_ROLE_ARN", "value": "unapproved-role"}
        )

    api.mutate_running = inject
    assert runner.execute_case(settings, tmp_path, 1, deadline) == 1, (
        "admission-injected identity must be rejected before the probe is armed"
    )
    assert api.armed is False, "no ARM may cross an admitted-spec drift"
    assert not any("prepare" in item or "run" in item for item in api.calls), (
        "database and runtime work must remain behind admission verification"
    )
    result = json.loads((case_dir / f"{runner.CASE_ID}.json").read_text())
    assert result["verdict"] == "FAIL", (
        "unknown admitted credentials cannot prove isolation"
    )
