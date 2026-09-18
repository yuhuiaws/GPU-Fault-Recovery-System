from __future__ import annotations

import pytest

from gpu_fault_release import regional_release_prerequisite_repair as repair
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional._prerequisite_repair_support import repair_release
from tests.regional.test_release_prerequisite_repair import prepare


def test_bootstrap_retains_only_completed_job_receipts_after_removing_live_journal(
    monkeypatch,
):
    release = repair_release(monkeypatch, bootstrap=True)
    repair.prepare_bootstrap_workflows(release)
    assert repair.REPAIR_KEY not in release.state
    proof = release.state["bootstrap_store_safety"]
    receipts = proof["proof_jobs"]
    assert set(receipts) == {"credential", "store"}
    assert receipts["store"]["uid"] == proof["job_uid"]
    assert all(
        item["status"] == "REMOVED" and item["owner_uid"] and item["namespace"]
        for item in receipts.values()
    ), "retained proof receipts must bind removed Jobs to their namespace and owner"
    assert not release.runner.jobs, "bootstrap proof must not leave live Jobs behind"
    assert "proof_jobs" not in release.state[repair.BOOTSTRAP_ORIGIN_KEY]
    for item in receipts.values():
        assert set(item) == {
            "name",
            "namespace",
            "uid",
            "owner_uid",
            "run_id",
            "spec_sha256",
            "status",
        }, "the receipt must not include a Secret, Pod command or environment"


def test_store_proof_ack_loss_cleans_its_original_job_and_retries_with_new_identity(
    monkeypatch,
):
    release = repair_release(monkeypatch, bootstrap=True)
    prepare(release, bootstrap=True)
    release.runner.create_ack_lost = True
    with pytest.raises(ReleaseError, match="ACK lost"):
        repair.bootstrap_workflow_proof(release)
    failed = dict(release.state[repair.REPAIR_KEY]["jobs"]["store"])
    assert failed["uid"] and failed["status"] == "REMOVED"
    assert not release.runner.jobs, "lost proof ACK must still clean the original Job"
    assert "bootstrap_store_safety" not in release.state
    repair.prepare_bootstrap_workflows(release)
    current = release.state["bootstrap_store_safety"]["proof_jobs"]["store"]
    assert current["run_id"] != failed["run_id"]
    assert current["status"] == "REMOVED"
    assert not release.runner.jobs, (
        "successful proof retry must clean its replacement Job"
    )
