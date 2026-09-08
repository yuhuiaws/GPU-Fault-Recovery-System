"""The consent refusals ``gpu-fault-admin deploy`` makes in its first minute.

They say exactly what the engine says (``regional_admin_commands`` and
``regional_schema_change``), so an operator sees one message whichever layer
stops the deploy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_fault.admin import deploy_consent as CONSENT
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault_release import regional_admin_commands as ADMIN
from gpu_fault_release import regional_schema_change as SCHEMA_CHANGE

SUPERSEDE = {ADMIN.SUPERSEDE_FAILED_TRANSACTION_ENV: "1"}
ACCEPT = {SCHEMA_CHANGE.ACCEPT_SCHEMA_CHANGE_ENV: "snapshot"}


@pytest.mark.parametrize("phase", sorted(CONSENT.SUPERSEDABLE_PHASES))
def test_a_different_candidate_over_a_failed_transaction_needs_the_flag(
    phase: str,
) -> None:
    refusal = CONSENT.consent_refusal(
        live_phase=phase,
        live_release_id="release-old",
        candidate_release_id="release-new",
        environment={},
    )

    assert refusal == ADMIN.foreign_candidate_resume_message(
        live_release_id="release-old", phase=phase, candidate_release_id="release-new"
    ), "the deploy entry and the engine say the same words"
    assert ADMIN.SUPERSEDE_FAILED_TRANSACTION_FLAG in refusal
    assert (
        CONSENT.consent_refusal(
            live_phase=phase,
            live_release_id="release-old",
            candidate_release_id="release-new",
            environment=SUPERSEDE,
        )
        is None
    ), "the flag is the consent"
    assert (
        CONSENT.consent_refusal(
            live_phase=phase,
            live_release_id="release-old",
            candidate_release_id="release-old",
            environment={},
        )
        is None
    ), "the failed release itself is resumed, not superseded"


def test_a_schema_change_under_auto_rollback_needs_acceptance() -> None:
    common = {
        "live_phase": "complete",
        "live_release_id": "release-old",
        "candidate_release_id": "release-new",
        "live_schema_version": 12,
        "candidate_schema_version": 13,
    }

    assert CONSENT.consent_refusal(**common, environment={}) == (
        SCHEMA_CHANGE.refusal_message()
    )
    assert CONSENT.consent_refusal(**common, environment=ACCEPT) is None
    assert (
        CONSENT.consent_refusal(**common, auto_rollback=False, environment={}) is None
    )
    assert (
        CONSENT.consent_refusal(**common, acceptance_recorded=True, environment={})
        is None
    ), "a resumed schema-change transaction carries the acceptance it started with"
    assert (
        CONSENT.consent_refusal(
            **{**common, "candidate_schema_version": None}, environment={}
        )
        is None
    ), "an unknown version is not compared; the engine will"


def test_refuse_unconsented_candidate_reads_state_and_manifest() -> None:
    with pytest.raises(BootstrapError, match="different release"):
        CONSENT.refuse_unconsented_candidate(
            live_state={"phase": "partial-convergence", "release_id": "release-old"},
            manifest={"release_id": "release-new"},
            auto_rollback=True,
            environment={},
        )
    with pytest.raises(BootstrapError, match="cannot be rolled back"):
        CONSENT.refuse_unconsented_candidate(
            live_state={
                "phase": "complete",
                "release_id": "release-old",
                "database_schema_version": 12,
            },
            manifest={"release_id": "release-new", "database_schema_version": 13},
            auto_rollback=True,
            environment={},
        )
    CONSENT.refuse_unconsented_candidate(
        live_state={
            "phase": "schema-ready",
            "release_id": "release-new",
            "database_schema_version": 12,
            "schema_change_acceptance": {"mode": "snapshot"},
        },
        manifest={"release_id": "release-new", "database_schema_version": 13},
        auto_rollback=True,
        environment={},
    )


def test_inner_hop_check_is_silent_without_a_site_or_a_manifest(tmp_path: Path) -> None:
    CONSENT.refuse_unconsented_release(
        state_dir=tmp_path, manifest_path=tmp_path / "missing.json", existing_site=None
    )
    CONSENT.refuse_unconsented_release(
        state_dir=tmp_path,
        manifest_path=tmp_path / "missing.json",
        existing_site={"spec": {}},
    )


def test_load_release_manifest_refuses_garbage(tmp_path: Path) -> None:
    path = tmp_path / "current-release.json"
    assert CONSENT.load_release_manifest(path) is None
    path.write_text(json.dumps({"release_id": "r"}), encoding="utf-8")
    assert CONSENT.load_release_manifest(path) == {"release_id": "r"}
    path.write_text("{", encoding="utf-8")
    with pytest.raises(BootstrapError, match="release manifest is invalid"):
        CONSENT.load_release_manifest(path)
