from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.e2e.regional.probes import notify008_protocol as contract
from tests.regional._cov95_notify008_support import (
    RUN_ID,
    report_record,
    variant_record,
)


def test_matrix_has_twelve_distinct_process_windows_and_explicit_ambiguity():
    variants = contract.variants()
    assert len(variants) == len({item.key for item in variants}) == 12, (
        "both delivery modes and notification kinds need all three process windows"
    )
    for item in variants:
        assert item.notification_id(RUN_ID).startswith(RUN_ID + "/"), (
            "every variant must retain its run-owned notification identity"
        )
        assert item.first_acceptances == (
            0 if item.crash == "before-provider" else 1
        ), "the pre-crash receipt count must distinguish unaccepted work"
        assert item.total_acceptances == (
            2 if item.crash == "accepted-before-commit" else 1
        ), "uncommitted acceptance must expose the duplicate window"
    assert contract.report_errors(report_record(), run_id=RUN_ID) == [], (
        "complete independent process observations must satisfy the proposed contract"
    )


@pytest.mark.parametrize("run_id", ["", "notify008-short", "notify008-" + "g" * 16, 3])
def test_run_identity_refuses_unowned_or_malformed_names(run_id):
    with pytest.raises(contract.ProbeError, match="run identity"):
        contract.run_identity(run_id)


@pytest.mark.parametrize(
    "arguments",
    [
        ("unknown", "before-provider", "gpu-reset"),
        ("inline", "unknown", "gpu-reset"),
        ("inline", "before-provider", "unknown"),
    ],
)
def test_unknown_variant_is_not_silently_mapped_to_a_known_window(arguments):
    with pytest.raises(contract.ProbeError, match="variant"):
        contract.Variant(*arguments)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("case_id", "GF-REGIONAL-NOTIFY-007"),
        ("run_id", "notify008-fedcba9876543210"),
        ("validation_scope", "LIVE"),
        ("provider", "SES"),
        ("postgres_major", True),
        ("database_is_unix_socket", False),
        ("production_credentials_loaded", True),
        ("exactly_once_proven", True),
        ("children_reaped", False),
        ("variants", []),
        ("variants", {}),
    ],
)
def test_report_refuses_wrong_scope_backend_or_incomplete_cleanup(field, bad):
    report = report_record()
    report[field] = bad
    assert contract.report_errors(report, run_id=RUN_ID), (
        f"report drift at {field} must prevent acceptance"
    )


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("variant", "inline/before-provider/gpu-reset"),
        ("notification_id", "foreign-notification"),
        ("crash_exitcode", 0),
        ("before_result", "SENT"),
        ("final_result", "FAILED"),
        ("stored_notifications", True),
        ("replay_added_acceptances", 1),
        ("early_retry_added_acceptances", 1),
        ("exactly_once_proven", True),
        ("duplicate_observed", False),
        ("supervisor_pid", None),
        ("provider_pid", 31),
        ("first_pid", True),
        ("replacement_pid", []),
        ("provider_message_id", "unobserved-acceptance"),
        ("before_receipts", []),
        ("receipts", []),
    ],
)
def test_variant_refuses_fabricated_commit_crash_or_provider_observations(field, bad):
    variant = contract.Variant("outbox", "accepted-before-commit", "gpu-reset")
    value = variant_record(variant)
    value[field] = bad
    assert contract.variant_errors(value, run_id=RUN_ID, variant=variant), (
        f"variant drift at {field} must prevent acceptance"
    )


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("run_id", "foreign"),
        ("notification_id", "foreign"),
        ("message_id", "not-an-independent-receipt"),
        ("provider_pid", True),
        ("provider_pid", 0),
        ("sequence", True),
        ("sequence", 0),
        ("sequence", 17),
    ],
)
def test_provider_receipts_are_bound_to_run_notification_and_process(field, bad):
    variant = contract.Variant("inline", "accepted-before-commit", "workload-restart")
    value = variant_record(variant)
    value["receipts"][0][field] = bad
    assert contract.variant_errors(value, run_id=RUN_ID, variant=variant), (
        f"a malformed provider receipt at {field} must be refused"
    )


@pytest.mark.parametrize("drift", ["duplicate", "provider", "history", "shape"])
def test_receipt_history_cannot_be_rewritten_or_duplicate_one_acceptance(drift):
    variant = contract.Variant("outbox", "accepted-before-commit", "gpu-reset")
    value = variant_record(variant)
    if drift == "duplicate":
        value["receipts"][1] = deepcopy(value["receipts"][0])
    elif drift == "provider":
        value["receipts"][0]["provider_pid"] = 99
    elif drift == "history":
        value["before_receipts"][0]["message_id"] = "simulated-" + "f" * 32
    else:
        value["receipts"][0] = {}
    assert contract.variant_errors(value, run_id=RUN_ID, variant=variant), (
        "independent provider history must survive the worker crash unchanged"
    )


def test_missing_probe_observations_are_refused():
    variant = contract.Variant("outbox", "accepted-before-commit", "gpu-reset")
    assert contract.variant_errors(None, run_id=RUN_ID, variant=variant), (
        "missing variant evidence must be refused"
    )
    assert contract.report_errors(None, run_id=RUN_ID), "missing report must be refused"


def test_plan_payload_digest_is_canonical_and_rejects_nonfinite_values():
    assert contract.digest({"a": 1, "b": 2}) == contract.digest({"b": 2, "a": 1}), (
        "dictionary insertion order must not change approved content identity"
    )
    assert contract.digest({"a": 1}) != contract.digest({"a": 2}), (
        "changing an approved value must change its digest"
    )
    with pytest.raises(ValueError):
        contract.digest({"budget": float("nan")})
