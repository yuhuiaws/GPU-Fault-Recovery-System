"""Acceptance scope, order and immutable runtime identity behavior."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
import yaml

from scripts.e2e.regional import acceptance_scope as scope
from scripts.e2e.regional import regional_case_contract as contract
from scripts.e2e.regional import runtime_identity_fields as identity
from scripts.e2e.regional import site_profile
from scripts.e2e.regional.regional_live_fixture import RUNTIME_IDENTITY_DEPLOYMENTS
from scripts.e2e.regional.remote_command_shapes import command_operations
from tests.regional import _cov95_residual_support as support

residual_isolation = support.residual_isolation


@pytest.mark.parametrize(
    "mode,reference",
    [
        ("unknown", ""),
        ("formal", "change-1"),
        ("selective", ""),
        ("selective", "bad reference"),
        ("selective", "a" * 129),
    ],
)
def test_invalid_execution_scope_refuses_before_evidence_can_claim_formal_sequence(
    mode, reference
):
    with pytest.raises(RuntimeError):
        scope.current_acceptance_scope(
            {scope.EXECUTION_SCOPE_ENV: mode, scope.SELECTION_REFERENCE_ENV: reference}
        )


@pytest.mark.parametrize(
    "value", [None, [], {}, {"case_id": "case-a"}, {"verdict": "PASS"}]
)
def test_non_case_documents_are_not_relabelled_by_selective_scope(monkeypatch, value):
    monkeypatch.setenv(scope.EXECUTION_SCOPE_ENV, "selective")
    monkeypatch.setenv(scope.SELECTION_REFERENCE_ENV, "change-1")
    assert scope.scoped_case_evidence(value) is value


def test_selective_scope_records_audit_reference_without_claiming_formal_completion(
    monkeypatch,
):
    monkeypatch.setenv(scope.EXECUTION_SCOPE_ENV, "selective")
    monkeypatch.setenv(scope.SELECTION_REFERENCE_ENV, "change-1")
    original = {"case_id": "case-a", "verdict": "PASS"}
    result = scope.scoped_case_evidence(original)
    assert result == {
        **original,
        "execution_scope": "selective",
        "selection_reference": "change-1",
        "formal_sequence_satisfied": False,
    }
    assert original == {"case_id": "case-a", "verdict": "PASS"}
    for name, value in scope.current_acceptance_scope().result_fields().items():
        wrong = "foreign" if not isinstance(value, bool) else True
        with pytest.raises(RuntimeError, match=name):
            scope.scoped_case_evidence({**original, name: wrong})


@pytest.fixture
def local_contract(monkeypatch, tmp_path, residual_isolation):
    importlib.reload(contract)
    order, catalog = tmp_path / "order.yaml", tmp_path / "catalog.yaml"
    with monkeypatch.context() as patch:
        patch.setattr(contract, "ORDER_PATH", order)
        patch.setattr(contract, "CATALOG_PATH", catalog)
        yield SimpleNamespace(module=contract, order=order, catalog=catalog)
    importlib.reload(contract)


def write_contract(case, order, entries=None):
    case.order.write_text(yaml.safe_dump(order))
    if entries is not None:
        case.catalog.write_text(yaml.safe_dump({"test_cases": entries}))


@pytest.mark.parametrize(
    "order",
    [
        [],
        {},
        {"phases": "wrong"},
        {
            "phases": [
                {
                    "sequence": 1,
                    "entries": [{"case": "GF-REGIONAL-X-001", "predecessor": 3}],
                }
            ]
        },
        {
            "phases": [
                {
                    "sequence": 1,
                    "entries": [
                        {"case": "GF-REGIONAL-X-001", "predecessor": "foreign"}
                    ],
                }
            ]
        },
        {
            "phases": [{"sequence": 1, "entries": [{"case": "GF-REGIONAL-X-001"}]}],
            "do_not_run": [{"case": "GF-REGIONAL-X-001"}],
        },
    ],
)
def test_invalid_order_cannot_yield_a_formal_predecessor(local_contract, order):
    write_contract(local_contract, order)
    with pytest.raises(contract.RegionalCaseContractError):
        contract.ordered_case_ids()


def test_case_metadata_uses_ordered_nonwrapper_predecessor_and_explicit_path(
    local_contract, tmp_path
):
    ids = [f"GF-REGIONAL-X-{number:03d}" for number in range(1, 4)]
    entries = [
        {
            "id": case_id,
            "title": case_id,
            "category": "unit",
            "level": "node",
            "risk": "read-only-signal-replay",
            "automation": "command",
            "procedure": "private procedure",
            "command": ["python3", "-m", "pytest"]
            if index == 1
            else ["private-runner"],
        }
        for index, case_id in enumerate(ids)
    ]
    entries.append({"id": "NONREGIONAL-IGNORED"})
    write_contract(
        local_contract,
        {
            "phases": [
                {"sequence": 2, "entries": [{"case": ids[2]}]},
                {
                    "sequence": 1,
                    "entries": [{"range": {"prefix": "X", "start": 1, "end": 2}}],
                },
            ]
        },
        entries,
    )
    assert contract.ordered_case_ids() == tuple(ids)
    assert contract.pytest_wrapper_case_ids() == (ids[1],)
    assert contract.formal_predecessor(ids[0]) is None
    assert contract.formal_predecessor(ids[2]) == ids[0]
    metadata = contract.case_metadata(ids[2])
    assert metadata.confirmation == "X003_EXECUTE"
    explicit = tmp_path / "private-proof.json"
    assert contract.predecessor_path(tmp_path, ids[2], explicit) == (
        ids[0],
        explicit.resolve(),
    )
    assert contract.predecessor_path(tmp_path, ids[0], explicit) == (None, None)
    with pytest.raises(contract.RegionalCaseContractError, match="not in"):
        contract.case_metadata("GF-REGIONAL-X-099")
    with pytest.raises(contract.RegionalCaseContractError, match="not in"):
        contract.formal_predecessor("GF-REGIONAL-X-099")


@pytest.mark.parametrize(
    "catalog",
    [
        [],
        {},
        {"test_cases": []},
        {"test_cases": [{"id": "GF-REGIONAL-X-001", "procedure": ""}]},
    ],
)
def test_missing_or_unusable_catalog_metadata_is_rejected(local_contract, catalog):
    local_contract.catalog.write_text(yaml.safe_dump(catalog))
    with pytest.raises(contract.RegionalCaseContractError):
        contract.case_metadata("GF-REGIONAL-X-001")


def safe_identity():
    return {
        "release_state": {
            "phase": "complete",
            "release_id": "release-a",
            "updated_at_epoch": 1,
        },
        "deployments": {
            plane: {
                name: {
                    "generation": 1,
                    "observed_generation": 1,
                    "desired_replicas": 1,
                    "ready_replicas": 1,
                    "updated_replicas": 1,
                    "available_replicas": 1,
                    "image": "private-image",
                }
                for name in names
            }
            for plane, names in RUNTIME_IDENTITY_DEPLOYMENTS.items()
        },
    }


@pytest.mark.parametrize(
    "change", ["allowed", "foreign-release", "unsafe", "missing-release"]
)
def test_runtime_identity_allowance_never_bypasses_safety_or_other_fields(
    tmp_path, change
):
    expected = safe_identity()
    current = deepcopy(expected)
    current["release_state"]["updated_at_epoch"] = 2
    if change == "foreign-release":
        current["release_state"]["release_id"] = "other"
    elif change == "unsafe":
        current["release_state"]["phase"] = "rolling-back"
        expected["release_state"]["phase"] = "rolling-back"
    elif change == "missing-release":
        current["release_state"] = expected["release_state"] = None
    path = tmp_path / "runtime.json"
    fixture = SimpleNamespace(runtime_identity=lambda: current)
    if change == "allowed":
        assert (
            identity.verify_runtime_identity_allowing(
                fixture,
                expected,
                evidence_path=path,
                stage="after",
                mutable_state_fields=("updated_at_epoch",),
            )
            == current
        )
    else:
        with pytest.raises(identity.RegionalFixtureError):
            identity.verify_runtime_identity_allowing(
                fixture,
                expected,
                evidence_path=path,
                stage="after",
                mutable_state_fields=("updated_at_epoch",),
            )
    assert json.loads(path.read_text()) == current
    assert identity.identity_without_state_fields(current, ()) is current


def test_head_and_batched_remote_operations_preserve_order_and_skip_empty_entries():
    assert command_operations(
        {
            "step": {"operation": "QUIESCE_GPU"},
            "batched_steps": [
                None,
                {},
                {"step": {}},
                {"step": {"operation": "RESTORE_GPU_SERVICES"}},
            ],
        }
    ) == ["QUIESCE_GPU", "RESTORE_GPU_SERVICES"]
    assert command_operations({}) == []


def test_profile_mapping_validation_and_late_parser_binding_are_real(
    tmp_path, monkeypatch
):
    path = tmp_path / "profile.json"
    path.write_text("[]")
    path.chmod(0o600)
    with pytest.raises(site_profile.SiteProfileError, match="mapping"):
        site_profile.load_site_profile(path)
    path.write_text(
        json.dumps({"arguments": {"cluster-id": "profile-cluster"}, "environment": {}})
    )
    monkeypatch.setenv(site_profile.SITE_PROFILE_ENV, str(path))
    monkeypatch.setattr(
        sys, "argv", ["private-parser", "--cluster-id", "explicit-cluster"]
    )
    parser = site_profile.bind_site_profile(argparse.ArgumentParser())
    parser.add_argument("--cluster-id")
    assert parser.parse_args().cluster_id == "explicit-cluster"
    namespace = argparse.Namespace()
    assert parser.parse_args([], namespace).cluster_id == "profile-cluster"
    assert site_profile.apply_site_environment(
        {"environment": {"KEY": "value"}}, {"KEY": " "}
    ) == ["KEY"]
    monkeypatch.delenv(site_profile.SITE_PROFILE_ENV)
    assert site_profile.install_site_profile([]) is None
