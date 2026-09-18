from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_e2e002_multicluster_fault as e2e
from scripts.e2e.regional import run_iso006_cluster_offline as offline
from scripts.e2e.regional import run_iso007_failed_recovery_isolation as asymmetric
from tests.regional._cov95_identity_isolation_support import (
    arguments,
    preflight_environment,
)
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional.test_multi_cluster_fixture_review import pod_document


@pytest.mark.parametrize("module", [e2e, offline])
@pytest.mark.parametrize("explicit", [False, True])
def test_multicluster_config_binds_run_attempt_and_explicit_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module: Any, explicit: bool
) -> None:
    args = arguments(module, tmp_path)
    if explicit:
        args.job_id, args.attempt_id = "chosen-job", "chosen-attempt"
        args.predecessor_evidence = str(tmp_path / "previous")
    config = module.configure(args)
    environment = config.environment()
    assert (
        environment["GPU_A_CLUSTER_ID"] == "a"
        and environment["GPU_B_CLUSTER_ID"] == "b"
    )
    assert environment["GPU_FAULT_SHARED_JOB_ID"] == config.job_id
    assert environment["GPU_FAULT_SHARED_ATTEMPT_ID"] == config.attempt_id
    if explicit:
        assert config.job_id == "chosen-job" and config.attempt_id == "chosen-attempt"
        assert config.predecessor_path == tmp_path / "previous"
    else:
        next_config = module.configure(
            SimpleNamespace(**{**vars(args), "attempt": args.attempt + 1})
        )
        assert next_config.job_id != config.job_id
        assert config.attempt_id == config.job_id + "-a001"
    if module is offline:
        assert config.restore_seconds <= 3600
    elif not explicit:
        assert asymmetric.configure(args).job_id != config.job_id
        assert asymmetric.configure(args).predecessor_path.parent.name == e2e.CASE_ID


@pytest.mark.parametrize("duration", [59, 1801])
def test_iso006_config_refuses_unapproved_duration(
    tmp_path: Path, duration: int
) -> None:
    args = arguments(offline, tmp_path)
    args.duration_seconds = duration
    with pytest.raises(offline.RegionalFixtureError, match="outside 60..1800"):
        offline.configure(args)


def test_iso006_requires_explicit_control_plane_cidrs(tmp_path: Path) -> None:
    args = arguments(offline, tmp_path)
    args.control_plane_cidr = []
    with pytest.raises(offline.RegionalFixtureError, match="CIDR is required"):
        offline.configure(args)


@pytest.mark.parametrize("module", [e2e, offline])
@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "predecessor",
        "count",
        "synthetic",
        "physical",
        "nodes-a",
        "nodes-b",
        "workload-a",
        "workload-b",
        "site",
        "tests",
    ],
)
def test_multicluster_preflight_records_exact_refusal_without_mutations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module: Any, defect: str
) -> None:
    settings, regions, registrations, reads = preflight_environment(
        monkeypatch, tmp_path, module
    )
    if defect == "predecessor":
        monkeypatch.setattr(
            module, "predecessor_evidence", lambda *args, **kwargs: {"valid": False}
        )
    elif defect == "count":
        registrations.pop()
    elif defect == "synthetic":
        registrations[1]["synthetic"] = True
    elif defect == "physical":
        registrations[1]["eks_cluster_arn"] = registrations[0]["eks_cluster_arn"]
    elif defect.startswith("nodes-"):
        regions[0 if defect == "nodes-a" else 1].nodes = []
    elif defect.startswith("workload-"):
        regions[0 if defect == "workload-a" else 1].workloads = [{"name": "foreign"}]
    elif defect == "site":
        settings = replace(settings, site_file=tmp_path / "absent")
    elif defect == "tests":
        monkeypatch.setattr(
            module, "focused_tests", lambda *args, **kwargs: {"passed": False}
        )
    result = module.read_only_preflight(settings, tmp_path)
    assert bool(result["errors"]) is (defect != "none")
    assert json.loads((tmp_path / "preflight.json").read_text()) == result
    assert reads and all(args[1] == "get" for args in reads)
    if defect == "none":
        details = module.plan_details(settings, result)
        assert details["preflight_identity"]["release_id"] == "unit-release"
        assert len(details["preflight_identity"]["cluster_a_node_uids"]) == 3
        assert details["shared_job_id"] == settings.job_id


@pytest.mark.parametrize("defect", ["not-ready", "claim"])
def test_iso006_preflight_requires_healthy_claims_and_node_readiness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    settings, regions, _registrations, _reads = preflight_environment(
        monkeypatch, tmp_path, offline
    )
    if defect == "not-ready":
        regions[1].nodes[0]["ready"] = "False"
    else:
        monkeypatch.setattr(
            offline,
            "claim_sample",
            lambda region: {"status": 503 if region is regions[1] else 200},
        )
    result = offline.read_only_preflight(settings, tmp_path)
    assert any(
        "not Ready" in error or "cannot claim" in error for error in result["errors"]
    ), result["errors"]


@pytest.mark.parametrize("module", [e2e, offline])
@pytest.mark.parametrize("cached", [None, {"passed": True}, {"passed": False}])
@pytest.mark.parametrize("code", [0, 1])
def test_focused_test_wrappers_reuse_proof_or_record_the_actual_returncode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module: Any, cached: Any, code: int
) -> None:
    monkeypatch.setattr(module, "reusable_focused_tests", lambda path: cached)
    calls = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, code, "unit stdout", "unit stderr")

    monkeypatch.setattr(module.RegionalLiveFixture, "run", run)
    result = module.focused_tests(tmp_path, reuse=True)
    if cached is not None:
        assert result == {**cached, "focused_tests_reused": True}
        assert calls == []
    else:
        assert result["passed"] is (code == 0)
        assert result["returncode"] == code
        assert calls[0][1]["check"] is False
        assert calls[0][1]["timeout"] == 300
        assert (tmp_path / "focused-tests.log").read_text() == "unit stdoutunit stderr"
        assert (tmp_path / "focused-tests.log").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "shape", ["literal", "field-ref", "missing", "ambiguous", "wrong-ref", "not-ready"]
)
def test_e2e002_executor_identity_is_explicit_and_not_ambiguous(shape: str) -> None:
    document = pod_document()
    pod = document["items"][0]
    entry: dict[str, Any] = {"name": "GPU_FAULT_EXECUTOR_ID"}
    if shape == "literal":
        entry["value"] = "a/executor"
    elif shape in {"field-ref", "wrong-ref"}:
        entry["valueFrom"] = {
            "fieldRef": {
                "fieldPath": "metadata.name" if shape == "field-ref" else "metadata.uid"
            }
        }
    pod["spec"]["containers"][0]["env"] = [
        {"name": "IGNORED", "value": "ignore"},
        entry,
    ]
    if shape == "ambiguous":
        entry["value"] = "first"
        pod["spec"]["containers"][1]["env"] = [
            {"name": "GPU_FAULT_EXECUTOR_ID", "value": "second"}
        ]
    elif shape == "not-ready":
        pod["status"]["phase"] = "Pending"
    region = SimpleNamespace(kubectl=lambda *args: json.dumps(document))
    if shape in {"literal", "field-ref"}:
        assert e2e.executor_identities(region) == [
            "a/executor" if shape == "literal" else "api"
        ]
    else:
        with pytest.raises(
            e2e.RegionalFixtureError, match="Ready|missing or ambiguous"
        ):
            e2e.executor_identities(region)


@pytest.mark.parametrize("defect", ["unreadable", "nonobject", "node-uid"])
def test_iso007_preflight_cannot_borrow_unbound_e2e002_pair_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    previous = tmp_path / "previous"
    if defect != "unreadable":
        previous.write_text(
            "[]" if defect == "nonobject" else '{"cluster_ids":["a","b"]}',
            encoding="ascii",
        )
    settings = SimpleNamespace(
        predecessor_path=previous,
        multi=SimpleNamespace(
            cluster_a=SimpleNamespace(cluster_id="a"),
            cluster_b=SimpleNamespace(cluster_id="b"),
        ),
    )
    node = {
        "name": "node",
        "uid": "" if defect == "node-uid" else "uid",
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "labels": {},
    }
    monkeypatch.setattr(
        e2e,
        "read_only_preflight",
        lambda *args, **kwargs: {
            "errors": [],
            "nodes_a": [copy.deepcopy(node)],
            "nodes_b": [copy.deepcopy(node)],
        },
    )
    assert asymmetric.read_only_preflight(settings, tmp_path)["errors"]


def test_iso007_plan_preserves_asymmetric_budget_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings, _regions, _registrations, _reads = preflight_environment(
        monkeypatch, tmp_path, e2e
    )
    preflight = e2e.read_only_preflight(settings, tmp_path)
    result = asymmetric.plan_details(settings, preflight)
    assert result["expected_outcomes"] == {
        "cluster_a": "terminal FAILED / RESTART_BUDGET_EXHAUSTED / no remote restart",
        "cluster_b": "terminal SUCCEEDED / one local workload restart",
    }
    assert result["stop_conditions"][0].startswith("E2E-002 predecessor"), (
        "ISO007 must require its same-pair E2E002 predecessor"
    )
    assert result["rollback"]["no_node_or_provider_mutation"] is True


@pytest.mark.parametrize("module", [e2e, offline, asymmetric])
def test_standard_entrypoints_delegate_only_to_the_guard(
    monkeypatch: pytest.MonkeyPatch, module: Any
) -> None:
    calls = []
    monkeypatch.setattr(
        module, "run_standard_case", lambda case: calls.append(case) or 1
    )
    assert module.main() == 1
    assert calls == [module.CASE]
