from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import cluster_batch_join as batch
from gpu_fault.admin import cluster_join as join
from gpu_fault.admin import cluster_join_evidence as evidence
from gpu_fault.admin import cluster_join_network as network_helpers
from gpu_fault.admin import cluster_join_readonly as readonly
from gpu_fault.admin import cluster_readiness as readiness
from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_join import JoinClusterRequest, join_cluster
from gpu_fault.admin.execution import current_deadline, deadline_scope
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from gpu_fault.admin.site import load_site
from tests.admin.test_admin_cluster_join import (
    GPU_B_ARN,
    Runner,
    _batch_execution,
    _membership_snapshot,
    _patch_batch_discovery,
    _prerequisites,
    _target,
)
from tests.admin.test_admin_cluster_join_rollback import Attempt, Commands
from tests.admin.test_admin_site import site_file


def test_join_rejects_the_cpu_eks_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = load_site(site_file(tmp_path))
    cpu_arn = site.release_config["cpu_eks_arn"]
    monkeypatch.setattr(
        join,
        "discover_cluster",
        lambda *_args, **_kwargs: replace(_target(), eks_arn=cpu_arn),
    )
    monkeypatch.setattr(
        join, "run_command", lambda *_args, **_kwargs: pytest.fail("command started")
    )
    monkeypatch.setattr(
        join, "run_driver", lambda *_args, **_kwargs: pytest.fail("driver started")
    )

    with pytest.raises(join.JoinTargetIdentityError, match="CPU EKS"):
        join_cluster(
            JoinClusterRequest(
                site=site, gpu_cluster_arn=cpu_arn, state_dir=tmp_path / "join"
            ),
            runner=Runner(),
        )

    assert len(load_site(site.source).release_config["clusters"]) == 1


@pytest.mark.parametrize("field", ["hyperpod_arn", "vpc_id", "node_recovery"])
def test_resumed_join_rejects_target_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    site = load_site(site_file(tmp_path))
    request = JoinClusterRequest(
        site=site, gpu_cluster_arn=GPU_B_ARN, state_dir=tmp_path / "join"
    )
    original = _target()
    monkeypatch.setattr(
        join,
        "discover_cluster",
        lambda *_args, **_kwargs: replace(original, **{field: "changed"}),
    )
    monkeypatch.setattr(
        join, "run_command", lambda *_args, **_kwargs: pytest.fail("command started")
    )

    with pytest.raises(join.JoinTargetIdentityError):
        join.validate_saved_join_target(
            request, {"target": asdict(original), "cluster_id": "hp-gpu-b"}, Runner()
        )


def test_readiness_refuses_a_late_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = load_site(site_file(tmp_path))
    clock = [100.0]
    calls: list[list[str]] = []
    monkeypatch.setattr(readiness.time, "monotonic", lambda: clock[0])

    def command(arguments, **kwargs):
        calls.append(list(arguments))
        assert 0 < kwargs["timeout_seconds"] <= 30
        if "exec" in arguments:
            clock[0] += 2
            output = json.dumps(
                {"cluster_id": "gpu-b", "ready": True, "nodes": [{"ready": True}]}
            )
        else:
            output = "ingress-pod"
        return subprocess.CompletedProcess(arguments, 0, output, "")

    monkeypatch.setattr(readiness, "run_command", command)

    with pytest.raises(BootstrapError, match="did not become ready"):
        readiness.wait_collector_readiness(site, "gpu-b", timeout_seconds=1)

    assert len(calls) == 2


@pytest.mark.parametrize(
    "report",
    [
        [],
        {"cluster_id": "another-cluster", "ready": True, "nodes": [{}]},
        {"cluster_id": "gpu-b", "ready": "true", "nodes": [{}]},
        {"cluster_id": "gpu-b", "ready": True, "nodes": None},
        {"cluster_id": "gpu-b", "ready": True, "nodes": []},
        {"cluster_id": "gpu-b", "ready": True, "nodes": [{"ready": False}]},
    ],
)
def test_readiness_refuses_unknown_response_identity_or_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, report: object
) -> None:
    site = load_site(site_file(tmp_path))
    clock = [100.0]
    monkeypatch.setattr(readiness.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        readiness.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    monkeypatch.setattr(
        readiness,
        "run_command",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, 0, json.dumps(report) if "exec" in arguments else "ingress", ""
        ),
    )

    with pytest.raises(BootstrapError, match="did not become ready"):
        readiness.wait_collector_readiness(
            site, "gpu-b", timeout_seconds=1, interval_seconds=5
        )

    assert clock[0] == 101


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_readiness_rejects_unbounded_timeouts(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        readiness.wait_collector_readiness(None, "gpu-b", timeout_seconds=timeout)


def test_membership_snapshot_uses_the_installed_ingress_and_supervised_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = load_site(site_file(tmp_path))
    calls: list[tuple[list[str], dict[str, Any]]] = []
    release = {
        "release_id": "release-a",
        "executor_image": "example@sha256:one",
        "rendered_manifest_sha256": "a" * 64,
    }
    registry = {
        "generation": 2,
        "content_sha256": "c" * 64,
        "cluster_states": {"gpu-a": "ACTIVE", "gpu-b": "PENDING"},
        "required_member_ids": [],
        "acked_member_ids": [],
        "missing_member_ids": [],
        "active_member_ids": [],
        "members": [],
        "converged": True,
    }

    def command(arguments, **kwargs):
        calls.append((list(arguments), kwargs))
        if "configmap" in arguments:
            output = json.dumps({"data": {"state.json": json.dumps(release)}})
        elif "exec" in arguments:
            output = json.dumps(registry)
        else:
            output = "ingress-pod"
        return subprocess.CompletedProcess(arguments, 0, output, "")

    monkeypatch.setattr(evidence, "run_command", command)

    first = evidence.membership_runtime_snapshot(site)
    release["rendered_manifest_sha256"] = "b" * 64
    membership_changed = evidence.membership_runtime_snapshot(site)
    assert (
        first["live_release_identity_sha256"]
        == membership_changed["live_release_identity_sha256"]
    )
    assert (
        first["live_release_state_sha256"]
        != membership_changed["live_release_state_sha256"]
    )
    release.update(
        phase="rolled-back",
        release_id="failed-release",
        previous={"release_id": "release-a"},
        rollback_result={"status": "PASSED"},
    )
    restored = evidence.membership_runtime_snapshot(site)
    assert (
        first["live_release_identity_sha256"]
        == restored["live_release_identity_sha256"]
    )
    release["executor_image"] = "example@sha256:two"
    second = evidence.membership_runtime_snapshot(site)

    assert any("app=gpu-fault-api-ha" in arguments for arguments, _kwargs in calls), (
        "membership snapshot did not query the installed API ingress"
    )
    assert all(0 < kwargs["timeout_seconds"] <= 60 for _arguments, kwargs in calls), (
        "membership probes must use a positive timeout of at most 60 seconds"
    )
    assert all("environment" in kwargs for _arguments, kwargs in calls), (
        "membership probes must use the supervised command environment"
    )
    assert (
        first["live_release_identity_sha256"] != second["live_release_identity_sha256"]
    )


@pytest.mark.parametrize("offset", [60.0, float("nan"), float("inf")])
def test_network_cache_refuses_future_or_nonfinite_observation_times(
    tmp_path: Path, offset: float
) -> None:
    site = load_site(site_file(tmp_path))
    cache = tmp_path / "network.json"
    write_json_atomic(
        cache,
        {
            "schema_version": 1,
            "site_sha256": site.source_sha256,
            "observed_at_epoch": datetime.now(timezone.utc).timestamp() + offset,
            "networks": [],
        },
    )

    assert readonly.load_network_baseline_cache(cache, site) is None


def test_interrupted_join_compensates_before_propagating_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    attempt.prepare_error = KeyboardInterrupt("cancelled")
    attempt.install(monkeypatch)

    with pytest.raises(KeyboardInterrupt, match="cancelled"):
        join_cluster(
            JoinClusterRequest(
                site=attempt.site,
                gpu_cluster_arn=GPU_B_ARN,
                state_dir=attempt.state_dir,
            ),
            runner=Runner(),
        )

    assert attempt.state()["phase"] == "ROLLED_BACK"
    assert attempt.membership == [True]
    assert not attempt.token_file.exists(), (
        "cancelled join left its original token file behind"
    )


def test_supervision_loss_never_starts_join_compensation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    attempt.prepare_error = ProcessSupervisionLost("ownership unproven")
    attempt.install(monkeypatch)

    with pytest.raises(ProcessSupervisionLost):
        join_cluster(
            JoinClusterRequest(
                site=attempt.site,
                gpu_cluster_arn=GPU_B_ARN,
                state_dir=attempt.state_dir,
            ),
            runner=Runner(),
        )

    assert attempt.rollouts == []
    assert attempt.commands.calls == []
    assert attempt.token_file.is_file(), (
        "supervision loss must retain the join token for recovery"
    )
    assert attempt.fleet_master_file.is_file(), (
        "supervision loss must retain the fleet master file for recovery"
    )
    assert attempt.state()["phase"] == "SUPERVISION_LOST"

    with pytest.raises(ProcessSupervisionLost, match="automatic retry is forbidden"):
        join_cluster(
            JoinClusterRequest(
                site=attempt.site,
                gpu_cluster_arn=GPU_B_ARN,
                state_dir=attempt.state_dir,
            ),
            runner=Runner(),
        )

    assert attempt.commands.calls == []


@pytest.mark.parametrize("step", ["ACTIVATION_STARTED", "ACTIVATED"])
def test_expired_activated_join_resumes_fail_forward_without_pending_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    attempt = Attempt(tmp_path)
    attempt.prepare_error = None
    attempt.extra_steps = ["VERIFIED", step]
    attempt.extra_evidence = {
        "VERIFIED": {
            "verified_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        }
    }
    attempt.install(monkeypatch)
    commits = []
    monkeypatch.setattr(
        join, "commit_membership", lambda *_args, **_kwargs: commits.append(True)
    )
    monkeypatch.setattr(
        join,
        "membership_runtime_snapshot",
        lambda _site: pytest.fail("PENDING verification ran after activation"),
    )

    result = join_cluster(
        JoinClusterRequest(
            site=attempt.site, gpu_cluster_arn=GPU_B_ARN, state_dir=attempt.state_dir
        ),
        runner=Runner(),
    )

    assert result["phase"] == "COMPLETED"
    assert commits == [True]
    assert attempt.rollouts == []
    assert attempt.token_file.is_file(), (
        "fail-forward completion must preserve the active cluster token"
    )


def test_failed_rollback_resumes_cleanup_before_starting_a_new_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    attempt.commands = Commands(failures=[("delete-role", 1, "AccessDenied")])
    attempt.run(monkeypatch, error="cluster deploy failed")
    assert attempt.state()["phase"] == "ROLLBACK_FAILED"
    assert attempt.token_file.is_file(), (
        "failed rollback must retain the join token for recovery"
    )
    attempt.commands = Commands()
    attempt.install(monkeypatch)
    monkeypatch.setattr(join, "discover_cluster", lambda *_args, **_kwargs: _target())

    def prepare(_request, **kwargs):
        assert kwargs["state"]["attempt"] == 2
        assert kwargs["state"]["completed_steps"] == []
        assert not attempt.token_file.exists(), (
            "new attempt started before the previous token file was cleaned up"
        )
        assert attempt.membership == [True]
        return {"phase": "NEW_ATTEMPT"}

    monkeypatch.setattr(join, "_prepare_execution", prepare)

    result = join_cluster(
        JoinClusterRequest(
            site=attempt.site, gpu_cluster_arn=GPU_B_ARN, state_dir=attempt.state_dir
        ),
        runner=Runner(),
    )

    assert result == {"phase": "NEW_ATTEMPT"}
    assert attempt.commands.matching("delete-role"), (
        "rollback retry did not delete the outstanding IAM role"
    )


def test_rollback_does_not_delete_a_recreated_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    attempt.install(monkeypatch)
    monkeypatch.setattr(
        join, "probe_join_namespace", lambda *_args, **_kwargs: "replacement-namespace"
    )

    with pytest.raises(BootstrapError, match="namespace changed before rollback"):
        join_cluster(
            JoinClusterRequest(
                site=attempt.site,
                gpu_cluster_arn=GPU_B_ARN,
                state_dir=attempt.state_dir,
            ),
            runner=Runner(),
        )

    assert attempt.commands.calls == []
    assert attempt.token_file.is_file(), (
        "namespace identity drift must retain the join token"
    )
    assert attempt.state()["phase"] == "ROLLBACK_STARTED"


def test_batch_rejects_mixed_sites_before_replacing_request_sites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    requests = tuple(
        JoinClusterRequest(site=load_site(site_file(root)), gpu_cluster_arn=GPU_B_ARN)
        for root in (first, second)
    )
    monkeypatch.setattr(
        batch,
        "reload_site_for_mutation",
        lambda _site: pytest.fail("mixed-site batch reached the mutation boundary"),
    )

    with pytest.raises(BootstrapError, match="same managed site"):
        batch.join_clusters(requests, runner_factory=Runner)


@pytest.mark.parametrize("failure_at", ["before", "after", "drift", "candidate"])
def test_shared_batch_verification_failure_records_every_affected_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_at: str
) -> None:
    path = site_file(tmp_path)
    document = yaml.safe_load(path.read_text())
    document["spec"]["autoRollback"] = False
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    site = load_site(path)
    requests = tuple(
        JoinClusterRequest(
            site=site,
            gpu_cluster_arn=f"arn:aws:eks:us-east-1:123456789012:cluster/gpu-{index}",
        )
        for index in range(2)
    )
    executions = {
        request.gpu_cluster_arn: _batch_execution(tmp_path, site, request, index)
        for index, request in enumerate(requests)
    }
    _patch_batch_discovery(monkeypatch, executions)
    monkeypatch.setattr(
        join,
        "run_driver",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(arguments, 0, "", ""),
    )
    monkeypatch.setattr(
        join, "wait_collector_readiness", lambda *_args: {"ready": True, "nodes": [{}]}
    )
    snapshots = []

    def snapshot(candidate):
        snapshots.append(True)
        if failure_at == "before" or (failure_at == "after" and len(snapshots) == 2):
            raise BootstrapError("shared snapshot failed")
        value = _membership_snapshot(candidate)
        if failure_at == "drift" and len(snapshots) == 2:
            value["registry_generation"] += 1
        if failure_at == "candidate" and len(snapshots) == 2:
            value["registry_cluster_states"]["gpu-0"] = "ACTIVE"
        return value

    monkeypatch.setattr(batch, "membership_runtime_snapshot", snapshot)

    with pytest.raises(BootstrapError, match="batch join completed with failed"):
        batch.join_clusters(requests, runner_factory=Runner)

    states = [
        json.loads(state.read_text())
        for state in (tmp_path / "join-cluster").glob("*/state.json")
    ]
    assert len(states) == 2
    assert all(state["phase"] == "FAILED" for state in states), (
        "shared verification failure must fail every batch attempt"
    )
    assert all(
        "ACTIVATION_STARTED" not in state["completed_steps"] for state in states
    ), "batch activation started despite failed shared verification"
    assert all("VERIFIED" not in state["completed_steps"] for state in states), (
        "failed shared verification marked a batch attempt verified"
    )


def test_batch_activation_advances_runtime_expectation_without_renewing_verification() -> (
    None
):
    observed = datetime.now(timezone.utc) - timedelta(minutes=10)
    record = {
        "verified_at": observed.isoformat(),
        "live_release_identity_sha256": "a" * 64,
        "registry_generation": 2,
        "registry_content_sha256": "b" * 64,
        "registry_cluster_states": {"gpu-a": "PENDING", "gpu-b": "PENDING"},
    }
    activated = {
        **record,
        "registry_generation": 3,
        "registry_content_sha256": "c" * 64,
        "registry_cluster_states": {"gpu-a": "ACTIVE", "gpu-b": "PENDING"},
    }

    advanced = evidence.advance_batch_verification(
        record, committed_evidence=record, final_identity=activated, cluster_id="gpu-a"
    )

    assert advanced["verified_at"] == observed.isoformat()
    assert advanced["registry_generation"] == 2
    assert advanced["post_verification_runtime"]["registry_generation"] == 3
    assert record["registry_cluster_states"]["gpu-a"] == "PENDING"

    activated["registry_cluster_states"]["gpu-b"] = "ACTIVE"
    with pytest.raises(BootstrapError, match="drifted during sibling"):
        evidence.advance_batch_verification(
            record,
            committed_evidence=record,
            final_identity=activated,
            cluster_id="gpu-a",
        )


def test_partial_join_failure_compensates_the_published_pending_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    attempt.prepare_error = None
    attempt.joined = False
    attempt.install(monkeypatch)

    def rollout(_site, mode, *, cluster_id=None):
        attempt.rollouts.append((mode, cluster_id))
        if mode == "join-cluster":
            raise BootstrapError("partial join failed")

    monkeypatch.setattr(join, "_run_rollout", rollout)

    with pytest.raises(BootstrapError, match="partial join failed"):
        join_cluster(
            JoinClusterRequest(
                site=attempt.site,
                gpu_cluster_arn=GPU_B_ARN,
                state_dir=attempt.state_dir,
            ),
            runner=Runner(),
        )

    assert attempt.rollouts == [
        ("join-cluster", "hp-gpu-b"),
        ("fail-cluster", "hp-gpu-b"),
    ]
    assert attempt.membership == [True]
    state = attempt.state()
    assert state["phase"] == "ROLLED_BACK"
    retained = Path(state["retained_cluster_token"]["path"])
    assert retained.parent == attempt.site.source.parent / "secure"
    assert retained.stat().st_mode & 0o777 == 0o600
    assert retained.stat().st_size == 64
    assert state["retained_cluster_token"]["hyperpod_arn"] == _target().hyperpod_arn


def test_rollback_refuses_to_clean_an_unexpectedly_active_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    attempt.install(monkeypatch)
    monkeypatch.setattr(
        join,
        "membership_runtime_snapshot",
        lambda _site: {"registry_cluster_states": {"hp-gpu-b": "ACTIVE"}},
    )

    with pytest.raises(BootstrapError, match="does not permit rollback"):
        join_cluster(
            JoinClusterRequest(
                site=attempt.site,
                gpu_cluster_arn=GPU_B_ARN,
                state_dir=attempt.state_dir,
            ),
            runner=Runner(),
        )

    assert attempt.rollouts == []
    assert attempt.commands.calls == []
    assert attempt.membership == []
    assert attempt.state()["phase"] == "ROLLBACK_FAILED"
    assert attempt.token_file.is_file(), (
        "active registration must retain its join token"
    )


def test_token_retention_survives_resetting_a_rolled_back_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    attempt.run(monkeypatch, error="cluster deploy failed")
    retained = dict(attempt.state()["retained_cluster_token"])
    monkeypatch.setattr(join, "discover_cluster", lambda *_args, **_kwargs: _target())

    def prepare(_request, **kwargs):
        assert kwargs["state"]["attempt"] == 2
        assert kwargs["state"]["retained_cluster_token"] == retained
        return {"phase": "RETRY"}

    monkeypatch.setattr(join, "_prepare_execution", prepare)

    assert join_cluster(
        JoinClusterRequest(
            site=attempt.site, gpu_cluster_arn=GPU_B_ARN, state_dir=attempt.state_dir
        ),
        runner=Runner(),
    ) == {"phase": "RETRY"}


@pytest.mark.parametrize("failure_at", ["ingress", "association", "visibility"])
def test_join_network_checkpoints_partial_work_before_reporting_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_at: str
) -> None:
    site = load_site(site_file(tmp_path))
    site.release_config["dns"] = {"hosted_zone_id": "Z123"}
    state: dict[str, Any] = {}

    class NetworkRunner:
        def aws_json(self, _region, *arguments):
            assert "get-hosted-zone" in arguments
            return {"VPCs": []}

        def run(self, arguments, **_kwargs):
            assert "associate-vpc-with-hosted-zone" in arguments
            record = json.loads((tmp_path / "state.json").read_text())
            assert (
                record["evidence"]["PREREQUISITES_READY"]["network"]["pending_mutation"]
                == "vpc-association"
            )
            if failure_at == "association":
                raise BootstrapError("association request outcome unknown")
            return ""

    def command(arguments, **_kwargs):
        if "192.0.2.21/32" in arguments and failure_at == "ingress":
            raise TimeoutError("ingress request timed out")
        return subprocess.CompletedProcess(arguments, 0, "", "")

    def wait(*_args, **_kwargs):
        raise BootstrapError("association visibility timed out")

    monkeypatch.setattr(
        join, "_existing_cluster_networks", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        join, "_gpu_nat_eips", lambda *_args: ["192.0.2.20", "192.0.2.21"]
    )
    monkeypatch.setattr(
        join, "ensure_executor_role", lambda *_args, **_kwargs: {"role_arn": "role"}
    )
    monkeypatch.setattr(
        join, "provision_node_action_keys", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(join, "run_command", command)
    monkeypatch.setattr(network_helpers, "wait_until", wait)

    with pytest.raises((BootstrapError, TimeoutError)):
        _prerequisites(site, tmp_path=tmp_path, state=state, runner=NetworkRunner())

    network = json.loads((tmp_path / "state.json").read_text())["evidence"][
        "PREREQUISITES_READY"
    ]["network"]
    assert network["complete"] is False
    assert "192.0.2.20" in network["created_ingress_eips"]
    if failure_at == "visibility":
        assert network["association_created"] is True
        assert network["pending_mutation"] is None
    else:
        assert network["pending_mutation"], (
            "an unconfirmed network request must retain its mutation intent"
        )


def test_unproven_network_mutation_never_becomes_a_clean_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    prerequisites = attempt.evidence()["PREREQUISITES_READY"]
    prerequisites["network"]["pending_mutation"] = "vpc-association"
    attempt.extra_evidence = {"PREREQUISITES_READY": prerequisites}

    attempt.run(monkeypatch, error="network mutation outcome is unproven")

    assert attempt.state()["phase"] == "ROLLBACK_FAILED"
    assert attempt.token_file.is_file(), (
        "unproven network mutation must retain the join token"
    )
    assert not attempt.commands.matching("disassociate-vpc-from-hosted-zone"), (
        "rollback detached a VPC with an unproven network mutation"
    )
    assert not attempt.commands.matching("revoke-security-group-ingress"), (
        "rollback revoked ingress with an unproven network mutation"
    )


def test_readonly_join_workers_inherit_the_callers_deadline() -> None:
    with deadline_scope("join test", 5) as deadline:
        observed = readonly.parallel_verify_and_discover(
            lambda: current_deadline() == deadline or pytest.fail("deadline lost"),
            current_deadline,
        )

    assert observed == deadline


@pytest.mark.parametrize("limit", [0, 9, -1, True, "2"])
def test_join_cluster_parallelism_never_silently_widens_or_coerces(
    tmp_path: Path, limit: object
) -> None:
    site = load_site(site_file(tmp_path))
    site.release_config["release"]["upgrade_max_parallel_clusters"] = limit

    with pytest.raises(BootstrapError, match="within 1..8"):
        batch.deploy_concurrency(site)


def test_rollback_retry_reloads_membership_before_starting_the_next_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    attempt.path.write_bytes(attempt.candidate_path.read_bytes())
    request = JoinClusterRequest(
        site=load_site(attempt.path),
        gpu_cluster_arn=GPU_B_ARN,
        state_dir=attempt.state_dir,
    )
    state = {"phase": "ROLLBACK_FAILED", "attempt": 1, "completed_steps": []}
    transaction = join.JoinAttempt(
        request, attempt.state_dir, attempt.state_dir / "state.json", state
    )

    def rollback(_request, **kwargs):
        document = yaml.safe_load(attempt.path.read_text())
        document["spec"]["clusters"] = document["spec"]["clusters"][:1]
        attempt.path.write_text(yaml.safe_dump(document, sort_keys=False))
        kwargs["state"]["phase"] = "ROLLED_BACK"

    monkeypatch.setattr(join, "_rollback", rollback)

    assert join.resume_join_rollback(transaction, Runner()) is True
    assert [
        cluster["cluster_id"]
        for cluster in transaction.request.site.release_config["clusters"]
    ] == ["gpu-a"]
    assert state["source_site_sha256"] == transaction.request.site.source_sha256
    assert state["attempt"] == 2


def test_local_cleanup_retry_uses_the_already_retained_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    attempt.run(monkeypatch, error="cluster deploy failed")
    state = attempt.state()
    assert not attempt.token_file.exists(), (
        "rollback left the original token file instead of retaining it"
    )
    retained = Path(state["retained_cluster_token"]["path"])
    state["phase"] = "ROLLBACK_FAILED"
    state["completed_steps"] = ["REMOTE_ROLLBACK_COMPLETED"]
    state["evidence"] = {
        **attempt.evidence(),
        "REMOTE_ROLLBACK_COMPLETED": {"registry_present": True},
    }
    write_json_atomic(attempt.state_dir / "state.json", state)
    attempt.commands.calls.clear()
    monkeypatch.setattr(
        join, "_prepare_execution", lambda *_args, **_kwargs: {"phase": "RETRY"}
    )

    result = join_cluster(
        JoinClusterRequest(
            site=attempt.site, gpu_cluster_arn=GPU_B_ARN, state_dir=attempt.state_dir
        ),
        runner=Runner(),
    )

    assert result == {"phase": "RETRY"}
    assert retained.is_file(), "local cleanup retry removed the retained cluster token"
    assert all("config" in arguments for arguments in attempt.commands.calls), (
        "local cleanup retry issued a non-kubeconfig command"
    )


def test_final_membership_identity_rejects_release_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = load_site(site_file(tmp_path))
    monkeypatch.setattr(
        evidence,
        "membership_runtime_snapshot",
        lambda _site: {
            "registry_cluster_states": {"gpu-b": "ACTIVE"},
            "live_release_identity_sha256": "b" * 64,
        },
    )

    with pytest.raises(BootstrapError, match="live release identity drifted"):
        evidence.final_membership_identity(
            site,
            cluster_id="gpu-b",
            candidate_site_sha256="c" * 64,
            verified_at=datetime.now(timezone.utc).isoformat(),
            verification_evidence={"live_release_identity_sha256": "a" * 64},
        )
