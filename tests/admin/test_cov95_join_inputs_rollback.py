from __future__ import annotations

import copy
from pathlib import Path

import pytest

from gpu_fault.admin import cluster_join as join
from gpu_fault.admin import cluster_join_rollback as rollback
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_join_inputs import prepare_local_inputs
from gpu_fault.admin.cluster_join_state import load_join_state
from tests.admin._cov95_join_support import JoinScenario, target
from tests.admin.test_admin_cluster_join_rollback import GPU_B_ARN, Attempt, Commands


@pytest.fixture
def local(tmp_path, monkeypatch):
    scenario = JoinScenario(tmp_path, monkeypatch)
    request = scenario.request()
    directory, path, state = load_join_state(request)
    return scenario, request, directory, path, state


def prepare(context):
    scenario, request, directory, path, state = context
    prepare_local_inputs(
        request,
        runner=scenario,
        target=target(),
        cluster_id="hp-gpu-b",
        state_dir=directory,
        state_path=path,
        state=state,
    )


def test_join_inputs_cannot_write_cpu_kubeconfig(local):
    scenario, request, _directory, _path, _state = local
    request.site.environment["KUBECONFIG"] = request.site.release_config[
        "cpu_kubeconfig"
    ]
    with pytest.raises(join.JoinTargetIdentityError, match="CPU kubeconfig"):
        prepare(local)
    assert scenario.commands == []


@pytest.mark.parametrize("bound", [False, True])
def test_existing_cluster_token_requires_exact_retained_ownership(local, bound):
    scenario, request, _directory, _path, state = local
    token = request.site.source.parent / "secure/hp-gpu-b.token"
    token.write_text("t" * 64)
    token.chmod(0o600)
    before = token.read_bytes()
    if bound:
        state["retained_cluster_token"] = {
            "path": str(token),
            "cluster_id": "hp-gpu-b",
            "eks_arn": target().eks_arn,
            "hyperpod_arn": target().hyperpod_arn,
        }
        prepare(local)
        assert "LOCAL_INPUTS_READY" in state["completed_steps"]
    else:
        with pytest.raises(
            join.JoinTargetIdentityError, match="no matching join ownership"
        ):
            prepare(local)
        assert scenario.commands == []
    assert token.read_bytes() == before


def test_join_inputs_refuse_existing_context_alias_before_update(local):
    scenario = local[0]
    scenario.gpu_kubeconfig.touch(mode=0o600)
    scenario.contexts.add(target().context)
    with pytest.raises(join.JoinTargetIdentityError, match="existing kubeconfig alias"):
        prepare(local)
    assert len(scenario.commands) == 1
    assert not scenario.commands[0][1].get("mutate"), (
        "alias refusal performed a mutation"
    )


@pytest.mark.parametrize("kind", ["foreign", "unobservable"])
def test_join_namespace_requires_matching_identity_after_creation(local, kind):
    scenario = local[0]
    if kind == "foreign":
        scenario.namespace_uids[target().context] = "foreign-namespace"
    else:
        scenario.namespace_read_override = ""
    with pytest.raises(BootstrapError, match="existing or replaced|not observable"):
        prepare(local)
    assert "LOCAL_INPUTS_READY" not in local[4]["completed_steps"]
    assert not (scenario.path.parent / "secure/hp-gpu-b.token").exists(), (
        "unproven namespace created a cluster credential"
    )


def test_prepared_local_inputs_reuse_same_namespace_and_token(local):
    prepare(local)
    original = copy.deepcopy(local[4]["evidence"]["LOCAL_INPUTS_READY"])
    token = Path(original["token_file"]).read_bytes()
    prepare(local)
    assert local[4]["evidence"]["LOCAL_INPUTS_READY"] == original
    assert Path(original["token_file"]).read_bytes() == token


@pytest.fixture
def attempt(tmp_path, monkeypatch):
    value = Attempt(tmp_path)
    value.install(monkeypatch)
    request = join.JoinClusterRequest(
        site=value.site, gpu_cluster_arn=GPU_B_ARN, state_dir=value.state_dir
    )
    directory, path, state = load_join_state(request)
    state["evidence"] = value.evidence()
    state["completed_steps"] = [*state["evidence"], "NODE_KEYS_STARTED", "JOINED"]
    return value, request, directory, path, state


def compensate(context):
    _attempt, request, directory, path, state = context
    rollback.rollback(request, state_dir=directory, state_path=path, state=state)


@pytest.mark.parametrize("step", ["ACTIVATION_STARTED", "ACTIVATED"])
def test_join_compensation_cannot_cross_activation_intent(attempt, step):
    attempt[4]["completed_steps"].append(step)
    with pytest.raises(BootstrapError, match="rollback is forbidden"):
        compensate(attempt)
    assert attempt[0].commands.calls == []
    assert attempt[0].rollouts == []


def test_join_compensation_requires_candidate_for_started_rollout(attempt):
    attempt[0].candidate_path.unlink()
    with pytest.raises(BootstrapError, match="candidate is missing"):
        compensate(attempt)
    assert attempt[0].commands.calls == []
    assert attempt[0].rollouts == []


@pytest.mark.parametrize("field", ["namespace_creation_started", "namespace_uid"])
def test_join_compensation_rejects_unknown_namespace_ownership(attempt, field):
    attempt[4]["evidence"]["LOCAL_INPUTS_READY"].pop(field)
    with pytest.raises(join.JoinTargetIdentityError, match="namespace"):
        compensate(attempt)
    assert attempt[0].commands.calls == []
    assert attempt[0].rollouts == []


@pytest.mark.parametrize(
    "field", ["node_uids", "expected_key_sha256", "cluster_id", "eks_arn"]
)
def test_join_compensation_never_removes_keys_without_complete_proof(attempt, field):
    attempt[4]["evidence"]["NODE_NAMES_VERIFIED"].pop(field)
    with pytest.raises(BootstrapError, match="node/key ownership proof"):
        compensate(attempt)
    assert attempt[0].cleared == []
    assert attempt[0].rollouts == []


@pytest.mark.parametrize("states", [None, {"hp-gpu-b": "ACTIVE"}, {}])
def test_join_compensation_rejects_unknown_or_active_registry_membership(
    attempt, monkeypatch, states
):
    monkeypatch.setattr(
        join,
        "membership_runtime_snapshot",
        lambda _site: {"registry_cluster_states": states},
    )
    with pytest.raises(BootstrapError, match="registry state"):
        compensate(attempt)
    assert attempt[0].commands.calls == []
    assert attempt[0].rollouts == []


@pytest.mark.parametrize(
    "fragment",
    [
        "delete-role-policy",
        "delete-role",
        "revoke-security-group-ingress",
        "disassociate-vpc-from-hosted-zone",
        "get-contexts",
    ],
)
def test_join_compensation_failure_is_persisted_and_retry_finishes(
    attempt, monkeypatch, fragment
):
    value, _request, _directory, _path, state = attempt
    commands = Commands(failures=[(fragment, 1, "example cleanup failure")])
    monkeypatch.setattr(join, "run_command", commands)
    with pytest.raises(BootstrapError, match="rollback|kubeconfig"):
        compensate(attempt)
    assert state["phase"] == "ROLLBACK_FAILED"
    prior = len(commands.matching("delete-role"))
    remote_complete = "REMOTE_ROLLBACK_COMPLETED" in state["completed_steps"]
    commands.failures = ()
    compensate(attempt)
    assert state["phase"] == "ROLLED_BACK"
    if remote_complete:
        assert len(commands.matching("delete-role")) == prior
    assert not value.fleet_master_file.exists(), (
        "finished rollback left its disposable master copy"
    )
    assert (value.site.source.parent / "secure/hp-gpu-b.token").is_file(), (
        "registered rollback discarded its cluster token"
    )


def test_unproven_network_mutation_keeps_compensation_nonterminal(attempt):
    attempt[4]["evidence"]["PREREQUISITES_READY"]["network"]["pending_mutation"] = (
        "vpc-association"
    )
    with pytest.raises(BootstrapError, match="outcome is unproven"):
        compensate(attempt)
    assert attempt[4]["phase"] == "ROLLBACK_FAILED"
    assert attempt[0].token_file.is_file(), (
        "unproven rollback removed its retry credential"
    )


def test_early_join_failure_without_keys_cleans_only_owned_namespace(attempt):
    value, _request, _directory, _path, state = attempt
    state["completed_steps"] = ["PRECHECKED", "DISCOVERED", "LOCAL_INPUTS_STARTED"]
    original = state["evidence"]
    state["evidence"] = {
        "PRECHECKED": {},
        "DISCOVERED": original["DISCOVERED"],
        "LOCAL_INPUTS_STARTED": original["LOCAL_INPUTS_READY"],
    }
    compensate(attempt)
    assert state["phase"] == "ROLLED_BACK"
    assert value.cleared == value.rollouts == []
    assert value.commands.matching("--raw"), "owned early namespace was not cleaned"
    assert not value.commands.matching("delete-role"), (
        "early local failure attempted IAM cleanup"
    )


@pytest.mark.parametrize("kind", ["missing", "different", "symlink"])
def test_registered_rollback_requires_retained_token_identity(attempt, kind):
    value, _request, _directory, _path, state = attempt
    retained = value.site.source.parent / "secure/hp-gpu-b.token"
    if kind == "missing":
        value.token_file.unlink()
    elif kind == "different":
        retained.write_text("x" * 64)
    else:
        retained.symlink_to(value.token_file)
    with pytest.raises(BootstrapError, match="unavailable for retry|conflicts"):
        compensate(attempt)
    assert state["phase"] == "ROLLBACK_FAILED"
    assert "REMOTE_ROLLBACK_COMPLETED" in state["completed_steps"]
