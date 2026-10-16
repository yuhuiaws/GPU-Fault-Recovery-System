"""A first deploy bootstrap would refuse stops before anything is recorded.

Mixed-site BOOT-029 stage 1: a first ``deploy`` given a HyperPod and a plain
EKS GPU ARN wrote ``initial-deploy-request.json`` and, through the email check,
the SNS topic and subscription; bootstrap refused the provider mix only later,
and the corrected rerun with one provider was refused as a conflicting request,
so the run needed a new state directory. The public hop now runs bootstrap's
read-only discovery on the request first (``first_deploy_precheck``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import deploy_command
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.deploy_identity import INITIAL_REQUEST_FILE
from gpu_fault.admin.first_deploy_precheck import NOTHING_CREATED, RECORDED_REQUEST
from tests.admin._first_deploy_discovery import (
    discovered_identity,
    stub_first_deploy_discovery,
)

CPU = "arn:aws:eks:us-east-1:123456789012:cluster/cpu"
HYPERPOD_GPU = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-hp"
PLAIN_GPU = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-eks"
SECOND_PLAIN_GPU = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-eks-2"
EMAIL = "operations@example.com"


def _deploy(state_dir: Path, *gpu_arns: str) -> argparse.Namespace:
    arguments = ["deploy", "--cpu-cluster-arn", CPU]
    for arn in gpu_arns:
        arguments.extend(["--gpu-cluster-arn", arn])
    arguments.extend(["--state-dir", str(state_dir), "--admin-email", EMAIL])
    return admin_cli.parser().parse_args(arguments)


@pytest.fixture
def source_deploys(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """The source preparer the public hop hands the deploy to; its first step
    is the email check that creates the SNS topic and the subscription."""

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        admin_cli, "run_source_deploy", lambda **kwargs: calls.append(kwargs) or 0
    )
    return calls


def _recorded(state_dir: Path) -> dict[str, object] | None:
    path = state_dir / INITIAL_REQUEST_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def test_a_mixed_provider_first_deploy_is_refused_before_anything_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_deploys
) -> None:
    state_dir = tmp_path / "state"
    asked = stub_first_deploy_discovery(
        monkeypatch, providers={PLAIN_GPU: "eks-managed"}
    )

    with pytest.raises(BootstrapError) as refused:
        admin_cli.run(_deploy(state_dir, HYPERPOD_GPU, PLAIN_GPU))

    message = str(refused.value)
    assert "must share one provider" in message
    assert f"hyperpod: {HYPERPOD_GPU}" in message, "the refusal names each provider"
    assert f"eks-managed: {PLAIN_GPU}" in message, "the refusal names each provider"
    assert "join-cluster" in message and "deploy --state-dir" in message, (
        "the refusal names the two-step procedure"
    )
    assert NOTHING_CREATED in message
    assert _recorded(state_dir) is None, "the request was recorded before the refusal"
    assert not (state_dir / "bootstrap-state.json").exists(), (
        "the email check's subscription request was recorded"
    )
    assert source_deploys == [], (
        "the source preparer (and its SNS topic/subscription check) ran"
    )
    assert asked == [(CPU, "cpu"), (HYPERPOD_GPU, "gpu"), (PLAIN_GPU, "gpu")]

    # The corrected rerun in the same state directory is a new request.
    assert admin_cli.run(_deploy(state_dir, HYPERPOD_GPU)) == 0
    assert source_deploys[-1]["gpu_cluster_arns"] == (HYPERPOD_GPU,)
    recorded = _recorded(state_dir)
    assert recorded is not None and recorded["gpu_cluster_arns"] == [HYPERPOD_GPU]


def test_a_single_provider_first_deploy_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_deploys
) -> None:
    state_dir = tmp_path / "state"
    providers = {PLAIN_GPU: "eks-managed", SECOND_PLAIN_GPU: "eks-managed"}
    asked = stub_first_deploy_discovery(monkeypatch, providers=providers)

    assert admin_cli.run(_deploy(state_dir, PLAIN_GPU, SECOND_PLAIN_GPU)) == 0

    assert len(source_deploys) == 1
    call = source_deploys[0]
    assert (call["cpu_cluster_arn"], call["gpu_cluster_arns"], call["admin_email"]) == (
        CPU,
        (PLAIN_GPU, SECOND_PLAIN_GPU),
        EMAIL,
    )
    assert _recorded(state_dir) == {
        "schema_version": 1,
        "state_dir": str(state_dir.resolve()),
        "cpu_cluster_arn": CPU,
        "gpu_cluster_arns": [PLAIN_GPU, SECOND_PLAIN_GPU],
        "admin_email": EMAIL,
    }
    assert sorted(asked) == sorted(
        [(CPU, "cpu"), (PLAIN_GPU, "gpu"), (SECOND_PLAIN_GPU, "gpu")]
    ), "the precheck discovers each cluster once, read-only"


def test_a_discovery_refusal_stops_the_first_deploy_before_the_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_deploys
) -> None:
    """Every refusal discovery makes on a first deploy (here: no node group
    carries the site's ownership label) comes before the request is recorded."""

    state_dir = tmp_path / "state"

    def discover(_runner, *, cluster_arn, role, **_options):
        if role == "gpu":
            raise BootstrapError(
                f"no managed node group of {cluster_arn} carries the ownership label"
            )
        return discovered_identity(cluster_arn, role=role)

    monkeypatch.setattr(admin_cli, "discover_cluster", discover)

    with pytest.raises(BootstrapError, match="carries the ownership label") as refused:
        admin_cli.run(_deploy(state_dir, HYPERPOD_GPU))

    assert NOTHING_CREATED in str(refused.value)
    assert _recorded(state_dir) is None, "the request was recorded before the refusal"
    assert source_deploys == [], "the source preparer ran after a discovery refusal"


def test_one_cluster_named_twice_is_refused_before_the_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_deploys
) -> None:
    """An EKS ARN and the HyperPod ARN of the same cluster are one GPU cluster;
    bootstrap refuses the target as duplicate, so the precheck does first."""

    state_dir = tmp_path / "state"
    stub_first_deploy_discovery(monkeypatch)
    sagemaker_spelling = "arn:aws:sagemaker:us-east-1:123456789012:cluster/gpu-hp"

    with pytest.raises(BootstrapError, match="duplicate GPU clusters") as refused:
        admin_cli.run(_deploy(state_dir, HYPERPOD_GPU, sagemaker_spelling))

    assert HYPERPOD_GPU in str(refused.value) and sagemaker_spelling in str(
        refused.value
    ), "the refusal names both spellings"
    assert _recorded(state_dir) is None, "the request was recorded before the refusal"
    assert source_deploys == [], "the source preparer ran for a duplicate target"


def test_a_recorded_request_is_prechecked_before_it_is_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_deploys
) -> None:
    """A state directory an earlier version pinned to a mixed request is told
    so before the email check and the gates, not after them."""

    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=0o700)
    legacy = {
        "schema_version": 1,
        "state_dir": str(state_dir.resolve()),
        "cpu_cluster_arn": CPU,
        "gpu_cluster_arns": [HYPERPOD_GPU, PLAIN_GPU],
        "admin_email": EMAIL,
    }
    path = state_dir / INITIAL_REQUEST_FILE
    path.write_text(json.dumps(legacy), encoding="utf-8")
    path.chmod(0o600)
    stub_first_deploy_discovery(monkeypatch, providers={PLAIN_GPU: "eks-managed"})

    with pytest.raises(BootstrapError, match="must share one provider") as refused:
        admin_cli.run(_deploy(state_dir, HYPERPOD_GPU, PLAIN_GPU))

    assert RECORDED_REQUEST in str(refused.value)
    assert NOTHING_CREATED not in str(refused.value), (
        "a recorded request cannot be called unrecorded"
    )
    assert _recorded(state_dir) == legacy
    assert source_deploys == [], "the source preparer ran for a refused resume"


def test_the_prepared_hop_does_not_discover_the_request_again(tmp_path: Path) -> None:
    """The inner hop's bootstrap discovers the clusters itself; the precheck
    belongs to the public hop only."""

    def discover(*_args, **_options):
        raise AssertionError("the prepared hop ran the first-deploy precheck")

    hooks = SimpleNamespace(discover_cluster=discover)
    arguments = argparse.Namespace(
        cpu_cluster_arn=CPU,
        gpu_cluster_arn=[HYPERPOD_GPU, PLAIN_GPU],
        alert_email=EMAIL,
        prepared_source_release=True,
    )

    resolved = deploy_command.resolve_deploy_identity(
        arguments,
        tmp_path / "state",
        hooks=hooks,  # type: ignore[arg-type]
    )

    assert resolved == (CPU, (HYPERPOD_GPU, PLAIN_GPU), EMAIL, ())
