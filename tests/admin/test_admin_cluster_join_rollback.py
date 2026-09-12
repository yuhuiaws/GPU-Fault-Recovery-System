"""Compensation when a join fails before activation.

``test_admin_cluster_join.py`` stubs ``_rollback`` out to prove the join state
machine calls it. These tests let the real one run: every resource the attempt
created has a matching undo, an undo that finds the resource already gone is not
an error, and an undo that genuinely fails leaves the evidence in place instead of
reporting a clean rollback.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pytest
import yaml

from gpu_fault.admin import cluster_join as admin_cluster_join
from gpu_fault.admin import cluster_join_commit as admin_cluster_join_commit
from gpu_fault.admin.bootstrap_common import BootstrapError, ClusterIdentity
from gpu_fault.admin.cluster_join import JoinClusterRequest, join_cluster
from gpu_fault.admin.site import load_site
from tests.admin.test_admin_site import site_file

GPU_B_ARN = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-b"
ROLE_ARN = "arn:aws:iam::123456789012:role/gpu-fault-gpu-b-executor"
INGRESS_EIP = "192.0.2.20"
HOSTED_ZONE = "Z123"


def _target() -> ClusterIdentity:
    return ClusterIdentity(
        input_arn=GPU_B_ARN,
        role="gpu",
        region="us-east-1",
        account_id="123456789012",
        hyperpod_arn="arn:aws:sagemaker:us-east-1:123456789012:cluster/hp-gpu-b",
        hyperpod_name="hp-gpu-b",
        eks_arn=GPU_B_ARN,
        eks_name="gpu-b",
        vpc_id="vpc-gpu-b",
        subnet_ids=("subnet-b",),
        node_recovery="None",
        context="gpu-fault-gpu-2-gpu-b",
    )


class Commands:
    """Stands in for every external process the rollback shells out to.

    ``failures`` maps a fragment of the command line to the result it returns, so
    a test can fail exactly one undo and leave the rest working.
    """

    def __init__(
        self,
        *,
        failures: Sequence[tuple[str, int, str]] = (),
        outputs: Sequence[tuple[str, str]] = (),
    ) -> None:
        self.failures = tuple(failures)
        self.outputs = tuple(outputs)
        self.calls: list[list[str]] = []

    def __call__(self, arguments: Sequence[Any], **_keywords: Any) -> SimpleNamespace:
        argv = [str(item) for item in arguments]
        self.calls.append(argv)
        line = " ".join(argv)
        for fragment, returncode, stderr in self.failures:
            if fragment in line:
                return SimpleNamespace(returncode=returncode, stdout="", stderr=stderr)
        for fragment, stdout in self.outputs:
            if fragment in line:
                return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def matching(self, fragment: str) -> list[list[str]]:
        return [argv for argv in self.calls if fragment in " ".join(argv)]


class Attempt:
    """A join that has reached ``JOINED`` and is about to fail.

    The evidence written here is the same shape ``_prepare_execution`` records,
    because the rollback reads nothing else: what it undoes is decided entirely by
    what the attempt claimed to have created.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.path = site_file(tmp_path)
        self.site = load_site(self.path)
        self.state_dir = tmp_path / "join-state"
        self.secure = self.state_dir / "secure"
        self.secure.mkdir(mode=0o700, parents=True)
        self.token_file = self.secure / "hp-gpu-b.token"
        self.fleet_master_file = self.secure / "fleet-master"
        for path in (self.token_file, self.fleet_master_file):
            path.write_text("f" * 64, encoding="utf-8")
            path.chmod(0o600)
        self.candidate_path = self._write_candidate()
        self.commands = Commands()
        self.rollouts: list[tuple[str, str | None]] = []
        self.membership: list[bool] = []
        self.cleared: list[list[str]] = []
        self.joined = True

    def _write_candidate(self) -> Path:
        target = _target()
        document = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        first = document["spec"]["clusters"][0]
        document["spec"]["clusters"].append(
            {
                "clusterId": "hp-gpu-b",
                "context": target.context,
                "region": target.region,
                "hyperpodClusterName": target.hyperpod_name,
                "eksClusterArn": target.eks_arn,
                "executorIrsaRoleArn": ROLE_ARN,
                "allowedNamespaces": ["gpu-fault-system", "training"],
                "controlPlaneUrl": "https://control.example",
                "tokenFile": str(self.token_file),
                "caFile": first["caFile"],
                "fleetMasterFile": str(self.fleet_master_file),
            }
        )
        path = self.state_dir / "candidate-site-001.yaml"
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        path.chmod(0o600)
        return path

    def evidence(self) -> dict[str, Any]:
        return {
            "PRECHECKED": {},
            "DISCOVERED": {
                "target": asdict(_target()),
                "cluster_id": "hp-gpu-b",
                "registry_snapshot": str(
                    self.state_dir / "installation-resources-before-001.json"
                ),
            },
            "LOCAL_INPUTS_READY": {
                "gpu_kubeconfig": str(self.state_dir / "gpu.kubeconfig"),
                "token_file": str(self.token_file),
                "fleet_master_file": str(self.fleet_master_file),
                "ca_file": self.site.release_config["clusters"][0]["ca_file"],
                "nodes": ["node-b"],
            },
            "PREREQUISITES_READY": {
                "executor_role": {
                    "role_arn": ROLE_ARN,
                    "ownership": "CREATED",
                    "inline_policy_name": "GPUFaultRegionalExecutor",
                },
                "network": {
                    "vpc_id": "vpc-gpu-b",
                    "nat_eips": [INGRESS_EIP],
                    "created_ingress_eips": [INGRESS_EIP],
                    "existing_vpc_ids": ["vpc-gpu-a"],
                    "hosted_zone_id": HOSTED_ZONE,
                    "association_created": True,
                },
                "node_keys": {"cluster_id": "hp-gpu-b"},
            },
            "CANDIDATE_READY": {"site_file": str(self.candidate_path)},
        }

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def prepare(_request: Any, **keywords: Any) -> Any:
            state = keywords["state"]
            state["completed_steps"] = [
                "PRECHECKED",
                "DISCOVERED",
                "LOCAL_INPUTS_READY",
                "PREREQUISITES_READY",
                "CANDIDATE_READY",
                *(["JOINED"] if self.joined else []),
            ]
            state["evidence"] = self.evidence()
            raise BootstrapError("cluster deploy failed")

        monkeypatch.setattr(admin_cluster_join, "_prepare_execution", prepare)
        monkeypatch.setattr(
            admin_cluster_join,
            "_run_rollout",
            lambda _site, mode, *, cluster_id=None: self.rollouts.append(
                (mode, cluster_id)
            ),
        )
        monkeypatch.setattr(
            admin_cluster_join,
            "_clear_installer_annotations",
            lambda _site, _target, nodes: self.cleared.append(list(nodes)),
        )
        monkeypatch.setattr(
            admin_cluster_join,
            "_remove_node_action_keys",
            lambda *_args, **_keywords: None,
        )
        monkeypatch.setattr(
            admin_cluster_join, "_wait_vpc_association_absent", lambda **_keywords: None
        )
        monkeypatch.setattr(
            admin_cluster_join_commit,
            "rollback_membership",
            lambda _request, *, execution, joined: self.membership.append(joined),
        )
        monkeypatch.setattr(admin_cluster_join.subprocess, "run", self.commands)

    def run(self, monkeypatch: pytest.MonkeyPatch, *, error: str) -> None:
        self.install(monkeypatch)
        with pytest.raises(BootstrapError, match=error):
            join_cluster(
                JoinClusterRequest(
                    site=self.site, gpu_cluster_arn=GPU_B_ARN, state_dir=self.state_dir
                ),
                runner=SimpleNamespace(dry_run=False),
            )

    def state(self) -> dict[str, Any]:
        return json.loads((self.state_dir / "state.json").read_text(encoding="utf-8"))


def test_a_failure_before_activation_undoes_everything_the_attempt_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each recorded resource has one undo, and the site keeps its own cluster.

    A join that stops halfway leaves an IAM role, an NLB ingress rule, a hosted
    zone association, a namespace and two secret files behind. Leaving any of them
    would make the next attempt reuse a resource this attempt half-configured, and
    the ingress rule in particular is a live hole in the control plane's NLB.
    """

    attempt = Attempt(tmp_path)

    attempt.run(monkeypatch, error="cluster deploy failed")

    assert attempt.rollouts == [("fail-cluster", "hp-gpu-b")]
    assert attempt.cleared == [["node-b"]]
    assert attempt.membership == [True]
    assert attempt.commands.matching("prepare-clean-redeploy.sh"), (
        "the GPU-side Kubernetes objects were not cleaned up"
    )
    revoke = attempt.commands.matching("revoke-security-group-ingress")
    assert [argv for argv in revoke if f"{INGRESS_EIP}/32" in argv], (
        "the NLB ingress rule created for this cluster was left in place"
    )
    assert attempt.commands.matching("disassociate-vpc-from-hosted-zone"), (
        "the private hosted zone stayed associated with the candidate VPC"
    )
    assert attempt.commands.matching("delete-role-policy"), (
        "the executor inline policy was left on the role"
    )
    assert attempt.commands.matching("delete-role"), (
        "the executor IAM role was left in the account"
    )
    assert attempt.commands.matching("delete-context"), (
        "the kubeconfig context for the candidate cluster was left behind"
    )
    namespace = attempt.commands.matching("delete namespace")
    assert namespace and "--ignore-not-found" in namespace[0]
    assert not attempt.token_file.exists(), "the join token file was left on disk"
    assert not attempt.fleet_master_file.exists(), (
        "the fleet master secret was left on disk"
    )
    assert len(load_site(attempt.path).release_config["clusters"]) == 1


def test_a_rollback_repoints_the_current_context_it_left_dangling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``update-kubeconfig --alias`` made the candidate current; undo that too.

    Deleting only the context left ``current-context`` naming a context that no
    longer exists, so every ``kubectl --kubeconfig gpu.kubeconfig`` call without
    ``--context`` failed after a rolled-back join (live, 2026-09-12). The first
    surviving context takes over.
    """

    attempt = Attempt(tmp_path)
    attempt.commands = Commands(
        outputs=(
            ("config current-context", "gpu-fault-gpu-2-gpu-b\n"),
            (
                "config get-contexts -o name",
                "gpu-fault-gpu-2-gpu-b\ngpu-fault-gpu-1-gpu-a\n",
            ),
        )
    )

    attempt.run(monkeypatch, error="cluster deploy failed")

    assert attempt.commands.matching("delete-context gpu-fault-gpu-2-gpu-b"), (
        "the candidate context was not deleted"
    )
    assert attempt.commands.matching("use-context gpu-fault-gpu-1-gpu-a"), (
        "current-context was left pointing at the deleted context"
    )


def test_a_rollback_unsets_the_current_context_when_no_context_remains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    attempt.commands = Commands(
        outputs=(
            ("config current-context", "gpu-fault-gpu-2-gpu-b\n"),
            ("config get-contexts -o name", "gpu-fault-gpu-2-gpu-b\n"),
        )
    )

    attempt.run(monkeypatch, error="cluster deploy failed")

    assert attempt.commands.matching("unset current-context"), (
        "a dangling current-context with no other context must be unset"
    )
    assert not attempt.commands.matching("use-context"), (
        "there was no surviving context to make current"
    )


def test_a_rollback_that_completes_clears_the_attempt_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the read-only steps survive a completed rollback.

    A resumed join must re-create what was undone; keeping ``PREREQUISITES_READY``
    would make the next attempt trust an IAM role that no longer exists.
    """

    attempt = Attempt(tmp_path)

    attempt.run(monkeypatch, error="cluster deploy failed")
    state = attempt.state()

    assert state["phase"] == "ROLLED_BACK"
    assert state["completed_steps"] == ["PRECHECKED", "DISCOVERED"]
    assert set(state["evidence"]) == {"PRECHECKED", "DISCOVERED"}
    assert "rollback_errors" not in state


def test_a_join_that_never_reached_joined_does_not_fail_the_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``fail-cluster`` is only meaningful once the cluster was actually joined.

    Running it against a cluster the control plane has never seen turns a clean
    rollback into a rollout error, which would then be reported as
    ``ROLLBACK_FAILED``.
    """

    attempt = Attempt(tmp_path)
    attempt.joined = False

    attempt.run(monkeypatch, error="cluster deploy failed")

    assert attempt.rollouts == []
    assert attempt.membership == [False]
    assert attempt.state()["phase"] == "ROLLED_BACK"


@pytest.mark.parametrize(
    "failure",
    [
        ("delete-role-policy", 254, "NoSuchEntity: cannot be found"),
        ("delete-role", 254, "NoSuchEntity: cannot be found"),
        ("revoke-security-group-ingress", 254, "InvalidPermission.NotFound"),
        ("disassociate-vpc-from-hosted-zone", 254, "VPCAssociationNotFound"),
        ("delete-context", 1, 'context "gpu-fault-gpu-2-gpu-b" not found'),
    ],
)
def test_a_resource_that_is_already_gone_is_not_a_rollback_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: tuple[str, int, str]
) -> None:
    """Rollback has to be re-runnable after a partial rollback.

    Every undo here is idempotent, so "it is not there" is the desired end state.
    Reporting ``ROLLBACK_FAILED`` for it would strand the attempt's evidence and
    block the retry that is the whole point of the state machine.
    """

    attempt = Attempt(tmp_path)
    attempt.commands = Commands(failures=[failure])

    attempt.run(monkeypatch, error="cluster deploy failed")

    assert attempt.state()["phase"] == "ROLLED_BACK"


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (
            ("prepare-clean-redeploy.sh", 1, "helm uninstall timed out"),
            "kubernetes/control rollback",
        ),
        (
            ("delete namespace", 1, "Error from server: etcdserver: timeout"),
            "namespace rollback",
        ),
        (("revoke-security-group-ingress", 254, "AccessDenied"), "network rollback"),
        (("delete-role-policy", 254, "AccessDenied"), "Executor policy rollback"),
        (("delete-role", 254, "AccessDenied"), "Executor role rollback"),
        (
            ("delete-context", 1, "error: open kubeconfig: permission denied"),
            "kube context rollback",
        ),
    ],
)
def test_an_undo_that_really_failed_is_reported_and_keeps_the_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: tuple[str, int, str],
    message: str,
) -> None:
    """A failed undo must not be reported as a clean rollback.

    ``ROLLED_BACK`` tells the operator the cluster is untouched and the attempt can
    be retried. If an undo failed, that is false: the leftover resource is named in
    ``rollback_errors`` and the evidence is kept so the retry can find it.
    """

    attempt = Attempt(tmp_path)
    attempt.commands = Commands(failures=[failure])

    attempt.run(monkeypatch, error=message)
    state = attempt.state()

    assert state["phase"] == "ROLLBACK_FAILED"
    assert any(message in item for item in state["rollback_errors"]), (
        f"the failed undo was not named in rollback_errors: {state['rollback_errors']}"
    )
    assert "PREREQUISITES_READY" in state["evidence"]
    assert attempt.membership == [], (
        "membership rollback ran on top of a failed resource rollback"
    )


def test_a_retry_after_rollback_starts_a_fresh_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read-only steps of a rolled-back attempt are not worth resuming.

    Live, 2026-09-12: the first join rolled back because discovery had derived
    Agent CIDRs that covered no node. The fix changed discovery, yet a retry
    would have replayed the recorded ``DISCOVERED`` target, and the deploy that
    shipped the fix had moved the site's ``repositoryRoot``, which the drift
    guard treated as a conflict. A rolled-back attempt is over: the retry
    archives it and rediscovers from scratch.
    """

    attempt = Attempt(tmp_path)
    attempt.run(monkeypatch, error="cluster deploy failed")
    assert attempt.state()["phase"] == "ROLLED_BACK", "the first attempt must roll back"
    state_path = attempt.state_dir / "state.json"
    recorded = attempt.state()
    recorded["source_site_non_membership_sha256"] = "0" * 64
    state_path.write_text(json.dumps(recorded), encoding="utf-8")

    seen: list[dict[str, Any]] = []

    def prepare(_request: Any, **keywords: Any) -> Any:
        seen.append(json.loads(json.dumps(keywords["state"])))
        raise BootstrapError("stop after reset")

    monkeypatch.setattr(admin_cluster_join, "_prepare_execution", prepare)
    with pytest.raises(BootstrapError, match="stop after reset"):
        join_cluster(
            JoinClusterRequest(
                site=attempt.site,
                gpu_cluster_arn=GPU_B_ARN,
                state_dir=attempt.state_dir,
            ),
            runner=SimpleNamespace(dry_run=False),
        )

    assert seen and seen[0]["attempt"] == 2, "the retry must open a new attempt"
    assert seen[0]["completed_steps"] == [], (
        "the retry must not resume the rolled-back attempt's discovery"
    )
    assert (attempt.state_dir / "state.attempt-001.json").exists(), (
        "the rolled-back attempt must be archived, not overwritten"
    )


EMPTY_REGISTRY = json.dumps(
    {
        "schema_version": 1,
        "cpu": {"resources": [{"kind": "deployment", "name": "gpu-fault-api"}]},
        "gpu": {"resources": []},
        "unregistered_resources": [],
    }
)


def test_a_release_that_installed_nothing_needs_no_gpu_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cleanup script refuses an empty inventory; do not ask it to clean one.

    Live, 2026-09-12: the release failed at the endpoint gate, before any GPU
    resource existed, and the rollback then failed on "cleanup resource
    inventory gpu.resources is empty" -- leaving the attempt ROLLBACK_FAILED
    and the membership undo skipped. When the installed-resource registry
    shows nothing on the GPU plane, the rollback skips the script.
    """

    attempt = Attempt(tmp_path)
    attempt.commands = Commands(
        outputs=(("collect_installed_resource_registry.py", EMPTY_REGISTRY),)
    )

    attempt.run(monkeypatch, error="cluster deploy failed")

    assert attempt.state()["phase"] == "ROLLED_BACK", "the rollback must complete"
    assert attempt.commands.matching("collect_installed_resource_registry.py"), (
        "the registry was not consulted"
    )
    assert not attempt.commands.matching("prepare-clean-redeploy.sh"), (
        "the cleanup script was run against an empty inventory"
    )
    assert attempt.membership == [True], "the membership undo must still run"


def test_a_retry_after_a_failed_rollback_finishes_the_undo_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ROLLBACK_FAILED`` keeps its evidence so the retry can finish the undo.

    The retry must not resume the attempt (its IAM roles and zone association
    are already gone) and must not open a new one on top of the leftovers
    (live, 2026-09-12: the control plane still listed the cluster as a
    member). It re-runs the rollback; once that is clean the attempt is
    archived and discovery starts over.
    """

    attempt = Attempt(tmp_path)
    attempt.commands = Commands(
        failures=[("prepare-clean-redeploy.sh", 1, "cleanup refused")]
    )
    attempt.run(monkeypatch, error="cleanup refused")
    assert attempt.state()["phase"] == "ROLLBACK_FAILED", "the first undo must fail"
    assert attempt.membership == [], "membership must stay until the undo is clean"
    # The deploy that ships a fix moves the site's repositoryRoot between the
    # attempts (live, 2026-09-12); a finished attempt is not in flight, so that
    # drift must not block the undo.
    state_path = attempt.state_dir / "state.json"
    recorded = attempt.state()
    recorded["source_site_non_membership_sha256"] = "0" * 64
    state_path.write_text(json.dumps(recorded), encoding="utf-8")

    attempt.commands = Commands()
    attempt.install(monkeypatch)
    seen: list[dict[str, Any]] = []

    def prepare(_request: Any, **keywords: Any) -> Any:
        seen.append(json.loads(json.dumps(keywords["state"])))
        raise BootstrapError("stop after reset")

    monkeypatch.setattr(admin_cluster_join, "_prepare_execution", prepare)
    with pytest.raises(BootstrapError, match="stop after reset"):
        join_cluster(
            JoinClusterRequest(
                site=attempt.site,
                gpu_cluster_arn=GPU_B_ARN,
                state_dir=attempt.state_dir,
            ),
            runner=SimpleNamespace(dry_run=False),
        )

    assert attempt.commands.matching("prepare-clean-redeploy.sh"), (
        "the retry did not re-run the failed undo"
    )
    assert attempt.membership == [True], "the retry must finish the membership undo"
    assert seen and seen[0]["attempt"] == 2, "a clean undo must open a fresh attempt"
    assert seen[0]["completed_steps"] == [], "the fresh attempt must rediscover"


def test_a_retry_whose_undo_still_fails_stops_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    attempt.commands = Commands(
        failures=[("prepare-clean-redeploy.sh", 1, "cleanup refused")]
    )
    attempt.run(monkeypatch, error="cleanup refused")

    attempt.install(monkeypatch)
    monkeypatch.setattr(
        admin_cluster_join,
        "_prepare_execution",
        lambda *_args, **_keywords: pytest.fail("the join must not proceed"),
    )
    with pytest.raises(BootstrapError, match="cleanup refused"):
        join_cluster(
            JoinClusterRequest(
                site=attempt.site,
                gpu_cluster_arn=GPU_B_ARN,
                state_dir=attempt.state_dir,
            ),
            runner=SimpleNamespace(dry_run=False),
        )

    assert attempt.state()["phase"] == "ROLLBACK_FAILED", "the evidence must be kept"
    assert attempt.membership == [], "membership must not be undone over leftovers"


def test_drift_still_blocks_an_attempt_that_is_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = Attempt(tmp_path)
    attempt.run(monkeypatch, error="cluster deploy failed")
    state_path = attempt.state_dir / "state.json"
    recorded = attempt.state()
    recorded["phase"] = "CANDIDATE_READY"
    recorded["source_site_non_membership_sha256"] = "0" * 64
    state_path.write_text(json.dumps(recorded), encoding="utf-8")

    with pytest.raises(BootstrapError, match="non-membership fields drifted"):
        join_cluster(
            JoinClusterRequest(
                site=attempt.site,
                gpu_cluster_arn=GPU_B_ARN,
                state_dir=attempt.state_dir,
            ),
            runner=SimpleNamespace(dry_run=False),
        )
