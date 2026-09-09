from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.admin import bootstrap_checkpoint as admin_bootstrap_checkpoint
from gpu_fault.admin import bootstrap_load_balancer as admin_bootstrap_load_balancer
from gpu_fault.admin import bootstrap_platform_probes as admin_bootstrap_platform_probes
from gpu_fault.admin import bootstrap_services as admin_bootstrap_services
from gpu_fault.admin.bootstrap_checkpoint import bind_bootstrap_inputs
from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    BootstrapMutationRequired,
    BootstrapRequest,
    BootstrapState,
    ClusterIdentity,
    ReadOnlyProbeRunner,
    compute_agent_config_digest,
    run_parallel,
)


def _cluster() -> ClusterIdentity:
    return ClusterIdentity(
        input_arn="arn:aws:eks:us-east-1:123456789012:cluster/control",
        role="cpu",
        region="us-east-1",
        account_id="123456789012",
        hyperpod_arn=("arn:aws:sagemaker:us-east-1:123456789012:cluster/control"),
        hyperpod_name="control",
        eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/control",
        eks_name="control",
        vpc_id="vpc-control",
        subnet_ids=("subnet-private-a", "subnet-private-b"),
        node_recovery="None",
        context="control",
    )


def test_agent_config_digest_uses_installer_environment(tmp_path: Path) -> None:
    calls = []
    installer_environment = {
        "GPU_FAULT_PYTHON_STACK_TOOL": "/opt/gpu-fault/tools/py-spy-current",
        "GPU_FAULT_QUIESCE_RESTORE_COMMAND": (
            "/opt/gpu-fault/current/venv/bin/gpu-fault-restore-gpu-services"
        ),
    }

    class Runner:
        def run(self, arguments, **kwargs):
            calls.append((arguments, kwargs))
            if arguments[0] == "bash":
                return json.dumps(installer_environment)
            assert (
                kwargs["env"]["GPU_FAULT_PYTHON_STACK_TOOL"]
                == installer_environment["GPU_FAULT_PYTHON_STACK_TOOL"]
            )
            assert (
                kwargs["env"]["GPU_FAULT_QUIESCE_RESTORE_COMMAND"]
                == installer_environment["GPU_FAULT_QUIESCE_RESTORE_COMMAND"]
            )
            return "a" * 64

    digest = compute_agent_config_digest(
        Runner(), repository_root=tmp_path, runtime_profile_version="profile-a"
    )

    assert digest == "a" * 64
    assert calls[0][0] == [
        "bash",
        str(tmp_path / "deploy/node/install-gpu-fault-collector.sh"),
        "--print-config-digest-environment",
    ]


def test_agent_config_digest_rejects_incomplete_installer_environment(
    tmp_path: Path,
) -> None:
    class Runner:
        def run(self, _arguments, **_kwargs):
            return json.dumps({"GPU_FAULT_PYTHON_STACK_TOOL": "/usr/bin/py-spy"})

    with pytest.raises(BootstrapError, match="unexpected keys"):
        compute_agent_config_digest(
            Runner(), repository_root=tmp_path, runtime_profile_version="profile-a"
        )


def test_parallel_bootstrap_persists_successes_when_another_task_fails(
    tmp_path: Path,
) -> None:
    state = BootstrapState(tmp_path / "state.json", site_id="test")

    def fail():
        raise BootstrapError("aurora failed")

    with pytest.raises(BootstrapError, match="aurora failed"):
        run_parallel(
            {"aurora": fail, "pki": lambda: {"certificate_arn": "arn:certificate"}},
            state=state,
        )

    reloaded = BootstrapState(tmp_path / "state.json", site_id="test")
    assert reloaded.value["completed_tasks"] == ["pki"]
    assert reloaded.value["resources"]["pki"] == {"certificate_arn": "arn:certificate"}


def test_parallel_bootstrap_probe_reuses_healthy_completed_task(tmp_path: Path) -> None:
    state = BootstrapState(tmp_path / "state.json", site_id="test")
    state.record("load_balancer_controller", {"generation": 1})
    state.complete("load_balancer_controller")
    calls = []

    result = run_parallel(
        {
            "load_balancer_controller": lambda: (
                calls.append("ensured") or {"generation": 2}
            )
        },
        state=state,
        probes={
            "load_balancer_controller": lambda: (
                calls.append("probed") or {"generation": 99}
            )
        },
        revalidate=frozenset({"load_balancer_controller"}),
    )

    assert calls == ["probed"]
    assert result["load_balancer_controller"] == {"generation": 1}
    assert state.value["resources"]["load_balancer_controller"] == {"generation": 1}


def test_healthy_iam_role_probe_performs_no_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust = admin_bootstrap_services.pod_identity_trust(_cluster())
    policy = {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Action": "example:Read", "Resource": "*"}],
    }
    calls = []

    class Runner:
        dry_run = False

        def run(self, arguments, **kwargs):
            calls.append((list(arguments), kwargs))
            if "get-role-policy" in arguments:
                return json.dumps({"PolicyDocument": policy})
            if "get-role" in arguments:
                return json.dumps(
                    {
                        "Role": {
                            "AssumeRolePolicyDocument": trust,
                            "Tags": [
                                {
                                    "Key": admin_bootstrap_services.SITE_TAG_KEY,
                                    "Value": "site-a",
                                }
                            ],
                        }
                    }
                )
            raise AssertionError(arguments)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stderr=""),
    )

    result = admin_bootstrap_services.probe_iam_role(
        Runner(),
        account_id="123456789012",
        role_name="gpu-fault-role",
        trust=trust,
        policy_name="ReadOnly",
        policy=policy,
        site_id="site-a",
    )

    assert result["role_arn"].endswith("/gpu-fault-role"), (
        "IAM role probe returned the wrong role identity"
    )
    assert all(not options.get("mutate") for _command, options in calls), (
        "healthy IAM role probe attempted mutation"
    )


def test_healthy_pod_identity_association_probe_performs_no_mutation() -> None:
    calls = []
    cluster = _cluster()
    role_arn = "arn:aws:iam::123456789012:role/gpu-fault-role"

    class Runner:
        dry_run = False

        def run(self, arguments, **kwargs):
            calls.append((list(arguments), kwargs))
            if "list-pod-identity-associations" in arguments:
                return json.dumps({"associations": [{"associationId": "assoc-a"}]})
            if "describe-pod-identity-association" in arguments:
                return json.dumps(
                    {"association": {"associationId": "assoc-a", "roleArn": role_arn}}
                )
            raise AssertionError(arguments)

    result = admin_bootstrap_services.probe_pod_identity_association(
        Runner(),
        cluster=cluster,
        namespace="gpu-fault-system",
        service_account="gpu-fault-control-plane",
        role_arn=role_arn,
    )

    assert result["association_id"] == "assoc-a"
    assert all(not options.get("mutate") for _command, options in calls), (
        "healthy Pod Identity association probe attempted mutation"
    )


def test_healthy_lbc_probe_skips_helm_and_iam_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cpu = _cluster()
    policy_arn = "arn:aws:iam::123456789012:policy/gpu-fault-site-a-lbc-policy"
    calls = []

    class Runner:
        dry_run = False

        def run(self, arguments, **kwargs):
            calls.append((list(arguments), kwargs))
            if "get-policy" in arguments:
                # One read carries the tags; there is no list-policy-tags call.
                return json.dumps(
                    {
                        "Policy": {
                            "Arn": policy_arn,
                            "Tags": [
                                {
                                    "Key": admin_bootstrap_services.SITE_TAG_KEY,
                                    "Value": "site-a",
                                }
                            ],
                        }
                    }
                )
            if "list-attached-role-policies" in arguments:
                return json.dumps({"AttachedPolicies": [{"PolicyArn": policy_arn}]})
            raise AssertionError(arguments)

    monkeypatch.setattr(
        admin_bootstrap_load_balancer, "_lbc_release_exists", lambda _kubeconfig: True
    )
    monkeypatch.setattr(
        admin_bootstrap_load_balancer, "_controller_ready", lambda *_args: True
    )
    monkeypatch.setattr(
        admin_bootstrap_load_balancer,
        "_lbc_release_matches",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        admin_bootstrap_load_balancer,
        "_ensure_pod_identity_agent",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        admin_bootstrap_load_balancer,
        "_ensure_role",
        lambda *_args, **_kwargs: {
            "role_arn": "arn:aws:iam::123456789012:role/gpu-fault-site-a-lbc",
            "ownership": "CREATED",
        },
    )
    monkeypatch.setattr(
        admin_bootstrap_load_balancer,
        "_ensure_service_account",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        admin_bootstrap_load_balancer,
        "_ensure_pod_identity_association",
        lambda *_args, **_kwargs: {
            "association_id": "assoc-a",
            "ownership": "CREATED",
            "cluster_name": cpu.eks_name,
            "namespace": "kube-system",
            "service_account": "aws-load-balancer-controller",
        },
    )
    monkeypatch.setattr(
        admin_bootstrap_load_balancer.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stderr=""),
    )

    result = admin_bootstrap_load_balancer.ensure_load_balancer_controller(
        ReadOnlyProbeRunner(Runner()),
        cpu=cpu,
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        state_dir=tmp_path,
        site_id="site-a",
    )

    assert result["reused"] is True
    assert all("upgrade" not in command for command, _options in calls), (
        "healthy LBC probe invoked Helm upgrade"
    )
    assert all(not options.get("mutate") for _command, options in calls), (
        "healthy LBC probe attempted mutation"
    )


def test_monitoring_resources_create_no_alerts_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The alerts SQS queue had no consumer; bootstrap creates the topic only.

    Every AWS call the monitoring task makes is recorded: none may name the
    ``sqs`` service, and the only subscription the topic gets is the email one.
    """

    cpu = _cluster()
    topic_arn = "arn:aws:sns:us-east-1:123456789012:gpu-fault-site-a-alerts"
    calls: list[list[str]] = []

    class Runner:
        dry_run = False

        def run(self, arguments, **kwargs):
            del kwargs
            calls.append(list(arguments))
            if "list-subscriptions-by-topic" in arguments:
                return json.dumps({"Subscriptions": []})
            raise AssertionError(arguments)

    monkeypatch.setattr(
        admin_bootstrap_services,
        "_ensure_amp_workspace",
        lambda _runner, **_kwargs: ("ws-a", True),
    )
    monkeypatch.setattr(
        admin_bootstrap_services,
        "ensure_sns_topic",
        lambda _runner, **_kwargs: (topic_arn, True, "a" * 32),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("monitoring bootstrap shelled out"),
    )

    result = admin_bootstrap_services.ensure_monitoring_resources(
        Runner(), state=None, cpu=cpu, site_id="site-a", alert_email=None
    )

    assert not any("sqs" in call for call in calls), (
        f"monitoring bootstrap issued an SQS call: {calls}"
    )
    assert not any("subscribe" in call for call in calls), (
        f"monitoring bootstrap subscribed something other than the email: {calls}"
    )
    assert not any(key.startswith(("sqs_", "queue_")) for key in result), (
        f"monitoring result still carries queue keys: {sorted(result)}"
    )
    assert result["sns_topic_arn"] == topic_arn
    assert result["email_subscription_arn"] is None


def test_parallel_bootstrap_ensures_only_after_probe_detects_drift(
    tmp_path: Path,
) -> None:
    state = BootstrapState(tmp_path / "state.json", site_id="test")
    state.record("load_balancer_controller", {"generation": 1})
    state.complete("load_balancer_controller")
    calls = []

    def probe():
        calls.append("probed")
        raise BootstrapMutationRequired("kubectl")

    result = run_parallel(
        {
            "load_balancer_controller": lambda: (
                calls.append("ensured") or {"generation": 2}
            )
        },
        state=state,
        probes={"load_balancer_controller": probe},
        revalidate=frozenset({"load_balancer_controller"}),
    )

    assert calls == ["probed", "ensured"]
    assert result["load_balancer_controller"] == {"generation": 2}


def test_read_only_probe_runner_blocks_mutation() -> None:
    calls = []

    class Runner:
        dry_run = False

        def run(self, arguments, **kwargs):
            calls.append((arguments, kwargs))
            return "read-result"

    probe = ReadOnlyProbeRunner(Runner())

    assert probe.run(["aws", "read"]) == "read-result"
    with pytest.raises(BootstrapMutationRequired):
        probe.run(["aws", "write"], mutate=True)
    assert len(calls) == 1


def _gpu_cluster() -> ClusterIdentity:
    return ClusterIdentity(
        input_arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
        role="gpu",
        region="us-east-1",
        account_id="123456789012",
        hyperpod_arn="arn:aws:sagemaker:us-east-1:123456789012:cluster/gpu-a",
        hyperpod_name="gpu-a",
        eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",
        eks_name="gpu-a",
        vpc_id="vpc-gpu-a",
        subnet_ids=("subnet-gpu-a",),
        node_recovery="None",
        context="gpu-a",
    )


def _mirror_repository_root(tmp_path: Path, name: str) -> Path:
    """Copy the files the input digests read into a writable repository root.

    The digests must react to the bytes of the scripts and Manifests the tasks
    apply, so a test that claims a digest tracks an asset edits that asset here
    instead of replacing the hash helper.
    """

    root = tmp_path / name
    real = Path(__file__).resolve().parents[2]
    for relative in (
        *admin_bootstrap_checkpoint.BOOTSTRAP_RECONCILE_SOURCES,
        *admin_bootstrap_checkpoint.BOOTSTRAP_TASK_ASSETS,
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((real / relative).read_bytes())
    return root


def _bind_release_inputs(
    tmp_path: Path,
    *,
    release_id: str,
    wheel_bytes: bytes = b"wheel-bytes",
    repository_root: Path | None = None,
) -> dict[str, str]:
    root = repository_root or _mirror_repository_root(tmp_path, "repository")
    state_dir = tmp_path / f"state-{release_id}-{len(wheel_bytes)}"
    state_dir.mkdir(exist_ok=True)
    wheel = tmp_path / "gpu_fault-0.1.0-py3-none-any.whl"
    wheel.write_bytes(wheel_bytes)
    manifest = state_dir / "release.json"
    manifest.write_text(
        json.dumps({"release_id": release_id, "wheel": str(wheel)}), encoding="utf-8"
    )
    state = BootstrapState(state_dir / "bootstrap-state.json", site_id="site-a")
    request = BootstrapRequest(
        cpu_cluster_arn="arn:aws:eks:us-east-1:123456789012:cluster/control",
        gpu_cluster_arns=("arn:aws:eks:us-east-1:123456789012:cluster/gpu-a",),
        repository_root=root,
        state_dir=state_dir,
        alert_email="ops@example.com",
    )
    bind_bootstrap_inputs(
        state,
        request=request,
        cpu=_cluster(),
        gpu_clusters=(_gpu_cluster(),),
        release={
            "release_id": release_id,
            "manifest": str(manifest),
            "agent_config_digest": "c" * 64,
            "images": {"runtime": "runtime@sha256:aaa", "adot": "adot@sha256:bbb"},
        },
    )
    return dict(state.value["task_input_sha256"])


def test_probe_before_ensure_tasks_survive_a_release_identity_change(
    tmp_path: Path,
) -> None:
    first = _bind_release_inputs(tmp_path, release_id="0100-a")
    second = _bind_release_inputs(tmp_path, release_id="0100-b")

    for name in ("monitoring_install", "aurora_refresh", "node_keys:gpu-a"):
        assert first[name] == second[name], (
            f"{name} re-runs on a release identity change alone"
        )
    assert first["release"] != second["release"], (
        "the release task must still notice a new release identity"
    )


# The tasks `bootstrap_tasks.py` re-proves with a read-only probe on every
# deploy, plus `aurora`, whose reconcilable inputs are the admin config digest and
# the CPU subnet ids and whose ensure path writes to RDS unconditionally.
PROBE_COVERED_TASKS = (
    "pod_identity_agent",
    "monitoring_install",
    "aurora_ready",
    "aurora_refresh",
    "load_balancer_controller",
    "control_plane_role",
    "email_notifications",
    "monitoring_resources",
    "aurora",
)
# The tasks with no probe: the digest is their only re-convergence trigger, so it
# still binds the bytes of `bootstrap.py`.
UNPROBED_TASKS = ("nlb_network", "pki")


def _refactored_repository_root(tmp_path: Path, name: str) -> Path:
    root = _mirror_repository_root(tmp_path, name)
    for relative in admin_bootstrap_checkpoint.BOOTSTRAP_RECONCILE_SOURCES:
        source = root / relative
        source.write_text(
            source.read_text(encoding="utf-8") + "\n# a refactor\n", encoding="utf-8"
        )
    return root


def test_task_digests_ignore_orchestration_source_edits(tmp_path: Path) -> None:
    """Editing the bootstrap code must not re-run what a probe already re-proves.

    Every task digest embedded the bytes of `bootstrap.py` and its siblings, so
    any refactor of the orchestration re-ran the heavy ensure paths -- an
    unconditional `rds modify-db-subnet-group` and its `rds wait`, a PKI
    regeneration, an NLB reconcile -- on a deploy where no desired resource
    changed. For a task that `bootstrap_tasks.py` revalidates, the read-only probe
    is the re-convergence trigger and the module bytes are noise.
    """

    baseline = _bind_release_inputs(tmp_path, release_id="0100-a")
    edited_root = _refactored_repository_root(tmp_path, "repository-edited-source")

    refactored = _bind_release_inputs(
        tmp_path, release_id="0100-a", repository_root=edited_root
    )

    unchanged = {name: refactored[name] for name in PROBE_COVERED_TASKS}
    assert unchanged == {name: baseline[name] for name in PROBE_COVERED_TASKS}, (
        "a code edit alone re-runs probe-covered bootstrap tasks: "
        + repr(
            sorted(
                name
                for name in PROBE_COVERED_TASKS
                if baseline[name] != refactored[name]
            )
        )
    )
    for name in baseline:
        if name.startswith(("node_keys:", "executor_role:")):
            assert baseline[name] == refactored[name], (
                f"a code edit alone re-runs {name}, which a probe re-proves"
            )


def test_unprobed_task_digests_still_track_the_code_that_converges_them(
    tmp_path: Path,
) -> None:
    """`nlb_network` and `pki` have no probe, so the digest is the only trigger.

    `bootstrap_tasks.py` revalidates the platform, notification and role tasks
    with a read-only probe, but never these two. Dropping `bootstrap.py` from
    their digests would leave a change to how the NLB or the PKI is converged with
    nothing at all to re-run it on the next deploy.
    """

    baseline = _bind_release_inputs(tmp_path, release_id="0100-a")
    edited_root = _refactored_repository_root(tmp_path, "repository-edited-unprobed")

    refactored = _bind_release_inputs(
        tmp_path, release_id="0100-a", repository_root=edited_root
    )

    for name in UNPROBED_TASKS:
        assert baseline[name] != refactored[name], (
            f"{name} has no probe and no source binding, so nothing re-runs it"
        )


def test_each_task_has_its_own_digest(tmp_path: Path) -> None:
    """Two tasks with the same inputs must still be checkpointed separately.

    Two notification tasks are built from the same routing identity, and a shared
    digest would let one task's checkpoint answer for the other the next time
    only one of them changes.
    """

    digests = _bind_release_inputs(tmp_path, release_id="0100-a")

    assert len(set(digests.values())) == len(digests), "two tasks share one digest"


def test_monitoring_install_digest_tracks_its_amp_assets(tmp_path: Path) -> None:
    baseline = _bind_release_inputs(tmp_path, release_id="0100-a")
    edited_root = _mirror_repository_root(tmp_path, "repository-edited-rules")
    rules = edited_root / "deploy/observability/amp-rules.yaml"
    rules.write_text(
        rules.read_text(encoding="utf-8") + "\n# an edited capacity rule\n",
        encoding="utf-8",
    )
    shifted_digests = _bind_release_inputs(
        tmp_path, release_id="0100-a", repository_root=edited_root
    )

    assert baseline["monitoring_install"] != shifted_digests["monitoring_install"], (
        "monitoring_install ignores a change to the AMP rules it publishes"
    )


def test_aurora_refresh_digest_tracks_the_wheel_bytes(tmp_path: Path) -> None:
    baseline = _bind_release_inputs(tmp_path, release_id="0100-a")
    rebuilt = _bind_release_inputs(
        tmp_path, release_id="0100-a", wheel_bytes=b"rebuilt-wheel-bytes"
    )

    assert baseline["aurora_refresh"] != rebuilt["aurora_refresh"], (
        "aurora_refresh ignores a rebuilt control-plane wheel"
    )


def test_healthy_monitoring_install_probe_runs_no_installer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cpu = _cluster()
    endpoint = (
        "https://aps-workspaces.us-east-1.amazonaws.com/"
        "workspaces/ws-a/api/v1/remote_write"
    )
    calls: list[list[str]] = []

    class Runner:
        dry_run = False

        def run(self, arguments, **kwargs):
            calls.append(list(arguments))
            if kwargs.get("mutate"):
                raise AssertionError("probe mutated live state")
            if "jsonpath={.spec.template.spec.containers[0].image}" in arguments:
                return "adot@sha256:bbb"
            if "jsonpath={.status.availableReplicas}" in arguments:
                return "1"
            if "jsonpath={.data}" in arguments:
                return json.dumps({"collector.yaml": f"endpoint: {endpoint}"})
            if "list-rule-groups-namespaces" in arguments:
                return json.dumps({"ruleGroupsNamespaces": [{"name": "capacity"}]})
            if "describe-alert-manager-definition" in arguments:
                return json.dumps({"alertManagerDefinition": {"status": {}}})
            if "list-pod-identity-associations" in arguments:
                return json.dumps({"associations": [{"associationId": "assoc-a"}]})
            if "describe-pod-identity-association" in arguments:
                return json.dumps(
                    {
                        "association": {
                            "associationId": "assoc-a",
                            "roleArn": (
                                "arn:aws:iam::123456789012:role/"
                                "gpu-fault-site-a-amp-writer"
                            ),
                        }
                    }
                )
            if "get-role-policy" in arguments:
                return json.dumps(
                    {
                        "PolicyDocument": {
                            "Version": "2012-10-17",
                            "Statement": [
                                {
                                    "Effect": "Allow",
                                    "Action": ["aps:RemoteWrite"],
                                    "Resource": (
                                        "arn:aws:aps:us-east-1:123456789012:"
                                        "workspace/ws-a"
                                    ),
                                }
                            ],
                        }
                    }
                )
            if "get-role" in arguments:
                return json.dumps(
                    {
                        "Role": {
                            "AssumeRolePolicyDocument": (
                                admin_bootstrap_services.pod_identity_trust(_cluster())
                            ),
                            "Tags": [
                                {
                                    "Key": admin_bootstrap_services.SITE_TAG_KEY,
                                    "Value": "site-a",
                                }
                            ],
                        }
                    }
                )
            raise AssertionError(arguments)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stderr=""),
    )

    admin_bootstrap_services.install_monitoring(
        ReadOnlyProbeRunner(Runner()),
        repository_root=tmp_path,
        cpu=cpu,
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        namespace="gpu-fault-system",
        site_id="site-a",
        monitoring={
            "workspace_id": "ws-a",
            "sns_topic_arn": "arn:aws:sns:us-east-1:123456789012:gpu-fault-site-a",
        },
        adot_image="adot@sha256:bbb",
        alert_email="ops@example.com",
        probe_only=True,
    )

    assert not any("install-amp-monitoring.sh" in " ".join(call) for call in calls), (
        "healthy monitoring probe ran the mutating installer"
    )


@pytest.mark.parametrize(
    ("image", "available", "detail"),
    [
        ("adot@sha256:old", "1", "image drift"),
        ("adot@sha256:bbb", "0", "scaled-down collector"),
    ],
)
def test_monitoring_install_probe_requires_ensure_on_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image: str,
    available: str,
    detail: str,
) -> None:
    class Runner:
        dry_run = False

        def run(self, arguments, **kwargs):
            if "jsonpath={.spec.template.spec.containers[0].image}" in arguments:
                return image
            if "jsonpath={.status.availableReplicas}" in arguments:
                return available
            raise AssertionError(arguments)

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stderr=""),
    )

    with pytest.raises(BootstrapMutationRequired):
        admin_bootstrap_platform_probes.assert_monitoring_install_current(
            ReadOnlyProbeRunner(Runner()),
            cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
            namespace="gpu-fault-system",
            region="us-east-1",
            adot_image="adot@sha256:bbb",
            workspace_id="ws-a",
        )


AURORA_IMAGE = "runtime@sha256:aaa"
AURORA_WHEEL = "gpu-fault-control-plane-wheel-0100-abcdef123456"
AURORA_SECRET_ARN = "arn:aws:secretsmanager:us-east-1:123456789012:secret:m"


def _aurora_cronjob_runner(
    *,
    image: str = AURORA_IMAGE,
    wheel_configmap: str = AURORA_WHEEL,
    master_secret_arn: str = AURORA_SECRET_ARN,
):
    """A kubectl that answers a jsonpath only when it names a single field.

    kubectl evaluates every ``{...}`` group in ``-o jsonpath=`` against the root
    document and concatenates the results, so a probe that glues a pod-spec
    prefix and a field expression together reads the whole pod spec instead of
    the field. This fake reproduces exactly that: it answers the three
    single-expression projections the probe is allowed to use, and returns the
    serialized pod spec for anything else -- which is what the live cluster
    returns for the concatenated form.
    """

    spec = {
        "containers": [
            {
                "image": image,
                "env": [
                    {
                        "name": "GPU_FAULT_AURORA_MASTER_SECRET_ARN",
                        "value": master_secret_arn,
                    }
                ],
            }
        ],
        "volumes": [{"name": "artifact", "configMap": {"name": wheel_configmap}}],
    }
    root = "{.spec.jobTemplate.spec.template.spec"
    answers = {
        f"jsonpath={root}.containers[0].image}}": image,
        f'jsonpath={root}.volumes[?(@.name=="artifact")].configMap.name}}': (
            wheel_configmap
        ),
        f"jsonpath={root}.containers[0]"
        '.env[?(@.name=="GPU_FAULT_AURORA_MASTER_SECRET_ARN")].value}': (
            master_secret_arn
        ),
    }

    class Runner:
        dry_run = False

        def run(self, arguments, **kwargs):
            requested = [item for item in arguments if item.startswith("jsonpath=")]
            assert len(requested) == 1, arguments
            return answers.get(requested[0], json.dumps(spec))

    return Runner()


def _assert_aurora_refresh(runner, tmp_path: Path) -> None:
    admin_bootstrap_platform_probes.assert_aurora_refresh_current(
        ReadOnlyProbeRunner(runner),
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        namespace="gpu-fault-system",
        wheel_configmap=AURORA_WHEEL,
        runtime_image=AURORA_IMAGE,
        master_secret_arn=AURORA_SECRET_ARN,
    )


def test_aurora_refresh_probe_passes_when_live_matches(tmp_path: Path) -> None:
    """A CronJob that already matches must not send the release into ensure.

    Every deploy paid for an apply, a verify Job and a 420s ``kubectl wait``
    because the probe's jsonpath read the pod spec instead of the image, so the
    comparison could never match. Reading a field back is the whole point of the
    probe: if this passes only by accident the ensure path runs forever.
    """

    _assert_aurora_refresh(_aurora_cronjob_runner(), tmp_path)


@pytest.mark.parametrize(
    ("drift", "detail"),
    [
        ({"image": "runtime@sha256:old"}, "a rolled-back runtime image"),
        (
            {"wheel_configmap": "gpu-fault-control-plane-wheel-0100-stale0000000"},
            "a stale wheel ConfigMap",
        ),
        (
            {"master_secret_arn": "arn:aws:secretsmanager:us-east-1:1:secret:other"},
            "a different master Secret",
        ),
    ],
)
def test_aurora_refresh_probe_requires_ensure_on_drift(
    tmp_path: Path, drift: dict[str, str], detail: str
) -> None:
    """Each of the three projections must be read, and each must fail closed."""

    with pytest.raises(BootstrapMutationRequired):
        _assert_aurora_refresh(_aurora_cronjob_runner(**drift), tmp_path)


def _node_key_probe_runner(
    nodes: tuple[str, ...], gpu_keys: tuple[str, ...], cpu_keys: tuple[str, ...]
):
    calls: list[list[str]] = []

    class Runner:
        dry_run = False

        def run(self, arguments, **kwargs):
            calls.append(list(arguments))
            if "nodes" in arguments:
                return "\n".join(nodes)
            if "--kubeconfig" in arguments:
                index = arguments.index("--kubeconfig") + 1
                keys = gpu_keys if "gpu" in arguments[index] else cpu_keys
                return "\n".join(keys)
            raise AssertionError(arguments)

    return Runner(), calls


def test_healthy_node_action_key_probe_reads_only_key_names(tmp_path: Path) -> None:
    nodes = ("hyperpod-node-a", "hyperpod-node-b")
    runner, calls = _node_key_probe_runner(nodes, nodes, nodes)

    admin_bootstrap_services.provision_node_action_keys(
        ReadOnlyProbeRunner(runner),
        repository_root=tmp_path,
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
        namespace="gpu-fault-system",
        cluster=_gpu_cluster(),
        cluster_id="gpu-a",
        fleet_master_file=tmp_path / "fleet-master",
        probe_only=True,
    )

    assert not any(
        "provision-node-action-keys.sh" in " ".join(call) for call in calls
    ), "healthy node key probe ran the mutating provisioning script"
    secret_reads = [call for call in calls if "secret" in call]
    assert len(secret_reads) == 2, (
        "node key probe did not compare both the GPU Secret and the CPU mirror"
    )
    assert all(
        call[call.index("-o") + 1].startswith("go-template=")
        and "$_" in call[call.index("-o") + 1]
        for call in secret_reads
    ), "node key probe read Secret values instead of key names"


def test_node_action_key_probe_requires_ensure_for_a_new_node(tmp_path: Path) -> None:
    runner, _calls = _node_key_probe_runner(
        ("hyperpod-node-a", "hyperpod-node-b"),
        ("hyperpod-node-a",),
        ("hyperpod-node-a",),
    )

    with pytest.raises(BootstrapMutationRequired):
        admin_bootstrap_services.provision_node_action_keys(
            ReadOnlyProbeRunner(runner),
            repository_root=tmp_path,
            cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
            gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
            namespace="gpu-fault-system",
            cluster=_gpu_cluster(),
            cluster_id="gpu-a",
            fleet_master_file=tmp_path / "fleet-master",
            probe_only=True,
        )


def test_node_action_key_probe_requires_ensure_for_a_stale_mirror(
    tmp_path: Path,
) -> None:
    nodes = ("hyperpod-node-a", "hyperpod-node-b")
    runner, _calls = _node_key_probe_runner(nodes, nodes, ("hyperpod-node-a",))

    with pytest.raises(BootstrapMutationRequired):
        admin_bootstrap_services.provision_node_action_keys(
            ReadOnlyProbeRunner(runner),
            repository_root=tmp_path,
            cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
            gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
            namespace="gpu-fault-system",
            cluster=_gpu_cluster(),
            cluster_id="gpu-a",
            fleet_master_file=tmp_path / "fleet-master",
            probe_only=True,
        )


def test_node_action_key_probe_requires_ensure_when_rotation_is_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nodes = ("hyperpod-node-a",)
    runner, _calls = _node_key_probe_runner(nodes, nodes, nodes)
    monkeypatch.setenv("GPU_FAULT_ROTATE_NODE_ACTION_KEY", "hyperpod-node-a")

    with pytest.raises(BootstrapMutationRequired):
        admin_bootstrap_services.provision_node_action_keys(
            ReadOnlyProbeRunner(runner),
            repository_root=tmp_path,
            cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
            gpu_kubeconfig=tmp_path / "gpu.kubeconfig",
            namespace="gpu-fault-system",
            cluster=_gpu_cluster(),
            cluster_id="gpu-a",
            fleet_master_file=tmp_path / "fleet-master",
            probe_only=True,
        )


def test_bootstrap_input_digest_invalidates_only_stale_task_checkpoints(
    tmp_path: Path,
) -> None:
    state = BootstrapState(tmp_path / "state.json", site_id="test")
    state.record("pki", {"certificate_arn": "arn:certificate"})
    state.complete("pki")

    task_digests = {"pki": "1" * 64, "aurora": "2" * 64}
    state.bind_inputs("a" * 64, task_digests)
    state.complete("pki")
    state.complete("aurora")
    state.bind_inputs("a" * 64, task_digests)
    assert state.value["completed_tasks"] == ["aurora", "pki"]

    state.bind_inputs("b" * 64, {"pki": "1" * 64, "aurora": "3" * 64})
    assert state.value["completed_tasks"] == ["pki"]
    assert state.value["resources"]["pki"] == {"certificate_arn": "arn:certificate"}


def _control_plane_identity_runner(calls: list[list[str]]):
    """A control plane whose identity already matches, recording every read.

    The interesting number in the tests below is how many times a converged
    deploy asks AWS the same question, so this fake answers every read of the
    healthy path and refuses any mutation.
    """

    trust = admin_bootstrap_services.pod_identity_trust(_cluster())
    policy = admin_bootstrap_services.control_plane_policy_document(
        region="us-east-1", account_id="123456789012"
    )
    role_arn = "arn:aws:iam::123456789012:role/gpu-fault-site-a-control"

    class Runner:
        dry_run = False

        def run(self, arguments, **kwargs):
            argv = list(arguments)
            calls.append(argv)
            if kwargs.get("mutate"):
                raise AssertionError(f"a converged identity was rewritten: {argv}")
            if argv[0] == "kubectl":
                return "serviceaccount/gpu-fault-control-plane"
            if "describe-addon" in argv:
                return json.dumps(
                    {
                        "addon": {
                            "addonArn": (
                                "arn:aws:eks:us-east-1:123456789012:addon/control/a"
                            )
                        }
                    }
                )
            if "list-tags-for-resource" in argv:
                return json.dumps(
                    {"tags": {admin_bootstrap_services.SITE_TAG_KEY: "site-a"}}
                )
            if "wait" in argv:
                return ""
            if "get-role-policy" in argv:
                return json.dumps({"PolicyDocument": policy})
            if "get-role" in argv:
                return json.dumps(
                    {
                        "Role": {
                            "AssumeRolePolicyDocument": trust,
                            "Tags": [
                                {
                                    "Key": admin_bootstrap_services.SITE_TAG_KEY,
                                    "Value": "site-a",
                                }
                            ],
                        }
                    }
                )
            if "list-pod-identity-associations" in argv:
                return json.dumps({"associations": [{"associationId": "assoc-a"}]})
            if "describe-pod-identity-association" in argv:
                return json.dumps(
                    {"association": {"associationId": "assoc-a", "roleArn": role_arn}}
                )
            raise AssertionError(argv)

        def aws_json(self, region, *arguments, **kwargs):
            return json.loads(
                self.run(
                    ["aws", *arguments, "--region", region, "--output", "json"],
                    **kwargs,
                )
            )

    return Runner()


def _forbidden_subprocess(calls: list[list[str]]):
    def run(arguments, **_kwargs):
        calls.append(list(arguments))
        raise AssertionError(
            f"a bootstrap read bypassed the command runner: {list(arguments)}"
        )

    return run


def test_pod_identity_agent_is_probed_once_per_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The add-on is described and waited for once, not once per caller.

    `revalidate_pod_identity_agent`, the control-plane role and the load balancer
    controller each ensured the add-on, and each one paid for a describe plus an
    `aws eks wait addon-active`. The add-on cannot change between two calls in
    one deploy, so the first successful read is the evidence for the rest of the
    run -- including for the ensure that follows a healthy read-only probe.
    """

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    calls: list[list[str]] = []
    runner = _control_plane_identity_runner(calls)
    monkeypatch.setattr(subprocess, "run", _forbidden_subprocess(calls))
    arguments: dict[str, Any] = {
        "cpu": _cluster(),
        "cpu_kubeconfig": tmp_path / "cpu.kubeconfig",
        "namespace": "gpu-fault-system",
        "site_id": "site-a",
    }

    probed = admin_bootstrap_services.ensure_control_plane_role(
        ReadOnlyProbeRunner(runner), **arguments
    )
    ensured = admin_bootstrap_services.ensure_control_plane_role(runner, **arguments)

    assert probed["role_arn"] == ensured["role_arn"]
    assert len([argv for argv in calls if "describe-addon" in argv]) == 1, (
        "the Pod Identity add-on was described more than once in one run"
    )
    assert len([argv for argv in calls if "addon-active" in argv]) == 1, (
        "the run waited for the same add-on to become active more than once"
    )


def test_ensure_role_reads_each_role_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """One IAM read answers both the existence question and the drift question.

    The silent `aws iam get-role` existence probe and the logged read returned
    the same document, and so did the two `get-role-policy` calls, so every role
    in the deploy cost four IAM reads to decide it was already correct.
    """

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    calls: list[list[str]] = []
    runner = _control_plane_identity_runner(calls)
    monkeypatch.setattr(subprocess, "run", _forbidden_subprocess(calls))

    admin_bootstrap_services.probe_iam_role(
        runner,
        account_id="123456789012",
        role_name="gpu-fault-site-a-control",
        trust=admin_bootstrap_services.pod_identity_trust(_cluster()),
        policy_name="GPUFaultRegionalObserve",
        policy=admin_bootstrap_services.control_plane_policy_document(
            region="us-east-1", account_id="123456789012"
        ),
        site_id="site-a",
    )

    assert len([argv for argv in calls if "get-role" in argv]) == 1, (
        "the role document was read twice to answer one question"
    )
    assert len([argv for argv in calls if "get-role-policy" in argv]) == 1, (
        "the inline policy was read twice to answer one question"
    )


def test_ensure_rds_ca_bundle_runs_the_pinned_script_against_the_cpu_cluster(
    tmp_path: Path,
) -> None:
    """The refresh CronJob mounts gpu-fault-rds-ca-bundle non-optionally; the
    bootstrap task must ship it before applying the CronJob (live 2026-09-08:
    the verify Job hung on the missing volume until the deploy timed out)."""
    from gpu_fault.admin.rds_ca_bundle import ensure_rds_ca_bundle

    calls: list[tuple[list[str], dict]] = []

    class Runner:
        dry_run = False

        def run(self, arguments, **kwargs):
            calls.append((list(arguments), kwargs))
            return ""

    ensure_rds_ca_bundle(
        Runner(),
        repository_root=tmp_path,
        cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
        namespace="gpu-fault-system",
    )

    assert len(calls) == 1, calls
    arguments, kwargs = calls[0]
    assert arguments == [
        "bash",
        str(tmp_path / "deploy/control-plane/tools/apply-rds-ca-bundle.sh"),
    ], arguments
    assert kwargs["mutate"] is True, kwargs
    assert kwargs["env"]["KUBECONFIG"] == str(tmp_path / "cpu.kubeconfig"), kwargs
    assert kwargs["env"]["GPU_FAULT_NAMESPACE"] == "gpu-fault-system", kwargs
