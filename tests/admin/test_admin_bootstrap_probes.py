from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
    trust = admin_bootstrap_services.pod_identity_trust()
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
        admin_bootstrap_services.subprocess,
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
            if "list-policy-tags" in arguments:
                return json.dumps(
                    {
                        "Tags": [
                            {
                                "Key": admin_bootstrap_services.SITE_TAG_KEY,
                                "Value": "site-a",
                            }
                        ]
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


def test_healthy_sqs_probe_skips_policy_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu = _cluster()
    topic_arn = "arn:aws:sns:us-east-1:123456789012:topic-a"
    queue_url = "https://sqs.us-east-1.amazonaws.com/123456789012/queue-a"
    queue_arn = "arn:aws:sqs:us-east-1:123456789012:queue-a"
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sns.amazonaws.com"},
                "Action": "sqs:SendMessage",
                "Resource": queue_arn,
                "Condition": {"ArnEquals": {"aws:SourceArn": topic_arn}},
            }
        ],
    }
    calls = []

    class Runner:
        dry_run = False

        def run(self, arguments, **kwargs):
            calls.append((list(arguments), kwargs))
            if "list-queue-tags" in arguments:
                return json.dumps(
                    {"Tags": {admin_bootstrap_services.SITE_TAG_KEY: "site-a"}}
                )
            if "get-queue-attributes" in arguments:
                return json.dumps(
                    {
                        "Attributes": {
                            "QueueArn": queue_arn,
                            "Policy": json.dumps(policy),
                        }
                    }
                )
            raise AssertionError(arguments)

    monkeypatch.setattr(
        admin_bootstrap_services.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=queue_url, stderr=""
        ),
    )

    result = admin_bootstrap_services.probe_sqs_queue(
        Runner(), cpu=cpu, site_id="site-a", topic_arn=topic_arn
    )

    assert result[:2] == (queue_url, queue_arn)
    assert all(not options.get("mutate") for _command, options in calls), (
        "healthy SQS probe rewrote queue state"
    )


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
                                admin_bootstrap_services.pod_identity_trust()
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
        admin_bootstrap_services.subprocess,
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
        admin_bootstrap_services.subprocess,
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


def test_aurora_refresh_probe_requires_ensure_on_wheel_drift(tmp_path: Path) -> None:
    class Runner:
        dry_run = False

        def run(self, arguments, **kwargs):
            if "jsonpath" in " ".join(arguments) and "image}" in " ".join(arguments):
                return "runtime@sha256:aaa"
            return "gpu-fault-control-plane-wheel-0100-stale0000000"

    with pytest.raises(BootstrapMutationRequired):
        admin_bootstrap_platform_probes.assert_aurora_refresh_current(
            ReadOnlyProbeRunner(Runner()),
            cpu_kubeconfig=tmp_path / "cpu.kubeconfig",
            namespace="gpu-fault-system",
            wheel_configmap="gpu-fault-control-plane-wheel-0100-abcdef123456",
            runtime_image="runtime@sha256:aaa",
            master_secret_arn="arn:aws:secretsmanager:us-east-1:123456789012:secret:m",
        )


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
