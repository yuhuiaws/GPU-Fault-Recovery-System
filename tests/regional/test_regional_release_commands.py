"""Regional release command tests: preflight, join, rollback and contracts.

Split from ``test_regional_release_orchestrator.py``, which keeps the rollout
engine itself (ordering, parallelism, gates, failure handling). What is left here
is the surface an administrator drives one command at a time, plus the artifact
contracts those commands verify: preflight binding, cluster join and removal,
rollback, deploy-path selection, live checkpoints, the rendered manifest and the
executor IAM boundary.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gpu_fault.admin.command_log import report_failure
from gpu_fault_release import regional_admin_commands as ADMIN_COMMANDS_MODULE
from gpu_fault_release import regional_release_iam as IAM_MODULE
from gpu_fault_release import regional_release_orchestration as ORCHESTRATION_MODULE
from gpu_fault_release import (
    regional_release_rollback_context as ROLLBACK_CONTEXT_MODULE,
)
from gpu_fault_release import regional_release_state as STATE_MODULE
from tests.regional._release_orchestrator_support import (
    CPU_EKS_ARN,
    GPU_EKS_ARN,
    REGION,
    ROOT,
    SNS_TOPIC_ARN,
    config_file,
    manifest_config_file,
)
from tests.regional._release_orchestrator_support import RELEASE_MODULE as MODULE

# `build_rollback_environment` is re-exported by the orchestration module but
# defined here, and a function resolves its globals in the module that defines
# it, so the digest stubs below have to be installed on this one.


class PreflightRunner:
    dry_run = False

    def __init__(self, *, gpu_eks_arn=GPU_EKS_ARN, gpu_node_recovery="None") -> None:
        self.gpu_eks_arn = gpu_eks_arn
        self.gpu_node_recovery = gpu_node_recovery

    def run(self, args, **kwargs):
        del kwargs
        if "config" in args and "view" in args:
            return CPU_EKS_ARN if "--kubeconfig" in args else self.gpu_eks_arn
        if args[:3] == ["aws", "sagemaker", "describe-cluster"]:
            cluster_name = args[args.index("--cluster-name") + 1]
            return json.dumps(
                {
                    "EksClusterArn": (
                        GPU_EKS_ARN if cluster_name == "hp-gpu-a" else CPU_EKS_ARN
                    ),
                    "NodeRecovery": (
                        self.gpu_node_recovery
                        if cluster_name == "hp-gpu-a"
                        else "Automatic"
                    ),
                }
            )
        if "--raw=/readyz" in args:
            return "ok"
        if "get" in args and "nodes" in args:
            return (
                '{"items":[{"metadata":{"name":"gpu-node-a"},"status":'
                '{"addresses":[{"type":"InternalIP","address":"10.0.1.10"}]}}]}'
            )
        raise AssertionError(f"unexpected preflight command: {args}")


def test_preflight_binds_contexts_and_hyperpod_to_config(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, PreflightRunner())
    monkeypatch.setattr(release, "_validate_executor_iam_role", lambda _target: None)

    MODULE.ensure_region_contexts(release)


def test_preflight_rejects_wrong_context_and_managed_gpu_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    wrong_context = MODULE.RegionalRelease(
        config,
        PreflightRunner(
            gpu_eks_arn=("arn:aws:eks:us-east-1:123456789012:cluster/unexpected-gpu")
        ),
    )
    monkeypatch.setattr(
        wrong_context, "_validate_executor_iam_role", lambda _target: None
    )

    with pytest.raises(MODULE.ReleaseError, match="does not match"):
        MODULE.ensure_region_contexts(wrong_context)

    managed_recovery = MODULE.RegionalRelease(
        config, PreflightRunner(gpu_node_recovery="Automatic")
    )
    monkeypatch.setattr(
        managed_recovery, "_validate_executor_iam_role", lambda _target: None
    )
    with pytest.raises(MODULE.ReleaseError, match="NodeRecovery=None"):
        MODULE.ensure_region_contexts(managed_recovery)


def test_join_cluster_requires_current_release_artifact(tmp_path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))

    assert release.wheel_cm.startswith("gpu-fault-control-plane-wheel-0100-"), (
        f"a joining cluster reads the wheel by name: {release.wheel_cm}"
    )
    assert release.bundle_cm.startswith("gpu-fault-node-installer-0100-"), (
        f"a joining cluster reads the installer bundle by name: {release.bundle_cm}"
    )
    assert MODULE.STATE_CONFIG_MAP == ("gpu-fault-regional-release-state")


def test_sync_release_state_records_a_noop_topology() -> None:
    calls = []
    release = SimpleNamespace(
        state={
            "rollback_completed_phases": ["rollback-verified"],
            "rollback_result": {"status": "PASSED"},
        },
        _capture_previous=lambda: {"live_runtime_image": "legacy-runtime:stable"},
        _save_state=lambda phase, **updates: calls.append((phase, updates)),
    )

    MODULE.sync_release_state(release)

    assert release.state == {}
    assert calls == [
        (
            "complete",
            {
                "previous": None,
                "release_diff": {"kind": "NOOP", "changed": []},
                "adopted_live_runtime_image": "legacy-runtime:stable",
                "transaction_committed": True,
                "release_lifecycle": "COMMITTED",
                "completed_phases": ["complete"],
                "completed_cluster_ids": [],
            },
        )
    ]


def test_join_cluster_requires_an_idle_remote_command_queue() -> None:
    """A join is refused while remote commands are still in flight.

    The cluster id is resolved first, so an unknown cluster is reported as
    unknown instead of being masked by whatever the queue happens to be doing.
    """

    unknown = SimpleNamespace(
        _target=lambda _cluster_id: (_ for _ in ()).throw(
            MODULE.ReleaseError("unknown cluster")
        ),
        _remote_commands_are_idle=lambda: pytest.fail(
            "the queue was probed for a cluster that is not in the config"
        ),
    )
    with pytest.raises(MODULE.ReleaseError, match="unknown cluster"):
        MODULE.join_target(unknown, "gpu-z")

    busy = SimpleNamespace(
        _target=lambda cluster_id: SimpleNamespace(cluster_id=cluster_id),
        _remote_commands_are_idle=lambda: False,
    )
    with pytest.raises(MODULE.ReleaseError, match="PENDING/LEASED/WAITING"):
        MODULE.join_target(busy, "gpu-a")

    idle = SimpleNamespace(
        _target=lambda cluster_id: SimpleNamespace(cluster_id=cluster_id),
        _remote_commands_are_idle=lambda: True,
    )

    assert MODULE.join_target(idle, "gpu-a").cluster_id == "gpu-a"


def rollback_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    metadata: dict[str, str],
    allow_email: bool,
    ses_configuration_set: str | None = None,
) -> dict[str, str]:
    monkeypatch.setattr(
        ROLLBACK_CONTEXT_MODULE,
        "admin_config_renderer_environment",
        lambda _admin_config: {},
    )
    monkeypatch.setattr(
        ROLLBACK_CONTEXT_MODULE, "notification_digest", lambda _notifications: "f" * 64
    )
    admin_config = SimpleNamespace(
        sha256=lambda: "a" * 64,
        role_sha256=lambda: {
            "ingress": "i" * 64,
            "worker": "w" * 64,
            "spool": "s" * 64,
        },
    )
    return ORCHESTRATION_MODULE.build_rollback_environment(
        rollback_config=SimpleNamespace(
            admin_config=admin_config,
            cpu_kubeconfig="/nonexistent/cpu.kubeconfig",
            aws_region=REGION,
            namespace="gpu-fault-system",
            notifications=SimpleNamespace(
                allow_email=allow_email,
                acknowledge_external_alert_channel=not allow_email,
                ses_configuration_set=ses_configuration_set,
            ),
            notification_environment=lambda: {
                "GPU_FAULT_NOTIFICATION_CHANNEL": "sns",
                "GPU_FAULT_SNS_TOPIC_ARN": SNS_TOPIC_ARN,
            },
        ),
        metadata=metadata,
        cpu_wheel="previous-cpu-wheel",
        cpu_sha="c" * 64,
        artifact="artifact",
        config_digest="config",
        runtime_profile_version="profile-v1",
        runtime_image="previous-runtime",
    )


def test_rollback_detects_legacy_component_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A previous release without component digests is pinned the legacy way.

    Rolling back to a release that predates the per-component digests must not
    make the control plane demand digests the old Agents cannot present, and a
    release that does record them must not be treated as legacy.
    """

    legacy = rollback_environment(monkeypatch, metadata={}, allow_email=True)
    modern = rollback_environment(
        monkeypatch,
        metadata={"required-regional-executor-artifact-sha256": "e" * 64},
        allow_email=True,
    )

    assert legacy["GPU_FAULT_LEGACY_COMPONENT_PINS"] == "true"
    assert legacy["GPU_FAULT_REQUIRED_AGENT_COMPATIBILITY_DIGEST"] == "artifact"
    assert modern["GPU_FAULT_LEGACY_COMPONENT_PINS"] == "false"
    assert (
        modern["GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST"] == "e" * 64
    )


def test_rollback_preserves_the_alerting_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rollback carries the notification settings, so alerting stays as declared.

    Dropping these would silently roll the region back into a state where nobody
    is told about the next fault.
    """

    enabled = rollback_environment(monkeypatch, metadata={}, allow_email=True)
    acknowledged = rollback_environment(monkeypatch, metadata={}, allow_email=False)

    assert enabled["GPU_FAULT_ALLOW_EMAIL"] == "true"
    assert enabled["GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL"] == "false"
    assert acknowledged["GPU_FAULT_ALLOW_EMAIL"] == "false"
    assert acknowledged["GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL"] == "true"
    assert enabled["GPU_FAULT_NOTIFICATION_CONFIG_SHA256"] == "f" * 64
    assert enabled["GPU_FAULT_NOTIFICATION_CHANNEL"] == "sns", (
        "a rolled-back role runs the alert-channel guard at startup and must "
        "be told which channel the snapshot shipped with"
    )
    assert enabled["GPU_FAULT_SNS_TOPIC_ARN"] == SNS_TOPIC_ARN
    assert enabled["GPU_FAULT_SES_CONFIGURATION_SET"] == "", (
        "an unset site re-renders the empty ConfigMap value, never a value "
        "inherited from the operator's shell"
    )

    declared = rollback_environment(
        monkeypatch, metadata={}, allow_email=True, ses_configuration_set="alerts-set"
    )

    assert declared["GPU_FAULT_SES_CONFIGURATION_SET"] == "alerts-set"


def test_last_cluster_can_be_removed_from_the_cpu_registry(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))

    class RegistryRunner:
        dry_run = False

        def __init__(self) -> None:
            self.registrations = None
            self.secret = {
                "metadata": {"resourceVersion": "1"},
                "data": {
                    "clusters.json": base64.b64encode(
                        json.dumps(
                            [{"cluster_id": config.clusters[0].cluster_id}]
                        ).encode()
                    ).decode()
                },
            }

        def run(self, arguments, **kwargs):
            if "patch" in arguments and "secret" in arguments:
                patch = json.loads(arguments[arguments.index("-p") + 1])
                encoded = patch[-1]["value"]
                self.secret["data"]["clusters.json"] = encoded
                self.registrations = json.loads(base64.b64decode(encoded))
            return ""

    runner = RegistryRunner()
    release = MODULE.RegionalRelease(config, runner)
    monkeypatch.setattr(
        release, "_get_json", lambda _arguments: json.loads(json.dumps(runner.secret))
    )

    release._update_registry(config.clusters[0], remove=True)

    assert runner.registrations == []


@pytest.mark.parametrize(
    ("state_exists", "phase", "expected"),
    [
        (False, None, "bootstrap"),
        (True, "bootstrap-cleaned", "bootstrap"),
        (True, "bootstrap-cleanup-started", "bootstrap"),
        (True, "bootstrap-cleanup-progress", "bootstrap"),
        (True, "bootstrap-cleanup-failed", "bootstrap"),
        (True, "bootstrap-cpu-ready", "bootstrap"),
        (True, "bootstrap-endpoint-ready", "bootstrap"),
        (True, "bootstrap-data-plane-progress", "bootstrap"),
        (True, "failed", "upgrade"),
        (True, "rolled-back", "upgrade"),
        (True, "complete", "upgrade"),
    ],
)
def test_deploy_selects_initial_or_upgrade_path(
    tmp_path: Path, monkeypatch, state_exists: bool, phase: str | None, expected: str
) -> None:
    module = MODULE
    admin = ADMIN_COMMANDS_MODULE
    release = module.RegionalRelease(
        module.ReleaseConfig.load(config_file(tmp_path)), module.Runner(dry_run=False)
    )
    calls: list[str] = []
    monkeypatch.setattr(release.runner, "probe", lambda _args, **_kwargs: state_exists)
    monkeypatch.setattr(release, "bootstrap", lambda: calls.append("bootstrap"))
    monkeypatch.setattr(release, "upgrade", lambda **_kwargs: calls.append("upgrade"))
    monkeypatch.setattr(release, "noop", lambda _diff: calls.append("noop"))
    if state_exists:
        monkeypatch.setattr(
            release,
            "_load_state",
            lambda: {
                "phase": phase,
                "release_id": release.release_id,
                "release_diff": {
                    "kind": "CONTROL_PLANE_ONLY",
                    "changed": ["control_plane_wheel"],
                },
            },
        )

    admin.run_deploy(release)

    assert calls == [expected]


# --------------------------------------------------------------------------- #
# M-23: a resume pins the approved-plan digest so the resumed apply refuses a
# working tree that drifted since the transaction was planned; a fresh
# transaction (initial bootstrap or a brand-new upgrade off a completed state)
# must NOT pin, or it would enforce the wrong (or previous) transaction's plan.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("state_exists", "phase", "should_pin"),
    [
        (False, None, False),  # fresh bootstrap: nothing planned yet
        (True, "bootstrap-cpu-ready", True),  # resuming a partial bootstrap
        (True, "bootstrap-data-plane-progress", True),
        (True, "failed", True),  # resuming an incomplete upgrade
        (True, "partial-convergence", True),
        (True, "rolled-back", False),  # re-entry renders a different plan
        (True, "complete", False),  # brand-new upgrade off a finished state
    ],
)
def test_deploy_pins_approved_plan_only_when_resuming(
    tmp_path: Path, monkeypatch, state_exists: bool, phase: str | None, should_pin: bool
) -> None:
    module = MODULE
    admin = ADMIN_COMMANDS_MODULE
    release = module.RegionalRelease(
        module.ReleaseConfig.load(config_file(tmp_path)), module.Runner(dry_run=False)
    )
    pins: list[str | None] = []
    monkeypatch.setattr(release.runner, "probe", lambda _args, **_kwargs: state_exists)
    monkeypatch.setattr(release, "bootstrap", lambda: None)
    monkeypatch.setattr(release, "upgrade", lambda **_kwargs: None)
    monkeypatch.setattr(release, "noop", lambda _diff: None)
    monkeypatch.setattr(
        release, "pin_approved_manifest_plan", lambda digest: pins.append(digest)
    )
    if state_exists:
        monkeypatch.setattr(
            release,
            "_load_state",
            lambda: {
                "phase": phase,
                # The same candidate as the recorded transaction: a plain resume,
                # not a supersede.
                "release_id": release.release_id,
                "approved_manifest_sha256": "plan-digest",
                "release_diff": {
                    "kind": "CONTROL_PLANE_ONLY",
                    "changed": ["control_plane_wheel"],
                },
            },
        )

    admin.run_deploy(release)

    assert pins == (["plan-digest"] if should_pin else [])


def test_resume_pins_the_approved_plan_digest(tmp_path: Path, monkeypatch) -> None:
    module = MODULE
    admin = ADMIN_COMMANDS_MODULE
    release = module.RegionalRelease(
        module.ReleaseConfig.load(config_file(tmp_path)), module.Runner(dry_run=False)
    )
    pins: list[str | None] = []
    resumed: list[bool] = []
    monkeypatch.setattr(
        release,
        "_load_state",
        lambda: {
            "phase": "failed",
            "release_id": release.release_id,
            "approved_manifest_sha256": "plan-digest",
            "release_diff": {
                "kind": "CONTROL_PLANE_ONLY",
                "changed": ["control_plane_wheel"],
            },
        },
    )
    monkeypatch.setattr(
        release, "pin_approved_manifest_plan", lambda digest: pins.append(digest)
    )
    monkeypatch.setattr(
        release, "upgrade", lambda **kwargs: resumed.append(kwargs.get("resume"))
    )

    admin.run_resume(release)

    # Pinned before the resumed apply runs, with the persisted plan digest.
    assert pins == ["plan-digest"]
    assert resumed == [True]


def test_saved_state_stickily_records_the_approved_manifest_digest(
    tmp_path: Path,
) -> None:
    # The persisted ``approved_manifest_sha256`` follows the pinned approved
    # digest when one is set (a resume), so a resume checkpoint cannot overwrite
    # the original plan digest with the resuming process's (drifted) tree.
    module = MODULE
    release = module.RegionalRelease(
        module.ReleaseConfig.load(config_file(tmp_path)), module.Runner(dry_run=True)
    )

    # Fresh transaction (nothing pinned): records the tree it is applying now.
    release.approved_manifest_digest = None
    module.save_state(release, "uploaded", previous=None)
    assert release.state["approved_manifest_sha256"] == release.rendered_manifest_digest

    # Resume (approved digest pinned): the pin wins even though the rendered
    # working-tree digest differs.
    release.approved_manifest_digest = "original-plan-digest"
    module.save_state(release, "cpu-staged", previous=None)
    assert release.state["approved_manifest_sha256"] == "original-plan-digest"
    assert release.rendered_manifest_digest != "original-plan-digest"


def test_bootstrap_live_checkpoint_recognizes_current_cpu_release(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))
    monkeypatch.setattr(
        release,
        "_config_map_data",
        lambda _name: {
            "required-agent-artifact-sha256": release.node_wheel_sha,
            "required-agent-compatibility-digest": (
                config.component_digests.get("node_runtime") or release.node_wheel_sha
            ),
            "required-regional-executor-artifact-sha256": (release.executor_wheel_sha),
            "required-regional-executor-compatibility-digest": (
                config.component_digests.get("executor") or release.executor_wheel_sha
            ),
            "required-agent-config-digest": config.agent_config_digest,
            "required-runtime-profile-version": config.runtime_profile_version,
        },
    )
    monkeypatch.setattr(
        release, "_deployment_wheel", lambda _kubectl, _deployment: release.wheel_cm
    )
    monkeypatch.setattr(
        release,
        "_get_json",
        lambda _arguments: {
            "metadata": {"generation": 7},
            "spec": {"replicas": 3},
            "status": {
                "observedGeneration": 7,
                "readyReplicas": 3,
                "updatedReplicas": 3,
                "availableReplicas": 3,
            },
        },
    )

    assert release._bootstrap_cpu_is_current() is True


def test_bootstrap_live_checkpoint_rejects_partial_cpu_rollout(
    tmp_path: Path, monkeypatch
) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))
    monkeypatch.setattr(
        release,
        "_config_map_data",
        lambda _name: {
            "required-agent-artifact-sha256": release.wheel_sha,
            "required-agent-config-digest": config.agent_config_digest,
            "required-runtime-profile-version": config.runtime_profile_version,
        },
    )
    monkeypatch.setattr(
        release, "_deployment_wheel", lambda _kubectl, _deployment: release.wheel_cm
    )
    monkeypatch.setattr(
        release,
        "_get_json",
        lambda _arguments: {
            "metadata": {"generation": 7},
            "spec": {"replicas": 3},
            "status": {
                "observedGeneration": 7,
                "readyReplicas": 2,
                "updatedReplicas": 3,
                "availableReplicas": 2,
            },
        },
    )

    assert release._bootstrap_cpu_is_current() is False


def test_regional_release_shell_has_valid_syntax() -> None:
    subprocess.run(
        [
            "bash",
            "-n",
            str(ROOT / "deploy/control-plane/regional/rollout-regional-release.sh"),
        ],
        check=True,
    )


def test_release_package_imports_with_only_src_on_pythonpath(tmp_path: Path) -> None:
    """The launcher puts ``src`` on PYTHONPATH and nothing else.

    Run from an unrelated directory so neither the repository root nor the
    package directory is on ``sys.path`` by accident: the orchestrator has to
    resolve every sibling through ``gpu_fault_release.*``.
    """
    completed = subprocess.run(
        [sys.executable, "-c", "import gpu_fault_release.rollout"],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT / "src"),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_regional_release_shell_launches_the_package_entry_point(
    tmp_path: Path,
) -> None:
    """The ``.sh`` path is the contract the admin CLI execs; argv passes through.

    ``--help`` is the one argument that exercises the full launcher (bash ->
    ``python3 -m gpu_fault_release.rollout`` -> argparse) without touching a
    cluster, and its usage line proves the request reached the orchestrator's
    own parser rather than a stub.
    """
    launcher = ROOT / "deploy/control-plane/regional/rollout-regional-release.sh"
    completed = subprocess.run(
        [str(launcher), "--help"],
        cwd=tmp_path,
        env={
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "PYTHONSAFEPATH"}
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
    assert "status" in completed.stdout
    assert "python3 -m gpu_fault_release.rollout" in launcher.read_text(
        encoding="utf-8"
    )


def test_gpu_deployment_manifest_is_stamped_with_release_sha(tmp_path) -> None:
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))
    release = MODULE.RegionalRelease(config, MODULE.Runner(dry_run=True))
    document = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "executor"},
        "spec": {
            "template": {
                "metadata": {"annotations": {"gpu-fault.io/artifact-sha256": "old"}}
            }
        },
    }
    rendered = release._stamp_gpu_deployments(json.dumps(document))
    stamped = next(yaml.safe_load_all(rendered))
    annotations = stamped["spec"]["template"]["metadata"]["annotations"]

    assert annotations["gpu-fault.io/artifact-sha256"] == release.executor_wheel_sha
    assert (
        annotations["gpu-fault.io/executor-wheel-sha256"] == release.executor_wheel_sha
    )
    assert annotations["gpu-fault.io/executor-compatibility-digest"] == (
        config.component_digests.get("executor") or release.executor_wheel_sha
    )
    assert "gpu-fault.io/control-plane-wheel-sha256" not in annotations
    assert annotations["gpu-fault.io/release-rollout"] == release.release_id
    assert annotations["gpu-fault.io/runtime-image"] == MODULE.DEFAULT_RUNTIME_IMAGE


def test_executor_iam_boundary_accepts_minimal_role() -> None:
    MODULE.validate_executor_iam_documents(
        "arn:aws:iam::1:role/executor",
        [
            {
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "sagemaker:DescribeCluster",
                            "sagemaker:ListClusterNodes",
                            "sagemaker:DescribeClusterNode",
                            "sagemaker:BatchRebootClusterNodes",
                            "s3:PutObject",
                        ],
                    }
                ]
            }
        ],
    )


@pytest.mark.parametrize(
    "action",
    [
        "ses:SendEmail",
        "sagemaker:BatchReplaceClusterNodes",
        "sagemaker:BatchDeleteClusterNodes",
        "sagemaker:*",
    ],
)
def test_executor_iam_boundary_rejects_excess_privilege(action: str) -> None:
    with pytest.raises(
        MODULE.ReleaseError, match="exceeds the regional data-plane boundary"
    ):
        MODULE.validate_executor_iam_documents(
            "arn:aws:iam::1:role/executor",
            [{"Statement": [{"Effect": "Allow", "Action": action}]}],
        )


def test_release_config_loads_content_addressed_manifest(tmp_path) -> None:
    config = MODULE.ReleaseConfig.load(manifest_config_file(tmp_path))

    assert config.wheel.name == "release.whl"
    assert config.bundle.name == "bundle.tar.gz"


def _iam_release(calls: list[tuple[str, ...]]):
    """A release whose IAM reads are recorded, with a shared managed policy.

    Both roles attach ``arn:aws:iam::1:policy/shared``, which is what a real
    fleet looks like: one policy authored once and attached to every cluster's
    executor role.
    """

    documents = {
        "list-role-policies": {"PolicyNames": []},
        "list-attached-role-policies": {
            "AttachedPolicies": [{"PolicyArn": "arn:aws:iam::1:policy/shared"}]
        },
        "get-policy": {"Policy": {"DefaultVersionId": "v3"}},
        "get-policy-version": {
            "PolicyVersion": {
                "Document": {
                    "Statement": [
                        {"Effect": "Allow", "Action": "sagemaker:DescribeCluster"}
                    ]
                }
            }
        },
    }

    class Runner:
        @staticmethod
        def run(arguments, **_kwargs):
            calls.append(tuple(arguments))
            return json.dumps(documents[arguments[2]])

    return SimpleNamespace(config=SimpleNamespace(aws_region=REGION), runner=Runner())


def _iam_target(cluster_id: str):
    return SimpleNamespace(
        executor_irsa_role_arn=f"arn:aws:iam::1:role/executor-{cluster_id}"
    )


def test_shared_executor_policy_is_read_once_per_fleet_snapshot() -> None:
    """Expanding two roles that share a policy costs one expansion, not two.

    Every cluster's executor role tends to attach the same managed policies, and
    each attached one costs a ``get-policy``/``get-policy-version`` pair. Inside
    one snapshot the second role reuses what the first resolved, so the cost of
    the check grows with the number of distinct policies rather than with the
    size of the fleet.
    """

    calls: list[tuple[str, ...]] = []
    release = _iam_release(calls)

    with STATE_MODULE.read_snapshot(release):
        for cluster_id in ("gpu-a", "gpu-b"):
            IAM_MODULE.validate_executor_iam_role(release, _iam_target(cluster_id))

    verbs = [item[2] for item in calls]

    assert verbs.count("get-policy") == 1, verbs
    assert verbs.count("get-policy-version") == 1, verbs
    # The role listings are per role and must not collapse: the roles differ.
    assert verbs.count("list-attached-role-policies") == 2, verbs
    assert not any("--region" in item for item in calls), (
        "IAM is global, and a --region would make the cache key differ from the "
        "command actually issued"
    )


def test_executor_policy_reads_do_not_survive_the_snapshot() -> None:
    """A later phase re-reads the role, because it may have been changed."""

    calls: list[tuple[str, ...]] = []
    release = _iam_release(calls)

    for _ in range(2):
        with STATE_MODULE.read_snapshot(release):
            IAM_MODULE.validate_executor_iam_role(release, _iam_target("gpu-a"))

    assert [item[2] for item in calls].count("get-policy-version") == 2


def test_executor_iam_boundary_rejects_allow_not_action() -> None:
    with pytest.raises(MODULE.ReleaseError, match="Allow/NotAction"):
        MODULE.validate_executor_iam_documents(
            "arn:aws:iam::1:role/executor",
            [{"Statement": [{"Effect": "Allow", "NotAction": "iam:*"}]}],
        )


def test_a_failed_captured_command_reports_what_it_printed(capsys) -> None:
    """A gate that fails has to say what it found, on whichever stream it used.

    The verifiers this runner drives -- alerting, manual command order, artifact
    scanning -- print their defect list on stdout and exit non-zero. Forwarding
    only stderr reduced that to ``command failed (1): python3``, which blocks a
    release for a reason the operator cannot read and cannot act on.
    """

    runner = MODULE.Runner()
    script = (
        "import sys; "
        "print('FAIL: live AMP namespace is missing groups: x'); "
        "print('context on stderr', file=sys.stderr); "
        "sys.exit(1)"
    )

    with pytest.raises(MODULE.ReleaseError, match=r"exited with status 1") as caught:
        runner.run([sys.executable, "-c", script], capture=True)
    reported = capsys.readouterr()

    assert "missing groups: x" in reported.err
    assert "context on stderr" in reported.err
    # stdout carries the machine-readable release report, so a failing command's
    # output must not be mixed into it.
    assert reported.out == ""
    # The verifier is our own script and has just been quoted in full, so the
    # rollout's own exit adds no ``command failed (1): python3`` on top of it and
    # exits with the verifier's status.
    assert "command failed" not in str(caught.value)
    assert report_failure("ERROR", caught.value) == 1
    assert capsys.readouterr().err == ""


def test_a_failed_foreign_command_names_the_step_and_its_last_words(capsys) -> None:
    """``command failed (1): kubectl`` named a binary; the step is in the argv."""

    runner = MODULE.Runner()

    with pytest.raises(MODULE.ReleaseError) as caught:
        runner.run(
            ["bash", "-c", "echo 'error: timed out waiting' >&2; exit 1"], capture=True
        )

    assert str(caught.value) == (
        "command failed (1): bash -c echo 'error: timed out waiting' >&2; exit 1: "
        "error: timed out waiting"
    )
    assert report_failure("ERROR", caught.value) == 2
    reported = capsys.readouterr().err
    assert reported.endswith("ERROR: " + str(caught.value) + "\n"), reported


def test_a_failed_sensitive_command_stays_silent(capsys) -> None:
    """Sensitive commands are marked sensitive because of their output.

    Their argv is already withheld from the trace; echoing the body of a failure
    would put the token back on the terminal and into the deploy log, which is
    the one place this repository will not put it.
    """

    runner = MODULE.Runner()
    script = "import sys; print('token-value-abc'); sys.exit(1)"

    with pytest.raises(MODULE.ReleaseError, match=r"exited with status 1"):
        runner.run([sys.executable, "-c", script], capture=True, sensitive=True)
    reported = capsys.readouterr()

    assert "token-value-abc" not in reported.err
    assert "token-value-abc" not in reported.out
    assert "<sensitive command>" in reported.err


def test_release_renders_one_runtime_image_across_gpu_roles(
    tmp_path, monkeypatch
) -> None:
    runtime_image = "registry.example/gpu-fault/python@sha256:" + "a" * 64
    monkeypatch.setenv("GPU_FAULT_RUNTIME_IMAGE", runtime_image)
    config = MODULE.ReleaseConfig.load(config_file(tmp_path))

    class RecordingRunner:
        dry_run = True

        def __init__(self) -> None:
            self.calls = []

        def run(self, args, **kwargs):
            self.calls.append((args, kwargs))
            return ""

    runner = RecordingRunner()
    release = MODULE.RegionalRelease(config, runner)
    target = config.clusters[0]

    release._apply_gpu_deployments(target, release.wheel_cm)
    rendered = [
        kwargs["input_text"]
        for args, kwargs in runner.calls
        if kwargs.get("input_text") and "--dry-run=server" not in args
    ]
    dry_run = [
        kwargs["input_text"]
        for args, kwargs in runner.calls
        if kwargs.get("input_text") and "--dry-run=server" in args
    ]

    assert len(rendered) == 3
    assert len(dry_run) == 3
    assert sorted(dry_run) == sorted(rendered)
    assert all(runtime_image in item for item in rendered), (
        f"every GPU manifest must carry the configured image {runtime_image}"
    )
    assert all(MODULE.DEFAULT_RUNTIME_IMAGE not in item for item in rendered), (
        "a manifest kept the built-in default image, so the configured one was "
        "only added rather than substituted"
    )
    assert any(f"value: {REGION}" in item for item in rendered), (
        "GPU manifests did not receive the configured Region"
    )
    assert all("REPLACE_WITH_AWS_REGION" not in item for item in rendered), (
        "GPU manifests retained an unresolved Region placeholder"
    )
    release._deploy_reconciler(
        target,
        wheel_cm=release.wheel_cm,
        bundle_cm=release.bundle_cm,
        artifact_sha=release.wheel_sha,
        config_digest=config.agent_config_digest,
    )
    reconciler_call = next(
        kwargs
        for args, kwargs in runner.calls
        if args and str(args[0]).endswith("deploy-node-installer-reconciler.sh")
    )
    assert reconciler_call["env"]["GPU_FAULT_RUNTIME_IMAGE"] == runtime_image
    assert reconciler_call["env"]["GPU_FAULT_RUNTIME_PROFILE"] == "hyperpod-v1"
    assert reconciler_call["env"]["GPU_FAULT_CLUSTER_ID"] == "gpu-a"
    assert reconciler_call["env"]["GPU_FAULT_HYPERPOD_CLUSTER"] == "hp-gpu-a"


def test_previous_node_installer_image_survives_an_empty_cluster_set() -> None:
    """Removing the last GPU cluster leaves no reconciler to read the Node
    Installer image from; the snapshot keeps the recorded image instead of
    judging the empty mapping inconsistent (remove-cluster's final sync-state
    failed there on 2026-09-12). With clusters the live reading still rules."""

    from gpu_fault_release import regional_release_state as state_module

    assert (
        state_module.previous_node_installer_image(
            capture_gpu=True,
            images={},
            recorded="repo@sha256:" + "a" * 64,
            configured="x",
        )
        == "repo@sha256:" + "a" * 64
    )
    assert (
        state_module.previous_node_installer_image(
            capture_gpu=True,
            images={},
            recorded="",
            configured="repo@sha256:" + "b" * 64,
        )
        == "repo@sha256:" + "b" * 64
    )
    assert (
        state_module.previous_node_installer_image(
            capture_gpu=True,
            images={"gpu-a": "repo@sha256:" + "c" * 64},
            recorded="repo@sha256:" + "a" * 64,
            configured="x",
        )
        == "repo@sha256:" + "c" * 64
    )
    with pytest.raises(state_module.ReleaseError, match="inconsistent"):
        state_module.previous_node_installer_image(
            capture_gpu=True,
            images={
                "gpu-a": "repo@sha256:" + "c" * 64,
                "gpu-b": "repo@sha256:" + "d" * 64,
            },
            recorded="",
            configured="",
        )
