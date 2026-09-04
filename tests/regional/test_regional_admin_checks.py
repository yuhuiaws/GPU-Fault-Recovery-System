from __future__ import annotations

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
CHECKS = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_admin_checks.py"
)
EVIDENCE = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_validation_evidence.py"
)
MONITORING = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_monitoring_safety.py"
)
PROBES = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_probes.py"
)
STATE = lazy_script_module(
    ROOT / "deploy/control-plane/regional/regional_release_state.py"
)


def _checks_module():
    return CHECKS.load()


class Config:
    site_name = "test-site"
    clusters = []


class Release:
    config = Config()


def test_health_report_exit_code_tracks_failures() -> None:
    healthy = CHECKS._report(
        "verify",
        Release(),
        [{"name": "a", "status": "PASS", "summary": "ok", "details": None}],
    )
    unhealthy = CHECKS._report(
        "verify",
        Release(),
        [{"name": "a", "status": "FAIL", "summary": "broken", "details": None}],
    )

    assert healthy["healthy"] is True
    assert CHECKS.report_exit_code(healthy) == 0
    assert unhealthy["healthy"] is False
    assert CHECKS.report_exit_code(unhealthy) == 1


def test_monitoring_rejects_duplicate_email_subscription_states() -> None:
    confirmed = {
        "Protocol": "email",
        "Endpoint": "ops@example.com",
        "SubscriptionArn": "arn:aws:sns:us-east-1:123456789012:topic:confirmed",
    }
    pending = {
        "Protocol": "email",
        "Endpoint": "ops@example.com",
        "SubscriptionArn": "PendingConfirmation",
    }

    with pytest.raises(CHECKS.ReleaseError, match="duplicate confirmed"):
        MONITORING.email_subscription_summary(
            [
                confirmed,
                {**confirmed, "SubscriptionArn": confirmed["SubscriptionArn"] + "2"},
            ],
            "ops@example.com",
        )
    with pytest.raises(CHECKS.ReleaseError, match="both confirmed and pending"):
        MONITORING.email_subscription_summary([confirmed, pending], "ops@example.com")


def test_quick_validation_evidence_requires_current_release_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {"release_id": "release-a", "phase": "complete"}
    evidence = tmp_path / "quick-validation.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_id": "release-a",
                "release_delivery_sha256": "a" * 64,
                "site_identity": {
                    "site_name": "test-site",
                    "aws_region": "us-east-1",
                    "cpu_eks_arn": ("arn:aws:eks:us-east-1:123456789012:cluster/cpu"),
                    "cluster_ids": ["gpu-a"],
                },
                "verified_at_epoch": int(time.time()),
                "checks": ["runtime_component_identity"],
                "release_state_sha256": EVIDENCE.canonical_sha256(state),
            }
        ),
        encoding="utf-8",
    )
    evidence.chmod(0o600)
    monkeypatch.setenv(CHECKS.QUICK_VALIDATION_EVIDENCE_ENV, str(evidence))
    release = SimpleNamespace(
        release_id="release-a",
        config=SimpleNamespace(
            site_name="test-site",
            aws_region="us-east-1",
            cpu_eks_arn="arn:aws:eks:us-east-1:123456789012:cluster/cpu",
            clusters=[SimpleNamespace(cluster_id="gpu-a")],
            release_delivery_sha256="a" * 64,
        ),
        _load_state=lambda: state,
    )

    value, fallback = EVIDENCE.quick_validation_evidence(release)

    assert fallback is None
    assert value is not None
    state["phase"] = "changed"
    value, fallback = EVIDENCE.quick_validation_evidence(release)
    assert value is None
    assert "changed after quick validation" in str(fallback)


def test_check_aggregates_exceptions_without_stopping() -> None:
    def broken():
        raise RuntimeError("boom")

    result = CHECKS._check("broken", broken)

    assert result["status"] == "FAIL"
    assert result["summary"] == "boom"


def test_empty_node_action_keys_are_valid_for_an_empty_gpu_registry(
    monkeypatch,
) -> None:
    values = {
        "gpu-fault-aurora": {"data": {"postgres-url": "x", "master-secret-arn": "y"}},
        "gpu-fault-control-plane-active": {
            "data": {
                "execution-token": base64.b64encode(b"a" * 32).decode(),
                "processor-replay-secret": base64.b64encode(b"b" * 32).decode(),
                "node-action-secret": base64.b64encode(b"c" * 32).decode(),
            }
        },
        "gpu-fault-node-action-keys": {"data": None},
    }
    monkeypatch.setattr(CHECKS, "_secret", lambda _release, name: values[name])
    release = SimpleNamespace(config=SimpleNamespace(clusters=[]))

    result = CHECKS.check_cpu_secrets(release)

    assert result.details["node_action_key_count"] == 0


def test_email_check_validates_declared_routing_and_site_context(monkeypatch) -> None:
    module = _checks_module()
    values = {
        "email-sender": "sender@example.com",
        "email-recipients": "ops@example.com,oncall@example.com",
        "email-subject-prefix": "[PROD]",
        "site-id": "site-a",
        "aws-account-id": "123456789012",
    }
    secret = {
        "data": {
            key: base64.b64encode(value.encode()).decode()
            for key, value in values.items()
        }
    }
    responses = iter(
        [
            {"VerifiedForSendingStatus": True},
            {"SendingEnabled": True, "ProductionAccessEnabled": True},
        ]
    )
    monkeypatch.setattr(module, "_aws_json", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(module, "_secret", lambda *_args, **_kwargs: secret)
    release = SimpleNamespace(
        config=SimpleNamespace(
            site_name="site-a",
            cpu_eks_arn="arn:aws:eks:us-west-2:123456789012:cluster/cpu",
            notifications=SimpleNamespace(
                allow_email=True,
                acknowledge_external_alert_channel=False,
                admin_email="owner@example.com",
                email_sender="sender@example.com",
                email_recipients=("ops@example.com", "oncall@example.com"),
                email_subject_prefix="[PROD]",
            ),
        )
    )

    result = module.check_email_notifications(release)

    assert result.details["recipient_count"] == 2
    assert result.details["site_id"] == "site-a"


def test_preflight_report_contains_all_required_domains(monkeypatch) -> None:
    module = _checks_module()
    names = (
        "_check_tools",
        "_check_local_inputs",
        "_check_aws_identity",
        "_check_contexts",
        "_check_cpu_capacity",
        "check_cpu_secrets",
        "_check_load_balancer_controller",
        "_check_nlb_inputs",
        "_check_aurora",
        "check_aurora_refresh",
        "check_email_notifications",
        "_check_monitoring",
        "workflow_safety_snapshot",
    )
    for name in names:
        monkeypatch.setattr(
            module, name, lambda *args, name=name: module.CheckValue(name)
        )

    report = module.build_preflight_report(Release())

    assert report["healthy"] is True
    assert report["summary"]["PASS"] == len(names)
    assert {item["name"] for item in report["checks"]} == {
        "tools",
        "local_inputs",
        "aws_identity",
        "regional_contexts",
        "cpu_capacity",
        "cpu_secrets",
        "load_balancer_controller",
        "nlb_inputs",
        "aurora",
        "aurora_credential_refresh",
        "email_notifications",
        "monitoring",
        "workflow_safety",
    }


def test_aurora_refresh_check_rejects_rbac_target_drift() -> None:
    module = _checks_module()
    cronjob = {
        "spec": {
            "jobTemplate": {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "env": [
                                        {
                                            "name": (
                                                "GPU_FAULT_AURORA_RESTART_DEPLOYMENTS"
                                            ),
                                            "value": (
                                                "gpu-fault-api-ha,"
                                                "gpu-fault-control-worker"
                                            ),
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                }
            }
        },
        "status": {},
    }
    role = {
        "rules": [
            {
                "apiGroups": ["apps"],
                "resources": ["deployments"],
                "resourceNames": ["gpu-fault-api-ha"],
                "verbs": ["get", "patch"],
            }
        ]
    }

    class AuroraRelease:
        config = SimpleNamespace(namespace="gpu-fault-system")

        @staticmethod
        def _cpu(*args):
            return args

        @staticmethod
        def _get_json(command):
            return role if "role" in command else cronjob

    with pytest.raises(module.ReleaseError, match="RBAC differs"):
        module.check_aurora_refresh(AuroraRelease())


def test_aurora_refresh_check_reports_converged_targets() -> None:
    module = _checks_module()
    targets = ["gpu-fault-api-ha", "gpu-fault-control-worker"]
    cronjob = {
        "spec": {
            "jobTemplate": {
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "env": [
                                        {
                                            "name": (
                                                "GPU_FAULT_AURORA_RESTART_DEPLOYMENTS"
                                            ),
                                            "value": ",".join(targets),
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                }
            }
        },
        "status": {},
    }
    role = {
        "rules": [
            {
                "apiGroups": ["apps"],
                "resources": ["deployments"],
                "resourceNames": targets,
                "verbs": ["get", "patch"],
            }
        ]
    }

    class AuroraRelease:
        config = SimpleNamespace(namespace="gpu-fault-system")

        @staticmethod
        def _cpu(*args):
            return args

        @staticmethod
        def _get_json(command):
            if "role" in command:
                return role
            if "job" in command:
                return {"items": []}
            return cronjob

    value = module.check_aurora_refresh(AuroraRelease())

    assert value.details["deployment_targets"] == targets


def test_health_report_uses_bounded_parallel_checks(monkeypatch) -> None:
    module = _checks_module()
    clusters = [
        SimpleNamespace(cluster_id="gpu-a"),
        SimpleNamespace(cluster_id="gpu-b"),
    ]
    release = SimpleNamespace(
        config=SimpleNamespace(site_name="test-site", clusters=clusters)
    )
    for name in (
        "_check_contexts",
        "check_cpu_secrets",
        "_check_cpu_workloads",
        "check_email_notifications",
        "check_aurora_refresh",
        "_verify_profile",
        "run_read_only_verifiers",
        "_check_runtime_component_identity",
        "_check_control_api",
        "_check_gpu_cluster",
        "_check_nlb_runtime",
        "_check_aurora",
        "_check_monitoring",
    ):
        monkeypatch.setattr(
            module, name, lambda *_args, name=name, **_kwargs: module.CheckValue(name)
        )

    worker_counts: list[int] = []
    submitted: list[str] = []

    class Completed:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class RecordingExecutor:
        def __init__(self, *, max_workers):
            worker_counts.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def submit(self, function, name, callback):
            submitted.append(name)
            return Completed(function(name, callback))

    monkeypatch.setattr(module, "ThreadPoolExecutor", RecordingExecutor)

    report = module.build_health_report(release, mode="verify")

    assert worker_counts == [8]
    assert submitted == [
        "regional_contexts",
        "cpu_secrets",
        "cpu_workloads",
        "email_notifications",
        "aurora_credential_refresh",
        "runtime_profile",
        "read_only_verifiers",
        "runtime_component_identity",
        "control_api",
        "gpu_cluster:gpu-a",
        "gpu_cluster:gpu-b",
        "nlb_runtime",
        "aurora",
        "monitoring",
    ]
    assert [item["name"] for item in report["checks"]] == submitted


def test_read_only_verifiers_reuse_exact_cpu_and_gpu_checks() -> None:
    module = _checks_module()
    targets = [
        SimpleNamespace(cluster_id="gpu-a", context="context-a"),
        SimpleNamespace(cluster_id="gpu-b", context="context-b"),
    ]
    release = SimpleNamespace(
        config=SimpleNamespace(
            namespace="gpu-fault-system",
            cpu_kubeconfig="/secure/cpu.kubeconfig",
            clusters=targets,
        ),
        runtime_image="registry/runtime@sha256:" + "a" * 64,
        executor_wheel_cm="gpu-fault-executor-wheel",
        runner=SimpleNamespace(
            run=lambda *_args, **_kwargs: pytest.fail(
                "an evidence-covered verifier was executed again"
            )
        ),
    )

    value = module.run_read_only_verifiers(
        release,
        reused_checks={
            "control_plane_role_split",
            "data_plane_executor:gpu-a",
            "data_plane_executor:gpu-b",
        },
    )

    assert value.details == {
        "cpu": {"reused": True},
        "clusters": {"gpu-a": {"reused": True}, "gpu-b": {"reused": True}},
    }


def control_api_release(
    *,
    executor_internal_error_baseline: int = 0,
    executor_internal_error_timestamp_baseline: float | None = 0.0,
) -> SimpleNamespace:
    previous: dict[str, int | float] = {
        "executor_internal_error_total": executor_internal_error_baseline
    }
    if executor_internal_error_timestamp_baseline is not None:
        previous["executor_internal_error_last_seen_timestamp_seconds"] = (
            executor_internal_error_timestamp_baseline
        )
    return SimpleNamespace(
        wheel_sha="a" * 64,
        node_wheel_sha="a" * 64,
        executor_wheel_sha="d" * 64,
        config=SimpleNamespace(
            agent_config_digest="b" * 64,
            component_digests={"node_runtime": "e" * 64, "executor": "f" * 64},
            runtime_profile_version="hyperpod-v1",
            health=SimpleNamespace(remote_command_max_unclaimed_seconds=300),
            clusters=[SimpleNamespace(cluster_id="gpu-a")],
        ),
        _load_state=lambda: {"previous": previous},
    )


def healthy_control_api_report() -> dict:
    return {
        "healthz": {"status": "ok"},
        "version": {
            "deployment_mode": "regional",
            "required_agent_artifact_sha256": "a" * 64,
            "required_agent_config_digest": "b" * 64,
            "compatible_agent_artifact_sha256s": [],
            "required_agent_compatibility_digest": "e" * 64,
            "compatible_agent_compatibility_digests": [],
            "compatible_agent_protocol_versions": [],
            "compatible_agent_config_digests": [],
            "required_runtime_profile_version": "hyperpod-v1",
            "required_node_action_key_version": 2,
            "required_regional_executor_protocol_version": 2,
            "compatible_regional_executor_protocol_versions": [],
            "required_regional_executor_artifact_sha256": "d" * 64,
            "compatible_regional_executor_artifact_sha256s": [],
            "required_regional_executor_compatibility_digest": "f" * 64,
            "compatible_regional_executor_compatibility_digests": [],
        },
        "registry": [{"cluster_id": "gpu-a"}],
        "clusters": {
            "gpu-a": {
                "expected_node_ids": ["node-a"],
                "agents": [
                    {
                        "node_id": "node-a",
                        "lifecycle_state": "ACTIVE",
                        "runtime_profile_version": "hyperpod-v1",
                    }
                ],
                "fleet_readiness": {"ready": True},
                "collector_readiness": {"ready": True},
            }
        },
        "remote_commands": {
            "oldest_unclaimed_age_seconds": 0,
            "executor_internal_error_total": 0,
            "executor_internal_error_last_seen_timestamp_seconds": 0.0,
        },
    }


def control_api_check(monkeypatch, report: dict):
    module = CHECKS.load()
    monkeypatch.setattr(module, "_control_api_report", lambda _release: report)
    return module, module._check_control_api


@pytest.mark.parametrize(("baseline", "current"), ((0, 0), (2, 2)))
def test_control_api_health_accepts_a_converged_fleet(
    monkeypatch, baseline: int, current: int
) -> None:
    report = healthy_control_api_report()
    report["remote_commands"]["executor_internal_error_total"] = current
    _module, check = control_api_check(monkeypatch, report)

    value = check(control_api_release(executor_internal_error_baseline=baseline))

    assert value.summary == "control-plane API, fleet and collectors are healthy"


def test_control_api_probe_does_not_inject_unknown_gpu_fault_variables() -> None:
    # The probe's inputs travel in the environment, and the Pod's own
    # environment already carries the execution token under a `GPU_FAULT_`
    # prefix; a probe input under that prefix would be indistinguishable from
    # one the runtime is entitled to read.
    source = PROBES.probe_source("control_api_inspect")

    assert "GPU_FAULT_ADMIN_" not in source
    assert "ADMIN_CLUSTER_IDS_JSON" in source
    assert "ADMIN_EXPECTED_NODES_JSON" in source


def test_control_api_health_accepts_internal_error_retention_cleanup(
    monkeypatch,
) -> None:
    report = healthy_control_api_report()
    report["remote_commands"].update(
        {
            "executor_internal_error_total": 1,
            "executor_internal_error_last_seen_timestamp_seconds": 100.0,
        }
    )
    _module, check = control_api_check(monkeypatch, report)

    value = check(
        control_api_release(
            executor_internal_error_baseline=2,
            executor_internal_error_timestamp_baseline=100.0,
        )
    )

    assert value.summary == "control-plane API, fleet and collectors are healthy"


def test_control_api_health_rejects_new_internal_error_timestamp(monkeypatch) -> None:
    report = healthy_control_api_report()
    report["remote_commands"].update(
        {
            "executor_internal_error_total": 1,
            "executor_internal_error_last_seen_timestamp_seconds": 101.0,
        }
    )
    module, check = control_api_check(monkeypatch, report)

    with pytest.raises(module.ReleaseError, match="observed during release"):
        check(
            control_api_release(
                executor_internal_error_baseline=2,
                executor_internal_error_timestamp_baseline=100.0,
            )
        )


def test_control_api_health_uses_count_for_legacy_release_state(monkeypatch) -> None:
    report = healthy_control_api_report()
    report["remote_commands"]["executor_internal_error_total"] = 1
    module, check = control_api_check(monkeypatch, report)

    with pytest.raises(module.ReleaseError, match="increased during release"):
        check(control_api_release(executor_internal_error_timestamp_baseline=None))


def test_control_api_health_uses_count_when_current_api_has_no_timestamp(
    monkeypatch,
) -> None:
    report = healthy_control_api_report()
    report["remote_commands"].pop("executor_internal_error_last_seen_timestamp_seconds")
    report["remote_commands"]["executor_internal_error_total"] = 1
    module, check = control_api_check(monkeypatch, report)

    with pytest.raises(module.ReleaseError, match="increased during release"):
        check(
            control_api_release(
                executor_internal_error_baseline=0,
                executor_internal_error_timestamp_baseline=0.0,
            )
        )


def test_executor_verifier_is_pinned_to_the_executor_wheel() -> None:
    """The data-plane verifier must be told the Executor wheel, not the CPU one.

    The control plane, the Executor and the node runtime are separate wheels
    with separate digests. Handing the verifier the control-plane ConfigMap
    would make it compare an Executor Pod against an artifact it never loads,
    and the check would pass through real Executor drift.
    """

    environments: list[dict[str, str]] = []

    class Runner:
        def run(self, _arguments, *, env=None, **_kwargs):
            environments.append(dict(env or {}))
            return "{}"

    release = SimpleNamespace(
        runner=Runner(),
        runtime_image="runtime:candidate",
        executor_wheel_cm="gpu-fault-executor-wheel-abc",
        cpu_wheel_cm="gpu-fault-control-plane-wheel-def",
        config=SimpleNamespace(
            namespace="gpu-fault-system",
            cpu_kubeconfig="/tmp/cpu.kubeconfig",
            clusters=[SimpleNamespace(cluster_id="gpu-a", context="gpu-a")],
        ),
    )

    EVIDENCE.read_only_verifier_details(release)

    pins = [
        environment["GPU_FAULT_EXPECTED_WHEEL_CONFIGMAP"]
        for environment in environments
        if "GPU_FAULT_EXPECTED_WHEEL_CONFIGMAP" in environment
    ]
    assert pins == [release.executor_wheel_cm]


def test_gpu_cluster_check_is_pinned_to_the_node_wheel(monkeypatch) -> None:
    """Node installer drift is measured against the node runtime wheel.

    The installer annotates each node with the artifact digest it installed. It
    installs the node-runtime wheel, so comparing that annotation with the
    control-plane wheel digest would report drift on every healthy node -- and
    comparing it with nothing at all would hide a node still running the
    previous release.
    """

    module = _checks_module()
    node_wheel_sha = "n" * 64
    installed = node_wheel_sha

    def nodes() -> dict:
        return {
            "items": [
                {
                    "metadata": {
                        "name": "node-a",
                        "annotations": {
                            "gpu-fault.io/installer-state": "Succeeded",
                            "gpu-fault.io/installer-artifact-sha256": installed,
                        },
                    },
                    "status": {"conditions": [{"type": "Ready", "status": "True"}]},
                }
            ]
        }

    def get_json(arguments):
        if "nodes" in arguments:
            return nodes()
        return {"spec": {"replicas": 1}, "status": {"readyReplicas": 1}}

    class Runner:
        def run(self, arguments, **_kwargs):
            return '{"status": "ok"}' if "-c" in arguments else "ready"

    release = SimpleNamespace(
        runner=Runner(),
        node_wheel_sha=node_wheel_sha,
        cpu_wheel_sha="c" * 64,
        config=SimpleNamespace(
            site_name="test-site",
            namespace="gpu-fault-system",
            clusters=[
                SimpleNamespace(
                    cluster_id="gpu-a", context="gpu-a", hyperpod_cluster_name="hp-a"
                )
            ],
        ),
        _get_json=get_json,
        _gpu=lambda _target, *arguments: list(arguments),
    )
    for name in (
        "_check_contexts",
        "check_cpu_secrets",
        "_check_cpu_workloads",
        "check_email_notifications",
        "check_aurora_refresh",
        "_verify_profile",
        "run_read_only_verifiers",
        "_check_runtime_component_identity",
        "_check_control_api",
        "_check_nlb_runtime",
        "_check_aurora",
        "_check_monitoring",
    ):
        monkeypatch.setattr(
            module, name, lambda *_args, name=name, **_kwargs: module.CheckValue(name)
        )

    def gpu_check() -> dict:
        report = module.build_health_report(release, mode="verify")
        return next(
            item for item in report["checks"] if item["name"] == "gpu_cluster:gpu-a"
        )

    assert gpu_check()["status"] == "PASS"
    installed = release.cpu_wheel_sha
    drifted = gpu_check()
    assert drifted["status"] == "FAIL"
    assert "artifact drift" in drifted["summary"]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"oldest_unclaimed_age_seconds": 301}, "oldest unclaimed"),
        (
            {
                "executor_internal_error_total": 1,
                "executor_internal_error_last_seen_timestamp_seconds": 1.0,
            },
            "internal",
        ),
    ],
)
def test_control_api_health_rejects_stuck_or_broken_commands(
    monkeypatch, updates: dict[str, int | float], message: str
) -> None:
    report = healthy_control_api_report()
    report["remote_commands"].update(updates)
    module, check = control_api_check(monkeypatch, report)

    with pytest.raises(module.ReleaseError, match=message):
        check(control_api_release())


def test_certificate_hostname_matching_is_single_label() -> None:
    module = CHECKS.load()

    assert module._hostname_matches("control.example", "control.example")
    assert module._hostname_matches("*.example", "control.example")
    assert not module._hostname_matches("*.example", "nested.control.example")


def _nlb_release(calls: list[str], *, certificate_status: str = "ISSUED"):
    not_after = datetime.now(timezone.utc) + timedelta(days=120)
    responses = {
        "describe-load-balancers": {
            "LoadBalancers": [
                {
                    "LoadBalancerArn": "arn:nlb",
                    "State": {"Code": "active"},
                    "DNSName": "nlb.example",
                }
            ]
        },
        "describe-listeners": {
            "Listeners": [
                {
                    "Port": 443,
                    "Protocol": "TLS",
                    "Certificates": [{"CertificateArn": "arn:cert"}],
                }
            ]
        },
        "describe-target-groups": {"TargetGroups": [{"TargetGroupArn": "arn:tg"}]},
        "describe-target-health": {
            "TargetHealthDescriptions": [
                {"TargetHealth": {"State": "healthy"}} for _ in range(3)
            ]
        },
        "describe-certificate": {
            "Certificate": {
                "Status": certificate_status,
                "NotAfter": not_after.isoformat(),
                "SubjectAlternativeNames": ["control.example"],
            }
        },
    }

    class Runner:
        @staticmethod
        def run(arguments, **_kwargs):
            operation = arguments[2]
            calls.append(operation)
            return json.dumps(responses[operation])

    return SimpleNamespace(
        runner=Runner(),
        config=SimpleNamespace(
            site_name="test-site",
            aws_region="us-east-1",
            nlb={"name": "gpu-fault-regional", "certificate_arn": "arn:cert"},
            health=SimpleNamespace(certificate_min_validity_days=30),
            clusters=[],
        ),
    )


def test_nlb_runtime_check_asks_independent_questions_together() -> None:
    """One check is one task, so its own round trips have to overlap.

    The load balancer and its certificate are named by configuration, and the
    listeners and target groups are both keyed by the load balancer ARN; only the
    target health has to wait for an answer. Read as a straight sequence this
    check was five serial AWS round trips inside a single health-report task,
    where the report's thread pool could not help it.
    """

    module = CHECKS.load()
    calls: list[str] = []
    release = _nlb_release(calls)
    release._read_snapshot = lambda: STATE.read_snapshot(release)

    with module._read_snapshot(release):
        value = module._check_nlb_runtime(release)

    assert value.details["healthy_targets"] == 3
    assert value.details["certificate"]["arn"] == "arn:cert"
    assert sorted(calls) == [
        "describe-certificate",
        "describe-listeners",
        "describe-load-balancers",
        "describe-target-groups",
        "describe-target-health",
    ]


def test_nlb_runtime_check_reports_the_first_failure_in_program_order() -> None:
    """Overlapping the reads must not make the reason depend on scheduling.

    A certificate that is about to expire and a load balancer that is missing are
    now read at the same time. The reason reported is the one the sequential
    version reported -- the load balancer -- because the futures are resolved in
    submission order rather than as they complete.
    """

    module = CHECKS.load()
    calls: list[str] = []
    release = _nlb_release(calls, certificate_status="PENDING_VALIDATION")
    release.config.nlb = {"name": "missing", "certificate_arn": "arn:cert"}
    original = module._aws_json

    def missing_load_balancer(release_value, arguments):
        if arguments[1] == "describe-load-balancers":
            return {"LoadBalancers": []}
        return original(release_value, arguments)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "_aws_json", missing_load_balancer)
        with pytest.raises(module.ReleaseError, match="was not found"):
            module._check_nlb_runtime(release)


def _aws_cache_release(calls: list[tuple[str, ...]], *, delay: float = 0.0):
    class Runner:
        @staticmethod
        def run(arguments, **_kwargs):
            calls.append(tuple(arguments))
            if delay:
                time.sleep(delay)
            return json.dumps({"call": len(calls)})

    release = SimpleNamespace(
        config=SimpleNamespace(site_name="test-site", aws_region="us-east-1"),
        runner=Runner(),
    )
    release._read_snapshot = lambda: STATE.read_snapshot(release)
    return release


def test_repeated_aws_reads_are_served_once_per_snapshot() -> None:
    """One report asks AWS the same question from several checks.

    `_check_aurora` and the preflight NLB check both describe the same ACM
    certificate, and the per-subnet route test falls back to the same VPC main
    route table once per subnet. Those are reads of one observation, so the
    snapshot answers them once -- and stops answering when it closes, because
    outside a snapshot there is no observation to be consistent with.
    """

    module = CHECKS.load()
    calls: list[tuple[str, ...]] = []
    release = _aws_cache_release(calls)
    query = ["acm", "describe-certificate", "--certificate-arn", "arn:acm:cert"]

    with module._read_snapshot(release):
        first = module._aws_json(release, query)
        second = module._aws_json(release, query)
    third = module._aws_json(release, query)

    assert first == second == {"call": 1}
    assert third == {"call": 2}
    assert len(calls) == 2


def test_concurrent_aws_reads_collapse_into_one_call() -> None:
    """The checks run on a thread pool, so the duplicates arrive at once.

    A cache that stored values rather than promises would let both threads miss
    before either finished, and the collapsing this exists for would only work
    for callers that happened not to overlap -- which, on a thread pool, is the
    ones that did not need it.
    """

    module = CHECKS.load()
    calls: list[tuple[str, ...]] = []
    release = _aws_cache_release(calls, delay=0.2)
    query = ["ec2", "describe-route-tables", "--filters", "Name=vpc-id,Values=vpc-a"]

    with module._read_snapshot(release):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(
                pool.map(lambda _index: module._aws_json(release, query), range(4))
            )

    assert results == [{"call": 1}] * 4
    assert len(calls) == 1


def test_state_changing_aws_calls_are_never_cached() -> None:
    """Only `describe-`/`get-`/`list-` may be served from the cache.

    Caching a call that changes AWS state would silently drop every repeat of
    it, and caching a read a caller is polling would make it wait for a value
    that can never arrive.
    """

    module = CHECKS.load()
    calls: list[tuple[str, ...]] = []
    release = _aws_cache_release(calls)
    mutation = ["ec2", "create-tags", "--resources", "i-abc"]

    with module._read_snapshot(release):
        module._aws_json(release, mutation)
        module._aws_json(release, mutation)

    assert len(calls) == 2
    assert not module._aws_read_only(mutation), "create-tags mutates and cannot cache"
    assert module._aws_read_only(["acm", "describe-certificate"]), "describe- is read"
    assert module._aws_read_only(["sesv2", "get-account"]), "get- is read"
    assert module._aws_read_only(["sns", "list-subscriptions-by-topic"]), (
        "list- is read"
    )
