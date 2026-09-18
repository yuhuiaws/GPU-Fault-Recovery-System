from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.capacity_defaults import INGRESS_REPLICAS
from gpu_fault_release import regional_admin_checks as checks
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional._cov95_release_checks import (
    CheckedRelease,
    finding,
    install_api_transport,
)


@pytest.fixture
def release(monkeypatch: pytest.MonkeyPatch) -> CheckedRelease:
    release = CheckedRelease()
    install_api_transport(monkeypatch, release)
    monkeypatch.setattr(checks.shutil, "which", lambda name: f"/example/bin/{name}")
    return release


def test_quick_report_validates_cpu_roles_and_public_api(
    release: CheckedRelease,
) -> None:
    report = checks.build_quick_health_report(release)
    assert report["healthy"] is True
    assert report["summary"] == {"PASS": 2, "WARN": 0, "FAIL": 0, "SKIP": 0}
    assert report["scope"] == "quick"
    assert report["reused_validation_checks"] == []
    assert release.prime_calls == 1
    assert checks.report_exit_code(report) == 0


@pytest.mark.parametrize(
    "field,value,problem",
    [
        ("deployment_mode", "standalone", "not running in regional mode"),
        ("required_agent_artifact_sha256", "other", "required Agent artifact"),
        ("required_agent_config_digest", "other", "required Agent config"),
        ("compatible_agent_artifact_sha256s", ["old"], "artifact compatibility window"),
        ("required_agent_compatibility_digest", "other", "Agent compatibility digest"),
        (
            "compatible_agent_compatibility_digests",
            ["old"],
            "Agent compatibility digest window",
        ),
        (
            "compatible_agent_protocol_versions",
            [2],
            "Agent protocol compatibility window",
        ),
        (
            "compatible_agent_config_digests",
            ["old"],
            "Agent config compatibility window",
        ),
        ("required_runtime_profile_version", "other", "required Runtime Profile"),
        ("required_node_action_key_version", 1, "unexpected Node Action key version"),
        (
            "compatible_regional_executor_protocol_versions",
            [1],
            "executor protocol compatibility window",
        ),
        (
            "required_regional_executor_artifact_sha256",
            "other",
            "required Executor artifact",
        ),
        (
            "compatible_regional_executor_artifact_sha256s",
            ["old"],
            "executor artifact compatibility window",
        ),
        (
            "required_regional_executor_compatibility_digest",
            "other",
            "required Executor compatibility digest",
        ),
        (
            "compatible_regional_executor_compatibility_digests",
            ["old"],
            "executor compatibility digest window",
        ),
    ],
)
def test_quick_report_rejects_every_open_or_drifted_pin(
    release: CheckedRelease, field: str, value: Any, problem: str
) -> None:
    release.api["version"][field] = value
    report = checks.build_quick_health_report(release)
    result = finding(report, "control_api")
    assert result["status"] == "FAIL"
    assert problem in result["summary"]
    assert checks.report_exit_code(report) == 1


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("shape", "non-object evidence"),
        ("health", "/healthz is not ok"),
        ("registry", "registry drift"),
        ("agents", "no registered Node Agents"),
        ("coverage", "Agent coverage drift"),
        ("lifecycle", "non-ACTIVE Node Agent"),
        ("profile", "Agent Runtime Profile drift"),
        ("fleet", "fleet readiness failed"),
        ("collector", "collector readiness failed"),
        ("remote-age", "oldest unclaimed remote"),
    ],
)
def test_quick_report_does_not_infer_health_from_partial_evidence(
    release: CheckedRelease, fault: str, problem: str
) -> None:
    if fault == "shape":
        release.api = []
    elif fault == "health":
        release.api["healthz"]["status"] = "failed"
    elif fault == "registry":
        release.api["registry"] = []
    elif fault == "remote-age":
        release.api["remote_commands"]["oldest_unclaimed_age_seconds"] = 31
    else:
        cluster = release.api["clusters"]["gpu-a"]
        if fault == "agents":
            cluster["agents"] = []
        elif fault == "coverage":
            cluster["expected_node_ids"] = ["different"]
        elif fault == "lifecycle":
            cluster["agents"][0]["lifecycle_state"] = "UNKNOWN"
        elif fault == "profile":
            cluster["agents"][0]["runtime_profile_version"] = "other"
        else:
            cluster["fleet_readiness" if fault == "fleet" else "collector_readiness"][
                "ready"
            ] = False
    result = finding(checks.build_quick_health_report(release), "control_api")
    assert result["status"] == "FAIL"
    assert problem in result["summary"]


@pytest.mark.parametrize(
    "baseline,current,problem",
    [
        ("invalid", 3, "timestamp is invalid"),
        (-1, 3, "timestamp is invalid"),
        (2, 3, "observed during release"),
        (3, 3, None),
        (3, 2, None),
    ],
)
def test_remote_error_timestamp_is_checked_even_when_retained_count_drops(
    release: CheckedRelease, baseline: Any, current: Any, problem: str | None
) -> None:
    field = "executor_internal_error_last_seen_timestamp_seconds"
    release.live_state = {
        "previous": {field: baseline, "executor_internal_error_total": 100}
    }
    release.api["remote_commands"] = {
        field: current,
        "executor_internal_error_total": 0,
    }
    result = finding(checks.build_quick_health_report(release), "control_api")
    assert result["status"] == ("FAIL" if problem else "PASS")
    if problem:
        assert problem in result["summary"]


def test_invalid_legacy_error_baseline_and_state_read_failure_remain_failures(
    release: CheckedRelease,
) -> None:
    release.live_state = {"previous": {"executor_internal_error_total": "invalid"}}
    result = finding(checks.build_quick_health_report(release), "control_api")
    assert result["status"] == "FAIL"
    assert "baseline is invalid" in result["summary"]
    release.live_state = ReleaseError("release state cannot be read")
    result = finding(checks.build_quick_health_report(release), "control_api")
    assert result["status"] == "FAIL"
    assert result["summary"] == "release state cannot be read"


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("unready", "Ready"),
        ("legacy", "overrides notification ConfigMap"),
        ("ingress-count", "exactly"),
        ("worker-zero", "control-worker is scaled to zero"),
        ("adot-zero", "adot is scaled to zero"),
        ("profile", "required Runtime Profile"),
        ("email", "email enablement"),
        ("ack", "external alert acknowledgement"),
    ],
)
def test_quick_cpu_health_rejects_partial_capacity_and_configuration_drift(
    release: CheckedRelease, fault: str, problem: str
) -> None:
    ingress = release.documents[("cpu", "deployment", "gpu-fault-api-ha")]
    if fault == "unready":
        ingress["status"]["readyReplicas"] = INGRESS_REPLICAS - 1
    elif fault == "legacy":
        ingress["spec"]["template"]["spec"]["containers"][0]["env"] = [
            {"name": "GPU_FAULT_ALLOW_EMAIL", "value": "false"}
        ]
    elif fault.endswith("zero"):
        name = (
            "gpu-fault-control-worker" if fault == "worker-zero" else "gpu-fault-adot"
        )
        release.documents[("cpu", "deployment", name)]["spec"]["replicas"] = 0
        release.documents[("cpu", "deployment", name)]["status"]["readyReplicas"] = 0
    elif fault == "ingress-count":
        ingress["spec"]["replicas"] = INGRESS_REPLICAS + 1
        ingress["status"]["readyReplicas"] = INGRESS_REPLICAS + 1
    else:
        field = {
            "profile": "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION",
            "email": "GPU_FAULT_ALLOW_EMAIL",
            "ack": "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL",
        }[fault]
        release.documents[("cpu", "configmap", "gpu-fault-api-ha-config-core")]["data"][
            field
        ] = "drift"
    result = finding(checks.build_quick_health_report(release), "cpu_workloads")
    assert result["status"] == "FAIL"
    assert problem in result["summary"]


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("none", None),
        ("missing", "no AWS Load Balancer Controller"),
        ("unready", "Ready"),
        ("zero", "Ready"),
    ],
)
def test_preflight_checks_controller_discovery_and_readiness(
    release: CheckedRelease, fault: str, problem: str | None
) -> None:
    items = release.documents[("cpu", "deployment", "")]["items"]
    if fault == "missing":
        items.pop()
    elif fault == "unready":
        items[-1]["status"]["readyReplicas"] = 0
    elif fault == "zero":
        items[-1]["spec"]["replicas"] = 0
        items[-1]["status"]["readyReplicas"] = 0
    result = finding(checks.build_preflight_report(release), "load_balancer_controller")
    assert result["status"] == ("FAIL" if problem else "PASS")
    if problem:
        assert problem in result["summary"]


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("none", None),
        ("missing-subnet", "do not exist"),
        ("vpc", "CPU EKS VPC"),
        ("az", "at least two AZs"),
        ("route", "route to an IGW"),
        ("sg", "Security Group is not"),
        ("rule", "exposes TCP 443"),
        ("certificate-expiry", "has no NotAfter"),
        ("expiring", "expires in"),
        ("sans", "SAN does not cover"),
        ("missing-config", "configuration is missing"),
    ],
)
def test_preflight_nlb_checks_topology_tls_and_explicit_exposure(
    release: CheckedRelease, fault: str, problem: str | None
) -> None:
    subnets = release.aws["ec2", "describe-subnets"]["Subnets"]
    certificate = release.aws["acm", "describe-certificate"]["Certificate"]
    if fault == "missing-subnet":
        subnets.pop()
    elif fault == "vpc":
        subnets[0]["VpcId"] = "other"
    elif fault == "az":
        subnets[1]["AvailabilityZone"] = subnets[0]["AvailabilityZone"]
    elif fault == "route":
        release.aws["ec2", "describe-route-tables"] = {"RouteTables": []}
    elif fault == "sg":
        release.aws["ec2", "describe-security-groups"]["SecurityGroups"] = []
    elif fault == "rule":
        release.aws["ec2", "describe-security-groups"]["SecurityGroups"][0][
            "IpPermissions"
        ][0]["Ipv6Ranges"] = [{"CidrIpv6": "::/0"}]
    elif fault == "certificate-expiry":
        certificate["NotAfter"] = None
    elif fault == "expiring":
        certificate["NotAfter"] = "2000-01-01T00:00:00Z"
    elif fault == "sans":
        certificate["SubjectAlternativeNames"] = ["other.test"]
    elif fault == "missing-config":
        release.config.nlb = {}
    result = finding(checks.build_preflight_report(release), "nlb_inputs")
    assert result["status"] == ("FAIL" if problem else "PASS")
    if problem:
        assert problem in result["summary"]


def test_nlb_subnets_can_use_main_route_table_and_single_label_certificate_wildcard(
    release: CheckedRelease,
) -> None:
    route = copy.deepcopy(release.aws["ec2", "describe-route-tables"])
    release.aws["ec2", "describe-route-tables"] = lambda arguments: (
        route
        if "Name=association.main,Values=true" in arguments
        else {"RouteTables": []}
    )
    release.aws["acm", "describe-certificate"]["Certificate"][
        "SubjectAlternativeNames"
    ] = ["*.example.test"]
    assert (
        finding(checks.build_preflight_report(release), "nlb_inputs")["status"]
        == "PASS"
    )
    release.config.clusters = tuple(
        replace(target, control_plane_url="https://deep.api.example.test")
        for target in release.config.clusters
    )
    assert (
        finding(checks.build_preflight_report(release), "nlb_inputs")["status"]
        == "FAIL"
    )


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("missing", "not found"),
        ("status", "is modifying"),
        ("members", "at least two"),
        ("instance-status", "at least two"),
        ("unconfigured", "not configured"),
    ],
)
def test_aurora_health_does_not_treat_partial_availability_as_ready(
    release: CheckedRelease, fault: str, problem: str
) -> None:
    if fault == "missing":
        release.aws["rds", "describe-db-clusters"] = {"DBClusters": []}
    elif fault == "status":
        release.aws["rds", "describe-db-clusters"]["DBClusters"][0]["Status"] = (
            "modifying"
        )
    elif fault == "unconfigured":
        release.config.health.aurora_cluster_id = None
    elif fault == "members":
        release.aws["rds", "describe-db-instances"]["DBInstances"].pop()
    else:
        release.aws["rds", "describe-db-instances"]["DBInstances"][0][
            "DBInstanceStatus"
        ] = "starting"
    result = finding(checks.build_preflight_report(release), "aurora")
    assert result["status"] == "FAIL"
    assert problem in result["summary"]


@pytest.mark.parametrize(
    "fault,problem",
    [("none", None), ("unconfigured", "not configured"), ("status", "expected ACTIVE")],
)
def test_monitoring_preflight_requires_active_workspace(
    release: CheckedRelease, fault: str, problem: str | None
) -> None:
    if fault == "unconfigured":
        release.config.health.amp_workspace_id = None
    elif fault == "status":
        release.aws["amp", "describe-workspace"]["workspace"]["status"][
            "statusCode"
        ] = "UPDATING"
    result = finding(checks.build_preflight_report(release), "monitoring")
    assert result["status"] == ("FAIL" if problem else "PASS")
    if problem:
        assert problem in result["summary"]
    else:
        assert result["details"]["confirmed_subscriptions"] == 1


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("none", None),
        ("empty", "no Ready HyperPod nodes"),
        ("installer", "installer is not Succeeded"),
        ("artifact", "artifact drift"),
        ("tls", "TLS probe did not return ok"),
    ],
)
def test_full_health_report_rechecks_gpu_node_installer_and_tls(
    release: CheckedRelease, fault: str, problem: str | None
) -> None:
    node = release.documents[("gpu-a", "nodes", "")]["items"][0]
    if fault == "empty":
        node["status"]["conditions"] = []
    elif fault == "installer":
        node["metadata"]["annotations"]["gpu-fault.io/installer-state"] = "Failed"
    elif fault == "artifact":
        node["metadata"]["annotations"]["gpu-fault.io/installer-artifact-sha256"] = (
            "other"
        )
    elif fault == "tls":
        release.tls = {"status": "failed"}
    result = finding(
        checks.build_health_report(release, mode="verify"), "gpu_cluster:gpu-a"
    )
    assert result["status"] == ("FAIL" if problem else "PASS")
    if problem:
        assert problem in result["summary"]
    else:
        assert result["details"]["nodes"] == ["node-a"]


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("none", None),
        ("unconfigured", "configuration is missing"),
        ("inactive", "not active"),
        ("listeners", "exactly one TLS"),
        ("certificate", "configured certificate"),
        ("groups", "no target group"),
        ("unhealthy", "only 2 healthy targets"),
    ],
)
def test_full_health_report_requires_nlb_tls_and_all_expected_targets(
    release: CheckedRelease, fault: str, problem: str | None
) -> None:
    if fault == "unconfigured":
        release.config.nlb = {}
    elif fault == "inactive":
        release.aws["elbv2", "describe-load-balancers"]["LoadBalancers"][0]["State"][
            "Code"
        ] = "provisioning"
    elif fault == "listeners":
        release.aws["elbv2", "describe-listeners"]["Listeners"] = []
    elif fault == "certificate":
        release.aws["elbv2", "describe-listeners"]["Listeners"][0]["Certificates"] = []
    elif fault == "groups":
        release.aws["elbv2", "describe-target-groups"]["TargetGroups"] = []
    elif fault == "unhealthy":
        release.aws["elbv2", "describe-target-health"]["TargetHealthDescriptions"].pop()
    result = finding(checks.build_health_report(release, mode="verify"), "nlb_runtime")
    assert result["status"] == ("FAIL" if problem else "PASS")
    if problem:
        assert problem in result["summary"]
    else:
        assert result["details"]["healthy_targets"] == 3


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("none", None),
        ("missing-field", "requires token_file"),
        ("missing-file", "required local input is missing"),
        ("short-token", "token must be"),
        ("whitespace-token", "token must be"),
        ("short-master", "fleet master must be"),
        ("permissions", "must not grant group/other"),
    ],
)
def test_preflight_local_inputs_enforce_permissions_and_format_without_exposing_values(
    release: CheckedRelease, tmp_path: Path, fault: str, problem: str | None
) -> None:
    cpu = tmp_path / "cpu.kubeconfig"
    cpu.write_text("example fake transport only")
    release.config.cpu_kubeconfig = str(cpu)
    token, master, ca = (
        tmp_path / name for name in ("cluster-input", "fleet-input", "ca")
    )
    token.write_text("example-cluster-placeholder-" + "x" * 32)
    master.write_text("example-fleet-placeholder-" + "y" * 32)
    ca.write_text("example certificate handled by fake openssl")
    token.chmod(0o600)
    master.chmod(0o600)
    target = replace(
        release.config.clusters[0],
        token_file=str(token),
        fleet_master_file=str(master),
        ca_file=str(ca),
    )
    if fault == "missing-field":
        target = replace(target, token_file=None)
    elif fault == "missing-file":
        ca.unlink()
    elif fault == "short-token":
        token.write_text("short")
    elif fault == "whitespace-token":
        token.write_text(" " + token.read_text())
    elif fault == "short-master":
        master.write_text("short")
    elif fault == "permissions":
        master.chmod(0o644)
    release.config.clusters = (target,)
    report = checks.build_preflight_report(release)
    result = finding(report, "local_inputs")
    assert result["status"] == ("FAIL" if problem else "PASS")
    if problem:
        assert problem in result["summary"]
    else:
        assert result["details"]["clusters"]["gpu-a"]["token_mode"] == "600"
        assert any(args[0] == "openssl" for args, _kwargs in release.runner.calls), (
            "valid local inputs still require the certificate validity check"
        )
    assert token.read_text() not in repr(report)


def test_missing_administrator_binary_is_a_failed_preflight(
    release: CheckedRelease, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        checks.shutil,
        "which",
        lambda name: None if name == "kubectl" else "/example/tool",
    )
    result = finding(checks.build_preflight_report(release), "tools")
    assert result["status"] == "FAIL"
    assert result["summary"] == "missing administrator tools: kubectl"


def test_disabled_email_requires_an_acknowledged_external_channel(
    release: CheckedRelease,
) -> None:
    assert checks.check_email_notifications(release).details == {"enabled": False}
    release.config.notifications.acknowledge_external_alert_channel = False
    with pytest.raises(ReleaseError, match="no administrator notification channel"):
        checks.check_email_notifications(release)


@pytest.mark.parametrize(
    "permission,violation",
    [
        ({"IpProtocol": "tcp"}, True),
        ({"IpProtocol": "tcp", "FromPort": "bad", "ToPort": 443}, True),
        ({"IpProtocol": "tcp", "FromPort": {}, "ToPort": 443}, True),
        ({"IpProtocol": "tcp", "FromPort": 444, "ToPort": 445}, False),
        (
            {
                "IpProtocol": "tcp",
                "FromPort": 443,
                "ToPort": 443,
                "Ipv6Ranges": [{"CidrIpv6": "::/0"}],
            },
            True,
        ),
    ],
)
def test_ambiguous_https_security_group_rules_fail_closed(
    permission: dict[str, Any], violation: bool
) -> None:
    assert (checks.nlb_https_rule_violation(permission) is not None) is violation
