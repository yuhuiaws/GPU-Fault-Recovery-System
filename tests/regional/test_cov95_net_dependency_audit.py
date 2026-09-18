"""NET-004 command protocol and boundary verdicts with no cloud or kube calls."""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import audit_net004_dependency_boundary as audit
from tests.regional._cov95_collect_net import no_external_effects  # noqa: F401


@pytest.fixture
def boundary(tmp_path: Path, monkeypatch: Any) -> Any:
    kubeconfig = tmp_path / "fixture-kubeconfig"
    kubeconfig.touch()
    for name in (
        "CPU_KUBECONFIG",
        "GPU_KUBECONFIG",
        "GPU_CONTEXT",
        "GPU_EKS_NAME",
        "AWS_REGION",
        "NAMESPACE",
        "CONTROL_APP",
        "CONTROL_SERVICE",
        "EXECUTOR_APP",
    ):
        monkeypatch.setattr(audit, name, getattr(audit, name))
    host = SimpleNamespace(
        calls=[],
        cpu_exec=0,
        missing_pods=False,
        contexts=[{"context": {"cluster": "unit/gpu-a"}}],
        service={
            "metadata": {
                "annotations": {
                    "service.beta.kubernetes.io/aws-load-balancer-name": "control"
                }
            },
            "status": {"loadBalancer": {"ingress": [{"hostname": "control.invalid"}]}},
        },
        lb={
            "LoadBalancers": [
                {
                    "DNSName": "control.invalid",
                    "LoadBalancerArn": "fixture-lb",
                    "Scheme": "internet-facing",
                    "Type": "network",
                    "State": {"Code": "active"},
                    "SecurityGroups": ["sg-fixture"],
                }
            ]
        },
        listeners={
            "Listeners": [
                {
                    "Port": 443,
                    "Protocol": "TLS",
                    "ListenerArn": "fixture-listener",
                    "SslPolicy": "fixture-policy",
                }
            ]
        },
        attributes={"Attributes": []},
        rules={
            "SecurityGroupRules": [
                {
                    "IsEgress": False,
                    "CidrIpv4": "192.0.2.1/32",
                    "IpProtocol": "tcp",
                    "FromPort": 443,
                    "ToPort": 443,
                },
                {"IsEgress": True, "CidrIpv4": "0.0.0.0/0", "IpProtocol": "-1"},
                {
                    "IsEgress": False,
                    "CidrIpv4": "192.0.2.2/32",
                    "IpProtocol": "udp",
                    "FromPort": 443,
                    "ToPort": 443,
                },
                {
                    "IsEgress": False,
                    "CidrIpv4": "192.0.2.3/32",
                    "IpProtocol": "tcp",
                    "FromPort": 80,
                    "ToPort": 80,
                },
                {
                    "IsEgress": False,
                    "IpProtocol": "tcp",
                    "FromPort": 443,
                    "ToPort": 443,
                },
            ]
        },
        eks={
            "cluster": {
                "name": "gpu-a",
                "status": "ACTIVE",
                "endpoint": "https://gpu.invalid",
                "certificateAuthority": {"data": "example-public-ca"},
                "resourcesVpcConfig": {
                    "vpcId": "vpc-fixture",
                    "endpointPublicAccess": True,
                    "endpointPrivateAccess": True,
                },
            }
        },
        nat={"NatGateways": [{"NatGatewayAddresses": [{"PublicIp": "192.0.2.1"}, {}]}]},
        pod={
            "metadata": {"name": "api-a", "uid": "api-a-uid"},
            "spec": {
                "containers": [
                    {
                        "name": "api",
                        "env": [{"name": "SAFE_NAME"}, {}],
                        "volumeMounts": [{"mountPath": "/safe"}, {}],
                    }
                ],
                "volumes": [{"secret": {"secretName": "safe-name"}}, {}],
                "serviceAccountName": "api",
            },
            "status": {
                "phase": "Running",
                "conditions": [{"type": "Ready", "status": "True"}],
                "containerStatuses": [
                    {"name": "api", "ready": True, "containerID": "unit-api"}
                ],
            },
        },
        runtime={"kubeconfig_env_names": [], "kubeconfig_files_present": []},
        reverse={"reachable": True, "status": 403, "accepted_boundary": True},
        network={
            "control_resolved_ips": ["192.0.2.10"],
            "nlb_resolved_ips": ["192.0.2.10"],
            "resolved_private": [False],
            "tls_version": "TLSv1.3",
            "tls_handshake_seconds": 0.1,
            "executor_timeout_seconds": 15,
            "egress_ip": "192.0.2.1",
        },
        kubeconfig=kubeconfig,
    )

    monkeypatch.setattr(audit, "run_fixture_command", dependency_transport(host))
    args = audit.build_parser().parse_args(
        [
            "--cpu-kubeconfig",
            str(kubeconfig),
            "--gpu-kubeconfig",
            str(kubeconfig),
            "--gpu-context",
            "gpu-a",
            "--region",
            "us-test-1",
        ]
    )
    host.args = args
    audit.configure(args)
    return host


def dependency_transport(host: Any):
    def command(argv: list[str], **kwargs: Any) -> Any:
        host.calls.append((argv, kwargs))
        if argv[0] == "aws":
            value = {
                "describe-load-balancers": host.lb,
                "describe-listeners": host.listeners,
                "describe-listener-attributes": host.attributes,
                "describe-security-group-rules": host.rules,
                "describe-cluster": host.eks,
                "describe-nat-gateways": host.nat,
            }[argv[2]]
            return SimpleNamespace(stdout=json.dumps(value))
        assert argv[0] == "kubectl"
        is_gpu = "--context" in argv
        if "config" in argv:
            value = {"contexts": host.contexts}
        elif "exec" in argv:
            assert (
                "/opt/gpu-fault/executor/bin/python" in argv
                if is_gpu
                else "/opt/gpu-fault/control-plane/bin/python" in argv
            )
            if is_gpu:
                value = host.network
            else:
                host.cpu_exec += 1
                value = host.runtime if host.cpu_exec % 2 else host.reverse
        elif "svc" in argv:
            value = host.service
        elif "deployment" in argv:
            name = argv[argv.index("deployment") + 1]
            value = {
                "metadata": {"uid": name + "-uid", "generation": 1},
                "spec": {"replicas": 1},
            }
        elif "pods" in argv:
            app = argv[argv.index("-l") + 1].removeprefix("app=")
            pod = deepcopy(host.pod)
            pod["metadata"] = {"name": app + "-a", "uid": app + "-a-uid"}
            value = {"items": [] if host.missing_pods else [pod]}
        elif argv[-1].startswith("jsonpath="):
            return SimpleNamespace(
                stdout=""
                if host.missing_pods
                else "executor-a executor-b"
                if is_gpu
                else "api-a"
            )
        else:
            value = deepcopy(host.pod)
            if "pod" in argv:
                name = argv[argv.index("pod") + 1]
                value["metadata"] = {"name": name, "uid": name + "-uid"}
        return SimpleNamespace(stdout=json.dumps(value))

    return command


@pytest.mark.parametrize(
    "internal,attribute", [(False, False), (True, False), (False, True)]
)
def test_audit_gathers_all_boundary_facts_through_checked_fake_transports(
    boundary: Any, internal: bool, attribute: bool
) -> None:
    host = boundary
    if internal:
        host.lb["LoadBalancers"][0]["Scheme"] = "internal"
        host.network["resolved_private"] = [True]
        host.eks["cluster"]["resourcesVpcConfig"]["endpointPublicAccess"] = False
        host.reverse["reachable"] = False
    if attribute:
        host.attributes["Attributes"] = [
            {"Key": "tcp.idle_timeout.seconds", "Value": "360"},
            {"Value": "ignored"},
        ]
    result = audit.audit()
    assert result["verdict"] == "PASS", result
    assert len(result["gpu_network_probes"]) == 2
    assert result["load_balancer"]["inbound_443_cidrs"] == ["192.0.2.1/32"]
    assert result["nlb_idle_timeout_seconds"] == (360 if attribute else 350)
    assert "certificate_authority_data" not in result["gpu_eks"], (
        "certificate data is not exported"
    )
    assert result["cpu_credential_boundary"]["suspect_env_names"] == []
    assert audit.GPU_EKS_NAME == "gpu-a"
    assert all(call[1]["timeout"] <= 120 for call in host.calls), (
        "all external audit reads must retain bounded timeouts"
    )


@pytest.mark.parametrize(
    "problem", ["ingress", "name", "lb-count", "dns", "listener", "pods"]
)
def test_fact_collection_errors_fail_audit_without_synthetic_pass(
    boundary: Any, problem: str
) -> None:
    host = boundary
    if problem == "ingress":
        host.service["status"]["loadBalancer"]["ingress"] = []
    elif problem == "name":
        host.service["metadata"]["annotations"] = {}
    elif problem == "lb-count":
        host.lb["LoadBalancers"] = []
    elif problem == "dns":
        host.lb["LoadBalancers"][0]["DNSName"] = "other.invalid"
    elif problem == "listener":
        host.listeners["Listeners"] = [{"Port": 80, "Protocol": "TCP"}]
    else:
        host.missing_pods = True
    result = audit.audit()
    assert result["verdict"] == "FAIL"
    assert "error" in result


@pytest.mark.parametrize(
    "problem",
    [
        "env",
        "secret",
        "mount",
        "runtime-env",
        "runtime-file",
        "reverse",
        "timeout",
        "tls",
        "dns",
        "allowlist",
        "egress",
    ],
)
def test_actual_boundary_checks_reject_failed_observations(
    boundary: Any, problem: str
) -> None:
    host = boundary
    if problem == "env":
        host.pod["spec"]["containers"][0]["env"].append({"name": "GPU_EKS_CONTEXT"})
    elif problem == "secret":
        host.pod["spec"]["volumes"].append(
            {"secret": {"secretName": "gpu-a-kubeconfig"}}
        )
    elif problem == "mount":
        host.pod["spec"]["containers"][0]["volumeMounts"].append(
            {"mountPath": "/root/.kube"}
        )
    elif problem == "runtime-env":
        host.runtime["kubeconfig_env_names"] = ["KUBECONFIG"]
    elif problem == "runtime-file":
        host.runtime["kubeconfig_files_present"] = ["/root/.kube/config"]
    elif problem == "reverse":
        host.reverse["accepted_boundary"] = False
    elif problem == "timeout":
        host.network["executor_timeout_seconds"] = 500
    elif problem == "tls":
        host.network["tls_version"] = None
    elif problem == "dns":
        host.network["control_resolved_ips"] = ["192.0.2.11"]
    elif problem == "allowlist":
        host.rules["SecurityGroupRules"].append(
            {"IsEgress": False, "CidrIpv4": "0.0.0.0/0", "IpProtocol": "-1"}
        )
    else:
        host.network["egress_ip"] = "192.0.2.99"
    result = audit.audit()
    assert result["verdict"] == "FAIL"
    expected = (
        "cpu_has_no_gpu_kubeconfig"
        if problem in {"env", "secret", "mount", "runtime-env", "runtime-file"}
        else "gpu_eks_reverse_boundary_is_enforced"
        if problem == "reverse"
        else "executor_timeout_is_below_nlb_idle_timeout"
        if problem == "timeout"
        else "tls_handshake_verified"
        if problem == "tls"
        else "all_executors_resolve_the_control_nlb"
        if problem == "dns"
        else "nlb_exposure_matches_deployment_mode"
    )
    assert expected in result["errors"]


@pytest.mark.parametrize(
    "missing",
    [
        "cpu_kubeconfig",
        "gpu_kubeconfig",
        "gpu_context",
        "region",
        "cpu-file",
        "gpu-file",
    ],
)
def test_configure_requires_explicit_existing_target_identity(
    boundary: Any, missing: str
) -> None:
    args = deepcopy(boundary.args)
    if missing.endswith("-file"):
        setattr(
            args,
            "cpu_kubeconfig" if missing == "cpu-file" else "gpu_kubeconfig",
            str(boundary.kubeconfig.parent / "absent"),
        )
    else:
        setattr(args, missing, "")
    with pytest.raises(audit.CaseError):
        audit.configure(args)


@pytest.mark.parametrize("contexts", [[], [{}, {}], [{"context": {}}]])
def test_derived_cluster_name_requires_single_bound_context(
    boundary: Any, contexts: Any
) -> None:
    boundary.contexts = contexts
    with pytest.raises(audit.CaseError, match="cluster"):
        audit.configure(boundary.args)


def test_explicit_cluster_name_and_failed_public_read_protocol(
    boundary: Any, monkeypatch: Any
) -> None:
    args = deepcopy(boundary.args)
    args.gpu_eks_name = "explicit-cluster"
    audit.configure(args)
    assert audit.GPU_EKS_NAME == "explicit-cluster"
    boundary.missing_pods = True
    for control in (True, False):
        with pytest.raises(audit.CaseError, match="no running"):
            audit.first_running_pod("app=fixture", control=control)
    monkeypatch.setattr(
        audit, "run_fixture_command", lambda *a, **k: SimpleNamespace(stdout="[]")
    )
    with pytest.raises(audit.CaseError, match="not an object"):
        audit.aws("ec2", "describe")


def test_unbound_main_writes_explicit_analysis_not_formal_evidence(
    boundary: Any, monkeypatch: Any, tmp_path: Path
) -> None:
    path = tmp_path / "analysis.json"
    monkeypatch.setattr(
        audit, "os", SimpleNamespace(**{**vars(os), "umask": lambda mode: None})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--cpu-kubeconfig",
            str(boundary.kubeconfig),
            "--gpu-kubeconfig",
            str(boundary.kubeconfig),
            "--gpu-context",
            "gpu-a",
            "--region",
            "us-test-1",
            "--output",
            str(path),
        ],
    )
    assert audit.main() == 0
    result = json.loads(path.read_text())
    assert result["validation_scope"] == "deployed-readonly-analysis"
    assert result["formal_sequence_satisfied"] is False
    assert path.stat().st_mode & 0o777 == 0o600
