#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


CPU_KUBECONFIG = Path()
GPU_KUBECONFIG = Path()
GPU_CONTEXT = ""
GPU_EKS_NAME = ""
NAMESPACE = "gpu-fault-system"
AWS_REGION = ""
CONTROL_SERVICE = "gpu-fault-api-nlb"
CONTROL_APP = "gpu-fault-api-ha"
EXECUTOR_APP = "gpu-fault-cluster-executor"
CASE_ID = "GF-REGIONAL-NET-004"
EXECUTOR_TIMEOUT_SECONDS = 15
TLS_NLB_IDLE_TIMEOUT_SECONDS = 350


class CaseError(RuntimeError):
    pass


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the NET-004 one-way regional network boundary."
    )
    parser.add_argument(
        "--cpu-kubeconfig",
        default=os.getenv("CPU_KUBECONFIG", ""),
    )
    parser.add_argument(
        "--gpu-kubeconfig",
        default=os.getenv("GPU_KUBECONFIG") or os.getenv("KUBECONFIG", ""),
    )
    parser.add_argument(
        "--gpu-context",
        default=(
            os.getenv("GPU_EKS_CONTEXT") or os.getenv("GPU_FAULT_DATAPLANE_CONTEXT", "")
        ),
    )
    parser.add_argument(
        "--gpu-eks-name",
        default=os.getenv("GPU_EKS_NAME", ""),
        help="optional; derived from the selected kube context when omitted",
    )
    parser.add_argument(
        "--region",
        default=(os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "")),
    )
    parser.add_argument("--namespace", default="gpu-fault-system")
    parser.add_argument("--control-service", default="gpu-fault-api-nlb")
    parser.add_argument("--control-app", default="gpu-fault-api-ha")
    parser.add_argument("--executor-app", default="gpu-fault-cluster-executor")
    parser.add_argument("--output", type=Path)
    return parser


def _derive_gpu_eks_name() -> str:
    value = json.loads(gpu("config", "view", "--minify", "-o", "json"))
    contexts = value.get("contexts", [])
    if len(contexts) != 1:
        raise CaseError("selected GPU context does not resolve to one cluster")
    cluster = str(contexts[0].get("context", {}).get("cluster") or "")
    if not cluster:
        raise CaseError("selected GPU context has no cluster identity")
    return cluster.rsplit("/", 1)[-1]


def configure(arguments: argparse.Namespace) -> None:
    global AWS_REGION
    global CONTROL_APP
    global CONTROL_SERVICE
    global CPU_KUBECONFIG
    global EXECUTOR_APP
    global GPU_CONTEXT
    global GPU_EKS_NAME
    global GPU_KUBECONFIG
    global NAMESPACE

    if not arguments.cpu_kubeconfig:
        raise CaseError("CPU_KUBECONFIG or --cpu-kubeconfig is required")
    if not arguments.gpu_kubeconfig:
        raise CaseError("GPU_KUBECONFIG/KUBECONFIG or --gpu-kubeconfig is required")
    if not arguments.gpu_context:
        raise CaseError("GPU_EKS_CONTEXT or --gpu-context is required")
    if not arguments.region:
        raise CaseError("AWS_REGION or --region is required")
    CPU_KUBECONFIG = Path(arguments.cpu_kubeconfig).expanduser().resolve()
    GPU_KUBECONFIG = Path(arguments.gpu_kubeconfig).expanduser().resolve()
    if not CPU_KUBECONFIG.is_file():
        raise CaseError(f"CPU kubeconfig does not exist: {CPU_KUBECONFIG}")
    if not GPU_KUBECONFIG.is_file():
        raise CaseError(f"GPU kubeconfig does not exist: {GPU_KUBECONFIG}")
    GPU_CONTEXT = arguments.gpu_context
    AWS_REGION = arguments.region
    NAMESPACE = arguments.namespace
    CONTROL_SERVICE = arguments.control_service
    CONTROL_APP = arguments.control_app
    EXECUTOR_APP = arguments.executor_app
    GPU_EKS_NAME = arguments.gpu_eks_name or _derive_gpu_eks_name()


def run(
    argv: list[str],
    *,
    stdin: str | None = None,
    timeout: int = 120,
) -> str:
    result = subprocess.run(
        argv,
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise CaseError(
            f"command failed ({result.returncode}): {' '.join(argv)}; "
            f"stderr={result.stderr.strip()}"
        )
    return result.stdout


def cpu(*args: str, stdin: str | None = None, timeout: int = 120) -> str:
    return run(
        [
            "kubectl",
            "--kubeconfig",
            str(CPU_KUBECONFIG),
            "-n",
            NAMESPACE,
            *args,
        ],
        stdin=stdin,
        timeout=timeout,
    )


def gpu(*args: str, stdin: str | None = None, timeout: int = 120) -> str:
    return run(
        [
            "kubectl",
            "--kubeconfig",
            str(GPU_KUBECONFIG),
            "--context",
            GPU_CONTEXT,
            "-n",
            NAMESPACE,
            *args,
        ],
        stdin=stdin,
        timeout=timeout,
    )


def aws(service: str, *args: str) -> dict:
    output = run(
        [
            "aws",
            service,
            *args,
            "--region",
            AWS_REGION,
            "--output",
            "json",
        ],
        timeout=120,
    )
    value = json.loads(output)
    if not isinstance(value, dict):
        raise CaseError(f"AWS {service} response is not an object")
    return value


def first_running_pod(label: str, *, control: bool) -> str:
    client = cpu if control else gpu
    value = client(
        "get",
        "pod",
        "-l",
        label,
        "--field-selector=status.phase=Running",
        "-o",
        "jsonpath={.items[0].metadata.name}",
    ).strip()
    if not value:
        raise CaseError(f"no running Pod for {label}")
    return value


def load_balancer_facts() -> dict:
    service = json.loads(cpu("get", "svc", CONTROL_SERVICE, "-o", "json"))
    ingress = service.get("status", {}).get("loadBalancer", {}).get("ingress", [])
    if len(ingress) != 1 or not ingress[0].get("hostname"):
        raise CaseError("control-plane Service has no single NLB hostname")
    nlb_dns = str(ingress[0]["hostname"])
    annotations = service.get("metadata", {}).get("annotations", {})
    lb_name = annotations.get("service.beta.kubernetes.io/aws-load-balancer-name")
    if not lb_name:
        raise CaseError("control-plane Service has no NLB name annotation")

    load_balancers = aws("elbv2", "describe-load-balancers", "--names", lb_name).get(
        "LoadBalancers",
        [],
    )
    if len(load_balancers) != 1:
        raise CaseError("expected one control-plane load balancer")
    lb = load_balancers[0]
    if lb.get("DNSName") != nlb_dns:
        raise CaseError("Service and AWS NLB DNS names differ")

    listeners = aws(
        "elbv2",
        "describe-listeners",
        "--load-balancer-arn",
        str(lb["LoadBalancerArn"]),
    ).get("Listeners", [])
    tls_listeners = [
        listener
        for listener in listeners
        if listener.get("Port") == 443 and listener.get("Protocol") == "TLS"
    ]
    if len(tls_listeners) != 1:
        raise CaseError("expected one TLS/443 NLB listener")
    listener = tls_listeners[0]
    attributes = aws(
        "elbv2",
        "describe-listener-attributes",
        "--listener-arn",
        str(listener["ListenerArn"]),
    ).get("Attributes", [])
    attribute_map = {
        str(item.get("Key")): str(item.get("Value"))
        for item in attributes
        if item.get("Key")
    }
    idle_raw = attribute_map.get("tcp.idle_timeout.seconds")
    if idle_raw is not None:
        idle_timeout = int(idle_raw)
        idle_source = "listener-attribute"
    else:
        idle_timeout = TLS_NLB_IDLE_TIMEOUT_SECONDS
        idle_source = "aws-tls-listener-fixed"

    security_group_ids = [str(value) for value in lb.get("SecurityGroups", [])]
    rules = aws(
        "ec2",
        "describe-security-group-rules",
        "--filters",
        "Name=group-id,Values=" + ",".join(security_group_ids),
    ).get("SecurityGroupRules", [])
    inbound_443_cidrs = sorted(
        {
            str(rule["CidrIpv4"])
            for rule in rules
            if not rule.get("IsEgress")
            and rule.get("CidrIpv4")
            and str(rule.get("IpProtocol")) in {"tcp", "-1"}
            and (
                str(rule.get("IpProtocol")) == "-1"
                or (
                    int(rule.get("FromPort", -1)) <= 443
                    and int(rule.get("ToPort", -1)) >= 443
                )
            )
        }
    )
    return {
        "dns_name": nlb_dns,
        "scheme": lb.get("Scheme"),
        "type": lb.get("Type"),
        "state": lb.get("State", {}).get("Code"),
        "security_group_ids": security_group_ids,
        "inbound_443_cidrs": inbound_443_cidrs,
        "listener_protocol": listener.get("Protocol"),
        "listener_port": listener.get("Port"),
        "listener_ssl_policy": listener.get("SslPolicy"),
        "listener_attributes": attribute_map,
        "idle_timeout_seconds": idle_timeout,
        "idle_timeout_source": idle_source,
    }


def eks_and_nat_facts() -> tuple[dict, list[str]]:
    cluster = aws("eks", "describe-cluster", "--name", GPU_EKS_NAME)["cluster"]
    vpc = cluster["resourcesVpcConfig"]
    nat_gateways = aws(
        "ec2",
        "describe-nat-gateways",
        "--filter",
        f"Name=vpc-id,Values={vpc['vpcId']}",
        "Name=state,Values=available",
    ).get("NatGateways", [])
    nat_public_ips = sorted(
        {
            str(address["PublicIp"])
            for gateway in nat_gateways
            for address in gateway.get("NatGatewayAddresses", [])
            if address.get("PublicIp")
        }
    )
    return (
        {
            "name": cluster.get("name"),
            "status": cluster.get("status"),
            "endpoint": cluster.get("endpoint"),
            "certificate_authority_data": cluster["certificateAuthority"]["data"],
            "vpc_id": vpc.get("vpcId"),
            "endpoint_public_access": vpc.get("endpointPublicAccess"),
            "endpoint_private_access": vpc.get("endpointPrivateAccess"),
            "public_access_cidrs": vpc.get("publicAccessCidrs", []),
        },
        nat_public_ips,
    )


def gpu_network_probes(nlb_dns: str, *, check_egress: bool) -> list[dict]:
    pods = gpu(
        "get",
        "pod",
        "-l",
        f"app={EXECUTOR_APP}",
        "--field-selector=status.phase=Running",
        "-o",
        "jsonpath={.items[*].metadata.name}",
    ).split()
    if not pods:
        raise CaseError("no running cluster executor Pods")
    script = f"""
import ipaddress
import json
import os
import socket
import ssl
import time
import urllib.request
from urllib.parse import urlsplit
from gpu_fault.cluster_executor import _regional_client_from_environment

nlb_dns = {nlb_dns!r}
check_egress = {check_egress!r}
url = os.environ["GPU_FAULT_CONTROL_PLANE_URL"]
host = urlsplit(url).hostname
if not host:
    raise RuntimeError("control-plane URL has no hostname")
control_ips = sorted({{
    item[4][0]
    for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
}})
nlb_ips = sorted({{
    item[4][0]
    for item in socket.getaddrinfo(nlb_dns, 443, type=socket.SOCK_STREAM)
}})
context = ssl.create_default_context(
    cafile=os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"]
)
started = time.monotonic()
with socket.create_connection((host, 443), timeout=15) as raw:
    with context.wrap_socket(raw, server_hostname=host) as tls:
        tls_version = tls.version()
handshake_seconds = time.monotonic() - started
egress_ip = None
if check_egress:
    with urllib.request.urlopen(
        "https://checkip.amazonaws.com",
        timeout=15,
    ) as response:
        egress_ip = response.read().decode().strip()
client_timeout = _regional_client_from_environment().timeout_seconds
print(json.dumps({{
    "pod": os.environ.get("HOSTNAME"),
    "control_host": host,
    "control_resolved_ips": control_ips,
    "nlb_resolved_ips": nlb_ips,
    "resolved_private": [
        ipaddress.ip_address(value).is_private
        for value in control_ips
    ],
    "tls_version": tls_version,
    "tls_handshake_seconds": round(handshake_seconds, 4),
    "egress_ip": egress_ip,
    "executor_timeout_seconds": client_timeout,
}}, sort_keys=True))
"""
    probes = []
    for pod in pods:
        output = gpu("exec", "-i", pod, "--", "python", "-", stdin=script)
        probes.append(json.loads(output.splitlines()[-1]))
    return probes


def cpu_credential_boundary(eks: dict) -> dict:
    pod = first_running_pod(f"app={CONTROL_APP}", control=True)
    pod_spec = json.loads(cpu("get", "pod", pod, "-o", "json"))
    containers = pod_spec["spec"]["containers"]
    env_names = sorted(
        {
            str(item.get("name"))
            for container in containers
            for item in container.get("env", [])
            if item.get("name")
        }
    )
    secret_names = sorted(
        {
            str(volume["secret"]["secretName"])
            for volume in pod_spec["spec"].get("volumes", [])
            if volume.get("secret", {}).get("secretName")
        }
    )
    mount_paths = sorted(
        {
            str(mount["mountPath"])
            for container in containers
            for mount in container.get("volumeMounts", [])
            if mount.get("mountPath")
        }
    )
    runtime_script = """
import json
import os
from pathlib import Path

paths = [
    Path.home() / ".kube" / "config",
    Path("/root/.kube/config"),
    Path("/home/app/.kube/config"),
]
print(json.dumps({
    "kubeconfig_env_names": sorted(
        name for name in os.environ if "KUBECONFIG" in name.upper()
    ),
    "kubeconfig_files_present": [
        str(path) for path in paths if path.exists()
    ],
}, sort_keys=True))
"""
    runtime = json.loads(
        cpu("exec", "-i", pod, "--", "python3", "-", stdin=runtime_script).splitlines()[
            -1
        ]
    )

    unauthenticated_script = f"""
import base64
import json
import ssl
import urllib.error
import urllib.request

endpoint = {str(eks["endpoint"])!r}
ca_data = {str(eks["certificate_authority_data"])!r}
pem = base64.b64decode(ca_data).decode()
context = ssl.create_default_context(cadata=pem)
request = urllib.request.Request(endpoint.rstrip("/") + "/api")
reachable = True
try:
    with urllib.request.urlopen(request, context=context, timeout=15) as response:
        status = response.status
except urllib.error.HTTPError as exc:
    status = exc.code
except (urllib.error.URLError, OSError, TimeoutError):
    reachable = False
    status = None
print(json.dumps({{
    "status": status,
    "reachable": reachable,
    "accepted_boundary": status in {{401, 403}},
}}))
"""
    unauthenticated = json.loads(
        cpu(
            "exec",
            "-i",
            pod,
            "--",
            "python3",
            "-",
            stdin=unauthenticated_script,
        ).splitlines()[-1]
    )
    suspect_env_names = [
        name
        for name in env_names
        if "KUBECONFIG" in name.upper() or "GPU_EKS" in name.upper()
    ]
    suspect_secret_names = [
        name
        for name in secret_names
        if "kubeconfig" in name.lower() or GPU_EKS_NAME.lower() in name.lower()
    ]
    suspect_mount_paths = [
        path
        for path in mount_paths
        if ".kube" in path.lower() or "kubeconfig" in path.lower()
    ]
    return {
        "pod": pod,
        "service_account": pod_spec["spec"].get("serviceAccountName"),
        "suspect_env_names": suspect_env_names,
        "suspect_secret_names": suspect_secret_names,
        "suspect_mount_paths": suspect_mount_paths,
        "runtime": runtime,
        "gpu_eks_unauthenticated_request": unauthenticated,
    }


def audit() -> dict:
    result: dict = {
        "case_id": CASE_ID,
        "verdict": "FAIL",
    }
    started = time.monotonic()
    try:
        load_balancer = load_balancer_facts()
        eks, nat_public_ips = eks_and_nat_facts()
        public_nlb = load_balancer["scheme"] == "internet-facing"
        internal_nlb = load_balancer["scheme"] == "internal"
        probes = gpu_network_probes(
            str(load_balancer["dns_name"]),
            check_egress=public_nlb,
        )
        credential_boundary = cpu_credential_boundary(eks)
        safe_eks = dict(eks)
        safe_eks.pop("certificate_authority_data", None)

        allowed_nat_cidrs = sorted(f"{value}/32" for value in nat_public_ips)
        same_nlb = all(
            probe["control_resolved_ips"] == probe["nlb_resolved_ips"]
            for probe in probes
        )
        internal_resolution = all(
            probe["resolved_private"] and all(probe["resolved_private"])
            for probe in probes
        )
        public_resolution = all(
            probe["resolved_private"] and not any(probe["resolved_private"])
            for probe in probes
        )
        allowed_nat_set = set(allowed_nat_cidrs)
        configured_cidrs = set(load_balancer["inbound_443_cidrs"])
        exact_public_allowlist = bool(configured_cidrs) and configured_cidrs <= (
            allowed_nat_set
        )
        observed_public_egress = all(
            probe["egress_ip"] in nat_public_ips
            and f"{probe['egress_ip']}/32" in load_balancer["inbound_443_cidrs"]
            for probe in probes
        )
        eks_request = credential_boundary["gpu_eks_unauthenticated_request"]
        checks = {
            "nlb_is_active_tls": (
                (public_nlb or internal_nlb)
                and load_balancer["type"] == "network"
                and load_balancer["state"] == "active"
                and load_balancer["listener_protocol"] == "TLS"
                and load_balancer["listener_port"] == 443
            ),
            "all_executors_resolve_the_control_nlb": same_nlb,
            "nlb_exposure_matches_deployment_mode": (
                internal_nlb and internal_resolution
            )
            or (
                public_nlb
                and public_resolution
                and exact_public_allowlist
                and observed_public_egress
            ),
            "tls_handshake_verified": all(
                str(probe["tls_version"]).startswith("TLSv1.")
                and float(probe["tls_handshake_seconds"]) < 15
                for probe in probes
            ),
            "executor_timeout_is_below_nlb_idle_timeout": all(
                float(probe["executor_timeout_seconds"]) == EXECUTOR_TIMEOUT_SECONDS
                and float(probe["executor_timeout_seconds"])
                < float(load_balancer["idle_timeout_seconds"])
                for probe in probes
            ),
            "cpu_has_no_gpu_kubeconfig": (
                not credential_boundary["suspect_env_names"]
                and not credential_boundary["suspect_secret_names"]
                and not credential_boundary["suspect_mount_paths"]
                and not credential_boundary["runtime"]["kubeconfig_env_names"]
                and not credential_boundary["runtime"]["kubeconfig_files_present"]
            ),
            "gpu_eks_reverse_boundary_is_enforced": (
                eks["endpoint_public_access"] is True
                and eks_request["reachable"] is True
                and eks_request["accepted_boundary"] is True
            )
            or (
                eks["endpoint_public_access"] is False
                and eks["endpoint_private_access"] is True
                and eks_request["reachable"] is False
            ),
        }
        errors = [name for name, passed in checks.items() if not passed]
        result = {
            "case_id": CASE_ID,
            "verdict": "PASS" if not errors else "FAIL",
            "errors": errors,
            "checks": checks,
            "load_balancer": load_balancer,
            "gpu_eks": safe_eks,
            "nat_public_ips": nat_public_ips,
            "gpu_network_probes": probes,
            "cpu_credential_boundary": credential_boundary,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    os.umask(0o077)
    try:
        configure(args)
        result = audit()
    except Exception as exc:
        result = {
            "case_id": CASE_ID,
            "verdict": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }
    if args.output is not None:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
