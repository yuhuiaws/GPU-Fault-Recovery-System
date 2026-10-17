"""What the nodes of a GPU node group carry: one read-only SSM inventory.

The plan reads the node group's current nodes and the apply's verification
reads the nodes after the roll, with the same POSIX ``sh`` script sent as SSM
``AWS-RunShellScript`` to every running instance of the group (found with
``ec2 describe-instances`` by the ``eks:cluster-name``/``eks:nodegroup-name``
tags). The script changes nothing: it reports whether ``/usr/bin/python3.12``
can build a venv and ``strace`` exists, the driver's CUDA major
(``nvidia-smi -q`` "CUDA Version"), every installed DCGM package with its
``version-release``, ``systemctl is-active nvidia-dcgm``, a running
``nv-hostengine`` and whether ``dcgmi discovery -l`` answers. The judgment is
made here, per DCGM mode:

* always: python3.12 venv and strace present;
* host mode: both DCGM 4 packages accepted by the pin (``DcgmPin``), the CUDA
  build matching the driver's CUDA major, ``nvidia-dcgm`` active and
  ``dcgmi discovery -l`` answering; a CUDA build of another major beside it
  is a finding;
* embedded mode: an active host engine is a finding -- it conflicts with the
  exporter's embedded engine.

Across the nodes, different builds of a DCGM package (the 4.7.0-1 / 4.7.0-2
drift of unpinned installs) and different driver CUDA majors are findings.

SSM needs neither Kubernetes access nor a debug Pod and works before any site
exists: the operator's identity needs ``ssm:SendCommand`` and
``ssm:GetCommandInvocation``, the node role ``AmazonSSMManagedInstanceCore``.
Without them the inventory is ``unavailable`` with the reason.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from gpu_fault.admin.bootstrap_common import BootstrapError, CommandRunner
from gpu_fault.admin.deadlines import remaining_timeout
from gpu_fault.admin.eks_nodegroup_user_data import (
    DCGM_CORE_PACKAGE,
    DCGM_CUDA_PACKAGE_PREFIX,
    DCGM_HOSTENGINE_HOST,
    DCGM_SERVICE,
    DcgmPin,
)
from gpu_fault.admin.process_supervisor import ensure_supervision_safe

__all__ = [
    "NodeFacts",
    "fleet_findings",
    "inventory_commands",
    "judge",
    "parse_inventory",
    "read_nodes",
]

SSM_DOCUMENT = "AWS-RunShellScript"
SSM_SEND_ATTEMPTS = 20
SSM_SEND_INTERVAL_SECONDS = 15.0
SSM_INTERVAL_SECONDS = 10.0
SSM_PENDING = frozenset({"Pending", "InProgress", "Delayed", "Cancelling"})
FACT_PREFIX = "gpu-fault-node "
_CUDA_PACKAGE = re.compile(rf"^{re.escape(DCGM_CUDA_PACKAGE_PREFIX)}(?P<major>[0-9]+)$")
MANUAL_CHECK = (
    "kubectl debug node/<node> -it --image=public.ecr.aws/amazonlinux/"
    'amazonlinux:2023 -- chroot /host sh -c "/usr/bin/python3.12 -c '
    "'import venv' && command -v strace\""
)


def inventory_commands(*, wait_for_cloud_init: bool) -> list[str]:
    """The read-only node script; after a roll it first waits for cloud-init,
    so a node whose user data still installs is not reported missing."""

    lines = (
        ["timeout 900 cloud-init status --wait >/dev/null 2>&1 || true"]
        if wait_for_cloud_init
        else []
    )
    return lines + [
        f'say() {{ echo "{FACT_PREFIX}$*"; }}',
        "if /usr/bin/python3.12 -c 'import venv' >/dev/null 2>&1; then "
        "say python3.12-venv ok; else say python3.12-venv missing; fi",
        "if command -v strace >/dev/null 2>&1; then say strace ok; "
        "else say strace missing; fi",
        'major="$(nvidia-smi -q 2>/dev/null | '
        "sed -n 's/^ *CUDA Version *: *\\([0-9][0-9]*\\)\\..*$/\\1/p' | "
        'head -n 1)"',
        'say driver-cuda-major "${major:-absent}"',
        f"rpm -qa --qf '{FACT_PREFIX}package %{{NAME}} %{{VERSION}}-%{{RELEASE}}\\n' "
        "'datacenter-gpu-manager*' 2>/dev/null || true",
        f'say service "$(systemctl is-active {DCGM_SERVICE} 2>/dev/null || true)"',
        "if pgrep -x nv-hostengine >/dev/null 2>&1; then say hostengine running; "
        "else say hostengine absent; fi",
        "if ! command -v dcgmi >/dev/null 2>&1; then say dcgmi-discovery absent; "
        "elif timeout 60 dcgmi discovery -l >/dev/null 2>&1; then "
        "say dcgmi-discovery ok; else say dcgmi-discovery failed; fi",
        "exit 0",
    ]


@dataclass(frozen=True)
class NodeFacts:
    instance_id: str
    ssm_status: str
    python_venv: str = ""
    strace: str = ""
    driver_cuda_major: str = ""
    packages: Mapping[str, str] = field(default_factory=dict)
    service: str = ""
    hostengine: str = ""
    dcgmi: str = ""

    @property
    def cuda_builds(self) -> dict[str, str]:
        """``{CUDA major: version-release}`` of the installed DCGM CUDA builds."""

        builds = {}
        for name, version in self.packages.items():
            match = _CUDA_PACKAGE.fullmatch(name)
            if match:
                builds[match.group("major")] = version
        return builds

    def describe(self) -> dict[str, Any]:
        return {
            "python3.12-venv": self.python_venv or "unknown",
            "strace": self.strace or "unknown",
            "driver_cuda_major": self.driver_cuda_major or "unknown",
            "dcgm_packages": dict(sorted(self.packages.items())),
            "nvidia-dcgm": self.service or "unknown",
            "nv-hostengine": self.hostengine or "unknown",
            "dcgmi_discovery": self.dcgmi or "unknown",
        }


def parse_inventory(instance_id: str, status: str, output: str) -> NodeFacts:
    values: dict[str, str] = {}
    packages: dict[str, str] = {}
    for line in output.splitlines():
        if not line.startswith(FACT_PREFIX):
            continue
        words = line[len(FACT_PREFIX) :].split()
        if not words:
            continue
        if words[0] == "package" and len(words) == 3:
            packages[words[1]] = words[2]
        else:
            values[words[0]] = " ".join(words[1:])
    major = values.get("driver-cuda-major", "")
    return NodeFacts(
        instance_id=instance_id,
        ssm_status=status,
        python_venv=values.get("python3.12-venv", ""),
        strace=values.get("strace", ""),
        driver_cuda_major="" if major == "absent" else major,
        packages=packages,
        service=values.get("service", ""),
        hostengine=values.get("hostengine", ""),
        dcgmi=values.get("dcgmi-discovery", ""),
    )


def _check(ok: bool, observed: str, expected: str) -> dict[str, Any]:
    return {"ok": ok, "observed": observed or "unknown", "expected": expected}


def judge(facts: NodeFacts, *, mode: str, pin: DcgmPin | None) -> dict[str, Any]:
    """The checks one node passes or fails for ``mode``, plus findings."""

    checks = {
        "python3.12-venv": _check(facts.python_venv == "ok", facts.python_venv, "ok"),
        "strace": _check(facts.strace == "ok", facts.strace, "ok"),
    }
    findings: list[str] = []
    builds = facts.cuda_builds
    major = facts.driver_cuda_major
    if mode == DCGM_HOSTENGINE_HOST and pin is not None:
        core = facts.packages.get(DCGM_CORE_PACKAGE, "")
        checks[DCGM_CORE_PACKAGE] = _check(pin.accepts(core), core, pin.value)
        expected = f"{DCGM_CUDA_PACKAGE_PREFIX}{major}" if major else ""
        checks["dcgm-cuda-flavor"] = _check(
            bool(major) and major in builds,
            ",".join(f"{DCGM_CUDA_PACKAGE_PREFIX}{item}" for item in sorted(builds)),
            expected or "the build of the driver's CUDA major (unreadable)",
        )
        checks["dcgm-cuda-version"] = _check(
            pin.accepts(builds.get(major)), builds.get(major, ""), pin.value
        )
        checks[DCGM_SERVICE] = _check(
            facts.service == "active", facts.service, "active"
        )
        checks["dcgmi-discovery"] = _check(facts.dcgmi == "ok", facts.dcgmi, "ok")
        for other in sorted(set(builds) - {major}):
            findings.append(
                f"{facts.instance_id}: {DCGM_CUDA_PACKAGE_PREFIX}{other} "
                f"{builds[other]} is installed beside the driver's CUDA {major or '?'}"
            )
    elif facts.service == "active" or facts.hostengine == "running":
        findings.append(
            f"{facts.instance_id}: a host DCGM engine is active ({DCGM_SERVICE} "
            f"{facts.service or 'unknown'}, nv-hostengine {facts.hostengine or 'unknown'}"
            "); it conflicts with the DCGM exporter's embedded engine"
        )
    return {
        "instance_id": facts.instance_id,
        "ssm_status": facts.ssm_status,
        "facts": facts.describe(),
        "checks": checks,
        "findings": findings,
        "passed": facts.ssm_status == "Success"
        and all(item["ok"] for item in checks.values()),
    }


def fleet_findings(nodes: Sequence[NodeFacts]) -> list[str]:
    """Differences across the nodes: DCGM builds and driver CUDA majors."""

    findings: list[str] = []
    names = sorted({name for item in nodes for name in item.packages})
    for name in names:
        versions = {
            item.instance_id: item.packages.get(name, "absent") for item in nodes
        }
        if len(set(versions.values())) > 1:
            findings.append(
                f"{name} differs across the nodes: "
                + ", ".join(f"{node} {version}" for node, version in versions.items())
            )
    majors = {item.instance_id: item.driver_cuda_major or "unknown" for item in nodes}
    if len(set(majors.values())) > 1:
        findings.append(
            "the driver CUDA major differs across the nodes: "
            + ", ".join(f"{node} {major}" for node, major in majors.items())
        )
    return findings


# --- SSM -----------------------------------------------------------------------


def _instances(
    runner: CommandRunner, *, region: str, cluster: str, nodegroup: str
) -> list[str]:
    reservations = runner.aws_json(
        region,
        "ec2",
        "describe-instances",
        "--filters",
        f"Name=tag:eks:cluster-name,Values={cluster}",
        f"Name=tag:eks:nodegroup-name,Values={nodegroup}",
        "Name=instance-state-name,Values=running",
    ).get("Reservations")
    return sorted(
        str(instance.get("InstanceId"))
        for reservation in reservations or ()
        for instance in (reservation or {}).get("Instances") or ()
        if instance.get("InstanceId")
    )


def _send(
    runner: CommandRunner,
    *,
    region: str,
    nodegroup: str,
    instances: Sequence[str],
    lines: list[str],
    sleep: Callable[[float], None],
) -> str:
    arguments = (
        "ssm",
        "send-command",
        "--document-name",
        SSM_DOCUMENT,
        "--comment",
        # SendCommand accepts at most 100 characters of comment.
        f"gpu-fault prepare-eks-nodegroup read-only inventory {nodegroup}"[:100],
        "--instance-ids",
        *instances,
        "--parameters",
        json.dumps({"commands": lines, "executionTimeout": ["1200"]}),
    )
    for attempt in range(1, SSM_SEND_ATTEMPTS + 1):
        ensure_supervision_safe()
        try:
            response = runner.aws_json(region, *arguments)
            command_id = str((response.get("Command") or {}).get("CommandId") or "")
            if not command_id:
                raise BootstrapError("ssm send-command returned no CommandId")
            return command_id
        except BootstrapError as exc:
            # A node that booted seconds ago is not registered with SSM yet.
            if "InvalidInstanceId" not in str(exc) or attempt == SSM_SEND_ATTEMPTS:
                raise
        sleep(SSM_SEND_INTERVAL_SECONDS)
    raise BootstrapError("ssm send-command was never accepted")


def _invocation(
    runner: CommandRunner,
    *,
    region: str,
    command_id: str,
    instance: str,
    deadline: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> NodeFacts:
    """One instance's result; ``deadline`` (``clock`` time) bounds every node."""

    while True:
        ensure_supervision_safe()
        try:
            invocation = runner.aws_json(
                region,
                "ssm",
                "get-command-invocation",
                "--command-id",
                command_id,
                "--instance-id",
                instance,
            )
            status = str(invocation.get("Status") or "")
        except BootstrapError as exc:
            if "InvocationDoesNotExist" not in str(exc):
                raise
            invocation, status = {}, "Pending"
        if status not in SSM_PENDING:
            return parse_inventory(
                instance, status, str(invocation.get("StandardOutputContent") or "")
            )
        if clock() >= deadline:
            return NodeFacts(instance_id=instance, ssm_status="TimedOutWaiting")
        sleep(SSM_INTERVAL_SECONDS)


def read_nodes(
    runner: CommandRunner,
    *,
    region: str,
    cluster: str,
    nodegroup: str,
    mode: str,
    pin: DcgmPin | None,
    after_roll: bool,
    timeout_seconds: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> dict[str, Any]:
    """The inventory of every running instance, judged for ``mode``/``pin``.

    ``outcome``: ``passed``/``failed`` (some node misses a check),
    ``skipped`` (no running instance) or ``unavailable`` (SSM refused or the
    nodes could not be listed), with ``nodes``, ``findings`` and, when
    unavailable, ``reason`` and ``remedy``.
    """

    lines = inventory_commands(wait_for_cloud_init=after_roll)
    try:
        instances = _instances(
            runner, region=region, cluster=cluster, nodegroup=nodegroup
        )
        if not instances:
            return {"outcome": "skipped", "reason": "no running instance", "nodes": []}
        command_id = _send(
            runner,
            region=region,
            nodegroup=nodegroup,
            instances=instances,
            lines=lines,
            sleep=sleep,
        )
        deadline = clock() + remaining_timeout(timeout_seconds)
        facts = [
            _invocation(
                runner,
                region=region,
                command_id=command_id,
                instance=item,
                deadline=deadline,
                sleep=sleep,
                clock=clock,
            )
            for item in instances
        ]
    except BootstrapError as exc:
        return {
            "outcome": "unavailable",
            "reason": str(exc),
            "remedy": (
                "the operator identity needs ssm:SendCommand and "
                "ssm:GetCommandInvocation, the node role AmazonSSMManagedInstanceCore; "
                f"or check by hand: {MANUAL_CHECK}"
            ),
            "nodes": [],
            "findings": [],
        }
    nodes = [judge(item, mode=mode, pin=pin) for item in facts]
    findings = [line for item in nodes for line in item["findings"]]
    findings += fleet_findings(facts)
    return {
        "outcome": "passed" if all(item["passed"] for item in nodes) else "failed",
        "command_id": command_id,
        "commands": lines,
        "nodes": nodes,
        "findings": findings,
        "driver_cuda_majors": sorted(
            {item.driver_cuda_major for item in facts if item.driver_cuda_major}
        ),
    }
