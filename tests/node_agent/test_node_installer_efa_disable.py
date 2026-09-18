"""EFA opt-out survives parsing and every rendered node-install handoff."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

from gpu_fault.node_installer_rendering import (
    InstallerIdentity,
    InstallerNode,
    load_installer_template,
    preflight_job,
    render_installer_job,
)
from tests.node_agent._deployment_support import NODE_SCRIPTS, ROOT

EFA_ENV = "GPU_FAULT_NODE_ALLOW_EFA_DRIVER_REMEDIATION"
OPERATIONS_ENV = "GPU_FAULT_NODE_ALLOWED_OPERATIONS"
RETENTION_ENV = "GPU_FAULT_NODE_ACTION_RETENTION_SECONDS"
DEPENDENCY_IMAGE = "registry.invalid/node-dependencies@sha256:" + "e" * 64


def _excerpt(script: str, pattern: str) -> str:
    matches = list(re.finditer(pattern, script, re.M | re.S))
    assert len(matches) == 1, f"expected one shell fragment for {pattern!r}"
    return matches[0].group()


def _installer_environment(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Execute only real parser/guard/allowlist fragments, never the installer."""

    installer = NODE_SCRIPTS[0].read_text(encoding="utf-8")
    defaults = "\n".join(
        re.findall(
            r"^(?:ALLOW_[A-Z_]+|ENABLE_NODE_AGENT|MEMORY_FIELD_DIAGNOSTIC_COMMAND|"
            r'NODE_ACTION_RETENTION_SECONDS)="[^"]*"$',
            installer,
            re.M,
        )
    )
    parser = _excerpt(installer, r"^while \[\[ \$# -gt 0 \]\]; do\n.*?^done\n")
    guard = _excerpt(installer, r'^elif \[\[ "\$\{ALLOW_GPU_RESET\}".*?^fi\n')
    operations = _excerpt(
        installer,
        r'^    NODE_ALLOWED_OPERATIONS="COLLECT_HUNG_TRIAGE.*?'
        r'UPDATE_SOFTWARE_FIRMWARE"\n    fi\n',
    )
    exports = "".join(
        _excerpt(
            installer,
            rf"^[ \t]+write_env {name}[ \t]+(?:\\\n[ \t]*)?"
            r'"\$\{[A-Z_]+\}"\n',
        )
        for name in (EFA_ENV, OPERATIONS_ENV, RETENTION_ENV)
    )
    program = (
        "set -euo pipefail\n"
        'die() { printf "%s\\n" "$*" >&2; exit 2; }\n'
        'require_value() { [[ $# -ge 2 && -n "$2" ]] || die "missing value"; }\n'
        'write_env() { printf "%s=%s\\n" "$1" "$2"; }\n'
        + defaults
        + "\n"
        + parser
        + 'if [[ "${ENABLE_NODE_AGENT}" == "true" ]]; then :\n'
        + guard
        + 'printf "node_agent=%s\\n" "${ENABLE_NODE_AGENT}"\n'
        + 'if [[ "${ENABLE_NODE_AGENT}" == "true" ]]; then\n'
        + operations
        + exports
        + "fi\n"
    )
    return subprocess.run(
        ["bash", "-c", program, "efa-parser", *arguments],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )


def _values(result: subprocess.CompletedProcess[str]) -> dict[str, str]:
    assert result.returncode == 0, result.stderr
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ((), "true"),
        (("--allow-efa-driver-remediation",), "true"),
        (("--disable-efa-driver-remediation",), "false"),
        (
            ("--allow-efa-driver-remediation", "--disable-efa-driver-remediation"),
            "false",
        ),
        (
            ("--disable-efa-driver-remediation", "--allow-efa-driver-remediation"),
            "true",
        ),
    ],
)
def test_efa_switch_controls_the_export_and_only_its_own_operation(
    arguments: tuple[str, ...], expected: str
) -> None:
    values = _values(
        _installer_environment(
            "--enable-node-agent",
            "--allow-driver-remediation",
            "--allow-firmware-update",
            *arguments,
        )
    )
    operations = values[OPERATIONS_ENV].split(",")
    assert values[EFA_ENV] == expected, "installer lost the EFA setting"
    assert ("REMEDIATE_EFA_DRIVER" in operations) is (expected == "true"), (
        "the advertised operation must agree with the EFA permission"
    )
    assert {"RESET_GPU", "REMEDIATE_DRIVER", "UPDATE_SOFTWARE_FIRMWARE"} <= set(
        operations
    ), "EFA opt-out removed unrelated operations"
    assert values[RETENTION_ENV] == str(30 * 24 * 60 * 60), (
        "the actual ledger default must match the 30-day help text"
    )


@pytest.mark.parametrize(
    "arguments",
    [(), ("--allow-efa-driver-remediation",), ("--disable-efa-driver-remediation",)],
)
def test_collector_only_install_keeps_the_default_on_switch_out_of_its_guard(
    arguments: tuple[str, ...],
) -> None:
    assert _values(_installer_environment(*arguments)) == {"node_agent": "false"}, (
        "EFA defaults must not enable or require a NodeAgent"
    )


@pytest.mark.parametrize(
    "argument",
    [
        "--allow-gpu-reset",
        "--allow-fabric-reset",
        "--allow-service-quiesce",
        "--allow-fabric-manager-restart",
        "--allow-field-diagnostic",
        "--allow-driver-remediation",
        "--allow-firmware-update",
    ],
)
def test_efa_opt_out_does_not_bypass_other_node_agent_guards(argument: str) -> None:
    result = _installer_environment("--disable-efa-driver-remediation", argument)
    assert result.returncode == 2, "a mutation flag bypassed the NodeAgent guard"
    assert "node mutation options require --enable-node-agent" in result.stderr, (
        "the installer must retain its existing mutation guard"
    )


def _render_job(
    tmp_path: Path, setting: str | None, connection_mode: str, offline: bool = False
) -> tuple[subprocess.CompletedProcess[str], Path]:
    trace = tmp_path / "kubectl-calls"
    node = {
        "metadata": {
            "uid": "fixture-node-uid",
            "labels": {"node.kubernetes.io/instance-type": "ml.p5.48xlarge"},
        },
        "status": {"addresses": [{"type": "InternalIP", "address": "192.0.2.20"}]},
    }
    stub = tmp_path / "kubectl"
    stub.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        'printf "%s\\n" "$*" >>"${KUBECTL_TRACE}"\n'
        'case "$*" in\n'
        "*'get node node-a -o json'*) printf '%s\\n' "
        + shlex.quote(json.dumps(node))
        + " ;;\n"
        "*'get secret gpu-fault-node-action-keys -o json'*) "
        'printf \'%s\\n\' \'{"data":{"node-a":""}}\' ;;\n'
        "*'get secret gpu-fault-regional-connection'*) : ;;\n"
        "*'get service gpu-fault-api-canary '*) printf '%s\\n' '192.0.2.10' ;;\n"
        '*) printf "unexpected fake kubectl call\\n" >&2; exit 99 ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    environment = {
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "HOME": str(tmp_path),
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONPYCACHEPREFIX": "/tmp/gpu-fault-pycache",
        "KUBECTL_TRACE": str(trace),
        "GPU_FAULT_CLUSTER_ID": "fixture-cluster",
        "GPU_FAULT_KUBECTL_CONTEXT": "fixture-context",
        "GPU_FAULT_CONNECTION_MODE": connection_mode,
        "GPU_FAULT_INSTALLER_ARTIFACT_SHA256": "a" * 64,
        "GPU_FAULT_INSTALLER_BUNDLE_SHA256": "b" * 64,
        "GPU_FAULT_INSTALLER_TEMPLATE_SHA256": "c" * 64,
        "GPU_FAULT_INSTALLER_CONFIG_DIGEST": "d" * 64,
        "GPU_FAULT_VERSION": "0.10.0",
    }
    if setting is not None:
        environment["GPU_FAULT_ENABLE_NODE_EFA_DRIVER_REMEDIATION"] = setting
    if offline:
        environment.update(
            GPU_FAULT_NODE_DEPENDENCY_IMAGE=DEPENDENCY_IMAGE,
            GPU_FAULT_NODE_WHEELHOUSE_SHA256="f" * 64,
        )
    result = subprocess.run(
        ["bash", str(NODE_SCRIPTS[4]), "--node", "node-a", "--render-only"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    return result, trace


@pytest.mark.parametrize("connection_mode", ["local", "regional"])
@pytest.mark.parametrize(
    "setting",
    ["TRUE", "False", "1", "0", "yes", "no", " true", "false ", "false\ntrue"],
)
def test_invalid_efa_setting_fails_before_any_node_or_secret_read(
    tmp_path: Path, setting: str, connection_mode: str
) -> None:
    result, trace = _render_job(tmp_path, setting, connection_mode)
    assert result.returncode == 2, "invalid EFA input was accepted"
    assert result.stderr == (
        "ERROR: GPU_FAULT_ENABLE_NODE_EFA_DRIVER_REMEDIATION must be true or false\n"
    ), "validation must identify the setting without echoing its contents"
    assert not trace.exists(), "invalid EFA input reached a Kubernetes read"
    assert not result.stdout, "invalid EFA input produced a Job"


@pytest.mark.parametrize("connection_mode", ["local", "regional"])
@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("setting", [None, "", "true", "false"])
def test_rendered_efa_setting_reaches_the_installer_and_node_bound_templates(
    tmp_path: Path, setting: str | None, connection_mode: str, offline: bool
) -> None:
    result, trace = _render_job(tmp_path, setting, connection_mode, offline)
    assert result.returncode == 0, result.stderr
    assert trace.exists(), "the rendered Job did not use the local node double"
    job = yaml.safe_load(result.stdout)
    container = job["spec"]["template"]["spec"]["containers"][0]
    expected = setting or "true"
    environment = {item["name"]: item.get("value") for item in container["env"]}
    assert environment["ENABLE_EFA_DRIVER_REMEDIATION"] == expected, (
        "the Job lost the EFA input"
    )
    words = shlex.split(container["args"][0], comments=True)
    assert "ENABLE_EFA_DRIVER_REMEDIATION=${ENABLE_EFA_DRIVER_REMEDIATION}" in words, (
        "chroot must forward the EFA setting to the node shell"
    )
    shells = [
        words[index + 4]
        for index, word in enumerate(words)
        if word == "/bin/bash"
        and words[index + 1 : index + 4] == ["-ceu", "-o", "pipefail"]
    ]
    assert len(shells) == 2, "expected separate preflight and install shell payloads"
    block = _excerpt(
        shells[1],
        r'^\s*if \[\[ "\$\{ENABLE_EFA_DRIVER_REMEDIATION\}" == "true" \]\]; then\n'
        r".*?^\s*fi\n",
    )
    flags = subprocess.run(
        [
            "bash",
            "-c",
            "set -euo pipefail\nremediation_args=()\n"
            + block
            + 'printf "%s\\n" "${remediation_args[@]}"\n',
        ],
        env={"PATH": "/usr/bin:/bin", "ENABLE_EFA_DRIVER_REMEDIATION": expected},
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert flags.returncode == 0, flags.stderr
    arguments = flags.stdout.splitlines()
    assert arguments == [
        "--allow-efa-driver-remediation"
        if expected == "true"
        else "--disable-efa-driver-remediation"
    ], "default-on installers require an explicit disable flag for false"
    install_words = shlex.split(shells[1], comments=True)
    install_start = install_words.index("deploy/node/install-gpu-fault-collector.sh")
    assert "${remediation_args[@]}" in install_words[install_start:], (
        "the installer invocation dropped its remediation arguments"
    )
    assert (
        _values(_installer_environment("--enable-node-agent", *arguments))[EFA_ENV]
        == expected
    ), "the Job and installer disagree about the EFA permission"

    if connection_mode == "regional":
        identity = InstallerIdentity(
            namespace="gpu-fault-system",
            config_digest="d" * 64,
            artifact_sha256="a" * 64,
            bundle_sha256="b" * 64,
            template_sha256="c" * 64,
            node_action_keys_secret="gpu-fault-node-action-keys",
            deadline_seconds=840,
            metrics_url_template="http://{node_ip}:9400/metrics",
            template_content_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
            node_dependency_image=DEPENDENCY_IMAGE if offline else "",
            node_wheelhouse_sha256="f" * 64 if offline else "",
        )
        template = load_installer_template(
            result.stdout,
            expected_sha256=identity.template_content_sha256,
            identity=identity,
            origin="EFA fixture",
        )
        node = InstallerNode("node-b", "fixture-uid-b", "192.0.2.21", "ml.p5.48xlarge")
        for rendered in (
            render_installer_job(template, node, identity, "efa-install"),
            preflight_job(template, node, identity, "efa-preflight"),
        ):
            bound = rendered["spec"]["template"]["spec"]["containers"][0]
            values = {item["name"]: item.get("value") for item in bound["env"]}
            assert values["ENABLE_EFA_DRIVER_REMEDIATION"] == expected, (
                "node binding changed the EFA setting"
            )
            assert bound["args"] == container["args"], (
                "node binding changed the trusted shell payload"
            )
