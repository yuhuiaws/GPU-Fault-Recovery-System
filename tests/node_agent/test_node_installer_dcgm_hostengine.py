"""The DCGM host engine mode on its way from the release to the Node Agent.

An ``eks-managed`` cluster whose node groups declare a host DCGM
(``gpu-fault.io/dcgm-hostengine=host``) is handed ``GPU_FAULT_DCGM_HOSTENGINE``
by the release. ``deploy-node-installer-reconciler.sh`` bakes it into the
installer template and the reconciler's environment, the installer Job hands it
to the node preflight and to ``install-gpu-fault-collector.sh``, which writes it
into the Agent environment, and the reconciler keeps it in the node identity
(``gpu-fault.io/installer-dcgm-hostengine``) so a mode change re-installs the
Agents whose environment still names the other mode. HyperPod, and an
``eks-managed`` render without the mode, stay byte for byte what they were.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from gpu_fault.node_installer_reconciler import (
    INSTALLER_ARTIFACT_ANNOTATION,
    INSTALLER_BUNDLE_ANNOTATION,
    INSTALLER_DIGEST_ANNOTATION,
    INSTALLER_NODE_ANNOTATIONS,
    INSTALLER_NODE_UID_ANNOTATION,
    INSTALLER_STATE_ANNOTATION,
    INSTALLER_TEMPLATE_ANNOTATION,
    INSTALLER_VERSION_ANNOTATION,
    NodeInstallerReconciler,
)
from gpu_fault.node_installer_rendering import INSTALLER_DCGM_HOSTENGINE_ANNOTATION
from tests.deploy.test_reconciler_template_override import (  # noqa: F401 - fixture
    override_environment,
    records,
    run_override,
)
from tests.node_agent._deployment_support import NODE_SCRIPTS, ROOT
from tests.node_agent.test_node_installer_reconciler import (
    ARTIFACT,
    BUNDLE,
    TEMPLATE,
    BatchApi,
    CoreApi,
    apply_patches,
    job,
    node,
    template,
)

JOB_SCRIPT = ROOT / "deploy/node/run-hyperpod-installer-job.sh"
FQDN = "ip-10-0-33-205.us-west-2.compute.internal"
PROVIDER_ID = "aws:///us-west-2c/i-00000000000000001"
#: What a host-mode render adds, and nothing else (run-hyperpod-installer-job.sh).
HOST_SPLICES = (
    '\n            - name: DCGM_HOSTENGINE\n              value: "host"',
    ' GPU_FAULT_DCGM_HOSTENGINE="${DCGM_HOSTENGINE}"',
    ' DCGM_HOSTENGINE="${DCGM_HOSTENGINE}"',
    ' --dcgm-hostengine "${DCGM_HOSTENGINE}"',
)


# --- the installer Job ---------------------------------------------------------


def _render_job(
    tmp_path: Path, *, provider: str | None, mode: str | None
) -> subprocess.CompletedProcess[str]:
    snapshot = tmp_path / "nodes.json"
    snapshot.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cluster_id": "gpu-test",
                "context": "gpu-context",
                "namespace": "gpu-fault-system",
                "node_action_keys_secret": "gpu-fault-node-action-keys",
                "connection_secret": "gpu-fault-regional-connection",
                "references_validated": True,
                "nodes": {
                    FQDN: {
                        "metadata": {
                            "uid": "node-uid-1",
                            "labels": {
                                "node.kubernetes.io/instance-type": "p5en.48xlarge"
                            },
                        },
                        "status": {
                            "addresses": [
                                {"type": "InternalIP", "address": "10.0.33.205"}
                            ]
                        },
                        "spec": {"providerID": PROVIDER_ID},
                    }
                },
            }
        )
    )
    snapshot.chmod(0o600)
    kubectl = tmp_path / "kubectl"
    kubectl.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    kubectl.chmod(0o755)
    environment = {
        "PATH": str(tmp_path) + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "GPU_FAULT_CLUSTER_ID": "gpu-test",
        "GPU_FAULT_KUBECTL_CONTEXT": "gpu-context",
        "GPU_FAULT_CONNECTION_MODE": "regional",
        "GPU_FAULT_INSTALLER_ARTIFACT_SHA256": "a" * 64,
        "GPU_FAULT_INSTALLER_CONFIG_DIGEST": "b" * 64,
        "GPU_FAULT_INSTALLER_TEMPLATE_SHA256": "d" * 64,
        "GPU_FAULT_VERSION": "0.10.0",
    }
    if provider is not None:
        environment["GPU_FAULT_CLUSTER_PROVIDER"] = provider
    if mode is not None:
        environment["GPU_FAULT_DCGM_HOSTENGINE"] = mode
    return subprocess.run(
        [
            "bash",
            str(JOB_SCRIPT),
            "--node",
            FQDN,
            "--node-inputs",
            str(snapshot),
            "--render-only",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )


def _rendered(tmp_path: Path, *, provider: str | None, mode: str | None) -> str:
    result = _render_job(tmp_path, provider=provider, mode=mode)
    assert result.returncode == 0, result.stderr
    return result.stdout


def _container(text: str) -> dict:
    return yaml.safe_load(text)["spec"]["template"]["spec"]["containers"][0]


def test_a_host_mode_job_hands_the_mode_to_the_preflight_and_the_installer(
    tmp_path: Path,
) -> None:
    text = _rendered(tmp_path, provider="eks-managed", mode="host")
    container = _container(text)
    environment = {item["name"]: item.get("value") for item in container["env"]}
    assert environment["DCGM_HOSTENGINE"] == "host", "the Job carries the mode"
    script = container["args"][-1]
    # The heredoc joins each chroot's backslash-continued env list into one line.
    lines = script.splitlines()
    preflight = next(
        line for line in lines if "GPU_FAULT_PREFLIGHT_HOST_ROOT=/" in line
    )
    install = next(
        line
        for line in lines
        if "chroot /host" in line and "INSTALLER_ACTIVE_DEADLINE_SECONDS=" in line
    )
    assert ' GPU_FAULT_DCGM_HOSTENGINE="${DCGM_HOSTENGINE}" ' in preflight, (
        "the node preflight learns the mode it has to check for"
    )
    assert ' DCGM_HOSTENGINE="${DCGM_HOSTENGINE}" ' in install, (
        "the chroot running the installer receives the mode"
    )
    fragment = script[script.index("quiesce_args=()") :]
    fragment = fragment[: fragment.index("fi\n") + 3]
    resolved = subprocess.run(
        ["bash", "-ceu", fragment + '\nprintf "%s\\n" "${quiesce_args[@]}"'],
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "CLUSTER_PROVIDER": "eks-managed",
            "DCGM_HOSTENGINE": "host",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert resolved.returncode == 0, resolved.stderr
    assert resolved.stdout.split() == [
        "--quiesce-sweep-kubernetes-device-holders",
        "--cluster-provider",
        "eks-managed",
        "--dcgm-hostengine",
        "host",
    ], "the installer writes the mode into the eks-managed Agent environment"


def test_a_render_without_the_mode_is_unchanged(tmp_path: Path) -> None:
    host = _rendered(tmp_path, provider="eks-managed", mode="host")
    embedded = _rendered(tmp_path, provider="eks-managed", mode="embedded")
    plain = _rendered(tmp_path, provider="eks-managed", mode=None)
    assert _rendered(tmp_path, provider="eks-managed", mode="") == plain, (
        "an empty mode renders exactly like an absent one"
    )
    assert host != plain and embedded != plain, "a set mode reaches the Job"
    stripped = host
    for splice in HOST_SPLICES:
        assert stripped.count(splice) == 1, f"the host render splices {splice!r}"
        stripped = stripped.replace(splice, "")
    assert stripped == plain, (
        "the mode changes nothing but its four splices: a mode-less render is "
        "byte for byte the template every node was installed from"
    )
    for text in (plain, _rendered(tmp_path, provider=None, mode=None)):
        assert "DCGM_HOSTENGINE" not in text and "dcgm-hostengine" not in text, (
            "HyperPod and a mode-less eks-managed render name no DCGM mode"
        )


@pytest.mark.parametrize(
    ("provider", "mode", "message"),
    [
        (None, "host", "GPU_FAULT_DCGM_HOSTENGINE is an eks-managed setting"),
        ("hyperpod", "embedded", "GPU_FAULT_DCGM_HOSTENGINE is an eks-managed setting"),
        ("eks-managed", "Host", "GPU_FAULT_DCGM_HOSTENGINE must be host or embedded"),
    ],
)
def test_the_job_refuses_a_mode_it_cannot_honour(
    tmp_path: Path, provider: str | None, mode: str, message: str
) -> None:
    result = _render_job(tmp_path, provider=provider, mode=mode)
    assert result.returncode == 2, result.stdout + result.stderr
    assert message in result.stderr, result.stderr


# --- install-gpu-fault-collector.sh ------------------------------------------------


def _excerpt(script: str, pattern: str) -> str:
    matches = list(re.finditer(pattern, script, re.M | re.S))
    assert len(matches) == 1, f"expected one shell fragment for {pattern!r}"
    return matches[0].group()


def _collector_environment(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the installer's own defaults, parser, provider guards and the
    eks-managed environment writes -- never the installer itself."""

    installer = NODE_SCRIPTS[0].read_text(encoding="utf-8")
    program = "\n".join(
        (
            "set -euo pipefail",
            'die() { printf "%s\\n" "$*" >&2; exit 2; }',
            'require_value() { [[ $# -ge 2 && -n "$2" ]] || die "missing value"; }',
            'write_env() { printf "%s=%s\\n" "$1" "$2"; }',
            _excerpt(installer, r'^CLUSTER_PROVIDER="hyperpod"$'),
            _excerpt(installer, r'^DCGM_HOSTENGINE="embedded"$'),
            _excerpt(installer, r"^while \[\[ \$# -gt 0 \]\]; do\n.*?^done$"),
            _excerpt(installer, r'^\[\[ "\$\{CLUSTER_PROVIDER\}" == "hyperpod" .*?$'),
            _excerpt(installer, r'^\[\[ "\$\{DCGM_HOSTENGINE\}" == "embedded" .*?$'),
            _excerpt(
                installer,
                r'^ +if \[\[ "\$\{CLUSTER_PROVIDER\}" == "eks-managed" \]\]; then '
                r"write_env GPU_FAULT_CLUSTER_PROVIDER .*?; fi$",
            ),
            _excerpt(
                installer,
                r'^ +if \[\[ "\$\{CLUSTER_PROVIDER\}" == "eks-managed" \]\]; then '
                r"write_env GPU_FAULT_DCGM_HOSTENGINE .*?; fi$",
            ),
        )
    )
    return subprocess.run(
        ["bash", "-c", program, "installer", *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ((), []),
        (("--cluster-provider", "hyperpod"), []),
        (
            ("--cluster-provider", "eks-managed"),
            [
                "GPU_FAULT_CLUSTER_PROVIDER=eks-managed",
                "GPU_FAULT_DCGM_HOSTENGINE=embedded",
            ],
        ),
        (
            ("--cluster-provider", "eks-managed", "--dcgm-hostengine", "host"),
            [
                "GPU_FAULT_CLUSTER_PROVIDER=eks-managed",
                "GPU_FAULT_DCGM_HOSTENGINE=host",
            ],
        ),
        (
            ("--cluster-provider", "eks-managed", "--dcgm-hostengine", "embedded"),
            [
                "GPU_FAULT_CLUSTER_PROVIDER=eks-managed",
                "GPU_FAULT_DCGM_HOSTENGINE=embedded",
            ],
        ),
    ],
)
def test_the_collector_installer_writes_the_mode_on_eks_managed_only(
    arguments: tuple[str, ...], expected: list[str]
) -> None:
    result = _collector_environment(*arguments)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == expected, (
        "the mode reaches the Agent environment on eks-managed only; a HyperPod "
        "Agent environment file stays byte for byte the same"
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ("--dcgm-hostengine", "host"),
        ("--cluster-provider", "hyperpod", "--dcgm-hostengine", "host"),
        ("--cluster-provider", "eks-managed", "--dcgm-hostengine", "Host"),
        ("--cluster-provider", "eks-managed", "--dcgm-hostengine", "disabled"),
    ],
)
def test_the_collector_installer_refuses_a_mode_it_cannot_honour(
    arguments: tuple[str, ...],
) -> None:
    result = _collector_environment(*arguments)
    assert result.returncode == 2, result.stdout + result.stderr
    assert (
        "--dcgm-hostengine must be embedded, or host with --cluster-provider "
        "eks-managed"
    ) in result.stderr, result.stderr
    assert result.stdout == "", "nothing reaches the Agent environment"


# --- deploy-node-installer-reconciler.sh ----------------------------------------


@pytest.fixture
def deploy_environment(request: pytest.FixtureRequest) -> dict[str, str]:
    """The stubbed reconciler deploy of ``test_reconciler_template_override``."""

    return dict(request.getfixturevalue("override_environment"))


def _reconciler_environment(tmp_path: Path) -> dict[str, str | None]:
    documents = records(tmp_path, "restores.jsonl")
    deployment = next(
        item
        for entry in documents
        for item in entry["documents"]
        if item.get("kind") == "Deployment"
    )
    container = next(
        item
        for item in deployment["spec"]["template"]["spec"]["containers"]
        if item["name"] == "reconciler"
    )
    return {item["name"]: item.get("value") for item in container["env"]}


def test_the_reconciler_identity_carries_the_mode_of_an_eks_cluster(
    deploy_environment: dict[str, str], tmp_path: Path
) -> None:
    for provider, mode, expected in (
        ("eks-managed", "host", "host"),
        ("eks-managed", "embedded", "embedded"),
        ("eks-managed", "", None),
        (None, "", None),
    ):
        (tmp_path / "restores.jsonl").unlink(missing_ok=True)
        environment = {**deploy_environment, "GPU_FAULT_DCGM_HOSTENGINE": mode}
        if provider is not None:
            environment["GPU_FAULT_CLUSTER_PROVIDER"] = provider
        result = run_override(environment)
        assert result.returncode == 0, result.stderr
        assert (
            _reconciler_environment(tmp_path).get("GPU_FAULT_DCGM_HOSTENGINE")
            == expected
        ), (
            f"{provider or 'hyperpod'}/{mode or 'unset'}: the reconciler compares "
            "the mode; a HyperPod or mode-less manifest does not name it"
        )


@pytest.mark.parametrize(
    ("provider", "mode", "message"),
    [
        (None, "host", "GPU_FAULT_DCGM_HOSTENGINE is an eks-managed setting"),
        ("eks-managed", "on", "GPU_FAULT_DCGM_HOSTENGINE must be host or embedded"),
    ],
)
def test_the_reconciler_deploy_refuses_a_mode_before_any_remote_call(
    deploy_environment: dict[str, str],
    tmp_path: Path,
    provider: str | None,
    mode: str,
    message: str,
) -> None:
    environment = {**deploy_environment, "GPU_FAULT_DCGM_HOSTENGINE": mode}
    if provider is not None:
        environment["GPU_FAULT_CLUSTER_PROVIDER"] = provider
    result = run_override(environment)
    assert result.returncode == 2, result.stdout + result.stderr
    assert message in result.stderr, result.stderr
    assert not (tmp_path / "calls.jsonl").exists(), "no cluster was read"


def test_a_fresh_template_bakes_the_mode_into_every_installer_job(
    deploy_environment: dict[str, str], tmp_path: Path
) -> None:
    """Without a pinned template the deploy renders one, mode included."""

    binary = tmp_path / "bin"
    (binary / "kubectl").rename(binary / "kubectl-recorder")
    (binary / "kubectl").write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """
            import os
            import sys
            from pathlib import Path

            import yaml

            args = sys.argv[1:]
            if "create" in args and "configmap" in args:
                name = args[args.index("configmap") + 1]
                source = next(a for a in args if a.startswith("--from-file="))
                key, _, path = source.removeprefix("--from-file=").partition("=")
                print(yaml.safe_dump({
                    "apiVersion": "v1",
                    "kind": "ConfigMap",
                    "metadata": {"name": name, "namespace": "gpu-fault-system"},
                    "data": {key: Path(path).read_text()},
                }))
                raise SystemExit(0)
            recorder = Path(sys.argv[0]).with_name("kubectl-recorder")
            os.execv(sys.executable, [sys.executable, str(recorder), *args])
            """
        ),
        encoding="utf-8",
    )
    (binary / "kubectl").chmod(0o755)
    environment = {
        **deploy_environment,
        "GPU_FAULT_CLUSTER_PROVIDER": "eks-managed",
        "GPU_FAULT_NODE_SELECTOR": "gpu-fault.io/control-plane=site-a",
        "GPU_FAULT_DCGM_HOSTENGINE": "host",
    }
    for name in (
        "GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP",
        "GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256",
    ):
        environment.pop(name)
    result = run_override(environment)
    assert result.returncode == 0, result.stderr
    config_map = next(
        item
        for entry in records(tmp_path, "restores.jsonl")
        for item in entry["documents"]
        if item.get("kind") == "ConfigMap"
    )
    container = _container(config_map["data"]["job.yaml"])
    job_environment = {item["name"]: item.get("value") for item in container["env"]}
    assert job_environment["DCGM_HOSTENGINE"] == "host", (
        "the template every installer and preflight Job of the cluster is "
        "rendered from carries the mode"
    )
    assert job_environment["CLUSTER_PROVIDER"] == "eks-managed", "provider baked"
    assert _reconciler_environment(tmp_path)["GPU_FAULT_DCGM_HOSTENGINE"] == "host"


# --- the reconciler's node identity --------------------------------------------


def _reconciler(
    core: CoreApi, batch: BatchApi, *, mode: str
) -> NodeInstallerReconciler:
    return NodeInstallerReconciler(
        core,
        batch,
        namespace="gpu-fault-system",
        cluster_name="hp-cluster-a",
        version="0.10.0",
        config_digest="config-sha",
        artifact_sha256=ARTIFACT,
        bundle_sha256=BUNDLE,
        template_sha256=TEMPLATE,
        job_template=template(),
        dcgm_metrics_url_template="http://{node_ip}:9400/metrics",
        dcgm_hostengine=mode,
    )


def _installed(**extra: str) -> dict[str, str]:
    return {
        INSTALLER_VERSION_ANNOTATION: "0.10.0",
        INSTALLER_DIGEST_ANNOTATION: "config-sha",
        INSTALLER_ARTIFACT_ANNOTATION: ARTIFACT,
        INSTALLER_BUNDLE_ANNOTATION: BUNDLE,
        INSTALLER_TEMPLATE_ANNOTATION: TEMPLATE,
        INSTALLER_NODE_UID_ANNOTATION: "node-uid",
        INSTALLER_STATE_ANNOTATION: "Succeeded",
        **extra,
    }


def _created_job_name(mode: str, annotations: dict[str, str]) -> str:
    batch = BatchApi()
    outcome = _reconciler(
        CoreApi([node(annotations=annotations)]), batch, mode=mode
    ).reconcile_once()
    assert outcome["created"] == 1, outcome
    return str(batch.created[0][1]["metadata"]["name"])


def test_a_host_mode_reconciler_reinstalls_an_agent_installed_without_the_mode() -> (
    None
):
    target = node(annotations=_installed())
    core = CoreApi([target])
    batch = BatchApi()
    reconciler = _reconciler(core, batch, mode="host")
    assert reconciler.reconcile_once()["created"] == 1, (
        "an Agent installed before host mode was declared still names embedded"
    )
    annotations = apply_patches(target, core)
    assert annotations[INSTALLER_DCGM_HOSTENGINE_ANNOTATION] == "host", (
        "the node is stamped with the mode its new Agent environment names"
    )
    annotations[INSTALLER_STATE_ANNOTATION] = "Succeeded"
    target.metadata.annotations = annotations
    core.patches.clear()
    assert reconciler.reconcile_once()["current"] == 1, "installed in host mode"
    assert not core.patches, "a current node is not patched"


def test_an_embedded_reconciler_reinstalls_an_agent_still_in_host_mode() -> None:
    target = node(
        annotations=_installed(**{INSTALLER_DCGM_HOSTENGINE_ANNOTATION: "host"})
    )
    core = CoreApi([target])
    batch = BatchApi(job(condition="Complete"))
    assert _reconciler(core, batch, mode="embedded").reconcile_once()["running"] == 1, (
        "a host-mode Agent on a cluster back in the embedded mode is replayed"
    )
    patch = core.patches[-1][1]["metadata"]["annotations"]
    assert patch[INSTALLER_DCGM_HOSTENGINE_ANNOTATION] is None, (
        "the stale host mark is removed with the rest of the identity"
    )
    assert apply_patches(target, core).get(INSTALLER_DCGM_HOSTENGINE_ANNOTATION) is None


def test_the_identity_of_hyperpod_and_embedded_nodes_is_unchanged() -> None:
    for mode in ("", "embedded"):
        core = CoreApi([node(annotations=_installed())])
        assert _reconciler(core, BatchApi(), mode=mode).reconcile_once()["current"] == 1
        assert not core.patches, f"mode {mode!r}: an installed node stays current"
    hyperpod = _created_job_name("", {})
    assert _created_job_name("embedded", {}) == hyperpod, (
        "an embedded-mode Job keeps the name a mode-less one always had"
    )
    assert _created_job_name("host", {}) != hyperpod, (
        "a host-mode install is its own Job, never the completed one of the old mode"
    )
    assert INSTALLER_DCGM_HOSTENGINE_ANNOTATION in INSTALLER_NODE_ANNOTATIONS, (
        "cluster removal and the clean redeploy clear the mark with the others"
    )


@pytest.mark.parametrize("mode", ["Host", "disabled", "true"])
def test_the_reconciler_refuses_an_unknown_mode(mode: str) -> None:
    with pytest.raises(ValueError, match="GPU_FAULT_DCGM_HOSTENGINE"):
        _reconciler(CoreApi([]), BatchApi(), mode=mode)


def _stub_node_batch(binary: Path) -> None:
    """``python3 -m gpu_fault_release.regional_node_batch`` answers with
    ``TEST_BATCH_SUMMARY`` and exits ``TEST_BATCH_STATUS``; anything else is
    the real interpreter."""

    (binary / "python3").write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """
            import os
            import sys

            if sys.argv[1:3] != ["-m", "gpu_fault_release.regional_node_batch"]:
                os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
            if os.environ.get("TEST_BATCH_SUMMARY"):
                print(os.environ["TEST_BATCH_SUMMARY"])
            status = int(os.environ.get("TEST_BATCH_STATUS", "0"))
            if status:
                print("ERROR: node preflight batch failed: node-a", file=sys.stderr)
            raise SystemExit(status)
            """
        ),
        encoding="utf-8",
    )
    (binary / "python3").chmod(0o755)


def test_the_preflight_verdict_carries_each_nodes_dcgm_state(
    deploy_environment: dict[str, str], tmp_path: Path
) -> None:
    _stub_node_batch(tmp_path / "bin")
    preflight = {**deploy_environment, "GPU_FAULT_RECONCILER_PREFLIGHT_ONLY": "true"}
    for summary, expected in (
        (
            {
                "node_count": 1,
                "status": "PASSED",
                "dcgm_hostengine": {"node-a": {"mode": "host", "state": "host-active"}},
            },
            '{"node_count":1,"status":"PASSED","template_config_map":'
            '"previous-template","dcgm_hostengine":{"node-a":{"mode":"host",'
            '"state":"host-active"}}}',
        ),
        (
            {"node_count": 1, "status": "PASSED"},
            '{"node_count":1,"status":"PASSED","template_config_map":'
            '"previous-template"}',
        ),
    ):
        result = run_override({**preflight, "TEST_BATCH_SUMMARY": json.dumps(summary)})
        assert result.returncode == 0, result.stderr
        lines = result.stdout.strip().splitlines()
        assert json.loads(lines[-2]) == summary, "the batch's summary is still shown"
        assert lines[-1] == expected, (
            "the release reads the DCGM state of every node off the last line; "
            "without one the line is byte for byte the historical verdict"
        )
    failed = run_override({**preflight, "TEST_BATCH_STATUS": "1"})
    assert failed.returncode == 1, failed.stdout + failed.stderr
    assert '"status":"PASSED"' not in failed.stdout, (
        "a failed batch is never reported as a passed preflight"
    )
    assert "node preflight batch failed: node-a" in failed.stderr, failed.stderr
