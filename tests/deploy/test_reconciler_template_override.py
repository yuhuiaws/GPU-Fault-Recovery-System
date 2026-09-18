from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped,unused-ignore]

from tests.deploy.test_installer_template_identity import (
    DEPENDENCY_IMAGE,
    INVENTORY_SHA,
    template_job,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deploy/node/deploy-node-installer-reconciler.sh"


@pytest.fixture
def override_environment(tmp_path: Path) -> dict[str, str]:
    binary = tmp_path / "bin"
    binary.mkdir()
    kubectl = binary / "kubectl"
    kubectl.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """
            import base64
            import json
            import os
            import sys
            from pathlib import Path
            import yaml

            root = Path(os.environ["TEST_TEMPLATE_ROOT"])
            args = sys.argv[1:]
            with (root / "calls.jsonl").open("a") as handle:
                handle.write(json.dumps(args) + "\\n")
            if "get" in args:
                index = args.index("get")
                kind = args[index + 1]
                name = args[index + 2] if len(args) > index + 2 else ""
                if kind == "configmap" and name == "previous-template":
                    print(json.dumps({"data": {"job.yaml": (root / "selected.yaml").read_bytes().decode()}}))
                elif kind == "configmap" and name == "old-bundle":
                    print(json.dumps({"binaryData": {"gpu-fault-node-installer-0.10.0.tar.gz": ""}}))
                elif kind == "configmap" and name == "old-wheel":
                    print(json.dumps({"binaryData": {"executor.whl": ""}}))
                elif kind == "secret":
                    if name == "gpu-fault-node-action-keys":
                        data = {"node-a": "fixture-key-material-not-used-by-any-service"}
                    elif name == "gpu-fault-regional-connection":
                        data = {
                            "cluster-id": "test-cluster",
                            "cluster-token": "fixture-only",
                            "ca.crt": "fixture-only",
                            "control-plane-url": "https://control.example",
                        }
                    else:
                        raise SystemExit("unexpected Secret reference")
                    print(json.dumps({"data": {
                        key: base64.b64encode(value.encode()).decode()
                        for key, value in data.items()
                    }}))
                elif kind == "nodes":
                    print(json.dumps({"items": [{
                        "metadata": {"name": "node-a", "uid": "uid-a", "labels": {
                            "node.kubernetes.io/instance-type": "ml.p5.48xlarge"
                        }},
                        "status": {"addresses": [{"type": "InternalIP", "address": "10.0.0.1"}]},
                    }]}))
                elif kind == "jobs":
                    print('{"items":[]}')
                else:
                    raise SystemExit("unexpected read")
            elif "apply" in args:
                text = Path(args[args.index("-f") + 1]).read_bytes().decode()
                documents = list(yaml.safe_load_all(text))
                dry_run = "--dry-run=server" in args
                name = "admissions.jsonl" if dry_run else "restores.jsonl"
                with (root / name).open("a") as handle:
                    handle.write(json.dumps({"text": text, "documents": documents}) + "\\n")
                if os.environ.get("TEST_DENY_SELECTED") and any(
                    item.get("kind") == "Job" for item in documents
                ):
                    raise SystemExit("selected template denied by admission")
            else:
                raise SystemExit("unexpected mutation")
            """
        ),
        encoding="utf-8",
    )
    kubectl.chmod(0o755)
    python = binary / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """
            import contextlib
            import io
            import json
            import os
            import sys
            from pathlib import Path

            if sys.argv[1:3] != ["-m", "gpu_fault_release.regional_node_batch"]:
                os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
            arguments = sys.argv[3:]
            template = Path(arguments[arguments.index("--template") + 1]).read_bytes().decode()
            # Host execution stays mocked; preparation uses the real batch CLI.
            from gpu_fault_release.regional_node_batch import main
            sys.argv = ["regional_node_batch", *(
                argument for argument in arguments if argument != "--run-preflights"
            )]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main()
            result = json.loads(output.getvalue())
            directory = Path(arguments[arguments.index("--output") + 1])
            jobs = {
                kind: [json.loads(path.read_text()) for path in sorted(directory.glob(kind + "-*.json"))]
                for kind in ("install", "preflight")
            }
            Path(os.environ["TEST_TEMPLATE_ROOT"], "batch.json").write_text(
                json.dumps({"arguments": arguments, "template": template, "jobs": jobs})
            )
            print(json.dumps({"status": "PASSED", "node_count": result["node_count"]}))
            """
        ),
        encoding="utf-8",
    )
    python.chmod(0o755)
    text = yaml.safe_dump(template_job()).replace("\n", "\r\n")
    (tmp_path / "selected.yaml").write_bytes(text.encode())
    return {
        "HOME": str(tmp_path),
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
        "PYTHONDONTWRITEBYTECODE": "1",
        "AWS_CONFIG_FILE": "/dev/null",
        "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
        "AWS_EC2_METADATA_DISABLED": "true",
        "KUBECONFIG": "/dev/null",
        "TEST_TEMPLATE_ROOT": str(tmp_path),
        "GPU_FAULT_KUBECTL_CONTEXT": "test-context",
        "GPU_FAULT_CLUSTER_ID": "test-cluster",
        "GPU_FAULT_INSTALLER_CONFIG_DIGEST": "old-config",
        "GPU_FAULT_INSTALLER_ARTIFACT_SHA256": "1" * 64,
        "GPU_FAULT_INSTALLER_BUNDLE_SHA256": "2" * 64,
        "GPU_FAULT_INSTALLER_TEMPLATE_SHA256": "3" * 64,
        "GPU_FAULT_NODE_COMPATIBILITY_DIGEST": "4" * 64,
        "GPU_FAULT_WHEEL_CONFIG_MAP": "old-wheel",
        "GPU_FAULT_EXECUTOR_WHEEL_FILENAME": "executor.whl",
        "GPU_FAULT_INSTALLER_CONFIG_MAP": "old-bundle",
        "GPU_FAULT_NODE_INSTALLER_IMAGE": "registry.example/installer@sha256:"
        + "c" * 64,
        "GPU_FAULT_NODE_DEPENDENCY_IMAGE": DEPENDENCY_IMAGE,
        "GPU_FAULT_NODE_WHEELHOUSE_SHA256": INVENTORY_SHA,
        "GPU_FAULT_INSTALLER_TEMPLATE_CONFIG_MAP": "previous-template",
        "GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256": hashlib.sha256(
            text.encode()
        ).hexdigest(),
        "GPU_FAULT_SYNC_INSTALLED_RESOURCE_REGISTRY": "false",
        "GPU_FAULT_WAIT_FOR_RECONCILER_ROLLOUT": "false",
    }


def run_override(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=environment,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )


def records(root: Path, name: str) -> list[dict[str, Any]]:
    path = root / name
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def test_preflight_dry_runs_and_batches_the_exact_selected_template(
    override_environment: dict[str, str], tmp_path: Path
) -> None:
    result = run_override(
        {
            **override_environment,
            "GPU_FAULT_RECONCILER_PREFLIGHT_ONLY": "true",
            "GPU_FAULT_REQUIRE_ROLLBACK_SLOT": "true",
        }
    )
    assert result.returncode == 0, result.stderr
    expected = (tmp_path / "selected.yaml").read_bytes().decode()
    admissions = records(tmp_path, "admissions.jsonl")
    assert admissions[0]["text"] == expected, "admission checked a different Job"
    batch = json.loads((tmp_path / "batch.json").read_text())
    assert batch["template"] == expected, "host preflight received a fresh template"
    for mode in ("install", "preflight"):
        jobs = batch["jobs"][mode]
        assert len(jobs) == 1, f"{mode} did not cover the selected node"
        pod = jobs[0]["spec"]["template"]["spec"]
        assert pod["nodeName"] == "node-a", f"{mode} selected a different node"
        assert pod["containers"][0]["args"] == ["previous-template-program"], (
            f"{mode} substituted the candidate program"
        )
        assert pod["initContainers"][0]["image"] == DEPENDENCY_IMAGE, (
            f"{mode} substituted a different dependency image"
        )
        job_environment = {
            item["name"]: item["value"] for item in pod["containers"][0]["env"]
        }
        assert job_environment["NODE_WHEELHOUSE_SHA256"] == INVENTORY_SHA, (
            f"{mode} substituted a different dependency inventory"
        )
        if mode == "preflight":
            assert job_environment["REQUIRE_ROLLBACK_SLOT"] == "true", (
                "the selected old template bypassed the required rollback-slot check"
            )
            assert job_environment["PREFLIGHT_ONLY"] == "true", (
                "the selected old template did not enter read-only preflight"
            )
            root = next(
                mount
                for mount in pod["containers"][0]["volumeMounts"]
                if mount["name"] == "host-root"
            )
            assert root["readOnly"] is True, "preflight retained a writable host root"
    for flag, environment in (
        ("--template-content-sha256", "GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256"),
        ("--node-dependency-image", "GPU_FAULT_NODE_DEPENDENCY_IMAGE"),
        ("--node-wheelhouse-sha256", "GPU_FAULT_NODE_WHEELHOUSE_SHA256"),
    ):
        arguments = batch["arguments"]
        assert arguments[arguments.index(flag) + 1] == override_environment[environment]
    assert not (tmp_path / "restores.jsonl").exists(), (
        "read-only preflight applied a Reconciler restore"
    )


def test_selected_template_admission_failure_stops_before_preflight(
    override_environment: dict[str, str], tmp_path: Path
) -> None:
    result = run_override(
        {
            **override_environment,
            "GPU_FAULT_RECONCILER_PREFLIGHT_ONLY": "true",
            "TEST_DENY_SELECTED": "true",
        }
    )
    assert result.returncode != 0, "the selected template bypassed admission"
    assert "selected template denied by admission" in result.stderr
    assert not (tmp_path / "batch.json").exists(), (
        "preflight started after admission failed"
    )
    assert not (tmp_path / "restores.jsonl").exists(), (
        "a rejected template reached Reconciler restoration"
    )


@pytest.mark.parametrize("preflight", ["true", "false"])
def test_template_drift_cannot_be_installed_as_a_new_trusted_pin(
    override_environment: dict[str, str], tmp_path: Path, preflight: str
) -> None:
    path = tmp_path / "selected.yaml"
    path.write_bytes(
        path.read_bytes().replace(b"previous-template-program", b"tampered-program")
    )
    result = run_override(
        {
            **override_environment,
            "GPU_FAULT_RECONCILER_PREFLIGHT_ONLY": preflight,
            "GPU_FAULT_FLEET_MASTER_FILE": "/must-not-be-read",
        }
    )
    assert result.returncode != 0, "current mutable bytes were blessed as a new pin"
    assert "does not match" in result.stderr
    assert not (tmp_path / "admissions.jsonl").exists(), (
        "an untrusted template reached admission"
    )
    assert not (tmp_path / "restores.jsonl").exists(), (
        "template drift reached Reconciler restoration"
    )
    assert not (tmp_path / "batch.json").exists(), (
        "template drift reached node Job preparation"
    )


def test_override_without_a_trusted_pin_fails_before_remote_reads(
    override_environment: dict[str, str], tmp_path: Path
) -> None:
    override_environment.pop("GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256")
    result = run_override(override_environment)
    assert result.returncode == 2
    assert (
        "requires a trusted GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256"
        in result.stderr
    )
    assert not (tmp_path / "calls.jsonl").exists(), (
        "a missing trusted pin did not stop remote calls"
    )


def test_restore_preserves_the_old_pin_and_actual_dependency_identity(
    override_environment: dict[str, str], tmp_path: Path
) -> None:
    result = run_override(override_environment)
    assert result.returncode == 0, result.stderr
    documents = records(tmp_path, "restores.jsonl")[0]["documents"]
    deployment = next(item for item in documents if item["kind"] == "Deployment")
    container = next(
        item
        for item in deployment["spec"]["template"]["spec"]["containers"]
        if item["name"] == "reconciler"
    )
    environment = {item["name"]: item.get("value") for item in container["env"]}
    for name in (
        "GPU_FAULT_INSTALLER_TEMPLATE_CONTENT_SHA256",
        "GPU_FAULT_INSTALLER_TEMPLATE_SHA256",
        "GPU_FAULT_NODE_DEPENDENCY_IMAGE",
        "GPU_FAULT_NODE_WHEELHOUSE_SHA256",
    ):
        assert environment[name] == override_environment[name], name
    assert (
        records(tmp_path, "admissions.jsonl")[0]["text"]
        == (tmp_path / "selected.yaml").read_bytes().decode()
    )
