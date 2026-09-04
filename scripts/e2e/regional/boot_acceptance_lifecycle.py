from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from gpu_fault.admin.site import load_site
from scripts.e2e.regional.boot_acceptance_common import (
    ROOT,
    BootAcceptanceError,
    SiteFixture,
    run,
    write_log,
)


def admin_command(
    *arguments: str,
    timeout: int = 21600,
) -> subprocess.CompletedProcess[str]:
    return run(
        [sys.executable, "-m", "gpu_fault.admin.cli", *arguments],
        check=False,
        timeout=timeout,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


def deployment_generations(site_file: Path) -> dict[str, Any]:
    site = load_site(site_file, repository_root=ROOT)
    cluster_ids = [str(item["cluster_id"]) for item in site.release_config["clusters"]]
    if not cluster_ids:
        raise BootAcceptanceError("site contains no GPU clusters")
    fixture = SiteFixture(site_file, cluster_ids[0])
    cpu = json.loads(
        fixture.regional.kubectl(
            "cpu",
            "get",
            "deployment",
            "-o",
            "json",
        )
    )
    gpu = {}
    for cluster_id in cluster_ids:
        target = SiteFixture(site_file, cluster_id)
        value = json.loads(
            target.regional.kubectl(
                "gpu",
                "get",
                "deployment",
                "-o",
                "json",
            )
        )
        gpu[cluster_id] = {
            item["metadata"]["name"]: item["metadata"].get("generation")
            for item in value.get("items", [])
        }
    return {
        "cpu": {
            item["metadata"]["name"]: item["metadata"].get("generation")
            for item in cpu.get("items", [])
        },
        "gpu": gpu,
    }


def run_boot016(arguments: argparse.Namespace, case_dir: Path) -> dict[str, Any]:
    state_dir = arguments.bootstrap_state_dir.resolve()
    if state_dir.exists() and any(state_dir.iterdir()):
        raise BootAcceptanceError(
            "BOOT-016 requires a new or empty isolated --bootstrap-state-dir"
        )
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    command = [
        "deploy",
        "--cpu-cluster-arn",
        arguments.cpu_cluster_arn,
    ]
    for value in arguments.gpu_cluster_arn:
        command.extend(["--gpu-cluster-arn", value])
    command.extend(
        [
            "--state-dir",
            str(state_dir),
            "--admin-email",
            arguments.admin_email,
        ]
    )
    first = admin_command(*command)
    write_log(case_dir / "first-deploy.log", first)
    if first.returncode:
        cleanup = None
        if (state_dir / "site.yaml").is_file():
            cleanup = admin_command(
                "uninstall",
                "--state-dir",
                str(state_dir),
                "--cpu-cluster",
                "keep",
                "--confirm",
                "UNINSTALL_GPU_FAULT",
            )
            write_log(case_dir / "failed-deploy-cleanup.log", cleanup)
        return {
            "verdict": "FAIL",
            "checks": {
                "first_deploy": False,
                "failure_cleanup": cleanup is None or cleanup.returncode == 0,
            },
            "limitations": [
                "The isolated deployment did not reach verification; no later "
                "BOOT-016 conclusion is valid."
            ],
        }
    site_file = state_dir / "site.yaml"
    if not site_file.is_file():
        raise BootAcceptanceError("admin deploy produced no managed site")
    site = load_site(site_file, repository_root=ROOT)
    clusters = [str(item["cluster_id"]) for item in site.release_config["clusters"]]
    before = deployment_generations(site_file)
    empty_guard = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/regional/test_regional_admin_commands.py::"
            "test_deploy_rejects_empty_cluster_set_during_bootstrap",
        ],
        check=False,
        timeout=600,
    )
    write_log(case_dir / "empty-bootstrap-guard.log", empty_guard)
    verify = admin_command("verify", "--state-dir", str(state_dir), timeout=1800)
    status = admin_command("status", "--state-dir", str(state_dir), timeout=1800)
    second = admin_command(*command)
    write_log(case_dir / "verify.log", verify)
    write_log(case_dir / "status.log", status)
    write_log(case_dir / "second-deploy.log", second)
    after = deployment_generations(site_file)
    live_fixture = SiteFixture(site_file, clusters[0])
    registry = json.loads(
        live_fixture.regional.kubectl(
            "cpu",
            "get",
            "secret",
            "gpu-fault-regional-clusters",
            "-o",
            "json",
        )
    )
    registry_clusters = sorted(
        item["cluster_id"]
        for item in json.loads(base64.b64decode(registry["data"]["clusters.json"]))
    )
    release_state = json.loads(
        live_fixture.regional.kubectl(
            "cpu",
            "get",
            "configmap",
            "gpu-fault-regional-release-state",
            "-o",
            "json",
        )
    )
    state = json.loads(release_state["data"]["state.json"])
    cpu_deployments = json.loads(
        live_fixture.regional.kubectl(
            "cpu",
            "get",
            "deployment",
            "-o",
            "json",
        )
    )
    gpu_deployment_items = []
    for cluster_id in clusters:
        target_fixture = SiteFixture(site_file, cluster_id)
        gpu_deployment_items.extend(
            json.loads(
                target_fixture.regional.kubectl(
                    "gpu",
                    "get",
                    "deployment",
                    "-o",
                    "json",
                )
            ).get("items", [])
        )
    registry_created = datetime.fromisoformat(
        str(registry["metadata"]["creationTimestamp"]).replace("Z", "+00:00")
    )
    workload_created = [
        datetime.fromisoformat(
            str(item["metadata"]["creationTimestamp"]).replace("Z", "+00:00")
        )
        for item in [
            *cpu_deployments.get("items", []),
            *gpu_deployment_items,
        ]
    ]
    checks = {
        "state_dir_mode_0700": state_dir.stat().st_mode & 0o777 == 0o700,
        "site_mode_0600": site_file.stat().st_mode & 0o777 == 0o600,
        "site_clusters_nonempty": bool(clusters),
        "empty_bootstrap_guard": empty_guard.returncode == 0,
        "registry_matches_site": registry_clusters == sorted(clusters),
        "registry_created_before_workloads": bool(workload_created)
        and all(registry_created <= item for item in workload_created),
        "release_state_completed": str(state.get("phase", "")).lower()
        in {"complete", "completed"},
        "verify_passed": verify.returncode == 0,
        "status_passed": status.returncode == 0,
        "rerun_passed": second.returncode == 0,
        "rerun_no_rollout_generation_change": before == after,
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "state_dir": str(state_dir),
        "cluster_count": len(clusters),
        "limitations": [
            "The runner uses only caller-approved existing EKS/HyperPod ARNs; "
            "it does not create, replace or damage GPU nodes."
        ],
    }


def run_boot017(
    arguments: argparse.Namespace,
    predecessor: dict[str, Any],
    case_dir: Path,
) -> dict[str, Any]:
    gate = run(
        [sys.executable, "scripts/check-manual-command-order.py", "--verbose"],
        check=False,
        timeout=600,
    )
    selftest = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/regional/test_production_safety_config.py::"
            "test_manual_command_order_gate_is_not_constant_green",
        ],
        check=False,
        timeout=600,
    )
    status = admin_command(
        "status",
        "--state-dir",
        str(arguments.bootstrap_state_dir.resolve()),
        timeout=1800,
    )
    write_log(case_dir / "manual-order.log", gate)
    write_log(case_dir / "manual-order-selftest.log", selftest)
    write_log(case_dir / "greenfield-status.log", status)
    checks = {
        "manual_command_order_gate": gate.returncode == 0,
        "constant_green_selftest": selftest.returncode == 0,
        "greenfield_sequence_reused_boot016": predecessor.get("valid", False),
        "greenfield_status_passed": status.returncode == 0,
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "limitations": [
            "The live linear deployment is the immediately preceding BOOT-016 "
            "run; BOOT-017 adds the full manual-order gate and its negative self-test."
        ],
    }


def copy_checkout(destination: Path) -> None:
    ignored = {
        ".git",
        ".venv",
        ".codex",
        "artifacts",
        "dist",
        "build",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
    }

    def ignore(_path: str, names: list[str]) -> set[str]:
        return {name for name in names if name in ignored}

    shutil.copytree(ROOT, destination, ignore=ignore)


def run_boot018(
    arguments: argparse.Namespace,
    case_dir: Path,
) -> dict[str, Any]:
    manifests: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="gpu-fault-boot018-") as temporary:
        base = Path(temporary)
        for mask in ("077", "002"):
            checkout = base / f"repo-{mask}"
            copy_checkout(checkout)
            environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
            completed = run(
                [
                    sys.executable,
                    "scripts/build-release-artifacts.py",
                    "--python",
                    sys.executable,
                ],
                cwd=checkout,
                env=environment,
                check=False,
                timeout=7200,
                umask=int(mask, 8),
            )
            write_log(case_dir / f"build-{mask}.log", completed)
            if completed.returncode:
                raise BootAcceptanceError(f"BOOT-018 build failed under umask {mask}")
            artifact_test = run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_artifact_consistency.py",
                ],
                cwd=checkout,
                env={
                    **environment,
                    "GPU_FAULT_REQUIRE_BUILD_ARTIFACTS": "1",
                },
                check=False,
                timeout=3600,
            )
            write_log(case_dir / f"artifact-tests-{mask}.log", artifact_test)
            if artifact_test.returncode:
                raise BootAcceptanceError(
                    f"BOOT-018 artifact tests failed under umask {mask}"
                )
            manifest = json.loads(
                (checkout / "dist/current-release.json").read_text(encoding="utf-8")
            )
            manifests[mask] = manifest
        left = manifests["077"]
        right = manifests["002"]
        identical = all(
            (
                left["release_id"] == right["release_id"],
                left["bundle_sha256"] == right["bundle_sha256"],
                *(
                    left["components"][name]["wheel_sha256"]
                    == right["components"][name]["wheel_sha256"]
                    and left["components"][name]["module_digest"]
                    == right["components"][name]["module_digest"]
                    for name in ("control_plane", "executor", "node_runtime")
                ),
            )
        )
        negative = base / "repo-077"
        target = negative / "src/gpu_fault/app/builtin_metric_contributors.py"
        target.write_text(
            target.read_text(encoding="utf-8")
            + "\n# BOOT-018 temporary digest probe\n",
            encoding="utf-8",
        )
        tamper = run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_artifact_consistency.py::"
                "test_built_component_wheels_match_their_source_closures",
            ],
            cwd=negative,
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT / "src"),
                "GPU_FAULT_REQUIRE_BUILD_ARTIFACTS": "1",
            },
            check=False,
            timeout=1200,
        )
        write_log(case_dir / "tamper-negative.log", tamper)
    verify = admin_command(
        "verify",
        "--state-dir",
        str(arguments.bootstrap_state_dir.resolve()),
        timeout=1800,
    )
    write_log(case_dir / "live-verify.log", verify)
    cleanup = None
    if not arguments.retain_bootstrap_site:
        cleanup = admin_command(
            "uninstall",
            "--state-dir",
            str(arguments.bootstrap_state_dir.resolve()),
            "--cpu-cluster",
            "keep",
            "--confirm",
            "UNINSTALL_GPU_FAULT",
        )
        write_log(case_dir / "bootstrap-site-cleanup.log", cleanup)
    checks = {
        "wheel_and_bundle_hashes_identical": identical,
        "tamper_negative_failed": tamper.returncode != 0,
        "live_runtime_identity_verify": verify.returncode == 0,
        "bootstrap_site_cleanup": cleanup is None or cleanup.returncode == 0,
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "release_id": manifests["077"]["release_id"],
        "limitations": [
            "Runtime identity is consumed from the administrator verify report; "
            "the fixture does not trust version strings or annotations."
        ],
    }
