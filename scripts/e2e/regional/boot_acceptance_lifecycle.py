from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from gpu_fault.admin.site import load_site
from scripts.e2e.regional.boot_acceptance_common import (
    ROOT,
    BootAcceptanceError,
    SiteFixture,
    run,
    write_log,
)

UNINSTALL_CONFIRMATION = "UNINSTALL_GPU_FAULT"
# `gpu-fault-admin verify` reuses the quick gate's runtime-identity evidence when
# this variable names it, and the reused check carries no per-replica digests.
# BOOT-018 needs the digests, so the variable is withheld from its verify run.
QUICK_VALIDATION_EVIDENCE_ENV = "GPU_FAULT_QUICK_VALIDATION_EVIDENCE"
RUNTIME_IDENTITY_CHECK = "runtime_component_identity"
RELEASE_METADATA_CONFIGMAP = "gpu-fault-release-metadata"
ACTIVE_AGENT_PROBE = r"""
import json
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext

now = datetime.now(timezone.utc)
agents = [
    {
        "cluster_id": item.cluster_id,
        "node_id": item.node_id,
        "artifact_sha256": item.artifact_sha256,
        "compatibility_digest": item.compatibility_digest or item.artifact_sha256,
    }
    for item in ApplicationContext.from_environment().store.list_agents()
    if getattr(item.lifecycle_state, "value", item.lifecycle_state) == "ACTIVE"
    and item.lease_expires_at is not None
    and item.lease_expires_at > now
]
print(json.dumps({"agents": agents}, sort_keys=True))
"""


def admin_command(
    *arguments: str,
    timeout: int = 21600,
    drop_environment: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for name in drop_environment:
        environment.pop(name, None)
    return run(
        [sys.executable, "-m", "gpu_fault.admin.cli", *arguments],
        check=False,
        timeout=timeout,
        env=environment,
    )


def uninstall_site(state_dir: Path) -> subprocess.CompletedProcess[str]:
    return admin_command(
        "uninstall",
        "--state-dir",
        str(state_dir),
        "--cpu-cluster",
        "keep",
        "--confirm",
        UNINSTALL_CONFIRMATION,
    )


def failure_outcome(exc: BaseException, *, limitation: str) -> dict[str, Any]:
    """The FAIL outcome a case records when its body raised.

    Kept separate from the runner's generic handler so a lifecycle case can
    run its cleanup *and* keep the frames of the original failure in the same
    evidence document.
    """

    return {
        "verdict": "FAIL",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exception(exc)[-12:],
        "checks": {},
        "limitations": [limitation],
    }


def site_identity(site_file: Path) -> dict[str, Any] | None:
    """``release_id``/``cluster_id`` of a managed site, or ``None`` if unreadable.

    Every case result carries the identity so the next case can bind its
    predecessor check to the same deployment; a site whose control plane cannot
    be reached yields nothing rather than a guessed identity.
    """

    try:
        site = load_site(site_file, repository_root=ROOT)
        clusters = [str(item["cluster_id"]) for item in site.release_config["clusters"]]
        if not clusters:
            return None
        return dict(SiteFixture(site_file, clusters[0]).regional.evidence_identity())
    except Exception:  # noqa: BLE001 - identity is best effort, never a verdict
        return None


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


def _boot016_deploy_command(
    arguments: argparse.Namespace, state_dir: Path
) -> list[str]:
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
    return command


def _boot016_verified_site(
    command: list[str],
    state_dir: Path,
    case_dir: Path,
) -> dict[str, Any]:
    first = admin_command(*command)
    write_log(case_dir / "first-deploy.log", first)
    if first.returncode:
        return {
            "verdict": "FAIL",
            "checks": {"first_deploy": False},
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
        **(site_identity(site_file) or {}),
        "limitations": [
            "The runner uses only caller-approved existing EKS/HyperPod ARNs; "
            "it does not create, replace or damage GPU nodes."
        ],
    }


def cleanup_isolated_site(
    state_dir: Path,
    case_dir: Path,
    *,
    log_name: str,
    retain: bool,
    uninstall: Callable[[Path], subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Uninstall the isolated site unless the operator asked to keep it.

    Returns the record the case stores beside its checks: whether a site was
    there to remove, whether it was retained, and the uninstall exit code. A
    missing ``site.yaml`` means the deploy never committed a site, so there is
    nothing gpu-fault-admin could remove.
    """

    site_present = (state_dir / "site.yaml").is_file()
    record: dict[str, Any] = {
        "site_present": site_present,
        "retained": retain,
        "uninstall_ran": False,
        "uninstall_returncode": None,
    }
    if not site_present or retain:
        return record
    completed = (uninstall or uninstall_site)(state_dir)
    write_log(case_dir / log_name, completed)
    record["uninstall_ran"] = True
    record["uninstall_returncode"] = completed.returncode
    return record


def run_boot016(arguments: argparse.Namespace, case_dir: Path) -> dict[str, Any]:
    state_dir = arguments.bootstrap_state_dir.resolve()
    if state_dir.exists() and any(state_dir.iterdir()):
        raise BootAcceptanceError(
            "BOOT-016 requires a new or empty isolated --bootstrap-state-dir"
        )
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    command = _boot016_deploy_command(arguments, state_dir)
    outcome = failure_outcome(
        BootAcceptanceError("BOOT-016 was interrupted before it recorded a verdict"),
        limitation="The isolated deployment did not reach verification; no later "
        "BOOT-016 conclusion is valid.",
    )
    try:
        outcome = _boot016_verified_site(command, state_dir, case_dir)
    except Exception as exc:
        # A deploy timeout (subprocess.TimeoutExpired), a kubectl/JSON error or
        # a BootAcceptanceError used to skip the uninstall entirely and leave
        # Aurora/NLB/AMP/ECR/IAM behind; the cleanup below runs for all of them.
        outcome = failure_outcome(
            exc,
            limitation="The case stopped at the first failed step; the isolated "
            "site was removed unless --retain-bootstrap-site was given.",
        )
    finally:
        # A PASS leaves the site for BOOT-017/018, which reuse it; anything
        # else removes it unless the operator explicitly keeps it for triage.
        passed = outcome.get("verdict") == "PASS"
        cleanup = cleanup_isolated_site(
            state_dir,
            case_dir,
            log_name="failed-deploy-cleanup.log",
            retain=passed or bool(arguments.retain_bootstrap_site),
        )
        outcome["cleanup"] = cleanup
        if cleanup["uninstall_ran"]:
            outcome.setdefault("checks", {})["failure_cleanup"] = (
                cleanup["uninstall_returncode"] == 0
            )
    return outcome


def boot016_reuse(
    predecessor: dict[str, Any],
    *,
    expected_state_dir: Path,
    read_document: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    """Whether the live site BOOT-017 runs against is BOOT-016's PASS site.

    ``predecessor["valid"]`` is ``True`` under selective scope with no
    greenfield run at all, so it cannot carry this check. The evidence
    document itself must say PASS and name the same ``state_dir``; its
    ``status_passed`` check is consumed instead of running ``admin status`` a
    second time on a site nothing has touched since.
    """

    result: dict[str, Any] = {
        "reused": "NOT_EVALUATED",
        "status_passed": "NOT_EVALUATED",
        "reason": None,
    }
    if str(predecessor.get("verdict") or "") != "PASS":
        result["reason"] = "BOOT-016 evidence verdict is not PASS"
        return result
    path = predecessor.get("path")
    if not path:
        result["reason"] = "BOOT-016 evidence path is unknown"
        return result
    try:
        document = read_document(Path(str(path)))
    except (OSError, ValueError) as exc:
        result["reason"] = f"cannot read BOOT-016 evidence: {exc}"
        return result
    if document.get("case_id") != "GF-REGIONAL-BOOT-016":
        result["reason"] = "evidence does not belong to BOOT-016"
        return result
    recorded = document.get("state_dir")
    if not recorded or Path(str(recorded)).resolve() != expected_state_dir.resolve():
        result["reason"] = "BOOT-016 state_dir differs from --bootstrap-state-dir"
        return result
    checks = document.get("checks") or {}
    result["reused"] = True
    result["status_passed"] = bool(checks.get("status_passed"))
    return result


def _read_json_document(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("evidence is not a JSON object")
    return value


def run_boot017(
    arguments: argparse.Namespace,
    predecessor: dict[str, Any],
    case_dir: Path,
) -> dict[str, Any]:
    state_dir = arguments.bootstrap_state_dir.resolve()
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
    write_log(case_dir / "manual-order.log", gate)
    write_log(case_dir / "manual-order-selftest.log", selftest)
    reuse = boot016_reuse(
        predecessor,
        expected_state_dir=state_dir,
        read_document=_read_json_document,
    )
    checks: dict[str, Any] = {
        "manual_command_order_gate": gate.returncode == 0,
        "constant_green_selftest": selftest.returncode == 0,
        "greenfield_sequence_reused_boot016": reuse["reused"],
        "greenfield_status_passed": reuse["status_passed"],
    }
    local_checks_passed = (
        checks["manual_command_order_gate"] and checks["constant_green_selftest"]
    )
    if not local_checks_passed:
        verdict = "FAIL"
    elif reuse["reused"] is True and reuse["status_passed"] is True:
        verdict = "PASS"
    elif reuse["reused"] == "NOT_EVALUATED":
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    return {
        "verdict": verdict,
        "checks": checks,
        "boot016_reuse": reuse,
        **(site_identity(state_dir / "site.yaml") or {}),
        "limitations": [
            "The live linear deployment is the immediately preceding BOOT-016 "
            "run; BOOT-017 adds the full manual-order gate and its negative "
            "self-test and consumes BOOT-016's status check instead of rerunning it."
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


ARTIFACT_TESTS = {
    # The first build runs the whole artifact suite; the second only has to
    # prove its wheels close over the same sources, everything else is the
    # same code path run twice.
    "077": "tests/test_artifact_consistency.py",
    "002": (
        "tests/test_artifact_consistency.py::"
        "test_built_component_wheels_match_their_source_closures"
    ),
}


def build_release_under_umask(
    mask: str,
    *,
    base: Path,
    case_dir: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = run,
    copy: Callable[[Path], None] = copy_checkout,
) -> dict[str, Any]:
    """Copy the checkout, build it under ``mask`` and run its artifact tests."""

    checkout = base / f"repo-{mask}"
    copy(checkout)
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    completed = runner(
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
    artifact_test = runner(
        [sys.executable, "-m", "pytest", "-q", ARTIFACT_TESTS[mask]],
        cwd=checkout,
        env={**environment, "GPU_FAULT_REQUIRE_BUILD_ARTIFACTS": "1"},
        check=False,
        timeout=3600,
    )
    write_log(case_dir / f"artifact-tests-{mask}.log", artifact_test)
    if artifact_test.returncode:
        raise BootAcceptanceError(f"BOOT-018 artifact tests failed under umask {mask}")
    manifest = json.loads(
        (checkout / "dist/current-release.json").read_text(encoding="utf-8")
    )
    return {"checkout": checkout, "manifest": manifest}


def parse_verify_report(stdout: str) -> dict[str, Any]:
    """The health report ``gpu-fault-admin verify`` printed.

    The report is an indented JSON document; the rollout wrapper may print
    plain lines ahead of it, so the parse starts at the first line that is
    exactly ``{``.
    """

    text = stdout.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        lines = text.splitlines()
        starts = [index for index, line in enumerate(lines) if line.strip() == "{"]
        if not starts:
            raise BootAcceptanceError("verify printed no JSON report") from None
        value = json.loads("\n".join(lines[starts[0] :]))
    if not isinstance(value, dict):
        raise BootAcceptanceError("verify report is not a JSON object")
    return value


def runtime_identity_matches_release(
    report: dict[str, Any],
    *,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    agents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare what verify saw against the release metadata itself.

    ``verify.returncode == 0`` only says verify's own expectations held. This
    reads the per-replica module digests from the report's
    ``runtime_component_identity`` check, the required pins from the release
    metadata ConfigMap and every ACTIVE Agent's artifact pin, and compares each
    with the built manifest's component digests. Any field that is missing is
    a failure, not a skip: a report without replica digests proves nothing.
    """

    reasons: list[str] = []
    components = manifest.get("components") or {}

    def component(name: str, field: str) -> str:
        value = str((components.get(name) or {}).get(field) or "")
        if len(value) != 64:
            reasons.append(f"manifest components.{name}.{field} is missing")
        return value

    control_digest = component("control_plane", "module_digest")
    executor_digest = component("executor", "module_digest")
    executor_wheel = component("executor", "wheel_sha256")
    node_wheel = component("node_runtime", "wheel_sha256")
    node_digest = component("node_runtime", "module_digest")

    check = next(
        (
            item
            for item in report.get("checks") or []
            if isinstance(item, dict) and item.get("name") == RUNTIME_IDENTITY_CHECK
        ),
        None,
    )
    replicas: dict[str, str] = {}
    if check is None or check.get("status") != "PASS":
        reasons.append("verify report has no passing runtime_component_identity check")
    else:
        details = check.get("details") or {}
        control = (details.get("control_plane") or {}).get("deployments")
        clusters = (details.get("executor") or {}).get("clusters")
        if not isinstance(control, dict) or not isinstance(clusters, dict):
            reasons.append("runtime_component_identity carries no replica digests")
        else:
            for deployment, pods in control.items():
                for pod, digest in (pods or {}).items():
                    replicas[f"cpu/{deployment}/{pod}"] = str(digest)
                    if str(digest) != control_digest:
                        reasons.append(f"cpu/{deployment}/{pod} digest differs")
            for cluster_id, deployments in clusters.items():
                for deployment, pods in (deployments or {}).items():
                    for pod, digest in (pods or {}).items():
                        replicas[f"gpu/{cluster_id}/{deployment}/{pod}"] = str(digest)
                        if str(digest) != executor_digest:
                            reasons.append(
                                f"gpu/{cluster_id}/{deployment}/{pod} digest differs"
                            )
            if not replicas:
                reasons.append("runtime_component_identity lists no replicas")

    for key, expected in (
        ("required-regional-executor-artifact-sha256", executor_wheel),
        ("required-regional-executor-compatibility-digest", executor_digest),
        ("required-agent-artifact-sha256", node_wheel),
        ("required-agent-compatibility-digest", node_digest),
    ):
        actual = str(metadata.get(key) or "")
        if not actual:
            reasons.append(f"release metadata lacks {key}")
        elif actual != expected:
            reasons.append(f"release metadata {key} differs from the manifest")

    if not agents:
        reasons.append("no ACTIVE Agent identity to compare")
    for agent in agents:
        label = f"{agent.get('cluster_id')}/{agent.get('node_id')}"
        if str(agent.get("artifact_sha256") or "") != node_wheel:
            reasons.append(f"agent {label} artifact pin differs from the manifest")
        if str(agent.get("compatibility_digest") or "") != node_digest:
            reasons.append(f"agent {label} compatibility digest differs")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "replica_count": len(replicas),
        "agent_count": len(agents),
        "expected": {
            "control_plane_module_digest": control_digest,
            "executor_module_digest": executor_digest,
            "executor_wheel_sha256": executor_wheel,
            "node_runtime_wheel_sha256": node_wheel,
            "node_runtime_module_digest": node_digest,
        },
    }


def _live_release_identity(state_dir: Path) -> dict[str, Any]:
    """The manifest, release metadata and ACTIVE Agents of the isolated site."""

    site_file = state_dir / "site.yaml"
    site = load_site(site_file, repository_root=ROOT)
    manifest_path = Path(str(site.release_config["release"]["manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clusters = [str(item["cluster_id"]) for item in site.release_config["clusters"]]
    if not clusters:
        raise BootAcceptanceError("BOOT-018 site contains no GPU clusters")
    fixture = SiteFixture(site_file, clusters[0])
    configmap = json.loads(
        fixture.regional.kubectl(
            "cpu",
            "get",
            "configmap",
            RELEASE_METADATA_CONFIGMAP,
            "-o",
            "json",
        )
    )
    agents = fixture.regional.cpu_python(ACTIVE_AGENT_PROBE)
    return {
        "manifest": manifest,
        "metadata": dict(configmap.get("data") or {}),
        "agents": list(agents.get("agents") or []),
        "identity": dict(fixture.regional.evidence_identity()),
    }


def boot018_body(state_dir: Path, case_dir: Path) -> dict[str, Any]:
    manifests: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="gpu-fault-boot018-") as temporary:
        base = Path(temporary)
        # The two umask builds share nothing but the source tree, so they run
        # side by side; a build failure in either aborts the case.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                mask: pool.submit(
                    build_release_under_umask,
                    mask,
                    base=base,
                    case_dir=case_dir,
                )
                for mask in ("077", "002")
            }
            for mask, future in futures.items():
                manifests[mask] = future.result()["manifest"]
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
        str(state_dir),
        timeout=1800,
        drop_environment=(QUICK_VALIDATION_EVIDENCE_ENV,),
    )
    write_log(case_dir / "live-verify.log", verify)
    live = _live_release_identity(state_dir)
    identity = runtime_identity_matches_release(
        parse_verify_report(verify.stdout),
        manifest=live["manifest"],
        metadata=live["metadata"],
        agents=live["agents"],
    )
    checks = {
        "wheel_and_bundle_hashes_identical": identical,
        "tamper_negative_failed": tamper.returncode != 0,
        "live_verify_passed": verify.returncode == 0,
        "live_runtime_identity_verify": identity["passed"],
    }
    return {
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "release_id": manifests["077"]["release_id"],
        "runtime_identity": identity,
        **live["identity"],
        "limitations": [
            "Runtime identity is read from the administrator verify report and "
            "compared with the release manifest, metadata ConfigMap and ACTIVE "
            "Agent pins; version strings and annotations are not trusted."
        ],
    }


def run_boot018(
    arguments: argparse.Namespace,
    case_dir: Path,
) -> dict[str, Any]:
    state_dir = arguments.bootstrap_state_dir.resolve()
    outcome = failure_outcome(
        BootAcceptanceError("BOOT-018 was interrupted before it recorded a verdict"),
        limitation="The case stopped before a verdict; the isolated site was "
        "removed unless --retain-bootstrap-site was given.",
    )
    try:
        outcome = boot018_body(state_dir, case_dir)
    except Exception as exc:
        # Build and artifact-test failures used to raise past the uninstall and
        # leave the BOOT-016 site (Aurora/NLB/AMP/ECR/IAM) behind.
        outcome = failure_outcome(
            exc,
            limitation="The case stopped at the first failed step; the isolated "
            "site was removed unless --retain-bootstrap-site was given.",
        )
    finally:
        cleanup = cleanup_isolated_site(
            state_dir,
            case_dir,
            log_name="bootstrap-site-cleanup.log",
            retain=bool(arguments.retain_bootstrap_site),
        )
        outcome["cleanup"] = cleanup
        checks = outcome.setdefault("checks", {})
        checks["bootstrap_site_cleanup"] = (
            not cleanup["uninstall_ran"] or cleanup["uninstall_returncode"] == 0
        )
        if outcome.get("verdict") == "PASS" and not checks["bootstrap_site_cleanup"]:
            outcome["verdict"] = "FAIL"
    return outcome
