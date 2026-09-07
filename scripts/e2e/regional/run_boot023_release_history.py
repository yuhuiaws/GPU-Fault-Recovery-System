#!/usr/bin/env python3
"""GF-REGIONAL-BOOT-023: release history, durable registry publish, rollback flag.

One NOOP release through the sanctioned path -- ``RegionalRelease.noop``, the
same call ``gpu-fault-admin deploy`` reaches when ``classify_release`` says
NOOP, driven from the release config ``boot020_release_candidates.py configs``
materialises as ``noop`` -- and three ARCH-H facts are read back from the site:

* H5: ``gpu-fault-release-history`` gained append-only entries naming the live
  release id, the phase, the state/plan digests, the operator identity and a
  redacted command line; the admin-side ndjson mirror carries the same tail;
  the previous-snapshot ConfigMaps stay within the retention bound and a NOOP
  does not touch them.
* H3: on every CPU Pod the durable registry head digest equals the Secret's
  configured digest and ``/healthz`` reports ``secret_drift=false``; the head
  generation does not move across a NOOP.
* H2: the noop manifest, re-read with ``database.rollback_compatible: true``,
  is refused by ``parse_delivery_identity`` with the explanatory error.

The NOOP classification is a hard stop, not a convenience: the release engine
is only ever asked to do what the live site already is, so the only things the
case can change are the history ConfigMap and its mirror. Plan mode does every
read and the offline refusal check; ``--execute`` adds the one ``noop`` call.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import boot023_verdicts as verdicts  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseRunner,
    add_live_arguments,
    run_standard_case,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    predecessor_evidence,
    required,
    run_case_main,
    runtime_identity_errors,
    settings_from_arguments,
)

CASE_ID = verdicts.CASE_ID
PREDECESSOR_CASE_ID = verdicts.PREDECESSOR_CASE_ID
CONFIRMATION = verdicts.CONFIRMATION
PROBE_SCRIPT = Path(__file__).with_name("probes") / "boot023_registry_probe.py"
HISTORY_DIR_ENV = "GPU_FAULT_RELEASE_HISTORY_DIR"
# The three CPU runtime Deployments; the inventory module is the source of
# truth and is consulted at run time, this tuple only names the fallback.
CPU_RUNTIME_DEPLOYMENTS = (
    "gpu-fault-api-ha",
    "gpu-fault-control-worker",
    "gpu-fault-telemetry-spool-worker",
)


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    noop_config: Path
    predecessor_path: Path

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_NOOP_RELEASE_CONFIG": str(self.noop_config),
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
        }


def configure(arguments: argparse.Namespace) -> Settings:
    predecessor = (
        Path(arguments.predecessor_evidence).expanduser().resolve()
        if arguments.predecessor_evidence
        else (
            arguments.run_dir
            / "cases"
            / PREDECESSOR_CASE_ID
            / f"{PREDECESSOR_CASE_ID}.json"
        ).resolve()
    )
    noop_config = (
        Path(
            required(
                arguments.noop_config or os.getenv("GPU_FAULT_NOOP_RELEASE_CONFIG", ""),
                "NOOP release config",
            )
        )
        .expanduser()
        .resolve()
    )
    if not noop_config.is_file():
        raise RegionalFixtureError(f"NOOP release config does not exist: {noop_config}")
    return Settings(
        regional=settings_from_arguments(arguments),
        noop_config=noop_config,
        predecessor_path=predecessor,
    )


# --------------------------------------------------------------------------- #
# Release engine access (the ``gpu_fault_release`` package, imported lazily so
# a plan-only run that never reaches the engine does not pay for it)
# --------------------------------------------------------------------------- #
def deploy_modules() -> dict[str, Any]:
    package = "gpu_fault_release"
    return {
        "rollout_regional_release": importlib.import_module(f"{package}.rollout"),
        **{
            name: importlib.import_module(f"{package}.{name}")
            for name in (
                "regional_release_config",
                "regional_release_diff",
                "regional_release_history",
                "regional_deployment_inventory",
            )
        },
    }


def cpu_runtime_deployments(modules: dict[str, Any]) -> tuple[str, ...]:
    names = getattr(
        modules["regional_deployment_inventory"], "CPU_RUNTIME_DEPLOYMENTS", ()
    )
    return tuple(str(name) for name in names) or CPU_RUNTIME_DEPLOYMENTS


def release_for(settings: Settings, modules: dict[str, Any], *, dry_run: bool) -> Any:
    rollout = modules["rollout_regional_release"]
    config = modules["regional_release_config"].ReleaseConfig.load(settings.noop_config)
    return rollout.RegionalRelease(config, rollout.Runner(dry_run=dry_run))


def classify(release: Any, modules: dict[str, Any]) -> dict[str, Any]:
    state = release._load_state()
    diff = modules["regional_release_diff"].classify_release(release, state)
    return {
        "kind": diff.kind.value,
        "changed": sorted(diff.changed),
        "release_id": str(state.get("release_id") or ""),
        "phase": state.get("phase"),
        "release_lifecycle": state.get("release_lifecycle"),
        "diff": diff.as_dict(),
    }


def manifest_with_rollback_flag(config_path: Path) -> dict[str, Any]:
    """The noop config's own manifest, re-read with the forbidden claim set."""

    document = json.loads(config_path.read_text(encoding="utf-8"))
    release = dict(document.get("release") or document)
    reference = release.get("manifest")
    if not reference:
        raise RegionalFixtureError("the NOOP release config names no manifest")
    manifest_path = Path(str(reference))
    if not manifest_path.is_absolute():
        manifest_path = config_path.parent / manifest_path
    manifest: dict[str, Any] = copy.deepcopy(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    database = dict(manifest.get("database") or {})
    database["rollback_compatible"] = True
    manifest["database"] = database
    return manifest


def rollback_flag_rejection(manifest: dict[str, Any], config_module: Any) -> str:
    """The ReleaseError message the parser raises, or "" when it accepts."""

    try:
        config_module.parse_delivery_identity(
            manifest, dict(manifest.get("components") or {})
        )
    except config_module.ReleaseError as exc:
        return str(exc)
    return ""


# --------------------------------------------------------------------------- #
# Site reads
# --------------------------------------------------------------------------- #
def history_entries(regional: RegionalLiveFixture) -> list[dict[str, Any]]:
    output = regional.kubectl(
        "cpu",
        "get",
        "configmap",
        verdicts.HISTORY_CONFIG_MAP,
        "-o",
        "json",
        "--ignore-not-found",
    ).strip()
    if not output:
        return []
    value = json.loads(output)
    raw = (value.get("data") or {}).get(verdicts.HISTORY_KEY)
    return verdicts.parse_history(str(raw or ""))


def previous_snapshot_groups(regional: RegionalLiveFixture) -> list[list[str]]:
    value = json.loads(
        regional.kubectl(
            "cpu",
            "get",
            "configmap",
            "-l",
            f"{verdicts.PREVIOUS_SNAPSHOT_LABEL}=true",
            "-o",
            "json",
        )
    )
    return verdicts.snapshot_groups(list(value.get("items") or []))


def pod_http_port(regional: RegionalLiveFixture, pod: str) -> int:
    value = json.loads(regional.kubectl("cpu", "get", "pod", pod, "-o", "json"))
    for container in value.get("spec", {}).get("containers", []):
        for port in container.get("ports") or []:
            if port.get("name") == "http":
                return int(port["containerPort"])
    raise RegionalFixtureError(f"{pod} exposes no containerPort named http")


def registry_probes(
    regional: RegionalLiveFixture, deployments: tuple[str, ...]
) -> list[dict[str, Any]]:
    script = PROBE_SCRIPT.read_text(encoding="utf-8")
    probes: list[dict[str, Any]] = []
    for app in deployments:
        for pod in regional.ready_pods("cpu", app):
            name = str(pod["name"])
            output = regional.kubectl(
                "cpu",
                "exec",
                "-i",
                name,
                "--",
                "python3",
                "-",
                str(pod_http_port(regional, name)),
                input_text=script,
                timeout=120,
            )
            value = json.loads(output.splitlines()[-1])
            probes.append({"pod": name, "app": app, **value})
    return probes


def focused_tests(case_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/regional/test_release_history.py",
        "tests/regional/test_release_schema_rollback_flag.py",
        "tests/regional/test_release_registry_durable_publish.py",
        "tests/regional/test_boot023_release_history.py",
    ]
    completed = RegionalLiveFixture.run(command, cwd=ROOT, check=False, timeout=600)
    path = case_dir / "focused-tests.log"
    path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    path.chmod(0o600)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
    }


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #
def read_only_preflight(settings: Settings, case_dir: Path) -> dict[str, Any]:
    modules = deploy_modules()
    regional = RegionalLiveFixture(settings.regional)
    deployments = cpu_runtime_deployments(modules)
    classification = classify(release_for(settings, modules, dry_run=True), modules)
    rejection = rollback_flag_rejection(
        manifest_with_rollback_flag(settings.noop_config),
        modules["regional_release_config"],
    )
    history = history_entries(regional)
    groups = previous_snapshot_groups(regional)
    probes = registry_probes(regional, deployments)
    predecessor = predecessor_evidence(settings.predecessor_path, PREDECESSOR_CASE_ID)
    runtime_identity = regional.runtime_identity()
    tests = focused_tests(case_dir)
    errors = verdicts.preflight_errors(
        classification=classification,
        release_id=classification["release_id"],
        state_phase=classification["phase"],
        probes=probes,
        rollback_flag_message=rejection,
        predecessor_valid=bool(predecessor["valid"]),
        tests_passed=tests["passed"],
        history_before=history,
        snapshot_groups_before=groups,
    )
    errors.extend(runtime_identity_errors(runtime_identity))
    result = {
        "release_id": classification["release_id"],
        "classification": classification,
        "rollback_flag_rejection": rejection,
        "history": history,
        "snapshot_groups": groups,
        "registry_probes": probes,
        "cpu_runtime_deployments": list(deployments),
        "predecessor": predecessor,
        "runtime_identity": runtime_identity,
        "focused_tests": tests,
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def plan_identity(preflight: dict[str, Any]) -> dict[str, Any]:
    classification = preflight["classification"]
    return {
        "release_id": preflight["release_id"],
        "classification": classification["kind"],
        "state_phase": classification["phase"],
        "history_entries": len(preflight["history"]),
        "snapshot_groups": len(preflight["snapshot_groups"]),
        "head_generations": sorted(
            {
                int(item.get("head_generation") or 0)
                for item in preflight["registry_probes"]
            }
        ),
    }


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "live-service-action",
        "predecessor": preflight["predecessor"],
        "noop_config": str(settings.noop_config),
        "mutation": (
            "run one NOOP release through RegionalRelease.noop (the path "
            "gpu-fault-admin deploy takes for a NOOP diff): validates the release, "
            "rewrites the release-state ConfigMap with the same content and appends "
            "to the gpu-fault-release-history ConfigMap and its ndjson mirror. No "
            "Deployment rolls, no Secret changes, no registry revision is published."
        ),
        "preflight_identity": plan_identity(preflight),
        "stop_conditions": [
            f"{PREDECESSOR_CASE_ID} has not passed in formal sequence",
            "the release config does not classify as NOOP against the live site",
            "the live release state is not a completed transaction",
            "any CPU Pod reports secret_drift or a durable head digest that differs "
            "from its Secret digest before the release",
            "the manifest with database.rollback_compatible: true is accepted by "
            "the release config parser",
            "the existing history or previous snapshots already violate their bounds",
            "focused regression tests fail",
            "runtime identity drifted between plan and execute",
        ],
        "rollback": {
            "release_is_a_noop_against_the_live_site": True,
            "no_deployment_secret_or_registry_revision_is_written": True,
            "history_configmap_is_append_only_and_bounded": True,
            "runtime_identity_is_verified_after_the_release": True,
        },
        "preflight": preflight,
    }


def verify_plan_identity(case_dir: Path, preflight: dict[str, Any]) -> None:
    plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned = plan["details"]["preflight_identity"]
    current = plan_identity(preflight)
    if current != planned:
        raise RegionalFixtureError(f"{CASE_ID} plan drifted: {planned} != {current}")


# --------------------------------------------------------------------------- #
# Execute
# --------------------------------------------------------------------------- #
def run_noop_release(
    settings: Settings, modules: dict[str, Any], case_dir: Path
) -> dict[str, Any]:
    history_dir = case_dir / "history"
    history_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.environ[HISTORY_DIR_ENV] = str(history_dir)
    release = release_for(settings, modules, dry_run=False)
    classification = classify(release, modules)
    errors = verdicts.classification_errors(classification)
    if errors:
        raise RegionalFixtureError("; ".join(errors))
    diff = modules["regional_release_diff"].ReleaseDiff(
        kind=modules["regional_release_diff"].ReleaseChangeKind(classification["kind"]),
        changed=frozenset(classification["changed"]),
    )
    started = datetime.now(timezone.utc)
    release.noop(diff)
    mirror = history_dir / verdicts.HISTORY_MIRROR_FILE
    return {
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "classification": classification,
        "history_dir": str(history_dir),
        "mirror_lines": (
            mirror.read_text(encoding="utf-8").splitlines() if mirror.is_file() else []
        ),
    }


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    """Drive the live case. Not exercised by the unit suite; the verdicts it
    calls are."""

    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    verify_plan_identity(case_dir, preflight)
    modules = deploy_modules()
    regional = RegionalLiveFixture(settings.regional)
    deployments = cpu_runtime_deployments(modules)
    release_id = str(preflight["release_id"])
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "release_id": release_id,
        "maintenance_window_end": maintenance_window_end.isoformat(),
    }
    errors: list[str] = []
    try:
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError("approved maintenance window has ended")
        history_before = preflight["history"]
        groups_before = preflight["snapshot_groups"]
        probes_before = preflight["registry_probes"]
        release = run_noop_release(settings, modules, case_dir)
        write_json_atomic(case_dir / "noop-release.json", release)
        history_after = history_entries(regional)
        write_json_atomic(case_dir / "history-after.json", {"entries": history_after})
        groups_after = previous_snapshot_groups(regional)
        probes_after = registry_probes(regional, deployments)
        write_json_atomic(
            case_dir / "registry-probes-after.json", {"probes": probes_after}
        )
        rejection = rollback_flag_rejection(
            manifest_with_rollback_flag(settings.noop_config),
            modules["regional_release_config"],
        )
        errors.extend(
            verdicts.history_append_errors(
                history_before, history_after, release_id=release_id
            )
        )
        appended = history_after[len(history_before) :]
        errors.extend(verdicts.mirror_errors(release["mirror_lines"], appended))
        errors.extend(verdicts.snapshot_retention_errors(groups_before, groups_after))
        errors.extend(verdicts.registry_probe_errors(probes_after))
        errors.extend(verdicts.registry_stability_errors(probes_before, probes_after))
        errors.extend(verdicts.rollback_flag_errors(rejection))
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "appended_history_entries": appended,
                "history_entries_before": len(history_before),
                "history_entries_after": len(history_after),
                "snapshot_groups_after": groups_after,
                "rollback_flag_rejection": rejection,
            }
        )
    except Exception as exc:  # noqa: BLE001 - recorded as the case error
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup: dict[str, Any] = {"errors": []}
        try:
            cleanup["runtime_identity"] = regional.verify_runtime_identity(
                preflight["runtime_identity"],
                evidence_path=case_dir / "runtime-identity-after-release.json",
                stage=f"after {CASE_ID} release",
            )
        except Exception as exc:  # noqa: BLE001 - a cleanup failure is a FAIL
            cleanup["errors"].append(f"runtime_identity: {type(exc).__name__}: {exc}")
        os.environ.pop(HISTORY_DIR_ENV, None)
        result["cleanup"] = cleanup
        if cleanup["errors"]:
            result["verdict"] = "FAIL"
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded BOOT-023 acceptance: one NOOP release through the "
            "sanctioned path, then read back the release history, the durable "
            "registry digests and the rollback_compatible refusal."
        )
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument(
        "--noop-config",
        default="",
        help="release config that classifies as NOOP against the live site "
        "(boot020_release_candidates.py configs -> noop.json)",
    )
    value.add_argument("--predecessor-evidence", default="")
    return value


CASE = CaseRunner(
    case_id=CASE_ID,
    confirmation=CONFIRMATION,
    parser=parser,
    configure=configure,
    read_only_preflight=read_only_preflight,
    plan_details=plan_details,
    execute_case=execute_case,
)


def main() -> int:
    return run_standard_case(CASE)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
