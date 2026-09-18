"""Approval-bound, two-process full-uninstall acceptance lifecycle."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpu_fault.admin.execution import deployment_deadline
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.acceptance_supervision import record_supervision_loss
from scripts.e2e.regional.boot032_contract import (
    CASE_ID,
    CONFIRMATION,
    RESTART_EXIT,
    Settings,
    UninstallCaseError,
    cluster_specs,
    mapping,
    read_document,
    require,
)
from scripts.e2e.regional.boot032_journal import native_state, read_case, save_case
from scripts.e2e.regional.boot032_native import (
    NativeBackend,
    RestartRequired,
    checked_locks,
)
from scripts.e2e.regional.live_driver_guard import (
    authorize_execution,
    details_sha256,
    source_digest,
)
from scripts.e2e.regional.regional_case_contract import predecessor_path
from scripts.e2e.regional.regional_live_fixture import predecessor_evidence


def process_identity() -> dict[str, Any]:
    # The kernel start tick distinguishes PID reuse within the same boot.
    boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    fields = (
        Path("/proc/self/stat").read_text(encoding="ascii").rsplit(")", 1)[-1].split()
    )
    require(
        bool(boot) and len(fields) > 19 and fields[19].isdigit(),
        "local process incarnation cannot be verified",
    )
    return {"pid": os.getpid(), "boot_id": boot, "start_ticks": fields[19]}


def predecessor(settings: Settings, binding: dict[str, Any]) -> dict[str, Any]:
    case_id, path = predecessor_path(settings.run_dir, CASE_ID, "")
    if case_id is None or path is None:
        return {"valid": True, "verdict": "NOT_REQUIRED"}
    runtime = binding["protected"]["runtime"]
    cluster_id = settings.protected_cluster_id
    members = {
        spec["cluster_id"]
        for spec in binding["inputs"]["protected"]["clusters"]
        if spec["plane"] == "gpu"
    }
    require(
        binding["inputs"]["protected_cluster_id"] == cluster_id
        and cluster_id in members
        and cluster_id
        in {spec["cluster_id"] for spec in cluster_specs(settings.protected)[1:]},
        "predecessor cluster is not the approved protected-site member",
    )
    selected = mapping(
        runtime.get(cluster_id), "selected protected runtime is unavailable"
    )
    release_state = mapping(
        selected.get("release_state"), "protected release state is unavailable"
    )
    release_id = release_state.get("release_id")
    if not isinstance(release_id, str) or not release_id.strip():
        raise UninstallCaseError("selected protected release identity is unavailable")
    return predecessor_evidence(
        path,
        case_id,
        release_id=release_id,
        cluster_id=cluster_id,
    )


def read_only_preflight(settings: Settings, case_dir: Path) -> dict[str, Any]:
    try:
        with deployment_deadline(
            "BOOT-032 read-only preflight", 900, recovery_seconds=0
        ):
            backend = NativeBackend(settings)
            journal = read_case(case_dir / "boot032-state.json")
            if journal is None:
                binding = backend.initial()
                binding["predecessor"] = predecessor(settings, binding)
            else:
                require(
                    journal["phase"] != "BLOCKED",
                    "case requires independent supervision recovery",
                )
                require(
                    journal["source_digest"] == source_digest(),
                    "case source changed after teardown began",
                )
                binding = journal["binding"]
                require(
                    journal["binding_sha256"] == details_sha256(binding),
                    "case approval binding changed",
                )
                backend.check(binding)
                require(
                    predecessor(settings, binding) == binding["predecessor"],
                    "predecessor evidence changed",
                )
            require(
                binding["predecessor"].get("valid") is True,
                "formal predecessor is not PASS",
            )
            return {"errors": [], "binding": binding}
    except Exception as error:
        return {
            "errors": [f"uninstall preflight refused: {type(error).__name__}"],
            "binding": None,
        }


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "destructive",
        "binding": preflight["binding"],
        "sacrificial_state_dir": str(settings.target.source.parent),
        "protected_accepted_site": str(settings.protected.source),
        "protected_cluster_id": settings.protected_cluster_id,
        "native_retirement": {
            "cpu_disposition": "delete",
            "final_snapshot_policy": "retain",
            "reset_database": False,
            "gpu_clusters": "preserve",
        },
        "procedure": [
            "checked native teardown through verified Kubernetes cleanup",
            "exit with RESTART_REQUIRED and preserve every native journal",
            "fresh process resumes the same approved retirement",
            "independent full-resource and protected-site verification",
        ],
        "stop_conditions": [
            "missing sacrificial provisioning provenance or fixture-purpose tags",
            "site, source, context, ownership or protected-site drift",
            "unknown resource read or unproved process supervision",
            "missing native checkpoint, original incarnation or final readback",
            "approved maintenance window ended",
        ],
        "recovery": (
            "retirement is irreversible; preserve journals and resume the identical "
            "scope under approval, never repair by retargeting or deleting journals"
        ),
    }


def case_result(
    settings: Settings,
    journal: dict[str, Any],
    *,
    verdict: str,
    status: str,
    checks: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    runtime = journal["binding"]["target"]["runtime"]
    cluster_id = sorted(runtime)[0]
    value = {
        "schema_version": 2,
        "report_type": "fault-acceptance",
        "case_id": CASE_ID,
        "verdict": verdict,
        "status": status,
        "release_id": runtime[cluster_id]["release_state"]["release_id"],
        "cluster_id": cluster_id,
        "protected_cluster_id": settings.protected_cluster_id,
        "fixture_id": settings.fixture_id,
        "binding_sha256": journal["binding_sha256"],
        "native_state_dir": str(settings.native_dir),
        "checks": checks or {},
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Requires a separately provisioned and explicitly tagged sacrificial site.",
            "GPU clusters and registered external resources are preserved.",
            "CPU Kubernetes absence is proved by prior native cleanup evidence and AWS CPU deletion.",
            "The controlled pause does not assert that asynchronous cloud deletions have stopped.",
        ],
    }
    if error is not None:
        value["error_type"] = type(error).__name__
        if isinstance(error, UninstallCaseError):
            value["error"] = str(error)
    write_json_atomic(settings.case_dir / f"{CASE_ID}.json", value)
    return value


def execute_case(
    settings: Settings, run_dir: Path, attempt: int, maintenance_window_end: datetime
) -> int:
    require(
        run_dir == settings.run_dir and attempt > 0, "case invocation scope is invalid"
    )
    if settings.arguments is None:
        raise UninstallCaseError("checked runner approval arguments are required")
    plan = read_document(settings.case_dir / "plan.json")
    binding = plan["details"]["binding"]
    authorized = authorize_execution(
        settings.arguments,
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        environment=settings.environment(),
        details=plan_details(settings, {"binding": binding}),
    )
    deadline = min(authorized, maintenance_window_end)
    write_json_atomic(
        settings.case_dir / f"{CASE_ID}.json",
        {
            "schema_version": 2,
            "report_type": "fault-acceptance",
            "case_id": CASE_ID,
            "attempt": attempt,
            "verdict": "FAIL",
            "status": "PREFLIGHT",
        },
    )
    path = settings.case_dir / "boot032-state.json"
    with checked_locks(settings, deadline):
        backend = NativeBackend(settings)
        backend.check(binding)
        require(
            predecessor(settings, binding) == binding["predecessor"]
            and binding["predecessor"].get("valid") is True,
            "formal predecessor evidence changed",
        )
        journal = read_case(path)
        if journal is None:
            require(
                not settings.native_dir.exists(),
                "unowned native uninstall journal exists",
            )
            journal = {
                "schema_version": 1,
                "case_id": CASE_ID,
                "binding": binding,
                "binding_sha256": details_sha256(binding),
                "source_digest": plan["source_digest"],
                "events": [],
                "pause": None,
            }
            save_case(path, journal, "STARTED")
        require(
            journal["binding"] == binding
            and journal["source_digest"] == plan["source_digest"]
            and journal["phase"] != "BLOCKED",
            "case journal no longer matches its approval",
        )
        return continue_case(settings, backend, path, journal)


def continue_case(
    settings: Settings, backend: NativeBackend, path: Path, journal: dict[str, Any]
) -> int:
    current_process = process_identity()
    try:
        native = native_state(settings)
        completed = native is not None and native["phase"] == "COMPLETED"
        if (
            journal["phase"] == "COMPLETED"
            or journal.get("final_registry_sha256")
            or completed
        ):
            require(
                journal.get("pause") is not None,
                "completed native work lacks its restart proof",
            )
            require(
                journal["pause"]["process"] != current_process,
                "completed teardown lacks a fresh-process restart",
            )
            checks = backend.final(journal["binding"], journal["pause"]["proof"])
            checks["completed_replay_read_only"] = True
            save_case(
                path,
                journal,
                "COMPLETED",
                final_registry_sha256=checks["final_registry_sha256"],
            )
            case_result(
                settings, journal, verdict="PASS", status="COMPLETED", checks=checks
            )
            return 0
        pause = journal.get("pause")
        if pause is not None:
            require(
                pause["process"] != current_process,
                "resume requires a fresh process incarnation",
            )
            save_case(path, journal, "RESUMING", resumed_process=current_process)
        try:
            backend.invoke(journal["binding"], pause=pause is None)
        except RestartRequired:
            require(pause is None, "controlled pause repeated after resume")
            proof = backend.paused(journal["binding"])
            save_case(
                path,
                journal,
                "RESTART_REQUIRED",
                pause={
                    "process": current_process,
                    "proof": proof,
                },
            )
            case_result(settings, journal, verdict="NOT_RUN", status="RESTART_REQUIRED")
            return RESTART_EXIT
        if pause is None:
            raise UninstallCaseError(
                "native teardown returned without the required restart checkpoint"
            )
        checks = backend.final(journal["binding"], pause["proof"])
        checks["fresh_process_resume"] = pause["process"] != current_process
        save_case(
            path,
            journal,
            "COMPLETED",
            final_registry_sha256=checks["final_registry_sha256"],
        )
        case_result(
            settings, journal, verdict="PASS", status="COMPLETED", checks=checks
        )
        return 0
    except BaseException as error:
        blocked = isinstance(error, ProcessSupervisionLost)
        if blocked:
            record_supervision_loss()
        save_case(
            path,
            journal,
            "BLOCKED" if blocked else "FAILED",
            error_type=type(error).__name__,
        )
        case_result(
            settings, journal, verdict="FAIL", status=journal["phase"], error=error
        )
        if not isinstance(error, Exception):
            raise
        return 1
