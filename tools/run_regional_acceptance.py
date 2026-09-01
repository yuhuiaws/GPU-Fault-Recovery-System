"""Run the complete regional acceptance inventory without automatic repair."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __package__:
    from .case_scheduler import (
        ExecutionPolicy,
        ResourceLock,
        run_scheduled_cases,
    )
    from .codex_acceptance import (
        CaseAnalysis,
        CodexAcceptanceBackend,
        CodexAcceptanceError,
        EvidenceObservation,
        MandatoryOrderEdge,
    )
    from .regional_acceptance_plan import (
        EnvironmentMode,
        ExecutorKind,
        PlanMode,
        RegionalAcceptanceCase,
        RegionalAcceptancePlan,
        compile_regional_acceptance_plan,
    )
    from .run_fault_test_cases import run_case
else:
    from case_scheduler import ExecutionPolicy, ResourceLock, run_scheduled_cases
    from codex_acceptance import (
        CaseAnalysis,
        CodexAcceptanceBackend,
        CodexAcceptanceError,
        EvidenceObservation,
        MandatoryOrderEdge,
    )
    from regional_acceptance_plan import (
        EnvironmentMode,
        ExecutorKind,
        PlanMode,
        RegionalAcceptanceCase,
        RegionalAcceptancePlan,
        compile_regional_acceptance_plan,
    )
    from run_fault_test_cases import run_case

DEFAULT_REPORT_DIR = ROOT / "artifacts" / "fault"
DEFAULT_ORDER = ROOT / "testcases" / "regional-execution-order.yaml"
DEFAULT_CATALOG = ROOT / "testcases" / "fault-scenarios.yaml"
DEFAULT_WORKERS = 4
DEFAULT_CODEX_BATCH_SIZE = 3
DEFAULT_REVIEW_WORKERS = 3
MAX_CAPTURED_OUTPUT_CHARS = 256 * 1024
_SAFE_ENVIRONMENT_NAMES = (
    "HOME",
    "PATH",
    "PYTHONPATH",
    "PYTHONPYCACHEPREFIX",
    "TMPDIR",
    "TMP",
    "TEMP",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
)
_INHERITED_ENVIRONMENT_REQUIREMENTS = {
    "GF-REGIONAL-CAP-005": ("GPU_FAULT_STORE_URL",),
}
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)(?P<prefix>"
    r"(?:authorization\s*:\s*bearer|"
    r"[A-Z0-9_.-]*(?:TOKEN|PASSWORD|PASSWD|SECRET|PRIVATE_KEY|"
    r"API_KEY|ACCESS_KEY|CREDENTIAL|DSN|DATABASE_URL|STORE_URL)"
    r"[A-Z0-9_.-]*[\"']?\s*[=:]\s*[\"']?)"
    r")(?P<value>[^\s,;\"']+)"
)
_URL_CREDENTIAL = re.compile(r"(?i)([a-z][a-z0-9+.-]*://)[^/@\s]+@")
_AWS_ACCESS_KEY = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_local_environment(
    source: Mapping[str, str] | None = None,
    *,
    extra_names: Sequence[str] = (),
) -> dict[str, str]:
    """Build a minimal child environment without ambient credentials."""

    source_environment = os.environ if source is None else source
    environment = {
        name: value
        for name in _SAFE_ENVIRONMENT_NAMES
        if (value := source_environment.get(name)) is not None
    }
    environment.setdefault("PATH", os.defpath)
    environment["GPU_FAULT_STORE_URL"] = ""
    environment["GPU_FAULT_TEST_POSTGRES_URL"] = ""
    for name in extra_names:
        value = source_environment.get(name)
        if value is None or not value:
            raise ValueError(
                f"required inherited environment variable is unset: {name}"
            )
        environment[name] = value
    return environment


def redact_text(value: str) -> str:
    """Redact common credential shapes before evidence is persisted."""

    redacted = _SENSITIVE_ASSIGNMENT.sub(
        lambda match: f"{match.group('prefix')}<redacted>",
        value,
    )
    redacted = _URL_CREDENTIAL.sub(r"\1<redacted>@", redacted)
    redacted = _AWS_ACCESS_KEY.sub("<redacted-aws-access-key>", redacted)
    if len(redacted) > MAX_CAPTURED_OUTPUT_CHARS:
        return (
            redacted[:MAX_CAPTURED_OUTPUT_CHARS]
            + "\n<output truncated by regional acceptance runner>"
        )
    return redacted


def sanitize_value(value: object) -> object:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): sanitize_value(item) for key, item in value.items()}
    return value


def secure_write_json(path: Path, value: Mapping[str, object]) -> None:
    """Atomically write a private report with mode 0600."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                sanitize_value(value),
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _case_payload(case: RegionalAcceptanceCase) -> dict[str, Any]:
    return {
        "id": case.id,
        "title": case.title,
        "category": case.category,
        "level": case.level,
        "risk": case.risk,
        "problem": case.problem,
        "injection": case.injection,
        "expected": list(case.expected),
        "automation": case.automation.value,
        "procedure": case.procedure,
        "related_pytest": case.related_pytest,
        "gate": case.gate,
        "phase": (
            f"{case.phase_sequence}: {case.phase_name}"
            if case.phase_sequence is not None
            else "DO_NOT_RUN"
        ),
    }


def _scheduler_policy(
    case: RegionalAcceptanceCase,
    mode: PlanMode,
) -> ExecutionPolicy:
    if mode is not PlanMode.FORMAL and (
        case.executor in {ExecutorKind.HUMAN, ExecutorKind.DO_NOT_RUN} or case.blocked
    ):
        return ExecutionPolicy(
            parallel_safe=True,
            locks=(ResourceLock("report-only", "shared"),),
            depends_on=case.execution.depends_on,
            failure_scope="case",
            environment="isolated",
        )
    return ExecutionPolicy(
        parallel_safe=case.execution.parallel_safe,
        locks=tuple(
            ResourceLock(lock.resource, lock.mode.value)
            for lock in case.execution.locks
        ),
        depends_on=case.execution.depends_on,
        failure_scope=case.execution.failure_scope.value,
        environment=case.execution.environment.value,
    )


def _base_result(
    case: RegionalAcceptanceCase,
    *,
    status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "id": case.id,
        "title": case.title,
        "category": case.category,
        "level": case.level,
        "risk": case.risk,
        "problem": case.problem,
        "injection": case.injection,
        "expected": list(case.expected),
        "status": status,
        "execution_status": status,
        "reason": reason,
        "procedure": case.procedure,
        "duration_seconds": 0.0,
        "regional": {
            "ordinal": case.ordinal,
            "phase_sequence": case.phase_sequence,
            "phase_name": case.phase_name,
            "maintenance_window": case.maintenance_window,
            "executor": case.executor.value,
            "local_proxy": case.local_proxy,
            "read_only": case.execution.read_only,
            "repair_allowed": case.execution.repair_allowed,
        },
    }


def _command_environment(
    case: RegionalAcceptanceCase,
    *,
    approved_inherited_cases: set[str],
) -> tuple[dict[str, str] | None, str | None]:
    if case.execution.environment is EnvironmentMode.ISOLATED:
        return build_local_environment(), None
    required = _INHERITED_ENVIRONMENT_REQUIREMENTS.get(case.id)
    if required is None:
        return None, (
            "BLOCKED_PREREQUISITE: inherited command has no reviewed "
            "environment allowlist"
        )
    if case.id not in approved_inherited_cases:
        return None, (
            "BLOCKED_PREREQUISITE: inherited command requires explicit "
            f"--allow-inherited-command {case.id}"
        )
    try:
        return build_local_environment(extra_names=required), None
    except ValueError as exc:
        return None, f"BLOCKED_PREREQUISITE: {exc}"


def _run_automated_case(
    case: RegionalAcceptanceCase,
    *,
    approved_inherited_cases: set[str],
) -> dict[str, Any]:
    environment, blocked_reason = _command_environment(
        case,
        approved_inherited_cases=approved_inherited_cases,
    )
    if blocked_reason is not None:
        return _base_result(case, status="BLOCKED", reason=blocked_reason)
    if environment is None:
        return _base_result(
            case,
            status="FAIL",
            reason="executor environment was not constructed",
        )
    payload = _case_payload(case)
    if case.executor is ExecutorKind.PYTEST:
        if len(case.pytest_nodeids) != 1:
            return _base_result(
                case,
                status="FAIL",
                reason="pytest executor requires exactly one nodeid",
            )
        payload["automation"] = "pytest"
        payload["pytest_nodeid"] = case.pytest_nodeids[0]
    elif case.executor is ExecutorKind.COMMAND:
        if not case.command:
            return _base_result(
                case,
                status="FAIL",
                reason="command executor has no command",
            )
        payload["automation"] = "command"
        payload["command"] = list(case.command)
    else:
        return _base_result(
            case,
            status="FAIL",
            reason=f"unsupported automated executor: {case.executor.value}",
        )

    raw_value = sanitize_value(run_case(payload, environment=environment))
    if not isinstance(raw_value, Mapping):
        return _base_result(
            case,
            status="FAIL",
            reason="automated executor returned a non-mapping result",
        )
    raw = dict(raw_value)
    raw["execution_status"] = raw["status"]
    raw["regional"] = _base_result(
        case,
        status="NOT_RUN",
        reason="metadata",
    )["regional"]
    if not case.local_proxy:
        return raw

    proxy_status = str(raw["status"])
    result = _base_result(
        case,
        status="NOT_RUN" if proxy_status == "PASS" else "FAIL",
        reason=(
            "manual procedure was not executed; associated local check passed"
            if proxy_status == "PASS"
            else "manual procedure was not executed; associated local check failed"
        ),
    )
    result["execution_status"] = "NOT_RUN"
    result["proxy_check"] = {
        "status": proxy_status,
        "executor": raw.get("executor"),
        "duration_seconds": raw.get("duration_seconds"),
        "output": raw.get("output", ""),
        "trace_error": raw.get("trace_error"),
    }
    return result


def _agent_result(
    case: RegionalAcceptanceCase,
    analysis: CaseAnalysis,
    plan: RegionalAcceptancePlan,
) -> dict[str, Any]:
    status_map = {
        "PASS": "NOT_RUN",
        "FAIL": "FAIL",
        "BLOCKED": "BLOCKED",
        "NEEDS_HUMAN": "NOT_RUN",
    }
    derived_dependents = list(plan.blocked_case_ids({case.id}))
    result = _base_result(
        case,
        status=status_map[analysis.status],
        reason=(
            "manual procedure was not executed; read-only Codex analysis "
            f"returned {analysis.status}"
        ),
    )
    result["execution_status"] = "NOT_RUN"
    result["analysis_status"] = analysis.status
    assessment = analysis.as_dict()
    assessment["reported_affected_dependents_untrusted"] = assessment.pop(
        "affected_dependents"
    )
    assessment["affected_dependents"] = derived_dependents
    result["agent_assessment"] = assessment
    return dict(sanitize_value(result))


def _agent_blocked_result(
    case: RegionalAcceptanceCase,
    error: BaseException,
) -> dict[str, Any]:
    result = _base_result(
        case,
        status="BLOCKED",
        reason=(
            "read-only Codex analysis could not complete: "
            f"{type(error).__name__}: {error}"
        ),
    )
    result["execution_status"] = "NOT_RUN"
    result["analysis_status"] = "BLOCKED"
    return dict(sanitize_value(result))


def _analysis_succeeded(result: Mapping[str, Any]) -> bool:
    if result.get("status") == "PASS":
        return True
    if result.get("analysis_status") == "PASS":
        return True
    proxy = result.get("proxy_check")
    return isinstance(proxy, Mapping) and proxy.get("status") == "PASS"


def _review_agent_results(
    *,
    backend: CodexAcceptanceBackend,
    plan_by_id: Mapping[str, RegionalAcceptanceCase],
    results: list[dict[str, Any]],
    max_workers: int,
) -> None:
    review_targets = [
        result
        for result in results
        if isinstance(result.get("agent_assessment"), Mapping)
    ]
    if not review_targets:
        return

    def review(result: dict[str, Any]) -> tuple[str, dict[str, object]]:
        case_id = str(result["id"])
        case = plan_by_id[case_id]
        assessment = result["agent_assessment"]
        if not isinstance(assessment, Mapping):
            return case_id, {
                "status": "BLOCKED",
                "summary": "agent assessment is not a mapping",
                "findings": [],
                "missing_evidence": [],
                "contradictions": [],
                "human_actions": ["Review the malformed assessment manually."],
            }
        raw_evidence = assessment.get("evidence")
        evidence: list[EvidenceObservation] = []
        if isinstance(raw_evidence, list):
            for item in raw_evidence:
                if not isinstance(item, Mapping):
                    continue
                source = item.get("source")
                observation = item.get("observation")
                if isinstance(source, str) and isinstance(observation, str):
                    evidence.append(EvidenceObservation(source, observation))
        try:
            outcome = backend.review_evidence(_case_payload(case), evidence)
            return case_id, outcome.as_dict()
        except CodexAcceptanceError as exc:
            return case_id, {
                "status": "BLOCKED",
                "summary": (
                    "independent evidence review could not complete: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "findings": [],
                "missing_evidence": [],
                "contradictions": [],
                "human_actions": ["Review the case evidence manually."],
            }

    by_id = {str(result["id"]): result for result in review_targets}
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="acceptance-review",
    ) as pool:
        futures = {pool.submit(review, result) for result in review_targets}
        for future in as_completed(futures):
            case_id, review_result = future.result()
            by_id[case_id]["independent_review"] = sanitize_value(review_result)


def execute_plan(
    plan: RegionalAcceptancePlan,
    *,
    workers: int,
    codex_batch_size: int,
    backend: CodexAcceptanceBackend | None,
    review_with_codex: bool,
    review_workers: int,
    approved_inherited_cases: set[str],
) -> list[dict[str, Any]]:
    plan_by_id = {case.id: case for case in plan.cases}
    scheduler_cases = [_case_payload(case) for case in plan.cases]

    def policy_for(payload: dict[str, Any]) -> ExecutionPolicy:
        return _scheduler_policy(plan_by_id[str(payload["id"])], plan.mode)

    def execute(
        payload: dict[str, Any],
        _policy: ExecutionPolicy,
    ) -> dict[str, Any]:
        case = plan_by_id[str(payload["id"])]
        if case.executor is ExecutorKind.DO_NOT_RUN:
            return _base_result(
                case,
                status="NOT_RUN",
                reason=case.do_not_run_reason or "case is DO_NOT_RUN",
            )
        if case.executor is ExecutorKind.HUMAN or case.blocked:
            return _base_result(
                case,
                status="NOT_RUN",
                reason=case.blocked_reason or "manual procedure was not executed",
            )
        if case.executor in {ExecutorKind.PYTEST, ExecutorKind.COMMAND}:
            return _run_automated_case(
                case,
                approved_inherited_cases=approved_inherited_cases,
            )
        if case.executor is ExecutorKind.CODEX_MANUAL:
            raise RuntimeError("codex-manual case bypassed the batch executor")
        return _base_result(
            case,
            status="FAIL",
            reason=f"unsupported executor: {case.executor.value}",
        )

    def batch_key_for(
        payload: dict[str, Any],
        _policy: ExecutionPolicy,
    ) -> str | None:
        case = plan_by_id[str(payload["id"])]
        return (
            "codex-manual"
            if case.executor is ExecutorKind.CODEX_MANUAL and not case.blocked
            else None
        )

    def execute_batch(
        payloads: Sequence[dict[str, Any]],
        _policies: Sequence[ExecutionPolicy],
    ) -> Mapping[str, dict[str, Any]]:
        batch_cases = [plan_by_id[str(payload["id"])] for payload in payloads]
        if backend is None:
            error = RuntimeError(
                "codex-manual executor requires --manual-backend codex-read-only"
            )
            return {case.id: _agent_blocked_result(case, error) for case in batch_cases}
        try:
            analyses = backend.analyze_manual_cases(
                [_case_payload(case) for case in batch_cases]
            )
        except CodexAcceptanceError as exc:
            return {case.id: _agent_blocked_result(case, exc) for case in batch_cases}
        return {
            case.id: _agent_result(case, analyses[case.id], plan)
            for case in batch_cases
        }

    results = run_scheduled_cases(
        scheduler_cases,
        policy_for=policy_for,
        execute=execute,
        batch_key_for=batch_key_for,
        execute_batch=execute_batch,
        max_batch_size=codex_batch_size,
        max_workers=workers,
        collect_all=plan.collect_all,
        is_successful=_analysis_succeeded,
    )
    if review_with_codex:
        if backend is None:
            raise ValueError("--review-with-codex requires a Codex manual backend")
        _review_agent_results(
            backend=backend,
            plan_by_id=plan_by_id,
            results=results,
            max_workers=review_workers,
        )
    return results


def _report_verdict(results: Sequence[Mapping[str, Any]]) -> str:
    statuses = {str(result.get("status")) for result in results}
    if "FAIL" in statuses:
        return "FAIL"
    if statuses.intersection({"BLOCKED", "NOT_RUN"}):
        return "PARTIAL"
    return "PASS"


def build_report(
    plan: RegionalAcceptancePlan,
    results: list[dict[str, Any]],
    *,
    started: float,
    workers: int,
    codex_batch_size: int,
    reviewed_plan: Path | None,
    order_path: Path,
    catalog_path: Path,
) -> dict[str, object]:
    status_counts = Counter(str(result["status"]) for result in results)
    executor_counts = Counter(
        str(result["regional"]["executor"])
        for result in results
        if isinstance(result.get("regional"), Mapping)
    )
    analysis_counts = Counter(
        str(result["analysis_status"])
        for result in results
        if "analysis_status" in result
    )
    proxy_counts = Counter(
        str(proxy["status"])
        for result in results
        if isinstance((proxy := result.get("proxy_check")), Mapping)
    )
    plan_payload = plan.as_dict()
    plan_digest = hashlib.sha256(
        json.dumps(
            plan_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    limitations = [
        "Test workers and Codex agents were read-only and never attempted repairs.",
        "Codex or proxy PASS does not promote a manual/live case to execution PASS.",
        "Only a reviewed deterministic DAG controls dependency blocking.",
        "This report does not authorize or execute live, destructive, AWS, "
        "Kubernetes, GPU, node, service, workload, or network mutations.",
    ]
    if plan.mode is not PlanMode.FORMAL:
        limitations.append(
            "collect-all and local-preacceptance modes are diagnostic, not formal "
            "phase-ordered regional acceptance."
        )
    return {
        "schema_version": 2,
        "report_type": "regional-acceptance-run",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "verdict": _report_verdict(results),
        "mode": plan.mode.value,
        "limitations": limitations,
        "sources": {
            "order": str(order_path),
            "catalog": str(catalog_path),
            "reviewed_plan": str(reviewed_plan) if reviewed_plan else None,
            "plan_sha256": plan_digest,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "workers": workers,
            "codex_batch_size": codex_batch_size,
        },
        "summary": {
            "total": len(results),
            "case_status": dict(sorted(status_counts.items())),
            "executor": dict(sorted(executor_counts.items())),
            "analysis_status": dict(sorted(analysis_counts.items())),
            "proxy_status": dict(sorted(proxy_counts.items())),
            "wall_duration_seconds": round(time.monotonic() - started, 3),
        },
        "plan": plan_payload,
        "results": results,
    }


def propose_dependencies(
    *,
    backend: CodexAcceptanceBackend,
    order_path: Path,
    catalog_path: Path,
) -> dict[str, object]:
    plan = compile_regional_acceptance_plan(
        mode=PlanMode.COLLECT_ALL,
        order_path=order_path,
        catalog_path=catalog_path,
    )
    runnable = [
        case for case in plan.cases if case.executor is not ExecutorKind.DO_NOT_RUN
    ]
    cases = [
        {
            "id": case.id,
            "title": case.title,
            "phase": f"{case.phase_sequence}: {case.phase_name}",
            "risk": case.risk,
            "summary": case.problem,
            "automation": case.automation.value,
            "expected": list(case.expected),
            "procedure": case.procedure,
            "category": case.category,
            "level": case.level,
        }
        for case in runnable
    ]
    mandatory_edges = [
        MandatoryOrderEdge(before=dependency, after=case.id)
        for case in runnable
        for dependency in case.mandatory_depends_on
    ]
    proposal = backend.propose_dependencies(cases, mandatory_edges)
    return {
        "schema_version": 1,
        "report_type": "regional-dependency-proposal",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trusted": False,
        "limitations": [
            "This proposal is advisory and must not be passed directly to the scheduler.",
            "A human-reviewed plan with current source digests is required before use.",
        ],
        "sources": {
            "order": str(order_path),
            "order_sha256": hashlib.sha256(order_path.read_bytes()).hexdigest(),
            "catalog": str(catalog_path),
            "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        },
        "proposal": proposal.as_dict(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or analyze the complete regional acceptance inventory without "
            "automatic repair."
        )
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in PlanMode],
        default=PlanMode.LOCAL_PREACCEPTANCE.value,
    )
    parser.add_argument("--order", type=Path, default=DEFAULT_ORDER)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--reviewed-plan", type=Path)
    parser.add_argument("--workers", type=_positive_integer, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--codex-batch-size",
        type=_positive_integer,
        default=DEFAULT_CODEX_BATCH_SIZE,
    )
    parser.add_argument(
        "--manual-backend",
        choices=("none", "codex-read-only"),
        default="none",
    )
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument(
        "--codex-timeout-seconds",
        type=_positive_float,
        default=900.0,
    )
    parser.add_argument("--review-with-codex", action="store_true")
    parser.add_argument(
        "--review-workers",
        type=_positive_integer,
        default=DEFAULT_REVIEW_WORKERS,
    )
    parser.add_argument(
        "--allow-inherited-command",
        action="append",
        default=[],
        dest="approved_inherited_cases",
    )
    parser.add_argument("--list-plan", action="store_true")
    parser.add_argument("--write-plan", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--propose-dependencies", action="store_true")
    parser.add_argument("--proposal-output", type=Path)
    return parser


def _backend_from_args(args: argparse.Namespace) -> CodexAcceptanceBackend | None:
    needs_backend = (
        args.manual_backend == "codex-read-only"
        or args.review_with_codex
        or args.propose_dependencies
    )
    if not needs_backend:
        return None
    return CodexAcceptanceBackend(
        ROOT,
        codex_binary=str(args.codex_binary),
        timeout_seconds=float(args.codex_timeout_seconds),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    backend = _backend_from_args(args)
    if args.propose_dependencies:
        if backend is None:
            raise RuntimeError("dependency proposal requires a Codex backend")
        output = args.proposal_output
        if output is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output = DEFAULT_REPORT_DIR / f"regional-dependency-proposal-{stamp}.json"
        proposal = propose_dependencies(
            backend=backend,
            order_path=args.order,
            catalog_path=args.catalog,
        )
        secure_write_json(output, proposal)
        print(f"PROPOSAL {output}")
        print("TRUSTED false")
        return 0

    plan = compile_regional_acceptance_plan(
        mode=args.mode,
        order_path=args.order,
        catalog_path=args.catalog,
        override_path=args.reviewed_plan,
    )
    if args.write_plan is not None:
        secure_write_json(args.write_plan, plan.as_dict())
    if args.list_plan:
        for case in plan.cases:
            print(
                f"{case.ordinal + 1:03d}\t{case.id}\t{case.executor.value}\t"
                f"{case.risk}\t{','.join(case.execution.depends_on) or '-'}"
            )
        return 0

    codex_cases = [
        case for case in plan.cases if case.executor is ExecutorKind.CODEX_MANUAL
    ]
    if codex_cases and backend is None:
        print(
            "ERROR: reviewed plan selects codex-manual cases but "
            "--manual-backend codex-read-only is not enabled",
            file=sys.stderr,
        )
        return 2
    started = time.monotonic()
    results = execute_plan(
        plan,
        workers=args.workers,
        codex_batch_size=args.codex_batch_size,
        backend=backend,
        review_with_codex=args.review_with_codex,
        review_workers=args.review_workers,
        approved_inherited_cases=set(args.approved_inherited_cases),
    )
    report = build_report(
        plan,
        results,
        started=started,
        workers=args.workers,
        codex_batch_size=args.codex_batch_size,
        reviewed_plan=args.reviewed_plan,
        order_path=args.order,
        catalog_path=args.catalog,
    )
    report_path = args.report
    if report_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = DEFAULT_REPORT_DIR / f"regional-acceptance-{stamp}.json"
    secure_write_json(report_path, report)
    summary = report["summary"]
    if not isinstance(summary, Mapping):
        raise RuntimeError("regional report summary is not a mapping")
    print(f"REPORT {report_path}")
    print(
        "SUMMARY "
        f"verdict={report['verdict']} "
        f"total={summary['total']} "
        f"case_status={json.dumps(summary['case_status'], sort_keys=True)}"
    )
    case_status = summary["case_status"]
    if not isinstance(case_status, Mapping):
        raise RuntimeError("regional report case_status is not a mapping")
    return 1 if case_status.get("FAIL", 0) or case_status.get("BLOCKED", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
