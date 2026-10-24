#!/usr/bin/env python3
"""GF-GENERAL-LIVE: one generic live acceptance runner for the general faults.

One variant per invocation, selected by ``--case GF-GENERAL-LIVE-NNN
--variant <name>``. The variant catalog (``general_live_registry``) binds each
required variant of an implemented category to its injector/verdict module, or
to the explicit reason it is NOT_RUN/BLOCKED in this phase. Plan-only by
default; ``--execute`` requires the case confirmation ``GENERALLIVE<NNN>_EXECUTE``
and, for a dangerous branch, its ``--approve`` tokens.

    run_general_live.py --site-profile P --run-dir R --case GF-GENERAL-LIVE-001 \
        --variant cpu_busy --plan
    run_general_live.py ... --variant cpu_busy --execute \
        --confirm GENERALLIVE001_EXECUTE --maintenance-window-end <UTC>

Every variant arms a time-limited automatic stop on the node before it mutates,
holds an independent restore in a LIFO cleanup stack, and tears every probe Pod
down in ``finally``; a cleanup that cannot prove it is clean fails the variant.
Nothing here posts a prefabricated finding or workflow -- the verdict reads the
product's own records only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.collector_action_guard import action_window  # noqa: E402
from scripts.e2e.regional.general_live_fixture import (  # noqa: E402
    Evidence,
    GeneralLiveSettings,
    VariantBlocked,
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
    add_arguments,
    configure,
    read_only_preflight,
    run_identity,
    utc_now,
)
from scripts.e2e.regional.general_live_registry import (  # noqa: E402
    APPROVALS,
    VariantSpec,
    load_registry,
    select,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    authorize_execution,
    build_plan,
    install_site_profile,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError  # noqa: E402
from scripts.e2e.regional.regional_case_contract import case_evidence_path  # noqa: E402
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalLiveFixture,
    install_abort_signals,
    run_case_main,
)

SCHEMA_VERSION = 2


def plan_details(
    spec: VariantSpec, settings: GeneralLiveSettings, preflight: dict[str, Any]
) -> dict[str, Any]:
    return {
        "variant": spec.describe(),
        "risk": {"L": "diagnostic", "M": "isolation", "H": "dangerous"}[spec.risk],
        "target_node": settings.node,
        "peer_node": settings.peer_node or None,
        "approvals_required": list(spec.approvals),
        "approvals_given": list(settings.approvals),
        "approval_meanings": {item: APPROVALS[item] for item in spec.approvals},
        "safety": {
            "automatic_stop": "every node mutation arms a systemd-run deadman first",
            "independent_restore": "LIFO cleanup stack; a failed cleanup fails the variant",
            "window_bounded": "actions refused once the maintenance window cannot fit them",
            "reads_product_records_only": True,
        },
        "preflight_errors": list(preflight.get("errors") or []),
        "planned_verdict": preflight.get("planned_verdict"),
    }


def evidence_skeleton(
    spec: VariantSpec, settings: GeneralLiveSettings, attempt: int, identity: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "report_type": "general-fault-live",
        "case_id": spec.case_id,
        "variant": spec.variant,
        "guard_key": spec.key,
        "proof_mode": spec.proof_mode,
        "attempt": attempt,
        "target_node": settings.node,
        "expected_outcome": spec.expected_outcome,
        "product_gap": spec.product_gap or None,
        "verdict": "FAIL",
        "started_at": utc_now().isoformat(),
        **identity,
    }


def not_run(result: dict[str, Any], reason: str, detail: str, status: str) -> dict[str, Any]:
    result["verdict"] = status
    result["reason"] = reason
    result["detail"] = detail
    result["ended_at"] = utc_now().isoformat()
    return result


def execute_variant(
    spec: VariantSpec,
    settings: GeneralLiveSettings,
    run_dir: Path,
    attempt: int,
    deadline: datetime,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    regional = RegionalLiveFixture(settings.regional)
    case_dir = run_dir / "cases" / spec.key
    evidence = Evidence(case_dir)
    identity = regional.evidence_identity()
    result = evidence_skeleton(spec, settings, attempt, identity)
    missing = [item for item in spec.approvals if item not in settings.approvals]
    if missing:
        return not_run(result, "NOT_AUTHORIZED", f"missing approvals {missing}", "NOT_RUN")
    if preflight.get("errors"):
        return not_run(
            result, "BLOCKED", "; ".join(preflight["errors"]), "BLOCKED"
        )
    ctx = VariantContext(
        spec=spec,
        settings=settings,
        regional=regional,
        case_dir=case_dir,
        run_id=run_identity(spec.case_id, spec.variant, attempt, run_dir),
        attempt=attempt,
        deadline=deadline,
        evidence=evidence,
    )
    result["run_id"] = ctx.run_id
    assert spec.execute is not None  # guaranteed by validate_spec for implemented specs
    try:
        with action_window(deadline):
            outcome = spec.execute(ctx)
        result.update(outcome)
        result.setdefault("verdict", "FAIL")
    except VariantBlocked as blocked:
        not_run(result, blocked.reason, blocked.detail, "BLOCKED")
    except VariantNotRun as skipped:
        not_run(result, skipped.reason, skipped.detail, "NOT_RUN")
    except VariantInconclusive as inconclusive:
        not_run(result, "INCONCLUSIVE", str(inconclusive), "INCONCLUSIVE")
    except BaseException as exc:  # noqa: BLE001 - recorded; cleanup still runs
        result["verdict"] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, Exception):
            _teardown(ctx, result)
            write_json_atomic(case_evidence_path(run_dir, spec.key), result)
            raise
    _teardown(ctx, result)
    if result["verdict"] == "PASS" and spec.expected_outcome == "FAIL":
        # The spec marks this variant as exposing a product gap: an effective
        # injection that nonetheless "passes" the chain is the FAIL the case
        # exists to record, not a PASS.
        result["verdict"] = "FAIL"
        result.setdefault("errors", []).append(
            f"effective injection did not expose the expected gap: {spec.product_gap}"
        )
    result["ended_at"] = utc_now().isoformat()
    write_json_atomic(case_evidence_path(run_dir, spec.key), result)
    return result


def _teardown(ctx: VariantContext, result: dict[str, Any]) -> None:
    cleanup = ctx.cleanup.run()
    probe_residuals = {}
    for name, handle in ctx.probes.items():
        try:
            probe_residuals[name] = handle.cleanup()
        except Exception as exc:  # noqa: BLE001 - recorded, verdict downgraded
            probe_residuals[name] = {"cleanup_error": f"{type(exc).__name__}: {exc}"}
    result["cleanup"] = cleanup
    result["probe_residuals"] = probe_residuals
    result["cleanup_pending"] = ctx.cleanup.pending()
    dirty = (
        ctx.cleanup.failed
        or ctx.cleanup.pending()
        or any(
            value is not False
            for residual in probe_residuals.values()
            if isinstance(residual, dict)
            for value in residual.values()
        )
    )
    if dirty and result["verdict"] in {"PASS", "FAIL"}:
        result["verdict"] = "FAIL"
        result.setdefault("errors", []).append("cleanup did not prove the node clean")
    ctx.evidence.write("result", result)


def main() -> int:
    install_site_profile()
    parser = argparse.ArgumentParser(description="Run one GF-GENERAL-LIVE variant.")
    add_arguments(parser)
    arguments = parser.parse_args()
    import os

    os.umask(0o077)
    install_abort_signals()
    registry = load_registry()
    spec = select(registry, str(arguments.case), str(arguments.variant))
    settings = configure(arguments)
    run_dir = Path(arguments.run_dir)
    case_dir = run_dir / "cases" / spec.key
    case_dir.mkdir(parents=True, exist_ok=True)
    regional = RegionalLiveFixture(settings.regional)
    preflight = read_only_preflight(settings, spec, regional)
    write_json_atomic(case_dir / "preflight.json", preflight)
    confirmation = spec.confirmation
    if not arguments.execute:
        passed = not preflight.get("errors")
        plan = build_plan(
            arguments=arguments,
            preflight_passed=passed,
            run_dir=run_dir,
            case_id=spec.key,
            attempt=arguments.attempt,
            confirmation=confirmation,
            environment=settings.environment(),
            details={**plan_details(spec, settings, preflight), "preflight": preflight},
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if (passed or spec.deferred) else 1
    if arguments.confirm != confirmation:
        raise RegionalFixtureError(f"confirmation must be exactly {confirmation}")
    deadline = authorize_execution(
        arguments,
        case_id=spec.key,
        confirmation=confirmation,
        environment=settings.environment(),
    )
    if spec.deferred:
        reason = spec.deferred.split(":", 1)[0]
        result = evidence_skeleton(
            spec, settings, arguments.attempt, regional.evidence_identity()
        )
        not_run(result, reason, spec.deferred, "NOT_RUN")
        write_json_atomic(case_evidence_path(run_dir, spec.key), result)
        print(json.dumps(result, sort_keys=True, default=str))
        return 0
    result = execute_variant(
        spec, settings, run_dir, arguments.attempt, deadline, preflight
    )
    print(json.dumps(result, sort_keys=True, default=str))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
