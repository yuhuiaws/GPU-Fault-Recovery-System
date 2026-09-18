"""Guarded entrypoint for the physical late-ownership companion proof."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.e2e.regional import run_destr015_parallel_branch_join as normal
from scripts.e2e.regional.late_ownership_live import (
    execute_case,
    read_only_preflight as read_only_preflight,
    require_companion_directory as require_companion_directory,
)
from scripts.e2e.regional.late_ownership_ordinary import require_external_ordinary
from scripts.e2e.regional.live_driver_guard import CaseSurface, run_selected_case
from scripts.e2e.regional.regional_live_fixture import run_case_main

CONFIRMATION = "PHYSICAL_LATE_OWNERSHIP_EXECUTE"
CASES = ("GF-REGIONAL-PREEMPT-033", "GF-REGIONAL-DESTR-015")
SCENARIOS = ("unchanged-owner", "ownership-drift", "late-sibling")


@dataclass(frozen=True)
class Settings:
    base: normal.Settings
    case_id: str
    scenario: str
    ordinary_destr015_evidence: Path

    def __post_init__(self) -> None:
        if self.case_id not in CASES or self.scenario not in SCENARIOS:
            raise ValueError("unsupported physical late-ownership case or scenario")

    @property
    def confirmation(self) -> str:
        return CONFIRMATION

    def environment(self) -> dict[str, str]:
        return {
            **self.base.environment(),
            "GPU_FAULT_ORDINARY_DESTR015_EVIDENCE": str(
                self.ordinary_destr015_evidence
            ),
        }


def parser() -> argparse.ArgumentParser:
    value = normal.parser(confirmation=CONFIRMATION)
    value.description = "Guarded physical late-ownership companion proof; never a model-only LIVE verdict."
    value.add_argument("--case", choices=CASES, required=True)
    value.add_argument("--scenario", choices=SCENARIOS, required=True)
    value.add_argument("--ordinary-destr015-evidence", type=Path, required=True)
    return value


def configure(arguments: argparse.Namespace) -> Settings:
    base = normal.configure(arguments)
    require_companion_directory(arguments.run_dir, base.predecessor_path)
    ordinary_path = arguments.ordinary_destr015_evidence.expanduser().absolute()
    require_external_ordinary(arguments.run_dir, ordinary_path)
    return Settings(base, arguments.case, arguments.scenario, ordinary_path)


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    details = normal.plan_details(settings.base, preflight)
    details.update(
        {
            "subproof": "physical-late-ownership",
            "scenario": settings.scenario,
            "related_cases": list(CASES),
            "promotes_ordinary_case": False,
            "manual_sequence": {
                "canonical_case": "GF-REGIONAL-DESTR-015",
                "preempt033_alias_investigation_only": True,
                "cross_reference_same_mechanism_receipts": list(CASES),
                "after_ordinary_case": "GF-REGIONAL-DESTR-015",
                "ordinary_destr015_evidence": str(settings.ordinary_destr015_evidence),
                "outside_formal_case_slot": True,
                "separate_run_directory_per_variant": True,
                "required_variants": list(SCENARIOS),
                "ordinary_predecessor_case": normal.PREDECESSOR_CASE_ID,
                "companion_result_path": f"cases/{settings.case_id}/late-ownership/{settings.scenario}/result.json",
                "companion_receipts_path": f"cases/{settings.case_id}/late-ownership/{settings.scenario}/receipts.json",
            },
            "mutation": (
                "create one owned two-node workload and an audited CPU workflow holder; "
                "attach calibrated exec witnesses to both Agents; execute actual STOP; "
                "park at the Agent's post-queue/pre-spawn challenge; make the approved "
                "UID-conditional ownership change or admit a real GPU sibling on the "
                "other node; release only into the installed ownership validator; "
                "drain commands, restore and independently verify nodes, then clean "
                "only the run's owned resources. No hardware fault injection, reboot "
                "or provider replacement is authorized."
            ),
            "boundary": "AGENT_PRE_SPAWN",
            "stop_conditions": [
                "any normal preflight, identity, release or predecessor check fails",
                "a deployed Agent does not require the final ownership protocol",
                "a physical witness cannot attach, calibrate or remain continuous",
                "STOP, source GPU-client absence or the late sibling is unproven",
                "UID/owner drift escapes the planned fixture scope",
                "controller loss, stale callback, permit expiry or cleanup failure",
                "any reset exec appears in a refused ownership experiment",
            ],
            "rollback": {
                "node_action_gate_fails_closed_on_caller_loss": True,
                "existing_node_quiesce_failsafe_is_retained": True,
                "no_rescheduling_before_host_baseline_verification": True,
                "uid_conditional_owned_workload_and_probe_cleanup": True,
                "unproven_quiescence_requires_operator_reconciliation": True,
            },
        }
    )
    return details


CASE = CaseSurface(
    parser=parser,
    configure=configure,
    read_only_preflight=read_only_preflight,
    plan_details=plan_details,
    execute_case=execute_case,
)


def main() -> int:
    return run_case_main(lambda: run_selected_case(CASE))
