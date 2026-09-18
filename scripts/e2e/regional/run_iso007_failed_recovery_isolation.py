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

from scripts.e2e.regional import run_e2e002_multicluster_fault as shared  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import write_json_atomic  # noqa: E402
from scripts.e2e.regional.live_driver_guard import CaseRunner, run_standard_case  # noqa: E402
from scripts.e2e.regional.regional_live_fixture import run_case_main  # noqa: E402
from scripts.e2e.regional.run_workload_acceptance import gpu_nodes_clean  # noqa: E402

CASE_ID = "GF-REGIONAL-ISO-007"
PREDECESSOR_CASE_ID = "GF-REGIONAL-E2E-002"
CONFIRMATION = "ISO007_A_BUDGET_DENIED_B_RECOVERS"


def parser() -> argparse.ArgumentParser:
    return shared.parser(
        confirmation=CONFIRMATION,
        description="Prove A's zero-budget recovery failure does not block B's recovery.",
    )


def configure(arguments: argparse.Namespace) -> shared.Settings:
    return shared.configure(
        arguments, case_id=CASE_ID, predecessor_case_id=PREDECESSOR_CASE_ID
    )


def read_only_preflight(
    settings: shared.Settings, case_dir: Path, *, reuse_focused_tests: bool = False
) -> dict[str, Any]:
    result = shared.read_only_preflight(
        settings,
        case_dir,
        reuse_focused_tests=reuse_focused_tests,
        predecessor_case_id=PREDECESSOR_CASE_ID,
    )
    try:
        predecessor = json.loads(settings.predecessor_path.read_text(encoding="utf-8"))
        pair = predecessor.get("cluster_ids")
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or set(pair)
            != {
                settings.multi.cluster_a.cluster_id,
                settings.multi.cluster_b.cluster_id,
            }
        ):
            result["errors"].append(
                "E2E-002 predecessor does not bind this cluster pair"
            )
    except (OSError, TypeError, ValueError, AttributeError):
        result["errors"].append("E2E-002 cluster-pair evidence is unreadable")
    for key in ("nodes_a", "nodes_b"):
        nodes = result.get(key) or []
        if not gpu_nodes_clean(nodes) or any(not node.get("uid") for node in nodes):
            result["errors"].append(f"{key} identity/readiness is not proven")
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def plan_details(
    settings: shared.Settings, preflight: dict[str, Any]
) -> dict[str, Any]:
    result = shared.plan_details(settings, preflight)
    result["mutation"] = (
        "create run-owned same-identity workloads with A restart budget 0 and B "
        "restart budget 1, then replay software XID11 concurrently; no online "
        "allowlist, policy, network, node or provider mutation"
    )
    result["expected_outcomes"] = {
        "cluster_a": "terminal FAILED / RESTART_BUDGET_EXHAUSTED / no remote restart",
        "cluster_b": "terminal SUCCEEDED / one local workload restart",
    }
    result["stop_conditions"] = [
        "E2E-002 predecessor is not PASS for this release and cluster pair",
        *result["stop_conditions"][1:],
        "A fails for any reason other than its explicit zero restart budget",
        "either workload's observed budget differs from its submitted budget",
        "A issues a restart command or either GPU node inventory changes",
    ]
    result["validation_limitations"] = [
        "Software event replay and an explicit workload budget refusal, not a "
        "hardware failure, executor crash or network partition."
    ]
    return result


def execute_case(
    settings: shared.Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    return shared.execute_case(
        settings,
        run_dir,
        attempt,
        maintenance_window_end,
        case_id=CASE_ID,
        predecessor_case_id=PREDECESSOR_CASE_ID,
        expect_a_budget_denial=True,
        preflight_reader=read_only_preflight,
    )


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
