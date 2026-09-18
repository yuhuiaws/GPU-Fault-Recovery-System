from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gpu_fault.admin.release_state import live_release_state  # noqa: E402
from gpu_fault.admin.site import load_site  # noqa: E402
from scripts.e2e.regional.capacity_acceptance_base import (  # noqa: E402
    DEFAULT_B_LATENCY_FACTOR,
    CapError,
)
from scripts.e2e.regional.capacity_acceptance_cases import (  # noqa: E402
    CapacityAcceptanceCases,
)
from scripts.e2e.regional.capacity_scrape_companion import (  # noqa: E402
    capture_scrape_source,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
    connection_identity,
    install_site_profile,
)
from scripts.e2e.regional.regional_case_contract import (  # noqa: E402
    predecessor_path,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    predecessor_evidence,
)

CASE_IDS = tuple(f"GF-REGIONAL-CAP-{number:03d}" for number in range(1, 5))
CapHarness = CapacityAcceptanceCases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one isolated regional capacity acceptance case."
    )
    add_live_arguments(parser, confirmation="CASE_SPECIFIC_CONFIRMATION")
    parser.add_argument("--case", choices=CASE_IDS, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--predecessor-evidence", default="")
    parser.add_argument(
        "--b-latency-factor",
        type=float,
        default=DEFAULT_B_LATENCY_FACTOR,
        help=(
            "CAP-001: storm-phase cluster B p95 latency may be at most this "
            "multiple of the B-only baseline p95 measured before the storm"
        ),
    )
    return parser.parse_args()


def main() -> int:
    install_site_profile()
    args = parse_args()
    os.umask(0o077)
    if not math.isfinite(args.b_latency_factor) or args.b_latency_factor < 1:
        raise CapError("b_latency_factor must be finite and at least 1.0")
    site = load_site(args.site)
    environment = {
        "GPU_FAULT_SITE_FILE": str(site.source),
        "GPU_FAULT_CAPACITY_CASE": args.case,
        "GPU_FAULT_CONTROL_KUBECONFIG": str(site.release_config["cpu_kubeconfig"]),
    }
    connections = connection_identity(args, environment)
    if not connections.get("environment:GPU_FAULT_CONTROL_KUBECONFIG", {}).get(
        "sha256"
    ):
        raise CapError("the CPU kubeconfig identity is missing")
    release_id = live_release_state(site).get("release_id")
    if not isinstance(release_id, str) or not release_id.strip():
        raise CapError("the deployed release identity is missing")
    identity = {
        "release_id": release_id,
        "cluster_id": str(site.release_config["cpu_eks_arn"]),
    }
    confirmation = args.case.removeprefix("GF-REGIONAL-").replace("-", "") + "_EXECUTE"
    predecessor_id, path = predecessor_path(
        args.run_dir,
        args.case,
        args.predecessor_evidence,
    )
    predecessor = (
        predecessor_evidence(
            path,
            predecessor_id,
            release_id=identity["release_id"],
            # Only CAP predecessors describe this CPU cluster, not a GPU target.
            cluster_id=identity["cluster_id"] if predecessor_id in CASE_IDS else None,
        )
        if predecessor_id is not None and path is not None
        else {"valid": True, "case_id": None, "verdict": "NOT_REQUIRED"}
    )
    details = {
        "risk": "live-non-destructive",
        "site_sha256": site.source_sha256,
        "site_identity": {
            key: site.release_config[key]
            for key in ("site_name", "aws_region", "cpu_eks_arn", "namespace")
        },
        "release_id": release_id,
        "predecessor": predecessor,
        "b_latency_factor": args.b_latency_factor,
        "mutation": (
            "create disposable capacity probe Deployments, Services, "
            "Secrets and one isolated PostgreSQL database"
        ),
        "stop_conditions": [
            "formal predecessor evidence is not PASS",
            "the selected CPU EKS or release identity drifts",
            "a probe cannot create or drop its isolated database",
            "the production baseline changes",
        ],
        "rollback": {
            "runner_finally_deletes_probe_resources": True,
            "runner_finally_drops_isolated_databases": True,
            "production_registry_is_not_modified": True,
        },
    }
    scrape_source_binding = None
    if args.case == "GF-REGIONAL-CAP-002" and predecessor.get("valid") is True:
        scrape_source_binding = capture_scrape_source(
            kubeconfig=Path(site.release_config["cpu_kubeconfig"]),
            namespace=str(site.release_config["namespace"]),
            region=str(site.release_config["aws_region"]),
            workspace_id=str(site.release_config["health"]["amp_workspace_id"]),
        )
        details["scrape_source_binding"] = scrape_source_binding
        details["mutation"] += (
            "; create an owned static-scrape ADOT Pod and ConfigMap using only "
            "the probe token and the deployed collector identity"
        )
        details["stop_conditions"].extend(
            [
                "the deployed ADOT image, configuration or ServiceAccount drifts",
                "fresh probe metrics do not reach AMP",
                "probe saturation does not clear before collector cleanup",
            ]
        )
    if not args.execute:
        plan = build_plan(
            arguments=args,
            preflight_passed=predecessor.get("valid") is True,
            run_dir=args.run_dir,
            case_id=args.case,
            attempt=args.attempt,
            confirmation=confirmation,
            environment=environment,
            details=details,
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if predecessor.get("valid") is True else 1
    if args.confirm != confirmation:
        raise CapError(f"confirmation must be exactly {confirmation}")
    deadline = authorize_execution(
        args,
        case_id=args.case,
        confirmation=confirmation,
        environment=environment,
        details=details,
    )
    if predecessor.get("valid") is not True:
        raise CapError("formal predecessor evidence is not PASS")
    harness = CapHarness(
        site_path=args.site,
        run_dir=args.run_dir,
        case_id=args.case,
        predecessor=predecessor,
        b_latency_factor=args.b_latency_factor,
        rendered_site=site,
        evidence_identity=identity,
        scrape_source_binding=scrape_source_binding,
        maintenance_deadline=deadline,
    )
    return harness.run()


if __name__ == "__main__":
    raise SystemExit(main())
