from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.identity_acceptance_auth import (  # noqa: E402
    auth015_focused_tests,
    run_auth007,
    run_auth010,
    run_auth013,
    run_auth014,
    run_auth015,
    run_auth016,
)
from scripts.e2e.regional.identity_acceptance_common import (  # noqa: E402
    ClusterTarget,
    IdentityAcceptanceError,
    IdentityCaseFailure,
    IdentitySite,
    utc_now,
)
from scripts.e2e.regional.identity_acceptance_iso import (  # noqa: E402
    run_iso003,
    run_iso004,
    run_iso005,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
    install_site_profile,
    record_focused_tests,
    reusable_focused_tests,
)
from scripts.e2e.regional.regional_case_contract import (  # noqa: E402
    case_evidence_path,
    predecessor_path,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    predecessor_evidence,
)

CASE_IDS = (
    "GF-REGIONAL-AUTH-007",
    "GF-REGIONAL-AUTH-010",
    "GF-REGIONAL-AUTH-013",
    "GF-REGIONAL-AUTH-014",
    "GF-REGIONAL-AUTH-015",
    "GF-REGIONAL-AUTH-016",
    "GF-REGIONAL-ISO-003",
    "GF-REGIONAL-ISO-004",
    "GF-REGIONAL-ISO-005",
)


def case_plan(
    case_id: str,
    *,
    primary: ClusterTarget,
    secondary: ClusterTarget | None,
    nodes: tuple[str, ...],
    predecessor: dict[str, Any],
    evidence_identity: dict[str, str],
) -> dict[str, Any]:
    mutations = {
        "GF-REGIONAL-AUTH-007": (
            "disable one test registration, roll ingress/control-worker, then restore"
        ),
        "GF-REGIONAL-AUTH-010": "anonymous read-only route matrix",
        "GF-REGIONAL-AUTH-013": "TLS handshakes with empty CA and wrong hostname",
        "GF-REGIONAL-AUTH-014": "anonymous route audit plus external SG evidence",
        "GF-REGIONAL-AUTH-015": (
            "rotate one staging node key, scan two nodes, then restore key Secrets"
        ),
        "GF-REGIONAL-AUTH-016": (
            "rotate one test cluster token through a bounded overlap window, then "
            "drop the retiring digest and restore both Secrets"
        ),
        "GF-REGIONAL-ISO-003": "cross-cluster Fleet calls that must be rejected",
        "GF-REGIONAL-ISO-004": "cross-cluster spare-health query that must be rejected",
        "GF-REGIONAL-ISO-005": (
            "create a pause Deployment in a forbidden namespace, temporarily widen "
            "only the control-plane allowlist, then restore"
        ),
    }
    return {
        "risk": "case-defined",
        "predecessor": predecessor,
        "evidence_identity": evidence_identity,
        "primary": {
            "cluster_id": primary.cluster_id,
            "context": primary.context,
        },
        "secondary": (
            {
                "cluster_id": secondary.cluster_id,
                "context": secondary.context,
            }
            if secondary is not None
            else None
        ),
        "nodes": list(nodes),
        "mutation": mutations[case_id],
        "stop_conditions": [
            "formal predecessor evidence is not PASS",
            "site, Region, context or cluster binding drifts",
            "any Secret value would be written to evidence",
            "a control-plane or executor rejection differs from the exact contract",
            "registry, token, key, namespace or probe cleanup is incomplete",
        ],
        "rollback": {
            "registry_and_token_documents_are_kept_only_in_memory": True,
            "all_mutating_handlers_restore_in_finally": True,
            "host_probes_have_active_deadlines": True,
        },
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run one guarded regional AUTH/ISO identity-boundary acceptance case."
        )
    )
    add_live_arguments(value, confirmation="CASE_SPECIFIC_CONFIRMATION")
    value.add_argument("--case", choices=CASE_IDS, required=True)
    value.add_argument("--site", type=Path, required=True)
    value.add_argument("--cluster-id", default="")
    value.add_argument("--secondary-cluster-id", default="")
    value.add_argument("--node", action="append", default=[])
    value.add_argument("--fleet-master-file", type=Path)
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--outside-probe-evidence", type=Path)
    value.add_argument("--predecessor-evidence", default="")
    return value


def validate_case_arguments(
    arguments: argparse.Namespace,
    site: IdentitySite,
) -> tuple[ClusterTarget, ClusterTarget | None, tuple[str, ...]]:
    primary = site.target(arguments.cluster_id)
    secondary_cases = {
        "GF-REGIONAL-AUTH-007",
        "GF-REGIONAL-ISO-003",
        "GF-REGIONAL-ISO-004",
    }
    secondary = None
    if arguments.case in secondary_cases:
        if not arguments.secondary_cluster_id:
            raise IdentityAcceptanceError(
                f"{arguments.case} requires --secondary-cluster-id"
            )
        secondary = site.target(arguments.secondary_cluster_id)
        if secondary.cluster_id == primary.cluster_id:
            raise IdentityAcceptanceError("primary and secondary clusters must differ")
    nodes = tuple(arguments.node)
    if arguments.case == "GF-REGIONAL-AUTH-015":
        if (
            len(nodes) != 2
            or len(set(nodes)) != 2
            or arguments.fleet_master_file is None
            or not arguments.host_probe_image
        ):
            raise IdentityAcceptanceError(
                "AUTH-015 requires two distinct --node values, "
                "--fleet-master-file and --host-probe-image"
            )
        if not arguments.fleet_master_file.is_file():
            raise IdentityAcceptanceError("fleet master file does not exist")
        if "@sha256:" not in arguments.host_probe_image:
            raise IdentityAcceptanceError(
                "AUTH-015 host probe image must use an immutable digest"
            )
    if arguments.case == "GF-REGIONAL-AUTH-013":
        # Optional: one node plus a host probe image lets the case read the
        # per-node certificate-expiry timer; without them that check is
        # recorded as not evaluated and the case cannot PASS.
        if len(nodes) > 1 or bool(nodes) != bool(arguments.host_probe_image):
            raise IdentityAcceptanceError(
                "AUTH-013 takes at most one --node, together with --host-probe-image"
            )
        if nodes and "@sha256:" not in arguments.host_probe_image:
            raise IdentityAcceptanceError(
                "AUTH-013 host probe image must use an immutable digest"
            )
    return primary, secondary, nodes


def main() -> int:
    install_site_profile()
    arguments = parser().parse_args()
    os.umask(0o077)
    site = IdentitySite(arguments.site)
    primary, secondary, nodes = validate_case_arguments(arguments, site)
    # The release and cluster this run's evidence is bound to; the predecessor
    # must have been earned against the same pair, and the successor will ask
    # the same of this case's evidence.
    identity = site.regional(primary).evidence_identity()
    predecessor_id, path = predecessor_path(
        arguments.run_dir,
        arguments.case,
        arguments.predecessor_evidence,
    )
    predecessor = (
        predecessor_evidence(path, predecessor_id, **identity)
        if predecessor_id is not None and path is not None
        else {"valid": True, "case_id": None, "verdict": "NOT_REQUIRED"}
    )
    confirmation = (
        arguments.case.removeprefix("GF-REGIONAL-").replace("-", "") + "_EXECUTE"
    )
    environment = {
        "GPU_FAULT_IDENTITY_CASE": arguments.case,
        "GPU_FAULT_SITE_FILE": str(arguments.site.resolve()),
        "GPU_FAULT_PRIMARY_CLUSTER_ID": primary.cluster_id,
        "GPU_FAULT_SECONDARY_CLUSTER_ID": (
            secondary.cluster_id if secondary is not None else ""
        ),
        "GPU_FAULT_TARGET_NODES": ",".join(nodes),
    }
    case_dir = arguments.run_dir / "cases" / arguments.case
    case_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not arguments.execute:
        details = case_plan(
            arguments.case,
            primary=primary,
            secondary=secondary,
            nodes=nodes,
            predecessor=predecessor,
            evidence_identity=identity,
        )
        if arguments.case == "GF-REGIONAL-AUTH-015":
            # Run the focused pytest here so --execute can reuse the result
            # against an unchanged tree instead of paying for it twice.
            record_focused_tests(details, auth015_focused_tests())
        plan = build_plan(
            run_dir=arguments.run_dir,
            case_id=arguments.case,
            attempt=arguments.attempt,
            confirmation=confirmation,
            environment=environment,
            details=details,
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        focused_ok = details.get("focused_tests", {"passed": True}).get("passed")
        return 0 if predecessor.get("valid", False) and focused_ok is True else 1
    if arguments.confirm != confirmation:
        raise IdentityAcceptanceError(f"confirmation must be exactly {confirmation}")
    authorize_execution(
        arguments,
        case_id=arguments.case,
        confirmation=confirmation,
        environment=environment,
    )
    if not predecessor.get("valid", False):
        raise IdentityAcceptanceError("formal predecessor evidence is not PASS")
    started_at = utc_now()
    try:
        handlers: dict[str, Callable[[], dict[str, Any]]] = {
            "GF-REGIONAL-AUTH-007": lambda: run_auth007(
                site, primary, cast(ClusterTarget, secondary), case_dir=case_dir
            ),
            "GF-REGIONAL-AUTH-010": lambda: run_auth010(site, primary),
            "GF-REGIONAL-AUTH-013": lambda: run_auth013(
                site,
                primary,
                node=nodes[0] if nodes else "",
                host_probe_image=arguments.host_probe_image,
                case_dir=case_dir,
            ),
            "GF-REGIONAL-AUTH-014": lambda: run_auth014(
                site,
                primary,
                outside_probe_path=arguments.outside_probe_evidence,
            ),
            "GF-REGIONAL-AUTH-015": lambda: run_auth015(
                site,
                primary,
                nodes=cast(tuple[str, str], nodes),
                fleet_master_file=arguments.fleet_master_file,
                host_probe_image=arguments.host_probe_image,
                case_dir=case_dir,
                focused_tests=reusable_focused_tests(case_dir / "plan.json"),
            ),
            "GF-REGIONAL-AUTH-016": lambda: run_auth016(
                site, primary, case_dir=case_dir
            ),
            "GF-REGIONAL-ISO-003": lambda: run_iso003(
                site, primary, cast(ClusterTarget, secondary)
            ),
            "GF-REGIONAL-ISO-004": lambda: run_iso004(
                site, primary, cast(ClusterTarget, secondary)
            ),
            "GF-REGIONAL-ISO-005": lambda: run_iso005(
                site,
                primary,
                case_dir=case_dir,
            ),
        }
        outcome = handlers[arguments.case]()
    except IdentityCaseFailure as exc:
        # The handler restored what it could and kept its partial checks;
        # they are evidence of how far the case got, not a PASS.
        outcome = {
            "verdict": "FAIL",
            "error": f"{type(exc.__cause__ or exc).__name__}: {exc}",
            "partial": exc.details,
            "cleanup_errors": list(exc.details.get("cleanup_errors") or []),
            "limitations": [
                "The case stopped at the first failed step; the checks under "
                "'partial' were gathered before it and the restore state after."
            ],
        }
    except Exception as exc:
        outcome = {
            "verdict": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "limitations": [
                "The case stopped at the first failed assertion; later checks "
                "were not treated as executed."
            ],
        }
    result = {
        "schema_version": 2,
        "report_type": "fault-acceptance",
        "case_id": arguments.case,
        "verdict": outcome.get("verdict", "FAIL"),
        "started_at": started_at,
        "executed_at": utc_now(),
        "predecessor": predecessor,
        **identity,
        **{
            key: value
            for key, value in outcome.items()
            if key not in {"verdict", *identity}
        },
    }
    write_json_atomic(case_evidence_path(arguments.run_dir, arguments.case), result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
