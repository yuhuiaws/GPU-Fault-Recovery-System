from __future__ import annotations

import argparse

from gpu_fault.hyperpod import (
    HyperPodAction,
    HyperPodAdapterConfig,
    HyperPodLifecycleAdapter,
    HyperPodRecoveryState,
    HyperPodWorkloadRecoveryEvidence,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Read-only by default SageMaker HyperPod adapter"
    )
    result.add_argument("--cluster")
    result.add_argument("--region")
    subcommands = result.add_subparsers(dest="command", required=True)

    subcommands.add_parser("discover")
    ownership = subcommands.add_parser("ownership")
    ownership.add_argument(
        "--workload-recovery",
        choices=["enabled", "disabled", "unknown"],
        required=True,
    )
    ownership.add_argument("--evidence-source", default="operator-supplied")
    ownership.add_argument("--workload-id")

    for name in ("preflight", "reboot", "replace"):
        command = subcommands.add_parser(name)
        command.add_argument("--node", action="append", required=True)
        command.add_argument("--isolated-node", action="append", default=[])
        if name != "preflight":
            command.add_argument("--execute", action="store_true")
            command.add_argument("--confirm-cluster")
            command.add_argument("--workflow-fencing-token", type=int)
            command.add_argument("--expected-fencing-token", type=int)
            command.add_argument("--idempotency-key")
        else:
            command.add_argument(
                "--action",
                choices=["reboot", "replace"],
                default="reboot",
            )
    return result


def main() -> None:
    args = parser().parse_args()
    config = HyperPodAdapterConfig.from_environment(
        cluster_name=args.cluster,
        region_name=args.region,
    )
    adapter = HyperPodLifecycleAdapter(config)

    if args.command == "discover":
        print(adapter.discover().model_dump_json(indent=2))
        return

    if args.command == "ownership":
        state = {
            "enabled": HyperPodRecoveryState.ENABLED,
            "disabled": HyperPodRecoveryState.DISABLED,
            "unknown": HyperPodRecoveryState.UNKNOWN,
        }[args.workload_recovery]
        evidence = HyperPodWorkloadRecoveryEvidence(
            state=state,
            source=args.evidence_source,
            workload_id=args.workload_id,
        )
        print(adapter.resolve_recovery_ownership(evidence).model_dump_json(indent=2))
        return

    action = (
        HyperPodAction.REPLACE
        if (args.command == "replace" or getattr(args, "action", None) == "replace")
        else HyperPodAction.REBOOT
    )
    preflight = adapter.preflight(
        action,
        args.node,
        isolation_verified_nodes=args.isolated_node,
    )
    if args.command == "preflight" or not args.execute:
        print(preflight.model_dump_json(indent=2))
        return

    required = {
        "--confirm-cluster": args.confirm_cluster,
        "--workflow-fencing-token": args.workflow_fencing_token,
        "--expected-fencing-token": args.expected_fencing_token,
        "--idempotency-key": args.idempotency_key,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit("execution requires " + ", ".join(missing))
    result = adapter.submit(
        action,
        args.node,
        isolation_verified_nodes=args.isolated_node,
        confirm_cluster_name=args.confirm_cluster,
        workflow_fencing_token=args.workflow_fencing_token,
        expected_fencing_token=args.expected_fencing_token,
        idempotency_key=args.idempotency_key,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
