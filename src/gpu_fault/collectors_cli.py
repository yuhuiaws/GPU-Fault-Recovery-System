from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path

from gpu_fault.collector_registry import (
    COLLECTOR_REGISTRY,
    CollectorDescriptor,
    collector_registry_with_plugins,
)
from gpu_fault.collectors import (
    KubernetesHmaNodeCollector,
    KubernetesNodeResourceCollector,
    SqsHmaConsumer,
    context_from_environment,
    sink_from_environment,
)
from gpu_fault.collectors.models import CollectorContext
from gpu_fault.collectors.sinks import EventSink, OutboxFile
from gpu_fault.env_validation import validate_gpu_fault_environment
from gpu_fault.logging_setup import configure_logging

DEFAULT_OUTBOX_DIRECTORY = "/var/lib/gpu-fault/outbox"


def _add_outbox_parser(
    subcommands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """``gpu-fault-collector outbox {list,stats,requeue-dead}`` (ARCH-G2).

    Reads the NDJSON outbox a collector writes; never the control plane.
    ``list`` and ``stats`` show record metadata only -- a payload is fault
    evidence and does not belong on a terminal. ``requeue-dead`` flips
    dead-lettered records back to replayable and demands ``--yes`` because
    a replayed verdict is retried by every collector restart until it
    dead-letters again.

    ``requeue-dead`` runs its read-modify-write under the outbox's
    ``fcntl.flock`` (``<outbox>.lock``), so it waits for a collector that is
    buffering or replaying instead of racing it and losing one side's update
    (F7). It leaves records whose payload was truncated to a digest dead: only
    a 4 KB excerpt of those bodies exists, so they cannot be replayed.
    """

    outbox = subcommands.add_parser(
        "outbox", help="inspect or requeue a collector's durable outbox"
    )
    outbox.add_argument(
        "--outbox-path",
        default=os.getenv("GPU_FAULT_COLLECTOR_OUTBOX_PATH"),
        help=(
            "outbox file; defaults to GPU_FAULT_COLLECTOR_OUTBOX_PATH or "
            f"{DEFAULT_OUTBOX_DIRECTORY}/<collector>.ndjson with --collector"
        ),
    )
    outbox.add_argument(
        "--collector",
        default=None,
        help="collector subcommand name whose default outbox to read",
    )
    actions = outbox.add_subparsers(dest="outbox_action", required=True)
    actions.add_parser("list", help="one metadata line per buffered record")
    actions.add_parser("stats", help="depth, replayable/dead split, oldest failure")
    requeue = actions.add_parser(
        "requeue-dead", help="mark dead-lettered records replayable again"
    )
    requeue.add_argument(
        "--yes",
        action="store_true",
        help="required: confirm the dead records should be replayed",
    )
    requeue.add_argument(
        "--path",
        default=None,
        help="only requeue records for this control-plane path",
    )


def _outbox_path(args: argparse.Namespace) -> Path:
    if args.outbox_path:
        return Path(args.outbox_path)
    if args.collector:
        return Path(DEFAULT_OUTBOX_DIRECTORY) / f"{args.collector}.ndjson"
    raise SystemExit("outbox requires --outbox-path or --collector")


def run_outbox_command(args: argparse.Namespace) -> None:
    outbox = OutboxFile(_outbox_path(args))
    if args.outbox_action == "list":
        for index, record in enumerate(outbox.read()):
            print(OutboxFile.describe(index, record))
        return
    if args.outbox_action == "stats":
        print(json.dumps(outbox.stats(), sort_keys=True))
        return
    if not args.yes:
        raise SystemExit(
            "requeue-dead replays records the control plane already rejected; "
            "pass --yes to confirm"
        )
    requeued = outbox.requeue_dead(path_filter=args.path)
    print(f"requeued {requeued} dead record(s) in {outbox.path}")


def _add_node_id(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--node-id",
        default=os.getenv("NODE_NAME") or os.getenv("HOSTNAME"),
    )


def _add_interval(command: argparse.ArgumentParser, default: float) -> None:
    command.add_argument("--interval-seconds", type=float, default=default)


def _kernel_arguments(command: argparse.ArgumentParser) -> None:
    _add_node_id(command)
    command.add_argument(
        "--kmsg-path",
        default=os.getenv("GPU_FAULT_KMSG_PATH", "/dev/kmsg"),
    )


def _kubernetes_node_resources_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--interval-seconds",
        type=float,
        default=float(
            os.getenv(
                "GPU_FAULT_KUBERNETES_RESOURCE_INTERVAL_SECONDS",
                os.getenv(
                    "GPU_FAULT_KUBERNETES_EFA_INTERVAL_SECONDS",
                    "15",
                ),
            )
        ),
    )


def _dcgm_arguments(command: argparse.ArgumentParser) -> None:
    _add_node_id(command)
    command.add_argument(
        "--metrics-url",
        default=os.getenv(
            "GPU_FAULT_DCGM_METRICS_URL",
            "http://127.0.0.1:9400/metrics",
        ),
    )
    _add_interval(command, float(os.getenv("GPU_FAULT_METRICS_INTERVAL_SECONDS", "15")))


def _nvidia_smi_arguments(command: argparse.ArgumentParser) -> None:
    _add_node_id(command)
    _add_interval(command, float(os.getenv("GPU_FAULT_METRICS_INTERVAL_SECONDS", "30")))


def _host_arguments(command: argparse.ArgumentParser) -> None:
    _add_node_id(command)
    _add_interval(command, float(os.getenv("GPU_FAULT_HOST_INTERVAL_SECONDS", "15")))


def _logs_arguments(command: argparse.ArgumentParser) -> None:
    _add_node_id(command)
    _add_interval(command, float(os.getenv("GPU_FAULT_LOG_INTERVAL_SECONDS", "10")))


def _fabric_manager_arguments(command: argparse.ArgumentParser) -> None:
    _add_node_id(command)
    _add_interval(
        command,
        float(
            os.getenv(
                "GPU_FAULT_FABRIC_MANAGER_LOG_INTERVAL_SECONDS",
                "5",
            )
        ),
    )


def _training_progress_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--attempt-id",
        default=os.getenv("GPU_FAULT_ATTEMPT_ID"),
    )
    command.add_argument(
        "--rank",
        type=int,
        default=(int(os.environ["RANK"]) if os.getenv("RANK") else None),
    )
    command.add_argument(
        "--progress-file",
        default=os.getenv(
            "GPU_FAULT_TRAINING_PROGRESS_FILE",
            "/var/run/gpu-fault/progress.json",
        ),
    )
    _add_interval(
        command,
        float(
            os.getenv(
                "GPU_FAULT_TRAINING_PROGRESS_INTERVAL_SECONDS",
                "15",
            )
        ),
    )


# Per-command argparse options. A command absent here takes no options; a
# plugin collector reads its configuration from the environment.
CLI_ARGUMENTS: dict[str, Callable[[argparse.ArgumentParser], None]] = {
    "kernel": _kernel_arguments,
    "kubernetes-node-resources": _kubernetes_node_resources_arguments,
    "dcgm": _dcgm_arguments,
    "nvidia-smi": _nvidia_smi_arguments,
    "host": _host_arguments,
    "logs": _logs_arguments,
    "fabric-manager": _fabric_manager_arguments,
    "training-progress": _training_progress_arguments,
}


def parser(
    registry: Mapping[str, CollectorDescriptor] | None = None,
) -> argparse.ArgumentParser:
    registry = COLLECTOR_REGISTRY if registry is None else registry
    result = argparse.ArgumentParser(
        description="Read-only GPU fault signal collectors"
    )
    subcommands = result.add_subparsers(dest="command", required=True)
    for command in registry:
        subparser = subcommands.add_parser(command)
        add_arguments = CLI_ARGUMENTS.get(command)
        if add_arguments is not None:
            add_arguments(subparser)
    _add_outbox_parser(subcommands)
    return result


# Factories for the cluster-side commands whose collector modules are shared
# with the Lambda and control-plane surfaces; the node collectors carry their
# own ``build_from_environment``.
def build_sqs_hma(sink: EventSink, arguments: argparse.Namespace) -> SqsHmaConsumer:
    queue_url = os.getenv("GPU_FAULT_HMA_QUEUE_URL")
    if not queue_url:
        raise SystemExit("GPU_FAULT_HMA_QUEUE_URL is required")
    return SqsHmaConsumer(sink, queue_url)


def build_kubernetes_hma(
    sink: EventSink, context: CollectorContext, arguments: argparse.Namespace
) -> KubernetesHmaNodeCollector:
    return KubernetesHmaNodeCollector(sink, context)


def build_kubernetes_node_resources(
    sink: EventSink, context: CollectorContext, arguments: argparse.Namespace
) -> KubernetesNodeResourceCollector:
    return KubernetesNodeResourceCollector(
        sink,
        context,
        interval_seconds=arguments.interval_seconds,
        required_consecutive_samples=int(
            os.getenv(
                "GPU_FAULT_KUBERNETES_EFA_MISMATCH_SAMPLES",
                "2",
            )
        ),
        health_summary_seconds=int(
            os.getenv(
                "GPU_FAULT_KUBERNETES_EFA_HEALTH_SUMMARY_SECONDS",
                "300",
            )
        ),
        list_page_size=int(
            os.getenv("GPU_FAULT_KUBERNETES_NODE_LIST_PAGE_SIZE", "500")
        ),
    )


def main() -> None:
    configure_logging()
    registry = collector_registry_with_plugins()
    args = parser(registry).parse_args()
    if args.command == "outbox":
        # An operator command on a file: no control-plane URL, no collector.
        run_outbox_command(args)
        return
    validate_gpu_fault_environment(process_name="gpu-fault-collector")
    descriptor = registry[args.command]
    os.environ.setdefault(
        "GPU_FAULT_COLLECTOR_OUTBOX_PATH",
        f"/var/lib/gpu-fault/outbox/{args.command}.ndjson",
    )
    sink = sink_from_environment()
    context = (
        context_from_environment(discover_product=descriptor.needs_product_discovery)
        if descriptor.needs_context
        else None
    )
    descriptor.build(sink, context, args).run()


if __name__ == "__main__":
    main()
