from __future__ import annotations

import argparse
import logging
import os

from gpu_fault.collectors import (
    DcgmMetricsCollector,
    FabricManagerLogCollector,
    HostTelemetryCollector,
    KernelLogCollector,
    KubernetesNodeResourceCollector,
    KubernetesHmaNodeCollector,
    NodeLogCollector,
    NvidiaSmiMetricsCollector,
    SqsHmaConsumer,
    TrainingProgressCollector,
    context_from_environment,
    sink_from_environment,
)
from gpu_fault.env_validation import validate_gpu_fault_environment


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Read-only GPU fault signal collectors"
    )
    subcommands = result.add_subparsers(dest="command", required=True)
    kernel = subcommands.add_parser("kernel")
    kernel.add_argument(
        "--node-id",
        default=os.getenv("NODE_NAME") or os.getenv("HOSTNAME"),
    )
    kernel.add_argument(
        "--kmsg-path",
        default=os.getenv("GPU_FAULT_KMSG_PATH", "/dev/kmsg"),
    )
    subcommands.add_parser("kubernetes-hma")
    kubernetes_resources = subcommands.add_parser("kubernetes-node-resources")
    kubernetes_resources.add_argument(
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
    subcommands.add_parser("sqs-hma")
    dcgm = subcommands.add_parser("dcgm")
    dcgm.add_argument(
        "--node-id",
        default=os.getenv("NODE_NAME") or os.getenv("HOSTNAME"),
    )
    dcgm.add_argument(
        "--metrics-url",
        default=os.getenv(
            "GPU_FAULT_DCGM_METRICS_URL",
            "http://127.0.0.1:9400/metrics",
        ),
    )
    dcgm.add_argument(
        "--interval-seconds",
        type=float,
        default=float(os.getenv("GPU_FAULT_METRICS_INTERVAL_SECONDS", "15")),
    )
    smi = subcommands.add_parser("nvidia-smi")
    smi.add_argument(
        "--node-id",
        default=os.getenv("NODE_NAME") or os.getenv("HOSTNAME"),
    )
    smi.add_argument(
        "--interval-seconds",
        type=float,
        default=float(os.getenv("GPU_FAULT_METRICS_INTERVAL_SECONDS", "30")),
    )
    host = subcommands.add_parser("host")
    host.add_argument(
        "--node-id",
        default=os.getenv("NODE_NAME") or os.getenv("HOSTNAME"),
    )
    host.add_argument(
        "--interval-seconds",
        type=float,
        default=float(os.getenv("GPU_FAULT_HOST_INTERVAL_SECONDS", "15")),
    )
    logs = subcommands.add_parser("logs")
    logs.add_argument(
        "--node-id",
        default=os.getenv("NODE_NAME") or os.getenv("HOSTNAME"),
    )
    fabric = subcommands.add_parser("fabric-manager")
    fabric.add_argument(
        "--node-id",
        default=os.getenv("NODE_NAME") or os.getenv("HOSTNAME"),
    )
    fabric.add_argument(
        "--interval-seconds",
        type=float,
        default=float(
            os.getenv(
                "GPU_FAULT_FABRIC_MANAGER_LOG_INTERVAL_SECONDS",
                "5",
            )
        ),
    )
    progress = subcommands.add_parser("training-progress")
    progress.add_argument(
        "--attempt-id",
        default=os.getenv("GPU_FAULT_ATTEMPT_ID"),
    )
    progress.add_argument(
        "--rank",
        type=int,
        default=(int(os.environ["RANK"]) if os.getenv("RANK") else None),
    )
    progress.add_argument(
        "--progress-file",
        default=os.getenv(
            "GPU_FAULT_TRAINING_PROGRESS_FILE",
            "/var/run/gpu-fault/progress.json",
        ),
    )
    progress.add_argument(
        "--interval-seconds",
        type=float,
        default=float(
            os.getenv(
                "GPU_FAULT_TRAINING_PROGRESS_INTERVAL_SECONDS",
                "15",
            )
        ),
    )
    logs.add_argument(
        "--interval-seconds",
        type=float,
        default=float(os.getenv("GPU_FAULT_LOG_INTERVAL_SECONDS", "10")),
    )
    return result


def _configure_logging() -> None:
    """Give every collector process a root logging configuration.

    This is the same defect ``api.py:_configure_logging`` fixed for the
    control plane, left unfixed on the node side: nothing in a collector
    process ever called basicConfig, so root stayed at WARNING with no
    handlers. Consequences observed on a live node:

    * every ``LOGGER.info`` was discarded outright -- including
      ``"delivered DCGM batch %s: %s"``, which is the only place the
      node side reports *why* the edge filter decided to deliver. A
      journal with zero delivery lines therefore proved nothing about
      the filter, and looked identical to a dead collector.
    * every ``LOGGER.warning`` / ``LOGGER.exception`` fell through to
      ``logging.lastResort``, which prints the bare message with no
      timestamp, level or logger name -- so tracebacks in the journal
      could not be attributed to a collector or correlated in time.

    Left alone when root already has handlers so tests and embedders
    keep control of their own logging.
    """

    if logging.getLogger().handlers:
        return
    level = (
        (os.getenv("GPU_FAULT_LOG_LEVEL") or os.getenv("LOG_LEVEL") or "INFO")
        .strip()
        .upper()
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main() -> None:
    _configure_logging()
    validate_gpu_fault_environment(process_name="gpu-fault-collector")
    args = parser().parse_args()
    os.environ.setdefault(
        "GPU_FAULT_COLLECTOR_OUTBOX_PATH",
        f"/var/lib/gpu-fault/outbox/{args.command}.ndjson",
    )
    sink = sink_from_environment()
    if args.command == "sqs-hma":
        queue_url = os.getenv("GPU_FAULT_HMA_QUEUE_URL")
        if not queue_url:
            raise SystemExit("GPU_FAULT_HMA_QUEUE_URL is required")
        SqsHmaConsumer(sink, queue_url).run()
        return
    context = context_from_environment(
        discover_product=args.command
        in {
            "kernel",
            "dcgm",
            "nvidia-smi",
            "host",
            "logs",
            "fabric-manager",
        }
    )
    if args.command == "kernel":
        if not args.node_id:
            raise SystemExit("--node-id, NODE_NAME, or HOSTNAME is required")
        KernelLogCollector(
            sink,
            context,
            node_id=args.node_id,
            kmsg_path=args.kmsg_path,
        ).run()
        return
    if args.command == "dcgm":
        if not args.node_id:
            raise SystemExit("--node-id, NODE_NAME, or HOSTNAME is required")
        DcgmMetricsCollector(
            sink,
            context,
            node_id=args.node_id,
            metrics_url=args.metrics_url,
            interval_seconds=args.interval_seconds,
        ).run()
        return
    if args.command == "nvidia-smi":
        if not args.node_id:
            raise SystemExit("--node-id, NODE_NAME, or HOSTNAME is required")
        NvidiaSmiMetricsCollector(
            sink,
            context,
            node_id=args.node_id,
            interval_seconds=args.interval_seconds,
        ).run()
        return
    if args.command == "host":
        if not args.node_id:
            raise SystemExit("--node-id, NODE_NAME, or HOSTNAME is required")
        expected_gpu_count = os.getenv("GPU_FAULT_EXPECTED_GPU_COUNT")
        expected_efa_device_count = os.getenv("GPU_FAULT_EXPECTED_EFA_DEVICE_COUNT")
        HostTelemetryCollector(
            sink,
            context,
            node_id=args.node_id,
            interval_seconds=args.interval_seconds,
            filesystems=[
                item
                for item in os.getenv("GPU_FAULT_FILESYSTEMS", "/,/var,/tmp").split(",")
                if item
            ],
            required_interfaces=[
                item
                for item in os.getenv("GPU_FAULT_REQUIRED_INTERFACES", "").split(",")
                if item
            ],
            pci_devices_root=(os.getenv("GPU_FAULT_PCI_DEVICES_ROOT") or None),
            node_instance_type=(os.getenv("GPU_FAULT_NODE_INSTANCE_TYPE") or None),
            expected_gpu_count=(
                int(expected_gpu_count) if expected_gpu_count else None
            ),
            expected_efa_device_count=(
                int(expected_efa_device_count) if expected_efa_device_count else None
            ),
            inventory_mismatch_consecutive_samples=int(
                os.getenv(
                    "GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES",
                    "2",
                )
            ),
        ).run()
        return
    if args.command == "logs":
        if not args.node_id:
            raise SystemExit("--node-id, NODE_NAME, or HOSTNAME is required")
        NodeLogCollector(
            sink,
            context,
            node_id=args.node_id,
            interval_seconds=args.interval_seconds,
            training_log_paths=[
                item
                for item in os.getenv("GPU_FAULT_TRAINING_LOG_PATHS", "").split(",")
                if item
            ],
            state_path=os.getenv(
                "GPU_FAULT_LOG_STATE_PATH",
                "/var/lib/gpu-fault/log-collector-state.json",
            ),
        ).run()
        return
    if args.command == "fabric-manager":
        if not args.node_id:
            raise SystemExit("--node-id, NODE_NAME, or HOSTNAME is required")
        FabricManagerLogCollector(
            sink,
            context,
            node_id=args.node_id,
            interval_seconds=args.interval_seconds,
            journal_enabled=(
                os.getenv("GPU_FAULT_FABRIC_MANAGER_JOURNAL", "true").lower() == "true"
            ),
            journal_identifiers=tuple(
                item
                for item in os.getenv(
                    "GPU_FAULT_FABRIC_MANAGER_IDENTIFIERS",
                    "nvidia-fabricmanager,nv-fabricmanager",
                ).split(",")
                if item
            ),
            log_paths=[
                item
                for item in os.getenv("GPU_FAULT_FABRIC_MANAGER_LOG_PATHS", "").split(
                    ","
                )
                if item
            ],
            state_path=os.getenv(
                "GPU_FAULT_FABRIC_MANAGER_STATE_PATH",
                "/var/lib/gpu-fault/fabric-manager-collector-state.json",
            ),
        ).run()
        return
    if args.command == "training-progress":
        if not args.attempt_id or args.rank is None:
            raise SystemExit("training-progress requires --attempt-id and --rank")
        TrainingProgressCollector(
            sink,
            cluster_id=context.cluster_id,
            attempt_id=args.attempt_id,
            rank=args.rank,
            progress_path=args.progress_file,
            node_id=os.getenv("NODE_NAME") or os.getenv("HOSTNAME"),
            pod_uid=os.getenv("POD_UID"),
            container_name=os.getenv("CONTAINER_NAME", "trainer"),
            gpu_uuids=[
                item for item in os.getenv("GPU_FAULT_GPU_UUIDS", "").split(",") if item
            ],
            interval_seconds=args.interval_seconds,
        ).run()
        return
    if args.command == "kubernetes-node-resources":
        KubernetesNodeResourceCollector(
            sink,
            context,
            interval_seconds=args.interval_seconds,
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
        ).run()
        return
    KubernetesHmaNodeCollector(sink, context).run()


if __name__ == "__main__":
    main()
