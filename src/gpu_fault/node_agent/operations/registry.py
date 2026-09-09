from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.protocol import UnsupportedOperationError
from gpu_fault.operation_registry import (
    OperationAdapter,
    operations_for_adapter,
)

OPERATION_HANDLERS: dict[WorkflowOperation, str] = {
    WorkflowOperation.COLLECT_HUNG_TRIAGE: "_collect_hung_triage",
    WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE: ("_collect_diagnostic_bundle"),
    WorkflowOperation.RUN_DCGM_DIAGNOSTIC: "_run_dcgm_diagnostic",
    WorkflowOperation.RUN_FIELD_DIAGNOSTIC: "_run_field_diagnostic",
    WorkflowOperation.RUN_NVLINK74_WORKFLOW: "_run_field_diagnostic",
    WorkflowOperation.QUIESCE_GPU_SERVICES: "_execute_quiesce",
    WorkflowOperation.VERIFY_NO_GPU_CLIENTS: ("_execute_verify_no_clients"),
    WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT: ("_execute_health_snapshot"),
    WorkflowOperation.RESET_GPU: "_execute_reset_gpu",
    WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES: ("_execute_reset_all"),
    WorkflowOperation.RESTORE_GPU_SERVICES: "_execute_restore",
    WorkflowOperation.REMEDIATE_EFA_DRIVER: "_remediate_efa_driver",
    WorkflowOperation.REMEDIATE_DRIVER: "_remediate_driver",
    WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE: "_update_firmware",
    WorkflowOperation.RESTART_FABRIC_MANAGER: ("_restart_fabric_manager"),
}


def operation_handler_name(operation: WorkflowOperation) -> str:
    handler = OPERATION_HANDLERS.get(operation)
    if handler is None:
        raise UnsupportedOperationError(
            f"node action operation has no executor branch: {operation.value}"
        )
    return handler


def validate_operation_handlers(executor_type=None) -> None:
    expected = operations_for_adapter(OperationAdapter.NODE_ACTION)
    actual = frozenset(OPERATION_HANDLERS)
    if actual != expected:
        missing = sorted(item.value for item in expected - actual)
        extra = sorted(item.value for item in actual - expected)
        raise RuntimeError(
            f"NodeAction handler registry mismatch: missing={missing} extra={extra}"
        )
    if executor_type is None:
        return
    missing_handlers = sorted(
        name
        for name in OPERATION_HANDLERS.values()
        if not callable(getattr(executor_type, name, None))
    )
    if missing_handlers:
        raise RuntimeError(
            "NodeAction handlers are not implemented: " + ", ".join(missing_handlers)
        )


validate_operation_handlers()
