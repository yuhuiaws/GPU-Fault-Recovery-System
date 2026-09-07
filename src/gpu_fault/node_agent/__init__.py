from gpu_fault.lazy_exports import lazy_module

_EXPORTS = {
    "AgentHeartbeatRejected": (
        "gpu_fault.node_agent.heartbeat",
        "AgentHeartbeatRejected",
    ),
    "AgentHeartbeatReporter": (
        "gpu_fault.node_agent.heartbeat",
        "AgentHeartbeatReporter",
    ),
    "GpuServiceQuiesceManager": (
        "gpu_fault.node_agent.quiesce",
        "GpuServiceQuiesceManager",
    ),
    "NodeActionCommand": ("gpu_fault.node_agent.protocol", "NodeActionCommand"),
    "NodeActionExecutionState": (
        "gpu_fault.node_agent.protocol",
        "NodeActionExecutionState",
    ),
    "NodeActionExecutor": ("gpu_fault.node_agent.executor", "NodeActionExecutor"),
    "NodeActionLedger": ("gpu_fault.node_agent.ledger", "NodeActionLedger"),
    "NodeActionResult": ("gpu_fault.node_agent.protocol", "NodeActionResult"),
    "NodeActionStatus": ("gpu_fault.node_agent.protocol", "NodeActionStatus"),
    "NodeActionSubmission": ("gpu_fault.node_agent.protocol", "NodeActionSubmission"),
    "SignedNodeAction": ("gpu_fault.node_agent.protocol", "SignedNodeAction"),
    "UnsupportedOperationError": (
        "gpu_fault.node_agent.protocol",
        "UnsupportedOperationError",
    ),
    "agent_config_digest": ("gpu_fault.node_agent.config", "agent_config_digest"),
    "agent_config_payload": ("gpu_fault.node_agent.config", "agent_config_payload"),
    "canonical_command": ("gpu_fault.node_agent.protocol", "canonical_command"),
    "create_node_agent_app": ("gpu_fault.node_agent.app", "create_node_agent_app"),
    "executor_from_environment": (
        "gpu_fault.node_agent.config",
        "executor_from_environment",
    ),
    "heartbeat_reporter_from_environment": (
        "gpu_fault.node_agent.heartbeat",
        "heartbeat_reporter_from_environment",
    ),
    "print_config_digest": ("gpu_fault.node_agent.config", "print_config_digest"),
    "restore_gpu_services": ("gpu_fault.node_agent.quiesce", "restore_gpu_services"),
    "run": ("gpu_fault.node_agent.app", "run"),
    "sign_node_action": ("gpu_fault.node_agent.protocol", "sign_node_action"),
    "sign_result_query": ("gpu_fault.node_agent.protocol", "sign_result_query"),
    "validate_host_proc_root": (
        "gpu_fault.node_agent.config",
        "validate_host_proc_root",
    ),
    "verify_result_query": ("gpu_fault.node_agent.protocol", "verify_result_query"),
}

__getattr__, __dir__, __all__ = lazy_module(globals(), _EXPORTS)
