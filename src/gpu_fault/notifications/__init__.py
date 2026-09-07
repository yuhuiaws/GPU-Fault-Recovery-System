from gpu_fault.lazy_exports import lazy_module

_EXPORTS = {
    "FABRIC_RESET_EMAIL_TEMPLATE": (
        "gpu_fault.notifications.common",
        "FABRIC_RESET_EMAIL_TEMPLATE",
    ),
    "FABRIC_RESET_TEMPLATE_VERSION": (
        "gpu_fault.notifications.common",
        "FABRIC_RESET_TEMPLATE_VERSION",
    ),
    "GPU_COUNT_CHANGE_EMAIL_TEMPLATE": (
        "gpu_fault.notifications.common",
        "GPU_COUNT_CHANGE_EMAIL_TEMPLATE",
    ),
    "GPU_RESET_EMAIL_TEMPLATE": (
        "gpu_fault.notifications.common",
        "GPU_RESET_EMAIL_TEMPLATE",
    ),
    "GPU_RESET_TEMPLATE_VERSION": (
        "gpu_fault.notifications.common",
        "GPU_RESET_TEMPLATE_VERSION",
    ),
    "NOT_APPLICABLE_EMAIL_TEMPLATE": (
        "gpu_fault.notifications.common",
        "NOT_APPLICABLE_EMAIL_TEMPLATE",
    ),
    "NOT_APPLICABLE_TEMPLATE_VERSION": (
        "gpu_fault.notifications.common",
        "NOT_APPLICABLE_TEMPLATE_VERSION",
    ),
    "NVLINK74_SUPPORT_TEMPLATE_VERSION": (
        "gpu_fault.notifications.common",
        "NVLINK74_SUPPORT_TEMPLATE_VERSION",
    ),
    "RESTART_FABRIC_MANAGER_EMAIL_TEMPLATE": (
        "gpu_fault.notifications.common",
        "RESTART_FABRIC_MANAGER_EMAIL_TEMPLATE",
    ),
    "RESTART_FABRIC_MANAGER_TEMPLATE_VERSION": (
        "gpu_fault.notifications.common",
        "RESTART_FABRIC_MANAGER_TEMPLATE_VERSION",
    ),
    "RESTART_GUARD_TEMPLATE_VERSION": (
        "gpu_fault.notifications.common",
        "RESTART_GUARD_TEMPLATE_VERSION",
    ),
    "RESTART_NODE_EMAIL_TEMPLATE": (
        "gpu_fault.notifications.common",
        "RESTART_NODE_EMAIL_TEMPLATE",
    ),
    "RESTART_NODE_TEMPLATE_VERSION": (
        "gpu_fault.notifications.common",
        "RESTART_NODE_TEMPLATE_VERSION",
    ),
    "RESTART_WORKLOAD_EMAIL_TEMPLATE": (
        "gpu_fault.notifications.common",
        "RESTART_WORKLOAD_EMAIL_TEMPLATE",
    ),
    "RESTART_WORKLOAD_TEMPLATE_VERSION": (
        "gpu_fault.notifications.common",
        "RESTART_WORKLOAD_TEMPLATE_VERSION",
    ),
    "SXID_EVENT_TEMPLATE_VERSION": (
        "gpu_fault.notifications.common",
        "SXID_EVENT_TEMPLATE_VERSION",
    ),
    "XID_INVESTIGATORY_EMAIL_TEMPLATE": (
        "gpu_fault.notifications.common",
        "XID_INVESTIGATORY_EMAIL_TEMPLATE",
    ),
    "XID_INVESTIGATORY_TEMPLATE_VERSION": (
        "gpu_fault.notifications.common",
        "XID_INVESTIGATORY_TEMPLATE_VERSION",
    ),
    "HyperPodAdvisoryEmailBuilder": (
        "gpu_fault.notifications.hyperpod_advisory",
        "HyperPodAdvisoryEmailBuilder",
    ),
    "HardwareEscalationEmailBuilder": (
        "gpu_fault.notifications.hardware_escalation",
        "HardwareEscalationEmailBuilder",
    ),
    "Nvlink74SupportEmailBuilder": (
        "gpu_fault.notifications.nvlink74_support",
        "Nvlink74SupportEmailBuilder",
    ),
    "Nvlink74MechanicalEmailBuilder": (
        "gpu_fault.notifications.nvlink74_mechanical",
        "Nvlink74MechanicalEmailBuilder",
    ),
    "DcgmDiagnosticEmailBuilder": (
        "gpu_fault.notifications.dcgm_diagnostic",
        "DcgmDiagnosticEmailBuilder",
    ),
    "RestartGuardEmailBuilder": (
        "gpu_fault.notifications.restart_guard",
        "RestartGuardEmailBuilder",
    ),
    "WarmSpareReplacementEmailBuilder": (
        "gpu_fault.notifications.warm_spare",
        "WarmSpareReplacementEmailBuilder",
    ),
    "XidInvestigatoryEmailBuilder": (
        "gpu_fault.notifications.xid_investigatory",
        "XidInvestigatoryEmailBuilder",
    ),
    "EfaRdmaEventEmailBuilder": (
        "gpu_fault.notifications.efa_rdma",
        "EfaRdmaEventEmailBuilder",
    ),
    "HostResourceEventEmailBuilder": (
        "gpu_fault.notifications.host_resource",
        "HostResourceEventEmailBuilder",
    ),
    "HardwareInventoryEmailBuilder": (
        "gpu_fault.notifications.hardware_inventory",
        "HardwareInventoryEmailBuilder",
    ),
    "SxidEventEmailBuilder": (
        "gpu_fault.notifications.sxid_event",
        "SxidEventEmailBuilder",
    ),
    "NotApplicableEmailBuilder": (
        "gpu_fault.notifications.not_applicable",
        "NotApplicableEmailBuilder",
    ),
    "SesV2Client": ("gpu_fault.notifications.ses", "SesV2Client"),
    "SesNotificationConfig": ("gpu_fault.notifications.ses", "SesNotificationConfig"),
    "SesEmailNotifier": ("gpu_fault.notifications.ses", "SesEmailNotifier"),
    "DisabledNotificationNotifier": (
        "gpu_fault.notifications.ses",
        "DisabledNotificationNotifier",
    ),
    "notification_notifier_from_environment": (
        "gpu_fault.notifications.ses",
        "notification_notifier_from_environment",
    ),
}

__getattr__, __dir__, __all__ = lazy_module(globals(), _EXPORTS)
