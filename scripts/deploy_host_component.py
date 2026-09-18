from __future__ import annotations

DEPLOY_HOST_COMPONENT_SPEC = {
    "distribution": "gpu-fault-deploy-host",
    "roots": (
        "gpu_fault.admin.cli",
        "gpu_fault_release.regional_node_batch",
        # The cleanup scripts are assets outside the package import walk.
        "gpu_fault.admin.cluster_removal_rbac",
    ),
    "scripts": {
        "gpu-fault-admin": "gpu_fault.admin.cli:main",
        "gpu-training-submit": "gpu_fault.training_submit_cli:main",
        "gpu-fault-workload-annotate": "gpu_fault.workload_annotate_cli:main",
    },
    "entry_points": {},
    "data_globs": ("*.json", "*.yaml", "*.yml"),
}
