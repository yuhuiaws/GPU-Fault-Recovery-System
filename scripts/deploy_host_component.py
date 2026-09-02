from __future__ import annotations

DEPLOY_HOST_COMPONENT_SPEC = {
    "distribution": "gpu-fault-deploy-host",
    "roots": ("gpu_fault.admin_cli",),
    "scripts": {"gpu-fault-admin": "gpu_fault.admin_cli:main"},
    "entry_points": {},
    "data_globs": ("*.json", "*.yaml", "*.yml"),
}
