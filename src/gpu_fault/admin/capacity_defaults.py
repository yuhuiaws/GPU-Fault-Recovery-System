"""Control-plane shape constants shared by the admin config and its parser.

A leaf module on purpose: ``config_parser`` needs the control-worker default
and ``config`` needs the parser, so the constants cannot live in either.
"""

from __future__ import annotations

# Processes = ``replicas x UVICORN_WORKERS_PER_POD``. Every tier but the
# telemetry spool runs ``uvicorn --workers 4`` behind one port; /metrics is
# aggregated over the Pod's processes by ``gpu_fault.app.process_metrics``.
# Kept in step with the renderer
# (deploy/control-plane/tools/render_control_plane_role_split.py) and the
# base manifest's replica count.
UVICORN_WORKERS_PER_POD = 4
INGRESS_REPLICAS = 3
DEFAULT_CONTROL_WORKER_REPLICAS = 6
