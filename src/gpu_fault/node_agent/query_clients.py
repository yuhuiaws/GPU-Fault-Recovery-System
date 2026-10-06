"""Device holders that only read the GPU: ``nvidia-smi`` and its kind.

``nvidia-smi`` goes through NVML (``/dev/nvidiactl`` + ``/dev/nvidiaN``) and
never opens ``/dev/nvidia-uvm``, which every CUDA context needs. Live
(gpu-eks-cluster, XID 46): with persistence mode off -- the quiesce had stopped
``nvidia-persistenced`` -- one ``nvidia-smi`` query by the acceptance sampler
took long enough to be seen in both 0.5 s final samples, the reset was refused
``OWNERSHIP_FINAL_CLIENTS_CHANGED`` on a GPU nobody computed on, and the chain
parked for an operator. Stdlib only: the ownership boundary and the client
scan both import it, and the protocol module sits between them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: Process names (``comm``) a read-only NVML query runs under.
QUERY_ONLY_PROCESS_NAMES = frozenset({"nvidia-smi"})
#: Opening this node means a CUDA context, whatever the process is called.
UVM_DEVICE_PREFIX = "/dev/nvidia-uvm"


def query_only_device_client(
    proc_root: Path | str | None, client: Mapping[str, Any]
) -> bool:
    """Whether ``client`` is a read-only NVML query rather than a GPU user.

    True only on positive evidence: the process is one of
    ``QUERY_ONLY_PROCESS_NAMES`` *and* its descriptor table could be read in
    full and names no ``/dev/nvidia-uvm`` node. A process that is gone, that
    cannot be read, or that holds the UVM node is not query-only: the reset
    keeps refusing on it.
    """

    if proc_root is None or str(client.get("process_name", "")) not in (
        QUERY_ONLY_PROCESS_NAMES
    ):
        return False
    try:
        descriptors = list(
            (Path(proc_root) / str(client.get("pid", "")) / "fd").iterdir()
        )
    except OSError:
        return False
    for descriptor in descriptors:
        try:
            target = os.readlink(descriptor)
        except OSError:
            return False
        if target.startswith(UVM_DEVICE_PREFIX):
            return False
    return True
