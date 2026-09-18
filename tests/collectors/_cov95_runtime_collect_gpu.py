"""Fake nvidia-smi inventory, metrics and XML command responses."""

from __future__ import annotations

import subprocess


def gpu_runner(argv, **kwargs):
    if "-x" in argv:
        text = (
            "<nvidia_smi_log><gpu><temperature>"
            "<gpu_temp_slow_threshold>85 C</gpu_temp_slow_threshold>"
            "</temperature></gpu></nvidia_smi_log>"
        )
    elif "--query-gpu=index,uuid,pci.bus_id,name" in argv:
        text = "0,GPU-a,0000:01:00.0,H100\n"
    else:
        fields = argv[1].partition("=")[2].split(",")
        identity = {
            "index": "0",
            "uuid": "GPU-a",
            "name": "H100",
            "pci.bus_id": "0000:01:00.0",
        }
        text = ",".join(identity.get(field, "1") for field in fields)
    return subprocess.CompletedProcess(argv, 0, stdout=text, stderr="")
