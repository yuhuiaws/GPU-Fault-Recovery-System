"""Read-only synthetic isolation for the deployed DESTR-006/007 guard probes."""

ISOLATED_NODE_READER = r"""
from types import SimpleNamespace

from gpu_fault.adapters.common import (
    ANNOTATION_FENCING,
    ANNOTATION_INCIDENT,
    QUARANTINE_TAINT,
    quarantine_taint_value,
)


class IsolatedNodeReader:
    def __init__(self, incident_id, fencing_token, target_nodes):
        self.reads = []
        self.target_nodes = frozenset(target_nodes)
        self.node = {
            "metadata": {
                "resourceVersion": "deployed-probe",
                "annotations": {
                    ANNOTATION_INCIDENT: incident_id,
                    ANNOTATION_FENCING: str(fencing_token),
                },
            },
            "spec": {
                "unschedulable": True,
                "taints": [{
                    "key": QUARANTINE_TAINT,
                    "value": quarantine_taint_value(incident_id),
                    "effect": "NoSchedule",
                }],
            },
        }

    def read_node(self, name):
        if name not in self.target_nodes:
            raise KeyError("isolation probe cannot read an unrelated node")
        self.reads.append(name)
        return self.node


def isolated_kubernetes_adapter(incident_id, fencing_token, target_nodes):
    reader = IsolatedNodeReader(incident_id, fencing_token, target_nodes)
    return SimpleNamespace(core=reader, owner="gpu-fault-kubernetes-adapter"), reader
"""
