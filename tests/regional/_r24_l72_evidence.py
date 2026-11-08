"""R24 L72 evidence (GF-GENERAL-LIVE-013.post_repair_isolation_lifted @hp,
attempt 1): the product records of the rung read and S72's cordon-hold record,
minimized to the fields the grading reads (identities of executors, agents and
keys, and every incident reason, are dropped; the node's instance id is the
synthetic one the public-release check accepts)."""

from __future__ import annotations

from typing import Any

NODE = "hyperpod-i-00000000000000001"
BASELINE_BOOT = "f0fe4160-7ffd-4c54-aee1-19031e5e93cc"
BOOT_AFTER = "8f1c23af-65ff-4bcf-ba57-d3abbd34fa02"
OBSERVED: dict[str, Any] = {
    "incidents": [
        {
            "created_at": "2026-11-07T17:04:39.271048Z",
            "event_id": "reboot-after-workflow-2a6df2454fcf9e861eab982b",
            "event_type": "NODE_HEALTH",
            "incident_id": "inc-reboot-after-workflow-2a6df2454fcf9e861eab982b",
            "node_ids": ["hyperpod-i-00000000000000001"],
            "reasons": [],
            "state": "ACTION_PENDING",
            "workflow_request_id": "workflow-reboot-after-workflow-2a6df2454fcf9e861eab982b",
        },
        {
            "created_at": "2026-11-07T17:04:16.950870Z",
            "event_id": "host-hyperpod-i-00000000000000001-1794071056568950-efa_inventory_mismatch-node",
            "event_type": "NODE_HEALTH",
            "incident_id": "inc-host-hyperpod-i-00000000000000001-1794071056568950-efa_inventory_mismatch-node",
            "node_ids": ["hyperpod-i-00000000000000001"],
            "reasons": [],
            "state": "QUARANTINED",
            "workflow_request_id": "workflow-2a6df2454fcf9e861eab982b",
        },
    ],
    "workflows": [
        {
            "created_at": "2026-11-07T17:04:39.271048Z",
            "events": [
                {
                    "at": "2026-11-07T17:04:39.415941Z",
                    "code": "CLAIMED",
                    "kind": "CLAIM",
                    "operation": None,
                    "status": "RUNNING",
                    "step_index": None,
                },
                {
                    "at": "2026-11-07T17:04:39.425500Z",
                    "code": "STEP_SUCCEEDED",
                    "kind": "STEP_ATTEMPT",
                    "operation": "FREEZE_EVIDENCE",
                    "status": "SUCCEEDED",
                    "step_index": 0,
                },
                {
                    "at": "2026-11-07T17:04:39.443860Z",
                    "code": "STEP_WAITING",
                    "kind": "STEP_ATTEMPT",
                    "operation": "MARK_UNSCHEDULABLE",
                    "status": "WAITING",
                    "step_index": 1,
                },
                {
                    "at": "2026-11-07T17:04:39.724833Z",
                    "code": "STEP_SUCCEEDED",
                    "kind": "STEP_ATTEMPT",
                    "operation": "MARK_UNSCHEDULABLE",
                    "status": "SUCCEEDED",
                    "step_index": 1,
                },
                {
                    "at": "2026-11-07T17:04:39.744724Z",
                    "code": "STEP_WAITING",
                    "kind": "STEP_ATTEMPT",
                    "operation": "QUARANTINE",
                    "status": "WAITING",
                    "step_index": 2,
                },
                {
                    "at": "2026-11-07T17:04:39.948331Z",
                    "code": "STEP_SUCCEEDED",
                    "kind": "STEP_ATTEMPT",
                    "operation": "QUARANTINE",
                    "status": "SUCCEEDED",
                    "step_index": 2,
                },
                {
                    "at": "2026-11-07T17:04:39.965885Z",
                    "code": "STEP_WAITING",
                    "kind": "STEP_ATTEMPT",
                    "operation": "RESTART_NODE",
                    "status": "WAITING",
                    "step_index": 3,
                },
            ],
            "incident_id": "inc-reboot-after-workflow-2a6df2454fcf9e861eab982b",
            "official_action": "REBOOT_NODE",
            "official_steps": [
                {
                    "node_ids": ["hyperpod-i-00000000000000001"],
                    "operation": "FREEZE_EVIDENCE",
                    "parameters": {},
                    "workload_ids": [],
                },
                {
                    "node_ids": ["hyperpod-i-00000000000000001"],
                    "operation": "MARK_UNSCHEDULABLE",
                    "parameters": {},
                    "workload_ids": [],
                },
                {
                    "node_ids": ["hyperpod-i-00000000000000001"],
                    "operation": "QUARANTINE",
                    "parameters": {},
                    "workload_ids": [],
                },
                {
                    "node_ids": ["hyperpod-i-00000000000000001"],
                    "operation": "RESTART_NODE",
                    "parameters": {},
                    "workload_ids": [],
                },
                {
                    "node_ids": ["hyperpod-i-00000000000000001"],
                    "operation": "VALIDATE_GPU",
                    "parameters": {},
                    "workload_ids": [],
                },
                {
                    "node_ids": ["hyperpod-i-00000000000000001"],
                    "operation": "VALIDATE_HOST",
                    "parameters": {},
                    "workload_ids": [],
                },
                {
                    "node_ids": ["hyperpod-i-00000000000000001"],
                    "operation": "VALIDATE_FABRIC",
                    "parameters": {},
                    "workload_ids": [],
                },
                {
                    "node_ids": ["hyperpod-i-00000000000000001"],
                    "operation": "RESTORE_SCHEDULING",
                    "parameters": {},
                    "workload_ids": [],
                },
            ],
            "predecessor_workflow_id": None,
            "request_id": "workflow-reboot-after-workflow-2a6df2454fcf9e861eab982b",
            "status": "RUNNING",
            "step_executions": [
                {
                    "details": {},
                    "error": None,
                    "operation": "FREEZE_EVIDENCE",
                    "started_at": "2026-11-07T17:04:39.425529Z",
                    "status": "SUCCEEDED",
                    "step_index": 0,
                    "updated_at": "2026-11-07T17:04:39.425538Z",
                },
                {
                    "details": {
                        "isolated_nodes": ["hyperpod-i-00000000000000001"],
                        "node_baselines": {
                            "hyperpod-i-00000000000000001": {
                                "after": {
                                    "resource_version": "182827275",
                                    "taint_keys": [
                                        "gpu-fault.io/quarantined",
                                        "node.kubernetes.io/unschedulable",
                                    ],
                                    "unschedulable": True,
                                },
                                "before": {
                                    "resource_version": "182827070",
                                    "taint_keys": [
                                        "gpu-fault.io/quarantined",
                                        "node.kubernetes.io/unschedulable",
                                    ],
                                    "unschedulable": True,
                                },
                            }
                        },
                    },
                    "error": None,
                    "operation": "MARK_UNSCHEDULABLE",
                    "started_at": "2026-11-07T17:04:39.443887Z",
                    "status": "SUCCEEDED",
                    "step_index": 1,
                    "updated_at": "2026-11-07T17:04:39.724865Z",
                },
                {
                    "details": {
                        "isolated_nodes": ["hyperpod-i-00000000000000001"],
                        "node_baselines": {
                            "hyperpod-i-00000000000000001": {
                                "after": {
                                    "resource_version": "182827275",
                                    "taint_keys": [
                                        "gpu-fault.io/quarantined",
                                        "node.kubernetes.io/unschedulable",
                                    ],
                                    "unschedulable": True,
                                },
                                "before": {
                                    "resource_version": "182827275",
                                    "taint_keys": [
                                        "gpu-fault.io/quarantined",
                                        "node.kubernetes.io/unschedulable",
                                    ],
                                    "unschedulable": True,
                                },
                            }
                        },
                    },
                    "error": None,
                    "operation": "QUARANTINE",
                    "started_at": "2026-11-07T17:04:39.744753Z",
                    "status": "SUCCEEDED",
                    "step_index": 2,
                    "updated_at": "2026-11-07T17:04:39.948365Z",
                },
                {
                    "details": {
                        "action": "REBOOT",
                        "isolation_reasserted": ["hyperpod-i-00000000000000001"],
                        "observed_isolation": {
                            "hyperpod-i-00000000000000001": {
                                "fencing_token": "1",
                                "incident": "inc-reboot-after-workflow-2a6df2454fcf9e861eab982b",
                                "kubernetes_node": "hyperpod-i-00000000000000001",
                                "resource_version": "182827275",
                                "unschedulable": True,
                            }
                        },
                        "outcome_unknown": True,
                        "remote_status": "WAITING",
                        "requires_external_confirmation": True,
                        "submitted_nodes": ["98519593-5e91-4217-85e3-0d3e789e0cc8"],
                    },
                    "error": None,
                    "operation": "RESTART_NODE",
                    "started_at": "2026-11-07T17:04:39.965923Z",
                    "status": "WAITING",
                    "step_index": 3,
                    "updated_at": "2026-11-07T17:05:20.156013Z",
                },
            ],
        },
        {
            "created_at": "2026-11-07T17:04:16.950870Z",
            "events": [
                {
                    "at": "2026-11-07T17:04:17.120392Z",
                    "code": "CLAIMED",
                    "kind": "CLAIM",
                    "operation": None,
                    "status": "RUNNING",
                    "step_index": None,
                },
                {
                    "at": "2026-11-07T17:04:17.131375Z",
                    "code": "STEP_SUCCEEDED",
                    "kind": "STEP_ATTEMPT",
                    "operation": "FREEZE_EVIDENCE",
                    "status": "SUCCEEDED",
                    "step_index": 0,
                },
                {
                    "at": "2026-11-07T17:04:17.153473Z",
                    "code": "STEP_WAITING",
                    "kind": "STEP_ATTEMPT",
                    "operation": "MARK_UNSCHEDULABLE",
                    "status": "WAITING",
                    "step_index": 1,
                },
                {
                    "at": "2026-11-07T17:04:17.315268Z",
                    "code": "STEP_SUCCEEDED",
                    "kind": "STEP_ATTEMPT",
                    "operation": "MARK_UNSCHEDULABLE",
                    "status": "SUCCEEDED",
                    "step_index": 1,
                },
                {
                    "at": "2026-11-07T17:04:17.332437Z",
                    "code": "STEP_WAITING",
                    "kind": "STEP_ATTEMPT",
                    "operation": "REMEDIATE_EFA_DRIVER",
                    "status": "WAITING",
                    "step_index": 2,
                },
                {
                    "at": "2026-11-07T17:04:31.213437Z",
                    "code": "STEP_SUCCEEDED",
                    "kind": "STEP_ATTEMPT",
                    "operation": "REMEDIATE_EFA_DRIVER",
                    "status": "SUCCEEDED",
                    "step_index": 2,
                },
                {
                    "at": "2026-11-07T17:04:31.232738Z",
                    "code": "STEP_WAITING",
                    "kind": "STEP_ATTEMPT",
                    "operation": "RESTART_EFA_DEVICE_PLUGIN",
                    "status": "WAITING",
                    "step_index": 3,
                },
                {
                    "at": "2026-11-07T17:04:31.431369Z",
                    "code": "STEP_SUCCEEDED",
                    "kind": "STEP_ATTEMPT",
                    "operation": "RESTART_EFA_DEVICE_PLUGIN",
                    "status": "SUCCEEDED",
                    "step_index": 3,
                },
                {
                    "at": "2026-11-07T17:04:31.452786Z",
                    "code": "STEP_WAITING",
                    "kind": "STEP_ATTEMPT",
                    "operation": "TRIGGER_HEALTH_SNAPSHOT",
                    "status": "WAITING",
                    "step_index": 4,
                },
                {
                    "at": "2026-11-07T17:04:34.196036Z",
                    "code": "STEP_SUCCEEDED",
                    "kind": "STEP_ATTEMPT",
                    "operation": "TRIGGER_HEALTH_SNAPSHOT",
                    "status": "SUCCEEDED",
                    "step_index": 4,
                },
                {
                    "at": "2026-11-07T17:04:34.208860Z",
                    "code": "STEP_WAITING",
                    "kind": "STEP_ATTEMPT",
                    "operation": "VALIDATE_FABRIC",
                    "status": "WAITING",
                    "step_index": 5,
                },
                {
                    "at": "2026-11-07T17:04:39.243856Z",
                    "code": "STEP_FAILED",
                    "kind": "STEP_ATTEMPT",
                    "operation": "VALIDATE_FABRIC",
                    "status": "FAILED",
                    "step_index": 5,
                },
                {
                    "at": "2026-11-07T17:04:39.246230Z",
                    "code": "TERMINALIZED",
                    "kind": "TERMINAL",
                    "operation": None,
                    "status": "FAILED",
                    "step_index": None,
                },
            ],
            "incident_id": "inc-host-hyperpod-i-00000000000000001-1794071056568950-efa_inventory_mismatch-node",
            "official_action": "REMEDIATE_EFA_DRIVER",
            "official_steps": [
                {
                    "node_ids": ["hyperpod-i-00000000000000001"],
                    "operation": "FREEZE_EVIDENCE",
                    "parameters": {},
                    "workload_ids": [],
                },
                {
                    "node_ids": ["hyperpod-i-00000000000000001"],
                    "operation": "MARK_UNSCHEDULABLE",
                    "parameters": {},
                    "workload_ids": [],
                },
                {
                    "node_ids": ["hyperpod-i-00000000000000001"],
                    "operation": "REMEDIATE_EFA_DRIVER",
                    "parameters": {},
                    "workload_ids": [],
                },
                {
                    "node_ids": ["hyperpod-i-00000000000000001"],
                    "operation": "RESTART_EFA_DEVICE_PLUGIN",
                    "parameters": {},
                    "workload_ids": [],
                },
                {
                    "node_ids": ["hyperpod-i-00000000000000001"],
                    "operation": "TRIGGER_HEALTH_SNAPSHOT",
                    "parameters": {},
                    "workload_ids": [],
                },
                {
                    "node_ids": ["hyperpod-i-00000000000000001"],
                    "operation": "VALIDATE_FABRIC",
                    "parameters": {},
                    "workload_ids": [],
                },
                {
                    "node_ids": ["hyperpod-i-00000000000000001"],
                    "operation": "RESTORE_SCHEDULING",
                    "parameters": {},
                    "workload_ids": [],
                },
            ],
            "predecessor_workflow_id": None,
            "request_id": "workflow-2a6df2454fcf9e861eab982b",
            "status": "FAILED",
            "step_executions": [
                {
                    "details": {},
                    "error": None,
                    "operation": "FREEZE_EVIDENCE",
                    "started_at": "2026-11-07T17:04:17.131402Z",
                    "status": "SUCCEEDED",
                    "step_index": 0,
                    "updated_at": "2026-11-07T17:04:17.131410Z",
                },
                {
                    "details": {
                        "isolated_nodes": ["hyperpod-i-00000000000000001"],
                        "node_baselines": {
                            "hyperpod-i-00000000000000001": {
                                "after": {
                                    "resource_version": "182827069",
                                    "taint_keys": ["gpu-fault.io/quarantined"],
                                    "unschedulable": True,
                                },
                                "before": {
                                    "resource_version": "182825447",
                                    "taint_keys": [],
                                    "unschedulable": False,
                                },
                            }
                        },
                    },
                    "error": None,
                    "operation": "MARK_UNSCHEDULABLE",
                    "started_at": "2026-11-07T17:04:17.153500Z",
                    "status": "SUCCEEDED",
                    "step_index": 1,
                    "updated_at": "2026-11-07T17:04:17.315300Z",
                },
                {
                    "details": {},
                    "error": None,
                    "operation": "REMEDIATE_EFA_DRIVER",
                    "started_at": "2026-11-07T17:04:17.332464Z",
                    "status": "SUCCEEDED",
                    "step_index": 2,
                    "updated_at": "2026-11-07T17:04:31.213473Z",
                },
                {
                    "details": {},
                    "error": None,
                    "operation": "RESTART_EFA_DEVICE_PLUGIN",
                    "started_at": "2026-11-07T17:04:31.232767Z",
                    "status": "SUCCEEDED",
                    "step_index": 3,
                    "updated_at": "2026-11-07T17:04:31.431401Z",
                },
                {
                    "details": {},
                    "error": None,
                    "operation": "TRIGGER_HEALTH_SNAPSHOT",
                    "started_at": "2026-11-07T17:04:31.452815Z",
                    "status": "SUCCEEDED",
                    "step_index": 4,
                    "updated_at": "2026-11-07T17:04:34.196074Z",
                },
                {
                    "details": {},
                    "error": "validation failed on nodes: "
                    "hyperpod-i-00000000000000001 "
                    "(efa_inventory_active_count=15,expected=16)",
                    "operation": "VALIDATE_FABRIC",
                    "started_at": "2026-11-07T17:04:34.208888Z",
                    "status": "FAILED",
                    "step_index": 5,
                    "updated_at": "2026-11-07T17:04:39.243894Z",
                },
            ],
        },
    ],
}
HOLD: dict[str, Any] = {
    "last": {
        "boot_id": "f0fe4160-7ffd-4c54-aee1-19031e5e93cc",
        "incident": "inc-reboot-after-workflow-2a6df2454fcf9e861eab982b",
        "unschedulable": True,
    },
    "lifts": [
        {
            "lifted_at": "2026-11-07T17:04:41.435110+00:00",
            "node": "hyperpod-i-00000000000000001",
            "patch": {"spec": {"unschedulable": False}},
        },
        {
            "lifted_at": "2026-11-07T17:04:56.123515+00:00",
            "node": "hyperpod-i-00000000000000001",
            "patch": {"spec": {"unschedulable": False}},
        },
        {
            "lifted_at": "2026-11-07T17:05:07.476015+00:00",
            "node": "hyperpod-i-00000000000000001",
            "patch": {"spec": {"unschedulable": False}},
        },
        {
            "lifted_at": "2026-11-07T17:05:10.614414+00:00",
            "node": "hyperpod-i-00000000000000001",
            "patch": {"spec": {"unschedulable": False}},
        },
        {
            "lifted_at": "2026-11-07T17:05:13.794352+00:00",
            "node": "hyperpod-i-00000000000000001",
            "patch": {"spec": {"unschedulable": False}},
        },
    ],
    "stopped": "lift_cap",
    "successor_incident": "inc-reboot-after-workflow-2a6df2454fcf9e861eab982b",
}
