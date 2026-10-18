"""A reset that left a node's display stack unloaded says so in its completion mail.

The Agent reloads the ``nvidia_drm`` / ``nvidia_modeset`` a busy reset
unloaded, retries once and reports ``display_stack_restored`` per node
(``node_agent/operations/display_stack.py``). The reset is SUCCEEDED either
way and nothing is drained, quarantined or reset for a display stack, so the
operator reads it in the mail every successful reset already sends
(GPU_RESET_COMPLETED, FABRIC_RESET_COMPLETED): a subject marker and, per node,
what is missing, the last error and the exact remedy. A stack that is back, or
a reset that released nothing, leaves the mail as it was.
"""

from __future__ import annotations

from typing import Any

from gpu_fault.models import NotificationStatus
from gpu_fault.notifications.common import (
    FABRIC_RESET_TEMPLATE_VERSION,
    GPU_RESET_TEMPLATE_VERSION,
)
from gpu_fault.notifications.display_stack import (
    DISPLAY_STACK_SUBJECT_MARKER,
    display_stack_follow_up,
    unrestored_display_stacks,
)
from gpu_fault.notifications.restart_guard import RestartGuardEmailBuilder
from gpu_fault.regional import RemoteCommandStatus
from tests._builders import build_store
from tests.notifications.test_batched_remote_completion import (
    QUIESCE_NODE_RESULTS,
    _compound_command,
    _entry,
    _service,
)

ERROR = (
    "RuntimeError: modprobe exited with status 1: modprobe: ERROR: could not "
    "insert 'nvidia_drm': No such device"
)
NOT_RESTORED = {
    "status": "SUCCEEDED",
    "reset_gpu_uuids": ["GPU-a"],
    "released_kernel_modules": ["nvidia_drm", "nvidia_modeset"],
    "display_module_reload_attempts": [
        {"attempt": 1, "command": "modprobe nvidia_drm", "outcome": "failed"},
        {"attempt": 2, "command": "modprobe nvidia_drm", "outcome": "failed"},
    ],
    "display_stack_restored": False,
    "display_modules_missing": ["nvidia_drm", "nvidia_modeset"],
    "kernel_module_reload_error": ERROR,
    "display_stack_remedy": "modprobe nvidia_drm, or reboot the node",
}
RESTORED = {
    "status": "SUCCEEDED",
    "reset_gpu_uuids": ["GPU-a"],
    "released_kernel_modules": ["nvidia_drm", "nvidia_modeset"],
    "reloaded_kernel_modules": ["nvidia_modeset", "nvidia_drm"],
    "display_stack_restored": True,
}


def _gpu_mail(node_results: dict[str, Any]):
    return RestartGuardEmailBuilder().build_gpu_reset_completed(
        cluster_id="gpu-eks-cluster",
        incident_id="incident-a",
        workflow_id="workflow-a",
        event_id="xid-46",
        event_type="XID",
        policy_source="NVIDIA_CATALOG",
        official_action="RESET_GPU",
        reasons=["XID 46"],
        operation_id="workflow-a/4/RESET_GPU",
        node_ids=sorted(node_results),
        gpu_uuids=["GPU-a"],
        node_results=node_results,
        workload_ids=[],
    )


def _fabric_mail(node_results: dict[str, Any]):
    return RestartGuardEmailBuilder().build_fabric_reset_completed(
        cluster_id="gpu-eks-cluster",
        incident_id="incident-sxid",
        workflow_id="workflow-sxid",
        event_id="sxid-1",
        event_type="SXID",
        policy_source="NVIDIA_FABRIC_MANAGER",
        official_action="RESET_ALL_GPUS_AND_NVSWITCHES",
        reasons=["fatal trunk-link SXID"],
        operation_id="workflow-sxid/5/RESET_ALL_GPUS_NVSWITCHES",
        node_results=node_results,
        workload_ids=[],
        fabric_partition="gpu-eks-cluster/node-a/local-nvswitch",
        sxid=10003,
    )


def test_an_unrestored_stack_puts_the_remedy_in_the_gpu_reset_mail() -> None:
    notification = _gpu_mail({"node-a": NOT_RESTORED})

    assert notification.subject == (
        f"[NOTICE][GPU reset automatically]{DISPLAY_STACK_SUBJECT_MARKER} "
        "gpu-eks-cluster: GPU-a"
    )
    body = notification.body_text
    assert (
        "- Display stack: NOT restored on node-a: nvidia_drm, nvidia_modeset not "
        "loaded after 2 modprobe attempt(s); last error: " + ERROR
    ) in body
    assert "Impact: no /dev/dri on node-a for graphics, EGL or Vulkan" in body
    assert "CUDA is unaffected; nothing is drained, quarantined or reset" in body
    assert "  Remedy on node-a: modprobe nvidia_drm, or reboot the node\n" in body
    assert body.index("3. Follow-up") < body.index("Display stack"), body
    assert f"\n\nTemplate: {GPU_RESET_TEMPLATE_VERSION}" in body
    assert notification.category == "ACTION_COMPLETED", (
        "the reset completed; the mail category and its acceptance check stay"
    )
    assert notification.deduplication_key == (
        "gpu-eks-cluster/incident-a/gpu-reset/workflow-a/4/RESET_GPU"
    ), "one mail per reset, as before"


def test_a_restored_or_untouched_stack_leaves_the_mail_as_it_was() -> None:
    for results in (
        {"node-a": RESTORED},
        {"node-a": {"status": "SUCCEEDED", "reset_gpu_uuids": ["GPU-a"]}},
    ):
        notification = _gpu_mail(results)

        assert notification.subject == (
            "[NOTICE][GPU reset automatically] gpu-eks-cluster: GPU-a"
        )
        assert "Display stack" not in notification.body_text
        assert (
            f"after all validation passes\n\nTemplate: {GPU_RESET_TEMPLATE_VERSION}\n"
        ) in notification.body_text, "the follow-up block reads as before"
    assert display_stack_follow_up({"node-a": RESTORED}) == ""


def test_an_older_agents_reload_error_reads_the_same_way() -> None:
    """An Agent that predates the check reports only the error; the control
    plane rolls first, so its mail must already say so."""

    legacy = {
        "status": "SUCCEEDED",
        "released_kernel_modules": ["nvidia_drm", "nvidia_modeset"],
        "kernel_module_reload_error": "RuntimeError: modprobe exited with status 1",
    }
    notification = _gpu_mail({"node-a": legacy})

    assert DISPLAY_STACK_SUBJECT_MARKER in notification.subject, notification.subject
    assert "nvidia_drm, nvidia_modeset not loaded after 1 modprobe attempt(s)" in (
        notification.body_text
    )
    assert "Remedy on node-a: modprobe nvidia_drm, or reboot the node" in (
        notification.body_text
    )
    assert list(unrestored_display_stacks({"node-a": legacy})) == ["node-a"]
    assert unrestored_display_stacks({"node-a": {**legacy, **RESTORED}}) == {}, (
        "an explicit restored verdict wins over an error of an earlier attempt"
    )


def test_the_fabric_reset_mail_names_only_the_node_whose_stack_is_missing() -> None:
    notification = _fabric_mail(
        {
            "node-a": {**NOT_RESTORED, "reset_scope": "ALL_LOCAL_GPUS_AND_NVSWITCHES"},
            "node-b": {**RESTORED, "reset_scope": "ALL_LOCAL_GPUS_AND_NVSWITCHES"},
        }
    )

    assert notification.subject == (
        f"[NOTICE][GPU/NVSwitch reset automatically]{DISPLAY_STACK_SUBJECT_MARKER} "
        "gpu-eks-cluster: node-a, node-b"
    )
    assert "Display stack: NOT restored on node-a" in notification.body_text
    assert "node-b:" not in notification.body_text.split("3. Follow-up")[1]
    assert f"Template: {FABRIC_RESET_TEMPLATE_VERSION}" in notification.body_text

    clean = _fabric_mail({"node-b": RESTORED})
    assert DISPLAY_STACK_SUBJECT_MARKER not in clean.subject, clean.subject
    assert "Display stack" not in clean.body_text


def test_the_regional_completion_of_a_batched_reset_mails_the_remedy() -> None:
    """The live path: the RESET_GPU passenger of the QUIESCE carrier, its own
    node results, one mail when the carrier is terminal."""

    store = build_store()
    service, notifier = _service(store)
    command = _compound_command(
        store,
        batched_results={
            2: _entry(RemoteCommandStatus.SUCCEEDED, QUIESCE_NODE_RESULTS),
            3: _entry(
                RemoteCommandStatus.SUCCEEDED,
                {"node-a": {"verified_no_gpu_clients": True}},
            ),
            4: _entry(RemoteCommandStatus.SUCCEEDED, {"node-a": NOT_RESTORED}),
            5: _entry(RemoteCommandStatus.SUCCEEDED, {"node-a": {"restored": True}}),
        },
    )

    results = service.dispatch_remote_completion(command)

    assert [item.status for item in results] == [NotificationStatus.SENT], results
    [notification] = store.list_notifications()
    assert DISPLAY_STACK_SUBJECT_MARKER in notification.subject, notification.subject
    assert "Remedy on node-a: modprobe nvidia_drm, or reboot the node" in (
        notification.body_text
    )
    assert notification.category == "ACTION_COMPLETED"
    assert len(notifier.notifications) == 1
