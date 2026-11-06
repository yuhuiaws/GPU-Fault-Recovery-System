"""Pin the EFA/RDMA kernel-log vocabulary the RDMA node log rules grade.

Every line below is text the Linux EFA driver prints, built from the format
strings quoted in :mod:`gpu_fault.efa_kernel_log` (Linux v6.1.189 and
v6.12.103 ``drivers/infiniband/hw/efa/``, the amzn-drivers out-of-tree ``efa``
at tags ``efa_linux_3.3.1`` -- the ``extra/efa.ko`` 3.3.1g the nodes load --
and ``efa_linux_3.4.0``) through ``__ibdev_printk()``
(``drivers/infiniband/core/device.c``, ``"%s %s %s: %pV"`` = ``efa <pci bdf>
<ib device>: <message>``; the ib device is ``efa_N`` as registered and
``rdmap<bus>s<slot>`` once rdma-core's udev rule renamed it -- what every
runtime line of a live node carries; an ib device that is not registered yet
prints ``(null): <message>``) or ``dev_err()`` on the PCI device (``efa <pci
bdf>: <message>``). A change that regrades one of them must come with a new
kernel quote, not a new opinion.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.efa_kernel_log import (
    AMZN,
    AMZN_DRIVERS_EFA_COMMITS,
    AMZN_DRIVERS_EFA_SOURCES,
    EFA_DEGRADED_LINES,
    EFA_FATAL_LINES,
    EFA_INFORMATIONAL_LINES,
    EFA_LINUX_SOURCES,
    EFA_VOCABULARY,
    ERRNO_SOURCE,
    ETIME,
    IBDEV_PRINTK_FORMAT,
    IBDEV_PRINTK_SOURCE,
    LINUX_6_1,
    LINUX_6_12,
    RDMA_CORE_RENAME_RULE,
    RDMA_CORE_RENAME_SOURCES,
    EfaKernelLine,
    EfaLineGrade,
    grade_efa_line,
)
from gpu_fault.host_health import NodeHealthPolicy, NodeLogBatch, NodeLogEntry
from gpu_fault.log_rules import (
    LogHealthCategory,
    log_signal_priority,
    matching_log_rules,
)
from gpu_fault.models import RecoveryAction, Severity
from tests._builders import build_store

NOW = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
#: The name the driver registers (``ib_register_device(..., "efa_%d", ...)``).
DEV = "efa 0000:00:1d.0 efa_0: "
#: The name every runtime line of a live node carries: rdma-core's udev rule
#: (``rdma_rename %k NAME_FALLBACK``) renames the device after its PCI
#: location, ``rdma`` + ``p<bus>s<slot>`` (bus 0x55 = 85) -- the HyperPod
#: GL-013 driver survey's ``0000:55:00.0`` -> ``rdmap85s0``.
LIVE_DEV = "efa 0000:55:00.0 rdmap85s0: "
#: ``dev_err(&pdev->dev, ...)``: driver and PCI address only.
PCI_DEV = "efa 0000:55:00.0: "

#: The GF-GENERAL-LIVE-016 ``rdma_fatal`` reviewed line (b251 harness): what
#: amzn-drivers 3.3.1g's ``efa_com_cmd_exec()`` prints after an admin command
#: timed out (``err = -ETIME``), on the node's own BDF and renamed device.
LIVE_016_RDMA_FATAL = (
    LIVE_DEV + "Failed to process command GET_STATS (opcode 13) err -62"
)

#: The device stopped answering its admin queue, the queue is closed, or the
#: reset handshake failed: the node's fabric is gone until the driver reloads.
FATAL_LINES = [
    LIVE_016_RDMA_FATAL,
    # The brief's live form: the queue closed on a renamed device.
    LIVE_DEV + "Admin queue is closed",
    DEV + "Admin queue is closed",
    # ``efa_com_wait_and_process_admin_cq_polling()``: no node runs an admin
    # command in polling mode, but the line still means a silent device.
    DEV + "Wait for completion (polling) timeout",
    DEV + "The device didn't send any completion for admin cmd CREATE_QP(1) "
    "status 0 (ctx 0x00000000a1b2c3d4, sq producer: 12, sq consumer: 11, "
    "cq consumer: 11)",
    # amzn-drivers prints the completion context id instead of the pointer;
    # status 2 is EFA_CMD_SUBMITTED (3.3.1), 4 EFA_CMD_TIMED_OUT (3.4.0).
    DEV + "The device didn't send any completion for admin cmd CREATE_QP(1) "
    "status 0 (id: 7, sq producer: 12, sq consumer: 11, cq consumer: 11)",
    LIVE_DEV + "The device didn't send any completion for admin cmd GET_STATS(13) "
    "status 2 (id: 5, sq producer: 1234, sq consumer: 1233, cq consumer: 1233)",
    DEV + "The device sent a completion but the driver didn't receive any MSI-X "
    "interrupt for admin cmd DESTROY_QP(2) status 1 (ctx: 0x00000000a1b2c3d4, "
    "sq producer: 12, sq consumer: 11, cq consumer: 11)",
    DEV + "Reading register timed out. expected: req id[12] offset[0x58] actual: "
    "req id[11] offset[0x58]",
    # ``efa_com_cmd_exec()``: a submit failure closes the queue whatever the
    # errno; a process failure is the device's silence only with -ETIME (-62).
    DEV + "Failed to submit command CREATE_QP (opcode 1) err -19",
    DEV + "Failed to process command CREATE_QP (opcode 1) comp_status 0 err -62",
    DEV + "Failed to process command CREATE_QP (opcode 1) err -62",
    # amzn-drivers: ``efa_com_cmd_exec()`` closes the queue after this line and
    # prints nothing more.
    DEV + "Completion context[7] is used[1]",
    LIVE_DEV + "Completion context[5] is used[2]",
    # Probe/reset time: the ib device has no name yet, ``__ibdev_printk()``
    # prints ``(null): ``; at remove time the registered name is still there.
    "(null): Device isn't ready, abort com init 0x0",
    "(null): Device isn't ready, can't reset device",
    "(null): Reset indication didn't turn on",
    DEV + "Reset indication didn't turn off",
    LIVE_DEV + "Reset indication didn't turn off",
    # dmesg-style timestamp in front of the kernel line.
    "[ 1234.567890] efa 0000:00:1d.0 efa_0: Admin queue is closed",
    "[ 1234.567890] efa 0000:55:00.0 rdmap85s0: Admin queue is closed",
]

#: The driver reports the device or firmware misbehaving, or the device
#: failing to come up: read-only fabric diagnostics, no isolation.
DEGRADED_LINES = [
    DEV + "Received completion with unexpected command id[7], sq producer: 12, "
    "sq consumer: 11, cq consumer: 11",
    DEV + "Received completion with unexpected command id[7], status[1] "
    "sq producer[12], sq consumer[11], cq consumer[11]",
    LIVE_DEV + "Received completion with unexpected command id[2f], status[0] "
    "sq producer[12], sq consumer[11], cq consumer[11]",
    DEV + "Received completion with invalid checksum, cqe[12], calc[13], "
    "sq producer[12], sq consumer[11], cq consumer[11]",
    # In-tree: the same call prints the fatal "Failed to submit command".
    DEV + "Completion context for command_id 0x7 is occupied",
    DEV + "Completion event on non-existent CQ[42]",
    DEV + "Unknown event type received 9",
    DEV + "Unknown event was received or event with unimplemented handler",
    LIVE_DEV + "Unknown event was received or event with unimplemented handler",
    DEV + "Failed to create eq[-12]",
    DEV + "Failed to destroy EQ-3 [-110]",
    DEV + "Failed to get device attributes -22",
    DEV + "Failed to set aenq attributes: -22",
    DEV + "Trying to set unsupported aenq groups[0x3] supported[0x1]",
    "(null): EFA version is lower than the minimal version the driver supports",
    "(null): DMA width illegal value: 0",
    "(null): Invalid timeout value",
    # ``dev_err(&pdev->dev, ...)`` in efa_main.c: driver and PCI address only.
    "efa 0000:00:1d.0: EFA device init failed",
    "efa 0000:00:1d.0: Failed to enable MSI-X. error -28",
    "efa 0000:00:1d.0: Failed to enable MSI-X. irq_num 2",
    "efa 0000:00:1d.0: Failed to request irq efa-mgmnt@pci:0000:00:1d.0 (-16)",
    "efa 0000:00:1d.0: Doorbells BAR unavailable. Requested 0x4, available 0x1",
    PCI_DEV + "BARs unavailable. Requested 0x5, available 0x1",
    "efa 0000:00:1d.0: Failed to remap register bar",
    "efa 0000:00:1d.0: Failed to allocate QP table",
]

#: Printed in normal operation, or for one command that the device answered
#: with an error (a job asked for a resource the device could not give):
#: never a finding.
INFORMATIONAL_LINES = [
    DEV + "IB device registered",
    DEV + "Unregister ib device",
    LIVE_DEV + "Unregister ib device",
    "efa 0000:00:1d.0: Setup irq:35 name:efa-mgmnt@pci:0000:00:1d.0",
    "Elastic Fabric Adapter (EFA) v3.3.1g",
    "Couldn't register efa driver",
    # ``efa_com_comp_status_to_errno()``: the device answered
    # RESOURCE_ALLOCATION_FAILURE (-ENOMEM) / ILLEGAL_PARAMETER (-EINVAL).
    DEV + "Failed to process command CREATE_QP (opcode 1) comp_status 2 err -12",
    DEV + "Failed to process command CREATE_QP (opcode 1) err -22",
    LIVE_DEV + "Failed to process command CREATE_QP (opcode 1) err -95",
    DEV + "Failed to create qp [-12]",
    DEV + "Failed to create cq[-12]",
    DEV + "Failed to register mr [-12]",
    DEV + "Failed to de-register mr(lkey-4096) [-22]",
    DEV + "Failed to create ah for fe80:0000:0000:0000:0000:0000:0000:0001 [-12]",
    DEV + "Failed to allocate pd[-12]",
    DEV + "Failed to get stats type-0 scope-0.0 [-22]",
    # The per-command line under a timed-out GET_STATS: the fatal lines above
    # it carry the finding.
    LIVE_DEV + "Failed to get stats type-0 scope-0.0 [-62]",
    DEV + "Feature 5 isn't supported",
    DEV + "Failed to submit set_feature command 5 error: -22",
    DEV + "Failed to map DMA address",
    DEV + "chunk[3] dma_map_failed",
    DEV + "Unable to map pbl to DMA address",
    # Not EFA at all: the plugin's name is not the driver's device prefix.
    "Reconcile PyTorchJob gpu-fault-efa-plugin error: object has been modified",
    "aws-efa-k8s-device-plugin: fatal error: link down on efa_0",
]

#: The earlier harness text. The kernel tree never printed it.
PARAPHRASE = "efa: fatal error on device efa_0: link down, resetting"

#: Every way the driver's prefix stands before a graded message: the ib
#: device as registered and as renamed, the PCI device, the unnamed ib device
#: (with and without its parent), and a dmesg timestamp in front.
PREFIXES = [
    DEV,
    LIVE_DEV,
    "efa 0000:6e:00.0 rdmap110s0: ",
    PCI_DEV,
    "(null): ",
    "efa 0000:55:00.0 (null): ",
    "[ 1234.567890] " + LIVE_DEV,
]


def _evaluate(
    message: str,
    *,
    source: str = "dmesg",
    kernel: bool = True,
    fields: dict[str, str] | None = None,
    observed_at: datetime = NOW,
) -> list:
    if fields is None:
        fields = {"_TRANSPORT": "kernel"} if kernel else {"_TRANSPORT": "syslog"}
    return NodeHealthPolicy(build_store()).evaluate_logs(
        NodeLogBatch(
            cluster_id="cluster-a",
            node_id="node-a",
            collected_at=NOW,
            entries=[
                NodeLogEntry(
                    entry_id="efa-line",
                    source=source,
                    observed_at=observed_at,
                    message=message,
                    fields=fields,
                )
            ],
        )
    )


@pytest.mark.parametrize("message", FATAL_LINES)
def test_fatal_efa_lines_quarantine_from_a_trusted_kernel_origin(message: str) -> None:
    assert grade_efa_line(message) is EfaLineGrade.FATAL, message
    rules = matching_log_rules(message, "dmesg")
    assert [(r.category, r.severity, r.action) for r in rules] == [
        (LogHealthCategory.RDMA, Severity.CRITICAL, RecoveryAction.QUARANTINE)
    ], message
    [finding] = _evaluate(message)
    assert (
        finding.category.value,
        finding.severity,
        finding.recommended_action,
        finding.policy_source,
    ) == ("RDMA", Severity.CRITICAL, RecoveryAction.QUARANTINE, "SITE_KERNEL_LOG"), (
        finding
    )
    assert finding.diagnostic_parameters["stop_on_isolation"] is True, finding


@pytest.mark.parametrize("message", DEGRADED_LINES)
def test_degraded_efa_lines_request_fabric_diagnostics_only(message: str) -> None:
    assert grade_efa_line(message) is EfaLineGrade.DEGRADED, message
    rules = matching_log_rules(message, "dmesg")
    assert [(r.category, r.severity, r.action) for r in rules] == [
        (LogHealthCategory.RDMA, Severity.WARNING, RecoveryAction.RUN_DIAGNOSTICS)
    ], message
    [finding] = _evaluate(message)
    assert (
        finding.category.value,
        finding.severity,
        finding.recommended_action,
        finding.policy_source,
    ) == (
        "RDMA",
        Severity.WARNING,
        RecoveryAction.RUN_DIAGNOSTICS,
        "SITE_APPLICATION_LOG",
    ), finding
    assert finding.diagnostic_parameters["stop_on_isolation"] is False, finding


@pytest.mark.parametrize("message", INFORMATIONAL_LINES)
def test_informational_efa_lines_mint_no_finding(message: str) -> None:
    assert grade_efa_line(message) in {EfaLineGrade.INFORMATIONAL, None}, message
    assert matching_log_rules(message) == [], message
    assert _evaluate(message) == [], message


def test_a_paraphrase_is_not_kernel_vocabulary() -> None:
    # The live replay must carry what the kernel prints; the earlier text
    # does not exist in drivers/infiniband/hw/efa and grades nothing.
    assert grade_efa_line(PARAPHRASE) is None, PARAPHRASE
    assert matching_log_rules(PARAPHRASE, "dmesg") == [], PARAPHRASE
    assert _evaluate(PARAPHRASE) == [], PARAPHRASE


@pytest.mark.parametrize(
    "source, kernel",
    [("journal", False), ("training-log", True), ("training-log", False)],
)
def test_fatal_efa_text_from_an_untrusted_origin_is_ignored(
    source: str, kernel: bool
) -> None:
    assert _evaluate(LIVE_016_RDMA_FATAL, source=source, kernel=kernel) == [], (
        source,
        kernel,
    )
    assert matching_log_rules(LIVE_016_RDMA_FATAL, "training-log") == [], (
        "a workload cannot forge an EFA fatal line"
    )


def test_degraded_efa_text_from_a_workload_log_is_ignored() -> None:
    message = DEGRADED_LINES[0]
    assert matching_log_rules(message, "training-log") == [], message
    assert _evaluate(message, source="training-log", kernel=False) == [], message


#: Lines the kernel trust gate (``log_recovery_context.log_rule_decision``)
#: refuses for every rule only a kernel origin may satisfy: a journal line
#: another transport wrote (a host process's ``logger``, a unit's stdout), a
#: line of another boot, one older than the collection window or from its
#: future (``current_kernel_evidence``: 900 s before to 30 s after).
UNTRUSTED_KERNEL_ORIGINS = {
    "syslog-transport": {"source": "journal", "fields": {"_TRANSPORT": "syslog"}},
    "stdout-transport": {"source": "journal", "fields": {"_TRANSPORT": "stdout"}},
    "previous-boot": {
        "fields": {
            "_TRANSPORT": "kernel",
            "_BOOT_ID": "boot-before",
            "_COLLECTOR_BOOT_ID": "boot-now",
        }
    },
    "stale": {"observed_at": NOW - timedelta(seconds=901)},
    "future": {"observed_at": NOW + timedelta(seconds=31)},
}
#: ``efa_com_handle_single_admin_completion()``, amzn-drivers' spelling.
UNEXPECTED_COMPLETION = (
    "Received completion with unexpected command id[2f], status[0] "
    "sq producer[12], sq consumer[11], cq consumer[11]"
)


@pytest.mark.parametrize("prefix", [DEV, LIVE_DEV], ids=["registered", "renamed"])
@pytest.mark.parametrize("origin", sorted(UNTRUSTED_KERNEL_ORIGINS))
def test_degraded_efa_text_passes_the_fatal_lines_kernel_trust_gate(
    prefix: str, origin: str
) -> None:
    # The degraded rule is KERNEL_LOG_SOURCES like the fatal one, but the gate
    # behind the source tag used to run only for QUARANTINE rules: a
    # ``logger`` line in the journal or a previous boot's line opened a fabric
    # diagnostic on the node and its job.
    untrusted = UNTRUSTED_KERNEL_ORIGINS[origin]
    message = prefix + UNEXPECTED_COMPLETION
    assert grade_efa_line(message) is EfaLineGrade.DEGRADED, message
    assert _evaluate(message, **untrusted) == [], (origin, message)
    fatal = prefix + "Admin queue is closed"
    assert _evaluate(fatal, **untrusted) == [], (origin, "the fatal twin is refused")


@pytest.mark.parametrize("prefix", [DEV, LIVE_DEV], ids=["registered", "renamed"])
@pytest.mark.parametrize("age_seconds", [-30, 0, 900])
def test_a_current_boot_kernel_degraded_line_still_requests_diagnostics(
    prefix: str, age_seconds: int
) -> None:
    [finding] = _evaluate(
        prefix + UNEXPECTED_COMPLETION,
        fields={
            "_TRANSPORT": "kernel",
            "_BOOT_ID": "boot-now",
            "_COLLECTOR_BOOT_ID": "boot-now",
        },
        observed_at=NOW - timedelta(seconds=age_seconds),
    )
    assert (finding.severity, finding.recommended_action) == (
        Severity.WARNING,
        RecoveryAction.RUN_DIAGNOSTICS,
    ), finding


def test_node_side_priority_ranks_fatal_above_degraded_above_noise() -> None:
    # ``_limit_entries`` on the node orders the batch by this rank, and only
    # a line some rule matches is shipped at all.
    assert log_signal_priority(LIVE_016_RDMA_FATAL) == 2, "CRITICAL"
    assert log_signal_priority(DEGRADED_LINES[0]) == 1, "WARNING"
    assert log_signal_priority(INFORMATIONAL_LINES[0]) == -1, "no rule"


def test_legacy_generic_fabric_wording_still_quarantines() -> None:
    # Non-EFA fabrics keep the keyword rule: a trusted kernel line naming
    # RDMA/InfiniBand with a fatal/error/timeout/link-down word.
    for message in (
        "mlx5_core 0000:3b:00.0: infiniband mlx5_0: port 1 link down",
        "ib_core: rdma device mlx5_0 reported a fatal error",
    ):
        [rule] = matching_log_rules(message, "dmesg")
        assert (rule.category, rule.severity, rule.action) == (
            LogHealthCategory.RDMA,
            Severity.CRITICAL,
            RecoveryAction.QUARANTINE,
        ), message


# --------------------------------------------------------------------------- #
# b251: the ib device is renamed on every live node; the timeout is -ETIME
# --------------------------------------------------------------------------- #
#: printf conversions of a kernel format string (``%pI6`` and ``%#x`` too).
CONVERSION = re.compile(
    r"%(?:[-+ #0]*\d*(?:\.\d+)?)(?:hh|h|ll|l|z)?(?:p[A-Za-z0-9]*|[a-zA-Z])"
)
#: An example body for an entry whose pattern pins more than its format: the
#: process failure is fatal only with the timeout errno.
EXAMPLE_BODIES = {
    "Failed to process command %s (opcode %u) comp_status %d err %d": (
        "Failed to process command GET_STATS (opcode 13) comp_status 0 err -62"
    ),
    "Failed to process command %s (opcode %u) err %d": (
        "Failed to process command GET_STATS (opcode 13) err -62"
    ),
}


def _sample(conversion: str) -> str:
    if conversion.endswith("s"):
        return "GET_STATS"
    if conversion.startswith("%p"):
        return "00000000a1b2c3d4"
    if conversion.endswith("x"):
        return "0x58"
    return "13"


def _body(line: EfaKernelLine) -> str:
    """One message the driver prints for ``line``, every conversion filled."""

    body = EXAMPLE_BODIES.get(line.text) or CONVERSION.sub(
        lambda match: _sample(match.group(0)), line.text
    )
    assert re.fullmatch(line.regex(), body), (line.text, body)
    return body


GRADED_ENTRIES = [
    pytest.param(line, id=f"{line.grade.value}:{line.function}:{index}")
    for index, line in enumerate((*EFA_FATAL_LINES, *EFA_DEGRADED_LINES))
]
RULE_OF_GRADE = {
    EfaLineGrade.FATAL: (
        LogHealthCategory.RDMA,
        Severity.CRITICAL,
        RecoveryAction.QUARANTINE,
        2,
    ),
    EfaLineGrade.DEGRADED: (
        LogHealthCategory.RDMA,
        Severity.WARNING,
        RecoveryAction.RUN_DIAGNOSTICS,
        1,
    ),
}


@pytest.mark.parametrize("line", GRADED_ENTRIES)
@pytest.mark.parametrize("prefix", PREFIXES)
def test_every_graded_entry_keeps_its_grade_under_every_device_prefix(
    line: EfaKernelLine, prefix: str
) -> None:
    # rdma-core renames the ib device right after registration, so a live
    # node prints every runtime line under ``rdmap<bus>s<slot>``; the node
    # ships a line only when a rule matches it (log_signal_priority >= 0).
    message = prefix + _body(line)
    category, severity, action, priority = RULE_OF_GRADE[line.grade]
    assert grade_efa_line(message) is line.grade, message
    assert [
        (rule.category, rule.severity, rule.action)
        for rule in matching_log_rules(message, "dmesg")
    ] == [(category, severity, action)], message
    assert log_signal_priority(message) == priority, message


def test_a_renamed_device_line_is_shipped_and_quarantines_the_node() -> None:
    # Before b251 the prefix took only ``efa_N``: this line matched no rule,
    # ranked -1 and the node collector never shipped it.
    message = "efa 0000:55:00.0 rdmap85s0: Admin queue is closed"
    assert log_signal_priority(message) == 2, "shipped as CRITICAL"
    [finding] = _evaluate(message)
    assert (finding.severity, finding.recommended_action, finding.policy_source) == (
        Severity.CRITICAL,
        RecoveryAction.QUARANTINE,
        "SITE_KERNEL_LOG",
    ), finding


@pytest.mark.parametrize(
    "message",
    [
        # ETIMEDOUT (110) is no EFA admin path's return value.
        LIVE_DEV + "Failed to process command GET_STATS (opcode 13) err -110",
        DEV + "Failed to process command CREATE_QP (opcode 1) comp_status 0 err -110",
        # The errno is one number: -620 is not -62, -6 not -62.
        LIVE_DEV + "Failed to process command GET_STATS (opcode 13) err -620",
        LIVE_DEV + "Failed to process command GET_STATS (opcode 13) err -6",
    ],
)
def test_only_the_etime_errno_is_the_devices_silence(message: str) -> None:
    assert ETIME == 62, "include/uapi/asm-generic/errno.h: #define ETIME 62"
    assert grade_efa_line(message) is None, message
    assert matching_log_rules(message) == [], message
    assert _evaluate(message) == [], message


@pytest.mark.parametrize(
    "message",
    [
        # One token without a colon after the PCI address, nothing else.
        "efa 0000:55:00.0 rdmap85s0 extra: Admin queue is closed",
        # No PCI address: ``__ibdev_printk()`` never prints the ib device so.
        "efa rdmap85s0: Admin queue is closed",
        # A word that merely ends in ``efa``.
        "notefa 0000:55:00.0 rdmap85s0: Admin queue is closed",
        # Upper-case hex: the kernel prints a PCI name in lower case.
        "efa 0000:5A:00.0 rdmap90s0: Admin queue is closed",
    ],
)
def test_the_device_prefix_does_not_widen_beyond_the_kernels_forms(
    message: str,
) -> None:
    assert grade_efa_line(message) is None, message
    assert matching_log_rules(message, "dmesg") == [], message


def test_the_module_records_the_kernel_evidence_verbatim() -> None:
    base = "https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/tree/"
    assert EFA_LINUX_SOURCES == {
        "v6.1.189": base + "drivers/infiniband/hw/efa?h=v6.1.189",
        "v6.12.103": base + "drivers/infiniband/hw/efa?h=v6.12.103",
    }
    assert IBDEV_PRINTK_SOURCE == base + "drivers/infiniband/core/device.c?h=v6.1.189"
    assert IBDEV_PRINTK_FORMAT == "%s %s %s: %pV"
    assert ERRNO_SOURCE == base + "include/uapi/asm-generic/errno.h?h=v6.1.189"
    assert ETIME == 62, "#define ETIME 62 /* Timer expired */"
    amzn = "https://github.com/amzn/amzn-drivers/tree/"
    assert AMZN_DRIVERS_EFA_SOURCES == {
        "efa_linux_3.3.1": amzn + "efa_linux_3.3.1/kernel/linux/efa/src",
        "efa_linux_3.4.0": amzn + "efa_linux_3.4.0/kernel/linux/efa/src",
    }
    assert AMZN_DRIVERS_EFA_COMMITS == {
        "efa_linux_3.3.1": "f7ba3059356b13fdabfd61cb35aa0ed7059a81e8",
        "efa_linux_3.4.0": "9d5ca4021f7c9f8421e404c1e89edb0b1325372c",
    }
    rdma_core = "https://github.com/linux-rdma/rdma-core/blob/v65.0/kernel-boot/"
    assert RDMA_CORE_RENAME_SOURCES == {
        "rdma-persistent-naming.rules": rdma_core + "rdma-persistent-naming.rules",
        "rdma_rename.c": rdma_core + "rdma_rename.c",
    }
    assert RDMA_CORE_RENAME_RULE == (
        'ACTION=="add", SUBSYSTEM=="infiniband", PROGRAM="rdma_rename %k NAME_FALLBACK"'
    )
    urls = [
        *EFA_LINUX_SOURCES.values(),
        *AMZN_DRIVERS_EFA_SOURCES.values(),
        *RDMA_CORE_RENAME_SOURCES.values(),
        IBDEV_PRINTK_SOURCE,
        ERRNO_SOURCE,
    ]
    unpinned = [url for url in urls if "/master/" in url or "/main/" in url]
    assert unpinned == [], "every source is read at a tag, never a branch"
    both = frozenset({LINUX_6_1, LINUX_6_12})
    amzn_only = frozenset({AMZN})
    every = both | amzn_only
    assert (LINUX_6_1, LINUX_6_12, AMZN) == (
        "linux-v6.1.189",
        "linux-v6.12.103",
        "amzn-drivers-efa_linux_3.3.1",
    )
    wait = "efa_com_wait_and_process_admin_cq_interrupts"
    fatal = {
        (line.file, line.function, line.text, line.sources) for line in EFA_FATAL_LINES
    }
    assert fatal == {
        (
            "efa_com.c",
            "efa_com_reg_read32",
            "Reading register timed out. expected: req id[%u] offset[%#x] "
            "actual: req id[%u] offset[%#x]",
            every,
        ),
        ("efa_com.c", "efa_com_submit_admin_cmd", "Admin queue is closed", every),
        (
            "efa_com.c",
            "efa_com_wait_and_process_admin_cq_polling",
            "Wait for completion (polling) timeout",
            every,
        ),
        (
            "efa_com.c",
            wait,
            "The device sent a completion but the driver didn't receive any "
            "MSI-X interrupt for admin cmd %s(%d) status %d (ctx: 0x%p, "
            "sq producer: %d, sq consumer: %d, cq consumer: %d)",
            both,
        ),
        (
            "efa_com.c",
            wait,
            "The device sent a completion but the driver didn't receive any "
            "MSI-X interrupt for admin cmd %s(%d) status %d (id: %d, "
            "sq producer: %d, sq consumer: %d, cq consumer: %d)",
            amzn_only,
        ),
        (
            "efa_com.c",
            wait,
            "The device didn't send any completion for admin cmd %s(%d) status "
            "%d (ctx 0x%p, sq producer: %d, sq consumer: %d, cq consumer: %d)",
            both,
        ),
        (
            "efa_com.c",
            wait,
            "The device didn't send any completion for admin cmd %s(%d) status "
            "%d (id: %d, sq producer: %d, sq consumer: %d, cq consumer: %d)",
            amzn_only,
        ),
        (
            "efa_com.c",
            "efa_com_cmd_exec",
            "Failed to submit command %s (opcode %u) err %ld",
            both,
        ),
        (
            "efa_com.c",
            "efa_com_cmd_exec",
            "Failed to submit command %s (opcode %u) err %d",
            amzn_only,
        ),
        (
            "efa_com.c",
            "efa_com_cmd_exec",
            "Failed to process command %s (opcode %u) comp_status %d err %d",
            both,
        ),
        (
            "efa_com.c",
            "efa_com_cmd_exec",
            "Failed to process command %s (opcode %u) err %d",
            amzn_only,
        ),
        # 3.3.1 efa_com.c L731-736: the queue is closed, nothing more printed.
        (
            "efa_com.c",
            "efa_com_alloc_comp_ctx",
            "Completion context[%u] is used[%u]",
            amzn_only,
        ),
        (
            "efa_com.c",
            "efa_com_admin_init",
            "Device isn't ready, abort com init %#x",
            every,
        ),
        (
            "efa_com.c",
            "efa_com_dev_reset",
            "Device isn't ready, can't reset device",
            every,
        ),
        ("efa_com.c", "efa_com_dev_reset", "Reset indication didn't turn on", every),
        ("efa_com.c", "efa_com_dev_reset", "Reset indication didn't turn off", every),
    }
    errno_patterns = [
        line.pattern for line in EFA_FATAL_LINES if line.pattern is not None
    ]
    assert len(errno_patterns) == 2, errno_patterns
    assert all(
        pattern is not None and pattern.endswith(r"err -62(?!\d)")
        for pattern in errno_patterns
    ), errno_patterns
    # 3.3.1 efa_main.c L245 / L855 print both BAR lines too (not 6.12 only).
    bars = {
        line.text: line.sources
        for line in EFA_DEGRADED_LINES
        if "BAR" in line.text and "unavailable" in line.text
    }
    assert bars == {
        "Doorbells BAR unavailable. Requested %#x, available %#x": frozenset(
            {LINUX_6_12, AMZN}
        ),
        "BARs unavailable. Requested %#x, available %#x": frozenset({LINUX_6_12, AMZN}),
    }
    informational = {line.text for line in EFA_INFORMATIONAL_LINES}
    assert {"IB device registered", "Unregister ib device"} <= informational
    assert "Failed to create qp [%d]" in informational
    for line in EFA_VOCABULARY:
        assert line.sources and line.sources <= every, line
        assert line.text == line.text.strip() and "\n" not in line.text, line


@pytest.mark.parametrize(
    "message, grade",
    [
        *((message, EfaLineGrade.FATAL) for message in FATAL_LINES),
        *((message, EfaLineGrade.DEGRADED) for message in DEGRADED_LINES),
    ],
)
def test_each_graded_example_matches_exactly_one_grade(
    message: str, grade: EfaLineGrade
) -> None:
    # The three bodies are disjoint: a line never carries two grades, so the
    # highest-severity choice in ``evaluate_logs`` never has to arbitrate.
    others = {
        candidate
        for candidate in EfaLineGrade
        if candidate is not grade and _grade_matches(candidate, message)
    }
    assert others == set(), (message, grade, others)


def _grade_matches(grade: EfaLineGrade, message: str) -> bool:
    from gpu_fault.efa_kernel_log import (
        EFA_DEGRADED_PATTERN,
        EFA_FATAL_PATTERN,
        EFA_INFORMATIONAL_PATTERN,
    )

    pattern = {
        EfaLineGrade.FATAL: EFA_FATAL_PATTERN,
        EfaLineGrade.DEGRADED: EFA_DEGRADED_PATTERN,
        EfaLineGrade.INFORMATIONAL: EFA_INFORMATIONAL_PATTERN,
    }[grade]
    return pattern.search(message) is not None
