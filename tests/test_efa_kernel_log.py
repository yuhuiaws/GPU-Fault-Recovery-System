"""Pin the EFA/RDMA kernel-log vocabulary the RDMA node log rules grade.

Every line below is text the Linux EFA driver prints, built from the format
strings quoted in :mod:`gpu_fault.efa_kernel_log` (Linux v6.1.188 and
v6.12.103 ``drivers/infiniband/hw/efa/``, the amzn-drivers out-of-tree ``efa``
r3.4.0) through ``__ibdev_printk()`` (``drivers/infiniband/core/device.c``,
``"%s %s %s: %pV"`` = ``efa <pci bdf> efa_N: <message>``; an ib device that is
not registered yet prints ``(null): <message>``) or ``dev_err()`` on the PCI
device (``efa <pci bdf>: <message>``). A change that regrades one of them must
come with a new kernel quote, not a new opinion.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gpu_fault.efa_kernel_log import (
    AMZN,
    AMZN_DRIVERS_EFA_SOURCE,
    EFA_FATAL_LINES,
    EFA_INFORMATIONAL_LINES,
    EFA_LINUX_SOURCES,
    EFA_VOCABULARY,
    IBDEV_PRINTK_FORMAT,
    IBDEV_PRINTK_SOURCE,
    LINUX_6_1,
    LINUX_6_12,
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
DEV = "efa 0000:00:1d.0 efa_0: "

#: The GF-GENERAL-LIVE-016 ``rdma_fatal`` reviewed line (b189 harness):
#: ``efa_com_wait_and_process_admin_cq_polling()``.
LIVE_016_RDMA_FATAL = "efa 0000:00:1d.0 efa_0: Wait for completion (polling) timeout"

#: The device stopped answering its admin queue, the queue is closed, or the
#: reset handshake failed: the node's fabric is gone until the driver reloads.
FATAL_LINES = [
    LIVE_016_RDMA_FATAL,
    DEV + "Admin queue is closed",
    DEV + "The device didn't send any completion for admin cmd CREATE_QP(1) "
    "status 0 (ctx 0x00000000a1b2c3d4, sq producer: 12, sq consumer: 11, "
    "cq consumer: 11)",
    # amzn-drivers prints the completion context id instead of the pointer.
    DEV + "The device didn't send any completion for admin cmd CREATE_QP(1) "
    "status 0 (id: 7, sq producer: 12, sq consumer: 11, cq consumer: 11)",
    DEV + "The device sent a completion but the driver didn't receive any MSI-X "
    "interrupt for admin cmd DESTROY_QP(2) status 1 (ctx: 0x00000000a1b2c3d4, "
    "sq producer: 12, sq consumer: 11, cq consumer: 11)",
    DEV + "Reading register timed out. expected: req id[12] offset[0x58] actual: "
    "req id[11] offset[0x58]",
    # ``efa_com_cmd_exec()``: a submit failure closes the queue whatever the
    # errno; a process failure is the device's silence only with -ETIME (-110).
    DEV + "Failed to submit command CREATE_QP (opcode 1) err -19",
    DEV + "Failed to process command CREATE_QP (opcode 1) comp_status 0 err -110",
    DEV + "Failed to process command CREATE_QP (opcode 1) err -110",
    # Probe/reset time: the ib device has no name yet, ``__ibdev_printk()``
    # prints ``(null): ``; at remove time the registered name is still there.
    "(null): Device isn't ready, abort com init 0x0",
    "(null): Device isn't ready, can't reset device",
    "(null): Reset indication didn't turn on",
    DEV + "Reset indication didn't turn off",
    # dmesg-style timestamp in front of the kernel line.
    "[ 1234.567890] efa 0000:00:1d.0 efa_0: Admin queue is closed",
]

#: The driver reports the device or firmware misbehaving, or the device
#: failing to come up: read-only fabric diagnostics, no isolation.
DEGRADED_LINES = [
    DEV + "Received completion with unexpected command id[7], sq producer: 12, "
    "sq consumer: 11, cq consumer: 11",
    DEV + "Received completion with unexpected command id[7], status[1] "
    "sq producer[12], sq consumer[11], cq consumer[11]",
    DEV + "Received completion with invalid checksum, cqe[12], calc[13], "
    "sq producer[12], sq consumer[11], cq consumer[11]",
    DEV + "Completion context for command_id 0x7 is occupied",
    DEV + "Completion context[7] is used[1]",
    DEV + "Completion event on non-existent CQ[42]",
    DEV + "Unknown event type received 9",
    DEV + "Unknown event was received or event with unimplemented handler",
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
    "efa 0000:00:1d.0: Failed to remap register bar",
    "efa 0000:00:1d.0: Failed to allocate QP table",
]

#: Printed in normal operation, or for one command that the device answered
#: with an error (a job asked for a resource the device could not give):
#: never a finding.
INFORMATIONAL_LINES = [
    DEV + "IB device registered",
    DEV + "Unregister ib device",
    "efa 0000:00:1d.0: Setup irq:35 name:efa-mgmnt@pci:0000:00:1d.0",
    "Elastic Fabric Adapter (EFA) v3.4.0g",
    "Couldn't register efa driver",
    # ``efa_com_comp_status_to_errno()``: the device answered
    # RESOURCE_ALLOCATION_FAILURE (-ENOMEM) / ILLEGAL_PARAMETER (-EINVAL).
    DEV + "Failed to process command CREATE_QP (opcode 1) comp_status 2 err -12",
    DEV + "Failed to process command CREATE_QP (opcode 1) err -22",
    DEV + "Failed to create qp [-12]",
    DEV + "Failed to create cq[-12]",
    DEV + "Failed to register mr [-12]",
    DEV + "Failed to de-register mr(lkey-4096) [-22]",
    DEV + "Failed to create ah for fe80:0000:0000:0000:0000:0000:0000:0001 [-12]",
    DEV + "Failed to allocate pd[-12]",
    DEV + "Failed to get stats type-0 scope-0.0 [-22]",
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


def _evaluate(message: str, *, source: str = "dmesg", kernel: bool = True) -> list:
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
                    observed_at=NOW,
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


def test_the_module_records_the_kernel_evidence_verbatim() -> None:
    base = "https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/tree/"
    assert EFA_LINUX_SOURCES == {
        "v6.1.188": base + "drivers/infiniband/hw/efa?h=v6.1.188",
        "v6.12.103": base + "drivers/infiniband/hw/efa?h=v6.12.103",
    }
    assert IBDEV_PRINTK_SOURCE == base + "drivers/infiniband/core/device.c?h=v6.12.103"
    assert IBDEV_PRINTK_FORMAT == "%s %s %s: %pV"
    assert AMZN_DRIVERS_EFA_SOURCE == (
        "https://github.com/amzn/amzn-drivers/tree/master/kernel/linux/efa/src"
    )
    both = frozenset({LINUX_6_1, LINUX_6_12})
    amzn = frozenset({AMZN})
    every = both | amzn
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
            amzn,
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
            amzn,
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
            amzn,
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
            amzn,
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
