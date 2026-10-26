"""Reviewed vocabulary of the EFA (Elastic Fabric Adapter) kernel driver's log lines.

The RDMA node log rules (``log_rules.NODE_LOG_RULES``) decide from a trusted
kernel line whether the node's fabric device is gone (CRITICAL / QUARANTINE),
degraded (WARNING / RUN_DIAGNOSTICS) or merely talking (no finding). The EFA
driver prints no "link down" or "fatal error" line at all -- ``efa_query_port()``
(``efa_verbs.c``) reports ``IB_PORT_ACTIVE`` / ``IB_PORT_PHYS_STATE_LINK_UP``
unconditionally -- so the only words that mean a dead device are the admin
queue's. Everything graded here is quoted below from the source that prints
it; a change needs a new quote, not a new opinion.

Print format. Linux v6.12.103 (identical in v6.1.188), ``drivers/infiniband/core/device.c``
``__ibdev_printk()`` (https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/tree/drivers/infiniband/core/device.c?h=v6.12.103)::

    if (ibdev && ibdev->dev.parent)
        dev_printk_emit(level[1] - '0', ibdev->dev.parent, "%s %s %s: %pV",
                        dev_driver_string(ibdev->dev.parent),
                        dev_name(ibdev->dev.parent), dev_name(&ibdev->dev), vaf);
    else if (ibdev)
        printk("%s%s: %pV", level, dev_name(&ibdev->dev), vaf);

so ``ibdev_err(efa_dev, ...)`` reads ``efa 0000:00:1d.0 efa_0: <message>``.
``efa_main.c`` sets ``dev->ibdev.dev.parent`` and registers the name
(``ib_register_device(&dev->ibdev, "efa_%d", ...)``) in ``efa_ib_device_add()``
*after* ``efa_device_init()`` / ``efa_com_admin_init()``, so a probe-time line
has neither and prints ``(null): <message>``. ``dev_err(&pdev->dev, ...)``
reads ``efa 0000:00:1d.0: <message>``. ``include/rdma/ib_verbs.h``:
``ibdev_err_ratelimited`` is ``ibdev_err`` behind ``DEFINE_RATELIMIT_STATE``
(10 lines per 5 s by default), hence one line per burst.

Sources quoted (``drivers/infiniband/hw/efa/``):

* Linux v6.1.188 (the EKS GPU nodes, ``6.1.188-233.385.amzn2023``):
  https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/tree/drivers/infiniband/hw/efa?h=v6.1.188
* Linux v6.12.103 (HyperPod, ``6.12.103``):
  https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/tree/drivers/infiniband/hw/efa?h=v6.12.103
  ``efa_com.c`` is byte-identical between the two; ``efa_main.c`` differs in
  the MSI-X and BAR messages listed per version below.
* amzn-drivers out-of-tree ``efa`` r3.4.0 (``kernel/linux/efa/src``,
  https://github.com/amzn/amzn-drivers/tree/master/kernel/linux/efa/src): the
  EKS AL2023 AMI installs it (amazon-eks-ami
  ``templates/al2023/provisioners/install-efa.sh``: ``./efa_installer.sh
  --minimal -y``, no ``--skip-kmod``); the harness survey records no
  ``modinfo efa``, so both in-tree and out-of-tree spellings are graded.

FATAL -- the device stopped answering, the queue is closed, or the reset
handshake failed (``efa_com.c``):

* ``efa_com_reg_read32()``: ``"Reading register timed out. expected: req
  id[%u] offset[%#x] actual: req id[%u] offset[%#x]\\n"`` -- the readless MMIO
  read got no answer within ``EFA_REG_READ_TIMEOUT_US``.
* ``efa_com_submit_admin_cmd()``: ``"Admin queue is closed\\n"`` when
  ``!test_bit(EFA_AQ_STATE_RUNNING_BIT, &aq->state)``; returns ``-ENODEV``,
  and a failed ``__efa_com_submit_admin_cmd()`` clears that bit itself.
* ``efa_com_wait_and_process_admin_cq_polling()``: ``"Wait for completion
  (polling) timeout\\n"`` -- ``clear_bit(EFA_AQ_STATE_RUNNING_BIT, ...)``,
  ``err = -ETIME``.
* ``efa_com_wait_and_process_admin_cq_interrupts()``: ``"The device sent a
  completion but the driver didn't receive any MSI-X interrupt for admin cmd
  %s(%d) status %d (ctx: 0x%p, sq producer: %d, sq consumer: %d, cq consumer:
  %d)\\n"`` and ``"The device didn't send any completion for admin cmd %s(%d)
  status %d (ctx 0x%p, sq producer: %d, sq consumer: %d, cq consumer: %d)\\n"``
  -- both ``clear_bit(EFA_AQ_STATE_RUNNING_BIT, ...)``, ``-ETIME``.
  amzn-drivers prints ``(id: %d, ...)`` for the context.
* ``efa_com_cmd_exec()``: ``"Failed to submit command %s (opcode %u) err
  %ld\\n"`` (amzn-drivers ``err %d``; the queue is closed before or by the
  failure: in-tree a failed ``__efa_com_submit_admin_cmd()`` clears the bit,
  amzn-drivers returns only ``-ENODEV`` from the closed-queue check) and ``"Failed to
  process command %s (opcode %u) comp_status %d err %d\\n"`` (amzn-drivers:
  ``"Failed to process command %s (opcode %u) err %d\\n"``). The second is
  fatal only with ``err -110`` (``-ETIME``, the wait above); any other value
  comes from ``efa_com_comp_status_to_errno()`` -- the device answered
  (``RESOURCE_ALLOCATION_FAILURE`` -> ``-ENOMEM``, ``UNSUPPORTED_OPCODE`` ->
  ``-EOPNOTSUPP``, ``BAD_OPCODE`` / ``MALFORMED_REQUEST`` /
  ``ILLEGAL_PARAMETER`` / ``UNKNOWN_ERROR`` -> ``-EINVAL``) and is alive.
* ``efa_com_admin_init()``: ``"Device isn't ready, abort com init %#x\\n"``.
* ``efa_com_dev_reset()``: ``"Device isn't ready, can't reset device\\n"``,
  ``"Reset indication didn't turn on\\n"``, ``"Reset indication didn't turn
  off\\n"`` (called from ``efa_device_init()`` at probe and from
  ``efa_remove_device()`` / ``efa_shutdown()``).

DEGRADED -- the driver reports the device or firmware misbehaving, or the
device failing to come up; a workload cannot cause these:

* ``efa_com.c``: ``efa_com_reg_read32()`` ``"Reading register failed: wrong
  offset provided\\n"``; ``efa_com_get_comp_ctx()`` ``"Completion context for
  command_id %#x is occupied\\n"`` (amzn-drivers ``efa_com_alloc_comp_ctx()``
  ``"Completion context[%u] is used[%u]\\n"``); ``efa_com_handle_single_admin_completion()`` ``"Received
  completion with unexpected command id[%d], sq producer: %d, sq consumer: %d,
  cq consumer: %d\\n"`` (amzn-drivers ``"Received completion with unexpected
  command id[%x], status[%d] sq producer[%d], sq consumer[%d], cq
  consumer[%d]\\n"`` and ``efa_com_cqe_checksum_valid()`` ``"Received completion with invalid
  checksum, cqe[%u], calc[%u], sq producer[%d], sq consumer[%d], cq
  consumer[%d]\\n"``);
  ``efa_com_admin_init_aenq()`` ``"aenq handlers pointer is NULL\\n"``;
  ``efa_com_validate_version()`` ``"EFA version is lower than the minimal
  version the driver supports\\n"`` / ``"EFA ctrl version is lower than the
  minimal ctrl version the driver supports\\n"``; ``efa_com_get_dma_width()``
  ``"DMA width illegal value: %d\\n"``; ``efa_com_dev_reset()`` ``"Invalid
  timeout value\\n"``; ``efa_com_create_eq()`` ``"Failed to create eq[%d]\\n"``;
  ``efa_com_destroy_eq()`` ``"Failed to destroy EQ-%u [%d]\\n"``.
* ``efa_main.c``: ``unimplemented_aenq_handler()`` ``"Unknown event was
  received or event with unimplemented handler\\n"``; ``efa_process_comp_eqe()``
  ``"Completion event on non-existent CQ[%u]"``; ``efa_process_eqe()``
  ``"Unknown event type received %lu"``; ``efa_request_irq()`` ``"Failed to
  request irq %s (%d)\\n"``; ``efa_request_doorbell_bar()``
  ``"pci_request_selected_regions for bar %d failed %d\\n"`` and (v6.12)
  ``"Doorbells BAR unavailable. Requested %#x, available %#x\\n"``;
  ``efa_enable_msix()`` ``"Failed to enable MSI-X. irq_num %d\\n"`` /
  ``"Allocated %d MSI-X (out of %d requested)\\n"`` (v6.1) and ``"Failed to
  enable MSI-X. error %d\\n"`` (v6.12); ``efa_device_init()``
  ``"dma_set_mask_and_coherent failed %d\\n"``; ``efa_probe_device()``
  ``"pci_enable_device_mem() failed!\\n"``, ``"Device alloc failed\\n"``,
  (v6.12) ``"BARs unavailable. Requested %#x, available %#x\\n"``,
  ``"pci_request_selected_regions failed %d\\n"``, ``"Failed to remap register
  bar\\n"``, ``"Failed to init readless MMIO\\n"``, ``"EFA device init
  failed\\n"``; amzn-drivers adds ``efa_create_qp_table()`` ``"Failed to allocate
  QP table\\n"``, ``efa_alloc_dev_uar()`` ``"Failed to allocate UAR, err %d\\n"``
  and (``efa_com.c`` ``efa_com_admin_init()``) ``"Failed to init AH cache\\n"``.
* ``efa_com_cmd.c`` probe-time queries: ``"Failed to get device attributes
  %d\\n"``, ``"Failed to get device attr api version [%u < 1]\\n"``, ``"Failed
  to get queue attributes %d\\n"`` (amzn-drivers ``attributes1`` /
  ``attributes2``), ``"Failed to get network attributes %d\\n"``, ``"Failed to
  get event queue attributes %d\\n"``, ``"Failed to get hw hints %d\\n"``,
  ``efa_com_set_aenq_config()`` ``"Failed to get aenq attributes: %d\\n"``,
  ``"Trying to set unsupported aenq groups[%#x] supported[%#x]\\n"``, ``"Failed
  to set aenq attributes: %d\\n"``.

INFORMATIONAL -- normal operation, or one command the device answered with an
error (a job asked for a resource the device could not give); no finding:
``efa_ib_device_add()`` ``ibdev_info "IB device registered\\n"``;
``efa_ib_device_remove()`` ``ibdev_info "Unregister ib device\\n"``;
``efa_setup_mgmnt_irq()`` ``dev_info "Setup irq:%d name:%s\\n"``; amzn-drivers
``efa_init()`` ``pr_info("%s\\n", version)`` (``"Elastic Fabric Adapter (EFA)
v3.4.0g"``) and ``pr_err
"Couldn't register efa driver\\n"``; ``efa_com_cmd.c`` per-resource failures
``"Failed to create qp [%d]\\n"``, ``"Failed to modify qp-%u modify_mask[%#x]
[%d]\\n"``, ``"Failed to query qp-%u [%d]\\n"``, ``"Failed to destroy qp-%u
[%d]\\n"``, ``"Failed to create cq[%d]\\n"``, ``"Failed to destroy CQ-%u
[%d]\\n"``, ``"Failed to register mr [%d]\\n"``, ``"Failed to de-register
mr(lkey-%u) [%d]\\n"``, ``"Failed to create ah for %pI6 [%d]\\n"``, ``"Failed
to destroy ah-%d pd-%d [%d]\\n"``, ``"Failed to allocate pd[%d]\\n"``, ``"Failed
to deallocate pd-%u [%d]\\n"``, ``"Failed to allocate uar[%d]\\n"``, ``"Failed
to deallocate uar-%u [%d]\\n"``, ``"Failed to get stats type-%u scope-%u.%u
[%d]\\n"``, ``"Feature %d isn't supported\\n"``, ``"Failed to submit
get_feature command %d [%d]\\n"``, ``"Failed to submit set_feature command %d
error: %d\\n"``; ``efa_verbs.c`` ``efa_zalloc_mapped()`` ``"Failed to map DMA address\\n"``,
``pbl_chunk_list_create()`` ``"chunk[%u] dma_map_failed\\n"``,
``pbl_continuous_initialize()`` ``"Unable to map pbl to DMA address\\n"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

#: ``drivers/infiniband/hw/efa`` at the two kernels the site runs.
EFA_LINUX_SOURCES: dict[str, str] = {
    "v6.1.188": (
        "https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/tree/"
        "drivers/infiniband/hw/efa?h=v6.1.188"
    ),
    "v6.12.103": (
        "https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/tree/"
        "drivers/infiniband/hw/efa?h=v6.12.103"
    ),
}
#: The out-of-tree driver the aws-efa-installer ships (r3.4.0 reviewed).
AMZN_DRIVERS_EFA_SOURCE = (
    "https://github.com/amzn/amzn-drivers/tree/master/kernel/linux/efa/src"
)
#: ``__ibdev_printk()``: driver name, parent (PCI) name, ib device name.
IBDEV_PRINTK_SOURCE = (
    "https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/tree/"
    "drivers/infiniband/core/device.c?h=v6.12.103"
)
IBDEV_PRINTK_FORMAT = "%s %s %s: %pV"

#: Where a format string was read: the two site kernels and the out-of-tree
#: driver. An entry lists every source that prints its exact text.
LINUX_6_1 = "linux-v6.1.188"
LINUX_6_12 = "linux-v6.12.103"
AMZN = "amzn-drivers-r3.4.0"
ALL_SOURCES: frozenset[str] = frozenset({LINUX_6_1, LINUX_6_12, AMZN})
IN_TREE: frozenset[str] = frozenset({LINUX_6_1, LINUX_6_12})
ONLY_AMZN: frozenset[str] = frozenset({AMZN})
ONLY_6_1: frozenset[str] = frozenset({LINUX_6_1})
ONLY_6_12: frozenset[str] = frozenset({LINUX_6_12})
SINCE_6_12: frozenset[str] = frozenset({LINUX_6_12, AMZN})


class EfaLineGrade(StrEnum):
    #: The device stopped answering / the admin queue is closed / reset failed.
    FATAL = "FATAL"
    #: The driver reports the device misbehaving or failing to come up.
    DEGRADED = "DEGRADED"
    #: Normal operation or a per-command failure the device answered.
    INFORMATIONAL = "INFORMATIONAL"


@dataclass(frozen=True)
class EfaKernelLine:
    """One ``ibdev_*`` / ``dev_*`` / ``pr_*`` format string the driver prints."""

    file: str
    function: str
    #: The kernel format string verbatim, without the trailing newline.
    text: str
    grade: EfaLineGrade
    #: The sources (``LINUX_6_1`` / ``LINUX_6_12`` / ``AMZN``) printing ``text``.
    sources: frozenset[str] = ALL_SOURCES
    #: Regex for the message body when the format alone is not the rule
    #: (``None``: derived from ``text``, every conversion matches ``.+?``).
    pattern: str | None = None

    def regex(self) -> str:
        if self.pattern is not None:
            return self.pattern
        return format_regex(self.text)


#: printf conversions in the kernel's format strings (``%pI6`` included).
_CONVERSION = re.compile(
    r"%(?:[-+ #0]*\d*(?:\.\d+)?)(?:hh|h|ll|l|z)?(?:p[A-Za-z0-9]*|[a-zA-Z])"
)


def format_regex(text: str) -> str:
    """A regex for a kernel format string: literals verbatim, conversions ``.+?``."""

    return r".+?".join(re.escape(part) for part in _CONVERSION.split(text))


def _line(
    file: str,
    function: str,
    text: str,
    grade: EfaLineGrade,
    *,
    sources: frozenset[str] = ALL_SOURCES,
    pattern: str | None = None,
) -> EfaKernelLine:
    return EfaKernelLine(file, function, text, grade, sources, pattern)


_F = EfaLineGrade.FATAL
_D = EfaLineGrade.DEGRADED
_I = EfaLineGrade.INFORMATIONAL

EFA_FATAL_LINES: tuple[EfaKernelLine, ...] = (
    _line(
        "efa_com.c",
        "efa_com_reg_read32",
        "Reading register timed out. expected: req id[%u] offset[%#x] "
        "actual: req id[%u] offset[%#x]",
        _F,
    ),
    _line("efa_com.c", "efa_com_submit_admin_cmd", "Admin queue is closed", _F),
    _line(
        "efa_com.c",
        "efa_com_wait_and_process_admin_cq_polling",
        "Wait for completion (polling) timeout",
        _F,
    ),
    _line(
        "efa_com.c",
        "efa_com_wait_and_process_admin_cq_interrupts",
        "The device sent a completion but the driver didn't receive any MSI-X "
        "interrupt for admin cmd %s(%d) status %d (ctx: 0x%p, sq producer: %d, "
        "sq consumer: %d, cq consumer: %d)",
        _F,
        sources=IN_TREE,
    ),
    _line(
        "efa_com.c",
        "efa_com_wait_and_process_admin_cq_interrupts",
        "The device sent a completion but the driver didn't receive any MSI-X "
        "interrupt for admin cmd %s(%d) status %d (id: %d, sq producer: %d, "
        "sq consumer: %d, cq consumer: %d)",
        _F,
        sources=ONLY_AMZN,
    ),
    _line(
        "efa_com.c",
        "efa_com_wait_and_process_admin_cq_interrupts",
        "The device didn't send any completion for admin cmd %s(%d) status %d "
        "(ctx 0x%p, sq producer: %d, sq consumer: %d, cq consumer: %d)",
        _F,
        sources=IN_TREE,
    ),
    _line(
        "efa_com.c",
        "efa_com_wait_and_process_admin_cq_interrupts",
        "The device didn't send any completion for admin cmd %s(%d) status %d "
        "(id: %d, sq producer: %d, sq consumer: %d, cq consumer: %d)",
        _F,
        sources=ONLY_AMZN,
    ),
    _line(
        "efa_com.c",
        "efa_com_cmd_exec",
        "Failed to submit command %s (opcode %u) err %ld",
        _F,
        sources=IN_TREE,
    ),
    _line(
        "efa_com.c",
        "efa_com_cmd_exec",
        "Failed to submit command %s (opcode %u) err %d",
        _F,
        sources=ONLY_AMZN,
    ),
    # Only the device's silence (``-ETIME``, -110) is fatal; every other errno
    # is the device's own answer (``efa_com_comp_status_to_errno()``).
    _line(
        "efa_com.c",
        "efa_com_cmd_exec",
        "Failed to process command %s (opcode %u) comp_status %d err %d",
        _F,
        sources=IN_TREE,
        pattern=(
            r"Failed to process command .+? \(opcode .+?\) comp_status .+? "
            r"err -110(?!\d)"
        ),
    ),
    _line(
        "efa_com.c",
        "efa_com_cmd_exec",
        "Failed to process command %s (opcode %u) err %d",
        _F,
        sources=ONLY_AMZN,
        pattern=r"Failed to process command .+? \(opcode .+?\) err -110(?!\d)",
    ),
    _line(
        "efa_com.c",
        "efa_com_admin_init",
        "Device isn't ready, abort com init %#x",
        _F,
    ),
    _line(
        "efa_com.c", "efa_com_dev_reset", "Device isn't ready, can't reset device", _F
    ),
    _line("efa_com.c", "efa_com_dev_reset", "Reset indication didn't turn on", _F),
    _line("efa_com.c", "efa_com_dev_reset", "Reset indication didn't turn off", _F),
)

EFA_DEGRADED_LINES: tuple[EfaKernelLine, ...] = (
    _line(
        "efa_com.c",
        "efa_com_reg_read32",
        "Reading register failed: wrong offset provided",
        _D,
    ),
    _line(
        "efa_com.c",
        "efa_com_get_comp_ctx",
        "Completion context for command_id %#x is occupied",
        _D,
        sources=IN_TREE,
    ),
    _line(
        "efa_com.c",
        "efa_com_alloc_comp_ctx",
        "Completion context[%u] is used[%u]",
        _D,
        sources=ONLY_AMZN,
    ),
    _line(
        "efa_com.c",
        "efa_com_handle_single_admin_completion",
        "Received completion with unexpected command id[%d], sq producer: %d, "
        "sq consumer: %d, cq consumer: %d",
        _D,
        sources=IN_TREE,
    ),
    _line(
        "efa_com.c",
        "efa_com_handle_single_admin_completion",
        "Received completion with unexpected command id[%x], status[%d] "
        "sq producer[%d], sq consumer[%d], cq consumer[%d]",
        _D,
        sources=ONLY_AMZN,
    ),
    _line(
        "efa_com.c",
        "efa_com_cqe_checksum_valid",
        "Received completion with invalid checksum, cqe[%u], calc[%u], "
        "sq producer[%d], sq consumer[%d], cq consumer[%d]",
        _D,
        sources=ONLY_AMZN,
    ),
    _line("efa_com.c", "efa_com_admin_init_aenq", "aenq handlers pointer is NULL", _D),
    _line(
        "efa_com.c",
        "efa_com_validate_version",
        "EFA version is lower than the minimal version the driver supports",
        _D,
    ),
    _line(
        "efa_com.c",
        "efa_com_validate_version",
        "EFA ctrl version is lower than the minimal ctrl version the driver supports",
        _D,
    ),
    _line("efa_com.c", "efa_com_get_dma_width", "DMA width illegal value: %d", _D),
    _line("efa_com.c", "efa_com_dev_reset", "Invalid timeout value", _D),
    _line("efa_com.c", "efa_com_create_eq", "Failed to create eq[%d]", _D),
    _line("efa_com.c", "efa_com_destroy_eq", "Failed to destroy EQ-%u [%d]", _D),
    _line(
        "efa_main.c",
        "unimplemented_aenq_handler",
        "Unknown event was received or event with unimplemented handler",
        _D,
    ),
    _line(
        "efa_main.c",
        "efa_process_comp_eqe",
        "Completion event on non-existent CQ[%u]",
        _D,
    ),
    _line("efa_main.c", "efa_process_eqe", "Unknown event type received %lu", _D),
    _line("efa_main.c", "efa_request_irq", "Failed to request irq %s (%d)", _D),
    _line(
        "efa_main.c",
        "efa_request_doorbell_bar",
        "pci_request_selected_regions for bar %d failed %d",
        _D,
    ),
    _line(
        "efa_main.c",
        "efa_request_doorbell_bar",
        "Doorbells BAR unavailable. Requested %#x, available %#x",
        _D,
        sources=ONLY_6_12,
    ),
    _line(
        "efa_main.c",
        "efa_enable_msix",
        "Failed to enable MSI-X. irq_num %d",
        _D,
        sources=ONLY_6_1,
    ),
    _line(
        "efa_main.c",
        "efa_enable_msix",
        "Allocated %d MSI-X (out of %d requested)",
        _D,
        sources=ONLY_6_1,
    ),
    _line(
        "efa_main.c",
        "efa_enable_msix",
        "Failed to enable MSI-X. error %d",
        _D,
        sources=SINCE_6_12,
    ),
    _line(
        "efa_main.c",
        "efa_device_init",
        "dma_set_mask_and_coherent failed %d",
        _D,
    ),
    _line("efa_main.c", "efa_probe_device", "pci_enable_device_mem() failed!", _D),
    _line("efa_main.c", "efa_probe_device", "Device alloc failed", _D),
    _line(
        "efa_main.c",
        "efa_probe_device",
        "BARs unavailable. Requested %#x, available %#x",
        _D,
        sources=ONLY_6_12,
    ),
    _line(
        "efa_main.c",
        "efa_probe_device",
        "pci_request_selected_regions failed %d",
        _D,
    ),
    _line("efa_main.c", "efa_probe_device", "Failed to remap register bar", _D),
    _line("efa_main.c", "efa_probe_device", "Failed to init readless MMIO", _D),
    _line("efa_main.c", "efa_probe_device", "EFA device init failed", _D),
    _line(
        "efa_main.c",
        "efa_create_qp_table",
        "Failed to allocate QP table",
        _D,
        sources=ONLY_AMZN,
    ),
    _line(
        "efa_main.c",
        "efa_alloc_dev_uar",
        "Failed to allocate UAR, err %d",
        _D,
        sources=ONLY_AMZN,
    ),
    _line(
        "efa_com.c",
        "efa_com_admin_init",
        "Failed to init AH cache",
        _D,
        sources=ONLY_AMZN,
    ),
    _line(
        "efa_com_cmd.c",
        "efa_com_get_device_attr",
        "Failed to get device attributes %d",
        _D,
    ),
    _line(
        "efa_com_cmd.c",
        "efa_com_get_device_attr",
        "Failed to get device attr api version [%u < 1]",
        _D,
    ),
    _line(
        "efa_com_cmd.c",
        "efa_com_get_device_attr",
        "Failed to get queue attributes %d",
        _D,
        sources=IN_TREE,
    ),
    _line(
        "efa_com_cmd.c",
        "efa_com_get_device_attr",
        "Failed to get queue attributes1 %d",
        _D,
        sources=ONLY_AMZN,
    ),
    _line(
        "efa_com_cmd.c",
        "efa_com_get_device_attr",
        "Failed to get queue attributes2 %d",
        _D,
        sources=ONLY_AMZN,
    ),
    _line(
        "efa_com_cmd.c",
        "efa_com_get_device_attr",
        "Failed to get network attributes %d",
        _D,
    ),
    _line(
        "efa_com_cmd.c",
        "efa_com_get_device_attr",
        "Failed to get event queue attributes %d",
        _D,
    ),
    _line("efa_com_cmd.c", "efa_com_get_hw_hints", "Failed to get hw hints %d", _D),
    _line(
        "efa_com_cmd.c",
        "efa_com_set_aenq_config",
        "Failed to get aenq attributes: %d",
        _D,
    ),
    _line(
        "efa_com_cmd.c",
        "efa_com_set_aenq_config",
        "Trying to set unsupported aenq groups[%#x] supported[%#x]",
        _D,
    ),
    _line(
        "efa_com_cmd.c",
        "efa_com_set_aenq_config",
        "Failed to set aenq attributes: %d",
        _D,
    ),
)

EFA_INFORMATIONAL_LINES: tuple[EfaKernelLine, ...] = (
    _line("efa_main.c", "efa_ib_device_add", "IB device registered", _I),
    _line("efa_main.c", "efa_ib_device_remove", "Unregister ib device", _I),
    _line("efa_main.c", "efa_setup_mgmnt_irq", "Setup irq:%d name:%s", _I),
    # ``pr_info("%s\n", version)`` with ``version[] = DEVICE_NAME " v"
    # DRV_MODULE_VERSION``: "Elastic Fabric Adapter (EFA) v3.4.0g". Printed
    # without a device prefix, so it is listed for the record and matches
    # nothing: a bare ``%s`` would otherwise swallow every line.
    _line(
        "efa_main.c",
        "efa_init",
        "%s",
        _I,
        sources=ONLY_AMZN,
        pattern=r"Elastic Fabric Adapter \(EFA\) v\S+",
    ),
    _line(
        "efa_main.c",
        "efa_init",
        "Couldn't register efa driver",
        _I,
        sources=ONLY_AMZN,
    ),
    _line("efa_com_cmd.c", "efa_com_create_qp", "Failed to create qp [%d]", _I),
    _line(
        "efa_com_cmd.c",
        "efa_com_modify_qp",
        "Failed to modify qp-%u modify_mask[%#x] [%d]",
        _I,
    ),
    _line("efa_com_cmd.c", "efa_com_query_qp", "Failed to query qp-%u [%d]", _I),
    _line("efa_com_cmd.c", "efa_com_destroy_qp", "Failed to destroy qp-%u [%d]", _I),
    _line("efa_com_cmd.c", "efa_com_create_cq", "Failed to create cq[%d]", _I),
    _line("efa_com_cmd.c", "efa_com_destroy_cq", "Failed to destroy CQ-%u [%d]", _I),
    _line("efa_com_cmd.c", "efa_com_register_mr", "Failed to register mr [%d]", _I),
    _line(
        "efa_com_cmd.c",
        "efa_com_dereg_mr",
        "Failed to de-register mr(lkey-%u) [%d]",
        _I,
    ),
    _line(
        "efa_com_cmd.c", "efa_com_create_ah", "Failed to create ah for %pI6 [%d]", _I
    ),
    _line(
        "efa_com_cmd.c",
        "efa_com_destroy_ah",
        "Failed to destroy ah-%d pd-%d [%d]",
        _I,
    ),
    _line("efa_com_cmd.c", "efa_com_alloc_pd", "Failed to allocate pd[%d]", _I),
    _line("efa_com_cmd.c", "efa_com_dealloc_pd", "Failed to deallocate pd-%u [%d]", _I),
    _line("efa_com_cmd.c", "efa_com_alloc_uar", "Failed to allocate uar[%d]", _I),
    _line(
        "efa_com_cmd.c",
        "efa_com_dealloc_uar",
        "Failed to deallocate uar-%u [%d]",
        _I,
    ),
    _line(
        "efa_com_cmd.c",
        "efa_com_get_stats",
        "Failed to get stats type-%u scope-%u.%u [%d]",
        _I,
    ),
    _line("efa_com_cmd.c", "efa_com_get_feature_ex", "Feature %d isn't supported", _I),
    _line(
        "efa_com_cmd.c",
        "efa_com_get_feature_ex",
        "Failed to submit get_feature command %d [%d]",
        _I,
    ),
    _line(
        "efa_com_cmd.c",
        "efa_com_set_feature_ex",
        "Failed to submit set_feature command %d error: %d",
        _I,
    ),
    _line("efa_verbs.c", "efa_zalloc_mapped", "Failed to map DMA address", _I),
    _line("efa_verbs.c", "pbl_chunk_list_create", "chunk[%u] dma_map_failed", _I),
    _line(
        "efa_verbs.c",
        "pbl_continuous_initialize",
        "Unable to map pbl to DMA address",
        _I,
    ),
)

EFA_VOCABULARY: tuple[EfaKernelLine, ...] = (
    *EFA_FATAL_LINES,
    *EFA_DEGRADED_LINES,
    *EFA_INFORMATIONAL_LINES,
)

#: What stands before the message: ``efa <bdf> efa_N: `` (``ibdev_*`` on a
#: registered device), ``efa <bdf>: `` (``dev_*`` on the PCI device),
#: ``(null): `` (``ibdev_*`` before registration). A dmesg timestamp or a
#: ``[`` may precede it. Case-sensitive: the kernel's spelling is fixed.
EFA_DEVICE_PREFIX = (
    r"(?:^|[\s\[])(?:efa(?:\s+[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])?"
    r"(?:\s+efa_\d+)?|\(null\)):\s+"
)


def _body(grade: EfaLineGrade) -> str:
    return "|".join(
        f"(?:{line.regex()})" for line in EFA_VOCABULARY if line.grade is grade
    )


EFA_FATAL_REGEX = f"{EFA_DEVICE_PREFIX}(?:{_body(EfaLineGrade.FATAL)})"
EFA_DEGRADED_REGEX = f"{EFA_DEVICE_PREFIX}(?:{_body(EfaLineGrade.DEGRADED)})"
EFA_INFORMATIONAL_REGEX = f"{EFA_DEVICE_PREFIX}(?:{_body(EfaLineGrade.INFORMATIONAL)})"
EFA_FATAL_PATTERN = re.compile(EFA_FATAL_REGEX)
EFA_DEGRADED_PATTERN = re.compile(EFA_DEGRADED_REGEX)
EFA_INFORMATIONAL_PATTERN = re.compile(EFA_INFORMATIONAL_REGEX)


def grade_efa_line(message: str) -> EfaLineGrade | None:
    """The grade of one kernel line, ``None`` when it is not EFA vocabulary.

    Fatal wins over degraded wins over informational; the three bodies are
    disjoint by construction, the order only states the fail-safe.
    """

    if EFA_FATAL_PATTERN.search(message):
        return EfaLineGrade.FATAL
    if EFA_DEGRADED_PATTERN.search(message):
        return EfaLineGrade.DEGRADED
    if EFA_INFORMATIONAL_PATTERN.search(message):
        return EfaLineGrade.INFORMATIONAL
    return None
