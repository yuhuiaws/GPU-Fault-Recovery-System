"""Grade a kernel machine-check (MCE) log line: corrected, uncorrected, detail.

The node log rule for MCE (``log_rules.NODE_LOG_RULES``) is hardware-fatal:
CRITICAL / QUARANTINE. The kernel, however, prints every machine-check record
the same way whether the processor corrected it or not, and a corrected error
must end in diagnostics, never in isolation (fault table row "日志：MCE /
存储只读 / RDMA 致命 / 内核 lockup"). This module decides which it is from the
kernel's own output, never from prose. Everything it keys on is quoted below
from the source that prints it; a change here needs a new quote.

Linux v6.12, ``arch/x86/kernel/cpu/mce/core.c``
(https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/tree/arch/x86/kernel/cpu/mce/core.c?h=v6.12):

* ``__print_mce()``::

      pr_emerg(HW_ERR "CPU %d: Machine Check%s: %Lx Bank %d: %016Lx\\n",
               m->extcpu,
               (m->mcgstatus & MCG_STATUS_MCIP ? " Exception" : ""),
               m->mcgstatus, m->bank, m->status);

  " Exception" is printed only in #MC context (``MCG_STATUS_MCIP``); a polled
  or CMCI record prints "Machine Check:". The trailing word is IA32_MCi_STATUS.
  The lines after it -- ``"RIP%s %02x:<%016Lx> "``, ``"TSC %llx "`` (+ ``ADDR``
  / ``MISC`` / ``PPIN`` / ``SYND`` / ``IPID``), ``"PROCESSOR %u:%x TIME %llu
  SOCKET %u APIC %x microcode %x"`` and ``"Run the above through 'mcelog
  --ascii'"`` -- describe the record whose header carries the grade.
* ``mce_notify_irq()``: ``pr_info(HW_ERR "Machine check events logged\\n")``,
  rate-limited to two per minute: the notice that records were logged. It has
  no severity of its own; whatever was uncorrected is printed on its own line.
* ``mce_panic()``: ``pr_emerg(HW_ERR "Machine check: %s\\n", exp)`` with the
  severity-table message ("Processor context corrupt", ...) and the panic
  messages ("Fatal machine check", ...): never corrected.
* ``internal.h``: ``#define pr_fmt(fmt) "mce: " fmt``; ``include/linux/printk.h``:
  ``#define HW_ERR "[Hardware Error]: "``.

Linux v6.12, ``arch/x86/include/asm/mce.h``: ``MCI_STATUS_VAL BIT_ULL(63)``
(valid error), ``MCI_STATUS_UC BIT_ULL(61)`` (uncorrected error),
``MCI_STATUS_PCC BIT_ULL(57)`` (processor context corrupt),
``MCI_STATUS_DEFERRED BIT_ULL(44)`` (uncorrected error, deferred exception),
``MCG_STATUS_MCIP BIT_ULL(2)`` (machine check in progress).

Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 3B,
Order Number 253669-093US, September 2026, Section 18.3.2.2 "IA32_MCi_STATUS
MSRS" (https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html,
Volume 3B download https://cdrdv2.intel.com/v1/dl/getContent/671427):
"UC (error uncorrected) flag, bit 61 -- Indicates (when set) that the
processor did not or was not able to correct the error condition. When clear,
this flag indicates that the processor was able to correct the error
condition." The same section defines VAL (bit 63) and PCC (bit 57).

Linux v6.12, ``drivers/edac/mce_amd.c`` ``amd_decode_mce()``: first
``pr_emerg(HW_ERR "%s\\n", decode_error_status(m))`` with "Corrected error, no
action required." / "Deferred error, no action required." / "Uncorrected,
software restartable error." / "Uncorrected, software containable error." /
"System Fatal error."; then ``pr_emerg(HW_ERR "CPU:%d (%x:%x:%x)
MC%d_STATUS[%s|%s|%s|%s|%s", ..., ((m->status & MCI_STATUS_UC) ? "UE" :
(m->status & MCI_STATUS_DEFERRED) ? "-" : "CE"), ...)`` closed by ``pr_cont("]:
0x%016llx\\n", m->status)``, with ``pr_cont("|%sECC", ((ecc == 2) ? "C" :
"U"))``; then the detail lines ``"Error Addr: 0x%016llx"``, ``"PPIN:
0x%016llx"``, ``"IPID: 0x%016llx"`` (+ ``", Syndrome: 0x%016llx"``),
``"%s Ext. Error Code: %d"`` and the legacy ``"MC%d Error: "``.

Linux v6.12, ``drivers/edac/skx_common.c`` ``skx_mce_check_error()``: ``type =
"Exception"`` when ``MCG_STATUS_MCIP`` else ``"Event"``; ``"CPU %d: Machine
Check %s: 0x%llx Bank %d: 0x%llx"`` under the ``"EDAC skx MC%d: "`` prefix.
``drivers/edac/edac_mc.c``: ``"%d CE %s%son %s (...)"`` (``edac_ce_error``) and
``"%d UE %s%son %s (...)"`` (``edac_ue_error``) under ``"EDAC MC%d: "``.

Linux v6.12, firmware-first path: ``drivers/acpi/apei/ghes.c``
``ghes_print_estatus()`` prefixes every line with ``"%s{%u}" HW_ERR`` and
prints ``"Hardware error from APEI Generic Hardware Error Source: %d"``, then
``drivers/firmware/efi/cper.c`` ``cper_estatus_print()`` prints ``"It has been
corrected by h/w and requires no further action"`` for ``CPER_SEV_CORRECTED``,
``"event severity: %s"`` and per section ``"Error %d, type: %s"`` with
``severity_strs[] = {"recoverable", "fatal", "corrected", "info"}``; the
section body (``section_type``, ``error_status``, ``physical_address``, ...)
carries no severity.

Linux v6.12, ``arch/x86/kernel/cpu/mce/threshold.c`` / ``intel.c``: ``"CPU%d
BANK%d CMCI storm detected"``, ``"... CMCI storm subsided"``, ``"... CMCI
inherited storm"`` -- a storm of corrected errors.

Grading order, fail closed: an uncorrected marker wins over everything; a
decodable IA32_MCi_STATUS word decides by its bits; then a corrected marker;
then a known detail line (no grade of its own: the caller records no finding
for it, the header of the same record already did); anything else is
``UNKNOWN`` and keeps the rule's own CRITICAL grade.
"""

from __future__ import annotations

import re
from enum import StrEnum

#: IA32_MCi_STATUS / MCA_STATUS bits (Intel SDM Vol. 3B 18.3.2.2; asm/mce.h).
MCI_STATUS_VAL = 1 << 63
MCI_STATUS_UC = 1 << 61
MCI_STATUS_PCC = 1 << 57
MCI_STATUS_DEFERRED = 1 << 44


class MceGrade(StrEnum):
    CORRECTED = "CORRECTED"
    UNCORRECTED = "UNCORRECTED"
    #: A line that describes a record graded by its own header line.
    DETAIL = "DETAIL"
    UNKNOWN = "UNKNOWN"


_UNCORRECTED_WORDS = re.compile(
    r"\b(?:uncorrect(?:ed|able)|fatal|recoverable|deferred error)\b"
    r"|\bMachine Check Exception\b"
    r"|event severity:\s*(?:fatal|recoverable)\b"
    r"|\bError \d+, type:\s*(?:fatal|recoverable)\b",
    re.I,
)
#: EDAC / mce_amd tokens are upper case on purpose: "ce" and "ue" are syllables.
_EDAC_UNCORRECTED_TOKENS = re.compile(r"\b(?:UE|UECC)\b")
_EDAC_CORRECTED_TOKENS = re.compile(r"\b(?:CE|CECC)\b")
#: ``__print_mce`` / skx "Machine Check[ Event]: <mcgstatus> Bank <n>: <status>"
#: and mce_amd "MC<n>_STATUS[...]: 0x<status>".
_STATUS_WORD = re.compile(
    r"(?:\bMachine Check(?: Event)?:\s*(?:0x)?[0-9a-f]+\s+Bank\s+\d+:"
    r"|\bMC\d+_STATUS\[[^\]]*\]:)"
    r"\s*(?:0x)?([0-9a-f]{1,16})\b",
    re.I,
)
_CORRECTED_WORDS = re.compile(
    r"\bcorrect(?:ed|able)\b"
    r"|\bMachine check events logged\b"
    r"|\bCMCI\b.*\bstorm\b"
    r"|event severity:\s*info\b"
    r"|\bError \d+, type:\s*info\b",
    re.I,
)
_HW_ERR_DETAIL = re.compile(
    r"\[Hardware Error\]:\s*(?:RIP\b|TSC\b|PROCESSOR\b|Run the above through"
    r"|Error Addr:|PPIN:|IPID:|Syndrome:|MC\d+ Error:|Corrupted MC\d+ MCE info"
    r"|[^:\[\]]*\bExt\. Error Code:)"
)
_GHES_LINE = re.compile(r"^\s*\{\d+\}\[Hardware Error\]:")


def decode_mci_status(value: int) -> MceGrade:
    """Grade an IA32_MCi_STATUS word by VAL, UC, PCC and the AMD deferred bit."""

    if not value & MCI_STATUS_VAL:
        return MceGrade.UNKNOWN
    if value & (MCI_STATUS_UC | MCI_STATUS_PCC | MCI_STATUS_DEFERRED):
        return MceGrade.UNCORRECTED
    return MceGrade.CORRECTED


def grade_mce_line(message: str) -> MceGrade:
    """Grade one kernel line that matched the MCE log rule."""

    if _UNCORRECTED_WORDS.search(message) or _EDAC_UNCORRECTED_TOKENS.search(message):
        return MceGrade.UNCORRECTED
    status = _STATUS_WORD.search(message)
    if status is not None:
        return decode_mci_status(int(status.group(1), 16))
    if _CORRECTED_WORDS.search(message) or _EDAC_CORRECTED_TOKENS.search(message):
        return MceGrade.CORRECTED
    if _HW_ERR_DETAIL.search(message) or _GHES_LINE.match(message):
        return MceGrade.DETAIL
    return MceGrade.UNKNOWN


__all__ = [
    "MCI_STATUS_DEFERRED",
    "MCI_STATUS_PCC",
    "MCI_STATUS_UC",
    "MCI_STATUS_VAL",
    "MceGrade",
    "decode_mci_status",
    "grade_mce_line",
]
