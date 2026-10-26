"""Pin the corrected-vs-uncorrected grading of kernel machine-check lines.

Every message below is the text a Linux kernel prints, built from the quoted
format strings in :mod:`gpu_fault.mce_grading` (Linux v6.12 ``mce/core.c``,
``asm/mce.h``, ``edac/mce_amd.c``, ``edac/skx_common.c``, ``efi/cper.c``,
``apei/ghes.c``) or the IA32_MCi_STATUS bit layout of the Intel SDM Vol. 3B
section 18.3.2.2. A change that regrades one of them must come with a new
vendor quote, not a new opinion.
"""

from __future__ import annotations

import pytest

from gpu_fault.mce_grading import (
    MCI_STATUS_DEFERRED,
    MCI_STATUS_PCC,
    MCI_STATUS_UC,
    MCI_STATUS_VAL,
    MceGrade,
    decode_mci_status,
    grade_mce_line,
)

#: ``__print_mce()`` header of a polled (CMCI/timer) error: no " Exception"
#: because MCG_STATUS.MCIP is clear; IA32_MCi_STATUS 0x8c00004000010090 has
#: VAL set and UC/PCC clear -- the processor corrected it.
CORRECTED_HEADER = (
    "mce: [Hardware Error]: CPU 1: Machine Check: 0 Bank 7: 8c00004000010090"
)
#: The same header in #MC context (MCIP set) with UC|EN|PCC set: fatal.
EXCEPTION_HEADER = (
    "mce: [Hardware Error]: CPU 3: Machine Check Exception: 5 Bank 7: be00000000800400"
)
#: ``mce_notify_irq()``: the rate-limited notice that records were logged.
EVENTS_LOGGED = "mce: [Hardware Error]: Machine check events logged"


def _status(*bits: int) -> str:
    return f"{sum(bits):016x}"


@pytest.mark.parametrize(
    "message, grade",
    [
        (CORRECTED_HEADER, MceGrade.CORRECTED),
        (EXCEPTION_HEADER, MceGrade.UNCORRECTED),
        # Polled but uncorrected (UCNA / patrol scrub): UC set, no exception.
        (
            "mce: [Hardware Error]: CPU 3: Machine Check: 0 Bank 7: "
            + _status(MCI_STATUS_VAL, MCI_STATUS_UC, 0x10090),
            MceGrade.UNCORRECTED,
        ),
        # AMD deferred error (bit 44): uncorrected, exception deferred.
        (
            "mce: [Hardware Error]: CPU 9: Machine Check: 0 Bank 18: "
            + _status(MCI_STATUS_VAL, MCI_STATUS_DEFERRED, 0x11B),
            MceGrade.UNCORRECTED,
        ),
        # A header whose status word is not VAL is nothing the kernel logs;
        # fail closed (the rule's own CRITICAL grade stands).
        (
            "mce: [Hardware Error]: CPU 1: Machine Check: 0 Bank 7: "
            + _status(0x10090),
            MceGrade.UNKNOWN,
        ),
        # PCC without UC never happens in hardware; still never corrected.
        (
            "mce: [Hardware Error]: CPU 1: Machine Check: 0 Bank 7: "
            + _status(MCI_STATUS_VAL, MCI_STATUS_PCC),
            MceGrade.UNCORRECTED,
        ),
        # An exception header is uncorrected even when its own bank was
        # corrected: the #MC context means another bank was not.
        (
            "mce: [Hardware Error]: CPU 3: Machine Check Exception: 5 Bank 7: "
            + _status(MCI_STATUS_VAL, 0x10090),
            MceGrade.UNCORRECTED,
        ),
        (EVENTS_LOGGED, MceGrade.CORRECTED),
        # __print_mce() detail lines carry no grade of their own.
        (
            "mce: [Hardware Error]: TSC 2b5f3e1c0a ADDR 1234567890 MISC 86",
            MceGrade.DETAIL,
        ),
        (
            "mce: [Hardware Error]: PROCESSOR 0:50654 TIME 1761451200 SOCKET 0 APIC 6 "
            "microcode 2007006",
            MceGrade.DETAIL,
        ),
        (
            "mce: [Hardware Error]: RIP !INEXACT! 10:<ffffffff81b4c3a2> "
            "{native_irq_return_iret+0x0/0x0}",
            MceGrade.DETAIL,
        ),
        (
            "mce: [Hardware Error]: Run the above through 'mcelog --ascii'",
            MceGrade.DETAIL,
        ),
        # mce_panic() lines keep the rule's CRITICAL grade.
        (
            "mce: [Hardware Error]: Machine check: Processor context corrupt",
            MceGrade.UNKNOWN,
        ),
        (
            "mce: [Hardware Error]: Fake kernel panic: Fatal machine check",
            MceGrade.UNCORRECTED,
        ),
        # CMCI storm notices are corrected-error floods.
        ("mce: CPU3 BANK7 CMCI inherited storm", MceGrade.CORRECTED),
        ("CPU3 BANK7 CMCI storm detected", MceGrade.CORRECTED),
        # drivers/edac/mce_amd.c decode_error_status() + header.
        ("[Hardware Error]: Corrected error, no action required.", MceGrade.CORRECTED),
        ("[Hardware Error]: Deferred error, no action required.", MceGrade.UNCORRECTED),
        (
            "[Hardware Error]: Uncorrected, software restartable error.",
            MceGrade.UNCORRECTED,
        ),
        ("[Hardware Error]: System Fatal error.", MceGrade.UNCORRECTED),
        (
            "[Hardware Error]: CPU:0 (17:31:0) MC27_STATUS[-|CE|MiscV|-|AddrV|-|-|SyndV|CECC]: "
            "0x9c2040000000011b",
            MceGrade.CORRECTED,
        ),
        (
            "[Hardware Error]: CPU:0 (17:31:0) MC27_STATUS[Over|UE|MiscV|-|AddrV|PCC|-|SyndV|UECC]: "
            "0xfe0040000000011b",
            MceGrade.UNCORRECTED,
        ),
        ("[Hardware Error]: Error Addr: 0x000000207d6e9fc0", MceGrade.DETAIL),
        ("[Hardware Error]: IPID: 0x0000009600050f00", MceGrade.DETAIL),
        ("[Hardware Error]: PPIN: 0x00f1a2b3c4d5e6f7", MceGrade.DETAIL),
        (
            "[Hardware Error]: Unified Memory Controller Ext. Error Code: 0, DRAM ECC error.",
            MceGrade.DETAIL,
        ),
        # drivers/edac/skx_common.c: "Event" is the polled form of the header.
        (
            "EDAC skx MC0: CPU 0: Machine Check Event: 0x0 Bank 7: 0x8c00004000010090",
            MceGrade.CORRECTED,
        ),
        (
            "EDAC skx MC0: CPU 0: Machine Check Exception: 0x5 Bank 7: 0xbe00000000800400",
            MceGrade.UNCORRECTED,
        ),
        # drivers/edac/edac_mc.c: CE / UE tokens.
        (
            "EDAC MC0: 1 CE memory read error on CPU_SrcID#0_MC#0_Chan#0_DIMM#0 "
            "(channel:0 slot:0 page:0x1a2b3c offset:0x0 grain:32 syndrome:0x0 - err_code:0x0080:0x0090)",
            MceGrade.CORRECTED,
        ),
        (
            "EDAC MC0: 1 UE memory read error on CPU_SrcID#0_MC#0_Chan#0_DIMM#0 "
            "(channel:0 slot:0 page:0x1a2b3c offset:0x0 grain:32 - err_code:0x00a0:0x0090)",
            MceGrade.UNCORRECTED,
        ),
        # Firmware-first (APEI/GHES + CPER): the severity line grades the
        # block, its header and section lines are detail.
        (
            "{1}[Hardware Error]: Hardware error from APEI Generic Hardware Error Source: 0",
            MceGrade.DETAIL,
        ),
        (
            "{1}[Hardware Error]: It has been corrected by h/w and requires no further action",
            MceGrade.CORRECTED,
        ),
        ("{1}[Hardware Error]: event severity: corrected", MceGrade.CORRECTED),
        ("{1}[Hardware Error]: event severity: info", MceGrade.CORRECTED),
        ("{1}[Hardware Error]: event severity: recoverable", MceGrade.UNCORRECTED),
        ("{1}[Hardware Error]: event severity: fatal", MceGrade.UNCORRECTED),
        ("{1}[Hardware Error]:  Error 0, type: corrected", MceGrade.CORRECTED),
        ("{1}[Hardware Error]:  Error 0, type: fatal", MceGrade.UNCORRECTED),
        ("{1}[Hardware Error]:   section_type: memory error", MceGrade.DETAIL),
        ("{1}[Hardware Error]:   error_type: 2, single-bit ECC", MceGrade.DETAIL),
        (
            "{1}[Hardware Error]:   physical_address: 0x000000207d6e9fc0",
            MceGrade.DETAIL,
        ),
        # The GF-GENERAL-LIVE-016 reviewed lines of the earlier harness: not
        # kernel formats, graded by their words as before.
        (
            "mce: [Hardware Error]: Machine check events logged: corrected error CPU 1",
            MceGrade.CORRECTED,
        ),
        (
            "mce: [Hardware Error]: Machine check events logged: uncorrected error CPU 3",
            MceGrade.UNCORRECTED,
        ),
        # A corrected word never masks an uncorrected one on the same line.
        (
            "mce: uncorrectable hardware error; corrected errors were also logged",
            MceGrade.UNCORRECTED,
        ),
        ("mce: corrected hardware error", MceGrade.CORRECTED),
        # A line with none of the markers keeps the rule's grade.
        ("mce: [Hardware Error]: something new the kernel says", MceGrade.UNKNOWN),
    ],
)
def test_kernel_mce_lines_are_graded_by_their_status_bits_and_words(message, grade):
    assert grade_mce_line(message) is grade, message


@pytest.mark.parametrize(
    "value, grade",
    [
        (MCI_STATUS_VAL | 0x10090, MceGrade.CORRECTED),
        (MCI_STATUS_VAL | MCI_STATUS_UC | 0x10090, MceGrade.UNCORRECTED),
        (MCI_STATUS_VAL | MCI_STATUS_PCC, MceGrade.UNCORRECTED),
        (MCI_STATUS_VAL | MCI_STATUS_DEFERRED, MceGrade.UNCORRECTED),
        (0x10090, MceGrade.UNKNOWN),
        (0, MceGrade.UNKNOWN),
    ],
)
def test_mci_status_decode_follows_the_sdm_bit_layout(value, grade):
    assert decode_mci_status(value) is grade, hex(value)


def test_status_bits_are_the_architectural_positions():
    # Intel SDM Vol. 3B 18.3.2.2 / Linux asm/mce.h: VAL 63, UC 61, PCC 57;
    # AMD deferred exception bit 44 (asm/mce.h MCI_STATUS_DEFERRED).
    assert (MCI_STATUS_VAL, MCI_STATUS_UC, MCI_STATUS_PCC, MCI_STATUS_DEFERRED) == (
        1 << 63,
        1 << 61,
        1 << 57,
        1 << 44,
    ), "the bit positions are the vendor's, not ours"


def test_grading_is_case_insensitive_for_words_but_not_for_edac_tokens():
    assert grade_mce_line("MCE: CORRECTED ERROR") is MceGrade.CORRECTED, (
        "the corrected word is matched regardless of case"
    )
    # Lowercase "ce"/"ue" are ordinary syllables, not EDAC tokens.
    assert (
        grade_mce_line("mce: [Hardware Error]: the ce path, ue later")
        is MceGrade.UNKNOWN
    ), "only the upper-case EDAC tokens CE/UE/CECC/UECC grade a line"
