from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from gpu_fault.efa_kernel_log import EFA_DEGRADED_REGEX, EFA_FATAL_REGEX
from gpu_fault.models import RecoveryAction, Severity


class LogHealthCategory(StrEnum):
    MCE = "MCE"
    STORAGE = "STORAGE"
    MEMORY = "MEMORY"
    NCCL = "NCCL"
    TRAINING = "TRAINING"
    RDMA = "RDMA"
    SYSTEM_LOG = "SYSTEM_LOG"


#: Log-collector origin tags that a workload cannot forge. ``dmesg`` and
#: ``journal`` are read by the node collector from the kernel ring buffer and
#: the systemd journal (see ``collectors/logs/node.py`` ``_parse_journal``); a
#: training job cannot write into either. A hardware-fatal rule (MCE, ECC,
#: a dead RDMA/EFA device, kernel panic) may only fire from one of these,
#: because a match drives isolation/quarantine.
KERNEL_LOG_SOURCES: frozenset[str] = frozenset({"dmesg", "journal"})

#: Origins that a workload can influence -- the training log is a file the job
#: writes (``collectors/logs/node.py`` tags it ``training-log``). Only
#: observational rules (``NO_ACTION`` / ``RUN_DIAGNOSTICS``) may match from here, so a job
#: cannot manufacture a trusted quarantine by printing an ``NVRM: Xid`` line.
WORKLOAD_LOG_SOURCES: frozenset[str] = frozenset({"training-log"})

#: Observational rules accept every known origin, kernel and workload alike.
ALL_LOG_SOURCES: frozenset[str] = KERNEL_LOG_SOURCES | WORKLOAD_LOG_SOURCES


@dataclass(frozen=True)
class NodeLogRule:
    pattern: re.Pattern[str]
    category: LogHealthCategory
    severity: Severity
    action: RecoveryAction
    reason: str
    #: Collector origins that may satisfy this rule. A log entry whose
    #: ``source`` is outside this set never matches the rule, so an
    #: untrusted origin cannot drive a hardware-fatal action.
    sources: frozenset[str]

    def allows(self, source: str | None) -> bool:
        """Whether an entry from ``source`` may satisfy this rule.

        An entry that carries no origin (``None``) is treated as untrusted
        and never matches: the fail-safe is to skip, not to trust.
        """

        return source is not None and source in self.sources


NODE_LOG_RULES = (
    # Every x86 machine-check print carries one of the first three tokens
    # (``mce: [Hardware Error]: ...`` from mce/core.c, ``[Hardware Error]: ...``
    # from edac/mce_amd.c, ``{N}[Hardware Error]: ...`` from apei/ghes.c,
    # ``EDAC skx MC0: CPU n: Machine Check ...``). The EDAC core reports an
    # uncorrected memory error only as ``EDAC MC0: 1 UE memory read error on
    # ...`` (edac_mc.c ``edac_ue_error``), so the token ``UE`` -- upper case,
    # case-insensitivity switched off for it -- is the fourth way in; its
    # corrected twin ``CE`` is deliberately not a trigger (no finding). Whether
    # a matched line is corrected is decided by ``mce_grading``.
    NodeLogRule(
        re.compile(
            r"\b(machine check|mce:|hardware error|edac.*(?:uncorrect|(?-i:\bUE\b)))",
            re.I,
        ),
        LogHealthCategory.MCE,
        Severity.CRITICAL,
        RecoveryAction.QUARANTINE,
        "CPU or memory hardware error",
        KERNEL_LOG_SOURCES,
    ),
    NodeLogRule(
        re.compile(
            r"(remounting filesystem read-only|buffer i/o error|"
            r"blk_update_request.*error)",
            re.I,
        ),
        LogHealthCategory.STORAGE,
        Severity.CRITICAL,
        RecoveryAction.QUARANTINE,
        "persistent storage I/O failure",
        KERNEL_LOG_SOURCES,
    ),
    NodeLogRule(
        re.compile(r"\b(out of memory|oom-kill|killed process)\b", re.I),
        LogHealthCategory.MEMORY,
        Severity.WARNING,
        RecoveryAction.RUN_DIAGNOSTICS,
        "host or workload out-of-memory event",
        ALL_LOG_SOURCES,
    ),
    NodeLogRule(
        re.compile(r"\b(nccl).*(error|timeout|unhandled|abort)", re.I),
        LogHealthCategory.NCCL,
        Severity.WARNING,
        RecoveryAction.NO_ACTION,
        "NCCL collective failure",
        ALL_LOG_SOURCES,
    ),
    NodeLogRule(
        re.compile(
            r"(torch\.distributed|distributed backend|collective)"
            r".*(error|timeout|abort|failed)",
            re.I,
        ),
        LogHealthCategory.TRAINING,
        Severity.WARNING,
        RecoveryAction.NO_ACTION,
        "distributed training failure",
        ALL_LOG_SOURCES,
    ),
    # EFA prints no "link down" / "fatal error" line; what means a dead device
    # is its admin queue -- "Admin queue is closed", "Wait for completion
    # (polling) timeout", "The device didn't send any completion ...", a failed
    # reset handshake. The reviewed vocabulary, quoted from Linux
    # v6.1.188/v6.12.103 and amzn-drivers r3.4.0, lives in
    # ``gpu_fault.efa_kernel_log``; the kernel's spelling is fixed, so that part
    # is case-sensitive. Other fabrics keep the keyword form: a line naming
    # RDMA/InfiniBand with a fatal/error/timeout/link-down word.
    NodeLogRule(
        re.compile(
            rf"(?-i:{EFA_FATAL_REGEX})|"
            r"(?:\brdma\b|\binfiniband\b|\bib_[a-z0-9_]+\b).{0,160}"
            r"\b(?:fatal|error|timeout|(?:link|port)\s+down)\b",
            re.I,
        ),
        LogHealthCategory.RDMA,
        Severity.CRITICAL,
        RecoveryAction.QUARANTINE,
        "RDMA/EFA/InfiniBand failure",
        KERNEL_LOG_SOURCES,
    ),
    # The EFA driver reporting the device or firmware misbehaving, or the
    # device failing to probe (unexpected completion ids, EQ/CQ anomalies,
    # version/BAR/MSI-X failures): the device is still answering, so this is
    # read-only fabric diagnostics (FREEZE_EVIDENCE -> VALIDATE_FABRIC), not
    # isolation. Per-command failures the device answered ("Failed to create
    # qp [-12]") are a job's resource problem and match nothing here.
    NodeLogRule(
        re.compile(EFA_DEGRADED_REGEX),
        LogHealthCategory.RDMA,
        Severity.WARNING,
        RecoveryAction.RUN_DIAGNOSTICS,
        "RDMA/EFA device degraded (kernel driver reports a failing device or probe)",
        KERNEL_LOG_SOURCES,
    ),
    NodeLogRule(
        re.compile(
            r"(kernel panic|soft lockup|hard lockup|task .* blocked)",
            re.I,
        ),
        LogHealthCategory.SYSTEM_LOG,
        Severity.CRITICAL,
        RecoveryAction.QUARANTINE,
        "kernel liveness failure",
        KERNEL_LOG_SOURCES,
    ),
)


def matching_log_rules(message: str, source: str | None = None) -> list[NodeLogRule]:
    """Rules whose pattern matches ``message``.

    When ``source`` is given, a rule matches only if it also allows that
    origin -- the trusted-source gate. When ``source`` is omitted the origin
    is not considered; that form is only for node-side signal ranking, which
    never produces a trusted marker on its own. The authoritative,
    marker-producing matcher in :mod:`gpu_fault.host_health` always passes the
    entry's origin.
    """

    return [
        rule
        for rule in NODE_LOG_RULES
        if rule.pattern.search(message) and (source is None or rule.allows(source))
    ]


def log_signal_priority(message: str, source: str | None = None) -> int:
    ranks = {
        Severity.INFO: 0,
        Severity.WARNING: 1,
        Severity.CRITICAL: 2,
        Severity.FATAL: 3,
    }
    return max(
        (ranks[rule.severity] for rule in matching_log_rules(message, source)),
        default=-1,
    )
