from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re

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
#: RDMA/EFA link loss, kernel panic) may only fire from one of these, because a
#: match drives isolation/quarantine.
KERNEL_LOG_SOURCES: frozenset[str] = frozenset({"dmesg", "journal"})

#: Origins that a workload can influence -- the training log is a file the job
#: writes (``collectors/logs/node.py`` tags it ``training-log``). Only
#: observational rules (``RUN_DIAGNOSTICS``) may match from here, so a job
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
    NodeLogRule(
        re.compile(
            r"\b(machine check|mce:|hardware error|edac.*uncorrect)",
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
        RecoveryAction.RUN_DIAGNOSTICS,
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
        RecoveryAction.RUN_DIAGNOSTICS,
        "distributed training failure",
        ALL_LOG_SOURCES,
    ),
    NodeLogRule(
        re.compile(
            r"(?:\brdma\b|\binfiniband\b|\bib_[a-z0-9_]+\b|"
            r"(?:^|[\s\[])efa(?=[:\s])).{0,160}"
            r"\b(?:fatal|error|timeout|(?:link|port)\s+down)\b",
            re.I,
        ),
        LogHealthCategory.RDMA,
        Severity.CRITICAL,
        RecoveryAction.QUARANTINE,
        "RDMA/EFA/InfiniBand failure",
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
