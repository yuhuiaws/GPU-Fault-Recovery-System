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


@dataclass(frozen=True)
class NodeLogRule:
    pattern: re.Pattern[str]
    category: LogHealthCategory
    severity: Severity
    action: RecoveryAction
    reason: str


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
    ),
    NodeLogRule(
        re.compile(r"\b(out of memory|oom-kill|killed process)\b", re.I),
        LogHealthCategory.MEMORY,
        Severity.WARNING,
        RecoveryAction.RUN_DIAGNOSTICS,
        "host or workload out-of-memory event",
    ),
    NodeLogRule(
        re.compile(r"\b(nccl).*(error|timeout|unhandled|abort)", re.I),
        LogHealthCategory.NCCL,
        Severity.WARNING,
        RecoveryAction.RUN_DIAGNOSTICS,
        "NCCL collective failure",
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
    ),
)


def matching_log_rules(message: str) -> list[NodeLogRule]:
    return [rule for rule in NODE_LOG_RULES if rule.pattern.search(message)]


def log_signal_priority(message: str) -> int:
    ranks = {
        Severity.INFO: 0,
        Severity.WARNING: 1,
        Severity.CRITICAL: 2,
        Severity.FATAL: 3,
    }
    return max(
        (ranks[rule.severity] for rule in matching_log_rules(message)),
        default=-1,
    )
