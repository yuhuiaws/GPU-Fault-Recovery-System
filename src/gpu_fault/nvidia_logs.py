"""Normalize NVIDIA kernel and Fabric Manager fault records."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from gpu_fault.models import StrictModel, WorkloadState
from gpu_fault.policy import (
    NVIDIA_ALWAYS_FATAL_SXIDS,
    FaultPolicyDecision,
    SxidClassification,
    SxidEvent,
    SxidLinkScope,
    XidEvent,
    load_sxid_policy,
)

_SXID_CATALOG_CODES = frozenset(rule.sxid for rule in load_sxid_policy().rules)

_XID_PATTERN = re.compile(
    r"\bXid\b\s*(?:\((?:PCI:)?([0-9a-fA-F:.]+)\))?\s*"
    r"(?::|=)?\s*(\d+)",
    re.IGNORECASE,
)
# The bare tokens the node-side collectors filter on. A line that carries one
# of them but yields no code to ``_XID_PATTERN`` / ``_SXID_PATTERN`` is a
# format drift the ingest side must name, not swallow: no code is invented and
# the reason travels on the provider signal so the fault ingestion service can
# turn it into a finding.
_XID_TOKEN_PATTERN = re.compile(r"\bXid\b", re.IGNORECASE)
_SXID_TOKEN_PATTERN = re.compile(r"\bSXid\b", re.IGNORECASE)
UNPARSED_XID_REASON = "unparsed_xid_line"
UNPARSED_SXID_REASON = "unparsed_sxid_line"
UNCLASSIFIED_SXID_REASON = "unclassified_sxid"
# Retained for historical finding/notification decoding, not active HMA ingest.
UNSCHEDULABLE_WITHOUT_CODE_REASON = "hma_unschedulable_without_code"


def unresolved_reason_kind(reason: str) -> str:
    """The finding kind an unresolved provider-signal reason maps to."""

    for kind in (
        UNPARSED_XID_REASON,
        UNPARSED_SXID_REASON,
        UNSCHEDULABLE_WITHOUT_CODE_REASON,
    ):
        if reason.startswith(kind):
            return kind
    return UNCLASSIFIED_SXID_REASON


def _unparsed_line_reasons(
    message: str,
    xid_matches: list[tuple[str | None, int, str]],
    sxid_matches: list[tuple[str | None, int, str]],
) -> list[str]:
    reasons = []
    if not xid_matches and _XID_TOKEN_PATTERN.search(message):
        reasons.append(
            f"{UNPARSED_XID_REASON}: an Xid token is present but no "
            "code could be extracted"
        )
    if not sxid_matches and _SXID_TOKEN_PATTERN.search(message):
        reasons.append(
            f"{UNPARSED_SXID_REASON}: an SXid token is present but no "
            "code could be extracted"
        )
    return reasons


_XID_REGISTER_PATTERN = re.compile(r"\b0x([0-9a-fA-F]+)\b")
_NVLINK5_XIDS = frozenset(range(144, 151))
_NVLINK5_REGISTER_TOKEN = r"(?:0[xX][0-9a-fA-F]{1,8}|[0-9a-fA-F]{8})"
_NVLINK5_REGISTER_PAYLOAD_PATTERN = re.compile(
    r"\(\s*"
    rf"(?P<intr_info>{_NVLINK5_REGISTER_TOKEN})"
    r"(?:\s*,\s*|\s+)"
    rf"(?P<error_status>{_NVLINK5_REGISTER_TOKEN})"
    + "".join(rf"(?:\s*,\s*|\s+){_NVLINK5_REGISTER_TOKEN}" for _ in range(5))
    + r"\s*\)",
)
_SXID_PATTERN = re.compile(
    r"\bSXid\b\s*(?:\(PCI:([0-9a-fA-F:.]+)\))?\s*"
    r"(?::|=)?\s*(\d+)",
    re.IGNORECASE,
)
_TIMESTAMP_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?"
)
_SXID_SWITCH_PATTERN = re.compile(
    r"\bnvidia-nvswitch(?P<switch>\d+)\s*:", re.IGNORECASE
)
_SXID_LINK_PATTERN = re.compile(r"\bLink\s+(?P<link>\d+)\b", re.IGNORECASE)


def _scoped_event_id(cluster_id: str, event_id: str) -> str:
    digest = hashlib.sha256(cluster_id.encode("utf-8")).hexdigest()[:16]
    return f"{event_id}-cluster-{digest}"


class FaultSignalSource(StrEnum):
    # Historical evidence still carries these two retired source values.
    KUBERNETES_NODE = "KUBERNETES_NODE"
    CLOUDWATCH_LOG = "CLOUDWATCH_LOG"
    KERNEL_LOG = "KERNEL_LOG"
    FABRIC_MANAGER_LOG = "FABRIC_MANAGER_LOG"


class NvidiaKernelLogEvent(StrictModel):
    cluster_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    record_id: str
    observed_at: datetime
    source_monotonic_us: int | None = Field(default=None, ge=0)
    source_boot_id: str | None = None
    collected_at: datetime | None = None
    message: str
    runtime_profile_version: str | None = None
    product: str | None = None
    driver_branch: int | None = Field(default=None, ge=0)
    cuda_version: str | None = None
    workload_state: WorkloadState = WorkloadState.UNKNOWN
    affected_workload_ids: list[str] = Field(default_factory=list)
    checkpoint_manifest_ref: str | None = None
    evidence_ref: str | None = None


class FabricManagerLogEvent(StrictModel):
    cluster_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    record_id: str
    observed_at: datetime
    collected_at: datetime | None = None
    message: str
    source: str
    unit: str | None = None
    fields: dict[str, str] = Field(default_factory=dict)
    runtime_profile_version: str | None = None
    product: str | None = None
    driver_branch: int | None = Field(default=None, ge=0)
    cuda_version: str | None = None
    workload_state: WorkloadState = WorkloadState.UNKNOWN
    affected_workload_ids: list[str] = Field(default_factory=list)
    checkpoint_manifest_ref: str | None = None
    evidence_ref: str | None = None


class FaultSignal(StrictModel):
    signal_id: str
    source: FaultSignalSource
    cluster_id: str
    node_id: str
    observed_at: datetime
    health_status: str | None = None
    fault_types: list[str] = Field(default_factory=list)
    fault_reasons: list[str] = Field(default_factory=list)
    fault_details: list[str] = Field(default_factory=list)
    unschedulable_taint: bool = False
    raw_message: str | None = None
    xid_codes: list[int] = Field(default_factory=list)
    sxid_codes: list[int] = Field(default_factory=list)
    unresolved_reasons: list[str] = Field(default_factory=list)
    # The runtime profile the source record was collected under. An unresolved
    # line opens a node-health finding, and the health family cannot compile
    # even a FREEZE_EVIDENCE-only workflow without a profile to name the
    # evidence-capture owner; without this the finding blocked as
    # "runtime_profile_version is required for execution".
    runtime_profile_version: str | None = None


class NormalizedFaultBatch(StrictModel):
    provider_signals: list[FaultSignal]
    xid_events: list[XidEvent] = Field(default_factory=list)
    sxid_events: list[SxidEvent] = Field(default_factory=list)


class FaultIngestionResult(StrictModel):
    normalized: NormalizedFaultBatch
    decisions: list[FaultPolicyDecision] = Field(default_factory=list)
    #: How many fault lines in this batch could not be resolved into an
    #: XID/SXID event. ``decisions == [] and unresolved == 0`` is the only
    #: shape that means "nothing to act on"; a non-zero count says the
    #: provider reported a fault the control plane could not read, and an
    #: operator-review finding was opened for it. Always derived from the
    #: batch below so no caller can report a different number.
    unresolved: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _count_unresolved(self) -> FaultIngestionResult:
        self.unresolved = sum(
            len(signal.unresolved_reasons)
            for signal in self.normalized.provider_signals
        )
        return self


class NvidiaLogNormalizer:
    """Normalize local NVIDIA logs without inferring missing fault evidence."""

    def normalize_kernel(self, event: NvidiaKernelLogEvent) -> NormalizedFaultBatch:
        xid_matches = self._matches([event.message], _XID_PATTERN)
        sxid_matches = self._matches([event.message], _SXID_PATTERN)
        signal = FaultSignal(
            signal_id=(f"kernel-log-{self._safe_id(event.record_id)}"),
            source=FaultSignalSource.KERNEL_LOG,
            cluster_id=event.cluster_id,
            node_id=event.node_id,
            observed_at=event.observed_at,
            runtime_profile_version=event.runtime_profile_version,
            fault_details=[event.message],
            raw_message=event.message,
            xid_codes=sorted({code for _, code, _ in xid_matches}),
            sxid_codes=sorted({code for _, code, _ in sxid_matches}),
        )
        xids = [
            self._xid_event(
                event,
                event_id=_scoped_event_id(
                    event.cluster_id,
                    f"{signal.signal_id}-xid-{code}",
                ),
                xid=code,
                pci_bdf=pci_bdf,
                raw_message=text,
                source_event_time=None,
            )
            for pci_bdf, code, text in self._deduplicate_matches(xid_matches)
        ]
        sxids, unresolved = self._sxid_events(
            event,
            signal.signal_id,
            sxid_matches,
        )
        unresolved = [
            *_unparsed_line_reasons(event.message, xid_matches, sxid_matches),
            *unresolved,
        ]
        if unresolved:
            signal = signal.model_copy(update={"unresolved_reasons": unresolved})
        return NormalizedFaultBatch(
            provider_signals=[signal],
            xid_events=xids,
            sxid_events=sxids,
        )

    def normalize_fabric_manager(
        self, event: FabricManagerLogEvent
    ) -> NormalizedFaultBatch:
        xid_matches = self._matches([event.message], _XID_PATTERN)
        sxid_matches = self._matches([event.message], _SXID_PATTERN)
        signal = FaultSignal(
            signal_id=(f"fabric-manager-log-{self._safe_id(event.record_id)}"),
            source=FaultSignalSource.FABRIC_MANAGER_LOG,
            cluster_id=event.cluster_id,
            node_id=event.node_id,
            observed_at=event.observed_at,
            runtime_profile_version=event.runtime_profile_version,
            fault_details=[event.message],
            raw_message=event.message,
            xid_codes=sorted({code for _, code, _ in xid_matches}),
            sxid_codes=sorted({code for _, code, _ in sxid_matches}),
        )
        xids = [
            self._xid_event(
                event,
                event_id=_scoped_event_id(
                    event.cluster_id,
                    f"{signal.signal_id}-xid-{code}",
                ),
                xid=code,
                pci_bdf=pci_bdf,
                raw_message=text,
                source_event_time=self._text_timestamp(text),
            )
            for pci_bdf, code, text in self._deduplicate_matches(xid_matches)
        ]
        sxids, unresolved = self._sxid_events(
            event,
            signal.signal_id,
            sxid_matches,
        )
        unresolved = [
            *_unparsed_line_reasons(event.message, xid_matches, sxid_matches),
            *unresolved,
        ]
        if unresolved:
            signal = signal.model_copy(update={"unresolved_reasons": unresolved})
        return NormalizedFaultBatch(
            provider_signals=[signal],
            xid_events=xids,
            sxid_events=sxids,
        )

    @staticmethod
    def _xid_event(
        source: NvidiaKernelLogEvent | FabricManagerLogEvent,
        *,
        event_id: str,
        xid: int,
        pci_bdf: str | None,
        raw_message: str,
        observed_at: datetime | None = None,
        source_event_time: datetime | None = None,
    ) -> XidEvent:
        intr_info, error_status = NvidiaLogNormalizer._xid_nvlink5_registers(
            raw_message, xid
        )
        drill_id = NvidiaLogNormalizer._drill_id(raw_message)
        return XidEvent(
            event_id=event_id,
            cluster_id=source.cluster_id,
            node_id=source.node_id,
            observed_at=observed_at or source.observed_at,
            source_event_time=source_event_time,
            source_monotonic_us=getattr(source, "source_monotonic_us", None),
            source_boot_id=getattr(source, "source_boot_id", None),
            collected_at=(getattr(source, "collected_at", None) or source.observed_at),
            event_source=(
                FaultSignalSource.KERNEL_LOG.value
                if isinstance(source, NvidiaKernelLogEvent)
                else FaultSignalSource.FABRIC_MANAGER_LOG.value
            ),
            xid=xid,
            pci_bdf=pci_bdf,
            product=source.product,
            driver_branch=source.driver_branch,
            cuda_version=source.cuda_version,
            runtime_profile_version=source.runtime_profile_version,
            workload_state=source.workload_state,
            affected_workload_ids=source.affected_workload_ids,
            checkpoint_manifest_ref=source.checkpoint_manifest_ref,
            intr_info=intr_info,
            error_status=error_status,
            registers=(
                NvidiaLogNormalizer._xid_registers(raw_message, xid)
                if xid == 74
                else []
            ),
            nvlink_link_id=(
                NvidiaLogNormalizer._xid_nvlink_id(raw_message) if xid == 74 else None
            ),
            nvlink_link_identity_source=(
                "explicit-kernel-message"
                if xid == 74
                and NvidiaLogNormalizer._xid_nvlink_id(raw_message) is not None
                else None
            ),
            raw_message=raw_message,
            evidence_ref=source.evidence_ref,
            drill_id=drill_id,
        )

    @staticmethod
    def _drill_id(message: str) -> str | None:
        match = re.search(
            r"\bdrill[_-]id=([A-Za-z0-9][A-Za-z0-9._:-]{0,127})\b",
            message,
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    @staticmethod
    def _xid_registers(message: str, xid: int) -> list[int]:
        for match in _XID_PATTERN.finditer(message):
            if int(match.group(2)) != xid:
                continue
            return [
                int(value, 16)
                for value in _XID_REGISTER_PATTERN.findall(message[match.end() :])
            ]
        return []

    @staticmethod
    def _xid_nvlink5_registers(message: str, xid: int) -> tuple[int | None, int | None]:
        if xid not in _NVLINK5_XIDS:
            return None, None
        matches = list(_XID_PATTERN.finditer(message))
        for index, match in enumerate(matches):
            if int(match.group(2)) != xid:
                continue
            end = (
                matches[index + 1].start() if index + 1 < len(matches) else len(message)
            )
            payload = _NVLINK5_REGISTER_PAYLOAD_PATTERN.search(
                message, match.end(), end
            )
            if payload is None:
                continue
            return (
                int(payload.group("intr_info"), 16),
                int(payload.group("error_status"), 16),
            )
        return None, None

    @staticmethod
    def _xid_nvlink_id(message: str) -> int | None:
        match = re.search(
            r"\b(?:NVLink|Link)\s*(?:ID\s*)?[:=#]?\s*(\d+)\b",
            message,
            re.IGNORECASE,
        )
        return int(match.group(1)) if match else None

    def _sxid_events(
        self,
        source: NvidiaKernelLogEvent | FabricManagerLogEvent,
        signal_id: str,
        matches: list[tuple[str | None, int, str]],
        *,
        observed_at: datetime | None = None,
    ) -> tuple[list[SxidEvent], list[str]]:
        events = []
        unresolved = []
        for pci_bdf, code, text in self._deduplicate_matches(matches):
            classification = self._sxid_classification(text, code)
            if classification is None:
                unresolved.append(
                    f"SXID {code} classification is absent; raw "
                    "Fabric Manager evidence is required"
                )
                continue
            events.append(
                SxidEvent(
                    event_id=_scoped_event_id(
                        source.cluster_id,
                        (f"{signal_id}-sxid-{code}-{self._content_id(text)}"),
                    ),
                    cluster_id=source.cluster_id,
                    node_id=source.node_id,
                    observed_at=observed_at or source.observed_at,
                    source_event_time=(
                        observed_at
                        if observed_at is not None
                        else self._text_timestamp(text)
                    ),
                    source_monotonic_us=getattr(source, "source_monotonic_us", None),
                    source_boot_id=getattr(source, "source_boot_id", None),
                    collected_at=(
                        getattr(source, "collected_at", None) or source.observed_at
                    ),
                    event_source=(
                        FaultSignalSource.KERNEL_LOG.value
                        if isinstance(source, NvidiaKernelLogEvent)
                        else FaultSignalSource.FABRIC_MANAGER_LOG.value
                    ),
                    sxid=code,
                    classification=classification,
                    classification_source=(
                        "NVIDIA_FABRIC_MANAGER_CATALOG"
                        if code in _SXID_CATALOG_CODES
                        else "NVIDIA_FABRIC_MANAGER_RUNTIME"
                    ),
                    link_scope=self._sxid_scope(
                        text,
                        product=source.product,
                        classification=classification,
                    ),
                    link_scope_source=None,
                    switch_id=self._sxid_switch(text),
                    pci_bdf=pci_bdf,
                    port=self._sxid_link(text),
                    product=source.product,
                    runtime_profile_version=(source.runtime_profile_version),
                    workload_state=source.workload_state,
                    affected_workload_ids=(source.affected_workload_ids),
                    checkpoint_manifest_ref=(source.checkpoint_manifest_ref),
                    raw_message=text,
                    evidence_ref=source.evidence_ref,
                    drill_id=self._drill_id(text),
                )
            )
        return events, unresolved

    @staticmethod
    def _matches(
        texts: list[str], pattern: re.Pattern[str]
    ) -> list[tuple[str | None, int, str]]:
        matches = []
        for text in texts:
            for match in pattern.finditer(text):
                matches.append((match.group(1), int(match.group(2)), text))
        return matches

    @staticmethod
    def _deduplicate_matches(
        matches: list[tuple[str | None, int, str]],
    ) -> list[tuple[str | None, int, str]]:
        unique: dict[tuple[str | None, int, str], str] = {}
        for pci_bdf, code, text in matches:
            timestamp = _TIMESTAMP_PATTERN.search(text)
            occurrence = (
                NvidiaLogNormalizer._canonical_timestamp(timestamp.group(0))
                if timestamp
                else NvidiaLogNormalizer._content_id(text)
            )
            unique.setdefault((pci_bdf, code, occurrence), text)
        return [(pci_bdf, code, text) for (pci_bdf, code, _), text in unique.items()]

    @staticmethod
    def _sxid_classification(
        text: str,
        code: int,
    ) -> SxidClassification | None:
        lowered = text.lower()
        if "always fatal" in lowered:
            return SxidClassification.ALWAYS_FATAL
        if "non-fatal" in lowered or "nonfatal" in lowered:
            return SxidClassification.NON_FATAL
        if "fatal" in lowered:
            return (
                SxidClassification.ALWAYS_FATAL
                if code in NVIDIA_ALWAYS_FATAL_SXIDS
                else SxidClassification.FATAL
            )
        return None

    @staticmethod
    def _sxid_scope(
        text: str,
        *,
        product: str | None,
        classification: SxidClassification,
    ) -> SxidLinkScope:
        # Raw descriptions are evidence, not trusted topology. The API
        # resolves switch/port scope from fresh topology telemetry or a
        # pinned product invariant.
        return SxidLinkScope.UNKNOWN

    @staticmethod
    def _sxid_switch(text: str) -> str | None:
        match = _SXID_SWITCH_PATTERN.search(text)
        return match.group("switch") if match else None

    @staticmethod
    def _sxid_link(text: str) -> str | None:
        match = _SXID_LINK_PATTERN.search(text)
        return match.group("link") if match else None

    @staticmethod
    def _timestamp(value: Any, fallback: datetime) -> datetime:
        if not isinstance(value, str):
            return fallback
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return fallback
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    @staticmethod
    def _text_timestamp(value: str) -> datetime | None:
        match = _TIMESTAMP_PATTERN.search(value)
        if match is None:
            return None
        return NvidiaLogNormalizer._timestamp(
            match.group(0), datetime.now(timezone.utc)
        )

    @staticmethod
    def _safe_id(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "-", value)

    @staticmethod
    def _content_id(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _canonical_timestamp(value: str) -> str:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
        except ValueError:
            return value
