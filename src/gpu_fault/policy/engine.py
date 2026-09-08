from __future__ import annotations

import re
from collections import OrderedDict
from datetime import timedelta
from threading import RLock
from typing import Callable, NamedTuple

from gpu_fault.models import (
    MarkerScope,
    NodeMarker,
    RecoveryAction,
    Severity,
)
from gpu_fault.policy.catalog import (
    load_sxid_policy,
    load_xid_policy,
)
from gpu_fault.policy.models import (
    CONTAINMENT_SEVERITY_RANK,
    DEFAULT_DECISION_CACHE_SIZE,
    DIRECT_ACTION_MAP,
    OPERATOR_SEVERITY_RANK,
    RECOVERY_ACTION_SEVERITY_RANK,
    ActionDisposition,
    ActionSource,
    CatalogRule,
    Containment,
    DynamicRecoveryAction,
    FaultEventType,
    FaultPolicyDecision,
    NvlinkDecodeRule,
    SxidCatalogRule,
    SxidClassification,
    SxidEvent,
    SxidLinkScope,
    SxidPolicy,
    XidEvent,
    XidPolicy,
    _Resolution,
)
from gpu_fault.policy.nvlink74 import resolve_nvlink74
from gpu_fault.policy.product_families import (
    ProductFamilyPolicyMixin,
    ProductFamilyResolver,
)

# The catalog's Immediate Action token for XID 154. It is a workflow like any
# other NVIDIA workflow -- it is dispatched from _XID_WORKFLOW_RESOLVERS rather
# than from a hardcoded ``event.xid == 154`` branch in the middle of
# ``_resolve_xid``'s guard sequence -- but it is also the one token whose
# resolver consumes the recovery action parsed off the event itself.
#
# That action is propagated onto the companion events XID 154 summarises, where
# it overrides their own Immediate Action; on the XID 154 event it must not,
# because ``_workflow_xid_154`` additionally refuses to act while the correlation
# window is still open. So the propagation guard tests the catalog token, which
# keeps every piece of XID 154 routing keyed the same way as the table.
XID_154_ACTION = "XID_154"


class _XidCorrelation(NamedTuple):
    """The correlation evidence the caller read out of the store for one event.

    The engine never queries the store itself, so everything a workflow needs
    beyond the event and its catalog rule arrives here. Passing one value keeps
    every workflow resolver on the same signature, which is what lets them be
    dispatched from a table instead of a branch per NVIDIA action.
    """

    companion_events: list[XidEvent]
    xid45_window_closed: bool
    xid154_window_closed: bool


class GpuFaultPolicyEngine(ProductFamilyPolicyMixin):
    """Resolves one event against the pinned catalogs.

    The engine is deliberately close to stateless: correlation
    candidates are supplied by the caller (the coordinator reads them
    from the shared store) so two HA replicas reach the same verdict.
    The only retained state is a bounded idempotency cache, because an
    unbounded one grew ~10 KB per event and eventually OOMed a
    long-lived API pod.
    """

    def __init__(
        self,
        policy: XidPolicy | None = None,
        sxid_policy: SxidPolicy | None = None,
        *,
        decision_cache_size: int = DEFAULT_DECISION_CACHE_SIZE,
    ) -> None:
        if decision_cache_size < 1:
            raise ValueError("decision cache size must be positive")
        self.policy = policy or load_xid_policy()
        self.sxid_policy = sxid_policy or load_sxid_policy()
        self._rules = {rule.xid: rule for rule in self.policy.catalog_rules}
        if len(self._rules) != len(self.policy.catalog_rules):
            raise ValueError("official catalog contains duplicate XIDs")
        self._sxid_rules = {rule.sxid: rule for rule in self.sxid_policy.rules}
        if len(self._sxid_rules) != len(self.sxid_policy.rules):
            raise ValueError("official catalog contains duplicate SXIDs")
        self.decision_cache_size = decision_cache_size
        self._decisions: OrderedDict[str, FaultPolicyDecision] = OrderedDict()
        self._product_families = ProductFamilyResolver(self.policy.product_families)
        self._lock = RLock()

    def _remember(self, decision: FaultPolicyDecision) -> None:
        """Cache a verdict for idempotent replays, newest-wins."""
        self._decisions[decision.event_id] = decision
        self._decisions.move_to_end(decision.event_id)
        while len(self._decisions) > self.decision_cache_size:
            self._decisions.popitem(last=False)

    def _cached(self, event_id: str) -> FaultPolicyDecision | None:
        decision = self._decisions.get(event_id)
        if decision is not None:
            self._decisions.move_to_end(event_id)
        return decision

    def sxid_catalog_rule(self, sxid: int) -> SxidCatalogRule | None:
        return self._sxid_rules.get(sxid)

    def evaluate_xid(
        self,
        event: XidEvent,
        *,
        companion_events: list[XidEvent] | None = None,
        xid45_window_closed: bool = True,
        xid154_window_closed: bool = True,
    ) -> FaultPolicyDecision:
        with self._lock:
            duplicate = self._cached(event.event_id)
            if duplicate is not None:
                return duplicate.model_copy(update={"duplicate": True})

            resolution = self._resolve_xid(
                event,
                companion_events=companion_events,
                xid45_window_closed=xid45_window_closed,
                xid154_window_closed=xid154_window_closed,
            )
            decision = self._decision(
                event=event,
                event_type=FaultEventType.XID,
                resolution=resolution,
            )
            if decision.disposition is not ActionDisposition.PENDING_CORRELATION:
                self._remember(decision)
            return decision

    def evaluate_sxid(self, event: SxidEvent) -> FaultPolicyDecision:
        with self._lock:
            duplicate = self._cached(event.event_id)
            if duplicate is not None:
                return duplicate.model_copy(update={"duplicate": True})

            resolution = self._resolve_sxid(event)
            decision = self._decision(
                event=event,
                event_type=FaultEventType.SXID,
                resolution=resolution,
            )
            self._remember(decision)
            return decision

    def _resolve_xid(
        self,
        event: XidEvent,
        *,
        companion_events: list[XidEvent] | None = None,
        xid45_window_closed: bool = True,
        xid154_window_closed: bool = True,
    ) -> _Resolution:
        """Resolve one XID into exactly one NVIDIA-sourced outcome.

        The guards run before any workflow and in a fixed order: an XID outside
        the pinned catalog, one that does not apply to this product, and one
        whose evidence does not meet the catalog's version linkage can never
        reach a recovery action. Everything after them is dispatched by the
        catalog's own Immediate Action text.
        """

        correlation = _XidCorrelation(
            companion_events=companion_events or [],
            xid45_window_closed=xid45_window_closed,
            xid154_window_closed=xid154_window_closed,
        )
        rule = self._rules.get(event.xid)
        if rule is None:
            return _Resolution(
                source=ActionSource.SITE_SAFETY,
                disposition=ActionDisposition.SITE_SAFETY,
                official_action=None,
                investigatory_action=None,
                action=None,
                containment=Containment.UNKNOWN,
                reasons=[
                    "XID is absent from the pinned NVIDIA Catalog; "
                    "site safety quarantine is not an NVIDIA action"
                ],
                requires_operator=True,
            )

        for gate in (self._product_gate, self._version_gate):
            blocked = gate(event, rule)
            if blocked is not None:
                return blocked

        official = rule.immediate_action
        # The annotated action overrides a *companion* event's own Immediate
        # Action, never the XID 154 event carrying it; see XID_154_ACTION.
        if event.xid_154_action is not None and official != XID_154_ACTION:
            return self._dynamic_recovery_resolution(event.xid_154_action, rule)

        if official is None:
            return _Resolution(
                source=ActionSource.NVIDIA_CATALOG,
                disposition=ActionDisposition.MONITOR_ONLY,
                official_action=None,
                investigatory_action=rule.investigatory_action,
                action=RecoveryAction.NO_ACTION,
                containment=self._xid_containment(event.xid),
                reasons=[
                    "NVIDIA Catalog defines no Immediate Action"
                    + (
                        f": {rule.trigger_conditions}"
                        if rule.trigger_conditions
                        else ""
                    )
                ],
            )
        if official in DIRECT_ACTION_MAP:
            resolution = self._direct_resolution(
                official,
                rule,
                reason=(
                    f"exact NVIDIA Catalog {self.policy.catalog_version} "
                    f"Immediate Action for XID {event.xid}"
                ),
            )
            return self._with_xid_48_backlink(
                event,
                resolution,
                candidates=correlation.companion_events,
            )
        resolver = _XID_WORKFLOW_RESOLVERS.get(official)
        if resolver is not None:
            return resolver(self, event, rule, correlation)

        return _Resolution(
            source=ActionSource.NVIDIA_CATALOG,
            disposition=ActionDisposition.BLOCKED_WORKFLOW,
            official_action=official,
            investigatory_action=rule.investigatory_action,
            action=None,
            containment=self._xid_containment(event.xid),
            reasons=[
                f"NVIDIA workflow {official} is preserved exactly and "
                "has no approved executor in this release"
            ],
            requires_operator=True,
        )

    def _product_gate(self, event: XidEvent, rule: CatalogRule) -> _Resolution | None:
        if self._product_families.supports(
            event.product,
            rule.products,
            record_unknown=True,
        ):
            return None
        return _Resolution(
            source=ActionSource.NVIDIA_CATALOG,
            disposition=ActionDisposition.NOT_APPLICABLE,
            official_action=rule.immediate_action,
            investigatory_action=rule.investigatory_action,
            action=None,
            containment=Containment.UNKNOWN,
            reasons=[
                f"Catalog XID {event.xid} is not applicable to "
                f"product {event.product or 'UNKNOWN'}"
            ],
            requires_operator=True,
        )

    def _workflow_operator_review(
        self,
        event: XidEvent,
        rule: CatalogRule,
        correlation: _XidCorrelation,
    ) -> _Resolution:
        """Hand an NVIDIA action that has no automated step to an operator.

        ``UPDATE_SWFW`` is included because the update itself is a maintenance
        activity, but unlike the two inspection workflows it does not by itself
        require an operator to close the incident.
        """

        official = rule.immediate_action
        return _Resolution(
            source=ActionSource.NVIDIA_CATALOG,
            disposition=ActionDisposition.EXECUTABLE,
            official_action=official,
            investigatory_action=rule.investigatory_action,
            action=None,
            containment=self._xid_containment(event.xid),
            reasons=[
                f"exact NVIDIA Catalog {self.policy.catalog_version} "
                f"workflow for XID {event.xid}: {official}"
            ],
            requires_operator=official in {"CONTACT_SUPPORT", "CHECK_MECHANICALS"},
        )

    def _workflow_nvlink74(
        self,
        event: XidEvent,
        rule: CatalogRule,
        correlation: _XidCorrelation,
    ) -> _Resolution:
        """Delegate to the XID 74 register decode in ``policy/nvlink74.py``."""

        return resolve_nvlink74(
            event,
            rule,
            family=self._product_families.family(event.product),
        )

    def _version_gate(self, event: XidEvent, rule: CatalogRule) -> _Resolution | None:
        linkage = rule.xid154_linkage or ""
        driver_match = re.search(r"GPU driver R(\d+)", linkage)
        cuda_match = re.search(r"CUDA (\d+\.\d+)", linkage)
        missing = []
        if driver_match and event.driver_branch is None:
            missing.append("driver_branch")
        if cuda_match and event.cuda_version is None:
            missing.append("cuda_version")
        if missing:
            return _Resolution(
                source=ActionSource.NVIDIA_CATALOG,
                disposition=(ActionDisposition.BLOCKED_MISSING_EVIDENCE),
                official_action=rule.immediate_action,
                investigatory_action=rule.investigatory_action,
                action=None,
                containment=self._xid_containment(event.xid),
                reasons=[
                    "Catalog compatibility gate requires "
                    + ", ".join(missing)
                    + f" ({linkage})"
                ],
                requires_operator=True,
            )
        incompatible = []
        if (
            driver_match
            and event.driver_branch is not None
            and event.driver_branch < int(driver_match.group(1))
        ):
            incompatible.append(
                f"driver R{event.driver_branch} < R{driver_match.group(1)}"
            )
        if cuda_match and event.cuda_version is not None:
            actual_cuda = tuple(int(part) for part in event.cuda_version.split(".")[:2])
            required_cuda = tuple(int(part) for part in cuda_match.group(1).split("."))
            if actual_cuda < required_cuda:
                incompatible.append(
                    f"CUDA {event.cuda_version} < {cuda_match.group(1)}"
                )
        if incompatible:
            return _Resolution(
                source=ActionSource.NVIDIA_CATALOG,
                disposition=ActionDisposition.NOT_APPLICABLE,
                official_action=rule.immediate_action,
                investigatory_action=rule.investigatory_action,
                action=None,
                containment=self._xid_containment(event.xid),
                reasons=[
                    "Catalog reporting compatibility is not satisfied: "
                    + ", ".join(incompatible)
                ],
                requires_operator=True,
            )
        return None

    def _direct_resolution(
        self,
        official: str,
        rule: CatalogRule,
        *,
        source: ActionSource = ActionSource.NVIDIA_CATALOG,
        reason: str,
    ) -> _Resolution:
        action = DIRECT_ACTION_MAP[official]
        return _Resolution(
            source=source,
            disposition=(
                ActionDisposition.MONITOR_ONLY
                if action is RecoveryAction.NO_ACTION
                else ActionDisposition.EXECUTABLE
            ),
            official_action=official,
            investigatory_action=rule.investigatory_action,
            action=action,
            containment=self._xid_containment(rule.xid),
            reasons=[reason],
            pre_actions=(
                [
                    RecoveryAction.MARK_UNSCHEDULABLE,
                    RecoveryAction.STOP_WORKLOAD,
                ]
                if rule.xid == 95
                else []
            ),
        )

    def _dynamic_recovery_resolution(
        self,
        action: DynamicRecoveryAction,
        rule: CatalogRule,
        *,
        correlated_event_id: str | None = None,
    ) -> _Resolution:
        official = action.value
        if action is DynamicRecoveryAction.DRAIN_P2P:
            return _Resolution(
                source=ActionSource.NVIDIA_XID_154,
                disposition=ActionDisposition.EXECUTABLE,
                official_action=official,
                investigatory_action=rule.investigatory_action,
                action=RecoveryAction.STOP_WORKLOAD,
                containment=Containment.ALL_APPLICATIONS,
                reasons=[
                    "followed the driver-reported XID 154 Data Center "
                    "GPU Recovery Action DRAIN_P2P"
                ],
                pre_actions=[RecoveryAction.STOP_WORKLOAD],
                correlated_event_id=correlated_event_id,
            )
        mapped = (
            DynamicRecoveryAction.RESET_GPU
            if action is DynamicRecoveryAction.DRAIN_AND_RESET
            else action
        )
        resolution = self._direct_resolution(
            mapped.value,
            rule,
            source=ActionSource.NVIDIA_XID_154,
            reason=(
                "followed the driver-reported XID 154 Data Center "
                f"GPU Recovery Action {official}"
            ),
        )
        resolution.official_action = official
        resolution.correlated_event_id = correlated_event_id
        if action is DynamicRecoveryAction.DRAIN_AND_RESET:
            resolution.pre_actions = [
                RecoveryAction.MARK_UNSCHEDULABLE,
                RecoveryAction.STOP_WORKLOAD,
            ]
        return resolution

    def _workflow_xid_154(
        self,
        event: XidEvent,
        rule: CatalogRule,
        correlation: _XidCorrelation,
    ) -> _Resolution:
        window_closed = correlation.xid154_window_closed
        if event.xid_154_action is None:
            return _Resolution(
                source=ActionSource.NVIDIA_CATALOG,
                disposition=ActionDisposition.BLOCKED_MISSING_EVIDENCE,
                official_action=XID_154_ACTION,
                investigatory_action=rule.investigatory_action,
                action=None,
                containment=self._xid_containment(event.xid),
                reasons=[
                    "XID 154 message does not contain one of the official "
                    "recovery-action labels"
                ],
                requires_operator=True,
            )
        companion = self._find_xid154_companion(event, correlation.companion_events)
        if not window_closed:
            return _Resolution(
                source=ActionSource.NVIDIA_XID_154,
                disposition=ActionDisposition.PENDING_CORRELATION,
                official_action=event.xid_154_action.value,
                investigatory_action=rule.investigatory_action,
                action=None,
                containment=self._xid_containment(event.xid),
                reasons=["XID 154 companion correlation window is still open"],
                correlated_event_id=(
                    companion.event_id if companion is not None else None
                ),
            )
        return self._dynamic_recovery_resolution(
            event.xid_154_action,
            rule,
            correlated_event_id=(companion.event_id if companion is not None else None),
        )

    def _workflow_xid_45(
        self,
        event: XidEvent,
        rule: CatalogRule,
        correlation: _XidCorrelation,
    ) -> _Resolution:
        if not correlation.xid45_window_closed:
            return _Resolution(
                source=ActionSource.NVIDIA_CATALOG,
                disposition=ActionDisposition.PENDING_CORRELATION,
                official_action="WORKFLOW_XID_45",
                investigatory_action=rule.investigatory_action,
                action=None,
                containment=Containment.APPLICATION,
                reasons=["XID 45 correlation window is still open"],
            )
        companion_match = self._find_companion(event, correlation.companion_events)
        if companion_match is not None:
            companion, companion_resolution = companion_match
            companion_resolution.reasons.insert(
                0,
                f"XID 45 is not solo; followed the most severe "
                f"companion XID {companion.xid} guidance",
            )
            companion_resolution.correlated_event_id = companion.event_id
            return companion_resolution
        return _Resolution(
            source=ActionSource.NVIDIA_CATALOG,
            disposition=ActionDisposition.EXECUTABLE,
            official_action="RESTART_FM",
            investigatory_action=rule.investigatory_action,
            action=None,
            containment=Containment.APPLICATION,
            reasons=[
                "NVIDIA WORKFLOW_XID_45 classifies this as solo and "
                "requires Fabric Manager restart"
            ],
        )

    def _workflow_xid_48(
        self,
        event: XidEvent,
        rule: CatalogRule,
        correlation: _XidCorrelation,
    ) -> _Resolution:
        candidates = correlation.companion_events
        companions = self._find_related(event, {63, 64}, candidates=candidates)
        if companions:
            return _Resolution(
                source=ActionSource.NVIDIA_CATALOG,
                disposition=ActionDisposition.EXECUTABLE,
                official_action="DRAIN_AND_RESET",
                investigatory_action=rule.investigatory_action,
                action=RecoveryAction.RESET_GPU,
                containment=Containment.GPU,
                reasons=[
                    "NVIDIA WORKFLOW_XID_48 with XID 63/64 requires drain and GPU reset"
                ],
                pre_actions=[RecoveryAction.MARK_UNSCHEDULABLE],
                correlated_event_id=self._reset_equivalent_companion(
                    companions,
                    candidates=candidates,
                ),
            )
        return _Resolution(
            source=ActionSource.NVIDIA_CATALOG,
            disposition=ActionDisposition.EXECUTABLE,
            official_action="RESET_GPU",
            investigatory_action=rule.investigatory_action,
            action=RecoveryAction.RESET_GPU,
            containment=Containment.GPU,
            reasons=["NVIDIA WORKFLOW_XID_48 solo recovery action is RESET_GPU"],
        )

    def _reset_equivalent_companion(
        self,
        companions: list[XidEvent],
        *,
        candidates: list[XidEvent],
    ) -> str | None:
        """Pick a companion safe to share this GPU reset's incident.

        Merging XID 48 into a companion incident is how the duplicate
        reset of one GPU is avoided (XID 64 is itself RESET_GPU, so
        without this both events compile their own eight-step reset).
        But the merge in ``api.py:finalize_xid`` makes XID 48 adopt the
        companion's *existing* workflow, so merging into a companion
        that never created one silently drops the reset: XID 63 is
        IGNORE / no workflow, and merging into it left the double-bit
        error entirely unremediated.

        So only merge into a companion whose own resolution reaches the
        same containment strength. That is XID 64 (RESET_GPU) and never
        XID 63. Returning ``None`` keeps the independent incident,
        which is the fail-safe direction: a redundant reset is
        recoverable, a skipped one is not.
        """
        reset_rank = RECOVERY_ACTION_SEVERITY_RANK[RecoveryAction.RESET_GPU]
        eligible = [
            companion
            for companion in companions
            if RECOVERY_ACTION_SEVERITY_RANK.get(
                self._resolve_xid(companion, companion_events=candidates).action,
                0,
            )
            >= reset_rank
        ]
        if not eligible:
            return None
        return max(
            eligible,
            key=lambda item: (
                item.observed_at,
                item.event_id,
            ),
        ).event_id

    def _with_xid_48_backlink(
        self,
        event: XidEvent,
        resolution: _Resolution,
        *,
        candidates: list[XidEvent],
    ) -> _Resolution:
        """Make the XID 64 / XID 48 pairing dedupe in both orders.

        ``_workflow_xid_48`` only looks *forward*: it merges into a
        companion already in the store. So ``64 -> 48`` shares one
        reset, but ``48 -> 64`` produced two eight-step resets of the
        same GPU, because XID 64's own RESET_GPU never looked back at
        the XID 48 that preceded it.

        Correlating here instead of holding XID 48 open for the 30s
        companion window is deliberate. XID 48 is an uncorrectable
        double-bit error whose solo branch is already the strongest
        containment we have, so waiting would delay a cordon we apply
        regardless, and would still miss a companion arriving one
        second past the window.

        Only XID 64 backlinks, and only onto an XID 48: both resolve to
        RESET_GPU, so whichever incident wins carries an equivalent
        reset. XID 63 is excluded on both sides -- it is IGNORE and
        owns no workflow to adopt.
        """
        if event.xid != 64 or resolution.correlated_event_id:
            return resolution
        earlier = [
            companion
            for companion in self._find_related(event, {48}, candidates=candidates)
            if (companion.observed_at, companion.event_id)
            < (event.observed_at, event.event_id)
        ]
        if not earlier:
            return resolution
        companion = max(
            earlier,
            key=lambda item: (
                item.observed_at,
                item.event_id,
            ),
        )
        resolution.reasons.append(
            "shares the incident of the preceding XID 48 on this GPU "
            "so the uncorrectable error is reset once, not twice"
        )
        resolution.correlated_event_id = companion.event_id
        return resolution

    def _workflow_check_uvm(
        self,
        event: XidEvent,
        rule: CatalogRule,
        correlation: _XidCorrelation,
    ) -> _Resolution:
        if event.uvm_in_use is None:
            return _Resolution(
                source=ActionSource.NVIDIA_CATALOG,
                disposition=(ActionDisposition.BLOCKED_MISSING_EVIDENCE),
                official_action="CHECK_UVM",
                investigatory_action=rule.investigatory_action,
                action=None,
                containment=Containment.GPU,
                reasons=["CHECK_UVM requires explicit UVM/vGPU usage evidence"],
                requires_operator=True,
            )
        official = "RESET_GPU" if event.uvm_in_use else "IGNORE"
        return self._direct_resolution(
            official,
            rule,
            reason=(
                f"NVIDIA CHECK_UVM resolved from explicit uvm_in_use={event.uvm_in_use}"
            ),
        )

    def _workflow_nvlink5(
        self,
        event: XidEvent,
        rule: CatalogRule,
        correlation: _XidCorrelation,
    ) -> _Resolution:
        if (
            event.driver_branch is None
            or event.intr_info is None
            or event.error_status is None
        ):
            return _Resolution(
                source=ActionSource.NVIDIA_CATALOG,
                disposition=(ActionDisposition.BLOCKED_MISSING_EVIDENCE),
                official_action="WORKFLOW_NVLINK5_ERR",
                investigatory_action=rule.investigatory_action,
                action=None,
                containment=Containment.UNKNOWN,
                reasons=[
                    "NVLink5 decode requires driver branch, IntrInfo and Error Status"
                ],
                requires_operator=True,
            )

        is_v1 = event.driver_branch < self.policy.nvlink5.driver_boundary
        matched: list[NvlinkDecodeRule] = []
        action2: list[str] = []
        for decode in self.policy.nvlink5.decode_rules:
            if decode.xid != event.xid:
                continue
            pattern = decode.v1_pattern if is_v1 else decode.v2_pattern
            if not self._pattern_matches(event.intr_info, pattern):
                continue
            if not self._error_status_matches(event.error_status, decode.error_status):
                continue
            matched.append(decode)
            action2_pattern = (
                decode.action2_v1_pattern if is_v1 else decode.action2_v2_pattern
            )
            if (
                decode.action2
                and action2_pattern
                and self._pattern_matches(event.intr_info, action2_pattern)
            ):
                action2.append(decode.action2)

        if not matched:
            return _Resolution(
                source=ActionSource.NVIDIA_CATALOG,
                disposition=(ActionDisposition.BLOCKED_MISSING_EVIDENCE),
                official_action="WORKFLOW_NVLINK5_ERR",
                investigatory_action=rule.investigatory_action,
                action=None,
                containment=Containment.UNKNOWN,
                reasons=[
                    "IntrInfo/Error Status has no match in the pinned "
                    "official XID 144-150 decode table"
                ],
                decoded_subcode=self._subcode(event),
                requires_operator=True,
            )

        recovery_actions = {item.recovery_action for item in matched}.union(action2)
        xid154_action = event.xid_154_action
        if xid154_action is None:
            xid154 = self._find_xid154_event(event, correlation.companion_events)
            if xid154 is not None:
                xid154_action = xid154.xid_154_action
        if (
            "XID_154_EVAL" in recovery_actions
            and xid154_action is None
            and not correlation.xid154_window_closed
        ):
            return _Resolution(
                source=ActionSource.NVIDIA_CATALOG,
                disposition=ActionDisposition.PENDING_CORRELATION,
                official_action="XID_154",
                investigatory_action=rule.investigatory_action,
                action=None,
                containment=Containment.GPU,
                reasons=[
                    "NVLink5 decode requires waiting for a possible "
                    "companion XID 154 recovery action"
                ],
                decoded_subcode=self._subcode(event),
                matched_decode_rules=[item.subcode_name for item in matched],
            )
        if "RESET_GPU" in recovery_actions:
            official = "RESET_GPU"
        elif "XID_154_EVAL" in recovery_actions:
            official = xid154_action.value if xid154_action else "RESTART_APP"
        elif recovery_actions == {"IGNORE"}:
            official = "IGNORE"
        else:
            return _Resolution(
                source=ActionSource.NVIDIA_CATALOG,
                disposition=ActionDisposition.BLOCKED_WORKFLOW,
                official_action="WORKFLOW_NVLINK5_ERR",
                investigatory_action=rule.investigatory_action,
                action=None,
                containment=Containment.UNKNOWN,
                reasons=[
                    "official decode produced conflicting recovery "
                    f"actions: {sorted(recovery_actions)}"
                ],
                decoded_subcode=self._subcode(event),
                matched_decode_rules=[item.subcode_name for item in matched],
                requires_operator=True,
            )

        if "XID_154_EVAL" in recovery_actions and xid154_action:
            result = self._dynamic_recovery_resolution(xid154_action, rule)
        else:
            result = self._direct_resolution(
                official,
                rule,
                reason=(
                    "matched the pinned official XID 144-150 decode "
                    "table: " + ", ".join(item.subcode_name for item in matched)
                ),
            )
        result.decoded_subcode = self._subcode(event)
        result.matched_decode_rules = [item.subcode_name for item in matched]
        result.investigatory_action = (
            ", ".join(
                sorted(
                    {
                        item.investigatory_action
                        for item in matched
                        if item.investigatory_action
                    }
                )
            )
            or rule.investigatory_action
        )
        return result

    def _resolve_sxid(self, event: SxidEvent) -> _Resolution:
        family = self._product_families.family(event.product)
        for resolver in (
            self._resolve_sxid_product,
            self._resolve_sxid_integrity,
            self._resolve_sxid_official_action,
        ):
            result = resolver(event, family)
            if result is not None:
                return result
        return self._resolve_sxid_runtime(event)

    def _resolve_sxid_product(self, event, family):
        if family in {"B100", "GB200"}:
            return _Resolution(
                source=ActionSource.NVIDIA_FABRIC_MANAGER,
                disposition=ActionDisposition.NOT_APPLICABLE,
                official_action=None,
                investigatory_action="USE_DCGM_NVSDM_TELEMETRY",
                action=None,
                containment=Containment.FABRIC_PARTITION,
                reasons=[
                    f"Fabric Manager SXID reporting applies only to Hopper and earlier GPUs; {event.product} is Blackwell-family"
                ],
                requires_operator=True,
            )
        if event.product and family is None:
            return _Resolution(
                source=ActionSource.NVIDIA_FABRIC_MANAGER,
                disposition=ActionDisposition.BLOCKED_MISSING_EVIDENCE,
                official_action=None,
                investigatory_action="VERIFY_GPU_PRODUCT_GENERATION",
                action=None,
                containment=Containment.UNKNOWN,
                reasons=[
                    f"SXID applicability requires a recognized GPU generation; product {event.product} is unknown"
                ],
                requires_operator=True,
            )
        return None

    def _resolve_sxid_integrity(self, event, _family):
        if event.classification_source not in {
            "NVIDIA_FABRIC_MANAGER",
            "NVIDIA_FABRIC_MANAGER_CATALOG",
            "NVIDIA_FABRIC_MANAGER_RUNTIME",
        }:
            return _Resolution(
                source=ActionSource.NVIDIA_FABRIC_MANAGER,
                disposition=ActionDisposition.BLOCKED_MISSING_EVIDENCE,
                official_action=None,
                investigatory_action=None,
                action=None,
                containment=Containment.UNKNOWN,
                reasons=[
                    "SXID classification is not sourced from the pinned NVIDIA Fabric Manager table"
                ],
                requires_operator=True,
            )
        rule = self._sxid_rules.get(event.sxid)
        if rule is not None and event.classification is not rule.classification:
            return _Resolution(
                source=ActionSource.NVIDIA_FABRIC_MANAGER,
                disposition=ActionDisposition.BLOCKED_MISSING_EVIDENCE,
                official_action=rule.official_action,
                investigatory_action="VERIFY_SXID_LOG_INTEGRITY",
                action=None,
                containment=Containment.UNKNOWN,
                reasons=[
                    f"SXID {event.sxid} severity conflict: NVIDIA catalog={rule.classification.value}, observed={event.classification.value}; destructive recovery is blocked"
                ],
                requires_operator=True,
            )
        if rule is None and event.classification in {
            SxidClassification.ALWAYS_FATAL,
            SxidClassification.FATAL,
        }:
            # Fail closed on any destructive severity the pinned catalog does
            # not vouch for. Without this twin guard an unknown code claiming
            # FATAL fell through to the runtime trunk branch and was granted a
            # full RESET_ALL_GPUS_AND_NVSWITCHES on the strength of the
            # client-supplied classification alone (security review H-9).
            claimed = (
                "Always Fatal"
                if event.classification is SxidClassification.ALWAYS_FATAL
                else "Fatal"
            )
            return _Resolution(
                source=ActionSource.NVIDIA_FABRIC_MANAGER,
                disposition=ActionDisposition.BLOCKED_MISSING_EVIDENCE,
                official_action=None,
                investigatory_action="VERIFY_SXID_LOG_INTEGRITY",
                action=None,
                containment=Containment.UNKNOWN,
                reasons=[
                    f"SXID {event.sxid} claims {claimed} but is absent from pinned NVIDIA Table 23"
                ],
                requires_operator=True,
            )
        if event.classification is SxidClassification.ALWAYS_FATAL:
            return _Resolution(
                source=ActionSource.NVIDIA_FABRIC_MANAGER,
                disposition=ActionDisposition.EXECUTABLE,
                official_action="RESTART_BM",
                investigatory_action=rule.investigatory_action if rule else None,
                action=RecoveryAction.REBOOT_NODE,
                containment=Containment.NODE,
                reasons=[
                    f"NVIDIA Table 23 classifies SXID {event.sxid} as Always Fatal; HyperPod bare-metal recovery uses the official host-restart branch"
                ],
                pre_actions=[
                    RecoveryAction.MARK_UNSCHEDULABLE,
                    RecoveryAction.STOP_WORKLOAD,
                ],
            )
        return None

    def _resolve_sxid_official_action(self, event, _family):
        rule = self._sxid_rules.get(event.sxid)
        if rule is None:
            return None
        if rule.official_action == "RESET_ALL_GPUS_AND_NVSWITCHES":
            missing = self._sxid_missing_fabric_scope(event)
            if missing:
                return _Resolution(
                    source=ActionSource.NVIDIA_FABRIC_MANAGER,
                    disposition=ActionDisposition.BLOCKED_MISSING_EVIDENCE,
                    official_action=rule.official_action,
                    investigatory_action=rule.investigatory_action,
                    action=None,
                    containment=Containment.FABRIC_PARTITION,
                    reasons=[f"SXID {event.sxid} requires " + " and ".join(missing)],
                    pre_actions=[
                        RecoveryAction.MARK_UNSCHEDULABLE,
                        RecoveryAction.STOP_WORKLOAD,
                    ],
                    requires_operator=True,
                )
            return _Resolution(
                source=ActionSource.NVIDIA_FABRIC_MANAGER,
                disposition=ActionDisposition.EXECUTABLE,
                official_action=rule.official_action,
                investigatory_action=rule.investigatory_action,
                action=None,
                containment=Containment.FABRIC_PARTITION,
                reasons=[
                    f"NVIDIA SXID {event.sxid} rule requires a coordinated reset of all local GPUs and NVSwitches"
                ],
                pre_actions=[
                    RecoveryAction.MARK_UNSCHEDULABLE,
                    RecoveryAction.STOP_WORKLOAD,
                ],
            )
        if rule.official_action == "RESTART_VM":
            return _Resolution(
                source=ActionSource.NVIDIA_FABRIC_MANAGER,
                disposition=ActionDisposition.BLOCKED_WORKFLOW,
                official_action="RESTART_VM",
                investigatory_action=rule.investigatory_action,
                action=None,
                containment=Containment.FABRIC_PARTITION,
                reasons=[
                    f"NVIDIA SXID {event.sxid} restart-VM guidance is specific to Shared NVSwitch/vGPU deployments; HyperPod bare-metal has no equivalent guest VM lifecycle owner"
                ],
                requires_operator=True,
            )
        if rule.official_action == "CHECK_MECHANICALS":
            return _Resolution(
                source=ActionSource.NVIDIA_FABRIC_MANAGER,
                disposition=ActionDisposition.EXECUTABLE,
                official_action="CHECK_MECHANICALS",
                investigatory_action=rule.investigatory_action,
                action=RecoveryAction.ESCALATE_OPERATOR,
                containment=Containment.FABRIC_PARTITION,
                reasons=[
                    f"NVIDIA SXID {event.sxid} requires inspection of the NVLink mechanical connections"
                ],
                requires_operator=True,
            )
        return None

    def _resolve_sxid_runtime(self, event):
        rule = self._sxid_rules.get(event.sxid)
        if (
            event.classification is SxidClassification.FATAL
            and event.link_scope is not SxidLinkScope.UNKNOWN
            and event.link_scope_source
            not in {
                "TRUSTED_NVSWITCH_TOPOLOGY",
                "NVIDIA_PRODUCT_INVARIANT",
            }
        ):
            return _Resolution(
                source=ActionSource.NVIDIA_FABRIC_MANAGER,
                disposition=ActionDisposition.BLOCKED_MISSING_EVIDENCE,
                official_action="RESOLVE_TRUSTED_LINK_SCOPE",
                investigatory_action="QUERY_NVSWITCH_TOPOLOGY",
                action=None,
                containment=Containment.UNKNOWN,
                reasons=[
                    f"SXID {event.sxid} {event.link_scope.value} scope is not backed by trusted topology evidence"
                ],
                requires_operator=True,
            )
        if event.classification is SxidClassification.NON_FATAL:
            return _Resolution(
                source=ActionSource.NVIDIA_FABRIC_MANAGER,
                disposition=ActionDisposition.MONITOR_ONLY,
                official_action="IGNORE",
                investigatory_action=(
                    rule.investigatory_action
                    if rule and rule.investigatory_action
                    else "MONITOR_PROGRESS_AND_PERFORMANCE"
                ),
                action=RecoveryAction.NO_ACTION,
                containment=Containment.FABRIC_PARTITION,
                reasons=[
                    f"NVIDIA classifies SXID {event.sxid} as non-fatal; "
                    + (
                        f"catalog action={rule.official_action}"
                        if rule
                        else "the runtime severity drives the generic informational procedure"
                    )
                ],
            )
        if (
            event.classification is SxidClassification.FATAL
            and event.link_scope is SxidLinkScope.ACCESS
        ):
            if not event.participating_gpu_uuids:
                return _Resolution(
                    source=ActionSource.NVIDIA_FABRIC_MANAGER,
                    disposition=ActionDisposition.BLOCKED_MISSING_EVIDENCE,
                    official_action="RESET_PARTICIPATING_GPUS",
                    investigatory_action=None,
                    action=None,
                    containment=Containment.UNKNOWN,
                    reasons=[
                        "fatal access-link recovery requires the affected GPU and all participating workload GPUs"
                    ],
                    requires_operator=True,
                )
            return _Resolution(
                source=ActionSource.NVIDIA_FABRIC_MANAGER,
                disposition=ActionDisposition.EXECUTABLE,
                official_action="RESET_PARTICIPATING_GPUS",
                investigatory_action=None,
                action=RecoveryAction.RESET_GPU,
                containment=Containment.GPU,
                reasons=[
                    "Fabric Manager fatal access-link procedure resets the affected and all participating workload GPUs"
                ],
                pre_actions=[
                    RecoveryAction.MARK_UNSCHEDULABLE,
                    RecoveryAction.STOP_WORKLOAD,
                ],
            )
        if (
            event.classification is SxidClassification.FATAL
            and event.link_scope is SxidLinkScope.UNKNOWN
        ):
            return _Resolution(
                source=ActionSource.NVIDIA_FABRIC_MANAGER,
                disposition=ActionDisposition.BLOCKED_MISSING_EVIDENCE,
                official_action="RESET_ALL_GPUS_AND_NVSWITCHES",
                investigatory_action=None,
                action=None,
                containment=Containment.UNKNOWN,
                reasons=[
                    "fatal SXID requires a trusted access/trunk port-scope mapping"
                ],
                pre_actions=[
                    RecoveryAction.MARK_UNSCHEDULABLE,
                    RecoveryAction.STOP_WORKLOAD,
                ],
                requires_operator=True,
            )
        missing = self._sxid_missing_fabric_scope(event)
        if missing:
            return _Resolution(
                source=ActionSource.NVIDIA_FABRIC_MANAGER,
                disposition=ActionDisposition.BLOCKED_MISSING_EVIDENCE,
                official_action="RESET_ALL_GPUS_AND_NVSWITCHES",
                investigatory_action=None,
                action=None,
                containment=Containment.FABRIC_PARTITION,
                reasons=["fatal trunk SXID requires " + " and ".join(missing)],
                pre_actions=[
                    RecoveryAction.MARK_UNSCHEDULABLE,
                    RecoveryAction.STOP_WORKLOAD,
                ],
                requires_operator=True,
            )
        return _Resolution(
            source=ActionSource.NVIDIA_FABRIC_MANAGER,
            disposition=ActionDisposition.EXECUTABLE,
            official_action="RESET_ALL_GPUS_AND_NVSWITCHES",
            investigatory_action=None,
            action=None,
            containment=Containment.FABRIC_PARTITION,
            reasons=[
                "fatal trunk SXID requires stopping Fabric Manager and coordinated reset of all GPUs/NVSwitches"
            ],
            pre_actions=[
                RecoveryAction.MARK_UNSCHEDULABLE,
                RecoveryAction.STOP_WORKLOAD,
            ],
        )

    @staticmethod
    def _sxid_missing_fabric_scope(
        event: SxidEvent,
    ) -> list[str]:
        missing = []
        if not event.fabric_partition:
            missing.append("fabric_partition")
        if not event.participating_gpu_uuids:
            missing.append("complete node GPU inventory")
        return missing

    def _find_companion(
        self, event: XidEvent, candidates: list[XidEvent]
    ) -> tuple[XidEvent, _Resolution] | None:
        """Pick the most severe companion, not the most recent one.

        The catalog says XID 45 "will be seen in relation to" several
        errors at once, so ranking by arrival time made the verdict
        depend on log ordering: a late XID 95 could be shadowed by an
        earlier IGNORE-only XID 63 and downgrade a reset to NO_ACTION.
        """
        related = self._find_related(
            event,
            {item.xid for item in candidates if item.xid != 45},
            candidates=candidates,
        )
        if not related:
            return None
        resolved = [
            (
                item,
                self._resolve_xid(item, companion_events=candidates),
            )
            for item in related
        ]
        return max(
            resolved,
            key=lambda item: (
                *self._severity_rank(
                    item[0],
                    candidates,
                    resolution=item[1],
                ),
                item[0].observed_at,
                item[0].event_id,
            ),
        )

    def _severity_rank(
        self,
        candidate: XidEvent,
        candidates: list[XidEvent],
        *,
        resolution: _Resolution | None = None,
    ) -> tuple[int, int]:
        """Rank a companion by how much it forces us to contain."""
        resolution = resolution or self._resolve_xid(
            candidate, companion_events=candidates
        )
        action_rank = (
            OPERATOR_SEVERITY_RANK
            if resolution.action is None and resolution.requires_operator
            else RECOVERY_ACTION_SEVERITY_RANK.get(resolution.action, 0)
        )
        return (
            action_rank,
            CONTAINMENT_SEVERITY_RANK.get(resolution.containment, 0),
        )

    def _find_xid154_event(
        self, event: XidEvent, candidates: list[XidEvent]
    ) -> XidEvent | None:
        related = self._find_related(event, {154}, candidates=candidates)
        usable = [item for item in related if item.xid_154_action is not None]
        return max(
            usable,
            key=lambda item: item.observed_at,
            default=None,
        )

    def _find_xid154_companion(
        self, event: XidEvent, candidates: list[XidEvent]
    ) -> XidEvent | None:
        related = self._find_related(
            event,
            {item.xid for item in candidates if item.xid != 154},
            candidates=candidates,
        )
        return max(
            related,
            key=lambda item: item.observed_at,
            default=None,
        )

    def _find_related(
        self,
        event: XidEvent,
        xids: set[int],
        *,
        candidates: list[XidEvent],
    ) -> list[XidEvent]:
        window = timedelta(seconds=self.policy.companion_window_seconds)
        return [
            item
            for item in candidates
            if item.xid in xids
            and item.event_id != event.event_id
            and item.cluster_id == event.cluster_id
            and item.node_id == event.node_id
            and self._same_gpu(event, item)
            and (
                not event.source_boot_id
                or not item.source_boot_id
                or event.source_boot_id == item.source_boot_id
            )
            and abs(item.observed_at - event.observed_at) <= window
        ]

    @staticmethod
    def _same_gpu(left: XidEvent, right: XidEvent) -> bool:
        if left.gpu_uuid and right.gpu_uuid:
            return left.gpu_uuid == right.gpu_uuid
        if left.pci_bdf and right.pci_bdf:
            return left.pci_bdf.lower() == right.pci_bdf.lower()
        return False

    def _decision(
        self,
        *,
        event: XidEvent | SxidEvent,
        event_type: FaultEventType,
        resolution: _Resolution,
    ) -> FaultPolicyDecision:
        action = resolution.action
        safety_action = (
            RecoveryAction.QUARANTINE
            if action is None
            and resolution.disposition
            in {
                ActionDisposition.BLOCKED_WORKFLOW,
                ActionDisposition.BLOCKED_MISSING_EVIDENCE,
                ActionDisposition.NOT_APPLICABLE,
                ActionDisposition.SITE_SAFETY,
            }
            else None
        )
        marker_action = action or safety_action
        severity = (
            Severity.INFO
            if marker_action is RecoveryAction.NO_ACTION
            else Severity.WARNING
            if marker_action is RecoveryAction.RESTART_WORKLOAD
            else Severity.CRITICAL
            if marker_action is not None
            else Severity.WARNING
        )
        gpu_uuids = (
            [event.gpu_uuid]
            if isinstance(event, XidEvent) and event.gpu_uuid
            else event.participating_gpu_uuids
            if isinstance(event, SxidEvent)
            else []
        )
        fabric = [event.fabric_partition] if event.fabric_partition else []
        mapping_version = (
            self.sxid_policy.mapping_version
            if event_type is FaultEventType.SXID
            else self.policy.mapping_version
        )
        if isinstance(event, XidEvent):
            fault_class = {
                48: "GPU_MEMORY",
                63: "GPU_MEMORY",
                64: "GPU_MEMORY",
                74: "GPU_FABRIC",
                79: "GPU_INVENTORY",
                92: "GPU_MEMORY",
                94: "GPU_MEMORY",
                95: "GPU_MEMORY",
            }.get(event.xid, f"GPU_XID_{event.xid}")
            correlation_keys = {
                *([f"gpu:{event.gpu_uuid.lower()}"] if event.gpu_uuid else []),
                *([f"pci:{event.pci_bdf.strip().lower()}"] if event.pci_bdf else []),
                *(
                    [f"fabric:{event.fabric_partition.lower()}"]
                    if event.fabric_partition
                    else []
                ),
                *(
                    [f"link:{event.nvlink_link_id}"]
                    if event.nvlink_link_id is not None
                    else []
                ),
            }
        else:
            fault_class = "GPU_FABRIC"
            correlation_keys = {
                *(
                    f"gpu:{gpu_uuid.lower()}"
                    for gpu_uuid in event.participating_gpu_uuids
                ),
                *([f"pci:{event.pci_bdf.strip().lower()}"] if event.pci_bdf else []),
                *(
                    [f"fabric:{event.fabric_partition.lower()}"]
                    if event.fabric_partition
                    else []
                ),
                *([f"switch:{event.switch_id.lower()}"] if event.switch_id else []),
                *([f"port:{event.port.lower()}"] if event.port else []),
            }
        marker = NodeMarker(
            marker_id=f"marker-{event.event_id}",
            cluster_id=event.cluster_id,
            source=f"gpu-fault-policy/{resolution.source.value.lower()}",
            trusted=True,
            incident_id=f"inc-{event.event_id}",
            observed_at=event.observed_at,
            expires_at=event.observed_at
            + timedelta(seconds=self.policy.marker_ttl_seconds),
            scope=MarkerScope(
                node_ids=[event.node_id],
                gpu_uuids=gpu_uuids,
                pci_bdfs=[event.pci_bdf] if event.pci_bdf else [],
                fabric_partitions=fabric,
            ),
            source_event_time=event.source_event_time,
            source_monotonic_us=event.source_monotonic_us,
            source_boot_id=event.source_boot_id,
            collected_at=event.collected_at,
            ingested_at=event.ingested_at,
            event_source=event.event_source,
            severity=severity,
            recommended_action=marker_action,
            action_owner="gpu-fault-policy",
            mapping_version=mapping_version,
            policy_source=resolution.source.value,
            official_action=resolution.official_action,
            site_safety_action=(safety_action.value if safety_action else None),
            investigatory_action=(resolution.investigatory_action),
            action_disposition=resolution.disposition.value,
            fault_class=fault_class,
            correlation_keys=sorted(correlation_keys),
            raw_reason=(
                f"{event_type.value} "
                f"{event.xid if isinstance(event, XidEvent) else event.sxid}"
            ),
            raw_evidence_ref=event.evidence_ref,
            active=(
                marker_action is not None
                and marker_action is not RecoveryAction.NO_ACTION
            ),
        )
        return FaultPolicyDecision(
            event_id=event.event_id,
            event_type=event_type,
            policy_version=mapping_version,
            source=resolution.source,
            disposition=resolution.disposition,
            official_action=resolution.official_action,
            investigatory_action=resolution.investigatory_action,
            action=action,
            safety_action=safety_action,
            severity=severity,
            containment=resolution.containment,
            reasons=resolution.reasons,
            pre_actions=resolution.pre_actions,
            decoded_subcode=resolution.decoded_subcode,
            matched_decode_rules=resolution.matched_decode_rules,
            nvlink_link_id=(
                event.nvlink_link_id
                if isinstance(event, XidEvent) and event.xid == 74
                else None
            ),
            nvlink_occurrence_counts=(
                event.nvlink_occurrence_counts
                if isinstance(event, XidEvent) and event.xid == 74
                else {}
            ),
            requires_operator=resolution.requires_operator,
            correlated_event_id=resolution.correlated_event_id,
            marker=marker,
        )

    def _subcode(self, event: XidEvent) -> int | None:
        if event.intr_info is None or event.driver_branch is None:
            return None
        if event.driver_branch < self.policy.nvlink5.driver_boundary:
            return (event.intr_info >> 5) & 0x1F
        return event.intr_info & 0x7F

    @staticmethod
    def _pattern_matches(value: int, pattern: str) -> bool:
        if len(pattern) != 32 or re.fullmatch(r"[01-]{32}", pattern) is None:
            raise ValueError(
                "NVLink decode pattern must contain exactly 32 binary/wildcard bits"
            )
        for index, expected in enumerate(reversed(pattern)):
            if expected == "-":
                continue
            if ((value >> index) & 1) != int(expected):
                return False
        return True

    @staticmethod
    def _error_status_matches(value: int, expected: str | None) -> bool:
        if expected is None:
            return True
        try:
            values = {int(item.strip(), 16) for item in expected.split("/")}
        except ValueError as exc:
            raise ValueError(
                "NVLink errorStatus contains a non-hexadecimal value"
            ) from exc
        return value in values

    @staticmethod
    def _xid_containment(xid: int) -> Containment:
        if xid == 94:
            return Containment.APPLICATION
        if xid == 95:
            return Containment.ALL_APPLICATIONS
        return Containment.GPU


_XidWorkflowResolver = Callable[
    [GpuFaultPolicyEngine, XidEvent, CatalogRule, _XidCorrelation],
    _Resolution,
]

# Every catalog Immediate Action that is a workflow rather than a direct action
# (those are in DIRECT_ACTION_MAP) is routed from here, so adding an NVIDIA
# workflow is a table entry plus a resolver instead of another branch in the
# middle of _resolve_xid's guard sequence. A token absent from the table is
# deliberately BLOCKED_WORKFLOW: preserving the NVIDIA action verbatim without
# executing it is the fail-closed outcome, so a resolver nobody registered can
# never silently execute.
_XID_WORKFLOW_RESOLVERS: dict[str, _XidWorkflowResolver] = {
    XID_154_ACTION: GpuFaultPolicyEngine._workflow_xid_154,
    "WORKFLOW_XID_45": GpuFaultPolicyEngine._workflow_xid_45,
    "WORKFLOW_XID_48": GpuFaultPolicyEngine._workflow_xid_48,
    "WORKFLOW_NVLINK_ERR": GpuFaultPolicyEngine._workflow_nvlink74,
    "WORKFLOW_NVLINK5_ERR": GpuFaultPolicyEngine._workflow_nvlink5,
    "CHECK_UVM": GpuFaultPolicyEngine._workflow_check_uvm,
    "CONTACT_SUPPORT": GpuFaultPolicyEngine._workflow_operator_review,
    "CHECK_MECHANICALS": GpuFaultPolicyEngine._workflow_operator_review,
    "UPDATE_SWFW": GpuFaultPolicyEngine._workflow_operator_review,
}
