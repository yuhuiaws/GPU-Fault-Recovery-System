"""Runner side of the GF-GENERAL-LIVE DCGM field and GPU inventory replay.

The DCGM categories (019 ECC/remap/retirement, 020 NVLink counters, 021 fabric
state, 022 PCIe replay, 024 composites, 025 field structure, 026 GPU
identity) change one raw reading the node's metrics collector reads, and
nothing else (spec "证明层次与夹具就绪", ``live-replay``):

* the DCGM exporter fields go through ``tools/gpu_dcgm_field_replay.py`` (its
  own per-category profiles; the environment tool's allow-list is not
  widened);
* the inventory reading goes through ``tools/nvidia_smi_inventory_replay.py``,
  a pass-through wrapper first on the one unit's PATH;
* the boot-generation and expected-count inputs go through the same unit's
  private environment.

``probes/general_live_dcgm_probe.py`` installs those reviewed files under
``/run/gfrs-live/<digest>/`` with sha256 checks, serves the tool loopback-only
and pass-through until armed, repoints ``gpu-fault-metrics-collector.service``
through one ``/run`` drop-in and arms a deadman before it does. This module
drives that probe from the runner, keeps the replay's audit journal as the
oracle (``oracle.jsonl``), builds the reviewed plans from the node's own
survey, and owns the clean end of an injection: a counter finding is only
recovered by a fresh zero-delta sample, so the replay holds the elevated value
until the product has accepted one before it disarms.

Two node-state rules bracket every replay. Before anything starts, the survey
must show no replay object on the node: an earlier run's drop-in, proxy unit,
run directory or deadman is named in a ``NOT_RUN(FIXTURE_NOT_READY)`` and
nothing is installed; only this fixture's own leftovers whose deadman has
already elapsed (or never armed) are reclaimed first, through the probe's
``cleanup-leftover``, and never a drop-in or directory that is not this
fixture's. At the end, ``DcgmReplay.close`` runs the probe's ``close-run``
whatever state the run reached -- a repoint that raised after it wrote its
drop-in is undone like a finished run -- and names every object the probe
still finds in the error that fails the variant.

Nothing here decides a verdict and nothing posts a finding or workflow; the
category modules judge the product's own records (``general_live_gpu_store``)
with ``general_live_gpu_verdicts``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.e2e.regional.general_live_fixture import (
    VariantContext,
    VariantInconclusive,
    VariantNotRun,
    utc_now,
)
from scripts.e2e.regional.general_live_common import activity as node_activity
from scripts.e2e.regional.general_live_gpu_store import gpu_state
from scripts.e2e.regional.general_live_gpu_verdicts import node_hardware_errors
from scripts.e2e.regional.general_live_registry import LIVE_REPLAY, VariantSpec
from scripts.e2e.regional.general_live_verdicts import verdict
from scripts.e2e.regional.probes.general_live_dcgm_probe import PROFILE_NAME
from tools.gpu_dcgm_field_replay import PROFILES as FIELD_PROFILES

ROOT = Path(__file__).resolve().parents[3]
FIELD_TOOL = ROOT / "tools" / "gpu_dcgm_field_replay.py"
INVENTORY_WRAPPER = ROOT / "tools" / "nvidia_smi_inventory_replay.py"
PORT = 9412
#: The one unit the probe repoints (named in refusals; the probe owns the rule).
METRICS_UNIT = "gpu-fault-metrics-collector.service"
STATE_ROOT = "/run/gfrs-live"


class ReplayCleanupError(RuntimeError):
    """The replay's cleanup left named objects on the node (or could not prove
    it clean); the message lists each one."""


#: Normal samples the product must accept through the pass-through proxy
#: before a replay is armed (spec step 3: at least three normal samples).
PASS_THROUGH_READS = 3
#: The replay's deadline beyond a variant's own budget: the tool goes back to
#: pass-through by itself, and the deadman restores the unit later still.
TOOL_MARGIN_SECONDS = 300
DEADMAN_MARGIN_SECONDS = 600
#: The collector's scrape interval when the node did not report its own: the
#: product's default (``collectors_cli._dcgm_arguments``).
INTERVAL_KEY = "GPU_FAULT_METRICS_INTERVAL_SECONDS"
DEFAULT_INTERVAL_SECONDS = 15
#: The oracle gate after ``arm``: the plan moves only when the collector
#: scrapes, so the first replayed reading takes the plan's pass-through reads
#: plus one more scrape. The gate allows that many scrapes, two slack ones (a
#: skipped or late scrape) and a fixed grace for the restarted collector's
#: phase and the probe round trips, polling the journal in between.
SERVE_SLACK_READS = 2
SERVE_GRACE_SECONDS = 90
SERVE_POLL_SECONDS = 10
#: The NVLink error-counter families the product grades alike
#: (``gpu_fault.dcgm_fields.DCGM_FABRIC_CANONICAL_GROUPS``; every member is a
#: ``nvlink_*_error_total`` counter to ``gpu_metrics._link_decision`` and the
#: NVLink composites), each member in the order a plan prefers it: the legacy
#: per-class counter (``DCGM_FI_DEV_NVLINK_*_ERROR_COUNT_TOTAL``, NVLink3 and
#: earlier parts) before the DL aggregate (``DCGM_FI_DEV_NVLINK_ERROR_DL_*``,
#: the only NVLink error counters a Hopper/NVLink4 exporter publishes: live
#: 2026-10-26, p5en H200, DCGM 4.6/4.7, every 020 variant NOT_RUN "no GPU on
#: the node reports nvlink_crc_flit_error_total").
NVLINK_FAMILY_MEMBERS: dict[str, tuple[str, ...]] = {
    "nvlink_crc": (
        "nvlink_crc_flit_error_total",
        "nvlink_crc_aggregate_error_total",
        "nvlink_crc_data_error_total",
    ),
    "nvlink_replay": (
        "nvlink_replay_error_total",
        "nvlink_replay_aggregate_error_total",
    ),
    "nvlink_recovery": (
        "nvlink_recovery_error_total",
        "nvlink_recovery_aggregate_error_total",
    ),
}


# --------------------------------------------------------------------------- #
# Survey: what the node really reports
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NvlinkCounter:
    """The member of one NVLink error family this node reports.

    ``gpus`` are the exporter GPUs reporting it in index order and
    ``exporter_metrics`` the sample names it arrives under (for example
    ``DCGM_FI_DEV_NVLINK_ERROR_DL_CRC_total``), so a variant's evidence says
    which counter it exercised.
    """

    family: str
    canonical: str
    gpus: tuple[str, ...]
    exporter_metrics: tuple[str, ...]

    @property
    def generation(self) -> str:
        if "_aggregate_" in self.canonical:
            return "DL aggregate, NVLink4 and later"
        return "legacy per-class counter, NVLink3 and earlier"

    def note(self) -> str:
        metrics = ", ".join(self.exporter_metrics) or "exporter name not surveyed"
        return (
            f"NVLink {self.family.removeprefix('nvlink_')} counter exercised: "
            f"{self.canonical} ({self.generation}; exporter {metrics})"
        )

    def evidence(self) -> dict[str, Any]:
        return {
            "nvlink_family": self.family,
            "canonical_name": self.canonical,
            "exporter_metrics": list(self.exporter_metrics),
        }


@dataclass(frozen=True)
class ComponentPair:
    """Two components of one per-GPU composite, both reported by ``gpu``.

    ``primary`` anchors the plan (every exporter generation reports it);
    ``companion`` is the member of a candidate list the exporter really
    publishes for the same GPU, so a composite plan never targets a reading
    that is absent upstream -- the replay tool refuses to invent samples and
    would abort the armed run INCONCLUSIVE instead (live 2026-10-26, p5en
    H200: 024 ``fresh_components`` on ``retired_pages_sbe_total``, a counter
    Hopper does not publish). ``exporter_metrics`` are the sample names the
    companion arrives under, so a variant's evidence says which reading it
    exercised.
    """

    gpu: str
    primary: str
    companion: str
    exporter_metrics: tuple[str, ...]

    @property
    def components(self) -> tuple[str, str]:
        return (self.primary, self.companion)

    def keys(self) -> list[str]:
        """The oracle keys (``<gpu>/<field>``) of both components."""

        return [f"{self.gpu}/{name}" for name in self.components]

    def note(self) -> str:
        metrics = ", ".join(self.exporter_metrics) or "exporter name not surveyed"
        return (
            f"second component exercised: {self.companion} beside {self.primary} "
            f"on {self.gpu} (exporter {metrics})"
        )

    def evidence(self) -> dict[str, Any]:
        return {
            "gpu_uuid": self.gpu,
            "component_metrics": list(self.components),
            "second_component": self.companion,
            "exporter_metrics": list(self.exporter_metrics),
        }


@dataclass(frozen=True)
class GpuSurvey:
    """The probe's read-only survey, with the questions plans ask of it."""

    raw: dict[str, Any]

    @property
    def exporter(self) -> dict[str, Any]:
        return dict(self.raw.get("exporter") or {})

    @property
    def gpus(self) -> list[str]:
        """Exporter GPUs ordered by their index (UUIDs)."""

        entries = self.exporter.get("gpus") or {}

        def key(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
            raw = str(item[1].get("gpu_index") or "")
            return (int(raw) if raw.isdigit() else 999, item[0])

        return [gpu for gpu, _entry in sorted(entries.items(), key=key)]

    def entry(self, gpu: str) -> dict[str, Any]:
        return dict((self.exporter.get("gpus") or {}).get(gpu) or {})

    def value(self, gpu: str, name: str) -> float | None:
        value = (self.entry(gpu).get("values") or {}).get(name)
        return float(value) if isinstance(value, (int, float)) else None

    def has(self, gpu: str, name: str) -> bool:
        return self.value(gpu, name) is not None

    def model(self, gpu: str) -> str:
        return str(self.entry(gpu).get("model") or "")

    def pci_bdf(self, gpu: str) -> str:
        """The exporter's PCI BDF label of ``gpu`` (``00000000:1B:00.0`` form)."""

        return str(self.entry(gpu).get("pci_bdf") or "")

    def temperature_limits(self, gpu: str) -> dict[str, float]:
        """``nvidia-smi -q -x`` temperature limits of ``gpu`` (``survey --limits``)."""

        for item in self.raw.get("temperature_limits") or []:
            if item.get("gpu_uuid") == gpu:
                return {
                    str(name): float(value)
                    for name, value in (item.get("limits") or {}).items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
        return {}

    def setting(self, key: str, default: int) -> int:
        value = (self.raw.get("collector_env") or {}).get(key)
        try:
            return int(value) if value not in (None, "") else default
        except (TypeError, ValueError):
            return default

    @property
    def interval(self) -> int:
        return self.setting("GPU_FAULT_METRICS_INTERVAL_SECONDS", 15)

    @property
    def health_summary(self) -> int:
        return self.setting("GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS", 300)

    @property
    def confirmation(self) -> int:
        return self.setting("GPU_FAULT_DCGM_EDGE_CONFIRMATION_SAMPLES", 3)

    @property
    def sustain(self) -> int:
        return self.setting("GPU_FAULT_DCGM_COLLECTION_ERROR_SUSTAIN_SECONDS", 180)

    @property
    def inventory_interval(self) -> int:
        return self.setting("GPU_FAULT_GPU_INVENTORY_INTERVAL_SECONDS", 60)

    @property
    def boot_id(self) -> str:
        return str(self.raw.get("boot_id") or "")

    @property
    def inventory(self) -> list[dict[str, str]]:
        return [dict(item) for item in self.raw.get("gpu_inventory") or []]

    def gpus_with(self, name: str) -> list[str]:
        return [gpu for gpu in self.gpus if self.has(gpu, name)]

    def exporter_metrics(self, gpu: str, name: str) -> list[str]:
        """The exporter sample names ``gpu`` reports canonical ``name`` under."""

        names = (self.entry(gpu).get("metric_names") or {}).get(name) or []
        return [str(item) for item in names]

    def nvlink_counter(self, family: str) -> NvlinkCounter:
        """The member of NVLink error ``family`` this node reports.

        The product grades every member of a family alike, so a plan targets
        the one the exporter publishes: in ``NVLINK_FAMILY_MEMBERS`` order,
        the member the most GPUs report (a Hopper exporter has only the DL
        aggregate; an older part only the legacy per-class counter). No member
        on any GPU is ``NOT_RUN(CAPABILITY_ABSENT)`` naming the whole family.
        """

        members = NVLINK_FAMILY_MEMBERS[family]
        reporting = {member: self.gpus_with(member) for member in members}
        canonical = max(members, key=lambda member: len(reporting[member]))
        gpus = reporting[canonical]
        if not gpus:
            raise VariantNotRun(
                "CAPABILITY_ABSENT",
                f"no GPU on the node reports an NVLink {family.removeprefix('nvlink_')} "
                f"counter ({', '.join(members)})",
            )
        metrics = sorted(
            {name for gpu in gpus for name in self.exporter_metrics(gpu, canonical)}
        )
        return NvlinkCounter(
            family=family,
            canonical=canonical,
            gpus=tuple(gpus),
            exporter_metrics=tuple(metrics),
        )

    def target(self, name: str, *, avoid: Sequence[str] = ()) -> str:
        """The highest-index GPU that reports ``name`` (not in ``avoid``)."""

        candidates = [gpu for gpu in self.gpus_with(name) if gpu not in avoid]
        if not candidates:
            raise VariantNotRun(
                "CAPABILITY_ABSENT", f"no GPU on the node reports {name}"
            )
        return candidates[-1]

    def component_pair(
        self, primary: str, candidates: Sequence[str], *, avoid: Sequence[str] = ()
    ) -> ComponentPair:
        """``primary`` and the first of ``candidates`` one GPU reports beside it.

        A composite plan alters two readings of one GPU, and the exporter
        decides which second reading exists: Hopper retires no pages (row
        remapping replaced it), so it publishes no ``retired_pages_*`` while
        it reports the row-remap counters; a pre-Ampere part is the other way
        round. The candidates are tried in order and the highest-index GPU
        (not in ``avoid``) reporting both the primary and that candidate is
        the target. No candidate beside the primary on any GPU is
        ``NOT_RUN(CAPABILITY_ABSENT)`` naming every candidate -- before
        anything is installed on the node.
        """

        anchors = [gpu for gpu in self.gpus_with(primary) if gpu not in avoid]
        if not anchors:
            raise VariantNotRun(
                "CAPABILITY_ABSENT", f"no GPU on the node reports {primary}"
            )
        for companion in candidates:
            gpus = [gpu for gpu in anchors if self.has(gpu, companion)]
            if gpus:
                gpu = gpus[-1]
                return ComponentPair(
                    gpu=gpu,
                    primary=primary,
                    companion=companion,
                    exporter_metrics=tuple(self.exporter_metrics(gpu, companion)),
                )
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            f"no GPU on the node reports {primary} together with a second "
            f"component ({', '.join(candidates)})",
        )


def require_replayable(
    survey: GpuSurvey, *, reclaim: dict[str, dict[str, Any]] | None = None
) -> None:
    """The node can run a replay at all: one collector, a real exporter, and
    nothing of an earlier replay still on it (every leftover is named)."""

    if not survey.raw.get("venv_python"):
        raise VariantNotRun("CAPABILITY_ABSENT", "the node has no gpu_fault venv")
    tools = survey.raw.get("tools") or {}
    missing = [name for name in ("systemd-run", "systemctl") if not tools.get(name)]
    if missing:
        raise VariantNotRun("CAPABILITY_ABSENT", "the node lacks " + ", ".join(missing))
    unit = survey.raw.get("metrics_unit") or {}
    if unit.get("ActiveState") != "active":
        raise VariantNotRun(
            "FIXTURE_NOT_READY", "gpu-fault-metrics-collector.service is not active"
        )
    environment = survey.raw.get("collector_env") or {}
    mode = str(environment.get("GPU_FAULT_METRICS_MODE") or "dcgm")
    if mode != "dcgm":
        raise VariantNotRun(
            "CAPABILITY_ABSENT", f"the metrics collector runs in {mode} mode, not dcgm"
        )
    exporter = survey.exporter
    if exporter.get("error") or not exporter.get("gpus"):
        raise VariantNotRun(
            "FIXTURE_NOT_READY",
            "the real DCGM exporter scrape is not usable: "
            + str(exporter.get("error") or "no GPU"),
        )
    leftovers = residue_lines(survey.raw.get("residue") or {})
    for run_digest, receipt in sorted((reclaim or {}).items()):
        for error in receipt.get("errors") or []:
            leftovers.append(f"reclaim of run {run_digest} failed: {error}")
    if leftovers:
        raise VariantNotRun(
            "FIXTURE_NOT_READY",
            "replay objects are still on the node; nothing was started: "
            + "; ".join(leftovers),
        )


def residue_lines(residue: dict[str, Any]) -> list[str]:
    """Every replay object the survey found on the node, named precisely.

    A probe that groups its findings per run digest (``runs``) names each
    run's objects with its run id and whether its deadman restore is still
    pending; the flat lists of an older probe are named as they are. Foreign
    drop-ins and directories -- not this fixture's -- are named as left in
    place, and so is a tool that still answers on the replay port.
    """

    lines: list[str] = []
    runs = residue.get("runs")
    if isinstance(runs, dict):
        for run_digest in sorted(runs):
            state = runs[run_digest] or {}
            objects = [str(item) for item in state.get("objects") or []]
            if not objects:
                continue
            label = f"earlier run {state.get('run_id') or run_digest}"
            if state.get("deadman_pending"):
                label += " (deadman restore pending, left to fire)"
            lines.append(f"{label}: " + ", ".join(objects))
    else:
        lines.extend(f"drop-in {name}" for name in residue.get("metrics_dropins") or [])
        lines.extend(f"unit {name}" for name in residue.get("units") or [])
        lines.extend(
            f"run dir {STATE_ROOT}/{name}" for name in residue.get("run_dirs") or []
        )
    lines.extend(
        f"foreign drop-in {name} on {METRICS_UNIT} (not this fixture's; left in place)"
        for name in residue.get("foreign_dropins") or []
    )
    lines.extend(
        f"foreign directory {STATE_ROOT}/{name} (not this fixture's; left in place)"
        for name in residue.get("foreign_run_dirs") or []
    )
    if residue.get("port_answers"):
        lines.append(
            f"a replay tool still answers on 127.0.0.1:{residue.get('port') or PORT}"
        )
    return lines


def reclaim_leftovers(
    ctx: VariantContext, survey: GpuSurvey, arguments: Sequence[str]
) -> tuple[GpuSurvey, dict[str, dict[str, Any]]]:
    """Restore this fixture's own expired leftovers, then survey again.

    Only runs the probe marks ``reclaimable`` -- its own objects whose deadman
    is no longer pending -- are handed to ``cleanup-leftover``; a pending
    deadman, a foreign drop-in or directory is never touched here and stays
    for ``require_replayable`` to name.
    """

    runs = (survey.raw.get("residue") or {}).get("runs") or {}
    expired = sorted(
        run_digest
        for run_digest, state in runs.items()
        if isinstance(state, dict) and state.get("reclaimable")
    )
    if not expired:
        return survey, {}
    receipts = {
        run_digest: ctx.dcgm("cleanup-leftover", "--digest", run_digest, timeout=300)
        for run_digest in expired
    }
    ctx.evidence.write("dcgm-leftover-reclaim", receipts)
    return GpuSurvey(ctx.dcgm(*arguments, timeout=300)), receipts


def run_gpu_survey(ctx: VariantContext, *, limits: bool = False) -> GpuSurvey:
    arguments = ["survey", "--port", str(PORT), *(["--limits"] if limits else [])]
    survey = GpuSurvey(ctx.dcgm(*arguments, timeout=300))
    survey, reclaimed = reclaim_leftovers(ctx, survey, arguments)
    ctx.evidence.write("gpu-survey", survey.raw)
    require_replayable(survey, reclaim=reclaimed)
    return survey


def replay_uuid(run_id: str, salt: str = "") -> str:
    """A deterministic, clearly synthetic replay GPU UUID for one run."""

    digest = hashlib.sha256(f"{run_id}/{salt}".encode()).hexdigest()
    return f"GPU-{uuid.UUID(digest[:32])}"


def replay_boot_id(survey: GpuSurvey, run_id: str) -> tuple[str, str]:
    """A boot id that is not the current one, and where it came from.

    A real earlier boot of this node (its journal) when there is one; else a
    deterministic replay id -- the product compares boot ids for equality
    only, so either exercises the same branch.
    """

    for item in survey.raw.get("prior_boot_ids") or []:
        if item and item != survey.boot_id:
            return str(item), "journal --list-boots"
    digest = hashlib.sha256(f"boot/{run_id}".encode()).hexdigest()
    return str(uuid.UUID(digest[:32])), "replay id (no earlier boot in the journal)"


# --------------------------------------------------------------------------- #
# Plan builders
# --------------------------------------------------------------------------- #
def step(reads: int, *operations: dict[str, Any], hold: bool = False) -> dict[str, Any]:
    value: dict[str, Any] = {"reads": int(reads), "ops": list(operations)}
    if hold:
        value["until_advance"] = True
    return value


def add(name: str, gpus: Iterable[str], value: int = 1) -> dict[str, Any]:
    return {"op": "add", "field": name, "gpus": list(gpus), "value": int(value)}


def put(name: str, gpus: Iterable[str], value: int) -> dict[str, Any]:
    return {"op": "set", "field": name, "gpus": list(gpus), "value": int(value)}


def clear(name: str, gpus: Iterable[str]) -> dict[str, Any]:
    return {"op": "clear", "field": name, "gpus": list(gpus)}


def label(name: str, gpus: Iterable[str], key: str, value: str) -> dict[str, Any]:
    return {
        "op": "label",
        "field": name,
        "gpus": list(gpus),
        "label": key,
        "value": value,
    }


def plan(profile: str, *steps: dict[str, Any]) -> dict[str, Any]:
    """A field plan on one of the tool's profiles.

    The name is checked here, before the survey's node is touched, against
    both gates it has to pass on the node: the tool's ``PROFILES`` and the
    host controller's ``PROFILE_NAME`` shape. Live 2026-10-26 the controller
    accepted only ``[a-z]{3,16}`` while 022's XID-join plans use ``pcie_xid``,
    and the three variants ended INCONCLUSIVE at ``deploy-proxy`` with the
    refusal's message withheld.
    """

    if profile not in FIELD_PROFILES or PROFILE_NAME.fullmatch(profile) is None:
        raise VariantInconclusive(
            f"replay plan profile {profile!r} is not a field-tool profile the "
            "host controller serves (harness defect; nothing was deployed)"
        )
    return {"profile": profile, "steps": list(steps)}


def passthrough_plan() -> dict[str, Any]:
    """A field plan that changes nothing: the DCGM tool stays pass-through.

    Used by the inventory-only (026) variants, where only the nvidia-smi
    wrapper rewrites and the DCGM field tool is never armed.
    """

    return plan("completeness", step(1))


def baseline() -> dict[str, Any]:
    """Armed no-op reads first: the stale ring and the abort baseline fill."""

    return step(PASS_THROUGH_READS)


def leading_passthrough_reads(plan: dict[str, Any]) -> int:
    """The armed reads a plan spends before its first operation (``baseline``).

    Those reads serve nothing by design; the oracle gate waits past them.
    """

    count = 0
    for item in plan.get("steps") or []:
        if item.get("ops"):
            break
        count += int(item.get("reads") or 0)
    return count


def collector_interval(repoint: dict[str, Any], *, default: int) -> int:
    """The repointed collector's scrape interval, from its real environment.

    The probe's ``repoint-collector`` receipt carries the allow-listed keys of
    the restarted process's environment; the product reads this one as a
    float. A receipt without it (an older probe) or a malformed value keeps
    ``default``.
    """

    raw = (repoint.get("process_environment") or {}).get(INTERVAL_KEY)
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def at_limit_increment(interval: int, rate_per_minute: float = 8.0) -> int:
    """The largest per-read PCIe increment that never exceeds ``rate``.

    The product rates ``delta * 60 / elapsed`` between accepted batches whose
    ``observed_at`` are at least one collector interval apart, so
    ``floor(rate * interval / 60)`` per read stays at or below the rate.
    """

    return max(1, int(rate_per_minute * interval // 60))


def over_limit_increment(interval: int, rate_per_minute: float = 8.0) -> int:
    """A per-read PCIe increment whose rate clears the warning threshold.

    Doubles the per-interval budget and adds one, so even one skipped scrape
    (two intervals) still rates above the threshold.
    """

    return int(rate_per_minute * interval // 60) * 2 + 1


def under_limit_increment(interval: int, rate_per_minute: float = 8.0) -> int:
    """A per-read PCIe increment whose rate stays clearly under ``rate``.

    Half the at-limit budget, at least one: a raise that must never be the
    finding -- ``counter_reset`` grades the drop that follows it -- cannot sit
    on the 8/min edge, where two scrapes a little closer than one interval
    already rate above the threshold. One per read on a 15-s collector (4/min).
    """

    return max(1, at_limit_increment(interval, rate_per_minute) // 2)


def cross_boot_jump(
    interval: int, health_summary: int, rate_per_minute: float = 8.0
) -> int:
    """A one-read PCIe jump that rates over ``rate`` against any reading of
    the previous boot generation.

    The repointed collector's first scrape under the replayed boot id is the
    product's first reading of the new generation; the store's previous
    sample is the real boot's, delivered by the old collector at most one
    health summary, one scrape and the restart earlier. The jump clears the
    threshold over twice that window (plus a minute), so a product that rated
    across the boundary would have to mint and one that starts a new baseline
    must not.
    """

    window = 2 * (health_summary + interval) + 60
    return int(rate_per_minute * window // 60) + 1


# --------------------------------------------------------------------------- #
# The replay controller
# --------------------------------------------------------------------------- #
def _source(name: str, path: Path) -> tuple[str, str, str]:
    content = path.read_bytes()
    return (
        name,
        hashlib.sha256(content).hexdigest(),
        base64.b64encode(content).decode("ascii"),
    )


@dataclass
class DcgmReplay:
    """One reviewed field/inventory replay on the node, cleanup registered.

    The controller installs the reviewed sources, serves the tool, repoints
    the one collector unit (the deadman is armed first, inside the probe),
    and arms the plan. Its audit journal is the runner's oracle. ``close``
    disarms, restores the endpoint and stops the tool in that order; the
    runner registers it on the cleanup stack before the first mutation.
    """

    ctx: VariantContext
    plan: dict[str, Any]
    seconds: int
    gpu_uuid: str = ""
    inventory_plan: dict[str, Any] | None = None
    boot_id: str = ""
    unknown_expected_count: bool = False
    #: The repointed collector's scrape interval; ``deploy`` reads the real one.
    interval: int = DEFAULT_INTERVAL_SECONDS
    audit_offset: int = 0
    inventory_offset: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)
    inventory_records: list[dict[str, Any]] = field(default_factory=list)
    receipts: dict[str, Any] = field(default_factory=dict)
    repointed: bool = False
    armed: bool = False

    @property
    def inventory_wrapper(self) -> bool:
        return self.inventory_plan is not None

    def _install(self, name: str, path: Path) -> None:
        source_name, sha256, payload = _source(name, path)
        self.receipts[f"install-{name}"] = self.ctx.dcgm(
            "install-source",
            "--name",
            source_name,
            "--sha256",
            sha256,
            "--b64",
            payload,
            timeout=120,
        )

    def _install_json(self, name: str, document: dict[str, Any]) -> None:
        path = self.ctx.case_dir / name
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
        )
        self._install(name, path)

    def deploy(self) -> None:
        """Install sources, serve the tool, repoint the collector, arm later."""

        self.serve()
        self.repoint()

    def serve(self) -> None:
        """Install the reviewed sources and serve the tool pass-through.

        The cleanup is registered before the first file lands on the node.
        Nothing scrapes the tool until ``repoint`` moves the collector to it,
        so a plan armed between the two (``arm_before_repoint``) is first read
        by the restarted collector's own first scrape.
        """

        self.ctx.cleanup.push("dcgm-replay", self.close)
        self._install("gpu_dcgm_field_replay.py", FIELD_TOOL)
        self._install_json("plan.json", self.plan)
        if self.inventory_plan is not None:
            self._install("nvidia_smi_inventory_replay.py", INVENTORY_WRAPPER)
            self._install_json("inventory-plan.json", self.inventory_plan)
        self.receipts["deploy"] = self.ctx.dcgm(
            "deploy-proxy",
            "--tool",
            "gpu_dcgm_field_replay.py",
            "--profile",
            str(self.plan["profile"]),
            "--port",
            str(PORT),
            "--max-seconds",
            str(self.seconds + TOOL_MARGIN_SECONDS),
            timeout=120,
        )
        self.ctx.evidence.write("dcgm-replay-deploy", self.receipts)

    def repoint(self) -> None:
        """Repoint the one collector unit at the served tool (deadman first).

        The probe restarts the unit; its first scrape after the restart is the
        tool's first read, under the replayed boot id when one is given.
        """

        repoint = [
            "repoint-collector",
            "--port",
            str(PORT),
            "--restore-seconds",
            str(self.seconds + DEADMAN_MARGIN_SECONDS),
        ]
        if self.inventory_wrapper:
            repoint.append("--inventory-wrapper")
        if self.boot_id:
            repoint += ["--boot-id", self.boot_id]
        if self.unknown_expected_count:
            repoint.append("--unknown-expected-count")
        self.receipts["repoint"] = self.ctx.dcgm(*repoint, timeout=180)
        self.repointed = True
        self.interval = collector_interval(
            self.receipts["repoint"], default=self.interval
        )
        self.ctx.evidence.write("dcgm-replay-deploy", self.receipts)

    def arm(self) -> dict[str, Any]:
        receipt = self.ctx.dcgm("arm", "--port", str(PORT), timeout=60)
        self.armed = True
        self.receipts["arm"] = receipt
        self.ctx.evidence.write("dcgm-replay-arm", receipt)
        return receipt

    def advance(self) -> dict[str, Any]:
        """Leave a held plan step (``until_advance``) for the next one."""

        return self.ctx.dcgm("advance", "--port", str(PORT), timeout=60)

    def status(self) -> dict[str, Any]:
        return self.ctx.dcgm("status", "--port", str(PORT), timeout=60)

    def collector_restart(self) -> dict[str, Any]:
        """One restart of the metrics collector (the monitor-restart variant)."""

        receipt = self.ctx.dcgm("collector-restart", timeout=180)
        self.ctx.evidence.write("collector-restart", receipt)
        return receipt

    def audit(self) -> list[dict[str, Any]]:
        """New replay journal records since the last read (the oracle)."""

        return self._read_journal("replay.jsonl")

    def inventory_audit(self) -> list[dict[str, Any]]:
        return self._read_journal("inventory.jsonl")

    def _read_journal(self, name: str) -> list[dict[str, Any]]:
        attribute = "audit_offset" if name == "replay.jsonl" else "inventory_offset"
        store = self.records if name == "replay.jsonl" else self.inventory_records
        source = name.split(".", 1)[0]
        fresh: list[dict[str, Any]] = []
        while True:
            value = self.ctx.dcgm(
                "audit-read",
                "--journal",
                name,
                "--offset",
                str(getattr(self, attribute)),
                timeout=120,
            )
            records = list(value.get("records") or [])
            fresh.extend(records)
            setattr(
                self,
                attribute,
                int(value.get("next_offset") or getattr(self, attribute)),
            )
            if not records or getattr(self, attribute) >= int(value.get("total") or 0):
                break
        if fresh:
            store.extend(fresh)
            self.ctx.evidence.append_oracle(fresh, source=source)
        return fresh

    def inventory_rewritten(self) -> bool:
        """Whether the inventory wrapper has rewritten at least one real query."""

        self.inventory_audit()
        return any(record.get("rewritten") for record in self.inventory_records)

    def inventory_refusals(self) -> list[str]:
        """Why the wrapper left real inventory queries unchanged (final reasons).

        Every journal record is one real inventory query of the collector. One
        the wrapper could not apply -- an unreadable or expired plan, a GPU
        index not in the real inventory, a replay UUID that names a real GPU --
        is passed through unchanged for the run's whole life, so the first such
        record ends the gate. A failed real query (``returncode`` not 0) is no
        refusal: the product retries the inventory on its next tick.
        """

        return [
            "; ".join(str(item) for item in record.get("unchanged_reason") or [])
            or "unchanged"
            for record in self.inventory_records
            if not record.get("rewritten") and record.get("returncode") == 0
        ]

    def wait_inventory_rewritten(self, timeout_seconds: int) -> dict[str, Any]:
        """The oracle gate for an inventory replay, waited for: poll the
        wrapper's journal until it has rewritten one real inventory query.

        The collector queries the inventory on its own cadence -- first on its
        first tick after the restart (start-up spread plus start-up jitter),
        then every inventory interval -- so a journal judged once, right after
        the repoint, holds nothing (live 2026-10-26, plain EKS, 026
        ``bdf_changed`` and ``count_unchanged``: judged 14-15 s after the
        collector restart; the wrapper rewrote the first real query 20 s after
        it, 5 s after the verdict). A query the wrapper refused to rewrite is
        final and raised at once with the wrapper's reason; a deadline names
        what the journal held, so "no query reached the wrapper" reads
        differently from "every query failed".
        """

        def rewritten() -> dict[str, Any] | None:
            if self.inventory_rewritten():
                first = next(
                    record
                    for record in self.inventory_records
                    if record.get("rewritten")
                )
                return {
                    "rewritten_at": first.get("at"),
                    "inventory_queries": len(self.inventory_records),
                }
            refusals = self.inventory_refusals()
            if refusals:
                raise VariantInconclusive(
                    "the inventory replay was not effective: the wrapper left a real "
                    f"inventory query unchanged: {refusals[0]}"
                )
            return None

        value = self.ctx.wait_until(
            rewritten,
            timeout_seconds=timeout_seconds,
            poll_seconds=SERVE_POLL_SECONDS,
            name="inventory-rewritten",
        )
        if value is not None:
            return value
        failed = sum(
            1 for record in self.inventory_records if record.get("returncode") != 0
        )
        seen = (
            "no inventory query reached the wrapper"
            if not self.inventory_records
            else f"{len(self.inventory_records)} inventory quer"
            f"{'ies' if len(self.inventory_records) != 1 else 'y'} journaled, "
            f"{failed} failed"
        )
        raise VariantInconclusive(
            "the inventory replay was not effective: the wrapper never rewrote a real "
            f"inventory query within {timeout_seconds} s of the repoint: {seen}"
        )

    def _served(self, key: str) -> bool:
        return any(
            (record.get("served") or {}).get(key) is not None
            for record in self.records
            if record.get("armed")
        )

    def _aborted(self) -> str | None:
        for record in self.records:
            if record.get("aborted"):
                return str(record["aborted"])
        return None

    def armed_reads(self) -> list[dict[str, Any]]:
        """The journal's armed collector reads so far (events are not reads)."""

        return [
            record
            for record in self.records
            if record.get("armed") and record.get("read") is not None
        ]

    def served_any(self, key: str) -> bool:
        """Whether the replay ever served a non-null value for ``<gpu>/<field>``."""

        self.audit()
        return self._served(key)

    def aborted(self) -> str | None:
        self.audit()
        return self._aborted()

    def _gate(self, keys: Sequence[str]) -> list[str]:
        """Read the journal once; raise on an abort, return the unserved keys."""

        self.audit()
        aborted = self._aborted()
        if aborted:
            raise VariantInconclusive(f"the replay aborted: {aborted}")
        return [key for key in keys if not self._served(key)]

    def require_effective(self, *keys: str) -> None:
        """The oracle gate, judged now: the replay served ``keys``, no abort."""

        missing = self._gate(keys)
        if missing:
            raise VariantInconclusive("the replay never served " + ", ".join(missing))

    def serve_timeout_seconds(self) -> int:
        """How long the first replayed reading of this plan may take."""

        reads = leading_passthrough_reads(self.plan) + 1 + SERVE_SLACK_READS
        return reads * self.interval + SERVE_GRACE_SECONDS

    def wait_served(self, *keys: str) -> None:
        """The oracle gate, waited for: poll the journal until the replay has
        served every key once, the replay aborts, or the deadline passes.

        The plan only moves when the collector scrapes, and its leading
        pass-through reads serve nothing by design -- a journal judged right
        after ``arm`` holds nothing else (live 2026-10-25: two baseline reads
        of a 15-s collector, every 019 variant INCONCLUSIVE "never served").
        An abort is final and raised at once; a deadline names the reads the
        journal did hold, so a collector that never scraped the replay reads
        differently from one that had not reached the step yet.
        """

        timeout = self.serve_timeout_seconds()

        def served() -> dict[str, Any] | None:
            missing = self._gate(keys)
            if missing:
                return None
            return {"served": list(keys), "armed_reads": len(self.armed_reads())}

        value = self.ctx.wait_until(
            served,
            timeout_seconds=timeout,
            poll_seconds=SERVE_POLL_SECONDS,
            name="replay-served",
        )
        if value is not None:
            return
        reads = self.armed_reads()
        missing = [key for key in keys if not self._served(key)]
        seen = f"{len(reads)} armed read{'s' if len(reads) != 1 else ''}"
        if reads:
            seen += f", the last at plan step {reads[-1].get('step')}"
        raise VariantInconclusive(
            f"the replay never served {', '.join(missing)} within {timeout} s: {seen}"
        )

    def close(self) -> dict[str, Any]:
        """Disarm, restore the collector endpoint, stop the tool, prove it clean.

        One idempotent probe call (``close-run``) performs every step whatever
        the run reached -- the runner's own ``armed`` / ``repointed`` flags do
        not decide what is undone, because a probe call that raised midway
        (live 2026-10-25: ``repoint-collector`` after it had written the
        drop-in and restarted the collector) leaves the node changed with the
        flag unset. The journals are read first (the run directory goes with
        the stop), and every object the probe still finds is named in the
        ``ReplayCleanupError`` that fails the variant.
        """

        receipts: dict[str, Any] = {}
        journals: list[tuple[str, Callable[[], Any]]] = [("audit", self.audit)]
        if self.inventory_wrapper:
            journals.append(("inventory_audit", self.inventory_audit))
        for name, read in journals:
            try:
                read()
            except Exception as exc:  # noqa: BLE001 - recorded; the close still runs
                receipts[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
        closed = self.ctx.dcgm("close-run", "--port", str(PORT), timeout=300)
        self.armed = False
        self.repointed = False
        receipts["close"] = closed
        for key in ("disarm", "restore", "stop"):
            if key in closed:
                receipts[key] = closed[key]
        self.ctx.evidence.write("dcgm-replay-close", receipts)
        leftover = [str(item) for item in closed.get("leftover") or []]
        errors = [str(item) for item in closed.get("errors") or []]
        if leftover or errors or closed.get("clean") is not True:
            raise ReplayCleanupError(
                cleanup_message(self.ctx.settings.node, leftover, errors)
            )
        return receipts


def cleanup_message(node: str, leftover: Sequence[str], errors: Sequence[str]) -> str:
    """Name what the replay cleanup left on ``node`` and which steps failed."""

    parts = []
    if leftover:
        parts.append("left on the node: " + "; ".join(leftover))
    if errors:
        parts.append("failed steps: " + "; ".join(errors))
    if not parts:
        parts.append("the probe did not prove the run clean")
    return f"replay cleanup on {node} -- " + " -- ".join(parts)


def arm_and_confirm(replay: DcgmReplay, *served_keys: str) -> None:
    """Deploy, arm, and gate on the oracle once the collector reached the step.

    The gate waits (``DcgmReplay.wait_served``): the collector scrapes on its
    own interval and the plan's pass-through baseline reads serve nothing, so
    the journal is judged when it can hold a replayed reading, not right
    after ``arm``. Without keys there is no gate (the variant reads the
    journal itself).
    """

    replay.deploy()
    replay.arm()
    if served_keys:
        replay.wait_served(*served_keys)


def arm_before_repoint(replay: DcgmReplay, *served_keys: str) -> None:
    """Serve and arm the plan first, repoint the collector after, then gate.

    ``arm_and_confirm`` repoints before it arms, so a collector restarted under
    a replayed boot id scrapes its first reading pass-through and the boot
    boundary the product crosses is (v, real boot) -> (v, replayed boot): a
    zero delta no grading could get wrong (live 2026-10-26, 022
    ``source_boot_change``: the over-limit ramp that followed it was a
    same-generation storm the product rightly minted). Here the proxy is armed
    while the collector still scrapes the real exporter -- no read reaches the
    plan -- and the repoint's restart makes the plan's first read the first
    reading the product sees under the new boot, so the boundary itself
    carries the plan's first step.
    """

    replay.serve()
    replay.arm()
    replay.repoint()
    if served_keys:
        replay.wait_served(*served_keys)


def wait_accepted(
    ctx: VariantContext,
    since: Any,
    accept: Callable[[dict[str, Any]], bool],
    *,
    names: Sequence[str] = (),
    timeout_seconds: float,
    name: str,
) -> dict[str, Any]:
    """Poll the product's GPU records until ``accept`` holds (or the deadline)."""

    last: dict[str, Any] = {}

    def probe() -> dict[str, Any] | None:
        nonlocal last
        last = gpu_state(
            ctx.regional,
            ctx.settings.regional.cluster_id,
            ctx.settings.node,
            since,
            names=names,
        )
        return last if accept(last) else None

    value = ctx.wait_until(
        probe, timeout_seconds=timeout_seconds, poll_seconds=20, name=name
    )
    return value if value is not None else last


#: The evidence kinds a GPU category reads incidents and metrics from.
GPU_KINDS = ("GPU_METRICS",)


def activity_now(
    ctx: VariantContext, since: Any, *, metrics: Sequence[str] = ()
) -> dict[str, Any]:
    """The product's incidents/workflows/notifications for the node since then."""

    return node_activity(
        ctx, since, metrics=metrics or ("collector_observation_valid",), kinds=GPU_KINDS
    )


def wait_gpu_incident(
    ctx: VariantContext,
    since: Any,
    select: Callable[[dict[str, Any]], Sequence[dict[str, Any]]],
    *,
    metrics: Sequence[str] = (),
    timeout_seconds: float,
    name: str,
) -> dict[str, Any]:
    """Poll the product until ``select`` finds an incident (or the deadline)."""

    last: dict[str, Any] = {}

    def probe() -> dict[str, Any] | None:
        nonlocal last
        last = activity_now(ctx, since, metrics=metrics)
        return last if select(last) else None

    value = ctx.wait_until(
        probe, timeout_seconds=timeout_seconds, poll_seconds=20, name=name
    )
    return value if value is not None else last


def recover_counter(
    ctx: VariantContext, replay: DcgmReplay, since: Any, *, names: Sequence[str]
) -> dict[str, Any]:
    """Disarm the replay and wait until the product books a clean batch.

    A counter finding is cleared only by a later accepted sample whose delta
    is below threshold (``gpu_metrics.ingest``); reverting the counter is an
    invalid delta that keeps the finding. So the replay disarms (the real
    unchanged value returns), and the runner waits for the product to have
    accepted a fresh batch with no active finding of ``names``.
    """

    replay.close()
    ctx.cleanup.discard("dcgm-replay")

    def recovered(state: dict[str, Any]) -> bool:
        active = {
            item.get("canonical_name") for item in state.get("active_findings") or []
        }
        return not (active & set(names)) and bool(state.get("batches"))

    return wait_accepted(
        ctx,
        since,
        recovered,
        names=names,
        timeout_seconds=max(600, replay.seconds),
        name="recovery",
    )


def state_now(
    ctx: VariantContext, since: Any, *, names: Sequence[str] = ()
) -> dict[str, Any]:
    return gpu_state(
        ctx.regional,
        ctx.settings.regional.cluster_id,
        ctx.settings.node,
        since,
        names=names,
    )


def gpu_spec(
    case: str,
    variant: str,
    injection: str,
    expected: str,
    execute,
    *,
    risk: str = "L",
    flags: str = "S",
    lanes: tuple[str, ...] = ("eks", "hp"),
    approvals: tuple[str, ...] = (),
    max_minutes: int = 25,
    expected_outcome: str = "PASS",
    product_gap: str = "",
) -> VariantSpec:
    """One live-replay DCGM/inventory variant of a lane-R category."""

    return VariantSpec(
        case_id=case,
        variant=variant,
        proof_mode=LIVE_REPLAY,
        lanes=lanes,
        risk=risk,
        flags=flags,
        injection=injection,
        expected=expected,
        cleanup="disarm, restore the collector endpoint, remove every /run fixture",
        max_minutes=max_minutes,
        approvals=approvals,
        execute=execute,
        expected_outcome=expected_outcome,
        product_gap=product_gap,
    )


def result(
    stages: dict[str, list[str]],
    *,
    notes: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    inconclusive: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble a variant result from its per-stage error lists.

    ``proof_scope`` is always ``live-replay``: this lane only proves the
    processing chain, never the physical fault.
    """

    return {
        "verdict": verdict(stages, inconclusive=inconclusive),
        "proof_scope": "live-replay",
        "physical_fault_injected": False,
        "stages": stages,
        "errors": [
            f"{stage}: {error}" for stage, items in stages.items() for error in items
        ],
        "notes": notes or [],
        **(extra or {}),
    }


def no_hardware(state: dict[str, Any], *, allowed: Iterable[str] = ()) -> list[str]:
    """No workflow on the node planned an unapproved hardware operation."""

    return node_hardware_errors({"workflows": _all_workflows(state)}, allowed=allowed)


def _all_workflows(state: dict[str, Any]) -> list[dict[str, Any]]:
    return list(state.get("workflows") or [])


__all__ = [
    "DEFAULT_INTERVAL_SECONDS",
    "DcgmReplay",
    "GPU_KINDS",
    "ReplayCleanupError",
    "activity_now",
    "wait_gpu_incident",
    "gpu_spec",
    "no_hardware",
    "result",
    "GpuSurvey",
    "NVLINK_FAMILY_MEMBERS",
    "NvlinkCounter",
    "PASS_THROUGH_READS",
    "PORT",
    "add",
    "arm_and_confirm",
    "arm_before_repoint",
    "at_limit_increment",
    "baseline",
    "cleanup_message",
    "collector_interval",
    "cross_boot_jump",
    "passthrough_plan",
    "clear",
    "label",
    "leading_passthrough_reads",
    "over_limit_increment",
    "under_limit_increment",
    "plan",
    "put",
    "reclaim_leftovers",
    "recover_counter",
    "replay_boot_id",
    "replay_uuid",
    "require_replayable",
    "residue_lines",
    "run_gpu_survey",
    "state_now",
    "step",
    "utc_now",
    "wait_accepted",
]
