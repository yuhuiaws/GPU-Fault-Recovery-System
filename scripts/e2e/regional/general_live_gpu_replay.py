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

ROOT = Path(__file__).resolve().parents[3]
FIELD_TOOL = ROOT / "tools" / "gpu_dcgm_field_replay.py"
INVENTORY_WRAPPER = ROOT / "tools" / "nvidia_smi_inventory_replay.py"
PORT = 9412
#: Normal samples the product must accept through the pass-through proxy
#: before a replay is armed (spec step 3: at least three normal samples).
PASS_THROUGH_READS = 3
#: The replay's deadline beyond a variant's own budget: the tool goes back to
#: pass-through by itself, and the deadman restores the unit later still.
TOOL_MARGIN_SECONDS = 300
DEADMAN_MARGIN_SECONDS = 600


# --------------------------------------------------------------------------- #
# Survey: what the node really reports
# --------------------------------------------------------------------------- #
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

    def target(self, name: str, *, avoid: Sequence[str] = ()) -> str:
        """The highest-index GPU that reports ``name`` (not in ``avoid``)."""

        candidates = [gpu for gpu in self.gpus_with(name) if gpu not in avoid]
        if not candidates:
            raise VariantNotRun(
                "CAPABILITY_ABSENT", f"no GPU on the node reports {name}"
            )
        return candidates[-1]


def require_replayable(survey: GpuSurvey) -> None:
    """The node can run a replay at all: one collector, a real exporter."""

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
    residue = survey.raw.get("residue") or {}
    if residue.get("metrics_dropins") or residue.get("units"):
        raise VariantNotRun(
            "FIXTURE_NOT_READY",
            "a runtime drop-in or replay unit already changes the metrics collector",
        )


def run_gpu_survey(ctx: VariantContext, *, limits: bool = False) -> GpuSurvey:
    arguments = ["survey", *(["--limits"] if limits else [])]
    value = ctx.dcgm(*arguments, timeout=300)
    ctx.evidence.write("gpu-survey", value)
    survey = GpuSurvey(value)
    require_replayable(survey)
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

    def require_inventory(self) -> None:
        """The oracle gate for an inventory replay: the wrapper really rewrote."""

        aborted = [
            str(record.get("unchanged_reason"))
            for record in self.inventory_records
            if record.get("returncode") not in (0, None) and not record.get("rewritten")
        ]
        if not self.inventory_rewritten():
            detail = (
                ("; ".join(aborted))
                if aborted
                else "the wrapper never rewrote a real inventory query"
            )
            raise VariantInconclusive(
                f"the inventory replay was not effective: {detail}"
            )

    def served_any(self, key: str) -> bool:
        """Whether the replay ever served a non-null value for ``<gpu>/<field>``."""

        self.audit()
        return any(
            (record.get("served") or {}).get(key) is not None
            for record in self.records
            if record.get("armed")
        )

    def aborted(self) -> str | None:
        self.audit()
        for record in self.records:
            if record.get("aborted"):
                return str(record["aborted"])
        return None

    def require_effective(self, *keys: str) -> None:
        """The oracle gate: the replay really served ``keys`` and never aborted."""

        aborted = self.aborted()
        if aborted:
            raise VariantInconclusive(f"the replay aborted: {aborted}")
        missing = [key for key in keys if not self.served_any(key)]
        if missing:
            raise VariantInconclusive("the replay never served " + ", ".join(missing))

    def close(self) -> dict[str, Any]:
        """Disarm, restore the collector endpoint, stop the tool -- in order."""

        receipts: dict[str, Any] = {}
        if self.armed:
            try:
                receipts["disarm"] = self.ctx.dcgm(
                    "disarm", "--port", str(PORT), timeout=60
                )
                self.armed = False
            except Exception as exc:  # noqa: BLE001 - recorded; restore still runs
                receipts["disarm_error"] = f"{type(exc).__name__}: {exc}"
        if self.repointed:
            receipts["restore"] = self.ctx.dcgm("restore-collector", timeout=180)
            self.repointed = False
        receipts["stop"] = self.ctx.dcgm("stop-proxy", "--port", str(PORT), timeout=60)
        self.audit()
        if self.inventory_wrapper:
            self.inventory_audit()
        self.ctx.evidence.write("dcgm-replay-close", receipts)
        restore = receipts.get("restore") or {}
        stop = receipts.get("stop") or {}
        if restore and restore.get("restored_endpoint") not in {
            None,
            restore.get("original_endpoint"),
        }:
            raise VariantInconclusive("the collector endpoint was not restored")
        if stop and not stop.get("run_dir_removed", True):
            raise VariantInconclusive("the replay run directory was not removed")
        return receipts


def arm_and_confirm(replay: DcgmReplay, *served_keys: str) -> None:
    """Deploy, let a few normal samples through, arm, and gate on the oracle."""

    replay.deploy()
    replay.arm()
    if served_keys:
        replay.require_effective(*served_keys)


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
    "DcgmReplay",
    "GPU_KINDS",
    "activity_now",
    "wait_gpu_incident",
    "gpu_spec",
    "no_hardware",
    "result",
    "GpuSurvey",
    "PASS_THROUGH_READS",
    "PORT",
    "add",
    "arm_and_confirm",
    "at_limit_increment",
    "baseline",
    "passthrough_plan",
    "clear",
    "label",
    "over_limit_increment",
    "plan",
    "put",
    "recover_counter",
    "replay_boot_id",
    "replay_uuid",
    "require_replayable",
    "run_gpu_survey",
    "state_now",
    "step",
    "utc_now",
    "wait_accepted",
]
