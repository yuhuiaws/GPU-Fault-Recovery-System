"""GF-GENERAL-LIVE-025 DCGM collection incomplete: the field variants.

Phase 1 implemented the native scrape-timeout variants and left the five field
variants FIXTURE_NOT_READY. They drive the *structure* of one scrape through
the ``completeness`` profile (the reviewed tool never touches a safety reading
and never writes a value here) and read the product's parser / normaliser /
completeness check:

* ``missing_field`` -- a required field missing from *every* GPU: the
  whole-batch completeness check reports it (PASS);
* ``one_gpu_missing`` -- the same field missing from *one* GPU only: the
  product's completeness check is computed over the union of all GPUs'
  fields, so the field is still present on the others and the per-GPU
  omission is not reported. Expected FAIL (product gap);
* ``conflicting_alias`` -- two exporter aliases of one field with different
  values: the product drops the whole slot of that batch and reports it as
  ambiguous (``gpu_observation.normalize_gpu_observation``), so neither the
  injected value nor the published reading is accepted while the conflict is
  served and the slot's latest row stays the pre-conflict reading; graded from
  the first conflicting batch the product recorded, never over the plan's
  pass-through reads (``_alias_overwrite``). The pair is the name the node
  publishes the field under plus the product's other name for it
  (``alias_pair``), chosen from the survey before anything is installed; a
  node without such a pair is NOT_RUN(CAPABILITY_ABSENT);
* ``nan_negative`` -- a negative value is rejected and a NaN is dropped: neither
  slot's latest accepted sample moves past the first replayed scrape (the
  earlier real reading is retained; it never becomes a healthy zero);
* ``duplicate_slot`` -- a duplicated ``(device, field)`` slot is rejected while
  other GPUs' independent samples stay usable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from gpu_fault.collectors.gpu.dcgm import DCGM_METRICS
from scripts.e2e.regional.general_live_fixture import VariantContext, VariantNotRun
from scripts.e2e.regional.general_live_gpu_replay import (
    DcgmReplay,
    GpuSurvey,
    arm_and_confirm,
    baseline,
    gpu_spec,
    plan,
    recover_counter,
    result,
    run_gpu_survey,
    state_now,
    step,
    utc_now,
)
from scripts.e2e.regional.general_live_gpu_verdicts import (
    accepted_samples,
    active_findings,
    collection_errors,
    latest_value,
)
from tools.gpu_dcgm_field_replay import PROFILES as FIELD_PROFILES

CASE = "GF-GENERAL-LIVE-025"
REQUIRED = "pcie_replay_total"
#: The reviewed gauge the ``nan_negative`` NaN is served on: a required field
#: of the product's per-GPU completeness check, so the dropped slot is reported.
NAN_FIELD = "row_remap_failure"
#: The value served under the injected alias: a clock event bitmask the real
#: reading never carries at the same time (``0x40`` HW slowdown); halved when
#: the real reading happens to be it, so the two aliases always disagree.
ALIAS_VALUE = 64
#: The product's scrape-wide completeness message naming the omitted field
#: (``collectors/gpu/completeness.py``: ``DCGM exporter is missing required
#: fields: <fields>``); the GPU-scoped shape reads ``... fields on <gpu>: ...``.
BATCH_WIDE_OMISSION = re.compile(
    rf"missing required fields: .*\b{re.escape(REQUIRED)}\b"
)


@dataclass(frozen=True)
class AliasPair:
    """One canonical field, the exporter name the node publishes it under and
    the product's other name for it -- what the ``alias`` op serves."""

    gpu: str
    field: str
    published: str
    alias: str

    def describe(self) -> dict[str, str]:
        return {
            "field": self.field,
            "published": self.published,
            "injected": self.alias,
        }


def alias_families() -> dict[str, tuple[str, ...]]:
    """Canonical fields the product reads under two or more exporter names
    (``collectors/gpu/dcgm.DCGM_METRICS``) that the completeness profile may
    serve (``tools/gpu_dcgm_field_replay.PROFILES``).

    On the shipped product that is ``clock_throttle_reasons`` alone: DCGM 4
    renamed ``DCGM_FI_DEV_CLOCK_THROTTLE_REASONS`` to
    ``DCGM_FI_DEV_CLOCKS_EVENT_REASONS`` and an exporter publishes one of the
    two; the T.Limit pair is a thermal-safety reading the profile excludes.
    """

    names: dict[str, set[str]] = {}
    for exporter_name, (canonical, _unit) in DCGM_METRICS.items():
        names.setdefault(canonical, set()).add(exporter_name)
    allowed = FIELD_PROFILES["completeness"].fields
    return {
        canonical: tuple(sorted(members))
        for canonical, members in sorted(names.items())
        if len(members) >= 2 and canonical in allowed
    }


def alias_pair(survey: GpuSurvey) -> AliasPair:
    """The pair ``conflicting_alias`` serves, decided from the survey before
    anything is installed on the node.

    Live 2026-10-26 (eksb, H200, dcgm-exporter 4.6.1) the exporter published
    ``clock_throttle_reasons`` as ``DCGM_FI_DEV_CLOCK_THROTTLE_REASONS`` -- the
    very name the variant injected -- and the replay aborted mid-run ("is not
    a distinct alias"). No exporter publishes two names of one field, so the
    pair is the published name and the family member the node does *not*
    publish, on the highest-index GPU reporting the field. A node publishing
    none, or every, name of each family is ``NOT_RUN(CAPABILITY_ABSENT)``
    naming the candidates, with nothing armed.
    """

    candidates = []
    for field, members in alias_families().items():
        gpus = survey.gpus_with(field)
        published_by_gpu = {gpu: survey.exporter_metrics(gpu, field) for gpu in gpus}
        seen = sorted({name for names in published_by_gpu.values() for name in names})
        candidates.append(
            f"{field} ({'|'.join(members)}; published: {', '.join(seen) or 'none'})"
        )
        for gpu in reversed(gpus):
            published = published_by_gpu[gpu]
            bare = {name.removesuffix("_total") for name in published}
            spare = [name for name in members if name not in bare]
            if len(published) == 1 and spare:
                return AliasPair(
                    gpu=gpu, field=field, published=published[0], alias=spare[0]
                )
    raise VariantNotRun(
        "CAPABILITY_ABSENT",
        "no canonical field has two distinct exporter aliases on this node: "
        + "; ".join(candidates),
    )


def conflicting_value(real: float | None) -> int:
    return ALIAS_VALUE if real != ALIAS_VALUE else ALIAS_VALUE // 2


def _hold(ctx: VariantContext) -> int:
    return min(ctx.spec.max_minutes * 60, 16 * 60)


def _settle(ctx: VariantContext, survey) -> None:
    ctx.hold(min(survey.interval * 4 + 10, _hold(ctx)), name="settle")


def _state(ctx: VariantContext, since: Any, *names: str) -> dict[str, Any]:
    return state_now(ctx, since, names=list(names) or [REQUIRED])


def _batch_wide_omission(errors: list[str]) -> bool:
    """Whether the product reported ``REQUIRED`` missing from the whole scrape.

    The batch-wide message is ``DCGM exporter is missing required fields:
    <fields>``; a GPU-scoped gap reads ``... fields on <gpu>: <fields>`` and an
    invalid slot ``invalid or ambiguous GPU observation: <gpu>/<field>``. The
    stage used to accept any text carrying "missing required fields", which a
    real gap of another field inside the result window (it opens before the
    arm) would have satisfied without the replay.
    """

    return any(BATCH_WIDE_OMISSION.search(text) for text in errors)


def missing_field(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpus = survey.gpus_with(REQUIRED)
    if not gpus:
        raise VariantNotRun(
            "CAPABILITY_ABSENT", f"no GPU on the node reports {REQUIRED}"
        )
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "completeness",
            baseline(),
            step(40, {"op": "omit", "field": REQUIRED, "gpus": gpus}, hold=True),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, *[f"{gpu}/{REQUIRED}" for gpu in gpus])
    _settle(ctx, survey)
    doc = _state(ctx, since)
    errors = collection_errors(doc)
    stages = {
        "completeness": []
        if _batch_wide_omission(errors)
        else ["a batch-wide missing required field was not reported as incomplete"],
    }
    recover_counter(ctx, repl, since, names=[REQUIRED])
    return result(stages, extra={"collection_errors": errors[:5]})


def one_gpu_missing(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpus = survey.gpus_with(REQUIRED)
    if len(gpus) < 2:
        raise VariantNotRun(
            "CAPABILITY_ABSENT", "need at least two GPUs to hide a per-GPU omission"
        )
    target = gpus[-1]
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "completeness",
            baseline(),
            step(40, {"op": "omit", "field": REQUIRED, "gpus": [target]}, hold=True),
        ),
        _hold(ctx),
    )
    # The same gate as ``missing_field``: the settle starts once the omit is
    # served (the plan's pass-through reads come first), not right after the
    # arm -- the journal peek this used to make asserted nothing.
    arm_and_confirm(repl, f"{target}/{REQUIRED}")
    _settle(ctx, survey)
    doc = _state(ctx, since)
    errors = collection_errors(doc)
    flagged = any(target in text and REQUIRED in text for text in errors) or bool(
        active_findings(doc, "dcgm_field_completeness")
    )
    # expected_outcome=FAIL: PASS only once the product checks completeness per
    # GPU and flags the omission on ``target``.
    stages = {
        "per_gpu_detected": []
        if flagged
        else [
            f"{REQUIRED} missing on {target} only was not flagged: the whole-batch "
            "completeness check is computed over the union of all GPUs' fields"
        ],
    }
    recover_counter(ctx, repl, since, names=[REQUIRED])
    return result(stages, extra={"target_gpu": target, "collection_errors": errors[:5]})


def conflicting_alias(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    pair = alias_pair(survey)
    value = conflicting_value(survey.value(pair.gpu, pair.field))
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "completeness",
            baseline(),
            step(
                40,
                {
                    "op": "alias",
                    "field": pair.field,
                    "gpus": [pair.gpu],
                    "name": pair.alias,
                    "value": value,
                },
                hold=True,
            ),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, f"{pair.gpu}/{pair.field}#alias:{pair.alias}")
    served_at = ctx.clock()
    _settle(ctx, survey)
    doc = _state(ctx, since, pair.field)
    # The ambiguity error always names ``<gpu>/<field>``, on a one-GPU node too.
    replayed_at = _first_replayed_at(doc, pair.gpu, (pair.field,), single_gpu=False)
    overwritten, notes = _alias_overwrite(
        doc, pair, value, replayed_at=replayed_at, served_at=served_at
    )
    stages = {
        "ambiguous_rejected": []
        if replayed_at is not None
        else ["a conflicting field alias was not reported as an ambiguous slot"],
        "no_silent_overwrite": overwritten,
    }
    recover_counter(ctx, repl, since, names=[pair.field])
    return result(
        stages,
        notes=notes,
        extra={
            "gpu_uuid": pair.gpu,
            "alias_pair": {**pair.describe(), "value": value},
            "collection_errors": collection_errors(doc)[:5],
            "first_conflicting_batch_at": (
                replayed_at.isoformat() if replayed_at is not None else None
            ),
            "window_opened_at": (replayed_at or served_at).isoformat(),
            "latest": latest_value(doc, pair.field, pair.gpu),
        },
    )


def _when(text: Any) -> datetime | None:
    """A product record's ``observed_at`` as an aware datetime (None: unreadable)."""

    try:
        value = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nan_negative_target(survey) -> str:
    """The highest-index GPU reporting both replayed fields.

    The NaN field used to be taken on trust (S19 audit): a GPU whose exporter
    does not report it has no slot to serve the NaN on, and the tool would
    refuse to invent one.
    """

    gpus = [gpu for gpu in survey.gpus_with(REQUIRED) if survey.has(gpu, NAN_FIELD)]
    if not gpus:
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            f"no GPU on the node reports both {REQUIRED} and {NAN_FIELD}",
        )
    return gpus[-1]


def _names_slot(text: str, gpu: str, name: str, *, single_gpu: bool) -> bool:
    """Whether a collection error is about ``gpu``'s ``name`` slot.

    The invalid-observation error names ``<gpu>/<field>``; a required field
    only some GPUs lack is reported ``on <gpu>``, one no GPU reports in the
    batch-wide message, which names no GPU (a one-GPU node).
    """

    return name in text and (gpu in text or single_gpu)


def _first_replayed_at(
    doc: dict[str, Any], gpu: str, names: tuple[str, ...], *, single_gpu: bool
) -> datetime | None:
    """When the product first processed a replayed scrape: the earliest batch
    since the arm whose collection errors carry its rejection of one of the
    replayed slots (the negative value, the NaN slot read as missing, the
    ambiguous alias or duplicate slot)."""

    times = [
        at
        for batch in doc.get("batches") or []
        if any(
            _names_slot(str(text), gpu, name, single_gpu=single_gpu)
            for text in batch.get("collection_errors") or []
            for name in names
        )
        and (at := _when(batch.get("observed_at"))) is not None
    ]
    return min(times) if times else None


def _not_a_value(
    doc: dict[str, Any],
    name: str,
    gpu: str,
    *,
    kind: str,
    replayed_at: datetime | None,
) -> tuple[list[str], str | None]:
    """Errors if an invalid ``name`` reading became a value; else what was kept.

    The product's latest row of a slot is its latest *accepted* sample, so a
    row older than the first replayed scrape is the earlier real reading,
    retained because the invalid one was dropped -- the live 2026-10-26 run
    read that retained zero as "became a healthy zero". A row timestamped at
    or after that scrape, an accepted batch sample inside the replay, or a
    negative value anywhere is the conversion the variant forbids.
    """

    errors: list[str] = []
    accepted = [
        sample
        for sample in accepted_samples(doc, name, gpu)
        if replayed_at is not None
        and (at := _when(sample.get("observed_at"))) is not None
        and at >= replayed_at
    ]
    if accepted:
        values = sorted({str(sample.get("value")) for sample in accepted})
        errors.append(
            f"a {kind} {name} reading was accepted as a sample while it was "
            f"served: values {values}"
        )
    latest = latest_value(doc, name, gpu)
    if latest is None:
        return errors, f"{gpu}/{name} has no latest accepted sample"
    value = _float(latest.get("value"))
    at = _when(latest.get("observed_at"))
    if value is not None and value < 0:
        errors.append(
            f"a negative {name} reading is the latest accepted value: {value}"
        )
    elif replayed_at is None:
        errors.append(
            f"{name} latest value {value} cannot be judged: no batch since the arm "
            "carries the product's rejection of a replayed reading"
        )
    elif at is None:
        errors.append(
            f"{name} latest value {value} carries no readable observed_at: "
            f"{latest.get('observed_at')!r}"
        )
    elif at >= replayed_at:
        errors.append(
            f"a {kind} {name} reading became the latest accepted value {value} at "
            f"{at.isoformat()}, not before the first replayed scrape at "
            f"{replayed_at.isoformat()}"
        )
    else:
        return errors, (
            f"{gpu}/{name} kept its earlier accepted value {value} from "
            f"{at.isoformat()}, before the first replayed scrape at "
            f"{replayed_at.isoformat()}; the {kind} reading was dropped"
        )
    return errors, None


def _alias_overwrite(
    doc: dict[str, Any],
    pair: AliasPair,
    value: int,
    *,
    replayed_at: datetime | None,
    served_at: datetime,
) -> tuple[list[str], list[str]]:
    """Errors if the ambiguous slot produced a value; notes on what was kept.

    Two exporter names of one field disagreeing on one GPU are one ``(device,
    canonical)`` slot seen twice, and the product drops the whole slot of that
    batch: ``gpu_environment_policy.deduplicate_dcgm_aliases`` collapses an
    alias pair only when both values and labels are equal, and
    ``gpu_observation.normalize_gpu_observation`` marks every sample of a slot
    counted twice "invalid or ambiguous" and keeps none. The published reading
    goes with the injected one; ``gpu_metric_latest`` is upserted for accepted
    samples only, so the slot's latest row stays the pre-conflict reading. "No
    silent overwrite" is therefore: the injected value is never a sample or
    the latest (the real reading never equals it, ``conflicting_value``), and
    from the first conflicting batch the product recorded (``replayed_at``)
    the slot has no accepted sample at all and its latest row predates it.
    The plan's pass-through reads before the conflict carry the published
    reading legitimately and are not graded -- live 2026-10-26 15:30Z
    (HyperPod, H200) the collector's initial-baseline scrape after the repoint
    delivered 1.0 inside the result window, which opens before the arm, and
    the variant FAILed on it. Without the product's own record of the conflict
    the window opens at ``served_at``, when the oracle gate saw the alias
    served: every later read is on the held step.
    """

    opened = replayed_at or served_at
    anchor = (
        f"the first conflicting batch at {opened.isoformat()}"
        if replayed_at is not None
        else f"the oracle gate at {opened.isoformat()}"
    )
    gpu, field = pair.gpu, pair.field
    errors: list[str] = []
    injected: list[str] = []
    inside: list[str] = []
    for sample in accepted_samples(doc, field, gpu):
        at = _when(sample.get("observed_at"))
        if _float(sample.get("value")) == float(value):
            injected.append(str(sample.get("observed_at")))
        elif at is not None and at >= opened:
            inside.append(str(sample.get("value")))
    if injected:
        errors.append(
            f"the injected alias value {value} was accepted as a {field} sample "
            f"of {gpu} at {', '.join(injected)}"
        )
    if inside:
        errors.append(
            f"a {field} sample of {gpu} was accepted while the conflicting alias "
            f"was served (from {opened.isoformat()}): values {sorted(set(inside))}; "
            "the product drops the whole slot of an ambiguous batch"
        )
    latest = latest_value(doc, field, gpu)
    if latest is None:
        return errors, [f"{gpu}/{field} has no latest accepted sample"]
    kept = _float(latest.get("value"))
    at = _when(latest.get("observed_at"))
    if kept == float(value):
        errors.append(
            f"the injected alias value {value} is the latest accepted {field} "
            f"value of {gpu}"
        )
    elif at is None:
        errors.append(
            f"{field} latest value {kept} carries no readable observed_at: "
            f"{latest.get('observed_at')!r}"
        )
    elif at >= opened:
        errors.append(
            f"a {field} reading of {gpu} became the latest accepted value {kept} "
            f"at {at.isoformat()}, not before {anchor}"
        )
    else:
        return errors, [
            f"{gpu}/{field} kept its earlier accepted value {kept} from "
            f"{at.isoformat()}, before {anchor}; the ambiguous slot was dropped"
        ]
    return errors, []


def nan_negative(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpu = _nan_negative_target(survey)
    single_gpu = len(survey.gpus) == 1
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "completeness",
            baseline(),
            step(
                40,
                {"op": "negative", "field": REQUIRED, "gpus": [gpu]},
                {"op": "nan", "field": NAN_FIELD, "gpus": [gpu]},
                hold=True,
            ),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, f"{gpu}/{REQUIRED}", f"{gpu}/{NAN_FIELD}")
    _settle(ctx, survey)
    doc = _state(ctx, since, REQUIRED, NAN_FIELD)
    errors = collection_errors(doc)
    replayed_at = _first_replayed_at(
        doc, gpu, (REQUIRED, NAN_FIELD), single_gpu=single_gpu
    )
    negative_errors, negative_note = _not_a_value(
        doc, REQUIRED, gpu, kind="negative", replayed_at=replayed_at
    )
    nan_errors, nan_note = _not_a_value(
        doc, NAN_FIELD, gpu, kind="NaN", replayed_at=replayed_at
    )
    stages = {
        "negative_rejected": []
        if any(gpu in text and REQUIRED in text for text in errors)
        else ["a negative counter was not reported as an invalid observation"],
        "nan_dropped": []
        if any(
            _names_slot(text, gpu, NAN_FIELD, single_gpu=single_gpu) for text in errors
        )
        else [f"a NaN {NAN_FIELD} was not reported as a missing field of {gpu}"],
        "no_healthy_zero": [*negative_errors, *nan_errors],
    }
    recover_counter(ctx, repl, since, names=[REQUIRED, NAN_FIELD])
    return result(
        stages,
        notes=[note for note in (negative_note, nan_note) if note],
        extra={
            "gpu_uuid": gpu,
            "collection_errors": errors[:5],
            "first_replayed_batch_at": (
                replayed_at.isoformat() if replayed_at is not None else None
            ),
            "latest": {
                name: latest_value(doc, name, gpu) for name in (REQUIRED, NAN_FIELD)
            },
        },
    )


def duplicate_slot(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpus = survey.gpus_with(REQUIRED)
    if len(gpus) < 2:
        raise VariantNotRun(
            "CAPABILITY_ABSENT",
            "need a second GPU to show independent samples stay usable",
        )
    target, other = gpus[-1], gpus[0]
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "completeness",
            baseline(),
            step(
                40,
                {"op": "duplicate", "field": REQUIRED, "gpus": [target], "value": 7},
                hold=True,
            ),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, f"{target}/{REQUIRED}#duplicate")
    _settle(ctx, survey)
    doc = _state(ctx, since)
    errors = collection_errors(doc)
    replayed_at = _first_replayed_at(doc, target, (REQUIRED,), single_gpu=False)
    stages = {
        "duplicate_rejected": []
        if replayed_at is not None
        else ["a duplicate field slot was not reported as ambiguous"],
        "others_usable": _others_usable(doc, other, target, replayed_at=replayed_at),
    }
    recover_counter(ctx, repl, since, names=[REQUIRED])
    return result(
        stages,
        extra={
            "target_gpu": target,
            "collection_errors": errors[:5],
            "first_replayed_batch_at": (
                replayed_at.isoformat() if replayed_at is not None else None
            ),
        },
    )


def _others_usable(
    doc: dict[str, Any], other: str, target: str, *, replayed_at: datetime | None
) -> list[str]:
    """``other``'s reading of the duplicated field was accepted *inside* the
    replay: a batch sample at/after the first batch the product rejected the
    duplicate in, or its latest row timestamped there. A latest row is not
    windowed by the read's ``since`` -- the stage used to pass on any row the
    GPU ever had, from before the arm as well."""

    if replayed_at is None:
        return [
            f"{other}'s samples cannot be judged: no batch since the arm carries "
            f"the product's rejection of the duplicate slot on {target}"
        ]
    latest = latest_value(doc, REQUIRED, other)
    kept = _when(latest.get("observed_at")) if latest is not None else None
    if (kept is not None and kept >= replayed_at) or any(
        (at := _when(sample.get("observed_at"))) is not None and at >= replayed_at
        for sample in accepted_samples(doc, REQUIRED, other)
    ):
        return []
    return [
        f"a duplicate slot on {target} also dropped {other}'s independent sample: "
        f"no {REQUIRED} sample of {other} was accepted at or after the first "
        f"replayed scrape at {replayed_at.isoformat()} (latest row: "
        f"{latest.get('observed_at') if latest else None})"
    ]


VARIANTS = [
    gpu_spec(
        CASE,
        "missing_field",
        "omit a required field from every GPU",
        "the whole-batch completeness check reports the missing field",
        missing_field,
        max_minutes=14,
    ),
    gpu_spec(
        CASE,
        "one_gpu_missing",
        "omit a required field from one GPU only",
        "per-GPU completeness flags the omission and names the GPU and field group "
        "(batch 160: completeness is judged per GPU; another GPU's field no longer hides it)",
        one_gpu_missing,
        max_minutes=14,
    ),
    gpu_spec(
        CASE,
        "conflicting_alias",
        "serve the product's other exporter name of a field the node publishes under one name, with a different value",
        "the ambiguous slot is reported and dropped whole from the first conflicting batch on: neither the "
        "injected value nor the published reading is accepted while the conflict is served and the slot's "
        "latest row stays the pre-conflict reading (the pass-through reads before it are not graded); "
        "no distinct alias pair on the node is NOT_RUN before anything is installed",
        conflicting_alias,
        max_minutes=14,
    ),
    gpu_spec(
        CASE,
        "nan_negative",
        "serve a negative value and a NaN",
        "a negative value is rejected and a NaN is dropped; neither becomes a value: "
        "both slots keep their earlier accepted sample from before the replayed scrape",
        nan_negative,
        max_minutes=14,
    ),
    gpu_spec(
        CASE,
        "duplicate_slot",
        "serve a duplicated (device, field) slot",
        "the duplicate slot is rejected; other GPUs' independent samples stay usable",
        duplicate_slot,
        max_minutes=14,
    ),
]
