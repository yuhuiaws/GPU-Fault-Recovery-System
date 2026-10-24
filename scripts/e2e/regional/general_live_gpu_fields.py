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
  values: both are rejected as an ambiguous slot, never silently overwritten;
* ``nan_negative`` -- a negative value is rejected and a NaN is dropped: a slot
  never becomes a healthy zero;
* ``duplicate_slot`` -- a duplicated ``(device, field)`` slot is rejected while
  other GPUs' independent samples stay usable.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional.general_live_fixture import VariantContext, VariantNotRun
from scripts.e2e.regional.general_live_gpu_replay import (
    DcgmReplay,
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

CASE = "GF-GENERAL-LIVE-025"
REQUIRED = "pcie_replay_total"
ALIAS_FIELD = "clock_throttle_reasons"
ALIAS_NAME = "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS"


def _hold(ctx: VariantContext) -> int:
    return min(ctx.spec.max_minutes * 60, 16 * 60)


def _settle(ctx: VariantContext, survey) -> None:
    ctx.hold(min(survey.interval * 4 + 10, _hold(ctx)), name="settle")


def _state(ctx: VariantContext, since: Any, *names: str) -> dict[str, Any]:
    return state_now(ctx, since, names=list(names) or [REQUIRED])


def missing_field(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpus = survey.gpus_with(REQUIRED)
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
        if any(
            REQUIRED in text or "missing required fields" in text.lower()
            for text in errors
        )
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
    arm_and_confirm(repl)  # the oracle gate below reads the audit for the omit
    if not any(
        f"{target}/{REQUIRED}" in str((rec.get("served") or {})) for rec in repl.records
    ):
        repl.audit()
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
    gpu = survey.target(ALIAS_FIELD)
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
                    "field": ALIAS_FIELD,
                    "gpus": [gpu],
                    "name": ALIAS_NAME,
                    "value": 64,
                },
                hold=True,
            ),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl, f"{gpu}/{ALIAS_FIELD}#alias:{ALIAS_NAME}")
    _settle(ctx, survey)
    doc = _state(ctx, since, ALIAS_FIELD)
    errors = collection_errors(doc)
    samples = accepted_samples(doc, ALIAS_FIELD, gpu)
    stages = {
        "ambiguous_rejected": []
        if any(gpu in text and ALIAS_FIELD in text for text in errors)
        else ["a conflicting field alias was not reported as an ambiguous slot"],
        "no_silent_overwrite": []
        if not samples
        else [
            f"a conflicting alias slot was accepted as a value: {samples[-1].get('value')}"
        ],
    }
    recover_counter(ctx, repl, since, names=[ALIAS_FIELD])
    return result(stages, extra={"gpu_uuid": gpu, "collection_errors": errors[:5]})


def nan_negative(ctx: VariantContext) -> dict[str, Any]:
    survey = run_gpu_survey(ctx)
    gpu = survey.target(REQUIRED)
    since = utc_now()
    repl = DcgmReplay(
        ctx,
        plan(
            "completeness",
            baseline(),
            step(
                40,
                {"op": "negative", "field": REQUIRED, "gpus": [gpu]},
                {"op": "nan", "field": "row_remap_failure", "gpus": [gpu]},
                hold=True,
            ),
        ),
        _hold(ctx),
    )
    arm_and_confirm(repl)
    _settle(ctx, survey)
    doc = _state(ctx, since, REQUIRED, "row_remap_failure")
    errors = collection_errors(doc)
    negative_latest = latest_value(doc, REQUIRED, gpu)
    nan_latest = latest_value(doc, "row_remap_failure", gpu)
    stages = {
        "negative_rejected": []
        if any(gpu in text and REQUIRED in text for text in errors)
        else ["a negative counter was not reported as an invalid observation"],
        "no_healthy_zero": [],
    }
    if negative_latest is not None and float(negative_latest.get("value", -1)) >= 0:
        stages["no_healthy_zero"].append(
            "a negative reading became a non-negative value"
        )
    if nan_latest is not None and float(nan_latest.get("value", 1)) == 0:
        stages["no_healthy_zero"].append("a NaN reading became a healthy zero")
    recover_counter(ctx, repl, since, names=[REQUIRED])
    return result(stages, extra={"gpu_uuid": gpu, "collection_errors": errors[:5]})


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
    stages = {
        "duplicate_rejected": []
        if any(target in text and REQUIRED in text for text in errors)
        else ["a duplicate field slot was not reported as ambiguous"],
        "others_usable": []
        if latest_value(doc, REQUIRED, other) is not None
        else [
            f"a duplicate slot on {target} also dropped {other}'s independent sample"
        ],
    }
    recover_counter(ctx, repl, since, names=[REQUIRED])
    return result(stages, extra={"target_gpu": target, "collection_errors": errors[:5]})


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
        "per-GPU completeness must flag the omission; the whole-batch check hides it (spec FAIL)",
        one_gpu_missing,
        max_minutes=14,
        expected_outcome="FAIL",
        product_gap="src/gpu_fault/collectors/gpu/dcgm.py DcgmMetricsCollector.collect_text computes "
        "observed_fields = {sample.canonical_name for sample in samples} over ALL GPUs and passes it "
        "to src/gpu_fault/dcgm_fields.py missing_dcgm_metric_groups, so a field present on any GPU "
        "hides its absence on another GPU; there is no per-GPU completeness check in the metrics "
        "ingestion path (VALIDATE_GPU checks per-GPU presence only during a workflow, not here)",
    ),
    gpu_spec(
        CASE,
        "conflicting_alias",
        "serve two exporter aliases of one field with different values",
        "the ambiguous slot is rejected, never silently overwritten",
        conflicting_alias,
        max_minutes=14,
    ),
    gpu_spec(
        CASE,
        "nan_negative",
        "serve a negative value and a NaN",
        "a negative value is rejected and a NaN is dropped; neither becomes a healthy zero",
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
