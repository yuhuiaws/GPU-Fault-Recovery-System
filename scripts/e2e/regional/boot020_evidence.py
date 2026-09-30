"""BOOT-020 evidence identity and the resume contract around it.

The runner's evidence ``inputs`` must be equal on every attempt of one run or
``EvidenceRecorder`` refuses to resume. Live 2026-09-30 (fifth regional run)
the first execute built the candidates and recorded ``action=built`` with a
work directory; every later ``--resume`` found them present, computed
``action=reused`` and died on a message that named nothing. These helpers make
the inputs the run's identity (``evidence_inputs``), refuse a drifted resume by
key and value (``open_evidence``), and point a drifted plan at the two-command
resume sequence (``authorize_release_rolling``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from scripts.e2e.regional import boot020_release_prerequisites as prerequisites
from scripts.e2e.regional.acceptance_runner_common import EvidenceRecorder, utc_now
from scripts.e2e.regional.boot020_deployment_reads import AcceptanceCheckError
from scripts.e2e.regional.live_driver_guard import authorize_execution

# Bumped whenever the shape of the evidence ``inputs`` changes: an older
# document then fails the resume on this key first, which says why.
ACCEPTANCE_CONTRACT = 5
# The only supported way to continue a run whose candidates already exist.
# Both commands must carry the same ``--attempt`` and ``--resume`` (the plan
# binds every non-approval argument into ``arguments_sha256``), and must name
# the same site, candidates directory and GPU kubeconfig as the first attempt
# (the evidence ``inputs``).
RESUME_SEQUENCE = (
    "re-plan and execute with the same command line: "
    "run_boot020_release_rolling.py <same site/candidate arguments> "
    "--run-dir <run> --attempt <n> --resume --plan, then the identical command "
    "with --execute --confirm RUN_BOOT020_RELEASE_ROLLING "
    "--maintenance-window-end <end>; a different site, candidates directory or "
    "kubeconfig is a new run and needs a new --run-dir"
)


def evidence_inputs(
    arguments: argparse.Namespace,
    *,
    configs: dict[str, Path],
    candidates: dict[str, Any],
    gpu_kubeconfig: Path,
) -> dict[str, Any]:
    """The evidence ``inputs``: the run's identity, equal on every attempt.

    Built from the administrator state, the candidate files and the kubeconfig
    -- never from what this attempt did (``built`` vs ``reused``, work dir,
    timestamps), which goes into the ``release_candidates_provenance`` note.
    """

    state_dir = arguments.admin_state_dir.resolve()
    return {
        "acceptance_contract": ACCEPTANCE_CONTRACT,
        "admin_state_dir": str(state_dir),
        "admin_reference": arguments.admin_reference,
        "configs": {name: str(path) for name, path in configs.items()},
        "config_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in configs.items()
        },
        "release_candidates": prerequisites.candidate_identity(
            state_dir, configs, candidates
        ),
        "gpu_kubeconfig": str(gpu_kubeconfig),
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Nested input dicts as dotted keys; an empty nested dict stays a value."""

    if not isinstance(value, dict) or (prefix and not value):
        return {prefix: value}
    flat: dict[str, Any] = {}
    for key, item in value.items():
        flat.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    return flat


def evidence_input_drift(path: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    """Per differing input (dotted key), the recorded and the current value.

    Empty when there is no document yet or the inputs are equal. A malformed
    document is left for ``EvidenceRecorder`` to refuse.
    """

    if not path.exists():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    recorded = document.get("inputs") if isinstance(document, dict) else None
    if not isinstance(recorded, dict) or recorded == inputs:
        return {}
    before = _flatten(recorded)
    after = _flatten(inputs)
    return {
        key: {"recorded": before.get(key), "current": after.get(key)}
        for key in sorted(set(before) | set(after))
        if before.get(key) != after.get(key)
    }


def open_evidence(
    path: Path, *, case_id: str, inputs: dict[str, Any]
) -> EvidenceRecorder:
    """``EvidenceRecorder`` for this case, refusing a drifted resume by name.

    The recorder itself only says the inputs differ; the operator needs the
    keys and both values to see whether they resumed the wrong run or the
    identity is genuinely different (live 2026-09-30: four attempts to continue
    one run died on the bare message).
    """

    drift = evidence_input_drift(path, inputs)
    if drift:
        raise AcceptanceCheckError(
            "evidence inputs differ from the existing run at "
            + ", ".join(drift)
            + ": "
            + json.dumps(drift, sort_keys=True, default=str)
            + f"; to continue that run, {RESUME_SEQUENCE}"
        )
    return EvidenceRecorder(path, case_id=case_id, inputs=inputs)


def authorize_release_rolling(
    arguments: argparse.Namespace,
    *,
    case_id: str,
    confirmation: str,
    environment: dict[str, str],
    details: dict[str, Any],
    authorize: Callable[..., Any] = authorize_execution,
) -> None:
    """``authorize_execution`` with the resume sequence on a drifted plan.

    ``arguments_sha256`` covers ``--resume``, ``--start-stage`` and
    ``--attempt`` covers itself: a plan written without them cannot admit an
    execute with them, and the guard's message names only the key. The runner
    passes its own ``authorize_execution`` binding so the guard it calls is the
    one its module names (the entrypoint tests stub it there).
    """

    try:
        authorize(
            arguments,
            case_id=case_id,
            confirmation=confirmation,
            environment=environment,
            details=details,
        )
    except RuntimeError as exc:
        text = str(exc)
        if "drifted at arguments_sha256" in text or "drifted at attempt" in text:
            raise RuntimeError(
                f"{text}: the --plan and the --execute of one attempt must carry "
                "the same --attempt, --resume and --start-stage (and every other "
                f"non-approval argument); {RESUME_SEQUENCE}"
            ) from exc
        if "drifted at details" in text:
            raise RuntimeError(
                f"{text}: the plan was written against another candidate state "
                "(for example before the first execute built the candidates); "
                f"{RESUME_SEQUENCE}"
            ) from exc
        raise


def record_candidate_provenance(
    recorder: EvidenceRecorder,
    arguments: argparse.Namespace,
    candidates: dict[str, Any],
) -> list[dict[str, Any]]:
    """Append what this attempt did for the candidates: history, not identity.

    ``built`` / ``reused`` / ``explicit``, the work directory and the replicas
    delta live here (``release_candidates_provenance``), one entry per attempt,
    so the evidence still tells the reader which attempt built and which
    reused without any of it entering ``inputs``.
    """

    history = [
        *(recorder.document.get("release_candidates_provenance") or []),
        {
            "attempt": arguments.attempt,
            "recorded_at": utc_now(),
            "resume": arguments.resume,
            "start_stage": arguments.start_stage,
            **prerequisites.candidate_provenance(candidates),
        },
    ]
    recorded: list[dict[str, Any]] = recorder.note(
        "release_candidates_provenance", history
    )
    return recorded
