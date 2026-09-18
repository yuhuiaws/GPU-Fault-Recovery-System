"""Kubernetes omits empty args without changing the proof container command."""

from __future__ import annotations

import json
from typing import Any

import pytest

from gpu_fault_release import regional_release_probe_job as jobs
from gpu_fault_release import regional_release_store_proof as proof
from gpu_fault_release.regional_release_config import ReleaseError
from tests.regional._prerequisite_repair_support import repair_release


def set_args(document: dict[str, Any], value: list[str] | None) -> None:
    container = document["spec"]["template"]["spec"]["containers"][0]
    if value is None:
        container.pop("args", None)
    else:
        container["args"] = value


@pytest.mark.parametrize(
    ("requested", "admitted", "equivalent"),
    [
        ([], None, True),
        (None, [], True),
        ([], ["--unexpected"], False),
        (["--required"], None, False),
        (["--required"], [], False),
        (["--required"], ["--different"], False),
    ],
    ids=[
        "server-omits-empty",
        "server-emits-empty",
        "added-argument",
        "removed-argument",
        "emptied-argument",
        "changed-argument",
    ],
)
def test_proof_job_preserves_argument_semantics_through_admission(
    monkeypatch: pytest.MonkeyPatch,
    requested: list[str] | None,
    admitted: list[str] | None,
    equivalent: bool,
) -> None:
    release = repair_release(monkeypatch)
    release.runner.database_state = "uninitialized_empty"
    original_run = release.runner.run

    def admission(arguments: list[str], **options: Any) -> str:
        output = original_run(arguments, **options)
        if "--dry-run=server" in arguments and "-o" in arguments:
            document = json.loads(output)
            set_args(document, admitted)
            return json.dumps(document)
        return str(output)

    def run_job(
        instance: Any, document: dict[str, Any], checkpoint: jobs.Checkpoint
    ) -> dict[str, Any]:
        set_args(document, requested)
        return jobs.run_probe_job(instance, document, checkpoint)

    monkeypatch.setattr(release.runner, "run", admission)
    monkeypatch.setattr(proof, "run_probe_job", run_job)
    if equivalent:
        result = proof.bootstrap_store_proof(release, lambda _: None)
        assert result["database_state"] == "uninitialized_empty", (
            "an equivalent API representation prevented the bootstrap proof"
        )
        assert len(release.runner.created_jobs) == 1, (
            "the proof must create exactly one owned Job"
        )
        assert release.runner.jobs == {}, "the proof Job was not cleaned up"
    else:
        with pytest.raises(ReleaseError, match="Job admission identity differs"):
            proof.bootstrap_store_proof(release, lambda _: None)
        assert release.runner.created_jobs == [], (
            "changed arguments reached execution before rejection"
        )
