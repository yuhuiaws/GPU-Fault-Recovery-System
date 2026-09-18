from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import blast_acceptance_base as base
from scripts.e2e.regional import run_workload_acceptance as workload
from tests.regional._blast_containment_support import produce_containment
from tests.regional.test_blast_acceptance_review import make_runner
from tests.regional.test_identity_causal_review import lifecycle_harness


def containment_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Any, dict[str, Any], list[str], list[tuple[str, ...]]]:
    runner = make_runner(monkeypatch, tmp_path, base.CASE_IDS[0])
    runner.targets = [replace(runner.targets[0], cluster_id="a")]
    runner.e2e_dir = base.default_e2e_dir(runner.root_run_dir)
    runner.trusted_cpu_baseline = base.default_trusted_cpu_baseline(runner.e2e_dir)
    runner.e2e_dir.mkdir(parents=True)
    site, targets, events = lifecycle_harness(monkeypatch, runner.e2e_dir)
    outcome = workload.run_e2e001(
        site=site,
        target=targets[0],
        case_dir=runner.e2e_dir,
        job_id="job-test",
        attempt_id="attempt-test",
        host_probe_image="unit@sha256:" + "a" * 64,
        attempt=1,
        maintenance_window_end=datetime.now(timezone.utc) + timedelta(minutes=30),
    )
    assert outcome["verdict"] == "PASS"
    nodes = json.loads(runner.trusted_cpu_baseline.read_text())
    runner.predecessor = produce_containment(
        monkeypatch, runner.root_run_dir, nodes, release="release-test", cluster="a"
    )
    documents = {"nodes": nodes, "jobs": {"items": []}, "events": {"items": []}}
    original = runner.cpu_json
    reads: list[tuple[str, ...]] = []

    def cpu_json(*args: str) -> Any:
        reads.append(args)
        if args[0] == "get":
            return documents[args[1]]
        return original(*args)

    monkeypatch.setattr(runner, "cpu_json", cpu_json)
    return runner, documents, events, reads
