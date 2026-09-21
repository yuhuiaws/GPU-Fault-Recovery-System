"""The schema release Job is the only sanctioned schema creator; a fresh
database it creates starts the control-state tables in dedicated mode."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "deploy" / "migrations" / "postgres-schema-ensure-job.yaml"


def test_schema_ensure_job_seeds_fresh_databases_dedicated() -> None:
    document = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    container = document["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == ["gpu-fault-store-migrate"]
    assert container["args"] == [
        "--ensure-schema",
        "--fresh-control-state-mode",
        "dedicated",
    ], "a fresh install must not start on the legacy control-state path"
