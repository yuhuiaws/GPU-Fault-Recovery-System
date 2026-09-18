#!/usr/bin/env python3
"""Guarded full retirement of a separately provisioned BOOT-032 sacrificial site."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_fault.admin.site import load_site  # noqa: E402
from scripts.e2e.regional.boot032_contract import (  # noqa: E402
    CASE_ID,
    CONFIRMATION,
    Settings,
    checked_path,
    validate_sites,
)
from scripts.e2e.regional.boot032_lifecycle import (  # noqa: E402
    execute_case,
    plan_details,
    read_only_preflight,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseRunner,
    add_live_arguments,
    run_standard_case,
)
from scripts.e2e.regional.regional_live_fixture import run_case_main  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--fixture-id", required=True)
    value.add_argument("--protected-site", type=Path, required=True)
    value.add_argument("--protected-cluster-id", required=True)
    return value


def configure(arguments: argparse.Namespace) -> Settings:
    root = arguments.run_dir.expanduser().absolute()
    target = checked_path(root / "cases" / CASE_ID / "sacrificial" / "site.yaml")
    protected = checked_path(arguments.protected_site)
    settings = Settings(
        run_dir=root.resolve(),
        fixture_id=arguments.fixture_id,
        target=load_site(target),
        protected=load_site(protected),
        arguments=arguments,
        protected_cluster_id=arguments.protected_cluster_id,
    )
    validate_sites(settings)
    return settings


CASE = CaseRunner(
    case_id=CASE_ID,
    confirmation=CONFIRMATION,
    parser=parser,
    configure=configure,
    read_only_preflight=read_only_preflight,
    plan_details=plan_details,
    execute_case=execute_case,
)


def main() -> int:
    return run_standard_case(CASE)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
