#!/usr/bin/env python3
"""Guarded NOTIFY008 candidate; catalog registration remains a prerequisite."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)
    from scripts.e2e.regional.live_driver_guard import run_standard_case
    from scripts.e2e.regional.notify008_runner import CASE
    from scripts.e2e.regional.regional_live_fixture import run_case_main

    return run_case_main(lambda: run_standard_case(CASE))


if __name__ == "__main__":
    raise SystemExit(main())
