"""Guarded entrypoint; local replay cannot authorize physical acceptance."""

from scripts.e2e.regional.late_ownership_runner import (
    PhysicalBoundaryIO as PhysicalBoundaryIO,
    RunResult as RunResult,
    require_no_errors as require_no_errors,
    run_case as run_case,
)


def main() -> int:
    from scripts.e2e.regional.late_ownership_entry import main as live_main

    return live_main()


if __name__ == "__main__":
    raise SystemExit(main())
