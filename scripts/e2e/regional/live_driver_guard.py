from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

ROOT = Path(__file__).resolve().parents[3]
# The trees whose content decides whether a recorded focused-test result still
# speaks for the code about to run: the drivers, their unit tests, the package.
SOURCE_DIGEST_PATHS = ("scripts/e2e/regional", "tests/regional", "src")

if __package__:
    from .acceptance_runner_common import write_json_atomic
    from .acceptance_scope import current_acceptance_scope
    from .regional_live_fixture import RegionalFixtureError, install_abort_signals
    from .site_profile import (
        SITE_PROFILE_ENV,
        applied_site_profile,
        bind_site_profile,
        install_site_profile,
    )
else:
    from acceptance_runner_common import write_json_atomic
    from acceptance_scope import current_acceptance_scope
    from regional_live_fixture import RegionalFixtureError, install_abort_signals
    from site_profile import (
        SITE_PROFILE_ENV,
        applied_site_profile,
        bind_site_profile,
        install_site_profile,
    )

# Re-exported so the runners keep one import site for the live-driver contract:
# every one of them already imports `add_live_arguments` from here.
__all__ = [
    "CaseRunner",
    "CaseSurface",
    "add_live_arguments",
    "bind_site_profile",
    "authorize_execution",
    "build_plan",
    "environment_snapshot",
    "install_site_profile",
    "PlainCaseRunner",
    "details_sha256",
    "record_focused_tests",
    "reusable_focused_tests",
    "run_plain_case",
    "run_selected_case",
    "run_standard_case",
    "source_digest",
]


COMMON_ENVIRONMENT_KEYS = (
    "GPU_FAULT_CONTROL_KUBECONFIG",
    "GPU_FAULT_DATAPLANE_CONTEXT",
    "GPU_FAULT_PERF_AWS_REGION",
    "GPU_FAULT_PERF_CONTROL_NAMESPACE",
    "GPU_FAULT_PERF_DATAPLANE_NAMESPACE",
    "KUBECONFIG",
)


def add_live_arguments(
    parser: argparse.ArgumentParser,
    *,
    confirmation: str,
) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument(
        "--site-profile",
        default=os.getenv(SITE_PROFILE_ENV, ""),
        help=(
            "private file naming this site's clusters, contexts and nodes; "
            "explicit flags on this command line override it"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--confirm",
        default="",
        help=f"execute mode requires exactly {confirmation}",
    )
    parser.add_argument(
        "--maintenance-window-end",
        default=os.getenv("GPU_FAULT_ACCEPTANCE_WINDOW_END", ""),
        help="UTC ISO-8601 deadline required by execute mode",
    )
    # Every live runner passes through here, so this is the one place that has
    # to know the profile supplies arguments. It is evaluated at parse time, so
    # flags the runner adds after this call are still filled from the profile.
    bind_site_profile(parser)


def environment_snapshot(
    keys: tuple[str, ...] = COMMON_ENVIRONMENT_KEYS,
) -> dict[str, str]:
    missing = [key for key in keys if not os.getenv(key, "").strip()]
    if missing:
        raise RuntimeError(
            "required live-driver environment is missing: " + ", ".join(sorted(missing))
        )
    return {key: os.environ[key].strip() for key in keys}


def _deadline(value: str) -> datetime:
    if not value.strip():
        raise RuntimeError(
            "--maintenance-window-end or GPU_FAULT_ACCEPTANCE_WINDOW_END is required"
        )
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise RuntimeError("maintenance-window-end must include a timezone")
    return parsed.astimezone(timezone.utc)


def details_sha256(details: dict[str, Any]) -> str:
    """The digest ``build_plan`` records over the plan's ``details``.

    Canonical JSON (sorted keys, no whitespace, ``default=str`` for the odd
    datetime a preflight leaves in) so the same details always hash the same
    whatever order a runner built them in.
    """

    canonical = json.dumps(
        details,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_plan(
    *,
    run_dir: Path,
    case_id: str,
    attempt: int,
    confirmation: str,
    details: dict[str, Any],
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    scope = current_acceptance_scope()
    plan = {
        "schema_version": 2,
        "case_id": case_id,
        "attempt": attempt,
        "confirmation": confirmation,
        "environment": environment or environment_snapshot(),
        "site_profile": applied_site_profile(),
        **scope.plan_fields(),
        "details": details,
        "details_sha256": details_sha256(details),
        "mutation_performed": False,
    }
    write_json_atomic(run_dir / "cases" / case_id / "plan.json", plan)
    return plan


def authorize_execution(
    arguments: argparse.Namespace,
    *,
    case_id: str,
    confirmation: str,
    environment: dict[str, str] | None = None,
    details: dict[str, Any] | None = None,
) -> datetime:
    """Admit an ``--execute`` only against the plan the operator approved.

    Beyond the identity fields, the plan's ``details`` must still hash to the
    ``details_sha256`` written next to them -- the target node, the drill
    marker and the preflight verdict live in ``details``, and a plan edited
    after approval is not the plan that was approved. A runner that knows its
    current ``details`` at execute time passes them too, and they must hash to
    the same value; one that only learns them from the plan file leaves the
    parameter ``None`` and gets the on-disk integrity check alone.
    """

    if not arguments.execute:
        raise RuntimeError("execute authorization requested outside --execute mode")
    if arguments.confirm != confirmation:
        raise RuntimeError(f"confirmation must be exactly {confirmation}")
    deadline = _deadline(arguments.maintenance_window_end)
    if datetime.now(timezone.utc) >= deadline:
        raise RuntimeError("approved maintenance window has ended")
    path = arguments.run_dir / "cases" / case_id / "plan.json"
    if not path.is_file():
        raise RuntimeError("run --plan before --execute")
    plan = json.loads(path.read_text(encoding="utf-8"))
    scope = current_acceptance_scope()
    expected = {
        "schema_version": 2,
        "case_id": case_id,
        "attempt": arguments.attempt,
        "confirmation": confirmation,
        "environment": environment or environment_snapshot(),
        # A plan built against one site profile must not be executed under
        # another. Absent on both sides when no profile is in use, so plans
        # written before profiles existed still compare equal.
        "site_profile": applied_site_profile(),
        **scope.plan_fields(),
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise RuntimeError(f"live-driver plan drifted at {key}")
    recorded_details = plan.get("details")
    if not isinstance(recorded_details, dict):
        raise RuntimeError("live-driver plan drifted at details")
    recorded_digest = plan.get("details_sha256")
    if recorded_digest != details_sha256(recorded_details):
        raise RuntimeError("live-driver plan drifted at details_sha256")
    if details is not None and details_sha256(details) != recorded_digest:
        raise RuntimeError("live-driver plan drifted at details")
    return deadline


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    if completed.returncode:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def source_digest() -> str:
    """A digest of the source a focused-test result was computed against.

    ``git rev-parse HEAD`` plus the working-tree diff of the driver, test and
    package trees, so an uncommitted edit to any of them changes the digest
    while an unrelated edit (docs, deploy manifests) does not. Read-only.
    """

    head = _git_output("rev-parse", "HEAD").strip()
    diff = _git_output("diff", "HEAD", "--", *SOURCE_DIGEST_PATHS)
    return hashlib.sha256(f"{head}\n{diff}".encode()).hexdigest()


def record_focused_tests(details: dict[str, Any], result: dict[str, Any]) -> None:
    """Store a focused pytest result in plan ``details`` with its source digest."""

    details["focused_tests"] = result
    details["focused_tests_source_digest"] = source_digest()


def reusable_focused_tests(plan_path: Path) -> dict[str, Any] | None:
    """The plan's recorded focused-test result, if it still speaks for this tree.

    Returns the result only when it recorded ``passed: True`` and the source
    digest it was taken against equals ``source_digest()`` now; otherwise
    ``None``, and the runner re-runs pytest. A ``--plan`` that ran the tests
    minutes ago on the same tree is thereby not paid for twice in ``--execute``,
    while any edit in between forces a fresh run.
    """

    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(plan, dict):
        return None
    details = plan.get("details")
    if not isinstance(details, dict):
        return None
    recorded = details.get("focused_tests")
    digest = details.get("focused_tests_source_digest")
    if not isinstance(recorded, dict) or recorded.get("passed") is not True:
        return None
    if not isinstance(digest, str) or digest != source_digest():
        return None
    return dict(recorded)


class CaseSettings(Protocol):
    """What the shared ``main`` needs from a runner's configured settings."""

    def environment(self) -> dict[str, str]: ...


class SelectedCaseSettings(CaseSettings, Protocol):
    """Settings of a runner that picks its case at configure time."""

    @property
    def case_id(self) -> str: ...

    @property
    def confirmation(self) -> str: ...


S = TypeVar("S", bound=CaseSettings)
T = TypeVar("T", bound=SelectedCaseSettings)


@dataclass(frozen=True)
class CaseSurface(Generic[S]):
    """The functions one live runner hands to the shared ``main``.

    A dataclass rather than a module looked up by attribute, so a runner that
    forgets one of these fails while the module declares ``CASE`` -- at import,
    under the unit suite -- instead of on the first ``--execute`` against a
    cluster.
    """

    parser: Callable[[], argparse.ArgumentParser]
    configure: Callable[[argparse.Namespace], S]
    read_only_preflight: Callable[[S, Path], dict[str, Any]]
    plan_details: Callable[[S, dict[str, Any]], dict[str, Any]]
    execute_case: Callable[[S, Path, int, datetime], int]


@dataclass(frozen=True)
class CaseRunner(CaseSurface[S]):
    """A runner whose case identity is fixed by its module constants."""

    case_id: str
    confirmation: str


def _start_case(case: CaseSurface[S]) -> tuple[argparse.Namespace, S]:
    install_site_profile()
    arguments = case.parser().parse_args()
    os.umask(0o077)
    install_abort_signals()
    return arguments, case.configure(arguments)


def _case_dir(arguments: argparse.Namespace, case_id: str) -> Path:
    case_dir = Path(arguments.run_dir) / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def _plan_case(
    case: CaseSurface[S],
    arguments: argparse.Namespace,
    settings: S,
    *,
    case_dir: Path,
    case_id: str,
    confirmation: str,
) -> int:
    preflight = case.read_only_preflight(settings, case_dir)
    plan = build_plan(
        run_dir=arguments.run_dir,
        case_id=case_id,
        attempt=arguments.attempt,
        confirmation=confirmation,
        environment=settings.environment(),
        details=case.plan_details(settings, preflight),
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0 if not preflight["errors"] else 1


def _execute_case(
    case: CaseSurface[S],
    arguments: argparse.Namespace,
    settings: S,
    *,
    case_id: str,
    confirmation: str,
) -> int:
    deadline = authorize_execution(
        arguments,
        case_id=case_id,
        confirmation=confirmation,
        environment=settings.environment(),
    )
    return case.execute_case(
        settings,
        arguments.run_dir,
        arguments.attempt,
        deadline,
    )


def run_standard_case(case: CaseRunner[S]) -> int:
    """The ``main`` every fixed-identity live runner used to carry verbatim.

    Site profile first (the parser's defaults read the environment it fills),
    then parse, then a ``0o077`` umask before anything under ``--run-dir``
    exists, then the abort signals, then the case's own configuration. Without
    ``--execute`` this writes and prints the plan and exits 0 only when the
    read-only preflight found nothing wrong; with it, ``authorize_execution``
    must pass before ``execute_case`` runs, and its verdict is the exit code.
    """

    arguments, settings = _start_case(case)
    case_dir = _case_dir(arguments, case.case_id)
    if not arguments.execute:
        return _plan_case(
            case,
            arguments,
            settings,
            case_dir=case_dir,
            case_id=case.case_id,
            confirmation=case.confirmation,
        )
    return _execute_case(
        case,
        arguments,
        settings,
        case_id=case.case_id,
        confirmation=case.confirmation,
    )


@dataclass(frozen=True)
class PlainCaseRunner:
    """A runner with no configured settings.

    The plan is static (no read-only preflight, no environment snapshot) and
    the case reads the site from the environment only when it executes; the
    CMD/NET hold cases are this shape.
    """

    case_id: str
    confirmation: str
    parser: Callable[[], argparse.ArgumentParser]
    plan_details: Callable[[], dict[str, Any]]
    run_case: Callable[[Path, int, datetime], int]


def run_plain_case(case: PlainCaseRunner) -> int:
    """``run_standard_case`` for a ``PlainCaseRunner``.

    Site profile first, then parse, then the ``0o077`` umask before anything
    under ``--run-dir`` exists. Without ``--execute`` the static plan is
    printed and the exit code is 0; with it, ``authorize_execution`` must pass
    before ``run_case`` runs, and its verdict is the exit code.
    """

    install_site_profile()
    arguments = case.parser().parse_args()
    os.umask(0o077)
    if not arguments.execute:
        plan = build_plan(
            run_dir=arguments.run_dir,
            case_id=case.case_id,
            attempt=arguments.attempt,
            confirmation=case.confirmation,
            details=case.plan_details(),
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    deadline = authorize_execution(
        arguments, case_id=case.case_id, confirmation=case.confirmation
    )
    return case.run_case(arguments.run_dir, arguments.attempt, deadline)


def run_selected_case(case: CaseSurface[T]) -> int:
    """`run_standard_case` for a runner that selects its case from ``--case``.

    The identity comes from the configured settings, and a confirmation for a
    different case is refused as a fixture error before the live-driver guard
    sees it, so the operator reads which case they actually named.
    """

    arguments, settings = _start_case(case)
    case_dir = _case_dir(arguments, settings.case_id)
    if not arguments.execute:
        return _plan_case(
            case,
            arguments,
            settings,
            case_dir=case_dir,
            case_id=settings.case_id,
            confirmation=settings.confirmation,
        )
    if arguments.confirm != settings.confirmation:
        raise RegionalFixtureError(
            f"confirmation must be exactly {settings.confirmation}"
        )
    return _execute_case(
        case,
        arguments,
        settings,
        case_id=settings.case_id,
        confirmation=settings.confirmation,
    )
