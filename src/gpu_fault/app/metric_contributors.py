from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from threading import Lock

from gpu_fault.app.runtime import AppRuntime
from gpu_fault.plugins import PluginGroup, discover_plugins

LOGGER = logging.getLogger(__name__)

MetricContributor = Callable[[AppRuntime], list[str]]

ERRORS_METRIC = "gpu_fault_metrics_contributor_errors_total"


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class MetricContributorRegistry:
    """The ordered set of ``/metrics`` producers, rendered with two guards.

    Isolation (G-8 / H2-1): a contributor that raises used to fail the whole
    endpoint with a 500, so one store exception in a fleet-level family took
    the process-local gauges -- processor health, dispatcher liveness -- down
    with it, and the Pod that most needed watching became a blind spot. Each
    contributor now renders under its own ``try``; a failure is logged, counted
    per contributor and the remaining families still reach the scraper.

    Role gating (A-6): a contributor registered ``fleet_level=True`` reads a
    cluster-level fact out of the store (workflow census, orphan inspection,
    remote-command backlog). Every replica publishes the same number and the
    alerts already ``max by`` over them, so the ingress and spool-worker roles
    -- ``background_services_enabled=False`` -- skip them rather than repeat
    the worker tier's table scans on the tier that serves collectors.
    """

    def __init__(self) -> None:
        self._contributors: dict[str, MetricContributor] = {}
        self._fleet_level: set[str] = set()
        self._plugins_loaded = False
        self._errors: Counter[str] = Counter()
        self._reported: set[str] = set()
        self._lock = Lock()

    def register(
        self,
        name: str,
        contributor: MetricContributor,
        *,
        fleet_level: bool = False,
    ) -> None:
        if not name:
            raise ValueError("metric contributor name is required")
        if name in self._contributors:
            raise RuntimeError(f"duplicate metric contributor: {name}")
        self._contributors[name] = contributor
        if fleet_level:
            self._fleet_level.add(name)

    def _load_plugins(self) -> None:
        if self._plugins_loaded:
            return
        for name, entry_point in discover_plugins(
            PluginGroup.METRIC_CONTRIBUTORS
        ).items():
            self.register(name, entry_point.load())
        self._plugins_loaded = True

    def render(self, runtime: AppRuntime) -> list[str]:
        self._load_plugins()
        fleet_level_enabled = getattr(runtime, "background_services_enabled", True)
        lines: list[str] = []
        for name, contributor in self._contributors.items():
            if not fleet_level_enabled and name in self._fleet_level:
                continue
            try:
                lines.extend(contributor(runtime))
            except Exception as exc:
                self._record_failure(name, exc)
        lines.extend(self._error_lines())
        return lines

    def _record_failure(self, name: str, exc: Exception) -> None:
        with self._lock:
            self._errors[name] += 1
            first = name not in self._reported
            self._reported.add(name)
        # The traceback once per process per contributor; afterwards one line
        # per scrape, or a failing Aurora would fill the log every 15 s.
        if first:
            LOGGER.exception("metric contributor %s failed; family skipped", name)
        else:
            LOGGER.warning("metric contributor %s failed again: %s", name, exc)

    def _error_lines(self) -> list[str]:
        with self._lock:
            errors = dict(self._errors)
        lines = [
            f"# HELP {ERRORS_METRIC} Scrapes on which this contributor raised and "
            "its families were skipped; the other families still rendered "
            "(control-plane review 2026-09-08, G-8).",
            f"# TYPE {ERRORS_METRIC} counter",
        ]
        for name in self._contributors:
            lines.append(
                f'{ERRORS_METRIC}{{contributor="{_escape_label(name)}"}} '
                f"{errors.get(name, 0)}"
            )
        return lines

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._contributors)

    @property
    def fleet_level_names(self) -> tuple[str, ...]:
        return tuple(name for name in self._contributors if name in self._fleet_level)

    def error_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._errors)
