from __future__ import annotations

from collections.abc import Callable

from gpu_fault.app.runtime import AppRuntime
from gpu_fault.plugins import PluginGroup, discover_plugins


MetricContributor = Callable[[AppRuntime], list[str]]


class MetricContributorRegistry:
    def __init__(self) -> None:
        self._contributors: dict[str, MetricContributor] = {}
        self._plugins_loaded = False

    def register(
        self,
        name: str,
        contributor: MetricContributor,
    ) -> None:
        if not name:
            raise ValueError("metric contributor name is required")
        if name in self._contributors:
            raise RuntimeError(f"duplicate metric contributor: {name}")
        self._contributors[name] = contributor

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
        lines = []
        for contributor in self._contributors.values():
            lines.extend(contributor(runtime))
        return lines

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._contributors)
