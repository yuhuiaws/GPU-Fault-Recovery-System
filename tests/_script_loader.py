from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


class LazyScriptModule:
    """Load a path-only repository script on first test use."""

    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = path
        self._module: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._module is not None:
            return self._module
        spec = importlib.util.spec_from_file_location(self.name, self.path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load test script module {self.path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        previous = sys.dont_write_bytecode
        script_directory = str(self.path.parent)
        sys.path.insert(0, script_directory)
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(spec.name, None)
            raise
        finally:
            sys.dont_write_bytecode = previous
            sys.path.remove(script_directory)
        self._module = module
        return module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)


def lazy_script_module(name: str, path: Path) -> LazyScriptModule:
    return LazyScriptModule(name, path)
