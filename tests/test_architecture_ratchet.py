from __future__ import annotations

import ast
from pathlib import Path

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check-python-architecture.py"
architecture = lazy_script_module(SCRIPT)


def test_generated_architecture_baseline_is_exact_and_minimal() -> None:
    current = {
        "files": {"large.py": 1501, "small.py": 20},
        "functions": {"large.py:work": 201},
        "classes": {"large.py:Worker": 800},
    }

    baseline = architecture.generated_baseline(current, [])

    assert baseline == {
        "classes": {},
        "cycles": [],
        "files": {"large.py": 1501},
        "functions": {"large.py:work": 201},
    }
    assert architecture.baseline_failures(baseline, current, []) == []


def test_architecture_ratchet_rejects_slack_and_stale_entries() -> None:
    current = {
        "files": {"large.py": 1501},
        "functions": {"large.py:work": 201},
        "classes": {},
    }
    baseline = {
        "files": {"large.py": 1600, "deleted.py": 1700},
        "functions": {},
        "classes": {},
        "cycles": [["old.module", "old.peer"]],
    }

    failures = architecture.baseline_failures(
        baseline, current, [["new.module", "new.peer"]]
    )
    message = "\n".join(failures)

    assert "baseline has slack" in message
    assert "stale file baseline entry: deleted.py" in message
    assert "default limit 200" in message
    assert "new import cycle" in message
    assert "stale grandfathered import cycle" in message


def test_architecture_ratchet_rejects_growth() -> None:
    current = {"files": {"large.py": 1510}, "functions": {}, "classes": {}}
    baseline = {
        "files": {"large.py": 1501},
        "functions": {},
        "classes": {},
        "cycles": [],
    }

    failures = architecture.baseline_failures(baseline, current, [])

    assert failures == ["file large.py grew from 1501 to 1510 lines"]


def test_runtime_import_graph_ignores_type_checking_imports() -> None:
    tree = ast.parse(
        "from typing import TYPE_CHECKING\n"
        "import runtime_dependency\n"
        "if TYPE_CHECKING:\n"
        "    import typing_only_dependency\n"
    )
    collector = architecture.RuntimeImportCollector()

    collector.visit(tree)

    assert "runtime_dependency" in collector.targets
    assert "typing_only_dependency" not in collector.targets
