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


def test_shell_scripts_are_ratcheted_with_python(tmp_path: Path) -> None:
    """S18: ``*.sh`` under the scanned roots gets file and function sizes.

    The fixture function body has ``${var}`` expansions, an awk program, a
    ``# }`` comment, a heredoc whose body is a bare ``}`` and a regex ``\\{``,
    and a ``${value//\\/\\\\}`` whose closing brace is real. None of these may
    move the brace depth the wrong way; the function ends at its own
    top-level ``}`` and the file line count is the raw one.
    """
    root = tmp_path / "repo"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    body = [
        '    local value="${1:-}"  # } not a brace',
        '    awk \'BEGIN { print "x" }\' <<< "${value}"',
        "    cat <<EOF",
        "}",
        '{ "regex": "(\\\\{| )" }',
        "EOF",
        '    value="${value//\\\\/\\\\\\\\}"',
        '    if [[ -n "${value}" ]]; then',
        "        printf '%s\\n' \"${value}\"",
        "    fi",
    ]
    filler = ['    : "${value}"'] * (
        architecture.DEFAULT_FUNCTION_LIMIT + 1 - len(body) - 2
    )
    long_function = ["long_work() {", *body, *filler, "}"]
    short_function = ["short_work() {", *body, "}"]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        *long_function,
        *short_function,
        'long_work "$@"',
    ]
    padding = architecture.DEFAULT_FILE_LIMIT + 1 - len(lines)
    lines.extend(["# padding"] * padding)
    (scripts / "big.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (scripts / "small.sh").write_text(
        "#!/usr/bin/env bash\n" + "\n".join(short_function) + "\nshort_work\n",
        encoding="utf-8",
    )
    (scripts / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")

    current, cycles = architecture.collect_architecture(
        source_roots=(scripts,), root=root
    )

    assert cycles == []
    assert current["files"] == {
        "scripts/big.sh": architecture.DEFAULT_FILE_LIMIT + 1,
        "scripts/helper.py": 1,
        "scripts/small.sh": len(short_function) + 2,
    }
    assert current["functions"] == {
        "scripts/big.sh:long_work": len(long_function),
        "scripts/big.sh:short_work": len(short_function),
        "scripts/small.sh:short_work": len(short_function),
    }
    assert len(long_function) == architecture.DEFAULT_FUNCTION_LIMIT + 1
    baseline = architecture.generated_baseline(current, cycles)
    assert baseline["files"] == {"scripts/big.sh": architecture.DEFAULT_FILE_LIMIT + 1}
    assert baseline["functions"] == {
        "scripts/big.sh:long_work": architecture.DEFAULT_FUNCTION_LIMIT + 1
    }
    assert architecture.baseline_failures(baseline, current, cycles) == []


def test_shell_function_scanner_reads_this_repository_style() -> None:
    """The real deploy entry point: known function heads resolve to sizes."""
    text = (ROOT / "deploy/hyperpod/deploy.sh").read_text(encoding="utf-8")

    sizes = architecture.shell_functions(text)

    assert {"usage", "deploy_control_plane", "build_artifacts"} <= set(sizes)
    assert all(size > 0 for size in sizes.values()), (
        "every shell function must have a positive size"
    )
    assert sizes["deploy_control_plane"] > architecture.DEFAULT_FUNCTION_LIMIT


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
