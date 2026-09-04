"""The probe registry: names resolve, programs are sound, and the log is short.

The probe programs used to be triple-quoted literals inside the engine modules,
where a typo was caught by nothing at all until it failed inside a Pod
mid-release. Moving them to ``probes/*.py`` fixed that for the *bodies* -- ruff
and mypy see them now -- but it moved the failure for the *names*: a bad
``probe_source("...")`` argument is no longer a ``NameError`` at import, it is a
``FileNotFoundError`` on first use. These cases put that back at test time.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
REGIONAL = ROOT / "deploy/control-plane/regional"
PROBE_DIR = REGIONAL / "probes"

PROBES = lazy_script_module(REGIONAL / "regional_release_probes.py")
RUNNER = lazy_script_module(REGIONAL / "rollout_regional_release.py")

# The one probe that runs on the deploy host rather than in a Pod, so it is the
# only one allowed the deploy host's dependencies. Adding a name here means
# claiming the probe is never executed by `kubectl exec`; `probes/README.md`
# says so and the probe's own docstring has to say so too.
DEPLOY_HOST_PROBES = {"critical_amp_alerts"}
DEPLOY_HOST_MODULES = {"boto3", "botocore"}

# The probes that take parameters in argv. `store_io_rejection_series` does
# because the exec that runs it has no `-i`; `registry_client` does because its
# stdin is the request body and the method and path cannot travel in it. Every
# other probe reads stdin, the environment, or nothing.
ARGV_PROBES = {"store_io_rejection_series", "registry_client"}

# The one probe shipped in a Pod's `command` rather than through `kubectl exec`:
# it asks whether a Pod *in a GPU cluster* can reach and authenticate to the
# control plane, which nothing running on the administrator's host can answer.
POD_COMMAND_PROBES = {"gpu_endpoint_gate"}


def probe_names() -> list[str]:
    return sorted(path.stem for path in PROBE_DIR.glob("*.py"))


def call_sites() -> dict[str, list[Path]]:
    """Every ``probe_source("name")`` in the engine, by name."""

    pattern = re.compile(r'probe_source\(\s*"([a-z0-9_]+)"\s*\)')
    sites: dict[str, list[Path]] = {}
    for path in sorted(REGIONAL.glob("*.py")):
        for name in pattern.findall(path.read_text(encoding="utf-8")):
            sites.setdefault(name, []).append(path)
    return sites


def test_every_probe_name_in_the_engine_resolves() -> None:
    """A bad name fails here rather than inside a Pod, three phases into a roll."""

    sites = call_sites()
    assert sites, "no probe_source call sites found; the pattern must have drifted"
    missing = {
        name: [path.name for path in paths]
        for name, paths in sites.items()
        if not (PROBE_DIR / f"{name}.py").is_file()
    }
    assert missing == {}


def test_no_probe_is_unreferenced() -> None:
    """An orphaned probe is dead code that still ships in the release bundle."""

    assert set(probe_names()) == set(call_sites())


def test_no_engine_module_still_carries_an_inline_program() -> None:
    """The literals are what this whole directory exists to remove.

    Asked of the syntax tree rather than of the text, because the first version
    of this case matched on `= \"\"\"` and so walked straight past five more
    programs: two `r\"\"\"` literals in `regional_admin_checks`, two in
    `regional_release_online_registry`, and one written as a column of
    implicitly concatenated single-line strings in `regional_gpu_bootstrap`.
    """

    offenders: list[str] = []
    for path in sorted(REGIONAL.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            body = value.value
            # A Python program, as opposed to a message or a jsonpath: it
            # imports something, or it prints a JSON document.
            if "\nimport " in body or "\nfrom " in body or "print(json.dumps" in body:
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []


@pytest.mark.parametrize("name", probe_names())
def test_probe_parses_and_imports_only_what_it_may(name: str) -> None:
    """Nothing from the deploy tree exists inside the Pod, so it cannot be imported."""

    tree = ast.parse(PROBE_DIR.joinpath(f"{name}.py").read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    allowed = DEPLOY_HOST_MODULES if name in DEPLOY_HOST_PROBES else set()
    forbidden = {
        root
        for root in roots
        if root.startswith("regional_")
        or (
            root not in allowed and root not in ("gpu_fault",) and _is_third_party(root)
        )
    }
    assert forbidden == set()
    assert ast.get_docstring(tree), "a probe has to say what it reads and prints"


def _is_third_party(root: str) -> bool:
    import sys

    return root not in sys.stdlib_module_names


@pytest.mark.parametrize("name", probe_names())
def test_probe_uses_the_calling_convention_it_is_registered_for(name: str) -> None:
    """argv is the documented exception; a silent second one breaks a call site."""

    source = PROBE_DIR.joinpath(f"{name}.py").read_text(encoding="utf-8")
    reads_argv = "sys.argv" in source
    assert reads_argv == (name in ARGV_PROBES)


def test_a_deploy_host_probe_says_so_in_its_docstring() -> None:
    """The rule is only enforceable if the reader can see which probes it covers."""

    for name in DEPLOY_HOST_PROBES:
        tree = ast.parse(PROBE_DIR.joinpath(f"{name}.py").read_text(encoding="utf-8"))
        docstring = ast.get_docstring(tree) or ""
        assert "deploy host" in docstring


def test_a_pod_command_probe_says_so_in_its_docstring() -> None:
    """It is not exec'd into a running Pod, so the usual reading of it is wrong."""

    for name in POD_COMMAND_PROBES:
        tree = ast.parse(PROBE_DIR.joinpath(f"{name}.py").read_text(encoding="utf-8"))
        docstring = ast.get_docstring(tree) or ""
        assert "kubectl exec" in docstring


def test_a_missing_probe_is_reported_by_name() -> None:
    with pytest.raises(FileNotFoundError, match="no_such_probe"):
        PROBES.probe_source("no_such_probe")


def _echoed_line(capsys: pytest.CaptureFixture[str], arguments: list[str]) -> str:
    """The line the runner prints before it executes a command.

    Driven through `Runner.run` rather than the echo helper, because the echo is
    the only thing this behaviour exists for: a dry run prints the line and
    returns without launching anything.
    """

    RUNNER.Runner(dry_run=True).run(["/usr/bin/kubectl", *arguments])
    # Only the newline `print` added: a probe body ends in one of its own, and
    # dropping that would hide a program echoed in full instead of by name.
    return capsys.readouterr().err.removesuffix("\n")


def test_the_command_echo_names_a_probe_instead_of_pasting_its_body(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """24% of an upgrade's log lines were probe bodies, burying the kubectl trace."""

    source = PROBES.probe_source("workflow_safety")
    assert source.count("\n") > 20

    echoed = _echoed_line(capsys, ["exec", "pod-a", "--", "python3", "-c", source])

    assert echoed == ("+ kubectl exec pod-a -- python3 -c <probe:workflow_safety>")


def test_an_unrecognised_dash_c_payload_is_echoed_unchanged(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Only probes are relabelled; anything else stays readable as itself."""

    assert _echoed_line(capsys, ["-c", "print(1)"]) == "+ kubectl -c print(1)"
    assert PROBES.probe_label("print(1)") is None


def test_a_probe_body_that_is_not_after_dash_c_is_left_alone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The label is positional, so it must not rewrite an unrelated argument."""

    source = PROBES.probe_source("remote_command_stats")

    assert _echoed_line(capsys, ["-f", source]) == f"+ kubectl -f {source}"
