from __future__ import annotations

import subprocess
import sys

import pytest

from scripts.e2e.regional.late_ownership_probe_bundle import probe_program
from tools.run_fault_test_cases import build_isolated_environment


@pytest.mark.parametrize("role", ["reset-interval", "reset-interval-detached"])
def test_minimal_physical_probe_bundle_imports_without_checkout_script_fallback(
    tmp_path, role
):
    program, digest = probe_program(role)
    wrapper = """
import io, json, sys
program = sys.stdin.read()
sys.stdin = io.StringIO('\\n')
def deny_live(event, arguments):
    if event in {'subprocess.Popen', 'socket.connect', 'socket.bind'}:
        raise RuntimeError('invalid input must fail before host or network operations')
sys.addaudithook(deny_live)
try:
    exec(compile(program, '<measured-private-bundle>', 'exec'), {})
except json.JSONDecodeError:
    print('PRIVATE_BUNDLE_IMPORTED_AND_REFUSED_EMPTY_ARM')
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", wrapper],
        input=program,
        cwd=tmp_path,
        env=build_isolated_environment(),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert len(digest) == 64
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "PRIVATE_BUNDLE_IMPORTED_AND_REFUSED_EMPTY_ARM"
