from __future__ import annotations

import pytest

from gpu_fault.admin import command_log, diagnostics
from gpu_fault.admin.bootstrap_common import BootstrapError


def test_empty_and_sensitive_command_labels_do_not_expose_arguments():
    assert diagnostics.diagnostic_command([]) == "<no command>"
    assert (
        diagnostics.diagnostic_command(["example", "private-input"], sensitive=True)
        == "example"
    )
    assert command_log.is_own_driver([]) is False


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (["python3", "path/example.py"], "example.py"),
        (["python3", "-m"], "python3"),
        (["make", "-j4", "EXAMPLE=value"], "make"),
    ],
)
def test_driver_labels_remain_useful_with_partial_wrapper_arguments(
    arguments, expected
):
    assert command_log.driver_name(arguments) == expected


def test_wrapper_failure_reports_context_notes_before_its_own_reason(capsys):
    error = BootstrapError("example refusal")
    error.add_note("example task failed")
    assert command_log.report_failure("example", error) == 2
    assert capsys.readouterr().err.splitlines() == [
        "example: example task failed",
        "example: example refusal",
    ]


@pytest.mark.parametrize("newline", [False, True])
def test_oversized_diagnostic_input_never_emits_an_untrusted_tail(monkeypatch, newline):
    output = []
    monkeypatch.setattr(
        diagnostics,
        "write_diagnostic",
        lambda text, **options: output.append((text, options)) or True,
    )
    reader = diagnostics.DriverDiagnostics()
    reader.maximum = 16
    reader.feed("x" * 17 + ("\n" if newline else ""))
    reader.feed("\nexample\n")
    reader.finish()
    assert any("safe size limit" in text for text, _options in output), (
        "oversized diagnostic input lost its truncation warning"
    )
    assert all("xxxxx" not in text for text, _options in output), (
        "oversized diagnostic input exposed an untrusted tail"
    )


@pytest.mark.parametrize("accepted", [False, True])
def test_structured_progress_falls_back_to_final_output_when_console_refuses(
    monkeypatch, accepted
):
    output = []

    def write(text, **options):
        output.append((text, options.get("final", False)))
        return accepted

    monkeypatch.setattr(diagnostics, "write_diagnostic", write)
    reader = diagnostics.DriverDiagnostics()
    line = "bootstrap task=example start"
    reader.feed(line + "\n")
    reader.finish()
    assert output == [(line + "\n", False)] + (
        [] if accepted else [(line + "\n", True)]
    )


def test_partial_diagnostic_line_is_flushed_once_at_completion(monkeypatch):
    output = []
    monkeypatch.setattr(
        diagnostics,
        "write_diagnostic",
        lambda text, **options: output.append((text, options)) or True,
    )
    reader = diagnostics.DriverDiagnostics()
    reader.feed("example partial")
    reader.finish()
    reader.finish()
    assert output == [("example partial\n", {"final": True})]
