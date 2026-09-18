from __future__ import annotations

import json

import pytest

from gpu_fault.admin.diagnostics import diagnostic_command, diagnostic_text


@pytest.mark.parametrize(
    "text",
    [
        "Forbidden: token=sentinel-credential",
        'Invalid value: {"password": "sentinel-credential"}',
        "connection failed: postgresql://operator:sentinel-credential@db.test/store",
        "Authorization: Bearer sentinel-credential",
        "\n".join(
            (
                f"{'-' * 5}BEGIN PRIVATE KEY{'-' * 5}",
                "sentinel-credential",
                f"{'-' * 5}END PRIVATE KEY{'-' * 5}",
            )
        ),
    ],
)
def test_diagnostics_keep_the_failure_but_remove_credentials(text):
    output = diagnostic_text(text)
    assert "sentinel-credential" not in output
    assert "redacted" in output


def test_json_diagnostics_redact_secret_data_and_sensitive_environment_values():
    output = diagnostic_text(
        json.dumps(
            {
                "kind": "Secret",
                "reason": "Forbidden",
                "data": {"arbitrary": "sentinel-data"},
                "env": [{"name": "EXECUTION_TOKEN", "value": "sentinel-env"}],
            }
        )
    )
    assert "Forbidden" in output
    assert "sentinel-data" not in output
    assert "sentinel-env" not in output


def test_sensitive_failures_preserve_only_known_error_codes():
    output = diagnostic_text(
        "ResourceNotFoundException: sentinel-unlabelled-value", sensitive=True
    )
    assert "ResourceNotFoundException" in output
    assert "sentinel-unlabelled-value" not in output


def test_command_arguments_and_json_patches_are_redacted():
    shown = diagnostic_command(
        [
            "aws",
            "example",
            "--token",
            "sentinel-token",
            "--cli-input-json",
            '{"secretString":"sentinel-body"}',
        ]
    )
    assert "sentinel-token" not in shown
    assert "sentinel-body" not in shown
    assert "aws example" in shown


def test_diagnostic_output_is_bounded_and_keeps_the_final_reason():
    output = diagnostic_text("x" * 20000 + "\nForbidden: admission denied", limit=256)
    assert len(output) < 300
    assert output.endswith("Forbidden: admission denied"), (
        "truncation lost the final failure reason"
    )
