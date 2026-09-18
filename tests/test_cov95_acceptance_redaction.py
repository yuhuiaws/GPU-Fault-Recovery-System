from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools import run_regional_acceptance as runner


@pytest.mark.parametrize("shape", ["public", "keyword"])
def test_long_public_output_redaction_finishes_within_a_local_process_bound(
    shape,
) -> None:
    source = """
import json
import sys
from tools.run_regional_acceptance import MAX_CAPTURED_OUTPUT_CHARS, redact_text
unit = "x" if sys.argv[1] == "public" else "TOKEN"
text = unit * (MAX_CAPTURED_OUTPUT_CHARS // len(unit) + 20)
result = redact_text(text)
print(json.dumps({"length": len(result), "prefix": result[:10], "suffix": result[-57:]}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", source, shape],
        cwd=Path(__file__).resolve().parents[1],
        env=runner.build_local_environment(),
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert completed.returncode == 0, "pure local redaction must complete successfully"
    result = json.loads(completed.stdout)
    marker = "\n<output truncated by regional acceptance runner>"
    assert result["length"] == runner.MAX_CAPTURED_OUTPUT_CHARS + len(marker), (
        "redaction must preserve the configured cap for large noncredential text"
    )
    assert result["prefix"] == ("x" * 10 if shape == "public" else "TOKENTOKEN"), (
        "ordinary text without an assignment must not be mistaken for a credential"
    )
    assert result["suffix"].endswith(marker), "truncated output needs a visible marker"


def test_standard_bearer_header_redacts_the_value_after_its_separator() -> None:
    assert runner.redact_text("Authorization: Bearer fixture-only-value") == (
        "Authorization: Bearer <redacted>"
    ), "normal bearer whitespace must not leave the credential in a text report"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("COUNT=25", "COUNT=25"),
        ("COUNT=TOKEN=fixture-only", "COUNT=TOKEN=<redacted>"),
        ("GPU_FAULT_TOKEN=fixture-only", "GPU_FAULT_TOKEN=<redacted>"),
        ('"API_KEY": "fixture-only"', '"API_KEY": "<redacted>"'),
        (
            "https://fixture-user:fixture-password@example.invalid",
            "https://<redacted>@example.invalid",
        ),
        ("https://example.invalid/path", "https://example.invalid/path"),
        ("Authorization:Bearer fixture-only", "Authorization:Bearer <redacted>"),
    ],
)
def test_text_redaction_keeps_public_assignments_and_removes_credential_shapes(
    text, expected
) -> None:
    assert runner.redact_text(text) == expected, (
        "redaction must preserve noncredential context"
    )
