from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import release_image
from tests.test_release_image import fake_component_builder, inspected_image

ROOT = Path(__file__).resolve().parents[1]
TAG = "registry.example:5000/runtime:build-test"
CANARY = "synthetic-registry-diagnostic-canary"


@pytest.mark.parametrize(
    "error",
    [
        "manifest unknown",
        "manifest not found",
        "name unknown",
        "no such manifest",
        "manifest unknown: manifest unknown",
        "manifest unknown: Requested image not found",
        "name unknown: repository name not known to registry",
        f"{TAG}: not found",
        f"ERROR: {TAG}: not found\n",
        f"ERROR: {TAG}: manifest unknown",
        f"no such manifest: {TAG}",
        json.dumps({"errors": [{"code": "MANIFEST_UNKNOWN"}]}),
        json.dumps({"errors": [{"code": "NAME_UNKNOWN"}]}),
    ],
)
def test_only_identifiable_registry_absence_is_a_cache_miss(error):
    def run(command, **_kwargs):
        return subprocess.CompletedProcess(command, 1, "", error)

    assert (
        release_image.inspect_registry_image(
            ROOT, tag=TAG, platform="linux/amd64", expected_labels={}, runner=run
        )
        is None
    )


@pytest.mark.parametrize("returncode", [-9, 2, 125, 126, 127])
def test_launcher_and_process_failures_cannot_report_registry_absence(returncode):
    def run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command, returncode, "", f"ERROR: {TAG}: not found"
        )

    with pytest.raises(release_image.ReleaseImageError, match="inspection failed"):
        release_image.inspect_registry_image(
            ROOT, tag=TAG, platform="linux/amd64", expected_labels={}, runner=run
        )


@pytest.mark.parametrize(
    ("error", "hint"),
    [
        (
            'ERROR: error getting credentials - err: exec: "docker-credential-ecr-login": '
            'executable file not found in $PATH, out: ""',
            "credential helper installation",
        ),
        ("ERROR: credentials not found", "credential helper"),
        ("ERROR: unauthorized: manifest unknown", "registry login"),
        ("ERROR: denied: repository not found", "read permissions"),
        ("ERROR: failed to read config: file not found", "Docker Buildx"),
        ("ERROR: failed to parse response: manifest unknown", "Docker Buildx"),
        ("ERROR: no such host: registry.example", "connectivity"),
        ("ERROR: TLS certificate file not found", "TLS trust"),
        ("ERROR: connection timed out", "request time limits"),
        ("ERROR: registry.example/other:build-test: not found", "Docker Buildx"),
        ("ERROR: not found", "Docker Buildx"),
        ("manifest unknown\ncredential helper failed", "credential helper"),
        (f"ERROR: {TAG}: not found\nunauthorized", "registry login"),
        (
            json.dumps(
                {"errors": [{"code": "MANIFEST_UNKNOWN"}, {"code": "UNAUTHORIZED"}]}
            ),
            "registry login",
        ),
        (json.dumps({"errors": []}), "Docker Buildx"),
        (json.dumps({"errors": [{"message": "manifest unknown"}]}), "Docker Buildx"),
        ("", "Docker Buildx"),
    ],
)
def test_registry_errors_fail_closed_before_build_and_keep_actionable_hints(
    monkeypatch, error, hint
):
    commands = []
    monkeypatch.setattr(release_image, "build_component", fake_component_builder)

    def run(command, **_kwargs):
        commands.append(command)
        assert command[:3] == ["docker", "buildx", "imagetools"], (
            "a registry tool error triggered an image build"
        )
        return subprocess.CompletedProcess(command, 1, "", error)

    with pytest.raises(release_image.ReleaseImageError, match=hint):
        release_image.build_runtime_image(
            ROOT, repository="registry.example/runtime", push=True, runner=run
        )
    assert len(commands) == 1


@pytest.mark.parametrize(
    "error",
    [
        f"error getting credentials: helper output {CANARY}",
        f"unauthorized: token={CANARY} https://user:{CANARY}@registry.example/",
        json.dumps(
            {"errors": [{"code": "UNAUTHORIZED", "detail": {"password": CANARY}}]}
        ),
        f"unknown failure: {CANARY}",
    ],
)
def test_registry_error_output_never_echoes_credentials(error):
    def run(command, **_kwargs):
        return subprocess.CompletedProcess(command, 1, CANARY, error)

    with pytest.raises(release_image.ReleaseImageError) as caught:
        release_image.inspect_registry_image(
            ROOT, tag=TAG, platform="linux/amd64", expected_labels={}, runner=run
        )
    assert CANARY not in str(caught.value)
    assert "status 1" in str(caught.value)
    assert "check " in str(caught.value)


@pytest.mark.parametrize("stdout", ["", "{", "not-json", "null", "[]", '"text"'])
def test_successful_inspection_must_return_a_json_object(stdout):
    def run(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, stdout, "")

    with pytest.raises(release_image.ReleaseImageError, match="JSON"):
        release_image.inspect_registry_image(
            ROOT, tag=TAG, platform="linux/amd64", expected_labels={}, runner=run
        )


@pytest.mark.parametrize(
    ("failure", "hint"),
    [
        (FileNotFoundError(CANARY), "could not execute Docker"),
        (PermissionError(CANARY), "executable permissions"),
        (TimeoutError(CANARY), "timed out"),
        (
            subprocess.TimeoutExpired(
                ["docker", CANARY], 1, output=CANARY, stderr=CANARY
            ),
            "timed out",
        ),
        (UnicodeError(CANARY), "invalid text"),
    ],
)
def test_inspection_transport_exceptions_are_sanitized(failure, hint):
    def run(_command, **_kwargs):
        raise failure

    with pytest.raises(release_image.ReleaseImageError, match=hint) as caught:
        release_image.inspect_registry_image(
            ROOT, tag=TAG, platform="linux/amd64", expected_labels={}, runner=run
        )
    assert CANARY not in str(caught.value)
    assert caught.value.__suppress_context__, "raw tool exception was exposed"


def test_platform_mismatch_does_not_echo_untrusted_config_values():
    def run(command, **_kwargs):
        value = json.loads(inspected_image({}))
        value["image"]["linux/amd64"]["architecture"] = CANARY
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")

    with pytest.raises(release_image.ReleaseImageError, match="platform") as caught:
        release_image.inspect_registry_image(
            ROOT, tag=TAG, platform="linux/amd64", expected_labels={}, runner=run
        )
    assert CANARY not in str(caught.value)
