from __future__ import annotations

import copy
import json
import re
import subprocess

import pytest

from gpu_fault.admin import execution, release_artifacts
from gpu_fault.admin.bootstrap_common import BootstrapError

REGION = "us-east-1"
REGISTRY = "123456789012"
REPOSITORY_NAME = "gpu-fault/runtime-a"
REPOSITORY = f"{REGISTRY}.dkr.ecr.{REGION}.amazonaws.com/{REPOSITORY_NAME}"
DIGESTS = ["sha256:" + character * 64 for character in "abc"]
REFERENCES = [f"{REPOSITORY}@{digest}" for digest in DIGESTS]


def inventory(digests=DIGESTS):
    return {
        "imageDetails": [
            {
                "registryId": REGISTRY,
                "repositoryName": REPOSITORY_NAME,
                "imageDigest": digest,
            }
            for digest in digests
        ]
    }


@pytest.fixture
def aws(monkeypatch):
    commands = []
    response = {"value": inventory(), "returncode": 0, "stderr": ""}

    def run(command, **kwargs):
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            response["returncode"],
            response.get("stdout", json.dumps(response["value"])),
            response["stderr"],
        )

    monkeypatch.setattr(release_artifacts, "bounded_command", run)
    return commands, response


def test_ecr_batch_deduplicates_and_validates_all_digests_in_any_order(aws):
    commands, response = aws
    response["value"] = inventory(reversed(DIGESTS))
    assert release_artifacts.runtime_images_exist(
        region=REGION, references=[*REFERENCES, REFERENCES[0]]
    ), "complete ECR inventory was rejected"
    assert len(commands) == 1
    command, options = commands[0]
    assert command[:3] == ["aws", "ecr", "describe-images"]
    assert command[command.index("--region") + 1] == REGION
    assert command[command.index("--registry-id") + 1] == REGISTRY
    assert command[command.index("--repository-name") + 1] == REPOSITORY_NAME
    assert command[
        command.index("--image-ids") + 1 : command.index("--no-paginate")
    ] == [f"imageDigest={digest}" for digest in DIGESTS]
    assert options == {}


@pytest.mark.parametrize(
    "response",
    [
        None,
        [],
        {},
        {"imageDetails": None},
        {"imageDetails": {}},
        {"imageDetails": []},
        {"imageDetails": [None]},
        {"imageDetails": [{}]},
        inventory(DIGESTS[:2]),
        inventory([*DIGESTS, DIGESTS[0]]),
        inventory([*DIGESTS, "sha256:" + "d" * 64]),
        {**inventory(), "nextToken": "more-results"},
    ],
)
def test_malformed_or_incomplete_inventory_is_an_error_not_absence(aws, response):
    commands, result = aws
    result["value"] = response
    with pytest.raises(BootstrapError, match="inventory"):
        release_artifacts.runtime_images_exist(region=REGION, references=REFERENCES)
    assert len(commands) == 1


@pytest.mark.parametrize("stdout", ["", "{", "not-json"])
def test_unparseable_inventory_is_not_treated_as_reusable(aws, stdout):
    _commands, result = aws
    result["stdout"] = stdout
    with pytest.raises(BootstrapError, match="invalid JSON"):
        release_artifacts.runtime_images_exist(region=REGION, references=REFERENCES)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("registryId", "111122223333"),
        ("registryId", None),
        ("repositoryName", "gpu-fault/other"),
        ("repositoryName", None),
        ("imageDigest", "sha256:" + "d" * 64),
        ("imageDigest", []),
        ("imageDigest", None),
    ],
)
def test_response_identity_drift_fails_closed(aws, field, value):
    _commands, result = aws
    result["value"]["imageDetails"][1][field] = value
    with pytest.raises(BootstrapError, match="identity"):
        release_artifacts.runtime_images_exist(region=REGION, references=REFERENCES)


@pytest.mark.parametrize(
    "code", ["ImageNotFoundException", "RepositoryNotFoundException"]
)
@pytest.mark.parametrize("cli_message", [False, True])
def test_only_explicit_ecr_absence_is_a_cache_miss(aws, code, cli_message):
    _commands, result = aws
    result["returncode"] = 254
    result["stderr"] = (
        f"An error occurred ({code}) when calling the DescribeImages operation: missing"
        if cli_message
        else code
    )
    assert not release_artifacts.runtime_images_exist(
        region=REGION, references=REFERENCES
    ), "confirmed missing ECR image was treated as present"


@pytest.mark.parametrize(
    "error",
    [
        "AccessDeniedException",
        "ExpiredTokenException",
        "ThrottlingException",
        "connection timed out",
        "",
        "An error occurred (AccessDeniedException) when calling the DescribeImages "
        "operation: policy denies ImageNotFoundException handling",
        "credential helper executable not found: ImageNotFoundException",
        "An error occurred (ImageNotFoundException) when calling the Other "
        "operation: missing",
        "An error occurred (ImageNotFoundException) when calling the DescribeImages "
        "operation: missing\ncredential helper failed",
    ],
)
def test_registry_errors_never_turn_into_absence(aws, error):
    _commands, result = aws
    result["returncode"] = 254
    result["stderr"] = error
    with pytest.raises(BootstrapError, match="cannot verify"):
        release_artifacts.runtime_images_exist(region=REGION, references=REFERENCES)


@pytest.mark.parametrize(
    "references",
    [
        [],
        [REPOSITORY + ":latest"],
        [REPOSITORY + "@sha256:" + "a" * 63],
        [REPOSITORY + "@sha256:" + "z" * 64],
        ["registry.example/runtime@" + DIGESTS[0]],
        [REFERENCES[0].replace(REGION, "us-west-2")],
        [REFERENCES[0].replace(".amazonaws.com/", ".amazonaws.com.cn/")],
        [REFERENCES[0], REFERENCES[1].replace(REGISTRY, "111122223333")],
        [REFERENCES[0], REFERENCES[1].replace(REPOSITORY_NAME, "gpu-fault/other")],
        [f"{REPOSITORY}@sha256:{index:064x}" for index in range(101)],
    ],
)
def test_invalid_or_mixed_repository_batch_stops_before_aws(aws, references):
    commands, _result = aws
    with pytest.raises(BootstrapError):
        release_artifacts.runtime_images_exist(region=REGION, references=references)
    assert not commands, "refused image identity unexpectedly reached AWS"


@pytest.fixture
def signed_release(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "current-release.json").write_text(
        json.dumps({"schema_version": 4, "deployable": True, "staging_only": False})
    )
    (dist / "current-attestation.json").write_text(
        json.dumps({"source": {"dirty": False, "git_commit": "a" * 40}})
    )
    (dist / "current-attestation.bundle.json").write_text("{}")
    release = {
        "release_id": "release-a",
        "images": dict(zip(("runtime", "executor", "node_dependencies"), REFERENCES)),
    }
    calls = []
    monkeypatch.setattr(release_artifacts, "_git_output", lambda *_args: "a" * 40)
    monkeypatch.setattr(
        release_artifacts,
        "verify_prebuilt_release",
        lambda *_args, **_kwargs: calls.append("verify"),
    )

    def load(*_args, **_kwargs):
        assert calls == ["verify"]
        calls.append("load")
        return copy.deepcopy(release)

    monkeypatch.setattr(release_artifacts, "load_prebuilt_release", load)

    def reuse():
        return release_artifacts.load_reusable_signed_release(
            object(),
            repository_root=tmp_path,
            state_dir=tmp_path,
            region=REGION,
            runtime_repository=REPOSITORY,
            runtime_profile="profile-a",
            staging_only=False,
            impact_base="origin/main",
        )

    return release, calls, reuse


def test_real_split_release_reuse_verifies_then_makes_one_batched_read(
    signed_release, aws, monkeypatch
):
    release, calls, reuse = signed_release
    commands, _response = aws
    monkeypatch.setattr(
        release_artifacts,
        "runtime_image_exists",
        lambda **_kwargs: pytest.fail("split reuse called the sequential helper"),
    )
    assert reuse() == {**release, "release_reused": True}
    assert calls == ["verify", "load"]
    assert len(commands) == 1


def test_split_release_missing_digest_is_not_reused(signed_release, aws):
    _release, _calls, reuse = signed_release
    _commands, response = aws
    response["returncode"] = 254
    response["stderr"] = "ImageNotFoundException"
    assert reuse() is None


def test_split_release_checks_all_repositories_before_querying_ecr(signed_release, aws):
    release, _calls, reuse = signed_release
    commands, _response = aws
    release["images"]["node_dependencies"] = REFERENCES[2].replace(
        REPOSITORY_NAME, "gpu-fault/other"
    )
    assert reuse() is None
    assert not commands, "refused image identity unexpectedly reached AWS"


def test_invalid_signature_stops_before_any_registry_read(
    signed_release, aws, monkeypatch
):
    _release, _calls, reuse = signed_release
    commands, _response = aws

    def invalid(*_args, **_kwargs):
        raise BootstrapError("signature is invalid")

    monkeypatch.setattr(release_artifacts, "verify_prebuilt_release", invalid)
    with pytest.raises(BootstrapError, match="signature"):
        reuse()
    assert not commands, "refused image identity unexpectedly reached AWS"


def test_shared_image_release_keeps_the_single_image_helper_contract(
    signed_release, aws, monkeypatch
):
    release, _calls, reuse = signed_release
    commands, _response = aws
    release["images"] = {"runtime": REFERENCES[0]}
    checked = []
    monkeypatch.setattr(
        release_artifacts,
        "runtime_image_exists",
        lambda **kwargs: checked.append(kwargs) or True,
    )
    assert reuse() == {**release, "release_reused": True}
    assert checked == [{"region": REGION, "reference": REFERENCES[0]}]
    assert not commands, "refused image identity unexpectedly reached AWS"


@pytest.fixture(params=[DIGESTS, DIGESTS[:1]], ids=["batch", "single"])
def requested_digests(request):
    return request.param


@pytest.fixture
def check_images(requested_digests):
    def check():
        if len(requested_digests) == 1:
            return release_artifacts.runtime_image_exists(
                region=REGION, reference=REFERENCES[0]
            )
        return release_artifacts.runtime_images_exist(
            region=REGION, references=REFERENCES
        )

    return check


@pytest.mark.parametrize(
    ("returncode", "stderr", "expected"),
    [
        (0, "", True),
        (254, "ImageNotFoundException", False),
        (254, "RepositoryNotFoundException", False),
    ],
)
def test_both_ecr_helpers_preserve_success_and_absence_semantics(
    aws, check_images, requested_digests, returncode, stderr, expected
):
    commands, result = aws
    result.update(
        value=inventory(requested_digests), returncode=returncode, stderr=stderr
    )
    assert check_images() is expected
    assert len(commands) == 1


@pytest.mark.parametrize("kind", ["text", "json"])
def test_both_ecr_helpers_redact_error_output(aws, check_images, kind):
    _commands, result = aws
    canary = "synthetic-ecr-diagnostic-canary"
    error = (
        f"AccessDeniedException: token={canary} endpoint=https://user:{canary}@example.com"
        if kind == "text"
        else json.dumps(
            {
                "code": "AccessDeniedException",
                "credentials": {"password": canary},
                "message": f"Bearer {canary}",
            }
        )
    )
    result.update(returncode=254, stderr=error)
    with pytest.raises(BootstrapError) as caught:
        check_images()
    message = str(caught.value)
    assert canary not in message
    assert "AccessDeniedException" in message
    assert "<redacted" in message


@pytest.mark.parametrize("kind", ["process", "deadline"])
def test_timeouts_never_become_reuse_or_absence(check_images, monkeypatch, kind):
    canary = "synthetic-timeout-diagnostic-canary"

    def expired(command):
        if kind == "process":
            raise subprocess.TimeoutExpired(command, 2, output=canary, stderr=canary)
        raise execution.DeploymentDeadlineExceeded(canary)

    monkeypatch.setattr(release_artifacts, "bounded_command", expired)
    with pytest.raises(BootstrapError, match="deployment time budget") as caught:
        check_images()
    assert canary not in str(caught.value)
    assert caught.value.__suppress_context__, (
        "timeout exposed original subprocess arguments"
    )


def test_ecr_reads_have_a_default_bound_without_a_deployment_scope(
    check_images, requested_digests, monkeypatch
):
    monkeypatch.setattr(execution.time, "monotonic", lambda: 100.0)
    commands = []

    def owned(command, **kwargs):
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(
            command, 0, json.dumps(inventory(requested_digests)), ""
        )

    monkeypatch.setattr(execution, "run_owned_command", owned)
    assert check_images(), "valid bounded ECR lookup was rejected"
    assert len(commands) == 1
    assert commands[0][1]["timeout"] == 120
    assert commands[0][1]["expires_at"] == 220
    assert commands[0][1]["capture"] is True


def test_ecr_reads_honor_the_shorter_task_deadline(
    check_images, requested_digests, monkeypatch
):
    commands = []
    clock = [100.0]
    monkeypatch.setattr(execution.time, "monotonic", lambda: clock[0])

    def owned(command, **kwargs):
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(
            command, 0, json.dumps(inventory(requested_digests)), ""
        )

    monkeypatch.setattr(execution, "run_owned_command", owned)
    with execution.deployment_deadline("root", 60):
        with execution.deadline_scope("release image task", 9):
            clock[0] = 102.0
            assert check_images(), "valid bounded ECR lookup was rejected"
    assert len(commands) == 1
    options = commands[0][1]
    assert options["timeout"] == 7
    assert options["environment"][execution.DEADLINE_ENV] == "109.0"
    assert options["environment"][execution.DEADLINE_LABEL_ENV] == "release image task"


def test_inherited_root_and_hard_deadlines_bound_direct_ecr_reads(
    check_images, requested_digests, monkeypatch
):
    commands = []
    monkeypatch.setattr(execution.time, "monotonic", lambda: 100.0)
    monkeypatch.setenv(execution.DEADLINE_ENV, "107.0")
    monkeypatch.setenv(execution.HARD_DEADLINE_ENV, "105.0")

    def owned(command, **kwargs):
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(
            command, 0, json.dumps(inventory(requested_digests)), ""
        )

    monkeypatch.setattr(execution, "run_owned_command", owned)
    assert check_images(), "valid bounded ECR lookup was rejected"
    assert len(commands) == 1
    assert commands[0][1]["timeout"] == 5


def test_expired_task_deadline_prevents_any_ecr_process(check_images, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(execution.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        execution,
        "run_owned_command",
        lambda *_args, **_kwargs: pytest.fail("expired ECR read started a process"),
    )
    with execution.deadline_scope("release image task", 1):
        clock[0] = 102.0
        with pytest.raises(BootstrapError, match="deployment time budget"):
            check_images()


def test_single_image_query_binds_the_signed_account_region_repository_and_digest(aws):
    commands, result = aws
    result["value"] = inventory(DIGESTS[:1])
    assert release_artifacts.runtime_image_exists(
        region=REGION, reference=REFERENCES[0]
    ), "valid single-image inventory was rejected"
    assert len(commands) == 1
    command, _options = commands[0]
    assert command[command.index("--registry-id") + 1] == REGISTRY
    assert command[command.index("--region") + 1] == REGION
    assert command[command.index("--repository-name") + 1] == REPOSITORY_NAME
    assert command[
        command.index("--image-ids") + 1 : command.index("--no-paginate")
    ] == [f"imageDigest={DIGESTS[0]}"]


@pytest.mark.parametrize(
    "response",
    [
        None,
        [],
        {},
        {"imageDetails": []},
        {"imageDetails": [None]},
        inventory(DIGESTS[1:2]),
        inventory([DIGESTS[0], DIGESTS[0]]),
        inventory(DIGESTS[:2]),
        {**inventory(DIGESTS[:1]), "nextToken": "more"},
        {**inventory(DIGESTS[:1]), "nextToken": False},
        {**inventory(DIGESTS[:1]), "nextToken": []},
    ],
)
def test_single_image_inventory_must_be_complete_and_unambiguous(aws, response):
    _commands, result = aws
    result["value"] = response
    with pytest.raises(BootstrapError, match="inventory"):
        release_artifacts.runtime_image_exists(region=REGION, reference=REFERENCES[0])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("registryId", "111122223333"),
        ("registryId", None),
        ("repositoryName", "gpu-fault/other"),
        ("repositoryName", None),
        ("imageDigest", "sha256:" + "d" * 64),
        ("imageDigest", "sha256:" + "a" * 63),
        ("imageDigest", None),
    ],
)
def test_single_image_response_identity_drift_is_not_reuse(aws, field, value):
    _commands, result = aws
    result["value"] = inventory(DIGESTS[:1])
    result["value"]["imageDetails"][0][field] = value
    with pytest.raises(BootstrapError, match="identity"):
        release_artifacts.runtime_image_exists(region=REGION, reference=REFERENCES[0])


@pytest.mark.parametrize(
    "reference",
    [
        REPOSITORY + ":latest",
        REPOSITORY + "@sha256:" + "a" * 63,
        REPOSITORY + "@sha256:" + "z" * 64,
        "registry.example/runtime@" + DIGESTS[0],
        REFERENCES[0].replace(REGION, "us-west-2"),
        REFERENCES[0].replace(".amazonaws.com/", ".amazonaws.com.cn/"),
    ],
)
def test_invalid_single_image_identity_stops_before_aws(aws, reference):
    commands, _result = aws
    with pytest.raises(BootstrapError, match="reference"):
        release_artifacts.runtime_image_exists(region=REGION, reference=reference)
    assert not commands, "invalid single-image reference reached AWS"


@pytest.mark.parametrize("stdout", ["", "{", "not-json"])
def test_single_image_success_without_valid_json_is_an_error(aws, stdout):
    _commands, result = aws
    result["stdout"] = stdout
    with pytest.raises(BootstrapError, match="invalid JSON"):
        release_artifacts.runtime_image_exists(region=REGION, reference=REFERENCES[0])


@pytest.mark.parametrize(
    "error",
    [
        "credential helper executable not found: ImageNotFoundException",
        "An error occurred (AccessDeniedException) when calling the DescribeImages "
        "operation: ImageNotFoundException is not allowed",
        "An error occurred (ImageNotFoundException) when calling the Other "
        "operation: missing",
        "An error occurred (ImageNotFoundException) when calling the DescribeImages "
        "operation: missing\ncredential helper failed",
    ],
)
def test_single_image_errors_cannot_impersonate_registry_absence(aws, error):
    _commands, result = aws
    result.update(returncode=254, stderr=error)
    with pytest.raises(BootstrapError, match="cannot verify"):
        release_artifacts.runtime_image_exists(region=REGION, reference=REFERENCES[0])


@pytest.mark.parametrize("returncode", [-9, -15, 2, 125, 126, 127])
def test_failed_ecr_process_cannot_authorize_a_cache_miss(
    aws, check_images, returncode
):
    _commands, result = aws
    result.update(
        returncode=returncode,
        stderr="An error occurred (ImageNotFoundException) when calling "
        "the DescribeImages operation: missing",
    )
    with pytest.raises(BootstrapError, match="cannot verify"):
        check_images()


@pytest.mark.parametrize("kind", [FileNotFoundError, PermissionError])
def test_ecr_tool_startup_errors_are_sanitized_and_actionable(
    check_images, monkeypatch, kind
):
    canary = "synthetic-tool-error-canary"

    def unavailable(_command):
        raise kind(canary)

    monkeypatch.setattr(release_artifacts, "bounded_command", unavailable)
    with pytest.raises(BootstrapError, match="cannot execute AWS CLI") as caught:
        check_images()
    assert canary not in str(caught.value)
    assert caught.value.__suppress_context__, "tool failure leaked its raw exception"


def test_ecr_credential_helper_output_is_withheld_even_without_sensitive_field_names(
    aws, check_images
):
    _commands, response = aws
    canary = "synthetic-helper-output-canary"
    response.update(
        returncode=255,
        stderr=f"Error when retrieving credentials from custom-process: {canary}",
    )
    with pytest.raises(BootstrapError, match="credential_process") as caught:
        check_images()
    assert canary not in str(caught.value)


def test_ecr_invalid_text_is_a_sanitized_error_not_absence(check_images, monkeypatch):
    canary = "synthetic-invalid-text-canary"

    def unreadable(_command):
        raise UnicodeError(canary)

    monkeypatch.setattr(release_artifacts, "bounded_command", unreadable)
    with pytest.raises(BootstrapError, match="invalid text") as caught:
        check_images()
    assert canary not in str(caught.value)
    assert caught.value.__suppress_context__, "ECR decoding exposed raw tool output"


@pytest.mark.parametrize("outcome", ["present", "missing", "empty", "wrong-account"])
def test_schema_v3_signed_release_reuse_requires_the_single_image_proof(
    signed_release, aws, tmp_path, outcome
):
    release, calls, reuse = signed_release
    manifest = tmp_path / "dist/current-release.json"
    manifest.write_text(
        json.dumps({"schema_version": 3, "deployable": True, "staging_only": False})
    )
    release["images"] = {"runtime": REFERENCES[0]}
    commands, response = aws
    response["value"] = inventory(DIGESTS[:1])
    if outcome == "missing":
        response.update(returncode=254, stderr="ImageNotFoundException")
        assert reuse() is None
    elif outcome == "present":
        assert reuse() == {**release, "release_reused": True}
    else:
        if outcome == "empty":
            response["value"] = inventory([])
        else:
            response["value"]["imageDetails"][0]["registryId"] = "111122223333"
        with pytest.raises(BootstrapError, match="inventory"):
            reuse()
    assert calls == ["verify", "load"]
    assert len(commands) == 1


@pytest.fixture(scope="module")
def ecr_repository_model():
    from botocore.loaders import Loader

    return Loader().load_service_model("ecr", "service-2")["shapes"]["RepositoryName"]


@pytest.mark.parametrize(
    ("repository_name", "valid"),
    [
        ("runtime", True),
        ("runtime--stable", True),
        ("runtime---stable", True),
        ("runtime__stable", True),
        ("gpu--fault/team__gpu/runtime--stable", True),
        ("gpu_fault/runtime.a-b__c---d", True),
        ("ab", True),
        ("a/b", True),
        pytest.param("a" * 256, True, id="maximum-length"),
        ("", False),
        ("a", False),
        pytest.param("a" * 257, False, id="overlong"),
        ("/runtime", False),
        ("runtime/", False),
        ("team//runtime", False),
        ("team/-runtime", False),
        ("runtime--", False),
        ("runtime___stable", False),
        ("runtime..stable", False),
        ("runtime._stable", False),
        (".runtime", False),
        ("team/Runtime", False),
        ("runtime:tag", False),
        ("runtime@stable", False),
        ("runtime space", False),
        ("../runtime", False),
    ],
)
def test_repository_inputs_follow_ecr_service_model_before_transport(
    aws, requested_digests, ecr_repository_model, repository_name, valid
):
    model_accepts = (
        ecr_repository_model["min"]
        <= len(repository_name)
        <= ecr_repository_model["max"]
        and re.fullmatch(ecr_repository_model["pattern"], repository_name) is not None
    )
    assert model_accepts is valid, "repository test case disagrees with the ECR model"
    commands, response = aws
    response["value"] = inventory(requested_digests)
    for image in response["value"]["imageDetails"]:
        image["repositoryName"] = repository_name
    repository = f"{REGISTRY}.dkr.ecr.{REGION}.amazonaws.com/{repository_name}"
    references = [f"{repository}@{digest}" for digest in requested_digests]

    def check():
        if len(references) == 1:
            return release_artifacts.runtime_image_exists(
                region=REGION, reference=references[0]
            )
        return release_artifacts.runtime_images_exist(
            region=REGION, references=references
        )

    if not valid:
        with pytest.raises(BootstrapError, match="reference"):
            check()
        assert not commands, "invalid ECR repository reached the transport"
        return

    assert check(), "a service-model-valid ECR repository was rejected"
    assert len(commands) == 1
    command, _options = commands[0]
    assert command[command.index("--repository-name") + 1] == repository_name, (
        "ECR repository identity was rewritten during validation"
    )
    assert command[command.index("--registry-id") + 1] == REGISTRY
    assert command[command.index("--region") + 1] == REGION
    assert command[
        command.index("--image-ids") + 1 : command.index("--no-paginate")
    ] == [f"imageDigest={digest}" for digest in sorted(requested_digests)]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("registryId", "111122223333"),
        ("repositoryName", "gpu-fault/runtime_stable"),
        ("imageDigest", "sha256:" + "d" * 64),
    ],
)
def test_legal_repository_separators_do_not_weaken_response_binding(
    aws, requested_digests, field, value
):
    commands, response = aws
    repository_name = "gpu--fault/runtime__stable"
    response["value"] = inventory(requested_digests)
    for image in response["value"]["imageDetails"]:
        image["repositoryName"] = repository_name
    response["value"]["imageDetails"][0][field] = value
    repository = f"{REGISTRY}.dkr.ecr.{REGION}.amazonaws.com/{repository_name}"
    references = [f"{repository}@{digest}" for digest in requested_digests]

    with pytest.raises(BootstrapError, match="identity differs"):
        if len(references) == 1:
            release_artifacts.runtime_image_exists(
                region=REGION, reference=references[0]
            )
        else:
            release_artifacts.runtime_images_exist(region=REGION, references=references)

    assert len(commands) == 1, "the legal repository was rejected before its proof"
