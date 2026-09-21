from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Mapping

from gpu_fault.admin.bootstrap_common import (
    BootstrapError,
    CommandRunner,
    compute_agent_config_digest,
)
from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault.admin.execution import run_command as bounded_command
from gpu_fault.admin.postgres_grant import PostgresTestAllocation
from gpu_fault.admin.release_postgres import (
    isolated_postgres_allocation,
    require_postgres_cleanup,
)
from gpu_fault.admin.release_consent import (
    ACCEPT_SCHEMA_CHANGE_ENV,
    ALLOW_INFLIGHT_INSTALLS_ENV,
    SUPERSEDE_FAILED_TRANSACTION_ENV,
)

DEFAULT_ADOT_IMAGE_AMD64 = (
    "public.ecr.aws/aws-observability/aws-otel-collector@"
    "sha256:bb72328152c72fb9662056759b275f7cc85e115db12bbb114fbea9f68dc4816c"
)
STAGING_IMPACT_PLAN = Path("dist/staging-impact-plan.json")
# ECR RepositoryName service-model grammar, including its 2..256 character limit.
ECR_DIGEST_REFERENCE = re.compile(
    r"(?P<registry>[0-9]{12})\.dkr\.ecr\.(?P<region>[a-z0-9-]+)"
    r"\.amazonaws\.com(?P<china>\.cn)?/"
    r"(?P<repository>(?=[^@]{2,256}@)[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
    r"(?:/[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*)*)"
    r"@(?P<digest>sha256:[0-9a-f]{64})"
)


def _private_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise BootstrapError(f"{description} is missing: {resolved}")
    if resolved.stat().st_mode & 0o077:
        raise BootstrapError(f"{description} must not be group/other accessible")
    return resolved


def _git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise BootstrapError(
            f"git command failed while checking release source: "
            f"{(completed.stderr or '').strip()}"
        )
    return (completed.stdout or "").strip()


def load_reusable_signed_release(
    runner: CommandRunner,
    *,
    repository_root: Path,
    state_dir: Path,
    region: str,
    runtime_repository: str,
    runtime_profile: str,
    staging_only: bool,
    impact_base: str,
    cosign_public_key: Path | None = None,
) -> dict[str, Any] | None:
    manifest = repository_root / "dist/current-release.json"
    attestation = repository_root / "dist/current-attestation.json"
    bundle = repository_root / "dist/current-attestation.bundle.json"
    required = (manifest, attestation, bundle)
    if not all(path.is_file() for path in required):
        return None
    try:
        manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
        value = json.loads(attestation.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    manifest_tier = (
        manifest_value.get("staging_only", False)
        if isinstance(manifest_value, dict)
        else None
    )
    if (
        not isinstance(manifest_value, dict)
        or int(manifest_value.get("schema_version", 0)) < 3
        or manifest_value.get("deployable") is not True
        or not isinstance(manifest_tier, bool)
        or manifest_tier is not staging_only
    ):
        return None
    source = value.get("source") if isinstance(value, dict) else None
    if not isinstance(source, dict) or source.get("dirty") is not False:
        return None
    if staging_only and value.get("impact_base") != impact_base:
        return None
    current_commit = _git_output(repository_root, "rev-parse", "HEAD")
    if str(source.get("git_commit") or "") != current_commit:
        return None
    verify_prebuilt_release(
        runner,
        repository_root=repository_root,
        state_dir=state_dir,
        cosign_public_key=cosign_public_key,
        staging_only=staging_only,
    )
    release = load_prebuilt_release(
        runner,
        repository_root=repository_root,
        runtime_profile=runtime_profile,
        allow_staging=staging_only,
    )
    references = []
    for name in ("runtime", "executor", "node_dependencies"):
        runtime_reference = str(release["images"].get(name) or "")
        if not runtime_reference and name != "runtime":
            continue
        if not runtime_reference.startswith(
            runtime_repository.rstrip("/") + "@sha256:"
        ):
            return None
        references.append(runtime_reference)
    # Legacy shared-image callers retain the single-image compatibility helper.
    exists = (
        runtime_image_exists(region=region, reference=references[0])
        if len(references) == 1
        else runtime_images_exist(region=region, references=references)
    )
    if not exists:
        return None
    return {**release, "release_reused": True}


def runtime_images_exist(*, region: str, references: Sequence[str]) -> bool:
    """Prove the complete signed digest set with one repository-scoped ECR read."""
    registry = ""
    repository = ""
    digests: set[str] = set()
    for reference in references:
        match = ECR_DIGEST_REFERENCE.fullmatch(reference)
        if (
            match is None
            or match["region"] != region
            or bool(match["china"]) != region.startswith("cn-")
        ):
            raise BootstrapError("signed runtime image reference is invalid")
        if digests and (registry, repository) != (
            match["registry"],
            match["repository"],
        ):
            raise BootstrapError("signed runtime images must share one ECR repository")
        registry, repository = match["registry"], match["repository"]
        digests.add(match["digest"])
    if not digests or len(digests) > 100:
        raise BootstrapError("signed runtime image digest batch is invalid")
    completed = _run_runtime_image_check(
        [
            "aws",
            "ecr",
            "describe-images",
            "--region",
            region,
            "--registry-id",
            registry,
            "--repository-name",
            repository,
            "--image-ids",
            *(f"imageDigest={digest}" for digest in sorted(digests)),
            "--no-paginate",
            "--output",
            "json",
        ],
    )
    if completed.returncode:
        error = (completed.stderr or "").strip()
        # AWS CLI 2.35 prefixes its error line with ``aws: [ERROR]: `` (live
        # 2026-09-20); only that prefix is tolerated, nothing else may precede.
        code = re.fullmatch(
            r"(?:aws: \[ERROR\]: )?"
            r"An error occurred \((\w+)\) when calling the DescribeImages operation:"
            r"[^\r\n]*",
            error,
        )
        if completed.returncode in {1, 254, 255} and (code[1] if code else error) in {
            "ImageNotFoundException",
            "RepositoryNotFoundException",
        }:
            return False
        detail = (
            "AWS credential helper failed; check credential_process and tool installation"
            if any(
                marker in error.lower()
                for marker in (
                    "credential_process",
                    "credential helper",
                    "error when retrieving credentials",
                    "error getting credentials",
                )
            )
            else diagnostic_text(error) or "ECR returned no error detail"
        )
        raise BootstrapError(
            f"cannot verify signed runtime images: {detail}; "
            "check AWS CLI credentials, ECR read permissions and connectivity"
        )
    try:
        value = json.loads(completed.stdout or "")
    except json.JSONDecodeError as exc:
        raise BootstrapError("signed runtime image inventory is invalid JSON") from exc
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("imageDetails"), list)
        or value.get("nextToken") not in (None, "")
    ):
        raise BootstrapError("signed runtime image inventory is invalid or incomplete")
    observed: set[str] = set()
    for image in value["imageDetails"]:
        if not isinstance(image, dict):
            raise BootstrapError("signed runtime image inventory entry is invalid")
        digest = image.get("imageDigest")
        if (
            image.get("registryId") != registry
            or image.get("repositoryName") != repository
            or not isinstance(digest, str)
            or digest not in digests
            or digest in observed
        ):
            raise BootstrapError("signed runtime image inventory identity differs")
        observed.add(digest)
    if observed != digests:
        raise BootstrapError("signed runtime image inventory is incomplete")
    return True


def _run_runtime_image_check(
    arguments: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    try:
        return bounded_command(arguments)
    except (subprocess.TimeoutExpired, TimeoutError):
        raise BootstrapError(
            "cannot verify signed runtime image: ECR request exceeded its "
            "deployment time budget"
        ) from None
    except OSError:
        raise BootstrapError(
            "cannot verify signed runtime image: cannot execute AWS CLI; "
            "check the deployment tools and executable permissions"
        ) from None
    except UnicodeError:
        raise BootstrapError(
            "cannot verify signed runtime image: ECR returned invalid text; "
            "check AWS CLI output"
        ) from None


def runtime_image_exists(*, region: str, reference: str) -> bool:
    """Compatibility entry point with the same identity proof as split images."""
    return runtime_images_exist(region=region, references=(reference,))


# Deploy-scoped consent the CLI exports for the release engine. The release
# gates (static, pytest, postgres) run in a child process of that same deploy,
# and a suite that inherits the consent reads it as the state under test: with
# ``--accept-schema-change`` one rollback-compatibility test took the accepted
# branch (deploy #18, 2026-09-09); with ``--supersede-failed-transaction`` 35
# deploy/resume decision tests demanded a supersede (deploy #22). The gates
# judge the tree, not this deploy's consent, so the consent stays out of them.
#: The Makefile resolves ``PROMTOOL`` (and the audit tools) from the venv that
#: ``SUPPLY_CHAIN_PYTHON`` lives in; ``make ci-supply-chain-tools`` builds that
#: venv, by default under ``SUPPLY_CHAIN_TOOLS_DEFAULT``.
SUPPLY_CHAIN_PYTHON_ENV = "SUPPLY_CHAIN_PYTHON"
PROMTOOL_ENV = "PROMTOOL"
SITE_TOOLCHAIN_DIRECTORY = "toolchain"
SUPPLY_CHAIN_TOOLS_DEFAULT = Path("/tmp/gpu-fault-supply-chain-tools")


def _supply_chain_tools(venv: Path) -> tuple[Path, Path] | None:
    python, promtool = venv / "bin" / "python", venv / "bin" / "promtool"
    if all(path.is_file() and os.access(path, os.X_OK) for path in (python, promtool)):
        return python, promtool
    return None


def supply_chain_tools_environment(
    environment: Mapping[str, str], *, state_dir: Path
) -> dict[str, str]:
    """Bind the release gates' pinned tools to the site, not to the caller's shell.

    The gates ran ``make promtool-preflight`` with whatever ``SUPPLY_CHAIN_PYTHON``
    the operator's shell happened to export. A site whose ``<state-dir>/toolchain``
    held the pinned promtool still failed its redeploy from a fresh shell
    (2026-09-17), because nothing recorded where the tools were. Resolution
    order: the caller's explicit venv (its promtool must exist -- an explicit
    but incomplete venv is refused, never silently replaced), then the site's
    own ``<state-dir>/toolchain``, then the Makefile default; none of them is
    a fail-closed error naming the setup command.
    """

    bound = dict(environment)
    explicit = bound.get(SUPPLY_CHAIN_PYTHON_ENV, "").strip()
    if explicit:
        # The venv is where ``bin/python`` lives, never where its symlink chain
        # ends (``python -> python3.12 -> /usr/bin/python3.12`` left the gates
        # looking for ``/usr/bin/promtool``, 2026-09-18).
        tools = _supply_chain_tools(
            Path(os.path.abspath(Path(explicit).expanduser())).parent.parent
        )
        if tools is None:
            raise BootstrapError(
                f"{SUPPLY_CHAIN_PYTHON_ENV}={explicit} names a venv without an "
                "executable bin/python and bin/promtool; run "
                "make ci-supply-chain-tools SUPPLY_CHAIN_TOOLS_VENV=<that venv>"
            )
        bound.setdefault(PROMTOOL_ENV, str(tools[1]))
        return bound
    site_toolchain = state_dir / SITE_TOOLCHAIN_DIRECTORY
    for candidate in (site_toolchain, SUPPLY_CHAIN_TOOLS_DEFAULT):
        tools = _supply_chain_tools(candidate)
        if tools is not None:
            bound[SUPPLY_CHAIN_PYTHON_ENV] = str(tools[0])
            bound[PROMTOOL_ENV] = str(tools[1])
            return bound
    raise BootstrapError(
        "the release gates need the pinned supply-chain tools (promtool): none "
        f"under {site_toolchain} or {SUPPLY_CHAIN_TOOLS_DEFAULT}; run "
        f"make ci-supply-chain-tools SUPPLY_CHAIN_TOOLS_VENV={site_toolchain} "
        f"(or set {SUPPLY_CHAIN_PYTHON_ENV})"
    )


RELEASE_GATE_EXCLUDED_ENVIRONMENT = frozenset(
    {
        ACCEPT_SCHEMA_CHANGE_ENV,
        SUPERSEDE_FAILED_TRANSACTION_ENV,
        ALLOW_INFLIGHT_INSTALLS_ENV,
    }
)


def release_gate_environment(
    parent: Mapping[str, str],
    *,
    postgres_url: str,
    cosign_password: str | None,
) -> dict[str, str]:
    """The environment the release gates and the release build run under."""

    environment = {
        key: value
        for key, value in parent.items()
        if key not in RELEASE_GATE_EXCLUDED_ENVIRONMENT
    }
    environment["GPU_FAULT_TEST_POSTGRES_URL"] = postgres_url
    if cosign_password is not None:
        environment["COSIGN_PASSWORD"] = cosign_password
    return environment


@contextmanager
def isolated_postgres_url(runner: CommandRunner) -> Iterator[str]:
    """Compatibility view; builds also consume the allocation's private grant."""
    with isolated_postgres_allocation(runner) as allocation:
        yield allocation.url


def verify_prebuilt_release(
    runner: CommandRunner,
    *,
    repository_root: Path,
    state_dir: Path,
    cosign_public_key: Path | None = None,
    staging_only: bool = False,
) -> None:
    attestation = repository_root / "dist/current-attestation.json"
    bundle = repository_root / "dist/current-attestation.bundle.json"
    public_key = (
        (cosign_public_key or state_dir / "release-signing/cosign.pub")
        .expanduser()
        .resolve()
    )
    if not attestation.is_file() or not bundle.is_file():
        raise BootstrapError("signed release artifacts are missing after release-build")
    if not public_key.is_file():
        raise BootstrapError(f"cosign public key is missing: {public_key}")
    script = repository_root / "scripts/verify-release-attestation.py"
    runner.run(
        [
            sys.executable,
            str(script),
            "--attestation",
            str(attestation),
            "--bundle",
            str(bundle),
            "--cosign-key",
            str(public_key),
            *(["--allow-staging-release"] if staging_only else []),
        ],
        cwd=repository_root,
    )


def prepare_staging_impact_plan(
    runner: CommandRunner,
    *,
    repository_root: Path,
    impact_base: str,
    environment: Mapping[str, str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    path = repository_root / STAGING_IMPACT_PLAN
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(repository_root / "scripts/select-affected-tests.py"),
        "--base",
        impact_base,
        "--format",
        "json",
        "--write-plan",
        str(path),
    ]
    # Selection runs the promtool preflight itself, so it needs the same
    # site-bound tools as the gates.
    output = (
        runner.run(command, cwd=repository_root)
        if environment is None
        else runner.run(command, cwd=repository_root, env=dict(environment))
    )
    try:
        plan = json.loads(output)
    except json.JSONDecodeError as exc:
        raise BootstrapError("staging impact plan output is invalid") from exc
    if not isinstance(plan, dict) or not isinstance(plan.get("postgres"), bool):
        raise BootstrapError("staging impact plan is incomplete")
    return path, plan


def restore_main_ci_candidate(
    runner: CommandRunner,
    *,
    repository_root: Path,
) -> dict[str, Any] | None:
    output = runner.run(
        [
            sys.executable,
            str(repository_root / "scripts/restore_ci_candidate.py"),
            "--root",
            str(repository_root),
            "--destination",
            str(repository_root / "dist"),
        ],
        cwd=repository_root,
    )
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        raise BootstrapError("CI candidate restore returned invalid JSON") from exc
    if not isinstance(value, dict) or not isinstance(value.get("available"), bool):
        raise BootstrapError("CI candidate restore result is incomplete")
    if value["available"] is False:
        reason = str(value.get("reason") or "trusted main CI candidate unavailable")
        print(
            "release-build: trusted main CI candidate unavailable; "
            f"falling back to local gates: {reason}",
            file=sys.stderr,
        )
        return None
    gate = Path(str(value.get("ci_gate") or "")).resolve()
    try:
        gate.relative_to((repository_root / "dist").resolve())
    except ValueError as exc:
        raise BootstrapError("restored CI gate leaves repository dist") from exc
    if not gate.is_file():
        raise BootstrapError("restored CI gate is missing")
    return value


def load_verified_ci_candidate_receipt(
    runner: CommandRunner,
    *,
    repository_root: Path,
    state_dir: Path,
    cosign_public_key: Path | None = None,
) -> dict[str, Any] | None:
    commit = _git_output(repository_root, "rev-parse", "HEAD")
    receipt_root = state_dir / "ci-candidates" / commit
    receipt = receipt_root / "verification.json"
    signature = receipt_root / "verification.sigstore.json"
    if not receipt.exists() and not signature.exists():
        return None
    public_key = (
        (cosign_public_key or state_dir / "release-signing/cosign.pub")
        .expanduser()
        .resolve()
    )
    output = runner.run(
        [
            sys.executable,
            str(repository_root / "scripts/ci_candidate_receipt.py"),
            "verify",
            "--root",
            str(repository_root),
            "--state-dir",
            str(state_dir),
            "--public-key",
            str(public_key),
        ],
        cwd=repository_root,
    )
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        raise BootstrapError("CI candidate receipt returned invalid JSON") from exc
    if not isinstance(value, dict) or not isinstance(value.get("available"), bool):
        raise BootstrapError("CI candidate receipt result is incomplete")
    if value["available"] is False:
        return None
    gate = Path(str(value.get("ci_gate") or "")).resolve()
    try:
        gate.relative_to((repository_root / "dist").resolve())
    except ValueError as exc:
        raise BootstrapError(
            "verified CI candidate gate leaves repository dist"
        ) from exc
    if not gate.is_file():
        raise BootstrapError("verified CI candidate gate is missing")
    return value


def build_signed_release(
    runner: CommandRunner,
    *,
    repository_root: Path,
    state_dir: Path,
    region: str,
    runtime_repository: str,
    cache_repository: str | None,
    runtime_profile: str,
    cosign_signing_key: Path | None = None,
    cosign_public_key: Path | None = None,
    cosign_password_file: Path | None = None,
    staging_only: bool = False,
    impact_base: str = "origin/main",
) -> dict[str, Any]:
    if staging_only and not impact_base.strip():
        raise BootstrapError("staging release requires a non-empty impact base")
    require_postgres_cleanup(state_dir)
    if _git_output(
        repository_root,
        "status",
        "--porcelain",
        "--untracked-files=normal",
    ):
        raise BootstrapError("release build requires a clean source tree")
    reusable = load_reusable_signed_release(
        runner,
        repository_root=repository_root,
        state_dir=state_dir,
        region=region,
        runtime_repository=runtime_repository,
        runtime_profile=runtime_profile,
        staging_only=staging_only,
        impact_base=impact_base,
        cosign_public_key=cosign_public_key,
    )
    if reusable is not None:
        return reusable
    promoted_candidate = None
    if not staging_only:
        promoted_candidate = load_verified_ci_candidate_receipt(
            runner,
            repository_root=repository_root,
            state_dir=state_dir,
            cosign_public_key=cosign_public_key,
        )
        if promoted_candidate is None:
            promoted_candidate = restore_main_ci_candidate(
                runner,
                repository_root=repository_root,
            )
    release_source = (
        "staging_impact"
        if staging_only
        else "main_ci_candidate"
        if promoted_candidate is not None
        else "local_full_gate"
    )
    # Said up front, because it decides whether the next four minutes are spent
    # re-running gates that CI already passed on main.
    print(
        f"release-build: release_source={release_source}"
        + (
            " (no verified CI candidate for this tree; running the full gates locally)"
            if release_source == "local_full_gate"
            else ""
        ),
        file=sys.stderr,
        flush=True,
    )
    impact_plan_path: Path | None = None
    postgres_required = promoted_candidate is None
    tools_environment = supply_chain_tools_environment(os.environ, state_dir=state_dir)
    if staging_only:
        impact_plan_path, impact_plan = prepare_staging_impact_plan(
            runner,
            repository_root=repository_root,
            impact_base=impact_base,
            environment=tools_environment,
        )
        postgres_required = bool(impact_plan["postgres"])
    signing_dir = state_dir / "release-signing"
    signing_key = _private_file(
        cosign_signing_key or signing_dir / "cosign.key",
        "cosign signing key",
    )
    password_path = cosign_password_file or signing_dir / "cosign.password"
    password = None
    if password_path.expanduser().is_file():
        password = (
            _private_file(
                password_path,
                "cosign password file",
            )
            .read_text(encoding="utf-8")
            .strip()
        )
    registry = runtime_repository.split("/", 1)[0]
    login_password = runner.run(
        [
            "aws",
            "ecr",
            "get-login-password",
            "--region",
            region,
        ],
        sensitive=True,
    )
    runner.run(
        [
            "docker",
            "login",
            "--username",
            "AWS",
            "--password-stdin",
            registry,
        ],
        input_text=login_password + "\n",
        sensitive=True,
        mutate=True,
    )
    command = [
        "make",
        f"PYTHON={sys.executable}",
        (
            "release-build-staging"
            if staging_only
            else "release-build-promoted"
            if promoted_candidate is not None
            else "release-build"
        ),
        f"RUNTIME_IMAGE_REPOSITORY={runtime_repository}",
        f"COSIGN_SIGNING_KEY={signing_key}",
    ]
    if promoted_candidate is not None:
        command.append(f"CI_GATE={promoted_candidate['ci_gate']}")
    if staging_only:
        command.append(f"BASE={impact_base}")
        assert impact_plan_path is not None
        command.extend(
            [
                f"STAGING_IMPACT_PLAN={impact_plan_path}",
                "IMPACT_PLAN_PREPARED=1",
                f"COMPONENT_ARTIFACT_CACHE_ROOT={state_dir / 'component-artifacts'}",
            ]
        )
    if cache_repository:
        cache_ref = f"{cache_repository}:buildcache-linux-amd64"
        command.extend(
            [
                f"RUNTIME_IMAGE_CACHE_FROM=type=registry,ref={cache_ref}",
                (
                    "RUNTIME_IMAGE_CACHE_TO=type=registry,"
                    f"ref={cache_ref},mode=max,image-manifest=true,"
                    "oci-mediatypes=true"
                ),
            ]
        )
    postgres_context = (
        isolated_postgres_allocation(
            runner, repository_root=repository_root, state_dir=state_dir
        )
        if postgres_required
        else nullcontext(PostgresTestAllocation(""))
    )
    with postgres_context as postgres:
        environment = postgres.build_environment(
            release_gate_environment(
                tools_environment,
                postgres_url=postgres.url,
                cosign_password=password,
            )
        )
        runner.run(
            command,
            cwd=repository_root,
            env=environment,
            mutate=True,
            capture=False,
        )
    verify_prebuilt_release(
        runner,
        repository_root=repository_root,
        state_dir=state_dir,
        cosign_public_key=cosign_public_key,
        staging_only=staging_only,
    )
    release = load_prebuilt_release(
        runner,
        repository_root=repository_root,
        runtime_profile=runtime_profile,
        allow_staging=staging_only,
    )
    return {
        **release,
        "release_reused": False,
        "release_source": release_source,
    }


def load_prebuilt_release(
    runner: CommandRunner,
    *,
    repository_root: Path,
    runtime_profile: str,
    allow_staging: bool = False,
) -> dict[str, Any]:
    manifest = repository_root / "dist/current-release.json"
    if not manifest.is_file():
        raise BootstrapError(
            "dist/current-release.json is missing; build and sign the release in CI"
        )
    try:
        release = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapError(f"release manifest is invalid: {exc}") from exc
    if (
        int(release.get("schema_version", 0)) < 3
        or release.get("deployable") is not True
    ):
        raise BootstrapError("ARN bootstrap requires a deployable schema v3 release")
    tier = release.get("staging_only", False)
    if not isinstance(tier, bool):
        raise BootstrapError("release manifest staging_only must be a boolean")
    staging_only = tier
    if staging_only and not allow_staging:
        raise BootstrapError(
            "staging-only release requires explicit staging authorization"
        )
    images = dict((release.get("delivery") or {}).get("images") or {})
    image_names = ["runtime", "node_installer", "dcgm_exporter", "adot"]
    if release["schema_version"] == 4:
        image_names.extend(("executor", "node_dependencies"))
    elif release["schema_version"] != 3:
        raise BootstrapError("unsupported release manifest schema version")
    required_images = {
        name: str((images.get(name) or {}).get("reference") or "")
        for name in image_names
    }
    if any("@sha256:" not in value for value in required_images.values()):
        raise BootstrapError("release manifest image identity is incomplete")
    return {
        "manifest": str(manifest),
        "release_id": str(release["release_id"]),
        "images": required_images,
        "agent_config_digest": compute_agent_config_digest(
            runner,
            repository_root=repository_root,
            runtime_profile_version=runtime_profile,
        ),
        "staging_only": staging_only,
    }
