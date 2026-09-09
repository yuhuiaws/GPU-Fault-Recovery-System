"""Security guards for the node installer chain.

The HyperPod installer Job extracts a tarball inside ``chroot /host`` as
root and runs the installer it finds there, and the installer writes the
cluster bearer token and node action secret to ``/etc/gpu-fault/*.env``.
These tests pin the properties a security review found missing:

* the bundle digest is compared before any ``tar -x`` of it, and the
  inline pod shells fail closed on a broken pipeline;
* the bearer token travels as a 0600 file, never on the installer's argv;
* secret-bearing env files exist with mode 0600 before their content is
  written, instead of being created 0644 and tightened afterwards;
* the node runtime's dependency closure installs with ``--require-hashes``
  from a lock shipped in the bundle, and the wheel digest is a hard gate
  anchored outside the bundle.

Nothing here shells out to kubectl or aws; the scripts are checked with
``bash -n``, text assertions, and by running pure fragments under bash.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NODE_DIR = ROOT / "deploy" / "node"
INSTALLER = NODE_DIR / "install-gpu-fault-collector.sh"
JOB = NODE_DIR / "run-hyperpod-installer-job.sh"
BUNDLE_BUILDER = NODE_DIR / "build-node-installer-bundle.sh"
PREFLIGHT = NODE_DIR / "preflight-gpu-fault-node.sh"
EDITED_SCRIPTS = (INSTALLER, JOB, BUNDLE_BUILDER, PREFLIGHT)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_edited_scripts_are_valid_bash() -> None:
    for script in EDITED_SCRIPTS:
        result = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, f"{script.name}: {result.stderr}"


# --- H-6: secret env files are never observable at 0644 -----------------


def test_every_secret_env_file_is_pre_created_0600_before_its_redirect() -> None:
    """``{ ... } > file`` creates the file with the process umask (0644 here)
    and only a later ``chmod 0600`` closes it; the token and node action
    secret are readable by every local user in between. Either the script
    sets ``umask 0077`` first or each file is pre-created with mode 0600."""

    installer = _text(INSTALLER)
    redirects = [
        (match.start(), match.group(1))
        for match in re.finditer(
            r"(?<!>)>\s*(/etc/gpu-fault/[a-z-]+\.env)\b", installer
        )
    ]
    assert redirects, "expected the installer to write /etc/gpu-fault/*.env"
    umask_at = installer.find("umask 0077")
    for position, path in redirects:
        if 0 <= umask_at < position:
            continue
        pre_create = f"install -m 0600 /dev/null {path}"
        assert 0 <= installer.find(pre_create) < position, (
            f"{path} is written before it is created 0600 (missing '{pre_create}')"
        )


def test_node_agent_tls_key_is_pre_created_0600() -> None:
    installer = _text(INSTALLER)
    keyout = installer.index('-keyout "${NODE_AGENT_TLS_KEY}"')
    pre_create = installer.find('install -m 0600 /dev/null "${NODE_AGENT_TLS_KEY}"')
    assert 0 <= pre_create < keyout


# --- H-2 / H-3: bundle digest before extraction, pipefail everywhere ---


def _inline_shell_bodies(job: str) -> list[str]:
    """Return the body of every inline ``/bin/bash -c...`` shell in the Job."""

    pattern = re.compile(r"/bin/bash -c[a-z]*(?: -o pipefail)? '")
    starts = [match.end() for match in pattern.finditer(job)]
    assert len(starts) == 2, "expected the preflight and install inline shells"
    bodies = []
    for start in starts:
        end = job.index("'", start)
        bodies.append(job[start:end])
    return bodies


def test_every_inline_pod_shell_enables_pipefail() -> None:
    job = _text(JOB)
    for match in re.finditer(r'command: \["/bin/bash", ([^\]]*)\]', job):
        assert "pipefail" in match.group(1), match.group(0)
    inline = re.findall(r"/bin/bash -c[a-z]*( -o pipefail)? '", job)
    assert inline, "expected inline /bin/bash -c shells in the Job"
    assert all(inline), "an inline /bin/bash -c shell runs without pipefail"


def test_bundle_digest_is_verified_before_every_tar_extraction() -> None:
    job = _text(JOB)
    bodies = _inline_shell_bodies(job)
    extractions = 0
    for body in bodies:
        for match in re.finditer(r"tar\s+-x", body):
            extractions += 1
            before = body[: match.start()]
            assert "sha256sum" in before, "tar -x runs before sha256sum"
            assert "${INSTALLER_BUNDLE_SHA256}" in before, (
                "tar -x runs before the bundle digest comparison"
            )
            assert "bundle SHA-256 mismatch" in before
    assert extractions == 2, "expected the preflight and install tar -x calls"


def test_preflight_shell_refuses_an_empty_expected_digest() -> None:
    preflight_body = _inline_shell_bodies(_text(JOB))[0]
    assert "=~ ^[0-9a-f]{64}$" in preflight_body, (
        "the preflight shell must refuse to extract when no digest was rendered"
    )
    assert "exit 0" not in preflight_body


# --- H-7: the bearer token stays off argv -------------------------------


def test_installer_job_never_puts_the_bearer_token_on_argv() -> None:
    job = _text(JOB)
    assert '--token "' not in job
    assert "--token-file" in job
    # /usr/bin/env VAR=value puts the value on env's argv as well.
    assert 'CONTROL_PLANE_TOKEN="\\${CONTROL_PLANE_TOKEN}"' not in job
    assert "name: CONTROL_PLANE_TOKEN" not in job
    assert "install -m 0600 /connection-secret/cluster-token" in job
    assert "gpu-fault-control-plane-token-${INSTALL_RUN_ID}" in job
    volume = job.split("CONNECTION_SECRET_VOLUME=", 1)[1].split("EOF\n)", 1)[0]
    assert "key: cluster-token" in volume


def test_installer_accepts_a_token_file_and_keeps_token_for_compat() -> None:
    installer = _text(INSTALLER)
    assert '--token-file) require_value "$@"; TOKEN_FILE="$2"; shift 2 ;;' in installer
    assert '--token) require_value "$@"; TOKEN="$2"; shift 2 ;;' in installer
    assert "use only one bearer token input" in installer
    assert 'IFS= read -r TOKEN < "${TOKEN_FILE}"' in installer


# --- H-8: hash-pinned dependency install --------------------------------


def test_node_runtime_dependencies_install_with_require_hashes() -> None:
    installer = _text(INSTALLER)
    slot = installer.split("prepare_runtime_slot() {", 1)[1].split("\n}\n", 1)[0]
    assert "--require-hashes" in slot
    assert "--no-deps" in slot
    assert "[collectors]" not in slot, "extras resolve the closure online"
    assert "--upgrade" not in slot
    joined = re.sub(r"\\\n\s*", " ", slot)
    pip_calls = [line for line in joined.splitlines() if "-m pip install" in line]
    assert pip_calls, "expected a direct pip install of the wheel"
    for pip_call in pip_calls:
        assert "--require-hashes" in pip_call or "--no-index" in pip_call, pip_call
    pip_args = joined.split("PIP_ARGS=(", 1)[1].split(")", 1)[0]
    assert "--require-hashes" in pip_args and "--no-deps" in pip_args, pip_args
    assert '--requirement "${DEPENDENCY_LOCK}"' in pip_args


def test_installer_defaults_to_the_narrow_node_runtime_lock() -> None:
    """The node installer must install from the narrow node-runtime lock, not
    the broad control-plane runtime.lock. The narrow lock pins only the node
    collector/agent's true runtime import closure, so --require-hashes cannot
    pull the control-plane-only dependencies (psycopg, uvloop, ...)."""

    installer = _text(INSTALLER)
    assert 'DEPENDENCY_LOCK="${REPO_DIR}/requirements/node-runtime.lock"' in installer
    # The old broad default must be gone, so nothing silently reverts to it.
    assert 'DEPENDENCY_LOCK="${REPO_DIR}/requirements/runtime.lock"' not in installer


def test_bundle_ships_the_node_runtime_lock_and_preflight_requires_it() -> None:
    builder = _text(BUNDLE_BUILDER)
    preflight = _text(PREFLIGHT)
    assert "requirements/node-runtime.lock" in builder
    assert "/requirements/node-runtime.lock" in preflight
    # The bundle must not also ship or require the broad runtime.lock.
    assert "requirements/runtime.lock" not in builder
    assert "/requirements/runtime.lock" not in preflight


NODE_RUNTIME_LOCK = ROOT / "requirements" / "node-runtime.lock"


def test_node_runtime_lock_covers_only_the_node_closure() -> None:
    lock = _text(NODE_RUNTIME_LOCK)
    # The node collector/agent's true runtime imports. prometheus-client is
    # imported inside ``DcgmCollector.collect_text`` (a function-local import,
    # invisible to a top-level scan): the lock exported with
    # ``--no-emit-package prometheus-client`` shipped to the first canary node
    # on 2026-09-09 and every DCGM scrape died with ModuleNotFoundError, so
    # the node never delivered a metric and the installer verify failed.
    for name in (
        "boto3",
        "fastapi",
        "kubernetes",
        "prometheus-client",
        "pydantic",
        "pyyaml",
        "uvicorn",
    ):
        assert re.search(rf"^{name}==", lock, re.MULTILINE), name
    # Broad control-plane-only dependencies the node never imports must not be
    # dragged onto the host by the lock.
    for name in ("psycopg", "psycopg-binary", "uvloop", "httptools"):
        assert not re.search(rf"^{name}==", lock, re.MULTILINE), name


def _normalized_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def test_node_runtime_lock_pins_every_third_party_import_of_the_node_closure() -> None:
    """Every third-party module any shipped node-runtime module imports, at
    any nesting depth, must resolve to a distribution the lock pins. The lock
    is the only place the hash-pinned install can take a dependency from, so a
    module the lock omits is a ModuleNotFoundError on the node."""
    import ast
    import sys
    from importlib.metadata import packages_distributions

    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import component_wheels

    pinned = {
        _normalized_distribution(match.group(1))
        for match in re.finditer(r"(?m)^([A-Za-z0-9_.-]+)==", _text(NODE_RUNTIME_LOCK))
    }
    distributions = packages_distributions()
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    missing: set[str] = set()
    for module in sorted(component_wheels.component_modules("node_runtime")):
        tree = ast.parse(component_wheels.MODULES[module].read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported = [node.module]
            else:
                continue
            for name in imported:
                top = name.split(".", 1)[0]
                if top in stdlib or component_wheels.is_local_module(top):
                    continue
                candidates = {
                    _normalized_distribution(dist)
                    for dist in distributions.get(top, ())
                }
                assert candidates, f"{module} imports {top}, not installed in this venv"
                if not candidates & pinned:
                    missing.add(f"{module} -> {top} ({', '.join(sorted(candidates))})")
    assert missing == set(), sorted(missing)


def test_node_runtime_lock_is_hash_complete() -> None:
    """``pip install --require-hashes`` refuses any requirement line without a
    hash, so every pinned distribution in the lock must carry at least one
    ``--hash=sha256:`` pin."""

    lock = _text(NODE_RUNTIME_LOCK)
    blocks = re.split(r"(?m)^(?=[a-zA-Z0-9_.-]+==)", lock)
    pinned = 0
    for block in blocks:
        match = re.match(r"^([a-zA-Z0-9_.-]+)==", block)
        if not match:
            continue
        pinned += 1
        assert re.search(r"--hash=sha256:[0-9a-f]{64}", block), match.group(1)
    assert pinned, "the node-runtime lock pins no distributions"


# --- H-2: the wheel digest is a hard gate anchored outside the bundle ---

WHEEL_BLOCK_START = 'MANIFEST_WHEEL_SHA256=""'
WHEEL_BLOCK_END = (
    'sha256 ${WHEEL_SHA256} does not match expected ${EXPECTED_WHEEL_SHA256}"'
)


def _wheel_probe(target: Path) -> Path:
    installer = _text(INSTALLER)
    assert installer.count(WHEEL_BLOCK_START) == 1
    assert installer.count(WHEEL_BLOCK_END) == 1
    start = installer.index(WHEEL_BLOCK_START)
    end = installer.index(WHEEL_BLOCK_END) + len(WHEEL_BLOCK_END)
    probe = target / "probe.sh"
    probe.write_text(
        "set -euo pipefail\n"
        'REPO_DIR="$1"\n'
        'EXPECTED_WHEEL_SHA256="${2:-}"\n'
        "PYTHON_COMMAND=python3\n"
        'WHEEL=""\n'
        "die() { printf 'DIE: %s\\n' \"$*\"; exit 1; }\n"
        f"{installer[start:end]}\n"
        'printf \'WHEEL=%s\\nSHA=%s\\n\' "${WHEEL}" "${WHEEL_SHA256}"\n',
        encoding="utf-8",
    )
    return probe


def _run_probe(probe: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(probe), *arguments], capture_output=True, text=True, check=False
    )


def test_installer_refuses_to_install_without_an_expected_wheel_digest(
    tmp_path: Path,
) -> None:
    probe = _wheel_probe(tmp_path)
    bundle = tmp_path / "bundle"
    wheel = bundle / "dist/gpu_fault_node_runtime-0.10.0.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"bundle-wheel")

    result = _run_probe(probe, str(bundle))
    assert result.returncode != 0
    assert "expected wheel SHA-256 is required" in result.stdout

    result = _run_probe(probe, str(bundle), "not-a-digest")
    assert result.returncode != 0
    assert "expected wheel SHA-256 is required" in result.stdout


def test_installer_rejects_a_wheel_that_differs_from_the_expected_digest(
    tmp_path: Path,
) -> None:
    probe = _wheel_probe(tmp_path)
    bundle = tmp_path / "bundle"
    wheel = bundle / "dist/gpu_fault_node_runtime-0.10.0.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"tampered")

    result = _run_probe(probe, str(bundle), hashlib.sha256(b"original").hexdigest())
    assert result.returncode != 0
    assert "does not match expected" in result.stdout

    wheel.write_bytes(b"original")
    result = _run_probe(probe, str(bundle), hashlib.sha256(b"original").hexdigest())
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"SHA={hashlib.sha256(b'original').hexdigest()}" in result.stdout


def test_installer_cross_checks_the_release_manifest_against_the_expected_digest(
    tmp_path: Path,
) -> None:
    probe = _wheel_probe(tmp_path)
    checkout = tmp_path / "checkout"
    wheel = checkout / "dist/2ebb8d337fca/gpu_fault_node_runtime-0.10.0.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"checkout-wheel")
    digest = hashlib.sha256(b"checkout-wheel").hexdigest()
    (checkout / "dist/current-release.json").write_text(
        json.dumps(
            {"wheel": wheel.relative_to(checkout).as_posix(), "wheel_sha256": digest}
        ),
        encoding="utf-8",
    )

    result = _run_probe(probe, str(checkout), digest)
    assert result.returncode == 0, result.stdout + result.stderr

    # A manifest that disagrees with the caller's digest is a tampered
    # checkout, not a tie-break the manifest wins.
    other = hashlib.sha256(b"other").hexdigest()
    result = _run_probe(probe, str(checkout), other)
    assert result.returncode != 0
    assert "does not match" in result.stdout


def test_installer_job_passes_the_wheel_digest_from_outside_the_bundle() -> None:
    job = _text(JOB)
    install_body = _inline_shell_bodies(job)[1]
    assert '--wheel-sha256 "${INSTALLER_ARTIFACT_SHA256}"' in install_body
    assert "--dependency-lock" not in install_body or "requirements/runtime.lock" in (
        install_body
    )
