from __future__ import annotations

import ast
import configparser
import fnmatch
import hashlib
import io
import json
import re
import subprocess
import sys
import zipfile
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request

import pytest
from coverage import CoverageData

from scripts import ci_coverage_gate, ci_gate_artifacts, ci_unit_gate
from scripts.ci_coverage_config import (
    DEPLOYMENT_SHARDS,
    PARTITIONED_DOMAINS,
    POSTGRES_SHARDS,
    RUNTIME_SHARDS,
    is_postgres_shard,
    logical_test_domain,
    shard_arguments,
    shard_partition,
)
from scripts.ci_pytest_evidence import aggregate_pytest_results, ci_context
from scripts.component_wheels import APPLICATION_COMPONENT_NAMES, component_modules
from tools.coverage_objectives import SCOPES, measured_files
from tools.pytest_case_reporter import partition_for_nodeid
from tools.pytest_result_identity import source_identity

ROOT = Path(__file__).resolve().parents[1]


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)


def _identity_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    files = {
        ".github/workflows/ci.yml": "name: CI\n",
        "Makefile": "coverage-shard:\n\t@true\n",
        "config/ci-unit-gate.json": (ROOT / "config/ci-unit-gate.json").read_text(
            encoding="utf-8"
        ),
        "deploy/control-plane/regional/probes/node_probe.py": "VALUE = 'probe'\n",
        "src/gpu_fault_release/regional_dns.py": "VALUE = 'dns'\n",
        "src/gpu_fault_release/regional_release_config.py": ("VALUE = 'release'\n"),
        "deploy/control-plane/tools/cleanup_state.py": "VALUE = 'tool'\n",
        "deploy/manifest.yaml": "kind: Deployment\n",
        "docs/guide.md": "guide\n",
        "requirements/build.lock": "build==1\n",
        "src/gpu_fault/admin/config.py": "VALUE = 'shared'\n",
        "src/gpu_fault/admin/only.py": "VALUE = 'deployment'\n",
        "src/gpu_fault/failure_domains.py": "VALUE = 'domains'\n",
        "src/gpu_fault/release_state_snapshot.py": "VALUE = 'snapshot'\n",
        "src/gpu_fault/runtime.py": (
            "def choose(flag):\n    if flag:\n        return 1\n    return 0\n"
        ),
        "scripts/e2e/regional/sample.py": "VALUE = 'regional'\n",
        "testcases/fault-scenarios.yaml": "schema_version: 1\n",
        "tests/admin/test_admin.py": "def test_admin(): pass\n",
        "tests/conftest.py": "VALUE = 'shared tests'\n",
        "tests/store/test_postgres_store.py": "def test_postgres(): pass\n",
        "tests/test_case_scheduler.py": "def test_scheduler(): pass\n",
        "tests/test_runtime.py": "def test_runtime(): pass\n",
        "tools/case_scheduler.py": (
            "def choose(flag):\n    if flag:\n        return 1\n    return 0\n"
        ),
        "uv.lock": "version = 1\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    return root


def _committed_root(tmp_path: Path) -> Path:
    root = _identity_root(tmp_path)
    _git(root, "config", "user.name", "CI")
    _git(root, "config", "user.email", "ci@example.com")
    _git(root, "commit", "-qm", "test")
    return root


def _identity(root: Path, shard: str) -> dict:
    return ci_coverage_gate.shard_identity(
        root,
        shard,
        postgres_image="sha256:postgres" if is_postgres_shard(shard) else "",
        distributions={"coverage": "7.16.0", "pytest": "9.1.1"},
        environment_identity={
            "machine": "x86_64",
            "postgres_image": ("sha256:postgres" if is_postgres_shard(shard) else ""),
            "python_cache_tag": "cpython-312",
            "python_implementation": "CPython",
            "python_version": "3.12.14",
            "runner_environment": "github-hosted",
            "runner_image_os": "ubuntu24",
            "runner_image_version": "20260901.1",
            "runner_label": "ubuntu-latest",
            "sysconfig_platform": "linux-x86_64",
        },
        pytest_workers="4",
    )


def test_repository_inputs_include_untracked_and_exclude_unstaged_deletions(
    tmp_path: Path,
) -> None:
    root = _identity_root(tmp_path)
    before = _identity(root, "runtime_0")["sha256"]
    deleted = root / "src/gpu_fault/runtime.py"
    deleted.unlink()
    added = root / "src/gpu_fault/new module.py"
    added.write_text("VALUE = 'new'\n", encoding="utf-8")

    inputs = ci_gate_artifacts.repository_files(root)

    assert deleted not in inputs, "deleted tracked files are not candidate inputs"
    assert added in inputs, "untracked replacement files must enter the candidate"
    assert _identity(root, "runtime_0")["sha256"] != before, (
        "a deletion and replacement must invalidate a reusable shard identity"
    )


@pytest.mark.parametrize("replacement", ["directory", "symlink", "broken-symlink"])
def test_repository_inputs_still_reject_non_regular_indexed_files(
    tmp_path: Path, replacement: str
) -> None:
    root = _identity_root(tmp_path)
    target = root / "src/gpu_fault/runtime.py"
    target.unlink()
    if replacement == "directory":
        target.mkdir()
    else:
        destination = tmp_path / "outside.py"
        if replacement == "symlink":
            destination.write_text("VALUE = 'outside'\n", encoding="utf-8")
        target.symlink_to(destination)

    with pytest.raises(
        ci_gate_artifacts.GateArtifactError, match="unsupported gate input"
    ):
        ci_gate_artifacts.repository_files(root)


def test_repository_inputs_reject_a_vanished_untracked_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _identity_root(tmp_path)
    transient = root / "untracked.py"
    transient.write_text("VALUE = 1\n", encoding="utf-8")
    run = subprocess.run

    def disappear(arguments, **kwargs):
        completed = run(arguments, **kwargs)
        if "--deleted" in arguments:
            transient.unlink()
        return completed

    monkeypatch.setattr(ci_gate_artifacts.subprocess, "run", disappear)
    with pytest.raises(
        ci_gate_artifacts.GateArtifactError, match="unsupported gate input"
    ):
        ci_gate_artifacts.repository_files(root)


def test_repository_inputs_refuse_an_unreadable_deletion_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _identity_root(tmp_path)
    run = subprocess.run

    def deny_deletions(arguments, **kwargs):
        if "--deleted" in arguments:
            return subprocess.CompletedProcess(arguments, 1, b"", b"denied")
        return run(arguments, **kwargs)

    monkeypatch.setattr(ci_gate_artifacts.subprocess, "run", deny_deletions)
    with pytest.raises(
        ci_gate_artifacts.GateArtifactError, match="deleted gate inputs"
    ):
        ci_gate_artifacts.repository_files(root)


def test_coverage_test_partition_is_complete_and_disjoint() -> None:
    counts = ci_coverage_gate.validate_test_partition(ROOT)
    config = ci_coverage_gate.load_config(ROOT)
    assigned = {
        relative
        for shard in ci_coverage_gate.SHARDS
        for relative in ci_coverage_gate.pytest_targets(ROOT, shard)
    }
    collectable = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").rglob("test_*.py")
    }

    assert all(counts[domain] > 0 for domain in ci_coverage_gate.TEST_DOMAINS), (
        "every logical coverage domain must retain at least one test file"
    )
    assert assigned == collectable - set(config["tests"]["coverage_excluded_files"])
    assert sum(counts.values()) == len(assigned)


_POSTGRES_URL_READ = re.compile(
    r"(?:getenv|environ\.get|environ)\s*[\(\[]\s*['\"]GPU_FAULT_TEST_POSTGRES_URL['\"]"
)


def _postgres_gated_test_files(root: Path = ROOT) -> set[str]:
    """Every collected test file whose cases skip without a Postgres URL.

    The environment-read matcher remains independent from the partition code.
    Inspect actual expressions and imports, not strings containing generated
    child code or comments. A dict literal or ``monkeypatch.setenv`` mentioning
    the name is not a gate.
    """

    trees = {
        path: ast.parse(path.read_text(encoding="utf-8"))
        for path in (root / "tests").rglob("*.py")
    }
    readers = set()
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and node.args:
                receiver, key, opening = node.func, node.args[0], "("
            elif isinstance(node, ast.Subscript):
                receiver, key, opening = node.value, node.slice, "["
            else:
                continue
            if (
                isinstance(key, ast.Constant)
                and key.value == "GPU_FAULT_TEST_POSTGRES_URL"
                and _POSTGRES_URL_READ.search(
                    f"{ast.unparse(receiver)}{opening}{key.value!r}"
                )
            ):
                readers.add(path)
                break
    reader_modules = {
        ".".join(path.relative_to(root).with_suffix("").parts) for path in readers
    }
    gated: set[str] = set()
    for path, tree in trees.items():
        if not path.name.startswith("test_"):
            continue
        relative = path.relative_to(root).as_posix()
        if path in readers:
            gated.add(relative)
            continue
        if any(
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module in reader_modules
            for node in ast.walk(tree)
        ):
            gated.add(relative)
    return gated


def _makefile_postgres_tests() -> set[str]:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    block = makefile.split("POSTGRES_TESTS = \\\n", 1)[1].split("\n\n", 1)[0]
    entries = set()
    for line in block.splitlines():
        entry = line.strip().rstrip("\\").strip()
        if entry and not entry.startswith("#") and "=" not in entry:
            entries.add(entry)
    return entries


def test_every_postgres_gated_test_file_runs_in_the_postgres_shard() -> None:
    # G-1: the CI postgres shard used to name four files while sixty-odd test
    # modules skip without GPU_FAULT_TEST_POSTGRES_URL, and every other shard
    # clears the variable -- so an EXPLAIN or lock-order assertion could go
    # red in the repository without any CI job ever running it.
    config = ci_coverage_gate.load_config(ROOT)
    gated = _postgres_gated_test_files() - set(
        config["tests"]["coverage_excluded_files"]
    )
    shard = set(ci_coverage_gate.pytest_targets(ROOT, "postgres"))
    makefile = _makefile_postgres_tests()

    assert len(gated) > 40, sorted(gated)
    assert gated <= shard, sorted(gated - shard)
    assert makefile == shard, {
        "makefile_only": sorted(makefile - shard),
        "shard_only": sorted(shard - makefile),
    }


def test_native_activation_and_history_partitions_preserve_shared_backends() -> None:
    native = {
        "tests/metrics/test_kernel_pcie_composite_postgres.py",
        "tests/orchestration/test_provider_correlation_postgres.py",
        "tests/store/test_postgres_activation_inhibition.py",
        "tests/store/test_postgres_health_signal_semantics.py",
        "tests/store/test_postgres_terminal_quarantine.py",
    }
    history = "tests/store/test_job_recovery_history.py"
    shared = {
        "tests/metrics/test_kernel_pcie_composite.py",
        "tests/orchestration/test_provider_correlation_arbitration.py",
        "tests/store/test_activation_inhibition_claims.py",
        "tests/store/test_activation_inhibition_query.py",
        "tests/store/test_remote_command_claim_cancellation.py",
        "tests/store/test_health_signal_semantics.py",
        "tests/orchestration/test_terminal_quarantine_constraints.py",
    }
    config = ci_coverage_gate.load_config(ROOT)
    postgres = set(ci_coverage_gate.pytest_targets(ROOT, "postgres"))
    makefile = _makefile_postgres_tests()

    # Module-alias fixtures are not discovered by the environment-read matcher.
    assert native <= set(config["tests"]["postgres_files"]), (
        "native PostgreSQL fixtures need explicit serial shard ownership"
    )
    assert native | {history} <= postgres & makefile, (
        "native SQL cases and all three history backends must run in the serial gate"
    )
    assert shared.isdisjoint(postgres | makefile), (
        "native wrappers must not move their Memory/SQLite helpers into PostgreSQL"
    )
    for shard in ci_coverage_gate.SHARDS:
        if is_postgres_shard(shard):
            continue
        targets = set(ci_coverage_gate.pytest_targets(ROOT, shard))
        assert (native | {history}).isdisjoint(targets), (
            "PostgreSQL-dependent modules must not leak into ordinary coverage"
        )
        if logical_test_domain(shard) == "runtime":
            assert shared <= targets, (
                "every runtime nodeid partition must retain the shared backend tests"
            )


def test_postgres_gate_detection_reads_the_environment_not_mentions(
    tmp_path: Path,
) -> None:
    root = _identity_root(tmp_path)
    files = {
        "tests/store/_pg_support.py": (
            "import os\nPOSTGRES_URL = os.getenv('GPU_FAULT_TEST_POSTGRES_URL')\n"
        ),
        "tests/store/test_direct.py": (
            "import os, pytest\n"
            "if not os.environ.get('GPU_FAULT_TEST_POSTGRES_URL'):\n"
            "    pytest.skip('x', allow_module_level=True)\n"
        ),
        "tests/store/test_indirect.py": (
            "from tests.store._pg_support import POSTGRES_URL\n"
        ),
        "tests/regional/test_baseline.py": (
            "import os\nURL = os.environ['GPU_FAULT_TEST_POSTGRES_URL']\n"
        ),
        "tests/test_mention.py": (
            "def test_env(monkeypatch):\n"
            "    monkeypatch.setenv('GPU_FAULT_TEST_POSTGRES_URL', 'postgresql://x')\n"
        ),
        "tests/test_fstring_gate.py": (
            "import os\nVALUE = f\"{os.getenv('GPU_FAULT_TEST_POSTGRES_URL')}\"\n"
        ),
        "tests/test_generated_child.py": "PROGRAM = "
        + repr(
            "import os\n"
            "URL = os.environ['GPU_FAULT_TEST_POSTGRES_URL']\n"
            "from tests.store._pg_support import POSTGRES_URL\n"
        )
        + "\n",
        "tests/test_commented_reads.py": (
            "# os.getenv('GPU_FAULT_TEST_POSTGRES_URL')\n"
            "# from tests.store._pg_support import POSTGRES_URL\n"
        ),
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(root, "add", ".")

    postgres = set(ci_coverage_gate.pytest_targets(root, "postgres"))
    runtime = set(ci_coverage_gate.pytest_targets(root, "runtime_0"))
    deployment = set(ci_coverage_gate.pytest_targets(root, "deployment"))

    assert {
        "tests/store/test_direct.py",
        "tests/store/test_indirect.py",
        "tests/regional/test_baseline.py",
        "tests/store/test_postgres_store.py",
    } <= postgres
    assert "tests/test_mention.py" in runtime
    assert "tests/regional/test_baseline.py" not in deployment
    assert "tests/test_mention.py" not in postgres
    observed = _postgres_gated_test_files(root)
    actual_readers = {
        "tests/store/test_direct.py",
        "tests/store/test_indirect.py",
        "tests/regional/test_baseline.py",
        "tests/test_fstring_gate.py",
    }
    assert actual_readers <= observed and actual_readers <= postgres, (
        "both independent checks must retain executable and imported database reads"
    )
    for relative in (
        "tests/test_mention.py",
        "tests/test_generated_child.py",
        "tests/test_commented_reads.py",
    ):
        assert relative not in observed and relative not in postgres, (
            f"{relative} does not depend on an ambient PostgreSQL allocation"
        )
        assert relative in runtime, f"{relative} must remain an ordinary test"


@pytest.mark.parametrize("domain", sorted(PARTITIONED_DOMAINS))
def test_nodeid_partitions_are_stable_disjoint_and_complete(domain: str) -> None:
    nodeids = [f"tests/test_{domain}.py::test_case[{index}]" for index in range(1000)]
    count = PARTITIONED_DOMAINS[domain]
    shards = [shard for shard in ci_coverage_gate.SHARDS if shard.startswith(domain)]
    partitions = [
        {nodeid for nodeid in nodeids if partition_for_nodeid(nodeid, count) == index}
        for index in range(count)
    ]

    assert [shard_partition(shard) for shard in shards] == [
        (count, index) for index in range(count)
    ], "every physical shard of the domain must own exactly one partition slot"
    assert set.union(*partitions) == set(nodeids)
    assert sum(len(partition) for partition in partitions) == len(nodeids)
    assert max(map(len, partitions)) - min(map(len, partitions)) < 80 * count / 3


def _collected_by(arguments, test_files: list[str]) -> set[str]:
    """The test files pytest would collect for ``arguments``."""
    collected = set()
    for path in arguments.paths:
        if path.endswith(".py"):
            collected.add(path)
        else:
            collected.update(f for f in test_files if f.startswith(path + "/"))
    return collected - set(arguments.ignores)


@pytest.mark.parametrize("shard", ci_coverage_gate.SHARDS)
def test_shard_arguments_collect_exactly_the_shard_targets(shard: str) -> None:
    """Directory arguments with ``--ignore`` must select the same files as the
    explicit target list: nothing dropped, nothing from another shard."""
    test_files = sorted(
        path.relative_to(ROOT).as_posix()
        for path in ci_gate_artifacts.repository_files(ROOT)
        if path.relative_to(ROOT).as_posix().startswith("tests/")
        and path.name.startswith("test_")
        and path.suffix == ".py"
    )
    arguments = shard_arguments(ROOT, shard)
    targets = set(ci_coverage_gate.pytest_targets(ROOT, shard))
    directories = [path for path in arguments.paths if not path.endswith(".py")]

    assert _collected_by(arguments, test_files) == targets, (
        "the argument set must collect exactly the shard's test files"
    )
    assert set(arguments.ignores) <= {
        f for d in directories for f in test_files if f.startswith(d + "/")
    }, "every ignore must lie below a directory argument"
    assert not any(
        f.startswith(d + "/") for f in arguments.paths for d in directories
    ), "a file argument must not repeat a directory argument"
    assert arguments == shard_arguments(ROOT, shard), "arguments are deterministic"
    if logical_test_domain(shard) == "deployment":
        assert "tests/regional" in directories, (
            "the regional directory is the case this optimisation exists for"
        )


def test_shard_names_resolve_to_their_logical_domain() -> None:
    assert logical_test_domain("fault_runner") == "fault_runner"
    assert shard_partition("fault_runner") is None
    assert {logical_test_domain(shard) for shard in RUNTIME_SHARDS} == {"runtime"}
    assert {logical_test_domain(shard) for shard in DEPLOYMENT_SHARDS} == {"deployment"}
    assert all(is_postgres_shard(shard) for shard in POSTGRES_SHARDS), (
        "every postgres partition must be recognised as a PostgreSQL shard"
    )
    assert not any(
        is_postgres_shard(shard) for shard in (*RUNTIME_SHARDS, *DEPLOYMENT_SHARDS)
    ), "no runtime or deployment partition may be treated as PostgreSQL"
    with pytest.raises(ci_coverage_gate.CoverageGateError, match="unknown"):
        shard_partition("postgres")


def test_coverage_excludes_the_static_documentation_and_ci_tests() -> None:
    config = ci_coverage_gate.load_config(ROOT)
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    docs_block = makefile.split("DOCUMENTATION_TESTS = \\\n", 1)[1].split(
        "\nCI_TOOLING_TESTS =", 1
    )[0]
    docs = {
        line.strip().rstrip("\\").strip()
        for line in docs_block.splitlines()
        if line.strip()
    }
    tooling_block = makefile.split("CI_TOOLING_TESTS = \\\n", 1)[1].split(
        "\nPOSTGRES_TESTS =", 1
    )[0]
    tooling = {
        line.strip().rstrip("\\").strip()
        for line in tooling_block.splitlines()
        if line.strip()
    }
    promql_block = makefile.split("PROMQL_TESTS = \\\n", 1)[1].split(
        "\nDOCUMENTATION_TESTS =", 1
    )[0]
    promql = {
        line.strip().rstrip("\\").strip()
        for line in promql_block.splitlines()
        if line.strip()
    }

    assert promql == {
        "tests/metrics/test_closed_loop_promql.py",
        "tests/metrics/test_observability_evidence_promql.py",
        "tests/metrics/test_review_dashboard_promql.py",
        "tests/test_alert_rules_promtool.py",
    }, "the pinned static gate owns every native PromQL test file"
    assert docs | tooling | promql <= set(config["tests"]["coverage_excluded_files"])


def test_deployment_only_coverage_scope_matches_distribution_split() -> None:
    application_modules = {
        module
        for component in APPLICATION_COMPONENT_NAMES
        for module in component_modules(component)
    }
    # The deploy-host wheel also carries the release engine it imports
    # (``gpu_fault_release``); those modules are the deployment-only *root*
    # files below, not application-package files.
    deploy_host_only = {
        module
        for module in set(component_modules("deploy_host")) - application_modules
        if module == "gpu_fault" or module.startswith("gpu_fault.")
    }
    expected_package_files = {
        (
            "src/"
            + module.replace(".", "/")
            + (
                "/__init__.py"
                if (ROOT / ("src/" + module.replace(".", "/"))).is_dir()
                else ".py"
            )
        )
        for module in deploy_host_only
    }
    deploy_roots = set(
        ci_coverage_gate.load_config(ROOT)["coverage"]["deployment_only_sources"]
    )
    # The release orchestrator (``src/gpu_fault_release``) and the control-plane
    # tools are not application wheels: they are the modules sitting directly in
    # the deployment-only source roots, whichever tree the root lives in.
    # ``probes/`` is left out on purpose -- most probes are only parsed by
    # tests, never executed.
    expected_root_files = {
        path.relative_to(ROOT).as_posix()
        for path in ci_gate_artifacts.repository_files(ROOT)
        if path.suffix == ".py"
        and path.parent.relative_to(ROOT).as_posix() in deploy_roots
    }
    assert "src/gpu_fault_release" in deploy_roots
    assert "src/gpu_fault" not in deploy_roots

    measured = set(ci_coverage_gate.deployment_only_source_files(ROOT))
    assert {item for item in measured if item.startswith("src/gpu_fault/")} == (
        expected_package_files
    )
    assert {
        item for item in measured if not item.startswith("src/gpu_fault/")
    } == expected_root_files
    assert not any("/probes/" in item for item in measured), "probes are not measured"


def test_coverage_config_names_the_deploy_roots_as_deployment_only_sources() -> None:
    """Both objective scopes are measured; deploy roots remain deployment-only."""

    config = ci_coverage_gate.load_config(ROOT)
    coverage = config["coverage"]

    assert config["schema_version"] == 5
    assert "source" not in coverage
    assert coverage["sources"] == [
        "src/gpu_fault",
        "src/gpu_fault_release",
        "deploy/control-plane/tools",
        "scripts/e2e/regional",
        "tools",
    ]
    assert coverage["deployment_only_sources"] == [
        "src/gpu_fault_release",
        "deploy/control-plane/tools",
    ]
    assert all((ROOT / root).is_dir() for root in coverage["sources"]), (
        "every coverage source root must exist"
    )


def _mutated_config(mutate) -> dict:
    config = ci_coverage_gate.load_config(ROOT)
    mutate(config)
    return config


def _legacy_source_key(config: dict) -> None:
    config["coverage"]["source"] = "src/gpu_fault"


def _empty_sources(config: dict) -> None:
    config["coverage"]["sources"] = []


def _foreign_deployment_only_source(config: dict) -> None:
    config["coverage"]["deployment_only_sources"] = ["deploy/dataplane"]


def _every_source_deployment_only(config: dict) -> None:
    config["coverage"]["deployment_only_sources"] = list(config["coverage"]["sources"])


def _previous_schema(config: dict) -> None:
    config["schema_version"] = 2


@pytest.mark.parametrize(
    "mutate",
    [
        _legacy_source_key,
        _empty_sources,
        _foreign_deployment_only_source,
        _every_source_deployment_only,
        _previous_schema,
    ],
)
def test_coverage_config_rejects_an_incomplete_source_declaration(
    tmp_path: Path, mutate
) -> None:
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    (root / "config/ci-unit-gate.json").write_text(
        json.dumps(_mutated_config(mutate)), encoding="utf-8"
    )

    with pytest.raises(ci_coverage_gate.CoverageGateError):
        ci_coverage_gate.load_config(root)


def _coverage_ini_list(path: Path, option: str) -> set[str]:
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    if not parser.has_option("run", option):
        return set()
    return {
        line.strip() for line in parser.get("run", option).splitlines() if line.strip()
    }


@pytest.mark.parametrize("shard", sorted(ci_coverage_gate.SHARDS))
def test_only_the_deployment_shard_measures_the_deploy_roots(
    tmp_path: Path, shard: str
) -> None:
    """Runtime shards measure runtime and runner code without deployment roots.

    Their identities exclude ``deploy/``, so a runtime shard that measured the
    release orchestrator could be reused against a changed orchestrator.
    """

    root = _identity_root(tmp_path)
    output = tmp_path / "coverage.ini"

    ci_coverage_gate.write_coverage_config(root, shard, output)

    sources = _coverage_ini_list(output, "source")
    omitted = _coverage_ini_list(output, "omit")
    protocol = _identity(root, shard)["protocol"]
    assert "coverage_source" not in protocol
    assert set(protocol["coverage_sources"]) == sources
    if logical_test_domain(shard) == "deployment":
        assert sources == {
            "src/gpu_fault",
            "src/gpu_fault_release",
            "deploy/control-plane/tools",
            "scripts/e2e/regional",
            "tools",
        }
        assert omitted == set()
    else:
        assert sources == {"src/gpu_fault", "scripts/e2e/regional", "tools"}
        assert "src/gpu_fault/admin/only.py" in omitted
        assert not any(item.startswith("deploy/") for item in omitted), (
            "a root that is not measured has nothing to omit"
        )


def test_non_deployment_shard_measuring_deploy_source_is_rejected(
    tmp_path: Path,
) -> None:
    root = _identity_root(tmp_path)
    data_path = tmp_path / "coverage-data"
    data = CoverageData(basename=str(data_path))
    data.add_lines(
        {
            "src/gpu_fault/runtime.py": {1},
            "src/gpu_fault_release/regional_release_config.py": {1},
        }
    )
    data.write()

    with pytest.raises(
        ci_coverage_gate.CoverageGateError, match="deploy-host-only source"
    ):
        ci_coverage_gate.verify_coverage_data(root, "runtime_0", data_path)


@pytest.mark.parametrize(
    ("relative", "affected"),
    [
        ("requirements/build.lock", set(ci_coverage_gate.SHARDS)),
        ("Makefile", set(ci_coverage_gate.SHARDS)),
        ("src/gpu_fault/runtime.py", set(ci_coverage_gate.SHARDS)),
        ("src/gpu_fault/admin/only.py", set(DEPLOYMENT_SHARDS)),
        ("src/gpu_fault_release/regional_release_config.py", set(DEPLOYMENT_SHARDS)),
        ("deploy/control-plane/regional/probes/node_probe.py", set(DEPLOYMENT_SHARDS)),
        ("deploy/control-plane/tools/cleanup_state.py", set(DEPLOYMENT_SHARDS)),
        ("tests/test_runtime.py", set(RUNTIME_SHARDS)),
        ("tests/admin/test_admin.py", set(DEPLOYMENT_SHARDS)),
        ("tests/test_case_scheduler.py", {"fault_runner"}),
        ("tests/store/test_postgres_store.py", set(POSTGRES_SHARDS)),
        ("testcases/fault-scenarios.yaml", set(ci_coverage_gate.SHARDS)),
        ("tools/case_scheduler.py", set(ci_coverage_gate.SHARDS)),
        ("scripts/e2e/regional/sample.py", set(ci_coverage_gate.SHARDS)),
        ("tests/conftest.py", set(ci_coverage_gate.SHARDS)),
        ("docs/guide.md", set()),
        (".github/workflows/ci.yml", set()),
    ],
)
def test_shard_identity_changes_only_affected_domains(
    tmp_path: Path, relative: str, affected: set[str]
) -> None:
    root = _identity_root(tmp_path)
    before = {
        shard: _identity(root, shard)["sha256"] for shard in ci_coverage_gate.SHARDS
    }
    path = root / relative
    path.write_text(path.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")
    after = {
        shard: _identity(root, shard)["sha256"] for shard in ci_coverage_gate.SHARDS
    }

    assert {
        shard for shard in ci_coverage_gate.SHARDS if before[shard] != after[shard]
    } == affected


def _write_coverage_data(root: Path, path: Path, shard: str) -> None:
    data = CoverageData(basename=str(path))
    measured = [
        root / "src/gpu_fault/runtime.py",
        root / "tools/case_scheduler.py",
        root / "scripts/e2e/regional/sample.py",
        root / "src/gpu_fault/admin/config.py",
    ]
    if logical_test_domain(shard) == "deployment":
        # Every deployment-only module has a module floor, and a floor that
        # matches no measured file is itself a failure, so the shard that owns
        # them has to measure all of them.
        measured += [
            root / "src/gpu_fault/admin/only.py",
            root / "src/gpu_fault/failure_domains.py",
            root / "src/gpu_fault/release_state_snapshot.py",
            root / "src/gpu_fault_release/regional_release_config.py",
            root / "src/gpu_fault_release/regional_dns.py",
            root / "deploy/control-plane/tools/cleanup_state.py",
        ]
    data.add_arcs(
        {
            item.relative_to(root).as_posix(): (
                {(-1, 1), (1, -1), (-1, 2), (2, 3), (2, 4), (3, -1), (4, -1)}
                if item.name in {"runtime.py", "case_scheduler.py"}
                else {(-1, 1), (1, -1)}
            )
            for item in measured
        }
    )
    data.write()


def _pytest_value(root: Path, identity: dict, *, suite: str = "pytest") -> dict:
    shard = identity["shard"]
    targets = list(ci_coverage_gate.pytest_targets(root, shard))
    slot = shard_partition(shard)
    partition = list(slot) if slot is not None else None
    discovered = sorted(
        f"{target}::test_pass[{index}]"
        for target in targets
        for index in range(4 * partition[0] if partition is not None else 1)
    )
    selected = [
        nodeid
        for nodeid in discovered
        if partition is None
        or partition_for_nodeid(nodeid, partition[0]) == partition[1]
    ]
    original = source_identity(root)
    return {
        "schema_version": 1,
        "source_identity": original,
        "session": {
            "source_identity": original,
            "started_at": "2026-09-14T01:00:00+00:00",
            "finished_at": "2026-09-14T01:01:00+00:00",
            "exitstatus": 0,
            "collected_nodeids": selected,
            "discovered_nodeids": discovered,
            "collected_files": targets,
            "collection_errors": [],
            "collection_skips": [],
            "ci_context": ci_context(identity, suite),
            "selection": {
                "targets": list(shard_arguments(root, shard).paths),
                "partition": partition,
                "keyword": "",
                "markexpr": "",
                "deselect": [],
                "numprocesses": 0 if is_postgres_shard(shard) else 4,
                "stress_workers": "8" if suite == "postgres_stress" else "",
                "stress_rounds": "40" if suite == "postgres_stress" else "",
            },
        },
        "records": {
            nodeid: {
                "duration_seconds": 0.01,
                "output": "",
                "phases": {"setup": "passed", "call": "passed", "teardown": "passed"},
                "status": "PASS",
            }
            for nodeid in selected
        },
    }


def _write_pytest_results(
    path: Path, *, root: Path, identity: dict, suite: str = "pytest"
) -> None:
    path.write_text(
        json.dumps(_pytest_value(root, identity, suite=suite)), encoding="utf-8"
    )


def _write_durations(path: Path, shard: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shard": shard,
                "pytest": {"record_count": 1, "slowest": [], "wall_seconds": 1.0},
                "postgres_stress": (
                    {"record_count": 1, "slowest": [], "wall_seconds": 1.0}
                    if is_postgres_shard(shard)
                    else None
                ),
            }
        ),
        encoding="utf-8",
    )


def _set_main_environment(monkeypatch: pytest.MonkeyPatch, *, run_id: str) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repository")
    monkeypatch.setenv(
        "GITHUB_WORKFLOW_REF",
        "owner/repository/.github/workflows/ci.yml@refs/heads/main",
    )
    monkeypatch.setenv("GITHUB_RUN_ID", run_id)


def _build_shard(
    root: Path,
    destination: Path,
    shard: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_id: str = "123",
) -> dict:
    _set_main_environment(monkeypatch, run_id=run_id)
    destination.mkdir(parents=True, exist_ok=True)
    coverage_data = destination / ci_coverage_gate.COVERAGE_DATA_NAME
    pytest_results = destination / ci_coverage_gate.PYTEST_RESULTS_NAME
    durations = destination / ci_coverage_gate.DURATIONS_NAME
    stress = (
        destination / ci_coverage_gate.STRESS_RESULTS_NAME
        if is_postgres_shard(shard)
        else None
    )
    _write_coverage_data(root, coverage_data, shard)
    identity = _identity(root, shard)
    _write_pytest_results(pytest_results, root=root, identity=identity)
    _write_durations(durations, shard)
    if stress is not None:
        _write_pytest_results(
            stress, root=root, identity=identity, suite="postgres_stress"
        )
    gate = ci_coverage_gate.build_shard_gate(
        root,
        destination,
        identity=identity,
        coverage_data=coverage_data,
        pytest_results=pytest_results,
        durations=durations,
        stress_results=stress,
    )
    (destination / ci_coverage_gate.BUNDLE_NAME).write_text("{}", encoding="utf-8")
    return gate


def test_coverage_shard_gate_binds_evidence_and_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _committed_root(tmp_path)
    artifact_root = tmp_path / "artifact"
    gate = _build_shard(root, artifact_root, "runtime_0", monkeypatch)

    assert (
        ci_coverage_gate.verify_shard_gate(
            artifact_root / ci_coverage_gate.GATE_NAME,
            artifact_root,
            expected_identity=gate["identity"]["sha256"],
            expected_shard="runtime_0",
            require_trusted=True,
            source_root=root,
        )
        == gate
    )
    (artifact_root / ci_coverage_gate.DURATIONS_NAME).write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(
        ci_gate_artifacts.GateArtifactError, match="evidence does not match"
    ):
        ci_coverage_gate.verify_shard_gate(
            artifact_root / ci_coverage_gate.GATE_NAME, artifact_root, source_root=root
        )


def test_restore_reusable_shard_resigns_current_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _committed_root(tmp_path)
    source = tmp_path / "source"
    gate = _build_shard(root, source, "runtime_0", monkeypatch, run_id="99")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as value:
        for path in sorted(source.iterdir()):
            value.write(path, path.name)
    data = archive.getvalue()
    artifact = {
        "archive_download_url": "https://example.invalid/artifact",
        "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
    }
    monkeypatch.setattr(
        ci_coverage_gate,
        "find_reusable_artifact",
        lambda **_kwargs: (artifact, {"id": 99}),
    )
    monkeypatch.setattr(ci_coverage_gate, "download", lambda *_args: data)
    monkeypatch.setattr(ci_coverage_gate, "_is_ancestor", lambda *_args: True)

    destination = tmp_path / "restored"
    result = ci_coverage_gate.restore_reusable_shard(
        root=root,
        repository="owner/repository",
        token="token",
        identity=gate["identity"],
        destination=destination,
        current_run_id=100,
    )

    assert result["reused"] == "true"
    assert (destination / ci_coverage_gate.BASE_GATE_NAME).is_file(), (
        "restored evidence must retain the signed base gate"
    )
    _set_main_environment(monkeypatch, run_id="100")
    current = ci_coverage_gate.build_shard_gate(
        root,
        destination,
        identity=gate["identity"],
        coverage_data=destination / ci_coverage_gate.COVERAGE_DATA_NAME,
        pytest_results=destination / ci_coverage_gate.PYTEST_RESULTS_NAME,
        durations=destination / ci_coverage_gate.DURATIONS_NAME,
    )
    assert current["producer"]["run_id"] == "100"
    assert current["reused_from"]["producer_run_id"] == "99"


def test_restore_network_failure_falls_back_to_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _identity_root(tmp_path)
    identity = _identity(root, "runtime_0")
    monkeypatch.setattr(
        ci_coverage_gate,
        "find_reusable_artifact",
        lambda **_kwargs: (_ for _ in ()).throw(
            HTTPError("https://api.github.com/example", 503, "unavailable", {}, None)
        ),
    )

    result = ci_coverage_gate.restore_reusable_shard(
        root=root,
        repository="owner/repository",
        token="token",
        identity=identity,
        destination=tmp_path / "restored",
        current_run_id=100,
    )

    assert result["reused"] == "false"


def _unit_evidence(
    root: Path, *, source_root: Path, gates: dict
) -> tuple[Path, Path, Path, Path]:
    coverage = root / "coverage.json"
    coverage.write_text(
        json.dumps(
            {
                "meta": {"branch_coverage": True},
                "files": {
                    name: {
                        "summary": {
                            "num_statements": 100,
                            "covered_lines": 100,
                            "missing_lines": 0,
                            "num_branches": 100,
                            "covered_branches": 100,
                            "missing_branches": 0,
                        }
                    }
                    for scope in SCOPES
                    for name in measured_files(source_root, scope)
                },
                "totals": {"percent_covered": 100.0},
            }
        ),
        encoding="utf-8",
    )
    pytest_results = root / "pytest-case-results.json"
    pytest_results.write_text(
        json.dumps(
            aggregate_pytest_results(
                source_root,
                gates,
                resolve_identity=ci_coverage_gate.current_shard_identity,
            )
        ),
        encoding="utf-8",
    )
    fault_report = root / "fault-report.json"
    fault_report.write_text(
        json.dumps({"schema_version": 2, "verdict": "PASS", "results": []}),
        encoding="utf-8",
    )
    durations = root / "test-durations.json"
    durations.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "shards": {
                    shard: {
                        "pytest": {"wall_seconds": 1.0},
                        "postgres_stress": None,
                        "producer_run_id": "123",
                        "reused": False,
                    }
                    for shard in ci_coverage_gate.SHARDS
                },
                "slowest": [],
            }
        ),
        encoding="utf-8",
    )
    return coverage, pytest_results, fault_report, durations


def test_unit_gate_aggregates_all_signed_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _committed_root(tmp_path)
    shards_root = tmp_path / "shards"
    for shard in ci_coverage_gate.SHARDS:
        _build_shard(root, shards_root / shard, shard, monkeypatch)
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    coverage, pytest_results, fault_report, durations = _unit_evidence(
        evidence_root,
        source_root=root,
        gates=ci_coverage_gate.discover_shard_gates(shards_root, source_root=root),
    )
    artifact_root = tmp_path / "unit"
    _set_main_environment(monkeypatch, run_id="123")

    gate = ci_unit_gate.build_unit_gate(
        root,
        artifact_root,
        shards_root=shards_root,
        coverage_summary=coverage,
        pytest_results=pytest_results,
        fault_report=fault_report,
        durations=durations,
        require_run_id="123",
    )

    assert set(gate["shards"]) == set(ci_coverage_gate.SHARDS)
    assert (
        ci_unit_gate.verify_unit_gate(
            artifact_root / ci_unit_gate.GATE_NAME,
            artifact_root,
            expected_identity=gate["identity"]["sha256"],
            source_root=root,
        )
        == gate
    )
    bundle = artifact_root / "shards" / "runtime_0" / ci_coverage_gate.BUNDLE_NAME
    bundle.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ci_unit_gate.UnitGateError, match="artifact does not match"):
        ci_unit_gate.verify_unit_gate(
            artifact_root / ci_unit_gate.GATE_NAME, artifact_root, source_root=root
        )


def test_coverage_combine_uses_the_configured_data_file_basename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _committed_root(tmp_path)
    shards_root = tmp_path / "shards"
    for shard in ci_coverage_gate.SHARDS:
        _build_shard(root, shards_root / shard, shard, monkeypatch)
    output_root = tmp_path / "combined"

    ci_coverage_gate.combine_shards(
        root=root,
        shards_root=shards_root,
        output_root=output_root,
        python=sys.executable,
        require_run_id="123",
    )

    summary = json.loads((output_root / "coverage.json").read_text(encoding="utf-8"))
    merged = json.loads(
        (output_root / "pytest-case-results.json").read_text(encoding="utf-8")
    )
    assert summary["totals"]["percent_covered"] >= 78
    assert summary["meta"]["branch_coverage"] is True, (
        "branch-enabled shard proofs must combine real branch data"
    )
    expected_records = {
        nodeid
        for shard in ci_coverage_gate.SHARDS
        for nodeid in _pytest_value(root, _identity(root, shard))["records"]
    }
    assert set(merged["records"]) == expected_records and len(expected_records) == (
        sum(
            4 * count * len(ci_coverage_gate.pytest_targets(root, domain))
            for domain, count in PARTITIONED_DOMAINS.items()
        )
        + len(ci_coverage_gate.pytest_targets(root, "fault_runner"))
    ), "every partition slice of every domain survives the merge exactly once"


def test_reusable_artifact_requires_a_successful_producer_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = {
        "artifacts": [
            {
                "created_at": "2026-09-02T01:00:00Z",
                "expired": False,
                "id": 1,
                "workflow_run": {"head_branch": "main", "id": 10},
            },
            {
                "created_at": "2026-09-01T01:00:00Z",
                "expired": False,
                "id": 2,
                "workflow_run": {"head_branch": "main", "id": 11},
            },
        ]
    }

    def api(url: str, _token: str):
        if "actions/artifacts" in url:
            return artifacts
        if url.endswith("/jobs?per_page=100"):
            run_id = int(url.rsplit("/", 2)[1])
            return {
                "jobs": [
                    {
                        "conclusion": "failure" if run_id == 10 else "success",
                        "name": "coverage-runtime_0",
                        "status": "completed",
                    }
                ]
            }
        run_id = int(url.rsplit("/", 1)[1])
        return {
            "conclusion": "failure",
            "event": "push",
            "head_branch": "main",
            "id": run_id,
            "path": ".github/workflows/ci.yml",
            "status": "completed",
        }

    monkeypatch.setattr(ci_gate_artifacts, "api_json", api)

    found = ci_gate_artifacts.find_reusable_artifact(
        repository="owner/repository",
        token="token",
        name="gpu-fault-coverage-runtime-" + "a" * 64,
        current_run_id=None,
        required_job_name="coverage-runtime_0",
    )

    assert found is not None
    assert found[0]["id"] == 2


def test_artifact_redirect_drops_authentication_on_cross_origin() -> None:
    request = Request(
        "https://api.github.com/repos/owner/repository/actions/artifacts/1/zip",
        headers={"Authorization": "Bearer example"},
    )
    redirected = ci_gate_artifacts.ArtifactRedirectHandler().redirect_request(
        request,
        None,
        302,
        "Found",
        Message(),
        "https://objects.example.invalid/artifact.zip?signature=example",
    )

    assert redirected is not None
    assert redirected.get_header("Authorization") is None


def _coverage_report(files: dict[str, tuple[int, int]]) -> dict:
    """A coverage JSON report where each file covers ``reached`` of ``total``."""

    return {
        "files": {
            relative: {
                "summary": {
                    "num_statements": total,
                    "covered_lines": reached,
                    "num_branches": 0,
                    "covered_branches": 0,
                    "missing_branches": 0,
                    "missing_lines": total - reached,
                    "num_partial_branches": 0,
                }
            }
            for relative, (reached, total) in files.items()
        }
    }


def _write_report(path: Path, files: dict[str, tuple[int, int]]) -> Path:
    _git(path.parent, "init", "-q")
    for relative in files:
        source = path.parent / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("VALUE = 1\n", encoding="utf-8")
    path.write_text(json.dumps(_coverage_report(files)), encoding="utf-8")
    return path


def _module_floors_config(entries: list[dict]) -> dict:
    config = ci_coverage_gate.load_config(ROOT)
    config["coverage"]["module_floors"] = entries
    return config


def test_module_floors_cover_every_deployment_only_source_file() -> None:
    """No deployment-only module may sit outside a floor group.

    These files are omitted from every runtime shard, so the repository floor
    never sees them. A new one landing outside a group would be unmeasured in
    practice while every gate still reported green.
    """

    config = ci_coverage_gate.load_config(ROOT)
    patterns = [
        pattern
        for entry in config["coverage"]["module_floors"]
        for pattern in entry["globs"]
    ]

    uncovered = [
        relative
        for relative in ci_coverage_gate.deployment_only_source_files(ROOT)
        if not any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns)
    ]

    assert uncovered == [], "deployment-only modules without a module floor"


def test_module_floors_report_group_and_file_shortfalls(tmp_path: Path) -> None:
    config = _module_floors_config(
        [
            {
                "id": "family",
                "description": "test group",
                "globs": ["src/gpu_fault/admin/*.py"],
                "group_floor": 60,
                "file_floor": 30,
            }
        ]
    )
    report = _write_report(
        tmp_path / "coverage.json",
        {
            "src/gpu_fault/admin/good.py": (90, 100),
            "src/gpu_fault/admin/bad.py": (10, 100),
            "src/gpu_fault/runtime.py": (0, 100),
        },
    )

    violations = ci_coverage_gate.module_floor_violations(
        report, config=config, root=tmp_path
    )

    assert violations == [
        "src/gpu_fault/admin/bad.py covers 10.0% of 100 measurable points, "
        "below the family file floor of 30%",
        "module group family covers 50.0%, below its group floor of 60%",
    ]


def test_module_floors_pass_when_the_family_clears_both_floors(tmp_path: Path) -> None:
    config = _module_floors_config(
        [
            {
                "id": "family",
                "description": "test group",
                "globs": ["src/gpu_fault/admin/*.py"],
                "group_floor": 60,
                "file_floor": 30,
            }
        ]
    )
    report = _write_report(
        tmp_path / "coverage.json",
        {
            "src/gpu_fault/admin/good.py": (90, 100),
            "src/gpu_fault/admin/thin.py": (31, 100),
            # Nothing to measure: only imports and constants, so no floor applies.
            "src/gpu_fault/admin/constants.py": (0, 0),
        },
    )

    assert (
        ci_coverage_gate.module_floor_violations(report, config=config, root=tmp_path)
        == []
    )


def test_module_floor_group_that_matches_nothing_fails(tmp_path: Path) -> None:
    """A renamed module must not silently retire its floor."""

    config = _module_floors_config(
        [
            {
                "id": "renamed",
                "description": "test group",
                "globs": ["src/gpu_fault/gone_*.py"],
                "group_floor": 60,
                "file_floor": 30,
            }
        ]
    )
    report = _write_report(
        tmp_path / "coverage.json", {"src/gpu_fault/runtime.py": (90, 100)}
    )

    assert ci_coverage_gate.module_floor_violations(
        report, config=config, root=tmp_path
    ) == ["coverage module floor renamed matched no measured file"]


@pytest.mark.parametrize(
    ("entries", "reason"),
    [
        ([], "no group at all leaves every module unfloored"),
        (
            [
                {
                    "id": "family",
                    "description": "d",
                    "globs": [],
                    "group_floor": 60,
                    "file_floor": 30,
                }
            ],
            "a group with no globs matches nothing and cannot fail",
        ),
        (
            [
                {
                    "id": "family",
                    "description": "d",
                    "globs": ["src/gpu_fault/admin/*.py"],
                    "group_floor": 60,
                    "file_floor": 0,
                }
            ],
            "a file floor of zero passes an untested module",
        ),
        (
            [
                {
                    "id": "family",
                    "description": "d",
                    "globs": ["src/gpu_fault/admin/*.py"],
                    "group_floor": 20,
                    "file_floor": 30,
                }
            ],
            "a file floor above the group floor is contradictory",
        ),
        (
            [
                {
                    "id": "family",
                    "description": "d",
                    "globs": ["a"],
                    "group_floor": 60,
                    "file_floor": 30,
                },
                {
                    "id": "family",
                    "description": "d",
                    "globs": ["b"],
                    "group_floor": 60,
                    "file_floor": 30,
                },
            ],
            "a duplicate id makes it ambiguous which floor a failure names",
        ),
    ],
)
def test_module_floor_config_rejects_a_floor_that_cannot_fail(
    tmp_path: Path, entries: list[dict], reason: str
) -> None:
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    (root / "config/ci-unit-gate.json").write_text(
        json.dumps(_module_floors_config(entries)), encoding="utf-8"
    )

    with pytest.raises(ci_coverage_gate.CoverageGateError):
        ci_coverage_gate.load_config(root)


def test_module_floors_reject_an_unreadable_report(tmp_path: Path) -> None:
    report = tmp_path / "coverage.json"
    report.write_text("{}", encoding="utf-8")

    with pytest.raises(ci_coverage_gate.CoverageGateError, match="unreadable"):
        ci_coverage_gate.module_floor_violations(
            report, config=ci_coverage_gate.load_config(ROOT)
        )


def test_gate_zip_rejects_path_traversal(tmp_path: Path) -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as value:
        value.writestr("../outside", "unsafe")

    with pytest.raises(ci_gate_artifacts.GateArtifactError, match="unsafe"):
        ci_gate_artifacts.extract_archive(archive.getvalue(), tmp_path)
