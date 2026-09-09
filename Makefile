# Local development dependencies, including pytest-xdist, live in the
# repository venv. Source bundles and CI can still fall back to or explicitly
# override the interpreter.
PYTHON ?= $(firstword $(wildcard .venv/bin/python) python3)
# Quality gates import Python modules from deploy/. Keep interpreter caches out
# of the checkout so a prior gate cannot poison the later deploy layout check.
PYTHONPYCACHEPREFIX ?= /tmp/gpu-fault-pycache
export PYTHONPYCACHEPREFIX
# Default to a quarter of the cores, bounded to 4..16: the release gate runs
# this suite next to the artifact build and (in release mode) the static gates.
PYTEST_XDIST_WORKERS ?= $(shell $(PYTHON) -c "import os;print(max(4,min(16,(os.cpu_count() or 4)//4)))")
PYTEST_XDIST_DIST ?= worksteal
PYTEST_DURATIONS ?= 50
FAULT_TEST_WORKERS ?= 1
PARALLEL_FAULT_TEST_WORKERS ?= 4
FAULT_TEST_PYTEST_RESULTS ?= artifacts/fault/pytest-case-results.json
FAULT_TEST_REPORT ?=
COVERAGE_FLOOR ?= 78
COVERAGE_SHARD ?=
COVERAGE_SHARD_ROOT ?= artifacts/coverage-shards/$(COVERAGE_SHARD)
COVERAGE_SHARDS_ROOT ?= artifacts/coverage-shards
COVERAGE_COMBINED_ROOT ?= artifacts/coverage-combined
COVERAGE_LOCAL_JSON ?= artifacts/coverage-local.json
COVERAGE_INCLUDE_STRESS ?= 0
BASE ?= origin/main
COSIGN ?= cosign
RUNTIME_IMAGE_PLATFORM ?= linux/amd64
RELEASE_ATTESTATION := dist/current-attestation.json
RELEASE_ATTESTATION_BUNDLE := dist/current-attestation.bundle.json
PREBUILT_ATTESTATION ?= $(RELEASE_ATTESTATION)
PREBUILT_BUNDLE ?= $(RELEASE_ATTESTATION_BUNDLE)
STAGING_IMPACT_PLAN ?= dist/staging-impact-plan.json
SOURCE_COMPONENT_ARTIFACTS ?= dist/current-release.json
COMPONENT_ARTIFACT_CACHE_ROOT ?=
# Supply-chain tools (review items S8/S10). Versions are pinned here and only
# here; ci.yml installs them through `make ci-supply-chain-tools`, so the
# workflow cannot drift from the Makefile. They live in their own venv so
# their transitive dependencies never touch the hash-locked environment the
# other gates run in. `$(CI)` decides strictness: GitHub sets CI=true, so a
# missing tool is a hard failure there and a printed skip on a laptop.
PIP_AUDIT_VERSION ?= 2.10.1
CFN_LINT_VERSION ?= 1.56.0
CYCLONEDX_BOM_VERSION ?= 7.3.1
# promtool is a Go binary from the Prometheus release tarball, not a pip
# package: pinned by version AND by the tarball's published sha256 so the gate
# runs the bytes it was written against.
PROMTOOL_VERSION ?= 3.14.0
PROMTOOL_SHA256_LINUX_AMD64 ?= f665c6da19eb7ba399c915d30c7d9793c9b417bf8a749b504bc470678631478d
SUPPLY_CHAIN_TOOLS_VENV ?= /tmp/gpu-fault-supply-chain-tools
SUPPLY_CHAIN_PYTHON ?= $(firstword $(wildcard $(SUPPLY_CHAIN_TOOLS_VENV)/bin/python) $(PYTHON))
SUPPLY_CHAIN_BIN = $(dir $(SUPPLY_CHAIN_PYTHON))
# Every lock a shipped artifact is installed from: runtime.lock builds the
# runtime image, build.lock and deploy-host.lock go into the deploy-host bundle,
# and node-runtime.lock is what the node installer hash-pins onto every GPU node.
SHIPPED_LOCKS = requirements/build.lock requirements/runtime.lock requirements/deploy-host.lock requirements/node-runtime.lock
PIP_AUDIT_IGNORE_FILE = requirements/pip-audit-ignore.txt
CFN_TEMPLATES = deploy/aws/lambda/*.yaml
SBOM_DIR = dist/sbom
DEPLOY_HOST_PLATFORM ?= $(shell $(PYTHON) -c "from scripts.deploy_host_bundle import bundle_platform_id; print(bundle_platform_id())")
DEPLOY_HOST_ARCHIVE ?= dist/gpu-fault-deploy-host-$(DEPLOY_HOST_PLATFORM).tar.gz
DEPLOY_HOST_SIGNATURE_BUNDLE ?= dist/gpu-fault-deploy-host-$(DEPLOY_HOST_PLATFORM).sigstore.json
DEPLOY_HOST_VENV ?= .venv
QUALITY_SCRIPTS = scripts tools
QUALITY_SHELL_ROOTS = deploy scripts tools
YAMLLINT_CONFIG = .yamllint
# deploy/ and testcases/ were outside the scan, so the production manifest
# tree and the case catalog were the only YAML nobody linted. --strict is
# what makes the config load-bearing: without it yamllint reports every
# rule in .yamllint as a warning and still exits 0.
YAMLLINT_ROOTS = deploy examples testcases config scripts/e2e scripts/perf
DOCUMENTATION_TESTS = \
	tests/test_documentation_contracts.py \
	tests/test_deployment_manual_contracts.py \
	tests/test_fault_scenario_catalog.py \
	tests/regional/test_regional_acceptance_spec.py \
	tests/regional/test_regional_acceptance_fixtures.py \
	tests/regional/test_complete_acceptance_entries.py \
	tests/test_operations_manual_guide.py \
	tests/test_env_reference.py \
	tests/regional/test_regional_case_index.py \
	tests/test_fault_evidence.py \
	tests/test_doc_impact.py \
	tests/test_change_impact.py
CI_TOOLING_TESTS = \
	tests/test_script_assets.py \
	tests/test_ci_gate.py \
	tests/test_ci_unit_gate.py
POSTGRES_TESTS = \
	tests/app_services/test_cleanup_unbounded_kinds.py \
	tests/app_services/test_collector_ingestion_transaction.py \
	tests/execution/test_branch_settlement.py \
	tests/execution/test_workload_withdrawal.py \
	tests/metrics/test_processor_counter_mode_metric.py \
	tests/notifications/test_delivery_state.py \
	tests/orchestration/test_incident_closure.py \
	tests/processor/test_observation_interlock_liveness.py \
	tests/processor/test_queue_priority_tiers.py \
	tests/processor/test_routine_coalescing_boundaries.py \
	tests/processor/test_telemetry_spool.py \
	tests/regional/test_stuck_workflow_baseline.py \
	tests/store/test_active_workflow_incidents_bound.py \
	tests/store/test_cleanup_logging.py \
	tests/store/test_completion_decision_reads.py \
	tests/store/test_control_record_archive.py \
	tests/store/test_duplicate_event_fast_path.py \
	tests/store/test_evidence_pinned_to_incident.py \
	tests/store/test_health_signal_notified_latch.py \
	tests/store/test_incident_state_counts.py \
	tests/store/test_lease_renewal_half_life.py \
	tests/store/test_marker_retirement_history.py \
	tests/store/test_merge_duplicate_event_contract.py \
	tests/store/test_merge_executor_isolation.py \
	tests/store/test_operator_event_recording.py \
	tests/store/test_orphan_workflow_inspection.py \
	tests/store/test_postgres_admission_vs_claim.py \
	tests/store/test_postgres_claim_window.py \
	tests/store/test_postgres_core_guards.py \
	tests/store/test_postgres_dedicated_hot_state_startup.py \
	tests/store/test_postgres_ensure_schema_locks.py \
	tests/store/test_postgres_find_open_remote_command.py \
	tests/store/test_postgres_health_signal_claim_isolation.py \
	tests/store/test_postgres_index_builder.py \
	tests/store/test_postgres_lane_claim_guard.py \
	tests/store/test_postgres_merge_vs_executor.py \
	tests/store/test_postgres_notification_delivery_lock.py \
	tests/store/test_postgres_observation_sweep.py \
	tests/store/test_postgres_processor_claim.py \
	tests/store/test_postgres_processor_counters.py \
	tests/store/test_postgres_processor_expired_leases.py \
	tests/store/test_postgres_processor_legacy_paths.py \
	tests/store/test_postgres_reconnect.py \
	tests/store/test_postgres_remote_claim_cancellation.py \
	tests/store/test_postgres_remote_command_lock_key.py \
	tests/store/test_postgres_remote_sweeper.py \
	tests/store/test_postgres_spool_notify_trigger.py \
	tests/store/test_postgres_store.py \
	tests/store/test_postgres_store_review_indexes.py \
	tests/store/test_postgres_stuck_workflow_audit.py \
	tests/store/test_postgres_wakeup_triggers.py \
	tests/store/test_postgres_workflow_indexes.py \
	tests/store/test_postgres_workflow_lock_order.py \
	tests/store/test_preemption_pending_marker.py \
	tests/store/test_processor_batch_completion_contract.py \
	tests/store/test_processor_release_cas.py \
	tests/store/test_recent_markers_trusted_predicate.py \
	tests/store/test_reconcile_epoch_cas.py \
	tests/store/test_reconcile_narrow_reads.py \
	tests/store/test_remote_command_stale_fence.py \
	tests/store/test_save_incident_guard.py \
	tests/store/test_save_plan_guard.py \
	tests/store/test_save_workflow_guard.py \
	tests/store/test_schema_trigger_definitions.py \
	tests/store/test_store_contracts.py \
	tests/store/test_store_error_classification.py \
	tests/store/test_store_migrate.py \
	tests/store/test_workflow_scan_cursor.py \
	tests/store/test_workflow_scan_pushdown.py \
	tests/store/test_workflow_scan_whole_second.py

# POSTGRES_TESTS above: every test module that skips without
# GPU_FAULT_TEST_POSTGRES_URL (G-1), kept equal to
# scripts/ci_coverage_config.pytest_targets(root, "postgres") by
# tests/test_ci_unit_gate.py. Regenerate with
#   .venv/bin/python -c 'from pathlib import Path; from scripts.ci_coverage_config import pytest_targets; print("\n".join(pytest_targets(Path("."), "postgres")))'
COVERAGE_IGNORE_ARGS = $(foreach test,$(DOCUMENTATION_TESTS) $(CI_TOOLING_TESTS) $(POSTGRES_TESTS),--ignore=$(test))

.PHONY: test test-postgres test-postgres-stress test-shuffled test-parallel test-parallel-release test-impact regional-impact-plan impact-check coverage coverage-shard coverage-combine fault-test-cases fault-test-cases-ci fault-test-cases-with-cap005 run format check check-static check-static-sequential python-cache-clean html artifact-check runtime-image-check release-build release-build-promoted release-build-staging release-preflight release-deploy deploy-host-bundle deploy-host-sign deploy-host-setup deploy-host-setup-online deploy-host-check architecture-check architecture-baseline code-size-audit mypy-check mixin-check private-test-coupling-check test-source-assertion-check assert-message-check public-release-check ci-tooling-check docs-check docs-static-check doc-impact-check env-doc-check xid-catalog-check config-check case-index-check manual-command-order-check doc-reference-check doc-anchor-check fault-evidence-check deployment-contracts-update deployment-contracts-check deploy-check artifacts-safety-check artifacts-local-safety-check artifacts-retention yaml-check shell-check lazy-export-check cfn-lint-check doc-facts-check pip-audit-check sbom ci-supply-chain-tools promtool-check grafana-dashboards grafana-dashboards-check

test:
	$(PYTHON) -m pytest

# Same suite as test-parallel-release, but with the collection order shuffled,
# so a test that quietly depends on another one running first shows up here
# instead of on the day an unrelated test is added. The seed is drawn HERE and
# exported, never inside conftest.py: every xdist worker runs the collection
# hook, and a seed drawn per process would make the workers disagree about the
# collection, which xdist aborts on. Pass GPU_FAULT_TEST_SHUFFLE_SEED=<n> to
# replay a specific red run; the seed is printed here and in the pytest header.
test-shuffled:
	@seed="$${GPU_FAULT_TEST_SHUFFLE_SEED:-$$($(PYTHON) -c 'import secrets; print(secrets.randbelow(2 ** 32))')}"; \
		printf 'GPU_FAULT_TEST_SHUFFLE_SEED=%s\n' "$$seed"; \
		GPU_FAULT_TEST_POSTGRES_URL= \
		GPU_FAULT_TEST_SHUFFLE_SEED="$$seed" \
		$(PYTHON) -m pytest -n $(PYTEST_XDIST_WORKERS) \
			--dist=$(PYTEST_XDIST_DIST) \
			--durations=$(PYTEST_DURATIONS) \
			--ignore=tests/test_artifact_consistency.py

test-parallel:
	GPU_FAULT_TEST_POSTGRES_URL= \
		$(PYTHON) -m pytest -n $(PYTEST_XDIST_WORKERS) \
			--dist=$(PYTEST_XDIST_DIST) \
			--durations=$(PYTEST_DURATIONS)

test-parallel-release:
	GPU_FAULT_TEST_POSTGRES_URL= \
		$(PYTHON) -m pytest -n $(PYTEST_XDIST_WORKERS) \
			--dist=$(PYTEST_XDIST_DIST) \
			--durations=$(PYTEST_DURATIONS) \
			--ignore=tests/test_artifact_consistency.py

test-impact:
	$(PYTHON) scripts/select-affected-tests.py \
		--base "$(BASE)" \
		--execute

regional-impact-plan:
	$(PYTHON) scripts/select-affected-tests.py \
		--base "$(BASE)" \
		--regional-only

impact-check:
	$(PYTHON) scripts/select-affected-tests.py --check

coverage:
	@test -n "$${GPU_FAULT_TEST_POSTGRES_URL}" || \
		(printf 'GPU_FAULT_TEST_POSTGRES_URL is required\n' >&2; exit 2)
	$(MAKE) python-cache-clean
	$(PYTHON) -m coverage erase
	GPU_FAULT_TEST_POSTGRES_URL= \
	PYTEST_GPU_FAULT_CASE_REPORT="$(FAULT_TEST_PYTEST_RESULTS)" \
	$(PYTHON) -m pytest -n $(PYTEST_XDIST_WORKERS) \
		--dist=$(PYTEST_XDIST_DIST) \
		-p tools.pytest_case_reporter \
		$(COVERAGE_IGNORE_ARGS) \
		--cov=src/gpu_fault --cov=src/gpu_fault_release --cov=deploy/control-plane/tools \
		--cov-branch \
		--cov-report= \
		--durations=$(PYTEST_DURATIONS)
	$(PYTHON) -m pytest $(POSTGRES_TESTS) \
		--cov=src/gpu_fault --cov=src/gpu_fault_release --cov=deploy/control-plane/tools \
		--cov-branch \
		--cov-append \
		--cov-fail-under=$(COVERAGE_FLOOR) \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-report=json:$(COVERAGE_LOCAL_JSON) \
		--durations=$(PYTEST_DURATIONS)
	$(PYTHON) scripts/ci_coverage_gate.py module-floors \
		--coverage-json "$(COVERAGE_LOCAL_JSON)"

coverage-shard:
	@test -n "$(COVERAGE_SHARD)" || \
		(printf 'COVERAGE_SHARD is required\n' >&2; exit 2)
	$(PYTHON) scripts/ci_coverage_gate.py run \
		--shard "$(COVERAGE_SHARD)" \
		--python "$(PYTHON)" \
		--artifact-root "$(COVERAGE_SHARD_ROOT)" \
		--workers "$(PYTEST_XDIST_WORKERS)" \
		--dist "$(PYTEST_XDIST_DIST)" \
		--durations "$(PYTEST_DURATIONS)" \
		$(if $(filter true yes 1,$(COVERAGE_INCLUDE_STRESS)),--include-stress,)

coverage-combine:
	$(PYTHON) scripts/ci_coverage_gate.py combine \
		--python "$(PYTHON)" \
		--shards-root "$(COVERAGE_SHARDS_ROOT)" \
		--output-root "$(COVERAGE_COMBINED_ROOT)" \
		$(if $(GITHUB_RUN_ID),--require-run-id "$(GITHUB_RUN_ID)",)

test-postgres:
	@test -n "$${GPU_FAULT_TEST_POSTGRES_URL}" || \
		(printf 'GPU_FAULT_TEST_POSTGRES_URL is required\n' >&2; exit 2)
	$(PYTHON) -m pytest $(POSTGRES_TESTS) \
		--durations=$(PYTEST_DURATIONS)

test-postgres-stress:
	GPU_FAULT_POSTGRES_LOCK_STRESS_WORKERS=8 \
	GPU_FAULT_POSTGRES_LOCK_STRESS_ROUNDS=40 \
		$(MAKE) test-postgres PYTHON="$(PYTHON)"

fault-test-cases:
	$(PYTHON) tools/run_fault_test_cases.py \
		--workers $(FAULT_TEST_WORKERS)

fault-test-cases-ci:
	@if [ -f "$(FAULT_TEST_PYTEST_RESULTS)" ]; then \
		$(PYTHON) tools/run_fault_test_cases.py \
			--level unit \
			--level component \
			--pytest-results "$(FAULT_TEST_PYTEST_RESULTS)" \
			$(if $(FAULT_TEST_REPORT),--report "$(FAULT_TEST_REPORT)",) \
			--workers $(FAULT_TEST_WORKERS); \
	else \
		$(PYTHON) tools/run_fault_test_cases.py \
			--level unit \
			--level component \
			--batch-pytest \
			$(if $(FAULT_TEST_REPORT),--report "$(FAULT_TEST_REPORT)",) \
			--workers $(FAULT_TEST_WORKERS); \
	fi

fault-test-cases-with-cap005:
	@test -n "$${GPU_FAULT_STORE_URL}" || \
		(printf 'GPU_FAULT_STORE_URL is required\n' >&2; exit 2)
	$(PYTHON) tools/run_fault_test_cases.py \
		--also-case GF-REGIONAL-CAP-005 \
		--include-live \
		--workers $(PARALLEL_FAULT_TEST_WORKERS)

run:
	PYTHONPATH=src $(PYTHON) -m gpu_fault.api

format:
	$(PYTHON) -m ruff format src tests deploy $(QUALITY_SCRIPTS)
	$(PYTHON) -m ruff check --select I --fix tests

python-cache-clean:
	@tracked="$$(git ls-files | \
		awk '/(^|\/)__pycache__\// || /\.py(c|o)$$/')"; \
	if [ -n "$$tracked" ]; then \
		printf 'tracked Python cache artifacts must be removed:\n%s\n' "$$tracked" >&2; \
		exit 1; \
	fi
	find src tests deploy $(QUALITY_SCRIPTS) -type f \
		\( -name '*.pyc' -o -name '*.pyo' \) -delete
	find src tests deploy $(QUALITY_SCRIPTS) -depth \
		-type d -name '__pycache__' -empty -delete

check-static:
	$(MAKE) python-cache-clean
	env -u COSIGN_PASSWORD $(PYTHON) scripts/run_static_gates.py \
		--python "$(PYTHON)"
	$(MAKE) python-cache-clean

check-static-sequential:
	$(MAKE) python-cache-clean
	$(PYTHON) -m ruff format --check src tests deploy $(QUALITY_SCRIPTS)
	$(PYTHON) -m ruff check src tests deploy $(QUALITY_SCRIPTS)
	$(PYTHON) -m ruff check --select I tests
	$(MAKE) mypy-check
	$(PYTHON) -m compileall -q \
		src tests deploy $(QUALITY_SCRIPTS)
	$(MAKE) architecture-check
	$(MAKE) lazy-export-check
	$(MAKE) mixin-check
	$(MAKE) private-test-coupling-check
	$(MAKE) test-source-assertion-check
	$(MAKE) assert-message-check
	$(MAKE) public-release-check
	$(MAKE) xid-catalog-check
	$(MAKE) deployment-contracts-check
	$(MAKE) docs-check DOCS_PYTEST=0
	$(MAKE) config-check
	$(MAKE) deploy-check
	$(MAKE) artifacts-safety-check
	$(MAKE) yaml-check
	$(MAKE) cfn-lint-check
	$(MAKE) shell-check

check:
	env -u COSIGN_PASSWORD $(PYTHON) scripts/run_release_gates.py \
		--mode check \
		--python "$(PYTHON)"

architecture-check:
	$(PYTHON) scripts/check-python-architecture.py

architecture-baseline:
	$(PYTHON) scripts/check-python-architecture.py --write-baseline

# Every `_EXPORTS` tuple in a package __init__ and every Python entry point a
# manifest declares (Lambda `Handler:`) must import; the tuples are only
# executed on first attribute access, so nothing else exercises them.
lazy-export-check:
	$(PYTHON) scripts/check-lazy-exports.py

code-size-audit:
	@mkdir -p artifacts/audit
	@stamp="$$(date -u +%Y%m%dT%H%M%SZ)"; \
		$(PYTHON) scripts/audit-code-size.py \
			--root src/gpu_fault \
			--json-output "artifacts/audit/code-size-$${stamp}.json" \
			--markdown-output "artifacts/audit/code-size-$${stamp}.md"; \
		printf 'Wrote artifacts/audit/code-size-%s.{json,md}\n' "$${stamp}"

mypy-check:
	$(PYTHON) scripts/check-mypy-baseline.py

mixin-check:
	$(PYTHON) scripts/check-mixin-contracts.py

private-test-coupling-check:
	$(PYTHON) scripts/check-test-private-coupling.py

test-source-assertion-check:
	$(PYTHON) scripts/check-test-source-assertions.py

assert-message-check:
	$(PYTHON) scripts/check-assert-messages.py

public-release-check:
	$(PYTHON) scripts/check-public-release.py

ci-tooling-check:
	$(PYTHON) -m pytest $(CI_TOOLING_TESTS)

docs-check:
	$(MAKE) docs-static-check
	@if [ "$(DOCS_PYTEST)" != "0" ]; then \
		$(PYTHON) -m pytest $(DOCUMENTATION_TESTS); \
	fi

docs-static-check:
	$(MAKE) doc-impact-check
	$(MAKE) impact-check
	$(MAKE) env-doc-check
	$(MAKE) case-index-check
	$(MAKE) manual-command-order-check
	$(MAKE) doc-reference-check
	$(MAKE) doc-anchor-check
	$(MAKE) fault-evidence-check
	$(MAKE) doc-facts-check
	$(MAKE) grafana-dashboards-check

doc-impact-check:
	$(PYTHON) scripts/check-doc-impact.py

env-doc-check:
	$(PYTHON) scripts/generate-env-reference.py --check

xid-catalog-check:
	PYTHONPATH=src $(PYTHON) tools/generate_nvidia_xid_policy.py --check

case-index-check:
	PYTHONPATH=src $(PYTHON) scripts/build-regional-case-index.py --check

manual-command-order-check:
	$(PYTHON) scripts/check-manual-command-order.py

doc-reference-check:
	$(PYTHON) scripts/check-doc-references.py

doc-anchor-check:
	$(PYTHON) scripts/check-doc-anchors.py

fault-evidence-check:
	$(PYTHON) scripts/build-fault-evidence-index.py --check

# Declared prose facts (test-tree shape, collector auth model) against the code.
doc-facts-check:
	$(PYTHON) scripts/check-doc-facts.py

# Grafana dashboards are rendered from scripts/grafana_dashboard_catalog.py and
# deploy/observability/amp-rules.yaml (thresholds, runbook anchors); the JSON
# under deploy/observability/dashboards/ is checked in and must match.
grafana-dashboards:
	$(PYTHON) scripts/build-grafana-dashboards.py

grafana-dashboards-check:
	$(PYTHON) scripts/build-grafana-dashboards.py --check

config-check:
	PYTHONPATH=src $(PYTHON) -m gpu_fault.config_cli validate \
		deploy/control-plane/regional/generated

deployment-contracts-update: python-cache-clean
	PYTHON="$(PYTHON)" \
		deploy/control-plane/tools/update-deployment-contracts.sh

deployment-contracts-check: python-cache-clean
	PYTHON="$(PYTHON)" \
		deploy/control-plane/tools/update-deployment-contracts.sh --check

deploy-check: python-cache-clean
	$(PYTHON) scripts/check-deploy-layout.py

artifacts-safety-check:
	$(PYTHON) scripts/check-artifacts.py

artifacts-local-safety-check:
	$(PYTHON) scripts/check-artifacts.py --require-content

release-preflight:
	$(MAKE) artifacts-local-safety-check
	$(MAKE) artifact-check

runtime-image-check:
	@tmp_dir="$$(mktemp -d)"; trap 'rm -rf "$$tmp_dir"' EXIT; \
		$(PYTHON) scripts/build-release-runtime-image.py \
			--repository gpu-fault-runtime-local \
			--output "$$tmp_dir/descriptor.json"; \
		tag="$$(jq -r '.tag' "$$tmp_dir/descriptor.json")"; \
		control_digest="$$(jq -r '.components.control_plane.module_digest' \
			"$$tmp_dir/descriptor.json")"; \
		executor_digest="$$(jq -r '.components.executor.module_digest' \
			"$$tmp_dir/descriptor.json")"; \
		docker run --rm "$$tag" /bin/sh -c \
			'! python -c "import gpu_fault" >/dev/null 2>&1 && \
			/opt/gpu-fault/control-plane/bin/python -c "import pydantic" && \
			/opt/gpu-fault/executor/bin/python -c "import pydantic" && \
			! test -e /opt/gpu-fault/control-plane/bin/gpu-fault-admin && \
			test -x /opt/gpu-fault/control-plane/bin/gpu-fault-api && \
			test -x /opt/gpu-fault/control-plane/bin/gpu-fault-store-migrate && \
			test -x /opt/gpu-fault/executor/bin/gpu-fault-cluster-executor && \
			test -x /opt/gpu-fault/executor/bin/gpu-fault-completion-watcher && \
			test -x /opt/gpu-fault/executor/bin/gpu-fault-collector && \
			test -x /opt/gpu-fault/executor/bin/gpu-fault-node-installer-reconciler'; \
		test "$$control_digest" = "$$(docker run --rm "$$tag" \
			/opt/gpu-fault/control-plane/bin/python -c \
			'from gpu_fault import module_digest; print(module_digest())')"; \
		test "$$executor_digest" = "$$(docker run --rm "$$tag" \
			/opt/gpu-fault/executor/bin/python -c \
			'from gpu_fault import module_digest; print(module_digest())')"

release-build:
	@test -n "$(RUNTIME_IMAGE_REPOSITORY)" || \
		(printf 'RUNTIME_IMAGE_REPOSITORY is required\n' >&2; exit 2)
	@test -n "$${GPU_FAULT_TEST_POSTGRES_URL}" || \
		(printf 'GPU_FAULT_TEST_POSTGRES_URL is required\n' >&2; exit 2)
	@test -z "$$(git status --porcelain --untracked-files=normal)" || \
		(printf 'release-build requires a clean source tree\n' >&2; exit 2)
	@command -v "$(COSIGN)" >/dev/null || \
		(printf 'cosign is required\n' >&2; exit 2)
	env -u COSIGN_PASSWORD $(PYTHON) scripts/run_release_gates.py \
		--python "$(PYTHON)"
	env -u COSIGN_PASSWORD $(PYTHON) scripts/build-release-runtime-image.py \
		--repository "$(RUNTIME_IMAGE_REPOSITORY)" \
		--platform "$(RUNTIME_IMAGE_PLATFORM)" \
		--component-artifacts "$(SOURCE_COMPONENT_ARTIFACTS)" \
		$(foreach arg,$(RUNTIME_IMAGE_BUILD_ARGS),--build-arg "$(arg)") \
		$(if $(RUNTIME_IMAGE_CACHE_FROM),--cache-from "$(RUNTIME_IMAGE_CACHE_FROM)",) \
		$(if $(RUNTIME_IMAGE_CACHE_TO),--cache-to "$(RUNTIME_IMAGE_CACHE_TO)",) \
		$(if $(filter true yes 1,$(RUNTIME_IMAGE_FORCE_REBUILD)),--force-rebuild,) \
		--push
	env -u COSIGN_PASSWORD $(PYTHON) scripts/build-release-artifacts.py \
		--python "$(PYTHON)" \
		--runtime-image-descriptor dist/release-runtime-image.json \
		--reuse-artifacts-from "$(SOURCE_COMPONENT_ARTIFACTS)"
	GPU_FAULT_REQUIRE_BUILD_ARTIFACTS=1 \
		env -u COSIGN_PASSWORD $(PYTHON) -m pytest tests/test_artifact_consistency.py
	env -u COSIGN_PASSWORD $(MAKE) sbom PYTHON="$(PYTHON)"
	env -u COSIGN_PASSWORD $(PYTHON) scripts/build-release-attestation.py \
		--sbom-dir "$(SBOM_DIR)"
	$(COSIGN) sign-blob --yes \
		$(if $(COSIGN_SIGNING_KEY),--key "$(COSIGN_SIGNING_KEY)",) \
		--bundle "$(RELEASE_ATTESTATION_BUNDLE)" \
		"$(RELEASE_ATTESTATION)" >/dev/null

release-build-promoted:
	@test -n "$(RUNTIME_IMAGE_REPOSITORY)" || \
		(printf 'RUNTIME_IMAGE_REPOSITORY is required\n' >&2; exit 2)
	@test -n "$(CI_GATE)" || \
		(printf 'CI_GATE is required\n' >&2; exit 2)
	@test -z "$$(git status --porcelain --untracked-files=normal)" || \
		(printf 'release-build-promoted requires a clean source tree\n' >&2; exit 2)
	@command -v "$(COSIGN)" >/dev/null || \
		(printf 'cosign is required\n' >&2; exit 2)
	env -u COSIGN_PASSWORD $(PYTHON) scripts/ci_gate.py verify \
		--gate "$(CI_GATE)" \
		--dist dist
	env -u COSIGN_PASSWORD $(PYTHON) scripts/build-release-runtime-image.py \
		--repository "$(RUNTIME_IMAGE_REPOSITORY)" \
		--platform "$(RUNTIME_IMAGE_PLATFORM)" \
		--component-artifacts "$(SOURCE_COMPONENT_ARTIFACTS)" \
		$(foreach arg,$(RUNTIME_IMAGE_BUILD_ARGS),--build-arg "$(arg)") \
		$(if $(RUNTIME_IMAGE_CACHE_FROM),--cache-from "$(RUNTIME_IMAGE_CACHE_FROM)",) \
		$(if $(RUNTIME_IMAGE_CACHE_TO),--cache-to "$(RUNTIME_IMAGE_CACHE_TO)",) \
		$(if $(filter true yes 1,$(RUNTIME_IMAGE_FORCE_REBUILD)),--force-rebuild,) \
		--push
	env -u COSIGN_PASSWORD $(PYTHON) scripts/build-release-artifacts.py \
		--python "$(PYTHON)" \
		--runtime-image-descriptor dist/release-runtime-image.json \
		--reuse-artifacts-from "$(SOURCE_COMPONENT_ARTIFACTS)"
	GPU_FAULT_REQUIRE_BUILD_ARTIFACTS=1 \
		env -u COSIGN_PASSWORD $(PYTHON) -m pytest tests/test_artifact_consistency.py
	env -u COSIGN_PASSWORD $(MAKE) sbom PYTHON="$(PYTHON)"
	env -u COSIGN_PASSWORD $(PYTHON) scripts/build-release-attestation.py \
		--ci-gate "$(CI_GATE)" \
		--sbom-dir "$(SBOM_DIR)"
	$(COSIGN) sign-blob --yes \
		$(if $(COSIGN_SIGNING_KEY),--key "$(COSIGN_SIGNING_KEY)",) \
		--bundle "$(RELEASE_ATTESTATION_BUNDLE)" \
		"$(RELEASE_ATTESTATION)" >/dev/null

release-build-staging:
	@test -n "$(RUNTIME_IMAGE_REPOSITORY)" || \
		(printf 'RUNTIME_IMAGE_REPOSITORY is required\n' >&2; exit 2)
	@test -z "$$(git status --porcelain --untracked-files=normal)" || \
		(printf 'release-build-staging requires a clean source tree\n' >&2; exit 2)
	@command -v "$(COSIGN)" >/dev/null || \
		(printf 'cosign is required\n' >&2; exit 2)
	env -u COSIGN_PASSWORD $(MAKE) public-release-check PYTHON="$(PYTHON)"
	@if [ "$(IMPACT_PLAN_PREPARED)" != "1" ]; then \
		env -u COSIGN_PASSWORD $(PYTHON) scripts/select-affected-tests.py \
			--base "$(BASE)" \
			--format json \
			--write-plan "$(STAGING_IMPACT_PLAN)" >/dev/null; \
	fi
	env -u COSIGN_PASSWORD $(PYTHON) scripts/select-affected-tests.py \
		--base "$(BASE)" \
		--read-plan "$(STAGING_IMPACT_PLAN)" \
		--execute
	env -u COSIGN_PASSWORD $(PYTHON) scripts/build-release-artifacts.py \
		--python "$(PYTHON)" \
		--reuse-if-current \
		$(if $(COMPONENT_ARTIFACT_CACHE_ROOT),--component-cache-root "$(COMPONENT_ARTIFACT_CACHE_ROOT)",)
	env -u COSIGN_PASSWORD $(PYTHON) scripts/build-release-runtime-image.py \
		--repository "$(RUNTIME_IMAGE_REPOSITORY)" \
		--platform "$(RUNTIME_IMAGE_PLATFORM)" \
		--component-artifacts "$(SOURCE_COMPONENT_ARTIFACTS)" \
		$(foreach arg,$(RUNTIME_IMAGE_BUILD_ARGS),--build-arg "$(arg)") \
		$(if $(RUNTIME_IMAGE_CACHE_FROM),--cache-from "$(RUNTIME_IMAGE_CACHE_FROM)",) \
		$(if $(RUNTIME_IMAGE_CACHE_TO),--cache-to "$(RUNTIME_IMAGE_CACHE_TO)",) \
		$(if $(filter true yes 1,$(RUNTIME_IMAGE_FORCE_REBUILD)),--force-rebuild,) \
		--push
	env -u COSIGN_PASSWORD $(PYTHON) scripts/build-release-artifacts.py \
		--python "$(PYTHON)" \
		--runtime-image-descriptor dist/release-runtime-image.json \
		--reuse-artifacts-from "$(SOURCE_COMPONENT_ARTIFACTS)" \
		--staging-only
	GPU_FAULT_REQUIRE_BUILD_ARTIFACTS=1 \
		env -u COSIGN_PASSWORD $(PYTHON) -m pytest tests/test_artifact_consistency.py
	env -u COSIGN_PASSWORD $(MAKE) sbom PYTHON="$(PYTHON)"
	env -u COSIGN_PASSWORD $(PYTHON) scripts/build-release-attestation.py \
		--staging-only \
		--impact-base "$(BASE)" \
		--impact-plan "$(STAGING_IMPACT_PLAN)" \
		--sbom-dir "$(SBOM_DIR)"
	$(COSIGN) sign-blob --yes \
		$(if $(COSIGN_SIGNING_KEY),--key "$(COSIGN_SIGNING_KEY)",) \
		--bundle "$(RELEASE_ATTESTATION_BUNDLE)" \
		"$(RELEASE_ATTESTATION)" >/dev/null

release-deploy:
	@if [ -n "$(CPU_CLUSTER_ARN)$(GPU_CLUSTER_ARNS)" ]; then \
		test -n "$(CPU_CLUSTER_ARN)" || \
			(printf 'CPU_CLUSTER_ARN is required\n' >&2; exit 2); \
		test -n "$(GPU_CLUSTER_ARNS)" || \
			(printf 'GPU_CLUSTER_ARNS is required\n' >&2; exit 2); \
		test -n "$(STATE_DIR)" || \
			(printf 'STATE_DIR is required\n' >&2; exit 2); \
		PYTHONPATH=src $(PYTHON) -m gpu_fault.admin.cli deploy \
			--cpu-cluster-arn "$(CPU_CLUSTER_ARN)" \
			$(foreach arn,$(GPU_CLUSTER_ARNS),--gpu-cluster-arn "$(arn)") \
			--state-dir "$(STATE_DIR)" \
			$(if $(ADMIN_EMAIL),--admin-email "$(ADMIN_EMAIL)",) \
			$(if $(REPO_ROOT),--repo-root "$(REPO_ROOT)",); \
	else \
		test -f "$(PREBUILT_ATTESTATION)" || \
			(printf 'PREBUILT_ATTESTATION does not exist: %s\n' \
				"$(PREBUILT_ATTESTATION)" >&2; exit 2); \
		if [ -z "$(PREBUILT_SIGNATURE)" ] && [ -z "$(PREBUILT_BUNDLE)" ]; then \
			printf 'PREBUILT_SIGNATURE or PREBUILT_BUNDLE is required\n' >&2; \
			exit 2; \
		fi; \
		if [ -n "$(PREBUILT_SIGNATURE)" ] && [ ! -f "$(PREBUILT_SIGNATURE)" ]; then \
			printf 'PREBUILT_SIGNATURE does not exist: %s\n' \
				"$(PREBUILT_SIGNATURE)" >&2; exit 2; \
		fi; \
		if [ -n "$(PREBUILT_BUNDLE)" ] && [ ! -f "$(PREBUILT_BUNDLE)" ]; then \
			printf 'PREBUILT_BUNDLE does not exist: %s\n' \
				"$(PREBUILT_BUNDLE)" >&2; exit 2; \
		fi; \
		if [ -z "$(COSIGN_KEY)" ]; then \
			( test -n "$(PREBUILT_BUNDLE)" || test -n "$(PREBUILT_CERTIFICATE)" ) && \
			test -n "$(CERTIFICATE_IDENTITY)" && \
			test -n "$(CERTIFICATE_OIDC_ISSUER)" || \
			(printf 'keyless verification requires bundle/certificate identity and issuer\n' >&2; exit 2); \
		fi; \
		PYTHONPATH=src $(PYTHON) scripts/release_deploy.py \
			$(if $(SITE),--site "$(SITE)",) \
			--prebuilt-attestation "$(PREBUILT_ATTESTATION)" \
			$(if $(PREBUILT_SIGNATURE),--prebuilt-signature "$(PREBUILT_SIGNATURE)",) \
			$(if $(PREBUILT_BUNDLE),--prebuilt-bundle "$(PREBUILT_BUNDLE)",) \
			$(if $(PREBUILT_CERTIFICATE),--prebuilt-certificate "$(PREBUILT_CERTIFICATE)",) \
			$(if $(COSIGN_KEY),--cosign-key "$(COSIGN_KEY)",) \
			$(if $(CERTIFICATE_IDENTITY),--certificate-identity "$(CERTIFICATE_IDENTITY)",) \
			$(if $(CERTIFICATE_OIDC_ISSUER),--certificate-oidc-issuer "$(CERTIFICATE_OIDC_ISSUER)",); \
	fi

deploy-host-bundle:
	env -u COSIGN_PASSWORD $(PYTHON) scripts/build-deploy-host-bundle.py \
		--python "$(PYTHON)" \
		--output "$(DEPLOY_HOST_ARCHIVE)" $(if $(DEPLOY_HOST_WHEELHOUSE),--wheelhouse "$(DEPLOY_HOST_WHEELHOUSE)",) $(if $(DEPLOY_HOST_TOOLS_DIR),--tools-dir "$(DEPLOY_HOST_TOOLS_DIR)",) $(if $(filter true yes 1,$(DEPLOY_HOST_ALLOW_DIRTY)),--allow-dirty,)
	$(COSIGN) sign-blob --yes \
		$(if $(COSIGN_SIGNING_KEY),--key "$(COSIGN_SIGNING_KEY)",) \
		--bundle "$(DEPLOY_HOST_SIGNATURE_BUNDLE)" \
		"$(DEPLOY_HOST_ARCHIVE)" >/dev/null

deploy-host-sign:
	@test -f "$(DEPLOY_HOST_ARCHIVE)" || \
		(printf 'DEPLOY_HOST_ARCHIVE does not exist: %s\n' \
			"$(DEPLOY_HOST_ARCHIVE)" >&2; exit 2)
	$(COSIGN) sign-blob --yes \
		$(if $(COSIGN_SIGNING_KEY),--key "$(COSIGN_SIGNING_KEY)",) \
		--bundle "$(DEPLOY_HOST_SIGNATURE_BUNDLE)" \
		"$(DEPLOY_HOST_ARCHIVE)" >/dev/null

deploy-host-setup:
	@test -f "$(DEPLOY_HOST_ARCHIVE)" || \
		(printf 'DEPLOY_HOST_ARCHIVE does not exist: %s\n' \
			"$(DEPLOY_HOST_ARCHIVE)" >&2; exit 2)
	@test -f "$(DEPLOY_HOST_SIGNATURE_BUNDLE)" || \
		(printf 'DEPLOY_HOST_SIGNATURE_BUNDLE does not exist: %s\n' \
			"$(DEPLOY_HOST_SIGNATURE_BUNDLE)" >&2; exit 2)
	PYTHON=python3.12 scripts/setup-deploy-host.sh \
		--venv "$(DEPLOY_HOST_VENV)" \
		--bundle "$(DEPLOY_HOST_ARCHIVE)" \
		--signature-bundle "$(DEPLOY_HOST_SIGNATURE_BUNDLE)" $(if $(DEPLOY_HOST_COSIGN_KEY),--cosign-key "$(DEPLOY_HOST_COSIGN_KEY)",) $(if $(CERTIFICATE_IDENTITY),--certificate-identity "$(CERTIFICATE_IDENTITY)",) $(if $(CERTIFICATE_OIDC_ISSUER),--certificate-oidc-issuer "$(CERTIFICATE_OIDC_ISSUER)",)

deploy-host-setup-online:
	PYTHON=python3.12 scripts/setup-deploy-host.sh \
		--venv "$(DEPLOY_HOST_VENV)" \
		--allow-network

deploy-host-check:
	PYTHON=python3.12 scripts/setup-deploy-host.sh \
		--venv "$(DEPLOY_HOST_VENV)" \
		--check

# 本地保留策略（artifacts/README.md「Retention」）。只报告，不删除；真正
# 删除要人工加 --apply。artifacts/ 在 .gitignore 里，缺失时输出 0 条而不报错，
# 所以这个目标在新克隆上也能跑。故意不进 `make check`：它读的是本地工作区。
artifacts-retention:
	$(PYTHON) scripts/prune_artifacts.py

yaml-check:
	$(PYTHON) -m yamllint --strict -c $(YAMLLINT_CONFIG) $(YAMLLINT_ROOTS)

shell-check:
	find $(QUALITY_SHELL_ROOTS) -type f -name '*.sh' -print0 | \
		sort -z | xargs -0 -n1 bash -n
	@if command -v shellcheck >/dev/null 2>&1; then \
		find $(QUALITY_SHELL_ROOTS) -type f -name '*.sh' -print0 | \
			sort -z | xargs -0 shellcheck --severity=info; \
	else \
		printf 'shellcheck is not installed; bash syntax passed, CI runs shellcheck\\n'; \
	fi

# Pinned supply-chain tools in a venv of their own (see the variables at the
# top). CI and the release workflow run this before the checks below.
ci-supply-chain-tools:
	$(PYTHON) -m venv "$(SUPPLY_CHAIN_TOOLS_VENV)"
	"$(SUPPLY_CHAIN_TOOLS_VENV)/bin/python" -m pip install --quiet --upgrade pip
	"$(SUPPLY_CHAIN_TOOLS_VENV)/bin/python" -m pip install --quiet \
		pip-audit==$(PIP_AUDIT_VERSION) \
		cfn-lint==$(CFN_LINT_VERSION) \
		cyclonedx-bom==$(CYCLONEDX_BOM_VERSION)
	curl -sSfL --retry 3 -o "$(SUPPLY_CHAIN_TOOLS_VENV)/promtool.tar.gz" \
		"https://github.com/prometheus/prometheus/releases/download/v$(PROMTOOL_VERSION)/prometheus-$(PROMTOOL_VERSION).linux-amd64.tar.gz"
	printf '%s  %s\n' "$(PROMTOOL_SHA256_LINUX_AMD64)" "$(SUPPLY_CHAIN_TOOLS_VENV)/promtool.tar.gz" | sha256sum -c -
	tar -xzf "$(SUPPLY_CHAIN_TOOLS_VENV)/promtool.tar.gz" -C "$(SUPPLY_CHAIN_TOOLS_VENV)/bin" \
		--strip-components=1 "prometheus-$(PROMTOOL_VERSION).linux-amd64/promtool"

# verify-regional-alerting.py proves runbooks, annotations and aggregation;
# promtool proves the PromQL itself parses and type-checks. The Kubernetes
# PrometheusRule envelope is unwrapped by the script before promtool sees it.
promtool-check:
	@if [ -x "$(SUPPLY_CHAIN_BIN)promtool" ]; then \
		$(PYTHON) scripts/check-alert-rules.py \
			--promtool "$(SUPPLY_CHAIN_BIN)promtool" --version "$(PROMTOOL_VERSION)"; \
	elif [ -n "$(CI)" ]; then \
		printf 'promtool is required in CI: run make ci-supply-chain-tools\n' >&2; \
		exit 2; \
	else \
		printf 'promtool is not installed; PromQL check skipped, CI runs promtool %s\n' \
			"$(PROMTOOL_VERSION)"; \
	fi

# yamllint proves the Lambda templates are YAML; cfn-lint proves they are
# CloudFormation (resource schemas, intrinsic functions, property types).
cfn-lint-check:
	@if [ -x "$(SUPPLY_CHAIN_BIN)cfn-lint" ]; then \
		"$(SUPPLY_CHAIN_BIN)cfn-lint" $(CFN_TEMPLATES); \
	elif [ -n "$(CI)" ]; then \
		printf 'cfn-lint is required in CI: run make ci-supply-chain-tools\n' >&2; \
		exit 2; \
	else \
		printf 'cfn-lint is not installed; CloudFormation lint skipped, CI runs cfn-lint==%s\n' \
			"$(CFN_LINT_VERSION)"; \
	fi

# Known-vulnerability audit of every shipped lock, hash-pinned so the audited
# set is exactly what installs. Accepted findings live in
# $(PIP_AUDIT_IGNORE_FILE), one id per line with the fix version; remove the
# line when the lock moves past it.
pip-audit-check:
	@if [ -x "$(SUPPLY_CHAIN_BIN)pip-audit" ]; then \
		ignores="$$(sed -e 's/#.*//' $(PIP_AUDIT_IGNORE_FILE) | \
			awk 'NF { printf "--ignore-vuln %s ", $$1 }')"; \
		for lock in $(SHIPPED_LOCKS); do \
			printf 'pip-audit %s\n' "$$lock"; \
			"$(SUPPLY_CHAIN_BIN)pip-audit" --require-hashes --progress-spinner off \
				$$ignores --requirement "$$lock" || exit 1; \
		done; \
	elif [ -n "$(CI)" ]; then \
		printf 'pip-audit is required in CI: run make ci-supply-chain-tools\n' >&2; \
		exit 2; \
	else \
		printf 'pip-audit is not installed; dependency audit skipped, CI runs pip-audit==%s\n' \
			"$(PIP_AUDIT_VERSION)"; \
	fi

# CycloneDX SBOM per shipped lock. build-release-attestation.py --sbom-dir
# binds each file's SHA-256 into the attestation cosign signs, so the SBOM is
# covered by the release signature without a second signing step. Stale files
# are removed first: an SBOM from a previous lock must never be attested.
sbom:
	@rm -rf "$(SBOM_DIR)"
	@if [ -x "$(SUPPLY_CHAIN_BIN)cyclonedx-py" ]; then \
		mkdir -p "$(SBOM_DIR)"; \
		for lock in $(SHIPPED_LOCKS); do \
			name="$$(basename "$$lock" .lock)"; \
			"$(SUPPLY_CHAIN_BIN)cyclonedx-py" requirements "$$lock" \
				--output-reproducible --of JSON \
				-o "$(SBOM_DIR)/$$name.cdx.json" || exit 1; \
			printf 'sbom %s -> %s/%s.cdx.json\n' "$$lock" "$(SBOM_DIR)" "$$name"; \
		done; \
	elif [ -n "$(CI)" ]; then \
		printf 'cyclonedx-py is required in CI: run make ci-supply-chain-tools\n' >&2; \
		exit 2; \
	else \
		printf 'warning: cyclonedx-py is not installed; the release attestation will carry no SBOM (CI installs cyclonedx-bom==%s)\n' \
			"$(CYCLONEDX_BOM_VERSION)" >&2; \
	fi

# Rebuild the wheel and prove it is the current source. A manual roll
# uploads whatever wheel is sitting in dist/ into the wheel ConfigMap, so
# a stale wheel there ships code nobody reviewed; the annotations do not
# catch it because they record intent, not content.
artifact-check:
	$(PYTHON) scripts/build-release-artifacts.py \
		--python $(PYTHON)
	GPU_FAULT_REQUIRE_BUILD_ARTIFACTS=1 \
		$(PYTHON) -m pytest tests/test_artifact_consistency.py
	$(PYTHON) -c \
		"import shutil; shutil.rmtree('build', ignore_errors=True)"

html:
	./build-html.sh
