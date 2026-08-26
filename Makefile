PYTHON ?= python3
# Quality gates import Python modules from deploy/. Keep interpreter caches out
# of the checkout so a prior gate cannot poison the later deploy layout check.
PYTHONPYCACHEPREFIX ?= /tmp/gpu-fault-pycache
export PYTHONPYCACHEPREFIX
PYTEST_XDIST_WORKERS ?= 4
COVERAGE_FLOOR ?= 78
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
	tests/test_operations_manual_guide.py \
	tests/test_env_reference.py \
	tests/regional/test_regional_case_index.py \
	tests/test_fault_evidence.py \
	tests/test_doc_impact.py

.PHONY: test test-postgres test-parallel coverage fault-test-cases fault-test-cases-ci run format check python-cache-clean html artifact-check release-preflight release-deploy architecture-check architecture-baseline code-size-audit mypy-check mixin-check private-test-coupling-check assert-message-check public-release-check docs-check doc-impact-check env-doc-check xid-catalog-check config-check case-index-check manual-command-order-check doc-reference-check fault-evidence-check deployment-contracts-update deployment-contracts-check deploy-check artifacts-safety-check artifacts-local-safety-check artifacts-retention yaml-check shell-check

test:
	$(PYTHON) -m pytest

test-parallel:
	GPU_FAULT_TEST_POSTGRES_URL= \
		$(PYTHON) -m pytest -n $(PYTEST_XDIST_WORKERS)

coverage:
	GPU_FAULT_TEST_POSTGRES_URL= \
	$(PYTHON) -m pytest -n $(PYTEST_XDIST_WORKERS) \
		--cov=src/gpu_fault \
		--cov-branch \
		--cov-fail-under=$(COVERAGE_FLOOR) \
		--cov-report=term-missing \
		--cov-report=html

test-postgres:
	@test -n "$${GPU_FAULT_TEST_POSTGRES_URL}" || \
		(printf 'GPU_FAULT_TEST_POSTGRES_URL is required\n' >&2; exit 2)
	$(PYTHON) -m pytest \
		tests/store/test_postgres_store.py \
		tests/store/test_postgres_processor_claim.py \
		tests/store/test_postgres_reconnect.py \
		tests/store/test_store_contracts.py

fault-test-cases:
	$(PYTHON) tools/run_fault_test_cases.py

fault-test-cases-ci:
	$(PYTHON) tools/run_fault_test_cases.py \
		--level unit \
		--level component

run:
	PYTHONPATH=src $(PYTHON) -m gpu_fault.api

format:
	$(PYTHON) -m ruff format src deploy $(QUALITY_SCRIPTS)
	$(PYTHON) -m ruff format \
		--config 'format.skip-magic-trailing-comma=true' tests
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

check:
	$(MAKE) python-cache-clean
	$(PYTHON) -m ruff format --check src deploy $(QUALITY_SCRIPTS)
	$(PYTHON) -m ruff format --check \
		--config 'format.skip-magic-trailing-comma=true' tests
	$(PYTHON) -m ruff check src tests deploy $(QUALITY_SCRIPTS)
	$(PYTHON) -m ruff check --select I tests
	$(MAKE) mypy-check
	$(PYTHON) -m compileall -q \
		src tests deploy $(QUALITY_SCRIPTS)
	$(MAKE) architecture-check
	$(MAKE) mixin-check
	$(MAKE) private-test-coupling-check
	$(MAKE) assert-message-check
	$(MAKE) public-release-check
	$(MAKE) xid-catalog-check
	$(MAKE) deployment-contracts-check
	$(MAKE) docs-check
	$(MAKE) config-check
	$(MAKE) deploy-check
	$(MAKE) artifacts-safety-check
	$(MAKE) yaml-check
	$(MAKE) shell-check
	$(MAKE) artifact-check
	$(MAKE) test-parallel
	$(MAKE) python-cache-clean

architecture-check:
	$(PYTHON) scripts/check-python-architecture.py

architecture-baseline:
	$(PYTHON) scripts/check-python-architecture.py --write-baseline

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

assert-message-check:
	$(PYTHON) scripts/check-assert-messages.py

public-release-check:
	$(PYTHON) scripts/check-public-release.py

docs-check:
	$(MAKE) doc-impact-check
	$(MAKE) env-doc-check
	$(MAKE) case-index-check
	$(MAKE) manual-command-order-check
	$(MAKE) doc-reference-check
	$(MAKE) fault-evidence-check
	$(PYTHON) -m pytest $(DOCUMENTATION_TESTS)

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

fault-evidence-check:
	$(PYTHON) scripts/build-fault-evidence-index.py --check

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

release-deploy:
	PYTHONPATH=src $(PYTHON) scripts/release_deploy.py \
		$(if $(SITE),--site "$(SITE)",) \
		$(if $(ADMIN_EMAIL),--admin-email "$(ADMIN_EMAIL)",) \
		$(if $(PROFILE_APPROVAL),--profile-approval "$(PROFILE_APPROVAL)",)

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
			sort -z | xargs -0 shellcheck --severity=warning; \
	else \
		printf 'shellcheck is not installed; bash syntax passed, CI runs shellcheck\\n'; \
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
