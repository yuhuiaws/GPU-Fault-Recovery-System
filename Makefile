PYTHON ?= python3
# Quality gates import Python modules from deploy/. Keep interpreter caches out
# of the checkout so a prior gate cannot poison the later deploy layout check.
PYTHONPYCACHEPREFIX ?= /tmp/gpu-fault-pycache
export PYTHONPYCACHEPREFIX
PYTEST_XDIST_WORKERS ?= 4
COVERAGE_FLOOR ?= 78
BASE ?= origin/main
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
	tests/test_operations_manual_guide.py \
	tests/test_env_reference.py \
	tests/regional/test_regional_case_index.py \
	tests/test_fault_evidence.py \
	tests/test_doc_impact.py \
	tests/test_change_impact.py

.PHONY: test test-postgres test-postgres-stress test-parallel test-impact regional-impact-plan impact-check coverage fault-test-cases fault-test-cases-ci run format check python-cache-clean html artifact-check runtime-image-check release-build release-preflight release-deploy architecture-check architecture-baseline code-size-audit mypy-check mixin-check private-test-coupling-check assert-message-check public-release-check docs-check doc-impact-check env-doc-check xid-catalog-check config-check case-index-check manual-command-order-check doc-reference-check fault-evidence-check deployment-contracts-update deployment-contracts-check deploy-check terraform-check artifacts-safety-check artifacts-local-safety-check artifacts-retention yaml-check shell-check

test:
	$(PYTHON) -m pytest

test-parallel:
	GPU_FAULT_TEST_POSTGRES_URL= \
		$(PYTHON) -m pytest -n $(PYTEST_XDIST_WORKERS)

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

test-postgres-stress:
	GPU_FAULT_POSTGRES_LOCK_STRESS_WORKERS=8 \
	GPU_FAULT_POSTGRES_LOCK_STRESS_ROUNDS=40 \
		$(MAKE) test-postgres PYTHON="$(PYTHON)"

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
	$(MAKE) terraform-check
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
	$(MAKE) impact-check
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

terraform-check:
	@command -v terraform >/dev/null || \
		(printf 'terraform is required\n' >&2; exit 2)
	@data_dir="$$(mktemp -d)"; \
		trap 'rm -rf "$$data_dir"' EXIT; \
		TF_DATA_DIR="$$data_dir" terraform fmt -check -recursive \
			deploy/aws/regional-foundation; \
		TF_DATA_DIR="$$data_dir" terraform \
			-chdir=deploy/aws/regional-foundation \
			init -backend=false -input=false >/dev/null; \
		TF_DATA_DIR="$$data_dir" terraform \
			-chdir=deploy/aws/regional-foundation validate

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
	$(MAKE) check PYTHON="$(PYTHON)"
	$(MAKE) test-postgres-stress PYTHON="$(PYTHON)"
	$(PYTHON) scripts/build-release-runtime-image.py \
		--repository "$(RUNTIME_IMAGE_REPOSITORY)" \
		--platform "$(or $(RUNTIME_IMAGE_PLATFORM),linux/amd64)" \
		--push
	$(PYTHON) scripts/build-release-artifacts.py \
		--python "$(PYTHON)" \
		--runtime-image-descriptor dist/release-runtime-image.json
	GPU_FAULT_REQUIRE_BUILD_ARTIFACTS=1 \
		$(PYTHON) -m pytest tests/test_artifact_consistency.py
	$(PYTHON) scripts/build-release-attestation.py

release-deploy:
	@test -n "$(PREBUILT_ATTESTATION)" || \
		(printf 'PREBUILT_ATTESTATION is required\n' >&2; exit 2)
	@if [ -z "$(PREBUILT_SIGNATURE)" ] && [ -z "$(PREBUILT_BUNDLE)" ]; then \
		printf 'PREBUILT_SIGNATURE or PREBUILT_BUNDLE is required\n' >&2; \
		exit 2; \
	fi
	@if [ -z "$(COSIGN_KEY)" ]; then \
		( test -n "$(PREBUILT_BUNDLE)" || test -n "$(PREBUILT_CERTIFICATE)" ) && \
		test -n "$(CERTIFICATE_IDENTITY)" && \
		test -n "$(CERTIFICATE_OIDC_ISSUER)" || \
		(printf 'keyless verification requires PREBUILT_BUNDLE or PREBUILT_CERTIFICATE, plus CERTIFICATE_IDENTITY and CERTIFICATE_OIDC_ISSUER\n' >&2; exit 2); \
	fi
	PYTHONPATH=src $(PYTHON) scripts/release_deploy.py \
		$(if $(SITE),--site "$(SITE)",) \
		$(if $(PROFILE_APPROVAL),--profile-approval "$(PROFILE_APPROVAL)",) \
		--prebuilt-attestation "$(PREBUILT_ATTESTATION)" \
		$(if $(PREBUILT_SIGNATURE),--prebuilt-signature "$(PREBUILT_SIGNATURE)",) \
		$(if $(PREBUILT_BUNDLE),--prebuilt-bundle "$(PREBUILT_BUNDLE)",) \
		$(if $(PREBUILT_CERTIFICATE),--prebuilt-certificate "$(PREBUILT_CERTIFICATE)",) \
		$(if $(COSIGN_KEY),--cosign-key "$(COSIGN_KEY)",) \
		$(if $(CERTIFICATE_IDENTITY),--certificate-identity "$(CERTIFICATE_IDENTITY)",) \
		$(if $(CERTIFICATE_OIDC_ISSUER),--certificate-oidc-issuer "$(CERTIFICATE_OIDC_ISSUER)",)

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
