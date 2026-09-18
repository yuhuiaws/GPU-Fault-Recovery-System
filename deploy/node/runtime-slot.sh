#!/usr/bin/env bash

prepare_py_spy() {
    local tool_dir="$1"
    local binary="${tool_dir}/venv/bin/py-spy"
    local observed_sha

    if [[ -f "${tool_dir}/.complete" && -x "${binary}" ]]; then
        observed_sha="$(sha256sum "${binary}" | cut -d' ' -f1)"
        if [[ "${observed_sha}" == "${PY_SPY_BINARY_SHA256}" ]]; then
            "${binary}" --version >/dev/null ||
                die "py-spy installation verification failed"
            return
        fi
        die "existing py-spy tool slot failed integrity validation"
    fi
    if [[ -e "${tool_dir}" ]]; then
        rm -rf "${tool_dir}"
    fi
    install -d -m 0755 "${tool_dir}"
    "${PYTHON_COMMAND}" -m venv "${tool_dir}/venv"
    local -a source_args=()
    if [[ -n "${WHEELHOUSE}" ]]; then
        source_args=(--no-index --find-links "${WHEELHOUSE}")
    fi
    "${tool_dir}/venv/bin/python" -I -m pip install \
        --force-reinstall --no-deps --only-binary=:all: --require-hashes \
        --requirement "${REPO_DIR}/requirements/node-tools.lock" \
        "${source_args[@]}"
    observed_sha="$(sha256sum "${binary}" | cut -d' ' -f1)"
    [[ "${observed_sha}" == "${PY_SPY_BINARY_SHA256}" ]] ||
        die "py-spy binary SHA-256 mismatch"
    "${binary}" --version >/dev/null || die "py-spy installation verification failed"
    touch "${tool_dir}/.complete"
    chmod 0644 "${tool_dir}/.complete"
}
# Sourced by the node installer after the wheel and dependency lock are verified.

RUNTIME_INTEGRITY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/runtime_integrity.py"

runtime_dependency_identity() {
    "${PYTHON_COMMAND}" -I -S "${RUNTIME_INTEGRITY}" identity --lock "${DEPENDENCY_LOCK}"
}

runtime_slot_directory() {
    printf '%s/%s-%s\n' \
        "${RUNTIME_RELEASES_DIR:?node releases directory is required}" \
        "${WHEEL_SHA256}" "${RUNTIME_DEPENDENCY_SHA256}"
}

validate_previous_node_runtime() {
    if [[ -L "${RUNTIME_CURRENT_LINK}" || -d "${LEGACY_VENV_PATH}" ]]; then
        "${PYTHON_COMMAND}" -I -S -B "${RUNTIME_INTEGRITY}" previous \
            --root "${RUNTIME_ROOT}" >/dev/null ||
            die "previous node runtime failed integrity validation"
    fi
}

prepare_node_runtime() {
    RUNTIME_DEPENDENCY_SHA256="$(runtime_dependency_identity)"
    RUNTIME_RELEASE_DIR="$(runtime_slot_directory)"
    validate_previous_node_runtime
    prepare_runtime_slot "${RUNTIME_RELEASE_DIR}" "${WHEEL_SHA256}"
}

runtime_record_digest() {
    local release_dir="$1"
    local -a seal_args=()
    if [[ -f "${release_dir}/record.sha256" ]]; then
        seal_args=(--seal "${release_dir}/record.sha256")
    fi
    "${PYTHON_COMMAND}" -I -S "${RUNTIME_INTEGRITY}" validate \
        --venv "${release_dir}/venv" --lock "${DEPENDENCY_LOCK}" \
        --layer "$(runtime_dependency_directory)/venv" --wheel "${WHEEL}" "${seal_args[@]}"
}

runtime_dependency_directory() {
    printf '%s/%s\n' \
        "${RUNTIME_DEPENDENCIES_DIR:-${RUNTIME_RELEASES_DIR%/*}/dependencies}" \
        "${RUNTIME_DEPENDENCY_SHA256}"
}

validate_dependency_layer() {
    local directory="$1"
    local observed_record
    [[ -f "${directory}/.complete" && -f "${directory}/record.sha256" ]] || return 1
    [[ -f "${directory}/dependency.sha256" ]] || return 1
    [[ "$(<"${directory}/dependency.sha256")" == "${RUNTIME_DEPENDENCY_SHA256}" ]] || return 1
    [[ "$(runtime_dependency_identity)" == "${RUNTIME_DEPENDENCY_SHA256}" ]] || return 1
    observed_record="$("${PYTHON_COMMAND}" -I -S "${RUNTIME_INTEGRITY}" validate \
        --venv "${directory}/venv" --lock "${DEPENDENCY_LOCK}" \
        --seal "${directory}/record.sha256")" || return 1
    [[ "$(<"${directory}/record.sha256")" == "${observed_record}" ]]
}

prepare_dependency_layer() {
    local directory
    local record_digest
    local -a pip_args
    directory="$(runtime_dependency_directory)"
    if [[ -d "${directory}" ]]; then
        if validate_dependency_layer "${directory}"; then
            printf 'Reusing verified node dependency layer %s\n' "${RUNTIME_DEPENDENCY_SHA256}"
            return
        fi
        # Only a never-published build is disposable. An immutable layer can
        # still be referenced by the active or a rollback slot.
        [[ -f "${directory}/.building" && ! -f "${directory}/.complete" ]] ||
            die "published node dependency layer failed integrity validation"
        rm -rf "${directory}"
    fi
    install -d -m 0755 "${directory}"
    touch "${directory}/.building"
    "${PYTHON_COMMAND}" -m venv "${directory}/venv"
    pip_args=(install --require-hashes --no-deps --requirement "${DEPENDENCY_LOCK}")
    if [[ -n "${WHEELHOUSE}" ]]; then
        [[ -d "${WHEELHOUSE}" ]] || die "wheelhouse does not exist: ${WHEELHOUSE}"
        pip_args+=(--no-index --find-links "${WHEELHOUSE}")
    fi
    "${directory}/venv/bin/python" -I -m pip "${pip_args[@]}"
    record_digest="$("${PYTHON_COMMAND}" -I -S "${RUNTIME_INTEGRITY}" validate \
        --venv "${directory}/venv" --lock "${DEPENDENCY_LOCK}")" ||
        die "node dependency layer validation failed"
    printf '%s\n' "${record_digest}" > "${directory}/record.sha256"
    printf '%s\n' "${RUNTIME_DEPENDENCY_SHA256}" > "${directory}/dependency.sha256"
    install -m 0644 "${DEPENDENCY_LOCK}" "${directory}/dependency.lock"
    chmod 0644 "${directory}/record.sha256" "${directory}/dependency.sha256"
    touch "${directory}/.complete"
    chmod 0644 "${directory}/.complete"
    rm "${directory}/.building"
}

validate_runtime_slot() {
    local release_dir="$1"
    local expected_artifact="$2"
    local observed_record

    [[ -f "${release_dir}/.complete" ]] || return 1
    [[ -f "${release_dir}/artifact.sha256" ]] || return 1
    [[ -f "${release_dir}/record.sha256" ]] || return 1
    [[ -f "${release_dir}/dependency.sha256" ]] || return 1
    [[ "$(<"${release_dir}/artifact.sha256")" == "${expected_artifact}" ]] ||
        return 1
    [[ "$(<"${release_dir}/dependency.sha256")" == "${RUNTIME_DEPENDENCY_SHA256}" ]] ||
        return 1
    [[ -x "${release_dir}/venv/bin/gpu-fault-collector" ]] || return 1
    [[ -x "${release_dir}/venv/bin/gpu-fault-node-agent" ]] || return 1
    [[ -x "${release_dir}/venv/bin/gpu-fault-restore-gpu-services" ]] ||
        return 1
    validate_dependency_layer "$(runtime_dependency_directory)" || return 1
    observed_record="$(runtime_record_digest "${release_dir}")" || return 1
    [[ "$(<"${release_dir}/record.sha256")" == "${observed_record}" ]] || return 1
    "${release_dir}/venv/bin/python" -I -m pip check >/dev/null || return 1
    "${release_dir}/venv/bin/python" -I -c 'import gpu_fault' || return 1
    "${release_dir}/venv/bin/python" -I -B - <<'PY' || return 1
import importlib.metadata
import importlib.util

if importlib.util.find_spec("gpu_fault.collector_registry") is None:
    if importlib.metadata.entry_points().select(group="gpu_fault.collectors"):
        raise RuntimeError("runtime cannot validate installed Collector plugins")
else:
    from gpu_fault import collector_registry

    validate = getattr(collector_registry, "validate_collector_plugins", None)
    if validate is not None:
        validate()
    else:
        for descriptor in collector_registry.collector_registry_with_plugins().values():
            factory = importlib.metadata.EntryPoint(
                name=descriptor.cli_command,
                value=descriptor.factory,
                group="gpu_fault.collectors",
            ).load()
            if not callable(factory):
                raise RuntimeError("Collector factory is not callable")
PY
}

prepare_runtime_slot() {
    local release_dir="$1"
    local expected_artifact="$2"
    local record_digest
    local release_resolved
    local current_resolved
    local runtime_site
    local dependency_site

    if [[ -d "${release_dir}" ]]; then
        if validate_runtime_slot "${release_dir}" "${expected_artifact}"; then
            printf 'Reusing node runtime slot %s\n' "${expected_artifact}"
            return
        fi
        if [[ -n "${PREVIOUS_CURRENT_TARGET}" ]]; then
            release_resolved="$(readlink -f "${release_dir}")"
            current_resolved="$(readlink -f "${PREVIOUS_CURRENT_TARGET}")"
            if [[ "${release_resolved}" == "${current_resolved}" ]]; then
                die "active node runtime slot failed integrity validation"
            fi
        fi
        rm -rf "${release_dir}"
    fi
    prepare_dependency_layer
    install -d -m 0755 "${release_dir}"
    "${PYTHON_COMMAND}" -m venv "${release_dir}/venv"
    runtime_site="$("${PYTHON_COMMAND}" -I -S "${RUNTIME_INTEGRITY}" site --venv "${release_dir}/venv")"
    dependency_site="$("${PYTHON_COMMAND}" -I -S "${RUNTIME_INTEGRITY}" site \
        --venv "$(runtime_dependency_directory)/venv")"
    printf '%s\n' "${dependency_site}" > "${runtime_site}/gpu-fault-node-dependencies.pth"
    chmod 0644 "${runtime_site}/gpu-fault-node-dependencies.pth"
    "${release_dir}/venv/bin/python" -I -m pip install \
        --no-index --no-deps "${WHEEL}"
    record_digest="$(runtime_record_digest "${release_dir}")" ||
        die "node runtime RECORD validation failed"
    printf '%s\n' "${expected_artifact}" > "${release_dir}/artifact.sha256"
    printf '%s\n' "${record_digest}" > "${release_dir}/record.sha256"
    printf '%s\n' "${RUNTIME_DEPENDENCY_SHA256}" > "${release_dir}/dependency.sha256"
    chmod 0644 "${release_dir}/artifact.sha256" \
        "${release_dir}/record.sha256" "${release_dir}/dependency.sha256"
    touch "${release_dir}/.complete"
    chmod 0644 "${release_dir}/.complete"
    validate_runtime_slot "${release_dir}" "${expected_artifact}" ||
        die "candidate node runtime slot validation failed"
}
