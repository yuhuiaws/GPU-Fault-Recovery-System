#!/usr/bin/env bash
# Apply the checked-in role-split control-plane manifests and prove the
# split actually took.
#
# deploy.sh applies these manifests itself, ahead of the environment it
# sets on both tiers, and then runs only the verify step. This script is
# the standalone path: enabling or repairing the split on a control
# plane that is already deployed. Pass the release's wheel ConfigMap in
# GPU_FAULT_WHEEL_CONFIGMAP, or the pods come up mounting the
# unversioned name from the base manifest and stay in
# CreateContainerConfigError.
#
# Replaces enable-control-plane-role-split.sh, which rendered the split
# from whatever was already running. Here the manifests come from the
# repo (regional/generated/, produced by
# render-control-plane-role-split.sh), so a rebuilt cluster gets the
# same three tiers, and the verify step below fails the deploy when a
# consumer tier is missing or still carrying the ingress role - the
# failure mode that otherwise looks healthy: every replica Ready, every
# request accepted with a 202, and nothing ever claimed off the queue.
set -euo pipefail

NAMESPACE="${GPU_FAULT_NAMESPACE:-gpu-fault-system}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
GENERATED="${GPU_FAULT_ROLE_SPLIT_GENERATED_DIR:-${SCRIPT_DIR}/../regional/generated}"
KUBECONFIG_PATH="${KUBECONFIG:-}"
DEFAULT_RUNTIME_IMAGE="public.ecr.aws/docker/library/python:3.12-slim"
RUNTIME_IMAGE="${GPU_FAULT_RUNTIME_IMAGE:-${DEFAULT_RUNTIME_IMAGE}}"
FAST_ROLLOUT_TIMEOUT="5m"
FAST_ROLLOUT_TIMEOUT_SECONDS=$((10#${FAST_ROLLOUT_TIMEOUT%m} * 60))
AWS_REGION="${GPU_FAULT_AWS_REGION:?GPU_FAULT_AWS_REGION is required}"
RUNTIME_PROFILE_VERSION="$(
    printf '%s' \
        "${GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION:?GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION is required}"
)"
ALLOW_EMAIL="${GPU_FAULT_ALLOW_EMAIL:-true}"
ACKNOWLEDGE_NO_ALERT_CHANNEL="$(
    printf '%s' "${GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL:-false}"
)"
NOTIFICATION_CONFIG_SHA256="$(
    printf '%s' "${GPU_FAULT_NOTIFICATION_CONFIG_SHA256:-}"
)"
LEGACY_COMPONENT_PINS="${GPU_FAULT_LEGACY_COMPONENT_PINS:-false}"
PRESERVE_ROLE_CONFIG_MAPS="${GPU_FAULT_PRESERVE_ROLE_CONFIG_MAPS:-false}"
FORCE_ROLE_RESTART="${GPU_FAULT_FORCE_ROLE_RESTART:-false}"
CONTRACT_DIR="$(mktemp -d)"
trap 'rm -rf "${CONTRACT_DIR}"' EXIT

[[ "${AWS_REGION}" =~ ^[a-z0-9]+(-[a-z0-9]+)+-[0-9]+$ ]] || {
    echo "GPU_FAULT_AWS_REGION is not a valid AWS Region: ${AWS_REGION}" >&2
    exit 2
}
[[ -n "${RUNTIME_IMAGE}" &&
    "${RUNTIME_IMAGE}" != *[[:space:]#]* ]] || {
    echo "GPU_FAULT_RUNTIME_IMAGE must be a non-empty OCI image reference without whitespace or #" >&2
    exit 2
}
[[ "${RUNTIME_PROFILE_VERSION}" =~ ^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$ ]] || {
    echo "GPU_FAULT_REQUIRED_RUNTIME_PROFILE_VERSION is invalid" >&2
    exit 2
}
[[ "${ALLOW_EMAIL}" == "true" || "${ALLOW_EMAIL}" == "false" ]] || {
    echo "GPU_FAULT_ALLOW_EMAIL must be true or false" >&2
    exit 2
}
[[ "${ACKNOWLEDGE_NO_ALERT_CHANNEL}" == "true" ||
    "${ACKNOWLEDGE_NO_ALERT_CHANNEL}" == "false" ]] || {
    echo "GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL must be true or false" >&2
    exit 2
}
[[ "${NOTIFICATION_CONFIG_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "GPU_FAULT_NOTIFICATION_CONFIG_SHA256 must be a lowercase SHA-256" >&2
    exit 2
}
[[ "${LEGACY_COMPONENT_PINS}" == "true" ||
    "${LEGACY_COMPONENT_PINS}" == "false" ]] || {
    echo "GPU_FAULT_LEGACY_COMPONENT_PINS must be true or false" >&2
    exit 2
}
[[ "${PRESERVE_ROLE_CONFIG_MAPS}" == "true" ||
    "${PRESERVE_ROLE_CONFIG_MAPS}" == "false" ]] || {
    echo "GPU_FAULT_PRESERVE_ROLE_CONFIG_MAPS must be true or false" >&2
    exit 2
}
[[ "${FORCE_ROLE_RESTART}" == "true" ||
    "${FORCE_ROLE_RESTART}" == "false" ]] || {
    echo "GPU_FAULT_FORCE_ROLE_RESTART must be true or false" >&2
    exit 2
}
if [[ "${ALLOW_EMAIL}" != "true" &&
    "${ACKNOWLEDGE_NO_ALERT_CHANNEL}" != "true" ]]; then
    echo "email must be enabled or the external alert channel acknowledged" >&2
    exit 2
fi

kubectl_args=()
if [[ -n "${KUBECONFIG_PATH}" ]]; then
    kubectl_args+=(--kubeconfig "${KUBECONFIG_PATH}")
fi

if [[ ! -f "${GENERATED}/gpu-fault-control-worker.yaml" ]]; then
    echo "missing ${GENERATED}; run render-control-plane-role-split.sh" >&2
    exit 1
fi
if [[ ! -f "${GENERATED}/manifest-list.txt" ]]; then
    echo "missing ${GENERATED}/manifest-list.txt; re-render the role split" >&2
    exit 1
fi

grep -Ev '^(#|[[:space:]]*$)' "${GENERATED}/manifest-list.txt" |
    sort >"${CONTRACT_DIR}/expected.txt"
find "${GENERATED}" -maxdepth 1 -type f \
    -name 'gpu-fault-*.yaml' -printf '%f\n' |
    sort >"${CONTRACT_DIR}/actual.txt"
if ! diff -u \
    "${CONTRACT_DIR}/expected.txt" \
    "${CONTRACT_DIR}/actual.txt"; then
    echo "generated manifest set differs from manifest-list.txt; re-render instead of adding files by hand" >&2
    exit 1
fi

WHEEL_CONFIGMAP="${GPU_FAULT_WHEEL_CONFIGMAP:-gpu-fault-control-plane-wheel-0100}"
WHEEL_SHA256="${GPU_FAULT_WHEEL_SHA256:-}"
if [[ -z "${WHEEL_SHA256}" ]]; then
    WHEEL_SHA256="$(
        kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get configmap \
            "${WHEEL_CONFIGMAP}" \
            -o jsonpath='{.binaryData.gpu_fault_control_plane-0\.10\.0-py3-none-any\.whl}' |
            base64 -d |
            sha256sum |
            awk '{print $1}'
    )"
fi
[[ "${WHEEL_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "cannot derive wheel SHA-256 from ${WHEEL_CONFIGMAP}" >&2
    exit 1
}
RELEASE_ID="${WHEEL_SHA256:0:12}"

REQUIRED_AGENT_ARTIFACT_SHA256="${GPU_FAULT_REQUIRED_AGENT_ARTIFACT_SHA256:-${WHEEL_SHA256}}"
REQUIRED_AGENT_COMPATIBILITY_DIGEST="${GPU_FAULT_REQUIRED_AGENT_COMPATIBILITY_DIGEST:-${REQUIRED_AGENT_ARTIFACT_SHA256}}"
REQUIRED_AGENT_CONFIG_DIGEST="${GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST:-}"
REQUIRED_NODE_ACTION_KEY_VERSION="${GPU_FAULT_REQUIRED_NODE_ACTION_KEY_VERSION:-2}"
REQUIRED_AGENT_PROTOCOL_VERSION="$(
    PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}" \
        python3 -c \
        'from gpu_fault.fleet import CURRENT_AGENT_PROTOCOL_VERSION; print(CURRENT_AGENT_PROTOCOL_VERSION)'
)"
REQUIRED_AGENT_PROTOCOL_VERSION="${GPU_FAULT_REQUIRED_AGENT_PROTOCOL_VERSION:-${REQUIRED_AGENT_PROTOCOL_VERSION}}"
REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION="$(
    PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}" \
        python3 -c \
        'from gpu_fault.regional_compatibility import CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION; print(CURRENT_REGIONAL_EXECUTOR_PROTOCOL_VERSION)'
)"
REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION="${GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION:-${REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION}}"
REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256="${GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256:-}"
REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST="${GPU_FAULT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST:-${REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256}}"
COMPATIBLE_AGENT_ARTIFACT_SHA256S="${GPU_FAULT_COMPATIBLE_AGENT_ARTIFACT_SHA256S:-}"
COMPATIBLE_AGENT_COMPATIBILITY_DIGESTS="${GPU_FAULT_COMPATIBLE_AGENT_COMPATIBILITY_DIGESTS:-}"
COMPATIBLE_AGENT_PROTOCOL_VERSIONS="${GPU_FAULT_COMPATIBLE_AGENT_PROTOCOL_VERSIONS:-}"
COMPATIBLE_AGENT_CONFIG_DIGESTS="${GPU_FAULT_COMPATIBLE_AGENT_CONFIG_DIGESTS:-}"
COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS="${GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS:-}"
COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S="${GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S:-}"
COMPATIBLE_REGIONAL_EXECUTOR_COMPATIBILITY_DIGESTS="${GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_COMPATIBILITY_DIGESTS:-}"
FINALIZE_AGENT_PIN="${GPU_FAULT_FINALIZE_AGENT_PIN:-false}"
FINALIZE_DATA_PLANE_PIN="${GPU_FAULT_FINALIZE_DATA_PLANE_PIN:-${FINALIZE_AGENT_PIN}}"
PIN_METADATA_CHANGED="false"
TARGET_AGENT_ARTIFACT_SHA256="${REQUIRED_AGENT_ARTIFACT_SHA256}"
TARGET_AGENT_COMPATIBILITY_DIGEST="${REQUIRED_AGENT_COMPATIBILITY_DIGEST}"
TARGET_AGENT_CONFIG_DIGEST="${REQUIRED_AGENT_CONFIG_DIGEST}"
TARGET_AGENT_PROTOCOL_VERSION="${REQUIRED_AGENT_PROTOCOL_VERSION}"
TARGET_REGIONAL_EXECUTOR_PROTOCOL_VERSION="${REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION}"
TARGET_REGIONAL_EXECUTOR_ARTIFACT_SHA256="${REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256}"
TARGET_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST="${REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST}"

append_csv_value() {
    local current="$1"
    local value="$2"
    if [[ -z "${value}" ]]; then
        printf '%s' "${current}"
    elif [[ ",${current}," == *",${value},"* ]]; then
        printf '%s' "${current}"
    elif [[ -n "${current}" ]]; then
        printf '%s,%s' "${current}" "${value}"
    else
        printf '%s' "${value}"
    fi
}

if kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get configmap \
    gpu-fault-release-metadata >/dev/null 2>&1; then
    CURRENT_REQUIRED_AGENT_ARTIFACT_SHA256="$(
        kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get configmap \
            gpu-fault-release-metadata \
            -o jsonpath='{.data.required-agent-artifact-sha256}'
    )"
    CURRENT_REQUIRED_AGENT_PROTOCOL_VERSION="$(
        kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get configmap \
            gpu-fault-release-metadata \
            -o jsonpath='{.data.required-agent-protocol-version}'
    )"
    CURRENT_REQUIRED_AGENT_CONFIG_DIGEST="$(
        kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get configmap \
            gpu-fault-release-metadata \
            -o jsonpath='{.data.required-agent-config-digest}'
    )"
    CURRENT_COMPATIBLE_AGENT_ARTIFACT_SHA256S="$(
        kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get configmap \
            gpu-fault-release-metadata \
            -o jsonpath='{.data.compatible-agent-artifact-sha256s}'
    )"
    CURRENT_REQUIRED_AGENT_COMPATIBILITY_DIGEST="$(
        kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get configmap \
            gpu-fault-release-metadata \
            -o jsonpath='{.data.required-agent-compatibility-digest}'
    )"
    CURRENT_COMPATIBLE_AGENT_COMPATIBILITY_DIGESTS="$(
        kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get configmap \
            gpu-fault-release-metadata \
            -o jsonpath='{.data.compatible-agent-compatibility-digests}'
    )"
    CURRENT_COMPATIBLE_AGENT_PROTOCOL_VERSIONS="$(
        kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get configmap \
            gpu-fault-release-metadata \
            -o jsonpath='{.data.compatible-agent-protocol-versions}'
    )"
    CURRENT_COMPATIBLE_AGENT_CONFIG_DIGESTS="$(
        kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get configmap \
            gpu-fault-release-metadata \
            -o jsonpath='{.data.compatible-agent-config-digests}'
    )"
    CURRENT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION="$(
        kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get configmap \
            gpu-fault-release-metadata \
            -o jsonpath='{.data.required-regional-executor-protocol-version}'
    )"
    CURRENT_COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS="$(
        kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get configmap \
            gpu-fault-release-metadata \
            -o jsonpath='{.data.compatible-regional-executor-protocol-versions}'
    )"
    CURRENT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256="$(
        kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get configmap \
            gpu-fault-release-metadata \
            -o jsonpath='{.data.required-regional-executor-artifact-sha256}'
    )"
    CURRENT_COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S="$(
        kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get configmap \
            gpu-fault-release-metadata \
            -o jsonpath='{.data.compatible-regional-executor-artifact-sha256s}'
    )"
    CURRENT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST="$(
        kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get configmap \
            gpu-fault-release-metadata \
            -o jsonpath='{.data.required-regional-executor-compatibility-digest}'
    )"
    CURRENT_COMPATIBLE_REGIONAL_EXECUTOR_COMPATIBILITY_DIGESTS="$(
        kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get configmap \
            gpu-fault-release-metadata \
            -o jsonpath='{.data.compatible-regional-executor-compatibility-digests}'
    )"
    CURRENT_REQUIRED_AGENT_PROTOCOL_VERSION="${CURRENT_REQUIRED_AGENT_PROTOCOL_VERSION:-${GPU_FAULT_EXISTING_AGENT_PROTOCOL_VERSION:-3}}"
    if [[ "${FINALIZE_AGENT_PIN}" != "true" ]]; then
        if [[ -z "${GPU_FAULT_COMPATIBLE_AGENT_ARTIFACT_SHA256S+x}" ]]; then
            COMPATIBLE_AGENT_ARTIFACT_SHA256S="${CURRENT_COMPATIBLE_AGENT_ARTIFACT_SHA256S}"
        fi
        if [[ -z "${GPU_FAULT_COMPATIBLE_AGENT_PROTOCOL_VERSIONS+x}" ]]; then
            COMPATIBLE_AGENT_PROTOCOL_VERSIONS="${CURRENT_COMPATIBLE_AGENT_PROTOCOL_VERSIONS}"
        fi
        if [[ -z "${GPU_FAULT_COMPATIBLE_AGENT_CONFIG_DIGESTS+x}" ]]; then
            COMPATIBLE_AGENT_CONFIG_DIGESTS="${CURRENT_COMPATIBLE_AGENT_CONFIG_DIGESTS}"
        fi
        if [[ -n "${CURRENT_REQUIRED_AGENT_ARTIFACT_SHA256}" ]] &&
            [[ "${CURRENT_REQUIRED_AGENT_ARTIFACT_SHA256}" != "${TARGET_AGENT_ARTIFACT_SHA256}" ]]; then
            COMPATIBLE_AGENT_ARTIFACT_SHA256S="$(
                append_csv_value \
                    "${COMPATIBLE_AGENT_ARTIFACT_SHA256S}" \
                    "${TARGET_AGENT_ARTIFACT_SHA256}"
            )"
            REQUIRED_AGENT_ARTIFACT_SHA256="${CURRENT_REQUIRED_AGENT_ARTIFACT_SHA256}"
        fi
        if [[ -z "${GPU_FAULT_COMPATIBLE_AGENT_COMPATIBILITY_DIGESTS+x}" ]]; then
            COMPATIBLE_AGENT_COMPATIBILITY_DIGESTS="${CURRENT_COMPATIBLE_AGENT_COMPATIBILITY_DIGESTS}"
        fi
        CURRENT_EFFECTIVE_AGENT_COMPATIBILITY_DIGEST="${CURRENT_REQUIRED_AGENT_COMPATIBILITY_DIGEST:-${CURRENT_REQUIRED_AGENT_ARTIFACT_SHA256}}"
        if [[ -n "${CURRENT_EFFECTIVE_AGENT_COMPATIBILITY_DIGEST}" ]] &&
            [[ "${CURRENT_EFFECTIVE_AGENT_COMPATIBILITY_DIGEST}" != "${TARGET_AGENT_COMPATIBILITY_DIGEST}" ]]; then
            COMPATIBLE_AGENT_COMPATIBILITY_DIGESTS="$(
                append_csv_value \
                    "${COMPATIBLE_AGENT_COMPATIBILITY_DIGESTS}" \
                    "${TARGET_AGENT_COMPATIBILITY_DIGEST}"
            )"
            REQUIRED_AGENT_COMPATIBILITY_DIGEST="${CURRENT_EFFECTIVE_AGENT_COMPATIBILITY_DIGEST}"
        elif [[ -n "${CURRENT_EFFECTIVE_AGENT_COMPATIBILITY_DIGEST}" ]]; then
            REQUIRED_AGENT_COMPATIBILITY_DIGEST="${CURRENT_EFFECTIVE_AGENT_COMPATIBILITY_DIGEST}"
        fi
        if [[ "${CURRENT_REQUIRED_AGENT_PROTOCOL_VERSION}" != "${TARGET_AGENT_PROTOCOL_VERSION}" ]]; then
            COMPATIBLE_AGENT_PROTOCOL_VERSIONS="$(
                append_csv_value \
                    "${COMPATIBLE_AGENT_PROTOCOL_VERSIONS}" \
                    "${TARGET_AGENT_PROTOCOL_VERSION}"
            )"
            REQUIRED_AGENT_PROTOCOL_VERSION="${CURRENT_REQUIRED_AGENT_PROTOCOL_VERSION}"
        fi
        if [[ -n "${CURRENT_REQUIRED_AGENT_CONFIG_DIGEST}" ]] &&
            [[ "${CURRENT_REQUIRED_AGENT_CONFIG_DIGEST}" != "${TARGET_AGENT_CONFIG_DIGEST}" ]]; then
            COMPATIBLE_AGENT_CONFIG_DIGESTS="$(
                append_csv_value \
                    "${COMPATIBLE_AGENT_CONFIG_DIGESTS}" \
                    "${TARGET_AGENT_CONFIG_DIGEST}"
            )"
            REQUIRED_AGENT_CONFIG_DIGEST="${CURRENT_REQUIRED_AGENT_CONFIG_DIGEST}"
        fi
    else
        COMPATIBLE_AGENT_ARTIFACT_SHA256S=""
        COMPATIBLE_AGENT_COMPATIBILITY_DIGESTS=""
        COMPATIBLE_AGENT_PROTOCOL_VERSIONS=""
        COMPATIBLE_AGENT_CONFIG_DIGESTS=""
    fi
    CURRENT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION="${CURRENT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION:-${GPU_FAULT_EXISTING_REGIONAL_EXECUTOR_PROTOCOL_VERSION:-1}}"
    if [[ "${FINALIZE_DATA_PLANE_PIN}" != "true" ]]; then
        if [[ -z "${GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS+x}" ]]; then
            COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS="${CURRENT_COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS}"
        fi
        if [[ "${CURRENT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION}" != "${TARGET_REGIONAL_EXECUTOR_PROTOCOL_VERSION}" ]]; then
            COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS="$(
                append_csv_value \
                    "${COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS}" \
                    "${TARGET_REGIONAL_EXECUTOR_PROTOCOL_VERSION}"
            )"
            REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION="${CURRENT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION}"
        fi
        if [[ -z "${GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S+x}" ]]; then
            COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S="${CURRENT_COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S}"
        fi
        if [[ -n "${CURRENT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256}" ]] &&
            [[ "${CURRENT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256}" != "${TARGET_REGIONAL_EXECUTOR_ARTIFACT_SHA256}" ]]; then
            COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S="$(
                append_csv_value \
                    "${COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S}" \
                    "${TARGET_REGIONAL_EXECUTOR_ARTIFACT_SHA256}"
            )"
            REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256="${CURRENT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256}"
        elif [[ -z "${CURRENT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256}" ]]; then
            # Legacy control planes did not pin the executor artifact.
            # Keep the staged window unpinned, roll the executors, then let
            # finalize establish the first required artifact.
            REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256=""
        fi
        if [[ -z "${GPU_FAULT_COMPATIBLE_REGIONAL_EXECUTOR_COMPATIBILITY_DIGESTS+x}" ]]; then
            COMPATIBLE_REGIONAL_EXECUTOR_COMPATIBILITY_DIGESTS="${CURRENT_COMPATIBLE_REGIONAL_EXECUTOR_COMPATIBILITY_DIGESTS}"
        fi
        if [[ -n "${CURRENT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST}" ]] &&
            [[ "${CURRENT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST}" != "${TARGET_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST}" ]]; then
            COMPATIBLE_REGIONAL_EXECUTOR_COMPATIBILITY_DIGESTS="$(
                append_csv_value \
                    "${COMPATIBLE_REGIONAL_EXECUTOR_COMPATIBILITY_DIGESTS}" \
                    "${TARGET_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST}"
            )"
            REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST="${CURRENT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST}"
        elif [[ -z "${CURRENT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST}" ]]; then
            REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST=""
        fi
    else
        COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS=""
        COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S=""
        COMPATIBLE_REGIONAL_EXECUTOR_COMPATIBILITY_DIGESTS=""
    fi
    if [[ "${CURRENT_REQUIRED_AGENT_ARTIFACT_SHA256}" != "${REQUIRED_AGENT_ARTIFACT_SHA256}" ||
        "${CURRENT_COMPATIBLE_AGENT_ARTIFACT_SHA256S}" != "${COMPATIBLE_AGENT_ARTIFACT_SHA256S}" ||
        "${CURRENT_REQUIRED_AGENT_COMPATIBILITY_DIGEST}" != "${REQUIRED_AGENT_COMPATIBILITY_DIGEST}" ||
        "${CURRENT_COMPATIBLE_AGENT_COMPATIBILITY_DIGESTS}" != "${COMPATIBLE_AGENT_COMPATIBILITY_DIGESTS}" ||
        "${CURRENT_REQUIRED_AGENT_PROTOCOL_VERSION}" != "${REQUIRED_AGENT_PROTOCOL_VERSION}" ||
        "${CURRENT_COMPATIBLE_AGENT_PROTOCOL_VERSIONS}" != "${COMPATIBLE_AGENT_PROTOCOL_VERSIONS}" ||
        "${CURRENT_REQUIRED_AGENT_CONFIG_DIGEST}" != "${REQUIRED_AGENT_CONFIG_DIGEST}" ||
        "${CURRENT_COMPATIBLE_AGENT_CONFIG_DIGESTS}" != "${COMPATIBLE_AGENT_CONFIG_DIGESTS}" ||
        "${CURRENT_REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION}" != "${REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION}" ||
        "${CURRENT_COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS}" != "${COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS}" ||
        "${CURRENT_REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256}" != "${REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256}" ||
        "${CURRENT_COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S}" != "${COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S}" ||
        "${CURRENT_REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST}" != "${REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST}" ||
        "${CURRENT_COMPATIBLE_REGIONAL_EXECUTOR_COMPATIBILITY_DIGESTS}" != "${COMPATIBLE_REGIONAL_EXECUTOR_COMPATIBILITY_DIGESTS}" ]]; then
        PIN_METADATA_CHANGED="true"
    fi
fi
# ConfigMap consumers need a restart only when the effective pin metadata
# changed. Re-running an already-finalized release is therefore a no-op.
RELOAD_RELEASE_METADATA="${PIN_METADATA_CHANGED}"
if [[ -n "${REQUIRED_AGENT_CONFIG_DIGEST}" ]]; then
    [[ "${REQUIRED_AGENT_ARTIFACT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
        echo "required agent artifact SHA-256 is invalid" >&2
        exit 1
    }
    [[ "${REQUIRED_AGENT_CONFIG_DIGEST}" =~ ^[0-9a-f]{64}$ ]] || {
        echo "required agent config digest is invalid" >&2
        exit 1
    }
    [[ -z "${REQUIRED_AGENT_COMPATIBILITY_DIGEST}" ||
        "${REQUIRED_AGENT_COMPATIBILITY_DIGEST}" =~ ^[0-9a-f]{64}$ ]] || {
        echo "required agent compatibility digest is invalid" >&2
        exit 1
    }
    [[ "${REQUIRED_AGENT_PROTOCOL_VERSION}" =~ ^[1-9][0-9]*$ ]] || {
        echo "required agent protocol version is invalid" >&2
        exit 1
    }
    [[ "${REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION}" =~ ^[1-9][0-9]*$ ]] || {
        echo "required regional executor protocol version is invalid" >&2
        exit 1
    }
    [[ -z "${REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256}" ||
        "${REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
        echo "required regional executor artifact SHA-256 is invalid" >&2
        exit 1
    }
    [[ -z "${REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST}" ||
        "${REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST}" =~ ^[0-9a-f]{64}$ ]] || {
        echo "required regional executor compatibility digest is invalid" >&2
        exit 1
    }
    IFS=',' read -r -a compatible_artifacts <<< \
        "${COMPATIBLE_AGENT_ARTIFACT_SHA256S}"
    for artifact in "${compatible_artifacts[@]}"; do
        [[ -z "${artifact}" || "${artifact}" =~ ^[0-9a-f]{64}$ ]] || {
            echo "compatible agent artifact SHA-256 is invalid" >&2
            exit 1
        }
    done
    IFS=',' read -r -a compatible_agent_digests <<< \
        "${COMPATIBLE_AGENT_COMPATIBILITY_DIGESTS}"
    for digest in "${compatible_agent_digests[@]}"; do
        [[ -z "${digest}" || "${digest}" =~ ^[0-9a-f]{64}$ ]] || {
            echo "compatible agent compatibility digest is invalid" >&2
            exit 1
        }
    done
    IFS=',' read -r -a compatible_protocols <<< \
        "${COMPATIBLE_AGENT_PROTOCOL_VERSIONS}"
    for protocol in "${compatible_protocols[@]}"; do
        [[ -z "${protocol}" || "${protocol}" =~ ^[1-9][0-9]*$ ]] || {
            echo "compatible agent protocol version is invalid" >&2
            exit 1
        }
    done
    IFS=',' read -r -a compatible_configs <<< \
        "${COMPATIBLE_AGENT_CONFIG_DIGESTS}"
    for digest in "${compatible_configs[@]}"; do
        [[ -z "${digest}" || "${digest}" =~ ^[0-9a-f]{64}$ ]] || {
            echo "compatible agent config digest is invalid" >&2
            exit 1
        }
    done
    IFS=',' read -r -a compatible_executor_protocols <<< \
        "${COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS}"
    for protocol in "${compatible_executor_protocols[@]}"; do
        [[ -z "${protocol}" || "${protocol}" =~ ^[1-9][0-9]*$ ]] || {
            echo "compatible regional executor protocol version is invalid" >&2
            exit 1
        }
    done
    IFS=',' read -r -a compatible_executor_artifacts <<< \
        "${COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S}"
    for artifact in "${compatible_executor_artifacts[@]}"; do
        [[ -z "${artifact}" || "${artifact}" =~ ^[0-9a-f]{64}$ ]] || {
            echo "compatible regional executor artifact SHA-256 is invalid" >&2
            exit 1
        }
    done
    IFS=',' read -r -a compatible_executor_digests <<< \
        "${COMPATIBLE_REGIONAL_EXECUTOR_COMPATIBILITY_DIGESTS}"
    for digest in "${compatible_executor_digests[@]}"; do
        [[ -z "${digest}" || "${digest}" =~ ^[0-9a-f]{64}$ ]] || {
            echo "compatible regional executor compatibility digest is invalid" >&2
            exit 1
        }
    done
    kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" create configmap \
        gpu-fault-release-metadata \
        --from-literal=required-agent-artifact-sha256="${REQUIRED_AGENT_ARTIFACT_SHA256}" \
        --from-literal=compatible-agent-artifact-sha256s="${COMPATIBLE_AGENT_ARTIFACT_SHA256S}" \
        --from-literal=required-agent-compatibility-digest="${REQUIRED_AGENT_COMPATIBILITY_DIGEST}" \
        --from-literal=compatible-agent-compatibility-digests="${COMPATIBLE_AGENT_COMPATIBILITY_DIGESTS}" \
        --from-literal=required-agent-protocol-version="${REQUIRED_AGENT_PROTOCOL_VERSION}" \
        --from-literal=compatible-agent-protocol-versions="${COMPATIBLE_AGENT_PROTOCOL_VERSIONS}" \
        --from-literal=required-regional-executor-protocol-version="${REQUIRED_REGIONAL_EXECUTOR_PROTOCOL_VERSION}" \
        --from-literal=compatible-regional-executor-protocol-versions="${COMPATIBLE_REGIONAL_EXECUTOR_PROTOCOL_VERSIONS}" \
        --from-literal=required-regional-executor-artifact-sha256="${REQUIRED_REGIONAL_EXECUTOR_ARTIFACT_SHA256}" \
        --from-literal=compatible-regional-executor-artifact-sha256s="${COMPATIBLE_REGIONAL_EXECUTOR_ARTIFACT_SHA256S}" \
        --from-literal=required-regional-executor-compatibility-digest="${REQUIRED_REGIONAL_EXECUTOR_COMPATIBILITY_DIGEST}" \
        --from-literal=compatible-regional-executor-compatibility-digests="${COMPATIBLE_REGIONAL_EXECUTOR_COMPATIBILITY_DIGESTS}" \
        --from-literal=required-agent-config-digest="${REQUIRED_AGENT_CONFIG_DIGEST}" \
        --from-literal=compatible-agent-config-digests="${COMPATIBLE_AGENT_CONFIG_DIGESTS}" \
        --from-literal=required-node-action-key-version="${REQUIRED_NODE_ACTION_KEY_VERSION}" \
        --dry-run=client -o yaml |
        kubectl "${kubectl_args[@]}" apply -f -
elif ! kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get configmap \
    gpu-fault-release-metadata >/dev/null 2>&1; then
    echo "gpu-fault-release-metadata is missing; set GPU_FAULT_REQUIRED_AGENT_CONFIG_DIGEST for a greenfield deploy" >&2
    exit 1
fi

PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}" python3 -m \
    gpu_fault.config_cli validate "${GENERATED}"

apply_manifest() {
    local manifest="$1"
    render_manifest "${manifest}" |
        if [[ "${LEGACY_COMPONENT_PINS}" == "true" ]]; then
            PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}" python3 \
                "${SCRIPT_DIR}/filter_legacy_release_env.py"
        else
            cat
        fi |
        kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" apply -f -
}

remove_legacy_notification_env() {
    local deployment="$1"
    if kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get deployment \
        "${deployment}" >/dev/null 2>&1; then
        kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" set env \
            "deployment/${deployment}" \
            GPU_FAULT_ALLOW_EMAIL- \
            GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL-
    fi
}

render_manifest() {
    local manifest="$1"
    sed \
        -e "s/gpu-fault-control-plane-wheel-0100/${WHEEL_CONFIGMAP}/g" \
        -e "s/namespace: gpu-fault-system/namespace: ${NAMESPACE}/g" \
        -e "s/REPLACE_WITH_AWS_REGION/${AWS_REGION}/g" \
        -e "s#REPLACE_WITH_RUNTIME_PROFILE_VERSION#${RUNTIME_PROFILE_VERSION}#g" \
        -e "s#gpu-fault.io/artifact-sha256: .*#gpu-fault.io/artifact-sha256: ${WHEEL_SHA256}#g" \
        -e "s/GPU_FAULT_ALLOW_EMAIL: 'true'/GPU_FAULT_ALLOW_EMAIL: '${ALLOW_EMAIL}'/g" \
        -e "s/GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL: 'false'/GPU_FAULT_ACKNOWLEDGE_NO_ALERT_CHANNEL: '${ACKNOWLEDGE_NO_ALERT_CHANNEL}'/g" \
        -e "s#${DEFAULT_RUNTIME_IMAGE}#${RUNTIME_IMAGE}#g" \
        "${GENERATED}/${manifest}.yaml"
}

stamp_release() {
    local deployment="$1"
    kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" patch deployment \
        "${deployment}" --type=merge -p "$(
            jq -nc \
                --arg sha "${WHEEL_SHA256}" \
                --arg release "${RELEASE_ID}" \
                --arg runtime_image "${RUNTIME_IMAGE}" \
                --arg notification "${NOTIFICATION_CONFIG_SHA256}" \
                '{
                    spec: {
                        template: {
                            metadata: {
                                annotations: {
                                    "gpu-fault.io/artifact-sha256": $sha,
                                    "gpu-fault.io/control-plane-wheel-sha256": $sha,
                                    "gpu-fault.io/release-wheel-sha256": $sha,
                                    "gpu-fault.io/release-rollout": $release,
                                    "gpu-fault.io/runtime-image": $runtime_image,
                                    "gpu-fault.io/notification-config-sha256": $notification
                                }
                            }
                        }
                    }
                }'
        )"
}

startup_failure_reason() {
    local deployment="$1"
    local deployment_uid
    local revision
    local template_hash
    local pod
    local logs
    deployment_uid="$(
        kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get deployment \
            "${deployment}" -o jsonpath='{.metadata.uid}'
    )"
    revision="$(
        kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get deployment \
            "${deployment}" \
            -o jsonpath='{.metadata.annotations.deployment\.kubernetes\.io/revision}'
    )"
    template_hash="$(
        kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get replicasets \
            -l "app=${deployment}" -o json |
            jq -r \
                --arg uid "${deployment_uid}" \
                --arg revision "${revision}" \
                '.items[] |
                 select(
                   (.metadata.ownerReferences // []) |
                   any(.uid == $uid)
                 ) |
                 select(
                   (.metadata.annotations["deployment.kubernetes.io/revision"] // "") ==
                   $revision
                 ) |
                 .metadata.labels["pod-template-hash"]' |
            head -n 1
    )"
    [[ -n "${template_hash}" ]] || return 1
    while IFS= read -r pod; do
        [[ -n "${pod}" ]] || continue
        logs="$(
            kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" logs \
                "pod/${pod}" --all-containers=true --tail=200 --since=10m \
                2>/dev/null || true
        )"
        if grep -Fq \
            "received unknown GPU_FAULT_* environment variable(s)" \
            <<<"${logs}"; then
            printf 'unknown GPU_FAULT environment variable'
            return 0
        fi
        if grep -Fq \
            "regional cluster registrations require agent_endpoint_allowed_cidrs" \
            <<<"${logs}"; then
            printf 'regional cluster registry is missing Agent endpoint CIDRs'
            return 0
        fi
        if grep -Fq \
            "enabled regional cluster requires agent endpoint CIDRs" \
            <<<"${logs}"; then
            printf 'durable regional cluster records require CIDR migration'
            return 0
        fi
        if grep -Fq "validation error for RegionalClusterRegistration" \
            <<<"${logs}" &&
            grep -Fq "Extra inputs are not permitted" <<<"${logs}"; then
            printf 'regional cluster registry is incompatible with the target wheel'
            return 0
        fi
    done < <(
        kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get pods \
            -l "app=${deployment}" -o json |
            jq -r \
                --arg release "${RELEASE_ID}" \
                --arg template_hash "${template_hash}" \
                '.items[] |
                 select(
                   (.metadata.annotations["gpu-fault.io/release-rollout"] // "") ==
                   $release
                 ) |
                 select(
                   (.metadata.labels["pod-template-hash"] // "") == $template_hash
                 ) |
                 .metadata.name'
    )
    return 1
}

wait_for_rollout() {
    local deployment="$1"
    local deadline=$((SECONDS + FAST_ROLLOUT_TIMEOUT_SECONDS))
    local reason
    while ((SECONDS < deadline)); do
        if kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" rollout status \
            "deployment/${deployment}" --timeout=5s >/dev/null 2>&1; then
            return 0
        fi
        if reason="$(startup_failure_reason "${deployment}")"; then
            echo "deployment ${deployment} failed fast: ${reason}" >&2
            return 1
        fi
    done
    kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" rollout status \
        "deployment/${deployment}" --timeout=1s
}

if [[ "${PRESERVE_ROLE_CONFIG_MAPS}" != "true" ]]; then
    for config in "${GENERATED}"/gpu-fault-*-config-*.yaml; do
        name="$(basename "${config}" .yaml)"
        apply_manifest "${name}"
    done
fi

apply_manifest gpu-fault-api-ha-pdb
apply_manifest gpu-fault-control-worker-pdb
apply_manifest gpu-fault-telemetry-spool-worker-pdb
remove_legacy_notification_env gpu-fault-telemetry-spool-worker
apply_manifest gpu-fault-telemetry-spool-worker
stamp_release gpu-fault-telemetry-spool-worker
if [[ "${RELOAD_RELEASE_METADATA}" == "true" ||
    "${FORCE_ROLE_RESTART}" == "true" ]]; then
    kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" rollout restart \
        deployment/gpu-fault-telemetry-spool-worker
fi

# Admission remains off on the running ingress revision until a live
# consumer tier is proven ready.
wait_for_rollout gpu-fault-telemetry-spool-worker

remove_legacy_notification_env gpu-fault-control-worker
apply_manifest gpu-fault-control-worker
stamp_release gpu-fault-control-worker
if [[ "${RELOAD_RELEASE_METADATA}" == "true" ||
    "${FORCE_ROLE_RESTART}" == "true" ]]; then
    kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" rollout restart \
        deployment/gpu-fault-control-worker
fi

# An upgrade may carry processor pool values added by `kubectl set env`;
# a greenfield namespace has no ingress Deployment yet. Only run the
# cleanup when the old Deployment exists, otherwise the NotFound would
# abort the first deployment before ingress is created.
if kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" get deployment \
    gpu-fault-api-ha >/dev/null 2>&1; then
    kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" set env \
        deployment/gpu-fault-api-ha \
        GPU_FAULT_PROCESSOR_WORKERS- \
        GPU_FAULT_PROCESSOR_FAULT_WORKERS- \
        GPU_FAULT_PROCESSOR_FAULT_PRESSURE_EVIDENCE_WORKERS- \
        GPU_FAULT_PROCESSOR_OBSERVATION_WORKERS- \
        GPU_FAULT_PROCESSOR_GPU_TELEMETRY_WORKERS- \
        GPU_FAULT_PROCESSOR_HOST_TELEMETRY_WORKERS-
fi

wait_for_rollout gpu-fault-control-worker

remove_legacy_notification_env gpu-fault-api-ha
apply_manifest gpu-fault-api-ha-ingress
stamp_release gpu-fault-api-ha
if [[ "${RELOAD_RELEASE_METADATA}" == "true" ||
    "${FORCE_ROLE_RESTART}" == "true" ]]; then
    kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" rollout restart \
        deployment/gpu-fault-api-ha
fi
wait_for_rollout gpu-fault-api-ha

GPU_FAULT_NAMESPACE="${NAMESPACE}" \
GPU_FAULT_RUNTIME_IMAGE="${RUNTIME_IMAGE}" \
GPU_FAULT_RELEASE_ID="${RELEASE_ID}" \
    "${SCRIPT_DIR}/verify-control-plane-role-split.sh"
