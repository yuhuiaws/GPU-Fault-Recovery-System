#!/usr/bin/env bash
# Persistent, non-secret variables for one regional deployment.
# Install this file as mode 0600 outside the repository, replace every
# REPLACE_* value, and source it at the start of each operator session.

export AWS_REGION='us-west-2'
export NAMESPACE='gpu-fault-system'

export CPU_KUBECONFIG='REPLACE_WITH_ABSOLUTE_CPU_KUBECONFIG'
export CPU_EKS_NAME='REPLACE_WITH_CPU_EKS_NAME'
export CPU_HYPERPOD_CLUSTER='REPLACE_WITH_CPU_HYPERPOD_CLUSTER_NAME'

export GPU_EKS_CONTEXT='REPLACE_WITH_GPU_EKS_CONTEXT'
export GPU_EKS_NAME='REPLACE_WITH_GPU_EKS_NAME'
export GPU_EKS_ARN='REPLACE_WITH_GPU_EKS_ARN'
export CLUSTER_ID='REPLACE_WITH_STABLE_CLUSTER_ID'
export HYPERPOD_CLUSTER_NAME='REPLACE_WITH_HYPERPOD_CLUSTER_NAME'
export HYPERPOD_CONFIRM_CLUSTER_NAME='REPLACE_WITH_HYPERPOD_CLUSTER_NAME'
export ALLOWED_NAMESPACES='gpu-fault-system,training'

export CONTROL_PLANE_URL='REPLACE_AFTER_NLB_IS_READY'
export SECURE_DIR="${HOME}/.gpu-fault-secrets/${CLUSTER_ID}"

export GPU_FAULT_REPO_ROOT='REPLACE_WITH_ABSOLUTE_REPO_ROOT'
export RELEASE_MANIFEST="${GPU_FAULT_REPO_ROOT}/dist/current-release.json"
WHEEL="$(
  python3 "${GPU_FAULT_REPO_ROOT}/scripts/release-artifact-path.py" \
    wheel --manifest "${RELEASE_MANIFEST}"
)"
export WHEEL
NODE_BUNDLE="$(
  python3 "${GPU_FAULT_REPO_ROOT}/scripts/release-artifact-path.py" \
    bundle --manifest "${RELEASE_MANIFEST}"
)"
export NODE_BUNDLE
export EXECUTOR_IRSA_ROLE_ARN='REPLACE_WITH_EXECUTOR_IRSA_ROLE_ARN'
export GPU_FAULT_RUNTIME_IMAGE='public.ecr.aws/docker/library/python:3.12-slim'
export GPU_FAULT_NODE_INSTALLER_IMAGE='public.ecr.aws/amazonlinux/amazonlinux:2023'
export GPU_FAULT_DCGM_EXPORTER_IMAGE='nvcr.io/nvidia/k8s/dcgm-exporter:4.4.1-4.5.2-ubuntu22.04'
export GPU_FAULT_ADOT_IMAGE='602401143452.dkr.ecr.us-west-2.amazonaws.com/hyperpod/otel_collector:v1783977775530'

regional_require() {
    local name
    local value
    for name in "$@"; do
        value="${!name:-}"
        if [[ -z "${value}" || "${value}" == REPLACE_* ]]; then
            printf 'ERROR: %s is unset or still a placeholder\n' \
                "${name}" >&2
            return 2
        fi
    done
}
