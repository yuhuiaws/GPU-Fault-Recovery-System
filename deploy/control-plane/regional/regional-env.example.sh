#!/usr/bin/env bash
# Persistent, non-secret variables for one regional deployment.
# Install this file as mode 0600 outside the repository, replace every
# REPLACE_* value, and source it at the start of each operator session.

export AWS_REGION='REPLACE_WITH_AWS_REGION'
export NAMESPACE='gpu-fault-system'

export CPU_KUBECONFIG='REPLACE_WITH_ABSOLUTE_CPU_KUBECONFIG'
export CPU_EKS_NAME='REPLACE_WITH_CPU_EKS_NAME'
export CPU_EKS_ARN='REPLACE_WITH_CPU_EKS_ARN'
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
# In-Pod path of the RDS CA bundle. bootstrap (gpu-fault-admin deploy) bakes
# this value into the gpu-fault-aurora Secret's postgres-url as
# sslrootcert=..., so it must equal the path the CA bundle ConfigMap is
# mounted at in every Pod (see apply-rds-ca-bundle.sh and the control-plane,
# migration and credential-refresh manifests). The file need not exist on the
# deploy host: only the path value is baked in. bootstrap refuses to build the
# DSN if this is unset.
export GPU_FAULT_RDS_CA_BUNDLE='/etc/gpu-fault/rds/ca-bundle.pem'

export GPU_FAULT_RUNTIME_IMAGE='public.ecr.aws/docker/library/python:3.12-slim'
export GPU_FAULT_NODE_INSTALLER_IMAGE='public.ecr.aws/amazonlinux/amazonlinux:2023'
export GPU_FAULT_DCGM_EXPORTER_IMAGE='nvcr.io/nvidia/k8s/dcgm-exporter:4.4.1-4.5.2-ubuntu22.04'
export GPU_FAULT_ADOT_IMAGE='REPLACE_WITH_APPROVED_ADOT_IMAGE'

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

regional_validate_region() {
    regional_require AWS_REGION || return
    if [[ ! "${AWS_REGION}" =~ ^[a-z0-9]+(-[a-z0-9]+)+-[0-9]+$ ]]; then
        printf 'ERROR: AWS_REGION is not a valid AWS Region: %s\n' \
            "${AWS_REGION}" >&2
        return 2
    fi
}

regional_assert_eks_arn_region() {
    local name="$1"
    local arn="${!name:-}"
    regional_validate_region || return
    regional_require "${name}" || return
    if [[ "${arn}" != arn:*:eks:"${AWS_REGION}":*:cluster/* ]]; then
        printf 'ERROR: %s does not belong to AWS_REGION %s: %s\n' \
            "${name}" "${AWS_REGION}" "${arn}" >&2
        return 2
    fi
}
