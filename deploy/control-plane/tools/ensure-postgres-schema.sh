#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
KUBECONFIG="${GPU_FAULT_CONTROL_PLANE_KUBECONFIG:?required}"
NAMESPACE="${GPU_FAULT_NAMESPACE:-gpu-fault-system}"
WHEEL_CONFIGMAP="${GPU_FAULT_WHEEL_CONFIGMAP:?required}"
DEFAULT_RUNTIME_IMAGE="public.ecr.aws/docker/library/python:3.12-slim"
RUNTIME_IMAGE="${GPU_FAULT_RUNTIME_IMAGE:?required}"
JOB="gpu-fault-postgres-schema-ensure"

[[ "${RUNTIME_IMAGE}" != *[[:space:]#]* ]] || {
    printf 'ERROR: invalid GPU_FAULT_RUNTIME_IMAGE\n' >&2
    exit 2
}

kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" \
  delete "job/${JOB}" --ignore-not-found
sed \
  -e "s/REPLACE_WITH_WHEEL_CONFIGMAP/${WHEEL_CONFIGMAP}/g" \
  -e "s/namespace: gpu-fault-system/namespace: ${NAMESPACE}/g" \
  -e "s#${DEFAULT_RUNTIME_IMAGE}#${RUNTIME_IMAGE}#g" \
  "${ROOT}/deploy/migrations/postgres-schema-ensure-job.yaml" |
  kubectl --kubeconfig "${KUBECONFIG}" apply -f -
kubectl --kubeconfig "${KUBECONFIG}" -n "${NAMESPACE}" \
  wait --for=condition=complete "job/${JOB}" --timeout=12m
