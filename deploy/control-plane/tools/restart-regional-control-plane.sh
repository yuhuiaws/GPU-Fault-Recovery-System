#!/usr/bin/env bash
# Restart every running regional control-plane role after a referenced
# Secret or ConfigMap changes. Consumers restart before ingress so the
# new registry/config is loaded before new requests are accepted.
set -euo pipefail

: "${KUBECONFIG:?set KUBECONFIG to the CPU control-plane kubeconfig}"

NAMESPACE="${GPU_FAULT_NAMESPACE:-gpu-fault-system}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
found=0

for deployment in \
    gpu-fault-telemetry-spool-worker \
    gpu-fault-control-worker \
    gpu-fault-api-ha; do
    if ! kubectl -n "${NAMESPACE}" get deployment "${deployment}" \
        >/dev/null 2>&1; then
        continue
    fi
    found=1
    replicas="$(
        kubectl -n "${NAMESPACE}" get deployment "${deployment}" \
            -o jsonpath='{.spec.replicas}'
    )"
    replicas="${replicas:-0}"
    if (( replicas == 0 )); then
        printf 'Skipping %s: replicas=0\n' "${deployment}"
        continue
    fi
    kubectl -n "${NAMESPACE}" rollout restart \
        "deployment/${deployment}"
    kubectl -n "${NAMESPACE}" rollout status \
        "deployment/${deployment}" --timeout=10m
done

if (( found == 0 )); then
    printf 'No regional control-plane Deployments exist yet; nothing to restart.\n'
    exit 0
fi

GPU_FAULT_NAMESPACE="${NAMESPACE}" \
    "${SCRIPT_DIR}/verify-control-plane-role-split.sh"
