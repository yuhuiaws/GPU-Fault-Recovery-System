#!/usr/bin/env bash
# Render the checked-in regional role-split manifests.
#
# The repository intentionally contains no deployable kustomization.yaml.
# This script copies the explicitly named inputs into a temporary directory,
# installs the input template there as kustomization.yaml, and renders that
# isolated directory. The intermediate single-tier Deployment is never a
# supported apply target.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONTROL_PLANE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BASE_DIR="${CONTROL_PLANE_DIR}/base"
REGIONAL_DIR="${CONTROL_PLANE_DIR}/regional"
OUT_DIR="${GPU_FAULT_ROLE_SPLIT_OUT_DIR:-${REGIONAL_DIR}/generated}"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

mkdir -p "${OUT_DIR}"

install -m 0644 \
    "${BASE_DIR}/control-plane-deployment.yaml" \
    "${WORK_DIR}/control-plane-deployment.yaml"
install -m 0644 \
    "${REGIONAL_DIR}/regional-control-plane-prerequisites.yaml" \
    "${WORK_DIR}/regional-control-plane-prerequisites.yaml"
install -m 0644 \
    "${REGIONAL_DIR}/regional-control-plane-patch.yaml" \
    "${WORK_DIR}/regional-control-plane-patch.yaml"
install -m 0644 \
    "${BASE_DIR}/role-split-input.kustomization.yaml" \
    "${WORK_DIR}/kustomization.yaml"

kubectl kustomize "${WORK_DIR}" |
    python3 "${SCRIPT_DIR}/render_control_plane_role_split.py" \
        --out-dir "${OUT_DIR}"

echo "rendered:"
ls -1 "${OUT_DIR}"
