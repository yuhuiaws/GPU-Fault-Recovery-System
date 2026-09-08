#!/usr/bin/env bash
# Ship the AWS RDS global CA bundle into the cluster as a ConfigMap.
#
# The control-plane Postgres DSN uses sslmode=verify-full (see
# gpu_fault.admin.bootstrap._aurora_dsn): the connection is refused unless a
# trusted RDS CA bundle is mounted in the Pod and GPU_FAULT_RDS_CA_BUNDLE
# points at it. Without this ConfigMap the control-plane Deployment, the
# Aurora migration Job and the credential-refresh CronJob all mount a missing
# volume and stay in CreateContainerConfigError -- fail closed, never an
# unverified `require` connection.
#
# The bundle is fetched at deploy time from the official AWS truststore and
# checked against a pinned SHA-256 before it is admitted. A bundle whose
# digest does not match the pin aborts the deploy, exactly as the wheel and
# node-bundle digest gates elsewhere in this tree do. To roll the pin when AWS
# rotates the global bundle, download it, confirm its provenance, and replace
# RDS_CA_BUNDLE_SHA256 below with the new `sha256sum` value.
#
# Point GPU_FAULT_RDS_CA_BUNDLE_SOURCE at a local PEM to install from a
# vendored copy instead of the network; the same digest gate applies.
set -euo pipefail

NAMESPACE="${GPU_FAULT_NAMESPACE:-gpu-fault-system}"
CONFIGMAP_NAME="${GPU_FAULT_RDS_CA_BUNDLE_CONFIGMAP:-gpu-fault-rds-ca-bundle}"
# The single key inside the ConfigMap. It becomes the file name under the
# mount directory, so the in-Pod path is <mount dir>/ca-bundle.pem, which is
# the value every consumer sets GPU_FAULT_RDS_CA_BUNDLE to.
BUNDLE_KEY="ca-bundle.pem"
RDS_CA_BUNDLE_URL="${GPU_FAULT_RDS_CA_BUNDLE_URL:-https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem}"
# Pinned SHA-256 of the AWS RDS global-bundle.pem. Provenance:
#   curl -fsS https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem | sha256sum
# Fetched 2026-09-07; 108 certificates.
RDS_CA_BUNDLE_SHA256="e5bb2084ccf45087bda1c9bffdea0eb15ee67f0b91646106e466714f9de3c7e3"

for command in kubectl sha256sum; do
    command -v "${command}" >/dev/null 2>&1 || {
        echo "required command not found: ${command}" >&2
        exit 2
    }
done

[[ "${RDS_CA_BUNDLE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "RDS_CA_BUNDLE_SHA256 must be a lowercase SHA-256" >&2
    exit 2
}

kubectl_args=()
if [[ -n "${KUBECONFIG:-}" ]]; then
    kubectl_args+=(--kubeconfig "${KUBECONFIG}")
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
BUNDLE_FILE="${WORK_DIR}/${BUNDLE_KEY}"

SOURCE="${GPU_FAULT_RDS_CA_BUNDLE_SOURCE:-}"
if [[ -n "${SOURCE}" ]]; then
    [[ -f "${SOURCE}" ]] || {
        echo "GPU_FAULT_RDS_CA_BUNDLE_SOURCE is not a file: ${SOURCE}" >&2
        exit 2
    }
    cp "${SOURCE}" "${BUNDLE_FILE}"
else
    command -v curl >/dev/null 2>&1 || {
        echo "required command not found: curl" >&2
        exit 2
    }
    curl --fail --silent --show-error --location --max-time 60 \
        "${RDS_CA_BUNDLE_URL}" -o "${BUNDLE_FILE}"
fi

# Fail closed: a bundle whose digest does not match the pin never reaches the
# cluster. `sha256sum -c` returns non-zero on mismatch, and set -e aborts.
printf '%s  %s\n' "${RDS_CA_BUNDLE_SHA256}" "${BUNDLE_FILE}" |
    sha256sum -c - >/dev/null || {
    echo "RDS CA bundle digest does not match the pinned SHA-256; refusing" >&2
    echo "  expected: ${RDS_CA_BUNDLE_SHA256}" >&2
    echo "  actual:   $(sha256sum "${BUNDLE_FILE}" | awk '{print $1}')" >&2
    exit 1
}

kubectl "${kubectl_args[@]}" -n "${NAMESPACE}" create configmap \
    "${CONFIGMAP_NAME}" \
    --from-file="${BUNDLE_KEY}=${BUNDLE_FILE}" \
    --dry-run=client -o yaml |
    kubectl "${kubectl_args[@]}" apply -f -

echo "applied ConfigMap ${CONFIGMAP_NAME} in ${NAMESPACE} (${RDS_CA_BUNDLE_SHA256})"
