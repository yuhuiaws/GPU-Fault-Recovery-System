#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

AWS_REGION="${AWS_REGION:-us-west-2}"
ENABLE_CLOUDWATCH_HMA_COLLECTOR="$(
    printf '%s' "${GPU_FAULT_ENABLE_CLOUDWATCH_HMA_COLLECTOR:-false}" |
        tr '[:upper:]' '[:lower:]'
)"
[[ "${ENABLE_CLOUDWATCH_HMA_COLLECTOR}" == "true" ||
    "${ENABLE_CLOUDWATCH_HMA_COLLECTOR}" == "false" ]] || {
    printf 'ERROR: GPU_FAULT_ENABLE_CLOUDWATCH_HMA_COLLECTOR must be true or false\n' >&2
    exit 2
}
if [[ "${ENABLE_CLOUDWATCH_HMA_COLLECTOR}" == "false" ]]; then
    printf 'SKIP CloudWatch HMA collector (disabled by deployment policy)\n'
    exit 0
fi

EKS_CLUSTER_NAME="${EKS_CLUSTER_NAME:?EKS_CLUSTER_NAME is required}"
HYPERPOD_CLUSTER_NAME="${HYPERPOD_CLUSTER_NAME:?HYPERPOD_CLUSTER_NAME is required}"
ARTIFACT_BUCKET="${ARTIFACT_BUCKET:?ARTIFACT_BUCKET is required}"
NAMESPACE="${GPU_FAULT_NAMESPACE:-gpu-fault-system}"
STACK_NAME="${STACK_NAME:-gpu-fault-hma-cloudwatch}"
RUNTIME_PROFILE="${GPU_FAULT_RUNTIME_PROFILE:-hyperpod-v1}"

for command in aws kubectl python3.12 zip; do
    command -v "${command}" >/dev/null || {
        printf 'ERROR: missing command: %s\n' "${command}" >&2
        exit 1
    }
done

account_id="$(aws sts get-caller-identity --query Account --output text)"
oidc_url="$(
    aws eks describe-cluster --region "${AWS_REGION}" \
        --name "${EKS_CLUSTER_NAME}" \
        --query 'cluster.identity.oidc.issuer' --output text
)"
oidc_issuer="${oidc_url#https://}"
oidc_provider_arn="arn:aws:iam::${account_id}:oidc-provider/${oidc_issuer}"

log_group="$(
    aws logs describe-log-groups --region "${AWS_REGION}" \
        --log-group-name-prefix \
        "/aws/sagemaker/Clusters/${HYPERPOD_CLUSTER_NAME}/" \
        --query 'logGroups[0].logGroupName' --output text
)"
[[ "${log_group}" != "None" && -n "${log_group}" ]] || {
    printf 'ERROR: HyperPod HMA log group was not found\n' >&2
    exit 1
}

build_dir="$(mktemp -d)"
artifact_file="$(mktemp --suffix=.zip)"
rendered_template="$(mktemp --suffix=.yaml)"
trap 'rm -rf "${build_dir}"; rm -f "${artifact_file}" "${rendered_template}"' EXIT
python3.12 -m pip install --quiet --target "${build_dir}" \
    "${REPO_DIR}[collectors]"
rm -f "${artifact_file}"
(
    cd "${build_dir}"
    zip -qr "${artifact_file}" .
)
artifact_key="gpu-fault/cloudwatch-hma-$(
    sha256sum "${artifact_file}" | cut -d' ' -f1
).zip"
aws s3 cp "${artifact_file}" \
    "s3://${ARTIFACT_BUCKET}/${artifact_key}" \
    --region "${AWS_REGION}" --only-show-errors

sed "s|REPLACE_WITH_OIDC_ISSUER|${oidc_issuer}|g" \
    "${REPO_DIR}/deploy/aws/lambda/cloudwatch-hma-template.yaml" \
    > "${rendered_template}"
aws cloudformation deploy --region "${AWS_REGION}" \
    --stack-name "${STACK_NAME}" \
    --template-file "${rendered_template}" \
    --capabilities CAPABILITY_NAMED_IAM \
    --parameter-overrides \
        "ArtifactBucket=${ARTIFACT_BUCKET}" \
        "ArtifactKey=${artifact_key}" \
        "HmaLogGroup=${log_group}" \
        "ClusterId=${HYPERPOD_CLUSTER_NAME}" \
        "RuntimeProfileVersion=${RUNTIME_PROFILE}" \
        'NodeRegex=^SagemakerHealthMonitoringAgent/[^/]+/(?P<node_id>i-[a-f0-9]+)$' \
        "NodePrefix=hyperpod-" \
        "OidcProviderArn=${oidc_provider_arn}"

queue_url="$(
    # shellcheck disable=SC2016 # JMESPath backtick literal, not a shell expansion.
    aws cloudformation describe-stacks --region "${AWS_REGION}" \
        --stack-name "${STACK_NAME}" \
        --query 'Stacks[0].Outputs[?OutputKey==`QueueUrl`].OutputValue' \
        --output text
)"
consumer_role="$(
    # shellcheck disable=SC2016 # JMESPath backtick literal, not a shell expansion.
    aws cloudformation describe-stacks --region "${AWS_REGION}" \
        --stack-name "${STACK_NAME}" \
        --query \
        'Stacks[0].Outputs[?OutputKey==`ConsumerRoleArn`].OutputValue' \
        --output text
)"
sed "s|REPLACE_WITH_QUEUE_URL|${queue_url}|g" \
    "${REPO_DIR}/deploy/dataplane/optional/hma-cloudwatch-consumer.yaml" |
    kubectl apply -f -
kubectl -n "${NAMESPACE}" annotate serviceaccount \
    gpu-fault-hma-cloudwatch-consumer \
    "eks.amazonaws.com/role-arn=${consumer_role}" --overwrite
kubectl -n "${NAMESPACE}" rollout restart \
    deployment/gpu-fault-hma-cloudwatch-consumer
kubectl -n "${NAMESPACE}" rollout status \
    deployment/gpu-fault-hma-cloudwatch-consumer --timeout=10m

PYTHONDONTWRITEBYTECODE=1 python3 \
    "${REPO_DIR}/deploy/control-plane/tools/sync_installed_resource_registry.py" \
    --plane gpu \
    --namespace "${NAMESPACE}" \
    --release-id "${artifact_key##*/}"

aws logs describe-subscription-filters --region "${AWS_REGION}" \
    --log-group-name "${log_group}" \
    --query 'subscriptionFilters[].{name:filterName,destination:destinationArn}' \
    --output table
