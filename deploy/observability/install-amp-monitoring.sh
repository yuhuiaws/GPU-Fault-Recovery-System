#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
)"
AWS_REGION="${AWS_REGION:?AWS_REGION is required}"
CPU_EKS_CLUSTER="${CPU_EKS_CLUSTER:?CPU_EKS_CLUSTER is required}"
CPU_KUBECONFIG="${CPU_KUBECONFIG:-/tmp/gpu-fault-control-plane.kubeconfig}"
AMP_WORKSPACE_ID="${AMP_WORKSPACE_ID:?AMP_WORKSPACE_ID is required}"
SNS_TOPIC_NAME="${SNS_TOPIC_NAME:-gpu-fault-control-plane-alerts-${AWS_REGION}}"
IAM_ROLE_NAME="${IAM_ROLE_NAME:-gpu-fault-control-plane-amp-writer-${AWS_REGION}}"
NAMESPACE="${NAMESPACE:-gpu-fault-system}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-gpu-fault-adot}"
RULE_NAMESPACE="${RULE_NAMESPACE:-gpu-fault-control-plane-capacity}"
GPU_FAULT_ENABLE_ADOT="${GPU_FAULT_ENABLE_ADOT:-true}"
GPU_FAULT_ENABLE_AMP="${GPU_FAULT_ENABLE_AMP:-true}"
GPU_FAULT_REQUIRE_CONFIRMED_SNS_SUBSCRIPTION="${GPU_FAULT_REQUIRE_CONFIRMED_SNS_SUBSCRIPTION:-true}"
ADOT_IMAGE="${GPU_FAULT_ADOT_IMAGE:?GPU_FAULT_ADOT_IMAGE is required}"

if [[ "${GPU_FAULT_ENABLE_AMP}" != "true" ]]; then
    printf 'AMP integration disabled; built-in collector silence alerts remain active.\n'
    exit 0
fi
[[ "${AWS_REGION}" =~ ^[a-z0-9]+(-[a-z0-9]+)+-[0-9]+$ ]] || {
    printf 'ERROR: invalid AWS_REGION: %s\n' "${AWS_REGION}" >&2
    exit 2
}
[[ "${GPU_FAULT_REQUIRE_CONFIRMED_SNS_SUBSCRIPTION}" == "true" ||
    "${GPU_FAULT_REQUIRE_CONFIRMED_SNS_SUBSCRIPTION}" == "false" ]] || {
    printf 'ERROR: GPU_FAULT_REQUIRE_CONFIRMED_SNS_SUBSCRIPTION must be true or false\n' >&2
    exit 2
}
[[ -n "${ADOT_IMAGE}" &&
    "${ADOT_IMAGE}" != REPLACE_* &&
    "${ADOT_IMAGE}" != *[[:space:]#]* ]] || {
    printf 'ERROR: invalid GPU_FAULT_ADOT_IMAGE\n' >&2
    exit 2
}

for command in aws jq kubectl sed; do
    command -v "${command}" >/dev/null || {
        printf 'ERROR: %s is required\n' "${command}" >&2
        exit 1
    }
done

CPU_EKS_ARN="$(
    aws eks describe-cluster \
        --region "${AWS_REGION}" \
        --name "${CPU_EKS_CLUSTER}" \
        --query 'cluster.arn' \
        --output text
)"
KUBECONFIG_EKS_ARN="$(
    kubectl --kubeconfig "${CPU_KUBECONFIG}" config view --minify \
        -o jsonpath='{.contexts[0].context.cluster}'
)"
if [[ "${KUBECONFIG_EKS_ARN}" != "${CPU_EKS_ARN}" ]]; then
    printf 'ERROR: CPU kubeconfig EKS ARN does not match %s in %s\n' \
        "${CPU_EKS_CLUSTER}" "${AWS_REGION}" >&2
    exit 2
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
WORKSPACE_ARN="arn:aws:aps:${AWS_REGION}:${ACCOUNT_ID}:workspace/${AMP_WORKSPACE_ID}"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${IAM_ROLE_NAME}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

cat >"${TMP_DIR}/pod-identity-trust.json" <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "pods.eks.amazonaws.com"},
    "Action": ["sts:AssumeRole", "sts:TagSession"]
  }]
}
EOF

cat >"${TMP_DIR}/amp-write-policy.json" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["aps:RemoteWrite"],
    "Resource": "${WORKSPACE_ARN}"
  }]
}
EOF

if ! aws iam get-role --role-name "${IAM_ROLE_NAME}" >/dev/null 2>&1; then
    aws iam create-role \
        --role-name "${IAM_ROLE_NAME}" \
        --assume-role-policy-document \
        "file://${TMP_DIR}/pod-identity-trust.json" >/dev/null
fi
aws iam put-role-policy \
    --role-name "${IAM_ROLE_NAME}" \
    --policy-name gpu-fault-amp-remote-write \
    --policy-document "file://${TMP_DIR}/amp-write-policy.json"

SNS_TOPIC_ARN="$(
    aws sns create-topic \
        --region "${AWS_REGION}" \
        --name "${SNS_TOPIC_NAME}" \
        --query TopicArn --output text
)"
CURRENT_SNS_POLICY="$(
    aws sns get-topic-attributes \
        --region "${AWS_REGION}" \
        --topic-arn "${SNS_TOPIC_ARN}" \
        --query 'Attributes.Policy' \
        --output text
)"
jq \
    --arg account_id "${ACCOUNT_ID}" \
    --arg topic_arn "${SNS_TOPIC_ARN}" \
    --arg workspace_arn "${WORKSPACE_ARN}" \
    '
      .Statement = (
        [.Statement[] | select(.Sid != "AllowAmpAlertmanagerPublish")] +
        [{
          Sid: "AllowAmpAlertmanagerPublish",
          Effect: "Allow",
          Principal: {Service: "aps.amazonaws.com"},
          Action: "sns:Publish",
          Resource: $topic_arn,
          Condition: {
            StringEquals: {"AWS:SourceAccount": $account_id},
            ArnEquals: {"AWS:SourceArn": $workspace_arn}
          }
        }]
      )
    ' <<<"${CURRENT_SNS_POLICY}" >"${TMP_DIR}/sns-policy.json"
SNS_POLICY="$(jq -c . "${TMP_DIR}/sns-policy.json")"
aws sns set-topic-attributes \
    --region "${AWS_REGION}" \
    --topic-arn "${SNS_TOPIC_ARN}" \
    --attribute-name Policy \
    --attribute-value "${SNS_POLICY}"

if [[ -n "${GPU_FAULT_ALERT_EMAIL:-}" ]]; then
    aws sns subscribe \
        --region "${AWS_REGION}" \
        --topic-arn "${SNS_TOPIC_ARN}" \
        --protocol email \
        --notification-endpoint "${GPU_FAULT_ALERT_EMAIL}" >/dev/null
fi

sed \
    -e "s/REPLACE_WITH_AMP_WORKSPACE_ID/${AMP_WORKSPACE_ID}/g" \
    -e "s/REPLACE_WITH_AWS_REGION/${AWS_REGION}/g" \
    -e "s#REPLACE_WITH_ADOT_IMAGE#${ADOT_IMAGE}#g" \
    "${REPO_DIR}/deploy/observability/adot-control-plane.yaml" \
    >"${TMP_DIR}/adot-control-plane.yaml"
CURRENT_ADOT_REPLICAS="$(
    kubectl --kubeconfig "${CPU_KUBECONFIG}" \
        -n "${NAMESPACE}" get deployment gpu-fault-adot \
        -o jsonpath='{.spec.replicas}' 2>/dev/null || true
)"
if [[ "${GPU_FAULT_ENABLE_ADOT}" != "true" \
    && -n "${CURRENT_ADOT_REPLICAS}" ]]; then
    sed -i \
        "s/^  replicas: 1$/  replicas: ${CURRENT_ADOT_REPLICAS}/" \
        "${TMP_DIR}/adot-control-plane.yaml"
fi
kubectl --kubeconfig "${CPU_KUBECONFIG}" \
    apply -f "${TMP_DIR}/adot-control-plane.yaml"
if [[ "${GPU_FAULT_ENABLE_ADOT}" != "true" \
    && -n "${CURRENT_ADOT_REPLICAS}" \
    && "${CURRENT_ADOT_REPLICAS}" != "0" ]]; then
    kubectl --kubeconfig "${CPU_KUBECONFIG}" \
        -n "${NAMESPACE}" scale deployment/gpu-fault-adot \
        --replicas="${CURRENT_ADOT_REPLICAS}"
fi

ASSOCIATION_ID="$(
    aws eks list-pod-identity-associations \
        --region "${AWS_REGION}" \
        --cluster-name "${CPU_EKS_CLUSTER}" \
        --namespace "${NAMESPACE}" \
        --service-account "${SERVICE_ACCOUNT}" \
        --query 'associations[0].associationId' \
        --output text
)"
if [[ "${ASSOCIATION_ID}" == "None" ]]; then
    aws eks create-pod-identity-association \
        --region "${AWS_REGION}" \
        --cluster-name "${CPU_EKS_CLUSTER}" \
        --namespace "${NAMESPACE}" \
        --service-account "${SERVICE_ACCOUNT}" \
        --role-arn "${ROLE_ARN}" >/dev/null
else
    aws eks update-pod-identity-association \
        --region "${AWS_REGION}" \
        --cluster-name "${CPU_EKS_CLUSTER}" \
        --association-id "${ASSOCIATION_ID}" \
        --role-arn "${ROLE_ARN}" >/dev/null
fi

if aws amp describe-rule-groups-namespace \
    --region "${AWS_REGION}" \
    --workspace-id "${AMP_WORKSPACE_ID}" \
    --name "${RULE_NAMESPACE}" >/dev/null 2>&1; then
    aws amp put-rule-groups-namespace \
        --region "${AWS_REGION}" \
        --workspace-id "${AMP_WORKSPACE_ID}" \
        --name "${RULE_NAMESPACE}" \
        --data "fileb://${REPO_DIR}/deploy/observability/amp-rules.yaml" \
        >/dev/null
else
    aws amp create-rule-groups-namespace \
        --region "${AWS_REGION}" \
        --workspace-id "${AMP_WORKSPACE_ID}" \
        --name "${RULE_NAMESPACE}" \
        --data "fileb://${REPO_DIR}/deploy/observability/amp-rules.yaml" \
        >/dev/null
fi

sed \
    -e "s#REPLACE_WITH_SNS_TOPIC_ARN#${SNS_TOPIC_ARN}#g" \
    -e "s/REPLACE_WITH_AWS_REGION/${AWS_REGION}/g" \
    "${REPO_DIR}/deploy/observability/amp-alertmanager.yaml" \
    >"${TMP_DIR}/amp-alertmanager.yaml"
if aws amp describe-alert-manager-definition \
    --region "${AWS_REGION}" \
    --workspace-id "${AMP_WORKSPACE_ID}" >/dev/null 2>&1; then
    aws amp put-alert-manager-definition \
        --region "${AWS_REGION}" \
        --workspace-id "${AMP_WORKSPACE_ID}" \
        --data "fileb://${TMP_DIR}/amp-alertmanager.yaml" \
        >/dev/null
else
    aws amp create-alert-manager-definition \
        --region "${AWS_REGION}" \
        --workspace-id "${AMP_WORKSPACE_ID}" \
        --data "fileb://${TMP_DIR}/amp-alertmanager.yaml" \
        >/dev/null
fi

CONFIRMED_SUBSCRIPTIONS="$(
    aws sns list-subscriptions-by-topic \
        --region "${AWS_REGION}" \
        --topic-arn "${SNS_TOPIC_ARN}" \
        --query 'length(Subscriptions[?SubscriptionArn != `PendingConfirmation` && SubscriptionArn != `Deleted`])' \
        --output text
)"
if [[ "${GPU_FAULT_REQUIRE_CONFIRMED_SNS_SUBSCRIPTION}" == "true" &&
    "${CONFIRMED_SUBSCRIPTIONS}" == "0" ]]; then
    printf 'ERROR: SNS topic %s has no confirmed subscription; confirm the email subscription or attach an operations endpoint before enabling production alerts\n' \
        "${SNS_TOPIC_ARN}" >&2
    exit 1
fi

if [[ "${GPU_FAULT_ENABLE_ADOT}" == "true" ]]; then
    kubectl --kubeconfig "${CPU_KUBECONFIG}" \
        -n "${NAMESPACE}" scale deployment/gpu-fault-adot --replicas=1
    kubectl --kubeconfig "${CPU_KUBECONFIG}" \
        -n "${NAMESPACE}" rollout restart deployment/gpu-fault-adot
    kubectl --kubeconfig "${CPU_KUBECONFIG}" \
        -n "${NAMESPACE}" rollout status deployment/gpu-fault-adot \
        --timeout=300s
else
    # Do not scale down an already-enabled collector: an operator who turned
    # it on should not have it silently disabled by re-running the installer.
    printf 'ADOT collector left as-is (GPU_FAULT_ENABLE_ADOT=false).\n'
    printf 'Path-B metric alerts are inert while it has 0 replicas.\n'
    printf 'Enable with: kubectl -n %s scale deployment/gpu-fault-adot --replicas=1\n' \
        "${NAMESPACE}"
fi

printf 'AMP workspace: %s\n' "${WORKSPACE_ARN}"
printf 'SNS topic: %s\n' "${SNS_TOPIC_ARN}"
printf 'ADOT role: %s\n' "${ROLE_ARN}"

PYTHONDONTWRITEBYTECODE=1 python3 \
    "${REPO_DIR}/deploy/control-plane/tools/sync_installed_resource_registry.py" \
    --plane cpu \
    --kubeconfig "${CPU_KUBECONFIG}" \
    --namespace "${NAMESPACE}" \
    --release-id "${GPU_FAULT_RELEASE_ID:-observability}"
