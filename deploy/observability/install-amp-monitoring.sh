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
# The collector restart below is skipped when nothing it reads changed. This is
# the escape hatch for what an apply cannot describe: a Pod running the right
# manifest with stale credentials or a wedged exporter.
GPU_FAULT_FORCE_ADOT_RESTART="${GPU_FAULT_FORCE_ADOT_RESTART:-false}"

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

AMP_APPLY_TIMEOUT_SECONDS="${GPU_FAULT_AMP_APPLY_TIMEOUT_SECONDS:-300}"

# AMP validates a rule namespace and an Alertmanager definition
# asynchronously: the write returns immediately, the definition goes to
# CREATING/UPDATING, and until it settles a describe still answers with the
# previous one. Returning here before that happens makes the caller's own
# verification read the definition this installer just replaced -- which reads
# as "the live namespace is missing a group" on any release that adds one, and
# passes on the retry, i.e. exactly the shape of failure nobody trusts. It also
# hides the case that matters: a definition AMP rejected leaves the old rules
# serving, so the new alerts never fire and nothing else would say so.
wait_for_amp_definition() {
    local label="$1" query="$2"
    shift 2
    local deadline=$((SECONDS + AMP_APPLY_TIMEOUT_SECONDS))
    local status
    while :; do
        status="$("$@" --query "${query}" --output text)"
        case "${status}" in
        ACTIVE)
            return 0
            ;;
        CREATING | UPDATING) ;;
        *)
            printf 'ERROR: %s was not applied: AMP reports status %s\n' \
                "${label}" "${status}" >&2
            exit 1
            ;;
        esac
        if ((SECONDS >= deadline)); then
            printf 'ERROR: %s is still %s after %ss\n' \
                "${label}" "${status}" "${AMP_APPLY_TIMEOUT_SECONDS}" >&2
            exit 1
        fi
        sleep 5
    done
}

# AMP hands a definition back exactly as stored, so comparing the decoded bytes
# with the file about to be written says whether a put would change anything.
# An identical, ACTIVE definition is left alone: the put itself is cheap, but
# AMP then revalidates asynchronously and the release would wait out
# UPDATING -> ACTIVE for a definition that did not change.
amp_definition_is_current() {
    local file="$1" root="$2"
    shift 2
    local document status
    document="$("$@" --output json 2>/dev/null)" || return 1
    status="$(
        jq -r --arg root "${root}" '.[$root].status.statusCode // ""' \
            <<<"${document}"
    )"
    [[ "${status}" == "ACTIVE" ]] || return 1
    cmp -s "${file}" <(
        jq -r --arg root "${root}" '.[$root].data // ""' <<<"${document}" |
            base64 -d
    )
}

# Per-step wall clock, so a slow run of this installer can be attributed without
# re-running it under a profiler. On stdout, because the admin CLI runs this
# script with the output captured and only keeps stdout: on stderr these lines
# were discarded on every successful run, i.e. exactly the runs worth measuring.
STEP_STARTED_AT="${SECONDS}"
step_done() {
    printf 'amp-step-elapsed %ss %s\n' "$((SECONDS - STEP_STARTED_AT))" "$1"
    STEP_STARTED_AT="${SECONDS}"
}

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

# `put-role-policy` is an audited IAM mutation, and this installer issued one on
# every deploy for a document that changes only when the workspace does. Reading
# the inline policy first also answers whether the role exists at all: on a
# missing role the read fails the same way, so the create path below is entered
# without a second `get-role` in the steady state.
CURRENT_AMP_POLICY="$(
    aws iam get-role-policy \
        --role-name "${IAM_ROLE_NAME}" \
        --policy-name gpu-fault-amp-remote-write \
        --query PolicyDocument \
        --output json 2>/dev/null || true
)"
AMP_POLICY_CURRENT="false"
DESIRED_AMP_POLICY="$(jq -S -c . "${TMP_DIR}/amp-write-policy.json")"
if [[ -n "${CURRENT_AMP_POLICY}" ]]; then
    LIVE_AMP_POLICY="$(jq -S -c . <<<"${CURRENT_AMP_POLICY}" 2>/dev/null || true)"
    if [[ "${LIVE_AMP_POLICY}" == "${DESIRED_AMP_POLICY}" ]]; then
        AMP_POLICY_CURRENT="true"
        printf 'IAM role %s already grants remote write to this workspace.\n' \
            "${IAM_ROLE_NAME}"
    fi
elif ! aws iam get-role --role-name "${IAM_ROLE_NAME}" >/dev/null 2>&1; then
    aws iam create-role \
        --role-name "${IAM_ROLE_NAME}" \
        --assume-role-policy-document \
        "file://${TMP_DIR}/pod-identity-trust.json" >/dev/null
fi
if [[ "${AMP_POLICY_CURRENT}" != "true" ]]; then
    aws iam put-role-policy \
        --role-name "${IAM_ROLE_NAME}" \
        --policy-name gpu-fault-amp-remote-write \
        --policy-document "file://${TMP_DIR}/amp-write-policy.json"
fi
step_done iam-role

# `create-topic` is idempotent but not free, and the topic ARN is derivable, so
# the read comes first: an existing topic answers with its policy and the create
# is skipped. A topic that does not answer is created and then read.
SNS_TOPIC_ARN="arn:aws:sns:${AWS_REGION}:${ACCOUNT_ID}:${SNS_TOPIC_NAME}"
read_sns_policy() {
    aws sns get-topic-attributes \
        --region "${AWS_REGION}" \
        --topic-arn "${SNS_TOPIC_ARN}" \
        --query 'Attributes.Policy' \
        --output text
}
CURRENT_SNS_POLICY="$(read_sns_policy 2>/dev/null || true)"
if [[ -z "${CURRENT_SNS_POLICY}" || "${CURRENT_SNS_POLICY}" == "None" ]]; then
    SNS_TOPIC_ARN="$(
        aws sns create-topic \
            --region "${AWS_REGION}" \
            --name "${SNS_TOPIC_NAME}" \
            --query TopicArn --output text
    )"
    CURRENT_SNS_POLICY="$(read_sns_policy)"
fi
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
# The transform above only adds the Alertmanager publish statement, so a topic
# that already carries it is left alone: rewriting the policy on every deploy
# both hides real changes in CloudTrail and briefly republishes a policy that
# other statements (a queue subscription, an operator grant) live in.
# Statement order carries no meaning in an SNS policy, and SNS is free to hand
# the list back in a different order than it was written, so the comparison sorts
# by Sid first. Comparing the raw order would rewrite the policy on every deploy
# for a topic that already says exactly what it should.
canonical_sns_policy() {
    jq -S -c '.Statement |= sort_by(.Sid // "")'
}
DESIRED_SNS_POLICY="$(canonical_sns_policy <<<"${SNS_POLICY}")"
LIVE_SNS_POLICY="$(canonical_sns_policy <<<"${CURRENT_SNS_POLICY}" 2>/dev/null || true)"
if [[ "${DESIRED_SNS_POLICY}" == "${LIVE_SNS_POLICY}" ]]; then
    printf 'SNS topic %s already allows Alertmanager to publish.\n' \
        "${SNS_TOPIC_ARN}"
else
    aws sns set-topic-attributes \
        --region "${AWS_REGION}" \
        --topic-arn "${SNS_TOPIC_ARN}" \
        --attribute-name Policy \
        --attribute-value "${SNS_POLICY}"
fi
step_done sns-topic

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
ADOT_APPLY_OUTPUT="$(
    kubectl --kubeconfig "${CPU_KUBECONFIG}" \
        apply -f "${TMP_DIR}/adot-control-plane.yaml"
)"
printf '%s\n' "${ADOT_APPLY_OUTPUT}"
# The collector reads its ConfigMap once at start, so a changed ConfigMap or
# Deployment needs a restart. An unchanged pair does not, and restarting it
# anyway costs the old Pod's termination grace on every release. Pod identity
# changes below can still flip this back on.
ADOT_RESTART_REQUIRED="true"
if [[ "$(
    grep -Ec '^(configmap/gpu-fault-adot|deployment\.apps/gpu-fault-adot) unchanged$' \
        <<<"${ADOT_APPLY_OUTPUT}" || true
)" == "2" ]]; then
    ADOT_RESTART_REQUIRED="false"
fi
# The gate above skips the restart only for a pair that apply itself called
# `unchanged`; a `configured`, a `created`, or output it cannot parse all keep
# it. The override exists for what apply cannot describe: a Pod on the right
# manifest with stale credentials, or an exporter that stopped writing.
if [[ "${GPU_FAULT_FORCE_ADOT_RESTART}" == "true" ]]; then
    ADOT_RESTART_REQUIRED="true"
fi
if [[ "${GPU_FAULT_ENABLE_ADOT}" != "true" \
    && -n "${CURRENT_ADOT_REPLICAS}" \
    && "${CURRENT_ADOT_REPLICAS}" != "0" ]]; then
    kubectl --kubeconfig "${CPU_KUBECONFIG}" \
        -n "${NAMESPACE}" scale deployment/gpu-fault-adot \
        --replicas="${CURRENT_ADOT_REPLICAS}"
fi
step_done adot-apply

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
    # Fresh credentials reach the Pod only through a restart.
    ADOT_RESTART_REQUIRED="true"
else
    CURRENT_ASSOCIATION_ROLE_ARN="$(
        aws eks describe-pod-identity-association \
            --region "${AWS_REGION}" \
            --cluster-name "${CPU_EKS_CLUSTER}" \
            --association-id "${ASSOCIATION_ID}" \
            --query 'association.roleArn' \
            --output text
    )"
    if [[ "${CURRENT_ASSOCIATION_ROLE_ARN}" != "${ROLE_ARN}" ]]; then
        aws eks update-pod-identity-association \
            --region "${AWS_REGION}" \
            --cluster-name "${CPU_EKS_CLUSTER}" \
            --association-id "${ASSOCIATION_ID}" \
            --role-arn "${ROLE_ARN}" >/dev/null
        ADOT_RESTART_REQUIRED="true"
    fi
fi
step_done pod-identity

RULES_DEFINITION_CURRENT="false"
if amp_definition_is_current \
    "${REPO_DIR}/deploy/observability/amp-rules.yaml" \
    ruleGroupsNamespace \
    aws amp describe-rule-groups-namespace \
    --region "${AWS_REGION}" \
    --workspace-id "${AMP_WORKSPACE_ID}" \
    --name "${RULE_NAMESPACE}"; then
    RULES_DEFINITION_CURRENT="true"
    printf 'AMP rule namespace %s already matches the checked-in rules.\n' \
        "${RULE_NAMESPACE}"
elif aws amp describe-rule-groups-namespace \
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
if [[ "${RULES_DEFINITION_CURRENT}" != "true" ]]; then
    wait_for_amp_definition \
        "AMP rule namespace ${RULE_NAMESPACE}" \
        'ruleGroupsNamespace.status.statusCode' \
        aws amp describe-rule-groups-namespace \
        --region "${AWS_REGION}" \
        --workspace-id "${AMP_WORKSPACE_ID}" \
        --name "${RULE_NAMESPACE}"
fi
step_done amp-rules

sed \
    -e "s#REPLACE_WITH_SNS_TOPIC_ARN#${SNS_TOPIC_ARN}#g" \
    -e "s/REPLACE_WITH_AWS_REGION/${AWS_REGION}/g" \
    "${REPO_DIR}/deploy/observability/amp-alertmanager.yaml" \
    >"${TMP_DIR}/amp-alertmanager.yaml"
ALERTMANAGER_DEFINITION_CURRENT="false"
if amp_definition_is_current \
    "${TMP_DIR}/amp-alertmanager.yaml" \
    alertManagerDefinition \
    aws amp describe-alert-manager-definition \
    --region "${AWS_REGION}" \
    --workspace-id "${AMP_WORKSPACE_ID}"; then
    ALERTMANAGER_DEFINITION_CURRENT="true"
    printf 'AMP Alertmanager definition already matches the rendered one.\n'
elif aws amp describe-alert-manager-definition \
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
if [[ "${ALERTMANAGER_DEFINITION_CURRENT}" != "true" ]]; then
    wait_for_amp_definition \
        "AMP Alertmanager definition" \
        'alertManagerDefinition.status.statusCode' \
        aws amp describe-alert-manager-definition \
        --region "${AWS_REGION}" \
        --workspace-id "${AMP_WORKSPACE_ID}"
fi
step_done amp-alertmanager

if [[ "${GPU_FAULT_REQUIRE_CONFIRMED_SNS_SUBSCRIPTION}" == "true" ]]; then
    if [[ -n "${GPU_FAULT_ALERT_EMAIL:-}" ]]; then
        read -r CONFIRMED_SUBSCRIPTIONS PENDING_SUBSCRIPTIONS < <(
            aws sns list-subscriptions-by-topic \
                --region "${AWS_REGION}" \
                --topic-arn "${SNS_TOPIC_ARN}" \
                --output json |
                jq -r --arg endpoint "${GPU_FAULT_ALERT_EMAIL}" '
                  [
                    .Subscriptions[]?
                    | select(
                        ((.Protocol // "") | ascii_downcase) == "email"
                        and ((.Endpoint // "") | ascii_downcase)
                          == ($endpoint | ascii_downcase)
                    )
                  ] as $matches
                  | [
                      ($matches
                        | map(select(
                            ((.SubscriptionArn // "") | startswith("arn:"))
                          ))
                        | length),
                      ($matches
                        | map(select(
                            .SubscriptionArn == "PendingConfirmation"
                          ))
                        | length)
                    ]
                  | @tsv
                '
        )
        if [[ "${CONFIRMED_SUBSCRIPTIONS}" != "1" ||
            "${PENDING_SUBSCRIPTIONS}" != "0" ]]; then
            printf 'ERROR: SNS topic %s must have exactly one confirmed email subscription for the configured administrator endpoint and no pending duplicates (confirmed=%s pending=%s); confirm the existing request or remove duplicate subscriptions before retrying\n' \
                "${SNS_TOPIC_ARN}" \
                "${CONFIRMED_SUBSCRIPTIONS}" \
                "${PENDING_SUBSCRIPTIONS}" >&2
            exit 1
        fi
    else
        CONFIRMED_SUBSCRIPTIONS="$(
            aws sns list-subscriptions-by-topic \
                --region "${AWS_REGION}" \
                --topic-arn "${SNS_TOPIC_ARN}" \
                --query 'length(Subscriptions[?SubscriptionArn != `PendingConfirmation` && SubscriptionArn != `Deleted`])' \
                --output text
        )"
        if [[ "${CONFIRMED_SUBSCRIPTIONS}" == "0" ]]; then
            printf 'ERROR: SNS topic %s has no confirmed subscription; attach an operations endpoint before enabling production alerts\n' \
                "${SNS_TOPIC_ARN}" >&2
            exit 1
        fi
    fi
fi

if [[ "${GPU_FAULT_ENABLE_ADOT}" == "true" ]]; then
    step_done sns-subscription-check
    kubectl --kubeconfig "${CPU_KUBECONFIG}" \
        -n "${NAMESPACE}" scale deployment/gpu-fault-adot --replicas=1
    # Restart when something the Pod reads changed, or when a Deployment that
    # was already meant to be running is not fully available (a restart is the
    # repair for that). Scaling up from zero creates a fresh Pod by itself.
    if [[ "${ADOT_RESTART_REQUIRED}" == "true" ]]; then
        kubectl --kubeconfig "${CPU_KUBECONFIG}" \
            -n "${NAMESPACE}" rollout restart deployment/gpu-fault-adot
    elif [[ "${CURRENT_ADOT_REPLICAS:-0}" != "0" ]] &&
        ! kubectl --kubeconfig "${CPU_KUBECONFIG}" \
            -n "${NAMESPACE}" rollout status deployment/gpu-fault-adot \
            --timeout=5s >/dev/null 2>&1; then
        kubectl --kubeconfig "${CPU_KUBECONFIG}" \
            -n "${NAMESPACE}" rollout restart deployment/gpu-fault-adot
    else
        printf 'ADOT collector unchanged and available; restart skipped.\n'
    fi
    kubectl --kubeconfig "${CPU_KUBECONFIG}" \
        -n "${NAMESPACE}" rollout status deployment/gpu-fault-adot \
        --timeout=300s
    step_done adot-rollout
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
step_done installed-resource-registry
