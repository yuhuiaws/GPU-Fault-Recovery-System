#!/usr/bin/env bash
# Ship container logs off the cluster with the amazon-cloudwatch-observability
# EKS addon.
#
# Why a standalone script rather than a bootstrap task: the addon is an AWS-side
# object plus a namespace this repository does not render. Nothing here enters
# `module_digest` or the release transaction, so it can be installed and removed
# in its own window without touching a deploy. That is the whole reason this
# path was chosen first over an ADOT logs pipeline, which would put
# `adot-control-plane.yaml` into the rendered manifest digest.
#
# What it actually centralizes -- read this before believing a log is safe:
#   * `/aws/containerinsights/<cluster>/application` -- every container's
#     stdout/stderr in the cluster, which on the control plane is this system's
#     own logs. Central redaction (gpu_fault.logging_setup) is the hard
#     precondition; see docs/详细设计-v2.md §10.1.
#   * `/aws/containerinsights/<cluster>/dataplane` -- the docker/containerd/kubelet
#     journald units, plus the aws-node and kube-proxy containers.
#   * `/aws/containerinsights/<cluster>/host`      -- journald, in three inputs:
#     `_TRANSPORT=kernel` (what dmesg shows), `PRIORITY=0-6`, and
#     `SYSLOG_FACILITY=10`. The PRIORITY input is **unit-agnostic**: every systemd
#     unit on the node at info or above lands here, so wherever `gpu-fault-*`
#     units run their journal ships too. The only exclusion is by syslog facility
#     (mail, cron, authpriv), never by unit.
#
# What it does NOT centralize: the training log files (Fluent Bit has no tail input
# for arbitrary paths, only `/var/log/containers/*.log`), and anything at
# PRIORITY=7. Fault evidence is also unaffected either way -- the control plane
# never reads CloudWatch, so node-side collection remains the only channel that
# produces `NodeLogBatch.collection_errors`.
#
# Both classes of input start at the tail (`READ_FROM_HEAD=Off`), so nothing
# written before the install is ever shipped.
#
# Actions: `status` (default, read-only), `install`, `uninstall`.
set -euo pipefail

AWS_REGION="${AWS_REGION:?AWS_REGION is required}"
EKS_CLUSTER="${EKS_CLUSTER:?EKS_CLUSTER is required}"
EKS_KUBECONFIG="${EKS_KUBECONFIG:-/tmp/gpu-fault-control-plane.kubeconfig}"
# Read-only by default. An operator who meant to install says so; an operator who
# typed the wrong cluster name finds out from a report instead of from a bill.
ACTION="${GPU_FAULT_CLOUDWATCH_ACTION:-status}"
SITE_TAG_KEY="gpu-fault:site-id"
ADDON_NAME="amazon-cloudwatch-observability"
CLOUDWATCH_NAMESPACE="amazon-cloudwatch"
CLOUDWATCH_SERVICE_ACCOUNT="cloudwatch-agent"
AGENT_POLICY_ARN="arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
# Never-expire is the CloudWatch default for a log group Fluent Bit creates, and
# an unbounded application log group on a cluster that logs per request is the
# cost incident this switch exists to prevent. Retention is set on the groups
# *before* the addon starts, so there is no window in which lines land in a group
# that keeps them forever.
LOG_RETENTION_DAYS="${GPU_FAULT_CLOUDWATCH_RETENTION_DAYS:-30}"
LOG_ARRIVAL_TIMEOUT_SECONDS="${GPU_FAULT_CLOUDWATCH_LOG_TIMEOUT_SECONDS:-300}"
ROLLOUT_TIMEOUT_SECONDS="${GPU_FAULT_CLOUDWATCH_ROLLOUT_TIMEOUT_SECONDS:-300}"
# Both default to false on purpose, and both are about the GPU cluster:
#   * enhanced insights is the per-pod metric tier, priced per metric, and this
#     site already has its own metrics in AMP;
#   * accelerated compute metrics make the agent deploy its own dcgm-exporter,
#     which binds host port 9400 -- the port
#     `deploy/systemd/gpu-fault-dcgm-exporter.service` already holds on every GPU
#     node, so both would fight over it and one loses.
ENHANCED_INSIGHTS="${GPU_FAULT_CLOUDWATCH_ENHANCED_INSIGHTS:-false}"
ACCELERATED_METRICS="${GPU_FAULT_CLOUDWATCH_ACCELERATED_METRICS:-false}"
ALLOW_GPU_CLUSTER="${GPU_FAULT_CLOUDWATCH_ALLOW_GPU_CLUSTER:-false}"

case "${ACTION}" in
status | install | uninstall) ;;
*)
    printf 'ERROR: GPU_FAULT_CLOUDWATCH_ACTION must be status, install or uninstall\n' >&2
    exit 2
    ;;
esac
[[ "${AWS_REGION}" =~ ^[a-z0-9]+(-[a-z0-9]+)+-[0-9]+$ ]] || {
    printf 'ERROR: invalid AWS_REGION: %s\n' "${AWS_REGION}" >&2
    exit 2
}
for flag in ENHANCED_INSIGHTS ACCELERATED_METRICS ALLOW_GPU_CLUSTER; do
    value="${!flag}"
    [[ "${value}" == "true" || "${value}" == "false" ]] || {
        printf 'ERROR: %s must be true or false\n' "${flag}" >&2
        exit 2
    }
done
# CloudWatch rejects any other value, and it rejects it *after* the log group
# exists -- which would leave a never-expire group behind on a typo.
RETENTION_VALUES=" 1 3 5 7 14 30 60 90 120 150 180 365 400 545 731 1096 1827 2192 2557 2922 3288 3653 "
[[ "${RETENTION_VALUES}" == *" ${LOG_RETENTION_DAYS} "* ]] || {
    printf 'ERROR: GPU_FAULT_CLOUDWATCH_RETENTION_DAYS=%s is not a CloudWatch retention value\n' \
        "${LOG_RETENTION_DAYS}" >&2
    exit 2
}
safe_name() {
    # Mirrors gpu_fault.admin.bootstrap_common.safe_name. A real site id is
    # around 47 characters (region, role and a digest suffix), so
    # `gpu-fault-<site>-cloudwatch-agent` is past IAM's 64-character limit and has
    # to be shortened the same deterministic way every other name at this site is:
    # truncate and append a digest of the full string. Truncating without the
    # digest would collide between two sites whose ids share a prefix, and
    # anything non-deterministic would create a second role on every rerun and
    # leave the first one attached to nothing.
    local value="$1" maximum="$2" normalized digest
    normalized="$(
        printf '%s' "${value}" | tr '[:upper:]' '[:lower:]' |
            sed -e 's/[^a-z0-9-]\{1,\}/-/g' -e 's/^-*//' -e 's/-*$//'
    )"
    [[ -n "${normalized}" ]] || {
        printf 'ERROR: cannot derive a resource name from %s\n' "${value}" >&2
        exit 2
    }
    if ((${#normalized} <= maximum)); then
        printf '%s\n' "${normalized}"
        return
    fi
    digest="$(printf '%s' "${normalized}" | sha256sum | cut -c1-8)"
    printf '%s-%s\n' \
        "$(printf '%s' "${normalized:0:maximum-9}" | sed 's/-*$//')" "${digest}"
}

for command in aws jq kubectl sha256sum; do
    command -v "${command}" >/dev/null || {
        printf 'ERROR: %s is required\n' "${command}" >&2
        exit 1
    }
done

if [[ "${ACTION}" != "status" ]]; then
    SITE_ID="${SITE_ID:?SITE_ID is required for install and uninstall}"
    [[ "${SITE_ID}" =~ ^[a-z0-9][a-z0-9-]{0,61}[a-z0-9]$ ]] || {
        printf 'ERROR: invalid SITE_ID: %s\n' "${SITE_ID}" >&2
        exit 2
    }
    IAM_ROLE_NAME="${GPU_FAULT_CLOUDWATCH_ROLE_NAME:-$(
        safe_name "gpu-fault-${SITE_ID}-cloudwatch-agent" 64
    )}"
    [[ "${#IAM_ROLE_NAME}" -le 64 ]] || {
        printf 'ERROR: IAM role name exceeds 64 characters: %s\n' \
            "${IAM_ROLE_NAME}" >&2
        exit 2
    }
fi

# The guard that matters most: everything below names the cluster by string, and
# a kubeconfig pointing somewhere else would have the addon installed in one
# cluster while the readiness and log checks pass against another.
EKS_ARN="$(
    aws eks describe-cluster \
        --region "${AWS_REGION}" \
        --name "${EKS_CLUSTER}" \
        --query 'cluster.arn' \
        --output text
)"
KUBECONFIG_EKS_ARN="$(
    kubectl --kubeconfig "${EKS_KUBECONFIG}" config view --minify \
        -o jsonpath='{.contexts[0].context.cluster}'
)"
if [[ "${KUBECONFIG_EKS_ARN}" != "${EKS_ARN}" ]]; then
    printf 'ERROR: kubeconfig %s does not point at %s in %s\n' \
        "${EKS_KUBECONFIG}" "${EKS_CLUSTER}" "${AWS_REGION}" >&2
    exit 2
fi

LOG_GROUP_PREFIX="/aws/containerinsights/${EKS_CLUSTER}"
# `performance` is where the agent writes its embedded-metric records; it is
# created by the agent, not by Fluent Bit, and it is the one most often left
# unbounded because nobody thinks of it as a log group.
LOG_GROUPS=(
    "${LOG_GROUP_PREFIX}/application"
    "${LOG_GROUP_PREFIX}/dataplane"
    "${LOG_GROUP_PREFIX}/host"
    "${LOG_GROUP_PREFIX}/performance"
)

gpu_node_count() {
    kubectl --kubeconfig "${EKS_KUBECONFIG}" get nodes -o json |
        jq '[.items[] | select(.status.capacity["nvidia.com/gpu"] != null)] | length'
}

addon_status() {
    aws eks describe-addon \
        --region "${AWS_REGION}" \
        --query 'addon.status' \
        --output text \
        --cluster-name "${EKS_CLUSTER}" \
        --addon-name "${ADDON_NAME}" 2>/dev/null || printf 'ABSENT\n'
}

association_id() {
    aws eks list-pod-identity-associations \
        --region "${AWS_REGION}" \
        --cluster-name "${EKS_CLUSTER}" \
        --namespace "${CLOUDWATCH_NAMESPACE}" \
        --service-account "${CLOUDWATCH_SERVICE_ACCOUNT}" \
        --query 'associations[0].associationId' \
        --output text 2>/dev/null || printf 'None\n'
}

report_status() {
    printf 'addon %s: %s\n' "${ADDON_NAME}" "$(addon_status)"
    printf 'pod identity association (%s/%s): %s\n' \
        "${CLOUDWATCH_NAMESPACE}" "${CLOUDWATCH_SERVICE_ACCOUNT}" "$(association_id)"
    local group retention newest
    for group in "${LOG_GROUPS[@]}"; do
        retention="$(
            aws logs describe-log-groups \
                --region "${AWS_REGION}" \
                --log-group-name-prefix "${group}" \
                --query "logGroups[?logGroupName=='${group}'].retentionInDays | [0]" \
                --output text 2>/dev/null || printf 'None\n'
        )"
        newest="$(
            aws logs describe-log-streams \
                --region "${AWS_REGION}" \
                --log-group-name "${group}" \
                --order-by LastEventTime \
                --descending \
                --limit 1 \
                --query 'logStreams[0].lastEventTimestamp' \
                --output text 2>/dev/null || printf 'None\n'
        )"
        # `None` retention on an existing group is never-expire, which is the
        # state this script exists to keep the site out of.
        printf 'log group %s: retention=%s last_event_epoch_ms=%s\n' \
            "${group}" "${retention}" "${newest}"
    done
    if kubectl --kubeconfig "${EKS_KUBECONFIG}" get namespace \
        "${CLOUDWATCH_NAMESPACE}" >/dev/null 2>&1; then
        kubectl --kubeconfig "${EKS_KUBECONFIG}" -n "${CLOUDWATCH_NAMESPACE}" \
            get daemonset -o custom-columns=NAME:.metadata.name,DESIRED:.status.desiredNumberScheduled,READY:.status.numberReady
    else
        printf 'namespace %s does not exist\n' "${CLOUDWATCH_NAMESPACE}"
    fi
}

wait_for_daemonsets() {
    # Not `rollout status daemonset/<name>`: the set of DaemonSets the addon
    # creates depends on its version and on the configuration above (the
    # accelerated-compute one appears only when that tier is on), and naming them
    # here would turn a version bump into a failed install. Whatever is in the
    # namespace has to become ready instead.
    local deadline=$((SECONDS + ROLLOUT_TIMEOUT_SECONDS))
    local pending
    while :; do
        pending="$(
            kubectl --kubeconfig "${EKS_KUBECONFIG}" -n "${CLOUDWATCH_NAMESPACE}" \
                get daemonset -o json |
                jq -r '
                  [.items[]
                   | select((.status.desiredNumberScheduled // 0) != (.status.numberReady // 0))
                   | "\(.metadata.name) ready=\(.status.numberReady // 0)/\(.status.desiredNumberScheduled // 0)"]
                  | join(", ")
                '
        )"
        [[ -z "${pending}" ]] && return 0
        if ((SECONDS >= deadline)); then
            printf 'ERROR: DaemonSets not ready after %ss: %s\n' \
                "${ROLLOUT_TIMEOUT_SECONDS}" "${pending}" >&2
            exit 1
        fi
        sleep 10
    done
}

wait_for_log_arrival() {
    # Pod Ready is not the success criterion: the agent comes up Ready and then
    # fails every PutLogEvents when the pod identity association is missing or the
    # role lost its policy, which looks healthy from Kubernetes and ships nothing.
    # A stream in the application group is proof the whole chain works.
    #
    # `--limit`, not `--max-items`: the latter is client-side pagination, so the
    # CLI appends its own token line to the output and a scalar --query comes back
    # as two lines. That turns every comparison below into a comparison against
    # "1\nNone".
    local group="$1"
    local deadline=$((SECONDS + LOG_ARRIVAL_TIMEOUT_SECONDS))
    local streams
    while :; do
        streams="$(
            aws logs describe-log-streams \
                --region "${AWS_REGION}" \
                --log-group-name "${group}" \
                --limit 1 \
                --query 'length(logStreams)' \
                --output text 2>/dev/null || printf '0\n'
        )"
        [[ "${streams}" != "0" && "${streams}" != "None" ]] && return 0
        if ((SECONDS >= deadline)); then
            printf 'ERROR: no log stream in %s after %ss; the addon is installed but nothing is shipping. Check the pod identity association and the agent pod logs in %s\n' \
                "${group}" "${LOG_ARRIVAL_TIMEOUT_SECONDS}" "${CLOUDWATCH_NAMESPACE}" >&2
            exit 1
        fi
        sleep 10
    done
}

if [[ "${ACTION}" == "status" ]]; then
    report_status
    exit 0
fi

if [[ "${ACTION}" == "uninstall" ]]; then
    if [[ "$(addon_status)" != "ABSENT" ]]; then
        # No `--preserve`: leaving the DaemonSets behind unmanaged is the state
        # that later reads as "the addon is gone but pods are still shipping".
        aws eks delete-addon \
            --region "${AWS_REGION}" \
            --cluster-name "${EKS_CLUSTER}" \
            --addon-name "${ADDON_NAME}" >/dev/null
        aws eks wait addon-deleted \
            --region "${AWS_REGION}" \
            --cluster-name "${EKS_CLUSTER}" \
            --addon-name "${ADDON_NAME}"
    fi
    ASSOCIATION_ID="$(association_id)"
    if [[ "${ASSOCIATION_ID}" != "None" && -n "${ASSOCIATION_ID}" ]]; then
        aws eks delete-pod-identity-association \
            --region "${AWS_REGION}" \
            --cluster-name "${EKS_CLUSTER}" \
            --association-id "${ASSOCIATION_ID}" >/dev/null
    fi
    if aws iam get-role --role-name "${IAM_ROLE_NAME}" >/dev/null 2>&1; then
        ROLE_SITE="$(
            aws iam list-role-tags \
                --role-name "${IAM_ROLE_NAME}" \
                --query "Tags[?Key=='${SITE_TAG_KEY}'].Value | [0]" \
                --output text
        )"
        if [[ "${ROLE_SITE}" == "${SITE_ID}" ]]; then
            aws iam detach-role-policy \
                --role-name "${IAM_ROLE_NAME}" \
                --policy-arn "${AGENT_POLICY_ARN}" 2>/dev/null || true
            aws iam delete-role --role-name "${IAM_ROLE_NAME}"
        else
            # An untagged or foreign role was not created here. Deleting it would
            # break whatever does own it, and this script has no way to know what.
            printf 'IAM role %s is not tagged %s=%s; left in place\n' \
                "${IAM_ROLE_NAME}" "${SITE_TAG_KEY}" "${SITE_ID}"
        fi
    fi
    # The log groups are deliberately kept: they are the only remaining copy of
    # what the cluster said before it stopped shipping, and retention already
    # bounds them. Removing them is a separate, explicit decision.
    printf 'Log groups kept under %s (retention %s days).\n' \
        "${LOG_GROUP_PREFIX}" "${LOG_RETENTION_DAYS}"
    printf 'To discard that evidence as well, per group:\n'
    printf '  aws logs delete-log-group --region %s --log-group-name %s/application\n' \
        "${AWS_REGION}" "${LOG_GROUP_PREFIX}"
    exit 0
fi

# --- install ---------------------------------------------------------------

GPU_NODES="$(gpu_node_count)"
if [[ "${GPU_NODES}" != "0" && "${ALLOW_GPU_CLUSTER}" != "true" ]]; then
    printf 'ERROR: %s has %s GPU node(s). Three consequences there: the application log group would carry every training job stdout, the host group would duplicate the kernel XID lines and the gpu-fault-* unit journals that node-side collection already carries, and the agent'"'"'s own dcgm-exporter contends for host port 9400 with gpu-fault-dcgm-exporter. Set GPU_FAULT_CLOUDWATCH_ALLOW_GPU_CLUSTER=true, with GPU_FAULT_CLOUDWATCH_ACCELERATED_METRICS=false, only after deciding all three are acceptable\n' \
        "${EKS_CLUSTER}" "${GPU_NODES}" >&2
    exit 2
fi
if [[ "${GPU_NODES}" != "0" && "${ACCELERATED_METRICS}" == "true" ]]; then
    printf 'ERROR: GPU_FAULT_CLOUDWATCH_ACCELERATED_METRICS=true on a cluster with GPU nodes would deploy a second dcgm-exporter on host port 9400\n' >&2
    exit 2
fi

POD_IDENTITY_STATUS="$(
    aws eks describe-addon \
        --region "${AWS_REGION}" \
        --cluster-name "${EKS_CLUSTER}" \
        --addon-name eks-pod-identity-agent \
        --query 'addon.status' \
        --output text 2>/dev/null || printf 'ABSENT\n'
)"
if [[ "${POD_IDENTITY_STATUS}" != "ACTIVE" ]]; then
    printf 'ERROR: eks-pod-identity-agent is %s on %s; the agent has no way to assume a role. Bootstrap the site first\n' \
        "${POD_IDENTITY_STATUS}" "${EKS_CLUSTER}" >&2
    exit 1
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
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

if ! aws iam get-role --role-name "${IAM_ROLE_NAME}" >/dev/null 2>&1; then
    aws iam create-role \
        --role-name "${IAM_ROLE_NAME}" \
        --assume-role-policy-document "file://${TMP_DIR}/pod-identity-trust.json" \
        --tags "Key=${SITE_TAG_KEY},Value=${SITE_ID}" >/dev/null
fi
aws iam attach-role-policy \
    --role-name "${IAM_ROLE_NAME}" \
    --policy-arn "${AGENT_POLICY_ARN}"

# Before the addon, so the association exists the moment the agent pod starts.
# The service account does not exist yet; an association is by name, so this is
# allowed and avoids a first-start window with no credentials.
ASSOCIATION_ID="$(association_id)"
if [[ "${ASSOCIATION_ID}" == "None" || -z "${ASSOCIATION_ID}" ]]; then
    aws eks create-pod-identity-association \
        --region "${AWS_REGION}" \
        --cluster-name "${EKS_CLUSTER}" \
        --namespace "${CLOUDWATCH_NAMESPACE}" \
        --service-account "${CLOUDWATCH_SERVICE_ACCOUNT}" \
        --role-arn "${ROLE_ARN}" >/dev/null
else
    aws eks update-pod-identity-association \
        --region "${AWS_REGION}" \
        --cluster-name "${EKS_CLUSTER}" \
        --association-id "${ASSOCIATION_ID}" \
        --role-arn "${ROLE_ARN}" >/dev/null
fi

# Retention before the first byte: a group Fluent Bit creates itself defaults to
# never-expire, and setting retention afterwards does not shorten what already
# landed under the old policy.
for group in "${LOG_GROUPS[@]}"; do
    aws logs create-log-group \
        --region "${AWS_REGION}" \
        --log-group-name "${group}" \
        --tags "${SITE_TAG_KEY}=${SITE_ID}" 2>/dev/null || true
    aws logs put-retention-policy \
        --region "${AWS_REGION}" \
        --log-group-name "${group}" \
        --retention-in-days "${LOG_RETENTION_DAYS}"
done

jq -n \
    --argjson enhanced "${ENHANCED_INSIGHTS}" \
    --argjson accelerated "${ACCELERATED_METRICS}" \
    '{
      containerLogs: {enabled: true},
      agent: {
        config: {
          logs: {
            metrics_collected: {
              kubernetes: {
                enhanced_container_insights: $enhanced,
                accelerated_compute_metrics: $accelerated
              }
            }
          }
        }
      }
    }' >"${TMP_DIR}/addon-configuration.json"
ADDON_CONFIGURATION="$(jq -c . "${TMP_DIR}/addon-configuration.json")"

if [[ "$(addon_status)" == "ABSENT" ]]; then
    aws eks create-addon \
        --region "${AWS_REGION}" \
        --cluster-name "${EKS_CLUSTER}" \
        --addon-name "${ADDON_NAME}" \
        --configuration-values "${ADDON_CONFIGURATION}" \
        --tags "${SITE_TAG_KEY}=${SITE_ID}" >/dev/null
else
    # OVERWRITE resolves the conflict against whatever a previous run or an
    # operator left in the addon's own resources; without it an update stops
    # half-applied and reports CREATE_FAILED with no further explanation.
    aws eks update-addon \
        --region "${AWS_REGION}" \
        --cluster-name "${EKS_CLUSTER}" \
        --addon-name "${ADDON_NAME}" \
        --configuration-values "${ADDON_CONFIGURATION}" \
        --resolve-conflicts OVERWRITE >/dev/null
fi
aws eks wait addon-active \
    --region "${AWS_REGION}" \
    --cluster-name "${EKS_CLUSTER}" \
    --addon-name "${ADDON_NAME}"

wait_for_daemonsets
wait_for_log_arrival "${LOG_GROUP_PREFIX}/application"

printf 'Addon: %s (%s)\n' "${ADDON_NAME}" "${EKS_ARN}"
printf 'Agent role: %s\n' "${ROLE_ARN}"
printf 'Log groups: %s/{application,dataplane,host,performance} retention=%s days\n' \
    "${LOG_GROUP_PREFIX}" "${LOG_RETENTION_DAYS}"
printf 'Shipped from now on only: every input starts at the tail, so nothing written before this install is sent.\n'
printf 'Not shipped by this path: training log files; those stay with the node log collector, which is also the only channel that produces NodeLogBatch.collection_errors.\n'
printf 'Shipped whether or not you expected it: the host group takes every systemd unit at PRIORITY 0-6, gpu-fault-* units included.\n'
