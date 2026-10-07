#!/usr/bin/env bash
# GF-REGIONAL-BOOT-029 driver: the administrator deployment lifecycle as one
# ordered acceptance sequence on an isolated staging site.
#
#   1  cold first deploy (BuildKit caches, local runtime images and the site's
#      ECR image/buildcache tags are gone before the build)
#   2  remove-cluster of every GPU cluster, one ARN at a time in the given
#      order, ending in the zero-GPU-cluster steady state
#   3  one join-cluster re-attaching every GPU cluster (a plain-EKS member's
#      --gpu-nodegroup / --node-selector passed through as given)
#   4  uninstall --cpu-cluster keep --reset-database
#      (--aurora-final-snapshot retain|skip, default retain)
#   5  cold first deploy again, into a second empty state directory, strictly
#      by README.md's three documented steps: `make deploy-host-setup-online`,
#      `. .venv/bin/activate` and the four-argument `gpu-fault-admin deploy`.
#      The lines are read from the README (its `make ... check` verification
#      line is skipped: the deploy runs the release gates itself), rendered
#      with this run's ARNs, second state directory and email, and executed
#      from a pristine copy of --repo in a clean environment
#      (scripts/e2e/regional/boot029_readme.py). --email-wait-minutes applies
#      to stage 1 only: the README command has no wait, so an unconfirmed
#      SNS subscription ends the first stage-5 attempt with the deploy's own
#      refusal, and --stage 5 resumes after the mail was confirmed.
#
# Every admin command runs in the foreground and the next stage starts only
# after it has exited. The first failing stage stops the run and prints its
# number so `--stage N` can resume. Wall time and the sha256 of each stage's log
# go to <state-dir>/acceptance/admin-lifecycle-sequence.json; ARNs, the email
# and the paths appear there only as sha256 digests.
set -euo pipefail
umask 077

usage() {
  cat >&2 <<'USAGE'
usage: admin-lifecycle-sequence.sh --state-dir DIR --cpu-cluster-arn ARN
         --gpu-cluster-arn ARN [--gpu-cluster-arn ARN ...]
         [--gpu-nodegroup NODEGROUP | --node-selector SELECTOR  (plain-EKS members)]
         --admin-email EMAIL --repo CHECKOUT
         [--stage N] [--second-state-dir DIR] [--email-wait-minutes M  (stage 1 only)]
         [--previous-site-id ID] [--aurora-final-snapshot retain|skip]
         --confirm-isolated-build-host
USAGE
  exit 2
}

STATE_DIR="" CPU_ARN="" ADMIN_EMAIL="" REPO="" START_STAGE=1
GPU_ARNS=() EKS_FLAG_NAME="" EKS_FLAG_VALUE=""
SECOND_STATE_DIR="" EMAIL_WAIT=0 PREVIOUS_SITE_ID="" ISOLATED_BUILD_HOST=false
AURORA_FINAL_SNAPSHOT=retain
while [[ $# -gt 0 ]]; do
  case "$1" in
    --state-dir) STATE_DIR="$2"; shift 2 ;;
    --cpu-cluster-arn) CPU_ARN="$2"; shift 2 ;;
    --gpu-cluster-arn) [[ $# -ge 2 ]] || usage; GPU_ARNS+=("$2"); shift 2 ;;
    --gpu-nodegroup|--node-selector)
      [[ $# -ge 2 && -z "$EKS_FLAG_NAME" ]] || usage
      EKS_FLAG_NAME="$1"; EKS_FLAG_VALUE="$2"; shift 2 ;;
    --admin-email) ADMIN_EMAIL="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --stage) START_STAGE="$2"; shift 2 ;;
    --second-state-dir) SECOND_STATE_DIR="$2"; shift 2 ;;
    --email-wait-minutes) EMAIL_WAIT="$2"; shift 2 ;;
    --previous-site-id) PREVIOUS_SITE_ID="$2"; shift 2 ;;
    --aurora-final-snapshot) [[ $# -ge 2 ]] || usage; AURORA_FINAL_SNAPSHOT="$2"; shift 2 ;;
    --confirm-isolated-build-host) ISOLATED_BUILD_HOST=true; shift ;;
    *) usage ;;
  esac
done
[[ -n "$STATE_DIR" && -n "$CPU_ARN" && ${#GPU_ARNS[@]} -ge 1 && -n "$ADMIN_EMAIL" && -n "$REPO" ]] || usage
[[ -z "$EKS_FLAG_NAME" || -n "$EKS_FLAG_VALUE" ]] || usage
# The repeated flag for deploy/join-cluster, the per-ARN receipt arguments and
# the plain-EKS selector pass-through, in the operator's order.
GPU_ARN_ARGS=() GPU_RECEIPT_ARGS=() EKS_ARGS=() EKS_FLAG=""
for arn in "${GPU_ARNS[@]}"; do
  GPU_ARN_ARGS+=(--gpu-cluster-arn "$arn")
  GPU_RECEIPT_ARGS+=(--gpu-arn "$arn")
done
if [[ -n "$EKS_FLAG_NAME" ]]; then
  EKS_ARGS=("$EKS_FLAG_NAME" "$EKS_FLAG_VALUE")
  EKS_FLAG="${EKS_FLAG_NAME}=${EKS_FLAG_VALUE}"
fi
[[ "$START_STAGE" =~ ^[1-5]$ ]] || usage
[[ "$EMAIL_WAIT" =~ ^[0-9]+$ ]] || usage
[[ "$AURORA_FINAL_SNAPSHOT" =~ ^(retain|skip)$ ]] || usage
[[ "$ISOLATED_BUILD_HOST" == true ]] || usage
[[ -d "$REPO/deploy/image" ]] || { echo "--repo is not a checkout: $REPO" >&2; exit 2; }

REPO="$(realpath "$REPO")"
STATE_DIR="$(realpath -m "$STATE_DIR")"
SECOND_STATE_DIR="$(realpath -m "${SECOND_STATE_DIR:-${STATE_DIR}-second}")"
[[ "$STATE_DIR" != "$SECOND_STATE_DIR" ]] || usage
ACCEPT_DIR="${STATE_DIR}/acceptance"
RECORD="${ACCEPT_DIR}/admin-lifecycle-sequence.json"
CASE_ID="GF-REGIONAL-BOOT-029"
STAGE_NAMES=("" cold-first-deploy remove-cluster join-cluster uninstall-reset cold-redeploy-readme)
STAGE_LOG=""
CURRENT_STAGE=1
CURRENT_ATTEMPT=1
# The deploy reads its source tree from the working directory.
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO"
export BUILDKIT_PROGRESS=plain

log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
fail() { printf 'FAIL: %s\n' "$*" | tee -a "$STAGE_LOG" >&2; exit 1; }
digest_of() { printf '%s' "$1" | sha256sum | cut -c1-64; }

json_field() {  # FILE KEY -> value or empty
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))' "$1" "$2"
}

require_fresh_state_dir() {
  # Freshness is checked once by start_stage; a retry keeps its original state.
  if [[ -f "$1/bootstrap-state.json" ]]; then
    log "$(basename "$1") holds the original first-deploy checkpoint: resuming it"
    printf '{"resumed_first_deploy": true}\n' >>"${STAGE_LOG}.extra.json"
  elif [[ -e "$1/site.yaml" || -L "$1/site.yaml" ]]; then
    fail "$(basename "$1") already holds site.yaml: a first deploy needs an empty state directory"
  fi
  install -d -m 0700 "$1"
}

init_record() {
  install -d -m 0700 "$ACCEPT_DIR"
  python3 -m scripts.e2e.regional.boot029_receipts init --record "$RECORD" \
    --state "$STATE_DIR" --second-state "$SECOND_STATE_DIR" --stage "$START_STAGE" \
    --cpu-arn "$CPU_ARN" "${GPU_RECEIPT_ARGS[@]}" "--eks-flag=$EKS_FLAG" \
    --email "$ADMIN_EMAIL" \
    --repo "$REPO" --previous-site-id "$PREVIOUS_SITE_ID" --email-wait "$EMAIL_WAIT" \
    --aurora-final-snapshot "$AURORA_FINAL_SNAPSHOT"
}

record_stage() {  # N NAME STATUS SECONDS LOG REASON; LOG.extra.json holds JSON lines
  python3 -m scripts.e2e.regional.boot029_receipts finish --record "$RECORD" \
    --state "$STATE_DIR" --stage "$1" --name "$2" --status "$3" \
    --seconds "$4" --log "$5"
}

admin_binary() {
  local verb="$1" dir="" previous="" argument
  for argument in "$@"; do
    [[ "$previous" == "--state-dir" ]] && dir="$argument"
    [[ "$argument" == --state-dir=* ]] && dir="${argument#--state-dir=}"
    previous="$argument"
  done
  if [[ "$verb" == deploy || -z "$dir" ]]; then
    printf 'gpu-fault-admin'
    return
  fi
  python3 - "$dir" <<'PY'
from pathlib import Path
import sys
from gpu_fault.admin.deploy_host_binding import site_bound_admin
from gpu_fault.admin.site import SiteConfigError

try:
    admin = site_bound_admin(Path(sys.argv[1]))
except SiteConfigError as exc:
    print(f"gpu-fault-admin: {exc}", file=sys.stderr)
    raise SystemExit(2) from None
print(admin or "gpu-fault-admin")
PY
}

run_admin() {  # waits for the command to exit; never backgrounds a deploy
  local binary
  binary="$(admin_binary "$@")" || return "$?"
  log "$binary $1 ... (waiting for the process to exit)"
  local command=("$binary" "$@")
  if [[ "$binary" != gpu-fault-admin ]]; then
    command=(env -u PYTHONPATH -u PYTHONHOME -u GPU_FAULT_REPOSITORY_ROOT
      -u GPU_FAULT_REPO_ROOT "PATH=${binary%/*}:$PATH" "$binary" "$@")
  fi
  "${command[@]}" 2>&1 | tee -a "$STAGE_LOG" || return "$?"
}

assert_current_journal() {  # STATE_DIR [GPU_ARN ...]: default every GPU cluster
  local state="$1" policy=() arns=()
  shift
  if [[ $# -gt 0 ]]; then
    for arn in "$@"; do arns+=(--gpu-arn "$arn"); done
  else
    arns=("${GPU_RECEIPT_ARGS[@]}")
  fi
  if [[ "$CURRENT_STAGE" == 4 ]]; then
    policy=(--aurora-final-snapshot "$AURORA_FINAL_SNAPSHOT")
  fi
  python3 -m scripts.e2e.regional.boot029_receipts verify --record "$RECORD" \
    --state "$state" --stage "$CURRENT_STAGE" --cpu-arn "$CPU_ARN" "${arns[@]}" \
    ${policy[@]+"${policy[@]}"} >>"${STAGE_LOG}.extra.json"
}

# --- cold build ------------------------------------------------------------

prune_image_caches() {
  local builder images inspected
  # Read the whole report first: an early-exiting awk closes docker's pipe, and
  # under pipefail that turns a healthy builder into a silent stage failure.
  inspected="$(docker buildx inspect 2>&1)" ||
    fail "docker buildx inspect failed: ${inspected##*$'\n'}"
  builder="$(printf '%s\n' "$inspected" | awk '/^Name:/ {print $2; exit}')"
  [[ -n "$builder" ]] || fail "docker buildx inspect reported no builder name"
  log "pruning BuildKit caches (builder ${builder}) and local runtime images"
  docker buildx prune --builder "$builder" -af 2>&1 | tee -a "$STAGE_LOG"
  docker builder prune -af 2>&1 | tee -a "$STAGE_LOG"
  images="$(docker images --format '{{.Repository}}:{{.Tag}}')"
  { printf '%s\n' "$images" | grep -E '(^|/)gpu-fault[-/](runtime|executor|node-dependencies)' || true; } |
    xargs -r docker image rm -f 2>&1 | tee -a "$STAGE_LOG"
}

clear_site_ecr_tags() {  # SITE_ID: delete the site's image and buildcache tags, only if the repos exist
  python3 -m scripts.e2e.regional.boot029_receipts clear-ecr --record "$RECORD" \
    --state "$STATE_DIR" --stage "$CURRENT_STAGE" --cpu-arn "$CPU_ARN" \
    --previous-site-id "$1" >>"$STAGE_LOG"
}

assert_cold_build() {  # the stage log must show a full, uncached image build
  python3 -m scripts.e2e.regional.boot029_receipts cold --record "$RECORD" \
    --state "$1" --stage "$CURRENT_STAGE" --log "$STAGE_LOG" \
    >>"${STAGE_LOG}.extra.json"
}

# --- stages ----------------------------------------------------------------

run_path_deploy() {  # STATE_DIR: the driver's own four-argument deploy (stage 1)
  local wait=()
  [[ "$EMAIL_WAIT" -gt 0 ]] && wait=(--wait-for-email-confirmation "$EMAIL_WAIT")
  run_admin deploy --cpu-cluster-arn "$CPU_ARN" "${GPU_ARN_ARGS[@]}" \
    ${EKS_ARGS[@]+"${EKS_ARGS[@]}"} \
    --state-dir "$1" --admin-email "$ADMIN_EMAIL" ${wait[@]+"${wait[@]}"}
}

run_readme_deploy() {  # STATE_DIR: README.md's procedure, verbatim, clean environment
  log "README procedure from $REPO/README.md (waiting for the process to exit)"
  python3 -m scripts.e2e.regional.boot029_readme --readme "$REPO/README.md" \
    --repo "$REPO" --work-dir "$ACCEPT_DIR/stage-5-readme" --attempt "$CURRENT_ATTEMPT" \
    --cpu-arn "$CPU_ARN" "${GPU_RECEIPT_ARGS[@]}" ${EKS_ARGS[@]+"${EKS_ARGS[@]}"} \
    --state-dir "$1" \
    --admin-email "$ADMIN_EMAIL" --receipt "${STAGE_LOG}.extra.json" 2>&1 |
    tee -a "$STAGE_LOG" || return "$?"
}

deploy_fresh() {  # STATE_DIR SITE_ID_FOR_ECR_CHECK DEPLOY_FUNCTION
  require_fresh_state_dir "$1"
  # Do not destroy caches of an already-started build when resuming its deploy.
  if [[ ! -f "$1/bootstrap-state.json" ]]; then
    prune_image_caches
    clear_site_ecr_tags "$2"
  fi
  "$3" "$1"
  assert_cold_build "$1"
  assert_current_journal "$1"
}

stage_1() { deploy_fresh "$STATE_DIR" "$PREVIOUS_SITE_ID" run_path_deploy; }

stage_2() {  # one ARN at a time; every removed cluster's journal is asserted
  local removed=()
  for arn in "${GPU_ARNS[@]}"; do
    run_admin remove-cluster --state-dir "$STATE_DIR" --gpu-cluster-arn "$arn" \
      --confirm REMOVE_GPU_CLUSTER
    removed+=("$arn")
    assert_current_journal "$STATE_DIR" "${removed[@]}"
  done
}

stage_3() {  # one join-cluster for every GPU cluster; one journal per ARN
  run_admin join-cluster --state-dir "$STATE_DIR" "${GPU_ARN_ARGS[@]}" \
    ${EKS_ARGS[@]+"${EKS_ARGS[@]}"}
  assert_current_journal "$STATE_DIR"
}

stage_4() {
  run_admin uninstall --state-dir "$STATE_DIR" --cpu-cluster keep --reset-database \
    --aurora-final-snapshot "$AURORA_FINAL_SNAPSHOT" --confirm UNINSTALL_GPU_FAULT
  assert_current_journal "$STATE_DIR"
}

stage_5() {
  local site_id=""
  [[ -f "$STATE_DIR/bootstrap-state.json" ]] && site_id="$(json_field "$STATE_DIR/bootstrap-state.json" site_id)"
  deploy_fresh "$SECOND_STATE_DIR" "$site_id" run_readme_deploy
}

run_stage() {  # N
  local number="$1" name="${STAGE_NAMES[$1]}" started status reason="" attempt state code
  CURRENT_STAGE="$number"
  state="$STATE_DIR"
  [[ "$number" == 5 ]] && state="$SECOND_STATE_DIR"
  attempt="$(python3 -m scripts.e2e.regional.boot029_receipts start --record "$RECORD" --state "$state" --stage "$number")"
  CURRENT_ATTEMPT="$attempt"
  STAGE_LOG="${ACCEPT_DIR}/stage-${number}-${name}-attempt-${attempt}.log"
  [[ ! -e "$STAGE_LOG" ]] || fail "attempt log already exists"
  : >"$STAGE_LOG"
  started=$SECONDS
  log "== stage ${number}/5 ${name}"
  # A function invoked inside `if` inherits errexit suppression. Execute the
  # stage as a normal subshell command and capture its status afterwards.
  set +e
  ( set -e; "stage_${number}" )
  code=$?
  set -e
  if [[ "$code" == 0 ]]; then
    status=PASS
  else
    status=FAIL
    reason="$(grep -E '^FAIL: ' "$STAGE_LOG" | tail -n 1 || true)"
    reason="${reason:-a command exited non-zero; see the stage log}"
  fi
  record_stage "$number" "$name" "$status" "$((SECONDS - started))" "$STAGE_LOG" "$reason"
  log "== stage ${number}/5 ${name}: ${status} ($((SECONDS - started)) s)"
  if [[ "$status" == FAIL ]]; then
    echo "FAILED at stage ${number} (${name}): ${reason}" >&2
    echo "fix the cause, then resume with --stage ${number}" >&2
    exit 1
  fi
}

install -d -m 0700 "$ACCEPT_DIR"
[[ ! -L "$ACCEPT_DIR/sequence.lock" ]] || { echo "unsafe sequence lock" >&2; exit 1; }
exec {SEQUENCE_LOCK_FD}>"$ACCEPT_DIR/sequence.lock"
flock -n "$SEQUENCE_LOCK_FD" || { echo "another BOOT-029 sequence is running" >&2; exit 1; }
init_record
for stage in $(seq "$START_STAGE" 5); do
  run_stage "$stage"
done
log "${CASE_ID}: all five driver stages PASS; record ${RECORD}"
if [[ "$AURORA_FINAL_SNAPSHOT" == skip ]]; then
  log "GF-REGIONAL-BOOT-027: NOT_RUN (Aurora final snapshot explicitly skipped)"
fi
