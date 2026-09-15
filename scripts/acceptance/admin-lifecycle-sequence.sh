#!/usr/bin/env bash
# GF-REGIONAL-BOOT-029 driver: the administrator deployment lifecycle as one
# ordered acceptance sequence on an isolated staging site.
#
#   1  cold first deploy (BuildKit caches, local runtime images and the site's
#      ECR image/buildcache tags are gone before the build)
#   2  remove-cluster of the site's only GPU cluster
#   3  join-cluster --gpu-cluster-arn re-attach
#   4  uninstall --cpu-cluster keep --reset-database (--aurora-final-snapshot
#      retain|skip, default retain: skip only when the staging data is disposable)
#   5  cold first deploy again, into a second empty state directory
#
# Every admin command runs in the foreground and the next stage starts only
# after it has exited. The first failing stage stops the run and prints its
# number so `--stage N` can resume. Wall time and the sha256 of each stage's log
# go to <state-dir>/acceptance/admin-lifecycle-sequence.json; ARNs, the email
# and the paths appear there only as sha256 digests.
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: admin-lifecycle-sequence.sh --state-dir DIR --cpu-cluster-arn ARN
         --gpu-cluster-arn ARN --admin-email EMAIL --repo CHECKOUT
         [--stage N] [--second-state-dir DIR] [--email-wait-minutes M]
         [--previous-site-id ID] [--aurora-final-snapshot retain|skip]
USAGE
  exit 2
}

STATE_DIR="" CPU_ARN="" GPU_ARN="" ADMIN_EMAIL="" REPO="" START_STAGE=1
SECOND_STATE_DIR="" EMAIL_WAIT=0 PREVIOUS_SITE_ID="" AURORA_FINAL_SNAPSHOT=retain
while [[ $# -gt 0 ]]; do
  case "$1" in
    --state-dir) STATE_DIR="$2"; shift 2 ;;
    --cpu-cluster-arn) CPU_ARN="$2"; shift 2 ;;
    --gpu-cluster-arn) GPU_ARN="$2"; shift 2 ;;
    --admin-email) ADMIN_EMAIL="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --stage) START_STAGE="$2"; shift 2 ;;
    --second-state-dir) SECOND_STATE_DIR="$2"; shift 2 ;;
    --email-wait-minutes) EMAIL_WAIT="$2"; shift 2 ;;
    --previous-site-id) PREVIOUS_SITE_ID="$2"; shift 2 ;;
    --aurora-final-snapshot) AURORA_FINAL_SNAPSHOT="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "$STATE_DIR" && -n "$CPU_ARN" && -n "$GPU_ARN" && -n "$ADMIN_EMAIL" && -n "$REPO" ]] || usage
[[ "$START_STAGE" =~ ^[1-5]$ ]] || usage
[[ "$AURORA_FINAL_SNAPSHOT" =~ ^(retain|skip)$ ]] || usage
[[ -d "$REPO/deploy/image" ]] || { echo "--repo is not a checkout: $REPO" >&2; exit 2; }

STATE_DIR="$(realpath -m "$STATE_DIR")"
SECOND_STATE_DIR="$(realpath -m "${SECOND_STATE_DIR:-${STATE_DIR}-second}")"
ACCEPT_DIR="${STATE_DIR}/acceptance"
RECORD="${ACCEPT_DIR}/admin-lifecycle-sequence.json"
CASE_ID="GF-REGIONAL-BOOT-029"
STAGE_NAMES=("" cold-first-deploy remove-cluster join-cluster uninstall-reset cold-redeploy)
STAGE_LOG=""
# The deploy reads its source tree from the working directory.
cd "$REPO"
export BUILDKIT_PROGRESS=plain

log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
fail() { printf 'FAIL: %s\n' "$*" | tee -a "$STAGE_LOG" >&2; exit 1; }
digest_of() { printf '%s' "$1" | sha256sum | cut -c1-64; }

json_field() {  # FILE KEY -> value or empty
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))' "$1" "$2"
}

assert_phase() {  # FILE EXPECTED
  [[ -f "$1" ]] || fail "missing state file $(basename "$1")"
  local phase; phase="$(json_field "$1" phase)"
  [[ "$phase" == "$2" ]] || fail "$(basename "$1") phase=$phase, expected $2"
  printf '{"phase": "%s"}\n' "$phase" >>"${STAGE_LOG}.extra.json"
}

newest_state() {  # DIR -> newest DIR/*/state.json
  local best="" file
  for file in "$1"/*/state.json; do
    [[ -f "$file" ]] || continue
    [[ -z "$best" || "$file" -nt "$best" ]] && best="$file"
  done
  printf '%s' "$best"
}

require_fresh_state_dir() {
  [[ -f "$1/bootstrap-state.json" ]] && fail "$(basename "$1") already holds bootstrap-state.json: a first deploy needs an empty state directory"
  install -d -m 0700 "$1"
}

init_record() {
  install -d -m 0700 "$ACCEPT_DIR"
  python3 - "$RECORD" "$CASE_ID" "$(digest_of "$CPU_ARN")" "$(digest_of "$GPU_ARN")" \
    "$(digest_of "$ADMIN_EMAIL")" "$(digest_of "$STATE_DIR")" \
    "$(digest_of "$SECOND_STATE_DIR")" "$START_STAGE" "$AURORA_FINAL_SNAPSHOT" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
doc = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
doc.update(
    case_id=sys.argv[2],
    schema_version=1,
    inputs={
        "cpu_cluster_arn_sha256": sys.argv[3],
        "gpu_cluster_arn_sha256": sys.argv[4],
        "admin_email_sha256": sys.argv[5],
        "state_dir_sha256": sys.argv[6],
        "second_state_dir_sha256": sys.argv[7],
        "aurora_final_snapshot": sys.argv[9],
    },
    resumed_from_stage=int(sys.argv[8]),
)
doc.setdefault("stages", {})
path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

record_stage() {  # N NAME STATUS SECONDS LOG REASON; LOG.extra.json holds JSON lines
  python3 - "$RECORD" "$@" <<'PY'
import hashlib, json, pathlib, sys
path, number, name, status, seconds, log, reason = sys.argv[1:8]
record = pathlib.Path(path)
doc = json.loads(record.read_text(encoding="utf-8"))
extra = pathlib.Path(log + ".extra.json")
stage = {
    "name": name,
    "status": status,
    "wall_seconds": int(seconds),
    "log_sha256": hashlib.sha256(pathlib.Path(log).read_bytes()).hexdigest(),
    "failure": reason or None,
}
if extra.is_file():  # one JSON object per line, one per assertion
    for line in extra.read_text(encoding="utf-8").splitlines():
        stage.update(json.loads(line))
doc.setdefault("stages", {})[number] = stage
record.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_admin() {  # waits for the command to exit; never backgrounds a deploy
  log "gpu-fault-admin $1 ... (waiting for the process to exit)"
  gpu-fault-admin "$@" 2>&1 | tee -a "$STAGE_LOG"
}

# --- cold build ------------------------------------------------------------

prune_image_caches() {
  local builder
  builder="$(docker buildx inspect | awk '/^Name:/ {print $2; exit}')"
  [[ -n "$builder" ]] || fail "docker buildx inspect reported no builder name"
  log "pruning BuildKit caches (builder ${builder}) and local runtime images"
  docker buildx prune --builder "$builder" -af 2>&1 | tee -a "$STAGE_LOG"
  docker builder prune -af 2>&1 | tee -a "$STAGE_LOG"
  { docker images --format '{{.Repository}}:{{.Tag}}' | grep -E '(^|/)gpu-fault[-/]runtime' || true; } |
    xargs -r docker image rm -f 2>&1 | tee -a "$STAGE_LOG"
}

clear_site_ecr_tags() {  # SITE_ID: delete the site's image and buildcache tags, only if the repos exist
  local site_id="$1" digest repo kind ids
  if [[ -z "$site_id" ]]; then
    log "no site identifier yet: a fresh site has no ECR repositories to clear"
    return 0
  fi
  digest="$(digest_of "$site_id" | cut -c1-12)"
  for kind in runtime runtime-cache; do
    repo="gpu-fault/${kind}-${digest}"
    if ! aws ecr describe-repositories --repository-names "$repo" >/dev/null 2>&1; then
      log "ECR repository ${kind} is absent (expected before a first deploy)"
      continue
    fi
    if [[ "$kind" == runtime-cache ]]; then
      ids='[{"imageTag":"buildcache-linux-amd64"}]'
    else
      ids="$(aws ecr list-images --repository-name "$repo" --query imageIds --output json)"
    fi
    if [[ "$ids" != "[]" ]]; then
      aws ecr batch-delete-image --repository-name "$repo" --image-ids "$ids" >/dev/null
      log "deleted stale tags from ECR repository ${kind}"
    fi
  done
}

assert_cold_build() {  # the stage log must show a full, uncached image build
  local cached pulled steps
  cached="$(grep -cE '^#[0-9]+ CACHED$' "$STAGE_LOG" || true)"
  pulled="$(grep -cE '^#[0-9]+ extracting sha256:' "$STAGE_LOG" || true)"
  steps="$(grep -oE '^#[0-9]+ \[[0-9]+/[0-9]+\] ' "$STAGE_LOG" | sed -E 's|.*/([0-9]+)\] $|\1|' | head -n 1)"
  printf '{"cold_build": {"cached_layers": %s, "base_image_pulled": %s, "dockerfile_steps": %s}}\n' \
    "$cached" "$([[ "$pulled" -gt 0 ]] && echo true || echo false)" "${steps:-0}" >>"${STAGE_LOG}.extra.json"
  [[ -n "$steps" ]] || fail "the deploy log has no Dockerfile step lines: the runtime image was not built"
  [[ "$cached" == 0 ]] || fail "the image build used ${cached} CACHED layer(s); the build was not cold"
  [[ "$pulled" -gt 0 ]] || fail "the base image was not pulled during the build"
}

# --- stages ----------------------------------------------------------------

deploy_fresh() {  # STATE_DIR SITE_ID_FOR_ECR_CHECK
  require_fresh_state_dir "$1"
  prune_image_caches
  clear_site_ecr_tags "$2"
  local wait=()
  [[ "$EMAIL_WAIT" -gt 0 ]] && wait=(--wait-for-email-confirmation "$EMAIL_WAIT")
  run_admin deploy --cpu-cluster-arn "$CPU_ARN" --gpu-cluster-arn "$GPU_ARN" \
    --state-dir "$1" --admin-email "$ADMIN_EMAIL" "${wait[@]}"
  assert_cold_build
  assert_phase "$1/bootstrap-state.json" site-ready
}

stage_1() { deploy_fresh "$STATE_DIR" "$PREVIOUS_SITE_ID"; }

stage_2() {
  run_admin remove-cluster --state-dir "$STATE_DIR" --gpu-cluster-arn "$GPU_ARN" \
    --confirm REMOVE_GPU_CLUSTER
  assert_phase "$(newest_state "$STATE_DIR/remove-cluster")" COMPLETED
}

stage_3() {
  run_admin join-cluster --state-dir "$STATE_DIR" --gpu-cluster-arn "$GPU_ARN"
  assert_phase "$(newest_state "$STATE_DIR/join-cluster")" COMPLETED
}

stage_4() {
  run_admin uninstall --state-dir "$STATE_DIR" --cpu-cluster keep --reset-database \
    --aurora-final-snapshot "$AURORA_FINAL_SNAPSHOT" --confirm UNINSTALL_GPU_FAULT
  printf '{"aurora_final_snapshot": "%s"}\n' "$AURORA_FINAL_SNAPSHOT" >>"${STAGE_LOG}.extra.json"
  assert_phase "$STATE_DIR/uninstall/state.json" COMPLETED
}

stage_5() {
  local site_id=""
  [[ -f "$STATE_DIR/bootstrap-state.json" ]] && site_id="$(json_field "$STATE_DIR/bootstrap-state.json" site_id)"
  deploy_fresh "$SECOND_STATE_DIR" "$site_id"
}

run_stage() {  # N
  local number="$1" name="${STAGE_NAMES[$1]}" started status reason=""
  STAGE_LOG="${ACCEPT_DIR}/stage-${number}-${name}.log"
  rm -f "$STAGE_LOG" "${STAGE_LOG}.extra.json"
  : >"$STAGE_LOG"
  started=$SECONDS
  log "== stage ${number}/5 ${name}"
  if ( "stage_${number}" ); then
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

init_record
for stage in $(seq "$START_STAGE" 5); do
  run_stage "$stage"
done
log "${CASE_ID}: all five stages PASS; record ${RECORD}"
