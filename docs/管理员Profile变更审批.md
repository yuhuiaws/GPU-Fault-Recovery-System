# Runtime Profile 变更审批

本文面向区域站点管理员，说明检测到Runtime Profile策略变化后，如何审核计划、绑定外部
变更单、继续原四参数部署并保存审计证据。日常流程不使用内部site路径、
`PROFILE_APPROVAL`环境变量或隐藏deploy参数。

Profile是站点级授权策略。新版本生效后会被该站点当前纳管的全部GPU集群使用；GPU集群
加入和注销继续通过`join-cluster/remove-cluster`处理，不进入Profile审批的稳定站点
身份摘要。

## 1. 最短操作路径

第一次执行原四参数deploy：

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault \
  --admin-email <operations-email>
```

Profile变化时，命令会生成
`/secure/gpu-fault/release-deploy/profile-plan.json`并停止。审核计划并取得变更单批准
后执行：

```bash
STATE_DIR=/secure/gpu-fault
PLAN_SHA256="$(jq -er '.plan_sha256' \
  "${STATE_DIR}/release-deploy/profile-plan.json")"

gpu-fault-admin approve-profile \
  --state-dir "${STATE_DIR}" \
  --plan-sha256 "${PLAN_SHA256}" \
  --reference CHG-12345
```

最后重新执行完全相同的四参数deploy。多个GPU集群仍通过重复
`--gpu-cluster-arn <gpu-arn>`传入。

## 2. 状态流转

```text
四参数deploy
  |
  +-- Profile未变化 ----------------------------> 正常发布
  |
  +-- Profile变化 -> profile-plan.json -> 人工审核
                                      |
                                      +-- approve-profile
                                            |
                                            +-- 发布失败且计划未漂移 -> 原命令续跑
                                            +-- 计划漂移 -> SUPERSEDED -> 重新审核
                                            +-- 发布成功 -> CONSUMED
```

管理员不创建审批归档目录，也不复制计划文件。`approve-profile`自动创建
`release-deploy/profile-approvals/<plan_sha256>/`并写入审计文件。

## 3. 审核前提

执行审批前确认：

1. 使用生成计划时相同的`state-dir`、CPU ARN和管理员身份。
2. Release来自受信签名制品，普通deploy没有使用staging-only候选进入生产。
3. 外部变更单或维护窗口已经明确目标Profile和风险范围。
4. 若变化开放reset、reboot、warm spare或其他破坏性能力，已满足对应维护窗口和阶段验收。
5. 没有另一个审批或release命令正在使用同一`state-dir`。
6. `profile-plan.json`没有被人工编辑、复制回填或改变权限。

## 4. 审核计划

计划位置固定为：

```text
<state-dir>/release-deploy/profile-plan.json
```

查看管理员需要审核的字段：

```bash
STATE_DIR=/secure/gpu-fault
PLAN_FILE="${STATE_DIR}/release-deploy/profile-plan.json"

jq '{
  site_identity,
  site_identity_sha256,
  registration_cluster_id,
  versions: {
    current: .current_version,
    desired: .desired_version
  },
  change_kind,
  changes,
  current_policy_digest,
  policy_digest,
  live_profile_sha256,
  active_source_sha256,
  source_sha256,
  snapshot_sha256,
  plan_sha256,
  approval_required
}' "${PLAN_FILE}"
```

| 字段 | 审核要求 |
|---|---|
| `site_identity.site_name` | 必须是目标站点 |
| `site_identity.aws_region` | 必须是批准的Region |
| `site_identity.cpu_eks_arn` | 必须是该站点稳定CPU控制面 |
| `site_identity_sha256` | 由程序重新计算，防止可读身份与摘要不一致 |
| `registration_cluster_id` | Profile稳定注册锚点；不要求仍属于当前GPU集合 |
| `current_version` / `desired_version` | 必须符合变更单描述的版本迁移 |
| `change_kind` | 必须与风险和维护窗口一致 |
| `changes[]` | 每项capability、mode、owner、adapter变化均须解释 |
| `live_profile_sha256` | 审批时线上Profile baseline |
| `policy_digest` | 规范化目标策略摘要 |
| `source_sha256` | 开发者模板原始内容摘要 |
| `snapshot_sha256` | 将要发布的不可变Profile快照摘要 |
| `plan_sha256` | 整份审核计划的绑定值，必须写入变更单并传回审批命令 |

`change_kind`含义：

| 值 | 管理员判断 |
|---|---|
| `EXPANSIVE` | 扩大能力或执行权限；重点确认破坏性能力和维护窗口 |
| `OWNER_CHANGE` | owner或adapter变化；确认不存在第二个writer |
| `RESTRICTIVE` | 收紧或关闭能力；确认不会破坏当前恢复SLO |
| `IMPLEMENTATION_CHANGE` | observed实现或版本变化；确认实现已验收 |
| `UNKNOWN_BASELINE` | 线上基线不可证明；不得审批，先修复证据或状态 |

`UNCHANGED`不会生成待审批Profile变更。

## 5. 绑定外部批准

将审核过的`plan_sha256`记录到变更单。取得批准后，原样传回命令：

```bash
STATE_DIR=/secure/gpu-fault
PLAN_FILE="${STATE_DIR}/release-deploy/profile-plan.json"
PLAN_SHA256="$(jq -er '.plan_sha256' "${PLAN_FILE}")"

test "${PLAN_SHA256}" != "null"
test "${#PLAN_SHA256}" -eq 64

gpu-fault-admin approve-profile \
  --state-dir "${STATE_DIR}" \
  --plan-sha256 "${PLAN_SHA256}" \
  --reference CHG-12345
```

命令在同一文件锁内重新读取计划并比较SHA。当前计划与管理员传入值不一致时，不写任何
活动审批并立即失败。

审批成功输出包含：

```json
{
  "status": "APPROVED",
  "site_identity": {
    "site_name": "prod",
    "aws_region": "us-east-1",
    "cpu_eks_arn": "arn:aws:eks:us-east-1:123456789012:cluster/control"
  },
  "plan_sha256": "<reviewed-plan-sha256>",
  "reference": "CHG-12345"
}
```

同一计划已审批后不能用不同`reference`覆盖，因此执行命令前必须确认变更单号正确。

## 6. 继续部署

重新执行首次生成计划时完全相同的四参数deploy：

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault \
  --admin-email <operations-email>
```

发布器会重新计算站点身份、Profile目标和live baseline。全部匹配后才会生成不可变快照、
更新内部site并执行deploy、verify、stability和commit。

## 7. 完整命令模板

以下模板保留第一次deploy的真实退出码。只有命令失败且确实生成Profile计划时，才进入
审批分支；其他失败必须立即停止。第一次运行不设置`APPROVED_PLAN_SHA256`和
`CHANGE_REFERENCE`，脚本展示计划后退出；外部审批完成后设置这两个值并重新执行。

```bash
set -euo pipefail

STATE_DIR=/secure/gpu-fault
CPU_CLUSTER_ARN='<cpu-arn>'
GPU_CLUSTER_ARNS=('<gpu-arn>')
ADMIN_EMAIL='<operations-email>'
APPROVED_PLAN_SHA256="${APPROVED_PLAN_SHA256:-}"
CHANGE_REFERENCE="${CHANGE_REFERENCE:-}"
PLAN_FILE="${STATE_DIR}/release-deploy/profile-plan.json"

deploy() {
  local args=(
    --cpu-cluster-arn "${CPU_CLUSTER_ARN}"
    --state-dir "${STATE_DIR}"
    --admin-email "${ADMIN_EMAIL}"
  )
  local gpu_arn
  for gpu_arn in "${GPU_CLUSTER_ARNS[@]}"; do
    args+=(--gpu-cluster-arn "${gpu_arn}")
  done
  gpu-fault-admin deploy "${args[@]}"
}

set +e
deploy
DEPLOY_RC=$?
set -e

if [ "${DEPLOY_RC}" -eq 0 ]; then
  printf '%s\n' 'No Profile approval was required.'
  exit 0
fi

test -f "${PLAN_FILE}" || exit "${DEPLOY_RC}"

jq -e '{
  site_identity,
  current_version,
  desired_version,
  change_kind,
  changes,
  live_profile_sha256,
  policy_digest,
  snapshot_sha256,
  plan_sha256
}' "${PLAN_FILE}"

PLAN_SHA256="$(jq -er '.plan_sha256' "${PLAN_FILE}")"

if [ -z "${APPROVED_PLAN_SHA256}" ] || [ -z "${CHANGE_REFERENCE}" ]; then
  printf '%s\n' \
    'Stop: approve this plan externally, then set APPROVED_PLAN_SHA256 and CHANGE_REFERENCE.'
  exit 3
fi

test "${APPROVED_PLAN_SHA256}" = "${PLAN_SHA256}" || {
  printf '%s\n' 'Approved SHA does not match the current Profile plan.' >&2
  exit 4
}

gpu-fault-admin approve-profile \
  --state-dir "${STATE_DIR}" \
  --plan-sha256 "${APPROVED_PLAN_SHA256}" \
  --reference "${CHANGE_REFERENCE}"

deploy
gpu-fault-admin status --state-dir "${STATE_DIR}"
```

## 8. 成功证据

审批命令自动创建：

```text
<state-dir>/release-deploy/profile-approvals/<plan_sha256>/
  plan.json
  approval.json
```

发布成功后自动增加：

```text
consumed.json
```

计划漂移时自动增加：

```text
superseded.json
```

对应release状态位于：

```text
<state-dir>/release-deploy/<release-id>/state.json
```

成功标准：

1. release状态为`COMPLETED`。
2. `profile_approval_audit.status=CONSUMED`。
3. 审计中的`plan_sha256`与变更单完全一致。
4. 活动`profile-plan.json`和`profile-approval.json`已删除。
5. `gpu-fault-admin status`确认目标Profile已注册且无漂移。

## 9. 异常处理

| 现象 | 处理 |
|---|---|
| deploy失败但没有`profile-plan.json` | 不是审批暂停；按原错误修复，不执行审批 |
| `--plan-sha256`不匹配 | 计划在审核后发生变化；重新读取、重新审核并更新变更单 |
| `site_identity`或其摘要不匹配 | 停止；确认state-dir和CPU控制面身份 |
| site `source`路径漂移，但本地版本快照SHA与live SHA一致 | deploy自动使用该不可变快照恢复baseline并修正site |
| `UNKNOWN_BASELINE` | 本地快照缺失或SHA不一致；不审批，先恢复可信live Profile证据 |
| 发布失败，审批仍活动且计划未变 | 修复发布问题，重跑原四参数deploy，无需重复审批 |
| 旧审批标记为`SUPERSEDED` | 审核新`profile-plan.json`并重新审批 |
| 发布已完成但归档收尾失败 | 重跑原四参数deploy；系统以`ALREADY_APPLIED`完成消费 |
| 提示审批或发布正在进行 | 等待当前进程结束；不删除lock文件 |
| GPU集群加入或注销 | 使用正式join/remove命令；GPU集合不改变稳定Profile审批身份 |

## 10. 禁止操作

- 不手工创建`profile-approvals/<plan_sha256>/`目录。
- 不手工复制、移动或编辑`plan.json`、`approval.json`、`consumed.json`。
- 不删除活动审批、lock或失败release state来强制重跑。
- 不通过环境变量、隐藏参数或直接调用内部`release-deploy`注入审批引用。
- 不在同一维护窗口并行运行两个针对同一`state-dir`的审批或发布命令。
- 不把`plan_sha256`当作Secret；它是审计身份，不是凭据。
