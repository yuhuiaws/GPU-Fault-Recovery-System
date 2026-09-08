# Runtime Profile 变更审批

本文面向区域站点管理员，说明deploy因Runtime Profile策略变化停止后，如何审核计划、
绑定外部变更单、让同一条deploy继续并保存审计证据。整个流程只有两步，不需要`jq`、
内部site路径、`PROFILE_APPROVAL`环境变量或隐藏参数。

Profile是站点级授权策略。新版本生效后会被该站点当前纳管的全部GPU集群使用；GPU集群
加入和注销继续通过`join-cluster/remove-cluster`处理，不进入Profile审批的稳定站点
身份摘要。

## 1. 最短操作路径

第一步：照常执行四参数deploy。

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault \
  --admin-email <operations-email>
```

Profile变化时，命令写出`/secure/gpu-fault/release-deploy/profile-plan.json`并停止，
停止信息直接打印审核字段和续跑命令：

```text
Runtime Profile policy changed; the deploy stopped for review.
Review /secure/gpu-fault/release-deploy/profile-plan.json:
  site_identity:
    site_name: prod
    aws_region: us-east-1
    cpu_eks_arn: arn:aws:eks:us-east-1:123456789012:cluster/control
  version: regional-hyperpod-3f2a1c9b7d40 -> regional-hyperpod-8e51b0c2a7f3
  change_kind: EXPANSIVE
  changes:
    - gpuReset: mode OBSERVE->OWN
  live_profile_sha256: ...
  policy_digest: ...
  snapshot_sha256: ...
  plan_sha256: <plan_sha256>
Record plan_sha256 on the change request; once it is approved, rerun:
  gpu-fault-admin deploy --state-dir /secure/gpu-fault --approve-profile-plan <plan_sha256> --reference CHG-<id>
```

第二步：按第4节审核这些字段，把`plan_sha256`写入变更单，取得批准后原样执行打印出的
续跑命令，只把`CHG-<id>`换成真实变更单号：

```bash
gpu-fault-admin deploy \
  --state-dir /secure/gpu-fault \
  --approve-profile-plan <plan_sha256> \
  --reference CHG-12345
```

该命令先在同一state-dir上完成审批，再继续原部署直到verify、stability和commit结束。
没有第三步。

## 2. 状态流转

```text
四参数deploy
  |
  +-- Profile未变化 ----------------------------> 正常发布
  |
  +-- Profile变化 -> profile-plan.json -> 停止并打印续跑命令 -> 人工审核
                                      |
                                      +-- deploy --approve-profile-plan <sha> --reference <ref>
                                            |
                                            +-- 计划未漂移 -> 审批生效 -> 继续发布 -> CONSUMED
                                            +-- 计划漂移 -> 旧审批SUPERSEDED -> 重新停止并打印新摘要
                                            +-- 发布失败且计划未漂移 -> 重跑原四参数deploy即可续跑
```

管理员不创建审批归档目录，也不复制计划文件。续跑命令自动创建
`release-deploy/profile-approvals/<plan_sha256>/`并写入审计文件。

## 3. 审核前提

执行续跑命令前确认：

1. 使用生成计划时相同的`state-dir`和管理员身份。
2. Release来自受信签名制品，普通deploy没有使用staging-only候选进入生产。
3. 外部变更单或维护窗口已经明确目标Profile和风险范围。
4. 若变化开放reset、reboot、warm spare或其他破坏性能力，已满足对应维护窗口和阶段验收。
5. 没有另一个deploy命令正在使用同一`state-dir`。
6. `profile-plan.json`没有被人工编辑、复制回填或改变权限。

## 4. 审核计划

停止信息已经打印管理员必须审核的字段；需要完整内容时直接查看JSON文件：

```text
<state-dir>/release-deploy/profile-plan.json
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
| `plan_sha256` | 整份审核计划的绑定值，必须写入变更单并原样传给`--approve-profile-plan` |

`change_kind`含义：

| 值 | 管理员判断 |
|---|---|
| `EXPANSIVE` | 扩大能力或执行权限；重点确认破坏性能力和维护窗口 |
| `OWNER_CHANGE` | owner或adapter变化；确认不存在第二个writer |
| `RESTRICTIVE` | 收紧或关闭能力；确认不会破坏当前恢复SLO |
| `IMPLEMENTATION_CHANGE` | observed实现或版本变化；确认实现已验收 |
| `UNKNOWN_BASELINE` | 线上基线不可证明；不得审批，先修复证据或状态 |

`UNCHANGED`不会生成待审批Profile变更。

只修改模板、代码与site不变的部署同样会生成计划：staging层在判定`UNCHANGED`前先用
`plan_runtime_profile`把模板与线上Profile比对，存在待审批变更时直接走完整的
application release路径。

## 5. 绑定外部批准并继续部署

`--approve-profile-plan`只接受当前待审批计划的`plan_sha256`：

- state-dir下没有`profile-plan.json`时命令拒绝执行——第一次deploy不能带这个参数。
- 传入值与待审批计划摘要不一致时命令拒绝执行，并在错误里给出当前待审批的摘要；
  不写任何活动审批。
- 同一计划已审批后不能用不同`--reference`覆盖，因此执行前必须确认变更单号正确。

审批在同一文件锁内完成，记录审批人身份（STS调用者ARN）、变更单号、审批时间和站点
身份，随后同一进程继续发布。发布器重新计算站点身份、Profile目标和live baseline，
全部匹配后才生成不可变快照、更新内部site并执行deploy、verify、stability和commit。

若审批和发布之间模板、目标策略或live baseline又发生变化，旧审批被归档为
`SUPERSEDED`，命令重新停止并打印新的`plan_sha256`和续跑命令；按第4节重新审核。

## 6. 成功证据

续跑命令自动创建：

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

## 7. 异常处理

| 现象 | 处理 |
|---|---|
| deploy失败但没有`profile-plan.json` | 不是审批暂停；按原错误修复，不带`--approve-profile-plan`重跑 |
| `--approve-profile-plan`被拒绝：没有待审批计划 | 先跑不带该参数的四参数deploy，让它生成并打印计划 |
| `--approve-profile-plan`被拒绝：摘要不匹配 | 计划在审核后发生变化；按错误里给出的当前摘要重新审核并更新变更单 |
| `site_identity`或其摘要不匹配 | 停止；确认state-dir和CPU控制面身份 |
| site `source`路径漂移，但本地版本快照SHA与live SHA一致 | deploy自动使用该不可变快照恢复baseline并修正site |
| `UNKNOWN_BASELINE` | 本地快照缺失或SHA不一致；不审批，先恢复可信live Profile证据 |
| 发布失败，审批仍活动且计划未变 | 修复发布问题，重跑原四参数deploy，无需重复审批 |
| 旧审批标记为`SUPERSEDED` | 按新打印的`plan_sha256`重新审核，再执行新的续跑命令 |
| 发布已完成但归档收尾失败 | 重跑原四参数deploy；系统以`ALREADY_APPLIED`完成消费 |
| 提示审批或发布正在进行 | 等待当前进程结束；不删除lock文件 |
| GPU集群加入或注销 | 使用正式join/remove命令；GPU集合不改变稳定Profile审批身份 |

## 8. 禁止操作

- 不手工创建`profile-approvals/<plan_sha256>/`目录。
- 不手工复制、移动或编辑`plan.json`、`approval.json`、`consumed.json`。
- 不删除活动审批、lock或失败release state来强制重跑。
- 不通过环境变量、隐藏参数或直接调用内部`release-deploy`注入审批引用。
- 不在同一维护窗口并行运行两个针对同一`state-dir`的deploy命令。
- 不把`plan_sha256`当作Secret；它是审计身份，不是凭据。
