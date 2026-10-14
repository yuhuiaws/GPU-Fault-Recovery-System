# Runtime Profile 变更审批

本文面向区域站点管理员，说明deploy因Runtime Profile策略变化停止后，如何审核计划、
绑定外部变更单、让同一条deploy继续并保存审计证据。整个流程只有两步，不需要`jq`、
内部site路径、`PROFILE_APPROVAL`环境变量或隐藏参数。

Profile 是授权策略。站点 profile（`spec.runtimeProfile`）新版本生效后被该站点所有**沿用站点
profile** 的 GPU 集群使用；一个集群也可以拥有自己的 profile（`spec.clusters[].runtimeProfile`，
由 `join-cluster` 在站点 profile 的 `nodeReboot` owner 不能服务该集群 provider 时从对应示例 profile
填入：HyperPod 用 `gpu-fault-hyperpod-adapter`/`hyperpod-v1`，普通 EKS `eks-managed` 用
`gpu-fault-eks-adapter`/`eks-v1`）。每个在用版本都在控制面注册一次；集群条目的
`agentConfigDigest` 是按其版本计算的 Agent 配置摘要；集群注册表携带该版本与摘要，fleet pin 按
集群核对。GPU 集群加入和注销继续通过 `join-cluster/remove-cluster` 处理，不进入站点 profile 审批的
稳定站点身份摘要；但**集群自己的新 profile 版本同样要审批**：join 写出同一份
`release-deploy/profile-plan.json`（`change_kind: CLUSTER_PROFILE_NEW`，`registration_cluster_id` 为该集群），
用停止信息打印的 `join-cluster ... --approve-profile-plan <plan_sha256> --reference <CHG>` 审批后继续
（`deploy --gpu-cluster-arn` 新增集群时打印的也是这条 `join-cluster` 命令；不要改用 `deploy --approve-profile-plan`，见 1.2）；审批消费后写入不可变快照 `profiles/<version>.yaml`，后续同 provider 的集群复用。
多个集群需要同一版本时只出一份计划，一次审批覆盖全部（见 1.2）。

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

站点 Profile 的版本名跟随注册集群 provider 的家族：策略等于该家族示例模板、且已有的声明快照
（`profiles/eks-v1.yaml` / `profiles/hyperpod-v1.yaml`）携带同一策略时，回到声明版本 `eks-v1` / `hyperpod-v1`
并原样复用快照（活基线属于另一家族时，计划显示为 `regional-hyperpod-<digest12> -> eks-v1`）；否则为
`regional-<provider>-<digest12>`。模板的 `nodeReboot` owner 属于另一 provider 的 adapter 时，`deploy` 直接拒绝而不出计划。

### 1.1 已部署站点上集群自有 Profile 的版本升级

站点 Profile 无法重启其 provider 的 GPU 集群（HyperPod 站点上的纯 EKS 集群，或反之）pin 在自己的 Profile 版本上，
记录在 `spec.clusters[].runtimeProfile`（`version`、`source`、`templateSource`、`agentConfigDigest`）。首个版本是
provider 声明的版本（`eks-v1` / `hyperpod-v1`），由 `join-cluster` 审核（`change_kind: CLUSTER_PROFILE_NEW`，
`registration_cluster_id: <cluster>`）。快照 `profiles/<version>.yaml` 不可变：一个版本永不对应两种策略。

pin 之后 provider 示例 Profile 的策略变化（`templateSource` 与已 pin 快照不同）时，`deploy` 不复用版本名，而是
计算后继版本 `regional-<provider>-<digest12>`（模板策略摘要的前 12 位十六进制），写出计划并在 release bundle
与各项 gate 之前停止：

```text
Runtime Profile policy changed; the deploy stopped for review.
Review /secure/gpu-fault/release-deploy/profile-plan.json:
  ...
  cluster: gpu-eks-cluster
  version: eks-v1 -> regional-eks-8e51b0c2a7f3 (this cluster only)
  change_kind: CLUSTER_PROFILE_NEW
  changes:
    - cluster gpu-eks-cluster (eks-managed) moves from its own Runtime Profile eks-v1 to regional-eks-8e51b0c2a7f3: ...
  plan_sha256: <plan_sha256>
Record plan_sha256 on the change request; once it is approved, rerun:
  gpu-fault-admin deploy --state-dir /secure/gpu-fault --approve-profile-plan <plan_sha256> --reference CHG-<id>
```

续跑命令先审批该计划，写入 `profiles/regional-eks-<digest12>.yaml`，重新 pin 集群条目（version、source、
templateSource、本次发布的 `agentConfigDigest`），随发布注册该版本并继续；节点 wave 随后重新 pin 该集群的 Agent。
`eks-v1` 的快照原样保留。变化之后再加入的集群会推导出同一后继版本并复用其已审批快照。不要为了"腾出"版本名而删除
已 pin 的快照：快照就是该版本下被批准策略的证据。

### 1.2 多个集群需要同一版本

计划文件全站点只有一份，所以集群自有 Profile 的计划按**版本**出：一份 `CLUSTER_PROFILE_NEW` 计划列出本次操作中需要该版本的
全部集群——一次 `join-cluster` 的多个 `--gpu-cluster-arn`、`deploy --gpu-cluster-arn` 新增的 GPU 集群（两者都走批量 join），
或 `deploy` 要迁到同一后继版本的全部成员。`registration_cluster_id` 与快照锚点取其中 cluster id 最小者，无论哪条命令、按什么
顺序列出这些集群，算出的都是同一份计划，deploy 的早期 preflight 与站点收尾也算出同一份，一次审批覆盖全部。此时停止信息打印
`clusters: <id>, <id>` 与 `(these clusters only)`，`changes` 每个集群一行，末行说明该版本是这些集群共用的一次审批与一个不可变快照。

批量 join 与单集群 join 审核方式完全一致：在任何变更之前停止，计划覆盖的集群在各自 join 记录里记下这次停止，同批其余集群保持原状；
按打印的续跑命令（`gpu-fault-admin join-cluster`，同一批 `--gpu-cluster-arn`，加 `--approve-profile-plan <plan_sha256> --reference <CHG>`）继续。已纳管却没有自有
Profile、也没有已审批快照的集群（批量 join 计算集群 Profile 之前完成接入的普通 EKS 集群）由 `deploy` 以同样方式为其首个自有版本出计划
（对照站点 Profile 审核），审批后写入快照并 pin 条目；已 pin 版本的快照缺失仍然拒绝。

**新增集群的接入只能用打印的 `join-cluster` 命令续跑**，由 `deploy --gpu-cluster-arn` 发起的接入也一样，不能改用
`deploy --approve-profile-plan`（已知限制，执行顺序的修复随后续版本发布）：同时执行应用 release 的 deploy 先按站点 Profile
核对待审批记录、后接入新集群，这份 `CLUSTER_PROFILE_NEW` 审批在 join 之前就被判为过期并归档为 `SUPERSEDED`，join 随后再次停下、
写出同一份计划；而已归档的 `plan_sha256` 不能再次审批（`archived Profile approval differs from the active approval`）。不发布
应用 release 的 deploy 会直接接入并消费审批，但事先无法确定一次 deploy 是否发布。已经这样续跑过的按第 7 节处理。

审批只覆盖它所审的那份计划：计划在被消费前变化（例如需要该版本的集群集合变了），旧审批归档为 `SUPERSEDED`，写出新计划并停止，打印新摘要。
控制面按版本只保存一份 Profile，记录首次注册它的集群（`cluster_id`）；release 比较已注册的 Profile 时不比较这个注册锚点，所以 join 追加、
批量 join 与提交按 cluster id 排序、`deploy` 按 EKS ARN 重排、`remove-cluster` 移走锚点集群，都不会被当成策略漂移；`join-cluster` 的
release 把新版本注册在正在加入的集群名下。已存的 Profile 原样读取，不会被改写。

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
| `registration_cluster_id` | 站点当前的注册集群（`spec.runtimeProfile.registrationClusterId`），新版本以它注册；`remove-cluster` 移除它时会改为剩余的同 provider 成员。已注册版本与其不可变快照保留首次注册时的锚点，不要求仍属于当前GPU集合 |
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
| 新增集群的接入用 `deploy --approve-profile-plan` 续跑后审批归档为 `SUPERSEDED`，join 再次停下并打印同一摘要，再次审批被拒（`archived Profile approval differs from the active approval`） | 已知限制（1.2）：同一计划不能再次审批，没有命令能续跑这次接入；不要编辑、移动或删除 `profile-approvals/<plan_sha256>/` 与活动记录，保留 state-dir 交开发团队处理。新增集群一律用停止信息打印的 `join-cluster ... --approve-profile-plan` 续跑 |
| 集群自有版本上出现 `approved Runtime Profile snapshot ... has a different policy than ...` | 模板策略在已 pin 版本下变化；重跑 `deploy`——它会规划后继版本 `regional-<provider>-<digest12>`（第 1.1 节）并打印续跑命令。不要删除已 pin 的快照 |

## 8. 禁止操作

- 不手工创建`profile-approvals/<plan_sha256>/`目录。
- 不手工复制、移动或编辑`plan.json`、`approval.json`、`consumed.json`。
- 不删除活动审批、lock或失败release state来强制重跑。
- 不通过环境变量、隐藏参数或直接调用内部`release-deploy`注入审批引用。
- 不在同一维护窗口并行运行两个针对同一`state-dir`的deploy命令。
- 不把`plan_sha256`当作Secret；它是审计身份，不是凭据。
