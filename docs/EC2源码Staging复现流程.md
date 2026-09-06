# EC2源码统一部署流程

本文描述在一台受控CPU EC2上，从源码完成首次部署、dirty代码测试、失败续跑和后续升级。
四种场景使用完全相同的公共命令：

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault-staging \
  --admin-email <operations-email>
```

多个GPU集群重复传`--gpu-cluster-arn`。首集群形成基线后，其余集群通过最多4路的
独立batch join接入。操作者不提供release-ref、artifact、site、
release-build、release-deploy、bundle或venv路径。

## 1. 前提

- 已有至少3个Ready节点的CPU EKS或HyperPod EKS集群。
- 已有至少一个GPU HyperPod EKS集群，且`NodeRecovery=None`。
- GPU VPC已有NAT出口。
- EC2执行身份具有staging bootstrap所需的AWS、EKS和Kubernetes权限。
- EC2已安装Python 3.12、Git、Make、Docker/Buildx、AWS CLI、Cosign、kubectl、Helm、
  curl、jq、OpenSSL和sha256sum。
- 已按[CI 发布流程](CI发布流程.md)的bundle规则安装并激活`gpu-fault-admin`。
- CPU/GPU集群ARN属于同一账号和Region。

state目录必须位于Git仓库外。本流程不执行GPU reset、节点reboot、warm-spare切换或
fault injection。

## 2. 克隆和初始化

```bash
git clone https://github.com/yuhuiaws/GPU-Fault-Recovery-System.git
cd GPU-Fault-Recovery-System

git switch main
git pull --ff-only origin main
git status --short

make deploy-host-setup-online
. .venv/bin/activate
```

首次部署应从仓库根目录执行。成功后，内部状态会记录受信源码根目录，后续不要求操作者
提供repo路径。

## 3. 检查系统工具和身份

```bash
python3.12 --version
docker info
docker buildx version
aws --version
cosign version
kubectl version --client
helm version
aws sts get-caller-identity
```

任一命令失败时先修复EC2镜像、工具安装或AWS临时身份。

## 4. 设置公共输入

```bash
export CPU_CLUSTER_ARN='<CPU EKS或HyperPod ARN>'
export GPU_CLUSTER_ARN='<GPU EKS或HyperPod ARN>'
export STATE_DIR=/secure/gpu-fault-staging
export ADMIN_EMAIL='<运维邮箱>'
```

## 5. 首次和后续部署

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn "${CPU_CLUSTER_ARN}" \
  --gpu-cluster-arn "${GPU_CLUSTER_ARN}" \
  --state-dir "${STATE_DIR}" \
  --admin-email "${ADMIN_EMAIL}"
```

相同命令适用于：

1. state中不存在站点：首次部署；
2. state中已有站点：后续升级或NOOP验证；
3. 上一次执行中途失败：幂等续跑；
4. 开发人员修改了当前checkout：构建并部署新的源码候选。

## 6. 内部状态机

```text
state中不存在站点
  -> 扫描并固定当前源码身份
  -> 发现CPU/GPU集群
  -> 创建runtime/cache ECR
  -> 构建、签名并验签release
  -> 创建Aurora、NLB、DNS/PKI、监控和IAM
  -> 内部生成site
  -> 原子写完整registry -> CPU/endpoint -> 最多4路GPU bootstrap
  -> verify -> stability

state中已有站点
  -> 首次多GPU接入未完成时，验证原目标未变且当前成员是其单调子集
  -> 首次接入完成后，验证CPU/GPU规范化身份集合完全一致
  -> 扫描并固定当前源码身份
  -> 应用身份未变:
       -> deploy-host变化: 更新deploy-host并执行影响门禁和只读preflight
       -> 仅测试/文档变化: 只执行影响门禁
       -> 完全未变: 仅在site、live release、commit状态和NOOP分类均未漂移时复用授权
  -> 应用身份变化: 验签或构建新的签名release
  -> 计算NOOP/CONTROL_PLANE_ONLY/DATA_PLANE_COMPATIBLE/FULL
  -> upgrade -> verify -> stability
```

GPU集合变化不会由`deploy`隐式接受。新增或移除集群必须使用受检
`join-cluster/remove-cluster`流程。

## 7. Dirty和Clean两个发布等级

| 当前源码 | 内部门禁 | 发布等级 |
|---|---|---|
| dirty工作区 | public scan、影响测试、区域影响计划；不确定时升级全量 | `staging_only=true` |
| clean commit且`HEAD != origin/main` | 本地完整production门禁 | production |
| clean commit且`HEAD == origin/main` | 优先验签同commit main CI候选；不可用时回退本地完整门禁 | production |

clean与dirty源码都会先复制到`state-dir/source-snapshots/<fingerprint>/`下的隔离
checkout，后续build、签名、site和config-only发布只引用该checkout，不再引用可变的
开发工作区。clean源码保持原commit并生成production release；dirty工作区额外转换为
隔离的本地临时commit，当前分支、index和工作区不被修改，并生成staging-only release。
staging attestation绑定影响测试基线。普通生产验签默认拒绝staging-only release。

release-ref由内部根据Git commit或dirty快照计算，不是公共参数。

## 8. 自动复用

同一命令只在身份确实变化时重做工作：

| 对象 | 允许复用的条件 |
|---|---|
| 源码快照 | HEAD、tracked diff、未跟踪源码、文件权限和准备后tree摘要完全一致 |
| deploy-host bundle和venv | deploy-host依赖闭包、两个lock、Python ABI、OS/架构/libc、工具清单和签名一致 |
| deploy-host wheelhouse | `build.lock`、`deploy-host.lock`、平台和Python ABI一致 |
| deploy-host依赖层 | 两个lock和平台计算的dependency identity一致且健康 |
| source-only组件制品 | component build identity一致；delivery Manifest重新生成 |
| Runtime Image | 完整image input digest一致，且ECR digest仍存在 |
| 签名release | commit、发布等级、影响基线、签名和runtime repository一致 |
| AWS资源 | site ownership、ARN、标签、配置和生命周期策略一致 |
| Kubernetes发布 | release diff明确分类为NOOP或有限升级 |

deploy-host archive保存在
`STATE_DIR/deploy-host/by-content/<payload-sha256>/`，不再因无关应用commit变化重建。
Git commit、源码fingerprint、应用/deploy-host身份和成功模式写入单独的本地Cosign签名
`source-deploy-success.json`授权记录；记录缺失、签名不匹配或应用身份变化时不得走
deploy-host-only快速路径。`UNCHANGED`还要求签名记录中的site摘要、live release ID、
`phase=complete`、`transaction_committed=true`和release-state摘要与当前只读status
完全一致，且`next_deploy.kind=NOOP`；任一漂移都回到受检application release。

clean源码只有同时满足以下条件时才访问GitHub Actions：

1. 工作区干净；
2. `HEAD`与本地`refs/remotes/origin/main`完全相同；
3. 当前隔离checkout中没有已经验签可复用的候选；
4. 配置了权限`0600`的`GPU_FAULT_GITHUB_TOKEN_FILE`，或受控进程环境中的
   `GITHUB_TOKEN`/`GH_TOKEN`。

dirty工作区、个人clean commit或本地`origin/main`未指向当前commit时不查询GitHub，
直接使用本地影响门禁或完整门禁；实现不会自动`git fetch`。下载候选后会在任何AWS/ECR
动作前验证main run、commit/tree/repository、CI gate、unit gate、六个shard签名和完整
artifact清单。没有凭据、没有候选或API临时不可用时回退本地门禁；已经下载的候选若身份
或签名不匹配则fail closed。

本地production回退把静态/契约、普通pytest和PostgreSQL stress放在三个隔离cache目录中，
按CPU容量自适应为1至3路执行；可用
`GPU_FAULT_RELEASE_GATE_PARALLELISM=1..3`进一步收口。全部通过后才构建和检查artifact。
门禁集合与原`make check`加stress相同，不会降低coverage floor或省略PostgreSQL stress。
static分支内部继续把Ruff、mypy、compile、架构、代码契约、安全、部署配置、文档、
YAML和Shell拆成最多10路；`GPU_FAULT_STATIC_GATE_PARALLELISM=1..10`可主动收口。

门禁的mypy/ruff/pytest缓存目录位于`state-dir`下，dirty与clean路径都会复用。每次成功部署后
按保留规则清理`state-dir/source-snapshots/<fingerprint>/`与`state-dir/.deployer-venv.versions/*`：
被`source-deploy.json`、`source-deploy-success.json`引用的快照、当前部署的快照以及最新
`GPU_FAULT_SOURCE_SNAPSHOT_RETAINED`个（默认5，不要低于2）保留；venv版本保留当前激活、state绑定、
`.deployer-venv.previous`所指以及最新`GPU_FAULT_DEPLOY_HOST_VENV_VERSIONS_RETAINED`个（默认3）。
同一次deploy只在站点锁内收集一次部署前live evidence，部署成功后再收集一次作为成功记录。

任何签名、commit、平台、摘要、集群身份或资源状态不确定时均fail closed。
snapshot中的content-addressed release Manifest与签名材料会随site持续保留；
开发checkout后续运行测试或生成新的`dist/`不会改变已部署站点的
`status/verify/config`输入。

dirty staging会先生成一份绑定`BASE`、changed files和计划SHA的
`dist/staging-impact-plan.json`。静态检查、pytest、regional影响清单和attestation
共同消费该文件；读取时仍重新核对当前changed files，计划漂移会fail closed。只有计划
标记`postgres=true`时才启动临时PostgreSQL 16，普通非数据库修改不再为满足空参数检查而
启动容器。

组件缓存位于`STATE_DIR/component-artifacts/`，不属于开发checkout的`dist/`。它只复用
三个wheel和Node installer bundle的物理文件；新的隔离snapshot仍重算delivery identity
并生成自己的Manifest。缓存identity覆盖组件源码和Node bundle脚本/unit等输入，任何
不匹配或文件损坏都会自动回退重建。

统一源码部署安装的受信 deploy-host venv 会以`0600`文件绑定其所属
`state-dir`。此后该安装中的`gpu-fault-admin`若收到其他`--state-dir`，会在源码扫描、
AWS查询或Kubernetes访问之前直接拒绝。需要管理另一站点时必须使用由该站点自身部署
流程安装的deploy-host，不能复用其他state目录的CLI。

deploy-host bundle中的第三方依赖按lock和平台安装到
`STATE_DIR`相邻的共享dependency venv。项目wheel更新但依赖identity不变时只重建轻量
overlay venv，不再重复安装全部第三方wheel；依赖层校验失败时不会激活新venv。

## 9. 修改代码后的循环

已知相关测试时，可先执行一个最小测试获得快速反馈：

```bash
.venv/bin/python -m pytest -q tests/<known-related-test>.py
```

然后直接重复第5节的四参数命令。测试或部署失败时继续修改源码，再次重复同一命令。

staging测试全部通过后再提交正式commit：

```bash
git diff --check
git status --short
git add <本次确认的文件>
git commit -m "<change message>"
git status --short
```

工作区变为clean后，再重复同一四参数命令。此时内部切换到production等级，不会复用此前
的staging-only attestation；个人commit执行本地production门禁，只有
`HEAD == origin/main`时才尝试消费同commit签名main CI候选。

真实Runtime Profile变化仍必须经过独立审批。首次四参数命令会写出
`STATE_DIR/release-deploy/profile-plan.json`并停止；审核计划和变更单后执行：

```bash
gpu-fault-admin approve-profile \
  --state-dir "${STATE_DIR}" \
  --plan-sha256 "$(jq -er '.plan_sha256' \
    "${STATE_DIR}/release-deploy/profile-plan.json")" \
  --reference CHG-12345
```

然后重新执行原四参数`gpu-fault-admin deploy`。审批记录绑定具体计划摘要和live
Profile baseline，不通过隐藏deploy参数注入。发布失败且计划未变化时保留审批用于续跑；
模板或live baseline漂移时旧审批归档失效；verify、stability和commit全部成功后才消费
审批并删除活动记录。

## 10. 成功和失败

应用release发生变化时，命令成功返回前已经完成部署、verify、稳定窗口和release
summary。仅deploy-host变化时只更新部署机环境、执行影响门禁和只读preflight，不生成
应用release、不访问GPU rollout也不等待稳定窗口；仅测试/文档变化或源码完全未变时记录
NOOP，不执行AWS/Kubernetes mutation。操作者不需要再提供内部site路径执行第二条部署命令。

失败时：

1. 保留`STATE_DIR`；
2. 修复源码、权限、AWS资源或外部依赖；
3. 使用完全相同的四参数命令重跑；
4. 不删除bootstrap/release state，不手工修改artifact或在线编辑Deployment。

## 11. 生产职责分离

dirty候选只能用于隔离staging。正式生产推荐由GitHub Release CI为最终clean commit生成
签名制品；部署自动化把批准制品放入受信环境后，管理员仍执行同一四参数
`gpu-fault-admin deploy`。artifact和site路径由内部解析，不进入公共命令。
