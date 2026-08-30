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

多个GPU集群重复传`--gpu-cluster-arn`。操作者不提供release-ref、artifact、site、
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
  -> bootstrap -> deploy -> verify -> stability

state中已有站点
  -> 验证CPU/GPU规范化身份集合完全一致
  -> 扫描并固定当前源码身份
  -> 验签或构建新的签名release
  -> 计算NOOP/CONTROL_PLANE_ONLY/DATA_PLANE_COMPATIBLE/FULL
  -> upgrade -> verify -> stability
```

GPU集合变化不会由`deploy`隐式接受。新增或移除集群必须使用受检
`join-cluster/remove-cluster`流程。

## 7. Dirty和Clean两个发布等级

| 当前源码 | 内部门禁 | 发布等级 |
|---|---|---|
| dirty工作区 | public scan、影响测试、区域影响计划；不确定时升级全量 | `staging_only=true` |
| clean commit | 完整`make check`、PostgreSQL stress和制品一致性 | production |

dirty工作区会被复制到隔离的本地临时commit，当前分支、index和工作区不被修改。
staging attestation绑定影响测试基线。普通生产验签默认拒绝staging-only release。

release-ref由内部根据Git commit或dirty快照计算，不是公共参数。

## 8. 自动复用

同一命令只在身份确实变化时重做工作：

| 对象 | 允许复用的条件 |
|---|---|
| 源码快照 | HEAD、tracked diff和未跟踪源码摘要完全一致 |
| deploy-host bundle和venv | commit、平台、摘要和签名一致 |
| Runtime Image | 完整image input digest一致，且ECR digest仍存在 |
| 签名release | commit、发布等级、影响基线、签名和runtime repository一致 |
| AWS资源 | site ownership、ARN、标签、配置和生命周期策略一致 |
| Kubernetes发布 | release diff明确分类为NOOP或有限升级 |

任何签名、commit、平台、摘要、集群身份或资源状态不确定时均fail closed。

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

工作区变为clean后，再重复同一四参数命令。此时内部自动切换到完整production门禁，不会
复用此前的staging-only attestation。

真实Runtime Profile变化仍必须经过独立审批。普通四参数接口不把审批引用、profile plan
或内部release参数暴露给日常操作者；缺少批准时流程停止，不会降低门禁。

## 10. 成功和失败

命令成功返回前已经完成部署、verify、稳定窗口和release summary。操作者不需要再提供
内部site路径执行第二条部署命令。

失败时：

1. 保留`STATE_DIR`；
2. 修复源码、权限、AWS资源或外部依赖；
3. 使用完全相同的四参数命令重跑；
4. 不删除bootstrap/release state，不手工修改artifact或在线编辑Deployment。

## 11. 生产职责分离

dirty候选只能用于隔离staging。正式生产推荐由GitHub Release CI为最终clean commit生成
签名制品；部署自动化把批准制品放入受信环境后，管理员仍执行同一四参数
`gpu-fault-admin deploy`。artifact和site路径由内部解析，不进入公共命令。
