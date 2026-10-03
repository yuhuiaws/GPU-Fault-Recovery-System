# CI 发布流程

本文描述当前 GitHub Actions `CI` 与 `Release` workflow 从源码提交、质量门禁、
受信候选到签名发布制品的完整过程，面向 Release 工程师、维护发布自动化的开发者，
以及负责下载和验签制品的管理员。

本文只描述 CI 构建与发布制品。实际 AWS/Kubernetes 部署继续使用
[管理员快速部署](管理员快速部署.md)、[管理员日常运维](管理员日常运维.md)和
[开发者部署实现](开发者部署实现.md)定义的受检入口。
开发者修改代码后需要一条可直接执行的测试与上线主路径时，使用
[EC2源码统一部署流程](EC2源码Staging复现流程.md)。
单EC2上的dirty源码验证使用
[EC2源码Staging统一部署流程](EC2源码Staging复现流程.md)中的四参数
`gpu-fault-admin deploy`，不属于Release CI。

当前流程的实现事实源是：

- `.github/workflows/ci.yml`：并行质量门禁、main候选、CI gate和候选签名；
- `.github/workflows/release.yml`：候选解析、验签、ECR晋级和最终artifact上传；
- `Makefile` 中的 `release-build-promoted`、`deploy-host-sign`，以及受信本地环境使用的
  `release-build`、`deploy-host-bundle`；
- `scripts/ci_coverage_gate.py`、`scripts/ci_gate_artifacts.py`、
  `scripts/ci_unit_gate.py`、`scripts/ci_gate.py`、`scripts/resolve_ci_run.py`：
  内容寻址coverage shard、聚合unit门禁、main CI候选身份和Release run选择；
- `scripts/build-release-runtime-image.py`、`scripts/build-release-artifacts.py`、
  `scripts/build-release-attestation.py`：独立运行镜像、Manifest v4 和 attestation；
- `scripts/build-deploy-host-bundle.py`、`scripts/deploy_host_bundle.py`：部署机离线包。

修改上述入口时必须同步本文，不得只修改 workflow 或命令而保留旧流程说明。

## 1. 流程边界

当前主链路分成main CI和Release晋级两段：

```text
main push
  -> static、artifact、5个非PG coverage shard、1个PG shard并行
  -> 每个coverage shard:
       -> 计算域内容身份
       -> 命中历史成功main CI的同身份签名gate：验签并复用证据
       -> 未命中：执行本域pytest、branch coverage和duration采集
       -> 生成当前run gate并独立签名
  -> unit: 验签全部十二个物理shard
       -> coverage combine + 生产78% floor + per-module floor + 两范围95%语句/分支
       -> 合并pytest结果、生成fault report和duration汇总
       -> 生成并签名聚合unit gate
  -> artifact: source-only canonical组件制品 + 未签名deploy-host bundle
  -> test: 验证unit域gate，汇总本次static/artifact，生成并签名commit CI gate
  -> 上传 gpu-fault-ci-candidate

workflow_dispatch 或 v* tag
  -> checkout目标commit
  -> 解析该commit对应的成功main CI run
  -> 下载候选并验证CI gate签名、源码身份和完整文件清单
  -> 通过GitHub OIDC获取AWS临时身份并登录ECR
  -> make release-build-promoted
       -> 再次验证CI gate
       -> 用候选组件制品构建或复用不可变Runtime Image
       -> 生成deployable Manifest v4和绑定CI gate的attestation
       -> 运行最终制品一致性检查并签名attestation
  -> make deploy-host-sign
       -> 只签名CI已经构建的deploy-host archive，不重复构建
  -> 上传 dist/ 为 gpu-fault-release artifact
```

PR也运行相同的十二个fresh shard、统一coverage floor、PostgreSQL stress、`static`和
`artifact`，但shard不作main信任签名、不构建deploy-host bundle，也不生成可供Release
跨run下载的签名候选。最终job仍名为`test`，用于保持分支保护的单一聚合状态。

Release workflow只接受checkout得到的clean commit，并要求其候选来自
`refs/heads/main`上的成功push CI。它生成：

- Manifest中的`staging_only=false`；
- attestation中的`release_tier=production`；
- 绑定签名CI gate的晋级门禁记录。

`release-build-staging`只供统一CLI的内部源码准备器处理dirty隔离快照。该等级执行影响
选择并在不确定时升级全量门禁，但其Manifest固定为`staging_only=true`，普通生产验签
默认拒绝；GitHub Release workflow不会构建或上传该等级。

该 workflow：

- 会查询、构建和推送 ECR Runtime Image，并可读写独立 BuildKit cache repository；
- 不执行 `make release-deploy`；
- 不创建或修改 EKS、HyperPod、Aurora、NLB、Route53 或 Kubernetes 资源；
- 不持有 CPU/GPU kubeconfig，不接触站点 token、Node Action key 或数据库生产密码；
- 不把 GitHub Actions artifact 自动复制到 `/secure/release/`。

源码ARN部署在干净`HEAD == refs/remotes/origin/main`时也可自动下载同commit候选：
它先验证main run、CI/unit/十二个shard签名、commit/tree/repository和artifact清单，再进入
`release-build-promoted`。dirty源码和个人clean commit不查询GitHub。候选不可用时，
本地`release-build`先执行static/契约，再按CPU预算并行执行普通pytest、PostgreSQL
stress与source-only artifact构建；全部通过后才构建运行镜像、生成可部署Manifest并签名。
任一门禁失败会停止同组其他任务，已生成的source-only制品不能部署。这是受信本地复现路径，
不等同于伪造main CI结论。
static分支内部再并行Ruff、mypy、compile、架构、文档、部署配置、YAML、Shell、安全和
代码契约；默认`make check`在static后并行artifact与普通pytest。
quality gate的子进程不继承外层部署的deadline、API预算或shim PATH，测试因此使用自己的
时钟和fake工具。构建仍受外层进程监督和时间上限约束；该隔离不移除后续真实发布的预算。
本地PostgreSQL fallback使用受监督、精确CID绑定的私有授权协议；仅PG测试子进程更换
HOME/PGPASSFILE，其他构建和签名步骤不变。凭据与授权位于仓库外的
`<state-dir>/release-postgres`，不进普通artifact。创建或删除结果无法确认时，构建失败
并保留证据，已有签名制品也不能绕过未完成清理。该本地fallback不是Actions service的
生命周期所有者，后者仍由CI job管理。
有效本机allocation授权下，原`test-postgres-stress`门禁委托独立实例分片，
`POSTGRES_TEST_WORKERS`默认按核数四分之一取4至16（与`PYTEST_XDIST_WORKERS`同一推导）、允许1至16；每份拥有独立PG16、私有授权和串行`-n 0`
pytest进程。完整发现清单、互不重复且无遗漏的执行并集、成功阶段与零skip全部通过
才接受门禁，stress的8个竞争worker与40轮参数不变。任一分片失败时只请求其他分片停止
pytest，等待各自的受管清理结束，不同时终止清理监督进程。dirty影响计划要求完整门禁时
同样复用static之后的普通测试、PostgreSQL、artifact并行组；CI service、coverage追加
及自定义外部测试连接仍走原串行路径，本机报告不能充当CI shard回执。
本机分片的报告路径和分区选择绑定在当前pytest会话内，不改写报告器的公开环境变量常量，
也不把外层报告或分区设置传给嵌套测试。嵌套runner仍独立准备并校验自己的完整回执，
不能覆盖外层报告或继承外层分区而少跑测试。

可选 HMA CloudWatch 转发退役后，不再交付 Lambda/CloudFormation 模板，专用
cfn-lint 工具安装与 CI/static 目标随之删除。YAML、惰性导出、pip-audit、SBOM、
promtool 以及制品一致性检查仍执行；删除 collector 不能留下未解析的 Python 导出或
可重新部署的旧清单。

`promtool-check` 除语法检查外，还使用固定版本的真实 promtool 执行告警行为测试，
校验完整的成功 pytest 回执；缺少工具在本地和 CI 中都失败。相关两组测试由始终
重新执行的 static 门禁负责，不作为可复用 coverage shard 的历史证据。

dirty 候选的文件枚举包含未跟踪文件，并排除 Git 明确报告的工作区删除，删除同样改变
内容身份，不能复用原 shard。仍拒绝符号链接、目录、不可读取的 Git 删除清单，以及枚举
后消失且不在删除清单内的输入；不要求为了运行本地门禁而暂存或提交文件。

## 2. 触发、权限和输入

### 2.1 触发条件

`.github/workflows/ci.yml`只响应Pull Request和`main` push。普通feature branch push
不重复运行CI；同一PR或分支的新提交会通过concurrency取消旧run。

`.github/workflows/release.yml` 支持两种触发方式：

| 方式 | 用途 |
|---|---|
| `workflow_dispatch` | 经审批后晋级所选commit；可选指定成功CI run ID |
| 推送 `v*` tag | 晋级标签指向commit对应的成功main CI候选 |

两个workflow都使用`fetch-depth: 0`。Release默认按`git rev-parse HEAD`查询最近的
成功main push CI；显式`ci_run_id`只省略查询，后续源码commit、Git tree、repository和
artifact清单仍必须与当前checkout完全一致。

### 2.2 GitHub 权限

两个workflow只声明：

| 权限 | 用途 |
|---|---|
| `actions: read` | Release跨run下载受信CI候选 |
| `contents: read` | checkout 源码 |
| `id-token: write` | CI gate、release attestation、deploy-host archive的keyless签名，以及Release的AWS OIDC |

GitHub workflow不配置`COSIGN_SIGNING_KEY`，因此使用keyless `cosign sign-blob`。
Release在请求AWS身份之前先把CI gate的certificate identity固定为
`ci.yml@refs/heads/main`并验证OIDC issuer。部署侧同样必须验证批准的Release identity
和issuer，不能只检查文件存在或SHA-256。

### 2.3 Repository variables

Release workflow 消费以下 GitHub repository 或 environment variables：

| 名称 | 必需 | 作用 |
|---|---|---|
| `RELEASE_ROLE_ARN` | 是 | GitHub OIDC 要 assume 的发布角色 |
| `AWS_REGION` | 是 | ECR 登录和镜像操作 Region |
| `RUNTIME_IMAGE_REPOSITORY` | 是 | 不可变 Runtime Image 的 ECR repository URI |
| `RUNTIME_IMAGE_CACHE_REPOSITORY` | 否 | 独立 BuildKit registry cache repository URI |

两个 repository 值都必须匹配 ECR URI。cache repository 只用于加速 layer 构建，
不能作为可信发布制品，也不能替代 Runtime Image digest 校验。

CI测试还支持以下可选repository variables：

| 名称 | 默认 | 作用 |
|---|---|---|
| `CI_TEST_RUNNER` | `ubuntu-latest` | coverage和PostgreSQL shard使用的已配置larger/self-hosted Runner label |
| `CI_PYTEST_WORKERS` | `4` | 非PostgreSQL shard的xdist worker数；应与Runner CPU/IO容量匹配 |

未配置larger runner时不得填写不存在的label。worker数变化会进入shard协议身份；CI固定
使用`--dist=worksteal`减少长尾，但不会以增加worker代替慢测试分析。
整数设置必须与回执中的实际worker数精确相等；空覆盖值、负数、布尔值或非法文本直接
拒绝。`auto`/`logical`保留原请求模式并记录xdist实际解析出的正整数进程数，聚合机器
不重新计算CPU数。请求被worker cap缩小不能冒充原整数预算已执行；PostgreSQL始终`-n 0`。

只有CI的`postgres` shard创建临时PostgreSQL 16 service，并设置只指向该service的
`GPU_FAULT_TEST_POSTGRES_URL`。Release job不再创建PostgreSQL，也不重复执行已经由
签名CI gate证明的测试。该测试连接不是生产数据库连接，不得替换为生产Aurora。
`ci_postgres_grant.py` 将进程测试授权绑定到完整 service container ID、当前
repository/run/attempt/job 归属及仅 loopback 的端口映射。测试 URL 不含密码，
`HOME/.pgpass` 和授权记录位于 checkout、artifact 之外的私有 `RUNNER_TEMP` 目录，
目录权限 0700、文件 0600。证据上传后的 `always()` 清理只撤销该授权文件和环境引用；
service 生命周期仍由 Actions 管理，不按名称接管、重标记或删除容器。

部署机自动下载main CI候选时可使用权限`0600`的
`GPU_FAULT_GITHUB_TOKEN_FILE`，也可继承受控进程中的`GITHUB_TOKEN`或`GH_TOKEN`。
token只用于GitHub Actions只读API，不得进入命令输出、state、release artifact或site。
候选查找不会自动`git fetch`，本地`origin/main`不是当前HEAD时直接跳过。

## 3. CI 执行步骤

### 3.1 并行CI门禁

所有job使用Python 3.12和pip cache，并安装hash固定的`requirements/build.lock`后再
安装项目测试extras。CI的执行单元如下：

| job | 主要职责 |
|---|---|
| `static` | Ruff、mypy strict、compileall、架构、部署契约、文档与CI工具测试、配置、YAML、Shell和artifact安全检查 |
| `coverage-runtime_0..2` | 普通Runtime pytest按稳定nodeid哈希分为三份，分别采集branch coverage和duration证据 |
| `coverage-deployment_0..3` | 发布、区域编排、deploy-host管理员pytest按稳定nodeid哈希分为四份，分别采集对应coverage |
| `coverage-fault_runner` | fault scheduler/runner测试和duration证据 |
| `coverage-postgres_0..3` | PostgreSQL contract pytest按稳定nodeid哈希分为四份；每份在自己的job级PostgreSQL service上串行（`-n 0`）采集coverage，再对同一份执行8 workers × 40 rounds stress |
| `shuffled-order-0..3` | 与`test-parallel-release`相同的测试集按run id种子整体打乱后切成四个互不相交的子序列（`tests/conftest.py`），各自以xdist执行，暴露顺序耦合 |
| `unit` | 验签十二个物理shard、合并coverage、统一floor、fault report、duration汇总和聚合签名 |
| `artifact` | 构建并验证source-only三个组件wheel与Node bundle；main额外构建未签名deploy-host bundle |
| `test` | 聚合static/unit/shuffle/artifact并为main生成commit级CI gate |

`config/ci-unit-gate.json`定义测试分区、内容身份组、coverage协议和不重复进入coverage的
文档/CI测试。四个逻辑域按以下边界执行；`protocol.partitions`（与
`scripts/ci_coverage_config.py`的`PARTITIONED_DOMAINS`一致）把runtime、deployment和
postgres分别拆成三、四、四个物理shard，fault_runner保持一个，共十二个：

| shard | 测试边界 | 内容身份要点 |
|---|---|---|
| `runtime_0..2` | 除静态、部署、fault runner和PostgreSQL入口外的普通测试；每个具体nodeid按SHA-256取模只进入一份 | dependencies、Runtime源码、Runtime测试、共享fixture、分区总数和index |
| `deployment_0..3` | `tests/admin/`、可执行regional测试和release/deploy/artifact根测试；每个具体nodeid按SHA-256取模只进入一份 | dependencies、Runtime共享源码、部署/deploy-host-only源码、部署测试、分区总数和index |
| `fault_runner` | `tests/test_case_scheduler.py`；catalog完整契约仍由static执行 | dependencies、Runtime共享源码、testcases、runner/scheduler和runner测试 |
| `postgres_0..3` | `tests.postgres_files` 的完整显式清单；每个具体nodeid按SHA-256取模只进入一份，contract和stress两遍使用同一份 | dependencies、Runtime共享源码、PostgreSQL测试、实际PostgreSQL image、分区总数和index |

PostgreSQL shard 的入口以 `config/ci-unit-gate.json` 的 `tests.postgres_files` 为准，
并与 Makefile 的串行 PostgreSQL 清单检查一致；新增 native 回归不得只放入普通分片。

它主要覆盖`src/gpu_fault/store/postgres/**`、`store/contracts.py`、
`store/shared/**`、schema/DDL、连接池与重连，以及processor claim、lease、counter和
fencing。由于这些测试会消费共享model、Store contract和Runtime辅助代码，其身份保守
绑定Runtime源码，而不是只摘要`store/postgres/`；deploy-host-only管理员源码不在该
身份中。

每个shard身份还包含Python版本/ABI、Runner image、实际worker协议和已安装distribution；
PostgreSQL shard额外包含容器image ID。main push先按
`gpu-fault-coverage-<shard>-<identity>`查找历史artifact，只接受已完成的main push中
对应shard job成功的artifact；其他job失败不会废弃已经独立签名成功的shard。
命中后：

1. 验证历史shard gate的Cosign workflow identity、producer run和全部证据SHA；
2. 要求producer commit仍是当前commit的祖先；
3. 复用coverage、pytest和duration证据；
4. 生成带`reused_from`的当前run shard gate并重新签名。

未命中的shard只执行自己的pytest。八个非PostgreSQL shard在独立Runner上使用
`--dist=worksteal`；四个PostgreSQL shard各自拥有一个job级`postgres:16` service，
在上面串行执行自己那一份contract和stress。每个pytest命令同时打印`--durations=50`
并写入结构化`durations.json`。因此fresh run的墙钟由最慢shard决定，不再把普通测试、
PostgreSQL contract和stress串在同一job中。

shard的pytest命令行由`scripts/ci_coverage_config.py`的`shard_arguments`生成：能整体
归属本shard（其余shard的测试文件不超过两成）的目录直接作为参数，少数例外用
`--ignore=`排除，其余文件逐个列出。pytest对每个文件参数都会重新收集其所在目录
（`Session.collect`对裸文件路径不走收集缓存），所以把`tests/regional`的 843 个文件
逐个传入会构造约 80 万个 Module 节点，仅收集就比目录参数慢 3 倍（2026-10-03 实测
190 秒对 63 秒）。回执中的`selection.targets`记录的是实际参数，`collected_files`
仍必须精确等于本shard的文件清单，因此参数集合算错会让shard gate失败，而不是跑错
测试；`tests/test_ci_unit_gate.py`对每个shard证明参数集合恰好收集目标文件。

pytest nodeid是“测试文件路径 + 测试类/函数 + 参数化case ID”，例如：

```text
tests/collectors/test_xid_kmsg_catalog_replay.py::test_xid_kmsg_catalog_replay_b200[GF-XID-KMSG-B200-143]
```

它不是GPU或Kubernetes节点身份。按nodeid而不是按文件拆分，可以把同一文件中的数百个
参数化case分散到同一域的多个Runner；分区算法（`partition_for_nodeid`：nodeid的
SHA-256取模）确定、互斥且并集完整，`tests/test_ci_unit_gate.py`对每个分区域证明这
三点。同一域的各shard共享内容身份组，只在`partition_count`/`partition_index`上不同；
测试函数或参数ID变化会改变整个域的身份，使该域全部shard同时失效重跑。unit job合并时
还要求同一域的各shard报告完全相同的discovery，否则拒绝合并。

runtime、fault runner和PostgreSQL coverage显式排除deploy-host-only模块；
deployment shard采集完整应用源码覆盖。这样只修改独立管理员CLI代码时，旧Runtime
coverage不会携带变化模块的陈旧行号，只有deployment shard失效。最终`unit` job再次
核对十二个shard属于当前run、验签并执行`coverage combine`，只在生产范围达到78%组合
floor、各模块门禁通过，并且 production/runner 各自语句和分支均达到95%后继续。
所有分片测量 runner 的跨域调用，因此 runner 源码变化保守失效全部分片；纯
deploy-host 修改仍不失效 Runtime 分片。

per-module floor只在合并报告上执行：deployment-only模块被每个runtime shard排除，
单个shard的数据无法说明它们真实覆盖了多少。`config/ci-unit-gate.json`的
`coverage.module_floors`为每个族声明`group_floor`和`file_floor`，前者阻止整族被
仓库其余部分抬起来，后者阻止族内一个覆盖良好的模块替兄弟模块背书。没有可度量点的
空模块不计算单文件比例，整组没有可度量点仍失败。本地
`make coverage`用`ci_coverage_gate.py module-floors`执行同一检查。

计数按 `covered_lines + covered_branches` 计算，而不是从总量减去
`num_partial_branches`。后者只描述部分覆盖的分支行，遗漏整段未执行的分支；
缺失、非整数、负数及互相矛盾的统计均拒绝。95% 门禁独立检查两个范围的语句与分支，
参见 [覆盖率与场景矩阵](components/scenario-coverage.md)，不会缩小现有源码范围。

十二份 pytest 结果校验 discovery、选择清单、完整执行阶段及分片内容身份后合并；
数值worker协议还要匹配实际进程数及已记录的请求数；自动模式回执必须显式绑定请求
模式。缺失自动模式证明或计数类型错误均不复用，旧数值回执也不能绕过实际数量校验。
PostgreSQL stress 也必须提供完整且没有 skip 的回执。原回执的源码身份、session 和
首次执行 producer 均保留。schema-2 聚合报告以 `validated_source_identity` 记录当前
验证对象，不把历史执行改写为当前源码的新 session。分片 gate 使用 schema 2，聚合
unit gate 使用 schema 3，旧格式不得绕过新增门禁。
`make fault-test-cases-ci`直接把这些结果映射到64条unit/component fault case，不再次
启动pytest。文档/CI契约测试由当前commit的`static` job执行，因此docs或`.github/`
变化可以复用全部shard，但不能跳过当前static和artifact门禁。

本地 pytest reporter 额外记录开始/结束源码身份、时间、退出状态及实际 collection。
fault case 的 PASS 必须具备 setup/call/teardown 三个成功阶段；带 session 的报告若执行
失败或源码中途变化，不允许复用。场景验证对本地报告要求完整 session、七天新鲜度和
精确参数化数量；旧的签名 shard 合并报告不自动变成本地场景证明，仍由 CI 签名身份协议
保障其原有用途。需求文件、场景工具及其测试进入 fault-runner 内容身份；跨代码/文档的
需求引用完整性由当前 static 的 catalog 契约测试检查。

#### 3.1.1 Fresh实测基线

以下数据来自默认`ubuntu-latest`、4个xdist worker，未配置larger runner；用于回归比较，
不是固定SLA：

| main commit / run | 流程 | 端到端 | 最长测试路径 | 聚合 |
|---|---|---:|---:|---:|
| `863a557` / `33615185401` | 旧单体unit | 18分44秒 | unit 18分19秒；其中普通coverage 14分44秒 | 包含在同一unit |
| `a6dd742` / `33620340339` | 六个签名shard | 6分18秒 | `runtime_2` 4分53秒 | unit 57秒 |
| `f90fdad` / `33622247519` | 稳定修复后fresh | 6分10秒 | `runtime_1` 4分41秒 | unit 59秒 |
| `adcaadff` / `37124211673` | 六个shard + 单job打乱轮（测试集增长后） | 39分14秒 | `shuffled-order` 39分；`coverage-deployment` 36分；`coverage-postgres` 30分 | unit未执行 |
| `ci/faster-pipeline` / 见下文 | 十二个shard + 四个打乱shard | 见下文 | 见下文 | 见下文 |

2026-10-03 重新分片前，测试集已经增长到约 4.1 万条 nodeid：单job的打乱轮、单个
deployment shard和串行PostgreSQL shard各自需要 30-40 分钟，三者并行也让整条流水线
停在 40-45 分钟。重新分片只改变每个job拿到的切片，不改变测试集、coverage floor、
签名协议和打乱顺序本身（见 `tests/test_ci_unit_gate.py` 和
`tests/test_test_suite_contracts.py` 的分区证明）。每个job固定开销（checkout、
pip cache 安装、cosign）约 1 分钟，不是瓶颈。

单条用例的长尾不能靠分片摊薄：`tests/test_optional_postgres_collection.py` 的整套
收集探针约 4.5 分钟，`tests/regional/test_clean_redeploy_script.py` 的若干用例 40-55
秒，它们各自所在的shard不会短于这些用例。进一步缩短只能靠更大的Runner
（`CI_TEST_RUNNER` 配合 `CI_PYTEST_WORKERS`）或缩短这些用例本身。

后续比较应同时检查artifact的`test-durations.json`，避免只看总时长而遗漏Runner排队、
依赖安装或单项测试长尾。

`artifact-check`只生成一套canonical Control Plane、Executor、Node Runtime wheel和
Node bundle。source-only Manifest为`deployable=false`，但包含物理SHA、
`module_digest`、bundle内嵌wheel和component build identity。component identity还覆盖
Python/平台、build lock、组件源码以及全部Node bundle输入，不能用旧bundle搭配新源码。

deploy-host bundle在main CI artifact job中构建一次。`actions/cache`按两个lock、
Runner OS/architecture和Python 3.12保存wheelhouse；archive此时未签名，留给Release
workflow在候选验签成功后签名。

### 3.2 生成受信CI候选

聚合job `test`依赖三个job并逐一要求`success`。main push时它：

1. 下载artifact job生成的完整`dist/`；
2. 下载unit job本次生成的聚合签名gate，再次验签，并核对其内十二个shard gate、
   签名bundle、内容身份和证据；
3. 要求工作树干净，读取source-only schema v3 Manifest；
4. 记录Git commit、Git tree、repository、main CI workflow ref、run ID；
5. 对`dist/`中除CI gate自身外的每个文件记录相对路径、权限、大小和SHA-256；
6. 把unit gate的identity、producer run/commit、SHA和复用shard清单写入
   `domains.unit`；
7. keyless签名`dist/ci-gate.json`；
8. 上传`gpu-fault-ci-candidate`，保留30天。

CI gate只能由`ci.yml@refs/heads/main`生成。PR聚合job只给出分支保护结果，不签名或上传
该候选。

### 3.3 Release先验签后访问云端

Release checkout目标commit后，先解析匹配的成功main CI run并下载
`gpu-fault-ci-candidate`。随后依次：

1. 用精确CI workflow certificate identity和GitHub Actions issuer验证CI gate签名；
2. 验证gate的commit、tree和repository与当前checkout一致；
3. 重新计算候选Manifest SHA和完整文件清单；
4. 再次验证聚合unit gate和十二个coverage shard的独立Cosign签名；
5. 只有全部一致后才获取AWS OIDC身份、登录ECR和安装发布依赖。

因此手工提供其他run ID、tag指向未通过main CI的commit、候选缺文件或跨run混合制品都
会在AWS/ECR动作之前失败。

### 3.4 执行 `make release-build-promoted`

`release-build-promoted`要求`RUNTIME_IMAGE_REPOSITORY`、`CI_GATE`、clean工作树和
Cosign。它不重复`make check`或PostgreSQL stress，而是再次执行`ci_gate.py verify`，
然后继续以下步骤。

#### 3.4.1 构建或复用 Runtime Image

`scripts/build-release-runtime-image.py`先验证`artifact-check`留下的source-only
Manifest、delivery identity、wheel SHA、`module_digest`以及Node bundle内嵌wheel，
再根据以下输入计算规范化 image input digest：

- Dockerfile 和固定 base image digest；
- runtime dependency lock；
- 目标 platform 和 build args；
- 每个运行镜像自己的component wheel SHA-256 与 `module_digest`；
- 构建器定义的其他 Runtime Image 输入。

目标不可变tag不存在时，Release构建并推送镜像；tag已存在时，只有目标platform和全部
`gpu-fault.*` labels 与本次输入完全一致才允许复用。任一 label 不一致都会失败，不能把
已有 tag 当作普通 mutable tag 覆盖。
registry缺失必须来自明确的目标引用或registry错误码；credential helper、认证、TLS、
网络、进程和输出错误立即失败，不触发冷下载或重建，也不回显原始认证helper输出。
签名schema v3共享镜像与v4独立镜像使用同一ECR inventory验证，绑定Region、registry ID、
repository及完整digest集合；成功退出但返回空数据不代表镜像存在。

镜像内容及 Collector 插件验证在无网络、只读的独立容器中执行。验证器在启动命令前
绑定完整 CID、私有 CID 文件、归属标签及镜像身份；正常结束、超时或主线程中断均
清理同一容器及匿名卷，并确认容器不存在。创建或清理结果无法确认时停止门禁，保留
私有归属记录供调和，不能仅凭同名容器或同标签容器执行清理。

成功后生成：

```text
dist/release-runtime-image.json
```

该image-set descriptor使用schema v3，分别记录Control Plane、Executor及节点离线依赖的
OCI digest、平台、构建输入和组件/文件身份。最多3路构建，共用原不可变repository；
只改Control Plane wheel不会重建Executor镜像。每个CPU/Executor镜像只有一个应用包环境：
`/opt/gpu-fault/runtime`在隔离venv内安装全部hash锁定runtime依赖及本组件wheel，
不再把依赖安装到基础Python，也不使用`--system-site-packages`或`.pth`桥接。
镜像默认`PATH`和`VIRTUAL_ENV`指向该环境；原`control-plane`或`executor`目录仅为同一
环境的兼容软链接，不是第二套包。组件参数和wheel位于共同依赖安装层之后，两个镜像
仍可共享BuildKit依赖层。基础镜像的解释器、标准库和打包工具不作为第二套应用环境。
内容门禁实际执行默认`python`、兼容路径、`-I`导入、console入口和`pip check`，拒绝外部
包路径、错误组件或CLI解释器。仅已知旧Dockerfile摘要保留旧布局验证，仍验证组件digest、
插件及隔离边界；不改变descriptor格式、签名或旧release回滚身份。
节点wheelhouse按`node-runtime.lock`和
`node-tools.lock`下载固定hash的Linux amd64/Python 3.12 wheel，包含py-spy，并生成
inventory。wheelhouse不放入Node bundle或ConfigMap；inventory SHA由最终签名Manifest绑定。
BuildKit cache仍使用独立的mutable repository，但各组件分别向带
`-control-plane`、`-executor`、`-node-dependencies`后缀的tag导出；local cache使用同名
子目录。普通import保留旧共享缓存作为兼容读取源，固定digest或带selector的import不改写。
split构建的export只接受可明确划分的registry/local目标，CSV选项歧义、目标重叠和不支持
的export类型在任何构建开始前拒绝。缓存位置不进入可信release身份，也不能替代内容验签。
同一可信部署机可在下载wheel前复用已验证的依赖镜像：本地私有HMAC receipt绑定完整输入，
再对固定OCI digest重新核对registry labels和平台。缓存只保存receipt，不缓存或授权任意
wheel目录；缺失时冷构建，认证失败或身份漂移时拒绝。receipt及其本地认证密钥不随源码、
CI artifact或release交付，不能作为跨部署机的签名release证明。
旧schema v2共享镜像descriptor仍可读取，source-only候选仍为Manifest v3。
输出descriptor前，`verify_release_images`在禁网、只读容器中检查真实module digest、
组件隔离、入口和wheelhouse全部文件；失败不签名、不输出可部署descriptor。

#### 3.4.2 构建最终发布制品

`scripts/build-release-artifacts.py` 使用 Runtime Image descriptor 和已验证的
canonical制品生成最终release：

1. 复用 Control Plane wheel；
2. 复用 Cluster Executor wheel；
3. 复用 Node Runtime wheel；
4. 复用只包含该精确 Node Runtime wheel 的 Node installer bundle。

构建器重新计算source identity、四个物理SHA、三个`module_digest`和bundle内嵌wheel，
并逐项比较 Runtime Image descriptor。任何不一致都不会生成最终release。组件构建使用
`requirements/build.lock`中固定的build frontend/backend和`--no-isolation`，不会为
每个组件重复创建隔离构建环境。

首次生成canonical组件制品时，三个组件的独立构建子进程并行运行；Node bundle仅等待
Node Runtime wheel，不等待另外两个组件。构建输出按distribution匹配，`umask 022`
只作用于各构建子进程，不修改父进程的全局umask。所有构建任务结束并通过检查后才原子发布
Manifest；失败不会覆盖上一次已发布制品。

最终 `release_id` 由四个制品 SHA 和 delivery identity 计算，发布目录采用
`dist/<release-id>/`，并生成 `deployable=true` 的 Manifest v4：

```text
dist/<release-id>/release.json
dist/current-release.json
```

`current-release.json` 是同一 Manifest 的稳定入口，不是另一份可独立修改的事实源。

#### 3.4.3 独立制品一致性测试

Release设置`GPU_FAULT_REQUIRE_BUILD_ARTIFACTS=1`，运行
`tests/test_artifact_consistency.py`，验证源码、三个 wheel、Node bundle 和 Runtime
Image 中组件的实际内容一致。该检查失败时不得只保留或发布已推送的 OCI。

#### 3.4.4 生成并签名 release attestation

`scripts/build-release-attestation.py` 生成：

```text
dist/<release-id>/attestation.json
dist/current-attestation.json
```

attestation 绑定：

- `release_id`；
- `dist/current-release.json` 的 SHA-256；
- delivery identity SHA；
- Git commit 和干净工作树状态；
- `release_tier=production`；
- `dist/ci-gate.json`的SHA-256；
- `ci_gate.py verify`和最终artifact consistency的PASS结论。

main CI中的static、coverage、artifact和PostgreSQL stress结论保存在被绑定的CI gate中，
不伪装成Release job再次执行了`make check`。

随后 `cosign sign-blob` 对 `dist/current-attestation.json` 执行 keyless 签名并生成：

```text
dist/current-attestation.bundle.json
```

部署阶段必须同时校验签名身份、Manifest SHA、release ID 和 delivery identity。

### 3.5 签名deploy-host archive并上传

部署机离线包与 Node installer bundle 是两个独立制品。该步骤不会复用 Node bundle。

CI artifact job中的`scripts/build-deploy-host-bundle.py`：

1. 再次要求干净源码树；
2. 复制 `requirements/build.lock` 和 `requirements/deploy-host.lock`；
3. 从按两个lock、OS、架构和Python ABI缓存的目录补齐完整离线wheelhouse；
4. 在只安装`build.lock`的临时venv中构建独立
   `gpu_fault_deploy_host-*.whl`；
5. 加入部署机工具清单和可选审核后二进制；
6. 记录payload identity、平台、Python ABI、libc 和每个文件的 SHA-256、大小、权限；
7. 生成确定性 tar.gz 和 `.sha256` sidecar。

CI workflow使用`actions/cache`持久化`DEPLOY_HOST_WHEELHOUSE`。缓存miss时仍按
hash lock下载，cache hit时只校验和补齐缺失wheel。部署机安装始终使用 `--no-index`，
不能再次在线解析依赖。

deploy-host wheel以`gpu_fault.admin.cli`为根，安装管理员部署CLI及其Python闭包和
`deploy-host-tools.json`；闭包同时追踪`gpu_fault`与`gpu_fault_release`，包括进程监督、
私有stdio HTTP worker、SNS policy helper、前置刷新事务、Store proof和Job清理模块。
由独立部署脚本导入的workload RBAC检查与清理模块显式列为deploy-host构建根，
不能只依赖管理员CLI的Python导入闭包，否则源码checkout能运行而离线安装缺少模块。
这些helper不进入Control Plane、Executor或Node Runtime业务wheel。
非Python部署输入仍从受信checkout解析：共享
`deploy/observability/amp-sns-publish-policy.json`进入Manifest输入与observability组件
身份，数据库proof复用的`deploy/migrations/postgres-schema-preflight-job.yaml`也进入
Manifest输入。`scripts/deploy_source_identity.py`另以部署编排身份绑定release包源码和
`wait-for-kubernetes-job.sh`，不能只更新helper而复用未绑定的源码快照。
Control Plane Runtime wheel不再包含`gpu_fault.admin.cli`、
`gpu-fault-admin`入口或deploy-host工具清单。管理员代码变化因此只重建deploy-host
bundle；只有helper变化且Manifest、renderer、节点及业务组件输入不变时，才不改变
Runtime Image或应用release diff。共享asset变化仍进入其所属发布组件。

archive schema v2不再把Git commit写进payload。其内容身份覆盖deploy-host Python依赖
闭包、两个lock、构建器、管理员配置模板、工具清单、Python ABI、OS/架构/libc和可选工具
目录；相同payload跨应用commit生成同一archive。commit/tree/repository授权由独立签名
CI gate或部署机本地签名`source-deploy-success.json`记录承担。setup仍重新计算当前
checkout的payload身份，不能只凭缓存路径接受archive。

默认 Ubuntu x86-64、CPython 3.12 Runner 生成：

```text
dist/gpu-fault-deploy-host-linux-x86-64-cpython-312.tar.gz
dist/gpu-fault-deploy-host-linux-x86-64-cpython-312.tar.gz.sha256
```

Release验证完整候选后只运行`make deploy-host-sign`，对上述原始archive执行Cosign
keyless签名，不重新构建项目wheel或wheelhouse：

```text
dist/gpu-fault-deploy-host-linux-x86-64-cpython-312.sigstore.json
```

文件名由构建 Runner 的 OS、CPU architecture 和 Python cache tag 计算。其他平台必须在
对应受信环境重新构建和签名，不能手工改名。

所有步骤成功后，`actions/upload-artifact`将整个`dist/`上传为：

```text
artifact name: gpu-fault-release
retention: 30 days
```

`if-no-files-found: error` 保证空目录不会形成成功发布。该对象是 GitHub Actions
artifact，不是自动创建的 GitHub Release asset，也不会自动复制到部署机。

## 4. 发布制品清单

| 位置 | 生产者 | 用途 |
|---|---|---|
| `dist/ci-domains/unit/unit-gate.json` | main CI `unit` job | 聚合十二个当前run shard、统一coverage/fault/duration证据 |
| `dist/ci-domains/unit/unit-gate.bundle.json` | main CI `cosign sign-blob` | unit域gate的Sigstore签名 |
| `dist/ci-domains/unit/shards/<shard>/coverage-shard-gate.json` | 对应coverage job | 绑定单个shard内容身份、producer和coverage/pytest/duration证据 |
| 同目录`coverage-shard-gate.bundle.json` | 对应coverage job的`cosign sign-blob` | 单个shard的独立Sigstore签名 |
| `dist/ci-gate.json` | main CI `test` job | 绑定当前源码、source-only候选、unit域gate及组合质量门禁 |
| `dist/ci-gate.bundle.json` | main CI `cosign sign-blob` | CI gate的Sigstore签名材料 |
| ECR 中的不可变 Runtime Image digest | `build-release-runtime-image.py` | CPU Control Plane 和 GPU Executor 运行镜像 |
| `dist/release-runtime-image.json` | 同上 | 绑定 OCI digest、平台、输入和镜像内组件 |
| `dist/current-release.json` | `build-release-artifacts.py` | 当前 deployable Manifest v4 稳定入口 |
| `dist/<release-id>/release.json` | 同上 | 内容寻址 release Manifest |
| `dist/<release-id>/gpu_fault_control_plane-*.whl` | 同上 | CPU Control Plane |
| `dist/<release-id>/gpu_fault_cluster_executor-*.whl` | 同上 | GPU EKS Executor、Watcher、Collector、Reconciler |
| `dist/<release-id>/gpu_fault_node_runtime-*.whl` | 同上 | GPU 节点 Agent/Collector |
| `dist/<release-id>/gpu-fault-node-installer-*.tar.gz` | 同上 | GPU 节点安装脚本、unit 和精确 Node Runtime wheel |
| `dist/<release-id>/attestation.json` | `build-release-attestation.py` | 内容寻址 release attestation |
| `dist/current-attestation.json` | 同上 | 部署入口消费的 attestation |
| `dist/current-attestation.bundle.json` | `cosign sign-blob` | release attestation 的 Sigstore 签名材料 |
| `dist/gpu-fault-deploy-host-<platform>.tar.gz` | `build-deploy-host-bundle.py` | 部署机离线 Python 环境和工具 payload |
| 同名 `.tar.gz.sha256` | 同上 | 部署机离线包 SHA-256 sidecar |
| `dist/gpu-fault-deploy-host-<platform>.sigstore.json` | `cosign sign-blob` | 部署机离线包的 Sigstore 签名材料 |

不要混淆以下三个名称：

- **Node installer bundle**：安装 GPU 节点 Node Runtime，属于 Manifest v4，旧v3仍支持；
- **deploy-host bundle**：初始化部署机 venv，不安装 GPU 节点；
- **Sigstore bundle**：签名和透明日志验证材料，不包含运行软件。

## 5. 信任链

发布链路按以下关系逐层绑定：

1. 十二个coverage shard分别绑定本域内容、测试环境和证据，并由固定main CI identity签名；
2. unit域gate验签并聚合十二个当前run shard、统一coverage floor、fault和duration证据；
3. 当前commit CI gate验签并绑定unit域gate，同时绑定当前tree、static和artifact；
4. Runtime Image descriptor绑定OCI digest、构建输入和候选中的镜像组件；
5. Manifest v4绑定三个wheel、Node bundle、独立运行镜像、节点wheelhouse inventory和delivery identity；
6. attestation绑定Manifest SHA、release ID、delivery identity、源码状态和CI gate SHA；
7. Release Sigstore bundle证明attestation来自批准的GitHub OIDC identity；
8. deploy-host archive使用独立`gpu-fault-deploy-host` distribution、Manifest和签名，
   绑定平台、依赖身份和全部payload；schema v2的commit授权由CI gate或部署机本地签名
   成功记录单独承担，不写进可跨commit复用的archive。

仅有版本号、tag、文件名、ConfigMap 名或 Kubernetes annotation 均不足以替代上述绑定。
任何一层缺失、摘要不一致、身份不匹配或制品来自不同 workflow run 时都必须停止。

## 6. 下载、验签与部署消费

### 6.1 下载与受控落盘

Release 管理员应：

1. 选择与审批 commit 或 `v*` tag 对应且整体成功的 workflow run；
2. 下载完整 `gpu-fault-release` artifact，不按文件名拼接其他 run 的制品；
3. 将 release 内容恢复为匹配 checkout 下的 `dist/...` 布局；
4. 按需把 deploy-host archive、`.sha256` 和 `.sigstore.json` 复制到
   `/secure/release/` 等权限受控目录；
5. 保持文件内容和相对 Manifest 路径不变，不重新打包、重命名平台或改写 JSON。

`/secure/release/` 是管理员选择的受控落盘目录，不是 CI 生成或自动发现的路径。
GitHub Actions artifact 的 30 天保留期也不是生产制品长期保存策略；需要长期保留的
已批准 release 应在过期前进入受控制品库。

### 6.2 初始化部署机

当前 CI 使用 keyless 签名，因此正常消费方式是：

```bash
make deploy-host-setup \
  DEPLOY_HOST_ARCHIVE=/secure/release/gpu-fault-deploy-host-linux-x86-64-cpython-312.tar.gz \
  DEPLOY_HOST_SIGNATURE_BUNDLE=/secure/release/gpu-fault-deploy-host-linux-x86-64-cpython-312.sigstore.json \
  CERTIFICATE_IDENTITY=<approved-release-identity> \
  CERTIFICATE_OIDC_ISSUER=<approved-release-issuer> \
  DEPLOY_HOST_VENV=/secure/gpu-fault/deployer-venv
```

setup会在安装前验证签名、archive内容、平台兼容性与当前clean checkout的payload身份；
旧schema v1 bundle仍按内嵌Git commit精确匹配。
bundle Manifest绑定OS、CPU架构、Python实现/3.12 ABI cache tag、sysconfig platform和
libc实现；任一不匹配都fail closed。schema v2的payload复用不能替代第5节的commit授权
或当前release验签。依赖安装始终使用`--no-index`。bundle中的
`dependency_identity_sha256`覆盖两个lock和平台兼容信息；相同依赖身份只创建一次共享
依赖venv，后续不同项目wheel只创建轻量overlay venv并通过`.pth`引用共享
site-packages，不重复安装两个lock。依赖层缺失、损坏或身份不一致时fail closed；旧
bundle没有依赖身份时保留原完整安装路径。完整依赖、项目CLI和系统工具检查通过后才
原子替换目标venv；相同bundle重复执行只验证并复用。初始化器不调用系统包管理器，
报告和venv不得包含AWS凭据、token、私钥、数据库密码或kubeconfig。
deploy-host wheel交付`gpu-fault-admin`、`gpu-training-submit`和
`gpu-fault-workload-annotate`；后两者保持现有Control Plane兼容入口。
统一deploy内部选择新venv时，同时绑定CLI及shell helper的Python PATH，不依赖调用者
仍激活的旧venv。直接从受检CLI执行的site命令也采用其解释器目录，API预算shim仍优先。
bundle还携带经过Manifest摘要校验的`config/admin-config.example.yaml`，setup将其安装
到`<deploy-host-venv>/share/gpu-fault/admin-config.example.yaml`。该文件是管理员首次
部署前准备`0600`配置输入的只读模板，不包含凭据，也不会自动写入任何state-dir。

开发checkout统一运行`make deploy-host-setup-online`：先只读验证路径和版本配置，再以两个Make任务
并行执行两份hash锁的Python/admin CLI安装与现有`ci-supply-chain-tools`隔离工具准备。
两路均由系统Python 3.12初始化，可通过`DEPLOY_HOST_BOOTSTRAP_PYTHON`指定解释器路径；
无需添加`make -j`，任一路失败均等待已启动的另一路结束后返回失败，已完成的环境不回滚。
CI仍可单独调用`ci-supply-chain-tools`。
`DEPLOY_HOST_VENV`与`SUPPLY_CHAIN_TOOLS_VENV`必须分离；显式路径可包含空格。
`SUPPLY_CHAIN_PYTHON`须位于工具venv的`bin`目录，`PROMTOOL`可指向另一个显式安装路径
或PATH上已有的工具，但不得落入主venv或安装器自身的运行venv。
版本及Prometheus archive SHA-256仍只由Makefile声明。
Python工具先检查实际环境、包版本、入口和依赖一致性；promtool先核验缓存archive摘要，
再比较其中二进制与所选工具的字节，最后复用既有tool-only前检验证精确版本。
archive准备可与Python工具安装并行，二进制安装须等待两者完成；同一venv内的pip写入仍串行。
tool-only前检仅依赖Python标准库，完整规则检查和行为测试仍需要锁定环境。
只有缺失或不一致的工具才重新安装；缓存archive缺失或损坏时重新下载并验摘要。
在线Python初始化本身仍重建venv并按锁安装，重复执行不承诺完全离线。
以上工具步骤不进入签名`deploy-host-setup`，也不增加离线安装的工具或测试前提。

### 6.3 消费签名 release

受控制品同步自动化把同一CI artifact恢复到受信checkout后，管理员仍执行统一四参数命令：

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault \
  --admin-email <operations-email>
```

CLI内部解析并验签attestation、Manifest和OCI digest；artifact、certificate identity、
site和低层`release-deploy`参数不进入普通管理员命令。实际Kubernetes/AWS mutation继续
受Profile、维护窗口、回滚和fail-closed门禁约束。
公开`preflight`不修改资源；deploy driver内部的`preflight --for-deploy`可能先持久化
刷新器前置journal、更新选中的刷新器并创建/清理CPU proof Job，以恢复Store读取前提。
这属于已授权部署事务，不是CI门禁或公共只读检查；它不能替代Store安全证明、候选host
预检或业务启动前的schema门禁。完整顺序见[开发者部署实现](开发者部署实现.md)。

## 7. 失败与重跑

| 失败阶段 | 处理原则 |
|---|---|
| main CI任一并行job失败 | 修复对应静态、测试、PostgreSQL或artifact问题后提交新commit |
| 单个coverage shard未命中 | 只执行该shard；其他同身份shard继续复用 |
| GitHub artifact查询或下载临时不可用 | 安全回退fresh执行该shard；不把不可验证的历史证据当作通过 |
| shard签名、身份、证据或历史run不合法 | fail closed；不得复用该shard，调查artifact后重新执行 |
| coverage combine或78% floor失败 | 检查分片遗漏、陈旧数据或覆盖率回退；不得单独接受某个shard |
| per-module floor失败 | 为该族补测试；不得调低`group_floor`/`file_floor`或收窄globs绕过 |
| 找不到匹配main CI run | 先让目标commit通过main push CI；不得用其他commit候选代替 |
| CI gate签名、源码或文件清单失败 | 停止晋级；不得获取AWS身份或拼接其他run文件 |
| OIDC、AWS 凭据或 ECR 登录失败 | 修复 GitHub environment、角色信任或最小权限后重跑 |
| repository URI 校验失败 | 修正 GitHub variable；不得绕过 ECR URI 检查 |
| PostgreSQL stress 失败 | 修复并发/schema问题或 CI service；不得以普通 pytest 代替 |
| Runtime Image tag/label不一致 | 调查输入或 registry 漂移；不得覆盖不匹配的不可变 tag |
| wheel、Node bundle 或镜像一致性失败 | 停止发布并调查工具链、lock 或源码闭包漂移 |
| attestation/Cosign失败 | 检查 OIDC identity、`id-token` 权限和签名服务后重跑 |
| deploy-host bundle失败 | 检查 lock wheel、Python 3.12、平台和干净工作树 |
| artifact上传失败 | 整个 run 视为失败，不得只从 Runner 临时目录取文件部署 |

如果 OCI 已推送但后续步骤失败，该 OCI 不能单独视为可发布 release。必须得到完整、
签名且上传成功的 Manifest、attestation 和相关制品后才能交给部署阶段。

同一干净 commit 重跑时可以复用完全匹配的不可变 OCI；仍应把新 run 视为一套独立发布
结果，不能把两个 run 的签名或文件混合。

## 8. 本地受信环境复现

源码环境先执行`make deploy-host-setup-online`，一次准备管理员CLI和固定版本的隔离门禁工具；
CI保留独立`ci-supply-chain-tools`目标。本地完整测试、`make check`及release门禁会在
昂贵步骤之前执行`promtool-preflight`，验证可执行性和Makefile声明的精确版本，
并把路径传给pytest、xdist和子进程。工具缺失或版本不符立即失败，不记为skip，也不在
测试中自动下载安装。CI独立`promtool-check`的成功不能替代另一次本地测试的依赖前检。
定向影响测试只有选中Make清单中的PromQL测试时才要求工具，并向实际pytest子进程传入
前检返回的绝对路径；其他选择和已签名CI制品复用不因此增加本地测试或工具要求。

本地复现用于 Release 工程和故障定位，不替代正式 CI 审批。前置条件包括 Python 3.12、
隔离 PostgreSQL 16、ECR 登录、Docker Buildx、Cosign，以及已经安全设置但不打印的
`GPU_FAULT_TEST_POSTGRES_URL`。

```bash
test -n "${GPU_FAULT_TEST_POSTGRES_URL}"

COSIGN_SIGNING_KEY=/secure/release/cosign.key \
make PYTHON=.venv/bin/python release-build \
  RUNTIME_IMAGE_REPOSITORY=<ecr-repository>

COSIGN_SIGNING_KEY=/secure/release/cosign.key \
make PYTHON=.venv/bin/python deploy-host-bundle
```

本地示例使用受控私钥；GitHub workflow 使用 keyless OIDC。两种签名模式的部署验签参数
不同，不能把 keyless bundle 当作固定 public key 签名处理。

## 9. 修改与验证

修改发布流程时至少同步：

| 变化 | 必须审阅 |
|---|---|
| workflow trigger、权限、Runner 或 GitHub variables | 本文第 2、3 节 |
| CI并行job、CI gate或`release-build-promoted` | 本文第3.1至3.4节、开发者部署实现和运维手册 |
| `release-build-staging`或两级attestation边界 | EC2源码Staging流程、安全参考和本文第 1 节 |
| Manifest、wheel、Node bundle、attestation | 本文第 4、5 节及对应制品测试 |
| deploy-host bundle、共享依赖层、平台或 lock | 本文第3.5、6.2节和开发者部署实现 |
| artifact 名称、路径或保留期 | 本文第3.2、3.5、4、6.1节 |
| CI 与实际部署职责边界 | 本文第 1、6.3 节和管理员文档 |

只修改文档仍必须运行：

```bash
make docs-check
```
