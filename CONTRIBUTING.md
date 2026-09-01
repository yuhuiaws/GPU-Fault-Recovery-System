# Contributing

修改前按变化类型选择文档：

| 变化 | 必读文档 |
|---|---|
| 新增 operation、channel、Store、路由、插件或指标 | [GPU Fault 扩展指南](docs/扩展指南.md) |
| 修改 Manifest、renderer、systemd、AWS 资源、配置模型或管理员 CLI | [开发者部署实现](docs/开发者部署实现.md) |
| 修改 Release CI、签名、`dist/`制品或部署机离线bundle | [CI 发布流程](docs/CI发布流程.md)和[开发者部署实现](docs/开发者部署实现.md) |
| 完成代码修改并部署到staging或生产验证 | [EC2源码统一部署流程](docs/EC2源码Staging复现流程.md) |
| 同时新增代码能力和生产资源 | 两份都读，两套门禁都执行 |
| 只部署已有 release | [管理员快速部署](docs/管理员快速部署.md) |

operation/channel registry、显式授权、生产资源生命周期和管理员入口都不能只在调用侧
或线上增加字面量。扩展指南决定能力登记点，开发者部署实现决定生成、发布、验证、
回滚和卸载方式。

文档职责和权威顺序见 [docs/README.md](docs/README.md)。历史材料不得作为当前
实现依据。

`src/**/*.py`、`tests/**/*.py`、`scripts/**/*.sh` 和生产部署脚本受
[代码与文档影响契约](docs/code-doc-contracts.yaml)约束。修改这些文件时：

- 同时修改契约列出的相关文档；或
- 在 Pull Request 正文中填写 `Documentation-Impact: none`，并在
  `Documentation-Impact-Reason` 中说明为什么公共行为、命令和验收契约均未变化。

不能用空值、`N/A` 或 `TODO` 代替原因。CI 会基于目标分支和当前提交的 Git diff
执行 `scripts/check-doc-impact.py`。

本地确认属于纯内部改动时，可以显式运行：

```bash
GPU_FAULT_DOC_IMPACT=none \
GPU_FAULT_DOC_IMPACT_REASON='仅重构内部实现，公共行为和命令未变化' \
make PYTHON=.venv/bin/python check
```

日常开发先按相对目标分支的实际差异运行反向影响选择：

```bash
make test-impact BASE=origin/main
make regional-impact-plan BASE=origin/main
```

`test-impact`执行相关静态检查和pytest；`regional-impact-plan`只输出受影响的区域用例，
不会自动执行live或destructive case。无法匹配的文件、共享契约、Python/依赖/runtime
image、schema/事务变化或跨越三个以上影响域时会fail closed并升级为完整门禁。
版本候选和正式发布仍必须运行`make check`；完整区域验收只跟随首次上线、重大架构变化
或影响计划明确要求执行。规则与说明见
[变更影响与测试选择](docs/变更影响与测试选择.md)。

管理员首次部署由一个ARN命令完成基础资源、release build和应用部署：

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault \
  --admin-email <operations-email>
```

首次建站需要覆盖容量默认值时，可以额外传入严格、权限`0600`的
`--config <AdminConfig.yaml>`；不传仍是原四参数默认路径。已有站点的容量变化使用
`gpu-fault-admin capacity plan/apply`，详见
[管理员容量配置](docs/管理员容量配置.md)，不得通过shell环境变量或`kubectl set env`
替代。

开发者修改代码后也使用同一四参数命令；dirty/clean等级、影响测试、构建、验签和部署
由内部流程选择：

```bash
gpu-fault-admin deploy \
  --cpu-cluster-arn <cpu-arn> \
  --gpu-cluster-arn <gpu-arn> \
  --state-dir /secure/gpu-fault-staging \
  --admin-email <operations-email>
```

Runtime Profile策略变化时，首次deploy生成
`<state-dir>/release-deploy/profile-plan.json`并停止。审核后运行：

```bash
gpu-fault-admin approve-profile \
  --state-dir /secure/gpu-fault-staging \
  --plan-sha256 "$(jq -er '.plan_sha256' \
    /secure/gpu-fault-staging/release-deploy/profile-plan.json)" \
  --reference CHG-12345
```

随后重跑原四参数deploy。不得通过隐藏参数或环境变量注入审批引用。
完整管理员流程见[Runtime Profile变更审批](docs/管理员Profile变更审批.md)。

GitHub Actions从触发、OIDC/ECR、质量门禁、制品签名到`gpu-fault-release` artifact
交付的完整顺序见[CI 发布流程](docs/CI发布流程.md)。
开发者修改代码后的最短测试、staging部署和同制品生产晋级顺序见
[EC2源码统一部署流程](docs/EC2源码Staging复现流程.md)。

`release-build`、`release-deploy`、artifact路径、site生成路径和release-ref属于内部
发布实现，不是普通开发者或管理员参数。

`make check`的最终全量测试和`make test-parallel`使用4个worker且不连接外部
PostgreSQL。`make coverage`要求设置`GPU_FAULT_TEST_POSTGRES_URL`：先由4个worker
采集非PostgreSQL覆盖率，再串行追加隔离PostgreSQL 16测试库覆盖率，最后统一强制当前
78%的floor。覆盖率可以提高，不能通过调低`COVERAGE_FLOOR`掩盖未测试的新分支。
Make在checkout中检测到`.venv/bin/python`时会自动使用该解释器；源码包没有`.venv`
时回退到`python3`，显式`PYTHON=...`始终优先。

只修改文档时，仍必须运行：

```bash
make docs-check
```
