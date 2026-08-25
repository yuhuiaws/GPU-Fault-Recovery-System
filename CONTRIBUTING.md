# Contributing

修改代码或部署契约前，先阅读
[GPU Fault 扩展指南](docs/扩展指南.md)。其中 operation registry、channel
registry、Node Action handler、通知 builder、生产 Manifest lifecycle annotation、
systemd installed-unit inventory 和外部清理状态门禁是新增功能的强制入口，不能只在
调用侧或线上增加字面量。

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
make check
```

提交前至少运行：

```bash
make check
```

`make coverage` 使用 4 个 worker 跑同一套测试，并强制当前 78% 的覆盖率 floor。
为避免多个 worker 操作同一数据库，该目标不启用外部 PostgreSQL；设置
`GPU_FAULT_TEST_POSTGRES_URL` 后另跑 `make test-postgres`。覆盖率可以提高，
不能通过调低 `COVERAGE_FLOOR` 掩盖未测试的新分支。

只修改文档时，仍必须运行：

```bash
make docs-check
```
