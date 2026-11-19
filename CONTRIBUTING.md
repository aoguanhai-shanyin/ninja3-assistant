# 参与开发

先阅读 `AGENTS.md` 和 `docs/ARCHITECTURE.md`。提交保持小范围；说明触发条件、实际行为、验证方式和剩余限制。

从项目根目录使用 Python 3.12 环境，安装 `python -m pip install -r requirements-lint.txt -e .` 后执行：

```powershell
python -m ruff check src tests tools
python -m tools.testing.verify
```

`verify` 使用 Windows C++ Build Tools 编译 NinjaTouch，再运行公开核心测试和原生 mock 回归。Python 核心测试可单独执行 `python -m tools.testing.run_portable_tests`。验证范围见 [tests/README.md](tests/README.md)。

Git for Windows 可用 `git config --local core.hooksPath .githooks` 启用提交检查。先用 `git config --show-origin --get core.hooksPath` 确认现有设置。钩子检查暂存内容的空白和 Python lint。GitHub Actions 在 push 和 PR 上运行 lint 和 verify。

新动作先在训练所验证，检查连续画面中的角色动作、场景和进度。输入日志和 SDK 返回成功只能验证输入链路。

不要在 issue、PR、截图或测试数据中提交账号标识、配置、访问令牌、个人路径或未经许可的第三方素材。需要复现图时，优先提供自制合成样本或说明本地采样步骤。
