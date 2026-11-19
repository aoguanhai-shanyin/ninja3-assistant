# 验证范围

公开核心测试：

```powershell
python -m tools.testing.run_portable_tests
```

这些测试覆盖任务规划、活动状态机、实时帧与动作计划、取消、预算组合选牌、未知卡证据归档、策略及模拟输入，无需游戏素材和模拟器。

Windows 提交验证使用 `python -m tools.testing.verify`。它会编译 NinjaTouch，运行同一套核心测试，再通过 `--mock` 运行原生输入协议和计划回归。GitHub Actions 使用此入口，并单独运行 `python -m ruff check src tests tools`。

`python -m unittest discover -s tests -t . -q` 是本地完整回归入口。其他测试还需要本地截图、模型与历史 `data/` 数据，未随公开仓库分发。缺少这些数据时，不能把完整回归失败当成核心逻辑回归，也不能宣称完整套件已通过。

训练所过渡测试按固定 45 秒期限检查；原生延迟指令按各自执行窗口、循环计数按明确周期边界检查，避免混入下一周期。避障、绳索登板与稳定上分仍需实机验证。
