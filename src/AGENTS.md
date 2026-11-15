# Python 运行代码约定

本文件适用于 `src/` 及其子目录，补充根目录 `AGENTS.md`。以下路径均相对项目根目录。

## 模块职责

- `ninja3_assistant/entrypoints/`：Worker 参数、控制命令和事件输出；不承载活动策略。WinUI 是当前用户界面，`app.py` 是旧 Tk 入口，不能当作当前界面的修改位置。
- `ninja3_assistant/scheduling/`：任务各自实现 `run`；Plan 管任务、进度、重试及优先级；Scheduler 按优先级串行调度，不判断具体玩法。
- `ninja3_assistant/runtime/`：任务运行控制和共用实时战斗内核；自动战斗与活动复用同一套帧时效、动作计划和撤回机制。
- `ninja3_assistant/device/`：MuMu 采集和 C++ 输入协议适配；实际触控执行在 `native_touch/`，不在 Python 中新增竞争的输入循环。
- `ninja3_assistant/vision/`：通用场景识别和邀请处理；活动专用识别放在对应活动目录。
- `ninja3_assistant/activities/`：悬赏、情报模拟战和周末活动的流程与策略。
- `ninja3_assistant/combat/`：通用战斗及 Boss 策略；可复用能力放回共用内核，不为每个活动复制一份。

## 修改边界

- 使用 `ninja3_assistant.*` 绝对包导入；Worker 入口为 `python -m ninja3_assistant`。不添加根目录模块副本或 `sys.path` 引导。
- 项目与资源路径统一通过 `ninja3_assistant.paths` 解析；兼顾源码运行和冻结程序，不能依赖当前工作目录或业务文件层级。
- 导入模块不得启动游戏、进程或输入。采集工具、标注、训练和回放脚本放 `tools/`；测试放 `tests/` 对应领域。
- 高频路径避免重复加载模型、启动子进程、无界排队和大对象分配。性能优化分别测量采集、识别、排队与执行延迟；不未经测量就全局关闭 GC。
- 连续移动计划最多 3 秒，点击计划遵循现有更短的协议窗口；不能把移动期限直接套到点击。新帧可替换或撤回旧计划，过期动作不得补发。
- 暂停、停止、异常和场景退出必须取消剩余输入并释放手指。任务异常保留未完成状态，交由 Plan 重试，不冒充完成或中断全部任务。
- 识别状态、流程决策与设备执行保持职责分离；不得为了通过测试跳过识别边界、输入截止时间或停止屏障。
- 日常依据进度、领取状态和完整滚动扫描判断；不绑定固定任务类型。周末活动轮换时须确认玩法后再进入。
- 修改 Worker 的命令、事件或参数时，同时检查 `winui/` 消费端；JSON 行事件不能混入普通日志。

## 验证

在项目根目录使用 Python 3.12 环境，先执行 `python -m pip install -e .`。

- 无游戏素材的核心回归：`python -m tools.testing.run_portable_tests`。
- 按改动领域运行对应测试，例如：`python -m unittest tests.scheduling.test_task_planner tests.runtime.test_realtime_battle -q`。
- C++ 输入协议改动：先执行 `cmd /c native_touch\build.cmd`，再执行 `python -m unittest tests.device.test_native_touch_protocol tests.device.test_native_plans -q`；使用 mock，不连接正式游戏。
- 本地素材齐备时全量验证：`python -m unittest discover -s tests -t . -q`。
- 入口、导入或资源路径变更需检查冻结 Worker；完整构建统一使用 `tools/build/build_assistant.ps1 -Python <python.exe>`。

测试通过只能证明检查范围内的行为。移动、绳索及避障改动按根目录约定先在训练所验证连续画面，不把协议日志当作稳定上分证据。不提交私有素材、模型、截图和运行数据。
