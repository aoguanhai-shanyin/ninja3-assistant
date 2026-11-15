# 架构

WinUI 启动 Python Worker，Worker 执行任务计划。Plan 管理任务、进度、重试和优先级；Scheduler 只排序并串行调度；每个任务拥有自己的 `run`。

| 模块 | 职责 |
| --- | --- |
| `src/ninja3_assistant/scheduling/activity_task.py` | 活动任务接口与入口 |
| `src/ninja3_assistant/scheduling/task_planner.py` | 日常优先、后续目标、重试与对局恢复 |
| `src/ninja3_assistant/scheduling/task_scheduler.py` | 执行 Plan 提供的任务顺序 |
| `src/ninja3_assistant/runtime/realtime_battle.py` | 共用实时内核：新鲜帧、识别节奏、动作计划和撤回 |
| `src/ninja3_assistant/device/native_touch.py` | Python 到 C++ 的有界命令协议 |
| `native_touch/main.cpp` | 多手指触控执行、截止时间及释放 |
| `src/ninja3_assistant/activities/weekend/runner.py`、`src/ninja3_assistant/activities/weekend/obstacles.py`、`src/ninja3_assistant/activities/weekend/rope.py` | 竞速、正向避障及绳索处理 |

连续移动可提交最多 3 秒，点击计划使用更短的时间窗口；后续帧可替换或撤销。地面移动、应急跳跃、武器/卡牌按手指分配；暂停、停止和场景退出取消剩余动作。玩法策略不得另开相互竞争的输入循环。

运行代码位于 `src/ninja3_assistant/`，使用绝对包导入。`paths.py` 集中处理项目根目录与冻结程序资源位置；`packaging/` 集中维护入口和资源清单。测试按领域对应分组，工具与运行包分离。具体目录见 `DIRECTORY_LAYOUT.md`。

目录迁移不改变任务优先级、触控时序或玩法策略。数据与发布输出隔离，公开仓库不包含游戏派生素材。
