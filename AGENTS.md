# 忍3助手：工作约定与目录规范

## 当前入口

- 用户启动：项目根目录 `启动助手.cmd`。
- 当前程序唯一发布位置：`dist/Ninja3Assistant/Ninja3Assistant.exe`。
- 必须整套发布 WinUI 程序、`Ninja3Worker.exe`、`NinjaTouch.exe` 和资源；不能只换主 EXE。
- 根目录 README 只保留当前使用、构建、验证和目录入口。历史实验说明放 `docs/`。

## 目录职责

| 位置 | 用途 |
| --- | --- |
| `src/ninja3_assistant/` | 运行包：按调度、内核、设备、活动、战斗和入口分组；使用绝对包导入 |
| `packaging/` | PyInstaller 配置，入口与资源路径相对项目根目录解析 |
| `winui/` | Windows 界面源码，`bin/`、`obj/` 为生成文件 |
| `native_touch/` | C++ 输入内核源码、协议文档、原生构建入口 |
| `assets/` | 运行时模板、模型和 OCR 脚本；模板应能追溯到证据 |
| `configs/` | 项目配置；用户本地设置不混入源码 |
| `tests/` | 自动化测试；读取固定证据，不得启动正式游戏任务 |
| `tools/` | 构建、离线分析、诊断和显式实测工具；导入工具不得自动操作游戏 |
| `docs/` | 架构、目录说明、历史记录 |
| `build/` | 可再生的编译缓存、打包中间产物、构建日志 |
| `dist/Ninja3Assistant/` | 唯一当前完整发布包；运行记录通过兼容目录指向 `data/runtime/` |
| `data/runtime/` | 当前程序运行日志和截图，按活动/运行时间分目录 |
| `data/evidence/YYYY-MM-DD/` | 已保留的复现图、回归测试证据和验证结论 |
| `data/experiments/` | 临时分析、对照图和实验产物，按主题/日期分目录 |
| `D:/CodexArtifacts/` | 大体积实测录像、逐帧记录；在本项目中保留索引 |

旧 `data/hayabusa-*`、`data/intelligence-simulation`、`data/weekend-race` 等目录暂按历史证据维护。不能为了美观搬走仍被测试、模型清单引用的数据。旧发布路径的兼容联接不代表另一份当前版本。

根目录不新增截图、录像、日志、临时 JSON、带日期的发布包。`data/` 不再新增 `publish-*` 或打包缓存。新实验必须先选定输出子目录。

## 运行架构边界

- `src/ninja3_assistant/scheduling/activity_task.py`：任务接口，各任务自己的 `run`、开放时段和后续目标。
- `src/ninja3_assistant/scheduling/task_planner.py`：Plan 管任务、进度、重试和动态优先级；日常优先。
- `src/ninja3_assistant/scheduling/task_scheduler.py`：按 Plan 的优先级排序并串行执行；不写活动玩法判断。
- `src/ninja3_assistant/runtime/realtime_battle.py`：所有活动/自动战斗共用的实时内核，负责帧时效、动作计划和撤回。
- `src/ninja3_assistant/device/native_touch.py`、`native_touch/`：C++ 输入执行；不能在多个 Python 线程中另起相互竞争的滑步循环。
- 连续移动可以提前提交最多 3 秒；新帧可修改或撤回。暂停、停止和场景退出必须释放手指、取消剩余计划，旧帧不能恢复已撤回动作。
- 异常保留未完成状态，交给 Plan 重试；不能把异常当任务完成，或直接结束全部任务。
- 日常只看进度、领取状态和完整滚动扫描结果，不依赖固定任务类型。三项均确认领取才结束；有未完成项就继续打，结算后复查。

## 实测与修改原则

1. 修改前核对现有文件和运行进程，保留其他未完成工作。本目录使用 Git；修改前检查状态，保留其他未提交工作。
2. 游戏画面使用 MuMu SDK；操作使用 C++ NinjaTouch。ADB 可用于只读诊断，不用电脑鼠标自动化替代游戏输入。
3. 移动动作先在训练所隔离验证，再按用户已有授权接入活动。不要用正式积分局反复猜参数。
4. SDK 返回成功、单元测试通过、输入日志存在，都不等于游戏动作成功。必须检查连续画面中的角色动作、场景和实际进度。
5. 分别报告采集频率、识别频率、输入时序与游戏效果；不能混成一个 FPS，也不能将底层时序验证称为稳定避障/上分。
6. 界面默认显示游戏窗口；隐藏仅由用户选项控制。启用悬赏时，在安全间隙接受好友或族员的所选等级悬赏邀请；其他邀请应拒绝，不能只收起遮挡。
7. 目录整理不夹带玩法重构；移文件前查引用，更新后检查链接、入口、资源与包内容。保留原始实测记录和模型来源。
8. Windows 使用 PowerShell，不使用 bash。检索优先 `rg`；路径有空格时使用可靠的参数传递方式。
9. 自动化与实测优先使用 `tools/diagnostics/assistant_control.py`，通过 JSON 行或本机 HTTP 接口启动、暂停、继续、停止、读取状态和截图；游戏输入仍由 C++ NinjaTouch 执行。控制程序已覆盖的操作应优先调用接口。接口无法覆盖的必要界面检查可以使用 Computer Use，这是一项优先级约定，并非禁用；游戏采集和输入仍遵守第2条。一次实测只能有一个输入控制者，启动控制程序前先确认旧 Worker 已退出。

控制程序在项目根目录运行。启动本机服务后，可从另一条命令查询或控制；`start` 只传本轮需要的任务，不自动启用额外活动：

```powershell
python -m tools.diagnostics.assistant_control --folder '<MuMu安装目录>' --index 1 --worker dist/Ninja3Assistant/Ninja3Worker.exe --serve
python -m tools.diagnostics.assistant_control --command start --mode run --daily-tasks monthly limited blessing weapon curse bounty_shop daily_shop house bounty bounty_s village rank --bounty-grade A --hide-game-window
python -m tools.diagnostics.assistant_control --command status
python -m tools.diagnostics.assistant_control --command stop
python -m tools.diagnostics.assistant_control --command exit
```

`--hide-game-window` 依用户选项决定，悬赏等级沿用本轮所选等级；服务只监听 `127.0.0.1`，实测结束后停止 Worker 并退出控制服务。

## 构建与验证

- 统一构建：`tools/build/build_assistant.ps1 -Python <python.exe>`，输出到 `dist/Ninja3Assistant/`。
- 使用 Python 3.12 虚拟环境，安装 `requirements-dev.txt` 并执行 `pip install -e .`；构建前检查 `assets/asset-manifest.json` 所需的本地资源。
- C++：`cmd /c native_touch\build.cmd`，输出 `build/ninja-touch/NinjaTouch.exe`。
- 常用检查：`python -m unittest tests.activities.intelligence.test_tasks tests.activities.intelligence.test_flow tests.scheduling.test_task_planner tests.runtime.test_realtime_battle tests.device.test_native_touch -q`。
- 输入协议检查需要先构建 C++：`python -m unittest tests.device.test_native_touch_protocol -q`。
- 训练所工具：`tools/diagnostics/verify_skate_kernel.py`，支持 3 秒计划、途中撤回、替换、测试后暂停。
- 构建后核对发布包三组件和资源；用打包 worker 的无动作启动检查验证参数与导入。发布清单记录组件及源码哈希。
- 检查范围与剩余失败必须如实报告；已有失败不能靠改断言来伪造通过。

## 公开仓库边界

- 不提交游戏原始素材、训练权重、账号截图、录像、模拟器配置、个人路径、运行记录或发布二进制。
- 游戏识别资源由使用者自行准备，文件清单见 `assets/asset-manifest.json`。
- 无素材验证运行 `python -m tools.testing.run_portable_tests`；需要私有图像或模型的回归测试另行说明，不得把部分检查称为完整验收。

## 包与路径

- 资源目录统一通过 `ninja3_assistant.paths` 解析；禁止根据业务模块所在目录猜测项目根目录。
- Worker 入口为 `python -m ninja3_assistant`；不恢复根目录模块副本或 sys.path 引导兼容层。
- 工具按 `python -m tools.<分类>.<工具名>` 运行，测试按领域放入对应包。
