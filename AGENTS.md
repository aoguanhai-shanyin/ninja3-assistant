# 忍3助手：工作约定与目录规范

## 开发与提交检查

- 修改前运行 `git status --short`，保留已有未提交工作。目录整理限于文件布局，移动前查引用，改动后核对链接、入口、资源和发布包。
- Windows 使用 PowerShell，优先使用 `rg` 检索。通过引号或参数数组传递含空格的路径。
- 在项目根目录用 Python 3.12 运行 `python -m venv .venv` 创建环境，然后用 `.\.venv\Scripts\Activate.ps1` 激活。日常检查使用 `python -m pip install -r requirements-lint.txt -e .`。需要构建完整程序时使用 `python -m pip install -r requirements-dev.txt -e .`。

| 检查 | 命令与范围 |
| --- | --- |
| Python lint | `python -m ruff check src tests tools` 检查运行代码、测试和工具。 |
| Windows 验证 | `python -m tools.testing.verify` 编译 NinjaTouch，并运行无需游戏素材的核心测试和原生输入协议模拟测试。 |

- 在 Git for Windows 中，先用 `git config --get core.hooksPath` 查看已有配置，并检查现有本地钩子。确认所需钩子已保留后，运行 `git config --local core.hooksPath .githooks` 启用本仓库的 `pre-commit` 钩子。钩子检查暂存差异中的空白错误，并用 Ruff 检查 `src/`、`tests/` 和 `tools/` 中暂存的 Python 内容。
- GitHub 的 `Core tests` 工作流在推送和拉取请求时运行上述 lint 和 Windows 验证。按改动补做私有素材回归、WinUI 或实机检查，并报告实际通过、未通过及因环境缺失无法执行的检查。
- 重构前后运行同一组相关检查，区分已有失败和本次改动引入的回归。

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

## 画面识别与 WinML

| 场景 | 识别方式 |
| --- | --- |
| 情报模拟战卡牌头像 | WinML 批量匹配 |
| S 级悬赏徽记 | WinML，供反打和每日 S 级悬赏找队使用 |
| 隼白自动战斗 | WinML 检测攻击、定位玩家，以及备用的攻击颜色分割 |
| Boss 调试 | WinML 分割红色攻击特效 |
| 普通悬赏等级、固定按钮和弹窗 | 优先用小范围 OpenCV 模板匹配和颜色规则 |
| 须弥幻境的入口、编队、装备与日常进度 | 局部模板与 Windows OCR，战斗复用实时内核 |
| 答题、商品名称、活动和客户端更新提示等文字 | Windows OCR，按任务规则校验识别结果 |

- ONNX 会话统一通过 `ninja3_assistant.vision.winml.model_session` 创建。GPU 目前经由 Windows ML 加载已安装的 TensorRT RTX 提供程序。只在调用方明确允许的场景使用 CPU，离线回放可显式选 CPU。
- 批量识别或高频战斗检测适合评估 WinML。新增模型前用相同实测帧对照识别结果、首次加载和稳定运行耗时，确认对实际流程有收益。耗时的加载和预热应避开实时帧循环。
- 查找入口、滚动方向、购买确认、领取判断、维护和重试由任务流程处理。根据截图和日志修复状态判断与退出条件。
- 客户端更新弹窗优先于背景任务处理。确认提示和重启按钮后，用新帧复核再点击，待客户端恢复后继续调度未完成的任务。

## 游戏实测

1. 实测前核对运行进程，确认旧 Worker 已退出，每轮只保留一个输入控制者。游戏画面统一使用 MuMu SDK，游戏输入统一使用 C++ NinjaTouch。ADB 仅用于只读诊断。
2. 优先通过 `tools/diagnostics/assistant_control.py` 的 JSON 行或本机 HTTP 接口启动、暂停、继续、停止、读状态和截图。接口未覆盖的必要界面检查可使用 Computer Use。游戏采集与游戏输入遵循上一条。
3. 移动动作先在训练所隔离验证，再按用户已有授权接入活动。不要用正式积分局反复猜参数。训练所工具 `tools/diagnostics/verify_skate_kernel.py` 支持 3 秒计划、途中撤回、替换和测试后暂停。
4. 通过连续画面确认角色动作、场景和实际进度。分别报告采集频率、识别频率、输入时序和游戏效果。稳定避障和上分需要各自的实测证据。
5. 界面默认显示游戏窗口；隐藏仅由用户选项控制。启用悬赏时，在安全间隙接受好友或族员的所选等级悬赏邀请；其他邀请应拒绝，不能只收起遮挡。自建悬赏队伍人数不足时等待约 45 秒。加入其他队伍后，等待最多 60 秒开战，再尝试其他队伍。

控制程序在项目根目录运行。启动本机服务后，可从另一条命令查询或控制；`start` 只传本轮需要的任务，不自动启用额外活动：

```powershell
python -m tools.diagnostics.assistant_control --folder '<MuMu安装目录>' --index 1 --worker dist/Ninja3Assistant/Ninja3Worker.exe --serve
python -m tools.diagnostics.assistant_control --command start --mode run --daily-tasks monthly limited blessing weapon curse bounty_shop daily_shop house bounty bounty_s village rank --bounty-grade A --hide-game-window
python -m tools.diagnostics.assistant_control --command status
python -m tools.diagnostics.assistant_control --command stop
python -m tools.diagnostics.assistant_control --command exit
```

`--hide-game-window` 依用户选项决定，悬赏等级沿用本轮所选等级；服务只监听 `127.0.0.1`，实测结束后停止 Worker 并退出控制服务。

## 按改动补充验证与构建

- 快速回归按改动领域运行对应测试。只跑无需素材的核心 Python 测试时使用 `python -m tools.testing.run_portable_tests`。本地截图、模型与历史数据齐备时，运行 `python -m unittest discover -s tests -t . -q` 做完整回归。遇到失败先定位原因，保持测试断言与预期行为一致。
- 单独构建 C++ 使用 `cmd /c native_touch\build.cmd`，输出 `build/ninja-touch/NinjaTouch.exe`。
- 完整构建使用 `tools/build/build_assistant.ps1 -Python <python.exe>`。构建前检查 `assets/asset-manifest.json` 列出的本地资源并退出正在运行的助手。构建后核对 `dist/Ninja3Assistant/` 的三组件和资源，检查发布清单记录的组件及源码哈希，用打包 Worker 的无动作启动检查验证参数与导入。

## 公开仓库边界

- 不提交游戏原始素材、训练权重、账号截图、录像、模拟器配置、个人路径、运行记录或发布二进制。
- 游戏识别资源由使用者自行准备，文件清单见 `assets/asset-manifest.json`。
- 保留原始实测记录和模型来源，目录整理时核对依赖这些证据的测试和模型清单。

## 包与路径

- 资源目录统一通过 `ninja3_assistant.paths` 解析；禁止根据业务模块所在目录猜测项目根目录。
- Worker 入口为 `python -m ninja3_assistant`；不恢复根目录模块副本或 sys.path 引导兼容层。
- 工具按 `python -m tools.<分类>.<工具名>` 运行，测试按领域放入对应包。
