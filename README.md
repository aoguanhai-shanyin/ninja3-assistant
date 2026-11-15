# 忍3助手 · Ninja3 Assistant

面向 Windows / MuMu 的《忍者必须死3》实验自动化助手。WinUI 界面、Python 任务规划与视觉处理、C++ 多手指输入，共用可撤回的实时战斗内核。

**当前为源码发布。** 游戏识别素材、模型权重、录像及账号数据不随仓库提供；运行完整游戏任务前，需要自行准备有权使用的本地资源。项目非官方，与游戏或模拟器厂商无关联。

## 功能与状态

- 多任务自动规划：先完成已选活动的日常，再打满场次或继续对战；对局未结束的恢复不切换其他活动。
- 情报模拟战：滚动检查日常进度、领取奖励、选牌和对局；异常保留任务进度并重试。
- 周末同屏竞速：读取今日剩余积分场次，正向识别障碍；基础移动为“两跳 → 下滑”。绳索上跳跃前进，识别到上方地板时尝试登板，离绳后接回地面循环。
- 自动战斗：共享实时帧、指令截止时间、多手指计划、取消与暂停释放机制；提供录像回放与训练所诊断工具。
- 连续移动计划最长 3 秒，可续接、修改或撤回；武器/卡牌使用独立手指。

**避障、绳索登板与稳定上分尚未完成实战验收。** 输入日志和单元测试通过不等于游戏动作成功。已知完整回归限制见 [tests/README.md](tests/README.md)。

## 开发与构建

需要 Windows、Python 3.12、Visual Studio C++ Build Tools、Windows SDK 和支持项目目标框架的 .NET SDK。WinUI 依赖版本以 `winui/Ninja3Assistant.csproj` 为准。

```powershell
git clone https://github.com/tanwenhai/ninja3-assistant.git
cd ninja3-assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m tools.testing.run_portable_tests
```

公开核心测试无需游戏资源或模拟器。完整运行还需根据 [assets/README.md](assets/README.md) 准备本地资源：

```powershell
python -m tools.build.check_assets
./tools/build/build_assistant.ps1 -Python ./.venv/Scripts/python.exe
```

构建会生成完整程序到 `dist/Ninja3Assistant/`。先退出正在运行的助手，再构建；不能只替换主 EXE。

安装后也可执行 `ninja3-worker --help` 或 `python -m ninja3_assistant --help` 查看 Worker 参数。开发工具从项目根目录使用 `python -m tools.<分类>.<工具名>` 运行。

## 使用

1. 自行安装 MuMu 和游戏，将所选实例设置为 1280×720 横屏。
2. 构建完成后，双击根目录 `启动助手.cmd`。
3. 在设备页选择实例，在任务中心勾选活动和结束方式，然后启动。暂停、继续、停止位于任务页顶部。
4. 默认显示游戏窗口，可通过设备选项控制隐藏。首次修改动作，先在训练所验证。

不要上传包含账号标识、个人信息、配置或未经许可素材的日志和截图。使用时自行了解游戏及模拟器的相关规则。

## 项目结构

| 位置 | 内容 |
| --- | --- |
| `src/ninja3_assistant/` | 按领域分组的 Python 包：调度、实时内核、设备、活动、战斗与入口 |
| `winui/` | WinUI 界面与 Worker 启动 |
| `native_touch/` | C++ 输入执行及协议 |
| `assets/` | OCR 脚本、资源清单；识别素材本地准备 |
| `configs/` | 示例配置；用户实际配置不提交 |
| `tests/` | 与源码领域对应的测试包 |
| `tools/` | 按构建、采集、回放、模型、数据集和诊断分组的工具 |
| `packaging/` | PyInstaller 打包配置 |
| `docs/` | 架构与目录说明 |
| `build/`、`dist/`、`data/` | 本地输出、素材及记录，均不提交 |

[架构](docs/ARCHITECTURE.md) · [目录规范](docs/DIRECTORY_LAYOUT.md) · [开发约定](AGENTS.md) · [参与开发](CONTRIBUTING.md)

## 许可证

原创源码与文档采用 [MIT](LICENSE)。游戏内容、商标、SDK 和第三方模型不因此获得 MIT 授权，也未随源码分发；详见 [第三方说明](THIRD_PARTY_NOTICES.md)。
