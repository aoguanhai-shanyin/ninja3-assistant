# WinUI 界面约定

本文件适用于 `winui/` 及其子目录，补充根目录 `AGENTS.md`。以下路径均相对项目根目录。

## 文件职责

- `winui/MainWindow.xaml`：任务、设置、视频页面及启动、暂停、停止等控件布局。
- `winui/MainWindow.xaml.cs`：设置读写、模拟器列表、Worker 生命周期、任务状态与事件显示。
- `winui/MainWindow.Video.cs`：离线视频播放、逐帧请求及分析结果显示。
- `winui/App.xaml`、`winui/App.xaml.cs`：应用资源与启动。
- `winui/Ninja3Assistant.csproj`：WinUI 构建、资源和 Worker、NinjaTouch 发布复制规则。
- `winui/Assets/`：界面资源；`bin/`、`obj/` 是生成目录，不手改或提交。

## 界面与运行边界

- 界面负责收集选项、发送控制命令和显示状态；任务优先级、重试、识别和战斗策略属于 Python Worker，不能在 C# 重复实现。
- 支持多任务选择，由 Plan 安排日常优先；不恢复“只允许一个活动”的界面限制。
- 启动、暂停、继续、停止的可用状态随 Worker 实际状态更新。停止请求发出后不能立即假定进程已退出，也不能启动第二个竞争输入的 Worker。
- 游戏窗口默认可见，隐藏只能由用户选项控制；新增设置应保持旧设置文件可读取，并有明确默认值。
- 用户设置保存在现有用户配置位置；不把个人设置、模拟器路径或账号数据写进源码。
- 进程输出读取、扫描和等待使用异步方式，UI 更新通过 Dispatcher；不在 UI 线程新增阻塞等待。
- 处理异步回调时检查当前进程或请求标识，避免旧 Worker、旧视频帧覆盖新状态。退出或切换模式时释放对应进程、计时器和流。
- 离线视频只分析并展示建议，不能发送游戏输入；保留视频分析与实时任务之间的互斥控制。
- 界面文案使用简洁中文，展示用户可理解的任务状态；实现细节留在日志中。调整布局时保证控制按钮无需滚动即可找到，并检查小窗口和显示缩放。

## Worker 协议

- Worker 由同一发布目录下的 `Ninja3Worker.exe` 和配套 `_internal/` 组成。参数与 `src/ninja3_assistant/entrypoints/worker.py` 保持一致。
- 标准输入按行发送 JSON 控制命令，包括 `pause`、`resume`、`stop`；标准输出按行读取 `event`、`value` 事件，诊断信息与事件解析区分处理。
- 修改事件、参数或状态含义时同时更新 Python 端和相关测试；不能只改界面让状态看似成功。

## 验证与发布

- 仅改本文等说明文件时检查路径、命令和差异即可，无需重建程序。
- 界面代码改动可先在项目根目录执行 `dotnet build winui/Ninja3Assistant.csproj -c Release -p:Platform=x64 -r win-x64 -p:PublishReadyToRun=false`；这只是编译检查，不等于完整发布。
- 完整发布统一执行 `tools/build/build_assistant.ps1 -Python <python.exe>`；构建前按脚本要求退出现有发布程序及 Worker。
- 发布位置只有 `dist/Ninja3Assistant/`。必须包含 WinUI、Worker、NinjaTouch 及所需资源，不单独替换主 EXE；启动入口仍是根目录 `启动助手.cmd`。
- 布局改动检查实际渲染；状态控制改动检查启动、暂停、继续、停止和异常退出；视频改动检查导入、跳帧、播放及关闭。验证不得自行启动正式积分任务。
- 分别报告编译、界面观察和 Worker 联动验证结果；未执行的检查明确说明。
- 启停、暂停、恢复和运行状态验证优先使用 `tools/diagnostics/assistant_control.py`；只有接口未覆盖的必要界面检查（例如布局、托盘和关闭询问）再使用 Computer Use。不要为了执行日常任务反复点击助手界面。
