# 目录规范

Python 运行代码采用 `src` 包布局。根目录只放项目元数据、使用说明和启动脚本，不放业务模块或打包配置。

```text
src/ninja3_assistant/
  entrypoints/           Worker、旧版界面入口
  scheduling/            任务接口、Plan、Scheduler、导航与结果
  runtime/               运行控制、实时战斗内核、输入调度
  device/                MuMu SDK、原生触控、宏与设备能力
  vision/                通用场景、邀请与 GPU 分割
  activities/
    bounty/              悬赏流程
    intelligence/        情报模拟战
    weekend/             同屏竞速、障碍与绳索
  combat/
    boss/                通用 Boss 策略与诊断
    hayabusa/            隼白识别、策略与录像调试
  paths.py               项目根目录与冻结资源路径

tools/
  build/                 统一构建、资源预检
  capture/               画面与输入采集
  replay/                离线回放与渲染
  benchmarks/            时序及性能分析
  datasets/              标注、图册与数据集准备
  models/                模型生成、训练与评估
  diagnostics/           控制端、网络与显式实测
  testing/               公开核心测试入口

packaging/               PyInstaller 配置
winui/                   Windows 界面
native_touch/            C++ 输入内核
```

`tests/` 按 `activities/`、`combat/`、`device/`、`runtime/`、`scheduling/`、`vision/`、`entrypoints/` 和 `tools/` 对应分组。统一历史图像仍保留在本地 `tests/fixtures/`，不随源码发布。

开发先执行 `pip install -e .`。Worker 使用 `python -m ninja3_assistant`，工具从项目根目录使用 `python -m tools.<分类>.<工具名>`；不依赖 `sys.path` 修改或根目录兼容副本。业务代码通过 `ninja3_assistant.paths` 定位资源，不能按业务模块所在目录推算根目录。

`assets/`、`configs/` 的公开清单与示例保留在根目录。游戏派生素材、模型和实际配置只在本地。`build/`、`dist/`、`data/` 继续分别存放构建缓存、完整发布包和运行/实验记录，并由 Git 忽略。本次整理不搬动历史素材或录像，不改其模型来源。
