# 架构

WinUI 启动 Python Worker，Worker 执行任务计划。Plan 管理任务、进度、重试和优先级；Scheduler 只排序并串行调度；每个任务拥有自己的 `run`。

| 模块 | 职责 |
| --- | --- |
| `src/ninja3_assistant/scheduling/activity_task.py` | 活动任务接口与入口 |
| `src/ninja3_assistant/scheduling/task_planner.py` | 日常优先、后续目标、重试与对局恢复 |
| `src/ninja3_assistant/scheduling/task_scheduler.py` | 执行 Plan 提供的任务顺序 |
| `src/ninja3_assistant/game_ui/` | 页面观察、具名操作和菜单导航图 |
| `src/ninja3_assistant/runtime/realtime_battle.py` | 共用实时内核：新鲜帧、识别节奏、动作计划和撤回 |
| `src/ninja3_assistant/device/native_touch.py` | Python 到 C++ 的有界命令协议 |
| `native_touch/main.cpp` | 多手指触控执行、截止时间及释放 |
| `src/ninja3_assistant/activities/weekend/runner.py`、`src/ninja3_assistant/activities/weekend/obstacles.py`、`src/ninja3_assistant/activities/weekend/rope.py` | 竞速、正向避障及绳索处理 |

连续移动可提交最多 3 秒，点击计划使用更短的时间窗口；后续帧可替换或撤销。地面移动、应急跳跃、武器/卡牌按手指分配；暂停、停止和场景退出取消剩余动作。玩法策略不得另开相互竞争的输入循环。

运行代码位于 `src/ninja3_assistant/`，使用绝对包导入。`paths.py` 集中处理项目根目录与冻结程序资源位置；`packaging/` 集中维护入口和资源清单。测试按领域对应分组，工具与运行包分离。具体目录见 `DIRECTORY_LAYOUT.md`。

目录迁移不改变任务优先级、触控时序或玩法策略。数据与发布输出隔离，公开仓库不包含游戏派生素材。

## 游戏页面操作与导航

`GameUI.observe()` 返回当前页面 ID、已确认的控件和采集代次。活动根据识别结果选择具名操作，由 `GameUI.perform(observation, name)` 核对新帧、动作时限和运行状态后提交 NinjaTouch。购买、领取、编队和开战的条件与结果确认由各活动管理。高频战斗与抢队继续使用已有的实时输入机制。

`GameUI.plan_route(source, target)` 查询已登记的最短菜单路径。`GameUI.navigate(target)` 每次执行一个安全跳转，观察落点后重新规划，并在观察到目标页面时报告到达。抢队、购买、领奖和保存等需要活动主动选择的控件由活动执行。页面 ID 和跳转登记在 `game_ui/pages.py`。

需要读取业务状态时，可以传页面类型取得同一会话中的页面对象。页面在 `game_ui/` 中定义识别证据、只读字段和操作。活动决定何时执行动作。小屋页也供每日清理使用。

```python
from ninja3_assistant.game_ui.house import HousePage

house = ui.navigate(HousePage)
if house.can_collect_fragments:
    house.collect_fragments()

# 确认弹窗出现后读取当前页。
confirm = ui.read_page(HousePage)
if confirm and confirm.can_continue_training:
    confirm.continue_training()
```

`house.training_remaining` 是该帧确认的剩余秒数。控件匹配优先使用局部画面和前景掩码，动作提交前复核新帧中的目标及相关条件。未确认状态保留为未知。续练确认核对饭团费用和按钮，导航图仅负责进出小屋。

模拟器已经打开游戏时，可以通过 `assistant_control` 查看当前页面并单独验证菜单路径。服务只监听本机。导航与本版 Worker 按模拟器实例独占输入，旧版 Worker 需先退出。

```powershell
python -m tools.diagnostics.assistant_control --folder '<MuMu安装目录>' --index 1 --serve
python -m tools.diagnostics.assistant_control --command page
python -m tools.diagnostics.assistant_control --command navigate --page intelligence.tasks
python -m tools.diagnostics.assistant_control --command status
python -m tools.diagnostics.assistant_control --command exit
```

同样的命令可通过 JSON 行或 HTTP `POST /command` 发送。导航命令形如 `{"command":"navigate","page":"intelligence.tasks"}`。`status` 的 `navigation` 字段报告目标、状态、最后观察到的页面和已执行步骤。需要取消时发送 `stop`。

## 每日清理

`DailyTask` 的每个清理项目独立运行，共用 `activities/daily/vision.py` 的页面识别、`flow.py` 的无输入决策和 `runner.py` 的 SDK 取图及 C++ 执行。Plan 将清理放在活动日常之前，小屋到时后在安全检查点优先执行。当前限时签到支持十日签到和三千日答谢礼，活动轮换后需更新识别与入口，未确认状态保留未完成。

免费操作在提交前用新截图重查按钮中的免费标签，每次运行最多发送一次免费请求；结果页只点左侧确定。商店按忍币图标筛选，购买弹窗再确认货币；悬赏奖册买后会从列表移除，按商品图像识别当前位置。未知画面、库存或货币不计作完成。主动邀请开关控制自建悬赏队伍的好友和族员招募。自动接受开关控制所选等级的好友、族员邀请及跨任务安全交接。两项独立传给 Worker。截图、决策和原生输入事件保存在本地 `data/runtime/daily-cleanup/runs/`。

主页通过集会所与召唤锚点识别，按实际文字及图标定位可移动入口。首次说明页、邀请和奖励优先于背景页面。同一页面与等待原因持续 12 秒无进展时，限时签到入口缺失，以及已识别的试炼、竞技场和答题页面无法确认关键状态时返回 `DEFERRED_TODAY`，其他等待返回 `WAITING`。Plan 在内存中保留暂缓项目的未完成状态，运行跨过北京时间 05:00 时重新启用。等待期间每 5 秒保留证据。整帧与局部 OCR 结束后检查停止及暂停代次，恢复后重新取帧。

`resources.py` 按模拟器实例保存小屋结束时间。续练需5饭团，扫荡前预留 `max(0, 5 - floor(距结束秒数 / 1800))`，倒计时未知时预留5点。Plan 启动时读取倒计时，到期后把小屋排在其他可切换任务之前。对局先恢复到安全检查点，小屋异常冷却后恢复最高优先级。

通灵兽喂养根据已选奖励确认巡逻路线。竞技场奖励与老板娘答题各自有独立任务。答题在 `data/runtime/daily-cleanup/landlady-quiz.sqlite3` 保存游戏明确反馈的选项正误，以题干和选项文字匹配乱序答案。运行 `python -m tools.datasets.import_landlady_quiz '题库.csv'` 可更新本地基准表。CSV 使用 UTF-8 编码，表头为 `题目,答案`。游戏确认的正确答案优先，基准答案需匹配可见选项并通过错误记录筛选。读不到已知答案时在剩余选项中随机选择，随后继续记录游戏反馈。

`activities/bounty/daily_flow.py`、`daily_vision.py` 和 `daily_runner.py` 分别负责每日悬赏的菜单决策、页面识别和执行。普通悬赏按D/C/B/A单一等级清理3次及1次明确免费的购买次数；S召令作为独立日常清理当前最多7次累计奖励，未解锁和无召令状态不冒充完成。等级滑块仅传递所选普通等级，游戏逻辑不在WinUI中重复实现。

悬赏战斗通过 `combat.py` 复用 `RealtimeBattleKernel` 和C++多指计划：地面以两个控制按钮的白色轮廓识别并续接滑步，排除动态背景明暗；绳索撤回移动。Boss头像外框与Lv前缀出现、忍术控件可见时立即提交忍术，血条作为补充信号，不等待血条填充或移动按钮同时可见。忍术与左右武器使用独立手指，不等待能量达到100%。菜单才运行OCR；战斗中的未知过渡最多0.5秒保留已提交的有界计划，不续发或阻塞于OCR，超时再取消。确认邀请、结算、场景退出以及暂停、停止时及时取消输入。无可加入队伍时以等待状态交回Plan稍后重试，继续其他日常；结算后读取真实剩余次数。

## 情报模拟战选牌

`activities/intelligence/policy.py` 使用卡库费用、基础攻击、属性、职业及已支持的效果，在当前金币和空位限制内搜索多张商店卡的购买顺序与站位。每个可支付的商店子集保留 3 个站位候选，避免低攻击的特殊胜利组件被普通伤害方案全部挤掉。比较到第 5 回合的估算伤害、上场伤害、两回合增益及剩余金币伤害；相邻加成按上下左右计算。支持五虎、武器、火系、水系同行、雷系门槛及行列联动。每个商店位置最多购买一次；每次执行一步，新截图中的棋盘、商店和金币与预期一致时续接原组合的位置，状态变化时废弃余下方案并重新规划。暂停、换回合、刷新和对局结束清空组合。续接日志单独标记原组合估值，不将其当成剩余方案的新估值。

经济牌计入黑卡固定收入和御守的属性种类收入。后续收入不能支付当前回合购牌，末回合结束的收入没有购牌收益。未来金币按当前商店的中位攻击/费用估算再投资价值（换算限制在 0.5–2），受空位乘商店中位费用的容量限制。空位少于后续购牌轮次时，额外估算保留现有金币的机会价值，避免只比较眼前正伤害便填满最后空位；末回合不再计这项未来收益。未来投资金额从现有金币与已知后续收入的同一预算中扣除，剩余金额才计老板娘伤害，再与全存币方案取较高者，避免一份金币双算。没有假定未来商店的具体牌或固定基础补助。日志记录保留金币、空位、持币机会估值和空位稀缺度；这些都是启发式估算。轮回戒的两次刷新仅在回合结束后产生，每次按 `0.5 × 攻击/金币换算 × 后续回合数` 估值，并限制在可用空位内。执行刷新依据实际可见计数，计数下降后才允许继续使用额外刷新。

鬼斩特殊胜利要求触发角色与三种不同 ID 的鬼斩铠甲同时在场；重复铠甲和只有铠甲没有触发角色均不算。可见且付得起的获胜组合优先于普通伤害；已经满足时不再购牌或刷新，保留阵容准备。

这是有界估算，不保证全局最优。已有临时增益和历史上场次数仍未追踪。日志中的 `estimated_gain` 是综合收益估值，不是胜率。`selection` 记录预算、花费、剩余金币、组合、站位、联动增益、每回合收入、后续收入、经济增益、回合结束产生的刷新、特殊胜利条件及估算参数；界面状态同步显示中文选择理由。

未知棋盘卡保留占位，未知商店卡不猜测身份；已识别的卡仍可购入。没有已知可购买卡时短暂重读一次，仍无法识别则使用可用的本回合刷新，或结束购牌，避免无限等待。

空卡背增加灰度图案与边缘轮廓的联合匹配，不能仅凭没有认出卡牌就判空。深色卡背及绿色加号卡背的新增变体要求灰度与对比度归一化后的轮廓在同一位置对齐，独立的小号加号模板用于确认可放位置。头像被攻击/伤害提示遮住时，使用卡名、候选间隔和至少两块空间对齐的头像局部共同确认。只有卡名、只有头像或裁剪严重错位时继续保留未知。真实未知样本的人工复核及图像留在本地证据目录，回归同时检查真卡、空卡背、空白、噪声和姓名/头像冲突。

每次运行目录的 `unknown-cards/` 保存未知卡整帧、裁剪和 `index.jsonl`，同时向 `events.jsonl` 写入 `unknown_cards` 事件，包含位置、轮次、选中状态及前三个画面匹配候选。候选分数不等于识别确认。按缩略图去重，每个位置/轮次最多保存 3 个变体，每次运行最多 128 个裁剪。这些文件留在本地，不进入公开仓库。
