# Episode 0、1、2 分类重建与独立验收

基于 `batch/full402-h0-v1` 的 `b387a341024228f70013211c700ff9e089b8b7b0`，使用独立 `classified_reconstruction_v4` 工作流。配置为 `configs/ep012_reconstruction_v4.json`，任务组为 `ep012_reconstruction_v4_20261107`，结果位于 `runs/ep012_reconstruction_v4_20261107/`。任务清单严格为 0、1、2；先贯通 0，再最多两个 worker 处理 1、2，API 并发为一。402 条旧任务保持停止。

## 运行和恢复

```bash
.venv/bin/python -B scripts/run_full_reconstruction.py start --config configs/ep012_reconstruction_v4.json
.venv/bin/python -B scripts/run_full_reconstruction.py status --config configs/ep012_reconstruction_v4.json
.venv/bin/python -B scripts/run_full_reconstruction.py stop --config configs/ep012_reconstruction_v4.json
.venv/bin/python -B scripts/run_full_reconstruction.py resume --config configs/ep012_reconstruction_v4.json
.venv/bin/python -B scripts/accept_reconstruction.py --config configs/ep012_reconstruction_v4.json --episodes 0 1 2
```

后台执行使用 POSIX 独立会话、DEVNULL 输入和持久化日志。`job.json`、`heartbeat.json`、`logs/`、`session.json` 和 SQLite 为恢复依据。代码修复通过 `scripts/publish_reconstruction_code.py --config configs/ep012_reconstruction_v4.json --reason '具体修复证据'` 发布新快照；完成候选保持原代码和产物绑定。不得清空或迁移已花费预算。

## 统一状态与工具

`SceneBackend` 的资产倍率始终相对于不可变训练资产，重复应用相同参数不会叠乘。苹果初始 1.0，已有 scale110/XY1.05/Z1.10 与形状处理不重做。原始视觉尺寸约 86.625×86.212×90.593 mm；最终还记录物体坐标系视觉/碰撞尺寸和世界包围尺寸。每条复用已溯源 FK 初值、观测机器人相机与接触高度求解，额外苹果补偿和额外桌面抬高均为零。软层顶部 h0，硬桌顶部 h0−6 mm。

新增/恢复工具：

| 工具 | 真实操作 |
|---|---|
| inspect_environment | 检查分割、覆盖、射线与 h0 平面交点、桌面和物理支撑范围 |
| update_environment / fit_geometry | 绝对桌面深度/宽度/前沿与视觉桌布延展；视觉和碰撞桌面同步，软层区域不扩大 |
| review_camera_hypothesis | 对比平移、旋转、FOV 候选及两个物体重拟合；保留对照与回滚；非相机标定 |
| render_auxiliary | 独立副本侧视、斜视、中性光照图和法线统计 |
| diagnose_support | 原生 flex、接触身份/距离/力、速度、软层形变与固定一秒终态 |
| repair_support | 有证据的初始姿态比较、支撑求解、XY/尺度重拟合及复测；不固定物体或修改积分过程 |
| update_shape_or_asset | 盘子深度与有证据的小倾角；不任意更换苹果资产 |
| update_appearance / fit_appearance | 执行颜色、纹理混色、Phong 材质与光照拟合，冻结当前几何 |

Python、优化器和 stdio MCP 共用 `ReconstructionSession`。工具 schema 从同一注册和当前阶段生成；操作证据、前后参数、调用结果与复测存入 `mcp_plans/`、`mcp_tool_calls.jsonl`，本地工具调用独立记账。

## 分类、阈值和预算

五类依次为观测/相机/全场景覆盖、物体几何、初始支撑、外观、静置。每类预留首次和必要的修复后复查，后续只查询失败、证据不足或依赖改变的项目。通过状态带场景版本与依赖指纹；不能沿用失效的布面、尺度、相机或接触结论。几何失败不阻止独立外观/物理证据收集。请求缓存区分类别和证据版本。

每条最多二十次实际 Terra 请求，包括失败和重试；SDK 重试为零，至少保留一次最后只读审阅。初始/修复/外观搜索分别最多 600/2400/800 次渲染，诊断 600、导出复验 100，总计 4500；投影 10000，活动计算 7200 秒。旧四轮几何提前退出不用于分类调度。SQLite 账本与 task ID 绑定，恢复、回滚和代码修复不退款；最终只读后的独立复验可使用预留诊断/导出资源，不能重新搜索。

静置以精确加载状态、积分前为参考，恰好一秒：终态位移 ≤10 mm、终态转角 ≤10°、终态相关接触最大穿透 ≤2 mm。全程峰值、末段速度和软层持续运动只报告，不增加速度硬门槛。初始支撑独立使用 −1.2～+2.4 mm；苹果/盘子相互穿透 ≤1.2 mm。物体中心和边缘 ≤3.6 px，面积/轴误差 ≤6%，IoU ≥0.88。绝对桌布/顶部 IoU 为 0.964/0.88，错误背景比例 ≤0.012；相对未退化不能代替绝对正确。

`thresholds` 报告列出全部旧值、新值和依据。IoU 按 `1−1.2×(1−旧下限)`；其余质量误差放宽 1.2，静置独立放宽 2。资产、数值、保存重载和预算工程条件保持原要求；外观没有新造数值硬门槛。`improvement_attribution.json` 区分实际指标改善与新阈值/终态口径带来的判定变化。

## 最终落盘验收

`accept_from_disk` 从静态和动态便携目录复制到独立临时目录加载，检查资产闭包、状态、相机、尺寸、RGB/分割重载一致性和初始支撑。动态测试必须包含原生 flex 和保存的节点状态；不调用静态导出器替代动态模型。模型加载后保留保存速度，积分期间不清零、不固定物体，不选择有利终点。

最终 Terra 使用重载全图、同窗口局部、轮廓、辅助视图、原生接触/终态记录和独立验收报告。最终审阅绑定 XML、资产、保存状态、图像和诊断哈希，之后冻结。独立复验写入 `reacceptance/`，不覆盖冻结证据，也不重新请求模型。失败样本同样保留候选、原因、完整产物和预算，不能省略。

## 回归

相关重建测试集首轮合并结果：48 passed。覆盖终态与峰值差异、独立支撑阈值、二十次失败/重试预算和恢复、检查失效与跳过、真实 stdio MCP 更新进入便携导出、阶段能力一致性、不可将视觉延展当作物理支撑。测试不包括策略推理或训练。

## 实际修复与保留记录

原生接触诊断使用 `mj_id2name` 读取 flex 身份，并以原生 flex 回归验证一秒时长和保存速度。数值修复包括：收敛的早期坐标不再阻止后续倾角自由度；每个相机/倾角候选先联合投影重算 XY 和绝对尺度；桌布平面候选跨越可见性平台并按绝对覆盖门槛选取；几何改变后重建当前外观冻结上下文。苹果黄色按源网格 UV 对应的实际高度烘焙，替代不对应物体高度的图像行渐变；画面实现哈希进入场景和缓存身份。

episode 0 曾因恢复扫描错误提前形成失败终审候选，完整保存在 `episode_000000/revisions/failed_terminal_90f9d8bdd5b1250cad66/`。后继修复在同一 task ID、同一 SQLite 账本和二十次总上限内追加 `repair_revision` 事件；旧终审、旧产物和已花费请求均保留，没有退款或增加额度。原候选和最终候选均可独立重载。

重验另行导出 `camera_verified_from_xml.json`，其中角度和内参从实际 XML/保存状态求得；旧先验字典中的派生角度与真实投影参数分开记录。`reacceptance/history_*` 保留先前独立检查，`reacceptance/final` 是最终统一复验。

截至交付前的相关测试集为 56 passed；真实 stdio MCP 的外观声明、阶段权限与独立外观执行条件一致。实际各样本结果、失败项目和预算以统一入口、最终重验报告和原始 Terra 响应为准。

## 历史能力调查与接回依据

| 调查对象 | 原有脚本能力 | 仍依赖先验或人工的部分 | 本轮处理 |
|---|---|---|---|
| `scene.py` 的场景准备、`JointScene` 注册/应用/导出 | 已有视觉桌布环、桌面深度/宽度/前沿仿射变换、颜色和灯光参数；不是全部依靠人工改 XML | 扩展上限、灰度和名义盘径是预设；真实褶皱没有资产 | 移植成 `environment.py` 的不可变基准变换，共用 backend/session，MCP 实际更新并进入静态/动态导出；软支撑范围单独保留 |
| `head_camera.py` 的候选生成、`select_candidates()`、`rest_diagnostic()` | 已有多 mount/FOV 候选、数值选择、侧斜视、便携重载和一秒诊断 | 候选集/近似 OAK 安装、固定根位姿和42°来自配置与历史选择，非实物标定；仅凭模拟法线不能确定真实盘形 | 复用机器人 FK 先验；在限定边界内逐候选联合重拟合物体，比较/回滚，记录单视图歧义；静置改为独立落盘原生 flex 终态检查 |
| `full_prepare.py` 的 `fk_h0` | 已有机器人状态/FK布局、接触高度、训练资产核验、初始倍率1；原 full402 明确关闭桌布/桌面拟合 | 旧缓存 estimate 的具体姿态/光学偏移仍是先验，未经测量 | 保留原始估计及来源、逐条相机 FK；有证据地比较直立姿态并联合优化 XY/两物体尺度，不锁盘子倍率 |
| `workflow.py` 的工具路由和外观阶段 | 已有材质/光照搜索；`update_appearance` 被写入 `deferred_appearance_actions` | 全局几何/Terra门禁失败会阻断外观；已保存的建议未由此队列实际执行；旧几何4轮及停止条件截断分类证据 | 新分类调度独立执行外观工具并复测；任何类别失败不阻断其他独立诊断；保留受影响检查失效规则 |
| 观测分割与历史报告 | HSV/连通域、FK关联、遮挡掩码及覆盖指标由脚本生成 | `mask_review` 中部分说明由代码/配置写入，不能作为独立人工审核证据；刚性桌边和真实盘内轮廓未可靠标注 | Terra 接收分割预览、轮廓、真实/模拟全图与局部；明确 automatic/unreviewed、内轮廓低置信；不把旧“未退化”当全场景通过 |

旧54条全部初始支撑/重载通过、50条物体数值通过，而全景与静置失败，与上述固定范围、倾倒姿态、全局门禁及旧峰值口径共同相关。不能只靠改阈值解释：原三个苹果终态在新阈值下仍失败。完整比较见每条 `improvement_attribution.json`，旧报告、输入与账本由 `history_protection.json` 及最终哈希审计保护。

冻结配置继承了未使用的 `diagnostic.initial_velocities=zero_once` 旧标签。v4 的 `rest_from_disk()` 从保存状态直接恢复 qvel，完全不读取该标签；真实执行策略、加载速度和无重置证据写入 `physical_diagnostic.json`，非零保存速度回归覆盖该行为。为了保持运行配置身份，交付没有重写已冻结配置。

## 最终结果与复现命令

三条均完成构建、物体数值几何、初始支撑、终态静置和保存重载。episode 1、2 的全场景覆盖与 Terra 几何通过；episode 0 顶部桌布 IoU 为0.875310，低于0.88，盘内结构为 needs_review。三条外观和 Terra 综合终审均失败，原因是苹果色彩/高光、桌布真实褶皱纹理及盘内明暗仍不匹配。失败候选、原始响应和产物完整交付，不宣称全质量通过。

每条用满20次实际请求，合计60次；没有独立连通测试或额外终审请求。三条构建结束后协调器自然退出、运行数为零，full402 的 STOP 与旧结果不变。实际 worker 的 `run_identity.json`、运行快照、`code_continuations.jsonl`/代码过渡记录与最终工作树哈希区分“执行版本”和“交付版本”。

实际最终验证与打包入口：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_reconstruction_v4.py tests/test_reconstruction_v3.py tests/test_reconstruction_flow.py tests/test_full_batch.py tests/test_head_camera.py tests/test_correctness.py
.venv/bin/python -B scripts/accept_reconstruction.py --config configs/ep012_reconstruction_v4.json --episodes 0 1 2
.venv/bin/python -B scripts/audit_reconstruction_delivery.py --config configs/ep012_reconstruction_v4.json
.venv/bin/python -B scripts/summarize_reconstruction.py --config configs/ep012_reconstruction_v4.json
.venv/bin/python -B scripts/package_reconstruction.py --config configs/ep012_reconstruction_v4.json
```

统一入口 `runs/ep012_reconstruction_v4_20261107/index.html`；独立复验 `final_reacceptance.json`；完整性审计 `delivery_audit.json`；代码身份和包哈希 `code_delivery.json`；三条的 `DELIVERY.md` 提供实际尺寸、相机、终态/峰值、剩余问题及对应文件。`packages/modified_code.tar.gz` 包含修改后源码、配置、测试、文档与可应用补丁；三个 `episode_*.tar.gz` 包含便携场景、资产、状态、全部审阅/工具证据和独立复验结果。
