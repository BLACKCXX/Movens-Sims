# v5 实际 MCP 契约与证据

入口是 `ReconstructionSession`，`mcp_server.make_server()` 在 v5 工作流转到 `v5_tools.make_server()`。执行器使用 MCP 2.3.0 stdio 客户端，启动当前运行快照中的真实服务器；Responses 返回结构化操作计划，本地执行器再调用工具。模型请求、MCP 调用、搜索内的候选评估分别记账。

| 工具 | 主要输入 | 真实行为 |
| --- | --- | --- |
| `inspect_scene` | 无 | 当前候选、统一参数 registry、可执行参数组、目标、待选候选及剩余预算 |
| `diagnose_issue` | `issue_id`, `candidate_id`, `diagnostic` | 最新度量；按问题提供覆盖、真实支撑范围、侧/斜视、镜面关闭、中性照明、深度导出法线或原生软层静置诊断 |
| `optimize_issue` | `issue_id`, `objective_id`, `parameter_group`, `render_cap`, `candidate_id`, 可选 `bounds/evidence/confidence` | 在注册目标与边界中执行候选初筛及有界 Powell；暂存候选，等待模型比较 |
| `fit_appearance` | 同上，仅允许外观参数组 | 固定相机、几何及初始动态状态，实际拟合颜色、材质、共享照明或可选桌布纹理 |
| `select_candidate` | `issue_id`, `decision=keep/rollback`, `candidate_id`, `evidence` | 接纳或回滚暂存候选，并使受影响检查失效；不回退预算 |

`candidate_id` 必须等于当前所选场景身份。选择操作中的 ID 指当前所选场景，待选候选从会话状态读取，不能用旧候选覆盖新场景。冻结及最终只读阶段仅注册检查和诊断工具，优化实现也单独拒绝冻结场景的写操作。

拟合返回 `execution_status`：`completed`、`no_improvement`、`budget_exhausted`、`capability_gap` 或 `error`，以及新增候选、真实新增渲染、缓存命中、是否改变参数、是否保留、前后指标、停止原因和剩余预算。`completed` 通常代表已开展有效搜索并暂存改善候选，不等于已选用或质量通过；只有随后实际成功的 `select_candidate(keep)` 才算保留修改。搜索错误通过 MCP 外层 `ok/error` 与调用日志保留。零预算及已完成的重复搜索本地拦截。

外观参数组包括 `apple_color`、`apple_red`、`apple_yellow`、`apple_highlight`、`plate_material`、`shared_lighting`、`plate_and_lighting`、`cloth_color`、`cloth_optional`。目标包括 `apple_color_v5`、`apple_highlight_v5`、`plate_shading_v5`、`cloth_coarse_v5`、`scene_lighting_v5`。公共灯光只能使用预先声明的 `scene_lighting_v5`，同时反馈苹果、盘子、桌布的全部分项变化。细桌布纹理最多 80 次局部渲染，不能改变碰撞支撑。

几何组包括桌布/桌面覆盖、受限相机与联合 XY/尺度、物体布局、盘子姿态/纵深、初始支撑。几何变化需证据及至少 0.7 的置信度；相机边界始终相对原始先验，而不是累加前一次修改：平移范数≤20mm、旋转≤5°、FOV 42°±2°。源训练苹果倍率为 1.0，继承的 v4 已选倍率是相对该源资产的绝对倍率。

每个优化结果需真实图、修改前后同一区域图片与固定评价器比较。保留已有硬门槛；退化须报告。没有可靠真实盘内标注时，中性光照或模拟侧视不能证明真实盘形或盘子倒置。低置信度高光为软诊断，不将亮黄色果皮自动标为镜面高光。

本轮实际 schema、参数上下限、阶段权限在每条 `parameter_registry.json` 和 MCP 执行记录中保存；实际操作在 `mcp_tool_calls.jsonl`、`mcp_plans/`、`operations/`、`candidate_decisions.jsonl`，模型收到的工具声明及模板哈希在 `terra/*/attempt_*/request.json`。这些运行记录优先于本文说明。

材质采用原生渲染器有效的颜色纹理、`specular`、`shininess` 和灯光。MuJoCo 3.7 文档说明原生渲染器不使用 PBR roughness/metallic，且局部 geom RGBA 会覆盖材质 RGBA；本轮同时以真实参数扰动、纹理色卡和落盘重载验证可见效果。[MuJoCo 3.7 XML 参考](https://mujoco.readthedocs.io/en/3.7.0/XMLreference.html#asset-material)

数值搜索使用项目固定的 SciPy 1.14.1 有界 Powell 接口，并叠加本地候选数、真实渲染、活动计算时限和最终验收预留。[SciPy 1.14.1 Powell 文档](https://docs.scipy.org/doc/scipy-1.14.1/reference/optimize.minimize-powell.html)
