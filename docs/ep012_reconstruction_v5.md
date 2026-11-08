# Terra 主导闭环重建 v5

基于 main `6877a0c6691daa217d2ce1335b6a1309169d53f7`，继承 v4 原始模板、源资产相对参数、原始相机先验和已选相机、完整原生 flex 保存初态。新任务组 `ep012_reconstruction_v5_20261108`，仅0/1/2，完整输出在 `runs/ep012_reconstruction_v5_20261108/`。v4及full402历史和账本不改，full402保持STOP。没有训练、策略推理或测试。

新入口 `terra_closed_loop_v5`。真实 Responses 结构化计划 → stdio MCP →数值搜索暂存候选→真实图及分项反馈→Terra keep/rollback/换方案。数值优化器不提供审阅判断，工具成功不等于质量通过。issue_id稳定；类别及问题按依赖指纹失效。每条20次实际HTTP请求（含错误、重试和终审），总渲染4500，投影10000，活动计算7200秒；新子池100初值/900几何/2400外观/600诊断/500导出验收。保留最终请求及1200秒最终计算资源；预算恢复不清零。

MCP统一契约：inspect_scene、diagnose_issue、optimize_issue、fit_appearance、select_candidate。拟合绑定issue_id、objective_id、参数组、边界、局部渲染上限。返回completed/no_improvement/budget_exhausted/capability_gap/error、实际候选/渲染/缓存数、前后指标、暂存/保留状态和剩余预算。重复同一已完成搜索本地拦截；geometry和camera在外观拟合中冻结。

详细输入、阶段权限、候选选择语义及实际记录入口见 [MCP v5 契约](mcp_v5_contract.md)。同一图像未变化并不意味着没有新的搜索反馈：外观硬问题仍为 open/needs_evidence 时，不能仅因连续两次 needs_review 就跳过下一轮决策。真实盘形等同一证据反复不确定的项目可延后，仍不记通过，且最终完整审阅不可省略。

独立苹果/盘子原生Phong specular、shininess；网格高度UV红黄色和边界；共享主光方向/强度。沿用MuJoCo3.7原生EGL，不声称PBR roughness/metallic生效。修复迁移时必须保留主光默认specular=.3（不是.5）。真实场景参数扰动和导出重载验证记录在每条parameter_perturbations/。可选程序化桌布纹理采用表面UV，静态网格和动态flex共用映射，不改变接触几何。

评价保持整体Lab/RGB对照，新增固定真实区域红黄颜色/分布、高光软权重、盘内中央/内壁/盘沿/方向扇区和径向亮度梯度。评价区不按候选掩膜缩小。新指标用于优化诊断，没有新增任意外观硬门槛；高光confidence<.25时效果表显示证据不足，这不是质量通过门槛。允许预先声明scene_lighting_v5联合目标后明确权衡，各对象的退化必须报告；既有硬门槛保持。

桌布细纹理/细褶皱为non_blocking_limitation，有限80次可选优化渲染，不单独否决。覆盖、整体颜色、位置及真实支撑仍必要。此政策同时进入分类/终审提示词、结构化问题和机器汇总。它不代表真实褶皱已恢复。

完整五类首次审查和本轮最终验收不可跳过。首轮按观测/几何/初始支撑/外观/静置完成，随后对未解决或失效项反馈。系统、分类、修复、候选对比、终审五模板均按实际请求记录哈希。每条首次搜索后必须真实反馈，保留候选前后图、指标、回滚理由。

最终导出后由独立进程执行accept_reconstruction_v5.py：复制便携XML/assets/state至独立目录，重新渲染/测量几何、覆盖、外观、支撑、重载；重新真实积分固定1秒原生flex，保留准确加载初态qvel，不选终点、不固定物体、不在步进中清零。终态10mm/10deg/2mm，峰值与速度仅诊断。新验收证据提供给计费的最终只读Terra请求，绑定最终文件哈希。

```bash
.venv/bin/python -B scripts/run_full_reconstruction.py start --config configs/ep012_reconstruction_v5.json
.venv/bin/python -B scripts/run_full_reconstruction.py status --config configs/ep012_reconstruction_v5.json
.venv/bin/python -B scripts/run_full_reconstruction.py resume --config configs/ep012_reconstruction_v5.json
.venv/bin/python -B scripts/publish_reconstruction_code.py --config configs/ep012_reconstruction_v5.json --reason '证据与修复说明'
.venv/bin/python -B scripts/accept_reconstruction_v5.py --run runs/ep012_reconstruction_v5_20261108/episode_000000
.venv/bin/python -B scripts/report_reconstruction_v5.py --package
```

独立复验可再次运行但不能代替本轮最终审阅所绑定的首次新验收。终审后保持交付文件冻结；补测写入独立reacceptance目录，预算继续累计。

本轮相关重建回归已实际执行：74 passed（50.65秒），见运行目录 `verification/pytest_feedback_fix.log` 与 `test_receipt.json`。包括真实原生渲染与导出、实际 stdio MCP 暂存/选择、静置终态与峰值、独立支撑阈值、预算恢复/终审预留、依赖失效、空搜索与无改善、可选桌布政策、同相机初始化缓存、局部对照图和未解决外观的后续反馈。没有策略测试。

本轮三条已完成实际新优化、新导出、新进程落盘验收、新原生4000步/1秒积分和最终只读Terra审阅；协调器及worker已退出。统一入口 [本轮 index.html](../runs/ep012_reconstruction_v5_20261108/index.html)，[三条效果表](../runs/ep012_reconstruction_v5_20261108/rerun_effects_summary.csv)，[完整分项对比](../runs/ep012_reconstruction_v5_20261108/photometric_effect_details.csv)，[最终复验记录](../runs/ep012_reconstruction_v5_20261108/final_reacceptance.json)。执行完成不等于质量通过。

| Episode | 构建 | 几何数值 | 覆盖 | 外观 | 初始支撑 | 静置终态 | 重载 | Terra综合终审 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 完成 | 通过 | 通过 | 失败 | 通过 | 通过 | 通过 | fail |
| 1 | 完成 | 通过 | 通过 | 待复核 | 通过 | 通过 | 通过 | needs_review |
| 2 | 完成 | 通过 | 通过 | 待复核 | 通过 | 通过 | 通过 | needs_review |

0号Terra几何项另保留真实盘内形状待复核；1/2号外轮廓几何为pass_with_limitations。三条苹果仍有偏粉/品红或分布残差，盘内明暗边界仍偏强，桌布整体颜色仍有误差。细桌布纹理和褶皱单独记non_blocking_limitation；没有把这项政策变动写成图像改善或真实褶皱恢复。

| Episode | 苹果颜色目标 v4→v5 | 盘内综合亮度目标 v4→v5 | 顶部桌布 IoU v4→v5 | 结论 |
| --- | --- | --- | --- | --- |
| 0 | .141877→.049189 | .017219→.013703 | .875310→.926456 | 颜色、综合亮度和覆盖改善，边界/盘沿局部退化 |
| 1 | .141620→.036318 | .016483→.013091 | .882630→.882630 | 颜色和综合亮度改善，覆盖持平，边界/盘沿局部退化 |
| 2 | .139299→.039948 | .020420→.018226 | .882279→.882279 | 颜色和综合亮度改善，覆盖持平，边界/盘沿局部退化 |

误差越低越好，IoU越高越好。使用相同真实图、分辨率、固定真实评价区域和同一评价器；v4新增指标来自冻结图片补测，不覆盖历史。三条可信高光证据均弱（置信度约.167/.211/.184），效果结论为证据不足。完整梯度曲线显示最大峰主要位于外盘沿；`plate_region_breakdown.json`另外按原有central/wall/rim区域区分内侧和盘沿，不改任何原分数、权重或门槛。内侧最大相邻梯度v4→v5为4.682→6.779、4.148→6.219、4.735→5.270，均有退化，不能用综合亮度误差下降掩盖。也不能据此断言盘子倒置或真实盘形已知。

三条新物理终态，按苹果/盘子分别为：0号7.039/5.647mm、9.061/4.999°、.016/.331mm穿透；1号6.879/5.798mm、8.199/5.032°、.012/.343mm；2号7.290/5.718mm、8.672/5.050°、.013/.319mm。0号相对v4位移有小幅退化仍通过，1/2物理指标与v4持平；各条全过程峰值及末段速度完整保存，未追加速度门槛，也未使用峰值否决终态通过。新产物的静/动态XML、资产、状态、尺寸、相机、真实/初始/v4/v5图片、曲线、视频、时间序列、日志和模型原响应都在各episode目录。

| Episode | 实际HTTP | MCP调用 | 保留修改 | 累计原生渲染评估 | 活动计算秒 |
| --- | --- | --- | --- | --- | --- |
| 0 | 20 | 106 | 4 | 1459 | 1827.38 |
| 1 | 20 | 88 | 8 | 1360 | 1517.02 |
| 2 | 20 | 86 | 5 | 1437 | 1654.44 |

60次HTTP中59次完成有效响应，2号一次APIConnectionError无服务响应但照常计入；SDK隐式重试关闭，恢复后第19次反馈与第20次终审成功。280次MCP中273次正常返回，另有6次安全STOP和1次历史no_pending_candidate错误；保留17次修改不是17次模型请求或全部内部候选数。实际搜索候选、缓存、诊断、无效/空调用及中断后已完成的评估分别见`execution_audit.json`和`optimization_execution_audit.json`。最终已结算账本为`final_budget_usage.json`及SQLite备份；冻结验收JSON的budget_used是在活动计算预留期间拍摄的快照，可能包含未用预留，保留原件并单列最终结算值。

0号最终优化/模型审阅快照`e84011081429904e`，1/2为`76241a32419a6768`，各阶段更早快照与连续修复历史也保留。0号完成后修复了同相机初始化缓存、局部对照图生成和open外观问题的反馈调度。缓存错误实际只触发于2号，0号历史不存在后者延后条件；0号最终文件未改，并用最新代码再次独立完整验收：新渲染、几何、终态及峰值与原终审候选完全一致，重载RGB/掩膜差均0，没有追加第21次请求。证据见0号`reacceptance/final/equivalence_to_bound_acceptance.json`及根目录修复记录。

312项受保护历史文件/账本哈希全部未变，full402仍为STOP/stopped、0运行、历史54条完成。无Movens-Interact写入、策略推理、训练或策略测试。最终代码和三条产物包及完整执行快照见`packages/`与`package_manifest.json`；完整补丁包含新增文件，工作树未强制回退、未暂存或提交。
