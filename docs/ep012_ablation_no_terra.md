# 无 Terra 消融：现有流水线、原计算上限

基准 HEAD `a21409730c0bac65ba5aac4b81f2e630a5314f51`，初始工作树干净。独立工作流 `ablation_no_terra_v5`、配置 `configs/ep012_ablation_no_terra.json`、任务组/输出 `ep012_ablation_no_terra_20261108`，仅0/1/2。A是冻结的 `runs/ep012_reconstruction_v5_20261108`；B不读取A最终参数、最优候选或模型计划来做搜索。工具与共同v4热启动包含历史模型协助，本实验不是无任何历史模型影响的从零重建。

从A真实resolved_config复制所有本地上限、初值迁移、渲染、目标、参数边界、验收口径。每条4500总渲染、100初值/900几何/2400外观/600诊断/500导出，10000投影、7200秒活动计算，默认局部128、最大256，可选布纹理80，最终计算预留1200秒。HTTP严格0。`budget_comparison.json/csv`证明本地限制完全一致，恢复与工程修复累计同一账本。

## 固定调度与选择

用户已选择现有八组各一次。规则在0号运行前固定并保存哈希，三条共用，不按episode ID调整。先用已有诊断测量覆盖、轮廓、初始支撑和原生静置；覆盖失败先table_coverage，仍失败才camera_coverage；轮廓数值失败才object_layout；支撑/终态失败才support_initialization。每组至多一次，未解决项目保留；缺乏真实盘内标注，不据明暗推断盘形、姿态或倒置。

外观固定顺序：apple_color → apple_red → apple_yellow → apple_highlight → plate_material → shared_lighting → plate_and_lighting → cloth_color。外观阶段相机/几何固定，无额外回绕。可选细桌布纹理拟合默认跳过，能力与80次上限保留，记non_blocking_limitation，不豁免整体颜色、覆盖或物理支撑。

每步通过真实stdio MCP执行原有种子初筛、有界Powell和候选选择。Powell仍maxiter8、xtol.003、ftol1e-4、maxfev为原局部cap扣除种子后的余额；投影重拟合原45次、容差1e-7，不启用旧JSON中的projection_starts、appearance.subsets等非v5调度字段。数值搜索内核未改，见implementation_comparison.json。

保留条件是既有目标下降超过.0001且通过v5实际在用的guard_failures/gate，不新增外观阈值或改损失权重。共享灯光只用scene_lighting_v5，允许显式记录的分项退化，不能因某一对象的改善掩盖另一个对象退化。所有搜索暂存后通过select_candidate真正进入会话状态。几何操作B使用evidence中的数值来源、当前候选/观测身份与实际失败项，由MCP服务端重算核验；不伪造模型置信度，A原置信度路径保持。

持久步骤记录覆盖search_started、selection_started和终态。完成步骤不重跑；恢复待选候选的选择。如果中断搜索没有完整工具结果，保留已花成本和历史，保守终止该步骤，不重新授予128次预算或更换ID重放。导出/验收保留31次诊断和最多8次导出渲染余额，均在原池内，不重新分池。真实几何变化使相关图像/物理证据失效，最终验收永远从新产物重做。

## 模型禁用与验收

统一和遗留模型入口在凭据、缓存、假离线审阅、客户端和HTTP之前明确拒绝本工作流；账本也拒绝所有模型预留。不是靠缺少密钥关闭，不复制旧审阅作为B判断。结果中审阅=not_run_ablation，appearance_passed和visual_qualified为null；没有外观合格线时只报告指标改善/退化。

复用v5.initialize：源模板、v4绝对参数、已选相机、原生动态状态、v5默认材质迁移作用一次。共同初值逐条核验图像、掩膜、资产、参数、相机、qpos/qvel、模拟时间和种子。额外苹果补偿/桌面抬高0；软顶h0、硬顶h0−6mm；源苹果倍率1.0，继承v4源相对已选倍率。

最终由独立进程加载本轮便携静态/动态XML、资产和保存状态，重新渲染并测量几何、覆盖、外观、支撑、重载；重新执行固定1秒4000步原生flex积分。起点为准确加载初态，保留qvel，不固定物体、不清零、不把终态当新初态。既有IoU/像素/支撑/终态等门槛保持。峰值、末段速度只诊断。静态与动态场景分别交付。

## 执行与恢复

```bash
.venv/bin/python -B scripts/report_ablation_no_terra.py --prepare
.venv/bin/python -B scripts/run_full_reconstruction.py start --config configs/ep012_ablation_no_terra.json
.venv/bin/python -B scripts/run_full_reconstruction.py status --config configs/ep012_ablation_no_terra.json
.venv/bin/python -B scripts/run_full_reconstruction.py stop --config configs/ep012_ablation_no_terra.json
.venv/bin/python -B scripts/run_full_reconstruction.py resume --config configs/ep012_ablation_no_terra.json
.venv/bin/python -B scripts/report_ablation_no_terra.py --package
```

先0后1/2，最多两个worker，沿用后台协调器；PID/日志/恢复入口在job.json与logs/。独立复验可用原accept_reconstruction_v5.py --run新B目录，输出另一个reacceptance目录并继续累计预算，不能替代第一次新落盘验收。

统一index.html、ablation_summary.csv和ablation_metrics.csv展示真实/共同初态/A/B、红黄分布、高光弱证据、盘内中央/壁/沿及内外梯度、布色/覆盖、物理/重载、全部实际成本。A补测只读取冻结图，放comparison/A_v5下并单列资源；不优化A、不修改历史账本。最终结果和失败以实际产物为准，三条不足以推断402整体。402继续STOP；无策略推理、训练、策略测试或Movens-Interact写入。

## 本轮实际完成结果

三条均在2026-11-08实际重新搜索、选择候选、导出并在独立进程重新验收；没有运行时模型请求或旧模型响应复用。协调器06:31:04Z启动，先0，再1/2并行，07:08:20Z退出；三条均首次实际运行完成，没有重启或重复搜索。执行代码身份为 `fc23c24603eb7487cb15d78829ea709f244f2efb72c9fb0a9e05e18914459981`，完整源码快照和每条实际命令、PID、开始/结束时间均保留。

每条最终验收从本轮portable XML、本地资产和保存状态加载；独立验收PID依次393054、465970、466339。三条均重新渲染并真实执行4000步、固定1秒原生flex积分，未用旧物理报告替代。几何、覆盖、初始支撑、终态静置和重载全部通过，静态重载RGB MAE均0。重新计算的完整物体物理指标与A相同；峰值依旧作为诊断，不能用旧峰值门槛否定本轮约定的终态通过。

| episode | 流程/新落盘验收 | 苹果颜色目标 A→B | 盘内综合目标 A→B | 实际渲染 A→B | 活动秒 A→B |
|---|---|---|---|---|---|
| 0 | 完成，数值/工程项通过 | .049189→.047216（改善4.01%） | .013703→.013114（改善4.30%） | 1459→978 | 1827.38→1134.83 |
| 1 | 完成，数值/工程项通过 | .036318→.039592（退化9.02%） | .013091→.014692（退化12.23%） | 1360→822 | 1517.02→957.62 |
| 2 | 完成，数值/工程项通过 | .039948→.039341（改善1.52%） | .018226→.015383（改善15.60%） | 1437→855 | 1654.44→978.99 |

以上目标越低越好，只代表对应数值目标。外观状态为 `measured_no_hard_threshold`，Terra审阅为 `not_run_ablation`，不设自动视觉通过。所有上限不变，B合计2655次渲染、3071.45秒活动计算；A已结算合计4256次、4998.85秒。A0包含完成后的额外39次渲染、82.26秒复验，原完成口径与后续结算分别报告；B没有额外物理复验。A只读图像补测成本单列在 `comparison/supplement_usage.json`，不计作B搜索。总经过时间包含A历史开发和暂停，不能把全部时间差归因于模型。

| episode | 盘中心绝对亮度误差 A→B | 内侧梯度误差 A→B | 其他结果与局限 |
|---|---|---|---|
| 0 | .071261→1.923894，退化 | 5.846649→6.534974，退化 | 盘沿改善；布色基本持平；覆盖相同，顶部IoU .926456 |
| 1 | .665256→4.310756，退化 | 5.312165→4.340173，改善 | 盘沿、布色改善；苹果和盘内综合目标退化；覆盖相同 |
| 2 | 2.633226→1.303710，改善 | 4.336826→7.400215，退化 | 盘沿改善，布色退化；覆盖相同 |

内侧梯度使用同一冻结盘子区域中r≤.8的既有亮度曲线，另列整盘/盘沿梯度，不能混淆。0号综合盘内目标下降并不代表中心亮度或内部边界变好。全部红黄颜色、面积/空间分布、高光、中央/壁/沿/方向扇区/梯度分项保留在 `ablation_metrics.csv`、每条 `A_B_metrics.json` 和曲线图中。高光真实证据置信度低，三条均报告证据不足；真实盘形仍未知。细布纹理/褶皱可选政策不变，没有用政策变化制造质量改善。

可比性审计发现：A0历史运行中的部分搜索使用过不同材质实现或数值搜索内核，不能证明是只切换Terra的纯消融。最终A冻结图与B新图已经使用同一参考图、掩膜、分辨率和评价器补测，因此最终误差可比较；0号差异单独归因Terra仍证据不足。A1/2实际搜索内核与B一致。逐版本搜索映射保留在 `comparison/implementation_history.json`，主CSV与首页均标记限制；没有重跑或改动A。

83项相关重建回归通过（79.33秒），另有12项管理器回归通过（1.81秒）；包括真实材质渲染、stdio MCP选择与导出、零模型入口拦截、预算/步骤恢复、未完成搜索不重授额度、旧v5路径兼容。日志在 `verification/`。最终执行审计确认12597项受保护历史文件/账本未变，402仍STOP、54条历史完成、0运行。

最终整理实际命令如下，均不调用优化或模型。前两步生成公共比较与执行审计，第三步补齐逐版本归因及工具轨迹，第四步打包并核验最终绑定文件。不要只重新生成公共报告而遗漏第三步的归因说明。

```bash
.venv/bin/python -B scripts/report_ablation_no_terra.py
.venv/bin/python -B runs/ep012_ablation_no_terra_20261108/verification/final_audit.py
.venv/bin/python -B runs/ep012_ablation_no_terra_20261108/verification/finish_reports.py
.venv/bin/python -B runs/ep012_ablation_no_terra_20261108/verification/package_delivery.py
```

结果入口为 `runs/ep012_ablation_no_terra_20261108/index.html`；每条目录含真实/共同初态/A/B全图与局部图、静态/动态portable场景、相机/参数、静置视频/时间序列、独立验收、真实MCP及候选选择轨迹。`package_manifest.json`、`package_checks.json`、`code_delivery.json`提供交付包、绑定文件及修改代码校验。现有最终资产已经烘焙选定变换，加载不得再次施加参数倍率。三条已完成后停止，不恢复拟合、不启动402或策略测试。
