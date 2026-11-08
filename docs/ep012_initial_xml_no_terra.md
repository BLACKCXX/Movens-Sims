# 初始 XML 到无 Terra 重建

基准 main eeda434fd5a0450f60670766e066d120588e5d4d，开始时工作区干净。本轮仅0/1/2，真实目标为各自frame0。工作流仍为ablation_no_terra_v5，新增initialization.source_mode=initial_xml；缺省继续采用旧v4热启动。mode=fk_h0仍控制顶部支撑/最近可行XY，不随来源字段变更。旧任务、账本和402的STOP不变。

## 输入和初始化

配置 configs/ep012_initial_xml_no_terra.json，三条manifest为configs/ep012_initial_xml_no_terra_manifest.json；本轮唯一任务组和目录在配置中冻结，恢复不得更名或清账本。使用原始full402来源的Movens-Interact episode_XXXXXX/offset_000mm/scene/scene.xml、真实视频、parquet观测、FK estimate、h0及哈希侧文件。只接受本项目已展开的苹果/盘子/桌子/机器人实体契约，不隐式读取其他旧场景。不可变训练资产和Mount参考复用静态共享资源。

三条没有权威完整原生保存状态，明确使用fresh XML reset，初始qvel=0、time=0。原始reference.npz是观测/控制参考，run/states.npz属于旧仿真，不作为新状态输入。可显式声明joint_name_state，必须含关节名、qpos/qvel地址与尺寸、时间，按名校验恢复；不按数组顺序套用。归一化和后续导出保留速度/时间，原生flex状态连续可追踪。

复用full_prepare：支撑装配从源位置平移至软顶h0、硬顶h0−6mm（当前源需−5mm），无额外桌面抬高或苹果偏移。物体位置来自原始FK，源尺度1.0，历史尺寸处理已烘焙。按碰撞几何求支撑Z，必要时使用已有最近可支撑XY/相交分离；与原始FK差异独立留档，不能称图像已经拟合。相机由真实frame0观测43关节和近似Mount先验重新计算，42°FOV；源XML自带69°相机仅保留作原始视图证据，不继承旧选定相机。重新执行自动分割/遮挡处理，保留低可信与失败原因；其他帧只做一致性检查。

V5独立Phong、UV配色和共享灯光默认值迁移一次，不乘历史v4材质倍率。source_xml_identity、source_xml/portable、initialization、initial_xml_verification、feasible_initial_parameters及初始图保留原始/规范化/可行初值。原始XML渲染使用原始相机和声明的fresh状态，不能当作规范化相机结果。

## 搜索与预算

有效物体观测且轮廓数值失败时，先执行已有投影种子/粗筛/三个局部Powell起点/全分辨率细化。仅六个XY/整体尺度参数，既有种子、损失、正则、容差和停止规则不变。该初始几何阶段含控制/提案渲染至多600次，计入既有900次render_repair；剩余repair池用于原来的四组修复，不动100次初值池。通过真实MCP fit_geometry适配器暂存候选，再经select_candidate保留/回滚。

随后覆盖失败→table_coverage→仍失败camera_coverage；物体残差失败→object_layout；初始支撑或终态失败→support_initialization。外观固定apple_color、apple_red、apple_yellow、apple_highlight、plate_material、shared_lighting、plate_and_lighting、cloth_color，各一次。默认不拟合可选细布纹理，不额外改盘形。保护已通过硬检查，不因其他既有失败而阻止独立有效目标搜索。

全部预算与上轮相同：4500渲染，100初始/900几何/2400外观/600诊断/500导出，10000投影、7200活动秒，局部默认128/最大256，可选布纹理80，最终1200活动秒及39总/31诊断/8导出渲染预留。预处理从开始即计费，与会话/子进程共享持久账本；嵌套活动计算不重复计费。初始几何与分组搜索保存原始局部额度和确定性目标值前缀；恢复只重建求解器状态，再执行尚未完成的调用，不重算已付费前缀、不重授搜索额度。实现/输入不一致时明确阻止隐式续跑。

冷启动进程含历史读取审计拦截；v4/v5/旧消融、head-camera和full402运行结果不能作为隐藏回退。统一/遗留模型入口在缓存、凭据和客户端之前拒绝，预算预留也拒绝；没有离线假审阅。旧热启动路径保留原行为。

## 执行和验收

```bash
.venv/bin/python -B scripts/report_ablation_no_terra.py --config configs/ep012_initial_xml_no_terra.json --prepare
.venv/bin/python -B scripts/run_full_reconstruction.py start --config configs/ep012_initial_xml_no_terra.json
.venv/bin/python -B scripts/run_full_reconstruction.py status --config configs/ep012_initial_xml_no_terra.json
.venv/bin/python -B scripts/run_full_reconstruction.py resume --config configs/ep012_initial_xml_no_terra.json
.venv/bin/python -B scripts/report_ablation_no_terra.py --config configs/ep012_initial_xml_no_terra.json --package
```

先0后1/2，最多两个worker；日志、PID、真实命令及恢复信息在新目录。导出后自动启动独立accept_reconstruction_v5.py进程，只从本轮portable XML/资产/状态定义模型，重新渲染和固定1秒4000步原生flex积分。额外独立复验必须另写输出目录，继续原账本，不能覆盖冻结产物。

数值门槛、原始先验相机边界、支撑/终态口径不变；峰值/末段速度只作诊断。外观没有新增硬线，appearance_passed/visual_qualified为null，审阅not_run_ablation。细布纹理和褶皱可选，不豁免覆盖、整体颜色或物理支撑。弱高光和真实盘形未知单列。

统一index.html、cold_start_summary.csv、cold_start_metrics.csv提供三条初始化/流程/数值/外观测量/支撑/静置/重载/审阅状态，原始—初始—最终图、局部图、动态视频、完整接触/时间序列、候选/工具轨迹、实际费用和零模型证据。失败和未完成阶段同样交付。历史对照只有在本轮冻结后才可选执行，并标记初始化不同，不作为同初值模型消融。完成三条后停止，不运行20/402或任何策略/训练测试。

## 执行身份与恢复保护补测

正式三条使用同一冻结实现，优化/导出/验收身份为8edf14aa520dc05f0f491180b39d52c045fa09f9b7e2bd185884749d6cef1313，完整快照随结果交付。正式启动后只在交付源码增加一个恢复保护：若未完成搜索/待选候选遇到实现身份变化，明确拒绝隐式换签名重新授予局部额度；可用原冻结实现恢复，或执行保留原额度的明确兼容维护。旧warm-start默认行为不变。

该保护不改变正常首次执行或任何渲染、优化目标、数值算法、物理、资产与验收。verification/control_guard_equivalence.json证明移除新增保护函数和代码身份变更分支中的保护调用后，整个ablation模块AST与正式执行快照一致；正式三条的执行身份不改标。保护及前缀恢复3项补测通过。相关回归累计96个不同测试用例通过，部分定向补测重复，分别保留原日志和早期工程失败记录。

## 本轮实际结果

目录 `runs/ep012_initial_xml_no_terra_20261108T080904Z/`。协调器2026-11-08T08:36:18Z启动，先0，再1/2，09:29:40Z完成退出；三个worker均首次执行完成，无工程重试。三条分别由独立进程562809、664000、655573重新加载本轮便携场景、资产和保存状态，真实执行固定1秒4000步原生flex积分。具体进程身份以每条final_acceptance/acceptance.json为准；最终文件绑定哈希不再修改。

| episode | 初始化/构建 | 几何 | 覆盖 | 初始支撑 | 静置终态 | 重载 | 渲染 | 投影 | 活动秒 | 总经过秒 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|
| 0 | 完成 | 通过 | 失败 | 通过 | 通过 | 通过 | 1339 | 2238 | 1437.28 | 1505.41 |
| 1 | 完成 | 失败 | 通过 | 通过 | 通过 | 通过 | 1531 | 2042 | 1618.24 | 1691.55 |
| 2 | 完成 | 失败 | 失败 | 通过 | 通过 | 通过 | 1503 | 2653 | 1472.83 | 1538.31 |

0号剩余顶部桌布IoU .868956<.88（整体.966089通过）；1号盘边缘3.978px>3.6px、两轴误差6.395%/6.799%>6%；2号盘IoU .800795<.88、中心16.798px>3.6px、边缘13.536px>3.6px，整体布IoU .959814<.964、错误背景.012370>.012，顶部.880769通过。三条苹果几何均通过。原有一次分组搜索/边界/硬约束保护下未解决的残差保留，不因预算尚有余额额外增加轮次。

初始几何分别实际198/242/338次repair渲染；0/2提案因已有plate.axes保护未被采用，1采用。之后按同一固定规则执行覆盖、受限相机/布局及八组外观拟合；无人工挑选。0/1/2分别1281/1428/1414候选记录、9/10/7次保留，真实MCP46/50/44次、工具错误0、零候选优化0。不同统计口径分别列出，不用正常返回率或保留数代表质量。

外观没有新增合格线，三条均为measured_no_hard_threshold，审阅not_run_ablation，模型HTTP总计0。以下仅比较几何结束后到最终的外观拟合，几何和掩膜保持不变，误差越低越好：

| episode | 苹果颜色目标 | 盘内综合目标 | 桌布颜色目标 |
|---|---|---|---|
| 0 | .399655 → .043062 | .091783 → .017731 | .023255 → .022314 |
| 1 | .352110 → .040881 | .085405 → .017852 | .024049 → .022503 |
| 2 | .355565 → .038171 | .078470 → .046295 | .025123 → .021694 |

初始化到最终的完整分项变化另见cold_start_metrics.csv、photometric_effects.csv；纯外观阶段见每条appearance_only_effects.csv。高光证据置信度仅.167/.211/.184，盘内边界置信度约.047/.045/.047，不能据此确认高光恢复或真实盘形。细布纹理和褶皱未恢复，作为非阻断局限记录；未豁免覆盖、颜色和支撑。

每条交付原始XML及原始相机渲染、规范化可行初态、最终重载全图/局部图、静态/动态便携场景和state、最终参数/实际尺寸/相机、独立静置视频与接触/时间序列、分项曲线、MCP及候选轨迹。统一index.html、cold_start_summary.csv、final_reacceptance.json、execution_audit.json、zero_model_usage.json和六个压缩包由package_manifest.json索引。固定初态独立隔离回归证明不需要v4/v5/旧消融目录；正式三条也启用历史读取拦截。

用户询问后，在0号冻结产物上进行一次只读历史对照，保存comparison/episode_000000/。相对旧warm-start无Terra0号，苹果颜色.047216→.043062改善8.8%；盘内综合.013114→.017731退化35.2%；布色.021630→.022314退化3.2%；顶部IoU .926456→.868956由通过变失败。高光数值变化小但证据不足；盘中央亮度误差1.924→4.872退化、内侧梯度误差6.535→4.027改善。两者初始化来源不同，不能将差异单独归因于模型开关；对照未参与本轮搜索。

96个不同重建测试用例通过。所有32273份受保护历史文件、旧账本和STOP哈希不变，402仍54完成/0运行。本轮三条已完成后停止，没有策略推理、训练或策略测试。
