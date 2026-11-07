# 全402条初始场景重建

配置 `configs/full402_h0_v1.json`，正式后端仍为 mount_joint_v3；新增 `initialization.mode=fk_h0` 和持久化批次管理。只构建初始静态场景，不训练、不切片、不测试 checkpoint。Movens-Interact 只读，历史 runs 保持不变。

## 已核实来源与尺寸

原始 revision 固定为 d89c126a713c6632432a607c12661546ff4d6ea9，完整 ID 0–401。402 份原始状态、真实视频、已有 XML 均存在，XML 的63个独立资源均可读取。manifest 从原始元数据枚举，包含旧训练入选250条及排除152条，抓取成败不作为静态输入资格。

最新 wrist 30000+10000+10000 训练的 formal/config.json 指向 appletoplate_tableup005mm_training_v1/index.json，哈希一致。苹果源于 episode_000018_scale110_xy105_z110_roundbottom_tableup15mm_contact6/scene 的网格；整体1.10、XY1.05、Z1.10已烘焙，等效未缩放轴倍率1.155/1.155/1.21，另保留已存在的平顶/圆底处理。当前视觉包围尺寸86.625×86.212×90.593mm。导入和首轮优化额外尺度为1.0；范围0.5–2.0相对这份不可变训练资产定义，不再乘一次历史缩放链。

本轮经用户确认：soft_top=h0，rigid_top=h0−0.006m。旧来源是soft_top=h0+0.005m，导入时真实平移完整硬桌和软层节点−5mm，并重新计算苹果、盘子支撑Z。使用未压缩reset，节点qpos/速度为零，不使用旧仿真settle状态。苹果朝手掌补偿0mm。原始桌面/布面尺寸固定，不继承旧拟合扩大的桌面或增加布面外延。

逐条复用已有FK事件、窗口、手腕到苹果偏移、h0与XY，记录来源和置信度；释放偏移仍是未经真实标定验证的初值。episode179有抓取起点截断记录，继续处理但不将缺失证据当作质量通过。相机逐条用真实frame0观测43关节和已固定root假设计算，复用官方固定版本的d435安装参考及OAK近似，VFOV42°；并非真实相机标定。本批不开放相机、FOV和桌面几何修改。

参考始终为完整640×480 frame0，无裁图；1/3/6仅检查运动。像素HSV连通域以FK投影辅助选择，不把投影当真实物体掩码。另生成观测机器人模型的像素投影，膨胀为保守的可能遮挡区域并明确标为不确定估计；不宣称它是已验证的真实分割。顶部像素保持有效。低可信掩码阻止几何通过，分割失败保留初值并标记观测问题。

## 启动、查看、停止、恢复

```bash
cd /root/autodl-tmp/Movens-Sim
.venv/bin/python -B scripts/run_full_reconstruction.py start --config configs/full402_h0_v1.json
.venv/bin/python -B scripts/run_full_reconstruction.py status --config configs/full402_h0_v1.json
.venv/bin/python -B scripts/run_full_reconstruction.py stop --config configs/full402_h0_v1.json
.venv/bin/python -B scripts/run_full_reconstruction.py resume --config configs/full402_h0_v1.json
```

start使用脱离终端的POSIX会话和DEVNULL标准输入。协调器及worker执行批次冻结代码，不依赖后续对工作树代码的修改。正式构建由`pipeline.run_pipeline`调用统一后端，初始化为`full_prepare.prepare_full`；批次统一通过上述管理入口恢复。

stop只设置本批STOP请求，worker在评价/API安全边界保存；不会停止其他任务。resume校验冻结代码及任务配置，保留原SQLite预算和候选缓存，检查已完成产物是否仍完整。历史完成场景保持原代码绑定，不能用修改后的工作树静默改写其身份。API认证/模型错误暂停API阶段；修复外部配置后显式resume可解除暂停，不退还请求额度。

每条5次实际Terra请求、4轮几何、1200次优化渲染（600+400+200）、5000次投影、1800秒优化计算。渲染和请求执行前持久化占用，失败及中断不退款；一份全批SQLite中使用逐episode任务ID，旧episode0试跑预算另行保留。预处理机器人遮挡渲染也计入本条initial预算。模型精确为gpt-6.1-terra，SDK重试0，API并发1，最后一格只读。API等待、导出、静置耗时分别记录。

起始2个worker，每卡1个；线程固定1。依据cgroup实际32核/180GiB而非宿主128核/754GiB调度。首4条工程通过且资源满足配置才升至最多4个；CPU/内存/显存不足保持或降为2个。失败不换样本、不修改验收阈值；基础设施最多额外重试2次，观测分割错误直接记待复核。相同系统性错误连续3条时暂停并记录。磁盘保留100GB。

## 输出与接续

批次目录 `runs/full402_h0_v1/`：

- `job.json`、`heartbeat.json`、`handoff.json`：真实PID、心跳、命令；`tasks.sqlite`：任务领取与状态。
- `logs/coordinator.log`、`logs/episode_XXXXXX_attempt_NNN.log`：协调和逐条日志。
- `results.csv`从启动即覆盖402行，`summary.json`分别统计构建、视觉、物理和待复核，`index.html`持续更新视频/图像链接。
- 每条目录保存初始化/相机/来源、参数、候选历史、分项指标、Terra记录、独立XML和资产、重载证据、640×480/30FPS/5秒静态视频，以及单独1秒静置诊断。
- `runs/_task_budgets/full402_h0_v1.sqlite`：不可逆逐条预算；不要删除或用新run_id绕过本批限额。

几何数值与有效Terra审阅共同通过后才做外观。当前布面支持灰度明暗，没有测得布纹资产；不会声称恢复真实褶皱。最终交付由哈希绑定，终审后不改画面参数。构建完成不等于视觉/动力学合格；未通过也保留最佳候选。

## 首批运行与兼容修正

初始后台于2026-11-07T10:57:45Z启动，首两条分别用了2轮几何、316/320次渲染和3次实际Terra请求。实际模型标识gpt-6.1-terra，独立重载/视频规格检查通过，但视觉与物理质量未通过。episode0完成后恢复验证消耗0次新增模型/渲染。

episode3暴露旧FK盘子初值越过固定支撑边界9.15mm的问题。保留raw FK输入，构造带0.1mm边缘余量的最近可支撑数值搜索初值（该条移动9.247mm），记录`support_search_seed.json`。这不是朝手掌方向补偿，桌面和尺度不变。对相交初值再测试至多4个基于包围盒的分离位置，选最近的受支持位置作为搜索初值；图像优化仍决定最终XY/大小。真实来源33/44/54/103/260的编译碰撞检查已确认旧初值存在5.2–11.1mm相交，不能将这类约束问题误报为基础设施故障。

旧`frozen_code`、`code_versions/support_seed_v2`和未推广的v3验证证据均保留；当前新worker采用经过45项测试的`code_versions/feasible_initial_v4`。`active_code.json`及`rolling_code_transition.json`记录选择和兼容条件，仅允许输入/观测/配置不变的未完成任务接续，不重置预算。已完成场景保持原版本。

已实测安全STOP/恢复：episode4在MCP数值返修中停止，保留原拟合轮次预约和缓存；中断导出归档在该条code_compatibility中，恢复同一轮。版本接续前后的预算完全相等，证据`code_versions/support_seed_v2_adoption.json`。协调器升级时4个独立worker继续运行，接续记录在`verification/coordinator_rollover.json`；最新PID及进度始终以job/heartbeat/summary为准。首4条工程完成后已自动从2升至4个worker。
