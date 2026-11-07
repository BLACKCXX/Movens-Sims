# 重建场景的单条策略测试

这是用户明确授权的独立推理测试，不属于静态重建优化。Movens-Interact 和历史 runs 都是只读输入。不训练、不重新估计场景、不调用 Terra、不扩展批量。

配置为 `configs/policy_test_wrist48500_v1.json`，入口为 `scripts/test_reconstructed_policy.py`。使用原项目已有的 Python 3.11 训练环境（PyTorch 2.7.1+cu128、MuJoCo 3.7.0），不向 Movens-Sim 的独立环境安装训练栈。

```bash
cd /root/autodl-tmp/Movens-Sim
/root/autodl-tmp/Movens-Interact/.local/train-env/bin/python -B \
  scripts/test_reconstructed_policy.py \
  --config configs/policy_test_wrist48500_v1.json
```

同一试跑中断后加 `--resume`。已经完成时，`--resume` 校验输入和产物哈希并读取结果，不重新执行。`--phase prepare|run|render|evaluate` 用于分阶段处理；渲染或评价重试使用已经生成的状态，不重新推理。

本次输入是 mount_joint_v3 episode 0 的动态便携 XML/状态，使用真实 flex 而非展示用冻结网格。苹果和盘子的实际终值已写入 XML，不再次套用参数 JSON。补回同源 G1，只初始化与真实 frame 0 对齐的 43 维观测关节及 Mount 的固定 root 假设。缺少速度测量，机器人速度为零。保留新场景固定世界相机（VFOV 42°），删除机器人旧相机，不把它称为真实标定。

现有测试实现来自已执行的 color_generalization 冻结代码；只调用单 case 方法，不运行原 A/B、换色或批量入口。模型与手部分支代码取所选腕部微调训练的冻结版，使用 checkpoint 内嵌配置及两个归一化器。checkpoint 是本轮新增 48500 步的 best_val，不是最后的 50000 步，也不把初始化来源的步数相加。基础权重不复制大文件，保持只读引用并校验身份。

局部冻结代码的兼容修改记录在 `runtime_adaptations.json`：核对配置中的 checkpoint 步数；迁移状态后重新 forward；记录 t=0 初始观测；更新图像身份和视频标签；使用动态盘子逐步位姿评价承托。额外进程内适配读取网格真实 AABB、记录盘子位姿。所有数值阈值和 50 Hz / 30-20-10 推理协议保持原样。没有旧流程的额外 0.2 秒 settle。19.64 秒操作和 1 秒尾段均由原记录确定。

保存的场景可独立编译；`test/scene/scene.state.npz` 包含实际 qpos/qvel/ctrl/time，不能只加载 XML 默认值来代替 flex 的保存状态。`initialization/initial_state.npy` 是适配后模型的完整 integration state。`scene_adaptation_check.json` 验证原环境世界位姿、边界、相机、flex 和重载状态，机器人隐藏时检查 RGB 完全一致。

结果位于 `runs/policy_test_wrist48500_v1/episode_000000/`：`index.html`、`report.md`、`result.json`，实际测试场景 `test/scene/scene.xml`，视频 `test/videos/ego.mp4`、`side.mp4`，原测试指标 `test/evaluation.json`、`test/run/report.json`，以及预测计划、实际执行、全接触记录和断点。

工程跑通、抓取放置成功、物理质量和原重建质量分别报告。原重建静置未通过；episode 0 不在保存的 250 条训练/验证/测试划分中。本试跑不自动修正已知物理问题。发生严重数值或场景失败时保留中止状态和视频，不通过换场景、种子或放宽阈值重试。

完成后的只读审计入口（不会调用模型或执行物理步）：

```bash
/root/autodl-tmp/Movens-Interact/.local/train-env/bin/python -B \
  scripts/audit_reconstructed_policy_test.py \
  --config configs/policy_test_wrist48500_v1.json
```

它校验第一张模型输入的像素哈希、初始化、实际控制与对应预测限幅后的一致性，报告实际执行跟踪误差、指令限幅、完整进程耗时和视频元数据。结果为 `test_audit.json`、`audit_report.md`。本次为消除新记录器重复搜索关节的开销，在 1.6 秒控制边界保存状态后恢复；兼容记录保存在 `resume_maintenance/`，没有重新开始物理试跑。

原适配器状态数组中的 `source_frame_index` 使用 50Hz 控制参考索引，审计生成 `reference_frame_mapping.json` 明确此语义并给出 30FPS 时间对应帧号，原始状态与指标不被修改。模型实际输入始终是当前仿真 RGB；执行时间和视频内容不依赖这个参考索引。
