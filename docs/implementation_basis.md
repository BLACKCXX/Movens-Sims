# 实施依据与版本

本轮只重建初始静态桌面，不运行机器人策略、抓取轨迹或训练。

实际源目录 `/root/autodl-tmp/Movens-Interact`，原 Git HEAD 为 `633c731d3049aecc69f638ae8dce6250bbead65f`；该目录有本地修改，Git HEAD 不能代替实际文件哈希。每条导入的 evidence 和 source_manifest 记录实际代码、XML、资产和输入哈希。原项目保持只读，GPU 1 上的独立任务未干预。

最新登记入口是 `outputs/appletoplate_tableup005mm_active.json`，指向 `appletoplate_tableup005mm_20261105T020624Z`。训练索引也登记同一批次与 revision。已枚举历史场景家族；旧 softpad_final/repose/xycomp 是溯源资料。按完整输入的不同原 episode ID 升序冻结 0、1、2、3、4；episode 0 的旧抓放失败不影响本次输入资格。缺失输入和全部场景枚举见 `configs/discovery.json`。没有换用其他 episode。

真实目标是本地原始 `nvidia/GR00T-N1.7-AppleToPlate` ego 视频，revision 固定为 `d89c126a713c6632432a607c12661546ff4d6ea9`；导入时核对源索引 LFS SHA256，无需重新下载。旧仿真视频仅用于基线对应检查，不充当真实观测。FFmpeg/libdav1d 解码原始 AV1；OpenCV 随附解码器不能解码该源，未将它误报为真实视频缺失。

实际参考的外部文件：

| 工作 | 固定版本、已阅读文件 | 本项目采用的内容 |
| --- | --- | --- |
| [Agentic Real2Sim](https://github.com/agentic-real2sim/agentic_real2sim/tree/880389bda289d2e78154c73ae1b6d23b7a650557) | commit `880389bda289d2e78154c73ae1b6d23b7a650557`；README.md、ar2s/pipeline_artifacts.py 前110行、ar2s/agents_visual/subagents/scene_mismatch_prompt.md | 分阶段文件契约、每 episode 独立产物、真实/仿真图身份及可见不一致诊断；未部署其机器人优化 |
| [SceneSmith](https://github.com/nepfaff/scenesmith/tree/67cc408fd38334b4a926efef45e284302ed5055b) | commit `67cc408fd38334b4a926efef45e284302ed5055b`；README.md、scripts/export_scene_to_mujoco.py 前110行、scenesmith/prompts/data/manipuland_agent/critic_agent.yaml 前110行 | 建议、视觉评论、本地物理检查的职责区分；独立资产导出。其原生 Drake 与实验性 MuJoCo 导出不直接移植 |
| [MuJoCo Python / MjSpec 3.7](https://mujoco.readthedocs.io/en/3.7.0/python.html) 与 [建模](https://mujoco.readthedocs.io/en/3.7.0/modeling.html) | 实装 `mujoco==3.7.0` | MjSpec 解析与静态展示模型编译；保留碰撞分解与自由关节 |
| [OpenAI 视觉输入](https://developers.openai.com/api/docs/guides/images-vision) 与 [结构化输出](https://developers.openai.com/api/docs/guides/structured-outputs) | 实装 `openai==2.54.0`，实际模型标识见响应日志 | 官方 Responses API、标注图像身份、strict JSON schema、独立本地数值/单位/阶段验证 |

MuJoCo 3.7 对本来源的 MjSpec flex 编译出现 `pad_other ... no equality constraints or passive forces`，直接 `MjModel.from_xml_path` 可编译。因此物理副本保留原始 flex XML，直接编译；静态显示副本用 MjSpec 编译。该差异逐条记录，不把“能编译”当成物理正确。

基线使用原 loader 的完整 reset 核查，并明确选取旧 frame_map 登记的真实/source frame 0 对应仿真保存状态（post-settle）。XML 默认、reset 前后和所选状态分别记录；相机先从挂载 body 求世界变换，再建立世界固定相机。

真实首帧苹果与盘子清晰、无手遮挡，采用 HSV 连通域像素掩码并输出人工可检查的轮廓图。只拟合 frame 0。相邻初始帧仅检查图像变化，不宣称多视角重建或真实深度；未安装 DA3 或任何训练栈。

名义尺度由原盘子约 0.182m 直径固定；盘子世界位姿也作为坐标基准。相机用盘子投影矩初始化，随后有界 Powell 搜索。苹果的额外 scale 相对已烘焙资产，未重复叠加历史 scale110/XY105/Z110 或桌面高度补偿。桌面外延只受可见布面覆盖约束，隐藏尺寸仍存在很大歧义；不能解释为测量所得尺寸。

原始局部软层保留完整动态状态；静态显示使用对应三角面冻结网格。观察到的更大布面以独立网格继续到有界桌面外缘，未使用真实图像覆盖画面的平面。延展区域没有独立软材料验证。静态副本不能用来证明软层稳定。

原盘子是固定体，本批继承并显式标注，故不能给出盘子自由静置物理合格结论。苹果不锁定，诊断原生积分 1秒、4000步，继承初始速度，不逐步重置。接触、质量、摩擦与软层本构均未由 RGB 拟合。
