# Movens-Sim

首批 episode 000000–000004 已完成初始静态重建候选、两类视频、可移植 XML、逐对象指标和 GPT API 实测。

- [五条视频与对照报告](runs/batch_v1/index.html)
- [真实 / 优化前 / 优化后总览](runs/batch_v1/overview.jpg)
- [结果 CSV](runs/batch_v1/results.csv) 与 [首批报告](docs/first_batch_report.md)

统一图像门槛通过 4/5；完整场景视觉合格 0/5、物理合格 0/5、待复核 5/5。这是用于分析的三维重建候选，不是真实采集或已验证真实动力学。

## 运行

项目使用独立 Python 3.12 环境，MuJoCo 固定 3.7.0。未安装 Movens 训练栈。

当前主脚本默认使用 `configs/batch_v2.json`，模型为 `gpt-6.1-terra`，新结果目录为 `runs/batch_v2`。旧批次的配置和模型记录保留；恢复或查看旧批次时显式指定 `--config configs/batch_v1.json`。

```bash
cd /root/autodl-tmp/Movens-Sim

# 检查已有五条结果并恢复缺失阶段；已完整阶段不重新调用 API。
./scripts/movens-sim run --config configs/batch_v1.json --resume --gpt

# 显示实际执行范围，不运行拟合/诊断或调用 API。
./scripts/movens-sim run --config configs/batch_v1.json --resume --gpt --dry-run

# 只重建报告。
./scripts/movens-sim report --config configs/batch_v1.json

# 使用当前代码与 v2 prompt，在新目录独立重跑第一条。
./scripts/movens-sim run --config configs/batch_v2.json --episode 0 --gpt

# 不调用 API，独立运行本地拟合与离线契约验证。
./scripts/movens-sim run --config configs/batch_v2.json --run-root runs/local_v2 --local-only

# 独立阶段（须先完成同一 identity 的依赖阶段）。
./scripts/movens-sim prepare --config configs/batch_v2.json --episode 0 --gpt
./scripts/movens-sim fit --config configs/batch_v2.json --episode 0 --gpt --fit-stages camera layout size appearance
./scripts/movens-sim validate --config configs/batch_v2.json --episode 0 --gpt
./scripts/movens-sim render --config configs/batch_v2.json --episode 0 --gpt

# 必要验证。
MUJOCO_GL=egl .venv/bin/python -B -m pytest -q
MUJOCO_GL=egl .venv/bin/python -B scripts/check_delivery.py
```

`discover` 从实际原项目枚举并冻结清单，`import` 验证并导入；本批数据已导入，不必重新下载。不可变数据在 `data/episode_XXXXXX/`，结果在 `runs/batch_v1/episode_XXXXXX/`。现有 `Movens-Interact` 始终作为只读源。

批量模式不指定 `--episode`；`--concurrency 1..4` 提供有界进程并发，正式首批按 1 路顺序运行。每条先取得文件锁。输入、冻结 manifest、代码、参数配置、prompt 与 MuJoCo 版本均核对。变化后使用新结果目录；缺失或哈希不符的产物不会被当作完成。

原数值拟合与重力诊断代码保存在 `runs/batch_v1/code_executed_v0.1.0/`。后续 0.1.1 修复恢复计数、自动惯量模型的缓存变换、GPT 图像归档与调用预算。该维护版本对本批已做逐条确定性渲染兼容检查，原 identity/receipt 保存在 `compatibility/`，不是静默沿用改变过的代码状态。

## API

流水线优先读取 `OPENAI_API_KEY`，否则读取配置的外部 `api.env`。服务器现有文件的 `api` 名称仅在进程内兼容为密钥；不会执行 dotenv 中的 shell，也不会把密钥写入仓库、prompt、报告或日志。压缩包不包含密钥文件。

首批历史调用使用 `gpt-4.1-2025-04-14`；当前新工作默认改为 `gpt-6.1-terra`，继续使用官方 Responses API，模型来自配置。迁移接口验证记录在 `docs/model_switch_terra_check.json`。每轮发送三张有明确身份的 PNG、参数/单位/阶段与数值指标，不发送完整视频。strict JSON Schema 之后仍验证未知 ID、非有限数、单位、边界、阶段和 decision/update 语义。HTTP 重试、超时与总次数受限；无数值试跑预算时只记录建议。

`diagnostic_v1` 的真实语义错误（needs_review 同时给 updates）已保留并拒绝；v2 明确一致性规则后，对前两条复核并冻结，再评价后三条。五条 v2 均通过 API 与本地契约验证。原 v1 和数值候选记录完整保留。详情见 `gpt/`、`gpt_summary.json`、`prompt_freeze_diagnostic_v2.json`。

## 移植

在 Python 3.12 / 可用 EGL 的主机上解包项目后：

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
.venv/bin/pip install --no-deps -e .
./scripts/movens-sim run --config configs/batch_v1.json --resume --gpt
```

结果重载及已导入数据的独立运行不需要原 Movens-Interact 目录。manifest 内的旧绝对路径是来源记录；XML 的资产引用均为包内相对路径。新 API 调用需要当前主机的凭据。原始 AV1 参考帧由系统 FFmpeg 解码，视频也由 FFmpeg 编码。

实现依据与实际参考版本见 [implementation_basis.md](docs/implementation_basis.md)。
