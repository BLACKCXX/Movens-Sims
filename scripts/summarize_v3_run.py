#!/usr/bin/env python3
"""Add derived tables and a candid human report to an immutable rendered result."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import html
import json
from pathlib import Path

import mujoco
import numpy as np

from movens_sim.reconstruction.session import ReconstructionSession
from movens_sim.util import ROOT,read,write,sha,now


def summarize(run):
    run=Path(run).resolve();result=read(run/'result.json');config=read(run/'resolved_config.json');state=read(run/'session.json');before=result['metrics_before'];after=result['metrics_after'];initial=read(run/'compiled_asset_checks.json')['source_geometry'];final=result['geometry'];cam=read(run/'camera_actual.json')
    rows=[]
    for label in ('apple','plate'):
        for version,g in [('mount_layout_initial',initial[label]),('joint_geometry_final',final[label])]:
            rows.append(dict(version=version,object=label,source_relative_scale=g['absolute_scale'],x_m=g['xyz_m'][0],y_m=g['xyz_m'][1],z_m=g['xyz_m'][2],quaternion_wxyz=json.dumps(g['quaternion_wxyz']),dimensions_world_m=json.dumps(g['dimensions_world_m']),mass_assumption_kg=g['mass_kg']))
    with (run/'parameter_comparison.csv').open('w') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    session=ReconstructionSession(run,config)
    try:
        support=dict(camera_world_xyz_m=cam['xyz_m'],nominal_cloth_z_m=.7771160388486943,height_above_nominal_cloth_m=cam['xyz_m'][2]-.7771160388486943,height_above_saved_visible_surface_range_m=[cam['xyz_m'][2]-session.scene.support.height_range[1],cam['xyz_m'][2]-session.scene.support.height_range[0]],local_final_center_heights_m={label:float(session.scene.support(np.array([[session.scene.params[label+'.x'],session.scene.params[label+'.y']]]))[0]) for label in ('apple','plate')},note='camera_actual inherited per-object support-clearance fields describe the original Mount layout; this separate metadata recomputes final local support without modifying the frozen camera binding')
        support['height_above_final_local_support_m']={k:cam['xyz_m'][2]-v for k,v in support['local_final_center_heights_m'].items()};write(run/'camera_support_recheck.json',support)
    finally:session.close()
    statuses=[read(p) for p in (run/'terra').glob('*/attempt_*/status.json')];calls=[json.loads(x) for x in (run/'commands.jsonl').read_text().splitlines()]
    first=datetime.fromisoformat(calls[0]['utc']);end=datetime.fromisoformat(result['completed_utc']);elapsed=(end-first).total_seconds();api=sum(s.get('wall_seconds',0.) for s in statuses)
    usage=dict(input_tokens=sum(s.get('usage',{}).get('input_tokens',0) for s in statuses),output_tokens=sum(s.get('usage',{}).get('output_tokens',0) for s in statuses),unknown_attempts_without_usage=sum(s.get('status')=='interrupted_unknown' for s in statuses))
    diagnostics=read(run/'physical_diagnostic.json');dynamic_time=(run/'physical_diagnostic.json').stat().st_mtime-(run/'physics/authored/scene.xml').stat().st_mtime
    notes=state.get('maintenance_notes',[])
    result.update(maintenance_notes=notes,timing_summary=dict(first_launch_to_completed_seconds=elapsed,includes_code_maintenance_and_pause=True,optimization_active_seconds=result['optimization_wall_seconds'],api_attempt_wait_seconds=api,diagnostic_wall_seconds_from_artifact_timestamps=dynamic_time,diagnostic_wall_basis='File timestamps; approximation, not a separately instrumented timer'),recorded_usage=usage,numerical_round_detail=state['rounds'],batch_readiness='Formal versioned single-episode entry and shared backend are implemented; not recommended for batch reconstruction quality, no batch/concurrency validation was run',implementation_validation=read(run/'automated_validation.json') if (run/'automated_validation.json').exists() else None)
    write(run/'result.json',result);write(run/'timing_summary.json',result['timing_summary'])
    lines=['# episode_000000 · Mount 相机＋位置／大小联合拟合','', '**代码已实现并完成单条执行；本条质量未通过，保持冻结分析候选。**','', '| 指标 | Mount 固定尺寸布局基线 | 新联合拟合 |','|---|---:|---:|']
    for label,cn in [('apple','苹果'),('plate','盘子')]:
        a,b=before['objects'][label],after['objects'][label]
        for key,title,unit in [('iou','轮廓 IoU',''),('centroid_error_px','中心偏差',' px'),('symmetric_edge_distance_px','边缘距离',' px'),('area_relative_error','面积相对误差','')]:lines.append(f'| {cn} {title} | {a[key]:.4f}{unit} | {b[key]:.4f}{unit} |')
        lines.append(f'| {cn} 原始资产绝对尺度 | {initial[label]["absolute_scale"]:.5f} | {final[label]["absolute_scale"]:.5f} |')
    lines+=['',f'盘子可视尺寸从约 {initial["plate"]["dimensions_world_m"][0]*1000:.1f} mm 增至 {final["plate"]["dimensions_world_m"][0]*1000:.1f} mm；仅改变物体位置、整体尺寸和苹果必要朝向，相机/FOV/桌高/桌面外延固定。尺寸为名义源资产推断，未经实测。','',f'相机世界位置 `{cam["xyz_m"]}` m，世界俯视角 {cam["down_angle_deg"]:.4f}°，VFOV {cam["fovy_deg"]:.1f}°；名义布面净高 {support["height_above_nominal_cloth_m"]:.6f} m。不是实测相机标定。','', '## 状态与限制','',f'- 数值几何：{result["geometry_numeric_passed"]}。盘子边缘 {after["objects"]["plate"]["symmetric_edge_distance_px"]:.2f}px，长短轴误差 {after["objects"]["plate"]["axis_relative_errors"]}，超出冻结阈值。',f'- 初始支撑：{final["physical_valid"]}。对象 Z 从实际保存的变形三角面求得；不把延展桌布作为物理支撑。',f'- Terra 几何复核：{result["terra_geometry_passed"]}；最终只读结论 `{result["final_terra_verdict"]}`。',f'- 外观：`{result["appearance_status"]}`；未执行，红黄分布、布纹、盘内圈及光照仍有差异。',f'- 静置：盘子通过；苹果位移 {diagnostics["objects"]["apple"]["translation_max_m"]*1000:.2f}mm、转角 {diagnostics["objects"]["apple"]["rotation_max_deg"]:.2f}°、最大穿透 {diagnostics["objects"]["apple"]["penetration_max_m"]*1000:.2f}mm，未通过。没有采用静置后位姿替换视觉拟合。', '- 顶部浅色带及桌布/背景仍不一致。固定桌面来自旧 batch 的近边界扩展，桌沿真值与布面掩码仍有不确定性。盘子内圈缺少可靠标注，不能仅凭明暗判定几何正反或继续放大。','', '## 实际执行与预算','',f'- 几何轮数 {int(result["geometry_fit_rounds"])}：首轮有效联合搜索；第二轮因执行版本缺少朝向权限路由而产生 **0 次新渲染**，不能称为完成了倾斜验证。',f'- 优化渲染 {int(result["render_optimization_evaluations"])} / 1200，纯投影残差 {int(result["projection_residual_evaluations"])} / 5000；优化活动时间 {result["optimization_wall_seconds"]:.2f} / 1800 s。',f'- Terra 实际/保守计数 {int(result["terra_actual_http_requests"])} / 5：几何有效响应 1 次，最终请求中断、结果未知 1 次，同一冻结候选最终只读重试有效响应 1 次。模型精确为 `{result["model"]}`；未静默换模型，未发第六次。',f'- 已知 API 使用量：输入 {usage["input_tokens"]}、输出 {usage["output_tokens"]} tokens；未知请求的用量不可得。',f'- 首次正式入口到完成 {elapsed:.1f}s（含维护暂停）；API 尝试等待 {api:.1f}s，静置/导出从文件时间推算约 {dynamic_time:.1f}s。这些时间与 58 秒量级的优化活动时间分开。','', '## 执行版本修复记录','']
    lines+=['- '+n for n in notes]
    if result.get('time_accounting_audit'):
        audit=result['time_accounting_audit'];lines.append(f'- 初始执行版仅对内核分段计时（{audit["original_kernel_seconds"]:.2f}s）。根据持久化轮次起止时间保守补记求解器/检查点/缓存开销 {audit["added_overhead_seconds"]:.2f}s，现账本合计 {audit["accounted_full_optimization_seconds"]:.2f}s；未退还额度、未改上限。当前代码已对完整搜索计时，含嵌套防重复计数及超时中断测试。')
    lines+=['- 当前路由已修复，离线回归明确证明：Terra 的朝向证据能启用真实后端的盘子倾斜，并与 XY/尺寸联合拟合。冻结后未重新拟合本条，原失败历史和全部预算保留。','- `code_snapshot/` 是实际初次执行版本；`code_compatibility/` 保存只读维护的前后身份、当前源快照和独立像素零差校验。没有直接混用不相容完成状态。','', '## 复现与检查','', '```bash','scripts/movens-sim run --config configs/mount_joint_v3.json --episode 0 --terra --resume',f'PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python -B scripts/check_v3_delivery.py --run {run.relative_to(ROOT)}','```','', '新入口已接入正式流水线；当前配置只允许 episode 0，不自动跑其余样本。由于本条几何/静置及全图结构未通过，尚不推荐扩大批量，也未验证批量并发。','', '## 交付文件','', '- `comparison.png`：同尺寸五列；第五列明确外观未执行。','- `ego_static.mp4`：640×480、30FPS、5秒，固定相机固定布局的三维仿真。','- `settle_diagnostic.mp4`：两物体自由的独立短暂重力检查。','- `delivery/static/portable/scene.xml` 及同目录 `.state.npz`、`assets/`：便携静态场景。','- `physics/portable/scene.xml` 及状态/资产：动态诊断初值。','- `parameter_comparison.csv`、`metrics_*.json`、`camera*.json`、`asset_audit.json`。','- `terra/`、`mcp_tool_calls.jsonl`、`candidates.jsonl`、`budget_ledger.json`：全部请求、修复、接受/拒绝及开销。','- `delivery_checks.json`：独立目录加载、像素、视频、原始输入与旧产物保护检查。','', '本结果不是真实采集视频、交互回放、实测相机标定或已验证真实动力学。','']
    (run/'report.md').write_text('\n'.join(lines))
    notice='<aside style="border:1px solid #d99;padding:16px"><strong>未通过；外观未执行。</strong><p>第二轮朝向返修在执行版本中被权限路由挡住，零新渲染。当前代码已修复并离线验证；本冻结候选未再改动。3 次 Terra 计数包括 1 次中断未知请求。</p><p><a href="parameter_comparison.csv">参数对照</a> · <a href="delivery_checks.json">交付检查</a> · <a href="timing_summary.json">耗时</a> · <a href="report.md">完整报告</a></p></aside>'
    doc=(run/'index.html').read_text();marker='<aside style="border:1px solid #d99'
    if marker not in doc:(run/'index.html').write_text(doc.replace('<h1>',notice+'<h1>',1))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();r=summarize(args.run);print('Metadata-only report updated; frozen scene and image parameters unchanged:',r['delivery_binding']['scene_parameter_asset_hash'])
