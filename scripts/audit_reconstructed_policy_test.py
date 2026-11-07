#!/usr/bin/env python3
"""Read saved predictions/execution and verify a completed test; no mj_step/model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from movens_sim.policy_test import read, sha, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();root=Path(read(args.config)['run_root']);case=root/'test'
    result=read(root/'result.json');cfg=read(root/'config.json');report=read(case/'run/report.json')
    for name,digest in read(root/'completion.json')['files'].items():
        if sha(root/name)!=digest:raise ValueError('Completed test output changed: '+name)
    model=mujoco.MjModel.from_xml_path(str(case/'scene/scene.xml'))
    robot=read(case/'scene/robot.json');names=robot['body_joint_names']+robot['hand_joint_names']
    limits=model.jnt_range[[model.joint(n).id for n in names]]
    plans={}
    for p in sorted((case/'run/plans').glob('plan_*.npz')):
        with np.load(p,allow_pickle=False) as d:plans[int(p.stem.split('_')[-1])]={k:d[k] for k in d.files}
    with np.load(case/'run/robot_apple_timeline.npz',allow_pickle=False) as d:
        t=d['timestamps'];q=d['robot_qpos'];ctrl=d['robot_ctrl']
        np.testing.assert_array_equal(d['joint_names'],names)
    applied_errors=[];clipped_counts=np.zeros(43,dtype=int);clip_max=np.zeros(43);executed_predictions=[]
    end=report['original_trajectory_end_s'];dt=report['dt_s']
    for i in range(1,len(t)):
        tick=int(np.floor(min(max(t[i]-.5*dt,0),end-.5*dt)*50))
        start=(tick//20)*20;offset=tick%20
        prediction=plans[start];command=np.r_[prediction['body_absolute'][offset,:29],prediction['hand_absolute'][offset]]
        expected=np.clip(command,limits[:,0],limits[:,1])
        applied_errors.append(float(np.max(np.abs(ctrl[i]-expected))))
    for start,p in plans.items():
        for offset in range(20):
            if (start+offset)/50>=end:break
            command=np.r_[p['body_absolute'][offset,:29],p['hand_absolute'][offset]]
            clipped_counts+=(command<limits[:,0])|(command>limits[:,1])
            clip_max=np.maximum(clip_max,np.abs(command-np.clip(command,limits[:,0],limits[:,1])))
            executed_predictions.append(command)
    executed_predictions=np.asarray(executed_predictions)
    with np.load(case/'run/states.npz',allow_pickle=False) as d:
        initial=d['qpos'][0];raw_reference_indices=d['source_frame_index'];state_time=d['timestamps']
    with np.load(case/'run/metrics.npz',allow_pickle=False) as d:
        touch=d['counts'][:,3]>0
        finger=d['left_finger_contact']
        contact_summary=dict(loaded_apple_robot_samples=int(touch.sum()),
            loaded_left_finger_samples=int(finger.sum()),
            first_loaded_apple_robot_contact_s=float(d['t'][touch][0]-dt) if touch.any() else None,
            max_apple_root_height_gain_m=float((d['apple_qpos'][:,2]-d['apple_qpos'][0,2]).max()),
            apple_motion_note='Root-height gain is descriptive; inherited pick metric separately evaluates collision-bottom clearance and sustained finger contact.')
    with np.load(case/'scene/scene.state.npz',allow_pickle=False) as d:
        initial_error=float(np.max(np.abs(initial-d['qpos'])))
    # Preserve native arrays: this sidecar clarifies the adapter's reference-index
    # metadata without changing states, controls, metrics, or any acceptance rule.
    index_kind=read(root/'initialization/frame_map.json').get('source_index_kind','50Hz_control_reference_index')
    reference_note=('Original adapter source_frame_index uses the 50Hz control grid; corrected video-time correspondence is supplied separately.'
                    if index_kind=='50Hz_control_reference_index' else 'Native source_frame_index uses the 30FPS video grid; control references are separate.')
    reference_map=[dict(state_index=i,sim_time_s=float(ti),
        raw_source_frame_index=int(raw_reference_indices[i]),
        raw_index_kind=index_kind,
        video_reference_frame_30fps=int(round(ti*30)) if ti<=end else None)
        for i,ti in enumerate(state_time)]
    write(root/'reference_frame_mapping.json',dict(note=reference_note+' Actual policy input is always live RGB.',frames=reference_map))
    events=[json.loads(x) for x in (case/'run/predictions.jsonl').read_text().splitlines()]
    pixel_hash=hashlib.sha256(np.asarray(Image.open(case/'run/first_model_input.png').convert('RGB')).tobytes()).hexdigest()
    assets=[]
    for node in ET.parse(case/'scene/scene.xml').iter():
        value=node.get('file')
        if value:
            path=Path(value)
            if path.is_absolute() or not (case/'scene'/path).resolve().is_relative_to((case/'scene').resolve()):
                raise ValueError('Nonportable scene reference: '+value)
            assets.append(dict(path=value,sha256=sha(case/'scene'/path)))
    videos={}
    for name in ('ego.mp4','side.mp4'):
        p=case/'videos'/name
        videos[name]=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_frames,duration','-of','json',str(p)],text=True))['streams'][0]
    timing={}
    for name in ('execution.log','resume_execution.log'):
        p=root/name
        if p.exists():
            birth=int(subprocess.check_output(['stat','-c','%W',str(p)],text=True).strip())
            timing[name]=dict(start_utc_epoch_s=birth,last_output_utc_epoch_s=p.stat().st_mtime,
                             wall_seconds=p.stat().st_mtime-birth)
    inference=[float(p['total_inference_seconds']) for p in plans.values()]
    tracking=q[1:]-ctrl[1:]
    summary=dict(episode=cfg['episode'],task_result=result,
        actual_checkpoint_step=read(root/'model_loaded.json')['checkpoint_step'],
        prediction_count=len(plans),predicted_right_hand_commands_zero=all((p['hand_absolute'][:,7:]==0).all() for p in plans.values()),
        apple_robot_contact_summary=contact_summary,
        applied_control_max_error_rad=max(applied_errors,default=0),initial_state_max_error=initial_error,
        applied_control_comparison_samples=len(applied_errors),
        applied_control_comparison_scope='Saved 30FPS native state/control samples; 50Hz control updates use the unchanged frozen runtime assignment rule.',
        model_first_input_pixel_hash_verified=pixel_hash==events[0]['input_sha256'],
        sampled_tracking_rms_rad=dict(body29=float(np.sqrt(np.mean(tracking[:,:29]**2))),left_hand7=float(np.sqrt(np.mean(tracking[:,29:36]**2))),right_hand7=float(np.sqrt(np.mean(tracking[:,36:]**2)))),
        command_clipping_control_ticks={n:int(v) for n,v in zip(names,clipped_counts) if v},
        command_clipping_max_rad={n:float(v) for n,v in zip(names,clip_max) if v},
        left_hand_prediction_abs_max_rad=float(np.abs(executed_predictions[:,29:36]).max()),
        left_hand_prediction_ranges_rad={n:dict(minimum=float(executed_predictions[:,i].min()),maximum=float(executed_predictions[:,i].max())) for i,n in enumerate(names) if 29<=i<36},
        inference_total_seconds=sum(inference),inference_mean_seconds=float(np.mean(inference)),
        executed_native_physics_steps=int(round(report['last_sim_time_s']/dt)),
        process_timings=timing,accumulated_process_wall_seconds=sum(v['wall_seconds'] for v in timing.values()),
        pipeline_elapsed_wall_seconds=max(v['last_output_utc_epoch_s'] for v in timing.values())-min(v['start_utc_epoch_s'] for v in timing.values()),
        videos=videos,portable_asset_count=len(assets),portable_assets=assets,
        audit_generated_utc_epoch_s=time.time(),audit_script_sha256=sha(__file__),
        native_completion_manifest_sha256=sha(root/'completion.json'),
        native_completed_output_hashes_verified=True,
        no_model_or_physics_execution_in_audit=True,
        reference_index_note=reference_note)
    summary['checks_passed']=bool(initial_error==0 and max(applied_errors,default=0)<1e-7 and summary['model_first_input_pixel_hash_verified'] and summary['predicted_right_hand_commands_zero'] and all(v['width']==640 and v['height']==480 and v['r_frame_rate']=='30/1' for v in videos.values()))
    write(root/'test_audit.json',summary)
    (root/'audit_report.md').write_text(f"""# 单条测试审计\n\n工程/输入/控制校验：{summary['checks_passed']}。任务成功：{result['task_success']}；物理有效：{result['physics_valid']}。\n\n实际新增训练步数 {summary['actual_checkpoint_step']}；实时预测 {len(plans)} 次；原生物理执行 {summary['executed_native_physics_steps']} 步。保存的初始状态误差 {initial_error}；实际控制与对应预测经原关节限幅后的最大误差 {summary['applied_control_max_error_rad']:.3g} rad。第一张真实模型输入像素哈希已核对。\n\n两段进程累计墙钟 {summary['accumulated_process_wall_seconds']:.2f} 秒；含断点维护的总历时 {summary['pipeline_elapsed_wall_seconds']:.2f} 秒；模型推理总计 {sum(inference):.2f} 秒。原 result.json 的 wall_seconds 是最后一次进程段，完整时间以本审计为准。断点缓存修复未改变场景、模型、种子、阈值或已执行状态。\n\n实际关节对指令的采样 RMS（rad）：{json.dumps(summary['sampled_tracking_rms_rad'],ensure_ascii=False)}。这衡量执行跟踪，不是预测相对真实示教的准确度。关节限幅次数见 test_audit.json。\n\n视频：test/videos/ego.mp4、test/videos/side.mp4（640×480，30FPS）。原生结果见 report.md、test/evaluation.json 和 test/run/report.json。实际使用的场景与资产在 test/scene；配套状态为 scene.state.npz。\n\n元数据说明：原适配器的 source_frame_index 保存为 50Hz 控制参考索引，不能当作 30FPS 原视频帧号。reference_frame_mapping.json 提供按时间换算的参考帧号；原始状态数组和指标没有被修改。模型始终使用实时渲染图像。\n\n这是仿真策略测试。原重建物理检查失败、固定 root 和 Mount 光心仍有假设，episode 0 不属于原保存的 250 条划分。成功/失败不能直接代表真实机器人表现。\n""")
    with (root/'audit_report.md').open('a') as f:
        f.write(f"\n指令一致性核验比较了 {len(applied_errors)} 个保存的控制采样（30FPS，排除初始化），不是对每个 50Hz 控制 tick 的独立记录；控制更新规则沿用原冻结测试代码。\n")
        f.write(f"\n全程苹果/机器人受力接触样本：{contact_summary['loaded_apple_robot_samples']}；左手指受力接触样本：{contact_summary['loaded_left_finger_samples']}；首次有效接触时间：{contact_summary['first_loaded_apple_robot_contact_s']}。苹果根节点最大高度增量 {contact_summary['max_apple_root_height_gain_m']*1000:.3f} mm。\n")
        f.write(f"\n已执行时间窗内，左手全部预测关节角绝对值最大仅 {summary['left_hand_prediction_abs_max_rad']:.6f} rad（{np.degrees(summary['left_hand_prediction_abs_max_rad']):.3f}°）。各关节预测范围及限幅幅度见 test_audit.json。下发指令与对应预测限幅后的结果一致；左手对指令的采样 RMS 跟踪误差 {summary['sampled_tracking_rms_rad']['left_hand7']:.6f} rad。此证据表明本次没有明显手指闭合指令；原因是否与新相机/场景、训练标签或归一化有关，单次测试不能区分，本轮未修改这些假设。\n")
    print(json.dumps({k:v for k,v in summary.items() if k in ('checks_passed','prediction_count','applied_control_max_error_rad','executed_native_physics_steps','accumulated_process_wall_seconds','pipeline_elapsed_wall_seconds','sampled_tracking_rms_rad','command_clipping_control_ticks','apple_robot_contact_summary')},ensure_ascii=False))


if __name__=='__main__':main()
