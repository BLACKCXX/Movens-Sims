"""Versioned single/batch stages with atomic receipts, locks and checked resume."""
from __future__ import annotations
import argparse
import concurrent.futures
import copy
import csv
import fcntl
import html
import json
import os
import shutil
import sys
import time
from pathlib import Path
import mujoco
import numpy as np
from .util import ROOT,code_identity,digest,now,read,save_rgb,sha,video_info,write

STAGES=['prepare','fit','validate','render']

def fingerprint(imported,config,enable_gpt):
    return dict(format='run_identity_v1',input_manifest_sha256=sha(Path(imported)/'source_manifest.json'),frozen_batch_manifest_sha256=sha(ROOT/config['manifest']),imported_files=read(Path(imported)/'source_manifest.json')['imported_files'],config_sha256=digest(config),code_hashes=code_identity(),prompt_sha256=sha(ROOT/'prompts'/(config['prompt_version']+'.txt')),mujoco_version=mujoco.__version__,gpt_enabled=enable_gpt)

def receipt_valid(run,stage,identity):
    path=Path(run)/'receipts'/(stage+'.json')
    if not path.exists():return False
    r=read(path)
    if r.get('identity_sha256')!=digest(identity) or not r.get('complete'):return False
    for row in r['outputs']:
        p=Path(run)/row['path']
        if not p.is_file() or sha(p)!=row['sha256']:return False
    return True

def complete_receipt(run,stage,identity,paths,wall_seconds):
    outputs=[]
    for p in paths:
        p=Path(p)
        if not p.is_file() or p.stat().st_size==0:raise ValueError(f'Incomplete {stage} output {p}')
        outputs.append(dict(path=str(p.relative_to(run)),sha256=sha(p),bytes=p.stat().st_size))
    write(Path(run)/'receipts'/(stage+'.json'),dict(stage=stage,complete=True,completed_utc=now(),identity_sha256=digest(identity),wall_seconds=wall_seconds,outputs=outputs))

def prepare(imported,run):
    from .scene import prepare_scene,Scene
    from .vision import prepare_observation,load_observation,frame_at
    from .render import Renderer
    from .evaluate import image_metrics
    prepare_scene(imported,run);prepare_observation(imported,run)
    s=Scene(Path(run)/'scene/scene_baseline.xml');reference,target,valid=load_observation(run)
    with Renderer(s.model,reference.shape[1],reference.shape[0]) as r:masks=r.masks(s.data,s.geom_sets)
    before=image_metrics(masks,target,valid)
    baseline=np.asarray(__import__('PIL.Image',fromlist=['Image']).open(Path(run)/'baseline_ego.png'))
    before['rgb_mean_absolute_error']=float(np.abs(reference.astype(float)-baseline.astype(float)).mean())
    write(Path(run)/'metrics_before.json',before)
    old,_=frame_at(Path(imported)/'old_sim.mp4');save_rgb(Path(run)/'old_sim_frame0.png',old)
    with_robot=np.asarray(__import__('PIL.Image',fromlist=['Image']).open(Path(run)/'baseline_with_robot.png'))
    write(Path(run)/'baseline_correspondence.json',dict(source_frame=0,old_simulation_vs_original_loader_mae_0_255=float(np.abs(old.astype(float)-with_robot.astype(float)).mean()),old_video_lossy_encoded=True,baseline_robot_removed=True,source_state_index=read(Path(run)/'initial_transforms.json')['source_state_index'],camera_transform_errors=read(Path(run)/'resource_checks.json')['camera_world_position_error_m']))
    return [Path(run)/p for p in ['baseline_ego.png','baseline_with_robot.png','reference.png','reference_selection.json','masks/observations.npz','metrics_before.json','resource_checks.json','initial_transforms.json','scene/scene_baseline.xml','scene/scene_fitting.xml','scene/scene_physics_baseline.xml','scene/physics_initial_state.npz','scene/scene_context.json']]+list((Path(run)/'scene/assets').glob('*'))

def fitted_scene(run):
    from .scene import Scene
    s=Scene(Path(run)/'scene/scene_fitting.xml')
    if (Path(run)/'parameters_optimized.json').exists():s.apply(read(Path(run)/'parameters_optimized.json'))
    return s

def fit(run,config,enable_gpt,resume,stages=None):
    from .fit import Fitter,save_final_images
    from .gpt_adapter import run_suggestions
    s=fitted_scene(run);fitter=Fitter(s,run,config,resume=resume)
    try:
        numerical_complete=(Path(run)/'fit_summary.json').exists() and read(Path(run)/'fit_checkpoint.json').get('complete',False)
        if not (resume and numerical_complete):fitter.run_numeric(stages)
        # Vision advice can still be recorded after render budget exhaustion; no
        # proposal is applied without an available local trial evaluation.
        run_suggestions(fitter,config,enable_gpt)
        save_final_images(s,run);s.export();write(Path(run)/'fit_final_state.json',dict(params=s.params,evaluations=fitter.evaluations,best_loss=fitter.best_loss,completed_utc=now()))
    finally:fitter.close()
    return [Path(run)/p for p in ['optimized_ego.png','overlay.png','comparison.png','metrics_after.json','parameters_optimized.json','parameter_registry.json','fit_summary.json','fit_final_state.json','candidates.jsonl','gpt_summary.json','scene/scene_optimized.xml','scene/scene_optimized.state.npz','scene/scene_optimized.state.json']]+list((Path(run)/'gpt').rglob('*.json'))+[p for p in (Path(run)/'scene/assets').glob('depth_*.obj')]

def validate(run,config):
    from .evaluate import settle_diagnostic,geometry_metrics
    s=fitted_scene(run);selection=read(Path(run)/'reference_selection.json');physical=settle_diagnostic(s,run,config['diagnostic'],config['thresholds'],selection['width'],selection['height'])
    write(Path(run)/'geometry_checks.json',geometry_metrics(s))
    return [Path(run)/p for p in ['settle_diagnostic.mp4','physical_diagnostic.json','diagnostic_states.npz','diagnostic_timeline.json','geometry_checks.json','scene/scene_physics.xml','scene/scene_physics.state.npz','scene/scene_physics.state.json']]

def render(run,config):
    from .render import static_video
    s=fitted_scene(run);selection=read(Path(run)/'reference_selection.json');static_video(s,run,config['render'],selection['width'],selection['height'])
    return [Path(run)/p for p in ['ego_static.mp4','ego_static_definition.json']]

def finalize(run,entry,config,identity):
    from .evaluate import visual_pass
    after=read(Path(run)/'metrics_after.json');physical=read(Path(run)/'physical_diagnostic.json');gpt=read(Path(run)/'gpt_summary.json');before=read(Path(run)/'metrics_before.json')
    numerical_pass=visual_pass(after,config['thresholds'])
    review=['Single view; nominal scale and hidden geometry remain ambiguous','Real cloth wrinkles and apple red/yellow appearance are not fully reproduced','Table extents inferred from visible cloth coverage, not independently measured','Inherited fixed plate has no free-resting physical validation']
    if not numerical_pass:review.append('One or more fixed image/coverage thresholds not met')
    if not physical['physics_passed']:review.append('Physical diagnostic incomplete or outside threshold; see physical_diagnostic.json')
    result=dict(episode=entry['episode'],candidate=entry['candidate'],input_complete=True,processing_complete=all(receipt_valid(run,s,identity) for s in STAGES),image_thresholds_passed=numerical_pass,visual_qualified=False,visual_review_status='needs_review',visual_review_record='Batch report includes actual comparison images; known appearance/extent limits prevent full-scene visual approval',physical_qualified=physical['physics_passed'],needs_review=True,review_reasons=review,metrics_before=before,metrics_after=after,physical_diagnostic=physical,gpt_api_verified=gpt['api_actually_verified'],gpt_calls=gpt['actual_calls'],numeric_evaluations=read(Path(run)/'fit_final_state.json')['evaluations'],identity_sha256=digest(identity),definition='Initial fixed-camera static simulation, not real capture or an interaction replay',completed_utc=now())
    write(Path(run)/'result.json',result);return result

def process(entry,config,selected_stages,enable_gpt=False,resume=False,dry_run=False,fit_stages=None):
    ep=entry['episode'];imported=ROOT/'data'/f'episode_{ep:06}';run=ROOT/config['run_root']/f'episode_{ep:06}';run.mkdir(parents=True,exist_ok=True)
    if dry_run:return dict(episode=ep,planned_stages=selected_stages,run=str(run),input_present=(imported/'source_manifest.json').exists())
    from .ingest import import_episode
    imported=import_episode(entry,config['source_project']);identity=fingerprint(imported,config,enable_gpt)
    lock=(run/'.pipeline.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError(f'Episode {ep} already has an active pipeline')
    try:
        idfile=run/'run_identity.json'
        if idfile.exists() and read(idfile)!=identity:raise ValueError(f'Input/code/config/prompt identity changed for {run}. Choose a new --run-root; incompatible completion is never reused.')
        write(idfile,identity);shutil.copy2(imported/'source_manifest.json',run/'source_manifest.json');write(run/'resolved_config.json',config)
        commands=run/'commands.json'
        history=read(commands) if commands.exists() else []
        history.append(dict(utc=now(),argv=sys.argv,stages=selected_stages,gpt_enabled=enable_gpt,resume=resume));write(commands,history)
        for stage in selected_stages:
            if resume and receipt_valid(run,stage,identity):print(f'episode {ep:06} {stage}: verified resume',flush=True);continue
            for dependency in STAGES[:STAGES.index(stage)]:
                if not receipt_valid(run,dependency,identity):raise ValueError(f'{stage} requires complete compatible {dependency} for episode {ep}')
            # A changed/missing downstream stage receipt can never claim overall completion.
            if (run/'result.json').exists():
                old=read(run/'result.json');old['processing_complete']=False;write(run/'result.json',old)
            start=time.monotonic();write(run/'status.json',dict(episode=ep,stage=stage,status='running',utc=now()))
            print(f'episode {ep:06} {stage}: start',flush=True)
            if stage=='prepare':outputs=prepare(imported,run)
            elif stage=='fit':outputs=fit(run,config,enable_gpt,resume,fit_stages)
            elif stage=='validate':outputs=validate(run,config)
            elif stage=='render':outputs=render(run,config)
            complete_receipt(run,stage,identity,outputs,time.monotonic()-start)
            print(f'episode {ep:06} {stage}: complete {time.monotonic()-start:.2f}s',flush=True)
        if all(receipt_valid(run,s,identity) for s in STAGES):result=finalize(run,entry,config,identity)
        else:result=dict(episode=ep,processing_complete=False,finished_stages=[s for s in STAGES if receipt_valid(run,s,identity)])
        write(run/'status.json',dict(episode=ep,status='complete' if result.get('processing_complete') else 'partial',utc=now()));return result
    except Exception as e:
        write(run/'status.json',dict(episode=ep,status='error',error_type=type(e).__name__,message=str(e) if not 'API' in type(e).__name__ else 'See redacted API call status',utc=now()));raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def report(manifest,config):
    root=ROOT/config['run_root'];root.mkdir(parents=True,exist_ok=True);rows=[];cards=[]
    for e in manifest['entries']:
        ep=e['episode'];run=root/f'episode_{ep:06}';res=read(run/'result.json') if (run/'result.json').exists() else dict(episode=ep,input_complete=(ROOT/'data'/f'episode_{ep:06}/source_manifest.json').exists(),processing_complete=False,visual_qualified=False,physical_qualified=False,needs_review=True)
        row={k:res.get(k) for k in ['episode','input_complete','processing_complete','image_thresholds_passed','visual_qualified','physical_qualified','needs_review','numeric_evaluations','gpt_calls','gpt_api_verified']}
        for label in ('apple','plate'):
            for phase in ('before','after'):
                o=res.get('metrics_'+phase,{}).get('objects',{}).get(label,{})
                for metric in ('iou','symmetric_edge_distance_px','centroid_error_px'):row[f'{label}_{metric}_{phase}']=o.get(metric)
        row['cloth_coverage']=res.get('metrics_after',{}).get('table',{}).get('cloth_visible_background_coverage');row['apple_max_drift_m']=res.get('physical_diagnostic',{}).get('apple_translation_max_m');row['apple_max_penetration_m']=res.get('physical_diagnostic',{}).get('apple_contact_penetration_max_m');rows.append(row)
        prefix=f'episode_{ep:06}'
        cards.append(f'<article><h2>Episode {ep:06}</h2><p>静态仿真展示 · 非真实采集 · 非交互回放</p><video controls preload="metadata" src="{prefix}/ego_static.mp4"></video><details><summary>重力接触诊断（动态模型）</summary><video controls preload="none" src="{prefix}/settle_diagnostic.mp4"></video></details><p>真实 / 优化后 / 像素轮廓叠加</p><a href="{prefix}/comparison.png"><img src="{prefix}/comparison.png" alt="comparison"></a><p><a href="{prefix}/baseline_ego.png">优化前</a> · <a href="{prefix}/reference_masks_review.png">参考掩码</a> · <a href="{prefix}/result.json">完整结果</a> · <a href="{prefix}/scene/scene_optimized.xml">XML</a></p><p>{html.escape(str(row))}</p></article>')
    with (root/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    counts={k:sum(bool(r.get(k)) for r in rows) for k in ['input_complete','processing_complete','image_thresholds_passed','visual_qualified','physical_qualified','needs_review','gpt_api_verified']}
    write(root/'summary.json',dict(episodes=len(rows),counts=counts,manifest_sha256=sha(ROOT/config['manifest']),config_sha256=digest(config),definitions='Input complete, processed, numeric image thresholds, full scene visual qualification, physical qualification and review counts are separate. No real dynamics claim.',results=rows))
    doc='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Movens-Sim 初始静态重建</title><style>body{font:16px system-ui;background:#15191e;color:#e4e8ed;max-width:1200px;margin:32px auto;padding:0 20px}a{color:#7bcaff}article{background:#20262d;padding:22px;margin:24px 0;border-radius:12px}img{width:100%}video{max-width:640px;width:100%}p{overflow-wrap:anywhere}summary{cursor:pointer}</style><h1>Movens-Sim · 首批初始静态桌面</h1><p>固定第一视角、保持初始布局的三维 MuJoCo 渲染，默认 5 秒 / 30 FPS。重力检查是独立动态副本。单视角、名义尺度；源盘子固定，真实动力学未验证。</p><p>'+html.escape(str(counts))+'</p><p><a href="results.csv">results.csv</a> · <a href="summary.json">summary.json</a></p>'+''.join(cards)+'</html>'
    (root/'index.html').write_text(doc);return counts

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['discover','import','prepare','fit','validate','render','report','run']);p.add_argument('--config',default='configs/batch_v1.json');p.add_argument('--manifest');p.add_argument('--episode',type=int);p.add_argument('--run-root');p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--gpt','--terra',dest='gpt',action='store_true');p.add_argument('--local-only',action='store_true');p.add_argument('--concurrency',type=int,default=1);p.add_argument('--stage',choices=STAGES);p.add_argument('--fit-stages',nargs='+',choices=['camera','layout','size','appearance']);args=p.parse_args(argv)
    config=read(ROOT/args.config)
    if args.manifest:config['manifest']=args.manifest
    if args.run_root:config['run_root']=args.run_root
    if args.concurrency<1 or args.concurrency>4:p.error('Concurrency must be bounded to 1..4')
    if args.command=='discover':
        from .ingest import discover
        m=discover(config['source_project']);print([e['episode'] for e in m['entries']]);return
    manifest=read(ROOT/config['manifest']);entries=[e for e in manifest['entries'] if args.episode is None or e['episode']==args.episode]
    if not entries:p.error('Episode is not in the frozen manifest')
    if config.get('workflow') in ('mount_joint_v3','classified_reconstruction_v4','terra_closed_loop_v5'):
        from .reconstruction.workflow import run_pipeline
        if config.get('workflow')=='classified_reconstruction_v4':
            from .reconstruction.classified import run_pipeline
        if config.get('workflow')=='terra_closed_loop_v5':
            from .reconstruction.v5 import run_pipeline
        if args.episode is None and not args.dry_run:p.error('v3 pilot requires an explicit --episode; no implicit batch run')
        if args.concurrency!=1:p.error('Use the durable full-batch coordinator for concurrency; per-episode workers use concurrency=1')
        for entry in entries:
            result=run_pipeline(entry,config,args.gpt and not args.local_only,args.resume,args.dry_run,args.stage or args.command)
            if args.dry_run:print(json.dumps(result,ensure_ascii=False))
        return
    if args.command=='import':
        from .ingest import import_episode
        for e in entries:
            if args.dry_run:print(e)
            else:print(import_episode(e,config['source_project']))
        return
    if args.command=='report':print(report(manifest,config));return
    stages=[args.stage] if args.stage else STAGES if args.command=='run' else [args.command]
    jobs=[(e,config,stages,args.gpt and not args.local_only,args.resume,args.dry_run,args.fit_stages) for e in entries]
    if args.dry_run:
        for job in jobs:print(json.dumps(process(*job),ensure_ascii=False),flush=True)
        return
    errors=[]
    # Parallel process isolation is for EGL/physics workers, not separate AI agents.
    if args.concurrency==1:
        for job in jobs:
            try:process(*job)
            except Exception as exc:print(f'episode {job[0]["episode"]}: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True);errors.append(job[0]['episode'])
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.concurrency) as pool:
            pending={pool.submit(process,*job):job[0]['episode'] for job in jobs}
            for future,ep in pending.items():
                try:future.result()
                except Exception as exc:print(f'episode {ep}: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True);errors.append(ep)
    if not args.dry_run:print(report(manifest,config),flush=True)
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
