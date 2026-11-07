"""Formal versioned pipeline: Mount -> joint geometry -> Terra -> appearance -> freeze."""
from __future__ import annotations
import asyncio
import copy
import csv
import fcntl
import html
import json
import os
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

from .budget import BudgetExceeded
from .metrics import measure,gate
from .mcp_server import call_tools
from .prepare import prepare
from .session import ReconstructionSession
from .terra import request_review
from ..head_camera import labelled_comparison,rest_diagnostic
from ..render import Renderer,overlay,video_writer
from ..scene import look_at,quat_from_matrix
from ..util import ROOT,CODE_ROOT,append,digest,now,read,save_rgb,sha,video_info,write


def identity(run,config):
    code={str(p.relative_to(CODE_ROOT)):sha(p) for p in sorted((CODE_ROOT/'src').rglob('*.py'))}
    for p in [CODE_ROOT/'scripts/movens-sim',CODE_ROOT/'pyproject.toml',CODE_ROOT/'prompts'/(config['terra']['prompt_version']+'.txt')]:code[str(p.relative_to(CODE_ROOT))]=sha(p)
    return dict(format='mount_joint_v3_identity',code=code,config_sha256=digest(config),source_manifest_sha256=sha(run/'source_manifest.json'),observation_sha256=sha(run/'observation.json'),mujoco=mujoco.__version__)


def verify_sources(run):
    manifest=read(run/'source_manifest.json')
    for name,expected in {**manifest.get('source_files',{}),**manifest.get('immutable_assets',{})}.items():
        if not Path(name).is_file() or sha(name)!=expected:raise ValueError('Immutable source/asset changed: '+name)
    for name,expected in manifest['inputs'].items():
        path=run/name
        if not path.is_file() or sha(path)!=expected:raise ValueError('Immutable input missing or changed: '+name)
    for name,expected in read(run/'observation.json')['mask_sha256s'].items():
        if sha(run/name)!=expected:raise ValueError('Observation changed without versioned reevaluation: '+name)


def groups(model):
    result={}
    for label in ('apple','plate','table'):
        members={model.body(label+'_root').id}
        for b in range(model.nbody):
            if int(model.body_parentid[b]) in members:members.add(b)
        result[label]=np.flatnonzero(np.isin(model.geom_bodyid,list(members)))
    result['cloth']=np.array([g for g in range(model.ngeom) if model.geom(g).name.startswith('frozen_') or model.geom(g).name=='cloth_continuation'])
    return result


def cached_images(session,best=None):
    best=best or session.evaluator.best.get('appearance_640x480') or session.evaluator.best['geometry_640x480']
    with np.load(session.run/'evaluation_cache'/(best['cachekey']+'.npz')) as z:return z['rgb'].copy(),{k:z[k].copy() for k in ('apple','plate','cloth','table')}


def old_baseline(session):
    if session.config.get('initialization',{}).get('mode')=='fk_h0':return
    path=session.run/'metrics_old_batch.json'
    if path.exists():return
    ep=f"episode_{read(session.run/'source_manifest.json')['episode']:06}"
    source=ROOT/session.config['legacy_comparison_root']/ep/'scene/scene_optimized.xml'
    with session.budget.computation(dict(kind='old_baseline_comparison')):
        events=session.budget.reserve_render('render_initial',dict(tag='old_batch_same_observation_comparison',width=640,height=480,reference_frame=0))
        model=mujoco.MjModel.from_xml_path(str(source));data=mujoco.MjData(model)
        with np.load(source.with_suffix('.state.npz')) as z:data.qpos[:]=z['qpos'];data.qvel[:]=z['qvel']
        mujoco.mj_forward(model,data)
        with Renderer(model) as renderer:rgb=renderer.rgb(data);masks=renderer.masks(data,groups(model))
        metrics=measure(rgb,masks,session.observation,session.config);write(path,metrics);save_rgb(session.run/'old_batch_ego.png',rgb)
        for event in events:session.budget.finish(event)
        session.budget.record('render_passes',dict(rgb=1,segmentation=1,tag='old_batch'))


def selected_metrics(session):
    return session.evaluator.best['geometry_640x480']['metrics']


def actions_to_calls(session,parsed):
    calls=[];ids=set();shape=[];updates={}
    cameras=[a for a in parsed['proposed_actions'] if a['tool']=='review_camera_hypothesis']
    if cameras:
        a=cameras[0]
        return [dict(tool='review_camera_hypothesis',arguments=dict(evidence=a['evidence'],confidence=a['confidence'],_current_candidate_id=True))]
    for action in parsed['proposed_actions']:
        tool=action['tool'];values={row['param_id']:row['value'] for row in action['updates']}
        if tool=='update_appearance':
            session.state.setdefault('deferred_appearance_actions',[]).append(action);continue
        if tool=='update_shape_or_asset':
            shape.append(dict(tool=tool,arguments=dict(updates=values,evidence=action['evidence'],confidence=action['confidence'],_current_candidate_id=True)))
        elif tool=='propose_or_apply_object_update':updates.update(values)
        ids.update(action['parameter_ids']);ids.update(values)
    if not ids and not shape:return []
    # Center/size repairs always retain their joint degrees of freedom.
    for label in ('apple','plate'):
        if any(k.startswith(label+'.') for k in ids):ids.update([label+'.x',label+'.y',label+'.scale'])
    calls.append(dict(tool='checkpoint',arguments=dict(label='before_repair_'+str(len(session.state['rounds'])),_current_candidate_id=True)))
    calls.extend(shape)
    if updates:calls.append(dict(tool='propose_or_apply_object_update',arguments=dict(updates=updates,apply=True,_current_candidate_id=True)))
    fits=[a for a in parsed['proposed_actions'] if a['tool']!='update_appearance']
    calls.append(dict(tool='fit_geometry',arguments=dict(parameter_ids=sorted(ids),evidence='; '.join(a['evidence'] for a in fits),confidence=min(a['confidence'] for a in fits),_current_candidate_id=True)))
    return calls


def resume_pending_fit(session):
    pending=[e for e in session.budget.summary()['events'] if e['kind']=='geometry_rounds' and e['status']=='reserved']
    if not pending or not session.state['rounds']:return
    args=dict(parameter_ids=pending[-1]['detail'].get('parameter_ids'))
    if args['parameter_ids'] is None:
        for path in sorted(session.run.glob('mcp_repair_*.json'),reverse=True):
            calls=[row for row in read(path)['calls'] if row['tool']=='fit_geometry' and row['result'].get('error',{}).get('code')=='BatchPaused']
            if calls:
                args={k:v for k,v in calls[-1]['arguments'].items() if k!='candidate_id'};break
    session.state['operations'].append(dict(tool='resume_pending_geometry_round',reservation_id=pending[-1]['id'],parameter_ids=args['parameter_ids'],budget_reset=False));session.save()
    session.fit_geometry(**args)


def geometry_and_reviews(session,enabled):
    run=session.run;config=session.config
    old_baseline(session)
    resume_pending_fit(session)
    if not session.state['rounds']:
        print('Initial joint geometry fit: both source-relative scales active',flush=True)
        session.fit_geometry()
    if 'geometry_640x480' not in session.evaluator.best:raise RuntimeError('No valid baseline/candidate available')
    if not (run/'metrics_mount_baseline.json').exists():
        # The first full-resolution event is the selected mount/layout baseline.
        history=[json.loads(line) for line in (run/'candidates.jsonl').read_text().splitlines()]
        baseline=next(row for row in history if row['tag']=='mount_baseline' and row['metrics'])
        write(run/'metrics_mount_baseline.json',baseline['metrics'])
        p=baseline['params'];key=digest(dict(candidate=digest(dict(source=session.scene.source_hash,params=p,camera=session.scene.locked_camera))[:20],width=640,height=480,stage='geometry',renderer=config['render'],observation=session.observation['version'],evaluator=config['evaluator'],mujoco=mujoco.__version__))
        with np.load(run/'evaluation_cache'/(key+'.npz')) as z:save_rgb(run/'mount_layout_baseline.png',z['rgb'])
    baseline=read(run/'metrics_mount_baseline.json')
    while True:
        best=session.evaluator.best['geometry_640x480'];session.scene.apply(best['params'],internal=True);session.save()
        metrics=best['metrics'];numeric=gate(metrics,session.scene.geometry_metrics(),baseline,config)
        if (run/'mask_review.json').exists() and not read(run/'mask_review.json').get('mask_quality_ok',True):
            numeric['geometry_numeric_passed']=False;numeric['failures'].append('observation.mask_requires_review')
        write(run/'geometry_gate.json',numeric)
        rgb,masks=cached_images(session,best)
        print('Geometry',len(session.state['rounds']),'IoU',{k:round(v['iou'],4) for k,v in metrics['objects'].items()},'numeric pass',numeric['geometry_numeric_passed'],flush=True)
        status,parsed=request_review(session,metrics,rgb,masks,enabled=enabled)
        session.state.setdefault('reviews',[]).append(status);session.state['geometry_review_passed']=bool(parsed and parsed['verdict']=='pass' and numeric['geometry_numeric_passed']);session.save()
        if session.state['geometry_review_passed']:break
        if not parsed:
            session.state['stop_reason']='terra_unavailable_or_unverified';break
        if session.budget.remaining('geometry_rounds')<1 or session.budget.remaining('render_repair')<1 or session.budget.remaining('terra_requests')<=1:
            session.state['stop_reason']='geometry_or_review_budget_exhausted';break
        if not parsed['proposed_actions']:
            session.state['stop_reason']='review_without_executable_repair_plan';break
        calls=actions_to_calls(session,parsed)
        if not calls:
            session.state['stop_reason']='only_deferred_or_unsupported_actions';break
        before=best['candidate_id'];session.state['operations'].append(dict(review=status['folder'],actions=parsed['proposed_actions']));session.save()
        session.close()
        report=asyncio.run(call_tools(run,calls));write(run/f'mcp_repair_{len(session.state["rounds"]):02}.json',report)
        session=ReconstructionSession(run,config)
        if any(not row['result'].get('ok') for row in report['calls']):
            errors=[row['result'].get('error',{}) for row in report['calls'] if not row['result'].get('ok')]
            if any(e.get('code')=='BatchPaused' for e in errors):
                from .control import BatchPaused
                raise BatchPaused('stopped','MCP safe-stop checkpoint; geometry remains resumable',0)
            session.state['stop_reason']='tool_repair_rejected';break
        after=session.evaluator.best.get('geometry_640x480',{}).get('candidate_id')
        if before==after:
            session.state['stop_reason']='no_substantial_repair_improvement';break
    best=session.evaluator.best['geometry_640x480'];session.scene.apply(best['params'],internal=True)
    rgb,masks=cached_images(session,best);save_rgb(run/'joint_geometry_ego.png',rgb);save_rgb(run/'geometry_overlay.png',overlay(session.observation['rgb'],masks,session.observation['target']))
    write(run/'metrics_geometry.json',best['metrics']);write(run/'geometry_parameters.json',best['params']);write(run/'geometry_checks.json',session.scene.geometry_metrics());session.state['geometry_stage_complete']=True;session.save()
    return session


def appearance(session):
    run=session.run
    if not session.state['geometry_review_passed']:
        session.state['appearance_status']='blocked_geometry_or_terra_gate';session.save();return
    session.scene.stage='appearance';session.save();e=session.evaluator;e.stage='appearance';e.pool='render_appearance'
    geometry={k:v for k,v in session.scene.params.items() if 'geometry' in next(p['stages'] for p in session.scene.registry() if p['id']==k)}
    write(run/'geometry_frozen.json',dict(geometry_params=geometry,objects=selected_metrics(session)['objects'],candidate_id=session.scene.candidate_id(),camera=session.scene.locked_camera))
    try:
        with session.budget.computation(dict(kind='complete_appearance_search'),whole_task_slice=True):
            e.evaluate(session.scene.params.copy(),'appearance_baseline',640,480)
            for index,(ids,cap) in enumerate(zip(session.config['appearance']['subsets'],session.config['appearance']['render_subbudgets'])):
                e.optimize(ids,cap,f'appearance_step_{index}',640,480)
    except BudgetExceeded as exc:session.state['appearance_stop_reason']=str(exc)
    best=e.best.get('appearance_640x480')
    if best:session.scene.apply(best['params'],internal=True)
    session.state['appearance_status']='completed';session.save()


def reload_delivery(session,xml,expected_rgb,expected_masks):
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
    with np.load(xml.with_suffix('.state.npz')) as z:data.qpos[:]=z['qpos'];data.qvel[:]=z['qvel']
    mujoco.mj_forward(model,data)
    with Renderer(model) as renderer:rgb=renderer.rgb(data);masks=renderer.masks(data,groups(model))
    cam=model.camera('approx_ego_camera').id
    errors=dict(camera_position_m=float(np.max(abs(data.cam_xpos[cam]-session.scene.data.cam_xpos[session.scene.cam]))),camera_rotation=float(np.max(abs(data.cam_xmat[cam]-session.scene.data.cam_xmat[session.scene.cam]))),rgb_mae=float(np.abs(rgb.astype(float)-expected_rgb.astype(float)).mean()),mask_changed_pixels={k:int(np.count_nonzero(masks[k]!=expected_masks[k])) for k in ('apple','plate','cloth')},object_position_m={label:float(np.max(abs(data.xpos[model.body(label+'_root').id]-session.scene.data.xpos[session.scene.model.body(label+'_root').id]))) for label in ('apple','plate')})
    dimension_errors={}
    for label in ('apple','plate'):
        points=[]
        for g in groups(model)[label]:
            if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[g]!=1:continue
            mesh=int(model.geom_dataid[g]);a=int(model.mesh_vertadr[mesh]);n=int(model.mesh_vertnum[mesh]);points.append(model.mesh_vert[a:a+n]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g])
        dimension_errors[label]=float(np.max(abs(np.ptp(np.concatenate(points),axis=0)-np.ptp(session.scene.vertices(label),axis=0))))
    errors['dimension_m']=dimension_errors
    passed=errors['camera_position_m']<1e-9 and errors['camera_rotation']<1e-9 and errors['rgb_mae']<.05 and max(errors['mask_changed_pixels'].values())<=4 and max(dimension_errors.values())<1e-7 and max(errors['object_position_m'].values())<1e-9
    write(session.run/'reload_check.json',dict(passed=passed,errors=errors,xml_sha256=sha(xml),state_sha256=sha(xml.with_suffix('.state.npz')),source_independent=True,robot_removed='pelvis' not in [model.body(i).name for i in range(model.nbody)]))
    # Auxiliary views use an independent model; the delivered ego camera stays fixed.
    center=(data.xpos[model.body('apple_root').id]+data.xpos[model.body('plate_root').id])/2
    for name,offset in [('support_side',np.array([0.,-.7,.08])),('support_oblique',np.array([-.4,-.5,.4]))]:
        eye=center+offset;model.cam_pos[cam]=eye;model.cam_quat[cam]=quat_from_matrix(look_at(eye,center));model.cam_fovy[cam]=50;mujoco.mj_forward(model,data)
        with Renderer(model) as renderer:save_rgb(session.run/(name+'.png'),renderer.rgb(data))
        session.budget.record('frozen_output_render',dict(view=name,rgb_passes=1,used_for_search=False))
    return passed,rgb,masks


def export_and_review(session,enabled):
    run=session.run;scene=session.scene
    scene.stage='frozen';session.save();best=session.evaluator.best.get('appearance_640x480') or session.evaluator.best['geometry_640x480'];expected_rgb,expected_masks=cached_images(session,best)
    if not (run/'delivery_binding.json').exists():
        xml=scene.export(run/'delivery/static');passed,rgb,masks=reload_delivery(session,xml,expected_rgb,expected_masks)
        save_rgb(run/'final_ego.png',rgb);save_rgb(run/'overlay.png',overlay(session.observation['rgb'],masks,session.observation['target']));np.savez_compressed(run/'final_masks.npz',**masks)
        metrics=measure(rgb,masks,session.observation,session.config);write(run/'metrics_after.json',metrics)
        diagnostic_config=dict(diagnostic=session.config['diagnostic'],thresholds=session.config['acceptance'])
        print('Frozen export/reload complete; running separate free-object gravity diagnostic',flush=True)
        physical=rest_diagnostic(scene,run,diagnostic_config);physical['mass_policy']='Immutable source density preserved; uniform mass s^3 and inertia s^5; source values unmeasured, not fitted';write(run/'physical_diagnostic.json',physical)
        n=150;info=video_writer(run/'ego_static.mp4',(rgb for _ in range(n)),30);write(run/'ego_static_definition.json',dict(type='fixed_camera_initial_static_3d_simulation',real_capture=False,interaction_replay=False,dynamics_validated=False,state='delivery/static/portable/scene.state.npz',scene='delivery/static/portable/scene.xml',video=info))
        write(run/'parameters_final.json',scene.params);write(run/'parameter_registry.json',scene.registry());write(run/'camera_actual.json',scene.locked_camera)
        files={str(p.relative_to(run)):sha(p) for folder in (run/'delivery/static/portable',run/'physics/portable') for p in folder.rglob('*') if p.is_file()}
        for name in ('parameters_final.json','camera_actual.json','final_ego.png','final_masks.npz','ego_static.mp4','settle_diagnostic.mp4','physical_diagnostic.json','metrics_after.json','reload_check.json'):files[name]=sha(run/name)
        binding=dict(candidate_id=scene.candidate_id(),artifacts=files,scene_parameter_asset_hash=digest(files),reload_passed=passed);write(run/'delivery_binding.json',binding)
        session.budget.record('frozen_output_render',dict(view='ego',rgb_passes=1,segmentation_passes=1,video_frames=150,used_for_search=False))
    else:
        binding=read(run/'delivery_binding.json')
        for name,expected in binding['artifacts'].items():
            if not (run/name).exists() or sha(run/name)!=expected:raise ValueError('Frozen delivery is missing/changed: '+name)
        rgb=cv2.cvtColor(cv2.imread(str(run/'final_ego.png')),cv2.COLOR_BGR2RGB)
        with np.load(run/'final_masks.npz') as z:masks={k:z[k].copy() for k in z.files}
        metrics=read(run/'metrics_after.json')
    scene.stage='final_readonly';session.save()
    if not (run/'final_review.json').exists():
        status,parsed=request_review(session,metrics,rgb,masks,final=True,enabled=enabled,delivery_binding=binding)
        write(run/'final_review.json',dict(status=status,review=parsed,delivery_hash=binding['scene_parameter_asset_hash']))
    session.state['complete']=True;session.save()


def report(session):
    run=session.run;budget=session.budget.summary();write(run/'budget_ledger.json',budget)
    after=read(run/'metrics_after.json');before=read(run/'metrics_mount_baseline.json');physical=read(run/'physical_diagnostic.json');review=read(run/'final_review.json');binding=read(run/'delivery_binding.json');numeric=gate(after,session.scene.geometry_metrics(),before,session.config)
    boundary=[]
    for p in session.scene.registry():
        value=session.scene.params[p['id']];distance=min(value-p['lower'],p['upper']-value)/(p['upper']-p['lower'])
        if 'geometry' in p['stages'] and distance<.01:boundary.append(p['id'])
    statuses=[read(p) for p in (run/'terra').rglob('status.json')];actual_requests=sum(s.get('actual_http_attempts',0) for s in statuses)
    result=dict(episode=read(run/'source_manifest.json')['episode'],code_implemented=True,processing_complete=True,workflow='mount_joint_v3',geometry_numeric_passed=numeric['geometry_numeric_passed'],terra_geometry_passed=session.state['geometry_review_passed'],geometry_passed=numeric['geometry_numeric_passed'] and session.state['geometry_review_passed'],appearance_status=session.state.get('appearance_status','not_run'),physical_passed=physical['physics_passed'],reload_passed=read(run/'reload_check.json')['passed'],final_terra_status=review['status']['status'],final_terra_verdict=review['review']['verdict'] if review['review'] else 'unreviewed',model=session.config['terra']['model'],terra_requests_reserved=budget['used'].get('terra_requests',0),terra_actual_http_requests=actual_requests,geometry_fit_rounds=budget['used'].get('geometry_rounds',0),render_optimization_evaluations=budget['used'].get('render_total',0),projection_residual_evaluations=budget['used'].get('projection_evaluations',0),optimization_wall_seconds=budget['used'].get('optimization_seconds',0),stop_reason=session.state.get('stop_reason','geometry_review_passed'),boundary_hits=boundary,metrics_before=before,metrics_after=after,geometry=session.scene.geometry_metrics(),camera=session.scene.locked_camera,delivery_binding=binding,real_dynamics_validated=False,camera_calibrated=False,limitations=['Single-view scale/shape/mount ambiguity; root and optical offset unmeasured','Inherited table extent frozen including visual-only cloth continuation','Plate convex collision interior does not match all concave visual support surfaces','Numerical fit, Terra review, initial support and dynamic rest are distinct statuses'],completed_utc=now())
    result['visual_qualified']=bool(result['geometry_passed'] and result['final_terra_verdict']=='pass' and result['reload_passed'] and not boundary)
    write(run/'result.json',result)
    if session.config.get('batch',{}).get('enabled'):
        init=read(run/'initialization.json');observation=read(run/'observation.json')
        result['initialization']=init;result['observation_quality']=observation['review'];result['input_complete']=True
        result['needs_review']=not result['visual_qualified'] or not result['physical_passed'] or bool(init['needs_review_reasons']) or not observation['review']['mask_quality_ok']
        result['visual_qualified'] &= observation['review']['mask_quality_ok'] and not bool(init['needs_review_reasons'])
        result['limitations'][1]='Original table/cloth geometry fixed at h0; no fitted extent or cloth continuation added'
        result['construction_complete']=bool(result['processing_complete'] and result['reload_passed'])
        write(run/'result.json',result)
    names=['mount_layout_baseline.png','joint_geometry_ego.png','final_ego.png']
    labels=['REAL frame 0','Mount + fixed sizes/layout','Mount + JOINT XY/size fit','Final appearance' if result['appearance_status']=='completed' else 'Appearance NOT RUN; geometry candidate']
    if (run/'old_batch_ego.png').is_file():names.insert(0,'old_batch_ego.png');labels.insert(1,'OLD batch_v1 final')
    images=[session.observation['rgb']]+[cv2.cvtColor(cv2.imread(str(run/name)),cv2.COLOR_BGR2RGB) for name in names]
    save_rgb(run/'comparison.png',labelled_comparison(images,labels))
    csv_rows=[]
    for label in ('apple','plate'):
        a,b=before['objects'][label],after['objects'][label];csv_rows.append(dict(object=label,iou_before=a['iou'],iou_after=b['iou'],center_before_px=a['centroid_error_px'],center_after_px=b['centroid_error_px'],edge_before_px=a['symmetric_edge_distance_px'],edge_after_px=b['symmetric_edge_distance_px'],area_error_before=a['area_relative_error'],area_error_after=b['area_relative_error'],absolute_scale=session.scene.params[label+'.scale'],dimensions_m=json.dumps(result['geometry'][label]['dimensions_world_m'])))
    with (run/'results.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    summary={k:result[k] for k in ('geometry_passed','geometry_numeric_passed','terra_geometry_passed','appearance_status','physical_passed','reload_passed','final_terra_verdict','geometry_fit_rounds','render_optimization_evaluations','projection_residual_evaluations','terra_actual_http_requests','optimization_wall_seconds','boundary_hits','stop_reason')}
    text='# Movens-Sim Mount joint reconstruction v3\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n'+('\n'.join(f"- {row['object']}: IoU {row['iou_before']:.4f} → {row['iou_after']:.4f}, center {row['center_before_px']:.2f} → {row['center_after_px']:.2f}px, source scale {row['absolute_scale']:.4f}, dimensions {row['dimensions_m']} m" for row in csv_rows))+'\n\n'+('\n'.join('- '+x for x in result['limitations']))+'\n\nAppearance only runs after numeric/support AND actual Terra geometry approval. Failed candidates and spent budgets are retained. Frozen static video is not real capture or an interaction replay. See commands.jsonl, code_snapshot, source_manifest, camera_prior, asset_audit, observation, budget_ledger and terra/.\n'
    (run/'report.md').write_text(text)
    links=['report.md','results.csv','result.json','budget_ledger.json','camera_actual.json','asset_audit.json','reload_check.json','physical_diagnostic.json','final_review.json','delivery/static/portable/scene.xml']
    doc='<!doctype html><meta charset="utf-8"><title>Movens-Sim reconstruction</title><style>body{background:#171b21;color:#eee;font:16px system-ui;margin:24px}a{color:#8cd3ff}img{max-width:100%}video{width:640px;max-width:100%}pre{white-space:pre-wrap}</style><h1>Movens-Sim · fixed Mount, joint object size/layout</h1><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><img src="comparison.png"><p>'+ ' · '.join(f'<a href="{p}">{p}</a>' for p in links)+'</p><video controls src="ego_static.mp4"></video><video controls src="settle_diagnostic.mp4"></video><p>Static simulation / independent gravity diagnostic. No real capture, calibrated-camera or validated-real-dynamics claim.</p><img src="overlay.png"><img src="support_side.png">'
    (run/'index.html').write_text(doc)
    return result


def run_pipeline(entry,config,enabled=False,resume=False,dry_run=False,command='run'):
    ep=entry['episode'];run=ROOT/config['run_root']/f'episode_{ep:06}'
    if ep not in config['allowed_episodes']:raise ValueError('Episode not enabled in this pilot configuration')
    if dry_run:return dict(episode=ep,run=str(run),workflow=config['workflow'],budget=config['budget'],model=config['terra']['model'],api_will_be_called=enabled)
    if command=='report':
        if not (run/'result.json').is_file():raise ValueError('No completed result to report; report never starts fitting')
        return read(run/'result.json')
    run.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    with (run/'.workflow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (run/'run_identity.json').exists() and not resume:raise ValueError('Existing run requires --resume; task budget is not reset by another run-root')
        task_registry=ROOT/'runs/_task_budgets'/f'mount_joint_v3_episode_{ep:06}.json'
        if config.get('batch',{}).get('enabled'):task_registry=ROOT/'runs/_task_budgets'/config['task_group_id']/f'episode_{ep:06}.json'
        if task_registry.exists() and read(task_registry)['task_group_id']!=config['task_group_id']:raise ValueError('This trial already has a task-group ledger; changing run_id/task_id cannot bypass its limits')
        write(task_registry,dict(task_group_id=config['task_group_id'],episode=ep,ledger=str(ROOT/'runs/_task_budgets'/(config['task_group_id']+'.sqlite'))))
        if (run/'resolved_config.json').exists() and read(run/'resolved_config.json')!=config:raise ValueError('Configuration changed; do not reuse incompatible completed state')
        write(run/'resolved_config.json',config)
        if not (run/'source_manifest.json').exists():prepare(entry,config,run)
        verify_sources(run);ident=identity(run,config)
        if (run/'run_identity.json').exists():
            previous=read(run/'run_identity.json')
            if previous!=ident:
                proof_path=run/'code_compatibility/approved_continuation.json'
                proof=read(proof_path) if proof_path.exists() else {}
                if proof.get('previous_identity_sha256')!=digest(previous) or proof.get('next_identity_sha256')!=digest(ident):
                    raise ValueError('Code/input/prompt identity mismatch; preserve prior results and task budget for an explicit compatible continuation')
                if any(previous.get(k)!=ident.get(k) for k in ('config_sha256','source_manifest_sha256','observation_sha256','mujoco')):raise ValueError('Compatible code continuation cannot change inputs/config/observations')
                if (run/'session.json').exists() and read(run/'session.json').get('complete'):raise ValueError('Completed scenes keep their original code binding')
                write(run/'code_compatibility/previous_identity.json',previous);write(run/'run_identity.json',ident)
        else:
            write(run/'run_identity.json',ident)
            for name in ident['code']:
                dest=run/'code_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(CODE_ROOT/name,dest)
            write(run/'code_snapshot/resolved_config.json',config)
        append(run/'commands.jsonl',dict(argv=sys.argv,enabled_terra=enabled,utc=now(),identity_sha256=digest(ident)))
        if command=='prepare':return dict(episode=ep,prepared=True,run=str(run))
        session=ReconstructionSession(run,config)
        try:
            if command in ('validate','render') and not session.state.get('geometry_stage_complete'):raise ValueError('Requested stage requires an existing completed fit; it does not start optimization')
            if not (run/'compiled_asset_checks.json').exists():
                model=session.scene.model
                write(run/'compiled_asset_checks.json',dict(mujoco_version=mujoco.__version__,normal_values_finite=bool(np.isfinite(model.mesh_normal).all()),mesh_vertices_finite=bool(np.isfinite(model.mesh_vert).all()),source_geometry=session.scene.geometry_metrics(),plate_topology_note=read(run/'asset_audit.json')['plate_face_status'],scaling_policy='Positive source-relative scale; visual and convex collision assets compiled together; face winding retained',camera_model_body_id=int(model.cam_bodyid[session.scene.cam]),no_robot_body='pelvis' not in [model.body(i).name for i in range(model.nbody)]))
            if session.state.get('complete'):
                # Verify every frozen dependency; never skip a partial/deleted result.
                for name,expected in read(run/'delivery_binding.json')['artifacts'].items():
                    if not (run/name).exists() or sha(run/name)!=expected:raise ValueError('Incomplete frozen delivery: '+name)
                for name in ('ego_static.mp4','settle_diagnostic.mp4','final_review.json','result.json','report.md'):
                    if not (run/name).is_file():raise ValueError('Incomplete completed result: '+name)
                print('Resume verified: task counts preserved, no new optimization or model requests',flush=True);return read(run/'result.json')
            if not (run/'mcp_smoke.json').exists():
                session.close();smoke=asyncio.run(call_tools(run,[dict(tool='inspect_scene',arguments={})]));write(run/'mcp_smoke.json',smoke)
                if not smoke['calls'][0]['result'].get('ok'):raise RuntimeError('Actual MCP smoke call failed')
                session=ReconstructionSession(run,config)
            if not session.state.get('geometry_stage_complete'):session=geometry_and_reviews(session,enabled)
            if not session.state.get('appearance_status'):appearance(session)
            if command=='fit':return dict(episode=ep,fit_complete=True,state=session.state,budget=session.budget.summary()['used'])
            export_and_review(session,enabled);result=report(session)
            append(run/'timings.jsonl',dict(total_pipeline_wall_seconds=time.monotonic()-start,api_wait_seconds=sum(s.get('wall_seconds',0) for s in [read(p) for p in (run/'terra').rglob('status.json')]),optimization_seconds=session.budget.summary()['used'].get('optimization_seconds',0),utc=now()))
            print(json.dumps({k:result[k] for k in ('episode','geometry_passed','appearance_status','physical_passed','terra_actual_http_requests','render_optimization_evaluations','geometry_fit_rounds')},ensure_ascii=False),flush=True)
            return result
        finally:session.close()
