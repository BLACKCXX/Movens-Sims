"""Isolated three-episode workflow with category repair, version validity and disk acceptance."""
from __future__ import annotations
import asyncio
import copy
import csv
import fcntl
import html
import json
from pathlib import Path
import shutil
import sys
import time
import cv2
import numpy as np
from .acceptance import accept_from_disk,threshold_audit
from .budget import BudgetExceeded
from .control import BatchPaused,check_stop
from .mcp_server import call_tools
from .metrics import gate
from .operations import current_evaluation,coverage_seed
from .prepare import prepare
from .review_state import CATEGORIES,dependencies,valid_check,protected_slots
from .session import ReconstructionSession
from .terra import request_review
from ..head_camera import labelled_comparison
from ..render import overlay,video_writer
from ..util import ROOT,CODE_ROOT,read,write,sha,digest,save_rgb,append,now


def image_cache(session,best):
    with np.load(session.run/'evaluation_cache'/(best['cachekey']+'.npz')) as z:return z['rgb'].copy(),{k:z[k].copy() for k in ('apple','plate','cloth','table')}


def execute(session,calls,reason):
    """Every optimizer/diagnostic operation requested here uses actual stdio MCP."""
    run=session.run;config=session.config;index=len(session.state.get('executed_plans',[]));path=run/'mcp_plans'/f'plan_{index:03}.json'
    calls=[dict(tool=c['tool'],arguments={**c.get('arguments',{}),'_current_candidate_id':True}) for c in calls]
    record=dict(reason=reason,before_candidate_id=session.scene.candidate_id(),calls=calls,started_utc=now());write(path,record)
    session.state.setdefault('operations',[]).append(dict(plan=str(path.relative_to(run)),reason=reason,tools=[c['tool'] for c in calls]));session.save();session.close()
    result=asyncio.run(call_tools(run,calls));record.update(result=result,finished_utc=now());write(path,record)
    session=ReconstructionSession(run,config);session.state.setdefault('executed_plans',[]).append(str(path.relative_to(run)));session.save()
    errors=[r['result'].get('error',{}) for r in result['calls'] if not r['result'].get('ok')]
    if any(e.get('code')=='BatchPaused' for e in errors):raise BatchPaused('stopped','MCP checkpoint requested stop',0)
    if errors:session.state.setdefault('tool_failures',[]).append(dict(plan=str(path.relative_to(run)),errors=errors));session.save()
    return session,result


def action_calls(parsed):
    calls=[]
    for action in parsed['proposed_actions']:
        tool=action['tool'];updates={r['param_id']:r['value'] for r in action['updates']};args={}
        if tool in ('fit_geometry','fit_appearance'):
            ids=set(action['parameter_ids'])
            if tool=='fit_geometry':
                for label in ('apple','plate'):
                    if any(x.startswith(label+'.') for x in ids):ids.update([label+'.x',label+'.y',label+'.scale'])
                args.update(evidence=action['evidence'],confidence=action['confidence'])
            args['parameter_ids']=sorted(ids) or None
        elif tool in ('update_environment','update_shape_or_asset','repair_support'):
            args.update(updates=updates,evidence=action['evidence'],confidence=action['confidence'])
        elif tool=='review_camera_hypothesis':args.update(evidence=action['evidence'],confidence=action['confidence'])
        elif tool=='propose_or_apply_object_update':args.update(updates=updates,apply=True)
        elif tool=='update_appearance':args['updates']=updates
        calls.append(dict(tool=tool,arguments=args))
    return calls


def counts(session):
    result={c:0 for c in CATEGORIES}
    for event in session.budget.summary()['events']:
        if event['kind']=='terra_requests' and event['detail'].get('category') in result:result[event['detail']['category']]+=1
    session.state['category_attempts']=result
    return result


def numeric_category(session,category,metrics):
    result=gate(metrics,session.scene.geometry_metrics(),read(session.run/'metrics_mount_baseline.json'),session.config)
    if category=='observation':return not any(f.startswith('table.') for f in result['failures']) and read(session.run/'mask_review.json')['mask_quality_ok']
    if category=='geometry':return not any(not f.startswith('table.') for f in result['failures'])
    if category=='initial_support':return result['initial_support_passed']
    if category=='rest':
        path=session.run/'evidence'/session.scene.candidate_id()/'rest/physical_diagnostic.json'
        return path.exists() and read(path)['physics_passed']
    return np.isfinite(metrics['appearance_loss'])


def prepare_evidence(session,category):
    session.scene.stage='appearance' if category=='appearance' else 'geometry';session.state['active_category']=category;session.save()
    best=current_evaluation(session,session.scene.stage)
    if category in ('geometry','initial_support','appearance','rest'):
        session,_=execute(session,[dict(tool='render_auxiliary',arguments={})],f'{category}: obtain side/oblique/neutral-light evidence')
    if category in ('initial_support','rest'):
        session,_=execute(session,[dict(tool='diagnose_support',arguments={})],f'{category}: native flex/contact evidence')
    return session,current_evaluation(session,session.scene.stage)


def review_categories(session,enabled):
    config=session.config;run=session.run
    for iteration in range(20):
        check_stop()
        made_request=False;api_wait=False
        for category in CATEGORIES:
            progress=f'{iteration}/{category}'
            if progress in session.state.get('classification_progress',[]):continue
            if iteration and valid_check(session.state,category,session.scene.params,session.scene.locked_camera,session.observation['version']):
                session.state.setdefault('skipped_checks',[]).append(dict(category=category,iteration=iteration,candidate_id=session.scene.candidate_id(),reason='Passed dependency fingerprint unchanged'));session.save();continue
            attempted=counts(session);left=session.budget.remaining('terra_requests')
            if left<=1:continue
            if attempted[category]>=2 and left<=protected_slots(session.state,session.scene.params,session.scene.locked_camera,session.observation['version']):continue
            session,best=prepare_evidence(session,category);rgb,masks=image_cache(session,best)
            cached_check=session.state.get('category_checks',{}).get(category,{})
            reusable=cached_check.get('progress')==progress and cached_check.get('source_candidate_id')==session.scene.candidate_id() and (cached_check.get('status') or {}).get('status')=='validated'
            pending=session.state.get('pending_query')
            if not pending or pending.get('progress')!=progress:
                session.state['evidence_revision']=session.state.get('evidence_revision',0)+1
                session.state['pending_query']=dict(progress=progress,candidate_id=session.scene.candidate_id())
                session.save()
            before_used=session.budget.used('terra_requests')
            print(f'Category {category}, iteration {iteration}, scene {session.scene.candidate_id()}, API used {before_used:g}/20',flush=True)
            status=None;parsed=None
            try:
                if reusable:status,parsed={**cached_check['status'],'cache_reused_after_tool_stop':True},cached_check['review']
                else:status,parsed=request_review(session,best['metrics'],rgb,masks,enabled=enabled)
            except BatchPaused as exc:
                api_wait=True
                session.state.setdefault('request_failures',[]).append(dict(category=category,reason=str(exc),utc=now()))
                # Other categories still get their local evidence. The detached
                # coordinator resumes API transport with the same billed ledger.
                session.save()
            made_request |= session.budget.used('terra_requests')>before_used
            check=dict(category=category,progress=progress,passed=bool(parsed and parsed['verdict']=='pass' and numeric_category(session,category,best['metrics'])),valid=True,source_candidate_id=session.scene.candidate_id(),scene_revision=session.state.get('scene_revision',0),dependency_hash=dependencies(category,session.scene.params,session.scene.locked_camera,session.observation['version']),status=status,review=parsed,checked_utc=now())
            session.state.setdefault('category_checks',{})[category]=check;session.state.setdefault('review_history',[]).append(copy.deepcopy(check));session.state['pending_query']=None;session.save()
            calls=action_calls(parsed) if parsed else []
            if calls:session,_=execute(session,calls,f'{category}: execute actual Terra repair/evidence plan')
            # Deterministic repairs for known implementation gaps remain explicit
            # tools, with real pixel/physics evidence, not invented model verdicts.
            if category=='observation' and session.scene.params['cloth.coverage']==0:
                seed=coverage_seed(session.scene,session.observation);write(run/'coverage_initialization_evidence.json',seed)
                session,_=execute(session,[dict(tool='inspect_environment',arguments={}),dict(tool='update_environment',arguments=dict(updates=seed['updates'],evidence='Real full-frame cloth mask ray-plane bounds versus the original small patch; '+seed['source'],confidence=.95)),dict(tool='fit_geometry',arguments=dict(parameter_ids=['table.depth_scale','table.width_scale','table.front_extension'],evidence='Compare full-frame/top segmentation after ray-plane coverage initialization',confidence=.95))],'Codex: restore existing cloth/table extent capability using recorded projection evidence')
            if category=='appearance' and not session.state.get('appearance_status'):
                session,_=execute(session,[dict(tool='fit_appearance',arguments={})],'Codex: execute deferred color/material/light fitting on frozen current geometry')
            if category=='observation' and attempted[category]>=1 and not numeric_category(session,'observation',current_evaluation(session,'geometry')['metrics']) and not session.state.get('codex_camera_hypothesis_tested'):
                session,_=execute(session,[dict(tool='review_camera_hypothesis',arguments=dict(evidence='Residual top-cloth coverage remains after actual extent fitting; compare bounded camera/FOV hypotheses and both object residuals against the fixed-camera control. This is a reversible sensitivity test, not measured calibration.',confidence=.85))],'Codex: compare projection hypotheses after unsuccessful extent-only correction')
                session.state['codex_camera_hypothesis_tested']=True;session.save()
            if category=='geometry' and not numeric_category(session,'geometry',current_evaluation(session,'geometry')['metrics']) and not session.state.get('codex_plate_tilt_compared'):
                metric=current_evaluation(session,'geometry')['metrics']['objects']['plate']
                if metric['centroid_error_px']<=session.config['acceptance']['center_px_max'] and max(metric['axis_relative_errors'])>session.config['acceptance']['axes_relative_max']:
                    session,_=execute(session,[dict(tool='fit_geometry',arguments=dict(parameter_ids=['plate.x','plate.y','plate.scale','plate.rx','plate.ry'],evidence='After executed joint XY/scale fitting the plate has opposing minor/major axis residuals, while the neutral/side views retain an upward-facing profile. Compare an explicitly uncalibrated <=5-degree supported tilt hypothesis; retain numerical and later native-rest tests, and report single-view ambiguity.',confidence=.8))],'Codex: compare a bounded plate orientation hypothesis instead of repeating scale-only fits')
                    session.state['codex_plate_tilt_compared']=True;session.save()
            if category=='rest' and not numeric_category(session,'rest',current_evaluation(session,'geometry')['metrics']) and not any(c['tool']=='repair_support' for c in calls):
                session,_=execute(session,[dict(tool='repair_support',arguments=dict(updates={},evidence='Fixed one-second terminal diagnostic exceeds the agreed displacement/rotation limits; compare upright training-asset initialization and refit XY/scale',confidence=.9))],'Codex: compare support-preserving upright initialization')
            if parsed or session.budget.used('terra_requests')>before_used:session.state.setdefault('classification_progress',[]).append(progress)
            session.save()
        counts(session);session.save()
        if all(valid_check(session.state,c,session.scene.params,session.scene.locked_camera,session.observation['version']) for c in CATEGORIES):break
        if session.budget.remaining('terra_requests')<=1 or not enabled:break
        if api_wait:
            session.state['classification_complete']=False;session.save()
            raise BatchPaused('waiting_api','Independent local evidence collected; resume unqueried categories with the same cumulative request ledger',30)
        # A resumed scan may skip an already completed first round. It must
        # advance to the unfinished/recheck round instead of freezing early.
    session.state['classification_complete']=True;session.save();return session


def freeze_accept_and_review(session,enabled):
    run=session.run;scene=session.scene
    if not (run/'delivery_binding.json').exists():
        scene.stage='frozen';session.save();events=session.budget.reserve_render('render_export',dict(kind='expected_final_scene'))
        expected_rgb,expected_masks=scene.render();
        for event in events:session.budget.finish(event)
        scene.export(run/'delivery/static');scene.export(run/'delivery/dynamic',physics=True)
        write(run/'parameters_final.json',scene.params);write(run/'camera_actual.json',scene.locked_camera);write(run/'parameter_registry.json',scene.registry())
        save_rgb(run/'expected_final_ego.png',expected_rgb)
        acceptance,rgb,masks=accept_from_disk(run,session.config,run/'final_acceptance',session.budget,expected_rgb,expected_masks)
        save_rgb(run/'final_ego.png',rgb);np.savez_compressed(run/'final_masks.npz',**masks);save_rgb(run/'overlay.png',overlay(session.observation['rgb'],masks,session.observation['target']));write(run/'metrics_after.json',acceptance['metrics'])
        write(run/'reload_check.json',dict(passed=acceptance['reload_passed'],errors=acceptance['reload_errors'],source='Independent portable XML/state reload, not the in-memory backend'))
        for name in ('settle_diagnostic.mp4','physical_diagnostic.json','diagnostic_timeline.json','contact_timeline.json','diagnostic_final_state.npz','rest_terminal.png'):
            shutil.copy2(run/'final_acceptance/rest'/name,run/name)
        for name in ('support_side.png','support_oblique.png','neutral_ego.png'):shutil.copy2(run/'final_acceptance/views'/name,run/name)
        info=video_writer(run/'ego_static.mp4',(rgb for _ in range(150)),30);write(run/'ego_static_definition.json',dict(type='static_3D_reconstruction_simulation',real_capture=False,interaction_replay=False,video=info,scene='delivery/static/portable/scene.xml',state='delivery/static/portable/scene.state.npz'))
        for c,check in session.state.get('category_checks',{}).items():
            check['verified_final_candidate_id']=scene.candidate_id();check['final_dependency_unchanged']=check.get('dependency_hash')==dependencies(c,scene.params,scene.locked_camera,session.observation['version'])
        session.save()
        files={str(p.relative_to(run)):sha(p) for folder in (run/'delivery/static/portable',run/'delivery/dynamic/portable',run/'final_acceptance') for p in folder.rglob('*') if p.is_file()}
        for name in ('parameters_final.json','camera_actual.json','final_ego.png','final_masks.npz','metrics_after.json','reload_check.json','ego_static.mp4','settle_diagnostic.mp4','physical_diagnostic.json','diagnostic_timeline.json','contact_timeline.json','diagnostic_final_state.npz','support_side.png','support_oblique.png','neutral_ego.png'):files[name]=sha(run/name)
        binding=dict(candidate_id=scene.candidate_id(),artifacts=files,scene_parameter_asset_hash=digest(files),reload_passed=acceptance['reload_passed'],independent_disk_acceptance_sha256=sha(run/'final_acceptance/acceptance.json'));write(run/'delivery_binding.json',binding)
    binding=read(run/'delivery_binding.json')
    for name,h in binding['artifacts'].items():
        if not (run/name).is_file() or sha(run/name)!=h:raise ValueError('Frozen artifact changed: '+name)
    scene.stage='final_readonly';session.state['active_category']='final';session.save()
    if not (run/'final_review.json').exists():
        rgb=cv2.cvtColor(cv2.imread(str(run/'final_ego.png')),cv2.COLOR_BGR2RGB)
        with np.load(run/'final_masks.npz') as z:masks={k:z[k].copy() for k in z.files}
        status,parsed=request_review(session,read(run/'metrics_after.json'),rgb,masks,final=True,enabled=enabled,delivery_binding=binding)
        write(run/'final_review.json',dict(status=status,review=parsed,delivery_hash=binding['scene_parameter_asset_hash']))
    final=read(run/'final_review.json');acceptance=read(run/'final_acceptance/acceptance.json')
    if final.get('review') and final['status']['status']=='validated':
        numeric=dict(observation=acceptance['coverage_numeric_passed'] and acceptance['observation_identity_passed'],geometry=acceptance['geometry_numeric_passed'],initial_support=acceptance['initial_support_passed'],appearance=np.isfinite(acceptance['metrics']['appearance_loss']),rest=acceptance['physical_passed'])
        for row in final['review']['category_results']:
            category=row['category'];check=dict(category=category,passed=bool(row['verdict']=='pass' and numeric[category]),valid=True,source_candidate_id=scene.candidate_id(),dependency_hash=dependencies(category,scene.params,scene.locked_camera,session.observation['version']),status=final['status'],review=row,final_delivery_hash=binding['scene_parameter_asset_hash'],checked_utc=now())
            session.state.setdefault('category_checks',{})[category]=check;session.state.setdefault('review_history',[]).append(copy.deepcopy(check))
    session.state['complete']=True;session.save()


def report(session):
    run=session.run;c=session.config;acceptance=read(run/'final_acceptance/acceptance.json');review=read(run/'final_review.json');budget=session.budget.summary();write(run/'budget_ledger.json',budget)
    attempts=[read(p) for p in (run/'terra').glob('*/attempt_*/status.json')];usage={}
    for row in attempts:
        for k,v in (row.get('usage') or {}).items():
            if isinstance(v,(int,float)):usage[k]=usage.get(k,0)+v
    checks={category:bool(valid_check(session.state,category,session.scene.params,session.scene.locked_camera,session.observation['version'])) for category in CATEGORIES}
    verdict=review['review']['verdict'] if review.get('review') else 'unreviewed';ep=read(run/'source_manifest.json')['episode']
    result=dict(episode=ep,workflow=c['workflow'],processing_complete=True,construction_complete=acceptance['engineering_passed'],code_implemented=True,input_complete=True,geometry_numeric_passed=acceptance['geometry_numeric_passed'],geometry_passed=acceptance['geometry_numeric_passed'] and checks['geometry'],coverage_passed=acceptance['coverage_numeric_passed'] and checks['observation'],appearance_status=session.state.get('appearance_status','not_run'),appearance_passed=checks['appearance'],initial_support_passed=acceptance['initial_support_passed'],physical_passed=acceptance['physical_passed'],reload_passed=acceptance['reload_passed'],final_terra_status=review['status']['status'],final_terra_verdict=verdict,category_passed=checks,model=c['terra']['model'],terra_actual_http_requests=sum(x.get('actual_http_attempts',0) for x in attempts),terra_requests_reserved=budget['used'].get('terra_requests',0),api_usage=usage,geometry_fit_rounds=budget['used'].get('geometry_rounds',0),render_optimization_evaluations=sum(budget['used'].get(k,0) for k in ('render_initial','render_repair','render_appearance')),render_total=budget['used'].get('render_total',0),optimization_wall_seconds=budget['used'].get('optimization_seconds',0),geometry=acceptance['geometry'],metrics_after=acceptance['metrics'],metrics_before=read(run/'metrics_mount_baseline.json'),camera=session.scene.locked_camera,delivery_binding=read(run/'delivery_binding.json'),real_dynamics_validated=False,limitations=['Single-view dimensions, optical center and hardware intrinsics remain unmeasured','Visual cloth continuation has no soft support; original native flex patch retained','Inherited convex plate interior and real cloth wrinkles are not fully identified from this view'],completed_utc=now())
    result['visual_qualified']=bool(result['geometry_passed'] and result['coverage_passed'] and result['appearance_passed'] and result['reload_passed'] and verdict=='pass');result['needs_review']=not (result['visual_qualified'] and result['physical_passed'] and result['initial_support_passed']);write(run/'result.json',result)
    save_rgb(run/'comparison.png',labelled_comparison([session.observation['rgb'],cv2.cvtColor(cv2.imread(str(run/'mount_layout_baseline.png')),cv2.COLOR_BGR2RGB),cv2.cvtColor(cv2.imread(str(run/'final_ego.png')),cv2.COLOR_BGR2RGB)],['REAL frame 0','INITIAL reconstruction; source scales 1.0','FINAL exported/reloaded reconstruction']))
    old=ROOT/'runs/full402_h0_v1'/f'episode_{ep:06}'/'result.json';comparison=dict(threshold_audit=threshold_audit(c))
    if old.exists():
        historical=read(old);comparison.update(historical=historical['metrics_after'],current=acceptance['metrics'],old_candidate_under_new_object_thresholds={k:dict(iou=o['iou']>=c['acceptance']['iou_min'],center=o['centroid_error_px']<=c['acceptance']['center_px_max'],edge=o['symmetric_edge_distance_px']<=c['acceptance']['edge_px_max'],axes=max(o['axis_relative_errors'])<=c['acceptance']['axes_relative_max']) for k,o in historical['metrics_after']['objects'].items()},interpretation='Compare unchanged old candidate under new thresholds separately from actual metric/physics changes; terminal-vs-peak semantics are a criterion change, not a physical repair')
        from .attribution import physical_attribution
        comparison['physical']=physical_attribution(old.parent,acceptance['physical'])
    write(run/'improvement_attribution.json',comparison);write(run/'api_usage.json',dict(actual_attempts=result['terra_actual_http_requests'],reserved=result['terra_requests_reserved'],tokens=usage,category_attempts=counts(session)))
    summary={k:result[k] for k in ('episode','construction_complete','geometry_passed','coverage_passed','appearance_passed','initial_support_passed','physical_passed','reload_passed','final_terra_verdict','terra_actual_http_requests','render_total')}
    links=['result.json','final_acceptance/acceptance.json','final_review.json','parameters_final.json','camera_actual.json','asset_audit.json','budget_ledger.json','api_usage.json','improvement_attribution.json','delivery/static/portable/scene.xml','delivery/dynamic/portable/scene.xml','diagnostic_timeline.json','contact_timeline.json']
    (run/'report.md').write_text('# Episode '+str(ep)+' reconstruction acceptance\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\nStatic and dynamic portable scenes are independently loaded from disk. Rest uses exact loaded initial state and fixed 1-second terminal criteria. Peaks/velocities are diagnostic. No policy inference or training.\n\n'+json.dumps(result['geometry'],ensure_ascii=False,indent=2)+'\n')
    (run/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Movens-Sim episode '+str(ep)+'</title><style>body{font:16px system-ui;background:#161b22;color:#eee;margin:24px}a{color:#8cf}img{max-width:100%}video{width:640px;max-width:100%}pre{white-space:pre-wrap}</style><h1>Episode '+str(ep)+' · reconstruction acceptance</h1><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><img src="comparison.png"><p>'+' · '.join('<a href="'+p+'">'+p+'</a>' for p in links)+'</p><video controls src="ego_static.mp4"></video><video controls src="settle_diagnostic.mp4"></video><img src="support_side.png"><img src="support_oblique.png"><img src="neutral_ego.png"><p>REAL reference / SIM reconstruction. Native soft-layer rest diagnostic; no policy inference or training.</p>')
    write(run/'delivery_checks.json',dict(passed=acceptance['engineering_passed'],quality_passed=not result['needs_review'],binding_sha256=sha(run/'delivery_binding.json')))
    return result


def run_pipeline(entry,config,enabled=False,resume=False,dry_run=False,command='run'):
    from .workflow import identity,verify_sources
    ep=entry['episode'];run=ROOT/config['run_root']/f'episode_{ep:06}'
    if config['allowed_episodes']!=[0,1,2] or ep not in config['allowed_episodes']:raise ValueError('v4 permits exactly episodes 0,1,2')
    if dry_run:return dict(episode=ep,run=str(run),workflow=config['workflow'],budget=config['budget'],model=config['terra']['model'])
    if command=='report':return read(run/'result.json')
    run.mkdir(parents=True,exist_ok=True)
    with (run/'.workflow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (run/'resolved_config.json').exists() and read(run/'resolved_config.json')!=config:raise ValueError('Same task cannot change configuration or reset budget')
        write(run/'resolved_config.json',config)
        if not (run/'source_manifest.json').exists():prepare(entry,config,run)
        verify_sources(run);ident=identity(run,config)
        if (run/'run_identity.json').exists():
            old=read(run/'run_identity.json')
            if old!=ident:
                if (run/'session.json').exists() and read(run/'session.json').get('complete'):
                    for name,h in read(run/'delivery_binding.json')['artifacts'].items():
                        if sha(run/name)!=h:raise ValueError('Completed frozen artifact changed: '+name)
                    return read(run/'result.json')
                if any(old.get(k)!=ident.get(k) for k in ('config_sha256','source_manifest_sha256','observation_sha256','mujoco')):raise ValueError('Only code repair can continue the same task; inputs/config stay immutable')
                append(run/'code_continuations.jsonl',dict(previous=old,current=ident,reason='User-authorized implementation repair; same immutable inputs/config and cumulative budget',utc=now()))
        write(run/'run_identity.json',ident)
        code_folder=run/'code_versions'/digest(ident['code'])[:16]
        if not code_folder.exists():
            for name in ident['code']:
                path=code_folder/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(CODE_ROOT/name,path)
            write(code_folder/'resolved_config.json',config)
        append(run/'commands.jsonl',dict(argv=sys.argv,enabled_terra=enabled,utc=now(),identity_sha256=digest(ident)))
        if command=='prepare':return dict(episode=ep,prepared=True)
        session=ReconstructionSession(run,config)
        try:
            session.state['observation_version']=session.observation['version'];session.save()
            active=session.state.get('active_category');check=session.state.get('category_checks',{}).get(active,{})
            if active in CATEGORIES and check.get('source_candidate_id')==session.scene.candidate_id() and (check.get('status') or {}).get('status')=='validated' and 'progress' not in check:
                completed=session.state.get('classification_progress',[])
                check['progress']=next(f'{i}/{active}' for i in range(20) if f'{i}/{active}' not in completed);session.save()
            if session.state.get('complete'):
                for name,h in read(run/'delivery_binding.json')['artifacts'].items():
                    if sha(run/name)!=h:raise ValueError('Frozen artifact mismatch: '+name)
                return read(run/'result.json')
            if not session.state.get('initial_fit_complete'):
                if not (run/'metrics_mount_baseline.json').exists():
                    e=session.evaluator;e.stage='geometry';e.pool='render_initial';e.evaluate(session.scene.params.copy(),'mount_baseline',640,480);best=e.best['geometry_640x480'];rgb,_=image_cache(session,best);save_rgb(run/'mount_layout_baseline.png',rgb);write(run/'metrics_mount_baseline.json',best['metrics']);write(run/'initial_asset_dimensions.json',session.scene.geometry_metrics())
                # Upright training shape first; unconstrained orientation was the
                # documented cause of 17..32mm terminal apple motion in old runs.
                session.fit_geometry(['apple.x','apple.y','apple.scale','plate.x','plate.y','plate.scale'])
                session.state['initial_fit_complete']=True;session.save()
            if session.state.get('repair_revision') and not session.state.get('successor_seed_complete'):
                session.scene.stage='geometry';session.save()
                session,_=execute(session,[dict(tool='review_camera_hypothesis',arguments=dict(evidence='Preserved failed candidate: top coverage remained 0.87584 after repeated extent fitting. Compare bounded camera/FOV controls and both object residuals before another review; calibration remains uncertain.',confidence=.9))],'Successor repair: bounded projection comparison from preserved failure evidence')
                session,_=execute(session,[dict(tool='fit_geometry',arguments=dict(parameter_ids=['plate.x','plate.y','plate.scale','plate.rx','plate.ry'],evidence='Preserved plate residual: centered but minor axis undersized and major axis oversized after repeated XY/scale fits. Compare supported <=5-degree tilt, using neutral/side evidence, then recheck native rest; single-view pose remains hypothetical.',confidence=.85))],'Successor repair: compare supported plate tilt instead of repeated scale-only fitting')
                session.scene.stage='appearance';session.save();session,_=execute(session,[dict(tool='fit_appearance',arguments={})],'Successor repair: fit height-mapped apple color, material and light after geometry changes')
                session.state['successor_seed_complete']=True;session.state['codex_camera_hypothesis_tested']=True;session.state['codex_plate_tilt_compared']=True;session.save()
            if session.state.get('repair_revision') and not session.state.get('projected_hypotheses_executed'):
                session.scene.stage='geometry';session.save()
                session,_=execute(session,[dict(tool='review_camera_hypothesis',arguments=dict(evidence='Earlier camera probes only refined two raw candidates and rejected pose trials before recentering. Compare every bounded camera candidate after measured projection-moment XY/scale reseeding; keep the fixed-camera control.',confidence=.9)),dict(tool='fit_geometry',arguments=dict(parameter_ids=['plate.rx','plate.ry','plate.x','plate.y','plate.scale'],evidence='Earlier orientation fit stopped after unchanged XY coordinates before testing tilt. Compare all bounded tilt candidates with projection-based recenter/scale seeds and the actual shared renderer; verify support and one-second dynamics afterward.',confidence=.9))],'Corrected numerical comparison: jointly reseed position/scale for each bounded projection/orientation hypothesis')
                session.scene.stage='appearance';session.save();session,_=execute(session,[dict(tool='fit_appearance',arguments={})],'Refit height-mapped appearance after the accepted geometry hypothesis')
                session.state['projected_hypotheses_executed']=True;session.state['classification_progress']=[];session.state['classification_complete']=False;session.save()
            if not session.state.get('classification_complete'):session=review_categories(session,enabled)
            freeze_accept_and_review(session,enabled);result=report(session);print(json.dumps({k:result[k] for k in ('episode','construction_complete','geometry_passed','appearance_passed','physical_passed','final_terra_verdict','terra_actual_http_requests')},ensure_ascii=False),flush=True);return result
        finally:session.close()
