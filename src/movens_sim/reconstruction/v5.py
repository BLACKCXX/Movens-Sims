"""Terra-led issue feedback loop, with mandatory new disk acceptance per episode."""
from __future__ import annotations
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import cv2
import mujoco
import numpy as np
from .session import ReconstructionSession
from .review_state import CATEGORIES,dependencies,valid_check,protected_slots
from .budget import BudgetExceeded
from .control import check_stop,BatchPaused
from .classified import execute
from .metrics import measure,gate
from .terra import request_review
from .v5_tools import ISSUES,evaluate,images,diagnose_issue,native_rest,plate_views,execution_fingerprint
from .v5_material import defaults
from .workflow import identity,verify_sources
from ..head_camera import labelled_comparison
from ..render import video_writer
from ..util import ROOT,CODE_ROOT,append,read,write,sha,digest,save_rgb,now


def initialize(entry,config,run):
    """Reconstruct v4 from raw templates+absolute params, never rescale a final mesh."""
    import xml.etree.ElementTree as ET
    from .diagnostics import load_state
    old=ROOT/config['initialization']['warm_start_root']/f"episode_{entry['episode']:06}"
    binding=read(old/'delivery_binding.json')
    for name,h in binding['artifacts'].items():
        if sha(old/name)!=h:raise ValueError('V4 frozen artifact changed: '+name)
    shutil.copytree(old/'inputs',run/'inputs')
    for name in ('reference.png','reference_masks_review.png','mask_review.json','observation.json','camera.json','camera_prior.json','asset_audit.json'):
        shutil.copy2(old/name,run/name)
    shutil.copytree(old/'masks',run/'masks')
    params=read(old/'parameters_final.json');new=defaults(ET.parse(run/'inputs/template_static/scene.xml').getroot())
    for label in ('apple','plate'):
        for field in ('specular','shininess'):new[label+'.'+field]*=params['material.'+field]
    params.update(new);camera=read(old/'session.json')['camera_override']
    write(run/'initial_parameters.json',params)
    write(run/'session.json',dict(params=params,camera_override=camera,stage='geometry',shape_enabled=True,plate_tilt_enabled=True,rounds=[],operations=[],geometry_review_passed=False,category_checks={},issues={},source_v4_candidate=binding['candidate_id']))
    # Carry exact native state by joint name, including flex DOF velocities.
    m,d=load_state(old/'delivery/dynamic/portable/scene.xml')
    qa=m.jnt_qposadr;da=m.jnt_dofadr;qs=np.diff(np.r_[qa,m.nq]);ds=np.diff(np.r_[da,m.nv])
    state=run/'inputs/physics_state_by_name.npz'
    np.savez_compressed(state,joint_names=np.array([m.joint(i).name for i in range(m.njnt)]),qpos_addresses=qa,qpos_sizes=qs,dof_addresses=da,dof_sizes=ds,qpos=d.qpos,qvel=d.qvel,time=np.array(d.time))
    shutil.copy2(old/'delivery/dynamic/portable/scene.state.npz',run/'inputs/v4_dynamic_saved.npz')
    manifest=read(old/'source_manifest.json');manifest['inputs']={str(p.relative_to(run)):sha(p) for p in (run/'inputs').rglob('*') if p.is_file()}
    manifest['warm_start']=dict(candidate_id=binding['candidate_id'],delivery_binding_sha256=sha(old/'delivery_binding.json'),root=str(old),raw_templates=True,source_scale_applied_once=True,inherited_checks=False,saved_velocity_max=float(np.abs(d.qvel).max()))
    write(run/'source_manifest.json',manifest)
    baseline=run/'baseline_v4';baseline.mkdir()
    for name in ('final_ego.png','final_masks.npz','parameters_final.json','final_review.json'):shutil.copy2(old/name,baseline/name)
    write(baseline/'source_identity.json',dict(kind='baseline_supplemental_measurement',original_root=str(old),files={name:sha(old/name) for name in ('final_ego.png','final_masks.npz','parameters_final.json','final_review.json')},original_acceptance_sha256=sha(old/'reacceptance/final/acceptance.json'),baseline_mutated=False))


def initialize_checks(s):
    if s.state.get('warm_start_verified'):return
    row=evaluate(s,pool='render_initial',tag='new_v5_warm_start_render');rgb,masks=images(s,row)
    original=cv2.cvtColor(cv2.imread(str(s.run/'baseline_v4/final_ego.png')),cv2.COLOR_BGR2RGB)
    with np.load(s.run/'baseline_v4/final_masks.npz') as z:old_masks={k:z[k].copy() for k in z.files}
    error=float(np.abs(rgb.astype(float)-original).mean());mask_error={k:int(np.count_nonzero(masks[k]!=old_masks[k])) for k in ('apple','plate','cloth')}
    report=dict(rgb_mae=error,mask_changed_pixels=mask_error,passed=error<.05 and max(mask_error.values())<=4,source_scale_applied_once=True,candidate_id=s.scene.candidate_id(),checked_utc=now())
    # Verify actual exported native flex state, not just input NPZ provenance.
    from .diagnostics import load_state
    with s.budget.computation(dict(kind='warm_start_native_export_reload'),whole_task_slice=True):
        path=s.scene.export(s.run/'initial_native_verification',physics=True);m,d=load_state(path)
    old=s.run/'inputs/v4_dynamic_saved.npz'
    with np.load(old) as z:
        report['native_qpos_max_error']=float(np.max(np.abs(d.qpos-z['qpos'])))
        report['native_qvel_max_error']=float(np.max(np.abs(d.qvel-z['qvel'])))
        report['native_time_error']=abs(d.time-float(z['time']))
    report['passed'] &= report['native_qpos_max_error']<1e-9 and report['native_qvel_max_error']<1e-12 and report['native_time_error']<1e-12
    previous=s.run/'warm_start_verification.json'
    if previous.exists():shutil.copy2(previous,s.run/('warm_start_history_'+sha(previous)[:16]+'.json'))
    write(previous,report)
    if not report['passed']:raise ValueError('V4 warm-start reproduction failed: '+json.dumps(report))
    save_rgb(s.run/'initial_reconstruction.png',rgb);save_rgb(s.run/'mount_layout_baseline.png',rgb)
    write(s.run/'metrics_mount_baseline.json',row['metrics']);write(s.run/'initial_asset_dimensions.json',row['geometry'])
    baseline=measure(original,old_masks,s.observation,s.config);write(s.run/'baseline_v4/metrics_v5_evaluator.json',baseline)
    s.state['warm_start_verified']=True;s.save()


def perturbation_checks(s):
    """Actual current-scene interventions, restored before any Terra decisions."""
    if s.state.get('perturbations_complete') and (s.run/'parameter_perturbations/color_pipeline_probe.json').exists():return
    base=evaluate(s);rgb,masks=images(s,base);rows=[]
    tests=[('apple.specular',.7),('apple.shininess',.04),('plate.specular',.7),('plate.shininess',.04),('light.key_rx',20.),('light.key_ry',20.),('light.key_strength',.35),('apple.color_strength',1.),('cloth.texture_amplitude',.02)]
    folder=s.run/'parameter_perturbations';folder.mkdir(exist_ok=True)
    from .v5_diagnostics import color_pipeline_probe
    with s.budget.computation(dict(kind='native_color_pipeline_probe')):color_probe=color_pipeline_probe(s.budget)
    write(folder/'color_pipeline_probe.json',color_probe)
    if not color_probe['native_passthrough']:raise ValueError('Native RGB color-card transfer differs from source; investigate before fitting')
    for k,v in tests:
        p={**base['params'],k:v};row=evaluate(s,p,tag='perturbation_'+k);out,_=images(s,row)
        deltas={label:float(np.abs(out.astype(float)-rgb.astype(float))[mask].mean()) if mask.any() else 0. for label,mask in masks.items()}
        save_rgb(folder/(k+'.png'),out);rows.append(dict(parameter=k,value=v,rgb_mae_by_region=deltas,render_changed=bool(np.any(out!=rgb))))
    report=dict(renderer='MuJoCo '+mujoco.__version__+' native OpenGL EGL',rows=rows,all_effective=all(r['render_changed'] for r in rows),source_texture_read='PIL RGB uint8; source PNG has no explicit color profile',render_read='native RGB uint8; cv2 BGR converted exactly once',texture_write='PNG RGB uint8, no extra gamma or per-object exposure',native_coefficients='Native Phong convention; no claimed calibrated linear radiometry',geometry_restored=s.scene.params==base['params'],checked_utc=now())
    write(folder/'report.json',report)
    if not report['all_effective']:raise ValueError('Ineffective new renderer control: '+json.dumps(rows))
    s.state['perturbations_complete']=True;s.save()


def counts(s):
    d={c:0 for c in CATEGORIES}
    for e in s.budget.summary()['events']:
        if e['kind']=='terra_requests' and e['detail'].get('category') in d:d[e['detail']['category']]+=1
    s.state['category_attempts']=d;return d


def apply_review(s,parsed,status):
    category=s.state['active_category'];row=evaluate(s);before=s.scene.candidate_id();pending=s.state.get('pending_candidate')
    counts(s)
    plan_id=status.get('reservation_id',status['folder']);done=s.state.setdefault('applied_plan_actions',{}).setdefault(plan_id,[])
    for update in parsed['issue_updates']:
        key=update['issue_id'];spec=ISSUES[key];existing=s.state.setdefault('issues',{}).setdefault(key,dict(issue_id=f"{s.config['task_group_id']}/{read(s.run/'source_manifest.json')['episode']:06}/{key}",kind=key,category=spec[0],region=spec[1],blocking=spec[2],history=[]))
        existing.update(status=update['status'],evidence=update['evidence'],confidence=update['confidence'],operation_id=update['operation_id'],dependency_hash=dependencies(spec[0],s.scene.params,s.scene.locked_camera,s.observation['version']))
        existing['history'].append(dict(update=update,source_candidate_id=before,request=status['folder'],utc=now()))
    s.save();feedback=[]
    for action_index,action in enumerate(parsed['proposed_actions']):
        if action_index in done:continue
        tool=action['tool'];args=dict(issue_id=action['issue_id'])
        if tool in ('optimize_issue','fit_appearance'):
            # Every new search needs a remaining feedback and final model slot.
            if s.budget.remaining('terra_requests')<2 or s.budget.remaining('terra_requests')<=protected_slots(s.state,s.scene.params,s.scene.locked_camera,s.observation['version']):
                feedback.append(dict(tool=tool,execution_status='budget_exhausted',stop_reason='Model feedback/final slots reserved'));continue
            args.update(objective_id=action['objective_id'],parameter_group=action['parameter_group'],render_cap=action['render_cap'],bounds=action['bounds'],evidence=action['evidence'],confidence=action['confidence'])
        elif tool=='select_candidate':args.update(decision=action['decision'],evidence=action['evidence'])
        elif tool=='diagnose_issue':args.update(diagnostic=action['diagnostic'])
        s,r=execute(s,[dict(tool=tool,arguments=args)],'Terra v5 decision: '+action['evidence'])
        for call in r['calls']:feedback.append(dict(tool=tool,**call['result']))
        s.state.setdefault('applied_plan_actions',{}).setdefault(plan_id,[]).append(action_index);s.save()
    s.state.setdefault('tool_feedback',[]).extend(feedback)
    if pending and any(x.get('tool')=='select_candidate' and x.get('result',{}).get('candidate_retained') for x in feedback):
        for update in parsed['issue_updates']:
            if update['status']=='resolved' and update['operation_id']==pending['operation_id']:
                item=s.state['issues'][update['issue_id']];item.update(status='resolved',dependency_hash=dependencies(item['category'],s.scene.params,s.scene.locked_camera,s.observation['version']))
    # A review pass binds the actually selected version only when no unreviewed
    # modification remains; a pending search is never treated as resolved.
    verdict=parsed['category_results'][0]['verdict'];current=evaluate(s);g=gate(current['metrics'],current['geometry'],read(s.run/'metrics_mount_baseline.json'),s.config)
    numeric={'observation':not any(x.startswith('table.') for x in g['failures']), 'geometry':not any(not x.startswith('table.') for x in g['failures']), 'initial_support':g['initial_support_passed'],'appearance':True,'rest':True}
    checked=dict(passed=verdict in ('pass','pass_with_limitations') and numeric[category] and not s.state.get('pending_candidate'),valid=True,source_candidate_id=s.scene.candidate_id(),dependency_hash=dependencies(category,s.scene.params,s.scene.locked_camera,s.observation['version']),review=parsed,status=status,utc=now())
    s.state.setdefault('category_checks',{})[category]=checked;s.state.setdefault('review_history',[]).append(copy.deepcopy(checked));s.state['pending_model_plan']=None;s.save();return s


def collect_category_evidence(s,category):
    s.state['active_category']=category;s.save()
    mapping={'observation':['coverage'],'geometry':['plate_structure'],'initial_support':['initial_support'],'appearance':['plate_shading'],'rest':['rest']}
    for issue in mapping[category]:
        s,_=execute(s,[dict(tool='diagnose_issue',arguments=dict(issue_id=issue))],f'Automatic evidence only for required new v5 {category} first/recheck')
    return s


def review_needed(s,category,attempted):
    if valid_check(s.state,category,s.scene.params,s.scene.locked_camera,s.observation['version']):return False
    # Unchanged images do not mean unchanged optimization evidence: a failed
    # search or rollback has new feedback that Terra must receive. In particular
    # an open, expressible appearance problem must not inherit the two-review
    # deferral intended for uncertain real geometry. Let Terra change its search
    # hypothesis, or explicitly classify the remaining capability/evidence gap.
    if category=='appearance':
        required=[v for v in s.state.get('issues',{}).values() if v.get('category')==category and v.get('blocking')]
        if any(v.get('status') in ('open','needs_evidence') for v in required):return True
    fp=dependencies(category,s.scene.params,s.scene.locked_camera,s.observation['version'])
    previous=[r for r in s.state.get('review_history',[]) if (r.get('review',{}).get('category_results') or [{}])[0].get('category')==category]
    if attempted[category]>=2 and len(previous)>=2 and all(r.get('dependency_hash')==fp and r['review']['category_results'][0]['verdict']=='needs_review' for r in previous[-2:]):
        s.state.setdefault('deferred_checks',{})[category]=dict(dependency_hash=fp,reason='Two reviews of unchanged evidence remain uncertain; defer until new dependent evidence. This is not a pass and final complete disk review is mandatory.')
        return False
    return True


def next_category(s,attempted):
    first=[c for c in CATEGORIES if attempted[c]==0]
    if first:return first[0]
    unfinished=[c for c in CATEGORIES if review_needed(s,c,attempted)]
    # Pending comparisons are included in EVERY request, so they need not
    # monopolize the primary category. Continue fair rotation beyond round2.
    if unfinished:return min(unfinished,key=lambda c:(attempted[c],CATEGORIES.index(c)))
    pending=s.state.get('pending_candidate')
    return ISSUES[pending['issue_id']][0] if pending else None


def run_loop(s,enabled):
    for iteration in range(30):
        check_stop();attempted=counts(s)
        first=[c for c in CATEGORIES if attempted[c]==0]
        pending=s.state.get('pending_candidate')
        if first or pending:category=next_category(s,attempted)
        else:
            queue=[c for c in CATEGORIES if review_needed(s,c,attempted)]
            if not queue:break
            # Entitlements first, then appearance feedback rather than blind round robin.
            low=[c for c in queue if attempted[c]<2]
            category=next_category(s,attempted)
            if attempted[category]>=2 and s.budget.remaining('terra_requests')<=protected_slots(s.state,s.scene.params,s.scene.locked_camera,s.observation['version']):break
            terminal={'resolved','capability_gap','non_blocking_limitation'}
            required=[v for v in s.state.get('issues',{}).values() if v.get('blocking')]
            if not low and not pending and required and all(v.get('status') in terminal for v in required):
                s.state['local_stopping_reason']='All reported required issues resolved or explicitly capability-limited; final review remains mandatory';s.save();break
        if s.budget.remaining('terra_requests')<=1:break
        s=collect_category_evidence(s,category)
        # Only a real new request increments evidence revision. Cached responses
        # can resume execution after a stop without being relabeled as new calls.
        s.state['evidence_revision']=s.state.get('evidence_revision',0)+1;s.save();row=evaluate(s);rgb,masks=images(s,row)
        print(f"V5 Terra {category} selected={s.scene.candidate_id()} pending={bool(pending)} requests={s.budget.used('terra_requests'):g}/20",flush=True)
        status,parsed=request_review(s,row['metrics'],rgb,masks,enabled=enabled)
        if not parsed:
            if not enabled:break
            continue
        write(s.run/'last_decision.json',dict(status=status,review=parsed));s.state['pending_model_plan']=dict(status=status,review=parsed,category=category);s.save();s=apply_review(s,parsed,status)
        counts(s);s.save()
    # If no model slot remains to inspect a staged search, keep the last model-
    # selected checkpoint. Archive, do not silently accept an unreviewed trial.
    if s.state.get('pending_candidate'):
        issue=s.state['pending_candidate']['issue_id'];s,_=execute(s,[dict(tool='select_candidate',arguments=dict(issue_id=issue,decision='rollback',evidence='Local boundary: no remaining candidate-feedback opportunity; preserve last explicitly reviewed selection'))],'Finalization of unreviewed pending trial')
    return s


def export_and_accept(s):
    """Shared disk export and fresh independent acceptance; no model decision."""
    run=s.run
    if not (run/'delivery_binding.json').exists():
        row=evaluate(s,pool='render_export',tag='final_selected');rgb,masks=images(s,row)
        with s.budget.computation(dict(kind='final_static_dynamic_export'),whole_task_slice=True):
            s.scene.export(run/'delivery/static');s.scene.export(run/'delivery/dynamic',physics=True)
        actual_camera=copy.deepcopy(s.scene.locked_camera);rotation=np.asarray(actual_camera['rotation_camera_to_world'])
        actual_camera.update(down_angle_deg=float(np.degrees(np.arcsin(np.clip(rotation[2,2],-1,1)))),fy_px=float(240/np.tan(np.radians(actual_camera['fovy_deg']/2))),height_above_nominal_cloth_m=float(actual_camera['xyz_m'][2]-s.scene.support.height_range[1]),resolution=[640,480],calibrated=False,derived_metadata_source='Actual exported camera rotation/FOV and support height, not copied prior scalar angles')
        write(run/'parameters_final.json',s.scene.params);write(run/'camera_actual.json',actual_camera);write(run/'parameter_registry.json',s.scene.registry())
        save_rgb(run/'expected_final_ego.png',rgb);save_rgb(run/'final_ego.png',rgb);np.savez_compressed(run/'final_masks.npz',**masks)
        s.scene.stage='frozen';s.save()
        # Separate process: only portable XML/assets/saved state may define the
        # tested model. No in-memory SceneBackend or old physical report reused.
        command=[str(ROOT/'.venv/bin/python'),'-B',str(CODE_ROOT/'scripts/accept_reconstruction_v5.py'),'--run',str(run),'--output','final_acceptance']
        receipt=dict(command=command,started_utc=now(),parent_pid=os.getpid(),implementation=execution_fingerprint());write(run/'final_acceptance_process.json',receipt)
        with (run/'final_acceptance_process.log').open('a') as log:
            p=subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(CODE_ROOT/'src')},stdout=log,stderr=subprocess.STDOUT)
        receipt.update(returncode=p.returncode,ended_utc=now());write(run/'final_acceptance_process.json',receipt)
        if p.returncode:raise RuntimeError('Independent acceptance failed; read final_acceptance_process.log')
        acceptance=read(run/'final_acceptance/acceptance.json');shutil.copy2(run/'final_acceptance/reloaded_ego.png',run/'final_ego.png');shutil.copy2(run/'final_acceptance/reloaded_masks.npz',run/'final_masks.npz')
        write(run/'metrics_after.json',acceptance['metrics']);write(run/'reload_check.json',dict(passed=acceptance['reload_passed'],errors=acceptance['reload_errors']))
        for name in ('settle_diagnostic.mp4','physical_diagnostic.json','diagnostic_timeline.json','contact_timeline.json','diagnostic_final_state.npz','rest_terminal.png'):shutil.copy2(run/'final_acceptance/rest'/name,run/name)
        for name in ('support_side.png','support_oblique.png','neutral_ego.png'):shutil.copy2(run/'final_acceptance/views'/name,run/name)
        rgb=cv2.cvtColor(cv2.imread(str(run/'final_ego.png')),cv2.COLOR_BGR2RGB);video_writer(run/'ego_static.mp4',(rgb for _ in range(150)),30)
        files={str(p.relative_to(run)):sha(p) for directory in (run/'delivery',run/'final_acceptance') for p in directory.rglob('*') if p.is_file()}
        for name in ('parameters_final.json','camera_actual.json','final_ego.png','final_masks.npz','metrics_after.json','ego_static.mp4','settle_diagnostic.mp4','final_acceptance_process.json'):files[name]=sha(run/name)
        binding=dict(candidate_id=s.scene.candidate_id(),artifacts=files,scene_parameter_asset_hash=digest(files),independent_disk_acceptance_sha256=sha(run/'final_acceptance/acceptance.json'));write(run/'delivery_binding.json',binding)
    binding=read(run/'delivery_binding.json')
    for name,h in binding['artifacts'].items():
        if sha(run/name)!=h:raise ValueError('Frozen final artifact changed: '+name)
    return binding


def finalize(s,enabled):
    run=s.run;binding=export_and_accept(s)
    s.scene.stage='final_readonly';s.state['active_category']='final';s.save()
    if not (run/'final_review.json').exists() or not read(run/'final_review.json').get('review'):
        rgb=cv2.cvtColor(cv2.imread(str(run/'final_ego.png')),cv2.COLOR_BGR2RGB)
        with np.load(run/'final_masks.npz') as z:masks={k:z[k] for k in z.files}
        status,parsed=request_review(s,read(run/'metrics_after.json'),rgb,masks,True,enabled,binding)
        write(run/'final_review.json',dict(status=status,review=parsed,delivery_hash=binding['scene_parameter_asset_hash']))
    s.state['complete']=bool(read(run/'final_review.json').get('review'));s.save()
    return report(s)


def report(s):
    run=s.run;c=s.config;a=read(run/'final_acceptance/acceptance.json');review=read(run/'final_review.json');r=review.get('review');checks={x['category']:x['verdict'] in ('pass','pass_with_limitations') for x in r['category_results']} if r else {k:False for k in CATEGORIES}
    attempts=[read(p) for p in (run/'terra').glob('*/attempt_*/status.json')];budget=s.budget.summary();usage={}
    for row in attempts:
        for k,v in (row.get('usage') or {}).items():
            if isinstance(v,(int,float)):usage[k]=usage.get(k,0)+v
    actual=sum(x.get('actual_http_attempts',0) for x in attempts);ep=read(run/'source_manifest.json')['episode']
    calls=[json.loads(x) for x in (run/'mcp_tool_calls.jsonl').read_text().splitlines()] if (run/'mcp_tool_calls.jsonl').exists() else []
    optim=[x for x in calls if x['tool'] in ('optimize_issue','fit_appearance')];stats=dict(terra_http=actual,mcp=len(calls),diagnostic=sum(x['tool']=='diagnose_issue' for x in calls),optimization=len(optim),ineffective=sum((x.get('retest') or {}).get('new_candidates',0)==0 for x in optim),candidate_evaluations=sum((x.get('retest') or {}).get('candidate_evaluations',0) for x in optim),new_candidates=sum((x.get('retest') or {}).get('new_candidates',0) for x in optim),retained_changes=sum(x.get('candidate_retained',False) for x in s.state.get('candidate_decisions',[])),resolved_issues=sum(v.get('status')=='resolved' for v in s.state.get('issues',{}).values()))
    executed=bool(stats['new_candidates']>0 and len(attempts)>=6)
    result=dict(episode=ep,workflow=c['workflow'],processing_complete=s.state['complete'] and executed,construction_complete=a['engineering_passed'],geometry_numeric_passed=a['geometry_numeric_passed'],geometry_passed=a['geometry_numeric_passed'] and checks['geometry'],coverage_passed=a['coverage_numeric_passed'] and checks['observation'],appearance_passed=checks['appearance'],appearance_status='executed' if optim else 'not_run',initial_support_passed=a['initial_support_passed'],physical_passed=a['physical_passed'],reload_passed=a['reload_passed'],final_terra_verdict=r['verdict'] if r else 'unreviewed',final_terra_status=review['status']['status'],category_passed=checks,terra_actual_http_requests=actual,terra_requests_reserved=budget['used'].get('terra_requests',0),render_total=budget['used'].get('render_total',0),optimization_wall_seconds=budget['used'].get('optimization_seconds',0),model=c['terra']['model'],api_usage=usage,metrics_after=a['metrics'],geometry=a['geometry'],camera=a['camera_verified_from_xml'],execution_stats=stats,run_identity=read(run/'run_identity.json'),details_policy=c['v5']['policy'],real_dynamics_validated=False,completed_utc=now())
    result['visual_qualified']=bool(result['geometry_passed'] and result['coverage_passed'] and result['appearance_passed'] and result['reload_passed'] and r and r['verdict'] in ('pass','pass_with_limitations'));result['needs_review']=not (result['visual_qualified'] and result['physical_passed']);write(run/'result.json',result)
    write(run/'budget_ledger.json',budget);write(run/'api_usage.json',dict(actual_attempts=actual,tokens=usage,category_attempts=counts(s),historical_v4_requests=20));write(run/'execution_stats.json',stats)
    real=s.observation['rgb'];old=cv2.cvtColor(cv2.imread(str(run/'baseline_v4/final_ego.png')),cv2.COLOR_BGR2RGB);new=cv2.cvtColor(cv2.imread(str(run/'final_ego.png')),cv2.COLOR_BGR2RGB)
    save_rgb(run/'comparison.png',labelled_comparison([real,old,new],['REAL frame 0','V4 frozen final','V5 NEW optimized / independently reloaded']))
    for label in ('apple','plate'):
        yy,xx=np.where(s.observation['target'][label]);x0,x1=max(0,int(xx.min())-12),min(640,int(xx.max())+13);y0,y1=max(0,int(yy.min())-12),min(480,int(yy.max())+13)
        save_rgb(run/(label+'_comparison.png'),labelled_comparison([v[y0:y1,x0:x1] for v in (real,old,new)],['REAL','V4','V5']))
    from .v5_delivery import episode_report
    episode_report(run,c)
    return result


def run_pipeline(entry,config,enabled=False,resume=False,dry_run=False,command='run'):
    if config['allowed_episodes']!=[0,1,2] or entry['episode'] not in (0,1,2):raise ValueError('Only episodes 0,1,2 are authorized')
    run=ROOT/config['run_root']/f"episode_{entry['episode']:06}"
    if dry_run:return dict(episode=entry['episode'],workflow=config['workflow'],run=str(run),budget=config['budget'])
    run.mkdir(parents=True,exist_ok=True)
    with (run/'.workflow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (run/'resolved_config.json').exists() and read(run/'resolved_config.json')!=config:raise ValueError('Frozen v5 config cannot change on resume')
        write(run/'resolved_config.json',config)
        if not (run/'source_manifest.json').exists():initialize(entry,config,run)
        verify_sources(run);ident=identity(run,config)
        if (run/'run_identity.json').exists() and read(run/'run_identity.json')!=ident:
            previous=read(run/'run_identity.json')
            if (run/'session.json').exists() and read(run/'session.json').get('complete'):raise ValueError('Completed v5 revision cannot silently change code')
            if any(previous[k]!=ident[k] for k in ('config_sha256','source_manifest_sha256','observation_sha256','mujoco')):raise ValueError('Resume changed frozen inputs/config')
            append(run/'code_continuations.jsonl',dict(previous=previous,current=ident,utc=now(),budget_reset=False))
            state=read(run/'session.json');changed={name for name in set(previous['code'])|set(ident['code']) if previous['code'].get(name)!=ident['code'].get(name)}
            image_or_physics={'backend.py','render.py','v5_material.py','v5_metrics.py','metrics.py','diagnostics.py','acceptance.py','full_prepare.py'}
            affects_evidence=any(Path(name).name in image_or_physics for name in changed)
            if affects_evidence:
                state['category_checks']={}
                for issue in state.get('issues',{}).values():
                    if issue.get('status')=='resolved':issue.update(status='open',invalidation_reason='evidence_implementation_changed')
            append(run/'code_invalidation_audit.jsonl',dict(changed_files=sorted(changed),evidence_invalidated=affects_evidence,reason='Controller/repair-prompt-only changes preserve unchanged measurement dependencies; final complete disk checks still mandatory',utc=now()))
            if affects_evidence and state.get('pending_candidate'):
                write(run/'code_compatibility'/('pending_'+digest(previous['code'])[:12]+'.json'),state['pending_candidate']);state['pending_candidate']=None
            write(run/'session.json',state)
        write(run/'run_identity.json',ident);code=run/'code_versions'/digest(ident['code'])[:16]
        if not code.exists():
            for name in ident['code']:
                dest=code/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(CODE_ROOT/name,dest)
        append(run/'commands.jsonl',dict(argv=sys.argv,utc=now(),identity=digest(ident),actual_terra=enabled))
        s=ReconstructionSession(run,config)
        try:
            s.state['observation_version']=s.observation['version'];s.save()
            if s.state.get('complete'):
                for name,h in read(run/'delivery_binding.json')['artifacts'].items():
                    if sha(run/name)!=h:raise ValueError('Frozen artifact changed: '+name)
                return read(run/'result.json')
            initialize_checks(s);perturbation_checks(s)
            if command=='prepare':return dict(episode=entry['episode'],prepared=True,terra_requests=s.budget.used('terra_requests'))
            if s.state.get('pending_model_plan'):
                plan=s.state['pending_model_plan'];s.state['active_category']=plan['category'];s=apply_review(s,plan['review'],plan['status'])
            if not (run/'delivery_binding.json').exists():s=run_loop(s,enabled)
            result=finalize(s,enabled);print(json.dumps({k:result[k] for k in ('episode','processing_complete','coverage_passed','appearance_passed','physical_passed','terra_actual_http_requests')},ensure_ascii=False),flush=True);return result
        finally:s.close()
