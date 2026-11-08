"""One fixed pass over existing v5 tools, with no model or model-review cache."""
from __future__ import annotations
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

from .budget import BudgetExceeded
from .classified import execute
from .control import check_stop
from .metrics import gate
from .modes import NO_TERRA,no_terra
from .session import ReconstructionSession
from .v5 import initialize,initialize_checks,perturbation_checks,export_and_accept
from .v5_metrics import objective
from .v5_tools import evaluate,native_rest,guard_failures
from .workflow import identity,verify_sources
from ..util import ROOT,CODE_ROOT,read,write,sha,digest,append,now

APPEARANCE=(
    ('apple_color','apple_color_v5','apple_color'),
    ('apple_color','apple_color_v5','apple_red'),
    ('apple_color','apple_color_v5','apple_yellow'),
    ('apple_highlight','apple_highlight_v5','apple_highlight'),
    ('plate_shading','plate_shading_v5','plate_material'),
    ('cloth_color','scene_lighting_v5','shared_lighting'),
    ('plate_shading','scene_lighting_v5','plate_and_lighting'),
    ('cloth_color','cloth_coarse_v5','cloth_color'))
GEOMETRY=(('coverage','coverage_v5','table_coverage'),
          ('camera','coverage_v5','camera_coverage'),
          ('geometry','geometry_v5','object_layout'),
          ('initial_support','support_v5','support_initialization'))
POLICY=dict(version='fixed_existing_groups_once_v1',appearance=[x[2] for x in APPEARANCE],
            geometry=[x[2] for x in GEOMETRY],passes=1,cloth_optional=False,
            selection='existing_objective_minimum_improvement_and_v5_guard_failures',
            final_render_reserve=dict(render_total=39,render_diagnostic=31,render_export=8))


def validate_config(c):
    if not no_terra(c) or c['allowed_episodes']!=[0,1,2] or c['budget']['terra_requests']!=0:
        raise ValueError('Only the authorized zero-model three-episode ablation')
    if c['ablation']['policy']!=POLICY or c['terra'].get('enabled') is not False:
        raise ValueError('Fixed policy/model prohibition changed')


def numeric_trigger(s,group):
    row=evaluate(s,tag='ablation_numeric_trigger')
    g=gate(row['metrics'],row['geometry'],read(s.run/'metrics_mount_baseline.json'),s.config)
    quality=read(s.run/'mask_review.json')['mask_quality_ok']
    masks_ok=quality and all(int(row['metrics']['objects'][k]['target_visible_pixels'])>0 for k in ('apple','plate'))
    failures=sorted(g['failures']);eligible=False;reason='No relevant numeric failure'
    if group in ('table_coverage','camera_coverage'):
        eligible=masks_ok and any(x.startswith('table.') for x in failures)
        if group=='camera_coverage':
            table=s.state.get('ablation',{}).get('steps',{}).get('table_coverage',{})
            eligible &= table.get('status')=='done' and table.get('result',{}).get('execution_status') in ('completed','no_improvement')
            reason='Remaining full-scene coverage residual after one table/cloth search; bounded hypothesis, not a calibrated camera claim'
    elif group=='object_layout':
        object_checks={label+'.'+key for label in ('apple','plate') for key in ('iou','center','edge','area','axes')}
        eligible=masks_ok and bool(object_checks.intersection(failures))
    elif group=='support_initialization':
        physical,_=native_rest(s)
        eligible=not row['geometry']['physical_valid'] or not physical['physics_passed']
        if not physical['physics_passed']:failures+=['rest.terminal']
        reason='Measured initial support or fixed-one-second terminal rest failure'
    else:
        reason='No independent real inner-profile/orientation evidence; no shape inference from photometry'
    return dict(kind='numeric',candidate_id=s.scene.candidate_id(),observation_version=s.observation['version'],
                parameter_group=group,failures=failures,mask_quality_ok=bool(masks_ok),eligible=bool(eligible),reason=reason)


def validate_numeric_evidence(s,group,evidence):
    if not no_terra(s.config):raise ValueError('Numeric evidence is restricted to the ablation')
    try:submitted=json.loads(evidence)
    except (ValueError,TypeError):raise ValueError('Explicit numeric evidence JSON required')
    current=numeric_trigger(s,group)
    if submitted!=current or not current['eligible']:
        raise ValueError('Stale, ineligible or unverifiable numeric evidence')
    return current


def verify_common_initial(s):
    """Only A INITIAL files are read here; A final results never enter scheduling."""
    if s.state.get('common_initial_verified'):return
    from .diagnostics import load_state
    import numpy as np
    ep=read(s.run/'source_manifest.json')['episode']
    a=ROOT/s.config['ablation']['reference_root']/f'episode_{ep:06}'
    with s.budget.computation(dict(kind='ablation_initial_identity_audit')):
        am,ad=load_state(a/'initial_native_verification/portable/scene.xml')
        bm,bd=load_state(s.run/'initial_native_verification/portable/scene.xml')
        ac=am.camera('approx_ego_camera').id;bc=bm.camera('approx_ego_camera').id
        checks=dict(parameters=read(a/'initial_parameters.json')==read(s.run/'initial_parameters.json'),
                    reference=sha(a/'reference.png')==sha(s.run/'reference.png'),
                    masks=all(sha(a/name)==sha(s.run/name) for name in read(a/'observation.json')['mask_sha256s']),
                    immutable_assets=read(a/'source_manifest.json')['immutable_assets']==read(s.run/'source_manifest.json')['immutable_assets'],
                    qpos=np.array_equal(ad.qpos,bd.qpos),qvel=np.array_equal(ad.qvel,bd.qvel),time=ad.time==bd.time,
                    camera=np.array_equal(ad.cam_xpos[ac],bd.cam_xpos[bc]) and np.array_equal(ad.cam_xmat[ac],bd.cam_xmat[bc]) and am.cam_fovy[ac]==bm.cam_fovy[bc],
                    seed=read(a/'resolved_config.json')['fit']['seed']==s.config['fit']['seed'])
    write(s.run/'common_initial_verification.json',dict(checks=checks,passed=all(checks.values()),
          source='Same v4 raw templates, absolute parameters and native state, with identical v5 default migration',
          a_files_read=['initial_parameters.json','reference.png','observation.json','masks/','source_manifest.json','resolved_config.json','initial_native_verification/portable/'],
          a_final_parameters_or_decisions_used=False,checked_utc=now()))
    if not all(checks.values()):raise ValueError('A/B initial conditions differ: '+str(checks))
    s.state['common_initial_verified']=True;s.save()


def has_search_budget(s,group):
    reserve=POLICY['final_render_reserve'];pool='render_repair' if group in POLICY['geometry'] else 'render_appearance'
    return (s.budget.remaining('optimization_seconds')>s.config['v5']['final_compute_reserve_seconds'] and
            s.budget.remaining(pool)>=1 and s.budget.remaining('render_total')>reserve['render_total'] and
            s.budget.remaining('render_diagnostic')>=reserve['render_diagnostic']+(31 if pool=='render_repair' else 0) and
            s.budget.remaining('render_export')>=reserve['render_export'])


def selection_rule(s,pending,objective_id):
    before,after=pending['before'],pending['after']
    delta=objective(before['metrics'],objective_id,s.config)-objective(after['metrics'],objective_id,s.config)
    failures=guard_failures(s,after,before)
    keep=delta>s.config['acceptance']['minimum_objective_improvement'] and not failures
    return ('keep' if keep else 'rollback'),dict(source='fixed_numeric_rule',objective_id=objective_id,
              improvement=delta,guard_failures=failures,metrics_before=before['metrics'],metrics_after=after['metrics'],
              appearance_submetric_tradeoffs_permitted=objective_id=='scene_lighting_v5',new_acceptance_thresholds=False)


def search_step(s,issue,objective_id,group):
    steps=s.state['ablation']['steps'];step=steps.get(group)
    if step and step['status'] in ('done','skipped','interrupted'):return s
    if not step:
        if not has_search_budget(s,group):
            steps[group]=dict(status='skipped',reason='Final acceptance reserve or existing pool exhausted');s.save();return s
        evidence='Fixed one-pass existing v5 appearance group; geometry/camera fixed'
        if group in POLICY['geometry']:
            proof=numeric_trigger(s,group)
            if not proof['eligible']:
                steps[group]=dict(status='skipped',reason=proof['reason'],evidence=proof);s.save();return s
            evidence=json.dumps(proof,sort_keys=True)
        step=dict(status='search_started',issue_id=issue,objective_id=objective_id,parameter_group=group,
                  operation_id=f'op_{len(s.state.get("search_history",[])):04d}',candidate_id=s.scene.candidate_id(),
                  render_cap=s.config['v5']['local_search_cap'],evidence=evidence,started_utc=now())
        steps[group]=step;s.save()
        tool='optimize_issue' if group in POLICY['geometry'] else 'fit_appearance'
        s,response=execute(s,[dict(tool=tool,arguments=dict(issue_id=issue,objective_id=objective_id,
             parameter_group=group,render_cap=step['render_cap'],evidence=evidence))],f'Fixed ablation step {group}, once')
        payload=response['calls'][-1]['result'];step=s.state['ablation']['steps'][group]
        step['result']=payload.get('result',dict(execution_status='error',error=payload.get('error')));s.save()
    step=s.state['ablation']['steps'][group]
    # A completed child result or staged candidate may survive a parent stop.
    pending=s.state.get('pending_candidate')
    if 'result' not in step:
        found=next((r for r in s.state.get('search_history',[]) if r['operation_id']==step['operation_id'] and r['parameter_group']==group),None)
        if found:step['result']=found
        elif not pending:
            step.update(status='interrupted',reason='No complete tool result; preserve charged work, do not replay local search');s.save();return s
    if pending and pending['operation_id']==step['operation_id']:
        decision,evidence=selection_rule(s,pending,objective_id)
        step.update(status='selection_started',decision=decision);s.save()
        s,response=execute(s,[dict(tool='select_candidate',arguments=dict(issue_id=issue,decision=decision,
                    evidence=json.dumps(evidence,sort_keys=True)))],f'Fixed numeric choice for {group}')
        step=s.state['ablation']['steps'][group];step['selection']=response['calls'][-1]['result']
    else:
        decisions=[d for d in s.state.get('candidate_decisions',[]) if d['operation_id']==step['operation_id']]
        if decisions:step['selection']=dict(ok=True,result=decisions[-1],resumed_existing_selection=True)
    step.update(status='done',finished_utc=now(),selected_candidate_id=s.scene.candidate_id());s.save()
    return s


def run_fixed_pass(s):
    control=s.state.setdefault('ablation',dict(policy=POLICY,policy_hash=digest(POLICY),steps={},diagnostics_done=[]))
    if control['policy_hash']!=digest(POLICY):raise ValueError('Frozen controller policy changed')
    s.save()
    for issue in ('coverage','plate_structure','initial_support'):
        if issue not in s.state['ablation']['diagnostics_done']:
            s,_=execute(s,[dict(tool='diagnose_issue',arguments=dict(issue_id=issue))],f'No-model initial evidence: {issue}')
            s.state['ablation']['diagnostics_done'].append(issue);s.save()
    for spec in GEOMETRY:
        check_stop();s=search_step(s,*spec)
    # A remaining geometric failure is reported, not used to suppress independent appearance searches.
    for spec in APPEARANCE:
        check_stop();s=search_step(s,*spec)
    s.state['ablation']['cloth_detail']=dict(status='non_blocking_limitation',reason='Optional texture fit skipped by frozen policy; existing capability and 80-render cap retained')
    s.state['ablation']['plate_structure']=dict(status='needs_review',reason='No independent real inner profile annotation; no photometric shape repair')
    s.state['ablation']['finished_utc']=now();s.save();return s


def report(s):
    run=s.run;a=read(run/'final_acceptance/acceptance.json');budget=s.budget.summary()
    if budget['used'].get('terra_requests',0)!=0 or (run/'terra').exists():raise ValueError('Ablation model-use isolation violated')
    calls=[json.loads(x) for x in (run/'mcp_tool_calls.jsonl').read_text().splitlines()]
    optim=[x for x in calls if x['tool'] in ('optimize_issue','fit_appearance')]
    stats=dict(mcp=len(calls),diagnostic=sum(x['tool']=='diagnose_issue' for x in calls),optimization=len(optim),
         new_candidates=sum((x.get('retest') or {}).get('new_candidates',0) for x in optim),
         retained_changes=sum(d['candidate_retained'] for d in s.state.get('candidate_decisions',[])),
         stop_reasons={k:v.get('result',{}).get('stop_reason',v.get('reason','fixed_step_done')) for k,v in s.state['ablation']['steps'].items()})
    result=dict(episode=read(run/'source_manifest.json')['episode'],workflow=NO_TERRA,
        processing_complete=bool(s.state.get('complete') and s.state['ablation'].get('finished_utc') and optim),
        construction_complete=a['engineering_passed'],geometry_numeric_passed=a['geometry_numeric_passed'],geometry_passed=a['geometry_numeric_passed'],
        coverage_passed=a['coverage_numeric_passed'],appearance_passed=None,appearance_status='measured_no_hard_threshold',
        initial_support_passed=a['initial_support_passed'],physical_passed=a['physical_passed'],reload_passed=a['reload_passed'],
        visual_qualified=None,needs_review=True,final_terra_status='not_run_ablation',final_terra_verdict='not_run_ablation',
        terra_actual_http_requests=0,terra_requests_reserved=0,model=None,api_usage={},render_total=budget['used'].get('render_total',0),
        optimization_wall_seconds=budget['used'].get('optimization_seconds',0),projection_evaluations=budget['used'].get('projection_evaluations',0),
        metrics_after=a['metrics'],geometry=a['geometry'],camera=a['camera_verified_from_xml'],execution_stats=stats,
        run_identity=read(run/'run_identity.json'),completed_utc=now(),details_policy=s.config['v5']['policy'],real_dynamics_validated=False)
    started=read(run/'execution_started.json')['utc']
    from datetime import datetime
    result['elapsed_seconds']=(datetime.fromisoformat(result['completed_utc'])-datetime.fromisoformat(started)).total_seconds()
    write(run/'result.json',result);write(run/'budget_ledger.json',budget);write(run/'execution_stats.json',stats)
    write(run/'api_usage.json',dict(actual_attempts=0,reserved=0,model_review_status='not_run_ablation',cached_model_response_used=False,offline_fake_review_used=False,transport_guard='explicit before cache/credentials/client'))
    write(run/'model_review_status.json',dict(status='not_run_ablation',actual_http_requests=0,judgment=None))
    from .ablation_delivery import episode_report
    episode_report(run,s.config)
    return result


def run_pipeline(entry,config,enabled=False,resume=False,dry_run=False,command='run'):
    validate_config(config)
    if enabled:raise ValueError('Model execution explicitly forbidden in ablation')
    if entry['episode'] not in (0,1,2):raise ValueError('Only episodes 0,1,2')
    run=ROOT/config['run_root']/f"episode_{entry['episode']:06}"
    if dry_run:return dict(episode=entry['episode'],workflow=NO_TERRA,api_will_be_called=False,run=str(run),budget=config['budget'])
    run.mkdir(parents=True,exist_ok=True)
    with (run/'.workflow.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (run/'resolved_config.json').exists() and read(run/'resolved_config.json')!=config:raise ValueError('Frozen ablation configuration changed')
        write(run/'resolved_config.json',config)
        if not (run/'execution_started.json').exists():write(run/'execution_started.json',dict(utc=now()))
        if not (run/'source_manifest.json').exists():initialize(entry,config,run)
        verify_sources(run);ident=identity(run,config)
        if (run/'run_identity.json').exists() and read(run/'run_identity.json')!=ident:
            old=read(run/'run_identity.json')
            if read(run/'session.json').get('complete'):raise ValueError('Completed ablation files cannot be relabeled')
            if any(old[k]!=ident[k] for k in ('config_sha256','source_manifest_sha256','observation_sha256','mujoco')):raise ValueError('Frozen inputs/configuration changed')
            append(run/'code_continuations.jsonl',dict(previous=old,current=ident,budget_reset=False,utc=now()))
        write(run/'run_identity.json',ident)
        code=run/'code_versions'/digest(ident['code'])[:16]
        if not code.exists():
            for name in ident['code']:
                dest=code/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(CODE_ROOT/name,dest)
        append(run/'commands.jsonl',dict(argv=sys.argv,utc=now(),identity=digest(ident),actual_terra=False))
        s=ReconstructionSession(run,config)
        try:
            if s.state.get('complete'):
                for name,h in read(run/'delivery_binding.json')['artifacts'].items():
                    if sha(run/name)!=h:raise ValueError('Frozen final artifact changed: '+name)
                return read(run/'result.json')
            initialize_checks(s);verify_common_initial(s);perturbation_checks(s)
            if command=='prepare':return dict(episode=entry['episode'],prepared=True,terra_requests=0)
            if not (run/'delivery_binding.json').exists():s=run_fixed_pass(s)
            export_and_accept(s)
            s.scene.stage='frozen';s.state['complete']=True;s.save()
            return report(s)
        finally:s.close()
