"""Metered adapter for the existing initial geometry kernel and MCP selection."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
from .modes import initial_xml
from .budget import BudgetExceeded
from .search_replay import SearchReplay
from .optimize import fit_geometry
from .v5_tools import evaluate,images,guard_failures
from ..util import read,write,append,save_rgb,digest,now


def legacy_objective(row):
    return row['metrics']['image_loss']+.002*sum((np.log(row['params'][label+'.scale'])/np.log(2))**2 for label in ('apple','plate'))


def fit_initial_geometry(s,candidate_id):
    if not initial_xml(s.config):raise ValueError('Initial geometry adapter is only enabled for initial_xml')
    s.check_id(candidate_id)
    if s.scene.stage in ('frozen','final_readonly'):raise ValueError('Frozen scene')
    if s.state.get('pending_candidate'):raise ValueError('Select the pending candidate first')
    from .ablation import numeric_trigger
    proof=numeric_trigger(s,'object_layout')
    if not proof['eligible']:return dict(execution_status='capability_gap',stop_reason=proof['reason'],evidence=proof,new_candidates=0,new_render_evaluations=0,cache_hits=0,candidate_retained=False,parameters_changed=False)
    spec=s.config['initial_geometry'];ids=spec['parameter_ids'];s.scene.stage='geometry';s.save()
    control=s.state.setdefault('cold_start',{});operation=control.get('initial_operation')
    if not operation:
        render_start=s.budget.used('render_repair');base=evaluate(s,pool='render_repair',tag='initial_geometry_control');e=s.evaluator
        operation=dict(operation_id=f'op_{len(s.state.get("search_history",[])):04d}',base=base,render_start=render_start,cap=min(spec['render_cap'],int(s.budget.remaining('render_repair'))+int(s.budget.used('render_repair')-render_start)),round_event=s.budget.reserve('geometry_rounds',detail=dict(round_index=0,stage='initial_xml_geometry')),initial_evaluator=dict(best=copy.deepcopy(e.best),attempts=e.attempts,improvements=e.improvements),started_utc=now())
        control['initial_operation']=operation;s.save()
    base=operation['base'];e=s.evaluator;origin=operation['initial_evaluator']
    e.best=copy.deepcopy(origin['best']);e.attempts=origin['attempts'];e.improvements=origin['improvements']
    s.scene.locked_camera=base['camera'];s.scene.apply(base['params'],internal=True)
    progress=SearchReplay(s.run/'initial_geometry_progress.json',digest(dict(candidate=base['candidate_id'],observation=s.observation['version'],spec=spec)),dict(repair_render_start=operation['render_start'],render_cap=max(0,operation['cap']-1)))
    e.progress=progress;s.budget.projection_progress=progress
    before_count=sum(1 for _ in e.history.open()) if e.history.exists() else 0
    try:
        with s.budget.computation(dict(kind='initial_xml_geometry',operation_id=operation['operation_id']),reserve_seconds=s.budget.remaining('optimization_seconds')-1200):
            result=fit_geometry(e,ids,round_index=0,resume_event=operation['round_event'],render_pool='render_repair',render_cap=operation['cap'])
            best=e.best.get('geometry_640x480');after=evaluate(s,best['params'] if best else base['params'],pool='render_repair',tag='initial_geometry_proposed')
    finally:
        e.progress=None;s.budget.projection_progress=None;s.scene.locked_camera=base['camera'];s.scene.apply(base['params'],internal=True)
    progress.flush(force=True)
    improve=legacy_objective(base)-legacy_objective(after);guards=guard_failures(s,after,base)
    eligible=after['candidate_id']!=base['candidate_id'] and improve>s.config['acceptance']['minimum_objective_improvement'] and not guards
    records=[json.loads(line) for line in e.history.read_text().splitlines()] if e.history.exists() else []
    total_candidates=len({digest(x['params']) for x in records if x['metrics'] is not None})
    actual_cache_hits=sum(x['cache_hit'] for x in records)
    report=dict(execution_status='completed' if eligible else 'no_improvement',operation_id=operation['operation_id'],issue_id='geometry',parameter_group='initial_geometry',objective_id='geometry_v5',selection_objective='Existing image_loss + .002 sum((log(scale)/log(2))**2)',before_objective=legacy_objective(base),after_objective=legacy_objective(after),metrics_before=base['metrics'],metrics_after=after['metrics'],guard_failures=guards,new_candidates=total_candidates,candidate_evaluations=len(records),new_render_evaluations=s.budget.used('render_repair')-operation['render_start'],cache_hits=actual_cache_hits,replayed_objective_calls=progress.replayed,parameters_changed=eligible,candidate_retained=False,pending_review=eligible,stop_reason=result.get('stop_reason','Existing initial coarse/three local/full-resolution search complete'),kernel_result=result,remaining_budget={k:s.budget.remaining(k) for k in s.config['budget']})
    folder=s.run/'operations'/operation['operation_id'];folder.mkdir(parents=True,exist_ok=True)
    for name,row in [('before',base),('after',after if eligible else base)]:
        rgb,_=images(s,row);save_rgb(folder/(name+'.png'),rgb);write(folder/(name+'.json'),row)
    write(folder/'result.json',report)
    if eligible:s.state['pending_candidate']=dict(operation_id=operation['operation_id'],issue_id='geometry',before=base,after=after,stage='geometry')
    s.state.setdefault('search_history',[]).append(report);control['initial_operation_result']=report;s.save()
    return report


def run_initial_geometry(s):
    from .ablation import numeric_trigger,has_search_budget
    from .classified import execute
    control=s.state.setdefault('cold_start',{});step=control.get('initial_geometry')
    if step and step['status'] in ('done','skipped'):return s
    if not step:
        proof=numeric_trigger(s,'object_layout')
        if not proof['eligible'] or not has_search_budget(s,'object_layout'):
            control['initial_geometry']=dict(status='skipped',reason=proof['reason'] if not proof['eligible'] else 'Final reserve/pool exhausted',evidence=proof);s.save();return s
        control['initial_geometry']=dict(status='started',evidence=proof,started_utc=now());s.save()
    if not s.state['cold_start'].get('initial_operation_result'):
        s,response=execute(s,[dict(tool='fit_geometry',arguments={})],'Original XML initial joint XY/scale coarse fit, same kernel and original caps')
        payload=response['calls'][-1]['result']
        if not payload.get('ok'):raise RuntimeError('Initial geometry MCP failed: '+str(payload.get('error')))
    pending=s.state.get('pending_candidate')
    if pending and pending['operation_id']==s.state['cold_start']['initial_operation']['operation_id']:
        result=s.state['cold_start']['initial_operation_result'];decision='keep' if result['parameters_changed'] and not result['guard_failures'] else 'rollback'
        s,response=execute(s,[dict(tool='select_candidate',arguments=dict(issue_id='geometry',decision=decision,evidence=json.dumps(dict(source='existing_initial_geometry_objective_and_hard_guard',improvement=result['before_objective']-result['after_objective'],guard_failures=result['guard_failures']))))],'Numeric selection of original initial-fit staged candidate')
        if not response['calls'][-1]['result'].get('ok'):raise RuntimeError('Initial candidate selection failed')
    s.state['cold_start']['initial_geometry'].update(status='done',result=s.state['cold_start'].get('initial_operation_result'),finished_utc=now(),selected_candidate_id=s.scene.candidate_id());s.save();return s
