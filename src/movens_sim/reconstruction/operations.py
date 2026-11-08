"""Executable v4 capabilities shared by Python and the local MCP server."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .backend import SceneError
from .metrics import gate,nondegrading
from .environment import coverage_seed
from .diagnostics import auxiliary,rest_from_disk
from ..scene import quat_from_matrix
from ..util import write,read,digest

TOOLS={
 'inspect_environment':dict(stages=['geometry','appearance','frozen','final_readonly'],parameters='none',mutates=False),
 'update_environment':dict(stages=['geometry'],parameters='environment',mutates=True),
 'render_auxiliary':dict(stages=['geometry','appearance','frozen','final_readonly'],parameters='none',mutates=False),
 'diagnose_support':dict(stages=['geometry','appearance','frozen','final_readonly'],parameters='none',mutates=False),
 'repair_support':dict(stages=['geometry'],parameters='objects',mutates=True),
 'fit_geometry':dict(stages=['geometry'],parameters='geometry',mutates=True),
 'propose_or_apply_object_update':dict(stages=['geometry'],parameters='objects',mutates=True),
 'update_shape_or_asset':dict(stages=['geometry'],parameters='shape',mutates=True),
 'update_appearance':dict(stages=['appearance'],parameters='appearance',mutates=True),
 'fit_appearance':dict(stages=['appearance'],parameters='appearance',mutates=True),
 'review_camera_hypothesis':dict(stages=['geometry'],parameters='none',mutates=True),
}


def allowed_parameters(session,kind):
    entries=session.scene.registry()
    if kind=='none':return []
    if kind=='appearance':return [p['id'] for p in entries if 'appearance' in p['stages']]
    if kind=='environment':return [p['id'] for p in entries if p['id'].startswith('table.') or p['id']=='cloth.coverage']
    if kind=='shape':return ['plate.depth_ratio','plate.rx','plate.ry']
    return [p['id'] for p in entries if 'geometry' in p['stages'] and (kind!='objects' or p['id'].startswith(('apple.','plate.')))]


def capabilities(session):
    return dict(version='classified_v4',tools={k:{**v,'parameter_ids':allowed_parameters(session,v['parameters'])} for k,v in TOOLS.items() if session.scene.stage in v['stages'] and not (session.scene.stage in ('frozen','final_readonly') and v['mutates'])},absolute_source_scale=True,visual_cloth_support=False,physical_support='Native original flex patch only',camera_bounds=session.config['camera']['conditional_revision'],appearance='Phong plus supported texture tint; geometry unchanged',contact_repair='Evidence-backed initial pose/support; native mass/friction/flex coefficients preserved',single_view_calibrated=False)


def current_evaluation(session,stage=None,pool=None):
    e=session.evaluator;e.stage=stage or session.scene.stage;e.pool=pool or ('render_appearance' if e.stage=='appearance' else 'render_repair')
    key=e.best_key(640,480);best=e.best.get(key)
    if best and best['candidate_id']==session.scene.candidate_id():return best
    # Candidate identity includes camera, environment, shape and appearance.
    # A best snapshot from a different phase must never undo later tool edits.
    if e.stage=='appearance':
        geometry={k:v for k,v in session.scene.params.items() if k in allowed_parameters(session,'geometry')}
        frozen_path=session.run/'geometry_frozen.json';old=read(frozen_path) if frozen_path.exists() else {}
        if old.get('geometry_params')!=geometry:
            write(frozen_path,dict(geometry_params=geometry,candidate_id=session.scene.candidate_id(),purpose='New independent appearance entry after a geometry revision'))
    e.best.pop(key,None);e.evaluate(session.scene.params.copy(),'current_scene_baseline',640,480)
    session.save();return e.best[key]


def inspect_environment(session,candidate_id):
    session.check_id(candidate_id)
    metrics=read(session.run/'metrics_after.json') if session.scene.stage in ('frozen','final_readonly') else current_evaluation(session,'geometry')['metrics']
    return dict(candidate_id=session.scene.candidate_id(),coverage=metrics['table'],coverage_seed=coverage_seed(session.scene,session.observation),physical_support_bounds_m=session.scene.support.bounds.tolist(),table_original_bounds_m=session.scene.table_original_bounds.tolist(),display_cloth_support=False)


def update_environment(session,updates,evidence,confidence,candidate_id):
    session.check_id(candidate_id)
    if session.scene.stage!='geometry':raise SceneError('wrong_stage','environment')
    if not evidence or not np.isfinite(confidence) or not .7<=confidence<=1:raise SceneError('evidence_required','Finite confidence >=0.7 and coverage evidence required')
    allowed=allowed_parameters(session,'environment')
    if set(updates)-set(allowed):raise SceneError('invalid_environment_parameter',str(set(updates)-set(allowed)))
    best=current_evaluation(session,'geometry');before=session.scene.candidate_id()
    session.scene.validate(updates,'geometry');loss,metrics=session.evaluator.evaluate({**session.scene.params,**updates},'mcp_environment_update',640,480);session.save()
    return dict(before_candidate_id=before,candidate_id=session.scene.candidate_id(),accepted=before!=session.scene.candidate_id(),loss=loss,coverage_before=best['metrics']['table'],coverage_after=metrics['table'] if metrics else None,physical_support_changed=False)


def evidence_export(session):
    folder=session.run/'evidence'/session.scene.candidate_id()
    static=folder/'static/portable/scene.xml';dynamic=folder/'dynamic/portable/scene.xml'
    if not static.exists():session.scene.export(folder/'static')
    if not dynamic.exists():session.scene.export(folder/'dynamic',physics=True)
    return folder,static,dynamic


def render_auxiliary(session,candidate_id):
    session.check_id(candidate_id)
    if session.scene.stage in ('frozen','final_readonly') and (session.run/'final_acceptance/views/normal_evidence.json').exists():
        return dict(candidate_id=candidate_id,folder='final_acceptance/views',normal_evidence=read(session.run/'final_acceptance/views/normal_evidence.json'))
    folder,static,_=evidence_export(session)
    if not (folder/'views/normal_evidence.json').exists():
        with session.budget.computation(dict(kind='auxiliary')):auxiliary(static,folder/'views',session.budget)
    return dict(candidate_id=candidate_id,folder=str((folder/'views').relative_to(session.run)),normal_evidence=read(folder/'views/normal_evidence.json'))


def diagnose_support(session,candidate_id):
    session.check_id(candidate_id)
    if session.scene.stage in ('frozen','final_readonly') and (session.run/'final_acceptance/rest/physical_diagnostic.json').exists():
        return dict(candidate_id=candidate_id,folder='final_acceptance/rest',initial_support=read(session.run/'final_acceptance/acceptance.json')['geometry'],rest=read(session.run/'final_acceptance/rest/physical_diagnostic.json'))
    folder,_,dynamic=evidence_export(session);target=folder/'rest'
    if not (target/'physical_diagnostic.json').exists():
        with session.budget.computation(dict(kind='native_rest',candidate_id=candidate_id),whole_task_slice=True):
            result=rest_from_disk(dynamic,target,session.config,session.budget,video=session.budget.remaining('render_diagnostic')>=100)
    else:result=read(target/'physical_diagnostic.json');session.budget.record('diagnostic_cache_hit',dict(candidate_id=candidate_id))
    return dict(candidate_id=candidate_id,folder=str(target.relative_to(session.run)),initial_support=session.scene.geometry_metrics(),rest=result)


def fit_appearance(session,parameter_ids=None,candidate_id=None):
    if candidate_id:session.check_id(candidate_id)
    if session.scene.stage!='appearance':raise SceneError('wrong_stage','appearance')
    s=session.scene;e=session.evaluator;geometry={k:v for k,v in s.params.items() if k in allowed_parameters(session,'geometry')}
    current=current_evaluation(session,'appearance')
    write(session.run/'geometry_frozen.json',dict(geometry_params=geometry,objects=current['metrics']['objects'],candidate_id=s.candidate_id()))
    allowed=set(allowed_parameters(session,'appearance'))
    if parameter_ids is not None and (not parameter_ids or set(parameter_ids)-allowed):raise SceneError('invalid_parameter_subset','appearance')
    subsets=[parameter_ids] if parameter_ids else session.config['appearance']['subsets']
    caps=[min(160,int(session.budget.remaining('render_appearance')))] if parameter_ids else session.config['appearance']['render_subbudgets']
    before=current['metrics']['appearance_loss']
    with session.budget.computation(dict(kind='appearance_fit'),whole_task_slice=True):
        for i,(ids,cap) in enumerate(zip(subsets,caps)):
            if session.budget.remaining('render_appearance')<1:break
            e.optimize(ids,min(cap,int(session.budget.remaining('render_appearance'))),f'classified_appearance_{i}',640,480)
    session.state['appearance_status']='completed';session.save();best=e.best['appearance_640x480']
    return dict(candidate_id=s.candidate_id(),before_loss=before,after_loss=best['metrics']['appearance_loss'],geometry_unchanged=all(s.params[k]==v for k,v in geometry.items()),metrics=best['metrics']['appearance'])


def repair_support(session,updates,evidence,confidence,candidate_id):
    session.check_id(candidate_id)
    if session.scene.stage!='geometry':raise SceneError('wrong_stage','support repair')
    if not evidence or not np.isfinite(confidence) or not .7<=confidence<=1:raise SceneError('evidence_required','support repair')
    before=diagnose_support(session,candidate_id);old_params=session.scene.params.copy();old_best=copy.deepcopy(session.evaluator.best)
    if set(updates)-set(allowed_parameters(session,'objects')):raise SceneError('invalid_parameter_subset','support repair')
    proposal=updates or {'apple.rx':0.,'apple.ry':0.,'plate.rx':0.,'plate.ry':0.}
    session.scene.validate(proposal,'geometry');session.scene.apply(proposal)
    session.evaluator.best={};current_evaluation(session,'geometry')
    session.evaluator.optimize(['apple.x','apple.y','apple.scale','plate.x','plate.y','plate.scale'],min(140,int(session.budget.remaining('render_repair'))),'physical_initialization_refit',640,480)
    after=diagnose_support(session,session.scene.candidate_id());metrics=current_evaluation(session,'geometry')['metrics'];g=gate(metrics,session.scene.geometry_metrics(),None,session.config)
    object_failures=[f for f in g['failures'] if not f.startswith('table.')]
    old_count=sum(x['passed'] for x in before['rest']['objects'].values());new_count=sum(x['passed'] for x in after['rest']['objects'].values())
    old_error=sum(x['translation_final_m']/.01+x['rotation_final_deg']/10 for x in before['rest']['objects'].values());new_error=sum(x['translation_final_m']/.01+x['rotation_final_deg']/10 for x in after['rest']['objects'].values())
    accepted=not object_failures and (new_count>old_count or (new_count==old_count and new_error<old_error-1e-4))
    trial_id=session.scene.candidate_id()
    if not accepted:
        session.scene.apply(old_params,internal=True);session.evaluator.best=old_best;session.evaluator.checkpoint()
    session.save()
    return dict(accepted=accepted,candidate_id=session.scene.candidate_id(),trial_candidate_id=trial_id,updates=proposal,before=before['rest'],after=after['rest'],geometry_failures=object_failures,reference_reset=False,physical_coefficients_changed=False)


def camera_hypotheses(session,evidence,confidence,candidate_id):
    session.check_id(candidate_id)
    if session.scene.stage!='geometry' or not evidence or not np.isfinite(confidence) or not .7<=confidence<=1:raise SceneError('camera_evidence_required','Need geometry stage and explicit finite evidence')
    if session.scene.params['cloth.coverage']<=0:raise SceneError('coverage_first_required','Inspect/repair cloth extent before camera hypotheses')
    base=current_evaluation(session,'geometry');old_camera=copy.deepcopy(session.scene.locked_camera);old_params=session.scene.params.copy();old_best=copy.deepcopy(session.evaluator.best)
    prior=read(session.run/'camera.json');mount=np.asarray(read(session.run/'camera_prior.json')['priors'][prior['mount_profile']]['mount_rotation_to_world']);r=np.asarray(prior['rotation_camera_to_world']);settings=session.config['camera']['conditional_revision'];rows=[];selected=None
    candidates=[]
    for axis in range(3):
        for direction in (-1,1):
            delta=mount[:,axis]*direction*.005;candidates.append((f'translation_{axis}_{direction}',delta,0.,axis,0.))
            candidates.append((f'rotation_{axis}_{direction}',np.zeros(3),direction,axis,0.))
    candidates.extend([('fov_minus_2',np.zeros(3),0.,0,-2.),('fov_plus_2',np.zeros(3),0.,0,2.)])
    try:
        with session.budget.computation(dict(kind='bounded_camera_comparison'),whole_task_slice=True):
            for name,delta,angle,axis,dfov in candidates:
                camera=copy.deepcopy(prior);rotation=Rotation.from_rotvec(mount[:,axis]*np.radians(angle)).as_matrix()@r
                if np.linalg.norm(delta)>settings['translation_norm_max_m'] or abs(angle)>settings['rotation_norm_max_deg'] or abs(dfov)>settings['fovy_delta_max_deg']:raise SceneError('camera_bounds',name)
                camera.update(xyz_m=(np.asarray(prior['xyz_m'])+delta).tolist(),rotation_camera_to_world=rotation.tolist(),quaternion_wxyz=quat_from_matrix(rotation).tolist(),fovy_deg=prior['fovy_deg']+dfov,hypothesis=name)
                session.scene.locked_camera=camera;session.scene.apply(old_params,internal=True);session.evaluator.best={}
                from .optimize import projection_refit
                proposal=projection_refit(session.scene,session.observation,session.budget,old_params)
                loss,metrics=session.evaluator.evaluate(proposal,'camera_probe_'+name,640,480)
                if metrics:rows.append(dict(name=name,camera=camera,loss=loss,metrics=metrics,params=proposal.copy(),refitted=True))
            for row in sorted(rows,key=lambda x:x['loss'])[:2]:
                session.scene.locked_camera=row['camera'];session.scene.apply(row['params'],internal=True);session.evaluator.best={};current_evaluation(session,'geometry')
                session.evaluator.optimize(['apple.x','apple.y','apple.scale','plate.x','plate.y','plate.scale'],80,'camera_joint_layout_'+row['name'],640,480)
                best=session.evaluator.best['geometry_640x480'];row.update(loss=best['loss'],metrics=best['metrics'],params=best['params'],refitted=True)
            for row in rows:
                row['rejection_reasons']=nondegrading(row['metrics'],base['metrics'],session.config)
                row['acceptable']=row['loss']<base['loss']-session.config['acceptance']['minimum_objective_improvement'] and not row['rejection_reasons']
                if row['acceptable'] and (selected is None or row['loss']<selected['loss']):selected=row
    finally:
        session.scene.locked_camera=selected['camera'] if selected else old_camera;session.scene.apply(selected['params'] if selected else old_params,internal=True);session.evaluator.best=old_best
        if selected:session.evaluator.best={};current_evaluation(session,'geometry')
        session.evaluator.checkpoint();session.save()
    result=dict(control=dict(camera=old_camera,metrics=base['metrics']),hypotheses=rows,accepted=selected['name'] if selected else None,candidate_id=session.scene.candidate_id(),real_calibration=False,coverage_test_preceded_camera=True,evidence=evidence)
    path=session.run/'camera_hypotheses'/('comparison_'+digest(result)[:12]+'.json');write(path,result);return result
