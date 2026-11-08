"""Executable, metered optimization transactions exposed through real MCP."""
from __future__ import annotations
import copy
import io
import json
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from .backend import SceneError
from .budget import BudgetExceeded
from .metrics import measure,gate
from .review_state import dependencies
from .v5_metrics import OBJECTIVES,objective,POLICY
from ..util import append,read,write,digest,sha,save_rgb,now,CODE_ROOT

GROUPS={
 'apple_color':['apple.color_strength','apple.red_r','apple.red_g','apple.red_b','apple.yellow_r','apple.yellow_g','apple.yellow_b','apple.color_center','apple.color_width'],
 'apple_red':['apple.color_strength','apple.red_r','apple.red_g','apple.red_b'],
 'apple_yellow':['apple.yellow_r','apple.yellow_g','apple.yellow_b','apple.color_center','apple.color_width'],
 'apple_highlight':['apple.specular','apple.shininess'],
 'plate_material':['plate.albedo','plate.specular','plate.shininess'],
 'shared_lighting':['light.ambient','light.diffuse','light.key_rx','light.key_ry','light.key_strength'],
 'plate_and_lighting':['plate.albedo','plate.specular','plate.shininess','light.ambient','light.diffuse','light.key_strength'],
 'cloth_color':['cloth.gray'],'cloth_optional':['cloth.texture_amplitude','cloth.texture_scale'],
 'table_coverage':['table.depth_scale','table.width_scale','table.front_extension','cloth.coverage'],
 'camera_coverage':[],
 'object_layout':['apple.x','apple.y','apple.scale','plate.x','plate.y','plate.scale'],
 'plate_pose':['plate.x','plate.y','plate.scale','plate.rx','plate.ry'],
 'plate_profile':['plate.depth_ratio','plate.x','plate.y','plate.scale'],
 'support_initialization':['apple.rx','apple.ry','plate.rx','plate.ry']}
ISSUES={'coverage':('observation','cloth',True),'camera':('observation','scene',True),
 'geometry':('geometry','objects',True),'plate_structure':('geometry','plate',True),
 'initial_support':('initial_support','objects',True),
 'apple_color':('appearance','apple',True),'apple_highlight':('appearance','apple',True),
 'plate_shading':('appearance','plate',True),'cloth_color':('appearance','cloth',True),
 'cloth_detail':('appearance','cloth',False),'rest':('rest','objects',True)}
TOOLS=('inspect_scene','diagnose_issue','optimize_issue','fit_appearance','select_candidate')


def execution_fingerprint():
    files=list(Path(__file__).parent.glob('v5*.py'))+[Path(__file__).parent/'backend.py',Path(__file__).parent/'session.py',Path(__file__).parent.parent/'render.py',Path(__file__).parent/'appearance.py',Path(__file__).parent/'environment.py',Path(__file__).parent/'metrics.py']
    return digest({p.name:sha(p) for p in files})


def capabilities(s):
    b=s.budget;groups={k:v for k,v in GROUPS.items()}
    if b.remaining('render_appearance')<1 or b.remaining('optimization_seconds')<=1200:
        groups={k:v for k,v in groups.items() if k in ('table_coverage','camera_coverage','object_layout','plate_pose','plate_profile','support_initialization')}
    if b.remaining('render_repair')<1 or b.remaining('optimization_seconds')<=1200:
        groups={k:v for k,v in groups.items() if k not in ('table_coverage','camera_coverage','object_layout','plate_pose','plate_profile','support_initialization')}
    mutable=s.scene.stage not in ('frozen','final_readonly')
    return dict(version='v5.1',tools=list(TOOLS if mutable else ('inspect_scene','diagnose_issue')),
        parameter_groups=groups,objectives=list(OBJECTIVES),issues={k:dict(category=v[0],region=v[1],blocking=v[2]) for k,v in ISSUES.items()},
        registry=s.scene.registry(),pending_candidate=s.state.get('pending_candidate'),
        duplicate_searches=s.state.get('search_signatures',[]),policy=POLICY,
        appearance_geometry_locked=True,shared_light_rotation_norm_max_deg=60,
        camera_absolute_bounds=s.config['camera']['conditional_revision'],
        final_compute_reserve_seconds=1200,scene_scale='Absolute relative to original training asset; no repeated scale chain')


def group_stage(group):return 'geometry' if group in ('table_coverage','camera_coverage','object_layout','plate_pose','plate_profile','support_initialization') else 'appearance'


def camera_valid(s,camera):
    p=read(s.run/'camera.json');t=s.config['camera']['conditional_revision']
    distance=np.linalg.norm(np.asarray(camera['xyz_m'])-p['xyz_m'])
    r=Rotation.from_matrix(np.asarray(camera['rotation_camera_to_world'])@np.asarray(p['rotation_camera_to_world']).T).magnitude()*180/np.pi
    if distance>t['translation_norm_max_m']+1e-10 or r>t['rotation_norm_max_deg']+1e-8 or abs(camera['fovy_deg']-42)>2+1e-9:raise SceneError('camera_bounds','Absolute original prior bounds exceeded')


def evaluate(s,params=None,camera=None,pool='render_diagnostic',tag='diagnostic'):
    with s.budget.computation(dict(kind='v5_evaluate',tag=tag)):
        return _evaluate(s,params,camera,pool,tag)


def _evaluate(s,params=None,camera=None,pool='render_diagnostic',tag='diagnostic'):
    from .control import check_stop
    check_stop();scene=s.scene;params=(params or scene.params).copy();camera=copy.deepcopy(camera or scene.locked_camera)
    original=scene.params.copy();old_camera=copy.deepcopy(scene.locked_camera);start=time.monotonic()
    scene.validate({k:v for k,v in params.items() if original.get(k)!=v})
    identity=digest(dict(params=params,camera=camera,source=scene.source_hash,implementation=execution_fingerprint(),evaluator=s.config['evaluator'],observation=s.observation['version'],resolution=[640,480]))
    folder=s.run/'v5_cache';folder.mkdir(exist_ok=True);jp=folder/(identity+'.json');npz=folder/(identity+'.npz')
    try:
        camera_valid(s,camera);scene.validate({k:v for k,v in params.items() if original.get(k)!=v});scene.locked_camera=camera;scene.apply(params,internal=True)
        if jp.exists() and npz.exists():
            row=read(jp);row['cache_hit']=True;s.budget.record('cache_hit',dict(key=identity,tag=tag))
        else:
            events=s.budget.reserve_render(pool,dict(kind='v5_candidate',tag=tag,candidate_id=scene.candidate_id(),cache_key=identity))
            rgb,masks=scene.render();metrics=measure(rgb,masks,s.observation,s.config)
            np.savez_compressed(npz,rgb=rgb,**masks)
            row=dict(params=params,camera=camera,candidate_id=scene.candidate_id(),metrics=metrics,geometry=scene.geometry_metrics(),cache_key=identity,cache_hit=False)
            write(jp,row)
            for event in events:s.budget.finish(event,elapsed=time.monotonic()-start)
            s.budget.record('render_passes',dict(rgb=1,segmentation=1,view='v5_candidate',cache_key=identity))
        s.budget.record('evaluation_request',dict(tag=tag,candidate_id=row['candidate_id'],cache_hit=row['cache_hit']))
        return row
    finally:
        scene.locked_camera=old_camera;scene.apply(original,internal=True)


def images(s,row):
    with np.load(s.run/'v5_cache'/(row['cache_key']+'.npz')) as z:return z['rgb'].copy(),{k:z[k].copy() for k in ('apple','plate','cloth','table')}


def result(status,**kw):
    return dict(execution_status=status,new_candidates=0,new_render_evaluations=0,cache_hits=0,parameters_changed=False,candidate_retained=False,stop_reason=status,**kw)


def counters(b):return {k:b.used(k) for k in ('render_total','render_appearance','render_repair')}


def guard_failures(s,current,base):
    a=gate(current['metrics'],current['geometry'],read(s.run/'metrics_mount_baseline.json') if (s.run/'metrics_mount_baseline.json').exists() else None,s.config)
    b=gate(base['metrics'],base['geometry'],None,s.config)
    # Preserve every already-passing hard check, including coverage and geometry.
    old=set(b['failures']);return [x for x in a['failures'] if x not in old]


def physical_key(s):return digest(dict(parameters=dependencies('rest',s.scene.params,s.scene.locked_camera,s.observation['version']),state=sha(s.scene.physics_state),template=sha(s.scene.physics_path),diagnostics=sha(Path(__file__).parent/'diagnostics.py'),backend=sha(Path(__file__).parent/'backend.py'),limits=s.config['rest_acceptance']))


def native_rest(s,force=False,folder=None,video=True):
    from .diagnostics import rest_from_disk
    key=physical_key(s);folder=Path(folder) if folder else s.run/'physics_evidence'/key[:20]
    path=folder/'physical_diagnostic.json'
    if path.exists() and not force:
        s.budget.record('diagnostic_cache_hit',dict(physical_fingerprint=key));return read(path),folder
    with s.budget.computation(dict(kind='v5_native_rest',physical_fingerprint=key),whole_task_slice=True):
        xml=s.scene.export(folder/'dynamic',physics=True)
        report=rest_from_disk(xml,folder,s.config,s.budget,video=video)
    return report,folder


def plate_views(s):
    from .diagnostics import auxiliary,spend_render,finish_render,load_state
    from ..render import Renderer
    folder=s.run/'visual_evidence'/s.scene.candidate_id();path=folder/'views/diagnostic.json'
    if path.exists():return read(path)
    xml=s.scene.export(folder/'static');views=folder/'views';views.mkdir(exist_ok=True)
    auxiliary(xml,views,s.budget)
    m,d=load_state(xml);camera=m.camera('approx_ego_camera').id
    light_backup=(m.light_diffuse.copy(),m.light_specular.copy(),m.light_ambient.copy())
    def render(name):
        events=spend_render(s.budget,'render_diagnostic',dict(kind='v5_auxiliary',view=name))
        with Renderer(m) as r:rgb=r.rgb(d)
        save_rgb(views/(name+'.png'),rgb);finish_render(s.budget,events,dict(rgb=1,segmentation=0,view=name))
    m.mat_specular[:]=0;render('specular_off')
    # Truly neutral control: remove explicit key light and all specular terms.
    m.light_diffuse[:]=0;m.light_specular[:]=0;m.light_ambient[:]=0
    m.vis.headlight.ambient[:]=.55;m.vis.headlight.diffuse[:]=.35;m.vis.headlight.specular[:]=0
    render('neutral_all_lights_control')
    events=spend_render(s.budget,'render_diagnostic',dict(kind='depth_normals'))
    with Renderer(m) as r:
        r.update(d);r.renderer.enable_depth_rendering();depth=r.renderer.render().copy()
    finish_render(s.budget,events,dict(depth=1,view='depth_normals'))
    f=240/np.tan(np.radians(m.cam_fovy[camera]/2));y,x=np.indices(depth.shape)
    pos=np.stack([(x-320)*depth/f,-(y-240)*depth/f,-depth],2);dx=np.gradient(pos,axis=1);dy=np.gradient(pos,axis=0);norm=np.cross(dx,dy);norm/=np.maximum(np.linalg.norm(norm,axis=2,keepdims=True),1e-9)
    save_rgb(views/'depth_normals.png',np.clip((norm+1)*127.5,0,255).astype(np.uint8))
    report=dict(folder=str(views.relative_to(s.run)),views=['support_side','support_oblique','neutral_ego','specular_off','neutral_all_lights_control','depth_normals'],real_shape_verified=False,note='SIM structure/normal diagnostic only; fixed-key light is disabled in the true neutral control; no real inner-profile claim')
    write(path,report);return report


def diagnose_issue(s,issue_id,candidate_id,diagnostic='automatic'):
    with s.budget.computation(dict(kind='complete_v5_diagnosis',issue_id=issue_id),whole_task_slice=True):
        value=_diagnose_issue(s,issue_id,candidate_id,diagnostic)
    value['remaining_budget']={k:s.budget.remaining(k) for k in s.config['budget']}
    s.state.setdefault('diagnostics',{})[issue_id]=value;s.save();return value


def _diagnose_issue(s,issue_id,candidate_id,diagnostic='automatic'):
    s.check_id(candidate_id)
    if issue_id not in ISSUES:raise SceneError('unknown_issue',issue_id)
    if s.scene.stage in ('frozen','final_readonly'):
        return dict(execution_status='completed',issue_id=issue_id,read_only=True,acceptance=read(s.run/'final_acceptance/acceptance.json'))
    start=counters(s.budget);row=evaluate(s);evidence={}
    if issue_id in ('plate_structure','plate_shading','apple_highlight') or diagnostic=='plate_views':evidence['visual']=plate_views(s)
    if issue_id in ('initial_support','rest') or diagnostic=='native_rest':
        p,folder=native_rest(s);evidence['physical']=p;evidence['physical_folder']=str(folder.relative_to(s.run))
    if issue_id in ('coverage','camera'):
        from .environment import coverage_seed
        evidence['coverage']=coverage_seed(s.scene,s.observation);evidence['support_bounds']=s.scene.support.bounds.tolist()
    report=dict(execution_status='completed',issue_id=issue_id,candidate_id=row['candidate_id'],metrics=row['metrics'],geometry=row['geometry'],evidence=evidence,new_candidates=0,new_render_evaluations=int(s.budget.used('render_total')-start['render_total']),cache_hits=int(row['cache_hit']),parameters_changed=False,candidate_retained=False,remaining_budget={k:s.budget.remaining(k) for k in s.config['budget']})
    s.state.setdefault('diagnostics',{})[issue_id]=report;s.save();return report


def optimize_issue(s,issue_id,objective_id,parameter_group,render_cap,candidate_id,bounds=None,evidence='',confidence=0.):
    s.check_id(candidate_id)
    if s.scene.stage in ('frozen','final_readonly'):raise SceneError('frozen_scene','No optimization after final freeze')
    if issue_id not in ISSUES or objective_id not in OBJECTIVES or parameter_group not in GROUPS:raise SceneError('unknown_search','Registered issue, objective and group required')
    if parameter_group in ('shared_lighting','plate_and_lighting') and objective_id!='scene_lighting_v5':raise SceneError('joint_objective_required','Shared-light interventions require the declared scene_lighting_v5 objective')
    if s.state.get('pending_candidate'):return result('capability_gap',reason='Review pending candidate with select_candidate first')
    if not 1<=render_cap<=256:raise SceneError('invalid_budget','1..256 local render cap required')
    stage=group_stage(parameter_group);pool='render_repair' if stage=='geometry' else 'render_appearance'
    if parameter_group=='cloth_optional':render_cap=min(render_cap,80-int(s.state.get('cloth_optional_used',0)))
    if s.budget.remaining(pool)<1 or s.budget.remaining('optimization_seconds')<=1200 or render_cap<1:return result('budget_exhausted',remaining_budget={k:s.budget.remaining(k) for k in s.config['budget']})
    if not np.isfinite(confidence) or not 0<=confidence<=1:raise SceneError('invalid_confidence','Finite confidence in [0,1] required')
    if stage=='geometry' and (not evidence or confidence<.7):raise SceneError('evidence_required','Geometry hypothesis requires evidence/confidence >=.7')
    ids=GROUPS[parameter_group];reg={p['id']:p for p in s.scene.registry()};ranges={k:(reg[k]['lower'],reg[k]['upper']) for k in ids}
    for b in bounds or []:
        k=b['param_id']
        if k not in ranges or not ranges[k][0]<=b['lower']<b['upper']<=ranges[k][1]:raise SceneError('invalid_bounds',str(b))
        ranges[k]=(b['lower'],b['upper'])
    signature=digest(dict(candidate=candidate_id,objective=objective_id,group=parameter_group,bounds=ranges,implementation=execution_fingerprint(),observation=s.observation['version']))
    if signature in s.state.get('search_signatures',[]):return result('no_improvement',reason='Duplicate completed search intercepted locally',duplicate=True)
    s.scene.stage=stage;s.scene.plate_tilt_enabled=True;s.save()
    before=counters(s.budget);base=evaluate(s,pool=pool,tag='search_control');best=base;before_score=objective(base['metrics'],objective_id,s.config);best_score=before_score
    evaluations=[];seen=set();new_seen=set();cache_hits=0;stop='local_cap';start=time.monotonic();cap=min(render_cap,int(s.budget.remaining(pool)));evaluated=0;bad=[];search_seconds=max(.01,s.budget.remaining('optimization_seconds')-1200)
    def trial(params,camera=None,tag='search'):
        nonlocal best,best_score,cache_hits,evaluated
        if evaluated>=cap or time.monotonic()-start>=search_seconds:raise StopIteration
        evaluated+=1
        try:
            row=evaluate(s,params,camera,pool,tag);score=objective(row['metrics'],objective_id,s.config);guard=guard_failures(s,row,base)
            cache_hits+=int(row['cache_hit']);seen.add(row['candidate_id'])
            if not row['cache_hit']:new_seen.add(row['candidate_id'])
            record=dict(candidate_id=row['candidate_id'],cache_key=row['cache_key'],params=params,camera=row['camera'],objective_id=objective_id,objective=float(score),guard_failures=guard,metrics=row['metrics'],cache_hit=row['cache_hit'])
            evaluations.append(record);append(s.run/'candidates_v5.jsonl',record)
            if not guard and score<best_score-s.config['acceptance']['minimum_objective_improvement']:
                best=row;best_score=score
            return score+len(guard)*10
        except (SceneError,ValueError) as exc:
            bad.append(dict(reason=str(exc)[:180],params=params));return 1e3
    try:
        with s.budget.computation(dict(kind='v5_search',issue_id=issue_id,objective_id=objective_id,group=parameter_group),reserve_seconds=search_seconds):
            if parameter_group=='camera_coverage':
                from .optimize import projection_refit
                from ..scene import quat_from_matrix
                prior=read(s.run/'camera.json');r=np.asarray(prior['rotation_camera_to_world']);mount=np.asarray(prior['mount_rotation_to_world']);candidates=[]
                for axis in range(3):
                    for sign in (-1,1):
                        for angle in (1.,3.,5.):candidates.append((axis,sign*angle,0.,0.))
                        for delta in (.005,.01,.02):candidates.append((axis,0.,sign*delta,0.))
                candidates += [(0,0.,0.,v) for v in (-2.,-1.,1.,2.)]
                for axis,angle,delta,dfov in candidates:
                    camera=copy.deepcopy(prior);rotation=Rotation.from_rotvec(mount[:,axis]*np.radians(angle)).as_matrix()@r
                    camera.update(xyz_m=(np.asarray(prior['xyz_m'])+mount[:,axis]*delta).tolist(),rotation_camera_to_world=rotation.tolist(),quaternion_wxyz=quat_from_matrix(rotation).tolist(),fovy_deg=42.+dfov)
                    old=s.scene.locked_camera;s.scene.locked_camera=camera
                    try:params=projection_refit(s.scene,s.observation,s.budget,base['params'])
                    finally:s.scene.locked_camera=old
                    trial(params,camera,'camera_absolute_hypothesis')
            else:
                origin=base['params'].copy();lo=np.array([ranges[k][0] for k in ids]);hi=np.array([ranges[k][1] for k in ids]);span=hi-lo
                seeds=[origin]
                if parameter_group in ('apple_color','apple_red'):
                    for center in (.30,.48,.62):seeds.append({**origin,'apple.color_strength':1.,'apple.color_center':center,'apple.color_width':.06})
                if parameter_group in ('apple_highlight','plate_material','plate_and_lighting'):
                    label='apple' if parameter_group=='apple_highlight' else 'plate'
                    for spec,shine in [(0.,.1),(.5,.12),(.9,.35),(.5,.7)]:seeds.append({**origin,label+'.specular':spec,label+'.shininess':shine})
                if parameter_group=='shared_lighting':
                    for x,y,strength,ambient in [(0,0,.4,.7),(30,0,.7,.4),(-30,0,.7,.4),(0,30,.7,.4),(0,-30,.7,.4)]:seeds.append({**origin,'light.key_rx':x,'light.key_ry':y,'light.key_strength':strength,'light.ambient':ambient})
                if parameter_group=='plate_and_lighting':
                    # Cross the albedo/illumination valley together. Coordinate
                    # steps alone reject either half of this valid hypothesis.
                    for albedo,ambient,diffuse,key in [(.95,.72,.12,.15),(.9,.65,.2,.2),(.85,.55,.3,.25),(.95,.75,.1,.1)]:
                        seeds.append({**origin,'plate.albedo':albedo,'plate.specular':.05,'plate.shininess':.12,'light.ambient':ambient,'light.diffuse':diffuse,'light.key_strength':key})
                if parameter_group=='table_coverage':
                    from .environment import coverage_seed
                    seeds.append({**origin,**coverage_seed(s.scene,s.observation)['updates']})
                    for depth in np.linspace(1.1,2.5,15):seeds.append({**origin,'table.depth_scale':float(depth),'cloth.coverage':1.})
                    for width in np.linspace(1,1.6,7):seeds.append({**origin,'table.width_scale':float(width),'cloth.coverage':1.})
                if parameter_group=='support_initialization':seeds.append({**origin,'apple.rx':0.,'apple.ry':0.,'plate.rx':0.,'plate.ry':0.})
                for p in seeds:trial({**origin,**{k:float(np.clip(p[k],*ranges[k])) for k in ids}},tag='grouped_seed')
                # For optional texture, also test a nonzero seed to avoid a flat scale coordinate.
                if parameter_group=='cloth_optional':trial({**origin,'cloth.texture_amplitude':.025,'cloth.texture_scale':.05},tag='optional_texture_seed')
                local_origin=best['params'].copy()
                def fun(x):
                    p=local_origin.copy();p.update({k:float(v) for k,v in zip(ids,lo+x*span)});return trial(p,tag='bounded_powell')
                x0=np.clip((np.array([local_origin[k] for k in ids])-lo)/span,0,1)
                minimize(fun,x0,method='Powell',bounds=[(0,1)]*len(ids),options=dict(maxfev=max(1,cap-evaluated),maxiter=8,xtol=.003,ftol=1e-4))
    except (StopIteration,BudgetExceeded) as exc:stop='local_cap' if isinstance(exc,StopIteration) else str(exc)
    finally:
        s.scene.locked_camera=base['camera'];s.scene.apply(base['params'],internal=True)
    improved=best['candidate_id']!=base['candidate_id'] and best_score<before_score-s.config['acceptance']['minimum_objective_improvement']
    if improved and stage=='geometry':
        s.scene.locked_camera=best['camera'];s.scene.apply(best['params'],internal=True)
        rest,_=native_rest(s)
        if not rest['physics_passed']:improved=False;stop='Candidate failed fixed one-second physics; rolled back'
        s.scene.locked_camera=base['camera'];s.scene.apply(base['params'],internal=True)
    operation_id=f'op_{len(s.state.get("search_history",[])):04d}'
    report=dict(execution_status='completed' if improved else ('budget_exhausted' if s.budget.remaining(pool)<1 else 'no_improvement'),operation_id=operation_id,issue_id=issue_id,objective_id=objective_id,parameter_group=parameter_group,bounds=ranges,evidence=evidence,confidence=confidence,
        before_candidate_id=base['candidate_id'],candidate_id=best['candidate_id'] if improved else base['candidate_id'],new_candidates=len(new_seen-{base['candidate_id']}),unique_candidates_evaluated=len(seen),candidate_evaluations=evaluated,new_render_evaluations=int(s.budget.used('render_total')-before['render_total']),cache_hits=cache_hits,
        parameters_changed=improved,candidate_retained=False,pending_review=improved,before_objective=before_score,after_objective=best_score if improved else before_score,metrics_before=base['metrics'],metrics_after=best['metrics'] if improved else base['metrics'],stop_reason=stop,errors=bad[:12],remaining_budget={k:s.budget.remaining(k) for k in s.config['budget']})
    folder=s.run/'operations'/operation_id;folder.mkdir(parents=True,exist_ok=True)
    for name,row in [('before',base),('after',best if improved else base)]:
        rgb,_=images(s,row);save_rgb(folder/(name+'.png'),rgb);write(folder/(name+'.json'),row)
    write(folder/'result.json',report);write(folder/'evaluations.json',evaluations)
    if improved:s.state['pending_candidate']=dict(operation_id=operation_id,issue_id=issue_id,before=base,after=best,stage=stage)
    s.state.setdefault('search_signatures',[]).append(signature);s.state.setdefault('search_history',[]).append(report)
    if parameter_group=='cloth_optional':s.state['cloth_optional_used']=s.state.get('cloth_optional_used',0)+report['new_render_evaluations']
    s.save();return report


def select_candidate(s,issue_id,decision,candidate_id,evidence=''):
    s.check_id(candidate_id);pending=s.state.get('pending_candidate')
    if not pending:raise SceneError('no_pending_candidate','No staged optimization to resolve')
    if decision not in ('keep','rollback'):raise SceneError('invalid_decision',decision)
    if pending['issue_id']!=issue_id:raise SceneError('wrong_issue','Selection must address the staged issue')
    row=pending['after'] if decision=='keep' else pending['before']
    s.scene.locked_camera=row['camera'];s.scene.apply(row['params'],internal=True)
    selection=dict(execution_status='completed',issue_id=issue_id,operation_id=pending['operation_id'],decision=decision,candidate_id=s.scene.candidate_id(),parameters_changed=decision=='keep',candidate_retained=decision=='keep',new_candidates=0,new_render_evaluations=0,cache_hits=0,evidence=evidence,utc=now())
    append(s.run/'candidate_decisions.jsonl',selection);s.state.setdefault('candidate_decisions',[]).append(selection);s.state['pending_candidate']=None;s.save();return selection


def make_server(s):
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.tools import Tool
    import fcntl
    def invoke(name,**kw):
        start=time.monotonic();before=s.scene.candidate_id();params=s.scene.params.copy()
        with (s.run/'.tool.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:
                from .control import check_stop
                check_stop()
                if name=='inspect_scene':value=s.inspect_scene()
                elif name=='fit_appearance':
                    if group_stage(kw['parameter_group'])!='appearance':raise SceneError('wrong_stage','fit_appearance cannot edit geometry')
                    value=globals()['optimize_issue'](s,**kw)
                else:value=globals()[name](s,**kw)
                payload=dict(ok=True,result=value)
            except Exception as exc:payload=dict(ok=False,error=dict(code=getattr(exc,'code',type(exc).__name__),message=str(exc)[:800]),result=result('error'))
            append(s.run/'mcp_tool_calls.jsonl',dict(tool=name,arguments=kw,ok=payload['ok'],before_candidate_id=before,candidate_id=s.scene.candidate_id(),parameter_changes={k:dict(before=params.get(k),after=v) for k,v in s.scene.params.items() if params.get(k)!=v},retest=payload.get('result'),error=payload.get('error'),transport='real_stdio_mcp',wall_seconds=time.monotonic()-start,utc=now()))
            s.budget.record('mcp_local_call',dict(tool=name,ok=payload['ok'],execution_status=payload.get('result',{}).get('execution_status')),time.monotonic()-start)
            return payload
    def inspect_scene()->dict:return invoke('inspect_scene')
    def diagnose_issue(issue_id:str,candidate_id:str,diagnostic:str='automatic')->dict:return invoke('diagnose_issue',issue_id=issue_id,candidate_id=candidate_id,diagnostic=diagnostic)
    def optimize_issue(issue_id:str,objective_id:str,parameter_group:str,render_cap:int,candidate_id:str,bounds:list[dict]|None=None,evidence:str='',confidence:float=0.)->dict:
        return invoke('optimize_issue',issue_id=issue_id,objective_id=objective_id,parameter_group=parameter_group,render_cap=render_cap,candidate_id=candidate_id,bounds=bounds,evidence=evidence,confidence=confidence)
    def fit_appearance(issue_id:str,objective_id:str,parameter_group:str,render_cap:int,candidate_id:str,bounds:list[dict]|None=None,evidence:str='',confidence:float=0.)->dict:
        return invoke('fit_appearance',issue_id=issue_id,objective_id=objective_id,parameter_group=parameter_group,render_cap=render_cap,candidate_id=candidate_id,bounds=bounds,evidence=evidence,confidence=confidence)
    def select_candidate(issue_id:str,decision:str,candidate_id:str,evidence:str='')->dict:return invoke('select_candidate',issue_id=issue_id,decision=decision,candidate_id=candidate_id,evidence=evidence)
    funcs=[inspect_scene,diagnose_issue]
    if s.scene.stage not in ('frozen','final_readonly'):funcs += [optimize_issue,fit_appearance,select_candidate]
    tools=[]
    for fn in funcs:
        tool=Tool.from_function(fn);tool.description={'inspect_scene':'Inspect shared state, executable contracts and cumulative budget','diagnose_issue':'Actual current-image, neutral/specular-off/normal or native-soft-layer diagnostics','optimize_issue':'Metered registered-objective search; stages a candidate, requires Terra comparison and select_candidate','fit_appearance':'Effective grouped appearance search; geometry/camera fixed; reports no_improvement/budget_exhausted explicitly','select_candidate':'Keep or roll back the staged candidate after comparing all metrics/images; never refunds budget'}[fn.__name__]
        tool.parameters['additionalProperties']=False;p=tool.parameters['properties']
        for key,values in [('issue_id',list(ISSUES)),('objective_id',list(OBJECTIVES)),('parameter_group',list(GROUPS)),('decision',['keep','rollback']),('diagnostic',['automatic','plate_views','native_rest'])]:
            if key in p:p[key]['enum']=values
        if fn.__name__=='fit_appearance':p['parameter_group']['enum']=[g for g in GROUPS if group_stage(g)=='appearance']
        if 'render_cap' in p:p['render_cap'].update(minimum=1,maximum=256)
        tools.append(tool)
    return MCPServer('Movens-Sim-v5',version='5.1',tools=tools,log_level='WARNING')
