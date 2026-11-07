"""Coarse multi-start projection plus joint nondifferentiable render fitting."""
from __future__ import annotations
import copy
import time
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from .budget import BudgetExceeded
from .backend import SceneError
from .metrics import measure, nondegrading
from ..fit import polygon_features
from ..scene import matrix_from_quat
from ..util import append, digest, read, write


def observation_at(observation,width,height):
    if observation['valid'].shape==(height,width):return observation
    return dict(rgb=cv2.resize(observation['rgb'],(width,height),interpolation=cv2.INTER_AREA),target={k:cv2.resize(v.astype(np.uint8),(width,height),interpolation=cv2.INTER_NEAREST).astype(bool) for k,v in observation['target'].items()},valid=cv2.resize(observation['valid'].astype(np.uint8),(width,height),interpolation=cv2.INTER_NEAREST).astype(bool),version=observation['version'])


class Evaluator:
    def __init__(self,scene,observation,budget,run,config):
        self.scene,self.observation,self.budget,self.run,self.config=scene,observation,budget,Path(run),config
        self.pool='render_initial';self.stage='geometry';self.best={};self.attempts=0;self.cache=self.run/'evaluation_cache';self.cache.mkdir(exist_ok=True)
        self.history=self.run/'candidates.jsonl';self.improvements=0
        path=self.run/'search_checkpoint.json'
        if path.exists():
            state=read(path);self.best=state['best'];self.attempts=state['attempts'];self.improvements=state.get('improvements',0)

    def best_key(self,w,h):return f'{self.stage}_{w}x{h}'

    def checkpoint(self):write(self.run/'search_checkpoint.json',dict(best=self.best,attempts=self.attempts,improvements=self.improvements))

    def evaluate(self,params,tag,width=320,height=240):
        from .control import check_stop
        check_stop()
        self.attempts+=1;start=time.monotonic();old=self.scene.params.copy();key=self.best_key(width,height)
        self.budget.record('evaluation_request',dict(tag=tag,stage=self.stage,width=width,height=height))
        status='rejected';reason=None;metrics=None;geometry=None;loss=10.;accepted=False;cached=False;events=[]
        try:
            with self.budget.computation(dict(kind='candidate',tag=tag)):
                changes={k:v for k,v in params.items() if k not in old or old[k]!=v};self.scene.validate(changes,self.stage)
                self.scene.apply(params,internal=True);geometry=self.scene.geometry_metrics()
                if self.config.get('batch',{}).get('enabled') and not geometry['physical_valid']:
                    raise SceneError('physical_constraint',','.join(geometry['failures']))
                cachekey=digest(dict(candidate=self.scene.candidate_id(),width=width,height=height,stage=self.stage,renderer=self.config['render'],observation=self.observation['version'],evaluator=self.config['evaluator'],mujoco=__import__('mujoco').__version__))
                cached_path=self.cache/(cachekey+'.json');image_path=self.cache/(cachekey+'.npz')
                if cached_path.exists() and image_path.exists():
                    metrics=read(cached_path)['metrics'];cached=True;self.budget.record('cache_hit',dict(key=cachekey))
                else:
                    events=self.budget.reserve_render(self.pool,dict(tag=tag,candidate_id=self.scene.candidate_id(),width=width,height=height,reference_frame=0,view='fixed_ego'))
                    rgb,masks=self.scene.render(width,height);metrics=measure(rgb,masks,observation_at(self.observation,width,height),self.config)
                    np.savez_compressed(image_path,rgb=rgb,**masks)
                    write(cached_path,dict(metrics=metrics,params=params,candidate_id=self.scene.candidate_id()))
                    for event in events:self.budget.finish(event,elapsed=time.monotonic()-start)
                    self.budget.record('render_passes',dict(rgb=1,segmentation=1),time.monotonic()-start)
                prior=.002*sum((np.log(params[label+'.scale'])/np.log(2))**2 for label in ('apple','plate'))
                loss=metrics['appearance_loss'] if self.stage=='appearance' else metrics['image_loss']+prior
                previous=self.best.get(key)
                reasons=nondegrading(metrics,previous['metrics'],self.config) if previous else []
                if self.stage=='appearance' and (self.run/'geometry_frozen.json').exists():
                    frozen=read(self.run/'geometry_frozen.json')
                    if any(params[k]!=v for k,v in frozen['geometry_params'].items()):reasons.append('geometry_changed_during_appearance')
                    if metrics['objects']!=frozen.get('objects',metrics['objects']):
                        # Pixel masks, not appearance-derived inner RGB edges, must be unchanged.
                        for label in ('apple','plate'):
                            if abs(metrics['objects'][label]['iou']-frozen['objects'][label]['iou'])>1e-12:reasons.append('silhouette_changed_during_appearance')
                minimum=self.config['acceptance']['minimum_objective_improvement']
                accepted=(previous is None or loss<previous['loss']-minimum) and not reasons
                if accepted:
                    self.best[key]=dict(params=params.copy(),loss=float(loss),metrics=metrics,geometry=geometry,cachekey=cachekey,candidate_id=self.scene.candidate_id());self.improvements+=1
                    status='accepted'
                else:reason=','.join(reasons) if reasons else 'no_substantial_improvement'
        except (SceneError,ValueError,AssertionError) as exc:
            reason=getattr(exc,'code',type(exc).__name__)+': '+str(exc)[:400]
        finally:
            row=dict(attempt=self.attempts,tag=tag,stage=self.stage,pool=self.pool,params=params,loss=float(loss),metrics=metrics,geometry=geometry,accepted=accepted,status=status,rejection_reason=reason,cache_hit=cached,elapsed_seconds=time.monotonic()-start)
            append(self.history,row)
            restore=self.best.get(key,{}).get('params',old)
            self.scene.apply(restore,internal=True)
            self.checkpoint()
        return float(loss),metrics

    def optimize(self,ids,cap,tag,width,height,start_params=None):
        reg={p['id']:p for p in self.scene.registry()};key=self.best_key(width,height)
        origin=(start_params or self.best.get(key,{}).get('params',self.scene.params)).copy()
        lo=np.array([reg[k]['lower'] for k in ids]);hi=np.array([reg[k]['upper'] for k in ids]);span=hi-lo
        x0=(np.array([origin[k] for k in ids])-lo)/span;count=0;stale=0;last=self.improvements
        def objective(x):
            nonlocal count,stale,last
            if count>=cap or stale>=self.config['fit']['local_no_improvement_evaluations']:raise StopIteration
            count+=1;p=origin.copy();p.update({k:float(v) for k,v in zip(ids,lo+x*span)})
            loss,_=self.evaluate(p,tag,width,height)
            stale=0 if self.improvements>last else stale+1;last=self.improvements
            return loss
        try:
            minimize(objective,x0,method='Powell',bounds=[(0.,1.)]*len(ids),options=dict(maxfev=cap,maxiter=12,ftol=1e-4,xtol=1e-3))
        except StopIteration:pass
        return self.best.get(key)


def projection_candidates(scene,observation,budget,config):
    """Continuous geometry projection; every residual call uses the global ledger."""
    w,h=640,480;r=np.asarray(scene.locked_camera['rotation_camera_to_world']);eye=np.asarray(scene.locked_camera['xyz_m']);f=h/2/np.tan(np.radians(scene.locked_camera['fovy_deg']/2))
    reg={p['id']:p for p in scene.registry()};initial=scene.params.copy();candidates={};provenance=[]
    for label in ('apple','plate'):
        b=scene.model.body(label+'_root').id;position=scene.data.xpos[b];rotation=scene.data.xmat[b].reshape(3,3)
        local=(scene.vertices(label)-position)@rotation/initial[label+'.scale'];collision=(scene.sample_collision(label)-position)@rotation/initial[label+'.scale']
        yy,xx=np.where(observation['target'][label]);target=polygon_features(np.c_[xx,yy],w,h)
        ids=[label+'.x',label+'.y',label+'.scale']+(['apple.rx','apple.ry','apple.rz'] if label=='apple' else [])
        lo=np.array([reg[k]['lower'] for k in ids]);hi=np.array([reg[k]['upper'] for k in ids]);span=hi-lo
        starts=qmc.Sobol(len(ids),scramble=True,seed=config['fit']['seed']+(label=='plate')).random_base2(4)
        base=np.array([initial[k] for k in ids]);starts[0]=(base-lo)/span
        rows=[]
        def residual(x):
            budget.reserve('projection_evaluations',detail=dict(object=label))
            vals=lo+x*span;p=dict(zip(ids,vals));rot=matrix_from_quat(scene.initial_poses[label]['quaternion_wxyz'])
            if label=='apple':rot=Rotation.from_rotvec(vals[3:],degrees=True).as_matrix()@rot
            col=collision*vals[2]@rot.T;col[:,:2]+=vals[:2];height=scene.support(col[:,:2])
            if not np.isfinite(height).all():return np.r_[np.ones(5)*5,(np.log(vals[2])*.005)]
            z=-np.min(col[:,2]-height);world=local*vals[2]@rot.T+np.r_[vals[:2],z];view=(world-eye)@r;depth=-view[:,2]
            if depth.min()<.02:return np.r_[np.ones(5)*5,(np.log(vals[2])*.005)]
            pixels=np.c_[w/2+f*view[:,0]/depth,h/2-f*view[:,1]/depth]
            error=(polygon_features(pixels,w,h)-target)*[5,5,9,9,8]
            return np.r_[error,np.log(vals[2])*.005]
        object_projection_start=budget.used('projection_evaluations');object_projection_calls=0
        original_residual=residual
        def limited_residual(x):
            nonlocal object_projection_calls
            if object_projection_calls>=config['budget']['projection_evaluations']//2:raise BudgetExceeded('Per-object share of projection budget')
            object_projection_calls+=1
            return original_residual(x)
        with budget.computation(dict(kind='projection_seeds',object=label)):
            for seed in starts:
                try:
                    fit=least_squares(limited_residual,seed,bounds=(np.zeros(len(ids)),np.ones(len(ids))),diff_step=.002,max_nfev=config['fit']['projection_max_per_start'],ftol=1e-5,xtol=1e-5,gtol=1e-5)
                    p=initial.copy();p.update({k:float(v) for k,v in zip(ids,lo+fit.x*span)});rows.append((float(np.linalg.norm(fit.fun)),p))
                except BudgetExceeded:break
        rows.sort(key=lambda x:x[0]);candidates[label]=rows[:5] or [(10.,initial)]
        provenance.append(dict(object=label,starts_completed=len(rows),parameter_ids=ids,algorithm='Sobol dispersed starts and bounded continuous least_squares on projected hull moments; XY/scale/orientation joint, saved local support Z',best_residual=rows[0][0] if rows else None))
    seeds=[initial]
    for _,a in candidates['apple']:
        for _,b in candidates['plate']:
            p=initial.copy();p.update({k:v for k,v in a.items() if k.startswith('apple.')});p.update({k:v for k,v in b.items() if k.startswith('plate.')});seeds.append(p)
    return seeds,provenance


def fit_geometry(evaluator,ids=None,round_index=0,resume_event=None):
    config=evaluator.config;fc=config['fit'];scene=evaluator.scene;budget=evaluator.budget
    if scene.stage!='geometry':raise SceneError('wrong_stage','fit_geometry')
    evaluator.stage='geometry';evaluator.pool='render_initial' if round_index==0 else 'render_repair'
    ids=ids or ['apple.x','apple.y','apple.scale','apple.rx','apple.ry','apple.rz','plate.x','plate.y','plate.scale']
    round_event=resume_event or budget.reserve('geometry_rounds',detail=dict(round_index=round_index,parameter_ids=ids))
    for key in ids:
        if key not in {x['id'] for x in scene.registry() if 'geometry' in x['stages']}:raise SceneError('wrong_stage',key)
    start_used=budget.summary()['used'].get(evaluator.pool,0);cap=600 if round_index==0 else fc['repair_round_caps'][min(round_index-1,2)]
    result=dict(round_index=round_index,parameter_ids=ids,algorithm='Sobol projection seeds; multi-start normalized bounded Powell; joint XY/absolute scale/orientation; full-resolution refinement',substeps=[])
    try:
        if round_index==0:
            evaluator.evaluate(scene.params.copy(),'mount_baseline',640,480)
            evaluator.evaluate(scene.params.copy(),'initial_joint_baseline',320,240)
            if (evaluator.run/'projection_seeds.json').exists():seeds=read(evaluator.run/'projection_seeds.json')['seeds']
            else:
                seeds,provenance=projection_candidates(scene,evaluator.observation,budget,config);write(evaluator.run/'projection_seeds.json',dict(provenance=provenance,seeds=seeds))
            ranked=[]
            for p in seeds[:fc['coarse_render_candidates']]:
                loss,metrics=evaluator.evaluate(p,'coarse_joint_seed',320,240)
                if metrics:ranked.append((loss,p))
            ranked.sort(key=lambda x:x[0]);result['substeps'].append('coarse_multiple_starts_including_both_scales')
            for index,(_,p) in enumerate(ranked[:fc['local_start_count']]):evaluator.optimize(ids,fc['initial_render_subbudgets'][1]//fc['local_start_count'],f'joint_local_start_{index}',320,240,p)
            low=evaluator.best.get('geometry_320x240')
            if low:evaluator.evaluate(low['params'],'full_resolution_candidate',640,480)
            evaluator.optimize(ids,fc['initial_render_subbudgets'][2],'joint_full_resolution_refinement',640,480)
        else:
            evaluator.optimize(ids,cap,f'directed_geometry_repair_{round_index}',640,480)
    except BudgetExceeded as exc:result['stop_reason']=str(exc)
    best=evaluator.best.get('geometry_640x480')
    if best:scene.apply(best['params'],internal=True)
    result.update(render_evaluations=budget.summary()['used'].get(evaluator.pool,0)-start_used,best_candidate_id=best['candidate_id'] if best else None,best_loss=best['loss'] if best else None)
    write(evaluator.run/f'geometry_round_{round_index:02}.json',result)
    budget.finish(round_event)
    return result
