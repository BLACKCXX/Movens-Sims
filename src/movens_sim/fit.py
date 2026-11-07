"""Bounded staged render-and-measure optimization; no differentiable renderer assumed."""
from __future__ import annotations
import time
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation
from .scene import look_at, matrix_from_quat, quat_from_matrix
from .evaluate import image_metrics,geometry_metrics
from .render import Renderer,overlay
from .util import append,read,save_rgb,write
from .vision import describe,load_observation

class BudgetExhausted(Exception): pass

def polygon_features(points,w,h):
    hull=cv2.convexHull(np.asarray(points,dtype=np.float32).reshape(-1,1,2));m=cv2.moments(hull)
    if m['m00']<1e-8:return np.ones(5)*100
    return np.array([m['m10']/m['m00']/w,m['m01']/m['m00']/h,np.sqrt(abs(m['mu20']/m['m00']))/w,np.sqrt(abs(m['mu02']/m['m00']))/h,(m['mu11']/m['m00'])/(w*h)*10])

def projection_seed(scene,target,w,h):
    """Circle/nominal plate anchor gives a reproducible pinhole pose initialization."""
    plate=describe(target['plate']);center=np.array(plate['centroid_xy_px']);ellipse=plate['ellipse'];axes=sorted(ellipse[1],reverse=True)
    elevation=np.arcsin(np.clip(axes[1]/axes[0],.3,.995));f=h/2/np.tan(np.radians(scene.params['camera.fovy']/2))
    verts=scene.vertices('plate');diam=max(np.ptp(verts[:,0]),np.ptp(verts[:,1]));depth=f*diam/axes[0]
    direction=scene.plate_pos[:2]-scene.cam_pos[:2];direction/=np.linalg.norm(direction)
    eye=scene.plate_pos+np.r_[-direction*depth*np.cos(elevation),depth*np.sin(elevation)]
    r=look_at(eye,scene.plate_pos);offset=np.array([(center[0]-w/2)/f*depth,-(center[1]-h/2)/f*depth,-depth]);eye=scene.plate_pos-r@offset
    rv=Rotation.from_matrix(r@scene.cam_rot.T).as_rotvec(degrees=True)
    ids=['camera.'+a for a in 'xyz']+['camera.r'+a for a in 'xyz'];reg={p['id']:p for p in scene.registry()}
    x0=np.array([*eye,*rv]);lo=np.array([reg[k]['lower'] for k in ids]);hi=np.array([reg[k]['upper'] for k in ids]);x0=np.clip(x0,lo+1e-5,hi-1e-5)
    ys,xs=np.where(target['plate']);target_points=np.c_[xs,ys];feature_target=polygon_features(target_points,w,h);projection_calls=0
    def error(x):
        nonlocal projection_calls
        projection_calls+=1
        r=Rotation.from_rotvec(x[3:],degrees=True).as_matrix()@scene.cam_rot
        v=(verts-x[:3])@r;z=-v[:,2]
        if z.min()<.02:return np.r_[np.ones(5)*10,np.zeros(6)]
        pixels=np.c_[w/2+f*v[:,0]/z,h/2-f*v[:,1]/z];features=polygon_features(pixels,w,h)
        return np.r_[(features-feature_target)*np.array([5,5,9,9,8]),(x-x0)/np.array([.3,.3,.3,50,50,50])*.0002]
    fit=least_squares(error,x0,bounds=(lo,hi),diff_step=1e-3,max_nfev=100,xtol=1e-8,ftol=1e-8,gtol=1e-8)
    values={k:float(v) for k,v in zip(ids,fit.x)};scene.apply(values)
    return dict(method='Nominal plate ellipse + bounded continuous projection-moment initialization; not a metric camera calibration',projection_evaluations=projection_calls,params=values,projection_residual=float(np.linalg.norm(fit.fun)),plate_diameter_nominal_m=diam)

def apple_xy_seed(scene,target,w,h):
    center=np.array(describe(target['apple'])['centroid_xy_px']);f=h/2/np.tan(np.radians(scene.params['camera.fovy']/2));r=matrix_from_quat(scene.model.cam_quat[scene.cam]);eye=scene.model.cam_pos[scene.cam];ray=r@np.array([(center[0]-w/2)/f,-(center[1]-h/2)/f,-1.])
    z=scene.data.qpos[scene.aq+2];t=(z-eye[2])/ray[2];point=eye+t*ray
    regs={p['id']:p for p in scene.registry()};return {'apple.'+a:float(np.clip(point[i],regs['apple.'+a]['lower'],regs['apple.'+a]['upper'])) for i,a in enumerate('xy')}

class Fitter:
    def __init__(self,scene,run,config,resume=False):
        self.scene=scene;self.run=Path(run);self.config=config;self.fc=config['fit'];self.start=time.monotonic();self.evaluations=0
        self.reference,self.target,self.valid=load_observation(run,self.fc['width'],self.fc['height']);self.width=self.reference.shape[1];self.height=self.reference.shape[0]
        self.renderer=Renderer(scene.model,self.width,self.height);self.best=scene.params.copy();self.best_loss=float('inf');self.history=self.run/'candidates.jsonl';self.stage='baseline';self.stage_best=float('inf');self.last_metrics=None;self.completed_steps=[]
        if resume and (self.run/'fit_checkpoint.json').exists():
            checkpoint=read(self.run/'fit_checkpoint.json');scene.apply(checkpoint['best_params']);self.best=checkpoint['best_params'];self.evaluations=checkpoint['evaluations'];self.completed_steps=checkpoint.get('completed_steps',[]);self.best_loss=float(checkpoint['best_loss'])
        if resume:
            if (self.run/'render_budget.json').exists():self.evaluations=max(self.evaluations,read(self.run/'render_budget.json')['reserved_evaluations'])
            if self.history.exists():
                for line in self.history.read_text().splitlines():
                    try:self.evaluations=max(self.evaluations,int(json.loads(line)['index']))
                    except (ValueError,KeyError):continue
    def close(self):self.renderer.renderer.close()
    def evaluate(self,values,stage=None,tag='numeric',appearance=False):
        if self.evaluations>=self.fc['max_evaluations'] or time.monotonic()-self.start>self.fc['max_seconds']:raise BudgetExhausted
        old=self.scene.params.copy();self.scene.validate(values);self.evaluations+=1
        # Reserve before rendering so an interrupted or rejected trial cannot regain budget.
        write(self.run/'render_budget.json',dict(reserved_evaluations=self.evaluations,max_evaluations=self.fc['max_evaluations']))
        self.scene.apply(values)
        masks=self.renderer.masks(self.scene.data,self.scene.geom_sets);metrics=image_metrics(masks,self.target,self.valid)
        reg={p['id']:p for p in self.scene.registry()};regularization=0.
        for k in self.best:
            if k.startswith('apple.r'):regularization+=((self.scene.params[k]-reg[k]['initial'])/55)**2*.0001
        if self.stage=='camera':
            o=metrics['objects']['plate'];loss=1-o['iou']+.6*o['symmetric_edge_distance_px']/np.hypot(self.width,self.height)+.5*o['centroid_error_px']/np.hypot(self.width,self.height)
        else:loss=metrics['image_loss']+regularization
        color_error=None
        if appearance:
            rgb=self.renderer.rgb(self.scene.data);valid=self.valid.copy();valid[:int(self.height*.1)]=False
            # Shape remains in the loss; color comes only after geometry stages.
            color_error=float(np.abs(rgb.astype(float)-self.reference.astype(float))[valid].mean()/255)
            loss+=.20*color_error
        collision=max([max(0,-float(c.dist)) for c in self.scene.data.contact] or [0.])
        geometry_valid=bool(np.isfinite(self.scene.data.qpos).all() and collision<.01)
        accepted=bool(geometry_valid and loss<self.stage_best-1e-8)
        if accepted:self.stage_best=loss;self.best_loss=loss;self.best=self.scene.params.copy();self.last_metrics=metrics
        append(self.history,dict(index=self.evaluations,stage=self.stage,tag=tag,params=self.scene.params,metrics=metrics,objective=loss,color_mae_fraction=color_error,regularization=regularization,geometry_valid=geometry_valid,display_contact_penetration_m=collision,accepted=accepted,remaining_evaluations=self.fc['max_evaluations']-self.evaluations,wall_seconds=time.monotonic()-self.start))
        if accepted or self.evaluations%20==0:write(self.run/'fit_checkpoint.json',dict(best_params=self.best,evaluations=self.evaluations,stage=self.stage,best_loss=self.best_loss,complete=False,completed_steps=self.completed_steps))
        self.scene.apply(self.best) # Every unaccepted trial rolls back, including GPT suggestions.
        return loss if geometry_valid else 10.+loss
    def optimize_stage(self,stage,ids,max_calls):
        self.stage=stage;self.stage_best=float('inf');self.evaluate(self.best,tag='stage_start',appearance=stage=='appearance')
        start=self.evaluations;origin=self.best.copy();reg={p['id']:p for p in self.scene.registry()}
        # Local bounded Powell search, normalized to a useful physical step.
        steps={k:(.04 if reg[k]['unit']=='m' else 12 if reg[k]['unit']=='deg' else .12) for k in ids}
        bounds=[((reg[k]['lower']-origin[k])/steps[k],(reg[k]['upper']-origin[k])/steps[k]) for k in ids]
        def fun(x):
            if self.evaluations-start>=max_calls:raise BudgetExhausted
            params=origin.copy();params.update({k:float(origin[k]+steps[k]*v) for k,v in zip(ids,x)})
            # Floating roundoff at an optimizer bound is clipped explicitly.
            for k in ids:params[k]=float(np.clip(params[k],reg[k]['lower'],reg[k]['upper']))
            return self.evaluate(params,appearance=stage=='appearance')
        try:minimize(fun,np.zeros(len(ids)),method='Powell',bounds=bounds,options=dict(maxfev=max_calls,maxiter=8,xtol=.007,ftol=.0001))
        except BudgetExhausted:pass
        self.scene.apply(self.best)
    def run_numeric(self,stages=None):
        stages=stages or self.fc['stages'];all_start=self.evaluations
        def mark(step):
            self.completed_steps.append(step);write(self.run/'fit_checkpoint.json',dict(best_params=self.best,evaluations=self.evaluations,stage=self.stage,best_loss=self.best_loss,complete=False,completed_steps=self.completed_steps))
        try:
            if 'camera' in stages and 'camera' not in self.completed_steps:
                if self.evaluations==0:
                    seed=projection_seed(self.scene,self.target,self.width,self.height);write(self.run/'camera_initialization.json',seed)
                    self.stage='camera';self.stage_best=float('inf');self.evaluate(self.scene.params.copy(),tag='projection_seed')
                self.optimize_stage('camera',['camera.x','camera.y','camera.z','camera.rx','camera.ry','camera.rz'],140);mark('camera')
            if 'layout' in stages and 'layout' not in self.completed_steps:
                self.stage='layout';self.stage_best=float('inf');self.evaluate(self.best,tag='layout_start')
                seed=apple_xy_seed(self.scene,self.target,self.width,self.height);self.evaluate({**self.best,**seed},tag='ray_plane_centroid_seed')
                self.optimize_stage('layout',['apple.x','apple.y','apple.rx','apple.ry','apple.rz'],245);mark('layout')
            if 'size' in stages and 'size' not in self.completed_steps:self.optimize_stage('size',['apple.scale','table.depth_scale','table.width_scale','table.front_extension'],115);mark('size')
            if 'layout' in stages and 'layout_refine' not in self.completed_steps:self.optimize_stage('layout',['apple.x','apple.y'],65);mark('layout_refine')
            if 'appearance' in stages and 'appearance' not in self.completed_steps:self.optimize_stage('appearance',['cloth.gray','light.ambient','plate.albedo'],65);mark('appearance')
        except BudgetExhausted:pass
        self.scene.apply(self.best);elapsed=time.monotonic()-self.start
        write(self.run/'fit_checkpoint.json',dict(best_params=self.best,evaluations=self.evaluations,complete=True,best_loss=self.best_loss,wall_seconds=elapsed,completed_steps=self.completed_steps))
        write(self.run/'fit_summary.json',dict(evaluations=self.evaluations,new_evaluations=self.evaluations-all_start,wall_seconds=elapsed,seconds_per_evaluation=elapsed/max(1,self.evaluations-all_start),max_evaluations=self.fc['max_evaluations'],projection_evaluations=read(self.run/'camera_initialization.json').get('projection_evaluations',0) if (self.run/'camera_initialization.json').exists() else 0,algorithm='Plate projection moment seed, bounded stage-wise Powell, silhouette IoU/edge/centroid and weak orientation regularizer; appearance last',stages=stages,best_params=self.best,budget_status='exhausted' if self.evaluations>=self.fc['max_evaluations'] else 'finished_stages'))
        return self.best

def save_final_images(scene,run):
    reference,target,valid=load_observation(run)
    with Renderer(scene.model,reference.shape[1],reference.shape[0]) as r:
        rgb=r.rgb(scene.data);masks=r.masks(scene.data,scene.geom_sets)
    save_rgb(Path(run)/'optimized_ego.png',rgb);save_rgb(Path(run)/'overlay.png',overlay(reference,masks,target))
    save_rgb(Path(run)/'comparison.png',np.concatenate([reference,rgb,overlay(reference,masks,target)],axis=1))
    metrics=image_metrics(masks,target,valid);metrics['geometry']=geometry_metrics(scene);write(Path(run)/'metrics_after.json',metrics);write(Path(run)/'parameters_optimized.json',scene.params);write(Path(run)/'parameter_registry.json',scene.registry());return metrics
