"""Single-episode, bounded robot-mount camera experiment and evidence bundle.

Independent of legacy camera/scale/table optimization. No API calls are made.
"""
from __future__ import annotations

import argparse
import copy
import csv
import html
import json
import os
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .evaluate import whole_scene_metrics, head_camera_gates
from .ingest import portable_copy
from .render import Renderer, overlay, video_writer
from .robot_camera import prepare_prior
from .scene import FixedGeometryScene, look_at, numbers, quat_from_matrix
from .util import ROOT, append, code_identity, digest, now, read, save_rgb, sha, video_info, write
from .vision import frame_at, largest


def save_state_by_name(xml, state, output):
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
    if state:
        with np.load(state) as z:data.qpos[:]=z['qpos']
    names=[];addresses=[];sizes=[]
    for j in range(model.njnt):
        names.append(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j));addresses.append(int(model.jnt_qposadr[j]));sizes.append(7 if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else 1)
    np.savez_compressed(output,qpos=data.qpos,joint_names=np.array(names),qpos_addresses=addresses,qpos_sizes=sizes)


def free_objects(xml, remove_continuation=False):
    tree=ET.parse(xml)
    for label in ('apple','plate'):
        node=tree.find(f".//body[@name='{label}_root']")
        if node.find('freejoint') is None and node.find("joint[@type='free']") is None:ET.SubElement(node,'freejoint',name=label+'_freejoint')
        for body in node.iter('body'):body.set('gravcomp','0')
    if remove_continuation:
        world=tree.find('worldbody');node=world.find("geom[@name='cloth_continuation']")
        if node is not None:world.remove(node)
    tree.write(xml,encoding='utf-8',xml_declaration=True)


def prepare_inputs(entry,config,run):
    run=Path(run);base=ROOT/config['comparison_root']/f"episode_{entry['episode']:06}"
    imported=ROOT/'data'/f"episode_{entry['episode']:06}"
    prior=prepare_prior(entry,config,run)
    for name in ['reference.png','reference_selection.json','optimized_ego.png','parameters_optimized.json','metrics_after.json','initial_transforms.json']:
        dest=run/'inputs'/('batch_'+name if name!='reference.png' else name);shutil.copy2(base/name,dest)
    # Verify decoded real frame identity; never use the old dynamic diagnostic video.
    rgb,fps=frame_at(imported/'real_ego.mp4',config['robot']['reference_frame'])
    old=cv2.cvtColor(cv2.imread(str(base/'reference.png')),cv2.COLOR_BGR2RGB)
    if not np.array_equal(rgb,old):raise ValueError('Real reference differs from batch frame0')
    save_rgb(run/'reference.png',rgb)
    folder=run/'masks';folder.mkdir(exist_ok=True)
    with np.load(base/'masks/observations.npz') as z:
        targets={k:z[k].copy() for k in ('apple','plate')};valid=z['valid'].copy();occlusion=z['occlusion'].copy()
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    cloth=largest((hsv[:,:,1]<75)&(hsv[:,:,2]<155))
    for mask in targets.values():cloth&=~mask
    targets['cloth']=cloth
    np.savez_compressed(folder/'observations.npz',**targets,valid=valid,occlusion=occlusion)
    for k,mask in {**targets,'valid':valid,'occlusion':occlusion}.items():cv2.imwrite(str(folder/(k+'.png')),mask.astype(np.uint8)*255)
    preview=rgb.copy()
    for k,color in [('cloth',(80,170,255)),('apple',(255,80,0)),('plate',(0,255,120))]:
        contours,_=cv2.findContours(targets[k].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(preview,contours,-1,color,1)
    cv2.line(preview,(0,int(.2*rgb.shape[0])),(rgb.shape[1]-1,int(.2*rgb.shape[0])),(255,0,255),1)
    save_rgb(run/'reference_masks_review.png',preview)
    write(run/'mask_definition.json',dict(version='initial_frame_whole_cloth_v1',objects='Copied audited batch frame0 HSV/connected-component masks; hashes preserved',cloth='Largest low-saturation S<75, V<155 component with internal holes filled; object pixels removed',valid_pixels=int(valid.sum()),occlusion_pixels=int(occlusion.sum()),top_fraction=.2,top_excluded=False,manual_review='Review reference_masks_review.png before interpreting qualification',rigid_table_edge='Not identifiable independently; do not confuse cloth/background edge with rigid tabletop edge',reference_video_sha256=sha(imported/'real_ego.mp4'),frame_index=0,timestamp_s=0,fps=fps))
    geometries={}
    for variant in config['geometry_variants']:
        folder=run/'geometries'/variant;folder.mkdir(parents=True,exist_ok=True)
        static_source=base/'scene'/('scene_optimized.xml' if variant=='batch_final' else 'scene_baseline.xml')
        physics_source=base/'scene'/('scene_physics.xml' if variant=='batch_final' else 'scene_physics_baseline.xml')
        static_receipt=portable_copy(static_source,folder/'static')
        physics_receipt=portable_copy(physics_source,folder/'physics')
        state_source=base/'scene'/('scene_physics.state.npz' if variant=='batch_final' else 'physics_initial_state.npz')
        state_file=folder/'physics_state_by_name.npz';save_state_by_name(physics_source,state_source,state_file)
        free_objects(folder/'static/scene.xml');free_objects(folder/'physics/scene.xml',remove_continuation=variant=='imported_initial')
        geometries[variant]=dict(static=str((folder/'static/scene.xml').relative_to(ROOT)),physics=str((folder/'physics/scene.xml').relative_to(ROOT)),physics_state=str(state_file.relative_to(ROOT)))
        write(folder/'provenance.json',dict(variant=variant,static_source=str(static_source),static_source_sha256=sha(static_source),physics_source=str(physics_source),physics_source_sha256=sha(physics_source),static_resources=static_receipt,physics_resources=physics_receipt,object_sizes_fixed=True,table_geometry_fixed=True,appearance_fixed=True,source_plate_fixed=True,plate_freejoint_added_for_pose_and_diagnostic=True,original_geometry_definition='Exact imported baseline, without the visual cloth continuation ring' if variant=='imported_initial' else 'Exact exported batch_v1 final geometry/scales/materials, including its inherited near-bound table expansion',soft_state='Old saved post-settle environment flex geometry; not the robot observation state'))
    write(run/'geometry_inputs.json',geometries)
    write(run/'source_manifest.json',dict(episode=entry['episode'],entry=entry,data_revision=entry['data_revision'],imported_manifest_sha256=sha(imported/'source_manifest.json'),batch_final_xml_sha256=sha(base/'scene/scene_optimized.xml'),batch_final_image_sha256=sha(base/'optimized_ego.png'),real_reference_sha256=sha(run/'reference.png'),camera_prior_sha256=sha(run/'camera_prior.json'),geometry_variants=config['geometry_variants'],gpt_calls=0))
    return prior,geometries


def observations(run,width=None,height=None):
    with np.load(Path(run)/'masks/observations.npz') as z:
        targets={k:z[k].copy() for k in ('apple','plate','cloth')};valid=z['valid'].copy()
    if width is not None and valid.shape!=(height,width):
        targets={k:cv2.resize(v.astype(np.uint8),(width,height),interpolation=cv2.INTER_NEAREST).astype(bool) for k,v in targets.items()}
        valid=cv2.resize(valid.astype(np.uint8),(width,height),interpolation=cv2.INTER_NEAREST).astype(bool)
    return targets,valid


def make_scene(geometry):
    return FixedGeometryScene(ROOT/geometry['static'],ROOT/geometry['physics'],ROOT/geometry['physics_state'])


class LayoutFit:
    def __init__(self,scene,prior,fovy,folder,run,config,refine=False,resume=False):
        self.scene=scene;self.prior=prior;self.fovy=fovy;self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.run=Path(run);self.config=config;self.refine=refine;self.fc=config['fit'];self.start=time.monotonic()
        self.targets,self.valid=observations(run,self.fc['width'],self.fc['height']);self.renderer=Renderer(scene.model,self.fc['width'],self.fc['height'])
        scene.set_camera(prior,fovy);self.reg={x['id']:x for x in scene.registry(self.fc['apple_tilt_bound_deg'],self.fc['apple_yaw_bound_deg'])}
        self.best=scene.params.copy();self.best_loss=float('inf');self.evaluations=0;self.render_evaluations=0;self.completed=[];self.orientation_evidence=[];self.previous_seconds=0.
        if refine:
            for a in 'xyz':
                for prefix,unit,bound in [('mount.d','m',config['refinement']['max_translation_norm_m']),('mount.r','deg',config['refinement']['max_rotation_angle_deg'])]:
                    k=prefix+a;self.reg[k]=dict(id=k,unit=unit,coordinate_frame='mounting-link frame',initial=0.,lower=-bound,upper=bound,stage='mount_refinement');self.best[k]=0.
        if resume and (self.folder/'checkpoint.json').exists():
            old=read(self.folder/'checkpoint.json');self.best=old['best'];self.best_loss=old['best_loss'] if old['best_loss'] is not None else float('inf');self.completed=old['completed'];self.orientation_evidence=old.get('orientation_evidence',[]);self.evaluations=old['evaluations'];self.render_evaluations=old['render_evaluations'];self.previous_seconds=old.get('elapsed_seconds',0.)
            if (self.folder/'budget.json').exists():self.evaluations=max(self.evaluations,read(self.folder/'budget.json')['reserved'])
            self.apply(self.best)

    def camera(self,params):
        if not self.refine:return self.prior
        dt=np.array([params['mount.d'+a] for a in 'xyz']);rv=np.array([params['mount.r'+a] for a in 'xyz'])
        if np.linalg.norm(dt)>self.config['refinement']['max_translation_norm_m']+1e-10 or np.linalg.norm(rv)>self.config['refinement']['max_rotation_angle_deg']+1e-10:raise ValueError('Mount correction exceeds norm bound')
        mount=np.asarray(self.prior['mount_rotation_to_world']);rotation=mount@Rotation.from_rotvec(rv,degrees=True).as_matrix()@np.asarray(self.prior['link_from_mujoco'])
        return {**self.prior,'xyz_m':np.asarray(self.prior['xyz_m'])+mount@dt,'quaternion_wxyz':quat_from_matrix(rotation),'rotation_camera_to_world':rotation,'down_angle_deg':float(np.degrees(np.arcsin(rotation[2,2])))}

    def apply(self,params):
        for k,v in params.items():
            r=self.reg[k]
            if isinstance(v,bool) or not np.isfinite(v) or not r['lower']-1e-10<=v<=r['upper']+1e-10:raise ValueError('Invalid candidate '+k)
        self.scene.set_camera(self.camera(params),self.fovy)
        self.scene.apply({k:v for k,v in params.items() if not k.startswith('mount.')})

    def checkpoint(self,complete=False):
        write(self.folder/'checkpoint.json',dict(best=self.best,best_loss=self.best_loss if np.isfinite(self.best_loss) else None,evaluations=self.evaluations,render_evaluations=self.render_evaluations,completed=self.completed,orientation_evidence=self.orientation_evidence,elapsed_seconds=self.previous_seconds+time.monotonic()-self.start,complete=complete))

    def evaluate(self,params,tag):
        if self.evaluations>=self.fc['max_evaluations'] or self.previous_seconds+time.monotonic()-self.start>self.fc['max_seconds']:raise StopIteration
        self.evaluations+=1;write(self.folder/'budget.json',dict(reserved=self.evaluations,limit=self.fc['max_evaluations']))
        metrics=None;geometry=None;reason=None;loss=10.;accepted=False
        try:
            self.apply(params);geometry=self.scene.geometry_metrics()
            valid=all(g['footprint_inside_physical_support'] and g['penetration_m']<=.001 and g['visual_support_gap_m'] is not None and -.001<=g['visual_support_gap_m']<=.002 for k,g in geometry.items() if k in ('apple','plate'))
            if not valid:raise ValueError('Candidate fails local collision support')
            masks=self.renderer.masks(self.scene.data,self.scene.geom_sets);self.render_evaluations+=1
            metrics=whole_scene_metrics(masks,self.targets,self.valid,self.config)
            regularizer=self.config['evaluator']['weights']['pose_regularization']*sum((params['apple.r'+a]/self.reg['apple.r'+a]['upper'])**2 for a in 'xyz')
            loss=metrics['image_loss']+regularizer
            accepted=loss<self.best_loss-1e-8
            if accepted:self.best=params.copy();self.best_loss=loss
        except (ValueError,AssertionError) as exc:reason=str(exc)
        append(self.folder/'candidates.jsonl',dict(index=self.evaluations,tag=tag,params=params,objective=loss,metrics=metrics,geometry=geometry,accepted=accepted,rejection_reason=reason,elapsed_s=time.monotonic()-self.start))
        self.apply(self.best)
        if accepted or self.evaluations%20==0:self.checkpoint()
        return loss

    def optimize(self,ids,budget,tag):
        origin=self.best.copy();start=self.evaluations
        steps=np.array([.02 if self.reg[k]['unit']=='m' else 5. for k in ids])
        bounds=[((self.reg[k]['lower']-origin[k])/s,(self.reg[k]['upper']-origin[k])/s) for k,s in zip(ids,steps)]
        def objective(x):
            if self.evaluations-start>=budget:raise StopIteration
            p=origin.copy();p.update({k:float(np.clip(origin[k]+s*v,self.reg[k]['lower'],self.reg[k]['upper'])) for k,s,v in zip(ids,steps,x)})
            return self.evaluate(p,tag)
        try:minimize(objective,np.zeros(len(ids)),method='Powell',bounds=bounds,options=dict(maxfev=budget,maxiter=self.fc['max_iterations'],ftol=self.fc['ftol'],xtol=self.fc['xtol']))
        except StopIteration:pass

    def centroid_seed(self):
        s=self.scene;r=np.asarray(self.prior['rotation_camera_to_world']);eye=np.asarray(self.prior['xyz_m']);w=self.fc['width'];h=self.fc['height'];f=h/2/np.tan(np.radians(self.fovy/2));p=self.best.copy()
        for label in ('apple','plate'):
            yy,xx=np.where(self.targets[label]);ray=r@np.array([(xx.mean()-w/2)/f,-(yy.mean()-h/2)/f,-1.])
            z=s.initial_poses[label]['xyz_m'][2];point=eye+ray*((z-eye[2])/ray[2])
            for i,a in enumerate('xy'):
                k=label+'.'+a;p[k]=float(np.clip(point[i],self.reg[k]['lower'],self.reg[k]['upper']))
        return p

    def run_fit(self):
        try:
            if 'xy' not in self.completed:
                self.evaluate(self.best,'initial_local_support');self.evaluate(self.centroid_seed(),'ray_support_seed')
                self.optimize(['apple.x','apple.y','plate.x','plate.y'],280,'xy');self.completed.append('xy');self.checkpoint()
            if 'orientation' not in self.completed:
                before=self.best_loss;probes=[];original=self.best.copy();probe_best=original.copy();probe_loss=before
                for a in 'xyz':
                    for direction in (-1,1):
                        self.best=original.copy();self.best_loss=before
                        p=original.copy();p['apple.r'+a]=float(np.clip(p['apple.r'+a]+direction*5,self.reg['apple.r'+a]['lower'],self.reg['apple.r'+a]['upper']))
                        loss=self.evaluate(p,'orientation_evidence_probe');probes.append(dict(axis=a,step_deg=direction*5,objective=loss))
                        if loss<probe_loss:probe_best=p.copy();probe_loss=loss
                self.orientation_evidence=probes
                self.best=original;self.best_loss=before
                if probe_loss<before-.002:
                    self.best=probe_best;self.best_loss=probe_loss;self.apply(self.best)
                    self.optimize(['apple.rx','apple.ry','apple.rz'],200,'silhouette_supported_orientation')
                self.completed.append('orientation');self.checkpoint()
            if self.refine and 'mount' not in self.completed:
                self.optimize(['mount.d'+a for a in 'xyz']+['mount.r'+a for a in 'xyz']+['apple.x','apple.y','plate.x','plate.y'],400,'bounded_mount_correction');self.completed.append('mount');self.checkpoint()
            if 'xy_refine' not in self.completed:
                self.optimize(['apple.x','apple.y','plate.x','plate.y'],90,'xy_refine');self.completed.append('xy_refine')
        except StopIteration:pass
        self.apply(self.best);self.checkpoint(complete=True)
        near=[k for k,v in self.best.items() if min(v-self.reg[k]['lower'],self.reg[k]['upper']-v)<.01*(self.reg[k]['upper']-self.reg[k]['lower'])]
        write(self.folder/'fit_summary.json',dict(algorithm='Centroid ray/support seed; bounded Powell XY; orientation only after an observed >=0.002 objective improvement from 5deg probes; optional norm-bounded mounting-frame refinement',evaluations=self.evaluations,render_evaluations=self.render_evaluations,max_evaluations=self.fc['max_evaluations'],wall_seconds=time.monotonic()-self.start,completed=self.completed,best_params=self.best,best_loss=self.best_loss if np.isfinite(self.best_loss) else None,near_bounds=near,orientation_evidence=self.orientation_evidence,camera_fixed=not self.refine,gpt_calls=0,budget_exhausted=self.evaluations>=self.fc['max_evaluations']))
        write(self.folder/'parameter_registry.json',list(self.reg.values()))
        return self.best

    def close(self):self.renderer.renderer.close()


def render_metrics(scene,targets,valid,config,width=640,height=480):
    with Renderer(scene.model,width,height) as renderer:
        rgb=renderer.rgb(scene.data);masks=renderer.masks(scene.data,scene.geom_sets)
    return rgb,masks,whole_scene_metrics(masks,targets,valid,config)


def reload_check(scene,xml,rgb,masks):
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
    with np.load(xml.with_suffix('.state.npz')) as z:data.qpos[:]=z['qpos'];data.qvel[:]=z['qvel']
    mujoco.mj_forward(model,data);groups={}
    for name in ('apple','plate','table'):
        bodies={model.body(name+'_root').id}
        for b in range(model.nbody):
            if int(model.body_parentid[b]) in bodies:bodies.add(b)
        groups[name]=np.flatnonzero(np.isin(model.geom_bodyid,list(bodies)))
    groups['cloth']=[g for g in range(model.ngeom) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or '').startswith('frozen_') or (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or '')=='cloth_continuation']
    with Renderer(model,640,480) as r:other=r.rgb(data);newmasks=r.masks(data,groups)
    cam=model.camera('approx_ego_camera').id
    errors=dict(camera_position_m=float(np.max(np.abs(scene.data.cam_xpos[scene.cam]-data.cam_xpos[cam]))),camera_rotation_matrix=float(np.max(np.abs(scene.data.cam_xmat[scene.cam]-data.cam_xmat[cam]))),rgb_mae=float(np.abs(rgb.astype(float)-other.astype(float)).mean()),mask_changed_pixels={k:int(np.count_nonzero(masks[k]!=newmasks[k])) for k in ('apple','plate','cloth')},object_position_m={k:float(np.max(np.abs(scene.data.xpos[scene.model.body(k+'_root').id]-data.xpos[model.body(k+'_root').id]))) for k in ('apple','plate')})
    passed=errors['camera_position_m']<1e-9 and errors['camera_rotation_matrix']<1e-9 and max(errors['object_position_m'].values())<1e-9 and errors['rgb_mae']<.05 and max(errors['mask_changed_pixels'].values())<=4
    return dict(passed=passed,errors=errors,robot_removed='pelvis' not in [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b) for b in range(model.nbody)],portable_compile=True,xml_sha256=sha(xml),state_sha256=sha(xml.with_suffix('.state.npz')))


def auxiliary_views(scene,folder):
    saved=(scene.model.cam_pos[scene.cam].copy(),scene.model.cam_quat[scene.cam].copy(),float(scene.model.cam_fovy[scene.cam]))
    center=(scene.data.xpos[scene.apple]+scene.data.xpos[scene.plate])/2
    for name,offset in [('support_side',np.array([.0,-.7,.08])),('support_oblique',np.array([-.4,-.5,.4]))]:
        eye=center+offset;scene.set_camera(dict(xyz_m=eye,quaternion_wxyz=quat_from_matrix(look_at(eye,center))),50,lock=False)
        with Renderer(scene.model,640,480) as renderer:save_rgb(Path(folder)/(name+'.png'),renderer.rgb(scene.data))
    scene.set_camera(dict(xyz_m=saved[0],quaternion_wxyz=saved[1]),saved[2])


def save_candidate(scene,folder,run,config,baseline,metadata):
    folder=Path(folder);targets,valid=observations(run);reference=cv2.cvtColor(cv2.imread(str(Path(run)/'reference.png')),cv2.COLOR_BGR2RGB)
    rgb,masks,metrics=render_metrics(scene,targets,valid,config);geometry=scene.geometry_metrics();gates=head_camera_gates(metrics,baseline,geometry,config['thresholds'])
    fit=read(folder/'fit_summary.json');gates['boundary_passed']=not fit['near_bounds'];gates['static_passed']&=gates['boundary_passed']
    save_rgb(folder/'optimized_ego.png',rgb);save_rgb(folder/'overlay.png',overlay(reference,masks,targets))
    write(folder/'metrics_after.json',metrics);write(folder/'geometry_checks.json',geometry)
    xml=scene.export(folder/'scene');checks=reload_check(scene,xml,rgb,masks);write(folder/'reload_check.json',checks)
    gates['static_passed']&=checks['passed']
    actual_cam=dict(xyz_m=scene.data.cam_xpos[scene.cam].copy(),quaternion_wxyz=quat_from_matrix(scene.data.cam_xmat[scene.cam]),rotation_camera_to_world=scene.data.cam_xmat[scene.cam].reshape(3,3),fovy_deg=float(scene.model.cam_fovy[scene.cam]),fy_px=240/np.tan(np.radians(scene.model.cam_fovy[scene.cam]/2)),resolution=[640,480],down_angle_deg=float(np.degrees(np.arcsin(scene.data.cam_xmat[scene.cam].reshape(3,3)[2,2]))),height_above_nominal_cloth_m=float(scene.data.cam_xpos[scene.cam,2]-.7771160388486943),coordinate_frame='world meters, quaternion wxyz, camera -Z forward +Y up')
    actual_cam['height_above_object_support_m']={label:float(actual_cam['xyz_m'][2]-np.mean(np.asarray(scene.support(np.array([[scene.params[label+'.x'],scene.params[label+'.y']]]))))) for label in ('apple','plate')}
    write(folder/'camera_final.json',actual_cam);write(folder/'parameters_optimized.json',scene.params)
    result=dict(**metadata,gates=gates,objective=metrics['image_loss'],metrics=metrics,geometry=geometry,near_bounds=fit['near_bounds'],camera=actual_cam,reload_passed=checks['passed'],gpt_calls=0,definition='Static 3D MuJoCo reconstruction candidate, not a real capture; nominal scale and approximate robot mount')
    write(folder/'result.json',result)
    return result


def rest_diagnostic(scene,folder,config):
    if config.get('rest_acceptance',{}).get('basis')=='terminal':
        from .reconstruction.diagnostics import rest_from_disk
        path=scene.export(Path(folder)/'physics',physics=True)
        return rest_from_disk(path,folder,config)
    folder=Path(folder);path=scene.export(folder/'physics',physics=True)
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
    with np.load(path.with_suffix('.state.npz')) as z:data.qpos[:]=z['qpos'];data.qvel[:]=z['qvel']
    data.qfrc_applied[:]=0;data.xfrc_applied[:]=0;mujoco.mj_forward(model,data)
    roots={k:model.body(k+'_root').id for k in ('apple','plate')};members={};geom_labels={};initial={};stats={}
    for label,b in roots.items():
        bodies={b}
        for child in range(model.nbody):
            if int(model.body_parentid[child]) in bodies:bodies.add(child)
        members[label]=bodies
        for g in np.flatnonzero(np.isin(model.geom_bodyid,list(bodies))):geom_labels[int(g)]=label
        initial[label]=(data.xpos[b].copy(),data.xquat[b].copy())
        stats[label]=dict(translation_max_m=0.,rotation_max_deg=0.,penetration_max_m=0.,initial_penetration_m=0.,initial_min_contact_distance_m=None,source_inherited_mass_kg=float(sum(model.body_mass[list(bodies)])),has_freejoint=bool(model.body_jntnum[b]))
    def contacts(initial_frame=False):
        for c in data.contact:
            labels={geom_labels[int(g)] for g in c.geom if int(g) in geom_labels}
            for label in labels:
                st=stats[label];st['penetration_max_m']=max(st['penetration_max_m'],max(0.,-float(c.dist)))
                if initial_frame:
                    st['initial_penetration_m']=max(st['initial_penetration_m'],max(0.,-float(c.dist)))
                    old=st['initial_min_contact_distance_m'];st['initial_min_contact_distance_m']=float(c.dist) if old is None else min(old,float(c.dist))
    contacts(True);fps=config['diagnostic']['fps'];duration=config['diagnostic']['duration_s'];frames=[];timeline=[];warnings=np.array([w.number for w in data.warning]);finite=True;steps=0
    with Renderer(model,640,480) as renderer:
        for frame in range(int(round(fps*duration))+1):
            target=min(duration,frame/fps)
            while data.time+1e-12<target:
                mujoco.mj_step(model,data);steps+=1
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():finite=False;break
                contacts()
                for label,b in roots.items():
                    p,q=initial[label];st=stats[label]
                    st['translation_max_m']=max(st['translation_max_m'],float(np.linalg.norm(data.xpos[b]-p)))
                    st['rotation_max_deg']=max(st['rotation_max_deg'],float(np.degrees(2*np.arccos(np.clip(abs(data.xquat[b]@q),0,1)))))
            mujoco.mj_forward(model,data)
            if frame<int(round(fps*duration)):frames.append(renderer.rgb(data))
            timeline.append(dict(time_s=float(data.time),objects={k:dict(xyz_m=data.xpos[b].copy(),quaternion_wxyz=data.xquat[b].copy()) for k,b in roots.items()},qvel_max=float(np.max(np.abs(data.qvel))),contact_count=int(data.ncon)))
            if not finite:break
    info=video_writer(folder/'settle_diagnostic.mp4',frames,fps);warning_delta=np.array([w.number for w in data.warning])-warnings;t=config['thresholds']
    for st in stats.values():st['passed']=bool(finite and not warning_delta.any() and st['has_freejoint'] and st['translation_max_m']<=t['translation_m_max'] and st['rotation_max_deg']<=t['rotation_deg_max'] and st['penetration_max_m']<=t['penetration_m_max'])
    result=dict(version='free_apple_plate_zero_initial_velocity_gravity_v1',objects=stats,physics_passed=all(x['passed'] for x in stats.values()),numerically_finite=finite,warning_counts=warning_delta.tolist(),duration_s=float(data.time),steps=steps,gravity_m_s2=model.opt.gravity,zero_external_operations=True,initial_velocity_policy='zero once before integration; never reset or pinned during stepping',physical_parameter_source='Inherited XML explicit apple inertia and plate geom density/inertia inference; unmeasured, not fitted. Both visual/collision geometry and original soft model retained.',soft_geometry='Name-mapped saved source flex qpos, with zero initial node velocities',video=info,real_dynamics_validated=False)
    write(folder/'physical_diagnostic.json',result);write(folder/'diagnostic_timeline.json',timeline);np.savez_compressed(folder/'diagnostic_final_state.npz',qpos=data.qpos,qvel=data.qvel)
    return result


def load_case_scene(result,geometry):
    scene=make_scene(geometry);scene.registry();scene.set_camera(result['camera'],result['camera']['fovy_deg']);scene.apply(read(Path(result['folder'])/'parameters_optimized.json'))
    return scene


def labelled_comparison(images,labels):
    panels=[]
    for rgb,label in zip(images,labels):
        panel=np.zeros((512,640,3),np.uint8);panel[32:]=rgb;cv2.putText(panel,label,(10,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA);panels.append(panel)
    return np.concatenate(panels,axis=1)


def case_outputs(folder):
    return [p for p in Path(folder).rglob('*') if p.is_file() and 'receipts' not in p.parts and 'physics' not in p.parts and p.name not in ('settle_diagnostic.mp4','physical_diagnostic.json','diagnostic_timeline.json','diagnostic_final_state.npz')]


def systematic_bias(result,config):
    residuals=[result['metrics']['objects'][k]['centroid_residual_xy_px'] for k in ('apple','plate')]
    r=config['refinement'];center=False
    if all(v is not None for v in residuals):
        a,b=map(np.asarray,residuals)
        center=bool(min(np.linalg.norm(a),np.linalg.norm(b))>=r['systematic_centroid_error_px_min'] and a@b/(np.linalg.norm(a)*np.linalg.norm(b))>=r['shared_error_cosine_min'])
    ratios=[np.sqrt(result['metrics']['objects'][k]['render_visible_pixels']/max(1,result['metrics']['objects'][k]['target_visible_pixels']))-1 for k in ('apple','plate')]
    size=ratios[0]*ratios[1]>0 and min(map(abs,ratios))>=.05
    return bool(center or size)


def candidate_name(variant,profile,fovy,refine=False):
    return variant+'__'+profile+'__fovy_'+str(float(fovy)).replace('.','p')+('__refined' if refine else '')


def fit_candidate(variant,profile,fovy,prior,geometry,run,config,identity,baseline,resume,seed=None):
    from .pipeline import complete_receipt,receipt_valid
    name=candidate_name(variant,profile,fovy,seed is not None);folder=Path(run)/'candidates'/name
    if resume and receipt_valid(folder,'fit',identity):
        print(name+': verified resume',flush=True);return read(folder/'result.json')
    folder.mkdir(parents=True,exist_ok=True);scene=make_scene(geometry);scene.set_camera(prior,fovy)
    targets,valid=observations(run)
    rgb,_,before=render_metrics(scene,targets,valid,config);save_rgb(folder/'camera_only.png',rgb);write(folder/'metrics_before.json',before)
    local_config=copy.deepcopy(config)
    if seed is not None:local_config['fit']['max_evaluations']=config['refinement']['max_evaluations']
    fitter=LayoutFit(scene,prior,fovy,folder,run,local_config,seed is not None,resume)
    try:
        if seed is not None and not (resume and (folder/'checkpoint.json').exists()):
            fitter.best.update(read(Path(seed['folder'])/'parameters_optimized.json'));fitter.completed=['xy','orientation'];fitter.evaluate(fitter.best,'strict_prior_seed')
        fitter.run_fit()
        metadata=dict(name=name,geometry_variant=variant,mount_profile=profile,fovy_deg=fovy,refinement=seed is not None,folder=str(folder),strict_parent=seed['name'] if seed else None)
        result=save_candidate(scene,folder,run,config,baseline,metadata)
    finally:fitter.close()
    complete_receipt(folder,'fit',identity,case_outputs(folder),0.)
    print(name+f": IoU apple={result['metrics']['objects']['apple']['iou']:.4f}, plate={result['metrics']['objects']['plate']['iou']:.4f}; cloth={result['metrics']['table']['cloth_iou']:.4f}; static={result['gates']['static_passed']}",flush=True)
    return result


def select_candidates(results,variants):
    selected={}
    for variant in variants:
        cases=[r for r in results if r['geometry_variant']==variant]
        strict=[r for r in cases if not r['refinement']]
        best_strict=min(strict,key=lambda r:(not r['gates']['static_passed'],r['objective']))
        best=min(cases,key=lambda r:(not r['gates']['static_passed'],r['objective']))
        alternatives=[dict(name=r['name'],objective=r['objective'],plate_xyz_m=r['geometry']['plate']['xyz_m'],apple_xyz_m=r['geometry']['apple']['xyz_m'],fovy_deg=r['fovy_deg'],camera_xyz_m=r['camera']['xyz_m']) for r in cases if r['objective']<=best['objective']+.01 and r['name']!=best['name']]
        selected[variant]=dict(best=best['name'],best_strict=best_strict['name'],near_equal_candidates=alternatives)
    return selected


def build_report(run,config,results,selection,diagnostics):
    run=Path(run);by_name={r['name']:r for r in results};baseline=read(run/'batch_final_metrics.json');rows=[];cards=[]
    reference=cv2.cvtColor(cv2.imread(str(run/'reference.png')),cv2.COLOR_BGR2RGB)
    batch=cv2.cvtColor(cv2.imread(str(run/'inputs/batch_optimized_ego.png')),cv2.COLOR_BGR2RGB)
    for variant,sel in selection.items():
        best=by_name[sel['best']];folder=Path(best['folder']);scene=load_case_scene(best,read(run/'geometry_inputs.json')[variant]);auxiliary_views(scene,folder)
        fitted=cv2.cvtColor(cv2.imread(str(folder/'optimized_ego.png')),cv2.COLOR_BGR2RGB)
        camera_only=cv2.cvtColor(cv2.imread(str(folder/'camera_only.png')),cv2.COLOR_BGR2RGB)
        if variant=='batch_final':second=batch;second_label='batch_v1 final (unchanged)'
        else:
            original=make_scene(read(run/'geometry_inputs.json')[variant])
            with Renderer(original.model,640,480) as renderer:second=renderer.rgb(original.data)
            save_rgb(run/'imported_initial_original_camera.png',second);second_label='Imported original geometry/camera'
        comparison=labelled_comparison([reference,second,camera_only,fitted],['Real frame 0',second_label,f"Mount only; VFOV {best['fovy_deg']:g} deg",'Mount + supported object layout'])
        save_rgb(run/(variant+'_comparison.png'),comparison)
        before=read(folder/'metrics_before.json');after=best['metrics'];diag=diagnostics.get(best['name'],{})
        attribution=dict(geometry_variant=variant,chosen_candidate=best['name'],camera_only_metrics=before,layout_fitted_metrics=after,baseline_metrics=baseline if variant=='batch_final' else None,change_policy='Camera-only panel preserves this geometry branch initial object poses. Last panel changes only supported object poses. FOV is explicitly labelled; object sizes/materials/table are frozen.',remaining_uncertainty=['Nominal fixed root, no measured base pose','Actual camera model, optical offset, image processing and distortion unconfirmed','Old soft deformation is inherited; not newly measured','Single-view nominal scale; hidden table and object geometry remain ambiguous'],physical_diagnostic=diag)
        write(run/(variant+'_attribution.json'),attribution)
        cards.append(f'<article><h2>{html.escape(variant)}</h2><p>{html.escape(best["name"])}</p><img src="{variant}_comparison.png"><p>固定三维场景仿真；不是实拍。安装与光心偏移为近似，FOV 不是标定。</p><p>静态判据：{best["gates"]["static_passed"]}；物理判据：{diag.get("physics_passed",False)}</p><video controls src="candidates/{best["name"]}/ego_static.mp4"></video><video controls src="candidates/{best["name"]}/settle_diagnostic.mp4"></video><p><a href="candidates/{best["name"]}/overlay.png">轮廓</a> · <a href="candidates/{best["name"]}/support_side.png">支撑侧视</a> · <a href="candidates/{best["name"]}/scene/scene_optimized.xml">XML</a> · <a href="candidates/{best["name"]}/physical_diagnostic.json">静置记录</a></p></article>')
    for r in results:
        m=r['metrics'];d=diagnostics.get(r['name'],{})
        rows.append(dict(candidate=r['name'],geometry=r['geometry_variant'],mount=r['mount_profile'],fovy_deg=r['fovy_deg'],refinement=r['refinement'],apple_iou=m['objects']['apple']['iou'],plate_iou=m['objects']['plate']['iou'],apple_edge_px=m['objects']['apple']['symmetric_edge_distance_px'],plate_edge_px=m['objects']['plate']['symmetric_edge_distance_px'],apple_centroid_px=m['objects']['apple']['centroid_error_px'],plate_centroid_px=m['objects']['plate']['centroid_error_px'],cloth_iou=m['table']['cloth_iou'],top_cloth_iou=m['table']['top_cloth_iou'],false_background=m['table']['false_background_on_real_cloth_fraction'],static_passed=r['gates']['static_passed'],physical_checked=bool(d),physical_passed=d.get('physics_passed',False),near_bounds=';'.join(r['near_bounds']),objective=r['objective']))
    with (run/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    passed=[r for r in results if r['gates']['static_passed']]
    final=dict(episode=read(run/'source_manifest.json')['episode'],processing_complete=True,strict_candidate_count=sum(not r['refinement'] for r in results),refinement_candidate_count=sum(r['refinement'] for r in results),visual_and_support_passed=bool(passed),integration_eligible=bool(passed),recommended=False,mode_status='experimental_static_pass' if passed else 'validation_failed_not_promoted',physical_passed=any(diagnostics.get(r['name'],{}).get('physics_passed',False) for r in passed),selected=selection,baseline_metrics=baseline,gpt_calls=0,api_model_unused=config['gpt']['model'],numeric_evaluations=sum(read(Path(r['folder'])/'fit_summary.json')['evaluations'] for r in results),real_hardware_confirmed=False,real_dynamics_validated=False,thresholds=config['thresholds'],completed_utc=now())
    write(run/'result.json',final)
    doc='<!doctype html><html lang="zh"><meta charset="utf-8"><title>Head camera pilot · episode 0</title><style>body{font:16px system-ui;background:#171b20;color:#e6e8ec;max-width:1500px;margin:30px auto;padding:20px}img{width:100%}video{width:48%;max-width:640px}a{color:#7cd0ff}article{padding:20px;background:#232a32;margin:20px 0}</style><h1>G1 头部相机先验 · episode 000000</h1><p>两组固定几何；16 组严格先验，必要时各增加一组有界安装修正。顶区完整计分。API 调用为 0。</p><p><a href="results.csv">全部指标</a> · <a href="camera_prior.json">相机来源</a> · <a href="result.json">判定</a> · <a href="reference_masks_review.png">分割核查</a></p>'+''.join(cards)+'</html>'
    (run/'index.html').write_text(doc)
    return final


def run_experiment(config,episode,resume=False,dry_run=False,stage=None):
    import fcntl
    from .pipeline import complete_receipt,receipt_valid
    if episode not in config['allowed_episodes']:raise ValueError('This pilot only authorizes configured episode 0')
    entries=[e for e in read(ROOT/config['manifest'])['entries'] if e['episode']==episode]
    if len(entries)!=1:raise ValueError('Episode must appear exactly once in manifest')
    entry=entries[0];run=ROOT/config['run_root']/f'episode_{episode:06}'
    if dry_run:return dict(episode=episode,run=str(run),strict_cases=len(config['geometry_variants'])*len(config['robot']['mount_profiles'])*len(config['intrinsics']['fovy_candidates_deg']),max_numeric_evaluations=10800,gpt_calls=0,read_only_sources=True)
    run.mkdir(parents=True,exist_ok=True);lock=(run/'.pilot.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        imported=ROOT/'data'/f'episode_{episode:06}';base=ROOT/config['comparison_root']/f'episode_{episode:06}'
        raw=Path(config['source_project'])/'data/apple_raw'/entry['data_revision']/'data'/f'chunk-{episode//1000:03}'/f'episode_{episode:06}.parquet'
        identity=dict(version='head_camera_pilot_identity_v1',config_sha256=digest(config),code_hashes=code_identity(),manifest_sha256=sha(ROOT/config['manifest']),source_manifest_sha256=sha(imported/'source_manifest.json'),raw_parquet_sha256=sha(raw),baseline_hashes={n:sha(base/n) for n in ['scene/scene_optimized.xml','scene/scene_baseline.xml','scene/scene_physics.xml','scene/scene_physics_baseline.xml','scene/scene_physics.state.npz','scene/physics_initial_state.npz','masks/observations.npz','reference.png']},mujoco_version=mujoco.__version__)
        if (run/'run_identity.json').exists() and read(run/'run_identity.json')!=identity:raise ValueError('Incompatible input/code/config identity; preserve this attempt and use a new run root')
        write(run/'run_identity.json',identity);write(run/'resolved_config.json',config)
        append(run/'commands.jsonl',dict(utc=now(),command='robot_mount_prior pilot',episode=episode,resume=resume,stage=stage,gpt_calls=0))
        snapshot=run/'code_snapshot';snapshot.mkdir(exist_ok=True)
        for name in identity['code_hashes']:
            dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        stages=['prepare','fit','validate','render'] if stage is None else [stage]
        all_results=None
        for current in stages:
            for dependency in ['prepare','fit','validate','render'][:['prepare','fit','validate','render'].index(current)]:
                if not receipt_valid(run,dependency,identity):raise ValueError('Missing compatible '+dependency+' before '+current)
            if resume and receipt_valid(run,current,identity):print(current+': verified resume',flush=True);continue
            start=time.monotonic();write(run/'status.json',dict(stage=current,status='running',utc=now()));print(current+': start',flush=True)
            if current=='prepare':
                prepare_inputs(entry,config,run)
                review=config.get('mask_review',{})
                if not review.get('reviewed') or review.get('reference_sha256')!=sha(run/'reference.png') or review.get('preview_sha256')!=sha(run/'reference_masks_review.png'):raise ValueError('Exact reference and segmentation preview require a recorded visual review')
                write(run/'mask_review.json',dict(**review,objects='Visible contours match apple and plate; no robot arm overlaps',cloth='Boundary follows upper cloth edge; top band remains valid'))
                outputs=[p for folder in ['inputs','masks','geometries'] for p in (run/folder).rglob('*') if p.is_file()]+[run/n for n in ['camera_prior.json','reference.png','reference_masks_review.png','mask_definition.json','mask_review.json','geometry_inputs.json','source_manifest.json']]
            elif current=='fit':
                priors=read(run/'camera_prior.json')['priors'];geometries=read(run/'geometry_inputs.json');targets,valid=observations(run)
                old=make_scene(geometries['batch_final']);rgb,_,baseline=render_metrics(old,targets,valid,config);save_rgb(run/'batch_final_verified.png',rgb);write(run/'batch_final_metrics.json',baseline)
                all_results=[]
                for variant,geometry in geometries.items():
                    for profile,prior in priors.items():
                        for fovy in config['intrinsics']['fovy_candidates_deg']:
                            all_results.append(fit_candidate(variant,profile,fovy,prior,geometry,run,config,identity,baseline,resume))
                for variant,geometry in geometries.items():
                    strict=min([r for r in all_results if r['geometry_variant']==variant and not r['refinement']],key=lambda r:(not r['gates']['static_passed'],r['objective']))
                    trigger=config['refinement']['enabled'] and not strict['gates']['static_passed'] and systematic_bias(strict,config)
                    write(run/(variant+'_refinement_decision.json'),dict(triggered=trigger,strict_candidate=strict['name'],rule='Shared centroid residual >=3px with cosine >=0.8, OR same-sign projected linear size error >=5% for both objects. No broad camera search.'))
                    if trigger:all_results.append(fit_candidate(variant,strict['mount_profile'],strict['fovy_deg'],priors[strict['mount_profile']],geometry,run,config,identity,baseline,resume,strict))
                selection=select_candidates(all_results,config['geometry_variants']);write(run/'selection.json',selection);write(run/'candidate_index.json',all_results)
                outputs=[run/n for n in ['candidate_index.json','selection.json','batch_final_metrics.json','batch_final_verified.png']]
                outputs += [Path(r['folder'])/'receipts/fit.json' for r in all_results]
            elif current=='validate':
                results=read(run/'candidate_index.json');by_name={r['name']:r for r in results};geometries=read(run/'geometry_inputs.json');selection=read(run/'selection.json');diagnostics={}
                for variant,sel in selection.items():
                    for name in dict.fromkeys([sel['best_strict'],sel['best']]):
                        result=by_name[name];folder=Path(result['folder']);scene=load_case_scene(result,geometries[variant])
                        diagnostics[name]=rest_diagnostic(scene,folder,config)
                        print(name+': physical_passed='+str(diagnostics[name]['physics_passed']),flush=True)
                write(run/'diagnostics.json',diagnostics)
                outputs=[run/'diagnostics.json']+[p for name in diagnostics for p in (Path(by_name[name]['folder'])/'physics').rglob('*') if p.is_file()]
                outputs += [Path(by_name[name]['folder'])/n for name in diagnostics for n in ['physical_diagnostic.json','settle_diagnostic.mp4','diagnostic_timeline.json','diagnostic_final_state.npz']]
            else:
                results=read(run/'candidate_index.json');by_name={r['name']:r for r in results};selection=read(run/'selection.json');diagnostics=read(run/'diagnostics.json');geometries=read(run/'geometry_inputs.json')
                outputs=[]
                for variant,sel in selection.items():
                    for name in dict.fromkeys([sel['best_strict'],sel['best']]):
                        result=by_name[name];folder=Path(result['folder']);scene=load_case_scene(result,geometries[variant])
                        with Renderer(scene.model,640,480) as renderer:frame=renderer.rgb(scene.data)
                        info=video_writer(folder/'ego_static.mp4',(frame for _ in range(int(round(config['render']['fps']*config['render']['duration_s'])))),config['render']['fps'])
                        write(folder/'ego_static_definition.json',dict(type='fixed_camera_initial_static_3d_simulation',real_capture=False,dynamics_validated=False,video=info,scene='scene/scene_optimized.xml',state='scene/scene_optimized.state.npz',camera='approx_ego_camera',objects='fixed initial layout; no integration',fovy_deg=result['fovy_deg']))
                        outputs.extend([folder/'ego_static.mp4',folder/'ego_static_definition.json'])
                build_report(run,config,results,selection,diagnostics)
                outputs += [run/n for n in ['result.json','results.csv','index.html']]+list(run.glob('*_comparison.png'))+list(run.glob('*_attribution.json'))
            complete_receipt(run,current,identity,outputs,time.monotonic()-start)
            print(current+f': complete ({time.monotonic()-start:.1f}s)',flush=True)
        write(run/'status.json',dict(status='complete' if receipt_valid(run,'render',identity) else 'partial',utc=now()))
        return read(run/'result.json') if (run/'result.json').exists() else dict(processing_complete=False,run=str(run))
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/head_camera_pilot_v1.json');parser.add_argument('--episode',type=int,required=True);parser.add_argument('--resume',action='store_true');parser.add_argument('--dry-run',action='store_true');parser.add_argument('--stage',choices=['prepare','fit','validate','render']);parser.add_argument('--run-root')
    args=parser.parse_args(argv);config=read(ROOT/args.config)
    if args.run_root:config['run_root']=args.run_root
    result=run_experiment(config,args.episode,args.resume,args.dry_run,args.stage)
    print(json.dumps(result,ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),flush=True)


if __name__=='__main__':main()
