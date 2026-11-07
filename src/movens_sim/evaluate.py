"""Image, support geometry and genuinely integrated gravity diagnostics."""
from pathlib import Path
import time
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .util import read, sha, write

def edge(mask):
    return cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool)

def image_metrics(pred,target,valid):
    objects={};h,w=valid.shape
    for k,b0 in target.items():
        a=pred[k]&valid;b=b0&valid;union=(a|b).sum();inter=(a&b).sum()
        ya,xa=np.where(a);yb,xb=np.where(b)
        centroid=float(np.linalg.norm([xa.mean()-xb.mean(),ya.mean()-yb.mean()])) if len(xa) and len(xb) else float(np.hypot(w,h))
        ea=edge(a)&valid;eb=edge(b)&valid
        if ea.any() and eb.any():
            da=cv2.distanceTransform((~ea).astype(np.uint8),cv2.DIST_L2,5);db=cv2.distanceTransform((~eb).astype(np.uint8),cv2.DIST_L2,5);ed=float((da[eb].mean()+db[ea].mean())/2)
        else:ed=float(np.hypot(w,h))
        objects[k]=dict(iou=float(inter/max(1,union)),symmetric_edge_distance_px=ed,centroid_error_px=centroid,target_visible_pixels=int(b.sum()),render_visible_pixels=int(a.sum()),visible_target_fraction_in_valid_mask=float(b.sum()/max(1,b0.sum())),occlusion_policy='Evaluate only valid visible pixels; true hand/arm occlusion excluded by supplied pixel mask',keypoint_error_px=None,keypoint_note='No independently annotated 3D keypoints; centroid is reported separately')
    loss=float(np.mean([1-v['iou']+.6*v['symmetric_edge_distance_px']/np.hypot(w,h)+.5*v['centroid_error_px']/np.hypot(w,h) for v in objects.values()]))
    cloth_valid=valid.copy();cloth_valid[:int(.08*h)]=False
    for mask in target.values():cloth_valid&=~cv2.dilate(mask.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
    coverage=float((pred['cloth']&cloth_valid).sum()/max(1,cloth_valid.sum())) if 'cloth' in pred else 0.
    loss+=.2*(1-coverage)
    return dict(evaluator_version='visible_silhouette_v2',width=w,height=h,objects=objects,image_loss=loss,table=dict(iou=None,edge_distance_px=None,cloth_visible_background_coverage=coverage,coverage_region='Real image below top 8%, excluding object masks with a 3px margin; no robot occlusion in reference',reason='Full table boundary not visible; cloth coverage is a lower-bound extent constraint, not a unique table size measurement'),single_view_ambiguity=True)

def geometry_metrics(scene):
    out={};top=scene.context['table_top_z_m']
    for label in ('apple','plate'):
        pts=scene.vertices(label,collision=True)
        out[label]=dict(collision_bottom_z_m=float(pts[:,2].min()),nominal_support_top_z_m=top,signed_support_gap_m=float(pts[:,2].min()-top),aabb_world_m=[pts.min(axis=0).tolist(),pts.max(axis=0).tolist()])
    a=scene.data.xpos[scene.apple];p=scene.data.xpos[scene.plate]
    out['relative_apple_minus_plate_xyz_m']=(a-p).tolist()
    out['note']='Collision-vertex Z vs source nominal uncompressed support. Deformed local support/contact distances additionally checked in the dynamic model. Plate collision is decomposed convex pieces, not inferred from a concave visual mesh.'
    return out

def visual_pass(metrics,thresholds):
    return all(o['iou']>=thresholds['object_iou_min'] and o['symmetric_edge_distance_px']<=thresholds['edge_distance_px_max'] and o['centroid_error_px']<=thresholds['centroid_error_px_max'] for o in metrics['objects'].values()) and metrics['table']['cloth_visible_background_coverage']>=thresholds.get('cloth_coverage_min',.98)

def settle_diagnostic(scene,run,config,thresholds,width=640,height=480):
    from .render import Renderer,video_writer
    run=Path(run);path=scene.export('scene_physics.xml',physics=True);m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
    with np.load(path.with_suffix('.state.npz')) as z:d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel']
    # No external operation, no pinning or state/velocity resets during integration.
    d.qfrc_applied[:]=0;d.xfrc_applied[:]=0;mujoco.mj_forward(m,d)
    ai=m.body('apple_root').id;pi=m.body('plate_root').id
    initial=d.xpos[ai].copy();initial_q=d.xquat[ai].copy();initial_qvel=d.qvel.copy()
    fps=config['fps'];nframes=int(round(config['duration_s']*fps));start=time.monotonic();logs=[];states=[];finite=True;max_pen=0.;max_qvel=0.;max_drift=0.;max_angle=0.
    apple_bodies={ai}
    for b in range(m.nbody):
        if int(m.body_parentid[b]) in apple_bodies:apple_bodies.add(b)
    apple_geoms=set(np.where(np.isin(m.geom_bodyid,list(apple_bodies)))[0])
    step=int(0);min_dist=1.;min_support_force=0.;warnings_before=np.array([w.number for w in d.warning])
    records=[]
    with Renderer(m,width,height) as renderer:
        frames=[]
        for fi in range(nframes):
            target=fi/fps
            while d.time+1e-12<target:
                mujoco.mj_step(m,d);step+=1
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():finite=False;break
                max_qvel=max(max_qvel,float(np.max(np.abs(d.qvel))))
                for ci,c in enumerate(d.contact):
                    if any(int(g) in apple_geoms for g in c.geom):
                        dist=float(c.dist);min_dist=min(min_dist,dist);max_pen=max(max_pen,max(0,-dist))
                drift=float(np.linalg.norm(d.xpos[ai]-initial));angle=float(np.degrees(2*np.arccos(np.clip(abs(d.xquat[ai]@initial_q),0,1))))
                max_drift=max(max_drift,drift);max_angle=max(max_angle,angle)
            mujoco.mj_forward(m,d)
            frames.append(renderer.rgb(d));states.append(d.qpos.copy())
            records.append(dict(frame=fi,time_s=float(d.time),apple_xyz_m=d.xpos[ai].copy(),apple_quaternion_wxyz=d.xquat[ai].copy(),qvel_max=float(np.max(np.abs(d.qvel))),contact_count=int(d.ncon)))
            if not finite:break
        info=video_writer(run/'settle_diagnostic.mp4',frames,fps)
    # Integrate the remaining final interval too, even though video sample at t=duration-dt.
    while finite and d.time+1e-12<config['duration_s']:
        mujoco.mj_step(m,d);step+=1
        finite=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        drift=float(np.linalg.norm(d.xpos[ai]-initial));angle=float(np.degrees(2*np.arccos(np.clip(abs(d.xquat[ai]@initial_q),0,1))))
        max_drift=max(max_drift,drift);max_angle=max(max_angle,angle)
        for c in d.contact:
            if any(int(g) in apple_geoms for g in c.geom):max_pen=max(max_pen,max(0,-float(c.dist)))
    plate_joints=int(m.body_jntnum[pi]);warnings=np.array([w.number for w in d.warning])-warnings_before
    apple_pass=bool(finite and not warnings.any() and max_drift<=thresholds['translation_m_max'] and max_angle<=thresholds['rotation_deg_max'] and max_pen<=thresholds['penetration_m_max'])
    result=dict(version='native_gravity_diagnostic_v1',duration_s=float(d.time),requested_duration_s=config['duration_s'],physics_steps=step,timestep_s=m.opt.timestep,zero_external_operations=True,state_reset_during_integration=False,velocities_zeroed_during_integration=False,initial_velocity_source='Original post-settle source frame0 velocity, mapped by joint name',initial_max_abs_qvel=float(np.max(np.abs(initial_qvel))),numerically_finite=finite,warning_counts=warnings.tolist(),apple_translation_max_m=max_drift,apple_rotation_max_deg=max_angle,apple_contact_penetration_max_m=max_pen,max_abs_generalized_velocity=max_qvel,apple_local_rest_check_passed=apple_pass,plate_free_dynamics_checked=bool(plate_joints),plate_fixed_inherited_from_source=not bool(plate_joints),physics_passed=bool(apple_pass and plate_joints>0),limitations=['Inherited plate is fixed, so its free resting stability is unverified','Mass, friction and soft foundation are source assumptions without measurements','Short gravity diagnostic does not validate real dynamics'],wall_seconds=time.monotonic()-start,video=info,scene_sha256=sha(path))
    write(run/'physical_diagnostic.json',result);write(run/'diagnostic_timeline.json',records);np.savez_compressed(run/'diagnostic_states.npz',qpos=np.array(states),final_qpos=d.qpos,final_qvel=d.qvel)
    return result
