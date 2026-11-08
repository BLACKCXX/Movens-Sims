"""Diagnostics on XML/state bytes, including native dynamic flex contacts."""
from __future__ import annotations
from pathlib import Path
import time
import mujoco
import numpy as np
from .backend import SceneBackend,SupportSurface
from ..render import Renderer,video_writer
from ..scene import look_at,quat_from_matrix
from ..util import write,sha,save_rgb


def load_state(xml):
    xml=Path(xml);m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m)
    with np.load(xml.with_suffix('.state.npz')) as z:
        d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel'];d.time=float(z['time']) if 'time' in z else 0.
    mujoco.mj_forward(m,d)
    return m,d


def members_and_groups(m):
    descendants={};groups={}
    for label in ('apple','plate','table'):
        bodies={m.body(label+'_root').id}
        for b in range(m.nbody):
            if int(m.body_parentid[b]) in bodies:bodies.add(b)
        descendants[label]=bodies;groups[label]=np.flatnonzero(np.isin(m.geom_bodyid,list(bodies)))
    groups['cloth']=np.array([g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('frozen_') or m.geom(g).name=='cloth_continuation'],int)
    return descendants,groups


def loaded_geometry(m,d,params,limits):
    """Use the common geometric definitions, exclusively on the loaded model/state."""
    s=object.__new__(SceneBackend);s.model=m;s.data=d;s.params=params;s.support_limits=limits
    s.descendants,s.geom_sets=members_and_groups(m)
    vertices=[];faces=[];offset=0
    if m.nflex:
        for i in range(m.nflex):
            if m.flex_group[i]==5:continue
            a=int(m.flex_vertadr[i]);n=int(m.flex_vertnum[i]);v=d.flexvert_xpos[a:a+n].copy();v[:,2]+=m.flex_radius[i]
            a=int(m.flex_elemdataadr[i]);n=int(m.flex_elemnum[i]);f=m.flex_elem[a:a+3*n].reshape(-1,3).copy()
            down=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])[:,2]<0;f[down]=f[down][:,[0,2,1]]
            vertices.append(v);faces.append(f+offset);offset+=len(v)
    else:
        for g in s.geom_sets['cloth']:
            if not (m.geom(g).name or '').startswith('frozen_'):continue
            mi=int(m.geom_dataid[g]);v=s.geom_vertices(g);a=int(m.mesh_faceadr[mi]);n=int(m.mesh_facenum[mi])
            vertices.append(v);faces.append(m.mesh_face[a:a+n]+offset);offset+=len(v)
    if not vertices:raise ValueError('No physical soft support in loaded scene')
    s.support=SupportSurface(np.concatenate(vertices),np.concatenate(faces),top_only=True)
    result=s.geometry_metrics()
    for label in ('apple','plate'):
        b=m.body(label+'_root').id;r=d.xmat[b].reshape(3,3)
        result[label]['dimensions_body_m']=np.ptp((s.vertices(label)-d.xpos[b])@r,axis=0).tolist()
        result[label]['collision_dimensions_body_m']=np.ptp((s.vertices(label,True)-d.xpos[b])@r,axis=0).tolist()
    return result


def spend_render(budget,pool,detail):
    return budget.reserve_render(pool,detail) if budget else []


def finish_render(budget,events,detail):
    if budget:
        for event in events:budget.finish(event)
        budget.record('render_passes',detail)


def auxiliary(xml,folder,budget=None,pool='render_diagnostic'):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);m,d=load_state(xml);cam=m.camera('approx_ego_camera').id
    camera=(m.cam_pos[cam].copy(),m.cam_quat[cam].copy(),float(m.cam_fovy[cam]));center=np.mean([d.xpos[m.body(k+'_root').id] for k in ('apple','plate')],0)
    paths=[]
    for name,offset in [('support_side',[0,-.65,.045]),('support_oblique',[-.4,-.5,.4]),('neutral_ego',None)]:
        if offset is None:
            m.cam_pos[cam],m.cam_quat[cam],m.cam_fovy[cam]=camera;m.vis.headlight.ambient[:]=.22;m.vis.headlight.diffuse[:]=.45;m.vis.headlight.specular[:]=0
            m.mat_specular[:]=0
        else:
            eye=center+offset;m.cam_pos[cam]=eye;m.cam_quat[cam]=quat_from_matrix(look_at(eye,center));m.cam_fovy[cam]=50
        mujoco.mj_forward(m,d);events=spend_render(budget,pool,dict(kind='auxiliary',view=name))
        with Renderer(m) as renderer:rgb=renderer.rgb(d)
        path=folder/(name+'.png');save_rgb(path,rgb);paths.append(str(path));finish_render(budget,events,dict(rgb=1,view=name))
    write(folder/'normal_evidence.json',dict(finite=bool(np.isfinite(m.mesh_normal).all()),zero_normals=int((np.linalg.norm(m.mesh_normal,axis=1)<1e-8).sum()),normal_count=len(m.mesh_normal),neutral_illumination=dict(ambient=.22,diffuse=.45,specular=0),purpose='Independent diagnostic camera/light copies; outer profile versus material/illumination, not measured real normals'))
    return paths


def terminal_pass(st,thresholds,finite=True,warnings=False,free=True):
    return bool(finite and not warnings and free and st['translation_final_m']<=thresholds['translation_m_max'] and st['rotation_final_deg']<=thresholds['rotation_deg_max'] and st['penetration_final_m']<=thresholds['penetration_m_max'])


def rest_from_disk(xml,folder,config,budget=None,video=True):
    start=time.monotonic();xml=Path(xml);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    m,d=load_state(xml)
    if m.nflex<1:raise ValueError('Rest acceptance requires a dynamic scene with native flex state')
    thresholds=config['rest_acceptance'];duration=float(config['diagnostic']['duration_s']);fps=int(config['diagnostic']['fps'])
    if duration!=1. or thresholds.get('duration_s',1.)!=1.:raise ValueError('Predeclared rest duration is exactly one second')
    native_steps=round(duration/m.opt.timestep)
    if abs(native_steps*m.opt.timestep-duration)>1e-10:raise ValueError('Duration must be an exact native-step multiple')
    initial_time=float(d.time);desc,groups=members_and_groups(m);roots={k:m.body(k+'_root').id for k in ('apple','plate')}
    reference={k:(d.xpos[b].copy(),d.xquat[b].copy()) for k,b in roots.items()};flex0=d.flexvert_xpos.copy();initial_qvel=d.qvel.copy()
    stats={k:dict(translation_max_m=0.,rotation_max_deg=0.,penetration_max_m=0.,translation_final_m=0.,rotation_final_deg=0.,penetration_final_m=0.,initial_penetration_m=0.,has_freejoint=bool(m.body_jntnum[b]),source_inherited_mass_kg=float(m.body_mass[list(desc[k])].sum())) for k,b in roots.items()}
    warnings0=np.array([w.number for w in d.warning]);finite=True;timeline=[];frames=[];contact_records=[]
    def sample(record=False,initial=False):
        distances=d.contact.dist.copy();geoms=d.contact.geom.copy()
        rows={}
        for label,b in roots.items():
            p,q=reference[label];st=stats[label];translation=float(np.linalg.norm(d.xpos[b]-p));rotation=float(np.degrees(2*np.arccos(np.clip(abs(d.xquat[b]@q),0,1))))
            relevant=np.any(np.isin(geoms,groups[label]),axis=1);penetration=float(max(0.,-distances[relevant].min())) if relevant.any() else 0.
            st['translation_max_m']=max(st['translation_max_m'],translation);st['rotation_max_deg']=max(st['rotation_max_deg'],rotation);st['penetration_max_m']=max(st['penetration_max_m'],penetration)
            st.update(translation_final_m=translation,rotation_final_deg=rotation,penetration_final_m=penetration)
            if initial:st['initial_penetration_m']=penetration
            if record:
                j=m.joint(label+'_freejoint').id;a=int(m.jnt_dofadr[j]);vel=d.qvel[a:a+6]
                rows[label]=dict(xyz_m=d.xpos[b].tolist(),quaternion_wxyz=d.xquat[b].tolist(),translation_m=translation,rotation_deg=rotation,penetration_m=penetration,linear_speed_m_s=float(np.linalg.norm(vel[:3])),angular_speed_rad_s=float(np.linalg.norm(vel[3:])),contacts=int(relevant.sum()))
        if record:
            contact_rows=[]
            for i,c in enumerate(d.contact):
                labels=[k for k in roots if np.any(np.isin(c.geom,groups[k]))]
                if not labels:continue
                force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
                contact_rows.append(dict(objects=labels,geom_ids=c.geom.tolist(),geom_names=[m.geom(int(g)).name if g>=0 else None for g in c.geom],flex_ids=c.flex.tolist(),flex_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_FLEX,int(f)) if f>=0 else None for f in c.flex],distance_m=float(c.dist),force_contact_frame=force.tolist(),position_m=c.pos.tolist()))
            delta=d.flexvert_xpos-flex0
            timeline.append(dict(time_s=float(d.time-initial_time),objects=rows,soft_displacement_max_m=float(np.linalg.norm(delta,axis=1).max()),soft_vertical_min_m=float(delta[:,2].min()),soft_vertical_max_m=float(delta[:,2].max()),qvel_max=float(np.abs(d.qvel).max()),contact_count=int(d.ncon)))
            contact_records.append(dict(time_s=float(d.time-initial_time),contacts=contact_rows))
    sample(True,True)
    renderer=Renderer(m,640,480) if video else None
    def frame_image():
        events=spend_render(budget,'render_diagnostic',dict(kind='rest_video',frame=len(frames)))
        frames.append(renderer.rgb(d));finish_render(budget,events,dict(rgb=1,view='rest_ego'))
    try:
        if video:frame_image()
        next_frame=1
        for step in range(1,native_steps+1):
            mujoco.mj_step(m,d)
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():finite=False;break
            sample()
            if step==round(next_frame*duration*native_steps/(fps*duration)) or step==native_steps:
                mujoco.mj_forward(m,d);sample(True)
                if video and next_frame<round(fps*duration):frame_image()
                next_frame+=1
        mujoco.mj_forward(m,d);sample()
        if video:
            events=spend_render(budget,'render_diagnostic',dict(kind='rest_terminal_still'))
            save_rgb(folder/'rest_terminal.png',renderer.rgb(d));finish_render(budget,events,dict(rgb=1,view='rest_terminal'))
    finally:
        if renderer:renderer.renderer.close()
    warning_delta=np.array([w.number for w in d.warning])-warnings0
    for label,st in stats.items():
        st['passed']=terminal_pass(st,thresholds,finite,bool(warning_delta.any()),st['has_freejoint'])
        tail=[row['objects'][label] for row in timeline if row['time_s']>=.8]
        st['last_200ms_linear_speed_max_m_s']=max((x['linear_speed_m_s'] for x in tail),default=None)
        st['last_200ms_angular_speed_max_rad_s']=max((x['angular_speed_rad_s'] for x in tail),default=None)
        st['legacy_peak_passed']=bool(st['translation_max_m']<=.005 and st['rotation_max_deg']<=5. and st['penetration_max_m']<=.001)
    info=video_writer(folder/'settle_diagnostic.mp4',frames,fps) if video else None
    result=dict(version='terminal_rest_v4',objects=stats,physics_passed=bool(finite and not warning_delta.any() and all(x['passed'] for x in stats.values()) and abs(d.time-initial_time-1.)<1e-9),numerically_finite=finite,warning_counts=warning_delta.tolist(),duration_s=float(d.time-initial_time),steps=step,native_timestep_s=float(m.opt.timestep),reference='Exact loaded XML/state before first integration; no shifted reference or selected terminal time',thresholds=thresholds,initial_velocity_policy='Preserve saved initial qvel exactly; never reset, damp artificially or pin objects during integration',loaded_qvel_max=float(np.abs(initial_qvel).max()),dynamic_flex_count=m.nflex,soft_geometry='Native flex and exact saved node qpos/qvel, not the frozen display mesh',xml_sha256=sha(xml),state_sha256=sha(xml.with_suffix('.state.npz')),video=info,wall_seconds=time.monotonic()-start,real_dynamics_validated=False)
    write(folder/'physical_diagnostic.json',result);write(folder/'diagnostic_timeline.json',timeline);write(folder/'contact_timeline.json',contact_records)
    np.savez_compressed(folder/'diagnostic_final_state.npz',qpos=d.qpos,qvel=d.qvel,time=np.array(d.time),flexvert_world=d.flexvert_xpos)
    if budget:budget.record('rest_diagnostic',dict(xml_sha256=sha(xml),duration_s=duration,steps=step,video=video),result['wall_seconds'])
    return result
