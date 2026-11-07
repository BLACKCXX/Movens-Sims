"""Explicit world transforms, frozen display surface, and bounded scene parameters."""
from __future__ import annotations
import copy
import importlib.util
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .util import ROOT, read, save_rgb, sha, write

def numbers(x): return ' '.join(f'{float(v):.17g}' for v in x)
def quat_from_matrix(x):
    q=np.zeros(4);mujoco.mju_mat2Quat(q,np.asarray(x).ravel());return q
def matrix_from_quat(x):
    a=np.zeros(9);mujoco.mju_quat2Mat(a,np.asarray(x));return a.reshape(3,3)
def world_summary(model,data):
    result={}
    for name in ('apple_root','plate_root','table_root'):
        b=model.body(name).id;result[name]=dict(xyz_m=data.xpos[b].copy(),quaternion_wxyz=data.xquat[b].copy())
    cam=model.camera('approx_ego_camera').id
    result['camera']=dict(xyz_m=data.cam_xpos[cam].copy(),quaternion_wxyz=quat_from_matrix(data.cam_xmat[cam]),rotation_camera_to_world=data.cam_xmat[cam].reshape(3,3).copy(),fovy_deg=float(model.cam_fovy[cam]),convention='MuJoCo camera local -Z forward, +Y up; quaternion wxyz; world meters')
    return result

def prune_assets(root):
    asset=root.find('asset')
    if asset is None:return
    used={k:set() for k in ('mesh','material','texture','hfield')}
    for e in root.iter():
        if e in list(asset):continue
        for k in used:
            if e.get(k):used[k].add(e.get(k))
    for e in asset:
        if e.tag=='material' and e.get('name') in used['material']:
            if e.get('texture'):used['texture'].add(e.get('texture'))
            for child in e:
                if child.get('texture'):used['texture'].add(child.get('texture'))
    for e in list(asset):
        if e.tag in used and e.get('name') not in used[e.tag]:asset.remove(e)

def prepare_scene(imported, run):
    imported,run=Path(imported),Path(run);run.mkdir(parents=True,exist_ok=True)
    source=imported/'source';out=run/'scene';out.mkdir(exist_ok=True)
    # Use the actual loader, snapshotted verbatim, without robot trajectory playback.
    spec=importlib.util.spec_from_file_location('_upstream_scene_loader',imported/'evidence/simulation__common.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    robot=read(source/'robot.json');reset=copy.deepcopy(robot['reset_protocol']);reset['settle_steps']=0
    sim=mod.AppleSimulation(str(source/'scene.json'),reset_protocol=reset)
    model,data=sim.model,sim.data
    default=mujoco.MjData(model);mujoco.mj_forward(model,default)
    default_summary=world_summary(model,default);reset_summary=world_summary(model,data)
    with np.load(imported/'source_initial_state.npz') as z:
        data.qpos[:]=z['qpos'];data.qvel[:]=z['qvel'];data.ctrl[:]=z['ctrl'];mujoco.mj_forward(model,data)
        selected_index=int(z['source_state_index']);selected_time=float(z['source_timestamp_s'])
    selected=world_summary(model,data)
    write(run/'initial_transforms.json',dict(xml_default=default_summary,complete_reset_before_settle=reset_summary,source_frame_zero_after_settle=selected,source_state_index=selected_index,source_time_s=selected_time,source_settle_s=read(imported/'source_run_report.json')['settle_duration_s'],chosen='source_frame_zero_after_settle',rationale='Matches the saved frame map for real/source frame0 and old simulation video frame0. The original loader reset is separately audited, not silently mixed with XML defaults.'))
    np.savez_compressed(run/'source_world_state.npz',qpos=data.qpos,qvel=data.qvel,ctrl=data.ctrl,flexvert_world=data.flexvert_xpos,flex_elements=model.flex_elem)
    from .render import Renderer
    with Renderer(model,640,480) as renderer:
        save_rgb(run/'baseline_with_robot.png',renderer.rgb(data))
    # Canonicalize includes with MjSpec, then remove robot-only references from the environment.
    mj_spec=mujoco.MjSpec.from_file(str(source/'scene.xml'))
    spec_flex_issue=None
    try:
        mj_spec.compile();root=ET.fromstring(mj_spec.to_xml())
    except ValueError as exc:
        # MuJoCo 3.7 MjSpec loses passive flex bookkeeping in this source model.
        # Preserve the XML and use the verified direct compiler for the physical copy.
        spec_flex_issue=str(exc)
        root=ET.parse(source/'scene.xml').getroot()
        if root.findall('.//include'):raise ValueError('Flex MjSpec fallback with nested includes needs explicit flattening')
    # Rebase assets because the exported model lives in a new independent run directory.
    for a in root.findall('./asset/*'):
        if a.get('file'):
            src=(source/a.get('file')).resolve();dest=out/'assets'/src.name;dest.parent.mkdir(exist_ok=True)
            shutil.copy2(src,dest);a.set('file','assets/'+dest.name)
    world=root.find('worldbody');pelvis=world.find("body[@name='pelvis']")
    removed_names={e.get('name') for e in pelvis.iter() if e.get('name')};world.remove(pelvis)
    for tag in ('actuator','sensor','tendon','keyframe'):
        node=root.find(tag)
        if node is not None:root.remove(node)
    deleted=[]
    for tag in ('contact','equality'):
        section=root.find(tag)
        if section is None:continue
        for e in list(section):
            if any(v in removed_names for v in e.attrib.values()):deleted.append(dict(tag=e.tag,attributes=e.attrib.copy()));section.remove(e)
    cam=selected['camera'];ET.SubElement(world,'camera',name='approx_ego_camera',pos=numbers(cam['xyz_m']),quat=numbers(cam['quaternion_wxyz']),fovy=str(cam['fovy_deg']))
    # Map every remaining joint by name; never assume compiled ordering survives deletion.
    for elem in root.findall('.//joint'):
        name=elem.get('name')
        if not name:continue
        try:j=model.joint(name).id
        except KeyError:continue
        if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE:
            body=next((b for b in world.findall('body') if elem in list(b)),None)
            if body is not None:
                # qpos retained in state, not baked twice into node body positions.
                pass
    apple=world.find("body[@name='apple_root']");apple.set('pos',numbers(selected['apple_root']['xyz_m']));apple.set('quat',numbers(selected['apple_root']['quaternion_wxyz']))
    for i,g in enumerate(world.find("body[@name='table_root']").findall('.//geom')):
        if not g.get('name'):g.set('name',f'movens_table_visual_{i}')
    # The real frame shows a cloth-covered tabletop beyond the small source contact patch.
    # Continue its visible surface to the existing tabletop bounds without rescaling the table
    # or duplicating the deformed contact patch. This visual-only ring is explicitly untested
    # outside the source foundation; apple/plate remain inside the original physical patch.
    table_bodies={model.body('table_root').id}
    for b in range(model.nbody):
        if int(model.body_parentid[b]) in table_bodies:table_bodies.add(b)
    boxes=[g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in table_bodies and model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX]
    topbox=max(boxes,key=lambda g:np.prod(sorted(model.geom_size[g])[-2:]))
    corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*model.geom_size[topbox]
    world_corners=corners@data.geom_xmat[topbox].reshape(3,3).T+data.geom_xpos[topbox]
    outer_lo=world_corners[:,:2].min(axis=0);outer_hi=world_corners[:,:2].max(axis=0)
    inner_lo=data.flexvert_xpos[:,:2].min(axis=0);inner_hi=data.flexvert_xpos[:,:2].max(axis=0)
    z=float(np.median(data.flexvert_xpos[:,2]));ring=out/'assets/cloth_continuation.obj'
    def rectangle(lo,hi):return [[lo[0],lo[1],z],[hi[0],lo[1],z],[hi[0],hi[1],z],[lo[0],hi[1],z]]
    with ring.open('w') as f:
        vertices=rectangle(outer_lo,outer_hi)+rectangle(inner_lo,inner_hi)
        for v in vertices+[np.array(v)-[0,0,.0001] for v in vertices]:f.write('v '+numbers(v)+'\n')
        for i in range(4):
            j=(i+1)%4
            for face in [(i,j,j+4),(i,j+4,i+4),(i+8,j+12,j+8),(i+8,i+12,j+12),(i,i+8,j+8),(i,j+8,j),(i+4,j+4,j+12),(i+4,j+12,i+12)]:f.write('f '+' '.join(str(x+1) for x in face)+'\n')
    ET.SubElement(root.find('asset'),'mesh',name='cloth_continuation',file='assets/cloth_continuation.obj')
    ET.SubElement(world,'geom',name='cloth_continuation',type='mesh',mesh='cloth_continuation',contype='0',conaffinity='0',group='1',rgba=numbers(model.flex_rgba[1]))
    prune_assets(root)
    physics_path=out/'scene_physics_baseline.xml';ET.ElementTree(root).write(physics_path,encoding='utf-8',xml_declaration=True)
    pm=mujoco.MjModel.from_xml_path(str(physics_path));pd=mujoco.MjData(pm)
    mapping=[]
    for j in range(pm.njnt):
        name=mujoco.mj_id2name(pm,mujoco.mjtObj.mjOBJ_JOINT,j);old=model.joint(name).id
        n=7 if pm.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE else 1;v=6 if n==7 else 1
        pd.qpos[pm.jnt_qposadr[j]:pm.jnt_qposadr[j]+n]=data.qpos[model.jnt_qposadr[old]:model.jnt_qposadr[old]+n]
        pd.qvel[pm.jnt_dofadr[j]:pm.jnt_dofadr[j]+v]=data.qvel[model.jnt_dofadr[old]:model.jnt_dofadr[old]+v]
        mapping.append(dict(name=name,source_joint=old,environment_joint=j))
    mujoco.mj_forward(pm,pd)
    np.savez_compressed(out/'physics_initial_state.npz',qpos=pd.qpos,qvel=pd.qvel)
    # Static display freezes the *observed* deforming surface as independent mesh geometry.
    display=copy.deepcopy(root);dw=display.find('worldbody');asset=display.find('asset')
    for body in list(dw):
        if body.tag=='body' and body.get('name','').startswith('pad_node_'):dw.remove(body)
    deformable=display.find('deformable')
    surface_mapping=[]
    if deformable is not None:
        for i in range(model.nflex):
            if model.flex_group[i]==5:continue  # Hidden coincident contact layer, not a second visible surface.
            start=int(model.flex_vertadr[i]);count=int(model.flex_vertnum[i]);v=data.flexvert_xpos[start:start+count]
            es=int(model.flex_elemdataadr[i]);en=int(model.flex_elemnum[i]);faces=model.flex_elem[es:es+en*3].reshape(-1,3)
            name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_FLEX,i);meshfile=out/'assets'/f'frozen_{name}.obj'
            with meshfile.open('w') as f:
                for vertex in v:f.write('v '+numbers(vertex)+'\n')
                for face in faces:f.write('f '+' '.join(str(int(k)+1) for k in face)+'\n')
            ET.SubElement(asset,'mesh',name='frozen_'+name,file='assets/'+meshfile.name)
            ET.SubElement(dw,'geom',name='frozen_'+name,type='mesh',mesh='frozen_'+name,contype='0',conaffinity='0',group='1',rgba=numbers(model.flex_rgba[i]))
            surface_mapping.append(dict(flex_name=name,mesh=meshfile.name,source_vertex_range=[start,start+count],triangles=en,sha256=sha(meshfile),max_world_vertex_error_m=0.0))
        display.remove(deformable)
    # Keep only environment assets after MjSpec compiles out all includes.
    prune_assets(display)
    for e in display.findall('.//geom'):
        if e.get('name','').startswith('frozen_'):continue
    ET.ElementTree(display).write(out/'scene_fitting.xml',encoding='utf-8',xml_declaration=True)
    baseline=copy.deepcopy(display);bw=baseline.find('worldbody');bw.remove(bw.find("geom[@name='cloth_continuation']"));prune_assets(baseline)
    ET.ElementTree(baseline).write(out/'scene_baseline.xml',encoding='utf-8',xml_declaration=True)
    dm=mujoco.MjSpec.from_file(str(out/'scene_baseline.xml')).compile();dd=mujoco.MjData(dm);mujoco.mj_forward(dm,dd)
    camera_error=float(np.max(np.abs(dd.cam_xpos[0]-data.cam_xpos[model.camera('approx_ego_camera').id])))
    rotation_error=float(np.max(np.abs(dd.cam_xmat[0]-data.cam_xmat[model.camera('approx_ego_camera').id])))
    with Renderer(dm,640,480) as renderer:save_rgb(run/'baseline_ego.png',renderer.rgb(dd))
    meshes=[]
    for a in root.findall('./asset/mesh'):
        meshes.append(dict(name=a.get('name'),file=a.get('file'),scale=a.get('scale','1 1 1'),uses=[dict(name=g.get('name'),body_group=g.get('group'),class_name=g.get('class')) for g in root.findall(f".//geom[@mesh='{a.get('name')}']")]))
    write(run/'resource_checks.json',dict(mujoco_version=mujoco.__version__,source_compile=True,environment_compile=True,display_compile=True,mjspec_source_flex_compatibility_issue=spec_flex_issue,physical_compiler='MjModel.from_xml_path; source XML retained without lossy flex export',display_compiler='MjSpec.compile',source_nflex=model.nflex,environment_nflex=pm.nflex,display_nflex=dm.nflex,removed_robot_references=deleted,camera_world_position_error_m=camera_error,camera_world_rotation_matrix_error=rotation_error,mesh_scale_registry=meshes,scale_import_multiplier=1,additional_table_height_offset_m=0,plate_fixed_in_source=True,display_surface_mapping=surface_mapping,joint_state_mapping=mapping,static_surface_is_physics_validation=False))
    write(out/'scene_context.json',dict(table_top_z_m=float(read(source/'scene.json')['table_top_z']),apple_baseline=selected['apple_root'],plate_baseline=selected['plate_root'],camera_baseline=cam,nominal_scale_anchor='Existing plate mesh, diameter approximately 0.182m; nominal, not measured',initial_velocity_source='Saved frame0 post-settle velocities, preserved in dynamic diagnostic',soft_surface='Saved post-settle vertices; full original foundation and node state retained in physical copy',cloth_continuation=dict(method='Visual-only mesh ring from original patch to existing table bounds; no source image texture or billboard',outer_xy_m=[outer_lo.tolist(),outer_hi.tolist()],inner_xy_m=[inner_lo.tolist(),inner_hi.tolist()],new_support_dynamics_validated=False)))
    return out

def look_at(eye,target,up=np.array([0.,0.,1.])):
    z=np.asarray(eye)-target;z=z/np.linalg.norm(z);x=np.cross(up,z);x/=np.linalg.norm(x);y=np.cross(z,x)
    return np.column_stack((x,y,z))

class Scene:
    def __init__(self, path):
        self.path=Path(path);self.context=read(self.path.parent/'scene_context.json')
        self.model=mujoco.MjSpec.from_file(str(path)).compile();self.data=mujoco.MjData(self.model)
        self.cam=self.model.camera('approx_ego_camera').id;self.apple=self.model.body('apple_root').id;self.plate=self.model.body('plate_root').id
        self.aj=self.model.joint('apple_freejoint').id;self.aq=int(self.model.jnt_qposadr[self.aj])
        self.cam_pos=self.model.cam_pos[self.cam].copy();self.cam_rot=matrix_from_quat(self.model.cam_quat[self.cam]);self.apple_pos=self.data.qpos[self.aq:self.aq+3].copy();self.apple_quat=self.data.qpos[self.aq+3:self.aq+7].copy();self.plate_pos=self.model.body_pos[self.plate].copy()
        self.descendants={}
        for label,rootid in [('apple',self.apple),('plate',self.plate),('table',self.model.body('table_root').id)]:
            bodies={rootid}
            for b in range(self.model.nbody):
                if int(self.model.body_parentid[b]) in bodies:bodies.add(b)
            self.descendants[label]=bodies
        self.geom_sets={k:np.where(np.isin(self.model.geom_bodyid,list(v)))[0] for k,v in self.descendants.items()}
        self.appearance_rgba=self.model.geom_rgba.copy();self.flex_ids=[i for i in range(self.model.ngeom) if (mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith('frozen_') or (mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,i) or '')=='cloth_continuation']
        self.geom_sets['cloth']=np.array(self.flex_ids,dtype=int)
        self.plate_materials=sorted({int(self.model.geom_matid[g]) for g in self.geom_sets['plate'] if self.model.geom_group[g]==1 and self.model.geom_matid[g]>=0})
        self.mesh_vertices=self.model.mesh_vert.copy();self.geom_pos=self.model.geom_pos.copy();self.mesh_bounds=self.model.mesh_aabb.copy() if hasattr(self.model,'mesh_aabb') else None
        self.apple_meshes=sorted(set(int(self.model.geom_dataid[g]) for g in self.geom_sets['apple'] if self.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH))
        mujoco.mj_forward(self.model,self.data)
        self.depth_geoms={};self.depth_anchor=self.context.get('cloth_continuation',{}).get('outer_xy_m',[[.229,0]])[0][0]
        for g in list(self.geom_sets['table'])+self.flex_ids:
            name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
            if self.model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH or name.startswith('frozen_'):continue
            mi=int(self.model.geom_dataid[g]);a=int(self.model.mesh_vertadr[mi]);n=int(self.model.mesh_vertnum[mi]);rot=self.data.geom_xmat[g].reshape(3,3).copy();pos=self.data.geom_xpos[g].copy();world=self.mesh_vertices[a:a+n]@rot.T+pos
            self.depth_geoms[g]=dict(mesh=mi,adr=a,count=n,rotation=rot,position=pos,world=world,ring=name=='cloth_continuation')
        self.support_bottom_z=float(self.vertices('apple',collision=True)[:,2].min())
        self.params={x['id']:x['initial'] for x in self.registry()};self.apply(self.params)

    def registry(self):
        out=[]
        def add(id,unit,initial,lo,hi,stage,target,meaning,update):out.append(dict(id=id,unit=unit,initial=float(initial),lower=float(lo),upper=float(hi),stage=stage,target=target,coordinate_frame='world XYZ' if unit=='m' else 'specified below',meaning=meaning,update_method=update))
        for i,a in enumerate('xyz'):add('camera.'+a,'m',self.cam_pos[i],self.cam_pos[i]-.65,self.cam_pos[i]+.65,'camera','ego_camera','Absolute camera world position','model.cam_pos')
        for a in 'xyz':add('camera.r'+a,'deg',0,-85,85,'camera','ego_camera','Extrinsic world-axis rotation vector component relative to baseline; scipy Rotation.from_rotvec(degrees=True), not Euler','R_delta @ R_baseline')
        add('camera.fovy','deg',self.model.cam_fovy[self.cam],35,95,'camera','ego_camera','Vertical pinhole field of view; image center fixed','model.cam_fovy')
        for i,a in enumerate('xy'):add('apple.'+a,'m',self.apple_pos[i],self.apple_pos[i]-.22,self.apple_pos[i]+.22,'layout','apple','Absolute body world translation; Z rederived from collision vertices and source support','freejoint.qpos')
        for a in 'xyz':add('apple.r'+a,'deg',0,-55,55,'layout','apple','World rotation vector relative to recorded pose','R_delta @ R_baseline')
        # Plate remains the world and scale anchor; layout ambiguity is not hidden by moving all objects.
        add('apple.scale','ratio',1,.85,1.15,'size','apple','Uniform extra scale relative to already baked source asset; mass retained and inertia scaled by s² on export','all apple visual/collision meshes, local positions and inertial offset')
        add('table.depth_scale','ratio',1,1,2.5,'size','table','World X tabletop depth scale anchored at its near edge, existing height and width fixed. Minimum extent explaining cloth background; unseen dimensions remain ambiguous. Original soft contact patch not stretched.','Visual and collision table vertices transformed consistently; cloth outer ring continued to the same bounds')
        add('table.width_scale','ratio',1,1,1.6,'size','table','World Y tabletop width scale about world Y=0; bounded nominal extent explaining cloth coverage, not a measured size','All table visible/collision geometry; outer cloth corners only')
        add('table.front_extension','m',0,0,.15,'size','table','Extend the near edge toward negative world X; far edge unchanged; soft patch stays at its original world coordinates','Affine table vertices and outer cloth ring corners')
        add('cloth.gray','linear_rgb',.36,.06,.60,'appearance','cloth','Uniform gray albedo; does not explain real wrinkles','frozen geom rgba and physical flex rgba')
        add('light.ambient','linear_rgb',.35,.15,.65,'appearance','headlight','Ambient intensity shared RGB','model.vis.headlight.ambient')
        add('plate.albedo','linear_rgb',.8,.45,.9,'appearance','plate','Neutral plate material albedo after geometry stabilized','Shared plate visual material RGB; collision geometry unchanged')
        return out

    def validate(self, updates, stage=None):
        reg={p['id']:p for p in self.registry()}
        if len(updates)!=len(set(updates)):raise ValueError('Duplicate parameter ID')
        for k,v in updates.items():
            if k not in reg:raise ValueError('Unknown parameter '+k)
            p=reg[k]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v):raise ValueError('Non-finite/non-numeric '+k)
            if not p['lower']<=v<=p['upper']:raise ValueError('Out of bounds '+k)
            if stage is not None and p['stage']!=stage:raise ValueError('Wrong stage '+k)

    def apply(self, values):
        self.validate(values);self.params.update(values);p=self.params;m=self.model;d=self.data
        m.cam_pos[self.cam]=[p['camera.'+a] for a in 'xyz'];r=Rotation.from_rotvec([p['camera.r'+a] for a in 'xyz'],degrees=True).as_matrix()@self.cam_rot;m.cam_quat[self.cam]=quat_from_matrix(r);m.cam_fovy[self.cam]=p['camera.fovy']
        ar=Rotation.from_rotvec([p['apple.r'+a] for a in 'xyz'],degrees=True).as_matrix()@matrix_from_quat(self.apple_quat)
        d.qpos[self.aq:self.aq+3]=[p['apple.x'],p['apple.y'],self.apple_pos[2]];d.qpos[self.aq+3:self.aq+7]=quat_from_matrix(ar)
        scale=p['apple.scale']
        for mi in self.apple_meshes:
            a=int(m.mesh_vertadr[mi]);n=int(m.mesh_vertnum[mi]);m.mesh_vert[a:a+n]=self.mesh_vertices[a:a+n]*scale
        m.geom_pos[self.geom_sets['apple']]=self.geom_pos[self.geom_sets['apple']]*scale
        # Disable compile-time same-frame shortcuts when changing a mesh offset.
        # Otherwise auto-inertia models can silently keep using the old inertial offset.
        m.geom_sameframe[self.geom_sets['apple']]=0
        for info in self.depth_geoms.values():
            w=info['world'].copy();inner=self.context['cloth_continuation']['inner_xy_m']
            sel=((w[:,0]>inner[1][0]+.01)|(w[:,1]<inner[0][1]-.01)|(w[:,1]>inner[1][1]+.01)) if info['ring'] else np.ones(len(w),bool)
            nominal_depth=self.context['cloth_continuation']['outer_xy_m'][1][0]-self.depth_anchor
            w[sel,0]=self.depth_anchor-p['table.front_extension']+(w[sel,0]-self.depth_anchor)*(p['table.depth_scale']+p['table.front_extension']/nominal_depth)
            w[sel,1]*=p['table.width_scale']
            a=info['adr'];n=info['count'];m.mesh_vert[a:a+n]=(w-info['position'])@info['rotation']
        # Consistent support from the actual collision vertices; no per-step pose correction.
        mujoco.mj_forward(m,d)
        pts=self.vertices('apple',collision=True);bottom=float(pts[:,2].min());desired=self.support_bottom_z
        d.qpos[self.aq+2]+=desired-bottom
        if 'cloth.gray' in values:
            for g in self.flex_ids:m.geom_rgba[g]=[p['cloth.gray']]*3+[1]
        m.vis.headlight.ambient[:]=p['light.ambient']
        for mat in self.plate_materials:m.mat_rgba[mat,:3]=p['plate.albedo']
        mujoco.mj_forward(m,d)

    def vertices(self,label,collision=False):
        pts=[];m=self.model;d=self.data
        for g in self.geom_sets[label]:
            if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
            if collision and m.geom_contype[g]==0 and m.geom_conaffinity[g]==0:continue
            if not collision and m.geom_group[g]!=1:continue
            mi=int(m.geom_dataid[g]);a=int(m.mesh_vertadr[mi]);n=int(m.mesh_vertnum[mi]);v=m.mesh_vert[a:a+n]
            pts.append(v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
        return np.concatenate(pts) if pts else np.empty((0,3))

    def export(self, name='scene_optimized.xml', physics=False):
        output_filename=name
        source=self.path.parent/('scene_physics_baseline.xml' if physics else 'scene_fitting.xml')
        root=ET.parse(source);p=self.params;cam=root.find(".//camera[@name='approx_ego_camera']");cam.set('pos',numbers(self.model.cam_pos[self.cam]));cam.set('quat',numbers(self.model.cam_quat[self.cam]));cam.set('fovy',str(p['camera.fovy']))
        apple=root.find(".//body[@name='apple_root']");apple.set('pos',numbers(self.data.qpos[self.aq:self.aq+3]));apple.set('quat',numbers(self.data.qpos[self.aq+3:self.aq+7]))
        for e in apple.findall('.//inertial'):
            s=p['apple.scale']
            if e.get('pos'):e.set('pos',numbers(np.fromstring(e.get('pos'),sep=' ')*s))
            for k in ('fullinertia','diaginertia'):
                if e.get(k):e.set(k,numbers(np.fromstring(e.get(k),sep=' ')*s*s))
        used={e.get('mesh') for e in apple.findall('.//geom') if e.get('mesh')}
        # Refuse changing a shared mesh used outside apple: clone before that operation.
        for e in root.findall('./asset/mesh'):
            if e.get('name') in used:
                other=[g for g in root.findall(f".//geom[@mesh='{e.get('name')}']") if g not in apple.findall('.//geom')]
                if other:raise ValueError('Cannot scale shared mesh without cloning')
                e.set('scale',numbers(np.fromstring(e.get('scale','1 1 1'),sep=' ')*p['apple.scale']))
        # Export the same table deformation to both visible and collision geometry.
        # Boxes become exact convex vertex meshes after the world-axis affine map.
        for elem in root.findall(".//body[@name='table_root']//geom")+[g for g in root.findall('./worldbody/geom') if g.get('name')=='cloth_continuation']:
            name=elem.get('name');g=self.model.geom(name).id
            if self.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
                mi=int(self.model.geom_dataid[g]);a=int(self.model.mesh_vertadr[mi]);n=int(self.model.mesh_vertnum[mi]);v=self.model.mesh_vert[a:a+n]@self.data.geom_xmat[g].reshape(3,3).T+self.data.geom_xpos[g]
                a=int(self.model.mesh_faceadr[mi]);n=int(self.model.mesh_facenum[mi]);faces=self.model.mesh_face[a:a+n].copy()
            elif self.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:
                from scipy.spatial import ConvexHull
                v=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*self.model.geom_size[g];v=v@self.data.geom_xmat[g].reshape(3,3).T+self.data.geom_xpos[g]
                nominal_depth=self.context['cloth_continuation']['outer_xy_m'][1][0]-self.depth_anchor
                v[:,0]=self.depth_anchor-p['table.front_extension']+(v[:,0]-self.depth_anchor)*(p['table.depth_scale']+p['table.front_extension']/nominal_depth);v[:,1]*=p['table.width_scale'];faces=ConvexHull(v).simplices
                for face in faces:
                    if np.dot(np.cross(v[face[1]]-v[face[0]],v[face[2]]-v[face[0]]),v[face].mean(axis=0)-v.mean(axis=0))<0:face[[1,2]]=face[[2,1]]
            else:continue
            body=int(self.model.geom_bodyid[g]);v=(v-self.data.xpos[body])@self.data.xmat[body].reshape(3,3)
            filename=f'depth_{name}.obj';asset_name=f'depth_{name}';dest=self.path.parent/'assets'/filename
            with dest.open('w') as f:
                for point in v:f.write('v '+numbers(point)+'\n')
                for face in faces:f.write('f '+' '.join(str(int(k)+1) for k in face)+'\n')
            ET.SubElement(root.find('asset'),'mesh',name=asset_name,file='assets/'+filename)
            for k in ('size','quat','euler','axisangle','xyaxes','zaxis','fromto'):elem.attrib.pop(k,None)
            elem.set('pos','0 0 0');elem.set('type','mesh');elem.set('mesh',asset_name)
        if 'cloth.gray' in self.params:
            for e in root.findall('.//geom'):
                if e.get('name','').startswith('frozen_') or e.get('name')=='cloth_continuation':e.set('rgba',numbers([p['cloth.gray']]*3+[1]))
            for e in root.findall('./deformable/flex'):e.set('rgba',numbers([p['cloth.gray']]*3+[1]))
        head=root.find('./visual/headlight')
        if head is not None:head.set('ambient',numbers([p['light.ambient']]*3))
        for e in root.findall('./asset/material'):
            if e.get('name') in [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_MATERIAL,i) for i in self.plate_materials]:e.set('rgba',numbers([p['plate.albedo']]*3+[1]))
        out=self.path.parent/output_filename;root.write(out,encoding='utf-8',xml_declaration=True)
        # MjSpec parse/compile is the final model-editing check; preserve source attributes in saved XML.
        compiled=mujoco.MjModel.from_xml_path(str(out)) if physics else mujoco.MjSpec.from_file(str(out)).compile();data=mujoco.MjData(compiled)
        if physics:
            with np.load(self.path.parent/'physics_initial_state.npz') as z:
                data.qpos[:]=z['qpos'];data.qvel[:]=z['qvel']
            j=compiled.joint('apple_freejoint').id;q=compiled.jnt_qposadr[j];data.qpos[q:q+7]=self.data.qpos[self.aq:self.aq+7]
        mujoco.mj_forward(compiled,data)
        np.savez_compressed(out.with_suffix('.state.npz'),qpos=data.qpos,qvel=data.qvel,time=np.array(0.))
        write(out.with_suffix('.state.json'),dict(xml_sha256=sha(out),state_sha256=sha(out.with_suffix('.state.npz')),params=self.params,quaternion_order='wxyz',world_units='m',static_display=not physics,physical_parameters='Inherited, unmeasured; no mass/friction/contact fitting; apple extra uniform scale changes inertia s² at constant mass',source_context=self.context))
        return out
