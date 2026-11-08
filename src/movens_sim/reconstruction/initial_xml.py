"""Original XML/FK/frame0 ingestion for the existing no-model workflow."""
from __future__ import annotations
import copy
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import cv2
import mujoco
import numpy as np
from .budget import BudgetManager
from .full_prepare import prepare_full
from .v5_material import defaults
from ..ingest import resource_closure, portable_copy
from ..render import Renderer
from ..util import ROOT, CODE_ROOT, read, write, sha, digest, save_rgb, now


def task_budget(c, ep):
    return BudgetManager(ROOT/'runs/_task_budgets'/(c['task_group_id']+'.sqlite'),c['task_group_id']+f'/episode_{ep:06}',c['budget'])


def validate_entry(entry, c):
    required=('episode','scene_dir','estimate','real_video','parquet','h0_m','source_files','initial_state')
    missing=[k for k in required if k not in entry]
    if missing:raise ValueError('Initial XML entry missing: '+', '.join(missing))
    source=Path(entry['scene_dir']).resolve();project=Path(c['source_project']).resolve()
    if not source.is_relative_to(project) or source.parts[-3:]!=(f"episode_{entry['episode']:06}",'offset_000mm','scene'):
        raise ValueError('Initial XML must be the declared zero-offset Movens-Interact source scene')
    xml=source/'scene.xml'
    for name,h in entry['source_files'].items():
        if not Path(name).is_file() or sha(name)!=h:raise ValueError('Missing/changed original input: '+name)
    for p in (xml,Path(entry['estimate']),Path(entry['real_video']),Path(entry['parquet'])):
        if str(p) not in entry['source_files']:raise ValueError('Required input has no declared hash: '+str(p))
    root=ET.parse(xml).getroot()
    if root.findall('.//include'):raise ValueError('Initial XML contract requires the existing flattened project scene; include structure unsupported')
    for label in ('apple','plate','table'):
        if root.find(f".//body[@name='{label}_root']") is None:raise ValueError('Missing project entity: '+label+'_root')
    receipt=read(source/'batch_construction.json')
    if receipt['scene_sha256']!=sha(xml) or receipt['offset_mm']!=0 or receipt['base_offset_mm']!=0:
        raise ValueError('Original scene identity/zero-compensation mismatch')
    if abs(entry['h0_m']-receipt['original_estimated_table_top_m'])>1e-12:raise ValueError('Original FK h0 mismatch')
    if entry['initial_state']['mode'] not in ('fresh_xml_reset','joint_name_state'):raise ValueError('Unsupported initial state mode')
    if entry['initial_state']['mode']=='joint_name_state':
        spec=entry['initial_state'];p=Path(spec['path'])
        if not p.is_file() or sha(p)!=spec['sha256']:raise ValueError('Authoritative named state missing/changed')
    closure={str(p):sha(p) for xml_node,(_,refs) in resource_closure(xml).items() for p in [xml_node,*[x[2] for x in refs]]}
    init=c['initialization'];index=project/init['training_index'];formal=project/init['training_formal_config'];training=read(index);formal_config=read(formal)
    if sha(index)!=formal_config['data_provenance']['teacher_index_sha256']:raise ValueError('Last training source index identity mismatch')
    if source!=Path(training['source_batch'])/f"episode_{entry['episode']:06}"/'offset_000mm/scene':raise ValueError('Initial scene is not from the declared training source batch')
    shared=ROOT/init['data_root']/'shared';asset_records=read(shared/'assets.json');training_audit=read(shared/'training_asset_audit.json');geometry_audit=Path(training['source_batch'])/'shared/geometry_audit.json'
    if sha(geometry_audit)!=training_audit['source_geometry_audit_sha256'] or training_audit['scale_at_import']!=1.:raise ValueError('Baked training asset geometry identity mismatch')
    for name,h in closure.items():
        if name==str(xml):continue
        if name not in asset_records or asset_records[name]['sha256']!=h or sha(asset_records[name]['path'])!=h:raise ValueError('Original/shared asset closure identity mismatch: '+name)
    training_files={str(p):sha(p) for p in (index,formal,geometry_audit,shared/'assets.json',shared/'training_asset_audit.json')}
    return dict(training_source_files=training_files,training_apple=dict(scale_at_import=1.,visual_extent_m=training_audit['shape']['geometry']['visual']['extent_xyz_m'],scale_chain_already_baked=training_audit['scale_chain']),xml=str(xml),xml_sha256=sha(xml),resource_closure=closure,original_camera=[x.attrib for x in root.findall('.//camera')],initial_state=entry['initial_state'],checked_utc=now())


def named_state(model,data,path):
    qa=model.jnt_qposadr.copy();da=model.jnt_dofadr.copy()
    np.savez_compressed(path,joint_names=np.array([model.joint(i).name for i in range(model.njnt)]),qpos_addresses=qa,qpos_sizes=np.diff(np.r_[qa,model.nq]),dof_addresses=da,dof_sizes=np.diff(np.r_[da,model.nv]),qpos=data.qpos.copy(),qvel=data.qvel.copy(),time=np.array(data.time))


def apply_named_state(model,data,path,allow_missing=()):
    """Never assume joint array order survived robot removal/freejoint insertion."""
    with np.load(path) as z:
        required=('joint_names','qpos_addresses','qpos_sizes','dof_addresses','dof_sizes','qpos','qvel','time')
        if any(k not in z for k in required):raise ValueError('Named state needs qpos, qvel, time and joint address/size tables')
        names=[str(x) for x in z['joint_names']]
        if len(names)!=len(set(names)):raise ValueError('Named state contains duplicate joints')
        for i,name in enumerate(names):
            try:j=model.joint(name).id
            except KeyError:
                if name in allow_missing:continue
                raise ValueError('Named state joint missing in normalized model: '+name)
            q=int(model.jnt_qposadr[j]);v=int(model.jnt_dofadr[j]);qn=int(z['qpos_sizes'][i]);vn=int(z['dof_sizes'][i])
            actual_q=(model.jnt_qposadr[j+1] if j+1<model.njnt else model.nq)-q
            actual_v=(model.jnt_dofadr[j+1] if j+1<model.njnt else model.nv)-v
            if (qn,vn)!=(actual_q,actual_v):raise ValueError('Named state joint dimensions differ: '+name)
            a=int(z['qpos_addresses'][i]);b=int(z['dof_addresses'][i]);data.qpos[q:q+qn]=z['qpos'][a:a+qn];data.qvel[v:v+vn]=z['qvel'][b:b+vn]
        data.time=float(z['time'])
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or not np.isfinite(data.time):raise ValueError('Nonfinite named initial state')
    mujoco.mj_forward(model,data)


def initialize(entry,c,run):
    from .modes import install_initial_xml_guard
    install_initial_xml_guard(c)
    run=Path(run);marker=run/'initial_xml_prepared.json'
    if marker.exists():
        receipt=read(marker)
        if receipt['entry_sha256']!=digest(entry):raise ValueError('Initial XML entry changed on resume')
        return
    b=task_budget(c,entry['episode']);left=b.remaining('optimization_seconds')-c['v5']['final_compute_reserve_seconds']
    with b.computation(dict(kind='initial_xml_prepare'),reserve_seconds=left):
        audit=validate_entry(entry,c);write(run/'source_xml_identity.json',audit)
        # The full_prepare side is checkpointed separately from material/state completion.
        if not (run/'raw_prepare_done.json').exists():
            prepare_full(entry,c,run,budget=b)
            write(run/'raw_prepare_done.json',dict(entry_sha256=digest(entry),utc=now()))
        physics=run/'inputs/template_physics/scene.xml';m=mujoco.MjModel.from_xml_path(str(physics));d=mujoco.MjData(m)
        state_spec=entry['initial_state']
        if state_spec['mode']=='joint_name_state':
            # Normalize only the named environment state. Robot joints are deliberately removed.
            source_model=mujoco.MjModel.from_xml_path(audit['xml'])
            removed=[source_model.joint(i).name for i in range(source_model.njnt) if source_model.joint(i).name not in {m.joint(j).name for j in range(m.njnt)}]
            apply_named_state(m,d,state_spec['path'],allow_missing=removed)
            # Free object poses will be selected by the same explicit FK/support normalization;
            # retain their loaded velocities and the authoritative flex coordinates and time.
        mujoco.mj_forward(m,d);named_state(m,d,run/'inputs/physics_state_by_name.npz')
        if state_spec['mode']=='joint_name_state':
            from .full_prepare import freeze_surface
            root=ET.parse(physics).getroot()
            from ..scene import numbers
            for label in ('apple','plate'):
                bid=m.body(label+'_root').id;node=root.find(f".//body[@name='{label}_root']")
                node.set('pos',numbers(d.xpos[bid]));node.set('quat',numbers(d.xquat[bid]))
            ET.ElementTree(root).write(physics,encoding='utf-8',xml_declaration=True)
            static=freeze_surface(root,m,d,run/'inputs/template_static')
            ET.ElementTree(static).write(run/'inputs/template_static/scene.xml',encoding='utf-8',xml_declaration=True)
        p=read(run/'initial_parameters.json');p.update(defaults(ET.parse(run/'inputs/template_static/scene.xml').getroot()))
        p.update({'apple.scale':1.,'plate.scale':1.});write(run/'initial_parameters.json',p)
        # Preserve an original, portable, unmodified geometry/camera view separately.
        raw=run/'source_xml/portable'
        if not (run/'source_xml/render_receipt.json').exists():
            records=portable_copy(audit['xml'],raw);rm=mujoco.MjModel.from_xml_path(str(raw/'scene.xml'));rd=mujoco.MjData(rm)
            if state_spec['mode']=='joint_name_state':apply_named_state(rm,rd,state_spec['path'])
            mujoco.mj_forward(rm,rd);named_state(rm,rd,raw/'original_state_by_name.npz')
            np.savez_compressed(raw/'scene.state.npz',qpos=rd.qpos,qvel=rd.qvel,time=np.array(rd.time))
            events=b.reserve_render('render_initial',dict(tag='raw_source_xml_render'))
            with Renderer(rm) as renderer:rgb=renderer.rgb(rd)
            save_rgb(run/'raw_source_xml.png',rgb)
            for event in events:b.finish(event)
            b.record('render_passes',dict(rgb=1,segmentation=0,tag='raw_source_xml_render'))
            write(run/'source_xml/render_receipt.json',dict(resource_closure=records,camera='Original authored camera, not the normalized Mount prior',state=state_spec,xml_sha256=audit['xml_sha256']))
        prior=read(run/'camera_prior.json');prior['soft_surface_policy']='This task uses original native flex with explicitly declared fresh reset or authoritative named state; no historical settle'
        write(run/'camera_prior.json',prior)
        camera=read(run/'camera.json');camera['fov_source']='42 degree approximate Mount prior recomputed from this episode observed frame0; no optimized historical camera';write(run/'camera.json',camera)
        shutil.copy2(CODE_ROOT/'configs/head_camera_pilot_v1.json',run/'inputs/mount_prior_config.json')
        manifest=read(run/'source_manifest.json');manifest['source_files'].update(audit['resource_closure']);manifest['source_files'].update(audit['training_source_files']);manifest.update(source_mode='initial_xml',source_xml=audit,initial_state=dict(mode=state_spec['mode'],qvel_max=float(np.abs(d.qvel).max()),time=float(d.time),by_name=True,path='inputs/physics_state_by_name.npz'),initialization_implementation_sha256=sha(Path(__file__)),material_migration=dict(defaults_applied_once=True,legacy_v4_multipliers_used=False),historical_reconstruction_input=False)
        manifest['inputs']={str(p.relative_to(run)):sha(p) for p in (run/'inputs').rglob('*') if p.is_file()};write(run/'source_manifest.json',manifest)
        write(marker,dict(entry_sha256=digest(entry),source_xml_sha256=audit['xml_sha256'],source_mode='initial_xml',state_mode=state_spec['mode'],utc=now()))


def initialize_checks(s):
    if s.state.get('initial_xml_verified'):return
    from .v5_tools import evaluate, images
    from .diagnostics import load_state
    row=evaluate(s,pool='render_initial',tag='normalized_feasible_initial');rgb,masks=images(s,row)
    save_rgb(s.run/'initial_reconstruction.png',rgb);save_rgb(s.run/'mount_layout_baseline.png',rgb)
    np.savez_compressed(s.run/'initial_masks.npz',**masks)
    write(s.run/'metrics_mount_baseline.json',row['metrics']);write(s.run/'initial_asset_dimensions.json',row['geometry'])
    with s.budget.computation(dict(kind='initial_xml_normalization_verification')):
        xml=s.scene.export(s.run/'initial_native_verification',physics=True);m,d=load_state(xml)
        with np.load(s.scene.physics_state) as z:
            by_name={str(n):i for i,n in enumerate(z['joint_names'])};velocity_error=0.;position_error=0.
            for j in range(m.njnt):
                name=m.joint(j).name;i=by_name[name];a=int(z['dof_addresses'][i]);n=int(z['dof_sizes'][i]);v=int(m.jnt_dofadr[j]);velocity_error=max(velocity_error,float(np.max(np.abs(d.qvel[v:v+n]-z['qvel'][a:a+n]))))
                if name not in ('apple_freejoint','plate_freejoint'):
                    a=int(z['qpos_addresses'][i]);n=int(z['qpos_sizes'][i]);q=int(m.jnt_qposadr[j]);position_error=max(position_error,float(np.max(np.abs(d.qpos[q:q+n]-z['qpos'][a:a+n]))))
            time_error=abs(d.time-float(z['time']))
        init=read(s.run/'initialization.json');h0=init['h0_m'];camera=s.scene.locked_camera
        checks=dict(source_scale_one=all(read(s.run/'initial_parameters.json')[k+'.scale']==1. for k in ('apple','plate')),zero_offsets=init['extra_hand_direction_offset_m']==0. and s.config['initialization']['extra_apple_offset_m']==0.,h0=abs(init['soft_top_m']-h0)<1e-9 and abs(init['rigid_top_m']-(h0-.006))<1e-9,original_prior_camera=camera==read(s.run/'camera.json'),native_flex=m.nflex>0,qvel=velocity_error<1e-12,nonobject_qpos=position_error<1e-9,time=time_error<1e-12,finite=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),no_historical_input=read(s.run/'source_manifest.json')['historical_reconstruction_input'] is False)
        report=dict(passed=all(checks.values()),checks=checks,normalized_candidate_id=row['candidate_id'],raw_fk=read(s.run/'initialization.json'),raw_parameters=read(s.run/'initial_parameters.json'),feasible_parameters=s.scene.params.copy(),feasibility_changes=s.scene.initial_support_seed,geometry=row['geometry'],mask_quality=read(s.run/'mask_review.json'),velocity_error=velocity_error,nonobject_position_error=position_error,time_error=time_error,normalization_reasons=['Original support assembly translated to h0 / h0-6mm','Observed frame0 Mount prior, not authored 69-degree camera','Both objects free, native gravity retained','Collision support Z and nearest feasible XY from existing backend','V5 independent Phong defaults once; source-relative scales1'],checked_utc=now())
        write(s.run/'initial_xml_verification.json',report);write(s.run/'feasible_initial_parameters.json',s.scene.params)
        if not report['passed']:raise ValueError('Initial XML import/normalization failed: '+str(checks))
    s.state['initial_xml_verified']=True;s.save()
