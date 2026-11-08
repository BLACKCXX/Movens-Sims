"""402-source ingestion and h0 reset; no policy playback or source writes."""
from __future__ import annotations
import copy
from collections import Counter
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
import pyarrow.parquet as pq

from ..ingest import resource_closure
from ..scene import numbers, prune_assets
from ..robot_camera import prepare_prior
from ..head_camera import free_objects, save_state_by_name
from ..render import Renderer
from ..util import ROOT, CODE_ROOT, read, write, sha, digest, now, save_rgb
from ..vision import frame_at, largest, describe
from .budget import BudgetManager


def create_manifest(config):
    """Freeze every original ID, regardless of historical dynamic acceptance."""
    if config.get('workflow') in ('classified_reconstruction_v4','terra_closed_loop_v5'):
        manifest=read(ROOT/config['manifest'])
        if config['allowed_episodes']!=[0,1,2] or [e['episode'] for e in manifest['entries']]!=[0,1,2]:raise ValueError('The isolated task must contain exactly episodes 0,1,2')
        return manifest
    root=ROOT/config['run_root'];root.mkdir(parents=True,exist_ok=True)
    manifest_path=ROOT/config['manifest']
    if manifest_path.exists():
        manifest=read(manifest_path)
        if len(manifest['entries'])!=402 or [e['episode'] for e in manifest['entries']]!=list(range(402)):
            raise ValueError('Frozen full batch must contain exactly original IDs 0..401')
        return manifest
    source=Path(config['source_project']);init=config['initialization'];raw=source/'data/apple_raw'/init['revision']
    info=read(raw/'meta/info.json');episodes=[__import__('json').loads(x) for x in (raw/'meta/episodes.jsonl').read_text().splitlines()]
    if info['total_episodes']!=402 or sorted(x['episode_index'] for x in episodes)!=list(range(402)):
        raise ValueError('Raw dataset identity/count changed')
    training=read(source/init['training_index']);formal=read(source/init['training_formal_config'])
    if sha(source/init['training_index'])!=formal['data_provenance']['teacher_index_sha256']:
        raise ValueError('Latest training index checksum mismatch')
    selected={x['episode_index'] for x in training['episodes']}
    batch=Path(training['source_batch']);assets={};entries=[]
    shared=ROOT/init['data_root']/'shared';shared.mkdir(parents=True,exist_ok=True)
    for row in sorted(episodes,key=lambda x:x['episode_index']):
        ep=row['episode_index'];candidate=batch/f'episode_{ep:06}'/'offset_000mm';scene=candidate/'scene'
        parquet=raw/info['data_path'].format(episode_chunk=ep//1000,episode_index=ep)
        video=raw/info['video_path'].format(episode_chunk=ep//1000,episode_index=ep,video_key='observation.images.ego_view')
        entry=dict(episode=ep,data_revision=init['revision'],scene_dir=str(scene),candidate=str(candidate),
                   estimate=str(candidate.parent/'estimate.json'),real_video=str(video),parquet=str(parquet),
                   training_selected=ep in selected,old_sim_video=str(candidate/'videos/main_rgb.mp4'))
        missing=[];files={}
        required=[parquet,video,scene/'scene.xml',scene/'robot.json',scene/'prepared_manifest.json',scene/'batch_construction.json',candidate.parent/'estimate.json']
        for p in required:
            if not p.is_file():missing.append(str(p))
            else:files[str(p)]=sha(p)
        if not missing:
            try:
                closure=resource_closure(scene/'scene.xml')
                for xml,(_,refs) in closure.items():
                    if xml!=scene/'scene.xml':files[str(xml)]=sha(xml)
                    for _,_,p in refs:
                        if p in closure:continue
                        key=str(p)
                        if key not in assets:
                            h=sha(p);dest=shared/'assets'/(h[:20]+'_'+p.name);dest.parent.mkdir(exist_ok=True)
                            if not dest.exists():shutil.copy2(p,dest)
                            if sha(dest)!=h:raise ValueError('Shared asset checksum mismatch')
                            assets[key]=dict(path=str(dest),sha256=h,bytes=p.stat().st_size)
                receipt=read(scene/'batch_construction.json')
                if receipt['base_offset_mm']!=0 or receipt['offset_mm']!=0:raise ValueError('Nonzero hand-direction compensation in source')
                entry['h0_m']=receipt['original_estimated_table_top_m']
            except (OSError,ValueError) as exc:missing.append(str(exc))
        entry.update(source_files=files,input_complete=not missing,missing=missing);entries.append(entry)
    training_audit=read(batch/'shared/geometry_audit.json')
    write(shared/'assets.json',assets)
    for name in ('official_g1_rev10.urdf','official_mount_README.md'):
        shutil.copy2(ROOT/init['mount_evidence_root']/name,shared/name)
    write(shared/'training_asset_audit.json',dict(training_index=str(source/init['training_index']),training_index_sha256=sha(source/init['training_index']),
        formal_config=str(source/init['training_formal_config']),scale_chain=training_audit['scale_chain'],shape=training_audit['shape'],
        source_geometry_audit_sha256=sha(batch/'shared/geometry_audit.json'),scale_at_import=1.,extra_xy_compensation_m=0.,soft_top_minus_h0_m=0.,thickness_m=.006))
    manifest=dict(version='full402_h0_manifest_v1',created_utc=now(),frozen=True,data_revision=init['revision'],dataset=info['robot_type'],
                  selection='All original raw IDs, independent of training/dynamic success',entries=entries,assets_manifest=str(shared/'assets.json'),
                  source_project=str(source),training_selected=len(selected),training_excluded=402-len(selected),parent_experiment='pipeline_refactor_ep000000_v1; different h0 baseline, old ledger retained')
    manifest['selection_sha256']=digest(entries);write(manifest_path,manifest);return manifest


def shift_to_h0(tree, receipt, h0):
    """Translate the whole support assembly, never relabel an elevated model."""
    delta=float(h0-receipt['actual_uncompressed_soft_top_m']);thickness=receipt['thickness_m']
    if abs(thickness-.006)>1e-10:raise ValueError('Unexpected inherited cloth thickness')
    world=tree.find('worldbody')
    for body in world.findall('body'):
        if body.get('name')=='table_root' or body.get('name','').startswith('pad_node_'):
            xyz=np.fromstring(body.get('pos','0 0 0'),sep=' ');xyz[2]+=delta;body.set('pos',numbers(xyz))
    return delta


def freeze_surface(tree, model, data, out):
    """Thin closed display mesh; the upper skin is the physical flex outer face."""
    static=copy.deepcopy(tree);world=static.find('worldbody');asset=static.find('asset')
    for body in list(world):
        if body.get('name','').startswith('pad_node_'):world.remove(body)
    deform=static.find('deformable')
    if deform is not None:static.remove(deform)
    for i in range(model.nflex):
        if model.flex_group[i]==5:continue
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_FLEX,i);start=int(model.flex_vertadr[i]);n=int(model.flex_vertnum[i]);v=data.flexvert_xpos[start:start+n].copy()
        v[:,2]+=model.flex_radius[i]
        a=int(model.flex_elemdataadr[i]);m=int(model.flex_elemnum[i]);faces=model.flex_elem[a:a+3*m].reshape(-1,3).copy()
        # Orient upper faces upward; the support solver ignores the lower skin.
        down=np.cross(v[faces[:,1]]-v[faces[:,0]],v[faces[:,2]]-v[faces[:,0]])[:,2]<0
        faces[down]=faces[down][:,[0,2,1]]
        edges=Counter(tuple(sorted((int(f[j]),int(f[(j+1)%3])))) for f in faces for j in range(3))
        allfaces=[*faces,*(f[[0,2,1]]+n for f in faces)]
        for (a,b),count in edges.items():
            if count==1:allfaces.extend([[a,b,b+n],[a,b+n,a+n]])
        path=out/'assets'/('frozen_'+name+'.obj');path.parent.mkdir(exist_ok=True)
        with path.open('w') as f:
            for p in np.concatenate([v,v-[0,0,.0002]]):f.write('v '+numbers(p)+'\n')
            for face in allfaces:f.write('f '+' '.join(str(int(x)+1) for x in face)+'\n')
        mesh='frozen_'+name
        ET.SubElement(asset,'mesh',name=mesh,file=str(path))
        ET.SubElement(world,'geom',name=mesh,type='mesh',mesh=mesh,group='1',contype='0',conaffinity='0',rgba='.18747670785 .18747670785 .18747670785 1')
    prune_assets(static);return static


def _objects(model):
    groups={}
    for label in ('apple','plate'):
        ids={model.body(label+'_root').id}
        for i in range(model.nbody):
            if int(model.body_parentid[i]) in ids:ids.add(i)
        groups[label]=[g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in ids and (model.geom_contype[g] or model.geom_conaffinity[g])]
    return groups


def _component(mask, seed):
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    choices=[]
    for i in range(1,n):
        area=stats[i,cv2.CC_STAT_AREA];fraction=area/mask.size
        if not .002<fraction<.35:continue
        score=np.linalg.norm(centers[i]-seed)/np.hypot(*mask.shape)-.02*np.sqrt(fraction)
        choices.append((score,i))
    if not choices:raise ValueError('No plausible pixel component near the FK prior; observation needs review')
    chosen=min(choices)[1];out=labels==chosen
    contours,_=cv2.findContours(out.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    filled=out.astype(np.uint8);cv2.drawContours(filled,contours,-1,1,-1);return filled.astype(bool)


def _pixels(rgb,robot_mask,seeds):
    """Auditable pixel thresholds; projected robot region is an uncertainty mask."""
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    apple=_component((((hsv[:,:,0]<38)|(hsv[:,:,0]>170))&(hsv[:,:,1]>95)&(hsv[:,:,2]>75)),seeds['apple'])
    plate=_component((hsv[:,:,1]<55)&(hsv[:,:,2]>155),seeds['plate'])
    cloth=largest((hsv[:,:,1]<75)&(hsv[:,:,2]<155))&~apple&~plate
    occlusion=cv2.dilate(robot_mask.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    valid=~occlusion
    masks=dict(apple=apple,plate=plate,cloth=cloth);quality={}
    for label in ('apple','plate'):
        desc=describe(masks[label]);fraction=float(masks[label].mean());occ=float((masks[label]&occlusion).sum()/max(1,masks[label].sum()))
        desc.update(area_fraction=fraction,possible_robot_occlusion_fraction=occ,occluded_by_robot=occ>0.,confidence='automatic_unreviewed',quality_ok=.002<fraction<.35 and occ<.2)
        quality[label]=desc
    return masks,valid,occlusion,quality


def prepare_full(entry, config, run):
    run=Path(run);start=time.monotonic();ep=entry['episode'];init=config['initialization']
    run.mkdir(parents=True,exist_ok=True)
    if not entry['input_complete']:raise FileNotFoundError('; '.join(entry['missing']))
    for p,h in entry['source_files'].items():
        if not Path(p).is_file() or sha(p)!=h:raise ValueError('Immutable source changed: '+p)
    shared=ROOT/init['data_root']/'shared';assets=read(shared/'assets.json');source=Path(entry['scene_dir'])
    inputs=run/'inputs';inputs.mkdir(exist_ok=True);robotimport=inputs/'robot_import/source';robotimport.mkdir(parents=True,exist_ok=True)
    tree=ET.parse(source/'scene.xml').getroot()
    if tree.findall('.//include'):raise ValueError('Full source requires explicitly flattened includes')
    for node in tree.findall('./asset/*'):
        if node.get('file'):
            p=(source/node.get('file')).resolve();record=assets[str(p)]
            if sha(record['path'])!=record['sha256']:raise ValueError('Shared resource changed')
            node.set('file',record['path'])
    for comp in tree.findall('compiler'):
        for key in ('meshdir','texturedir','assetdir'):comp.attrib.pop(key,None)
    estimate=read(entry['estimate']);receipt=read(source/'batch_construction.json');h0=entry['h0_m']
    delta=shift_to_h0(tree,receipt,h0)
    # Existing zero-offset FK XY and same-episode orientation; no future trajectory replay.
    for label in ('apple','plate'):
        node=tree.find(f".//body[@name='{label}_root']");p=np.asarray(estimate[label+'_xyz_m']).copy();p[2]+=delta;node.set('pos',numbers(p))
    ET.ElementTree(tree).write(robotimport/'scene.xml',encoding='utf-8',xml_declaration=True)
    for name in ('robot.json','prepared_manifest.json'):shutil.copy2(source/name,robotimport/name)
    for name in ('official_g1_rev10.urdf','official_mount_README.md'):shutil.copy2(shared/name,inputs/name)
    camera_config=read(CODE_ROOT/'configs/head_camera_pilot_v1.json');camera_config['source_project']=config['source_project']
    prior=prepare_prior(entry,camera_config,run,imported=robotimport.parent,table_top=h0)
    camera=copy.deepcopy(prior['priors'][init['mount_profile']]);camera.update(fovy_deg=init['fovy_deg'],mount_profile=init['mount_profile'],prior_status='approximate_robot_mount_not_calibrated',fov_source='Frozen historical selected Mount candidate, not measured intrinsics',root_source=prior['root'],resolution=[640,480])
    write(run/'camera.json',camera)
    world=tree.find('worldbody')
    # Use the same camera before and after deleting the robot.
    for parent in tree.iter():
        for child in list(parent):
            if child.tag=='camera':parent.remove(child)
    ET.SubElement(world,'camera',name='approx_ego_camera',pos=numbers(camera['xyz_m']),quat=numbers(camera['quaternion_wxyz']),fovy=str(camera['fovy_deg']))
    ET.ElementTree(tree).write(robotimport/'scene.xml',encoding='utf-8',xml_declaration=True)
    full=mujoco.MjModel.from_xml_path(str(robotimport/'scene.xml'));fd=mujoco.MjData(full)
    for name,value in prior['observed_joints_rad'].items():fd.qpos[full.jnt_qposadr[full.joint(name).id]]=value
    mujoco.mj_forward(full,fd)
    pelvis=full.body('pelvis').id;robotids={pelvis}
    for b in range(full.nbody):
        if int(full.body_parentid[b]) in robotids:robotids.add(b)
    budget=BudgetManager(ROOT/'runs/_task_budgets'/(config['task_group_id']+'.sqlite'),f"{config['task_group_id']}/episode_{ep:06}",config['budget'])
    occfile=run/'masks/robot_mount_projection.png';occfile.parent.mkdir(exist_ok=True)
    if occfile.exists():robotmask=cv2.imread(str(occfile),0)>0
    else:
        events=budget.reserve_render('render_initial',dict(tag='observed_robot_occlusion_initialization',width=640,height=480,reference_frame=0))
        with Renderer(full) as renderer:
            robotmask=renderer.masks(fd,dict(robot=np.flatnonzero(np.isin(full.geom_bodyid,list(robotids)))))['robot']
        cv2.imwrite(str(occfile),robotmask.astype(np.uint8)*255)
        for event in events:budget.finish(event)
        budget.record('render_passes',dict(rgb=0,segmentation=1,tag='observed_robot_occlusion_initialization'))
    pelvis_node=world.find("body[@name='pelvis']");removed={n.get('name') for n in pelvis_node.iter() if n.get('name')};world.remove(pelvis_node)
    for tag in ('actuator','sensor','tendon','keyframe'):
        node=tree.find(tag)
        if node is not None:tree.remove(node)
    for tag in ('contact','equality'):
        section=tree.find(tag)
        if section is not None:
            for node in list(section):
                if any(v in removed for v in node.attrib.values()):section.remove(node)
    for i,g in enumerate(world.find("body[@name='table_root']").iter('geom')):
        if not g.get('name'):g.set('name',f'movens_table_visual_{i}')
    for flex in tree.findall('./deformable/flex'):flex.set('rgba',numbers([init['cloth_gray']]*3+[1]))
    prune_assets(tree)
    physics=inputs/'template_physics';physics.mkdir(exist_ok=True);physics_xml=physics/'scene.xml';ET.ElementTree(tree).write(physics_xml,encoding='utf-8',xml_declaration=True);free_objects(physics_xml)
    pm=mujoco.MjModel.from_xml_path(str(physics_xml));pd=mujoco.MjData(pm);mujoco.mj_forward(pm,pd)
    # Recompute Z from real compiled collision vertices, including the current shape.
    for label,gs in _objects(pm).items():
        points=[];bid=pm.body(label+'_root').id
        for g in gs:
            mesh=int(pm.geom_dataid[g]);a=int(pm.mesh_vertadr[mesh]);n=int(pm.mesh_vertnum[mesh]);points.append(pm.mesh_vert[a:a+n]@pd.geom_xmat[g].reshape(3,3).T+pd.geom_xpos[g])
        bottom=float(np.concatenate(points)[:,2].min());j=pm.joint(label+'_freejoint').id;pd.qpos[pm.jnt_qposadr[j]+2]+=h0+init['initial_clearance_m']-bottom
    mujoco.mj_forward(pm,pd)
    tree=ET.parse(physics_xml).getroot()
    for label in ('apple','plate'):
        bid=pm.body(label+'_root').id;node=tree.find(f".//body[@name='{label}_root']");node.set('pos',numbers(pd.xpos[bid]));node.set('quat',numbers(pd.xquat[bid]))
    ET.ElementTree(tree).write(physics_xml,encoding='utf-8',xml_declaration=True)
    save_state_by_name(physics_xml,None,inputs/'physics_state_by_name.npz')
    surface_top=float(pd.flexvert_xpos[:,2].max()+pm.flex_radius.max())
    if abs(surface_top-h0)>1e-9:raise ValueError('Actual flex outer face does not equal h0')
    staticdir=inputs/'template_static';staticdir.mkdir(exist_ok=True);static=freeze_surface(tree,pm,pd,staticdir);ET.ElementTree(static).write(staticdir/'scene.xml',encoding='utf-8',xml_declaration=True)
    write(run/'initial_parameters.json',{'apple.scale':1.,'plate.scale':1.,'cloth.gray':init['cloth_gray'],'light.ambient':.35,'light.diffuse':.7,'plate.albedo':.8})
    write(run/'initialization.json',dict(episode=ep,h0_m=h0,soft_top_m=surface_top,rigid_top_m=h0-.006,source_soft_top_m=receipt['actual_uncompressed_soft_top_m'],support_delta_m=delta,
        extra_hand_direction_offset_m=0.,table_geometry_fitted=False,cloth_geometry_fitted=False,reset='fresh uncompressed source geometry; zero slide qpos and velocities; no old settle',
        events=estimate['events'],windows=estimate.get('event_windows_50hz'),wrist_offset_source=estimate.get('wrist_offset_source'),confidence=estimate.get('confidence'),
        needs_review_reasons=estimate.get('needs_review_reasons',[]),camera_state='observed frame0, not desired action or settle',h0_semantics='Original pre-softpad estimated contact plane, now used as outer soft surface; rigid top is h0 minus inherited thickness'))
    rgb,fps=frame_at(entry['real_video'],0);save_rgb(run/'reference.png',rgb)
    projection=budget.reserve('projection_evaluations',2,dict(tag='FK_component_association'))
    rot=np.asarray(camera['rotation_camera_to_world']);fy=240/np.tan(np.radians(camera['fovy_deg']/2));seeds={}
    for label in ('apple','plate'):
        p=(pd.xpos[pm.body(label+'_root').id]-np.array(camera['xyz_m']))@rot
        seeds[label]=np.array([320+fy*p[0]/max(1e-6,-p[2]),240-fy*p[1]/max(1e-6,-p[2])])
    masks,valid,occlusion,quality=_pixels(rgb,robotmask,seeds);budget.finish(projection)
    np.savez_compressed(run/'masks/observations.npz',**masks,valid=valid,occlusion=occlusion)
    preview=rgb.copy()
    for label,color in [('apple',(255,80,0)),('plate',(0,255,120)),('cloth',(80,170,255))]:
        mask=masks[label];cv2.imwrite(str(run/'masks'/(label+'.png')),mask.astype(np.uint8)*255);cv2.imwrite(str(run/'masks'/(label+'_edge.png')),cv2.Canny(mask.astype(np.uint8)*255,80,180))
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(preview,contours,-1,color,1)
    for label,mask in [('valid',valid),('occlusion',occlusion)]:cv2.imwrite(str(run/'masks'/(label+'.png')),mask.astype(np.uint8)*255)
    save_rgb(run/'reference_masks_review.png',preview)
    temporal=[];grey=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    raw=pq.read_table(entry['parquet'],columns=['observation.state']);state0=np.array(raw.slice(0,1).to_pylist()[0]['observation.state'])
    for index in (1,3,6):
        if index>=raw.num_rows:continue
        frame,_=frame_at(entry['real_video'],index);save_rgb(inputs/f'real_frame_{index:03}.png',frame);changes={}
        for label in ('apple','plate','cloth'):
            pts=cv2.goodFeaturesToTrack(grey,100,.01,5,mask=masks[label].astype(np.uint8)*255)
            if pts is None:continue
            end,status,_=cv2.calcOpticalFlowPyrLK(grey,cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY),pts,None)
            deltaflow=(end-pts).reshape(-1,2)[status.reshape(-1).astype(bool)]
            if len(deltaflow):changes[label]=float(np.median(np.linalg.norm(deltaflow,axis=1)))
        temporal.append(dict(frame=index,timestamp_s=index/fps,median_feature_flow_px=changes,observed_joint_max_delta_rad=float(np.max(abs(np.array(raw.slice(index,1).to_pylist()[0]['observation.state'])-state0))),usage='consistency only, never fitted'))
    maskfiles={str(p.relative_to(run)):sha(p) for p in (run/'masks').glob('*.png')};obs_version='full402_frame0_v1_'+digest(maskfiles)[:16]
    review=dict(reviewed=False,method='HSV pixel components, full frame; FK projected robot pixels conservatively excluded as uncertain, not verified real segmentation',objects=quality,mask_quality_ok=all(q['quality_ok'] for q in quality.values()),occlusion_pixels=int(occlusion.sum()),top_excluded=False)
    write(run/'mask_review.json',review);write(run/'observation.json',dict(version=obs_version,frame=0,timestamp_s=0.,fps=fps,resolution=[640,480],crop='none',mask_sha256s=maskfiles,review=review,fit_frames=[0],temporal=temporal,heldout_fit_frames=[],camera_model='centered pinhole; unmeasured distortion'))
    write(run/'asset_audit.json',dict(training_asset=read(shared/'training_asset_audit.json'),source_scale_definition='1.0 is the immutable baked latest-training apple; no extra 1.155/1.21 multiplication',plate_face_status='Inherited upward-facing visual plate; convex collision interior is not independently verified',cloth_texture='No measured cloth texture asset; gray albedo only, no recovered wrinkles'))
    hashes={str(p.relative_to(run)):sha(p) for p in inputs.rglob('*') if p.is_file()}
    write(run/'source_manifest.json',dict(episode=ep,data_revision=entry['data_revision'],real_video_local=entry['real_video'],real_video_sha256=entry['source_files'][entry['real_video']],source_files=entry['source_files'],inputs=hashes,observation_version=obs_version,
        immutable_assets={r['path']:r['sha256'] for r in assets.values()},source_project_read_only=True,training_selected=entry['training_selected'],initialization='fk_h0',prepare_seconds=time.monotonic()-start))
    print(f'Prepared episode {ep:06}: h0={h0:.6f}, training apple scale=1, camera FOV={camera["fovy_deg"]}, mask_quality={review["mask_quality_ok"]}',flush=True)
    return run
