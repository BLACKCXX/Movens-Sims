"""Import immutable object templates, reviewed observations and verified mount FK."""
from __future__ import annotations
import copy
import hashlib
import shutil
import time
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import mujoco
import numpy as np
import pyarrow.parquet as pq

from .backend import absolute_assets
from ..head_camera import free_objects
from ..ingest import portable_copy
from ..robot_camera import prepare_prior
from ..util import ROOT,digest,read,sha,write,save_rgb
from ..vision import frame_at,describe,segment


def replace_objects(environment, original, poses):
    """Keep the frozen environment, replace only objects with immutable source data."""
    base=absolute_assets(ET.parse(environment).getroot(),Path(environment).parent)
    source=absolute_assets(ET.parse(original).getroot(),Path(original).parent)
    asset=base.find('asset');world=base.find('worldbody');source_asset=source.find('asset')
    required=set()
    objects=[]
    for label in ('apple','plate'):
        old=world.find(f"body[@name='{label}_root']")
        if old is None:raise ValueError('Object must be world-rooted')
        world.remove(old);node=copy.deepcopy(source.find(f".//body[@name='{label}_root']"))
        for key in ('euler','xyaxes','zaxis','axisangle'):node.attrib.pop(key,None)
        node.set('pos',' '.join(map(str,poses[label]['xyz_m'])));node.set('quat',' '.join(map(str,poses[label]['quaternion_wxyz'])))
        for geom in node.iter('geom'):
            for key in ('mesh','material'):
                if geom.get(key):required.add(geom.get(key))
        objects.append(node)
    changed=True
    while changed:
        changed=False
        for elem in source_asset:
            if elem.get('name') in required and elem.get('texture') and elem.get('texture') not in required:required.add(elem.get('texture'));changed=True
    renames={name:'v3_original__'+name for name in required}
    for elem in source_asset:
        if elem.get('name') not in required:continue
        node=copy.deepcopy(elem)
        for key in ('name','texture'):
            if node.get(key) in renames:node.set(key,renames[node.get(key)])
        asset.append(node)
    for obj in objects:
        for geom in obj.iter('geom'):
            for key in ('mesh','material'):
                if geom.get(key) in renames:geom.set(key,renames[geom.get(key)])
        world.append(obj)
    return base


def prepare(entry,config,run):
    if config.get('initialization',{}).get('mode')=='fk_h0':
        from .full_prepare import prepare_full
        return prepare_full(entry,config,run)
    run=Path(run);run.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    ep=f"episode_{entry['episode']:06}";prior_run=ROOT/config['mount_source_root']/ep;legacy=ROOT/config['legacy_comparison_root']/ep
    selection=read(prior_run/'selection.json');variant=config['camera']['selection_variant'];selected_name=selection[variant]['best_strict'];selected=prior_run/'candidates'/selected_name
    result=read(selected/'result.json');camera=read(selected/'camera_final.json');poses=read(selected/'geometry_checks.json')
    for name in ('official_g1_rev10.urdf','official_mount_README.md',ep+'.parquet'):
        target=run/'inputs'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(prior_run/'inputs'/name,target)
    oldconfig=read(ROOT/'configs/head_camera_pilot_v1.json');fk=prepare_prior(entry,oldconfig,run)
    profile=result['mount_profile'];computed=fk['priors'][profile]
    np.testing.assert_allclose(camera['xyz_m'],computed['xyz_m'],atol=1e-10)
    np.testing.assert_allclose(camera['rotation_camera_to_world'],computed['rotation_camera_to_world'],atol=1e-10)
    camera.update(prior_status='approximate_robot_mount_not_calibrated',mount_profile=profile,selected_historical_candidate=selected_name,fov_source='Actual selected historical candidate; hardware/intrinsics not confirmed',root_source=fk['root'],observed_state_frame=fk['frame_index'],timestamp_s=fk['timestamp_s'])
    write(run/'camera.json',camera)
    rgb,fps=frame_at(ROOT/'data'/ep/'real_ego.mp4',0)
    if sha(prior_run/'reference.png')!=read(prior_run/'mask_review.json')['reference_sha256']:raise ValueError('Audited observation identity changed')
    if not np.array_equal(rgb,cv2.cvtColor(cv2.imread(str(prior_run/'reference.png')),cv2.COLOR_BGR2RGB)):raise ValueError('Decoded frame0 differs from Mount reference')
    save_rgb(run/'reference.png',rgb);shutil.copytree(prior_run/'masks',run/'masks',dirs_exist_ok=True)
    shutil.copy2(prior_run/'reference_masks_review.png',run/'reference_masks_review.png');shutil.copy2(prior_run/'mask_review.json',run/'mask_review.json')
    maskfiles={str(p.relative_to(run)):sha(p) for p in (run/'masks').glob('*.png')};obs_version='reviewed_frame0_v3_'+digest(maskfiles)[:16]
    temporal=[];raw=pq.read_table(run/'inputs'/(ep+'.parquet'),columns=['frame_index','timestamp','observation.state']);rows=raw.slice(0,7).to_pylist();state0=np.array(rows[0]['observation.state'])
    grey=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    with np.load(run/'masks/observations.npz') as z:targets={k:z[k] for k in ('apple','plate','cloth')}
    for index in (1,3,6):
        image,_=frame_at(ROOT/'data'/ep/'real_ego.mp4',index);save_rgb(run/'inputs'/f'real_frame_{index:03}.png',image)
        newgrey=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);flow={}
        for label,mask in targets.items():
            points=cv2.goodFeaturesToTrack(grey,100,.01,5,mask=mask.astype(np.uint8)*255)
            if points is None:continue
            end,status,_=cv2.calcOpticalFlowPyrLK(grey,newgrey,points,None);delta=(end-points).reshape(-1,2)[status.reshape(-1).astype(bool)]
            flow[label]=dict(median_flow_norm_px=float(np.median(np.linalg.norm(delta,axis=1))),median_flow_xy_px=np.median(delta,axis=0).tolist(),tracked=len(delta))
        temporal.append(dict(frame=index,timestamp_s=rows[index]['timestamp'],observed_joint_max_delta_rad=float(np.max(abs(np.array(rows[index]['observation.state'])-state0))),flow=flow,usage='Read-only consistency only; no static/multi-view triangulation claim'))
    write(run/'observation.json',dict(version=obs_version,frame=0,timestamp_s=0.,fps=fps,resolution=[640,480],crop='none',camera_model='centered pinhole approximation, no measured distortion',mask_sha256s=maskfiles,review=read(run/'mask_review.json'),temporal=temporal,fit_frames=[0],heldout_fit_frames=[],no_new_triangulation_claim=True))
    geometry=read(prior_run/'geometry_inputs.json');original=geometry['imported_initial']
    for kind in ('static','physics'):
        original_copy=run/'inputs'/('original_'+kind);portable_copy(ROOT/original[kind],original_copy)
        env=selected/('scene/scene_optimized.xml' if kind=='static' else 'physics/scene_physics.xml')
        root=replace_objects(env,original_copy/'scene.xml',poses);staging=run/'inputs'/('staging_'+kind);staging.mkdir(exist_ok=True);source=staging/'scene.xml';ET.ElementTree(root).write(source,encoding='utf-8',xml_declaration=True)
        portable_copy(source,run/'inputs'/('template_'+kind))
    shutil.copy2(ROOT/original['physics_state'],run/'inputs/physics_state_by_name.npz')
    scale=read(legacy/'parameters_optimized.json')['apple.scale']
    appearance=ET.parse(selected/'scene/scene_optimized.xml');head=appearance.find('visual/headlight');cloth=next(g for g in appearance.findall('.//geom') if g.get('name','').startswith('frozen_'))
    plate=appearance.find(".//body[@name='plate_root']");material=next(g.get('material') for g in plate.iter('geom') if g.get('material'));albedo=appearance.find(f"./asset/material[@name='{material}']")
    initial={'apple.scale':scale,'plate.scale':1.,'light.ambient':float(np.fromstring(head.get('ambient'),sep=' ').mean()),'light.diffuse':float(np.fromstring(head.get('diffuse'),sep=' ').mean()),'cloth.gray':float(np.fromstring(cloth.get('rgba'),sep=' ')[:3].mean()),'plate.albedo':float(np.fromstring(albedo.get('rgba'),sep=' ')[:3].mean())}
    write(run/'initial_parameters.json',initial)
    audit=dict(immutable_object_source=original,source_scale_definition='1.0 = imported original object mesh and authored scale chain, not the already batch-scaled object',apple_initial_absolute_scale=scale,plate_initial_absolute_scale=1.,shared_mesh_policy='Object-specific clones on fresh source tree',units='m; nominal source units, not independently measured',plate_visual_collision_gap_evidence=read(legacy/'plate_support_surface.json'),plate_face_status='Existing ray audit has center below rim: visible authored side is concave upward; RGB impression alone does not establish inversion. Normals/inner shape still need visual review.')
    write(run/'asset_audit.json',audit)
    inputs={str(p.relative_to(run)):sha(p) for p in (run/'inputs').rglob('*') if p.is_file()}
    write(run/'source_manifest.json',dict(episode=entry['episode'],data_revision=entry['data_revision'],real_video_sha256=sha(ROOT/'data'/ep/'real_ego.mp4'),real_video_local=str(ROOT/'data'/ep/'real_ego.mp4'),selected_mount_candidate=selected_name,mount_candidate_result_sha256=sha(selected/'result.json'),original_asset_manifest_sha256=sha(ROOT/'data'/ep/'source_manifest.json'),inputs=inputs,observation_version=obs_version,prepare_seconds=time.monotonic()-start,source_project_read_only=True))
    return selected


def load_observation(run):
    run=Path(run)
    with np.load(run/'masks/observations.npz') as z:target={k:z[k].copy() for k in ('apple','plate','cloth')};valid=z['valid'].copy()
    return dict(rgb=cv2.cvtColor(cv2.imread(str(run/'reference.png')),cv2.COLOR_BGR2RGB),target=target,valid=valid,version=read(run/'observation.json')['version'])
