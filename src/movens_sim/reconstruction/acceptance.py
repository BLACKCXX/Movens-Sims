"""Independent final-artifact acceptance. Never exports/rebuilds a scene or fits it."""
from __future__ import annotations
from pathlib import Path
import shutil
import sqlite3
import tempfile
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from .diagnostics import load_state,loaded_geometry,members_and_groups,rest_from_disk,auxiliary,spend_render,finish_render
from .metrics import measure,gate
from .prepare import load_observation
from ..render import Renderer
from ..util import ROOT,CODE_ROOT,read,write,sha,digest,save_rgb,now,video_info


def threshold_audit(config):
    old=read(ROOT/'configs/full402_h0_v1.json')['acceptance'];new=config['acceptance'];rows=[]
    for k in ('iou_min','center_px_max','edge_px_max','area_relative_max','axes_relative_max','iou_degradation_max','center_degradation_px','edge_degradation_px'):
        rows.append(dict(check=k,old=old[k],new=new[k],basis='1 - 1.2*(1-old)' if k=='iou_min' else 'old*1.2'))
    extras=[('cloth_iou_min',.97,.964,'restored absolute gate; 1-1.2*(1-old)'),('top_cloth_iou_min',.9,.88,'restored absolute gate; 1-1.2*(1-old)'),('false_background_max',.01,.012,'old*1.2'),('cloth_iou_degradation_max',.01,.012,'old*1.2'),('top_cloth_iou_degradation_max',.02,.024,'old*1.2'),('false_background_degradation_max',.005,.006,'old*1.2'),('initial_support_gap_min_m',-.001,-.0012,'signed tolerance magnitude*1.2'),('initial_support_gap_max_m',.002,.0024,'old*1.2'),('object_pair_penetration_m_max',.001,.0012,'old*1.2'),('rest_translation_m_max',.005,.01,'peak -> terminal; tolerance*2'),('rest_rotation_deg_max',5.,10.,'peak -> terminal; tolerance*2'),('rest_penetration_m_max',.001,.002,'peak -> terminal contact maximum; tolerance*2')]
    rows += [dict(check=k,old=a,new=b,basis=basis) for k,a,b,basis in extras]
    return dict(thresholds=rows,unchanged=['1 second duration','asset identity and finite values','portable reload RGB MAE <0.05, masks <=4 pixels, position/camera <1e-9m, dimensions <1e-7m','budget caps and irreversible accounting','minimum objective improvement 0.0001 and search bounds'],appearance_gate='Actual category/final Terra review plus reported RGB/Lab errors; no new unagreed numeric cutoff',velocity_gate=False,relative_cloth_nondegradation_is_not_absolute_correctness=True)


def artifact_assets(xml):
    xml=Path(xml);rows=[]
    for node in ET.parse(xml).findall('./asset/*'):
        if not node.get('file'):continue
        ref=Path(node.get('file'));path=(xml.parent/ref).resolve()
        rows.append(dict(reference=str(ref),local=not ref.is_absolute() and path.is_relative_to(xml.parent.resolve()),exists=path.is_file(),sha256=sha(path) if path.is_file() else None))
    return rows


def accept_from_disk(run,config,folder=None,budget=None,expected_rgb=None,expected_masks=None,with_rest=True):
    run=Path(run);folder=Path(folder) if folder else run/'reacceptance'/now().replace(':','').replace('+','_');folder.mkdir(parents=True,exist_ok=True)
    params=read(run/'parameters_final.json');camera=read(run/'camera_actual.json');observation=load_observation(run)
    static=run/'delivery/static/portable/scene.xml';dynamic=run/'delivery/dynamic/portable/scene.xml'
    artifacts={};portability={};loaded={}
    with tempfile.TemporaryDirectory(prefix='movens_final_artifacts_') as temporary:
        for kind,xml in [('static',static),('dynamic',dynamic)]:
            assets=artifact_assets(xml)
            if not all(x['exists'] and x['local'] for x in assets):raise ValueError('Delivery has a missing/external asset: '+kind)
            dest=Path(temporary)/kind;shutil.copytree(xml.parent,dest);path=dest/'scene.xml';m,d=load_state(path)
            geometry=loaded_geometry(m,d,params,config['initial_support']);loaded[kind]=(m,d)
            finite=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and np.isfinite(m.mesh_vert).all() and np.isfinite(m.mesh_normal).all())
            cam=m.camera('approx_ego_camera').id
            portability[kind]=dict(loaded_in_independent_directory=True,assets=assets,numerically_finite=finite,geometry=geometry,robot_removed='pelvis' not in [m.body(i).name for i in range(m.nbody)],camera_attached_to_world=int(m.cam_bodyid[cam])==0,free_objects={k:bool(m.body_jntnum[m.body(k+'_root').id]) for k in ('apple','plate')},native_flex_count=m.nflex,xml_sha256=sha(xml),state_sha256=sha(xml.with_suffix('.state.npz')))
            portability[kind]['passed']=bool(finite and portability[kind]['robot_removed'] and portability[kind]['camera_attached_to_world'] and all(portability[kind]['free_objects'].values()) and (kind=='static' or m.nflex>0))
            artifacts[kind]=dict(xml_sha256=sha(xml),state_sha256=sha(xml.with_suffix('.state.npz')),assets={x['reference']:x['sha256'] for x in assets})
        m,d=loaded['static'];events=spend_render(budget,'render_export',dict(kind='independent_final_reload'))
        with Renderer(m) as renderer:rgb=renderer.rgb(d);masks=renderer.masks(d,members_and_groups(m)[1])
        finish_render(budget,events,dict(rgb=1,segmentation=1,view='independent_final_reload'))
        if expected_rgb is None:expected_rgb=cv2.cvtColor(cv2.imread(str(run/'final_ego.png')),cv2.COLOR_BGR2RGB)
        if expected_masks is None:
            with np.load(run/'final_masks.npz') as z:expected_masks={k:z[k].copy() for k in z.files}
        cam=m.camera('approx_ego_camera').id;position_error=float(np.abs(d.cam_xpos[cam]-np.asarray(camera['xyz_m'])).max());rotation_error=float(np.abs(d.cam_xmat[cam].reshape(3,3)-np.asarray(camera['rotation_camera_to_world'])).max());fov_error=abs(float(m.cam_fovy[cam])-camera['fovy_deg'])
        actual_camera=dict(xyz_m=d.cam_xpos[cam].tolist(),rotation_camera_to_world=d.cam_xmat[cam].reshape(3,3).tolist(),fovy_deg=float(m.cam_fovy[cam]),down_angle_deg=float(np.degrees(np.arcsin(np.clip(d.cam_xmat[cam].reshape(3,3)[2,2],-1,1)))),fy_px=float(240/np.tan(np.radians(m.cam_fovy[cam]/2))),resolution=[640,480],coordinate_frame='World meters; camera -Z forward, +Y up',source='Independent portable XML and saved-state reload',calibrated=False)
        actual_camera['reported_prior_derived_metadata']={k:camera[k] for k in ('down_angle_deg','height_above_nominal_cloth_m') if k in camera}
        reload_errors=dict(rgb_mae=float(np.abs(rgb.astype(float)-expected_rgb.astype(float)).mean()),mask_changed_pixels={k:int(np.count_nonzero(masks[k]!=expected_masks[k])) for k in ('apple','plate','cloth')},camera_position_m=position_error,camera_rotation=rotation_error,fovy_deg=fov_error)
        static_geo=portability['static']['geometry'];dynamic_geo=portability['dynamic']['geometry']
        reload_errors['static_dynamic_object_position_m']={k:float(np.abs(np.asarray(static_geo[k]['xyz_m'])-dynamic_geo[k]['xyz_m']).max()) for k in ('apple','plate')}
        reload_errors['static_dynamic_dimensions_m']={k:float(np.abs(np.asarray(static_geo[k]['dimensions_world_m'])-dynamic_geo[k]['dimensions_world_m']).max()) for k in ('apple','plate')}
        reload_passed=bool(reload_errors['rgb_mae']<.05 and max(reload_errors['mask_changed_pixels'].values())<=4 and position_error<1e-9 and rotation_error<1e-9 and fov_error<1e-9 and max(reload_errors['static_dynamic_object_position_m'].values())<1e-9 and max(reload_errors['static_dynamic_dimensions_m'].values())<1e-7 and all(x['passed'] for x in portability.values()))
    save_rgb(folder/'reloaded_ego.png',rgb);np.savez_compressed(folder/'reloaded_masks.npz',**masks)
    write(folder/'camera_verified_from_xml.json',actual_camera)
    metrics=measure(rgb,masks,observation,config);baseline=read(run/'metrics_mount_baseline.json') if (run/'metrics_mount_baseline.json').exists() else None;numeric=gate(metrics,static_geo,baseline,config)
    if with_rest:
        if budget:
            with budget.computation(dict(kind='independent_artifact_rest'),whole_task_slice=True):physical=rest_from_disk(dynamic,folder/'rest',config,budget)
        else:physical=rest_from_disk(dynamic,folder/'rest',config)
    else:physical=None
    auxiliary(static,folder/'views',budget,'render_export')
    initial=bool(static_geo['physical_valid'] and dynamic_geo['physical_valid'])
    if physical:initial &= all(x['initial_penetration_m']<=-config['initial_support']['gap_min_m'] for x in physical['objects'].values())
    source=read(run/'source_manifest.json');changed=[]
    for name,h in {**source.get('source_files',{}),**source.get('immutable_assets',{})}.items():
        if not Path(name).is_file() or sha(name)!=h:changed.append(name)
    observations_unchanged=all(sha(run/name)==h for name,h in read(run/'observation.json')['mask_sha256s'].items())
    observation_arrays_match=True
    for label in ('apple','plate','cloth'):
        path=run/'masks'/(label+'.png')
        if path.exists():observation_arrays_match &= np.array_equal(cv2.imread(str(path),0)>0,observation['target'][label])
    with np.load(run/'masks/observations.npz') as z:
        for label in ('valid','occlusion'):
            path=run/'masks'/(label+'.png')
            if path.exists() and label in z:observation_arrays_match &= np.array_equal(cv2.imread(str(path),0)>0,z[label])
    from ..vision import frame_at
    decoded_reference,_=frame_at(source['real_video_local'],0)
    real_reference_matches_source=bool(np.array_equal(decoded_reference,observation['rgb']))
    observations_unchanged &= observation_arrays_match and real_reference_matches_source
    task=f"{config['task_group_id']}/episode_{source['episode']:06}";ledger=ROOT/'runs/_task_budgets'/(config['task_group_id']+'.sqlite')
    with sqlite3.connect(ledger) as db:used=dict(db.execute('SELECT kind,SUM(amount) FROM events WHERE task=? GROUP BY kind',(task,)).fetchall())
    budget_passed=all(used.get(k,0)<=limit+1e-9 for k,limit in config['budget'].items())
    object_failures=[x for x in numeric['failures'] if not x.startswith('table.')]
    result=dict(version='disk_acceptance_v4',episode=source['episode'],candidate_id=read(run/'session.json')['candidate_id'],artifacts=artifacts,artifact_scene_hash=digest(artifacts),geometry=static_geo,dynamic_geometry=dynamic_geo,metrics=metrics,geometry_numeric_passed=not object_failures,coverage_numeric_passed=not any(x.startswith('table.') for x in numeric['failures']),initial_support_passed=initial,physical_passed=physical['physics_passed'] if physical else None,physical=physical,reload_passed=reload_passed,reload_errors=reload_errors,portability=portability,source_identity_passed=not changed,changed_sources=changed,observation_identity_passed=observations_unchanged,budget_passed=budget_passed,budget_used=used,numeric_failures=numeric['failures'],thresholds=threshold_audit(config),read_only_artifacts=True,checked_utc=now())
    result['engineering_passed']=bool(reload_passed and not changed and observations_unchanged and budget_passed)
    result['real_reference_matches_source']=real_reference_matches_source;result['target_arrays_match_saved_masks']=bool(observation_arrays_match)
    result['camera_verified_from_xml']=actual_camera
    result['validator_code']={name:sha(CODE_ROOT/'src/movens_sim/reconstruction'/name) for name in ('acceptance.py','diagnostics.py','backend.py')}
    result['numeric_quality_passed']=bool(result['engineering_passed'] and result['geometry_numeric_passed'] and result['coverage_numeric_passed'] and initial and result['physical_passed'])
    write(folder/'acceptance.json',result)
    return result,rgb,masks
