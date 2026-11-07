#!/usr/bin/env python3
"""Independent delivery checks, including load after copying away from source paths."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID','0')
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np
from movens_sim.render import Renderer
from movens_sim.evaluate import image_metrics
from movens_sim.vision import load_observation,frame_at,segment
from movens_sim.util import ROOT,read,write,sha,video_info,save_rgb

def check(run):
    run=Path(run);checks={};selection=read(run/'reference_selection.json');expected=read(run/'metrics_after.json')
    with tempfile.TemporaryDirectory(prefix='movens_sim_portable_') as tmp:
        dest=Path(tmp)/'scene';shutil.copytree(run/'scene',dest)
        for filename in ['scene_optimized.xml','scene_physics.xml']:
            tree=ET.parse(dest/filename)
            for node in tree.iter():
                for attr in ['file','fileleft','fileright','fileup','filedown','filefront','fileback']:
                    if node.get(attr):
                        path=Path(node.get(attr));assert not path.is_absolute(),str(path);assert (dest/path).is_file()
            m=mujoco.MjModel.from_xml_path(str(dest/filename));d=mujoco.MjData(m)
            with np.load((dest/filename).with_suffix('.state.npz')) as z:d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel']
            mujoco.mj_forward(m,d);assert m.nu==0 and m.nsensor==0;assert mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'pelvis')<0
            assert m.jnt_type[m.joint('apple_freejoint').id]==mujoco.mjtJoint.mjJNT_FREE
            checks[filename]=dict(portable_compile=True,nq=m.nq,nv=m.nv,nflex=m.nflex,robot_removed=True)
            if filename=='scene_optimized.xml':
                geoms={}
                for label in ('apple','plate','table'):
                    bodies={m.body(label+'_root').id}
                    for b in range(m.nbody):
                        if int(m.body_parentid[b]) in bodies:bodies.add(b)
                    geoms[label]=np.where(np.isin(m.geom_bodyid,list(bodies)))[0]
                geoms['cloth']=[g for g in range(m.ngeom) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '').startswith('frozen_') or mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g)=='cloth_continuation']
                rgb,target,valid=load_observation(run)
                with Renderer(m,selection['width'],selection['height']) as r:rendered=r.rgb(d);masks=r.masks(d,geoms)
                save_rgb(run/'export_reload.png',rendered);metrics=image_metrics(masks,target,valid)
                checks['reloaded_metric_max_iou_difference']=max(abs(metrics['objects'][k]['iou']-expected['objects'][k]['iou']) for k in target)
                assert checks['reloaded_metric_max_iou_difference']<.002
                from PIL import Image
                original=np.array(Image.open(run/'optimized_ego.png'))
                checks['reloaded_render_mae_0_255']=float(np.abs(original.astype(float)-rendered.astype(float)).mean());assert checks['reloaded_render_mae_0_255']<.1
    for filename,frames in [('ego_static.mp4',150),('settle_diagnostic.mp4',30)]:
        info=video_info(run/filename);assert info['frames']==frames and info['fps']==30 and info['width']==selection['width'] and info['height']==selection['height'];checks[filename]=info
    resource=read(run/'resource_checks.json');assert resource['camera_world_position_error_m']<1e-9 and resource['camera_world_rotation_matrix_error']<1e-9
    checks['camera_world_transform_correct']=True
    imported=ROOT/'data'/run.name;manifest=read(imported/'source_manifest.json')
    for row in manifest['resource_closure']:
        assert sha(imported/'source'/row['portable'])==row['portable_sha256']
        assert sha(row['source'])==row['source_sha256']
    checks['source_assets_unchanged_and_imports_intact']=True
    # Independent pixel consistency on unused early frames, with motion caveats.
    heldout=[]
    for index in (1,3,6):
        frame,fps=frame_at(imported/'real_ego.mp4',index);target=segment(frame);met=image_metrics(masks,target,valid)
        heldout.append(dict(frame_index=index,timestamp_s=index/fps,metrics=met,camera_assumption='Same fitted camera for image-consistency check only; any early camera/scene drift is not independently resolved, so this is not multi-view 3D validation'))
    write(run/'unused_initial_frame_checks.json',heldout)
    checks['passed']=True;write(run/'delivery_checks.json',checks);return checks

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-root',default='runs/batch_v1');args=p.parse_args();root=ROOT/args.run_root;results={}
    for run in sorted(root.glob('episode_*')):
        if (run/'result.json').exists():
            try:results[run.name]=check(run);print(run.name,'passed',flush=True)
            except Exception as e:results[run.name]=dict(passed=False,error_type=type(e).__name__,message=str(e));print(run.name,'FAILED',type(e).__name__,str(e),flush=True)
    write(root/'delivery_checks.json',dict(passed=bool(results) and all(x.get('passed') for x in results.values()),episodes=results))
    if not all(x.get('passed') for x in results.values()):raise SystemExit(1)
