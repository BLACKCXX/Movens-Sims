#!/usr/bin/env python3
"""Reproduce uploaded PNG bytes from each logged state and verify request hashes."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID','0')
from pathlib import Path
import shutil
from movens_sim.scene import Scene
from movens_sim.render import Renderer,overlay
from movens_sim.vision import load_observation
from movens_sim.util import ROOT,read,write,save_rgb,sha

root=ROOT/'runs/batch_v1';results=[]
for run in sorted(root.glob('episode_*')):
    if not (run/'result.json').exists():continue
    reference,target,valid=load_observation(run)
    scene=Scene(run/'scene/scene_fitting.xml')
    for request in sorted((run/'gpt').glob('round_*/request.json')):
        log=read(request);state=log['state'];folder=request.parent;scene.apply(state['current_params'])
        with Renderer(scene.model,reference.shape[1],reference.shape[0]) as renderer:rgb=renderer.rgb(scene.data);masks=renderer.masks(scene.data,scene.geom_sets)
        for filename,image in [('reference.png',reference),('render.png',rgb),('overlay.png',overlay(reference,masks,target))]:save_rgb(folder/filename,image)
        archived=[dict(path=name,actual_sha256=sha(folder/name),request_sha256=expected['sha256'],exact_match=sha(folder/name)==expected['sha256']) for name,expected in zip(['reference.png','render.png','overlay.png'],state['images'])]
        record=dict(request=str(request.relative_to(ROOT)),images=archived,reconstruction='Regenerated from saved request parameters and immutable source geometry; SHA256 proves exact uploaded PNG identity when exact_match is true')
        write(folder/'image_archive.json',record);results.append(record)
write(root/'gpt_image_archive_checks.json',dict(all_exact=all(i['exact_match'] for r in results for i in r['images']),requests=results))
print('Requests archived',len(results),'all exact',all(i['exact_match'] for r in results for i in r['images']),flush=True)
