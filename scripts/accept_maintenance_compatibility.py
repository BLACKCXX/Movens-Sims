#!/usr/bin/env python3
"""One explicit, checked v0.1.0 -> v0.1.1 compatibility upgrade for completed batch v1.

This never adopts an incompatible input/config/prompt or an unfinished result.
Original execution identities and receipts remain in compatibility history.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID','0')
import shutil
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from movens_sim.scene import Scene
from movens_sim.render import Renderer
from movens_sim.pipeline import fingerprint,receipt_valid,finalize,report,STAGES
from movens_sim.util import ROOT,read,write,sha,digest,now

config=read(ROOT/'configs/batch_v1.json');manifest=read(ROOT/config['manifest']);root=ROOT/config['run_root'];records=[]
snapshot=root/'code_executed_v0.1.0';recorded=read(snapshot/'snapshot.json')['code_hashes']
for filename,expected in recorded.items():assert sha(snapshot/filename)==expected
for entry in manifest['entries']:
    run=root/f"episode_{entry['episode']:06}";imported=ROOT/'data'/run.name;old=read(run/'run_identity.json');new=fingerprint(imported,config,True)
    if old==new:continue
    previous_path=run/'compatibility/maintenance_v0.1.1.json'
    previous=read(previous_path) if previous_path.exists() else None
    assert old['code_hashes']==recorded or (previous and previous['new_resume_identity']==old),'Unexpected execution/resume version'
    for key,value in old.items():
        if key!='code_hashes':assert new[key]==value,f'Incompatible {key}'
    assert sha(snapshot/'configs/manifest.json')==sha(ROOT/config['manifest'])
    assert read(run/'result.json')['processing_complete'] and all(receipt_valid(run,s,old) for s in STAGES)
    assert read(run/'delivery_checks.json')['passed']
    # The corrected auto-inertia same-frame case is absent in every actual source.
    m=mujoco.MjModel.from_xml_path(str(run/'scene/scene_fitting.xml'));bodies={m.body('apple_root').id}
    for b in range(m.nbody):
        if int(m.body_parentid[b]) in bodies:bodies.add(b)
    apple_geoms=np.where(np.isin(m.geom_bodyid,list(bodies)))[0];assert np.all(m.geom_sameframe[apple_geoms]==0)
    s=Scene(run/'scene/scene_fitting.xml');s.apply(read(run/'parameters_optimized.json'))
    with Renderer(s.model,640,480) as renderer:rgb=renderer.rgb(s.data)
    expected=np.array(Image.open(run/'optimized_ego.png'));mae=float(np.abs(rgb.astype(float)-expected.astype(float)).mean());assert mae<.01
    assert read(run/'fit_final_state.json')['evaluations']<=config['fit']['max_evaluations']
    history=run/'compatibility'/('v0.1.0_execution' if not previous else 'maintenance_'+digest(old)[:12]);history.mkdir(parents=True,exist_ok=True)
    if previous:shutil.copy2(previous_path,history/'previous_maintenance.json')
    shutil.copy2(run/'run_identity.json',history/'run_identity.json');shutil.copytree(run/'receipts',history/'receipts',dirs_exist_ok=True);shutil.copy2(run/'result.json',history/'result.json')
    record=dict(version='maintenance_compatibility_v1',utc=now(),old_execution_identity=old,new_resume_identity=new,validated_static_render_mae=mae,actual_mesh_sameframe_flags_all_zero=True,existing_inputs_config_prompt_unchanged=True,original_artifacts_retained=True,changes=['Restore finite best_loss and conservative render count on resume','Disable stale same-frame mesh shortcuts for auto-inertia scenes (not present in these five)','Persist exact GPT input images, HTTP attempt budget and structured response cache','Retain read-only GPT advice at exhausted numeric budget','Check frozen manifest identity and immutable candidate mapping'],no_numerical_or_physical_rerun_claimed=True,script_sha256=sha(__file__))
    write(run/'compatibility/maintenance_v0.1.1.json',record);write(run/'run_identity.json',new)
    for stage in STAGES:
        path=run/'receipts'/(stage+'.json');r=read(path);r['identity_sha256']=digest(new);r['compatibility_upgrade']='compatibility/maintenance_v0.1.1.json';write(path,r)
    result=finalize(run,entry,config,new);result['execution_versions']=dict(numerical_fit_and_gravity='code_executed_v0.1.0',resume_and_maintenance='v0.1.1; checked compatible with saved outputs',prompt_fit='diagnostic_v1',prompt_readonly_review='diagnostic_v2');write(run/'result.json',result);records.append(dict(episode=entry['episode'],compatible=True,render_mae=mae))
write(root/'maintenance_compatibility.json',dict(upgrades=records,original_execution_snapshot='code_executed_v0.1.0',purpose='Explicit verified compatibility upgrade; no silent reuse of changed or incompatible completion'))
print(report(manifest,config),flush=True)
