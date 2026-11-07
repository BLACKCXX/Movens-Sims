#!/usr/bin/env python3
"""Independent read-only checks of frozen v3 artifacts, budgets and protected inputs."""
import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

from movens_sim.render import Renderer
from movens_sim.reconstruction.workflow import groups
from movens_sim.util import ROOT,read,write,sha,digest,video_info,now


def check(run):
    run=Path(run).resolve();binding=read(run/'delivery_binding.json');checks={}
    changed=[name for name,expected in binding['artifacts'].items() if not (run/name).is_file() or sha(run/name)!=expected]
    checks['frozen_binding']=dict(passed=not changed,changed=changed,hash=binding['scene_parameter_asset_hash'])
    portability={}
    with tempfile.TemporaryDirectory(prefix='movens_v3_independent_') as temp:
        for label,relative in [('static','delivery/static/portable'),('physics','physics/portable')]:
            source=run/relative;target=Path(temp)/label;shutil.copytree(source,target);xml=target/'scene.xml';root=ET.parse(xml)
            refs=[]
            for node in root.findall('.//*[@file]'):
                ref=Path(node.get('file'));resolved=(xml.parent/ref).resolve()
                refs.append(dict(path=str(ref),local=not ref.is_absolute() and resolved.is_relative_to(target),exists=resolved.is_file()))
            if not all(r['local'] and r['exists'] for r in refs):raise ValueError('External resource in '+label)
            model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
            with np.load(xml.with_suffix('.state.npz')) as z:data.qpos[:]=z['qpos'];data.qvel[:]=z['qvel']
            mujoco.mj_forward(model,data);cam=model.camera('approx_ego_camera').id
            result=dict(loaded_from_independent_directory=True,resources=len(refs),all_resources_local=True,qpos_finite=bool(np.isfinite(data.qpos).all()),camera_attached_to_world=int(model.cam_bodyid[cam])==0,apple_free=bool(model.body_jntnum[model.body('apple_root').id]),plate_free=bool(model.body_jntnum[model.body('plate_root').id]),robot_removed='pelvis' not in [model.body(i).name for i in range(model.nbody)])
            if label=='static':
                with Renderer(model) as renderer:rgb=renderer.rgb(data);masks=renderer.masks(data,groups(model))
                expected=cv2.cvtColor(cv2.imread(str(run/'final_ego.png')),cv2.COLOR_BGR2RGB);result['rgb_mae']=float(np.abs(rgb.astype(float)-expected.astype(float)).mean())
                with np.load(run/'final_masks.npz') as z:result['mask_changed_pixels']={k:int(np.count_nonzero(masks[k]!=z[k])) for k in ('apple','plate','cloth')}
                result['passed']=result['rgb_mae']<.05 and max(result['mask_changed_pixels'].values())<=4
            else:result['passed']=True
            result['passed']&=all(result[k] for k in ('all_resources_local','qpos_finite','camera_attached_to_world','apple_free','plate_free','robot_removed'));portability[label]=result
    checks['portability']=portability
    checks['videos']={}
    for filename,expected_frames in [('ego_static.mp4',150),('settle_diagnostic.mp4',30)]:
        info=video_info(run/filename);checks['videos'][filename]=dict(**info,passed=info['frames']==expected_frames and info['width']==640 and info['height']==480 and abs(info['fps']-30)<1e-9)
    review=read(run/'final_review.json');request=read(run/review['status']['folder']/'request.json')
    checks['final_review_binding']=dict(passed=review['delivery_hash']==binding['scene_parameter_asset_hash']==request['state']['delivery_binding']['scene_parameter_asset_hash'],model=review['status'].get('response_model',review['status']['model']),verdict=review['review']['verdict'] if review['review'] else 'unreviewed',api_response_validated=review['status']['status']=='validated',review_is_terminal=not review['review']['proposed_actions'] if review['review'] else True)
    config=read(run/'resolved_config.json');path=ROOT/'runs/_task_budgets'/(config['task_group_id']+'.sqlite')
    with sqlite3.connect(path) as db:
        used=dict(db.execute('select kind,sum(amount) from events where task=? group by kind',(config['task_group_id'],)).fetchall())
        render_reserved=db.execute("select count(*) from events where task=? and kind='render_total' and status='reserved'",(config['task_group_id'],)).fetchone()[0]
    limits=config['budget'];attempts=[read(p) for p in (run/'terra').glob('*/attempt_*/status.json')]
    http=sum(p.get('actual_http_attempts',int(p.get('actual_http_request_started',False))) for p in attempts)
    checks['budget']=dict(used={k:v for k,v in used.items() if v},limits=limits,passed=all(used.get(k,0)<=v for k,v in limits.items()),actual_http_attempts=http,actual_ledger_agree=http==used.get('terra_requests',0),unfinished_render_reservations=render_reserved,validated_responses=sum(p.get('status')=='validated' for p in attempts),unknown_responses=sum(p.get('status')=='interrupted_unknown' for p in attempts))
    inventory=read(ROOT/'runs/head_camera_pilot_v1/episode_000000/protected_batch_hashes.json');old_changed=[name for name,expected in inventory.items() if not (ROOT/name).is_file() or sha(ROOT/name)!=expected]
    checks['protected_batch']=dict(files=len(inventory),passed=not old_changed,changed=old_changed)
    source=read(ROOT/f"data/episode_{read(run/'source_manifest.json')['episode']:06}/source_manifest.json")
    closure_changed=[row['source'] for row in source['resource_closure'] if not Path(row['source']).is_file() or sha(row['source'])!=row['source_sha256']]
    checks['readonly_original_assets']=dict(files=len(source['resource_closure']),passed=not closure_changed,changed=closure_changed)
    frozen_masks=read(run/'observation.json')['mask_sha256s'];checks['observation']=dict(passed=all(sha(run/name)==expected for name,expected in frozen_masks.items()),version=read(run/'observation.json')['version'])
    checks['passed']=bool(checks['frozen_binding']['passed'] and all(x['passed'] for x in portability.values()) and all(x['passed'] for x in checks['videos'].values()) and checks['final_review_binding']['passed'] and checks['budget']['passed'] and checks['budget']['actual_ledger_agree'] and checks['protected_batch']['passed'] and checks['readonly_original_assets']['passed'] and checks['observation']['passed'])
    checks['checked_utc']=now();checks['interpretation']='Delivery/integrity check only, not visual quality or validated real dynamics. No parameter changes or new optimization/API requests.'
    write(run/'delivery_checks.json',checks);return checks


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();checks=check(args.run);print(json.dumps(checks,ensure_ascii=False,indent=2));raise SystemExit(0 if checks['passed'] else 1)
