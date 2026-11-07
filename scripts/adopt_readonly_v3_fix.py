#!/usr/bin/env python3
"""Explicit audited code continuation for an already frozen v3 delivery.

Never adopts changed geometry/render/observation code, configuration or assets.
Never clears a phase, counter, old identity or executed-source snapshot.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import cv2
import mujoco
import numpy as np

from movens_sim.reconstruction.workflow import identity,verify_sources
from movens_sim.reconstruction.session import ReconstructionSession
from movens_sim.render import Renderer
from movens_sim.util import ROOT,read,write,sha,digest,now


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);args=p.parse_args();run=Path(args.run).resolve()
    config=read(run/'resolved_config.json');old=read(run/'run_identity.json');new=identity(run,config)
    if old==new:print('Already matches current code; no changes');return
    state=read(run/'session.json')
    if state['stage']!='final_readonly' or not state.get('geometry_stage_complete'):raise ValueError('Only terminal frozen state can adopt this read-only fix')
    for key in ('config_sha256','source_manifest_sha256','observation_sha256','mujoco'):
        if old[key]!=new[key]:raise ValueError('Not a code-only compatible continuation: '+key)
    changed={name for name,value in new['code'].items() if old['code'].get(name)!=value}
    permitted={'src/movens_sim/reconstruction/'+name+'.py' for name in ('budget','session','mcp_server','terra','workflow')}
    if changed-permitted:raise ValueError('Geometry/renderer/input code changed; cannot adopt: '+str(changed-permitted))
    verify_sources(run);binding=read(run/'delivery_binding.json')
    for name,value in binding['artifacts'].items():
        if sha(run/name)!=value:raise ValueError('Frozen artifact changed: '+name)
    # The actual delivered model is rendered independently, without selecting a
    # candidate or updating any scene parameter. This is a read-only reload check.
    xml=run/'delivery/static/portable/scene.xml';model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
    with np.load(xml.with_suffix('.state.npz')) as z:data.qpos[:]=z['qpos'];data.qvel[:]=z['qvel']
    mujoco.mj_forward(model,data)
    with Renderer(model) as renderer:rgb=renderer.rgb(data)
    reference=cv2.cvtColor(cv2.imread(str(run/'final_ego.png')),cv2.COLOR_BGR2RGB)
    mae=float(np.abs(rgb.astype(float)-reference.astype(float)).mean())
    if mae>.05:raise ValueError('Read-only delivery render differs')
    folder=run/'code_compatibility'/str(time.time_ns());folder.mkdir(parents=True)
    write(folder/'run_identity_before.json',old);write(folder/'session_before.json',state)
    for name in new['code']:
        dest=folder/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    write(folder/'adoption.json',dict(created_utc=now(),changed_files=sorted(changed),reason='Fix evidence-to-plate-tilt permission routing and interrupted-final logging, with frozen-scene-only retry and stage entry guards. No geometry rerun or parameter update permitted for this already terminal trial.',render_mae=mae,frozen_delivery_hash=binding['scene_parameter_asset_hash'],script_sha256=sha(__file__),budget_reset=False,old_identity_sha256=digest(old),new_identity_sha256=digest(new)))
    state['maintenance_notes']=['The actual directed orientation repair produced zero new renders because the executed router omitted permission activation. Fixed and regression-tested in current source; this already frozen trial is not refit.','An interrupted final request is conservatively counted. Only an identical-candidate final read-only retry is allowed, within the original five-request cap.']
    state['stop_reason']='orientation_repair_blocked_by_executed_router_bug_and_terminal_scene_preserved'
    write(run/'session.json',state);write(run/'run_identity.json',new)
    print('Adopted explicit compatible read-only maintenance; frozen RGB MAE',mae,'budget unchanged')


if __name__=='__main__':main()
