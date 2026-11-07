#!/usr/bin/env python3
"""Render the actual portable frozen scene without source project, API or fitting."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import mujoco
import numpy as np
from movens_sim.render import Renderer,video_writer
from movens_sim.util import read,write,sha,save_rgb


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output-directory',required=True);a=p.parse_args();run=Path(a.run).resolve();out=Path(a.output_directory).resolve()
    if out==run:raise ValueError('Use a separate output directory; frozen delivery cannot be overwritten')
    out.mkdir(parents=True,exist_ok=True);xml=run/'delivery/static/portable/scene.xml';binding=read(run/'delivery_binding.json')
    for node in ET.parse(xml).findall('.//*[@file]'):
        file=Path(node.get('file'))
        if file.is_absolute() or not (xml.parent/file).resolve().is_relative_to(xml.parent):raise ValueError('Nonportable scene asset')
    model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model)
    with np.load(xml.with_suffix('.state.npz')) as z:data.qpos[:]=z['qpos'];data.qvel[:]=z['qvel']
    mujoco.mj_forward(model,data)
    with Renderer(model,640,480) as renderer:rgb=renderer.rgb(data)
    save_rgb(out/'ego.png',rgb);video=video_writer(out/'ego_static.mp4',(rgb for _ in range(150)),30)
    expected=cv2.cvtColor(cv2.imread(str(run/'final_ego.png')),cv2.COLOR_BGR2RGB);mae=float(np.abs(rgb.astype(float)-expected.astype(float)).mean())
    write(out/'replay_check.json',dict(scene_sha256=sha(xml),frozen_binding=binding['scene_parameter_asset_hash'],rgb_mae_vs_delivered=mae,video=video,no_optimization=True,no_model_request=True,no_integration=True,definition='Fixed-camera 3D static simulation; no real capture or interaction replay'))
    print('Frozen scene replayed; pixel MAE',mae,'video',out/'ego_static.mp4')


if __name__=='__main__':main()
