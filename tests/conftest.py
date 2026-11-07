import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID','0')
import pytest
from movens_sim.util import write

@pytest.fixture
def toy_scene(tmp_path):
    folder=tmp_path/'scene';folder.mkdir();assets=folder/'assets';assets.mkdir()
    (assets/'tetra.obj').write_text('v -.04 -.04 -.04\nv .04 -.04 -.04\nv 0 .04 -.04\nv 0 0 .05\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n')
    xml='''<mujoco><compiler angle="radian"/><option gravity="0 0 -9.81"/>
    <asset><mesh name="apple_mesh" file="assets/tetra.obj"/><mesh name="plate_mesh" file="assets/tetra.obj"/><material name="plate_mat" rgba=".8 .8 .8 1"/></asset>
    <worldbody><camera name="approx_ego_camera" pos=".35 -.5 1.1" xyaxes="1 0 0 0 .5 .8660254" fovy="69"/>
    <body name="table_root" pos="0 0 .7"><geom name="table_box" type="box" size="1 1 .1" group="0"/></body>
    <body name="apple_root" pos=".4 .08 .84"><freejoint name="apple_freejoint"/><geom name="apple_vis" type="mesh" mesh="apple_mesh" group="1" contype="0" conaffinity="0"/><geom name="apple_col" type="mesh" mesh="apple_mesh" group="0"/></body>
    <body name="plate_root" pos=".42 -.15 .84"><geom name="plate_vis" type="mesh" mesh="plate_mesh" group="1" material="plate_mat" contype="0" conaffinity="0"/><geom name="plate_col" type="mesh" mesh="plate_mesh" group="0"/></body></worldbody></mujoco>'''
    import mujoco
    raw=folder/'source.xml';raw.write_text(xml);spec=mujoco.MjSpec.from_file(str(raw));spec.compile();canonical=spec.to_xml()
    # Scene consumes the prepared/canonical MJCF stage, as in the real pipeline.
    for name in ['scene_fitting.xml','scene_baseline.xml','scene_physics_baseline.xml']:(folder/name).write_text(canonical)
    write(folder/'scene_context.json',dict(table_top_z_m=.8,cloth_continuation=dict(outer_xy_m=[[-1,-1],[1,1]],inner_xy_m=[[-.5,-.5],[.5,.5]])))
    return folder
