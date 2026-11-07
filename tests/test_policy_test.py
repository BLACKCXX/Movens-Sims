"""Behavioral checks for the single-case policy compatibility boundary."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from movens_sim.policy_test import geom_bounds, merge_robot, migrate_environment, video_schedule


def test_video_and_control_indices_use_distinct_time_grids():
    rows=video_schedule(.14,.2)
    assert rows[0]['source_frame_index'] == 1
    assert rows[0]['control_reference_index_50hz'] == 2
    assert rows[3]['source_frame_index'] == 4
    assert rows[4]['source_frame_index'] is None
    assert rows[4]['phase'] == 'tail_verification'


def test_joint_migration_uses_names_and_full_free_state():
    env = '<body name="apple"><freejoint name="apple_free"/><geom size=".03"/></body><body name="pad"><joint name="pad_z" type="slide"/><geom size=".01"/></body>'
    robot = '<body name="robot"><joint name="robot_joint"/><geom size=".02"/></body>'
    a = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+env+'</worldbody></mujoco>')
    b = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+robot+env+'</worldbody></mujoco>')
    da,db = mujoco.MjData(a),mujoco.MjData(b)
    da.qpos[:3] = [.3,.1,.8]
    da.qpos[7] = -.003
    da.qvel[:] = np.arange(a.nv)*.01
    db.qpos[0] = .25
    names = migrate_environment(a,da,b,db)
    assert names == ['apple_free','pad_z']
    assert db.qpos[0] == .25
    np.testing.assert_array_equal(db.qpos[1:],da.qpos)
    np.testing.assert_array_equal(db.qvel[1:],da.qvel)


def test_missing_joint_rejects_incompatible_state():
    a=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint name="pad"/><geom size=".01"/></body></worldbody></mujoco>')
    b=mujoco.MjModel.from_xml_string('<mujoco/>')
    with pytest.raises(ValueError,match='missing'):
        migrate_environment(a,mujoco.MjData(a),b,mujoco.MjData(b))


VERTICES = '-1 -1 -1  1 -1 -1  -1 1 -1  1 1 -1  -1 -1 1  1 -1 1  -1 1 1  1 1 1'


def test_mesh_bounds_and_robot_shared_assets_do_not_modify_environment(tmp_path: Path):
    env=tmp_path/'env.xml';robot=tmp_path/'robot.xml';out=tmp_path/'out.xml'
    env.write_text(f'<mujoco><asset><mesh name="shared" vertex="{VERTICES}" scale=".1 .2 .3"/></asset><worldbody><geom name="environment" type="mesh" mesh="shared" pos="1 2 3"/><camera name="approx_ego_camera" fovy="42"/></worldbody></mujoco>')
    robot.write_text(f'<mujoco><asset><mesh name="shared" vertex="{VERTICES}" scale=".5 .5 .5"/></asset><worldbody><body name="pelvis"><geom name="robot_geom" type="mesh" mesh="shared"/><camera name="approx_ego_camera" fovy="69"/></body></worldbody></mujoco>')
    merge_robot(env,robot,out,{'root':{'xyz_m':[0,0,.711],'quaternion_wxyz':[1,0,0,0]}})
    a=mujoco.MjModel.from_xml_path(str(env));b=mujoco.MjModel.from_xml_path(str(out))
    da,db=mujoco.MjData(a),mujoco.MjData(b)
    mujoco.mj_forward(a,da);mujoco.mj_forward(b,db)
    lo,hi=geom_bounds(b,db,b.geom('environment').id)
    np.testing.assert_allclose(lo,[.9,1.8,2.7],atol=1e-7)
    np.testing.assert_allclose(hi,[1.1,2.2,3.3],atol=1e-7)
    for x,y in zip(geom_bounds(a,da,0),(lo,hi)):
        np.testing.assert_array_equal(x,y)
    tree=ET.parse(out)
    assert len(tree.findall('.//camera')) == 1
    assert b.cam_fovy[b.camera('approx_ego_camera').id] == 42
    assert b.geom_dataid[b.geom('environment').id] != b.geom_dataid[b.geom('robot_geom').id]
    assert 'policy_g1_shared' in [b.mesh(i).name for i in range(b.nmesh)]
