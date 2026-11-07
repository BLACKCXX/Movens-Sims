import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.spatial.transform import Rotation

from movens_sim.evaluate import whole_scene_metrics
from movens_sim.head_camera import free_objects, LayoutFit, systematic_bias
from movens_sim.render import Renderer
from movens_sim.robot_camera import camera_transform, observed_joint_values
from movens_sim.scene import FixedGeometryScene
from movens_sim.util import ROOT,read,write,save_rgb


@pytest.fixture
def fixed_scene(toy_scene):
    mesh=toy_scene/'assets/support.obj'
    mesh.write_text('v -.2 -.6 .8\nv .8 -.6 .8\nv .8 .6 .8\nv -.2 .6 .8\nv .3 0 .81\nf 1 2 5\nf 2 3 5\nf 3 4 5\nf 4 1 5\nf 1 3 2\nf 1 4 3\n')
    path=toy_scene/'scene_fitting.xml';tree=ET.parse(path)
    ET.SubElement(tree.find('asset'),'mesh',name='support',file='assets/support.obj')
    ET.SubElement(tree.find('worldbody'),'geom',name='frozen_pad_other',type='mesh',mesh='support',group='1',contype='0',conaffinity='0')
    tree.write(path);free_objects(path)
    scene=FixedGeometryScene(path);scene.registry();return scene


def test_robot_mount_optical_axis_and_local_world_composition():
    p=np.array([.1,.2,.7]);r=Rotation.from_euler('zyx',[12,-8,3],degrees=True).as_matrix();t=np.array([.057,.017,.43]);offset=np.array([.01,0,0])
    camera=camera_transform(p,r,t,47.6,offset);mount=r@Rotation.from_euler('y',47.6,degrees=True).as_matrix()
    np.testing.assert_allclose(camera['forward_world'],mount[:,0],atol=1e-12)
    np.testing.assert_allclose(camera['xyz_m'],p+r@t+mount@offset,atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(camera['rotation_camera_to_world']),1.,atol=1e-12)
    assert not np.isclose(camera['down_angle_deg'],47.6)


def test_parquet_reads_observation_and_checks_alignment(tmp_path):
    path=tmp_path/'raw.parquet';state=np.arange(43,dtype=float)/100
    pq.write_table(pa.table({'frame_index':[0],'timestamp':[0.0],'observation.state':[state.tolist()],'action':[(-state).tolist()]}),path)
    mapping={'source_body_names':['body'+str(i) for i in range(29)],'hand_state_to_canonical':list(reversed(range(14)))}
    robot={'hand_joint_names':['hand'+str(i) for i in range(14)]}
    values,row=observed_joint_values(path,mapping,robot,0,0.,.001)
    assert values['body12']==state[12] and values['hand0']==state[42] and row['timestamp']==0
    with pytest.raises(ValueError):observed_joint_values(path,mapping,robot,0,.2,.001)


def test_fixed_camera_local_support_pose_bounds_and_reload(fixed_scene,tmp_path):
    s=fixed_scene;prior=camera_transform(np.array([.0,-.5,1.2]),np.eye(3),np.zeros(3),40,np.zeros(3));s.set_camera(prior,69)
    camera=s.data.cam_xpos[s.cam].copy();s.apply({'apple.x':.35,'plate.x':.5});z1=s.data.xpos[s.apple,2]
    s.apply({'apple.x':.6});z2=s.data.xpos[s.apple,2]
    assert abs(z1-z2)>.0001
    np.testing.assert_array_equal(s.data.cam_xpos[s.cam],camera)
    before=s.data.qpos.copy();params=s.params.copy()
    for bad in [{'camera.z':1.},{'apple.x':float('nan')},{'plate.x':9.}]:
        with pytest.raises(ValueError):s.apply(bad)
        np.testing.assert_array_equal(s.data.qpos,before);assert s.params==params
    xml=s.export(tmp_path/'portable');model=mujoco.MjModel.from_xml_path(str(xml));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    np.testing.assert_allclose(data.cam_xpos[0],camera,atol=1e-12)
    for label in ('apple','plate'):np.testing.assert_allclose(data.xpos[model.body(label+'_root').id],s.data.xpos[s.model.body(label+'_root').id],atol=1e-12)


def test_whole_scene_scoring_includes_top_without_exclusion():
    valid=np.ones((120,160),bool);apple=np.zeros_like(valid);apple[70:85,20:35]=1;plate=np.zeros_like(valid);plate[70:100,90:130]=1;cloth=~(apple|plate)
    target=dict(apple=apple,plate=plate,cloth=cloth);pred={k:v.copy() for k,v in target.items()};config=read(ROOT/'configs/head_camera_pilot_v1.json')
    same=whole_scene_metrics(pred,target,valid,config);pred['cloth'][:9]=False;bad=whole_scene_metrics(pred,target,valid,config)
    assert same['table']['top_cloth_iou']==1 and bad['table']['top_cloth_iou']<.7
    assert not bad['table']['top_excluded'] and bad['image_loss']>same['image_loss']


def test_shared_size_bias_triggers_even_after_centers_fit():
    config=read(ROOT/'configs/head_camera_pilot_v1.json');objects={k:dict(centroid_residual_xy_px=[0.,0.],target_visible_pixels=100,render_visible_pixels=60) for k in ('apple','plate')}
    assert systematic_bias({'metrics':{'objects':objects}},config)
    objects['apple']['render_visible_pixels']=150
    assert not systematic_bias({'metrics':{'objects':objects}},config)


def test_mount_norm_constraint_and_invalid_candidate_rollback(fixed_scene,tmp_path):
    s=fixed_scene;run=tmp_path/'run';(run/'masks').mkdir(parents=True)
    prior=camera_transform(np.array([.0,0.,1.2]),np.eye(3),np.zeros(3),47.6,np.zeros(3));s.set_camera(prior,69);s.apply(s.params)
    with Renderer(s.model,320,240) as renderer:masks=renderer.masks(s.data,s.geom_sets)
    np.savez_compressed(run/'masks/observations.npz',apple=masks['apple'],plate=masks['plate'],cloth=masks['cloth'],valid=np.ones((240,320),bool))
    config=read(ROOT/'configs/head_camera_pilot_v1.json');config['fit']['max_evaluations']=2
    fitter=LayoutFit(s,prior,69,run/'candidate',run,config,refine=True)
    try:
        p=fitter.best.copy();p.update({'mount.dx':.02,'mount.dy':.02})
        with pytest.raises(ValueError):fitter.camera(p)
        fitter.evaluate(fitter.best,'baseline');before=fitter.best.copy();fitter.evaluate(p,'invalid_norm')
        assert fitter.best==before
        with pytest.raises(StopIteration):fitter.evaluate(before,'budget')
    finally:fitter.close()
    resumed=LayoutFit(s,prior,69,run/'candidate',run,config,refine=True,resume=True)
    try:
        assert resumed.evaluations==2 and resumed.best==before
        with pytest.raises(StopIteration):resumed.evaluate(resumed.best,'interrupted_budget_recovery')
    finally:resumed.close()
