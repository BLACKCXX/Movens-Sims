import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from movens_sim.head_camera import free_objects
from movens_sim.reconstruction.backend import SceneBackend,SceneError
from movens_sim.reconstruction.budget import BudgetManager,BudgetExceeded
from movens_sim.util import ROOT,read


@pytest.fixture
def backend(toy_scene):
    mesh=toy_scene/'assets/support.obj'
    mesh.write_text('v -.2 -.6 .8\nv .8 -.6 .8\nv .8 .6 .8\nv -.2 .6 .8\nv .3 0 .81\nf 1 2 5\nf 2 3 5\nf 3 4 5\nf 4 1 5\nf 1 3 2\nf 1 4 3\n')
    path=toy_scene/'scene_fitting.xml';tree=ET.parse(path)
    ET.SubElement(tree.find('asset'),'mesh',name='support',file='assets/support.obj')
    ET.SubElement(tree.find('worldbody'),'geom',name='frozen_support',type='mesh',mesh='support',group='1',contype='0',conaffinity='0')
    # Exercise mesh sharing between different objects, not just independent files.
    for geom in tree.findall(".//body[@name='plate_root']//geom"):geom.set('mesh','apple_mesh')
    tree.write(path);free_objects(path)
    scene=SceneBackend(path)
    yield scene
    scene.close()


def test_absolute_scale_collision_support_mass_shared_and_rollback(backend):
    s=backend;initial=s.geometry_metrics();plate=s.vertices('plate').copy();camera=s.data.cam_xpos[s.cam].copy();old=s.params.copy()
    s.apply({'apple.scale':1.2});scaled=s.geometry_metrics();pts=s.vertices('apple').copy()
    np.testing.assert_allclose(np.array(scaled['apple']['dimensions_world_m'])/initial['apple']['dimensions_world_m'],1.2,rtol=1e-5)
    np.testing.assert_allclose(np.ptp(s.vertices('apple',True),axis=0)/np.ptp(np.array(initial['apple']['aabb_world_m']),axis=0),1.2,rtol=1e-5)
    assert abs(scaled['apple']['mass_kg']/initial['apple']['mass_kg']-1.2**3)<1e-7
    assert abs(scaled['apple']['signed_support_gap_m'])<1e-10
    s.apply({'apple.scale':1.2});np.testing.assert_array_equal(s.vertices('apple'),pts)
    np.testing.assert_array_equal(s.vertices('plate'),plate);np.testing.assert_array_equal(s.data.cam_xpos[s.cam],camera)
    accepted=s.params.copy();before=s.vertices('apple').copy()
    for update in [{'apple.x':.799},{'apple.scale':float('nan')},{'plate.scale':2.01},{'camera.z':1.}]:
        with pytest.raises(SceneError):s.apply(update)
        assert s.params==accepted;np.testing.assert_array_equal(s.vertices('apple'),before)
    s.apply({'apple.scale':old['apple.scale']});np.testing.assert_allclose(s.geometry_metrics()['apple']['dimensions_world_m'],initial['apple']['dimensions_world_m'],rtol=1e-5)


def test_portable_export_reload_geometry_camera_and_pixels(backend,tmp_path):
    s=backend;s.apply({'plate.scale':1.15,'apple.rx':12.})
    rgb,masks=s.render();path=s.export(tmp_path/'delivered')
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
    with np.load(path.with_suffix('.state.npz')) as z:data.qpos[:]=z['qpos']
    mujoco.mj_forward(model,data)
    from movens_sim.render import Renderer
    with Renderer(model) as renderer:other=renderer.rgb(data)
    np.testing.assert_allclose(other,rgb,atol=0)
    np.testing.assert_allclose(data.cam_xpos,s.data.cam_xpos,atol=1e-12)
    for label in ('apple','plate'):np.testing.assert_allclose(data.xpos[model.body(label+'_root').id],s.data.xpos[s.model.body(label+'_root').id],atol=1e-12)


def test_stage_permissions_shape_capability_and_frozen_scene(backend):
    s=backend
    with pytest.raises(SceneError):s.apply({'light.ambient':.4})
    with pytest.raises(SceneError):s.apply({'plate.depth_ratio':.9})
    s.stage='appearance'
    with pytest.raises(SceneError):s.apply({'plate.scale':1.1})
    s.apply({'light.ambient':.4});s.stage='frozen'
    with pytest.raises(SceneError):s.apply({'light.ambient':.3})


def test_global_budget_failures_resumption_and_terminal_reservation(tmp_path):
    limits=read(ROOT/'configs/mount_joint_v3.json')['budget'];path=tmp_path/'budget.sqlite';b=BudgetManager(path,'same-task',limits)
    resumed=BudgetManager(path,'same-task',limits)
    for i in range(4):resumed.reserve('geometry_rounds')
    with pytest.raises(BudgetExceeded):resumed.reserve('geometry_rounds')
    for i in range(600):resumed.reserve_render('render_initial',{'candidate':i})
    with pytest.raises(BudgetExceeded):resumed.reserve_render('render_initial',{})
    assert resumed.summary()['used']['render_total']==600
    for i in range(400):resumed.reserve_render('render_repair',{})
    for i in range(200):resumed.reserve_render('render_appearance',{})
    with pytest.raises(BudgetExceeded):resumed.reserve_render('render_appearance',{})
    for i in range(4):b.reserve_terra(detail={'attempt':i})
    with pytest.raises(BudgetExceeded):b.reserve_terra()
    resumed.reserve_terra(final=True)
    with pytest.raises(BudgetExceeded):resumed.reserve_terra(final=True)
    assert resumed.summary()['used']['terra_requests']==5
    altered=limits.copy();altered['terra_requests']=6
    with pytest.raises(ValueError):BudgetManager(path,'same-task',altered)


def test_unknown_final_retry_is_same_frozen_candidate_and_cannot_reopen_geometry(tmp_path):
    limits=read(ROOT/'configs/mount_joint_v3.json')['budget'];b=BudgetManager(tmp_path/'retry.sqlite','terminal-task',limits)
    b.reserve_terra(detail={'candidate_id':'a'})
    b.reserve_terra(final=True,detail={'candidate_id':'a','delivery_hash':'frozen'})
    with pytest.raises(BudgetExceeded):b.reserve_terra(detail={'candidate_id':'a'})
    with pytest.raises(BudgetExceeded):b.reserve_terra(final=True,detail={'candidate_id':'different','delivery_hash':'frozen'})
    with pytest.raises(BudgetExceeded):b.reserve_render('render_repair',{})
    with pytest.raises(BudgetExceeded):b.reserve('geometry_rounds')
    b.reserve_terra(final=True,detail={'candidate_id':'a','delivery_hash':'frozen'})
    assert b.used('terra_requests')==3


def test_nonfinite_budget_reservation_never_creates_spendable_credit(tmp_path):
    limits=read(ROOT/'configs/mount_joint_v3.json')['budget'];b=BudgetManager(tmp_path/'finite.sqlite','finite',limits)
    for amount in [float('nan'),float('inf'),-1,True]:
        with pytest.raises(ValueError):b.reserve('render_total',amount)
    assert b.used('render_total')==0


def test_active_time_is_nested_without_double_count_and_interrupts_optimizer_work(tmp_path):
    import time
    limits=read(ROOT/'configs/mount_joint_v3.json')['budget'];limits['optimization_seconds']=.08;b=BudgetManager(tmp_path/'time.sqlite','time',limits)
    with b.computation(whole_task_slice=True):
        with b.computation():time.sleep(.02)
    assert .015<b.used('optimization_seconds')<.05
    with pytest.raises(BudgetExceeded):
        with b.computation(whole_task_slice=True):time.sleep(.2)
    assert .07<b.used('optimization_seconds')<.1
