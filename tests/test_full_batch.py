"""Behavioral checks for h0 construction and durable batch controls, no API."""
import concurrent.futures
import json
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET
import cv2
import mujoco
import numpy as np
import pytest

from movens_sim.reconstruction.full_prepare import shift_to_h0,_pixels
from movens_sim.reconstruction.backend import SupportSurface
from movens_sim.reconstruction.budget import BudgetManager,BudgetExceeded
from movens_sim.reconstruction.control import BatchPaused,check_stop,api_slot
from movens_sim.full_batch import connect,publish,process_stamp,alive
from test_reconstruction_flow import session_fixture


@pytest.mark.parametrize('h0',[.71,.83])
def test_actual_compiled_h0_translation_preserves_thickness_and_xy(h0):
    root=ET.fromstring(f'''<mujoco><worldbody><body name="table_root" pos=".6 .1 {h0-.001-.1}"><geom type="box" size=".5 .4 .1"/></body><body name="pad_node_0" pos=".3 .2 {h0+.005-.0001}"><geom type="sphere" size=".0001"/></body></worldbody></mujoco>''')
    before=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    delta=shift_to_h0(root,dict(actual_uncompressed_soft_top_m=h0+.005,thickness_m=.006),h0)
    after=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(after);mujoco.mj_forward(after,d)
    assert delta==pytest.approx(-.005)
    assert d.geom_xpos[0,2]+after.geom_size[0,2]==pytest.approx(h0-.006)
    assert d.geom_xpos[1,2]+after.geom_size[1,0]==pytest.approx(h0)
    np.testing.assert_array_equal(before.geom_size,after.geom_size)
    np.testing.assert_array_equal(before.body_pos[:,:2],after.body_pos[:,:2])


def test_support_uses_upper_skin_of_flat_closed_display_mesh():
    v=np.array([[0,0,.8],[1,0,.8],[0,1,.8],[0,0,.7998],[1,0,.7998],[0,1,.7998]])
    faces=np.array([[3,5,4],[0,1,2]])
    support=SupportSurface(v,faces,top_only=True)
    assert support([[.2,.2]])[0]==pytest.approx(.8)


def test_full_frame_masks_and_possible_robot_occlusion_are_not_assumed_absent():
    rgb=np.full((480,640,3),45,np.uint8);cv2.circle(rgb,(120,40),25,(210,15,15),-1);cv2.ellipse(rgb,(430,330),(65,35),0,0,360,(230,230,230),-1)
    robot=np.zeros((480,640),bool);robot[20:65,120:150]=True
    masks,valid,occ,quality=_pixels(rgb,robot,{'apple':np.array([120,40]),'plate':np.array([430,330])})
    assert masks['apple'][:48].any(),'Top pixels must remain observable'
    assert occ.any() and not valid[40,125]
    assert quality['apple']['possible_robot_occlusion_fraction']>.2 and not quality['apple']['quality_ok']


def test_episode_ledgers_are_independent_and_restart_does_not_refund(tmp_path):
    limits=dict(terra_requests=5,geometry_rounds=4,render_total=3,render_initial=3,render_repair=0,render_appearance=0,projection_evaluations=5,optimization_seconds=10)
    path=tmp_path/'ledger.sqlite'
    a=BudgetManager(path,'batch/episode_000000',limits);b=BudgetManager(path,'batch/episode_000001',limits)
    for i in range(3):a.reserve_render('render_initial',{'i':i})
    with pytest.raises(BudgetExceeded):BudgetManager(path,a.task_id,limits).reserve_render('render_initial',{})
    assert b.used('render_total')==0;b.reserve_render('render_initial',{})
    for i in range(4):a.reserve_terra(False,dict(candidate_id='same'))
    with pytest.raises(BudgetExceeded):a.reserve_terra(False,{})
    a.reserve_terra(True,dict(candidate_id='same'))
    with pytest.raises(BudgetExceeded):BudgetManager(path,a.task_id,limits).reserve_terra(True,dict(candidate_id='same'))
    assert b.used('terra_requests')==0


def test_concurrent_reservations_cannot_cross_global_render_limit(tmp_path):
    limits=dict(terra_requests=5,geometry_rounds=4,render_total=2,render_initial=2,projection_evaluations=5,optimization_seconds=10)
    path=tmp_path/'ledger.sqlite';BudgetManager(path,'ep',limits)
    def reserve(i):
        try:BudgetManager(path,'ep',limits).reserve_render('render_initial',dict(i=i));return True
        except BudgetExceeded:return False
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:ok=list(pool.map(reserve,range(8)))
    assert sum(ok)==2


def test_stop_prevents_work_and_api_circuit_prevents_request(tmp_path,monkeypatch):
    stop=tmp_path/'STOP';stop.write_text('operator stop');monkeypatch.setenv('MOVENS_BATCH_STOP_FILE',str(stop))
    with pytest.raises(BatchPaused):check_stop()
    stop.unlink();(tmp_path/'api_circuit.json').write_text(json.dumps(dict(manual_pause=True,reason='offline fixture auth error')))
    with pytest.raises(BatchPaused):
        with api_slot({'batch':{'control_root':str(tmp_path)}}):pytest.fail('Must not reach HTTP dispatch')


def test_batch_report_keeps_every_original_id_and_separate_quality_statuses(tmp_path):
    with connect(tmp_path) as db:
        db.execute('CREATE TABLE tasks(episode INTEGER PRIMARY KEY,status TEXT,attempts INTEGER,errors INTEGER,not_before REAL,pid INTEGER,stamp TEXT,gpu INTEGER,started REAL,stage TEXT,reason TEXT,quality TEXT)')
        for ep in range(402):db.execute('INSERT INTO tasks VALUES (?,?,0,0,0,NULL,NULL,NULL,0,NULL,NULL,NULL)',(ep,'pending'))
        quality=dict(quality=dict(construction_complete=True,visual_qualified=False,physical_passed=False,needs_review=True))
        db.execute("UPDATE tasks SET status='complete',quality=? WHERE episode=0",(json.dumps(quality),))
    result=publish(tmp_path,{'terra':{'model':'OFFLINE_FIXTURE'}})
    assert result['total']==402 and result['counts']['pending']==401
    assert result['quality']==dict(construction_complete=1,visual_qualified=0,physical_passed=0,needs_review=1)
    assert len((tmp_path/'results.csv').read_text().splitlines())==403


def test_overhanging_fk_gets_idempotent_feasible_search_seed_without_table_or_scale_change(session_fixture,tmp_path):
    from movens_sim.reconstruction.backend import SceneBackend
    s=session_fixture;tree=ET.ElementTree(s.scene.template)
    tree.find(".//body[@name='plate_root']").set('pos','-.19 -.15 .85')
    path=tmp_path/'overhang.xml';tree.write(path)
    a=SceneBackend(path,initialize_supported=True);b=SceneBackend(path,initialize_supported=True)
    assert a.initial_support_seed['plate']['delta_xy_m'][0]>.02
    assert a.params['plate.scale']==1. and a.geometry_metrics()['physical_valid']
    assert a.params==b.params
    np.testing.assert_array_equal(a.support.bounds,b.support.bounds)
    a.close();b.close()


def test_pending_directed_fit_resumes_same_parameter_subset(session_fixture,monkeypatch):
    from movens_sim.reconstruction.workflow import resume_pending_fit
    s=session_fixture;s.state['rounds']=[dict(round_index=0)]
    initial=s.budget.reserve('geometry_rounds',detail=dict(round_index=0));s.budget.finish(initial)
    pending=s.budget.reserve('geometry_rounds',detail=dict(round_index=1,parameter_ids=['plate.x','plate.y','plate.scale']))
    calls=[];monkeypatch.setattr(s,'fit_geometry',lambda **kw:calls.append(kw))
    resume_pending_fit(s)
    assert calls==[dict(parameter_ids=['plate.x','plate.y','plate.scale'])]
    assert s.budget.used('geometry_rounds')==2


def test_locked_batch_camera_is_absent_from_model_action_schema(session_fixture):
    from movens_sim.reconstruction.terra import response_schema
    s=session_fixture;s.scene.allow_camera_revision=False
    tools=response_schema(s.scene)['properties']['proposed_actions']['items']['properties']['tool']['enum']
    assert 'review_camera_hypothesis' not in tools


def test_intersecting_fk_initial_layout_gets_supported_seed_without_size_changes(session_fixture,tmp_path):
    from movens_sim.reconstruction.backend import SceneBackend,SceneError
    s=session_fixture;tree=ET.ElementTree(s.scene.template)
    tree.find(".//body[@name='plate_root']").set('pos',' '.join(map(str,s.scene.initial_poses['apple']['xyz_m']+np.array([.02,.005,0.]))))
    path=tmp_path/'intersecting.xml';tree.write(path)
    with pytest.raises(SceneError):SceneBackend(path)
    scene=SceneBackend(path,initialize_supported=True)
    assert scene.geometry_metrics()['physical_valid'] and scene.initial_support_seed['plate']
    assert scene.params['apple.scale']==scene.params['plate.scale']==1.
    assert 'AABB-separated' in scene.initial_support_seed['plate']['reason']
    scene.close()
