"""Offline fixtures are isolated from all actual episode/API review records."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import httpx
import mujoco
import numpy as np
import pytest
from openai import OpenAI

from movens_sim.util import ROOT,read,write,save_rgb
from movens_sim.ingest import portable_copy
from movens_sim.head_camera import free_objects,save_state_by_name
from movens_sim.scene import quat_from_matrix
from movens_sim.reconstruction.backend import SceneBackend,SceneError
from movens_sim.reconstruction.session import ReconstructionSession
from movens_sim.reconstruction.metrics import gate
from movens_sim.reconstruction.terra import request_review,validate_review
from movens_sim.reconstruction.workflow import appearance,export_and_review


@pytest.fixture
def session_fixture(toy_scene,tmp_path,monkeypatch):
    import xml.etree.ElementTree as ET
    import movens_sim.reconstruction.session as module
    # Put this fixture's ledger under tmp_path, never the actual trial task.
    monkeypatch.setattr(module,'ROOT',tmp_path)
    mesh=toy_scene/'assets/support.obj';mesh.write_text('v -.2 -.6 .8\nv .8 -.6 .8\nv .8 .6 .8\nv -.2 .6 .8\nv .3 0 .81\nf 1 2 5\nf 2 3 5\nf 3 4 5\nf 4 1 5\nf 1 3 2\nf 1 4 3\n')
    path=toy_scene/'scene_fitting.xml';tree=ET.parse(path);ET.SubElement(tree.find('asset'),'mesh',name='support',file='assets/support.obj');ET.SubElement(tree.find('worldbody'),'geom',name='frozen_support',type='mesh',mesh='support',group='1',contype='0',conaffinity='0');tree.write(path);free_objects(path)
    backend=SceneBackend(path);rgb,masks=backend.render()
    cam=backend.cam;camera=dict(xyz_m=backend.data.cam_xpos[cam].tolist(),quaternion_wxyz=quat_from_matrix(backend.data.cam_xmat[cam]).tolist(),rotation_camera_to_world=backend.data.cam_xmat[cam].reshape(3,3).tolist(),fovy_deg=float(backend.model.cam_fovy[cam]),prior_status='OFFLINE_TEST_FIXTURE')
    run=tmp_path/'fixture';(run/'inputs').mkdir(parents=True);(run/'masks').mkdir()
    portable_copy(path,run/'inputs/template_static');portable_copy(path,run/'inputs/template_physics');save_state_by_name(path,None,run/'inputs/physics_state_by_name.npz')
    write(run/'initial_parameters.json',{});write(run/'camera.json',camera);write(run/'camera_prior.json',dict(fixture=True));save_rgb(run/'reference.png',rgb)
    np.savez_compressed(run/'masks/observations.npz',**masks,valid=np.ones((480,640),bool))
    write(run/'observation.json',dict(version='OFFLINE_FIXTURE_V1'));write(run/'source_manifest.json',dict(episode=0,fixture=True))
    config=read(ROOT/'configs/mount_joint_v3.json');config['task_group_id']='offline_fixture_'+uuid.uuid4().hex;config['budget']['render_appearance']=8
    write(run/'resolved_config.json',config);backend.close();session=ReconstructionSession(run,config)
    yield session
    session.close()


def test_api_failed_attempts_retries_resume_and_final_slot_are_hard_limited(session_fixture,monkeypatch):
    import movens_sim.reconstruction.terra as terra
    s=session_fixture;monkeypatch.setattr(terra,'credentials',lambda config:'offline_fixture_key_never_sent_to_network')
    s.evaluator.evaluate(s.scene.params.copy(),'fixture_baseline',640,480);best=s.evaluator.best['geometry_640x480']
    with np.load(s.run/'evaluation_cache'/(best['cachekey']+'.npz')) as z:rgb=z['rgb'];masks={k:z[k] for k in ('apple','plate','cloth','table')}
    sent=[]
    def factory(**kwargs):
        assert kwargs['max_retries']==0
        hooks=kwargs['http_client'].event_hooks;kwargs['http_client'].close()
        def failure(request):
            sent.append(request.url.path)
            return httpx.Response(503,json={'error':{'message':'OFFLINE FIXTURE transient failure','type':'fixture','code':'fixture'}})
        kwargs['http_client']=httpx.Client(transport=httpx.MockTransport(failure),event_hooks=hooks)
        return OpenAI(**kwargs)
    for i in range(4):
        status,parsed=request_review(s,best['metrics'],rgb,masks,enabled=True,client_factory=factory)
        assert parsed is None and status['actual_http_attempts']==1
    status,_=request_review(s,best['metrics'],rgb,masks,enabled=True,client_factory=lambda **kw:pytest.fail('Fifth nonfinal call must be blocked before constructing client'))
    assert status['status']=='blocked_budget'
    # Simulate restart, with exactly the same ledger and remaining final slot.
    s.budget=type(s.budget)(s.budget.path,s.budget.task_id,s.config['budget'])
    s.scene.stage='final_readonly'
    status,_=request_review(s,best['metrics'],rgb,masks,final=True,enabled=True,client_factory=factory)
    assert status['actual_http_attempts']==1
    status,_=request_review(s,best['metrics'],rgb,masks,final=True,enabled=True,client_factory=lambda **kw:pytest.fail('Sixth request blocked'))
    assert status['status']=='blocked_budget' and len(sent)==5
    attempts=list((s.run/'terra').glob('*/attempt_*/status.json'))
    assert len(attempts)==5 and sum(read(p)['actual_http_attempts'] for p in attempts)==5
    assert s.budget.used('terra_requests')==5


def test_tools_and_optimizer_share_scale_support_and_checkpoint_rollback(session_fixture):
    s=session_fixture;old=s.scene.params.copy();candidate=s.scene.candidate_id();s.checkpoint('before',candidate)
    trial=s.propose_or_apply_object_update({'plate.scale':1.2},candidate,apply=False)
    assert s.scene.params==old
    params={**old,'plate.scale':1.2};_,metrics=s.evaluator.evaluate(params,'same_backend_optimizer',640,480)
    row=json.loads((s.run/'candidates.jsonl').read_text().splitlines()[-1])
    np.testing.assert_allclose(trial['geometry']['plate']['dimensions_world_m'],row['geometry']['plate']['dimensions_world_m'])
    used=s.budget.used('render_total');s.scene.shape_enabled=True;s.restore('before',s.scene.candidate_id())
    assert s.scene.params==old and not s.scene.shape_enabled and s.budget.used('render_total')==used
    assert s.evaluator.best=={},'Explicit checkpoint restores the optimizer best snapshot too; spent budget stays used'
    with pytest.raises(SceneError):s.review_camera_hypothesis('One plate only',.9,s.scene.candidate_id())


def test_evidence_route_enables_real_plate_tilt_before_joint_fit(session_fixture,monkeypatch):
    import movens_sim.reconstruction.session as module
    from movens_sim.reconstruction.workflow import actions_to_calls
    s=session_fixture
    parsed=dict(proposed_actions=[dict(tool='fit_geometry',parameter_ids=['plate.rx','plate.ry','plate.scale'],updates=[],evidence='Opposing plate axis residuals after center/area alignment',confidence=.79)])
    calls=actions_to_calls(s,parsed);fit=next(c for c in calls if c['tool']=='fit_geometry')
    assert fit['arguments']['confidence']==.79 and fit['arguments']['evidence']
    with pytest.raises(SceneError):s.fit_geometry(['plate.rx'],s.scene.candidate_id())
    assert s.budget.used('geometry_rounds')==0
    def actual_backend_probe(evaluator,ids,index,resume_event):
        assert evaluator.scene.plate_tilt_enabled
        evaluator.scene.apply({'plate.rx':1.})
        assert evaluator.scene.geometry_metrics()['physical_valid']
        return dict(parameter_ids=ids,round_index=index)
    monkeypatch.setattr(module,'fit_geometry',actual_backend_probe)
    s.fit_geometry(fit['arguments']['parameter_ids'],s.scene.candidate_id(),fit['arguments']['evidence'],fit['arguments']['confidence'])
    assert s.scene.params['plate.rx']==1.


def test_unsupported_or_unapproved_fit_subset_rejected_before_round_reservation(session_fixture):
    s=session_fixture
    for ids in [[],['plate.scale','plate.scale'],['unknown'],['light.ambient'],['plate.depth_ratio']]:
        with pytest.raises(SceneError):s.fit_geometry(ids,s.scene.candidate_id(),evidence='fixture evidence',confidence=.9)
    for confidence in [float('nan'),float('inf'),1.1]:
        with pytest.raises(SceneError):s.fit_geometry(['plate.rx'],s.scene.candidate_id(),evidence='fixture evidence',confidence=confidence)
    assert s.budget.used('geometry_rounds')==0


def test_offline_full_state_flow_appearance_freeze_reload_and_terminal_readonly(session_fixture,monkeypatch):
    import movens_sim.reconstruction.terra as terra
    s=session_fixture;monkeypatch.setattr(terra,'credentials',lambda config:None)
    s.evaluator.evaluate(s.scene.params.copy(),'fixture_truth_baseline',640,480);best=s.evaluator.best['geometry_640x480']
    write(s.run/'metrics_mount_baseline.json',best['metrics'])
    with np.load(s.run/'evaluation_cache'/(best['cachekey']+'.npz')) as z:rgb=z['rgb'];masks={k:z[k] for k in ('apple','plate','cloth','table')}
    status,parsed=request_review(s,best['metrics'],rgb,masks,enabled=False)
    assert parsed is None and status['status']=='prepared_offline' and s.budget.used('terra_requests')==0
    appearance(s);assert s.state['appearance_status']=='blocked_geometry_or_terra_gate'
    # Explicit synthetic geometry approval for exercising the remaining state
    # machine, never represented as an actual model review in any result.
    fixture=dict(candidate_id=s.scene.candidate_id(),stage='GEOMETRY',verdict='pass',issues=[],proposed_actions=[],deferred_appearance_notes=['OFFLINE SYNTHETIC INTEGRATION FIXTURE'])
    validate_review(fixture,s);write(s.run/'synthetic_geometry_approval_fixture.json',fixture);s.state['geometry_review_passed']=True
    original={k:v for k,v in s.scene.params.items() if 'geometry' in next(p['stages'] for p in s.scene.registry() if p['id']==k)}
    appearance(s)
    assert s.state['appearance_status']=='completed' and all(s.scene.params[k]==v for k,v in original.items())
    export_and_review(s,enabled=False)
    assert s.scene.stage=='final_readonly' and read(s.run/'reload_check.json')['passed']
    assert read(s.run/'final_review.json')['status']['status']=='prepared_offline'
    assert s.budget.used('terra_requests')==0 and s.budget.used('render_appearance')<=8
    with pytest.raises(SceneError):s.update_appearance({'cloth.gray':.3},s.scene.candidate_id())
    assert read(s.run/'ego_static_definition.json')['video']['frames']==150
