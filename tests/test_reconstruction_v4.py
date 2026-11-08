"""v4 behavior contracts; synthetic fixtures never count as actual Terra reviews."""
import asyncio
import copy
from pathlib import Path
import httpx
import mujoco
import numpy as np
import pytest
from openai import OpenAI
from test_reconstruction_flow import session_fixture
from movens_sim.util import ROOT,read,write,save_rgb
from movens_sim.reconstruction.budget import BudgetManager,BudgetExceeded
from movens_sim.reconstruction.diagnostics import terminal_pass,load_state,loaded_geometry
from movens_sim.reconstruction.metrics import gate
from movens_sim.reconstruction.review_state import CATEGORIES,dependencies,invalidate,valid_check,protected_slots
from movens_sim.reconstruction.session import ReconstructionSession
from movens_sim.reconstruction.mcp_server import call_tools,make_server


@pytest.fixture
def v4(session_fixture,monkeypatch):
    s=session_fixture;run=s.run;task=s.config['task_group_id'];s.close()
    c=read(ROOT/'configs/ep012_reconstruction_v4.json');c['task_group_id']=task+'_v4';c['batch']['control_root']=None
    write(run/'resolved_config.json',c);write(run/'mask_review.json',dict(mask_quality_ok=True,reviewed=False))
    state=read(run/'observation.json');state['mask_sha256s']={};write(run/'observation.json',state)
    monkeypatch.setenv('MOVENS_SIM_PROJECT_ROOT',str(run.parent))
    s=ReconstructionSession(run,c)
    yield s
    s.close()


def test_terminal_and_peak_are_distinct_without_velocity_gate():
    limits=dict(translation_m_max=.01,rotation_deg_max=10.,penetration_m_max=.002)
    st=dict(translation_final_m=.008,rotation_final_deg=8.,penetration_final_m=.0018,translation_max_m=.03,rotation_max_deg=40.,penetration_max_m=.006,last_speed=100.)
    assert terminal_pass(st,limits)
    for key,value in [('translation_final_m',.01001),('rotation_final_deg',10.01),('penetration_final_m',.002001)]:
        assert not terminal_pass({**st,key:value},limits)
    assert not terminal_pass(st,limits,finite=False)
    assert not terminal_pass(st,limits,free=False)


def test_initial_support_threshold_is_independent_and_backend_reads_config(v4):
    s=v4.scene;b=s.model.body('apple_root').id;j=s.model.joint('apple_freejoint').id;q=int(s.model.jnt_qposadr[j]);s.data.qpos[q+2]+=.0022;mujoco.mj_forward(s.model,s.data)
    assert s.geometry_metrics()['physical_valid']
    s.support_limits={**s.support_limits,'gap_max_m':.002}
    assert 'apple.support' in s.geometry_metrics()['failures']
    assert v4.config['rest_acceptance']['penetration_m_max']==.002
    assert v4.config['initial_support']['pair_penetration_m_max']==.0012


def test_iou_error_conversion_and_absolute_cloth_gate(v4):
    s=v4;s.evaluator.evaluate(s.scene.params.copy(),'fixture_baseline',640,480);m=copy.deepcopy(s.evaluator.best['geometry_640x480']['metrics'])
    for o in m['objects'].values():o.update(iou=.89,centroid_error_px=3.5,symmetric_edge_distance_px=3.5,area_relative_error=.059,axis_relative_errors=[.059,.059])
    m['table'].update(cloth_iou=.5,top_cloth_iou=.0,false_background_on_real_cloth_fraction=.5)
    result=gate(m,s.scene.geometry_metrics(),m,s.config)
    assert not any(x.startswith(('apple.','plate.')) for x in result['failures'])
    assert 'table.cloth_iou.absolute' in result['failures']
    assert s.config['acceptance']['iou_min']==pytest.approx(1-1.2*(1-.9))


def test_category_validity_skip_and_invalidation():
    params={'apple.x':.4,'apple.scale':1.,'plate.scale':1.,'cloth.coverage':1.,'cloth.gray':.2};camera={'fovy_deg':42};obs='fixture'
    state=dict(observation_version=obs,category_checks={c:dict(passed=True,valid=True,source_candidate_id='old',dependency_hash=dependencies(c,params,camera,obs)) for c in CATEGORIES})
    assert all(valid_check(state,c,params,camera,obs) for c in CATEGORIES)
    changed={**params,'cloth.gray':.3};invalidate(state,params,changed,camera,camera)
    assert not valid_check(state,'appearance',changed,camera,obs)
    assert valid_check(state,'rest',changed,camera,obs)
    changed2={**changed,'apple.scale':1.1};invalidate(state,changed,changed2,camera,camera)
    assert not any(valid_check(state,c,changed2,camera,obs) for c in CATEGORIES)
    fresh=dict(category_attempts={});assert protected_slots(fresh,params,camera,obs)==11


def test_twenty_request_restore_and_terminal_reservation(tmp_path):
    limits=read(ROOT/'configs/ep012_reconstruction_v4.json')['budget'];path=tmp_path/'ledger.sqlite';b=BudgetManager(path,'v4',limits)
    for i in range(19):b.reserve_terra(detail=dict(category=CATEGORIES[i%5],candidate_id='fixture'))
    resumed=BudgetManager(path,'v4',limits)
    with pytest.raises(BudgetExceeded):resumed.reserve_terra()
    resumed.reserve_terra(True,dict(candidate_id='fixture',delivery_hash='same'))
    with pytest.raises(BudgetExceeded):resumed.reserve_terra(True,dict(candidate_id='fixture',delivery_hash='same'))
    with pytest.raises(BudgetExceeded):resumed.reserve_render('render_repair',{})
    resumed.reserve_render('render_export',dict(readonly_reacceptance=True))
    assert resumed.used('terra_requests')==20


def test_real_mcp_environment_update_is_exported_and_cannot_create_support(v4,tmp_path):
    s=v4;original_bounds=s.scene.support.bounds.copy();s.scene.apply({'cloth.coverage':1.});rgb,masks=s.scene.render()
    save_rgb(s.run/'reference.png',rgb);np.savez_compressed(s.run/'masks/observations.npz',**masks,valid=np.ones((480,640),bool));s.scene.apply({'cloth.coverage':0.});s.save();s.close()
    result=asyncio.run(call_tools(s.run,[dict(tool='update_environment',arguments=dict(updates={'cloth.coverage':1.},evidence='Synthetic target includes the full bounded table cloth',confidence=.95,_current_candidate_id=True))]))
    assert result['calls'][0]['result']['ok'],result
    current=ReconstructionSession(s.run,s.config)
    try:
        assert current.scene.params['cloth.coverage']==1.
        np.testing.assert_array_equal(current.scene.support.bounds,original_bounds)
        path=current.scene.export(tmp_path/'portable');m,d=load_state(path)
        g=m.geom('cloth_continuation').id;assert m.geom_contype[g]==0 and m.geom_conaffinity[g]==0
        np.testing.assert_allclose(current.scene.render()[0],__import__('movens_sim.render',fromlist=['Renderer']).Renderer(m).rgb(d),atol=0)
        report=loaded_geometry(m,d,current.scene.params,current.config['initial_support']);assert report['physical_support_xy_bounds_m']==original_bounds.tolist()
        assert current.budget.summary()['used']['mcp_local_call']==0
    finally:current.close()


def test_model_schema_and_mcp_stage_capabilities_match(v4):
    from movens_sim.reconstruction.classified_terra import response_schema
    s=v4;s.scene.stage='appearance';s.state['active_category']='appearance';s.save()
    schema=response_schema(s.scene);names=schema['properties']['proposed_actions']['items']['properties']['tool']['enum']
    assert 'fit_appearance' in names and 'update_appearance' in names
    assert 'update_environment' not in names and 'repair_support' not in names
    s.scene.stage='final_readonly';schema=response_schema(s.scene,True)
    assert schema['properties']['proposed_actions']['maxItems']==0


def test_failed_http_requests_and_retries_share_twenty_cap(v4,monkeypatch):
    import movens_sim.reconstruction.terra as terra
    s=v4;s.state['active_category']='observation';s.state['evidence_revision']=1;s.save();s.evaluator.evaluate(s.scene.params.copy(),'api_fixture',640,480);best=s.evaluator.best['geometry_640x480']
    with np.load(s.run/'evaluation_cache'/(best['cachekey']+'.npz')) as z:rgb=z['rgb'];masks={k:z[k] for k in ('apple','plate','cloth','table')}
    monkeypatch.setattr(terra,'credentials',lambda cfg:'offline-fixture-key-not-sent-to-network');sent=[]
    def factory(**kwargs):
        assert kwargs['max_retries']==0;hooks=kwargs['http_client'].event_hooks;kwargs['http_client'].close()
        def failure(request):sent.append(1);return httpx.Response(503,json={'error':{'message':'fixture','type':'fixture','code':'fixture'}})
        kwargs['http_client']=httpx.Client(transport=httpx.MockTransport(failure),event_hooks=hooks);return OpenAI(**kwargs)
    for _ in range(19):
        status,parsed=terra.request_review(s,best['metrics'],rgb,masks,client_factory=factory)
        assert parsed is None and status['actual_http_attempts']==1
    status,_=terra.request_review(s,best['metrics'],rgb,masks,client_factory=lambda **kw:pytest.fail('Final slot is reserved'))
    assert status['status']=='blocked_budget'
    s.scene.stage='final_readonly';s.state['active_category']='final';status,_=terra.request_review(s,best['metrics'],rgb,masks,final=True,delivery_binding={'scene_parameter_asset_hash':'fixture'},client_factory=factory)
    assert len(sent)==20 and s.budget.used('terra_requests')==20


def test_native_flex_contact_names_fixed_duration_and_saved_velocity(tmp_path):
    from movens_sim.reconstruction.diagnostics import rest_from_disk
    xml=tmp_path/'scene.xml'
    xml.write_text('''<mujoco><option timestep="0.01" gravity="0 0 -9.81" cone="elliptic" integrator="implicitfast"><flag island="disable"/></option>
    <worldbody><camera name="approx_ego_camera" pos="0 -2 2" xyaxes="1 0 0 0 0.7 0.7"/>
    <body name="table_root"><geom type="box" pos="0 0 .7" size="1 1 .1"/></body>
    <body name="n0" pos="-1 -1 .8"><joint type="slide" axis="0 0 1" stiffness="100" damping="1"/><geom type="sphere" size=".001" mass=".001" contype="0" conaffinity="0" group="5"/></body>
    <body name="n1" pos="1 -1 .8"><joint type="slide" axis="0 0 1" stiffness="100" damping="1"/><geom type="sphere" size=".001" mass=".001" contype="0" conaffinity="0" group="5"/></body>
    <body name="n2" pos="0 1 .8"><joint type="slide" axis="0 0 1" stiffness="100" damping="1"/><geom type="sphere" size=".001" mass=".001" contype="0" conaffinity="0" group="5"/></body>
    <body name="apple_root" pos=".1 .1 .84"><freejoint name="apple_freejoint"/><geom name="apple_col" type="sphere" size=".04" mass=".1"/></body>
    <body name="plate_root" pos=".2 -.2 .84"><freejoint name="plate_freejoint"/><geom name="plate_col" type="sphere" size=".04" mass=".1"/></body>
    </worldbody><deformable><flex name="native_test_pad" dim="2" radius=".0001" body="n0 n1 n2" vertex="0 0 0 0 0 0 0 0 0" element="0 1 2"/></deformable></mujoco>''')
    m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m);d.qvel[0]=.001
    np.savez_compressed(xml.with_suffix('.state.npz'),qpos=d.qpos,qvel=d.qvel,time=np.array(0.))
    c=read(ROOT/'configs/ep012_reconstruction_v4.json');result=rest_from_disk(xml,tmp_path/'diagnostic',c,video=False)
    assert result['duration_s']==pytest.approx(1.) and result['steps']==100
    assert result['loaded_qvel_max']==pytest.approx(.001)
    contacts=read(tmp_path/'diagnostic/contact_timeline.json')
    assert any('native_test_pad' in x['flex_names'] for row in contacts for x in row['contacts'])
    assert read(tmp_path/'diagnostic/diagnostic_timeline.json')[0]['time_s']==0.


def test_yellow_appearance_follows_mesh_height_not_atlas_rows(tmp_path):
    from movens_sim.reconstruction.appearance import height_atlas,apple_texture
    from PIL import Image
    mesh=tmp_path/'uv.obj';mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 1\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n')
    atlas=height_atlas(str(mesh),64,64)
    assert atlas[10,10]>.7 and atlas[52,5]<.25
    texture=tmp_path/'red.png';Image.fromarray(np.tile(np.array([180,25,15],np.uint8),(64,64,1))).save(texture)
    first=apple_texture(str(texture),str(mesh),.8);second=apple_texture(str(texture),str(mesh),.8)
    assert first==second and apple_texture.cache_info().hits>=1


def test_failed_frozen_successor_preserves_cumulative_budget(tmp_path):
    limits=read(ROOT/'configs/ep012_reconstruction_v4.json')['budget'];b=BudgetManager(tmp_path/'ledger.sqlite','ep012_reconstruction_v4_20261107/episode_000000',limits)
    for i in range(9):b.reserve_terra(detail=dict(candidate_id='old'))
    b.reserve_terra(True,dict(candidate_id='old',delivery_hash='oldhash'));proof=tmp_path/'failed_review.json';write(proof,dict(delivery_hash='oldhash',status=dict(status='validated'),review=dict(candidate_id='old',verdict='fail')))
    b.begin_repair_revision('old','oldhash','Regression: preserve failed candidate and repair unfinished classification',proof)
    assert b.used('terra_requests')==10
    b.reserve_render('render_repair',{})
    for i in range(9):b.reserve_terra(detail=dict(candidate_id='new'))
    with pytest.raises(BudgetExceeded):b.reserve_terra()
    b.reserve_terra(True,dict(candidate_id='new',delivery_hash='newhash'))
    assert b.used('terra_requests')==20
    with pytest.raises(BudgetExceeded):b.reserve_terra(True,dict(candidate_id='new',delivery_hash='newhash'))


def test_resume_completed_first_round_advances_to_required_rechecks(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import movens_sim.reconstruction.classified as module
    events=[]
    class Budget:
        def remaining(self,k):return 20-len(events)
        def used(self,k):return len(events)
        def summary(self):return dict(events=events)
    scene=SimpleNamespace(params={'cloth.coverage':1.},locked_camera={},stage='geometry',candidate_id=lambda:'fixture')
    s=SimpleNamespace(scene=scene,budget=Budget(),state=dict(classification_progress=['0/'+c for c in CATEGORIES],appearance_status='completed'),config={},run=tmp_path,observation={'version':'fixture'},save=lambda:None)
    def evidence(s,c):s.state['active_category']=c;return s,{'metrics':{}}
    def request(s,*args,**kwargs):
        c=s.state['active_category'];events.append(dict(kind='terra_requests',detail=dict(category=c)))
        return {'status':'validated','folder':'fixture'},{'verdict':'pass','proposed_actions':[]}
    monkeypatch.setattr(module,'prepare_evidence',evidence);monkeypatch.setattr(module,'current_evaluation',lambda *a,**k:{'metrics':{}});monkeypatch.setattr(module,'image_cache',lambda *a:(None,None));monkeypatch.setattr(module,'request_review',request);monkeypatch.setattr(module,'numeric_category',lambda *a:True)
    module.review_categories(s,True)
    assert len(events)==5 and all('1/'+c in s.state['classification_progress'] for c in CATEGORIES)


def test_api_failure_keeps_independent_evidence_and_unqueried_categories_resumable(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import movens_sim.reconstruction.classified as module
    from movens_sim.reconstruction.control import BatchPaused
    events=[];prepared=[]
    class Budget:
        def remaining(self,k):return 20-len(events)
        def used(self,k):return len(events)
        def summary(self):return dict(events=events)
    scene=SimpleNamespace(params={'cloth.coverage':1.},locked_camera={},stage='geometry',candidate_id=lambda:'fixture')
    s=SimpleNamespace(scene=scene,budget=Budget(),state=dict(appearance_status='completed'),config={},run=tmp_path,observation={'version':'fixture'},save=lambda:None)
    def evidence(s,c):prepared.append(c);s.state['active_category']=c;return s,{'metrics':{}}
    def failed(s,*args,**kwargs):
        if not events:events.append(dict(kind='terra_requests',detail=dict(category=s.state['active_category'])))
        raise BatchPaused('waiting_api','fixture API circuit',30)
    monkeypatch.setattr(module,'prepare_evidence',evidence);monkeypatch.setattr(module,'current_evaluation',lambda *a,**k:{'metrics':{}});monkeypatch.setattr(module,'image_cache',lambda *a:(None,None));monkeypatch.setattr(module,'request_review',failed);monkeypatch.setattr(module,'numeric_category',lambda *a:True)
    with pytest.raises(BatchPaused):module.review_categories(s,True)
    assert prepared==list(CATEGORIES) and not s.state['classification_complete']
    assert s.state['classification_progress']==['0/observation'] and len(events)==1


def test_flat_early_coordinates_do_not_stop_before_later_free_parameters():
    from types import SimpleNamespace
    from movens_sim.reconstruction.optimize import Evaluator
    e=object.__new__(Evaluator);ids=['p0','p1','p2','p3','p4'];params={k:.5 for k in ids};e.stage='geometry';e.improvements=0;e.config=dict(workflow='classified_reconstruction_v4',fit=dict(local_no_improvement_evaluations=40));e.scene=SimpleNamespace(params=params.copy(),registry=lambda:[dict(id=k,lower=0.,upper=1.) for k in ids]);e.best={'geometry_640x480':dict(params=params.copy(),loss=.09)}
    def evaluate(p,*args):
        loss=(p['p4']-.8)**2
        if loss<e.best['geometry_640x480']['loss']-1e-8:e.best['geometry_640x480']=dict(params=p.copy(),loss=loss);e.improvements+=1
        return loss,{}
    e.evaluate=evaluate;e.optimize(ids,220,'fixture',640,480)
    assert abs(e.best['geometry_640x480']['params']['p4']-.8)<.03


def test_appearance_baseline_rebinds_after_geometry_revision(v4):
    from movens_sim.reconstruction.operations import current_evaluation,allowed_parameters
    s=v4;old={k:v for k,v in s.scene.params.items() if k in allowed_parameters(s,'geometry')};write(s.run/'geometry_frozen.json',dict(geometry_params=old))
    s.scene.apply({'plate.scale':1.04});s.scene.stage='appearance';s.save()
    best=current_evaluation(s,'appearance')
    assert best['candidate_id']==s.scene.candidate_id()
    assert read(s.run/'geometry_frozen.json')['geometry_params']['plate.scale']==1.04


def test_real_mcp_appearance_description_matches_independent_stage(v4):
    s=v4;s.scene.stage='appearance';s.state['geometry_review_passed']=False;s.save();s.close()
    result=asyncio.run(call_tools(s.run,[dict(tool='inspect_scene',arguments={})]))
    tools={x['name']:x for x in result['tools']}
    assert 'independent' in tools['update_appearance']['description'].lower()
    assert 'update_environment' not in tools and 'fit_appearance' in tools
    assert result['calls'][0]['result']['ok']
