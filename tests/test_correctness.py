import copy
import shutil
from pathlib import Path
import cv2
import jsonschema
import mujoco
import numpy as np
import pytest
from movens_sim.ingest import portable_copy
from movens_sim.scene import Scene
from movens_sim.render import Renderer,video_writer
from movens_sim.evaluate import image_metrics
from movens_sim.gpt_adapter import response_schema,validate_response
from movens_sim.fit import Fitter,BudgetExhausted
from movens_sim.pipeline import complete_receipt,receipt_valid
from movens_sim.util import write,read

def test_portable_resource_closure_after_source_removed(tmp_path):
    src=tmp_path/'source';src.mkdir();(src/'meshes').mkdir();(src/'parts').mkdir()
    (src/'meshes/t.obj').write_text('v 0 0 0\nv .1 0 0\nv 0 .1 0\nv 0 0 .1\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n')
    (src/'parts/asset.xml').write_text('<mujocoinclude><asset><mesh name="t" file="t.obj"/></asset></mujocoinclude>')
    (src/'model.xml').write_text('<mujoco><compiler meshdir="meshes"/><include file="parts/asset.xml"/><worldbody><geom type="mesh" mesh="t"/></worldbody></mujoco>')
    out=tmp_path/'portable';receipt=portable_copy(src/'model.xml',out);shutil.rmtree(src)
    m=mujoco.MjModel.from_xml_path(str(out/'scene.xml'));assert m.nmesh==1
    assert any(x['source_sha256']!=x['portable_sha256'] for x in receipt if x['portable'].endswith('.xml'))

def test_camera_pose_and_mesh_export_reload(toy_scene):
    s=Scene(toy_scene/'scene_fitting.xml');s.apply({'camera.x':.37,'camera.rx':5.,'camera.rz':-8.,'apple.x':.43,'apple.ry':15.,'apple.scale':1.12,'table.depth_scale':1.2,'table.width_scale':1.1,'table.front_extension':.03})
    output=s.export();m=mujoco.MjModel.from_xml_path(str(output));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    np.testing.assert_allclose(s.data.cam_xpos[0],d.cam_xpos[0],atol=1e-10)
    np.testing.assert_allclose(s.data.cam_xmat[0],d.cam_xmat[0],atol=1e-10)
    np.testing.assert_allclose(s.data.xpos[s.apple],d.xpos[m.body('apple_root').id],atol=1e-10)
    with Renderer(s.model,320,240) as r:a=r.masks(s.data,s.geom_sets)
    with Renderer(m,320,240) as r:b=r.masks(d,{'apple':[m.geom('apple_vis').id],'plate':[m.geom('plate_vis').id]})
    assert image_metrics(a,b,np.ones((240,320),bool))['objects']['apple']['iou']>.999
    assert (output.with_suffix('.state.npz')).is_file()

def test_mesh_scale_updates_existing_renderer(toy_scene):
    s=Scene(toy_scene/'scene_fitting.xml')
    with Renderer(s.model,320,240) as r:
        small=r.masks(s.data,s.geom_sets)['apple'].sum();s.apply({'apple.scale':1.15});large=r.masks(s.data,s.geom_sets)['apple'].sum()
    assert large>small*1.1

def fixture_response():
    return dict(decision='propose',observations=[dict(evidence='Fixture only',uncertainty='No model call')],updates=[dict(param_id='apple.x',value=.4,unit='m',rationale='fixture')],expected_effect=[dict(metric='apple.centroid_error_px',direction='decrease',rationale='fixture')],review_reason='')

def test_gpt_contract_rejects_unknown_nonfinite_unit_and_stage(toy_scene):
    s=Scene(toy_scene/'scene_fitting.xml');base=fixture_response();assert validate_response(base,s,'layout')=={'apple.x':.4}
    for key,value,stage in [('value',float('nan'),'layout'),('value',10.,'layout'),('unit','deg','layout'),('param_id','mass','layout'),('value',.4,'camera')]:
        r=copy.deepcopy(base);r['updates'][0][key]=value
        with pytest.raises((ValueError,jsonschema.ValidationError)):validate_response(r,s,stage)
    r=copy.deepcopy(base);r['updates'].append(copy.deepcopy(r['updates'][0]))
    with pytest.raises(ValueError):validate_response(r,s,'layout')
    for values in [{'apple.x':float('inf')},{'apple.x':True},{'apple.x':-20.}]:
        old=s.params.copy()
        with pytest.raises(ValueError):s.apply(values)
        assert s.params==old

def test_invalid_numeric_candidate_rolls_back_and_budget_is_bounded(toy_scene,tmp_path):
    s=Scene(toy_scene/'scene_fitting.xml');run=tmp_path/'run';(run/'masks').mkdir(parents=True)
    with Renderer(s.model,320,240) as renderer:rgb=renderer.rgb(s.data);masks=renderer.masks(s.data,s.geom_sets)
    cv2.imwrite(str(run/'reference.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));np.savez_compressed(run/'masks/observations.npz',apple=masks['apple'],plate=masks['plate'],valid=np.ones((240,320),bool))
    config=dict(fit=dict(width=320,height=240,max_evaluations=2,max_seconds=10,stages=['layout']))
    f=Fitter(s,run,config)
    try:
        f.stage='layout';f.evaluate(s.params.copy());best=s.params.copy();f.evaluate({**best,'apple.x':best['apple.x']+.1});assert s.params==best
        with pytest.raises(BudgetExhausted):f.evaluate(best)
        assert f.evaluations==2
    finally:f.close()
    resumed=Fitter(s,run,config,resume=True)
    try:
        assert resumed.evaluations==2 and np.isfinite(resumed.best_loss)
        assert s.params==best
        with pytest.raises(BudgetExhausted):resumed.evaluate(best)
    finally:resumed.close()

def test_receipt_does_not_resume_incomplete_or_changed_artifact(tmp_path):
    run=tmp_path/'run';run.mkdir();output=run/'artifact';output.write_text('complete');identity=dict(input='original',config='v1')
    complete_receipt(run,'fit',identity,[output],.1);assert receipt_valid(run,'fit',identity)
    assert not receipt_valid(run,'fit',dict(input='changed',config='v1'))
    output.write_text('truncated');assert not receipt_valid(run,'fit',identity)
    output.unlink();assert not receipt_valid(run,'fit',identity)

def test_video_frame_fps_resolution(tmp_path):
    frame=np.zeros((48,64,3),np.uint8);frame[:,:,1]=100
    result=video_writer(tmp_path/'test.mp4',(frame for _ in range(10)),30)
    assert result['frames']==10 and result['fps']==30 and result['width']==64 and result['height']==48

def test_image_evaluator_matches_and_detects_shifts():
    a=np.zeros((120,160),bool);a[40:80,60:100]=True;valid=np.ones_like(a);target={'apple':a,'plate':a}
    same=image_metrics(target,target,valid);assert same['objects']['apple']['iou']==1 and same['objects']['apple']['symmetric_edge_distance_px']==0
    shifted={'apple':np.roll(a,10,axis=1),'plate':a};result=image_metrics(shifted,target,valid)
    assert result['objects']['apple']['iou']<.7 and result['objects']['apple']['centroid_error_px']==10

def gpt_run_fixture(scene,run):
    (run/'masks').mkdir(parents=True)
    with Renderer(scene.model,320,240) as renderer:rgb=renderer.rgb(scene.data);masks=renderer.masks(scene.data,scene.geom_sets)
    cv2.imwrite(str(run/'reference.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));np.savez_compressed(run/'masks/observations.npz',apple=masks['apple'],plate=masks['plate'],valid=np.ones((240,320),bool))
    write(run/'reference_selection.json',dict(objects=dict(apple=dict(occluded_by_robot=False),plate=dict(occluded_by_robot=False))))

def test_gpt_offline_contract_and_exact_input_images(toy_scene,tmp_path,monkeypatch):
    import movens_sim.gpt_adapter as adapter
    from movens_sim.util import ROOT,sha
    scene=Scene(toy_scene/'scene_fitting.xml');run=ROOT/'runs'/'_offline_contract_test'
    # request logs use project-relative paths, matching the real pipeline layout.
    try:
        gpt_run_fixture(scene,run);config=read(ROOT/'configs/batch_v1.json');monkeypatch.setattr(adapter,'credentials',lambda c:None)
        status,parsed=adapter.call_or_prepare(scene,run,config,'appearance',0,dict(render_evaluations=0),enabled=False)
        assert status['status']=='prepared_offline' and status['offline_contract_validated'] and parsed is None
        folder=run/'gpt/round_00';request=read(folder/'request.json')
        assert sha(folder/'reference.png')==request['image_sha256s'][0]
        assert sha(folder/'render.png')==request['image_sha256s'][1]
        assert sha(folder/'overlay.png')==request['image_sha256s'][2]
    finally:shutil.rmtree(run,ignore_errors=True)

def test_gpt_http_budget_prevents_any_network_call(toy_scene,tmp_path,monkeypatch):
    import movens_sim.gpt_adapter as adapter
    from movens_sim.util import ROOT
    scene=Scene(toy_scene/'scene_fitting.xml');run=ROOT/'runs'/'_http_budget_test'
    try:
        gpt_run_fixture(scene,run);write(run/'gpt/gpt_budget.json',dict(http_attempts_used=12,max_http_attempts=12))
        config=read(ROOT/'configs/batch_v1.json');monkeypatch.setattr(adapter,'credentials',lambda c:'fixture-key-never-sent')
        monkeypatch.setattr(adapter,'OpenAI',lambda **kw:pytest.fail('Network client must not be constructed at exhausted budget'))
        status,parsed=adapter.call_or_prepare(scene,run,config,'appearance',0,dict(render_evaluations=0),enabled=True)
        assert status['status']=='api_budget_exhausted' and parsed is None and not status['api_call_performed']
    finally:shutil.rmtree(run,ignore_errors=True)
