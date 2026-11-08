"""Cold-start contracts; regression fixtures are separate from actual0/1/2 runs."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import mujoco
import numpy as np
import pytest
from movens_sim.util import ROOT,read,write
from movens_sim.reconstruction.modes import initial_xml
from movens_sim.reconstruction.initial_xml import named_state,apply_named_state,validate_entry
from movens_sim.reconstruction.search_replay import SearchReplay
from movens_sim.reconstruction.ablation import validate_config
from test_reconstruction_v5 import v5
from test_reconstruction_flow import session_fixture


def test_cold_config_independent_source_and_identical_limits():
    c=read(ROOT/'configs/ep012_initial_xml_no_terra.json');old=read(ROOT/'configs/ep012_ablation_no_terra.json');validate_config(c)
    assert initial_xml(c) and not initial_xml(old)
    assert c['budget']==old['budget'] and c['initialization']['mode']=='fk_h0'
    assert 'warm_start_root' not in c['initialization'] and 'reference_root' not in c['ablation']
    assert c['initial_geometry']['render_pool']=='render_repair' and c['initial_geometry']['render_cap']==600
    assert c['fit']==old['fit'] and c['v5']==old['v5'] and c['acceptance']==old['acceptance']
    m=read(ROOT/c['manifest']);assert [x['episode'] for x in m['entries']]==[0,1,2]
    for e in m['entries']:
        assert e['initial_state']['mode']=='fresh_xml_reset'
        assert 'Movens-Interact/outputs/' in e['scene_dir'] and '/runs/' not in e['scene_dir']


def test_named_state_preserves_qpos_qvel_time_across_joint_reordering(tmp_path):
    first=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="a"><joint name="ja" type="slide"/><geom size=".1"/></body><body name="b"><joint name="jb" type="slide"/><geom size=".1"/></body></worldbody></mujoco>')
    second=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="b"><joint name="jb" type="slide"/><geom size=".1"/></body><body name="a"><joint name="ja" type="slide"/><geom size=".1"/></body></worldbody></mujoco>')
    data=mujoco.MjData(first);data.qpos[:]=[.12,.34];data.qvel[:]=[.56,.78];data.time=.91
    p=tmp_path/'state.npz';named_state(first,data,p);other=mujoco.MjData(second);apply_named_state(second,other,p)
    np.testing.assert_array_equal(other.qpos,[.34,.12]);np.testing.assert_array_equal(other.qvel,[.78,.56]);assert other.time==.91


def test_replay_resumes_original_objective_prefix_without_reexecuting_it(tmp_path):
    p=tmp_path/'progress.json';a=SearchReplay(p,'fixed',dict(cap=3));calls=[]
    for i in range(2):assert a.call({'x':i},lambda i=i:calls.append(i) or i*i)==(i*i,False)
    def interrupted():raise RuntimeError('synthetic interruption')
    with pytest.raises(RuntimeError):a.call({'x':2},interrupted)
    b=SearchReplay(p,'fixed',dict(cap=99));assert b.data['metadata']['cap']==3
    for i in range(2):assert b.call({'x':i},lambda:pytest.fail('Reexecuted charged prefix'))==(i*i,True)
    assert b.call({'x':2},lambda:4)==(4,False)
    assert len(b.data['calls'])==3 and b.replayed==2 and calls==[0,1]
    with pytest.raises(ValueError,match='identity changed'):SearchReplay(p,'new')
    with pytest.raises(ValueError,match='prefix diverged'):SearchReplay(p,'fixed').call({'x':9},lambda:0)


def test_missing_source_fields_fail_explicitly_without_history_fallback():
    c=read(ROOT/'configs/ep012_initial_xml_no_terra.json')
    with pytest.raises(ValueError,match='entry missing'):validate_entry({'episode':0},c)
    e=read(ROOT/c['manifest'])['entries'][0];e['scene_dir']=str(ROOT/'runs/ep012_reconstruction_v5_20261108/episode_000000/delivery/static/portable')
    with pytest.raises(ValueError,match='declared zero-offset'):validate_entry(e,c)


def test_cold_search_replays_interrupted_prefix_with_original_cap(v5,monkeypatch):
    import movens_sim.reconstruction.v5_tools as tools
    s=v5;s.config['workflow']='ablation_no_terra_v5';s.config['initialization']['source_mode']='initial_xml';s.scene.stage='appearance';s.scene.apply({'cloth.gray':.55});s.save()
    original=tools.minimize
    def stop(*a,**k):raise RuntimeError('synthetic stop after existing seeds')
    monkeypatch.setattr(tools,'minimize',stop)
    with pytest.raises(RuntimeError,match='synthetic stop'):tools.optimize_issue(s,'cloth_color','cloth_coarse_v5','cloth_color',16,s.scene.candidate_id())
    p=next((s.run/'search_progress').glob('*.json'));prefix=read(p);used=s.budget.used('render_appearance')
    monkeypatch.setattr(tools,'minimize',original)
    result=tools.optimize_issue(s,'cloth_color','cloth_coarse_v5','cloth_color',16,s.scene.candidate_id())
    assert result['candidate_evaluations']<=16 and result['new_render_evaluations']<=17
    assert read(p)['metadata']['cap']==16
    assert read(p)['calls'][:len(prefix['calls'])]==prefix['calls']
    assert s.budget.used('render_appearance')>=used and len(s.state['search_history'])==1
    assert s.budget.used('terra_requests')==0


def test_initial_geometry_reuses_kernel_repair_pool_and_stages_selection(v5):
    from movens_sim.reconstruction.cold_geometry import fit_initial_geometry
    from movens_sim.reconstruction.v5_tools import select_candidate
    s=v5;s.config['workflow']='ablation_no_terra_v5';s.config['initialization']['source_mode']='initial_xml';s.config['initial_geometry']=read(ROOT/'configs/ep012_initial_xml_no_terra.json')['initial_geometry']
    s.scene.stage='geometry';s.scene.apply({'apple.scale':.8,'plate.scale':.8});s.save();params=s.scene.params.copy();initial_used=s.budget.used('render_initial')
    result=fit_initial_geometry(s,s.scene.candidate_id())
    assert result['new_render_evaluations']>0 and result['new_render_evaluations']<=600
    assert s.budget.used('render_initial')==initial_used and s.budget.used('geometry_rounds')==1
    assert s.scene.params==params,'Numerical kernel result must remain staged until select_candidate'
    assert 'Sobol' in result['kernel_result']['algorithm']
    if s.state.get('pending_candidate'):
        select_candidate(s,'geometry','keep',s.scene.candidate_id(),'Existing numeric objective and hard guards')
        assert s.scene.params!=params



def test_existing_projection_refit_default_path_is_unchanged(v5):
    from movens_sim.reconstruction.optimize import projection_refit
    s=v5;p=projection_refit(s.scene,s.observation,s.budget,s.scene.params)
    assert all(np.isfinite(p[k]) for k in ('apple.x','apple.y','apple.scale','plate.x','plate.y','plate.scale'))
    assert s.budget.used('projection_evaluations')>0


def test_missing_object_mask_preserves_other_independent_targets():
    import cv2
    from movens_sim.reconstruction.full_prepare import _pixels
    rgb=np.full((480,640,3),60,np.uint8);cv2.ellipse(rgb,(420,330),(65,40),0,0,360,(230,230,230),-1)
    seeds={'apple':np.array([120,100]),'plate':np.array([420,330])};robot=np.zeros((480,640),bool)
    with pytest.raises(ValueError):_pixels(rgb,robot,seeds)
    masks,valid,occlusion,quality=_pixels(rgb,robot,seeds,allow_partial=True)
    assert not masks['apple'].any() and masks['plate'].any() and masks['cloth'].any()
    assert not quality['apple']['quality_ok'] and quality['apple']['failure_reason'] and quality['plate']['quality_ok']


def test_guard_can_improve_existing_failure_and_protect_passing_check(v5):
    from movens_sim.reconstruction.v5_tools import evaluate,guard_failures
    s=v5;s.scene.apply({'apple.scale':.8,'plate.scale':.8});before=evaluate(s)
    after=evaluate(s,{**s.scene.params,'apple.scale':1.,'plate.scale':1.})
    assert not guard_failures(s,after,before)
    bad=copy.deepcopy(after);bad['metrics']['objects']['plate']['centroid_error_px']=20.
    assert 'plate.center' in guard_failures(s,bad,before)


def test_replay_recovers_slots_after_last_periodic_checkpoint(tmp_path):
    p=tmp_path/'long_progress.json';a=SearchReplay(p,'same',dict(cap=100))
    for i in range(77):a.call({'x':i},lambda i=i:i+3)
    def stop():raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):a.call({'x':77},stop)
    b=SearchReplay(p,'same');assert len(b.data['calls'])==78
    for i in range(77):assert b.call({'x':i},lambda:pytest.fail('Reexecuted prefix'))==(i+3,True)
    assert b.call({'x':77},lambda:80)==(80,False)
    b.flush(force=True);assert len(read(p)['calls'])==78


def test_cold_prepare_has_no_old_experiment_config_or_result_dependency(tmp_path,monkeypatch):
    from movens_sim.reconstruction import cold_delivery
    c=read(ROOT/'configs/ep012_initial_xml_no_terra.json');c['run_root']=str(tmp_path/'new_run')
    actual=read
    def declared_only(path):
        text=str(path)
        if 'ep012_reconstruction_' in text or 'ep012_ablation_no_terra' in text or '/runs/' in text and str(tmp_path) not in text:
            pytest.fail('Historical configuration/result dependency: '+text)
        return actual(path)
    monkeypatch.setattr(cold_delivery,'read',declared_only)
    assert cold_delivery.prepare(c)['all_local_limits_identical']


def test_code_change_cannot_reopen_inflight_allowance(tmp_path):
    from movens_sim.reconstruction.ablation import guard_code_continuation
    from movens_sim.reconstruction.budget import BudgetManager
    c=read(ROOT/'configs/ep012_initial_xml_no_terra.json');b=BudgetManager(tmp_path/'budget.sqlite','synthetic_cold',c['budget'])
    b.reserve_render('render_repair',{'already_spent':True});before=b.summary()['used']
    for state in ({'cold_start':{'initial_geometry':{'status':'started'}}},{'ablation':{'steps':{'apple_color':{'status':'search_started','render_cap':128}}}},{'pending_candidate':{'operation_id':'original'}}):
        write(tmp_path/'session.json',state)
        with pytest.raises(ValueError,match='cold_inflight_code_change'):guard_code_continuation(c,tmp_path)
        assert b.summary()['used']==before and read(tmp_path/'session.json')==state
    write(tmp_path/'session.json',{'ablation':{'steps':{'apple_color':{'status':'done'}}}});guard_code_continuation(c,tmp_path)
    guard_code_continuation({'workflow':'terra_closed_loop_v5'},tmp_path/'not_used')

def test_isolated_original_input_prepare_real_mcp_export_independent_accept(tmp_path):
    """Actual original0 import and new1s physics, not an actual experiment result."""
    project=ROOT;c=read(project/'configs/ep012_initial_xml_no_terra.json');m=read(project/c['manifest'])
    c['task_group_id']='regression_cold_'+uuid.uuid4().hex;c['run_root']='runs/isolated_cold';c['manifest']='configs/isolated_inputs.json';c['batch']['control_root']=None;c['initialization']['data_root']=str(project/'data/full402_h0_v1')
    write(tmp_path/c['manifest'],m);write(tmp_path/'config.json',c)
    (tmp_path/'.venv').symlink_to(project/'.venv',target_is_directory=True)
    (tmp_path/'configs').mkdir(exist_ok=True)
    import shutil
    shutil.copy2(project/'configs/full402_h0_v1.json',tmp_path/'configs/full402_h0_v1.json')
    code=r'''
import json,sys
from pathlib import Path
from movens_sim.util import ROOT,read,write
from movens_sim.reconstruction.ablation import run_pipeline
from movens_sim.reconstruction.session import ReconstructionSession
from movens_sim.reconstruction.classified import execute
from movens_sim.reconstruction.v5 import export_and_accept
from movens_sim.reconstruction import terra
from movens_sim import gpt_adapter
c=read(ROOT/'config.json');entry=read(ROOT/c['manifest'])['entries'][0]
prepared=run_pipeline(entry,c,command='prepare');assert prepared['prepared']
run=ROOT/c['run_root']/'episode_000000';s=ReconstructionSession(run,c)
try:
    for old in ('ep012_reconstruction_v4_20261107','ep012_reconstruction_v5_20261108','ep012_ablation_no_terra_20261108'):
        try:Path(sys.argv[1],'runs',old,'result.json').read_text()
        except PermissionError:pass
        else:raise AssertionError('Historical read was not blocked')
    for fn,args in [(terra.request_review,(s,{},None,{})),(terra._request_review,(s,{},None,{})),(gpt_adapter.call_or_prepare,(None,run,c,'geometry',0,None))]:
        try:fn(*args,enabled=False)
        except RuntimeError as e:assert 'model_disabled_ablation' in str(e)
        else:raise AssertionError('Model/cache path reached')
    s.scene.stage='appearance';s.scene.apply({'cloth.gray':.55});s.save()
    s,out=execute(s,[dict(tool='fit_appearance',arguments=dict(issue_id='cloth_color',objective_id='cloth_coarse_v5',parameter_group='cloth_color',render_cap=28))],'Isolated regression: real existing numerical fit')
    assert out['calls'][-1]['result']['ok']
    assert out['calls'][-1]['result']['result']['new_candidates']>0
    if s.state.get('pending_candidate'):
        s,out=execute(s,[dict(tool='select_candidate',arguments=dict(issue_id='cloth_color',decision='keep',evidence='Fixture numerical improvement and original hard guard'))],'Isolated regression: actual selection')
        assert out['calls'][-1]['result']['ok']
    binding=export_and_accept(s);a=read(run/'final_acceptance/acceptance.json')
    assert a['engineering_passed'] and a['reload_passed'] and a['fresh_native_integration']
    assert a['physical']['steps']==4000 and abs(a['physical']['duration_s']-1)<1e-9
    assert a['validator_pid']!=a['parent_pid']
    assert read(run/'initial_xml_verification.json')['passed']
    assert not (run/'baseline_v4').exists() and not (run/'terra').exists()
    assert s.budget.used('terra_requests')==0 and s.budget.used('render_initial')>=3
    assert any(x['detail'].get('kind')=='initial_xml_prepare' for x in s.budget.summary()['events'])
    print(json.dumps(dict(isolated=True,engineering=a['engineering_passed'],fresh_steps=a['physical']['steps'],validator_pid=a['validator_pid'],model_requests=0)))
finally:s.close()
'''
    env={**os.environ,'MOVENS_SIM_PROJECT_ROOT':str(tmp_path),'PYTHONPATH':str(project/'src'),'MUJOCO_GL':'egl','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    result=subprocess.run([sys.executable,'-B','-u','-c',code,str(project)],cwd=project,env=env,capture_output=True,text=True,timeout=900)
    (tmp_path/'isolation.log').write_text(result.stdout+'\n'+result.stderr)
    assert result.returncode==0,result.stdout[-2000:]+'\n'+result.stderr[-6000:]
