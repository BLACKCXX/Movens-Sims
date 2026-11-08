"""Behavior tests; synthetic fixture is not one of the three actual ablation runs."""
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from test_reconstruction_v5 import v5
from test_reconstruction_flow import session_fixture
from movens_sim.util import ROOT,read,write,now
from movens_sim.reconstruction.ablation import POLICY,APPEARANCE,validate_config,search_step,selection_rule,has_search_budget,report,numeric_trigger
from movens_sim.reconstruction.budget import BudgetManager,BudgetExceeded
from movens_sim.reconstruction.modes import assert_model_allowed
from movens_sim.reconstruction.session import ReconstructionSession
from movens_sim.reconstruction.v5_tools import optimize_issue,evaluate


@pytest.fixture
def b(v5):
    s=v5;c=copy.deepcopy(s.config);c['workflow']='ablation_no_terra_v5';c['task_group_id']+='_no_terra'
    c['budget']['terra_requests']=0;c['terra']['enabled']=False;c['ablation']=dict(policy=POLICY)
    s.state['ablation']=dict(policy=POLICY,policy_hash=__import__('movens_sim.util',fromlist=['digest']).digest(POLICY),steps={},diagnostics_done=[])
    s.save();s.close();write(s.run/'resolved_config.json',c)
    current=ReconstructionSession(s.run,c)
    yield current
    current.close()


def test_config_limits_and_fixed_groups_are_identical_to_frozen_A():
    a=read(ROOT/'runs/ep012_reconstruction_v5_20261108/resolved_config.json');b=read(ROOT/'configs/ep012_ablation_no_terra.json')
    validate_config(b)
    assert b['budget']=={**a['budget'],'terra_requests':0}
    for key in ('fit','camera','evaluator','acceptance','initial_support','rest_acceptance','v5','appearance'):
        assert b[key]==a[key]
    assert [s[2] for s in APPEARANCE]==['apple_color','apple_red','apple_yellow','apple_highlight','plate_material','shared_lighting','plate_and_lighting','cloth_color']
    assert POLICY['passes']==1 and not POLICY['cloth_optional']


def test_model_paths_fail_before_credentials_caches_or_offline_fixtures(tmp_path,monkeypatch):
    from movens_sim.reconstruction import terra
    from movens_sim import gpt_adapter
    config={'workflow':'ablation_no_terra_v5'};s=SimpleNamespace(config=config,run=tmp_path)
    def forbidden(*a,**kw):pytest.fail('Credentials/cache/client path reached')
    monkeypatch.setattr(terra,'credentials',forbidden);monkeypatch.setattr(terra,'bundle',forbidden)
    monkeypatch.setattr(gpt_adapter,'request_bundle',forbidden)
    for fn in (terra.request_review,terra._request_review):
        with pytest.raises(RuntimeError,match='model_disabled_ablation'):fn(s,{},None,{},enabled=False)
    with pytest.raises(RuntimeError,match='model_disabled_ablation'):
        gpt_adapter.call_or_prepare(None,tmp_path,config,'geometry',0,None,enabled=True)
    assert not list(tmp_path.iterdir())
    assert_model_allowed({'workflow':'terra_closed_loop_v5'})


def test_zero_model_budget_cannot_reserve_final_or_normal_request(tmp_path):
    limits=read(ROOT/'configs/ep012_ablation_no_terra.json')['budget'];p=tmp_path/'budget.sqlite'
    a=BudgetManager(p,'synthetic_test',limits)
    for final in (False,True):
        with pytest.raises(BudgetExceeded,match='forbidden'):a.reserve_terra(final)
    assert a.used('terra_requests')==0 and not a.summary()['events']
    a.reserve('render_appearance',3);a.reserve('render_total',3)
    resumed=BudgetManager(p,'synthetic_test',limits)
    assert resumed.used('render_total')==3
    with pytest.raises(ValueError,match='changed'):BudgetManager(p,'synthetic_test',{**limits,'render_total':4501})


def test_numeric_geometry_evidence_does_not_forge_model_confidence(b):
    s=b;proof=numeric_trigger(s,'table_coverage');assert not proof['eligible']
    with pytest.raises(ValueError,match='numeric evidence'):
        optimize_issue(s,'coverage','coverage_v5','table_coverage',12,s.scene.candidate_id(),evidence='model says so',confidence=1.)
    proof['eligible']=True
    with pytest.raises(ValueError,match='Stale, ineligible'):
        optimize_issue(s,'coverage','coverage_v5','table_coverage',12,s.scene.candidate_id(),evidence=json.dumps(proof),confidence=0.)


def test_real_mcp_search_choice_export_and_resume_with_no_model(b,tmp_path,monkeypatch):
    from movens_sim.reconstruction.diagnostics import load_state
    from movens_sim.render import Renderer
    s=b;s.scene.stage='appearance';s.scene.apply({'cloth.gray':.55});s.save();before=s.scene.params.copy()
    # The real stdio child also uses the zero-model configuration and guard.
    s=search_step(s,'cloth_color','cloth_coarse_v5','cloth_color')
    try:
        assert s.scene.params['cloth.gray']!=before['cloth.gray']
        assert s.budget.used('terra_requests')==0 and not (s.run/'terra').exists()
        calls=(s.run/'mcp_tool_calls.jsonl').read_text();assert '"tool": "fit_appearance"' in calls and '"tool": "select_candidate"' in calls
        expected=s.scene.render()[0];xml=s.scene.export(tmp_path/'new_B_scene');m,d=load_state(xml)
        with Renderer(m) as renderer:actual=renderer.rgb(d)
        np.testing.assert_array_equal(actual,expected)
        used=s.budget.used('render_total');s=search_step(s,'cloth_color','cloth_coarse_v5','cloth_color')
        assert s.budget.used('render_total')==used and (s.run/'mcp_tool_calls.jsonl').read_text()==calls
        # Reporting semantics only: a clearly synthetic acceptance receipt tests
        # removal of the old minimum-HTTP/model-verdict completion dependency.
        row=evaluate(s);write(s.run/'final_acceptance/acceptance.json',dict(engineering_passed=True,geometry_numeric_passed=True,coverage_numeric_passed=True,initial_support_passed=True,physical_passed=True,reload_passed=True,metrics=row['metrics'],geometry=row['geometry'],camera_verified_from_xml=s.scene.locked_camera))
        write(s.run/'execution_started.json',dict(utc=now()));s.state['complete']=True;s.state['ablation']['finished_utc']=now();s.save()
        write(s.run/'run_identity.json',dict(code={'fixture':'explicit_synthetic_identity'}))
        monkeypatch.setattr('movens_sim.reconstruction.ablation_delivery.episode_report',lambda *args:None)
        result=report(s)
        assert result['processing_complete'] and result['terra_actual_http_requests']==0
        assert result['final_terra_verdict']=='not_run_ablation' and result['appearance_passed'] is None and result['visual_qualified'] is None
    finally:s.close()


def test_existing_numeric_selection_rolls_back_regression_and_reports_shared_tradeoff(b):
    s=b;row=evaluate(s);after=copy.deepcopy(row)
    after['metrics']['v5']['objectives']['scene_lighting_v5']-=.001
    # No extra per-object appearance hard gate: keep the declared joint gain.
    after['metrics']['v5']['apple_color']['red_lab_mae']+=2
    pending=dict(before=row,after=after)
    decision,evidence=selection_rule(s,pending,'scene_lighting_v5')
    assert decision=='keep' and evidence['appearance_submetric_tradeoffs_permitted']
    after['metrics']['objects']['apple']['iou']=.1
    assert selection_rule(s,pending,'scene_lighting_v5')[0]=='rollback'


def test_final_reserve_and_interrupted_search_are_not_replayed(b):
    s=b;s.state['ablation']['steps']['apple_color']=dict(status='search_started',operation_id='op_missing',parameter_group='apple_color')
    s.save();before=s.budget.used('render_total');s=search_step(s,'apple_color','apple_color_v5','apple_color')
    assert s.state['ablation']['steps']['apple_color']['status']=='interrupted' and s.budget.used('render_total')==before
    s.budget.reserve('optimization_seconds',s.budget.remaining('optimization_seconds')-1200)
    assert not has_search_budget(s,'apple_red')
    s=search_step(s,'apple_color','apple_color_v5','apple_red')
    assert s.state['ablation']['steps']['apple_red']['status']=='skipped'


def test_pending_real_mcp_result_resumes_selection_without_repeating_search(b):
    from movens_sim.reconstruction.classified import execute
    s=b;s.scene.stage='appearance';s.scene.apply({'cloth.gray':.55})
    s.state['ablation']['steps']['cloth_color']=dict(status='search_started',operation_id='op_0000',parameter_group='cloth_color')
    s.save();s,response=execute(s,[dict(tool='fit_appearance',arguments=dict(issue_id='cloth_color',objective_id='cloth_coarse_v5',parameter_group='cloth_color',render_cap=28))],'Synthetic interruption after real MCP search')
    try:
        assert s.state['pending_candidate'];before=s.budget.used('render_total')
        s=search_step(s,'cloth_color','cloth_coarse_v5','cloth_color')
        assert s.budget.used('render_total')==before and not s.state['pending_candidate']
        calls=[json.loads(x) for x in (s.run/'mcp_tool_calls.jsonl').read_text().splitlines()]
        assert sum(x['tool']=='fit_appearance' for x in calls)==1 and sum(x['tool']=='select_candidate' for x in calls)==1
        # Parent stopped after a successful selection but before recording done.
        s.state['ablation']['steps']['cloth_color']['status']='selection_started';s.save()
        s=search_step(s,'cloth_color','cloth_coarse_v5','cloth_color')
        assert len((s.run/'mcp_tool_calls.jsonl').read_text().splitlines())==len(calls)
    finally:s.close()


def test_fixed_controller_dispatches_each_existing_group_once(b,monkeypatch):
    import movens_sim.reconstruction.ablation as module
    calls=[]
    def step(s,issue,objective_id,group):calls.append(group);return s
    monkeypatch.setattr(module,'search_step',step)
    monkeypatch.setattr(module,'execute',lambda s,*args:(s,{}))
    module.run_fixed_pass(b)
    assert calls==[x[2] for x in module.GEOMETRY]+[x[2] for x in APPEARANCE]
    assert len(calls)==len(set(calls)) and 'cloth_optional' not in calls
