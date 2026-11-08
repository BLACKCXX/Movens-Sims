"""Real renderer/MCP contracts plus isolated state-machine tests; no live API."""
import asyncio
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import pytest
from PIL import Image
from test_reconstruction_flow import session_fixture
from movens_sim.util import ROOT,read,write,save_rgb
from movens_sim.reconstruction.session import ReconstructionSession
from movens_sim.reconstruction.v5_tools import evaluate,images,optimize_issue,select_candidate,execution_fingerprint,camera_valid
from movens_sim.reconstruction.v5_metrics import appearance_metrics
from movens_sim.reconstruction.review_state import dependencies,invalidate,valid_check,CATEGORIES
from movens_sim.reconstruction.backend import SceneError
from movens_sim.reconstruction.budget import BudgetManager,BudgetExceeded


@pytest.fixture
def v5(session_fixture,monkeypatch):
    old=session_fixture;run=old.run;task=old.config['task_group_id'];old.close()
    for kind in ('static','physics'):
        p=run/'inputs'/('template_'+kind)/'scene.xml';root=ET.parse(p).getroot();asset=root.find('asset')
        texture=p.parent/'assets/test_rgb.png';img=np.zeros((64,64,3),np.uint8);img[:]=[180,65,20];Image.fromarray(img).save(texture)
        ET.SubElement(asset,'texture',name='apple_tex',type='2d',file=str(texture));ET.SubElement(asset,'material',name='apple_mat',texture='apple_tex',specular='.1',shininess='.2')
        for g in root.findall(".//body[@name='apple_root']//geom"):
            if g.get('group')=='1':g.set('material','apple_mat')
        for mesh in asset.findall('mesh'):
            if 'apple' not in mesh.get('name',''):continue
            path=(p.parent/mesh.get('file')).resolve();lines=path.read_text().splitlines()
            if not any(s.startswith('vt ') for s in lines):
                out=[s for s in lines if not s.startswith('f ')]+['vt 0 0','vt 1 0','vt .5 .15','vt .5 1']
                out+=['f '+' '.join(x+'/'+x for x in s.split()[1:]) for s in lines if s.startswith('f ')];path.write_text('\n'.join(out)+'\n')
        ET.SubElement(root.find('worldbody'),'light',name='probe_light',directional='true',dir='0 0 -1',diffuse='.8 .8 .8')
        ET.ElementTree(root).write(p)
    config=read(ROOT/'configs/ep012_reconstruction_v5.json');config['task_group_id']=task+'_v5';config['batch']['control_root']=None
    config['initialization']['mode']='fixture';write(run/'resolved_config.json',config)
    camera=read(run/'camera.json');camera['fovy_deg']=42.;camera['mount_rotation_to_world']=np.eye(3).tolist();write(run/'camera.json',camera)
    write(run/'session.json',dict(stage='geometry',shape_enabled=False,plate_tilt_enabled=False,camera_override=camera,rounds=[],operations=[],geometry_review_passed=False))
    write(run/'mask_review.json',dict(mask_quality_ok=True));write(run/'observation.json',dict(version='v5_synthetic_fixture',mask_sha256s={}))
    monkeypatch.setenv('MOVENS_SIM_PROJECT_ROOT',str(run.parent));s=ReconstructionSession(run,config);s.scene.apply({'light.ambient':.1,'light.diffuse':.1,'light.key_strength':.2},internal=True);s.state['observation_version']=s.observation['version'];s.save()
    base=evaluate(s);rgb,masks=images(s,base);s.observation['rgb']=rgb;s.observation['target']=masks
    s.observation['version']='v5_exact_synthetic_truth';write(run/'observation.json',dict(version=s.observation['version'],mask_sha256s={}))
    save_rgb(run/'reference.png',rgb);np.savez_compressed(run/'masks/observations.npz',**masks,valid=s.observation['valid']);write(run/'metrics_mount_baseline.json',base['metrics'])
    yield s
    s.close()


def test_independent_native_material_light_texture_and_portable_reload(v5,tmp_path):
    from movens_sim.reconstruction.diagnostics import load_state
    from movens_sim.render import Renderer
    s=v5;base=evaluate(s);rgb,masks=images(s,base)
    for key,value,target in [('apple.color_strength',1.,'apple'),('apple.specular',.9,'apple'),('plate.specular',.9,'plate'),('light.key_rx',25.,'apple')]:
        control=s.scene.params.copy()
        if key.endswith('.specular'):control[target+'.shininess']=.005
        base_image=images(s,evaluate(s,control))[0]
        row=evaluate(s,{**control,key:value});im,_=images(s,row)
        assert np.abs(im.astype(float)-base_image.astype(float))[masks[target]].max()>0,(key,target)
        if key=='apple.specular':np.testing.assert_array_equal(im[masks['plate']],base_image[masks['plate']])
    s.scene.stage='appearance';s.scene.apply({'apple.color_strength':1.,'apple.red_g':.03,'apple.specular':.7,'plate.specular':.2,'light.key_rx':15.})
    expected=s.scene.render()[0];path=s.scene.export(tmp_path/'portable');m,d=load_state(path)
    with Renderer(m) as renderer:actual=renderer.rgb(d)
    np.testing.assert_array_equal(expected,actual)


def test_zero_budget_no_improvement_and_staged_candidate_are_distinct(v5):
    s=v5;s.scene.stage='appearance'
    # Exact target already matches: a real search can run without improvement.
    r=optimize_issue(s,'cloth_color','cloth_coarse_v5','cloth_color',12,s.scene.candidate_id())
    assert r['execution_status']=='no_improvement' and r['candidate_evaluations']>0 and r['new_candidates']>0 and not r['candidate_retained']
    duplicate=optimize_issue(s,'cloth_color','cloth_coarse_v5','cloth_color',12,s.scene.candidate_id())
    assert duplicate['execution_status']=='no_improvement' and duplicate['new_candidates']==0 and duplicate['duplicate']
    left=s.budget.remaining('render_appearance');s.budget.reserve('render_appearance',left)
    r=optimize_issue(s,'apple_color','apple_color_v5','apple_color',12,s.scene.candidate_id())
    assert r['execution_status']=='budget_exhausted' and r['new_render_evaluations']==0


def test_real_mcp_optimization_and_model_choice_enter_export(v5,tmp_path):
    from movens_sim.reconstruction.mcp_server import call_tools
    s=v5;s.scene.stage='appearance';s.scene.apply({'cloth.gray':.55});s.save();old=s.scene.params.copy();s.close()
    response=asyncio.run(call_tools(s.run,[dict(tool='fit_appearance',arguments=dict(issue_id='cloth_color',objective_id='cloth_coarse_v5',parameter_group='cloth_color',render_cap=28,_current_candidate_id=True))]))
    assert response['calls'][0]['result']['ok'],response
    current=ReconstructionSession(s.run,s.config)
    try:
        result=response['calls'][0]['result']['result'];assert result['execution_status']=='completed' and result['pending_review']
        assert current.scene.params==old and current.state['pending_candidate']
    finally:current.close()
    selection=asyncio.run(call_tools(s.run,[dict(tool='select_candidate',arguments=dict(issue_id='cloth_color',decision='keep',evidence='Synthetic target exact gray; real MCP lowered fixed cloth objective',_current_candidate_id=True))]))
    assert selection['calls'][0]['result']['result']['candidate_retained']
    current=ReconstructionSession(s.run,s.config)
    try:
        assert current.scene.params['cloth.gray']!=old['cloth.gray']
        path=current.scene.export(tmp_path/'selected');root=ET.parse(path).getroot();g=next(x for x in root.findall('.//geom') if x.get('name','').startswith('frozen_'))
        assert float(g.get('rgba').split()[0])==pytest.approx(current.scene.params['cloth.gray'])
        assert current.budget.used('render_appearance')>0
    finally:current.close()


def test_fixed_regions_and_photometric_materials_do_not_invalidate_physics(v5):
    s=v5;base=evaluate(s);row=evaluate(s,{**s.scene.params,'plate.specular':.8,'apple.color_strength':1.})
    assert row['metrics']['v5']['region_fingerprint']==base['metrics']['v5']['region_fingerprint']
    params=s.scene.params.copy();camera=s.scene.locked_camera;obs=s.observation['version']
    state=dict(observation_version=obs,category_checks={c:dict(passed=True,valid=True,dependency_hash=dependencies(c,params,camera,obs)) for c in CATEGORIES})
    updated={**params,'apple.specular':.6};invalidate(state,params,updated,camera,camera)
    assert valid_check(state,'rest',updated,camera,obs) and valid_check(state,'geometry',updated,camera,obs)
    assert not valid_check(state,'appearance',updated,camera,obs)


def test_camera_bounds_never_accumulate(v5):
    s=v5;p=copy.deepcopy(s.scene.locked_camera);p['xyz_m'][0]+=.019;camera_valid(s,p)
    s.scene.locked_camera=p;q=copy.deepcopy(p);q['xyz_m'][0]+=.005
    with pytest.raises(SceneError):camera_valid(s,q)


def test_v5_export_preserves_named_saved_velocity(v5,tmp_path):
    from movens_sim.reconstruction.diagnostics import load_state
    s=v5;m=s.scene.model;d=s.scene.data;da=m.jnt_dofadr;qa=m.jnt_qposadr
    source=tmp_path/'saved_by_name.npz';velocity=np.linspace(.001,.004,m.nv)
    np.savez_compressed(source,joint_names=np.array([m.joint(i).name for i in range(m.njnt)]),qpos_addresses=qa,qpos_sizes=np.diff(np.r_[qa,m.nq]),qpos=d.qpos,dof_addresses=da,dof_sizes=np.diff(np.r_[da,m.nv]),qvel=velocity,time=np.array(.25))
    s.scene.physics_state=source;path=s.scene.export(tmp_path/'dynamic',physics=True);other,loaded=load_state(path)
    np.testing.assert_array_equal(loaded.qvel,velocity);assert loaded.time==.25


def test_optional_details_policy_does_not_waive_hard_coverage(v5):
    from movens_sim.reconstruction.v5_terra import validate_review
    from movens_sim.reconstruction.metrics import gate
    s=v5;s.state['active_category']='appearance'
    issues=[dict(issue_id=k,status='resolved',evidence='fixture required issue satisfied',confidence=.8,operation_id='fixture') for k in ('apple_color','apple_highlight','plate_shading','cloth_color')]+[dict(issue_id='cloth_detail',status='non_blocking_limitation',evidence='Fine wrinkles absent',confidence=1.,operation_id='')]
    parsed=dict(candidate_id=s.scene.candidate_id(),stage='APPEARANCE',verdict='fail',category_results=[dict(category='appearance',verdict='fail',evidence='Only optional fine wrinkles')],issue_updates=issues,proposed_actions=[],tradeoff_rationale='',next_hypothesis='')
    validate_review(parsed,s);assert parsed['category_results'][0]['verdict']=='pass_with_limitations'
    metrics=evaluate(s)['metrics'];metrics['table']['top_cloth_iou']=.87531
    assert 'table.top_cloth_iou.absolute' in gate(metrics,s.scene.geometry_metrics(),None,s.config)['failures']


def test_budget_resume_final_reserve_and_contract(v5,tmp_path):
    s=v5;b=s.budget;b.reserve_terra(detail=dict(candidate_id='x'));again=BudgetManager(b.path,b.task_id,s.config['budget']);assert again.used('terra_requests')==1
    for _ in range(18):again.reserve_terra(detail=dict(candidate_id='x'))
    with pytest.raises(BudgetExceeded):again.reserve_terra()
    again.reserve_terra(True,dict(candidate_id='x',delivery_hash='hash'))
    with pytest.raises(BudgetExceeded):again.reserve_terra(True,dict(candidate_id='x',delivery_hash='hash'))
    from movens_sim.reconstruction.v5_tools import make_server
    server=make_server(s);assert server is not None


def test_installed_native_rgb_texture_transfer():
    from movens_sim.reconstruction.v5_diagnostics import color_pipeline_probe
    result=color_pipeline_probe();assert result['native_passthrough'],result


def test_recolor_preserves_dark_upper_stalk_uv_and_skin_detail(tmp_path):
    import io
    from movens_sim.reconstruction.v5_material import recolor
    from movens_sim.reconstruction.appearance import height_atlas
    mesh=tmp_path/'stalk_uv.obj';mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 1\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n')
    atlas=height_atlas(str(mesh),64,64);img=np.full((64,64,3),[170,40,20],np.uint8);stalk=(atlas>.7)&(np.indices(atlas.shape)[1]<12);img[stalk]=[55,30,12]
    buf=io.BytesIO();Image.fromarray(img).save(buf,format='PNG')
    data=recolor(buf.getvalue(),str(mesh),(1.,.42,.1,1.,.1,.03,1.,.88,.1));out=np.asarray(Image.open(io.BytesIO(data)))
    np.testing.assert_array_equal(out[stalk],img[stalk]);assert np.abs(out[~stalk].astype(float)-img[~stalk]).max()>10


def test_v5_nonfinite_search_and_geometry_appearance_contract(v5):
    s=v5
    with pytest.raises(SceneError):evaluate(s,{**s.scene.params,'apple.red_r':float('nan')})
    with pytest.raises(SceneError):optimize_issue(s,'coverage','coverage_v5','table_coverage',10,s.scene.candidate_id(),evidence='fixture',confidence=float('nan'))
    from movens_sim.reconstruction.v5_tools import make_server,GROUPS,group_stage
    assert group_stage('plate_profile')=='geometry' and 'plate.depth_ratio' in GROUPS['plate_profile']
    with pytest.raises(SceneError):s.fit_appearance(['plate.depth_ratio'],s.scene.candidate_id())


def test_pending_appearance_chain_cannot_spend_other_category_recheck_slots(v5):
    from movens_sim.reconstruction.v5_terra import bundle
    s=v5;s.state['active_category']='appearance';s.state['category_attempts']={c:1 for c in CATEGORIES};s.state['category_attempts']['appearance']=10
    s.state['pending_candidate']=None;s.state['evidence_revision']=1
    row=evaluate(s);rgb,masks=images(s,row)
    for _ in range(14):s.budget.reserve_terra(detail=dict(candidate_id=s.scene.candidate_id(),category='appearance'))
    folder,body=bundle(s,row['metrics'],rgb,masks)
    state=read(folder/'request.json')['state']
    assert not state['mutations_allowed'],'Six remaining slots must preserve feedback plus four other rechecks and final'


def test_failed_categories_rotate_beyond_second_round_with_pending_appearance(v5):
    from movens_sim.reconstruction.v5 import next_category
    s=v5;s.state['category_checks']={};s.state['pending_candidate']={'issue_id':'plate_shading'}
    attempted=dict(observation=2,geometry=3,initial_support=2,appearance=8,rest=2)
    assert next_category(s,attempted)=='observation'
    attempted['observation']=4
    assert next_category(s,attempted)=='initial_support'


def test_joint_lighting_preserves_geometry_and_reports_every_object(v5):
    s=v5;geometry={k:v for k,v in s.scene.params.items() if k in {p['id'] for p in s.scene.registry() if 'geometry' in p['stages']}};camera=copy.deepcopy(s.scene.locked_camera)
    report=optimize_issue(s,'plate_shading','scene_lighting_v5','plate_and_lighting',32,s.scene.candidate_id(),evidence='Joint photometric hypothesis under unchanged geometry',confidence=.9)
    assert report['candidate_evaluations']>0 and report['new_render_evaluations']>0
    assert all(s.scene.params[k]==v for k,v in geometry.items()) and s.scene.locked_camera==camera
    for key in ('metrics_before','metrics_after'):assert set(report[key]['appearance'])=={'apple','plate','cloth'}
    assert not report['candidate_retained']


def test_unchanged_uncertain_geometry_defers_without_becoming_a_pass(v5):
    from movens_sim.reconstruction.v5 import next_category,review_needed
    s=v5;obs=s.observation['version'];camera=s.scene.locked_camera;params=s.scene.params
    s.state['category_checks']={c:dict(passed=c!='geometry',valid=True,dependency_hash=dependencies(c,params,camera,obs)) for c in CATEGORIES}
    review=dict(category_results=[dict(category='geometry',verdict='needs_review')]);s.state['review_history']=[dict(dependency_hash=dependencies('geometry',params,camera,obs),review=review) for _ in range(2)]
    attempted={c:2 for c in CATEGORIES};s.state['pending_candidate']=None
    assert next_category(s,attempted) is None and not valid_check(s.state,'geometry',params,camera,obs)
    assert 'geometry' in s.state['deferred_checks']
    s.scene.stage='geometry';s.scene.apply({'plate.scale':1.05})
    assert review_needed(s,'geometry',attempted)
    assert next_category(s,attempted)=='observation','Scale also invalidates coverage, which is checked first'


def test_unchanged_image_does_not_hide_open_appearance_search_feedback(v5):
    from movens_sim.reconstruction.v5 import next_category,review_needed
    s=v5;obs=s.observation['version'];camera=s.scene.locked_camera;params=s.scene.params
    s.state['category_checks']={c:dict(passed=c!='appearance',valid=True,dependency_hash=dependencies(c,params,camera,obs)) for c in CATEGORIES}
    review=dict(category_results=[dict(category='appearance',verdict='needs_review')])
    s.state['review_history']=[dict(dependency_hash=dependencies('appearance',params,camera,obs),review=review) for _ in range(2)]
    s.state['issues']={'apple_color':dict(category='appearance',blocking=True,status='open')}
    s.state['tool_feedback']=[dict(tool='fit_appearance',execution_status='no_improvement',new_candidates=47)]
    s.state['pending_candidate']=None;attempted={c:2 for c in CATEGORIES}
    assert review_needed(s,'appearance',attempted)
    assert next_category(s,attempted)=='appearance'
    assert not valid_check(s.state,'appearance',params,camera,obs)
    s.state['issues']['apple_color']['status']='capability_gap'
    assert not review_needed(s,'appearance',attempted),'Explicit capability limit may stop unchanged evidence, never an open repair'


def test_comparison_supports_real_fixed_window_crops_and_original_full_frames():
    from movens_sim.head_camera import labelled_comparison
    crop=np.full((128,130,3),[70,120,200],np.uint8)
    image=labelled_comparison([crop,crop,crop],['REAL','V4','V5'])
    assert image.shape==(160,390,3)
    np.testing.assert_array_equal(image[32:,:130],crop)
    full=np.zeros((480,640,3),np.uint8)
    assert labelled_comparison([full,full],['REAL','SIM']).shape==(512,1280,3)


def test_warm_start_same_camera_cannot_reuse_pre_environment_model(v5):
    s=v5;s.scene.stage='geometry';s.scene.apply({'cloth.coverage':1.,'table.depth_scale':1.2});expected=s.scene.render()[0];params=s.scene.params.copy();s.save()
    write(s.run/'initial_parameters.json',params);s.close()
    current=ReconstructionSession(s.run,s.config)
    try:
        actual=current.scene.render()[0];np.testing.assert_array_equal(actual,expected)
        assert current.scene.params['cloth.coverage']==1.
        assert current.scene.locked_camera==read(s.run/'camera.json')
    finally:current.close()
