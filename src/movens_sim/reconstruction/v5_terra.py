"""Version-bound multi-template Terra decisions with complete tool feedback."""
import base64
import copy
import json
import cv2
import jsonschema
import numpy as np
from .backend import SceneError
from .review_state import CATEGORIES,protected_slots
from .v5_metrics import OBJECTIVES,POLICY
from .v5_tools import GROUPS,ISSUES,TOOLS
from ..util import CODE_ROOT,digest,read,write,sha,save_rgb,now


def review_key(s,final=False):
    return ('final_readonly' if final else s.state.get('active_category','observation'))+'_'+s.scene.candidate_id()+'_'+str(s.state.get('evidence_revision',0))


def response_schema(scene,final=False):
    def obj(p):return dict(type='object',properties=p,required=list(p),additionalProperties=False)
    def enum(values):return dict(type='string',enum=list(values))
    text=dict(type='string');number=dict(type='number')
    update=obj(dict(issue_id=enum(ISSUES),status=enum(['open','needs_evidence','resolved','capability_gap','non_blocking_limitation']),evidence=text,confidence=dict(type='number',minimum=0,maximum=1),operation_id=text))
    bound=obj(dict(param_id=enum([p['id'] for p in scene.registry()]),lower=number,upper=number))
    action=obj(dict(tool=enum(TOOLS[1:]),issue_id=enum(ISSUES),objective_id=enum(OBJECTIVES),parameter_group=enum(GROUPS),render_cap=dict(type='integer',minimum=0,maximum=256),bounds=dict(type='array',items=bound),decision=enum(['none','keep','rollback']),diagnostic=enum(['automatic','plate_views','native_rest']),evidence=text,confidence=dict(type='number',minimum=0,maximum=1)))
    category=obj(dict(category=enum(CATEGORIES),verdict=enum(['pass','pass_with_limitations','needs_review','fail']),evidence=text))
    return obj(dict(candidate_id=text,stage=enum(['OBSERVATION','GEOMETRY','INITIAL_SUPPORT','APPEARANCE','REST','FINAL_READONLY']),verdict=enum(['pass','pass_with_limitations','repair','needs_review','fail']),category_results=dict(type='array',items=category,minItems=5 if final else 1,maxItems=5 if final else 1),issue_updates=dict(type='array',items=update),proposed_actions=dict(type='array',items=action,maxItems=0 if final else 3),tradeoff_rationale=text,next_hypothesis=text))


def validate_review(parsed,s,final=False):
    jsonschema.validate(parsed,response_schema(s.scene,final))
    if parsed['candidate_id']!=s.scene.candidate_id():raise SceneError('stale_candidate','Terra must address current selected scene')
    expected='FINAL_READONLY' if final else s.state['active_category'].upper()
    if parsed['stage']!=expected:raise SceneError('wrong_stage',expected)
    if sorted(r['category'] for r in parsed['category_results'])!=sorted(CATEGORIES if final else [s.state['active_category']]):raise SceneError('category_contract','Exactly requested categories required')
    for action in parsed['proposed_actions']:
        if action['tool'] in ('optimize_issue','fit_appearance'):
            if not action['render_cap']:raise SceneError('invalid_local_budget','Search requires positive cap')
            if action['tool']=='fit_appearance' and action['parameter_group'] in ('table_coverage','camera_coverage','object_layout','plate_pose','plate_profile','support_initialization'):raise SceneError('wrong_stage','Appearance tool cannot edit geometry')
            reg={p['id']:p for p in s.scene.registry()};ids=GROUPS[action['parameter_group']]
            for b in action['bounds']:
                if b['param_id'] not in ids:raise SceneError('invalid_bounds',b['param_id'])
                p=reg[b['param_id']]
                if not p['lower']<=b['lower']<b['upper']<=p['upper']:raise SceneError('invalid_bounds',b['param_id'])
    # Optional cloth details cannot veto otherwise satisfied required appearance.
    required={k for k,v in ISSUES.items() if v[0]=='appearance' and v[2]}
    reports={r['issue_id']:r for r in parsed['issue_updates']}
    if required<=reports.keys() and all(reports[k]['status']=='resolved' for k in required):
        for row in parsed['category_results']:
            if row['category']=='appearance' and row['verdict'] in ('fail','needs_review') and reports.get('cloth_detail',{}).get('status')=='non_blocking_limitation':
                row['verdict']='pass_with_limitations';parsed['tradeoff_rationale']+='; machine policy '+POLICY+' excludes fine cloth details only'
        if final and all(r['verdict'] in ('pass','pass_with_limitations') for r in parsed['category_results']):parsed['verdict']='pass_with_limitations'
    return parsed


def bundle(s,metrics,rgb,masks,final=False,delivery_binding=None):
    folder=s.run/'terra'/review_key(s,final);folder.mkdir(parents=True,exist_ok=True)
    save_rgb(folder/'real.png',s.observation['rgb']);save_rgb(folder/'render.png',rgb)
    entries=[('REAL observation frame 0',folder/'real.png'),('CURRENT SELECTED SIM scene',folder/'render.png')]
    from ..render import overlay
    save_rgb(folder/'contours.png',overlay(s.observation['rgb'],masks,s.observation['target']));entries.append(('Real/selected-simulation contours',folder/'contours.png'))
    pending=s.state.get('pending_candidate');previous=None
    if pending:
        op=s.run/'operations'/pending['operation_id'];previous=read(op/'result.json')
        entries.extend([('SEARCH BEFORE selected scene',op/'before.png'),('SEARCH AFTER staged candidate, not yet retained',op/'after.png')])
    for label in ('apple','plate'):
        yy,xx=np.where(s.observation['target'][label]);box=(max(0,int(xx.min())-12),max(0,int(yy.min())-12),min(640,int(xx.max())+13),min(480,int(yy.max())+13)) if len(xx) else (0,0,640,480)
        x0,y0,x1,y1=box;parts=[s.observation['rgb'][y0:y1,x0:x1],rgb[y0:y1,x0:x1]]
        if pending:
            other=cv2.cvtColor(cv2.imread(str(s.run/'operations'/pending['operation_id']/'after.png')),cv2.COLOR_BGR2RGB);parts.append(other[y0:y1,x0:x1])
        path=folder/(label+'_comparison.png');save_rgb(path,np.concatenate(parts,1));entries.append((label+' fixed-window REAL / selected SIM'+(' / staged SIM' if pending else ''),path))
    diag=s.state.get('diagnostics',{})
    relevant=('geometry','appearance','initial_support','rest')
    if final:
        views=s.run/'final_acceptance/views'
        for name in ('support_side','support_oblique','neutral_ego','specular_off','neutral_all_lights_control','depth_normals'):
            path=views/(name+'.png')
            if path.exists():entries.append(('SIM-only auxiliary '+name,path))
    elif s.state.get('active_category') in relevant:
        for d in reversed(list(diag.values())):
            visual=d.get('evidence',{}).get('visual')
            if visual:
                for name in visual['views']:
                    if name not in ('support_side','support_oblique','neutral_all_lights_control','depth_normals'):continue
                    p=s.run/visual['folder']/(name+'.png')
                    if p.exists():entries.append(('SIM diagnostic '+name+'; no real shape proof',p))
                break
    acceptance=read(s.run/'final_acceptance/acceptance.json') if final else None
    state=dict(candidate_id=s.scene.candidate_id(),stage='FINAL_READONLY' if final else s.state['active_category'].upper(),requested_categories=list(CATEGORIES) if final else [s.state['active_category']],
        issues=s.state.get('issues',{}),category_checks=s.state.get('category_checks',{}),parameters=s.scene.params,camera=acceptance['camera_verified_from_xml'] if final else s.scene.locked_camera,
        capabilities=s.capabilities(),metrics=metrics,geometry=s.scene.geometry_metrics(),thresholds=s.config['acceptance'],initial_support=s.config['initial_support'],rest_thresholds=s.config['rest_acceptance'],
        budget={k:v for k,v in s.budget.summary().items() if k!='events'},pending_operation=previous,
        executed_results=s.state.get('tool_feedback',[])[-6:],diagnostics=diag,search_history=s.state.get('search_history',[])[-5:],
        category_entitlements=s.state.get('category_attempts',{}),independent_disk_acceptance=acceptance,delivery_binding=delivery_binding,
        details_policy=POLICY,tradeoff_policy='Explicit predeclared joint objective with all per-object deltas; preserve hard gates',
        mutations_allowed=not final and s.budget.remaining('terra_requests')>max(2,protected_slots(s.state,s.scene.params,s.scene.locked_camera,s.observation['version'])+1))
    templates=s.config['terra']['templates'];names=['system','classification'] if not final else ['system','final']
    if not final:names+=['repair']
    if pending or s.state.get('tool_feedback'):names+=['comparison']
    texts={name:(CODE_ROOT/'prompts'/templates[name]).read_text() for name in names}
    content=[dict(type='input_text',text=json.dumps(state,ensure_ascii=False,allow_nan=False))]
    for title,path in entries:content.extend([dict(type='input_text',text=title),dict(type='input_image',image_url='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode(),detail='high')])
    schema=response_schema(s.scene,final);cfg=s.config['terra'];body=dict(model=cfg['model'],instructions='\n\n'.join(texts.values()),input=[dict(role='user',content=content)],text=dict(format=dict(type='json_schema',name='terra_closed_loop_v5',strict=True,schema=schema)),max_output_tokens=cfg['max_output_tokens'],store=False)
    write(folder/'request.json',dict(request_sha256=digest(body),template_sha256s={name:sha(CODE_ROOT/'prompts'/templates[name]) for name in names},state=state,model=cfg['model'],schema=schema,images=[dict(identity=t,path=str(p.relative_to(s.run)),sha256=sha(p)) for t,p in entries],created_utc=now()))
    write(folder/'schema.json',schema);return folder,body
