"""Category-scoped evidence and strict contracts, using the common billed transport."""
from __future__ import annotations
import base64
import json
import cv2
import jsonschema
import numpy as np
from .backend import SceneError
from .operations import TOOLS,allowed_parameters
from .review_state import CATEGORIES,dependencies
from ..render import overlay
from ..util import CODE_ROOT,digest,read,write,save_rgb,sha,now


def review_key(session,final=False):
    category='final_readonly' if final else session.state['active_category']
    evidence=session.state.get('evidence_revision',0)
    return f'{category}_{session.scene.candidate_id()}_{evidence}'


def response_schema(scene,final=False):
    def obj(properties):return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)
    text=dict(type='string');tools=[k for k,v in TOOLS.items() if scene.stage in v['stages'] and not (scene.stage in ('frozen','final_readonly') and v['mutates'])]
    registry=scene.registry();editable=[p['id'] for p in registry if scene.stage in p['stages']] or [p['id'] for p in registry]
    issue=obj(dict(object=dict(type='string',enum=['apple','plate','cloth','table','camera','scene']),issue_type=dict(type='string',enum=['position','size','orientation','shape_asset','support','camera_global_projection','observation_error','appearance','rest']),evidence=text,confidence=dict(type='number',minimum=0,maximum=1),expected_metric=text))
    update=obj(dict(param_id=dict(type='string',enum=editable),value=dict(type='number'),unit=dict(type='string',enum=sorted({p['unit'] for p in registry}))))
    action=obj(dict(tool=dict(type='string',enum=tools or ['inspect_environment']),parameter_ids=dict(type='array',items=dict(type='string',enum=editable)),updates=dict(type='array',items=update),evidence=text,confidence=dict(type='number',minimum=0,maximum=1)))
    category=obj(dict(category=dict(type='string',enum=list(CATEGORIES)),verdict=dict(type='string',enum=['pass','fail','needs_review']),evidence=text))
    return obj(dict(candidate_id=text,stage=dict(type='string',enum=['OBSERVATION','GEOMETRY','INITIAL_SUPPORT','APPEARANCE','REST','FINAL_READONLY']),verdict=dict(type='string',enum=['pass','repair','needs_review','fail']),category_results=dict(type='array',items=category,minItems=5 if final else 1,maxItems=5 if final else 1),issues=dict(type='array',items=issue),proposed_actions=dict(type='array',maxItems=0 if final else 4,items=action),deferred_appearance_notes=dict(type='array',items=text)))


def validate_review(parsed,session,final=False):
    jsonschema.validate(parsed,response_schema(session.scene,final))
    if parsed['candidate_id']!=session.scene.candidate_id():raise SceneError('stale_candidate','Review scene identity mismatch')
    expected='FINAL_READONLY' if final else session.state['active_category'].upper()
    if parsed['stage']!=expected:raise SceneError('wrong_stage',expected)
    categories=[x['category'] for x in parsed['category_results']]
    if sorted(categories)!=sorted(CATEGORIES if final else [session.state['active_category']]):raise SceneError('category_contract','Exact requested categories required')
    if parsed['verdict']=='pass' and any(x['verdict']!='pass' for x in parsed['category_results']):raise SceneError('contradictory_pass','Every requested category must pass')
    registry={p['id']:p for p in session.scene.registry()}
    for action in parsed['proposed_actions']:
        spec=TOOLS[action['tool']]
        if session.scene.stage not in spec['stages']:raise SceneError('wrong_stage',action['tool'])
        allowed=set(allowed_parameters(session,spec['parameters']));ids=action['parameter_ids'];keys=[x['param_id'] for x in action['updates']]
        if len(ids)!=len(set(ids)) or len(keys)!=len(set(keys)) or set(ids+keys)-allowed:raise SceneError('invalid_parameter_subset',action['tool'])
        for row in action['updates']:
            p=registry[row['param_id']]
            if row['unit']!=p['unit'] or not np.isfinite(row['value']) or not p['lower']<=row['value']<=p['upper']:raise SceneError('invalid_parameter',row['param_id'])
    return parsed


def bundle(session,metrics,rgb,masks,final=False,delivery_binding=None):
    run=session.run;folder=run/'terra'/review_key(session,final);folder.mkdir(parents=True,exist_ok=True)
    save_rgb(folder/'real.png',session.observation['rgb']);save_rgb(folder/'render.png',rgb);save_rgb(folder/'overlay.png',overlay(session.observation['rgb'],masks,session.observation['target']))
    images=[('REAL full frame',folder/'real.png'),('SIM full frame of current candidate',folder/'render.png'),('Target/SIM contours',folder/'overlay.png')]
    for label in ('apple','plate'):
        yy,xx=np.where(session.observation['target'][label]|masks[label]);x0,x1=(max(0,int(xx.min())-15),min(640,int(xx.max())+16)) if len(xx) else (0,640);y0,y1=(max(0,int(yy.min())-15),min(480,int(yy.max())+16)) if len(yy) else (0,480)
        crop=np.concatenate([session.observation['rgb'][y0:y1,x0:x1],rgb[y0:y1,x0:x1]],1);path=folder/(label+'_real_sim.png');save_rgb(path,crop);images.append((label+' REAL left / SIM right, same window and magnification',path))
    category=session.state.get('active_category','observation')
    ev=run/'evidence'/session.scene.candidate_id();diagnostic=None
    if final:
        view_root=run/'final_acceptance/views';rest_root=run/'final_acceptance/rest'
    else:view_root=ev/'views';rest_root=ev/'rest'
    if category=='observation' and (run/'reference_masks_review.png').exists():images.append(('Automatic real-image segmentation; projected robot exclusion remains uncertain',run/'reference_masks_review.png'))
    if category in ('geometry','initial_support','appearance','rest') or final:
        for name in ('support_side','support_oblique','neutral_ego'):
            path=view_root/(name+'.png')
            if path.exists():images.append(('SIM diagnostic '+name,path))
    if (rest_root/'physical_diagnostic.json').exists():
        diagnostic=read(rest_root/'physical_diagnostic.json')
        timeline=read(rest_root/'diagnostic_timeline.json');contacts=read(rest_root/'contact_timeline.json')
        diagnostic={**diagnostic,'timeline_initial':timeline[0],'timeline_terminal':timeline[-1],'last_200ms':timeline[-7:],'terminal_contacts':contacts[-1]['contacts'][:18]}
        path=rest_root/'rest_terminal.png'
        if path.exists():images.append(('SIM at fixed one-second terminal time; not the reconstructed initial pose',path))
    acceptance=read(run/'final_acceptance/acceptance.json') if final and (run/'final_acceptance/acceptance.json').exists() else None
    tool_evidence=[]
    calls_path=run/'mcp_tool_calls.jsonl'
    if calls_path.exists():
        for row in [json.loads(line) for line in calls_path.read_text().splitlines()][-24:]:
            if row['tool']=='inspect_scene':continue
            result=row.get('retest') or {};keys=('accepted','candidate_id','before_candidate_id','loss','before_loss','after_loss','best_candidate_id','best_loss','render_evaluations','stop_reason','coverage_before','coverage_after','geometry_failures','geometry_unchanged')
            tool_evidence.append(dict(tool=row['tool'],ok=row['ok'],error=row.get('error'),parameter_changes=row.get('parameter_changes'),result={k:result[k] for k in keys if k in result}))
    state=dict(candidate_id=session.scene.candidate_id(),scene_revision=session.state.get('scene_revision',0),stage='FINAL_READONLY' if final else category.upper(),requested_categories=list(CATEGORIES) if final else [category],parameters=session.scene.params,registry=session.scene.registry(),camera=session.scene.locked_camera,geometry=session.scene.geometry_metrics(),metrics=metrics,acceptance_thresholds=session.config['acceptance'],initial_support_thresholds=session.config['initial_support'],rest_thresholds=session.config['rest_acceptance'],capabilities=session.capabilities(),category_checks=session.state.get('category_checks',{}),budget={k:v for k,v in session.budget.summary().items() if k!='events'},physical_diagnostic=diagnostic,normal_evidence=read(view_root/'normal_evidence.json') if (view_root/'normal_evidence.json').exists() else None,independent_disk_acceptance=acceptance,delivery_binding=delivery_binding,previous_operations=session.state.get('operations',[])[-12:],executed_tool_evidence=tool_evidence,mask_review=read(run/'mask_review.json'))
    content=[dict(type='input_text',text=json.dumps(state,ensure_ascii=False,allow_nan=False))]
    for title,path in images:content.extend([dict(type='input_text',text=title),dict(type='input_image',image_url='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode(),detail='high')])
    prompt=CODE_ROOT/'prompts'/(session.config['terra']['prompt_version']+'.txt');schema=response_schema(session.scene,final);cfg=session.config['terra']
    body=dict(model=cfg['model'],instructions=prompt.read_text(),input=[dict(role='user',content=content)],text=dict(format=dict(type='json_schema',name='classified_review_v4',strict=True,schema=schema)),max_output_tokens=cfg['max_output_tokens'],store=False)
    write(folder/'request.json',dict(request_sha256=digest(body),model=cfg['model'],prompt_sha256=sha(prompt),schema=schema,state=state,images=[dict(identity=t,path=str(p.relative_to(run)),sha256=sha(p)) for t,p in images],created_utc=now()))
    write(folder/'schema.json',schema)
    return folder,body
