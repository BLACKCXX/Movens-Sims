"""Exact-model Responses vision review; one reserved slot per actual HTTP POST."""
from __future__ import annotations
import base64
import json
import time
import shutil
from pathlib import Path
import cv2
import httpx
import jsonschema
import numpy as np
from openai import OpenAI

from .backend import SceneError
from .budget import BudgetExceeded
from ..gpt_adapter import credentials
from ..render import overlay
from ..util import ROOT,CODE_ROOT,digest,now,read,save_rgb,sha,write


ISSUES=['position','size','orientation','shape_asset','support','camera_global_projection','observation_error','appearance']
ACTION_TOOLS=['fit_geometry','propose_or_apply_object_update','update_shape_or_asset','update_appearance','review_camera_hypothesis']


def response_schema(scene,final=False):
    if getattr(scene,'v5',False):
        from .v5_terra import response_schema as schema
        return schema(scene,final)
    if getattr(scene,'classified',False):
        from .classified_terra import response_schema as categorical_schema
        return categorical_schema(scene,final)
    def obj(properties):return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)
    text=dict(type='string');unit=dict(type='string',enum=sorted({p['unit'] for p in scene.registry()}))
    editable=[p['id'] for p in scene.registry() if scene.stage in p['stages']] or [p['id'] for p in scene.registry()]
    issue=obj(dict(object=dict(type='string',enum=['apple','plate','cloth','table','camera','scene']),issue_type=dict(type='string',enum=ISSUES),evidence=text,confidence=dict(type='number',minimum=0,maximum=1),expected_metric=text))
    update=obj(dict(param_id=dict(type='string',enum=editable),value=dict(type='number'),unit=unit))
    tools=[t for t in ACTION_TOOLS if t!='review_camera_hypothesis' or getattr(scene,'allow_camera_revision',True)]
    action=obj(dict(tool=dict(type='string',enum=tools),parameter_ids=dict(type='array',items=dict(type='string',enum=editable)),updates=dict(type='array',items=update),evidence=text,confidence=dict(type='number',minimum=0,maximum=1)))
    return obj(dict(candidate_id=text,stage=dict(type='string',enum=['GEOMETRY','FINAL_READONLY']),verdict=dict(type='string',enum=['pass','repair','needs_review','fail']),issues=dict(type='array',items=issue),proposed_actions=dict(type='array',maxItems=0 if final else 4,items=action),deferred_appearance_notes=dict(type='array',items=text)))


def validate_review(parsed,session,final=False):
    if session.scene.v5:
        from .v5_terra import validate_review as validate
        return validate(parsed,session,final)
    if session.scene.classified:
        from .classified_terra import validate_review as categorical_validate
        return categorical_validate(parsed,session,final)
    jsonschema.validate(parsed,response_schema(session.scene,final))
    if parsed['candidate_id']!=session.scene.candidate_id():raise SceneError('stale_candidate','Terra candidate mismatch')
    if parsed['stage']!=('FINAL_READONLY' if final else 'GEOMETRY'):raise SceneError('wrong_stage','Terra stage mismatch')
    registry={p['id']:p for p in session.scene.registry()}
    for action in parsed['proposed_actions']:
        values={}
        for row in action['updates']:
            key=row['param_id']
            if key in values:raise SceneError('duplicate_parameter',key)
            if registry[key]['unit']!=row['unit']:raise SceneError('unit_conflict',key)
            values[key]=row['value']
        # Shape permission can only be granted by the actual shape tool with evidence.
        shape=session.scene.shape_enabled;tilt=getattr(session.scene,'plate_tilt_enabled',False)
        try:
            if action['tool']=='update_shape_or_asset' and action['confidence']>=.7:
                session.scene.shape_enabled=True;session.scene.plate_tilt_enabled=True
            session.scene.validate(values,'appearance' if action['tool']=='update_appearance' else 'geometry')
        finally:session.scene.shape_enabled=shape;session.scene.plate_tilt_enabled=tilt
    if final and parsed['proposed_actions']:raise SceneError('final_is_readonly','No final modifications')
    return parsed


def bundle(session,metrics,rgb,masks,final=False,delivery_binding=None):
    if session.scene.v5:
        from .v5_terra import bundle as v5_bundle
        return v5_bundle(session,metrics,rgb,masks,final,delivery_binding)
    if session.scene.classified:
        from .classified_terra import bundle as categorical_bundle
        return categorical_bundle(session,metrics,rgb,masks,final,delivery_binding)
    run=session.run;candidate=session.scene.candidate_id();stage='FINAL_READONLY' if final else 'GEOMETRY';folder=run/'terra'/(stage.lower()+'_'+candidate);folder.mkdir(parents=True,exist_ok=True)
    save_rgb(folder/'real.png',session.observation['rgb']);save_rgb(folder/'render.png',rgb);save_rgb(folder/'overlay.png',overlay(session.observation['rgb'],masks,session.observation['target']))
    images=[('REAL FULL FRAME',folder/'real.png'),('SIM FULL FRAME',folder/'render.png'),('CONTOUR OVERLAY',folder/'overlay.png')]
    windows={}
    for label in ('apple','plate'):
        union=session.observation['target'][label]|masks[label];y,x=np.where(union);pad=20
        x0,x1=max(0,int(x.min())-pad),min(640,int(x.max())+pad+1);y0,y1=max(0,int(y.min())-pad),min(480,int(y.max())+pad+1)
        crop=np.concatenate([session.observation['rgb'][y0:y1,x0:x1],rgb[y0:y1,x0:x1]],axis=1);path=folder/(label+'_same_window.png');save_rgb(path,crop);images.append((f'{label.upper()} REAL LEFT / SIM RIGHT, identical window, scale 1:1',path));windows[label]=dict(xyxy=[x0,y0,x1,y1],scale=1.)
    prompt=CODE_ROOT/'prompts'/(session.config['terra']['prompt_version']+'.txt');schema=response_schema(session.scene,final)
    available=[t for t in ACTION_TOOLS if t!='review_camera_hypothesis' or getattr(session.scene,'allow_camera_revision',True)]
    status=dict(candidate_id=candidate,stage=stage,parameters=session.scene.params,dimensions_and_support=session.scene.geometry_metrics(),camera=session.scene.locked_camera,registry=session.scene.registry(),metrics=metrics,acceptance=session.config['acceptance'],budget={k:v for k,v in session.budget.summary().items() if k!='events'},tools=session.capabilities(),allowed_action_tools=[] if final else available,previous_operations=session.state.get('operations',[]),local_windows=windows,delivery_binding=delivery_binding,settle_result=read(run/'physical_diagnostic.json') if (run/'physical_diagnostic.json').exists() else None)
    content=[dict(type='input_text',text=json.dumps(status,ensure_ascii=False,allow_nan=False))]
    for identity,path in images:content.extend([dict(type='input_text',text=identity),dict(type='input_image',image_url='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode(),detail='high')])
    config=session.config['terra'];body=dict(model=config['model'],instructions=prompt.read_text(),input=[dict(role='user',content=content)],text=dict(format=dict(type='json_schema',name='mount_joint_review_v3',strict=True,schema=schema)),max_output_tokens=config['max_output_tokens'],store=False)
    write(folder/'request.json',dict(request_sha256=digest(body),model=config['model'],api='Responses',prompt_version=config['prompt_version'],prompt_sha256=sha(prompt),schema=schema,state=status,images=[dict(identity=i,path=str(p.relative_to(run)),sha256=sha(p)) for i,p in images],created_utc=now()))
    write(folder/'schema.json',schema)
    return folder,body


def request_review(session,metrics,rgb,masks,final=False,enabled=True,delivery_binding=None,client_factory=OpenAI):
    from .control import api_slot, record_api_status
    with api_slot(session.config):
        status,parsed=_request_review(session,metrics,rgb,masks,final,enabled,delivery_binding,client_factory)
        if enabled:record_api_status(session.config,status)
        return status,parsed


def _request_review(session,metrics,rgb,masks,final=False,enabled=True,delivery_binding=None,client_factory=OpenAI):
    existing=session.run/'terra'/(('final_readonly_' if final else 'geometry_')+session.scene.candidate_id())
    if session.scene.classified:
        if session.scene.v5:from .v5_terra import review_key
        else:from .classified_terra import review_key
        existing=session.run/'terra'/review_key(session,final)
    for statusfile in sorted(existing.glob('attempt_*/status.json')):
        status=read(statusfile)
        if status.get('status')=='validated' and (statusfile.parent/'parsed.json').exists():
            parsed=read(statusfile.parent/'parsed.json');validate_review(parsed,session,final)
            return {**status,'cache_reused':True},parsed
    folder,body=bundle(session,metrics,rgb,masks,final,delivery_binding);status_path=folder/'status.json'
    if status_path.exists() and (folder/'parsed.json').exists():
        previous=read(status_path)
        if previous.get('request_sha256')==digest(body) and previous.get('status')=='validated':return previous,read(folder/'parsed.json')
    key=credentials(session.config['terra']);status=dict(model=body['model'],request_sha256=digest(body),actual_http_request_started=False,api_response_received=False,stage='final_readonly' if final else 'geometry',folder=str(folder.relative_to(session.run)))
    if not enabled or not key:
        fixture=dict(candidate_id=session.scene.candidate_id(),stage='FINAL_READONLY' if final else 'GEOMETRY',verdict='needs_review',issues=[],proposed_actions=[],deferred_appearance_notes=['Explicit synthetic offline schema fixture; no model judgment'])
        if session.scene.classified:
            fixture['stage']='FINAL_READONLY' if final else session.state['active_category'].upper()
            fixture['category_results']=[dict(category=c,verdict='needs_review',evidence='Offline fixture; no model judgment') for c in (session.config['classification']['categories'] if final else [session.state['active_category']])]
        validate_review(fixture,session,final);write(folder/'offline_fixture.json',fixture);status.update(status='prepared_offline',offline_contract_validated=True,credential_available=bool(key));write(status_path,status);return status,None
    try:event=session.budget.reserve_terra(final,dict(candidate_id=session.scene.candidate_id(),category=session.state.get('active_category','geometry') if not final else 'final',request_sha256=digest(body),delivery_hash=delivery_binding.get('scene_parameter_asset_hash') if delivery_binding else None))
    except BudgetExceeded:
        status['status']='blocked_budget';write(status_path,status);return status,None
    attempt=folder/('attempt_'+event[:12]);attempt.mkdir()
    for name in ('request.json','schema.json'):shutil.copy2(folder/name,attempt/name)
    folder=attempt;status_path=folder/'status.json';status.update(reservation_id=event,folder=str(folder.relative_to(session.run)))
    write(status_path,status);started=time.monotonic();sent=0
    def on_request(request):
        nonlocal sent
        if sent or request.method!='POST' or not request.url.path.endswith('/responses'):raise BudgetExceeded('Unexpected extra HTTP request blocked')
        sent+=1;status['actual_http_request_started']=True;write(status_path,status)
    http=httpx.Client(event_hooks={'request':[on_request]},follow_redirects=False)
    client=None
    try:
        client=client_factory(api_key=key,base_url=session.config['terra']['base_url'],max_retries=0,timeout=session.config['terra']['timeout_seconds'],http_client=http)
        response=client.responses.create(**body);raw=response.model_dump(mode='json');write(folder/'response.json',raw);status.update(api_response_received=True,response_id=response.id,response_model=response.model,usage=raw.get('usage'),response_status=response.status)
        if response.status!='completed':raise SceneError('incomplete_response',response.status)
        if response.model!=body['model']:raise SceneError('model_mismatch','Exact configured model required')
        parsed=json.loads(response.output_text);validate_review(parsed,session,final);write(folder/'parsed.json',parsed);status.update(status='validated',verdict=parsed['verdict'])
        return_value=parsed
    except (KeyboardInterrupt,SystemExit):
        status.update(status='interrupted_unknown',error_type='InterruptedBeforeResponse')
        raise
    except Exception as exc:
        status.update(status='api_or_contract_error',error_type=type(exc).__name__,http_status=getattr(exc,'status_code',None));return_value=None
    finally:
        status['wall_seconds']=time.monotonic()-started;status['actual_http_attempts']=sent;session.budget.finish(event,status.get('status','interrupted_unknown'),status['wall_seconds']);write(status_path,status)
        if client is not None:client.close()
        else:http.close()
    return status,return_value
