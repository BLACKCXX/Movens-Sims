"""Official Responses API vision suggestions, strict contract and local trial/rollback."""
from __future__ import annotations
import base64
import copy
import json
import os
import shutil
import time
from pathlib import Path
import jsonschema
import httpx
import numpy as np
from dotenv import dotenv_values
from openai import OpenAI
from .util import ROOT,append,digest,now,read,sha,write

def credentials(config):
    values=dotenv_values(config['env_file']) if Path(config['env_file']).is_file() else {}
    # Legacy key name explicitly supplied in this server's api.env. Do not source shell code.
    return os.environ.get('OPENAI_API_KEY') or values.get('OPENAI_API_KEY') or values.get('api')

def response_schema(registry):
    def obj(props):return dict(type='object',properties=props,required=list(props),additionalProperties=False)
    string={'type':'string'}
    return obj(dict(decision=dict(type='string',enum=['propose','accept_candidate','needs_review']),observations=dict(type='array',items=obj(dict(evidence=string,uncertainty=string))),updates=dict(type='array',maxItems=3,items=obj(dict(param_id=dict(type='string',enum=[p['id'] for p in registry]),value={'type':'number'},unit=dict(type='string',enum=sorted({p['unit'] for p in registry})),rationale=string))),expected_effect=dict(type='array',items=obj(dict(metric=string,direction=dict(type='string',enum=['increase','decrease','unchanged']),rationale=string))),review_reason=string))

def validate_response(response,scene,stage,schema=None):
    jsonschema.validate(response,schema or response_schema(scene.registry()))
    updates={};reg={p['id']:p for p in scene.registry()}
    for row in response['updates']:
        k=row['param_id']
        if k in updates:raise ValueError('Duplicate parameter ID')
        if row['unit']!=reg[k]['unit']:raise ValueError('Unit conflict '+k)
        updates[k]=row['value']
    scene.validate(updates,stage=stage)
    if response['decision']!='propose' and updates:raise ValueError('Non-propose decision cannot edit parameters')
    return updates

def request_bundle(scene,run,config,stage,round_index,budget,previous=None):
    from .fit import save_final_images
    save_final_images(scene,run);folder=Path(run)/'gpt'/f'round_{round_index:02}';folder.mkdir(parents=True,exist_ok=True)
    prompt_path=ROOT/'prompts'/(config['prompt_version']+'.txt');prompt=prompt_path.read_text();registry=scene.registry();schema=response_schema(registry)
    image_specs=[('REAL INITIAL RGB',Path(run)/'reference.png'),('CURRENT SIMULATION RGB',Path(run)/'optimized_ego.png'),('PIXEL CONTOUR OVERLAY',Path(run)/'overlay.png')]
    status=dict(task_boundary='Initial static scene, no actions, nominal plate scale anchor',prompt_version=config['prompt_version'],stage=stage,parameter_registry=registry,current_params=scene.params,editable_parameter_ids=[p['id'] for p in registry if p['stage']==stage],metrics=read(Path(run)/'metrics_after.json'),occlusion=read(Path(run)/'reference_selection.json')['objects'],previous_round=previous,remaining_budget=budget,coordinate_note='World meters; world-axis rotation vector degrees relative to baseline; no image-axis/world-axis equivalence',images=[dict(identity=label,path=str(path.relative_to(ROOT)),sha256=sha(path)) for label,path in image_specs])
    content=[dict(type='input_text',text=json.dumps(status,ensure_ascii=False))]
    for label,path in image_specs:
        content.append(dict(type='input_text',text=label));content.append(dict(type='input_image',image_url='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode(),detail='high'))
    body=dict(model=config['gpt']['model'],instructions=prompt,input=[dict(role='user',content=content)],text=dict(format=dict(type='json_schema',name='static_scene_diagnostic',strict=True,schema=schema)),max_output_tokens=config['gpt']['max_output_tokens'],store=False)
    previous=folder/'request.json'
    if previous.exists() and read(previous).get('request_sha256')!=digest(body):
        archive=Path(run)/'gpt/archive';archive.mkdir(exist_ok=True)
        dest=archive/(folder.name+'_'+read(previous).get('request_sha256','unknown')[:12])
        if dest.exists():dest=archive/(dest.name+'_'+str(time.time_ns()))
        shutil.move(str(folder),str(dest));folder.mkdir()
    for (_,path),filename in zip(image_specs,['reference.png','render.png','overlay.png']):shutil.copy2(path,folder/filename)
    log=dict(created_utc=now(),model=body['model'],api='Responses',endpoint=config['gpt']['base_url'],prompt_sha256=sha(prompt_path),prompt=prompt,schema=schema,state=status,request_sha256=digest(body),image_sha256s=[x['sha256'] for x in status['images']])
    write(folder/'request.json',log);write(folder/'schema.json',schema)
    return folder,body,schema

def call_or_prepare(scene,run,config,stage,round_index,budget,enabled=False,previous=None):
    from .reconstruction.modes import assert_model_allowed
    assert_model_allowed(config)
    folder,body,schema=request_bundle(scene,run,config,stage,round_index,budget,previous)
    key=credentials(config['gpt']);result=dict(round_index=round_index,stage=stage,model=config['gpt']['model'],enabled=enabled,credential_present=bool(key),request_sha256=digest(body),status='prepared_offline',api_call_performed=False)
    if enabled and key and (folder/'call_status.json').exists() and (folder/'parsed.json').exists():
        old=read(folder/'call_status.json')
        if old.get('request_sha256')==digest(body) and old.get('status')=='api_validated':
            parsed=read(folder/'parsed.json');validate_response(parsed,scene,stage,schema);old['api_call_performed']=False;old['cache_reused']=True;return old,parsed
    if not enabled or not key:
        # Explicitly synthetic offline contract fixture, never reported as a model answer.
        fixture=dict(decision='needs_review',observations=[dict(evidence='Offline contract validation only',uncertainty='No API response obtained')],updates=[],expected_effect=[],review_reason='Offline fixture; no model judgment')
        validate_response(fixture,scene,stage,schema);write(folder/'offline_fixture.json',fixture);result['offline_contract_validated']=True;write(folder/'call_status.json',result);return result,None
    budget_path=Path(run)/'gpt/gpt_budget.json'
    if budget_path.exists():budget=read(budget_path)
    else:
        historical=sum(1 for p in (Path(run)/'gpt').rglob('call_status.json') if read(p).get('enabled') and read(p).get('status','').startswith('api_'))
        budget=dict(http_attempts_used=historical,max_http_attempts=config['gpt']['max_calls'],historical_attempts_note='Legacy successful logical requests counted as at least one attempt each; new attempts logged individually')
    if budget['http_attempts_used']>=config['gpt']['max_calls']:
        result['status']='api_budget_exhausted';write(folder/'call_status.json',result);return result,None
    attempts=[]
    def on_request(request):
        if budget['http_attempts_used']>=config['gpt']['max_calls']:raise ValueError('GPT HTTP attempt budget exhausted')
        budget['http_attempts_used']+=1;write(budget_path,budget);attempts.append(dict(method=request.method,path=request.url.path,utc=now()));result['api_call_performed']=True
    def on_response(response):
        if attempts:attempts[-1]['http_status']=response.status_code
    http_client=httpx.Client(event_hooks=dict(request=[on_request],response=[on_response]))
    client=OpenAI(api_key=key,base_url=config['gpt']['base_url'],timeout=config['gpt']['timeout_seconds'],max_retries=config['gpt']['max_retries'],http_client=http_client)
    start=time.monotonic()
    try:
        response=client.responses.create(**body)
        raw=response.model_dump(mode='json');write(folder/'response.json',raw)
        result.update(response_id=response.id,response_model=response.model,usage=raw.get('usage'),response_status=response.status,wall_seconds=time.monotonic()-start,http_attempts=attempts)
        if response.status!='completed':raise ValueError('Incomplete API response')
        parsed=json.loads(response.output_text);updates=validate_response(parsed,scene,stage,schema);write(folder/'parsed.json',parsed);result.update(status='api_validated',updates=updates,decision=parsed['decision']);write(folder/'call_status.json',result);return result,parsed
    except Exception as e:
        # Do not print HTTP exception strings, headers, clients or credential values.
        result.update(status='api_or_contract_error',error_type=type(e).__name__,http_status=getattr(e,'status_code',None),wall_seconds=time.monotonic()-start,http_attempts=attempts)
        write(folder/'call_status.json',result);return result,None
    finally:client.close()

def run_suggestions(fitter,config,enabled):
    scene=fitter.scene;run=fitter.run;folder=Path(run)/'gpt';folder.mkdir(exist_ok=True);results=[];previous=None
    # First two episodes check the same v1 prompt, last three freeze it. No per-episode edits.
    rounds=min(config['gpt']['rounds'],config['gpt']['max_calls'])
    for i in range(rounds):
        stage=['layout','appearance'][i%2];budget=dict(render_evaluations=config['fit']['max_evaluations']-fitter.evaluations,gpt_calls=config['gpt']['max_calls']-i)
        status,parsed=call_or_prepare(scene,run,config,stage,i,budget,enabled,previous)
        if parsed and parsed['updates'] and fitter.evaluations+2>config['fit']['max_evaluations']:
            status.update(accepted=False,rollback=False,candidate_status='not_evaluated_render_budget_exhausted')
        elif parsed and parsed['updates']:
            before=scene.params.copy();fitter.stage=stage;fitter.stage_best=float('inf');fitter.evaluate(before,tag='pre_gpt',appearance=stage=='appearance');before_loss=fitter.best_loss
            try:
                updates=validate_response(parsed,scene,stage);loss=fitter.evaluate({**before,**updates},tag=f'gpt_round_{i:02}',appearance=stage=='appearance')
                status.update(trial_objective=loss,prior_objective=before_loss,accepted=scene.params!=before,rollback=scene.params==before)
            except Exception as e:scene.apply(before);status.update(accepted=False,rollback=True,local_error=type(e).__name__)
        else:status.update(accepted=False,rollback=False)
        previous=dict(status=status,observations=parsed['observations'] if parsed else None);results.append(status);write(folder/f'round_{i:02}'/'candidate_result.json',status)
    recorded=[read(p) for p in (Path(run)/'gpt').rglob('call_status.json')]
    request_ids={r.get('request_sha256') for r in recorded if r.get('enabled') and r.get('status') in ('api_validated','api_or_contract_error')}
    budget_path=Path(run)/'gpt/gpt_budget.json'
    write(Path(run)/'gpt_summary.json',dict(prompt_version=config['prompt_version'],configured_model=config['gpt']['model'],max_calls=config['gpt']['max_calls'],actual_calls=len(request_ids),http_attempts_used=read(budget_path)['http_attempts_used'] if budget_path.exists() else len(request_ids),results=results,api_actually_verified=any(r.get('status')=='api_validated' for r in recorded)))
    return results
