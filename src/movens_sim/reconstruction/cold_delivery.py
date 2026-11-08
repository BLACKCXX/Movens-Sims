"""Self-contained initial-XML receipts and reports; no historical runs are read."""
from __future__ import annotations
import csv
import html
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time
import cv2
import numpy as np
from .ablation import POLICY,validate_config
from .ablation_delivery import STYLE,table,csv_write,ledger
from .prepare import load_observation
from ..head_camera import labelled_comparison
from ..util import ROOT,CODE_ROOT,read,write,sha,digest,now,save_rgb

LOCAL_LIMITS={'terra_requests': 0, 'geometry_rounds': 100, 'render_total': 4500, 'render_initial': 100, 'render_repair': 900, 'render_appearance': 2400, 'projection_evaluations': 10000, 'optimization_seconds': 7200, 'render_diagnostic': 600, 'render_export': 500}
CONTRACT_GROUPS=('fit', 'acceptance', 'initial_support', 'rest_acceptance', 'evaluator', 'render', 'diagnostic', 'v5')
SHARED_CONTRACT_SHA256='66de6668fc72dc70a551ffd8ce121daee12a044607ec8895d29124b877946f98'


def prepare(c):
    validate_config(c);root=ROOT/c['run_root'];root.mkdir(parents=True,exist_ok=True)
    # Freeze the inherited contract in code, with no A/B run OR old experiment-config read.
    rows=[]
    for k,v in LOCAL_LIMITS.items():
        if c['budget'][k]!=v:raise ValueError('Original local limit changed: '+k)
        rows.append(dict(setting='budget.'+k,original=v,current=c['budget'][k],unchanged=True))
    contract={k:c[k] for k in CONTRACT_GROUPS};contract['camera_bounds']=c['camera']['conditional_revision']
    if digest(contract)!=SHARED_CONTRACT_SHA256:raise ValueError('Frozen inherited numerical/quality contract changed')
    for key in ('local_search_cap','max_local_search_cap','cloth_optional_render_cap','final_compute_reserve_seconds'):
        rows.append(dict(setting='v5.'+key,original=c['v5'][key],current=c['v5'][key],unchanged=True))
    manifest=read(ROOT/c['manifest'])
    if [e['episode'] for e in manifest['entries']]!=[0,1,2]:raise ValueError('Only0/1/2 initial XML entries')
    definition=dict(config=c,manifest_sha256=sha(ROOT/c['manifest']),policy=POLICY,initial_geometry=c['initial_geometry'],historical_runtime_inputs_required=False)
    path=root/'experiment_definition.json'
    if path.exists() and read(path)!=definition:raise ValueError('Frozen cold-start definition changed')
    write(path,definition);write(root/'budget_comparison.json',dict(all_local_limits_identical=True,rows=rows));csv_write(root/'budget_comparison.csv',rows)
    write(root/'fixed_rules.json',dict(policy=POLICY,initial_geometry=c['initial_geometry'],source_mode='initial_xml'))
    write(root/'input_contract.json',dict(required=['scene_dir/scene.xml','XML asset closure','real_video frame0','parquet observation.state frame0','FK estimate','h0','robot.json','prepared_manifest.json','batch_construction.json','initial_state choice'],entity_mapping={'apple':'apple_root','plate':'plate_root','table':'table_root','robot':'pelvis'},state='fresh_xml_reset by default for these sources; explicit joint_name_state must contain qpos/qvel/time and name/address/size tables',unsupported='Other scene families or unflattened includes fail explicitly; no fallback to old reconstruction'))
    return dict(prepared=True,model_requests=0,all_local_limits_identical=True)


def flatten(value,prefix=''):
    out={}
    if isinstance(value,dict):
        for k,v in value.items():out.update(flatten(v,prefix+'.'+k if prefix else k))
    elif isinstance(value,(float,int)) and not isinstance(value,bool):out[prefix]=value
    return out


def episode_report(run,c):
    run=Path(run);r=read(run/'result.json');a=read(run/'final_acceptance/acceptance.json');obs=load_observation(run)
    names=['reference.png','raw_source_xml.png','initial_reconstruction.png','final_ego.png'];images=[cv2.cvtColor(cv2.imread(str(run/n)),cv2.COLOR_BGR2RGB) for n in names]
    labels=['REAL FRAME0','RAW XML / AUTHORED CAMERA','NORMALIZED FEASIBLE INITIAL','FINAL RELOADED / NO TERRA']
    save_rgb(run/'comparison.png',labelled_comparison(images,labels))
    for label in ('apple','plate'):
        y,x=np.where(obs['target'][label])
        if len(x):x0,x1=max(0,int(x.min())-16),min(640,int(x.max())+17);y0,y1=max(0,int(y.min())-16),min(480,int(y.max())+17)
        else:x0,x1,y0,y1=0,640,0,480
        save_rgb(run/(label+'_comparison.png'),labelled_comparison([im[y0:y1,x0:x1] for im in images],labels))
    before=read(run/'metrics_mount_baseline.json');after=a['metrics'];bf,af=flatten(before),flatten(after)
    metrics=[dict(metric=k,initial=bf.get(k),final=v,delta=v-bf[k] if k in bf else None) for k,v in af.items()]
    write(run/'initial_final_metrics.json',metrics);csv_write(run/'initial_final_metrics.csv',metrics)
    keys=['episode','source_mode','initialization_passed','processing_complete','construction_complete','geometry_passed','coverage_passed','appearance_status','initial_support_passed','physical_passed','reload_passed','final_terra_verdict','render_total','optimization_wall_seconds','elapsed_seconds']
    links=['source_xml_identity.json','initial_xml_verification.json','initialization.json','feasible_initial_parameters.json','source_manifest.json','mask_review.json','camera_prior.json','parameters_final.json','camera_actual.json','initial_asset_dimensions.json','initial_final_metrics.csv','result.json','final_acceptance/acceptance.json','delivery/static/portable/scene.xml','delivery/dynamic/portable/scene.xml','api_usage.json','budget_ledger.json','mcp_tool_calls.jsonl','candidate_decisions.jsonl','candidates.jsonl','candidates_v5.jsonl','diagnostic_timeline.json','contact_timeline.json','optimization_trace.html']
    summary={k:r.get(k) for k in keys};text='<!doctype html><meta charset="utf-8">'+STYLE+'<h1>Initial XML · Episode '+str(r['episode'])+'</h1><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><img src="comparison.png"><img src="apple_comparison.png"><img src="plate_comparison.png"><p>Original input → h0/FK normalization → feasible search seed → actual numerical optimization. No historical warm start, runtime model or cached/fake review. Source camera and normalized Mount prior differ explicitly. Appearance has no new hard pass threshold; highlight/real plate-shape uncertainty remains. Fine cloth details optional; coverage/color/support required.</p><p>'+' · '.join('<a href="'+p+'">'+p+'</a>' for p in links if (run/p).exists() or p=='optimization_trace.html')+'</p><video controls src="settle_diagnostic.mp4"></video><img src="final_acceptance/photometric/plate_luminance_curve.png">'
    (run/'index.html').write_text(text);(run/'report.md').write_text('# Initial XML reconstruction '+str(r['episode'])+'\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    trace='<!doctype html><meta charset="utf-8">'+STYLE+'<h1>Real MCP numerical search / candidate decisions</h1><p>Trial AFTER is a proposed candidate and may be rolled back. No visual-model judgment.</p>'
    decisions=[json.loads(x) for x in (run/'candidate_decisions.jsonl').read_text().splitlines()] if (run/'candidate_decisions.jsonl').exists() else []
    for p in sorted((run/'operations').glob('*/result.json')):
        result=read(p);base='operations/'+p.parent.name+'/'
        trace+='<h2>'+html.escape(p.parent.name+' · '+result['parameter_group'])+'</h2>'+''.join('<img style="width:32%" src="'+x+'">' for x in ['reference.png',base+'before.png',base+'after.png'])+'<pre>'+html.escape(json.dumps({k:result.get(k) for k in ['execution_status','new_candidates','new_render_evaluations','cache_hits','before_objective','after_objective','stop_reason']},ensure_ascii=False,indent=2))+'</pre><pre>'+html.escape(json.dumps([d for d in decisions if d['operation_id']==result['operation_id']],ensure_ascii=False,indent=2))+'</pre><a href="'+base+'result.json">All metric deltas</a>'
    (run/'optimization_trace.html').write_text(trace)
    write(run/'delivery_checks.json',dict(passed=a['engineering_passed'],fresh_native_integration=a['fresh_native_integration'],model_requests=0,review_status='not_run_ablation'))


def build(c,package=False):
    prepare(c);root=ROOT/c['run_root'];rows=[];audits=[];details=[]
    for ep in (0,1,2):
        run=root/f'episode_{ep:06}';r=read(run/'result.json');a=read(run/'final_acceptance/acceptance.json');b=read(run/'delivery_binding.json');used,events=ledger(c,ep)
        changed=[name for name,h in b['artifacts'].items() if not (run/name).is_file() or sha(run/name)!=h]
        checks=dict(final_bound_files_unchanged=not changed,initial_xml_import=read(run/'initial_xml_verification.json')['passed'],no_history_input=read(run/'source_manifest.json')['historical_reconstruction_input'] is False,zero_model=used.get('terra_requests',0)==0 and not (run/'terra').exists() and not (run/'final_review.json').exists(),independent_native_integration=a['fresh_native_integration'] and a['validator_pid']!=a['parent_pid'] and a['physical']['steps']==4000 and abs(a['physical']['duration_s']-1)<1e-9,budget=all(used.get(k,0)<=v+1e-9 for k,v in c['budget'].items()),engineering=a['engineering_passed'])
        audits.append(dict(episode=ep,passed=all(checks.values()),checks=checks,changed_bound_files=changed));write(run/'execution_audit.json',audits[-1]);write(run/'accounting.json',dict(used=used,events=events,note='Original XML preparation, search, diagnostics, export and fresh independent acceptance share this durable task ledger; no baseline costs mixed in'))
        row={k:r.get(k) for k in ['episode','source_mode','initialization_passed','processing_complete','construction_complete','geometry_passed','coverage_passed','appearance_status','initial_support_passed','physical_passed','reload_passed','final_terra_verdict','render_total','projection_evaluations','optimization_wall_seconds','elapsed_seconds']}
        quality=read(run/'mask_review.json');row.update(mask_quality_ok=quality['mask_quality_ok'],apple_mask_quality_ok=quality['objects']['apple']['quality_ok'],plate_mask_quality_ok=quality['objects']['plate']['quality_ok'])
        row.update(actual_model_HTTP=0,stop_reasons=json.dumps(r['execution_stats']['stop_reasons'],ensure_ascii=False),final_acceptance='episode_'+f'{ep:06}'+'/final_acceptance/acceptance.json',execution_version=digest(r['run_identity']['code']))
        for k,v in used.items():
            if k in c['budget']:row['used_'+k]=v
        bf,af=flatten(read(run/'metrics_mount_baseline.json')),flatten(a['metrics'])
        for key in ('apple_color_v5','apple_highlight_v5','plate_shading_v5','cloth_coarse_v5'):
            path='v5.objectives.'+key;row[key+'_initial']=bf[path];row[key+'_final']=af[path];row[key+'_delta']=af[path]-bf[path]
        for k,v in af.items():details.append(dict(episode=ep,metric=k,initial=bf.get(k),final=v,delta=v-bf[k] if k in bf else None))
        candidates=[]
        for name in ('candidates.jsonl','candidates_v5.jsonl'):
            if (run/name).exists():candidates.extend(json.loads(x) for x in (run/name).read_text().splitlines())
        row['candidate_evaluations']=len(candidates);row['retained_changes']=r['execution_stats']['retained_changes'];rows.append(row)
        episode_report(run,c)
    csv_write(root/'cold_start_summary.csv',rows);write(root/'cold_start_summary.json',rows);csv_write(root/'cold_start_metrics.csv',details)
    report=dict(execution_complete=all(r['processing_complete'] for r in rows),audit_passed=all(a['passed'] for a in audits),model_requests=0,historical_runtime_inputs_required=False,episodes=audits,generated_utc=now());write(root/'delivery_summary.json',report);write(root/'execution_audit.json',dict(passed=report['audit_passed'],episodes=audits,actual_model_HTTP=0))
    body='<h1>Movens-Sim · Original XML / FK → No Terra</h1><p>Exactly episodes0/1/2, real frame0 targets, independent new portable/native1s acceptance. Original assets and numerical limits retained. No v4/v5 optimized scene, selected camera/mask/state, or model response is an initialization input. Engineering completion is separate from numerical quality and appearance qualification.</p>'+table(rows,['episode','initialization_passed','processing_complete','geometry_passed','coverage_passed','appearance_status','initial_support_passed','physical_passed','reload_passed','final_terra_verdict','render_total','optimization_wall_seconds'])
    for ep in (0,1,2):body+=f'<h2><a href="episode_{ep:06}/index.html">Episode {ep}</a></h2><img src="episode_{ep:06}/comparison.png">'
    links=['cold_start_summary.csv','cold_start_metrics.csv','budget_comparison.json','input_contract.json','fixed_rules.json','delivery_summary.json','execution_audit.json','run_commands.md','verification/pytest.log','packages/']
    body+='<p>'+' · '.join('<a href="'+x+'">'+x+'</a>' for x in links)+'</p>';(root/'index.html').write_text('<!doctype html><meta charset="utf-8">'+STYLE+body)
    if package:
        dest=root/'packages';dest.mkdir(exist_ok=True)
        for ep in (0,1,2):
            run=root/f'episode_{ep:06}'
            with tarfile.open(dest/(run.name+'.tar.gz'),'w:gz') as tar:tar.add(run,arcname=run.name,filter=lambda x:None if '/v5_cache/' in x.name or '/evaluation_cache/' in x.name else x)
        write(root/'package_manifest.json',dict(packages={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in dest.glob('*.tar.gz')},created_utc=now()))
    return report
