"""A/B reporting only. This module is never consulted to choose a B candidate."""
from pathlib import Path
import collections
import csv
import html
import io
import json
import sqlite3
import subprocess
import tarfile
import time
from datetime import datetime

import cv2
import numpy as np
from ..util import ROOT,CODE_ROOT,read,write,sha,digest,now,save_rgb,video_info
from ..head_camera import labelled_comparison
from .ablation import POLICY,validate_config
from .v5_metrics import appearance_metrics

STYLE='<style>body{font:16px system-ui;margin:24px;background:#151b23;color:#eee}a{color:#8cd}img{max-width:100%}video{width:640px;max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #566;padding:7px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'


def table(rows,keys):
    return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table>'


def csv_write(path,rows):
    out=io.StringIO();keys=list(dict.fromkeys(k for row in rows for k in row));w=csv.DictWriter(out,keys);w.writeheader();w.writerows(rows);Path(path).write_text(out.getvalue())


def prepare(c):
    validate_config(c);start=time.monotonic();root=ROOT/c['run_root'];root.mkdir(parents=True,exist_ok=True)
    a=ROOT/c['ablation']['reference_root'];base=read(a/'resolved_config.json');limits=[]
    for key,v in base['budget'].items():
        b=c['budget'][key];same=b==v
        if key!='terra_requests' and not same:raise ValueError('A/B local budget differs: '+key)
        if key=='terra_requests' and b!=0:raise ValueError('Nonzero model budget')
        limits.append(dict(setting='budget.'+key,A=v,B=b,local_limit_unchanged=same if key!='terra_requests' else 'model_disabled'))
    for group in ('fit','camera','acceptance','initial_support','rest_acceptance','evaluator','render','diagnostic','initialization','appearance','v5'):
        if c[group]!=base[group]:raise ValueError('A/B shared configuration differs: '+group)
    for k in ('local_search_cap','max_local_search_cap','cloth_optional_render_cap','final_compute_reserve_seconds'):
        limits.append(dict(setting='v5.'+k,A=base['v5'][k],B=c['v5'][k],local_limit_unchanged=True))
    frozen=root/'experiment_definition.json'
    definition=dict(config=c,policy_sha256=digest(POLICY),baseline_config_sha256=sha(a/'resolved_config.json'))
    if frozen.exists() and read(frozen)!=definition:raise ValueError('Experiment definition changed; no reset permitted')
    write(frozen,definition);write(root/'budget_comparison.json',dict(all_local_limits_identical=True,rows=limits));csv_write(root/'budget_comparison.csv',limits)
    # Preserve A, v4 and full402 evidence; read-only hash collection never runs A.
    protected=root/'history_protection.json'
    if not protected.exists():
        files={str(p):sha(p) for p in a.rglob('*') if p.is_file()}
        files.update(read(a/'history_protection.json')['files'])
        for p in (ROOT/'runs/_task_budgets').glob('*.sqlite'):
            if p.name.startswith(('ep012_reconstruction_v','full402_h0')):files[str(p)]=sha(p)
        for p in (ROOT/'runs/full402_h0_v1/STOP',ROOT/'runs/full402_h0_v1/summary.json'):files[str(p)]=sha(p)
        write(protected,dict(files=files,created_utc=now()))
    changed=[name for name,h in read(protected)['files'].items() if not Path(name).is_file() or sha(name)!=h]
    if changed:raise ValueError('Protected baseline changed: '+str(changed[:5]))
    original=subprocess.check_output(['git','show','HEAD:src/movens_sim/reconstruction/v5_tools.py'],cwd=ROOT,text=True)
    current=(CODE_ROOT/'src/movens_sim/reconstruction/v5_tools.py').read_text()
    kernel=lambda x:x[x.index('    ids=GROUPS[parameter_group]'):x.index('    operation_id=f\'op_')]
    if kernel(original)!=kernel(current):raise ValueError('A numerical search kernel was modified')
    receipt=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         search_kernel_unchanged=True,search_kernel_sha256=digest(kernel(current)),
         powell=dict(maxiter=8,xtol=.003,ftol=1e-4,maxfev='remaining local cap after the SAME grouped seeds'),
         projection_refit=dict(max_nfev=45,diff_step=.002,ftol=1e-7,xtol=1e-7,gtol=1e-7),
         unchanged_modules={name:sha(CODE_ROOT/'src/movens_sim'/name) for name in ('render.py','reconstruction/v5_material.py','reconstruction/v5_metrics.py','reconstruction/optimize.py','reconstruction/diagnostics.py','reconstruction/acceptance.py')},
         legacy_fields_not_driving_v5=['appearance.subsets','appearance.render_subbudgets','fit.projection_starts','fit.coarse_render_candidates','fit.local_start_count','fit.initial_render_subbudgets','fit.repair_round_caps','fit.local_no_improvement_evaluations','classification.maximum_passes','geometry_rounds (legacy fitter counter; fixed v5 controller has at most four geometry searches)'],
         seeds='Existing deterministic grouped seeds; fit.seed is preserved, no new random starts',
         A_source_versions={f'episode_{ep:06}':digest(read(a/f'episode_{ep:06}/run_identity.json')['code']) for ep in (0,1,2)},
         A_episode0_known_difference='session warm-cache correction and execution_fingerprint session.py inclusion; seed/Powell/objectives unchanged',
         preparation_wall_seconds=time.monotonic()-start,created_utc=now())
    write(root/'implementation_comparison.json',receipt);write(root/'fixed_rules.json',POLICY)
    return receipt


def episode_report(run,c):
    r=read(run/'result.json');a=read(run/'final_acceptance/acceptance.json')
    real=cv2.cvtColor(cv2.imread(str(run/'reference.png')),cv2.COLOR_BGR2RGB);initial=cv2.cvtColor(cv2.imread(str(run/'initial_reconstruction.png')),cv2.COLOR_BGR2RGB);final=cv2.cvtColor(cv2.imread(str(run/'final_ego.png')),cv2.COLOR_BGR2RGB)
    save_rgb(run/'comparison.png',labelled_comparison([real,initial,final],['REAL','COMMON INITIAL','B NO TERRA']))
    keys=('episode','processing_complete','construction_complete','geometry_passed','coverage_passed','appearance_status','initial_support_passed','physical_passed','reload_passed','final_terra_verdict','render_total','optimization_wall_seconds','elapsed_seconds')
    summary={k:r[k] for k in keys};links=['result.json','common_initial_verification.json','warm_start_verification.json','parameters_final.json','camera_actual.json','final_acceptance/acceptance.json','model_review_status.json','api_usage.json','budget_ledger.json','mcp_tool_calls.jsonl','candidate_decisions.jsonl','delivery/static/portable/scene.xml','delivery/dynamic/portable/scene.xml','physical_diagnostic.json','diagnostic_timeline.json','contact_timeline.json','session.json']
    (run/'index.html').write_text('<!doctype html><meta charset="utf-8">'+STYLE+'<h1>B Episode '+str(r['episode'])+'</h1><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><img src="comparison.png"><p>零运行时模型请求；外观只测量，不自动判合格。真实盘形和弱高光证据保留不确定性；桌布细节可选不豁免覆盖、整体颜色或支撑。</p><p>'+' · '.join('<a href="'+p+'">'+p+'</a>' for p in links)+'</p><video controls src="settle_diagnostic.mp4"></video><img src="final_acceptance/photometric/plate_luminance_curve.png">')
    (run/'report.md').write_text('# No-Terra ablation '+str(r['episode'])+'\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    write(run/'delivery_checks.json',dict(passed=a['engineering_passed'],fresh_native_integration=a['fresh_native_integration'],model_requests=0,review_status='not_run_ablation'))


def ledger(c,ep):
    p=ROOT/'runs/_task_budgets'/(c['task_group_id']+'.sqlite');task=c['task_group_id']+f'/episode_{ep:06}'
    with sqlite3.connect(f'file:{p}?mode=ro',uri=True) as db:
        events=[dict(zip(('kind','amount','detail','created','elapsed','status'),r)) for r in db.execute('SELECT kind,amount,detail,created,elapsed,status FROM events WHERE task=? ORDER BY created',(task,))]
    for e in events:e['detail']=json.loads(e['detail'])
    used=collections.defaultdict(float)
    for e in events:used[e['kind']]+=e['amount']
    return dict(used),events


def counters(run):
    candidates=[json.loads(x) for x in (run/'candidates_v5.jsonl').read_text().splitlines()] if (run/'candidates_v5.jsonl').exists() else []
    decisions=[json.loads(x) for x in (run/'candidate_decisions.jsonl').read_text().splitlines()] if (run/'candidate_decisions.jsonl').exists() else []
    calls=[json.loads(x) for x in (run/'mcp_tool_calls.jsonl').read_text().splitlines()] if (run/'mcp_tool_calls.jsonl').exists() else []
    return dict(candidate_evaluations=len(candidates),unique_candidates=len({x['candidate_id'] for x in candidates}),
                new_rendered_candidate_identities=len({x['candidate_id'] for x in candidates if not x['cache_hit']}),
                search_cache_hits=sum(x['cache_hit'] for x in candidates),retained_changes=sum(x['candidate_retained'] for x in decisions),
                rolled_back=sum(x['decision']=='rollback' for x in decisions),mcp_calls=len(calls),
                diagnostic_calls=sum(x['tool']=='diagnose_issue' for x in calls),
                optimization_calls=sum(x['tool'] in ('fit_appearance','optimize_issue') for x in calls),
                no_improvement_calls=sum((x.get('retest') or {}).get('execution_status')=='no_improvement' for x in calls),
                tool_errors=sum(not x['ok'] for x in calls))


def elapsed(run):
    result=read(run/'result.json');commands=[json.loads(x) for x in (run/'commands.jsonl').read_text().splitlines()]
    start=min(x['utc'] for x in commands);end=result['completed_utc']
    receipts=[read(p) for p in (run.parent/'control/receipts').glob(run.name+'_attempt_*.json')]
    return dict(first_command_utc=start,result_completed_utc=end,
                end_to_end_seconds=(datetime.fromisoformat(end)-datetime.fromisoformat(start)).total_seconds(),
                worker_process_wall_seconds=sum(x.get('wall_seconds',0) for x in receipts),
                interpretation='End-to-end includes engineering pauses/restarts; A was developed during its run, so this is not an isolated causal model-latency estimate')


def comparable_metrics(a):
    m=a['metrics'];v=m['v5'];out={}
    for k,x in v['objectives'].items():out[k]=(x,True)
    for group,keys in dict(apple_color=('red_lab_mae','yellow_lab_mae','area_fraction_error','spatial_distribution_mae'),apple_highlight=('position_error_normalized','soft_area_error','soft_distribution_mae'),plate_shading=('radial_luminance_mae','regional_luminance_mae','sector_luminance_mae','gradient_mae')).items():
        for key in keys:out[group+'.'+key]=(v[group][key],True)
    for k in ('cloth_iou','top_cloth_iou','false_background_on_real_cloth_fraction'):out[k]=(m['table'][k],k=='false_background_on_real_cloth_fraction')
    for label in ('apple','plate','cloth'):out[label+'.fixed_lab_mae']=(v['overall_fixed_lab'][label],True)
    out['cloth.contrast_error']=(abs(v['cloth']['contrast_sim']-v['cloth']['contrast_real']),True)
    for k,x in v['plate_shading']['regions'].items():out['plate.'+k+'_luminance_error']=(abs(x['sim']-x['real']),True)
    # Same pre-existing A report definition: central/wall bins <=.8, not a new objective.
    profile=[x for x in v['plate_shading']['radial_profile'] if x['pixels']>0]
    for name,inner in (('full_plate',False),('central_wall_r_le_0_8',True)):
        pairs=[(x,y) for x,y in zip(profile,profile[1:]) if not inner or y['radius']<=.8]
        values={channel:max((abs(y[channel]-x[channel]) for x,y in pairs),default=0.) for channel in ('real','sim')}
        out['plate.'+name+'_gradient_error']=(abs(values['sim']-values['real']),True)
    for label in ('apple','plate'):
        for key in ('iou','centroid_error_px','symmetric_edge_distance_px','area_relative_error'):
            out[label+'.'+key]=(m['objects'][label][key],key!='iou')
        for key in ('translation_final_m','rotation_final_deg','penetration_final_m','translation_max_m','rotation_max_deg','penetration_max_m','last_200ms_linear_speed_max_m_s','last_200ms_angular_speed_max_rad_s'):
            out[label+'.'+key]=(a['physical']['objects'][label][key],True)
        out[label+'.initial_support_signed_gap_m']=(a['geometry'][label]['signed_support_gap_m'],None)
    return out


def plots(run,aa,bb):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
    for label,metrics,channel in [('REAL',aa,'real'),('A TERRA',aa,'sim'),('B NO TERRA',bb,'sim')]:
        p=metrics['v5']['plate_shading']['radial_profile'];x=[v['radius'] for v in p];y=[v[channel] for v in p]
        axes[0].plot(x,y,label=label);axes[1].plot([(a+b)/2 for a,b in zip(x,x[1:])],np.abs(np.diff(y)),label=label)
    for ax in axes:ax.legend();ax.axvline(.4,color='gray',ls=':');ax.axvline(.8,color='gray',ls=':');ax.grid(alpha=.2)
    axes[0].set(ylabel='OpenCV L (0..255)',title='Fixed REAL ellipse: central / wall / rim')
    axes[1].set(xlabel='Reference normalized radius',ylabel='Adjacent |delta L|',title='Outer-rim gradient is separate from inner-profile evidence')
    fig.tight_layout();fig.savefig(run/'plate_A_B_profile.png',dpi=140);plt.close(fig)


def build(c,package=False):
    root=ROOT/c['run_root'];aroot=ROOT/c['ablation']['reference_root'];ac=read(aroot/'resolved_config.json');rows=[];details=[];audits=[];supp_start=time.monotonic()
    if not all((root/f'episode_{ep:06}/result.json').exists() for ep in (0,1,2)):raise ValueError('All three new B results are required before A/B comparison')
    for ep in (0,1,2):
        run=root/f'episode_{ep:06}';old=aroot/run.name;r=read(run/'result.json');aa=read(old/'final_acceptance/acceptance.json');bb=read(run/'final_acceptance/acceptance.json')
        # Read-only A frozen-image remeasurement, outside B optimization accounting.
        from .prepare import load_observation
        from .metrics import measure
        obs=load_observation(run);image=cv2.cvtColor(cv2.imread(str(old/'final_ego.png')),cv2.COLOR_BGR2RGB)
        with np.load(old/'final_masks.npz') as z:masks={k:z[k] for k in z.files}
        measured=measure(image,masks,obs,ac);supp=root/'comparison/A_v5'/run.name;supp.mkdir(parents=True,exist_ok=True)
        write(supp/'metrics.json',measured);write(supp/'provenance.json',dict(kind='read_only_frozen_image_supplement',reference_sha256=sha(run/'reference.png'),a_image_sha256=sha(old/'final_ego.png'),optimization=False,native_renders=0,model_requests=0,scoring_unchanged=digest(measured)==digest(aa['metrics'])))
        comparable=measured['v5']['region_fingerprint']==bb['metrics']['v5']['region_fingerprint'] and digest(measured)==digest(aa['metrics'])
        left=comparable_metrics(aa);right=comparable_metrics(bb);local=[]
        for metric,(a,lower) in left.items():
            b=right[metric][0];delta=b-a;tolerance=c['acceptance']['minimum_objective_improvement'] if metric in aa['metrics']['v5']['objectives'] else 1e-9
            conclusion='基本持平' if abs(delta)<=tolerance else ('改善' if ((delta<0)==lower) else '退化')
            if lower is None:conclusion='报告签名间隙；按既定区间验收'
            if not comparable or ('highlight' in metric and measured['v5']['apple_highlight']['confidence']<.25):conclusion='证据不足'
            item=dict(episode=ep,metric=metric,A=a,B=b,delta=delta,lower_is_better=lower,conclusion=conclusion);local.append(item);details.append(item)
        ause,ae=ledger(ac,ep);buse,be=ledger(c,ep);acount,bcount=counters(old),counters(run);atime,btime=elapsed(old),elapsed(run);binding=read(run/'delivery_binding.json');bad=[n for n,h in binding['artifacts'].items() if sha(run/n)!=h]
        caps_ok=all(buse.get(k,0)<=v+1e-8 for k,v in c['budget'].items())
        audit=dict(episode=ep,passed=not bad and caps_ok and comparable and buse.get('terra_requests',0)==0 and bb['fresh_native_integration'] and bb['validator_pid']!=bb['parent_pid'] and read(run/'common_initial_verification.json')['passed'],changed_bound_files=bad,comparable=comparable,actual_model_requests=buse.get('terra_requests',0),new_native_integration=bb['fresh_native_integration'],engineering=bb['engineering_passed'])
        audits.append(audit);write(run/'ablation_audit.json',audit);write(run/'A_B_metrics.json',local)
        row={k:r[k] for k in ('episode','processing_complete','construction_complete','geometry_passed','coverage_passed','appearance_status','initial_support_passed','physical_passed','reload_passed','final_terra_verdict')}
        row.update(comparable=comparable,A_actual_renders=ause.get('render_total',0),B_actual_renders=buse.get('render_total',0),render_delta=buse.get('render_total',0)-ause.get('render_total',0),A_active_seconds=ause.get('optimization_seconds',0),B_active_seconds=buse.get('optimization_seconds',0),active_seconds_delta=buse.get('optimization_seconds',0)-ause.get('optimization_seconds',0),A_projections=ause.get('projection_evaluations',0),B_projections=buse.get('projection_evaluations',0),B_elapsed_seconds=r['elapsed_seconds'],B_new_candidates=r['execution_stats']['new_candidates'],B_retained_changes=r['execution_stats']['retained_changes'],A_historical_terra_verdict=read(old/'result.json')['final_terra_verdict'],B_actual_HTTP=0,stop_reason=json.dumps(r['execution_stats']['stop_reasons'],ensure_ascii=False))
        for label,used,count,timing,source in [('A',ause,acount,atime,old),('B',buse,bcount,btime,run)]:
            for key,value in count.items():row[label+'_'+key]=value
            for key,value in timing.items():row[label+'_'+key]=value
            for pool in ('render_initial','render_repair','render_appearance','render_diagnostic','render_export'):
                row[label+'_'+pool]=used.get(pool,0)
            saved=read(source/'result.json')
            row[label+'_post_completion_extra_renders']=used.get('render_total',0)-saved['render_total']
            row[label+'_post_completion_extra_active_seconds']=used.get('optimization_seconds',0)-saved['optimization_wall_seconds']
        for x in local:
            for key in ('A','B','delta','conclusion'):row[x['metric']+'_'+key]=x[key]
        for group in ('apple_color','apple_highlight','plate_shading'):
            if 'confidence' in measured['v5'][group]:row[group+'_evidence_confidence']=measured['v5'][group]['confidence']
        row['plate_boundary_evidence_confidence']=measured['v5']['plate_shading']['boundary_confidence'];rows.append(row)
        images=[obs['rgb'],cv2.cvtColor(cv2.imread(str(run/'initial_reconstruction.png')),cv2.COLOR_BGR2RGB),image,cv2.cvtColor(cv2.imread(str(run/'final_ego.png')),cv2.COLOR_BGR2RGB)]
        save_rgb(run/'comparison.png',labelled_comparison(images,['REAL','COMMON INITIAL','A TERRA V5','B NO TERRA']))
        for label in ('apple','plate'):
            yy,xx=np.where(obs['target'][label]);x0,x1=max(0,int(xx.min())-12),min(640,int(xx.max())+13);y0,y1=max(0,int(yy.min())-12),min(480,int(yy.max())+13)
            save_rgb(run/(label+'_comparison.png'),labelled_comparison([im[y0:y1,x0:x1] for im in images],['REAL','COMMON','A TERRA','B NO TERRA']))
        plots(run,aa['metrics'],bb['metrics'])
        p=run/'index.html';text=p.read_text().split('<section id="ablation-comparison">')[0];p.write_text(text+'<section id="ablation-comparison"><h2>A / B</h2><img src="apple_comparison.png"><img src="plate_comparison.png"><img src="plate_A_B_profile.png">'+table(local,['metric','A','B','delta','conclusion'])+'</section>')
        write(run/'accounting.json',dict(A=ause,B=buse,A_events=ae,B_events=be,A_counters=acount,B_counters=bcount,A_timing=atime,B_timing=btime,note='Settled durable ledgers, all resumes included. Diagnostic/export pools and post-completion extra verification are shown separately. Search candidates include interrupted archived evaluations, not only successful tool returns. A supplemental comparison is not charged to B search.'))
    csv_write(root/'ablation_summary.csv',rows);csv_write(root/'ablation_metrics.csv',details);write(root/'ablation_summary.json',rows)
    changes=[n for n,h in read(root/'history_protection.json')['files'].items() if not Path(n).is_file() or sha(n)!=h]
    full402=read(ROOT/'runs/full402_h0_v1/summary.json');history=dict(protected_files=len(read(root/'history_protection.json')['files']),changed=changes,full402_stop=(ROOT/'runs/full402_h0_v1/STOP').exists(),full402_running=full402['running'])
    write(root/'history_integrity.json',history)
    report=dict(execution_complete=all(r['processing_complete'] for r in rows),audit_passed=all(a['passed'] for a in audits) and not changes and history['full402_stop'] and not history['full402_running'],episodes=audits,model_requests=0,visual_review_status='not_run_ablation',conclusion='Only three fixed cases; measured differences do not establish whole-402 performance',generated_utc=now())
    write(root/'delivery_summary.json',report);write(root/'comparison/supplement_usage.json',dict(kind='A frozen-image supplement and A/B report generation',wall_seconds=time.monotonic()-supp_start,optimization=False,native_renders=0,model_requests=0,charged_to_B_search=False))
    keys=['episode','processing_complete','geometry_passed','coverage_passed','appearance_status','initial_support_passed','physical_passed','reload_passed','final_terra_verdict','A_actual_renders','B_actual_renders','A_active_seconds','B_active_seconds']
    body='<h1>Movens-Sim A Terra v5 / B no Terra</h1><p>Same v4 warm start, existing tools, search kernels and local caps. B uses one frozen eight-group pass; no runtime model, cached review or fake verdict. Historical tools/warm start include model-assisted development. Appearance metrics are not an automatic quality pass.</p>'+table(rows,keys)
    for ep in (0,1,2):body+=f'<h2><a href="episode_{ep:06}/index.html">Episode {ep}</a></h2><img src="episode_{ep:06}/comparison.png">'
    body+='<p>'+ ' · '.join('<a href="'+p+'">'+p+'</a>' for p in ('ablation_summary.csv','ablation_metrics.csv','budget_comparison.json','fixed_rules.json','implementation_comparison.json','history_integrity.json','delivery_summary.json','verification/pytest.log','run_commands.md','packages/'))+'</p>'
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8">'+STYLE+body)
    if package:
        dest=root/'packages';dest.mkdir(exist_ok=True);files=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=ROOT,text=True).splitlines()+subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines();files=sorted({p for p in files if not p.startswith(('runs/','data/','.venv/')) and (ROOT/p).is_file()})
        with tarfile.open(dest/'modified_code.tar.gz','w:gz') as tar:
            for name in files:tar.add(ROOT/name,arcname=name)
        for ep in (0,1,2):
            run=root/f'episode_{ep:06}'
            with tarfile.open(dest/(run.name+'.tar.gz'),'w:gz') as tar:tar.add(run,arcname=run.name,filter=lambda info:None if '/v5_cache/' in info.name else info)
        write(root/'package_manifest.json',dict(packages={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in dest.glob('*.tar.gz')},source_files={p:sha(ROOT/p) for p in files}))
    return report
