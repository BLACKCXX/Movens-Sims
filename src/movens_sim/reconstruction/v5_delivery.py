"""Three fixed rows, new execution receipts and directly comparable v4/v5 metrics."""
from pathlib import Path
import csv
import html
import io
import json
import tarfile
import subprocess
import numpy as np
from ..util import ROOT,CODE_ROOT,read,write,sha,digest,now
from .v5_metrics import POLICY


def comparison(run,c):
    before=read(run/'baseline_v4/metrics_v5_evaluator.json');after=read(run/'final_acceptance/acceptance.json')['metrics'];rows=[]
    for k,b in before['v5']['objectives'].items():
        a=after['v5']['objectives'][k];delta=a-b;conclusion='基本持平' if abs(delta)<=c['acceptance']['minimum_objective_improvement'] else ('改善' if delta<0 else '退化')
        if k=='apple_highlight_v5' and before['v5']['apple_highlight']['confidence']<.25:conclusion='证据不足'
        rows.append(dict(metric=k,v4=b,v5=a,delta=delta,lower_is_better=True,conclusion=conclusion))
    for k in ('cloth_iou','top_cloth_iou','false_background_on_real_cloth_fraction'):
        b=before['table'][k];a=after['table'][k];delta=a-b;lower='false_background' in k
        conclusion='基本持平' if abs(delta)<=1e-9 else ('改善' if (delta<0)==lower else '退化')
        rows.append(dict(metric=k,v4=b,v5=a,delta=delta,lower_is_better=lower,conclusion=conclusion))
    for label in ('apple','plate'):
        for key in ('iou','centroid_error_px','symmetric_edge_distance_px'):
            b=before['objects'][label][key];a=after['objects'][label][key];delta=a-b;lower=key!='iou'
            rows.append(dict(metric=label+'.'+key,v4=b,v5=a,delta=delta,lower_is_better=lower,conclusion='基本持平' if abs(delta)<=1e-9 else ('改善' if (delta<0)==lower else '退化')))
    old=ROOT/c['initialization']['warm_start_root']/run.name;old_physics=read(old/'reacceptance/final/acceptance.json')['physical'];new_physics=read(run/'final_acceptance/acceptance.json')['physical']
    for label in ('apple','plate'):
        for key in ('translation_final_m','rotation_final_deg','penetration_final_m'):
            b=old_physics['objects'][label][key];a=new_physics['objects'][label][key]
            rows.append(dict(metric=label+'.'+key,v4=b,v5=a,delta=a-b,lower_is_better=True,conclusion='基本持平' if abs(a-b)<1e-9 else ('改善' if a<b else '退化')))
    result=dict(rows=rows,reference_unchanged=True,baseline_kind='Frozen v4 image/masks evaluated with v5 evaluator; v4 physics from frozen independent acceptance, v5 physics newly integrated',evaluator=c['evaluator'],fixed_region_fingerprint=after['v5']['region_fingerprint'],baseline_region_fingerprint=before['v5']['region_fingerprint'],policy_change=dict(version=POLICY,optional=['fine cloth texture','fine cloth wrinkles'],required=['coverage','position','overall cloth color','true support'],interpretation='Policy eligibility is separate from measured visual change; no hard thresholds relaxed'),checked_utc=now())
    write(run/'v4_v5_comparison.json',result);return result


def table(rows,columns):
    return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in columns)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(row.get(k,'')))+'</td>' for k in columns)+'</tr>' for row in rows)+'</table>'


STYLE='<style>body{font:16px system-ui;margin:24px;background:#151b23;color:#eee}a{color:#8cd}img{max-width:100%}video{width:640px;max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #566;padding:7px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'


def episode_report(run,c):
    r=read(run/'result.json');comp=comparison(run,c);a=read(run/'final_acceptance/acceptance.json');review=read(run/'final_review.json')
    keys=('episode','processing_complete','construction_complete','geometry_passed','coverage_passed','appearance_passed','initial_support_passed','physical_passed','reload_passed','final_terra_verdict','terra_actual_http_requests','render_total')
    summary={k:r[k] for k in keys};links=['result.json','warm_start_verification.json','parameter_perturbations/report.json','v4_v5_comparison.json','final_acceptance/acceptance.json','final_acceptance_process.json','final_review.json','session.json','parameters_final.json','camera_actual.json','asset_audit.json','execution_stats.json','api_usage.json','budget_ledger.json','mcp_tool_calls.jsonl','candidate_decisions.jsonl','delivery/static/portable/scene.xml','delivery/dynamic/portable/scene.xml','diagnostic_timeline.json','contact_timeline.json','physical_diagnostic.json','operations/','terra/']
    body='<h1>Episode '+str(r['episode'])+' · v5 本轮真实重跑</h1><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><p>REAL / frozen V4 / NEW V5。最终物理证据由独立进程从落盘动态 XML 和保存状态重新积分1秒生成。无策略推理或训练。</p><img src="comparison.png"><img src="apple_comparison.png"><img src="plate_comparison.png"><img src="final_acceptance/photometric/plate_luminance_curve.png">'+table(comp['rows'],['metric','v4','v5','delta','conclusion'])+'<p>'+' · '.join('<a href="'+p+'">'+p+'</a>' for p in links)+'</p><video controls src="ego_static.mp4"></video><video controls src="settle_diagnostic.mp4"></video><img src="final_acceptance/views/neutral_all_lights_control.png"><img src="final_acceptance/views/depth_normals.png"><h2>最终 Terra 实际结论</h2><pre>'+html.escape(json.dumps(review.get('review'),ensure_ascii=False,indent=2))+'</pre>'
    (run/'index.html').write_text('<!doctype html><meta charset="utf-8">'+STYLE+body)
    (run/'report.md').write_text('# Episode '+str(r['episode'])+' v5 重建验收\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n[新图片、曲线、视频与全部证据](index.html)\n\n'+json.dumps(a['geometry'],ensure_ascii=False,indent=2)+'\n')
    write(run/'delivery_checks.json',dict(passed=a['engineering_passed'],actual_new_optimization=r['execution_stats']['new_candidates']>0,fresh_native_integration=a['fresh_native_integration'],final_binding=sha(run/'delivery_binding.json'),quality_passed=not r['needs_review']))


def build(c,package=False):
    root=ROOT/c['run_root'];rows=[];effects=[];audit=[]
    for ep in (0,1,2):
        run=root/f'episode_{ep:06}'
        if not (run/'result.json').exists():rows.append(dict(episode=ep,processing_complete=False,status='incomplete',path=str(run)));continue
        r=read(run/'result.json');episode_report(run,c);binding=read(run/'delivery_binding.json');a=read(run/'final_acceptance/acceptance.json');review=read(run/'final_review.json')
        bad=[name for name,h in binding['artifacts'].items() if not (run/name).is_file() or sha(run/name)!=h]
        calls=[json.loads(x) for x in (run/'mcp_tool_calls.jsonl').read_text().splitlines()];requests=[read(p) for p in (run/'terra').glob('*/attempt_*/status.json')]
        template_roles=set()
        for path in (run/'terra').glob('*/attempt_*/request.json'):template_roles.update(read(path).get('template_sha256s',{}))
        checks=dict(frozen_files_unchanged=not bad,review_binding=review['delivery_hash']==binding['scene_parameter_asset_hash'],new_independent_integration=a.get('fresh_native_integration',False),different_validator_process=a['validator_pid']!=a['parent_pid'],new_optimization=any((x.get('retest') or {}).get('new_candidates',0)>0 for x in calls),all_templates_actually_loaded=template_roles>={'system','classification','repair','comparison','final'},actual_http_equals_ledger=sum(x.get('actual_http_attempts',0) for x in requests)==r['terra_requests_reserved']<=20,fixed_regions_comparable=read(run/'v4_v5_comparison.json')['fixed_region_fingerprint']==read(run/'v4_v5_comparison.json')['baseline_region_fingerprint'])
        audit.append(dict(episode=ep,checks=checks,passed=all(checks.values()),bad_files=bad))
        row={k:r[k] for k in ('episode','processing_complete','construction_complete','geometry_passed','coverage_passed','appearance_passed','initial_support_passed','physical_passed','reload_passed','final_terra_verdict','terra_actual_http_requests','render_total')};row['path']=str(run);row['execution_version']=digest(r['run_identity']['code']);rows.append(row)
        for x in read(run/'v4_v5_comparison.json')['rows']:effects.append(dict(episode=ep,execution_version=row['execution_version'],**x))
    for name,data in [('results.csv',rows),('rerun_effects.csv',effects)]:
        if data:
            fields=list(dict.fromkeys(k for row in data for k in row));out=io.StringIO();w=csv.DictWriter(out,fields);w.writeheader();w.writerows(data);(root/name).write_text(out.getvalue())
    complete=len(audit)==3 and all(x['passed'] for x in audit) and all(x['processing_complete'] for x in rows)
    result=dict(workflow=c['workflow'],episodes=rows,execution_complete=complete,quality_all_passed=all(x.get('final_terra_verdict') in ('pass','pass_with_limitations') for x in rows),audit=audit,full402_stop_present=(ROOT/'runs/full402_h0_v1/STOP').exists(),policy_inference=False,training=False,generated_utc=now())
    write(root/'delivery_summary.json',result);write(root/'delivery_audit.json',dict(passed=complete,episodes=audit));write(root/'rerun_effects.json',effects)
    body='<h1>Movens-Sim v5 · 0 / 1 / 2 本轮真实重跑与落盘验收</h1><p>执行完成与质量通过分别报告。桌布细节可选政策不豁免覆盖、整体颜色或真实支撑。402任务继续停止；没有策略推理或训练。</p>'+table(rows,list(dict.fromkeys(k for row in rows for k in row)))
    for ep in (0,1,2):
        p=f'episode_{ep:06}';body+='<h2><a href="'+p+'/index.html">Episode '+str(ep)+'</a></h2><img src="'+p+'/comparison.png">'
    body+='<p><a href="rerun_effects.csv">三条固定效果表 CSV</a> · <a href="rerun_effects.json">全部指标差值</a> · <a href="delivery_audit.json">真实执行与完整性审计</a> · <a href="verification/pytest.log">回归测试</a> · <a href="packages/">代码和三条产物包</a></p>'
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8">'+STYLE+body)
    if package:
        dest=root/'packages';dest.mkdir(exist_ok=True);changed=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=ROOT,text=True).splitlines();untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines();files=sorted(set(changed+untracked));files=[p for p in files if not p.startswith(('runs/','data/','.venv/')) and (ROOT/p).is_file()]
        patch=root/'implementation.patch';patch.write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT))
        with tarfile.open(dest/'modified_code.tar.gz','w:gz') as tar:
            for name in files:tar.add(ROOT/name,arcname=name)
            tar.add(patch,arcname='implementation.patch')
        for ep in (0,1,2):
            run=root/f'episode_{ep:06}'
            with tarfile.open(dest/(run.name+'.tar.gz'),'w:gz') as tar:
                tar.add(run,arcname=run.name,filter=lambda info:None if '/code_versions/' in info.name or '/v5_cache/' in info.name else info)
        write(root/'code_delivery.json',dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files={p:sha(ROOT/p) for p in files},packages={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in dest.glob('*.tar.gz')},working_tree_version=digest({p:sha(ROOT/p) for p in files})))
    return result
