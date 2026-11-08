#!/usr/bin/env python3
"""Read-only scene integrity/API accounting audit and convenient numeric tables."""
import argparse
from collections import Counter
import csv
import io
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from movens_sim.util import ROOT,read,write,sha,video_info,now

def csv_file(path,rows):
    buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);path.write_text(buf.getvalue())

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();c=read(ROOT/a.config);root=ROOT/c['run_root'];out=[]
    for ep in [0,1,2]:
        run=root/f'episode_{ep:06}';binding=read(run/'delivery_binding.json');review=read(run/'final_review.json');request=read(run/review['status']['folder']/'request.json');acceptance=read(run/'reacceptance/final/acceptance.json');ledger=read(run/'budget_ledger.json');events=ledger['events'];used=ledger['used']
        changed=[name for name,h in binding['artifacts'].items() if not (run/name).is_file() or sha(run/name)!=h]
        attempts=[read(f) for f in (run/'terra').glob('*/attempt_*/status.json')];actual=sum(r.get('actual_http_attempts',int(r.get('actual_http_request_started',False))) for r in attempts)
        frames={name:video_info(run/name) for name in ('ego_static.mp4','settle_diagnostic.mp4')}
        video_ok=all(v['width']==640 and v['height']==480 and v['fps']==30 and v['frames']==n for v,n in zip(frames.values(),(150,30)))
        review_bound=review['delivery_hash']==binding['scene_parameter_asset_hash']==request['state']['delivery_binding']['scene_parameter_asset_hash']
        image_changes=[r['path'] for r in request['images'] if sha(run/r['path'])!=r['sha256']]
        mcp=[e for e in events if e['kind']=='mcp_local_call'];native=[e for e in events if e['kind']=='rest_diagnostic'];passes=Counter()
        for e in events:
            if e['kind']=='render_passes':
                for k in ('rgb','segmentation'):passes[k]+=e['detail'].get(k,0)
            if e['kind']=='render_export' and e['status']=='completed' and e['detail'].get('kind')=='expected_final_scene':passes.update(rgb=1,segmentation=1)
        cost=dict(task_id=ledger['task_id'],limits=ledger['limits'],reserved_evaluations={k:v for k,v in used.items() if k.startswith('render_') or k in ('terra_requests','projection_evaluations')},completed_render_passes=dict(passes),active_compute_seconds=used.get('optimization_seconds',0),api_service_wall_seconds=sum(r.get('wall_seconds',0) for r in attempts),mcp_local_calls=len(mcp),mcp_by_tool=dict(Counter(e['detail']['tool'] for e in mcp)),mcp_tool_wall_seconds=sum(e['elapsed'] for e in mcp),native_rest_runs=len(native),native_integrated_steps=sum(e['detail'].get('steps',0) for e in native),note='MCP/native durations overlap active compute and must not be added to it; optimization render evaluations contain RGB and segmentation passes, reported separately here.')
        write(run/'cost_report.json',cost)
        rows=[]
        for t in read(run/'reacceptance/final/rest/diagnostic_timeline.json'):
            for label,obj in t['objects'].items():rows.append(dict(time_s=t['time_s'],object=label,translation_mm=obj['translation_m']*1000,rotation_deg=obj['rotation_deg'],penetration_mm=obj['penetration_m']*1000,linear_speed_m_s=obj['linear_speed_m_s'],angular_speed_rad_s=obj['angular_speed_rad_s'],contacts=obj['contacts'],soft_displacement_max_mm=t['soft_displacement_max_m']*1000))
        csv_file(run/'diagnostic_timeseries.csv',rows)
        metrics=[]
        for label,st in acceptance['physical']['objects'].items():metrics.append(dict(object=label,terminal_translation_mm=st['translation_final_m']*1000,terminal_rotation_deg=st['rotation_final_deg'],terminal_penetration_mm=st['penetration_final_m']*1000,peak_translation_mm=st['translation_max_m']*1000,peak_rotation_deg=st['rotation_max_deg'],peak_penetration_mm=st['penetration_max_m']*1000,terminal_passed=st['passed']))
        csv_file(run/'rest_terminal_and_peaks.csv',metrics)
        passed=bool(not changed and not image_changes and review_bound and video_ok and actual==used.get('terra_requests',0)<=20 and acceptance['engineering_passed'] and acceptance['artifact_scene_hash']==read(run/'final_acceptance/acceptance.json')['artifact_scene_hash'])
        row=dict(episode=ep,passed=passed,changed_artifacts=changed,changed_review_images=image_changes,review_bound=review_bound,actual_requests=actual,reserved_requests=used.get('terra_requests'),video_metadata=frames,rest_terminal_and_peaks=metrics,cost_report='cost_report.json');write(run/'delivery_audit.json',row);out.append(row)
        print(ep,'delivery integrity',passed,'actual API requests',actual,flush=True)
    write(root/'delivery_audit.json',dict(passed=all(r['passed'] for r in out),episodes=out,checked_utc=now(),policy_inference=False,training=False))
    with sqlite3.connect(ROOT/'runs/_task_budgets'/(c['task_group_id']+'.sqlite')) as source:
        with sqlite3.connect(root/'verification/final_budget_ledger.sqlite') as destination:source.backup(destination)
    if not all(r['passed'] for r in out):raise SystemExit(1)

if __name__=='__main__':main()
