#!/usr/bin/env python3
"""Preserve a failed frozen candidate and continue a successor under the SAME 20-call cap."""
import argparse
from pathlib import Path
import shutil
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from movens_sim.util import ROOT,read,write,sha,now
from movens_sim.full_batch import alive,process_stamp
from movens_sim.reconstruction.budget import BudgetManager

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--episode',type=int,required=True);p.add_argument('--reason',required=True);a=p.parse_args();c=read(ROOT/a.config);root=ROOT/c['run_root'];run=root/f'episode_{a.episode:06}'
    if c['workflow']!='classified_reconstruction_v4' or a.episode not in [0,1,2]:raise ValueError('Isolated v4 task only')
    job=read(root/'job.json')
    if alive(job['pid']) and process_stamp(job['pid'])==job['stamp']:raise ValueError('Stop the isolated coordinator at a safe checkpoint first')
    review=read(run/'final_review.json');binding=read(run/'delivery_binding.json')
    if not review.get('review') or review['review']['verdict']=='pass':raise ValueError('Only failed frozen candidates may get a successor')
    for name,h in binding['artifacts'].items():
        if sha(run/name)!=h:raise ValueError('Original frozen candidate changed: '+name)
    archive=run/'revisions'/('failed_terminal_'+binding['candidate_id']);archive.mkdir(parents=True,exist_ok=False)
    for name in ('delivery','final_acceptance','terra','mcp_plans','code_versions'):
        if (run/name).exists():shutil.copytree(run/name,archive/name)
    for path in run.iterdir():
        if path.is_file() and not path.name.startswith('.'):shutil.copy2(path,archive/path.name)
    for name,h in binding['artifacts'].items():
        if sha(archive/name)!=h:raise ValueError('Archive verification failed: '+name)
    ledger=BudgetManager(ROOT/'runs/_task_budgets'/(c['task_group_id']+'.sqlite'),f"{c['task_group_id']}/episode_{a.episode:06}",c['budget']);before=ledger.used('terra_requests')
    ledger.begin_repair_revision(binding['candidate_id'],binding['scene_parameter_asset_hash'],a.reason,archive/'final_review.json')
    for name in ('delivery','final_acceptance'):shutil.rmtree(run/name)
    for name in binding['artifacts']:
        path=run/name
        if path.is_file():path.unlink()
    for name in ('final_review.json','result.json','report.md','index.html','delivery_binding.json','delivery_checks.json'):(run/name).unlink(missing_ok=True)
    state=read(run/'session.json');state.update(complete=False,classification_complete=False,stage='geometry',active_category=None,classification_progress=[],pending_query=None,repair_revision=dict(parent_candidate_id=binding['candidate_id'],archive=str(archive.relative_to(run)),reason=a.reason,terra_spent_before=before,started_utc=now()))
    write(run/'session.json',state)
    with sqlite3.connect(root/'tasks.sqlite') as db:db.execute("UPDATE tasks SET status='stopped',stage='successor_repair',reason=?,quality=NULL WHERE episode=?",(a.reason,a.episode))
    write(root/'verification'/f'successor_episode_{a.episode:06}.json',dict(archive=str(archive),original_binding_verified=True,terra_before=before,terra_after=ledger.used('terra_requests'),same_task_id=ledger.task_id,limit=20,reason=a.reason))
    print(str(archive))

if __name__=='__main__':main()
