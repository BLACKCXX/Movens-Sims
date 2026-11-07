#!/usr/bin/env python3
"""Conservatively add omitted whole-round overhead from the first executed timer.

This never refunds budget, changes an upper limit, starts search, or changes a
candidate. Current code wraps complete searches, so it needs no reconciliation.
"""
import argparse
from pathlib import Path
import json

from movens_sim.reconstruction.budget import BudgetManager
from movens_sim.util import ROOT,read,write,now


def reconcile(run):
    run=Path(run).resolve();record=run/'timing_reconciliation.json'
    if record.exists():return read(record)
    executed=(run/'code_snapshot/src/movens_sim/reconstruction/session.py').read_text()
    if 'whole_task_slice=True' in executed:raise ValueError('Executed code already times complete fits; no additive audit is needed')
    config=read(run/'resolved_config.json');b=BudgetManager(ROOT/'runs/_task_budgets'/(config['task_group_id']+'.sqlite'),config['task_group_id'],config['budget']);summary=b.summary();rounds=[]
    for event in summary['events']:
        if event['kind']!='geometry_rounds':continue
        index=event['detail']['round_index'];file=run/f'geometry_round_{index:02}.json'
        if not file.exists() or event['status']!='completed':raise ValueError('Cannot reconcile unfinished round')
        seconds=file.stat().st_mtime-event['created']
        if seconds<0:raise ValueError('Inconsistent wall timestamps')
        rounds.append(dict(round_index=index,start_utc_unix=event['created'],summary_mtime_unix=file.stat().st_mtime,seconds=seconds))
    outside=sum(e['amount'] for e in summary['events'] if e['kind']=='optimization_seconds' and e['detail'].get('kind')=='old_baseline_comparison')
    before=b.used('optimization_seconds');target=sum(r['seconds'] for r in rounds)+outside;added=max(0.,target-before)
    if added:
        event=b.reserve('optimization_seconds',added,dict(kind='audit_add_omitted_round_overhead',source='Durable pre-fit reservation UTC to immutable round-summary mtime; includes solver/checkpoint/cache overhead, excludes API and diagnostics',rounds=rounds))
        b.finish(event,'audit_reconciled',added)
    result=dict(recorded_utc=now(),basis='Conservative whole-round wall from durable artifact timestamps; initial executed version timed inner evaluation/projection units only. No spending refunded; current source wraps the whole search.',original_kernel_seconds=before,added_overhead_seconds=added,accounted_full_optimization_seconds=b.used('optimization_seconds'),rounds=rounds,outside_fit_baseline_seconds=outside,search_performed=False,scene_modified=False,upper_limits_modified=False)
    write(record,result);write(run/'budget_ledger.json',b.summary());output=read(run/'result.json');output['optimization_kernel_seconds_original']=before;output['optimization_wall_seconds']=result['accounted_full_optimization_seconds'];output['time_accounting_audit']=result;write(run/'result.json',output)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();print(json.dumps(reconcile(a.run),ensure_ascii=False,indent=2))
