#!/usr/bin/env python3
"""Reaccept the three final artifact bundles; no fitting, policy or new model requests."""
import argparse
import json
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MUJOCO_GL','egl')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from movens_sim.util import ROOT,read,write,sha,now
from movens_sim.reconstruction.acceptance import accept_from_disk
from movens_sim.reconstruction.budget import BudgetManager

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--episodes',type=int,nargs='+',default=[0,1,2]);a=p.parse_args();c=read(ROOT/a.config)
    if c['workflow']!='classified_reconstruction_v4' or sorted(a.episodes)!=[0,1,2] or c['allowed_episodes']!=[0,1,2]:raise ValueError('Exactly the authorized three episodes are required')
    rows=[]
    for ep in a.episodes:
        run=ROOT/c['run_root']/f'episode_{ep:06}';binding=read(run/'delivery_binding.json')
        changed=[name for name,h in binding['artifacts'].items() if not (run/name).is_file() or sha(run/name)!=h]
        if changed:raise ValueError('Frozen artifact changed: '+str(changed))
        b=BudgetManager(ROOT/'runs/_task_budgets'/(c['task_group_id']+'.sqlite'),f"{c['task_group_id']}/episode_{ep:06}",c['budget'])
        previous=run/'reacceptance/final'
        if previous.exists():previous.rename(previous.with_name('history_'+now().replace(':','').replace('+','_')))
        result,_,_=accept_from_disk(run,c,run/'reacceptance/final',b);review=read(run/'final_review.json');result['final_review_binding_passed']=review['delivery_hash']==binding['scene_parameter_asset_hash'];result['final_terra_verdict']=review['review']['verdict'] if review.get('review') else 'unreviewed'
        write(run/'reacceptance/final/acceptance.json',result);write(run/'budget_ledger.json',b.summary());rows.append(dict(episode=ep,engineering_passed=result['engineering_passed'],geometry=result['geometry_numeric_passed'],coverage=result['coverage_numeric_passed'],initial_support=result['initial_support_passed'],rest=result['physical_passed'],reload=result['reload_passed'],final_terra=result['final_terra_verdict'],report=str(run/'reacceptance/final/acceptance.json')))
        print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
    write(ROOT/c['run_root']/'final_reacceptance.json',dict(episodes=rows,checked_utc=now(),policy_inference=False,training=False))

if __name__=='__main__':main()
