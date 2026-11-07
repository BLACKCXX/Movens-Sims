#!/usr/bin/env python3
"""Ordered, read-only vision review; numeric render budgets never reset."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID','0')
import shutil
import time
from pathlib import Path
from movens_sim.gpt_adapter import call_or_prepare
from movens_sim.pipeline import fitted_scene,complete_receipt,finalize,report
from movens_sim.util import ROOT,read,write,sha,now

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prompt-version',default='diagnostic_v2');p.add_argument('--round-index',type=int,default=3);args=p.parse_args()
    base_config=read(ROOT/'configs/batch_v1.json');config=base_config.copy();config['prompt_version']=args.prompt_version;manifest=read(ROOT/config['manifest']);root=ROOT/config['run_root'];records=[]
    for index,e in enumerate(manifest['entries']):
        run=root/f"episode_{e['episode']:06}";scene=fitted_scene(run);started=time.monotonic()
        review_path=run/'gpt'/('read_only_review_'+args.prompt_version+'.json')
        if review_path.exists():
            records.append(read(review_path));continue
        if index==2:
            write(root/('prompt_freeze_'+args.prompt_version+'.json'),dict(version=config['prompt_version'],sha256=sha(ROOT/'prompts'/(args.prompt_version+'.txt')),model=config['gpt']['model'],first_two=[r['episode'] for r in records],first_two_api_valid=all(r['call_status']=='api_validated' for r in records),freeze_utc=now(),scope='Ordered post-fit diagnostic evaluation; no prompt text or schema changes after first two episodes. Numeric histories are retained separately.'))
        status,parsed=call_or_prepare(scene,run,config,'appearance',args.round_index,dict(render_evaluations_remaining=0,reason='Read-only final diagnostic; no numerical candidate acceptance'),enabled=True)
        record=dict(episode=e['episode'],phase='read_only_postfit_review',prompt_version=config['prompt_version'],prompt_sha256=sha(ROOT/'prompts'/(args.prompt_version+'.txt')),call_status=status['status'],model=status['model'],actual_api_call=status.get('api_call_performed',False),parameter_updates_applied=False,model_decision=parsed['decision'] if parsed else None,observations=parsed['observations'] if parsed else None,review_reason=parsed['review_reason'] if parsed else None,code_sha256=sha(__file__),wall_seconds=time.monotonic()-started)
        write(review_path,record);records.append(record)
        # Preserve the previous stage receipt and old API summary before supplementing them.
        history=run/'receipts_history';history.mkdir(exist_ok=True)
        shutil.copy2(run/'receipts/fit.json',history/('fit_before_'+args.prompt_version+'.json'));shutil.copy2(run/'gpt_summary.json',history/('gpt_summary_before_'+args.prompt_version+'.json'))
        summary=read(run/'gpt_summary.json');summary['actual_calls']=summary.get('actual_calls',0)+int(record['actual_api_call']);summary['api_actually_verified']=summary.get('api_actually_verified',False) or status['status']=='api_validated';summary['readonly_review']=record;write(run/'gpt_summary.json',summary)
        receipt=read(run/'receipts/fit.json');outputs=[run/r['path'] for r in receipt['outputs']]+list((run/'gpt').rglob('*.json'))
        outputs=list(dict.fromkeys(outputs));identity=read(run/'run_identity.json');complete_receipt(run,'fit',identity,outputs,receipt['wall_seconds']+record['wall_seconds'])
        finalize(run,e,base_config,identity)
        print(f"episode {e['episode']:06} read-only GPT review: {status['status']}",flush=True)
    write(root/('gpt_phase_evaluation_'+args.prompt_version+'.json'),dict(records=records,prompt_changes=0,first_two=records[:2],frozen_evaluation=records[2:],numeric_budget_reset=False,api_credentials_logged=False))
    print(report(manifest,base_config),flush=True)

if __name__=='__main__':main()
