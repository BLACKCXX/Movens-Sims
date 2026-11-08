#!/usr/bin/env python3
"""Independent process: reload final files and perform NEW native integration."""
import argparse
import os
from pathlib import Path
import sys
import time
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('MUJOCO_GL','egl')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from movens_sim.util import ROOT,CODE_ROOT,read,write,sha,now
from movens_sim.reconstruction.acceptance import accept_from_disk
from movens_sim.reconstruction.budget import BudgetManager
from movens_sim.reconstruction.v5_diagnostics import extra_views,plots


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',default='reacceptance/final');a=p.parse_args();run=Path(a.run).resolve();c=read(run/'resolved_config.json')
    if c['workflow']!='terra_closed_loop_v5' or c['allowed_episodes']!=[0,1,2]:raise ValueError('Only authorized isolated v5 episodes')
    ep=read(run/'source_manifest.json')['episode'];started=now();folder=run/a.output
    if (folder/'acceptance.json').exists():
        archive=folder.with_name(folder.name+'_history_'+started.replace(':','').replace('+','_'));folder.rename(archive)
    b=BudgetManager(ROOT/'runs/_task_budgets'/(c['task_group_id']+'.sqlite'),f"{c['task_group_id']}/episode_{ep:06}",c['budget'])
    with b.computation(dict(kind='complete_independent_v5_acceptance'),whole_task_slice=True):
        report,rgb,masks=accept_from_disk(run,c,folder,b)
        extra_views(run/'delivery/static/portable/scene.xml',folder/'views',b)
        plots(report['metrics'],folder/'photometric')
    report.update(version='disk_acceptance_v5.1',started_utc=started,ended_utc=now(),validator_pid=os.getpid(),parent_pid=os.getppid(),command=sys.argv,code_root=str(CODE_ROOT),fresh_native_integration=True,policy_inference=False,training=False,details_policy=c['v5']['policy'],execution_code={str(p.relative_to(CODE_ROOT)):sha(p) for p in (CODE_ROOT/'src/movens_sim/reconstruction').glob('v5*.py')})
    write(folder/'acceptance.json',report);print({k:report[k] for k in ('episode','engineering_passed','geometry_numeric_passed','coverage_numeric_passed','initial_support_passed','physical_passed','reload_passed')},flush=True)

if __name__=='__main__':main()
