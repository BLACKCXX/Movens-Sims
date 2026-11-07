#!/usr/bin/env python3
"""Replace only our detached coordinator while its independent workers continue."""
from pathlib import Path
import os
import signal
import sqlite3
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from movens_sim.util import ROOT,read,write,now
from movens_sim.full_batch import alive,process_stamp,start

root=ROOT/'runs/full402_h0_v1';job=read(root/'job.json');pid=job['pid'];workers=[]
if not alive(pid) or process_stamp(pid)!=job['stamp']:raise RuntimeError('Coordinator identity changed')
command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
if 'movens_sim.full_batch coordinate' not in command or str(root/'resolved_config.json') not in command:raise RuntimeError('Refusing to signal an unrelated process')
frozen=False
try:
    for attempt in range(20):
        os.kill(pid,signal.SIGSTOP);frozen=True
        try:
            with sqlite3.connect(root/'tasks.sqlite',timeout=.1) as db:
                db.execute('BEGIN IMMEDIATE');workers=db.execute("SELECT episode,pid,stamp FROM tasks WHERE status='running'").fetchall();db.rollback()
            if all(w[1] and (not alive(w[1]) or process_stamp(w[1])==w[2]) for w in workers):break
        except sqlite3.OperationalError:pass
        os.kill(pid,signal.SIGCONT);frozen=False;time.sleep(.1)
    else:raise RuntimeError('Could not find a committed coordinator checkpoint')
    os.kill(pid,signal.SIGKILL);frozen=False  # PID only, never the worker groups.
    for attempt in range(100):
        if not alive(pid):break
        time.sleep(.05)
    new=start(read(root/'resolved_config.json'),resume=True)
    write(root/'verification/coordinator_rollover.json',dict(old_job=job,new_job=new,workers_preserved=workers,worker_groups_signalled=False,only_coordinator_replaced=True,utc=now()))
    print({'new_pid':new['pid'],'independent_workers_preserved':len(workers)})
finally:
    if frozen and alive(pid):os.kill(pid,signal.SIGCONT)
