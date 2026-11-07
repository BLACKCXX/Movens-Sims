"""Detached, bounded, durable full-402 reconstruction coordinator."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import csv
import fcntl
import html
import io
import json
import os
import re
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import traceback

from .util import ROOT,CODE_ROOT,read,write,sha,digest,now,video_info


def connect(root):
    db=sqlite3.connect(root/'tasks.sqlite',timeout=30,isolation_level='IMMEDIATE');db.row_factory=sqlite3.Row
    db.execute('PRAGMA synchronous=FULL');return db


def alive(pid):
    try:return Path(f'/proc/{int(pid)}/stat').read_text().split(')')[1].split()[0]!='Z'
    except (ValueError,FileNotFoundError,ProcessLookupError):return False


def process_stamp(pid):
    try:return Path(f'/proc/{int(pid)}/stat').read_text().split(')')[1].split()[19]
    except (FileNotFoundError,ProcessLookupError):return None


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);yield


def setup(config):
    from .reconstruction.full_prepare import create_manifest
    root=ROOT/config['run_root'];root.mkdir(parents=True,exist_ok=True)
    m=create_manifest(config)
    for record in read(m['assets_manifest']).values():
        asset=Path(record['path'])
        if sha(asset)!=record['sha256']:raise ValueError('Shared input asset changed: '+str(asset))
        asset.chmod(0o444)
    target=root/'resolved_config.json'
    if target.exists() and read(target)!=config:raise ValueError('Batch config changed; resume cannot relabel prior work or budgets')
    write(target,config)
    frozen=root/'frozen_code'
    if not (root/'code_ready.json').exists():
        if frozen.exists():shutil.rmtree(frozen)  # Incomplete unpublished code snapshot only.
        for folder in ('src','prompts','configs'):
            shutil.copytree(CODE_ROOT/folder,frozen/folder,ignore=shutil.ignore_patterns('__pycache__'))
        (frozen/'scripts').mkdir();shutil.copy2(CODE_ROOT/'scripts/movens-sim',frozen/'scripts/movens-sim');shutil.copy2(CODE_ROOT/'pyproject.toml',frozen/'pyproject.toml')
        write(root/'code_ready.json',{str(p.relative_to(frozen)):sha(p) for p in frozen.rglob('*') if p.is_file()})
    verify_code(root)
    with connect(root) as db:
        db.execute('CREATE TABLE IF NOT EXISTS tasks (episode INTEGER PRIMARY KEY, status TEXT, attempts INTEGER DEFAULT 0, errors INTEGER DEFAULT 0, not_before REAL DEFAULT 0, pid INTEGER, stamp TEXT, gpu INTEGER, started REAL, stage TEXT, reason TEXT, quality TEXT)')
        for e in m['entries']:
            db.execute('INSERT OR IGNORE INTO tasks(episode,status,reason) VALUES (?,?,?)',(e['episode'],'pending' if e['input_complete'] else 'missing_input', '; '.join(e['missing'])))
        # A stale database completion is not evidence that a delivery still
        # exists. Never skip deleted/corrupted bound files during resume.
        for row in db.execute("SELECT episode FROM tasks WHERE status='complete'").fetchall():
            run=root/f"episode_{row['episode']:06}";bad=[]
            try:
                for name,h in read(run/'delivery_binding.json')['artifacts'].items():
                    if not (run/name).is_file() or sha(run/name)!=h:bad.append(name)
                for name in ('result.json','report.md','delivery_checks.json'):
                    if not (run/name).is_file():bad.append(name)
            except (OSError,ValueError,KeyError):bad.append('delivery_binding.json')
            if bad:db.execute("UPDATE tasks SET status='failed',stage='delivery_integrity',reason=?,quality=NULL WHERE episode=?",('Incomplete completed delivery: '+','.join(bad),row['episode']))
    write(root/'handoff.json',dict(root=str(root),config=str(target),manifest=str(ROOT/config['manifest']),model=config['terra']['model'],initial_workers=config['batch']['initial_workers'],max_workers=config['batch']['max_workers'],
        commands={a:f"cd {ROOT} && .venv/bin/python -B scripts/run_full_reconstruction.py {a} --config configs/full402_h0_v1.json" for a in ('status','stop','resume')},
        scope='Static scene reconstruction only; no training, slicing or checkpoint tests',source_project_read_only=True))
    return root,m


def verify_code(root):
    selected=read(root/'active_code.json') if (root/'active_code.json').exists() else dict(code_root='frozen_code',hash_manifest='code_ready.json')
    for name,h in read(root/selected['hash_manifest']).items():
        p=root/selected['code_root']/name
        if not p.is_file() or sha(p)!=h:raise ValueError('Frozen execution code missing/changed: '+name)


def environment(root,gpu=None):
    code=read(root/'active_code.json')['code_root'] if (root/'active_code.json').exists() else 'frozen_code'
    env={**os.environ,'PYTHONPATH':str(root/code/'src'),'MOVENS_SIM_PROJECT_ROOT':str(ROOT),
         'MOVENS_BATCH_STOP_FILE':str(root/'STOP'),'MUJOCO_GL':'egl',
         'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
    if gpu is not None:env['MUJOCO_EGL_DEVICE_ID']=str(gpu)
    return env


def _receipt(root,ep,attempt):return root/'control/receipts'/f'episode_{ep:06}_attempt_{attempt:03}.json'


def worker(config,ep,attempt):
    from .reconstruction.workflow import run_pipeline
    from .reconstruction.control import BatchPaused
    root=ROOT/config['run_root'];run=root/f'episode_{ep:06}';receipt=_receipt(root,ep,attempt);start=time.monotonic()
    entry=next(e for e in read(ROOT/config['manifest'])['entries'] if e['episode']==ep)
    result=dict(episode=ep,attempt=attempt,pid=os.getpid(),started_utc=now(),status='failed')
    try:
        cv=__import__('cv2');cv.setNumThreads(1)
        approve_rolling_code(root,run,config)
        data=run_pipeline(entry,config,enabled=True,resume=True)
        checks={}
        for name,frames in [('ego_static.mp4',150),('settle_diagnostic.mp4',30)]:
            info=video_info(run/name);checks[name]=info
            if (info['width'],info['height'],info['fps'],info['frames'])!=(640,480,30.,frames):raise ValueError('Final video metadata mismatch: '+name)
        if not data['reload_passed']:raise ValueError('Export reload did not reproduce the fitted candidate')
        write(run/'delivery_checks.json',dict(passed=True,videos=checks,reload=read(run/'reload_check.json'),artifact_binding=sha(run/'delivery_binding.json'),checks_utc=now()))
        result.update(status='complete',quality={k:data.get(k) for k in ['processing_complete','construction_complete','visual_qualified','physical_passed','needs_review','geometry_passed','appearance_status','terra_actual_http_requests','terra_requests_reserved','render_optimization_evaluations','geometry_fit_rounds','optimization_wall_seconds']},
                      objects={k:{s:v.get(s) for s in ['iou','centroid_error_px','symmetric_edge_distance_px','area_relative_error']} for k,v in data['metrics_after']['objects'].items()})
    except BatchPaused as exc:result.update(status=exc.stage,reason=str(exc),retry_after=exc.retry_after)
    except Exception as exc:
        observation_error=str(exc).startswith(('No plausible pixel component','No segmented component'))
        constraint_error=getattr(exc,'code',None) in ('outside_support','initial_object_too_wide','physical_constraint','invalid_support_or_intersection','no_feasible_initial_layout') or str(exc).startswith('No valid baseline/candidate')
        classification='observation_error' if observation_error else 'reconstruction_constraint' if constraint_error else 'engineering_error'
        result.update(error_type=type(exc).__name__,reason=str(exc)[:900],retryable=classification=='engineering_error',classification=classification);run.mkdir(parents=True,exist_ok=True)
        (run/f'error_attempt_{attempt:03}.log').write_text(traceback.format_exc());traceback.print_exc()
    result['wall_seconds']=time.monotonic()-start;result['ended_utc']=now();write(receipt,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def approve_rolling_code(root,run,config):
    """Only a published, hash-bound code transition may update an unfinished run."""
    if not (run/'run_identity.json').exists() or not (root/'rolling_code_transition.json').exists():return
    from .reconstruction.workflow import identity,verify_sources
    previous=read(run/'run_identity.json');current=identity(run,config)
    if previous==current:return
    transition=read(root/'rolling_code_transition.json')
    if digest(previous['code']) not in transition['from_code_sha256s'] or digest(current['code'])!=transition['to_code_sha256']:return
    if (run/'session.json').exists() and read(run/'session.json').get('complete'):return
    verify_sources(run)
    if any(previous.get(k)!=current.get(k) for k in ('config_sha256','source_manifest_sha256','observation_sha256','mujoco')):raise ValueError('Rolling continuation cannot change inputs/config/observations')
    proof=run/'code_compatibility/approved_continuation.json'
    if proof.exists():write(proof.with_name('prior_proof_'+sha(proof)[:12]+'.json'),read(proof))
    write(proof,dict(previous_identity_sha256=digest(previous),next_identity_sha256=digest(current),reason=transition['reason'],transition_sha256=sha(root/'rolling_code_transition.json'),config_inputs_observations_unchanged=True,budgets_unchanged=True,approved_utc=now()))


def resources(previous=None):
    cgroup=Path('/sys/fs/cgroup');memory_max=(cgroup/'memory.max').read_text().strip();current=int((cgroup/'memory.current').read_text())
    maxbytes=int(memory_max) if memory_max!='max' else 180*1024**3
    usage=int(dict(x.split() for x in (cgroup/'cpu.stat').read_text().splitlines())['usage_usec']);t=time.monotonic()
    cores=(usage-previous['cpu_usage_usec'])/1e6/max(.01,t-previous['monotonic']) if previous else 0.
    quota=(cgroup/'cpu.max').read_text().split();limit=float(quota[0])/float(quota[1]) if quota[0]!='max' else os.cpu_count()
    rows=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    gpus=[]
    if rows.returncode==0:
        for line in rows.stdout.splitlines():
            a,b,c=line.split(',');gpus.append(dict(id=int(a),free_mib=int(b),utilization=int(c)))
    return dict(available_memory_gib=(maxbytes-current)/1024**3,cgroup_memory_limit_gib=maxbytes/1024**3,cpu_limit_cores=limit,cpu_used_cores=cores,
                disk_free_gb=shutil.disk_usage(ROOT).free/1e9,gpus=gpus,cpu_usage_usec=usage,monotonic=t)


def publish(root,config,phase='running',resource=None):
    with connect(root) as db:rows=[dict(r) for r in db.execute('SELECT * FROM tasks ORDER BY episode')]
    counts={s:sum(r['status']==s for r in rows) for s in ['pending','running','complete','failed','missing_input','retry','waiting_api','stopped']}
    training={}
    if config.get('manifest') and (ROOT/config['manifest']).exists():training={e['episode']:e['training_selected'] for e in read(ROOT/config['manifest'])['entries']}
    qualified={k:0 for k in ['construction_complete','visual_qualified','physical_passed','needs_review']};csvrows=[];cards=[]
    for r in rows:
        q=json.loads(r['quality']) if r['quality'] else {};quality=q.get('quality',{});objects=q.get('objects',{})
        for k in qualified:qualified[k]+=bool(quality.get(k))
        row=dict(episode=r['episode'],training_selected=training.get(r['episode']),status=r['status'],stage=r['stage'],input_complete=r['status']!='missing_input',attempts=r['attempts'],reason=r['reason'],**{k:quality.get(k) for k in ['construction_complete','visual_qualified','physical_passed','needs_review','appearance_status','geometry_fit_rounds','render_optimization_evaluations','terra_actual_http_requests','terra_requests_reserved']})
        for label in ('apple','plate'):
            for key in ('iou','centroid_error_px','symmetric_edge_distance_px'):row[label+'_'+key]=objects.get(label,{}).get(key)
        csvrows.append(row);ep=f"episode_{r['episode']:06}"
        links=f'<a href="{ep}/index.html">图像与报告</a> · <a href="{ep}/ego_static.mp4">静态视频</a> · <a href="{ep}/settle_diagnostic.mp4">静置检查</a> · <a href="{ep}/delivery/static/portable/scene.xml">XML</a>' if r['status']=='complete' else f'<a href="logs/{ep}_attempt_{r["attempts"]:03}.log">日志</a>'
        cards.append(f'<tr><td>{r["episode"]}</td><td>{html.escape(r["status"])}</td><td>{html.escape(r["stage"] or "")}</td><td>{links}</td><td>{html.escape(r["reason"] or "")}</td></tr>')
    buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    temp=root/'results.csv.tmp';temp.write_text(buf.getvalue());temp.replace(root/'results.csv')
    summary=dict(phase=phase,total=402,counts=counts,quality=qualified,pending_total=sum(r['status'] not in ['complete','failed','missing_input'] for r in rows),
                 model=config['terra']['model'],resources=resource,updated_utc=now(),coordinator_pid=os.getpid(),running=[{k:r[k] for k in ['episode','pid','gpu','stage']} for r in rows if r['status']=='running'])
    write(root/'summary.json',summary)
    doc='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Movens-Sim 全402条</title><style>body{font:16px system-ui;background:#171b21;color:#eee;margin:24px}a{color:#8cd3ff}td{padding:8px;border-bottom:1px solid #444}pre{white-space:pre-wrap}</style><h1>Movens-Sim · 402条 h0 静态重建</h1><p>固定 Mount 相机；静态仿真展示，非真实采集、非交互回放。构建、视觉、物理结果分别统计。</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><a href="results.csv">全402行汇总表</a><table><tr><th>Episode</th><th>状态</th><th>阶段</th><th>产物</th><th>原因</th></tr>'+''.join(cards)+'</table>'
    p=root/'index.html.tmp';p.write_text(doc);p.replace(root/'index.html');return summary


def coordinate(config):
    root=ROOT/config['run_root'];settings=config['batch'];children={};previous=None;workers=settings['initial_workers'];last_publish=0
    with locked(root/'control/coordinator.lock'):
        def stop(signum,frame):write(root/'STOP',dict(requested_utc=now(),signal=signum))
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        while True:
            t=time.time()
            with connect(root) as db:running=[dict(r) for r in db.execute("SELECT * FROM tasks WHERE status='running'")]
            for row in running:
                ep=row['episode'];p=children.get(ep)
                if p is not None:p.poll()
                active=alive(row['pid']) and process_stamp(row['pid'])==row['stamp']
                if active and t-row['started']>settings['worker_timeout_seconds']:
                    os.killpg(row['pid'],signal.SIGTERM);active=False
                if active:
                    stage='prepare';session=root/f'episode_{ep:06}/session.json'
                    if session.exists():
                        d=read(session);stage=d.get('stage','geometry')
                        if d.get('complete'):stage='delivery_checks'
                    with connect(root) as db:db.execute('UPDATE tasks SET stage=? WHERE episode=?',(stage,ep))
                    continue
                receipt=_receipt(root,ep,row['attempts']);data=read(receipt) if receipt.exists() else dict(status='failed',error_type='WorkerExited',reason='Process exited without a durable result receipt')
                status=data['status'];errors=row['errors'];not_before=0
                if status=='failed':
                    errors+=1
                    if data.get('retryable',True) and errors<=settings['engineering_retries']:status='retry';not_before=t+30*errors
                if status=='waiting_api':not_before=t+data.get('retry_after',30)
                with connect(root) as db:db.execute('UPDATE tasks SET status=?,errors=?,not_before=?,pid=NULL,stage=?,reason=?,quality=? WHERE episode=?',
                    (status,errors,not_before,'complete' if status=='complete' else data.get('error_type',status),data.get('reason'),json.dumps(data) if status=='complete' else None,ep))
                children.pop(ep,None)
                if status=='failed' and data.get('classification')=='engineering_error':
                    fingerprint=re.sub(r'episode_\d+','episode_ID',data.get('error_type','')+':'+data.get('reason',''))
                    path=root/'control/engineering_failures.json';history=read(path) if path.exists() else []
                    history.append(dict(episode=ep,fingerprint=fingerprint,utc=now()));write(path,history)
                    if len(history)>=3 and len({r['fingerprint'] for r in history[-3:]})==1:write(root/'control/system_pause.json',dict(reason=fingerprint,stage=data.get('error_type'),utc=now()))
            with connect(root) as db:
                counts=dict(db.execute('SELECT status,COUNT(*) FROM tasks GROUP BY status').fetchall());running_count=counts.get('running',0)
            if time.time()-last_publish>=settings['heartbeat_seconds']:
                snapshot=resources(previous);previous=snapshot
                stable=counts.get('complete',0)>=settings['scale_up_after_completed'] and counts.get('failed',0)==0
                capacity=snapshot['available_memory_gib']>=settings['scale_up_available_memory_gib'] and snapshot['cpu_used_cores']<settings['scale_up_cpu_cores_max'] and len(snapshot['gpus'])>=2 and all(g['free_mib']>=settings['scale_up_gpu_free_mib'] and g['utilization']<80 for g in snapshot['gpus'])
                workers=settings['max_workers'] if stable and capacity else settings['initial_workers']
                publish(root,config,'stopping' if (root/'STOP').exists() else 'running',snapshot)
                write(root/'heartbeat.json',dict(pid=os.getpid(),utc=now(),workers=workers,running=running_count));last_publish=time.time()
            if (root/'STOP').exists():
                if not running_count:publish(root,config,'stopped',previous);break
                time.sleep(1);continue
            if counts.get('complete',0)+counts.get('failed',0)+counts.get('missing_input',0)==402:
                publish(root,config,'complete',previous);break
            circuit=read(root/'control/api_circuit.json') if (root/'control/api_circuit.json').exists() else {}
            paused=circuit.get('manual_pause') or circuit.get('paused_until',0)>t
            if (root/'control/system_pause.json').exists() or (previous and previous['disk_free_gb']<settings['reserve_disk_gb']):
                publish(root,config,'paused_system_or_disk',previous);time.sleep(5);continue
            if paused and counts.get('waiting_api',0)>=8:time.sleep(2);continue
            while running_count<workers:
                with connect(root) as db:
                    db.execute('BEGIN IMMEDIATE')
                    statuses="('pending','retry','stopped')" if paused else "('pending','retry','waiting_api','stopped')"
                    row=db.execute(f"SELECT * FROM tasks WHERE status IN {statuses} AND not_before<=? ORDER BY CASE WHEN status='pending' THEN 1 ELSE 0 END,episode LIMIT 1",(t,)).fetchone()
                    if row is None:break
                    row=dict(row);ep=row['episode'];attempt=row['attempts']+1
                    if row['status']=='waiting_api' and paused:break
                    occupied={g:db.execute("SELECT COUNT(*) FROM tasks WHERE status='running' AND gpu=?",(g,)).fetchone()[0] for g in settings['gpu_ids']}
                    gpu=min(settings['gpu_ids'],key=lambda g:occupied[g])
                    db.execute("UPDATE tasks SET status='running',attempts=?,started=?,stage='starting',gpu=? WHERE episode=?",(attempt,t,gpu,ep))
                log=root/'logs'/f'episode_{ep:06}_attempt_{attempt:03}.log';log.parent.mkdir(exist_ok=True)
                command=[str(ROOT/'.venv/bin/python'),'-B','-u','-m','movens_sim.full_batch','worker','--config',str(root/'resolved_config.json'),'--episode',str(ep),'--attempt',str(attempt)]
                with log.open('a') as f:p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,env=environment(root,gpu),start_new_session=True)
                children[ep]=p
                with connect(root) as db:db.execute('UPDATE tasks SET pid=?,stamp=? WHERE episode=?',(p.pid,process_stamp(p.pid),ep))
                running_count+=1
                print(f'Start episode {ep:06} attempt {attempt}, PID {p.pid}, EGL GPU {gpu}',flush=True)
            time.sleep(1)


def start(config,resume=False):
    root=ROOT/config['run_root'];root.mkdir(parents=True,exist_ok=True)
    with locked(root/'control/start.lock'):
        if (root/'job.json').exists():
            job=read(root/'job.json')
            if alive(job['pid']) and process_stamp(job['pid'])==job.get('stamp'):
                if resume:
                    (root/'STOP').unlink(missing_ok=True);(root/'control/system_pause.json').unlink(missing_ok=True);(root/'control/api_circuit.json').unlink(missing_ok=True)
                return dict(already_running=True,**job)
            if not resume:raise ValueError('Existing batch requires resume; do not create a new task to reset budgets')
        root,m=setup(config)
        if resume:
            (root/'STOP').unlink(missing_ok=True);(root/'control/system_pause.json').unlink(missing_ok=True);(root/'control/api_circuit.json').unlink(missing_ok=True)
        log=root/'logs/coordinator.log';log.parent.mkdir(exist_ok=True)
        command=[str(ROOT/'.venv/bin/python'),'-B','-u','-m','movens_sim.full_batch','coordinate','--config',str(root/'resolved_config.json')]
        with log.open('a') as f:p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,env=environment(root),start_new_session=True)
        job=dict(pid=p.pid,pgid=p.pid,stamp=process_stamp(p.pid),started_utc=now(),command=command,log=str(log),stdin='DEVNULL',detached=True,root=str(root))
        if (root/'job.json').exists():
            from .util import append
            append(root/'job_history.jsonl',read(root/'job.json'))
        write(root/'job.json',job);return job


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['start','status','stop','resume','coordinate','worker','prepare'])
    p.add_argument('--config',default='configs/full402_h0_v1.json');p.add_argument('--episode',type=int);p.add_argument('--attempt',type=int,default=1);a=p.parse_args(argv)
    config=read(ROOT/a.config);root=ROOT/config['run_root']
    if a.action in ('start','resume'):print(json.dumps(start(config,a.action=='resume'),indent=2,ensure_ascii=False))
    elif a.action=='coordinate':coordinate(config)
    elif a.action=='worker':worker(config,a.episode,a.attempt)
    elif a.action=='prepare':setup(config);print(json.dumps(publish(root,config,'prepared'),indent=2))
    elif a.action=='stop':
        write(root/'STOP',dict(requested_utc=now(),reason='operator safe stop'));print('STOP persisted; this batch checkpoints at safe boundaries')
    else:
        d=read(root/'summary.json') if (root/'summary.json').exists() else dict(status='not_started')
        job=read(root/'job.json') if (root/'job.json').exists() else {};d['coordinator_alive']=alive(job.get('pid',0));d['job']=job;print(json.dumps(d,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
