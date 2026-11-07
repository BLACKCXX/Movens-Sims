"""Batch stop/circuit controls; no scene or budget rollback."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import fcntl
import os
import time
from ..util import read, write, now


class BatchPaused(RuntimeError):
    def __init__(self, stage, reason, retry_after=30):
        super().__init__(reason);self.stage=stage;self.retry_after=retry_after


def check_stop():
    stop=os.environ.get('MOVENS_BATCH_STOP_FILE')
    if stop and Path(stop).exists():raise BatchPaused('stopped','Batch STOP requested at a safe checkpoint',0)


@contextmanager
def api_slot(config):
    root=config.get('batch',{}).get('control_root')
    if not root:
        yield;return
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'api.lock').open('a') as lock:
        while True:
            check_stop()
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(.25)
        try:
            circuit=read(root/'api_circuit.json') if (root/'api_circuit.json').exists() else {}
            if circuit.get('manual_pause') or circuit.get('paused_until',0)>time.time():
                raise BatchPaused('waiting_api',circuit.get('reason','API circuit paused'),max(30,circuit.get('paused_until',time.time()+60)-time.time()))
            yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def record_api_status(config,status):
    root=config.get('batch',{}).get('control_root')
    if not root:return
    root=Path(root);path=root/'api_circuit.json';d=read(path) if path.exists() else {}
    if status['status']=='validated':
        write(path,dict(consecutive_failures=0,updated_utc=now(),last_model=status['model']));return
    if status['status']=='blocked_budget':return
    d['consecutive_failures']=d.get('consecutive_failures',0)+1;code=status.get('http_status')
    d.update(updated_utc=now(),reason=f"{status['status']}: {status.get('error_type','credentials/model review unavailable')}",http_status=code)
    if code in (401,403,404) or status.get('error_type')=='model_mismatch' or status['status']=='prepared_offline':d['manual_pause']=True
    if d['consecutive_failures']>=3:d['paused_until']=time.time()+120
    write(path,d)
    if d.get('manual_pause') or d.get('consecutive_failures',0)>=3:raise BatchPaused('waiting_api',d['reason'],120)
    raise BatchPaused('waiting_api',d['reason'],30*d['consecutive_failures'])
