"""Task-scoped, transactional, irreversible reservations shared across processes."""
from __future__ import annotations

import contextlib
import json
import sqlite3
import time
import uuid
import signal
import threading
import math
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


class BudgetManager:
    def __init__(self, path, task_id, limits):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.task_id, self.limits = task_id, limits
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, limits TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, task TEXT, kind TEXT, amount REAL, status TEXT, detail TEXT, created REAL, elapsed REAL DEFAULT 0)')
            encoded = json.dumps(limits, sort_keys=True)
            db.execute('INSERT OR IGNORE INTO tasks VALUES (?,?)', (task_id, encoded))
            if db.execute('SELECT limits FROM tasks WHERE id=?', (task_id,)).fetchone()[0] != encoded:
                raise ValueError('Task limits changed; existing budget cannot be reset or enlarged')

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level='IMMEDIATE')
        db.execute('PRAGMA synchronous=FULL')
        return db

    def reserve(self, kind, amount=1, detail=None):
        from .control import check_stop
        check_stop()
        if kind not in self.limits or isinstance(amount,bool) or not isinstance(amount,(int,float)) or not math.isfinite(amount) or amount < 0:
            raise ValueError('Unknown budget kind or negative reservation')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if kind=='geometry_rounds' and db.execute("SELECT COUNT(*) FROM events WHERE task=? AND kind='terra_final'",(self.task_id,)).fetchone()[0]:
                raise BudgetExceeded('Task is frozen for terminal read-only review')
            used = db.execute('SELECT COALESCE(SUM(amount),0) FROM events WHERE task=? AND kind=?', (self.task_id, kind)).fetchone()[0]
            if used + amount > self.limits[kind] + 1e-9:
                raise BudgetExceeded(f'{kind}: {used:g}+{amount:g} exceeds {self.limits[kind]:g}')
            event = uuid.uuid4().hex
            db.execute('INSERT INTO events(id,task,kind,amount,status,detail,created) VALUES (?,?,?,?,?,?,?)',
                       (event, self.task_id, kind, amount, 'reserved', json.dumps(detail or {}, sort_keys=True), time.time()))
        return event

    def finish(self, event, status='completed', elapsed=0.):
        with self.connect() as db:
            db.execute('UPDATE events SET status=?,elapsed=? WHERE id=? AND task=?', (status, elapsed, event, self.task_id))

    def remaining(self, kind):
        return max(0., self.limits[kind] - self.used(kind))

    def used(self,kind):
        with self.connect() as db:return db.execute('SELECT COALESCE(SUM(amount),0) FROM events WHERE task=? AND kind=?',(self.task_id,kind)).fetchone()[0]

    def reserve_render(self, pool, detail):
        from .control import check_stop
        check_stop()
        # Reserve the pool and total in one transaction. Neither may be refunded.
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT COUNT(*) FROM events WHERE task=? AND kind='terra_final'",(self.task_id,)).fetchone()[0]:raise BudgetExceeded('Task is frozen; no further optimization renders')
            for kind in ('render_total', pool):
                used = db.execute('SELECT COALESCE(SUM(amount),0) FROM events WHERE task=? AND kind=?', (self.task_id, kind)).fetchone()[0]
                if used + 1 > self.limits[kind]:
                    raise BudgetExceeded(kind)
            ids = []
            for kind in ('render_total', pool):
                event = uuid.uuid4().hex
                db.execute('INSERT INTO events(id,task,kind,amount,status,detail,created) VALUES (?,?,?,?,?,?,?)',
                           (event, self.task_id, kind, 1, 'reserved', json.dumps(detail, sort_keys=True), time.time()))
                ids.append(event)
        return ids

    def reserve_terra(self, final=False, detail=None):
        # One request always remains available for the terminal read-only review.
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            rows = db.execute("SELECT kind,COALESCE(SUM(amount),0) FROM events WHERE task=? AND kind IN ('terra_requests','terra_final') GROUP BY kind", (self.task_id,)).fetchall()
            used = dict(rows)
            limit = self.limits['terra_requests'] if final else self.limits['terra_requests']-1
            if (used.get('terra_final', 0) and not final) or used.get('terra_requests', 0) >= limit:
                raise BudgetExceeded('Terra request limit/final reservation')
            if used.get('terra_final',0) and final:
                marker=db.execute("SELECT created,detail FROM events WHERE task=? AND kind='terra_final' ORDER BY created LIMIT 1",(self.task_id,)).fetchone()
                binding=json.loads(marker[1])
                if not binding.get('candidate_id'):
                    previous=db.execute("SELECT detail FROM events WHERE task=? AND kind='terra_requests' AND created<=? ORDER BY created DESC LIMIT 1",(self.task_id,marker[0])).fetchone()
                    binding=json.loads(previous[0]) if previous else {}
                if binding.get('candidate_id')!=(detail or {}).get('candidate_id'):raise BudgetExceeded('Terminal retry must inspect the identical frozen candidate')
                if binding.get('delivery_hash') and binding['delivery_hash']!=(detail or {}).get('delivery_hash'):raise BudgetExceeded('Frozen delivery binding changed')
            event = uuid.uuid4().hex
            db.execute('INSERT INTO events(id,task,kind,amount,status,detail,created) VALUES (?,?,?,?,?,?,?)',
                       (event,self.task_id,'terra_requests',1,'reserved',json.dumps(detail or {},sort_keys=True),time.time()))
            if final and not used.get('terra_final',0):
                db.execute('INSERT INTO events(id,task,kind,amount,status,detail,created) VALUES (?,?,?,?,?,?,?)',
                           (uuid.uuid4().hex,self.task_id,'terra_final',1,'reserved',json.dumps(detail or {},sort_keys=True),time.time()))
        return event

    @contextlib.contextmanager
    def computation(self, detail=None, whole_task_slice=False):
        # Reserve a bounded active-wall lease before computation. On a crash the
        # outstanding lease stays charged. Completed leases record actual time.
        if getattr(self,'_time_depth',0):
            self._time_depth+=1
            try:yield
            finally:self._time_depth-=1
            return
        left = self.remaining('optimization_seconds')
        if left <= 0:
            raise BudgetExceeded('optimization_seconds')
        lease = left if whole_task_slice else min(120., left)
        event = self.reserve('optimization_seconds', lease, detail)
        start = time.monotonic()
        self._time_depth=1
        guarded=threading.current_thread() is threading.main_thread()
        old_handler=None
        if guarded:
            def deadline(signum,frame):raise BudgetExceeded('Optimization active-wall lease expired')
            old_handler=signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,lease)
        try:
            yield
        finally:
            self._time_depth=0
            if guarded:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
            elapsed = time.monotonic()-start
            with self.connect() as db:
                db.execute('UPDATE events SET amount=?,status=?,elapsed=? WHERE id=? AND task=?',
                           (max(elapsed, .000001), 'completed' if elapsed <= lease else 'overrun',elapsed,event,self.task_id))
            if elapsed > lease:
                raise BudgetExceeded('Single computation exceeded reserved wall lease')

    def record(self, kind, detail=None, elapsed=0.):
        with self.connect() as db:
            db.execute('INSERT INTO events(id,task,kind,amount,status,detail,created,elapsed) VALUES (?,?,?,?,?,?,?,?)',
                       (uuid.uuid4().hex,self.task_id,kind,0,'recorded',json.dumps(detail or {},sort_keys=True),time.time(),elapsed))

    def summary(self):
        with self.connect() as db:
            used = dict(db.execute('SELECT kind,SUM(amount) FROM events WHERE task=? GROUP BY kind',(self.task_id,)).fetchall())
            events = [dict(zip(['id','kind','amount','status','detail','created','elapsed'], row)) for row in db.execute('SELECT id,kind,amount,status,detail,created,elapsed FROM events WHERE task=? ORDER BY created',(self.task_id,))]
        for row in events:
            row['detail'] = json.loads(row['detail'])
        return dict(task_id=self.task_id,limits=self.limits,used=used,events=events)
