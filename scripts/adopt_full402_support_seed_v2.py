#!/usr/bin/env python3
"""Checked compatible continuation of the new batch; never edit old snapshots."""
from pathlib import Path
import sys
import shutil
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from movens_sim.util import ROOT,read,write,sha,digest,now,append
from movens_sim.full_batch import alive,connect,locked
from movens_sim.reconstruction.workflow import identity,verify_sources


def main():
    root=ROOT/'runs/full402_h0_v1';c=read(root/'resolved_config.json')
    with locked(root/'control/start.lock'):
        if alive(read(root/'job.json')['pid']):raise RuntimeError('Safe-stop this batch before adopting compatible code')
        with connect(root) as db:
            if db.execute("SELECT COUNT(*) FROM tasks WHERE status='running'").fetchone()[0]:raise RuntimeError('Workers have not checkpointed')
        import sqlite3
        ledger=ROOT/'runs/_task_budgets/full402_h0_v1.sqlite'
        def spent():
            with sqlite3.connect(ledger) as db:return db.execute('SELECT task,kind,SUM(amount) FROM events GROUP BY task,kind ORDER BY task,kind').fetchall()
        before=spent();version=root/'code_versions/support_seed_v2'
        if version.exists():raise RuntimeError('This code version already exists; never rewrite a published snapshot')
        for folder in ('src','prompts','configs'):shutil.copytree(ROOT/folder,version/folder,ignore=shutil.ignore_patterns('__pycache__'))
        (version/'scripts').mkdir();shutil.copy2(ROOT/'scripts/movens-sim',version/'scripts/movens-sim');shutil.copy2(ROOT/'pyproject.toml',version/'pyproject.toml')
        hashes={str(p.relative_to(version)):sha(p) for p in version.rglob('*') if p.is_file()};write(root/'code_versions/support_seed_v2_hashes.json',hashes)
        changes=[]
        for run in sorted(root.glob('episode_*')):
            if not (run/'run_identity.json').exists():continue
            if (run/'session.json').exists() and read(run/'session.json').get('complete'):continue
            verify_sources(run);old=read(run/'run_identity.json');new=identity(run,c)
            if any(old.get(k)!=new.get(k) for k in ('config_sha256','source_manifest_sha256','observation_sha256','mujoco')):raise ValueError('Cannot adopt changed input/config/observations')
            changed=[k for k in new['code'] if old['code'].get(k)!=new['code'][k]]
            allowed={'src/movens_sim/full_batch.py','src/movens_sim/reconstruction/backend.py','src/movens_sim/reconstruction/session.py','src/movens_sim/reconstruction/optimize.py','src/movens_sim/reconstruction/terra.py','src/movens_sim/reconstruction/workflow.py'}
            if set(changed)-allowed:raise ValueError('Unexpected source changes: '+str(set(changed)-allowed))
            # v1 swallowed a safe STOP inside MCP and prematurely exported.
            # No terminal model request was sent: preserve this interrupted
            # export as evidence, then resume the SAME pending fit reservation.
            interrupted=[]
            for p in run.glob('mcp_repair_*.json'):
                interrupted.extend(row for row in read(p)['calls'] if row['tool']=='fit_geometry' and row['result'].get('error',{}).get('code')=='BatchPaused')
            restored=False
            if interrupted:
                with sqlite3.connect(ledger) as db:
                    count=db.execute("SELECT COUNT(*) FROM events WHERE task=? AND kind='terra_final'",(f"full402_h0_v1/{run.name}",)).fetchone()[0]
                if count:raise ValueError('Cannot reopen a model-terminal candidate')
                state=read(run/'session.json');archive=run/'code_compatibility/interrupted_export_v1';archive.mkdir(parents=True)
                write(archive/'session.json',state)
                names=['delivery','physics','delivery_binding.json','reload_check.json','final_ego.png','final_masks.npz','overlay.png','metrics_after.json','physical_diagnostic.json','ego_static.mp4','ego_static_definition.json','settle_diagnostic.mp4','diagnostic_timeline.json','diagnostic_final_state.npz','parameters_final.json','parameter_registry.json','camera_actual.json','support_side.png','support_oblique.png']
                for name in names:
                    p=run/name
                    if p.exists():shutil.move(str(p),str(archive/name))
                state['stage']='geometry';state['complete']=False
                for key in ('geometry_stage_complete','appearance_status','stop_reason'):state.pop(key,None)
                write(run/'session.json',state);restored=True
            proof=dict(previous_identity_sha256=digest(old),next_identity_sha256=digest(new),changed_code=changed,reason='Feasible support search seed for an overhanging FK prior; safe-stop fit continuation; locked-camera tool contract',
                       config_inputs_observations_unchanged=True,old_snapshot_preserved=True,budgets_unchanged=True,interrupted_export_restored=restored,approved_utc=now())
            write(run/'code_compatibility/approved_continuation.json',proof);changes.append(dict(episode=run.name,**proof))
        after=spent()
        if before!=after:raise AssertionError('Code adoption must not refund or reserve budget')
        write(root/'code_versions/support_seed_v2_adoption.json',dict(changes=changes,budgets_before=before,budgets_after=after,old_code='frozen_code',new_code='code_versions/support_seed_v2',utc=now()))
        write(root/'active_code.json',dict(code_root='code_versions/support_seed_v2',hash_manifest='code_versions/support_seed_v2_hashes.json',reason='Checked compatible code continuation; old scenes and budgets retained'))
        print(json.dumps(dict(adopted=len(changes),budget_unchanged=True,old_snapshot_preserved=True)))


if __name__=='__main__':main()
